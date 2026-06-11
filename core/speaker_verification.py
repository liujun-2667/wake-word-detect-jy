import numpy as np
import os
import json
import time
import uuid

import config
from core.audio_preprocessing import compute_mfcc, load_audio, VAD


def cosine_similarity(v1, v2):
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot / (norm1 * norm2))


def extract_speaker_feature(signal, sample_rate=None):
    if sample_rate is None:
        sample_rate = config.SAMPLE_RATE

    vad = VAD()
    speech_signal, _ = vad.cut_speech(signal)
    if speech_signal is None or len(speech_signal) < int(sample_rate * 0.1):
        speech_signal = signal

    mfcc = compute_mfcc(speech_signal, sample_rate=sample_rate,
                        num_mfcc=config.SPEAKER_MFCC_NUM,
                        use_delta=True, use_delta_delta=False)

    num_mfcc = config.SPEAKER_MFCC_NUM
    mfcc_static = mfcc[:, :num_mfcc]
    mfcc_delta = mfcc[:, num_mfcc:2 * num_mfcc]

    mfcc_mean = np.mean(mfcc_static, axis=0)
    delta_mean = np.mean(mfcc_delta, axis=0)
    delta_std = np.std(mfcc_delta, axis=0)

    feature = np.concatenate([mfcc_mean, delta_mean, delta_std])
    return feature.astype(np.float64)


def extract_speaker_feature_from_file(file_path):
    signal, sr = load_audio(file_path)
    return extract_speaker_feature(signal, sample_rate=sr)


class VerificationStats:
    def __init__(self, stats_file=None):
        if stats_file is None:
            stats_file = config.SPEAKER_STATS_FILE
        self.stats_file = stats_file
        self.data = self._load()

    def _default_data(self):
        return {
            "total_verifications": 0,
            "high_count": 0,
            "medium_count": 0,
            "rejected_count": 0,
            "blacklist_blocks": 0,
            "speaker_stats": {}
        }

    def _load(self):
        if os.path.exists(self.stats_file):
            try:
                with open(self.stats_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for key in self._default_data():
                    if key not in data:
                        data[key] = self._default_data()[key]
                return data
            except Exception:
                pass
        return self._default_data()

    def _save(self):
        os.makedirs(os.path.dirname(self.stats_file), exist_ok=True)
        with open(self.stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def record(self, confidence_level, speaker_id=None, speaker_name=None, blocked_by=None):
        self.data["total_verifications"] += 1
        if blocked_by:
            self.data["blacklist_blocks"] += 1
            self.data["rejected_count"] += 1
        elif confidence_level == "high":
            self.data["high_count"] += 1
        elif confidence_level == "medium":
            self.data["medium_count"] += 1
        else:
            self.data["rejected_count"] += 1

        if speaker_id and not blocked_by:
            if speaker_id not in self.data["speaker_stats"]:
                self.data["speaker_stats"][speaker_id] = {
                    "name": speaker_name or "",
                    "verify_count": 0,
                    "last_verify_time": None
                }
            self.data["speaker_stats"][speaker_id]["verify_count"] += 1
            self.data["speaker_stats"][speaker_id]["last_verify_time"] = time.strftime("%Y-%m-%d %H:%M:%S")

        self._save()

    def get_stats(self):
        total = self.data["total_verifications"]
        high = self.data["high_count"]
        medium = self.data["medium_count"]
        rejected = self.data["rejected_count"]
        pass_rate = (high + medium) / total if total > 0 else 0.0
        return {
            "total_verifications": total,
            "pass_rate": float(pass_rate),
            "high_count": high,
            "medium_count": medium,
            "rejected_count": rejected,
            "blacklist_blocks": self.data["blacklist_blocks"],
            "high_ratio": float(high / total) if total > 0 else 0.0,
            "medium_ratio": float(medium / total) if total > 0 else 0.0,
            "rejected_ratio": float(rejected / total) if total > 0 else 0.0,
            "speaker_stats": dict(self.data["speaker_stats"])
        }

    def reset(self):
        self.data = self._default_data()
        self._save()


class SpeakerManager:
    def __init__(self):
        self.speakers_dir = config.SPEAKERS_DIR
        os.makedirs(self.speakers_dir, exist_ok=True)
        self.speakers = {}
        self.stats = VerificationStats()
        self._load_all_speakers()

    def _load_all_speakers(self):
        self.speakers = {}
        if not os.path.exists(self.speakers_dir):
            return
        for filename in os.listdir(self.speakers_dir):
            if filename.endswith('.json'):
                try:
                    filepath = os.path.join(self.speakers_dir, filename)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    speaker_id = data.get('id')
                    if speaker_id:
                        self.speakers[speaker_id] = data
                except Exception:
                    pass

    def _save_speaker(self, speaker_data):
        filepath = os.path.join(self.speakers_dir, f"{speaker_data['id']}.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(speaker_data, f, ensure_ascii=False, indent=2)

    def register_speaker(self, name, audio_files, speaker_type="whitelist"):
        if len(audio_files) < config.SPEAKER_MIN_SAMPLES:
            return False, f"至少需要 {config.SPEAKER_MIN_SAMPLES} 段音频样本"

        if speaker_type not in ("whitelist", "blacklist"):
            return False, "speaker_type 必须为 whitelist 或 blacklist"

        features = []
        for audio_file in audio_files:
            try:
                feat = extract_speaker_feature_from_file(audio_file)
                features.append(feat)
            except Exception as e:
                return False, f"音频处理失败: {str(e)}"

        template = np.mean(features, axis=0)

        speaker_id = str(uuid.uuid4())
        speaker_data = {
            'id': speaker_id,
            'name': name,
            'template': template.tolist(),
            'num_samples': len(audio_files),
            'created_at': time.strftime("%Y-%m-%d %H:%M:%S"),
            'created_at_unix': time.time(),
            'speaker_type': speaker_type,
            'verify_count': 0,
            'last_verify_time': None
        }

        self._save_speaker(speaker_data)
        self.speakers[speaker_id] = speaker_data

        type_label = "黑名单" if speaker_type == "blacklist" else "白名单"
        return True, f"说话人 '{name}' ({type_label}) 注册成功 (ID: {speaker_id})"

    def delete_speaker(self, speaker_id):
        if speaker_id not in self.speakers:
            return False, f"说话人 {speaker_id} 不存在"

        filepath = os.path.join(self.speakers_dir, f"{speaker_id}.json")
        if os.path.exists(filepath):
            os.remove(filepath)

        del self.speakers[speaker_id]
        return True, f"说话人已删除"

    def get_speaker_list(self):
        result = []
        for spk_id, spk_data in self.speakers.items():
            result.append({
                'id': spk_data['id'],
                'name': spk_data['name'],
                'num_samples': spk_data.get('num_samples', 0),
                'created_at': spk_data.get('created_at', ''),
                'speaker_type': spk_data.get('speaker_type', 'whitelist'),
                'verify_count': spk_data.get('verify_count', 0),
                'last_verify_time': spk_data.get('last_verify_time')
            })
        result.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        return result

    def get_speaker_count(self):
        return len(self.speakers)

    def _update_voiceprint(self, speaker_id, new_feature, decay=None):
        if decay is None:
            decay = config.SPEAKER_EMA_DECAY

        spk_data = self.speakers.get(speaker_id)
        if spk_data is None:
            return

        old_template = np.array(spk_data['template'])
        updated_template = (1.0 - decay) * old_template + decay * new_feature
        spk_data['template'] = updated_template.tolist()
        spk_data['num_samples'] = spk_data.get('num_samples', 0) + 1
        spk_data['verify_count'] = spk_data.get('verify_count', 0) + 1
        spk_data['last_verify_time'] = time.strftime("%Y-%m-%d %H:%M:%S")

        self._save_speaker(spk_data)

    def verify_speaker(self, signal, high_threshold=None, medium_threshold=None,
                       blacklist_threshold=None, ema_decay=None):
        if high_threshold is None:
            high_threshold = config.SPEAKER_HIGH_CONFIDENCE_THRESHOLD
        if medium_threshold is None:
            medium_threshold = config.SPEAKER_MEDIUM_CONFIDENCE_THRESHOLD
        if blacklist_threshold is None:
            blacklist_threshold = config.SPEAKER_BLACKLIST_THRESHOLD
        if ema_decay is None:
            ema_decay = config.SPEAKER_EMA_DECAY

        if not config.SPEAKER_VERIFICATION_ENABLED:
            return {
                'verified': True,
                'speaker_id': None,
                'speaker_name': None,
                'confidence': 0.0,
                'confidence_level': 'high',
                'skipped': True,
                'reason': 'verification_disabled',
                'blocked_by': None
            }

        whitelist_speakers = {k: v for k, v in self.speakers.items()
                              if v.get('speaker_type', 'whitelist') == 'whitelist'}
        blacklist_speakers = {k: v for k, v in self.speakers.items()
                              if v.get('speaker_type') == 'blacklist'}

        if len(whitelist_speakers) == 0:
            return {
                'verified': True,
                'speaker_id': None,
                'speaker_name': None,
                'confidence': 0.0,
                'confidence_level': 'high',
                'skipped': True,
                'reason': 'no_whitelist_speakers_registered',
                'blocked_by': None
            }

        try:
            test_feature = extract_speaker_feature(signal)
        except Exception:
            result = {
                'verified': False,
                'speaker_id': None,
                'speaker_name': None,
                'confidence': 0.0,
                'confidence_level': 'rejected',
                'skipped': False,
                'reason': 'feature_extraction_failed',
                'blocked_by': None
            }
            self.stats.record('rejected')
            return result

        blocked_by = None
        if blacklist_speakers:
            for spk_id, spk_data in blacklist_speakers.items():
                template = np.array(spk_data['template'])
                score = cosine_similarity(test_feature, template)
                if score >= blacklist_threshold:
                    blocked_by = spk_id
                    result = {
                        'verified': False,
                        'speaker_id': None,
                        'speaker_name': None,
                        'confidence': float(score),
                        'confidence_level': 'rejected',
                        'skipped': False,
                        'reason': 'blacklisted',
                        'blocked_by': blocked_by,
                        'blocked_by_name': spk_data.get('name', ''),
                        'high_threshold': high_threshold,
                        'medium_threshold': medium_threshold,
                        'blacklist_threshold': blacklist_threshold
                    }
                    self.stats.record('rejected', blocked_by=blocked_by)
                    return result

        best_speaker_id = None
        best_speaker_name = None
        best_score = -1.0

        for spk_id, spk_data in whitelist_speakers.items():
            template = np.array(spk_data['template'])
            score = cosine_similarity(test_feature, template)
            if score > best_score:
                best_score = score
                best_speaker_id = spk_id
                best_speaker_name = spk_data['name']

        if best_score >= high_threshold:
            confidence_level = 'high'
            verified = True
            self._update_voiceprint(best_speaker_id, test_feature, decay=ema_decay)
            self.stats.record('high', speaker_id=best_speaker_id, speaker_name=best_speaker_name)
        elif best_score >= medium_threshold:
            confidence_level = 'medium'
            verified = True
            spk_data = self.speakers.get(best_speaker_id, {})
            spk_data['verify_count'] = spk_data.get('verify_count', 0) + 1
            spk_data['last_verify_time'] = time.strftime("%Y-%m-%d %H:%M:%S")
            self._save_speaker(spk_data)
            self.stats.record('medium', speaker_id=best_speaker_id, speaker_name=best_speaker_name)
        else:
            confidence_level = 'rejected'
            verified = False
            self.stats.record('rejected')

        return {
            'verified': verified,
            'speaker_id': best_speaker_id if verified else None,
            'speaker_name': best_speaker_name if verified else None,
            'confidence': float(best_score),
            'confidence_level': confidence_level,
            'skipped': False,
            'blocked_by': None,
            'high_threshold': high_threshold,
            'medium_threshold': medium_threshold,
            'blacklist_threshold': blacklist_threshold
        }

    def verify_speaker_file(self, file_path, **kwargs):
        signal, sr = load_audio(file_path)
        return self.verify_speaker(signal, **kwargs)

    def get_verification_stats(self):
        return self.stats.get_stats()

    def reset_verification_stats(self):
        self.stats.reset()

    def reload_speakers(self):
        self._load_all_speakers()
