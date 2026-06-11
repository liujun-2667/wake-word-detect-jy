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


class SpeakerManager:
    def __init__(self):
        self.speakers_dir = config.SPEAKERS_DIR
        os.makedirs(self.speakers_dir, exist_ok=True)
        self.speakers = {}
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

    def register_speaker(self, name, audio_files):
        if len(audio_files) < config.SPEAKER_MIN_SAMPLES:
            return False, f"至少需要 {config.SPEAKER_MIN_SAMPLES} 段音频样本"

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
            'created_at_unix': time.time()
        }

        self._save_speaker(speaker_data)
        self.speakers[speaker_id] = speaker_data

        return True, f"说话人 '{name}' 注册成功 (ID: {speaker_id})"

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
                'created_at': spk_data.get('created_at', '')
            })
        result.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        return result

    def get_speaker_count(self):
        return len(self.speakers)

    def verify_speaker(self, signal, threshold=None):
        if threshold is None:
            threshold = config.SPEAKER_VERIFICATION_THRESHOLD

        if not config.SPEAKER_VERIFICATION_ENABLED:
            return {
                'verified': True,
                'speaker_id': None,
                'speaker_name': None,
                'confidence': 0.0,
                'skipped': True,
                'reason': 'verification_disabled'
            }

        if len(self.speakers) == 0:
            return {
                'verified': True,
                'speaker_id': None,
                'speaker_name': None,
                'confidence': 0.0,
                'skipped': True,
                'reason': 'no_speakers_registered'
            }

        try:
            test_feature = extract_speaker_feature(signal)
        except Exception:
            return {
                'verified': False,
                'speaker_id': None,
                'speaker_name': None,
                'confidence': 0.0,
                'skipped': False,
                'reason': 'feature_extraction_failed'
            }

        best_speaker_id = None
        best_speaker_name = None
        best_score = -1.0

        for spk_id, spk_data in self.speakers.items():
            template = np.array(spk_data['template'])
            score = cosine_similarity(test_feature, template)
            if score > best_score:
                best_score = score
                best_speaker_id = spk_id
                best_speaker_name = spk_data['name']

        verified = best_score >= threshold

        return {
            'verified': verified,
            'speaker_id': best_speaker_id if verified else None,
            'speaker_name': best_speaker_name if verified else None,
            'confidence': float(best_score),
            'skipped': False,
            'threshold': threshold,
            'num_speakers': len(self.speakers)
        }

    def verify_speaker_file(self, file_path, threshold=None):
        signal, sr = load_audio(file_path)
        return self.verify_speaker(signal, threshold=threshold)

    def reload_speakers(self):
        self._load_all_speakers()
