import numpy as np
import os
import json
import pickle
from collections import defaultdict
import config
from core.audio_preprocessing import compute_mfcc, VAD, load_audio, compute_rms, classify_noise_level, compute_adaptive_threshold
from core.dtw import dtw_similarity, average_template, multi_template_similarity
from core.wake_word_cnn import MultiClassWakeWordCNN, TORCH_AVAILABLE


class WakeWordDetector:
    def __init__(self, threshold=None, dtw_weight=None, cnn_weight=None):
        self.threshold = threshold if threshold else config.WAKE_WORD_THRESHOLD
        self.dtw_weight = dtw_weight if dtw_weight else config.DTW_WEIGHT
        self.cnn_weight = cnn_weight if cnn_weight else config.CNN_WEIGHT

        self.wake_words = {}
        self.dtw_templates = defaultdict(list)
        self.avg_templates = {}
        self.cnn_model = None
        self.vad = VAD()

        self.detection_stats = defaultdict(lambda: {"total": 0, "correct": 0})

        self.smoothed_rms = 0.0
        self.current_noise_level = "moderate"
        self.current_adaptive_threshold = self.threshold
        self.noise_alpha = config.NOISE_SMOOTHING_ALPHA

        self._load_existing_models()

    def _load_existing_models(self):
        if not os.path.exists(config.WAKE_WORDS_DIR):
            return

        wake_word_list = []
        for name in os.listdir(config.WAKE_WORDS_DIR):
            wake_word_dir = os.path.join(config.WAKE_WORDS_DIR, name)
            if os.path.isdir(wake_word_dir):
                self.wake_words[name] = {"samples": 0, "accuracy": 0.0}

                templates_file = os.path.join(wake_word_dir, "templates.pkl")
                if os.path.exists(templates_file):
                    with open(templates_file, 'rb') as f:
                        templates_data = pickle.load(f)
                        self.dtw_templates[name] = templates_data.get('templates', [])
                        self.avg_templates[name] = templates_data.get('avg_template', None)
                        self.wake_words[name]["samples"] = len(self.dtw_templates[name])

                stats_file = os.path.join(wake_word_dir, "stats.json")
                if os.path.exists(stats_file):
                    with open(stats_file, 'r') as f:
                        stats = json.load(f)
                        self.detection_stats[name] = stats
                        self.wake_words[name]["accuracy"] = stats.get("accuracy", 0.0)

                wake_word_list.append(name)

        if TORCH_AVAILABLE:
            cnn_model_path = os.path.join(config.WAKE_WORDS_DIR, "cnn_model.pth")
            if os.path.exists(cnn_model_path) and wake_word_list:
                try:
                    self.cnn_model = MultiClassWakeWordCNN(num_wake_words=len(wake_word_list))
                    self.cnn_model.load(cnn_model_path)
                except Exception as e:
                    print(f"Failed to load CNN model: {e}")
                    self.cnn_model = None

    def register_wake_word(self, name, audio_files):
        if len(self.wake_words) >= config.MAX_WAKE_WORDS and name not in self.wake_words:
            return False, f"已达到最大唤醒词数量 ({config.MAX_WAKE_WORDS})"

        if len(audio_files) < 3:
            return False, "至少需要3段音频样本"

        templates = []
        for audio_file in audio_files:
            signal, sr = load_audio(audio_file)
            speech_signal, _ = self.vad.cut_speech(signal)

            if speech_signal is not None and len(speech_signal) > 0:
                mfcc_features = compute_mfcc(speech_signal)
                templates.append(mfcc_features)
            else:
                mfcc_features = compute_mfcc(signal)
                templates.append(mfcc_features)

        if len(templates) < 3:
            return False, "有效的语音样本不足3个"

        self.dtw_templates[name] = templates
        self.avg_templates[name] = average_template(templates)
        self.wake_words[name] = {
            "samples": len(templates),
            "accuracy": 0.0
        }

        self._save_wake_word_templates(name)
        self._retrain_cnn()

        return True, f"唤醒词 '{name}' 注册成功，共 {len(templates)} 个有效样本"

    def add_sample(self, name, audio_file):
        if name not in self.wake_words:
            return False, f"唤醒词 '{name}' 不存在"

        signal, sr = load_audio(audio_file)
        speech_signal, _ = self.vad.cut_speech(signal)

        use_signal = speech_signal if (speech_signal is not None and len(speech_signal) > 0) else signal
        mfcc_features = compute_mfcc(use_signal)
        self.dtw_templates[name].append(mfcc_features)
        self.avg_templates[name] = average_template(self.dtw_templates[name])
        self.wake_words[name]["samples"] = len(self.dtw_templates[name])

        self._save_wake_word_templates(name)
        self._retrain_cnn()

        return True, f"已添加样本，当前共 {len(self.dtw_templates[name])} 个样本"

    def detect(self, signal, return_details=False):
        current_rms = compute_rms(signal)
        if self.smoothed_rms == 0.0:
            self.smoothed_rms = current_rms
        else:
            self.smoothed_rms = self.noise_alpha * self.smoothed_rms + (1 - self.noise_alpha) * current_rms

        self.current_noise_level = classify_noise_level(self.smoothed_rms)
        self.current_adaptive_threshold = compute_adaptive_threshold(self.current_noise_level, self.threshold)

        effective_threshold = self.current_adaptive_threshold

        speech_signal, vad_result = self.vad.cut_speech(signal)

        if speech_signal is None or len(speech_signal) < 1000:
            if len(signal) >= 1000:
                speech_signal = signal
            else:
                result = {
                    "detected": False,
                    "wake_word": None,
                    "confidence": 0.0,
                    "dtw_confidence": {},
                    "cnn_confidence": {},
                    "combined_confidence": {},
                    "vad_result": vad_result if return_details else None,
                    "noise_level": self.current_noise_level,
                    "noise_rms": self.smoothed_rms,
                    "adaptive_threshold": effective_threshold
                }
                return result

        mfcc_features = compute_mfcc(speech_signal)

        dtw_scores = {}
        for name, templates in self.dtw_templates.items():
            if templates:
                score = multi_template_similarity(mfcc_features, templates)
                dtw_scores[name] = float(score)

        cnn_scores = {}
        if self.cnn_model and self.cnn_model.is_trained:
            try:
                _, _, cnn_results = self.cnn_model.predict(mfcc_features)
                cnn_scores = {k: v for k, v in cnn_results.items() if k != 'background'}
            except Exception:
                cnn_scores = {}

        combined_scores = {}
        all_wake_words = set(list(dtw_scores.keys()) + list(cnn_scores.keys()))

        for name in all_wake_words:
            dtw_score = dtw_scores.get(name, 0.0)
            cnn_score = cnn_scores.get(name, 0.0)

            if cnn_score > 0:
                combined = self.dtw_weight * dtw_score + self.cnn_weight * cnn_score
            else:
                combined = dtw_score

            combined_scores[name] = float(combined)

        best_wake_word = None
        best_confidence = 0.0
        if combined_scores:
            best_wake_word = max(combined_scores, key=combined_scores.get)
            best_confidence = combined_scores[best_wake_word]

        detected = best_confidence >= effective_threshold

        result = {
            "detected": detected,
            "wake_word": best_wake_word if detected else None,
            "confidence": best_confidence,
            "dtw_confidence": dtw_scores,
            "cnn_confidence": cnn_scores,
            "combined_confidence": combined_scores,
            "vad_result": vad_result if return_details else None,
            "noise_level": self.current_noise_level,
            "noise_rms": self.smoothed_rms,
            "adaptive_threshold": effective_threshold
        }

        return result

    def detect_file(self, file_path, return_details=False):
        signal, sr = load_audio(file_path)
        return self.detect(signal, return_details=return_details)

    def sliding_window_detect(self, signal, window_duration=1.0, step_duration=0.5):
        sample_rate = config.SAMPLE_RATE
        window_samples = int(window_duration * sample_rate)
        step_samples = int(step_duration * sample_rate)

        results = []
        start = 0

        while start + window_samples <= len(signal):
            window_signal = signal[start:start + window_samples]
            result = self.detect(window_signal)
            result["start_time"] = start / sample_rate
            result["end_time"] = (start + window_samples) / sample_rate
            results.append(result)
            start += step_samples

        return results

    def _retrain_cnn(self):
        if not TORCH_AVAILABLE:
            self.cnn_model = None
            return

        if len(self.wake_words) == 0:
            self.cnn_model = None
            return

        wake_word_data = {}
        all_features = []

        for name, templates in self.dtw_templates.items():
            if templates:
                wake_word_data[name] = templates
                all_features.extend(templates)

        if not wake_word_data:
            self.cnn_model = None
            return

        negative_features = self._generate_negative_features(all_features)

        num_wake_words = len(wake_word_data)
        try:
            self.cnn_model = MultiClassWakeWordCNN(num_wake_words=num_wake_words)
            self.cnn_model.train(wake_word_data, negative_features, epochs=30, batch_size=8)
            cnn_model_path = os.path.join(config.WAKE_WORDS_DIR, "cnn_model.pth")
            self.cnn_model.save(cnn_model_path)
        except Exception as e:
            print(f"CNN training failed: {e}")
            self.cnn_model = None

    def _generate_negative_features(self, positive_features, num_negatives=20):
        negatives = []

        if not positive_features:
            for _ in range(num_negatives):
                length = np.random.randint(30, 80)
                feat = np.random.randn(length, 39) * 0.1
                negatives.append(feat)
            return negatives

        for _ in range(num_negatives):
            template = positive_features[np.random.randint(len(positive_features))]
            length = template.shape[0]
            noise = np.random.randn(length, 39) * 0.5
            distorted = template * 0.3 + noise
            negatives.append(distorted)

            reversed_feat = np.flip(template, axis=0)
            negatives.append(reversed_feat)

        return negatives

    def _save_wake_word_templates(self, name):
        wake_word_dir = os.path.join(config.WAKE_WORDS_DIR, name)
        os.makedirs(wake_word_dir, exist_ok=True)

        templates_file = os.path.join(wake_word_dir, "templates.pkl")
        with open(templates_file, 'wb') as f:
            pickle.dump({
                'templates': self.dtw_templates[name],
                'avg_template': self.avg_templates.get(name)
            }, f)

    def update_stats(self, wake_word_name, is_correct):
        self.detection_stats[wake_word_name]["total"] += 1
        if is_correct:
            self.detection_stats[wake_word_name]["correct"] += 1

        total = self.detection_stats[wake_word_name]["total"]
        correct = self.detection_stats[wake_word_name]["correct"]
        accuracy = correct / total if total > 0 else 0.0
        self.wake_words[wake_word_name]["accuracy"] = accuracy

        wake_word_dir = os.path.join(config.WAKE_WORDS_DIR, wake_word_name)
        if os.path.exists(wake_word_dir):
            stats_file = os.path.join(wake_word_dir, "stats.json")
            with open(stats_file, 'w') as f:
                json.dump(self.detection_stats[wake_word_name], f, indent=2)

    def get_wake_word_list(self):
        result = []
        for name, info in self.wake_words.items():
            result.append({
                "name": name,
                "samples": info["samples"],
                "accuracy": info["accuracy"]
            })
        return result

    def get_environment_status(self):
        return {
            "noise_level": self.current_noise_level,
            "noise_rms": float(self.smoothed_rms),
            "adaptive_threshold": float(self.current_adaptive_threshold),
            "base_threshold": float(self.threshold)
        }

    def delete_wake_word(self, name):
        if name not in self.wake_words:
            return False, f"唤醒词 '{name}' 不存在"

        del self.wake_words[name]
        if name in self.dtw_templates:
            del self.dtw_templates[name]
        if name in self.avg_templates:
            del self.avg_templates[name]
        if name in self.detection_stats:
            del self.detection_stats[name]

        wake_word_dir = os.path.join(config.WAKE_WORDS_DIR, name)
        if os.path.exists(wake_word_dir):
            import shutil
            shutil.rmtree(wake_word_dir)

        self._retrain_cnn()

        return True, f"唤醒词 '{name}' 已删除"

    def reload_models(self):
        self.wake_words.clear()
        self.dtw_templates.clear()
        self.avg_templates.clear()
        self.cnn_model = None
        self.detection_stats.clear()
        self._load_existing_models()
