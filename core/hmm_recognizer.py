import numpy as np
from hmmlearn import hmm
import os
import pickle
import config
from core.audio_preprocessing import compute_mfcc, load_audio, VAD


class HMMCommandRecognizer:
    def __init__(self):
        self.models = {}
        self.command_names = []
        self.vad = VAD()
        self._load_existing_models()

    def _load_existing_models(self):
        if not os.path.exists(config.COMMANDS_DIR):
            return

        for name in os.listdir(config.COMMANDS_DIR):
            model_dir = os.path.join(config.COMMANDS_DIR, name)
            model_file = os.path.join(model_dir, "hmm_model.pkl")
            if os.path.isdir(model_dir) and os.path.exists(model_file):
                try:
                    with open(model_file, 'rb') as f:
                        model_data = pickle.load(f)
                        self.models[name] = model_data
                    self.command_names.append(name)
                except Exception as e:
                    print(f"Failed to load HMM model for {name}: {e}")

    def train_command(self, command_name, audio_files, num_states=5, num_mixtures=3):
        features_list = []
        for audio_file in audio_files:
            signal, sr = load_audio(audio_file)
            speech_signal, _ = self.vad.cut_speech(signal)
            use_signal = speech_signal if (speech_signal is not None and len(speech_signal) > 0) else signal
            mfcc = compute_mfcc(use_signal)
            features_list.append(mfcc)

        if len(features_list) < 2:
            return False, "训练样本不足，至少需要2个有效样本"

        all_features = np.vstack(features_list)
        lengths = [len(f) for f in features_list]

        model = hmm.GMMHMM(
            n_components=num_states,
            n_mix=num_mixtures,
            covariance_type="diag",
            n_iter=50,
            random_state=42
        )

        try:
            model.fit(all_features, lengths=lengths)

            model_data = {
                "model": model,
                "num_samples": len(features_list),
                "num_states": num_states,
                "num_mixtures": num_mixtures
            }

            self.models[command_name] = model_data
            if command_name not in self.command_names:
                self.command_names.append(command_name)

            self._save_model(command_name, model_data)

            return True, f"命令 '{command_name}' 训练完成，共 {len(features_list)} 个有效样本"
        except Exception as e:
            return False, f"训练失败: {str(e)}"

    def recognize(self, signal):
        speech_signal, _ = self.vad.cut_speech(signal)
        if speech_signal is None or len(speech_signal) < 1000:
            if len(signal) >= 1000:
                speech_signal = signal
            else:
                return {
                    "command": "unknown_command",
                    "confidence": 0.0,
                    "all_scores": {},
                    "normalized_scores": {},
                    "top_candidates": []
                }

        mfcc_features = compute_mfcc(speech_signal)

        scores = {}
        for name, model_data in self.models.items():
            try:
                model = model_data["model"]
                score = model.score(mfcc_features)
                scores[name] = float(score)
            except Exception as e:
                scores[name] = float('-inf')

        if not scores:
            return {
                "command": "unknown_command",
                "confidence": 0.0,
                "all_scores": scores,
                "normalized_scores": {},
                "top_candidates": []
            }

        max_score = max(scores.values())
        min_score = min(v for v in scores.values() if v != float('-inf')) if any(v != float('-inf') for v in scores.values()) else 0

        if max_score == float('-inf'):
            return {
                "command": "unknown_command",
                "confidence": 0.0,
                "all_scores": scores,
                "normalized_scores": {},
                "top_candidates": []
            }

        if max_score == min_score:
            normalized_scores = {k: 1.0 for k, v in scores.items() if v != float('-inf')}
        else:
            normalized_scores = {k: (v - min_score) / (max_score - min_score + 1e-10)
                                 for k, v in scores.items() if v != float('-inf')}

        sorted_candidates = sorted(normalized_scores.items(), key=lambda x: x[1], reverse=True)
        top_k = sorted_candidates[:config.TOP_K_CANDIDATES]
        top_candidates = [{"command": cmd, "confidence": float(conf)} for cmd, conf in top_k]

        best_command = sorted_candidates[0][0] if sorted_candidates else "unknown_command"
        best_confidence = normalized_scores.get(best_command, 0.0)

        should_reject = False
        rejection_reason = None

        if best_confidence < config.COMMAND_MIN_CONFIDENCE:
            should_reject = True
            rejection_reason = "low_confidence"

        if len(sorted_candidates) >= 2:
            top1_conf = sorted_candidates[0][1]
            top2_conf = sorted_candidates[1][1]
            if (top1_conf - top2_conf) < config.COMMAND_REJECT_MARGIN:
                should_reject = True
                rejection_reason = "low_margin"

        if should_reject:
            final_command = "unclear_command"
        elif best_confidence < config.COMMAND_CONFIDENCE_THRESHOLD:
            final_command = "unknown_command"
        else:
            final_command = best_command

        result = {
            "command": final_command,
            "confidence": float(best_confidence),
            "all_scores": scores,
            "normalized_scores": normalized_scores,
            "top_candidates": top_candidates
        }

        if rejection_reason:
            result["rejection_reason"] = rejection_reason

        return result

    def recognize_file(self, file_path):
        signal, sr = load_audio(file_path)
        return self.recognize(signal)

    def _save_model(self, command_name, model_data):
        command_dir = os.path.join(config.COMMANDS_DIR, command_name)
        os.makedirs(command_dir, exist_ok=True)

        model_file = os.path.join(command_dir, "hmm_model.pkl")
        with open(model_file, 'wb') as f:
            pickle.dump(model_data, f)

    def get_command_list(self):
        result = []
        for name in self.command_names:
            model_data = self.models.get(name, {})
            result.append({
                "name": name,
                "samples": model_data.get("num_samples", 0),
                "states": model_data.get("num_states", 0),
                "mixtures": model_data.get("num_mixtures", 0)
            })
        return result

    def delete_command(self, command_name):
        if command_name not in self.models:
            return False, f"命令 '{command_name}' 不存在"

        del self.models[command_name]
        if command_name in self.command_names:
            self.command_names.remove(command_name)

        command_dir = os.path.join(config.COMMANDS_DIR, command_name)
        if os.path.exists(command_dir):
            import shutil
            shutil.rmtree(command_dir)

        return True, f"命令 '{command_name}' 已删除"

    def reload_models(self):
        self.models.clear()
        self.command_names.clear()
        self._load_existing_models()


class DigitRecognizer:
    def __init__(self):
        self.digits = {}
        self.vad = VAD()
        self._load_existing_models()

    def _load_existing_models(self):
        if not os.path.exists(config.DIGITS_DIR):
            return

        for digit_name in os.listdir(config.DIGITS_DIR):
            digit_dir = os.path.join(config.DIGITS_DIR, digit_name)
            model_file = os.path.join(digit_dir, "hmm_model.pkl")
            if os.path.isdir(digit_dir) and os.path.exists(model_file):
                try:
                    with open(model_file, 'rb') as f:
                        model_data = pickle.load(f)
                        self.digits[digit_name] = model_data
                except Exception as e:
                    print(f"Failed to load digit model for {digit_name}: {e}")

    def train_digit(self, digit, audio_files, num_states=3, num_mixtures=2):
        digit_str = str(digit)
        features_list = []

        for audio_file in audio_files:
            signal, sr = load_audio(audio_file)
            speech_signal, _ = self.vad.cut_speech(signal)
            use_signal = speech_signal if (speech_signal is not None and len(speech_signal) > 0) else signal
            mfcc = compute_mfcc(use_signal)
            features_list.append(mfcc)

        if len(features_list) < 2:
            return False, "训练样本不足，至少需要2个有效样本"

        all_features = np.vstack(features_list)
        lengths = [len(f) for f in features_list]

        model = hmm.GMMHMM(
            n_components=num_states,
            n_mix=num_mixtures,
            covariance_type="diag",
            n_iter=50,
            random_state=42
        )

        try:
            model.fit(all_features, lengths=lengths)

            model_data = {
                "model": model,
                "num_samples": len(features_list),
                "num_states": num_states,
                "num_mixtures": num_mixtures
            }

            self.digits[digit_str] = model_data
            self._save_model(digit_str, model_data)

            return True, f"数字 '{digit_str}' 训练完成，共 {len(features_list)} 个有效样本"
        except Exception as e:
            return False, f"训练失败: {str(e)}"

    def recognize_digit(self, signal):
        speech_signal, _ = self.vad.cut_speech(signal)
        if speech_signal is None or len(speech_signal) < 500:
            if len(signal) >= 500:
                speech_signal = signal
            else:
                return {"digit": None, "confidence": 0.0, "all_scores": {}}

        mfcc_features = compute_mfcc(speech_signal)

        scores = {}
        for digit_str, model_data in self.digits.items():
            try:
                model = model_data["model"]
                score = model.score(mfcc_features)
                scores[digit_str] = float(score)
            except Exception:
                scores[digit_str] = float('-inf')

        if not scores:
            return {"digit": None, "confidence": 0.0, "all_scores": {}}

        max_score = max(scores.values())
        if max_score == float('-inf'):
            return {"digit": None, "confidence": 0.0, "all_scores": scores}

        best_digit = max(scores, key=scores.get)
        min_score = min(v for v in scores.values() if v != float('-inf'))

        if max_score == min_score:
            confidence = 1.0
        else:
            confidence = (max_score - min_score) / (max_score - min_score + 1e-10)

        return {
            "digit": best_digit,
            "confidence": float(confidence),
            "all_scores": scores
        }

    def recognize_digits_sequence(self, signal, max_digits=3):
        speech_signal, vad_result = self.vad.cut_speech(signal)
        if speech_signal is None:
            if len(signal) >= 500:
                speech_signal = signal
            else:
                return {"digits": [], "confidence": 0.0}

        mfcc_features = compute_mfcc(speech_signal)
        total_frames = len(mfcc_features)

        best_sequence = []
        best_score = float('-inf')

        for num_digits in range(1, min(max_digits, total_frames // 10) + 1):
            segment_length = total_frames // num_digits

            digits_sequence = []
            total_confidence = 0.0

            for i in range(num_digits):
                start = i * segment_length
                end = start + segment_length if i < num_digits - 1 else total_frames
                segment = mfcc_features[start:end]

                if len(segment) < 5:
                    continue

                result = self._recognize_from_features(segment)
                if result["digit"] is not None:
                    digits_sequence.append(result["digit"])
                    total_confidence += result["confidence"]

            if len(digits_sequence) == num_digits:
                avg_confidence = total_confidence / num_digits
                if avg_confidence > best_score:
                    best_score = avg_confidence
                    best_sequence = digits_sequence

        if best_sequence:
            return {
                "digits": best_sequence,
                "number": int(''.join(best_sequence)),
                "confidence": float(best_score)
            }
        else:
            return {"digits": [], "number": None, "confidence": 0.0}

    def _recognize_from_features(self, mfcc_features):
        scores = {}
        for digit_str, model_data in self.digits.items():
            try:
                model = model_data["model"]
                score = model.score(mfcc_features)
                scores[digit_str] = float(score)
            except Exception:
                scores[digit_str] = float('-inf')

        if not scores:
            return {"digit": None, "confidence": 0.0}

        max_score = max(scores.values())
        if max_score == float('-inf'):
            return {"digit": None, "confidence": 0.0}

        best_digit = max(scores, key=scores.get)
        min_score = min(v for v in scores.values() if v != float('-inf'))

        if max_score == min_score:
            confidence = 1.0
        else:
            confidence = (max_score - min_score) / (max_score - min_score + 1e-10)

        return {"digit": best_digit, "confidence": float(confidence)}

    def _save_model(self, digit_str, model_data):
        digit_dir = os.path.join(config.DIGITS_DIR, digit_str)
        os.makedirs(digit_dir, exist_ok=True)

        model_file = os.path.join(digit_dir, "hmm_model.pkl")
        with open(model_file, 'wb') as f:
            pickle.dump(model_data, f)

    def get_digit_list(self):
        result = []
        for digit_str, model_data in self.digits.items():
            result.append({
                "digit": digit_str,
                "samples": model_data.get("num_samples", 0),
                "states": model_data.get("num_states", 0),
                "mixtures": model_data.get("num_mixtures", 0)
            })
        return result

    def reload_models(self):
        self.digits.clear()
        self._load_existing_models()
