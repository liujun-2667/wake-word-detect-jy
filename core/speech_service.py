import numpy as np
import os
import json
import time
from collections import deque
import config
from core.audio_preprocessing import compute_mfcc, VAD, load_audio, NoiseEstimator, spectral_subtraction
from core.wake_word_detector import WakeWordDetector
from core.hmm_recognizer import HMMCommandRecognizer, DigitRecognizer
from core.speaker_verification import SpeakerManager


class SpeechRecognitionService:
    def __init__(self):
        self.wake_word_detector = WakeWordDetector()
        self.command_recognizer = HMMCommandRecognizer()
        self.digit_recognizer = DigitRecognizer()
        self.speaker_manager = SpeakerManager()
        self.vad = VAD()
        self.noise_estimator = NoiseEstimator()

        self.is_initialized = False
        self.detection_history = deque(maxlen=1000)
        self._initialize_default_models()

    def _initialize_default_models(self):
        wake_words = self.wake_word_detector.get_wake_word_list()
        commands = self.command_recognizer.get_command_list()
        digits = self.digit_recognizer.get_digit_list()

        if len(wake_words) > 0 or len(commands) > 0 or len(digits) > 0:
            self.is_initialized = True
            print(f"已加载 {len(wake_words)} 个唤醒词, {len(commands)} 个命令, {len(digits)} 个数字模型")
        else:
            print("警告: 未加载任何模型，请先注册唤醒词和训练命令模型")

    def detect_wake_word(self, signal, return_details=False):
        result = self.wake_word_detector.detect(signal, return_details=return_details)
        self._record_detection(
            wake_word=result.get("wake_word"),
            detected=result.get("detected", False),
            confidence=result.get("confidence", 0.0),
            duration=float(len(signal) / config.SAMPLE_RATE),
            noise_level=result.get("noise_level"),
            adaptive_threshold=result.get("adaptive_threshold")
        )
        return result

    def _record_detection(self, wake_word, detected, confidence, duration, noise_level=None, adaptive_threshold=None):
        record = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "timestamp_unix": time.time(),
            "wake_word": wake_word if detected else "未命中",
            "detected": detected,
            "confidence": float(confidence),
            "duration": float(duration),
            "noise_level": noise_level,
            "adaptive_threshold": adaptive_threshold
        }
        self.detection_history.append(record)

    def get_detection_history(self, limit=100, wake_word_filter=None):
        history = list(self.detection_history)
        if wake_word_filter and wake_word_filter != "全部":
            if wake_word_filter == "未命中":
                history = [r for r in history if not r["detected"]]
            else:
                history = [r for r in history if r["wake_word"] == wake_word_filter]
        history = history[-limit:]
        history.reverse()
        return history

    def get_detection_stats(self):
        history = list(self.detection_history)
        total = len(history)
        if total == 0:
            return {
                "total_detections": 0,
                "hit_rate": 0.0,
                "avg_confidence": 0.0,
                "wake_word_distribution": {}
            }

        hits = sum(1 for r in history if r["detected"])
        hit_rate = hits / total if total > 0 else 0.0
        avg_confidence = sum(r["confidence"] for r in history) / total if total > 0 else 0.0

        distribution = {}
        for r in history:
            ww = r["wake_word"]
            if ww not in distribution:
                distribution[ww] = 0
            distribution[ww] += 1

        return {
            "total_detections": total,
            "hit_rate": float(hit_rate),
            "avg_confidence": float(avg_confidence),
            "wake_word_distribution": distribution
        }

    def detect_wake_word_file(self, file_path, return_details=False):
        signal, sr = load_audio(file_path)
        return self.detect_wake_word(signal, return_details=return_details)

    def recognize_command(self, signal):
        command_result = self.command_recognizer.recognize(signal)
        command_name = command_result["command"]
        confidence = command_result["confidence"]

        params = {}
        top_candidates = command_result.get("top_candidates", [])
        rejection_reason = command_result.get("rejection_reason")

        if command_name != "unknown_command" and command_name != "unclear_command" and command_name in config.COMMANDS:
            command_config = config.COMMANDS[command_name]
            if command_config.get("has_number", False):
                param_name = command_config.get("param_name", "target")
                digit_result = self.digit_recognizer.recognize_digits_sequence(signal)
                if digit_result.get("number") is not None:
                    params[param_name] = digit_result["number"]
                    avg_confidence = (confidence + digit_result["confidence"]) / 2
                    confidence = avg_confidence

        result = {
            "command": command_name,
            "confidence": float(confidence),
            "params": params,
            "duration": float(len(signal) / config.SAMPLE_RATE),
            "top_candidates": top_candidates
        }

        if rejection_reason:
            result["rejection_reason"] = rejection_reason

        return result

    def recognize_command_file(self, file_path):
        signal, sr = load_audio(file_path)
        return self.recognize_command(signal)

    def full_recognition(self, signal):
        wake_result = self.detect_wake_word(signal, return_details=True)

        result = {
            "wake_word_detected": wake_result["detected"],
            "wake_word": wake_result["wake_word"],
            "wake_word_confidence": wake_result["confidence"],
            "command": None,
            "command_confidence": 0.0,
            "params": {},
            "duration": float(len(signal) / config.SAMPLE_RATE),
            "vad_result": wake_result.get("vad_result"),
            "noise_level": wake_result.get("noise_level"),
            "noise_rms": wake_result.get("noise_rms"),
            "adaptive_threshold": wake_result.get("adaptive_threshold"),
            "speaker_verified": False,
            "speaker_id": None,
            "speaker_confidence": 0.0,
            "confidence_level": "rejected",
            "blocked_by": None,
            "blocked_by_name": None
        }

        if wake_result["detected"]:
            speaker_result = self.speaker_manager.verify_speaker(signal)
            result["speaker_verified"] = speaker_result["verified"]
            result["speaker_id"] = speaker_result.get("speaker_id")
            result["speaker_confidence"] = speaker_result["confidence"]
            result["confidence_level"] = speaker_result.get("confidence_level", "rejected")
            result["blocked_by"] = speaker_result.get("blocked_by")
            result["blocked_by_name"] = speaker_result.get("blocked_by_name")

            if not speaker_result["verified"] and not speaker_result.get("skipped", False):
                result["wake_word_detected"] = False

        if result["wake_word_detected"]:
            sample_rate = config.SAMPLE_RATE
            window_duration = config.COMMAND_WINDOW_DURATION
            window_samples = int(window_duration * sample_rate)

            if len(signal) > window_samples:
                command_signal = signal[-window_samples:]
            else:
                command_signal = signal

            command_result = self.recognize_command(command_signal)
            result["command"] = command_result["command"]
            result["command_confidence"] = command_result["confidence"]
            result["params"] = command_result["params"]
            if "top_candidates" in command_result:
                result["top_candidates"] = command_result["top_candidates"]
            if "rejection_reason" in command_result:
                result["rejection_reason"] = command_result["rejection_reason"]

        return result

    def full_recognition_file(self, file_path):
        signal, sr = load_audio(file_path)
        return self.full_recognition(signal)

    def register_wake_word(self, name, audio_files):
        success, message = self.wake_word_detector.register_wake_word(name, audio_files)
        if success:
            self.is_initialized = True
        return success, message

    def add_wake_word_sample(self, name, audio_file):
        return self.wake_word_detector.add_sample(name, audio_file)

    def delete_wake_word(self, name):
        return self.wake_word_detector.delete_wake_word(name)

    def get_wake_words(self):
        return self.wake_word_detector.get_wake_word_list()

    def train_command(self, command_name, audio_files, num_states=5, num_mixtures=3):
        success, message = self.command_recognizer.train_command(
            command_name, audio_files, num_states, num_mixtures
        )
        if success:
            self.is_initialized = True
        return success, message

    def delete_command(self, command_name):
        return self.command_recognizer.delete_command(command_name)

    def get_commands(self):
        return self.command_recognizer.get_command_list()

    def train_digit(self, digit, audio_files, num_states=3, num_mixtures=2):
        return self.digit_recognizer.train_digit(digit, audio_files, num_states, num_mixtures)

    def get_digits(self):
        return self.digit_recognizer.get_digit_list()

    def register_speaker(self, name, audio_files, speaker_type="whitelist"):
        return self.speaker_manager.register_speaker(name, audio_files, speaker_type=speaker_type)

    def delete_speaker(self, speaker_id):
        return self.speaker_manager.delete_speaker(speaker_id)

    def get_speakers(self):
        return self.speaker_manager.get_speaker_list()

    def get_speaker_count(self):
        return self.speaker_manager.get_speaker_count()

    def get_verification_stats(self):
        return self.speaker_manager.get_verification_stats()

    def reset_verification_stats(self):
        return self.speaker_manager.reset_verification_stats()

    def verify_speaker(self, signal):
        return self.speaker_manager.verify_speaker(signal)

    def verify_speaker_file(self, file_path):
        return self.speaker_manager.verify_speaker_file(file_path)

    def reload_all_models(self):
        self.wake_word_detector.reload_models()
        self.command_recognizer.reload_models()
        self.digit_recognizer.reload_models()
        self.speaker_manager.reload_speakers()

        wake_words = self.wake_word_detector.get_wake_word_list()
        commands = self.command_recognizer.get_command_list()
        digits = self.digit_recognizer.get_digit_list()
        self.is_initialized = len(wake_words) > 0 or len(commands) > 0 or len(digits) > 0

    def get_system_status(self):
        env_status = self.wake_word_detector.get_environment_status()
        return {
            "is_initialized": self.is_initialized,
            "num_wake_words": len(self.wake_word_detector.get_wake_word_list()),
            "num_commands": len(self.command_recognizer.get_command_list()),
            "num_digits": len(self.digit_recognizer.get_digit_list()),
            "num_speakers": self.speaker_manager.get_speaker_count(),
            "speaker_verification_enabled": config.SPEAKER_VERIFICATION_ENABLED,
            "speaker_verification_threshold": config.SPEAKER_VERIFICATION_THRESHOLD,
            "speaker_high_confidence_threshold": config.SPEAKER_HIGH_CONFIDENCE_THRESHOLD,
            "speaker_medium_confidence_threshold": config.SPEAKER_MEDIUM_CONFIDENCE_THRESHOLD,
            "speaker_blacklist_threshold": config.SPEAKER_BLACKLIST_THRESHOLD,
            "speaker_ema_decay": config.SPEAKER_EMA_DECAY,
            "wake_word_threshold": config.WAKE_WORD_THRESHOLD,
            "command_confidence_threshold": config.COMMAND_CONFIDENCE_THRESHOLD,
            "sample_rate": config.SAMPLE_RATE,
            "environment": env_status
        }


class StreamingWakeWordDetector:
    def __init__(self, service, window_duration=1.5, step_duration=0.5):
        self.service = service
        self.window_duration = window_duration
        self.step_duration = step_duration
        self.sample_rate = config.SAMPLE_RATE

        self.window_samples = int(window_duration * self.sample_rate)
        self.step_samples = int(step_duration * self.sample_rate)

        self.buffer = np.array([], dtype=np.float32)
        self.last_detection_time = -1
        self.wake_word_detected = False
        self.detected_wake_word = None
        self.confidence_history = []

    def reset(self):
        self.buffer = np.array([], dtype=np.float32)
        self.last_detection_time = -1
        self.wake_word_detected = False
        self.detected_wake_word = None
        self.confidence_history = []

    def process_audio(self, audio_data):
        self.buffer = np.concatenate([self.buffer, audio_data])

        results = []

        while len(self.buffer) >= self.window_samples:
            window_signal = self.buffer[:self.window_samples]

            result = self.service.detect_wake_word(window_signal)
            current_time = (len(self.buffer) - self.window_samples) / self.sample_rate

            result["timestamp"] = current_time
            results.append(result)

            self.confidence_history.append({
                "time": current_time,
                "confidence": result["confidence"],
                "wake_word": result["wake_word"]
            })

            if result["detected"] and not self.wake_word_detected:
                self.wake_word_detected = True
                self.detected_wake_word = result["wake_word"]
                self.last_detection_time = current_time

            self.buffer = self.buffer[self.step_samples:]

        return results

    def get_buffer_duration(self):
        return len(self.buffer) / self.sample_rate

    def get_confidence_history(self, max_points=100):
        if len(self.confidence_history) <= max_points:
            return self.confidence_history
        step = len(self.confidence_history) // max_points
        return self.confidence_history[::step]
