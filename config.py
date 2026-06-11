import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SAMPLE_RATE = 16000
FRAME_LENGTH = 0.025
FRAME_SHIFT = 0.01
PREEMPHASIS_COEFF = 0.97
NUM_MFCC = 13
NUM_MEL_FILTERS = 40
FFT_SIZE = 512
WINDOW_TYPE = "hamming"

VAD_ENERGY_THRESHOLD = 0.005
VAD_ZCR_THRESHOLD = 0.05
VAD_MIN_SILENCE_DURATION = 0.3
VAD_MIN_SPEECH_DURATION = 0.1

WAKE_WORD_THRESHOLD = 0.75
DTW_WEIGHT = 0.6
CNN_WEIGHT = 0.4
MAX_WAKE_WORDS = 5

NOISE_LEVEL_QUIET_RMS = 0.01
NOISE_LEVEL_MODERATE_RMS = 0.05
ADAPTIVE_THRESHOLD_MIN = 0.60
ADAPTIVE_THRESHOLD_MAX = 0.90
NOISE_SMOOTHING_ALPHA = 0.5

COMMAND_REJECT_MARGIN = 0.15
COMMAND_MIN_CONFIDENCE = 0.3
TOP_K_CANDIDATES = 3

COMMAND_WINDOW_DURATION = 2.0
COMMAND_CONFIDENCE_THRESHOLD = 0.5

MODELS_DIR = os.path.join(BASE_DIR, "models")
WAKE_WORDS_DIR = os.path.join(MODELS_DIR, "wake_words")
COMMANDS_DIR = os.path.join(MODELS_DIR, "commands")
DIGITS_DIR = os.path.join(MODELS_DIR, "digits")
SPEAKERS_DIR = os.path.join(MODELS_DIR, "speakers")
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")
TEMP_DIR = os.path.join(BASE_DIR, "temp")

SPEAKER_VERIFICATION_ENABLED = True
SPEAKER_VERIFICATION_THRESHOLD = 0.85
SPEAKER_MIN_SAMPLES = 5
SPEAKER_MFCC_NUM = 13

MAX_CONCURRENT_STREAMS = 10
FRAME_PROCESS_TIMEOUT_MS = 50

CNN_INPUT_SHAPE = (39, None, 1)

COMMANDS = {
    "turn_on_light": {"patterns": ["打开灯", "开灯", "打开电灯"], "has_number": False},
    "turn_off_light": {"patterns": ["关闭灯", "关灯", "关闭电灯"], "has_number": False},
    "turn_on_ac": {"patterns": ["打开空调", "开空调"], "has_number": False},
    "turn_off_ac": {"patterns": ["关闭空调", "关空调"], "has_number": False},
    "set_temperature": {"patterns": ["设置温度到", "温度调到", "温度设为"], "has_number": True, "param_name": "target"},
    "play_music": {"patterns": ["播放音乐", "放音乐", "播放歌曲"], "has_number": False},
    "pause_music": {"patterns": ["暂停", "暂停音乐", "停止播放"], "has_number": False},
    "next_song": {"patterns": ["下一首", "下一首歌"], "has_number": False},
    "set_volume": {"patterns": ["音量调到", "音量设为", "音量调到"], "has_number": True, "param_name": "target"},
}
