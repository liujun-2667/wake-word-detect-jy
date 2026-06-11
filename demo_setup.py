import numpy as np
import os
import sys
import config
from core.audio_preprocessing import save_audio, load_audio, compute_mfcc, VAD
from core.speech_service import SpeechRecognitionService


def generate_synthetic_audio(duration=1.0, freq=440, sample_rate=None):
    if sample_rate is None:
        sample_rate = config.SAMPLE_RATE
    t = np.linspace(0, duration, int(sample_rate * duration))
    signal = 0.5 * np.sin(2 * np.pi * freq * t)

    noise = np.random.randn(len(signal)) * 0.02
    signal = signal + noise

    signal = signal / np.max(np.abs(signal)) * 0.8
    return signal.astype(np.float32)


def generate_speech_like_signal(duration=1.0, base_freq=200, sample_rate=None):
    if sample_rate is None:
        sample_rate = config.SAMPLE_RATE

    t = np.linspace(0, duration, int(sample_rate * duration))

    signal = np.zeros_like(t)

    num_formants = 3
    formant_freqs = [base_freq, base_freq * 2.5, base_freq * 3.8]
    formant_amps = [1.0, 0.6, 0.3]

    for freq, amp in zip(formant_freqs, formant_amps):
        signal += amp * np.sin(2 * np.pi * freq * t)

    mod_freq = 3
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * mod_freq * t)
    signal = signal * envelope

    noise = np.random.randn(len(signal)) * 0.05
    signal = signal + noise

    fade_len = int(0.05 * sample_rate)
    fade_in = np.linspace(0, 1, fade_len)
    fade_out = np.linspace(1, 0, fade_len)
    signal[:fade_len] *= fade_in
    signal[-fade_len:] *= fade_out

    signal = signal / np.max(np.abs(signal)) * 0.8
    return signal.astype(np.float32)


def setup_demo_data():
    print("正在创建演示数据...")

    os.makedirs(config.SAMPLES_DIR, exist_ok=True)
    os.makedirs(config.TEMP_DIR, exist_ok=True)

    wake_words = {
        "xiao_zhu_shou": 220,
        "ni_hao_xiao_yi": 180,
    }

    for name, freq in wake_words.items():
        sample_dir = os.path.join(config.SAMPLES_DIR, f"wake_word_{name}")
        os.makedirs(sample_dir, exist_ok=True)

        for i in range(4):
            duration = 0.8 + np.random.random() * 0.4
            freq_var = freq * (0.95 + np.random.random() * 0.1)
            signal = generate_speech_like_signal(duration, freq_var)

            silence_len = int(0.1 * config.SAMPLE_RATE)
            silence = np.zeros(silence_len, dtype=np.float32)
            full_signal = np.concatenate([silence, signal, silence])

            file_path = os.path.join(sample_dir, f"sample_{i+1}.wav")
            save_audio(file_path, full_signal)
            print(f"  生成: {file_path}")

    commands = {
        "turn_on_light": 250,
        "turn_off_light": 270,
        "play_music": 200,
        "pause_music": 220,
    }

    for cmd_name, freq in commands.items():
        sample_dir = os.path.join(config.SAMPLES_DIR, f"cmd_{cmd_name}")
        os.makedirs(sample_dir, exist_ok=True)

        for i in range(3):
            duration = 1.0 + np.random.random() * 0.5
            freq_var = freq * (0.95 + np.random.random() * 0.1)
            signal = generate_speech_like_signal(duration, freq_var)

            silence_len = int(0.1 * config.SAMPLE_RATE)
            silence = np.zeros(silence_len, dtype=np.float32)
            full_signal = np.concatenate([silence, signal, silence])

            file_path = os.path.join(sample_dir, f"sample_{i+1}.wav")
            save_audio(file_path, full_signal)
            print(f"  生成: {file_path}")

    digits = [str(d) for d in range(10)]
    for digit in digits:
        sample_dir = os.path.join(config.SAMPLES_DIR, f"digit_{digit}")
        os.makedirs(sample_dir, exist_ok=True)

        base_freq = 300 + int(digit) * 30

        for i in range(3):
            duration = 0.5 + np.random.random() * 0.2
            freq_var = base_freq * (0.95 + np.random.random() * 0.1)
            signal = generate_speech_like_signal(duration, freq_var)

            silence_len = int(0.05 * config.SAMPLE_RATE)
            silence = np.zeros(silence_len, dtype=np.float32)
            full_signal = np.concatenate([silence, signal, silence])

            file_path = os.path.join(sample_dir, f"sample_{i+1}.wav")
            save_audio(file_path, full_signal)
            print(f"  生成: {file_path}")

    print("\n演示数据创建完成!")
    return True


def train_demo_models():
    print("\n正在训练演示模型...")

    service = SpeechRecognitionService()

    print("\n=== 训练唤醒词 ===")
    wake_words = {
        "xiao_zhu_shou": "wake_word_xiao_zhu_shou",
        "ni_hao_xiao_yi": "wake_word_ni_hao_xiao_yi",
    }

    for name, dir_name in wake_words.items():
        sample_dir = os.path.join(config.SAMPLES_DIR, dir_name)
        samples = [os.path.join(sample_dir, f"sample_{i+1}.wav") for i in range(4)]

        print(f"  训练唤醒词: {name}...")
        success, msg = service.register_wake_word(name, samples)
        print(f"    结果: {msg}")

    print("\n=== 训练命令模型 ===")
    commands = {
        "turn_on_light": "cmd_turn_on_light",
        "turn_off_light": "cmd_turn_off_light",
        "play_music": "cmd_play_music",
        "pause_music": "cmd_pause_music",
    }

    for cmd_name, dir_name in commands.items():
        sample_dir = os.path.join(config.SAMPLES_DIR, dir_name)
        samples = [os.path.join(sample_dir, f"sample_{i+1}.wav") for i in range(3)]

        print(f"  训练命令: {cmd_name}...")
        success, msg = service.train_command(cmd_name, samples, num_states=4, num_mixtures=2)
        print(f"    结果: {msg}")

    print("\n=== 训练数字模型 ===")
    for digit in range(10):
        dir_name = f"digit_{digit}"
        sample_dir = os.path.join(config.SAMPLES_DIR, dir_name)
        samples = [os.path.join(sample_dir, f"sample_{i+1}.wav") for i in range(3)]

        print(f"  训练数字: {digit}...")
        success, msg = service.train_digit(digit, samples, num_states=3, num_mixtures=2)
        print(f"    结果: {msg}")

    print("\n演示模型训练完成!")
    return True


def run_demo_test():
    print("\n=== 运行演示测试 ===")

    service = SpeechRecognitionService()

    test_wake_word = os.path.join(config.SAMPLES_DIR, "wake_word_xiao_zhu_shou", "sample_1.wav")
    test_command = os.path.join(config.SAMPLES_DIR, "cmd_turn_on_light", "sample_1.wav")

    print("\n1. 唤醒词检测测试:")
    result = service.detect_wake_word_file(test_wake_word)
    print(f"   检测结果: {'检测到' if result['detected'] else '未检测到'}")
    print(f"   唤醒词: {result.get('wake_word', 'N/A')}")
    print(f"   置信度: {result['confidence']:.4f}")

    print("\n2. 命令识别测试:")
    result = service.recognize_command_file(test_command)
    print(f"   命令: {result['command']}")
    print(f"   置信度: {result['confidence']:.4f}")
    print(f"   参数: {result['params']}")

    print("\n3. 完整识别测试 (合成唤醒词+命令):")
    ww_signal, sr = load_audio(test_wake_word)
    cmd_signal, _ = load_audio(test_command)
    full_signal = np.concatenate([ww_signal, cmd_signal])

    full_path = os.path.join(config.TEMP_DIR, "demo_full_test.wav")
    save_audio(full_path, full_signal)

    result = service.full_recognition_file(full_path)
    print(f"   唤醒词检测: {'是' if result['wake_word_detected'] else '否'}")
    print(f"   唤醒词: {result.get('wake_word', 'N/A')}")
    print(f"   命令: {result.get('command', 'N/A')}")
    print(f"   命令置信度: {result.get('command_confidence', 0):.4f}")
    print(f"   音频时长: {result['duration']:.2f}秒")

    print("\n演示测试完成!")


if __name__ == "__main__":
    setup_demo_data()
    train_demo_models()
    run_demo_test()

    print("\n" + "=" * 50)
    print("🎉 演示设置完成!")
    print("=" * 50)
    print("\n启动后端服务:")
    print("  python server/app.py")
    print("\n启动Web界面:")
    print("  streamlit run app.py")
