import numpy as np
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from core.audio_preprocessing import compute_mfcc, VAD, framing, windowing, preemphasis
from core.dtw import dtw_similarity
from core.speech_service import SpeechRecognitionService


def test_frame_processing_time():
    print("=== 单帧处理性能测试 ===")

    sample_rate = config.SAMPLE_RATE
    frame_length = config.FRAME_LENGTH
    frame_shift = config.FRAME_SHIFT

    signal = np.random.randn(int(sample_rate * 10)).astype(np.float32)
    signal = signal / np.max(np.abs(signal)) * 0.5

    preemphasized = preemphasis(signal)
    frames, _ = framing(preemphasized, sample_rate)
    windowed = windowing(frames)

    single_frame = windowed[0]

    start_time = time.time()
    iterations = 1000
    for _ in range(iterations):
        from scipy.fft import fft
        fft_size = config.FFT_SIZE
        _ = fft(single_frame, n=fft_size)
    end_time = time.time()

    fft_time = (end_time - start_time) / iterations * 1000
    print(f"  FFT处理时间: {fft_time:.3f} ms/帧")

    short_signal = np.random.randn(int(sample_rate * 0.1)).astype(np.float32)
    start_time = time.time()
    for _ in range(100):
        _ = compute_mfcc(short_signal)
    end_time = time.time()

    mfcc_time = (end_time - start_time) / 100 * 1000
    print(f"  MFCC处理时间 (100ms音频): {mfcc_time:.3f} ms")

    vad = VAD()
    start_time = time.time()
    for _ in range(100):
        _ = vad.detect(signal[:int(sample_rate * 0.5)])
    end_time = time.time()

    vad_time = (end_time - start_time) / 100 * 1000
    print(f"  VAD检测时间 (0.5秒音频): {vad_time:.3f} ms")

    return fft_time < 50 and mfcc_time < 100


def test_mfcc_extraction():
    print("\n=== MFCC特征提取测试 ===")

    sample_rate = config.SAMPLE_RATE
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    signal = 0.5 * np.sin(2 * np.pi * 440 * t)
    signal = signal.astype(np.float32)

    start_time = time.time()
    mfcc = compute_mfcc(signal)
    end_time = time.time()

    processing_time = (end_time - start_time) * 1000
    print(f"  1秒音频MFCC提取时间: {processing_time:.2f} ms")
    print(f"  MFCC特征形状: {mfcc.shape}")
    print(f"  帧数: {mfcc.shape[0]}, 特征维度: {mfcc.shape[1]}")

    expected_frames = int((duration - config.FRAME_LENGTH) / config.FRAME_SHIFT) + 1
    print(f"  预期帧数: {expected_frames}")

    return mfcc.shape[1] == 39


def test_dtw_performance():
    print("\n=== DTW匹配性能测试 ===")

    seq1 = np.random.randn(50, 39).astype(np.float32)
    seq2 = np.random.randn(60, 39).astype(np.float32)

    start_time = time.time()
    iterations = 100
    for _ in range(iterations):
        _ = dtw_similarity(seq1, seq2)
    end_time = time.time()

    dtw_time = (end_time - start_time) / iterations * 1000
    print(f"  单次DTW匹配时间: {dtw_time:.2f} ms")
    print(f"  序列长度: {len(seq1)} vs {len(seq2)}")

    return dtw_time < 100


def test_vad():
    print("\n=== VAD语音活动检测测试 ===")

    sample_rate = config.SAMPLE_RATE
    duration = 3.0
    t = np.linspace(0, duration, int(sample_rate * duration))

    signal = np.random.randn(len(t)) * 0.01
    speech_start = int(0.5 * sample_rate)
    speech_end = int(2.0 * sample_rate)
    t_speech = np.linspace(0, 1.5, speech_end - speech_start)
    signal[speech_start:speech_end] += 0.5 * np.sin(2 * np.pi * 300 * t_speech)

    signal = signal.astype(np.float32)

    vad = VAD()
    start_time = time.time()
    result = vad.detect(signal)
    end_time = time.time()

    processing_time = (end_time - start_time) * 1000
    print(f"  3秒音频VAD处理时间: {processing_time:.2f} ms")
    print(f"  检测到语音段数量: {len(result['speech_segments'])}")

    if result["speech_segments"]:
        for i, seg in enumerate(result["speech_segments"]):
            print(f"    段{i+1}: {seg['start_time']:.2f}s - {seg['end_time']:.2f}s")

    return len(result["speech_segments"]) >= 1


def test_full_pipeline():
    print("\n=== 完整处理流水线测试 ===")

    sample_rate = config.SAMPLE_RATE
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    signal = 0.5 * np.sin(2 * np.pi * 220 * t)
    noise = np.random.randn(len(signal)) * 0.05
    signal = (signal + noise).astype(np.float32)

    start_time = time.time()

    vad = VAD()
    vad_result = vad.detect(signal)

    mfcc = compute_mfcc(signal)

    end_time = time.time()

    processing_time = (end_time - start_time) * 1000
    print(f"  2秒音频完整处理时间: {processing_time:.2f} ms")
    print(f"  实时率: {processing_time / (duration * 1000):.2f}x 实时")

    return processing_time < duration * 1000


def test_concurrent_simulation():
    print("\n=== 并发处理模拟测试 ===")

    sample_rate = config.SAMPLE_RATE
    num_streams = 5

    signals = []
    for i in range(num_streams):
        duration = 1.0 + i * 0.2
        t = np.linspace(0, duration, int(sample_rate * duration))
        signal = 0.5 * np.sin(2 * np.pi * (200 + i * 50) * t)
        signals.append(signal.astype(np.float32))

    start_time = time.time()

    results = []
    for signal in signals:
        mfcc = compute_mfcc(signal)
        results.append(mfcc.shape)

    end_time = time.time()

    total_time = (end_time - start_time) * 1000
    avg_time = total_time / num_streams

    print(f"  处理 {num_streams} 个流总时间: {total_time:.2f} ms")
    print(f"  平均每流处理时间: {avg_time:.2f} ms")

    return True


def run_all_tests():
    print("=" * 60)
    print("语音唤醒词检测系统 - 性能测试")
    print("=" * 60)

    tests = [
        ("单帧处理时间", test_frame_processing_time),
        ("MFCC特征提取", test_mfcc_extraction),
        ("DTW匹配性能", test_dtw_performance),
        ("VAD语音检测", test_vad),
        ("完整处理流水线", test_full_pipeline),
        ("并发处理模拟", test_concurrent_simulation),
    ]

    passed = 0
    total = len(tests)

    for name, test_func in tests:
        try:
            result = test_func()
            status = "[PASS] 通过" if result else "[WARN] 待优化"
            print(f"\n  {name}: {status}")
            if result:
                passed += 1
        except Exception as e:
            print(f"\n  {name}: [ERROR] 错误 - {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"测试结果: {passed}/{total} 项通过")
    print("=" * 60)

    if passed == total:
        print("\n所有性能测试通过!")
    else:
        print(f"\n有 {total - passed} 项需要优化")

    return passed == total


if __name__ == "__main__":
    run_all_tests()
