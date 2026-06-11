import numpy as np
from scipy.signal import lfilter
from scipy.signal.windows import hamming
from scipy.fft import fft, ifft
import config


def preemphasis(signal, coeff=None):
    if coeff is None:
        coeff = config.PREEMPHASIS_COEFF
    return lfilter([1, -coeff], [1], signal)


def deemphasis(signal, coeff=None):
    if coeff is None:
        coeff = config.PREEMPHASIS_COEFF
    return lfilter([1], [1, -coeff], signal)


def framing(signal, sample_rate=None, frame_length=None, frame_shift=None):
    if sample_rate is None:
        sample_rate = config.SAMPLE_RATE
    if frame_length is None:
        frame_length = config.FRAME_LENGTH
    if frame_shift is None:
        frame_shift = config.FRAME_SHIFT

    frame_length_samples = int(round(frame_length * sample_rate))
    frame_shift_samples = int(round(frame_shift * sample_rate))
    signal_length = len(signal)

    if signal_length <= frame_length_samples:
        num_frames = 1
    else:
        num_frames = 1 + int(np.ceil((signal_length - frame_length_samples) / frame_shift_samples))

    pad_length = int((num_frames - 1) * frame_shift_samples + frame_length_samples)
    zeros = np.zeros((pad_length - signal_length,))
    pad_signal = np.concatenate((signal, zeros))

    indices = np.tile(np.arange(0, frame_length_samples), (num_frames, 1)) + \
              np.tile(np.arange(0, num_frames * frame_shift_samples, frame_shift_samples),
                      (frame_length_samples, 1)).T
    indices = np.array(indices, dtype=np.int32)
    frames = pad_signal[indices]

    return frames, frame_length_samples


def windowing(frames, window_type=None):
    if window_type is None:
        window_type = config.WINDOW_TYPE

    if window_type == "hamming":
        window = hamming(frames.shape[1])
    else:
        window = np.ones(frames.shape[1])

    return frames * window


def compute_energy(frames):
    return np.sum(frames ** 2, axis=1) / frames.shape[1]


def compute_zcr(frames):
    signs = np.sign(frames)
    signs[signs == 0] = -1
    return np.sum(np.abs(signs[:, 1:] - signs[:, :-1]), axis=1) / (2 * frames.shape[1])


class VAD:
    def __init__(self, sample_rate=None, energy_threshold=None, zcr_threshold=None,
                 min_silence_duration=None, min_speech_duration=None):
        self.sample_rate = sample_rate if sample_rate else config.SAMPLE_RATE
        self.energy_threshold = energy_threshold if energy_threshold else config.VAD_ENERGY_THRESHOLD
        self.zcr_threshold = zcr_threshold if zcr_threshold else config.VAD_ZCR_THRESHOLD
        self.min_silence_duration = min_silence_duration if min_silence_duration else config.VAD_MIN_SILENCE_DURATION
        self.min_speech_duration = min_speech_duration if min_speech_duration else config.VAD_MIN_SPEECH_DURATION

    def detect(self, signal):
        frames, _ = framing(signal, self.sample_rate)
        windowed_frames = windowing(frames)

        energy = compute_energy(windowed_frames)
        zcr = compute_zcr(windowed_frames)

        energy_max = np.max(energy)
        energy_min = np.min(energy)
        energy_range = energy_max - energy_min

        if energy_range > energy_max * 0.5 and len(energy) > 5:
            energy_sorted = np.sort(energy)
            noise_floor = np.mean(energy_sorted[:max(1, len(energy_sorted) // 5)])
            energy_threshold = max(self.energy_threshold, noise_floor * 2)
        else:
            energy_threshold = self.energy_threshold

        high_energy_mask = energy > energy_threshold
        low_energy_zcr_mask = (energy > energy_threshold * 0.5) & (zcr > self.zcr_threshold)
        speech_mask = high_energy_mask | low_energy_zcr_mask

        frame_shift = config.FRAME_SHIFT
        min_silence_frames = int(self.min_silence_duration / frame_shift)
        min_speech_frames = int(self.min_speech_duration / frame_shift)

        speech_mask = self._remove_short_silence(speech_mask, min_silence_frames)
        speech_mask = self._remove_short_speech(speech_mask, min_speech_frames)

        speech_segments = self._get_speech_segments(speech_mask, frame_shift)

        return {
            "speech_mask": speech_mask,
            "energy": energy,
            "zcr": zcr,
            "speech_segments": speech_segments,
            "num_frames": len(frames)
        }

    def _remove_short_silence(self, speech_mask, min_silence_frames):
        result = speech_mask.copy()
        i = 0
        while i < len(result):
            if not result[i]:
                j = i
                while j < len(result) and not result[j]:
                    j += 1
                if j - i < min_silence_frames and i > 0 and j < len(result):
                    result[i:j] = True
                i = j
            else:
                i += 1
        return result

    def _remove_short_speech(self, speech_mask, min_speech_frames):
        result = speech_mask.copy()
        i = 0
        while i < len(result):
            if result[i]:
                j = i
                while j < len(result) and result[j]:
                    j += 1
                if j - i < min_speech_frames:
                    result[i:j] = False
                i = j
            else:
                i += 1
        return result

    def _get_speech_segments(self, speech_mask, frame_shift):
        segments = []
        i = 0
        while i < len(speech_mask):
            if speech_mask[i]:
                start = i
                while i < len(speech_mask) and speech_mask[i]:
                    i += 1
                end = i
                segments.append({
                    "start_frame": start,
                    "end_frame": end,
                    "start_time": start * frame_shift,
                    "end_time": end * frame_shift
                })
            else:
                i += 1
        return segments

    def cut_speech(self, signal):
        result = self.detect(signal)
        if not result["speech_segments"]:
            return None, result

        frame_shift_samples = int(config.FRAME_SHIFT * self.sample_rate)
        speech_parts = []
        for seg in result["speech_segments"]:
            start_sample = seg["start_frame"] * frame_shift_samples
            end_sample = seg["end_frame"] * frame_shift_samples
            speech_parts.append(signal[start_sample:min(end_sample, len(signal))])

        if speech_parts:
            return np.concatenate(speech_parts), result
        return None, result


class NoiseEstimator:
    def __init__(self, sample_rate=None):
        self.sample_rate = sample_rate if sample_rate else config.SAMPLE_RATE
        self.noise_spectrum = None
        self.num_frames_estimated = 0

    def estimate_from_silence(self, signal, num_frames=10):
        frames, _ = framing(signal, self.sample_rate)
        windowed_frames = windowing(frames)

        if self.noise_spectrum is None:
            self.noise_spectrum = np.zeros(config.FFT_SIZE // 2 + 1)

        for i in range(min(num_frames, len(windowed_frames))):
            spectrum = np.abs(fft(windowed_frames[i], n=config.FFT_SIZE))[:config.FFT_SIZE // 2 + 1]
            self.noise_spectrum += spectrum
            self.num_frames_estimated += 1

        self.noise_spectrum /= self.num_frames_estimated

    def update(self, frame_spectrum, alpha=0.95):
        if self.noise_spectrum is None:
            self.noise_spectrum = frame_spectrum
            self.num_frames_estimated = 1
        else:
            self.noise_spectrum = alpha * self.noise_spectrum + (1 - alpha) * frame_spectrum
            self.num_frames_estimated += 1


def spectral_subtraction(signal, noise_estimator, sample_rate=None, alpha=2.0, beta=0.01):
    if sample_rate is None:
        sample_rate = config.SAMPLE_RATE

    frames, frame_length = framing(signal, sample_rate)
    windowed_frames = windowing(frames)

    enhanced_frames = []
    fft_size = config.FFT_SIZE

    for i in range(len(windowed_frames)):
        frame = windowed_frames[i]
        spectrum = fft(frame, n=fft_size)
        magnitude = np.abs(spectrum)
        phase = np.angle(spectrum)

        mag_half = magnitude[:fft_size // 2 + 1]

        if noise_estimator.noise_spectrum is not None:
            noise_mag = noise_estimator.noise_spectrum
            enhanced_mag = np.sqrt(np.maximum(mag_half ** alpha - noise_mag ** alpha, beta * mag_half ** alpha))
        else:
            enhanced_mag = mag_half

        enhanced_full = np.concatenate([enhanced_mag, enhanced_mag[-2:0:-1]])
        enhanced_spectrum = enhanced_full * np.exp(1j * phase)
        enhanced_frame = np.real(ifft(enhanced_spectrum, n=fft_size))[:frame_length]
        enhanced_frames.append(enhanced_frame)

    enhanced_signal = overlap_add(enhanced_frames, frame_length, int(config.FRAME_SHIFT * sample_rate))
    return enhanced_signal


def overlap_add(frames, frame_length, frame_shift):
    num_frames = len(frames)
    output_length = (num_frames - 1) * frame_shift + frame_length
    output = np.zeros(output_length)
    window_sum = np.zeros(output_length)

    for i in range(num_frames):
        start = i * frame_shift
        end = start + frame_length
        output[start:end] += frames[i]
        window_sum[start:end] += 1

    output = output / np.maximum(window_sum, 1e-10)
    return output


def mel_filterbank(num_filters=None, fft_size=None, sample_rate=None, low_freq=0, high_freq=None):
    if num_filters is None:
        num_filters = config.NUM_MEL_FILTERS
    if fft_size is None:
        fft_size = config.FFT_SIZE
    if sample_rate is None:
        sample_rate = config.SAMPLE_RATE
    if high_freq is None:
        high_freq = sample_rate / 2

    def hz_to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    def mel_to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    low_mel = hz_to_mel(low_freq)
    high_mel = hz_to_mel(high_freq)
    mel_points = np.linspace(low_mel, high_mel, num_filters + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((fft_size + 1) * hz_points / sample_rate).astype(int)

    filterbank = np.zeros((num_filters, fft_size // 2 + 1))
    for i in range(1, num_filters + 1):
        left = bin_points[i - 1]
        center = bin_points[i]
        right = bin_points[i + 1]

        for j in range(left, center):
            filterbank[i - 1, j] = (j - left) / (center - left)
        for j in range(center, right):
            filterbank[i - 1, j] = (right - j) / (right - center)

    return filterbank


def compute_mfcc(signal, sample_rate=None, num_mfcc=None, use_delta=True, use_delta_delta=True):
    if sample_rate is None:
        sample_rate = config.SAMPLE_RATE
    if num_mfcc is None:
        num_mfcc = config.NUM_MFCC

    preemphasized = preemphasis(signal)
    frames, frame_length = framing(preemphasized, sample_rate)
    windowed_frames = windowing(frames)

    fft_size = config.FFT_SIZE
    filterbank = mel_filterbank()

    mfcc_features = []

    for frame in windowed_frames:
        spectrum = fft(frame, n=fft_size)
        magnitude = np.abs(spectrum)[:fft_size // 2 + 1]
        power = magnitude ** 2

        mel_power = np.dot(filterbank, power)
        log_mel = np.log(np.maximum(mel_power, 1e-10))

        from scipy.fft import dct
        mfcc = dct(log_mel, type=2, norm='ortho')[:num_mfcc]
        mfcc_features.append(mfcc)

    mfcc_features = np.array(mfcc_features)

    if use_delta or use_delta_delta:
        features = [mfcc_features]

        if use_delta:
            delta = compute_delta(mfcc_features)
            features.append(delta)

        if use_delta_delta:
            delta_delta = compute_delta(features[1] if use_delta else compute_delta(mfcc_features))
            features.append(delta_delta if use_delta else compute_delta(features[0]))

        return np.concatenate(features, axis=1)

    return mfcc_features


def compute_delta(features, window=2):
    num_frames, num_features = features.shape
    delta = np.zeros_like(features)

    for i in range(num_frames):
        start = max(0, i - window)
        end = min(num_frames - 1, i + window)
        indices = np.arange(start, end + 1)
        weights = indices - i

        if np.sum(weights ** 2) != 0:
            delta[i] = np.dot(weights, features[start:end + 1]) / np.sum(weights ** 2)

    return delta


def load_audio(file_path, sample_rate=None):
    import soundfile as sf
    if sample_rate is None:
        sample_rate = config.SAMPLE_RATE

    signal, sr = sf.read(file_path)

    if len(signal.shape) > 1:
        signal = np.mean(signal, axis=1)

    if sr != sample_rate:
        from scipy.signal import resample
        num_samples = int(len(signal) * sample_rate / sr)
        signal = resample(signal, num_samples)

    if np.max(np.abs(signal)) > 0:
        signal = signal / np.max(np.abs(signal))

    return signal.astype(np.float32), sample_rate


def save_audio(file_path, signal, sample_rate=None):
    import soundfile as sf
    if sample_rate is None:
        sample_rate = config.SAMPLE_RATE
    sf.write(file_path, signal, sample_rate)


def compute_rms(signal):
    if len(signal) == 0:
        return 0.0
    return float(np.sqrt(np.mean(signal ** 2)))


def classify_noise_level(rms_value):
    if rms_value < config.NOISE_LEVEL_QUIET_RMS:
        return "quiet"
    elif rms_value < config.NOISE_LEVEL_MODERATE_RMS:
        return "moderate"
    else:
        return "noisy"


def compute_adaptive_threshold(noise_level, base_threshold=None):
    if base_threshold is None:
        base_threshold = config.WAKE_WORD_THRESHOLD

    if noise_level == "quiet":
        adjustment = -0.10
    elif noise_level == "moderate":
        adjustment = 0.0
    else:
        adjustment = 0.10

    adaptive = base_threshold + adjustment
    return float(max(config.ADAPTIVE_THRESHOLD_MIN, min(config.ADAPTIVE_THRESHOLD_MAX, adaptive)))
