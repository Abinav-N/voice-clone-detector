"""
Audio data augmentation for voice clone detection training and robustness testing.
Includes:
- Additive noise at controlled SNR (dB)
- Gain modification
- Time shifting
- Codec compression simulation (telephone 8kHz resample and mu-law companding)
"""
import random
import numpy as np
import librosa


def add_noise_at_snr(y: np.ndarray, snr_db: float) -> np.ndarray:
    """Add zero-mean Gaussian noise at the specified Signal-to-Noise Ratio (dB)."""
    p_signal = np.mean(y ** 2) + 1e-10
    p_noise = p_signal / (10.0 ** (snr_db / 10.0))
    noise = np.random.randn(len(y)).astype(y.dtype) * np.sqrt(p_noise)
    return y + noise


def change_gain(y: np.ndarray, min_gain_db: float = -6.0, max_gain_db: float = 6.0) -> np.ndarray:
    """Apply random gain in decibels."""
    gain_db = np.random.uniform(min_gain_db, max_gain_db)
    gain_factor = 10.0 ** (gain_db / 20.0)
    return y * gain_factor


def shift_time(y: np.ndarray, max_shift_sec: float = 0.25, sr: int = 16000) -> np.ndarray:
    """Randomly shift audio in time with circular roll."""
    shift = int(np.random.uniform(-max_shift_sec, max_shift_sec) * sr)
    if shift == 0:
        return y
    return np.roll(y, shift)


def simulate_codec_compression(y: np.ndarray, sr: int = 16000, codec: str = "random") -> np.ndarray:
    """
    Simulate speech codec compression artifacts:
    - telephone: 8kHz downsample and restore (G.711 / narrowband telephone channel)
    - mu_law: 8-bit non-linear quantization
    """
    if codec == "random":
        codec = random.choice(["telephone", "mu_law"])

    if codec == "telephone":
        y_8k = librosa.resample(y, orig_sr=sr, target_sr=8000)
        return librosa.resample(y_8k, orig_sr=8000, target_sr=sr)
    elif codec == "mu_law":
        mu = 255.0
        y_clip = np.clip(y, -1.0, 1.0)
        comp = np.sign(y_clip) * np.log1p(mu * np.abs(y_clip)) / np.log1p(mu)
        quant = np.round((comp + 1.0) * 127.5) / 127.5 - 1.0
        decomp = np.sign(quant) * ((1.0 + mu) ** np.abs(quant) - 1.0) / mu
        return decomp.astype(y.dtype)
    return y


def apply_augmentations(y: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    Apply stochastic audio data augmentation for robust training.
    """
    # 1. Gain change (50% probability)
    if random.random() < 0.5:
        y = change_gain(y)

    # 2. Time shift (50% probability)
    if random.random() < 0.5:
        y = shift_time(y, max_shift_sec=0.2, sr=sr)

    # 3. Additive noise (50% probability with SNR between 10 dB and 30 dB)
    if random.random() < 0.5:
        snr_db = random.choice([10.0, 15.0, 20.0, 25.0, 30.0])
        y = add_noise_at_snr(y, snr_db)

    # 4. Codec / compression artifacts (40% probability)
    if random.random() < 0.4:
        y = simulate_codec_compression(y, sr=sr)

    # Normalize max amplitude to avoid clipping distortion
    max_val = np.max(np.abs(y)) + 1e-6
    if max_val > 1.0:
        y = y / max_val

    return y.astype(np.float32)
