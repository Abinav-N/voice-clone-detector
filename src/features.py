"""
Feature extraction: waveform -> (3, n_mfcc, T) tensor of
[MFCC, delta-MFCC, delta2-MFCC], fixed-length padded/cropped.

This representation is deliberately simple and standard (same family of
features used in early ASVspoof baselines) so it's easy to explain and
easy to debug.
"""
import numpy as np
import librosa

SAMPLE_RATE = 16000
N_MFCC = 40
N_FFT = 512
HOP_LENGTH = 160          # 10ms hop at 16kHz
WIN_LENGTH = 400          # 25ms window at 16kHz
FIXED_FRAMES = 300        # ~3 seconds of audio; pad/crop everything to this


def load_audio(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load audio, force mono, resample to target sample rate."""
    y, orig_sr = librosa.load(path, sr=sr, mono=True)
    return y


def extract_mfcc_stack(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Returns array of shape (3, N_MFCC, FIXED_FRAMES):
    channel 0 = MFCC, channel 1 = delta, channel 2 = delta-delta.
    """
    if len(y) < WIN_LENGTH:
        # pad very short clips before framing
        y = np.pad(y, (0, WIN_LENGTH - len(y)))

    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT,
        hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
    )
    delta = librosa.feature.delta(mfcc, order=1)
    delta2 = librosa.feature.delta(mfcc, order=2)

    stacked = np.stack([mfcc, delta, delta2], axis=0)  # (3, n_mfcc, T)
    stacked = _pad_or_crop(stacked, FIXED_FRAMES)

    # per-utterance normalization (helps a lot with cross-dataset generalization)
    mean = stacked.mean(axis=(1, 2), keepdims=True)
    std = stacked.std(axis=(1, 2), keepdims=True) + 1e-6
    stacked = (stacked - mean) / std

    return stacked.astype(np.float32)


def _pad_or_crop(feat: np.ndarray, target_frames: int) -> np.ndarray:
    """Pad with zeros or crop along the time axis (last axis) to target_frames."""
    t = feat.shape[-1]
    if t == target_frames:
        return feat
    if t > target_frames:
        # take a random-ish but deterministic center crop
        start = (t - target_frames) // 2
        return feat[..., start:start + target_frames]
    pad_width = target_frames - t
    return np.pad(feat, ((0, 0), (0, 0), (0, pad_width)), mode="constant")


def extract_features_from_path(path: str) -> np.ndarray:
    y = load_audio(path)
    return extract_mfcc_stack(y)
