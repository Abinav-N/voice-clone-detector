"""
Feature extraction: waveform -> (3, n_mfcc, T) tensor of
[feature, delta, delta-delta], fixed-length padded/cropped.

Supports both:
- 'mfcc': 40 Mel-Frequency Cepstral Coefficients + derivatives
- 'logmel': 40 Log-Mel Spectrogram bands + derivatives
And sliding window extraction for long audio clips (>3 seconds).
"""
import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SRC_DIR.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import librosa
try:
    from augment import apply_augmentations
except ImportError:
    from src.augment import apply_augmentations

SAMPLE_RATE = 16000
N_MFCC = 40
N_FFT = 512
HOP_LENGTH = 160          # 10ms hop at 16kHz
WIN_LENGTH = 400          # 25ms window at 16kHz
FIXED_FRAMES = 300        # ~3 seconds of audio; pad/crop everything to this
WINDOW_SAMPLES = SAMPLE_RATE * 3   # 48,000 samples for 3 seconds


def load_audio(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load audio, force mono, resample to target sample rate."""
    y, orig_sr = librosa.load(path, sr=sr, mono=True)
    return y


def _normalize_tensor(stacked: np.ndarray) -> np.ndarray:
    """Per-utterance channel-wise normalization."""
    mean = stacked.mean(axis=(1, 2), keepdims=True)
    std = stacked.std(axis=(1, 2), keepdims=True) + 1e-6
    stacked = (stacked - mean) / std
    return stacked.astype(np.float32)


def extract_mfcc_stack(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Returns array of shape (3, N_MFCC, FIXED_FRAMES):
    channel 0 = MFCC, channel 1 = delta, channel 2 = delta-delta.
    """
    if len(y) < WIN_LENGTH:
        y = np.pad(y, (0, WIN_LENGTH - len(y)))

    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT,
        hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
    )
    delta = librosa.feature.delta(mfcc, order=1)
    delta2 = librosa.feature.delta(mfcc, order=2)

    stacked = np.stack([mfcc, delta, delta2], axis=0)  # (3, n_mfcc, T)
    stacked = _pad_or_crop(stacked, FIXED_FRAMES)
    return _normalize_tensor(stacked)


def extract_logmel_stack(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Returns array of shape (3, N_MFCC, FIXED_FRAMES):
    channel 0 = Log-Mel Spectrogram, channel 1 = delta, channel 2 = delta-delta.
    """
    if len(y) < WIN_LENGTH:
        y = np.pad(y, (0, WIN_LENGTH - len(y)))

    melspec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MFCC, n_fft=N_FFT,
        hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
    )
    logmel = librosa.power_to_db(melspec, ref=np.max)
    delta = librosa.feature.delta(logmel, order=1)
    delta2 = librosa.feature.delta(logmel, order=2)

    stacked = np.stack([logmel, delta, delta2], axis=0)
    stacked = _pad_or_crop(stacked, FIXED_FRAMES)
    return _normalize_tensor(stacked)


def extract_features(y: np.ndarray, feature_type: str = "mfcc", sr: int = SAMPLE_RATE) -> np.ndarray:
    """Extract features according to feature_type ('mfcc' or 'logmel')."""
    if feature_type == "logmel":
        return extract_logmel_stack(y, sr=sr)
    elif feature_type == "mfcc":
        return extract_mfcc_stack(y, sr=sr)
    else:
        raise ValueError(f"Unknown feature_type: {feature_type}. Expected 'mfcc' or 'logmel'.")


def _pad_or_crop(feat: np.ndarray, target_frames: int) -> np.ndarray:
    """Pad with zeros or crop along the time axis (last axis) to target_frames."""
    t = feat.shape[-1]
    if t == target_frames:
        return feat
    if t > target_frames:
        start = (t - target_frames) // 2
        return feat[..., start:start + target_frames]
    pad_width = target_frames - t
    return np.pad(feat, ((0, 0), (0, 0), (0, pad_width)), mode="constant")


def extract_features_from_path(
    path: str,
    feature_type: str = "mfcc",
    augment: bool = False,
) -> np.ndarray:
    """Load audio and extract features with optional online data augmentation."""
    y = load_audio(path)
    if augment:
        y = apply_augmentations(y, sr=SAMPLE_RATE)
    return extract_features(y, feature_type=feature_type)


def extract_sliding_windows(
    y: np.ndarray,
    feature_type: str = "mfcc",
    window_sec: float = 3.0,
    hop_sec: float = 1.5,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    Slide a window across long audio clips and return an array of feature tensors.
    Output shape: (num_windows, 3, 40, FIXED_FRAMES).
    If audio <= window_sec, returns (1, 3, 40, FIXED_FRAMES).
    """
    win_len = int(window_sec * sr)
    hop_len = int(hop_sec * sr)

    if len(y) <= win_len:
        feat = extract_features(y, feature_type=feature_type, sr=sr)
        return np.expand_dims(feat, axis=0)

    # Slide across waveform
    chunks = []
    start = 0
    while start + win_len <= len(y):
        chunk_y = y[start:start + win_len]
        chunks.append(extract_features(chunk_y, feature_type=feature_type, sr=sr))
        start += hop_len

    # If the end had a tail that wasn't covered, take final chunk
    if start < len(y) and len(y) - start >= WIN_LENGTH:
        chunk_y = y[-win_len:]
        chunks.append(extract_features(chunk_y, feature_type=feature_type, sr=sr))

    return np.stack(chunks, axis=0)  # (N, 3, 40, 300)


def extract_sliding_windows_from_path(
    path: str,
    feature_type: str = "mfcc",
    window_sec: float = 3.0,
    hop_sec: float = 1.5,
) -> np.ndarray:
    """Load long audio and extract sliding window features."""
    y = load_audio(path)
    return extract_sliding_windows(y, feature_type=feature_type, window_sec=window_sec, hop_sec=hop_sec)


def precompute_features_for_manifest(manifest_csv: str, cache_dir: str, feature_type: str = "mfcc"):
    """Precompute feature stacks (.npy) for all audio files listed in a manifest CSV."""
    import os
    import hashlib
    from pathlib import Path
    import pandas as pd
    from tqdm import tqdm

    df = pd.read_csv(manifest_csv)
    if "filepath" not in df.columns:
        raise ValueError(f"Manifest {manifest_csv} must have 'filepath' column.")

    os.makedirs(cache_dir, exist_ok=True)
    print(f"Precomputing {feature_type} features for {len(df)} files from '{manifest_csv}' -> '{cache_dir}'")
    for fpath in tqdm(df["filepath"], desc=f"Caching {feature_type}"):
        abs_path = os.path.abspath(fpath)
        h = hashlib.sha256(f"{abs_path}_{feature_type}".encode("utf-8")).hexdigest()[:16]
        stem = Path(fpath).stem[:24]
        out_file = os.path.join(cache_dir, f"{stem}_{feature_type}_{h}.npy")
        if not os.path.isfile(out_file):
            feat = extract_features_from_path(fpath, feature_type=feature_type)
            np.save(out_file, feat)
    print("Done precomputing features.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Extract and cache audio features.")
    ap.add_argument("--manifest", help="Manifest CSV to precompute features for")
    ap.add_argument("--feature", choices=["mfcc", "logmel"], default="mfcc", help="Feature type")
    ap.add_argument("--cache-dir", default=".cache/features", help="Output directory for .npy caches")
    ap.add_argument("--test-audio", help="Optional test audio file to print output shape for")
    args = ap.parse_args()

    if args.manifest:
        precompute_features_for_manifest(args.manifest, args.cache_dir, feature_type=args.feature)
    elif args.test_audio:
        feat = extract_features_from_path(args.test_audio, feature_type=args.feature)
        print(f"Extracted {args.feature} shape: {feat.shape} for {args.test_audio}")
    else:
        dummy_y = np.random.randn(SAMPLE_RATE * 3).astype(np.float32)
        out_mfcc = extract_features(dummy_y, "mfcc")
        out_logmel = extract_features(dummy_y, "logmel")
        print(f"Sanity check: MFCC shape={out_mfcc.shape}, Log-mel shape={out_logmel.shape}")
