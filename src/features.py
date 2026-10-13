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


def precompute_features_for_manifest(manifest_csv: str, cache_dir: str):
    """Precompute MFCC stack (.npy) for all audio files listed in a manifest CSV."""
    import os
    import hashlib
    from pathlib import Path
    import pandas as pd
    from tqdm import tqdm

    df = pd.read_csv(manifest_csv)
    if "filepath" not in df.columns:
        raise ValueError(f"Manifest {manifest_csv} must have 'filepath' column.")

    os.makedirs(cache_dir, exist_ok=True)
    print(f"Precomputing features for {len(df)} files from '{manifest_csv}' -> '{cache_dir}'")
    for fpath in tqdm(df["filepath"], desc="Caching MFCCs"):
        abs_path = os.path.abspath(fpath)
        h = hashlib.sha256(abs_path.encode("utf-8")).hexdigest()[:16]
        stem = Path(fpath).stem[:24]
        out_file = os.path.join(cache_dir, f"{stem}_{h}.npy")
        if not os.path.isfile(out_file):
            feat = extract_features_from_path(fpath)
            np.save(out_file, feat)
    print("Done precomputing features.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Extract and cache MFCC features.")
    ap.add_argument("--manifest", help="Manifest CSV to precompute features for")
    ap.add_argument("--cache-dir", default=".cache/features", help="Output directory for .npy caches")
    ap.add_argument("--test-audio", help="Optional test audio file to print output shape for")
    args = ap.parse_args()

    if args.manifest:
        precompute_features_for_manifest(args.manifest, args.cache_dir)
    elif args.test_audio:
        feat = extract_features_from_path(args.test_audio)
        print(f"Extracted features shape: {feat.shape} for {args.test_audio}")
    else:
        # Default synthetic sanity check
        dummy_y = np.random.randn(SAMPLE_RATE * 3).astype(np.float32)
        out = extract_mfcc_stack(dummy_y)
        print("Sanity check passed. Synthetic features shape:", out.shape)
