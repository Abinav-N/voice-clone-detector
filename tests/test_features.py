"""
Unit tests for feature extraction and preprocessing (Task 22).
"""
import numpy as np
import pytest

from features import (
    extract_mfcc_stack,
    extract_logmel_stack,
    extract_features,
    _pad_or_crop,
    _normalize_tensor,
    extract_features_from_path,
    extract_sliding_windows,
    SAMPLE_RATE,
    N_MFCC,
    FIXED_FRAMES,
)
from augment import (
    add_noise_at_snr,
    change_gain,
    shift_time,
    simulate_codec_compression,
    apply_augmentations,
)


def test_pad_or_crop():
    # Short tensor: pad to 300
    short_feat = np.ones((3, 40, 150), dtype=np.float32)
    padded = _pad_or_crop(short_feat, target_frames=FIXED_FRAMES)
    assert padded.shape == (3, 40, FIXED_FRAMES)
    assert np.all(padded[:, :, :150] == 1.0)
    assert np.all(padded[:, :, 150:] == 0.0)

    # Long tensor: crop to 300
    long_feat = np.ones((3, 40, 500), dtype=np.float32)
    cropped = _pad_or_crop(long_feat, target_frames=FIXED_FRAMES)
    assert cropped.shape == (3, 40, FIXED_FRAMES)

    # Exact size: unchanged
    exact_feat = np.ones((3, 40, FIXED_FRAMES), dtype=np.float32)
    same = _pad_or_crop(exact_feat, target_frames=FIXED_FRAMES)
    assert same.shape == (3, 40, FIXED_FRAMES)


def test_extract_mfcc_stack():
    y = np.random.randn(SAMPLE_RATE * 3).astype(np.float32)
    feat = extract_mfcc_stack(y, sr=SAMPLE_RATE)
    assert feat.shape == (3, N_MFCC, FIXED_FRAMES)
    assert feat.dtype == np.float32
    # Verify per-channel normalization: mean ~ 0, std ~ 1
    for c in range(3):
        assert abs(feat[c].mean()) < 0.1
        assert abs(feat[c].std() - 1.0) < 0.1


def test_extract_logmel_stack():
    y = np.random.randn(SAMPLE_RATE * 3).astype(np.float32)
    feat = extract_logmel_stack(y, sr=SAMPLE_RATE)
    assert feat.shape == (3, N_MFCC, FIXED_FRAMES)
    assert feat.dtype == np.float32


def test_extract_features_dispatch():
    y = np.random.randn(SAMPLE_RATE * 2).astype(np.float32)
    mfcc_feat = extract_features(y, feature_type="mfcc")
    assert mfcc_feat.shape == (3, N_MFCC, FIXED_FRAMES)

    logmel_feat = extract_features(y, feature_type="logmel")
    assert logmel_feat.shape == (3, N_MFCC, FIXED_FRAMES)

    with pytest.raises(ValueError):
        extract_features(y, feature_type="unknown_type")


def test_extract_features_from_path(dummy_audio_path):
    feat = extract_features_from_path(dummy_audio_path, feature_type="mfcc", augment=False)
    assert feat.shape == (3, N_MFCC, FIXED_FRAMES)
    assert feat.dtype == np.float32


def test_extract_sliding_windows(dummy_long_audio_path):
    import soundfile as sf
    y, sr = sf.read(dummy_long_audio_path)
    windows = extract_sliding_windows(y, feature_type="mfcc", window_sec=3.0, hop_sec=1.5, sr=sr)
    assert windows.ndim == 4
    assert windows.shape[0] > 1  # 7 seconds audio has at least 3 windows with 1.5s hop
    assert windows.shape[1:] == (3, N_MFCC, FIXED_FRAMES)


def test_augmentations():
    y = np.random.randn(SAMPLE_RATE * 2).astype(np.float32)
    y_noise = add_noise_at_snr(y, snr_db=15.0)
    assert y_noise.shape == y.shape
    assert not np.allclose(y_noise, y)

    y_gain = change_gain(y, min_gain_db=2.0, max_gain_db=4.0)
    assert y_gain.shape == y.shape

    y_shift = shift_time(y, max_shift_sec=0.1, sr=SAMPLE_RATE)
    assert y_shift.shape == y.shape

    y_comp = simulate_codec_compression(y, sr=SAMPLE_RATE, codec="telephone")
    assert y_comp.shape == y.shape

    y_all = apply_augmentations(y, sr=SAMPLE_RATE)
    assert len(y_all) > 0
