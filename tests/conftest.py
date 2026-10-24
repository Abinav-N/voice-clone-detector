"""
Shared pytest fixtures and path configuration.
"""
import sys
import tempfile
import os
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf

# Ensure repo root and src are on sys.path
TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
SRC_DIR = REPO_ROOT / "src"

for p in (REPO_ROOT, SRC_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


@pytest.fixture
def dummy_audio_path(tmp_path):
    """Generate a clean 3-second 16kHz sine wave WAV file."""
    sr = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

    wav_path = tmp_path / "dummy_sine.wav"
    sf.write(str(wav_path), y, sr)
    return str(wav_path)


@pytest.fixture
def dummy_long_audio_path(tmp_path):
    """Generate a 7-second 16kHz sine wave WAV file for sliding window tests."""
    sr = 16000
    duration = 7.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = (0.3 * np.sin(2 * np.pi * 300.0 * t)).astype(np.float32)

    wav_path = tmp_path / "dummy_long.wav"
    sf.write(str(wav_path), y, sr)
    return str(wav_path)


@pytest.fixture
def dummy_short_audio_path(tmp_path):
    """Generate a short 0.5-second 16kHz WAV file."""
    sr = 16000
    duration = 0.5
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = (0.4 * np.sin(2 * np.pi * 500.0 * t)).astype(np.float32)

    wav_path = tmp_path / "dummy_short.wav"
    sf.write(str(wav_path), y, sr)
    return str(wav_path)


@pytest.fixture
def dummy_corrupt_audio_path(tmp_path):
    """Generate an unreadable corrupt audio file."""
    bad_path = tmp_path / "corrupt.wav"
    bad_path.write_bytes(b"NOT_A_VALID_RIFF_HEADER_CORRUPT_BYTES")
    return str(bad_path)
