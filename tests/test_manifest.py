"""
Unit tests for manifest builders and dataset loader (Task 21 & 22).
"""
import csv
from pathlib import Path
import pytest
import torch

from build_manifest import write_manifest, parse_asvspoof_protocol, build_from_folder, build_from_itw
from dataset import VoiceSpoofDataset


def test_write_manifest(tmp_path):
    out_csv = tmp_path / "manifest.csv"
    rows = [
        ("path/to/a.wav", "bonafide", "-"),
        ("path/to/b.wav", "spoof", "A07"),
    ]
    write_manifest(rows, str(out_csv))
    assert out_csv.is_file()

    with open(out_csv, "r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
    assert reader[0] == ["filepath", "label", "attack_id"]
    assert reader[1] == ["path/to/a.wav", "bonafide", "-"]
    assert reader[2] == ["path/to/b.wav", "spoof", "A07"]


def test_parse_asvspoof_protocol(tmp_path):
    proto_file = tmp_path / "protocol.txt"
    proto_file.write_text(
        "LA_0001 LA_D_10001 - - bonafide\n"
        "LA_0002 LA_D_10002 - A01 spoof\n"
        "LA_0003 LA_D_10003 - A07 spoof\n",
        encoding="utf-8"
    )
    audio_dir = tmp_path / "flac"
    # audio_dir doesn't exist yet -> dry run returns all rows
    rows = parse_asvspoof_protocol(str(proto_file), str(audio_dir))
    assert len(rows) == 3
    assert rows[0][1] == "bonafide"
    assert rows[0][2] == "-"
    assert rows[1][1] == "spoof"
    assert rows[1][2] == "A01"
    assert rows[2][2] == "A07"


def test_build_from_folder(tmp_path):
    root = tmp_path / "dataset"
    (root / "real").mkdir(parents=True)
    (root / "fake").mkdir(parents=True)

    (root / "real" / "clip1.wav").write_bytes(b"dummy1")
    (root / "real" / "clip2.wav").write_bytes(b"dummy2")
    (root / "fake" / "clip3.wav").write_bytes(b"dummy3")

    rows = build_from_folder(str(root))
    assert len(rows) == 3
    labels = [r[1] for r in rows]
    assert labels.count("bonafide") == 2
    assert labels.count("spoof") == 1


def test_build_from_itw(tmp_path):
    root = tmp_path / "itw"
    root.mkdir()
    meta_csv = root / "meta.csv"
    meta_csv.write_text(
        "file,label,speaker\n"
        "clipA.wav,bona-fide,spk1\n"
        "clipB.wav,spoof,spk2\n",
        encoding="utf-8"
    )
    # create matching audio files
    (root / "clipA.wav").write_bytes(b"dummyA")
    (root / "clipB.wav").write_bytes(b"dummyB")

    rows = build_from_itw(str(root), meta_path=str(meta_csv))
    assert len(rows) == 2
    assert rows[0][1] == "bonafide"
    assert rows[1][1] == "spoof"


def test_dataset_loader_and_corrupt_file_handling(tmp_path, dummy_audio_path, dummy_corrupt_audio_path):
    """Verify VoiceSpoofDataset loads good audio and handles corrupt files gracefully."""
    manifest_csv = tmp_path / "test_manifest.csv"
    rows = [
        (dummy_audio_path, "bonafide", "-"),
        (dummy_corrupt_audio_path, "spoof", "A01"),
    ]
    write_manifest(rows, str(manifest_csv))

    ds = VoiceSpoofDataset(str(manifest_csv), feature_type="mfcc")
    assert len(ds) == 2

    # Sample 0: good audio
    feat0, label0 = ds[0]
    assert isinstance(feat0, torch.Tensor)
    assert feat0.shape == (3, 40, 300)
    assert label0.item() == 0

    # Sample 1: corrupt audio - must not raise exception, must return zero tensor
    feat1, label1 = ds[1]
    assert isinstance(feat1, torch.Tensor)
    assert feat1.shape == (3, 40, 300)
    assert torch.all(feat1 == 0.0)
    assert label1.item() == 1


def test_dataset_max_samples(tmp_path, dummy_audio_path):
    manifest_csv = tmp_path / "large_manifest.csv"
    rows = [(dummy_audio_path, "bonafide", "-")] * 10
    write_manifest(rows, str(manifest_csv))

    ds = VoiceSpoofDataset(str(manifest_csv), max_samples=3)
    assert len(ds) == 3
