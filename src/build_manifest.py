"""
Build a generic (filepath,label) CSV manifest from various dataset layouts,
then split into train/val/test.

Supported --dataset values:

  folder     Generic layout:  root/real/*.wav  and  root/fake/*.wav
             (works for the "Fake-or-Real" Kaggle dataset if you rename
             its subfolders to real/fake, and for WaveFake if you point
             it at a real/ folder plus each vocoder's fake folder)

  asvspoof   ASVspoof2019 LA layout, using the official protocol file:
             root/
               ASVspoof2019_LA_train/flac/*.flac
               ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt
             Protocol line format:
               SPEAKER_ID  FILENAME  -  ATTACK_ID  bonafide|spoof

Usage:
  python build_manifest.py --dataset folder --root /path/to/for_dataset --out manifests/
  python build_manifest.py --dataset asvspoof --root /path/to/asvspoof2019 \
      --protocol /path/to/protocol.txt --audio-dir /path/to/flac --out manifests/
"""
import argparse
import os
import random
import csv
from pathlib import Path


def build_from_folder(root: str):
    """root/real/*, root/fake/* -> list of (filepath, label)"""
    rows = []
    for label, subdir in [("bonafide", "real"), ("spoof", "fake")]:
        d = Path(root) / subdir
        if not d.exists():
            raise FileNotFoundError(f"Expected folder {d} — rename your dataset's "
                                     f"subfolders to 'real' and 'fake', or edit this script.")
        for ext in ("*.wav", "*.flac", "*.mp3"):
            for f in d.rglob(ext):
                rows.append((str(f), label))
    return rows


def build_from_asvspoof(protocol_path: str, audio_dir: str):
    """Parse an ASVspoof2019 LA protocol file into (filepath, label) rows."""
    rows = []
    audio_dir = Path(audio_dir)
    with open(protocol_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            filename, label = parts[1], parts[-1]  # e.g. "LA_T_1234567", "bonafide"/"spoof"
            fpath = audio_dir / f"{filename}.flac"
            if fpath.exists():
                rows.append((str(fpath), label))
    return rows


def split_and_write(rows, out_dir: str, val_frac=0.15, test_frac=0.15, seed=42):
    random.seed(seed)
    random.shuffle(rows)
    n = len(rows)
    n_val = int(n * val_frac)
    n_test = int(n * test_frac)
    val_rows = rows[:n_val]
    test_rows = rows[n_val:n_val + n_test]
    train_rows = rows[n_val + n_test:]

    os.makedirs(out_dir, exist_ok=True)
    for name, split_rows in [("train", train_rows), ("val", val_rows), ("test", test_rows)]:
        path = os.path.join(out_dir, f"{name}.csv")
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filepath", "label"])
            writer.writerows(split_rows)
        n_bonafide = sum(1 for _, l in split_rows if l == "bonafide")
        n_spoof = sum(1 for _, l in split_rows if l == "spoof")
        print(f"{name}: {len(split_rows)} rows ({n_bonafide} bonafide, {n_spoof} spoof) -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["folder", "asvspoof"], required=True)
    ap.add_argument("--root", required=True, help="dataset root (used by 'folder' mode)")
    ap.add_argument("--protocol", help="protocol file path (asvspoof mode)")
    ap.add_argument("--audio-dir", help="audio directory (asvspoof mode)")
    ap.add_argument("--out", default="manifests/")
    ap.add_argument("--no-split", action="store_true",
                     help="Write all rows to a single CSV instead of splitting into "
                          "train/val/test. Use this when your dataset already comes "
                          "pre-split (e.g. Fake-or-Real has training/validation/testing "
                          "folders already) — run this script once per folder.")
    ap.add_argument("--split-name", default="manifest",
                     help="Output filename (without .csv) when --no-split is used, "
                          "e.g. 'train', 'val', or 'test'")
    args = ap.parse_args()

    if args.dataset == "folder":
        rows = build_from_folder(args.root)
    else:
        if not args.protocol or not args.audio_dir:
            raise ValueError("asvspoof mode requires --protocol and --audio-dir")
        rows = build_from_asvspoof(args.protocol, args.audio_dir)

    print(f"Found {len(rows)} total audio files")

    if args.no_split:
        os.makedirs(args.out, exist_ok=True)
        path = os.path.join(args.out, f"{args.split_name}.csv")
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filepath", "label"])
            writer.writerows(rows)
        n_bonafide = sum(1 for _, l in rows if l == "bonafide")
        n_spoof = sum(1 for _, l in rows if l == "spoof")
        print(f"{args.split_name}: {len(rows)} rows ({n_bonafide} bonafide, {n_spoof} spoof) -> {path}")
    else:
        split_and_write(rows, args.out)


if __name__ == "__main__":
    main()
