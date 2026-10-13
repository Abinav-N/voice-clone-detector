"""
Build CSV manifests (filepath,label,attack_id) from various dataset layouts.

Supported --dataset values:

  folder     Generic folder layout: root/real/*.wav and root/fake/*.wav
             (works for the Fake-or-Real Kaggle dataset and WaveFake)

  asvspoof   Official ASVspoof 2019 LA benchmark layout using official protocol files
             for train, dev, and eval. Keeps attack ID (A01-A19, or - for bonafide).
             Generates official train.csv, val.csv (dev), and test.csv (eval).

  itw        In-the-Wild dataset (Müller et al., 2022) using meta.csv.
             Produces a test-only manifest (test.csv / itw_test.csv).

Usage:
  python src/build_manifest.py --dataset folder --root /path/to/for_dataset --out manifests/
  python src/build_manifest.py --dataset asvspoof --root /path/to/asvspoof2019 --out manifests/
  python src/build_manifest.py --dataset itw --root /path/to/in_the_wild --out manifests/
"""
import argparse
import os
import random
import csv
from pathlib import Path


def write_manifest(rows, out_path: str):
    """Write list of (filepath, label, attack_id) to CSV."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "label", "attack_id"])
        writer.writerows(rows)
    n_bonafide = sum(1 for _, l, _ in rows if l == "bonafide")
    n_spoof = sum(1 for _, l, _ in rows if l == "spoof")
    print(f"  -> {len(rows)} rows ({n_bonafide} bonafide, {n_spoof} spoof) written to {out_path}")


def build_from_folder(root: str):
    """root/real/*, root/fake/* -> list of (filepath, label, attack_id)"""
    rows = []
    for label, subdir in [("bonafide", "real"), ("spoof", "fake")]:
        d = Path(root) / subdir
        if not d.exists():
            raise FileNotFoundError(
                f"Expected folder {d} — ensure subfolders are named 'real' and 'fake', "
                f"or point to a folder containing real/ and fake/ subdirectories."
            )
        for ext in ("*.wav", "*.flac", "*.mp3"):
            for f in d.rglob(ext):
                rows.append((str(f), label, "-"))
    return rows


def parse_asvspoof_protocol(protocol_path: str, audio_dir: str):
    """
    Parse an ASVspoof 2019 protocol file.
    Protocol line format:
        SPEAKER_ID  FILENAME  ENVIRONMENT_ID  ATTACK_ID  KEY(bonafide|spoof)
    Returns: list of (filepath, label, attack_id)
    """
    rows = []
    audio_dir = Path(audio_dir)
    protocol_path = Path(protocol_path)
    if not protocol_path.is_file():
        raise FileNotFoundError(f"ASVspoof protocol file not found: {protocol_path}")

    with open(protocol_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            filename = parts[1]
            attack_id = parts[3]
            label = parts[4].lower()

            # Find matching audio file
            fpath = audio_dir / f"{filename}.flac"
            if not fpath.exists():
                fpath_wav = audio_dir / f"{filename}.wav"
                if fpath_wav.exists():
                    fpath = fpath_wav

            # Check if file exists or if audio_dir doesn't exist yet (allows mock/dry-run)
            if fpath.exists() or not audio_dir.exists():
                rows.append((str(fpath), label, attack_id))

    return rows


def build_official_asvspoof(root: str, out_dir: str,
                            protocols_dir: str = None,
                            train_protocol: str = None, dev_protocol: str = None, eval_protocol: str = None,
                            train_audio: str = None, dev_audio: str = None, eval_audio: str = None):
    """
    Build official train, dev, and eval manifests from ASVspoof 2019 LA layout.
    Generates train.csv, val.csv (dev), and test.csv (eval) with attack_id column.
    """
    root_path = Path(root) if root else Path(".")

    # Discover protocols directory
    p_dir = Path(protocols_dir) if protocols_dir else None
    if p_dir is None:
        for candidate in [
            root_path / "ASVspoof2019_LA_cm_protocols",
            root_path / "protocols",
            root_path / "cm_protocols",
            root_path,
        ]:
            if candidate.is_dir():
                p_dir = candidate
                break

    # Discover protocol files
    def find_file(explicit, candidates, search_dir):
        if explicit and Path(explicit).is_file():
            return Path(explicit)
        if search_dir and search_dir.is_dir():
            for name in candidates:
                p = search_dir / name
                if p.is_file():
                    return p
            for p in search_dir.rglob("*.txt"):
                for name in candidates:
                    if name.lower() in p.name.lower():
                        return p
        return None

    tr_proto = find_file(train_protocol, ["ASVspoof2019.LA.cm.train.trn.txt", "train.trn.txt"], p_dir)
    dv_proto = find_file(dev_protocol, ["ASVspoof2019.LA.cm.dev.trl.txt", "dev.trl.txt"], p_dir)
    ev_proto = find_file(eval_protocol, ["ASVspoof2019.LA.cm.eval.trl.txt", "eval.trl.txt"], p_dir)

    # Discover audio directories
    def find_audio_dir(explicit, candidates):
        if explicit and Path(explicit).exists():
            return Path(explicit)
        for c in candidates:
            p = root_path / c
            if p.is_dir():
                return p
        return root_path / candidates[0]

    tr_audio = find_audio_dir(train_audio, ["ASVspoof2019_LA_train/flac", "ASVspoof2019_LA_train", "train/flac", "train"])
    dv_audio = find_audio_dir(dev_audio, ["ASVspoof2019_LA_dev/flac", "ASVspoof2019_LA_dev", "dev/flac", "dev"])
    ev_audio = find_audio_dir(eval_audio, ["ASVspoof2019_LA_eval/flac", "ASVspoof2019_LA_eval", "eval/flac", "eval"])

    if not tr_proto or not dv_proto or not ev_proto:
        missing = []
        if not tr_proto: missing.append("train protocol")
        if not dv_proto: missing.append("dev protocol")
        if not ev_proto: missing.append("eval protocol")
        raise FileNotFoundError(
            f"Could not find official ASVspoof protocol files: {missing}. "
            f"Searched in: {p_dir}. Please specify via --protocols-dir or explicit flags."
        )

    print("Building official ASVspoof 2019 LA manifests:")
    print(f"  Train: protocol={tr_proto}, audio={tr_audio}")
    train_rows = parse_asvspoof_protocol(str(tr_proto), str(tr_audio))
    write_manifest(train_rows, os.path.join(out_dir, "train.csv"))

    print(f"  Dev/Val: protocol={dv_proto}, audio={dv_audio}")
    dev_rows = parse_asvspoof_protocol(str(dv_proto), str(dv_audio))
    write_manifest(dev_rows, os.path.join(out_dir, "val.csv"))
    # Also write dev.csv alias
    write_manifest(dev_rows, os.path.join(out_dir, "dev.csv"))

    print(f"  Eval/Test: protocol={ev_proto}, audio={ev_audio}")
    eval_rows = parse_asvspoof_protocol(str(ev_proto), str(ev_audio))
    write_manifest(eval_rows, os.path.join(out_dir, "test.csv"))
    # Also write eval.csv alias
    write_manifest(eval_rows, os.path.join(out_dir, "eval.csv"))

    print(f"\nAll three official ASVspoof manifests successfully generated in {out_dir}")


def build_from_itw(root: str, meta_path: str = None, audio_dir: str = None):
    """
    Parse In-the-Wild (ITW) dataset meta.csv -> list of (filepath, label, attack_id).
    Supports common meta.csv column layouts (file/speaker/label).
    """
    root_path = Path(root) if root else Path(".")
    if not meta_path:
        meta_candidates = [root_path / "meta.csv", root_path / "metadata.csv"]
        for c in meta_candidates:
            if c.is_file():
                meta_path = str(c)
                break
        if not meta_path:
            # Search recursively for meta.csv
            found = list(root_path.glob("**/meta.csv"))
            if found:
                meta_path = str(found[0])
            else:
                raise FileNotFoundError(f"Could not find meta.csv in {root_path}. Specify via --meta.")

    meta_file = Path(meta_path)
    if not meta_file.is_file():
        raise FileNotFoundError(f"meta.csv not found at: {meta_file}")

    # Determine audio directory
    if not audio_dir:
        for cand in [root_path / "wav", root_path / "audio", root_path / "clips", root_path, meta_file.parent]:
            if cand.is_dir():
                audio_dir = cand
                break
    audio_dir = Path(audio_dir) if audio_dir else meta_file.parent

    rows = []
    # Auto-detect delimiter
    with open(meta_file, "r", encoding="utf-8") as f:
        first_line = f.readline()
        delimiter = "\t" if "\t" in first_line else (";" if ";" in first_line else ",")

    with open(meta_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        # Normalize column header keys
        headers = {k.strip().lower(): k for k in (reader.fieldnames or [])}

        file_col = headers.get("file") or headers.get("filename") or headers.get("filepath") or headers.get("path")
        label_col = headers.get("label") or headers.get("key") or headers.get("target") or headers.get("class")
        speaker_col = headers.get("speaker") or headers.get("speaker_id") or headers.get("attack_id")

        if not file_col or not label_col:
            raise ValueError(
                f"meta.csv must contain filename and label columns. Found columns: {reader.fieldnames}"
            )

        for r in reader:
            fname = r[file_col].strip()
            raw_label = r[label_col].strip().lower()

            # Map label to standard bonafide / spoof
            if raw_label in ("bona-fide", "bonafide", "real", "human", "0"):
                label = "bonafide"
            elif raw_label in ("spoof", "fake", "cloned", "synthetic", "1"):
                label = "spoof"
            else:
                continue

            attack_id = r[speaker_col].strip() if (speaker_col and speaker_col in r and r[speaker_col]) else "-"

            # Resolve file path
            fpath = audio_dir / fname
            if not fpath.exists():
                fpath_wav = audio_dir / f"{fname}.wav"
                if fpath_wav.exists():
                    fpath = fpath_wav

            if fpath.exists() or not audio_dir.exists():
                rows.append((str(fpath), label, attack_id))

    return rows


def split_and_write(rows, out_dir: str, val_frac=0.15, test_frac=0.15, seed=42):
    """Random split into train/val/test manifests with attack_id column."""
    random.seed(seed)
    random.shuffle(rows)
    n = len(rows)
    n_val = int(n * val_frac)
    n_test = int(n * test_frac)
    val_rows = rows[:n_val]
    test_rows = rows[n_val:n_val + n_test]
    train_rows = rows[n_val + n_test:]

    for name, split_rows in [("train", train_rows), ("val", val_rows), ("test", test_rows)]:
        path = os.path.join(out_dir, f"{name}.csv")
        write_manifest(split_rows, path)


def main():
    ap = argparse.ArgumentParser(description="Build CSV manifests for voice clone detection.")
    ap.add_argument("--dataset", choices=["folder", "asvspoof", "itw"], required=True,
                    help="Dataset format: 'folder' (real/fake subfolders), "
                         "'asvspoof' (official 2019 LA protocols), or 'itw' (In-the-Wild meta.csv)")
    ap.add_argument("--root", help="Dataset root directory")
    ap.add_argument("--out", default="manifests/", help="Output directory for CSV manifests")

    # ASVspoof options
    ap.add_argument("--protocols-dir", help="Directory containing official ASVspoof protocol files")
    ap.add_argument("--train-protocol", help="Explicit path to train protocol txt")
    ap.add_argument("--dev-protocol", help="Explicit path to dev protocol txt")
    ap.add_argument("--eval-protocol", help="Explicit path to eval protocol txt")
    ap.add_argument("--train-audio", help="Explicit path to train audio directory")
    ap.add_argument("--dev-audio", help="Explicit path to dev audio directory")
    ap.add_argument("--eval-audio", help="Explicit path to eval audio directory")
    ap.add_argument("--protocol", help="Single protocol file path (legacy single-protocol mode)")
    ap.add_argument("--audio-dir", help="Audio directory (for ITW or single-protocol mode)")

    # ITW options
    ap.add_argument("--meta", help="Path to In-the-Wild meta.csv file")

    # Generic options
    ap.add_argument("--no-split", action="store_true",
                    help="Write rows to a single manifest instead of splitting")
    ap.add_argument("--split-name", default="manifest",
                    help="Output manifest filename (without .csv) when --no-split is used")

    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.dataset == "folder":
        if not args.root:
            raise ValueError("--root is required for folder mode")
        rows = build_from_folder(args.root)
        print(f"Found {len(rows)} audio files in {args.root}")
        if args.no_split:
            path = os.path.join(args.out, f"{args.split_name}.csv")
            write_manifest(rows, path)
        else:
            split_and_write(rows, args.out)

    elif args.dataset == "asvspoof":
        # If user explicitly provides a single protocol and audio-dir, support single-file mode
        if args.protocol and args.audio_dir:
            rows = parse_asvspoof_protocol(args.protocol, args.audio_dir)
            split_name = args.split_name if args.no_split else "asvspoof"
            out_file = os.path.join(args.out, f"{split_name}.csv")
            write_manifest(rows, out_file)
        else:
            # Official ASVspoof 3-way split: train, dev, eval
            build_official_asvspoof(
                root=args.root,
                out_dir=args.out,
                protocols_dir=args.protocols_dir,
                train_protocol=args.train_protocol,
                dev_protocol=args.dev_protocol,
                eval_protocol=args.eval_protocol,
                train_audio=args.train_audio,
                dev_audio=args.dev_audio,
                eval_audio=args.eval_audio,
            )

    elif args.dataset == "itw":
        # In-the-Wild: produces a test-only manifest
        rows = build_from_itw(root=args.root, meta_path=args.meta, audio_dir=args.audio_dir)
        print(f"Parsed {len(rows)} samples from In-the-Wild dataset")
        split_name = args.split_name if args.split_name != "manifest" else "test"
        out_file = os.path.join(args.out, f"{split_name}.csv")
        write_manifest(rows, out_file)
        # Also write itw_test.csv alias
        if split_name != "itw_test":
            write_manifest(rows, os.path.join(args.out, "itw_test.csv"))
        print(f"In-the-Wild test manifest successfully written to {out_file}")


if __name__ == "__main__":
    main()
