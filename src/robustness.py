"""
Task 14: Robustness Evaluation
-------------------------------
Re-evaluates a trained VoiceCloneCNN on degraded audio conditions to simulate
real-world phone-scam scenarios:

  - Telephone-band (8 kHz): downsample to 8 kHz then upsample back to 16 kHz,
    losing frequencies above 4 kHz.
  - Additive white Gaussian noise at SNRs of 20, 10, 5 dB.

Produces an EER-vs-condition table suitable for the project report.

Usage:
    python src/robustness.py \\
        --checkpoint checkpoints/best.pt \\
        --manifest   manifests/eval.csv  \\
        --out        results/robustness.json
"""
import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model import VoiceCloneCNN
from features import (
    load_audio, extract_features, SAMPLE_RATE, N_MFCC,
)
from metrics import compute_eer


# --------------------------------------------------------------------------- #
#  Audio degradation helpers                                                   #
# --------------------------------------------------------------------------- #

def degrade_telephone(y: np.ndarray, sr: int = SAMPLE_RATE, tel_sr: int = 8000) -> np.ndarray:
    """Simulate telephone channel: downsample to tel_sr, then resample back."""
    import librosa
    y_8k = librosa.resample(y, orig_sr=sr, target_sr=tel_sr)
    y_back = librosa.resample(y_8k, orig_sr=tel_sr, target_sr=sr)
    # Ensure same length as input after resampling round-trip
    if len(y_back) > len(y):
        y_back = y_back[:len(y)]
    elif len(y_back) < len(y):
        y_back = np.pad(y_back, (0, len(y) - len(y_back)))
    return y_back.astype(np.float32)


def add_noise(y: np.ndarray, snr_db: float) -> np.ndarray:
    """Add white Gaussian noise at the given SNR (dB) to a waveform."""
    signal_power = np.mean(y ** 2)
    if signal_power < 1e-12:
        return y  # silence — no noise addition possible
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = np.random.randn(len(y)).astype(np.float32) * np.sqrt(noise_power)
    return (y + noise).astype(np.float32)


# --------------------------------------------------------------------------- #
#  Dataset                                                                     #
# --------------------------------------------------------------------------- #

class DegradedDataset(Dataset):
    """Load clips from manifest, apply degradation, and extract features."""

    def __init__(
        self,
        manifest_path: str,
        condition: str,
        feature_type: str = "mfcc",
        max_samples: int = None,
        snr_db: float = None,
    ):
        """
        Args:
            condition: one of 'clean', 'telephone', 'noise_XX' (XX = SNR in dB).
            snr_db:    SNR level; required when condition == 'noise'.
        """
        df = pd.read_csv(manifest_path)
        if max_samples is not None:
            df = df.sample(n=min(max_samples, len(df)), random_state=42)
        self.df = df.reset_index(drop=True)
        self.condition = condition
        self.feature_type = feature_type
        self.snr_db = snr_db

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        y = load_audio(row["filepath"])

        # Apply condition-specific degradation
        if self.condition == "telephone":
            y = degrade_telephone(y)
        elif self.condition == "noise" and self.snr_db is not None:
            y = add_noise(y, self.snr_db)

        feat = extract_features(y, feature_type=self.feature_type)  # (3, 40, 300)
        label = int(row["label_idx"])
        return torch.from_numpy(feat), label


# --------------------------------------------------------------------------- #
#  Single-condition evaluation                                                 #
# --------------------------------------------------------------------------- #

def evaluate_condition(
    model: VoiceCloneCNN,
    manifest_path: str,
    condition: str,
    feature_type: str = "mfcc",
    batch_size: int = 32,
    max_samples: int = None,
    device: torch.device = None,
    snr_db: float = None,
    num_workers: int = 0,
) -> dict:
    """Evaluate model on one degraded-audio condition, return metrics dict."""
    ds = DegradedDataset(
        manifest_path, condition=condition,
        feature_type=feature_type, max_samples=max_samples, snr_db=snr_db,
    )
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )

    all_scores, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for feats, labels in tqdm(loader, desc=f"  [{condition}]", leave=False):
            feats = feats.to(device)
            logits = model(feats)
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_scores.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
    preds = (scores > 0.5).astype(int)

    eer, eer_thresh = compute_eer(labels, scores)
    acc = float((preds == labels).mean())

    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(labels, scores))
    except (ImportError, ValueError):
        auc = None

    return {
        "condition": condition,
        "snr_db": snr_db,
        "n_samples": int(len(labels)),
        "eer": float(eer),
        "eer_pct": float(eer * 100),
        "eer_threshold": float(eer_thresh),
        "auc": auc,
        "accuracy": acc,
        "accuracy_pct": float(acc * 100),
    }


# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(
        description="Robustness evaluation: phone-channel and additive noise conditions."
    )
    ap.add_argument("--checkpoint", required=True, help="Path to trained checkpoint (.pt)")
    ap.add_argument("--manifest", required=True, help="Eval manifest CSV")
    ap.add_argument(
        "--snrs", nargs="+", type=float, default=[20.0, 10.0, 5.0],
        help="SNR levels (dB) to test. Default: 20 10 5"
    )
    ap.add_argument("--feature", choices=["mfcc", "logmel"], default="mfcc")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--out", default=None, help="Output JSON path for results")
    ap.add_argument("--results-dir", default="results", help="Directory for saving results")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(42)
    num_workers = 4 if os.name != "nt" else 0

    # Load model
    print(f"\n[Robustness] Loading checkpoint: {args.checkpoint}")
    model = VoiceCloneCNN().to(device)
    try:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    except Exception:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    print(f"  Feature type in checkpoint: {ckpt.get('feature_type', args.feature)}")

    # Build list of (label, condition, snr_db) tuples
    conditions = [
        ("clean",      "clean",      None),
        ("telephone",  "telephone",  None),
    ]
    for snr in sorted(args.snrs, reverse=True):
        conditions.append((f"noise_{int(snr)}dB", "noise", snr))

    all_results = []
    print(f"\n  Running {len(conditions)} conditions on: {args.manifest}")
    for label, cond, snr in conditions:
        print(f"\n  Condition: {label}")
        metrics = evaluate_condition(
            model=model,
            manifest_path=args.manifest,
            condition=cond,
            feature_type=ckpt.get("feature_type", args.feature),
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            device=device,
            snr_db=snr,
            num_workers=num_workers,
        )
        metrics["label"] = label
        all_results.append(metrics)

    # ----------------------------------------------------------------------- #
    #  Print EER-vs-condition table                                            #
    # ----------------------------------------------------------------------- #
    print(f"\n{'='*65}")
    print("  ROBUSTNESS EVALUATION — EER vs. Condition")
    print(f"{'='*65}")
    print(f"  {'Condition':<20} {'N':<8} {'EER (%)':<12} {'AUC':<8} {'Accuracy (%)'}")
    print(f"  {'-'*60}")
    for r in all_results:
        auc_s = f"{r['auc']:.4f}" if r["auc"] is not None else "N/A  "
        print(
            f"  {r['label']:<20} {r['n_samples']:<8} "
            f"{r['eer_pct']:<12.2f} {auc_s:<8} {r['accuracy_pct']:.2f}"
        )
    print(f"  {'='*60}")

    # ----------------------------------------------------------------------- #
    #  Save results                                                            #
    # ----------------------------------------------------------------------- #
    out_path = args.out
    if not out_path:
        os.makedirs(args.results_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(args.results_dir, f"robustness_{ts}.json")

    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    payload = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "manifest": os.path.abspath(args.manifest),
        "timestamp": datetime.now().isoformat(),
        "feature_type": ckpt.get("feature_type", args.feature),
        "conditions": all_results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
