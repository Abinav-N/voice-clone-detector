import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Universal import hardening
_SRC_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SRC_DIR.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

try:
    from dataset import VoiceSpoofDataset
    from model import VoiceCloneCNN
    from metrics import compute_eer
except ImportError:
    from src.dataset import VoiceSpoofDataset
    from src.model import VoiceCloneCNN
    from src.metrics import compute_eer


def run_evaluation(
    checkpoint_path: str,
    manifest_path: str,
    batch_size: int = 32,
    max_samples: int = None,
    cache: bool = False,
    cache_dir: str = ".cache/features",
    device: torch.device = None,
    results_dir: str = "results",
    name: str = None,
    save_results: bool = True,
):
    """
    Run evaluation on a CSV manifest using a trained VoiceCloneCNN checkpoint.
    Computes overall metrics, per-attack-type EER error analysis, and optionally
    saves raw scores and summary metrics to results/<name>.json and .csv.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resolved_cache = cache_dir if cache else None
    ds = VoiceSpoofDataset(manifest_path, max_samples=max_samples, cache_dir=resolved_cache)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4 if os.name != "nt" else 0,
        pin_memory=(device.type == "cuda"),
    )

    model = VoiceCloneCNN().to(device)
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    all_scores, all_labels = [], []
    with torch.no_grad():
        for feats, labels in loader:
            feats = feats.to(device)
            logits = model(feats)
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_scores.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
    preds = (scores > 0.5).astype(int)

    eer, eer_threshold = compute_eer(labels, scores)
    try:
        auc = float(roc_auc_score(labels, scores))
        auc_str = f"{auc:.4f}"
    except ValueError:
        auc = None
        auc_str = "N/A (only one class present)"

    accuracy = float((preds == labels).mean())
    cm = confusion_matrix(labels, preds).tolist()
    clf_report = classification_report(
        labels, preds, target_names=["bonafide", "spoof"], output_dict=True, zero_division=0
    )

    # Detailed results dataframe
    df_results = ds.df.copy()
    df_results["p_spoof"] = scores
    df_results["p_real"] = 1.0 - scores
    df_results["prediction"] = preds
    df_results["predicted_label"] = np.where(preds == 1, "spoof", "bonafide")
    df_results["correct"] = (preds == labels)

    # Task 7: Per-attack-type evaluation (EER computed against bonafide baseline)
    per_attack_eer = {}
    per_attack_stats = {}

    has_attack_id = "attack_id" in df_results.columns
    bonafide_mask = (df_results["label_idx"] == 0)
    bonafide_scores = scores[bonafide_mask]

    if has_attack_id:
        spoof_mask = (df_results["label_idx"] == 1)
        spoof_attacks = df_results[spoof_mask]

        if len(spoof_attacks) > 0 and spoof_attacks["attack_id"].nunique() >= 1:
            for attack, group in spoof_attacks.groupby("attack_id"):
                total_att = len(group)
                detected = int((group["prediction"] == 1).sum())
                rate = float((detected / total_att) * 100) if total_att > 0 else 0.0
                mean_p = float(group["p_spoof"].mean())

                att_spoof_scores = group["p_spoof"].values

                # Compute EER against bonafide speech
                if len(bonafide_scores) > 0 and len(att_spoof_scores) > 0:
                    sub_scores = np.concatenate([bonafide_scores, att_spoof_scores])
                    sub_labels = np.concatenate([
                        np.zeros(len(bonafide_scores), dtype=int),
                        np.ones(len(att_spoof_scores), dtype=int),
                    ])
                    att_eer, att_thresh = compute_eer(sub_labels, sub_scores)
                    att_eer_val = float(att_eer)
                else:
                    att_eer_val = None

                per_attack_eer[str(attack)] = att_eer_val
                per_attack_stats[str(attack)] = {
                    "count": total_att,
                    "detected_as_fake": detected,
                    "detection_rate_pct": rate,
                    "mean_p_spoof": mean_p,
                    "eer": att_eer_val,
                    "eer_pct": (att_eer_val * 100) if att_eer_val is not None else None,
                }

    metrics = {
        "manifest": os.path.abspath(manifest_path),
        "checkpoint": os.path.abspath(checkpoint_path),
        "timestamp": datetime.now().isoformat(),
        "num_samples": int(len(labels)),
        "num_bonafide": int((labels == 0).sum()),
        "num_spoof": int((labels == 1).sum()),
        "eer": float(eer),
        "eer_pct": float(eer * 100),
        "eer_threshold": float(eer_threshold),
        "auc": auc,
        "accuracy": accuracy,
        "accuracy_pct": float(accuracy * 100),
        "confusion_matrix": cm,
        "classification_report": clf_report,
        "per_attack_eer": per_attack_eer,
        "per_attack_stats": per_attack_stats,
    }

    # Task 8: Results saving to results/<name>.json and .csv
    saved_paths = {}
    if save_results and results_dir:
        os.makedirs(results_dir, exist_ok=True)
        if not name:
            stem = Path(manifest_path).stem
            name = f"eval_{stem}"

        json_path = os.path.join(results_dir, f"{name}.json")
        csv_path = os.path.join(results_dir, f"{name}.csv")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        df_results.to_csv(csv_path, index=False)
        saved_paths = {"json": json_path, "csv": csv_path}

    return metrics, df_results, saved_paths


def main():
    ap = argparse.ArgumentParser(description="Evaluate VoiceCloneCNN detector on a CSV manifest.")
    ap.add_argument("--checkpoint", required=True, help="path to model checkpoint (.pt)")
    ap.add_argument("--manifest", required=True, help="CSV manifest to evaluate on")
    ap.add_argument("--batch-size", type=int, default=32)
    # Task 3: max-samples for smoke tests
    ap.add_argument("--max-samples", type=int, default=None, help="max samples to evaluate on (for smoke tests)")
    # Task 4: Feature caching
    ap.add_argument("--cache", action="store_true", help="enable reading/writing cached .npy features")
    ap.add_argument("--cache-dir", default=".cache/features", help="directory for cached .npy features")
    # Task 8: Results saving
    ap.add_argument("--results-dir", default="results", help="directory to save evaluation metrics and raw scores")
    ap.add_argument("--name", default=None, help="base filename for results saving (results/<name>.json, .csv)")
    ap.add_argument("--no-save", action="store_true", help="skip saving results to disk")
    args = ap.parse_args()

    metrics, df_results, saved_paths = run_evaluation(
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        cache=args.cache,
        cache_dir=args.cache_dir,
        results_dir=None if args.no_save else args.results_dir,
        name=args.name,
        save_results=not args.no_save,
    )

    print(f"\n========================================================================")
    print(f"EVALUATION SUMMARY: {args.manifest}")
    print(f"========================================================================")
    print(f"Total Samples: {metrics['num_samples']} (Bonafide: {metrics['num_bonafide']}, Spoof: {metrics['num_spoof']})")
    print(f"Equal Error Rate (EER): {metrics['eer_pct']:.2f}% (operating threshold={metrics['eer_threshold']:.3f})")
    auc_str = f"{metrics['auc']:.4f}" if metrics['auc'] is not None else "N/A"
    print(f"ROC AUC:                {auc_str}")
    print(f"Accuracy:               {metrics['accuracy_pct']:.2f}%")
    print()
    print("Confusion Matrix (rows=true, cols=pred; 0=bonafide, 1=spoof):")
    print(np.array(metrics["confusion_matrix"]))
    print()
    print("Classification Report:")
    print(classification_report(
        df_results["label_idx"], df_results["prediction"], target_names=["bonafide", "spoof"], zero_division=0
    ))

    # Task 7: Print per-attack-type error analysis table
    if metrics["per_attack_stats"]:
        print("\n========================================================================")
        print("PER-ATTACK-TYPE ERROR ANALYSIS (EER against Bonafide Speech)")
        print("========================================================================")
        print(f"{'Attack ID':<12} {'Count':<8} {'Detected Fake':<16} {'Detect Rate (%)':<18} {'EER (%)':<10}")
        print("-" * 68)
        summary_items = []
        for att, stats in sorted(metrics["per_attack_stats"].items()):
            eer_display = f"{stats['eer_pct']:.2f}%" if stats['eer_pct'] is not None else "N/A"
            print(f"{att:<12} {stats['count']:<8} {stats['detected_as_fake']:<16} {stats['detection_rate_pct']:<18.1f} {eer_display:<10}")
            if stats['eer_pct'] is not None:
                summary_items.append(f"{att}: {stats['eer_pct']:.1f}%")
            else:
                summary_items.append(f"{att}: N/A")

        print("-" * 68)
        if summary_items:
            print("Attack EER Summary: " + ", ".join(summary_items))

    # Task 8: Report saved files
    if saved_paths:
        print("\n========================================================================")
        print("SAVED EVALUATION ARTIFACTS")
        print("========================================================================")
        print(f"  Metrics JSON: {saved_paths['json']}")
        print(f"  Raw Scores:   {saved_paths['csv']}")


if __name__ == "__main__":
    main()
