"""
Cross-Dataset Generalization Evaluation (cross_eval.py)
Evaluates a trained model checkpoint on both ASVspoof (benchmark/in-domain) and
In-the-Wild (unseen real-world deepfakes), printing a side-by-side comparison table
that quantifies the generalization gap (drop in AUC/Accuracy, increase in EER).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch
import pandas as pd

# Add src to sys.path
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluate import run_evaluation


def cross_dataset_evaluate(
    checkpoint: str,
    asvspoof_manifest: str,
    itw_manifest: str,
    batch_size: int = 32,
    max_samples: int = None,
    cache: bool = False,
    cache_dir: str = ".cache/features",
    out_json: str = None,
):
    """
    Evaluate checkpoint across both datasets and return comparative metrics and table.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Evaluating checkpoint: {checkpoint}")

    print(f"\n[1/2] Evaluating on ASVspoof benchmark ({asvspoof_manifest})...")
    asv_metrics, _, _ = run_evaluation(
        checkpoint_path=checkpoint,
        manifest_path=asvspoof_manifest,
        batch_size=batch_size,
        max_samples=max_samples,
        cache=cache,
        cache_dir=cache_dir,
        device=device,
        save_results=False,
    )

    print(f"\n[2/2] Evaluating on In-the-Wild unseen dataset ({itw_manifest})...")
    itw_metrics, _, _ = run_evaluation(
        checkpoint_path=checkpoint,
        manifest_path=itw_manifest,
        batch_size=batch_size,
        max_samples=max_samples,
        cache=cache,
        cache_dir=cache_dir,
        device=device,
        save_results=False,
    )

    # Compute Generalization Gap
    delta_eer = itw_metrics["eer_pct"] - asv_metrics["eer_pct"]
    delta_acc = itw_metrics["accuracy_pct"] - asv_metrics["accuracy_pct"]
    if itw_metrics["auc"] is not None and asv_metrics["auc"] is not None:
        delta_auc = itw_metrics["auc"] - asv_metrics["auc"]
        delta_auc_str = f"{delta_auc:+.4f} ({'drop' if delta_auc < 0 else 'gain'})"
        itw_auc_str = f"{itw_metrics['auc']:.4f}"
        asv_auc_str = f"{asv_metrics['auc']:.4f}"
    else:
        delta_auc = None
        delta_auc_str = "N/A"
        itw_auc_str = "N/A"
        asv_auc_str = "N/A"

    delta_thresh = itw_metrics["eer_threshold"] - asv_metrics["eer_threshold"]

    # Build comparison data
    table_rows = [
        ("Total Samples", f"{asv_metrics['num_samples']:,}", f"{itw_metrics['num_samples']:,}", "-"),
        ("Bonafide / Spoof", f"{asv_metrics['num_bonafide']:,} / {asv_metrics['num_spoof']:,}",
                            f"{itw_metrics['num_bonafide']:,} / {itw_metrics['num_spoof']:,}", "-"),
        ("Accuracy (%)", f"{asv_metrics['accuracy_pct']:.2f}%", f"{itw_metrics['accuracy_pct']:.2f}%",
                         f"{delta_acc:+.2f}% ({'drop' if delta_acc < 0 else 'gain'})"),
        ("Equal Error Rate (EER)", f"{asv_metrics['eer_pct']:.2f}%", f"{itw_metrics['eer_pct']:.2f}%",
                                   f"{delta_eer:+.2f}% ({'increase' if delta_eer > 0 else 'decrease'})"),
        ("ROC AUC", asv_auc_str, itw_auc_str, delta_auc_str),
        ("Operating Threshold", f"{asv_metrics['eer_threshold']:.3f}", f"{itw_metrics['eer_threshold']:.3f}",
                                f"{delta_thresh:+.3f}"),
    ]

    # Print formatted comparative table
    print("\n" + "=" * 94)
    print("                      CROSS-DATASET GENERALIZATION EVALUATION TABLE")
    print("=" * 94)
    print(f"{'Metric':<26} {'ASVspoof (In-Domain)':<24} {'In-the-Wild (Unseen)':<24} {'Generalization Gap':<20}")
    print("-" * 94)
    for metric, asv_val, itw_val, gap in table_rows:
        print(f"{metric:<26} {asv_val:<24} {itw_val:<24} {gap:<20}")
    print("=" * 94)

    print("\nKey Observations:")
    if delta_eer > 0:
        print(f"  * Generalization Gap: Val EER increased by {delta_eer:.2f} percentage points on In-the-Wild.")
    else:
        print(f"  * Generalization Gap: EER difference is {delta_eer:.2f} percentage points.")

    if delta_acc < 0:
        print(f"  * Accuracy Drop: Accuracy decreased by {abs(delta_acc):.2f}% when testing on unseen real-world fakes.")

    results_data = {
        "checkpoint": os.path.abspath(checkpoint),
        "asvspoof_manifest": os.path.abspath(asvspoof_manifest),
        "itw_manifest": os.path.abspath(itw_manifest),
        "asvspoof_metrics": asv_metrics,
        "itw_metrics": itw_metrics,
        "generalization_gap": {
            "delta_eer_pct": float(delta_eer),
            "delta_accuracy_pct": float(delta_acc),
            "delta_auc": float(delta_auc) if delta_auc is not None else None,
            "delta_threshold": float(delta_thresh),
        },
    }

    if out_json:
        os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=2)
        print(f"\nComparative results saved to '{out_json}'")

    return results_data


def main():
    ap = argparse.ArgumentParser(description="Cross-dataset evaluation: ASVspoof vs In-the-Wild generalization gap.")
    ap.add_argument("--checkpoint", default="checkpoints/best.pt", help="path to model checkpoint (.pt)")
    ap.add_argument("--asvspoof", required=True, help="path to ASVspoof eval/test CSV manifest")
    ap.add_argument("--itw", required=True, help="path to In-the-Wild CSV manifest")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-samples", type=int, default=None, help="limit samples per dataset for quick smoke tests")
    ap.add_argument("--cache", action="store_true", help="enable feature caching")
    ap.add_argument("--cache-dir", default=".cache/features", help="cache directory for .npy features")
    ap.add_argument("--out", default=None, help="optional path to save results JSON (e.g. results/cross_eval.json)")
    args = ap.parse_args()

    cross_dataset_evaluate(
        checkpoint=args.checkpoint,
        asvspoof_manifest=args.asvspoof,
        itw_manifest=args.itw,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        cache=args.cache,
        cache_dir=args.cache_dir,
        out_json=args.out,
    )


if __name__ == "__main__":
    main()
