"""
Plot Generator (plots.py)
Generates publication-quality evaluation and training figures:
1. ROC Curve (Receiver Operating Characteristic with AUC and EER)
2. DET Curve (Detection Error Tradeoff curve using probit scale)
3. Confusion Matrix (annotated 2x2 matrix with counts and percentages)
4. Score Histogram (Real vs Fake P(spoof) score distribution)
5. Training Curves (Loss and Val EER / Accuracy across epochs from history.csv)

All 5 figures are saved as high-resolution PNGs in the specified output directory.
"""
import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

# Ensure src/ is on sys.path
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from metrics import compute_eer
from evaluate import run_evaluation

# Dark modern styling parameters
STYLE_BG = "#0f172a"
STYLE_PANEL = "#1e293b"
STYLE_TEXT = "#f8fafc"
STYLE_SUBTEXT = "#94a3b8"
STYLE_GRID = "#334155"
STYLE_ACCENT_BLUE = "#38bdf8"
STYLE_ACCENT_RED = "#f43f5e"
STYLE_ACCENT_GREEN = "#10b981"
STYLE_ACCENT_PURPLE = "#a855f7"


def setup_dark_ax(ax, title: str, xlabel: str, ylabel: str):
    """Apply polished dark theme styling to an axis."""
    ax.set_facecolor(STYLE_PANEL)
    ax.set_title(title, color=STYLE_TEXT, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, color=STYLE_SUBTEXT, fontsize=11, labelpad=8)
    ax.set_ylabel(ylabel, color=STYLE_SUBTEXT, fontsize=11, labelpad=8)
    ax.tick_params(colors=STYLE_SUBTEXT, labelsize=10)
    for spine in ax.spines.values():
        spine.set_color(STYLE_GRID)
    ax.grid(True, linestyle="--", alpha=0.35, color=STYLE_GRID)


def plot_roc_curve(labels: np.ndarray, scores: np.ndarray, out_path: str):
    """Generate and save ROC curve."""
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    eer, eer_thresh = compute_eer(labels, scores)
    try:
        auc = roc_auc_score(labels, scores)
        auc_str = f"AUC = {auc:.4f}"
    except ValueError:
        auc_str = "AUC = N/A"

    fig, ax = plt.subplots(figsize=(7, 6), facecolor=STYLE_BG)
    setup_dark_ax(ax, "Receiver Operating Characteristic (ROC) Curve",
                  "False Positive Rate (FPR)", "True Positive Rate (TPR)")

    ax.plot(fpr, tpr, color=STYLE_ACCENT_BLUE, linewidth=2.5,
            label=f"CNN Model ({auc_str})")
    ax.fill_between(fpr, tpr, color=STYLE_ACCENT_BLUE, alpha=0.15)

    # Random chance baseline
    ax.plot([0, 1], [0, 1], linestyle="--", color=STYLE_SUBTEXT, alpha=0.7, label="Random Guess (AUC = 0.50)")

    # Mark EER operating point (FPR = EER, TPR = 1 - EER)
    ax.scatter([eer], [1 - eer], color=STYLE_ACCENT_RED, s=90, zorder=5,
               label=f"EER Operating Point ({eer * 100:.2f}%)")

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.legend(facecolor=STYLE_BG, edgecolor=STYLE_GRID, labelcolor=STYLE_TEXT, loc="lower right", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  [1/5] ROC Curve saved -> {out_path}")


def plot_det_curve(labels: np.ndarray, scores: np.ndarray, out_path: str):
    """
    Generate and save Detection Error Tradeoff (DET) curve using standard normal deviate (probit) scale.
    """
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    eer, _ = compute_eer(labels, scores)

    # Filter boundary zeroes and ones for probit transform
    eps = 1e-4
    valid = (fpr > eps) & (fpr < 1 - eps) & (fnr > eps) & (fnr < 1 - eps)

    fig, ax = plt.subplots(figsize=(7, 6), facecolor=STYLE_BG)
    setup_dark_ax(ax, "Detection Error Tradeoff (DET) Curve",
                  "False Positive Rate (%) [Normal Deviate Scale]",
                  "False Negative Rate (%) [Normal Deviate Scale]")

    # Select standard probability tick values
    ticks_pct = np.array([0.1, 0.5, 1, 2, 5, 10, 20, 40])
    ticks_prob = ticks_pct / 100.0
    ticks_probit = norm.ppf(ticks_prob)

    if np.any(valid):
        sub_fpr = fpr[valid]
        sub_fnr = fnr[valid]
        probit_fpr = norm.ppf(sub_fpr)
        probit_fnr = norm.ppf(sub_fnr)
        ax.plot(probit_fpr, probit_fnr, color=STYLE_ACCENT_PURPLE, linewidth=2.5,
                label=f"DET Curve (EER = {eer * 100:.2f}%)")

    # Diagonal EER line (FNR = FPR)
    lim_min, lim_max = norm.ppf(0.001), norm.ppf(0.50)
    ax.plot([lim_min, lim_max], [lim_min, lim_max], linestyle=":", color=STYLE_SUBTEXT, alpha=0.6,
            label="EER Line (FNR = FPR)")

    # Mark EER point
    eer_probit = norm.ppf(max(min(eer, 1 - eps), eps))
    ax.scatter([eer_probit], [eer_probit], color=STYLE_ACCENT_RED, s=90, zorder=5,
               label=f"EER Point ({eer * 100:.2f}%)")

    ax.set_xticks(ticks_probit)
    ax.set_xticklabels([f"{p:g}%" for p in ticks_pct])
    ax.set_yticks(ticks_probit)
    ax.set_yticklabels([f"{p:g}%" for p in ticks_pct])
    ax.set_xlim([ticks_probit[0] - 0.2, ticks_probit[-1] + 0.2])
    ax.set_ylim([ticks_probit[0] - 0.2, ticks_probit[-1] + 0.2])

    ax.legend(facecolor=STYLE_BG, edgecolor=STYLE_GRID, labelcolor=STYLE_TEXT, loc="upper right", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  [2/5] DET Curve saved -> {out_path}")


def plot_confusion_matrix(labels: np.ndarray, scores: np.ndarray, out_path: str, threshold: float = 0.5):
    """Generate and save annotated Confusion Matrix heatmap."""
    preds = (scores > threshold).astype(int)
    cm = confusion_matrix(labels, preds)

    fig, ax = plt.subplots(figsize=(6.5, 5.5), facecolor=STYLE_BG)
    ax.set_facecolor(STYLE_PANEL)

    # Normalized percentages along each true class row
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors=STYLE_SUBTEXT, labelsize=9)
    cbar.outline.set_edgecolor(STYLE_GRID)
    cbar.set_label("Fraction within True Class", color=STYLE_SUBTEXT, fontsize=10)

    classes = ["Real (Human)", "Fake (Spoof)"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(classes, color=STYLE_TEXT, fontsize=11, fontweight="bold")
    ax.set_yticklabels(classes, color=STYLE_TEXT, fontsize=11, fontweight="bold")

    ax.set_title("Confusion Matrix", color=STYLE_TEXT, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", color=STYLE_SUBTEXT, fontsize=11, labelpad=8)
    ax.set_ylabel("True Ground Truth", color=STYLE_SUBTEXT, fontsize=11, labelpad=8)
    for spine in ax.spines.values():
        spine.set_color(STYLE_GRID)

    # Text annotations in each cell
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            pct = cm_norm[i, j] * 100
            txt_color = "white" if cm_norm[i, j] > 0.5 else "#cbd5e1"
            ax.text(j, i, f"{count:,}\n({pct:.1f}%)",
                    ha="center", va="center", color=txt_color, fontsize=13, fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  [3/5] Confusion Matrix saved -> {out_path}")


def plot_score_histogram(labels: np.ndarray, scores: np.ndarray, out_path: str, threshold: float = 0.5):
    """Generate and save Real vs Fake score histogram."""
    real_scores = scores[labels == 0]
    fake_scores = scores[labels == 1]

    fig, ax = plt.subplots(figsize=(7.5, 5), facecolor=STYLE_BG)
    setup_dark_ax(ax, "Score Distribution: P(spoof) for Real vs Fake Audio",
                  "Predicted P(spoof)", "Sample Count")

    bins = np.linspace(0.0, 1.0, 41)
    if len(real_scores) > 0:
        ax.hist(real_scores, bins=bins, color=STYLE_ACCENT_GREEN, alpha=0.65,
                edgecolor=STYLE_BG, label=f"Real Human ({len(real_scores):,} clips)")
    if len(fake_scores) > 0:
        ax.hist(fake_scores, bins=bins, color=STYLE_ACCENT_RED, alpha=0.65,
                edgecolor=STYLE_BG, label=f"Fake / Cloned ({len(fake_scores):,} clips)")

    # Decision threshold line
    ax.axvline(threshold, color=STYLE_TEXT, linestyle="--", linewidth=2,
               label=f"Decision Threshold ({threshold:.2f})")

    ax.set_xlim([-0.02, 1.02])
    ax.legend(facecolor=STYLE_BG, edgecolor=STYLE_GRID, labelcolor=STYLE_TEXT, loc="upper center", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  [4/5] Score Histogram saved -> {out_path}")


def plot_training_curves(history_csv: str, out_path: str):
    """Generate and save Loss and Metric training curves from history.csv."""
    if not os.path.isfile(history_csv):
        print(f"  [5/5] Warning: '{history_csv}' not found. Generating placeholder curve.")
        fig, ax = plt.subplots(figsize=(8, 4), facecolor=STYLE_BG)
        setup_dark_ax(ax, "Training Curves (No history.csv found)", "Epoch", "Value")
        ax.text(0.5, 0.5, "history.csv not found.\nTrain a model using train.py to produce training history.",
                ha="center", va="center", color=STYLE_SUBTEXT, fontsize=12)
        fig.tight_layout()
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        return

    df = pd.read_csv(history_csv)
    epochs = df["epoch"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor=STYLE_BG)

    # Subplot 1: Losses
    setup_dark_ax(ax1, "Training & Validation Loss", "Epoch", "Cross-Entropy Loss")
    ax1.plot(epochs, df["train_loss"], color=STYLE_ACCENT_BLUE, marker="o", linewidth=2, label="Train Loss")
    ax1.plot(epochs, df["val_loss"], color=STYLE_ACCENT_RED, marker="s", linewidth=2, label="Val Loss")
    ax1.legend(facecolor=STYLE_BG, edgecolor=STYLE_GRID, labelcolor=STYLE_TEXT, loc="upper right")

    # Subplot 2: Val EER & Accuracy
    setup_dark_ax(ax2, "Validation Metrics", "Epoch", "Val EER (%) [Lower is Better]")
    ax2_acc = ax2.twinx()
    ax2_acc.set_ylabel("Val Accuracy (%) [Higher is Better]", color=STYLE_ACCENT_GREEN, fontsize=11, labelpad=8)
    ax2_acc.tick_params(colors=STYLE_ACCENT_GREEN, labelsize=10)
    for spine in ax2_acc.spines.values():
        spine.set_color(STYLE_GRID)

    p1, = ax2.plot(epochs, df["val_eer"] * 100, color=STYLE_ACCENT_PURPLE, marker="^", linewidth=2, label="Val EER (%)")
    p2, = ax2_acc.plot(epochs, df["val_acc"] * 100, color=STYLE_ACCENT_GREEN, marker="d", linewidth=2, label="Val Acc (%)")

    # Mark best EER epoch
    best_idx = df["val_eer"].idxmin()
    best_epoch = df.loc[best_idx, "epoch"]
    best_eer_val = df.loc[best_idx, "val_eer"] * 100
    ax2.scatter([best_epoch], [best_eer_val], color="#facc15", s=140, marker="*", zorder=6,
                label=f"Best EER: {best_eer_val:.2f}% (Ep {int(best_epoch)})")

    lines = [p1, p2]
    ax2.legend(lines, [l.get_label() for l in lines], facecolor=STYLE_BG, edgecolor=STYLE_GRID,
               labelcolor=STYLE_TEXT, loc="center right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  [5/5] Training Curves saved -> {out_path}")


def generate_all_plots(
    scores_csv: str = None,
    history_csv: str = "checkpoints/history.csv",
    out_dir: str = "plots",
    checkpoint: str = None,
    manifest: str = None,
    cache: bool = False,
    cache_dir: str = ".cache/features",
    max_samples: int = None,
):
    """
    Produce ROC curve, DET curve, confusion matrix, score histogram, and training curves.
    Creates all five images in out_dir.
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1. Obtain labels and scores
    labels = None
    scores = None

    if scores_csv and os.path.isfile(scores_csv):
        print(f"Loading evaluated scores from '{scores_csv}'...")
        df_scores = pd.read_csv(scores_csv)
        if "label_idx" in df_scores.columns:
            labels = df_scores["label_idx"].values
        elif "label" in df_scores.columns:
            labels = df_scores["label"].str.lower().map({"bonafide": 0, "real": 0, "spoof": 1, "fake": 1}).values

        if "p_spoof" in df_scores.columns:
            scores = df_scores["p_spoof"].values
        elif "score" in df_scores.columns:
            scores = df_scores["score"].values

    elif checkpoint and manifest:
        print(f"Evaluating checkpoint '{checkpoint}' on manifest '{manifest}' to generate plot data...")
        metrics, df_results, _ = run_evaluation(
            checkpoint_path=checkpoint,
            manifest_path=manifest,
            max_samples=max_samples,
            cache=cache,
            cache_dir=cache_dir,
            save_results=False,
        )
        labels = df_results["label_idx"].values
        scores = df_results["p_spoof"].values

    else:
        # Check standard results folder for existing scores CSV
        results_dir = Path("results")
        candidates = list(results_dir.glob("*.csv")) if results_dir.is_dir() else []
        if candidates:
            cand = str(candidates[0])
            print(f"Found evaluation results at '{cand}'. Using for plots...")
            df_scores = pd.read_csv(cand)
            if "label_idx" in df_scores.columns and "p_spoof" in df_scores.columns:
                labels = df_scores["label_idx"].values
                scores = df_scores["p_spoof"].values

    if labels is None or scores is None:
        raise ValueError(
            "Could not load evaluation scores! Please provide either:\n"
            "  --scores <path/to/results.csv> (generated by evaluate.py)\n"
            "OR\n"
            "  --checkpoint <best.pt> --manifest <test.csv>"
        )

    print(f"\nGenerating all 5 publication-ready plots into '{out_dir}/'...")
    roc_path = os.path.join(out_dir, "roc_curve.png")
    det_path = os.path.join(out_dir, "det_curve.png")
    cm_path = os.path.join(out_dir, "confusion_matrix.png")
    hist_path = os.path.join(out_dir, "score_histogram.png")
    train_path = os.path.join(out_dir, "training_curves.png")

    plot_roc_curve(labels, scores, roc_path)
    plot_det_curve(labels, scores, det_path)
    plot_confusion_matrix(labels, scores, cm_path)
    plot_score_histogram(labels, scores, hist_path)
    plot_training_curves(history_csv, train_path)

    print("\nAll 5 plot images successfully generated:")
    print(f"  1. {roc_path}")
    print(f"  2. {det_path}")
    print(f"  3. {cm_path}")
    print(f"  4. {hist_path}")
    print(f"  5. {train_path}")

    return {
        "roc_curve": roc_path,
        "det_curve": det_path,
        "confusion_matrix": cm_path,
        "score_histogram": hist_path,
        "training_curves": train_path,
    }


def main():
    ap = argparse.ArgumentParser(description="Generate 5 publication-quality evaluation and training figures.")
    ap.add_argument("--scores", help="path to evaluation scores CSV (produced by evaluate.py)")
    ap.add_argument("--history", default="checkpoints/history.csv", help="path to history.csv (default: checkpoints/history.csv)")
    ap.add_argument("--out-dir", default="plots", help="output directory for PNG images (default: plots)")
    ap.add_argument("--checkpoint", help="optional checkpoint path to evaluate directly")
    ap.add_argument("--manifest", help="optional manifest CSV to evaluate directly")
    ap.add_argument("--cache", action="store_true", help="enable feature cache when evaluating checkpoint")
    ap.add_argument("--cache-dir", default=".cache/features", help="feature cache directory")
    ap.add_argument("--max-samples", type=int, default=None, help="limit samples when evaluating checkpoint")
    args = ap.parse_args()

    generate_all_plots(
        scores_csv=args.scores,
        history_csv=args.history,
        out_dir=args.out_dir,
        checkpoint=args.checkpoint,
        manifest=args.manifest,
        cache=args.cache,
        cache_dir=args.cache_dir,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
