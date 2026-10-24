"""
AI Voice Clone Detector - Web Dashboard
========================================
Tab 1  🔍 Classify Audio      – Upload/record, get REAL/FAKE verdict, waveform,
                                  MFCC spectrogram + Grad-CAM heatmap (Task 18).
Tab 2  📦 Batch Classify       – Drop in up to 50 clips, adjustable threshold
                                  slider, download predictions CSV (Task 20).
Tab 3  📊 Results Browser      – Browse saved results/*.json (ROC, per-attack
                                  table, cross-dataset table) (Task 19).
Tab 4  📈 Training Curves      – Loss / EER / Accuracy curves from history.csv.
"""
import io
import os
import sys
import glob
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

# ---------------------------------------------------------------------------
# Ensure src/ is on sys.path
# ---------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model import VoiceCloneCNN
from features import (
    load_audio, extract_mfcc_stack, extract_sliding_windows,
    extract_features, SAMPLE_RATE, N_MFCC, FIXED_FRAMES,
)

# ---------------------------------------------------------------------------
# Global model cache
# ---------------------------------------------------------------------------
_CACHED_MODEL = None
_CACHED_MODEL_PATH = None
_CACHED_THRESHOLD = 0.5      # EER threshold from checkpoint (Task 16)
_CACHED_FEATURE = "mfcc"     # feature type from checkpoint
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model(checkpoint_path: str):
    """Load model from checkpoint, caching it in memory.

    Also caches the EER threshold and feature type stored in the checkpoint.
    """
    global _CACHED_MODEL, _CACHED_MODEL_PATH, _CACHED_THRESHOLD, _CACHED_FEATURE
    checkpoint_path = os.path.abspath(checkpoint_path)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    if _CACHED_MODEL is not None and _CACHED_MODEL_PATH == checkpoint_path:
        return _CACHED_MODEL

    model = VoiceCloneCNN().to(_DEVICE)
    try:
        ckpt = torch.load(checkpoint_path, map_location=_DEVICE, weights_only=True)
    except Exception:
        ckpt = torch.load(checkpoint_path, map_location=_DEVICE, weights_only=False)
    state_dict = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    if isinstance(ckpt, dict):
        _CACHED_THRESHOLD = float(ckpt.get("eer_threshold", 0.5))
        _CACHED_FEATURE = ckpt.get("feature_type", "mfcc")
    else:
        _CACHED_THRESHOLD = 0.5
        _CACHED_FEATURE = "mfcc"

    _CACHED_MODEL = model
    _CACHED_MODEL_PATH = checkpoint_path
    return model


# ---------------------------------------------------------------------------
# Task 18 – Grad-CAM
# ---------------------------------------------------------------------------

def compute_gradcam(model: VoiceCloneCNN, feat_tensor: torch.Tensor) -> np.ndarray:
    """
    Compute Grad-CAM for the SPOOF (class 1) output w.r.t. the last conv layer
    (model.features[-1], which is a conv_block Sequential).

    Returns a heatmap of shape (H, W) in [0, 1] resized to (N_MFCC, FIXED_FRAMES).
    """
    model.eval()

    # Hook storage
    activations = {}
    gradients = {}

    # Target: last conv layer inside the last conv_block
    # model.features is nn.Sequential of 4 conv_blocks; each block is Sequential
    last_block = model.features[-1]          # conv_block Sequential
    target_layer = last_block[0]             # nn.Conv2d

    def fwd_hook(module, inp, out):
        activations["feat"] = out.detach()

    def bwd_hook(module, grad_in, grad_out):
        gradients["feat"] = grad_out[0].detach()

    fwd_handle = target_layer.register_forward_hook(fwd_hook)
    bwd_handle = target_layer.register_full_backward_hook(bwd_hook)

    # Forward pass with gradient tracking
    x = feat_tensor.clone().requires_grad_(False)
    x = x.unsqueeze(0).to(_DEVICE)  # (1, 3, 40, 300)

    # We need gradients, so run outside no_grad
    out = model(x)                          # (1, 2)
    score = out[0, 1]                       # P(spoof) logit

    model.zero_grad()
    score.backward()

    fwd_handle.remove()
    bwd_handle.remove()

    # Grad-CAM: global average pool gradients over spatial dims
    grads = gradients["feat"][0]            # (C, H, W)
    acts  = activations["feat"][0]          # (C, H, W)

    weights = grads.mean(dim=(1, 2), keepdim=True)   # (C, 1, 1)
    cam = (weights * acts).sum(dim=0)                # (H, W)
    cam = torch.clamp(cam, min=0)                    # ReLU

    # Normalise to [0, 1]
    cam_np = cam.cpu().numpy().astype(np.float32)
    cam_min, cam_max = cam_np.min(), cam_np.max()
    if cam_max - cam_min > 1e-6:
        cam_np = (cam_np - cam_min) / (cam_max - cam_min)
    else:
        cam_np = np.zeros_like(cam_np)

    # Resize to original MFCC shape via simple interpolation
    from scipy.ndimage import zoom
    zoom_h = N_MFCC / cam_np.shape[0]
    zoom_w = FIXED_FRAMES / cam_np.shape[1]
    cam_resized = zoom(cam_np, (zoom_h, zoom_w), order=1)
    return cam_resized.astype(np.float32)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_waveform(y: np.ndarray, sr: int = SAMPLE_RATE):
    fig, ax = plt.subplots(figsize=(8, 2.5), facecolor="#0f172a")
    ax.set_facecolor("#0f172a")
    time_axis = np.linspace(0, len(y) / sr, num=len(y))
    ax.plot(time_axis, y, color="#38bdf8", linewidth=0.8, alpha=0.9)
    ax.set_title("Audio Waveform", color="#f8fafc", fontsize=12, pad=10, fontweight="bold")
    ax.set_xlabel("Time (seconds)", color="#94a3b8", fontsize=10)
    ax.set_ylabel("Amplitude", color="#94a3b8", fontsize=10)
    ax.tick_params(colors="#94a3b8", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.grid(True, linestyle="--", alpha=0.25, color="#64748b")
    fig.tight_layout()
    return fig


def plot_mfcc_with_gradcam(mfcc_channel: np.ndarray, cam: np.ndarray = None, is_fake: bool = False):
    """
    Plot the MFCC spectrogram with an optional Grad-CAM heatmap overlay.
    cam: (40, 300) float32 in [0,1], or None to skip overlay.
    """
    fig, ax = plt.subplots(figsize=(8, 3.2), facecolor="#0f172a")
    ax.set_facecolor("#0f172a")

    im = ax.imshow(
        mfcc_channel,
        aspect="auto", origin="lower",
        cmap="magma", interpolation="nearest",
        alpha=0.85 if cam is not None else 1.0,
    )
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.ax.tick_params(colors="#94a3b8", labelsize=8)
    cbar.outline.set_edgecolor("#334155")
    cbar.set_label("Normalized Energy", color="#94a3b8", fontsize=9)

    # Task 18 – Grad-CAM overlay
    if cam is not None:
        overlay_cmap = "Reds" if is_fake else "Greens"
        ax.imshow(
            cam,
            aspect="auto", origin="lower",
            cmap=overlay_cmap, alpha=0.45,
            interpolation="bilinear",
            extent=[0, mfcc_channel.shape[1], 0, mfcc_channel.shape[0]],
        )
        title_suffix = " + Grad-CAM (red = pushed → FAKE)" if is_fake else " + Grad-CAM (green = pushed → REAL)"
    else:
        title_suffix = ""

    feat_label = _CACHED_FEATURE.upper() if _CACHED_FEATURE else "MFCC"
    ax.set_title(
        f"{feat_label} Acoustic Fingerprint (40 bands){title_suffix}",
        color="#f8fafc", fontsize=11, pad=10, fontweight="bold"
    )
    ax.set_xlabel("Time Frames (Fixed Length = 300)", color="#94a3b8", fontsize=10)
    ax.set_ylabel(f"{feat_label} Index (1-40)", color="#94a3b8", fontsize=10)
    ax.tick_params(colors="#94a3b8", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#334155")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Tab 1 – Single classify (Tasks 15, 16, 18)
# ---------------------------------------------------------------------------

def classify_audio(audio_path: str, checkpoint_path: str):
    """Classify a single audio clip. Returns verdict, confidence, waveform,
    and MFCC+Grad-CAM overlay."""
    if not audio_path:
        return "Please upload or record an audio file.", {}, None, None

    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        return (
            f"❌ Checkpoint not found: '{checkpoint_path}'.\n"
            "Train a model with `python src/train.py` first.",
            {}, None, None,
        )

    try:
        model = get_model(checkpoint_path)
    except Exception as e:
        return f"❌ Error loading checkpoint: {e}", {}, None, None

    try:
        y = load_audio(audio_path, sr=SAMPLE_RATE)
        duration_sec = len(y) / SAMPLE_RATE

        # Sliding-window for long clips (Task 15)
        windows = extract_sliding_windows(
            y, feature_type=_CACHED_FEATURE, window_sec=3.0, hop_sec=1.5
        )  # (N, 3, 40, 300)
        n_windows = windows.shape[0]

        # Inference
        x = torch.from_numpy(windows).to(_DEVICE)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[:, 1]

        window_scores = probs.cpu().numpy()
        p_spoof = float(window_scores.mean())
        p_real = 1.0 - p_spoof

        threshold = _CACHED_THRESHOLD
        is_fake = p_spoof > threshold
        verdict_badge = "🔴 FAKE (AI-generated / Cloned)" if is_fake else "🟢 REAL (Human Speech)"

        # Task 18 – Grad-CAM on first window
        feat_tensor = torch.from_numpy(windows[0])   # (3, 40, 300)
        try:
            cam = compute_gradcam(model, feat_tensor)
        except Exception:
            cam = None

        window_note = f"  ({n_windows} window(s), avg)" if n_windows > 1 else ""
        verdict_text = (
            f"### Verdict: **{verdict_badge}**\n\n"
            f"- **Confidence:** {max(p_real, p_spoof) * 100:.1f}%{window_note}\n"
            f"- **P(fake):** {p_spoof * 100:.2f}%  |  **P(real):** {p_real * 100:.2f}%\n"
            f"- **Decision Threshold (EER):** `{threshold:.4f}`\n"
            f"- **Duration:** {duration_sec:.2f} s\n"
            + (f"- **Grad-CAM:** {'red regions pushed model toward FAKE' if is_fake else 'green regions pushed model toward REAL'}"
               if cam is not None else "")
        )
        confidences = {"Real Human Speech": p_real, "AI Cloned / Spoof": p_spoof}

        fig_wave = plot_waveform(y)
        fig_mfcc = plot_mfcc_with_gradcam(windows[0][0], cam=cam, is_fake=is_fake)

        return verdict_text, confidences, fig_wave, fig_mfcc

    except Exception as e:
        return f"❌ Error during inference: {e}", {}, None, None


# ---------------------------------------------------------------------------
# Tab 2 – Batch classify (Task 20)
# ---------------------------------------------------------------------------

def batch_classify(files, checkpoint_path: str, threshold: float):
    """Classify multiple audio files and return a Dataframe + CSV download path."""
    if not files:
        return pd.DataFrame(), None, "⚠️ No files uploaded."

    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        return pd.DataFrame(), None, f"❌ Checkpoint not found: '{checkpoint_path}'"

    try:
        model = get_model(checkpoint_path)
    except Exception as e:
        return pd.DataFrame(), None, f"❌ Model load error: {e}"

    rows = []
    errors = []

    # files is a list of file paths (gr.File with type="filepath")
    for fpath in files:
        name = os.path.basename(fpath)
        try:
            y = load_audio(fpath, sr=SAMPLE_RATE)
            duration_sec = round(len(y) / SAMPLE_RATE, 2)

            windows = extract_sliding_windows(
                y, feature_type=_CACHED_FEATURE, window_sec=3.0, hop_sec=1.5
            )
            x = torch.from_numpy(windows).to(_DEVICE)
            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=1)[:, 1]

            p_spoof = float(probs.cpu().numpy().mean())
            p_real = 1.0 - p_spoof
            is_fake = p_spoof > threshold
            verdict = "FAKE" if is_fake else "REAL"

            rows.append({
                "filename": name,
                "verdict": verdict,
                "p_fake (%)": round(p_spoof * 100, 2),
                "p_real (%)": round(p_real * 100, 2),
                "threshold": round(threshold, 4),
                "duration_s": duration_sec,
                "n_windows": windows.shape[0],
            })
        except Exception as e:
            errors.append(f"{name}: {e}")
            rows.append({
                "filename": name,
                "verdict": "ERROR",
                "p_fake (%)": None,
                "p_real (%)": None,
                "threshold": round(threshold, 4),
                "duration_s": None,
                "n_windows": None,
            })

    df = pd.DataFrame(rows)

    # Write CSV to a temp file for download
    tmp = tempfile.NamedTemporaryFile(
        delete=False, suffix=".csv", prefix="batch_results_"
    )
    df.to_csv(tmp.name, index=False)
    tmp.close()

    n_fake = (df["verdict"] == "FAKE").sum()
    n_real = (df["verdict"] == "REAL").sum()
    summary = (
        f"### Batch Results  ({len(df)} files)\n\n"
        f"- 🔴 **FAKE:** {n_fake}  |  🟢 **REAL:** {n_real}\n"
        f"- Threshold used: `{threshold:.4f}`\n"
        + (f"\n⚠️ Errors in {len(errors)} file(s):\n" + "\n".join(f"  - {e}" for e in errors) if errors else "")
    )

    return df, tmp.name, summary


# ---------------------------------------------------------------------------
# Tab 3 – Results Browser (Task 19)
# ---------------------------------------------------------------------------

def list_result_files(results_dir: str = "results"):
    """Return a list of .json filenames found in results_dir."""
    if not os.path.isdir(results_dir):
        return []
    return sorted(glob.glob(os.path.join(results_dir, "*.json")))


def load_result(json_path: str):
    """Parse a results JSON and render summary markdown + plots."""
    if not json_path or not os.path.isfile(json_path):
        return "⚠️ Select a valid result file.", None, None, None

    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return f"❌ Failed to read file: {e}", None, None, None

    # ---- Summary markdown ----
    lines = [f"## 📄 {os.path.basename(json_path)}\n"]

    # Detect result type: robustness vs evaluate vs baseline
    if "conditions" in data:
        # Robustness result
        lines.append("### Robustness Evaluation — EER vs. Condition\n")
        rows = []
        for c in data["conditions"]:
            rows.append({
                "Condition": c.get("label", c.get("condition", "?")),
                "N": c.get("n_samples", ""),
                "EER (%)": f"{c['eer_pct']:.2f}" if c.get("eer_pct") is not None else "N/A",
                "AUC": f"{c['auc']:.4f}" if c.get("auc") is not None else "N/A",
                "Accuracy (%)": f"{c['accuracy_pct']:.2f}" if c.get("accuracy_pct") is not None else "N/A",
            })
        rob_df = pd.DataFrame(rows)
        lines.append(rob_df.to_markdown(index=False))
        summary_md = "\n".join(lines)
        return summary_md, _plot_robustness(data), None, rob_df

    elif "eer" in data and "per_attack_stats" in data:
        # Standard evaluate.py result
        lines += [
            f"- **EER:** `{data['eer_pct']:.2f}%`  |  **AUC:** `{data.get('auc') or 'N/A'}`",
            f"- **Accuracy:** `{data.get('accuracy_pct', 0):.2f}%`",
            f"- **Samples:** {data.get('num_samples', '?')}  "
            f"(bonafide: {data.get('num_bonafide', '?')}, spoof: {data.get('num_spoof', '?')})",
            f"- **Threshold:** `{data.get('eer_threshold', 0.5):.4f}`",
        ]
        # Per-attack table
        per_att = data.get("per_attack_stats", {})
        att_rows = []
        for att, stats in sorted(per_att.items()):
            att_rows.append({
                "Attack": att,
                "Count": stats.get("count", ""),
                "Detect Rate (%)": f"{stats.get('detection_rate_pct', 0):.1f}",
                "EER (%)": f"{stats['eer_pct']:.2f}" if stats.get("eer_pct") is not None else "N/A",
            })
        att_df = pd.DataFrame(att_rows) if att_rows else pd.DataFrame()
        if not att_df.empty:
            lines.append("\n### Per-Attack EER Table\n")
            lines.append(att_df.to_markdown(index=False))
        summary_md = "\n".join(lines)
        return summary_md, _plot_roc(data), _plot_per_attack(per_att), att_df

    else:
        # Baseline or other
        lines.append("```json\n" + json.dumps(data, indent=2)[:3000] + "\n```")
        summary_md = "\n".join(lines)
        return summary_md, None, None, pd.DataFrame()


def _plot_roc(data: dict):
    """Placeholder ROC-like bar chart from saved EER metrics."""
    per_att = data.get("per_attack_stats", {})
    if not per_att:
        return None

    attacks = sorted(per_att.keys())
    eers = [per_att[a]["eer_pct"] if per_att[a].get("eer_pct") is not None else 0.0 for a in attacks]

    fig, ax = plt.subplots(figsize=(max(6, len(attacks) * 0.6 + 2), 4), facecolor="#0f172a")
    ax.set_facecolor("#0f172a")
    colors = ["#f43f5e" if e > 20 else "#f59e0b" if e > 10 else "#10b981" for e in eers]
    bars = ax.bar(attacks, eers, color=colors, edgecolor="#334155", linewidth=0.8)
    ax.axhline(y=data.get("eer_pct", 0), color="#38bdf8", linestyle="--", linewidth=1.5, label=f"Overall EER {data.get('eer_pct', 0):.1f}%")
    ax.set_title("Per-Attack EER (%)", color="#f8fafc", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Attack ID", color="#94a3b8", fontsize=10)
    ax.set_ylabel("EER (%)", color="#94a3b8", fontsize=10)
    ax.tick_params(colors="#94a3b8", labelsize=9)
    ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#f8fafc")
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.grid(True, linestyle="--", alpha=0.2, color="#64748b", axis="y")
    fig.tight_layout()
    return fig


def _plot_per_attack(per_att: dict):
    """Detection-rate bar chart per attack type."""
    if not per_att:
        return None
    attacks = sorted(per_att.keys())
    rates = [per_att[a].get("detection_rate_pct", 0) for a in attacks]

    fig, ax = plt.subplots(figsize=(max(6, len(attacks) * 0.6 + 2), 4), facecolor="#0f172a")
    ax.set_facecolor("#0f172a")
    colors = ["#10b981" if r >= 80 else "#f59e0b" if r >= 50 else "#f43f5e" for r in rates]
    ax.bar(attacks, rates, color=colors, edgecolor="#334155", linewidth=0.8)
    ax.set_ylim(0, 110)
    ax.set_title("Detection Rate per Attack (%)", color="#f8fafc", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Attack ID", color="#94a3b8", fontsize=10)
    ax.set_ylabel("Detection Rate (%)", color="#94a3b8", fontsize=10)
    ax.tick_params(colors="#94a3b8", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.grid(True, linestyle="--", alpha=0.2, color="#64748b", axis="y")
    fig.tight_layout()
    return fig


def _plot_robustness(data: dict):
    """EER vs. condition bar chart for robustness results."""
    conditions = data.get("conditions", [])
    if not conditions:
        return None
    labels = [c.get("label", c.get("condition", "?")) for c in conditions]
    eers = [c.get("eer_pct", 0) for c in conditions]

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.0 + 2), 4), facecolor="#0f172a")
    ax.set_facecolor("#0f172a")
    colors = ["#38bdf8" if l == "clean" else "#a855f7" if l == "telephone" else "#f43f5e" for l in labels]
    ax.bar(labels, eers, color=colors, edgecolor="#334155", linewidth=0.8)
    ax.set_title("EER (%) vs. Acoustic Condition", color="#f8fafc", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Condition", color="#94a3b8", fontsize=10)
    ax.set_ylabel("EER (%)", color="#94a3b8", fontsize=10)
    ax.tick_params(colors="#94a3b8", labelsize=9, rotation=20)
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.grid(True, linestyle="--", alpha=0.2, color="#64748b", axis="y")
    fig.tight_layout()
    return fig


def refresh_results_dropdown(results_dir: str):
    files = list_result_files(results_dir)
    return gr.Dropdown(choices=files, value=files[0] if files else None)


# ---------------------------------------------------------------------------
# Tab 4 – Training Curves
# ---------------------------------------------------------------------------

def plot_training_curves(csv_file_or_path: str):
    if not csv_file_or_path:
        csv_file_or_path = "checkpoints/history.csv"
    path = csv_file_or_path.name if hasattr(csv_file_or_path, "name") else str(csv_file_or_path)

    if not os.path.isfile(path):
        return (
            f"⚠️ History file not found at: '{path}'.\n"
            "Train a model with `python src/train.py` to produce `history.csv`.",
            None, None, None,
        )
    try:
        df = pd.read_csv(path)
        required = {"epoch", "train_loss", "val_loss", "val_eer", "val_acc"}
        if not required.issubset(df.columns):
            return f"⚠️ CSV missing columns. Found: {list(df.columns)}", None, None, None

        epochs = df["epoch"]

        fig_loss, ax = plt.subplots(figsize=(7, 3.8), facecolor="#0f172a")
        ax.set_facecolor("#0f172a")
        ax.plot(epochs, df["train_loss"], label="Train Loss", color="#38bdf8", marker="o", linewidth=2)
        ax.plot(epochs, df["val_loss"], label="Val Loss", color="#f43f5e", marker="s", linewidth=2)
        ax.set_title("Training & Validation Loss", color="#f8fafc", fontsize=12, pad=10, fontweight="bold")
        ax.set_xlabel("Epoch", color="#94a3b8"); ax.set_ylabel("Loss", color="#94a3b8")
        ax.tick_params(colors="#94a3b8", labelsize=9)
        ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#f8fafc")
        for s in ax.spines.values(): s.set_color("#334155")
        ax.grid(True, linestyle="--", alpha=0.25, color="#64748b")
        fig_loss.tight_layout()

        fig_metrics, ax_eer = plt.subplots(figsize=(7, 3.8), facecolor="#0f172a")
        ax_eer.set_facecolor("#0f172a")
        ax_acc = ax_eer.twinx()
        p1, = ax_eer.plot(epochs, df["val_eer"] * 100, color="#a855f7", marker="^", linewidth=2, label="Val EER (%)")
        p2, = ax_acc.plot(epochs, df["val_acc"] * 100, color="#10b981", marker="d", linewidth=2, label="Val Accuracy (%)")
        ax_eer.set_title("Val EER & Accuracy", color="#f8fafc", fontsize=12, pad=10, fontweight="bold")
        ax_eer.set_xlabel("Epoch", color="#94a3b8")
        ax_eer.set_ylabel("EER (%) ↓", color="#a855f7")
        ax_acc.set_ylabel("Accuracy (%) ↑", color="#10b981")
        ax_eer.tick_params(colors="#94a3b8", labelsize=9)
        ax_acc.tick_params(colors="#94a3b8", labelsize=9)
        ax_eer.legend([p1, p2], [p1.get_label(), p2.get_label()],
                      facecolor="#1e293b", edgecolor="#334155", labelcolor="#f8fafc")
        for s in ax_eer.spines.values(): s.set_color("#334155")
        for s in ax_acc.spines.values(): s.set_color("#334155")
        ax_eer.grid(True, linestyle="--", alpha=0.25, color="#64748b")
        fig_metrics.tight_layout()

        best = df.loc[df["val_eer"].idxmin()]
        summary = (
            f"### Training Summary ({len(df)} epochs)\n\n"
            f"- **Best Val EER:** `{best['val_eer']*100:.2f}%` (Epoch {int(best['epoch'])})\n"
            f"- **Val Accuracy at Best EER:** `{best['val_acc']*100:.2f}%`\n"
            f"- **Final Train Loss:** `{df['train_loss'].iloc[-1]:.4f}`"
        )
        return summary, fig_loss, fig_metrics, df

    except Exception as e:
        return f"❌ Error: {e}", None, None, None


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
custom_theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="cyan",
    neutral_hue="slate",
)

with gr.Blocks(
    title="AI Voice Clone Detector",
    theme=custom_theme,
    css=".gradio-container { max-width: 1200px; margin: auto; }",
) as demo:
    gr.Markdown(
        """
        # 🎙️ AI Voice Clone Detector Dashboard
        Detect AI-synthesized and cloned voices using **MFCC/Log-mel features** and a **lightweight 2D CNN**.
        """
    )

    with gr.Tabs():

        # ================================================================
        # TAB 1 – Single Classify  (Tasks 15, 16, 18)
        # ================================================================
        with gr.Tab("🔍 Classify Audio"):
            with gr.Row():
                with gr.Column(scale=1):
                    audio_input = gr.Audio(
                        sources=["upload", "microphone"],
                        type="filepath",
                        label="Input Audio Clip (.wav / .flac / .mp3)",
                    )
                    checkpoint_input_1 = gr.Textbox(
                        value="checkpoints/best.pt",
                        label="Model Checkpoint Path",
                        placeholder="checkpoints/best.pt",
                    )
                    classify_btn = gr.Button("🚀 Run Classifier + Grad-CAM", variant="primary", size="lg")

                with gr.Column(scale=1):
                    verdict_output = gr.Markdown(
                        "Upload an audio file and click **Run Classifier** to view predictions."
                    )
                    confidence_label = gr.Label(num_top_classes=2, label="Confidence Breakdown")

            with gr.Row():
                with gr.Column():
                    waveform_plot = gr.Plot(label="Time-Domain Waveform")
                with gr.Column():
                    mfcc_plot = gr.Plot(label="MFCC + Grad-CAM Heatmap (Task 18)")

            classify_btn.click(
                fn=classify_audio,
                inputs=[audio_input, checkpoint_input_1],
                outputs=[verdict_output, confidence_label, waveform_plot, mfcc_plot],
            )

        # ================================================================
        # TAB 2 – Batch Classify  (Task 20)
        # ================================================================
        with gr.Tab("📦 Batch Classify"):
            gr.Markdown(
                "Upload up to **50 audio clips** at once. Adjust the decision threshold, "
                "then download a CSV with all predictions."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    batch_files = gr.File(
                        label="Audio Files (.wav / .flac / .mp3)",
                        file_count="multiple",
                        file_types=[".wav", ".flac", ".mp3", ".ogg", ".m4a"],
                    )
                    checkpoint_input_2 = gr.Textbox(
                        value="checkpoints/best.pt",
                        label="Model Checkpoint Path",
                    )
                    threshold_slider = gr.Slider(
                        minimum=0.01, maximum=0.99, step=0.01,
                        value=0.5,
                        label="Decision Threshold  (0 = strict REAL, 1 = strict FAKE)",
                        info="EER threshold from checkpoint is auto-loaded when the model is run.",
                    )
                    batch_btn = gr.Button("🚀 Classify All Files", variant="primary", size="lg")

                with gr.Column(scale=1):
                    batch_summary = gr.Markdown("Upload files and click **Classify All Files**.")
                    batch_csv_download = gr.File(label="📥 Download Predictions CSV")

            batch_table = gr.Dataframe(
                label="Batch Prediction Results",
                interactive=False,
                wrap=True,
            )

            batch_btn.click(
                fn=batch_classify,
                inputs=[batch_files, checkpoint_input_2, threshold_slider],
                outputs=[batch_table, batch_csv_download, batch_summary],
            )

        # ================================================================
        # TAB 3 – Results Browser  (Task 19)
        # ================================================================
        with gr.Tab("📊 Results Browser"):
            gr.Markdown(
                "Browse evaluation artefacts saved by `evaluate.py`, `robustness.py`, "
                "or `cross_eval.py` in the `results/` directory."
            )
            with gr.Row():
                with gr.Column(scale=2):
                    results_dir_input = gr.Textbox(
                        value="results",
                        label="Results Directory",
                        placeholder="results/",
                    )
                with gr.Column(scale=3):
                    results_file_dropdown = gr.Dropdown(
                        choices=list_result_files("results"),
                        label="Select Result File",
                        interactive=True,
                    )
                with gr.Column(scale=1):
                    refresh_btn = gr.Button("🔄 Refresh List")
                    load_result_btn = gr.Button("📂 Load Result", variant="primary")

            results_summary = gr.Markdown("Select a result file and click **Load Result**.")

            with gr.Row():
                with gr.Column():
                    results_plot1 = gr.Plot(label="EER / Attack Chart")
                with gr.Column():
                    results_plot2 = gr.Plot(label="Detection Rate Chart")

            results_table = gr.Dataframe(label="Detailed Table", interactive=False)

            refresh_btn.click(
                fn=refresh_results_dropdown,
                inputs=[results_dir_input],
                outputs=[results_file_dropdown],
            )
            load_result_btn.click(
                fn=load_result,
                inputs=[results_file_dropdown],
                outputs=[results_summary, results_plot1, results_plot2, results_table],
            )

        # ================================================================
        # TAB 4 – Training Curves
        # ================================================================
        with gr.Tab("📈 Training Curves"):
            with gr.Row():
                with gr.Column(scale=1):
                    history_input = gr.Textbox(
                        value="checkpoints/history.csv",
                        label="History CSV Path",
                        placeholder="checkpoints/history.csv",
                    )
                    load_history_btn = gr.Button("📊 Load & Plot Curves", variant="primary")
                    curves_summary = gr.Markdown("Click **Load & Plot Curves** to visualize training.")

                with gr.Column(scale=1):
                    history_table = gr.Dataframe(label="Epoch-by-Epoch History")

            with gr.Row():
                with gr.Column():
                    loss_curve_plot = gr.Plot(label="Train & Validation Loss")
                with gr.Column():
                    metrics_curve_plot = gr.Plot(label="Val EER & Accuracy")

            load_history_btn.click(
                fn=plot_training_curves,
                inputs=[history_input],
                outputs=[curves_summary, loss_curve_plot, metrics_curve_plot, history_table],
            )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, theme=custom_theme)

