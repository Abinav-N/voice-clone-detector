"""
AI Voice Clone Detector - Web Dashboard
Provides an interactive Gradio interface:
1. Classify Tab: Upload or record audio, view waveform & MFCC spectrogram, and get REAL/FAKE verdict with confidence.
2. Training Curves Tab: Visualize loss, validation EER, and accuracy across epochs from history.csv.
"""
import os
import sys
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import torch
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

# Ensure src/ is on sys.path
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model import VoiceCloneCNN
from features import (
    load_audio, extract_mfcc_stack, extract_sliding_windows,
    SAMPLE_RATE, N_MFCC, FIXED_FRAMES,
)

# Global model cache to avoid reloading on every request
_CACHED_MODEL = None
_CACHED_MODEL_PATH = None
_CACHED_THRESHOLD = 0.5     # EER threshold from checkpoint (Task 16)
_CACHED_FEATURE = "mfcc"    # feature type from checkpoint
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model(checkpoint_path: str):
    """Load model weights from checkpoint, caching the loaded model in memory.

    Task 16: also caches the EER threshold and feature type stored in the checkpoint.
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

    # Task 16: store EER threshold and feature type from checkpoint
    if isinstance(ckpt, dict):
        _CACHED_THRESHOLD = float(ckpt.get("eer_threshold", 0.5))
        _CACHED_FEATURE = ckpt.get("feature_type", "mfcc")
    else:
        _CACHED_THRESHOLD = 0.5
        _CACHED_FEATURE = "mfcc"

    _CACHED_MODEL = model
    _CACHED_MODEL_PATH = checkpoint_path
    return model


def plot_waveform(y: np.ndarray, sr: int = SAMPLE_RATE):
    """Generate a clean, styled waveform plot."""
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


def plot_mfcc(mfcc_channel: np.ndarray):
    """Generate a clean, styled 40-MFCC spectrogram heatmap."""
    fig, ax = plt.subplots(figsize=(8, 3.2), facecolor="#0f172a")
    ax.set_facecolor("#0f172a")
    # mfcc_channel is (40, 300)
    im = ax.imshow(
        mfcc_channel,
        aspect="auto",
        origin="lower",
        cmap="magma",
        interpolation="nearest"
    )
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.ax.tick_params(colors="#94a3b8", labelsize=8)
    cbar.outline.set_edgecolor("#334155")
    cbar.set_label("Normalized Energy", color="#94a3b8", fontsize=9)

    ax.set_title("MFCC Acoustic Fingerprint (40 Mel Coefficients)", color="#f8fafc", fontsize=12, pad=10, fontweight="bold")
    ax.set_xlabel("Time Frames (Fixed Length = 300)", color="#94a3b8", fontsize=10)
    ax.set_ylabel("MFCC Index (1-40)", color="#94a3b8", fontsize=10)
    ax.tick_params(colors="#94a3b8", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#334155")
    fig.tight_layout()
    return fig


def classify_audio(audio_path: str, checkpoint_path: str):
    """Process uploaded audio, run CNN inference, and generate waveform + MFCC plots.

    Task 15 – Long-audio: sliding-window inference for clips > 3 s.
    Task 16 – Threshold: uses EER threshold stored in the checkpoint.
    """
    if not audio_path:
        return "Please upload or record an audio file.", {}, None, None

    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        err_msg = (
            f"❌ Checkpoint file not found: '{checkpoint_path}'.\n"
            "Please train a model using `train.py` or specify an existing checkpoint path."
        )
        return err_msg, {}, None, None

    try:
        model = get_model(checkpoint_path)
    except Exception as e:
        return f"❌ Error loading model checkpoint: {str(e)}", {}, None, None

    try:
        y = load_audio(audio_path, sr=SAMPLE_RATE)
        duration_sec = len(y) / SAMPLE_RATE

        # Task 15: sliding-window over long clips; single window for short clips
        windows = extract_sliding_windows(
            y, feature_type=_CACHED_FEATURE, window_sec=3.0, hop_sec=1.5
        )  # (N, 3, 40, 300)
        n_windows = windows.shape[0]

        x = torch.from_numpy(windows).to(_DEVICE)      # (N, 3, 40, 300)
        with torch.no_grad():
            logits = model(x)                           # (N, 2)
            probs = torch.softmax(logits, dim=1)[:, 1]  # P(spoof) per window

        window_scores = probs.cpu().numpy()
        p_spoof = float(window_scores.mean())           # average across windows
        p_real = 1.0 - p_spoof

        # Task 16: use EER threshold from checkpoint
        threshold = _CACHED_THRESHOLD
        is_fake = p_spoof > threshold
        verdict_badge = "🔴 FAKE (AI-generated / Cloned)" if is_fake else "🟢 REAL (Human Speech)"

        # MFCC plot: use the first window's channel-0 features
        feat_first = windows[0]   # (3, 40, 300)

        window_note = (
            f"  ({n_windows} window(s), avg score)" if n_windows > 1 else ""
        )
        verdict_text = (
            f"### Verdict: **{verdict_badge}**\n\n"
            f"- **Confidence:** {max(p_real, p_spoof) * 100:.1f}%{window_note}\n"
            f"- **Spoof Probability (P_fake):** {p_spoof * 100:.2f}%\n"
            f"- **Human Probability (P_real):** {p_real * 100:.2f}%\n"
            f"- **Decision Threshold (EER):** {threshold:.4f}\n"
            f"- **Duration:** {duration_sec:.2f} s ({len(y)} samples @ {SAMPLE_RATE} Hz)"
        )

        confidences = {
            "Real Human Speech": p_real,
            "AI Cloned / Spoof": p_spoof,
        }

        fig_waveform = plot_waveform(y, sr=SAMPLE_RATE)
        fig_mfcc = plot_mfcc(feat_first[0])  # Channel 0 — MFCC / Log-mel

        return verdict_text, confidences, fig_waveform, fig_mfcc

    except Exception as e:
        return f"❌ Error during inference: {str(e)}", {}, None, None


def plot_training_curves(csv_file_or_path: str):
    """Parse history.csv and generate loss and metric curves."""
    if not csv_file_or_path:
        csv_file_or_path = "checkpoints/history.csv"

    # Support both file path string or uploaded file object
    path = csv_file_or_path.name if hasattr(csv_file_or_path, "name") else str(csv_file_or_path)

    if not os.path.isfile(path):
        return (
            f"⚠️ History file not found at: '{path}'.\n"
            "Train a model with `python src/train.py` to produce a `history.csv` file.",
            None,
            None,
            None
        )

    try:
        df = pd.read_csv(path)
        required_cols = {"epoch", "train_loss", "val_loss", "val_eer", "val_acc"}
        if not required_cols.issubset(set(df.columns)):
            return (
                f"⚠️ CSV must contain columns: {list(required_cols)}. Found: {list(df.columns)}",
                None,
                None,
                None
            )

        epochs = df["epoch"]

        # 1. Loss Figure
        fig_loss, ax_loss = plt.subplots(figsize=(7, 3.8), facecolor="#0f172a")
        ax_loss.set_facecolor("#0f172a")
        ax_loss.plot(epochs, df["train_loss"], label="Train Loss", color="#38bdf8", marker="o", linewidth=2)
        ax_loss.plot(epochs, df["val_loss"], label="Validation Loss", color="#f43f5e", marker="s", linewidth=2)
        ax_loss.set_title("Training & Validation Loss", color="#f8fafc", fontsize=12, pad=10, fontweight="bold")
        ax_loss.set_xlabel("Epoch", color="#94a3b8", fontsize=10)
        ax_loss.set_ylabel("Loss (Cross-Entropy)", color="#94a3b8", fontsize=10)
        ax_loss.tick_params(colors="#94a3b8", labelsize=9)
        ax_loss.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#f8fafc", loc="upper right")
        for spine in ax_loss.spines.values():
            spine.set_color("#334155")
        ax_loss.grid(True, linestyle="--", alpha=0.25, color="#64748b")
        fig_loss.tight_layout()

        # 2. Metrics Figure (EER and Accuracy)
        fig_metrics, ax_eer = plt.subplots(figsize=(7, 3.8), facecolor="#0f172a")
        ax_eer.set_facecolor("#0f172a")
        ax_acc = ax_eer.twinx()

        p1, = ax_eer.plot(epochs, df["val_eer"] * 100, label="Val EER (%)", color="#a855f7", marker="^", linewidth=2)
        p2, = ax_acc.plot(epochs, df["val_acc"] * 100, label="Val Accuracy (%)", color="#10b981", marker="d", linewidth=2)

        ax_eer.set_title("Validation EER and Accuracy", color="#f8fafc", fontsize=12, pad=10, fontweight="bold")
        ax_eer.set_xlabel("Epoch", color="#94a3b8", fontsize=10)
        ax_eer.set_ylabel("Equal Error Rate (%) [Lower is Better]", color="#a855f7", fontsize=10)
        ax_acc.set_ylabel("Accuracy (%) [Higher is Better]", color="#10b981", fontsize=10)

        ax_eer.tick_params(colors="#94a3b8", labelsize=9)
        ax_acc.tick_params(colors="#94a3b8", labelsize=9)
        ax_eer.legend([p1, p2], [p1.get_label(), p2.get_label()], facecolor="#1e293b", edgecolor="#334155", labelcolor="#f8fafc", loc="center right")

        for spine in ax_eer.spines.values():
            spine.set_color("#334155")
        for spine in ax_acc.spines.values():
            spine.set_color("#334155")
        ax_eer.grid(True, linestyle="--", alpha=0.25, color="#64748b")
        fig_metrics.tight_layout()

        # Summary text
        best_row = df.loc[df["val_eer"].idxmin()]
        summary = (
            f"### Training Summary ({len(df)} epochs)\n\n"
            f"- **Best Validation EER:** `{best_row['val_eer'] * 100:.2f}%` (Epoch {int(best_row['epoch'])})\n"
            f"- **Validation Accuracy at Best EER:** `{best_row['val_acc'] * 100:.2f}%`\n"
            f"- **Final Training Loss:** `{df['train_loss'].iloc[-1]:.4f}`\n"
            f"- **Final Validation Loss:** `{df['val_loss'].iloc[-1]:.4f}`"
        )

        return summary, fig_loss, fig_metrics, df

    except Exception as e:
        return f"❌ Error reading or plotting history: {str(e)}", None, None, None


# Build Gradio UI
custom_theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="cyan",
    neutral_hue="slate"
)

with gr.Blocks(title="AI Voice Clone Detector") as demo:
    gr.Markdown(
        """
        # 🎙️ AI Voice Clone Detector Dashboard
        Detect AI-synthesized and cloned voices from human speech using **MFCC acoustic features** and a **lightweight 2D CNN**.
        """
    )

    with gr.Tabs():
        # --- TAB 1: CLASSIFY ---
        with gr.Tab("🔍 Classify Audio"):
            with gr.Row():
                with gr.Column(scale=1):
                    audio_input = gr.Audio(
                        sources=["upload", "microphone"],
                        type="filepath",
                        label="Input Audio Clip (.wav, .flac, .mp3)"
                    )
                    checkpoint_input = gr.Textbox(
                        value="checkpoints/best.pt",
                        label="Model Checkpoint Path",
                        placeholder="Path to best.pt"
                    )
                    classify_btn = gr.Button("🚀 Run Classifier", variant="primary", size="lg")

                with gr.Column(scale=1):
                    verdict_output = gr.Markdown("Upload an audio file and click **Run Classifier** to view predictions.")
                    confidence_label = gr.Label(num_top_classes=2, label="Confidence Breakdown")

            with gr.Row():
                with gr.Column():
                    waveform_plot = gr.Plot(label="Time-Domain Waveform")
                with gr.Column():
                    mfcc_plot = gr.Plot(label="40-Channel MFCC Spectrogram")

            classify_btn.click(
                fn=classify_audio,
                inputs=[audio_input, checkpoint_input],
                outputs=[verdict_output, confidence_label, waveform_plot, mfcc_plot]
            )

        # --- TAB 2: TRAINING CURVES ---
        with gr.Tab("📈 Training Curves"):
            with gr.Row():
                with gr.Column(scale=1):
                    history_input = gr.Textbox(
                        value="checkpoints/history.csv",
                        label="History CSV File Path",
                        placeholder="checkpoints/history.csv"
                    )
                    load_history_btn = gr.Button("📊 Load & Plot Curves", variant="primary")
                    curves_summary = gr.Markdown("Click **Load & Plot Curves** to visualize training metrics.")

                with gr.Column(scale=1):
                    history_table = gr.Dataframe(label="Epoch-by-Epoch History")

            with gr.Row():
                with gr.Column():
                    loss_curve_plot = gr.Plot(label="Training & Validation Loss")
                with gr.Column():
                    metrics_curve_plot = gr.Plot(label="Validation EER & Accuracy")

            load_history_btn.click(
                fn=plot_training_curves,
                inputs=[history_input],
                outputs=[curves_summary, loss_curve_plot, metrics_curve_plot, history_table]
            )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, theme=custom_theme)
