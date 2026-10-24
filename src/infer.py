"""
Single-clip inference for the AI Voice Clone Detector.

Tasks implemented:
  Task 15 – Long-audio inference: slides a 3 s window across clips longer than
             3 seconds and averages the window scores, giving one stable verdict.
  Task 16 – Threshold choice: uses the EER threshold saved in the checkpoint
             instead of the hard-coded 0.5.
"""
import argparse
import sys
from pathlib import Path

# Universal import hardening
_SRC_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SRC_DIR.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch

try:
    from features import (
        load_audio,
        extract_features,
        extract_sliding_windows,
        SAMPLE_RATE,
        WINDOW_SAMPLES,
    )
    from model import VoiceCloneCNN
except ImportError:
    from src.features import (
        load_audio,
        extract_features,
        extract_sliding_windows,
        SAMPLE_RATE,
        WINDOW_SAMPLES,
    )
    from src.model import VoiceCloneCNN


# --------------------------------------------------------------------------- #
#  Model loading                                                               #
# --------------------------------------------------------------------------- #

def load_model(checkpoint_path: str, device):
    """Load model and return (model, eer_threshold, feature_type)."""
    model = VoiceCloneCNN().to(device)
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    state_dict = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    # Task 16: read EER threshold from checkpoint; fall back to 0.5 if absent
    eer_threshold = ckpt.get("eer_threshold", 0.5) if isinstance(ckpt, dict) else 0.5
    feature_type = ckpt.get("feature_type", "mfcc") if isinstance(ckpt, dict) else "mfcc"

    return model, float(eer_threshold), feature_type


# --------------------------------------------------------------------------- #
#  Inference                                                                   #
# --------------------------------------------------------------------------- #

def classify(
    model,
    wav_path: str,
    device,
    threshold: float = 0.5,
    feature_type: str = "mfcc",
    window_sec: float = 3.0,
    hop_sec: float = 1.5,
) -> dict:
    """
    Classify one audio clip.

    Task 15 – Long-audio: if the clip is longer than window_sec, a sliding
    window is applied and scores are averaged before thresholding.
    Task 16 – Uses the supplied EER threshold instead of 0.5.

    Returns a dict with label, per-window scores, final p_spoof, and p_real.
    """
    y = load_audio(wav_path)
    duration_sec = len(y) / SAMPLE_RATE

    # Task 15: sliding-window or single-clip extraction
    windows = extract_sliding_windows(
        y,
        feature_type=feature_type,
        window_sec=window_sec,
        hop_sec=hop_sec,
    )  # shape: (N, 3, 40, 300) — N >= 1

    n_windows = windows.shape[0]

    # Batch inference over all windows
    x = torch.from_numpy(windows).to(device)          # (N, 3, 40, 300)
    with torch.no_grad():
        logits = model(x)                              # (N, 2)
        probs = torch.softmax(logits, dim=1)[:, 1]    # (N,)  — P(spoof)

    window_scores = probs.cpu().numpy().tolist()
    p_spoof = float(np.mean(window_scores))           # average across windows
    p_real = 1.0 - p_spoof

    # Task 16: use EER threshold
    is_fake = p_spoof > threshold
    label = "FAKE (AI-generated)" if is_fake else "REAL (human)"

    return {
        "label": label,
        "p_spoof": p_spoof,
        "p_real": p_real,
        "threshold_used": threshold,
        "n_windows": n_windows,
        "window_scores": window_scores,
        "duration_sec": duration_sec,
        "feature_type": feature_type,
    }


# --------------------------------------------------------------------------- #
#  CLI entry point                                                             #
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(
        description="Single-clip inference with sliding-window support and EER threshold."
    )
    ap.add_argument("--checkpoint", required=True, help="Path to checkpoint (.pt)")
    ap.add_argument("--wav", required=True, help="Path to audio file (.wav/.flac/.mp3)")
    ap.add_argument(
        "--threshold", type=float, default=None,
        help="Override decision threshold (default: use EER threshold from checkpoint)"
    )
    ap.add_argument(
        "--window-sec", type=float, default=3.0,
        help="Sliding window length in seconds (default: 3.0)"
    )
    ap.add_argument(
        "--hop-sec", type=float, default=1.5,
        help="Sliding window hop in seconds (default: 1.5)"
    )
    ap.add_argument(
        "--feature", choices=["mfcc", "logmel"], default=None,
        help="Feature type override (default: read from checkpoint)"
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, eer_threshold, ckpt_feature = load_model(args.checkpoint, device)

    # Allow CLI overrides
    threshold = args.threshold if args.threshold is not None else eer_threshold
    feature_type = args.feature if args.feature is not None else ckpt_feature

    print(f"\n  Checkpoint  : {args.checkpoint}")
    print(f"  Feature     : {feature_type}")
    print(f"  Threshold   : {threshold:.4f} "
          f"({'EER from checkpoint' if args.threshold is None else 'CLI override'})")

    result = classify(
        model, args.wav, device,
        threshold=threshold,
        feature_type=feature_type,
        window_sec=args.window_sec,
        hop_sec=args.hop_sec,
    )

    print(f"\n  File        : {args.wav}")
    print(f"  Duration    : {result['duration_sec']:.2f}s  ({result['n_windows']} window(s))")
    print(f"  -> {result['label']}")
    print(f"     P(fake)  = {result['p_spoof'] * 100:.1f}%")
    print(f"     P(real)  = {result['p_real'] * 100:.1f}%")

    if result["n_windows"] > 1:
        per_w = ", ".join(f"{s * 100:.1f}%" for s in result["window_scores"])
        print(f"     Per-window P(fake): [{per_w}]")


if __name__ == "__main__":
    main()
