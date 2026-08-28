import argparse

import torch

from features import extract_features_from_path
from model import VoiceCloneCNN


def load_model(checkpoint_path: str, device):
    model = VoiceCloneCNN().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def classify(model, wav_path: str, device) -> dict:
    feat = extract_features_from_path(wav_path)          # (3, n_mfcc, T)
    x = torch.from_numpy(feat).unsqueeze(0).to(device)    # (1, 3, n_mfcc, T)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
    p_spoof = probs[1].item()
    label = "FAKE (AI-generated)" if p_spoof > 0.5 else "REAL (human)"
    return {"label": label, "p_spoof": p_spoof, "p_real": 1 - p_spoof}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--wav", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    result = classify(model, args.wav, device)

    print(f"{args.wav}")
    print(f"  -> {result['label']}  (P(fake)={result['p_spoof']*100:.1f}%, "
          f"P(real)={result['p_real']*100:.1f}%)")


if __name__ == "__main__":
    main()
