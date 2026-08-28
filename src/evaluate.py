import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

from dataset import VoiceSpoofDataset
from model import VoiceCloneCNN
from metrics import compute_eer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", required=True, help="CSV manifest to evaluate on")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = VoiceSpoofDataset(args.manifest)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = VoiceCloneCNN().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
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
    auc = roc_auc_score(labels, scores)

    print(f"EER:  {eer*100:.2f}%  (threshold={eer_threshold:.3f})")
    print(f"AUC:  {auc:.4f}")
    print()
    print("Confusion matrix (rows=true, cols=pred; 0=bonafide, 1=spoof):")
    print(confusion_matrix(labels, preds))
    print()
    print(classification_report(labels, preds, target_names=["bonafide", "spoof"]))


if __name__ == "__main__":
    main()
