import argparse
import os
import csv

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import VoiceSpoofDataset
from model import VoiceCloneCNN
from metrics import compute_eer


def evaluate(model, loader, device):
    model.eval()
    all_scores, all_labels = [], []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for feats, labels in loader:
            feats, labels = feats.to(device), labels.to(device)
            logits = model(feats)
            loss = criterion(logits, labels)
            total_loss += loss.item() * feats.size(0)
            probs = torch.softmax(logits, dim=1)[:, 1]  # P(spoof)
            all_scores.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
    eer, _ = compute_eer(labels, scores)
    acc = ((scores > 0.5).astype(int) == labels).mean()
    return total_loss / len(loader.dataset), eer, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="path to train manifest CSV")
    ap.add_argument("--val", required=True, help="path to val manifest CSV")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--out", default="checkpoints/")
    ap.add_argument("--history", default=None, help="path to history CSV (defaults to <out>/history.csv)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds = VoiceSpoofDataset(args.train)
    val_ds = VoiceSpoofDataset(args.val)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    model = VoiceCloneCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_eer = float("inf")
    history_path = args.history if args.history else os.path.join(args.out, "history.csv")
    with open(history_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_eer", "val_acc"])

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for feats, labels in pbar:
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(feats)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * feats.size(0)
            pbar.set_postfix(loss=loss.item())

        train_loss = running_loss / len(train_ds)
        val_loss, val_eer, val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} val_EER={val_eer*100:.2f}% val_acc={val_acc*100:.2f}%")

        with open(history_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{val_eer:.6f}", f"{val_acc:.6f}"])

        if val_eer < best_eer:
            best_eer = val_eer
            ckpt_path = os.path.join(args.out, "best.pt")
            torch.save({"model_state": model.state_dict(), "val_eer": val_eer}, ckpt_path)
            print(f"  -> new best (EER={val_eer*100:.2f}%), saved to {ckpt_path}")

    print(f"Training done. Best val EER: {best_eer*100:.2f}%. History logged to {history_path}")


if __name__ == "__main__":
    main()
