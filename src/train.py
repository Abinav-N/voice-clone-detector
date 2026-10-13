import argparse
import os
import csv

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import VoiceSpoofDataset
from model import VoiceCloneCNN
from metrics import compute_eer


def evaluate(model, loader, device, criterion=None):
    model.eval()
    all_scores, all_labels = [], []
    total_loss = 0.0
    if criterion is None:
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
    ap = argparse.ArgumentParser(description="Train VoiceCloneCNN detector.")
    ap.add_argument("--train", required=True, help="path to train manifest CSV")
    ap.add_argument("--val", required=True, help="path to val manifest CSV")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--out", default="checkpoints/")
    ap.add_argument("--history", default=None, help="path to history CSV (defaults to <out>/history.csv)")
    # Task 3: max-samples for smoke tests
    ap.add_argument("--max-samples", type=int, default=None, help="limit samples per split for quick smoke tests")
    # Task 4: Feature caching
    ap.add_argument("--cache", action="store_true", help="enable feature caching to .npy files on disk")
    ap.add_argument("--cache-dir", default=".cache/features", help="directory to store cached .npy features")
    ap.add_argument("--precompute", action="store_true", help="precompute and cache all features before training")
    # Task 5: Class-imbalance handling
    ap.add_argument("--weighted-loss", action="store_true", help="use class-weighted loss computed from train label counts")
    # Task 6: LR scheduler and early stopping
    ap.add_argument("--patience", type=int, default=5, help="early stopping patience (epochs without val EER improvement, 0 to disable)")
    ap.add_argument("--lr-patience", type=int, default=3, help="patience for ReduceLROnPlateau (epochs)")
    ap.add_argument("--lr-factor", type=float, default=0.5, help="factor to multiply lr on plateau")
    ap.add_argument("--min-lr", type=float, default=1e-6, help="minimum lr for ReduceLROnPlateau")
    ap.add_argument("--min-delta", type=float, default=1e-4, help="minimum improvement in val EER to qualify as new best")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cache_dir = args.cache_dir if (args.cache or args.precompute) else None
    if cache_dir:
        print(f"Feature caching active: storing/reading .npy files at '{cache_dir}'")

    train_ds = VoiceSpoofDataset(args.train, max_samples=args.max_samples, cache_dir=cache_dir)
    val_ds = VoiceSpoofDataset(args.val, max_samples=args.max_samples, cache_dir=cache_dir)

    if args.precompute and cache_dir:
        train_ds.precompute_cache()
        val_ds.precompute_cache()

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = VoiceCloneCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=args.min_lr,
    )

    # Class-imbalance weighted loss
    if args.weighted_loss:
        n_bonafide = int((train_ds.df["label_idx"] == 0).sum())
        n_spoof = int((train_ds.df["label_idx"] == 1).sum())
        total = len(train_ds)
        if n_bonafide == 0 or n_spoof == 0:
            print("Warning: One of the classes has 0 samples. Falling back to unweighted CrossEntropyLoss.")
            criterion = nn.CrossEntropyLoss()
        else:
            w0 = total / (2.0 * n_bonafide)
            w1 = total / (2.0 * n_spoof)
            weights = torch.tensor([w0, w1], dtype=torch.float32).to(device)
            criterion = nn.CrossEntropyLoss(weight=weights)
            print(f"Class-weighted loss active (bonafide weight={w0:.3f} [{n_bonafide} clips], spoof weight={w1:.3f} [{n_spoof} clips])")
    else:
        criterion = nn.CrossEntropyLoss()

    best_eer = float("inf")
    epochs_no_improve = 0
    history_path = args.history if args.history else os.path.join(args.out, "history.csv")
    with open(history_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_eer", "val_acc", "lr"])

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
        val_loss, val_eer, val_acc = evaluate(model, val_loader, device, criterion=criterion)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch}: train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} val_EER={val_eer*100:.2f}% val_acc={val_acc*100:.2f}% lr={current_lr:.6e}")

        with open(history_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{val_eer:.6f}", f"{val_acc:.6f}", f"{current_lr:.6e}"])

        # Learning rate scheduler step
        old_lr = current_lr
        scheduler.step(val_eer)
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr < old_lr:
            print(f"  -> Learning rate reduced: {old_lr:.6e} -> {new_lr:.6e}")

        # Model checkpointing & early stopping
        if val_eer < best_eer - args.min_delta:
            best_eer = val_eer
            epochs_no_improve = 0
            ckpt_path = os.path.join(args.out, "best.pt")
            torch.save({"model_state": model.state_dict(), "val_eer": float(val_eer)}, ckpt_path)
            print(f"  -> new best (EER={val_eer*100:.2f}%), saved to {ckpt_path}")
        else:
            epochs_no_improve += 1
            if args.patience > 0:
                print(f"  -> No val EER improvement for {epochs_no_improve}/{args.patience} epochs (best: {best_eer*100:.2f}%)")
                if epochs_no_improve >= args.patience:
                    print(f"\n[Early Stopping] val EER has not improved for {args.patience} consecutive epochs. Early stopping at epoch {epoch}.")
                    break

    print(f"Training done. Best val EER: {best_eer*100:.2f}%. History logged to {history_path}")


if __name__ == "__main__":
    main()
