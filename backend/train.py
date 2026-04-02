"""
Training script for Multi-Path Fusion Network on BraTS dataset.

Usage with real BraTS data:
  python train.py --data_dir /path/to/BraTS --epochs 50 --batch_size 4

For pipeline testing with 4-channel synthetic data:
  python train.py --synthetic --epochs 5
"""

import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from model import MultiPathFusionNet, CombinedLoss, compute_all_dice, dice_score


# ─────────────────────────────────────────────────────────────
# 4-Channel Synthetic Dataset (for pipeline testing only)
# ─────────────────────────────────────────────────────────────

class SyntheticBraTS4ChDataset(Dataset):
    """
    Generates 4-channel synthetic brain MRI slices with segmentation masks
    for pipeline testing. Produces tensors matching the real BraTS format:
      image: (4, 256, 256) — simulating T1, T1ce, T2, FLAIR
      mask:  (256, 256) — classes {0, 1, 2, 3}
    """

    def __init__(self, n_samples: int = 200, img_size: int = 256):
        self.n = n_samples
        self.sz = img_size

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        rng = np.random.RandomState(idx)
        h = w = self.sz

        mask = np.zeros((h, w), dtype=np.int64)
        Y, X = np.ogrid[:h, :w]
        cy = h // 2 + rng.randint(-10, 11)
        cx = w // 2 + rng.randint(-10, 11)

        # Brain region
        brain = ((X - cx) ** 2 / 85**2 + (Y - cy) ** 2 / 100**2) <= 1.0

        # Tumor parameters
        tx = cx + rng.randint(-25, 26)
        ty = cy + rng.randint(-25, 26)
        s = rng.uniform(0.6, 1.4)

        wt = (((X - tx) ** 2 / (45*s)**2 + (Y - ty)**2 / (50*s)**2) <= 1.0) & brain
        tc = (((X - tx)**2 / (25*s)**2 + (Y - ty)**2 / (28*s)**2) <= 1.0) & wt
        etx, ety = tx + rng.randint(-5, 6), ty + rng.randint(-5, 6)
        et = (((X - etx)**2 / (12*s)**2 + (Y - ety)**2 / (13*s)**2) <= 1.0) & tc

        mask[wt] = 1
        mask[tc] = 2
        mask[et] = 3

        # Generate 4 modality channels with slightly different intensity profiles
        channels = []
        for mod_idx in range(4):
            img = np.zeros((h, w), dtype=np.float32)
            base_offset = mod_idx * 0.05
            img[brain] = rng.uniform(0.3 + base_offset, 0.5 + base_offset, int(brain.sum()))

            wm = ((X - cx) ** 2 / 70**2 + (Y - cy) ** 2 / 85**2) <= 1.0
            img[wm] = rng.uniform(0.5 + base_offset, 0.7 + base_offset, int(wm.sum()))

            img[wt] = rng.uniform(0.65 + base_offset, 0.8 + base_offset, int(wt.sum()))
            img[tc] = rng.uniform(0.78 + base_offset, 0.9 + base_offset, int(tc.sum()))
            img[et] = rng.uniform(0.88, 1.0, int(et.sum()))

            img += rng.normal(0, 0.02, (h, w)).astype(np.float32)
            img = np.clip(img, 0, 1)
            channels.append(img)

        img_t = torch.from_numpy(np.stack(channels, axis=0))  # (4, H, W)
        mask_t = torch.from_numpy(mask)                         # (H, W)
        return img_t, mask_t


# ─────────────────────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Dataset
    if args.synthetic:
        train_ds = SyntheticBraTS4ChDataset(n_samples=400, img_size=256)
        val_ds   = SyntheticBraTS4ChDataset(n_samples=80,  img_size=256)
        print(f"Using 4-channel synthetic dataset for pipeline testing: "
              f"{len(train_ds)} train / {len(val_ds)} val")
    else:
        from dataset import BraTSDataset
        if not args.data_dir:
            raise ValueError(
                "BraTS data directory required. Use --data_dir /path/to/BraTS "
                "or --synthetic for pipeline testing."
            )
        train_ds = BraTSDataset(data_dir=args.data_dir, img_size=256)
        val_ds   = BraTSDataset(data_dir=args.data_dir, img_size=256, slices_per_volume=5)
        print(f"Using BraTS dataset: {len(train_ds)} train / {len(val_ds)} val")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model — 4 input channels for T1, T1ce, T2, FLAIR
    model = MultiPathFusionNet(in_channels=4, num_classes=4).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {params:,}")

    # Optimizer + LR scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = CombinedLoss(alpha=0.5)

    best_dice = 0.0
    save_dir = Path(__file__).parent / "models"
    save_dir.mkdir(exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # ── Train ────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        t0 = time.time()

        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        # ── Validate ─────────────────────────────────────────
        model.eval()
        val_dice = {"whole_tumor": 0, "tumor_core": 0, "enhancing_tumor": 0}
        n_val = 0

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(device)
                pred_logits = model(imgs)
                pred_masks  = pred_logits.argmax(dim=1).cpu()
                for i in range(imgs.size(0)):
                    d = compute_all_dice(pred_masks[i], masks[i])
                    for k in val_dice:
                        val_dice[k] += d[k]
                    n_val += 1

        for k in val_dice:
            val_dice[k] /= max(n_val, 1)

        avg_val_dice = sum(val_dice.values()) / 3
        elapsed = time.time() - t0

        print(
            f"Epoch [{epoch:3d}/{args.epochs}] | Loss: {avg_loss:.4f} | "
            f"WT: {val_dice['whole_tumor']:.1f}% | "
            f"TC: {val_dice['tumor_core']:.1f}% | "
            f"ET: {val_dice['enhancing_tumor']:.1f}% | "
            f"Avg: {avg_val_dice:.1f}% | {elapsed:.1f}s"
        )

        # Save best model
        if avg_val_dice > best_dice:
            best_dice = avg_val_dice
            path = save_dir / "best_model.pth"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "dice_scores": val_dice,
            }, str(path))
            print(f"  ✓ Best model saved (avg Dice: {best_dice:.1f}%) → {path}")

    print(f"\nTraining complete. Best Avg Dice: {best_dice:.1f}%")
    print("Reference target (Wu et al. 2023): WT≥90%, TC≥90%, ET≥85%")


# ─────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MPFNet for Brain Tumor Segmentation")
    parser.add_argument("--data_dir",   type=str, default=None,
                        help="Path to BraTS training data directory")
    parser.add_argument("--synthetic",  action="store_true", default=False,
                        help="Use 4-channel synthetic data for pipeline testing")
    parser.add_argument("--epochs",     type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr",         type=float, default=1e-3)
    args = parser.parse_args()

    if not args.synthetic and not args.data_dir:
        print("ERROR: No data directory specified. Use --data_dir or --synthetic.")
        print("  For real training: python train.py --data_dir /path/to/BraTS")
        print("  For pipeline test: python train.py --synthetic --epochs 5")
        exit(1)

    train(args)
