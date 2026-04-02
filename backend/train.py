"""
Training script for Multi-Path Fusion Network on BraTS dataset.

Automatically downloads the BraTS 2020 dataset from Kaggle via kagglehub.
No manual dataset setup required.

Usage:
  python train.py                         # auto-downloads BraTS, trains, saves weights
  python train.py --epochs 100            # custom epochs
  python train.py --data_dir /my/brats    # use local dataset instead of downloading
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from model import MultiPathFusionNet, CombinedLoss, compute_all_dice
from dataset import BraTSDataset, find_brats_training_dir

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
KAGGLE_DATASET = "awsaf49/brats20-dataset-training-validation"
DEFAULT_DATA_DIR = Path(__file__).parent / "data" / "BraTS2020"
SAVE_DIR = Path(__file__).parent / "models"
IMG_SIZE = 256
VAL_SPLIT = 0.2  # 20% of patients for validation


# ─────────────────────────────────────────────────────────────
# Dataset download
# ─────────────────────────────────────────────────────────────

def _has_patient_dirs(path: Path) -> bool:
    """Check if a directory contains BraTS patient subdirectories."""
    if not path.is_dir():
        return False
    for child in path.iterdir():
        if child.is_dir() and (
            list(child.glob("*_t1.nii*")) or list(child.glob("*_t1.nii.gz"))
        ):
            return True
    return False


def download_brats_dataset() -> Path:
    """
    Download BraTS 2020 dataset via kagglehub if not already present.
    Returns the path to the directory containing patient folders.
    """
    # Check if data already exists in backend/data/BraTS2020
    if DEFAULT_DATA_DIR.exists():
        try:
            root = find_brats_training_dir(str(DEFAULT_DATA_DIR))
            print(f"✓ Dataset already exists at {root}")
            return root
        except FileNotFoundError:
            pass

    import kagglehub

    print(f"📦 Downloading BraTS 2020 dataset from Kaggle ({KAGGLE_DATASET})...")
    print("   This may take several minutes on first run.")
    downloaded_path = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    print(f"✓ Downloaded to {downloaded_path}")

    # Find the training data root within the download
    training_root = find_brats_training_dir(str(downloaded_path))
    print(f"✓ Found training data at {training_root}")

    # Create a symlink at backend/data/BraTS2020 for convenience
    DEFAULT_DATA_DIR.parent.mkdir(parents=True, exist_ok=True)
    if not DEFAULT_DATA_DIR.exists():
        try:
            DEFAULT_DATA_DIR.symlink_to(training_root)
            print(f"✓ Linked {DEFAULT_DATA_DIR} → {training_root}")
        except OSError:
            # Symlink may fail on some systems; that's fine, we use the path directly
            pass

    return training_root


# ─────────────────────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Resolve dataset path ─────────────────────────────────
    if args.data_dir:
        data_root = find_brats_training_dir(args.data_dir)
        print(f"Using local dataset: {data_root}")
    else:
        data_root = download_brats_dataset()

    # ── Discover patients and split into train/val ───────────
    patient_dirs = sorted([
        d for d in Path(data_root).iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])
    n_patients = len(patient_dirs)
    n_val = max(1, int(n_patients * VAL_SPLIT))
    n_train = n_patients - n_val

    train_indices = list(range(n_train))
    val_indices = list(range(n_train, n_patients))
    print(f"Patients: {n_patients} total → {n_train} train / {n_val} val")

    # ── Create datasets ──────────────────────────────────────
    train_ds = BraTSDataset(
        data_dir=str(data_root),
        img_size=IMG_SIZE,
        patient_indices=train_indices,
    )
    val_ds = BraTSDataset(
        data_dir=str(data_root),
        img_size=IMG_SIZE,
        patient_indices=val_indices,
        slices_per_volume=5,
    )
    print(f"Samples: {len(train_ds)} train / {len(val_ds)} val")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ── Model — 4 input channels for T1, T1ce, T2, FLAIR ────
    model = MultiPathFusionNet(in_channels=4, num_classes=4).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {params:,}")

    # ── Optimizer + LR scheduler ─────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = CombinedLoss(alpha=0.5)

    best_dice = 0.0
    SAVE_DIR.mkdir(exist_ok=True)

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
        val_dice = {"whole_tumor": 0.0, "tumor_core": 0.0, "enhancing_tumor": 0.0}
        n_val_samples = 0

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(device)
                pred_logits = model(imgs)
                pred_masks = pred_logits.argmax(dim=1).cpu()
                for i in range(imgs.size(0)):
                    d = compute_all_dice(pred_masks[i], masks[i])
                    for k in val_dice:
                        val_dice[k] += d[k]
                    n_val_samples += 1

        for k in val_dice:
            val_dice[k] /= max(n_val_samples, 1)

        avg_val_dice = sum(val_dice.values()) / 3
        elapsed = time.time() - t0

        print(
            f"Epoch [{epoch:3d}/{args.epochs}] | Loss: {avg_loss:.4f} | "
            f"WT: {val_dice['whole_tumor']:.1f}% | "
            f"TC: {val_dice['tumor_core']:.1f}% | "
            f"ET: {val_dice['enhancing_tumor']:.1f}% | "
            f"Avg: {avg_val_dice:.1f}% | {elapsed:.1f}s"
        )

        # ── Save best model ──────────────────────────────────
        if avg_val_dice > best_dice:
            best_dice = avg_val_dice
            path = SAVE_DIR / "best_model.pth"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "dice_scores": val_dice,
            }, str(path))
            print(f"  ✓ Best model saved (avg Dice: {best_dice:.1f}%) → {path}")

    print(f"\nTraining complete. Best Avg Dice: {best_dice:.1f}%")
    print(f"Model weights saved to: {SAVE_DIR / 'best_model.pth'}")
    print("Reference target (Wu et al. 2023): WT≥90%, TC≥90%, ET≥85%")


# ─────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MPFNet for Brain Tumor Segmentation")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to local BraTS data (skips Kaggle download)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train(args)
