"""
Training script for Multi-Path Fusion Network on LGG MRI Segmentation dataset.

Automatically downloads the dataset from Kaggle via kagglehub (no credentials required).
No manual dataset setup required.

Usage:
  python train.py                         # auto-downloads dataset, trains, saves weights
  python train.py --epochs 100            # custom epochs
  python train.py --data_dir /my/data     # use local dataset instead of downloading
"""

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model import MultiPathFusionNet, CombinedLoss, compute_dice
from dataset import LGGDataset, find_lgg_data_dir

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
KAGGLE_DATASET = "mateuszbuda/lgg-mri-segmentation"
DEFAULT_DATA_DIR = Path(__file__).parent / "data" / "lgg_dataset"
SAVE_DIR = Path(__file__).parent / "models"
IMG_SIZE = 256
VAL_SPLIT = 0.2  # 20% of patients for validation


# ─────────────────────────────────────────────────────────────
# Dataset download
# ─────────────────────────────────────────────────────────────

def download_lgg_dataset() -> Path:
    """
    Download LGG MRI Segmentation dataset via kagglehub if not already present.
    Returns the path to the directory containing patient folders.
    """
    # Check if data already exists
    if DEFAULT_DATA_DIR.exists():
        try:
            root = find_lgg_data_dir(str(DEFAULT_DATA_DIR))
            print(f"✓ Dataset already exists at {root}")
            return root
        except FileNotFoundError:
            pass

    import kagglehub

    print(f"📦 Downloading LGG MRI Segmentation dataset ({KAGGLE_DATASET})...")
    print("   This may take a few minutes on first run.")
    downloaded_path = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    print(f"✓ Downloaded to {downloaded_path}")

    # Find the data root within the download
    data_root = find_lgg_data_dir(str(downloaded_path))
    print(f"✓ Found training data at {data_root}")

    # Create a symlink for convenience
    DEFAULT_DATA_DIR.parent.mkdir(parents=True, exist_ok=True)
    if not DEFAULT_DATA_DIR.exists():
        try:
            DEFAULT_DATA_DIR.symlink_to(data_root)
            print(f"✓ Linked {DEFAULT_DATA_DIR} → {data_root}")
        except OSError as e:
            print(f"⚠ Could not create symlink at {DEFAULT_DATA_DIR}: {e}")
            print(f"  Using downloaded path directly: {data_root}")

    return data_root


# ─────────────────────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Resolve dataset path ─────────────────────────────────
    if args.data_dir:
        data_root = find_lgg_data_dir(args.data_dir)
        print(f"Using local dataset: {data_root}")
    else:
        data_root = download_lgg_dataset()

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
    train_ds = LGGDataset(
        data_dir=str(data_root),
        img_size=IMG_SIZE,
        patient_indices=train_indices,
    )
    val_ds = LGGDataset(
        data_dir=str(data_root),
        img_size=IMG_SIZE,
        patient_indices=val_indices,
    )
    print(f"Samples: {len(train_ds)} train / {len(val_ds)} val")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ── Model — 1 input channel (grayscale), 2 classes (background + tumor) ──
    model = MultiPathFusionNet(in_channels=1, num_classes=2).to(device)
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
        total_tumor_dice = 0.0
        n_val_samples = 0

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(device)
                pred_logits = model(imgs)
                pred_masks = pred_logits.argmax(dim=1).cpu()
                for i in range(imgs.size(0)):
                    d = compute_dice(pred_masks[i], masks[i], class_idx=1)
                    total_tumor_dice += d
                    n_val_samples += 1

        avg_tumor_dice = total_tumor_dice / max(n_val_samples, 1)
        elapsed = time.time() - t0

        print(
            f"Epoch [{epoch:3d}/{args.epochs}] | Loss: {avg_loss:.4f} | "
            f"Tumor Dice: {avg_tumor_dice:.1f}% | {elapsed:.1f}s"
        )

        # ── Save best model ──────────────────────────────────
        if avg_tumor_dice > best_dice:
            best_dice = avg_tumor_dice
            path = SAVE_DIR / "best_model.pth"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "tumor_dice": avg_tumor_dice,
            }, str(path))
            print(f"  ✓ Best model saved (Tumor Dice: {best_dice:.1f}%) → {path}")

    print(f"\nTraining complete. Best Tumor Dice: {best_dice:.1f}%")
    print(f"Model weights saved to: {SAVE_DIR / 'best_model.pth'}")


# ─────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MPFNet for Brain Tumor Segmentation")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to local LGG dataset (skips Kaggle download)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train(args)
