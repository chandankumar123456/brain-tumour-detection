"""
BraTS Dataset Loader for Multi-Path Fusion Network.
Loads NIfTI (.nii / .nii.gz) files with 4 MRI modalities and segmentation masks.

Supports BraTS 2019/2020/2021 folder structures:
  BraTS20_Training_001/
    BraTS20_Training_001_t1.nii.gz
    BraTS20_Training_001_t1ce.nii.gz
    BraTS20_Training_001_t2.nii.gz
    BraTS20_Training_001_flair.nii.gz
    BraTS20_Training_001_seg.nii.gz

Usage:
  dataset = BraTSDataset(data_dir="/path/to/BraTS2020_TrainingData")
  img, mask = dataset[0]   # img: (4, 256, 256), mask: (256, 256)
"""

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import nibabel as nib
except ImportError:
    nib = None


IMG_SIZE = 256

# BraTS label mapping:
#   0 = Background
#   1 = Necrotic / Non-Enhancing Tumor (NCR/NET) → maps to Whole Tumor
#   2 = Peritumoral Edema (ED)                    → maps to Whole Tumor
#   4 = GD-Enhancing Tumor (ET)                   → maps to Enhancing Tumor
#
# Standard evaluation regions:
#   Whole Tumor (WT)    = labels {1, 2, 4}  → class 1
#   Tumor Core (TC)     = labels {1, 4}     → class 2
#   Enhancing Tumor (ET)= labels {4}        → class 3


def _map_brats_labels(seg: np.ndarray) -> np.ndarray:
    """
    Map raw BraTS segmentation labels to model classes:
      0 → 0 (Background)
      1, 2 → 1 (Whole Tumor)
      1, 4 → 2 (Tumor Core)  — only voxels that are also core
      4 → 3 (Enhancing Tumor)

    We assign the MOST SPECIFIC label to each voxel (enhancing > core > whole).
    """
    mapped = np.zeros_like(seg, dtype=np.int64)
    # Whole Tumor: all tumor labels
    mapped[(seg == 1) | (seg == 2) | (seg == 4)] = 1
    # Tumor Core: NCR/NET + ET (labels 1 and 4)
    mapped[(seg == 1) | (seg == 4)] = 2
    # Enhancing Tumor: label 4
    mapped[seg == 4] = 3
    return mapped


def _find_modality_file(patient_dir: Path, modality: str) -> Path:
    """Find a NIfTI file for a given modality in a patient directory."""
    patterns = [
        f"*_{modality}.nii.gz",
        f"*_{modality}.nii",
        f"*{modality}.nii.gz",
        f"*{modality}.nii",
    ]
    for pattern in patterns:
        matches = list(patient_dir.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"No {modality} file found in {patient_dir}. "
        f"Expected pattern: *_{modality}.nii.gz"
    )


def find_brats_training_dir(base_path: str | Path) -> Path:
    """
    Walk directory tree to find the directory containing BraTS patient folders.
    Handles various BraTS directory structures (2019/2020/2021).

    Returns the path containing patient subdirectories (e.g. BraTS20_Training_001/).
    """
    base = Path(base_path)
    candidates = [
        base / "BraTS2020_TrainingData" / "MICCAI_BraTS2020_TrainingData",
        base / "MICCAI_BraTS2020_TrainingData",
        base / "BraTS2020_TrainingData",
        base / "BraTS2021_TrainingData",
        base / "MICCAI_BraTS2021_TrainingData",
        base,
    ]
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        for child in candidate.iterdir():
            if child.is_dir() and (
                list(child.glob("*_t1.nii*")) or list(child.glob("*_t1.nii.gz"))
            ):
                return candidate

    raise FileNotFoundError(
        f"Cannot find BraTS training data (patient directories) in {base_path}. "
        f"Expected structure: .../BraTS20_Training_001/*_t1.nii.gz"
    )


class BraTSDataset(Dataset):
    """
    Dataset loader for BraTS challenge data.

    Expects directory structure:
      data_dir/
        PatientXXX/
          *_t1.nii.gz
          *_t1ce.nii.gz
          *_t2.nii.gz
          *_flair.nii.gz
          *_seg.nii.gz

    Each __getitem__ returns a 2D axial slice with all 4 modalities:
      image: (4, IMG_SIZE, IMG_SIZE) float32 tensor, normalized to [0, 1]
      mask:  (IMG_SIZE, IMG_SIZE) int64 tensor with classes {0, 1, 2, 3}

    Args:
        data_dir: Path to BraTS training data directory.
        img_size: Target image size (default 256).
        slices_per_volume: Number of axial slices to sample per volume.
            If None, uses all slices containing tumor.
        patient_indices: Optional list of patient directory indices to use
            (for train/val splitting). If None, uses all patients.
    """

    def __init__(
        self,
        data_dir: str,
        img_size: int = IMG_SIZE,
        slices_per_volume: int | None = None,
        patient_indices: list[int] | None = None,
    ):
        if nib is None:
            raise ImportError(
                "nibabel is required for BraTS data loading. "
                "Install with: pip install nibabel"
            )

        self.img_size = img_size
        self.data_dir = Path(data_dir)

        if not self.data_dir.exists():
            raise FileNotFoundError(f"BraTS data directory not found: {data_dir}")

        # Discover patient directories (only those with seg files for training)
        all_patient_dirs = sorted([
            d for d in self.data_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ])

        if not all_patient_dirs:
            raise FileNotFoundError(
                f"No patient directories found in {data_dir}. "
                f"Expected structure: data_dir/PatientXXX/*_t1.nii.gz"
            )

        # Filter to requested patient indices
        if patient_indices is not None:
            patient_dirs = [all_patient_dirs[i] for i in patient_indices
                           if i < len(all_patient_dirs)]
        else:
            patient_dirs = all_patient_dirs

        # Build index of (patient_dir, slice_idx) pairs
        self.samples: list[tuple[Path, int]] = []
        modalities = ["t1", "t1ce", "t2", "flair"]

        for pdir in patient_dirs:
            try:
                # Validate all files exist
                for mod in modalities:
                    _find_modality_file(pdir, mod)
                seg_path = _find_modality_file(pdir, "seg")

                # Load segmentation to find tumor-containing slices
                seg_nii = nib.load(str(seg_path))
                seg_data = np.asarray(seg_nii.dataobj, dtype=np.int16)

                # Get slices with tumor (non-zero labels)
                tumor_slices = []
                for z in range(seg_data.shape[2]):
                    if seg_data[:, :, z].any():
                        tumor_slices.append(z)

                if not tumor_slices:
                    continue

                if slices_per_volume is not None:
                    # Sample evenly spaced slices
                    step = max(1, len(tumor_slices) // slices_per_volume)
                    tumor_slices = tumor_slices[::step][:slices_per_volume]

                for z in tumor_slices:
                    self.samples.append((pdir, z))

            except FileNotFoundError:
                continue

        if not self.samples:
            raise FileNotFoundError(
                f"No valid samples found in {data_dir}. "
                f"Check that patient directories contain all 4 modalities and seg files."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        patient_dir, slice_idx = self.samples[idx]

        # Load all 4 modalities for this slice
        channels = []
        for modality in ["t1", "t1ce", "t2", "flair"]:
            path = _find_modality_file(patient_dir, modality)
            vol = np.asarray(nib.load(str(path)).dataobj, dtype=np.float32)
            slc = vol[:, :, slice_idx]
            channels.append(slc)

        # Load segmentation mask
        seg_path = _find_modality_file(patient_dir, "seg")
        seg_vol = np.asarray(nib.load(str(seg_path)).dataobj, dtype=np.int16)
        seg_slice = seg_vol[:, :, slice_idx]

        # Stack channels: (4, H, W)
        image = np.stack(channels, axis=0)

        # Per-channel normalization to [0, 1]
        for c in range(4):
            ch = image[c]
            ch_min, ch_max = ch.min(), ch.max()
            if ch_max - ch_min > 1e-8:
                image[c] = (ch - ch_min) / (ch_max - ch_min)
            else:
                image[c] = 0.0

        # Map BraTS labels to model classes
        mask = _map_brats_labels(seg_slice)

        # Resize to target size using bilinear for image, nearest for mask
        image_tensor = torch.from_numpy(image).unsqueeze(0)  # (1, 4, H, W)
        image_tensor = torch.nn.functional.interpolate(
            image_tensor, size=(self.img_size, self.img_size),
            mode="bilinear", align_corners=False
        ).squeeze(0)  # (4, img_size, img_size)

        mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()  # (1, 1, H, W)
        mask_tensor = torch.nn.functional.interpolate(
            mask_tensor, size=(self.img_size, self.img_size),
            mode="nearest"
        ).squeeze(0).squeeze(0).long()  # (img_size, img_size)

        return image_tensor, mask_tensor
