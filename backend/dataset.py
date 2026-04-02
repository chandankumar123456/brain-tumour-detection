"""
LGG MRI Segmentation Dataset Loader for Multi-Path Fusion Network.
Loads PNG/TIF image-mask pairs from the mateuszbuda/lgg-mri-segmentation dataset.

Dataset structure:
  kaggle_3m/
    TCGA_CS_4941_19960909/
      TCGA_CS_4941_19960909_1.tif       (MRI image)
      TCGA_CS_4941_19960909_1_mask.tif  (binary mask)
      ...

Usage:
  dataset = LGGDataset(data_dir="/path/to/kaggle_3m")
  img, mask = dataset[0]   # img: (1, 256, 256), mask: (256, 256)
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


IMG_SIZE = 256


def find_lgg_data_dir(base_path: str | Path) -> Path:
    """
    Locate the directory containing patient subdirectories in the LGG dataset.
    Handles the kagglehub download structure where data may be nested.
    """
    base = Path(base_path)
    # The dataset typically extracts to a 'kaggle_3m' subdirectory
    candidates = [
        base / "kaggle_3m",
        base / "lgg-mri-segmentation" / "kaggle_3m",
        base,
    ]
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        # Check if this directory contains patient subdirectories with .tif files
        for child in candidate.iterdir():
            if child.is_dir() and list(child.glob("*_mask.tif")):
                return candidate
    raise FileNotFoundError(
        f"Cannot find LGG dataset (patient directories with *_mask.tif) in {base_path}."
    )


class LGGDataset(Dataset):
    """
    Dataset loader for LGG MRI Segmentation data.

    Expects directory structure:
      data_dir/
        TCGA_XX_XXXX/
          TCGA_XX_XXXX_N.tif        (MRI image)
          TCGA_XX_XXXX_N_mask.tif   (binary tumor mask)

    Each __getitem__ returns:
      image: (1, 256, 256) float32 tensor, normalized to [0, 1]
      mask:  (256, 256) int64 tensor with classes {0, 1}

    Args:
        data_dir: Path to LGG data directory containing patient subdirectories.
        img_size: Target image size (default 256).
        patient_indices: Optional list of patient directory indices to use
            (for train/val splitting). If None, uses all patients.
    """

    def __init__(
        self,
        data_dir: str,
        img_size: int = IMG_SIZE,
        patient_indices: list[int] | None = None,
    ):
        self.img_size = img_size
        self.data_dir = Path(data_dir)

        if not self.data_dir.exists():
            raise FileNotFoundError(f"LGG data directory not found: {data_dir}")

        # Discover patient directories
        all_patient_dirs = sorted([
            d for d in self.data_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ])

        if not all_patient_dirs:
            raise FileNotFoundError(
                f"No patient directories found in {data_dir}."
            )

        # Filter to requested patient indices
        if patient_indices is not None:
            if any(i >= len(all_patient_dirs) for i in patient_indices):
                raise IndexError(
                    f"patient_indices contains values >= {len(all_patient_dirs)} "
                    f"(number of patient directories in {data_dir})"
                )
            patient_dirs = [all_patient_dirs[i] for i in patient_indices]
        else:
            patient_dirs = all_patient_dirs

        # Build index of (image_path, mask_path) pairs
        self.samples: list[tuple[Path, Path]] = []

        for pdir in patient_dirs:
            mask_files = sorted(pdir.glob("*_mask.tif"))
            for mask_path in mask_files:
                # Derive image path from mask path: remove '_mask' suffix
                image_name = mask_path.name.replace("_mask.tif", ".tif")
                image_path = mask_path.parent / image_name
                if image_path.exists():
                    self.samples.append((image_path, mask_path))

        if not self.samples:
            raise FileNotFoundError(
                f"No valid image-mask pairs found in {data_dir}. "
                f"Expected *_mask.tif files alongside corresponding *.tif images."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.samples[idx]

        # Load image as grayscale, resize, normalize to [0, 1]
        img = Image.open(image_path).convert("L")
        img = img.resize((self.img_size, self.img_size), Image.LANCZOS)
        img_arr = np.array(img, dtype=np.float32)

        img_min, img_max = img_arr.min(), img_arr.max()
        if img_max - img_min > 1e-8:
            img_arr = (img_arr - img_min) / (img_max - img_min)
        else:
            img_arr = np.zeros_like(img_arr)

        # Shape: (1, H, W)
        image_tensor = torch.from_numpy(img_arr).unsqueeze(0)

        # Load mask, resize, convert to binary {0, 1}
        mask = Image.open(mask_path).convert("L")
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
        mask_arr = np.array(mask, dtype=np.float32)

        # Binarize: any non-zero pixel is tumor (class 1)
        mask_arr = (mask_arr > 0).astype(np.int64)
        mask_tensor = torch.from_numpy(mask_arr)  # (H, W)

        return image_tensor, mask_tensor
