"""
PyTorch Dataset for the BDAPPV solar panel segmentation dataset.

Each sample is (image, mask):
  image: float tensor of shape (3, H, W), values in [0, 1]
  mask:  float tensor of shape (H, W),    values in {0, 1}
"""

from pathlib import Path
from typing import Optional, Callable

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class SolarPanelDataset(Dataset):
    """Loads paired aerial images and segmentation masks from disk."""

    def __init__(
            self,
            image_dir: str | Path,
            mask_dir: str | Path,
            transform: Optional[Callable] = None,
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform

        # only keep filenames that exist in both folders
        mask_files = set(p.name for p in self.mask_dir.glob("*.png"))
        image_files = set(p.name for p in self.image_dir.glob("*.png"))
        self.filenames = sorted(mask_files & image_files)

        if len(self.filenames) == 0:
            raise RuntimeError(f"No matching files found in {image_dir} and {mask_dir}")

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        filename = self.filenames[idx]

        # load and convert image to RGB (some are palette mode)
        image = Image.open(self.image_dir / filename).convert("RGB")
        image = np.array(image, dtype=np.float32) / 255.0  # normalize to [0, 1]

        # load mask as grayscale, binarize to {0, 1}
        mask = Image.open(self.mask_dir / filename)
        mask = np.array(mask, dtype=np.float32)
        mask = (mask > 127).astype(np.float32)

        # apply augmentations if provided (albumentations expects HWC numpy)
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # convert to torch tensors with channel-first layout for image
        # (H, W, C) -> (C, H, W) via permute
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1)
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)

        return image, mask