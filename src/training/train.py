"""
Training script for solar panel segmentation.

Designed to run on Kaggle GPU. Locally we can run a few steps to verify
the pipeline works, but full training needs a GPU.
"""

import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from src.data.dataset import SolarPanelDataset
from src.models.unet import build_unet
from src.training.losses import build_loss, compute_iou


def train(
        data_dir: str | Path,
        epochs: int = 30,
        batch_size: int = 8,
        lr: float = 1e-3,
        val_split: float = 0.15,
        device: str = "auto",
        num_workers: int = 0,
):
    """Train the U-Net model.

    Args:
        data_dir: path to the folder containing img/ and mask/ subdirectories
        epochs: number of full passes through the training data
        batch_size: samples per batch
        lr: peak learning rate for OneCycleLR scheduler
        val_split: fraction of data to hold out for validation
        device: "auto" picks GPU if available, else CPU
        num_workers: DataLoader worker processes (0 = main process only, safe on Windows)
    """

    # --- device setup ---
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    # --- dataset and split ---
    img_dir = Path(data_dir) / "img"
    mask_dir = Path(data_dir) / "mask"

    full_dataset = SolarPanelDataset(img_dir, mask_dir)
    n_total = len(full_dataset)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val

    train_dataset, val_dataset = random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"Dataset split: {n_train} train, {n_val} val")

    # --- dataloaders ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )

    # --- model, loss, optimizer ---
    model = build_unet().to(device)
    loss_fn = build_loss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # OneCycleLR: warms up LR then anneals it down over training
    # improves convergence vs a fixed learning rate
    scheduler = OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
    )

    # --- training loop ---
    best_val_iou = 0.0

    for epoch in range(epochs):
        epoch_start = time.time()

        # --- train phase ---
        model.train()
        train_loss = 0.0

        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)

            # zero gradients from previous batch
            optimizer.zero_grad()

            # forward pass
            logits = model(images)

            # squeeze channel dim from logits to match mask shape
            loss = loss_fn(logits.squeeze(1), masks)

            # backward pass
            loss.backward()

            # update weights
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # --- validation phase ---
        model.eval()
        val_loss = 0.0
        val_iou = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                logits = model(images)
                loss = loss_fn(logits.squeeze(1), masks)

                val_loss += loss.item()
                val_iou += compute_iou(logits, masks)

        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)

        epoch_time = time.time() - epoch_start

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"train loss: {avg_train_loss:.4f} | "
            f"val loss: {avg_val_loss:.4f} | "
            f"val IoU: {avg_val_iou:.4f} | "
            f"time: {epoch_time:.1f}s"
        )

        # save best model
        if avg_val_iou > best_val_iou:
            best_val_iou = avg_val_iou
            torch.save(model.state_dict(), "models/best_model.pth")
            print(f"  → new best model saved (IoU: {best_val_iou:.4f})")

    print(f"\nTraining complete. Best val IoU: {best_val_iou:.4f}")
    return best_val_iou


if __name__ == "__main__":
    from pathlib import Path

    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data" / "raw" / "bdappv" / "google"

    print(f"Looking for data in: {data_dir}")
    print(f"Data dir exists: {data_dir.exists()}")

    train(
        data_dir=data_dir,
        epochs=2,
        batch_size=2,
        device="auto",
    )