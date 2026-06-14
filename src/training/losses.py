"""
Loss function and metric for solar panel segmentation.

We use Dice loss because the dataset has severe class imbalance (only ~1.85%
of pixels are panel on average). Cross-entropy would let the model "cheat"
by predicting all-background. Dice forces the model to actually overlap
predictions with ground truth, ignoring easy background pixels.

IoU (Jaccard) is the metric we report. It measures overlap quality the
same way Dice does, but it's not used to drive training (not smoothly
differentiable when thresholded).
"""

import torch
import segmentation_models_pytorch as smp


def build_loss() -> torch.nn.Module:
    """Dice loss for binary segmentation.

    smp.losses.DiceLoss applies sigmoid internally and computes
    1 - (2 * intersection) / (predicted + truth), summed over pixels.

    mode="binary": single output channel (panel vs background)
    from_logits=True: model outputs raw logits, loss applies sigmoid itself
    """
    return smp.losses.DiceLoss(mode="binary", from_logits=True)


def compute_iou(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """Compute IoU for one batch.

    Args:
        logits: raw model output, shape (B, 1, H, W)
        target: ground truth mask, shape (B, H, W) with values in {0, 1}
        threshold: cutoff for binarizing predicted probabilities

    Returns:
        Mean IoU across the batch, as a Python float.
    """
    # logits -> probabilities -> binary predictions
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).long()  # shape (B, 1, H, W)

    # drop the channel dim to match target shape
    preds = preds.squeeze(1)  # (B, H, W)
    target = target.long()

    # compute per-batch IoU using smp's stats helper
    tp, fp, fn, tn = smp.metrics.get_stats(
        preds, target, mode="binary"
    )
    iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
    return iou.item()