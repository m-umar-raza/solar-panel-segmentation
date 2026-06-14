"""
U-Net model for solar panel segmentation.

We use segmentation-models-pytorch (smp), which provides U-Net plus many
encoder options. ResNet34 pretrained on ImageNet is a solid default:
small enough to train fast on Kaggle GPU, large enough for our task.
"""

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


def build_unet(
        encoder_name: str = "resnet34",
        encoder_weights: str = "imagenet",
        in_channels: int = 3,
        classes: int = 1,
) -> nn.Module:
    """Build a U-Net with a pretrained encoder.

    Args:
        encoder_name: backbone for the encoder, e.g. "resnet34", "resnet50", "efficientnet-b0".
        encoder_weights: "imagenet" for pretrained, None for random init.
        in_channels: 3 for RGB images.
        classes: number of output channels. 1 for binary segmentation.

    Returns:
        A PyTorch nn.Module ready for training. Output is raw logits
        (no sigmoid applied), shape (B, classes, H, W).
    """
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
    )
    return model