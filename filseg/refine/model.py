"""Refiner network: a small U-Net on the 4-channel instance crop."""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch


def build_refiner(pretrained: bool = False, encoder: str = "resnet34") -> torch.nn.Module:
    """U-Net (4 input channels, 1 output logit); ImageNet encoder weights only for training."""
    return smp.Unet(
        encoder_name=encoder,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=4,
        classes=1,
        decoder_channels=(128, 64, 32, 16, 8),
        activation=None,
    )
