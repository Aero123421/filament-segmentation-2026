"""Network definition and checkpoint loading.

Every file in ``weights/`` is a dictionary ``{"model": state_dict, "args":
training_arguments}`` written by :mod:`filseg.train`.  The network is rebuilt
from the recorded ``encoder`` and ``decoder`` names, so other folds or another
segmentation-models-pytorch encoder can be bundled without code changes.

All released networks are U-Nets with one grayscale input channel and one
output channel (the filament logit).  Weights are stored in FP16 (BatchNorm
statistics may be FP32 to avoid overflow); they are loaded into an FP32 model,
and inference uses FP16 autocast on CUDA (see :mod:`filseg.infer`).
"""

from __future__ import annotations

from pathlib import Path

import segmentation_models_pytorch as smp
import torch

DECODER_CHANNELS = (128, 64, 32, 16, 8)
DECODERS = {"unet": smp.Unet, "unetplusplus": smp.UnetPlusPlus, "fpn": smp.FPN}

__all__ = ["DECODERS", "DECODER_CHANNELS", "build_model", "count_parameters", "load_checkpoint", "load_models"]


def build_model(args: dict, pretrained: bool = False) -> torch.nn.Module:
    """Build the network described by a checkpoint's ``args``.

    Args:
        args: Training arguments stored in the checkpoint.  Only ``encoder``
            (default ``"resnet34"``), ``decoder`` (default ``"unet"``) and
            ``preprocess`` (must be ``"raw"``: grayscale input) are used.
        pretrained: Initialise the encoder with ImageNet weights (training
            only; downloads the weights on first use).

    Returns:
        A segmentation-models-pytorch model with one input and one output channel.
    """
    encoder = str(args.get("encoder", "resnet34"))
    decoder = str(args.get("decoder", "unet"))
    preprocess = str(args.get("preprocess", "raw"))
    if preprocess != "raw":
        raise ValueError(f"unsupported preprocess mode {preprocess!r}: this release feeds raw grayscale images only")
    if decoder not in DECODERS:
        raise ValueError(f"unknown decoder {decoder!r}; expected one of {sorted(DECODERS)}")
    kwargs: dict = {
        "encoder_name": encoder,
        "encoder_weights": "imagenet" if pretrained else None,
        "in_channels": 1,
        "classes": 1,
    }
    if decoder in ("unet", "unetplusplus"):
        kwargs["decoder_channels"] = DECODER_CHANNELS
    return DECODERS[decoder](**kwargs)


def count_parameters(model: torch.nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_checkpoint(path: str | Path) -> dict:
    """Load a release checkpoint safely (tensors and plain Python values only).

    Raises:
        ValueError: If the file is not a ``{"model", "args"}`` dictionary.
    """
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"{path}: expected a checkpoint dict with 'model' and 'args' keys")
    checkpoint.setdefault("args", {})
    return checkpoint


def load_models(paths: list[str | Path], device: torch.device) -> tuple[list[torch.nn.Module], list[dict]]:
    """Load checkpoints into eval-mode FP32 models on ``device``.

    Returns:
        ``(models, args)`` where ``args[i]`` is the training-argument dict of
        ``models[i]``.
    """
    if not paths:
        raise ValueError("at least one checkpoint is required")
    models: list[torch.nn.Module] = []
    all_args: list[dict] = []
    for path in paths:
        checkpoint = load_checkpoint(path)
        args = dict(checkpoint["args"])
        model = build_model(args)
        model.load_state_dict(checkpoint["model"], strict=True)
        models.append(model.to(device).float().eval())
        all_args.append(args)
    return models, all_args
