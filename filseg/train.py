"""Retrain the segmentation networks (optional: the release already ships the weights).

Recipe of the bundled checkpoints, per fold:

- **Stage 1**: 1024x1024 crops, 30,000 steps, ImageNet-initialised encoder, LR 3e-4.
- **Stage 2**: the full 1888x1888 view, 2,000 steps, LR 3e-5, initialised from the
  stage-1 checkpoint of the same fold.  The sample stream continues where stage 1
  stopped (``--sample-offset 120000`` = 30,000 steps x batch 4).
- **Both**: batch 4, AdamW (weight decay 1e-4), cosine decay of the LR to 10 % of
  its initial value, FP16 autocast with gradient scaling, loss = binary
  cross-entropy + 0.5 x soft Dice, and an exponential moving average (EMA) of the
  weights (decay 0.999, warmed up); the EMA weights are what gets saved.

Sampling: training sample ``i`` has its own random generator seeded with
``seed + 1009 * (i + sample_offset)``, so the stream does not depend on the number
of data-loader workers.  A sample picks one (image, annotation set) slot
uniformly, so images labelled by more annotators are drawn more often; centres the
crop on a random filament pixel with probability 0.7 (otherwise anywhere); applies
one of the 8 rotations/reflections; and with probability 0.5 changes contrast
(x0.85-1.15) and brightness (+-0.08).  The target is that annotator's mask
(``--target individual``) or the mean over all annotators (``--target soft``).
Images of the held-out ``--fold`` are never used for training.

Outputs in ``--out``: ``fold<k>.pt`` (FP16 parameters and FP32 BatchNorm
statistics, < 100 MB), ``oof/<stem>.npz`` (uint8 probabilities of the held-out
images, no test-time augmentation), ``history_fold<k>.json``, ``meta_fold<k>.json``.

Speed options for smaller GPUs such as a 24 GB L4 (same recipe, only the numerical
path changes): ``--channels-last`` and ``--fused-adamw`` (about 26 % faster), and
``--grad-checkpointing``, which recomputes the timm encoder blocks during the
backward pass so that stage 2 fits in 24 GB.

Usage (one fold; see README for the full commands)::

    python -m filseg.train --data prepared --out runs/stage1 --fold 0 --crop 1024 --steps 30000 --lr 3e-4
    python -m filseg.train --data prepared --out runs/stage2 --fold 0 --crop 1888 --steps 2000 --lr 3e-5 \
        --sample-offset 120000 --init-from runs/stage1/fold0.pt
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import math
import random
import time
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from filseg.infer import predict_crop_probability
from filseg.model import DECODERS, build_model
from filseg.preprocess import CROP_SIZE, paste_to_full, prob_to_uint8

POSITIVE_CROP_PROBABILITY = 0.7
DICE_WEIGHT = 0.5
WEIGHT_DECAY = 1e-4
MIN_LR_RATIO = 0.1
EMA_DECAY = 0.999
LOG_EVERY = 200
MAX_CHECKPOINT_BYTES = 100_000_000  # GitHub's per-file limit

__all__ = ["EMA", "CropDataset", "export_state", "loss_fn", "train"]


class CropDataset(Dataset):
    """Deterministic stream of augmented training crops (see the module docstring).

    Args:
        records: Manifest records of the training images.
        cache: ``stem -> (image, masks)`` arrays from :mod:`filseg.prepare`.
        crop: Crop size in pixels (at most 1888).
        target: ``"individual"`` or ``"soft"``.
        seed: Base seed of the stream.
        length: Number of samples (steps x batch).
        offset: Index of the first sample in the stream.
    """

    def __init__(
        self, records: list[dict], cache: dict, crop: int, target: str, seed: int, length: int, offset: int = 0
    ):
        if target not in ("individual", "soft"):
            raise ValueError(f"target must be 'individual' or 'soft', got {target!r}")
        if not 0 < crop <= CROP_SIZE:
            raise ValueError(f"crop must be in (0, {CROP_SIZE}], got {crop}")
        self.slots = [(r["stem"], j) for r in records for j in range(r["k"])]
        self.cache = cache
        self.crop = crop
        self.target = target
        self.seed = seed
        self.length = length
        self.offset = offset
        # Filament pixels (on a 4-px grid) that crops can be centred on.
        self.positives = {
            r["stem"]: np.column_stack(np.where(cache[r["stem"]][1].max(0)[::4, ::4] > 0)) * 4 for r in records
        }

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng(self.seed + (index + self.offset) * 1009)
        stem, annotator = self.slots[int(rng.integers(len(self.slots)))]
        image, masks = self.cache[stem]
        c = self.crop
        positives = self.positives[stem]
        if rng.random() < POSITIVE_CROP_PROBABILITY and len(positives):
            cy, cx = positives[int(rng.integers(len(positives)))]
            y = int(np.clip(cy - rng.integers(c // 4, c * 3 // 4), 0, CROP_SIZE - c))
            x = int(np.clip(cx - rng.integers(c // 4, c * 3 // 4), 0, CROP_SIZE - c))
        else:
            y, x = rng.integers(0, CROP_SIZE - c + 1, size=2)
        crop = image[y : y + c, x : x + c].astype(np.float32) / 255
        window = masks[:, y : y + c, x : x + c]
        if self.target == "individual":
            target = window[annotator].astype(np.float32)
        else:
            target = window.mean(0, dtype=np.float32)
        k = int(rng.integers(4))
        crop, target = np.rot90(crop, k), np.rot90(target, k)
        if rng.random() < 0.5:
            crop, target = crop[:, ::-1], target[:, ::-1]
        if rng.random() < 0.5:
            crop = np.clip((crop - 0.5) * rng.uniform(0.85, 1.15) + 0.5 + rng.uniform(-0.08, 0.08), 0, 1)
        return torch.from_numpy(np.ascontiguousarray(crop[None])), torch.from_numpy(np.ascontiguousarray(target[None]))


class EMA:
    """Exponential moving average of the weights, decay ``min(decay, (1 + u) / (10 + u))``."""

    def __init__(self, model: torch.nn.Module, decay: float = EMA_DECAY):
        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        self.decay = decay
        self.updates = 0

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        """Blend the current weights into the average (integer buffers are copied)."""
        self.updates += 1
        decay = min(self.decay, (1 + self.updates) / (10 + self.updates))
        source = model.state_dict()
        averaged, current = [], []
        for name, value in self.model.state_dict().items():
            if value.is_floating_point():
                averaged.append(value)
                current.append(source[name].detach())
            else:
                value.copy_(source[name].detach())
        torch._foreach_lerp_(averaged, current, 1 - decay)


def loss_fn(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy + 0.5 x soft Dice (squared denominator, works with soft targets)."""
    logits = logits.float()
    p = logits.sigmoid()
    dims = (1, 2, 3)
    dice = 1 - (2 * (p * target).sum(dims) + 1) / (p.square().sum(dims) + target.square().sum(dims) + 1)
    return F.binary_cross_entropy_with_logits(logits, target) + DICE_WEIGHT * dice.mean()


def export_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """State dict for distribution: FP16 parameters, FP32 floating buffers.

    BatchNorm running variances can exceed the FP16 range after long training,
    so floating buffers stay FP32.  Non-finite values are rejected.
    """
    parameter_names = {name for name, _ in model.named_parameters()}
    state = {}
    for name, value in model.state_dict().items():
        value = value.detach().cpu()
        if value.is_floating_point():
            value = value.half() if name in parameter_names else value.float()
        state[name] = value
    bad = [name for name, value in state.items() if value.is_floating_point() and not torch.isfinite(value).all()]
    if bad:
        raise RuntimeError(f"non-finite values in exported tensors, e.g. {bad[:3]}")
    return state


def enable_grad_checkpointing(model: torch.nn.Module) -> list[torch.nn.Module]:
    """Recompute the timm encoder blocks in the backward pass (saves activation memory).

    Returns the model's BatchNorm layers, whose running statistics must not be
    updated a second time during the recomputation (see :func:`frozen_running_stats`).
    """
    encoder = getattr(getattr(model, "encoder", None), "model", None)
    if encoder is None or not hasattr(encoder, "set_grad_checkpointing"):
        raise ValueError("--grad-checkpointing needs a timm encoder (an encoder name starting with 'tu-')")
    encoder.set_grad_checkpointing(True)
    return [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]


@contextlib.contextmanager
def frozen_running_stats(batchnorms: list[torch.nn.Module]) -> Iterator[None]:
    """Momentum 0 while backward recomputes checkpointed blocks: normalisation is
    unchanged (batch statistics), but running statistics are updated only once."""
    saved = [m.momentum for m in batchnorms]
    try:
        for m in batchnorms:
            m.momentum = 0.0
        yield
    finally:
        for m, momentum in zip(batchnorms, saved, strict=True):
            m.momentum = momentum


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_initializer(path: str, args: argparse.Namespace, model: torch.nn.Module) -> str:
    """Load the same-fold stage-1 weights into ``model``; returns the file's SHA-256."""
    initial = torch.load(path, map_location="cpu", weights_only=True)
    trained = initial.get("args", {})
    for key in ("fold", "encoder", "decoder", "seed"):
        if trained.get(key) != getattr(args, key):
            raise ValueError(
                f"--init-from was trained with {key}={trained.get(key)!r}, this run uses {getattr(args, key)!r}"
            )
    model.load_state_dict(initial["model"], strict=True)
    return _sha256(path)


def train(args: argparse.Namespace) -> dict:
    """Train one fold and write the checkpoint, out-of-fold maps and logs; returns the metadata."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    on_cuda = device.type == "cuda"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if on_cuda:
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)
    cv2.setNumThreads(0)
    torch.backends.cudnn.benchmark = True
    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records = json.loads((data / "manifest.json").read_text())
    train_records = [r for r in records if r["fold"] != args.fold]
    val_records = [r for r in records if r["fold"] == args.fold]
    if not train_records or not val_records:
        raise ValueError(f"fold {args.fold} leaves no training or no validation images")
    cache = {}
    for r in records:
        with np.load(data / f"{r['stem']}.npz") as arrays:
            cache[r["stem"]] = (arrays["image"], arrays["masks"])

    dataset = CropDataset(
        train_records, cache, args.crop, args.target, args.seed, args.steps * args.batch, args.sample_offset
    )
    loader = DataLoader(
        dataset, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=on_cuda,
        persistent_workers=args.workers > 0,
    )
    model_args = {"encoder": args.encoder, "decoder": args.decoder, "preprocess": "raw"}
    model = build_model(model_args, pretrained=not args.init_from)
    initial_sha = _load_initializer(args.init_from, args, model) if args.init_from else None
    model = model.to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    batchnorms = enable_grad_checkpointing(model) if args.grad_checkpointing else []
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY,
        **({"fused": True} if args.fused_adamw and on_cuda else {}),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=on_cuda)
    ema = EMA(model)
    model.train()
    stop_at = args.stop_after or args.steps
    if not 0 < stop_at <= args.steps:
        raise ValueError(f"--stop-after must be in (0, {args.steps}], got {args.stop_after}")
    history = []
    if on_cuda:
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for step, (x, y) in enumerate(loader, 1):
        x, y = x.to(device, non_blocking=on_cuda), y.to(device, non_blocking=on_cuda)
        if args.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        lr = args.lr * (MIN_LR_RATIO + (1 - MIN_LR_RATIO) * 0.5 * (1 + math.cos(math.pi * (step - 1) / args.steps)))
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=on_cuda):
            loss = loss_fn(model(x), y)
        scale_before = scaler.get_scale()
        with frozen_running_stats(batchnorms):
            scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= scale_before:  # the optimiser step was not skipped
            ema.update(model)
        if step == 1 or step % LOG_EVERY == 0:
            row = {
                "step": step, "loss": float(loss.detach()), "lr": lr,
                "elapsed_s": round(time.perf_counter() - start, 1),
            }
            history.append(row)
            print(json.dumps(row), flush=True)
        if step >= stop_at:
            break
    if on_cuda:
        torch.cuda.synchronize()
    train_s = time.perf_counter() - start

    state = export_state(ema.model)
    checkpoint_args = {
        **vars(args), **model_args, "variant": "ema", "checkpoint_dtype_policy": "fp16_parameters_fp32_buffers",
    }
    checkpoint = out / f"fold{args.fold}.pt"
    torch.save({"model": state, "args": checkpoint_args}, checkpoint)
    if checkpoint.stat().st_size >= MAX_CHECKPOINT_BYTES:
        raise RuntimeError(f"{checkpoint} is {checkpoint.stat().st_size} bytes, above the 100 MB limit")

    # Out-of-fold probabilities of exactly the weights that were saved.
    evaluator = build_model(checkpoint_args).to(device).eval()
    evaluator.load_state_dict(state, strict=True)
    (out / "oof").mkdir(exist_ok=True)
    for r in val_records:
        prob = predict_crop_probability([evaluator], cache[r["stem"]][0], "none", "fp16", device)
        np.savez_compressed(out / "oof" / f"{r['stem']}.npz", prob=prob_to_uint8(paste_to_full(prob)))

    meta = {
        "args": checkpoint_args,
        "train_s": train_s,
        "sec_per_step": train_s / stop_at,
        "ema_updates": ema.updates,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9 if on_cuda else 0.0,
        "train_stems": len(train_records),
        "val_stems": len(val_records),
        "gpu": torch.cuda.get_device_name() if on_cuda else "cpu",
        "torch": torch.__version__,
        "smp": smp.__version__,
        "checkpoint_sha256": _sha256(checkpoint),
        "initial_checkpoint_sha256": initial_sha,
    }
    (out / f"history_fold{args.fold}.json").write_text(json.dumps(history, indent=2) + "\n")
    (out / f"meta_fold{args.fold}.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def parser() -> argparse.ArgumentParser:
    """Command-line options (defaults = stage 1 of the released recipe)."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="folder written by `python -m filseg.prepare`")
    p.add_argument("--out", required=True, help="output folder of this stage")
    p.add_argument("--fold", type=int, required=True, help="held-out fold (0-4); the model trains on the other four")
    p.add_argument("--encoder", default="resnet34",
                   help="segmentation-models-pytorch encoder, e.g. tu-tf_efficientnetv2_s")
    p.add_argument("--decoder", default="unet", choices=sorted(DECODERS))
    p.add_argument("--target", default="individual", choices=["individual", "soft"])
    p.add_argument("--crop", type=int, default=1024, help="crop size: 1024 in stage 1, 1888 in stage 2")
    p.add_argument("--steps", type=int, default=30000, help="optimiser steps (also the cosine horizon)")
    p.add_argument("--stop-after", type=int, default=0,
                   help="stop after this many steps, keeping --steps as the cosine horizon (0 = run to --steps)")
    p.add_argument("--lr", type=float, default=3e-4, help="initial learning rate (stage 2: 3e-5)")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--seed", type=int, default=20261103)
    p.add_argument("--sample-offset", type=int, default=0, help="first sample index (stage 2: 120000)")
    p.add_argument("--init-from", default="", help="same-fold stage-1 checkpoint (stage 2)")
    p.add_argument("--workers", type=int, default=4, help="data-loader processes (do not affect the samples)")
    p.add_argument("--channels-last", action="store_true", help="NHWC memory layout (faster on recent GPUs)")
    p.add_argument("--fused-adamw", action="store_true", help="fused AdamW kernel (faster)")
    p.add_argument("--grad-checkpointing", action="store_true",
                   help="recompute encoder blocks (timm encoders) to save memory")
    return p


def main() -> None:
    """Command-line entry point: train one fold and print the metadata."""
    print(json.dumps(train(parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
