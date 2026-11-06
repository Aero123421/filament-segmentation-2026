"""Inference: test-time augmentation, fold ensemble and the end-to-end pipeline.

For one 2048x2048 image, :class:`FilamentPipeline` does:

1. take the central 1888x1888 crop (:mod:`filseg.preprocess`);
2. for every checkpoint and every test-time view (``d4`` = the 8 rotations /
   reflections of the square), transform the crop, run the U-Net, apply the
   sigmoid and undo the transform;
3. average the views of each checkpoint, then average the checkpoints of one
   model family, and paste the result onto the 2048x2048 canvas;
4. optionally round each family's map to uint8 and average the families in
   the uint8 domain (staged quantization, see :func:`quantized_mean_uint8`);
   with a single family this is one rounding, exactly the storage format of
   the out-of-fold maps on which the post-processing was tuned;
5. turn the probability map into disjoint instances, either with the
   mean-probability gate (:mod:`filseg.postprocess`) or with the component
   tree (:mod:`filseg.tree`), and, if enabled, refine each instance with
   the optional stage-2 refiner.

Averaging the 8 views and the fold models smooths out orientation-dependent
errors.  Every view is an exact pixel permutation, so no interpolation is
involved.  On CUDA the forward pass runs under FP16 autocast, on CPU in FP32;
the two differ only at the rounding level.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from filseg.model import load_models
from filseg.postprocess import PARAMETERS, probs_to_labels
from filseg.preprocess import crop_center, paste_to_full, prob_to_uint8

__all__ = [
    "TTA_MODES",
    "FilamentPipeline",
    "PipelineConfig",
    "Prediction",
    "apply_transform",
    "family_keys",
    "invert_transform",
    "predict_crop_probability",
    "quantized_mean_uint8",
    "tta_transforms",
]

# A view is (number of 90-degree rotations, mirror left-right after rotating).
_VIEWS: dict[str, list[tuple[int, bool]]] = {
    "none": [(0, False)],
    "hflip": [(0, False), (0, True)],
    "vflip": [(0, False), (2, True)],  # rot180 + mirror == vertical flip
    "hvflip": [(0, False), (0, True), (2, True), (2, False)],
    "d4": [(k, flip) for k in range(4) for flip in (False, True)],
}
TTA_MODES = tuple(_VIEWS)


def tta_transforms(mode: str) -> list[tuple[int, bool]]:
    """Return the ``(rot90_k, flip_lr)`` views of a TTA mode (``d4`` has 8)."""
    if mode not in _VIEWS:
        raise ValueError(f"unknown TTA mode {mode!r}; choose from {list(TTA_MODES)}")
    return list(_VIEWS[mode])


def apply_transform(array: np.ndarray, view: tuple[int, bool]) -> np.ndarray:
    """Rotate by ``k`` x 90 degrees, then optionally mirror (first two axes)."""
    k, flip = view
    out = np.rot90(array, k, axes=(0, 1)) if k else array
    return np.flip(out, axis=1) if flip else out


def invert_transform(array: np.ndarray, view: tuple[int, bool]) -> np.ndarray:
    """Exactly undo :func:`apply_transform` (mirror first, then rotate back)."""
    k, flip = view
    out = np.flip(array, axis=1) if flip else array
    return np.rot90(out, -k, axes=(0, 1)) if k else out


def _forward(model: torch.nn.Module, crop: np.ndarray, device: torch.device, precision: str) -> np.ndarray:
    """One forward pass: uint8 ``(H, W)`` image -> float32 ``(H, W)`` probability.

    The input is scaled to [0, 1] and zero-padded to a multiple of 32 (the
    U-Net's total stride); the padding is cropped off the output.
    """
    h, w = crop.shape[:2]
    chw = crop[None] if crop.ndim == 2 else crop.transpose(2, 0, 1)
    chw = np.ascontiguousarray(chw, dtype=np.float32) / 255.0
    chw = np.pad(chw, ((0, 0), (0, (-h) % 32), (0, (-w) % 32)), mode="constant")
    x = torch.from_numpy(chw[None]).to(device)
    with torch.inference_mode():
        if precision == "fp16" and device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(x)
        else:
            logits = model(x)
        return logits.float().sigmoid().cpu().numpy()[0, 0, :h, :w]


def predict_crop_probability(
    models: list[torch.nn.Module],
    crop: np.ndarray,
    tta_mode: str = "d4",
    precision: str = "fp16",
    device: torch.device | None = None,
) -> np.ndarray:
    """Mean probability of a crop over all TTA views of all models (float32)."""
    if device is None:
        device = next(models[0].parameters()).device
    views = tta_transforms(tta_mode)
    total: np.ndarray | None = None
    for model in models:
        model_total: np.ndarray | None = None
        for view in views:
            prob = _forward(model, np.ascontiguousarray(apply_transform(crop, view)), device, precision)
            prob = np.ascontiguousarray(invert_transform(prob, view))
            model_total = prob if model_total is None else model_total + prob
        model_mean = model_total / len(views)
        total = model_mean if total is None else total + model_mean
    return (total / len(models)).astype(np.float32)


def quantized_mean_uint8(arrays: list[np.ndarray]) -> np.ndarray:
    """Equal mean of uint8 probability maps, rounded once more to uint8.

    This is the canonical staged quantization of the released ensembles:
    every model family is first rounded to uint8 on its own, the uint8 maps
    are summed in uint16 and divided with ``np.rint``.  Averaging the float
    maps first and rounding once would differ by up to one uint8 step on
    boundary pixels, so the two orders are not interchangeable.
    """
    if not arrays:
        raise ValueError("at least one family map is required")
    total = np.zeros(arrays[0].shape, np.uint16)
    for array in arrays:
        if array.dtype != np.uint8 or array.shape != total.shape:
            raise ValueError("families must be identically shaped uint8 maps")
        total += array
    return np.rint(total.astype(np.float32) / len(arrays)).astype(np.uint8)


def family_keys(all_args: list[dict]) -> list[tuple]:
    """Family key of each checkpoint: (encoder, decoder, target, seed).

    Checkpoints sharing a key form one model family for staged quantization:
    the S family (ResNet34, soft targets) and the V family (EfficientNetV2-S,
    individual targets) of the combined pipeline are two families, while the
    five folds of a single recipe are one.  The order of first appearance is
    kept, so the family order follows ``pipeline_config.json``.
    """
    keys = []
    for args in all_args:
        keys.append((str(args.get("encoder", "resnet34")), str(args.get("decoder", "unet")),
                     str(args.get("target", "individual")), args.get("seed")))
    return keys


@dataclass
class PipelineConfig:
    """Settings of the released pipeline (the contents of ``pipeline_config.json``).

    Attributes:
        weights: Checkpoint file names inside ``weights/`` (one per fold model).
        postprocess: ``threshold``, ``close_radius``, ``min_area``, ``min_mean``
            of the mean-probability gate (unused for the component tree, kept
            so older configurations stay readable).
        tta: Test-time augmentation mode (``d4`` = 8 views).
        precision: ``fp16`` (autocast, CUDA only) or ``fp32``.
        quantize_uint8: Round each model family's probability to uint8 and
            average the families in the uint8 domain (staged quantization),
            as in the stored out-of-fold maps.
        weight_groups: Optional explicit families for staged quantization:
            a partition of ``weights`` into groups, one per family.  When
            omitted, families are derived from the checkpoints themselves
            (encoder, decoder, target, seed; see :func:`family_keys`).
        postprocess_method: ``"mean_gate"`` (threshold + gate) or
            ``"component_tree"`` (multi-threshold tree + LightGBM selection).
        tree_model: Tree-model JSON file name inside ``weights/`` (required
            for ``"component_tree"``); a portable text model, never a pickle.
        tree_tau: Global antichain threshold of the tree selection.
        tree_min_area: Smallest component kept as a tree node (pixels).
        tree_close_radius: Closing radius of the tree levels (0 = none).
        refiner_enabled: Apply the optional stage-2 refiner.
        refiner_weights: Refiner checkpoint names inside ``weights/refiner/``.
        refine_threshold: Probability threshold of the refiner output.
    """

    weights: list[str]
    postprocess: dict
    tta: str = "d4"
    precision: str = "fp16"
    quantize_uint8: bool = False
    weight_groups: list[list[str]] | None = None
    postprocess_method: str = "mean_gate"
    tree_model: str | None = None
    tree_tau: float = 0.26
    tree_min_area: int = 50
    tree_close_radius: int = 0
    refiner_enabled: bool = False
    refiner_weights: list[str] = field(default_factory=list)
    refine_threshold: float = 0.40

    def __post_init__(self) -> None:
        tta_transforms(self.tta)  # validates the mode
        if self.precision not in ("fp16", "fp32"):
            raise ValueError(f"precision must be 'fp16' or 'fp32', got {self.precision!r}")
        if not self.weights:
            raise ValueError("the configuration lists no weights")
        if self.refiner_enabled and not self.refiner_weights:
            raise ValueError("refiner_enabled is true but no refiner_weights are listed")
        unknown = set(self.postprocess) - set(PARAMETERS)
        if unknown:
            raise ValueError(f"unknown post-processing parameter(s) {sorted(unknown)}")
        if self.postprocess_method not in ("mean_gate", "component_tree"):
            raise ValueError(
                f"postprocess_method must be 'mean_gate' or 'component_tree', got {self.postprocess_method!r}"
            )
        if self.postprocess_method == "component_tree" and not self.tree_model:
            raise ValueError("postprocess_method is 'component_tree' but no tree_model is listed")
        if not 0.0 < self.tree_tau < 2.0:
            raise ValueError(f"tree_tau must be in (0, 2), got {self.tree_tau}")
        if self.tree_min_area < 1 or self.tree_close_radius < 0:
            raise ValueError("tree_min_area must be positive and tree_close_radius non-negative")
        if self.weight_groups is not None:
            flat = [name for group in self.weight_groups for name in group]
            if sorted(flat) != sorted(self.weights) or len(flat) != len(set(flat)):
                raise ValueError("weight_groups must partition weights (each checkpoint in exactly one family)")
            if any(not group for group in self.weight_groups):
                raise ValueError("weight_groups must not contain an empty family")

    @classmethod
    def from_json(cls, path: str | Path) -> PipelineConfig:
        """Read a ``pipeline_config.json`` file; unknown keys are an error."""
        data = json.loads(Path(path).read_text())
        allowed = set(cls.__dataclass_fields__)
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"{path}: unknown key(s) {sorted(unknown)}; allowed: {sorted(allowed)}")
        return cls(**data)


@dataclass
class Prediction:
    """Result for one image.

    Attributes:
        probability: ``(2048, 2048)`` float32 ensemble probability.
        labels: ``(2048, 2048)`` int32 instance labels (0 = background).
        infer_s: Seconds spent in the networks (TTA x ensemble).
        postprocess_s: Seconds spent turning the probability into instances.
    """

    probability: np.ndarray
    labels: np.ndarray
    infer_s: float
    postprocess_s: float


class FilamentPipeline:
    """The released pipeline: image -> probability map -> disjoint instances.

    Example::

        pipeline = FilamentPipeline.from_release("path/to/repository")
        result = pipeline.predict(read_image("image.jpeg"))
        rows = labels_to_rows("image", result.labels)
    """

    def __init__(self, config: PipelineConfig, weights_dir: str | Path, device: str | torch.device | None = None):
        self.config = config
        self.device = torch.device(device) if device is not None else _default_device()
        # FP16 autocast is a CUDA feature; CPU inference always runs in FP32.
        self.precision = config.precision if self.device.type == "cuda" else "fp32"
        weights_dir = Path(weights_dir)
        self.models, self.model_args = load_models([weights_dir / name for name in config.weights], self.device)
        # Families for staged quantization: explicit groups win; otherwise the
        # checkpoints are grouped by (encoder, decoder, target, seed).
        if config.weight_groups is not None:
            index = {name: i for i, name in enumerate(config.weights)}
            self.model_groups: list[list[torch.nn.Module]] = [
                [self.models[index[name]] for name in group] for group in config.weight_groups
            ]
            self.family_names = [f"family {i + 1} ({len(g)} models)" for i, g in enumerate(self.model_groups)]
        else:
            keys = family_keys(self.model_args)
            order: list[tuple] = []
            for key in keys:
                if key not in order:
                    order.append(key)
            self.model_groups = [[m for m, k in zip(self.models, keys, strict=True) if k == key] for key in order]
            self.family_names = [f"{key[0]}/{key[2]} (seed {key[3]})" for key in order]
        self.tree_model = None
        if config.postprocess_method == "component_tree":
            from filseg.tree import load_tree_model

            self.tree_model = load_tree_model(weights_dir / str(config.tree_model))
        self.refiner = None
        if config.refiner_enabled:
            from filseg.refine import load_refiners

            refiner_paths = [weights_dir / "refiner" / name for name in config.refiner_weights]
            self.refiner = load_refiners(refiner_paths, self.device)

    @classmethod
    def from_release(cls, root: str | Path, device: str | torch.device | None = None) -> FilamentPipeline:
        """Load ``pipeline_config.json`` and ``weights/`` from a release directory."""
        root = Path(root)
        return cls(PipelineConfig.from_json(root / "pipeline_config.json"), root / "weights", device)

    @property
    def n_views(self) -> int:
        """Number of test-time views per model."""
        return len(tta_transforms(self.config.tta))

    def warm_up(self) -> None:
        """Run each model once on a blank crop so CUDA start-up is not timed."""
        if self.device.type == "cuda":
            blank = np.zeros((1888, 1888), np.uint8)
            predict_crop_probability(self.models, blank, "none", self.precision, self.device)

    def predict_uint8(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        """Return the staged-quantized 2048x2048 uint8 probability and seconds.

        Each family's float map is rounded to uint8 first; the families are
        then averaged with :func:`quantized_mean_uint8`.  This is the exact
        operation that produced the stored ensemble maps.
        """
        crop = crop_center(image)
        _synchronize(self.device)
        start = time.perf_counter()
        family_maps = []
        for group in self.model_groups:
            prob_crop = predict_crop_probability(group, crop, self.config.tta, self.precision, self.device)
            family_maps.append(prob_to_uint8(paste_to_full(prob_crop)))
        result = quantized_mean_uint8(family_maps)
        _synchronize(self.device)
        return result, time.perf_counter() - start

    def predict_probability(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        """Return the 2048x2048 ensemble probability of one image and the seconds it took."""
        if self.config.quantize_uint8:
            quantized, seconds = self.predict_uint8(image)
            return quantized.astype(np.float32) / 255, seconds
        crop = crop_center(image)
        _synchronize(self.device)
        start = time.perf_counter()
        prob_crop = predict_crop_probability(self.models, crop, self.config.tta, self.precision, self.device)
        _synchronize(self.device)
        seconds = time.perf_counter() - start
        return paste_to_full(prob_crop), seconds

    def labels_from_probability(self, image: np.ndarray, prob: np.ndarray) -> np.ndarray:
        """Stage-1 instance labels from a probability map (no refiner)."""
        if self.config.postprocess_method == "component_tree":
            from filseg.tree import tree_to_labels

            assert self.tree_model is not None
            return tree_to_labels(
                prob, image, self.tree_model, tau=self.config.tree_tau,
                min_area=self.config.tree_min_area, close_radius=self.config.tree_close_radius,
            )
        return probs_to_labels(prob, self.config.postprocess)

    def postprocess(self, image: np.ndarray, prob: np.ndarray) -> np.ndarray:
        """Instance labels from a probability map (plus the refiner, if enabled)."""
        labels = self.labels_from_probability(image, prob)
        if self.refiner is not None:
            from filseg.refine import refine_labels

            labels, _ = refine_labels(image, prob, labels, self.refiner, threshold=self.config.refine_threshold)
        return labels

    def predict(self, image: np.ndarray) -> Prediction:
        """Run the whole pipeline on one ``(2048, 2048)`` uint8 image."""
        prob, infer_s = self.predict_probability(image)
        start = time.perf_counter()
        labels = self.postprocess(image, prob)
        return Prediction(prob, labels, infer_s, time.perf_counter() - start)


def _default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
