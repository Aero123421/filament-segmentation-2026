"""Optional stage 2: re-segment each stage-1 instance in its own crop.

Rules for pasting the refined masks back (fixed before evaluation):

- the refined probability is thresholded (default 0.40, chosen cross-fold) and
  only the refined components touching the stage-1 instance are kept;
- instances whose crop had to be scaled below ``scale_gate`` are not refined;
- an empty refinement, or one smaller than 15 % of the stage-1 instance, keeps
  the stage-1 mask (the refiner reshapes instances, it never deletes them);
- overlaps are resolved in order of decreasing refined area: a pixel goes to
  the instance with the higher refined probability, and an instance whose mask
  is at least half covered by instances placed before it is dropped.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from .crops import CROP_H, CROP_W, CropGeom, crop_geometry, extract_crop, paste_crop
from .model import build_refiner

SEED_SCORE = 0.9  # arbitration score of an instance that keeps its stage-1 mask
MIN_KEEP_FRACTION = 0.15
MAX_COVERED_FRACTION = 0.5

__all__ = ["RefinerEnsemble", "load_refiners", "refine_labels"]


class RefinerEnsemble:
    """Refiner models averaged with 4-fold flip test-time augmentation."""

    def __init__(self, models: list[torch.nn.Module], device: torch.device):
        self.models = models
        self.device = device

    @torch.inference_mode()
    def predict_crops(self, crops: np.ndarray, batch: int = 32, flips: bool = True) -> np.ndarray:
        """``(N, 4, H, W)`` uint8 crops -> ``(N, H, W)`` float32 mean probability."""
        n = len(crops)
        out = np.zeros((n, CROP_H, CROP_W), np.float32)
        if n == 0:
            return out
        variants = [(False, False)] + ([(True, False), (False, True), (True, True)] if flips else [])
        for model in self.models:
            acc = np.zeros((n, CROP_H, CROP_W), np.float32)
            for flip_h, flip_v in variants:
                for i in range(0, n, batch):
                    chunk = crops[i : i + batch].astype(np.float32) / 255.0
                    if flip_h:
                        chunk = chunk[:, :, :, ::-1]
                    if flip_v:
                        chunk = chunk[:, :, ::-1, :]
                    x = torch.from_numpy(np.ascontiguousarray(chunk)).to(self.device)
                    with torch.autocast("cuda", dtype=torch.float16, enabled=self.device.type == "cuda"):
                        logits = model(x)
                    p = logits.float().sigmoid().cpu().numpy()[:, 0]
                    if flip_v:
                        p = p[:, ::-1, :]
                    if flip_h:
                        p = p[:, :, ::-1]
                    acc[i : i + batch] += np.ascontiguousarray(p)
            out += acc / len(variants)
        return out / max(len(self.models), 1)


def load_refiners(paths: list[str | Path], device: torch.device) -> RefinerEnsemble:
    """Load refiner checkpoints (``{"model": state_dict, ...}``) into an ensemble."""
    models = []
    for path in paths:
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
        state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        model = build_refiner(pretrained=False)
        model.load_state_dict(state, strict=True)
        models.append(model.to(device).float().eval())
    return RefinerEnsemble(models, device)


def _seed_touching(refined: np.ndarray, seed_crop: np.ndarray) -> np.ndarray:
    """Keep the 8-connected components of ``refined`` that intersect the seed."""
    n, components = cv2.connectedComponents(refined.astype(np.uint8), connectivity=8)
    if n <= 1:
        return refined
    keep = np.zeros(n, bool)
    keep[np.unique(components[seed_crop > 0])] = True
    keep[0] = False
    return keep[components]


def _seed_fallback(seed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return seed, np.where(seed, SEED_SCORE, 0.0).astype(np.float32)


def _refined_mask(
    crop_prob: np.ndarray, crop: np.ndarray, geom: CropGeom, seed: np.ndarray, threshold: float, stats: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Full-size ``(mask, score)`` of one refined instance, or its stage-1 fallback."""
    refined = _seed_touching(crop_prob > threshold, crop[2] > 0)
    area_crop = int(refined.sum())
    if area_crop == 0:
        stats["n_empty_fallback"] += 1
        return _seed_fallback(seed)
    if area_crop / max(geom.scale**2, 1e-6) < MIN_KEEP_FRACTION * int(seed.sum()):
        stats["n_shrink_guarded"] += 1
        return _seed_fallback(seed)
    full_prob = paste_crop(np.where(refined, crop_prob, 0.0).astype(np.float32), geom)
    mask = full_prob > threshold
    stats["n_refined"] += 1
    return mask, np.where(mask, full_prob, 0.0).astype(np.float32)


def _arbitrate(masks: dict[int, np.ndarray], scores: dict[int, np.ndarray], shape: tuple, stats: dict) -> np.ndarray:
    """Merge possibly overlapping masks into one compact label image."""
    areas = {k: int(masks[k].sum()) for k in masks}
    priority = sorted(masks, key=lambda k: -areas[k])  # stable: ties keep insertion order
    owner = np.zeros(shape, np.int32)
    best = np.full(shape, -1.0, np.float32)
    for new_id, k in enumerate(priority, 1):
        mask = masks[k]
        if not mask.any():
            continue
        covered = float((mask & (owner > 0)).sum()) / max(int(mask.sum()), 1)
        if covered >= MAX_COVERED_FRACTION and new_id > 1:
            stats["n_absorbed"] += 1
            continue
        claim = mask & (scores[k] > best)
        owner[claim] = new_id
        best[claim] = scores[k][claim]
    if int(owner.max()) == 0:
        return owner
    present = [i for i in range(1, int(owner.max()) + 1) if (owner == i).any()]
    lookup = np.zeros(int(owner.max()) + 1, np.int32)
    lookup[present] = np.arange(1, len(present) + 1, dtype=np.int32)
    return lookup[owner]


def refine_labels(
    image: np.ndarray,
    prob: np.ndarray,
    labels: np.ndarray,
    ensemble: RefinerEnsemble,
    threshold: float = 0.40,
    scale_gate: float = 0.5,
    batch: int = 32,
    flips: bool = True,
) -> tuple[np.ndarray, dict]:
    """Refine stage-1 instance labels; returns ``(labels, counts of each rule)``."""
    n_instances = int(labels.max())
    stat_keys = ("n_refined", "n_kept_seed", "n_scale_gated", "n_shrink_guarded", "n_absorbed", "n_empty_fallback")
    stats = {key: 0 for key in stat_keys}
    stats["n_seed"] = n_instances
    if n_instances == 0:
        return labels.copy(), stats
    order, geoms, crops = [], [], []
    for k in range(1, n_instances + 1):
        seed = labels == k
        geom = crop_geometry(seed)
        if geom is None:
            continue
        if geom.scale < scale_gate:
            stats["n_scale_gated"] += 1
            continue
        order.append(k)
        geoms.append(geom)
        crops.append(extract_crop(image, prob, seed, (labels > 0) & ~seed, geom))
    masks: dict[int, np.ndarray] = {}
    scores: dict[int, np.ndarray] = {}
    if crops:
        stacked = np.stack(crops)
        crop_probs = ensemble.predict_crops(stacked, batch=batch, flips=flips)
        for j, k in enumerate(order):
            masks[k], scores[k] = _refined_mask(crop_probs[j], stacked[j], geoms[j], labels == k, threshold, stats)
    for k in range(1, n_instances + 1):
        if k not in masks:
            masks[k], scores[k] = _seed_fallback(labels == k)
            stats["n_kept_seed"] += 1
    return _arbitrate(masks, scores, labels.shape, stats), stats
