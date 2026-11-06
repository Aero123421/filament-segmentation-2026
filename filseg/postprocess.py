"""Post-processing: probability map -> disjoint filament instances.

Four steps, with the parameters from ``pipeline_config.json``:

1. **Threshold**: a pixel is foreground if ``p > threshold`` (strict).
2. **Closing** (optional): a morphological closing with a disk of radius
   ``close_radius`` pixels bridges small gaps along a filament, so one
   filament is not split into several instances.  ``0`` disables it.
3. **Candidates**: every 8-connected foreground component is one candidate
   instance.
4. **Mean-probability gate**: a candidate is kept only if its area is at
   least ``min_area`` pixels and the mean probability inside it is at least
   ``min_mean``.  Most false positives are faint, diffuse blobs with a low
   mean probability, while real filaments have a confident core.

Every pixel receives at most one label, so instances can never overlap.  The
four parameters were tuned on out-of-fold predictions (README, "Results").
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import find_objects

PARAMETERS = ("threshold", "close_radius", "min_area", "min_mean")

__all__ = ["PARAMETERS", "PostprocessTrace", "close_binary", "probs_to_labels", "trace_postprocess"]


@dataclass
class PostprocessTrace:
    """Intermediate results of the post-processing, for inspection and plots.

    Attributes:
        binary: ``p > threshold`` (uint8, 0/1).
        closed: ``binary`` after the closing (the same array if it is disabled).
        candidates: int32 component labels of ``closed`` (0 = background).
        areas: Pixel area of candidate ``i + 1`` at index ``i``.
        mean_probs: Mean probability of each candidate (NaN when it was not
            needed because the area test already failed).
        kept: Whether each candidate passed the gate.
        labels: Final int32 instance labels ``1..N`` (0 = background).
    """

    binary: np.ndarray
    closed: np.ndarray
    candidates: np.ndarray
    areas: np.ndarray
    mean_probs: np.ndarray
    kept: np.ndarray
    labels: np.ndarray


def _check_params(params: dict) -> tuple[float, int, int, float | None]:
    unknown = set(params) - set(PARAMETERS)
    if unknown:
        raise ValueError(f"unknown post-processing parameter(s) {sorted(unknown)}; supported: {list(PARAMETERS)}")
    if "threshold" not in params:
        raise ValueError("post-processing needs a 'threshold'")
    threshold = float(params["threshold"])
    close_radius = int(params.get("close_radius", 0))
    min_area = int(params.get("min_area", 0))
    min_mean = float(params["min_mean"]) if params.get("min_mean") is not None else None
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    if close_radius < 0 or min_area < 0:
        raise ValueError("close_radius and min_area must be non-negative")
    return threshold, close_radius, min_area, min_mean


def _check_prob(prob: np.ndarray) -> np.ndarray:
    p = np.asarray(prob)
    if p.ndim != 2 or not np.issubdtype(p.dtype, np.floating):
        raise ValueError("prob must be a 2D float array in [0, 1] (decode stored uint8 maps with q / 255)")
    if not np.isfinite(p).all() or p.min(initial=0.0) < 0 or p.max(initial=0.0) > 1:
        raise ValueError("prob contains non-finite values or values outside [0, 1]")
    return np.ascontiguousarray(p, dtype=np.float32)


def close_binary(binary: np.ndarray, radius: int) -> np.ndarray:
    """Morphological closing of a uint8 binary image with a disk of ``radius`` px (0 = no-op)."""
    if radius <= 0:
        return binary
    size = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)


def trace_postprocess(prob: np.ndarray, params: dict) -> PostprocessTrace:
    """Run the post-processing and keep every intermediate result.

    Args:
        prob: ``(H, W)`` float probability map with values in [0, 1].
        params: ``threshold`` (required), ``close_radius``, ``min_area`` and
            ``min_mean`` (optional; a missing ``min_mean`` disables the gate).

    Returns:
        A :class:`PostprocessTrace`; ``trace.labels`` is the final result.
    """
    threshold, close_radius, min_area, min_mean = _check_params(params)
    p = _check_prob(prob)
    binary = (p > threshold).astype(np.uint8)
    closed = close_binary(binary, close_radius)
    n, candidates, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.int64)
    kept = areas >= min_area
    mean_probs = np.full(len(areas), np.nan)
    if min_mean is not None and kept.any():
        for index, window in enumerate(find_objects(candidates, max_label=n - 1)):
            if window is None or not kept[index]:
                continue
            mean_probs[index] = float(p[window][candidates[window] == index + 1].mean())
            kept[index] = mean_probs[index] >= min_mean
    if n <= 1:
        labels = np.zeros(p.shape, dtype=np.int32)
    else:
        # Kept candidates are renumbered 1..N in their original (raster) order.
        lookup = np.r_[0, np.cumsum(kept) * kept].astype(np.int32)
        labels = lookup[candidates]
    return PostprocessTrace(binary, closed, candidates, areas, mean_probs, kept, labels)


def probs_to_labels(prob: np.ndarray, params: dict) -> np.ndarray:
    """Return disjoint int32 instance labels (0 = background) for a probability map."""
    return trace_postprocess(prob, params).labels
