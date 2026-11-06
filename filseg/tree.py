"""Component tree: multi-threshold candidates scored by a LightGBM model.

Stage 1 of the combined pipeline.  Instead of one threshold, the probability
map is thresholded at 15 levels (``DEFAULT_LEVELS``); the 8-connected
components at successive levels are nested (closing is disabled, so nesting
is exact), forming a forest.  Every node is described by 43 probability,
shape and local-contrast features (:data:`TREE_FEATURE_NAMES`).  A LightGBM
regressor, distributed as portable JSON holding the text model (never a
pickle), predicts each node's expected IoU contribution ``u``.  An exact
dynamic program then selects the maximum-weight set of pairwise disjoint
nodes (an antichain) for a global threshold ``tau``::

    maximise  sum(u_i) - tau * (number selected)

The selected nodes, rendered in tree order, are the stage-1 instances.
The deployment model was refitted on all 707 training stems; the number of
leaves and ``tau`` are recorded in ``pipeline_config.json`` so a retrained
tree can be swapped in by editing that file and replacing the model file.

This module is self-contained (NumPy / OpenCV / SciPy / LightGBM only) and
reproduces, operation for operation, the probabilities-to-labels path that
produced the reference submission from the stored uint8 probability maps.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import find_objects

from filseg.postprocess import close_binary

DEFAULT_LEVELS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                  0.55, 0.60, 0.65, 0.70, 0.80)

FEATURE_NAMES = [
    "log_area", "mean_p", "max_p", "p50", "p75", "p90", "p95",
    "std_p", "core_area_60", "core_area_75", "core_area_90",
    "core_fraction_60", "core_fraction_75", "core_fraction_90",
    "elongation", "extent", "log_width", "radius", "radius_max",
]
GATE_FEATURE_NAMES = FEATURE_NAMES + [
    "solidity", "compactness", "perimeter_area", "core_mean_p",
    "intensity_mean", "intensity_std", "intensity_p10", "core_intensity",
    "ring_contrast", "ring_dark_fraction", "ring_std",
]
TREE_FEATURE_NAMES = GATE_FEATURE_NAMES + [
    "threshold", "parent_area_ratio", "parent_mean_delta", "child_count",
    "child_area_fraction", "largest_child_fraction", "child_mean_max",
    "sibling_count", "root_area_ratio", "image_mean", "image_std",
    "image_p10", "image_p90",
]

__all__ = [
    "DEFAULT_LEVELS",
    "GATE_FEATURE_NAMES",
    "TREE_FEATURE_NAMES",
    "ComponentTree",
    "build_component_tree",
    "component_features",
    "gate_features",
    "load_tree_model",
    "predict_utility",
    "select_antichain",
    "tree_to_labels",
    "validate_parent",
]


def component_features(prob: np.ndarray, labels: np.ndarray, stats=None):
    """Descriptors in ascending positive-label order, plus areas."""
    n = int(labels.max())
    areas = np.bincount(labels.ravel(), minlength=n + 1)[1:]
    if n == 0:
        return np.empty((0, len(FEATURE_NAMES)), np.float32), areas
    if stats is None:
        slices = find_objects(labels, max_label=n)
    else:
        slices = [
            (slice(int(y), int(y + h)), slice(int(x), int(x + w)))
            for x, y, w, h, _ in stats[1:]
        ]
    hh, ww = prob.shape
    cy0, cx0 = (hh - 1) / 2, (ww - 1) / 2
    radius_scale = min(hh, ww) / 2
    rows = []
    for k, sl in enumerate(slices, 1):
        if sl is None:
            raise ValueError("Labels must be contiguous positive integers")
        mask = labels[sl] == k
        yy, xx = np.nonzero(mask)
        values = prob[sl][mask]
        area = len(values)
        p50, p75, p90, p95 = np.quantile(values, [0.5, 0.75, 0.9, 0.95])
        core = [int(np.count_nonzero(values > t)) for t in (0.6, 0.75, 0.9)]
        x = xx.astype(np.float64)
        y = yy.astype(np.float64)
        x -= x.mean()
        y -= y.mean()
        cov = np.array([[np.mean(x * x), np.mean(x * y)],
                        [np.mean(x * y), np.mean(y * y)]])
        eig = np.maximum(np.linalg.eigvalsh(cov), 1 / 12)
        elongation = float(np.sqrt(eig[1] / eig[0]))
        width = float(np.sqrt(12 * eig[0]))
        radius = np.hypot(xx + sl[1].start - cx0,
                          yy + sl[0].start - cy0) / radius_scale
        rows.append([
            np.log1p(area), values.mean(), values.max(), p50, p75, p90, p95,
            values.std(), *core, *[a / area for a in core], elongation,
            area / mask.size, np.log1p(width), radius.mean(), radius.max(),
        ])
    return np.asarray(rows, np.float32), areas


def gate_features(prob: np.ndarray, labels: np.ndarray, image: np.ndarray,
                  stats=None, base=None) -> tuple[np.ndarray, np.ndarray]:
    """Probability, shape and local H-alpha contrast, without any GT input."""
    img = np.asarray(image)
    if img.shape != prob.shape or img.ndim != 2 or not np.isfinite(img).all():
        raise ValueError("image must be finite grayscale with prob.shape")
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255
    else:
        img = img.astype(np.float32)
        if np.any(img < 0) or np.any(img > 1):
            raise ValueError("float image must be in [0,1]")
    features, areas = component_features(prob, labels, stats) if base is None else base
    n = len(areas)
    if n == 0:
        return np.empty((0, len(GATE_FEATURE_NAMES)), np.float32), areas
    objects = find_objects(labels, max_label=n)
    extra = []
    for k, sl in enumerate(objects, 1):
        if sl is None:
            raise ValueError("Labels must be contiguous positive integers")
        y = slice(max(0, sl[0].start - 13), min(labels.shape[0], sl[0].stop + 13))
        x = slice(max(0, sl[1].start - 13), min(labels.shape[1], sl[1].stop + 13))
        mask = labels[y, x] == k
        yy, xx = np.nonzero(mask)
        hull = cv2.convexHull(np.column_stack((xx, yy)).astype(np.int32))
        canvas = np.zeros(mask.shape, np.uint8)
        cv2.fillConvexPoly(canvas, hull, 1)
        solidity = len(yy) / max(int(canvas.sum()), len(yy))
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST,
                                       cv2.CHAIN_APPROX_SIMPLE)
        perimeter = sum(cv2.arcLength(c, True) for c in contours)
        values = img[y, x][mask]
        high = mask & (prob[y, x] > .75)
        pvalues = prob[y, x][high]
        core_mean = float(pvalues.mean()) if len(pvalues) else float(prob[y, x][mask].max())
        distance = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
        ring = (distance > 3) & (distance <= 12) & (labels[y, x] == 0)
        rv = img[y, x][ring]
        rm = float(np.median(rv)) if len(rv) else float(values.mean())
        rs = float(rv.std()) if len(rv) else 0.0
        extra.append([
            solidity, 4 * np.pi * len(yy) / max(perimeter ** 2, 1), perimeter / len(yy), core_mean,
            values.mean(), values.std(), np.quantile(values, .1),
            float(img[y, x][high].mean()) if high.any() else float(values.mean()),
            np.clip((rm - float(values.mean())) / max(rs, .02), -10, 10),
            np.mean(values < rm - max(rs, .02)), rs,
        ])
    return np.column_stack((features, np.asarray(extra, np.float32))), areas


@dataclass
class ComponentTree:
    """A laminar candidate forest; node masks are stored as bbox crops."""

    shape: tuple[int, int]
    parent: np.ndarray
    level: np.ndarray
    features: np.ndarray
    areas: np.ndarray
    instances: list[tuple[int, int, np.ndarray]]

    def labels(self, selected) -> np.ndarray:
        """Render selected nodes, rejecting overlaps instead of overwriting."""
        out = np.zeros(self.shape, np.int32)
        for k, index in enumerate(np.asarray(selected, dtype=int), 1):
            y, x, mask = self.instances[index]
            view = out[y:y + mask.shape[0], x:x + mask.shape[1]]
            if np.any(view[mask]):
                raise ValueError("selected nodes are not a pixel-disjoint antichain")
            view[mask] = k
        return out


def validate_parent(parent):
    p = np.asarray(parent)
    if p.ndim != 1 or not np.issubdtype(p.dtype, np.integer):
        raise ValueError("parent must be a 1D integer array")
    if np.any(p < -1) or np.any(p >= np.arange(len(p))):
        raise ValueError("each parent must be -1 or precede its child")
    return p


def select_antichain(parent, utility, tau=0.22, cost=None):
    """Exact O(nodes) DP, including the empty antichain.

    Equal objective values prefer descendants, and zero gains select nothing.
    The deployment score is utility=u_hat, cost=1, a global tau for all images.
    """
    p = validate_parent(parent)
    u = np.asarray(utility, np.float64)
    c = np.ones(len(p)) if cost is None else np.asarray(cost, np.float64)
    if u.shape != p.shape or c.shape != p.shape or not np.isfinite(u).all() or not np.isfinite(c).all():
        raise ValueError("utility/cost must be finite vectors with parent.shape")
    if not np.isfinite(tau) or np.any(c <= 0):
        raise ValueError("tau must be finite and costs positive")
    children_best = np.zeros(len(p), np.float64)
    take = np.zeros(len(p), bool)
    for i in range(len(p) - 1, -1, -1):
        own = u[i] - float(tau) * c[i]
        take[i] = own > max(0., children_best[i])
        best = max(0., own, children_best[i])
        if p[i] >= 0:
            children_best[p[i]] += best
    selected = []
    blocked = np.zeros(len(p), bool)
    for i in range(len(p)):
        if p[i] >= 0:
            blocked[i] = blocked[p[i]] or take[p[i]]
        if take[i] and not blocked[i]:
            selected.append(i)
    return np.asarray(selected, np.int32)


def _probability(prob):
    p = np.asarray(prob)
    if p.ndim != 2 or not np.issubdtype(p.dtype, np.floating):
        raise ValueError("prob must be a 2D float array in [0,1]; decode uint8 /255")
    if not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1):
        raise ValueError("prob contains nonfinite values or values outside [0,1]")
    return np.ascontiguousarray(p, dtype=np.float32)


def build_component_tree(prob, image, levels=DEFAULT_LEVELS, min_area=50,
                         close_radius=0) -> ComponentTree:
    """Build a laminar candidate forest; keep crop masks, never N full maps."""
    prob = _probability(prob)
    if image is None:
        raise ValueError("component_tree requires the grayscale source image")
    img = np.asarray(image)
    if img.ndim != 2 or img.shape != prob.shape or not np.isfinite(img).all():
        raise ValueError("image must be finite grayscale with prob.shape")
    img = img.astype(np.float32) / 255 if img.dtype == np.uint8 else img.astype(np.float32)
    if np.any(img < 0) or np.any(img > 1):
        raise ValueError("float image must be in [0,1]")
    levels = np.asarray(levels, float)
    bad_levels = (levels.ndim != 1 or not len(levels) or np.any(np.diff(levels) <= 0)
                  or not np.isfinite(levels).all() or np.any(levels < 0) or np.any(levels > 1))
    if bad_levels:
        raise ValueError("levels must be strictly increasing within [0,1]")
    if int(min_area) != min_area or min_area < 1:
        raise ValueError("min_area must be a positive integer")
    if int(close_radius) != close_radius or close_radius < 0:
        raise ValueError("close_radius must be a nonnegative integer")
    yy, xx = np.ogrid[:prob.shape[0], :prob.shape[1]]
    disk = ((yy - (prob.shape[0] - 1) / 2) ** 2 + (xx - (prob.shape[1] - 1) / 2) ** 2
            < (.88 * min(prob.shape) / 2) ** 2)
    vals = img[disk]
    image_stats = [vals.mean(), vals.std(), *np.quantile(vals, [.1, .9])] if len(vals) else [0] * 4
    parents, node_levels, rows, areas, instances = [], [], [], [], []
    previous = np.zeros(prob.shape, np.int32)
    for li, t in enumerate(levels):
        binary = close_binary((prob > float(t)).astype(np.uint8), int(close_radius))
        _, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        valid = np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] >= min_area) + 1
        lut = np.zeros(len(stats), np.int32)
        lut[valid] = np.arange(1, len(valid) + 1)
        labels = lut[labels]
        compact_stats = stats[np.r_[0, valid]]
        f, a = gate_features(prob, labels, img, compact_stats)
        offset = len(parents)
        for j, (x, y, w, h, _area) in enumerate(compact_stats[1:]):
            x, y, w, h = map(int, (x, y, w, h))
            mask = labels[y:y + h, x:x + w] == j + 1
            parent = -1
            if li:
                pv = previous[y:y + h, x:x + w][mask]
                if pv.min() <= 0 or pv.min() != pv.max():
                    raise ValueError("morphology did not preserve component nesting")
                parent = int(pv[0]) - 1
            parents.append(parent)
            node_levels.append(li)
            instances.append((y, x, mask))
        rows.extend(f)
        areas.extend(a)
        previous = np.where(labels > 0, labels + offset, 0).astype(np.int32)
    parent = np.asarray(parents, np.int32)
    level = np.asarray(node_levels, np.int16)
    area = np.asarray(areas, np.int64)
    base = np.asarray(rows, np.float32).reshape(-1, len(GATE_FEATURE_NAMES))
    extra = np.zeros((len(parent), len(TREE_FEATURE_NAMES) - len(GATE_FEATURE_NAMES)), np.float32)
    nchild = np.zeros(len(parent), np.int32)
    child_area, largest, child_mean = (np.zeros(len(parent)) for _ in range(3))
    roots = np.empty(len(parent), np.int32)
    for i, par in enumerate(parent):
        roots[i] = i if par < 0 else roots[par]
        if par >= 0:
            nchild[par] += 1
            child_area[par] += area[i]
            largest[par] = max(largest[par], area[i])
            child_mean[par] = max(child_mean[par], base[i, 1])
    for i, par in enumerate(parent):
        extra[i] = [levels[level[i]], area[i] / area[par] if par >= 0 else 1,
                    base[i, 1] - base[par, 1] if par >= 0 else 0,
                    nchild[i], child_area[i] / area[i], largest[i] / area[i],
                    child_mean[i], nchild[par] - 1 if par >= 0 else 0,
                    area[i] / area[roots[i]], *image_stats]
    return ComponentTree(prob.shape, parent, level, np.c_[base, extra], area, instances)


@lru_cache(maxsize=48)
def _booster(model_str):
    import lightgbm as lgb
    return lgb.Booster(model_str=model_str)


def load_tree_model(path: str | Path) -> dict:
    """Load a portable tree-model JSON (LightGBM text model, no pickle)."""
    model = json.loads(Path(path).read_text())
    if model.get("feature_names") != TREE_FEATURE_NAMES:
        raise ValueError(f"{path}: tree model feature names/order do not match")
    if model.get("kind") == "bagged_exact_direct":
        members = model.get("members")
        if not isinstance(members, list) or not members:
            raise ValueError(f"{path}: bagged tree model needs a non-empty members list")
        for member in members:
            if member.get("feature_names") != TREE_FEATURE_NAMES or "u_model" not in member:
                raise ValueError(f"{path}: bagged tree member needs feature_names and u_model")
        return model
    if model.get("kind") == "hurdle":
        if "h_model" not in model or "m_model" not in model:
            raise ValueError(f"{path}: hurdle model needs h_model and m_model")
    elif "u_model" not in model:
        raise ValueError(f"{path}: tree model needs u_model")
    return model


def _predict_single(features: np.ndarray, model: dict):
    """Utilities of one non-bagged tree model (hurdle or direct)."""
    if model["kind"] == "hurdle":
        h = np.clip(_booster(model["h_model"]).predict(features, num_threads=1), 0, 1)
        m = np.clip(_booster(model["m_model"]).predict(features, num_threads=1), .5, 2)
        return h * m, h, m
    u = np.clip(_booster(model["u_model"]).predict(features, num_threads=1), 0, 2)
    return u, np.full(len(features), np.nan), np.full(len(features), np.nan)


def predict_utility(features, model, return_parts=False):
    """Predict node utilities; ``model`` is a dict from :func:`load_tree_model`.

    A ``bagged_exact_direct`` model averages the (clipped) utilities of its
    members; it has no hurdle parts, so ``return_parts`` is rejected for it.
    """
    if isinstance(model, (str, Path)):
        model = load_tree_model(model)
    if model.get("feature_names") != TREE_FEATURE_NAMES:
        raise ValueError("tree model feature names/order do not match")
    x = np.asarray(features)
    if x.ndim != 2 or x.shape[1] != len(TREE_FEATURE_NAMES):
        raise ValueError("tree feature shape does not match model")
    if not len(x):
        empty = np.zeros(0)
        return (empty, empty, empty) if return_parts else empty
    if model.get("kind") == "bagged_exact_direct":
        if return_parts:
            raise ValueError("bagged tree model has no hurdle parts")
        return np.mean([_predict_single(x, member)[0] for member in model["members"]], axis=0)
    u, h, m = _predict_single(x, model)
    return (u, h, m) if return_parts else u


def tree_to_labels(prob, image, model, tau=0.26, levels=DEFAULT_LEVELS,
                   min_area=50, close_radius=0) -> np.ndarray:
    """Stage-1 instance labels from a probability map via the component tree."""
    tree = build_component_tree(prob, image, levels, min_area, close_radius)
    score = predict_utility(tree.features, model)
    selected = select_antichain(tree.parent, score, tau)
    return tree.labels(selected)
