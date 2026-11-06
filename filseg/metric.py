"""Panoptic Quality (PQ), as computed by the organisers' self-evaluation notebook.

- Every annotation set (one annotator's labels of one image) is a separate
  entry; the predictions of that image are compared with each set.
- A (ground truth, prediction) pair with IoU > 0.5 (strict) is a true positive.
- A prediction without such a partner is a false positive; a ground-truth
  filament without one is a false negative.
- ``PQ = sum(TP IoU) / (TP + 0.5 FP + 0.5 FN)``, accumulated over all entries
  (not averaged per image).  ``SQ`` is the mean TP IoU and ``RQ = TP / (TP +
  0.5 FP + 0.5 FN)``, so ``PQ = SQ x RQ``.

Command line (prints the metrics as JSON)::

    python -m filseg.metric solution.csv submission.csv

A ``solution.csv`` for training images can be built with
:func:`solution_from_coco`; its ``filament_id`` values are
``<annotator id>-<stem>_<k>``.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as mask_util

from filseg.preprocess import FULL_SIZE

IOU_THRESHOLD = 0.5

__all__ = ["aggregate_pq", "evaluate_dataframes", "iou_matrix_from_rles", "solution_from_coco"]


def _as_rle(rle: str | dict) -> dict:
    """pycocotools RLE dict from a counts string or an RLE dict."""
    if isinstance(rle, dict):
        counts = rle["counts"]
        return {"size": list(rle["size"]), "counts": counts.encode("ascii") if isinstance(counts, str) else counts}
    return {"size": [FULL_SIZE, FULL_SIZE], "counts": str(rle).encode("ascii")}


def iou_matrix_from_rles(gt_rles: Sequence, pred_rles: Sequence) -> tuple[np.ndarray, np.ndarray]:
    """Exact IoU and Dice matrices, both of shape ``(n_gt, n_pred)``."""
    n_gt, n_pred = len(gt_rles), len(pred_rles)
    if n_gt == 0 or n_pred == 0:
        return np.zeros((n_gt, n_pred)), np.zeros((n_gt, n_pred))
    gt = [_as_rle(r) for r in gt_rles]
    pred = [_as_rle(r) for r in pred_rles]
    iou = np.asarray(mask_util.iou(pred, gt, [0] * n_gt), dtype=np.float64).T  # pycocotools returns (pred, gt)
    area_sum = np.asarray(mask_util.area(gt), np.float64)[:, None] + np.asarray(mask_util.area(pred), np.float64)[None]
    # intersection = IoU * (|A| + |B|) / (1 + IoU), rounded to whole pixels
    intersection = np.rint(np.where(iou > 0, iou * area_sum / (1.0 + iou), 0.0))
    dice = np.divide(2.0 * intersection, area_sum, out=np.zeros_like(intersection), where=area_sum > 0)
    return iou, dice


def aggregate_pq(entries: Iterable[dict]) -> dict:
    """PQ, SQ, RQ, TP/FP/FN and match statistics over entries.

    Each entry is ``{"iou_matrix": (n_gt, n_pred) array, "n_gt": int,
    "n_pred": int}`` with an optional ``"dice_matrix"``.  ``gt_one_to_many``
    counts ground-truth filaments overlapped (IoU > 0) by two or more
    predictions; ``pred_many_to_one`` counts predictions overlapping two or
    more ground-truth filaments.
    """
    tp_ious: list[float] = []
    positive_ious: list[float] = []
    positive_dice: list[float] = []
    fp = fn = n_gt_total = n_pred_total = 0
    gt_degree: Counter = Counter()
    pred_degree: Counter = Counter()
    for entry in entries:
        iou = np.asarray(entry["iou_matrix"], dtype=np.float64)
        n_gt, n_pred = int(entry["n_gt"]), int(entry["n_pred"])
        n_gt_total += n_gt
        n_pred_total += n_pred
        if iou.size:
            positive_ious.extend(iou[iou > 0].tolist())
            if entry.get("dice_matrix") is not None:
                positive_dice.extend(np.asarray(entry["dice_matrix"], np.float64)[iou > 0].tolist())
        if n_gt == 0 or n_pred == 0:
            fp += n_pred if n_gt == 0 else 0
            fn += n_gt if n_pred == 0 else 0
            gt_degree.update([0] * n_gt)
            pred_degree.update([0] * n_pred)
            continue
        hit = iou > IOU_THRESHOLD
        tp_ious.extend(iou[hit].tolist())
        fp += int((~hit.any(axis=0)).sum())
        fn += int((~hit.any(axis=1)).sum())
        gt_degree.update((iou > 0).sum(axis=1).tolist())
        pred_degree.update((iou > 0).sum(axis=0).tolist())
    tp = len(tp_ious)
    denominator = tp + 0.5 * fp + 0.5 * fn
    return {
        "pq": float(sum(tp_ious) / denominator) if denominator else 0.0,
        "sq": float(np.mean(tp_ious)) if tp else 0.0,
        "rq": float(tp / denominator) if denominator else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_gt": n_gt_total,
        "n_pred": n_pred_total,
        "mean_iou_pos": float(np.mean(positive_ious)) if positive_ious else 0.0,
        "mean_dice_pos": float(np.mean(positive_dice)) if positive_dice else 0.0,
        "gt_one_to_many": int(sum(c for d, c in gt_degree.items() if d >= 2)),
        "pred_many_to_one": int(sum(c for d, c in pred_degree.items() if d >= 2)),
    }


def evaluate_dataframes(solution: pd.DataFrame, submission: pd.DataFrame) -> tuple[dict, list[dict]]:
    """Score a submission against a solution (both with the two submission columns).

    Solution rows are grouped into entries by the ``filament_id`` prefix before
    the first ``"_"`` (``<annotator id>-<stem>``); predictions are grouped by
    stem.  Returns ``(metrics, per_entry)``.
    """
    solution_keys = solution["filament_id"].astype(str).str.split("_", n=1).str[0]
    gt_groups: dict[str, list[str]] = {}
    for key, rle in zip(solution_keys, solution["segmentation_rle"].astype(str), strict=True):
        gt_groups.setdefault(key, []).append(rle)
    pred_groups: dict[str, list[str]] = {}
    ids = submission["filament_id"].astype(str)
    for filament_id, rle in zip(ids, submission["segmentation_rle"].astype(str), strict=True):
        pred_groups.setdefault(filament_id.split("_", 1)[0], []).append(rle)
    entries, per_entry = [], []
    for key, gt_rles in gt_groups.items():
        stem = key.split("-", 1)[1] if "-" in key else key
        pred_rles = pred_groups.get(stem, [])
        iou, dice = iou_matrix_from_rles(gt_rles, pred_rles)
        entries.append({"iou_matrix": iou, "dice_matrix": dice, "n_gt": len(gt_rles), "n_pred": len(pred_rles)})
        per_entry.append({"entry": key, "stem": stem, "n_gt": len(gt_rles), "n_pred": len(pred_rles)})
    return aggregate_pq(entries), per_entry


def solution_from_coco(json_path: str | Path, stems: Iterable[str] | None = None) -> pd.DataFrame:
    """Ground truth of the training JSON in submission format, one row per filament.

    ``filament_id`` is ``<annotation-set id>_<k>`` (``k`` in JSON order), the
    layout the organisers' evaluation expects.  ``stems`` restricts the rows to
    those images (for example one validation fold).
    """
    raw = json.loads(Path(json_path).read_text())
    keep = None if stems is None else set(stems)
    set_ids = sorted(str(im["id"]) for im in raw["images"] if keep is None or Path(im["file_name"]).stem in keep)
    by_set: dict[str, list[dict]] = {}
    for annotation in raw["annotations"]:
        by_set.setdefault(str(annotation["image_id"]), []).append(annotation)
    rows = []
    for set_id in set_ids:
        for k, annotation in enumerate(by_set.get(set_id, [])):
            rles = mask_util.frPyObjects(annotation["segmentation"], FULL_SIZE, FULL_SIZE)
            rle = (rles[0] if len(rles) == 1 else mask_util.merge(rles)) if isinstance(rles, list) else rles
            rows.append({"filament_id": f"{set_id}_{k}", "segmentation_rle": rle["counts"].decode("ascii")})
    return pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])


def main() -> None:
    """``python -m filseg.metric SOLUTION SUBMISSION``: print the metrics as JSON."""
    parser = argparse.ArgumentParser(description="Panoptic Quality of a submission CSV against a solution CSV.")
    parser.add_argument("solution")
    parser.add_argument("submission")
    args = parser.parse_args()
    metrics, _ = evaluate_dataframes(pd.read_csv(args.solution), pd.read_csv(args.submission))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
