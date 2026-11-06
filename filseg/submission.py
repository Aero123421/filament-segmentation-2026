"""Submission format: instance labels -> ``filament_id, segmentation_rle`` rows.

One row per predicted filament.  ``filament_id`` is ``"<image stem>_<k>"``
(``k = 1`` is the largest instance of that image) and ``segmentation_rle`` is
the compressed COCO run-length encoding (pycocotools ``counts`` string) of a
2048x2048 binary mask.  Images without predictions simply have no rows.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as mask_util

from filseg.preprocess import FULL_SIZE

COLUMNS = ["filament_id", "segmentation_rle"]

__all__ = ["COLUMNS", "decode_mask", "encode_mask", "labels_to_rows", "validate_submission", "write_submission"]


def encode_mask(mask: np.ndarray) -> str:
    """Encode a binary ``(H, W)`` mask as a compressed COCO RLE counts string."""
    rle = mask_util.encode(np.asfortranarray((mask > 0).astype(np.uint8)))
    counts = rle["counts"]
    return counts.decode("ascii") if isinstance(counts, bytes) else str(counts)


def decode_mask(counts: str, height: int = FULL_SIZE, width: int = FULL_SIZE) -> np.ndarray:
    """Decode a compressed COCO RLE counts string to a ``(H, W)`` uint8 mask."""
    return mask_util.decode({"size": [height, width], "counts": counts}).astype(np.uint8)


def labels_to_rows(stem: str, labels: np.ndarray) -> list[dict]:
    """Submission rows for one image, largest instance first.

    ``labels`` must be an int label image (0 = background); since each pixel
    carries one label, the encoded instances are disjoint by construction.
    """
    areas = np.bincount(labels.ravel())
    instance_ids = [k for k in sorted(range(1, len(areas)), key=lambda k: (-int(areas[k]), k)) if areas[k] > 0]
    return [
        {"filament_id": f"{stem}_{rank}", "segmentation_rle": encode_mask(labels == k)}
        for rank, k in enumerate(instance_ids, 1)
    ]


def write_submission(rows: list[dict], path: str | Path) -> Path:
    """Write rows to a CSV with the competition's two columns."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=COLUMNS).to_csv(path, index=False)
    return path


def validate_submission(path: str | Path, expected_stems: list[str] | None = None) -> dict:
    """Check a submission CSV and return summary counts.

    Checks: the two required columns, unique ``filament_id`` values of the form
    ``<stem>_<k>``, every RLE decodes to a non-empty 2048x2048 mask, instances
    of one image never overlap, and (if given) every stem is a known image.

    Raises:
        ValueError: On the first failed check, naming the offending row or image.
    """
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(df.columns) != COLUMNS:
        raise ValueError(f"expected columns {COLUMNS}, got {list(df.columns)}")
    if not df["filament_id"].is_unique:
        raise ValueError("duplicate filament_id values")
    by_stem: dict[str, list[dict]] = {}
    for filament_id, counts in zip(df["filament_id"], df["segmentation_rle"], strict=True):
        stem, _, k = filament_id.rpartition("_")
        if not stem or not k.isdigit():
            raise ValueError(f"filament_id {filament_id!r} is not of the form <stem>_<k>")
        if not counts:
            raise ValueError(f"{filament_id}: empty RLE string")
        by_stem.setdefault(stem, []).append({"size": [FULL_SIZE, FULL_SIZE], "counts": counts.encode("ascii")})
    if expected_stems is not None:
        unknown = sorted(set(by_stem) - set(expected_stems))
        if unknown:
            raise ValueError(f"{len(unknown)} stem(s) are not test images, e.g. {unknown[:3]}")
    n_pixels = 0
    for stem, rles in by_stem.items():
        areas = mask_util.area(rles)
        if (areas == 0).any():
            raise ValueError(f"{stem}: empty instance mask")
        union = int(mask_util.area(mask_util.merge(rles, intersect=False)))
        if union != int(areas.sum()):  # disjoint masks <=> union area == sum of areas
            raise ValueError(f"{stem}: instances overlap ({int(areas.sum()) - union} shared pixels)")
        n_pixels += union
    return {
        "n_instances": len(df),
        "n_images_with_instances": len(by_stem),
        "n_images_expected": None if expected_stems is None else len(expected_stems),
        "mean_instances_per_image": len(df) / max(len(by_stem), 1),
        "total_instance_pixels": n_pixels,
    }
