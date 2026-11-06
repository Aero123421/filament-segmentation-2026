"""Build the training cache from the competition files (CPU only, run once).

For every training image (stem) this writes ``<out>/<stem>.npz`` with

- ``image``: the central 1888x1888 crop of the grayscale JPEG (uint8);
- ``masks``: one 1888x1888 uint8 mask per annotation set, stacked in sorted
  annotation-set order; each mask is the union of that annotator's filaments,

plus ``<out>/manifest.json`` with one record per stem: ``stem``, ``fold`` (from
``folds.csv``), ``ids`` (annotation-set ids) and ``k`` (number of sets).
Keeping the annotators separate lets :mod:`filseg.train` supervise each crop
with one annotator's labels (``individual``) or with their mean (``soft``).

Usage::

    python -m filseg.prepare --train-dir /path/to/MAGFiLO_1.0_Kaggle_2026/train \
        --folds folds.csv --out prepared
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_util

from filseg.preprocess import FULL_SIZE, crop_center, read_image

ANNOTATION_FILE = "MAGFiLO_1.0_Annotations_kaggle2026_train.json"

__all__ = ["annotation_sets", "annotator_mask", "prepare", "read_folds"]


def read_folds(path: str | Path) -> dict[str, int]:
    """``stem -> fold`` from a ``folds.csv`` with ``stem`` and ``fold`` columns."""
    with open(path, newline="") as handle:
        return {row["stem"]: int(row["fold"]) for row in csv.DictReader(handle)}


def annotation_sets(json_path: str | Path) -> tuple[dict[str, list[str]], dict[str, list[dict]]]:
    """Group the COCO annotations of the training JSON.

    Returns:
        ``(sets_by_stem, rles_by_set)``: the sorted annotation-set ids of each
        stem (stems sorted), and the RLE of every filament of each set.
    """
    raw = json.loads(Path(json_path).read_text())
    sets_by_stem: dict[str, list[str]] = defaultdict(list)
    for image in raw["images"]:
        sets_by_stem[Path(image["file_name"]).stem].append(str(image["id"]))
    rles_by_set: dict[str, list[dict]] = defaultdict(list)
    for annotation in raw["annotations"]:
        rle = mask_util.frPyObjects(annotation["segmentation"], FULL_SIZE, FULL_SIZE)
        rles_by_set[str(annotation["image_id"])].append(mask_util.merge(rle) if isinstance(rle, list) else rle)
    return {stem: sorted(ids) for stem, ids in sorted(sets_by_stem.items())}, rles_by_set


def annotator_mask(rles: list[dict]) -> np.ndarray:
    """Union of one annotation set's filaments as a 2048x2048 uint8 mask."""
    if not rles:
        return np.zeros((FULL_SIZE, FULL_SIZE), np.uint8)
    return mask_util.decode(mask_util.merge(rles)).astype(np.uint8)


def prepare(train_dir: str | Path, folds_csv: str | Path, out_dir: str | Path) -> list[dict]:
    """Write the cache for every training stem and return the manifest records."""
    train_dir, out_dir = Path(train_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    folds = read_folds(folds_csv)
    sets_by_stem, rles_by_set = annotation_sets(train_dir / ANNOTATION_FILE)
    missing = sorted(set(sets_by_stem) - set(folds))
    if missing:
        raise ValueError(f"{len(missing)} training stem(s) have no fold in {folds_csv}, e.g. {missing[:3]}")
    records = []
    start = time.perf_counter()
    for index, (stem, set_ids) in enumerate(sets_by_stem.items(), 1):
        image = read_image(train_dir / "train_images" / f"{stem}.jpeg")
        masks = np.stack([crop_center(annotator_mask(rles_by_set[set_id])) for set_id in set_ids])
        np.savez_compressed(out_dir / f"{stem}.npz", image=crop_center(image), masks=masks)
        records.append({"stem": stem, "fold": folds[stem], "ids": set_ids, "k": len(set_ids)})
        if index % 50 == 0 or index == len(sets_by_stem):
            print(f"prepared {index}/{len(sets_by_stem)} images ({time.perf_counter() - start:.0f}s)", flush=True)
    (out_dir / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    return records


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Build the training cache (cropped images + per-annotator masks).")
    parser.add_argument("--train-dir", required=True, help="competition 'train' folder (train_images/ + the JSON)")
    parser.add_argument("--folds", required=True, help="folds.csv shipped with this repository")
    parser.add_argument("--out", required=True, help="output folder (about 1.6 GB for the 707 training images)")
    args = parser.parse_args()
    prepare(args.train_dir, args.folds, args.out)


if __name__ == "__main__":
    main()
