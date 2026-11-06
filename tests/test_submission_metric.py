"""Submission rows, CSV validation and the Panoptic Quality metric."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from filseg.metric import aggregate_pq, evaluate_dataframes, solution_from_coco
from filseg.submission import decode_mask, encode_mask, labels_to_rows, validate_submission, write_submission


def _square(y: int, x: int, size: int) -> np.ndarray:
    mask = np.zeros((2048, 2048), np.uint8)
    mask[y : y + size, x : x + size] = 1
    return mask


def test_rle_roundtrip() -> None:
    mask = _square(100, 300, 60)
    assert np.array_equal(decode_mask(encode_mask(mask)), mask)


def test_rows_are_numbered_largest_first() -> None:
    labels = np.zeros((2048, 2048), np.int32)
    labels[0:10, 0:10] = 1  # 100 px
    labels[100:130, 100:130] = 2  # 900 px
    rows = labels_to_rows("20110101000000Ch", labels)
    assert [r["filament_id"] for r in rows] == ["20110101000000Ch_1", "20110101000000Ch_2"]
    assert decode_mask(rows[0]["segmentation_rle"]).sum() == 900


def test_validate_accepts_a_good_file(tmp_path: Path) -> None:
    labels = np.zeros((2048, 2048), np.int32)
    labels[0:10, 0:10] = 1
    labels[50:60, 0:10] = 2
    path = write_submission(labels_to_rows("20110101000000Ch", labels), tmp_path / "s.csv")
    summary = validate_submission(path, expected_stems=["20110101000000Ch", "20110102000000Ch"])
    assert summary["n_instances"] == 2 and summary["n_images_with_instances"] == 1


def test_validate_rejects_overlaps_and_unknown_stems(tmp_path: Path) -> None:
    a, b = encode_mask(_square(0, 0, 20)), encode_mask(_square(10, 10, 20))
    overlapping = tmp_path / "overlap.csv"
    pd.DataFrame({"filament_id": ["s_1", "s_2"], "segmentation_rle": [a, b]}).to_csv(overlapping, index=False)
    with pytest.raises(ValueError, match="overlap"):
        validate_submission(overlapping)
    with pytest.raises(ValueError, match="not test images"):
        validate_submission(overlapping, expected_stems=["other"])


def test_pq_semantics() -> None:
    """TP iff IoU > 0.5 (strict); PQ = sum(TP IoU) / (TP + FP/2 + FN/2)."""
    iou = np.array([[0.9, 0.0], [0.0, 0.5]])  # one TP; IoU exactly 0.5 is not a match
    metrics = aggregate_pq([{"iou_matrix": iou, "n_gt": 2, "n_pred": 2}])
    assert (metrics["tp"], metrics["fp"], metrics["fn"]) == (1, 1, 1)
    assert metrics["pq"] == pytest.approx(0.9 / 2.0)
    empty_gt = aggregate_pq([{"iou_matrix": np.zeros((0, 3)), "n_gt": 0, "n_pred": 3}])
    assert (empty_gt["fp"], empty_gt["pq"]) == (3, 0.0)


def test_each_annotation_set_is_a_separate_entry() -> None:
    gt = encode_mask(_square(0, 0, 40))
    solution = pd.DataFrame(
        {"filament_id": ["010101-20110101000000Ch_0", "020202-20110101000000Ch_0"], "segmentation_rle": [gt, gt]}
    )
    submission = pd.DataFrame({"filament_id": ["20110101000000Ch_1"], "segmentation_rle": [gt]})
    metrics, entries = evaluate_dataframes(solution, submission)
    assert len(entries) == 2
    assert (metrics["tp"], metrics["fp"], metrics["fn"], metrics["pq"]) == (2, 0, 0, 1.0)


def test_solution_from_coco(tmp_path: Path) -> None:
    square = [[10.0, 10.0, 50.0, 10.0, 50.0, 50.0, 10.0, 50.0]]
    coco = {
        "images": [
            {"id": "010101-20110101000000Ch", "file_name": "20110101000000Ch.jpeg"},
            {"id": "010101-20110102000000Ch", "file_name": "20110102000000Ch.jpeg"},
        ],
        "annotations": [
            {"image_id": "010101-20110101000000Ch", "segmentation": square},
            {"image_id": "010101-20110102000000Ch", "segmentation": square},
        ],
    }
    path = tmp_path / "train.json"
    path.write_text(json.dumps(coco))
    solution = solution_from_coco(path, stems=["20110101000000Ch"])
    assert solution["filament_id"].tolist() == ["010101-20110101000000Ch_0"]
    assert decode_mask(solution["segmentation_rle"][0]).sum() > 1000
