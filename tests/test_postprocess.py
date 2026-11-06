"""Post-processing: threshold, closing, components and the mean-probability gate."""

import numpy as np
import pytest

from filseg.postprocess import probs_to_labels, trace_postprocess


def test_threshold_is_strict() -> None:
    """Pixels exactly at the threshold are background."""
    assert probs_to_labels(np.full((8, 8), 0.5, np.float32), {"threshold": 0.5}).max() == 0
    assert probs_to_labels(np.full((8, 8), 0.51, np.float32), {"threshold": 0.5}).max() == 1


def test_mean_gate_removes_weak_component() -> None:
    """A diffuse low-probability blob is removed; a confident one is kept."""
    prob = np.zeros((64, 64), np.float32)
    prob[4:20, 4:20] = 0.9
    prob[30:50, 30:50] = 0.5
    labels = probs_to_labels(prob, {"threshold": 0.45, "min_area": 10, "min_mean": 0.7})
    assert labels.max() == 1
    assert np.count_nonzero(labels) == 16 * 16


def test_min_area_removes_small_component() -> None:
    prob = np.zeros((32, 32), np.float32)
    prob[2:4, 2:4] = 0.9  # 4 px
    prob[10:20, 10:20] = 0.9  # 100 px
    labels = probs_to_labels(prob, {"threshold": 0.5, "min_area": 5})
    assert labels.max() == 1 and np.count_nonzero(labels) == 100


def test_closing_bridges_a_small_gap() -> None:
    """Two segments separated by a 2-px gap become one instance with closing."""
    prob = np.zeros((40, 80), np.float32)
    prob[18:22, 5:38] = 0.9
    prob[18:22, 40:75] = 0.9
    assert probs_to_labels(prob, {"threshold": 0.5, "close_radius": 0}).max() == 2
    assert probs_to_labels(prob, {"threshold": 0.5, "close_radius": 2}).max() == 1


def test_labels_are_compact_and_in_raster_order() -> None:
    prob = np.zeros((30, 30), np.float32)
    prob[1:5, 1:5] = 0.9  # first in raster order
    prob[10:14, 1:5] = 0.3  # removed by the gate
    prob[20:28, 20:28] = 0.95
    labels = probs_to_labels(prob, {"threshold": 0.2, "min_mean": 0.8})
    assert labels.dtype == np.int32
    assert sorted(np.unique(labels)) == [0, 1, 2]
    assert labels[2, 2] == 1 and labels[24, 24] == 2


def test_trace_matches_labels() -> None:
    rng = np.random.default_rng(0)
    prob = rng.random((64, 64)).astype(np.float32)
    params = {"threshold": 0.6, "close_radius": 1, "min_area": 3, "min_mean": 0.7}
    trace = trace_postprocess(prob, params)
    assert np.array_equal(trace.labels, probs_to_labels(prob, params))
    assert int(trace.kept.sum()) == int(trace.labels.max())
    assert np.isnan(trace.mean_probs[trace.areas < 3]).all()


@pytest.mark.parametrize(
    "params",
    [
        {"threshold": 0.5, "smooth_sigma": 1.0}, {"min_area": 10}, {"threshold": 1.5},
        {"threshold": 0.5, "close_radius": -1},
    ],
)
def test_invalid_parameters_are_rejected(params: dict) -> None:
    with pytest.raises(ValueError):
        probs_to_labels(np.zeros((4, 4), np.float32), params)


@pytest.mark.parametrize(
    "prob", [np.zeros((4, 4), np.uint8), np.full((4, 4), 1.5, np.float32), np.zeros(4, np.float32)]
)
def test_invalid_probability_maps_are_rejected(prob: np.ndarray) -> None:
    with pytest.raises(ValueError):
        probs_to_labels(prob, {"threshold": 0.5})
