"""Combined pipeline: staged quantization, component tree and configuration.

The data-dependent parity checks at the bottom read the competition inputs
(stored SV maps, exported stage-1 seeds) when they are present next to the
repository and skip otherwise; the exactness they assert was verified on the
full test set by ``scripts/muse_hr2_parity.py`` (see the release report).
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from filseg.infer import PipelineConfig, family_keys, quantized_mean_uint8
from filseg.tree import TREE_FEATURE_NAMES, load_tree_model, select_antichain, tree_to_labels

ROOT = Path(__file__).resolve().parents[3]  # repository root (tests/ -> template -> scripts -> root)
_WEIGHTS_DIR = Path(__file__).resolve().parents[1] / "weights"
# Template fixture; in a built release the same model is bundled as P_tree.json.
TREE_JSON = _WEIGHTS_DIR / "tree_l15_t26.json" if (_WEIGHTS_DIR / "tree_l15_t26.json").exists() else (
    _WEIGHTS_DIR / "P_tree.json")


def test_quantized_mean_rounds_like_the_canonical_rule() -> None:
    a = np.array([[0, 1, 254]], np.uint8)
    b = np.array([[1, 2, 255]], np.uint8)
    # Means 0.5, 1.5, 254.5 round to even (np.rint), exactly as stored maps do.
    assert quantized_mean_uint8([a, b]).tolist() == [[0, 2, 254]]
    assert quantized_mean_uint8([a]).tolist() == a.tolist()
    with pytest.raises(ValueError):
        quantized_mean_uint8([])
    with pytest.raises(ValueError):
        quantized_mean_uint8([a.astype(np.float32)])


def test_staged_quantization_differs_from_flat_rounding() -> None:
    """Two families at 0.5/255 offsets: staged keeps both steps, flat loses one."""
    family_a = np.array([[100]], np.uint8)
    family_b = np.array([[101]], np.uint8)
    assert quantized_mean_uint8([family_a, family_b]).tolist() == [[100]]  # rint(100.5) -> 100
    # The flat float mean of the underlying probabilities would round the same
    # way only by accident; the staged rule is defined on uint8 inputs alone.


def test_family_keys_group_folds_of_one_recipe() -> None:
    args = (
        [{"encoder": "resnet34", "target": "soft", "seed": 1}] * 2
        + [{"encoder": "tu-tf_efficientnetv2_s", "target": "individual", "seed": 1}] * 2
    )
    keys = family_keys(args)
    assert len(set(keys)) == 2 and keys[0] != keys[2] and keys[0] == keys[1]


def test_select_antichain_prefers_descendants_and_can_be_empty() -> None:
    parent = np.array([-1, 0, 0], np.int32)
    # Parent alone (0.30-0.26) loses to the two children together.
    assert select_antichain(parent, np.array([0.30, 0.30, 0.30]), tau=0.26).tolist() == [1, 2]
    # Nothing clears the threshold: the empty antichain is allowed.
    assert select_antichain(parent, np.array([0.10, 0.10, 0.10]), tau=0.26).tolist() == []
    with pytest.raises(ValueError):
        select_antichain(np.array([0], np.int32), np.array([1.0]))


def test_bundled_tree_model_is_portable_json() -> None:
    model = load_tree_model(TREE_JSON)
    assert model["feature_names"] == TREE_FEATURE_NAMES and len(TREE_FEATURE_NAMES) == 43
    assert model["kind"] == "exact_direct" and "u_model" in model  # LightGBM text, no pickle
    from filseg.tree import predict_utility

    scores = predict_utility(np.zeros((2, 43), np.float32), model)
    assert scores.shape == (2,) and np.isfinite(scores).all()


def test_bagged_tree_averages_member_utilities(tmp_path: Path) -> None:
    """The P3 tree (bagged_exact_direct) is the mean of its members' utilities."""
    from filseg.tree import predict_utility

    single = load_tree_model(TREE_JSON)
    bagged = {"feature_names": TREE_FEATURE_NAMES, "kind": "bagged_exact_direct", "members": [single, single]}
    path = tmp_path / "bagged.json"
    path.write_text(json.dumps(bagged))
    loaded = load_tree_model(path)
    features = np.random.default_rng(0).normal(size=(5, 43)).astype(np.float32)
    assert np.array_equal(predict_utility(features, loaded), predict_utility(features, single))
    with pytest.raises(ValueError, match="no hurdle parts"):
        predict_utility(features, loaded, return_parts=True)
    path.write_text(json.dumps({"feature_names": TREE_FEATURE_NAMES, "kind": "bagged_exact_direct", "members": []}))
    with pytest.raises(ValueError, match="members"):
        load_tree_model(path)


def test_tree_to_labels_on_a_synthetic_blob() -> None:
    prob = np.zeros((256, 256), np.float32)
    prob[100:140, 60:200] = 0.9
    image = np.full((256, 256), 128, np.uint8)
    labels = tree_to_labels(prob, image, load_tree_model(TREE_JSON), tau=0.26)
    assert labels.dtype == np.int32 and labels.max() >= 1
    assert set(np.unique(labels)) == set(range(int(labels.max()) + 1))  # compact, disjoint


def test_combined_config_validation() -> None:
    with pytest.raises(ValueError, match="tree_model"):
        PipelineConfig(weights=["a.pt"], postprocess={"threshold": 0.5}, postprocess_method="component_tree")
    with pytest.raises(ValueError, match="partition"):
        PipelineConfig(
            weights=["a.pt", "b.pt"], postprocess={"threshold": 0.5},
            postprocess_method="component_tree", tree_model="tree.json", weight_groups=[["a.pt"]],
        )
    config = PipelineConfig(
        weights=["s.pt", "v.pt"], postprocess={"threshold": 0.5},
        postprocess_method="component_tree", tree_model="tree.json",
        weight_groups=[["s.pt"], ["v.pt"]], quantize_uint8=True,
    )
    assert config.tree_tau == 0.26


@pytest.mark.skipif(
    not (ROOT / "experiments/sol_r3_mix_sv/test").exists() or os.environ.get("FILSEG_SKIP_PARITY"),
    reason="competition probability maps are not present in this checkout",
)
def test_staged_and_tree_parity_on_one_stem() -> None:
    """Stored S/V maps -> staged SV map -> tree labels, exactly as exported."""
    stem = sorted(p.stem for p in (ROOT / "experiments/sol_r3_mix_sv/test").glob("*.npz"))[0]

    def load(folder: Path) -> np.ndarray:
        with np.load(folder / f"{stem}.npz") as z:
            return z["prob"]

    staged = quantized_mean_uint8([
        load(ROOT / "experiments/sol_r3_r34_soft30k_ema_native2k_ema/test"),
        load(ROOT / "experiments/sol_r3_v2s_30k_ema_native2k_ema/test"),
    ])
    assert np.array_equal(staged, load(ROOT / "experiments/sol_r3_mix_sv/test"))

    import cv2

    prob = staged.astype(np.float32) / 255
    image = cv2.imread(str(ROOT / f"data/MAGFiLO_1.0_Kaggle_2026/test/test_images/{stem}.jpeg"),
                       cv2.IMREAD_GRAYSCALE)
    got = tree_to_labels(prob, image, load_tree_model(TREE_JSON), tau=0.26, min_area=50, close_radius=0)
    with np.load(ROOT / f"experiments/op_combined_export/test/{stem}.npz", allow_pickle=False) as z:
        sel = json.loads(str(z["selections"]))["unique"][0]
        pos = {int(n): i for i, n in enumerate(z["node_ids"])}
        want = np.zeros((2048, 2048), np.int32)
        for k, node in enumerate(sel, 1):
            i = pos[int(node)]
            y, x, h, w = (int(v) for v in z["boxes"][i])
            m = np.unpackbits(z["bits"][z["offs"][i]:z["offs"][i + 1]])[:h * w].reshape(h, w).astype(bool)
            want[y:y + h, x:x + w][m] = k
    assert np.array_equal(got, want)
