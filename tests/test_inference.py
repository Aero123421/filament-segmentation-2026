"""Image geometry, test-time augmentation, checkpoints and the end-to-end pipeline (CPU)."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from filseg.infer import FilamentPipeline, PipelineConfig, apply_transform, invert_transform, tta_transforms
from filseg.model import build_model, load_checkpoint
from filseg.preprocess import crop_center, paste_to_full, prob_to_uint8, read_image
from filseg.submission import labels_to_rows


@pytest.mark.parametrize("mode", ["none", "hflip", "vflip", "hvflip", "d4"])
def test_tta_views_are_exactly_invertible(mode: str) -> None:
    image = np.random.default_rng(0).random((37, 53)).astype(np.float32)
    for view in tta_transforms(mode):
        assert np.array_equal(invert_transform(apply_transform(image, view), view), image)


def test_d4_has_eight_distinct_views() -> None:
    image = np.arange(16).reshape(4, 4)
    assert len({apply_transform(image, v).tobytes() for v in tta_transforms("d4")}) == 8
    with pytest.raises(ValueError):
        tta_transforms("rotate45")


def test_crop_paste_and_uint8() -> None:
    image = np.random.default_rng(1).integers(0, 256, (2048, 2048), dtype=np.uint8)
    crop = crop_center(image)
    canvas = paste_to_full(crop.astype(np.float32))
    assert crop.shape == (1888, 1888)
    assert np.array_equal(canvas[80:1968, 80:1968], crop) and not canvas[:80].any()
    prob = np.array([[0.0, 0.5, 1.0]], np.float32)
    assert prob_to_uint8(prob).tolist() == [[0, 128, 255]]


def test_read_image_checks_size(tmp_path: Path) -> None:
    cv2.imwrite(str(tmp_path / "small.jpeg"), np.zeros((100, 100), np.uint8))
    with pytest.raises(ValueError, match="2048x2048"):
        read_image(tmp_path / "small.jpeg")
    with pytest.raises(FileNotFoundError):
        read_image(tmp_path / "missing.jpeg")


def test_unsupported_models_fail_clearly() -> None:
    with pytest.raises(ValueError, match="preprocess"):
        build_model({"preprocess": "radial_clahe"})
    with pytest.raises(ValueError, match="decoder"):
        build_model({"decoder": "deeplab"})


def test_config_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "pipeline_config.json"
    path.write_text(json.dumps({"weights": ["a.pt"], "postprocess": {"threshold": 0.5}, "tta": "d4", "typo": 1}))
    with pytest.raises(ValueError, match="typo"):
        PipelineConfig.from_json(path)
    with pytest.raises(ValueError):
        PipelineConfig(weights=["a.pt"], postprocess={"threshold": 0.5}, tta="d8")


def test_pipeline_end_to_end_on_cpu(tmp_path: Path) -> None:
    """A random ResNet18 U-Net checkpoint runs through the whole pipeline."""
    torch.manual_seed(0)
    model = build_model({"encoder": "resnet18"})
    state = {k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()}
    torch.save({"model": state, "args": {"encoder": "resnet18", "decoder": "unet"}}, tmp_path / "fold0.pt")
    assert set(load_checkpoint(tmp_path / "fold0.pt")) == {"model", "args"}
    config = PipelineConfig(
        weights=["fold0.pt"], postprocess={"threshold": 0.5, "min_area": 10, "min_mean": 0.6}, tta="none"
    )
    pipeline = FilamentPipeline(config, tmp_path, device="cpu")
    assert pipeline.precision == "fp32"
    result = pipeline.predict(np.zeros((2048, 2048), np.uint8))
    assert result.probability.shape == (2048, 2048) and result.probability.dtype == np.float32
    assert result.labels.shape == (2048, 2048) and result.labels.dtype == np.int32
    assert not result.probability[:80].any()  # the border outside the crop is background
    assert isinstance(labels_to_rows("x", result.labels), list)
