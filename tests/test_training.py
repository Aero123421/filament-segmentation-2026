"""Training data preparation, the crop sampler, checkpoint export and gradient checkpointing (CPU)."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from filseg.model import build_model
from filseg.prepare import prepare
from filseg.train import CropDataset, enable_grad_checkpointing, export_state, frozen_running_stats, loss_fn


def _synthetic_cache() -> tuple[list[dict], dict]:
    rng = np.random.default_rng(0)
    records, cache = [], {}
    for i, k in enumerate([1, 3]):
        masks = np.zeros((k, 1888, 1888), np.uint8)
        masks[0, 900:960, 400:1400] = 1
        if k == 3:  # annotators disagree: the second marks half of it, the third nothing
            masks[1, 900:930, 400:1400] = 1
        records.append({"stem": f"s{i}", "fold": 0, "k": k})
        cache[f"s{i}"] = (rng.integers(0, 256, (1888, 1888), dtype=np.uint8), masks)
    return records, cache


def test_sampler_is_deterministic_and_offset_continues_the_stream() -> None:
    records, cache = _synthetic_cache()
    stream = CropDataset(records, cache, crop=1024, target="individual", seed=1, length=20)
    shifted = CropDataset(records, cache, crop=1024, target="individual", seed=1, length=20, offset=5)
    x, y = stream[7]
    assert x.shape == y.shape == (1, 1024, 1024) and x.dtype == torch.float32
    assert torch.equal(x, stream[7][0]) and torch.equal(y, stream[7][1])
    assert torch.equal(shifted[2][0], stream[7][0])


def test_soft_target_is_the_annotator_mean() -> None:
    records, cache = _synthetic_cache()
    soft = CropDataset(records, cache, crop=1888, target="soft", seed=3, length=16)
    values = np.unique(np.concatenate([soft[i][1].numpy().ravel() for i in range(16)]))
    allowed = np.array([0.0, 1 / 3, 2 / 3, 1.0], np.float32)
    assert np.isclose(values[:, None], allowed[None]).any(axis=1).all()
    assert np.isclose(values, 1 / 3).any() and np.isclose(values, 2 / 3).any()


def test_loss_is_low_for_a_perfect_prediction() -> None:
    target = torch.zeros(1, 1, 32, 32)
    target[..., 8:24, 8:24] = 1
    assert loss_fn(target * 40 - 20, target) < 0.01 < loss_fn(torch.zeros_like(target), target)


def test_export_keeps_buffers_fp32_and_rejects_non_finite() -> None:
    model = build_model({"encoder": "resnet18"})
    state = export_state(model)
    assert state["encoder.conv1.weight"].dtype == torch.float16
    assert state["encoder.bn1.running_var"].dtype == torch.float32
    model.encoder.bn1.running_var.fill_(float("inf"))
    with pytest.raises(RuntimeError, match="non-finite"):
        export_state(model)


def test_grad_checkpointing_gives_the_same_gradients_and_statistics() -> None:
    """Recomputing timm encoder blocks must change neither gradients nor BatchNorm statistics."""
    torch.manual_seed(0)
    reference = build_model({"encoder": "tu-tf_efficientnetv2_s"}).train()
    checkpointed = build_model({"encoder": "tu-tf_efficientnetv2_s"}).train()
    checkpointed.load_state_dict(reference.state_dict())
    batchnorms = enable_grad_checkpointing(checkpointed)
    x, y = torch.rand(2, 1, 64, 64), (torch.rand(2, 1, 64, 64) > 0.5).float()
    loss_fn(reference(x), y).backward()
    loss = loss_fn(checkpointed(x), y)
    with frozen_running_stats(batchnorms):
        loss.backward()
    grads = dict(reference.named_parameters())
    for name, parameter in checkpointed.named_parameters():
        assert torch.allclose(parameter.grad, grads[name].grad, rtol=1e-4, atol=1e-6), name
    stats = reference.state_dict()
    for name, value in checkpointed.state_dict().items():
        if "running" in name:
            assert torch.allclose(value, stats[name], rtol=1e-5, atol=1e-7), name
    with pytest.raises(ValueError, match="timm encoder"):
        enable_grad_checkpointing(build_model({"encoder": "resnet18"}))


def test_short_cpu_training_run(tmp_path: Path) -> None:
    """A two-step training run on CPU writes a checkpoint and OOF maps (no CUDA needed)."""
    import argparse

    from filseg.train import train

    data = tmp_path / "prepared"
    data.mkdir()
    records = []
    for stem, fold in (("s0", 1), ("s1", 0)):
        image = np.full((1888, 1888), 100, np.uint8)
        masks = np.zeros((1, 1888, 1888), np.uint8)
        masks[0, 900:960, 400:1400] = 1
        np.savez_compressed(data / f"{stem}.npz", image=image, masks=masks)
        records.append({"stem": stem, "fold": fold, "ids": [f"01-{stem}"], "k": 1})
    (data / "manifest.json").write_text(json.dumps(records))
    args = argparse.Namespace(
        data=str(data), out=str(tmp_path / "runs"), fold=0, encoder="resnet18", decoder="unet",
        target="individual", crop=64, steps=2, stop_after=0, lr=3e-4, batch=2, seed=1, sample_offset=0,
        init_from="", workers=0, channels_last=False, fused_adamw=False, grad_checkpointing=False,
    )
    meta = train(args)
    assert meta["val_stems"] == 1 and meta["train_stems"] == 1
    assert (tmp_path / "runs" / "fold0.pt").exists()
    assert (tmp_path / "runs" / "oof" / "s1.npz").exists()


def test_prepare_writes_crops_masks_and_manifest(tmp_path: Path) -> None:
    train = tmp_path / "train"
    (train / "train_images").mkdir(parents=True)
    for stem in ("20110101000000Ch", "20110102000000Ch"):
        cv2.imwrite(str(train / "train_images" / f"{stem}.jpeg"), np.full((2048, 2048), 100, np.uint8))
    polygon = [[200.0, 200.0, 400.0, 200.0, 400.0, 260.0, 200.0, 260.0]]
    coco = {
        "images": [
            {"id": "02-20110101000000Ch", "file_name": "20110101000000Ch.jpeg"},
            {"id": "01-20110101000000Ch", "file_name": "20110101000000Ch.jpeg"},
            {"id": "01-20110102000000Ch", "file_name": "20110102000000Ch.jpeg"},
        ],
        "annotations": [{"image_id": "01-20110101000000Ch", "segmentation": polygon}],
    }
    (train / "MAGFiLO_1.0_Annotations_kaggle2026_train.json").write_text(json.dumps(coco))
    (tmp_path / "folds.csv").write_text("stem,fold\n20110101000000Ch,0\n20110102000000Ch,3\n")
    records = prepare(train, tmp_path / "folds.csv", tmp_path / "prepared")
    expected = {"stem": "20110101000000Ch", "fold": 0, "ids": ["01-20110101000000Ch", "02-20110101000000Ch"], "k": 2}
    assert records[0] == expected
    with np.load(tmp_path / "prepared" / "20110101000000Ch.npz") as arrays:
        assert arrays["image"].shape == (1888, 1888)
        assert arrays["masks"].shape == (2, 1888, 1888)
        assert arrays["masks"][0].sum() > 10000 and arrays["masks"][1].sum() == 0  # annotator 02 drew nothing
    assert json.loads((tmp_path / "prepared" / "manifest.json").read_text()) == records
