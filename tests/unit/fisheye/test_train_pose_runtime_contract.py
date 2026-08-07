from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from fisheye.training.train_pose import (
    _apply_pose_loader_training_param_overrides,
    _observe_pose_runtime_optimizer,
    _observe_pose_runtime_batch,
    _validate_pose_effective_arguments,
    _write_pose_runtime_receipt,
)


def _requested() -> dict[str, object]:
    return {
        "imgsz": 512,
        "pose": 12.0,
        "kobj": 1.0,
        "box": 7.5,
        "cls": 0.5,
        "dfl": 1.5,
        "lr0": 0.001,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "optimizer": "AdamW",
        "rect": False,
        "augment": False,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "fliplr": 0.0,
        "flipud": 0.0,
        "erasing": 0.0,
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "cutmix": 0.0,
        "auto_augment": None,
        "multi_scale": False,
        "workers": 24,
        "seed": 42,
    }


def test_effective_pose_arguments_require_exact_match() -> None:
    requested = _requested()
    receipt = _validate_pose_effective_arguments(
        requested, SimpleNamespace(**requested)
    )
    assert receipt["status"] == "exact_match"
    assert receipt["effective"]["imgsz"] == 512

    changed = dict(requested)
    changed["pose"] = 18.0
    with pytest.raises(ValueError, match='"pose"'):
        _validate_pose_effective_arguments(
            requested, SimpleNamespace(**changed)
        )


def test_pose_loader_overrides_preserve_losses_and_disable_unreachable_transforms() -> None:
    requested = _requested()
    requested.update(
        {
            "augment": True,
            "num_workers": 24,
            "persistent_workers": True,
            "prefetch_factor": 2,
            "deterministic_val": True,
        }
    )

    effective, loader = _apply_pose_loader_training_param_overrides(requested)

    assert effective["pose"] == 12.0
    assert effective["kobj"] == 1.0
    assert effective["box"] == 7.5
    assert effective["workers"] == 24
    assert effective["mosaic"] == 0.0
    assert effective["auto_augment"] is None
    assert effective["multi_scale"] is False
    assert effective["optimizer"] == "AdamW"
    assert effective["augment"] is False
    assert loader["augmentation_enabled"] is True
    assert loader["num_workers"] == 24
    assert loader["persistent_workers"] is True


def test_pose_runtime_optimizer_receipt_uses_instantiated_optimizer() -> None:
    optimizer = type("AdamW", (), {})()
    optimizer.param_groups = [
        {
            "lr": 0.001,
            "initial_lr": 0.001,
            "betas": (0.9, 0.999),
            "weight_decay": 0.0005,
        }
    ]
    receipt = _observe_pose_runtime_optimizer(
        {"requested_training_params": {"optimizer": "AdamW"}},
        SimpleNamespace(optimizer=optimizer),
    )

    assert receipt["status"] == "verified"
    assert receipt["effective_class"] == "AdamW"
    assert receipt["parameter_groups"][0]["lr"] == pytest.approx(0.001)
    assert receipt["parameter_groups"][0]["betas"] == pytest.approx([0.9, 0.999])


def test_pose_runtime_batch_receipt_captures_actual_tensor_contract(tmp_path) -> None:
    state: dict[str, object] = {
        "run_dir": str(tmp_path),
        "status": "effective_arguments_verified",
        "model_input_shape_hw": [64, 64],
        "effective_arguments": {"status": "exact_match"},
    }
    raw = torch.full((2, 3, 64, 64), 128, dtype=torch.uint8)
    normalized = raw.float() / 255.0

    _observe_pose_runtime_batch(
        state,
        raw_batch=raw,
        normalized_batch=normalized,
    )

    assert state["status"] == "runtime_batch_verified"
    assert state["first_batch"]["raw_shape_nchw"] == [2, 3, 64, 64]
    receipt_path = _write_pose_runtime_receipt(state)
    assert receipt_path is not None
    document = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert document["schema_id"] == "palette.pose_training_runtime_receipt.v1"
    assert document["payload"]["first_batch"]["raw_dtype"] == "uint8"
    assert len(document["payload_sha256"]) == 64


def test_pose_runtime_batch_rejects_undeclared_shape() -> None:
    state = {"model_input_shape_hw": [64, 64]}
    raw = torch.zeros((1, 3, 32, 32), dtype=torch.uint8)

    with pytest.raises(ValueError, match="model-input contract"):
        _observe_pose_runtime_batch(
            state,
            raw_batch=raw,
            normalized_batch=raw.float(),
        )
