"""Tests for detect training config audit utility."""

from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.utils.check_detect_training_config import audit_detect_training_config, main


def _write_config(path: Path, training_params: dict) -> Path:
    zarr_path = path.parent / "sample.zarr"
    zarr_path.mkdir(parents=True, exist_ok=True)
    (zarr_path / "zarr.json").write_text("{}", encoding="utf-8")

    config = {
        "train": "./dummy_train.txt",
        "val": "./dummy_val.txt",
        "nc": 1,
        "names": ["fish"],
        "datasets": {
            "sample": {
                "zarr_path": str(zarr_path),
                "source_type": "manual",
                "input_format": "gray",
            }
        },
        "task": "detect",
        "training_params": training_params,
    }
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_audit_detect_training_config_reports_ignored_keys(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path / "detect.yaml",
        {
            "model": "yolo11n.pt",
            "epochs": 1,
            "batch": 4,
            "imgsz": 64,
            "lr0": 0.002,
            "momentum": 0.9,
            "weight_decay": 0.0005,
            "patience": 10,
            "device": "0",
            "rect": True,
            "mosaic": 0.0,
            "auto_augment": None,
        },
    )

    summary = audit_detect_training_config(config_path)
    assert summary["ignored_training_param_keys"] == ["auto_augment", "mosaic"]
    assert summary["ignored_augment_keys"] == ["auto_augment", "mosaic"]
    assert "mosaic" not in summary["effective_training_params"]
    assert summary["notes"]["optimizer_auto_may_override_lr0_momentum"] is True


def test_main_strict_returns_nonzero_when_keys_are_ignored(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path / "detect.yaml",
        {
            "model": "yolo11n.pt",
            "epochs": 1,
            "batch": 4,
            "imgsz": 64,
            "lr0": 0.002,
            "momentum": 0.9,
            "weight_decay": 0.0005,
            "patience": 10,
            "device": "0",
            "rect": True,
            "mosaic": 0.0,
        },
    )

    assert main([str(config_path), "--strict"]) == 2
