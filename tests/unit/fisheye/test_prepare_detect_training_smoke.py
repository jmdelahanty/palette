"""Smoke checks for detection preflight invocation audit metadata."""

import json
import hashlib
import sqlite3
from pathlib import Path
import sys

import numpy as np
import yaml
import zarr

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.diagnostics import prepare_detect_training
from fisheye.registry.db import Registry


def _write_base_detect_config(path: Path) -> None:
    payload = {
        "train": "./dummy_train.txt",
        "val": "./dummy_val.txt",
        "nc": 1,
        "names": ["fish"],
        "datasets": {},
        "training_params": {
            "model": "yolo11n.pt",
            "epochs": 1,
            "batch": 1,
            "imgsz": 640,
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _create_minimal_detect_zarr(path: Path) -> None:
    root = zarr.open_group(str(path), mode="w")
    root.attrs["session_uuid"] = "session_smoke_001"

    raw = root.create_group("raw_video")
    raw.create_array(
        "images_ds",
        data=np.zeros((2, 16, 16), dtype=np.uint8),
        chunks=(1, 16, 16),
    )

    refined_parent = root.create_group("refined_detect_runs")
    refined_parent.attrs["latest"] = "refined_detect_smoke_001"
    refined_parent.create_group("refined_detect_smoke_001").create_group("manual")

    crop_parent = root.create_group("crop_runs")
    crop_parent.attrs["latest"] = "crop_smoke_001"
    crop_group = crop_parent.create_group("crop_smoke_001")
    crop_group.attrs["detection_source_type"] = "manual"
    crop_group.attrs["detection_source_path"] = "refined_detect_runs/refined_detect_smoke_001/manual"
    crop_group.attrs["detect_review_status"] = {"state": "approved"}
    crop_group.attrs["crop_review_status"] = {"state": "approved"}
    crop_group.create_array(
        "bbox_norm_coords",
        data=np.array([[0.5, 0.5, 0.2, 0.2]], dtype=np.float32),
        chunks=(1, 4),
    )
    crop_group.create_array(
        "detection_source",
        data=np.array([0], dtype=np.int8),
        chunks=(1,),
    )
    crop_group.create_array(
        "frame_indices",
        data=np.array([0], dtype=np.int64),
        chunks=(1,),
    )


def test_prepare_detect_training_persists_invocation_metadata(monkeypatch, tmp_path: Path) -> None:
    zarr_path = tmp_path / "sample.zarr"
    _create_minimal_detect_zarr(zarr_path)

    base_config_path = tmp_path / "detect_config.yaml"
    _write_base_detect_config(base_config_path)

    registry_path = tmp_path / "palette_registry.sqlite"
    monkeypatch.setenv("PALETTE_REGISTRY_PATH", str(registry_path))
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(
        "fisheye.utils.system.get_git_info",
        lambda: {
            "commit_hash": "abc123",
            "short_hash": "abc123",
            "branch": "main",
            "is_dirty": False,
        },
    )
    monkeypatch.setattr(
        "fisheye.utils.system.get_environment_summary",
        lambda: {
            "environment_type": "conda",
            "environment_name": "pytest-env",
            "python_version": "3.11",
            "total_packages": 3,
            "key_packages": {"numpy": "0.0-test"},
        },
    )
    monkeypatch.setattr(
        "fisheye.utils.system.get_platform_info",
        lambda **_kwargs: {
            "hostname": "pytest-host",
            "username": "pytest-user",
            "system": "Linux",
            "release": "test",
        },
    )

    argv = [
        str(zarr_path),
        "--base-config",
        str(base_config_path),
        "--set-name",
        "smoke_set",
        "--set-version",
        "1",
    ]
    prepare_detect_training.main(argv)

    manifest_path = tmp_path / "runs" / "manifests" / "detect" / "smoke_set_v001.manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    invocation = manifest.get("invocation")
    assert isinstance(invocation, dict)
    assert invocation.get("tool") == "fisheye.diagnostics.prepare_detect_training"
    assert invocation.get("argv") == argv
    assert invocation.get("args", {}).get("set_name") == "smoke_set"
    assert invocation.get("git", {}).get("commit_hash") == "abc123"
    assert invocation.get("environment", {}).get("environment_name") == "pytest-env"

    with sqlite3.connect(registry_path) as conn:
        row = conn.execute(
            "SELECT invocation_json, query_filter FROM training_sets WHERE set_id = ?",
            ("detect_smoke_set_v001",),
        ).fetchone()
    assert row is not None
    invocation_json, query_filter_json = row
    assert invocation_json is not None
    registry_invocation = json.loads(invocation_json)
    assert registry_invocation.get("argv") == argv
    query_filter = json.loads(query_filter_json)
    assert query_filter.get("set_name") == "smoke_set"


def test_registry_record_training_run_persists_invocation_json(tmp_path: Path) -> None:
    registry_path = tmp_path / "palette_registry.sqlite"
    registry = Registry(registry_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("abc", encoding="utf-8")
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    invocation = {
        "tool": "fisheye.training.train_detection",
        "argv": ["config.yaml", "--log-registry"],
    }
    final_metrics = {
        "precision": 0.91,
        "recall": 0.88,
        "mAP50": 0.95,
        "mAP50_95": 0.72,
    }
    registry.record_training_run(
        run_id="run_smoke_001",
        set_id="detect_smoke_set_v001",
        config_path=config_path,
        manifest_path=None,
        model_path=None,
        metrics_path=None,
        config_sha256=config_hash,
        status="success",
        final_metrics=final_metrics,
        invocation=invocation,
    )
    registry.close()

    with sqlite3.connect(registry_path) as conn:
        row = conn.execute(
            "SELECT invocation_json, config_sha256, status, final_metrics_json FROM training_runs WHERE run_id = ?",
            ("run_smoke_001",),
        ).fetchone()
    assert row is not None
    assert row[0] is not None
    assert json.loads(row[0]) == invocation
    assert row[1] == config_hash
    assert row[2] == "success"
    assert json.loads(row[3]) == final_metrics
