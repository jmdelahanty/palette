"""Tests for keypoint registry preflight wrapper."""

import json
from pathlib import Path
import sys

import numpy as np
import yaml
import zarr

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.registry.db import Registry
from fisheye.utils import prepare_keypoint_training_from_registry as wrapper


def _mock_invocation_sources(monkeypatch) -> None:
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


def _write_base_pose_config(path: Path) -> None:
    payload = {
        "train": "./",
        "val": "./",
        "nc": 1,
        "names": ["fish"],
        "task": "pose",
        "kpt_shape": [3, 3],
        "datasets": {},
        "training_params": {
            "model": "yolov8n-pose.pt",
            "epochs": 1,
            "batch": 2,
            "imgsz": 256,
            "lr0": 0.001,
            "momentum": 0.9,
            "weight_decay": 0.0005,
            "patience": 1,
            "device": "0",
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _create_minimal_pose_zarr(path: Path) -> None:
    root = zarr.open_group(str(path), mode="w")
    root.attrs["session_uuid"] = "session_pose_001"

    raw = root.create_group("raw_video")
    raw.create_array(
        "images_ds",
        data=np.zeros((4, 16, 16), dtype=np.uint8),
        chunks=(1, 16, 16),
    )

    crop_parent = root.create_group("crop_runs")
    crop_parent.attrs["latest"] = "crop_pose_001"
    crop_group = crop_parent.create_group("crop_pose_001")
    crop_group.attrs["detection_source_type"] = "filtered"
    crop_group.create_array(
        "roi_images",
        data=np.zeros((4, 64, 64), dtype=np.uint8),
        chunks=(1, 64, 64),
    )

    kp_parent = root.create_group("keypoints_runs")
    kp_parent.attrs["latest"] = "kp_pose_001"
    kp_group = kp_parent.create_group("kp_pose_001")
    kp_group.attrs["method"] = "traditional_pose"
    kp_group.attrs["keypoints_timestamp_utc"] = "2026-02-06T00:00:00+00:00"
    kp_group.attrs["source_crop_run"] = "crop_pose_001"
    kp_group.attrs["success_rate"] = 0.75
    kp_group.attrs["keypoints_processed"] = 4
    kp_group.create_array(
        "keypoints_roi",
        data=np.zeros((4, 3, 2), dtype=np.float32),
        chunks=(1, 3, 2),
    )
    kp_group.create_array(
        "detection_success",
        data=np.array([True, True, True, False], dtype=np.bool_),
        chunks=(4,),
    )


def _seed_registry(registry_path: Path, zarr_path: Path) -> None:
    db = Registry(registry_path)
    db.upsert_dataset(
        "dataset_pose_001",
        session_uuid="session_pose_001",
        zarr_path=zarr_path,
    )
    db.upsert_provenance(
        "dataset_pose_001",
        provenance={},
        context={"canvas_name": "DefaultScreen"},
        protocol_name=None,
        protocol_hash=None,
        acquisition={
            "dish_design": "cedar",
            "has_images_ds": True,
            "has_images_ds_rgb": False,
            "downsample_formats_json": '["gray"]',
        },
        zarr_purpose=None,
    )
    db.close()


def test_prepare_keypoint_from_registry_writes_outputs_and_registers_set(monkeypatch, tmp_path: Path) -> None:
    zarr_path = tmp_path / "pose_sample.zarr"
    _create_minimal_pose_zarr(zarr_path)
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry(registry_path, zarr_path)
    base_config_path = tmp_path / "pose_base.yaml"
    _write_base_pose_config(base_config_path)
    _mock_invocation_sources(monkeypatch)
    dataset_root = tmp_path / "datasets"
    monkeypatch.setenv("PALETTE_TRAINING_DATASETS_ROOT", str(dataset_root))

    monkeypatch.chdir(tmp_path)
    rc = wrapper.main(
        [
            "--registry",
            str(registry_path),
            "--base-config",
            str(base_config_path),
            "--source-type",
            "filtered",
            "--input-format",
            "gray",
            "--set-name",
            "pose_smoke_set",
            "--set-version",
            "1",
            "--register",
        ]
    )
    assert rc == 0

    out_config = dataset_root / "pose_pose_smoke_set_v001" / "pose_pose_smoke_set_v001.yaml"
    out_manifest = dataset_root / "pose_pose_smoke_set_v001" / "pose_pose_smoke_set_v001.manifest.json"
    assert out_config.exists()
    assert out_manifest.exists()

    cfg = yaml.safe_load(out_config.read_text(encoding="utf-8"))
    assert cfg["task"] == "pose"
    dataset_cfg = cfg["datasets"]["pose_sample"]
    assert dataset_cfg["source_type"] == "filtered"
    assert dataset_cfg["keypoint_run"] == "kp_pose_001"

    manifest = json.loads(out_manifest.read_text(encoding="utf-8"))
    assert manifest["set_id"] == "pose_pose_smoke_set_v001"
    assert manifest["datasets"][0]["source_crop_run"] == "crop_pose_001"
    assert manifest["datasets"][0]["keypoints_total"] == 4
    assert manifest["datasets"][0]["keypoints_successful"] == 3
    assert manifest["datasets"][0]["dish_design"] == "cedar"
    assert manifest["datasets"][0]["canvas_name"] == "DefaultScreen"

    db = Registry(registry_path)
    row = db.conn.execute(
        "SELECT set_id, dataset_ids_json FROM training_sets WHERE set_id = ?",
        ("pose_pose_smoke_set_v001",),
    ).fetchone()
    db.close()
    assert row is not None
    assert row["set_id"] == "pose_pose_smoke_set_v001"
    assert json.loads(row["dataset_ids_json"]) == ["dataset_pose_001"]


def test_prepare_keypoint_from_registry_dry_run_prints_generated_artifacts(capsys, monkeypatch, tmp_path: Path) -> None:
    zarr_path = tmp_path / "pose_sample.zarr"
    _create_minimal_pose_zarr(zarr_path)
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry(registry_path, zarr_path)
    base_config_path = tmp_path / "pose_base.yaml"
    _write_base_pose_config(base_config_path)
    _mock_invocation_sources(monkeypatch)
    monkeypatch.setenv("PALETTE_TRAINING_DATASETS_ROOT", str(tmp_path / "datasets"))

    rc = wrapper.main(
        [
            "--registry",
            str(registry_path),
            "--base-config",
            str(base_config_path),
            "--source-type",
            "filtered",
            "--input-format",
            "gray",
            "--dry-run",
        ]
    )
    assert rc == 0

    captured = capsys.readouterr().out
    assert "Keypoint Training Preflight" in captured
    assert "--- Generated Config (YAML) ---" in captured
    assert "--- Training Manifest (JSON) ---" in captured


def test_prepare_keypoint_from_registry_auto_set_name_when_omitted(monkeypatch, tmp_path: Path) -> None:
    zarr_path = tmp_path / "pose_sample.zarr"
    _create_minimal_pose_zarr(zarr_path)
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry(registry_path, zarr_path)
    base_config_path = tmp_path / "pose_base.yaml"
    _write_base_pose_config(base_config_path)
    _mock_invocation_sources(monkeypatch)
    dataset_root = tmp_path / "datasets"
    monkeypatch.setenv("PALETTE_TRAINING_DATASETS_ROOT", str(dataset_root))

    rc = wrapper.main(
        [
            "--registry",
            str(registry_path),
            "--base-config",
            str(base_config_path),
            "--source-type",
            "filtered",
            "--input-format",
            "gray",
            "--register",
        ]
    )
    assert rc == 0

    manifests = sorted(dataset_root.glob("pose_*_v001/*.manifest.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    set_name = manifest["set_name"]
    set_id = manifest["set_id"]
    assert set_name.startswith("cedar_defaultscreen_filtered_gray_latest_traditional_")
    assert len(set_name.rsplit("_", 1)[-1]) == 8
    assert set_id == f"pose_{set_name}_v001"

    db = Registry(registry_path)
    row = db.conn.execute(
        "SELECT set_id FROM training_sets WHERE set_id = ?",
        (set_id,),
    ).fetchone()
    db.close()
    assert row is not None
