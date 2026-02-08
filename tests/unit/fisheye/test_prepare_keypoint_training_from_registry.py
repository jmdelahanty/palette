"""Tests for keypoint registry preflight wrapper."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest
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


def _create_minimal_pose_zarr(
    path: Path,
    *,
    keypoints_rows: int = 4,
    roi_rows: int = 4,
    success_rows: int = 4,
    include_success_rate: bool = True,
    include_source_crop_run: bool = True,
    create_refined_run: bool = False,
    refined_usable_rows: int = 0,
    review_state: str | None = None,
    review_intended_use: str | None = None,
) -> None:
    root = zarr.open_group(str(path), mode="w")
    root.attrs["session_uuid"] = "session_pose_001"

    raw = root.create_group("raw_video")
    raw.create_array(
        "images_ds",
        data=np.zeros((keypoints_rows, 16, 16), dtype=np.uint8),
        chunks=(1, 16, 16),
    )

    crop_parent = root.create_group("crop_runs")
    crop_parent.attrs["latest"] = "crop_pose_001"
    crop_group = crop_parent.create_group("crop_pose_001")
    crop_group.attrs["detection_source_type"] = "filtered"
    crop_group.create_array(
        "roi_images",
        data=np.zeros((roi_rows, 64, 64), dtype=np.uint8),
        chunks=(1, 64, 64),
    )

    kp_parent = root.create_group("keypoints_runs")
    kp_parent.attrs["latest"] = "kp_pose_001"
    kp_group = kp_parent.create_group("kp_pose_001")
    kp_group.attrs["method"] = "traditional_pose"
    kp_group.attrs["keypoints_timestamp_utc"] = "2026-02-06T00:00:00+00:00"
    if include_source_crop_run:
        kp_group.attrs["source_crop_run"] = "crop_pose_001"
    if include_success_rate:
        kp_group.attrs["success_rate"] = 0.75
    kp_group.attrs["keypoints_processed"] = keypoints_rows
    kp_group.create_array(
        "keypoints_roi",
        data=np.zeros((keypoints_rows, 3, 2), dtype=np.float32),
        chunks=(1, 3, 2),
    )
    kp_group.create_array(
        "detection_success",
        data=np.array([True] * max(success_rows - 1, 0) + [False], dtype=np.bool_),
        chunks=(max(success_rows, 1),),
    )

    if create_refined_run:
        refined_parent = root.create_group("refined_keypoints_runs")
        refined_parent.attrs["latest"] = "refined_pose_001"
        refined_group = refined_parent.create_group("refined_pose_001")
        refined_group.attrs["source_keypoints_run"] = "kp_pose_001"
        refined_group.attrs["created_utc"] = "2026-02-07T00:00:00+00:00"
        if review_state is not None or review_intended_use is not None:
            refined_group.attrs["keypoint_review_status"] = {
                "state": review_state or "approved",
                "intended_use": review_intended_use or "training",
                "timestamp": "2026-02-07T00:00:00+00:00",
            }
        usable = np.array(
            [True] * max(refined_usable_rows, 0) + [False] * max(keypoints_rows - refined_usable_rows, 0),
            dtype=np.bool_,
        )
        refined_group.create_array("usable_keypoints", data=usable, chunks=(max(keypoints_rows, 1),))


def _seed_registry(registry_path: Path, zarr_path: Path) -> None:
    db = Registry(registry_path)
    root = zarr.open_group(str(zarr_path), mode="r")
    db.register_from_root(root, zarr_path)
    # Keep deterministic provenance values used by assertions.
    db.upsert_provenance(
        "session_pose_001",
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
    assert json.loads(row["dataset_ids_json"]) == ["session_pose_001"]


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


def test_prepare_keypoint_from_registry_requires_source_crop_run(monkeypatch, tmp_path: Path) -> None:
    zarr_path = tmp_path / "pose_sample.zarr"
    _create_minimal_pose_zarr(zarr_path, include_source_crop_run=False)
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry(registry_path, zarr_path)
    base_config_path = tmp_path / "pose_base.yaml"
    _write_base_pose_config(base_config_path)
    _mock_invocation_sources(monkeypatch)
    monkeypatch.setenv("PALETTE_TRAINING_DATASETS_ROOT", str(tmp_path / "datasets"))

    with pytest.raises(ValueError, match="missing source_crop_run"):
        wrapper.main(
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


def test_prepare_keypoint_from_registry_fails_on_roi_keypoint_row_mismatch(monkeypatch, tmp_path: Path) -> None:
    zarr_path = tmp_path / "pose_sample.zarr"
    _create_minimal_pose_zarr(zarr_path, keypoints_rows=4, roi_rows=3)
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry(registry_path, zarr_path)
    base_config_path = tmp_path / "pose_base.yaml"
    _write_base_pose_config(base_config_path)
    _mock_invocation_sources(monkeypatch)
    monkeypatch.setenv("PALETTE_TRAINING_DATASETS_ROOT", str(tmp_path / "datasets"))

    with pytest.raises(ValueError, match="roi/keypoint row mismatch"):
        wrapper.main(
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


def test_prepare_keypoint_from_registry_fails_on_detection_success_row_mismatch(monkeypatch, tmp_path: Path) -> None:
    zarr_path = tmp_path / "pose_sample.zarr"
    _create_minimal_pose_zarr(
        zarr_path,
        keypoints_rows=4,
        roi_rows=4,
        success_rows=3,
        include_success_rate=False,
    )
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry(registry_path, zarr_path)
    base_config_path = tmp_path / "pose_base.yaml"
    _write_base_pose_config(base_config_path)
    _mock_invocation_sources(monkeypatch)
    monkeypatch.setenv("PALETTE_TRAINING_DATASETS_ROOT", str(tmp_path / "datasets"))

    with pytest.raises(ValueError, match="detection_success row mismatch"):
        wrapper.main(
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


def test_prepare_keypoint_from_registry_enforces_review_status_and_quality(monkeypatch, tmp_path: Path) -> None:
    zarr_path = tmp_path / "pose_sample.zarr"
    _create_minimal_pose_zarr(
        zarr_path,
        keypoints_rows=4,
        roi_rows=4,
        success_rows=4,
        include_success_rate=True,
        create_refined_run=True,
        refined_usable_rows=3,
        review_state="approved",
        review_intended_use="training",
    )
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
            "manual",
            "--input-format",
            "gray",
            "--min-usable-keypoints-rate",
            "0.70",
            "--require-review-state",
            "approved",
            "--require-review-intended-use",
            "training",
            "--dry-run",
        ]
    )
    assert rc == 0


def test_prepare_keypoint_from_registry_fails_when_review_status_missing(monkeypatch, tmp_path: Path) -> None:
    zarr_path = tmp_path / "pose_sample.zarr"
    _create_minimal_pose_zarr(
        zarr_path,
        keypoints_rows=4,
        roi_rows=4,
        success_rows=4,
        include_success_rate=True,
        create_refined_run=True,
        refined_usable_rows=3,
        review_state=None,
        review_intended_use=None,
    )
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry(registry_path, zarr_path)
    base_config_path = tmp_path / "pose_base.yaml"
    _write_base_pose_config(base_config_path)
    _mock_invocation_sources(monkeypatch)
    monkeypatch.setenv("PALETTE_TRAINING_DATASETS_ROOT", str(tmp_path / "datasets"))

    with pytest.raises(SystemExit, match="No datasets remain after keypoint quality filtering"):
        wrapper.main(
            [
                "--registry",
                str(registry_path),
                "--base-config",
                str(base_config_path),
                "--source-type",
                "manual",
                "--input-format",
                "gray",
                "--require-review-state",
                "approved",
                "--dry-run",
            ]
        )


def test_prepare_keypoint_from_registry_exclusion_is_nonfatal(monkeypatch, tmp_path: Path) -> None:
    zarr_path = tmp_path / "pose_sample.zarr"
    _create_minimal_pose_zarr(
        zarr_path,
        keypoints_rows=4,
        roi_rows=4,
        success_rows=4,
        include_success_rate=True,
        create_refined_run=True,
        refined_usable_rows=3,
        review_state=None,
        review_intended_use=None,
    )
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry(registry_path, zarr_path)
    base_config_path = tmp_path / "pose_base.yaml"
    _write_base_pose_config(base_config_path)
    _mock_invocation_sources(monkeypatch)
    monkeypatch.setenv("PALETTE_TRAINING_DATASETS_ROOT", str(tmp_path / "datasets"))

    with pytest.raises(SystemExit, match="No datasets remain after keypoint quality filtering"):
        wrapper.main(
            [
                "--registry",
                str(registry_path),
                "--base-config",
                str(base_config_path),
                "--source-type",
                "manual",
                "--input-format",
                "gray",
                "--require-review-state",
                "approved",
                "--dry-run",
            ]
        )


def test_prepare_keypoint_from_registry_review_gate_falls_back_to_reviewed_source_run(
    monkeypatch, tmp_path: Path
) -> None:
    zarr_path = tmp_path / "pose_sample.zarr"
    _create_minimal_pose_zarr(
        zarr_path,
        keypoints_rows=4,
        roi_rows=4,
        success_rows=4,
        include_success_rate=True,
        create_refined_run=True,
        refined_usable_rows=3,
        review_state="approved",
        review_intended_use="training",
    )

    root = zarr.open_group(str(zarr_path), mode="a")
    kp_parent = root["keypoints_runs"]
    kp_parent["kp_pose_001"].attrs["method"] = "yolo_pose"

    kp_group = kp_parent.create_group("kp_pose_002")
    kp_group.attrs["method"] = "traditional_pose"
    kp_group.attrs["keypoints_timestamp_utc"] = "2026-02-08T00:00:00+00:00"
    kp_group.attrs["source_crop_run"] = "crop_pose_001"
    kp_group.attrs["success_rate"] = 1.0
    kp_group.attrs["keypoints_processed"] = 4
    kp_group.create_array(
        "keypoints_roi",
        data=np.zeros((4, 3, 2), dtype=np.float32),
        chunks=(1, 3, 2),
    )
    kp_group.create_array(
        "detection_success",
        data=np.array([True, True, True, True], dtype=np.bool_),
        chunks=(4,),
    )
    kp_parent.attrs["latest"] = "kp_pose_002"

    registry_path = tmp_path / "registry.sqlite"
    _seed_registry(registry_path, zarr_path)
    base_config_path = tmp_path / "pose_base.yaml"
    _write_base_pose_config(base_config_path)
    _mock_invocation_sources(monkeypatch)
    monkeypatch.setenv("PALETTE_TRAINING_DATASETS_ROOT", str(tmp_path / "datasets"))

    out_config = tmp_path / "pose_config.yaml"
    out_manifest = tmp_path / "pose_manifest.json"
    rc = wrapper.main(
        [
            "--registry",
            str(registry_path),
            "--base-config",
            str(base_config_path),
            "--source-type",
            "manual",
            "--input-format",
            "gray",
            "--keypoint-run",
            "latest_traditional",
            "--require-review-state",
            "approved",
            "--require-review-intended-use",
            "training",
            "--allow-cross-method-review-fallback",
            "--out-config",
            str(out_config),
            "--out-manifest",
            str(out_manifest),
        ]
    )
    assert rc == 0

    manifest = json.loads(out_manifest.read_text(encoding="utf-8"))
    dataset = manifest["datasets"][0]
    assert dataset["keypoint_run_selector"] == "latest_traditional_quality"
    assert dataset["keypoint_run_resolved"] == "kp_pose_001"
    assert any("cross-method fallback" in warning for warning in dataset["warnings"])


def test_prepare_keypoint_from_registry_review_gate_is_strict_without_fallback_flag(
    monkeypatch, tmp_path: Path
) -> None:
    zarr_path = tmp_path / "pose_sample.zarr"
    _create_minimal_pose_zarr(
        zarr_path,
        keypoints_rows=4,
        roi_rows=4,
        success_rows=4,
        include_success_rate=True,
        create_refined_run=True,
        refined_usable_rows=3,
        review_state="approved",
        review_intended_use="training",
    )

    root = zarr.open_group(str(zarr_path), mode="a")
    kp_parent = root["keypoints_runs"]
    kp_parent["kp_pose_001"].attrs["method"] = "yolo_pose"

    kp_group = kp_parent.create_group("kp_pose_002")
    kp_group.attrs["method"] = "traditional_pose"
    kp_group.attrs["keypoints_timestamp_utc"] = "2026-02-08T00:00:00+00:00"
    kp_group.attrs["source_crop_run"] = "crop_pose_001"
    kp_group.attrs["success_rate"] = 1.0
    kp_group.attrs["keypoints_processed"] = 4
    kp_group.create_array(
        "keypoints_roi",
        data=np.zeros((4, 3, 2), dtype=np.float32),
        chunks=(1, 3, 2),
    )
    kp_group.create_array(
        "detection_success",
        data=np.array([True, True, True, True], dtype=np.bool_),
        chunks=(4,),
    )
    kp_parent.attrs["latest"] = "kp_pose_002"

    registry_path = tmp_path / "registry.sqlite"
    _seed_registry(registry_path, zarr_path)
    base_config_path = tmp_path / "pose_base.yaml"
    _write_base_pose_config(base_config_path)
    _mock_invocation_sources(monkeypatch)
    monkeypatch.setenv("PALETTE_TRAINING_DATASETS_ROOT", str(tmp_path / "datasets"))

    with pytest.raises(SystemExit, match="No datasets remain after keypoint quality filtering"):
        wrapper.main(
            [
                "--registry",
                str(registry_path),
                "--base-config",
                str(base_config_path),
                "--source-type",
                "manual",
                "--input-format",
                "gray",
                "--keypoint-run",
                "latest_traditional",
                "--require-review-state",
                "approved",
                "--require-review-intended-use",
                "training",
                "--dry-run",
            ]
        )


def test_prepare_keypoint_from_registry_accepts_legacy_refined_group_name(monkeypatch, tmp_path: Path) -> None:
    zarr_path = tmp_path / "pose_sample.zarr"
    _create_minimal_pose_zarr(
        zarr_path,
        keypoints_rows=4,
        roi_rows=4,
        success_rows=4,
        include_success_rate=True,
        create_refined_run=False,
    )

    root = zarr.open_group(str(zarr_path), mode="a")
    legacy_parent = root.create_group("keypoints_refined_runs")
    legacy_parent.attrs["latest"] = "refined_pose_legacy_001"
    refined = legacy_parent.create_group("refined_pose_legacy_001")
    refined.attrs["source_keypoints_run"] = "kp_pose_001"
    refined.attrs["created_utc"] = "2026-02-08T00:00:00+00:00"
    refined.attrs["keypoint_review_status"] = {
        "state": "approved",
        "intended_use": "training",
        "timestamp": "2026-02-08T00:00:00+00:00",
    }
    refined.create_array(
        "usable_keypoints",
        data=np.array([True, True, True, False], dtype=np.bool_),
        chunks=(4,),
    )

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
            "manual",
            "--input-format",
            "gray",
            "--require-review-state",
            "approved",
            "--require-review-intended-use",
            "training",
            "--min-usable-keypoints-rate",
            "0.70",
            "--dry-run",
        ]
    )
    assert rc == 0


def test_prepare_keypoint_from_registry_fails_closed_on_stale_quality_row(
    monkeypatch, tmp_path: Path
) -> None:
    zarr_path = tmp_path / "pose_sample.zarr"
    _create_minimal_pose_zarr(
        zarr_path,
        keypoints_rows=4,
        roi_rows=4,
        success_rows=4,
        include_success_rate=True,
        create_refined_run=True,
        refined_usable_rows=3,
        review_state="approved",
        review_intended_use="training",
    )
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry(registry_path, zarr_path)
    db = Registry(registry_path)
    db.conn.execute("UPDATE keypoint_quality SET zarr_mtime_ns = zarr_mtime_ns - 1;")
    db.conn.commit()
    db.close()

    base_config_path = tmp_path / "pose_base.yaml"
    _write_base_pose_config(base_config_path)
    _mock_invocation_sources(monkeypatch)
    monkeypatch.setenv("PALETTE_TRAINING_DATASETS_ROOT", str(tmp_path / "datasets"))

    with pytest.raises(ValueError, match="filesystem mtime"):
        wrapper.main(
            [
                "--registry",
                str(registry_path),
                "--base-config",
                str(base_config_path),
                "--source-type",
                "manual",
                "--input-format",
                "gray",
                "--require-review-state",
                "approved",
                "--require-review-intended-use",
                "training",
                "--min-usable-keypoints-rate",
                "0.70",
                "--dry-run",
            ]
        )


def test_prepare_keypoint_from_registry_fails_closed_on_quality_divergence(
    monkeypatch, tmp_path: Path
) -> None:
    zarr_path = tmp_path / "pose_sample.zarr"
    _create_minimal_pose_zarr(
        zarr_path,
        keypoints_rows=4,
        roi_rows=4,
        success_rows=4,
        include_success_rate=True,
        create_refined_run=True,
        refined_usable_rows=3,
        review_state="approved",
        review_intended_use="training",
    )
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry(registry_path, zarr_path)
    db = Registry(registry_path)
    db.conn.execute("UPDATE keypoint_quality SET review_state = 'pending';")
    db.conn.commit()
    db.close()

    base_config_path = tmp_path / "pose_base.yaml"
    _write_base_pose_config(base_config_path)
    _mock_invocation_sources(monkeypatch)
    monkeypatch.setenv("PALETTE_TRAINING_DATASETS_ROOT", str(tmp_path / "datasets"))

    with pytest.raises(ValueError, match="review metadata divergence"):
        wrapper.main(
            [
                "--registry",
                str(registry_path),
                "--base-config",
                str(base_config_path),
                "--source-type",
                "manual",
                "--input-format",
                "gray",
                "--require-review-state",
                "approved",
                "--require-review-intended-use",
                "training",
                "--min-usable-keypoints-rate",
                "0.70",
                "--dry-run",
            ]
        )
