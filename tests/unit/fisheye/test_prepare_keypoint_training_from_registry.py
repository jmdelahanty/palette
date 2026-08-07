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


def _test_skeleton(keypoint_count: int) -> list[list[int]]:
    if int(keypoint_count) == 3:
        return [[0, 1], [0, 2], [1, 2]]
    if int(keypoint_count) == 5:
        return [[0, 1], [0, 2], [1, 2], [3, 1], [3, 2], [0, 4]]
    return [[idx, idx + 1] for idx in range(max(0, int(keypoint_count) - 1))]


def test_prepare_keypoint_skeleton_signature_helpers() -> None:
    assert wrapper._normalize_kpt_shape([3, 3]) == (3, 3)
    assert wrapper._normalize_kpt_shape([3, "3"]) == (3, 3)
    assert wrapper._normalize_kpt_shape([0, 3]) is None
    assert wrapper._normalize_kpt_shape(None) is None
    assert (
        wrapper._format_skeleton_signature(skeleton_id="pose_skel_a", kpt_shape=(3, 3))
        == "skeleton_id=pose_skel_a, kpt_shape=[3,3]"
    )


def test_prepare_keypoint_requires_exact_ordered_skeleton_source(tmp_path: Path) -> None:
    root = zarr.open_group(str(tmp_path / "missing_edges.zarr"), mode="w")
    group = root.create_group("keypoints")
    group.attrs["skeleton_id"] = "pose_skel_unpublished"
    group.attrs["pose_schema"] = {
        "name": "unpublished_schema",
        "skeleton_id": "pose_skel_unpublished",
        "kpt_shape": [3, 2],
    }
    with pytest.raises(ValueError, match="no exact ordered skeleton edge declaration"):
        wrapper._resolve_keypoint_skeleton(group, keypoint_count=3)


def _mock_invocation_sources(monkeypatch) -> None:
    monkeypatch.setattr(
        "fisheye.shared.system_metadata.get_git_info",
        lambda: {
            "commit_hash": "abc123",
            "short_hash": "abc123",
            "branch": "main",
            "is_dirty": False,
        },
    )
    monkeypatch.setattr(
        "fisheye.shared.system_metadata.get_environment_summary",
        lambda: {
            "environment_type": "conda",
            "environment_name": "pytest-env",
            "python_version": "3.11",
            "total_packages": 3,
            "key_packages": {"numpy": "0.0-test"},
        },
    )
    monkeypatch.setattr(
        "fisheye.shared.system_metadata.get_platform_info",
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
    session_uuid: str = "session_pose_001",
    keypoints_rows: int = 4,
    keypoint_count: int = 3,
    roi_rows: int = 4,
    success_rows: int = 4,
    include_success_rate: bool = True,
    include_source_crop_run: bool = True,
    create_refined_run: bool = False,
    refined_usable_rows: int = 0,
    review_state: str | None = None,
    review_intended_use: str | None = None,
    skeleton_id: str | None = "pose_skel_traditional_v1",
    kpt_shape: tuple[int, int] | None = (3, 3),
    pose_schema_name: str | None = "traditional_v1",
    refined_run_name: str = "refined_pose_001",
    refined_keypoint_count: int | None = None,
    refined_skeleton_id: str | None = None,
    refined_runtime_kpt_shape: tuple[int, int] | None = None,
    refined_pose_schema_name: str | None = None,
    detection_source_type: str = "refined",
    source_roi_pixel_contract_name: str | None = "orange_mono_pynvvc_luma_uint8_v1",
    source_roi_read_mode: str | None = "materialized_crop_run",
    source_roi_cache_backend: str | None = None,
    input_mode_effective: str | None = None,
) -> None:
    root = zarr.open_group(str(path), mode="w")
    root.attrs["session_uuid"] = session_uuid

    raw = root.create_group("raw_video")
    raw.create_array(
        "images_ds",
        data=np.zeros((keypoints_rows, 16, 16), dtype=np.uint8),
        chunks=(1, 16, 16),
    )

    crop_parent = root.create_group("crop_runs")
    crop_parent.attrs["latest"] = "crop_pose_001"
    crop_group = crop_parent.create_group("crop_pose_001")
    crop_group.attrs["detection_source_type"] = detection_source_type
    if source_roi_pixel_contract_name is not None:
        crop_group.attrs["roi_pixel_contract_name"] = source_roi_pixel_contract_name
        crop_group.attrs["roi_pixel_contract"] = {
            "name": source_roi_pixel_contract_name
        }
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
    kp_group.attrs["keypoint_labels"] = [
        f"kpt_{idx}" for idx in range(int(keypoint_count))
    ]
    if skeleton_id is not None:
        kp_group.attrs["skeleton_id"] = skeleton_id
    if kpt_shape is not None:
        kp_group.attrs["kpt_shape"] = [int(kpt_shape[0]), int(kpt_shape[1])]
    if pose_schema_name is not None or kpt_shape is not None:
        pose_schema_payload = {}
        if pose_schema_name is not None:
            pose_schema_payload["name"] = str(pose_schema_name)
        if kpt_shape is not None:
            pose_schema_payload["kpt_shape"] = [int(kpt_shape[0]), int(kpt_shape[1])]
        pose_schema_payload["edges"] = _test_skeleton(keypoint_count)
        kp_group.attrs["pose_schema"] = pose_schema_payload
    kp_group.attrs["keypoint_skeleton"] = _test_skeleton(keypoint_count)
    if include_source_crop_run:
        kp_group.attrs["source_crop_run"] = "crop_pose_001"
    if source_roi_pixel_contract_name is not None:
        kp_group.attrs["source_roi_pixel_contract_name"] = (
            source_roi_pixel_contract_name
        )
        kp_group.attrs["source_roi_pixel_contract"] = {
            "name": source_roi_pixel_contract_name
        }
    if source_roi_read_mode is not None:
        kp_group.attrs["source_roi_read_mode"] = source_roi_read_mode
    if source_roi_cache_backend is not None:
        kp_group.attrs["source_roi_cache_backend"] = source_roi_cache_backend
    if input_mode_effective is not None:
        kp_group.attrs["input_mode_effective"] = input_mode_effective
    if include_success_rate:
        kp_group.attrs["success_rate"] = 0.75
    kp_group.attrs["keypoints_processed"] = keypoints_rows
    kp_group.create_array(
        "keypoints_roi",
        data=np.zeros((keypoints_rows, int(keypoint_count), 2), dtype=np.float32),
        chunks=(1, int(keypoint_count), 2),
    )
    kp_group.create_array(
        "detection_success",
        data=np.array([True] * max(success_rows - 1, 0) + [False], dtype=np.bool_),
        chunks=(max(success_rows, 1),),
    )

    if create_refined_run:
        refined_parent = root.create_group("refined_keypoints_runs")
        refined_parent.attrs["latest"] = refined_run_name
        refined_group = refined_parent.create_group(refined_run_name)
        refined_group.attrs["source_keypoints_run"] = "kp_pose_001"
        refined_group.attrs["created_utc"] = "2026-02-07T00:00:00+00:00"
        resolved_refined_keypoint_count = int(
            refined_keypoint_count
            if refined_keypoint_count is not None
            else keypoint_count
        )
        resolved_refined_shape = (
            refined_runtime_kpt_shape
            if refined_runtime_kpt_shape is not None
            else (resolved_refined_keypoint_count, 2)
        )
        resolved_refined_schema_name = refined_pose_schema_name or pose_schema_name
        resolved_refined_skeleton_id = refined_skeleton_id or skeleton_id
        refined_group.attrs["keypoint_labels"] = [
            f"refined_kpt_{idx}" for idx in range(resolved_refined_keypoint_count)
        ]
        if resolved_refined_skeleton_id is not None:
            refined_group.attrs["skeleton_id"] = resolved_refined_skeleton_id
        if resolved_refined_shape is not None:
            refined_group.attrs["kpt_shape"] = [
                int(resolved_refined_shape[0]),
                int(resolved_refined_shape[1]),
            ]
        if (
            resolved_refined_schema_name is not None
            or resolved_refined_shape is not None
        ):
            refined_pose_schema_payload = {}
            if resolved_refined_schema_name is not None:
                refined_pose_schema_payload["name"] = str(resolved_refined_schema_name)
            if resolved_refined_skeleton_id is not None:
                refined_pose_schema_payload["skeleton_id"] = str(
                    resolved_refined_skeleton_id
                )
            if resolved_refined_shape is not None:
                refined_pose_schema_payload["kpt_shape"] = [
                    int(resolved_refined_shape[0]),
                    int(resolved_refined_shape[1]),
                ]
            refined_pose_schema_payload["edges"] = _test_skeleton(
                resolved_refined_keypoint_count
            )
            refined_group.attrs["pose_schema"] = refined_pose_schema_payload
        refined_group.attrs["keypoint_skeleton"] = _test_skeleton(
            resolved_refined_keypoint_count
        )
        if review_state is not None or review_intended_use is not None:
            refined_group.attrs["keypoint_review_status"] = {
                "state": review_state or "approved",
                "intended_use": review_intended_use or "training",
                "timestamp": "2026-02-07T00:00:00+00:00",
            }
        refined_group.create_array(
            "keypoints_roi",
            data=np.zeros(
                (keypoints_rows, resolved_refined_keypoint_count, 2), dtype=np.float32
            ),
            chunks=(1, resolved_refined_keypoint_count, 2),
        )
        usable = np.array(
            [True] * max(refined_usable_rows, 0)
            + [False] * max(keypoints_rows - refined_usable_rows, 0),
            dtype=np.bool_,
        )
        refined_group.create_array(
            "usable_keypoints", data=usable, chunks=(max(keypoints_rows, 1),)
        )


def _seed_registry(registry_path: Path, zarr_path: Path) -> None:
    db = Registry(registry_path)
    root = zarr.open_group(str(zarr_path), mode="r")
    dataset_id = str(root.attrs.get("session_uuid") or "session_pose_001")
    db.register_from_root(root, zarr_path)
    # Keep deterministic provenance values used by assertions.
    db.upsert_provenance(
        dataset_id,
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


def test_prepare_keypoint_from_registry_writes_outputs_and_registers_set(
    monkeypatch, tmp_path: Path
) -> None:
    zarr_path = tmp_path / "pose_sample.zarr"
    _create_minimal_pose_zarr(zarr_path)
    # Historical writers sometimes persisted percentage-valued success_rate
    # attributes. The exact boolean payload must remain authoritative.
    root = zarr.open_group(str(zarr_path), mode="a", use_consolidated=False)
    root["keypoints_runs/kp_pose_001"].attrs["success_rate"] = 75.0
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

    out_config = (
        dataset_root / "pose_pose_smoke_set_v001" / "pose_pose_smoke_set_v001.yaml"
    )
    out_manifest = (
        dataset_root
        / "pose_pose_smoke_set_v001"
        / "pose_pose_smoke_set_v001.manifest.json"
    )
    assert out_config.exists()
    assert out_manifest.exists()

    cfg = yaml.safe_load(out_config.read_text(encoding="utf-8"))
    assert cfg["task"] == "pose"
    dataset_cfg = cfg["datasets"]["pose_sample"]
    assert dataset_cfg["source_type"] == "refined"
    assert dataset_cfg["keypoint_run"] == "kp_pose_001"

    manifest = json.loads(out_manifest.read_text(encoding="utf-8"))
    assert manifest["set_id"] == "pose_pose_smoke_set_v001"
    assert manifest["datasets"][0]["source_crop_run"] == "crop_pose_001"
    assert manifest["datasets"][0]["keypoints_total"] == 4
    assert manifest["datasets"][0]["keypoints_successful"] == 3
    assert manifest["datasets"][0]["dish_design"] == "cedar"
    assert manifest["datasets"][0]["canvas_name"] == "DefaultScreen"
    assert (
        manifest["required_roi_pixel_contract_name"]
        == "orange_mono_pynvvc_luma_uint8_v1"
    )
    assert manifest["keypoint_contract_policy"]["status"] == "single_contract"
    assert manifest["keypoint_contract_policy"]["contract_counts"] == {
        "orange_mono_pynvvc_luma_uint8_v1": 1
    }
    assert (
        manifest["datasets"][0]["source_roi_pixel_contract_name"]
        == "orange_mono_pynvvc_luma_uint8_v1"
    )
    assert manifest["datasets"][0]["source_roi_read_mode"] == "materialized_crop_run"
    assert manifest["datasets"][0]["recording_id"] == "session_pose_001"
    assert manifest["datasets"][0]["leakage_group"] == {
        "id": "recording:session_pose_001",
        "source": "recording_fallback",
        "subject_ids": [],
        "recording_started_utc": None,
    }

    db = Registry(registry_path)
    row = db.conn.execute(
        "SELECT set_id, dataset_ids_json FROM training_sets WHERE set_id = ?",
        ("pose_pose_smoke_set_v001",),
    ).fetchone()
    db.close()
    assert row is not None
    assert row["set_id"] == "pose_pose_smoke_set_v001"
    assert json.loads(row["dataset_ids_json"]) == ["session_pose_001"]


def test_prepare_keypoint_from_registry_dry_run_prints_generated_artifacts(
    capsys, monkeypatch, tmp_path: Path
) -> None:
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


def test_prepare_keypoint_from_registry_auto_set_name_when_omitted(
    monkeypatch, tmp_path: Path
) -> None:
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
    assert set_name.startswith("cedar_defaultscreen_refined_gray_latest_traditional_")
    assert len(set_name.rsplit("_", 1)[-1]) == 8
    assert set_id == f"pose_{set_name}_v001"

    db = Registry(registry_path)
    row = db.conn.execute(
        "SELECT set_id FROM training_sets WHERE set_id = ?",
        (set_id,),
    ).fetchone()
    db.close()
    assert row is not None


def test_prepare_keypoint_from_registry_defaults_manifest_source_type_to_refined(
    monkeypatch, tmp_path: Path
) -> None:
    zarr_path = tmp_path / "pose_sample.zarr"
    _create_minimal_pose_zarr(zarr_path, detection_source_type="refined")
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
            "--input-format",
            "gray",
            "--out-config",
            str(out_config),
            "--out-manifest",
            str(out_manifest),
        ]
    )
    assert rc == 0

    cfg = yaml.safe_load(out_config.read_text(encoding="utf-8"))
    assert cfg["datasets"]["pose_sample"]["source_type"] == "refined"

    manifest = json.loads(out_manifest.read_text(encoding="utf-8"))
    assert manifest["source_type_requested"] == "refined"
    assert manifest["source_type"] == "refined"
    assert manifest["datasets"][0]["source_type_resolved"] == "refined"


def test_prepare_keypoint_from_registry_rejects_missing_keypoint_contract(
    monkeypatch, tmp_path: Path
) -> None:
    zarr_path = tmp_path / "pose_sample.zarr"
    _create_minimal_pose_zarr(zarr_path, source_roi_pixel_contract_name=None)
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry(registry_path, zarr_path)
    base_config_path = tmp_path / "pose_base.yaml"
    _write_base_pose_config(base_config_path)
    _mock_invocation_sources(monkeypatch)
    monkeypatch.setenv("PALETTE_TRAINING_DATASETS_ROOT", str(tmp_path / "datasets"))

    with pytest.raises(ValueError, match="missing keypoint ROI pixel contracts"):
        wrapper.main(
            [
                "--registry",
                str(registry_path),
                "--base-config",
                str(base_config_path),
                "--input-format",
                "gray",
                "--dry-run",
            ]
        )


def test_prepare_keypoint_from_registry_rejects_mixed_contracts_without_compatibility(
    monkeypatch, tmp_path: Path
) -> None:
    zarr_a = tmp_path / "pose_a.zarr"
    zarr_b = tmp_path / "pose_b.zarr"
    _create_minimal_pose_zarr(
        zarr_a,
        session_uuid="session_pose_a",
        source_roi_pixel_contract_name="orange_mono_pynvvc_luma_uint8_v1",
    )
    _create_minimal_pose_zarr(
        zarr_b,
        session_uuid="session_pose_b",
        source_roi_pixel_contract_name="nv12_luma_plane_uint8",
        source_roi_read_mode="flat_bin_roi_cache",
        source_roi_cache_backend="flat_bin_v1",
        input_mode_effective="tensor",
    )
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry(registry_path, zarr_a)
    _seed_registry(registry_path, zarr_b)
    base_config_path = tmp_path / "pose_base.yaml"
    _write_base_pose_config(base_config_path)
    _mock_invocation_sources(monkeypatch)
    monkeypatch.setenv("PALETTE_TRAINING_DATASETS_ROOT", str(tmp_path / "datasets"))

    with pytest.raises(ValueError, match="Mixed keypoint ROI pixel contracts detected"):
        wrapper.main(
            [
                "--registry",
                str(registry_path),
                "--base-config",
                str(base_config_path),
                "--input-format",
                "gray",
                "--dry-run",
            ]
        )


def test_prepare_keypoint_from_registry_allows_mixed_contracts_with_explicit_group(
    monkeypatch, tmp_path: Path
) -> None:
    zarr_a = tmp_path / "pose_a.zarr"
    zarr_b = tmp_path / "pose_b.zarr"
    _create_minimal_pose_zarr(
        zarr_a,
        session_uuid="session_pose_a",
        source_roi_pixel_contract_name="orange_mono_pynvvc_luma_uint8_v1",
    )
    _create_minimal_pose_zarr(
        zarr_b,
        session_uuid="session_pose_b",
        source_roi_pixel_contract_name="nv12_luma_plane_uint8",
        source_roi_read_mode="flat_bin_roi_cache",
        source_roi_cache_backend="flat_bin_v1",
        input_mode_effective="tensor",
    )
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry(registry_path, zarr_a)
    _seed_registry(registry_path, zarr_b)
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
            "--input-format",
            "gray",
            "--compatible-keypoint-contract",
            "orange_mono_pynvvc_luma_uint8_v1",
            "--compatible-keypoint-contract",
            "nv12_luma_plane_uint8",
            "--out-config",
            str(out_config),
            "--out-manifest",
            str(out_manifest),
        ]
    )
    assert rc == 0

    manifest = json.loads(out_manifest.read_text(encoding="utf-8"))
    policy = manifest["keypoint_contract_policy"]
    assert policy["status"] == "mixed_explicit_allowed"
    assert policy["required_roi_pixel_contract_name"] is None
    assert policy["contract_counts"] == {
        "nv12_luma_plane_uint8": 1,
        "orange_mono_pynvvc_luma_uint8_v1": 1,
    }
    assert policy["read_mode_counts"] == {
        "flat_bin_roi_cache": 1,
        "materialized_crop_run": 1,
    }
    assert policy["cache_backend_counts"] == {"<missing>": 1, "flat_bin_v1": 1}
    assert policy["input_mode_counts"] == {"<missing>": 1, "tensor": 1}
    assert manifest["query_filter"]["compatible_keypoint_contracts"] == [
        "orange_mono_pynvvc_luma_uint8_v1",
        "nv12_luma_plane_uint8",
    ]
    dataset_contracts = {
        dataset["dataset_id"]: dataset["required_roi_pixel_contract_name"]
        for dataset in manifest["datasets"]
    }
    assert dataset_contracts == {
        "session_pose_a": "orange_mono_pynvvc_luma_uint8_v1",
        "session_pose_b": "nv12_luma_plane_uint8",
    }


def test_prepare_keypoint_from_registry_rejects_non_refined_crop_lineage(
    monkeypatch, tmp_path: Path
) -> None:
    zarr_path = tmp_path / "pose_sample.zarr"
    _create_minimal_pose_zarr(zarr_path, detection_source_type="filtered")
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry(registry_path, zarr_path)
    base_config_path = tmp_path / "pose_base.yaml"
    _write_base_pose_config(base_config_path)
    _mock_invocation_sources(monkeypatch)
    monkeypatch.setenv("PALETTE_TRAINING_DATASETS_ROOT", str(tmp_path / "datasets"))

    with pytest.raises(
        ValueError, match=r"crop lineage detection_source_type in .*manual.*refined"
    ):
        wrapper.main(
            [
                "--registry",
                str(registry_path),
                "--base-config",
                str(base_config_path),
                "--input-format",
                "gray",
                "--dry-run",
            ]
        )


def test_prepare_keypoint_from_registry_requires_source_crop_run(
    monkeypatch, tmp_path: Path
) -> None:
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
                "--input-format",
                "gray",
                "--dry-run",
            ]
        )


def test_prepare_keypoint_from_registry_fails_on_roi_keypoint_row_mismatch(
    monkeypatch, tmp_path: Path
) -> None:
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
                "--input-format",
                "gray",
                "--dry-run",
            ]
        )


def test_prepare_keypoint_from_registry_fails_on_detection_success_row_mismatch(
    monkeypatch, tmp_path: Path
) -> None:
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
                "--input-format",
                "gray",
                "--dry-run",
            ]
        )


def test_prepare_keypoint_from_registry_enforces_review_status_and_quality(
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


def test_prepare_keypoint_from_registry_fails_when_review_status_missing(
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
        review_state=None,
        review_intended_use=None,
    )
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry(registry_path, zarr_path)
    base_config_path = tmp_path / "pose_base.yaml"
    _write_base_pose_config(base_config_path)
    _mock_invocation_sources(monkeypatch)
    monkeypatch.setenv("PALETTE_TRAINING_DATASETS_ROOT", str(tmp_path / "datasets"))

    with pytest.raises(
        SystemExit, match="No datasets remain after keypoint quality filtering"
    ):
        wrapper.main(
            [
                "--registry",
                str(registry_path),
                "--base-config",
                str(base_config_path),
                "--input-format",
                "gray",
                "--require-review-state",
                "approved",
                "--dry-run",
            ]
        )


def test_prepare_keypoint_from_registry_exclusion_is_nonfatal(
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
        review_state=None,
        review_intended_use=None,
    )
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry(registry_path, zarr_path)
    base_config_path = tmp_path / "pose_base.yaml"
    _write_base_pose_config(base_config_path)
    _mock_invocation_sources(monkeypatch)
    monkeypatch.setenv("PALETTE_TRAINING_DATASETS_ROOT", str(tmp_path / "datasets"))

    with pytest.raises(
        SystemExit, match="No datasets remain after keypoint quality filtering"
    ):
        wrapper.main(
            [
                "--registry",
                str(registry_path),
                "--base-config",
                str(base_config_path),
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
    assert dataset["keypoint_run_selector"] == "quality"
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

    with pytest.raises(
        SystemExit, match="No datasets remain after keypoint quality filtering"
    ):
        wrapper.main(
            [
                "--registry",
                str(registry_path),
                "--base-config",
                str(base_config_path),
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


def test_prepare_keypoint_from_registry_accepts_legacy_refined_group_name(
    monkeypatch, tmp_path: Path
) -> None:
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
    db.conn.execute(
        "UPDATE keypoint_quality SET usable_keypoints = usable_keypoints - 1;"
    )
    db.conn.commit()
    db.close()

    base_config_path = tmp_path / "pose_base.yaml"
    _write_base_pose_config(base_config_path)
    _mock_invocation_sources(monkeypatch)
    monkeypatch.setenv("PALETTE_TRAINING_DATASETS_ROOT", str(tmp_path / "datasets"))

    with pytest.raises(ValueError, match="usable_keypoints divergence"):
        wrapper.main(
            [
                "--registry",
                str(registry_path),
                "--base-config",
                str(base_config_path),
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


def test_prepare_keypoint_from_registry_fails_on_mixed_skeleton_signatures(
    monkeypatch, tmp_path: Path
) -> None:
    zarr_a = tmp_path / "pose_a.zarr"
    zarr_b = tmp_path / "pose_b.zarr"
    _create_minimal_pose_zarr(
        zarr_a,
        session_uuid="session_pose_a",
        skeleton_id="pose_skel_a",
        kpt_shape=(3, 3),
        pose_schema_name="schema_a",
    )
    _create_minimal_pose_zarr(
        zarr_b,
        session_uuid="session_pose_b",
        skeleton_id="pose_skel_b",
        kpt_shape=(3, 3),
        pose_schema_name="schema_b",
    )
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry(registry_path, zarr_a)
    _seed_registry(registry_path, zarr_b)
    base_config_path = tmp_path / "pose_base.yaml"
    _write_base_pose_config(base_config_path)
    _mock_invocation_sources(monkeypatch)
    monkeypatch.setenv("PALETTE_TRAINING_DATASETS_ROOT", str(tmp_path / "datasets"))

    with pytest.raises(ValueError) as excinfo:
        wrapper.main(
            [
                "--registry",
                str(registry_path),
                "--base-config",
                str(base_config_path),
                "--input-format",
                "gray",
                "--dry-run",
            ]
        )
    message = str(excinfo.value)
    assert "Mixed skeleton identities detected" in message
    assert "session_pose_a" in message
    assert "session_pose_b" in message
    assert "skeleton_id=pose_skel_a" in message
    assert "skeleton_id=pose_skel_b" in message


def test_prepare_keypoint_from_registry_prefers_refined_annotation_source_skeleton(
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
        skeleton_id="pose_skel_traditional_v1",
        kpt_shape=(3, 3),
        pose_schema_name="traditional_v1",
        refined_run_name="refined_pose_v2_001",
        refined_keypoint_count=5,
        refined_skeleton_id="pose_skel_traditional_v2",
        refined_runtime_kpt_shape=(5, 2),
        refined_pose_schema_name="traditional_v2",
    )
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
            "--input-format",
            "gray",
            "--require-review-state",
            "approved",
            "--require-review-intended-use",
            "training",
            "--min-usable-keypoints-rate",
            "0.70",
            "--out-config",
            str(out_config),
            "--out-manifest",
            str(out_manifest),
        ]
    )
    assert rc == 0

    cfg = yaml.safe_load(out_config.read_text(encoding="utf-8"))
    assert cfg["kpt_shape"] == [5, 3]

    manifest = json.loads(out_manifest.read_text(encoding="utf-8"))
    assert manifest["pose_schema"]["skeleton_id"] == "pose_skel_traditional_v2"
    assert manifest["pose_schema"]["kpt_shape"] == [5, 3]
    assert manifest["pose_schema"]["keypoint_labels"] == [
        "refined_kpt_0",
        "refined_kpt_1",
        "refined_kpt_2",
        "refined_kpt_3",
        "refined_kpt_4",
    ]
    assert manifest["pose_schema"]["skeleton"] == _test_skeleton(5)
    assert manifest["datasets"][0]["skeleton"] == _test_skeleton(5)
    dataset = manifest["datasets"][0]
    assert dataset["annotation_source_kind"] == "refined"
    assert dataset["annotation_source_parent"] == "refined_keypoints_runs"
    assert dataset["annotation_source_run"] == "refined_pose_v2_001"
    assert (
        dataset["keypoints_array_path"]
        == "refined_keypoints_runs/refined_pose_v2_001/keypoints_roi"
    )
    assert (
        dataset["detection_success_path"]
        == "refined_keypoints_runs/refined_pose_v2_001/usable_keypoints"
    )
    assert dataset["skeleton_id"] == "pose_skel_traditional_v2"
    assert dataset["kpt_shape"] == [5, 3]


def test_prepare_keypoint_from_registry_filters_to_requested_skeleton(
    monkeypatch, tmp_path: Path
) -> None:
    zarr_v1 = tmp_path / "pose_v1.zarr"
    zarr_v2 = tmp_path / "pose_v2.zarr"
    _create_minimal_pose_zarr(
        zarr_v1,
        session_uuid="session_pose_v1",
        create_refined_run=True,
        refined_usable_rows=3,
        review_state="approved",
        review_intended_use="training",
        skeleton_id="pose_skel_traditional_v1",
        kpt_shape=(3, 3),
        pose_schema_name="traditional_v1",
        refined_run_name="refined_pose_v1_001",
        refined_keypoint_count=3,
        refined_skeleton_id="pose_skel_traditional_v1",
        refined_runtime_kpt_shape=(3, 2),
        refined_pose_schema_name="traditional_v1",
    )
    _create_minimal_pose_zarr(
        zarr_v2,
        session_uuid="session_pose_v2",
        create_refined_run=True,
        refined_usable_rows=3,
        review_state="approved",
        review_intended_use="training",
        skeleton_id="pose_skel_traditional_v1",
        kpt_shape=(3, 3),
        pose_schema_name="traditional_v1",
        refined_run_name="refined_pose_v2_001",
        refined_keypoint_count=5,
        refined_skeleton_id="pose_skel_traditional_v2",
        refined_runtime_kpt_shape=(5, 2),
        refined_pose_schema_name="traditional_v2",
    )
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry(registry_path, zarr_v1)
    _seed_registry(registry_path, zarr_v2)
    base_config_path = tmp_path / "pose_base.yaml"
    _write_base_pose_config(base_config_path)
    _mock_invocation_sources(monkeypatch)
    monkeypatch.setenv("PALETTE_TRAINING_DATASETS_ROOT", str(tmp_path / "datasets"))

    out_manifest = tmp_path / "pose_manifest.json"
    rc = wrapper.main(
        [
            "--registry",
            str(registry_path),
            "--base-config",
            str(base_config_path),
            "--input-format",
            "gray",
            "--require-review-state",
            "approved",
            "--require-review-intended-use",
            "training",
            "--min-usable-keypoints-rate",
            "0.70",
            "--skeleton-id",
            "pose_skel_traditional_v2",
            "--out-manifest",
            str(out_manifest),
        ]
    )
    assert rc == 0

    manifest = json.loads(out_manifest.read_text(encoding="utf-8"))
    assert manifest["query_filter"]["skeleton_id"] == "pose_skel_traditional_v2"
    assert manifest["pose_schema"]["keypoint_labels"] == [
        "refined_kpt_0",
        "refined_kpt_1",
        "refined_kpt_2",
        "refined_kpt_3",
        "refined_kpt_4",
    ]
    assert [dataset["dataset_id"] for dataset in manifest["datasets"]] == [
        "session_pose_v2"
    ]
    assert manifest["quality_exclusions"] == [
        {
            "dataset_id": "session_pose_v1",
            "zarr_path": str(zarr_v1),
            "reason": "skeleton_id_mismatch:pose_skel_traditional_v1!=pose_skel_traditional_v2",
        }
    ]
