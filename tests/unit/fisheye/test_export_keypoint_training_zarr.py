"""Tests for keypoint merged-export skeleton identity guardrails."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import zarr

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.utils import export_keypoint_training_zarr as mod
from fisheye.utils.export_keypoint_training_zarr import (
    _discover_merge_sources,
    _format_skeleton_signature,
    _normalize_kpt_shape,
    _export_merged,
    validate_merged_keypoint_training_zarr,
)
from fisheye.shared.detect_reason_codec import write_reason_columns


def test_export_keypoint_skeleton_signature_helpers() -> None:
    assert _normalize_kpt_shape([3, 3]) == (3, 3)
    assert _normalize_kpt_shape(["3", 3]) == (3, 3)
    assert _normalize_kpt_shape([3, 0]) is None
    assert (
        _format_skeleton_signature(skeleton_id="pose_skel_shared", kpt_shape=(3, 3))
        == "skeleton_id=pose_skel_shared, kpt_shape=[3,3]"
    )


def _write_source_pose_zarr(
    path: Path,
    *,
    skeleton_id: str,
    detection_source_type: str = "refined",
    kpt_shape: tuple[int, int] = (3, 3),
    keypoint_count: int = 3,
    keypoint_labels: list[str] | None = None,
    refined_reasons: list[str] | None = None,
    refined_run_name: str = "refined_kp_pose_001",
    refined_skeleton_id: str | None = None,
    refined_runtime_kpt_shape: tuple[int, int] | None = None,
    refined_keypoint_count: int | None = None,
    refined_pose_schema_name: str | None = None,
) -> None:
    root = zarr.open_group(str(path), mode="w")

    crop_parent = root.create_group("crop_runs")
    crop_parent.attrs["latest"] = "crop_pose_001"
    crop = crop_parent.create_group("crop_pose_001")
    crop.attrs["detection_source_type"] = detection_source_type
    crop.attrs["roi_pixel_contract"] = {
        "name": "legacy_training_gray_uint8_v1",
        "channels": "gray",
        "dtype": "uint8",
    }
    crop.create_array(
        "roi_images",
        data=np.zeros((4, 16, 16), dtype=np.uint8),
        chunks=(2, 16, 16),
    )
    crop.create_array(
        "bbox_norm_coords",
        data=np.zeros((4, 4), dtype=np.float32),
        chunks=(4, 4),
    )
    crop.create_array(
        "source_refined_row_ids",
        data=np.array([1000, 1001, 1002, 1003], dtype=np.int64),
        chunks=(4,),
    )
    crop.create_array(
        "source_detect_row_index",
        data=np.array([2000, -1, 2002, 2003], dtype=np.int32),
        chunks=(4,),
    )

    kp_parent = root.create_group("keypoints_runs")
    kp_parent.attrs["latest"] = "kp_pose_001"
    kp = kp_parent.create_group("kp_pose_001")
    kp.attrs["source_crop_run"] = "crop_pose_001"
    kp.attrs["method"] = "traditional_pose"
    kp.attrs["skeleton_id"] = skeleton_id
    kp.attrs["kpt_shape"] = [int(kpt_shape[0]), int(kpt_shape[1])]
    kp.attrs["pose_schema"] = {
        "name": f"{skeleton_id}_schema",
        "skeleton_id": skeleton_id,
        "kpt_shape": [int(kpt_shape[0]), int(kpt_shape[1])],
    }
    kp.attrs["keypoint_labels"] = keypoint_labels or [f"kpt_{idx}" for idx in range(int(keypoint_count))]
    kp.create_array(
        "keypoints_roi",
        data=np.zeros((4, int(keypoint_count), 2), dtype=np.float32),
        chunks=(4, int(keypoint_count), 2),
    )
    kp.create_array(
        "detection_success",
        data=np.array([True, True, False, True], dtype=np.bool_),
        chunks=(4,),
    )
    if (
        refined_reasons is not None
        or refined_skeleton_id is not None
        or refined_runtime_kpt_shape is not None
        or refined_keypoint_count is not None
        or refined_pose_schema_name is not None
    ):
        refined_parent = root.create_group("refined_keypoints_runs")
        refined_parent.attrs["latest"] = refined_run_name
        refined = refined_parent.create_group(refined_run_name)
        refined.attrs["source_keypoints_run"] = "kp_pose_001"
        refined.attrs["source_crop_run"] = "crop_pose_001"
        refined.attrs["created_utc"] = "2026-02-27T00:00:00+00:00"
        resolved_refined_keypoint_count = int(
            refined_keypoint_count if refined_keypoint_count is not None else keypoint_count
        )
        resolved_refined_shape = (
            refined_runtime_kpt_shape
            if refined_runtime_kpt_shape is not None
            else (resolved_refined_keypoint_count, 2)
        )
        resolved_refined_skeleton_id = refined_skeleton_id or skeleton_id
        resolved_refined_schema_name = (
            refined_pose_schema_name or f"{resolved_refined_skeleton_id}_schema"
        )
        refined.attrs["skeleton_id"] = resolved_refined_skeleton_id
        refined.attrs["kpt_shape"] = [int(resolved_refined_shape[0]), int(resolved_refined_shape[1])]
        refined.attrs["pose_schema"] = {
            "name": resolved_refined_schema_name,
            "skeleton_id": resolved_refined_skeleton_id,
            "kpt_shape": [int(resolved_refined_shape[0]), int(resolved_refined_shape[1])],
        }
        refined.attrs["keypoint_labels"] = [
            f"refined_kpt_{idx}" for idx in range(resolved_refined_keypoint_count)
        ]
        refined.create_array(
            "keypoints_roi",
            data=np.zeros((4, resolved_refined_keypoint_count, 2), dtype=np.float32),
            chunks=(4, resolved_refined_keypoint_count, 2),
        )
        refined.create_array(
            "usable_keypoints",
            data=np.array([True, True, False, True], dtype=np.bool_),
            chunks=(4,),
        )
        write_reason_columns(
            refined,
            np.asarray(refined_reasons, dtype=object),
            chunk_size=4,
            include_reason_text=True,
            overwrite=True,
        )


def _manifest_for_single_source(path: Path) -> dict:
    return {
        "input_format": "gray",
        "source_type": "refined",
        "pose_schema": {
            "kpt_shape": [3, 3],
        },
        "datasets": [
            {
                "name": "dataset_single",
                "dataset_id": "dataset_single",
                "zarr_path": str(path),
                "input_format": "gray",
                "source_crop_run": "crop_pose_001",
                "keypoint_run": "kp_pose_001",
            }
        ],
    }


def _manifest_for_sources(path_a: Path, path_b: Path) -> dict:
    return {
        "input_format": "gray",
        "source_type": "refined",
        "pose_schema": {
            "kpt_shape": [3, 3],
        },
        "datasets": [
            {
                "name": "dataset_a",
                "dataset_id": "dataset_a",
                "zarr_path": str(path_a),
                "input_format": "gray",
                "source_crop_run": "crop_pose_001",
                "keypoint_run": "kp_pose_001",
            },
            {
                "name": "dataset_b",
                "dataset_id": "dataset_b",
                "zarr_path": str(path_b),
                "input_format": "gray",
                "source_crop_run": "crop_pose_001",
                "keypoint_run": "kp_pose_001",
            },
        ],
    }


def test_discover_merge_sources_accepts_single_skeleton_identity(tmp_path: Path) -> None:
    zarr_a = tmp_path / "source_a.zarr"
    zarr_b = tmp_path / "source_b.zarr"
    _write_source_pose_zarr(zarr_a, skeleton_id="pose_skel_shared")
    _write_source_pose_zarr(zarr_b, skeleton_id="pose_skel_shared")
    manifest = _manifest_for_sources(zarr_a, zarr_b)

    specs, layout = _discover_merge_sources(
        manifest,
        expected_input_format="gray",
        row_gate_policy="raw_success",
    )

    assert len(specs) == 2
    assert layout["skeleton_id"] == "pose_skel_shared"
    assert tuple(layout["kpt_shape"]) == (3, 3)


def test_discover_merge_sources_prefers_crop_resolved_source_type(tmp_path: Path) -> None:
    zarr_path = tmp_path / "source_pose.zarr"
    _write_source_pose_zarr(zarr_path, skeleton_id="pose_skel_shared", detection_source_type="refined")
    manifest = _manifest_for_single_source(zarr_path)
    manifest["source_type"] = "filtered"

    specs, _ = _discover_merge_sources(
        manifest,
        expected_input_format="gray",
        row_gate_policy="raw_success",
    )

    assert specs[0].source_type_resolved == "refined"


def test_discover_merge_sources_rejects_non_refined_crop_lineage(tmp_path: Path) -> None:
    zarr_path = tmp_path / "source_pose.zarr"
    _write_source_pose_zarr(zarr_path, skeleton_id="pose_skel_shared", detection_source_type="filtered")
    manifest = _manifest_for_single_source(zarr_path)

    with pytest.raises(ValueError, match="keypoint merged export requires crop lineage detection_source_type='refined'"):
        _discover_merge_sources(
            manifest,
            expected_input_format="gray",
            row_gate_policy="raw_success",
        )


def test_discover_merge_sources_accepts_required_roi_pixel_contract(tmp_path: Path) -> None:
    zarr_path = tmp_path / "source_pose.zarr"
    _write_source_pose_zarr(zarr_path, skeleton_id="pose_skel_shared")
    manifest = _manifest_for_single_source(zarr_path)
    manifest["required_roi_pixel_contract_name"] = "legacy_training_gray_uint8_v1"

    specs, _ = _discover_merge_sources(
        manifest,
        expected_input_format="gray",
        row_gate_policy="raw_success",
    )

    assert specs[0].roi_pixel_contract_name == "legacy_training_gray_uint8_v1"
    assert specs[0].roi_pixel_contract == {
        "name": "legacy_training_gray_uint8_v1",
        "channels": "gray",
        "dtype": "uint8",
    }


def test_discover_merge_sources_rejects_required_roi_pixel_contract_mismatch(tmp_path: Path) -> None:
    zarr_path = tmp_path / "source_pose.zarr"
    _write_source_pose_zarr(zarr_path, skeleton_id="pose_skel_shared")
    manifest = _manifest_for_single_source(zarr_path)
    manifest["required_roi_pixel_contract_name"] = "orange_mono_pynvvc_luma_uint8_v1"

    with pytest.raises(ValueError, match="ROI pixel contract mismatch"):
        _discover_merge_sources(
            manifest,
            expected_input_format="gray",
            row_gate_policy="raw_success",
        )


def test_discover_merge_sources_rejects_mixed_skeleton_identities(tmp_path: Path) -> None:
    zarr_a = tmp_path / "source_a.zarr"
    zarr_b = tmp_path / "source_b.zarr"
    _write_source_pose_zarr(zarr_a, skeleton_id="pose_skel_a")
    _write_source_pose_zarr(zarr_b, skeleton_id="pose_skel_b")
    manifest = _manifest_for_sources(zarr_a, zarr_b)

    with pytest.raises(ValueError) as excinfo:
        _discover_merge_sources(
            manifest,
            expected_input_format="gray",
            row_gate_policy="raw_success",
        )
    message = str(excinfo.value)
    assert "Mixed skeleton identities detected" in message
    assert "dataset_a" in message
    assert "dataset_b" in message
    assert "skeleton_id=pose_skel_a" in message
    assert "skeleton_id=pose_skel_b" in message


def test_discover_merge_sources_rejects_manifest_keypoint_label_mismatch(tmp_path: Path) -> None:
    zarr_path = tmp_path / "source_pose.zarr"
    _write_source_pose_zarr(
        zarr_path,
        skeleton_id="pose_skel_shared",
        keypoint_labels=["eye_left", "tail_tip", "bladder"],
    )
    manifest = _manifest_for_single_source(zarr_path)
    manifest["keypoint_labels"] = ["eye_left", "tail_tip", "swim_bladder"]
    manifest["datasets"][0]["keypoint_labels"] = ["tail_tip", "eye_left", "swim_bladder"]

    with pytest.raises(ValueError, match="dataset keypoint_labels"):
        _discover_merge_sources(
            manifest,
            expected_input_format="gray",
            row_gate_policy="raw_success",
        )


def test_discover_merge_sources_rejects_mixed_keypoint_label_sets(tmp_path: Path) -> None:
    zarr_a = tmp_path / "source_a.zarr"
    zarr_b = tmp_path / "source_b.zarr"
    labels_a = ["eye_left", "tail_tip", "bladder"]
    labels_b = ["eye_right", "tail_tip", "bladder"]
    _write_source_pose_zarr(zarr_a, skeleton_id="pose_skel_shared", keypoint_labels=labels_a)
    _write_source_pose_zarr(zarr_b, skeleton_id="pose_skel_shared", keypoint_labels=labels_b)
    manifest = _manifest_for_sources(zarr_a, zarr_b)

    with pytest.raises(ValueError, match="Mixed keypoint label sets detected"):
        _discover_merge_sources(
            manifest,
            expected_input_format="gray",
            row_gate_policy="raw_success",
        )


def test_discover_merge_sources_raw_success_plus_box_only_includes_tagged_rows(tmp_path: Path) -> None:
    zarr_path = tmp_path / "source_box_only.zarr"
    _write_source_pose_zarr(
        zarr_path,
        skeleton_id="pose_skel_shared",
        refined_reasons=["clean", "clean", "fish_present_no_keypoints", "clean"],
    )
    manifest = _manifest_for_single_source(zarr_path)

    specs, _layout = _discover_merge_sources(
        manifest,
        expected_input_format="gray",
        row_gate_policy="raw_success_plus_box_only",
    )

    assert len(specs) == 1
    spec = specs[0]
    assert spec.row_gate_policy == "raw_success_plus_box_only"
    assert spec.sample_count == 4
    assert spec.row_gate_raw_success_true == 3
    assert spec.row_gate_box_only_true == 1
    assert spec.row_gate_box_only_selected == 1
    assert spec.box_only_selected_mask is not None
    assert spec.box_only_selected_mask.tolist() == [False, False, True, False]


def test_discover_merge_sources_prefers_refined_annotation_skeleton_identity(tmp_path: Path) -> None:
    zarr_path = tmp_path / "source_refined_v2.zarr"
    _write_source_pose_zarr(
        zarr_path,
        skeleton_id="pose_skel_traditional_v1",
        kpt_shape=(3, 3),
        keypoint_count=3,
        refined_run_name="refined_kp_pose_v2_001",
        refined_skeleton_id="pose_skel_traditional_v2",
        refined_runtime_kpt_shape=(5, 2),
        refined_keypoint_count=5,
        refined_pose_schema_name="traditional_v2",
    )
    manifest = _manifest_for_single_source(zarr_path)
    manifest["pose_schema"] = {
        "skeleton_id": "pose_skel_traditional_v2",
        "kpt_shape": [5, 3],
    }
    manifest["datasets"][0]["refined_keypoint_run"] = "refined_kp_pose_v2_001"

    specs, layout = _discover_merge_sources(
        manifest,
        expected_input_format="gray",
        row_gate_policy="auto",
    )

    assert len(specs) == 1
    spec = specs[0]
    assert spec.row_gate_policy == "refined_usable"
    assert spec.row_gate_refined_run == "refined_kp_pose_v2_001"
    assert spec.keypoints_path == "refined_keypoints_runs/refined_kp_pose_v2_001/keypoints_roi"
    assert spec.success_path == "refined_keypoints_runs/refined_kp_pose_v2_001/usable_keypoints"
    assert spec.skeleton_id == "pose_skel_traditional_v2"
    assert spec.kpt_shape == (5, 3)
    assert layout["skeleton_id"] == "pose_skel_traditional_v2"
    assert tuple(layout["kpt_shape"]) == (5, 3)


def test_export_merged_uses_refined_keypoint_shape_for_written_arrays(tmp_path: Path) -> None:
    zarr_path = tmp_path / "source_refined_v2.zarr"
    _write_source_pose_zarr(
        zarr_path,
        skeleton_id="pose_skel_traditional_v1",
        kpt_shape=(3, 3),
        keypoint_count=3,
        refined_run_name="refined_kp_pose_v2_001",
        refined_skeleton_id="pose_skel_traditional_v2",
        refined_runtime_kpt_shape=(5, 2),
        refined_keypoint_count=5,
        refined_pose_schema_name="traditional_v2",
    )
    manifest = _manifest_for_single_source(zarr_path)
    manifest["set_id"] = "pose_set_v2"
    manifest["pose_schema"] = {
        "skeleton_id": "pose_skel_traditional_v2",
        "kpt_shape": [5, 3],
    }
    manifest["datasets"][0]["refined_keypoint_run"] = "refined_kp_pose_v2_001"

    manifest_path = tmp_path / "pose_set_v2.manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    out_zarr = tmp_path / "pose_set_v2_merged.zarr"

    result = _export_merged(
        manifest_payload=manifest,
        manifest_path=manifest_path,
        out_zarr=out_zarr,
        merged_dataset_id=None,
        overwrite=True,
        train_ratio=0.8,
        val_ratio=0.2,
        test_ratio=0.0,
        seed=42,
        copy_batch_size=128,
        row_gate_policy="auto",
        invocation={},
    )

    assert result.kpt_shape == (5, 3)

    root = zarr.open_group(str(out_zarr), mode="r")
    keypoints = np.asarray(root["keypoints_runs"][result.run_name]["keypoints_roi"][:], dtype=np.float32)
    assert keypoints.shape == (3, 5, 2)
    assert np.asarray(root["source_index/source_roi_idx"][:], dtype=np.int64).tolist() == [0, 1, 3]
    assert np.asarray(root["source_index/source_refined_row_ids"][:], dtype=np.int64).tolist() == [
        1000,
        1001,
        1003,
    ]
    assert np.asarray(root["source_index/source_detect_row_index"][:], dtype=np.int64).tolist() == [
        2000,
        -1,
        2003,
    ]

    summary = validate_merged_keypoint_training_zarr(out_zarr)
    assert summary["kpt_shape"] == [5, 3]


def _write_min_manifest(path: Path, *, set_id: str = "pose_set_v001") -> None:
    payload = {
        "set_id": set_id,
        "set_name": "pose_set",
        "input_format": "gray",
        "source_type": "refined",
        "datasets": [
            {
                "name": "dataset_a",
                "dataset_id": "dataset_a",
                "zarr_path": "/tmp/dataset_a.zarr",
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_main_auto_aggregates_keypoint_data_card_by_default(tmp_path: Path, monkeypatch) -> None:
    manifest_path = tmp_path / "pose_set_v001.manifest.json"
    _write_min_manifest(manifest_path)

    out_dir = tmp_path / "out"
    out_zarr = out_dir / "zarr" / "pose_set_v001_merged.zarr"
    out_manifest = out_dir / "pose_set_v001.manifest.json"

    merge_result = SimpleNamespace(
        input_format="gray",
        total_samples=4,
        source_specs=[],
        source_type="refined",
        run_name="merged_export_smoke",
    )

    monkeypatch.setattr(mod, "_export_merged", lambda **_kwargs: merge_result)
    monkeypatch.setattr(
        mod,
        "validate_merged_keypoint_training_zarr",
        lambda *_args, **_kwargs: {
            "total_samples": 4,
            "split_counts": {"train": 3, "val": 1, "test": 0},
            "source_count": 1,
        },
    )
    monkeypatch.setattr(mod, "_write_merge_summary", lambda **_kwargs: None)
    monkeypatch.setattr(
        mod,
        "_build_merged_manifest_payload",
        lambda **_kwargs: {"set_id": "pose_set_v001", "datasets": [], "merged_export": {"source_datasets": []}},
    )
    monkeypatch.setattr(mod, "_write_merged_config", lambda **_kwargs: None)

    class _RegistryPaths:
        path = tmp_path / "registry.sqlite"

    monkeypatch.setattr(
        mod.RegistryPaths,
        "from_env",
        classmethod(lambda cls, _cwd: _RegistryPaths()),  # type: ignore[misc]
    )

    captured: dict[str, object] = {}

    def _fake_card(*, cli: list[str], required: bool) -> int:
        captured["cli"] = list(cli)
        captured["required"] = bool(required)
        return 0

    monkeypatch.setattr(mod, "_run_keypoint_data_card_aggregation", _fake_card)

    rc = mod.main(
        [
            "--manifest",
            str(manifest_path),
            "--merge",
            "--out-dir",
            str(out_dir),
            "--out-zarr",
            str(out_zarr),
            "--overwrite",
        ]
    )
    assert rc == 0
    assert out_manifest.exists()

    assert captured["required"] is True
    card_cli = [str(item) for item in captured["cli"]]
    assert "--manifest" in card_cli and str(out_manifest) in card_cli
    assert "--merged-zarr" in card_cli and str(out_zarr) in card_cli
    assert "--registry" in card_cli
    assert str(tmp_path / "registry.sqlite") in card_cli


def test_main_no_aggregate_training_data_card_disables_aggregation(tmp_path: Path, monkeypatch) -> None:
    manifest_path = tmp_path / "pose_set_v001.manifest.json"
    _write_min_manifest(manifest_path)

    out_dir = tmp_path / "out"
    out_zarr = out_dir / "zarr" / "pose_set_v001_merged.zarr"

    merge_result = SimpleNamespace(
        input_format="gray",
        total_samples=2,
        source_specs=[],
        source_type="refined",
        run_name="merged_export_smoke",
    )

    monkeypatch.setattr(mod, "_export_merged", lambda **_kwargs: merge_result)
    monkeypatch.setattr(
        mod,
        "validate_merged_keypoint_training_zarr",
        lambda *_args, **_kwargs: {
            "total_samples": 2,
            "split_counts": {"train": 1, "val": 1, "test": 0},
            "source_count": 1,
        },
    )
    monkeypatch.setattr(mod, "_write_merge_summary", lambda **_kwargs: None)
    monkeypatch.setattr(
        mod,
        "_build_merged_manifest_payload",
        lambda **_kwargs: {"set_id": "pose_set_v001", "datasets": [], "merged_export": {"source_datasets": []}},
    )
    monkeypatch.setattr(mod, "_write_merged_config", lambda **_kwargs: None)

    called = {"card": False}

    def _fake_card(*, cli: list[str], required: bool) -> int:
        del cli, required
        called["card"] = True
        return 0

    monkeypatch.setattr(mod, "_run_keypoint_data_card_aggregation", _fake_card)

    rc = mod.main(
        [
            "--manifest",
            str(manifest_path),
            "--merge",
            "--out-dir",
            str(out_dir),
            "--out-zarr",
            str(out_zarr),
            "--overwrite",
            "--no-aggregate-training-data-card",
        ]
    )
    assert rc == 0
    assert called["card"] is False
