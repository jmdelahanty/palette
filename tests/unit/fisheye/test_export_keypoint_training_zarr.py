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
from fisheye.shared.zarr_helpers import open_zarr_group_direct


THREE_POINT_SKELETON = [[0, 1], [0, 2], [1, 2]]
FIVE_POINT_SKELETON = [[0, 1], [0, 2], [1, 2], [3, 1], [3, 2], [0, 4]]


def _skeleton_for_count(keypoint_count: int) -> list[list[int]]:
    if int(keypoint_count) == 3:
        return [list(edge) for edge in THREE_POINT_SKELETON]
    if int(keypoint_count) == 5:
        return [list(edge) for edge in FIVE_POINT_SKELETON]
    return [[idx, idx + 1] for idx in range(max(0, int(keypoint_count) - 1))]


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
    strict_v2_names: bool = False,
    roi_hw: tuple[int, int] = (16, 16),
    roi_value: int = 0,
    keypoint_xy: tuple[float, float] = (0.0, 0.0),
    keypoint_dtype: np.dtype = np.dtype(np.float32),
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
        data=np.full((4, *roi_hw), roi_value, dtype=np.uint8),
        chunks=(2, *roi_hw),
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
    raw_video = root.create_group("raw_video")
    raw_video.create_array(
        "original_frame_indices",
        data=np.array([100, 101, 102, 103], dtype=np.int64),
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
        "edges": _skeleton_for_count(keypoint_count),
    }
    kp.attrs["keypoint_skeleton"] = _skeleton_for_count(keypoint_count)
    kp.attrs["keypoint_labels"] = keypoint_labels or [f"kpt_{idx}" for idx in range(int(keypoint_count))]
    kp.create_array(
        "keypoints_roi",
        data=np.broadcast_to(
            np.asarray(keypoint_xy, dtype=keypoint_dtype),
            (4, int(keypoint_count), 2),
        ).copy(),
        chunks=(4, int(keypoint_count), 2),
    )
    kp.create_array(
        "pose_success" if strict_v2_names else "detection_success",
        data=np.array([True, True, False, True], dtype=np.bool_),
        chunks=(4,),
    )
    kp.create_array(
        "frame_indices",
        data=np.array([0, 1, 2, 3], dtype=np.int64),
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
        if strict_v2_names:
            refined.attrs["source_bindings"] = {
                "schema_id": "palette.refined_keypoint.source_bindings",
                "schema_version": 2,
                "raw_keypoint_snapshot": {
                    "stage": "keypoints",
                    "run_id": "kp_pose_001",
                    "run_path": "keypoints_runs/kp_pose_001",
                },
            }
        else:
            refined.attrs["source_keypoints_run"] = "kp_pose_001"
        refined.attrs["source_crop_run"] = "crop_pose_001"
        refined.attrs["created_utc"] = "2026-02-27T00:00:00+00:00"
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
        resolved_refined_skeleton_id = refined_skeleton_id or skeleton_id
        resolved_refined_schema_name = (
            refined_pose_schema_name or f"{resolved_refined_skeleton_id}_schema"
        )
        refined.attrs["skeleton_id"] = resolved_refined_skeleton_id
        refined.attrs["kpt_shape"] = [
            int(resolved_refined_shape[0]),
            int(resolved_refined_shape[1]),
        ]
        refined.attrs["pose_schema"] = {
            "name": resolved_refined_schema_name,
            "skeleton_id": resolved_refined_skeleton_id,
            "kpt_shape": [int(resolved_refined_shape[0]), int(resolved_refined_shape[1])],
            "edges": _skeleton_for_count(resolved_refined_keypoint_count),
        }
        refined.attrs["keypoint_skeleton"] = _skeleton_for_count(
            resolved_refined_keypoint_count
        )
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
        refined.create_array(
            "frame_indices",
            data=np.array([0, 1, 2, 3], dtype=np.int64),
            chunks=(4,),
        )
        write_reason_columns(
            refined,
            np.asarray(refined_reasons, dtype=object),
            chunk_size=4,
            overwrite=True,
        )


def _manifest_for_single_source(path: Path) -> dict:
    return {
        "input_format": "gray",
        "source_type": "refined",
        "pose_schema": {
            "kpt_shape": [3, 3],
            "skeleton": THREE_POINT_SKELETON,
        },
        "datasets": [
            {
                "name": "dataset_single",
                "dataset_id": "dataset_single",
                "recording_id": "recording_single",
                "leakage_group": {
                    "id": "recording:recording_single",
                    "source": "recording_fallback",
                },
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
            "skeleton": THREE_POINT_SKELETON,
        },
        "datasets": [
            {
                "name": "dataset_a",
                "dataset_id": "dataset_a",
                "recording_id": "recording_a",
                "leakage_group": {
                    "id": "recording:recording_a",
                    "source": "recording_fallback",
                },
                "zarr_path": str(path_a),
                "input_format": "gray",
                "source_crop_run": "crop_pose_001",
                "keypoint_run": "kp_pose_001",
            },
            {
                "name": "dataset_b",
                "dataset_id": "dataset_b",
                "recording_id": "recording_b",
                "leakage_group": {
                    "id": "recording:recording_b",
                    "source": "recording_fallback",
                },
                "zarr_path": str(path_b),
                "input_format": "gray",
                "source_crop_run": "crop_pose_001",
                "keypoint_run": "kp_pose_001",
            },
        ],
    }


def test_discover_merge_sources_accepts_single_skeleton_identity(
    tmp_path: Path,
) -> None:
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
    assert layout["skeleton"] == THREE_POINT_SKELETON
    assert specs[0].skeleton == THREE_POINT_SKELETON


def test_discover_merge_sources_requires_exact_manifest_skeleton(tmp_path: Path) -> None:
    zarr_path = tmp_path / "source_pose.zarr"
    _write_source_pose_zarr(zarr_path, skeleton_id="pose_skel_shared")
    manifest = _manifest_for_single_source(zarr_path)
    manifest["pose_schema"].pop("skeleton")
    with pytest.raises(ValueError, match="exact ordered skeleton edge list"):
        _discover_merge_sources(
            manifest,
            expected_input_format="gray",
            row_gate_policy="raw_success",
        )


def test_discover_merge_sources_rejects_ordered_skeleton_disagreement(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "source_pose.zarr"
    _write_source_pose_zarr(zarr_path, skeleton_id="pose_skel_shared")
    manifest = _manifest_for_single_source(zarr_path)
    manifest["pose_schema"]["skeleton"] = [[0, 2], [0, 1], [1, 2]]
    with pytest.raises(ValueError, match="exact ordered skeleton edges"):
        _discover_merge_sources(
            manifest,
            expected_input_format="gray",
            row_gate_policy="raw_success",
        )


def test_discover_merge_sources_prefers_crop_resolved_source_type(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "source_pose.zarr"
    _write_source_pose_zarr(
        zarr_path, skeleton_id="pose_skel_shared", detection_source_type="refined"
    )
    manifest = _manifest_for_single_source(zarr_path)
    manifest["source_type"] = "filtered"

    specs, _ = _discover_merge_sources(
        manifest,
        expected_input_format="gray",
        row_gate_policy="raw_success",
    )

    assert specs[0].source_type_resolved == "refined"


def test_discover_merge_sources_rejects_non_refined_crop_lineage(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "source_pose.zarr"
    _write_source_pose_zarr(
        zarr_path, skeleton_id="pose_skel_shared", detection_source_type="filtered"
    )
    manifest = _manifest_for_single_source(zarr_path)

    with pytest.raises(
        ValueError, match=r"crop lineage detection_source_type in .*manual.*refined"
    ):
        _discover_merge_sources(
            manifest,
            expected_input_format="gray",
            row_gate_policy="raw_success",
        )


def test_discover_merge_sources_accepts_required_roi_pixel_contract(
    tmp_path: Path,
) -> None:
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


def test_discover_merge_sources_reads_mutable_run_metadata_directly(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "source_pose.zarr"
    _write_source_pose_zarr(zarr_path, skeleton_id="pose_skel_shared")
    zarr.consolidate_metadata(str(zarr_path))

    crop = open_zarr_group_direct(zarr_path / "crop_runs" / "crop_pose_001", mode="r+")
    crop.attrs["roi_pixel_contract"] = {
        "name": "reviewed_gray_uint8_v2",
        "channels": "gray",
        "dtype": "uint8",
    }
    manifest = _manifest_for_single_source(zarr_path)
    manifest["required_roi_pixel_contract_name"] = "reviewed_gray_uint8_v2"

    specs, _ = _discover_merge_sources(
        manifest,
        expected_input_format="gray",
        row_gate_policy="raw_success",
    )

    assert specs[0].roi_pixel_contract_name == "reviewed_gray_uint8_v2"


def test_discover_merge_sources_honors_exact_refined_run_hidden_from_snapshot(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "source_pose.zarr"
    _write_source_pose_zarr(
        zarr_path,
        skeleton_id="pose_skel_shared",
        refined_skeleton_id="pose_skel_shared",
        refined_run_name="refined_visible_001",
    )
    zarr.consolidate_metadata(str(zarr_path))

    parent = open_zarr_group_direct(zarr_path / "refined_keypoints_runs", mode="r+")
    refined = parent.create_group("refined_reviewed_002")
    refined.attrs.update(
        {
            "source_keypoints_run": "kp_pose_001",
            "source_crop_run": "crop_pose_001",
            "created_utc": "2026-08-07T00:00:00+00:00",
            "skeleton_id": "pose_skel_shared",
            "kpt_shape": [3, 2],
            "keypoint_labels": ["kpt_0", "kpt_1", "kpt_2"],
        }
    )
    refined.create_array(
        "keypoints_roi",
        data=np.ones((4, 3, 2), dtype=np.float32),
        chunks=(4, 3, 2),
    )
    refined.create_array(
        "usable_keypoints",
        data=np.array([True, True, False, True], dtype=np.bool_),
        chunks=(4,),
    )
    refined.create_array(
        "frame_indices",
        data=np.arange(4, dtype=np.int64),
        chunks=(4,),
    )

    manifest = _manifest_for_single_source(zarr_path)
    manifest["datasets"][0].update(
        {
            "annotation_source_parent": "refined_keypoints_runs",
            "refined_keypoint_run": "refined_reviewed_002",
            "keypoints_array_path": (
                "refined_keypoints_runs/refined_reviewed_002/keypoints_roi"
            ),
            "detection_success_path": (
                "refined_keypoints_runs/refined_reviewed_002/usable_keypoints"
            ),
        }
    )

    specs, _ = _discover_merge_sources(
        manifest,
        expected_input_format="gray",
        row_gate_policy="refined_usable",
    )

    assert specs[0].keypoints_path == (
        "refined_keypoints_runs/refined_reviewed_002/keypoints_roi"
    )
    assert specs[0].row_gate_refined_run == "refined_reviewed_002"


def test_discover_merge_sources_rejects_required_roi_pixel_contract_mismatch(
    tmp_path: Path,
) -> None:
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


def test_discover_merge_sources_rejects_mixed_skeleton_identities(
    tmp_path: Path,
) -> None:
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


def test_discover_merge_sources_rejects_manifest_keypoint_label_mismatch(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "source_pose.zarr"
    _write_source_pose_zarr(
        zarr_path,
        skeleton_id="pose_skel_shared",
        keypoint_labels=["eye_left", "tail_tip", "bladder"],
    )
    manifest = _manifest_for_single_source(zarr_path)
    manifest["keypoint_labels"] = ["eye_left", "tail_tip", "swim_bladder"]
    manifest["datasets"][0]["keypoint_labels"] = [
        "tail_tip",
        "eye_left",
        "swim_bladder",
    ]

    with pytest.raises(ValueError, match="dataset keypoint_labels"):
        _discover_merge_sources(
            manifest,
            expected_input_format="gray",
            row_gate_policy="raw_success",
        )


def test_discover_merge_sources_rejects_mixed_keypoint_label_sets(
    tmp_path: Path,
) -> None:
    zarr_a = tmp_path / "source_a.zarr"
    zarr_b = tmp_path / "source_b.zarr"
    labels_a = ["eye_left", "tail_tip", "bladder"]
    labels_b = ["eye_right", "tail_tip", "bladder"]
    _write_source_pose_zarr(
        zarr_a, skeleton_id="pose_skel_shared", keypoint_labels=labels_a
    )
    _write_source_pose_zarr(
        zarr_b, skeleton_id="pose_skel_shared", keypoint_labels=labels_b
    )
    manifest = _manifest_for_sources(zarr_a, zarr_b)

    with pytest.raises(ValueError, match="Mixed keypoint label sets detected"):
        _discover_merge_sources(
            manifest,
            expected_input_format="gray",
            row_gate_policy="raw_success",
        )


def test_discover_merge_sources_raw_success_plus_box_only_includes_tagged_rows(
    tmp_path: Path,
) -> None:
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


def test_discover_merge_sources_prefers_refined_annotation_skeleton_identity(
    tmp_path: Path,
) -> None:
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
        "skeleton": FIVE_POINT_SKELETON,
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
    assert (
        spec.keypoints_path
        == "refined_keypoints_runs/refined_kp_pose_v2_001/keypoints_roi"
    )
    assert (
        spec.success_path
        == "refined_keypoints_runs/refined_kp_pose_v2_001/usable_keypoints"
    )
    assert spec.skeleton_id == "pose_skel_traditional_v2"
    assert spec.kpt_shape == (5, 3)
    assert layout["skeleton_id"] == "pose_skel_traditional_v2"
    assert tuple(layout["kpt_shape"]) == (5, 3)


def test_discover_merge_sources_supports_strict_v2_success_and_source_binding(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "source_refined_strict_v2.zarr"
    _write_source_pose_zarr(
        zarr_path,
        skeleton_id="pose_skel_traditional_v2",
        kpt_shape=(5, 3),
        keypoint_count=5,
        refined_run_name="refined_kp_pose_v2_001",
        refined_skeleton_id="pose_skel_traditional_v2",
        refined_runtime_kpt_shape=(5, 2),
        refined_keypoint_count=5,
        refined_pose_schema_name="traditional_v2",
        strict_v2_names=True,
    )
    annotation_zarr = tmp_path / "reviewed_compaction.zarr"
    annotation_root = zarr.open_group(str(annotation_zarr), mode="w")
    annotation_parent = annotation_root.create_group("refined_keypoints_runs")
    annotation = annotation_parent.create_group("refined_kp_pose_v2_001")
    annotation.attrs.update(
        {
            "source_bindings": {
                "schema_id": "palette.refined_keypoint.source_bindings",
                "schema_version": 2,
                "raw_keypoint_snapshot": {
                    "stage": "keypoints",
                    "run_id": "kp_pose_001",
                    "run_path": "keypoints_runs/kp_pose_001",
                },
            },
            "skeleton_id": "pose_skel_traditional_v2",
            "kpt_shape": [5, 2],
            "pose_schema": {
                "name": "traditional_v2",
                "skeleton_id": "pose_skel_traditional_v2",
                "kpt_shape": [5, 2],
                "edges": FIVE_POINT_SKELETON,
            },
            "keypoint_skeleton": FIVE_POINT_SKELETON,
            "keypoint_labels": [f"refined_kpt_{idx}" for idx in range(5)],
        }
    )
    annotation.create_array(
        "keypoints_roi",
        data=np.zeros((4, 5, 2), dtype=np.float32),
        chunks=(4, 5, 2),
    )
    annotation.create_array(
        "usable_keypoints",
        data=np.array([True, False, False, True], dtype=np.bool_),
        chunks=(4,),
    )
    manifest = _manifest_for_single_source(zarr_path)
    manifest["pose_schema"] = {
        "skeleton_id": "pose_skel_traditional_v2",
        "kpt_shape": [5, 3],
        "skeleton": FIVE_POINT_SKELETON,
    }
    manifest["datasets"][0]["refined_keypoint_run"] = "refined_kp_pose_v2_001"
    manifest["datasets"][0]["annotation_zarr_path"] = str(annotation_zarr)

    specs, _layout = _discover_merge_sources(
        manifest,
        expected_input_format="gray",
        row_gate_policy="refined_usable",
    )

    assert len(specs) == 1
    spec = specs[0]
    assert spec.annotation_zarr == annotation_zarr
    assert spec.sample_count == 2
    assert spec.row_gate_policy == "refined_usable"
    assert spec.row_gate_refined_run == "refined_kp_pose_v2_001"
    assert spec.keypoints_path == (
        "refined_keypoints_runs/refined_kp_pose_v2_001/keypoints_roi"
    )
    assert spec.success_path == (
        "refined_keypoints_runs/refined_kp_pose_v2_001/usable_keypoints"
    )


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
        "skeleton": FIVE_POINT_SKELETON,
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
    assert result.skeleton == FIVE_POINT_SKELETON

    root = zarr.open_group(str(out_zarr), mode="a", use_consolidated=False)
    keypoints = np.asarray(root["keypoints_runs"][result.run_name]["keypoints_roi"][:], dtype=np.float32)
    assert keypoints.shape == (3, 5, 2)
    assert np.asarray(
        root["source_index/source_roi_idx"][:], dtype=np.int64
    ).tolist() == [0, 1, 3]
    assert np.asarray(
        root["source_index/source_refined_row_ids"][:], dtype=np.int64
    ).tolist() == [
        1000,
        1001,
        1003,
    ]
    assert np.asarray(
        root["source_index/source_detect_row_index"][:], dtype=np.int64
    ).tolist() == [
        2000,
        -1,
        2003,
    ]

    summary = validate_merged_keypoint_training_zarr(out_zarr)
    assert summary["kpt_shape"] == [5, 3]
    assert summary["skeleton"] == FIVE_POINT_SKELETON
    keypoint_group = root[f"keypoints_runs/{result.run_name}"]
    assert keypoint_group.attrs["keypoint_skeleton"] == FIVE_POINT_SKELETON
    assert keypoint_group.attrs["pose_schema"]["skeleton"] == FIVE_POINT_SKELETON
    training_export = dict(root.attrs["training_export"])
    assert training_export["schema_version"] == "2.0.0"
    assert training_export["skeleton"] == FIVE_POINT_SKELETON
    assert training_export["pose_schema"]["skeleton"] == FIVE_POINT_SKELETON

    training_export["skeleton"] = [
        FIVE_POINT_SKELETON[1],
        FIVE_POINT_SKELETON[0],
        *FIVE_POINT_SKELETON[2:],
    ]
    root.attrs["training_export"] = training_export
    with pytest.raises(ValueError, match="pose_schema.skeleton disagrees"):
        validate_merged_keypoint_training_zarr(out_zarr)


def test_export_merged_keeps_mixed_lineage_out_of_surface_source_type(
    tmp_path: Path,
) -> None:
    zarr_a = tmp_path / "source_refined.zarr"
    zarr_b = tmp_path / "source_manual.zarr"
    _write_source_pose_zarr(
        zarr_a, skeleton_id="pose_skel_shared", detection_source_type="refined"
    )
    _write_source_pose_zarr(
        zarr_b, skeleton_id="pose_skel_shared", detection_source_type="manual"
    )
    manifest = _manifest_for_sources(zarr_a, zarr_b)
    manifest["source_type_requested"] = "refined"

    manifest_path = tmp_path / "pose_set_mixed_lineage.manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    out_zarr = tmp_path / "pose_set_mixed_lineage_merged.zarr"

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
        row_gate_policy="raw_success",
        invocation={},
    )

    assert result.source_type == "refined"
    assert result.source_type_counts == {"manual": 1, "refined": 1}

    root = zarr.open_group(str(out_zarr), mode="r", use_consolidated=False)
    crop = root[f"crop_runs/{result.run_name}"]
    assert crop.attrs["detection_source_type"] == "refined"
    assert crop.attrs["source_type_resolved_counts"] == {"manual": 1, "refined": 1}
    training_export = dict(root.attrs["training_export"])
    assert training_export["source_type_requested"] == "refined"
    assert training_export["source_type_resolved"] == "refined"
    assert training_export["source_type_resolved_counts"] == {"manual": 1, "refined": 1}


def test_export_v2_pads_without_resize_and_groups_splits_by_source(
    tmp_path: Path,
) -> None:
    zarr_a = tmp_path / "source_16.zarr"
    zarr_b = tmp_path / "source_12.zarr"
    _write_source_pose_zarr(
        zarr_a,
        skeleton_id="pose_skel_shared",
        roi_hw=(16, 16),
        roi_value=3,
        keypoint_xy=(2.0, 3.0),
    )
    _write_source_pose_zarr(
        zarr_b,
        skeleton_id="pose_skel_shared",
        roi_hw=(12, 12),
        roi_value=7,
        keypoint_xy=(1.0, 2.0),
    )
    manifest = _manifest_for_sources(zarr_a, zarr_b)
    manifest["set_id"] = "pose_set_padded_v2"
    manifest_path = tmp_path / "pose_set_padded_v2.manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    out_zarr = tmp_path / "pose_set_padded_v2_merged.zarr"

    result = _export_merged(
        manifest_payload=manifest,
        manifest_path=manifest_path,
        out_zarr=out_zarr,
        merged_dataset_id=None,
        overwrite=True,
        train_ratio=0.5,
        val_ratio=0.5,
        test_ratio=0.0,
        seed=42,
        copy_batch_size=2,
        row_gate_policy="raw_success",
        invocation={},
        roi_transform_mode="pad_to_shape",
        target_roi_hw=(20, 20),
        split_unit="source_dataset",
    )

    root = zarr.open_group(str(out_zarr), mode="r", use_consolidated=False)
    roi = np.asarray(root[f"crop_runs/{result.run_name}/roi_images"][:])
    keypoints = np.asarray(root[f"keypoints_runs/{result.run_name}/keypoints_roi"][:])
    source_idx = np.asarray(root["source_index/source_dataset_idx"][:], dtype=np.int64)
    train_idx = np.asarray(root["splits/train_indices"][:], dtype=np.int64)
    val_idx = np.asarray(root["splits/val_indices"][:], dtype=np.int64)

    assert roi.shape == (6, 20, 20)
    assert np.all(roi[0, 2:18, 2:18] == 3)
    assert np.all(roi[0, :2] == 0)
    assert np.all(roi[3, 4:16, 4:16] == 7)
    assert np.all(roi[3, :4] == 0)
    assert keypoints[0, 0].tolist() == [4.0, 5.0]
    assert keypoints[3, 0].tolist() == [5.0, 6.0]
    assert set(source_idx[train_idx].tolist()).isdisjoint(
        set(source_idx[val_idx].tolist())
    )
    assert root["splits"].attrs["strategy"] == "source_dataset_grouped"

    transforms = [
        json.loads(str(value))
        for value in np.asarray(
            root["source_index/source_roi_transform_json"][:], dtype=object
        )
    ]
    assert transforms[0]["pad_before_yx"] == [2, 2]
    assert transforms[1]["pad_before_yx"] == [4, 4]
    assert all(transform["resize_applied"] is False for transform in transforms)

    summary = validate_merged_keypoint_training_zarr(out_zarr)
    assert summary["total_samples"] == 6


def test_export_v2_rejects_roi_larger_than_padding_target(tmp_path: Path) -> None:
    source = tmp_path / "source_24.zarr"
    _write_source_pose_zarr(source, skeleton_id="pose_skel_shared", roi_hw=(24, 24))
    manifest = _manifest_for_single_source(source)

    with pytest.raises(ValueError, match="exceeds padding target"):
        _discover_merge_sources(
            manifest,
            expected_input_format="gray",
            row_gate_policy="raw_success",
            roi_transform_mode="pad_to_shape",
            target_roi_hw=(20, 20),
        )


def test_immutable_v3_publication_is_consolidated_and_selector_ineligible(
    tmp_path: Path,
) -> None:
    zarr_a = tmp_path / "source_a.zarr"
    zarr_b = tmp_path / "source_b.zarr"
    _write_source_pose_zarr(zarr_a, skeleton_id="pose_skel_shared", roi_hw=(16, 16))
    _write_source_pose_zarr(zarr_b, skeleton_id="pose_skel_shared", roi_hw=(12, 12))
    manifest = _manifest_for_sources(zarr_a, zarr_b)
    manifest["set_id"] = "immutable_pose_v2"
    manifest_path = tmp_path / "immutable_pose_v2.manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    scratch = tmp_path / "bounded_scratch"
    scratch.mkdir()
    target = tmp_path / "published" / "immutable_pose_v2.zarr"

    result = mod._publish_immutable_merged(
        scratch_root=scratch,
        manifest_payload=manifest,
        manifest_path=manifest_path,
        out_zarr=target,
        merged_dataset_id="immutable_pose_v2",
        train_ratio=0.5,
        val_ratio=0.5,
        test_ratio=0.0,
        seed=42,
        copy_batch_size=2,
        row_gate_policy="raw_success",
        invocation={},
        roi_transform_mode="pad_to_shape",
        target_roi_hw=(20, 20),
        split_unit="leakage_group",
    )

    assert result.total_samples == 6
    direct = zarr.open_group(str(target), mode="r", use_consolidated=False)
    consolidated = zarr.open_group(str(target), mode="r", use_consolidated=True)
    assert direct.attrs["training_artifact_status"] == "complete"
    assert direct.attrs["training_artifact_mutability"] == "immutable"
    assert direct.attrs["stage_selector_eligible"] is False
    assert direct.attrs["registry_activation"] == "deferred"
    assert direct.attrs["training_export"]["storage"]["schema_version"] == 3
    assert direct["splits"].attrs["strategy"] == "biological_acquisition_grouped_v1"
    assert "source_frame_idx" not in direct["source_index"]
    assert np.asarray(
        direct["source_index/source_sample_row_index"][:], dtype=np.int64
    ).tolist() == [0, 1, 3, 0, 1, 3]
    assert np.asarray(
        direct["source_index/source_acquisition_frame_index"][:], dtype=np.int64
    ).tolist() == [100, 101, 103, 100, 101, 103]
    publication = consolidated.attrs["immutable_training_publication"]
    assert publication["task"] == "keypoints"
    assert publication["schema_version"] == 2
    published_source_manifest = target.with_name(f"{target.stem}.source_manifest.json")
    assert published_source_manifest.read_bytes() == manifest_path.read_bytes()
    assert publication["manifest_path"] == str(published_source_manifest.resolve())
    assert publication["manifest_sha256"] == mod._sha256(published_source_manifest)
    publication_validation = publication["validation"]
    assert publication_validation["published_zarr_path"] == str(target.resolve())
    assert "zarr_path" not in publication_validation
    assert not list(target.parent.glob(f".{target.name}.publish_tmp.*"))
    assert not list(
        target.parent.glob(f".{published_source_manifest.name}.publish_tmp.*")
    )


def test_v3_leakage_group_keeps_related_sources_in_one_split(tmp_path: Path) -> None:
    sources = [tmp_path / f"source_{name}.zarr" for name in ("a", "b", "c")]
    for source in sources:
        _write_source_pose_zarr(source, skeleton_id="pose_skel_shared")
    manifest = _manifest_for_sources(sources[0], sources[1])
    manifest["datasets"].append(
        {
            "name": "dataset_c",
            "dataset_id": "dataset_c",
            "recording_id": "recording_c",
            "leakage_group": {
                "id": "subject:subject_c",
                "source": "registered_subject",
            },
            "zarr_path": str(sources[2]),
            "input_format": "gray",
            "source_crop_run": "crop_pose_001",
            "keypoint_run": "kp_pose_001",
        }
    )
    for dataset in manifest["datasets"][:2]:
        dataset["leakage_group"] = {
            "id": "subject:shared_subject",
            "source": "registered_subject",
        }
    manifest_path = tmp_path / "manifest_v3.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "merged_v3.zarr"

    result = _export_merged(
        manifest_payload=manifest,
        manifest_path=manifest_path,
        out_zarr=output,
        merged_dataset_id="merged_v3",
        overwrite=False,
        train_ratio=0.5,
        val_ratio=0.5,
        test_ratio=0.0,
        seed=42,
        copy_batch_size=2,
        row_gate_policy="raw_success",
        invocation={},
        split_unit="leakage_group",
        use_storage_contract_v2=True,
    )

    root = zarr.open_group(str(output), mode="r", use_consolidated=False)
    source_idx = np.asarray(root["source_index/source_dataset_idx"][:], dtype=np.int64)
    group_ids = np.asarray(root["source_index/leakage_group_id"][:], dtype=object)
    train = np.asarray(root["splits/train_indices"][:], dtype=np.int64)
    validation = np.asarray(root["splits/val_indices"][:], dtype=np.int64)
    train_groups = set(str(value) for value in group_ids[source_idx[train]].tolist())
    validation_groups = set(
        str(value) for value in group_ids[source_idx[validation]].tolist()
    )

    assert result.split_strategy == "biological_acquisition_grouped_v1"
    assert train_groups.isdisjoint(validation_groups)
    assert set(source_idx[train].tolist()).issuperset({0, 1}) or set(
        source_idx[validation].tolist()
    ).issuperset({0, 1})
    assert validate_merged_keypoint_training_zarr(output)["total_samples"] == 9

    writable = zarr.open_group(str(output), mode="a", use_consolidated=False)
    writable["source_index/leakage_group_id"][1] = "subject:subject_c"
    with pytest.raises(ValueError, match="share a group"):
        validate_merged_keypoint_training_zarr(output)


def test_checked_dtype_policy_normalizes_float64_to_float32_with_receipt(
    tmp_path: Path,
) -> None:
    source_float64 = tmp_path / "source_float64.zarr"
    source_float32 = tmp_path / "source_float32.zarr"
    _write_source_pose_zarr(
        source_float64,
        skeleton_id="pose_skel_shared",
        keypoint_xy=(0.123456789, 15.987654321),
        keypoint_dtype=np.dtype(np.float64),
    )
    _write_source_pose_zarr(
        source_float32,
        skeleton_id="pose_skel_shared",
        keypoint_xy=(2.5, 7.25),
        keypoint_dtype=np.dtype(np.float32),
    )
    manifest = _manifest_for_sources(source_float64, source_float32)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="keypoint dtype mismatch"):
        _discover_merge_sources(
            manifest,
            expected_input_format="gray",
            row_gate_policy="raw_success",
            keypoint_dtype_policy="strict",
        )

    out_zarr = tmp_path / "checked_float32.zarr"
    result = _export_merged(
        manifest_payload=manifest,
        manifest_path=manifest_path,
        out_zarr=out_zarr,
        merged_dataset_id="checked_float32",
        overwrite=False,
        train_ratio=0.5,
        val_ratio=0.5,
        test_ratio=0.0,
        seed=42,
        copy_batch_size=2,
        row_gate_policy="raw_success",
        invocation={},
        keypoint_dtype_policy="float32_checked",
    )

    root = zarr.open_group(str(out_zarr), mode="r", use_consolidated=False)
    keypoints = root[f"keypoints_runs/{result.run_name}/keypoints_roi"]
    dtype_receipt = dict(root.attrs["training_export"])["keypoint_dtype"]
    by_id = {entry["dataset_id"]: entry for entry in dtype_receipt["per_source"]}

    assert np.dtype(keypoints.dtype) == np.dtype(np.float32)
    assert result.keypoint_dtype_policy == "float32_checked"
    assert by_id["dataset_a"]["transform"] == "float64_to_float32_checked"
    assert by_id["dataset_b"]["transform"] == "identity_float32"
    assert 0.0 < by_id["dataset_a"]["max_abs_round_trip_error"] < 1.0e-5
    assert by_id["dataset_b"]["max_abs_round_trip_error"] == 0.0


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


def test_main_auto_aggregates_keypoint_data_card_by_default(
    tmp_path: Path, monkeypatch
) -> None:
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
        source_type_counts={"refined": 1},
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
        lambda **_kwargs: {
            "set_id": "pose_set_v001",
            "datasets": [],
            "merged_export": {"source_datasets": []},
        },
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


def test_main_no_aggregate_training_data_card_disables_aggregation(
    tmp_path: Path, monkeypatch
) -> None:
    manifest_path = tmp_path / "pose_set_v001.manifest.json"
    _write_min_manifest(manifest_path)

    out_dir = tmp_path / "out"
    out_zarr = out_dir / "zarr" / "pose_set_v001_merged.zarr"

    merge_result = SimpleNamespace(
        input_format="gray",
        total_samples=2,
        source_specs=[],
        source_type="refined",
        source_type_counts={"refined": 1},
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
        lambda **_kwargs: {
            "set_id": "pose_set_v001",
            "datasets": [],
            "merged_export": {"source_datasets": []},
        },
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
