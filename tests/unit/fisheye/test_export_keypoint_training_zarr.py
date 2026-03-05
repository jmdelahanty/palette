"""Tests for keypoint merged-export skeleton identity guardrails."""

from pathlib import Path
import sys

import numpy as np
import pytest
import zarr

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.utils.export_keypoint_training_zarr import (
    _discover_merge_sources,
    _format_skeleton_signature,
    _normalize_kpt_shape,
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
    kpt_shape: tuple[int, int] = (3, 3),
    keypoint_count: int = 3,
    refined_reasons: list[str] | None = None,
) -> None:
    root = zarr.open_group(str(path), mode="w")

    crop_parent = root.create_group("crop_runs")
    crop_parent.attrs["latest"] = "crop_pose_001"
    crop = crop_parent.create_group("crop_pose_001")
    crop.attrs["detection_source_type"] = "filtered"
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
    kp.attrs["keypoint_labels"] = [f"kpt_{idx}" for idx in range(int(keypoint_count))]
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
    if refined_reasons is not None:
        refined_parent = root.create_group("refined_keypoints_runs")
        refined_parent.attrs["latest"] = "refined_kp_pose_001"
        refined = refined_parent.create_group("refined_kp_pose_001")
        refined.attrs["source_keypoints_run"] = "kp_pose_001"
        refined.attrs["source_crop_run"] = "crop_pose_001"
        refined.attrs["created_utc"] = "2026-02-27T00:00:00+00:00"
        refined.create_array(
            "keypoints_roi",
            data=np.zeros((4, int(keypoint_count), 2), dtype=np.float32),
            chunks=(4, int(keypoint_count), 2),
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
        "source_type": "filtered",
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
        "source_type": "filtered",
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
