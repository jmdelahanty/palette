"""Small real-Zarr contract check; run outside the Codex sandbox."""

from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.shared.run_provenance import build_writer_run_provenance
from fisheye.shared.zarr_run_completion import (
    mark_run_complete,
    mark_run_started,
    require_runs_parent,
)
from fisheye.training.materialize_pose_head_crops import (
    build_pose_head_crop_plan,
    materialize_pose_head_crops,
)


KEYPOINT_ID = "reviewed_v2"
CROP_ID = "source_crop"
TARGET_ID = "pose_head_192_traditional_v1_v001"


def _complete(run: zarr.Group, parent: zarr.Group, name: str) -> None:
    mark_run_started(run, run_name=name, stage="fixture")
    mark_run_complete(
        run,
        parent_group=parent,
        run_name=name,
        run_provenance=build_writer_run_provenance(
            command="test_materialize_pose_head_crops",
            input_run_ids={},
        ),
    )


def _fixture(
    path: Path, *, selected_keypoints: bool = False, legacy_refined_boxes: bool = False
) -> zarr.Group:
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    root.attrs["zarr_purpose"] = "training"
    root.attrs["source_h5_fingerprint"] = "fixture-source"
    raw = root.create_group("raw_video")
    pixels = (
        np.arange(3 * 256 * 256, dtype=np.int64).reshape(3, 256, 256).astype(np.uint8)
    )
    raw.create_array("images_full", data=pixels, chunks=(2, 256, 256))
    raw.create_array("original_frame_indices", data=np.array([10, 20, 30]))

    crop_parent = require_runs_parent(root, "crop_runs")
    crop = crop_parent.create_group(CROP_ID)
    if legacy_refined_boxes:
        crop.attrs["source_refined_run"] = "reviewed_detection"
        crop.create_array("frame_indices", data=np.array([0, 1, 2]))
        crop.create_array("detection_indices", data=np.array([0, 1, 2]))
        crop.create_array("source_refined_row_ids", data=np.array([0, 1, 2]))
    else:
        crop.create_array(
            "bbox_img_xyxy",
            data=np.array(
                [[90, 90, 110, 110], [0, 0, 20, 20], [160, 160, 180, 180]],
                dtype=np.float32,
            ),
        )
        crop.create_array("source_training_row_indices", data=np.array([0, 1, 2]))
    crop.create_array("source_frame_indices", data=np.array([10, 20, 30]))
    crop.create_array("roi_coordinates_full", data=np.zeros((3, 2), dtype=np.int32))
    _complete(crop, crop_parent, CROP_ID)

    kp_parent = require_runs_parent(
        root, "keypoints_runs" if selected_keypoints else "refined_keypoints_runs"
    )
    keypoints = kp_parent.create_group(KEYPOINT_ID)
    keypoints.attrs["source_crop_run"] = CROP_ID
    if legacy_refined_boxes:
        keypoints.attrs["source_refined_run"] = "reviewed_detection"
    keypoints.attrs["skeleton_id"] = "pose_skel_traditional_v2"
    keypoints.attrs["keypoint_labels"] = [
        "swim_bladder",
        "eye_left",
        "eye_right",
        "snout_tip",
        "tail_tip",
    ]
    points = np.array(
        [
            [[100, 100], [110, 110], [90, 110], [120, 120], [150, 150]],
            [[10, 10], [20, 20], [30, 20], [40, 40], [50, 50]],
            [[170, 170], [180, 180], [160, 180], [190, 190], [200, 200]],
        ],
        dtype=np.float64,
    )
    keypoints.create_array("keypoints_img", data=points)
    keypoints.create_array("keypoints_roi", data=points.copy())
    keypoints.create_array(
        "detection_success" if selected_keypoints else "usable_keypoints",
        data=np.array([True, False, True]),
    )
    if legacy_refined_boxes:
        keypoints.create_array("frame_indices", data=np.array([0, 1, 2]))
        keypoints.create_array("detection_indices", data=np.array([0, 1, 2]))
    else:
        keypoints.create_array("source_crop_row_ids", data=np.array([0, 1, 2]))
    keypoints.create_array("source_frame_indices", data=np.array([10, 20, 30]))
    _complete(keypoints, kp_parent, KEYPOINT_ID)
    if legacy_refined_boxes:
        detection_parent = require_runs_parent(root, "refined_detect_runs")
        detection = detection_parent.create_group("reviewed_detection")
        table = detection.create_group("instances")
        table.create_array("refined_row_ids", data=np.array([0, 1, 2]))
        table.create_array("frame_indices", data=np.array([0, 1, 2]))
        table.create_array(
            "bbox_norm_coords",
            data=np.array(
                [
                    [100 / 256, 100 / 256, 20 / 256, 20 / 256],
                    [10 / 256, 10 / 256, 20 / 256, 20 / 256],
                    [170 / 256, 170 / 256, 20 / 256, 20 / 256],
                ],
                dtype=np.float64,
            ),
        )
        _complete(detection, detection_parent, "reviewed_detection")
        detection_parent.attrs["authoritative_run"] = "reviewed_detection"
    return root


def test_materializes_only_usable_rows_without_changing_selectors(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "training.zarr"
    _fixture(archive)
    result = materialize_pose_head_crops(
        zarr_path=archive,
        source_keypoint_run=KEYPOINT_ID,
        source_crop_run=CROP_ID,
        run_id=TARGET_ID,
        scratch_root=tmp_path,
        apply=True,
    )
    assert result["status"] == "materialized"
    root = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    run = root[f"crop_runs/{TARGET_ID}"]
    assert run.attrs["stage_selector_eligible"] is False
    assert root["crop_runs"].attrs["latest_complete"] == CROP_ID
    assert root["refined_keypoints_runs"].attrs["latest_complete"] == KEYPOINT_ID
    np.testing.assert_array_equal(run["source_keypoint_row_ids"][:], [0, 2])
    np.testing.assert_array_equal(run["keypoint_visibility"][:], np.full((2, 3), 2))
    np.testing.assert_array_equal(
        run["roi_images"][0], root["raw_video/images_full"][0, 4:196, 4:196]
    )
    np.testing.assert_array_equal(
        run["keypoints_roi"][0], [[96, 96], [106, 106], [86, 106]]
    )


def test_refuses_wrong_source_frame_and_crop_binding(tmp_path: Path) -> None:
    root = _fixture(tmp_path / "training.zarr")
    root[f"refined_keypoints_runs/{KEYPOINT_ID}"].attrs["source_crop_run"] = (
        "another_crop"
    )
    with pytest.raises(ValueError, match="not bound"):
        build_pose_head_crop_plan(
            root, source_keypoint_run=KEYPOINT_ID, source_crop_run=CROP_ID
        )
    root[f"refined_keypoints_runs/{KEYPOINT_ID}"].attrs["source_crop_run"] = CROP_ID
    root[f"refined_keypoints_runs/{KEYPOINT_ID}/source_frame_indices"][0] = 999
    with pytest.raises(ValueError, match="frame identities disagree"):
        build_pose_head_crop_plan(
            root, source_keypoint_run=KEYPOINT_ID, source_crop_run=CROP_ID
        )


def test_selected_keypoint_family_uses_its_bound_crop(tmp_path: Path) -> None:
    archive = tmp_path / "training.zarr"
    _fixture(archive, selected_keypoints=True)
    result = materialize_pose_head_crops(
        zarr_path=archive,
        source_keypoint_group="keypoints_runs",
        source_keypoint_run=KEYPOINT_ID,
        source_crop_run=CROP_ID,
        run_id=TARGET_ID,
        scratch_root=tmp_path,
        apply=True,
    )
    assert result["row_count"] == 2
    root = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    run = root[f"crop_runs/{TARGET_ID}"]
    assert run.attrs["schema_version"] == 2
    assert run.attrs["source_bindings"]["source_keypoint_group"] == "keypoints_runs"
    assert root["keypoints_runs"].attrs["latest_complete"] == KEYPOINT_ID


def test_legacy_authoritative_norm_boxes_join_by_frame_and_detection(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "training.zarr"
    _fixture(archive, legacy_refined_boxes=True)
    result = materialize_pose_head_crops(
        zarr_path=archive,
        source_box_mode="legacy_authoritative_refined_detection_norm_cxcywh",
        source_keypoint_run=KEYPOINT_ID,
        source_crop_run=CROP_ID,
        run_id=TARGET_ID,
        scratch_root=tmp_path,
        apply=True,
    )
    assert result["row_count"] == 2
    root = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    run = root[f"crop_runs/{TARGET_ID}"]
    assert run.attrs["schema_version"] == 3
    assert run.attrs["source_bindings"]["source_detection_run"] == "reviewed_detection"
    np.testing.assert_array_equal(
        run["bbox_img_xyxy"][:], [[90, 90, 110, 110], [160, 160, 180, 180]]
    )
