from __future__ import annotations

from io import StringIO

import numpy as np
import pytest
import zarr
from rich.console import Console

from fisheye.segmentation.eye_segmentation import (
    _copy_lineage_arrays_from_crop_with_keypoint_fallback,
)


def _console_capture() -> tuple[Console, StringIO]:
    buffer = StringIO()
    console = Console(file=buffer, force_terminal=False, color_system=None)
    return console, buffer


def test_copy_lineage_prefers_crop_when_crop_and_keypoint_match(tmp_path) -> None:
    root = zarr.open_group(str(tmp_path / "lineage_match.zarr"), mode="w")
    crop_group = root.require_group("crop_runs").create_group("crop_001")
    kp_group = root.require_group("refined_keypoints_runs").create_group("kp_001")
    run_group = root.require_group("eye_masks_runs").create_group("eye_masks_001")

    frame_indices = np.array([0, 1, 2], dtype=np.int32)
    detection_indices = np.array([10, 11, 12], dtype=np.int32)
    frame_counts = np.array([1, 1, 1], dtype=np.int32)

    crop_group.create_array("frame_indices", data=frame_indices, chunks=(2,), overwrite=True)
    crop_group.create_array("detection_indices", data=detection_indices, chunks=(2,), overwrite=True)
    crop_group.create_array("frame_counts", data=frame_counts, chunks=(2,), overwrite=True)

    kp_group.create_array("frame_indices", data=frame_indices, chunks=(1,), overwrite=True)
    kp_group.create_array("detection_indices", data=detection_indices, chunks=(1,), overwrite=True)
    kp_group.create_array("frame_counts", data=frame_counts, chunks=(1,), overwrite=True)

    console, _ = _console_capture()
    _copy_lineage_arrays_from_crop_with_keypoint_fallback(
        run_group=run_group,
        crop_group=crop_group,
        crop_run="crop_001",
        kp_group=kp_group,
        keypoint_group_name="refined_keypoints_runs",
        keypoint_run="kp_001",
        total_rois=3,
        console=console,
    )

    np.testing.assert_array_equal(run_group["frame_indices"][:], frame_indices)
    np.testing.assert_array_equal(run_group["detection_indices"][:], detection_indices)
    np.testing.assert_array_equal(run_group["frame_counts"][:], frame_counts)


def test_copy_lineage_falls_back_to_keypoints_for_missing_crop_array(tmp_path) -> None:
    root = zarr.open_group(str(tmp_path / "lineage_fallback.zarr"), mode="w")
    crop_group = root.require_group("crop_runs").create_group("crop_001")
    kp_group = root.require_group("refined_keypoints_runs").create_group("kp_001")
    run_group = root.require_group("eye_masks_runs").create_group("eye_masks_001")

    frame_indices = np.array([0, 1, 2], dtype=np.int32)
    detection_indices = np.array([20, 21, 22], dtype=np.int32)
    frame_counts = np.array([1, 1, 1], dtype=np.int32)

    crop_group.create_array("frame_indices", data=frame_indices, overwrite=True)
    crop_group.create_array("frame_counts", data=frame_counts, overwrite=True)

    kp_group.create_array("frame_indices", data=frame_indices, overwrite=True)
    kp_group.create_array("detection_indices", data=detection_indices, overwrite=True)
    kp_group.create_array("frame_counts", data=frame_counts, overwrite=True)

    console, buffer = _console_capture()
    _copy_lineage_arrays_from_crop_with_keypoint_fallback(
        run_group=run_group,
        crop_group=crop_group,
        crop_run="crop_001",
        kp_group=kp_group,
        keypoint_group_name="refined_keypoints_runs",
        keypoint_run="kp_001",
        total_rois=3,
        console=console,
    )

    np.testing.assert_array_equal(run_group["detection_indices"][:], detection_indices)
    output = buffer.getvalue()
    assert "falling back to" in output
    assert "refined_keypoints_runs/kp_001/detection_indices" in output


def test_copy_lineage_raises_on_crop_keypoint_mismatch(tmp_path) -> None:
    root = zarr.open_group(str(tmp_path / "lineage_mismatch.zarr"), mode="w")
    crop_group = root.require_group("crop_runs").create_group("crop_001")
    kp_group = root.require_group("refined_keypoints_runs").create_group("kp_001")
    run_group = root.require_group("eye_masks_runs").create_group("eye_masks_001")

    crop_group.create_array("frame_indices", data=np.array([0, 1, 2], dtype=np.int32), overwrite=True)
    crop_group.create_array("detection_indices", data=np.array([3, 4, 5], dtype=np.int32), overwrite=True)
    crop_group.create_array("frame_counts", data=np.array([1, 1, 1], dtype=np.int32), overwrite=True)

    kp_group.create_array("frame_indices", data=np.array([0, 2, 1], dtype=np.int32), overwrite=True)
    kp_group.create_array("detection_indices", data=np.array([3, 4, 5], dtype=np.int32), overwrite=True)
    kp_group.create_array("frame_counts", data=np.array([1, 1, 1], dtype=np.int32), overwrite=True)

    console, _ = _console_capture()
    with pytest.raises(ValueError, match="lineage mismatch.*frame_indices"):
        _copy_lineage_arrays_from_crop_with_keypoint_fallback(
            run_group=run_group,
            crop_group=crop_group,
            crop_run="crop_001",
            kp_group=kp_group,
            keypoint_group_name="refined_keypoints_runs",
            keypoint_run="kp_001",
            total_rois=3,
            console=console,
        )


def test_copy_lineage_raises_when_frame_counts_sum_does_not_match_rois(tmp_path) -> None:
    root = zarr.open_group(str(tmp_path / "lineage_bad_counts.zarr"), mode="w")
    crop_group = root.require_group("crop_runs").create_group("crop_001")
    kp_group = root.require_group("refined_keypoints_runs").create_group("kp_001")
    run_group = root.require_group("eye_masks_runs").create_group("eye_masks_001")

    crop_group.create_array("frame_indices", data=np.array([0, 1, 2], dtype=np.int32), overwrite=True)
    crop_group.create_array("detection_indices", data=np.array([3, 4, 5], dtype=np.int32), overwrite=True)
    crop_group.create_array("frame_counts", data=np.array([1, 1, 0], dtype=np.int32), overwrite=True)

    console, _ = _console_capture()
    with pytest.raises(ValueError, match="frame_counts' sums to 2, expected 3"):
        _copy_lineage_arrays_from_crop_with_keypoint_fallback(
            run_group=run_group,
            crop_group=crop_group,
            crop_run="crop_001",
            kp_group=kp_group,
            keypoint_group_name="refined_keypoints_runs",
            keypoint_run="kp_001",
            total_rois=3,
            console=console,
        )
