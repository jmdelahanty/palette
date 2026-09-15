"""Preservation checks for the fixed sensor-pixel pose-head crop recipe."""

import numpy as np
import pytest

from fisheye.training.pose_head_crop_geometry import (
    POSE_HEAD_KEYPOINT_LABELS,
    fixed_pose_head_origins,
    project_pose_head_keypoints,
)


def test_fixed_origins_truncate_and_clamp_without_padding() -> None:
    boxes = np.array(
        [
            [0.0, 0.0, 40.0, 40.0],
            [100.2, 200.2, 121.2, 221.2],
            [960.0, 960.0, 999.0, 999.0],
        ],
        dtype=np.float64,
    )
    actual = fixed_pose_head_origins(
        boxes, frame_shape_hw=(1000, 1000), crop_size_px=192
    )
    np.testing.assert_array_equal(actual, [[0, 0], [14, 114], [808, 808]])


def test_project_three_points_from_extended_schema_and_mark_outside() -> None:
    source_labels = ("swim_bladder", "eye_left", "eye_right", "snout_tip", "tail_tip")
    source = np.array(
        [
            [
                [110.0, 110.0],
                [120.0, 130.0],
                [90.0, 130.0],
                [125.0, 120.0],
                [500.0, 500.0],
            ],
            [[20.0, 20.0], [200.0, 20.0], [40.0, 30.0], [25.0, 30.0], [500.0, 500.0]],
        ],
        dtype=np.float64,
    )
    origins = np.array([[16, 16], [0, 0]], dtype=np.int32)
    coordinates, visibility = project_pose_head_keypoints(
        source, source_labels=source_labels, origins_xy=origins, crop_size_px=192
    )
    assert POSE_HEAD_KEYPOINT_LABELS == ("swim_bladder", "eye_left", "eye_right")
    np.testing.assert_array_equal(coordinates[0], [[94, 94], [104, 114], [74, 114]])
    np.testing.assert_array_equal(coordinates[1], [[20, 20], [200, 20], [40, 30]])
    np.testing.assert_array_equal(visibility, [[2, 2, 2], [2, 0, 2]])


def test_refuse_missing_or_duplicate_required_label() -> None:
    points = np.zeros((1, 3, 2), dtype=np.float64)
    origins = np.zeros((1, 2), dtype=np.int32)
    with pytest.raises(ValueError, match="eye_right"):
        project_pose_head_keypoints(
            points,
            source_labels=("swim_bladder", "eye_left", "tail_tip"),
            origins_xy=origins,
        )
    with pytest.raises(ValueError, match="duplicate"):
        project_pose_head_keypoints(
            points,
            source_labels=("swim_bladder", "eye_left", "eye_left"),
            origins_xy=origins,
        )


def test_refuse_nonfinite_or_malformed_box_geometry() -> None:
    with pytest.raises(ValueError, match="finite"):
        fixed_pose_head_origins(
            np.array([[0.0, 0.0, np.nan, 10.0]]),
            frame_shape_hw=(1000, 1000),
        )
    with pytest.raises(ValueError, match="positive"):
        fixed_pose_head_origins(
            np.array([[10.0, 0.0, 0.0, 10.0]]),
            frame_shape_hw=(1000, 1000),
        )
    with pytest.raises(ValueError, match="frame"):
        fixed_pose_head_origins(
            np.array([[0.0, 0.0, 10.0, 10.0]]),
            frame_shape_hw=(128, 128),
        )
