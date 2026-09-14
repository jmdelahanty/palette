"""Preservation checks for recovery from the surviving merged training sets."""

import numpy as np
import pytest

from fisheye.training.recover_merged_training_recording import (
    build_recording_row_join,
    select_centered_head_crops,
)


def test_join_preserves_detect_only_rows_and_exact_boxes() -> None:
    join = build_recording_row_join(
        recording_id="rec_a",
        pose_source_ids=("rec_a:zpose", "rec_b:zpose"),
        pose_dataset_idx=np.array([0, 0, 1], dtype=np.int32),
        pose_frame_idx=np.array([2, 4, 2], dtype=np.int64),
        pose_boxes=np.array(
            [[0.3, 0.4, 0.1, 0.1], [0.5, 0.4, 0.1, 0.1], [0, 0, 1, 1]], dtype=np.float32
        ),
        detect_source_ids=("rec_a", "rec_b"),
        detect_dataset_idx=np.array([0, 0, 0, 1], dtype=np.int32),
        detect_frame_idx=np.array([1, 2, 4, 2], dtype=np.int64),
        detect_boxes=np.array(
            [
                [0.1, 0.4, 0.1, 0.1],
                [0.3, 0.4, 0.1, 0.1],
                [0.5, 0.4, 0.1, 0.1],
                [0, 0, 1, 1],
            ],
            dtype=np.float32,
        ),
    )
    assert join.pose_merged_rows.tolist() == [0, 1]
    assert join.detect_merged_rows.tolist() == [0, 1, 2]
    assert join.pose_to_detect_local.tolist() == [1, 2]
    assert join.detect_only_local.tolist() == [0]


def test_join_rejects_duplicate_or_conflicting_source_identity() -> None:
    args = dict(
        recording_id="rec_a",
        pose_source_ids=("rec_a:zpose",),
        pose_dataset_idx=np.array([0], dtype=np.int32),
        pose_frame_idx=np.array([2], dtype=np.int64),
        pose_boxes=np.array([[0.3, 0.4, 0.1, 0.1]], dtype=np.float32),
        detect_source_ids=("rec_a",),
        detect_dataset_idx=np.array([0], dtype=np.int32),
        detect_frame_idx=np.array([2], dtype=np.int64),
        detect_boxes=np.array([[0.3, 0.4, 0.1, 0.1]], dtype=np.float32),
    )
    with pytest.raises(ValueError, match="duplicate detection frame"):
        build_recording_row_join(
            **{
                **args,
                "detect_dataset_idx": np.array([0, 0]),
                "detect_frame_idx": np.array([2, 2]),
                "detect_boxes": np.repeat(args["detect_boxes"], 2, axis=0),
            }
        )
    with pytest.raises(ValueError, match="box mismatch"):
        build_recording_row_join(
            **{
                **args,
                "detect_boxes": np.array([[0.2, 0.4, 0.1, 0.1]], dtype=np.float32),
            }
        )


def test_detect_only_recording_keeps_all_detector_rows_without_invented_pose() -> None:
    join = build_recording_row_join(
        recording_id="rec_a",
        pose_source_ids=("rec_b:zpose",),
        pose_dataset_idx=np.array([0], dtype=np.int32),
        pose_frame_idx=np.array([2], dtype=np.int64),
        pose_boxes=np.array([[0.3, 0.4, 0.1, 0.1]], dtype=np.float32),
        detect_source_ids=("rec_a",),
        detect_dataset_idx=np.array([0, 0], dtype=np.int32),
        detect_frame_idx=np.array([1, 2], dtype=np.int64),
        detect_boxes=np.array(
            [[0.1, 0.4, 0.1, 0.1], [0.3, 0.4, 0.1, 0.1]], dtype=np.float32
        ),
        allow_no_pose=True,
    )
    assert join.pose_source_dataset_id is None
    assert join.pose_merged_rows.size == 0
    assert join.detect_merged_rows.tolist() == [0, 1]
    assert join.detect_only_local.tolist() == [0, 1]


def test_centered_crop_keeps_only_fully_visible_head_rows() -> None:
    points = np.array(
        [
            [[256, 256], [260, 250], [250, 250]],
            [[256, 330], [260, 370], [250, 360]],
        ],
        dtype=np.float64,
    )
    kept, translated, excluded = select_centered_head_crops(
        points, image_shape=(512, 512)
    )
    assert kept.tolist() == [0]
    assert excluded.tolist() == [1]
    np.testing.assert_array_equal(translated[0], points[0] - 160)
