from __future__ import annotations

import numpy as np

from fisheye.refinement.subject_eye_assignment import (
    _split_union_by_keypoints,
    _split_union_by_keypoints_batch_into,
    assign_eyes_union_to_lr,
)


def test_vectorized_keypoint_split_matches_row_reference() -> None:
    union = np.zeros((4, 18, 20), dtype=np.uint8)
    union[0, 4:9, 3:8] = 1
    union[0, 4:9, 12:17] = 1
    union[1, 2:16, 8:12] = 1
    union[2, 8, :] = 1
    union[3, :, 10] = 1

    eye_left = np.asarray(
        [
            [5.0, 6.0],
            [8.0, 8.0],
            [4.0, 8.0],
            [10.0, 4.0],
        ],
        dtype=np.float32,
    )
    eye_right = np.asarray(
        [
            [14.0, 6.0],
            [12.0, 8.0],
            [15.0, 8.0],
            [10.0, 14.0],
        ],
        dtype=np.float32,
    )

    left_out = np.zeros_like(union, dtype=np.uint8)
    right_out = np.zeros_like(union, dtype=np.uint8)

    _split_union_by_keypoints_batch_into(
        union,
        eye_left,
        eye_right,
        row_indices=np.arange(int(union.shape[0])),
        left_out=left_out,
        right_out=right_out,
        batch_size=2,
    )

    for row_idx in range(int(union.shape[0])):
        expected_left, expected_right = _split_union_by_keypoints(
            union[row_idx],
            eye_left[row_idx],
            eye_right[row_idx],
        )
        np.testing.assert_array_equal(left_out[row_idx].astype(bool), expected_left)
        np.testing.assert_array_equal(right_out[row_idx].astype(bool), expected_right)


def test_assign_eyes_union_records_subphase_timings() -> None:
    union = np.zeros((4, 32, 32), dtype=np.uint8)
    union[0, 8:16, 7:15] = 1
    union[0, 8:16, 18:26] = 1
    union[2, 8:16, 7:15] = 1
    union[2, 8:16, 18:26] = 1
    union[3, 8:16, 7:15] = 1
    union[3, 8:16, 18:26] = 1

    keypoints = np.zeros((4, 5, 2), dtype=np.float32)
    keypoints[:, 0, :] = np.asarray([11.0, 12.0], dtype=np.float32)
    keypoints[:, 1, :] = np.asarray([22.0, 12.0], dtype=np.float32)
    keypoints[3, 1, :] = keypoints[3, 0, :]
    success = np.asarray([True, True, False, True], dtype=bool)

    result = assign_eyes_union_to_lr(
        union,
        keypoints_roi=keypoints,
        keypoint_success=success,
        eye_keypoint_indices=(0, 1),
        split_batch_size=2,
    )

    assert set(result.phase_seconds) >= {
        "split_by_keypoint",
        "select_components",
        "measure_ellipse",
        "reason_labels",
    }
    assert result.summary["status_counts"]["assigned"] == 1
    assert result.summary["status_counts"]["failed_empty_union"] == 1
    assert result.summary["status_counts"]["failed_keypoint_status"] == 1
    assert result.summary["status_counts"]["failed_coincident_eye_keypoints"] == 1
    assert np.count_nonzero(result.masks["eye_left"][0]) > 0
    assert np.count_nonzero(result.masks["eye_right"][0]) > 0
