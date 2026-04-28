from __future__ import annotations

import numpy as np

from fisheye.pose.body_frame import (
    BODY_FRAME_REASON_DEGENERATE_FORWARD_AXIS,
    BODY_FRAME_REASON_LEFT_RIGHT_UNRESOLVED,
    compute_keypoint_body_frame,
)
from fisheye.shared.detect_reason_codec import decode_reason_bytes


def test_keypoint_body_frame_points_forward_and_anatomical_left() -> None:
    keypoints = np.asarray(
        [
            [
                [0.0, 0.0],   # swim_bladder
                [2.0, -1.0],  # eye_left
                [2.0, 1.0],   # eye_right
            ]
        ],
        dtype=np.float32,
    )

    frame = compute_keypoint_body_frame(
        keypoints,
        keypoint_indices={"swim_bladder": 0, "eye_left": 1, "eye_right": 2},
    )

    assert frame.valid.tolist() == [True]
    np.testing.assert_allclose(frame.origin_xy[0], [2.0, 0.0])
    np.testing.assert_allclose(frame.forward_axis_xy[0], [1.0, 0.0])
    np.testing.assert_allclose(frame.left_axis_xy[0], [0.0, -1.0])
    np.testing.assert_allclose(frame.heading_deg[0], 0.0)


def test_keypoint_body_frame_marks_degenerate_forward_axis_invalid() -> None:
    keypoints = np.asarray(
        [
            [
                [1.0, 1.0],
                [0.0, 0.0],
                [2.0, 2.0],
            ]
        ],
        dtype=np.float32,
    )

    frame = compute_keypoint_body_frame(
        keypoints,
        keypoint_indices={"swim_bladder": 0, "eye_left": 1, "eye_right": 2},
    )

    assert frame.valid.tolist() == [False]
    assert decode_reason_bytes(frame.failure_reason_bytes).tolist() == [BODY_FRAME_REASON_DEGENERATE_FORWARD_AXIS]


def test_keypoint_body_frame_marks_unresolved_left_right_invalid() -> None:
    keypoints = np.asarray(
        [
            [
                [0.0, 0.0],
                [2.0, 0.0],
                [2.0, 0.0],
            ]
        ],
        dtype=np.float32,
    )

    frame = compute_keypoint_body_frame(
        keypoints,
        keypoint_indices={"swim_bladder": 0, "eye_left": 1, "eye_right": 2},
    )

    assert frame.valid.tolist() == [False]
    assert decode_reason_bytes(frame.failure_reason_bytes).tolist() == [BODY_FRAME_REASON_LEFT_RIGHT_UNRESOLVED]
