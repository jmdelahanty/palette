from __future__ import annotations

import numpy as np
import pytest

from fisheye.shared.pose_inference_failure import (
    PoseInferenceFailureCode,
    pose_inference_failure_code_map_json,
    pose_inference_failure_histogram,
    validate_pose_inference_failure_codes,
)


def test_pose_inference_failure_codes_are_exact_and_aligned() -> None:
    codes = np.asarray(
        [
            PoseInferenceFailureCode.NONE,
            PoseInferenceFailureCode.NO_POSE_DETECTION_ABOVE_THRESHOLD,
            PoseInferenceFailureCode.KEYPOINT_PAYLOAD_MISSING,
            PoseInferenceFailureCode.KEYPOINT_PAYLOAD_EMPTY,
            PoseInferenceFailureCode.INSUFFICIENT_KEYPOINT_COUNT,
        ],
        dtype=np.uint8,
    )
    success = np.asarray([True, False, False, False, False], dtype=bool)

    validate_pose_inference_failure_codes(codes, pose_success=success)

    assert pose_inference_failure_code_map_json() == {
        "0": "none",
        "1": "no_pose_detection_above_threshold",
        "2": "keypoint_payload_missing",
        "3": "keypoint_payload_empty",
        "4": "insufficient_keypoint_count",
    }
    assert pose_inference_failure_histogram(codes) == {
        "none": 1,
        "no_pose_detection_above_threshold": 1,
        "keypoint_payload_missing": 1,
        "keypoint_payload_empty": 1,
        "insufficient_keypoint_count": 1,
    }


def test_pose_inference_failure_codes_reject_unknown_or_misaligned_values() -> None:
    with pytest.raises(ValueError, match="undeclared"):
        validate_pose_inference_failure_codes(
            np.asarray([255], dtype=np.uint8),
            pose_success=np.asarray([False], dtype=bool),
        )

    with pytest.raises(ValueError, match="zero exactly"):
        validate_pose_inference_failure_codes(
            np.asarray([PoseInferenceFailureCode.NONE], dtype=np.uint8),
            pose_success=np.asarray([False], dtype=bool),
        )

    with pytest.raises(ValueError, match="uint8"):
        validate_pose_inference_failure_codes(
            np.asarray([PoseInferenceFailureCode.NONE], dtype=np.int64),
            pose_success=np.asarray([True], dtype=bool),
        )
