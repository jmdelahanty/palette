"""Exact row-level failure semantics for terminal pose inference.

These codes belong to immutable producer evidence.  They are intentionally
kept outside the frozen raw-keypoint-v2 logical schema until the taxonomy has
been exercised by production-shaped canaries and adopted by consumers through
an explicit schema revision.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Mapping

import numpy as np


POSE_INFERENCE_FAILURE_SCHEMA_ID = "palette.pose_inference_failure_codes"
POSE_INFERENCE_FAILURE_SCHEMA_VERSION = 1
POSE_INFERENCE_FAILURE_ARRAY_PATH = "pose_failure_codes"


class PoseInferenceFailureCode(IntEnum):
    """Terminal outcome for one requested pose row."""

    NONE = 0
    NO_POSE_DETECTION_ABOVE_THRESHOLD = 1
    KEYPOINT_PAYLOAD_MISSING = 2
    KEYPOINT_PAYLOAD_EMPTY = 3
    INSUFFICIENT_KEYPOINT_COUNT = 4


POSE_INFERENCE_FAILURE_CODE_MAP: Mapping[int, str] = {
    PoseInferenceFailureCode.NONE: "none",
    PoseInferenceFailureCode.NO_POSE_DETECTION_ABOVE_THRESHOLD: (
        "no_pose_detection_above_threshold"
    ),
    PoseInferenceFailureCode.KEYPOINT_PAYLOAD_MISSING: "keypoint_payload_missing",
    PoseInferenceFailureCode.KEYPOINT_PAYLOAD_EMPTY: "keypoint_payload_empty",
    PoseInferenceFailureCode.INSUFFICIENT_KEYPOINT_COUNT: (
        "insufficient_keypoint_count"
    ),
}


def pose_inference_failure_code_map_json() -> dict[str, str]:
    """Return the exact JSON-safe persisted code registry."""

    return {
        str(int(code)): label
        for code, label in POSE_INFERENCE_FAILURE_CODE_MAP.items()
    }


def validate_pose_inference_failure_codes(
    values: np.ndarray,
    *,
    pose_success: np.ndarray,
) -> None:
    """Require exact dtype, registry membership, and success alignment."""

    codes = np.asarray(values)
    success = np.asarray(pose_success)
    if codes.dtype != np.dtype(np.uint8) or codes.ndim != 1:
        raise ValueError("pose_failure_codes must have exact uint8 shape [N].")
    if success.dtype != np.dtype(bool) or success.shape != codes.shape:
        raise ValueError(
            "pose_success must have exact bool shape matching pose_failure_codes."
        )
    allowed = np.asarray(
        sorted(int(code) for code in POSE_INFERENCE_FAILURE_CODE_MAP),
        dtype=np.uint8,
    )
    if np.any(~np.isin(codes, allowed)):
        raise ValueError("pose_failure_codes contains an undeclared code.")
    no_failure = codes == np.uint8(PoseInferenceFailureCode.NONE)
    if not np.array_equal(no_failure, success):
        raise ValueError(
            "pose_failure_codes must be zero exactly for successful pose rows."
        )


def pose_inference_failure_histogram(values: np.ndarray) -> dict[str, int]:
    """Return an exact complete label-keyed count table."""

    codes = np.asarray(values)
    if codes.dtype != np.dtype(np.uint8) or codes.ndim != 1:
        raise ValueError("pose_failure_codes must have exact uint8 shape [N].")
    allowed = {int(code) for code in POSE_INFERENCE_FAILURE_CODE_MAP}
    observed = {int(code) for code in np.unique(codes)}
    if not observed.issubset(allowed):
        raise ValueError("pose_failure_codes contains an undeclared code.")
    return {
        label: int(np.count_nonzero(codes == np.uint8(code)))
        for code, label in POSE_INFERENCE_FAILURE_CODE_MAP.items()
    }


__all__ = [
    "POSE_INFERENCE_FAILURE_ARRAY_PATH",
    "POSE_INFERENCE_FAILURE_CODE_MAP",
    "POSE_INFERENCE_FAILURE_SCHEMA_ID",
    "POSE_INFERENCE_FAILURE_SCHEMA_VERSION",
    "PoseInferenceFailureCode",
    "pose_inference_failure_code_map_json",
    "pose_inference_failure_histogram",
    "validate_pose_inference_failure_codes",
]
