from __future__ import annotations

import numpy as np


def build_frame_keypoint_counts(
    frame_indices: np.ndarray,
    detection_success: np.ndarray,
    *,
    frame_axis_len: int,
    keypoint_count: int,
) -> np.ndarray:
    """Build the per-frame keypoint-count summary on the run's frame axis."""

    frame_axis_len = int(frame_axis_len)
    keypoint_count = int(keypoint_count)
    if frame_axis_len < 0:
        raise ValueError(f"frame_axis_len must be non-negative, got {frame_axis_len}")
    if keypoint_count < 0:
        raise ValueError(f"keypoint_count must be non-negative, got {keypoint_count}")

    frames = np.asarray(frame_indices, dtype=np.int64)
    success = np.asarray(detection_success, dtype=bool)
    if frames.shape[0] != success.shape[0]:
        raise ValueError(
            "frame_indices and detection_success must have matching lengths "
            f"({frames.shape[0]} != {success.shape[0]})"
        )
    if frames.size:
        invalid = np.flatnonzero((frames < 0) | (frames >= frame_axis_len))
        if invalid.size:
            first = int(invalid[0])
            raise ValueError(
                "frame_indices contains out-of-bounds frame "
                f"{int(frames[first])} at row {first} for frame axis length {frame_axis_len}"
            )

    counts = np.zeros(frame_axis_len, dtype=np.int32)
    if frames.size and np.any(success):
        counts[frames[success]] = keypoint_count
    return counts
