"""Compatibility-neutral access to detection instance tables."""

from __future__ import annotations

from typing import Any

import numpy as np

DETECTION_INSTANCE_REQUIRED_ARRAYS = (
    "frame_indices",
    "bbox_norm_coords",
)


def resolve_detection_instance_table(run: Any) -> Any:
    """Return ``instances`` for strict runs or the legacy run-root table.

    This is a reader boundary only.  It does not create aliases and it never
    changes which run is selected.
    """

    table = run.get("instances")
    if table is not None and all(
        name in table for name in DETECTION_INSTANCE_REQUIRED_ARRAYS
    ):
        return table
    return run


def read_detection_frame_counts(table: Any, *, n_frames: int) -> np.ndarray:
    """Read compatibility counts or derive them from the canonical CSR index."""

    count = int(n_frames)
    if count < 0:
        raise ValueError("n_frames cannot be negative.")
    if "frame_row_offsets" in table:
        offsets = np.asarray(table["frame_row_offsets"][:], dtype=np.int64)
        if offsets.shape != (count + 1,):
            raise ValueError(
                "Detection frame_row_offsets length differs from n_frames + 1."
            )
        if not offsets.size or int(offsets[0]) != 0 or np.any(np.diff(offsets) < 0):
            raise ValueError("Detection frame_row_offsets is malformed.")
        differences = np.diff(offsets)
        if differences.size and int(np.max(differences)) > np.iinfo(np.int32).max:
            raise ValueError("Per-frame detection cardinality exceeds int32.")
        return differences.astype(np.int32, copy=False)
    for name in ("frame_counts", "n_detections"):
        if name in table:
            values = np.asarray(table[name][:], dtype=np.int32)
            if values.shape != (count,):
                raise ValueError(f"Detection {name} length differs from n_frames.")
            return values
    frames = np.asarray(table["frame_indices"][:], dtype=np.int64)
    if frames.size and (np.any(frames < 0) or np.any(frames >= count)):
        raise ValueError("Detection frame_indices are outside n_frames.")
    return np.bincount(frames, minlength=count).astype(np.int32, copy=False)


__all__ = [
    "DETECTION_INSTANCE_REQUIRED_ARRAYS",
    "read_detection_frame_counts",
    "resolve_detection_instance_table",
]
