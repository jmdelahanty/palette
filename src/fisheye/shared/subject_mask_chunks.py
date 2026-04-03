"""Shared chunk-layout policy for subject-mask stage arrays.

These arrays are usually accessed and edited one ROI at a time, not as tiled
sub-regions of the ROI. Favor full-spatial ROI chunks with modest row depth so
incremental writes remain practical while file counts stay bounded.
"""

from __future__ import annotations

SUBJECT_MASK_STORAGE_ROW_CHUNK = 16
SUBJECT_MASK_METRIC_ROW_CHUNK = 256


def _clamp_positive_chunk(preferred: int, total: int) -> int:
    if preferred <= 0:
        raise ValueError("preferred chunk size must be positive.")
    if total <= 0:
        return 1
    return max(1, min(int(preferred), int(total)))


def subject_mask_storage_chunks(total_rows: int, height: int, width: int) -> tuple[int, int, int, int]:
    """Return canonical chunks for dense subject-mask ROI arrays."""
    return (
        _clamp_positive_chunk(SUBJECT_MASK_STORAGE_ROW_CHUNK, total_rows),
        1,
        max(1, int(height)),
        max(1, int(width)),
    )


def subject_mask_metric_row_chunk(total_rows: int) -> int:
    """Return canonical row chunk depth for subject-mask metric arrays."""
    return _clamp_positive_chunk(SUBJECT_MASK_METRIC_ROW_CHUNK, total_rows)
