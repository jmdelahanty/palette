"""Shared chunk-layout policy for subject-mask stage arrays.

These arrays are usually accessed and edited one ROI at a time, not as tiled
sub-regions of the ROI. Favor full-spatial ROI chunks with modest row depth so
incremental writes remain practical while file counts stay bounded.
"""

from __future__ import annotations

SUBJECT_MASK_STORAGE_ROW_CHUNK = 16
SUBJECT_MASK_METRIC_ROW_CHUNK = 256
REFINED_SUBJECT_MASK_DASK_CHUNK_ALIGNMENT = "refined_subject_mask_metric_row_chunk"


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


def refined_subject_mask_storage_row_chunk(total_rows: int, row_chunk: int | None = None) -> int:
    """Return the row chunk depth for dense refined-subject-mask ROI arrays."""
    preferred = SUBJECT_MASK_STORAGE_ROW_CHUNK if row_chunk is None else int(row_chunk)
    return _clamp_positive_chunk(preferred, total_rows)


def subject_mask_metric_row_chunk(total_rows: int) -> int:
    """Return canonical row chunk depth for subject-mask metric arrays."""
    return _clamp_positive_chunk(SUBJECT_MASK_METRIC_ROW_CHUNK, total_rows)


def refined_subject_mask_storage_chunks(
    total_rows: int,
    height: int,
    width: int,
    row_chunk: int | None = None,
) -> tuple[int, int, int, int]:
    """Return canonical chunks for dense refined-subject-mask ROI arrays.

    Refined runs currently share the same full-ROI storage policy as raw
    subject-mask runs, but keep a distinct helper so the refined contract can
    evolve independently if review/edit access patterns diverge later.
    """
    return (
        refined_subject_mask_storage_row_chunk(total_rows, row_chunk),
        1,
        max(1, int(height)),
        max(1, int(width)),
    )


def refined_subject_mask_metric_row_chunk(total_rows: int) -> int:
    """Return canonical row chunk depth for refined-subject-mask metric arrays."""
    return subject_mask_metric_row_chunk(total_rows)


def refined_subject_mask_dask_worker_row_chunk(total_rows: int, requested_chunk_size: int) -> int:
    """Return a worker chunk size that cannot split refined-subject metric chunks.

    Refined-subject analysis workers commonly update multiple row-aligned metric
    arrays. Those arrays use ``refined_subject_mask_metric_row_chunk`` on the
    first axis, so Dask worker ranges must be at least one metric chunk and, for
    larger requests, an integer multiple of that physical chunk size.
    """

    requested = max(1, int(requested_chunk_size))
    metric_chunk = refined_subject_mask_metric_row_chunk(total_rows)
    if requested <= metric_chunk:
        return int(metric_chunk)
    return int(((requested + metric_chunk - 1) // metric_chunk) * metric_chunk)
