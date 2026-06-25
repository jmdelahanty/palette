"""Shared helpers for propagating ROI row-lineage arrays between stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import zarr

from fisheye.shared.zarr.chunk_profiles import (
    CRIMSON_LINEAGE_PRELOAD_ARRAYS,
    create_geometry_preload_array,
)

ROW_ALIGNMENT_ARRAYS: Tuple[str, ...] = (
    "frame_indices",
    "frame_counts",
    "detection_indices",
)
ROW_IDENTITY_ARRAYS: Tuple[str, ...] = (
    "source_frame_indices",
    "source_clip_indices",
    "source_clip_local_frame_indices",
    "source_crop_row_ids",
    "source_refined_row_ids",
    "source_detect_row_index",
)
ROW_LINEAGE_ARRAYS: Tuple[str, ...] = ROW_ALIGNMENT_ARRAYS + ROW_IDENTITY_ARRAYS
PER_ROI_ROW_LINEAGE_ARRAYS = {
    "frame_indices",
    "source_frame_indices",
    "source_clip_indices",
    "source_clip_local_frame_indices",
    "source_crop_row_ids",
    "detection_indices",
    "source_refined_row_ids",
    "source_detect_row_index",
}
COUNT_ROW_LINEAGE_ARRAYS = {"frame_counts"}
SOURCE_CROP_ROW_IDS_ARRAY = "source_crop_row_ids"


@dataclass(frozen=True)
class RowLineageCopyResult:
    copied: Tuple[str, ...]
    missing: Tuple[str, ...]
    fallback_copied: Tuple[str, ...] = ()


def normalize_chunks_for_data(
    chunks: object,
    data_shape: Sequence[int],
    *,
    default_chunk_len: int = 1024,
) -> Optional[Tuple[int, ...]]:
    if not data_shape:
        return None
    if not chunks:
        return tuple(max(1, min(int(dim), int(default_chunk_len))) for dim in data_shape)
    if isinstance(chunks, int):
        chunks_tuple = (int(chunks),)
    else:
        chunks_tuple = tuple(int(item) for item in chunks)  # type: ignore[arg-type]
    normalized = []
    for axis, dim in enumerate(data_shape):
        chunk_value = chunks_tuple[axis] if axis < len(chunks_tuple) else chunks_tuple[-1]
        normalized.append(max(1, min(int(dim), int(chunk_value))))
    return tuple(normalized)


def validate_row_lineage_array(
    name: str,
    data: np.ndarray,
    *,
    total_rois: Optional[int],
) -> None:
    if data.ndim == 0:
        raise ValueError(f"row-lineage array '{name}' must be 1D, got scalar value")
    if total_rois is None:
        return
    if name in PER_ROI_ROW_LINEAGE_ARRAYS and int(data.shape[0]) != int(total_rois):
        raise ValueError(
            f"row-lineage array '{name}' has {data.shape[0]} rows, expected {total_rois}"
        )
    if name in COUNT_ROW_LINEAGE_ARRAYS:
        count_sum = int(np.sum(data, dtype=np.int64))
        if count_sum != int(total_rois):
            raise ValueError(
                f"row-lineage array '{name}' sums to {count_sum}, expected {total_rois}"
            )


def _read_array(source_group: zarr.Group, name: str) -> tuple[object, np.ndarray] | None:
    source_arr = source_group.get(name)
    if source_arr is None:
        return None
    return source_arr, np.asarray(source_arr[:])


def _as_array(value: object | None) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return np.asarray(value)
    try:
        return np.asarray(value[:])  # type: ignore[index]
    except Exception:
        return np.asarray(value)


def direct_source_crop_row_ids(total_rois: int) -> np.ndarray:
    """Return explicit crop-row IDs for a run that is row-aligned to its crop run."""

    return np.arange(int(total_rois), dtype=np.int64)


def _validate_source_crop_row_frame_alignment(
    source_crop_row_ids: np.ndarray,
    *,
    crop_group: zarr.Group | None,
    frame_indices: object | None,
    total_rois: int,
) -> None:
    """Ensure source crop rows point at the same frame index as the target rows."""

    validate_row_lineage_array(
        SOURCE_CROP_ROW_IDS_ARRAY,
        np.asarray(source_crop_row_ids),
        total_rois=total_rois,
    )
    if crop_group is None or "frame_indices" not in crop_group:
        return
    crop_frames = np.asarray(crop_group["frame_indices"][:], dtype=np.int64).reshape(-1)
    crop_row_ids = np.asarray(source_crop_row_ids, dtype=np.int64).reshape(-1)
    if crop_row_ids.size and (int(crop_row_ids.min()) < 0 or int(crop_row_ids.max()) >= int(crop_frames.shape[0])):
        raise ValueError("source_crop_row_ids contains rows outside the referenced source_crop_run.")
    source_frames = _as_array(frame_indices)
    if source_frames is None:
        return
    source_frames = np.asarray(source_frames, dtype=np.int64).reshape(-1)
    if int(source_frames.shape[0]) != int(crop_row_ids.shape[0]):
        raise ValueError(
            "frame_indices length does not match source_crop_row_ids length "
            f"({source_frames.shape[0]} != {crop_row_ids.shape[0]})."
        )
    crop_frames_for_rows = crop_frames[crop_row_ids]
    if not np.array_equal(source_frames, crop_frames_for_rows):
        raise ValueError("source_crop_row_ids do not map to matching crop frame_indices.")


def resolve_source_crop_row_ids(
    source_group: zarr.Group,
    crop_group: zarr.Group | None,
    *,
    total_rois: int,
    frame_indices: object | None = None,
    allow_direct_row_fallback: bool = True,
) -> object | None:
    """Resolve explicit source crop rows, synthesizing only verified direct alignment.

    Modern mask/keypoint consumers should use this array instead of assuming row
    ``N`` in a downstream run maps to row ``N`` in ``crop_runs/<source_crop_run>``.
    For legacy direct crop-row-aligned runs, the helper synthesizes ``0..N-1``
    only when the referenced crop run has the same row count and, when available,
    matching ``frame_indices``.
    """

    payload = _read_array(source_group, SOURCE_CROP_ROW_IDS_ARRAY)
    if payload is not None:
        source_array, data = payload
        _validate_source_crop_row_frame_alignment(
            np.asarray(data, dtype=np.int64).reshape(-1),
            crop_group=crop_group,
            frame_indices=frame_indices,
            total_rois=total_rois,
        )
        return source_array

    if not allow_direct_row_fallback or crop_group is None:
        return None
    crop_frame_indices = crop_group.get("frame_indices")
    if crop_frame_indices is None:
        return None
    if int(crop_frame_indices.shape[0]) != int(total_rois):
        return None
    fallback = direct_source_crop_row_ids(total_rois)
    _validate_source_crop_row_frame_alignment(
        fallback,
        crop_group=crop_group,
        frame_indices=frame_indices,
        total_rois=total_rois,
    )
    return fallback


def write_direct_source_crop_row_ids(
    target_group: zarr.Group,
    *,
    total_rois: int,
    overwrite: bool = True,
) -> None:
    """Write ``source_crop_row_ids`` for outputs that preserve crop row order."""

    data = direct_source_crop_row_ids(total_rois)
    _write_array(target_group, SOURCE_CROP_ROW_IDS_ARRAY, data, source_array=None, overwrite=overwrite)


def _write_array(
    target_group: zarr.Group,
    name: str,
    data: np.ndarray,
    *,
    source_array: object | None,
    overwrite: bool,
    geometry_preload_names: frozenset[str] | None = None,
) -> None:
    if geometry_preload_names is not None and name in geometry_preload_names:
        create_geometry_preload_array(target_group, name, data=data, overwrite=overwrite)
        return

    chunks = normalize_chunks_for_data(getattr(source_array, "chunks", None), data.shape)
    if overwrite and name in target_group:
        try:
            del target_group[name]
        except Exception:
            pass
    kwargs = {"data": data, "overwrite": overwrite}
    if chunks is not None:
        kwargs["chunks"] = chunks
    try:
        target_group.create_array(name, **kwargs)
    except TypeError:
        kwargs.pop("chunks", None)
        target_group.create_array(name, **kwargs)


def copy_row_lineage_array(
    target_group: zarr.Group,
    source_group: zarr.Group,
    name: str,
    *,
    total_rois: Optional[int] = None,
    overwrite: bool = True,
    use_geometry_preload_profile: bool = False,
) -> bool:
    payload = _read_array(source_group, name)
    if payload is None:
        return False
    source_array, data = payload
    validate_row_lineage_array(name, data, total_rois=total_rois)
    _write_array(
        target_group,
        name,
        data,
        source_array=source_array,
        overwrite=overwrite,
        geometry_preload_names=(
            CRIMSON_LINEAGE_PRELOAD_ARRAYS if use_geometry_preload_profile else None
        ),
    )
    return True


def copy_row_lineage_arrays(
    target_group: zarr.Group,
    source_group: zarr.Group,
    *,
    names: Iterable[str] = ROW_LINEAGE_ARRAYS,
    total_rois: Optional[int] = None,
    overwrite: bool = True,
    use_geometry_preload_profile: bool = False,
) -> RowLineageCopyResult:
    copied = []
    missing = []
    for name in names:
        if copy_row_lineage_array(
            target_group,
            source_group,
            name,
            total_rois=total_rois,
            overwrite=overwrite,
            use_geometry_preload_profile=use_geometry_preload_profile,
        ):
            copied.append(str(name))
        else:
            missing.append(str(name))
    return RowLineageCopyResult(copied=tuple(copied), missing=tuple(missing))


def copy_row_lineage_arrays_from_sources(
    target_group: zarr.Group,
    source_arrays: Mapping[str, object | None],
    *,
    names: Iterable[str] = ROW_LINEAGE_ARRAYS,
    total_rois: Optional[int] = None,
    overwrite: bool = True,
    use_geometry_preload_profile: bool = False,
) -> RowLineageCopyResult:
    copied = []
    missing = []
    for name in names:
        source_array = source_arrays.get(name)
        if source_array is None:
            missing.append(str(name))
            continue
        data = np.asarray(source_array[:])  # type: ignore[index]
        validate_row_lineage_array(name, data, total_rois=total_rois)
        _write_array(
            target_group,
            name,
            data,
            source_array=source_array,
            overwrite=overwrite,
            geometry_preload_names=(
                CRIMSON_LINEAGE_PRELOAD_ARRAYS if use_geometry_preload_profile else None
            ),
        )
        copied.append(str(name))
    return RowLineageCopyResult(copied=tuple(copied), missing=tuple(missing))


def copy_row_lineage_arrays_with_fallback(
    target_group: zarr.Group,
    primary_group: zarr.Group,
    fallback_group: zarr.Group,
    *,
    names: Iterable[str] = ROW_LINEAGE_ARRAYS,
    total_rois: Optional[int] = None,
    overwrite: bool = True,
    use_geometry_preload_profile: bool = False,
) -> RowLineageCopyResult:
    copied = []
    missing = []
    fallback_copied = []
    for name in names:
        primary = _read_array(primary_group, name)
        fallback = _read_array(fallback_group, name)
        if primary is not None and fallback is not None:
            _primary_array, primary_data = primary
            _fallback_array, fallback_data = fallback
            if primary_data.shape != fallback_data.shape or not np.array_equal(primary_data, fallback_data):
                raise ValueError(f"row-lineage mismatch for '{name}' between primary and fallback sources")

        chosen = primary if primary is not None else fallback
        if chosen is None:
            missing.append(str(name))
            continue
        source_array, data = chosen
        validate_row_lineage_array(name, data, total_rois=total_rois)
        _write_array(
            target_group,
            name,
            data,
            source_array=source_array,
            overwrite=overwrite,
            geometry_preload_names=(
                CRIMSON_LINEAGE_PRELOAD_ARRAYS if use_geometry_preload_profile else None
            ),
        )
        copied.append(str(name))
        if primary is None:
            fallback_copied.append(str(name))
    return RowLineageCopyResult(
        copied=tuple(copied),
        missing=tuple(missing),
        fallback_copied=tuple(fallback_copied),
    )


def assert_optional_row_lineage_equal(
    reference: Mapping[str, object],
    other: Mapping[str, object],
    *,
    names: Iterable[str] = ROW_LINEAGE_ARRAYS,
    require_presence_match: bool = True,
) -> None:
    for name in names:
        reference_array = reference.get(name)
        other_array = other.get(name)
        if reference_array is None and other_array is None:
            continue
        if reference_array is None or other_array is None:
            if require_presence_match:
                raise ValueError(f"Alignment mismatch: one source is missing {name}.")
            continue
        reference_data = np.asarray(reference_array[:])
        other_data = np.asarray(other_array[:])
        if reference_data.shape != other_data.shape:
            raise ValueError(f"Alignment mismatch for {name}: {reference_data.shape} != {other_data.shape}.")
        if not np.array_equal(reference_data, other_data):
            raise ValueError(f"Alignment mismatch for {name}")


def _assert_optional_row_lineage_sources_equal(
    reference_arrays: Mapping[str, object | None],
    other_arrays: Mapping[str, object | None],
    *,
    names: Iterable[str],
    require_presence_match: bool,
) -> None:
    for name in names:
        reference_array = reference_arrays.get(name)
        other_array = other_arrays.get(name)
        if reference_array is None and other_array is None:
            continue
        if reference_array is None or other_array is None:
            if require_presence_match:
                raise ValueError(f"Alignment mismatch: one source is missing {name}.")
            continue
        reference_data = np.asarray(reference_array[:])  # type: ignore[index]
        other_data = np.asarray(other_array[:])  # type: ignore[index]
        if reference_data.shape != other_data.shape:
            raise ValueError(f"Alignment mismatch for {name}: {reference_data.shape} != {other_data.shape}.")
        if not np.array_equal(reference_data, other_data):
            raise ValueError(f"Alignment mismatch for {name}")


def assert_row_lineage_alignment_equal(
    reference: Mapping[str, object],
    other: Mapping[str, object],
) -> None:
    assert_optional_row_lineage_equal(
        reference,
        other,
        names=ROW_ALIGNMENT_ARRAYS,
        require_presence_match=True,
    )
    assert_optional_row_lineage_equal(
        reference,
        other,
        names=ROW_IDENTITY_ARRAYS,
        require_presence_match=False,
    )


def assert_row_lineage_sources_equal(
    reference_arrays: Mapping[str, object | None],
    other_arrays: Mapping[str, object | None],
) -> None:
    _assert_optional_row_lineage_sources_equal(
        reference_arrays,
        other_arrays,
        names=ROW_ALIGNMENT_ARRAYS,
        require_presence_match=True,
    )
    _assert_optional_row_lineage_sources_equal(
        reference_arrays,
        other_arrays,
        names=ROW_IDENTITY_ARRAYS,
        require_presence_match=False,
    )
