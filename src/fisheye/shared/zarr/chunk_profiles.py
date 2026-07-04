"""Shared Zarr chunk profiles for Palette writer surfaces.

These helpers are intentionally conservative.  They apply only to small,
read-mostly metadata tables that are loaded in full by consumers such as
Crimson, not to dense pixels, masks, or editable row-oriented payloads.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


GEOMETRY_PRELOAD_STORAGE_PROFILE_ID = "geometry_preload_v1"
GEOMETRY_PRELOAD_CHUNK_POLICY_VERSION = GEOMETRY_PRELOAD_STORAGE_PROFILE_ID
GEOMETRY_PRELOAD_ROW_CHUNK = 16_384
GEOMETRY_PRELOAD_STORAGE_CLASS = "small_readmostly_preload_table"
GEOMETRY_PRELOAD_REASON = (
    "Crimson-validated read-mostly metadata surface; consumers preload or "
    "defer-load the full array rather than seeking one physical chunk per frame."
)

CRIMSON_LINEAGE_PRELOAD_ARRAYS = frozenset(
    {
        "detection_indices",
        "frame_indices",
        "instance_key",
        "source_detect_row_index",
        "source_refined_row_ids",
    }
)


def geometry_preload_chunks_for_shape(
    shape: Sequence[int],
    *,
    row_chunk: int = GEOMETRY_PRELOAD_ROW_CHUNK,
) -> tuple[int, ...] | None:
    """Return row-major chunks for a small table-like array shape."""

    shape_tuple = tuple(int(dim) for dim in shape)
    if not shape_tuple:
        return None
    first_dim = max(1, shape_tuple[0])
    first_chunk = max(1, min(int(row_chunk), first_dim))
    trailing = tuple(max(1, int(dim)) for dim in shape_tuple[1:])
    return (first_chunk, *trailing)


def geometry_preload_chunks_for_data(
    data: Any,
    *,
    row_chunk: int = GEOMETRY_PRELOAD_ROW_CHUNK,
) -> tuple[int, ...] | None:
    """Return row-major chunks using ``data.shape`` or ``np.asarray(data).shape``."""

    shape = getattr(data, "shape", None)
    if shape is None:
        shape = np.asarray(data).shape
    return geometry_preload_chunks_for_shape(shape, row_chunk=row_chunk)


def geometry_preload_attrs(
    *,
    reason: str = GEOMETRY_PRELOAD_REASON,
    row_chunk: int = GEOMETRY_PRELOAD_ROW_CHUNK,
) -> dict[str, Any]:
    """Return provenance attrs for arrays/groups using this storage profile."""

    return {
        "storage_profile_id": GEOMETRY_PRELOAD_STORAGE_PROFILE_ID,
        "chunk_policy_version": GEOMETRY_PRELOAD_CHUNK_POLICY_VERSION,
        "storage_profile_class": GEOMETRY_PRELOAD_STORAGE_CLASS,
        "storage_profile_row_chunk": int(row_chunk),
        "storage_profile_reason": str(reason),
    }


def stamp_geometry_preload_attrs(
    node: Any,
    *,
    reason: str = GEOMETRY_PRELOAD_REASON,
    row_chunk: int = GEOMETRY_PRELOAD_ROW_CHUNK,
) -> None:
    """Stamp a Zarr array or group with geometry-preload profile attrs."""

    attrs = getattr(node, "attrs", None)
    if attrs is None:
        return
    attrs.update(geometry_preload_attrs(reason=reason, row_chunk=row_chunk))


def create_geometry_preload_array(
    group: Any,
    name: str,
    *,
    data: Any | None = None,
    shape: Sequence[int] | None = None,
    dtype: Any | None = None,
    overwrite: bool = True,
    reason: str = GEOMETRY_PRELOAD_REASON,
    row_chunk: int = GEOMETRY_PRELOAD_ROW_CHUNK,
    attrs: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> Any:
    """Create a small read-mostly array with the geometry-preload profile."""

    if "chunks" not in kwargs:
        chunk_shape = (
            geometry_preload_chunks_for_shape(shape, row_chunk=row_chunk)
            if shape is not None
            else geometry_preload_chunks_for_data(data, row_chunk=row_chunk)
        )
        if chunk_shape is not None:
            kwargs["chunks"] = chunk_shape
    if data is not None:
        kwargs["data"] = data
    if shape is not None:
        kwargs["shape"] = tuple(int(dim) for dim in shape)
    if dtype is not None:
        kwargs["dtype"] = dtype
    kwargs["overwrite"] = overwrite
    array = group.create_array(name, **kwargs)
    stamp_geometry_preload_attrs(array, reason=reason, row_chunk=row_chunk)
    if attrs:
        array.attrs.update(dict(attrs))
    return array
