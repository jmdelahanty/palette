"""Dense materialization boundary for Palette mask storage encodings."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import zarr

from .mask_bitpack import (
    MASK_BITPACKED_AXIS,
    MASK_BITPACKED_BITORDER,
    MASK_BITPACKED_ENCODING,
    MASK_BITPACKED_LAYOUT,
    MASK_BITPACKED_SCHEMA_ID,
    MASK_BITPACKED_VALUE_SEMANTICS,
    pack_binary_mask_stack,
    packed_width_bytes,
    unpack_binary_mask_stack,
)
from .mask_rle import (
    MASK_RLE_ENCODING,
    MASK_RLE_SCHEMA_ID,
    MASK_RLE_VALUE_SEMANTICS,
    EncodedMaskComponentStack,
    concatenate_encoded_mask_components,
    concatenate_encoded_mask_component_stacks,
    decode_binary_mask_rle,
    encode_mask_component_rle,
    encode_mask_component_stack_rle,
)
from .zarr_helpers import zarr_attrs_dict, zarr_child_group, zarr_group_keys


class MaskStoreError(ValueError):
    """Raised when a mask store cannot resolve or materialize masks."""


MASK_RLE_VALIDATION_MODES = ("full", "invariants", "none")


def _mask_labels(group: Any, channel_count: int | None = None) -> tuple[str, ...]:
    attrs = zarr_attrs_dict(group)
    for key in ("mask_labels", "component_names", "labels"):
        value = attrs.get(key)
        if isinstance(value, (list, tuple)) and value:
            return tuple(str(item) for item in value)
    if channel_count is None:
        return ()
    return tuple(f"component_{idx}" for idx in range(int(channel_count)))


def _normalize_indices(values: Sequence[int] | np.ndarray | slice | int | None, size: int) -> np.ndarray:
    if values is None:
        return np.arange(int(size), dtype=np.int64)
    if isinstance(values, slice):
        return np.arange(int(size), dtype=np.int64)[values]
    if isinstance(values, (int, np.integer)):
        arr = np.asarray([int(values)], dtype=np.int64)
    else:
        arr = np.asarray(list(values), dtype=np.int64).reshape(-1)
    if np.any(arr < 0) or np.any(arr >= int(size)):
        raise MaskStoreError(f"Index selection {arr.tolist()} is out of bounds for size {size}.")
    return arr


def _safe_component_group_name(component_name: str, component_index: int) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in str(component_name)).strip("_")
    if not safe:
        safe = "component"
    return f"{int(component_index):02d}_{safe}"


def _contiguous_slice(indices: np.ndarray) -> slice | None:
    values = np.asarray(indices, dtype=np.int64).reshape(-1)
    if values.size == 0:
        return slice(0, 0)
    start = int(values[0])
    stop = start + int(values.size)
    if np.array_equal(values, np.arange(start, stop, dtype=np.int64)):
        return slice(start, stop)
    return None


def _create_array(group: Any, name: str, data: np.ndarray, *, chunks: tuple[int, ...] | int | None = None) -> Any:
    kwargs: dict[str, Any] = {"data": data, "overwrite": True}
    if chunks is not None:
        kwargs["chunks"] = chunks
    return group.create_array(name, **kwargs)


def _component_rle_logical_bytes(encoded: EncodedMaskComponentStack) -> int:
    total = 0
    for component in encoded.components:
        total += int(
            component.counts.nbytes
            + component.indptr.nbytes
            + component.present.nbytes
            + component.area_px.nbytes
            + component.bbox_xyxy.nbytes
        )
    return int(total)


def _normalize_component_names(values: Sequence[str] | None) -> tuple[str, ...] | None:
    if values is None:
        return None
    names = tuple(str(value) for value in values)
    if not names:
        raise MaskStoreError("Component selection must not be empty.")
    if len(set(names)) != len(names):
        raise MaskStoreError(f"Component selection contains duplicates: {names!r}.")
    return names


def _component_selection_indices(
    all_names: Sequence[str],
    selected_names: Sequence[str],
) -> tuple[int, ...]:
    names = tuple(str(value) for value in all_names)
    selected = tuple(str(value) for value in selected_names)
    name_to_index = {name: idx for idx, name in enumerate(names)}
    missing = [name for name in selected if name not in name_to_index]
    if missing:
        raise MaskStoreError(f"Unknown mask component(s) {missing!r}; available components are {names!r}.")
    return tuple(int(name_to_index[name]) for name in selected)


def _normalize_row_selection(values: Sequence[int] | np.ndarray | slice | int | None, size: int) -> tuple[int, ...] | None:
    if values is None:
        return None
    return tuple(int(value) for value in _normalize_indices(values, size))


def _normalize_mask_rle_validation_mode(
    validation_mode: str | None,
    *,
    validate_roundtrip: bool,
) -> str:
    if validation_mode is None:
        return "full" if bool(validate_roundtrip) else "none"
    mode = str(validation_mode).strip().lower()
    if mode not in MASK_RLE_VALIDATION_MODES:
        raise ValueError(
            f"mask_rle validation_mode must be one of {MASK_RLE_VALIDATION_MODES}; got {validation_mode!r}."
        )
    return mode


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _attr_truthy(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def mask_store_encodings(*, has_dense: bool, has_rle: bool, has_bitpacked: bool = False) -> list[str]:
    encodings: list[str] = []
    if has_dense:
        encodings.append("dense_uint8")
    if has_bitpacked:
        encodings.append("bitpacked_binary_v1")
    if has_rle:
        encodings.append("component_rle_v1")
    return encodings


def update_mask_storage_attrs(run_group: Any, *, has_dense: bool, has_rle: bool, has_bitpacked: bool = False) -> None:
    encodings = mask_store_encodings(has_dense=has_dense, has_rle=has_rle, has_bitpacked=has_bitpacked)
    run_group.attrs["mask_store_encodings"] = encodings
    run_group.attrs["mask_storage_encoding"] = "+".join(encodings) if encodings else "missing"
    run_group.attrs["masks_roi_materialized"] = bool(has_dense)
    run_group.attrs["mask_bitpacked_materialized"] = bool(has_bitpacked)
    run_group.attrs["mask_rle_materialized"] = bool(has_rle)


def materialize_dense_masks_roi_from_store(
    run_group: Any,
    *,
    chunk_size: int = 256,
    overwrite: bool = False,
    source_path: str = "",
) -> dict[str, object]:
    """Create or refresh the dense ``masks_roi`` compatibility cache from compact storage."""

    has_dense = "masks_roi" in run_group
    has_rle = "mask_rle" in run_group
    has_bitpacked = "mask_bitpacked" in run_group
    if has_dense and not overwrite:
        update_mask_storage_attrs(run_group, has_dense=True, has_rle=has_rle, has_bitpacked=has_bitpacked)
        return {"status": "existing", "encoding": "dense_uint8", "has_dense_after": True}
    if not has_rle and not has_bitpacked:
        raise MaskStoreError(
            "Dense masks_roi cannot be materialized because no compact mask store is available."
        )

    mask_store = open_mask_store(run_group, source_path=source_path, prefer="auto")
    if mask_store.encoding == "dense_uint8":
        raise MaskStoreError("Dense masks_roi cannot be materialized because a compact mask store is unavailable.")

    if has_dense:
        del run_group["masks_roi"]

    from .subject_mask_chunks import refined_subject_mask_storage_chunks

    n_rows, n_channels, height, width = mask_store.shape
    chunks = refined_subject_mask_storage_chunks(int(n_rows), int(height), int(width))
    dense = run_group.create_array(
        "masks_roi",
        shape=(int(n_rows), int(n_channels), int(height), int(width)),
        chunks=chunks,
        dtype="uint8",
        fill_value=0,
        overwrite=True,
    )

    row_chunk = max(1, int(chunk_size))
    rows_written = 0
    for start in range(0, int(n_rows), row_chunk):
        stop = min(int(n_rows), start + row_chunk)
        dense[start:stop] = mask_store.read_dense(rows=slice(start, stop))
        rows_written += int(stop - start)

    run_group.attrs["masks_roi_materialized"] = True
    run_group.attrs["masks_roi_materialized_from"] = mask_store.storage_surface
    run_group.attrs["masks_roi_materialized_at_utc"] = _utc_now()
    run_group.attrs["masks_roi_materialization_chunk_size"] = int(row_chunk)
    update_mask_storage_attrs(run_group, has_dense=True, has_rle=has_rle, has_bitpacked=has_bitpacked)
    return {
        "status": "materialized",
        "encoding": "dense_uint8",
        "source_encoding": mask_store.encoding,
        "rows": int(n_rows),
        "channels": int(n_channels),
        "shape": [int(n_rows), int(n_channels), int(height), int(width)],
        "chunks": [int(value) for value in chunks],
        "rows_written": int(rows_written),
        "has_dense_after": True,
    }


def _delete_attr_if_present(attrs: Any, key: str) -> None:
    try:
        del attrs[key]
    except KeyError:
        return
    except Exception:
        if hasattr(attrs, "__contains__") and key not in attrs:
            return
        raise


def clear_mask_rle_stale_attrs(run_group: Any) -> None:
    """Mark the compact mask store current after it has been regenerated."""

    run_group.attrs["mask_rle_stale"] = False
    for key in (
        "mask_rle_stale_at_utc",
        "mask_rle_stale_reason",
        "mask_rle_stale_component_names",
        "mask_rle_stale_row_count",
        "mask_rle_stale_row_min",
        "mask_rle_stale_row_max",
    ):
        _delete_attr_if_present(run_group.attrs, key)


def is_mask_rle_stale(run_group: Any) -> bool:
    """Return whether the compact RLE mirror is known stale relative to dense masks."""

    return _attr_truthy(zarr_attrs_dict(run_group).get("mask_rle_stale"))


def mark_mask_rle_stale_attrs(
    run_group: Any,
    *,
    reason: str,
    updated_components: Sequence[str] | None = None,
    updated_rows: Sequence[int] = (),
    updated_at_utc: str | None = None,
) -> bool:
    """Mark compact ``mask_rle`` stale after authoritative dense-mask edits.

    Returns ``False`` when the run has no compact store to mark.
    """

    if "mask_rle" not in run_group:
        return False
    row_values = [int(row) for row in updated_rows]
    run_group.attrs["mask_rle_stale"] = True
    run_group.attrs["mask_rle_stale_at_utc"] = str(updated_at_utc or _utc_now())
    run_group.attrs["mask_rle_stale_reason"] = str(reason)
    run_group.attrs["mask_rle_stale_component_names"] = (
        [str(component) for component in updated_components] if updated_components is not None else []
    )
    run_group.attrs["mask_rle_stale_row_count"] = int(len(row_values))
    if row_values:
        run_group.attrs["mask_rle_stale_row_min"] = int(min(row_values))
        run_group.attrs["mask_rle_stale_row_max"] = int(max(row_values))
    return True


def _dense_mask_shape(dense_masks: Any) -> tuple[int, int, int, int]:
    shape = tuple(int(value) for value in dense_masks.shape)
    if len(shape) == 3:
        n_rows, height, width = shape
        return (n_rows, 1, height, width)
    if len(shape) == 4:
        n_rows, n_channels, height, width = shape
        return (n_rows, n_channels, height, width)
    raise MaskStoreError(f"Expected dense masks with shape (N,C,H,W) or (N,H,W), got {shape!r}.")


def _mask_bitpacked_logical_bytes(mask_bitpacked: Any) -> int | None:
    packed = mask_bitpacked.get("masks_packed") if hasattr(mask_bitpacked, "get") else None
    if packed is None or not hasattr(packed, "shape"):
        return None
    try:
        return int(np.prod(tuple(int(value) for value in packed.shape), dtype=np.int64))
    except Exception:
        return None


def validate_bitpacked_mask_store_invariants(
    run_group: Any,
    *,
    expected_shape: Sequence[int] | None = None,
    component_names: Sequence[str] | None = None,
    source_path: str = "",
) -> dict[str, object]:
    """Validate compact ``mask_bitpacked`` structure without dense pixel round-trips."""

    bitpacked_group = zarr_child_group(run_group, "mask_bitpacked")
    if bitpacked_group is None:
        raise MaskStoreError(f"{source_path or '<mask_run>'} is missing mask_bitpacked.")
    attrs = zarr_attrs_dict(bitpacked_group)

    def fail(message: str) -> None:
        raise MaskStoreError(f"Bitpacked mask invariant validation failed: {message}")

    if attrs.get("schema_id") != MASK_BITPACKED_SCHEMA_ID:
        fail(f"schema_id {attrs.get('schema_id')!r} != {MASK_BITPACKED_SCHEMA_ID!r}.")
    if attrs.get("mask_encoding") != MASK_BITPACKED_ENCODING:
        fail(f"mask_encoding {attrs.get('mask_encoding')!r} != {MASK_BITPACKED_ENCODING!r}.")
    if attrs.get("layout") != MASK_BITPACKED_LAYOUT:
        fail(f"layout {attrs.get('layout')!r} != {MASK_BITPACKED_LAYOUT!r}.")
    if attrs.get("packed_axis") != MASK_BITPACKED_AXIS:
        fail(f"packed_axis {attrs.get('packed_axis')!r} != {MASK_BITPACKED_AXIS!r}.")
    if attrs.get("packed_bitorder") != MASK_BITPACKED_BITORDER:
        fail(f"packed_bitorder {attrs.get('packed_bitorder')!r} != {MASK_BITPACKED_BITORDER!r}.")

    logical_shape_raw = attrs.get("logical_shape")
    encoded_shape_raw = attrs.get("encoded_shape")
    if not isinstance(logical_shape_raw, (list, tuple)) or len(logical_shape_raw) != 4:
        fail("logical_shape must contain [N, C, H, W].")
    if not isinstance(encoded_shape_raw, (list, tuple)) or len(encoded_shape_raw) != 4:
        fail("encoded_shape must contain [N, C, H, packed_width_bytes].")
    logical_shape = tuple(int(value) for value in logical_shape_raw)
    encoded_shape = tuple(int(value) for value in encoded_shape_raw)
    n_rows, n_channels, height, width = logical_shape
    if min(logical_shape) <= 0:
        fail(f"logical_shape dimensions must be positive, got {logical_shape!r}.")
    expected_packed_width = packed_width_bytes(int(width))
    if encoded_shape != (int(n_rows), int(n_channels), int(height), int(expected_packed_width)):
        fail(f"encoded_shape {encoded_shape!r} does not match expected packed logical shape.")
    packed_width_attr = int(attrs.get("packed_width_bytes", -1))
    if packed_width_attr != int(expected_packed_width):
        fail(f"packed_width_bytes {packed_width_attr} != expected {expected_packed_width}.")
    names_raw = attrs.get("component_names")
    if not isinstance(names_raw, (list, tuple)) or len(names_raw) != int(n_channels):
        fail("component_names attr must match channel count.")
    names = tuple(str(value) for value in names_raw)
    if component_names is not None and tuple(str(value) for value in component_names) != names:
        fail(f"component_names {names!r} do not match expected {tuple(component_names)!r}.")
    if expected_shape is not None:
        expected = tuple(int(value) for value in expected_shape)
        if len(expected) == 3:
            expected = (int(expected[0]), 1, int(expected[1]), int(expected[2]))
        if expected != logical_shape:
            fail(f"logical_shape {logical_shape!r} != expected {expected!r}.")
    packed = bitpacked_group.get("masks_packed") if hasattr(bitpacked_group, "get") else None
    if packed is None or not hasattr(packed, "shape"):
        fail("missing masks_packed array.")
    if tuple(int(value) for value in packed.shape) != encoded_shape:
        fail(f"masks_packed shape {tuple(int(value) for value in packed.shape)!r} != encoded_shape {encoded_shape!r}.")
    if str(getattr(packed, "dtype", "")) != "uint8":
        fail(f"masks_packed dtype {getattr(packed, 'dtype', None)!r} != uint8.")
    return {
        "status": "passed",
        "mode": "invariants",
        "rows_checked": int(n_rows),
        "channels_checked": int(n_channels),
        "shape": [int(n_rows), int(n_channels), int(height), int(width)],
        "encoded_shape": [int(value) for value in encoded_shape],
        "logical_dense_validation": False,
    }


def validate_bitpacked_mask_store_against_dense(
    run_group: Any,
    dense_masks: Any,
    *,
    row_chunk_size: int = 256,
    source_path: str = "",
) -> dict[str, object]:
    """Validate compact ``mask_bitpacked`` by streaming dense-vs-decoded comparisons."""

    n_rows, n_channels, height, width = _dense_mask_shape(dense_masks)
    mask_store = open_mask_store(run_group, source_path=source_path, prefer="bitpacked")
    expected_shape = (int(n_rows), int(n_channels), int(height), int(width))
    if tuple(int(value) for value in mask_store.shape) != expected_shape:
        raise MaskStoreError(
            f"Bitpacked round-trip shape mismatch: decoded {mask_store.shape!r} != dense {expected_shape!r}."
        )
    rows_checked = 0
    chunks_checked = 0
    row_chunk = max(1, int(row_chunk_size))
    for start in range(0, int(n_rows), row_chunk):
        stop = min(int(n_rows), start + row_chunk)
        expected = np.asarray(dense_masks[start:stop], dtype=np.uint8)
        if expected.ndim == 3:
            expected = expected[:, None, :, :]
        expected = (expected > 0).astype(np.uint8, copy=False)
        actual = mask_store.read_dense(rows=slice(start, stop))
        if not np.array_equal(actual, expected):
            mismatch = np.argwhere(actual != expected)
            first = mismatch[0].tolist() if mismatch.size else []
            raise MaskStoreError(
                "Bitpacked round-trip validation failed for "
                f"rows {start}:{stop}; first mismatch at local index {first}."
            )
        rows_checked += int(stop - start)
        chunks_checked += 1
    return {
        "status": "passed",
        "rows_checked": int(rows_checked),
        "channels_checked": int(n_channels),
        "chunks_checked": int(chunks_checked),
        "row_chunk_size": int(row_chunk),
    }


def _validate_bitpacked_selection_against_dense(
    run_group: Any,
    dense_masks: Any,
    *,
    rows: Sequence[int],
    channels: Sequence[int],
    source_path: str = "",
) -> dict[str, object]:
    mask_store = open_mask_store(run_group, source_path=source_path, prefer="bitpacked")
    row_indices = tuple(int(value) for value in rows)
    channel_indices = tuple(int(value) for value in channels)
    if not row_indices or not channel_indices:
        return {
            "status": "passed",
            "rows_checked": 0,
            "channels_checked": int(len(channel_indices)),
            "cells_checked": 0,
        }
    for row_idx in row_indices:
        expected = np.asarray(dense_masks[int(row_idx) : int(row_idx) + 1, list(channel_indices)], dtype=np.uint8)
        if expected.ndim == 3:
            expected = expected[:, None, :, :]
        expected = (expected > 0).astype(np.uint8, copy=False)
        actual = mask_store.read_dense(rows=[int(row_idx)], channels=list(channel_indices))
        if not np.array_equal(actual, expected):
            mismatch = np.argwhere(actual != expected)
            first = mismatch[0].tolist() if mismatch.size else []
            raise MaskStoreError(
                "Bitpacked scoped refresh validation failed for "
                f"row {int(row_idx)} channels {list(channel_indices)!r}; first mismatch at {first}."
            )
    return {
        "status": "passed",
        "rows_checked": int(len(row_indices)),
        "channels_checked": int(len(channel_indices)),
        "cells_checked": int(len(row_indices) * len(channel_indices)),
    }


def write_bitpacked_mask_store_from_dense(
    run_group: Any,
    dense_masks: Any,
    *,
    component_names: Sequence[str] | None = None,
    overwrite: bool = True,
    encode_row_chunk_size: int = 256,
    validate_roundtrip: bool = True,
    validation_mode: str | None = None,
    extra_attrs: Mapping[str, Any] | None = None,
    progress_callback: Callable[..., None] | None = None,
) -> dict[str, object]:
    """Stream dense masks into fixed-size width-bitpacked storage."""

    n_rows, n_channels, height, width = _dense_mask_shape(dense_masks)
    names = tuple(str(value) for value in component_names) if component_names is not None else _mask_labels(run_group, n_channels)
    if len(names) != int(n_channels):
        raise MaskStoreError(f"Expected {n_channels} component names for dense masks, got {len(names)}.")
    validation_mode = _normalize_mask_rle_validation_mode(
        validation_mode,
        validate_roundtrip=bool(validate_roundtrip),
    )
    if "mask_bitpacked" in run_group:
        if not overwrite:
            raise MaskStoreError("mask_bitpacked already exists; pass overwrite=True to replace it.")
        del run_group["mask_bitpacked"]

    def emit(event: str, **payload: object) -> None:
        if progress_callback is not None:
            progress_callback(str(event), **payload)

    row_chunk = max(1, int(encode_row_chunk_size))
    packed_width = packed_width_bytes(int(width))
    bitpacked_group = run_group.create_group("mask_bitpacked")
    bitpacked_group.attrs.update(
        {
            "schema_id": MASK_BITPACKED_SCHEMA_ID,
            "mask_encoding": MASK_BITPACKED_ENCODING,
            "mask_value_semantics": MASK_BITPACKED_VALUE_SEMANTICS,
            "layout": MASK_BITPACKED_LAYOUT,
            "logical_shape": [int(n_rows), int(n_channels), int(height), int(width)],
            "encoded_shape": [int(n_rows), int(n_channels), int(height), int(packed_width)],
            "component_names": list(names),
            "packed_axis": MASK_BITPACKED_AXIS,
            "packed_bitorder": MASK_BITPACKED_BITORDER,
            "packed_width_bytes": int(packed_width),
        }
    )
    if extra_attrs:
        bitpacked_group.attrs.update(dict(extra_attrs))
    packed_chunks = (
        min(max(1, int(n_rows)), row_chunk),
        1,
        int(height),
        int(packed_width),
    )
    packed_array = bitpacked_group.create_array(
        "masks_packed",
        shape=(int(n_rows), int(n_channels), int(height), int(packed_width)),
        chunks=packed_chunks,
        dtype="uint8",
        fill_value=0,
        overwrite=True,
    )
    phase_seconds: dict[str, float] = {}
    emit(
        "bitpacked_encode_start",
        rows=int(n_rows),
        channels=int(n_channels),
        height=int(height),
        width=int(width),
        encode_row_chunk_size=int(row_chunk),
    )
    encode_started = time.perf_counter()
    rows_written = 0
    for start in range(0, int(n_rows), row_chunk):
        stop = min(int(n_rows), start + row_chunk)
        dense_chunk = np.asarray(dense_masks[start:stop], dtype=np.uint8)
        if dense_chunk.ndim == 3:
            dense_chunk = dense_chunk[:, None, :, :]
        packed_array[start:stop] = pack_binary_mask_stack(dense_chunk)
        rows_written += int(stop - start)
    phase_seconds["encode_write"] = float(time.perf_counter() - encode_started)
    logical_bytes = _mask_bitpacked_logical_bytes(bitpacked_group)
    emit(
        "bitpacked_encode_end",
        duration_seconds=phase_seconds["encode_write"],
        rows_written=int(rows_written),
        logical_bytes=int(logical_bytes or 0),
    )

    update_mask_storage_attrs(
        run_group,
        has_dense="masks_roi" in run_group,
        has_rle="mask_rle" in run_group,
        has_bitpacked=True,
    )
    run_group.attrs["mask_bitpacked_schema_id"] = MASK_BITPACKED_SCHEMA_ID
    run_group.attrs["mask_bitpacked_encoding"] = MASK_BITPACKED_ENCODING
    run_group.attrs["mask_bitpacked_layout"] = MASK_BITPACKED_LAYOUT
    summary: dict[str, object] = {
        "status": "written",
        "encoding": MASK_BITPACKED_ENCODING,
        "layout": MASK_BITPACKED_LAYOUT,
        "source_encoding": "dense_uint8",
        "rows": int(n_rows),
        "channels": int(n_channels),
        "shape": [int(n_rows), int(n_channels), int(height), int(width)],
        "encoded_shape": [int(n_rows), int(n_channels), int(height), int(packed_width)],
        "component_names": list(names),
        "logical_bytes": int(logical_bytes or 0),
        "encode_row_chunk_size": int(row_chunk),
        "validation_mode": str(validation_mode),
        "phase_seconds": dict(phase_seconds),
    }
    if validation_mode == "full":
        validation_started = time.perf_counter()
        validation = validate_bitpacked_mask_store_against_dense(
            run_group,
            dense_masks,
            row_chunk_size=row_chunk,
            source_path=str(extra_attrs.get("source_path", "")) if extra_attrs else "",
        )
        phase_seconds["roundtrip_validation"] = float(time.perf_counter() - validation_started)
        summary["phase_seconds"] = dict(phase_seconds)
        summary["mask_bitpacked_validation"] = {"mode": "full", **validation}
        summary["roundtrip_validation"] = validation
    elif validation_mode == "invariants":
        validation_started = time.perf_counter()
        validation = validate_bitpacked_mask_store_invariants(
            run_group,
            expected_shape=(int(n_rows), int(n_channels), int(height), int(width)),
            component_names=names,
            source_path=str(extra_attrs.get("source_path", "")) if extra_attrs else "",
        )
        phase_seconds["invariant_validation"] = float(time.perf_counter() - validation_started)
        summary["phase_seconds"] = dict(phase_seconds)
        summary["mask_bitpacked_validation"] = validation
        summary["roundtrip_validation"] = {"status": "skipped", "reason": "validation_mode=invariants"}
    else:
        summary["mask_bitpacked_validation"] = {
            "status": "skipped",
            "mode": "none",
            "reason": "validation_mode=none",
        }
        summary["roundtrip_validation"] = {"status": "skipped", "reason": "validation_mode=none"}
    return summary


def refresh_bitpacked_mask_store_from_dense(
    run_group: Any,
    *,
    component_names: Sequence[str] | None = None,
    refresh_components: Sequence[str] | None = None,
    refresh_rows: Sequence[int] | np.ndarray | slice | int | None = None,
    encode_row_chunk_size: int = 256,
    source_path: str = "",
    validation_mode: str | None = "invariants",
) -> dict[str, object]:
    """Refresh compact ``mask_bitpacked`` from dense ``masks_roi``.

    When no component or row selection is supplied, this performs a full
    overwrite through ``write_bitpacked_mask_store_from_dense``. With a
    selection, it rewrites only the selected fixed-size packed row/channel
    slices, preserving unrelated rows and components.
    """

    dense = run_group.get("masks_roi") if hasattr(run_group, "get") else None
    if dense is None or not hasattr(dense, "shape"):
        raise MaskStoreError("Cannot refresh compact mask_bitpacked because dense masks_roi is unavailable.")
    n_rows, n_channels, height, width = _dense_mask_shape(dense)
    names = tuple(str(value) for value in component_names) if component_names is not None else _mask_labels(run_group, n_channels)
    if len(names) != int(n_channels):
        raise MaskStoreError(f"Expected {n_channels} component names for dense masks, got {len(names)}.")
    selected_names = _normalize_component_names(refresh_components)
    selected_indices = (
        _component_selection_indices(names, selected_names)
        if selected_names is not None
        else tuple(range(int(n_channels)))
    )
    selected_rows = _normalize_row_selection(refresh_rows, int(n_rows))
    selected_row_indices = selected_rows if selected_rows is not None else tuple(range(int(n_rows)))
    validation_mode = _normalize_mask_rle_validation_mode(validation_mode, validate_roundtrip=False)

    if selected_names is None and selected_rows is None:
        summary = write_bitpacked_mask_store_from_dense(
            run_group,
            dense,
            component_names=names,
            overwrite=True,
            encode_row_chunk_size=max(1, int(encode_row_chunk_size)),
            validation_mode=validation_mode,
            extra_attrs={
                "source_path": str(source_path),
                "refreshed_from": "masks_roi",
                "refreshed_at_utc": _utc_now(),
            },
        )
        refreshed_at_utc = _utc_now()
        run_group.attrs["mask_bitpacked_refreshed_from"] = "masks_roi"
        run_group.attrs["mask_bitpacked_refreshed_at_utc"] = refreshed_at_utc
        run_group.attrs["mask_bitpacked_refresh_source_path"] = str(source_path)
        run_group.attrs["mask_bitpacked_refresh_row_count"] = int(n_rows)
        summary.update(
            {
                "status": "bitpacked_refreshed",
                "refresh_scope": "all_components_all_rows",
                "rows": int(n_rows),
                "channels": int(n_channels),
                "has_bitpacked_after": True,
            }
        )
        return summary

    bitpacked_group = zarr_child_group(run_group, "mask_bitpacked")
    if bitpacked_group is None:
        raise MaskStoreError(
            "Cannot refresh selected compact mask_bitpacked rows/components because mask_bitpacked does not exist. "
            "Run a full --refresh-bitpacked first."
        )
    bitpacked_attrs = zarr_attrs_dict(bitpacked_group)
    raw_names = bitpacked_attrs.get("component_names")
    if tuple(str(value) for value in raw_names or ()) != names:
        raise MaskStoreError(
            "Cannot refresh selected compact mask_bitpacked components because dense mask labels "
            f"{names!r} do not match existing mask_bitpacked component_names {raw_names!r}."
        )
    validate_bitpacked_mask_store_invariants(
        run_group,
        expected_shape=(int(n_rows), int(n_channels), int(height), int(width)),
        component_names=names,
        source_path=str(source_path),
    )
    packed = bitpacked_group["masks_packed"]
    for row_idx in selected_row_indices:
        for component_idx in selected_indices:
            dense_cell = np.asarray(
                dense[int(row_idx) : int(row_idx) + 1, int(component_idx) : int(component_idx) + 1],
                dtype=np.uint8,
            )
            packed[int(row_idx) : int(row_idx) + 1, int(component_idx) : int(component_idx) + 1] = (
                pack_binary_mask_stack(dense_cell)
            )

    validation: dict[str, object]
    if validation_mode == "none":
        validation = {"status": "skipped", "mode": "none", "reason": "validation_mode=none"}
    else:
        validation = _validate_bitpacked_selection_against_dense(
            run_group,
            dense,
            rows=selected_row_indices,
            channels=selected_indices,
            source_path=str(source_path),
        )
        validation["mode"] = "selection"

    refreshed_at_utc = _utc_now()
    bitpacked_group.attrs["refreshed_from"] = "masks_roi"
    bitpacked_group.attrs["refreshed_at_utc"] = refreshed_at_utc
    bitpacked_group.attrs["refreshed_component_names"] = [names[idx] for idx in selected_indices]
    bitpacked_group.attrs["refreshed_component_count"] = int(len(selected_indices))
    bitpacked_group.attrs["refreshed_row_count"] = int(len(selected_row_indices))
    if selected_row_indices:
        bitpacked_group.attrs["refreshed_row_min"] = int(min(selected_row_indices))
        bitpacked_group.attrs["refreshed_row_max"] = int(max(selected_row_indices))
    update_mask_storage_attrs(run_group, has_dense=True, has_rle="mask_rle" in run_group, has_bitpacked=True)
    run_group.attrs["mask_bitpacked_refreshed_from"] = "masks_roi"
    run_group.attrs["mask_bitpacked_refreshed_at_utc"] = refreshed_at_utc
    run_group.attrs["mask_bitpacked_refresh_source_path"] = str(source_path)
    run_group.attrs["mask_bitpacked_refresh_row_count"] = int(len(selected_row_indices))
    run_group.attrs["mask_bitpacked_refreshed_component_names"] = [names[idx] for idx in selected_indices]
    run_group.attrs["mask_bitpacked_refreshed_component_count"] = int(len(selected_indices))
    return {
        "status": "bitpacked_refreshed",
        "encoding": MASK_BITPACKED_ENCODING,
        "source_encoding": "dense_uint8",
        "refresh_scope": "selection",
        "refreshed_component_names": [names[idx] for idx in selected_indices],
        "refreshed_component_indices": [int(value) for value in selected_indices],
        "refreshed_rows": [int(value) for value in selected_row_indices],
        "rows": int(len(selected_row_indices)),
        "channels": int(len(selected_indices)),
        "shape": [int(n_rows), int(n_channels), int(height), int(width)],
        "has_bitpacked_after": True,
        "mask_bitpacked_validation": validation,
    }


def _write_encoded_component_rle_groups(
    rle_group: Any,
    encoded: EncodedMaskComponentStack,
    *,
    count_chunk_bytes: int,
    row_chunk_size: int,
) -> None:
    row_chunk = max(1, int(row_chunk_size))
    count_chunk_values = max(1, int(count_chunk_bytes) // np.dtype(np.uint32).itemsize)
    components_group = rle_group.require_group("components")
    for component in encoded.components:
        group_name = _safe_component_group_name(component.component_name, component.component_index)
        if group_name in components_group:
            del components_group[group_name]
        component_group = components_group.create_group(group_name)
        component_group.attrs.update(
            {
                "component_name": str(component.component_name),
                "component_index": int(component.component_index),
                "encoded_shape_hw": [int(component.shape_hw[0]), int(component.shape_hw[1])],
            }
        )
        _create_array(
            component_group,
            "counts",
            component.counts,
            chunks=(min(max(1, int(component.counts.size)), count_chunk_values),),
        )
        _create_array(
            component_group,
            "indptr",
            component.indptr,
            chunks=(min(max(1, int(component.indptr.size)), row_chunk + 1),),
        )
        _create_array(
            component_group,
            "present",
            component.present,
            chunks=(min(max(1, int(component.present.shape[0])), row_chunk),),
        )
        _create_array(
            component_group,
            "area_px",
            component.area_px,
            chunks=(min(max(1, int(component.area_px.shape[0])), row_chunk),),
        )
        _create_array(
            component_group,
            "bbox_xyxy",
            component.bbox_xyxy,
            chunks=(min(max(1, int(component.bbox_xyxy.shape[0])), row_chunk), 4),
        )


def write_encoded_component_rle_mask_store(
    run_group: Any,
    encoded: EncodedMaskComponentStack,
    *,
    overwrite: bool = True,
    count_chunk_bytes: int = 4 * 1024 * 1024,
    row_chunk_size: int = 256,
    extra_attrs: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    """Write an encoded component-separated RLE mask store under ``mask_rle``."""

    if "mask_rle" in run_group:
        if not overwrite:
            raise MaskStoreError("mask_rle already exists; pass overwrite=True to replace it.")
        del run_group["mask_rle"]

    rle_group = run_group.create_group("mask_rle")
    component_names = [component.component_name for component in encoded.components]
    rle_group.attrs.update(
        {
            "schema_id": MASK_RLE_SCHEMA_ID,
            "mask_encoding": MASK_RLE_ENCODING,
            "mask_value_semantics": MASK_RLE_VALUE_SEMANTICS,
            "encoded_shape_hw": [int(encoded.shape_hw[0]), int(encoded.shape_hw[1])],
            "layout": "component_groups",
            "component_names": list(component_names),
            "n_rows": int(encoded.n_rows),
            "component_count": int(len(encoded.components)),
        }
    )
    if extra_attrs:
        rle_group.attrs.update(dict(extra_attrs))

    _write_encoded_component_rle_groups(
        rle_group,
        encoded,
        count_chunk_bytes=count_chunk_bytes,
        row_chunk_size=row_chunk_size,
    )

    update_mask_storage_attrs(
        run_group,
        has_dense="masks_roi" in run_group,
        has_rle=True,
        has_bitpacked="mask_bitpacked" in run_group,
    )
    run_group.attrs["mask_rle_schema_id"] = MASK_RLE_SCHEMA_ID
    run_group.attrs["mask_rle_encoding"] = MASK_RLE_ENCODING
    run_group.attrs["mask_rle_layout"] = "component_groups"
    clear_mask_rle_stale_attrs(run_group)

    return {
        "status": "written",
        "encoding": "component_rle_v1",
        "layout": "component_groups",
        "n_rows": int(encoded.n_rows),
        "component_count": int(len(encoded.components)),
        "component_names": list(component_names),
        "shape_hw": [int(encoded.shape_hw[0]), int(encoded.shape_hw[1])],
        "logical_bytes": _component_rle_logical_bytes(encoded),
    }


def _component_rle_empty_stack(
    *,
    n_rows: int,
    n_channels: int,
    height: int,
    width: int,
    names: Sequence[str],
) -> EncodedMaskComponentStack:
    return encode_mask_component_stack_rle(
        np.zeros((int(n_rows), int(n_channels), int(height), int(width)), dtype=np.uint8),
        component_names=tuple(str(value) for value in names),
    )


def _encode_dense_component_rle_serial(
    dense_masks: Any,
    *,
    n_rows: int,
    n_channels: int,
    height: int,
    width: int,
    names: Sequence[str],
    row_chunk: int,
    row_offset: int = 0,
) -> EncodedMaskComponentStack:
    shards: list[EncodedMaskComponentStack] = []
    for start in range(int(row_offset), int(row_offset) + int(n_rows), int(row_chunk)):
        stop = min(int(row_offset) + int(n_rows), start + int(row_chunk))
        dense_chunk = np.asarray(dense_masks[start:stop], dtype=np.uint8)
        if dense_chunk.ndim == 3:
            dense_chunk = dense_chunk[:, None, :, :]
        shards.append(encode_mask_component_stack_rle(dense_chunk, component_names=tuple(names)))
    if shards:
        return concatenate_encoded_mask_component_stacks(shards)
    return _component_rle_empty_stack(
        n_rows=0,
        n_channels=int(n_channels),
        height=int(height),
        width=int(width),
        names=names,
    )


def _row_shards(n_rows: int, *, shard_count: int, row_chunk: int) -> list[tuple[int, int]]:
    if int(n_rows) <= 0:
        return []
    rows_per_chunk = max(1, int(row_chunk))
    total_chunks = int(np.ceil(int(n_rows) / rows_per_chunk))
    requested = max(1, min(int(shard_count), total_chunks))
    edges = [
        min(int(n_rows), int(np.floor(chunk_idx * total_chunks / requested)) * rows_per_chunk)
        for chunk_idx in range(requested + 1)
    ]
    edges[-1] = int(n_rows)
    shards: list[tuple[int, int]] = []
    for start, stop in zip(edges[:-1], edges[1:]):
        if int(stop) > int(start):
            shards.append((int(start), int(stop)))
    return shards


def _encode_component_rle_process_worker(
    zarr_path: str,
    run_path: str,
    source_array: str,
    start_row: int,
    stop_row: int,
    row_chunk: int,
    component_names: tuple[str, ...],
) -> EncodedMaskComponentStack:
    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    run_group = root[str(run_path)]
    dense_masks = run_group[str(source_array)]
    shape = tuple(int(value) for value in dense_masks.shape)
    if len(shape) == 3:
        _total_rows, height, width = shape
        n_channels = 1
    elif len(shape) == 4:
        _total_rows, n_channels, height, width = shape
    else:
        raise MaskStoreError(f"Expected dense masks with shape (N,C,H,W) or (N,H,W), got {shape!r}.")
    return _encode_dense_component_rle_serial(
        dense_masks,
        n_rows=max(0, int(stop_row) - int(start_row)),
        n_channels=int(n_channels),
        height=int(height),
        width=int(width),
        names=component_names,
        row_chunk=max(1, int(row_chunk)),
        row_offset=int(start_row),
    )


def _encode_dense_component_rle(
    dense_masks: Any,
    *,
    n_rows: int,
    n_channels: int,
    height: int,
    width: int,
    names: Sequence[str],
    row_chunk: int,
    encode_workers: int = 1,
    source_zarr_path: str | Path | None = None,
    source_run_path: str | None = None,
    source_array: str = "masks_roi",
) -> tuple[EncodedMaskComponentStack, dict[str, object]]:
    """Encode dense masks in row shards, returning one parent-writable payload."""

    worker_count = max(1, int(encode_workers))
    if worker_count <= 1 or source_zarr_path is None or not source_run_path or int(n_rows) <= int(row_chunk):
        encoded = _encode_dense_component_rle_serial(
            dense_masks,
            n_rows=int(n_rows),
            n_channels=int(n_channels),
            height=int(height),
            width=int(width),
            names=names,
            row_chunk=max(1, int(row_chunk)),
        )
        return encoded, {
            "encode_backend": "serial",
            "encode_workers": 1,
            "encode_shard_count": 1 if int(n_rows) > 0 else 0,
        }

    shards = _row_shards(int(n_rows), shard_count=worker_count, row_chunk=max(1, int(row_chunk)))
    if len(shards) <= 1:
        encoded = _encode_dense_component_rle_serial(
            dense_masks,
            n_rows=int(n_rows),
            n_channels=int(n_channels),
            height=int(height),
            width=int(width),
            names=names,
            row_chunk=max(1, int(row_chunk)),
        )
        return encoded, {
            "encode_backend": "serial",
            "encode_workers": 1,
            "encode_shard_count": 1 if int(n_rows) > 0 else 0,
        }

    component_names = tuple(str(value) for value in names)
    with ProcessPoolExecutor(max_workers=min(worker_count, len(shards))) as pool:
        futures = [
            pool.submit(
                _encode_component_rle_process_worker,
                str(source_zarr_path),
                str(source_run_path),
                str(source_array),
                int(start),
                int(stop),
                int(row_chunk),
                component_names,
            )
            for start, stop in shards
        ]
        encoded_shards = [future.result() for future in futures]
    encoded = concatenate_encoded_mask_component_stacks(encoded_shards)
    return encoded, {
        "encode_backend": "process_shards",
        "encode_workers": int(min(worker_count, len(shards))),
        "encode_shard_count": int(len(shards)),
        "encode_row_shards": [[int(start), int(stop)] for start, stop in shards],
    }


def _encode_dense_selected_component_rle_serial(
    dense_masks: Any,
    *,
    n_rows: int,
    height: int,
    width: int,
    names: Sequence[str],
    selected_indices: Sequence[int],
    row_chunk: int,
) -> EncodedMaskComponentStack:
    """Encode selected dense channels while preserving their original component indices."""

    all_names = tuple(str(value) for value in names)
    indices = tuple(int(value) for value in selected_indices)
    component_shards: dict[int, list[Any]] = {component_idx: [] for component_idx in indices}
    if int(n_rows) <= 0:
        components = tuple(
            encode_mask_component_rle(
                np.zeros((0, int(height), int(width)), dtype=np.uint8),
                component_name=all_names[component_idx],
                component_index=component_idx,
            )
            for component_idx in indices
        )
        return EncodedMaskComponentStack(components=components, shape_hw=(int(height), int(width)), n_rows=0)

    row_step = max(1, int(row_chunk))
    for start in range(0, int(n_rows), row_step):
        stop = min(int(n_rows), start + row_step)
        for component_idx in indices:
            if len(getattr(dense_masks, "shape", ())) == 3:
                if component_idx != 0:
                    raise MaskStoreError(
                        f"Cannot refresh component index {component_idx} from single-channel dense masks."
                    )
                dense_chunk = np.asarray(dense_masks[start:stop], dtype=np.uint8)
            else:
                dense_chunk = np.asarray(dense_masks[start:stop, component_idx], dtype=np.uint8)
            component_shards[component_idx].append(
                encode_mask_component_rle(
                    dense_chunk,
                    component_name=all_names[component_idx],
                    component_index=component_idx,
                )
            )

    components = tuple(
        concatenate_encoded_mask_components(component_shards[component_idx])
        for component_idx in indices
    )
    return EncodedMaskComponentStack(
        components=components,
        shape_hw=(int(height), int(width)),
        n_rows=int(n_rows),
    )


def _clear_mask_rle_stale_for_refreshed_components(
    run_group: Any,
    *,
    refreshed_components: Sequence[str],
    all_components: Sequence[str],
) -> None:
    """Clear stale markers only when the recorded stale scope was fully refreshed."""

    if not is_mask_rle_stale(run_group):
        run_group.attrs["mask_rle_stale"] = False
        return

    refreshed = {str(value) for value in refreshed_components}
    all_names = {str(value) for value in all_components}
    raw_stale = run_group.attrs.get("mask_rle_stale_component_names")
    if not isinstance(raw_stale, (list, tuple)) or not raw_stale:
        if refreshed == all_names:
            clear_mask_rle_stale_attrs(run_group)
        return

    remaining = [str(name) for name in raw_stale if str(name) not in refreshed]
    if not remaining:
        clear_mask_rle_stale_attrs(run_group)
        return
    run_group.attrs["mask_rle_stale"] = True
    run_group.attrs["mask_rle_stale_component_names"] = remaining


def write_component_rle_mask_store_from_dense(
    run_group: Any,
    dense_masks: Any,
    *,
    component_names: Sequence[str] | None = None,
    overwrite: bool = True,
    encode_row_chunk_size: int = 256,
    encode_workers: int = 1,
    source_zarr_path: str | Path | None = None,
    source_run_path: str | None = None,
    source_array: str = "masks_roi",
    validate_roundtrip: bool = True,
    validation_mode: str | None = None,
    count_chunk_bytes: int = 4 * 1024 * 1024,
    extra_attrs: Mapping[str, Any] | None = None,
    progress_callback: Callable[..., None] | None = None,
) -> dict[str, object]:
    """Stream dense masks into component-separated RLE storage.

    The dense source is read in row chunks so full ``(N,C,H,W)`` masks do not need
    to fit in host memory at once.
    """

    shape = tuple(int(v) for v in dense_masks.shape)
    if len(shape) == 3:
        n_rows, height, width = shape
        n_channels = 1
    elif len(shape) == 4:
        n_rows, n_channels, height, width = shape
    else:
        raise MaskStoreError(f"Expected dense masks with shape (N,C,H,W) or (N,H,W), got {shape!r}.")

    names = tuple(str(value) for value in component_names) if component_names is not None else _mask_labels(run_group, n_channels)
    if len(names) != int(n_channels):
        raise MaskStoreError(f"Expected {n_channels} component names for dense masks, got {len(names)}.")
    validation_mode = _normalize_mask_rle_validation_mode(
        validation_mode,
        validate_roundtrip=bool(validate_roundtrip),
    )

    def emit(event: str, **payload: object) -> None:
        if progress_callback is not None:
            progress_callback(str(event), **payload)

    phase_seconds: dict[str, float] = {}
    row_chunk = max(1, int(encode_row_chunk_size))
    encode_started = time.perf_counter()
    emit(
        "component_rle_encode_start",
        rows=int(n_rows),
        channels=int(n_channels),
        height=int(height),
        width=int(width),
        encode_row_chunk_size=int(row_chunk),
        encode_workers=max(1, int(encode_workers)),
        source_zarr_path=str(source_zarr_path) if source_zarr_path is not None else None,
        source_run_path=str(source_run_path) if source_run_path is not None else None,
        source_array=str(source_array),
    )
    encoded, encode_summary = _encode_dense_component_rle(
        dense_masks,
        n_rows=int(n_rows),
        n_channels=int(n_channels),
        height=int(height),
        width=int(width),
        names=names,
        row_chunk=row_chunk,
        encode_workers=max(1, int(encode_workers)),
        source_zarr_path=source_zarr_path,
        source_run_path=source_run_path,
        source_array=source_array,
    )
    phase_seconds["encode"] = float(time.perf_counter() - encode_started)
    emit(
        "component_rle_encode_end",
        duration_seconds=phase_seconds["encode"],
        logical_bytes=_component_rle_logical_bytes(encoded),
        **encode_summary,
    )
    write_started = time.perf_counter()
    emit(
        "component_rle_parent_write_start",
        rows=int(encoded.n_rows),
        components=int(len(encoded.components)),
        logical_bytes=_component_rle_logical_bytes(encoded),
    )
    summary = write_encoded_component_rle_mask_store(
        run_group,
        encoded,
        overwrite=overwrite,
        count_chunk_bytes=count_chunk_bytes,
        row_chunk_size=row_chunk,
        extra_attrs=extra_attrs,
    )
    phase_seconds["parent_write"] = float(time.perf_counter() - write_started)
    emit(
        "component_rle_parent_write_end",
        duration_seconds=phase_seconds["parent_write"],
        stored_logical_bytes=int(summary.get("logical_bytes") or 0),
        component_count=int(summary.get("component_count") or 0),
    )
    summary.update(
        {
            "source_encoding": "dense_uint8",
            "encode_row_chunk_size": int(row_chunk),
            "validation_mode": str(validation_mode),
            "mask_rle_validation_mode": str(validation_mode),
            "roundtrip_validation_requested": validation_mode == "full",
            "phase_seconds": dict(phase_seconds),
            **encode_summary,
        }
    )
    if validation_mode == "full":
        validation_started = time.perf_counter()
        emit(
            "component_rle_roundtrip_validation_start",
            rows=int(n_rows),
            channels=int(n_channels),
            row_chunk_size=int(row_chunk),
        )
        validation = validate_component_rle_mask_store_against_dense(
            run_group,
            dense_masks,
            row_chunk_size=row_chunk,
            source_path=str(extra_attrs.get("source_path", "")) if extra_attrs else "",
        )
        phase_seconds["roundtrip_validation"] = float(time.perf_counter() - validation_started)
        emit(
            "component_rle_roundtrip_validation_end",
            duration_seconds=phase_seconds["roundtrip_validation"],
            **validation,
        )
        summary["phase_seconds"] = dict(phase_seconds)
        summary["roundtrip_validation"] = validation
        summary["mask_rle_validation"] = {"mode": "full", **validation}
    elif validation_mode == "invariants":
        validation_started = time.perf_counter()
        emit(
            "component_rle_invariant_validation_start",
            rows=int(n_rows),
            channels=int(n_channels),
        )
        validation = validate_component_rle_mask_store_invariants(
            run_group,
            expected_shape=(int(n_rows), int(n_channels), int(height), int(width)),
            component_names=names,
            source_path=str(extra_attrs.get("source_path", "")) if extra_attrs else "",
        )
        phase_seconds["invariant_validation"] = float(time.perf_counter() - validation_started)
        emit(
            "component_rle_invariant_validation_end",
            duration_seconds=phase_seconds["invariant_validation"],
            **validation,
        )
        summary["phase_seconds"] = dict(phase_seconds)
        summary["mask_rle_validation"] = validation
        summary["roundtrip_validation"] = {
            "status": "skipped",
            "reason": "validation_mode=invariants",
        }
    else:
        summary["mask_rle_validation"] = {
            "status": "skipped",
            "mode": "none",
            "reason": "validation_mode=none",
        }
        summary["roundtrip_validation"] = {
            "status": "skipped",
            "reason": "validation_mode=none",
        }
    return summary


def validate_component_rle_mask_store_invariants(
    run_group: Any,
    *,
    expected_shape: Sequence[int] | None = None,
    component_names: Sequence[str] | None = None,
    source_path: str = "",
) -> dict[str, object]:
    """Validate compact ``mask_rle`` structure without dense pixel round-trips."""

    rle_group = zarr_child_group(run_group, "mask_rle")
    if rle_group is None:
        raise MaskStoreError(f"{source_path or '<mask_run>'} is missing mask_rle.")
    attrs = zarr_attrs_dict(rle_group)

    def fail(message: str) -> None:
        raise MaskStoreError(f"Component RLE invariant validation failed: {message}")

    if attrs.get("schema_id") != MASK_RLE_SCHEMA_ID:
        fail(f"schema_id {attrs.get('schema_id')!r} != {MASK_RLE_SCHEMA_ID!r}.")
    if attrs.get("mask_encoding") != MASK_RLE_ENCODING:
        fail(f"mask_encoding {attrs.get('mask_encoding')!r} != {MASK_RLE_ENCODING!r}.")
    if attrs.get("layout") != "component_groups":
        fail(f"layout {attrs.get('layout')!r} != 'component_groups'.")

    raw_shape = attrs.get("encoded_shape_hw")
    if not isinstance(raw_shape, (list, tuple)) or len(raw_shape) != 2:
        fail("encoded_shape_hw must contain [height, width].")
    height, width = (int(raw_shape[0]), int(raw_shape[1]))
    if height < 1 or width < 1:
        fail(f"encoded_shape_hw must be positive, got {(height, width)!r}.")
    total_pixels = int(height * width)

    raw_names = attrs.get("component_names")
    if not isinstance(raw_names, (list, tuple)) or not raw_names:
        fail("component_names attr must be a non-empty list.")
    names = tuple(str(value) for value in raw_names)
    if component_names is not None and tuple(str(value) for value in component_names) != names:
        fail(f"component_names {names!r} do not match expected {tuple(component_names)!r}.")

    try:
        n_rows = int(attrs["n_rows"])
        component_count = int(attrs["component_count"])
    except Exception as exc:
        raise MaskStoreError("Component RLE invariant validation failed: missing n_rows/component_count attrs.") from exc
    if n_rows < 0:
        fail(f"n_rows must be non-negative, got {n_rows}.")
    if component_count != len(names):
        fail(f"component_count {component_count} != len(component_names) {len(names)}.")

    if expected_shape is not None:
        expected = tuple(int(value) for value in expected_shape)
        if len(expected) == 3:
            expected = (int(expected[0]), 1, int(expected[1]), int(expected[2]))
        if len(expected) != 4:
            fail(f"expected_shape must have 3 or 4 dimensions, got {expected_shape!r}.")
        if expected != (n_rows, component_count, height, width):
            fail(f"shape {(n_rows, component_count, height, width)!r} != expected {expected!r}.")

    components_parent = zarr_child_group(rle_group, "components")
    if components_parent is None:
        fail("missing components group.")
    groups_by_index = _component_groups_by_index(rle_group)
    total_count_values = 0
    total_present_rows = 0
    components_checked = 0
    for component_idx, component_name in enumerate(names):
        component_group = groups_by_index.get(int(component_idx))
        if component_group is None:
            fail(f"missing component group for index {component_idx} ({component_name!r}).")
        component_attrs = zarr_attrs_dict(component_group)
        if str(component_attrs.get("component_name")) != str(component_name):
            fail(
                f"component {component_idx} name {component_attrs.get('component_name')!r} "
                f"!= {component_name!r}."
            )
        if int(component_attrs.get("component_index", -1)) != int(component_idx):
            fail(f"component {component_name!r} has wrong component_index attr.")
        for array_name in ("counts", "indptr", "present", "area_px", "bbox_xyxy"):
            if array_name not in component_group:
                fail(f"component {component_name!r} missing {array_name}.")

        counts = np.asarray(component_group["counts"][:])
        indptr = np.asarray(component_group["indptr"][:], dtype=np.int64)
        present = np.asarray(component_group["present"][:], dtype=bool)
        area_px = np.asarray(component_group["area_px"][:], dtype=np.int64)
        bbox_xyxy = np.asarray(component_group["bbox_xyxy"][:], dtype=np.int64)

        if indptr.shape != (n_rows + 1,):
            fail(f"component {component_name!r} indptr shape {indptr.shape!r} != {(n_rows + 1,)!r}.")
        if present.shape != (n_rows,):
            fail(f"component {component_name!r} present shape {present.shape!r} != {(n_rows,)!r}.")
        if area_px.shape != (n_rows,):
            fail(f"component {component_name!r} area_px shape {area_px.shape!r} != {(n_rows,)!r}.")
        if bbox_xyxy.shape != (n_rows, 4):
            fail(f"component {component_name!r} bbox_xyxy shape {bbox_xyxy.shape!r} != {(n_rows, 4)!r}.")
        if int(indptr[0]) != 0:
            fail(f"component {component_name!r} indptr[0] must be 0.")
        row_count_lengths = np.diff(indptr)
        if np.any(row_count_lengths <= 0):
            first = int(np.flatnonzero(row_count_lengths <= 0)[0])
            fail(f"component {component_name!r} row {first} has no RLE count payload.")
        if np.any(row_count_lengths < 0):
            first = int(np.flatnonzero(row_count_lengths < 0)[0])
            fail(f"component {component_name!r} indptr decreases at row {first}.")
        if int(indptr[-1]) != int(counts.shape[0]):
            fail(
                f"component {component_name!r} indptr[-1] {int(indptr[-1])} "
                f"!= counts length {int(counts.shape[0])}."
            )
        if counts.size:
            row_sums = np.add.reduceat(np.asarray(counts, dtype=np.int64), indptr[:-1])
            bad_sum = np.flatnonzero(row_sums != total_pixels)
            if bad_sum.size:
                first = int(bad_sum[0])
                fail(
                    f"component {component_name!r} row {first} RLE count sum "
                    f"{int(row_sums[first])} != {total_pixels}."
                )
        if np.any(area_px < 0) or np.any(area_px > total_pixels):
            first = int(np.flatnonzero((area_px < 0) | (area_px > total_pixels))[0])
            fail(f"component {component_name!r} row {first} has out-of-bounds area {int(area_px[first])}.")
        if not np.array_equal(present, area_px > 0):
            first = int(np.flatnonzero(present != (area_px > 0))[0])
            fail(f"component {component_name!r} row {first} present does not match area_px > 0.")

        x0 = bbox_xyxy[:, 0]
        y0 = bbox_xyxy[:, 1]
        x1 = bbox_xyxy[:, 2]
        y1 = bbox_xyxy[:, 3]
        present_rows = present
        if np.any(present_rows):
            sane = (
                (x0[present_rows] >= 0)
                & (y0[present_rows] >= 0)
                & (x1[present_rows] <= width)
                & (y1[present_rows] <= height)
                & (x1[present_rows] > x0[present_rows])
                & (y1[present_rows] > y0[present_rows])
            )
            if not np.all(sane):
                first_present = np.flatnonzero(present_rows)
                first_bad = int(first_present[np.flatnonzero(~sane)[0]])
                fail(f"component {component_name!r} row {first_bad} has invalid present bbox.")
        absent_rows = ~present_rows
        if np.any(absent_rows) and np.any(bbox_xyxy[absent_rows] != 0):
            first_absent = np.flatnonzero(absent_rows)
            first_bad = int(first_absent[np.flatnonzero(np.any(bbox_xyxy[absent_rows] != 0, axis=1))[0]])
            fail(f"component {component_name!r} row {first_bad} has non-zero absent bbox.")

        components_checked += 1
        total_count_values += int(counts.shape[0])
        total_present_rows += int(np.count_nonzero(present))

    return {
        "status": "passed",
        "mode": "invariants",
        "rows_checked": int(n_rows),
        "channels_checked": int(component_count),
        "components_checked": int(components_checked),
        "counts_values_checked": int(total_count_values),
        "present_rows": int(total_present_rows),
        "shape": [int(n_rows), int(component_count), int(height), int(width)],
        "logical_dense_validation": False,
    }


def validate_component_rle_mask_store_against_dense(
    run_group: Any,
    dense_masks: Any,
    *,
    row_chunk_size: int = 256,
    source_path: str = "",
) -> dict[str, object]:
    """Validate compact ``mask_rle`` by streaming dense-vs-decoded comparisons."""

    dense_shape = tuple(int(v) for v in dense_masks.shape)
    if len(dense_shape) == 3:
        n_rows, height, width = dense_shape
        n_channels = 1
        squeeze_channel = True
    elif len(dense_shape) == 4:
        n_rows, n_channels, height, width = dense_shape
        squeeze_channel = False
    else:
        raise MaskStoreError(f"Expected dense masks with shape (N,C,H,W) or (N,H,W), got {dense_shape!r}.")

    mask_store = open_mask_store(
        run_group,
        source_path=source_path,
        prefer="rle",
        allow_stale_rle=True,
    )
    expected_shape = (int(n_rows), int(n_channels), int(height), int(width))
    if tuple(int(value) for value in mask_store.shape) != expected_shape:
        raise MaskStoreError(
            f"RLE round-trip shape mismatch: decoded {mask_store.shape!r} != dense {expected_shape!r}."
        )

    row_chunk = max(1, int(row_chunk_size))
    rows_checked = 0
    chunks_checked = 0
    for start in range(0, int(n_rows), row_chunk):
        stop = min(int(n_rows), start + row_chunk)
        expected = np.asarray(dense_masks[start:stop], dtype=np.uint8)
        if squeeze_channel:
            expected = expected[:, None, :, :]
        expected = (expected > 0).astype(np.uint8, copy=False)
        actual = mask_store.read_dense(rows=slice(start, stop))
        if not np.array_equal(actual, expected):
            mismatch = np.argwhere(actual != expected)
            first = mismatch[0].tolist() if mismatch.size else []
            raise MaskStoreError(
                "RLE round-trip validation failed for "
                f"rows {start}:{stop}; first mismatch at local index {first}."
            )
        rows_checked += int(stop - start)
        chunks_checked += 1

    return {
        "status": "passed",
        "rows_checked": int(rows_checked),
        "channels_checked": int(n_channels),
        "chunks_checked": int(chunks_checked),
        "row_chunk_size": int(row_chunk),
    }


def refresh_component_rle_mask_store_from_dense(
    run_group: Any,
    *,
    component_names: Sequence[str] | None = None,
    refresh_components: Sequence[str] | None = None,
    encode_row_chunk_size: int = 256,
    count_chunk_bytes: int = 4 * 1024 * 1024,
    source_path: str = "",
    clear_stale: bool = True,
) -> dict[str, object]:
    """Regenerate compact ``mask_rle`` from the current dense ``masks_roi``.

    When ``refresh_components`` is provided, only the selected component groups
    under ``mask_rle/components`` are replaced. This is the review/edit repair
    path for component-scoped dense edits.
    """

    dense = run_group.get("masks_roi") if hasattr(run_group, "get") else None
    if dense is None or not hasattr(dense, "shape"):
        raise MaskStoreError("Cannot refresh compact mask_rle because dense masks_roi is unavailable.")
    dense_shape = tuple(int(value) for value in dense.shape)
    if len(dense_shape) not in {3, 4}:
        raise MaskStoreError(f"Cannot refresh compact mask_rle from unsupported dense mask shape {dense_shape!r}.")
    if len(dense_shape) == 3:
        n_rows, height, width = dense_shape
        n_channels = 1
    else:
        n_rows, n_channels, height, width = dense_shape

    names = tuple(str(value) for value in component_names) if component_names is not None else _mask_labels(run_group, n_channels)
    if len(names) != int(n_channels):
        raise MaskStoreError(f"Expected {n_channels} component names for dense masks, got {len(names)}.")

    selected_names = _normalize_component_names(refresh_components)
    if selected_names is not None:
        rle_group = zarr_child_group(run_group, "mask_rle")
        if rle_group is None:
            raise MaskStoreError(
                "Cannot refresh selected compact mask_rle components because mask_rle does not exist. "
                "Run a full --refresh-rle first."
            )
        rle_attrs = zarr_attrs_dict(rle_group)
        raw_rle_names = rle_attrs.get("component_names")
        if tuple(str(value) for value in raw_rle_names or ()) != names:
            raise MaskStoreError(
                "Cannot refresh selected compact mask_rle components because dense mask labels "
                f"{names!r} do not match existing mask_rle component_names {raw_rle_names!r}."
            )
        raw_shape_hw = rle_attrs.get("encoded_shape_hw")
        if (
            not isinstance(raw_shape_hw, (list, tuple))
            or len(raw_shape_hw) != 2
            or (int(raw_shape_hw[0]), int(raw_shape_hw[1])) != (int(height), int(width))
            or int(rle_attrs.get("n_rows", -1)) != int(n_rows)
            or int(rle_attrs.get("component_count", -1)) != int(n_channels)
        ):
            raise MaskStoreError(
                "Cannot refresh selected compact mask_rle components because existing mask_rle "
                "shape metadata does not match dense masks_roi."
            )
        selected_indices = _component_selection_indices(names, selected_names)
        row_chunk = max(1, int(encode_row_chunk_size))
        encoded = _encode_dense_selected_component_rle_serial(
            dense,
            n_rows=int(n_rows),
            height=int(height),
            width=int(width),
            names=names,
            selected_indices=selected_indices,
            row_chunk=row_chunk,
        )
        _write_encoded_component_rle_groups(
            rle_group,
            encoded,
            count_chunk_bytes=count_chunk_bytes,
            row_chunk_size=row_chunk,
        )
        refreshed_at_utc = _utc_now()
        rle_group.attrs["refreshed_from"] = "masks_roi"
        rle_group.attrs["refreshed_at_utc"] = refreshed_at_utc
        rle_group.attrs["refreshed_component_names"] = list(selected_names)
        rle_group.attrs["refreshed_component_count"] = int(len(selected_names))
        update_mask_storage_attrs(run_group, has_dense=True, has_rle=True, has_bitpacked="mask_bitpacked" in run_group)
        run_group.attrs["mask_rle_refreshed_from"] = "masks_roi"
        run_group.attrs["mask_rle_refreshed_at_utc"] = refreshed_at_utc
        run_group.attrs["mask_rle_refresh_source_path"] = str(source_path)
        run_group.attrs["mask_rle_refresh_row_count"] = int(n_rows)
        run_group.attrs["mask_rle_refreshed_component_names"] = list(selected_names)
        run_group.attrs["mask_rle_refreshed_component_count"] = int(len(selected_names))
        if clear_stale:
            _clear_mask_rle_stale_for_refreshed_components(
                run_group,
                refreshed_components=selected_names,
                all_components=names,
            )
        validation = validate_component_rle_mask_store_invariants(
            run_group,
            expected_shape=(int(n_rows), int(n_channels), int(height), int(width)),
            component_names=names,
            source_path=str(source_path),
        )
        return {
            "status": "rle_refreshed",
            "encoding": "component_rle_v1",
            "source_encoding": "dense_uint8",
            "refresh_scope": "components",
            "refreshed_component_names": list(selected_names),
            "refreshed_component_indices": [int(value) for value in selected_indices],
            "component_count": int(len(selected_names)),
            "rows": int(n_rows),
            "channels": int(n_channels),
            "shape": [int(value) for value in dense_shape],
            "has_rle_after": True,
            "mask_rle_stale_after": bool(run_group.attrs.get("mask_rle_stale", False)),
            "logical_bytes": _component_rle_logical_bytes(encoded),
            "mask_rle_validation": validation,
        }

    summary = write_component_rle_mask_store_from_dense(
        run_group,
        dense,
        component_names=names,
        overwrite=True,
        count_chunk_bytes=count_chunk_bytes,
        encode_row_chunk_size=max(1, int(encode_row_chunk_size)),
        extra_attrs={
            "source_path": str(source_path),
            "refreshed_from": "masks_roi",
            "refreshed_at_utc": _utc_now(),
        },
    )
    update_mask_storage_attrs(run_group, has_dense=True, has_rle=True, has_bitpacked="mask_bitpacked" in run_group)
    refreshed_at_utc = _utc_now()
    run_group.attrs["mask_rle_refreshed_from"] = "masks_roi"
    run_group.attrs["mask_rle_refreshed_at_utc"] = refreshed_at_utc
    run_group.attrs["mask_rle_refresh_source_path"] = str(source_path)
    run_group.attrs["mask_rle_refresh_row_count"] = int(dense_shape[0])
    if clear_stale:
        clear_mask_rle_stale_attrs(run_group)
    summary.update(
        {
            "status": "rle_refreshed",
            "encoding": "component_rle_v1",
            "source_encoding": "dense_uint8",
            "refresh_scope": "all_components",
            "rows": int(dense_shape[0]),
            "channels": int(n_channels),
            "shape": [int(value) for value in dense_shape],
            "has_rle_after": True,
            "mask_rle_stale_after": bool(run_group.attrs.get("mask_rle_stale", False)),
        }
    )
    return summary


@dataclass(frozen=True)
class MaskStore:
    """Reader that materializes dense masks from the selected physical encoding."""

    group: Any
    encoding: str
    mask_labels: tuple[str, ...]
    shape: tuple[int, int, int, int]
    source_path: str
    dense_array: Any | None = None
    bitpacked_group: Any | None = None
    bitpacked_array: Any | None = None
    rle_group: Any | None = None
    component_groups: Mapping[int, Any] | None = None

    @property
    def n_rows(self) -> int:
        return int(self.shape[0])

    @property
    def n_channels(self) -> int:
        return int(self.shape[1])

    @property
    def shape_hw(self) -> tuple[int, int]:
        return (int(self.shape[2]), int(self.shape[3]))

    @property
    def storage_surface(self) -> str:
        """Physical mask surface backing this logical dense reader."""

        if self.encoding == "dense_uint8":
            return "masks_roi"
        if self.encoding == "bitpacked_binary_v1":
            return "mask_bitpacked"
        if self.encoding == "component_rle_v1":
            return "mask_rle"
        raise MaskStoreError(f"Unsupported mask store encoding {self.encoding!r}.")

    @property
    def storage_path(self) -> str:
        """Best-effort logical path to the selected physical mask surface."""

        surface = self.storage_surface
        base = str(self.source_path or "").rstrip("/")
        return f"{base}/{surface}" if base else surface

    def component_index(self, component_name: str) -> int:
        try:
            return self.mask_labels.index(str(component_name))
        except ValueError as exc:
            raise MaskStoreError(f"Component {component_name!r} is not present in mask store {self.source_path}.") from exc

    def resolve_channels(
        self,
        channels: Sequence[int | str] | int | str | slice | None = None,
    ) -> tuple[np.ndarray, tuple[str, ...]]:
        if channels is None:
            indices = np.arange(self.n_channels, dtype=np.int64)
        elif isinstance(channels, slice):
            indices = np.arange(self.n_channels, dtype=np.int64)[channels]
        elif isinstance(channels, str):
            indices = np.asarray([self.component_index(channels)], dtype=np.int64)
        elif isinstance(channels, (int, np.integer)):
            indices = _normalize_indices(int(channels), self.n_channels)
        else:
            resolved: list[int] = []
            for value in channels:
                if isinstance(value, str):
                    resolved.append(self.component_index(value))
                else:
                    resolved.append(int(value))
            indices = _normalize_indices(resolved, self.n_channels)
        return indices, tuple(self.mask_labels[int(idx)] for idx in indices)

    def read_dense(
        self,
        rows: Sequence[int] | np.ndarray | slice | int | None = None,
        channels: Sequence[int | str] | int | str | slice | None = None,
    ) -> np.ndarray:
        row_indices = _normalize_indices(rows, self.n_rows)
        channel_indices, _names = self.resolve_channels(channels)
        if self.encoding == "dense_uint8":
            return self._read_dense_array(row_indices, channel_indices)
        if self.encoding == "bitpacked_binary_v1":
            return self._read_bitpacked(row_indices, channel_indices)
        if self.encoding == "component_rle_v1":
            return self._read_component_rle(row_indices, channel_indices)
        raise MaskStoreError(f"Unsupported mask store encoding {self.encoding!r}.")

    def _read_dense_array(self, rows: np.ndarray, channels: np.ndarray) -> np.ndarray:
        if self.dense_array is None:
            raise MaskStoreError(f"{self.source_path} has no dense mask array.")
        dense_shape = tuple(int(value) for value in self.dense_array.shape)
        row_slice = _contiguous_slice(rows)
        channel_slice = _contiguous_slice(channels)
        if len(dense_shape) == 3:
            if np.any(channels != 0):
                raise MaskStoreError(f"{self.source_path} has a single-channel dense mask array.")
            if row_slice is not None:
                output_3d = np.asarray(self.dense_array[row_slice, :, :], dtype=np.uint8)
                return (output_3d[:, None, :, :] > 0).astype(np.uint8, copy=False)
            output = np.zeros((int(rows.size), int(channels.size), *self.shape_hw), dtype=np.uint8)
            for out_row, row_idx in enumerate(rows):
                output[out_row, 0] = np.asarray(self.dense_array[int(row_idx)], dtype=np.uint8)
            return (output > 0).astype(np.uint8, copy=False)
        if row_slice is not None and channel_slice is not None:
            output = np.asarray(self.dense_array[row_slice, channel_slice, :, :], dtype=np.uint8)
            return (output > 0).astype(np.uint8, copy=False)
        output = np.zeros((int(rows.size), int(channels.size), *self.shape_hw), dtype=np.uint8)
        for out_row, row_idx in enumerate(rows):
            for out_channel, channel_idx in enumerate(channels):
                output[out_row, out_channel] = np.asarray(
                    self.dense_array[int(row_idx), int(channel_idx)],
                    dtype=np.uint8,
                )
        return (output > 0).astype(np.uint8, copy=False)

    def _read_bitpacked(self, rows: np.ndarray, channels: np.ndarray) -> np.ndarray:
        if self.bitpacked_array is None:
            raise MaskStoreError(f"{self.source_path}/mask_bitpacked has no masks_packed array.")
        row_slice = _contiguous_slice(rows)
        channel_slice = _contiguous_slice(channels)
        if row_slice is not None and channel_slice is not None:
            packed = np.asarray(self.bitpacked_array[row_slice, channel_slice, :, :], dtype=np.uint8)
            return unpack_binary_mask_stack(packed, logical_width=int(self.shape[3]))
        output = np.zeros((int(rows.size), int(channels.size), *self.shape_hw), dtype=np.uint8)
        for out_row, row_idx in enumerate(rows):
            for out_channel, channel_idx in enumerate(channels):
                packed = np.asarray(
                    self.bitpacked_array[int(row_idx) : int(row_idx) + 1, int(channel_idx) : int(channel_idx) + 1],
                    dtype=np.uint8,
                )
                output[out_row, out_channel] = unpack_binary_mask_stack(
                    packed,
                    logical_width=int(self.shape[3]),
                )[0, 0]
        return output

    def _read_component_rle(self, rows: np.ndarray, channels: np.ndarray) -> np.ndarray:
        groups = self.component_groups or {}
        output = np.zeros((int(rows.size), int(channels.size), *self.shape_hw), dtype=np.uint8)
        for out_channel, channel_idx in enumerate(channels):
            component = groups.get(int(channel_idx))
            if component is None:
                raise MaskStoreError(
                    f"{self.source_path}/mask_rle is missing component index {int(channel_idx)} "
                    f"({self.mask_labels[int(channel_idx)]!r})."
                )
            counts = component["counts"]
            indptr = component["indptr"]
            for out_row, row_idx in enumerate(rows):
                start = int(indptr[int(row_idx)])
                stop = int(indptr[int(row_idx) + 1])
                output[out_row, out_channel] = decode_binary_mask_rle(counts[start:stop], self.shape_hw)
        return output


def _component_groups_by_index(rle_group: Any) -> dict[int, Any]:
    components_parent = zarr_child_group(rle_group, "components")
    if components_parent is None:
        return {}
    groups: dict[int, Any] = {}
    for group_name in zarr_group_keys(components_parent):
        group = zarr_child_group(components_parent, group_name)
        if group is None:
            continue
        attrs = zarr_attrs_dict(group)
        raw_index = attrs.get("component_index")
        if raw_index is None:
            prefix = str(group_name).split("_", 1)[0]
            raw_index = prefix if prefix.isdigit() else None
        if raw_index is None:
            continue
        groups[int(raw_index)] = group
    return groups


def _component_rle_shape(run_group: Any, labels: tuple[str, ...], rle_group: Any) -> tuple[int, int, int, int]:
    attrs = zarr_attrs_dict(rle_group)
    raw_shape = attrs.get("encoded_shape_hw") or attrs.get("shape_hw") or attrs.get("mask_shape")
    if not isinstance(raw_shape, (list, tuple)) or len(raw_shape) != 2:
        raise MaskStoreError("Component RLE group is missing encoded_shape_hw attr.")
    component_groups = _component_groups_by_index(rle_group)
    if not component_groups:
        raise MaskStoreError("Component RLE group has no component payload groups.")
    first = component_groups[min(component_groups)]
    if "indptr" not in first:
        raise MaskStoreError("Component RLE group is missing indptr array.")
    n_rows = int(first["indptr"].shape[0]) - 1
    return (n_rows, len(labels), int(raw_shape[0]), int(raw_shape[1]))


def _bitpacked_shape(labels: tuple[str, ...], bitpacked_group: Any) -> tuple[int, int, int, int]:
    attrs = zarr_attrs_dict(bitpacked_group)
    raw_shape = attrs.get("logical_shape")
    if not isinstance(raw_shape, (list, tuple)) or len(raw_shape) != 4:
        raise MaskStoreError("Bitpacked mask group is missing logical_shape attr.")
    shape = tuple(int(value) for value in raw_shape)
    if len(labels) != int(shape[1]):
        raise MaskStoreError(
            f"Bitpacked mask component count {shape[1]} does not match labels {len(labels)}."
        )
    packed = bitpacked_group.get("masks_packed") if hasattr(bitpacked_group, "get") else None
    if packed is None or not hasattr(packed, "shape"):
        raise MaskStoreError("Bitpacked mask group is missing masks_packed array.")
    packed_shape = tuple(int(value) for value in packed.shape)
    expected_packed = (int(shape[0]), int(shape[1]), int(shape[2]), packed_width_bytes(int(shape[3])))
    if packed_shape != expected_packed:
        raise MaskStoreError(
            f"Bitpacked masks_packed shape {packed_shape!r} does not match logical shape {shape!r}."
        )
    return (int(shape[0]), int(shape[1]), int(shape[2]), int(shape[3]))


def open_mask_store(
    run_group: Any,
    *,
    source_path: str = "",
    prefer: str = "dense",
    allow_stale_rle: bool = False,
) -> MaskStore:
    """Open a mask store from a run group.

    ``prefer='dense'`` preserves current behavior when both dense and compact
    surfaces are present. Use ``prefer='bitpacked'`` or ``prefer='rle'`` to
    exercise compact reads. Stale compact RLE is rejected unless
    ``allow_stale_rle`` is explicit.
    """

    dense = run_group.get("masks_roi") if hasattr(run_group, "get") else None
    bitpacked_group = zarr_child_group(run_group, "mask_bitpacked")
    rle_group = zarr_child_group(run_group, "mask_rle")
    dense_available = dense is not None and hasattr(dense, "shape")
    bitpacked_array = bitpacked_group.get("masks_packed") if bitpacked_group is not None and hasattr(bitpacked_group, "get") else None
    bitpacked_available = bitpacked_array is not None and hasattr(bitpacked_array, "shape")
    rle_available = rle_group is not None

    if prefer not in {"dense", "bitpacked", "rle", "auto"}:
        raise ValueError("prefer must be one of: dense, bitpacked, rle, auto.")

    def open_dense() -> MaskStore:
        dense_shape = tuple(int(value) for value in dense.shape)
        if len(dense_shape) == 3:
            shape = (int(dense_shape[0]), 1, int(dense_shape[1]), int(dense_shape[2]))
        elif len(dense_shape) == 4:
            shape = dense_shape
        else:
            raise MaskStoreError(f"Unsupported dense mask shape {dense_shape!r}.")
        labels = _mask_labels(run_group, int(shape[1]))
        return MaskStore(
            group=run_group,
            encoding="dense_uint8",
            mask_labels=labels or tuple(f"component_{idx}" for idx in range(shape[1])),
            shape=(int(shape[0]), int(shape[1]), int(shape[2]), int(shape[3])),
            source_path=source_path,
            dense_array=dense,
        )

    def open_bitpacked() -> MaskStore:
        if bitpacked_group is None or bitpacked_array is None:
            raise MaskStoreError(f"{source_path or '<mask_run>'} has no bitpacked mask store.")
        labels = _mask_labels(run_group)
        if not labels:
            attrs = zarr_attrs_dict(bitpacked_group)
            raw_labels = attrs.get("component_names")
            if isinstance(raw_labels, (list, tuple)):
                labels = tuple(str(value) for value in raw_labels)
        if not labels:
            raise MaskStoreError("Cannot open bitpacked mask store without mask_labels/component_names.")
        shape = _bitpacked_shape(labels, bitpacked_group)
        return MaskStore(
            group=run_group,
            encoding=MASK_BITPACKED_ENCODING,
            mask_labels=labels,
            shape=shape,
            source_path=source_path,
            bitpacked_group=bitpacked_group,
            bitpacked_array=bitpacked_array,
        )

    def open_rle() -> MaskStore:
        if is_mask_rle_stale(run_group) and not allow_stale_rle:
            raise MaskStoreError(
                f"{source_path or '<mask_run>'}/mask_rle is marked stale relative to dense masks_roi. "
                "Refresh it with materialize_refined_subject_mask_store --refresh-rle --apply, "
                "prefer dense masks, or pass allow_stale_rle=True for explicit diagnostics."
            )
        labels = _mask_labels(run_group)
        if not labels:
            attrs = zarr_attrs_dict(rle_group)
            raw_labels = attrs.get("component_names")
            if isinstance(raw_labels, (list, tuple)):
                labels = tuple(str(value) for value in raw_labels)
        if not labels:
            raise MaskStoreError("Cannot open component RLE mask store without mask_labels/component_names.")
        shape = _component_rle_shape(run_group, labels, rle_group)
        return MaskStore(
            group=run_group,
            encoding="component_rle_v1",
            mask_labels=labels,
            shape=shape,
            source_path=source_path,
            rle_group=rle_group,
            component_groups=_component_groups_by_index(rle_group),
        )

    preference_order: tuple[str, ...]
    if prefer == "dense":
        preference_order = ("dense", "bitpacked", "rle")
    elif prefer == "bitpacked":
        preference_order = ("bitpacked", "dense", "rle")
    elif prefer == "rle":
        preference_order = ("rle", "dense", "bitpacked")
    else:
        preference_order = ("dense", "bitpacked", "rle")

    errors: list[str] = []
    for candidate in preference_order:
        try:
            if candidate == "dense" and dense_available:
                return open_dense()
            if candidate == "bitpacked" and bitpacked_available:
                return open_bitpacked()
            if candidate == "rle" and rle_available:
                return open_rle()
        except MaskStoreError as exc:
            if candidate == prefer:
                raise
            errors.append(str(exc))
            continue
    detail = f" Details: {'; '.join(errors)}" if errors else ""
    raise MaskStoreError(f"{source_path or '<mask_run>'} has no readable masks_roi, mask_bitpacked, or mask_rle.{detail}")


__all__ = [
    "MaskStore",
    "MaskStoreError",
    "clear_mask_rle_stale_attrs",
    "is_mask_rle_stale",
    "mark_mask_rle_stale_attrs",
    "mask_store_encodings",
    "materialize_dense_masks_roi_from_store",
    "open_mask_store",
    "refresh_component_rle_mask_store_from_dense",
    "refresh_bitpacked_mask_store_from_dense",
    "update_mask_storage_attrs",
    "validate_bitpacked_mask_store_against_dense",
    "validate_bitpacked_mask_store_invariants",
    "validate_component_rle_mask_store_against_dense",
    "write_bitpacked_mask_store_from_dense",
    "write_component_rle_mask_store_from_dense",
    "write_encoded_component_rle_mask_store",
]
