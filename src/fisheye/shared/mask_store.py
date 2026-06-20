"""Dense materialization boundary for Palette mask storage encodings."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .mask_rle import (
    MASK_RLE_ENCODING,
    MASK_RLE_SCHEMA_ID,
    MASK_RLE_VALUE_SEMANTICS,
    EncodedMaskComponentStack,
    concatenate_encoded_mask_component_stacks,
    decode_binary_mask_rle,
    encode_mask_component_stack_rle,
)
from .zarr_helpers import zarr_attrs_dict, zarr_child_group, zarr_group_keys


class MaskStoreError(ValueError):
    """Raised when a mask store cannot resolve or materialize masks."""


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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _attr_truthy(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def mask_store_encodings(*, has_dense: bool, has_rle: bool) -> list[str]:
    encodings: list[str] = []
    if has_dense:
        encodings.append("dense_uint8")
    if has_rle:
        encodings.append("component_rle_v1")
    return encodings


def update_mask_storage_attrs(run_group: Any, *, has_dense: bool, has_rle: bool) -> None:
    encodings = mask_store_encodings(has_dense=has_dense, has_rle=has_rle)
    run_group.attrs["mask_store_encodings"] = encodings
    run_group.attrs["mask_storage_encoding"] = "+".join(encodings) if encodings else "missing"
    run_group.attrs["masks_roi_materialized"] = bool(has_dense)
    if has_rle:
        run_group.attrs["mask_rle_materialized"] = True


def materialize_dense_masks_roi_from_store(
    run_group: Any,
    *,
    chunk_size: int = 256,
    overwrite: bool = False,
    source_path: str = "",
) -> dict[str, object]:
    """Create or refresh the dense ``masks_roi`` compatibility cache from RLE."""

    has_dense = "masks_roi" in run_group
    has_rle = "mask_rle" in run_group
    if has_dense and not overwrite:
        update_mask_storage_attrs(run_group, has_dense=True, has_rle=has_rle)
        return {"status": "existing", "encoding": "dense_uint8", "has_dense_after": True}
    if not has_rle:
        raise MaskStoreError("Dense masks_roi cannot be materialized because compact mask_rle storage is unavailable.")

    mask_store = open_mask_store(run_group, source_path=source_path, prefer="rle")
    if mask_store.encoding != "component_rle_v1":
        raise MaskStoreError("Dense masks_roi cannot be materialized because compact mask_rle storage is unavailable.")

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
    run_group.attrs["masks_roi_materialized_from"] = "mask_rle"
    run_group.attrs["masks_roi_materialized_at_utc"] = _utc_now()
    run_group.attrs["masks_roi_materialization_chunk_size"] = int(row_chunk)
    update_mask_storage_attrs(run_group, has_dense=True, has_rle=True)
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

    row_chunk = max(1, int(row_chunk_size))
    count_chunk_values = max(1, int(count_chunk_bytes) // np.dtype(np.uint32).itemsize)
    components_group = rle_group.require_group("components")
    for component in encoded.components:
        component_group = components_group.require_group(
            _safe_component_group_name(component.component_name, component.component_index)
        )
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

    update_mask_storage_attrs(run_group, has_dense="masks_roi" in run_group, has_rle=True)
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


def write_component_rle_mask_store_from_dense(
    run_group: Any,
    dense_masks: Any,
    *,
    component_names: Sequence[str] | None = None,
    overwrite: bool = True,
    encode_row_chunk_size: int = 256,
    validate_roundtrip: bool = True,
    count_chunk_bytes: int = 4 * 1024 * 1024,
    extra_attrs: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    """Stream dense masks into component-separated RLE storage.

    The dense source is read in row chunks so full ``(N,C,H,W)`` masks do not need
    to fit in host memory at once.
    """

    shape = tuple(int(v) for v in dense_masks.shape)
    if len(shape) == 3:
        n_rows, height, width = shape
        n_channels = 1
        squeeze_channel = True
    elif len(shape) == 4:
        n_rows, n_channels, height, width = shape
        squeeze_channel = False
    else:
        raise MaskStoreError(f"Expected dense masks with shape (N,C,H,W) or (N,H,W), got {shape!r}.")

    names = tuple(str(value) for value in component_names) if component_names is not None else _mask_labels(run_group, n_channels)
    if len(names) != int(n_channels):
        raise MaskStoreError(f"Expected {n_channels} component names for dense masks, got {len(names)}.")

    row_chunk = max(1, int(encode_row_chunk_size))
    shards: list[EncodedMaskComponentStack] = []
    for start in range(0, int(n_rows), row_chunk):
        stop = min(int(n_rows), start + row_chunk)
        dense_chunk = np.asarray(dense_masks[start:stop], dtype=np.uint8)
        if squeeze_channel:
            dense_chunk = dense_chunk[:, None, :, :]
        shards.append(encode_mask_component_stack_rle(dense_chunk, component_names=names))

    if shards:
        encoded = concatenate_encoded_mask_component_stacks(shards)
    else:
        encoded = encode_mask_component_stack_rle(
            np.zeros((0, int(n_channels), int(height), int(width)), dtype=np.uint8),
            component_names=names,
        )
    summary = write_encoded_component_rle_mask_store(
        run_group,
        encoded,
        overwrite=overwrite,
        count_chunk_bytes=count_chunk_bytes,
        row_chunk_size=row_chunk,
        extra_attrs=extra_attrs,
    )
    summary.update(
        {
            "source_encoding": "dense_uint8",
            "encode_row_chunk_size": int(row_chunk),
            "roundtrip_validation_requested": bool(validate_roundtrip),
        }
    )
    if validate_roundtrip:
        validation = validate_component_rle_mask_store_against_dense(
            run_group,
            dense_masks,
            row_chunk_size=row_chunk,
            source_path=str(extra_attrs.get("source_path", "")) if extra_attrs else "",
        )
        summary["roundtrip_validation"] = validation
    return summary


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
    encode_row_chunk_size: int = 256,
    count_chunk_bytes: int = 4 * 1024 * 1024,
    source_path: str = "",
    clear_stale: bool = True,
) -> dict[str, object]:
    """Regenerate compact ``mask_rle`` from the current dense ``masks_roi``."""

    dense = run_group.get("masks_roi") if hasattr(run_group, "get") else None
    if dense is None or not hasattr(dense, "shape"):
        raise MaskStoreError("Cannot refresh compact mask_rle because dense masks_roi is unavailable.")
    dense_shape = tuple(int(value) for value in dense.shape)
    if len(dense_shape) not in {3, 4}:
        raise MaskStoreError(f"Cannot refresh compact mask_rle from unsupported dense mask shape {dense_shape!r}.")

    summary = write_component_rle_mask_store_from_dense(
        run_group,
        dense,
        component_names=component_names,
        overwrite=True,
        count_chunk_bytes=count_chunk_bytes,
        encode_row_chunk_size=max(1, int(encode_row_chunk_size)),
        extra_attrs={
            "source_path": str(source_path),
            "refreshed_from": "masks_roi",
            "refreshed_at_utc": _utc_now(),
        },
    )
    update_mask_storage_attrs(run_group, has_dense=True, has_rle=True)
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
            "rows": int(dense_shape[0]),
            "channels": int(dense_shape[1]) if len(dense_shape) == 4 else 1,
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


def open_mask_store(
    run_group: Any,
    *,
    source_path: str = "",
    prefer: str = "dense",
    allow_stale_rle: bool = False,
) -> MaskStore:
    """Open a mask store from a run group.

    ``prefer='dense'`` preserves current behavior when both dense and compact
    surfaces are present. Use ``prefer='rle'`` to exercise compact reads. Stale
    compact RLE is rejected unless ``allow_stale_rle`` is explicit.
    """

    dense = run_group.get("masks_roi") if hasattr(run_group, "get") else None
    rle_group = zarr_child_group(run_group, "mask_rle")
    dense_available = dense is not None and hasattr(dense, "shape")
    rle_available = rle_group is not None

    if prefer not in {"dense", "rle", "auto"}:
        raise ValueError("prefer must be one of: dense, rle, auto.")
    if prefer == "dense" and dense_available:
        labels = _mask_labels(run_group, int(dense.shape[1]) if len(tuple(dense.shape)) >= 2 else None)
        if len(tuple(dense.shape)) == 3:
            shape = (int(dense.shape[0]), 1, int(dense.shape[1]), int(dense.shape[2]))
        elif len(tuple(dense.shape)) == 4:
            shape = tuple(int(v) for v in dense.shape)
        else:
            raise MaskStoreError(f"Unsupported dense mask shape {tuple(dense.shape)!r}.")
        return MaskStore(
            group=run_group,
            encoding="dense_uint8",
            mask_labels=labels or tuple(f"component_{idx}" for idx in range(shape[1])),
            shape=(int(shape[0]), int(shape[1]), int(shape[2]), int(shape[3])),
            source_path=source_path,
            dense_array=dense,
        )

    if rle_available:
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

    if dense_available:
        return open_mask_store(run_group, source_path=source_path, prefer="dense")
    raise MaskStoreError(f"{source_path or '<mask_run>'} has neither masks_roi nor mask_rle.")


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
    "update_mask_storage_attrs",
    "validate_component_rle_mask_store_against_dense",
    "write_component_rle_mask_store_from_dense",
    "write_encoded_component_rle_mask_store",
]
