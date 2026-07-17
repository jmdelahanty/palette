"""Shared helpers for columnar structured data in Palette Zarr archives.

The storage contract keeps each field of a NumPy structured array in its own
Zarr array. Fixed-width byte-string fields are represented as two-dimensional
``uint8`` arrays so TensorStore and other consumers can read them reliably.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import zarr
from zarr import Array as ZarrArray, Group as ZarrGroup

from fisheye.shared.json_safety import decode_null_terminated_text

__all__ = [
    "COLUMNAR_SHARD_ALIGNMENT_POLICY",
    "COLUMNAR_SHARD_POLICY",
    "COLUMNAR_STORAGE_SCHEMA_ID",
    "DEFAULT_COLUMNAR_SHARD_ROWS",
    "load_structured_dataset",
    "pick_chunks",
    "pick_shards",
    "read_columnar_dataset",
    "store_array",
    "write_columnar_dataset",
]

COLUMNAR_STORAGE_SCHEMA_ID = "palette.columnar_zarr_storage.v1"
DEFAULT_COLUMNAR_SHARD_ROWS = 262_144
COLUMNAR_SHARD_POLICY = "multi_chunk_capped"
COLUMNAR_SHARD_ALIGNMENT_POLICY = (
    "ceil_requested_rows_to_logical_chunk_grid_capped_to_array"
)


def pick_chunks(shape: Tuple[int, ...]) -> Optional[Tuple[int, ...]]:
    """Choose the established logical chunk layout for stored arrays."""

    if len(shape) == 0:
        return None
    if shape[0] == 0:
        return (1,) + shape[1:]
    if len(shape) == 1:
        return (min(4096, shape[0]),)
    first_dim = min(1024, shape[0])
    if first_dim <= 0:
        return None
    return (first_dim,) + shape[1:]


def pick_shards(
    shape: Tuple[int, ...],
    chunks: Optional[Tuple[int, ...]],
    *,
    shard_rows: Optional[int] = DEFAULT_COLUMNAR_SHARD_ROWS,
) -> Optional[Tuple[int, ...]]:
    """Choose a row-aligned outer shard without changing logical chunks.

    Sharding is omitted for scalars, empty arrays, and arrays containing only
    one logical row chunk. Capping the shard to the array's chunk-grid extent
    prevents short two-dimensional arrays from acquiring enormous empty shard
    indexes when their first-axis logical chunk is small.
    """

    if shard_rows is None:
        return None
    requested = int(shard_rows)
    if requested <= 0:
        raise ValueError("shard_rows must be positive when provided.")
    if not shape or not chunks:
        return None
    if len(shape) != len(chunks):
        raise ValueError(
            f"shape and chunks must have the same rank; got shape={shape!r}, chunks={chunks!r}."
        )
    row_count = int(shape[0])
    inner_rows = int(chunks[0])
    if row_count <= 0:
        return None
    if inner_rows <= 0:
        raise ValueError("Logical row chunks must be positive.")

    logical_row_chunks = (row_count + inner_rows - 1) // inner_rows
    if logical_row_chunks <= 1:
        return None
    aligned_requested = ((requested + inner_rows - 1) // inner_rows) * inner_rows
    maximum_useful_rows = logical_row_chunks * inner_rows
    effective_rows = min(aligned_requested, maximum_useful_rows)
    if effective_rows <= inner_rows:
        return None
    return (int(effective_rows), *tuple(int(value) for value in chunks[1:]))


def _to_string_list(data: np.ndarray) -> List[str]:
    """Convert a string-like NumPy array to text values."""

    strings: List[str] = []
    for value in data:
        if isinstance(value, (bytes, np.bytes_, str)):
            strings.append(decode_null_terminated_text(value, errors="ignore"))
        elif value is None:
            strings.append("")
        else:
            strings.append(str(value))
    return strings


def store_array(
    parent: zarr.Group,
    name: str,
    data: np.ndarray,
    attrs: Optional[Dict[str, object]] = None,
    *,
    shard_rows: Optional[int] = DEFAULT_COLUMNAR_SHARD_ROWS,
) -> zarr.Array:
    """Store a complete NumPy array with aligned, single-writer sharding."""

    if name in parent:
        del parent[name]

    if data.dtype.kind in ("S", "O", "U"):
        values = _to_string_list(data)
        if values:
            max_len = min(max(len(str(value).encode("utf-8")) for value in values), 512)
            max_len = 2 ** (max_len - 1).bit_length()
        else:
            max_len = 128

        encoded = np.zeros((len(values), max_len), dtype=np.uint8)
        for index, value in enumerate(values):
            byte_data = str(value).encode("utf-8")[:max_len]
            encoded[index, : len(byte_data)] = np.frombuffer(byte_data, dtype=np.uint8)

        stored_data = encoded
    else:
        stored_data = data

    chunks = pick_chunks(tuple(int(value) for value in stored_data.shape))
    shards = pick_shards(
        tuple(int(value) for value in stored_data.shape),
        chunks,
        shard_rows=shard_rows,
    )
    create_kwargs: Dict[str, object] = {
        "data": stored_data,
        "overwrite": True,
    }
    if chunks is not None:
        create_kwargs["chunks"] = chunks
    if shards is not None:
        create_kwargs["shards"] = shards
    arr = parent.create_array(name, **create_kwargs)

    if attrs:
        for attr_name, attr_value in attrs.items():
            arr.attrs[attr_name] = attr_value

    if shard_rows is None:
        skip_reason = "disabled"
    elif not stored_data.shape:
        skip_reason = "scalar"
    elif int(stored_data.shape[0]) <= 0:
        skip_reason = "empty_array"
    elif shards is None:
        skip_reason = "single_logical_row_chunk"
    else:
        skip_reason = None
    arr.attrs.update(
        {
            "palette_storage_schema_id": COLUMNAR_STORAGE_SCHEMA_ID,
            "palette_storage_writer": "fisheye.shared.zarr.columnar.store_array",
            "palette_physical_layout": (
                "indexed_sharding_v1" if shards is not None else "regular_chunks_v1"
            ),
            "palette_logical_chunk_shape": (
                list(chunks) if chunks is not None else None
            ),
            "palette_shard_rows_requested": (
                int(shard_rows) if shard_rows is not None else None
            ),
            "palette_shard_rows_effective": (
                int(shards[0]) if shards is not None else None
            ),
            "palette_shard_shape": list(shards) if shards is not None else None,
            "palette_shard_alignment_policy": COLUMNAR_SHARD_ALIGNMENT_POLICY,
            "palette_sharding_skip_reason": skip_reason,
        }
    )

    return arr


def write_columnar_dataset(
    parent: zarr.Group,
    name: str,
    data: np.ndarray,
    attrs: Optional[Dict[str, object]] = None,
    *,
    shard_rows: Optional[int] = DEFAULT_COLUMNAR_SHARD_ROWS,
) -> zarr.Group:
    """Store a structured array as a group of field-aligned Zarr arrays."""

    if data.dtype.names is None:
        raise ValueError("write_columnar_dataset requires a structured dtype.")

    if name in parent:
        del parent[name]

    group = parent.create_group(name)
    field_names = list(data.dtype.names)
    group.attrs["storage_layout"] = "columnar"
    group.attrs["field_names"] = field_names
    group.attrs["field_dtypes"] = {
        field: str(data.dtype.fields[field][0]) for field in field_names
    }

    if attrs:
        for attr_name, attr_value in attrs.items():
            group.attrs[attr_name] = attr_value

    sharded_field_count = 0
    for field in field_names:
        array = store_array(
            group,
            field,
            np.asarray(data[field]),
            shard_rows=shard_rows,
        )
        if array.attrs["palette_physical_layout"] == "indexed_sharding_v1":
            sharded_field_count += 1

    group.attrs.update(
        {
            "columnar_storage_schema_id": COLUMNAR_STORAGE_SCHEMA_ID,
            "columnar_shard_rows_requested": (
                int(shard_rows) if shard_rows is not None else None
            ),
            "columnar_shard_alignment_policy": COLUMNAR_SHARD_ALIGNMENT_POLICY,
            "columnar_sharding_policy": COLUMNAR_SHARD_POLICY,
            "columnar_sharded_field_count": int(sharded_field_count),
            "columnar_regular_field_count": int(len(field_names) - sharded_field_count),
        }
    )

    return group


def read_columnar_dataset(group: zarr.Group) -> np.ndarray:
    """Load a structured NumPy array from a columnar Zarr group."""

    if not isinstance(group, ZarrGroup):
        raise TypeError("Expected a Zarr group for columnar dataset.")
    field_names = list(group.attrs.get("field_names", []))
    if not field_names:
        raise ValueError("Columnar group missing 'field_names' attribute.")

    field_dtypes = group.attrs.get("field_dtypes", {})
    arrays = []
    dtype = []
    for field in field_names:
        arr = group[field][:]
        if field in field_dtypes:
            field_dtype = np.dtype(field_dtypes[field])
            dtype.append((field, field_dtype))
            if field_dtype.kind == "S" and arr.ndim == 2 and arr.dtype == np.uint8:
                decoded = np.empty(arr.shape[0], dtype=field_dtype)
                for index in range(arr.shape[0]):
                    decoded[index] = arr[index].tobytes().rstrip(b"\x00")
                arrays.append(decoded)
            else:
                arrays.append(arr)
        else:
            dtype.append((field, arr.dtype))
            arrays.append(arr)

    structured = np.empty(len(arrays[0]), dtype=dtype)
    for field, arr in zip(field_names, arrays):
        structured[field] = arr
    return structured


def load_structured_dataset(
    parent: zarr.Group,
    name: str,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Load either a columnar group or a historical structured Zarr array."""

    node = parent.get(name)
    if node is None:
        raise KeyError(f"Dataset '{name}' not found in group '{parent.path}'.")
    if isinstance(node, ZarrArray):
        return node[:], dict(node.attrs)
    if isinstance(node, ZarrGroup):
        return read_columnar_dataset(node), dict(node.attrs)
    raise TypeError(f"Unsupported node type for '{name}': {type(node)}")
