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
    "load_structured_dataset",
    "pick_chunks",
    "read_columnar_dataset",
    "store_array",
    "write_columnar_dataset",
]


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
) -> zarr.Array:
    """Store a NumPy array in Zarr, replacing any existing node."""

    if name in parent:
        del parent[name]

    if data.dtype.names:
        arr = parent.create_array(
            name,
            data=data,
            chunks=pick_chunks(data.shape),
            overwrite=True,
        )
        if attrs:
            for attr_name, attr_value in attrs.items():
                arr.attrs[attr_name] = attr_value
        return arr

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

        arr = parent.create_array(
            name,
            data=encoded,
            chunks=pick_chunks(encoded.shape),
            overwrite=True,
        )
    else:
        arr = parent.create_array(
            name,
            data=data,
            chunks=pick_chunks(data.shape),
            overwrite=True,
        )

    if attrs:
        for attr_name, attr_value in attrs.items():
            arr.attrs[attr_name] = attr_value

    return arr


def write_columnar_dataset(
    parent: zarr.Group,
    name: str,
    data: np.ndarray,
    attrs: Optional[Dict[str, object]] = None,
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

    for field in field_names:
        store_array(group, field, np.asarray(data[field]))

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
