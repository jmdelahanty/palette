"""Lossless HDF5 logical types and bounded row/payload iteration for this profile."""

from __future__ import annotations

import math
from typing import Iterator

import h5py
import numpy as np

from .common import (
    BLOCK_BYTES,
    MAX_DATASET_BYTES,
    MAX_DEPTH,
    MAX_JSON_BYTES,
    MAX_ROWS,
    exact_keys,
    require,
)


def type_descriptor(type_id, *, nested=False, _depth=0) -> dict:
    require(_depth <= MAX_DEPTH, "native_type_descriptor_depth")
    kind, size = type_id.get_class(), type_id.get_size()
    require(size > 0, "invalid_dataset_type")
    if kind == h5py.h5t.INTEGER:
        require(
            (size == 1 or type_id.get_order() == h5py.h5t.ORDER_LE)
            and size in (1, 2, 4, 8),
            "unsupported_integer_type",
        )
        require(
            type_id.get_precision() == size * 8 and type_id.get_offset() == 0,
            "nonstandard_integer_precision",
        )
        return {
            "class": "integer",
            "size_bytes": size,
            "byte_order": "none" if size == 1 else "little_endian",
            "signed": type_id.get_sign() == h5py.h5t.SGN_2,
        }
    if kind == h5py.h5t.FLOAT:
        require(
            type_id.equal(h5py.h5t.IEEE_F32LE) or type_id.equal(h5py.h5t.IEEE_F64LE),
            "unsupported_float_type",
        )
        return {
            "class": "ieee_float",
            "size_bytes": size,
            "byte_order": "little_endian",
        }
    if kind == h5py.h5t.STRING:
        variable = bool(type_id.is_variable_str())
        require(not (nested and variable), "nested_variable_string_unsupported")
        require(
            type_id.get_cset() in (h5py.h5t.CSET_UTF8, h5py.h5t.CSET_ASCII),
            "unsupported_string_charset",
        )
        result = {
            "class": "string",
            "variable_length": variable,
            "character_set": (
                "utf8" if type_id.get_cset() == h5py.h5t.CSET_UTF8 else "ascii"
            ),
        }
        if not variable:
            padding = {
                h5py.h5t.STR_NULLTERM: "null_terminated",
                h5py.h5t.STR_NULLPAD: "null_padded",
                h5py.h5t.STR_SPACEPAD: "space_padded",
            }
            require(type_id.get_strpad() in padding, "unsupported_string_padding")
            result.update(size_bytes=size, padding=padding[type_id.get_strpad()])
        return result
    if kind == h5py.h5t.ARRAY:
        dimensions = list(type_id.get_array_dims())
        require(0 < len(dimensions) <= 8, "invalid_array_type_rank")
        return {
            "class": "array",
            "dimensions": dimensions,
            "base": type_descriptor(
                type_id.get_super(), nested=True, _depth=_depth + 1
            ),
            "size_bytes": size,
        }
    if kind == h5py.h5t.COMPOUND:
        require(type_id.get_nmembers() <= 4096, "native_compound_member_budget")
        members, end = [], 0
        for index in range(type_id.get_nmembers()):
            member = type_id.get_member_type(index)
            offset = type_id.get_member_offset(index)
            name = type_id.get_member_name(index).decode("utf-8")
            require(
                name and name not in (".", "..") and "/" not in name,
                "invalid_compound_member_name",
            )
            require(offset == end, "compound_padding_unsupported")
            members.append(
                {
                    "name": name,
                    "offset": offset,
                    "type": type_descriptor(member, nested=True, _depth=_depth + 1),
                }
            )
            end += member.get_size()
        require(members and end == size, "compound_tail_padding_unsupported")
        return {"class": "packed_compound", "size_bytes": size, "members": members}
    raise ValueError(f"unsupported_hdf5_type:{kind}")


def dtype_from_descriptor(spec: dict, *, _depth=0) -> np.dtype:
    require(type(spec) is dict and _depth <= MAX_DEPTH, "native_type_descriptor_depth")
    kind = spec["class"]
    if kind == "integer":
        exact_keys(
            spec, ("class", "size_bytes", "byte_order", "signed"), "integer_type"
        )
        require(
            type(spec["size_bytes"]) is int
            and spec["size_bytes"] in (1, 2, 4, 8)
            and type(spec["signed"]) is bool
            and spec["byte_order"]
            == ("none" if spec["size_bytes"] == 1 else "little_endian"),
            "native_integer_type",
        )
        return np.dtype(("<i" if spec["signed"] else "<u") + str(spec["size_bytes"]))
    if kind == "ieee_float":
        exact_keys(spec, ("class", "size_bytes", "byte_order"), "float_type")
        require(
            type(spec["size_bytes"]) is int
            and spec["size_bytes"] in (4, 8)
            and spec["byte_order"] == "little_endian",
            "native_float_type",
        )
        return np.dtype("<f" + str(spec["size_bytes"]))
    if kind == "string":
        exact_keys(
            spec,
            {"class", "variable_length", "character_set"}
            | (set() if spec["variable_length"] else {"size_bytes", "padding"}),
            "string_type",
        )
        require(
            type(spec["variable_length"]) is bool
            and spec["character_set"] in ("ascii", "utf8"),
            "native_string_type",
        )
        if not spec["variable_length"]:
            require(
                type(spec["size_bytes"]) is int
                and 0 < spec["size_bytes"] <= MAX_DATASET_BYTES
                and spec["padding"]
                in ("null_terminated", "null_padded", "space_padded"),
                "native_fixed_string_type",
            )
        else:
            require(_depth == 0, "native_nested_variable_string")
        return h5py.string_dtype(
            "utf-8" if spec["character_set"] == "utf8" else "ascii",
            length=None if spec["variable_length"] else spec["size_bytes"],
        )
    if kind == "array":
        exact_keys(spec, ("class", "dimensions", "base", "size_bytes"), "array_type")
        dimensions = spec["dimensions"]
        require(
            type(dimensions) is list
            and 0 < len(dimensions) <= 8
            and all(type(value) is int and value > 0 for value in dimensions),
            "native_array_dimensions",
        )
        base = dtype_from_descriptor(spec["base"], _depth=_depth + 1)
        require(
            math.prod(dimensions) * base.itemsize
            == spec["size_bytes"]
            <= MAX_DATASET_BYTES,
            "native_array_type_budget",
        )
        return np.dtype((base, tuple(dimensions)))
    if kind == "packed_compound":
        exact_keys(spec, ("class", "size_bytes", "members"), "compound_type")
        require(
            type(spec["members"]) is list and 0 < len(spec["members"]) <= 4096,
            "native_compound_member_budget",
        )
        end, names, formats = 0, set(), []
        for member in spec["members"]:
            exact_keys(member, ("name", "offset", "type"), "compound_member")
            name = member["name"]
            require(
                isinstance(name, str)
                and name not in ("", ".", "..")
                and "/" not in name
                and name not in names
                and member["offset"] == end,
                "native_compound_layout",
            )
            names.add(name)
            formats.append(dtype_from_descriptor(member["type"], _depth=_depth + 1))
            end += formats[-1].itemsize
            require(end <= MAX_DATASET_BYTES, "native_compound_byte_budget")
        require(end == spec["size_bytes"], "native_compound_itemsize")
        return np.dtype(
            {
                "names": [m["name"] for m in spec["members"]],
                "formats": formats,
                "offsets": [m["offset"] for m in spec["members"]],
                "itemsize": spec["size_bytes"],
            }
        )
    raise ValueError(f"unsupported_logical_type:{kind}")


def check_dataset_budget(dataset: h5py.Dataset) -> dict:
    require(
        dataset.shape is not None and dataset.ndim <= 8,
        f"invalid_dataset_space:{dataset.name}",
    )
    require(
        not dataset.is_virtual and not dataset.external,
        f"dataset_storage_not_internal:{dataset.name}",
    )
    spec = type_descriptor(dataset.id.get_type())
    count = math.prod(dataset.shape)
    require(count <= MAX_ROWS, f"dataset_row_budget_exceeded:{dataset.name}")
    if not (spec["class"] == "string" and spec["variable_length"]):
        require(
            dataset.dtype.itemsize <= MAX_DATASET_BYTES
            and count * dataset.dtype.itemsize <= MAX_DATASET_BYTES,
            f"dataset_byte_budget_exceeded:{dataset.name}",
        )
    return spec


def read_bounded_string(
    dataset: h5py.Dataset, selection=(), *, limit=MAX_JSON_BYTES
) -> bytes:
    """Use fixed-width HDF5 destinations, rejecting rather than accepting truncation.

    Reading a VL string into an object dtype allocates its declared size before
    Python can check it. Fixed-width conversion bounds the destination buffer;
    a full buffer is retried up to limit+1, never treated as complete evidence.
    HDF5's own internal conversion/chunk caches remain library-managed.
    """
    encoding = h5py.check_string_dtype(dataset.dtype).encoding
    width = min(16 * 1024, limit + 1)
    while True:
        value = dataset.astype(h5py.string_dtype(encoding, length=width))[selection]
        raw = string_bytes(value)
        require(len(raw) <= limit, f"vlen_byte_budget_exceeded:{dataset.name}")
        if len(raw) < width:
            return raw
        width = min(width * 2, limit + 1)


def iter_blocks(
    dataset: h5py.Dataset, *, block_bytes: int = BLOCK_BYTES
) -> Iterator[tuple[int, np.ndarray]]:
    """Iterate first-axis hyperslabs; no unbounded numeric dataset[()] reads."""
    require(type(block_bytes) is int and block_bytes > 0, "invalid_block_budget")
    if dataset.dtype.hasobject:
        require(
            dataset.dtype.kind == "O"
            and h5py.check_string_dtype(dataset.dtype) is not None,
            "nested_variable_string_unsupported",
        )
        total = 0
        if not dataset.shape:
            yield 0, np.asarray(read_bounded_string(dataset), dtype=object)
            return
        for start in range(dataset.shape[0]):
            block = np.empty((1, *dataset.shape[1:]), dtype=object)
            for tail in np.ndindex(dataset.shape[1:]):
                raw = read_bounded_string(
                    dataset, (start, *tail), limit=MAX_JSON_BYTES - total
                )
                total += len(raw)
                block[(0, *tail)] = raw
            yield start, block
        return
    if not dataset.shape:
        yield 0, np.asarray(dataset[()])
        return
    row_bytes = dataset.dtype.itemsize * math.prod(dataset.shape[1:])
    rows = max(1, min(4096, block_bytes // max(1, row_bytes)))
    for start in range(0, dataset.shape[0], rows):
        yield start, dataset[start : min(start + rows, dataset.shape[0])]


def string_bytes(value) -> bytes:
    return value.encode("utf-8") if isinstance(value, str) else bytes(value)


def iter_payload(dataset: h5py.Dataset) -> Iterator[bytes]:
    spec = check_dataset_budget(dataset)
    variable = spec["class"] == "string" and spec["variable_length"]
    for _, block in iter_blocks(dataset):
        if variable:
            for value in block.reshape(-1):
                data = string_bytes(value)
                yield len(data).to_bytes(8, "little")
                yield data
        else:
            # Scalar fixed strings need their declared trailing bytes, not the
            # shorter np.bytes_ scalar returned by NumPy's string conversion.
            yield np.asarray(block, dtype=dataset.dtype).tobytes(order="C")


def dataset_bytes(dataset: h5py.Dataset) -> bytes:
    """Read one bounded scalar UTF-8 JSON/hash payload, without length framing."""
    spec = check_dataset_budget(dataset)
    if dataset.ndim == 1 and dataset.dtype == np.dtype("u1"):
        require(dataset.size <= MAX_JSON_BYTES, f"json_budget_exceeded:{dataset.name}")
        return b"".join(block.tobytes() for _, block in iter_blocks(dataset))
    require(
        dataset.shape == () and spec["class"] == "string",
        f"scalar_text_required:{dataset.name}",
    )
    data = (
        read_bounded_string(dataset)
        if spec["variable_length"]
        else string_bytes(dataset[()])
    )
    require(len(data) <= MAX_JSON_BYTES, f"json_budget_exceeded:{dataset.name}")
    return data
