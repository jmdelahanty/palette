"""Typed native attribute snapshots; no lossy JSON float/string coercion."""

import math

import h5py
import numpy as np

from .common import MAX_JSON_BYTES, MAX_ROWS, UnifiedH5ContractError, require
from .hdf5_types import type_descriptor


def _variable_attribute(attr):
    count = math.prod(attr.shape)
    require(count <= MAX_ROWS, "attribute_element_budget")
    if count == 0:
        return []
    max_width = MAX_JSON_BYTES // count
    require(max_width > 0, "attribute_byte_budget")
    width = min(16 * 1024, max_width)
    while True:
        memory_type = h5py.h5t.C_S1.copy()
        memory_type.set_size(width)
        memory_type.set_cset(attr.get_type().get_cset())
        memory_type.set_strpad(h5py.h5t.STR_NULLPAD)
        buffer = np.zeros(attr.shape, dtype=f"S{width}")
        attr.read(buffer, mtype=memory_type)
        values = [bytes(value) for value in buffer.reshape(-1)]
        if all(len(value) < width for value in values):
            return [value.hex() for value in values]
        require(width < max_width, "attribute_variable_byte_budget_or_truncation")
        width = min(2 * width, max_width)


def native_attributes(node):
    values = {}
    total = 0
    for name in node.attrs:
        attr = node.attrs.get_id(name)
        require(attr.shape is not None and len(attr.shape) <= 8, "attribute_rank")
        spec = type_descriptor(attr.get_type())
        descriptor = {"shape": list(attr.shape), "type": spec}
        if spec["class"] == "string" and spec["variable_length"]:
            descriptor["values_hex"] = _variable_attribute(attr)
            size = sum(len(value) // 2 for value in descriptor["values_hex"])
        else:
            size = math.prod(attr.shape) * attr.dtype.itemsize
            require(size <= MAX_JSON_BYTES, "attribute_byte_budget")
            buffer = np.empty(attr.shape, dtype=attr.dtype)
            attr.read(buffer)
            descriptor["payload_hex"] = buffer.tobytes(order="C").hex()
        total += size
        require(total <= MAX_JSON_BYTES, "node_attribute_byte_budget")
        values[name] = descriptor
    return values


def string_attributes(descriptors):
    """Decode copied scalar string attribute descriptors to text.

    Citrus writes metadata such as ``/metadata/subject`` as scalar string
    attributes. Anything else (arrays, numbers, undecodable bytes) is refused
    rather than coerced.
    """

    decoded = {}
    for name, descriptor in descriptors.items():
        spec = descriptor.get("type", {})
        require(
            spec.get("class") == "string" and descriptor.get("shape") == [],
            f"attribute_not_scalar_string:{name}",
        )
        if spec.get("variable_length"):
            values = descriptor.get("values_hex")
            require(
                isinstance(values, list) and len(values) == 1,
                f"attribute_value_missing:{name}",
            )
            raw = bytes.fromhex(values[0])
        else:
            raw = bytes.fromhex(descriptor["payload_hex"])
            if spec.get("padding") == "space_padded":
                raw = raw.rstrip(b" ")
            else:
                raw = raw.split(b"\0", 1)[0]
        encoding = "utf-8" if spec.get("character_set") == "utf8" else "ascii"
        try:
            decoded[name] = raw.decode(encoding)
        except UnicodeDecodeError as exc:
            raise UnifiedH5ContractError(f"attribute_not_text:{name}") from exc
    return decoded
