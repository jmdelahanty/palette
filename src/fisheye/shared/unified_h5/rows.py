"""Exact native keys and enum references, with bounded disk-backed indexes."""

from __future__ import annotations

import numpy as np

from .common import KeyIndex, require, text
from .hdf5_types import iter_blocks
from .schema import table_schema


def row_key(row, fields):
    return tuple(
        (
            text(row[name], name, empty=True)
            if row.dtype[name].kind == "S"
            else int(row[name])
        )
        for name in fields
    )


def index_table(dataset, index: KeyIndex, namespace: str, *, reason: str):
    fields = text(dataset.attrs["key_fields"], "key_fields", empty=True).split(",")
    if fields == [""] or fields in (["event_row_index"], ["trial_row_index"]):
        return
    for start, block in iter_blocks(dataset):
        for offset, row in enumerate(block):
            index.add(namespace, row_key(row, fields), start + offset, reason=reason)


def validate_table_relations(h5, descriptors):
    """Validate present canonical tables, without requiring absent components."""
    with KeyIndex() as index:
        for path in descriptors:
            index_table(h5[path], index, path, reason=f"canonical_key_duplicate:{path}")

        def fields_valid(values, fields, path):
            for field in fields:
                name = field["name"]
                if field["dtype"] == "compound":
                    fields_valid(values[name], field["fields"], path + ":" + name)
                meaning = field["meaning"]
                if meaning.startswith("enum:"):
                    enum_path = meaning[5:]
                    require(enum_path in descriptors, f"enum_table_missing:{enum_path}")
                    for value in np.unique(values[name]):
                        require(
                            index.lookup(enum_path, (int(value),)) is not None,
                            f"unknown_enum_value:{path}:{name}:{int(value)}",
                        )

        for path in descriptors:
            for _, block in iter_blocks(h5[path]):
                fields_valid(block, table_schema(path)["fields"], path)
                if (
                    path.startswith("/components/")
                    and "stimulus_frame_num" in block.dtype.names
                ):
                    for value in np.unique(block["stimulus_frame_num"]):
                        require(
                            index.lookup("/frames/stimulus", (int(value),)) is not None,
                            f"component_frame_unresolved:{path}:{int(value)}",
                        )
