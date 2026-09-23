"""Validate the exact pinned Citrus tables; do not infer them from emitted rows."""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from hashlib import sha256
from importlib.resources import files

import h5py
import numpy as np

from .common import canonical_json, internal_path, parse_json, require, same_json, text
from .hdf5_types import check_dataset_budget, dataset_bytes, iter_blocks, iter_payload

APPEARANCE = "/components/visual_appearance/states"
APPEARANCE_MANIFEST = "/components/visual_appearance/replay_dependency_manifest_json"
CONTRACT_HASHES = {
    "experimental_h5_core_v1.json": "febb7516ff96f359b6784bf137682abe387b7922270e9637ebc58b369e07f9c1",
    "experimental_h5_geometry_v1.json": "4c0c707192cf1fa6ceafb2f632672972f41dc7e4c7cc5cdb0c0e3b7920a1474b",
    "experimental_h5_identity_claims_v1.json": "83ca66b336f3e3a40ebaafc869d7febd020509e238243381dfe871cbeb89507e",
    "object_appearance_replay_dependency_manifest_v1.json": "eccfcfcb3c6e66549b2fdfcfcf88fdc4fa9def3d6ce4a375494d011b04f2607c",
}


@lru_cache(maxsize=4)
def _contract(name: str) -> dict:
    data = files(__package__).joinpath("contracts").joinpath(name).read_bytes()
    require(
        sha256(data).hexdigest() == CONTRACT_HASHES[name],
        f"packaged_contract_drift:{name}",
    )
    return parse_json(data, label=name)


def contract(name: str) -> dict:
    return deepcopy(_contract(name))


def read_json(h5, path: str, *, canonical: bool = False) -> dict:
    require(
        path in h5 and isinstance(h5[path], h5py.Dataset),
        f"missing_json_dataset:{path}",
    )
    return parse_json(dataset_bytes(h5[path]), label=path, canonical=canonical)


def table_schema(path: str) -> dict:
    template = path
    if path.startswith("/definitions/enums/") and len(path.split("/")) == 4:
        template = "/definitions/enums/{component}"
    elif (
        path.startswith("/correspondence/")
        and path.endswith("/sources")
        and path != "/correspondence/frames/sources"
    ):
        require(len(path.split("/")) == 4, f"unknown_table:{path}")
        template = "/correspondence/{component}/sources"
    matches = [
        table
        for table in _contract("experimental_h5_core_v1.json")["tables"]
        if table["path"] == template
    ]
    require(len(matches) == 1, f"unknown_table:{path}")
    result = deepcopy(matches[0])
    result["path"] = path
    return result


def _validate_fields(datatype, fields, itemsize: int, path: str) -> None:
    require(
        datatype.get_class() == h5py.h5t.COMPOUND and datatype.get_size() == itemsize,
        f"table_itemsize_or_type:{path}",
    )
    require(datatype.get_nmembers() == len(fields), f"table_field_count:{path}")
    for index, field in enumerate(fields):
        name = field["name"]
        require(
            datatype.get_member_name(index).decode("utf-8") == name
            and datatype.get_member_offset(index) == field["offset"],
            f"table_field_layout:{path}:{name}",
        )
        member = datatype.get_member_type(index)
        if field["dtype"] == "compound":
            _validate_fields(
                member, field["fields"], field["itemsize"], f"{path}:{name}"
            )
        elif field["dtype"] == "utf8_nullterm":
            require(
                member.get_class() == h5py.h5t.STRING
                and not member.is_variable_str()
                and member.get_cset() == h5py.h5t.CSET_UTF8
                and member.get_strpad() == h5py.h5t.STR_NULLTERM
                and member.get_size() == field["count"],
                f"table_utf8_type:{path}:{name}",
            )
        else:
            expected = np.dtype(field["dtype"])
            if field["count"] != 1:
                require(
                    member.get_class() == h5py.h5t.ARRAY
                    and member.get_array_dims() == (field["count"],),
                    f"table_array_type:{path}:{name}",
                )
                member = member.get_super()
            require(
                member.equal(h5py.h5t.py_create(expected)),
                f"table_field_type:{path}:{name}",
            )


def _validate_values(values, fields, path: str) -> None:
    for field in fields:
        name = field["name"]
        if field["dtype"] == "compound":
            _validate_values(values[name], field["fields"], f"{path}:{name}")
        if field.get("boolean"):
            require(
                bool(np.all((values[name] == 0) | (values[name] == 1))),
                f"invalid_boolean:{path}:{name}",
            )
        validity = field.get("validity_field")
        if validity:
            require(
                bool(np.all(values[name][values[validity] == 0] == 0)),
                f"invalid_value_fill:{path}:{name}",
            )
        if field["dtype"] == "utf8_nullterm":
            for value in values[name].reshape(-1):
                raw = bytes(value)
                require(len(raw) < field["count"], f"unterminated_text:{path}:{name}")
                text(raw.split(b"\0", 1)[0], f"{path}:{name}", empty=True)


def appearance_dependencies(h5) -> dict:
    dataset = h5[APPEARANCE]
    require(
        dataset.attrs.get("replay_dependency_manifest_ref") == APPEARANCE_MANIFEST,
        "appearance_replay_reference_mismatch:manifest",
    )
    manifest = read_json(h5, APPEARANCE_MANIFEST, canonical=True)
    require(
        same_json(
            manifest, _contract("object_appearance_replay_dependency_manifest_v1.json")
        ),
        "appearance_replay_manifest_mismatch",
    )
    bindings = {
        "appearance_profile_index": "profile_definition_ref",
        "protocol_semantic_identity": "protocol_semantic_ref",
        "chaser_runtime_state": "stimulus_state_table_ref",
        "stimulus_frame_timeline": "frame_table_ref",
        "renderer_snapshot": "renderer_snapshot_ref",
        "presentation_timing": "presentation_timing_ref",
        "appearance_enum_definitions": "appearance_enum_group_ref",
        "build_identity": "build_identity_ref",
    }
    values = {}
    for dependency in manifest["adapters"][0]["dependencies"]:
        role, path = dependency["role"], internal_path(dependency["object_ref"])
        require(path in h5, f"appearance_replay_dependency_missing:{role}")
        if role in bindings:
            require(
                dataset.attrs.get(bindings[role]) == path,
                f"appearance_replay_reference_mismatch:{role}",
            )
        for name in dependency["required_attributes"]:
            value = text(h5[path].attrs.get(name), f"{path}@{name}")
            values.setdefault(role, {})[name] = value
    require(
        "/protocol/executed/execution_index_json" in h5,
        "appearance_execution_index_missing",
    )
    return values


def describe_table(h5, path: str) -> dict:
    schema = table_schema(path)
    require(path in h5 and isinstance(h5[path], h5py.Dataset), f"missing_table:{path}")
    dataset = h5[path]
    check_dataset_budget(dataset)
    require(dataset.ndim == schema["rank"], f"table_rank:{path}")
    _validate_fields(dataset.id.get_type(), schema["fields"], schema["itemsize"], path)
    for name, expected in (
        ("schema_id", schema["id"]),
        ("schema_version", schema["schema_version"]),
        ("key_fields", schema["key_fields"]),
    ):
        require(
            dataset.attrs.get(name) == expected,
            f"table_attribute_mismatch:{path}:{name}",
        )
    require(
        dataset.attrs.get_id("schema_version").shape == ()
        and dataset.attrs.get_id("schema_version").dtype == np.dtype("<u8"),
        f"table_version_width:{path}",
    )
    references = {}
    for name in schema["required_reference_attributes"]:
        reference = internal_path(dataset.attrs.get(name))
        require(reference in h5, f"table_reference_missing:{path}:{name}")
        references[name] = reference
    expected_refs = {
        "frame_table_ref": "/frames/stimulus",
        "frame_context_ref": "/frames/stimulus",
        "acquisition_binding_ref": "/correspondence/acquisition/binding_json",
        "protocol_snapshot_ref": "/protocol/authored",
        "enum_group_ref": "/definitions/enums",
        "clock_domain_ref": "/timing/clock_domains/citrus_process",
        "coordinate_context_ref": "/geometry/runtime/contract_json",
    }
    for name, expected in expected_refs.items():
        if name in references:
            require(
                references[name] == expected, f"table_reference_mismatch:{path}:{name}"
            )
    if "state_table_ref" in references:
        require(
            references["state_table_ref"] == f'/components/{path.split("/")[2]}/states',
            f"table_reference_mismatch:{path}:state_table_ref",
        )
    descriptor = {
        "schema_id": "citrus.experimental_h5.core_dataset_digest",
        "schema_version": 1,
        "encoding": "core_schema_json_and_packed_le_v1",
        "table": schema,
        "shape": list(dataset.shape),
        "references": references,
    }
    if path == APPEARANCE:
        descriptor["required_dependency_attribute_values"] = appearance_dependencies(h5)
    header = canonical_json(descriptor)
    hasher = sha256(
        b"citrus.experimental_h5.core_dataset_digest.v1\n"
        + len(header).to_bytes(8, "little")
        + header
    )
    for _, block in iter_blocks(dataset):
        _validate_values(block, schema["fields"], path)
        hasher.update(block.tobytes(order="C"))
    descriptor["content_sha256"] = "sha256:" + hasher.hexdigest()
    return descriptor


def describe_internal_dataset(dataset, component_id: str) -> dict:
    spec = check_dataset_budget(dataset)
    variable = spec["class"] == "string" and spec["variable_length"]
    if variable:
        logical_size = sum(
            len(value.encode("utf-8") if isinstance(value, str) else bytes(value))
            for _, block in iter_blocks(dataset)
            for value in block.reshape(-1)
        )
    else:
        logical_size = dataset.size * dataset.dtype.itemsize
    result = {
        "kind": "internal_dataset",
        "component_id": component_id,
        "path": dataset.name,
        "encoding": "hdf5_closed_logical_values_v1",
        "shape": list(dataset.shape),
        "type": spec,
        "logical_payload_size_bytes": logical_size,
    }
    header = canonical_json(result)
    hasher = sha256(
        b"citrus.experimental_h5.internal_dataset_digest.v1\n"
        + len(header).to_bytes(8, "little")
        + header
    )
    for block in iter_payload(dataset):
        hasher.update(block)
    result["content_sha256"] = "sha256:" + hasher.hexdigest()
    return result
