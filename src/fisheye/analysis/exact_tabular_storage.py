"""Byte-planned rematerialization for exact compact analysis runs.

Scientific writers may continue computing into their established node-local
layout.  This module owns the final physical boundary for an explicit storage
candidate: it replans every declared fixed-width array from its actual bytes,
creates the destination through the shared array factory, and writes complete
non-overlapping physical units.  It does not select a profile, publish a run,
move parent pointers, or grant selector eligibility.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from fisheye.analysis._exact_tabular_run_schema import EXCLUDED_REPORT_PREFIXES
from fisheye.shared.zarr.analysis_array_contracts import AnalysisArrayDeclaration
from fisheye.shared.zarr.analysis_storage_planning import (
    AnalysisArrayStorageFacts,
    AnalysisStoragePlanReceipt,
    analysis_storage_plan_receipt_from_manifest,
    plan_analysis_storage,
)
from fisheye.shared.zarr.array_factory import (
    array_metadata_declaration_from_plan,
    create_array_from_plan,
)
from fisheye.shared.zarr.storage_profiles import StorageProfile


ANALYSIS_STORAGE_PLAN_RECEIPT_ATTR = "analysis_storage_plan_receipt"
ANALYSIS_STORAGE_PLAN_DIGEST_ATTR = "analysis_storage_plan_payload_sha256"
ANALYSIS_STORAGE_PROFILE_ID_ATTR = "analysis_storage_profile_id"
ANALYSIS_STORAGE_PROFILE_ROLE_ATTR = "analysis_storage_profile_role"
ANALYSIS_STORAGE_PROFILE_ROLE = "explicit_unpromoted_candidate"
_RESERVED_ARRAY_ATTRIBUTES = frozenset(
    {
        "logical_schema_id",
        "logical_schema_version",
        "storage_policy_version",
        "storage_profile_id",
        "codec_profile_id",
        "access_pattern",
        "write_mode",
    }
)


def _array_at_path(group: Any, path: str) -> Any:
    node = group
    for component in path.split("/"):
        node = node[component]
    return node


def _parent_and_leaf(group: Any, path: str) -> tuple[Any, str]:
    components = path.split("/")
    parent = group
    for component in components[:-1]:
        child = parent.get(component)
        if child is None:
            child = parent.create_group(component)
        parent = child
    return parent, components[-1]


def _iter_arrays(group: Any, prefix: str = ""):
    for name, array in sorted(group.arrays(), key=lambda item: item[0]):
        yield f"{prefix}/{name}" if prefix else str(name), array
    for name, child in sorted(group.groups(), key=lambda item: item[0]):
        child_prefix = f"{prefix}/{name}" if prefix else str(name)
        yield from _iter_arrays(child, child_prefix)


def _growth_axis(declaration: AnalysisArrayDeclaration) -> int | None:
    axes = declaration.contract.axis_names
    if not axes:
        return None
    # Frame-major matrices such as detector_signal[signal, frame] must grow and
    # shard along time.  All ordinary compact tables grow along their row axis.
    if "frame" in axes:
        return axes.index("frame")
    return 0


def _access_unit_semantics(declaration: AnalysisArrayDeclaration) -> str:
    growth_axis = _growth_axis(declaration)
    if growth_axis is None:
        return "one complete scalar value"
    axis_name = declaration.contract.axis_names[growth_axis]
    trailing = tuple(
        name
        for index, name in enumerate(declaration.contract.axis_names)
        if index != growth_axis
    )
    return (
        f"one complete {axis_name} unit; all non-growth semantic axes "
        f"remain indivisible: {trailing!r}"
    )


def build_exact_tabular_storage_receipt(
    run_group: Any,
    *,
    declarations: Sequence[AnalysisArrayDeclaration],
    profile: StorageProfile,
) -> AnalysisStoragePlanReceipt:
    """Replan the exact observed scientific arrays from uncompressed bytes."""

    if not isinstance(profile, StorageProfile):
        raise TypeError("profile must be an explicitly supplied StorageProfile.")
    declaration_by_path = {
        declaration.path: declaration for declaration in declarations
    }
    if len(declaration_by_path) != len(declarations):
        raise ValueError("Exact tabular declarations contain duplicate paths.")
    facts: dict[str, AnalysisArrayStorageFacts] = {}
    for path, declaration in sorted(declaration_by_path.items()):
        array = _array_at_path(run_group, path)
        facts[path] = AnalysisArrayStorageFacts(
            path=path,
            shape=tuple(int(value) for value in array.shape),
            dtype=array.dtype,
            growth_axis=_growth_axis(declaration),
            access_unit_semantics=_access_unit_semantics(declaration),
        )
    return plan_analysis_storage(
        declarations,
        facts,
        profile=profile,
    )


def exact_tabular_fill_value(declaration: AnalysisArrayDeclaration) -> object:
    """Return the frozen physical initialization value for a complete snapshot."""

    dtype_name = declaration.contract.dtype.numpy_dtype
    if dtype_name is None:
        raise ValueError(
            f"{declaration.path}: variable-width arrays require another factory."
        )
    dtype = np.dtype(dtype_name)
    if dtype.kind in {"f", "c"}:
        return float("nan")
    if dtype.kind == "b":
        return False
    return 0


def _copy_group_attributes_and_structure(source: Any, destination: Any) -> None:
    destination.attrs.update(dict(source.attrs))
    for name, source_child in sorted(source.groups(), key=lambda item: item[0]):
        destination_child = destination.create_group(name)
        _copy_group_attributes_and_structure(source_child, destination_child)


def _write_by_physical_units(
    destination: Any,
    source: Any,
    *,
    entry: Any,
) -> None:
    plan = entry.plan
    growth_axis = entry.facts.growth_axis
    if plan.chunk_shape is None or growth_axis is None:
        destination[...] = np.asarray(source[...])
        return
    unit_shape = plan.shard_shape or plan.chunk_shape
    unit_extent = int(unit_shape[growth_axis])
    axis_extent = int(plan.logical_shape[growth_axis])
    selection = [slice(None)] * len(plan.logical_shape)
    for start in range(0, axis_extent, unit_extent):
        stop = min(start + unit_extent, axis_extent)
        selection[growth_axis] = slice(start, stop)
        index = tuple(selection)
        destination[index] = np.asarray(source[index])


def rematerialize_exact_tabular_candidate(
    source_run: Any,
    destination_run: Any,
    *,
    receipt: AnalysisStoragePlanReceipt,
) -> None:
    """Copy one exact run into its planned physical layout.

    The destination must be a fresh group.  Declared scientific arrays use the
    shared factory.  Arrays under the schema's explicitly excluded report and
    visualization namespaces are copied as non-authoritative artifacts; any
    other undeclared array fails closed.
    """

    if list(destination_run.array_keys()) or list(destination_run.group_keys()):
        raise ValueError("Exact tabular candidate destination must be empty.")
    parsed = analysis_storage_plan_receipt_from_manifest(receipt.as_manifest())
    if parsed.as_manifest() != receipt.as_manifest():
        raise ValueError("Exact tabular candidate receipt is not executable.")
    _copy_group_attributes_and_structure(source_run, destination_run)
    entries = {entry.declaration.path: entry for entry in receipt.entries}
    observed_arrays = dict(_iter_arrays(source_run))
    unexpected = sorted(
        path
        for path in set(observed_arrays) - set(entries)
        if path.split("/", 1)[0] not in EXCLUDED_REPORT_PREFIXES
    )
    if unexpected:
        raise ValueError(
            f"Exact tabular source contains undeclared scientific arrays: {unexpected!r}."
        )
    for path in sorted(entries):
        entry = entries[path]
        source = _array_at_path(source_run, path)
        parent, leaf = _parent_and_leaf(destination_run, path)
        attributes = {
            str(key): value
            for key, value in dict(source.attrs).items()
            if key not in _RESERVED_ARRAY_ATTRIBUTES
        }
        destination = create_array_from_plan(
            parent,
            name=leaf,
            contract=entry.declaration.contract,
            plan=entry.plan,
            fill_value=exact_tabular_fill_value(entry.declaration),
            attributes=attributes,
        )
        _write_by_physical_units(destination, source, entry=entry)

    for path in sorted(set(observed_arrays) - set(entries)):
        source = observed_arrays[path]
        parent, leaf = _parent_and_leaf(destination_run, path)
        destination = parent.create_array(
            leaf,
            data=np.asarray(source[...]),
            overwrite=False,
        )
        destination.attrs.update(dict(source.attrs))


def persist_exact_tabular_storage_receipt(
    run_group: Any,
    receipt: AnalysisStoragePlanReceipt,
) -> dict[str, object]:
    manifest = receipt.as_manifest()
    run_group.attrs[ANALYSIS_STORAGE_PLAN_RECEIPT_ATTR] = manifest
    run_group.attrs[ANALYSIS_STORAGE_PLAN_DIGEST_ATTR] = manifest["payload_digest"]
    run_group.attrs[ANALYSIS_STORAGE_PROFILE_ID_ATTR] = receipt.profile.profile_id
    run_group.attrs[ANALYSIS_STORAGE_PROFILE_ROLE_ATTR] = ANALYSIS_STORAGE_PROFILE_ROLE
    run_group.attrs["stage_selector_eligible"] = False
    return manifest


def _normalized_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalized_metadata(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_normalized_metadata(child) for child in value]
    if value == "NaN":
        return {"palette_exact_float": "nan"}
    if isinstance(value, (float, np.floating)) and math.isnan(float(value)):
        return {"palette_exact_float": "nan"}
    return value


def validate_exact_tabular_storage_receipt(
    run_group: Any,
    *,
    declarations: Sequence[AnalysisArrayDeclaration],
) -> tuple[str, ...]:
    """Replan a persisted receipt and verify every direct array declaration."""

    errors: list[str] = []
    persisted = run_group.attrs.get(ANALYSIS_STORAGE_PLAN_RECEIPT_ATTR)
    if not isinstance(persisted, Mapping):
        return ("analysis storage-plan receipt is absent or not an object",)
    try:
        parsed = analysis_storage_plan_receipt_from_manifest(persisted)
        expected = build_exact_tabular_storage_receipt(
            run_group,
            declarations=declarations,
            profile=parsed.profile,
        )
    except Exception as exc:
        return (f"analysis storage plan could not be reconstructed: {exc}",)
    if dict(persisted) != expected.as_manifest():
        errors.append("analysis storage-plan receipt differs from executable planning")
    if run_group.attrs.get(ANALYSIS_STORAGE_PLAN_DIGEST_ATTR) != persisted.get(
        "payload_digest"
    ):
        errors.append("analysis storage-plan redundant digest binding mismatch")
    if run_group.attrs.get(ANALYSIS_STORAGE_PROFILE_ID_ATTR) != parsed.profile.profile_id:
        errors.append("analysis storage profile identity binding mismatch")
    if (
        run_group.attrs.get(ANALYSIS_STORAGE_PROFILE_ROLE_ATTR)
        != ANALYSIS_STORAGE_PROFILE_ROLE
    ):
        errors.append("analysis storage profile role is not the candidate role")
    if run_group.attrs.get("stage_selector_eligible") is not False:
        errors.append("analysis storage candidate is not selector-ineligible")

    entry_by_path = {entry.declaration.path: entry for entry in expected.entries}
    declaration_by_path = {
        declaration.path: declaration for declaration in declarations
    }
    if set(entry_by_path) != set(declaration_by_path):
        return (*errors, "analysis storage declaration inventory mismatch")
    for path in sorted(entry_by_path):
        try:
            array = _array_at_path(run_group, path)
            raw = array.metadata.to_dict()
            attributes = raw.get("attributes")
            if not isinstance(attributes, Mapping):
                raise ValueError("array metadata attributes are not an object")
            nonreserved = {
                str(key): value
                for key, value in attributes.items()
                if key not in _RESERVED_ARRAY_ATTRIBUTES
            }
            expected_metadata = array_metadata_declaration_from_plan(
                contract=declaration_by_path[path].contract,
                plan=entry_by_path[path].plan,
                fill_value=exact_tabular_fill_value(declaration_by_path[path]),
                attributes=nonreserved,
            )
            observed_metadata = {
                key: value
                for key, value in raw.items()
                if key not in {"zarr_format", "node_type", "consolidated_metadata"}
            }
            if _normalized_metadata(observed_metadata) != _normalized_metadata(
                expected_metadata
            ):
                errors.append(
                    f"{path}: array metadata differs from resolved chunks/shards/codecs"
                )
        except Exception as exc:
            errors.append(f"{path}: physical metadata validation failed: {exc}")
    return tuple(errors)


__all__ = [
    "ANALYSIS_STORAGE_PLAN_DIGEST_ATTR",
    "ANALYSIS_STORAGE_PLAN_RECEIPT_ATTR",
    "ANALYSIS_STORAGE_PROFILE_ID_ATTR",
    "ANALYSIS_STORAGE_PROFILE_ROLE",
    "ANALYSIS_STORAGE_PROFILE_ROLE_ATTR",
    "build_exact_tabular_storage_receipt",
    "exact_tabular_fill_value",
    "persist_exact_tabular_storage_receipt",
    "rematerialize_exact_tabular_candidate",
    "validate_exact_tabular_storage_receipt",
]
