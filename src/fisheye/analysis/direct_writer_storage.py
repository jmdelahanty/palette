"""Byte-planned array creation for the two guarded direct analytics writers.

This module is intentionally local to the tail-posture and bout-classification
writers.  It adapts their frozen logical declarations to the shared analysis
storage planner without selecting a profile or changing publication policy.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from fisheye.shared.zarr.analysis_array_contracts import AnalysisArrayDeclaration
from fisheye.shared.zarr.analysis_storage_planning import (
    AnalysisArrayStorageFacts,
    AnalysisStoragePlanReceipt,
    plan_analysis_storage,
)
from fisheye.shared.zarr.array_factory import (
    array_metadata_declaration_from_plan,
    create_array_from_plan,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.storage_profiles import (
    StorageProfile,
    storage_profile_from_manifest,
)

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


def _normalized_metadata(value: Any) -> Any:
    """Normalize tuple/list and NaN representation for exact metadata comparison."""

    if isinstance(value, Mapping):
        return {str(key): _normalized_metadata(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_normalized_metadata(child) for child in value]
    if value == "NaN":
        return {"palette_exact_float": "nan"}
    if isinstance(value, (float, np.floating)) and math.isnan(float(value)):
        return {"palette_exact_float": "nan"}
    return value


def _array_at_path(run_group: Any, path: str) -> Any:
    node = run_group
    for component in path.split("/"):
        node = node[component]
    return node


def _parent_and_leaf(run_group: Any, path: str) -> tuple[Any, str]:
    components = path.split("/")
    parent = run_group
    for component in components[:-1]:
        child = parent.get(component)
        if child is None:
            child = parent.create_group(component)
        parent = child
    return parent, components[-1]


def build_direct_writer_storage_receipt(
    *,
    declarations: Sequence[AnalysisArrayDeclaration],
    arrays_by_path: Mapping[str, np.ndarray],
    access_unit_semantics: Mapping[str, str],
    profile: StorageProfile,
    dimensions: Mapping[str, int],
) -> AnalysisStoragePlanReceipt:
    """Resolve one exact candidate receipt from the actual fixed-width arrays."""

    if not isinstance(profile, StorageProfile):
        raise TypeError("profile must be an explicitly supplied StorageProfile.")
    declaration_paths = {declaration.path for declaration in declarations}
    if set(arrays_by_path) != declaration_paths:
        raise ValueError("Direct-writer arrays must exactly match their declarations.")
    if set(access_unit_semantics) != declaration_paths:
        raise ValueError(
            "Direct-writer access-unit semantics must exactly match declarations."
        )
    facts: dict[str, AnalysisArrayStorageFacts] = {}
    for path in sorted(declaration_paths):
        data = np.asarray(arrays_by_path[path])
        facts[path] = AnalysisArrayStorageFacts(
            path=path,
            shape=tuple(int(value) for value in data.shape),
            dtype=data.dtype,
            access_unit_semantics=access_unit_semantics[path],
        )
    return plan_analysis_storage(
        declarations,
        facts,
        profile=profile,
        dimensions=dimensions,
    )


def persist_direct_writer_storage_receipt(
    run_group: Any,
    receipt: AnalysisStoragePlanReceipt,
) -> dict[str, object]:
    """Persist the canonical candidate receipt and redundant identity bindings."""

    manifest = receipt.as_manifest()
    run_group.attrs[ANALYSIS_STORAGE_PLAN_RECEIPT_ATTR] = manifest
    run_group.attrs[ANALYSIS_STORAGE_PLAN_DIGEST_ATTR] = manifest["payload_digest"]
    run_group.attrs[ANALYSIS_STORAGE_PROFILE_ID_ATTR] = receipt.profile.profile_id
    run_group.attrs[ANALYSIS_STORAGE_PROFILE_ROLE_ATTR] = ANALYSIS_STORAGE_PROFILE_ROLE
    return manifest


def create_direct_writer_arrays_from_receipt(
    run_group: Any,
    *,
    receipt: AnalysisStoragePlanReceipt,
    arrays_by_path: Mapping[str, np.ndarray],
    fill_values: Mapping[str, object],
) -> None:
    """Create and populate every declared array through the shared factory."""

    entries = {entry.declaration.path: entry for entry in receipt.entries}
    if set(entries) != set(arrays_by_path) or set(entries) != set(fill_values):
        raise ValueError(
            "Receipt, payload arrays, and fill values must have identical paths."
        )
    for path in sorted(entries):
        entry = entries[path]
        values = np.asarray(arrays_by_path[path])
        parent, leaf = _parent_and_leaf(run_group, path)
        if parent.get(leaf) is not None:
            raise ValueError(f"Direct-writer destination {path!r} already exists.")
        destination = create_array_from_plan(
            parent,
            name=leaf,
            contract=entry.declaration.contract,
            plan=entry.plan,
            fill_value=fill_values[path],
        )
        if values.size:
            destination[...] = values


def validate_direct_writer_storage_receipt(
    run_group: Any,
    *,
    declarations: Sequence[AnalysisArrayDeclaration],
    access_unit_semantics: Mapping[str, str],
    fill_values: Mapping[str, object],
    dimensions: Mapping[str, int],
) -> tuple[str, ...]:
    """Deeply recompute a persisted receipt and verify physical metadata."""

    errors: list[str] = []
    persisted = run_group.attrs.get(ANALYSIS_STORAGE_PLAN_RECEIPT_ATTR)
    if not isinstance(persisted, Mapping):
        return ("analysis storage-plan receipt is absent or not an object",)
    if set(persisted) != {
        "schema_id",
        "schema_version",
        "payload",
        "payload_digest",
    }:
        return ("analysis storage-plan receipt has an unexpected field set",)
    payload = persisted.get("payload")
    digest = persisted.get("payload_digest")
    if not isinstance(payload, Mapping) or type(digest) is not str:
        return ("analysis storage-plan payload or digest has the wrong type",)
    try:
        recomputed_digest = canonical_json_sha256(payload)
    except Exception as exc:
        return (f"analysis storage-plan payload is not canonical JSON: {exc}",)
    if digest != recomputed_digest:
        errors.append("analysis storage-plan payload digest mismatch")
    if run_group.attrs.get(ANALYSIS_STORAGE_PLAN_DIGEST_ATTR) != digest:
        errors.append("analysis storage-plan redundant digest binding mismatch")
    profile_value = payload.get("storage_profile")
    if not isinstance(profile_value, Mapping):
        errors.append("analysis storage profile is absent or not an object")
        return tuple(errors)
    try:
        profile = storage_profile_from_manifest(profile_value)
    except Exception as exc:
        errors.append(f"analysis storage profile is invalid: {exc}")
        return tuple(errors)
    if run_group.attrs.get(ANALYSIS_STORAGE_PROFILE_ID_ATTR) != profile.profile_id:
        errors.append("analysis storage profile identity binding mismatch")
    if (
        run_group.attrs.get(ANALYSIS_STORAGE_PROFILE_ROLE_ATTR)
        != ANALYSIS_STORAGE_PROFILE_ROLE
    ):
        errors.append("analysis storage profile role is not the candidate role")

    declaration_by_path = {
        declaration.path: declaration for declaration in declarations
    }
    try:
        facts = {
            path: AnalysisArrayStorageFacts(
                path=path,
                shape=tuple(
                    int(value) for value in _array_at_path(run_group, path).shape
                ),
                dtype=_array_at_path(run_group, path).dtype,
                access_unit_semantics=access_unit_semantics[path],
            )
            for path in declaration_by_path
        }
        expected = plan_analysis_storage(
            declarations,
            facts,
            profile=profile,
            dimensions=dimensions,
        )
    except Exception as exc:
        errors.append(f"analysis storage plan could not be recomputed: {exc}")
        return tuple(errors)
    expected_manifest = expected.as_manifest()
    if dict(persisted) != expected_manifest:
        errors.append("analysis storage-plan receipt differs from executable planning")

    if set(fill_values) != set(declaration_by_path):
        errors.append("analysis storage fill-value inventory is incomplete")
        return tuple(errors)
    entries = {entry.declaration.path: entry for entry in expected.entries}
    for path in sorted(declaration_by_path):
        try:
            array = _array_at_path(run_group, path)
            raw_metadata = array.metadata.to_dict()
            attributes = raw_metadata.get("attributes")
            if not isinstance(attributes, Mapping):
                raise ValueError("array metadata attributes are not an object")
            nonreserved_attributes = {
                str(key): value
                for key, value in attributes.items()
                if key not in _RESERVED_ARRAY_ATTRIBUTES
            }
            expected_metadata = array_metadata_declaration_from_plan(
                contract=declaration_by_path[path].contract,
                plan=entries[path].plan,
                fill_value=fill_values[path],
                attributes=nonreserved_attributes,
            )
            observed_metadata = {
                key: value
                for key, value in raw_metadata.items()
                if key not in {"zarr_format", "node_type", "consolidated_metadata"}
            }
            metadata_errors = (
                ()
                if _normalized_metadata(observed_metadata)
                == _normalized_metadata(expected_metadata)
                else ("array metadata differs from the resolved storage plan",)
            )
        except Exception as exc:
            errors.append(f"{path}: physical metadata validation failed: {exc}")
            continue
        errors.extend(f"{path}: {message}" for message in metadata_errors)
    return tuple(errors)


__all__ = [
    "ANALYSIS_STORAGE_PLAN_DIGEST_ATTR",
    "ANALYSIS_STORAGE_PLAN_RECEIPT_ATTR",
    "ANALYSIS_STORAGE_PROFILE_ID_ATTR",
    "ANALYSIS_STORAGE_PROFILE_ROLE",
    "ANALYSIS_STORAGE_PROFILE_ROLE_ATTR",
    "build_direct_writer_storage_receipt",
    "create_direct_writer_arrays_from_receipt",
    "persist_direct_writer_storage_receipt",
    "validate_direct_writer_storage_receipt",
]
