"""Byte-planned physical contract for opt-in stimulus-response v3 candidates."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import zarr

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
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.storage_profiles import PUBLISHED_HTTP_V1, StorageProfile
from fisheye.shared.zarr.stimulus_response_schema import (
    STIMULUS_RESPONSE_STORAGE_PROFILE_ROLE,
    STIMULUS_RESPONSE_STORAGE_PROFILE_ROLE_ATTR,
    expected_table_names,
    stimulus_response_candidate_fill_value,
    stimulus_response_array_declarations,
    table_contract,
)
from fisheye.shared.zarr_helpers import zarr_store_metadata_publication_lock

STIMULUS_RESPONSE_CANDIDATE_PROFILE_ID = PUBLISHED_HTTP_V1.profile_id
STIMULUS_RESPONSE_STORAGE_PLAN_RECEIPT_ATTR = "analysis_storage_plan_receipt"
STIMULUS_RESPONSE_STORAGE_PLAN_DIGEST_ATTR = "analysis_storage_plan_payload_sha256"
STIMULUS_RESPONSE_STORAGE_PROFILE_ID_ATTR = "analysis_storage_profile_id"
STIMULUS_RESPONSE_METADATA_EQUIVALENCE_ATTR = (
    "analysis_storage_direct_consolidated_equivalence"
)
STIMULUS_RESPONSE_CANDIDATE_ATTR = "stimulus_response_storage_candidate"
STIMULUS_RESPONSE_CANDIDATE_SCHEMA_ID = "palette.stimulus_response.storage_candidate"
STIMULUS_RESPONSE_CANDIDATE_SCHEMA_VERSION = 1
STIMULUS_RESPONSE_METADATA_EQUIVALENCE_SCHEMA_ID = (
    "palette.stimulus_response.metadata_equivalence"
)
STIMULUS_RESPONSE_METADATA_EQUIVALENCE_SCHEMA_VERSION = 1
STIMULUS_RESPONSE_METADATA_NORMALIZATION = (
    "zarr_v3_metadata_without_consolidated_envelopes_or_equivalence_receipt_v1"
)

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


def _require_candidate_profile(profile: StorageProfile) -> StorageProfile:
    if not isinstance(profile, StorageProfile):
        raise TypeError("storage profile must be an explicit StorageProfile.")
    if (
        profile.profile_id != STIMULUS_RESPONSE_CANDIDATE_PROFILE_ID
        or profile.as_manifest() != PUBLISHED_HTTP_V1.as_manifest()
    ):
        raise ValueError(
            "Stimulus-response v3 candidate supports only the exact "
            f"{STIMULUS_RESPONSE_CANDIDATE_PROFILE_ID!r} profile."
        )
    return profile


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


def _shape_and_dtype(value: Any) -> tuple[tuple[int, ...], np.dtype]:
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is None or dtype is None:
        array = np.asarray(value)
        shape = array.shape
        dtype = array.dtype
    return tuple(int(extent) for extent in shape), np.dtype(dtype)


def stimulus_response_dimensions(
    arrays_by_path: Mapping[str, Any],
    *,
    bundles: Sequence[str],
) -> dict[str, int]:
    """Bind every symbolic row axis to the exact table cardinality."""

    dimensions: dict[str, int] = {}
    axis_by_table: dict[str, str] = {}
    for declaration in stimulus_response_array_declarations(
        bundles=bundles,
        byte_planner_adopted=True,
    ):
        table_name = declaration.path.split("/", 1)[0]
        row_axis = declaration.contract.axis_names[0]
        previous_axis = axis_by_table.setdefault(table_name, row_axis)
        if previous_axis != row_axis:
            raise ValueError(f"{table_name} declarations disagree on their row axis.")
    for table_name in expected_table_names(bundles):
        table = table_contract(table_name, bundles=bundles)
        row_counts = {
            _shape_and_dtype(arrays_by_path[f"{table_name}/{field_name}"])[0][0]
            for field_name in table.field_names
        }
        if len(row_counts) != 1:
            raise ValueError(f"{table_name} arrays have inconsistent row counts.")
        row_count = row_counts.pop()
        row_axis = axis_by_table[table_name]
        previous = dimensions.get(row_axis)
        if previous is not None and previous != row_count:
            raise ValueError(
                f"Symbolic dimension {row_axis!r} disagrees across tables: "
                f"{previous} versus {row_count}."
            )
        dimensions[row_axis] = row_count
    return dimensions


def stimulus_response_fill_values(
    *,
    bundles: Sequence[str],
) -> dict[str, object]:
    """Return path-exact physical fills matching the frozen semantic contract."""

    fills: dict[str, object] = {}
    for table_name in expected_table_names(bundles):
        table = table_contract(table_name, bundles=bundles)
        for field_name, field in table.fields:
            path = f"{table_name}/{field_name}"
            fills[path] = stimulus_response_candidate_fill_value(
                field_name,
                field,
            )
    return fills


def stimulus_response_access_unit_semantics(
    *,
    bundles: Sequence[str],
) -> dict[str, str]:
    """Name the complete table row that cannot be split by byte planning."""

    return {
        declaration.path: (
            "one complete row of table "
            f"{declaration.path.split('/', 1)[0]!r}, including the complete "
            "fixed-width UTF-8 byte record when present"
        )
        for declaration in stimulus_response_array_declarations(
            bundles=bundles,
            byte_planner_adopted=True,
        )
    }


def build_stimulus_response_storage_receipt(
    *,
    arrays_by_path: Mapping[str, Any],
    bundles: Sequence[str],
    profile: StorageProfile,
) -> AnalysisStoragePlanReceipt:
    """Plan every exact v3 array from concrete uncompressed byte facts."""

    profile = _require_candidate_profile(profile)
    declarations = stimulus_response_array_declarations(
        bundles=bundles,
        byte_planner_adopted=True,
    )
    declaration_paths = {declaration.path for declaration in declarations}
    if set(arrays_by_path) != declaration_paths:
        raise ValueError(
            "Stimulus-response candidate payload must exactly match its declarations."
        )
    semantics = stimulus_response_access_unit_semantics(bundles=bundles)
    facts: dict[str, AnalysisArrayStorageFacts] = {}
    for path, values in arrays_by_path.items():
        shape, dtype = _shape_and_dtype(values)
        facts[path] = AnalysisArrayStorageFacts(
            path=path,
            shape=shape,
            dtype=dtype,
            access_unit_semantics=semantics[path],
        )
    return plan_analysis_storage(
        declarations,
        facts,
        profile=profile,
        dimensions=stimulus_response_dimensions(
            arrays_by_path,
            bundles=bundles,
        ),
    )


def persist_stimulus_response_storage_receipt(
    run_group: Any,
    receipt: AnalysisStoragePlanReceipt,
) -> dict[str, object]:
    """Persist the exact unpromoted-candidate plan and lifecycle envelope."""

    _require_candidate_profile(receipt.profile)
    manifest = receipt.as_manifest()
    run_group.attrs[STIMULUS_RESPONSE_STORAGE_PLAN_RECEIPT_ATTR] = manifest
    run_group.attrs[STIMULUS_RESPONSE_STORAGE_PLAN_DIGEST_ATTR] = manifest[
        "payload_digest"
    ]
    run_group.attrs[STIMULUS_RESPONSE_STORAGE_PROFILE_ID_ATTR] = (
        receipt.profile.profile_id
    )
    run_group.attrs[STIMULUS_RESPONSE_STORAGE_PROFILE_ROLE_ATTR] = (
        STIMULUS_RESPONSE_STORAGE_PROFILE_ROLE
    )
    run_group.attrs[STIMULUS_RESPONSE_CANDIDATE_ATTR] = {
        "schema_id": STIMULUS_RESPONSE_CANDIDATE_SCHEMA_ID,
        "schema_version": STIMULUS_RESPONSE_CANDIDATE_SCHEMA_VERSION,
        "profile_id": receipt.profile.profile_id,
        "status": "unpromoted_selector_ineligible",
        "write_ownership": "serial_single_writer_whole_shard",
    }
    return manifest


def create_stimulus_response_arrays_from_receipt(
    run_group: Any,
    *,
    receipt: AnalysisStoragePlanReceipt,
    arrays_by_path: Mapping[str, np.ndarray],
    bundles: Sequence[str],
) -> None:
    """Create and populate every array serially through the shared v3 factory."""

    expected = build_stimulus_response_storage_receipt(
        arrays_by_path=arrays_by_path,
        bundles=bundles,
        profile=receipt.profile,
    )
    if receipt.as_manifest() != expected.as_manifest():
        raise ValueError(
            "Stimulus-response creation receipt differs from executable planning."
        )
    fills = stimulus_response_fill_values(bundles=bundles)
    entries = {entry.declaration.path: entry for entry in receipt.entries}
    if set(entries) != set(arrays_by_path) or set(entries) != set(fills):
        raise ValueError("Stimulus-response receipt/payload/fill inventories disagree.")
    for path in sorted(entries):
        entry = entries[path]
        if entry.plan.is_sharded and (
            entry.plan.write_ownership != "whole_shard_single_writer"
        ):
            raise ValueError(f"{path}: sharded candidate lacks whole-shard ownership.")
        parent, leaf = _parent_and_leaf(run_group, path)
        if parent.get(leaf) is not None:
            raise ValueError(f"Stimulus-response destination {path!r} already exists.")
        destination = create_array_from_plan(
            parent,
            name=leaf,
            contract=entry.declaration.contract,
            plan=entry.plan,
            fill_value=fills[path],
        )
        values = np.asarray(arrays_by_path[path])
        if values.size:
            destination[...] = values


def _normalized_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _normalized_metadata(child)
            for key, child in value.items()
            if key
            not in {
                "consolidated_metadata",
                STIMULUS_RESPONSE_METADATA_EQUIVALENCE_ATTR,
            }
        }
    if isinstance(value, (tuple, list)):
        return [_normalized_metadata(child) for child in value]
    if value == "NaN":
        return {"palette_exact_float": "nan"}
    if isinstance(value, (float, np.floating)) and math.isnan(float(value)):
        return {"palette_exact_float": "nan"}
    return value


def _metadata_equivalence_document(
    run_group: Any,
    *,
    bundles: Sequence[str],
) -> dict[str, object]:
    declarations = stimulus_response_array_declarations(
        bundles=bundles,
        byte_planner_adopted=True,
    )
    table_names = expected_table_names(bundles)
    groups: dict[str, object] = {}
    for group_path in ("", *table_names):
        group = run_group if not group_path else run_group[group_path]
        groups[group_path] = _normalized_metadata(group.metadata.to_dict())
    arrays = {
        declaration.path: _normalized_metadata(
            _array_at_path(run_group, declaration.path).metadata.to_dict()
        )
        for declaration in declarations
    }
    return {
        "schema_id": "palette.stimulus_response.normalized_zarr_metadata",
        "schema_version": 1,
        "normalization": STIMULUS_RESPONSE_METADATA_NORMALIZATION,
        "groups": groups,
        "arrays": arrays,
    }


def _metadata_equivalence_receipt(
    run_group: Any,
    *,
    run_path: str,
    profile_id: str,
    bundles: Sequence[str],
) -> dict[str, object]:
    declarations = stimulus_response_array_declarations(
        bundles=bundles,
        byte_planner_adopted=True,
    )
    table_names = expected_table_names(bundles)
    metadata_document = _metadata_equivalence_document(
        run_group,
        bundles=bundles,
    )
    payload = {
        "run_path": run_path,
        "profile_id": profile_id,
        "normalization": STIMULUS_RESPONSE_METADATA_NORMALIZATION,
        "array_declaration_count": len(declarations),
        "group_declaration_count": 1 + len(table_names),
        "normalized_metadata_sha256": canonical_json_sha256(metadata_document),
        "result": "direct_and_consolidated_metadata_equal",
    }
    return {
        "schema_id": STIMULUS_RESPONSE_METADATA_EQUIVALENCE_SCHEMA_ID,
        "schema_version": STIMULUS_RESPONSE_METADATA_EQUIVALENCE_SCHEMA_VERSION,
        "payload": payload,
        "payload_digest": canonical_json_sha256(payload),
    }


def validate_stimulus_response_metadata_equivalence(
    run_group: Any,
) -> tuple[str, ...]:
    """Validate exact, current metadata-equivalence evidence for one candidate."""

    value = run_group.attrs.get(STIMULUS_RESPONSE_METADATA_EQUIVALENCE_ATTR)
    if not isinstance(value, Mapping) or set(value) != {
        "schema_id",
        "schema_version",
        "payload",
        "payload_digest",
    }:
        return ("metadata-equivalence receipt is absent or not exact",)
    if (
        value.get("schema_id") != STIMULUS_RESPONSE_METADATA_EQUIVALENCE_SCHEMA_ID
        or value.get("schema_version")
        != STIMULUS_RESPONSE_METADATA_EQUIVALENCE_SCHEMA_VERSION
    ):
        return ("metadata-equivalence receipt schema identity mismatch",)
    payload = value.get("payload")
    if not isinstance(payload, Mapping) or set(payload) != {
        "run_path",
        "profile_id",
        "normalization",
        "array_declaration_count",
        "group_declaration_count",
        "normalized_metadata_sha256",
        "result",
    }:
        return ("metadata-equivalence receipt payload is not exact",)
    if value.get("payload_digest") != canonical_json_sha256(payload):
        return ("metadata-equivalence receipt payload digest mismatch",)
    errors: list[str] = []
    run_path = payload.get("run_path")
    if (
        type(run_path) is not str
        or not run_path
        or run_path.startswith("/")
        or "//" in run_path
    ):
        errors.append("metadata-equivalence run path is invalid")
    observed_run_path = str(getattr(run_group, "path", "")).strip("/")
    if observed_run_path and run_path != observed_run_path:
        errors.append("metadata-equivalence run path binding mismatch")
    profile_id = run_group.attrs.get(STIMULUS_RESPONSE_STORAGE_PROFILE_ID_ATTR)
    if payload.get("profile_id") != profile_id:
        errors.append("metadata-equivalence profile binding mismatch")
    if payload.get("normalization") != STIMULUS_RESPONSE_METADATA_NORMALIZATION:
        errors.append("metadata-equivalence normalization identity mismatch")
    if payload.get("result") != "direct_and_consolidated_metadata_equal":
        errors.append("metadata-equivalence result is not accepted")
    bundles = run_group.attrs.get("stimulus_response_v3_bundles")
    if not isinstance(bundles, list) or any(type(item) is not str for item in bundles):
        errors.append("metadata-equivalence bundle declaration is invalid")
        return tuple(errors)
    declarations = stimulus_response_array_declarations(
        bundles=bundles,
        byte_planner_adopted=True,
    )
    table_names = expected_table_names(bundles)
    if payload.get("array_declaration_count") != len(declarations):
        errors.append("metadata-equivalence array count mismatch")
    if payload.get("group_declaration_count") != 1 + len(table_names):
        errors.append("metadata-equivalence group count mismatch")
    try:
        current_digest = canonical_json_sha256(
            _metadata_equivalence_document(run_group, bundles=bundles)
        )
    except Exception as exc:
        errors.append(f"metadata-equivalence declarations cannot be read: {exc}")
    else:
        if payload.get("normalized_metadata_sha256") != current_digest:
            errors.append("metadata-equivalence receipt is stale")
    return tuple(errors)


def _validate_candidate_envelope(run_group: Any, *, profile_id: str) -> tuple[str, ...]:
    expected = {
        "schema_id": STIMULUS_RESPONSE_CANDIDATE_SCHEMA_ID,
        "schema_version": STIMULUS_RESPONSE_CANDIDATE_SCHEMA_VERSION,
        "profile_id": profile_id,
        "status": "unpromoted_selector_ineligible",
        "write_ownership": "serial_single_writer_whole_shard",
    }
    errors: list[str] = []
    if run_group.attrs.get(STIMULUS_RESPONSE_CANDIDATE_ATTR) != expected:
        errors.append("stimulus-response storage candidate envelope is not exact")
    if run_group.attrs.get("stage_selector_eligible") is not False:
        errors.append("stimulus-response storage candidate is selector eligible")
    return tuple(errors)


def validate_stimulus_response_storage_receipt(run_group: Any) -> tuple[str, ...]:
    """Replan the persisted receipt and validate every live physical declaration."""

    errors: list[str] = []
    persisted = run_group.attrs.get(STIMULUS_RESPONSE_STORAGE_PLAN_RECEIPT_ATTR)
    if not isinstance(persisted, Mapping):
        return ("analysis storage-plan receipt is absent or not an object",)
    try:
        parsed = analysis_storage_plan_receipt_from_manifest(persisted)
        _require_candidate_profile(parsed.profile)
    except Exception as exc:
        return (f"analysis storage-plan receipt is invalid: {exc}",)
    if run_group.attrs.get(STIMULUS_RESPONSE_STORAGE_PLAN_DIGEST_ATTR) != persisted.get(
        "payload_digest"
    ):
        errors.append("analysis storage-plan redundant digest binding mismatch")
    if (
        run_group.attrs.get(STIMULUS_RESPONSE_STORAGE_PROFILE_ID_ATTR)
        != parsed.profile.profile_id
    ):
        errors.append("analysis storage profile identity binding mismatch")
    if (
        run_group.attrs.get(STIMULUS_RESPONSE_STORAGE_PROFILE_ROLE_ATTR)
        != STIMULUS_RESPONSE_STORAGE_PROFILE_ROLE
    ):
        errors.append("analysis storage profile role is not the candidate role")
    errors.extend(
        _validate_candidate_envelope(
            run_group,
            profile_id=parsed.profile.profile_id,
        )
    )

    bundles = run_group.attrs.get("stimulus_response_v3_bundles")
    if not isinstance(bundles, list) or any(
        type(value) is not str for value in bundles
    ):
        errors.append("stimulus-response bundle declaration is invalid")
        return tuple(errors)
    declarations = stimulus_response_array_declarations(
        bundles=bundles,
        byte_planner_adopted=True,
    )
    arrays_by_path = {
        declaration.path: _array_at_path(run_group, declaration.path)
        for declaration in declarations
    }
    try:
        expected = build_stimulus_response_storage_receipt(
            arrays_by_path=arrays_by_path,
            bundles=bundles,
            profile=parsed.profile,
        )
    except Exception as exc:
        errors.append(f"analysis storage plan could not be recomputed: {exc}")
        return tuple(errors)
    if dict(persisted) != expected.as_manifest():
        errors.append("analysis storage-plan receipt differs from executable planning")

    fills = stimulus_response_fill_values(bundles=bundles)
    entry_by_path = {entry.declaration.path: entry for entry in expected.entries}
    declaration_by_path = {
        declaration.path: declaration for declaration in declarations
    }
    for path in sorted(declaration_by_path):
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
                fill_value=fills[path],
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
                    f"{path}: array metadata differs from resolved storage plan"
                )
        except Exception as exc:
            errors.append(f"{path}: physical metadata validation failed: {exc}")
    return tuple(errors)


def consolidate_and_validate_stimulus_response_metadata(
    root: Any,
    *,
    run_path: str,
) -> dict[str, object]:
    """Consolidate one immutable generation and compare all run declarations."""

    if str(getattr(root, "path", "")).strip("/"):
        raise ValueError("Metadata consolidation requires the archive root group.")
    with zarr_store_metadata_publication_lock(root.store):
        direct_run = root[run_path]
        persisted = direct_run.attrs.get(
            STIMULUS_RESPONSE_STORAGE_PLAN_RECEIPT_ATTR
        )
        if not isinstance(persisted, Mapping):
            raise ValueError("Stimulus-response candidate has no storage-plan receipt.")
        parsed = analysis_storage_plan_receipt_from_manifest(persisted)
        _require_candidate_profile(parsed.profile)
        bundles = direct_run.attrs.get("stimulus_response_v3_bundles")
        if not isinstance(bundles, list):
            raise ValueError("Stimulus-response candidate has no exact bundle list.")
        receipt = _metadata_equivalence_receipt(
            direct_run,
            run_path=run_path,
            profile_id=parsed.profile.profile_id,
            bundles=bundles,
        )
        direct_run.attrs[STIMULUS_RESPONSE_METADATA_EQUIVALENCE_ATTR] = receipt
        zarr.consolidate_metadata(root.store)
        direct_root = zarr.open_group(root.store, mode="r", use_consolidated=False)
        consolidated_root = zarr.open_group(
            root.store, mode="r", use_consolidated=True
        )
        direct_run = direct_root[run_path]
        consolidated_run = consolidated_root[run_path]
        direct_document = _metadata_equivalence_document(direct_run, bundles=bundles)
        consolidated_document = _metadata_equivalence_document(
            consolidated_run,
            bundles=bundles,
        )
        if direct_document != consolidated_document:
            raise ValueError("Stimulus-response direct/consolidated metadata differ.")
        for label, group in (
            ("direct", direct_run),
            ("consolidated", consolidated_run),
        ):
            equivalence_errors = validate_stimulus_response_metadata_equivalence(group)
            if equivalence_errors:
                raise ValueError(
                    f"Stimulus-response {label} metadata-equivalence receipt is invalid: "
                    + "; ".join(equivalence_errors)
                )
        return receipt


__all__ = [
    "STIMULUS_RESPONSE_CANDIDATE_ATTR",
    "STIMULUS_RESPONSE_CANDIDATE_PROFILE_ID",
    "STIMULUS_RESPONSE_METADATA_EQUIVALENCE_ATTR",
    "STIMULUS_RESPONSE_METADATA_EQUIVALENCE_SCHEMA_ID",
    "STIMULUS_RESPONSE_METADATA_EQUIVALENCE_SCHEMA_VERSION",
    "STIMULUS_RESPONSE_STORAGE_PLAN_DIGEST_ATTR",
    "STIMULUS_RESPONSE_STORAGE_PLAN_RECEIPT_ATTR",
    "STIMULUS_RESPONSE_STORAGE_PROFILE_ID_ATTR",
    "build_stimulus_response_storage_receipt",
    "consolidate_and_validate_stimulus_response_metadata",
    "create_stimulus_response_arrays_from_receipt",
    "persist_stimulus_response_storage_receipt",
    "stimulus_response_access_unit_semantics",
    "stimulus_response_dimensions",
    "stimulus_response_fill_values",
    "validate_stimulus_response_metadata_equivalence",
    "validate_stimulus_response_storage_receipt",
]
