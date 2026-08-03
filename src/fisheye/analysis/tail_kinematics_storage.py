"""Shared byte-planner adapter for opt-in tail-kinematics candidates."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import zarr

from fisheye.analysis.tail_kinematics_schema import (
    TailKinematicsDimensions,
    build_tail_kinematics_array_declarations,
    infer_tail_kinematics_dimensions,
    tail_kinematics_access_unit_semantics,
    tail_kinematics_array_shapes_and_dtypes,
    tail_kinematics_fill_values,
    validate_tail_kinematics_array_schema,
)
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
ANALYSIS_STORAGE_METADATA_EQUIVALENCE_ATTR = (
    "analysis_storage_direct_consolidated_equivalence"
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


def build_tail_kinematics_storage_receipt(
    dimensions: TailKinematicsDimensions,
    *,
    profile: StorageProfile,
) -> AnalysisStoragePlanReceipt:
    """Plan every exact array using concrete byte facts, never row constants."""

    if not isinstance(profile, StorageProfile):
        raise TypeError("profile must be an explicitly supplied StorageProfile.")
    declarations = build_tail_kinematics_array_declarations(
        include_source_revision_bundle=dimensions.include_source_revision_bundle,
        byte_planner_adopted=True,
    )
    concrete = tail_kinematics_array_shapes_and_dtypes(dimensions)
    semantics = tail_kinematics_access_unit_semantics(
        include_source_revision_bundle=dimensions.include_source_revision_bundle
    )
    facts = {
        path: AnalysisArrayStorageFacts(
            path=path,
            shape=shape,
            dtype=dtype,
            access_unit_semantics=semantics[path],
        )
        for path, (shape, dtype) in concrete.items()
    }
    return plan_analysis_storage(
        declarations,
        facts,
        profile=profile,
        dimensions=dimensions.contract_dimensions,
    )


def create_tail_kinematics_arrays_from_receipt(
    run_group: Any,
    *,
    receipt: AnalysisStoragePlanReceipt,
    dimensions: TailKinematicsDimensions,
) -> None:
    """Create the complete candidate inventory through the shared array factory."""

    expected = build_tail_kinematics_storage_receipt(
        dimensions,
        profile=receipt.profile,
    )
    if receipt.as_manifest() != expected.as_manifest():
        raise ValueError(
            "Tail-kinematics creation receipt differs from executable planning."
        )
    fills = tail_kinematics_fill_values(
        include_source_revision_bundle=dimensions.include_source_revision_bundle
    )
    entries = {entry.declaration.path: entry for entry in receipt.entries}
    if set(entries) != set(fills):
        raise ValueError("Tail-kinematics receipt and fill inventory disagree.")
    for path in sorted(entries):
        parent, leaf = _parent_and_leaf(run_group, path)
        if parent.get(leaf) is not None:
            raise ValueError(f"Tail-kinematics destination {path!r} already exists.")
        entry = entries[path]
        create_array_from_plan(
            parent,
            name=leaf,
            contract=entry.declaration.contract,
            plan=entry.plan,
            fill_value=fills[path],
        )


def persist_tail_kinematics_storage_receipt(
    run_group: Any,
    receipt: AnalysisStoragePlanReceipt,
) -> dict[str, object]:
    manifest = receipt.as_manifest()
    run_group.attrs[ANALYSIS_STORAGE_PLAN_RECEIPT_ATTR] = manifest
    run_group.attrs[ANALYSIS_STORAGE_PLAN_DIGEST_ATTR] = manifest["payload_digest"]
    run_group.attrs[ANALYSIS_STORAGE_PROFILE_ID_ATTR] = receipt.profile.profile_id
    run_group.attrs[ANALYSIS_STORAGE_PROFILE_ROLE_ATTR] = ANALYSIS_STORAGE_PROFILE_ROLE
    return manifest


def _validate_array_physical_metadata(
    run_group: Any,
    receipt: AnalysisStoragePlanReceipt,
    dimensions: TailKinematicsDimensions,
) -> tuple[str, ...]:
    errors: list[str] = []
    declarations = build_tail_kinematics_array_declarations(
        include_source_revision_bundle=dimensions.include_source_revision_bundle,
        byte_planner_adopted=True,
    )
    declaration_by_path = {
        declaration.path: declaration for declaration in declarations
    }
    entry_by_path = {entry.declaration.path: entry for entry in receipt.entries}
    fills = tail_kinematics_fill_values(
        include_source_revision_bundle=dimensions.include_source_revision_bundle
    )
    if set(declaration_by_path) != set(entry_by_path) or set(fills) != set(
        entry_by_path
    ):
        return ("tail-kinematics physical metadata inventory is inconsistent",)
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
            expected = array_metadata_declaration_from_plan(
                contract=declaration_by_path[path].contract,
                plan=entry_by_path[path].plan,
                fill_value=fills[path],
                attributes=nonreserved,
            )
            observed = {
                key: value
                for key, value in raw.items()
                if key not in {"zarr_format", "node_type", "consolidated_metadata"}
            }
            if _normalized_metadata(observed) != _normalized_metadata(expected):
                errors.append(
                    f"{path}: array metadata differs from resolved chunks/shards/codecs"
                )
        except Exception as exc:
            errors.append(f"{path}: physical metadata validation failed: {exc}")
    return tuple(errors)


def validate_tail_kinematics_storage_receipt(run_group: Any) -> tuple[str, ...]:
    """Replan the persisted receipt and validate every live physical declaration."""

    errors: list[str] = []
    persisted = run_group.attrs.get(ANALYSIS_STORAGE_PLAN_RECEIPT_ATTR)
    if not isinstance(persisted, Mapping):
        return ("analysis storage-plan receipt is absent or not an object",)
    try:
        parsed = analysis_storage_plan_receipt_from_manifest(persisted)
    except Exception as exc:
        return (f"analysis storage-plan receipt is invalid: {exc}",)
    try:
        dimensions = infer_tail_kinematics_dimensions(run_group)
        expected = build_tail_kinematics_storage_receipt(
            dimensions,
            profile=parsed.profile,
        )
    except Exception as exc:
        return (f"analysis storage plan could not be recomputed: {exc}",)
    if dict(persisted) != expected.as_manifest():
        errors.append("analysis storage-plan receipt differs from executable planning")
    digest = persisted.get("payload_digest")
    if run_group.attrs.get(ANALYSIS_STORAGE_PLAN_DIGEST_ATTR) != digest:
        errors.append("analysis storage-plan redundant digest binding mismatch")
    if (
        run_group.attrs.get(ANALYSIS_STORAGE_PROFILE_ID_ATTR)
        != parsed.profile.profile_id
    ):
        errors.append("analysis storage profile identity binding mismatch")
    if (
        run_group.attrs.get(ANALYSIS_STORAGE_PROFILE_ROLE_ATTR)
        != ANALYSIS_STORAGE_PROFILE_ROLE
    ):
        errors.append("analysis storage profile role is not the candidate role")
    errors.extend(
        validate_tail_kinematics_array_schema(
            run_group,
            byte_planner_adopted=True,
        )
    )
    errors.extend(_validate_array_physical_metadata(run_group, expected, dimensions))
    return tuple(errors)


def consolidate_and_validate_tail_kinematics_metadata(
    root: Any,
    *,
    run_path: str,
) -> int:
    """Publish one metadata generation and compare every direct/inline array."""

    if str(getattr(root, "path", "")).strip("/"):
        raise ValueError("Metadata consolidation requires the archive root group.")
    zarr.consolidate_metadata(root.store)
    direct_root = zarr.open_group(
        root.store,
        mode="r",
        use_consolidated=False,
    )
    consolidated_root = zarr.open_group(
        root.store,
        mode="r",
        use_consolidated=True,
    )
    direct_run = direct_root[run_path]
    consolidated_run = consolidated_root[run_path]
    dimensions = infer_tail_kinematics_dimensions(direct_run)
    declarations = build_tail_kinematics_array_declarations(
        include_source_revision_bundle=dimensions.include_source_revision_bundle,
        byte_planner_adopted=True,
    )
    for declaration in declarations:
        direct = _array_at_path(direct_run, declaration.path).metadata.to_dict()
        consolidated = _array_at_path(
            consolidated_run, declaration.path
        ).metadata.to_dict()
        if _normalized_metadata(direct) != _normalized_metadata(consolidated):
            raise ValueError(
                f"{declaration.path}: direct/consolidated metadata differ."
            )
    return len(declarations)


__all__ = [
    "ANALYSIS_STORAGE_METADATA_EQUIVALENCE_ATTR",
    "ANALYSIS_STORAGE_PLAN_DIGEST_ATTR",
    "ANALYSIS_STORAGE_PLAN_RECEIPT_ATTR",
    "ANALYSIS_STORAGE_PROFILE_ID_ATTR",
    "ANALYSIS_STORAGE_PROFILE_ROLE",
    "ANALYSIS_STORAGE_PROFILE_ROLE_ATTR",
    "build_tail_kinematics_storage_receipt",
    "consolidate_and_validate_tail_kinematics_metadata",
    "create_tail_kinematics_arrays_from_receipt",
    "persist_tail_kinematics_storage_receipt",
    "validate_tail_kinematics_storage_receipt",
]
