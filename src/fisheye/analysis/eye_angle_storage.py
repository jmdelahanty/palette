"""Opt-in byte-planned physical storage for compact-v7 eye-angle runs.

This module deliberately owns a candidate profile rather than registering or
promoting a repository-wide profile.  The established eye-angle writer remains
the default.  Callers must explicitly request this candidate, and candidate
runs remain selector-ineligible until a later benchmark/promotion decision.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping

import numpy as np

from fisheye.analysis.eye_angle_schema import (
    EyeAngleDimensions,
    build_eye_angle_array_declarations,
    canonical_exact_json_bytes,
    collect_eye_angle_arrays,
)
from fisheye.shared.zarr.analysis_storage_planning import (
    AnalysisArrayStorageFacts,
    AnalysisArrayStoragePlanReceipt,
    AnalysisStoragePlanReceipt,
    plan_analysis_storage,
)
from fisheye.shared.zarr.array_factory import (
    create_array_from_plan,
    validate_array_metadata_declaration_from_plan,
)
from fisheye.shared.zarr.storage_intent import AccessPattern
from fisheye.shared.zarr.storage_profiles import KIB, MIB, StorageProfile

EYE_ANGLE_LEGACY_EXPLICIT_STORAGE = "legacy_explicit_chunks"
EYE_ANGLE_ACCESS_AWARE_CANDIDATE_PROFILE_ID = "eye_angle_access_aware_candidate_v1"
EYE_ANGLE_STORAGE_PROFILE_CHOICES = (
    EYE_ANGLE_LEGACY_EXPLICIT_STORAGE,
    EYE_ANGLE_ACCESS_AWARE_CANDIDATE_PROFILE_ID,
)
EYE_ANGLE_STORAGE_PLAN_ATTR = "eye_angle_storage_plan"
EYE_ANGLE_STORAGE_CANDIDATE_ATTR = "eye_angle_storage_candidate"

EYE_ANGLE_NAN_FILL_PATHS = frozenset(
    {
        "roi_angles",
        "frame_angles",
        "roi_vectors",
        "support/ellipse_major",
        "support/ellipse_minor",
        "support/ellipse_ratio",
        "support/body_frame/origin_xy",
        "support/body_frame/forward_axis_xy",
        "support/body_frame/left_axis_xy",
        "support/body_frame/heading_deg",
    }
)
EYE_ANGLE_FALSE_FILL_PATHS = frozenset(
    {
        "angle_channel_index/roi_available",
        "angle_channel_index/frame_available",
        "vector_channel_index/roi_available",
        "vector_channel_index/frame_available",
        "qa_channel_index/roi_available",
        "qa_channel_index/frame_available",
        "support/body_frame/valid",
    }
)
EYE_ANGLE_ZERO_FILL_PATHS = frozenset(
    {
        "roi_qa",
        "frame_qa",
        "angle_channel_index/name",
        "angle_channel_index/representation",
        "angle_channel_index/eye",
        "angle_channel_index/value_kind",
        "angle_channel_index/units",
        "angle_channel_index/source_channel",
        "angle_channel_index/formula",
        "angle_channel_index/compatibility_alias_of",
        "vector_channel_index/name",
        "vector_channel_index/representation",
        "vector_channel_index/eye",
        "vector_channel_index/value_kind",
        "vector_channel_index/units",
        "qa_channel_index/name",
        "qa_channel_index/value_kind",
        "qa_channel_index/dtype",
        "support/instance_key",
        "support/source_acquisition_frame_index",
        "support/frame_indices",
        "support/time_seconds",
        "support/frame_time_seconds",
        "support/body_frame/failure_reason_bytes",
    }
)


# This profile is intentionally local to the eye-angle writer.  It is not a
# registered or promoted production profile.  Complete logical rows remain
# intact inside approximately 1 MiB independently decodable inner chunks;
# immutable arrays large enough to benefit are packed into approximately
# 32 MiB indexed shards.  Small eager semantic tables stay one regular object.
EYE_ANGLE_ACCESS_AWARE_CANDIDATE_V1 = StorageProfile(
    profile_id=EYE_ANGLE_ACCESS_AWARE_CANDIDATE_PROFILE_ID,
    target_chunk_bytes=1 * MIB,
    min_chunk_bytes=512 * KIB,
    max_chunk_bytes=2 * MIB,
    eager_max_bytes=8 * MIB,
    target_shard_bytes=32 * MIB,
    per_row_target_shard_bytes=32 * MIB,
    max_shard_bytes=64 * MIB,
    max_payload_objects=4_096,
    codec_profile_id="zstd_fast_v1",
    shard_immutable=True,
    shard_owned_appends=True,
    target_chunk_bytes_by_access=((AccessPattern.EAGER, 1 * MIB),),
)


@dataclass(frozen=True)
class EyeAngleStorageIssue:
    code: str
    path: str
    message: str


def is_eye_angle_storage_candidate(profile_id: str) -> bool:
    return profile_id == EYE_ANGLE_ACCESS_AWARE_CANDIDATE_PROFILE_ID


def _resolved_shape(
    shape_template: tuple[str | int, ...], dimensions: Mapping[str, int]
) -> tuple[int, ...]:
    return tuple(
        int(dimensions[value]) if isinstance(value, str) else int(value)
        for value in shape_template
    )


def build_eye_angle_candidate_storage_plan(
    dimensions: EyeAngleDimensions,
) -> AnalysisStoragePlanReceipt:
    """Derive the exact 41-array candidate plan from bytes, not row literals."""

    declarations = build_eye_angle_array_declarations(byte_planner_adopted=True)
    declared_paths = {declaration.path for declaration in declarations}
    fill_paths = (
        EYE_ANGLE_NAN_FILL_PATHS
        | EYE_ANGLE_FALSE_FILL_PATHS
        | EYE_ANGLE_ZERO_FILL_PATHS
    )
    if fill_paths != declared_paths:
        raise RuntimeError(
            "Eye-angle physical fill inventory differs from the exact 41-array "
            f"schema: missing={sorted(declared_paths - fill_paths)!r}, "
            f"unexpected={sorted(fill_paths - declared_paths)!r}."
        )
    if (
        EYE_ANGLE_NAN_FILL_PATHS & EYE_ANGLE_FALSE_FILL_PATHS
        or EYE_ANGLE_NAN_FILL_PATHS & EYE_ANGLE_ZERO_FILL_PATHS
        or EYE_ANGLE_FALSE_FILL_PATHS & EYE_ANGLE_ZERO_FILL_PATHS
    ):
        raise RuntimeError("Eye-angle physical fill classes overlap.")
    resolved = dimensions.contract_dimensions
    facts = {
        declaration.path: AnalysisArrayStorageFacts(
            path=declaration.path,
            shape=_resolved_shape(declaration.contract.shape_template, resolved),
            dtype=np.dtype(declaration.contract.dtype.numpy_dtype),
            access_unit_semantics=(
                "one complete logical record on the declared growth axis; all "
                "fixed trailing semantic axes remain indivisible"
            ),
        )
        for declaration in declarations
    }
    return plan_analysis_storage(
        declarations,
        facts,
        profile=EYE_ANGLE_ACCESS_AWARE_CANDIDATE_V1,
        dimensions=resolved,
    )


def eye_angle_storage_entries_by_path(
    receipt: AnalysisStoragePlanReceipt | None,
) -> dict[str, AnalysisArrayStoragePlanReceipt]:
    if receipt is None:
        return {}
    return {entry.declaration.path: entry for entry in receipt.entries}


def eye_angle_planned_fill_value(entry: AnalysisArrayStoragePlanReceipt) -> Any:
    """Return the exact frozen semantic fill for one maintained array."""

    path = entry.declaration.path
    dtype = np.dtype(entry.facts.dtype)
    if path in EYE_ANGLE_NAN_FILL_PATHS:
        if dtype.kind != "f":
            raise ValueError(f"{path}: NaN fill requires a floating dtype.")
        return float("nan")
    if path in EYE_ANGLE_FALSE_FILL_PATHS:
        if dtype != np.dtype(bool):
            raise ValueError(f"{path}: false fill requires the bool dtype.")
        return False
    if path in EYE_ANGLE_ZERO_FILL_PATHS:
        if dtype == np.dtype(bool):
            raise ValueError(f"{path}: bool arrays must use the false fill class.")
        return 0
    raise ValueError(f"{path}: no exact eye-angle physical fill is declared.")


def _metadata_json_copy(array: Any) -> dict[str, Any]:
    return json.loads(json.dumps(array.metadata.to_dict()))


def _is_zarr_v3_nan_fill(value: Any) -> bool:
    return (type(value) is float and np.isnan(value)) or value == "NaN"


def _normalize_nan_fill_for_comparison(
    declaration: Mapping[str, Any],
    *,
    path: str,
) -> dict[str, Any]:
    normalized = dict(declaration)
    if path in EYE_ANGLE_NAN_FILL_PATHS and _is_zarr_v3_nan_fill(
        normalized.get("fill_value")
    ):
        normalized["fill_value"] = "NaN"
    return normalized


def create_eye_angle_array_from_entry(
    group: Any,
    *,
    name: str,
    entry: AnalysisArrayStoragePlanReceipt,
    data: np.ndarray | None = None,
) -> Any:
    """Create and optionally fill one candidate array from its exact plan."""

    array = create_array_from_plan(
        group,
        name=name,
        contract=entry.declaration.contract,
        plan=entry.plan,
        fill_value=eye_angle_planned_fill_value(entry),
    )
    if data is not None:
        observed = np.asarray(data)
        errors = entry.declaration.contract.validate_observation(
            observed,
            dimensions=dict(entry.resolved_dimensions),
        )
        if errors or tuple(observed.shape) != entry.plan.logical_shape:
            raise ValueError(
                f"{entry.declaration.path}: planned write data mismatch: "
                + "; ".join(errors or ("shape differs from resolved plan",))
            )
        if observed.size:
            array[...] = observed
    return array


def validate_eye_angle_candidate_storage(
    run_group: Any,
    *,
    dimensions: EyeAngleDimensions,
) -> tuple[EyeAngleStorageIssue, ...]:
    """Recompute the receipt and validate direct Zarr metadata for all arrays."""

    issues: list[EyeAngleStorageIssue] = []
    expected = build_eye_angle_candidate_storage_plan(dimensions)
    expected_manifest = expected.as_manifest()
    persisted = run_group.attrs.get(EYE_ANGLE_STORAGE_PLAN_ATTR)
    try:
        receipt_matches = canonical_exact_json_bytes(
            persisted,
            path=f"$.{EYE_ANGLE_STORAGE_PLAN_ATTR}",
        ) == canonical_exact_json_bytes(expected_manifest)
    except (TypeError, ValueError):
        receipt_matches = False
    if not receipt_matches:
        issues.append(
            EyeAngleStorageIssue(
                "storage_plan_receipt_mismatch",
                EYE_ANGLE_STORAGE_PLAN_ATTR,
                "Persisted receipt must exactly equal the recomputed 41-array plan.",
            )
        )

    arrays = collect_eye_angle_arrays(run_group)
    entries = eye_angle_storage_entries_by_path(expected)
    for path in sorted(entries):
        array = arrays.get(path)
        if array is None:
            issues.append(
                EyeAngleStorageIssue(
                    "storage_array_missing",
                    path,
                    "Planned eye-angle array is missing.",
                )
            )
            continue
        entry = entries[path]
        try:
            declaration = _metadata_json_copy(array)
        except (AttributeError, TypeError, ValueError) as exc:
            issues.append(
                EyeAngleStorageIssue(
                    "storage_metadata_unreadable",
                    path,
                    str(exc),
                )
            )
            continue
        expected_fill = eye_angle_planned_fill_value(entry)
        declaration_for_plan = dict(declaration)
        validation_fill = expected_fill
        if path in EYE_ANGLE_NAN_FILL_PATHS:
            if not _is_zarr_v3_nan_fill(declaration.get("fill_value")):
                issues.append(
                    EyeAngleStorageIssue(
                        "storage_fill_value_mismatch",
                        path,
                        "Physical fill_value must be the Zarr-v3 NaN representation.",
                    )
                )
            # The shared metadata comparator uses Python equality, where NaN
            # is intentionally unequal to itself. Validate the semantic NaN
            # above, then substitute one inert sentinel on both sides solely
            # for comparison of every other declaration field.
            declaration_for_plan["fill_value"] = 0
            validation_fill = 0
        errors = validate_array_metadata_declaration_from_plan(
            declaration_for_plan,
            contract=entry.declaration.contract,
            plan=entry.plan,
            fill_value=validation_fill,
        )
        issues.extend(
            EyeAngleStorageIssue("storage_metadata_mismatch", path, error)
            for error in errors
        )
    return tuple(issues)


def validate_eye_angle_direct_consolidated_storage(
    direct_run: Any,
    consolidated_run: Any,
    *,
    dimensions: EyeAngleDimensions,
) -> tuple[EyeAngleStorageIssue, ...]:
    """Validate both metadata views and their exact array declarations.

    The active writer intentionally does not consolidate a mutable recording
    root.  An atomic publisher or benchmark may call this after root metadata
    consolidation and before considering a candidate for visibility.
    """

    issues = [
        *validate_eye_angle_candidate_storage(
            direct_run,
            dimensions=dimensions,
        ),
        *validate_eye_angle_candidate_storage(
            consolidated_run,
            dimensions=dimensions,
        ),
    ]
    direct_arrays = collect_eye_angle_arrays(direct_run)
    consolidated_arrays = collect_eye_angle_arrays(consolidated_run)
    if set(direct_arrays) != set(consolidated_arrays):
        issues.append(
            EyeAngleStorageIssue(
                "direct_consolidated_path_mismatch",
                "arrays",
                "Direct and consolidated array path inventories differ.",
            )
        )
        return tuple(issues)
    for path in sorted(direct_arrays):
        direct = _normalize_nan_fill_for_comparison(
            _metadata_json_copy(direct_arrays[path]),
            path=path,
        )
        consolidated = _normalize_nan_fill_for_comparison(
            _metadata_json_copy(consolidated_arrays[path]),
            path=path,
        )
        if direct != consolidated:
            issues.append(
                EyeAngleStorageIssue(
                    "direct_consolidated_metadata_mismatch",
                    path,
                    "Direct and consolidated array declarations differ.",
                )
            )
    return tuple(issues)


__all__ = [
    "EYE_ANGLE_ACCESS_AWARE_CANDIDATE_PROFILE_ID",
    "EYE_ANGLE_ACCESS_AWARE_CANDIDATE_V1",
    "EYE_ANGLE_LEGACY_EXPLICIT_STORAGE",
    "EYE_ANGLE_FALSE_FILL_PATHS",
    "EYE_ANGLE_NAN_FILL_PATHS",
    "EYE_ANGLE_STORAGE_CANDIDATE_ATTR",
    "EYE_ANGLE_STORAGE_PLAN_ATTR",
    "EYE_ANGLE_STORAGE_PROFILE_CHOICES",
    "EYE_ANGLE_ZERO_FILL_PATHS",
    "EyeAngleStorageIssue",
    "build_eye_angle_candidate_storage_plan",
    "create_eye_angle_array_from_entry",
    "eye_angle_storage_entries_by_path",
    "is_eye_angle_storage_candidate",
    "validate_eye_angle_candidate_storage",
    "validate_eye_angle_direct_consolidated_storage",
]
