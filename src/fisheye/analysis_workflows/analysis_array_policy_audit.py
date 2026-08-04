"""Executable logical and physical policy audit for analytics arrays.

Family suite validators prove that an execution suite contains the live array
inventory for one maintained analysis family.  This module supplies the common
second half of that gate: every observed array must carry an exact logical
declaration, enter the shared byte planner, resolve through a registered Zarr
v3 codec profile, preserve complete access units inside independently readable
inner chunks, and expose effective chunk, shard, and object estimates.

The audit is evidence only.  It does not create arrays, select a production
profile, activate a selector, or register a candidate.
"""

from __future__ import annotations

from collections import Counter
from math import prod
from typing import Any, Mapping

from fisheye.shared.zarr.analysis_benchmark_suite import (
    ANALYSIS_BENCHMARK_SUITE_SCHEMA_VERSION,
    require_analysis_benchmark_suite_manifest,
)
from fisheye.shared.zarr.analysis_storage_planning import (
    AnalysisArrayStoragePlanReceipt,
    analysis_storage_plan_receipt_from_manifest,
)
from fisheye.shared.zarr.benchmark_contracts import FULL_SCAN_READ_V1
from fisheye.shared.zarr.codec_profiles import (
    ZSTD_FAST_V1,
    CodecProfile,
    get_codec_profile,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.storage_intent import (
    STORAGE_POLICY_VERSION,
    AccessPattern,
    WriteMode,
)


ANALYSIS_ARRAY_POLICY_AUDIT_SCHEMA_ID = "palette.analysis_array_policy_audit"
ANALYSIS_ARRAY_POLICY_AUDIT_SCHEMA_VERSION = 1


def _exact_optional_semantics(value: object, *, label: str) -> str | None:
    """Require either one exact semantic label or explicit not-applicable null."""

    if value is None:
        return None
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{label} must be null or one nonempty exact string")
    return value


def _codec_policy(codec: CodecProfile) -> dict[str, object]:
    if codec != ZSTD_FAST_V1:
        raise ValueError("analytics codec profile differs from the pinned Zarr-v3 policy")
    return codec.as_manifest()


def _array_policy_record(
    entry: AnalysisArrayStoragePlanReceipt,
) -> dict[str, object]:
    declaration = entry.declaration
    contract = declaration.contract
    facts = entry.facts
    plan = entry.plan

    if declaration.byte_planner_adopted is not True:
        raise ValueError(f"{declaration.path}: shared byte planner is not adopted")
    if contract.dtype.variable_length or contract.dtype.numpy_dtype is None:
        raise ValueError(
            f"{declaration.path}: analytics candidates require one exact fixed-width dtype"
        )
    if (
        type(contract.description) is not str
        or not contract.description
        or contract.description != contract.description.strip()
    ):
        raise ValueError(f"{declaration.path}: description must be one exact string")
    units = _exact_optional_semantics(
        contract.units,
        label=f"{declaration.path} units",
    )
    coordinate_space = _exact_optional_semantics(
        contract.coordinate_space,
        label=f"{declaration.path} coordinate_space",
    )
    if len(contract.shape_template) != len(contract.axis_names):
        raise ValueError(f"{declaration.path}: shape and axis ranks differ")
    if tuple(facts.shape) != tuple(plan.logical_shape):
        raise ValueError(f"{declaration.path}: observed and planned shapes differ")
    if plan.policy_version != STORAGE_POLICY_VERSION:
        raise ValueError(f"{declaration.path}: storage policy version differs")
    if plan.chunk_shape is None:
        raise ValueError(
            f"{declaration.path}: analytics array policy does not support scalar payloads"
        )

    chunk_shape = tuple(plan.chunk_shape)
    if len(chunk_shape) != len(facts.shape) or any(value <= 0 for value in chunk_shape):
        raise ValueError(f"{declaration.path}: effective chunk shape is invalid")
    if any(
        chunk % unit != 0
        for chunk, unit in zip(chunk_shape, facts.access_unit_shape, strict=True)
    ):
        raise ValueError(
            f"{declaration.path}: inner chunk splits one declared access unit"
        )
    expected_chunk_nbytes = int(facts.dtype.itemsize) * prod(chunk_shape)
    if plan.chunk_nbytes != expected_chunk_nbytes:
        raise ValueError(f"{declaration.path}: effective chunk bytes differ")

    shard_shape = None if plan.shard_shape is None else tuple(plan.shard_shape)
    independently_readable_inner_chunks = True
    if shard_shape is not None:
        if len(shard_shape) != len(chunk_shape) or any(
            shard % chunk != 0
            for shard, chunk in zip(shard_shape, chunk_shape, strict=True)
        ):
            raise ValueError(f"{declaration.path}: shard does not contain whole chunks")
        inner_chunks_per_shard = prod(
            shard // chunk
            for shard, chunk in zip(shard_shape, chunk_shape, strict=True)
        )
        if inner_chunks_per_shard <= 1:
            raise ValueError(f"{declaration.path}: indexed shard contains one inner chunk")
        if declaration.write_mode is WriteMode.RANDOM_UPDATE:
            raise ValueError(f"{declaration.path}: editable array cannot use indexed shards")
        if plan.write_ownership != "whole_shard_single_writer":
            raise ValueError(f"{declaration.path}: sharded writer lacks whole-shard ownership")
        expected_shard_nbytes = expected_chunk_nbytes * inner_chunks_per_shard
        if plan.shard_nbytes != expected_shard_nbytes:
            raise ValueError(f"{declaration.path}: effective shard bytes differ")
    elif plan.shard_nbytes is not None:
        raise ValueError(f"{declaration.path}: unsharded plan records shard bytes")

    return {
        "path": declaration.path,
        "required": declaration.required,
        "logical_schema_id": contract.schema_id,
        "logical_schema_version": contract.schema_version,
        "dtype": contract.dtype.as_manifest(),
        "shape_template": list(contract.shape_template),
        "axis_names": list(contract.axis_names),
        "units": units,
        "coordinate_space": coordinate_space,
        "null_semantics": declaration.null_semantics,
        "fill_semantics": declaration.fill_semantics,
        "access_pattern": declaration.access_pattern.value,
        "write_mode": declaration.write_mode.value,
        "authority_role": declaration.authority_role.value,
        "physical_policy_owner": declaration.physical_policy_owner,
        "byte_planner_adopted": True,
        "access_unit_shape": list(facts.access_unit_shape),
        "access_unit_semantics": facts.access_unit_semantics,
        "effective_chunk_shape": list(chunk_shape),
        "effective_shard_shape": (
            None if shard_shape is None else list(shard_shape)
        ),
        "chunk_nbytes": plan.chunk_nbytes,
        "shard_nbytes": plan.shard_nbytes,
        "inner_chunk_count": plan.estimated_chunk_count,
        "payload_object_estimate": plan.estimated_payload_objects,
        "object_budget_satisfied": plan.object_budget_satisfied,
        "shard_byte_budget_satisfied": plan.shard_byte_budget_satisfied,
        "write_ownership": plan.write_ownership,
        "independently_readable_inner_chunks": independently_readable_inner_chunks,
    }


def _full_scan_paths(benchmark_suite: Mapping[str, Any]) -> frozenset[str]:
    cases = benchmark_suite["payload"]["array_cases"]
    result: set[str] = set()
    for record in cases:
        if not isinstance(record, Mapping):
            raise ValueError("analysis benchmark suite contains a non-object case")
        case = record.get("case")
        if not isinstance(case, Mapping):
            raise ValueError("analysis benchmark suite case is not an object")
        workload = case.get("workload")
        if not isinstance(workload, Mapping):
            raise ValueError("analysis benchmark suite workload is not an object")
        if workload.get("workload_id") == FULL_SCAN_READ_V1.workload_id:
            path = record.get("array_path")
            if type(path) is not str or path in result:
                raise ValueError("full-scan workload path is invalid or duplicated")
            result.add(path)
    return frozenset(result)


def _analysis_array_policy_payload(
    stage_id: str,
    benchmark_suite: Mapping[str, Any],
) -> dict[str, object]:
    if type(stage_id) is not str or not stage_id:
        raise ValueError("stage_id must be one exact nonempty string")
    require_analysis_benchmark_suite_manifest(benchmark_suite, require_current=True)
    suite_payload = benchmark_suite["payload"]
    if suite_payload["family_id"] != stage_id:
        raise ValueError("analysis array policy stage differs from suite family")
    receipt_manifest = suite_payload["storage_plan_receipt"]
    receipt = analysis_storage_plan_receipt_from_manifest(receipt_manifest)
    if not receipt.entries:
        raise ValueError("analysis array policy requires one nonempty array inventory")
    codec = get_codec_profile(receipt.profile.codec_profile_id)
    codec_manifest = _codec_policy(codec)
    arrays = [_array_policy_record(entry) for entry in receipt.entries]
    array_paths = frozenset(str(record["path"]) for record in arrays)
    full_scan_paths = _full_scan_paths(benchmark_suite)
    if full_scan_paths != array_paths:
        raise ValueError(
            "complete-scan benchmark coverage differs from the planned array inventory"
        )

    access_counts = Counter(str(record["access_pattern"]) for record in arrays)
    write_counts = Counter(str(record["write_mode"]) for record in arrays)
    authority_counts = Counter(str(record["authority_role"]) for record in arrays)
    return {
        "stage_id": stage_id,
        "family_id": suite_payload["family_id"],
        "benchmark_suite_schema_version": ANALYSIS_BENCHMARK_SUITE_SCHEMA_VERSION,
        "benchmark_suite_payload_digest": benchmark_suite["payload_digest"],
        "storage_plan_payload_digest": receipt_manifest["payload_digest"],
        "storage_profile": receipt.profile.as_manifest(),
        "codec_profile": codec_manifest,
        "policy_scope": {
            "zarr_array_access_classes": [item.value for item in AccessPattern],
            "complete_scan_workload": "one_required_full_scan_case_per_array",
            "artifact_byte_stream_scope": (
                "non_zarr_artifacts_are_classified_outside_array_intent"
            ),
            "independently_readable_inner_chunks_inside_shards": True,
            "production_profile_promoted": False,
            "selector_or_registry_mutation_authorized": False,
        },
        "summary": {
            "array_count": len(arrays),
            "required_array_count": sum(bool(record["required"]) for record in arrays),
            "optional_array_count": sum(
                not bool(record["required"]) for record in arrays
            ),
            "fixed_width_array_count": len(arrays),
            "byte_planner_adopted_count": len(arrays),
            "sharded_array_count": sum(
                record["effective_shard_shape"] is not None for record in arrays
            ),
            "inner_chunk_count": sum(
                int(record["inner_chunk_count"]) for record in arrays
            ),
            "payload_object_estimate": sum(
                int(record["payload_object_estimate"]) for record in arrays
            ),
            "access_pattern_counts": dict(sorted(access_counts.items())),
            "write_mode_counts": dict(sorted(write_counts.items())),
            "authority_role_counts": dict(sorted(authority_counts.items())),
            "complete_scan_case_count": len(full_scan_paths),
        },
        "arrays": arrays,
    }


def build_analysis_array_policy_audit(
    stage_id: str,
    benchmark_suite: Mapping[str, Any],
) -> dict[str, object]:
    """Audit one family-bound suite after its live schema validator succeeds."""

    payload = _analysis_array_policy_payload(stage_id, benchmark_suite)
    result = {
        "schema_id": ANALYSIS_ARRAY_POLICY_AUDIT_SCHEMA_ID,
        "schema_version": ANALYSIS_ARRAY_POLICY_AUDIT_SCHEMA_VERSION,
        "payload": payload,
        "payload_digest": canonical_json_sha256(payload),
    }
    require_analysis_array_policy_audit(
        result,
        stage_id=stage_id,
        benchmark_suite=benchmark_suite,
    )
    return result


def require_analysis_array_policy_audit(
    value: Mapping[str, Any],
    *,
    stage_id: str,
    benchmark_suite: Mapping[str, Any],
) -> None:
    """Rebuild and compare one persisted policy audit against its exact suite."""

    if not isinstance(value, Mapping) or set(value) != {
        "schema_id",
        "schema_version",
        "payload",
        "payload_digest",
    }:
        raise ValueError("analysis array policy audit envelope field set differs")
    if (
        value["schema_id"] != ANALYSIS_ARRAY_POLICY_AUDIT_SCHEMA_ID
        or value["schema_version"] != ANALYSIS_ARRAY_POLICY_AUDIT_SCHEMA_VERSION
    ):
        raise ValueError("analysis array policy audit schema identity differs")
    payload = value["payload"]
    if not isinstance(payload, Mapping) or value["payload_digest"] != canonical_json_sha256(
        payload
    ):
        raise ValueError("analysis array policy audit payload digest differs")
    if payload.get("stage_id") != stage_id:
        raise ValueError("analysis array policy audit stage identity differs")

    expected_payload = _analysis_array_policy_payload(stage_id, benchmark_suite)
    if dict(payload) != expected_payload:
        raise ValueError("analysis array policy audit differs from executable evidence")


def require_analysis_array_policy_suite(
    stage_id: str,
    benchmark_suite: Mapping[str, Any],
) -> None:
    """Require common array-policy completeness without persisting an audit."""

    build_analysis_array_policy_audit(stage_id, benchmark_suite)


__all__ = [
    "ANALYSIS_ARRAY_POLICY_AUDIT_SCHEMA_ID",
    "ANALYSIS_ARRAY_POLICY_AUDIT_SCHEMA_VERSION",
    "build_analysis_array_policy_audit",
    "require_analysis_array_policy_audit",
    "require_analysis_array_policy_suite",
]
