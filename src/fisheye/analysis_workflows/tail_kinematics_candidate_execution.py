"""Typed suite and identity helpers for tail-kinematics candidates.

This module is deliberately family-owned.  It reconstructs the exact 21-array
core (plus the atomic two-array source-revision bundle when present), binds the
live canonical subject-shape authority used by the scientific computation, and
defines the closed invocation payload that the shared invocation catalog can
adopt without changing this family's execution semantics.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping

import numpy as np

from fisheye.analysis.tail_kinematics_schema import (
    TAIL_KINEMATICS_ARRAY_SCHEMA_ATTR,
    TailKinematicsDimensions,
    build_tail_kinematics_array_declarations,
    infer_tail_kinematics_dimensions,
    validate_tail_kinematics_array_schema,
)
from fisheye.analysis.tail_kinematics_storage import (
    build_tail_kinematics_storage_receipt,
)
from fisheye.analysis_workflows.analysis_candidate_invocation import (
    ANALYSIS_CANDIDATE_INVOCATION_SCHEMA_ID,
    ANALYSIS_CANDIDATE_INVOCATION_SCHEMA_VERSION,
    CandidateInvocationContract,
)
from fisheye.shared.coordinate_frame_record import array_values_sha256
from fisheye.shared.zarr.analysis_benchmark_suite import (
    AnalysisBenchmarkScale,
    build_analysis_benchmark_suite,
    require_analysis_benchmark_suite_manifest,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.storage_profiles import get_storage_profile

TAIL_KINEMATICS_EXECUTION_FAMILY_ID = "tail_kinematics"
TAIL_KINEMATICS_CORE_ARRAY_COUNT = 21
TAIL_KINEMATICS_REVISION_BUNDLE_ARRAY_COUNT = 2
TAIL_KINEMATICS_EXECUTION_PROFILE_ID = "published_http_v1"
TAIL_KINEMATICS_LOGICAL_EQUALITY_CONTRACT = "tail_kinematics_declared_arrays_v1"
TAIL_KINEMATICS_COORDINATE_VALIDATOR_REF = (
    "fisheye.analysis_workflows.tail_kinematics_candidate_execution:"
    "build_tail_kinematics_coordinate_evidence"
)
TAIL_KINEMATICS_SOURCE_STAGING_MODE = "canonical_subject_shape_physical_subset_v1"
TAIL_KINEMATICS_REVISION_BUNDLE_MODE = "atomic_source_mirror_v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_COPY_BACKENDS = frozenset({"python", "rsync"})

TAIL_KINEMATICS_INVOCATION_FIELDS = frozenset(
    {
        "source_subject_shape_run",
        "source_tail_coordinate_manifest_sha256",
        "source_subject_shape_manifest_sha256",
        "source_logical_schema_mode",
        "tail_angle_sample_count",
        "block_rows",
        "output_shard_rows",
        "execution_backend",
        "num_workers",
        "source_staging_mode",
        "source_revision_bundle_mode",
        "storage_profile_id",
        "copy_backend",
        "keep_scratch",
        "check_capacity",
    }
)


def _attrs(group: Any) -> dict[str, Any]:
    attrs = group.attrs
    return dict(attrs.asdict() if hasattr(attrs, "asdict") else dict(attrs))


def _array_at_path(group: Any, path: str) -> Any:
    node = group
    for component in path.split("/"):
        node = node[component]
    return node


def _require_sha256(value: object, *, label: str) -> str:
    if type(value) is not str or not _SHA256.fullmatch(value):
        raise ValueError(f"{label} must be one lowercase SHA-256")
    return value


def _require_positive_int(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be one positive exact integer")
    return value


def _require_run_name(value: object, *, label: str) -> str:
    if type(value) is not str or not _RUN_NAME.fullmatch(value):
        raise ValueError(f"{label} must be one exact run name")
    return value


def require_tail_kinematics_invocation_parameters(
    parameters: object,
) -> Mapping[str, Any]:
    """Validate the exact future shared ``tail_kinematics_v1`` grammar."""

    if not isinstance(parameters, Mapping) or set(parameters) != set(
        TAIL_KINEMATICS_INVOCATION_FIELDS
    ):
        raise ValueError("tail-kinematics invocation parameter field set differs")
    _require_run_name(
        parameters["source_subject_shape_run"],
        label="source_subject_shape_run",
    )
    _require_sha256(
        parameters["source_tail_coordinate_manifest_sha256"],
        label="source_tail_coordinate_manifest_sha256",
    )
    _require_sha256(
        parameters["source_subject_shape_manifest_sha256"],
        label="source_subject_shape_manifest_sha256",
    )
    for field in (
        "tail_angle_sample_count",
        "block_rows",
        "output_shard_rows",
        "num_workers",
    ):
        _require_positive_int(parameters[field], label=field)
    if parameters["tail_angle_sample_count"] < 2:
        raise ValueError("tail_angle_sample_count must be at least two")
    if parameters["execution_backend"] != "serial":
        raise ValueError("tail-kinematics candidate execution_backend must be serial")
    if parameters["num_workers"] != 1:
        raise ValueError("tail-kinematics candidate num_workers must equal one")
    if parameters["source_staging_mode"] != TAIL_KINEMATICS_SOURCE_STAGING_MODE:
        raise ValueError("tail-kinematics source_staging_mode differs")
    if (
        parameters["source_revision_bundle_mode"]
        != TAIL_KINEMATICS_REVISION_BUNDLE_MODE
    ):
        raise ValueError("tail-kinematics source_revision_bundle_mode differs")
    if parameters["source_logical_schema_mode"] != (
        "exact_arrays_legacy_receipt_optional_v1"
    ):
        raise ValueError("tail-kinematics source_logical_schema_mode differs")
    if parameters["storage_profile_id"] != TAIL_KINEMATICS_EXECUTION_PROFILE_ID:
        raise ValueError("tail-kinematics storage_profile_id differs")
    if parameters["copy_backend"] not in _COPY_BACKENDS:
        raise ValueError("tail-kinematics copy_backend must be python or rsync")
    for field in ("keep_scratch", "check_capacity"):
        if type(parameters[field]) is not bool:
            raise TypeError(f"{field} must be an exact bool")
    return parameters


def build_tail_kinematics_invocation(
    *,
    source_subject_shape_run: str,
    source_tail_coordinate_manifest_sha256: str,
    source_subject_shape_manifest_sha256: str,
    tail_angle_sample_count: int,
    block_rows: int,
    output_shard_rows: int,
    storage_profile_id: str,
    copy_backend: str,
    keep_scratch: bool,
    check_capacity: bool,
) -> dict[str, object]:
    """Build the family-owned invocation pending shared-catalog activation."""

    parameters = {
        "source_subject_shape_run": source_subject_shape_run,
        "source_tail_coordinate_manifest_sha256": (
            source_tail_coordinate_manifest_sha256
        ),
        "source_subject_shape_manifest_sha256": (source_subject_shape_manifest_sha256),
        "source_logical_schema_mode": "exact_arrays_legacy_receipt_optional_v1",
        "tail_angle_sample_count": tail_angle_sample_count,
        "block_rows": block_rows,
        "output_shard_rows": output_shard_rows,
        "execution_backend": "serial",
        "num_workers": 1,
        "source_staging_mode": TAIL_KINEMATICS_SOURCE_STAGING_MODE,
        "source_revision_bundle_mode": TAIL_KINEMATICS_REVISION_BUNDLE_MODE,
        "storage_profile_id": storage_profile_id,
        "copy_backend": copy_backend,
        "keep_scratch": keep_scratch,
        "check_capacity": check_capacity,
    }
    require_tail_kinematics_invocation_parameters(parameters)
    payload = {
        "contract_id": CandidateInvocationContract.TAIL_KINEMATICS_V1.value,
        "parameters": json.loads(
            json.dumps(parameters, sort_keys=True, separators=(",", ":"))
        ),
    }
    return {
        "schema_id": ANALYSIS_CANDIDATE_INVOCATION_SCHEMA_ID,
        "schema_version": ANALYSIS_CANDIDATE_INVOCATION_SCHEMA_VERSION,
        "payload": payload,
        "payload_digest": canonical_json_sha256(payload),
    }


def require_tail_kinematics_invocation_manifest(value: Mapping[str, Any]) -> None:
    """Deeply validate one family-local invocation envelope."""

    if not isinstance(value, Mapping) or set(value) != {
        "schema_id",
        "schema_version",
        "payload",
        "payload_digest",
    }:
        raise ValueError("tail-kinematics invocation envelope field set differs")
    if (
        value["schema_id"] != ANALYSIS_CANDIDATE_INVOCATION_SCHEMA_ID
        or type(value["schema_version"]) is not int
        or value["schema_version"] != ANALYSIS_CANDIDATE_INVOCATION_SCHEMA_VERSION
    ):
        raise ValueError("tail-kinematics invocation schema identity differs")
    payload = value["payload"]
    if not isinstance(payload, Mapping) or set(payload) != {
        "contract_id",
        "parameters",
    }:
        raise ValueError("tail-kinematics invocation payload field set differs")
    if payload["contract_id"] != CandidateInvocationContract.TAIL_KINEMATICS_V1.value:
        raise ValueError("tail-kinematics invocation contract differs")
    require_tail_kinematics_invocation_parameters(payload["parameters"])
    if value["payload_digest"] != canonical_json_sha256(payload):
        raise ValueError("tail-kinematics invocation payload digest differs")


def _validate_exact_tail_run(run_group: Any) -> TailKinematicsDimensions:
    dimensions = infer_tail_kinematics_dimensions(run_group)
    attrs = _attrs(run_group)
    byte_planner_adopted = bool(attrs.get("byte_planner_adopted"))
    declarations = build_tail_kinematics_array_declarations(
        include_source_revision_bundle=dimensions.include_source_revision_bundle,
        byte_planner_adopted=byte_planner_adopted,
    )
    errors: list[str] = []
    if TAIL_KINEMATICS_ARRAY_SCHEMA_ATTR in attrs:
        errors.extend(
            validate_tail_kinematics_array_schema(
                run_group,
                byte_planner_adopted=byte_planner_adopted,
            )
        )
    else:
        expected_paths = {declaration.path for declaration in declarations}
        observed_paths: set[str] = set()

        def visit(group: Any, prefix: str = "") -> None:
            for name, _array in group.arrays():
                observed_paths.add(f"{prefix}/{name}" if prefix else str(name))
            for name, child in group.groups():
                child_prefix = f"{prefix}/{name}" if prefix else str(name)
                visit(child, child_prefix)

        visit(run_group)
        if observed_paths != expected_paths:
            errors.append(
                "legacy source exact array inventory differs: "
                f"missing={sorted(expected_paths - observed_paths)!r}, "
                f"unexpected={sorted(observed_paths - expected_paths)!r}"
            )
        for declaration in declarations:
            try:
                node = _array_at_path(run_group, declaration.path)
            except (KeyError, TypeError):
                continue
            errors.extend(
                f"{declaration.path}: {message}"
                for message in declaration.contract.validate_observation(
                    node,
                    dimensions=dimensions.contract_dimensions,
                )
            )
    if errors:
        raise ValueError(
            "tail-kinematics run differs from the exact logical schema: "
            + "; ".join(errors)
        )
    expected_count = TAIL_KINEMATICS_CORE_ARRAY_COUNT + (
        TAIL_KINEMATICS_REVISION_BUNDLE_ARRAY_COUNT
        if dimensions.include_source_revision_bundle
        else 0
    )
    if len(declarations) != expected_count:
        raise RuntimeError("tail-kinematics exact array count changed")
    return dimensions


def compute_tail_kinematics_logical_hashes(run_group: Any) -> dict[str, object]:
    """Hash all 21 or 23 exact decoded arrays with path/dtype/shape framing."""

    dimensions = _validate_exact_tail_run(run_group)
    declarations = build_tail_kinematics_array_declarations(
        include_source_revision_bundle=dimensions.include_source_revision_bundle,
        byte_planner_adopted=bool(_attrs(run_group).get("byte_planner_adopted")),
    )
    records: list[dict[str, object]] = []
    for declaration in declarations:
        values = np.asarray(_array_at_path(run_group, declaration.path)[:])
        records.append(
            {
                "path": declaration.path,
                "dtype": values.dtype.str,
                "shape": [int(value) for value in values.shape],
                "array_values_sha256": array_values_sha256(values),
            }
        )
    records.sort(key=lambda item: str(item["path"]))
    return {
        "contract_id": TAIL_KINEMATICS_LOGICAL_EQUALITY_CONTRACT,
        "optional_revision_bundle_present": (dimensions.include_source_revision_bundle),
        "arrays": records,
    }


def tail_kinematics_logical_manifest_sha256(run_group: Any) -> str:
    return canonical_json_sha256(compute_tail_kinematics_logical_hashes(run_group))


def build_tail_kinematics_execution_suite(
    source_run: Any,
    *,
    scale_id: str,
    description: str,
    seed: int = 17,
    repetitions: int = 5,
) -> dict[str, object]:
    """Build one suite from a live exact tail-kinematics logical source."""

    dimensions = _validate_exact_tail_run(source_run)
    receipt = build_tail_kinematics_storage_receipt(
        dimensions,
        profile=get_storage_profile(TAIL_KINEMATICS_EXECUTION_PROFILE_ID),
    )
    result = build_analysis_benchmark_suite(
        family_id=TAIL_KINEMATICS_EXECUTION_FAMILY_ID,
        scale=AnalysisBenchmarkScale(
            scale_id=scale_id,
            dimensions=receipt.dimensions,
            description=description,
        ),
        storage_receipt=receipt,
        seed=seed,
        repetitions=repetitions,
    )
    require_tail_kinematics_execution_suite(
        TAIL_KINEMATICS_EXECUTION_FAMILY_ID,
        result,
    )
    return result


def require_tail_kinematics_execution_suite(
    stage_id: str,
    benchmark_suite: Mapping[str, Any],
) -> None:
    """Reconstruct and require the exact live 21/23-array candidate plan."""

    require_analysis_benchmark_suite_manifest(benchmark_suite)
    payload = benchmark_suite["payload"]
    if (
        stage_id != TAIL_KINEMATICS_EXECUTION_FAMILY_ID
        or payload["family_id"] != stage_id
    ):
        raise ValueError("tail-kinematics benchmark suite family differs")
    raw_dimensions = payload["scale"]["dimensions"]
    if not isinstance(raw_dimensions, Mapping) or set(raw_dimensions) not in (
        {"n_rows", "n_tail_samples"},
        {"n_components", "n_rows", "n_tail_samples"},
    ):
        raise ValueError("tail-kinematics benchmark dimensions differ")
    if any(type(value) is not int for value in raw_dimensions.values()):
        raise ValueError("tail-kinematics dimensions must be exact integers")
    dimensions = TailKinematicsDimensions(
        n_rows=raw_dimensions["n_rows"],
        n_tail_samples=raw_dimensions["n_tail_samples"],
        n_components=raw_dimensions.get("n_components"),
    )
    expected = build_tail_kinematics_storage_receipt(
        dimensions,
        profile=get_storage_profile(TAIL_KINEMATICS_EXECUTION_PROFILE_ID),
    ).as_manifest()
    if payload["storage_plan_receipt"] != expected:
        raise ValueError(
            "tail-kinematics benchmark storage plan differs from executable planning"
        )
    expected_count = TAIL_KINEMATICS_CORE_ARRAY_COUNT + (
        TAIL_KINEMATICS_REVISION_BUNDLE_ARRAY_COUNT
        if dimensions.include_source_revision_bundle
        else 0
    )
    if len(expected["payload"]["arrays"]) != expected_count:
        raise RuntimeError("tail-kinematics planned array count changed")


def _record_digest(record: Any, *, label: str) -> str:
    digest = getattr(record, "record_sha256", None)
    if type(digest) is not str or not _SHA256.fullmatch(digest):
        raise ValueError(f"{label} record digest is invalid")
    return digest


def build_tail_kinematics_coordinate_evidence(
    *,
    source_publication: Any,
    candidate_publication: Any,
) -> dict[str, object]:
    """Bind one recomputation to its live source and new canonical authority."""

    source_manifest = _record_digest(
        source_publication.manifest,
        label="source tail publication",
    )
    source_shape_manifest = _record_digest(
        source_publication.source.manifest,
        label="source subject-shape publication",
    )
    candidate_shape_manifest = _record_digest(
        candidate_publication.source.manifest,
        label="candidate subject-shape publication",
    )
    candidate_manifest = _record_digest(
        candidate_publication.manifest,
        label="candidate tail publication",
    )
    if source_shape_manifest != candidate_shape_manifest:
        raise ValueError(
            "tail-kinematics candidate uses a different subject-shape authority"
        )
    published_ref = getattr(candidate_publication.manifest, "record_ref", None)
    if type(published_ref) is not str or not published_ref:
        raise ValueError("candidate tail publication reference is invalid")
    authorities = [
        {"role": "canonical_subject_shape", "sha256": source_shape_manifest},
        {"role": "source_tail_kinematics", "sha256": source_manifest},
    ]
    validation_payload = {
        "schema_id": "palette.tail_kinematics_candidate_coordinate_validation",
        "schema_version": 1,
        "source_authority_digests": authorities,
        "published_authority_sha256": candidate_manifest,
        "published_authority_ref": published_ref,
    }
    return {
        "role": "canonical_producer",
        "status": "verified_canonical_publication",
        "source_authority_digests": authorities,
        "published_authority_sha256": candidate_manifest,
        "published_authority_ref": published_ref,
        "temporal_axis_sha256": None,
        "temporal_axis_ref": None,
        "validator_ref": TAIL_KINEMATICS_COORDINATE_VALIDATOR_REF,
        "validation_receipt_sha256": canonical_json_sha256(validation_payload),
        "coordinate_gate_passed": True,
    }


__all__ = [
    "TAIL_KINEMATICS_COORDINATE_VALIDATOR_REF",
    "TAIL_KINEMATICS_CORE_ARRAY_COUNT",
    "TAIL_KINEMATICS_EXECUTION_FAMILY_ID",
    "TAIL_KINEMATICS_EXECUTION_PROFILE_ID",
    "TAIL_KINEMATICS_INVOCATION_FIELDS",
    "TAIL_KINEMATICS_LOGICAL_EQUALITY_CONTRACT",
    "TAIL_KINEMATICS_REVISION_BUNDLE_ARRAY_COUNT",
    "build_tail_kinematics_coordinate_evidence",
    "build_tail_kinematics_execution_suite",
    "build_tail_kinematics_invocation",
    "compute_tail_kinematics_logical_hashes",
    "require_tail_kinematics_execution_suite",
    "require_tail_kinematics_invocation_manifest",
    "require_tail_kinematics_invocation_parameters",
    "tail_kinematics_logical_manifest_sha256",
]
