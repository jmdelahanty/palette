"""Typed suite and identity helpers for tail-kinematics candidates.

This module is deliberately family-owned.  It reconstructs the exact 21-array
core (plus the atomic two-array source-revision bundle when present), binds the
live canonical subject-shape authority used by the scientific computation, and
defines the closed invocation payload that the shared invocation catalog can
adopt without changing this family's execution semantics.
"""

from __future__ import annotations

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
    CandidateInvocationContract,
    build_tail_kinematics_invocation,
    require_candidate_invocation_manifest,
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
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _attrs(group: Any) -> dict[str, Any]:
    attrs = group.attrs
    return dict(attrs.asdict() if hasattr(attrs, "asdict") else dict(attrs))


def _array_at_path(group: Any, path: str) -> Any:
    node = group
    for component in path.split("/"):
        node = node[component]
    return node


def require_tail_kinematics_invocation_manifest(value: Mapping[str, Any]) -> None:
    """Validate through the single shared invocation-contract owner."""

    require_candidate_invocation_manifest(
        value,
        expected_contract=CandidateInvocationContract.TAIL_KINEMATICS_V1,
        expected_profile_id=TAIL_KINEMATICS_EXECUTION_PROFILE_ID,
    )


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


def build_tail_kinematics_logical_hashes_from_array_digests(
    run_group: Any,
    array_content_sha256: Mapping[str, str],
) -> dict[str, object]:
    """Build exact logical equality evidence from one sealed array digest set."""

    dimensions = _validate_exact_tail_run(run_group)
    declarations = build_tail_kinematics_array_declarations(
        include_source_revision_bundle=dimensions.include_source_revision_bundle,
        byte_planner_adopted=bool(_attrs(run_group).get("byte_planner_adopted")),
    )
    expected_paths = {declaration.path for declaration in declarations}
    if set(array_content_sha256) != expected_paths:
        raise ValueError(
            "tail-kinematics array digest inventory differs from the exact schema"
        )
    records: list[dict[str, object]] = []
    for declaration in declarations:
        node = _array_at_path(run_group, declaration.path)
        digest = array_content_sha256.get(declaration.path)
        if type(digest) is not str or not _SHA256.fullmatch(digest):
            raise ValueError(
                f"tail-kinematics array digest is invalid for {declaration.path!r}"
            )
        records.append(
            {
                "path": declaration.path,
                "dtype": np.dtype(node.dtype).str,
                "shape": [int(value) for value in node.shape],
                "array_values_sha256": digest,
            }
        )
    records.sort(key=lambda item: str(item["path"]))
    return {
        "contract_id": TAIL_KINEMATICS_LOGICAL_EQUALITY_CONTRACT,
        "optional_revision_bundle_present": (dimensions.include_source_revision_bundle),
        "arrays": records,
    }


def compute_tail_kinematics_logical_hashes(run_group: Any) -> dict[str, object]:
    """Hash all 21 or 23 exact decoded arrays with path/dtype/shape framing."""

    dimensions = _validate_exact_tail_run(run_group)
    declarations = build_tail_kinematics_array_declarations(
        include_source_revision_bundle=dimensions.include_source_revision_bundle,
        byte_planner_adopted=bool(_attrs(run_group).get("byte_planner_adopted")),
    )
    digests: dict[str, str] = {}
    for declaration in declarations:
        values = np.asarray(_array_at_path(run_group, declaration.path)[:])
        digests[declaration.path] = array_values_sha256(values)
    return build_tail_kinematics_logical_hashes_from_array_digests(
        run_group,
        digests,
    )


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
    "TAIL_KINEMATICS_LOGICAL_EQUALITY_CONTRACT",
    "TAIL_KINEMATICS_REVISION_BUNDLE_ARRAY_COUNT",
    "build_tail_kinematics_coordinate_evidence",
    "build_tail_kinematics_execution_suite",
    "build_tail_kinematics_invocation",
    "compute_tail_kinematics_logical_hashes",
    "require_tail_kinematics_execution_suite",
    "require_tail_kinematics_invocation_manifest",
    "tail_kinematics_logical_manifest_sha256",
]
