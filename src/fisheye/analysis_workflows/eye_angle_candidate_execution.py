"""Exact suite, logical-equality, and coordinate helpers for eye-angle candidates.

This module is deliberately family-local.  It reconstructs the maintained
compact-v7 schema and access-aware physical plan instead of trusting declarations
copied into an execution request.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from fisheye.analysis.eye_angle_schema import (
    CANONICAL_ANGLE_CHANNELS,
    EyeAngleDimensions,
    ROI_QA_CHANNELS,
    ROI_VECTOR_CHANNELS,
    build_eye_angle_array_declarations,
    eye_angle_dimensions_from_run_attrs,
    validate_eye_angle_compact_run,
)
from fisheye.analysis.eye_angle_storage import (
    build_eye_angle_candidate_storage_plan,
)
from fisheye.shared.coordinate_frame_record import array_values_sha256
from fisheye.shared.zarr.analysis_benchmark_suite import (
    AnalysisBenchmarkScale,
    build_analysis_benchmark_suite,
    require_analysis_benchmark_suite_manifest,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


EYE_ANGLE_EXECUTION_FAMILY_ID = "eye_angles"
EYE_ANGLE_LOGICAL_ARRAY_COUNT = 41
EYE_ANGLE_LOGICAL_EQUALITY_CONTRACT = "eye_angle_compact_v7_arrays_v1"
EYE_ANGLE_COORDINATE_VALIDATOR_REF = (
    "fisheye.analysis_workflows.eye_angle_candidate_execution:"
    "build_eye_angle_bound_source_evidence"
)


def _attrs(group: Any) -> dict[str, Any]:
    attrs = group.attrs
    return dict(attrs.asdict() if hasattr(attrs, "asdict") else dict(attrs))


def _array_at_path(group: Any, path: str) -> Any:
    node = group
    for component in path.split("/"):
        node = node[component]
    return node


def eye_angle_dimensions_from_suite(
    benchmark_suite: Mapping[str, Any],
) -> EyeAngleDimensions:
    """Parse the closed eye-angle dimensions from one validated suite."""

    require_analysis_benchmark_suite_manifest(benchmark_suite)
    dimensions = benchmark_suite["payload"]["scale"]["dimensions"]
    if not isinstance(dimensions, Mapping) or set(dimensions) != {
        "angle_block_width",
        "n_angle_channels",
        "n_frames",
        "n_qa_channels",
        "n_roi_rows",
        "n_vector_channels",
    }:
        raise ValueError("eye-angle benchmark dimensions differ")
    expected_static = {
        "n_angle_channels": len(CANONICAL_ANGLE_CHANNELS),
        "n_vector_channels": len(ROI_VECTOR_CHANNELS),
        "n_qa_channels": len(ROI_QA_CHANNELS),
    }
    for field, expected in expected_static.items():
        if type(dimensions[field]) is not int or dimensions[field] != expected:
            raise ValueError(f"eye-angle benchmark {field} differs")
    return EyeAngleDimensions(
        n_roi_rows=dimensions["n_roi_rows"],
        n_frames=dimensions["n_frames"],
        angle_block_width=dimensions["angle_block_width"],
    )


def require_eye_angle_execution_suite(
    stage_id: str,
    benchmark_suite: Mapping[str, Any],
) -> None:
    """Require the suite to equal the live 41-array byte-planned contract."""

    require_analysis_benchmark_suite_manifest(benchmark_suite)
    payload = benchmark_suite["payload"]
    if stage_id != EYE_ANGLE_EXECUTION_FAMILY_ID or payload["family_id"] != stage_id:
        raise ValueError("eye-angle benchmark suite family differs")
    dimensions = eye_angle_dimensions_from_suite(benchmark_suite)
    expected = build_eye_angle_candidate_storage_plan(dimensions).as_manifest()
    if payload["storage_plan_receipt"] != expected:
        raise ValueError(
            "eye-angle benchmark storage plan differs from the live 41-array plan"
        )
    records = expected["payload"]["arrays"]
    if len(records) != EYE_ANGLE_LOGICAL_ARRAY_COUNT:
        raise RuntimeError("live eye-angle storage plan no longer contains 41 arrays")


def build_eye_angle_execution_suite(
    source_run: Any,
    *,
    scale_id: str,
    description: str,
    seed: int = 17,
    repetitions: int = 5,
) -> dict[str, object]:
    """Build a shared benchmark suite from one valid compact-v7 source run."""

    issues = validate_eye_angle_compact_run(source_run)
    if issues:
        raise ValueError(
            "eye-angle source is not exact compact-v7: "
            + "; ".join(f"{item.code}:{item.path}:{item.message}" for item in issues)
        )
    dimensions = eye_angle_dimensions_from_run_attrs(_attrs(source_run))
    receipt = build_eye_angle_candidate_storage_plan(dimensions)
    result = build_analysis_benchmark_suite(
        family_id=EYE_ANGLE_EXECUTION_FAMILY_ID,
        scale=AnalysisBenchmarkScale(
            scale_id=scale_id,
            dimensions=tuple(sorted(receipt.dimensions)),
            description=description,
        ),
        storage_receipt=receipt,
        seed=seed,
        repetitions=repetitions,
    )
    require_eye_angle_execution_suite(EYE_ANGLE_EXECUTION_FAMILY_ID, result)
    return result


def compute_eye_angle_logical_hashes(run_group: Any) -> dict[str, object]:
    """Hash all 41 exact decoded arrays with dtype/shape/path framing."""

    issues = validate_eye_angle_compact_run(run_group)
    if issues:
        raise ValueError(
            "eye-angle logical hashing requires exact compact-v7: "
            + "; ".join(f"{item.code}:{item.path}:{item.message}" for item in issues)
        )
    dimensions = eye_angle_dimensions_from_run_attrs(_attrs(run_group))
    declarations = build_eye_angle_array_declarations(byte_planner_adopted=False)
    records: list[dict[str, object]] = []
    for declaration in declarations:
        node = _array_at_path(run_group, declaration.path)
        values = np.asarray(node[:])
        expected_shape = tuple(
            dimensions.contract_dimensions[item] if isinstance(item, str) else item
            for item in declaration.contract.shape_template
        )
        expected_dtype = np.dtype(declaration.contract.dtype.numpy_dtype)
        if values.shape != expected_shape or values.dtype != expected_dtype:
            raise ValueError(
                f"{declaration.path} differs from its exact logical declaration"
            )
        records.append(
            {
                "path": declaration.path,
                "dtype": values.dtype.str,
                "shape": list(values.shape),
                "array_values_sha256": array_values_sha256(values),
            }
        )
    records.sort(key=lambda item: str(item["path"]))
    if len(records) != EYE_ANGLE_LOGICAL_ARRAY_COUNT:
        raise RuntimeError("eye-angle logical projection no longer contains 41 arrays")
    return {
        "contract_id": EYE_ANGLE_LOGICAL_EQUALITY_CONTRACT,
        "arrays": records,
    }


def eye_angle_logical_manifest_sha256(run_group: Any) -> str:
    return canonical_json_sha256(compute_eye_angle_logical_hashes(run_group))


def _authority_digest(value: object, *, label: str) -> str:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} authority is missing")
    digest = value.get("record_sha256")
    if (
        type(digest) is not str
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"{label} authority digest is invalid")
    body = dict(value)
    body.pop("record_sha256")
    if canonical_json_sha256(body) != digest:
        raise ValueError(f"{label} authority self-digest differs")
    return digest


def build_eye_angle_bound_source_evidence(
    source_contracts: Mapping[str, Any],
) -> dict[str, object]:
    """Build source-only coordinate evidence from freshly resolved authorities."""

    if not isinstance(source_contracts, Mapping) or set(source_contracts) != {
        "eye_geometry",
        "keypoints",
        "diagnostic_base_keypoints",
        "resolved_arrays",
    }:
        raise ValueError("eye-angle source-contract field set differs")
    geometry = source_contracts["eye_geometry"]
    keypoints = source_contracts["keypoints"]
    if not isinstance(geometry, Mapping) or not isinstance(keypoints, Mapping):
        raise ValueError("eye-angle bound source contracts are malformed")
    authorities = [
        {
            "role": "canonical_keypoints",
            "sha256": _authority_digest(
                keypoints.get("canonical_keypoint_authority"),
                label="canonical keypoint",
            ),
        },
        {
            "role": "subject_shape_eye_geometry",
            "sha256": _authority_digest(
                geometry.get("source_authority"),
                label="subject-shape eye geometry",
            ),
        },
    ]
    validation_payload = {
        "schema_id": "palette.eye_angle_bound_source_validation",
        "schema_version": 1,
        "source_authority_digests": authorities,
        "source_contracts_sha256": canonical_json_sha256(source_contracts),
    }
    return {
        "role": "bound_derivative",
        "status": "verified_bound_source",
        "source_authority_digests": authorities,
        "published_authority_sha256": None,
        "published_authority_ref": None,
        "temporal_axis_sha256": None,
        "temporal_axis_ref": None,
        "validator_ref": EYE_ANGLE_COORDINATE_VALIDATOR_REF,
        "validation_receipt_sha256": canonical_json_sha256(validation_payload),
        "coordinate_gate_passed": True,
    }


__all__ = [
    "EYE_ANGLE_COORDINATE_VALIDATOR_REF",
    "EYE_ANGLE_EXECUTION_FAMILY_ID",
    "EYE_ANGLE_LOGICAL_ARRAY_COUNT",
    "EYE_ANGLE_LOGICAL_EQUALITY_CONTRACT",
    "build_eye_angle_bound_source_evidence",
    "build_eye_angle_execution_suite",
    "compute_eye_angle_logical_hashes",
    "eye_angle_dimensions_from_suite",
    "eye_angle_logical_manifest_sha256",
    "require_eye_angle_execution_suite",
]
