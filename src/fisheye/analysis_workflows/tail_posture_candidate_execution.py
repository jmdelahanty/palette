"""Exact suite, identity, and coordinate evidence for tail-posture v3.

This family-owned module closes the scientific surface used by the typed
candidate runner.  Shared invocation-envelope and adapter-catalog ownership
remain in their central modules; this file deliberately owns only the exact
tail-posture parameter grammar, ten-array equality projection, and authority
evidence that those shared modules adopt.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

import numpy as np

from fisheye.analysis.tail_posture_view_schema import (
    TAIL_POSTURE_VIEW_ACCESS_UNIT_SEMANTICS,
    TAIL_POSTURE_VIEW_CANDIDATE_ARRAY_DECLARATIONS,
    TAIL_POSTURE_VIEW_RUN_SCHEMA_ID,
    TAIL_POSTURE_VIEW_RUN_SCHEMA_VERSION,
    TailPostureViewDimensions,
    validate_tail_posture_view_arrays,
)
from fisheye.shared.coordinate_frame_record import array_values_sha256
from fisheye.shared.detect_reason_codec import decode_reason_bytes
from fisheye.shared.zarr.analysis_benchmark_suite import (
    AnalysisBenchmarkScale,
    build_analysis_benchmark_suite,
    require_analysis_benchmark_suite_manifest,
)
from fisheye.shared.zarr.analysis_storage_planning import (
    AnalysisArrayStorageFacts,
    AnalysisStoragePlanReceipt,
    plan_analysis_storage,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.storage_profiles import get_storage_profile

TAIL_POSTURE_EXECUTION_FAMILY_ID = "tail_posture_view"
TAIL_POSTURE_EXECUTION_PROFILE_ID = "published_http_v1"
TAIL_POSTURE_LOGICAL_EQUALITY_CONTRACT = "tail_posture_v3_arrays_v1"
TAIL_POSTURE_INVOCATION_CONTRACT_ID = "tail_posture_v1"
TAIL_POSTURE_ARRAY_COUNT = 10
TAIL_POSTURE_COORDINATE_VALIDATOR_REF = (
    "fisheye.analysis_workflows.tail_posture_candidate_execution:"
    "build_tail_posture_coordinate_evidence"
)
TAIL_POSTURE_SOURCE_IDENTITY_SCHEMA_ID = (
    "palette.tail_posture_execution_source_identity"
)
TAIL_POSTURE_SOURCE_IDENTITY_SCHEMA_VERSION = 1

_RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COPY_BACKENDS = frozenset({"python", "rsync"})
_HEAD_SOURCES = frozenset({"head_endpoint_xy", "snout_tip_xy"})
_SCIENTIFIC_IDENTITY_FIELDS = (
    "schema_id",
    "schema_version",
    "method",
    "method_version",
    "row_axis",
    "view_family",
    "compatible_tool",
    "dependency_policy",
    "source_subject_shape_run",
    "source_subject_shape_path",
    "source_subject_shape_publication_manifest_sha256",
    "source_refined_subject_masks_run",
    "source_tail_kinematics_run",
    "source_tail_geometry_kind",
    "head_source",
    "keypoint_count",
    "angle_count",
    "angle_units_primary",
    "angle_convention",
    "keypoint_order",
    "tail_base_definition",
    "tail_tip_definition",
    "acquisition_frame_index_source",
    "row_lineage_copied",
    "row_lineage_missing",
    "source_refs",
    "algorithm_provenance",
    "reason_encoding",
    "reason_bytes_width",
    "reason_bytes_null_terminated",
)


def _attrs(group: Any) -> dict[str, Any]:
    attrs = group.attrs
    return dict(attrs.asdict() if hasattr(attrs, "asdict") else dict(attrs))


def _require_sha256(value: object, *, label: str) -> str:
    if type(value) is not str or not _SHA256.fullmatch(value):
        raise ValueError(f"{label} must be one lowercase SHA-256")
    return value


def _require_run_name(value: object, *, label: str) -> str:
    if type(value) is not str or not _RUN_NAME.fullmatch(value):
        raise ValueError(f"{label} must be one exact run name")
    return value


def infer_tail_posture_dimensions(run_group: Any) -> TailPostureViewDimensions:
    """Infer the three symbolic dimensions from the exact direct arrays."""

    try:
        rows = int(run_group["instance_key"].shape[0])
        keypoint_shape = tuple(
            int(value) for value in run_group["tail_keypoints_xy"].shape
        )
        angle_shape = tuple(int(value) for value in run_group["tail_angle_rad"].shape)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("tail-posture dimensions cannot be inferred") from exc
    if len(keypoint_shape) != 3 or len(angle_shape) != 2:
        raise ValueError("tail-posture keypoint or angle rank differs")
    return TailPostureViewDimensions(
        n_rows=rows,
        n_keypoints=keypoint_shape[1],
        n_angles=angle_shape[1],
    )


def _require_exact_tail_posture_run(run_group: Any) -> TailPostureViewDimensions:
    dimensions = infer_tail_posture_dimensions(run_group)
    issues = validate_tail_posture_view_arrays(run_group, dimensions=dimensions)
    if issues:
        detail = "; ".join(
            f"{issue.code}:{issue.path}:{issue.message}" for issue in issues
        )
        raise ValueError(f"tail-posture exact schema differs: {detail}")
    if (
        run_group.attrs.get("schema_id") != TAIL_POSTURE_VIEW_RUN_SCHEMA_ID
        or run_group.attrs.get("schema_version") != TAIL_POSTURE_VIEW_RUN_SCHEMA_VERSION
    ):
        raise ValueError("tail-posture run schema identity differs")
    expected = {
        declaration.path
        for declaration in TAIL_POSTURE_VIEW_CANDIDATE_ARRAY_DECLARATIONS
    }
    observed = {str(name) for name in run_group.array_keys()}
    if observed != expected or len(expected) != TAIL_POSTURE_ARRAY_COUNT:
        raise ValueError("tail-posture direct array inventory is not exactly ten")
    require_tail_posture_semantics(run_group, dimensions=dimensions)
    return dimensions


def require_tail_posture_semantics(
    run_group: Any,
    *,
    dimensions: TailPostureViewDimensions,
) -> None:
    """Require exact lineage, fill, validity, and dual-angle semantics."""

    rows = dimensions.n_rows
    instance_key = np.asarray(run_group["instance_key"][:])
    crop_rows = np.asarray(run_group["source_crop_row_ids"][:])
    frame_index = np.asarray(run_group["source_acquisition_frame_index"][:])
    valid = np.asarray(run_group["valid"][:], dtype=bool)
    if instance_key.size != np.unique(instance_key).size:
        raise ValueError("tail-posture instance_key values are not unique")
    if np.any(crop_rows < 0) or np.any(frame_index < 0):
        raise ValueError("tail-posture crop/frame lineage contains a sentinel")
    reasons = np.asarray(
        decode_reason_bytes(np.asarray(run_group["failure_reason_bytes"][:])),
        dtype=object,
    )
    if reasons.shape != (rows,):
        raise ValueError("tail-posture decoded reason shape differs")
    floating = (
        np.asarray(run_group["head_xy"][:]),
        np.asarray(run_group["head_yaw_rad"][:]),
        np.asarray(run_group["tail_keypoints_xy"][:]),
        np.asarray(run_group["tail_angle_rad"][:]),
        np.asarray(run_group["tail_angle_deg"][:]),
    )
    for row in range(rows):
        values = [item[row].reshape(-1) for item in floating]
        if bool(valid[row]):
            if str(reasons[row]) != "ok" or any(
                not np.all(np.isfinite(value)) for value in values
            ):
                raise ValueError(
                    "valid tail-posture rows require reason ok and finite floats"
                )
        elif (
            not str(reasons[row])
            or str(reasons[row]) == "ok"
            or any(not np.all(np.isnan(value)) for value in values)
        ):
            raise ValueError(
                "invalid tail-posture rows require a non-ok reason and all-NaN floats"
            )
    radians = np.asarray(run_group["tail_angle_rad"][:], dtype=np.float32)
    degrees = np.asarray(run_group["tail_angle_deg"][:], dtype=np.float32)
    if np.any(valid) and not np.allclose(
        degrees[valid],
        np.rad2deg(radians[valid]).astype(np.float32),
        rtol=1.0e-6,
        atol=1.0e-5,
    ):
        raise ValueError("tail-posture radian/degree payloads disagree")


def compute_tail_posture_logical_hashes(run_group: Any) -> dict[str, object]:
    """Hash all ten decoded arrays with exact path, dtype, and shape framing."""

    dimensions = _require_exact_tail_posture_run(run_group)
    records: list[dict[str, object]] = []
    for declaration in sorted(
        TAIL_POSTURE_VIEW_CANDIDATE_ARRAY_DECLARATIONS,
        key=lambda item: item.path,
    ):
        values = np.asarray(run_group[declaration.path][:])
        records.append(
            {
                "path": declaration.path,
                "dtype": values.dtype.str,
                "shape": [int(value) for value in values.shape],
                "array_values_sha256": array_values_sha256(values),
            }
        )
    return {
        "contract_id": TAIL_POSTURE_LOGICAL_EQUALITY_CONTRACT,
        "dimensions": dimensions.contract_dimensions,
        "arrays": records,
    }


def tail_posture_logical_manifest_sha256(run_group: Any) -> str:
    return canonical_json_sha256(compute_tail_posture_logical_hashes(run_group))


def build_tail_posture_scientific_identity(run_group: Any) -> dict[str, object]:
    """Return the exact storage-independent scientific attribute projection."""

    attrs = _attrs(run_group)
    identity = {name: attrs.get(name) for name in _SCIENTIFIC_IDENTITY_FIELDS}
    source_name = identity["source_subject_shape_run"]
    source_path = identity["source_subject_shape_path"]
    if (
        type(source_name) is not str
        or not _RUN_NAME.fullmatch(source_name)
        or source_path != f"analysis/subject_shape_runs/{source_name}"
    ):
        raise ValueError("tail-posture subject-shape source identity differs")
    _require_sha256(
        identity["source_subject_shape_publication_manifest_sha256"],
        label="source subject-shape publication manifest",
    )
    source_tail = identity["source_tail_kinematics_run"]
    expected_refs = {
        "subject_shape_run": source_path,
        "subject_shape_body_component": f"{source_path}/components/subject_body",
    }
    if source_tail is not None:
        _require_run_name(source_tail, label="source_tail_kinematics_run")
        expected_refs["tail_kinematics_run"] = (
            f"analysis/tail_kinematics_runs/{source_tail}"
        )
    expected = {
        "schema_id": TAIL_POSTURE_VIEW_RUN_SCHEMA_ID,
        "schema_version": TAIL_POSTURE_VIEW_RUN_SCHEMA_VERSION,
        "method": "tail_posture_view_from_subject_shape",
        "method_version": 1,
        "row_axis": "observation_instance",
        "view_family": "megabouts_compatible",
        "compatible_tool": "megabouts",
        "dependency_policy": "no_megabouts_dependency_required",
        "source_tail_geometry_kind": "subject_shape_tail_curve_resample",
        "angle_units_primary": "rad",
        "angle_convention": "megabouts_cumulative_segment_angle",
        "keypoint_order": "tail_base_to_tail_tip",
        "tail_base_definition": (
            "subject_shape.components.subject_body.tail_sample_xy[:,0]"
        ),
        "tail_tip_definition": (
            "subject_shape.components.subject_body.tail_sample_xy[:,-1]"
        ),
        "acquisition_frame_index_source": "source_acquisition_frame_index",
        "row_lineage_copied": [
            "instance_key",
            "source_crop_row_ids",
            "source_acquisition_frame_index",
        ],
        "row_lineage_missing": [],
        "source_refs": expected_refs,
        "algorithm_provenance": {
            "implementation": "independent_palette_compatible",
            "compatible_with": (
                "megabouts.tracking_data.convert_tracking."
                "compute_angles_from_keypoints"
            ),
            "copies_megabouts_code": False,
            "requires_megabouts_install": False,
        },
        "reason_encoding": "utf8-null-terminated",
        "reason_bytes_width": 64,
        "reason_bytes_null_terminated": True,
    }
    for name, value in expected.items():
        if identity.get(name) != value:
            raise ValueError(f"tail-posture scientific identity {name!r} differs")
    if identity["head_source"] not in _HEAD_SOURCES:
        raise ValueError("tail-posture head source is unsupported")
    keypoints = identity["keypoint_count"]
    angles = identity["angle_count"]
    if (
        type(keypoints) is not int
        or keypoints < 2
        or type(angles) is not int
        or angles != keypoints - 1
    ):
        raise ValueError("tail-posture scientific dimensions differ")
    refined = identity["source_refined_subject_masks_run"]
    if refined is not None:
        _require_run_name(refined, label="source_refined_subject_masks_run")
    canonical_json_sha256(identity)
    return identity


def _storage_receipt(run_group: Any) -> AnalysisStoragePlanReceipt:
    dimensions = _require_exact_tail_posture_run(run_group)
    facts = {
        declaration.path: AnalysisArrayStorageFacts(
            path=declaration.path,
            shape=tuple(int(value) for value in run_group[declaration.path].shape),
            dtype=np.dtype(run_group[declaration.path].dtype),
            access_unit_semantics=TAIL_POSTURE_VIEW_ACCESS_UNIT_SEMANTICS[
                declaration.path
            ],
        )
        for declaration in TAIL_POSTURE_VIEW_CANDIDATE_ARRAY_DECLARATIONS
    }
    return plan_analysis_storage(
        TAIL_POSTURE_VIEW_CANDIDATE_ARRAY_DECLARATIONS,
        facts,
        profile=get_storage_profile(TAIL_POSTURE_EXECUTION_PROFILE_ID),
        dimensions=dimensions.contract_dimensions,
    )


def build_tail_posture_execution_suite(
    source_run: Any,
    *,
    seed: int = 17,
    repetitions: int = 5,
) -> dict[str, object]:
    receipt = _storage_receipt(source_run)
    result = build_analysis_benchmark_suite(
        family_id=TAIL_POSTURE_EXECUTION_FAMILY_ID,
        scale=AnalysisBenchmarkScale(
            scale_id="explicit_tail_posture_v3_run",
            dimensions=receipt.dimensions,
            description=(
                "Exact ten-array tail-posture v3 candidate recomputed from its "
                "bound canonical subject-shape authority."
            ),
        ),
        storage_receipt=receipt,
        seed=seed,
        repetitions=repetitions,
    )
    require_tail_posture_execution_suite(TAIL_POSTURE_EXECUTION_FAMILY_ID, result)
    return result


def require_tail_posture_execution_suite(
    stage_id: str,
    benchmark_suite: Mapping[str, Any],
) -> None:
    if stage_id != TAIL_POSTURE_EXECUTION_FAMILY_ID:
        raise ValueError("tail-posture suite validator owns only tail_posture_view")
    require_analysis_benchmark_suite_manifest(benchmark_suite)
    payload = benchmark_suite["payload"]
    if payload["family_id"] != stage_id:
        raise ValueError("tail-posture benchmark family differs")
    receipt = payload["storage_plan_receipt"]
    receipt_payload = receipt["payload"]
    dimensions_raw = payload["scale"]["dimensions"]
    if (
        not isinstance(dimensions_raw, Mapping)
        or set(dimensions_raw) != {"n_rows", "n_keypoints", "n_angles"}
        or any(type(value) is not int for value in dimensions_raw.values())
    ):
        raise ValueError("tail-posture benchmark dimensions differ")
    dimensions = TailPostureViewDimensions(**dict(dimensions_raw))
    records = receipt_payload.get("arrays")
    if not isinstance(records, list) or len(records) != TAIL_POSTURE_ARRAY_COUNT:
        raise ValueError("tail-posture storage plan must contain ten arrays")
    facts: dict[str, AnalysisArrayStorageFacts] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("tail-posture storage-plan array record is invalid")
        observed = record.get("observed_facts")
        if not isinstance(observed, Mapping):
            raise ValueError("tail-posture storage-plan facts are absent")
        path = observed.get("path")
        shape = observed.get("shape")
        dtype = observed.get("dtype")
        if (
            type(path) is not str
            or not isinstance(shape, list)
            or any(type(value) is not int for value in shape)
            or type(dtype) is not str
            or path not in TAIL_POSTURE_VIEW_ACCESS_UNIT_SEMANTICS
        ):
            raise ValueError("tail-posture storage-plan facts differ")
        facts[path] = AnalysisArrayStorageFacts(
            path=path,
            shape=tuple(shape),
            dtype=np.dtype(dtype),
            access_unit_semantics=TAIL_POSTURE_VIEW_ACCESS_UNIT_SEMANTICS[path],
        )
    expected = plan_analysis_storage(
        TAIL_POSTURE_VIEW_CANDIDATE_ARRAY_DECLARATIONS,
        facts,
        profile=get_storage_profile(TAIL_POSTURE_EXECUTION_PROFILE_ID),
        dimensions=dimensions.contract_dimensions,
    ).as_manifest()
    if receipt != expected:
        raise ValueError(
            "tail-posture benchmark plan differs from executable byte planning"
        )


def require_tail_posture_invocation_parameters(value: object) -> Mapping[str, Any]:
    """Validate the exact parameter map proposed for shared ``tail_posture_v1``."""

    fields = {
        "source_schema_id",
        "source_schema_version",
        "source_logical_schema_mode",
        "source_subject_shape_run",
        "source_tail_posture_manifest_sha256",
        "source_subject_shape_manifest_sha256",
        "source_tail_kinematics_run",
        "source_tail_kinematics_manifest_sha256",
        "view_family",
        "head_source",
        "keypoint_count",
        "execution_backend",
        "num_workers",
        "source_staging_mode",
        "storage_profile_id",
        "copy_backend",
        "keep_scratch",
        "check_capacity",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("tail-posture invocation parameter field set differs")
    if (
        value["source_schema_id"] != TAIL_POSTURE_VIEW_RUN_SCHEMA_ID
        or type(value["source_schema_version"]) is not int
        or value["source_schema_version"] != TAIL_POSTURE_VIEW_RUN_SCHEMA_VERSION
        or value["source_logical_schema_mode"] != "exact_tail_posture_v3_arrays_v1"
    ):
        raise ValueError("tail-posture source schema identity differs")
    _require_run_name(
        value["source_subject_shape_run"], label="source_subject_shape_run"
    )
    _require_sha256(
        value["source_tail_posture_manifest_sha256"],
        label="source_tail_posture_manifest_sha256",
    )
    _require_sha256(
        value["source_subject_shape_manifest_sha256"],
        label="source_subject_shape_manifest_sha256",
    )
    source_tail = value["source_tail_kinematics_run"]
    source_tail_digest = value["source_tail_kinematics_manifest_sha256"]
    if source_tail is None:
        if source_tail_digest is not None:
            raise ValueError("tail-kinematics digest requires a source run")
    else:
        _require_run_name(source_tail, label="source_tail_kinematics_run")
        _require_sha256(
            source_tail_digest,
            label="source_tail_kinematics_manifest_sha256",
        )
    if value["view_family"] != "megabouts_compatible":
        raise ValueError("tail-posture view_family differs")
    if value["head_source"] not in _HEAD_SOURCES:
        raise ValueError("tail-posture head_source differs")
    if type(value["keypoint_count"]) is not int or value["keypoint_count"] < 2:
        raise ValueError("tail-posture keypoint_count must be an exact integer >= 2")
    if value["execution_backend"] != "serial" or value["num_workers"] != 1:
        raise ValueError("tail-posture typed execution requires one serial writer")
    if value["source_staging_mode"] != "logical_array_snapshot_v1":
        raise ValueError("tail-posture source_staging_mode differs")
    if value["storage_profile_id"] != TAIL_POSTURE_EXECUTION_PROFILE_ID:
        raise ValueError("tail-posture storage profile differs")
    if value["copy_backend"] not in _COPY_BACKENDS:
        raise ValueError("tail-posture copy backend differs")
    for field in ("keep_scratch", "check_capacity"):
        if type(value[field]) is not bool:
            raise TypeError(f"{field} must be an exact bool")
    return value


def build_tail_posture_coordinate_evidence(
    *,
    source_publication: Any,
    candidate_publication: Any,
    source_tail_kinematics_manifest_sha256: str | None,
) -> dict[str, object]:
    """Bind one recomputation to its source tail and subject-shape authorities."""

    if source_publication.kind != "tail_posture_view":
        raise ValueError("source coordinate authority is not tail posture")
    if candidate_publication.kind != "tail_posture_view":
        raise ValueError("candidate coordinate authority is not tail posture")
    source_manifest = _require_sha256(
        source_publication.manifest.record_sha256,
        label="source tail-posture manifest",
    )
    candidate_manifest = _require_sha256(
        candidate_publication.manifest.record_sha256,
        label="candidate tail-posture manifest",
    )
    source_shape = _require_sha256(
        source_publication.source.manifest.record_sha256,
        label="source subject-shape manifest",
    )
    candidate_shape = _require_sha256(
        candidate_publication.source.manifest.record_sha256,
        label="candidate subject-shape manifest",
    )
    if source_shape != candidate_shape:
        raise ValueError("tail-posture candidate subject-shape authority differs")
    authorities = [
        {"role": "canonical_subject_shape", "sha256": source_shape},
        {"role": "source_tail_posture", "sha256": source_manifest},
    ]
    if source_tail_kinematics_manifest_sha256 is not None:
        authorities.append(
            {
                "role": "bound_tail_kinematics_lineage",
                "sha256": _require_sha256(
                    source_tail_kinematics_manifest_sha256,
                    label="bound tail-kinematics manifest",
                ),
            }
        )
    validation = {
        "schema_id": "palette.tail_posture_candidate_coordinate_validation",
        "schema_version": 1,
        "source_authority_digests": authorities,
        "published_authority_sha256": candidate_manifest,
        "published_authority_ref": candidate_publication.manifest.record_ref,
    }
    return {
        "role": "canonical_producer",
        "status": "verified_canonical_publication",
        "source_authority_digests": authorities,
        "published_authority_sha256": candidate_manifest,
        "published_authority_ref": candidate_publication.manifest.record_ref,
        "temporal_axis_sha256": None,
        "temporal_axis_ref": None,
        "validator_ref": TAIL_POSTURE_COORDINATE_VALIDATOR_REF,
        "validation_receipt_sha256": canonical_json_sha256(validation),
        "coordinate_gate_passed": True,
    }


__all__ = [
    "TAIL_POSTURE_ARRAY_COUNT",
    "TAIL_POSTURE_COORDINATE_VALIDATOR_REF",
    "TAIL_POSTURE_EXECUTION_FAMILY_ID",
    "TAIL_POSTURE_EXECUTION_PROFILE_ID",
    "TAIL_POSTURE_INVOCATION_CONTRACT_ID",
    "TAIL_POSTURE_LOGICAL_EQUALITY_CONTRACT",
    "build_tail_posture_coordinate_evidence",
    "build_tail_posture_execution_suite",
    "build_tail_posture_scientific_identity",
    "compute_tail_posture_logical_hashes",
    "infer_tail_posture_dimensions",
    "require_tail_posture_execution_suite",
    "require_tail_posture_invocation_parameters",
    "require_tail_posture_semantics",
    "tail_posture_logical_manifest_sha256",
]
