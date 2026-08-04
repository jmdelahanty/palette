"""Exact execution contract for bout-classification v2 storage candidates.

Classifier outputs are not recomputed for a physical-layout experiment.  One
explicit completed v2 authority is replayed through the maintained direct
writer on node-local scratch, then its twenty decoded arrays are required to
match exactly.  The replay preserves the classifier and dependency identity
while avoiding hardware-dependent inference drift.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

import numpy as np

from fisheye.analysis.bout_classification_schema import (
    BOUT_CLASSIFICATION_ACCESS_UNIT_SEMANTICS,
    BOUT_CLASSIFICATION_CANDIDATE_ARRAY_DECLARATIONS,
    BOUT_CLASSIFICATION_FIELD_NAMES,
    BOUT_CLASSIFICATION_RUN_SCHEMA_ID,
    BOUT_CLASSIFICATION_RUN_SCHEMA_VERSION,
    CATEGORY_LABEL_BYTES_WIDTH,
    FAILURE_REASON_BYTES_WIDTH,
    BoutClassificationDimensions,
    validate_bout_classification_arrays,
)
from fisheye.analysis.megabouts_classifier import (
    MegaboutsClassificationResult,
    MegaboutsRuntime,
)
from fisheye.analysis.megabouts_classifier_inputs import MegaboutsClassifierInputPack
from fisheye.shared.coordinate_frame_record import array_values_sha256
from fisheye.shared.detect_reason_codec import decode_reason_bytes
from fisheye.shared.zarr.analysis_benchmark_suite import (
    AnalysisBenchmarkScale,
    build_analysis_benchmark_suite,
    require_analysis_benchmark_suite_manifest,
)
from fisheye.shared.zarr.analysis_storage_planning import (
    AnalysisArrayStorageFacts,
    plan_analysis_storage,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.storage_profiles import get_storage_profile

BOUT_CLASSIFICATION_EXECUTION_FAMILY_ID = "bout_classification"
BOUT_CLASSIFICATION_EXECUTION_PROFILE_ID = "published_http_v1"
BOUT_CLASSIFICATION_LOGICAL_EQUALITY_CONTRACT = "bout_classification_v2_arrays_v1"
BOUT_CLASSIFICATION_INVOCATION_CONTRACT_ID = "bout_classification_v1"
BOUT_CLASSIFICATION_ARRAY_COUNT = 20
BOUT_CLASSIFICATION_COORDINATE_VALIDATOR_REF = (
    "fisheye.analysis_workflows.bout_classification_candidate_execution:"
    "build_bout_classification_coordinate_evidence"
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COPY_BACKENDS = frozenset({"python", "rsync"})
_SCIENTIFIC_IDENTITY_FIELDS = (
    "schema_id",
    "schema_version",
    "method",
    "method_version",
    "adapter_method",
    "adapter_method_version",
    "classifier_family",
    "classifier_name",
    "classifier_version",
    "classifier_input_mode",
    "megabouts_preprocessing",
    "megabouts_segmentation",
    "source_mode",
    "row_axis",
    "invalid_window_policy",
    "source_fps",
    "window_duration_s",
    "window_frames",
    "megabouts_time_sampling",
    "source_bout_count",
    "valid_source_window_count",
    "invalid_source_window_count",
    "classified_bout_count",
    "source_refs",
    "parameters",
    "tail_angle_conversion",
    "trajectory_conversion",
    "invalid_frame_policy",
)
_DEPENDENCY_DIGEST_FIELDS = (
    "tail_posture_publication_manifest_sha256",
    "tail_posture_source_subject_shape_publication_manifest_sha256",
    "positions_mm_coordinate_descriptor_sha256",
    "track_motion_manifest_sha256",
    "swim_bout_source_track_motion_manifest_sha256",
)


def _attrs(group: Any) -> dict[str, Any]:
    attrs = group.attrs
    return dict(attrs.asdict() if hasattr(attrs, "asdict") else dict(attrs))


def _require_sha256(value: object, *, label: str) -> str:
    if type(value) is not str or not _SHA256.fullmatch(value):
        raise ValueError(f"{label} must be one lowercase SHA-256")
    return value


def infer_bout_classification_dimensions(
    run_group: Any,
) -> BoutClassificationDimensions:
    try:
        rows = int(run_group["per_bout/source_bout_id"].shape[0])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "bout-classification row dimension cannot be inferred"
        ) from exc
    declared = run_group.attrs.get("source_bout_count")
    if type(declared) is not int or declared != rows:
        raise ValueError("bout-classification source_bout_count differs")
    return BoutClassificationDimensions(n_bouts=rows)


def require_exact_bout_classification_run(
    run_group: Any,
) -> BoutClassificationDimensions:
    dimensions = infer_bout_classification_dimensions(run_group)
    issues = validate_bout_classification_arrays(run_group, dimensions=dimensions)
    if issues:
        detail = "; ".join(
            f"{issue.code}:{issue.path}:{issue.message}" for issue in issues
        )
        raise ValueError(f"bout-classification exact schema differs: {detail}")
    if (
        run_group.attrs.get("schema_id") != BOUT_CLASSIFICATION_RUN_SCHEMA_ID
        or run_group.attrs.get("schema_version")
        != BOUT_CLASSIFICATION_RUN_SCHEMA_VERSION
    ):
        raise ValueError("bout-classification run schema identity differs")
    if tuple(run_group.group_keys()) != ("per_bout",) or tuple(run_group.array_keys()):
        raise ValueError("bout-classification run inventory differs")
    per_bout = run_group["per_bout"]
    if set(per_bout.array_keys()) != set(BOUT_CLASSIFICATION_FIELD_NAMES) or tuple(
        per_bout.group_keys()
    ):
        raise ValueError("bout-classification per_bout inventory differs")
    require_bout_classification_semantics(run_group, dimensions=dimensions)
    return dimensions


def _decode_text(values: np.ndarray, *, width: int, label: str) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype != np.dtype("uint8") or array.shape[1:] != (width,):
        raise ValueError(f"{label} must be exact uint8 rows of width {width}")
    decoded = np.asarray(decode_reason_bytes(array), dtype=object)
    if decoded.shape != (array.shape[0],) or any(not str(value) for value in decoded):
        raise ValueError(f"{label} contains an empty or malformed string")
    return decoded


def require_bout_classification_semantics(
    run_group: Any,
    *,
    dimensions: BoutClassificationDimensions,
) -> None:
    per_bout = run_group["per_bout"]
    arrays = {
        name: np.asarray(per_bout[name][:]) for name in BOUT_CLASSIFICATION_FIELD_NAMES
    }
    rows = dimensions.n_bouts
    classified = arrays["classified"].astype(bool, copy=False)
    valid = arrays["valid"].astype(bool, copy=False)
    source_valid = arrays["source_window_valid"].astype(bool, copy=False)
    skipped = ~classified
    labels = _decode_text(
        arrays["category_label_bytes"],
        width=CATEGORY_LABEL_BYTES_WIDTH,
        label="category_label_bytes",
    )
    reasons = _decode_text(
        arrays["failure_reason_bytes"],
        width=FAILURE_REASON_BYTES_WIDTH,
        label="failure_reason_bytes",
    )
    if np.unique(arrays["source_bout_id"]).size != rows or np.any(
        arrays["source_bout_id"] < 0
    ):
        raise ValueError("source_bout_id must be unique and nonnegative")
    for start, end in (
        ("start_frame", "end_frame"),
        ("window_start_frame", "window_end_frame"),
    ):
        if np.any(arrays[start] < 0) or np.any(arrays[end] < arrays[start]):
            raise ValueError(f"{start}/{end} intervals are invalid")
    if np.any(classified & ~source_valid) or np.any(valid != classified):
        raise ValueError("classification validity bitmaps differ from writer semantics")
    for name in ("HB1_frame", "HB1_offset_frames", "category_id", "subcategory_id"):
        if np.any(skipped & (arrays[name] != -1)):
            raise ValueError(f"unclassified {name} must be -1")
    if np.any(skipped & (arrays["sign"] != 0)) or np.any(
        skipped & ~np.isnan(arrays["probability"])
    ):
        raise ValueError("unclassified sign/probability fill differs")
    if np.any(classified & (arrays["category_id"] < 0)) or np.any(
        classified & ~np.isfinite(arrays["probability"])
    ):
        raise ValueError("classified category/probability values are invalid")
    expected_hb1 = arrays["window_start_frame"] + arrays["HB1_offset_frames"].astype(
        np.int64
    )
    if np.any(classified & (arrays["HB1_offset_frames"] < 0)) or np.any(
        classified & (arrays["HB1_frame"] != expected_hb1)
    ):
        raise ValueError("classified HB1 frame arithmetic differs")
    for name in ("tail_valid_fraction", "traj_valid_fraction"):
        values = arrays[name]
        if np.any(~np.isfinite(values)) or np.any(values < 0) or np.any(values > 1):
            raise ValueError(f"{name} must be finite in [0,1]")
    if any(
        str(labels[index]) != "skipped_invalid_window"
        for index in np.flatnonzero(skipped)
    ):
        raise ValueError("unclassified category label sentinel differs")
    if any(str(reasons[index]) != "ok" for index in np.flatnonzero(classified)):
        raise ValueError("classified failure reason must be ok")


def compute_bout_classification_logical_hashes(run_group: Any) -> dict[str, object]:
    dimensions = require_exact_bout_classification_run(run_group)
    arrays = []
    for declaration in sorted(
        BOUT_CLASSIFICATION_CANDIDATE_ARRAY_DECLARATIONS,
        key=lambda item: item.path,
    ):
        values = np.asarray(run_group[declaration.path][:])
        arrays.append(
            {
                "path": declaration.path,
                "dtype": values.dtype.str,
                "shape": [int(value) for value in values.shape],
                "array_values_sha256": array_values_sha256(values),
            }
        )
    return {
        "contract_id": BOUT_CLASSIFICATION_LOGICAL_EQUALITY_CONTRACT,
        "dimensions": dimensions.contract_dimensions,
        "arrays": arrays,
    }


def bout_classification_logical_manifest_sha256(run_group: Any) -> str:
    return canonical_json_sha256(compute_bout_classification_logical_hashes(run_group))


def build_bout_classification_scientific_identity(
    run_group: Any,
) -> dict[str, object]:
    require_exact_bout_classification_run(run_group)
    attrs = _attrs(run_group)
    missing = [name for name in _SCIENTIFIC_IDENTITY_FIELDS if name not in attrs]
    if missing:
        raise ValueError(
            "bout-classification scientific identity is incomplete: "
            + ", ".join(missing)
        )
    identity = {name: attrs[name] for name in _SCIENTIFIC_IDENTITY_FIELDS}
    refs = identity["source_refs"]
    if not isinstance(refs, Mapping) or any(
        type(key) is not str or type(value) is not str for key, value in refs.items()
    ):
        raise ValueError("bout-classification source_refs differs")
    for field in _DEPENDENCY_DIGEST_FIELDS:
        _require_sha256(refs.get(field), label=f"source_refs.{field}")
    if (
        refs["track_motion_manifest_sha256"]
        != refs["swim_bout_source_track_motion_manifest_sha256"]
    ):
        raise ValueError("swim-bout and track-motion authorities differ")
    if run_group["per_bout"].attrs.get("source_swim_bout_path") != refs.get(
        "swim_bout_level"
    ):
        raise ValueError("per_bout swim-bout source binding differs")
    canonical_json_sha256(identity)
    return identity


def build_bout_classification_source_identity(run_group: Any) -> dict[str, object]:
    payload = {
        "scientific_identity": build_bout_classification_scientific_identity(run_group),
        "logical_manifest": compute_bout_classification_logical_hashes(run_group),
    }
    return {
        "schema_id": "palette.bout_classification_execution_source_identity",
        "schema_version": 1,
        "payload": payload,
        "payload_digest": canonical_json_sha256(payload),
    }


def _storage_receipt(run_group: Any):
    dimensions = require_exact_bout_classification_run(run_group)
    facts = {
        declaration.path: AnalysisArrayStorageFacts(
            path=declaration.path,
            shape=tuple(int(value) for value in run_group[declaration.path].shape),
            dtype=np.dtype(run_group[declaration.path].dtype),
            access_unit_semantics=BOUT_CLASSIFICATION_ACCESS_UNIT_SEMANTICS[
                declaration.path
            ],
        )
        for declaration in BOUT_CLASSIFICATION_CANDIDATE_ARRAY_DECLARATIONS
    }
    return plan_analysis_storage(
        BOUT_CLASSIFICATION_CANDIDATE_ARRAY_DECLARATIONS,
        facts,
        profile=get_storage_profile(BOUT_CLASSIFICATION_EXECUTION_PROFILE_ID),
        dimensions=dimensions.contract_dimensions,
    )


def build_bout_classification_execution_suite(
    source_run: Any,
    *,
    seed: int = 17,
    repetitions: int = 5,
) -> dict[str, object]:
    receipt = _storage_receipt(source_run)
    result = build_analysis_benchmark_suite(
        family_id=BOUT_CLASSIFICATION_EXECUTION_FAMILY_ID,
        scale=AnalysisBenchmarkScale(
            scale_id="explicit_bout_classification_v2_run",
            dimensions=receipt.dimensions,
            description=(
                "Exact twenty-array bout-classification v2 direct-writer replay."
            ),
        ),
        storage_receipt=receipt,
        seed=seed,
        repetitions=repetitions,
    )
    require_bout_classification_execution_suite(
        BOUT_CLASSIFICATION_EXECUTION_FAMILY_ID, result
    )
    return result


def require_bout_classification_execution_suite(
    stage_id: str,
    benchmark_suite: Mapping[str, Any],
) -> None:
    if stage_id != BOUT_CLASSIFICATION_EXECUTION_FAMILY_ID:
        raise ValueError("bout-classification suite owns only bout_classification")
    require_analysis_benchmark_suite_manifest(benchmark_suite)
    payload = benchmark_suite["payload"]
    if payload["family_id"] != stage_id:
        raise ValueError("bout-classification benchmark family differs")
    dimensions_raw = payload["scale"]["dimensions"]
    if not isinstance(dimensions_raw, Mapping) or set(dimensions_raw) != {"n_bouts"}:
        raise ValueError("bout-classification benchmark dimensions differ")
    dimensions = BoutClassificationDimensions(n_bouts=dimensions_raw["n_bouts"])
    records = payload["storage_plan_receipt"]["payload"].get("arrays")
    if not isinstance(records, list) or len(records) != BOUT_CLASSIFICATION_ARRAY_COUNT:
        raise ValueError("bout-classification plan must contain twenty arrays")
    facts = {}
    for record in records:
        observed = record.get("observed_facts") if isinstance(record, Mapping) else None
        if not isinstance(observed, Mapping):
            raise ValueError("bout-classification storage facts are absent")
        path = observed.get("path")
        if path not in BOUT_CLASSIFICATION_ACCESS_UNIT_SEMANTICS:
            raise ValueError("bout-classification storage path differs")
        facts[path] = AnalysisArrayStorageFacts(
            path=path,
            shape=tuple(observed["shape"]),
            dtype=np.dtype(observed["dtype"]),
            access_unit_semantics=BOUT_CLASSIFICATION_ACCESS_UNIT_SEMANTICS[path],
        )
    expected = plan_analysis_storage(
        BOUT_CLASSIFICATION_CANDIDATE_ARRAY_DECLARATIONS,
        facts,
        profile=get_storage_profile(BOUT_CLASSIFICATION_EXECUTION_PROFILE_ID),
        dimensions=dimensions.contract_dimensions,
    ).as_manifest()
    if payload["storage_plan_receipt"] != expected:
        raise ValueError("bout-classification benchmark plan differs")


def require_bout_classification_invocation_parameters(
    value: object,
) -> Mapping[str, Any]:
    fields = {
        "source_schema_id",
        "source_schema_version",
        "source_logical_schema_mode",
        "source_scientific_identity_sha256",
        "writer_replay_mode",
        "execution_backend",
        "num_workers",
        "source_staging_mode",
        "storage_profile_id",
        "copy_backend",
        "keep_scratch",
        "check_capacity",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("bout-classification invocation field set differs")
    if (
        value["source_schema_id"] != BOUT_CLASSIFICATION_RUN_SCHEMA_ID
        or type(value["source_schema_version"]) is not int
        or value["source_schema_version"] != BOUT_CLASSIFICATION_RUN_SCHEMA_VERSION
        or value["source_logical_schema_mode"]
        != "exact_bout_classification_v2_arrays_v1"
    ):
        raise ValueError("bout-classification source schema identity differs")
    _require_sha256(
        value["source_scientific_identity_sha256"],
        label="source_scientific_identity_sha256",
    )
    if value["writer_replay_mode"] != "exact_result_direct_writer_replay_v1":
        raise ValueError("bout-classification writer replay mode differs")
    if value["execution_backend"] != "serial" or value["num_workers"] != 1:
        raise ValueError("bout-classification replay requires one serial writer")
    if value["source_staging_mode"] != "source_run_snapshot_copy_v1":
        raise ValueError("bout-classification source staging mode differs")
    if value["storage_profile_id"] != BOUT_CLASSIFICATION_EXECUTION_PROFILE_ID:
        raise ValueError("bout-classification storage profile differs")
    if value["copy_backend"] not in _COPY_BACKENDS:
        raise ValueError("bout-classification copy backend differs")
    for field in ("keep_scratch", "check_capacity"):
        if type(value[field]) is not bool:
            raise TypeError(f"{field} must be an exact bool")
    return value


def reconstruct_bout_classification_writer_inputs(
    run_group: Any,
) -> tuple[MegaboutsClassifierInputPack, MegaboutsClassificationResult]:
    """Reconstruct the maintained writer inputs from one exact v2 result."""

    dimensions = require_exact_bout_classification_run(run_group)
    per_bout = run_group["per_bout"]
    values = {
        name: np.asarray(per_bout[name][:]) for name in BOUT_CLASSIFICATION_FIELD_NAMES
    }
    parameters = dict(run_group.attrs["parameters"])
    window_frames = int(run_group.attrs["window_frames"])
    n_bouts = dimensions.n_bouts
    source_valid = values["source_window_valid"].astype(bool, copy=False)
    failure_reason = _decode_text(
        values["failure_reason_bytes"],
        width=FAILURE_REASON_BYTES_WIDTH,
        label="failure_reason_bytes",
    )
    pack = MegaboutsClassifierInputPack(
        tail_array=np.zeros((n_bouts, 10, window_frames), dtype=np.float32),
        traj_array=np.zeros((n_bouts, 3, window_frames), dtype=np.float32),
        tail_valid=np.repeat(source_valid[:, None], window_frames, axis=1),
        traj_valid=np.repeat(source_valid[:, None], window_frames, axis=1),
        traj_reference_valid=source_valid.copy(),
        source_bout_id=values["source_bout_id"],
        source_start_frame=values["start_frame"],
        source_end_frame=values["end_frame"],
        window_start_frame=values["window_start_frame"],
        window_end_frame=values["window_end_frame"],
        tail_valid_fraction=values["tail_valid_fraction"],
        traj_valid_fraction=values["traj_valid_fraction"],
        max_consecutive_tail_invalid=values["max_consecutive_tail_invalid"],
        max_consecutive_traj_invalid=values["max_consecutive_traj_invalid"],
        valid_bout=source_valid,
        failure_reason=failure_reason,
        source_refs=dict(run_group.attrs["source_refs"]),
        parameters=parameters,
    )
    classified_indices = np.flatnonzero(values["classified"]).astype(np.int64)
    category_labels = _decode_text(
        values["category_label_bytes"],
        width=CATEGORY_LABEL_BYTES_WIDTH,
        label="category_label_bytes",
    )
    runtime: MegaboutsRuntime | None = None
    classifier_version = run_group.attrs.get("classifier_version")
    if classifier_version is not None:
        max_category = int(
            np.max(values["category_id"][classified_indices], initial=-1)
        )
        labels = [f"category_{index}" for index in range(max_category + 1)]
        for row in classified_indices.tolist():
            category = int(values["category_id"][row])
            label = str(category_labels[row])
            if labels[category] != f"category_{category}" and labels[category] != label:
                raise ValueError("one classifier category has inconsistent labels")
            labels[category] = label
        runtime = MegaboutsRuntime(
            classifier_class=object(),
            tracking_config_class=object(),
            segmentation_config_class=object(),
            category_names=tuple(labels),
            package_version=str(run_group.attrs.get("megabouts_package_version")),
            package_path=str(run_group.attrs.get("megabouts_package_path")),
            source_repo=None,
            git_commit=None,
        )
    result = MegaboutsClassificationResult(
        classified_indices=classified_indices,
        classif_results={
            "cat": values["category_id"][classified_indices],
            "subcat": values["subcategory_id"][classified_indices],
            "sign": values["sign"][classified_indices],
            "proba": values["probability"][classified_indices],
            "first_half_beat": values["HB1_offset_frames"][classified_indices],
        },
        runtime=runtime,
    )
    return pack, result


def build_bout_classification_coordinate_evidence(
    source_run: Any,
    candidate_run: Any,
) -> dict[str, object]:
    source_identity = build_bout_classification_scientific_identity(source_run)
    if build_bout_classification_scientific_identity(candidate_run) != source_identity:
        raise ValueError("bout-classification candidate scientific identity differs")
    refs = source_identity["source_refs"]
    authorities = sorted(
        (
            {"role": field.removesuffix("_sha256"), "sha256": refs[field]}
            for field in _DEPENDENCY_DIGEST_FIELDS
        ),
        key=lambda record: record["role"],
    )
    validation = {
        "schema_id": "palette.bout_classification_coordinate_validation",
        "schema_version": 1,
        "source_authority_digests": authorities,
        "scientific_identity_sha256": canonical_json_sha256(source_identity),
    }
    return {
        "role": "bound_derivative",
        "status": "verified_bound_source",
        "source_authority_digests": authorities,
        "published_authority_sha256": None,
        "published_authority_ref": None,
        "temporal_axis_sha256": None,
        "temporal_axis_ref": None,
        "validator_ref": BOUT_CLASSIFICATION_COORDINATE_VALIDATOR_REF,
        "validation_receipt_sha256": canonical_json_sha256(validation),
        "coordinate_gate_passed": True,
    }


__all__ = [
    "BOUT_CLASSIFICATION_ARRAY_COUNT",
    "BOUT_CLASSIFICATION_COORDINATE_VALIDATOR_REF",
    "BOUT_CLASSIFICATION_EXECUTION_FAMILY_ID",
    "BOUT_CLASSIFICATION_EXECUTION_PROFILE_ID",
    "BOUT_CLASSIFICATION_INVOCATION_CONTRACT_ID",
    "BOUT_CLASSIFICATION_LOGICAL_EQUALITY_CONTRACT",
    "bout_classification_logical_manifest_sha256",
    "build_bout_classification_coordinate_evidence",
    "build_bout_classification_execution_suite",
    "build_bout_classification_scientific_identity",
    "build_bout_classification_source_identity",
    "compute_bout_classification_logical_hashes",
    "infer_bout_classification_dimensions",
    "reconstruct_bout_classification_writer_inputs",
    "require_bout_classification_execution_suite",
    "require_bout_classification_invocation_parameters",
    "require_bout_classification_semantics",
    "require_exact_bout_classification_run",
]
