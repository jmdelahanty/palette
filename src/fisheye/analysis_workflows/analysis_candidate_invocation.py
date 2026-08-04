"""Closed, digest-bound invocation contracts for analysis candidates.

The execution-adapter catalog names one invocation contract per maintained
family.  This module owns the corresponding call payloads.  Runner-affecting
values must be present here rather than supplied as unauthenticated Python or
command-line arguments after an execution request has been signed.
"""

from __future__ import annotations

from enum import Enum
import json
import math
import re
from typing import Any, Mapping

from fisheye.shared.zarr.manifest_digest import canonical_json_sha256

ANALYSIS_CANDIDATE_INVOCATION_SCHEMA_ID = "palette.analysis_candidate_invocation"
ANALYSIS_CANDIDATE_INVOCATION_SCHEMA_VERSION = 1

_RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_PROFILE_ID = re.compile(r"^[a-z][a-z0-9_]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COPY_BACKENDS = frozenset({"python", "rsync"})
_EYE_SCHEDULERS = frozenset({"single-threaded", "threads", "processes", "distributed"})


class CandidateInvocationContract(str, Enum):
    """Closed typed call shapes; none permits an open kwargs payload."""

    TRACK_FLAT_V1 = "track_flat_v1"
    EXACT_TABULAR_V1 = "exact_tabular_v1"
    OCCUPANCY_V1 = "occupancy_v1"
    EYE_ANGLES_V1 = "eye_angles_v1"
    SUBJECT_SHAPE_V1 = "subject_shape_v1"
    TAIL_KINEMATICS_V1 = "tail_kinematics_v1"
    STIMULUS_RESPONSE_V1 = "stimulus_response_v1"
    STIMULUS_EPOCHS_V1 = "stimulus_epochs_v1"
    CHASER_DISTANCE_BASE_V1 = "chaser_distance_base_v1"
    TAIL_POSTURE_V1 = "tail_posture_v1"
    BOUT_CLASSIFICATION_V1 = "bout_classification_v1"


def _json_copy(value: object) -> Any:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("candidate invocation must be strict JSON") from exc
    return json.loads(encoded)


def _require_exact_fields(
    value: object,
    fields: set[str],
    *,
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{label} field set differs")
    return value


def _require_bool(value: object, *, label: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{label} must be an exact bool")
    return value


def _require_positive_int(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be one positive exact integer")
    return value


def _require_copy_backend(value: object) -> str:
    if type(value) is not str or value not in _COPY_BACKENDS:
        raise ValueError("copy_backend must be python or rsync")
    return value


def _require_profile_id(value: object) -> str:
    if type(value) is not str or not _PROFILE_ID.fullmatch(value):
        raise ValueError("storage_profile_id must be one canonical profile ID")
    return value


def _require_run_name(value: object, *, label: str) -> str:
    if type(value) is not str or value in {".", ".."} or not _RUN_NAME.fullmatch(value):
        raise ValueError(f"{label} must be one exact run name, not a path")
    return value


def _require_sha256(value: object, *, label: str) -> str:
    if type(value) is not str or not _SHA256.fullmatch(value):
        raise ValueError(f"{label} must be one lowercase SHA-256")
    return value


def _require_exact_tabular(parameters: object) -> Mapping[str, Any]:
    parsed = _require_exact_fields(
        parameters,
        {"storage_profile_id", "copy_backend", "keep_scratch"},
        label="exact-tabular invocation parameters",
    )
    _require_profile_id(parsed["storage_profile_id"])
    _require_copy_backend(parsed["copy_backend"])
    _require_bool(parsed["keep_scratch"], label="keep_scratch")
    return parsed


def _require_occupancy(parameters: object) -> Mapping[str, Any]:
    from .occupancy_candidate_execution import (
        require_occupancy_invocation_parameters,
    )

    return require_occupancy_invocation_parameters(parameters)


def _require_track_flat(parameters: object) -> Mapping[str, Any]:
    parsed = _require_exact_fields(
        parameters,
        {
            "source_schema_id",
            "source_schema_version",
            "source_run_type",
            "source_motion_authority_sha256",
            "storage_profile_id",
            "physical_bundle_mode",
            "copy_backend",
            "keep_scratch",
        },
        label="track-flat invocation parameters",
    )
    if parsed["source_schema_id"] != "analysis.track_kinematics_runs":
        raise ValueError("track-flat source_schema_id differs")
    if (
        type(parsed["source_schema_version"]) is not int
        or parsed["source_schema_version"] != 1
    ):
        raise ValueError("track-flat source_schema_version differs")
    if parsed["source_run_type"] != "offline":
        raise ValueError("track-flat source_run_type must be offline")
    digest = parsed["source_motion_authority_sha256"]
    if type(digest) is not str or not _SHA256.fullmatch(digest):
        raise ValueError("source_motion_authority_sha256 must be one SHA-256")
    _require_profile_id(parsed["storage_profile_id"])
    if parsed["physical_bundle_mode"] != "excluded_from_flat_candidate_v1":
        raise ValueError("track-flat physical_bundle_mode differs")
    _require_copy_backend(parsed["copy_backend"])
    _require_bool(parsed["keep_scratch"], label="keep_scratch")
    return parsed


def _require_eye_angles(parameters: object) -> Mapping[str, Any]:
    parsed = _require_exact_fields(
        parameters,
        {
            "subject_shape_run",
            "keypoint_run",
            "storage_profile_id",
            "chunk_rows",
            "angle_chunk_rows",
            "angle_chunk_columns",
            "output_shard_rows",
            "angle_shard_columns",
            "execution_backend",
            "scheduler",
            "num_workers",
            "shard_workers",
            "native_threads",
            "fps_source",
            "fps",
            "smoothing_window",
            "copy_backend",
            "keep_scratch",
            "check_capacity",
        },
        label="eye-angle invocation parameters",
    )
    _require_run_name(parsed["subject_shape_run"], label="subject_shape_run")
    _require_run_name(parsed["keypoint_run"], label="keypoint_run")
    _require_profile_id(parsed["storage_profile_id"])
    for field in (
        "chunk_rows",
        "angle_chunk_rows",
        "angle_chunk_columns",
        "output_shard_rows",
        "angle_shard_columns",
        "num_workers",
        "shard_workers",
        "native_threads",
    ):
        _require_positive_int(parsed[field], label=field)
    if parsed["angle_chunk_columns"] < 3:
        raise ValueError("angle_chunk_columns must preserve all three angle bundles")
    if parsed["execution_backend"] != "serial_driver":
        raise ValueError("eye-angle candidate execution_backend must be serial_driver")
    if (
        type(parsed["scheduler"]) is not str
        or parsed["scheduler"] not in _EYE_SCHEDULERS
    ):
        raise ValueError("eye-angle scheduler is unsupported")
    fps_source = parsed["fps_source"]
    fps = parsed["fps"]
    if fps_source == "authoritative_recording_metadata":
        if fps is not None:
            raise ValueError("metadata FPS selection requires fps=null")
    elif fps_source == "explicit_override":
        if (
            type(fps) not in {int, float}
            or not math.isfinite(float(fps))
            or float(fps) <= 0
        ):
            raise ValueError("explicit FPS must be finite and positive")
    else:
        raise ValueError("eye-angle fps_source is unsupported")
    smoothing = parsed["smoothing_window"]
    if smoothing is not None:
        _require_positive_int(smoothing, label="smoothing_window")
    _require_copy_backend(parsed["copy_backend"])
    _require_bool(parsed["keep_scratch"], label="keep_scratch")
    _require_bool(parsed["check_capacity"], label="check_capacity")
    return parsed


def _require_subject_shape(parameters: object) -> Mapping[str, Any]:
    parsed = _require_exact_fields(
        parameters,
        {
            "source_schema_id",
            "source_schema_version",
            "source_profile_id",
            "source_manifest_sha256",
            "source_refined_subject_masks_run",
            "source_refined_authority_sha256",
            "source_staging_mode",
            "storage_profile_id",
            "block_rows",
            "output_shard_rows",
            "execution_backend",
            "scheduler",
            "num_workers",
            "shard_copy_workers",
            "native_threads",
            "copy_backend",
            "keep_scratch",
            "check_capacity",
        },
        label="subject-shape invocation parameters",
    )
    if (
        parsed["source_schema_id"] != "analysis.subject_shape_runs"
        or type(parsed["source_schema_version"]) is not int
        or parsed["source_schema_version"] != 4
        or parsed["source_profile_id"] != "analysis.subject_shape.full_anatomy_v4"
    ):
        raise ValueError("subject-shape invocation source schema differs")
    _require_sha256(
        parsed["source_manifest_sha256"],
        label="source_manifest_sha256",
    )
    _require_run_name(
        parsed["source_refined_subject_masks_run"],
        label="source_refined_subject_masks_run",
    )
    _require_sha256(
        parsed["source_refined_authority_sha256"],
        label="source_refined_authority_sha256",
    )
    if parsed["source_staging_mode"] != "archive_snapshot_copy_v1":
        raise ValueError("subject-shape source_staging_mode differs")
    if parsed["storage_profile_id"] != "subject_shape_access_aware_candidate_v1":
        raise ValueError("subject-shape storage_profile_id differs")
    for field in (
        "block_rows",
        "output_shard_rows",
        "num_workers",
        "shard_copy_workers",
        "native_threads",
    ):
        _require_positive_int(parsed[field], label=field)
    if parsed["execution_backend"] not in {
        "serial_driver",
        "dask_worker_chunks",
    }:
        raise ValueError("subject-shape execution_backend differs")
    if type(parsed["scheduler"]) is not str or parsed["scheduler"] not in (
        _EYE_SCHEDULERS
    ):
        raise ValueError("subject-shape scheduler is unsupported")
    _require_copy_backend(parsed["copy_backend"])
    _require_bool(parsed["keep_scratch"], label="keep_scratch")
    _require_bool(parsed["check_capacity"], label="check_capacity")
    return parsed


def _require_tail_kinematics(parameters: object) -> Mapping[str, Any]:
    parsed = _require_exact_fields(
        parameters,
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
        },
        label="tail-kinematics invocation parameters",
    )
    _require_run_name(
        parsed["source_subject_shape_run"],
        label="source_subject_shape_run",
    )
    _require_sha256(
        parsed["source_tail_coordinate_manifest_sha256"],
        label="source_tail_coordinate_manifest_sha256",
    )
    _require_sha256(
        parsed["source_subject_shape_manifest_sha256"],
        label="source_subject_shape_manifest_sha256",
    )
    for field in (
        "tail_angle_sample_count",
        "block_rows",
        "output_shard_rows",
        "num_workers",
    ):
        _require_positive_int(parsed[field], label=field)
    if parsed["tail_angle_sample_count"] < 2:
        raise ValueError("tail_angle_sample_count must be at least two")
    if parsed["execution_backend"] != "serial":
        raise ValueError("tail-kinematics execution_backend must be serial")
    if parsed["num_workers"] != 1:
        raise ValueError("tail-kinematics num_workers must equal one")
    if parsed["source_staging_mode"] != "canonical_subject_shape_physical_subset_v1":
        raise ValueError("tail-kinematics source_staging_mode differs")
    if parsed["source_revision_bundle_mode"] != "atomic_source_mirror_v1":
        raise ValueError("tail-kinematics source_revision_bundle_mode differs")
    if parsed["source_logical_schema_mode"] != (
        "exact_arrays_legacy_receipt_optional_v1"
    ):
        raise ValueError("tail-kinematics source_logical_schema_mode differs")
    if parsed["storage_profile_id"] != "published_http_v1":
        raise ValueError("tail-kinematics storage_profile_id differs")
    _require_copy_backend(parsed["copy_backend"])
    _require_bool(parsed["keep_scratch"], label="keep_scratch")
    _require_bool(parsed["check_capacity"], label="check_capacity")
    return parsed


def _require_stimulus_epochs(parameters: object) -> Mapping[str, Any]:
    parsed = _require_exact_fields(
        parameters,
        {
            "source_schema_id",
            "source_schema_version",
            "candidate_schema_id",
            "candidate_schema_version",
            "source_stimulus_fingerprint_algorithm",
            "source_stimulus_fingerprint",
            "source_epoch_lineage_hash",
            "source_staging_mode",
            "storage_profile_id",
            "copy_backend",
            "keep_scratch",
        },
        label="stimulus-epoch invocation parameters",
    )
    if (
        parsed["source_schema_id"] != "palette.stimulus_epoch_windows.v1"
        or type(parsed["source_schema_version"]) is not int
        or parsed["source_schema_version"] != 1
    ):
        raise ValueError("stimulus-epoch source schema identity differs")
    if (
        parsed["candidate_schema_id"] != "palette.stimulus_epoch_windows.v2"
        or type(parsed["candidate_schema_version"]) is not int
        or parsed["candidate_schema_version"] != 2
    ):
        raise ValueError("stimulus-epoch candidate schema identity differs")
    if parsed["source_stimulus_fingerprint_algorithm"] != (
        "sha256_canonical_stimulus_group_logical_tree_v1"
    ):
        raise ValueError("stimulus-epoch source fingerprint algorithm differs")
    _require_sha256(
        parsed["source_stimulus_fingerprint"],
        label="source_stimulus_fingerprint",
    )
    _require_sha256(
        parsed["source_epoch_lineage_hash"],
        label="source_epoch_lineage_hash",
    )
    if parsed["source_staging_mode"] != "epoch_and_stimulus_logical_copy_v1":
        raise ValueError("stimulus-epoch source_staging_mode differs")
    _require_profile_id(parsed["storage_profile_id"])
    _require_copy_backend(parsed["copy_backend"])
    _require_bool(parsed["keep_scratch"], label="keep_scratch")
    return parsed


def _require_chaser_distance_base(parameters: object) -> Mapping[str, Any]:
    parsed = _require_exact_fields(
        parameters,
        {
            "source_schema_id",
            "source_schema_version",
            "candidate_schema_id",
            "candidate_schema_version",
            "source_authority_binding_sha256",
            "projection_id",
            "source_staging_mode",
            "storage_profile_id",
            "copy_backend",
            "keep_scratch",
        },
        label="chaser-distance-base invocation parameters",
    )
    if (
        parsed["source_schema_id"] != "palette.chaser_distance.v1"
        or type(parsed["source_schema_version"]) is not int
        or parsed["source_schema_version"] != 1
    ):
        raise ValueError("chaser-distance source schema identity differs")
    if (
        parsed["candidate_schema_id"]
        != "palette.chaser_distance.sealed_base_storage_candidate.v2"
        or type(parsed["candidate_schema_version"]) is not int
        or parsed["candidate_schema_version"] != 2
    ):
        raise ValueError("chaser-distance candidate schema identity differs")
    _require_sha256(
        parsed["source_authority_binding_sha256"],
        label="source_authority_binding_sha256",
    )
    if parsed["projection_id"] != "sealed_base_30_arrays_v1":
        raise ValueError("chaser-distance projection_id differs")
    if parsed["source_staging_mode"] != "sealed_base_logical_copy_v1":
        raise ValueError("chaser-distance source_staging_mode differs")
    _require_profile_id(parsed["storage_profile_id"])
    _require_copy_backend(parsed["copy_backend"])
    _require_bool(parsed["keep_scratch"], label="keep_scratch")
    return parsed


def _require_stimulus_response(parameters: object) -> Mapping[str, Any]:
    from .stimulus_response_candidate_execution import (
        require_stimulus_response_invocation_parameters,
    )

    return require_stimulus_response_invocation_parameters(parameters)


def _require_tail_posture(parameters: object) -> Mapping[str, Any]:
    from .tail_posture_candidate_execution import (
        require_tail_posture_invocation_parameters,
    )

    return require_tail_posture_invocation_parameters(parameters)


def _require_bout_classification(parameters: object) -> Mapping[str, Any]:
    from .bout_classification_candidate_execution import (
        require_bout_classification_invocation_parameters,
    )

    return require_bout_classification_invocation_parameters(parameters)


_PARAMETER_VALIDATORS = {
    CandidateInvocationContract.EXACT_TABULAR_V1: _require_exact_tabular,
    CandidateInvocationContract.OCCUPANCY_V1: _require_occupancy,
    CandidateInvocationContract.TRACK_FLAT_V1: _require_track_flat,
    CandidateInvocationContract.EYE_ANGLES_V1: _require_eye_angles,
    CandidateInvocationContract.SUBJECT_SHAPE_V1: _require_subject_shape,
    CandidateInvocationContract.TAIL_KINEMATICS_V1: _require_tail_kinematics,
    CandidateInvocationContract.STIMULUS_EPOCHS_V1: _require_stimulus_epochs,
    CandidateInvocationContract.CHASER_DISTANCE_BASE_V1: (
        _require_chaser_distance_base
    ),
    CandidateInvocationContract.STIMULUS_RESPONSE_V1: _require_stimulus_response,
    CandidateInvocationContract.TAIL_POSTURE_V1: _require_tail_posture,
    CandidateInvocationContract.BOUT_CLASSIFICATION_V1: (_require_bout_classification),
}


def candidate_invocation_contract_is_frozen(
    value: CandidateInvocationContract | str,
) -> bool:
    """Return whether an executable exact parameter grammar exists."""

    try:
        contract = CandidateInvocationContract(value)
    except (TypeError, ValueError):
        return False
    return contract in _PARAMETER_VALIDATORS


def require_candidate_invocation_manifest(
    value: Mapping[str, Any],
    *,
    expected_contract: CandidateInvocationContract | str | None = None,
    expected_profile_id: str | None = None,
) -> None:
    """Deeply validate one invocation and optional adapter bindings."""

    envelope = _require_exact_fields(
        value,
        {"schema_id", "schema_version", "payload", "payload_digest"},
        label="candidate invocation envelope",
    )
    if (
        envelope["schema_id"] != ANALYSIS_CANDIDATE_INVOCATION_SCHEMA_ID
        or type(envelope["schema_version"]) is not int
        or envelope["schema_version"] != ANALYSIS_CANDIDATE_INVOCATION_SCHEMA_VERSION
    ):
        raise ValueError("candidate invocation schema identity differs")
    payload = _require_exact_fields(
        envelope["payload"],
        {"contract_id", "parameters"},
        label="candidate invocation payload",
    )
    _json_copy(value)
    if envelope["payload_digest"] != canonical_json_sha256(payload):
        raise ValueError("candidate invocation payload digest differs")
    try:
        contract = CandidateInvocationContract(payload["contract_id"])
    except (TypeError, ValueError) as exc:
        raise ValueError("candidate invocation contract is unsupported") from exc
    validator = _PARAMETER_VALIDATORS.get(contract)
    if validator is None:
        raise ValueError("candidate invocation parameters are not yet frozen")
    parameters = validator(payload["parameters"])
    if expected_contract is not None:
        try:
            expected = CandidateInvocationContract(expected_contract)
        except (TypeError, ValueError) as exc:
            raise ValueError("expected invocation contract is unsupported") from exc
        if contract is not expected:
            raise ValueError("candidate invocation contract differs from adapter")
    if (
        expected_profile_id is not None
        and parameters["storage_profile_id"] != expected_profile_id
    ):
        raise ValueError("candidate invocation storage profile differs from adapter")


def _build_invocation(
    contract: CandidateInvocationContract,
    parameters: Mapping[str, Any],
) -> dict[str, object]:
    payload = {
        "contract_id": contract.value,
        "parameters": _json_copy(parameters),
    }
    result = {
        "schema_id": ANALYSIS_CANDIDATE_INVOCATION_SCHEMA_ID,
        "schema_version": ANALYSIS_CANDIDATE_INVOCATION_SCHEMA_VERSION,
        "payload": payload,
        "payload_digest": canonical_json_sha256(payload),
    }
    require_candidate_invocation_manifest(result, expected_contract=contract)
    return result


def build_exact_tabular_invocation(
    *,
    storage_profile_id: str,
    copy_backend: str,
    keep_scratch: bool,
) -> dict[str, object]:
    return _build_invocation(
        CandidateInvocationContract.EXACT_TABULAR_V1,
        {
            "storage_profile_id": storage_profile_id,
            "copy_backend": copy_backend,
            "keep_scratch": keep_scratch,
        },
    )


def build_occupancy_invocation(
    *,
    source_spatiotemporal_identity_sha256: str,
    storage_profile_id: str,
    copy_backend: str,
    keep_scratch: bool,
) -> dict[str, object]:
    return _build_invocation(
        CandidateInvocationContract.OCCUPANCY_V1,
        {
            "source_spatiotemporal_identity_sha256": (
                source_spatiotemporal_identity_sha256
            ),
            "storage_profile_id": storage_profile_id,
            "copy_backend": copy_backend,
            "keep_scratch": keep_scratch,
        },
    )


def build_track_flat_invocation(
    *,
    source_motion_authority_sha256: str,
    storage_profile_id: str,
    copy_backend: str,
    keep_scratch: bool,
) -> dict[str, object]:
    return _build_invocation(
        CandidateInvocationContract.TRACK_FLAT_V1,
        {
            "source_schema_id": "analysis.track_kinematics_runs",
            "source_schema_version": 1,
            "source_run_type": "offline",
            "source_motion_authority_sha256": source_motion_authority_sha256,
            "storage_profile_id": storage_profile_id,
            "physical_bundle_mode": "excluded_from_flat_candidate_v1",
            "copy_backend": copy_backend,
            "keep_scratch": keep_scratch,
        },
    )


def build_eye_angle_invocation(
    *,
    subject_shape_run: str,
    keypoint_run: str,
    storage_profile_id: str,
    chunk_rows: int,
    angle_chunk_rows: int,
    angle_chunk_columns: int,
    output_shard_rows: int,
    angle_shard_columns: int,
    execution_backend: str,
    scheduler: str,
    num_workers: int,
    shard_workers: int,
    native_threads: int,
    fps: float | None,
    smoothing_window: int | None,
    copy_backend: str,
    keep_scratch: bool,
    check_capacity: bool,
) -> dict[str, object]:
    return _build_invocation(
        CandidateInvocationContract.EYE_ANGLES_V1,
        {
            "subject_shape_run": subject_shape_run,
            "keypoint_run": keypoint_run,
            "storage_profile_id": storage_profile_id,
            "chunk_rows": chunk_rows,
            "angle_chunk_rows": angle_chunk_rows,
            "angle_chunk_columns": angle_chunk_columns,
            "output_shard_rows": output_shard_rows,
            "angle_shard_columns": angle_shard_columns,
            "execution_backend": execution_backend,
            "scheduler": scheduler,
            "num_workers": num_workers,
            "shard_workers": shard_workers,
            "native_threads": native_threads,
            "fps_source": (
                "authoritative_recording_metadata"
                if fps is None
                else "explicit_override"
            ),
            "fps": fps,
            "smoothing_window": smoothing_window,
            "copy_backend": copy_backend,
            "keep_scratch": keep_scratch,
            "check_capacity": check_capacity,
        },
    )


def build_subject_shape_invocation(
    *,
    source_manifest_sha256: str,
    source_refined_subject_masks_run: str,
    source_refined_authority_sha256: str,
    storage_profile_id: str,
    block_rows: int,
    output_shard_rows: int,
    execution_backend: str,
    scheduler: str,
    num_workers: int,
    shard_copy_workers: int,
    native_threads: int,
    copy_backend: str,
    keep_scratch: bool,
    check_capacity: bool,
) -> dict[str, object]:
    return _build_invocation(
        CandidateInvocationContract.SUBJECT_SHAPE_V1,
        {
            "source_schema_id": "analysis.subject_shape_runs",
            "source_schema_version": 4,
            "source_profile_id": "analysis.subject_shape.full_anatomy_v4",
            "source_manifest_sha256": source_manifest_sha256,
            "source_refined_subject_masks_run": source_refined_subject_masks_run,
            "source_refined_authority_sha256": source_refined_authority_sha256,
            "source_staging_mode": "archive_snapshot_copy_v1",
            "storage_profile_id": storage_profile_id,
            "block_rows": block_rows,
            "output_shard_rows": output_shard_rows,
            "execution_backend": execution_backend,
            "scheduler": scheduler,
            "num_workers": num_workers,
            "shard_copy_workers": shard_copy_workers,
            "native_threads": native_threads,
            "copy_backend": copy_backend,
            "keep_scratch": keep_scratch,
            "check_capacity": check_capacity,
        },
    )


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
    return _build_invocation(
        CandidateInvocationContract.TAIL_KINEMATICS_V1,
        {
            "source_subject_shape_run": source_subject_shape_run,
            "source_tail_coordinate_manifest_sha256": (
                source_tail_coordinate_manifest_sha256
            ),
            "source_subject_shape_manifest_sha256": (
                source_subject_shape_manifest_sha256
            ),
            "source_logical_schema_mode": ("exact_arrays_legacy_receipt_optional_v1"),
            "tail_angle_sample_count": tail_angle_sample_count,
            "block_rows": block_rows,
            "output_shard_rows": output_shard_rows,
            "execution_backend": "serial",
            "num_workers": 1,
            "source_staging_mode": "canonical_subject_shape_physical_subset_v1",
            "source_revision_bundle_mode": "atomic_source_mirror_v1",
            "storage_profile_id": storage_profile_id,
            "copy_backend": copy_backend,
            "keep_scratch": keep_scratch,
            "check_capacity": check_capacity,
        },
    )


def build_stimulus_epoch_invocation(
    *,
    source_stimulus_fingerprint: str,
    source_epoch_lineage_hash: str,
    storage_profile_id: str,
    copy_backend: str,
    keep_scratch: bool,
) -> dict[str, object]:
    return _build_invocation(
        CandidateInvocationContract.STIMULUS_EPOCHS_V1,
        {
            "source_schema_id": "palette.stimulus_epoch_windows.v1",
            "source_schema_version": 1,
            "candidate_schema_id": "palette.stimulus_epoch_windows.v2",
            "candidate_schema_version": 2,
            "source_stimulus_fingerprint_algorithm": (
                "sha256_canonical_stimulus_group_logical_tree_v1"
            ),
            "source_stimulus_fingerprint": source_stimulus_fingerprint,
            "source_epoch_lineage_hash": source_epoch_lineage_hash,
            "source_staging_mode": "epoch_and_stimulus_logical_copy_v1",
            "storage_profile_id": storage_profile_id,
            "copy_backend": copy_backend,
            "keep_scratch": keep_scratch,
        },
    )


def build_chaser_distance_base_invocation(
    *,
    source_authority_binding_sha256: str,
    storage_profile_id: str,
    copy_backend: str,
    keep_scratch: bool,
) -> dict[str, object]:
    return _build_invocation(
        CandidateInvocationContract.CHASER_DISTANCE_BASE_V1,
        {
            "source_schema_id": "palette.chaser_distance.v1",
            "source_schema_version": 1,
            "candidate_schema_id": (
                "palette.chaser_distance.sealed_base_storage_candidate.v2"
            ),
            "candidate_schema_version": 2,
            "source_authority_binding_sha256": (source_authority_binding_sha256),
            "projection_id": "sealed_base_30_arrays_v1",
            "source_staging_mode": "sealed_base_logical_copy_v1",
            "storage_profile_id": storage_profile_id,
            "copy_backend": copy_backend,
            "keep_scratch": keep_scratch,
        },
    )


def build_stimulus_response_invocation(
    *,
    source_track_kinematics_scope: str,
    source_track_kinematics_run: str,
    source_track_motion_manifest_sha256: str,
    source_stimulus_run: str,
    source_stimulus_logical_tree_sha256: str,
    source_stimulus_coordinate_lineage_sha256: str,
    source_bout_mode: str,
    source_swim_bout_run: str | None,
    source_swim_bout_logical_tree_sha256: str | None,
    scientific_parameters: Mapping[str, Any],
    execution_backend: str,
    source_staging_mode: str,
    storage_profile_id: str,
    copy_backend: str,
    keep_scratch: bool,
    check_capacity: bool,
) -> dict[str, object]:
    return _build_invocation(
        CandidateInvocationContract.STIMULUS_RESPONSE_V1,
        {
            "source_track_kinematics_scope": source_track_kinematics_scope,
            "source_track_kinematics_run": source_track_kinematics_run,
            "source_track_motion_manifest_sha256": (
                source_track_motion_manifest_sha256
            ),
            "source_stimulus_run": source_stimulus_run,
            "source_stimulus_logical_tree_sha256": (
                source_stimulus_logical_tree_sha256
            ),
            "source_stimulus_coordinate_lineage_sha256": (
                source_stimulus_coordinate_lineage_sha256
            ),
            "source_bout_mode": source_bout_mode,
            "source_swim_bout_run": source_swim_bout_run,
            "source_swim_bout_logical_tree_sha256": (
                source_swim_bout_logical_tree_sha256
            ),
            "scientific_parameters": dict(scientific_parameters),
            "execution_backend": execution_backend,
            "source_staging_mode": source_staging_mode,
            "storage_profile_id": storage_profile_id,
            "copy_backend": copy_backend,
            "keep_scratch": keep_scratch,
            "check_capacity": check_capacity,
        },
    )


def build_tail_posture_invocation(
    *,
    source_schema_id: str,
    source_schema_version: int,
    source_logical_schema_mode: str,
    source_subject_shape_run: str,
    source_tail_posture_manifest_sha256: str,
    source_subject_shape_manifest_sha256: str,
    source_tail_kinematics_run: str | None,
    source_tail_kinematics_manifest_sha256: str | None,
    view_family: str,
    head_source: str,
    keypoint_count: int,
    execution_backend: str,
    num_workers: int,
    source_staging_mode: str,
    storage_profile_id: str,
    copy_backend: str,
    keep_scratch: bool,
    check_capacity: bool,
) -> dict[str, object]:
    return _build_invocation(
        CandidateInvocationContract.TAIL_POSTURE_V1,
        {
            "source_schema_id": source_schema_id,
            "source_schema_version": source_schema_version,
            "source_logical_schema_mode": source_logical_schema_mode,
            "source_subject_shape_run": source_subject_shape_run,
            "source_tail_posture_manifest_sha256": (
                source_tail_posture_manifest_sha256
            ),
            "source_subject_shape_manifest_sha256": (
                source_subject_shape_manifest_sha256
            ),
            "source_tail_kinematics_run": source_tail_kinematics_run,
            "source_tail_kinematics_manifest_sha256": (
                source_tail_kinematics_manifest_sha256
            ),
            "view_family": view_family,
            "head_source": head_source,
            "keypoint_count": keypoint_count,
            "execution_backend": execution_backend,
            "num_workers": num_workers,
            "source_staging_mode": source_staging_mode,
            "storage_profile_id": storage_profile_id,
            "copy_backend": copy_backend,
            "keep_scratch": keep_scratch,
            "check_capacity": check_capacity,
        },
    )


def build_bout_classification_invocation(
    *,
    source_scientific_identity_sha256: str,
    storage_profile_id: str,
    copy_backend: str,
    keep_scratch: bool,
    check_capacity: bool,
) -> dict[str, object]:
    return _build_invocation(
        CandidateInvocationContract.BOUT_CLASSIFICATION_V1,
        {
            "source_schema_id": "analysis.bout_classification_runs",
            "source_schema_version": 2,
            "source_logical_schema_mode": ("exact_bout_classification_v2_arrays_v1"),
            "source_scientific_identity_sha256": (source_scientific_identity_sha256),
            "writer_replay_mode": "exact_result_direct_writer_replay_v1",
            "execution_backend": "serial",
            "num_workers": 1,
            "source_staging_mode": "source_run_snapshot_copy_v1",
            "storage_profile_id": storage_profile_id,
            "copy_backend": copy_backend,
            "keep_scratch": keep_scratch,
            "check_capacity": check_capacity,
        },
    )


__all__ = [
    "ANALYSIS_CANDIDATE_INVOCATION_SCHEMA_ID",
    "ANALYSIS_CANDIDATE_INVOCATION_SCHEMA_VERSION",
    "CandidateInvocationContract",
    "build_bout_classification_invocation",
    "build_chaser_distance_base_invocation",
    "build_exact_tabular_invocation",
    "build_occupancy_invocation",
    "build_eye_angle_invocation",
    "build_stimulus_response_invocation",
    "build_subject_shape_invocation",
    "build_stimulus_epoch_invocation",
    "build_tail_kinematics_invocation",
    "build_tail_posture_invocation",
    "build_track_flat_invocation",
    "candidate_invocation_contract_is_frozen",
    "require_candidate_invocation_manifest",
]
