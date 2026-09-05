"""Strict core-behavior admission behind the generic cohort interface.

The execution report is the immutable recording bundle roster.  Its run names
are never accepted as scientific authority by themselves: this adapter opens
each named Zarr publication through the same strict binders used by the
standalone analytics exports, proves their shared acquisition authority, and
seals those results into the generic validated-behavior bundle set.

This module deliberately does not publish Parquet or define another manifest
family.  It supplies one installed bundle profile to the existing
``validated_behavior/v1`` publication engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from fisheye.analytics_exports.activity_spatial_time_bins import (
    BoundActivitySpatialSources,
    bind_activity_spatial_sources,
)
from fisheye.analytics_exports.eye_trace_samples import (
    bind_eye_trace_source,
    eye_trace_projection_contract,
)
from fisheye.analytics_exports.kinematics_samples import (
    BoundKinematicsSamplesSource,
    CORE_MOTION_PROJECTION_PROFILE_ID,
    CORE_MOTION_SOURCE_SURFACE_PROFILE_ID,
    KINEMATICS_SOURCE_SURFACE_PROFILE_ID,
    bind_kinematics_samples_source,
    core_motion_projection_contract,
    kinematics_projection_contract,
)
from fisheye.analytics_exports.tail_trace_samples import (
    BoundTailTraceSources,
    bind_tail_trace_sources,
    tail_trace_projection_contract,
)
from fisheye.analytics_exports.validated_behavior_core_behavior_contracts import (
    CANONICAL_SWIM_BOUTS_CAPABILITY,
    CORE_BEHAVIOR_CAPABILITY_KEYS,
    CORE_BEHAVIOR_CAPABILITY_PROFILE_ID,
    CORE_BEHAVIOR_CAPABILITY_PROFILE_ID_V1,
    CORE_BEHAVIOR_EXPORT_PROFILE_ID,
    CORE_BEHAVIOR_EXPORT_PROFILE_ID_V1,
    CROSS_GRAIN_JOIN_AUTHORITY,
    EYE_TRACE_CAPABILITY,
    KINEMATICS_SAMPLES_CAPABILITY,
    KINEMATICS_SAMPLES,
    SUBJECT_BODY_FRAME_CAPABILITY,
    TAIL_TRACE_CAPABILITY,
)
from fisheye.shared.pixel_frame_authority import (
    BoundAcquisitionCameraFrame,
    load_persisted_acquisition_camera_authority,
)
from fisheye.shared.subject_shape_coordinate_publication import (
    BoundSubjectShapeCoordinatePublication,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr_io import open_zarr_root

from .validated_behavior_cohort import (
    CAPABILITY_STATES,
    build_capability_contract,
    build_validated_behavior_bundle_set,
    validate_validated_behavior_bundle_set,
)
from .contracts import TemporalPolicy
from .core_authority_roster import (
    BoutAuthorityIdentity,
    build_core_authority_roster,
    core_roster_bout_identity,
)
from .validated_behavior_cohort_adapters import (
    sha256_file,
    validate_membership_current_sources,
)
from .validated_behavior_source_admission import (
    CORE_BEHAVIOR_EXECUTION_ADMISSION_ROLE,
    CORE_BEHAVIOR_EXECUTION_SCHEMA_ID,
    CORE_BEHAVIOR_EXECUTION_SCHEMA_VERSION,
    bind_core_behavior_execution_report,
)

CORE_BEHAVIOR_BUNDLE_ADAPTER_ID = "core_behavior_execution_report_v3"
CORE_BEHAVIOR_BUNDLE_METHOD_ID = "strict_named_authority_bundle_v1"
CORE_BEHAVIOR_BUNDLE_STATUS = "complete"
SUPPORTED_CORE_BEHAVIOR_EXPORT_PROFILE_IDS = frozenset(
    {CORE_BEHAVIOR_EXPORT_PROFILE_ID_V1, CORE_BEHAVIOR_EXPORT_PROFILE_ID}
)


class CoreBehaviorCohortAdapterError(ValueError):
    """A report-selected core-behavior source is absent or inconsistent."""


def _fail(message: str) -> None:
    raise CoreBehaviorCohortAdapterError(message)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _mapping(value: object, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{field_name} must be one object.")
    return value


def _sealed(body: Mapping[str, Any]) -> dict[str, Any]:
    normalized = _plain(body)
    return {**normalized, "payload_sha256": canonical_json_sha256(normalized)}


def _require_export_profile_id(value: object) -> str:
    profile_id = str(value)
    if profile_id not in SUPPORTED_CORE_BEHAVIOR_EXPORT_PROFILE_IDS:
        _fail(f"Unsupported core-behavior export profile: {profile_id!r}.")
    return profile_id


def core_behavior_capability_contract(
    export_profile_id: str = CORE_BEHAVIOR_EXPORT_PROFILE_ID,
) -> dict[str, Any]:
    """Return the closed capability vocabulary for this bundle profile."""

    reasons: dict[str, tuple[str | None, ...]] = {
        "complete": (None,),
        "inapplicable": ("member_not_admitted",),
        "invalid": ("invalid_source_authority",),
        "review_required": ("source_review_required",),
        "stale": ("source_stale",),
        "unavailable": (
            "blocked_by_invalid_membership",
            "blocked_by_unavailable_membership",
        ),
    }
    if set(reasons) != set(CAPABILITY_STATES):  # pragma: no cover - constant guard
        _fail("Core-behavior capability-state vocabulary is incomplete.")
    profile_id = _require_export_profile_id(export_profile_id)
    capability_profile_id = (
        CORE_BEHAVIOR_CAPABILITY_PROFILE_ID_V1
        if profile_id == CORE_BEHAVIOR_EXPORT_PROFILE_ID_V1
        else CORE_BEHAVIOR_CAPABILITY_PROFILE_ID
    )
    return build_capability_contract(
        profile_id=capability_profile_id,
        keys=CORE_BEHAVIOR_CAPABILITY_KEYS,
        reason_codes_by_state=reasons,
    )


def subject_body_frame_projection_contract() -> dict[str, Any]:
    """Describe the exact source-camera body-frame projection used by the table."""

    return _sealed(
        {
            "schema_id": "palette.subject_body_frame_samples.projection",
            "schema_version": 1,
            "source_grain": "one_canonical_subject_shape_observation",
            "output_grain": "one_subject_body_frame_observation",
            "row_selection": "all_subject_shape_rows_in_persisted_order",
            "time_formula": "source_acquisition_frame_index/source_video_metadata.fps",
            "coordinate_space": "source_camera_image_px",
            "validity_authority": "body_frame/axis_valid",
            "failure_reason_authority": "body_frame/failure_reason_bytes",
            "invalid_float_semantics": "source_ieee_nan_not_arrow_null",
        }
    )


def canonical_swim_bout_projection_contract() -> dict[str, Any]:
    """Describe the exact maintained default-signal bout event projection."""

    return _sealed(
        {
            "schema_id": "palette.canonical_swim_bouts.projection",
            "schema_version": 1,
            "source_grain": "one_selected_track_default_signal_bout",
            "output_grain": "one_track_bout",
            "row_selection": "all_default_signal_bouts_sorted_by_track_then_bout_id",
            "duration_source": "tables/bouts.duration_s",
            "path_length_source": "tables/bouts.path_length_mm",
            "net_displacement_source": "tables/bouts.net_displacement_mm",
            "mean_speed_source": "tables/bouts.mean_speed_mm_s",
            "peak_speed_source": "tables/bouts.peak_physical_speed_mm_s",
            "tortuosity_formula": "path_length_mm/net_displacement_mm_when_displacement_gt_1e-6_else_nan",
            "invalid_float_semantics": "source_ieee_nan_not_arrow_null",
        }
    )


def _subject_shape_binding(
    publication: BoundSubjectShapeCoordinatePublication,
) -> dict[str, Any]:
    run = publication._run
    temporal = publication.temporal_authority
    acquisition = temporal.acquisition_frame
    source_rate = acquisition.record.source_video_metadata.get("fps")
    if isinstance(source_rate, bool) or not isinstance(source_rate, (int, float)):
        _fail("Subject-shape acquisition authority lacks exact FPS.")
    body: dict[str, Any] = {
        "schema_id": "palette.subject_body_frame_samples.source_binding",
        "schema_version": 1,
        "stage_id": "subject_shape",
        "run_name": publication.run_path.rsplit("/", 1)[-1],
        "run_path": publication.run_path,
        "source_schema_id": run.attrs.get("schema_id"),
        "source_schema_version": run.attrs.get("schema_version"),
        "publication_manifest_sha256": publication.manifest.record_sha256,
        "row_count": int(publication.row_identity.leading_dimension),
        "row_identity_sha256": publication.row_identity.record_sha256,
        "temporal_authority_sha256": temporal.record_sha256,
        "acquisition_camera_frame_sha256": acquisition.record_sha256,
        "recording_id": temporal.record.recording_id,
        "camera_id": temporal.record.camera_id,
        "source_total_frames": temporal.record.source_total_frames,
        "source_sample_rate_hz": float(source_rate),
        "body_frame_record_sha256": publication.body_frame.record_sha256,
        "heading_semantics_sha256": publication.heading_semantics.record_sha256,
        "origin_coordinate_descriptor_sha256": publication.descriptors[
            "body_frame/origin_xy"
        ].descriptor.digest(),
        "forward_coordinate_descriptor_sha256": publication.descriptors[
            "body_frame/forward_axis_xy"
        ].descriptor.digest(),
        "left_coordinate_descriptor_sha256": publication.descriptors[
            "body_frame/left_axis_xy"
        ].descriptor.digest(),
        "completion_snapshot": {
            "status": run.attrs.get("palette_run_completion_status"),
            "completed_at_utc": run.attrs.get("palette_run_completed_at_utc"),
            "selector_eligible": publication.selector_eligible,
        },
    }
    return _sealed(body)


def _join_authority(
    *,
    acquisition: BoundAcquisitionCameraFrame,
    track: BoundKinematicsSamplesSource,
    eye: Mapping[str, Any],
    shape: Mapping[str, Any],
) -> dict[str, Any]:
    record = acquisition.record
    source_rate = record.source_video_metadata.get("fps")
    if isinstance(source_rate, bool) or not isinstance(source_rate, (int, float)):
        _fail("Acquisition authority lacks exact source FPS.")
    expected = (
        record.recording_id,
        record.camera_id,
        int(record.source_total_frames),
        float(source_rate),
    )
    observed_shape = (
        shape["recording_id"],
        shape["camera_id"],
        int(shape["source_total_frames"]),
        float(shape["source_sample_rate_hz"]),
    )
    if observed_shape != expected:
        _fail(
            "Track/body sources disagree with acquisition recording, camera, or time."
        )
    if float(track.binding["source_sample_rate_hz"]) != expected[3]:
        _fail("Track kinematics and acquisition authority disagree on FPS.")
    if int(eye["frame_count"]) != expected[2]:
        _fail("Eye trace does not close the acquisition camera-frame axis.")
    body = {
        "schema_id": "palette.validated_behavior.cross_grain_join_authority",
        "schema_version": 1,
        "recording_id": record.recording_id,
        "camera_id": record.camera_id,
        "source_total_frames": int(record.source_total_frames),
        "source_sample_rate_hz": float(source_rate),
        "acquisition_camera_frame_ref": acquisition.record_ref,
        "acquisition_camera_frame_sha256": acquisition.record_sha256,
        "source_video_metadata_sha256": record.source_video_metadata_sha256,
        "join_rules": {
            "frame_key_alone_authoritative": False,
            "eye_to_acquisition": {
                "keys": ["recording_id", "source_acquisition_frame_index"],
                "cardinality": "one_eye_row_per_acquisition_frame",
            },
            "motion_to_eye": {
                "keys": ["recording_id", "source_acquisition_frame_index"],
                "cardinality": "many_tracks_to_zero_or_one_eye_frame",
            },
            "motion_to_body": {
                "keys": [
                    "recording_id",
                    "source_acquisition_frame_index",
                    "source_instance_key",
                ],
                "cardinality": "zero_or_one_motion_row_to_zero_or_one_body_observation",
                "requires_source_instance_key_valid": True,
            },
            "tail_to_body": {
                "keys": ["recording_id", "instance_key"],
                "cardinality": "many_tail_samples_to_one_body_observation",
            },
            "bout_to_motion": {
                "keys": ["recording_id", "track_id"],
                "interval_fields": [
                    "start_acquisition_frame_id",
                    "end_acquisition_frame_id",
                ],
                "cardinality": "one_bout_to_many_motion_samples_in_closed_interval",
            },
        },
    }
    return _sealed(body)


def _capability_binding(
    *,
    profile_id: str,
    source_binding: Mapping[str, Any],
    projection_contract: Mapping[str, Any],
    join_authority_sha256: str,
) -> dict[str, Any]:
    return {
        "profile_id": profile_id,
        "source_binding": _plain(source_binding),
        "projection_contract": _plain(projection_contract),
        "join_authority_sha256": join_authority_sha256,
    }


@dataclass(frozen=True)
class BoundCoreBehaviorCohortSources:
    """One fully admitted report and its exact five-grain source authorities."""

    report_path: Path
    report_binding: Mapping[str, Any]
    report: Mapping[str, Any]
    root: Any = field(repr=False, compare=False)
    acquisition: BoundAcquisitionCameraFrame = field(repr=False, compare=False)
    track: BoundKinematicsSamplesSource = field(repr=False, compare=False)
    eye: Mapping[str, Any]
    subject_shape: BoundSubjectShapeCoordinatePublication = field(
        repr=False, compare=False
    )
    subject_shape_binding: Mapping[str, Any]
    tail: BoundTailTraceSources = field(repr=False, compare=False)
    bouts: BoundActivitySpatialSources = field(repr=False, compare=False)
    join_authority: Mapping[str, Any]
    capability_bindings: Mapping[str, Mapping[str, Any]]
    core_authority_roster: Mapping[str, Any]
    bout_authority_identity: BoutAuthorityIdentity


def bind_core_behavior_cohort_sources(
    report_path: str | Path,
    *,
    expected_analysis_zarr: str | Path,
    expected_recording_id: str,
    export_profile_id: str = CORE_BEHAVIOR_EXPORT_PROFILE_ID,
) -> BoundCoreBehaviorCohortSources:
    """Resolve one report through all five strict scientific source binders."""

    selected_export_profile_id = _require_export_profile_id(export_profile_id)
    source_path = Path(expected_analysis_zarr).expanduser().resolve()
    binding, report = bind_core_behavior_execution_report(
        report_path,
        recording_id=expected_recording_id,
        analysis_zarr=source_path,
    )
    root = open_zarr_root(source_path, mode="r", use_consolidated=True)
    _ownership, acquisition = load_persisted_acquisition_camera_authority(root)
    if acquisition.record.recording_id != expected_recording_id:
        _fail("Acquisition authority binds another recording.")
    runs = report["runs"]
    track = bind_kinematics_samples_source(
        root,
        zarr_path=source_path,
        expected_recording_id=expected_recording_id,
        track_kinematics_run=runs["track_kinematics"]["run_name"],
        track_scope="offline",
        source_surface_profile_id=(
            CORE_MOTION_SOURCE_SURFACE_PROFILE_ID
            if selected_export_profile_id == CORE_BEHAVIOR_EXPORT_PROFILE_ID
            else KINEMATICS_SOURCE_SURFACE_PROFILE_ID
        ),
    )
    if track.binding["run_path"] != runs["track_kinematics"]["run_path"]:
        _fail("Strict track binder resolved another execution-report run.")
    eye = bind_eye_trace_source(
        root,
        zarr_path=source_path,
        expected_recording_id=expected_recording_id,
        eye_angle_run=runs["eye_angles"]["run_name"],
    )
    if eye["run_path"] != runs["eye_angles"]["run_path"]:
        _fail("Strict eye binder resolved another execution-report run.")
    tail = bind_tail_trace_sources(
        root,
        zarr_path=source_path,
        tail_kinematics_run=runs["tail_kinematics"]["run_name"],
        subject_shape_run=runs["subject_shape"]["run_name"],
        track_kinematics_run=runs["track_kinematics"]["run_name"],
        track_scope="offline",
        prebound_track_source=track,
    )
    if tail.binding["tail_run_path"] != runs["tail_kinematics"]["run_path"]:
        _fail("Strict tail binder resolved another execution-report run.")
    shape_publication = tail.subject_shape_publication
    if shape_publication is None:  # pragma: no cover - maintained binder invariant
        _fail("Tail binder did not retain its strict subject-shape authority.")
    if shape_publication.run_path != runs["subject_shape"]["run_path"]:
        _fail("Strict subject-shape binder resolved another execution-report run.")
    shape_binding = _subject_shape_binding(shape_publication)
    track_ids = [int(item["track_id"]) for item in track.binding["tracks"]]
    if len(track_ids) != 1:
        _fail(
            "Core-behavior report v3 names one swim-bout run and therefore requires "
            "exactly one track; a multi-track report needs an explicit run map."
        )
    bouts = bind_activity_spatial_sources(
        root,
        zarr_path=source_path,
        recording_id=expected_recording_id,
        track_kinematics_run=runs["track_kinematics"]["run_name"],
        track_scope="offline",
        swim_bout_runs_by_track={track_ids[0]: runs["swim_bouts"]["run_name"]},
        prebound_track_source=track,
    )
    bout_binding = bouts.bout_sources[track_ids[0]].binding
    if bout_binding["run_path"] != runs["swim_bouts"]["run_path"]:
        _fail("Strict swim-bout binder resolved another execution-report run.")
    if (
        tail.binding["track_source_binding"] != track.binding
        or bouts.track_source.binding != track.binding
    ):
        _fail("Tail or bout source binds another track-kinematics authority.")

    join = _join_authority(
        acquisition=acquisition,
        track=track,
        eye=eye,
        shape=shape_binding,
    )
    join_sha = str(join["payload_sha256"])
    try:
        temporal_policy = TemporalPolicy.from_mapping(
            _mapping(report["temporal_policy"], field_name="temporal_policy")
        )
    except ValueError as exc:  # pragma: no cover - admission validates first
        _fail(f"Core-behavior report temporal policy is invalid: {exc}")
    source_sample_rate_hz = float(track.binding["source_sample_rate_hz"])
    requested_rate = temporal_policy.kinematics_export_rate_hz(
        source_sample_rate_hz=source_sample_rate_hz
    )
    if selected_export_profile_id == CORE_BEHAVIOR_EXPORT_PROFILE_ID_V1:
        kinematics_profile_id = "kinematics_samples_v1"
        kinematics_projection = kinematics_projection_contract(
            source_sample_rate_hz=source_sample_rate_hz,
            requested_sample_rate_hz=requested_rate,
        )
    else:
        kinematics_profile_id = CORE_MOTION_PROJECTION_PROFILE_ID
        kinematics_projection = core_motion_projection_contract(
            source_sample_rate_hz=source_sample_rate_hz,
            requested_sample_rate_hz=requested_rate,
            arrow_schema_sha256=KINEMATICS_SAMPLES.payload_sha256,
        )
    capability_bindings: dict[str, Mapping[str, Any]] = {
        CROSS_GRAIN_JOIN_AUTHORITY: join,
        KINEMATICS_SAMPLES_CAPABILITY: _capability_binding(
            profile_id=kinematics_profile_id,
            source_binding=track.binding,
            projection_contract=kinematics_projection,
            join_authority_sha256=join_sha,
        ),
        SUBJECT_BODY_FRAME_CAPABILITY: _capability_binding(
            profile_id="subject_body_frame_samples_v1",
            source_binding=shape_binding,
            projection_contract=subject_body_frame_projection_contract(),
            join_authority_sha256=join_sha,
        ),
        EYE_TRACE_CAPABILITY: _capability_binding(
            profile_id="eye_trace_samples_v1",
            source_binding=eye,
            projection_contract=eye_trace_projection_contract(),
            join_authority_sha256=join_sha,
        ),
        TAIL_TRACE_CAPABILITY: _capability_binding(
            profile_id="tail_trace_samples_v1",
            source_binding=tail.binding,
            projection_contract=tail_trace_projection_contract(),
            join_authority_sha256=join_sha,
        ),
        CANONICAL_SWIM_BOUTS_CAPABILITY: _capability_binding(
            profile_id="canonical_swim_bouts_v1",
            source_binding=bout_binding,
            projection_contract=canonical_swim_bout_projection_contract(),
            join_authority_sha256=join_sha,
        ),
    }
    authority_roster = build_core_authority_roster(
        recording_id=expected_recording_id,
        analysis_zarr=source_path,
        execution_report_binding=binding,
        capability_bindings=capability_bindings,
    )
    bout_identity = core_roster_bout_identity(
        authority_roster,
        bout_source=bouts.bout_sources[track_ids[0]],
        track_binding=track.binding,
    )
    return BoundCoreBehaviorCohortSources(
        report_path=Path(report_path).expanduser().resolve(),
        report_binding=binding,
        report=report,
        root=root,
        acquisition=acquisition,
        track=track,
        eye=eye,
        subject_shape=shape_publication,
        subject_shape_binding=shape_binding,
        tail=tail,
        bouts=bouts,
        join_authority=join,
        capability_bindings=capability_bindings,
        core_authority_roster=authority_roster,
        bout_authority_identity=bout_identity,
    )


def _complete_member(
    report_path: str | Path,
    *,
    membership_member: Mapping[str, Any],
    export_profile_id: str,
) -> dict[str, Any]:
    receipts = list(membership_member["admission_receipts"])
    if (
        len(receipts) != 1
        or receipts[0]["role"] != CORE_BEHAVIOR_EXECUTION_ADMISSION_ROLE
    ):
        _fail("Core-behavior bundle requires one execution-report admission receipt.")
    bound = bind_core_behavior_cohort_sources(
        report_path,
        expected_analysis_zarr=membership_member["analysis_zarr"],
        expected_recording_id=membership_member["recording_id"],
        export_profile_id=export_profile_id,
    )
    if _plain(bound.report_binding) != _plain(receipts[0]):
        _fail("Core-behavior bundle path differs from its admission receipt.")
    capabilities = {
        key: {
            "state": "complete",
            "reason_code": None,
            "detail": None,
            "binding": _plain(bound.capability_bindings[key]),
        }
        for key in CORE_BEHAVIOR_CAPABILITY_KEYS
    }
    inventory = {
        "execution_report": _plain(bound.report_binding),
        "capability_bindings": _plain(bound.capability_bindings),
    }
    return {
        "recording_id": membership_member["recording_id"],
        "bundle_state": "complete",
        "reason_code": None,
        "bundle": {
            "adapter_id": CORE_BEHAVIOR_BUNDLE_ADAPTER_ID,
            "path": str(bound.report_path),
            "file_sha256": bound.report_binding["file_sha256"],
            "record_sha256": bound.report["record_sha256"],
            "schema_id": CORE_BEHAVIOR_EXECUTION_SCHEMA_ID,
            "schema_version": CORE_BEHAVIOR_EXECUTION_SCHEMA_VERSION,
            "method_id": CORE_BEHAVIOR_BUNDLE_METHOD_ID,
            "status": CORE_BEHAVIOR_BUNDLE_STATUS,
            "receipt_bindings": [_plain(bound.report_binding)],
            "binding_inventory_sha256": canonical_json_sha256(inventory),
        },
        "capabilities": capabilities,
    }


def _nonadmitted_member(member: Mapping[str, Any]) -> dict[str, Any]:
    state = str(member["membership_state"])
    if state == "admitted":
        _fail("Admitted core-behavior members require one complete report.")
    if state == "invalid":
        capability_state = "invalid"
        reason = "invalid_source_authority"
    elif state == "excluded":
        capability_state = "inapplicable"
        reason = "member_not_admitted"
    else:
        capability_state = "unavailable"
        reason = "blocked_by_unavailable_membership"
    return {
        "recording_id": member["recording_id"],
        "bundle_state": state,
        "reason_code": member["reason_code"],
        "bundle": None,
        "capabilities": {
            key: {
                "state": capability_state,
                "reason_code": reason,
                "detail": member["disposition_evidence"]["detail"],
                "binding": None,
            }
            for key in CORE_BEHAVIOR_CAPABILITY_KEYS
        },
    }


def _bundle_profile(
    capability_contract: Mapping[str, Any],
    *,
    export_profile_id: str,
) -> dict[str, Any]:
    return {
        "adapter_id": CORE_BEHAVIOR_BUNDLE_ADAPTER_ID,
        "bundle_schema_id": CORE_BEHAVIOR_EXECUTION_SCHEMA_ID,
        "bundle_schema_version": CORE_BEHAVIOR_EXECUTION_SCHEMA_VERSION,
        "bundle_method_id": CORE_BEHAVIOR_BUNDLE_METHOD_ID,
        "bundle_status": CORE_BEHAVIOR_BUNDLE_STATUS,
        "capability_contract_sha256": capability_contract["record_sha256"],
        "export_profile_id": _require_export_profile_id(export_profile_id),
        "publication_surface": "validated_behavior/v1",
        "join_authority_profile": "cross_grain_join_authority_v1",
    }


def build_bundle_set_from_core_behavior_execution_reports(
    *,
    bundle_set_id: str,
    membership: Mapping[str, Any],
    membership_path: str | Path,
    report_paths_by_recording: Mapping[str, str | Path],
    bundle_root: str | Path,
    palette_commit: str,
    created_at_utc: str,
    export_profile_id: str = CORE_BEHAVIOR_EXPORT_PROFILE_ID,
) -> dict[str, Any]:
    """Adapt completed execution reports into the generic bundle-set schema."""

    validated_membership = validate_membership_current_sources(membership)
    membership_file = Path(membership_path).expanduser().resolve()
    if not membership_file.is_file():
        raise FileNotFoundError(
            f"Membership manifest does not exist: {membership_file}"
        )
    admitted = {
        member["recording_id"]
        for member in validated_membership["members"]
        if member["membership_state"] == "admitted"
    }
    if set(report_paths_by_recording) != admitted:
        _fail("Report paths must name every and only admitted membership record.")
    selected_export_profile_id = _require_export_profile_id(export_profile_id)
    members = [
        (
            _complete_member(
                report_paths_by_recording[member["recording_id"]],
                membership_member=member,
                export_profile_id=selected_export_profile_id,
            )
            if member["membership_state"] == "admitted"
            else _nonadmitted_member(member)
        )
        for member in validated_membership["members"]
    ]
    contract = core_behavior_capability_contract(selected_export_profile_id)
    return build_validated_behavior_bundle_set(
        bundle_set_id=bundle_set_id,
        membership=validated_membership,
        membership_path=membership_file,
        membership_file_sha256=sha256_file(membership_file),
        bundle_root=bundle_root,
        bundle_profile=_bundle_profile(
            contract,
            export_profile_id=selected_export_profile_id,
        ),
        capability_contract=contract,
        members=members,
        palette_commit=palette_commit,
        created_at_utc=created_at_utc,
    )


def validate_core_behavior_bundle_set_current_sources(
    value: object,
    *,
    membership: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Re-open every exact report-selected source and compare all bindings."""

    validated_membership = validate_membership_current_sources(membership)
    bundle_set = validate_validated_behavior_bundle_set(
        value,
        membership=validated_membership,
    )
    bundle_profile = _mapping(bundle_set["bundle_profile"], field_name="bundle_profile")
    export_profile_id = _require_export_profile_id(
        bundle_profile.get("export_profile_id")
    )
    contract = core_behavior_capability_contract(export_profile_id)
    if _plain(bundle_profile) != _bundle_profile(
        contract,
        export_profile_id=export_profile_id,
    ):
        _fail("Bundle set is not the installed core-behavior report profile.")
    if _plain(bundle_set["capability_contract"]) != _plain(contract):
        _fail("Core-behavior bundle set uses another capability contract.")
    members_by_id = {
        member["recording_id"]: member for member in validated_membership["members"]
    }
    for item in bundle_set["members"]:
        if item["bundle_state"] != "complete":
            continue
        rebuilt = _complete_member(
            item["bundle"]["path"],
            membership_member=members_by_id[item["recording_id"]],
            export_profile_id=export_profile_id,
        )
        if _plain(rebuilt["bundle"]) != _plain(item["bundle"]) or _plain(
            rebuilt["capabilities"]
        ) != _plain(item["capabilities"]):
            _fail("Current core-behavior sources differ from their bundle binding.")
    return bundle_set


__all__ = [
    "CANONICAL_SWIM_BOUTS_CAPABILITY",
    "CORE_BEHAVIOR_BUNDLE_ADAPTER_ID",
    "CORE_BEHAVIOR_CAPABILITY_KEYS",
    "CORE_BEHAVIOR_EXPORT_PROFILE_ID",
    "CORE_BEHAVIOR_EXPORT_PROFILE_ID_V1",
    "CROSS_GRAIN_JOIN_AUTHORITY",
    "CoreBehaviorCohortAdapterError",
    "EYE_TRACE_CAPABILITY",
    "KINEMATICS_SAMPLES_CAPABILITY",
    "SUBJECT_BODY_FRAME_CAPABILITY",
    "SUPPORTED_CORE_BEHAVIOR_EXPORT_PROFILE_IDS",
    "TAIL_TRACE_CAPABILITY",
    "BoundCoreBehaviorCohortSources",
    "bind_core_behavior_cohort_sources",
    "build_bundle_set_from_core_behavior_execution_reports",
    "canonical_swim_bout_projection_contract",
    "core_behavior_capability_contract",
    "subject_body_frame_projection_contract",
    "validate_core_behavior_bundle_set_current_sources",
]
