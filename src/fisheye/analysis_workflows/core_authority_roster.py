"""One fail-closed authority interface for core behavior and paradigm consumers.

The core workflow already resolves each scientific source through its strict
profile-specific loader.  This module does not discover those sources again.
It normalizes the resulting bindings into one sealed roster that downstream
paradigm planners can consume without learning the individual source grammars.

It also defines the normalized swim-bout identity used when a legacy paradigm
bundle and the selected core roster both claim ``canonical_swim_bouts``.  Raw
profile rows are deliberately not compared: equality requires the complete
normalized authority record, including the selected event payload.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from fisheye.analysis._exact_tabular_run_schema import MANIFEST_ATTRIBUTE
from fisheye.analysis.swim_bout_frame_axis import canonical_frame_axis_sha256
from fisheye.analysis.swim_bout_io import (
    SwimBoutTables,
    load_exact_selector_ineligible_default_swim_bout_tables,
)
from fisheye.analytics_exports.validated_behavior_core_behavior_contracts import (
    CANONICAL_SWIM_BOUTS_CAPABILITY,
    CORE_BEHAVIOR_CAPABILITY_KEYS,
    CROSS_GRAIN_JOIN_AUTHORITY,
    SUBJECT_BODY_FRAME_CAPABILITY,
    SUBJECT_BODY_FRAME_SOURCE_PROFILE_ID,
)
from fisheye.analytics_exports.kinematics_samples import (
    BoundKinematicsSamplesSource,
    CORE_MOTION_SOURCE_SURFACE_PROFILE_ID,
    bind_kinematics_samples_source,
)
from fisheye.analytics_exports.activity_spatial_time_bins import (
    BoundActivitySpatialSources,
    bind_activity_spatial_sources,
)
from fisheye.shared.coordinate_frame_record import array_values_sha256
from fisheye.shared.subject_shape_coordinate_publication import (
    BoundSubjectShapeCoordinatePublication,
    load_persisted_subject_shape_coordinate_publication,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr_io import open_zarr_root

from .validated_behavior_source_admission import (
    CORE_BEHAVIOR_EXECUTION_ADMISSION_ROLE,
    CORE_BEHAVIOR_EXECUTION_SCHEMA_ID,
    CORE_BEHAVIOR_EXECUTION_SCHEMA_VERSION,
)

CORE_AUTHORITY_ROSTER_SCHEMA_ID = "palette.core_behavior.authority_roster"
CORE_AUTHORITY_ROSTER_SCHEMA_VERSION = 1
BOUT_AUTHORITY_IDENTITY_SCHEMA_ID = "palette.core_behavior.bout_authority_identity"
BOUT_AUTHORITY_IDENTITY_SCHEMA_VERSION = 1
BOUT_AUTHORITY_COMPARISON_SCHEMA_ID = "palette.core_behavior.bout_authority_comparison"
BOUT_AUTHORITY_COMPARISON_SCHEMA_VERSION = 1
CORE_AUTHORITY_CONSUMPTION_SCHEMA_ID = (
    "palette.core_behavior.authority_consumption_receipt"
)
CORE_AUTHORITY_CONSUMPTION_SCHEMA_VERSION = 1

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_CAPABILITY_BINDING_FIELDS = frozenset(
    {
        "profile_id",
        "source_binding",
        "projection_contract",
        "join_authority_sha256",
    }
)
_ROSTER_FIELDS = frozenset(
    {
        "schema_id",
        "schema_version",
        "recording_id",
        "analysis_zarr",
        "execution_report_binding",
        "cross_grain_join_authority",
        "capability_bindings",
        "record_sha256",
    }
)
_BOUT_IDENTITY_FIELDS = frozenset(
    {
        "schema_id",
        "schema_version",
        "recording_id",
        "analysis_zarr",
        "publication",
        "motion_authority",
        "frame_axis",
        "selection",
        "selected_events",
        "source_binding_sha256",
        "record_sha256",
    }
)
_BOUT_PUBLICATION_FIELDS = frozenset(
    {
        "run_path",
        "schema_id",
        "schema_version",
        "layout",
        "completion_status",
        "selector_eligible",
        "array_manifest_sha256",
    }
)
_BOUT_MOTION_FIELDS = frozenset(
    {
        "scope",
        "run_path",
        "manifest_sha256",
        "verification_sha256",
        "track_id",
        "track_row_start",
        "track_row_stop",
    }
)
_BOUT_FRAME_AXIS_FIELDS = frozenset(
    {
        "source_sample_rate_hz",
        "count",
        "first_frame",
        "last_frame",
        "canonical_frame_axis_sha256",
        "array_values_sha256",
    }
)
_BOUT_SELECTION_FIELDS = frozenset(
    {
        "candidate_id",
        "candidate_name",
        "signal_id",
        "signal_name",
        "signal_level",
        "signal_role",
        "signal_source_level",
    }
)
_BOUT_EVENT_FIELDS = frozenset({"dtype", "count", "content_sha256"})
_CORE_CONSUMPTION_FIELDS = frozenset(
    {
        "schema_id",
        "schema_version",
        "consumer_id",
        "recording_id",
        "analysis_zarr",
        "core_authority_roster_sha256",
        "required_capabilities",
        "capability_binding_digests",
        "selected_track_id",
        "record_sha256",
    }
)
_BOUND_CORE_MOTION_SEAL = object()


class CoreAuthorityRosterError(ValueError):
    """A core roster or normalized authority record is absent or inconsistent."""


def _fail(message: str) -> None:
    raise CoreAuthorityRosterError(message)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{label} must be one object.")
    return value


def _text(value: object, *, label: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        _fail(f"{label} must be non-empty normalized text.")
    return value


def _digest(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        _fail(f"{label} must be one lowercase SHA-256 digest.")
    return value


def _int(value: object, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        _fail(f"{label} must be an integer >= {minimum}.")
    return value


def _number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{label} must be numeric.")
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        _fail(f"{label} must be positive and finite.")
    return number


def _verify_sealed_mapping(
    value: object,
    *,
    label: str,
    digest_field: str = "payload_sha256",
) -> dict[str, Any]:
    record = _plain(_mapping(value, label=label))
    digest = _digest(record.get(digest_field), label=f"{label} {digest_field}")
    body = {key: item for key, item in record.items() if key != digest_field}
    if canonical_json_sha256(body) != digest:
        _fail(f"{label} {digest_field} is stale.")
    return record


def build_subject_body_frame_source_binding(
    publication: BoundSubjectShapeCoordinatePublication,
) -> dict[str, Any]:
    """Normalize one strict subject-shape publication for the core roster."""

    if type(publication) is not BoundSubjectShapeCoordinatePublication:
        _fail("Subject body-frame binding requires one strict publication.")
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
    return {**body, "payload_sha256": canonical_json_sha256(body)}


def _validated_capability_bindings(
    value: object,
) -> tuple[dict[str, Any], dict[str, Any]]:
    bindings = _plain(_mapping(value, label="core capability bindings"))
    expected_keys = set(CORE_BEHAVIOR_CAPABILITY_KEYS)
    if set(bindings) != expected_keys:
        _fail(
            "Core capability roster is not closed: "
            f"missing={sorted(expected_keys - set(bindings))!r}, "
            f"unexpected={sorted(set(bindings) - expected_keys)!r}."
        )
    join = _verify_sealed_mapping(
        bindings[CROSS_GRAIN_JOIN_AUTHORITY],
        label="cross-grain join authority",
    )
    join_digest = _digest(
        join["payload_sha256"], label="cross-grain join authority digest"
    )
    normalized: dict[str, Any] = {CROSS_GRAIN_JOIN_AUTHORITY: join}
    for capability in CORE_BEHAVIOR_CAPABILITY_KEYS:
        if capability == CROSS_GRAIN_JOIN_AUTHORITY:
            continue
        binding = _plain(
            _mapping(bindings[capability], label=f"{capability} capability binding")
        )
        if set(binding) != _CAPABILITY_BINDING_FIELDS:
            _fail(f"{capability} capability binding is not a closed record.")
        _text(binding.get("profile_id"), label=f"{capability} profile ID")
        source = _verify_sealed_mapping(
            binding.get("source_binding"),
            label=f"{capability} source binding",
        )
        projection = _verify_sealed_mapping(
            binding.get("projection_contract"),
            label=f"{capability} projection contract",
        )
        if binding.get("join_authority_sha256") != join_digest:
            _fail(f"{capability} binds another cross-grain join authority.")
        normalized[capability] = {
            "profile_id": binding["profile_id"],
            "source_binding": source,
            "projection_contract": projection,
            "join_authority_sha256": join_digest,
        }
    return normalized, join


def build_core_authority_roster(
    *,
    recording_id: str,
    analysis_zarr: str | Path,
    execution_report_binding: Mapping[str, Any],
    capability_bindings: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Seal one already-resolved core capability set for downstream consumers."""

    selected_recording = _text(recording_id, label="recording ID")
    source = Path(analysis_zarr).expanduser().resolve()
    report = _plain(
        _mapping(execution_report_binding, label="execution report binding")
    )
    if set(report) != {
        "role",
        "path",
        "file_sha256",
        "record_sha256",
        "schema_id",
        "schema_version",
    }:
        _fail("Execution-report binding is not a closed admission receipt.")
    if report.get("role") != CORE_BEHAVIOR_EXECUTION_ADMISSION_ROLE:
        _fail("Execution-report admission role is not the core workflow role.")
    _text(report.get("path"), label="execution-report path")
    _digest(report.get("file_sha256"), label="execution-report file digest")
    _digest(report.get("record_sha256"), label="execution-report record digest")
    if (
        report.get("schema_id") != CORE_BEHAVIOR_EXECUTION_SCHEMA_ID
        or report.get("schema_version") != CORE_BEHAVIOR_EXECUTION_SCHEMA_VERSION
    ):
        _fail("Execution-report admission schema is unsupported.")

    normalized, join = _validated_capability_bindings(capability_bindings)
    if join.get("recording_id") != selected_recording:
        _fail("Cross-grain join authority binds another recording.")
    for capability, raw in normalized.items():
        if capability == CROSS_GRAIN_JOIN_AUTHORITY:
            continue
        binding = raw["source_binding"]
        bound_recording = binding.get("recording_id")
        if bound_recording is not None and bound_recording != selected_recording:
            _fail(f"{capability} source binding names another recording.")
        bound_path = binding.get("zarr_path")
        if bound_path is not None and Path(str(bound_path)).resolve() != source:
            _fail(f"{capability} source binding names another analysis Zarr.")

    body = {
        "schema_id": CORE_AUTHORITY_ROSTER_SCHEMA_ID,
        "schema_version": CORE_AUTHORITY_ROSTER_SCHEMA_VERSION,
        "recording_id": selected_recording,
        "analysis_zarr": str(source),
        "execution_report_binding": report,
        "cross_grain_join_authority": join,
        "capability_bindings": normalized,
    }
    return {**body, "record_sha256": canonical_json_sha256(body)}


def validate_core_authority_roster(value: object) -> Mapping[str, Any]:
    """Validate a persisted roster without resolving any source a second time."""

    record = _plain(_mapping(value, label="core authority roster"))
    if set(record) != _ROSTER_FIELDS:
        _fail("Core authority roster field set is not exact.")
    if (
        record.get("schema_id") != CORE_AUTHORITY_ROSTER_SCHEMA_ID
        or record.get("schema_version") != CORE_AUTHORITY_ROSTER_SCHEMA_VERSION
    ):
        _fail("Core authority roster schema is unsupported.")
    digest = _digest(record.get("record_sha256"), label="core authority roster digest")
    body = {key: item for key, item in record.items() if key != "record_sha256"}
    if canonical_json_sha256(body) != digest:
        _fail("Core authority roster digest is stale.")
    rebuilt = build_core_authority_roster(
        recording_id=record["recording_id"],
        analysis_zarr=record["analysis_zarr"],
        execution_report_binding=record["execution_report_binding"],
        capability_bindings=record["capability_bindings"],
    )
    if rebuilt != record:
        _fail("Core authority roster does not normalize to its sealed identity.")
    return MappingProxyType(record)


@dataclass(frozen=True)
class BoutAuthorityIdentity:
    """One normalized, fully proven selected swim-bout authority."""

    record: Mapping[str, Any]

    @property
    def record_sha256(self) -> str:
        return str(self.record["record_sha256"])


@dataclass(frozen=True)
class BoutAuthorityComparison:
    """Typed result of comparing two normalized bout authority identities."""

    status: str
    reason_code: str
    differing_fields: tuple[str, ...]
    selected_event_content_equal: bool | None
    record: Mapping[str, Any]


def build_core_authority_consumption_receipt(
    roster: Mapping[str, Any],
    *,
    consumer_id: str,
    required_capabilities: Sequence[str],
    selected_track_id: int,
) -> dict[str, Any]:
    """Bind one downstream computation to an exact subset of the core roster."""

    validated = validate_core_authority_roster(roster)
    consumer = _text(consumer_id, label="core-authority consumer ID")
    requested = tuple(sorted(set(required_capabilities)))
    if not requested or len(requested) != len(tuple(required_capabilities)):
        _fail("Required core capabilities must be a non-empty unique sequence.")
    known = set(CORE_BEHAVIOR_CAPABILITY_KEYS)
    if not set(requested).issubset(known):
        _fail(
            "Core-authority consumer requests unknown capabilities: "
            f"{sorted(set(requested) - known)!r}."
        )
    if CROSS_GRAIN_JOIN_AUTHORITY not in requested:
        _fail("Every paradigm dependency must consume the cross-grain join authority.")
    track_id = _int(selected_track_id, label="selected core track ID")
    capabilities = validated["capability_bindings"]
    track_source = capabilities["kinematics_samples"]["source_binding"]
    track_ids = tuple(int(item["track_id"]) for item in track_source["tracks"])
    if track_ids.count(track_id) != 1:
        _fail("Selected core track ID does not resolve exactly once in the roster.")

    digests: dict[str, Any] = {}
    for capability in requested:
        binding = capabilities[capability]
        if capability == CROSS_GRAIN_JOIN_AUTHORITY:
            digests[capability] = {
                "profile_id": "cross_grain_join_authority_v1",
                "source_binding_sha256": binding["payload_sha256"],
                "projection_contract_sha256": None,
                "join_authority_sha256": binding["payload_sha256"],
            }
        else:
            digests[capability] = {
                "profile_id": binding["profile_id"],
                "source_binding_sha256": binding["source_binding"]["payload_sha256"],
                "projection_contract_sha256": binding["projection_contract"][
                    "payload_sha256"
                ],
                "join_authority_sha256": binding["join_authority_sha256"],
            }
    body = {
        "schema_id": CORE_AUTHORITY_CONSUMPTION_SCHEMA_ID,
        "schema_version": CORE_AUTHORITY_CONSUMPTION_SCHEMA_VERSION,
        "consumer_id": consumer,
        "recording_id": validated["recording_id"],
        "analysis_zarr": validated["analysis_zarr"],
        "core_authority_roster_sha256": validated["record_sha256"],
        "required_capabilities": list(requested),
        "capability_binding_digests": digests,
        "selected_track_id": track_id,
    }
    return {**body, "record_sha256": canonical_json_sha256(body)}


def validate_core_authority_consumption_receipt(
    value: object,
    *,
    roster: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Prove a consumer receipt was derived from this exact current roster."""

    record = _plain(_mapping(value, label="core-authority consumption receipt"))
    if set(record) != _CORE_CONSUMPTION_FIELDS:
        _fail("Core-authority consumption receipt field set is not exact.")
    if (
        record.get("schema_id") != CORE_AUTHORITY_CONSUMPTION_SCHEMA_ID
        or record.get("schema_version") != CORE_AUTHORITY_CONSUMPTION_SCHEMA_VERSION
    ):
        _fail("Core-authority consumption receipt schema is unsupported.")
    digest = _digest(
        record.get("record_sha256"), label="core-authority consumption digest"
    )
    body = {key: item for key, item in record.items() if key != "record_sha256"}
    if canonical_json_sha256(body) != digest:
        _fail("Core-authority consumption receipt digest is stale.")
    required = record.get("required_capabilities")
    if not isinstance(required, list) or any(
        type(item) is not str for item in required
    ):
        _fail("Core-authority consumption capability list is invalid.")
    rebuilt = build_core_authority_consumption_receipt(
        roster,
        consumer_id=record["consumer_id"],
        required_capabilities=required,
        selected_track_id=record["selected_track_id"],
    )
    if rebuilt != record:
        _fail("Core-authority consumption receipt binds another roster or selection.")
    return MappingProxyType(record)


@dataclass(frozen=True, init=False, eq=False)
class BoundCoreMotionAndBouts:
    """The roster-selected core motion and bout authorities for one consumer."""

    roster: Mapping[str, Any]
    root: Any = field(repr=False, compare=False)
    track: BoundKinematicsSamplesSource = field(repr=False, compare=False)
    bouts: BoundActivitySpatialSources = field(repr=False, compare=False)
    bout_identities: Mapping[int, BoutAuthorityIdentity]
    _verification_seal: object = field(repr=False, compare=False)

    def __init__(self, *, _verification_seal: object | None = None, **values: Any):
        if _verification_seal is not _BOUND_CORE_MOTION_SEAL:
            _fail(
                "Core motion/bout bindings can only be minted by the roster resolver."
            )
        for name, value in values.items():
            if name == "bout_identities":
                value = MappingProxyType(dict(value))
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_verification_seal", _BOUND_CORE_MOTION_SEAL)

    @property
    def roster_sha256(self) -> str:
        return str(self.roster["record_sha256"])

    @property
    def track_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self.bout_identities))


def bind_subject_body_frame_from_core_roster(
    bound: BoundCoreMotionAndBouts,
) -> BoundSubjectShapeCoordinatePublication:
    """Reopen the roster-selected body authority through its strict loader.

    The roster supplies the exact publication identity and the normal
    subject-shape loader proves that immutable publication still matches it.
    No selector or alternate body-frame grammar is consulted.
    """

    if (
        type(bound) is not BoundCoreMotionAndBouts
        or bound._verification_seal is not _BOUND_CORE_MOTION_SEAL
    ):
        raise TypeError("bound must be a resolver-minted BoundCoreMotionAndBouts.")
    validated = validate_core_authority_roster(bound.roster)
    capability = _mapping(
        validated["capability_bindings"][SUBJECT_BODY_FRAME_CAPABILITY],
        label="subject body-frame capability",
    )
    if capability.get("profile_id") != SUBJECT_BODY_FRAME_SOURCE_PROFILE_ID:
        _fail("Paradigm body consumers require the canonical subject-body profile.")
    expected = _mapping(
        capability.get("source_binding"),
        label="subject body-frame source binding",
    )
    run_path = _text(expected.get("run_path"), label="subject body-frame run path")
    publication = load_persisted_subject_shape_coordinate_publication(
        bound.root,
        run_path,
    )
    observed = build_subject_body_frame_source_binding(publication)
    if _plain(observed) != _plain(expected):
        _fail("Live subject body-frame source differs from the selected roster.")

    join = _mapping(
        validated["cross_grain_join_authority"],
        label="core cross-grain join authority",
    )
    expected_domain = (
        validated["recording_id"],
        join.get("camera_id"),
        join.get("source_total_frames"),
        join.get("source_sample_rate_hz"),
        join.get("acquisition_camera_frame_sha256"),
    )
    observed_domain = (
        observed.get("recording_id"),
        observed.get("camera_id"),
        observed.get("source_total_frames"),
        observed.get("source_sample_rate_hz"),
        observed.get("acquisition_camera_frame_sha256"),
    )
    if observed_domain != expected_domain:
        _fail("Subject body frame and core join authority name another frame domain.")
    return publication


def _dtype_record(dtype: np.dtype[Any]) -> Any:
    return _plain(dtype.descr if dtype.names else dtype.str)


def build_bout_authority_identity(
    *,
    recording_id: str,
    analysis_zarr: str | Path,
    run_path: str,
    run_schema_id: str,
    run_schema_version: int,
    run_layout: str,
    completion_status: str,
    selector_eligible: bool,
    array_manifest_sha256: str,
    motion_scope: str,
    motion_run_path: str,
    motion_manifest_sha256: str,
    motion_verification_sha256: str,
    track_id: int,
    track_row_start: int,
    track_row_stop: int,
    source_sample_rate_hz: float,
    frame_axis: np.ndarray,
    candidate_id: int,
    candidate_name: str,
    signal_id: int,
    signal_name: str,
    signal_level: str,
    signal_role: str,
    signal_source_level: str | None,
    selected_events: np.ndarray,
    source_binding_sha256: str,
) -> BoutAuthorityIdentity:
    """Build the one profile-neutral identity compared by composite planning."""

    frames = np.asarray(frame_axis)
    events = np.asarray(selected_events)
    if frames.dtype != np.dtype("int64") or frames.ndim != 1:
        _fail("Bout authority frame axis must be a one-dimensional int64 array.")
    if events.ndim != 1:
        _fail("Bout authority selected-event table must be one-dimensional.")
    start = _int(track_row_start, label="track row start")
    stop = _int(track_row_stop, label="track row stop")
    if stop < start or stop - start != int(frames.size):
        _fail("Bout authority track row bounds do not close the frame axis.")
    body = {
        "schema_id": BOUT_AUTHORITY_IDENTITY_SCHEMA_ID,
        "schema_version": BOUT_AUTHORITY_IDENTITY_SCHEMA_VERSION,
        "recording_id": _text(recording_id, label="recording ID"),
        "analysis_zarr": str(Path(analysis_zarr).expanduser().resolve()),
        "publication": {
            "run_path": _text(run_path, label="swim-bout run path"),
            "schema_id": _text(run_schema_id, label="swim-bout schema ID"),
            "schema_version": _int(
                run_schema_version, label="swim-bout schema version", minimum=1
            ),
            "layout": _text(run_layout, label="swim-bout layout"),
            "completion_status": _text(
                completion_status, label="swim-bout completion status"
            ),
            "selector_eligible": selector_eligible,
            "array_manifest_sha256": _digest(
                array_manifest_sha256, label="swim-bout array-manifest digest"
            ),
        },
        "motion_authority": {
            "scope": _text(motion_scope, label="track-motion scope"),
            "run_path": _text(motion_run_path, label="track-motion run path"),
            "manifest_sha256": _digest(
                motion_manifest_sha256, label="track-motion manifest digest"
            ),
            "verification_sha256": _digest(
                motion_verification_sha256,
                label="track-motion verification digest",
            ),
            "track_id": _int(track_id, label="track ID"),
            "track_row_start": start,
            "track_row_stop": stop,
        },
        "frame_axis": {
            "source_sample_rate_hz": _number(
                source_sample_rate_hz, label="source sample rate"
            ),
            "count": int(frames.size),
            "first_frame": int(frames[0]) if frames.size else None,
            "last_frame": int(frames[-1]) if frames.size else None,
            "canonical_frame_axis_sha256": canonical_frame_axis_sha256(frames),
            "array_values_sha256": array_values_sha256(frames),
        },
        "selection": {
            "candidate_id": _int(candidate_id, label="candidate ID"),
            "candidate_name": _text(candidate_name, label="candidate name"),
            "signal_id": _int(signal_id, label="signal ID"),
            "signal_name": _text(signal_name, label="signal name"),
            "signal_level": _text(signal_level, label="signal level"),
            "signal_role": _text(signal_role, label="signal role"),
            "signal_source_level": (
                None
                if signal_source_level is None
                else _text(signal_source_level, label="signal source level")
            ),
        },
        "selected_events": {
            "dtype": _dtype_record(events.dtype),
            "count": int(events.size),
            "content_sha256": array_values_sha256(events),
        },
        "source_binding_sha256": _digest(
            source_binding_sha256, label="swim-bout source-binding digest"
        ),
    }
    if type(selector_eligible) is not bool:
        _fail("Swim-bout selector eligibility must be one exact boolean.")
    record = {**body, "record_sha256": canonical_json_sha256(body)}
    return BoutAuthorityIdentity(record=MappingProxyType(record))


def validate_bout_authority_identity(value: object) -> BoutAuthorityIdentity:
    """Validate and normalize one serialized bout identity."""

    raw = value.record if isinstance(value, BoutAuthorityIdentity) else value
    record = _plain(_mapping(raw, label="bout authority identity"))
    if set(record) != _BOUT_IDENTITY_FIELDS:
        _fail("Bout authority identity field set is not exact.")
    if (
        record.get("schema_id") != BOUT_AUTHORITY_IDENTITY_SCHEMA_ID
        or record.get("schema_version") != BOUT_AUTHORITY_IDENTITY_SCHEMA_VERSION
    ):
        _fail("Bout authority identity schema is unsupported.")
    digest = _digest(record.get("record_sha256"), label="bout authority digest")
    body = {key: item for key, item in record.items() if key != "record_sha256"}
    if canonical_json_sha256(body) != digest:
        _fail("Bout authority identity digest is stale.")
    publication = _mapping(record["publication"], label="bout publication identity")
    motion = _mapping(record["motion_authority"], label="bout motion authority")
    axis = _mapping(record["frame_axis"], label="bout frame-axis identity")
    selection = _mapping(record["selection"], label="bout selection identity")
    events = _mapping(record["selected_events"], label="bout event identity")
    if set(publication) != _BOUT_PUBLICATION_FIELDS:
        _fail("Bout publication identity field set is not exact.")
    if set(motion) != _BOUT_MOTION_FIELDS:
        _fail("Bout motion-authority field set is not exact.")
    if set(axis) != _BOUT_FRAME_AXIS_FIELDS:
        _fail("Bout frame-axis identity field set is not exact.")
    if set(selection) != _BOUT_SELECTION_FIELDS:
        _fail("Bout selection identity field set is not exact.")
    if set(events) != _BOUT_EVENT_FIELDS:
        _fail("Bout selected-event identity field set is not exact.")
    _text(record.get("recording_id"), label="recording ID")
    _text(record.get("analysis_zarr"), label="analysis Zarr")
    _text(publication.get("run_path"), label="swim-bout run path")
    _text(publication.get("schema_id"), label="swim-bout schema ID")
    _int(publication.get("schema_version"), label="swim-bout schema version", minimum=1)
    _text(publication.get("layout"), label="swim-bout layout")
    if publication.get("completion_status") != "complete":
        _fail("Bout publication is not complete.")
    if type(publication.get("selector_eligible")) is not bool:
        _fail("Bout selector eligibility must be one exact boolean.")
    _digest(publication.get("array_manifest_sha256"), label="array manifest digest")
    _text(motion.get("scope"), label="track-motion scope")
    _text(motion.get("run_path"), label="track-motion run path")
    _digest(motion.get("manifest_sha256"), label="motion manifest digest")
    _digest(motion.get("verification_sha256"), label="motion verification digest")
    track_start = _int(motion.get("track_row_start"), label="track row start")
    track_stop = _int(motion.get("track_row_stop"), label="track row stop")
    _int(motion.get("track_id"), label="track ID")
    if track_stop < track_start:
        _fail("Bout track row bounds are reversed.")
    _number(axis.get("source_sample_rate_hz"), label="source sample rate")
    frame_count = _int(axis.get("count"), label="frame-axis count")
    if frame_count != track_stop - track_start:
        _fail("Bout frame-axis count does not close the track row bounds.")
    if frame_count == 0:
        if axis.get("first_frame") is not None or axis.get("last_frame") is not None:
            _fail("Empty bout frame axis must have null endpoints.")
    else:
        first_frame = _int(axis.get("first_frame"), label="first frame")
        last_frame = _int(axis.get("last_frame"), label="last frame")
        if last_frame < first_frame:
            _fail("Bout frame-axis endpoints are reversed.")
    _digest(axis.get("canonical_frame_axis_sha256"), label="frame-axis digest")
    _digest(axis.get("array_values_sha256"), label="frame-axis values digest")
    _int(selection.get("candidate_id"), label="candidate ID")
    _text(selection.get("candidate_name"), label="candidate name")
    _int(selection.get("signal_id"), label="signal ID")
    _text(selection.get("signal_name"), label="signal name")
    _text(selection.get("signal_level"), label="signal level")
    source_level = selection.get("signal_source_level")
    if source_level is not None:
        _text(source_level, label="signal source level")
    _int(events.get("count"), label="selected-event count")
    if not isinstance(events.get("dtype"), (str, list)):
        _fail("Selected-event dtype declaration is invalid.")
    _digest(events.get("content_sha256"), label="selected-event digest")
    _digest(record.get("source_binding_sha256"), label="source-binding digest")
    _text(selection.get("signal_role"), label="signal role")
    return BoutAuthorityIdentity(record=MappingProxyType(record))


def bout_authority_identity_from_core_source(
    *,
    recording_id: str,
    analysis_zarr: str | Path,
    bout_source: Any,
    track_binding: Mapping[str, Any],
) -> BoutAuthorityIdentity:
    """Normalize the strict selector-visible bout source in a core roster."""

    binding = _verify_sealed_mapping(
        bout_source.binding, label="core swim-bout source binding"
    )
    attrs = _plain(bout_source.events.run_attrs)
    manifest = _mapping(attrs.get(MANIFEST_ATTRIBUTE), label="swim-bout array manifest")
    manifest_sha = canonical_json_sha256(_plain(manifest))
    if binding.get("source_array_manifest_sha256") != manifest_sha:
        _fail("Core swim-bout binding names another array manifest.")
    tracks = [
        _mapping(item, label="track binding record")
        for item in _mapping(track_binding, label="track binding").get("tracks", [])
    ]
    track_id = _int(binding.get("track_id"), label="core swim-bout track ID")
    selected_tracks = [item for item in tracks if item.get("track_id") == track_id]
    if len(selected_tracks) != 1:
        _fail("Core swim-bout source does not resolve one exact track record.")
    track = selected_tracks[0]
    frame_axis = np.asarray(bout_source.frame_axis)
    events = np.asarray(bout_source.events.bouts)
    if binding.get("frame_axis_array_values_sha256") != array_values_sha256(frame_axis):
        _fail("Core swim-bout frame-axis receipt is stale.")
    if binding.get("bout_content_sha256") != array_values_sha256(events):
        _fail("Core swim-bout event receipt is stale.")
    motion_run_path = _text(track_binding.get("run_path"), label="core motion run path")
    return build_bout_authority_identity(
        recording_id=recording_id,
        analysis_zarr=analysis_zarr,
        run_path=binding["run_path"],
        run_schema_id=binding["source_schema_id"],
        run_schema_version=binding["source_schema_version"],
        run_layout=attrs.get("layout"),
        completion_status=binding["completion_snapshot"]["status"],
        selector_eligible=binding["completion_snapshot"]["selector_eligible"],
        array_manifest_sha256=manifest_sha,
        motion_scope=track_binding["scope"],
        motion_run_path=motion_run_path,
        motion_manifest_sha256=track_binding["source_manifest_sha256"],
        motion_verification_sha256=track_binding["source_publication_commit_sha256"],
        track_id=track_id,
        track_row_start=0,
        track_row_stop=int(track["sample_count"]),
        source_sample_rate_hz=binding["source_sample_rate_hz"],
        frame_axis=frame_axis,
        candidate_id=binding["candidate_id"],
        candidate_name=binding["candidate_name"],
        signal_id=binding["signal_id"],
        signal_name=binding["signal_name"],
        signal_level=binding["speed_level"],
        signal_role=bout_source.events.signal.role,
        signal_source_level=bout_source.events.signal.source_level,
        selected_events=events,
        source_binding_sha256=binding["payload_sha256"],
    )


def _phase_c_binding_digest(binding: Mapping[str, Any]) -> str:
    record = _plain(binding)
    digest = _digest(record.get("sha256"), label="Phase-C swim-bout binding digest")
    body = {key: value for key, value in record.items() if key != "sha256"}
    if canonical_json_sha256(body) != digest:
        _fail("Phase-C swim-bout binding digest is stale.")
    return digest


def bout_authority_identity_from_phase_c_source(
    *,
    root: Any,
    recording_id: str,
    analysis_zarr: str | Path,
    binding: Mapping[str, Any],
    source_sample_rate_hz: float,
) -> BoutAuthorityIdentity:
    """Reopen and normalize the exact selector-ineligible Phase-C bout source."""

    source_binding = _plain(_mapping(binding, label="Phase-C swim-bout binding"))
    binding_sha = _phase_c_binding_digest(source_binding)
    run_path = _text(source_binding.get("run_path"), label="Phase-C bout run path")
    prefix = "analysis/swim_bout_runs/"
    if not run_path.startswith(prefix) or "/" in run_path[len(prefix) :]:
        _fail("Phase-C swim-bout binding does not name one exact run.")
    tables = load_exact_selector_ineligible_default_swim_bout_tables(
        root, run_name=run_path[len(prefix) :]
    )
    return bout_authority_identity_from_phase_c_tables(
        recording_id=recording_id,
        analysis_zarr=analysis_zarr,
        binding=source_binding,
        tables=tables,
        source_binding_sha256=binding_sha,
        source_sample_rate_hz=source_sample_rate_hz,
    )


def bout_authority_identity_from_phase_c_tables(
    *,
    recording_id: str,
    analysis_zarr: str | Path,
    binding: Mapping[str, Any],
    tables: SwimBoutTables,
    source_sample_rate_hz: float,
    source_binding_sha256: str | None = None,
) -> BoutAuthorityIdentity:
    """Normalize already-loaded Phase-C tables after strict source validation."""

    source_binding = _plain(_mapping(binding, label="Phase-C swim-bout binding"))
    binding_sha = source_binding_sha256 or _phase_c_binding_digest(source_binding)
    attrs = _plain(tables.run_attrs)
    manifest = _mapping(attrs.get(MANIFEST_ATTRIBUTE), label="swim-bout array manifest")
    frame_axis = np.asarray(tables.series.get("frame_indices"))
    events = np.asarray(tables.bouts)
    track_id = _int(source_binding.get("track_id"), label="Phase-C track ID")
    run_name = _text(
        attrs.get("source_track_kinematics_run"), label="Phase-C motion run name"
    )
    observed = {
        "run_path": tables.run_path,
        "track_id": tables.candidate.track_id,
        "candidate_id": tables.candidate.candidate_id,
        "signal_id": tables.signal.signal_id,
        "signal_level": tables.signal.speed_level,
    }
    expected = {
        "run_path": source_binding.get("run_path"),
        "track_id": track_id,
        "candidate_id": source_binding.get("default_candidate_id"),
        "signal_id": source_binding.get("default_signal_id"),
        "signal_level": source_binding.get("default_signal_level"),
    }
    if observed != expected:
        _fail("Phase-C swim-bout tables differ from their exact bundle binding.")
    return build_bout_authority_identity(
        recording_id=recording_id,
        analysis_zarr=analysis_zarr,
        run_path=tables.run_path,
        run_schema_id=attrs.get("schema_id"),
        run_schema_version=attrs.get("schema_version"),
        run_layout=attrs.get("layout"),
        completion_status=attrs.get("palette_run_completion_status"),
        selector_eligible=attrs.get("stage_selector_eligible"),
        array_manifest_sha256=canonical_json_sha256(_plain(manifest)),
        motion_scope="provider",
        motion_run_path=f"analysis/track_kinematics_runs/provider/{run_name}",
        motion_manifest_sha256=source_binding["source_track_motion_manifest_sha256"],
        motion_verification_sha256=source_binding[
            "source_track_motion_verification_digest"
        ],
        track_id=track_id,
        track_row_start=source_binding["track_row_start"],
        track_row_stop=source_binding["track_row_stop"],
        source_sample_rate_hz=source_sample_rate_hz,
        frame_axis=frame_axis,
        candidate_id=tables.candidate.candidate_id,
        candidate_name=tables.candidate.candidate_name,
        signal_id=tables.signal.signal_id,
        signal_name=tables.signal.signal_name,
        signal_level=tables.signal.speed_level,
        signal_role=tables.signal.role,
        signal_source_level=tables.signal.source_level,
        selected_events=events,
        source_binding_sha256=binding_sha,
    )


def compare_bout_authority_identities(
    left: BoutAuthorityIdentity | Mapping[str, Any] | None,
    right: BoutAuthorityIdentity | Mapping[str, Any] | None,
) -> BoutAuthorityComparison:
    """Return equal, conflict, or not_proven without choosing a preferred source."""

    if left is None or right is None:
        missing = (
            "both"
            if left is None and right is None
            else "left" if left is None else "right"
        )
        status = "not_proven"
        reason = f"missing_{missing}_bout_authority"
        differing: tuple[str, ...] = ()
        content_equal: bool | None = None
        left_digest = None
        right_digest = None
    else:
        left_identity = validate_bout_authority_identity(left)
        right_identity = validate_bout_authority_identity(right)
        left_digest = left_identity.record_sha256
        right_digest = right_identity.record_sha256
        fields = tuple(
            name
            for name in sorted(_BOUT_IDENTITY_FIELDS - {"record_sha256"})
            if _plain(left_identity.record[name]) != _plain(right_identity.record[name])
        )
        content_equal = (
            left_identity.record["selected_events"]["content_sha256"]
            == right_identity.record["selected_events"]["content_sha256"]
        )
        if fields:
            status = "conflict"
            reason = "bout_authority_identity_conflict"
            differing = fields
        else:
            status = "equal"
            reason = "same_normalized_bout_authority"
            differing = ()
    body = {
        "schema_id": BOUT_AUTHORITY_COMPARISON_SCHEMA_ID,
        "schema_version": BOUT_AUTHORITY_COMPARISON_SCHEMA_VERSION,
        "status": status,
        "reason_code": reason,
        "left_record_sha256": left_digest,
        "right_record_sha256": right_digest,
        "differing_fields": list(differing),
        "selected_event_content_equal": content_equal,
    }
    record = {**body, "record_sha256": canonical_json_sha256(body)}
    return BoutAuthorityComparison(
        status=status,
        reason_code=reason,
        differing_fields=differing,
        selected_event_content_equal=content_equal,
        record=MappingProxyType(record),
    )


def core_roster_bout_identity(
    roster: Mapping[str, Any],
    *,
    bout_source: Any,
    track_binding: Mapping[str, Any],
) -> BoutAuthorityIdentity:
    """Build the selected bout identity only after validating the whole roster."""

    validated = validate_core_authority_roster(roster)
    bout_binding = validated["capability_bindings"][CANONICAL_SWIM_BOUTS_CAPABILITY]
    if _plain(bout_binding["source_binding"]) != _plain(bout_source.binding):
        _fail("Core roster and bound swim-bout source differ.")
    return bout_authority_identity_from_core_source(
        recording_id=validated["recording_id"],
        analysis_zarr=validated["analysis_zarr"],
        bout_source=bout_source,
        track_binding=track_binding,
    )


def bind_core_motion_and_bouts_from_roster(
    roster: Mapping[str, Any],
) -> BoundCoreMotionAndBouts:
    """Reopen only the core capabilities required by motion/bout consumers.

    This is the planner/executor boundary for paradigm computations.  The
    roster selects exact paths and profiles; the strict source binders then
    prove the current immutable publications still match.  No selector,
    fallback, or unrelated eye/tail source is opened.
    """

    validated = validate_core_authority_roster(roster)
    source = Path(validated["analysis_zarr"])
    recording_id = str(validated["recording_id"])
    capabilities = validated["capability_bindings"]
    motion_capability = _mapping(
        capabilities["kinematics_samples"], label="core motion capability"
    )
    if motion_capability.get("profile_id") != CORE_MOTION_SOURCE_SURFACE_PROFILE_ID:
        _fail("Paradigm motion consumers require the complete core-motion v2 profile.")
    expected_track = _mapping(
        motion_capability.get("source_binding"), label="core motion source binding"
    )
    run_path = _text(expected_track.get("run_path"), label="core motion run path")
    prefix = "analysis/track_kinematics_runs/"
    if not run_path.startswith(prefix):
        _fail("Core motion roster path is not a canonical track-kinematics run.")
    suffix = run_path[len(prefix) :]
    parts = suffix.split("/")
    if len(parts) != 2 or parts[0] not in {"online", "offline"}:
        _fail("Core motion roster path does not contain one exact scope and run.")
    scope, run_name = parts

    root = open_zarr_root(source, mode="r", use_consolidated=True)
    track = bind_kinematics_samples_source(
        root,
        zarr_path=source,
        expected_recording_id=recording_id,
        track_kinematics_run=run_name,
        track_scope=scope,
        source_surface_profile_id=CORE_MOTION_SOURCE_SURFACE_PROFILE_ID,
    )
    if _plain(track.binding) != _plain(expected_track):
        _fail("Live core motion source differs from the selected authority roster.")

    bout_capability = _mapping(
        capabilities[CANONICAL_SWIM_BOUTS_CAPABILITY],
        label="canonical swim-bout capability",
    )
    expected_bout = _mapping(
        bout_capability.get("source_binding"),
        label="canonical swim-bout source binding",
    )
    track_id = _int(expected_bout.get("track_id"), label="canonical bout track ID")
    bout_run_path = _text(
        expected_bout.get("run_path"), label="canonical swim-bout run path"
    )
    bout_prefix = "analysis/swim_bout_runs/"
    if (
        not bout_run_path.startswith(bout_prefix)
        or "/" in bout_run_path[len(bout_prefix) :]
    ):
        _fail("Canonical swim-bout roster path does not name one exact run.")
    bouts = bind_activity_spatial_sources(
        root,
        zarr_path=source,
        recording_id=recording_id,
        track_kinematics_run=run_name,
        track_scope=scope,
        swim_bout_runs_by_track={track_id: bout_run_path[len(bout_prefix) :]},
        prebound_track_source=track,
    )
    bound_bout = bouts.bout_sources[track_id]
    if _plain(bound_bout.binding) != _plain(expected_bout):
        _fail("Live canonical bout source differs from the selected authority roster.")
    identity = core_roster_bout_identity(
        validated,
        bout_source=bound_bout,
        track_binding=track.binding,
    )
    return BoundCoreMotionAndBouts(
        _verification_seal=_BOUND_CORE_MOTION_SEAL,
        roster=validated,
        root=root,
        track=track,
        bouts=bouts,
        bout_identities=MappingProxyType({track_id: identity}),
    )


__all__ = [
    "BOUT_AUTHORITY_COMPARISON_SCHEMA_ID",
    "BOUT_AUTHORITY_IDENTITY_SCHEMA_ID",
    "BoundCoreMotionAndBouts",
    "BoutAuthorityComparison",
    "BoutAuthorityIdentity",
    "CORE_AUTHORITY_ROSTER_SCHEMA_ID",
    "CORE_AUTHORITY_CONSUMPTION_SCHEMA_ID",
    "CoreAuthorityRosterError",
    "bout_authority_identity_from_core_source",
    "bout_authority_identity_from_phase_c_source",
    "bout_authority_identity_from_phase_c_tables",
    "bind_core_motion_and_bouts_from_roster",
    "bind_subject_body_frame_from_core_roster",
    "build_bout_authority_identity",
    "build_core_authority_consumption_receipt",
    "build_core_authority_roster",
    "build_subject_body_frame_source_binding",
    "compare_bout_authority_identities",
    "core_roster_bout_identity",
    "validate_bout_authority_identity",
    "validate_core_authority_consumption_receipt",
    "validate_core_authority_roster",
]
