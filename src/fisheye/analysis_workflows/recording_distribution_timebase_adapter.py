"""Exact session-time projection for recording-distribution scope adapters."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from fisheye.analysis_workflows.chaser_relative_frame_validation_receipt import (
    load_chaser_relative_frame_targeted_source_handle,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.group_statistics.recording_distribution_scopes import (
    RecordingDistributionScope,
)

from .validated_recording_behavior_source import ValidatedRecordingBehaviorSource


SESSION_TIMEBASE_SCHEMA_ID = "palette.analysis.recording_session_timebase"
SESSION_TIMEBASE_SCHEMA_VERSION = 1
SESSION_TIMEBASE_MAPPING_POLICY_ID = "exact_acquisition_frame_lookup_no_interpolation_v1"


class RecordingDistributionTimebaseError(ValueError):
    """The recording bundle cannot supply one exact session-time projection."""


def _fail(message: str) -> None:
    raise RecordingDistributionTimebaseError(message)


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{field} must be one object.")
    return value


def _readonly(values: Any, *, dtype: Any, field: str) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 1 or result.dtype != np.dtype(dtype):
        _fail(f"{field} must be one exact {np.dtype(dtype)} vector.")
    result = np.array(result, copy=True, order="C")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class RecordingSessionTimebase:
    """One bundle-bound frame-to-session-time lookup with no interpolation."""

    acquisition_frame_id: np.ndarray
    timestamp_ns_session: np.ndarray
    timestamp_valid: np.ndarray
    source_binding: Mapping[str, Any]

    def __post_init__(self) -> None:
        frames = _readonly(
            self.acquisition_frame_id,
            dtype=np.int64,
            field="acquisition_frame_id",
        )
        timestamps = _readonly(
            self.timestamp_ns_session,
            dtype=np.int64,
            field="timestamp_ns_session",
        )
        valid = _readonly(self.timestamp_valid, dtype=bool, field="timestamp_valid")
        if frames.shape != timestamps.shape or frames.shape != valid.shape:
            _fail("Session-time axes do not share one row count.")
        if frames.size and np.any(np.diff(frames) <= 0):
            _fail("Session-time acquisition frames must be strictly increasing.")
        valid_timestamps = timestamps[valid]
        if valid_timestamps.size > 1 and np.any(np.diff(valid_timestamps) <= 0):
            _fail("Valid session timestamps must be strictly increasing.")
        binding = dict(self.source_binding)
        if not binding:
            _fail("Session-time source binding must not be empty.")
        object.__setattr__(self, "acquisition_frame_id", frames)
        object.__setattr__(self, "timestamp_ns_session", timestamps)
        object.__setattr__(self, "timestamp_valid", valid)
        object.__setattr__(self, "source_binding", MappingProxyType(binding))

    @property
    def binding(self) -> Mapping[str, Any]:
        body = {
            "schema_id": SESSION_TIMEBASE_SCHEMA_ID,
            "schema_version": SESSION_TIMEBASE_SCHEMA_VERSION,
            "mapping_policy_id": SESSION_TIMEBASE_MAPPING_POLICY_ID,
            **dict(self.source_binding),
        }
        return MappingProxyType(
            {**body, "timebase_sha256": canonical_json_sha256(body)}
        )

    def map_frames(self, frames: Any) -> tuple[np.ndarray, np.ndarray]:
        """Map exact acquisition frames; missing or invalid rows stay uncovered."""

        requested = np.asarray(frames)
        if requested.ndim != 1 or requested.dtype.kind not in "iu":
            _fail("Requested frame axis must be one integer vector.")
        requested = requested.astype(np.int64, copy=False)
        indexes = np.searchsorted(self.acquisition_frame_id, requested, side="left")
        inside = indexes < self.acquisition_frame_id.size
        exact = np.zeros(requested.shape, dtype=bool)
        exact[inside] = self.acquisition_frame_id[indexes[inside]] == requested[inside]
        timestamps = np.zeros(requested.shape, dtype=np.int64)
        valid = np.zeros(requested.shape, dtype=bool)
        timestamps[exact] = self.timestamp_ns_session[indexes[exact]]
        valid[exact] = self.timestamp_valid[indexes[exact]]
        return timestamps, valid


def load_bundle_recording_session_timebase(
    source: ValidatedRecordingBehaviorSource,
) -> RecordingSessionTimebase:
    """Load the paired-relative consensus clock sealed by a validated bundle."""

    if type(source) is not ValidatedRecordingBehaviorSource:
        _fail("Session-time adapter requires one validated recording source.")
    axis_binding = _mapping(
        source.bundle["source_bindings"].get("row_axis_timing_and_scale"),
        field="row_axis_timing_and_scale binding",
    )
    if axis_binding.get("binding_type") != "paired_relative_frame_consensus_v1":
        _fail("Bundle has an unsupported row-axis/timing binding.")
    authority = _mapping(
        axis_binding.get("authority"), field="row-axis/timing authority"
    )
    timing = _mapping(
        authority.get("shared_timing_semantics"),
        field="shared timing semantics",
    )
    if timing.get("timestamp_field") != "timestamp_ns_session":
        _fail("Bundle timing authority is not timestamp_ns_session.")
    sealed = _mapping(axis_binding.get("sealed_by"), field="timing sealed_by")
    keypoint_seal = _mapping(sealed.get("keypoint"), field="timing keypoint seal")
    child = source.scientific_child("chaser_relative_keypoint")
    if dict(child.binding) != dict(keypoint_seal):
        _fail("Session-time seal differs from the keypoint relative child.")
    handle = load_chaser_relative_frame_targeted_source_handle(
        keypoint_seal["receipt_path"],
        required_base_arrays=(
            "acquisition_frame_id",
            "timestamp_ns",
            "timestamp_valid",
        ),
        required_body_arrays=(),
        collapsed_frame_arrays=(
            "acquisition_frame_id",
            "timestamp_ns",
            "timestamp_valid",
        ),
        expected_analysis_zarr=source.analysis_zarr,
        expected_recording_id=source.recording_id,
        expected_run_name=str(keypoint_seal["run_path"]).rsplit("/", 1)[-1],
    )
    observed = {
        "run_path": handle.run_path,
        "manifest_sha256": handle.manifest_sha256,
        "payload_digest": handle.payload_digest,
        "receipt_sha256": handle.receipt_digest,
    }
    expected = {
        "run_path": keypoint_seal["run_path"],
        "manifest_sha256": keypoint_seal["manifest_sha256"],
        "payload_digest": keypoint_seal["payload_digest"],
        "receipt_sha256": keypoint_seal["receipt_sha256"],
    }
    if observed != expected:
        _fail("Targeted session-time evidence differs from the validated bundle.")
    return RecordingSessionTimebase(
        acquisition_frame_id=handle.frame_array("acquisition_frame_id"),
        timestamp_ns_session=handle.frame_array("timestamp_ns"),
        timestamp_valid=handle.frame_array("timestamp_valid"),
        source_binding={
            "recording_id": source.recording_id,
            "bundle_sha256": source.bundle_sha256,
            "row_axis_timing_authority_sha256": canonical_json_sha256(authority),
            "relative_frame_run_path": handle.run_path,
            "relative_frame_manifest_sha256": handle.manifest_sha256,
            "relative_frame_payload_digest": handle.payload_digest,
            "relative_frame_receipt_sha256": handle.receipt_digest,
            "timing_policy": dict(timing),
        },
    )


def require_scope_timebase_binding(
    scopes: Sequence[RecordingDistributionScope],
    timebase: RecordingSessionTimebase | None,
) -> bool:
    """Prove every requested time scope names this exact clock authority."""

    time_scopes = tuple(scope for scope in scopes if scope.axis_kind == "session_time_ns")
    if not time_scopes:
        return False
    if type(timebase) is not RecordingSessionTimebase:
        _fail("Session-time scopes require one exact recording timebase.")
    expected = dict(timebase.binding)
    for scope in time_scopes:
        if scope.source_binding.get("timebase") != expected:
            _fail(f"Scope {scope.scope_id!r} binds another session-time authority.")
    return True


__all__ = [
    "RecordingDistributionTimebaseError",
    "RecordingSessionTimebase",
    "SESSION_TIMEBASE_MAPPING_POLICY_ID",
    "SESSION_TIMEBASE_SCHEMA_ID",
    "SESSION_TIMEBASE_SCHEMA_VERSION",
    "load_bundle_recording_session_timebase",
    "require_scope_timebase_binding",
]
