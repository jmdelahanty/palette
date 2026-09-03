"""Bundle-backed provider-motion inputs for recording distributions."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from fisheye.group_statistics.recording_behavior_distributions import (
    RecordingDistributionMetricInput,
)
from fisheye.group_statistics.recording_distribution_scopes import (
    RecordingDistributionScope,
    ScopeMaskProjection,
    sample_scope_masks,
    transition_scope_masks,
    validate_scope_registry,
)
from fisheye.group_statistics.validated_behavior_distribution_specs import (
    DistributionMetricSpec,
)

from .validated_recording_behavior_source import (
    ProviderMotionTrackProjection,
    ValidatedRecordingBehaviorSource,
)
from .recording_distribution_timebase_adapter import (
    RecordingSessionTimebase,
    require_scope_timebase_binding,
)


PROVIDER_TIME_WEIGHT_POLICY_ID = "provider_motion_delta_seconds_positive_v1"

_VALUE_ARRAY_BY_COLUMN = MappingProxyType(
    {
        "speed_filtered_mm_s": "speed_filtered_mm",
        "speed_smoothed_mm_s": "speed_smoothed_mm",
        "frame_path_distance_smoothed_mm": "frame_path_distance_smoothed_mm",
        "delta_heading_smoothed_deg": "delta_heading_smoothed_degrees",
        "angular_velocity_smoothed_deg_s": "angular_velocity_smoothed_deg_s",
        "angular_speed_smoothed_deg_s": "angular_speed_smoothed_deg_s",
    }
)
_ANGULAR_VALIDITY_IDS = frozenset(
    {"angular_sample_valid_and_transition_valid_positive_time_v1"}
)
_LINEAR_VALIDITY_IDS = frozenset(
    {"linear_sample_valid_and_transition_valid_positive_time_v1"}
)
_BASE_ARRAYS = (
    "track_sample_key",
    "source_acquisition_frame_index",
    "linear_sample_valid",
    "angular_sample_valid",
    "smoothed_heading_degrees",
    "transition_valid",
    "delta_frames",
    "delta_seconds",
)


class RecordingDistributionMotionAdapterError(ValueError):
    """The validated provider-motion source cannot supply safe metric rows."""


def _fail(message: str) -> None:
    raise RecordingDistributionMotionAdapterError(message)


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{field} must be one object.")
    return value


def _fps_from_transitions(arrays: Mapping[str, np.ndarray]) -> float:
    delta_frames = np.asarray(arrays["delta_frames"], dtype=np.float64)
    delta_s = np.asarray(arrays["delta_seconds"], dtype=np.float64)
    transition_valid = np.asarray(arrays["transition_valid"], dtype=bool)
    valid = (
        transition_valid
        & np.isfinite(delta_s)
        & (delta_s > 0)
        & np.isfinite(delta_frames)
        & (delta_frames > 0)
    )
    if not np.any(valid):
        _fail("Provider motion has no valid transition for FPS verification.")
    rates = delta_frames[valid] / delta_s[valid]
    fps = float(np.median(rates))
    if (
        not math.isfinite(fps)
        or fps <= 0
        or not np.allclose(rates, fps, rtol=5e-5, atol=5e-5)
    ):
        _fail("Provider-motion transitions do not close one exact FPS.")
    return fps


def _constant(value: Any, count: int) -> np.ndarray:
    result = np.empty(count, dtype=object)
    result[:] = value
    return result


@dataclass(frozen=True, slots=True)
class ProviderMotionDistributionContext:
    projection: ProviderMotionTrackProjection
    arrays: Mapping[str, np.ndarray]
    fps: float
    sample_scopes: ScopeMaskProjection
    transition_scopes: ScopeMaskProjection
    valid_duration_s_by_scope: Mapping[str, float]
    provider_role: str
    provider_id: str
    provider_digest: str
    session_timebase: RecordingSessionTimebase | None = None
    time_weight_policy_id: str = PROVIDER_TIME_WEIGHT_POLICY_ID


def load_provider_motion_distribution_context(
    source: ValidatedRecordingBehaviorSource,
    scopes: Sequence[RecordingDistributionScope],
    *,
    value_columns: Sequence[str],
    session_timebase: RecordingSessionTimebase | None = None,
) -> ProviderMotionDistributionContext:
    """Load and validate the exact motion arrays needed by selected metrics."""

    if type(source) is not ValidatedRecordingBehaviorSource:
        _fail("Provider-motion adapter requires one validated recording source.")
    ordered = validate_scope_registry(scopes)
    columns = tuple(dict.fromkeys(str(value) for value in value_columns))
    unknown = sorted(set(columns) - set(_VALUE_ARRAY_BY_COLUMN))
    if unknown:
        _fail(f"Unsupported provider-motion value columns: {unknown!r}.")
    requested = tuple(
        dict.fromkeys((*_BASE_ARRAYS, *(_VALUE_ARRAY_BY_COLUMN[name] for name in columns)))
    )
    projection = source.provider_motion_track_projection(requested)
    arrays = projection.arrays
    keys = np.asarray(arrays["track_sample_key"])
    frames = np.asarray(arrays["source_acquisition_frame_index"], dtype=np.int64)
    if (
        keys.shape != (frames.size, 2)
        or np.any(keys[:, 0] != projection.track_id)
        or not np.array_equal(keys[:, 1], frames)
        or (frames.size and np.any(np.diff(frames) <= 0))
    ):
        _fail("Provider-motion track keys or acquisition frames are inconsistent.")
    requires_session_time = require_scope_timebase_binding(
        ordered, session_timebase
    )
    if session_timebase is not None and type(session_timebase) is not RecordingSessionTimebase:
        _fail("session_timebase must be one exact RecordingSessionTimebase.")
    if requires_session_time:
        assert session_timebase is not None
        timestamps_ns, timestamp_valid = session_timebase.map_frames(frames)
        previous_frames = frames - np.asarray(arrays["delta_frames"], dtype=np.int64)
        previous_ns, previous_valid = session_timebase.map_frames(previous_frames)
        delta_ns = timestamps_ns - previous_ns
        sample_scopes = sample_scope_masks(
            ordered,
            acquisition_frame_id=frames,
            timestamp_ns_session=timestamps_ns,
            timestamp_valid=timestamp_valid,
        )
        transition_scopes = transition_scope_masks(
            ordered,
            acquisition_frame_id=frames,
            acquisition_frame_delta=arrays["delta_frames"],
            timestamp_ns_session=timestamps_ns,
            timestamp_delta_ns=delta_ns,
            timestamp_valid=timestamp_valid & previous_valid,
        )
    else:
        sample_scopes = sample_scope_masks(
            ordered,
            acquisition_frame_id=frames,
        )
        transition_scopes = transition_scope_masks(
            ordered,
            acquisition_frame_id=frames,
            acquisition_frame_delta=arrays["delta_frames"],
        )
    fps = _fps_from_transitions(arrays)
    linear_valid = np.asarray(arrays["linear_sample_valid"], dtype=bool)
    durations = {
        scope.scope_id: float(
            np.count_nonzero(linear_valid & sample_scopes.masks[scope.scope_id])
        )
        / fps
        for scope in ordered
    }
    fish_binding = _mapping(
        source.bundle["source_bindings"].get("fish_position_keypoint"),
        field="fish_position_keypoint binding",
    )
    authority = _mapping(
        fish_binding.get("authority"), field="fish_position_keypoint authority"
    )
    provider_id = str(authority.get("provider_id") or "")
    provider_digest = str(authority.get("provider_digest") or "")
    if not provider_id or len(provider_digest) != 64:
        _fail("Keypoint provider identity is absent from the validated bundle.")
    return ProviderMotionDistributionContext(
        projection=projection,
        arrays=arrays,
        fps=fps,
        sample_scopes=sample_scopes,
        transition_scopes=transition_scopes,
        valid_duration_s_by_scope=MappingProxyType(durations),
        provider_role="keypoint",
        provider_id=provider_id,
        provider_digest=provider_digest,
        session_timebase=session_timebase,
    )


def provider_motion_distribution_inputs(
    source: ValidatedRecordingBehaviorSource,
    scopes: Sequence[RecordingDistributionScope],
    metric_specs: Sequence[DistributionMetricSpec],
    *,
    session_timebase: RecordingSessionTimebase | None = None,
) -> tuple[ProviderMotionDistributionContext, tuple[RecordingDistributionMetricInput, ...]]:
    """Project selected motion metrics into the generic recording reducer."""

    selected = tuple(
        spec for spec in metric_specs if spec.source_surface == "provider_motion_samples"
    )
    if not selected:
        _fail("No provider-motion metric specifications were selected.")
    context = load_provider_motion_distribution_context(
        source,
        scopes,
        value_columns=tuple(spec.value_column for spec in selected),
        session_timebase=session_timebase,
    )
    arrays = context.arrays
    row_count = int(np.asarray(arrays["source_acquisition_frame_index"]).size)
    group = {"provider_role": _constant(context.provider_role, row_count)}
    identity_values = {
        "position_provider_id": context.provider_id,
        "position_provider_digest": context.provider_digest,
        "source_run_path": context.projection.run_path,
        "source_manifest_sha256": context.projection.manifest_sha256,
        "source_verification_digest": context.projection.verification_digest,
        "track_id": context.projection.track_id,
        "time_weight_policy_id": context.time_weight_policy_id,
        "scope_timebase_sha256": (
            context.session_timebase.binding["timebase_sha256"]
            if context.session_timebase is not None
            else "not_required_for_frame_scopes"
        ),
    }
    identity_arrays = {
        name: _constant(value, row_count) for name, value in identity_values.items()
    }
    transition_valid = np.asarray(arrays["transition_valid"], dtype=bool)
    linear_valid = np.asarray(arrays["linear_sample_valid"], dtype=bool)
    angular_valid = np.asarray(arrays["angular_sample_valid"], dtype=bool)
    result: list[RecordingDistributionMetricInput] = []
    for spec in selected:
        if spec.validity_policy_id in _ANGULAR_VALIDITY_IDS:
            valid = angular_valid & transition_valid
        elif spec.validity_policy_id in _LINEAR_VALIDITY_IDS:
            valid = linear_valid & transition_valid
        else:
            _fail(
                f"{spec.metric_id}: unsupported provider-motion validity policy."
            )
        result.append(
            RecordingDistributionMetricInput(
                spec=spec,
                values=np.asarray(arrays[_VALUE_ARRAY_BY_COLUMN[spec.value_column]]),
                valid=valid,
                scope_projection=context.sample_scopes,
                group_arrays=group,
                source_identity_arrays=identity_arrays,
                source_identity_fallback=identity_values,
                time_weights_s=np.asarray(arrays["delta_seconds"]),
                time_scope_projection=context.transition_scopes,
                valid_duration_s_by_scope=context.valid_duration_s_by_scope,
            )
        )
    return context, tuple(result)


__all__ = [
    "PROVIDER_TIME_WEIGHT_POLICY_ID",
    "ProviderMotionDistributionContext",
    "RecordingDistributionMotionAdapterError",
    "load_provider_motion_distribution_context",
    "provider_motion_distribution_inputs",
]
