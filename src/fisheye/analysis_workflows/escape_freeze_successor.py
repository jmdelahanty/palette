"""Exact-trial escape/freeze successor.

Speed-defined escape events and body-frame high-turn evidence are intentionally
separate.  The speed tier can be computed from exact trials, motion, distance,
and bouts alone.  The optional high-turn tier annotates those same events; it
never changes the speed threshold or silently supplies missing orientation.

Every event attaches to one controller-trial row.  Counts remain present when
an event-aligned recapture trace is unusable, with a separate trace exclusion
reason as required by the composable chaser contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from fisheye.analysis_workflows.controller_trial_successor import (
    PreparedControllerTrials,
)
from fisheye.analysis_workflows.core_motion_source_handle import (
    validate_core_motion_dependency_record,
)
from fisheye.analysis_workflows.generalized_bout_response_successor import (
    PreparedGeneralizedBoutResponse,
    exact_provider_frame_projection,
)
from fisheye.shared.coordinate_frame_record import array_values_sha256
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256

SCHEMA_ID = "palette.analysis.chaser_escape_freeze"
SCHEMA_VERSION = 2
PREPARED_SCHEMA_ID = "palette.analysis.chaser_escape_freeze.prepared_successor"
PREPARED_SCHEMA_VERSION = 1
METHOD_ID = "exact_trial_speed_escape_optional_high_turn_freeze_v1"

RESPONSE_CLASS_INSUFFICIENT = 0
RESPONSE_CLASS_ESCAPE = 1
RESPONSE_CLASS_FREEZE = 2
RESPONSE_CLASS_OTHER = 3

TRACE_REASON_VALID = 0
TRACE_REASON_NO_POST_EVENT_DISTANCE = 1
TRACE_REASON_EVENT_FRAME_UNAVAILABLE = 2


class EscapeFreezeSuccessorError(ValueError):
    """Raised when exact-trial escape/freeze facts cannot be prepared."""


def _fail(message: str) -> None:
    raise EscapeFreezeSuccessorError(message)


def _readonly(value: Any) -> np.ndarray:
    result = np.array(value, copy=True, order="C")
    result.setflags(write=False)
    return result


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _text(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        _fail(f"{name} must be one non-empty exact string.")
    return value


def _digest(value: object, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{name} must be one lowercase SHA-256 digest.")
    return value


def _vector(value: Any, *, name: str, dtype: Any, size: int) -> np.ndarray:
    result = np.asarray(value)
    if result.dtype != np.dtype(dtype) or result.shape != (size,):
        _fail(
            f"{name} must have exact dtype {np.dtype(dtype).str!r} and "
            f"shape {(size,)!r}."
        )
    return result


def _float_vector(value: Any, *, name: str, size: int) -> np.ndarray:
    result = np.asarray(value)
    if result.dtype.kind != "f" or result.shape != (size,):
        _fail(f"{name} must be one floating array with shape {(size,)!r}.")
    return np.asarray(result, dtype=np.float64)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class EscapeFreezeInput:
    recording_id: str
    source_motion_run_path: str
    source_motion_manifest_sha256: str
    source_speed_level: str
    source_motion_frame_projection: Mapping[str, Any]
    controller_trials: PreparedControllerTrials
    bout_response: PreparedGeneralizedBoutResponse
    n_frames: int
    n_chasers: int
    acquisition_frame_id_by_frame: np.ndarray
    timestamp_ns_by_frame: np.ndarray
    timestamp_valid_by_frame: np.ndarray
    speed_mm_s_by_frame: np.ndarray
    speed_valid_by_frame: np.ndarray
    chaser_identity_code: np.ndarray
    distance_mm: np.ndarray
    distance_valid: np.ndarray
    source_core_authority: Mapping[str, Any] | None = None
    escape_speed_threshold_mm_s: float = 20.0
    high_turn_threshold_deg: float = 45.0
    freeze_speed_threshold_mm_s: float = 2.0
    freeze_window_s: float = 1.0
    freeze_fraction_threshold: float = 0.8
    minimum_freeze_valid_fraction: float = 0.5
    threshold_sweep_mm_s: Sequence[float] = (10.0, 15.0, 20.0, 25.0, 30.0)


@dataclass(frozen=True, slots=True)
class PreparedEscapeFreeze:
    recording_id: str
    n_trials: int
    n_events: int
    arrays: Mapping[str, np.ndarray]
    manifest: Mapping[str, Any]

    def array(self, name: str) -> np.ndarray:
        try:
            return self.arrays[name]
        except KeyError as exc:
            raise KeyError(f"Unknown escape/freeze array {name!r}.") from exc

    @property
    def payload_digest(self) -> str:
        return str(self.manifest["payload_digest"])


def _declarations(arrays: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    return [
        {
            "path": name,
            "dtype": np.asarray(values).dtype.str,
            "shape": list(np.asarray(values).shape),
            "content_sha256": array_values_sha256(np.asarray(values)),
        }
        for name, values in sorted(arrays.items())
    ]


def _positive_finite(value: object, *, name: str, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{name} must be one finite number.")
    result = float(value)
    if not math.isfinite(result) or (result < 0 if allow_zero else result <= 0):
        _fail(
            f"{name} must be {'non-negative' if allow_zero else 'positive'} and finite."
        )
    return result


def prepare_escape_freeze_successor(source: EscapeFreezeInput) -> PreparedEscapeFreeze:
    """Prepare per-event, per-trial, sweep, and recording-level facts."""

    if type(source) is not EscapeFreezeInput:
        raise TypeError("source must be one EscapeFreezeInput.")
    recording_id = _text(source.recording_id, name="recording_id")
    _text(source.source_motion_run_path, name="source_motion_run_path")
    _digest(source.source_motion_manifest_sha256, name="source_motion_manifest_sha256")
    _text(source.source_speed_level, name="source_speed_level")
    if type(source.controller_trials) is not PreparedControllerTrials:
        raise TypeError("controller_trials must be one prepared exact successor.")
    if type(source.bout_response) is not PreparedGeneralizedBoutResponse:
        raise TypeError("bout_response must be one prepared generalized successor.")
    if (
        source.controller_trials.recording_id != recording_id
        or source.bout_response.recording_id != recording_id
    ):
        _fail("Escape/freeze sources belong to different recordings.")
    if (
        source.controller_trials.payload_digest
        != source.bout_response.manifest["sources"]["controller_trial_payload_sha256"]
    ):
        _fail("Bout-response controller-trial binding is stale.")
    if type(source.n_frames) is not int or source.n_frames < 0:
        _fail("n_frames must be one non-negative exact integer.")
    if type(source.n_chasers) is not int or source.n_chasers <= 0:
        _fail("n_chasers must be one positive exact integer.")
    projection = source.source_motion_frame_projection
    projection_schema = (
        projection.get("schema_id") if isinstance(projection, Mapping) else None
    )
    if (
        projection_schema
        not in {
            "palette.provider_motion.relative_frame_projection",
            "palette.core_motion.relative_frame_projection",
        }
        or projection.get("schema_version") != 1
        or projection.get("join_policy")
        != "left_join_missing_provider_rows_invalid_no_interpolation"
        or projection.get("relative_frame_count") != source.n_frames
        or projection.get("fallback") != "prohibited"
    ):
        _fail("source_motion_frame_projection is absent, malformed, or permissive.")
    core_authority: dict[str, Any] | None = None
    if source.source_core_authority is not None:
        try:
            core_authority = dict(
                validate_core_motion_dependency_record(source.source_core_authority)
            )
        except (TypeError, ValueError) as exc:
            _fail(f"Core motion dependency record is invalid: {exc}")
        if (
            projection_schema != "palette.core_motion.relative_frame_projection"
            or core_authority.get("recording_id") != recording_id
            or core_authority.get("motion_run_path") != source.source_motion_run_path
            or core_authority.get("motion_manifest_sha256")
            != source.source_motion_manifest_sha256
            or core_authority.get("core_authority_roster_sha256")
            != projection.get("core_authority_roster_sha256")
        ):
            _fail("Core motion dependency record is absent, stale, or inconsistent.")
        response_core = source.bout_response.manifest["sources"].get("core_authority")
        if not isinstance(response_core, Mapping) or _plain(response_core) != _plain(
            core_authority
        ):
            _fail("Escape/freeze and bout-response core authorities differ.")
    elif projection_schema == "palette.core_motion.relative_frame_projection":
        _fail("Core motion projection lacks its sealed authority dependency.")
    if (
        source.controller_trials.n_frames != source.n_frames
        or source.controller_trials.n_chasers != source.n_chasers
        or source.bout_response.n_chasers != source.n_chasers
    ):
        _fail("Escape/freeze dimensions differ from prepared dependencies.")
    n_frames, n_chasers = source.n_frames, source.n_chasers
    n_rows = n_frames * n_chasers
    frame_ids = _vector(
        source.acquisition_frame_id_by_frame,
        name="acquisition_frame_id_by_frame",
        dtype=np.int64,
        size=n_frames,
    )
    if np.unique(frame_ids).size != n_frames:
        _fail("Acquisition frame identities are duplicated.")
    frame_lookup = {int(value): row for row, value in enumerate(frame_ids.tolist())}
    timestamp = _vector(
        source.timestamp_ns_by_frame,
        name="timestamp_ns_by_frame",
        dtype=np.int64,
        size=n_frames,
    )
    timestamp_valid = _vector(
        source.timestamp_valid_by_frame,
        name="timestamp_valid_by_frame",
        dtype=bool,
        size=n_frames,
    )
    speed = _float_vector(
        source.speed_mm_s_by_frame,
        name="speed_mm_s_by_frame",
        size=n_frames,
    )
    speed_valid = _vector(
        source.speed_valid_by_frame,
        name="speed_valid_by_frame",
        dtype=bool,
        size=n_frames,
    )
    codes = _vector(
        source.chaser_identity_code,
        name="chaser_identity_code",
        dtype=np.uint16,
        size=n_rows,
    ).reshape(n_frames, n_chasers)
    if n_frames and np.any(codes != codes[:1, :]):
        _fail("Chaser identity changed along the fixed chaser axis.")
    chaser_codes = (
        codes[0] if n_frames else np.arange(1, n_chasers + 1, dtype=np.uint16)
    )
    distance = _float_vector(
        source.distance_mm,
        name="distance_mm",
        size=n_rows,
    ).reshape(n_frames, n_chasers)
    distance_valid = _vector(
        source.distance_valid,
        name="distance_valid",
        dtype=bool,
        size=n_rows,
    ).reshape(n_frames, n_chasers)

    escape_threshold = _positive_finite(
        source.escape_speed_threshold_mm_s,
        name="escape_speed_threshold_mm_s",
    )
    high_turn_threshold = _positive_finite(
        source.high_turn_threshold_deg,
        name="high_turn_threshold_deg",
    )
    freeze_speed = _positive_finite(
        source.freeze_speed_threshold_mm_s,
        name="freeze_speed_threshold_mm_s",
        allow_zero=True,
    )
    freeze_window = _positive_finite(source.freeze_window_s, name="freeze_window_s")
    for name in ("freeze_fraction_threshold", "minimum_freeze_valid_fraction"):
        value = _positive_finite(getattr(source, name), name=name, allow_zero=True)
        if value > 1:
            _fail(f"{name} must be in [0, 1].")
    sweep = np.asarray(tuple(source.threshold_sweep_mm_s), dtype=np.float64)
    if (
        sweep.ndim != 1
        or sweep.size == 0
        or np.any(~np.isfinite(sweep))
        or np.any(sweep <= 0)
        or np.any(np.diff(sweep) <= 0)
    ):
        _fail("threshold_sweep_mm_s must be positive, finite, and strictly increasing.")

    trials = source.controller_trials.arrays
    n_trials = source.controller_trials.n_trials
    trial_chaser = np.asarray(trials["chaser_identity_code"], dtype=np.uint16)
    trial_start = np.asarray(trials["start_source_frame_row"], dtype=np.int64)
    trial_end = np.asarray(trials["end_source_frame_row_exclusive"], dtype=np.int64)
    trial_trigger_id = np.asarray(
        trials["trigger_acquisition_frame_id"], dtype=np.int64
    )
    dense_trial = np.asarray(
        trials["trial_row_id_by_source_row"], dtype=np.int64
    ).reshape(n_frames, n_chasers)

    bout = source.bout_response.arrays
    bout_trial = np.asarray(bout["controller_trial_row_id"], dtype=np.int64)
    bout_base_valid = np.asarray(bout["base_valid"], dtype=bool)
    bout_peak = np.asarray(bout["bout_peak_speed_mm_s"], dtype=np.float64)
    event_mask = (
        bout_base_valid
        & (bout_trial >= 0)
        & (bout_trial < n_trials)
        & np.isfinite(bout_peak)
        & (bout_peak >= escape_threshold)
    )
    event_source_rows = np.flatnonzero(event_mask)
    n_events = int(event_source_rows.size)
    event_trial = bout_trial[event_source_rows]
    if np.any(np.bincount(event_trial, minlength=n_trials) < 0):  # pragma: no cover
        _fail("Invalid event/trial count.")
    event_arrays: dict[str, np.ndarray] = {
        "event_row_id": np.arange(n_events, dtype=np.int64),
        "event_source_bout_chaser_row_id": np.asarray(
            bout["bout_chaser_row_id"], dtype=np.int64
        )[event_source_rows],
        "event_bout_row_id": np.asarray(bout["bout_row_id"], dtype=np.int64)[
            event_source_rows
        ],
        "event_bout_id": np.asarray(bout["bout_id"], dtype=np.int64)[event_source_rows],
        "event_controller_trial_row_id": event_trial.astype(np.int64),
        "event_chaser_identity_code": np.asarray(
            bout["chaser_identity_code"], dtype=np.uint16
        )[event_source_rows],
        "event_onset_acquisition_frame_id": np.asarray(
            bout["start_acquisition_frame_id"], dtype=np.int64
        )[event_source_rows],
        "event_peak_speed_mm_s": bout_peak[event_source_rows].astype(np.float32),
        "event_distance_at_onset_mm": np.asarray(
            bout["distance_at_onset_mm"], dtype=np.float32
        )[event_source_rows],
        "event_separation_gain_mm": np.asarray(
            bout["delta_distance_mm"], dtype=np.float32
        )[event_source_rows],
        "event_directed_valid": np.asarray(bout["directed_valid"], dtype=bool)[
            event_source_rows
        ],
        "event_turn_deg": np.asarray(bout["turn_deg"], dtype=np.float32)[
            event_source_rows
        ],
        "event_high_turn": np.zeros(n_events, dtype=bool),
        "event_latency_from_trigger_s": np.full(n_events, np.nan, dtype=np.float64),
        "event_trigger_distance_mm": np.full(n_events, np.nan, dtype=np.float32),
        "event_recaptured": np.zeros(n_events, dtype=bool),
        "event_recapture_latency_s": np.full(n_events, np.nan, dtype=np.float64),
        "event_trace_valid": np.zeros(n_events, dtype=bool),
        "event_trace_exclusion_reason_code": np.full(
            n_events, TRACE_REASON_EVENT_FRAME_UNAVAILABLE, dtype=np.uint8
        ),
    }
    event_arrays["event_high_turn"] = (
        event_arrays["event_directed_valid"]
        & np.isfinite(event_arrays["event_turn_deg"])
        & (np.abs(event_arrays["event_turn_deg"]) >= high_turn_threshold)
    )

    trial_code_to_axis = {int(code): axis for axis, code in enumerate(chaser_codes)}
    for event in range(n_events):
        trial_row = int(event_trial[event])
        code = int(event_arrays["event_chaser_identity_code"][event])
        if int(trial_chaser[trial_row]) != code or code not in trial_code_to_axis:
            _fail("An escape event attached across chaser identities.")
        c = trial_code_to_axis[code]
        trigger_frame = frame_lookup.get(int(trial_trigger_id[trial_row]), -1)
        event_frame = frame_lookup.get(
            int(event_arrays["event_onset_acquisition_frame_id"][event]), -1
        )
        if trigger_frame >= 0:
            if distance_valid[trigger_frame, c]:
                event_arrays["event_trigger_distance_mm"][event] = distance[
                    trigger_frame, c
                ]
            if (
                event_frame >= 0
                and timestamp_valid[trigger_frame]
                and timestamp_valid[event_frame]
            ):
                event_arrays["event_latency_from_trigger_s"][event] = (
                    timestamp[event_frame] - timestamp[trigger_frame]
                ) / 1e9
        if event_frame < 0:
            continue
        post_frames = np.arange(
            event_frame + 1,
            int(trial_end[trial_row]),
            dtype=np.int64,
        )
        if post_frames.size:
            post_frames = post_frames[dense_trial[post_frames, c] == trial_row]
        usable = (
            post_frames[distance_valid[post_frames, c]]
            if post_frames.size
            else np.zeros(0, dtype=np.int64)
        )
        if usable.size == 0:
            event_arrays["event_trace_exclusion_reason_code"][
                event
            ] = TRACE_REASON_NO_POST_EVENT_DISTANCE
            continue
        event_arrays["event_trace_valid"][event] = True
        event_arrays["event_trace_exclusion_reason_code"][event] = TRACE_REASON_VALID
        onset_distance = float(event_arrays["event_distance_at_onset_mm"][event])
        recapture = usable[distance[usable, c] <= onset_distance]
        if recapture.size:
            event_arrays["event_recaptured"][event] = True
            first = int(recapture[0])
            if timestamp_valid[event_frame] and timestamp_valid[first]:
                event_arrays["event_recapture_latency_s"][event] = (
                    timestamp[first] - timestamp[event_frame]
                ) / 1e9

    trial_arrays: dict[str, np.ndarray] = {
        "trial_row_id": np.arange(n_trials, dtype=np.int64),
        "trial_chaser_identity_code": trial_chaser.copy(),
        "trial_logged_id": np.asarray(trials["logged_trial_id"], dtype=np.int64).copy(),
        "trial_ordinal": np.asarray(trials["trial_ordinal"], dtype=np.int32).copy(),
        "trial_envelope_frame_count": np.asarray(
            trials["envelope_frame_count"], dtype=np.int64
        ).copy(),
        "trial_gap_frame_count": np.asarray(
            trials["gap_frame_count"], dtype=np.int64
        ).copy(),
        "trial_gap_fraction": np.asarray(
            trials["gap_fraction"], dtype=np.float64
        ).copy(),
        "trial_logged_active_id_unavailable_count": np.zeros(n_trials, dtype=np.int64),
        "trial_trigger_acquisition_frame_id": trial_trigger_id.copy(),
        "trial_trigger_distance_mm": np.full(n_trials, np.nan, dtype=np.float32),
        "trial_valid_time_s": np.zeros(n_trials, dtype=np.float64),
        "trial_bout_count": np.zeros(n_trials, dtype=np.int64),
        "trial_escape_event_count": np.zeros(n_trials, dtype=np.int64),
        "trial_high_turn_escape_count": np.zeros(n_trials, dtype=np.int64),
        "trial_escape_event_rate_per_min": np.full(n_trials, np.nan, dtype=np.float64),
        "trial_first_escape_latency_s": np.full(n_trials, np.nan, dtype=np.float64),
        "trial_mean_separation_gain_mm": np.full(n_trials, np.nan, dtype=np.float64),
        "trial_recapture_fraction": np.full(n_trials, np.nan, dtype=np.float64),
        "trial_freeze_valid_fraction": np.full(n_trials, np.nan, dtype=np.float64),
        "trial_freeze_low_speed_fraction": np.full(n_trials, np.nan, dtype=np.float64),
        "trial_escape_speed_class": np.zeros(n_trials, dtype=bool),
        "trial_freeze_candidate": np.zeros(n_trials, dtype=bool),
        "trial_response_class_code": np.full(
            n_trials, RESPONSE_CLASS_INSUFFICIENT, dtype=np.uint8
        ),
    }
    bout_all_trial = np.asarray(bout["controller_trial_row_id"], dtype=np.int64)
    envelope_trial = np.asarray(
        trials["trial_envelope_row_id_by_source_row"], dtype=np.int64
    ).reshape(n_frames, n_chasers)
    unresolved_active_id = np.asarray(
        trials["logged_active_trial_id_unavailable"], dtype=bool
    ).reshape(n_frames, n_chasers)
    for trial_row in range(n_trials):
        code = int(trial_chaser[trial_row])
        if code not in trial_code_to_axis:
            _fail("Controller trial names an unknown chaser identity.")
        c = trial_code_to_axis[code]
        start, end = int(trial_start[trial_row]), int(trial_end[trial_row])
        trigger = frame_lookup.get(int(trial_trigger_id[trial_row]), -1)
        if trigger >= 0 and distance_valid[trigger, c]:
            trial_arrays["trial_trigger_distance_mm"][trial_row] = distance[trigger, c]
        frames = np.arange(start, end, dtype=np.int64)
        envelope_frames = frames[envelope_trial[frames, c] == trial_row]
        trial_arrays["trial_logged_active_id_unavailable_count"][trial_row] = int(
            np.count_nonzero(unresolved_active_id[envelope_frames, c])
        )
        member_frames = frames[dense_trial[frames, c] == trial_row]
        if member_frames.size > 1:
            prev, current = member_frames[:-1], member_frames[1:]
            adjacent = current == prev + 1
            valid_dt = (
                adjacent
                & timestamp_valid[prev]
                & timestamp_valid[current]
                & (timestamp[current] > timestamp[prev])
            )
            trial_arrays["trial_valid_time_s"][trial_row] = float(
                np.sum((timestamp[current[valid_dt]] - timestamp[prev[valid_dt]]) / 1e9)
            )
        trial_bout_mask = bout_base_valid & (bout_all_trial == trial_row)
        trial_arrays["trial_bout_count"][trial_row] = int(
            np.count_nonzero(trial_bout_mask)
        )
        event_rows = np.flatnonzero(event_trial == trial_row)
        count = int(event_rows.size)
        trial_arrays["trial_escape_event_count"][trial_row] = count
        trial_arrays["trial_high_turn_escape_count"][trial_row] = int(
            np.count_nonzero(event_arrays["event_high_turn"][event_rows])
        )
        seconds = float(trial_arrays["trial_valid_time_s"][trial_row])
        if seconds > 0:
            trial_arrays["trial_escape_event_rate_per_min"][trial_row] = count / (
                seconds / 60.0
            )
        if count:
            latency = event_arrays["event_latency_from_trigger_s"][event_rows]
            finite = latency[np.isfinite(latency)]
            if finite.size:
                trial_arrays["trial_first_escape_latency_s"][trial_row] = float(
                    np.min(finite)
                )
            gain = event_arrays["event_separation_gain_mm"][event_rows]
            finite_gain = gain[np.isfinite(gain)]
            if finite_gain.size:
                trial_arrays["trial_mean_separation_gain_mm"][trial_row] = float(
                    np.mean(finite_gain)
                )
            trace = event_arrays["event_trace_valid"][event_rows]
            if np.any(trace):
                trial_arrays["trial_recapture_fraction"][trial_row] = float(
                    np.mean(event_arrays["event_recaptured"][event_rows][trace])
                )

        if trigger >= 0:
            window_end_ns = timestamp[trigger] + int(round(freeze_window * 1e9))
            window = frames[(frames >= trigger) & (timestamp[frames] <= window_end_ns)]
            members = window[dense_trial[window, c] == trial_row]
            valid_speed = members[speed_valid[members] & np.isfinite(speed[members])]
            if window.size:
                trial_arrays["trial_freeze_valid_fraction"][trial_row] = (
                    valid_speed.size / window.size
                )
            if valid_speed.size:
                trial_arrays["trial_freeze_low_speed_fraction"][trial_row] = float(
                    np.mean(speed[valid_speed] <= freeze_speed)
                )
        coverage = float(trial_arrays["trial_freeze_valid_fraction"][trial_row])
        low = float(trial_arrays["trial_freeze_low_speed_fraction"][trial_row])
        escape = count > 0
        freeze = (
            not escape
            and math.isfinite(coverage)
            and coverage >= float(source.minimum_freeze_valid_fraction)
            and math.isfinite(low)
            and low >= float(source.freeze_fraction_threshold)
        )
        trial_arrays["trial_escape_speed_class"][trial_row] = escape
        trial_arrays["trial_freeze_candidate"][trial_row] = freeze
        if escape:
            trial_arrays["trial_response_class_code"][trial_row] = RESPONSE_CLASS_ESCAPE
        elif freeze:
            trial_arrays["trial_response_class_code"][trial_row] = RESPONSE_CLASS_FREEZE
        elif math.isfinite(coverage) and coverage >= float(
            source.minimum_freeze_valid_fraction
        ):
            trial_arrays["trial_response_class_code"][trial_row] = RESPONSE_CLASS_OTHER

    sweep_count = n_trials * sweep.size
    sweep_arrays: dict[str, np.ndarray] = {
        "sweep_row_id": np.arange(sweep_count, dtype=np.int64),
        "sweep_trial_row_id": np.repeat(
            np.arange(n_trials, dtype=np.int64), sweep.size
        ),
        "sweep_speed_threshold_mm_s": np.tile(sweep, n_trials).astype(np.float32),
        "sweep_escape_event_count": np.zeros(sweep_count, dtype=np.int64),
        "sweep_escape_event_rate_per_min": np.full(
            sweep_count, np.nan, dtype=np.float64
        ),
    }
    for row in range(sweep_count):
        trial_row = int(sweep_arrays["sweep_trial_row_id"][row])
        threshold = float(sweep_arrays["sweep_speed_threshold_mm_s"][row])
        mask = (
            bout_base_valid
            & (bout_all_trial == trial_row)
            & np.isfinite(bout_peak)
            & (bout_peak >= threshold)
        )
        count = int(np.count_nonzero(mask))
        sweep_arrays["sweep_escape_event_count"][row] = count
        seconds = float(trial_arrays["trial_valid_time_s"][trial_row])
        if seconds > 0:
            sweep_arrays["sweep_escape_event_rate_per_min"][row] = count / (
                seconds / 60.0
            )

    recording_arrays = {
        "recording_trial_count": np.asarray([n_trials], dtype=np.int64),
        "recording_escape_trial_count": np.asarray(
            [np.count_nonzero(trial_arrays["trial_escape_speed_class"])],
            dtype=np.int64,
        ),
        "recording_freeze_trial_count": np.asarray(
            [np.count_nonzero(trial_arrays["trial_freeze_candidate"])],
            dtype=np.int64,
        ),
        "recording_escape_event_count": np.asarray([n_events], dtype=np.int64),
        "recording_high_turn_escape_event_count": np.asarray(
            [np.count_nonzero(event_arrays["event_high_turn"])], dtype=np.int64
        ),
        "recording_trace_usable_event_count": np.asarray(
            [np.count_nonzero(event_arrays["event_trace_valid"])], dtype=np.int64
        ),
    }
    arrays = {**event_arrays, **trial_arrays, **sweep_arrays, **recording_arrays}
    readonly = {name: _readonly(values) for name, values in arrays.items()}
    manifest_body = {
        "schema_id": PREPARED_SCHEMA_ID,
        "schema_version": PREPARED_SCHEMA_VERSION,
        "scientific_schema": {
            "schema_id": SCHEMA_ID,
            "schema_version": SCHEMA_VERSION,
            "method_id": METHOD_ID,
            "event_unit": "speed_thresholded_exact_swim_bout_x_chaser",
            "trial_unit": "exact_logged_controller_trial",
        },
        "recording_id": recording_id,
        "sources": {
            "motion": {
                "run_path": source.source_motion_run_path,
                "manifest_sha256": source.source_motion_manifest_sha256,
                "speed_level": source.source_speed_level,
                "relative_frame_projection": dict(projection),
            },
            "controller_trial_payload_sha256": source.controller_trials.payload_digest,
            "bout_response_payload_sha256": source.bout_response.payload_digest,
            **(
                {"core_authority": core_authority} if core_authority is not None else {}
            ),
        },
        "parameters": {
            "escape_speed_threshold_mm_s": escape_threshold,
            "high_turn_threshold_deg": high_turn_threshold,
            "freeze_speed_threshold_mm_s": freeze_speed,
            "freeze_window_s": freeze_window,
            "freeze_fraction_threshold": float(source.freeze_fraction_threshold),
            "minimum_freeze_valid_fraction": float(
                source.minimum_freeze_valid_fraction
            ),
            "threshold_sweep_mm_s": sweep.tolist(),
        },
        "dimensions": {
            "n_trials": n_trials,
            "n_events": n_events,
            "n_sweep_rows": sweep_count,
        },
        "policy": {
            "speed_escape": "bout_peak_speed_greater_equal_threshold",
            "high_turn_tier": "optional_directed_annotation_separate_from_speed_class",
            "freeze": "no_speed_escape_and_low_speed_fraction_with_coverage_gate",
            "trial_attachment": "exactly_one_controller_trial_row_at_bout_onset",
            "event_counts": "retained_even_when_recapture_trace_unusable",
            "recapture": "first_post_event_exact_trial_member_at_or_below_onset_distance",
            "fallback_trial_segmentation": "prohibited",
            "trial_gaps": (
                "excluded_from_membership_time_and_event_attachment;"
                "retained_as_coverage_evidence"
            ),
        },
        "identity_registries": {
            "response_class": {
                "0": "insufficient_valid_freeze_window",
                "1": "speed_escape",
                "2": "freeze_candidate",
                "3": "other_response",
            },
            "trace_exclusion_reason": {
                "0": "valid",
                "1": "no_post_event_valid_distance_in_trial",
                "2": "event_frame_unavailable",
            },
        },
        "array_declarations": _declarations(readonly),
        "selector_eligible": False,
        "selection": "none",
        "production_authority": False,
        "registry_update": False,
    }
    manifest = _freeze(
        {
            **manifest_body,
            "payload_digest": canonical_json_sha256(manifest_body),
        }
    )
    return PreparedEscapeFreeze(
        recording_id=recording_id,
        n_trials=n_trials,
        n_events=n_events,
        arrays=MappingProxyType(readonly),
        manifest=manifest,
    )


def escape_freeze_input_from_handles(
    relative_frame: Any,
    provider_motion: Any,
    controller_trials: PreparedControllerTrials,
    bout_response: PreparedGeneralizedBoutResponse,
    *,
    track_id: int,
    speed_level: str = "filtered",
    escape_speed_threshold_mm_s: float = 20.0,
    high_turn_threshold_deg: float = 45.0,
    freeze_speed_threshold_mm_s: float = 2.0,
    freeze_window_s: float = 1.0,
    freeze_fraction_threshold: float = 0.8,
    minimum_freeze_valid_fraction: float = 0.5,
    threshold_sweep_mm_s: Sequence[float] = (10.0, 15.0, 20.0, 25.0, 30.0),
) -> EscapeFreezeInput:
    """Bind exact relative/motion handles and prepared response dependencies."""

    from fisheye.analysis_workflows.chaser_relative_frame_source_handle import (
        ChaserRelativeFrameSourceHandle,
    )
    from fisheye.analysis_workflows.core_motion_source_handle import (
        CoreMotionTrackSourceHandle,
        core_motion_dependency_record,
    )
    from fisheye.analysis_workflows.generalized_bout_response_successor import (
        exact_core_motion_frame_projection,
    )
    from fisheye.analysis_workflows.materializers.provider_epoch_behavior_summary import (
        SUPPORTED_SPEED_LEVELS,
        _track_slice,
    )
    from fisheye.analysis_workflows.provider_track_motion_source_handle import (
        ProviderTrackMotionSourceHandle,
    )

    if type(relative_frame) is not ChaserRelativeFrameSourceHandle:
        raise TypeError("relative_frame must be a strict loader-minted handle.")
    provider_mode = type(provider_motion) is ProviderTrackMotionSourceHandle
    core_mode = type(provider_motion) is CoreMotionTrackSourceHandle
    if not provider_mode and not core_mode:
        raise TypeError(
            "provider_motion must be one strict provider or core motion handle."
        )
    if type(controller_trials) is not PreparedControllerTrials:
        raise TypeError("controller_trials must be one prepared exact successor.")
    if type(bout_response) is not PreparedGeneralizedBoutResponse:
        raise TypeError("bout_response must be one prepared generalized successor.")
    if type(track_id) is not int or track_id < 0:
        _fail("track_id must be one non-negative exact integer.")
    if type(speed_level) is not str or speed_level not in SUPPORTED_SPEED_LEVELS:
        _fail(f"speed_level must be one of {SUPPORTED_SPEED_LEVELS!r}.")
    relative_frame.assert_current()
    if provider_mode:
        provider_motion.assert_current()
    else:
        provider_motion.assert_verified()
    if relative_frame.analysis_zarr_path != provider_motion.analysis_zarr_path:
        _fail("Escape/freeze sources do not belong to one exact archive.")
    if not (
        relative_frame.recording_id
        == controller_trials.recording_id
        == bout_response.recording_id
    ):
        _fail("Escape/freeze sources belong to different recordings.")
    if relative_frame.run_manifest.get("scale_policy", {}).get("unit") != "mm":
        _fail("Chaser-relative physical distance is not explicitly in millimeters.")
    relative_source = bout_response.manifest["sources"]["relative_frame"]
    motion_source = bout_response.manifest["sources"]["motion"]
    if (
        controller_trials.manifest["source_relative_frame"]["run_path"]
        != relative_frame.run_path
        or controller_trials.manifest["source_relative_frame"]["manifest_sha256"]
        != relative_frame.manifest_sha256
        or relative_source["run_path"] != relative_frame.run_path
        or relative_source["manifest_sha256"] != relative_frame.manifest_sha256
        or motion_source["run_path"] != provider_motion.run_path
        or motion_source["manifest_sha256"]
        != (
            provider_motion.provider_manifest_sha256
            if provider_mode
            else provider_motion.source_manifest_sha256
        )
        or bout_response.manifest["sources"]["controller_trial_payload_sha256"]
        != controller_trials.payload_digest
    ):
        _fail("Escape/freeze dependency binding is stale or mixed across sources.")
    if core_mode:
        core_envelope = relative_frame.context.get("core_authority")
        core_record = (
            core_envelope.get("record") if isinstance(core_envelope, Mapping) else None
        )
        response_core = bout_response.manifest["sources"].get("core_authority")
        if (
            not isinstance(core_record, Mapping)
            or not isinstance(response_core, Mapping)
            or core_record.get("core_authority_roster_sha256")
            != provider_motion.core_authority_roster_sha256
            or response_core.get("core_authority_roster_sha256")
            != provider_motion.core_authority_roster_sha256
            or core_record.get("core_motion", {}).get("track_id")
            != provider_motion.track_id
            or provider_motion.track_id != track_id
        ):
            _fail("Escape/freeze dependencies bind different core authority.")
    if provider_mode:
        rows = _track_slice(provider_motion, track_id=track_id)
        provider_frames = np.asarray(
            provider_motion.source_acquisition_frame_index[rows], dtype=np.int64
        )
        motion_manifest_sha256 = provider_motion.provider_manifest_sha256
    else:
        rows = slice(0, provider_motion.sample_count)
        provider_frames = np.asarray(
            provider_motion.array("source_acquisition_frame_index"),
            dtype=np.int64,
        )
        motion_manifest_sha256 = provider_motion.source_manifest_sha256
    acquisition_matrix = relative_frame.base_frame_chaser("acquisition_frame_id")
    timestamp_matrix = relative_frame.base_frame_chaser("timestamp_ns")
    timestamp_valid_matrix = relative_frame.base_frame_chaser("timestamp_valid")
    if relative_frame.n_frames and (
        not np.all(acquisition_matrix == acquisition_matrix[:, :1])
        or not np.all(timestamp_matrix == timestamp_matrix[:, :1])
        or not np.all(timestamp_valid_matrix == timestamp_valid_matrix[:, :1])
    ):
        _fail("Relative-frame acquisition/timing evidence differs across chasers.")
    relative_frames = np.asarray(acquisition_matrix[:, 0], dtype=np.int64)
    if provider_mode:
        provider_rows_by_relative, provider_present, provider_projection = (
            exact_provider_frame_projection(provider_frames, relative_frames)
        )
    else:
        provider_rows_by_relative, provider_present, provider_projection = (
            exact_core_motion_frame_projection(
                provider_frames,
                relative_frames,
                core_authority_roster_sha256=(
                    provider_motion.core_authority_roster_sha256
                ),
            )
        )
    try:
        if provider_mode:
            source_speed = np.asarray(
                provider_motion.array(f"speed_{speed_level}_mm")[rows],
                dtype=np.float64,
            )
            source_linear_valid = np.asarray(
                provider_motion.array("linear_sample_valid")[rows], dtype=bool
            )
        else:
            core_speed_path = {
                "filtered": "movement/speed/filtered/mm",
                "smoothed": "movement/speed/smoothed/mm",
            }.get(speed_level)
            if core_speed_path is None:
                _fail("Core motion supports only filtered or smoothed physical speed.")
            source_speed = np.asarray(
                provider_motion.array(core_speed_path),
                dtype=np.float64,
            )
            source_linear_valid = np.asarray(
                provider_motion.array("sample_valid"), dtype=bool
            )
    except KeyError as exc:
        raise EscapeFreezeSuccessorError(
            f"Provider motion lacks required speed level {speed_level!r}."
        ) from exc
    if (
        source_speed.shape != provider_frames.shape
        or source_linear_valid.shape != provider_frames.shape
    ):
        _fail("Provider speed/validity arrays differ from provider frame IDs.")
    speed = np.full(relative_frame.n_frames, np.nan, dtype=np.float64)
    linear_valid = np.zeros(relative_frame.n_frames, dtype=bool)
    speed[provider_present] = source_speed[provider_rows_by_relative[provider_present]]
    linear_valid[provider_present] = source_linear_valid[
        provider_rows_by_relative[provider_present]
    ]
    return EscapeFreezeInput(
        recording_id=relative_frame.recording_id,
        source_motion_run_path=provider_motion.run_path,
        source_motion_manifest_sha256=motion_manifest_sha256,
        source_speed_level=speed_level,
        source_motion_frame_projection=provider_projection,
        controller_trials=controller_trials,
        bout_response=bout_response,
        n_frames=relative_frame.n_frames,
        n_chasers=relative_frame.n_chasers,
        acquisition_frame_id_by_frame=relative_frames,
        timestamp_ns_by_frame=np.asarray(timestamp_matrix[:, 0], dtype=np.int64),
        timestamp_valid_by_frame=np.asarray(timestamp_valid_matrix[:, 0], dtype=bool),
        speed_mm_s_by_frame=speed,
        speed_valid_by_frame=linear_valid & np.isfinite(speed),
        chaser_identity_code=relative_frame.base_array("chaser_identity_code"),
        distance_mm=np.asarray(
            relative_frame.base_array("relative_distance_physical"),
            dtype=np.float64,
        ),
        distance_valid=relative_frame.base_array("relative_physical_valid"),
        source_core_authority=(
            core_motion_dependency_record(provider_motion) if core_mode else None
        ),
        escape_speed_threshold_mm_s=escape_speed_threshold_mm_s,
        high_turn_threshold_deg=high_turn_threshold_deg,
        freeze_speed_threshold_mm_s=freeze_speed_threshold_mm_s,
        freeze_window_s=freeze_window_s,
        freeze_fraction_threshold=freeze_fraction_threshold,
        minimum_freeze_valid_fraction=minimum_freeze_valid_fraction,
        threshold_sweep_mm_s=threshold_sweep_mm_s,
    )


def prepare_escape_freeze_successor_from_handles(
    relative_frame: Any,
    provider_motion: Any,
    controller_trials: PreparedControllerTrials,
    bout_response: PreparedGeneralizedBoutResponse,
    **kwargs: Any,
) -> PreparedEscapeFreeze:
    """Prepare escape/freeze from exact current archive dependencies."""

    return prepare_escape_freeze_successor(
        escape_freeze_input_from_handles(
            relative_frame,
            provider_motion,
            controller_trials,
            bout_response,
            **kwargs,
        )
    )


__all__ = [
    "METHOD_ID",
    "PREPARED_SCHEMA_ID",
    "PREPARED_SCHEMA_VERSION",
    "SCHEMA_ID",
    "SCHEMA_VERSION",
    "EscapeFreezeInput",
    "EscapeFreezeSuccessorError",
    "PreparedEscapeFreeze",
    "escape_freeze_input_from_handles",
    "prepare_escape_freeze_successor",
    "prepare_escape_freeze_successor_from_handles",
]
