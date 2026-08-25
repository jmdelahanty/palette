"""Provider-aware generalized bout-response successor.

This product asks a reusable question for every selected swim bout: where was
each chaser at bout onset, what did the bout do to fish--chaser separation, and
was the bout directed toward or away from that chaser when body-frame evidence
is available?  The distance/motion base is independent of body orientation;
directed fields form an explicitly optional extension over the same bout by
chaser rows.

One exact swim-bout signal is required.  The successor never concatenates or
discovers alternate signal levels and therefore cannot reproduce the legacy
multi-level bout duplication failure.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from fisheye.analysis_workflows.controller_trial_successor import (
    PreparedControllerTrials,
    SEMANTIC_ROLE_CODES,
    TRIAL_GAP_REASON_NOT_GAP,
    TRIAL_GAP_REASON_TRIAL_ID_MISMATCH,
    semantic_role_codes_from_handles as _semantic_role_codes_from_handles,
)
from fisheye.shared.coordinate_frame_record import array_values_sha256
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


SCHEMA_ID = "palette.analysis.generalized_chaser_bout_response"
SCHEMA_VERSION = 1
PREPARED_SCHEMA_ID = (
    "palette.analysis.generalized_chaser_bout_response.prepared_successor"
)
PREPARED_SCHEMA_VERSION = 1
METHOD_ID = "exact_signal_bout_x_chaser_distance_motion_with_body_extension_v1"

ROLE_CODES = SEMANTIC_ROLE_CODES
ATTACHMENT_REASON_VALID = 0
ATTACHMENT_REASON_FRAME_UNAVAILABLE = 1
ATTACHMENT_REASON_OUTSIDE_SEMANTIC_SELECTION = 2
ATTACHMENT_REASON_TRIAL_UNAVAILABLE = 3


class GeneralizedBoutResponseSuccessorError(ValueError):
    """Raised when exact bout-response rows cannot be constructed."""


def _fail(message: str) -> None:
    raise GeneralizedBoutResponseSuccessorError(message)


def _readonly(value: Any, *, dtype: Any | None = None) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def _digest(value: object, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{name} must be one lowercase SHA-256 digest.")
    return value


def _text(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        _fail(f"{name} must be one non-empty exact string.")
    return value


def _vector(
    value: Any,
    *,
    name: str,
    dtype: Any,
    size: int,
) -> np.ndarray:
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


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _wrap_deg(value: np.ndarray) -> np.ndarray:
    return (np.asarray(value, dtype=np.float64) + 180.0) % 360.0 - 180.0


@dataclass(frozen=True, slots=True)
class GeneralizedBoutResponseInput:
    recording_id: str
    source_relative_frame_run_path: str
    source_relative_frame_manifest_sha256: str
    source_motion_run_path: str
    source_motion_manifest_sha256: str
    source_swim_bout_run_path: str
    source_swim_bout_lineage_sha256: str
    source_signal_id: int
    source_signal_level: str
    source_semantic_selection_manifest_sha256: str
    source_controller_trial_payload_sha256: str
    source_motion_frame_projection: Mapping[str, Any]
    n_frames: int
    n_chasers: int
    acquisition_frame_id_by_frame: np.ndarray
    timestamp_ns_by_frame: np.ndarray
    timestamp_valid_by_frame: np.ndarray
    transition_valid_by_frame: np.ndarray
    semantic_role_code_by_frame: np.ndarray
    chaser_identity_code: np.ndarray
    distance_mm: np.ndarray
    distance_valid: np.ndarray
    controller_trial_row_id: np.ndarray
    controller_trial_envelope_row_id: np.ndarray
    controller_trial_gap_reason_code: np.ndarray
    bout_id: np.ndarray
    bout_start_acquisition_frame_id: np.ndarray
    bout_end_acquisition_frame_id: np.ndarray
    bout_peak_speed_mm_s: np.ndarray
    bout_mean_speed_mm_s: np.ndarray
    bout_duration_s: np.ndarray
    bout_path_length_mm: np.ndarray
    bout_net_displacement_mm: np.ndarray
    body_heading_deg_by_frame: np.ndarray | None = None
    body_heading_valid_by_frame: np.ndarray | None = None
    chaser_bearing_deg: np.ndarray | None = None
    chaser_bearing_valid: np.ndarray | None = None
    distance_bin_edges_mm: Sequence[float] = (0.0, 8.0, 16.0, 30.0, 50.0, math.inf)


@dataclass(frozen=True, slots=True)
class PreparedGeneralizedBoutResponse:
    recording_id: str
    n_bouts: int
    n_chasers: int
    n_bout_chaser_rows: int
    arrays: Mapping[str, np.ndarray]
    manifest: Mapping[str, Any]

    def array(self, name: str) -> np.ndarray:
        try:
            return self.arrays[name]
        except KeyError as exc:
            raise KeyError(f"Unknown generalized bout-response array {name!r}.") from exc

    @property
    def payload_digest(self) -> str:
        return str(self.manifest["payload_digest"])


def _array_declarations(arrays: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    return [
        {
            "path": name,
            "dtype": np.asarray(value).dtype.str,
            "shape": list(np.asarray(value).shape),
            "content_sha256": array_values_sha256(np.asarray(value)),
        }
        for name, value in sorted(arrays.items())
    ]


def _frame_lookup(frame_ids: np.ndarray) -> dict[int, int]:
    if np.unique(frame_ids).size != frame_ids.size:
        _fail("Acquisition frame identities are duplicated.")
    if frame_ids.size > 1 and np.any(np.diff(frame_ids) <= 0):
        _fail("Acquisition frame identities must be strictly increasing.")
    return {int(value): index for index, value in enumerate(frame_ids.tolist())}


def exact_provider_frame_projection(
    provider_frame_ids: np.ndarray,
    relative_frame_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build a fail-closed exact left join onto the relative-frame axis."""

    provider = np.asarray(provider_frame_ids)
    relative = np.asarray(relative_frame_ids)
    for values, name in ((provider, "provider"), (relative, "relative")):
        if values.dtype != np.dtype(np.int64) or values.ndim != 1:
            _fail(f"{name} acquisition frame IDs must be one int64 vector.")
        if values.size > 1 and np.any(np.diff(values) <= 0):
            _fail(
                f"{name} acquisition frame IDs must be unique and strictly increasing."
            )
    positions = np.searchsorted(provider, relative)
    present = positions < provider.size
    if provider.size:
        clipped = np.minimum(positions, provider.size - 1)
        present &= provider[clipped] == relative
    elif relative.size:
        present[:] = False
    rows = np.full(relative.size, -1, dtype=np.int64)
    rows[present] = positions[present]
    rows.setflags(write=False)
    present.setflags(write=False)
    matched = int(np.count_nonzero(present))
    record = {
        "schema_id": "palette.provider_motion.relative_frame_projection",
        "schema_version": 1,
        "join_key": "exact_acquisition_frame_id",
        "join_policy": "left_join_missing_provider_rows_invalid_no_interpolation",
        "provider_frame_count": int(provider.size),
        "relative_frame_count": int(relative.size),
        "matched_relative_frame_count": matched,
        "missing_relative_frame_count": int(relative.size - matched),
        "provider_only_frame_count": int(provider.size - matched),
        "provider_frame_ids_sha256": array_values_sha256(provider),
        "relative_frame_ids_sha256": array_values_sha256(relative),
        "provider_row_index_by_relative_frame_sha256": array_values_sha256(rows),
        "provider_frame_present_sha256": array_values_sha256(present),
        "fallback": "prohibited",
    }
    return rows, present, record


def _summary_arrays(
    *,
    roles: np.ndarray,
    chaser_codes: np.ndarray,
    distance: np.ndarray,
    distance_valid: np.ndarray,
    timestamp_ns: np.ndarray,
    timestamp_valid: np.ndarray,
    transition_valid: np.ndarray,
    onset_role: np.ndarray,
    onset_chaser: np.ndarray,
    onset_distance: np.ndarray,
    base_valid: np.ndarray,
    peak_speed: np.ndarray,
    duration: np.ndarray,
    path_length: np.ndarray,
    displacement: np.ndarray,
    edges: np.ndarray,
) -> dict[str, np.ndarray]:
    role_values = np.asarray(sorted(set(ROLE_CODES.values())), dtype=np.uint8)
    chaser_values = np.asarray(sorted(set(chaser_codes.tolist())), dtype=np.uint16)
    rows = int(role_values.size * chaser_values.size * (edges.size - 1))
    summary: dict[str, np.ndarray] = {
        "summary_role_code": np.zeros(rows, dtype=np.uint8),
        "summary_chaser_identity_code": np.zeros(rows, dtype=np.uint16),
        "summary_distance_bin_index": np.zeros(rows, dtype=np.int16),
        "summary_distance_bin_start_mm": np.zeros(rows, dtype=np.float32),
        "summary_distance_bin_end_mm": np.zeros(rows, dtype=np.float32),
        "summary_valid_time_s": np.zeros(rows, dtype=np.float64),
        "summary_bout_count": np.zeros(rows, dtype=np.int64),
        "summary_bout_rate_per_min": np.full(rows, np.nan, dtype=np.float64),
        "summary_median_peak_speed_mm_s": np.full(rows, np.nan, dtype=np.float64),
        "summary_median_duration_s": np.full(rows, np.nan, dtype=np.float64),
        "summary_median_path_length_mm": np.full(rows, np.nan, dtype=np.float64),
        "summary_median_net_displacement_mm": np.full(rows, np.nan, dtype=np.float64),
    }
    n_frames, n_chasers = distance.shape
    dt = np.zeros(n_frames, dtype=np.float64)
    if n_frames > 1:
        usable_dt = (
            timestamp_valid[:-1]
            & timestamp_valid[1:]
            & transition_valid[1:]
            & (timestamp_ns[1:] > timestamp_ns[:-1])
        )
        dt[:-1][usable_dt] = (
            timestamp_ns[1:][usable_dt] - timestamp_ns[:-1][usable_dt]
        ) / 1e9
    out = 0
    for role in role_values:
        for chaser_pos, code in enumerate(chaser_values):
            matches = np.flatnonzero(chaser_codes == code)
            if matches.size != 1:
                _fail("Chaser identity registry is duplicated or incomplete.")
            c = int(matches[0])
            for band in range(edges.size - 1):
                low, high = float(edges[band]), float(edges[band + 1])
                frame_mask = (
                    (roles == role)
                    & distance_valid[:, c]
                    & np.isfinite(distance[:, c])
                    & (distance[:, c] >= low)
                    & (distance[:, c] < high)
                )
                seconds = float(np.sum(dt[frame_mask]))
                bout_mask = (
                    base_valid
                    & (onset_role == role)
                    & (onset_chaser == code)
                    & np.isfinite(onset_distance)
                    & (onset_distance >= low)
                    & (onset_distance < high)
                )
                summary["summary_role_code"][out] = role
                summary["summary_chaser_identity_code"][out] = code
                summary["summary_distance_bin_index"][out] = band
                summary["summary_distance_bin_start_mm"][out] = low
                summary["summary_distance_bin_end_mm"][out] = high
                summary["summary_valid_time_s"][out] = seconds
                count = int(np.count_nonzero(bout_mask))
                summary["summary_bout_count"][out] = count
                if seconds > 0:
                    summary["summary_bout_rate_per_min"][out] = count / (seconds / 60.0)
                for values, name in (
                    (peak_speed, "summary_median_peak_speed_mm_s"),
                    (duration, "summary_median_duration_s"),
                    (path_length, "summary_median_path_length_mm"),
                    (displacement, "summary_median_net_displacement_mm"),
                ):
                    finite = bout_mask & np.isfinite(values)
                    if np.any(finite):
                        summary[name][out] = float(np.median(values[finite]))
                out += 1
    return summary


def prepare_generalized_bout_response_successor(
    source: GeneralizedBoutResponseInput,
) -> PreparedGeneralizedBoutResponse:
    """Prepare exact bout-by-chaser facts and valid-time band summaries."""

    if type(source) is not GeneralizedBoutResponseInput:
        raise TypeError("source must be one GeneralizedBoutResponseInput.")
    recording_id = _text(source.recording_id, name="recording_id")
    for name in (
        "source_relative_frame_run_path",
        "source_motion_run_path",
        "source_swim_bout_run_path",
        "source_signal_level",
    ):
        _text(getattr(source, name), name=name)
    for name in (
        "source_relative_frame_manifest_sha256",
        "source_motion_manifest_sha256",
        "source_swim_bout_lineage_sha256",
        "source_semantic_selection_manifest_sha256",
        "source_controller_trial_payload_sha256",
    ):
        _digest(getattr(source, name), name=name)
    if type(source.source_signal_id) is not int or source.source_signal_id < 0:
        _fail("source_signal_id must be one non-negative exact integer.")
    if type(source.n_frames) is not int or source.n_frames < 0:
        _fail("n_frames must be one non-negative exact integer.")
    if type(source.n_chasers) is not int or source.n_chasers <= 0:
        _fail("n_chasers must be one positive exact integer.")
    projection = _plain(source.source_motion_frame_projection)
    if (
        not isinstance(projection, dict)
        or projection.get("schema_id")
        != "palette.provider_motion.relative_frame_projection"
        or projection.get("schema_version") != 1
        or projection.get("join_policy")
        != "left_join_missing_provider_rows_invalid_no_interpolation"
        or projection.get("relative_frame_count") != source.n_frames
        or projection.get("fallback") != "prohibited"
    ):
        _fail("source_motion_frame_projection is absent, malformed, or permissive.")
    n_frames, n_chasers = source.n_frames, source.n_chasers
    n_rows = n_frames * n_chasers
    frame_ids = _vector(
        source.acquisition_frame_id_by_frame,
        name="acquisition_frame_id_by_frame",
        dtype=np.int64,
        size=n_frames,
    )
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
    transition_valid = _vector(
        source.transition_valid_by_frame,
        name="transition_valid_by_frame",
        dtype=bool,
        size=n_frames,
    )
    roles = _vector(
        source.semantic_role_code_by_frame,
        name="semantic_role_code_by_frame",
        dtype=np.uint8,
        size=n_frames,
    )
    if np.any(~np.isin(roles, np.asarray([0, *ROLE_CODES.values()], dtype=np.uint8))):
        _fail("semantic_role_code_by_frame contains an unknown role code.")
    chaser_flat = _vector(
        source.chaser_identity_code,
        name="chaser_identity_code",
        dtype=np.uint16,
        size=n_rows,
    )
    chaser_matrix = chaser_flat.reshape(n_frames, n_chasers)
    if n_frames and np.any(chaser_matrix != chaser_matrix[:1, :]):
        _fail("Chaser identity changed along the fixed chaser axis.")
    chaser_codes = (
        chaser_matrix[0]
        if n_frames
        else np.arange(1, n_chasers + 1, dtype=np.uint16)
    )
    if np.unique(chaser_codes).size != n_chasers:
        _fail("Chaser identity codes are duplicated.")
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
    if np.any(distance_valid & ~np.isfinite(distance)):
        _fail("A valid physical distance is non-finite.")
    trial_by_row = _vector(
        source.controller_trial_row_id,
        name="controller_trial_row_id",
        dtype=np.int64,
        size=n_rows,
    ).reshape(n_frames, n_chasers)
    trial_envelope_by_row = _vector(
        source.controller_trial_envelope_row_id,
        name="controller_trial_envelope_row_id",
        dtype=np.int64,
        size=n_rows,
    ).reshape(n_frames, n_chasers)
    trial_gap_reason_by_row = _vector(
        source.controller_trial_gap_reason_code,
        name="controller_trial_gap_reason_code",
        dtype=np.uint8,
        size=n_rows,
    ).reshape(n_frames, n_chasers)
    if np.any(trial_by_row < -1) or np.any(trial_envelope_by_row < -1):
        _fail("Controller-trial row identities must use -1 or a nonnegative row.")
    if np.any(trial_gap_reason_by_row > TRIAL_GAP_REASON_TRIAL_ID_MISMATCH):
        _fail("controller_trial_gap_reason_code contains an unknown registry code.")
    exact_member = trial_by_row >= 0
    envelope_member = trial_envelope_by_row >= 0
    gap_member = envelope_member & ~exact_member
    if np.any(exact_member & (trial_envelope_by_row != trial_by_row)):
        _fail("Exact controller-trial membership differs from envelope identity.")
    if np.any(~envelope_member & (trial_gap_reason_by_row != TRIAL_GAP_REASON_NOT_GAP)):
        _fail("A controller-trial gap reason exists outside every trial envelope.")
    if np.any(exact_member & (trial_gap_reason_by_row != TRIAL_GAP_REASON_NOT_GAP)):
        _fail("An exact controller-trial member is incorrectly reason-coded as a gap.")
    if np.any(gap_member & (trial_gap_reason_by_row == TRIAL_GAP_REASON_NOT_GAP)):
        _fail("A controller-trial envelope gap lacks a reason code.")

    n_bouts = int(np.asarray(source.bout_id).size)
    bout_id = _vector(source.bout_id, name="bout_id", dtype=np.int64, size=n_bouts)
    if np.unique(bout_id).size != n_bouts:
        _fail("Selected signal contains duplicate bout IDs.")
    start_ids = _vector(
        source.bout_start_acquisition_frame_id,
        name="bout_start_acquisition_frame_id",
        dtype=np.int64,
        size=n_bouts,
    )
    end_ids = _vector(
        source.bout_end_acquisition_frame_id,
        name="bout_end_acquisition_frame_id",
        dtype=np.int64,
        size=n_bouts,
    )
    if np.any(end_ids < start_ids):
        _fail("A selected bout ends before it starts.")
    bout_metrics = {
        "bout_peak_speed_mm_s": _float_vector(
            source.bout_peak_speed_mm_s,
            name="bout_peak_speed_mm_s",
            size=n_bouts,
        ),
        "bout_mean_speed_mm_s": _float_vector(
            source.bout_mean_speed_mm_s,
            name="bout_mean_speed_mm_s",
            size=n_bouts,
        ),
        "bout_duration_s": _float_vector(
            source.bout_duration_s,
            name="bout_duration_s",
            size=n_bouts,
        ),
        "bout_path_length_mm": _float_vector(
            source.bout_path_length_mm,
            name="bout_path_length_mm",
            size=n_bouts,
        ),
        "bout_net_displacement_mm": _float_vector(
            source.bout_net_displacement_mm,
            name="bout_net_displacement_mm",
            size=n_bouts,
        ),
    }
    edges = np.asarray(tuple(source.distance_bin_edges_mm), dtype=np.float64)
    if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0):
        _fail("distance_bin_edges_mm must be one strictly increasing vector.")

    lookup = _frame_lookup(frame_ids)
    start_row = np.asarray([lookup.get(int(value), -1) for value in start_ids], dtype=np.int64)
    end_row = np.asarray([lookup.get(int(value), -1) for value in end_ids], dtype=np.int64)
    frame_available = (start_row >= 0) & (end_row >= 0)
    role = np.zeros(n_bouts, dtype=np.uint8)
    role[frame_available] = roles[start_row[frame_available]]

    pair_count = n_bouts * n_chasers
    pair_bout_index = np.repeat(np.arange(n_bouts, dtype=np.int64), n_chasers)
    pair_chaser_pos = np.tile(np.arange(n_chasers, dtype=np.int64), n_bouts)
    pair_start_row = np.repeat(start_row, n_chasers)
    pair_end_row = np.repeat(end_row, n_chasers)
    pair_frame_available = np.repeat(frame_available, n_chasers)
    pair_role = np.repeat(role, n_chasers)
    pair_chaser_code = np.tile(chaser_codes, n_bouts)
    pair_distance_onset = np.full(pair_count, np.nan, dtype=np.float64)
    pair_distance_end = np.full(pair_count, np.nan, dtype=np.float64)
    pair_trial = np.full(pair_count, -1, dtype=np.int64)
    pair_trial_envelope = np.full(pair_count, -1, dtype=np.int64)
    pair_trial_gap_reason = np.full(
        pair_count, TRIAL_GAP_REASON_NOT_GAP, dtype=np.uint8
    )
    base_valid = np.zeros(pair_count, dtype=bool)
    for row in np.flatnonzero(pair_frame_available):
        f0 = int(pair_start_row[row])
        f1 = int(pair_end_row[row])
        c = int(pair_chaser_pos[row])
        if distance_valid[f0, c]:
            pair_distance_onset[row] = distance[f0, c]
        if distance_valid[f1, c]:
            pair_distance_end[row] = distance[f1, c]
        pair_trial[row] = trial_by_row[f0, c]
        pair_trial_envelope[row] = trial_envelope_by_row[f0, c]
        pair_trial_gap_reason[row] = trial_gap_reason_by_row[f0, c]
        base_valid[row] = (
            pair_role[row] != 0
            and distance_valid[f0, c]
            and distance_valid[f1, c]
        )
    pair_delta = pair_distance_end - pair_distance_onset
    reason = np.full(
        pair_count,
        ATTACHMENT_REASON_VALID,
        dtype=np.uint8,
    )
    reason[~pair_frame_available] = ATTACHMENT_REASON_FRAME_UNAVAILABLE
    reason[pair_frame_available & (pair_role == 0)] = (
        ATTACHMENT_REASON_OUTSIDE_SEMANTIC_SELECTION
    )
    reason[
        pair_frame_available & (pair_role != 0) & (pair_trial < 0)
    ] = ATTACHMENT_REASON_TRIAL_UNAVAILABLE

    body_present = all(
        value is not None
        for value in (
            source.body_heading_deg_by_frame,
            source.body_heading_valid_by_frame,
            source.chaser_bearing_deg,
            source.chaser_bearing_valid,
        )
    )
    if body_present:
        heading = _float_vector(
            source.body_heading_deg_by_frame,
            name="body_heading_deg_by_frame",
            size=n_frames,
        )
        heading_valid = _vector(
            source.body_heading_valid_by_frame,
            name="body_heading_valid_by_frame",
            dtype=bool,
            size=n_frames,
        )
        bearing = _float_vector(
            source.chaser_bearing_deg,
            name="chaser_bearing_deg",
            size=n_rows,
        ).reshape(n_frames, n_chasers)
        bearing_valid = _vector(
            source.chaser_bearing_valid,
            name="chaser_bearing_valid",
            dtype=bool,
            size=n_rows,
        ).reshape(n_frames, n_chasers)
    elif any(
        value is not None
        for value in (
            source.body_heading_deg_by_frame,
            source.body_heading_valid_by_frame,
            source.chaser_bearing_deg,
            source.chaser_bearing_valid,
        )
    ):
        _fail("Body-frame bout-response inputs must be all present or all absent.")
    else:
        heading = np.full(n_frames, np.nan, dtype=np.float64)
        heading_valid = np.zeros(n_frames, dtype=bool)
        bearing = np.full((n_frames, n_chasers), np.nan, dtype=np.float64)
        bearing_valid = np.zeros((n_frames, n_chasers), dtype=bool)

    turn = np.full(n_bouts, np.nan, dtype=np.float64)
    turn_valid = frame_available.copy()
    turn_valid[frame_available] &= (
        heading_valid[start_row[frame_available]]
        & heading_valid[end_row[frame_available]]
    )
    turn[turn_valid] = _wrap_deg(
        heading[end_row[turn_valid]] - heading[start_row[turn_valid]]
    )
    pair_turn = np.repeat(turn, n_chasers)
    pair_bearing = np.full(pair_count, np.nan, dtype=np.float64)
    directed_valid = np.zeros(pair_count, dtype=bool)
    for row in np.flatnonzero(pair_frame_available):
        bout = int(pair_bout_index[row])
        f0 = int(pair_start_row[row])
        c = int(pair_chaser_pos[row])
        if bearing_valid[f0, c]:
            pair_bearing[row] = bearing[f0, c]
        directed_valid[row] = (
            base_valid[row]
            and turn_valid[bout]
            and bearing_valid[f0, c]
            and np.isfinite(pair_bearing[row])
        )
    turn_toward = np.zeros(pair_count, dtype=bool)
    turn_toward[directed_valid] = (
        np.sign(pair_turn[directed_valid])
        == np.sign(pair_bearing[directed_valid])
    )

    arrays: dict[str, np.ndarray] = {
        "bout_chaser_row_id": np.arange(pair_count, dtype=np.int64),
        "bout_row_id": pair_bout_index,
        "bout_id": np.repeat(bout_id, n_chasers),
        "chaser_identity_code": pair_chaser_code,
        "source_signal_id": np.full(pair_count, source.source_signal_id, dtype=np.int32),
        "start_acquisition_frame_id": np.repeat(start_ids, n_chasers),
        "end_acquisition_frame_id": np.repeat(end_ids, n_chasers),
        "semantic_role_code": pair_role,
        "controller_trial_row_id": pair_trial,
        "controller_trial_envelope_row_id": pair_trial_envelope,
        "controller_trial_gap_reason_code": pair_trial_gap_reason,
        "attachment_reason_code": reason,
        "base_valid": base_valid,
        "directed_valid": directed_valid,
        "distance_at_onset_mm": pair_distance_onset.astype(np.float32),
        "distance_at_end_mm": pair_distance_end.astype(np.float32),
        "delta_distance_mm": pair_delta.astype(np.float32),
        "bearing_at_onset_deg": pair_bearing.astype(np.float32),
        "turn_deg": pair_turn.astype(np.float32),
        "turn_toward_chaser": turn_toward,
    }
    for name, values in bout_metrics.items():
        arrays[name] = np.repeat(values, n_chasers).astype(np.float32)
    arrays["bout_tortuosity"] = np.divide(
        arrays["bout_path_length_mm"],
        arrays["bout_net_displacement_mm"],
        out=np.full(pair_count, np.nan, dtype=np.float32),
        where=arrays["bout_net_displacement_mm"] > 1e-6,
    )
    arrays.update(
        _summary_arrays(
            roles=roles,
            chaser_codes=chaser_codes,
            distance=distance,
            distance_valid=distance_valid,
            timestamp_ns=timestamp,
            timestamp_valid=timestamp_valid,
            transition_valid=transition_valid,
            onset_role=pair_role,
            onset_chaser=pair_chaser_code,
            onset_distance=pair_distance_onset,
            base_valid=base_valid,
            peak_speed=arrays["bout_peak_speed_mm_s"],
            duration=arrays["bout_duration_s"],
            path_length=arrays["bout_path_length_mm"],
            displacement=arrays["bout_net_displacement_mm"],
            edges=edges,
        )
    )
    readonly = {name: _readonly(value) for name, value in arrays.items()}
    body = {
        "schema_id": PREPARED_SCHEMA_ID,
        "schema_version": PREPARED_SCHEMA_VERSION,
        "scientific_schema": {
            "schema_id": SCHEMA_ID,
            "schema_version": SCHEMA_VERSION,
            "method_id": METHOD_ID,
            "row_unit": "selected_swim_bout_x_chaser",
            "summary_unit": "semantic_role_x_chaser_x_distance_band",
            "body_extension_present": body_present,
        },
        "recording_id": recording_id,
        "sources": {
            "relative_frame": {
                "run_path": source.source_relative_frame_run_path,
                "manifest_sha256": source.source_relative_frame_manifest_sha256,
            },
            "motion": {
                "run_path": source.source_motion_run_path,
                "manifest_sha256": source.source_motion_manifest_sha256,
                "relative_frame_projection": projection,
            },
            "swim_bouts": {
                "run_path": source.source_swim_bout_run_path,
                "lineage_sha256": source.source_swim_bout_lineage_sha256,
                "signal_id": source.source_signal_id,
                "signal_level": source.source_signal_level,
            },
            "semantic_selection_manifest_sha256": (
                source.source_semantic_selection_manifest_sha256
            ),
            "controller_trial_payload_sha256": (
                source.source_controller_trial_payload_sha256
            ),
        },
        "dimensions": {
            "n_frames": n_frames,
            "n_chasers": n_chasers,
            "n_bouts": n_bouts,
            "n_bout_chaser_rows": pair_count,
            "n_summary_rows": int(readonly["summary_role_code"].size),
        },
        "distance_bin_edges_mm": [
            float(value) if np.isfinite(value) else None for value in edges
        ],
        "policy": {
            "bout_signal": "one_explicit_default_signal_only",
            "bout_attachment": "exact_acquisition_frame_identity",
            "trial_attachment": "onset_row_exact_controller_trial_membership",
            "trial_envelope": (
                "retained_for_visualization_and_censoring_not_event_membership"
            ),
            "rate_denominator": "valid_transition_time_in_distance_band",
            "directed_metrics": "optional_body_frame_extension_no_motion_heading_fallback",
            "unattached_bouts": "retained_with_reason_code",
        },
        "identity_registries": {
            "semantic_role": {str(value): key for key, value in ROLE_CODES.items()},
            "attachment_reason": {
                "0": "valid_or_trial_optional",
                "1": "frame_unavailable",
                "2": "outside_semantic_selection",
                "3": "controller_trial_unavailable_at_onset",
            },
        },
        "array_declarations": _array_declarations(readonly),
        "selector_eligible": False,
        "selection": "none",
        "production_authority": False,
        "registry_update": False,
    }
    manifest = _freeze({**body, "payload_digest": canonical_json_sha256(body)})
    return PreparedGeneralizedBoutResponse(
        recording_id=recording_id,
        n_bouts=n_bouts,
        n_chasers=n_chasers,
        n_bout_chaser_rows=pair_count,
        arrays=MappingProxyType(readonly),
        manifest=manifest,
    )


def semantic_role_codes_from_handles(
    relative_frame: Any,
    semantic_selection: Any,
) -> np.ndarray:
    """Project exact semantic role bounds onto a relative acquisition axis."""
    return _semantic_role_codes_from_handles(relative_frame, semantic_selection)


def generalized_bout_response_input_from_handles(
    relative_frame: Any,
    semantic_selection: Any,
    controller_trials: PreparedControllerTrials,
    provider_motion: Any,
    *,
    swim_bout_run_name: str,
    track_id: int,
    include_body_extension: bool = True,
    distance_bin_edges_mm: Sequence[float] = (
        0.0,
        8.0,
        16.0,
        30.0,
        50.0,
        math.inf,
    ),
) -> GeneralizedBoutResponseInput:
    """Bind exact archive handles into the generalized successor input.

    The swim-bout source must be an explicitly named complete
    selector-ineligible v8 candidate already sealed to this provider-motion
    track.  Direct and consolidated reads must yield the same selected bout
    table and binding.
    """

    from fisheye.analysis.swim_bout_io import (
        load_exact_selector_ineligible_default_swim_bout_tables,
    )
    from fisheye.analysis_workflows.chaser_relative_frame_source_handle import (
        ChaserRelativeFrameSourceHandle,
    )
    from fisheye.analysis_workflows.materializers.provider_epoch_behavior_summary import (
        _swim_bout_binding,
        _track_slice,
    )
    from fisheye.analysis_workflows.protocol_semantic_chaser_selection_publication import (
        ProtocolSemanticChaserSelectionSourceHandle,
    )
    from fisheye.analysis_workflows.provider_track_motion_source_handle import (
        ProviderTrackMotionSourceHandle,
    )
    from fisheye.shared.zarr_io import open_zarr_root

    if type(relative_frame) is not ChaserRelativeFrameSourceHandle:
        raise TypeError("relative_frame must be a strict loader-minted handle.")
    if type(semantic_selection) is not ProtocolSemanticChaserSelectionSourceHandle:
        raise TypeError("semantic_selection must be a strict loader-minted handle.")
    if type(provider_motion) is not ProviderTrackMotionSourceHandle:
        raise TypeError("provider_motion must be a strict loader-minted handle.")
    if type(controller_trials) is not PreparedControllerTrials:
        raise TypeError("controller_trials must be one prepared exact successor.")
    if type(swim_bout_run_name) is not str or not swim_bout_run_name.strip():
        _fail("swim_bout_run_name must be one explicit non-empty run name.")
    if type(track_id) is not int or track_id < 0:
        _fail("track_id must be one non-negative exact integer.")
    if type(include_body_extension) is not bool:
        _fail("include_body_extension must be the exact boolean.")
    relative_frame.assert_current()
    semantic_selection.assert_current()
    provider_motion.assert_current()
    if not (
        relative_frame.analysis_zarr_path == provider_motion.analysis_zarr_path
        == semantic_selection.analysis_zarr
    ):
        _fail("Successor sources do not belong to one exact analysis archive.")
    if not (
        relative_frame.recording_id
        == semantic_selection.recording_id
        == controller_trials.recording_id
    ):
        _fail("Successor sources belong to different recordings.")
    if relative_frame.run_manifest.get("scale_policy", {}).get("unit") != "mm":
        _fail("Chaser-relative physical distance is not explicitly in millimeters.")
    if (
        controller_trials.manifest["source_relative_frame"]["run_path"]
        != relative_frame.run_path
        or controller_trials.manifest["source_relative_frame"]["manifest_sha256"]
        != relative_frame.manifest_sha256
        or controller_trials.manifest["semantic_selection"]["manifest_sha256"]
        != semantic_selection.manifest_sha256
    ):
        _fail("Controller-trial dependency is not bound to the exact input handles.")

    rows = _track_slice(provider_motion, track_id=track_id)
    provider_frames = np.asarray(
        provider_motion.source_acquisition_frame_index[rows], dtype=np.int64
    )
    relative_frames_matrix = relative_frame.base_frame_chaser("acquisition_frame_id")
    relative_frames = (
        np.asarray(relative_frames_matrix[:, 0], dtype=np.int64)
        if relative_frame.n_frames
        else np.asarray([], dtype=np.int64)
    )
    provider_rows_by_relative, provider_present, provider_projection = (
        exact_provider_frame_projection(provider_frames, relative_frames)
    )

    archive = relative_frame.analysis_zarr_path
    root_consolidated = open_zarr_root(archive, mode="r", use_consolidated=True)
    root_direct = open_zarr_root(archive, mode="r", use_consolidated=False)
    try:
        tables = load_exact_selector_ineligible_default_swim_bout_tables(
            root_consolidated, run_name=swim_bout_run_name
        )
        direct_tables = load_exact_selector_ineligible_default_swim_bout_tables(
            root_direct, run_name=swim_bout_run_name
        )
        binding, lineage_hash, _frame_hash = _swim_bout_binding(
            tables,
            provider=provider_motion,
            rows=rows,
            track_id=track_id,
        )
        direct_binding, direct_lineage_hash, _direct_frame_hash = _swim_bout_binding(
            direct_tables,
            provider=provider_motion,
            rows=rows,
            track_id=track_id,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise GeneralizedBoutResponseSuccessorError(
            f"Exact swim-bout source validation failed: {exc}"
        ) from exc
    if (
        binding != direct_binding
        or lineage_hash != direct_lineage_hash
        or array_values_sha256(tables.bouts)
        != array_values_sha256(direct_tables.bouts)
    ):
        _fail("Swim-bout direct and consolidated selections differ.")

    bouts = np.asarray(tables.bouts)
    names = set(bouts.dtype.names or ())
    required_fields = {
        "bout_id",
        "start_frame",
        "end_frame",
        "peak_physical_speed_mm_s",
        "mean_speed_mm_s",
        "duration_s",
        "path_length_mm",
        "net_displacement_mm",
    }
    missing_fields = sorted(required_fields - names)
    if missing_fields:
        _fail(f"Exact swim-bout table lacks required fields: {missing_fields!r}.")
    roles = semantic_role_codes_from_handles(relative_frame, semantic_selection)
    timestamp = relative_frame.base_frame_chaser("timestamp_ns")
    timestamp_valid = relative_frame.base_frame_chaser("timestamp_valid")
    if relative_frame.n_frames and (
        not np.all(timestamp == timestamp[:, :1])
        or not np.all(timestamp_valid == timestamp_valid[:, :1])
    ):
        _fail("Relative-frame timing evidence is not constant across chasers.")
    source_transition_valid = np.asarray(
        provider_motion.array("transition_valid")[rows], dtype=bool
    )
    if source_transition_valid.shape != provider_frames.shape:
        _fail("Provider transition-validity axis differs from provider frame IDs.")
    transition_valid = np.zeros(relative_frame.n_frames, dtype=bool)
    transition_valid[provider_present] = source_transition_valid[
        provider_rows_by_relative[provider_present]
    ]

    body_heading: np.ndarray | None = None
    body_heading_valid: np.ndarray | None = None
    bearing: np.ndarray | None = None
    bearing_valid: np.ndarray | None = None
    if include_body_extension:
        if not relative_frame.body_available:
            _fail("Requested generalized body extension is unavailable.")
        heading_matrix = relative_frame.body_frame_chaser("body_heading_deg")
        heading_valid_matrix = relative_frame.body_frame_chaser("body_heading_valid")
        if relative_frame.n_frames and (
            not np.allclose(heading_matrix, heading_matrix[:, :1], equal_nan=True)
            or not np.all(heading_valid_matrix == heading_valid_matrix[:, :1])
        ):
            _fail("Relative-frame body heading is not constant across chasers.")
        body_heading = np.asarray(heading_matrix[:, 0], dtype=np.float64)
        body_heading_valid = np.asarray(heading_valid_matrix[:, 0], dtype=bool)
        bearing = np.asarray(
            relative_frame.body_array("body_bearing_deg"), dtype=np.float64
        )
        bearing_valid = np.asarray(
            relative_frame.body_array("body_bearing_valid"), dtype=bool
        )

    return GeneralizedBoutResponseInput(
        recording_id=relative_frame.recording_id,
        source_relative_frame_run_path=relative_frame.run_path,
        source_relative_frame_manifest_sha256=relative_frame.manifest_sha256,
        source_motion_run_path=provider_motion.run_path,
        source_motion_manifest_sha256=provider_motion.provider_manifest_sha256,
        source_swim_bout_run_path=str(binding["run_path"]),
        source_swim_bout_lineage_sha256=lineage_hash,
        source_signal_id=int(tables.signal.signal_id),
        source_signal_level=str(tables.signal.speed_level),
        source_semantic_selection_manifest_sha256=semantic_selection.manifest_sha256,
        source_controller_trial_payload_sha256=controller_trials.payload_digest,
        source_motion_frame_projection=provider_projection,
        n_frames=relative_frame.n_frames,
        n_chasers=relative_frame.n_chasers,
        acquisition_frame_id_by_frame=relative_frames,
        timestamp_ns_by_frame=np.asarray(timestamp[:, 0], dtype=np.int64),
        timestamp_valid_by_frame=np.asarray(timestamp_valid[:, 0], dtype=bool),
        transition_valid_by_frame=transition_valid,
        semantic_role_code_by_frame=roles,
        chaser_identity_code=relative_frame.base_array("chaser_identity_code"),
        distance_mm=np.asarray(
            relative_frame.base_array("relative_distance_physical"), dtype=np.float64
        ),
        distance_valid=relative_frame.base_array("relative_physical_valid"),
        controller_trial_row_id=controller_trials.array(
            "trial_row_id_by_source_row"
        ),
        controller_trial_envelope_row_id=controller_trials.array(
            "trial_envelope_row_id_by_source_row"
        ),
        controller_trial_gap_reason_code=controller_trials.array(
            "trial_gap_reason_code_by_source_row"
        ),
        bout_id=np.asarray(bouts["bout_id"], dtype=np.int64),
        bout_start_acquisition_frame_id=np.asarray(
            bouts["start_frame"], dtype=np.int64
        ),
        bout_end_acquisition_frame_id=np.asarray(bouts["end_frame"], dtype=np.int64),
        bout_peak_speed_mm_s=np.asarray(
            bouts["peak_physical_speed_mm_s"], dtype=np.float64
        ),
        bout_mean_speed_mm_s=np.asarray(bouts["mean_speed_mm_s"], dtype=np.float64),
        bout_duration_s=np.asarray(bouts["duration_s"], dtype=np.float64),
        bout_path_length_mm=np.asarray(bouts["path_length_mm"], dtype=np.float64),
        bout_net_displacement_mm=np.asarray(
            bouts["net_displacement_mm"], dtype=np.float64
        ),
        body_heading_deg_by_frame=body_heading,
        body_heading_valid_by_frame=body_heading_valid,
        chaser_bearing_deg=bearing,
        chaser_bearing_valid=bearing_valid,
        distance_bin_edges_mm=distance_bin_edges_mm,
    )


def prepare_generalized_bout_response_successor_from_handles(
    relative_frame: Any,
    semantic_selection: Any,
    controller_trials: PreparedControllerTrials,
    provider_motion: Any,
    **kwargs: Any,
) -> PreparedGeneralizedBoutResponse:
    """Prepare the generalized successor from exact current archive inputs."""

    return prepare_generalized_bout_response_successor(
        generalized_bout_response_input_from_handles(
            relative_frame,
            semantic_selection,
            controller_trials,
            provider_motion,
            **kwargs,
        )
    )


__all__ = [
    "METHOD_ID",
    "PREPARED_SCHEMA_ID",
    "PREPARED_SCHEMA_VERSION",
    "ROLE_CODES",
    "SCHEMA_ID",
    "SCHEMA_VERSION",
    "GeneralizedBoutResponseInput",
    "GeneralizedBoutResponseSuccessorError",
    "PreparedGeneralizedBoutResponse",
    "exact_provider_frame_projection",
    "generalized_bout_response_input_from_handles",
    "prepare_generalized_bout_response_successor",
    "prepare_generalized_bout_response_successor_from_handles",
    "semantic_role_codes_from_handles",
]
