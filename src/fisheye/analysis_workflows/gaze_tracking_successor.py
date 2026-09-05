"""Exact-source gaze-to-chaser tracking successor.

Gaze and chaser bearing are compared only in the fish body frame.  The
successor requires independently valid eye-orientation and body-frame bearing
sources, keeps invalid rows explicit, and never substitutes world-frame gaze,
motion heading, or nasal-positive eye-angle fields.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from fisheye.analysis_workflows.generalized_bout_response_successor import (
    ROLE_CODES,
)
from fisheye.analysis_workflows.core_paradigm_authority import (
    core_paradigm_dependency_from_relative_frame,
    validate_core_paradigm_source_dependency,
)
from fisheye.shared.coordinate_frame_record import array_values_sha256
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256

SCHEMA_ID = "palette.analysis.chaser_gaze_tracking"
SCHEMA_VERSION = 3
PREPARED_SCHEMA_ID = "palette.analysis.chaser_gaze_tracking.prepared_successor"
PREPARED_SCHEMA_VERSION = 2
METHOD_ID = "exact_eye_body_frame_gaze_real_rotated_controls_dynamic_v2"

DEFAULT_VIRTUAL_ROTATIONS_DEG = (60.0, 120.0, 180.0, 240.0, 300.0)

EYE_LEFT = 1
EYE_RIGHT = 2


class GazeTrackingSuccessorError(ValueError):
    """Raised when an exact gaze successor cannot be prepared."""


def _fail(message: str) -> None:
    raise GazeTrackingSuccessorError(message)


def _readonly(value: Any) -> np.ndarray:
    result = np.array(value, copy=True, order="C")
    result.setflags(write=False)
    return result


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


def _float_array(value: Any, *, name: str, shape: tuple[int, ...]) -> np.ndarray:
    result = np.asarray(value)
    if result.dtype.kind != "f" or result.shape != shape:
        _fail(f"{name} must be one floating array with shape {shape!r}.")
    return np.asarray(result, dtype=np.float64)


def _wrap_deg(value: Any) -> np.ndarray:
    return (np.asarray(value, dtype=np.float64) + 180.0) % 360.0 - 180.0


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _linear_fit(
    x: np.ndarray,
    y: np.ndarray,
    valid: np.ndarray,
    *,
    minimum_samples: int,
    minimum_span_deg: float,
) -> tuple[int, float, float, float]:
    use = np.asarray(valid, dtype=bool) & np.isfinite(x) & np.isfinite(y)
    x_use, y_use = (
        np.asarray(x, dtype=np.float64)[use],
        np.asarray(y, dtype=np.float64)[use],
    )
    count = int(x_use.size)
    if count < minimum_samples or float(np.ptp(x_use)) < minimum_span_deg:
        return count, math.nan, math.nan, math.nan
    centered_x = x_use - float(np.mean(x_use))
    denominator = float(np.dot(centered_x, centered_x))
    if denominator <= 0:
        return count, math.nan, math.nan, math.nan
    gain = float(np.dot(centered_x, y_use - float(np.mean(y_use))) / denominator)
    intercept = float(np.mean(y_use) - gain * np.mean(x_use))
    correlation = float(np.corrcoef(x_use, y_use)[0, 1])
    return count, gain, intercept, correlation


def _dynamic_fit(
    bearing_deg: np.ndarray,
    gaze_deg: np.ndarray,
    valid: np.ndarray,
    acquisition_frame_id: np.ndarray,
    timestamp_ns: np.ndarray,
    timestamp_valid: np.ndarray,
    *,
    maximum_lag_s: float,
    minimum_samples: int,
) -> tuple[int, float, float, int, float]:
    """Fit wrapped gaze changes to same/future contiguous bearing changes."""

    bearing = np.asarray(bearing_deg, dtype=np.float64).reshape(-1)
    gaze = np.asarray(gaze_deg, dtype=np.float64).reshape(-1)
    usable = np.asarray(valid, dtype=bool).reshape(-1)
    frames = np.asarray(acquisition_frame_id, dtype=np.int64).reshape(-1)
    timestamps = np.asarray(timestamp_ns, dtype=np.int64).reshape(-1)
    time_valid = np.asarray(timestamp_valid, dtype=bool).reshape(-1)
    if not (
        bearing.size
        == gaze.size
        == usable.size
        == frames.size
        == timestamps.size
        == time_valid.size
    ):
        _fail("Dynamic gaze inputs have inconsistent frame lengths.")
    if bearing.size < 2:
        return 0, math.nan, math.nan, 0, math.nan
    transition = (
        usable[:-1]
        & usable[1:]
        & time_valid[:-1]
        & time_valid[1:]
        & (frames[1:] == frames[:-1] + 1)
        & (timestamps[1:] > timestamps[:-1])
    )
    interval_s = (timestamps[1:] - timestamps[:-1]).astype(np.float64) / 1e9
    finite_interval = interval_s[transition & np.isfinite(interval_s)]
    if not finite_interval.size:
        return 0, math.nan, math.nan, 0, math.nan
    typical_interval_s = float(np.median(finite_interval))
    if not math.isfinite(typical_interval_s) or typical_interval_s <= 0:
        return 0, math.nan, math.nan, 0, math.nan
    maximum_lag_frames = max(0, int(math.floor(maximum_lag_s / typical_interval_s)))
    delta_bearing = _wrap_deg(np.diff(bearing))
    delta_gaze = _wrap_deg(np.diff(gaze))
    invalid_prefix = np.concatenate(
        (np.asarray([0], dtype=np.int64), np.cumsum(~transition, dtype=np.int64))
    )
    best: tuple[int, float, float, int, float] | None = None
    for lag in range(maximum_lag_frames + 1):
        length = transition.size - lag
        if length <= 0:
            break
        candidate = transition[:length] & transition[lag : lag + length]
        if lag:
            contiguous = (
                invalid_prefix[lag + 1 : lag + 1 + length] - invalid_prefix[:length]
            ) == 0
            candidate &= contiguous
        x = delta_bearing[:length][candidate]
        y = delta_gaze[lag : lag + length][candidate]
        count = int(x.size)
        if count < minimum_samples:
            continue
        centered_x = x - float(np.mean(x))
        centered_y = y - float(np.mean(y))
        denominator = float(np.dot(centered_x, centered_x))
        x_norm = math.sqrt(denominator) if denominator > 0 else 0.0
        y_norm = float(np.linalg.norm(centered_y))
        if denominator <= 0 or x_norm <= 0 or y_norm <= 0:
            continue
        gain = float(np.dot(centered_x, centered_y) / denominator)
        correlation = float(np.dot(centered_x, centered_y) / (x_norm * y_norm))
        if lag:
            start_indices = np.flatnonzero(candidate) + 1
            lag_seconds = float(
                np.median(
                    (
                        timestamps[start_indices + lag] - timestamps[start_indices]
                    ).astype(np.float64)
                    / 1e9
                )
            )
        else:
            lag_seconds = 0.0
        if lag_seconds > maximum_lag_s + 1e-12:
            continue
        result = (count, gain, correlation, lag, lag_seconds)
        if best is None or correlation > best[2]:
            best = result
    return best or (0, math.nan, math.nan, 0, math.nan)


def _rotated_virtual_candidates(
    *,
    chaser_xy_px: np.ndarray,
    chaser_valid: np.ndarray,
    chaser_occurrence: np.ndarray,
    center_xy_px: np.ndarray,
    rotations_deg: tuple[float, ...],
    minimum_separation_px: float,
    maximum_collision_fraction: float,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Build accepted rotated references and retain candidate exclusion evidence."""

    real = np.asarray(chaser_xy_px, dtype=np.float64)
    valid = np.asarray(chaser_valid, dtype=bool) & np.asarray(
        chaser_occurrence, dtype=bool
    )
    center = np.asarray(center_xy_px, dtype=np.float64)
    records: list[dict[str, Any]] = []
    positions: list[np.ndarray] = []
    next_reference = 0
    for parent in range(real.shape[1]):
        relative = real[:, parent, :] - center
        for rotation_deg in rotations_deg:
            theta = math.radians(rotation_deg)
            cosine, sine = math.cos(theta), math.sin(theta)
            rotated = (
                np.column_stack(
                    (
                        relative[:, 0] * cosine - relative[:, 1] * sine,
                        relative[:, 0] * sine + relative[:, 1] * cosine,
                    )
                )
                + center
            )
            collision_fractions: list[float] = []
            for other in range(real.shape[1]):
                rows = (
                    valid[:, parent]
                    & valid[:, other]
                    & np.isfinite(rotated).all(axis=1)
                    & np.isfinite(real[:, other, :]).all(axis=1)
                )
                gaps = np.linalg.norm(rotated[rows] - real[rows, other, :], axis=1)
                collision_fractions.append(
                    float(np.mean(gaps < minimum_separation_px)) if gaps.size else 0.0
                )
            maximum_fraction = max(collision_fractions, default=0.0)
            accepted = maximum_fraction <= maximum_collision_fraction
            reference_row_id = next_reference if accepted else -1
            records.append(
                {
                    "parent_chaser_position": parent,
                    "rotation_deg": rotation_deg,
                    "maximum_collision_fraction": maximum_fraction,
                    "accepted": accepted,
                    "reference_row_id": reference_row_id,
                }
            )
            if accepted:
                positions.append(rotated)
                next_reference += 1
    stacked = (
        np.stack(positions, axis=1)
        if positions
        else np.empty((real.shape[0], 0, 2), dtype=np.float64)
    )
    return records, stacked


@dataclass(frozen=True, slots=True)
class GazeTrackingInput:
    recording_id: str
    source_relative_frame_run_path: str
    source_relative_frame_manifest_sha256: str
    source_eye_run_path: str
    source_eye_manifest_sha256: str
    source_eye_convention_receipt_sha256: str
    source_eye_channel_policy: str
    source_semantic_selection_manifest_sha256: str
    source_radial_run_path: str
    source_radial_manifest_sha256: str
    source_radial_payload_sha256: str
    source_arena_geometry_and_scale: Mapping[str, Any]
    arena_center_xy_px: np.ndarray
    arena_radius_px: float
    arena_radius_mm: float
    pixels_per_mm: float
    n_frames: int
    n_chasers: int
    acquisition_frame_id_by_frame: np.ndarray
    timestamp_ns_by_frame: np.ndarray
    timestamp_valid_by_frame: np.ndarray
    semantic_role_code_by_frame: np.ndarray
    chaser_identity_code: np.ndarray
    fish_position_xy_px: np.ndarray
    fish_position_valid: np.ndarray
    chaser_position_xy_px: np.ndarray
    chaser_position_valid: np.ndarray
    chaser_occurrence_member: np.ndarray
    body_origin_xy_px: np.ndarray
    body_forward_axis_xy: np.ndarray
    body_left_axis_xy: np.ndarray
    body_axes_valid: np.ndarray
    distance_mm: np.ndarray
    distance_valid: np.ndarray
    chaser_bearing_deg: np.ndarray
    chaser_bearing_valid: np.ndarray
    gaze_signed_deg: np.ndarray
    gaze_valid: np.ndarray
    vergence_deg: np.ndarray
    vergence_valid: np.ndarray
    lock_threshold_deg: float = 10.0
    minimum_lock_duration_s: float = 0.1
    maximum_tracking_distance_mm: float = 50.0
    accessible_quantiles: tuple[float, float] = (0.025, 0.975)
    virtual_rotations_deg: tuple[float, ...] = DEFAULT_VIRTUAL_ROTATIONS_DEG
    minimum_virtual_separation_mm: float = 8.0
    maximum_virtual_collision_fraction: float = 0.05
    maximum_dynamic_lag_s: float = 0.5
    minimum_regression_samples: int = 30
    minimum_regression_span_deg: float = 5.0
    core_authority_dependency: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class PreparedGazeTracking:
    recording_id: str
    n_gaze_rows: int
    n_summary_rows: int
    n_lock_events: int
    arrays: Mapping[str, np.ndarray]
    manifest: Mapping[str, Any]

    def array(self, name: str) -> np.ndarray:
        try:
            return self.arrays[name]
        except KeyError as exc:
            raise KeyError(f"Unknown gaze-tracking array {name!r}.") from exc

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


def prepare_gaze_tracking_successor(source: GazeTrackingInput) -> PreparedGazeTracking:
    """Prepare frame/eye/chaser rows, summaries, and contiguous lock events."""

    if type(source) is not GazeTrackingInput:
        raise TypeError("source must be one GazeTrackingInput.")
    recording_id = _text(source.recording_id, name="recording_id")
    for name in (
        "source_relative_frame_run_path",
        "source_eye_run_path",
        "source_eye_channel_policy",
        "source_radial_run_path",
    ):
        _text(getattr(source, name), name=name)
    for name in (
        "source_relative_frame_manifest_sha256",
        "source_eye_manifest_sha256",
        "source_eye_convention_receipt_sha256",
        "source_semantic_selection_manifest_sha256",
        "source_radial_manifest_sha256",
        "source_radial_payload_sha256",
    ):
        _digest(getattr(source, name), name=name)
    try:
        core_authority = validate_core_paradigm_source_dependency(
            source.core_authority_dependency,
            recording_id=recording_id,
            source_relative_frame_run_path=source.source_relative_frame_run_path,
            source_relative_frame_manifest_sha256=(
                source.source_relative_frame_manifest_sha256
            ),
        )
    except (TypeError, ValueError) as exc:
        _fail(f"Core-authority dependency is invalid: {exc}")
    if not isinstance(source.source_arena_geometry_and_scale, Mapping):
        _fail("source_arena_geometry_and_scale must be one exact source record.")
    try:
        canonical_json_sha256(dict(source.source_arena_geometry_and_scale))
    except (TypeError, ValueError) as exc:
        raise GazeTrackingSuccessorError(
            f"source_arena_geometry_and_scale is not strict JSON: {exc}"
        ) from exc
    if type(source.n_frames) is not int or source.n_frames < 0:
        _fail("n_frames must be one non-negative exact integer.")
    if type(source.n_chasers) is not int or source.n_chasers <= 0:
        _fail("n_chasers must be one positive exact integer.")
    n_frames, n_chasers = source.n_frames, source.n_chasers
    n_rel = n_frames * n_chasers
    frame_ids = _vector(
        source.acquisition_frame_id_by_frame,
        name="acquisition_frame_id_by_frame",
        dtype=np.int64,
        size=n_frames,
    )
    if np.unique(frame_ids).size != n_frames:
        _fail("Acquisition frame identities are duplicated.")
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
    role = _vector(
        source.semantic_role_code_by_frame,
        name="semantic_role_code_by_frame",
        dtype=np.uint8,
        size=n_frames,
    )
    if np.any(~np.isin(role, np.asarray([0, *ROLE_CODES.values()], dtype=np.uint8))):
        _fail("semantic_role_code_by_frame contains an unknown code.")
    codes = _vector(
        source.chaser_identity_code,
        name="chaser_identity_code",
        dtype=np.uint16,
        size=n_rel,
    ).reshape(n_frames, n_chasers)
    if n_frames and np.any(codes != codes[:1, :]):
        _fail("Chaser identity changed along the fixed chaser axis.")
    chaser_codes = (
        codes[0] if n_frames else np.arange(1, n_chasers + 1, dtype=np.uint16)
    )
    if np.unique(chaser_codes).size != n_chasers:
        _fail("Chaser identity codes are duplicated.")
    fish_xy = _float_array(
        source.fish_position_xy_px,
        name="fish_position_xy_px",
        shape=(n_frames, 2),
    )
    fish_valid = _vector(
        source.fish_position_valid,
        name="fish_position_valid",
        dtype=bool,
        size=n_frames,
    )
    chaser_xy = _float_array(
        source.chaser_position_xy_px,
        name="chaser_position_xy_px",
        shape=(n_rel, 2),
    ).reshape(n_frames, n_chasers, 2)
    chaser_position_valid = _vector(
        source.chaser_position_valid,
        name="chaser_position_valid",
        dtype=bool,
        size=n_rel,
    ).reshape(n_frames, n_chasers)
    occurrence = _vector(
        source.chaser_occurrence_member,
        name="chaser_occurrence_member",
        dtype=bool,
        size=n_rel,
    ).reshape(n_frames, n_chasers)
    body_origin = _float_array(
        source.body_origin_xy_px,
        name="body_origin_xy_px",
        shape=(n_frames, 2),
    )
    body_forward = _float_array(
        source.body_forward_axis_xy,
        name="body_forward_axis_xy",
        shape=(n_frames, 2),
    )
    body_left = _float_array(
        source.body_left_axis_xy,
        name="body_left_axis_xy",
        shape=(n_frames, 2),
    )
    body_axes_valid = _vector(
        source.body_axes_valid,
        name="body_axes_valid",
        dtype=bool,
        size=n_frames,
    )
    for values, valid, name in (
        (fish_xy, fish_valid, "fish position"),
        (chaser_xy, chaser_position_valid, "chaser position"),
        (body_origin, body_axes_valid, "body origin"),
        (body_forward, body_axes_valid, "body forward axis"),
        (body_left, body_axes_valid, "body left axis"),
    ):
        if np.any(valid & ~np.isfinite(values).all(axis=-1)):
            _fail(f"A valid {name} row is non-finite.")
    arena_center = np.asarray(source.arena_center_xy_px, dtype=np.float64)
    if arena_center.shape != (2,) or np.any(~np.isfinite(arena_center)):
        _fail("arena_center_xy_px must be one finite xy pair.")
    geometry_numbers: dict[str, float] = {}
    for name in (
        "arena_radius_px",
        "arena_radius_mm",
        "pixels_per_mm",
    ):
        value = getattr(source, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            _fail(f"{name} must be one positive finite number.")
        result = float(value)
        if not math.isfinite(result) or result <= 0:
            _fail(f"{name} must be one positive finite number.")
        geometry_numbers[name] = result
    if not math.isclose(
        geometry_numbers["arena_radius_px"] / geometry_numbers["pixels_per_mm"],
        geometry_numbers["arena_radius_mm"],
        rel_tol=1e-6,
        abs_tol=1e-9,
    ):
        _fail("Reviewed arena radius and relative-frame scale disagree.")
    distance = _float_array(
        source.distance_mm,
        name="distance_mm",
        shape=(n_rel,),
    ).reshape(n_frames, n_chasers)
    distance_valid = _vector(
        source.distance_valid,
        name="distance_valid",
        dtype=bool,
        size=n_rel,
    ).reshape(n_frames, n_chasers)
    bearing = _float_array(
        source.chaser_bearing_deg,
        name="chaser_bearing_deg",
        shape=(n_rel,),
    ).reshape(n_frames, n_chasers)
    bearing_valid = _vector(
        source.chaser_bearing_valid,
        name="chaser_bearing_valid",
        dtype=bool,
        size=n_rel,
    ).reshape(n_frames, n_chasers)
    gaze = _float_array(
        source.gaze_signed_deg,
        name="gaze_signed_deg",
        shape=(n_frames, 2),
    )
    gaze_valid = np.asarray(source.gaze_valid)
    if gaze_valid.dtype != np.dtype(bool) or gaze_valid.shape != (n_frames, 2):
        _fail("gaze_valid must be exact bool with shape (n_frames, 2).")
    vergence = _float_array(
        source.vergence_deg,
        name="vergence_deg",
        shape=(n_frames,),
    )
    vergence_valid = _vector(
        source.vergence_valid,
        name="vergence_valid",
        dtype=bool,
        size=n_frames,
    )
    for values, valid, name in (
        (distance, distance_valid, "distance"),
        (bearing, bearing_valid, "bearing"),
        (gaze, gaze_valid, "gaze"),
        (vergence, vergence_valid, "vergence"),
    ):
        if np.any(valid & ~np.isfinite(values)):
            _fail(f"A valid {name} value is non-finite.")

    parameters: dict[str, float] = {}
    for name in (
        "lock_threshold_deg",
        "minimum_lock_duration_s",
        "maximum_tracking_distance_mm",
        "minimum_virtual_separation_mm",
        "maximum_dynamic_lag_s",
        "minimum_regression_span_deg",
    ):
        value = getattr(source, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            _fail(f"{name} must be one positive finite number.")
        value = float(value)
        if not math.isfinite(value) or value <= 0:
            _fail(f"{name} must be one positive finite number.")
        parameters[name] = value
    collision_fraction = source.maximum_virtual_collision_fraction
    if (
        isinstance(collision_fraction, bool)
        or not isinstance(collision_fraction, (int, float))
        or not math.isfinite(float(collision_fraction))
        or not 0 <= float(collision_fraction) <= 1
    ):
        _fail("maximum_virtual_collision_fraction must be finite in [0, 1].")
    parameters["maximum_virtual_collision_fraction"] = float(collision_fraction)
    if (
        type(source.minimum_regression_samples) is not int
        or source.minimum_regression_samples < 3
    ):
        _fail("minimum_regression_samples must be one exact integer at least 3.")
    rotations = np.asarray(source.virtual_rotations_deg, dtype=np.float64)
    if (
        rotations.ndim != 1
        or rotations.size == 0
        or rotations.size > 35
        or np.any(~np.isfinite(rotations))
        or np.any((rotations <= 0) | (rotations >= 360))
        or np.unique(rotations).size != rotations.size
    ):
        _fail("virtual_rotations_deg must be unique finite angles in (0, 360).")
    quantiles = np.asarray(source.accessible_quantiles, dtype=np.float64)
    if (
        quantiles.shape != (2,)
        or np.any(~np.isfinite(quantiles))
        or not (0 <= quantiles[0] < quantiles[1] <= 1)
    ):
        _fail("accessible_quantiles must be an ordered pair in [0, 1].")
    eye_range = np.full((2, 2), np.nan, dtype=np.float64)
    for eye in range(2):
        values = gaze[:, eye][gaze_valid[:, eye]]
        if values.size:
            eye_range[eye] = np.quantile(values, quantiles)

    n_rows = n_frames * 2 * n_chasers
    frame_index = np.repeat(np.arange(n_frames, dtype=np.int64), 2 * n_chasers)
    eye_pos = np.tile(np.repeat(np.arange(2, dtype=np.int64), n_chasers), n_frames)
    chaser_pos = np.tile(np.arange(n_chasers, dtype=np.int64), n_frames * 2)
    row_gaze = gaze[frame_index, eye_pos]
    row_bearing = bearing[frame_index, chaser_pos]
    row_distance = distance[frame_index, chaser_pos]
    valid = (
        (role[frame_index] != 0)
        & gaze_valid[frame_index, eye_pos]
        & bearing_valid[frame_index, chaser_pos]
        & distance_valid[frame_index, chaser_pos]
        & (row_distance <= parameters["maximum_tracking_distance_mm"])
    )
    accessible = np.zeros(n_rows, dtype=bool)
    for eye in range(2):
        low, high = eye_range[eye]
        if np.isfinite(low) and np.isfinite(high):
            eye_rows = eye_pos == eye
            accessible[eye_rows] = (row_bearing[eye_rows] >= low) & (
                row_bearing[eye_rows] <= high
            )
    error = _wrap_deg(row_gaze - row_bearing)
    lock = valid & accessible & (np.abs(error) <= parameters["lock_threshold_deg"])

    arrays: dict[str, np.ndarray] = {
        "gaze_row_id": np.arange(n_rows, dtype=np.int64),
        "acquisition_frame_id": frame_ids[frame_index],
        "semantic_role_code": role[frame_index],
        "eye_code": (eye_pos + 1).astype(np.uint8),
        "chaser_identity_code": chaser_codes[chaser_pos],
        "distance_mm": row_distance.astype(np.float32),
        "bearing_deg": row_bearing.astype(np.float32),
        "gaze_signed_deg": row_gaze.astype(np.float32),
        "vergence_deg": vergence[frame_index].astype(np.float32),
        "valid": valid,
        "accessible": accessible,
        "gaze_error_deg": error.astype(np.float32),
        "lock_on": lock,
    }

    role_values = np.asarray(sorted(ROLE_CODES.values()), dtype=np.uint8)
    summary_count = int(role_values.size * 2 * n_chasers)
    summary: dict[str, np.ndarray] = {
        "summary_row_id": np.arange(summary_count, dtype=np.int64),
        "summary_role_code": np.zeros(summary_count, dtype=np.uint8),
        "summary_eye_code": np.zeros(summary_count, dtype=np.uint8),
        "summary_chaser_identity_code": np.zeros(summary_count, dtype=np.uint16),
        "summary_valid_sample_count": np.zeros(summary_count, dtype=np.int64),
        "summary_accessible_sample_count": np.zeros(summary_count, dtype=np.int64),
        "summary_lock_sample_count": np.zeros(summary_count, dtype=np.int64),
        "summary_lock_fraction": np.full(summary_count, np.nan, dtype=np.float64),
        "summary_median_abs_error_deg": np.full(
            summary_count, np.nan, dtype=np.float64
        ),
        "summary_tracking_gain": np.full(summary_count, np.nan, dtype=np.float64),
        "summary_tracking_intercept_deg": np.full(
            summary_count, np.nan, dtype=np.float64
        ),
        "summary_tracking_correlation": np.full(
            summary_count, np.nan, dtype=np.float64
        ),
        "summary_regression_sample_count": np.zeros(summary_count, dtype=np.int64),
        "summary_dynamic_zero_lag_gain": np.full(
            summary_count, np.nan, dtype=np.float64
        ),
        "summary_dynamic_zero_lag_correlation": np.full(
            summary_count, np.nan, dtype=np.float64
        ),
        "summary_dynamic_zero_lag_sample_count": np.zeros(
            summary_count, dtype=np.int64
        ),
        "summary_dynamic_best_lag_gain": np.full(
            summary_count, np.nan, dtype=np.float64
        ),
        "summary_dynamic_best_lag_correlation": np.full(
            summary_count, np.nan, dtype=np.float64
        ),
        "summary_dynamic_best_lag_frames": np.zeros(summary_count, dtype=np.int32),
        "summary_dynamic_best_lag_seconds": np.full(
            summary_count, np.nan, dtype=np.float64
        ),
        "summary_dynamic_best_lag_sample_count": np.zeros(
            summary_count, dtype=np.int64
        ),
    }
    out = 0
    for role_code in role_values:
        for eye_code in (EYE_LEFT, EYE_RIGHT):
            for code in chaser_codes:
                mask = (
                    (arrays["semantic_role_code"] == role_code)
                    & (arrays["eye_code"] == eye_code)
                    & (arrays["chaser_identity_code"] == code)
                )
                use = mask & valid
                access = use & accessible
                summary["summary_role_code"][out] = role_code
                summary["summary_eye_code"][out] = eye_code
                summary["summary_chaser_identity_code"][out] = code
                summary["summary_valid_sample_count"][out] = int(np.count_nonzero(use))
                access_count = int(np.count_nonzero(access))
                lock_count = int(np.count_nonzero(mask & lock))
                summary["summary_accessible_sample_count"][out] = access_count
                summary["summary_lock_sample_count"][out] = lock_count
                if access_count:
                    summary["summary_lock_fraction"][out] = lock_count / access_count
                    summary["summary_median_abs_error_deg"][out] = float(
                        np.median(np.abs(error[access]))
                    )
                count, gain, intercept, correlation = _linear_fit(
                    row_bearing,
                    row_gaze,
                    access,
                    minimum_samples=source.minimum_regression_samples,
                    minimum_span_deg=parameters["minimum_regression_span_deg"],
                )
                summary["summary_regression_sample_count"][out] = count
                summary["summary_tracking_gain"][out] = gain
                summary["summary_tracking_intercept_deg"][out] = intercept
                summary["summary_tracking_correlation"][out] = correlation
                eye_position = eye_code - 1
                chaser_position = int(np.flatnonzero(chaser_codes == code)[0])
                frame_access = (
                    (role == role_code)
                    & gaze_valid[:, eye_position]
                    & bearing_valid[:, chaser_position]
                    & distance_valid[:, chaser_position]
                    & (
                        distance[:, chaser_position]
                        <= parameters["maximum_tracking_distance_mm"]
                    )
                    & (bearing[:, chaser_position] >= eye_range[eye_position, 0])
                    & (bearing[:, chaser_position] <= eye_range[eye_position, 1])
                )
                zero = _dynamic_fit(
                    bearing[:, chaser_position],
                    gaze[:, eye_position],
                    frame_access,
                    frame_ids,
                    timestamp,
                    timestamp_valid,
                    maximum_lag_s=0.0,
                    minimum_samples=source.minimum_regression_samples,
                )
                best = _dynamic_fit(
                    bearing[:, chaser_position],
                    gaze[:, eye_position],
                    frame_access,
                    frame_ids,
                    timestamp,
                    timestamp_valid,
                    maximum_lag_s=parameters["maximum_dynamic_lag_s"],
                    minimum_samples=source.minimum_regression_samples,
                )
                summary["summary_dynamic_zero_lag_sample_count"][out] = zero[0]
                summary["summary_dynamic_zero_lag_gain"][out] = zero[1]
                summary["summary_dynamic_zero_lag_correlation"][out] = zero[2]
                summary["summary_dynamic_best_lag_sample_count"][out] = best[0]
                summary["summary_dynamic_best_lag_gain"][out] = best[1]
                summary["summary_dynamic_best_lag_correlation"][out] = best[2]
                summary["summary_dynamic_best_lag_frames"][out] = best[3]
                summary["summary_dynamic_best_lag_seconds"][out] = best[4]
                out += 1
    arrays.update(summary)

    candidate_records, virtual_xy = _rotated_virtual_candidates(
        chaser_xy_px=chaser_xy,
        chaser_valid=chaser_position_valid,
        chaser_occurrence=occurrence,
        center_xy_px=arena_center,
        rotations_deg=tuple(float(value) for value in rotations),
        minimum_separation_px=(
            parameters["minimum_virtual_separation_mm"]
            * geometry_numbers["pixels_per_mm"]
        ),
        maximum_collision_fraction=parameters["maximum_virtual_collision_fraction"],
    )
    accepted_records = [record for record in candidate_records if record["accepted"]]
    n_virtual = len(accepted_records)
    arrays.update(
        {
            "virtual_candidate_row_id": np.arange(
                len(candidate_records), dtype=np.int64
            ),
            "virtual_candidate_parent_chaser_identity_code": np.asarray(
                [
                    chaser_codes[record["parent_chaser_position"]]
                    for record in candidate_records
                ],
                dtype=np.uint16,
            ),
            "virtual_candidate_rotation_deg": np.asarray(
                [record["rotation_deg"] for record in candidate_records],
                dtype=np.float32,
            ),
            "virtual_candidate_max_collision_fraction": np.asarray(
                [record["maximum_collision_fraction"] for record in candidate_records],
                dtype=np.float64,
            ),
            "virtual_candidate_accepted": np.asarray(
                [record["accepted"] for record in candidate_records], dtype=bool
            ),
            "virtual_candidate_reference_row_id": np.asarray(
                [record["reference_row_id"] for record in candidate_records],
                dtype=np.int64,
            ),
            "virtual_reference_row_id": np.arange(n_virtual, dtype=np.int64),
            "virtual_reference_parent_chaser_identity_code": np.asarray(
                [
                    chaser_codes[record["parent_chaser_position"]]
                    for record in accepted_records
                ],
                dtype=np.uint16,
            ),
            "virtual_reference_rotation_deg": np.asarray(
                [record["rotation_deg"] for record in accepted_records],
                dtype=np.float32,
            ),
        }
    )
    virtual_distance = np.full((n_frames, n_virtual), np.nan, dtype=np.float64)
    virtual_bearing = np.full((n_frames, n_virtual), np.nan, dtype=np.float64)
    virtual_base_valid = np.zeros((n_frames, n_virtual), dtype=bool)
    for reference_index, record in enumerate(accepted_records):
        parent = int(record["parent_chaser_position"])
        position_valid = (
            chaser_position_valid[:, parent]
            & occurrence[:, parent]
            & fish_valid
            & body_axes_valid
            & np.isfinite(virtual_xy[:, reference_index, :]).all(axis=1)
        )
        distance_values = (
            np.linalg.norm(virtual_xy[:, reference_index, :] - fish_xy, axis=1)
            / geometry_numbers["pixels_per_mm"]
        )
        relative_values = virtual_xy[:, reference_index, :] - body_origin
        forward_values = np.sum(relative_values * body_forward, axis=1)
        left_values = np.sum(relative_values * body_left, axis=1)
        relative_norm = np.linalg.norm(relative_values, axis=1)
        position_valid &= (
            np.isfinite(distance_values)
            & np.isfinite(forward_values)
            & np.isfinite(left_values)
            & (relative_norm > 0)
        )
        virtual_distance[position_valid, reference_index] = distance_values[
            position_valid
        ]
        virtual_bearing[position_valid, reference_index] = np.rad2deg(
            np.arctan2(left_values[position_valid], forward_values[position_valid])
        )
        virtual_base_valid[:, reference_index] = position_valid

    virtual_summary_count = int(role_values.size * 2 * n_virtual)
    virtual_summary: dict[str, np.ndarray] = {
        "virtual_summary_row_id": np.arange(virtual_summary_count, dtype=np.int64),
        "virtual_summary_role_code": np.zeros(virtual_summary_count, dtype=np.uint8),
        "virtual_summary_eye_code": np.zeros(virtual_summary_count, dtype=np.uint8),
        "virtual_summary_reference_row_id": np.zeros(
            virtual_summary_count, dtype=np.int64
        ),
        "virtual_summary_valid_sample_count": np.zeros(
            virtual_summary_count, dtype=np.int64
        ),
        "virtual_summary_accessible_sample_count": np.zeros(
            virtual_summary_count, dtype=np.int64
        ),
        "virtual_summary_lock_sample_count": np.zeros(
            virtual_summary_count, dtype=np.int64
        ),
        "virtual_summary_lock_fraction": np.full(
            virtual_summary_count, np.nan, dtype=np.float64
        ),
        "virtual_summary_median_abs_error_deg": np.full(
            virtual_summary_count, np.nan, dtype=np.float64
        ),
        "virtual_summary_tracking_gain": np.full(
            virtual_summary_count, np.nan, dtype=np.float64
        ),
        "virtual_summary_tracking_intercept_deg": np.full(
            virtual_summary_count, np.nan, dtype=np.float64
        ),
        "virtual_summary_tracking_correlation": np.full(
            virtual_summary_count, np.nan, dtype=np.float64
        ),
        "virtual_summary_regression_sample_count": np.zeros(
            virtual_summary_count, dtype=np.int64
        ),
        "virtual_summary_dynamic_zero_lag_gain": np.full(
            virtual_summary_count, np.nan, dtype=np.float64
        ),
        "virtual_summary_dynamic_zero_lag_correlation": np.full(
            virtual_summary_count, np.nan, dtype=np.float64
        ),
        "virtual_summary_dynamic_zero_lag_sample_count": np.zeros(
            virtual_summary_count, dtype=np.int64
        ),
        "virtual_summary_dynamic_best_lag_gain": np.full(
            virtual_summary_count, np.nan, dtype=np.float64
        ),
        "virtual_summary_dynamic_best_lag_correlation": np.full(
            virtual_summary_count, np.nan, dtype=np.float64
        ),
        "virtual_summary_dynamic_best_lag_frames": np.zeros(
            virtual_summary_count, dtype=np.int32
        ),
        "virtual_summary_dynamic_best_lag_seconds": np.full(
            virtual_summary_count, np.nan, dtype=np.float64
        ),
        "virtual_summary_dynamic_best_lag_sample_count": np.zeros(
            virtual_summary_count, dtype=np.int64
        ),
    }
    virtual_out = 0
    for role_code in role_values:
        for eye_code in (EYE_LEFT, EYE_RIGHT):
            eye_position = eye_code - 1
            for reference_index in range(n_virtual):
                base = (
                    (role == role_code)
                    & gaze_valid[:, eye_position]
                    & virtual_base_valid[:, reference_index]
                    & (
                        virtual_distance[:, reference_index]
                        <= parameters["maximum_tracking_distance_mm"]
                    )
                )
                accessible_virtual = (
                    base
                    & (
                        virtual_bearing[:, reference_index]
                        >= eye_range[eye_position, 0]
                    )
                    & (
                        virtual_bearing[:, reference_index]
                        <= eye_range[eye_position, 1]
                    )
                )
                virtual_error = _wrap_deg(
                    gaze[:, eye_position] - virtual_bearing[:, reference_index]
                )
                locked_virtual = accessible_virtual & (
                    np.abs(virtual_error) <= parameters["lock_threshold_deg"]
                )
                virtual_summary["virtual_summary_role_code"][virtual_out] = role_code
                virtual_summary["virtual_summary_eye_code"][virtual_out] = eye_code
                virtual_summary["virtual_summary_reference_row_id"][
                    virtual_out
                ] = reference_index
                valid_count = int(np.count_nonzero(base))
                accessible_count = int(np.count_nonzero(accessible_virtual))
                lock_count = int(np.count_nonzero(locked_virtual))
                virtual_summary["virtual_summary_valid_sample_count"][
                    virtual_out
                ] = valid_count
                virtual_summary["virtual_summary_accessible_sample_count"][
                    virtual_out
                ] = accessible_count
                virtual_summary["virtual_summary_lock_sample_count"][
                    virtual_out
                ] = lock_count
                if accessible_count:
                    virtual_summary["virtual_summary_lock_fraction"][virtual_out] = (
                        lock_count / accessible_count
                    )
                    virtual_summary["virtual_summary_median_abs_error_deg"][
                        virtual_out
                    ] = float(np.median(np.abs(virtual_error[accessible_virtual])))
                fit = _linear_fit(
                    virtual_bearing[:, reference_index],
                    gaze[:, eye_position],
                    accessible_virtual,
                    minimum_samples=source.minimum_regression_samples,
                    minimum_span_deg=parameters["minimum_regression_span_deg"],
                )
                virtual_summary["virtual_summary_regression_sample_count"][
                    virtual_out
                ] = fit[0]
                virtual_summary["virtual_summary_tracking_gain"][virtual_out] = fit[1]
                virtual_summary["virtual_summary_tracking_intercept_deg"][
                    virtual_out
                ] = fit[2]
                virtual_summary["virtual_summary_tracking_correlation"][virtual_out] = (
                    fit[3]
                )
                zero = _dynamic_fit(
                    virtual_bearing[:, reference_index],
                    gaze[:, eye_position],
                    accessible_virtual,
                    frame_ids,
                    timestamp,
                    timestamp_valid,
                    maximum_lag_s=0.0,
                    minimum_samples=source.minimum_regression_samples,
                )
                best = _dynamic_fit(
                    virtual_bearing[:, reference_index],
                    gaze[:, eye_position],
                    accessible_virtual,
                    frame_ids,
                    timestamp,
                    timestamp_valid,
                    maximum_lag_s=parameters["maximum_dynamic_lag_s"],
                    minimum_samples=source.minimum_regression_samples,
                )
                virtual_summary["virtual_summary_dynamic_zero_lag_sample_count"][
                    virtual_out
                ] = zero[0]
                virtual_summary["virtual_summary_dynamic_zero_lag_gain"][
                    virtual_out
                ] = zero[1]
                virtual_summary["virtual_summary_dynamic_zero_lag_correlation"][
                    virtual_out
                ] = zero[2]
                virtual_summary["virtual_summary_dynamic_best_lag_sample_count"][
                    virtual_out
                ] = best[0]
                virtual_summary["virtual_summary_dynamic_best_lag_gain"][
                    virtual_out
                ] = best[1]
                virtual_summary["virtual_summary_dynamic_best_lag_correlation"][
                    virtual_out
                ] = best[2]
                virtual_summary["virtual_summary_dynamic_best_lag_frames"][
                    virtual_out
                ] = best[3]
                virtual_summary["virtual_summary_dynamic_best_lag_seconds"][
                    virtual_out
                ] = best[4]
                virtual_out += 1
    arrays.update(virtual_summary)

    control: dict[str, np.ndarray] = {
        "control_summary_row_id": np.arange(summary_count, dtype=np.int64),
        "control_role_code": summary["summary_role_code"].copy(),
        "control_eye_code": summary["summary_eye_code"].copy(),
        "control_chaser_identity_code": summary["summary_chaser_identity_code"].copy(),
        "control_virtual_reference_count": np.zeros(summary_count, dtype=np.int64),
        "control_tracking_gain_virtual_valid_count": np.zeros(
            summary_count, dtype=np.int64
        ),
        "control_tracking_gain_excess_vs_virtual": np.full(
            summary_count, np.nan, dtype=np.float64
        ),
        "control_dynamic_zero_lag_gain_virtual_valid_count": np.zeros(
            summary_count, dtype=np.int64
        ),
        "control_dynamic_zero_lag_gain_excess_vs_virtual": np.full(
            summary_count, np.nan, dtype=np.float64
        ),
        "control_dynamic_best_lag_gain_virtual_valid_count": np.zeros(
            summary_count, dtype=np.int64
        ),
        "control_dynamic_best_lag_gain_excess_vs_virtual": np.full(
            summary_count, np.nan, dtype=np.float64
        ),
        "control_lock_fraction_virtual_valid_count": np.zeros(
            summary_count, dtype=np.int64
        ),
        "control_lock_fraction_excess_vs_virtual": np.full(
            summary_count, np.nan, dtype=np.float64
        ),
        "control_median_abs_error_virtual_valid_count": np.zeros(
            summary_count, dtype=np.int64
        ),
        "control_median_abs_error_improvement_vs_virtual_deg": np.full(
            summary_count, np.nan, dtype=np.float64
        ),
    }
    metric_pairs = (
        (
            "summary_tracking_gain",
            "virtual_summary_tracking_gain",
            "control_tracking_gain_virtual_valid_count",
            "control_tracking_gain_excess_vs_virtual",
            1.0,
        ),
        (
            "summary_dynamic_zero_lag_gain",
            "virtual_summary_dynamic_zero_lag_gain",
            "control_dynamic_zero_lag_gain_virtual_valid_count",
            "control_dynamic_zero_lag_gain_excess_vs_virtual",
            1.0,
        ),
        (
            "summary_dynamic_best_lag_gain",
            "virtual_summary_dynamic_best_lag_gain",
            "control_dynamic_best_lag_gain_virtual_valid_count",
            "control_dynamic_best_lag_gain_excess_vs_virtual",
            1.0,
        ),
        (
            "summary_lock_fraction",
            "virtual_summary_lock_fraction",
            "control_lock_fraction_virtual_valid_count",
            "control_lock_fraction_excess_vs_virtual",
            1.0,
        ),
        (
            "summary_median_abs_error_deg",
            "virtual_summary_median_abs_error_deg",
            "control_median_abs_error_virtual_valid_count",
            "control_median_abs_error_improvement_vs_virtual_deg",
            -1.0,
        ),
    )
    reference_parent_codes = arrays["virtual_reference_parent_chaser_identity_code"]
    for summary_index in range(summary_count):
        role_code = summary["summary_role_code"][summary_index]
        eye_code = summary["summary_eye_code"][summary_index]
        chaser_code = summary["summary_chaser_identity_code"][summary_index]
        reference_ids = np.flatnonzero(reference_parent_codes == chaser_code)
        control["control_virtual_reference_count"][summary_index] = reference_ids.size
        virtual_rows = (
            (virtual_summary["virtual_summary_role_code"] == role_code)
            & (virtual_summary["virtual_summary_eye_code"] == eye_code)
            & np.isin(
                virtual_summary["virtual_summary_reference_row_id"], reference_ids
            )
        )
        for real_name, virtual_name, count_name, output_name, direction in metric_pairs:
            values = virtual_summary[virtual_name][virtual_rows]
            finite = values[np.isfinite(values)]
            control[count_name][summary_index] = finite.size
            real_value = float(summary[real_name][summary_index])
            if finite.size and math.isfinite(real_value):
                virtual_mean = float(np.mean(finite))
                control[output_name][summary_index] = (
                    real_value - virtual_mean
                    if direction > 0
                    else virtual_mean - real_value
                )
    arrays.update(control)

    event_records: list[tuple[int, int, int, int, int, float, int, float]] = []
    for role_code in role_values:
        for eye_code in (EYE_LEFT, EYE_RIGHT):
            for code in chaser_codes:
                mask = (
                    (arrays["semantic_role_code"] == role_code)
                    & (arrays["eye_code"] == eye_code)
                    & (arrays["chaser_identity_code"] == code)
                    & lock
                )
                frame_mask = np.zeros(n_frames, dtype=bool)
                frame_mask[frame_index[mask]] = True
                frame_contiguous = (
                    (frame_ids[1:] == frame_ids[:-1] + 1)
                    & timestamp_valid[:-1]
                    & timestamp_valid[1:]
                    & (timestamp[1:] > timestamp[:-1])
                )
                starts = np.flatnonzero(
                    frame_mask
                    & np.concatenate(
                        (
                            np.asarray([True]),
                            (~frame_mask[:-1]) | (~frame_contiguous),
                        )
                    )
                )
                ends_inclusive = np.flatnonzero(
                    frame_mask
                    & np.concatenate(
                        (
                            (~frame_mask[1:]) | (~frame_contiguous),
                            np.asarray([True]),
                        )
                    )
                )
                if starts.size != ends_inclusive.size:
                    _fail("Lock-event frame segmentation is inconsistent.")
                for start, end_inclusive in zip(
                    starts.tolist(), ends_inclusive.tolist(), strict=True
                ):
                    end = end_inclusive + 1
                    duration = math.nan
                    if timestamp_valid[start] and timestamp_valid[end_inclusive]:
                        duration = (timestamp[end_inclusive] - timestamp[start]) / 1e9
                    if not math.isfinite(duration):
                        continue
                    # A one-sample event has zero timestamp span. Preserve a
                    # conservative zero duration and apply the requested gate.
                    if duration < parameters["minimum_lock_duration_s"]:
                        continue
                    event_row_mask = mask & (frame_index >= start) & (frame_index < end)
                    event_records.append(
                        (
                            int(role_code),
                            int(eye_code),
                            int(code),
                            int(frame_ids[start]),
                            int(frame_ids[end_inclusive]),
                            float(duration),
                            int(end - start),
                            float(np.median(np.abs(error[event_row_mask]))),
                        )
                    )
    n_events = len(event_records)
    event_arrays = {
        "lock_event_row_id": np.arange(n_events, dtype=np.int64),
        "lock_event_role_code": np.asarray(
            [row[0] for row in event_records], dtype=np.uint8
        ),
        "lock_event_eye_code": np.asarray(
            [row[1] for row in event_records], dtype=np.uint8
        ),
        "lock_event_chaser_identity_code": np.asarray(
            [row[2] for row in event_records], dtype=np.uint16
        ),
        "lock_event_start_acquisition_frame_id": np.asarray(
            [row[3] for row in event_records], dtype=np.int64
        ),
        "lock_event_end_acquisition_frame_id_inclusive": np.asarray(
            [row[4] for row in event_records], dtype=np.int64
        ),
        "lock_event_duration_s": np.asarray(
            [row[5] for row in event_records], dtype=np.float64
        ),
        "lock_event_sample_count": np.asarray(
            [row[6] for row in event_records], dtype=np.int64
        ),
        "lock_event_median_abs_error_deg": np.asarray(
            [row[7] for row in event_records], dtype=np.float32
        ),
    }
    arrays.update(event_arrays)
    readonly = {name: _readonly(values) for name, values in arrays.items()}
    manifest_body = {
        "schema_id": PREPARED_SCHEMA_ID,
        "schema_version": PREPARED_SCHEMA_VERSION,
        "scientific_schema": {
            "schema_id": SCHEMA_ID,
            "schema_version": SCHEMA_VERSION,
            "method_id": METHOD_ID,
            "row_unit": "acquisition_frame_x_eye_x_chaser",
            "summary_unit": "semantic_role_x_eye_x_chaser",
            "event_unit": "contiguous_lock_on_interval",
            "virtual_candidate_unit": "chaser_x_rotation",
            "virtual_summary_unit": "semantic_role_x_eye_x_accepted_virtual_reference",
            "control_summary_unit": "semantic_role_x_eye_x_real_chaser",
        },
        "recording_id": recording_id,
        **(
            {"core_authority": dict(core_authority)}
            if core_authority is not None
            else {}
        ),
        "sources": {
            "relative_frame": {
                "run_path": source.source_relative_frame_run_path,
                "manifest_sha256": source.source_relative_frame_manifest_sha256,
            },
            "eye_orientation": {
                "run_path": source.source_eye_run_path,
                "manifest_sha256": source.source_eye_manifest_sha256,
                "convention_receipt_sha256": source.source_eye_convention_receipt_sha256,
                "channel_policy": source.source_eye_channel_policy,
            },
            "semantic_selection_manifest_sha256": (
                source.source_semantic_selection_manifest_sha256
            ),
            "radial_near_field_geometry_authority": {
                "run_path": source.source_radial_run_path,
                "manifest_sha256": source.source_radial_manifest_sha256,
                "scientific_payload_sha256": source.source_radial_payload_sha256,
                "arena_geometry_and_scale": dict(
                    source.source_arena_geometry_and_scale
                ),
            },
        },
        "parameters": {
            **parameters,
            "accessible_quantiles": quantiles.tolist(),
            "empirical_eye_range_deg": eye_range.tolist(),
            "virtual_rotations_deg": rotations.tolist(),
            "minimum_regression_samples": source.minimum_regression_samples,
        },
        "arena": {
            "center_xy_px": arena_center.tolist(),
            "radius_px": geometry_numbers["arena_radius_px"],
            "radius_mm": geometry_numbers["arena_radius_mm"],
            "pixels_per_mm": geometry_numbers["pixels_per_mm"],
        },
        "dimensions": {
            "n_frames": n_frames,
            "n_chasers": n_chasers,
            "n_gaze_rows": n_rows,
            "n_summary_rows": summary_count,
            "n_lock_events": n_events,
            "n_virtual_candidates": len(candidate_records),
            "n_virtual_references": n_virtual,
            "n_virtual_summary_rows": virtual_summary_count,
            "n_control_summary_rows": summary_count,
        },
        "policy": {
            "gaze_field": "directed_left_right_gaze_signed_deg_in_fish_body_frame",
            "bearing_field": "exact_chaser_body_bearing_deg_anatomical_left_positive",
            "world_frame_gaze": "prohibited",
            "nasal_positive_eye_angle": "prohibited_for_object_bearing_comparison",
            "orientation_fallback": "prohibited",
            "invalid_rows": "retained_and_excluded_from_summaries",
            "cohort_inference_unit": "recording_fish",
            "virtual_control_geometry": (
                "rotate_each_exact_real_chaser_trajectory_about_reviewed_arena_center"
            ),
            "virtual_collision_denominator": (
                "frames_where_parent_virtual_and_compared_real_positions_are_valid_and_present"
            ),
            "virtual_collision_exclusion": (
                "exclude_candidate_when_max_real_chaser_collision_fraction_exceeds_threshold"
            ),
            "virtual_null_denominator": (
                "finite_accepted_virtual_metric_values_with_count_persisted_per_real_summary"
            ),
            "control_direction": (
                "gain_and_lock_real_minus_virtual_mean;error_virtual_mean_minus_real"
            ),
            "dynamic_tracking": (
                "wrapped_contiguous_frame_deltas_zero_and_causal_nonnegative_lags"
            ),
            "dynamic_lag_selection": "maximum_correlation_within_exact_maximum_lag",
        },
        "identity_registries": {
            "eye": {"1": "left", "2": "right"},
            "semantic_role": {str(value): name for name, value in ROLE_CODES.items()},
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
    return PreparedGazeTracking(
        recording_id=recording_id,
        n_gaze_rows=n_rows,
        n_summary_rows=summary_count,
        n_lock_events=n_events,
        arrays=MappingProxyType(readonly),
        manifest=manifest,
    )


def gaze_tracking_input_from_handles(
    relative_frame: Any,
    semantic_selection: Any,
    eye_gaze: Any,
    radial_near_field: Any,
    *,
    lock_threshold_deg: float = 10.0,
    minimum_lock_duration_s: float = 0.1,
    maximum_tracking_distance_mm: float = 50.0,
    accessible_quantiles: tuple[float, float] = (0.025, 0.975),
) -> GazeTrackingInput:
    """Bind exact relative, semantic, and reviewed eye-gaze handles."""

    from fisheye.analysis_workflows.chaser_relative_frame_source_handle import (
        ChaserRelativeFrameSourceHandle,
    )
    from fisheye.analysis_workflows.eye_gaze_source_handle import (
        EyeGazeSourceHandle,
    )
    from fisheye.analysis_workflows.composable_chaser_successor_publication import (
        ComposableChaserSuccessorSourceHandle,
    )
    from fisheye.analysis_workflows.generalized_bout_response_successor import (
        semantic_role_codes_from_handles,
    )
    from fisheye.analysis_workflows.protocol_semantic_chaser_selection_publication import (
        ProtocolSemanticChaserSelectionSourceHandle,
    )

    if type(relative_frame) is not ChaserRelativeFrameSourceHandle:
        raise TypeError("relative_frame must be a strict loader-minted handle.")
    if type(semantic_selection) is not ProtocolSemanticChaserSelectionSourceHandle:
        raise TypeError("semantic_selection must be a strict loader-minted handle.")
    if type(eye_gaze) is not EyeGazeSourceHandle:
        raise TypeError("eye_gaze must be a strict loader-minted reviewed handle.")
    if type(radial_near_field) is not ComposableChaserSuccessorSourceHandle:
        raise TypeError("radial_near_field must be a strict composable handle.")
    core_authority = core_paradigm_dependency_from_relative_frame(relative_frame)
    semantic_selection.assert_current()
    eye_gaze.assert_current()
    radial_near_field.assert_current()
    if not relative_frame.body_available:
        _fail("Gaze tracking requires the exact relative-frame body extension.")
    if not (
        relative_frame.analysis_zarr_path
        == semantic_selection.analysis_zarr
        == eye_gaze.analysis_zarr_path
        == radial_near_field.analysis_zarr
    ):
        _fail("Gaze successor sources do not belong to one exact archive.")
    if not (
        relative_frame.recording_id
        == semantic_selection.recording_id
        == eye_gaze.recording_id
        == radial_near_field.recording_id
    ):
        _fail("Gaze successor sources belong to different recordings.")
    if relative_frame.run_manifest.get("scale_policy", {}).get("unit") != "mm":
        _fail("Chaser-relative physical distance is not explicitly in millimeters.")
    if radial_near_field.successor_kind != "chaser_radial_near_field":
        _fail("Gaze tracking requires one exact radial geometry authority.")
    radial_scientific = radial_near_field.scientific_manifest
    radial_core_authority = radial_scientific.get("core_authority")
    if (
        dict(radial_core_authority)
        if isinstance(radial_core_authority, Mapping)
        else radial_core_authority
    ) != (dict(core_authority) if core_authority is not None else None):
        _fail("Radial and gaze successors bind different core authorities.")
    radial_sources = radial_scientific.get("sources")
    if not isinstance(radial_sources, Mapping):
        _fail("Radial gaze geometry authority lacks source bindings.")
    if radial_sources.get("relative_frame") != {
        "run_path": relative_frame.run_path,
        "manifest_sha256": relative_frame.manifest_sha256,
    }:
        _fail("Radial gaze geometry authority uses another relative frame.")
    if radial_sources.get("protocol_semantic_selection") != {
        "run_path": semantic_selection.run_path,
        "manifest_sha256": semantic_selection.manifest_sha256,
    }:
        _fail("Radial gaze geometry authority uses another semantic selection.")
    radial_provider = radial_scientific.get("position_provider")
    fish_authority = relative_frame.source_authorities.get("fish_position")
    if (
        not isinstance(radial_provider, Mapping)
        or not isinstance(fish_authority, Mapping)
        or radial_provider.get("provider_id") != fish_authority.get("provider_id")
        or radial_provider.get("provider_digest")
        != fish_authority.get("provider_digest")
        or radial_provider.get("status") != "first_class_explicit_authority"
    ):
        _fail("Radial gaze geometry authority uses another position provider.")
    geometry_binding = radial_sources.get("arena_geometry_and_scale")
    arena = radial_scientific.get("arena")
    if not isinstance(geometry_binding, Mapping) or not isinstance(arena, Mapping):
        _fail("Radial gaze geometry authority lacks reviewed arena evidence.")
    center = np.asarray(arena.get("center_xy_px"), dtype=np.float64)
    radius_px = float(arena.get("radius_px", math.nan))
    radius_mm = float(arena.get("radius_mm", math.nan))
    scale = relative_frame.run_manifest.get("scale_policy", {})
    pixels_per_mm = float(scale.get("pixels_per_unit", math.nan))
    acquisition_matrix = relative_frame.base_frame_chaser("acquisition_frame_id")
    timestamp_matrix = relative_frame.base_frame_chaser("timestamp_ns")
    timestamp_valid_matrix = relative_frame.base_frame_chaser("timestamp_valid")
    if relative_frame.n_frames and (
        not np.all(acquisition_matrix == acquisition_matrix[:, :1])
        or not np.all(timestamp_matrix == timestamp_matrix[:, :1])
        or not np.all(timestamp_valid_matrix == timestamp_valid_matrix[:, :1])
    ):
        _fail("Relative-frame acquisition/timing evidence differs across chasers.")
    fish_matrix = relative_frame.base_frame_chaser("fish_position_xy_px")
    fish_valid_matrix = relative_frame.base_frame_chaser("fish_position_valid")
    for values, name in (
        (fish_matrix, "fish_position_xy_px"),
        (fish_valid_matrix, "fish_position_valid"),
    ):
        reference = values[:, :1, ...]
        if values.dtype.kind == "f":
            repeated = np.array_equal(
                values, np.broadcast_to(reference, values.shape), equal_nan=True
            )
        else:
            repeated = np.array_equal(values, np.broadcast_to(reference, values.shape))
        if not repeated:
            _fail(f"Relative-frame {name} differs across chaser rows.")
    body_origin_matrix = relative_frame.body_frame_chaser("body_origin_xy_px")
    body_forward_matrix = relative_frame.body_frame_chaser("body_forward_axis_xy")
    body_left_matrix = relative_frame.body_frame_chaser("body_left_axis_xy")
    body_valid_matrix = relative_frame.body_frame_chaser("body_axes_valid")
    for values, name in (
        (body_origin_matrix, "body_origin_xy_px"),
        (body_forward_matrix, "body_forward_axis_xy"),
        (body_left_matrix, "body_left_axis_xy"),
        (body_valid_matrix, "body_axes_valid"),
    ):
        reference = values[:, :1, ...]
        repeated = (
            np.array_equal(
                values, np.broadcast_to(reference, values.shape), equal_nan=True
            )
            if values.dtype.kind == "f"
            else np.array_equal(values, np.broadcast_to(reference, values.shape))
        )
        if not repeated:
            _fail(f"Relative-frame {name} differs across chaser rows.")
    acquisition = np.asarray(acquisition_matrix[:, 0], dtype=np.int64)
    gaze, gaze_valid, vergence, vergence_valid = eye_gaze.align_to_acquisition_frames(
        acquisition
    )
    roles = semantic_role_codes_from_handles(relative_frame, semantic_selection)
    return GazeTrackingInput(
        recording_id=relative_frame.recording_id,
        source_relative_frame_run_path=relative_frame.run_path,
        source_relative_frame_manifest_sha256=relative_frame.manifest_sha256,
        source_eye_run_path=eye_gaze.run_path,
        source_eye_manifest_sha256=eye_gaze.logical_manifest_sha256,
        source_eye_convention_receipt_sha256=(eye_gaze.convention_receipt_sha256),
        source_eye_channel_policy=(
            f"{eye_gaze.channel_variant}:"
            f"{','.join(eye_gaze.gaze_channel_names)}:"
            f"{eye_gaze.vergence_channel_name}"
        ),
        source_semantic_selection_manifest_sha256=(semantic_selection.manifest_sha256),
        source_radial_run_path=radial_near_field.run_path,
        source_radial_manifest_sha256=radial_near_field.manifest_sha256,
        source_radial_payload_sha256=radial_near_field.scientific_payload_sha256,
        source_arena_geometry_and_scale=dict(geometry_binding),
        arena_center_xy_px=center,
        arena_radius_px=radius_px,
        arena_radius_mm=radius_mm,
        pixels_per_mm=pixels_per_mm,
        n_frames=relative_frame.n_frames,
        n_chasers=relative_frame.n_chasers,
        acquisition_frame_id_by_frame=acquisition,
        timestamp_ns_by_frame=np.asarray(timestamp_matrix[:, 0], dtype=np.int64),
        timestamp_valid_by_frame=np.asarray(timestamp_valid_matrix[:, 0], dtype=bool),
        semantic_role_code_by_frame=roles,
        chaser_identity_code=relative_frame.base_array("chaser_identity_code"),
        fish_position_xy_px=np.asarray(fish_matrix[:, 0, :], dtype=np.float64),
        fish_position_valid=np.asarray(fish_valid_matrix[:, 0], dtype=bool),
        chaser_position_xy_px=np.asarray(
            relative_frame.base_array("chaser_position_xy_px"), dtype=np.float64
        ),
        chaser_position_valid=relative_frame.base_array("chaser_position_valid"),
        chaser_occurrence_member=relative_frame.base_array("chaser_occurrence_member"),
        body_origin_xy_px=np.asarray(body_origin_matrix[:, 0, :], dtype=np.float64),
        body_forward_axis_xy=np.asarray(body_forward_matrix[:, 0, :], dtype=np.float64),
        body_left_axis_xy=np.asarray(body_left_matrix[:, 0, :], dtype=np.float64),
        body_axes_valid=np.asarray(body_valid_matrix[:, 0], dtype=bool),
        distance_mm=np.asarray(
            relative_frame.base_array("relative_distance_physical"),
            dtype=np.float64,
        ),
        distance_valid=relative_frame.base_array("relative_physical_valid"),
        chaser_bearing_deg=np.asarray(
            relative_frame.body_array("body_bearing_deg"), dtype=np.float64
        ),
        chaser_bearing_valid=relative_frame.body_array("body_bearing_valid"),
        gaze_signed_deg=np.asarray(gaze, dtype=np.float64),
        gaze_valid=np.asarray(gaze_valid, dtype=bool),
        vergence_deg=np.asarray(vergence, dtype=np.float64),
        vergence_valid=np.asarray(vergence_valid, dtype=bool),
        core_authority_dependency=core_authority,
        lock_threshold_deg=lock_threshold_deg,
        minimum_lock_duration_s=minimum_lock_duration_s,
        maximum_tracking_distance_mm=maximum_tracking_distance_mm,
        accessible_quantiles=accessible_quantiles,
    )


def prepare_gaze_tracking_successor_from_handles(
    relative_frame: Any,
    semantic_selection: Any,
    eye_gaze: Any,
    radial_near_field: Any,
    **kwargs: Any,
) -> PreparedGazeTracking:
    """Prepare gaze tracking from exact current source handles."""

    return prepare_gaze_tracking_successor(
        gaze_tracking_input_from_handles(
            relative_frame,
            semantic_selection,
            eye_gaze,
            radial_near_field,
            **kwargs,
        )
    )


__all__ = [
    "METHOD_ID",
    "PREPARED_SCHEMA_ID",
    "PREPARED_SCHEMA_VERSION",
    "SCHEMA_ID",
    "SCHEMA_VERSION",
    "GazeTrackingInput",
    "GazeTrackingSuccessorError",
    "PreparedGazeTracking",
    "gaze_tracking_input_from_handles",
    "prepare_gaze_tracking_successor",
    "prepare_gaze_tracking_successor_from_handles",
]
