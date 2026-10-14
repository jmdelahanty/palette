"""Generic moving-grating feature extraction for stimulus-response analysis.

Grating directions follow agent-contracts ``citrus-palette/angles``
(``stimulus_angle_convention_v1``, PR 91, vendored under
``fisheye/shared/contracts``):

* the drift direction is resolved per protocol era
  (``legacy.moving_grating_drift_rule``): orientation era
  ``drift = orientation + 180`` at positive speed and ``orientation`` at
  negative speed; motion era ``drift = motion_direction_degrees`` for either
  speed sign; unit-circle era ``motion_direction_degrees`` read natively; zero
  speed is static and has no direction;
* legacy clockwise canvas angles become ``atan2_negative_y_x_degrees`` by sign
  (``legacy_canvas_angle_to_degrees``);
* the canvas direction reaches the source-camera raster only as a vector
  through the canvas-to-camera Jacobian, never by adding an angular offset, so
  a reflected rig (negative determinant) reverses rotation sense correctly.

The camera angle is then directly comparable with Palette track headings,
which use the same convention on the source-camera raster.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import numpy as np
from fisheye.shared.heading_convention import (
    LEGACY_CANVAS_ANGLE,
    STIMULUS_ANGLE_CONVENTION,
    finite_difference_jacobian,
    legacy_canvas_angle_to_degrees,
    map_stimulus_direction,
    stimulus_angle_vector,
    wrap_delta_degrees,
)


_MOVING_GRATING = "MOVING_GRATING"

GRATING_DRIFT_RULE_ID = "citrus_palette.stimulus_angle_convention.v1.moving_grating_drift"
GRATING_DIRECTION_MAPPING_ID = "selected_canvas_to_source_camera_jacobian_at_arena_center_v1"

DRIFT_ERA_ORIENTATION = "orientation"
DRIFT_ERA_MOTION = "motion"
DRIFT_ERA_UNIT_CIRCLE = "unit_circle"

DRIFT_STATUS_RESOLVED = "drift_resolved"
DRIFT_STATUS_STATIC = "static_zero_speed"
DRIFT_STATUS_DIRECTION_MISSING = "direction_field_missing"
DRIFT_STATUS_DIRECTION_AMBIGUOUS = "orientation_and_motion_both_present"
DRIFT_STATUS_SPEED_MISSING = "speed_missing"
DRIFT_STATUS_NONFINITE = "nonfinite_direction_or_speed"
DRIFT_STATUS_REFERENCE_FRAME_UNSUPPORTED = "direction_reference_frame_unsupported"

_STIMULUS_CANVAS_FRAME = "STIMULUS_CANVAS"


class GratingDirectionError(ValueError):
    """A moving-grating step has no admissible drift direction."""


@dataclass(frozen=True)
class GratingDrift:
    """Protocol-level drift of one moving-grating step, on the stimulus canvas."""

    status: str
    era: Optional[str]
    authored_convention: str
    authored_field: Optional[str]
    authored_degrees: Optional[float]
    speed_pps: Optional[float]
    canvas_degrees: Optional[float]
    direction_reference_frame: Optional[str]

    @property
    def drifting(self) -> bool:
        return self.status == DRIFT_STATUS_RESOLVED

    def record(self) -> Dict[str, Any]:
        return {
            "grating_drift_rule": GRATING_DRIFT_RULE_ID,
            "grating_drift_status": self.status,
            "grating_drift_era": self.era,
            "grating_drift_authored_convention": self.authored_convention,
            "grating_drift_authored_field": self.authored_field,
            "grating_drift_authored_deg": self.authored_degrees,
            "grating_drift_canvas_deg": self.canvas_degrees,
            "grating_drift_canvas_convention": STIMULUS_ANGLE_CONVENTION,
            "direction_reference_frame": self.direction_reference_frame,
        }


def _flatten_stimulus_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Return the canonical Citrus protocol parameter dict for a step."""

    if not isinstance(params, dict):
        return {}
    nested = params.get("parameters")
    if isinstance(nested, dict):
        return dict(nested)
    return dict(params)


def _optional_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    return float(value)


def _signed_speed(params: Mapping[str, Any]) -> Optional[float]:
    for key in ("speed_pps", "speed_mm_per_sec", "speed_mm_s", "grating_speed_mm_s"):
        value = _optional_float(params.get(key))
        if value is not None:
            return value
    return None


def resolve_grating_drift(
    params: Mapping[str, Any],
    *,
    stimulus_angle_convention: str = LEGACY_CANVAS_ANGLE,
) -> GratingDrift:
    """Drift direction of one moving-grating step in canvas unit-circle degrees.

    ``stimulus_angle_convention`` is the convention the recording's admitted
    binding fixed for authored angles (legacy clockwise for every historical
    import). Refusals are carried in ``status``, never as a default angle.
    """

    if stimulus_angle_convention not in (LEGACY_CANVAS_ANGLE, STIMULUS_ANGLE_CONVENTION):
        raise GratingDirectionError(
            f"unknown stimulus angle convention {stimulus_angle_convention!r}"
        )
    motion = _optional_float(params.get("motion_direction_degrees"))
    orientation = _optional_float(params.get("orientation_degrees"))
    reference_frame = params.get("direction_reference_frame")
    speed = _signed_speed(params)

    def result(status, era=None, field=None, authored=None, canvas=None):
        return GratingDrift(
            status=status,
            era=era,
            authored_convention=stimulus_angle_convention,
            authored_field=field,
            authored_degrees=authored,
            speed_pps=speed,
            canvas_degrees=canvas,
            direction_reference_frame=reference_frame,
        )

    if motion is not None and orientation is not None:
        # Citrus never wrote both; a file that has both is not one era.
        return result(DRIFT_STATUS_DIRECTION_AMBIGUOUS)
    if stimulus_angle_convention == STIMULUS_ANGLE_CONVENTION:
        era, field, authored = DRIFT_ERA_UNIT_CIRCLE, "motion_direction_degrees", motion
    elif motion is not None:
        era, field, authored = DRIFT_ERA_MOTION, "motion_direction_degrees", motion
    else:
        era, field, authored = DRIFT_ERA_ORIENTATION, "orientation_degrees", orientation
    if authored is None:
        return result(DRIFT_STATUS_DIRECTION_MISSING, era, field)
    if reference_frame not in (None, _STIMULUS_CANVAS_FRAME):
        # A CAMERA_VIEW-authored direction is not a canvas angle; Palette has
        # no binding for that raster yet, so it refuses rather than guess.
        return result(DRIFT_STATUS_REFERENCE_FRAME_UNSUPPORTED, era, field, authored)
    if speed is None:
        return result(DRIFT_STATUS_SPEED_MISSING, era, field, authored)
    if not (math.isfinite(authored) and math.isfinite(speed)):
        return result(DRIFT_STATUS_NONFINITE, era, field, authored)
    if speed == 0.0:
        return result(DRIFT_STATUS_STATIC, era, field, authored)
    if era == DRIFT_ERA_UNIT_CIRCLE:
        canvas = authored
    else:
        flips = era == DRIFT_ERA_ORIENTATION and speed > 0.0
        drift_legacy = authored + 180.0 if flips else authored
        canvas = float(legacy_canvas_angle_to_degrees(drift_legacy))
    return result(DRIFT_STATUS_RESOLVED, era, field, authored, float(canvas))


def step_stimulus_angle_convention(step: Any) -> str:
    """The authored-angle convention recorded for one imported step.

    Import metadata v2 records it; earlier imports only ever came from legacy
    (clockwise) files.
    """

    raw_params = step.stimulus_params if isinstance(step.stimulus_params, dict) else {}
    moving_attrs = raw_params.get("moving_grating")
    if isinstance(moving_attrs, dict):
        recorded = moving_attrs.get("grating_drift_authored_convention")
        if recorded is not None:
            return str(recorded)
    return LEGACY_CANVAS_ANGLE


def resolve_step_grating_drift(step: Any) -> GratingDrift:
    """``resolve_grating_drift`` for one imported ``ProtocolStep``."""

    raw_params = step.stimulus_params if isinstance(step.stimulus_params, dict) else {}
    return resolve_grating_drift(
        _flatten_stimulus_params(raw_params),
        stimulus_angle_convention=step_stimulus_angle_convention(step),
    )


@dataclass(frozen=True)
class GratingCameraDirection:
    """A resolved drift mapped into the source-camera raster."""

    drift: GratingDrift
    camera_degrees: float
    det_sign: int
    canvas_point_px: tuple[float, float]

    def record(self) -> Dict[str, Any]:
        return {
            **self.drift.record(),
            "grating_direction_camera_deg": self.camera_degrees,
            "grating_direction_camera_convention": STIMULUS_ANGLE_CONVENTION,
            "direction_mapping_id": GRATING_DIRECTION_MAPPING_ID,
            "direction_mapping_det_sign": self.det_sign,
            "direction_mapping_canvas_point_px": list(self.canvas_point_px),
        }


def map_grating_drift_to_camera(
    drift: GratingDrift,
    *,
    canvas_point_px: Sequence[float],
    canvas_to_camera: Callable[[np.ndarray], np.ndarray],
) -> GratingCameraDirection:
    """Map a resolved canvas drift into the source-camera raster.

    ``canvas_to_camera`` maps ``(N, 2)`` selected-canvas pixels to source-camera
    pixels (the recording's bound transform). Refuses a step whose drift is not
    resolved; never substitutes a default direction.
    """

    if not drift.drifting or drift.canvas_degrees is None:
        raise GratingDirectionError(
            f"moving-grating step has no drift direction: {drift.status}"
        )
    point = (float(canvas_point_px[0]), float(canvas_point_px[1]))
    jacobian = finite_difference_jacobian(canvas_to_camera, point)
    camera_degrees, det_sign = map_stimulus_direction(drift.canvas_degrees, jacobian)
    return GratingCameraDirection(
        drift=drift,
        camera_degrees=camera_degrees,
        det_sign=det_sign,
        canvas_point_px=point,
    )


def resolve_grating_speed_mm_s(step: Any) -> float:
    """Grating drift speed magnitude from canonical protocol params.

    The drift direction already absorbs the sign of the authored speed (see
    ``resolve_grating_drift``), so the speed along the drift is its magnitude.
    """

    raw_params = step.stimulus_params if isinstance(step.stimulus_params, dict) else {}
    params = _flatten_stimulus_params(raw_params)
    for key in ("grating_speed_mm_s", "speed_mm_per_sec", "speed_mm_s"):
        if key in params and params[key] is not None:
            return abs(float(params[key]))
    moving_attrs = raw_params.get("moving_grating")
    if isinstance(moving_attrs, dict):
        for key in ("speed_mm_s", "speed_mm_per_sec", "grating_speed_mm_s"):
            if moving_attrs.get(key) is not None:
                return abs(float(moving_attrs[key]))
    return 0.0


def _grating_direction_vector(direction_deg: float) -> np.ndarray:
    """Unit vector on the y-down source-camera raster for a camera drift angle."""

    x, y = stimulus_angle_vector(direction_deg)
    return np.array([float(x), float(y)], dtype=np.float64)

def compute_grating_per_frame(
    tracks: Sequence[Any],
    step: Any,
    grating_dir_deg: float,
    fps: float,
) -> Dict[str, np.ndarray]:
    """Per-frame grating alignment metrics for one MOVING_GRATING step.

    Returns arrays shaped (n_fish, n_step_frames).
    """

    sf, ef = step.start_frame, step.end_frame
    n_step = max(ef - sf, 1)
    n_fish = len(tracks)

    frame_indices = np.arange(sf, ef, dtype=np.int64)
    valid = np.zeros((n_fish, n_step), dtype=bool)
    det_src = np.full((n_fish, n_step), -1, dtype=np.int8)
    alignment_angle = np.zeros((n_fish, n_step), dtype=np.float32)
    alignment_cos = np.zeros((n_fish, n_step), dtype=np.float32)
    speed_along = np.zeros((n_fish, n_step), dtype=np.float32)
    ang_vel = np.zeros((n_fish, n_step), dtype=np.float32)

    grating_dir_rad = np.deg2rad(grating_dir_deg)

    for i, t in enumerate(tracks):
        heading_step = t.heading_deg[sf:ef]
        speed_step = t.speed_mm[sf:ef]
        valid_step = (
            t.valid[sf:ef]
            & np.isfinite(heading_step)
            & np.isfinite(speed_step)
            & np.isfinite(t.angular_velocity[sf:ef])
        )
        angvel_step = t.angular_velocity[sf:ef]

        valid[i] = valid_step
        det_src[i] = t.detection_source[sf:ef]

        # Alignment angle: heading - grating direction, wrapped to [-180, 180].
        raw_diff = heading_step - grating_dir_deg
        alignment_angle[i] = wrap_delta_degrees(raw_diff, dtype=raw_diff.dtype)
        alignment_cos[i] = np.cos(np.deg2rad(alignment_angle[i]))

        # Speed projected onto grating direction.
        heading_rad = np.deg2rad(heading_step)
        speed_along[i] = speed_step * np.cos(heading_rad - grating_dir_rad)

        ang_vel[i] = angvel_step

        inv = ~valid_step
        alignment_angle[i, inv] = 0.0
        alignment_cos[i, inv] = 0.0
        speed_along[i, inv] = 0.0
        ang_vel[i, inv] = 0.0

    return {
        "frame_indices": frame_indices,
        "valid": valid,
        "detection_source": det_src,
        "alignment_angle_deg": alignment_angle,
        "alignment_cos": alignment_cos,
        "speed_along_grating_mm_s": speed_along,
        "angular_velocity_deg_s": ang_vel,
    }


def compute_grating_per_fish(
    per_frame: Dict[str, np.ndarray],
    tracks: Sequence[Any],
    step: Any,
    fps: float,
    grating_speed_mm_s: float = 0.0,
    follow_threshold: float = 0.5,
    follow_window_s: float = 1.0,
    grating_dir_deg: float | None = None,
) -> Dict[str, np.ndarray]:
    """Per-fish summary grating metrics for one step."""

    sf, ef = step.start_frame, step.end_frame
    n_fish = len(tracks)

    a_cos = per_frame["alignment_cos"]
    a_angle = per_frame["alignment_angle_deg"]
    spd_along = per_frame["speed_along_grating_mm_s"]

    mean_acos = np.zeros(n_fish, dtype=np.float32)
    rvl = np.zeros(n_fish, dtype=np.float32)
    frac_follow = np.zeros(n_fish, dtype=np.float32)
    frac_oppose = np.zeros(n_fish, dtype=np.float32)
    frac_perp = np.zeros(n_fish, dtype=np.float32)
    spd_wt_align = np.zeros(n_fish, dtype=np.float32)
    opt_gain = np.zeros(n_fish, dtype=np.float32)
    drift_along = np.zeros(n_fish, dtype=np.float32)
    drift_perp = np.zeros(n_fish, dtype=np.float32)
    latency = np.full(n_fish, np.nan, dtype=np.float32)

    if grating_dir_deg is None:
        raise GratingDirectionError(
            "grating_dir_deg is required: the camera drift direction comes from "
            "map_grating_drift_to_camera."
        )
    # Positions are y-down source-camera mm; the perpendicular is the drift
    # rotated 90 degrees counter-clockwise on the image (its left).
    along_vec = _grating_direction_vector(grating_dir_deg).astype(np.float32)
    perp_vec = _grating_direction_vector(float(grating_dir_deg) + 90.0).astype(np.float32)

    follow_window_frames = max(1, int(follow_window_s * fps))

    for i, t in enumerate(tracks):
        v = t.valid[sf:ef]
        n_valid = int(v.sum())
        if n_valid == 0:
            continue

        cos_v = a_cos[i][v]
        angle_rad = np.deg2rad(a_angle[i][v])
        spd_v = t.speed_mm[sf:ef][v]
        spd_along_v = spd_along[i][v]

        mean_acos[i] = float(np.mean(cos_v))

        mean_vec = np.mean(np.exp(1j * angle_rad))
        rvl[i] = float(np.abs(mean_vec))

        frac_follow[i] = float((cos_v > 0).sum()) / n_valid
        frac_oppose[i] = float((cos_v < 0).sum()) / n_valid
        frac_perp[i] = float((np.abs(cos_v) < 0.25).sum()) / n_valid

        total_spd = float(np.sum(spd_v))
        if total_spd > 0:
            spd_wt_align[i] = float(np.sum(spd_v * cos_v)) / total_spd

        if grating_speed_mm_s > 0:
            opt_gain[i] = float(np.mean(spd_along_v)) / grating_speed_mm_s

        pos_step = t.positions_mm[sf:ef]
        valid_pos = pos_step[v]
        if valid_pos.shape[0] >= 2:
            displacement = valid_pos[-1] - valid_pos[0]
            drift_along[i] = float(np.dot(displacement, along_vec))
            drift_perp[i] = float(np.dot(displacement, perp_vec))

        full_cos = a_cos[i]
        full_valid = v
        if follow_window_frames <= full_cos.shape[0]:
            for start in range(full_cos.shape[0] - follow_window_frames + 1):
                window = slice(start, start + follow_window_frames)
                w_valid = full_valid[window]
                n_wv = int(w_valid.sum())
                if n_wv < follow_window_frames * 0.5:
                    continue
                w_cos = full_cos[window][w_valid]
                if float(np.mean(w_cos)) > follow_threshold:
                    latency[i] = float(start) / fps if fps > 0 else 0.0
                    break

    return {
        "mean_alignment_cos": mean_acos,
        "resultant_vector_length": rvl,
        "fraction_following": frac_follow,
        "fraction_opposing": frac_oppose,
        "fraction_perpendicular": frac_perp,
        "speed_weighted_alignment": spd_wt_align,
        "optomotor_gain": opt_gain,
        "drift_along_grating_mm": drift_along,
        "drift_perp_grating_mm": drift_perp,
        "latency_to_follow_s": latency,
    }


def compute_grating_time_series(
    per_frame: Dict[str, np.ndarray],
    tracks: Sequence[Any],
    step: Any,
    fps: float,
    bin_size_s: float = 1.0,
    grating_speed_mm_s: float = 0.0,
) -> Dict[str, np.ndarray]:
    """Binned temporal dynamics for one MOVING_GRATING step."""

    sf, ef = step.start_frame, step.end_frame
    n_step = max(ef - sf, 1)
    n_fish = len(tracks)

    bin_size_frames = max(1, int(bin_size_s * fps))
    n_bins = max(1, (n_step + bin_size_frames - 1) // bin_size_frames)

    bin_center_s = np.zeros(n_bins, dtype=np.float32)
    acos_binned = np.zeros((n_fish, n_bins), dtype=np.float32)
    speed_binned = np.zeros((n_fish, n_bins), dtype=np.float32)
    follow_binned = np.zeros((n_fish, n_bins), dtype=np.float32)
    gain_binned = np.zeros((n_fish, n_bins), dtype=np.float32)

    a_cos = per_frame["alignment_cos"]
    spd_along = per_frame["speed_along_grating_mm_s"]

    for b in range(n_bins):
        bs = b * bin_size_frames
        be = min(bs + bin_size_frames, n_step)
        bin_center_s[b] = ((bs + be) / 2.0) / fps if fps > 0 else 0.0

        for i, t in enumerate(tracks):
            v = t.valid[sf + bs:sf + be]
            n_v = int(v.sum())
            if n_v == 0:
                continue
            cos_bin = a_cos[i, bs:be][v]
            spd_bin = t.speed_mm[sf + bs:sf + be][v]
            spd_along_bin = spd_along[i, bs:be][v]

            acos_binned[i, b] = float(np.mean(cos_bin))
            speed_binned[i, b] = float(np.mean(spd_bin))
            follow_binned[i, b] = float((cos_bin > 0).sum()) / n_v
            if grating_speed_mm_s > 0:
                gain_binned[i, b] = float(np.mean(spd_along_bin)) / grating_speed_mm_s

    return {
        "bin_center_s": bin_center_s,
        "alignment_cos": acos_binned,
        "speed_mm_s": speed_binned,
        "fraction_following": follow_binned,
        "optomotor_gain": gain_binned,
    }
