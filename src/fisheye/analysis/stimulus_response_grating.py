"""Generic moving-grating feature extraction for stimulus-response analysis."""

from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np


_MOVING_GRATING = "MOVING_GRATING"


def _flatten_stimulus_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Return the canonical Citrus protocol parameter dict for a step."""

    if not isinstance(params, dict):
        return {}
    nested = params.get("parameters")
    if isinstance(nested, dict):
        return dict(nested)
    return dict(params)


def resolve_grating_direction(
    step: Any,
    offset_deg: float = 0.0,
) -> float:
    """Extract grating drift direction in camera-space degrees.

    Reads ``orientation_degrees`` (or fallbacks) from step stimulus_params and
    applies the configured projector-to-camera angular correction. The returned
    direction is wrapped to ``[0, 360)`` for stable persisted labels.
    """

    raw_params = step.stimulus_params if isinstance(step.stimulus_params, dict) else {}
    moving_attrs = raw_params.get("moving_grating")
    if isinstance(moving_attrs, dict):
        status = str(moving_attrs.get("direction_mapping_status", ""))
        camera_direction = moving_attrs.get("grating_direction_camera_deg")
        if status == "configured_camera_offset" and camera_direction is not None:
            return float(camera_direction) % 360.0

    params = _flatten_stimulus_params(raw_params)
    direction = 0.0
    for key in ("orientation_degrees", "angle_degrees", "grating_orientation"):
        if key in params and params[key] is not None:
            direction = params[key]
            break
    else:
        if isinstance(moving_attrs, dict):
            for key in ("orientation_degrees_authored", "grating_direction_camera_deg"):
                if moving_attrs.get(key) is not None:
                    direction = moving_attrs[key]
                    break
    return (float(direction) + float(offset_deg)) % 360.0


def resolve_grating_speed_mm_s(step: Any) -> float:
    """Resolve moving-grating speed from canonical protocol params."""

    raw_params = step.stimulus_params if isinstance(step.stimulus_params, dict) else {}
    params = _flatten_stimulus_params(raw_params)
    for key in ("grating_speed_mm_s", "speed_mm_per_sec", "speed_mm_s"):
        if key in params and params[key] is not None:
            return float(params[key])
    moving_attrs = raw_params.get("moving_grating")
    if isinstance(moving_attrs, dict):
        for key in ("speed_mm_s", "speed_mm_per_sec", "grating_speed_mm_s"):
            if moving_attrs.get(key) is not None:
                return float(moving_attrs[key])
    return 0.0


def _wrap_angle(angle_deg: np.ndarray) -> np.ndarray:
    """Wrap angles to [-180, +180]."""

    return ((angle_deg + 180.0) % 360.0) - 180.0


def _grating_direction_vector(direction_deg: float) -> np.ndarray:
    """Return a unit vector in camera/mm coordinates for grating drift."""

    rad = np.deg2rad(direction_deg)
    return np.array([np.cos(rad), np.sin(rad)], dtype=np.float64)


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
        alignment_angle[i] = _wrap_angle(raw_diff)
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
        grating_dir_deg = resolve_grating_direction(step)
    grating_dir_rad = np.deg2rad(grating_dir_deg)
    along_vec = np.array([np.cos(grating_dir_rad), np.sin(grating_dir_rad)], dtype=np.float32)
    perp_vec = np.array([-np.sin(grating_dir_rad), np.cos(grating_dir_rad)], dtype=np.float32)

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
