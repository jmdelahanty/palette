"""Radial OMR metrics for concentric-grating stimulus-response analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


CONCENTRIC_RADIAL_OMR_METHOD_VERSION = "stimulus_response_concentric_radial_omr_v1"
CONCENTRIC_RADIAL_OMR_DEFAULT_WINDOW_LENGTHS_S = (10.0, 30.0, 60.0)
CONCENTRIC_RADIAL_OMR_DEFAULT_EARLY_RESPONSE_WINDOWS_S = (5.0, 10.0)


@dataclass
class ConcentricRadialOMRStepData:
    """Radial OMR outputs for one CONCENTRIC_GRATING step."""

    per_frame: Dict[str, np.ndarray]
    per_fish: Dict[str, np.ndarray]
    per_bout: Dict[str, np.ndarray]
    windows: Dict[str, np.ndarray]
    early_windows: Dict[str, np.ndarray]
    attrs: Dict[str, Any]


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0 or not np.isfinite(denominator):
        return float("nan")
    value = float(numerator) / float(denominator)
    return value if np.isfinite(value) else float("nan")


def _flatten_stimulus_params(params: Any) -> Dict[str, Any]:
    if not isinstance(params, dict):
        return {}
    nested = params.get("parameters")
    if isinstance(nested, dict):
        out = dict(nested)
    else:
        out = dict(params)
    concentric = params.get("concentric_grating")
    if isinstance(concentric, dict):
        out.update({f"concentric_grating.{key}": value for key, value in concentric.items()})
        for key, value in concentric.items():
            out.setdefault(key, value)
    return out


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _as_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    return None


def resolve_concentric_radial_polarity(step: Any) -> Dict[str, Any]:
    """Resolve authored concentric-grating radial polarity from step metadata.

    Positive sign means outward/expanding local radial motion; negative sign
    means inward/contracting local radial motion. Current Citrus recordings
    provide authored intent only, so validation is normally false.
    """

    params = _flatten_stimulus_params(getattr(step, "stimulus_params", {}))
    sign_value = _first_present(
        params.get("stimulus_radial_sign_observed"),
        params.get("radial_sign_observed"),
        params.get("stimulus_radial_sign_authored"),
        params.get("radial_sign_authored"),
    )
    observed = _first_present(
        params.get("stimulus_radial_polarity_observed"),
        params.get("radial_polarity_observed"),
    )
    authored = _first_present(
        params.get("stimulus_radial_polarity_authored"),
        params.get("radial_polarity_authored"),
    )

    sign: Optional[int] = None
    if sign_value is not None:
        try:
            sign = 1 if float(sign_value) >= 0 else -1
        except (TypeError, ValueError):
            sign = None
    if sign is None:
        expanding = _as_bool(params.get("is_expanding"))
        if expanding is not None:
            sign = 1 if expanding else -1
            authored = "expanding" if expanding else "contracting"
    if sign is None and isinstance(authored, str):
        lowered = authored.lower()
        if lowered == "expanding":
            sign = 1
        elif lowered == "contracting":
            sign = -1
    if sign is None:
        sign = 1
        authored = authored or "expanding"

    validated = bool(_as_bool(
        _first_present(
            params.get("stimulus_radial_polarity_validated"),
            params.get("radial_polarity_validated"),
        )
    ) or False)

    effective_source = "observed" if observed is not None and validated else "authored"
    effective = observed if effective_source == "observed" else authored
    if effective is None:
        effective = "expanding" if sign > 0 else "contracting"

    return {
        "stimulus_radial_sign": int(sign),
        "stimulus_radial_polarity": str(effective),
        "stimulus_radial_sign_authored": int(sign),
        "stimulus_radial_polarity_authored": str(authored or ("expanding" if sign > 0 else "contracting")),
        "stimulus_radial_polarity_observed": str(observed) if observed is not None else None,
        "stimulus_radial_sign_observed": int(sign) if observed is not None else None,
        "stimulus_radial_polarity_source": str(
            _first_present(params.get("stimulus_radial_polarity_source"), params.get("radial_polarity_source"), "protocol_parameters.is_expanding")
        ),
        "stimulus_radial_polarity_validated": validated,
        "effective_stimulus_radial_polarity_source": effective_source,
    }


def _valid_position_radii(
    track: Any,
    start_frame: int,
    end_frame: int,
    center_mm: Tuple[float, float],
) -> Dict[str, np.ndarray]:
    start = max(int(start_frame), 0)
    end = min(int(end_frame), int(track.valid.shape[0]))
    frames = np.arange(start, end, dtype=np.int64)
    if frames.size == 0:
        return {"frames": frames, "radius": np.array([], dtype=np.float64)}
    valid = (
        track.valid[frames]
        & np.isfinite(track.positions_mm[frames]).all(axis=1)
    )
    frames = frames[valid]
    if frames.size == 0:
        return {"frames": frames, "radius": np.array([], dtype=np.float64)}
    rel = track.positions_mm[frames].astype(np.float64) - np.asarray(center_mm, dtype=np.float64)
    radius = np.linalg.norm(rel, axis=1)
    return {"frames": frames, "radius": radius}


def _radial_transition_components(
    track: Any,
    start_frame: int,
    end_frame: int,
    center_mm: Tuple[float, float],
    fps: float,
    radial_singularity_epsilon_mm: float,
) -> Dict[str, np.ndarray]:
    """Frame-to-frame radial/tangential components around a stimulus center."""

    start = max(int(start_frame), 0)
    end = min(int(end_frame), int(track.valid.shape[0]))
    frames = np.arange(max(start + 1, 1), end, dtype=np.int64)
    empty = np.array([], dtype=np.float64)
    if frames.size == 0:
        return {
            "frames": frames,
            "path": empty,
            "radial_outward": empty,
            "tangential_ccw": empty,
            "dt": empty,
            "speed": empty,
            "radius_mid": empty,
        }

    pos0 = track.positions_mm[frames - 1].astype(np.float64)
    pos1 = track.positions_mm[frames].astype(np.float64)
    valid = (
        track.valid[frames]
        & track.valid[frames - 1]
        & np.isfinite(pos0).all(axis=1)
        & np.isfinite(pos1).all(axis=1)
    )
    frames = frames[valid]
    pos0 = pos0[valid]
    pos1 = pos1[valid]
    if frames.size == 0:
        return {
            "frames": frames,
            "path": empty,
            "radial_outward": empty,
            "tangential_ccw": empty,
            "dt": empty,
            "speed": empty,
            "radius_mid": empty,
        }

    center = np.asarray(center_mm, dtype=np.float64)
    midpoint = 0.5 * (pos0 + pos1)
    rel_mid = midpoint - center
    radius_mid = np.linalg.norm(rel_mid, axis=1)
    basis_valid = radius_mid >= float(radial_singularity_epsilon_mm)
    frames = frames[basis_valid]
    pos0 = pos0[basis_valid]
    pos1 = pos1[basis_valid]
    rel_mid = rel_mid[basis_valid]
    radius_mid = radius_mid[basis_valid]
    if frames.size == 0:
        return {
            "frames": frames,
            "path": empty,
            "radial_outward": empty,
            "tangential_ccw": empty,
            "dt": empty,
            "speed": empty,
            "radius_mid": empty,
        }

    r_hat = rel_mid / radius_mid[:, None]
    theta_hat = np.column_stack((-r_hat[:, 1], r_hat[:, 0]))
    dx = pos1 - pos0
    path = np.linalg.norm(dx, axis=1)
    radial_outward = np.einsum("ij,ij->i", dx, r_hat)
    tangential_ccw = np.einsum("ij,ij->i", dx, theta_hat)

    dt = track.time_seconds[frames].astype(np.float64) - track.time_seconds[frames - 1].astype(np.float64)
    fallback_dt = 1.0 / fps if fps > 0 else 0.0
    dt[~np.isfinite(dt) | (dt <= 0.0)] = fallback_dt
    speed = track.speed_mm[frames].astype(np.float64)
    speed[~np.isfinite(speed)] = 0.0
    return {
        "frames": frames,
        "path": path,
        "radial_outward": radial_outward,
        "tangential_ccw": tangential_ccw,
        "dt": dt,
        "speed": speed,
        "radius_mid": radius_mid,
    }


def _summary_for_window(
    track: Any,
    start_frame: int,
    end_frame: int,
    center_mm: Tuple[float, float],
    stimulus_radial_sign: int,
    fps: float,
    moving_threshold_mm_s: float,
    projection_speed_deadzone_mm_s: float,
    radial_singularity_epsilon_mm: float,
    arena_radius_mm: Optional[float],
) -> Dict[str, float | int]:
    comps = _radial_transition_components(
        track,
        start_frame,
        end_frame,
        center_mm,
        fps,
        radial_singularity_epsilon_mm,
    )
    radial = comps["radial_outward"]
    tangential = comps["tangential_ccw"]
    path = comps["path"]
    dt = comps["dt"]
    speed = comps["speed"]
    aligned = float(stimulus_radial_sign) * radial

    total_path = float(np.sum(path)) if path.size else 0.0
    total_aligned = float(np.sum(aligned)) if aligned.size else 0.0
    total_radial = float(np.sum(radial)) if radial.size else 0.0
    total_tangential = float(np.sum(tangential)) if tangential.size else 0.0
    valid_transition_count = int(path.size)
    frames_possible = max(min(int(end_frame), track.valid.shape[0]) - max(int(start_frame), 0) - 1, 0)
    coverage = float(valid_transition_count) / float(frames_possible) if frames_possible > 0 else 0.0

    moving = speed >= float(moving_threshold_mm_s)
    deadzone = float(projection_speed_deadzone_mm_s) * dt
    correct = moving & (aligned > deadzone)
    opposing = moving & (aligned < -deadzone)
    correct_s = float(np.sum(dt[correct])) if dt.size else 0.0
    opposing_s = float(np.sum(dt[opposing])) if dt.size else 0.0
    classified_s = correct_s + opposing_s

    radii = _valid_position_radii(track, start_frame, end_frame, center_mm)["radius"]
    start_radius = float(radii[0]) if radii.size else float("nan")
    end_radius = float(radii[-1]) if radii.size else float("nan")
    mean_radius = float(np.mean(radii)) if radii.size else float("nan")
    net_displacement = 0.0
    if radii.size >= 2:
        pos_frames = _valid_position_radii(track, start_frame, end_frame, center_mm)["frames"]
        first = int(pos_frames[0])
        last = int(pos_frames[-1])
        net_displacement = float(np.linalg.norm(
            track.positions_mm[last].astype(np.float64) - track.positions_mm[first].astype(np.float64)
        ))

    radius_extent = float(arena_radius_mm) if arena_radius_mm is not None and np.isfinite(arena_radius_mm) and arena_radius_mm > 0 else float("nan")
    quality = 0
    if valid_transition_count == 0:
        quality = 1
    elif total_path <= 0.0:
        quality = 2

    return {
        "omr_path_index": _safe_ratio(total_aligned, total_path),
        "radial_path_index": _safe_ratio(total_radial, total_path),
        "omr_net_direction_index": _safe_ratio(float(stimulus_radial_sign) * (end_radius - start_radius), net_displacement),
        "tangential_bias_index": _safe_ratio(total_tangential, total_path),
        "stimulus_aligned_radial_displacement_mm": total_aligned,
        "radial_displacement_integrated_mm": total_radial,
        "tangential_displacement_mm": total_tangential,
        "path_length_mm": total_path,
        "net_displacement_mm": net_displacement,
        "valid_transition_count": valid_transition_count,
        "coverage_fraction": coverage,
        "time_fraction_correct_classified": _safe_ratio(correct_s, classified_s),
        "time_choice_index": _safe_ratio(correct_s - opposing_s, classified_s),
        "time_correct_s": correct_s,
        "time_opposing_s": opposing_s,
        "time_classified_s": classified_s,
        "start_radius_mm": start_radius,
        "end_radius_mm": end_radius,
        "mean_radius_mm": mean_radius,
        "start_radius_norm": _safe_ratio(start_radius, radius_extent),
        "end_radius_norm": _safe_ratio(end_radius, radius_extent),
        "mean_radius_norm": _safe_ratio(mean_radius, radius_extent),
        "available_outward_space_at_start_mm": (
            radius_extent - start_radius
            if np.isfinite(radius_extent) and np.isfinite(start_radius) else float("nan")
        ),
        "available_inward_space_at_start_mm": start_radius,
        "quality_flag": quality,
    }


def _bout_score_for_bounds(
    track: Any,
    bout: Any,
    step_start_frame: int,
    step_end_frame: int,
    center_mm: Tuple[float, float],
    stimulus_radial_sign: int,
    fps: float,
    radial_singularity_epsilon_mm: float,
) -> Dict[str, float | int]:
    start = max(int(bout.start_frame), int(step_start_frame), 0)
    end = min(int(bout.end_frame), int(step_end_frame) - 1, track.valid.shape[0] - 1)
    if end <= start:
        return {
            "start_radius_mm": float("nan"),
            "end_radius_mm": float("nan"),
            "mean_radius_mm": float("nan"),
            "radial_displacement_endpoint_mm": float("nan"),
            "radial_displacement_integrated_mm": float("nan"),
            "stimulus_aligned_radial_displacement_mm": float("nan"),
            "tangential_displacement_mm": float("nan"),
            "path_length_mm": float("nan"),
            "net_displacement_mm": float("nan"),
            "radial_omr_score": float("nan"),
            "radial_net_direction_score": float("nan"),
            "tangential_bias_score": float("nan"),
            "valid_radial_basis": 0,
            "quality_flag": 1,
        }

    comps = _radial_transition_components(
        track, start, end + 1, center_mm, fps, radial_singularity_epsilon_mm,
    )
    radii_data = _valid_position_radii(track, start, end + 1, center_mm)
    radii = radii_data["radius"]
    radius_start = float(radii[0]) if radii.size else float("nan")
    radius_end = float(radii[-1]) if radii.size else float("nan")
    radius_mean = float(np.mean(radii)) if radii.size else float("nan")
    radial_endpoint = radius_end - radius_start if np.isfinite(radius_start) and np.isfinite(radius_end) else float("nan")

    radial_integrated = float(np.sum(comps["radial_outward"])) if comps["radial_outward"].size else float("nan")
    tangential = float(np.sum(comps["tangential_ccw"])) if comps["tangential_ccw"].size else float("nan")
    path = float(np.sum(comps["path"])) if comps["path"].size else float("nan")
    aligned = float(stimulus_radial_sign) * radial_integrated if np.isfinite(radial_integrated) else float("nan")
    net_displacement = float("nan")
    if radii_data["frames"].size >= 2:
        first = int(radii_data["frames"][0])
        last = int(radii_data["frames"][-1])
        net_displacement = float(np.linalg.norm(
            track.positions_mm[last].astype(np.float64) - track.positions_mm[first].astype(np.float64)
        ))

    quality = 0
    if not comps["path"].size:
        quality = 1
    elif not np.isfinite(path) or path <= 0.0:
        quality = 2

    return {
        "start_radius_mm": radius_start,
        "end_radius_mm": radius_end,
        "mean_radius_mm": radius_mean,
        "radial_displacement_endpoint_mm": radial_endpoint,
        "radial_displacement_integrated_mm": radial_integrated,
        "stimulus_aligned_radial_displacement_mm": aligned,
        "tangential_displacement_mm": tangential,
        "path_length_mm": path,
        "net_displacement_mm": net_displacement,
        "radial_omr_score": _safe_ratio(aligned, path),
        "radial_net_direction_score": _safe_ratio(float(stimulus_radial_sign) * radial_endpoint, net_displacement),
        "tangential_bias_score": _safe_ratio(tangential, path),
        "valid_radial_basis": int(quality == 0),
        "quality_flag": quality,
    }


def _label_score(score: float, projection_deadzone: float) -> int:
    if not np.isfinite(score):
        return 0
    if score > projection_deadzone:
        return 1
    if score < -projection_deadzone:
        return -1
    return 0


def _compute_per_frame(
    tracks: Sequence[Any],
    step: Any,
    center_mm: Tuple[float, float],
    stimulus_radial_sign: int,
    fps: float,
    radial_singularity_epsilon_mm: float,
) -> Dict[str, np.ndarray]:
    sf, ef = int(step.start_frame), int(step.end_frame)
    n_step = max(ef - sf, 1)
    n_fish = len(tracks)
    frame_indices = np.arange(sf, ef, dtype=np.int64)
    valid_basis = np.zeros((n_fish, n_step), dtype=bool)
    radius = np.zeros((n_fish, n_step), dtype=np.float32)
    radial_speed = np.zeros((n_fish, n_step), dtype=np.float32)
    tangential_speed = np.zeros((n_fish, n_step), dtype=np.float32)
    aligned_speed = np.zeros((n_fish, n_step), dtype=np.float32)

    for i, track in enumerate(tracks):
        radii_data = _valid_position_radii(track, sf, ef, center_mm)
        for frame, r in zip(radii_data["frames"], radii_data["radius"]):
            idx = int(frame) - sf
            if 0 <= idx < n_step:
                radius[i, idx] = float(r)
        comps = _radial_transition_components(
            track, sf, ef, center_mm, fps, radial_singularity_epsilon_mm,
        )
        for j, frame in enumerate(comps["frames"]):
            idx = int(frame) - sf
            if 0 <= idx < n_step:
                dt = float(comps["dt"][j])
                if dt <= 0:
                    continue
                valid_basis[i, idx] = True
                radial_speed[i, idx] = float(comps["radial_outward"][j] / dt)
                tangential_speed[i, idx] = float(comps["tangential_ccw"][j] / dt)
                aligned_speed[i, idx] = float(stimulus_radial_sign) * radial_speed[i, idx]

    return {
        "frame_indices": frame_indices,
        "valid_radial_basis": valid_basis,
        "radius_mm": radius,
        "radial_speed_outward_mm_s": radial_speed,
        "tangential_speed_ccw_mm_s": tangential_speed,
        "stimulus_aligned_radial_speed_mm_s": aligned_speed,
    }


def _compute_windows(
    tracks: Sequence[Any],
    step: Any,
    center_mm: Tuple[float, float],
    stimulus_radial_sign: int,
    fps: float,
    moving_threshold_mm_s: float,
    projection_speed_deadzone_mm_s: float,
    radial_singularity_epsilon_mm: float,
    arena_radius_mm: Optional[float],
    window_lengths_s: Sequence[float],
    bouts_by_fish: Optional[Dict[int, Sequence[Any]]],
) -> Dict[str, np.ndarray]:
    full_length_s = float(getattr(step, "duration_s", 0.0))
    requested = [
        float(v) for v in window_lengths_s
        if float(v) > 0.0 and (full_length_s <= 0.0 or float(v) < full_length_s)
    ]
    if full_length_s > 0.0 and not any(abs(v - full_length_s) < 1e-6 for v in requested):
        requested.append(full_length_s)

    out: Dict[str, List[Any]] = {
        "window_id": [],
        "fish_id": [],
        "start_frame": [],
        "end_frame": [],
        "start_time_s": [],
        "end_time_s": [],
        "window_length_s": [],
        "omr_path_index": [],
        "time_choice_index": [],
        "coverage_fraction": [],
        "mean_radius_norm": [],
        "n_bouts": [],
        "quality_flag": [],
    }
    wid = 0
    for window_s in requested:
        window_frames = max(1, int(round(window_s * fps))) if fps > 0 else max(1, int(step.end_frame) - int(step.start_frame))
        cursor = int(step.start_frame)
        while cursor < int(step.end_frame):
            w_start = cursor
            w_end = min(cursor + window_frames, int(step.end_frame))
            actual_len_s = (w_end - w_start) / fps if fps > 0 else 0.0
            for track in tracks:
                summary = _summary_for_window(
                    track,
                    w_start,
                    w_end,
                    center_mm,
                    stimulus_radial_sign,
                    fps,
                    moving_threshold_mm_s,
                    projection_speed_deadzone_mm_s,
                    radial_singularity_epsilon_mm,
                    arena_radius_mm,
                )
                bouts = []
                if bouts_by_fish is not None:
                    bouts = [
                        b for b in bouts_by_fish.get(track.fish_id, [])
                        if b.start_frame < w_end and b.end_frame >= w_start
                    ]
                out["window_id"].append(wid)
                out["fish_id"].append(track.fish_id)
                out["start_frame"].append(w_start)
                out["end_frame"].append(w_end)
                out["start_time_s"].append((w_start - int(step.start_frame)) / fps if fps > 0 else 0.0)
                out["end_time_s"].append((w_end - int(step.start_frame)) / fps if fps > 0 else 0.0)
                out["window_length_s"].append(actual_len_s)
                out["omr_path_index"].append(float(summary["omr_path_index"]))
                out["time_choice_index"].append(float(summary["time_choice_index"]))
                out["coverage_fraction"].append(float(summary["coverage_fraction"]))
                out["mean_radius_norm"].append(float(summary["mean_radius_norm"]))
                out["n_bouts"].append(len(bouts))
                out["quality_flag"].append(int(summary["quality_flag"]))
            wid += 1
            cursor = w_end

    return {
        "window_id": np.array(out["window_id"], dtype=np.int32),
        "fish_id": np.array(out["fish_id"], dtype=np.int32),
        "start_frame": np.array(out["start_frame"], dtype=np.int64),
        "end_frame": np.array(out["end_frame"], dtype=np.int64),
        "start_time_s": np.array(out["start_time_s"], dtype=np.float32),
        "end_time_s": np.array(out["end_time_s"], dtype=np.float32),
        "window_length_s": np.array(out["window_length_s"], dtype=np.float32),
        "omr_path_index": np.array(out["omr_path_index"], dtype=np.float32),
        "time_choice_index": np.array(out["time_choice_index"], dtype=np.float32),
        "coverage_fraction": np.array(out["coverage_fraction"], dtype=np.float32),
        "mean_radius_norm": np.array(out["mean_radius_norm"], dtype=np.float32),
        "n_bouts": np.array(out["n_bouts"], dtype=np.int32),
        "quality_flag": np.array(out["quality_flag"], dtype=np.int8),
    }


def _compute_early_windows(
    tracks: Sequence[Any],
    step: Any,
    center_mm: Tuple[float, float],
    stimulus_radial_sign: int,
    fps: float,
    moving_threshold_mm_s: float,
    projection_speed_deadzone_mm_s: float,
    radial_singularity_epsilon_mm: float,
    arena_radius_mm: Optional[float],
    early_window_lengths_s: Sequence[float],
    bouts_by_fish: Optional[Dict[int, Sequence[Any]]],
) -> Dict[str, np.ndarray]:
    requested = sorted({float(v) for v in early_window_lengths_s if float(v) > 0.0})
    out = _compute_windows(
        tracks,
        step,
        center_mm,
        stimulus_radial_sign,
        fps,
        moving_threshold_mm_s,
        projection_speed_deadzone_mm_s,
        radial_singularity_epsilon_mm,
        arena_radius_mm,
        requested,
        bouts_by_fish,
    )
    # Keep only windows that start at the step onset; _compute_windows adds the
    # full-step length automatically, which is not an early-response window.
    if out["window_id"].size == 0:
        return out
    requested_arr = np.round(np.array(requested, dtype=np.float32), 6)
    keep = (
        (out["start_frame"] == int(step.start_frame))
        & np.isin(np.round(out["window_length_s"].astype(np.float32), 6), requested_arr)
    )
    return {name: arr[keep] for name, arr in out.items()}


def compute_step_concentric_radial_omr_metrics(
    tracks: Sequence[Any],
    step: Any,
    center_mm: Tuple[float, float],
    fps: float,
    *,
    moving_threshold_mm_s: float,
    bouts_by_fish: Optional[Dict[int, Sequence[Any]]] = None,
    projection_deadzone: float = 0.0,
    projection_speed_deadzone_mm_s: float = 0.0,
    window_lengths_s: Sequence[float] = CONCENTRIC_RADIAL_OMR_DEFAULT_WINDOW_LENGTHS_S,
    early_window_lengths_s: Sequence[float] = CONCENTRIC_RADIAL_OMR_DEFAULT_EARLY_RESPONSE_WINDOWS_S,
    radial_singularity_epsilon_mm: float = 0.5,
    arena_radius_mm: Optional[float] = None,
    center_source: str = "unavailable",
) -> ConcentricRadialOMRStepData:
    """Compute radial OMR metrics for one CONCENTRIC_GRATING step."""

    polarity = resolve_concentric_radial_polarity(step)
    stimulus_radial_sign = int(polarity["stimulus_radial_sign"])
    n_fish = len(tracks)
    fish_ids = np.array([t.fish_id for t in tracks], dtype=np.int32)

    per_frame = _compute_per_frame(
        tracks,
        step,
        center_mm,
        stimulus_radial_sign,
        fps,
        radial_singularity_epsilon_mm,
    )

    per_fish: Dict[str, np.ndarray] = {
        "fish_id": fish_ids,
        "omr_path_index": np.full(n_fish, np.nan, dtype=np.float32),
        "radial_path_index": np.full(n_fish, np.nan, dtype=np.float32),
        "omr_net_direction_index": np.full(n_fish, np.nan, dtype=np.float32),
        "tangential_bias_index": np.full(n_fish, np.nan, dtype=np.float32),
        "stimulus_aligned_radial_displacement_mm": np.zeros(n_fish, dtype=np.float32),
        "radial_displacement_integrated_mm": np.zeros(n_fish, dtype=np.float32),
        "tangential_displacement_mm": np.zeros(n_fish, dtype=np.float32),
        "path_length_mm": np.zeros(n_fish, dtype=np.float32),
        "net_displacement_mm": np.zeros(n_fish, dtype=np.float32),
        "valid_transition_count": np.zeros(n_fish, dtype=np.int32),
        "coverage_fraction": np.zeros(n_fish, dtype=np.float32),
        "time_fraction_correct_classified": np.full(n_fish, np.nan, dtype=np.float32),
        "time_choice_index": np.full(n_fish, np.nan, dtype=np.float32),
        "time_correct_s": np.zeros(n_fish, dtype=np.float32),
        "time_opposing_s": np.zeros(n_fish, dtype=np.float32),
        "time_classified_s": np.zeros(n_fish, dtype=np.float32),
        "start_radius_mm": np.full(n_fish, np.nan, dtype=np.float32),
        "end_radius_mm": np.full(n_fish, np.nan, dtype=np.float32),
        "mean_radius_mm": np.full(n_fish, np.nan, dtype=np.float32),
        "start_radius_norm": np.full(n_fish, np.nan, dtype=np.float32),
        "end_radius_norm": np.full(n_fish, np.nan, dtype=np.float32),
        "mean_radius_norm": np.full(n_fish, np.nan, dtype=np.float32),
        "available_outward_space_at_start_mm": np.full(n_fish, np.nan, dtype=np.float32),
        "available_inward_space_at_start_mm": np.full(n_fish, np.nan, dtype=np.float32),
        "bout_fraction_correct_classified": np.full(n_fish, np.nan, dtype=np.float32),
        "bout_fraction_correct_all": np.full(n_fish, np.nan, dtype=np.float32),
        "bout_choice_index": np.full(n_fish, np.nan, dtype=np.float32),
        "bout_count_total": np.zeros(n_fish, dtype=np.int32),
        "bout_count_correct": np.zeros(n_fish, dtype=np.int32),
        "bout_count_opposing": np.zeros(n_fish, dtype=np.int32),
        "bout_count_ambiguous": np.zeros(n_fish, dtype=np.int32),
        "first_aligned_bout_id": np.full(n_fish, -1, dtype=np.int32),
        "first_aligned_bout_start_frame": np.full(n_fish, -1, dtype=np.int64),
        "first_aligned_bout_latency_s": np.full(n_fish, np.nan, dtype=np.float32),
        "first_aligned_bout_score": np.full(n_fish, np.nan, dtype=np.float32),
        "first_opposing_bout_id": np.full(n_fish, -1, dtype=np.int32),
        "first_opposing_bout_start_frame": np.full(n_fish, -1, dtype=np.int64),
        "first_opposing_bout_latency_s": np.full(n_fish, np.nan, dtype=np.float32),
        "first_opposing_bout_score": np.full(n_fish, np.nan, dtype=np.float32),
        "quality_flag": np.zeros(n_fish, dtype=np.int8),
    }

    all_fish_id: List[int] = []
    all_bout_id: List[int] = []
    all_start: List[int] = []
    all_end: List[int] = []
    all_start_radius: List[float] = []
    all_end_radius: List[float] = []
    all_mean_radius: List[float] = []
    all_radial_endpoint: List[float] = []
    all_radial_integrated: List[float] = []
    all_aligned: List[float] = []
    all_tangential: List[float] = []
    all_path: List[float] = []
    all_net: List[float] = []
    all_score: List[float] = []
    all_net_score: List[float] = []
    all_tangential_score: List[float] = []
    all_label: List[int] = []
    all_valid_basis: List[int] = []
    all_quality: List[int] = []

    for i, track in enumerate(tracks):
        summary = _summary_for_window(
            track,
            step.start_frame,
            step.end_frame,
            center_mm,
            stimulus_radial_sign,
            fps,
            moving_threshold_mm_s,
            projection_speed_deadzone_mm_s,
            radial_singularity_epsilon_mm,
            arena_radius_mm,
        )
        for key, value in summary.items():
            if key in per_fish:
                per_fish[key][i] = value

        bouts = []
        if bouts_by_fish is not None:
            bouts = [
                b for b in bouts_by_fish.get(track.fish_id, [])
                if b.start_frame < step.end_frame and b.end_frame >= step.start_frame
            ]

        correct = opposing = ambiguous = 0
        for bout in bouts:
            scored = _bout_score_for_bounds(
                track,
                bout,
                step.start_frame,
                step.end_frame,
                center_mm,
                stimulus_radial_sign,
                fps,
                radial_singularity_epsilon_mm,
            )
            label = _label_score(float(scored["radial_omr_score"]), projection_deadzone)
            if label > 0:
                correct += 1
            elif label < 0:
                opposing += 1
            else:
                ambiguous += 1

            all_fish_id.append(track.fish_id)
            all_bout_id.append(int(bout.bout_id))
            all_start.append(int(bout.start_frame))
            all_end.append(int(bout.end_frame))
            all_start_radius.append(float(scored["start_radius_mm"]))
            all_end_radius.append(float(scored["end_radius_mm"]))
            all_mean_radius.append(float(scored["mean_radius_mm"]))
            all_radial_endpoint.append(float(scored["radial_displacement_endpoint_mm"]))
            all_radial_integrated.append(float(scored["radial_displacement_integrated_mm"]))
            all_aligned.append(float(scored["stimulus_aligned_radial_displacement_mm"]))
            all_tangential.append(float(scored["tangential_displacement_mm"]))
            all_path.append(float(scored["path_length_mm"]))
            all_net.append(float(scored["net_displacement_mm"]))
            all_score.append(float(scored["radial_omr_score"]))
            all_net_score.append(float(scored["radial_net_direction_score"]))
            all_tangential_score.append(float(scored["tangential_bias_score"]))
            all_label.append(label)
            all_valid_basis.append(int(scored["valid_radial_basis"]))
            all_quality.append(int(scored["quality_flag"]))

            if label != 0:
                latency_start = max(int(bout.start_frame), int(step.start_frame))
                latency_s = (
                    (latency_start - int(step.start_frame)) / fps
                    if fps > 0 else float("nan")
                )
                if label > 0 and per_fish["first_aligned_bout_id"][i] < 0:
                    per_fish["first_aligned_bout_id"][i] = int(bout.bout_id)
                    per_fish["first_aligned_bout_start_frame"][i] = latency_start
                    per_fish["first_aligned_bout_latency_s"][i] = latency_s
                    per_fish["first_aligned_bout_score"][i] = scored["radial_omr_score"]
                if label < 0 and per_fish["first_opposing_bout_id"][i] < 0:
                    per_fish["first_opposing_bout_id"][i] = int(bout.bout_id)
                    per_fish["first_opposing_bout_start_frame"][i] = latency_start
                    per_fish["first_opposing_bout_latency_s"][i] = latency_s
                    per_fish["first_opposing_bout_score"][i] = scored["radial_omr_score"]

        total = correct + opposing + ambiguous
        classified = correct + opposing
        per_fish["bout_count_total"][i] = total
        per_fish["bout_count_correct"][i] = correct
        per_fish["bout_count_opposing"][i] = opposing
        per_fish["bout_count_ambiguous"][i] = ambiguous
        per_fish["bout_fraction_correct_classified"][i] = _safe_ratio(correct, classified)
        per_fish["bout_fraction_correct_all"][i] = _safe_ratio(correct, total)
        per_fish["bout_choice_index"][i] = _safe_ratio(correct - opposing, classified)

    per_bout = {
        "fish_id": np.array(all_fish_id, dtype=np.int32),
        "bout_id": np.array(all_bout_id, dtype=np.int32),
        "start_frame": np.array(all_start, dtype=np.int64),
        "end_frame": np.array(all_end, dtype=np.int64),
        "start_radius_mm": np.array(all_start_radius, dtype=np.float32),
        "end_radius_mm": np.array(all_end_radius, dtype=np.float32),
        "mean_radius_mm": np.array(all_mean_radius, dtype=np.float32),
        "radial_displacement_endpoint_mm": np.array(all_radial_endpoint, dtype=np.float32),
        "radial_displacement_integrated_mm": np.array(all_radial_integrated, dtype=np.float32),
        "stimulus_aligned_radial_displacement_mm": np.array(all_aligned, dtype=np.float32),
        "tangential_displacement_mm": np.array(all_tangential, dtype=np.float32),
        "path_length_mm": np.array(all_path, dtype=np.float32),
        "net_displacement_mm": np.array(all_net, dtype=np.float32),
        "radial_omr_score": np.array(all_score, dtype=np.float32),
        "radial_net_direction_score": np.array(all_net_score, dtype=np.float32),
        "tangential_bias_score": np.array(all_tangential_score, dtype=np.float32),
        "omr_label": np.array(all_label, dtype=np.int8),
        "valid_radial_basis": np.array(all_valid_basis, dtype=bool),
        "quality_flag": np.array(all_quality, dtype=np.int8),
    }

    windows = _compute_windows(
        tracks,
        step,
        center_mm,
        stimulus_radial_sign,
        fps,
        moving_threshold_mm_s,
        projection_speed_deadzone_mm_s,
        radial_singularity_epsilon_mm,
        arena_radius_mm,
        window_lengths_s,
        bouts_by_fish,
    )
    early_windows = _compute_early_windows(
        tracks,
        step,
        center_mm,
        stimulus_radial_sign,
        fps,
        moving_threshold_mm_s,
        projection_speed_deadzone_mm_s,
        radial_singularity_epsilon_mm,
        arena_radius_mm,
        early_window_lengths_s,
        bouts_by_fish,
    )

    params = _flatten_stimulus_params(getattr(step, "stimulus_params", {}))
    attrs = {
        "method_version": CONCENTRIC_RADIAL_OMR_METHOD_VERSION,
        "coordinate_system": "camera_mm_polar_about_stimulus_center",
        "stimulus_center_mm": [float(center_mm[0]), float(center_mm[1])],
        "stimulus_center_source": center_source,
        "stimulus_radial_polarity": polarity["stimulus_radial_polarity"],
        "stimulus_radial_sign": stimulus_radial_sign,
        "stimulus_radial_polarity_authored": polarity["stimulus_radial_polarity_authored"],
        "stimulus_radial_sign_authored": polarity["stimulus_radial_sign_authored"],
        "stimulus_radial_polarity_observed": polarity["stimulus_radial_polarity_observed"],
        "stimulus_radial_sign_observed": polarity["stimulus_radial_sign_observed"],
        "stimulus_radial_polarity_source": polarity["stimulus_radial_polarity_source"],
        "stimulus_radial_polarity_validated": polarity["stimulus_radial_polarity_validated"],
        "effective_stimulus_radial_polarity_source": polarity["effective_stimulus_radial_polarity_source"],
        "radial_singularity_epsilon_mm": float(radial_singularity_epsilon_mm),
        "arena_radius_mm": (
            float(arena_radius_mm)
            if arena_radius_mm is not None and np.isfinite(arena_radius_mm)
            else None
        ),
        "projection_deadzone": float(projection_deadzone),
        "projection_speed_deadzone_mm_s": float(projection_speed_deadzone_mm_s),
        "moving_threshold_mm_s": float(moving_threshold_mm_s),
        "window_lengths_s": [float(v) for v in window_lengths_s],
        "early_response_window_lengths_s": [float(v) for v in early_window_lengths_s],
        "baseline_correction": "none",
        "concentric_grating_role": str(_first_present(params.get("stimulus_role"), params.get("concentric_grating_role"), "unknown")),
        "detector_estimator_policy": "bout_boundaries_from_detector_physical_metrics_from_positions",
        "position_source_array": "positions_mm",
        "speed_source_array": "speed_smoothed_mm",
        "radial_coordinate_convention": "outward_positive; stimulus_aligned = stimulus_radial_sign * outward_radial",
        "tangential_coordinate_convention": "counterclockwise_positive_about_stimulus_center",
        "quality_flag_codes": {
            "0": "ok",
            "1": "no_valid_radial_transitions_or_invalid_bout",
            "2": "no_movement",
        },
    }
    for key in (
        "spatial_freq_rpp",
        "speed_pps",
        "spatial_freq_cycles_per_mm",
        "speed_mm_per_sec",
        "speed_mm_s",
        "temporal_frequency_hz",
        "actual_rendered_temporal_frequency_hz",
        "target_radius_min_mm",
        "target_radius_max_mm",
        "target_radius_source",
        "centering_success_fraction_threshold",
    ):
        if key in params:
            attrs[key] = params[key]

    return ConcentricRadialOMRStepData(
        per_frame=per_frame,
        per_fish=per_fish,
        per_bout=per_bout,
        windows=windows,
        early_windows=early_windows,
        attrs=attrs,
    )


__all__ = [
    "CONCENTRIC_RADIAL_OMR_DEFAULT_EARLY_RESPONSE_WINDOWS_S",
    "CONCENTRIC_RADIAL_OMR_DEFAULT_WINDOW_LENGTHS_S",
    "CONCENTRIC_RADIAL_OMR_METHOD_VERSION",
    "ConcentricRadialOMRStepData",
    "compute_step_concentric_radial_omr_metrics",
    "resolve_concentric_radial_polarity",
]
