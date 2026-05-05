"""OMR metrics for moving-grating stimulus-response analysis."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import zarr

from fisheye.analysis.stimulus_response_grating import (
    _flatten_stimulus_params,
    _grating_direction_vector,
)
from fisheye.shared.coordinate_transform import load_calibration_transform


OMR_METHOD_VERSION = "stimulus_response_omr_v3"
OMR_DEFAULT_WINDOW_LENGTHS_S = (10.0, 30.0, 60.0)
OMR_DEFAULT_EARLY_RESPONSE_WINDOWS_S = (5.0, 10.0)


@dataclass
class OMRStepData:
    """OMR responsiveness outputs for one MOVING_GRATING step."""

    per_fish: Dict[str, np.ndarray]
    per_bout: Dict[str, np.ndarray]
    windows: Dict[str, np.ndarray]
    early_windows: Dict[str, np.ndarray]
    attrs: Dict[str, Any]


def _distance_for_window(track: Any, start_frame: int, end_frame: int) -> float:
    """Return physical path length over [start_frame, end_frame)."""
    start = max(int(start_frame), 0)
    end = min(int(end_frame), track.valid.shape[0])
    if end <= start + 1:
        return 0.0
    if track.cumulative_path_distance_mm is not None:
        last = end - 1
        first = start
        if track.valid[first] and track.valid[last]:
            return float(
                track.cumulative_path_distance_mm[last]
                - track.cumulative_path_distance_mm[first]
            )
    positions = track.positions_mm[start:end]
    valid = track.valid[start:end]
    if positions.shape[0] < 2:
        return 0.0
    deltas = np.linalg.norm(np.diff(positions.astype(np.float64), axis=0), axis=1)
    valid_edges = valid[:-1] & valid[1:]
    return float(np.sum(deltas[valid_edges]))


def _finite_or_nan(value: float) -> float:
    value = float(value)
    return value if np.isfinite(value) else float("nan")


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0 or not np.isfinite(denominator):
        return float("nan")
    return _finite_or_nan(numerator / denominator)


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _moving_grating_provenance_attrs(step: Any, grating_dir_deg: float) -> Dict[str, Any]:
    """Return local stimulus-direction provenance for a moving-grating OMR group."""

    raw_params = step.stimulus_params if isinstance(step.stimulus_params, dict) else {}
    moving_attrs = raw_params.get("moving_grating")
    if not isinstance(moving_attrs, dict):
        moving_attrs = {}
    params = _flatten_stimulus_params(raw_params)
    out: Dict[str, Any] = {
        "stimulus_direction_deg": float(grating_dir_deg),
        "grating_direction_camera_deg": float(grating_dir_deg),
        "orientation_degrees_authored": _first_present(
            moving_attrs.get("orientation_degrees_authored"),
            params.get("orientation_degrees"),
            params.get("angle_degrees"),
            params.get("grating_orientation"),
        ),
        "camera_to_projector_offset_deg": moving_attrs.get("camera_to_projector_offset_deg"),
        "direction_mapping_source": moving_attrs.get("direction_mapping_source"),
        "direction_mapping_status": moving_attrs.get("direction_mapping_status"),
        "direction_mapping_validated": moving_attrs.get("direction_mapping_validated"),
    }
    for key, candidates in {
        "spatial_freq_rpp": (
            moving_attrs.get("spatial_freq_rpp"),
            params.get("spatial_freq_rpp"),
            params.get("spatial_freq_cpp"),
        ),
        "spatial_freq_cycles_per_mm": (
            moving_attrs.get("spatial_freq_cycles_per_mm"),
            params.get("spatial_freq_cycles_per_mm"),
        ),
        "speed_pps": (moving_attrs.get("speed_pps"), params.get("speed_pps")),
        "speed_mm_s": (
            moving_attrs.get("speed_mm_s"),
            moving_attrs.get("speed_mm_per_sec"),
            params.get("speed_mm_s"),
            params.get("speed_mm_per_sec"),
            params.get("grating_speed_mm_s"),
        ),
        "speed_mm_per_sec": (
            moving_attrs.get("speed_mm_per_sec"),
            moving_attrs.get("speed_mm_s"),
            params.get("speed_mm_per_sec"),
            params.get("speed_mm_s"),
            params.get("grating_speed_mm_s"),
        ),
        "temporal_frequency_hz": (
            moving_attrs.get("temporal_frequency_hz"),
            params.get("temporal_frequency_hz"),
        ),
        "actual_rendered_temporal_frequency_hz": (
            moving_attrs.get("actual_rendered_temporal_frequency_hz"),
            params.get("actual_rendered_temporal_frequency_hz"),
        ),
    }.items():
        value = _first_present(*candidates)
        if value is not None:
            out[key] = value
    return out


def _first_finite_float(*values: Any) -> Optional[float]:
    for value in values:
        if value is None:
            continue
        if isinstance(value, np.generic):
            value = value.item()
        try:
            out = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(out):
            return out
    return None


def _group_attrs(group: Any) -> Dict[str, Any]:
    if group is None or not hasattr(group, "attrs"):
        return {}
    try:
        return dict(group.attrs)
    except Exception:
        return {}


def _get_child_group(group: Any, key: str) -> Any:
    if group is None:
        return None
    getter = getattr(group, "get", None)
    if callable(getter):
        try:
            child = getter(key)
            if child is not None:
                return child
        except Exception:
            pass
    try:
        return group[key]
    except Exception:
        return None


def _iter_child_group_names(group: Any) -> List[str]:
    if group is None:
        return []
    keys_fn = getattr(group, "group_keys", None)
    if callable(keys_fn):
        try:
            return [str(key) for key in keys_fn()]
        except Exception:
            pass
    keys_fn = getattr(group, "keys", None)
    if callable(keys_fn):
        names: List[str] = []
        try:
            for key in keys_fn():
                child = _get_child_group(group, str(key))
                if isinstance(child, zarr.Group):
                    names.append(str(key))
        except Exception:
            return names
        return names
    return []


def _calibration_attr_sources(root: Any, stimulus_run: Optional[str]) -> List[Tuple[str, Dict[str, Any]]]:
    sources: List[Tuple[str, Dict[str, Any]]] = []
    analysis = _get_child_group(root, "analysis")
    analysis_cal = _get_child_group(analysis, "calibration")
    sources.append(("analysis/calibration", _group_attrs(analysis_cal)))

    root_cal = _get_child_group(root, "calibration")
    sources.append(("calibration", _group_attrs(root_cal)))

    stim_parent = _get_child_group(analysis, "stimulus_runs")
    stim_name = stimulus_run
    if not stim_name and hasattr(stim_parent, "attrs"):
        latest = stim_parent.attrs.get("latest")
        stim_name = str(latest) if latest is not None else None
    stim_group = _get_child_group(stim_parent, stim_name) if stim_name else None
    stim_cal = _get_child_group(stim_group, "calibration")
    for camera_id in _iter_child_group_names(stim_cal):
        sources.append((
            f"analysis/stimulus_runs/{stim_name}/calibration/{camera_id}",
            _group_attrs(_get_child_group(stim_cal, camera_id)),
        ))
    return [(name, attrs) for name, attrs in sources if attrs]


def _resolve_omr_arena_geometry_mm(
    root: Any,
    stimulus_run: Optional[str],
    direction_xy: np.ndarray,
) -> Tuple[Optional[Tuple[float, float]], Optional[float], str]:
    """Resolve arena center and half-extent along the grating axis in mm."""

    cal = load_calibration_transform(root, stimulus_run=stimulus_run)
    pixel_to_mm = _first_finite_float(cal.get("pixel_to_mm"))
    projector_pixels_per_mm = _first_finite_float(cal.get("pixels_per_mm_projector"))
    sources = _calibration_attr_sources(root, stimulus_run)

    center_mm: Optional[Tuple[float, float]] = None
    center_source = ""
    if cal.get("arena_center_px") is not None and pixel_to_mm is not None:
        cx, cy = cal["arena_center_px"]
        center_mm = (float(cx) * pixel_to_mm, float(cy) * pixel_to_mm)
        center_source = "arena_center_px_camera"
    for source_name, attrs in sources:
        if center_mm is None:
            cx = _first_finite_float(
                attrs.get("arena_center_x_mm"),
                attrs.get("experimental_area_center_x_mm"),
            )
            cy = _first_finite_float(
                attrs.get("arena_center_y_mm"),
                attrs.get("experimental_area_center_y_mm"),
            )
            if cx is not None and cy is not None:
                center_mm = (cx, cy)
                center_source = f"{source_name}:center_mm"
        if center_mm is None:
            cx = _first_finite_float(attrs.get("experimental_area_center_x_px"))
            cy = _first_finite_float(attrs.get("experimental_area_center_y_px"))
            pixels_per_mm = _first_finite_float(
                attrs.get("pixels_per_mm_projector"),
                projector_pixels_per_mm,
            )
            if cx is not None and cy is not None and pixels_per_mm is not None and pixels_per_mm > 0:
                # Citrus experimental-area fields are in stimulus/projector
                # pixels, matching local arena millimetres after projector
                # scale conversion. Do not convert them with camera pixel_to_mm.
                center_mm = (cx / pixels_per_mm, cy / pixels_per_mm)
                center_source = f"{source_name}:experimental_area_center_projector_px"
        if center_mm is None:
            cx = _first_finite_float(attrs.get("arena_center_x_px"))
            cy = _first_finite_float(attrs.get("arena_center_y_px"))
            if cx is not None and cy is not None and pixel_to_mm is not None:
                center_mm = (cx * pixel_to_mm, cy * pixel_to_mm)
                center_source = f"{source_name}:arena_center_camera_px"
        if center_mm is None:
            w_mm = _first_finite_float(attrs.get("sub_arena_width_mm"))
            h_mm = _first_finite_float(attrs.get("sub_arena_height_mm"))
            if w_mm is not None and h_mm is not None and w_mm > 0 and h_mm > 0:
                center_mm = (0.5 * w_mm, 0.5 * h_mm)
                center_source = f"{source_name}:sub_arena_size_mm"
        if center_mm is None:
            w_px = _first_finite_float(attrs.get("sub_arena_width_px"))
            h_px = _first_finite_float(attrs.get("sub_arena_height_px"))
            pixels_per_mm = _first_finite_float(
                attrs.get("pixels_per_mm_projector"),
                projector_pixels_per_mm,
            )
            if w_px is not None and h_px is not None and pixels_per_mm is not None and pixels_per_mm > 0:
                center_mm = (0.5 * w_px / pixels_per_mm, 0.5 * h_px / pixels_per_mm)
                center_source = f"{source_name}:sub_arena_size_projector_px"

    extent_mm: Optional[float] = None
    extent_source = ""
    for source_name, attrs in sources:
        radius_mm = _first_finite_float(
            attrs.get("arena_radius_mm"),
            attrs.get("experimental_area_radius_mm"),
        )
        if radius_mm is not None and radius_mm > 0:
            extent_mm = radius_mm
            extent_source = f"{source_name}:radius_mm"
            break

        radius_px = _first_finite_float(attrs.get("experimental_area_radius_px"))
        pixels_per_mm = _first_finite_float(
            attrs.get("pixels_per_mm_projector"),
            projector_pixels_per_mm,
        )
        if radius_px is not None and radius_px > 0 and pixels_per_mm is not None and pixels_per_mm > 0:
            extent_mm = radius_px / pixels_per_mm
            extent_source = f"{source_name}:experimental_area_radius_projector_px"
            break

        radius_px = _first_finite_float(attrs.get("arena_radius_px"))
        if radius_px is not None and radius_px > 0 and pixel_to_mm is not None:
            extent_mm = radius_px * pixel_to_mm
            extent_source = f"{source_name}:arena_radius_camera_px"
            break

        width_mm = _first_finite_float(
            attrs.get("arena_width_mm"),
            attrs.get("experimental_area_width_mm"),
            attrs.get("sub_arena_width_mm"),
        )
        height_mm = _first_finite_float(
            attrs.get("arena_height_mm"),
            attrs.get("experimental_area_height_mm"),
            attrs.get("sub_arena_height_mm"),
        )
        if width_mm is None or height_mm is None:
            width_px = _first_finite_float(
                attrs.get("arena_width_px"),
                attrs.get("experimental_area_width_px"),
                attrs.get("sub_arena_width_px"),
            )
            height_px = _first_finite_float(
                attrs.get("arena_height_px"),
                attrs.get("experimental_area_height_px"),
                attrs.get("sub_arena_height_px"),
            )
            if width_px is not None and height_px is not None and pixel_to_mm is not None:
                width_mm = width_px * pixel_to_mm
                height_mm = height_px * pixel_to_mm
        if width_mm is not None and height_mm is not None and width_mm > 0 and height_mm > 0:
            extent_mm = 0.5 * (
                abs(float(direction_xy[0])) * width_mm
                + abs(float(direction_xy[1])) * height_mm
            )
            extent_source = f"{source_name}:axis_projected_rectangle"
            break

    if center_mm is None and extent_mm is None:
        return None, None, "unavailable"

    parts = []
    if center_mm is not None:
        parts.append(center_source or "center")
    if extent_mm is not None:
        parts.append(extent_source or "extent")
    return center_mm, extent_mm, ";".join(parts)


def _position_axis_metrics(
    track: DenseTrack,
    start_frame: int,
    end_frame: int,
    direction_xy: np.ndarray,
    arena_center_mm: Optional[Tuple[float, float]],
    arena_axis_extent_mm: Optional[float],
) -> Dict[str, float]:
    """Project fish occupancy onto the stimulus axis for one step/window."""

    keys = {
        "start_position_axis_mm": float("nan"),
        "end_position_axis_mm": float("nan"),
        "mean_position_axis_mm": float("nan"),
        "start_position_axis_norm": float("nan"),
        "end_position_axis_norm": float("nan"),
        "mean_position_axis_norm": float("nan"),
        "fraction_time_correct_side": float("nan"),
        "available_forward_space_at_start_mm": float("nan"),
        "available_backward_space_at_start_mm": float("nan"),
        "available_forward_space_at_start_norm": float("nan"),
        "available_backward_space_at_start_norm": float("nan"),
        "opportunity_normalized_parallel_displacement": float("nan"),
    }
    if arena_center_mm is None:
        return keys

    start = max(int(start_frame), 0)
    end = min(int(end_frame), int(track.valid.shape[0]))
    if end <= start:
        return keys

    frame_valid = (
        track.valid[start:end]
        & np.isfinite(track.positions_mm[start:end]).all(axis=1)
    )
    if not np.any(frame_valid):
        return keys

    positions = track.positions_mm[start:end][frame_valid].astype(np.float64)
    center = np.asarray(arena_center_mm, dtype=np.float64)
    axis = (positions - center) @ direction_xy.astype(np.float64)
    keys["start_position_axis_mm"] = float(axis[0])
    keys["end_position_axis_mm"] = float(axis[-1])
    keys["mean_position_axis_mm"] = float(np.mean(axis))
    keys["fraction_time_correct_side"] = float(np.mean(axis > 0.0))

    extent = float(arena_axis_extent_mm) if arena_axis_extent_mm is not None else float("nan")
    if np.isfinite(extent) and extent > 0.0:
        start_norm = keys["start_position_axis_mm"] / extent
        end_norm = keys["end_position_axis_mm"] / extent
        mean_norm = keys["mean_position_axis_mm"] / extent
        keys["start_position_axis_norm"] = _finite_or_nan(start_norm)
        keys["end_position_axis_norm"] = _finite_or_nan(end_norm)
        keys["mean_position_axis_norm"] = _finite_or_nan(mean_norm)
        keys["available_forward_space_at_start_mm"] = _finite_or_nan(extent - keys["start_position_axis_mm"])
        keys["available_backward_space_at_start_mm"] = _finite_or_nan(extent + keys["start_position_axis_mm"])
        keys["available_forward_space_at_start_norm"] = _finite_or_nan(1.0 - start_norm)
        keys["available_backward_space_at_start_norm"] = _finite_or_nan(1.0 + start_norm)
        parallel = keys["end_position_axis_mm"] - keys["start_position_axis_mm"]
        denom = (
            keys["available_forward_space_at_start_mm"]
            if parallel >= 0.0 else keys["available_backward_space_at_start_mm"]
        )
        keys["opportunity_normalized_parallel_displacement"] = _safe_ratio(parallel, denom)
    return keys


def _valid_transition_components(
    track: DenseTrack,
    start_frame: int,
    end_frame: int,
    direction_xy: np.ndarray,
    fps: float,
) -> Dict[str, np.ndarray]:
    """Frame-to-frame physical displacement components for one window.

    Returned arrays are indexed by the current frame in each transition. A
    transition from frame ``t-1`` to ``t`` is included only when both frames are
    valid. This prevents OMR displacement from silently crossing tracking gaps.
    """

    start = max(int(start_frame), 0)
    end = min(int(end_frame), int(track.valid.shape[0]))
    current_frames = np.arange(max(start + 1, 1), end, dtype=np.int64)
    if current_frames.size == 0:
        empty_float = np.array([], dtype=np.float64)
        return {
            "frames": current_frames,
            "dx": np.empty((0, 2), dtype=np.float64),
            "path": empty_float,
            "parallel": empty_float,
            "dt": empty_float,
            "speed": empty_float,
        }

    valid = (
        track.valid[current_frames]
        & track.valid[current_frames - 1]
        & np.isfinite(track.positions_mm[current_frames]).all(axis=1)
        & np.isfinite(track.positions_mm[current_frames - 1]).all(axis=1)
    )
    frames = current_frames[valid]
    if frames.size == 0:
        empty_float = np.array([], dtype=np.float64)
        return {
            "frames": frames,
            "dx": np.empty((0, 2), dtype=np.float64),
            "path": empty_float,
            "parallel": empty_float,
            "dt": empty_float,
            "speed": empty_float,
        }

    dx = (
        track.positions_mm[frames].astype(np.float64)
        - track.positions_mm[frames - 1].astype(np.float64)
    )
    path = np.linalg.norm(dx, axis=1)
    parallel = dx @ direction_xy.astype(np.float64)

    dt = (
        track.time_seconds[frames].astype(np.float64)
        - track.time_seconds[frames - 1].astype(np.float64)
    )
    fallback_dt = 1.0 / fps if fps > 0 else 0.0
    dt[~np.isfinite(dt) | (dt <= 0.0)] = fallback_dt
    speed = track.speed_mm[frames].astype(np.float64)

    return {
        "frames": frames,
        "dx": dx,
        "path": path,
        "parallel": parallel,
        "dt": dt,
        "speed": speed,
    }


def _omr_summary_for_window(
    track: DenseTrack,
    start_frame: int,
    end_frame: int,
    direction_xy: np.ndarray,
    fps: float,
    moving_threshold_mm_s: float,
    projection_speed_deadzone_mm_s: float,
    arena_center_mm: Optional[Tuple[float, float]] = None,
    arena_axis_extent_mm: Optional[float] = None,
) -> Dict[str, float | int]:
    """Compute physical OMR summary metrics for one fish/window."""

    components = _valid_transition_components(
        track, start_frame, end_frame, direction_xy, fps,
    )
    parallel = components["parallel"]
    path = components["path"]
    dt = components["dt"]
    speed = components["speed"]

    total_parallel = float(np.sum(parallel)) if parallel.size else 0.0
    total_path = float(np.sum(path)) if path.size else 0.0
    valid_transition_count = int(parallel.size)

    frames_possible = max(min(int(end_frame), track.valid.shape[0]) - max(int(start_frame), 0) - 1, 0)
    coverage = (
        float(valid_transition_count) / float(frames_possible)
        if frames_possible > 0 else 0.0
    )

    if valid_transition_count > 0:
        valid_frames = np.flatnonzero(track.valid[max(int(start_frame), 0):min(int(end_frame), track.valid.shape[0])])
    else:
        valid_frames = np.array([], dtype=np.int64)
    if valid_frames.size >= 2:
        offset = max(int(start_frame), 0)
        first_frame = int(valid_frames[0]) + offset
        last_frame = int(valid_frames[-1]) + offset
        net_dx = (
            track.positions_mm[last_frame].astype(np.float64)
            - track.positions_mm[first_frame].astype(np.float64)
        )
        net_displacement = float(np.linalg.norm(net_dx))
        net_parallel = float(net_dx @ direction_xy)
    else:
        net_displacement = 0.0
        net_parallel = 0.0

    moving = speed >= float(moving_threshold_mm_s)
    deadzone = float(projection_speed_deadzone_mm_s) * dt
    correct = moving & (parallel > deadzone)
    opposing = moving & (parallel < -deadzone)
    correct_s = float(np.sum(dt[correct])) if dt.size else 0.0
    opposing_s = float(np.sum(dt[opposing])) if dt.size else 0.0
    classified_s = correct_s + opposing_s

    if valid_transition_count == 0:
        quality_flag = 1
    elif total_path <= 0.0:
        quality_flag = 2
    else:
        quality_flag = 0

    result: Dict[str, float | int] = {
        "omr_path_index": _safe_ratio(total_parallel, total_path),
        "omr_net_direction_index": _safe_ratio(net_parallel, net_displacement),
        "parallel_displacement_mm": total_parallel,
        "net_displacement_mm": net_displacement,
        "path_length_mm": total_path,
        "valid_transition_count": valid_transition_count,
        "coverage_fraction": coverage,
        "time_fraction_correct_classified": _safe_ratio(correct_s, classified_s),
        "time_choice_index": _safe_ratio(correct_s - opposing_s, classified_s),
        "time_correct_s": correct_s,
        "time_opposing_s": opposing_s,
        "time_classified_s": classified_s,
        "quality_flag": quality_flag,
    }
    result.update(_position_axis_metrics(
        track,
        start_frame,
        end_frame,
        direction_xy,
        arena_center_mm,
        arena_axis_extent_mm,
    ))
    return result


def _bout_omr_score_for_bounds(
    track: DenseTrack,
    bout: BoutEntry,
    start_frame: int,
    end_frame: int,
    direction_xy: np.ndarray,
) -> Tuple[float, float, float, float]:
    """Return per-bout OMR score and displacement/path components in bounds."""

    start = max(int(bout.start_frame), int(start_frame), 0)
    end = min(int(bout.end_frame), int(end_frame) - 1, track.valid.shape[0] - 1)
    if end <= start or not (track.valid[start] and track.valid[end]):
        return float("nan"), float("nan"), float("nan"), float("nan")

    displacement_xy = (
        track.positions_mm[end].astype(np.float64)
        - track.positions_mm[start].astype(np.float64)
    )
    bout_displacement = float(np.linalg.norm(displacement_xy))
    parallel = float(displacement_xy @ direction_xy)
    score = _safe_ratio(parallel, bout_displacement)
    path = _distance_for_window(track, start, end + 1)
    return score, parallel, bout_displacement, path


def _bout_omr_score(
    track: DenseTrack,
    bout: BoutEntry,
    step: ProtocolStep,
    direction_xy: np.ndarray,
) -> Tuple[float, float, float, float]:
    """Return per-bout OMR score and displacement/path components."""

    return _bout_omr_score_for_bounds(
        track,
        bout,
        step.start_frame,
        step.end_frame,
        direction_xy,
    )


def _bout_omr_label(score: float, projection_deadzone: float) -> Tuple[int, int]:
    """Classify one per-bout OMR score into aligned/opposing/ambiguous."""

    if not np.isfinite(score):
        return 0, 1
    if score > projection_deadzone:
        return 1, 0
    if score < -projection_deadzone:
        return -1, 0
    return 0, 0


def _weighted_bout_omr_summary(
    labels: Sequence[int],
    parallel_displacements_mm: Sequence[float],
    bout_displacements_mm: Sequence[float],
    bout_path_lengths_mm: Sequence[float],
) -> Dict[str, float]:
    """Summarize bout-direction evidence weighted by physical movement."""

    label_arr = np.asarray(labels, dtype=np.int8)
    parallel = np.asarray(parallel_displacements_mm, dtype=np.float64)
    displacement = np.asarray(bout_displacements_mm, dtype=np.float64)
    path = np.asarray(bout_path_lengths_mm, dtype=np.float64)

    finite_path = np.isfinite(parallel) & np.isfinite(path) & (path > 0.0)
    finite_displacement = np.isfinite(displacement) & (displacement > 0.0)
    aligned = label_arr > 0
    opposing = label_arr < 0

    total_parallel = float(np.sum(parallel[finite_path])) if finite_path.size else 0.0
    total_path = float(np.sum(path[finite_path])) if finite_path.size else 0.0
    total_displacement = (
        float(np.sum(displacement[finite_displacement])) if finite_displacement.size else 0.0
    )
    aligned_path = float(np.sum(path[finite_path & aligned])) if finite_path.size else 0.0
    opposing_path = float(np.sum(path[finite_path & opposing])) if finite_path.size else 0.0
    aligned_displacement = (
        float(np.sum(displacement[finite_displacement & aligned]))
        if finite_displacement.size else 0.0
    )
    opposing_displacement = (
        float(np.sum(displacement[finite_displacement & opposing]))
        if finite_displacement.size else 0.0
    )
    classifiable_path = aligned_path + opposing_path
    classifiable_displacement = aligned_displacement + opposing_displacement

    return {
        "bout_path_index": _safe_ratio(total_parallel, total_path),
        "bout_parallel_displacement_sum_mm": total_parallel,
        "bout_path_length_sum_mm": total_path,
        "bout_displacement_sum_mm": total_displacement,
        "bout_classified_path_length_sum_mm": classifiable_path,
        "bout_classified_displacement_sum_mm": classifiable_displacement,
        "bout_fraction_correct_weighted_by_path": _safe_ratio(aligned_path, classifiable_path),
        "bout_fraction_correct_weighted_by_displacement": _safe_ratio(
            aligned_displacement,
            classifiable_displacement,
        ),
        "bout_classifiable_path_fraction": _safe_ratio(classifiable_path, total_path),
        "bout_classifiable_displacement_fraction": _safe_ratio(
            classifiable_displacement,
            total_displacement,
        ),
    }


def compute_step_omr_metrics(
    tracks: Sequence[DenseTrack],
    step: ProtocolStep,
    grating_dir_deg: float,
    fps: float,
    *,
    moving_threshold_mm_s: float,
    bouts_by_fish: Optional[Dict[int, List[BoutEntry]]] = None,
    projection_deadzone: float = 0.0,
    projection_speed_deadzone_mm_s: float = 0.0,
    window_lengths_s: Sequence[float] = OMR_DEFAULT_WINDOW_LENGTHS_S,
    early_window_lengths_s: Sequence[float] = OMR_DEFAULT_EARLY_RESPONSE_WINDOWS_S,
    position_anchor: str = "positions_mm",
    arena_center_mm: Optional[Tuple[float, float]] = None,
    arena_axis_extent_mm: Optional[float] = None,
    arena_geometry_source: str = "unavailable",
) -> "OMRStepData":
    """Compute OMR responsiveness metrics for one static MOVING_GRATING step."""

    direction_xy = _grating_direction_vector(grating_dir_deg)
    fish_ids = np.array([t.fish_id for t in tracks], dtype=np.int32)
    n_fish = len(tracks)

    per_fish: Dict[str, np.ndarray] = {
        "fish_id": fish_ids,
        "omr_path_index": np.full(n_fish, np.nan, dtype=np.float32),
        "omr_net_direction_index": np.full(n_fish, np.nan, dtype=np.float32),
        "parallel_displacement_mm": np.zeros(n_fish, dtype=np.float32),
        "net_displacement_mm": np.zeros(n_fish, dtype=np.float32),
        "path_length_mm": np.zeros(n_fish, dtype=np.float32),
        "valid_transition_count": np.zeros(n_fish, dtype=np.int32),
        "coverage_fraction": np.zeros(n_fish, dtype=np.float32),
        "bout_fraction_correct_classified": np.full(n_fish, np.nan, dtype=np.float32),
        "bout_fraction_correct_all": np.full(n_fish, np.nan, dtype=np.float32),
        "bout_choice_index": np.full(n_fish, np.nan, dtype=np.float32),
        "bout_path_index": np.full(n_fish, np.nan, dtype=np.float32),
        "bout_fraction_correct_weighted_by_path": np.full(n_fish, np.nan, dtype=np.float32),
        "bout_fraction_correct_weighted_by_displacement": np.full(n_fish, np.nan, dtype=np.float32),
        "bout_parallel_displacement_sum_mm": np.zeros(n_fish, dtype=np.float32),
        "bout_path_length_sum_mm": np.zeros(n_fish, dtype=np.float32),
        "bout_displacement_sum_mm": np.zeros(n_fish, dtype=np.float32),
        "bout_classified_path_length_sum_mm": np.zeros(n_fish, dtype=np.float32),
        "bout_classified_displacement_sum_mm": np.zeros(n_fish, dtype=np.float32),
        "bout_classifiable_path_fraction": np.full(n_fish, np.nan, dtype=np.float32),
        "bout_classifiable_displacement_fraction": np.full(n_fish, np.nan, dtype=np.float32),
        "bout_count_total": np.zeros(n_fish, dtype=np.int32),
        "bout_count_correct": np.zeros(n_fish, dtype=np.int32),
        "bout_count_opposing": np.zeros(n_fish, dtype=np.int32),
        "bout_count_ambiguous": np.zeros(n_fish, dtype=np.int32),
        "time_fraction_correct_classified": np.full(n_fish, np.nan, dtype=np.float32),
        "time_choice_index": np.full(n_fish, np.nan, dtype=np.float32),
        "time_correct_s": np.zeros(n_fish, dtype=np.float32),
        "time_opposing_s": np.zeros(n_fish, dtype=np.float32),
        "time_classified_s": np.zeros(n_fish, dtype=np.float32),
        "start_position_axis_mm": np.full(n_fish, np.nan, dtype=np.float32),
        "end_position_axis_mm": np.full(n_fish, np.nan, dtype=np.float32),
        "mean_position_axis_mm": np.full(n_fish, np.nan, dtype=np.float32),
        "start_position_axis_norm": np.full(n_fish, np.nan, dtype=np.float32),
        "end_position_axis_norm": np.full(n_fish, np.nan, dtype=np.float32),
        "mean_position_axis_norm": np.full(n_fish, np.nan, dtype=np.float32),
        "fraction_time_correct_side": np.full(n_fish, np.nan, dtype=np.float32),
        "available_forward_space_at_start_mm": np.full(n_fish, np.nan, dtype=np.float32),
        "available_backward_space_at_start_mm": np.full(n_fish, np.nan, dtype=np.float32),
        "available_forward_space_at_start_norm": np.full(n_fish, np.nan, dtype=np.float32),
        "available_backward_space_at_start_norm": np.full(n_fish, np.nan, dtype=np.float32),
        "opportunity_normalized_parallel_displacement": np.full(n_fish, np.nan, dtype=np.float32),
        "first_aligned_bout_id": np.full(n_fish, -1, dtype=np.int32),
        "first_aligned_bout_start_frame": np.full(n_fish, -1, dtype=np.int64),
        "first_aligned_bout_latency_s": np.full(n_fish, np.nan, dtype=np.float32),
        "first_aligned_bout_score": np.full(n_fish, np.nan, dtype=np.float32),
        "first_opposing_bout_id": np.full(n_fish, -1, dtype=np.int32),
        "first_opposing_bout_start_frame": np.full(n_fish, -1, dtype=np.int64),
        "first_opposing_bout_latency_s": np.full(n_fish, np.nan, dtype=np.float32),
        "first_opposing_bout_score": np.full(n_fish, np.nan, dtype=np.float32),
        "first_classified_bout_id": np.full(n_fish, -1, dtype=np.int32),
        "first_classified_bout_start_frame": np.full(n_fish, -1, dtype=np.int64),
        "first_classified_bout_latency_s": np.full(n_fish, np.nan, dtype=np.float32),
        "first_classified_bout_score": np.full(n_fish, np.nan, dtype=np.float32),
        "quality_flag": np.zeros(n_fish, dtype=np.int8),
    }

    all_fish_id: List[int] = []
    all_bout_id: List[int] = []
    all_start: List[int] = []
    all_end: List[int] = []
    all_score: List[float] = []
    all_parallel: List[float] = []
    all_displacement: List[float] = []
    all_path: List[float] = []
    all_label: List[int] = []
    all_quality: List[int] = []

    for i, track in enumerate(tracks):
        summary = _omr_summary_for_window(
            track,
            step.start_frame,
            step.end_frame,
            direction_xy,
            fps,
            moving_threshold_mm_s,
            projection_speed_deadzone_mm_s,
            arena_center_mm,
            arena_axis_extent_mm,
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
        track_labels: List[int] = []
        track_parallel: List[float] = []
        track_displacement: List[float] = []
        track_path: List[float] = []
        for bout in bouts:
            score, parallel, displacement, path = _bout_omr_score(
                track, bout, step, direction_xy,
            )
            label, quality = _bout_omr_label(score, projection_deadzone)
            if label > 0:
                correct += 1
            elif label < 0:
                opposing += 1
            else:
                ambiguous += 1
            track_labels.append(label)
            track_parallel.append(parallel)
            track_displacement.append(displacement)
            track_path.append(path)

            all_fish_id.append(track.fish_id)
            all_bout_id.append(bout.bout_id)
            all_start.append(bout.start_frame)
            all_end.append(bout.end_frame)
            all_score.append(score)
            all_parallel.append(parallel)
            all_displacement.append(displacement)
            all_path.append(path)
            all_label.append(label)
            all_quality.append(quality)

            if label != 0:
                latency_start = max(int(bout.start_frame), int(step.start_frame))
                latency_s = (
                    (latency_start - int(step.start_frame)) / fps
                    if fps > 0 else float("nan")
                )
                if per_fish["first_classified_bout_id"][i] < 0:
                    per_fish["first_classified_bout_id"][i] = int(bout.bout_id)
                    per_fish["first_classified_bout_start_frame"][i] = latency_start
                    per_fish["first_classified_bout_latency_s"][i] = latency_s
                    per_fish["first_classified_bout_score"][i] = score
                if label > 0 and per_fish["first_aligned_bout_id"][i] < 0:
                    per_fish["first_aligned_bout_id"][i] = int(bout.bout_id)
                    per_fish["first_aligned_bout_start_frame"][i] = latency_start
                    per_fish["first_aligned_bout_latency_s"][i] = latency_s
                    per_fish["first_aligned_bout_score"][i] = score
                if label < 0 and per_fish["first_opposing_bout_id"][i] < 0:
                    per_fish["first_opposing_bout_id"][i] = int(bout.bout_id)
                    per_fish["first_opposing_bout_start_frame"][i] = latency_start
                    per_fish["first_opposing_bout_latency_s"][i] = latency_s
                    per_fish["first_opposing_bout_score"][i] = score

        total = correct + opposing + ambiguous
        classified = correct + opposing
        per_fish["bout_count_total"][i] = total
        per_fish["bout_count_correct"][i] = correct
        per_fish["bout_count_opposing"][i] = opposing
        per_fish["bout_count_ambiguous"][i] = ambiguous
        per_fish["bout_fraction_correct_classified"][i] = _safe_ratio(correct, classified)
        per_fish["bout_fraction_correct_all"][i] = _safe_ratio(correct, total)
        per_fish["bout_choice_index"][i] = _safe_ratio(correct - opposing, classified)
        weighted = _weighted_bout_omr_summary(
            track_labels,
            track_parallel,
            track_displacement,
            track_path,
        )
        for key, value in weighted.items():
            per_fish[key][i] = value

    per_bout = {
        "fish_id": np.array(all_fish_id, dtype=np.int32),
        "bout_id": np.array(all_bout_id, dtype=np.int32),
        "start_frame": np.array(all_start, dtype=np.int64),
        "end_frame": np.array(all_end, dtype=np.int64),
        "per_bout_omr_score": np.array(all_score, dtype=np.float32),
        "parallel_displacement_mm": np.array(all_parallel, dtype=np.float32),
        "bout_displacement_mm": np.array(all_displacement, dtype=np.float32),
        "bout_path_length_mm": np.array(all_path, dtype=np.float32),
        "correct_label": np.array(all_label, dtype=np.int8),
        "quality_flag": np.array(all_quality, dtype=np.int8),
    }

    windows = _compute_omr_windows(
        tracks,
        step,
        direction_xy,
        fps,
        moving_threshold_mm_s=moving_threshold_mm_s,
        projection_speed_deadzone_mm_s=projection_speed_deadzone_mm_s,
        window_lengths_s=window_lengths_s,
        bouts_by_fish=bouts_by_fish,
        arena_center_mm=arena_center_mm,
        arena_axis_extent_mm=arena_axis_extent_mm,
    )

    early_windows = _compute_omr_early_windows(
        tracks,
        step,
        direction_xy,
        fps,
        moving_threshold_mm_s=moving_threshold_mm_s,
        projection_deadzone=projection_deadzone,
        projection_speed_deadzone_mm_s=projection_speed_deadzone_mm_s,
        early_window_lengths_s=early_window_lengths_s,
        bouts_by_fish=bouts_by_fish,
        arena_center_mm=arena_center_mm,
        arena_axis_extent_mm=arena_axis_extent_mm,
    )

    attrs = {
        "method_version": OMR_METHOD_VERSION,
        "stimulus_direction_source": "static_step_params",
        "detector_estimator_policy": "bout_boundaries_from_detector_physical_metrics_from_positions",
        "position_source_array": "positions_mm",
        "position_anchor": position_anchor,
        "speed_source_array": "speed_smoothed_mm",
        "arena_center_mm": list(arena_center_mm) if arena_center_mm is not None else None,
        "arena_axis_extent_mm": (
            float(arena_axis_extent_mm)
            if arena_axis_extent_mm is not None and np.isfinite(arena_axis_extent_mm)
            else None
        ),
        "arena_geometry_source": arena_geometry_source,
        "arena_position_axis_definition": (
            "dot(position_mm - arena_center_mm, stimulus_direction_xy); "
            "normalized values divide by arena_axis_extent_mm"
        ),
        "projection_deadzone": float(projection_deadzone),
        "projection_speed_deadzone_mm_s": float(projection_speed_deadzone_mm_s),
        "moving_threshold_mm_s": float(moving_threshold_mm_s),
        "window_lengths_s": [float(v) for v in window_lengths_s],
        "early_response_window_lengths_s": [float(v) for v in early_window_lengths_s],
        "weighted_bout_metric_policy": (
            "bout_path_index includes all finite bout path; weighted correct fractions "
            "include only aligned/opposing classifiable bouts"
        ),
        "quality_flag_codes": {
            "0": "ok",
            "1": "no_valid_transitions_or_invalid_bout",
            "2": "no_movement",
        },
    }
    attrs.update(_moving_grating_provenance_attrs(step, grating_dir_deg))
    return OMRStepData(
        per_fish=per_fish,
        per_bout=per_bout,
        windows=windows,
        early_windows=early_windows,
        attrs=attrs,
    )


def _compute_omr_windows(
    tracks: Sequence[DenseTrack],
    step: ProtocolStep,
    direction_xy: np.ndarray,
    fps: float,
    *,
    moving_threshold_mm_s: float,
    projection_speed_deadzone_mm_s: float,
    window_lengths_s: Sequence[float],
    bouts_by_fish: Optional[Dict[int, List[BoutEntry]]] = None,
    arena_center_mm: Optional[Tuple[float, float]] = None,
    arena_axis_extent_mm: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """Compute non-overlapping windowed OMR metrics for a grating step."""

    full_length_s = float(step.duration_s)
    requested_lengths = [
        float(v) for v in window_lengths_s
        if float(v) > 0.0 and (full_length_s <= 0.0 or float(v) < full_length_s)
    ]
    if full_length_s > 0.0 and not any(abs(v - full_length_s) < 1e-6 for v in requested_lengths):
        requested_lengths.append(full_length_s)

    window_id: List[int] = []
    fish_id: List[int] = []
    start_frame: List[int] = []
    end_frame: List[int] = []
    start_time_s: List[float] = []
    end_time_s: List[float] = []
    window_length_s_out: List[float] = []
    omr_path_index: List[float] = []
    time_choice_index: List[float] = []
    coverage_fraction: List[float] = []
    mean_position_axis_norm: List[float] = []
    fraction_time_correct_side: List[float] = []
    n_bouts: List[int] = []
    quality_flag: List[int] = []

    wid = 0
    for window_length_s in requested_lengths:
        window_frames = max(1, int(round(window_length_s * fps))) if fps > 0 else max(1, step.end_frame - step.start_frame)
        cursor = int(step.start_frame)
        while cursor < int(step.end_frame):
            w_start = cursor
            w_end = min(cursor + window_frames, int(step.end_frame))
            actual_len_s = (w_end - w_start) / fps if fps > 0 else 0.0
            for track in tracks:
                summary = _omr_summary_for_window(
                    track,
                    w_start,
                    w_end,
                    direction_xy,
                    fps,
                    moving_threshold_mm_s,
                    projection_speed_deadzone_mm_s,
                    arena_center_mm,
                    arena_axis_extent_mm,
                )
                bouts = []
                if bouts_by_fish is not None:
                    bouts = [
                        b for b in bouts_by_fish.get(track.fish_id, [])
                        if b.start_frame < w_end and b.end_frame >= w_start
                    ]
                window_id.append(wid)
                fish_id.append(track.fish_id)
                start_frame.append(w_start)
                end_frame.append(w_end)
                start_time_s.append((w_start - step.start_frame) / fps if fps > 0 else 0.0)
                end_time_s.append((w_end - step.start_frame) / fps if fps > 0 else 0.0)
                window_length_s_out.append(actual_len_s)
                omr_path_index.append(float(summary["omr_path_index"]))
                time_choice_index.append(float(summary["time_choice_index"]))
                coverage_fraction.append(float(summary["coverage_fraction"]))
                mean_position_axis_norm.append(float(summary["mean_position_axis_norm"]))
                fraction_time_correct_side.append(float(summary["fraction_time_correct_side"]))
                n_bouts.append(len(bouts))
                quality_flag.append(int(summary["quality_flag"]))
            wid += 1
            cursor = w_end

    return {
        "window_id": np.array(window_id, dtype=np.int32),
        "fish_id": np.array(fish_id, dtype=np.int32),
        "start_frame": np.array(start_frame, dtype=np.int64),
        "end_frame": np.array(end_frame, dtype=np.int64),
        "start_time_s": np.array(start_time_s, dtype=np.float32),
        "end_time_s": np.array(end_time_s, dtype=np.float32),
        "window_length_s": np.array(window_length_s_out, dtype=np.float32),
        "omr_path_index": np.array(omr_path_index, dtype=np.float32),
        "time_choice_index": np.array(time_choice_index, dtype=np.float32),
        "coverage_fraction": np.array(coverage_fraction, dtype=np.float32),
        "mean_position_axis_norm": np.array(mean_position_axis_norm, dtype=np.float32),
        "fraction_time_correct_side": np.array(fraction_time_correct_side, dtype=np.float32),
        "n_bouts": np.array(n_bouts, dtype=np.int32),
        "quality_flag": np.array(quality_flag, dtype=np.int8),
    }


def _compute_omr_early_windows(
    tracks: Sequence[DenseTrack],
    step: ProtocolStep,
    direction_xy: np.ndarray,
    fps: float,
    *,
    moving_threshold_mm_s: float,
    projection_deadzone: float,
    projection_speed_deadzone_mm_s: float,
    early_window_lengths_s: Sequence[float],
    bouts_by_fish: Optional[Dict[int, List[BoutEntry]]] = None,
    arena_center_mm: Optional[Tuple[float, float]] = None,
    arena_axis_extent_mm: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """Compute fixed-from-onset early OMR summaries for each grating step."""

    requested_lengths = sorted({float(v) for v in early_window_lengths_s if float(v) > 0.0})

    window_id: List[int] = []
    fish_id: List[int] = []
    start_frame: List[int] = []
    end_frame: List[int] = []
    window_length_s_out: List[float] = []
    actual_window_length_s: List[float] = []
    omr_path_index: List[float] = []
    omr_net_direction_index: List[float] = []
    parallel_displacement_mm: List[float] = []
    net_displacement_mm: List[float] = []
    path_length_mm: List[float] = []
    time_fraction_correct_classified: List[float] = []
    time_choice_index: List[float] = []
    coverage_fraction: List[float] = []
    start_position_axis_norm: List[float] = []
    end_position_axis_norm: List[float] = []
    mean_position_axis_norm: List[float] = []
    fraction_time_correct_side: List[float] = []
    n_bouts: List[int] = []
    n_aligned_bouts: List[int] = []
    n_opposing_bouts: List[int] = []
    n_ambiguous_bouts: List[int] = []
    bout_path_index: List[float] = []
    bout_fraction_correct_weighted_by_path: List[float] = []
    bout_fraction_correct_weighted_by_displacement: List[float] = []
    quality_flag: List[int] = []

    for wid, window_length_s in enumerate(requested_lengths):
        window_frames = (
            max(1, int(math.ceil(window_length_s * fps)))
            if fps > 0 else max(1, int(step.end_frame) - int(step.start_frame))
        )
        w_start = int(step.start_frame)
        w_end = min(w_start + window_frames, int(step.end_frame))
        actual_len_s = (w_end - w_start) / fps if fps > 0 else 0.0

        for track in tracks:
            summary = _omr_summary_for_window(
                track,
                w_start,
                w_end,
                direction_xy,
                fps,
                moving_threshold_mm_s,
                projection_speed_deadzone_mm_s,
                arena_center_mm,
                arena_axis_extent_mm,
            )

            bouts = []
            if bouts_by_fish is not None:
                bouts = [
                    b for b in bouts_by_fish.get(track.fish_id, [])
                    if b.start_frame < w_end and b.end_frame >= w_start
                ]

            labels: List[int] = []
            parallels: List[float] = []
            displacements: List[float] = []
            paths: List[float] = []
            aligned_count = opposing_count = ambiguous_count = 0
            for bout in bouts:
                score, parallel, displacement, path = _bout_omr_score_for_bounds(
                    track,
                    bout,
                    w_start,
                    w_end,
                    direction_xy,
                )
                label, _quality = _bout_omr_label(score, projection_deadzone)
                if label > 0:
                    aligned_count += 1
                elif label < 0:
                    opposing_count += 1
                else:
                    ambiguous_count += 1
                labels.append(label)
                parallels.append(parallel)
                displacements.append(displacement)
                paths.append(path)

            weighted = _weighted_bout_omr_summary(labels, parallels, displacements, paths)

            window_id.append(wid)
            fish_id.append(track.fish_id)
            start_frame.append(w_start)
            end_frame.append(w_end)
            window_length_s_out.append(window_length_s)
            actual_window_length_s.append(actual_len_s)
            omr_path_index.append(float(summary["omr_path_index"]))
            omr_net_direction_index.append(float(summary["omr_net_direction_index"]))
            parallel_displacement_mm.append(float(summary["parallel_displacement_mm"]))
            net_displacement_mm.append(float(summary["net_displacement_mm"]))
            path_length_mm.append(float(summary["path_length_mm"]))
            time_fraction_correct_classified.append(float(summary["time_fraction_correct_classified"]))
            time_choice_index.append(float(summary["time_choice_index"]))
            coverage_fraction.append(float(summary["coverage_fraction"]))
            start_position_axis_norm.append(float(summary["start_position_axis_norm"]))
            end_position_axis_norm.append(float(summary["end_position_axis_norm"]))
            mean_position_axis_norm.append(float(summary["mean_position_axis_norm"]))
            fraction_time_correct_side.append(float(summary["fraction_time_correct_side"]))
            n_bouts.append(len(bouts))
            n_aligned_bouts.append(aligned_count)
            n_opposing_bouts.append(opposing_count)
            n_ambiguous_bouts.append(ambiguous_count)
            bout_path_index.append(float(weighted["bout_path_index"]))
            bout_fraction_correct_weighted_by_path.append(float(weighted["bout_fraction_correct_weighted_by_path"]))
            bout_fraction_correct_weighted_by_displacement.append(
                float(weighted["bout_fraction_correct_weighted_by_displacement"])
            )
            quality_flag.append(int(summary["quality_flag"]))

    return {
        "window_id": np.array(window_id, dtype=np.int32),
        "fish_id": np.array(fish_id, dtype=np.int32),
        "start_frame": np.array(start_frame, dtype=np.int64),
        "end_frame": np.array(end_frame, dtype=np.int64),
        "window_length_s": np.array(window_length_s_out, dtype=np.float32),
        "actual_window_length_s": np.array(actual_window_length_s, dtype=np.float32),
        "omr_path_index": np.array(omr_path_index, dtype=np.float32),
        "omr_net_direction_index": np.array(omr_net_direction_index, dtype=np.float32),
        "parallel_displacement_mm": np.array(parallel_displacement_mm, dtype=np.float32),
        "net_displacement_mm": np.array(net_displacement_mm, dtype=np.float32),
        "path_length_mm": np.array(path_length_mm, dtype=np.float32),
        "time_fraction_correct_classified": np.array(time_fraction_correct_classified, dtype=np.float32),
        "time_choice_index": np.array(time_choice_index, dtype=np.float32),
        "coverage_fraction": np.array(coverage_fraction, dtype=np.float32),
        "start_position_axis_norm": np.array(start_position_axis_norm, dtype=np.float32),
        "end_position_axis_norm": np.array(end_position_axis_norm, dtype=np.float32),
        "mean_position_axis_norm": np.array(mean_position_axis_norm, dtype=np.float32),
        "fraction_time_correct_side": np.array(fraction_time_correct_side, dtype=np.float32),
        "n_bouts": np.array(n_bouts, dtype=np.int32),
        "n_aligned_bouts": np.array(n_aligned_bouts, dtype=np.int32),
        "n_opposing_bouts": np.array(n_opposing_bouts, dtype=np.int32),
        "n_ambiguous_bouts": np.array(n_ambiguous_bouts, dtype=np.int32),
        "bout_path_index": np.array(bout_path_index, dtype=np.float32),
        "bout_fraction_correct_weighted_by_path": np.array(
            bout_fraction_correct_weighted_by_path,
            dtype=np.float32,
        ),
        "bout_fraction_correct_weighted_by_displacement": np.array(
            bout_fraction_correct_weighted_by_displacement,
            dtype=np.float32,
        ),
        "quality_flag": np.array(quality_flag, dtype=np.int8),
    }


def compute_global_omr_metrics(
    fish_ids: Sequence[int],
    step_omr_data: Sequence["OMRStepData"],
) -> Dict[str, np.ndarray]:
    """Aggregate OMR metrics across all eligible moving-grating steps."""

    n_fish = len(fish_ids)
    fish_id_arr = np.array(list(fish_ids), dtype=np.int32)
    eligible_step_count = np.zeros(n_fish, dtype=np.int32)
    eligible_window_count = np.zeros(n_fish, dtype=np.int32)
    omr_path_sum = np.zeros(n_fish, dtype=np.float64)
    omr_path_count = np.zeros(n_fish, dtype=np.int32)
    total_parallel = np.zeros(n_fish, dtype=np.float64)
    total_path = np.zeros(n_fish, dtype=np.float64)
    total_bouts = np.zeros(n_fish, dtype=np.int32)
    total_correct = np.zeros(n_fish, dtype=np.int32)
    total_opposing = np.zeros(n_fish, dtype=np.int32)
    total_ambiguous = np.zeros(n_fish, dtype=np.int32)
    total_bout_parallel = np.zeros(n_fish, dtype=np.float64)
    total_bout_path = np.zeros(n_fish, dtype=np.float64)
    total_bout_displacement = np.zeros(n_fish, dtype=np.float64)
    total_bout_classified_path = np.zeros(n_fish, dtype=np.float64)
    total_bout_classified_displacement = np.zeros(n_fish, dtype=np.float64)
    total_bout_weighted_path_correct_numerator = np.zeros(n_fish, dtype=np.float64)
    total_bout_weighted_displacement_correct_numerator = np.zeros(n_fish, dtype=np.float64)
    total_time_correct = np.zeros(n_fish, dtype=np.float64)
    total_time_opposing = np.zeros(n_fish, dtype=np.float64)
    coverage_sum = np.zeros(n_fish, dtype=np.float64)
    coverage_count = np.zeros(n_fish, dtype=np.int32)
    correct_side_sum = np.zeros(n_fish, dtype=np.float64)
    correct_side_count = np.zeros(n_fish, dtype=np.int32)
    start_axis_norm_sum = np.zeros(n_fish, dtype=np.float64)
    start_axis_norm_count = np.zeros(n_fish, dtype=np.int32)
    end_axis_norm_sum = np.zeros(n_fish, dtype=np.float64)
    end_axis_norm_count = np.zeros(n_fish, dtype=np.int32)
    mean_axis_norm_sum = np.zeros(n_fish, dtype=np.float64)
    mean_axis_norm_count = np.zeros(n_fish, dtype=np.int32)
    min_first_aligned_latency = np.full(n_fish, np.inf, dtype=np.float64)

    fish_to_idx = {int(fid): i for i, fid in enumerate(fish_id_arr)}
    for omr in step_omr_data:
        pf = omr.per_fish
        for row, fid_raw in enumerate(pf["fish_id"]):
            idx = fish_to_idx.get(int(fid_raw))
            if idx is None:
                continue
            eligible_step_count[idx] += 1
            path_index = float(pf["omr_path_index"][row])
            if np.isfinite(path_index):
                omr_path_sum[idx] += path_index
                omr_path_count[idx] += 1
            total_parallel[idx] += float(pf["parallel_displacement_mm"][row])
            total_path[idx] += float(pf["path_length_mm"][row])
            total_bouts[idx] += int(pf["bout_count_total"][row])
            total_correct[idx] += int(pf["bout_count_correct"][row])
            total_opposing[idx] += int(pf["bout_count_opposing"][row])
            total_ambiguous[idx] += int(pf["bout_count_ambiguous"][row])
            bout_parallel = float(pf.get("bout_parallel_displacement_sum_mm", np.zeros_like(pf["fish_id"]))[row])
            bout_path = float(pf.get("bout_path_length_sum_mm", np.zeros_like(pf["fish_id"]))[row])
            bout_displacement = float(pf.get("bout_displacement_sum_mm", np.zeros_like(pf["fish_id"]))[row])
            bout_classified_path = float(
                pf.get("bout_classified_path_length_sum_mm", np.zeros_like(pf["fish_id"]))[row]
            )
            bout_classified_displacement = float(
                pf.get("bout_classified_displacement_sum_mm", np.zeros_like(pf["fish_id"]))[row]
            )
            weighted_path_fraction = float(
                pf.get("bout_fraction_correct_weighted_by_path", np.full_like(pf["fish_id"], np.nan, dtype=np.float32))[row]
            )
            weighted_displacement_fraction = float(
                pf.get(
                    "bout_fraction_correct_weighted_by_displacement",
                    np.full_like(pf["fish_id"], np.nan, dtype=np.float32),
                )[row]
            )
            total_bout_parallel[idx] += bout_parallel
            total_bout_path[idx] += bout_path
            total_bout_displacement[idx] += bout_displacement
            total_bout_classified_path[idx] += bout_classified_path
            total_bout_classified_displacement[idx] += bout_classified_displacement
            if np.isfinite(weighted_path_fraction):
                total_bout_weighted_path_correct_numerator[idx] += (
                    weighted_path_fraction * bout_classified_path
                )
            if np.isfinite(weighted_displacement_fraction):
                total_bout_weighted_displacement_correct_numerator[idx] += (
                    weighted_displacement_fraction * bout_classified_displacement
                )
            total_time_correct[idx] += float(pf["time_correct_s"][row])
            total_time_opposing[idx] += float(pf["time_opposing_s"][row])
            coverage = float(pf["coverage_fraction"][row])
            if np.isfinite(coverage):
                coverage_sum[idx] += coverage
                coverage_count[idx] += 1
            correct_side = float(pf["fraction_time_correct_side"][row])
            if np.isfinite(correct_side):
                correct_side_sum[idx] += correct_side
                correct_side_count[idx] += 1
            start_axis = float(pf["start_position_axis_norm"][row])
            if np.isfinite(start_axis):
                start_axis_norm_sum[idx] += start_axis
                start_axis_norm_count[idx] += 1
            end_axis = float(pf["end_position_axis_norm"][row])
            if np.isfinite(end_axis):
                end_axis_norm_sum[idx] += end_axis
                end_axis_norm_count[idx] += 1
            mean_axis = float(pf["mean_position_axis_norm"][row])
            if np.isfinite(mean_axis):
                mean_axis_norm_sum[idx] += mean_axis
                mean_axis_norm_count[idx] += 1
            aligned_latency = float(pf["first_aligned_bout_latency_s"][row])
            if np.isfinite(aligned_latency):
                min_first_aligned_latency[idx] = min(min_first_aligned_latency[idx], aligned_latency)

        if "fish_id" in omr.windows:
            for fid_raw in omr.windows["fish_id"]:
                idx = fish_to_idx.get(int(fid_raw))
                if idx is not None:
                    eligible_window_count[idx] += 1

    classified_bouts = total_correct + total_opposing
    classified_time = total_time_correct + total_time_opposing

    omr_path_index_mean = np.full(n_fish, np.nan, dtype=np.float32)
    omr_path_index_weighted = np.full(n_fish, np.nan, dtype=np.float32)
    bout_fraction = np.full(n_fish, np.nan, dtype=np.float32)
    bout_choice = np.full(n_fish, np.nan, dtype=np.float32)
    bout_path_index = np.full(n_fish, np.nan, dtype=np.float32)
    bout_fraction_weighted_by_path = np.full(n_fish, np.nan, dtype=np.float32)
    bout_fraction_weighted_by_displacement = np.full(n_fish, np.nan, dtype=np.float32)
    time_choice = np.full(n_fish, np.nan, dtype=np.float32)
    coverage_fraction = np.full(n_fish, np.nan, dtype=np.float32)
    mean_fraction_time_correct_side = np.full(n_fish, np.nan, dtype=np.float32)
    mean_start_position_axis_norm = np.full(n_fish, np.nan, dtype=np.float32)
    mean_end_position_axis_norm = np.full(n_fish, np.nan, dtype=np.float32)
    mean_mean_position_axis_norm = np.full(n_fish, np.nan, dtype=np.float32)
    first_aligned_bout_latency_s_min = np.full(n_fish, np.nan, dtype=np.float32)
    quality_flag = np.zeros(n_fish, dtype=np.int8)

    for i in range(n_fish):
        omr_path_index_mean[i] = _safe_ratio(omr_path_sum[i], float(omr_path_count[i]))
        omr_path_index_weighted[i] = _safe_ratio(total_parallel[i], total_path[i])
        bout_fraction[i] = _safe_ratio(float(total_correct[i]), float(classified_bouts[i]))
        bout_choice[i] = _safe_ratio(float(total_correct[i] - total_opposing[i]), float(classified_bouts[i]))
        bout_path_index[i] = _safe_ratio(total_bout_parallel[i], total_bout_path[i])
        bout_fraction_weighted_by_path[i] = _safe_ratio(
            total_bout_weighted_path_correct_numerator[i],
            total_bout_classified_path[i],
        )
        bout_fraction_weighted_by_displacement[i] = _safe_ratio(
            total_bout_weighted_displacement_correct_numerator[i],
            total_bout_classified_displacement[i],
        )
        time_choice[i] = _safe_ratio(total_time_correct[i] - total_time_opposing[i], classified_time[i])
        coverage_fraction[i] = _safe_ratio(coverage_sum[i], float(coverage_count[i]))
        mean_fraction_time_correct_side[i] = _safe_ratio(
            correct_side_sum[i], float(correct_side_count[i]),
        )
        mean_start_position_axis_norm[i] = _safe_ratio(
            start_axis_norm_sum[i], float(start_axis_norm_count[i]),
        )
        mean_end_position_axis_norm[i] = _safe_ratio(
            end_axis_norm_sum[i], float(end_axis_norm_count[i]),
        )
        mean_mean_position_axis_norm[i] = _safe_ratio(
            mean_axis_norm_sum[i], float(mean_axis_norm_count[i]),
        )
        if np.isfinite(min_first_aligned_latency[i]):
            first_aligned_bout_latency_s_min[i] = min_first_aligned_latency[i]
        if eligible_step_count[i] == 0:
            quality_flag[i] = 1
        elif total_path[i] <= 0.0:
            quality_flag[i] = 2

    return {
        "fish_id": fish_id_arr,
        "eligible_step_count": eligible_step_count,
        "eligible_window_count": eligible_window_count,
        "omr_path_index_mean": omr_path_index_mean,
        "omr_path_index_weighted_by_path": omr_path_index_weighted,
        "bout_fraction_correct_classified": bout_fraction,
        "bout_choice_index": bout_choice,
        "bout_path_index": bout_path_index,
        "bout_fraction_correct_weighted_by_path": bout_fraction_weighted_by_path,
        "bout_fraction_correct_weighted_by_displacement": bout_fraction_weighted_by_displacement,
        "time_choice_index": time_choice,
        "mean_fraction_time_correct_side": mean_fraction_time_correct_side,
        "mean_start_position_axis_norm": mean_start_position_axis_norm,
        "mean_end_position_axis_norm": mean_end_position_axis_norm,
        "mean_mean_position_axis_norm": mean_mean_position_axis_norm,
        "first_aligned_bout_latency_s_min": first_aligned_bout_latency_s_min,
        "total_path_length_mm": total_path.astype(np.float32),
        "total_parallel_displacement_mm": total_parallel.astype(np.float32),
        "total_bouts": total_bouts,
        "total_bout_correct": total_correct,
        "total_bout_opposing": total_opposing,
        "total_bout_ambiguous": total_ambiguous,
        "total_bout_parallel_displacement_mm": total_bout_parallel.astype(np.float32),
        "total_bout_path_length_mm": total_bout_path.astype(np.float32),
        "total_bout_displacement_mm": total_bout_displacement.astype(np.float32),
        "coverage_fraction": coverage_fraction,
        "quality_flag": quality_flag,
    }
