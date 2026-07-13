"""Pure builders for stimulus-independent baseline behavior export tables.

The recording Zarr remains authoritative.  These helpers accept already
resolved, lineage-compatible arrays and construct portable recording/track
rows without opening or mutating a Zarr store themselves.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np


METHOD = "baseline_behavior_export"
METHOD_VERSION = "1"
COORDINATE_FRAME = "arena_centered_mm"
COORDINATE_ORIGIN = "arena_center"
TIME_BIN_POLICY = "fixed_width_from_baseline_start_half_open_v1"
SAMPLE_POLICY = "source_frame_modulo_stride_v1"
FULL_SAMPLE_POLICY = "all_source_samples_v1"
BOUNDARY_DISTANCE_METHOD = "circle_radius_minus_center_distance_v1"
WALL_FRACTION_DENOMINATOR = "valid_position_frames"


@dataclass(frozen=True)
class BaselineWindow:
    window_id: int
    label: str
    start_frame: int
    end_frame: int
    start_time_s: float
    end_time_s: float
    duration_s: float


@dataclass(frozen=True)
class BaselineArrays:
    fps: float
    arena_xy_px: np.ndarray
    position_valid: np.ndarray
    arena_center_x_px: float
    arena_center_y_px: float
    arena_radius_px: float
    pixels_per_mm: float
    wall_band_mm: float
    track_frames: np.ndarray
    track_time_s: np.ndarray | None
    speed_mm_s: np.ndarray | None
    frame_path_distance_mm: np.ndarray | None
    heading_deg: np.ndarray | None
    sample_valid: np.ndarray | None
    bout_event_frames: np.ndarray | None


def is_baseline_label(value: object) -> bool:
    """Return whether an epoch label denotes the pre-stimulus baseline."""

    label = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return label in {
        "baseline",
        "pre",
        "pre_event",
        "pre_static",
        "pre_stimulus",
        "prestimulus",
    }


def _finite_or_none(value: object) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _aligned_numeric(values: np.ndarray | None, size: int) -> np.ndarray:
    if values is None:
        return np.full(size, np.nan, dtype=np.float64)
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    if data.shape[0] != size:
        return np.full(size, np.nan, dtype=np.float64)
    return data


def _aligned_bool(values: np.ndarray | None, size: int, *, default: bool) -> np.ndarray:
    if values is None:
        return np.full(size, default, dtype=bool)
    data = np.asarray(values, dtype=bool).reshape(-1)
    if data.shape[0] != size:
        return np.full(size, default, dtype=bool)
    return data


def _validated(inputs: BaselineArrays) -> BaselineArrays:
    fps = float(inputs.fps)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("baseline export requires a positive finite fps")
    xy = np.asarray(inputs.arena_xy_px, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("arena_xy_px must have shape (frames, 2)")
    position_valid = np.asarray(inputs.position_valid, dtype=bool).reshape(-1)
    if position_valid.shape[0] != xy.shape[0]:
        raise ValueError("position_valid must align with arena_xy_px")
    frames = np.asarray(inputs.track_frames, dtype=np.int64).reshape(-1)
    radius = float(inputs.arena_radius_px)
    pixels_per_mm = float(inputs.pixels_per_mm)
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("baseline export requires a positive arena radius")
    if not math.isfinite(pixels_per_mm) or pixels_per_mm <= 0:
        raise ValueError("baseline export requires positive pixels_per_mm")
    return BaselineArrays(
        fps=fps,
        arena_xy_px=xy,
        position_valid=position_valid,
        arena_center_x_px=float(inputs.arena_center_x_px),
        arena_center_y_px=float(inputs.arena_center_y_px),
        arena_radius_px=radius,
        pixels_per_mm=pixels_per_mm,
        wall_band_mm=max(0.0, float(inputs.wall_band_mm)),
        track_frames=frames,
        track_time_s=inputs.track_time_s,
        speed_mm_s=inputs.speed_mm_s,
        frame_path_distance_mm=inputs.frame_path_distance_mm,
        heading_deg=inputs.heading_deg,
        sample_valid=inputs.sample_valid,
        bout_event_frames=inputs.bout_event_frames,
    )


def _position_surfaces(inputs: BaselineArrays) -> dict[str, np.ndarray | float]:
    xy = inputs.arena_xy_px
    relative_px = xy - np.asarray(
        [inputs.arena_center_x_px, inputs.arena_center_y_px], dtype=np.float64
    )
    relative_mm = relative_px / inputs.pixels_per_mm
    center_distance_mm = np.sqrt(np.sum(relative_mm**2, axis=1))
    arena_radius_mm = inputs.arena_radius_px / inputs.pixels_per_mm
    distance_to_boundary_mm = np.maximum(0.0, arena_radius_mm - center_distance_mm)
    finite = np.isfinite(relative_mm).all(axis=1)
    in_arena = center_distance_mm <= arena_radius_mm
    valid = inputs.position_valid & finite & in_arena
    wall = valid & (
        center_distance_mm >= max(0.0, arena_radius_mm - inputs.wall_band_mm)
    )
    return {
        "relative_mm": relative_mm,
        "center_distance_mm": center_distance_mm,
        "arena_radius_mm": arena_radius_mm,
        "distance_to_boundary_mm": distance_to_boundary_mm,
        "valid": valid,
        "wall": wall,
    }


def _finite_summary(values: np.ndarray) -> dict[str, float | int | None]:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    return {
        "count": int(data.size),
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "p95": float(np.percentile(data, 95.0)),
        "max": float(np.max(data)),
    }


def _entropy_features(relative_mm: np.ndarray, radius_mm: float, grid_size: int) -> dict[str, Any]:
    grid_size = int(grid_size)
    if grid_size < 2:
        raise ValueError("spatial_grid_size must be at least 2")
    points = np.asarray(relative_mm, dtype=np.float64)
    if points.size == 0:
        return {
            "spatial_grid_size": grid_size,
            "spatial_valid_sample_count": 0,
            "spatial_visited_cell_count": 0,
            "spatial_entropy_normalized": None,
            "spatial_max_cell_fraction": None,
            "quadrant_entropy_normalized": None,
            "quadrant_max_fraction": None,
        }
    fractions = (points / (2.0 * float(radius_mm))) + 0.5
    indexes = np.floor(fractions * grid_size).astype(np.int64)
    indexes = np.clip(indexes, 0, grid_size - 1)
    flat = indexes[:, 1] * grid_size + indexes[:, 0]
    counts = np.bincount(flat, minlength=grid_size * grid_size).astype(np.float64)
    probabilities = counts[counts > 0] / float(np.sum(counts))
    entropy = -float(np.sum(probabilities * np.log(probabilities))) / math.log(
        grid_size * grid_size
    )

    quadrants = (points[:, 0] >= 0).astype(np.int64) + 2 * (
        points[:, 1] >= 0
    ).astype(np.int64)
    quadrant_counts = np.bincount(quadrants, minlength=4).astype(np.float64)
    quadrant_probabilities = quadrant_counts[quadrant_counts > 0] / float(
        np.sum(quadrant_counts)
    )
    quadrant_entropy = -float(
        np.sum(quadrant_probabilities * np.log(quadrant_probabilities))
    ) / math.log(4.0)
    return {
        "spatial_grid_size": grid_size,
        "spatial_valid_sample_count": int(points.shape[0]),
        "spatial_visited_cell_count": int(np.count_nonzero(counts)),
        "spatial_entropy_normalized": entropy,
        "spatial_max_cell_fraction": float(np.max(probabilities)),
        "quadrant_entropy_normalized": quadrant_entropy,
        "quadrant_max_fraction": float(np.max(quadrant_probabilities)),
    }


def _track_surfaces(inputs: BaselineArrays) -> dict[str, np.ndarray]:
    count = int(inputs.track_frames.shape[0])
    return {
        "frames": inputs.track_frames,
        "time_s": _aligned_numeric(inputs.track_time_s, count),
        "speed": _aligned_numeric(inputs.speed_mm_s, count),
        "path": _aligned_numeric(inputs.frame_path_distance_mm, count),
        "heading": _aligned_numeric(inputs.heading_deg, count),
        "valid": _aligned_bool(inputs.sample_valid, count, default=True),
    }


def _window_frame_bounds(window: BaselineWindow, frame_count: int) -> tuple[int, int]:
    start = max(0, int(window.start_frame))
    end = min(frame_count - 1, int(window.end_frame))
    return start, end


def _bout_count(inputs: BaselineArrays, start: int, end_exclusive: int) -> int:
    if inputs.bout_event_frames is None:
        return 0
    frames = np.asarray(inputs.bout_event_frames, dtype=np.int64).reshape(-1)
    return int(np.count_nonzero((frames >= start) & (frames < end_exclusive)))


def build_summary_metrics(
    inputs: BaselineArrays,
    window: BaselineWindow,
    *,
    spatial_grid_size: int = 12,
    source_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one recording/track baseline summary payload."""

    inputs = _validated(inputs)
    positions = _position_surfaces(inputs)
    tracks = _track_surfaces(inputs)
    start, end = _window_frame_bounds(window, inputs.arena_xy_px.shape[0])
    end_exclusive = max(start, end + 1)
    frame_slice = slice(start, end_exclusive)
    total_frames = max(0, end_exclusive - start)
    valid_positions = np.asarray(positions["valid"], dtype=bool)[frame_slice]
    valid_count = int(np.count_nonzero(valid_positions))
    center_values = np.asarray(positions["center_distance_mm"], dtype=np.float64)[
        frame_slice
    ][valid_positions]
    center_summary = _finite_summary(center_values)
    boundary_summary = _finite_summary(
        np.asarray(positions["distance_to_boundary_mm"], dtype=np.float64)[frame_slice][
            valid_positions
        ]
    )
    wall_count = int(
        np.count_nonzero(np.asarray(positions["wall"], dtype=bool)[frame_slice])
    )

    track_mask = (
        (tracks["frames"] >= start)
        & (tracks["frames"] < end_exclusive)
        & tracks["valid"]
    )
    speed_summary = _finite_summary(tracks["speed"][track_mask])
    path_values = tracks["path"][track_mask]
    finite_path = path_values[np.isfinite(path_values)]
    total_path = float(np.sum(finite_path)) if finite_path.size else None
    radius_mm = float(positions["arena_radius_mm"])
    entropy = _entropy_features(
        np.asarray(positions["relative_mm"], dtype=np.float64)[frame_slice][
            valid_positions
        ],
        radius_mm,
        spatial_grid_size,
    )
    duration_s = max(0.0, float(window.duration_s))
    bouts = _bout_count(inputs, start, end_exclusive)
    source_summary = dict(source_summary or {})
    row = {
        "baseline_method": METHOD,
        "baseline_method_version": METHOD_VERSION,
        "baseline_window_id": int(window.window_id),
        "baseline_window_label": str(window.label),
        "start_frame": start,
        "end_frame": end,
        "start_time_s": float(window.start_time_s),
        "end_time_s": float(window.end_time_s),
        "duration_s": duration_s,
        "total_frame_count": total_frames,
        "valid_frame_count": valid_count,
        "missing_frame_count": max(0, total_frames - valid_count),
        "tracking_dropout_fraction": (
            1.0 - (float(valid_count) / float(total_frames)) if total_frames else None
        ),
        "speed_sample_count": int(speed_summary["count"]),
        "mean_speed_mm_s": speed_summary["mean"],
        "median_speed_mm_s": speed_summary["median"],
        "p95_speed_mm_s": speed_summary["p95"],
        "max_speed_mm_s": speed_summary["max"],
        "total_path_mm": total_path,
        "bout_count": bouts,
        "bout_rate_per_min": bouts / (duration_s / 60.0) if duration_s > 0 else None,
        "arena_radius_mm": radius_mm,
        "wall_band_mm": float(inputs.wall_band_mm),
        "expected_uniform_wall_fraction": 1.0
        - (max(0.0, radius_mm - float(inputs.wall_band_mm)) / radius_mm) ** 2,
        "experimental_area_geometry_type": "circle",
        "boundary_distance_method": BOUNDARY_DISTANCE_METHOD,
        "wall_fraction_denominator": WALL_FRACTION_DENOMINATOR,
        "wall_frame_count": wall_count,
        "wall_fraction": float(wall_count) / float(valid_count) if valid_count else None,
        "mean_distance_from_arena_center_mm": center_summary["mean"],
        "median_distance_from_arena_center_mm": center_summary["median"],
        "p95_distance_from_arena_center_mm": center_summary["p95"],
        "mean_distance_to_arena_boundary_mm": boundary_summary["mean"],
        "median_distance_to_arena_boundary_mm": boundary_summary["median"],
        "p95_distance_to_arena_boundary_mm": boundary_summary["p95"],
        "mean_center_distance_norm": (
            float(center_summary["mean"]) / radius_mm
            if center_summary["mean"] is not None and radius_mm > 0
            else None
        ),
        "median_center_distance_norm": (
            float(center_summary["median"]) / radius_mm
            if center_summary["median"] is not None and radius_mm > 0
            else None
        ),
        "coordinate_frame": COORDINATE_FRAME,
        "coordinate_origin": COORDINATE_ORIGIN,
        "x_axis_direction": "right",
        "y_axis_direction": "down",
    }
    row.update(entropy)
    for name in (
        "median_bout_duration_s",
        "mean_bout_duration_s",
        "median_bout_path_length_mm",
        "mean_bout_path_length_mm",
        "median_abs_bout_net_heading_change_deg",
        "mean_abs_bout_net_heading_change_deg",
        "median_inter_bout_interval_s",
        "mean_inter_bout_interval_s",
    ):
        row[name] = _finite_or_none(source_summary.get(name))
    return row


def _circular_heading(values: np.ndarray) -> tuple[float | None, float | None]:
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return None, None
    radians = np.deg2rad(data)
    x = float(np.mean(np.cos(radians)))
    y = float(np.mean(np.sin(radians)))
    angle = math.degrees(math.atan2(y, x))
    return angle, math.sqrt(x * x + y * y)


def build_time_bin_metrics(
    inputs: BaselineArrays,
    window: BaselineWindow,
    *,
    time_bin_s: float = 5.0,
) -> list[dict[str, Any]]:
    """Build fixed-width temporal summaries across one baseline window."""

    inputs = _validated(inputs)
    width_s = float(time_bin_s)
    if not math.isfinite(width_s) or width_s <= 0:
        raise ValueError("baseline time_bin_s must be positive and finite")
    positions = _position_surfaces(inputs)
    tracks = _track_surfaces(inputs)
    start, end = _window_frame_bounds(window, inputs.arena_xy_px.shape[0])
    end_exclusive = max(start, end + 1)
    duration_s = max(0.0, float(window.duration_s))
    bin_count = max(1, int(math.ceil(duration_s / width_s)))
    rows: list[dict[str, Any]] = []
    for bin_index in range(bin_count):
        relative_start_s = bin_index * width_s
        relative_end_s = min(duration_s, (bin_index + 1) * width_s)
        bin_start = min(
            end_exclusive,
            start + int(math.floor(relative_start_s * inputs.fps + 1e-9)),
        )
        bin_end_exclusive = min(
            end_exclusive,
            start + int(math.ceil(relative_end_s * inputs.fps - 1e-9)),
        )
        if bin_end_exclusive <= bin_start:
            bin_end_exclusive = min(end_exclusive, bin_start + 1)
        frame_slice = slice(bin_start, bin_end_exclusive)
        expected_frames = max(0, bin_end_exclusive - bin_start)
        position_valid = np.asarray(positions["valid"], dtype=bool)[frame_slice]
        valid_count = int(np.count_nonzero(position_valid))
        relative_xy = np.asarray(positions["relative_mm"], dtype=np.float64)[frame_slice][
            position_valid
        ]
        center = np.asarray(positions["center_distance_mm"], dtype=np.float64)[
            frame_slice
        ][position_valid]
        center_summary = _finite_summary(center)
        boundary = np.asarray(positions["distance_to_boundary_mm"], dtype=np.float64)[
            frame_slice
        ][position_valid]
        boundary_summary = _finite_summary(boundary)
        wall_count = int(
            np.count_nonzero(np.asarray(positions["wall"], dtype=bool)[frame_slice])
        )

        track_mask = (
            (tracks["frames"] >= bin_start)
            & (tracks["frames"] < bin_end_exclusive)
            & tracks["valid"]
        )
        speed_summary = _finite_summary(tracks["speed"][track_mask])
        path_values = tracks["path"][track_mask]
        finite_path = path_values[np.isfinite(path_values)]
        heading_mean, heading_resultant = _circular_heading(tracks["heading"][track_mask])
        rows.append(
            {
                "baseline_method": METHOD,
                "baseline_method_version": METHOD_VERSION,
                "baseline_window_id": int(window.window_id),
                "baseline_window_label": str(window.label),
                "time_bin_index": bin_index,
                "relative_start_s": relative_start_s,
                "relative_end_s": relative_end_s,
                "time_bin_duration_s": max(0.0, relative_end_s - relative_start_s),
                "source_start_frame": bin_start,
                "source_end_frame": max(bin_start, bin_end_exclusive - 1),
                "expected_frame_count": expected_frames,
                "valid_position_count": valid_count,
                "valid_position_fraction": (
                    float(valid_count) / float(expected_frames) if expected_frames else None
                ),
                "speed_sample_count": int(speed_summary["count"]),
                "mean_speed_mm_s": speed_summary["mean"],
                "median_speed_mm_s": speed_summary["median"],
                "p95_speed_mm_s": speed_summary["p95"],
                "distance_travelled_mm": (
                    float(np.sum(finite_path)) if finite_path.size else None
                ),
                "mean_center_distance_mm": center_summary["mean"],
                "median_center_distance_mm": center_summary["median"],
                "mean_distance_to_arena_boundary_mm": boundary_summary["mean"],
                "median_distance_to_arena_boundary_mm": boundary_summary["median"],
                "experimental_area_geometry_type": "circle",
                "boundary_distance_method": BOUNDARY_DISTANCE_METHOD,
                "wall_fraction_denominator": WALL_FRACTION_DENOMINATOR,
                "wall_frame_count": wall_count,
                "wall_fraction": (
                    float(wall_count) / float(valid_count) if valid_count else None
                ),
                "representative_position_method": "median_valid_arena_position",
                "representative_x_mm": (
                    float(np.median(relative_xy[:, 0])) if relative_xy.size else None
                ),
                "representative_y_mm": (
                    float(np.median(relative_xy[:, 1])) if relative_xy.size else None
                ),
                "mean_heading_deg": heading_mean,
                "heading_resultant": heading_resultant,
                "bout_count": _bout_count(inputs, bin_start, bin_end_exclusive),
                "coordinate_frame": COORDINATE_FRAME,
                "coordinate_origin": COORDINATE_ORIGIN,
                "x_axis_direction": "right",
                "y_axis_direction": "down",
                "time_bin_policy": TIME_BIN_POLICY,
            }
        )
    return rows


def build_sample_metrics(
    inputs: BaselineArrays,
    window: BaselineWindow,
    *,
    target_sample_rate_hz: float = 10.0,
    full_resolution: bool = False,
) -> list[dict[str, Any]]:
    """Build deterministic long-form kinematic samples for one baseline."""

    inputs = _validated(inputs)
    target_rate = float(target_sample_rate_hz)
    if not full_resolution and (not math.isfinite(target_rate) or target_rate <= 0):
        raise ValueError("baseline target sample rate must be positive and finite")
    stride = 1 if full_resolution else max(1, int(round(inputs.fps / target_rate)))
    nominal_rate = inputs.fps / float(stride)
    policy = FULL_SAMPLE_POLICY if full_resolution else SAMPLE_POLICY
    positions = _position_surfaces(inputs)
    tracks = _track_surfaces(inputs)
    start, end = _window_frame_bounds(window, inputs.arena_xy_px.shape[0])
    end_exclusive = max(start, end + 1)
    selected = np.flatnonzero(
        (tracks["frames"] >= start)
        & (tracks["frames"] < end_exclusive)
        & (((tracks["frames"] - start) % stride) == 0)
    )
    effective_rate = (
        float(selected.size) / float(window.duration_s)
        if float(window.duration_s) > 0
        else 0.0
    )
    rows: list[dict[str, Any]] = []
    relative_mm = np.asarray(positions["relative_mm"], dtype=np.float64)
    center_distance = np.asarray(positions["center_distance_mm"], dtype=np.float64)
    position_valid = np.asarray(positions["valid"], dtype=bool)
    wall = np.asarray(positions["wall"], dtype=bool)
    radius_mm = float(positions["arena_radius_mm"])
    for source_sample_index in selected.tolist():
        frame = int(tracks["frames"][source_sample_index])
        position_ok = 0 <= frame < position_valid.shape[0] and bool(position_valid[frame])
        sample_ok = bool(tracks["valid"][source_sample_index])
        source_time = tracks["time_s"][source_sample_index]
        if not math.isfinite(float(source_time)):
            source_time = float(frame) / inputs.fps
        x_mm = float(relative_mm[frame, 0]) if position_ok else None
        y_mm = float(relative_mm[frame, 1]) if position_ok else None
        rows.append(
            {
                "baseline_method": METHOD,
                "baseline_method_version": METHOD_VERSION,
                "baseline_window_id": int(window.window_id),
                "baseline_window_label": str(window.label),
                "source_sample_index": int(source_sample_index),
                "source_frame": frame,
                "source_time_s": float(source_time),
                "relative_time_s": (float(frame - start) / inputs.fps),
                "x_arena_mm": x_mm,
                "y_arena_mm": y_mm,
                "x_arena_fraction": (
                    (x_mm / (2.0 * radius_mm)) + 0.5 if x_mm is not None else None
                ),
                "y_arena_fraction": (
                    (y_mm / (2.0 * radius_mm)) + 0.5 if y_mm is not None else None
                ),
                "speed_mm_s": _finite_or_none(tracks["speed"][source_sample_index]),
                "heading_deg": _finite_or_none(tracks["heading"][source_sample_index]),
                "frame_path_distance_mm": _finite_or_none(
                    tracks["path"][source_sample_index]
                ),
                "center_distance_mm": (
                    float(center_distance[frame]) if position_ok else None
                ),
                "distance_to_arena_boundary_mm": (
                    float(np.asarray(positions["distance_to_boundary_mm"])[frame])
                    if position_ok
                    else None
                ),
                "wall": bool(wall[frame]) if position_ok else None,
                "experimental_area_geometry_type": "circle",
                "boundary_distance_method": BOUNDARY_DISTANCE_METHOD,
                "position_valid": position_ok,
                "sample_valid": sample_ok,
                "sampling_policy": policy,
                "sampling_stride_frames": stride,
                "requested_sample_rate_hz": None if full_resolution else target_rate,
                "source_sample_rate_hz": float(inputs.fps),
                "nominal_sample_rate_hz": nominal_rate,
                "effective_sample_rate_hz": effective_rate,
                "coordinate_frame": COORDINATE_FRAME,
                "coordinate_origin": COORDINATE_ORIGIN,
                "x_axis_direction": "right",
                "y_axis_direction": "down",
            }
        )
    return rows


__all__ = [
    "BOUNDARY_DISTANCE_METHOD",
    "BaselineArrays",
    "BaselineWindow",
    "COORDINATE_FRAME",
    "COORDINATE_ORIGIN",
    "FULL_SAMPLE_POLICY",
    "METHOD",
    "METHOD_VERSION",
    "SAMPLE_POLICY",
    "WALL_FRACTION_DENOMINATOR",
    "TIME_BIN_POLICY",
    "build_sample_metrics",
    "build_summary_metrics",
    "build_time_bin_metrics",
    "is_baseline_label",
]
