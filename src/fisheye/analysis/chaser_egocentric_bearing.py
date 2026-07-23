"""Compute fish-centric chaser bearing components for chaser-distance runs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
import json
import re
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import zarr  # noqa: E402

from fisheye.analysis.chaser_behavior import resolve_configured_chaser_behaviors
from fisheye.analysis.chaser_distance_runs import (
    ChaserDistanceWindow,
    _artifact_signature,
    _bytes_array,
    _display_epoch_label,
    _write_array,
    write_chaser_dashboard_spec_artifact,
    write_goodcopbadcop_chaser_dashboard_spec_artifact,
)
from fisheye.analysis.chaser_distance_io import (
    ChaserDistanceReadSnapshot,
    load_chaser_distance_run,
    reject_unsealed_chaser_derived_publication,
)
from fisheye.analysis.track_kinematics_io import load_track_kinematics_track
from fisheye.shared.json_safety import decode_null_terminated_text, json_attr_safe
from fisheye.shared.plot_artifacts import write_png_visualization_artifact
from fisheye.shared.system_metadata import get_git_info


SCHEMA_ID = "palette.chaser_egocentric_bearing.v1"
SCHEMA_VERSION = 1
METHOD = "offline_track_heading_to_chaser_egocentric_bearing"
METHOD_VERSION = "1"
COMPONENT_PARENT_NAME = "egocentric_bearing"
PRE_POST_POLAR_PNG_ARTIFACT_NAME = "egocentric_bearing_pre_post_polar_png"
PRE_POST_POLAR_POINT_CLOUD_PNG_ARTIFACT_NAME = "egocentric_bearing_pre_post_polar_point_cloud_png"
PRE_POST_POLAR_VISUALIZATION_CONTRACT_ID = (
    "palette.chaser_egocentric_bearing.pre_post_polar_density.v2"
)
PRE_POST_POLAR_POINT_CLOUD_VISUALIZATION_CONTRACT_ID = (
    "palette.chaser_egocentric_bearing.pre_post_polar_point_cloud.v2"
)
STATIC_VISUALIZATION_RENDERER = "fisheye.analysis.chaser_egocentric_bearing"
STATIC_VISUALIZATION_RENDERER_VERSION = "2"
DEFAULT_HEADING_LEVEL = "smoothed"
REQUIRED_TRACK_SCOPE = "offline"
STATIC_POLAR_DISPLAY_DISTANCE_BIN_WIDTH_MM = 5.0
STATIC_POLAR_DISPLAY_BEARING_BIN_WIDTH_DEG = 30.0
STATIC_POLAR_DISPLAY_COLOR_CMAX_QUANTILE = 0.98
STATIC_CHASER_FALLBACK_COLORS = (
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#9333ea",
    "#ea580c",
    "#0891b2",
)
HEADING_LEVEL_TO_ARRAY = {
    "raw": "heading_degrees",
    "heading_raw": "heading_degrees",
    "smoothed": "smoothed_heading_degrees",
    "heading_smoothed": "smoothed_heading_degrees",
}
ANGLE_CONVENTION = (
    "arena_xy_y_down_positions_converted_to_math_y_up_angles; "
    "heading_degrees_ccw_from_+x; egocentric_bearing_deg=wrap(object_bearing-heading); "
    "0=in_front; positive=anatomical_left"
)


@dataclass(frozen=True)
class ChaserEgocentricBearingResult:
    zarr_path: str
    recording_id: str
    component_name: str
    chaser_distance_run_name: str
    chaser_distance_run_path: str
    source_track_kinematics_run: str
    source_track_kinematics_scope: str
    source_track_kinematics_track_id: int
    source_track_kinematics_track_path: str
    source_heading_array: str
    heading_level: str
    fps: float
    total_frames: int
    pixels_per_mm_projector: float
    chaser_indices: np.ndarray
    chaser_behavior_labels: tuple[str, ...]
    chaser_color_hex: Mapping[int, str]
    camera_frame_id: np.ndarray
    stimulus_epoch_window_id: np.ndarray
    windows: tuple[ChaserDistanceWindow, ...]
    fish_heading_deg: np.ndarray
    fish_heading_valid: np.ndarray
    object_vector_arena_xy: np.ndarray
    distance_mm: np.ndarray
    bearing_deg: np.ndarray
    alignment_cos: np.ndarray
    lateral_sin: np.ndarray
    valid: np.ndarray
    epoch_valid_frame_count: np.ndarray
    epoch_circular_mean_bearing_deg: np.ndarray
    epoch_circular_resultant_length: np.ndarray
    epoch_mean_alignment_cos: np.ndarray
    epoch_mean_lateral_sin: np.ndarray
    epoch_fraction_front_45: np.ndarray
    epoch_fraction_lateral_45: np.ndarray
    epoch_fraction_behind_45: np.ndarray
    distance_bin_edges_mm: np.ndarray
    distance_bin_centers_mm: np.ndarray
    bearing_bin_edges_deg: np.ndarray
    bearing_bin_centers_deg: np.ndarray
    histogram_counts: np.ndarray
    histogram_probability: np.ndarray


def _open_root(zarr_path: Path, *, mode: str) -> zarr.Group:
    return zarr.open_group(str(zarr_path), mode=mode, use_consolidated=False)


def wrap_degrees_signed(values: np.ndarray | float) -> np.ndarray:
    """Wrap degree values into the signed [-180, 180) interval."""

    array = np.asarray(values, dtype=np.float64)
    return ((array + 180.0) % 360.0) - 180.0


def normalize_heading_level(value: str) -> str:
    key = str(value).strip().lower()
    if key not in HEADING_LEVEL_TO_ARRAY:
        expected = ", ".join(sorted(HEADING_LEVEL_TO_ARRAY))
        raise ValueError(f"Unsupported heading level {value!r}; expected one of: {expected}")
    return "smoothed" if key == "heading_smoothed" else "raw" if key == "heading_raw" else key


def _normalize_track_scope(value: str) -> str:
    scope = str(value).strip() or REQUIRED_TRACK_SCOPE
    if scope != REQUIRED_TRACK_SCOPE:
        raise ValueError(
            "Egocentric chaser bearing currently requires an offline track-kinematics run "
            f"under analysis/track_kinematics_runs/{REQUIRED_TRACK_SCOPE}; got scope={scope!r}."
        )
    return scope


def _unit_color_to_hex(red: object, green: object, blue: object) -> Optional[str]:
    try:
        channels = [float(red), float(green), float(blue)]
    except Exception:
        return None
    if not all(np.isfinite(value) for value in channels):
        return None
    values = [int(round(max(0.0, min(1.0, value)) * 255.0)) for value in channels]
    return f"#{values[0]:02x}{values[1]:02x}{values[2]:02x}"


def _chaser_colors_from_protocol_payload(payload: Mapping[str, Any]) -> dict[int, str]:
    steps = payload.get("steps")
    if not isinstance(steps, list):
        return {}
    for step in steps:
        if not isinstance(step, Mapping):
            continue
        parameters = step.get("parameters")
        if not isinstance(parameters, Mapping):
            continue
        chasers = parameters.get("chasers")
        if not isinstance(chasers, list):
            continue
        colors: dict[int, str] = {}
        for index, chaser in enumerate(chasers):
            if not isinstance(chaser, Mapping):
                continue
            color = _unit_color_to_hex(
                chaser.get("color_r"),
                chaser.get("color_g"),
                chaser.get("color_b"),
            )
            if color:
                colors[int(index)] = color
        if colors:
            return colors
    return {}


def _source_stimulus_path_from_run_group(run_group: zarr.Group) -> Optional[str]:
    raw_path = getattr(run_group, "attrs", {}).get("source_stimulus_path")
    if raw_path:
        return str(raw_path).strip().strip("/")
    source_refs = getattr(run_group, "attrs", {}).get("source_refs")
    if isinstance(source_refs, Mapping):
        raw_path = source_refs.get("source_stimulus_path")
        if raw_path:
            return str(raw_path).strip().strip("/")
    return None


def _load_chaser_color_hex(
    root: zarr.Group,
    run_group: zarr.Group,
    chaser_indices: np.ndarray,
) -> dict[int, str]:
    stimulus_path = _source_stimulus_path_from_run_group(run_group)
    if not stimulus_path:
        return {}
    try:
        stimulus_group = root[stimulus_path]
    except Exception:
        return {}
    protocol_json = getattr(stimulus_group, "attrs", {}).get("protocol_json")
    if not protocol_json:
        return {}
    try:
        payload = json.loads(str(protocol_json))
    except Exception:
        return {}
    by_protocol_index = _chaser_colors_from_protocol_payload(payload)
    if not by_protocol_index:
        return {}
    out: dict[int, str] = {}
    for chaser_index in np.asarray(chaser_indices, dtype=np.int64).reshape(-1).tolist():
        color = by_protocol_index.get(int(chaser_index))
        if color:
            out[int(chaser_index)] = color
    return out


def _load_configured_chaser_behavior_labels(
    root: zarr.Group,
    run_group: zarr.Group,
    chaser_indices: np.ndarray,
) -> tuple[str, ...]:
    stimulus_path = _source_stimulus_path_from_run_group(run_group)
    if not stimulus_path:
        return tuple("unknown" for _ in chaser_indices)
    stimulus_group = root.get(stimulus_path)
    if stimulus_group is None:
        return tuple("unknown" for _ in chaser_indices)
    protocol_json = getattr(stimulus_group, "attrs", {}).get("protocol_json")
    if not protocol_json:
        return tuple("unknown" for _ in chaser_indices)
    try:
        payload = json.loads(str(protocol_json))
        configured = resolve_configured_chaser_behaviors(payload)
    except (TypeError, ValueError, json.JSONDecodeError):
        return tuple("unknown" for _ in chaser_indices)
    by_index = {item.chaser_index: item.behavior_class for item in configured}
    return tuple(by_index.get(int(index), "unknown") for index in chaser_indices)


def compute_egocentric_chaser_bearing(
    *,
    fish_arena_xy: np.ndarray,
    chaser_arena_xy: np.ndarray,
    fish_heading_deg: np.ndarray,
    fish_valid: Optional[np.ndarray] = None,
    chaser_valid: Optional[np.ndarray] = None,
    fish_heading_valid: Optional[np.ndarray] = None,
    distance_mm: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return vector, bearing, alignment, lateral component, and validity arrays.

    Arena positions use image-style coordinates with y increasing downward.
    Heading arrays are interpreted as degrees counterclockwise from +x in a
    math-y-up coordinate frame, matching the keypoint/track-kinematics heading
    convention used elsewhere in Palette.
    """

    fish_xy = np.asarray(fish_arena_xy, dtype=np.float64)
    chaser_xy = np.asarray(chaser_arena_xy, dtype=np.float64)
    heading = np.asarray(fish_heading_deg, dtype=np.float64).reshape(-1)
    if fish_xy.ndim != 2 or fish_xy.shape[1] != 2:
        raise ValueError("fish_arena_xy must have shape (frame, xy).")
    if chaser_xy.ndim != 3 or chaser_xy.shape[2] != 2:
        raise ValueError("chaser_arena_xy must have shape (frame, chaser, xy).")

    n = min(fish_xy.shape[0], chaser_xy.shape[0], heading.shape[0])
    n_chasers = int(chaser_xy.shape[1])
    fish_xy = fish_xy[:n]
    chaser_xy = chaser_xy[:n]
    heading = heading[:n]

    if fish_valid is None:
        fish_valid_arr = np.isfinite(fish_xy).all(axis=1)
    else:
        fish_valid_values = np.asarray(fish_valid, dtype=bool).reshape(-1)
        if fish_valid_values.shape[0] < n:
            raise ValueError(f"fish_valid length {fish_valid_values.shape[0]} is shorter than frame count {n}.")
        fish_valid_arr = fish_valid_values[:n]
    if chaser_valid is None:
        chaser_valid_arr = np.isfinite(chaser_xy).all(axis=2)
    else:
        chaser_valid_values = np.asarray(chaser_valid, dtype=bool)
        if chaser_valid_values.ndim != 2 or chaser_valid_values.shape[0] < n or chaser_valid_values.shape[1] < n_chasers:
            raise ValueError(
                "chaser_valid shape does not cover the position arrays "
                f"(chaser_valid={chaser_valid_values.shape}, frames={n}, chasers={n_chasers})."
            )
        chaser_valid_arr = chaser_valid_values[:n, :n_chasers]
    if fish_heading_valid is None:
        heading_valid_arr = np.isfinite(heading)
    else:
        heading_valid_values = np.asarray(fish_heading_valid, dtype=bool).reshape(-1)
        if heading_valid_values.shape[0] < n:
            raise ValueError(
                f"fish_heading_valid length {heading_valid_values.shape[0]} is shorter than frame count {n}."
            )
        heading_valid_arr = heading_valid_values[:n] & np.isfinite(heading)
    if distance_mm is None:
        distance_finite_arr = np.ones((n, n_chasers), dtype=bool)
    else:
        distance_values = np.asarray(distance_mm, dtype=np.float64)
        if distance_values.ndim == 1:
            distance_values = distance_values.reshape(-1, 1)
        if distance_values.ndim != 2:
            raise ValueError("distance_mm must have shape (frame, chaser).")
        if distance_values.shape[0] < n or distance_values.shape[1] < n_chasers:
            raise ValueError(
                "distance_mm shape does not cover the position arrays "
                f"(distance_mm={distance_values.shape}, frames={n}, chasers={n_chasers})."
            )
        distance_finite_arr = np.isfinite(distance_values[:n, :n_chasers])

    vector = chaser_xy - fish_xy[:, None, :]
    vector_finite = np.isfinite(vector).all(axis=2)
    valid = fish_valid_arr[:, None] & chaser_valid_arr & heading_valid_arr[:, None] & vector_finite & distance_finite_arr

    object_bearing_world_deg = np.rad2deg(np.arctan2(-vector[:, :, 1], vector[:, :, 0]))
    bearing = wrap_degrees_signed(object_bearing_world_deg - heading[:, None])
    bearing = np.where(valid, bearing, np.nan).astype(np.float32)

    bearing_rad = np.deg2rad(bearing.astype(np.float64))
    alignment_cos = np.where(valid, np.cos(bearing_rad), np.nan).astype(np.float32)
    lateral_sin = np.where(valid, np.sin(bearing_rad), np.nan).astype(np.float32)
    vector = np.where(valid[:, :, None], vector, np.nan).astype(np.float32)
    return vector, bearing, alignment_cos, lateral_sin, valid


def _resolve_chaser_distance_run(
    root: zarr.Group,
    run_name: str,
) -> tuple[ChaserDistanceReadSnapshot, str, str]:
    snapshot = load_chaser_distance_run(
        root,
        run_name=str(run_name).strip() or "latest",
    )
    return snapshot, snapshot.run_name, snapshot.run_path


def _decode_text_column(data: np.ndarray) -> list[str]:
    values = np.asarray(data)
    if values.ndim == 2 and values.dtype.kind in ("u", "i"):
        return [decode_null_terminated_text(row).strip() for row in values]
    return [decode_null_terminated_text(value).strip() for value in values.reshape(-1)]


def _read_windows(
    distance: ChaserDistanceReadSnapshot,
    *,
    fps: float,
) -> tuple[ChaserDistanceWindow, ...]:
    ids = np.asarray(distance.epoch_window_id, dtype=np.int32).reshape(-1)
    labels = list(distance.epoch_labels)
    starts = np.asarray(distance.epoch_start_frame, dtype=np.int64).reshape(-1)
    ends = np.asarray(distance.epoch_end_frame, dtype=np.int64).reshape(-1)
    n = min(ids.shape[0], len(labels), starts.shape[0], ends.shape[0])
    if n == 0:
        return ()
    safe_fps = float(fps) if np.isfinite(fps) and fps > 0 else 1.0
    return tuple(
        ChaserDistanceWindow(
            window_id=int(ids[i]),
            label=str(labels[i]),
            start_frame=int(starts[i]),
            end_frame=int(ends[i]),
            start_time_s=float(starts[i]) / safe_fps,
            end_time_s=(float(ends[i]) + 1.0) / safe_fps,
            duration_s=max(0.0, (float(ends[i]) - float(starts[i]) + 1.0) / safe_fps),
        )
        for i in range(n)
    )


def _component_name_from_source(
    *,
    track_scope: str,
    track_run: str,
    track_id: int,
    heading_level: str,
) -> str:
    raw = f"track_{track_scope}_{track_run}_id_{int(track_id)}_{heading_level}"
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_")
    return sanitized or "track_heading"


def _dense_heading_from_track(
    root: zarr.Group,
    *,
    total_frames: int,
    track_kinematics_run: str,
    track_scope: str,
    track_id: int,
    heading_level: str,
) -> tuple[str, str, str, np.ndarray, np.ndarray]:
    normalized_level = normalize_heading_level(heading_level)
    resolved_scope = _normalize_track_scope(track_scope)
    array_name = HEADING_LEVEL_TO_ARRAY[normalized_level]
    try:
        track = load_track_kinematics_track(
            root,
            run_name=track_kinematics_run,
            scope=resolved_scope,
            track_id=int(track_id),
            required_speed_levels=(),
        )
    except ValueError as exc:
        raise ValueError(
            "Egocentric chaser bearing requires an offline track-kinematics run before it can be computed. "
            "Create analysis/track_kinematics_runs/offline/<run> for this archive, then rerun this command. "
            f"Resolver error: {exc}"
        ) from exc
    source = track.smoothed_heading_degrees if array_name == "smoothed_heading_degrees" else track.heading_degrees
    if source is None:
        raise ValueError(f"Track heading source array is missing: {track.track_path}/{array_name}")
    source = np.asarray(source, dtype=np.float64).reshape(-1)
    frames = np.asarray(track.frame_indices, dtype=np.int64).reshape(-1)
    n = min(frames.shape[0], source.shape[0])
    frames = frames[:n]
    source = source[:n]
    dense = np.full(int(total_frames), np.nan, dtype=np.float32)
    dense_valid = np.zeros(int(total_frames), dtype=bool)
    sample_valid = None
    if track.sample_valid is not None:
        sample_valid = np.asarray(track.sample_valid, dtype=bool).reshape(-1)
        n = min(n, sample_valid.shape[0])
        frames = frames[:n]
        source = source[:n]
        sample_valid = sample_valid[:n]
    for row_idx, frame in enumerate(frames):
        frame_i = int(frame)
        if frame_i < 0 or frame_i >= int(total_frames):
            continue
        value = float(source[row_idx])
        dense[frame_i] = value
        is_valid = np.isfinite(value)
        if sample_valid is not None:
            is_valid = is_valid and bool(sample_valid[row_idx])
        dense_valid[frame_i] = bool(is_valid)
    return (
        track.run_name,
        track.track_path,
        f"{track.track_path}/{array_name}",
        dense,
        dense_valid,
    )


def _summarize_epochs(
    *,
    bearing_deg: np.ndarray,
    alignment_cos: np.ndarray,
    lateral_sin: np.ndarray,
    valid: np.ndarray,
    windows: Sequence[ChaserDistanceWindow],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_windows = len(windows)
    n_chasers = int(bearing_deg.shape[1])
    counts = np.zeros((n_windows, n_chasers), dtype=np.int64)
    circular_mean = np.full((n_windows, n_chasers), np.nan, dtype=np.float32)
    resultant = np.full((n_windows, n_chasers), np.nan, dtype=np.float32)
    mean_alignment = np.full((n_windows, n_chasers), np.nan, dtype=np.float32)
    mean_lateral = np.full((n_windows, n_chasers), np.nan, dtype=np.float32)
    front = np.full((n_windows, n_chasers), np.nan, dtype=np.float32)
    lateral = np.full((n_windows, n_chasers), np.nan, dtype=np.float32)
    behind = np.full((n_windows, n_chasers), np.nan, dtype=np.float32)
    abs_bearing = np.abs(np.asarray(bearing_deg, dtype=np.float64))

    for w_idx, window in enumerate(windows):
        start = max(0, int(window.start_frame))
        end = min(valid.shape[0] - 1, int(window.end_frame))
        if end < start:
            continue
        for c_idx in range(n_chasers):
            mask = valid[start : end + 1, c_idx] & np.isfinite(bearing_deg[start : end + 1, c_idx])
            angles = np.asarray(bearing_deg[start : end + 1, c_idx][mask], dtype=np.float64)
            if angles.size == 0:
                continue
            counts[w_idx, c_idx] = int(angles.size)
            vector = np.mean(np.exp(1j * np.deg2rad(angles)))
            circular_mean[w_idx, c_idx] = float(wrap_degrees_signed(np.asarray(np.rad2deg(np.angle(vector)))))
            resultant[w_idx, c_idx] = float(np.abs(vector))
            mean_alignment[w_idx, c_idx] = float(np.nanmean(alignment_cos[start : end + 1, c_idx][mask]))
            mean_lateral[w_idx, c_idx] = float(np.nanmean(lateral_sin[start : end + 1, c_idx][mask]))
            abs_values = abs_bearing[start : end + 1, c_idx][mask]
            front[w_idx, c_idx] = float(np.mean(abs_values <= 45.0))
            lateral[w_idx, c_idx] = float(np.mean((abs_values > 45.0) & (abs_values < 135.0)))
            behind[w_idx, c_idx] = float(np.mean(abs_values >= 135.0))
    return counts, circular_mean, resultant, mean_alignment, mean_lateral, front, lateral, behind


def _histogram_bins(
    distance_mm: np.ndarray,
    *,
    distance_bin_width_mm: float,
    bearing_bin_width_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    distance_width = float(distance_bin_width_mm)
    bearing_width = float(bearing_bin_width_deg)
    if not np.isfinite(distance_width) or distance_width <= 0:
        raise ValueError("distance_bin_width_mm must be positive.")
    if not np.isfinite(bearing_width) or bearing_width <= 0 or bearing_width > 180:
        raise ValueError("bearing_bin_width_deg must be in (0, 180].")
    finite_distance = np.asarray(distance_mm, dtype=np.float64)
    finite_distance = finite_distance[np.isfinite(finite_distance)]
    max_distance = float(np.max(finite_distance)) if finite_distance.size else distance_width
    max_edge = max(distance_width, float(np.ceil(max_distance / distance_width) * distance_width))
    distance_edges = np.arange(0.0, max_edge + distance_width * 0.5, distance_width, dtype=np.float32)
    if distance_edges.shape[0] < 2:
        distance_edges = np.asarray([0.0, distance_width], dtype=np.float32)
    bearing_edges = np.arange(-180.0, 180.0 + bearing_width * 0.5, bearing_width, dtype=np.float32)
    if bearing_edges[-1] < 180.0:
        bearing_edges = np.append(bearing_edges, np.float32(180.0))
    distance_centers = ((distance_edges[:-1] + distance_edges[1:]) / 2.0).astype(np.float32)
    bearing_centers = ((bearing_edges[:-1] + bearing_edges[1:]) / 2.0).astype(np.float32)
    return distance_edges, distance_centers, bearing_edges, bearing_centers


def _compute_histograms(
    *,
    distance_mm: np.ndarray,
    bearing_deg: np.ndarray,
    valid: np.ndarray,
    windows: Sequence[ChaserDistanceWindow],
    distance_edges: np.ndarray,
    bearing_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_windows = len(windows)
    n_chasers = int(bearing_deg.shape[1])
    n_distance_bins = int(distance_edges.shape[0] - 1)
    n_bearing_bins = int(bearing_edges.shape[0] - 1)
    counts = np.zeros((n_windows, n_chasers, n_distance_bins, n_bearing_bins), dtype=np.uint32)
    probability = np.zeros((n_windows, n_chasers, n_distance_bins, n_bearing_bins), dtype=np.float32)
    for w_idx, window in enumerate(windows):
        start = max(0, int(window.start_frame))
        end = min(valid.shape[0] - 1, int(window.end_frame))
        if end < start:
            continue
        for c_idx in range(n_chasers):
            mask = (
                valid[start : end + 1, c_idx]
                & np.isfinite(distance_mm[start : end + 1, c_idx])
                & np.isfinite(bearing_deg[start : end + 1, c_idx])
            )
            if not np.any(mask):
                continue
            hist, _distance_edges, _bearing_edges = np.histogram2d(
                np.asarray(distance_mm[start : end + 1, c_idx][mask], dtype=np.float64),
                np.asarray(bearing_deg[start : end + 1, c_idx][mask], dtype=np.float64),
                bins=[distance_edges.astype(np.float64), bearing_edges.astype(np.float64)],
            )
            counts[w_idx, c_idx, :, :] = hist.astype(np.uint32, copy=False)
            total = float(np.sum(hist))
            if total > 0:
                probability[w_idx, c_idx, :, :] = (hist / total).astype(np.float32)
    return counts, probability


def build_chaser_egocentric_bearing_result(
    zarr_path: Path,
    *,
    chaser_distance_run: str = "latest",
    track_kinematics_run: str = "latest",
    track_scope: str = "offline",
    track_id: int = 0,
    heading_level: str = DEFAULT_HEADING_LEVEL,
    component_name: Optional[str] = None,
    distance_bin_width_mm: float = 2.0,
    bearing_bin_width_deg: float = 15.0,
) -> ChaserEgocentricBearingResult:
    root = _open_root(zarr_path, mode="r")
    normalized_track_scope = _normalize_track_scope(track_scope)
    distance, distance_run_name, distance_run_path = _resolve_chaser_distance_run(
        root,
        chaser_distance_run,
    )
    fps = float(distance.fps)
    pixels_per_mm_projector = float(distance.pixels_per_mm_projector)

    camera_frame_id = np.asarray(distance.camera_frame_id, dtype=np.int64)
    stimulus_epoch_window_id = np.asarray(
        distance.stimulus_epoch_window_id,
        dtype=np.int32,
    )
    fish_xy = np.asarray(distance.fish_centroid_arena_xy, dtype=np.float32)
    chaser_xy = np.asarray(distance.chaser_arena_xy, dtype=np.float32)
    fish_valid = np.asarray(distance.fish_valid, dtype=bool)
    chaser_valid = np.asarray(distance.chaser_valid, dtype=bool)
    distance_mm = np.asarray(distance.distance_mm, dtype=np.float32)
    chaser_indices = np.asarray(distance.chaser_index, dtype=np.int16)
    # Behavior roles/colors are not part of bearing mathematics and are not yet
    # sealed.  Preserve identity-only output and use deterministic presentation
    # colors rather than reading mutable protocol semantics.
    chaser_behavior_labels: tuple[str, ...] = ()
    chaser_color_hex = {
        int(index): STATIC_CHASER_FALLBACK_COLORS[column % len(STATIC_CHASER_FALLBACK_COLORS)]
        for column, index in enumerate(chaser_indices)
    }
    total_frames = int(camera_frame_id.shape[0])
    if distance.total_frames != total_frames:
        raise ValueError(
            "Chaser-distance typed frame-axis mismatch: "
            f"authority total_frames={distance.total_frames}, "
            f"frames/camera_frame_id length={total_frames}."
        )
    expected_frame_shapes = {
        "frames/stimulus_epoch_window_id": stimulus_epoch_window_id.shape[:1],
        "positions/fish_centroid_arena_xy": fish_xy.shape[:1],
        "positions/chaser_arena_xy": chaser_xy.shape[:1],
        "positions/fish_valid": fish_valid.shape[:1],
        "positions/chaser_valid": chaser_valid.shape[:1],
        "distances/distance_mm": distance_mm.shape[:1],
    }
    mismatched = {name: shape for name, shape in expected_frame_shapes.items() if shape != (total_frames,)}
    if mismatched:
        raise ValueError(f"Chaser-distance run arrays disagree on camera-frame axis: {mismatched}")
    windows = _read_windows(distance, fps=fps)

    normalized_heading_level = normalize_heading_level(heading_level)
    resolved_track_run, track_path, heading_array_path, heading_dense, heading_valid = _dense_heading_from_track(
        root,
        total_frames=int(total_frames),
        track_kinematics_run=str(track_kinematics_run),
        track_scope=normalized_track_scope,
        track_id=int(track_id),
        heading_level=normalized_heading_level,
    )
    component = component_name or _component_name_from_source(
        track_scope=normalized_track_scope,
        track_run=resolved_track_run,
        track_id=int(track_id),
        heading_level=normalized_heading_level,
    )

    vector, bearing, alignment, lateral, valid = compute_egocentric_chaser_bearing(
        fish_arena_xy=fish_xy,
        chaser_arena_xy=chaser_xy,
        fish_heading_deg=heading_dense,
        fish_valid=fish_valid,
        chaser_valid=chaser_valid,
        fish_heading_valid=heading_valid,
        distance_mm=distance_mm,
    )
    summaries = _summarize_epochs(
        bearing_deg=bearing,
        alignment_cos=alignment,
        lateral_sin=lateral,
        valid=valid,
        windows=windows,
    )
    distance_edges, distance_centers, bearing_edges, bearing_centers = _histogram_bins(
        distance_mm,
        distance_bin_width_mm=float(distance_bin_width_mm),
        bearing_bin_width_deg=float(bearing_bin_width_deg),
    )
    hist_counts, hist_probability = _compute_histograms(
        distance_mm=distance_mm,
        bearing_deg=bearing,
        valid=valid,
        windows=windows,
        distance_edges=distance_edges,
        bearing_edges=bearing_edges,
    )

    return ChaserEgocentricBearingResult(
        zarr_path=str(zarr_path),
        recording_id=distance.recording_id,
        component_name=component,
        chaser_distance_run_name=distance_run_name,
        chaser_distance_run_path=distance_run_path,
        source_track_kinematics_run=resolved_track_run,
        source_track_kinematics_scope=normalized_track_scope,
        source_track_kinematics_track_id=int(track_id),
        source_track_kinematics_track_path=track_path,
        source_heading_array=heading_array_path,
        heading_level=normalized_heading_level,
        fps=fps,
        total_frames=int(total_frames),
        pixels_per_mm_projector=float(pixels_per_mm_projector),
        chaser_indices=chaser_indices,
        chaser_behavior_labels=chaser_behavior_labels,
        chaser_color_hex=chaser_color_hex,
        camera_frame_id=camera_frame_id,
        stimulus_epoch_window_id=stimulus_epoch_window_id,
        windows=windows,
        fish_heading_deg=heading_dense.astype(np.float32),
        fish_heading_valid=heading_valid,
        object_vector_arena_xy=vector,
        distance_mm=distance_mm.astype(np.float32),
        bearing_deg=bearing,
        alignment_cos=alignment,
        lateral_sin=lateral,
        valid=valid,
        epoch_valid_frame_count=summaries[0],
        epoch_circular_mean_bearing_deg=summaries[1],
        epoch_circular_resultant_length=summaries[2],
        epoch_mean_alignment_cos=summaries[3],
        epoch_mean_lateral_sin=summaries[4],
        epoch_fraction_front_45=summaries[5],
        epoch_fraction_lateral_45=summaries[6],
        epoch_fraction_behind_45=summaries[7],
        distance_bin_edges_mm=distance_edges,
        distance_bin_centers_mm=distance_centers,
        bearing_bin_edges_deg=bearing_edges,
        bearing_bin_centers_deg=bearing_centers,
        histogram_counts=hist_counts,
        histogram_probability=hist_probability,
    )


def _pre_post_window_indices(windows: Sequence[ChaserDistanceWindow]) -> tuple[Optional[int], Optional[int]]:
    pre_idx = None
    post_idx = None
    non_training: list[int] = []
    for idx, window in enumerate(windows):
        label = str(window.label).strip().lower()
        if "train" not in label:
            non_training.append(idx)
        if pre_idx is None and "pre" in label:
            pre_idx = idx
        if post_idx is None and "post" in label:
            post_idx = idx
    if pre_idx is None and non_training:
        pre_idx = non_training[0]
    if post_idx is None and len(non_training) > 1:
        post_idx = non_training[-1]
    return pre_idx, post_idx


def _chaser_column_index(result: ChaserEgocentricBearingResult, chaser_index: int) -> Optional[int]:
    matches = np.flatnonzero(np.asarray(result.chaser_indices, dtype=np.int64) == int(chaser_index))
    if matches.size:
        return int(matches[0])
    if 0 <= int(chaser_index) < int(result.chaser_indices.shape[0]):
        return int(chaser_index)
    return None


def _chaser_display_label(
    result: ChaserEgocentricBearingResult,
    chaser_index: int,
    chaser_col_idx: int | None,
) -> str:
    behavior = (
        result.chaser_behavior_labels[chaser_col_idx]
        if chaser_col_idx is not None and chaser_col_idx < len(result.chaser_behavior_labels)
        else "unknown"
    )
    return f"chaser {int(chaser_index)} — {behavior}"


def _bearing_probability_for_panel(
    result: ChaserEgocentricBearingResult,
    *,
    window_idx: int,
    chaser_col_idx: int,
) -> tuple[np.ndarray, int]:
    probability = np.sum(
        np.asarray(result.histogram_probability[window_idx, chaser_col_idx, :, :], dtype=np.float64),
        axis=0,
    )
    sample_count = int(np.sum(result.histogram_counts[window_idx, chaser_col_idx, :, :]))
    if sample_count > 0:
        return probability.astype(np.float64, copy=False), sample_count

    window = result.windows[window_idx]
    start = max(0, int(window.start_frame))
    end = min(int(result.valid.shape[0]) - 1, int(window.end_frame))
    if end < start:
        return np.zeros(result.bearing_bin_centers_deg.shape[0], dtype=np.float64), 0
    mask = (
        result.valid[start : end + 1, chaser_col_idx]
        & np.isfinite(result.bearing_deg[start : end + 1, chaser_col_idx])
    )
    bearings = np.asarray(result.bearing_deg[start : end + 1, chaser_col_idx][mask], dtype=np.float64)
    if bearings.size == 0:
        return np.zeros(result.bearing_bin_centers_deg.shape[0], dtype=np.float64), 0
    counts, _edges = np.histogram(bearings, bins=np.asarray(result.bearing_bin_edges_deg, dtype=np.float64))
    total = int(np.sum(counts))
    if total <= 0:
        return np.zeros(result.bearing_bin_centers_deg.shape[0], dtype=np.float64), 0
    return (counts.astype(np.float64) / float(total)), total


def _positive_quantile(values: np.ndarray, quantile: float) -> float:
    finite_positive = np.asarray(values, dtype=np.float64)
    finite_positive = finite_positive[np.isfinite(finite_positive) & (finite_positive > 0.0)]
    if finite_positive.size == 0:
        return 1.0
    return max(float(np.quantile(finite_positive, float(quantile))), float(np.finfo(np.float64).eps))


def _chaser_plot_color(result: ChaserEgocentricBearingResult, chaser_index: int, fallback_index: int) -> str:
    color = result.chaser_color_hex.get(int(chaser_index))
    if color:
        return str(color)
    return STATIC_CHASER_FALLBACK_COLORS[int(fallback_index) % len(STATIC_CHASER_FALLBACK_COLORS)]


def _bearing_distance_for_panel(
    result: ChaserEgocentricBearingResult,
    *,
    window_idx: int,
    chaser_col_idx: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    window = result.windows[window_idx]
    start = max(0, int(window.start_frame))
    end = min(int(result.valid.shape[0]) - 1, int(window.end_frame))
    if end < start:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64), 0
    bearing = np.asarray(result.bearing_deg[start : end + 1, chaser_col_idx], dtype=np.float64)
    distance = np.asarray(result.distance_mm[start : end + 1, chaser_col_idx], dtype=np.float64)
    mask = result.valid[start : end + 1, chaser_col_idx] & np.isfinite(bearing) & np.isfinite(distance)
    bearing = bearing[mask]
    distance = distance[mask]
    return bearing, distance, int(bearing.shape[0])


def _display_bearing_edges_deg(width_deg: float = STATIC_POLAR_DISPLAY_BEARING_BIN_WIDTH_DEG) -> np.ndarray:
    width = float(width_deg)
    if not np.isfinite(width) or width <= 0.0:
        width = float(STATIC_POLAR_DISPLAY_BEARING_BIN_WIDTH_DEG)
    bin_count = max(1, int(round(360.0 / width)))
    return np.linspace(-180.0, 180.0, bin_count + 1, dtype=np.float64)


def _pre_post_panel_specs(
    result: ChaserEgocentricBearingResult,
) -> tuple[tuple[tuple[str, Optional[int]], ...], tuple[tuple[int, Optional[int]], ...]]:
    pre_idx, post_idx = _pre_post_window_indices(result.windows)
    return (
        (
            ("pre", pre_idx),
            ("post", post_idx),
        ),
        (
            (0, _chaser_column_index(result, 0)),
            (1, _chaser_column_index(result, 1)),
        ),
    )


def _pre_post_radial_limit_mm(
    result: ChaserEgocentricBearingResult,
    *,
    panel_specs: Sequence[tuple[str, Optional[int]]],
    chaser_specs: Sequence[tuple[int, Optional[int]]],
    distance_bin_width_mm: float,
) -> float:
    radial_max = 0.0
    for _phase_label, window_idx in panel_specs:
        if window_idx is None:
            continue
        for _chaser_index, chaser_col_idx in chaser_specs:
            if chaser_col_idx is None:
                continue
            _bearings, distances, sample_count = _bearing_distance_for_panel(
                result,
                window_idx=int(window_idx),
                chaser_col_idx=int(chaser_col_idx),
            )
            if sample_count > 0 and distances.size:
                radial_max = max(radial_max, float(np.nanmax(distances)))
    width = max(float(distance_bin_width_mm), float(np.finfo(np.float64).eps))
    if not np.isfinite(radial_max) or radial_max <= 0.0:
        return width
    return float(np.ceil(radial_max / width) * width)


def _distance_bearing_density_for_panel(
    result: ChaserEgocentricBearingResult,
    *,
    window_idx: int,
    chaser_col_idx: int,
    radial_limit_mm: float,
    distance_bin_width_mm: float = STATIC_POLAR_DISPLAY_DISTANCE_BIN_WIDTH_MM,
    bearing_bin_width_deg: float = STATIC_POLAR_DISPLAY_BEARING_BIN_WIDTH_DEG,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    bearings, distances, sample_count = _bearing_distance_for_panel(
        result,
        window_idx=window_idx,
        chaser_col_idx=chaser_col_idx,
    )
    distance_edges = np.arange(
        0.0,
        max(float(radial_limit_mm), float(distance_bin_width_mm)) + float(distance_bin_width_mm) * 0.5,
        float(distance_bin_width_mm),
        dtype=np.float64,
    )
    if distance_edges.shape[0] < 2:
        distance_edges = np.asarray([0.0, float(distance_bin_width_mm)], dtype=np.float64)
    bearing_edges = _display_bearing_edges_deg(float(bearing_bin_width_deg))
    if sample_count <= 0:
        return (
            np.zeros((bearing_edges.shape[0] - 1, distance_edges.shape[0] - 1), dtype=np.float64),
            bearing_edges,
            distance_edges,
            0,
        )
    clipped_distance = np.clip(distances, distance_edges[0], np.nextafter(distance_edges[-1], distance_edges[0]))
    counts, _bearing_edges, _distance_edges = np.histogram2d(
        bearings,
        clipped_distance,
        bins=(bearing_edges, distance_edges),
    )
    total = float(np.sum(counts))
    probability = counts.astype(np.float64) / total if total > 0.0 else counts.astype(np.float64)
    return probability, bearing_edges, distance_edges, sample_count


def _setup_egocentric_polar_axis(ax: Any, *, radial_limit_mm: float) -> None:
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(1)
    ax.set_thetagrids([0, 90, 180, 270], labels=["front", "left", "behind", "right"])
    ax.set_ylim(0.0, float(radial_limit_mm))
    ax.set_rlabel_position(135)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=8)


def _draw_polar_density(
    ax: Any,
    probability: np.ndarray,
    *,
    bearing_edges: np.ndarray,
    distance_edges: np.ndarray,
    cmap: Any,
    norm: Any,
) -> None:
    bearing_centers_rad = np.deg2rad((bearing_edges[:-1] + bearing_edges[1:]) * 0.5)
    bearing_widths_rad = np.deg2rad(np.diff(bearing_edges)) * 0.98
    distance_starts = distance_edges[:-1]
    distance_widths = np.diff(distance_edges)
    for distance_idx, (distance_start, distance_width) in enumerate(zip(distance_starts, distance_widths)):
        values = probability[:, distance_idx]
        positive = np.isfinite(values) & (values > 0.0)
        if not np.any(positive):
            continue
        ax.bar(
            bearing_centers_rad[positive],
            np.full(int(np.sum(positive)), float(distance_width), dtype=np.float64),
            width=bearing_widths_rad[positive],
            bottom=float(distance_start),
            align="center",
            color=cmap(norm(values[positive])),
            edgecolor="none",
            linewidth=0.0,
        )


def render_egocentric_bearing_pre_post_polar_png(
    result: ChaserEgocentricBearingResult,
    *,
    dpi: int = 150,
) -> bytes:
    """Render pre/post egocentric chaser-bearing distance-density maps.

    Panels are arranged as rows = pre/post and columns = chaser 0/chaser 1.
    The polar zero points upward, matching the egocentric convention that
    bearing 0 means the chaser is in front of the fish; positive bearings are
    counterclockwise and therefore anatomical left.
    """

    panel_specs, chaser_specs = _pre_post_panel_specs(result)
    distance_width = float(STATIC_POLAR_DISPLAY_DISTANCE_BIN_WIDTH_MM)
    bearing_width = float(STATIC_POLAR_DISPLAY_BEARING_BIN_WIDTH_DEG)
    radial_limit = _pre_post_radial_limit_mm(
        result,
        panel_specs=panel_specs,
        chaser_specs=chaser_specs,
        distance_bin_width_mm=distance_width,
    )

    panel_values: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray, int]] = {}
    all_probabilities: list[np.ndarray] = []
    for row_idx, (_phase_label, window_idx) in enumerate(panel_specs):
        if window_idx is None:
            continue
        for col_idx, (_chaser_index, chaser_col_idx) in enumerate(chaser_specs):
            if chaser_col_idx is None:
                continue
            probability, bearing_edges, distance_edges, sample_count = _distance_bearing_density_for_panel(
                result,
                window_idx=int(window_idx),
                chaser_col_idx=int(chaser_col_idx),
                radial_limit_mm=radial_limit,
                distance_bin_width_mm=distance_width,
                bearing_bin_width_deg=bearing_width,
            )
            panel_values[(row_idx, col_idx)] = (probability, bearing_edges, distance_edges, sample_count)
            if sample_count > 0:
                all_probabilities.append(probability.reshape(-1))

    if all_probabilities:
        color_cmax = _positive_quantile(
            np.concatenate(all_probabilities),
            STATIC_POLAR_DISPLAY_COLOR_CMAX_QUANTILE,
        )
    else:
        color_cmax = 1.0
    cmap = plt.get_cmap("viridis")
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=float(color_cmax))

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(9.0, 8.0),
        subplot_kw={"projection": "polar"},
        constrained_layout=True,
    )
    for row_idx, (phase_label, window_idx) in enumerate(panel_specs):
        window_label = result.windows[window_idx].label if window_idx is not None else phase_label
        for col_idx, (chaser_index, chaser_col_idx) in enumerate(chaser_specs):
            ax = axes[row_idx, col_idx]
            _setup_egocentric_polar_axis(ax, radial_limit_mm=radial_limit)
            title_window = _display_epoch_label(str(window_label))
            chaser_color = _chaser_plot_color(result, int(chaser_index), col_idx)
            ax.set_title(
                f"{title_window} - {_chaser_display_label(result, chaser_index, chaser_col_idx)}",
                fontsize=10,
                pad=12,
                color=chaser_color,
            )
            if window_idx is None or chaser_col_idx is None:
                ax.text(0.5, 0.5, "not available", ha="center", va="center", transform=ax.transAxes)
                continue
            probability, bearing_edges, distance_edges, sample_count = panel_values.get(
                (row_idx, col_idx),
                (
                    np.zeros((0, 0), dtype=np.float64),
                    _display_bearing_edges_deg(bearing_width),
                    np.asarray([0.0, distance_width], dtype=np.float64),
                    0,
                ),
            )
            if sample_count <= 0:
                ax.text(0.5, 0.5, "no valid frames", ha="center", va="center", transform=ax.transAxes)
                continue
            _draw_polar_density(
                ax,
                probability,
                bearing_edges=bearing_edges,
                distance_edges=distance_edges,
                cmap=cmap,
                norm=norm,
            )
            ax.text(
                0.02,
                0.02,
                f"n={sample_count:,}",
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=8,
            )

    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(
        sm,
        ax=axes.ravel().tolist(),
        shrink=0.78,
        pad=0.06,
        label="fraction/bin",
    )
    fig.suptitle(
        (
            "Egocentric chaser bearing density, pre/post "
            f"({distance_width:g} mm x {bearing_width:g} deg display bins)\n{result.recording_id}"
        ),
        fontsize=12,
    )
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=int(dpi))
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def _point_cloud_for_panel(
    result: ChaserEgocentricBearingResult,
    *,
    window_idx: int,
    chaser_col_idx: int,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    bearing, distance, sample_count = _bearing_distance_for_panel(
        result,
        window_idx=window_idx,
        chaser_col_idx=chaser_col_idx,
    )
    if sample_count <= 0:
        return bearing, distance, 0
    max_count = max(1, int(max_points))
    if sample_count > max_count:
        step = max(1, int(np.ceil(sample_count / float(max_count))))
        bearing = bearing[::step]
        distance = distance[::step]
    return bearing, distance, sample_count


def render_egocentric_bearing_pre_post_point_cloud_png(
    result: ChaserEgocentricBearingResult,
    *,
    dpi: int = 150,
    max_points_per_panel: int = 5000,
) -> bytes:
    """Render pre/post egocentric chaser-bearing distance point clouds.

    Panels are arranged as rows = pre/post and columns = chaser 0/chaser 1.
    Radius is fish-to-chaser distance in mm. Angle follows the egocentric
    convention used by the interactive marimo point cloud: 0 means in front of
    the fish, positive is anatomical left.
    """

    panel_specs, chaser_specs = _pre_post_panel_specs(result)
    radial_limit = _pre_post_radial_limit_mm(
        result,
        panel_specs=panel_specs,
        chaser_specs=chaser_specs,
        distance_bin_width_mm=float(STATIC_POLAR_DISPLAY_DISTANCE_BIN_WIDTH_MM),
    )

    panel_values: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, int]] = {}
    for row_idx, (_phase_label, window_idx) in enumerate(panel_specs):
        if window_idx is None:
            continue
        for col_idx, (_chaser_index, chaser_col_idx) in enumerate(chaser_specs):
            if chaser_col_idx is None:
                continue
            bearings, distances, sample_count = _point_cloud_for_panel(
                result,
                window_idx=int(window_idx),
                chaser_col_idx=int(chaser_col_idx),
                max_points=int(max_points_per_panel),
            )
            panel_values[(row_idx, col_idx)] = (bearings, distances, sample_count)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(9.0, 8.2),
        subplot_kw={"projection": "polar"},
        constrained_layout=True,
    )
    for row_idx, (phase_label, window_idx) in enumerate(panel_specs):
        window_label = result.windows[window_idx].label if window_idx is not None else phase_label
        for col_idx, (chaser_index, chaser_col_idx) in enumerate(chaser_specs):
            ax = axes[row_idx, col_idx]
            _setup_egocentric_polar_axis(ax, radial_limit_mm=radial_limit)
            title_window = _display_epoch_label(str(window_label))
            color = _chaser_plot_color(result, int(chaser_index), col_idx)
            ax.set_title(
                f"{title_window} - {_chaser_display_label(result, chaser_index, chaser_col_idx)}",
                fontsize=10,
                pad=12,
                color=color,
            )
            if window_idx is None or chaser_col_idx is None:
                ax.text(0.5, 0.5, "not available", ha="center", va="center", transform=ax.transAxes)
                continue
            bearings, distances, sample_count = panel_values.get(
                (row_idx, col_idx),
                (np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64), 0),
            )
            if sample_count <= 0:
                ax.text(0.5, 0.5, "no valid frames", ha="center", va="center", transform=ax.transAxes)
                continue
            ax.scatter(
                np.deg2rad(bearings),
                distances,
                s=4.0,
                c=color,
                alpha=0.18,
                linewidths=0.0,
            )
            rendered_count = int(bearings.shape[0])
            text = f"n={sample_count:,}"
            if rendered_count < sample_count:
                text += f"\nshown={rendered_count:,}"
            ax.text(0.02, 0.02, text, transform=ax.transAxes, ha="left", va="bottom", fontsize=8)

    fig.suptitle(
        f"Egocentric chaser bearing point clouds, pre/post\n{result.recording_id}",
        fontsize=12,
    )
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=int(dpi))
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def write_chaser_egocentric_bearing_component(
    zarr_path: Path,
    result: ChaserEgocentricBearingResult,
    *,
    overwrite: bool = False,
    write_png: bool = True,
    write_interactive_spec: bool = True,
) -> str:
    root = _open_root(zarr_path, mode="a")
    reject_unsealed_chaser_derived_publication(
        root,
        run_name=result.chaser_distance_run_name,
        run_path=result.chaser_distance_run_path,
        relative_path=f"{COMPONENT_PARENT_NAME}/{result.component_name}",
    )
    run_group = root[result.chaser_distance_run_path]
    parent = run_group.require_group(COMPONENT_PARENT_NAME)
    component_name = result.component_name
    if component_name in parent:
        if not overwrite:
            raise ValueError(f"Egocentric-bearing component already exists: {result.chaser_distance_run_path}/{COMPONENT_PARENT_NAME}/{component_name}")
        del parent[component_name]
    component = parent.create_group(component_name)

    frames = component.require_group("frames")
    _write_array(frames, "camera_frame_id", result.camera_frame_id)
    _write_array(frames, "stimulus_epoch_window_id", result.stimulus_epoch_window_id)
    _write_array(frames, "fish_heading_deg", result.fish_heading_deg)
    _write_array(frames, "fish_heading_valid", result.fish_heading_valid)
    frames.attrs.update(
        {
            "row_axis": "camera_frames",
            "fish_heading_deg_convention": "degrees_ccw_from_+x_math_y_up",
            "source_heading_array": result.source_heading_array,
        }
    )

    per_chaser = component.require_group("per_chaser")
    _write_array(per_chaser, "chaser_index", result.chaser_indices)
    _write_array(
        per_chaser,
        "behavior_class_label_bytes",
        _bytes_array(result.chaser_behavior_labels, width=32),
    )
    _write_array(per_chaser, "object_vector_arena_xy", result.object_vector_arena_xy)
    _write_array(per_chaser, "distance_mm", result.distance_mm)
    _write_array(per_chaser, "bearing_deg", result.bearing_deg)
    _write_array(per_chaser, "alignment_cos", result.alignment_cos)
    _write_array(per_chaser, "lateral_sin", result.lateral_sin)
    _write_array(per_chaser, "valid", result.valid)
    per_chaser.attrs.update(
        {
            "row_axis": "camera_frames",
            "column_axis": "chasers",
            "object_vector_coordinate_frame": "arena_relative_canvas_px",
            "angle_convention": ANGLE_CONVENTION,
            "axis_order": {
                "object_vector_arena_xy": ["camera_frame", "chaser", "xy"],
                "distance_mm": ["camera_frame", "chaser"],
                "bearing_deg": ["camera_frame", "chaser"],
                "alignment_cos": ["camera_frame", "chaser"],
                "lateral_sin": ["camera_frame", "chaser"],
                "valid": ["camera_frame", "chaser"],
            },
        }
    )

    epoch_summary = component.require_group("epoch_summary")
    _write_array(epoch_summary, "window_id", np.asarray([w.window_id for w in result.windows], dtype=np.int32))
    _write_array(epoch_summary, "label_bytes", _bytes_array([w.label for w in result.windows]))
    _write_array(epoch_summary, "start_frame", np.asarray([w.start_frame for w in result.windows], dtype=np.int64))
    _write_array(epoch_summary, "end_frame", np.asarray([w.end_frame for w in result.windows], dtype=np.int64))
    _write_array(epoch_summary, "valid_frame_count", result.epoch_valid_frame_count)
    _write_array(epoch_summary, "circular_mean_bearing_deg", result.epoch_circular_mean_bearing_deg)
    _write_array(epoch_summary, "circular_resultant_length", result.epoch_circular_resultant_length)
    _write_array(epoch_summary, "mean_alignment_cos", result.epoch_mean_alignment_cos)
    _write_array(epoch_summary, "mean_lateral_sin", result.epoch_mean_lateral_sin)
    _write_array(epoch_summary, "fraction_front_45", result.epoch_fraction_front_45)
    _write_array(epoch_summary, "fraction_lateral_45", result.epoch_fraction_lateral_45)
    _write_array(epoch_summary, "fraction_behind_45", result.epoch_fraction_behind_45)
    epoch_summary.attrs.update(
        {
            "row_axis": "stimulus_epoch_windows",
            "column_axis": "chasers",
            "front_definition": "abs(bearing_deg) <= 45",
            "lateral_definition": "45 < abs(bearing_deg) < 135",
            "behind_definition": "abs(bearing_deg) >= 135",
        }
    )

    hist = component.require_group("distance_bearing_histogram")
    _write_array(hist, "window_id", np.asarray([w.window_id for w in result.windows], dtype=np.int32))
    _write_array(hist, "chaser_index", result.chaser_indices)
    _write_array(hist, "distance_bin_edges_mm", result.distance_bin_edges_mm)
    _write_array(hist, "distance_bin_centers_mm", result.distance_bin_centers_mm)
    _write_array(hist, "bearing_bin_edges_deg", result.bearing_bin_edges_deg)
    _write_array(hist, "bearing_bin_centers_deg", result.bearing_bin_centers_deg)
    _write_array(hist, "hist_counts", result.histogram_counts)
    _write_array(hist, "hist_probability", result.histogram_probability)
    hist.attrs.update(
        {
            "hist_axis_order": ["window", "chaser", "distance_bin", "bearing_bin"],
            "distance_unit": "mm",
            "bearing_unit": "deg",
            "probability_normalization": "sum(hist_probability) == 1 for non-empty window/chaser pairs",
        }
    )

    git = get_git_info(Path(__file__).resolve().parents[3])
    source_refs = {
        "source_chaser_distance_run": result.chaser_distance_run_name,
        "source_chaser_distance_path": result.chaser_distance_run_path,
        "source_track_kinematics_run": result.source_track_kinematics_run,
        "source_track_kinematics_scope": result.source_track_kinematics_scope,
        "source_track_kinematics_track_id": int(result.source_track_kinematics_track_id),
        "source_track_kinematics_track_path": result.source_track_kinematics_track_path,
        "source_heading_array": result.source_heading_array,
    }
    parameters = {
        "heading_level": result.heading_level,
        "angle_convention": ANGLE_CONVENTION,
        "distance_bin_width_mm": float(result.distance_bin_edges_mm[1] - result.distance_bin_edges_mm[0])
        if result.distance_bin_edges_mm.shape[0] > 1
        else None,
        "bearing_bin_width_deg": float(result.bearing_bin_edges_deg[1] - result.bearing_bin_edges_deg[0])
        if result.bearing_bin_edges_deg.shape[0] > 1
        else None,
        "static_polar_display_distance_bin_width_mm": float(STATIC_POLAR_DISPLAY_DISTANCE_BIN_WIDTH_MM),
        "static_polar_display_bearing_bin_width_deg": float(STATIC_POLAR_DISPLAY_BEARING_BIN_WIDTH_DEG),
        "static_polar_display_color_cmax_quantile": float(STATIC_POLAR_DISPLAY_COLOR_CMAX_QUANTILE),
    }
    summary = {
        "chaser_indices": result.chaser_indices.astype(int).tolist(),
        "chaser_color_hex": {str(key): value for key, value in result.chaser_color_hex.items()},
        "heading_valid_frame_count": int(np.sum(result.fish_heading_valid)),
        "egocentric_valid_frame_count": np.sum(result.valid, axis=0).astype(int).tolist(),
        "epoch_labels": [w.label for w in result.windows],
        "epoch_mean_alignment_cos": result.epoch_mean_alignment_cos.tolist(),
        "epoch_fraction_front_45": result.epoch_fraction_front_45.tolist(),
        "epoch_fraction_lateral_45": result.epoch_fraction_lateral_45.tolist(),
        "epoch_fraction_behind_45": result.epoch_fraction_behind_45.tolist(),
    }
    component.attrs.update(
        json_attr_safe(
            {
                "schema_id": SCHEMA_ID,
                "schema_version": SCHEMA_VERSION,
                "method": METHOD,
                "method_version": METHOD_VERSION,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "component_name": component_name,
                "recording_id": result.recording_id,
                "row_axis": "camera_frames",
                "source_refs": source_refs,
                "parameters": parameters,
                "summary": summary,
                "status": "complete",
                "git_commit": git.get("commit_hash"),
                "git_branch": git.get("branch"),
                "git_dirty": git.get("is_dirty"),
                "provenance": {
                    "stage": "chaser_egocentric_bearing",
                    "created_by": "fisheye.analysis.chaser_egocentric_bearing",
                    "inputs": source_refs,
                    "parameters": parameters,
                },
            }
        )
    )
    parent.attrs["latest"] = component_name
    parent.attrs["latest_complete"] = component_name

    component_path = f"{result.chaser_distance_run_path}/{COMPONENT_PARENT_NAME}/{component_name}"
    if write_png:
        common_source_runs = {
            "chaser_distance_run": result.chaser_distance_run_name,
            "track_kinematics_run": result.source_track_kinematics_run,
            "egocentric_bearing_component": component_name,
        }
        common_extra_attrs = {
            "egocentric_bearing_schema_id": SCHEMA_ID,
            "angle_convention": ANGLE_CONVENTION,
            "summary": json_attr_safe(summary),
        }
        artifact_specs = (
            (
                PRE_POST_POLAR_PNG_ARTIFACT_NAME,
                PRE_POST_POLAR_VISUALIZATION_CONTRACT_ID,
                render_egocentric_bearing_pre_post_polar_png(result),
                (
                    "Pre/post static polar density maps of egocentric chaser bearing and "
                    "fish-to-chaser distance for chaser 0 and chaser 1."
                ),
                {
                    "bearing_deg": f"{component_path}/per_chaser/bearing_deg",
                    "distance_mm": f"{component_path}/per_chaser/distance_mm",
                    "valid": f"{component_path}/per_chaser/valid",
                    "hist_probability": f"{component_path}/distance_bearing_histogram/hist_probability",
                    "hist_counts": f"{component_path}/distance_bearing_histogram/hist_counts",
                    "bearing_bin_edges_deg": f"{component_path}/distance_bearing_histogram/bearing_bin_edges_deg",
                    "epoch_summary": f"{component_path}/epoch_summary",
                },
            ),
            (
                PRE_POST_POLAR_POINT_CLOUD_PNG_ARTIFACT_NAME,
                PRE_POST_POLAR_POINT_CLOUD_VISUALIZATION_CONTRACT_ID,
                render_egocentric_bearing_pre_post_point_cloud_png(result),
                (
                    "Pre/post static polar point clouds of egocentric chaser bearing "
                    "and fish-to-chaser distance for chaser 0 and chaser 1."
                ),
                {
                    "bearing_deg": f"{component_path}/per_chaser/bearing_deg",
                    "distance_mm": f"{component_path}/per_chaser/distance_mm",
                    "valid": f"{component_path}/per_chaser/valid",
                    "epoch_summary": f"{component_path}/epoch_summary",
                },
            ),
        )
        for artifact_name, visualization_contract_id, png_bytes, description, source_paths in artifact_specs:
            write_png_visualization_artifact(
                component,
                artifact_name,
                png_bytes,
                description=description,
                created_by="fisheye.analysis.chaser_egocentric_bearing",
                visualization_contract_id=visualization_contract_id,
                renderer=STATIC_VISUALIZATION_RENDERER,
                renderer_version=STATIC_VISUALIZATION_RENDERER_VERSION,
                role="analysis_distribution",
                artifact_signature=_artifact_signature(
                    {
                        "schema_id": SCHEMA_ID,
                        "artifact_name": artifact_name,
                        "visualization_contract_id": visualization_contract_id,
                        "renderer": STATIC_VISUALIZATION_RENDERER,
                        "renderer_version": STATIC_VISUALIZATION_RENDERER_VERSION,
                        "component_name": component_name,
                        "source_refs": source_refs,
                        "parameters": parameters,
                    }
                ),
                source_paths=source_paths,
                source_runs=common_source_runs,
                parameters=parameters,
                extra_attrs=common_extra_attrs,
                overwrite=True,
            )

    if write_interactive_spec:
        write_chaser_dashboard_spec_artifact(
            root,
            run_group,
            run_name=result.chaser_distance_run_name,
            run_path=result.chaser_distance_run_path,
            source_refs=source_refs,
            parameters=parameters,
            summary=summary,
            overwrite=True,
        )
        # Refresh the historical artifact too so existing readers see the
        # newly written egocentric component during the compatibility window.
        write_goodcopbadcop_chaser_dashboard_spec_artifact(
            root,
            run_group,
            run_name=result.chaser_distance_run_name,
            run_path=result.chaser_distance_run_path,
            source_refs=source_refs,
            parameters=parameters,
            summary=summary,
            overwrite=True,
        )

    return component_path


def _result_payload(result: ChaserEgocentricBearingResult, *, applied_path: Optional[str]) -> dict[str, Any]:
    return {
        "schema_id": SCHEMA_ID,
        "zarr_path": result.zarr_path,
        "recording_id": result.recording_id,
        "component_name": result.component_name,
        "applied_path": applied_path,
        "chaser_distance_run": result.chaser_distance_run_name,
        "source_heading_array": result.source_heading_array,
        "heading_level": result.heading_level,
        "chaser_color_hex": {str(key): value for key, value in result.chaser_color_hex.items()},
        "heading_valid_frame_count": int(np.sum(result.fish_heading_valid)),
        "egocentric_valid_frame_count": np.sum(result.valid, axis=0).astype(int).tolist(),
        "windows": [
            {
                "label": window.label,
                "valid_frame_count": result.epoch_valid_frame_count[i].astype(int).tolist(),
                "mean_alignment_cos": result.epoch_mean_alignment_cos[i].tolist(),
                "fraction_front_45": result.epoch_fraction_front_45[i].tolist(),
                "fraction_lateral_45": result.epoch_fraction_lateral_45[i].tolist(),
                "fraction_behind_45": result.epoch_fraction_behind_45[i].tolist(),
            }
            for i, window in enumerate(result.windows)
        ],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path, help="Analysis zarr archive.")
    parser.add_argument("--chaser-distance-run", default="latest")
    parser.add_argument("--track-kinematics-run", default="latest")
    parser.add_argument(
        "--track-scope",
        default=REQUIRED_TRACK_SCOPE,
        choices=(REQUIRED_TRACK_SCOPE,),
        help="Track-kinematics scope. Egocentric chaser bearing currently requires offline kinematics.",
    )
    parser.add_argument("--track-id", type=int, default=0)
    parser.add_argument("--heading-level", default=DEFAULT_HEADING_LEVEL, choices=sorted(HEADING_LEVEL_TO_ARRAY))
    parser.add_argument("--component-name")
    parser.add_argument("--distance-bin-width-mm", type=float, default=2.0)
    parser.add_argument("--bearing-bin-width-deg", type=float, default=15.0)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-png", action="store_true")
    parser.add_argument("--no-interactive-spec", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = build_chaser_egocentric_bearing_result(
        Path(args.zarr_path),
        chaser_distance_run=str(args.chaser_distance_run),
        track_kinematics_run=str(args.track_kinematics_run),
        track_scope=str(args.track_scope),
        track_id=int(args.track_id),
        heading_level=str(args.heading_level),
        component_name=args.component_name,
        distance_bin_width_mm=float(args.distance_bin_width_mm),
        bearing_bin_width_deg=float(args.bearing_bin_width_deg),
    )
    applied_path = None
    if args.apply:
        applied_path = write_chaser_egocentric_bearing_component(
            Path(args.zarr_path),
            result,
            overwrite=bool(args.overwrite),
            write_png=not bool(args.no_png),
            write_interactive_spec=not bool(args.no_interactive_spec),
        )
    payload = _result_payload(result, applied_path=applied_path)
    if args.json:
        print(json.dumps(json_attr_safe(payload), indent=2, sort_keys=True))
    else:
        print(f"recording_id: {result.recording_id}")
        print(f"chaser_distance_run: {result.chaser_distance_run_name}")
        print(f"source_heading_array: {result.source_heading_array}")
        print(f"heading_valid_frame_count: {int(np.sum(result.fish_heading_valid)):,}")
        print(f"egocentric_valid_frame_count: {np.sum(result.valid, axis=0).astype(int).tolist()}")
        for i, window in enumerate(result.windows):
            alignment = ", ".join(
                "nan" if not np.isfinite(v) else f"{float(v):.3f}"
                for v in result.epoch_mean_alignment_cos[i]
            )
            print(f"  {window.label}: mean_alignment_cos=[{alignment}]")
        if applied_path:
            print(f"wrote: {applied_path}")
        else:
            print("dry_run: pass --apply to write the egocentric-bearing component")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
