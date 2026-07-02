"""Quick plotting helpers for track kinematics outputs.

Generates speed-vs-time, heading-vs-time, and position heatmaps for a
selected track within an analysis/track_kinematics_runs entry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import zarr
from rich.console import Console

from fisheye.analysis.swim_bout_io import SwimBoutIOError, load_swim_bout_tables
from fisheye.analysis.track_kinematics_io import TrackKinematicsTrackTables, load_track_kinematics_track
from fisheye.shared.plot_artifacts import (
    write_interactive_plot_spec_artifact,
    write_png_visualization_artifact,
)
from fisheye.shared.stage_provenance import build_stage_provenance
from fisheye.utils.system import get_environment_info, get_git_info
from fisheye.utils.zarr_io import open_zarr_root

from .chaser_metrics_loader import load_chaser_metrics


TRACK_KINEMATICS_PLOT_SPEC_SCHEMA_ID = "palette.plot_spec.track_kinematics_summary.v1"
TRACK_KINEMATICS_PLOT_RENDERER = "palette-track-kinematics-summary-v1"
TRACK_KINEMATICS_PNG_PREFIX = "track_kinematics_summary"


@dataclass
class _PersistedMetricsBundle:
    camera_frame_ids: np.ndarray
    has_offline: np.ndarray


def _extract_smoothed_series(
    run_group: zarr.Group,
    unit: str,
    indices: np.ndarray,
) -> Optional[np.ndarray]:
    if indices.size == 0:
        return None

    if unit == "mm":
        candidates = (
            "distance_to_target_smoothed_mm",
            "distance_to_target_interpolated_mm",
        )
    else:
        candidates = (
            "distance_to_target_smoothed_px",
            "distance_to_target_interpolated_px",
        )

    max_idx = int(indices.max()) if indices.size else -1
    for name in candidates:
        if name not in run_group:
            continue
        series = np.asarray(run_group[name], dtype=np.float32)
        if series.ndim == 0 or series.shape[0] <= max_idx:
            continue
        values = series[indices]
        if values.size == 0:
            continue
        return values.astype(np.float64)
    return None


def resolve_track_kinematics_runs(
    root: zarr.Group,
    requested: Optional[str],
    console: Console,
    *,
    include_online: bool,
    include_offline: bool,
) -> List[Tuple[str, zarr.Group]]:
    if "analysis" not in root or "track_kinematics_runs" not in root["analysis"]:
        raise ValueError("No track_kinematics_runs group found under analysis/.")

    parent = root["analysis/track_kinematics_runs"]
    online_parent = parent.get("online")
    offline_parent = parent.get("offline")

    def get_run(group: Optional[zarr.Group], name: str) -> Optional[zarr.Group]:
        if group is None:
            return None
        return group.get(name)

    def iter_runs() -> Iterable[Tuple[str, str, zarr.Group]]:
        if include_online and online_parent is not None:
            for name in online_parent.group_keys():
                yield ("online", name, online_parent[name])
        if include_offline and offline_parent is not None:
            for name in offline_parent.group_keys():
                yield ("offline", name, offline_parent[name])

    def resolve_requested(name: str) -> Optional[Tuple[str, str, zarr.Group]]:
        if "/" in name:
            prefix, run = name.split("/", 1)
            if prefix == "online":
                group = get_run(online_parent, run)
                if group is not None:
                    return ("online", run, group)
            elif prefix == "offline":
                group = get_run(offline_parent, run)
                if group is not None:
                    return ("offline", run, group)
            return None
        # Search both parents for bare run names
        if include_online:
            group = get_run(online_parent, name)
            if group is not None:
                return ("online", name, group)
        if include_offline:
            group = get_run(offline_parent, name)
            if group is not None:
                return ("offline", name, group)
        return None

    runs: List[Tuple[str, zarr.Group]] = []

    if requested:
        resolved = resolve_requested(requested)
        if resolved is None:
            raise ValueError(f"Track kinematics run '{requested}' not found.")
        run_type, name, group = resolved
        console.print(f"Using track kinematics run: [cyan]{run_type}/{name}[/cyan]")
        runs.append((f"{run_type}/{name}", group))
        return runs

    preferred: List[Tuple[str, str, zarr.Group]] = []
    if include_online and online_parent is not None:
        latest_online = online_parent.attrs.get("latest")
        if latest_online and latest_online in online_parent:
            preferred.append(("online", latest_online, online_parent[latest_online]))
    if include_offline and offline_parent is not None:
        latest_offline = offline_parent.attrs.get("latest")
        if latest_offline and latest_offline in offline_parent:
            preferred.append(("offline", latest_offline, offline_parent[latest_offline]))

    # legacy attribute storing path like 'online/run'
    legacy_latest = parent.attrs.get("latest")
    if legacy_latest and isinstance(legacy_latest, str):
        resolved = resolve_requested(legacy_latest)
        if resolved and resolved not in preferred:
            preferred.append(resolved)

    seen = set()
    for run_type, name, group in preferred:
        key = (run_type, name)
        if key in seen:
            continue
        seen.add(key)
        console.print(f"Using track kinematics run: [cyan]{run_type}/{name}[/cyan]")
        runs.append((f"{run_type}/{name}", group))

    if not runs:
        console.print("[yellow]No track kinematics runs resolved via 'latest'; scanning available runs.[/yellow]")
        for run_type, name, group in iter_runs():
            runs.append((f"{run_type}/{name}", group))

    return runs


def resolve_track_kinematics_run(
    root: zarr.Group,
    requested: Optional[str],
    console: Console,
) -> tuple[str, zarr.Group]:
    runs = resolve_track_kinematics_runs(
        root,
        requested,
        console,
        include_online=True,
        include_offline=True,
    )
    if not runs:
        raise ValueError("No track kinematics runs available.")
    return runs[0]


def resolve_track(group: zarr.Group, track_id: Optional[int], console: Console) -> tuple[int, zarr.Group]:
    track_ids = group["track_ids"][:]
    if track_ids.size == 0:
        raise ValueError("Track kinematics run contains no tracks.")

    if track_id is None:
        track_id = int(track_ids[0])
        console.print(f"Using track id: [cyan]{track_id}[/cyan]")
    elif track_id not in track_ids:
        raise ValueError(f"Track id {track_id} not found in run. Available IDs: {track_ids.tolist()}")

    track_group_name = f"id_{int(track_id)}"
    if track_group_name not in group["tracks"]:
        raise ValueError(
            f"Track group '{track_group_name}' missing under track kinematics run."
        )
    return int(track_id), group["tracks"][track_group_name]


def pick_units(
    run_group: zarr.Group,
    track_group: zarr.Group,
    track_tables: Optional[TrackKinematicsTrackTables] = None,
) -> tuple[str, np.ndarray, np.ndarray]:
    pixel_to_mm = run_group.attrs.get("pixel_to_mm")
    positions_mm = (
        track_tables.positions_mm
        if track_tables is not None and track_tables.positions_mm is not None
        else track_group["positions_mm"][:]
    )
    if pixel_to_mm and np.isfinite(positions_mm).any():
        return "mm", positions_mm[:, 0], positions_mm[:, 1]
    positions_px = (
        track_tables.positions_px
        if track_tables is not None and track_tables.positions_px is not None
        else track_group["positions_px"][:]
    )
    return "px", positions_px[:, 0], positions_px[:, 1]


def resolve_swim_bout_spans(
    root: zarr.Group,
    requested: Optional[str],
    console: Console,
    speed_level: str = "smoothed",
) -> Tuple[List[Tuple[float, float]], Optional[str]]:
    requested_level = f"speed_{speed_level}" if not str(speed_level).startswith("speed_") else str(speed_level)
    used_level_fallback = False
    try:
        swim_payload = load_swim_bout_tables(root, run_name=requested or "latest", speed_level=speed_level)
    except SwimBoutIOError as exc:
        if "Speed level" not in str(exc):
            raise ValueError(str(exc)) from exc
        try:
            swim_payload = load_swim_bout_tables(root, run_name=requested or "latest")
        except SwimBoutIOError as fallback_exc:
            raise ValueError(str(fallback_exc)) from fallback_exc
        if swim_payload.signal.speed_level:
            used_level_fallback = True
            console.print(
                f"[yellow]Warning:[/yellow] Speed level '{requested_level}' not found, "
                f"using default '{swim_payload.signal.speed_level}'"
            )

    run_name = swim_payload.run_name
    source_speed_level = swim_payload.signal.speed_level
    if source_speed_level:
        if requested_level != source_speed_level and not used_level_fallback:
            console.print(
                f"[yellow]Warning:[/yellow] Speed level '{requested_level}' not found, "
                f"using default '{source_speed_level}'"
            )
        label_suffix = f" ({source_speed_level})"
    else:
        label_suffix = ""

    bout_array = swim_payload.bouts
    method = swim_payload.candidate.detection_method
    full_label = f"{run_name}{label_suffix} ({method})"

    spans: List[Tuple[float, float]] = []
    if bout_array.size == 0:
        return spans, full_label

    names = bout_array.dtype.names or ()
    if "start_time_s" in names and "end_time_s" in names:
        starts = bout_array["start_time_s"]
        ends = bout_array["end_time_s"]
    elif "start_frame" in names and "end_frame" in names:
        fps: Optional[float] = None
        provenance = swim_payload.run_attrs.get("provenance")
        if isinstance(provenance, dict):
            params = provenance.get("parameters") or {}
            if isinstance(params, dict):
                fps_val = params.get("fps")
                if fps_val is not None:
                    fps = float(fps_val)
        if not fps:
            # Try to get FPS from run-level attrs
            fps = swim_payload.run_attrs.get('fps')
        if not fps:
            raise ValueError(
                f"Swim bout run '{run_name}' lacks time fields and FPS information required to convert frames."
            )
        starts = bout_array["start_frame"] / float(fps)
        ends = bout_array["end_frame"] / float(fps)
    else:
        raise ValueError(f"Swim bout run '{run_name}' lacks usable start/end fields for plotting.")

    for start, stop in zip(starts, ends):
        if not (np.isfinite(start) and np.isfinite(stop)):
            continue
        if stop < start:
            continue
        spans.append((float(start), float(stop)))

    return spans, full_label


def resolve_stimulus_run(root: zarr.Group, run_group: zarr.Group) -> Optional[str]:
    inputs = run_group.attrs.get("inputs")
    if isinstance(inputs, dict):
        for key in ("stimulus_run", "source_stimulus_run"):
            value = inputs.get(key)
            if isinstance(value, str) and value:
                return value
    alt = run_group.attrs.get("stimulus_run")
    if isinstance(alt, str) and alt:
        return alt
    stim_parent = root.get("analysis/stimulus_runs")
    if stim_parent is not None:
        latest = stim_parent.attrs.get("latest")
        if isinstance(latest, str) and latest:
            return latest
    return None


def _resolve_metrics_run(run_group: zarr.Group) -> Optional[str]:
    direct = run_group.attrs.get("source_metrics_run")
    if isinstance(direct, str) and direct:
        return direct

    inputs = run_group.attrs.get("inputs")
    if isinstance(inputs, dict):
        for key in ("source_metrics_run", "metrics_run"):
            value = inputs.get(key)
            if isinstance(value, str) and value:
                return value

    provenance = run_group.attrs.get("provenance")
    if isinstance(provenance, dict):
        provenance_inputs = provenance.get("inputs")
        if isinstance(provenance_inputs, dict):
            for key in ("source_metrics_run", "metrics_run"):
                value = provenance_inputs.get(key)
                if isinstance(value, str) and value:
                    return value
        parameters = provenance.get("parameters")
        if isinstance(parameters, dict):
            for key in ("source_metrics_run", "metrics_run"):
                value = parameters.get(key)
                if isinstance(value, str) and value:
                    return value
    return None


def _resolve_track_times(
    run_group: zarr.Group,
    track_group: zarr.Group,
    frame_indices: np.ndarray,
) -> np.ndarray:
    time_seconds = np.asarray(track_group["time_seconds"][:], dtype=np.float64)
    if time_seconds.shape[0] == frame_indices.shape[0]:
        return time_seconds

    fps = run_group.attrs.get("fps")
    if fps and np.isfinite(fps) and fps > 0:
        return frame_indices.astype(np.float64) / float(fps)

    return np.arange(frame_indices.shape[0], dtype=np.float64)


def _map_distance_to_track(
    bundle,
    run_group: zarr.Group,
    track_group: zarr.Group,
    distance_values: np.ndarray,
    availability: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame_indices = track_group["frame_indices"][:].astype(np.int64, copy=False)
    if frame_indices.size == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    time_seconds = _resolve_track_times(run_group, track_group, frame_indices)
    frame_to_idx = {int(frame): idx for idx, frame in enumerate(bundle.camera_frame_ids)}

    times: List[float] = []
    distances: List[float] = []
    indices: List[int] = []
    for frame, timestamp in zip(frame_indices, time_seconds):
        idx = frame_to_idx.get(int(frame))
        if idx is None:
            continue
        if availability is not None:
            if idx >= availability.shape[0] or not availability[idx]:
                continue
        value = distance_values[idx]
        if not np.isfinite(value):
            continue
        times.append(float(timestamp))
        distances.append(float(value))
        indices.append(int(idx))

    return (
        np.asarray(times, dtype=np.float64),
        np.asarray(distances, dtype=np.float64),
        np.asarray(indices, dtype=np.int64),
    )


def compute_online_chaser_distance(
    zarr_path: Path,
    stim_run: str,
    run_group: zarr.Group,
    track_group: zarr.Group,
    track_id: int,
    console: Console,
) -> Optional[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], str]]:
    metrics_run = _resolve_metrics_run(run_group)
    try:
        bundle = load_chaser_metrics(
            zarr_path,
            stimulus_run=stim_run,
            metrics_run=metrics_run,
            chaser_index=track_id,
        )
    except Exception as exc:  # pragma: no cover - defensive
        console.print(
            f"[yellow]Warning:[/yellow] Unable to load chaser metrics ({exc}); skipping distance plot."
        )
        return None

    online = bundle.online
    distance_mm = online.get("distance_to_target_mm")
    distance_px = online.get("distance_to_target_px")

    unit: Optional[str] = None
    distances: Optional[np.ndarray] = None
    if distance_mm is not None:
        candidate = np.asarray(distance_mm, dtype=np.float64)
        if np.isfinite(candidate).any():
            unit = "mm"
            distances = candidate
    if distances is None and distance_px is not None:
        candidate = np.asarray(distance_px, dtype=np.float64)
        if np.isfinite(candidate).any():
            unit = "px"
            distances = candidate

    if distances is None or unit is None:
        console.print("[yellow]Warning:[/yellow] Online distance metrics unavailable; skipping distance plot.")
        return None

    times, values, _ = _map_distance_to_track(bundle, run_group, track_group, distances)
    if times.size == 0:
        console.print("[yellow]Warning:[/yellow] No overlapping online metrics for this track.")
        return None

    return times, values, None, unit


def compute_offline_chaser_distance(
    zarr_path: Path,
    stim_run: str,
    run_group: zarr.Group,
    track_group: zarr.Group,
    track_id: int,
    console: Console,
) -> Optional[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], str]]:
    distance_px = run_group.get("distance_to_target_px")
    distance_mm = run_group.get("distance_to_target_mm")
    camera_frames = run_group.get("camera_frame_ids")
    availability = run_group.get("has_offline")

    if distance_px is not None or distance_mm is not None:
        if camera_frames is None or availability is None:
            console.print(
                "[yellow]Warning:[/yellow] Offline track kinematics run is missing camera_frame_ids/has_offline; falling back to metrics load."
            )
        else:
            unit: Optional[str] = None
            distances: Optional[np.ndarray] = None
            if distance_mm is not None:
                candidate = np.asarray(distance_mm, dtype=np.float64)
                if np.isfinite(candidate).any():
                    unit = "mm"
                    distances = candidate
            if distances is None and distance_px is not None:
                candidate = np.asarray(distance_px, dtype=np.float64)
                if np.isfinite(candidate).any():
                    unit = "px"
                    distances = candidate

            if distances is not None and unit is not None:
                bundle_like = _PersistedMetricsBundle(
                    camera_frame_ids=np.asarray(camera_frames, dtype=np.int64),
                    has_offline=np.asarray(availability, dtype=bool),
                )
                times, values, indices = _map_distance_to_track(
                    bundle_like,
                    run_group,
                    track_group,
                    distances,
                    availability=bundle_like.has_offline,
                )
                if times.size == 0:
                    console.print("[yellow]Warning:[/yellow] No overlapping offline metrics for this track.")
                    return None

                smoothed = _extract_smoothed_series(run_group, unit, indices)
                console.print("[dim]Using offline metrics embedded in the track kinematics run.[/dim]")
                return times, values, smoothed, unit

    metrics_run = _resolve_metrics_run(run_group)
    try:
        bundle = load_chaser_metrics(
            zarr_path,
            stimulus_run=stim_run,
            metrics_run=metrics_run,
            chaser_index=track_id,
        )
    except Exception as exc:  # pragma: no cover - defensive
        console.print(
            f"[yellow]Warning:[/yellow] Unable to load chaser metrics ({exc}); skipping distance plot."
        )
        return None

    offline = bundle.offline
    has_offline_raw = offline.get("has_offline")
    if has_offline_raw is None:
        console.print(
            "[yellow]Warning:[/yellow] Legacy offline chaser metrics not available; skipping distance plot. "
            "Use chaser_distance_runs plus chaser_egocentric_bearing for current GoodCopBadCop/CRA analyses."
        )
        return None

    has_offline = np.asarray(has_offline_raw, dtype=bool)
    if not has_offline.any():
        console.print(
            "[yellow]Warning:[/yellow] Legacy offline chaser metrics missing for this run; skipping distance plot. "
            "Use chaser_distance_runs plus chaser_egocentric_bearing for current GoodCopBadCop/CRA analyses."
        )
        return None

    distance_mm = offline.get("distance_mm")
    distance_px = offline.get("distance_px")

    unit: Optional[str] = None
    distances: Optional[np.ndarray] = None
    if distance_mm is not None:
        candidate = np.asarray(distance_mm, dtype=np.float64)
        if np.isfinite(candidate).any():
            unit = "mm"
            distances = candidate
    if distances is None and distance_px is not None:
        candidate = np.asarray(distance_px, dtype=np.float64)
        if np.isfinite(candidate).any():
            unit = "px"
            distances = candidate

    if distances is None or unit is None:
        console.print("[yellow]Warning:[/yellow] Offline distance metrics unavailable; skipping distance plot.")
        return None

    times, values, indices = _map_distance_to_track(
        bundle,
        run_group,
        track_group,
        distances,
        availability=has_offline,
    )
    if times.size == 0:
        console.print("[yellow]Warning:[/yellow] No overlapping offline metrics for this track.")
        return None

    metrics_label = bundle.provenance.get("metrics_run")
    if isinstance(metrics_label, str) and metrics_label:
        console.print(f"[dim]Using offline metrics run: {metrics_label}[/dim]")

    smoothed = _extract_smoothed_series(run_group, unit, indices)

    return times, values, smoothed, unit


def _format_processing_metadata(run_group: zarr.Group) -> str:
    """Extract and format processing metadata from run_group attributes."""
    attrs = run_group.attrs

    # Try to get parameters from provenance first, then fall back to top-level attrs
    provenance = attrs.get("provenance", {})
    params = provenance.get("parameters", {}) if isinstance(provenance, dict) else {}

    # Extract key parameters
    fps = params.get("fps") or attrs.get("fps")
    smoothing_seconds = params.get("smoothing_seconds") or attrs.get("smoothing_seconds")
    smoothing_method = params.get("smoothing_method") or attrs.get("smoothing_method", "moving_average")
    savgol_polyorder = params.get("savgol_polyorder") or attrs.get("savgol_polyorder")

    # Hysteresis parameters
    hysteresis_enabled = params.get("hysteresis_enabled")
    if hysteresis_enabled is None:
        hysteresis_enabled = attrs.get("hysteresis_enabled", False)

    hysteresis_high = params.get("hysteresis_high_px") or attrs.get("hysteresis_high_px")
    hysteresis_low = params.get("hysteresis_low_px") or attrs.get("hysteresis_low_px")
    hysteresis_min = params.get("hysteresis_min_frames") or attrs.get("hysteresis_min_frames")

    # Coordinate space
    coord_space = params.get("coordinate_space") or attrs.get("coordinate_space", "unknown")

    # Build metadata text
    lines = []

    if fps is not None:
        lines.append(f"FPS: {fps:.1f}")

    if smoothing_seconds is not None:
        lines.append(f"Smoothing window: {smoothing_seconds:.3f}s")

    if smoothing_method:
        if smoothing_method == "savitzky_golay" and savgol_polyorder is not None:
            lines.append(f"Smoothing: Savitzky-Golay (order {savgol_polyorder})")
        elif smoothing_method == "savitzky_golay":
            lines.append(f"Smoothing: Savitzky-Golay")
        else:
            lines.append(f"Smoothing: Moving average")

    if hysteresis_enabled:
        hyst_parts = []
        if hysteresis_high is not None:
            hyst_parts.append(f"high={hysteresis_high:.1f}px")
        if hysteresis_low is not None:
            hyst_parts.append(f"low={hysteresis_low:.1f}px")
        if hysteresis_min is not None:
            hyst_parts.append(f"min={hysteresis_min} frames")
        if hyst_parts:
            lines.append(f"Hysteresis: {', '.join(hyst_parts)}")
        else:
            lines.append("Hysteresis: enabled")
    else:
        lines.append("Hysteresis: disabled")

    lines.append(f"Coordinate space: {coord_space}")

    return " | ".join(lines)


def _artifact_signature(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _track_artifact_names(track_id: int) -> tuple[str, str]:
    suffix = f"track_{int(track_id)}"
    return (
        f"{TRACK_KINEMATICS_PNG_PREFIX}_{suffix}_png",
        f"{TRACK_KINEMATICS_PNG_PREFIX}_{suffix}_interactive",
    )


def _track_source_paths(run_name: str, track_group: zarr.Group, run_group: zarr.Group, track_id: int) -> dict[str, str]:
    run_path = f"analysis/track_kinematics_runs/{run_name}"
    track_path = f"{run_path}/tracks/id_{int(track_id)}"
    source_paths: dict[str, str] = {
        "run": run_path,
        "track": track_path,
    }
    movement_speed_levels = {
        "raw": "speed_raw",
        "filtered": "speed_filtered",
        "smoothed": "speed_smoothed",
        "averaged": "speed_averaged",
    }
    movement = track_group.get("movement")
    speed_parent = movement.get("speed") if movement is not None else None
    if speed_parent is not None:
        for level_name, flat_prefix in movement_speed_levels.items():
            if level_name not in speed_parent:
                continue
            level_group = speed_parent[level_name]
            movement_path = f"{track_path}/movement/speed/{level_name}"
            for unit in ("px", "mm"):
                if unit in level_group:
                    source_paths[f"{flat_prefix}_{unit}"] = f"{movement_path}/{unit}"
                accel_key = f"acceleration_{unit}"
                if accel_key in level_group:
                    if level_name == "smoothed":
                        source_paths[accel_key] = f"{movement_path}/{accel_key}"
                    source_paths[f"{flat_prefix}_{accel_key}"] = f"{movement_path}/{accel_key}"
                smooth_accel_key = f"smoothed_acceleration_{unit}"
                if smooth_accel_key in level_group:
                    if level_name == "smoothed":
                        source_paths[smooth_accel_key] = f"{movement_path}/{smooth_accel_key}"
                    source_paths[f"{flat_prefix}_{smooth_accel_key}"] = (
                        f"{movement_path}/{smooth_accel_key}"
                    )
                path_key = f"frame_path_distance_{unit}"
                if path_key in level_group:
                    source_paths[f"{flat_prefix}_{path_key}"] = (
                        f"{movement_path}/{path_key}"
                    )
                    source_paths[f"frame_path_distance_{level_name}_{unit}"] = (
                        f"{movement_path}/{path_key}"
                    )

    track_arrays = (
        "time_seconds",
        "frame_indices",
        "positions_px",
        "positions_mm",
        "speed_raw_px",
        "speed_raw_mm",
        "speed_filtered_px",
        "speed_filtered_mm",
        "speed_smoothed_px",
        "speed_smoothed_mm",
        "speed_averaged_px",
        "speed_averaged_mm",
        "acceleration_px",
        "acceleration_mm",
        "smoothed_heading_degrees",
        "smoothed_acceleration_px",
        "smoothed_acceleration_mm",
        "delta_heading_degrees",
        "angular_velocity_deg_s",
        "angular_velocity_raw_deg_s",
        "angular_speed_raw_deg_s",
        "delta_heading_smoothed_degrees",
        "angular_velocity_smoothed_deg_s",
        "angular_speed_smoothed_deg_s",
        "cumulative_path_distance_px",
        "cumulative_path_distance_mm",
        "sample_valid",
        "sample_reason_code",
        "transition_valid",
        "transition_reason_code",
    )
    for name in track_arrays:
        if name in track_group and name not in source_paths:
            source_paths[name] = f"{track_path}/{name}"

    run_arrays = (
        "distance_to_target_px",
        "distance_to_target_mm",
        "distance_to_target_smoothed_px",
        "distance_to_target_smoothed_mm",
        "distance_to_target_interpolated_px",
        "distance_to_target_interpolated_mm",
        "camera_frame_ids",
        "has_offline",
    )
    for name in run_arrays:
        if name in run_group:
            source_paths[name] = f"{run_path}/{name}"
    return source_paths


def _series_spec(label: str, path_key: str, source_paths: dict[str, str], *, color: str) -> Optional[dict[str, str]]:
    if path_key not in source_paths:
        return None
    return {"label": label, "path_key": path_key, "color": color}


def _compact_series(items: Iterable[Optional[dict[str, str]]]) -> list[dict[str, str]]:
    return [item for item in items if item is not None]


def _build_track_interactive_spec(
    *,
    run_name: str,
    run_group: zarr.Group,
    track_group: zarr.Group,
    track_id: int,
    source_paths: dict[str, str],
    bins: int,
    swim_bout_label: Optional[str],
    swim_bout_requested: Optional[str],
    speed_level: str,
    distance_series_present: bool,
    stimulus_run: Optional[str],
) -> dict[str, Any]:
    unit_label, _pos_x, _pos_y = pick_units(run_group, track_group)
    speed_unit_suffix = "_mm" if unit_label == "mm" else "_px"
    accel_path = "smoothed_acceleration_mm" if unit_label == "mm" else "smoothed_acceleration_px"
    cumulative_path = "cumulative_path_distance_mm" if unit_label == "mm" else "cumulative_path_distance_px"
    position_path = "positions_mm" if unit_label == "mm" else "positions_px"

    speed_series = _compact_series(
        [
            _series_spec("Raw", f"speed_raw{speed_unit_suffix}", source_paths, color="tab:gray"),
            _series_spec("Filtered", f"speed_filtered{speed_unit_suffix}", source_paths, color="tab:green"),
            _series_spec("Smoothed", f"speed_smoothed{speed_unit_suffix}", source_paths, color="tab:blue"),
            _series_spec("Averaged", f"speed_averaged{speed_unit_suffix}", source_paths, color="tab:purple"),
        ]
    )
    panels: list[dict[str, Any]] = [
        {
            "id": "speed",
            "kind": "timeseries",
            "x_path_key": "time_seconds",
            "y_label": f"Speed ({unit_label}/s)",
            "series": speed_series,
            "interval_overlay": "swim_bouts" if swim_bout_label else None,
        },
        {
            "id": "acceleration",
            "kind": "timeseries",
            "x_path_key": "time_seconds",
            "y_label": f"Acceleration ({unit_label}/s^2)",
            "series": _compact_series(
                [_series_spec("Smoothed acceleration", accel_path, source_paths, color="tab:red")]
            ),
        },
        {
            "id": "turning",
            "kind": "timeseries",
            "x_path_key": "time_seconds",
            "y_label": "Turning (deg/s)",
            "series": _compact_series(
                [
                    _series_spec(
                        "Smoothed angular speed",
                        "angular_speed_smoothed_deg_s",
                        source_paths,
                        color="tab:cyan",
                    ),
                    _series_spec(
                        "Smoothed angular velocity",
                        "angular_velocity_smoothed_deg_s",
                        source_paths,
                        color="tab:blue",
                    ),
                    _series_spec(
                        "Raw angular speed",
                        "angular_speed_raw_deg_s",
                        source_paths,
                        color="tab:gray",
                    ),
                ]
            ),
            "interval_overlay": "swim_bouts" if swim_bout_label else None,
        },
        {
            "id": "heading",
            "kind": "timeseries",
            "x_path_key": "time_seconds",
            "y_label": "Heading (deg)",
            "series": _compact_series(
                [_series_spec("Smoothed heading", "smoothed_heading_degrees", source_paths, color="tab:orange")]
            ),
        },
        {
            "id": "cumulative_path_distance",
            "kind": "timeseries",
            "x_path_key": "time_seconds",
            "y_label": f"Cumulative path distance ({unit_label})",
            "series": _compact_series(
                [_series_spec("Cumulative path distance", cumulative_path, source_paths, color="tab:green")]
            ),
        },
        {
            "id": "position_density",
            "kind": "hist2d",
            "position_path_key": position_path,
            "unit": unit_label,
            "bins": int(bins),
        },
    ]
    if distance_series_present:
        distance_series = _compact_series(
            [
                _series_spec("Distance to target", f"distance_to_target{speed_unit_suffix}", source_paths, color="tab:purple"),
                _series_spec(
                    "Smoothed distance to target",
                    f"distance_to_target_smoothed{speed_unit_suffix}",
                    source_paths,
                    color="tab:pink",
                ),
                _series_spec(
                    "Interpolated distance to target",
                    f"distance_to_target_interpolated{speed_unit_suffix}",
                    source_paths,
                    color="tab:pink",
                ),
            ]
        )
        panels.insert(
            2,
            {
                "id": "distance_to_target",
                "kind": "timeseries",
                "x_path_key": "time_seconds",
                "y_label": f"Distance to target ({unit_label})",
                "series": distance_series,
            },
        )

    return {
        "schema_id": TRACK_KINEMATICS_PLOT_SPEC_SCHEMA_ID,
        "title": f"Movement summary - track {int(track_id)}",
        "run_name": run_name,
        "track_id": int(track_id),
        "renderer": TRACK_KINEMATICS_PLOT_RENDERER,
        "source_paths": source_paths,
        "panels": panels,
        "overlays": {
            "swim_bouts": {
                "enabled": bool(swim_bout_label),
                "requested_run": swim_bout_requested,
                "resolved_label": swim_bout_label,
                "speed_level": speed_level,
            }
        },
        "stimulus_run": stimulus_run,
    }


def write_track_kinematics_plot_artifacts(
    *,
    zarr_path: Optional[Path],
    run_group: zarr.Group,
    run_name: str,
    track_group: zarr.Group,
    track_id: int,
    png_bytes: bytes,
    bins: int,
    artifact_dpi: int,
    swim_bout_label: Optional[str],
    swim_bout_requested: Optional[str],
    speed_level: str,
    distance_series_present: bool,
    stimulus_run: Optional[str],
    console: Console,
    command: Optional[str] = None,
) -> None:
    png_artifact_name, spec_artifact_name = _track_artifact_names(track_id)
    source_paths = _track_source_paths(run_name, track_group, run_group, track_id)
    source_runs: dict[str, Any] = {"track_kinematics": run_name}
    if stimulus_run:
        source_runs["stimulus"] = stimulus_run
    if swim_bout_label:
        source_runs["swim_bout"] = swim_bout_label
    parameters = {
        "bins": int(bins),
        "artifact_dpi": int(artifact_dpi),
        "swim_bout_run": swim_bout_requested,
        "speed_level": speed_level,
        "distance_overlay": bool(distance_series_present),
    }
    signature = _artifact_signature(
        {
            "schema_id": TRACK_KINEMATICS_PLOT_SPEC_SCHEMA_ID,
            "run_name": run_name,
            "track_id": int(track_id),
            "source_paths": source_paths,
            "source_runs": source_runs,
            "parameters": parameters,
        }
    )
    created_at_utc = datetime.now(timezone.utc).isoformat()
    env_info = get_environment_info(
        disk_path=str(zarr_path) if zarr_path is not None else None,
        capture_env_vars=False,
    )
    provenance = build_stage_provenance(
        stage="track_kinematics_visualization",
        created_at_utc=created_at_utc,
        parameters={
            **parameters,
            "plot_schema_id": TRACK_KINEMATICS_PLOT_SPEC_SCHEMA_ID,
            "renderer": TRACK_KINEMATICS_PLOT_RENDERER,
        },
        inputs={
            "zarr_path": str(zarr_path) if zarr_path is not None else None,
            "source_paths": source_paths,
            "source_runs": source_runs,
            "track_id": int(track_id),
            "run_name": run_name,
        },
        command=command,
        version=TRACK_KINEMATICS_PLOT_RENDERER,
        git=get_git_info(),
        environment=env_info.get("environment"),
        platform=env_info.get("platform"),
        artifacts={
            "png_artifact": f"visualizations/{png_artifact_name}",
            "interactive_artifact": f"visualizations/{spec_artifact_name}",
            "artifact_signature": signature,
        },
    )

    write_png_visualization_artifact(
        run_group,
        png_artifact_name,
        png_bytes,
        description="Track kinematics summary PNG",
        created_by="fisheye.analysis.plot_track_kinematics",
        artifact_signature=signature,
        created_at_utc=created_at_utc,
        source_paths=source_paths,
        source_runs=source_runs,
        parameters=parameters,
        extra_attrs={
            "plot_schema_id": TRACK_KINEMATICS_PLOT_SPEC_SCHEMA_ID,
            "track_id": int(track_id),
            "run_name": run_name,
            "provenance": provenance,
        },
    )
    spec = _build_track_interactive_spec(
        run_name=run_name,
        run_group=run_group,
        track_group=track_group,
        track_id=track_id,
        source_paths=source_paths,
        bins=bins,
        swim_bout_label=swim_bout_label,
        swim_bout_requested=swim_bout_requested,
        speed_level=speed_level,
        distance_series_present=distance_series_present,
        stimulus_run=stimulus_run,
    )
    write_interactive_plot_spec_artifact(
        run_group,
        spec_artifact_name,
        spec,
        description="Track kinematics interactive plot spec",
        created_by="fisheye.analysis.plot_track_kinematics",
        renderer=TRACK_KINEMATICS_PLOT_RENDERER,
        artifact_signature=signature,
        created_at_utc=created_at_utc,
        snapshot_artifact=png_artifact_name,
        source_paths=source_paths,
        source_runs=source_runs,
        parameters=parameters,
        extra_attrs={
            "plot_schema_id": TRACK_KINEMATICS_PLOT_SPEC_SCHEMA_ID,
            "track_id": int(track_id),
            "run_name": run_name,
            "provenance": provenance,
        },
    )
    console.print(
        "[green]Wrote zarr visualization artifacts:[/green] "
        f"visualizations/{png_artifact_name}, visualizations/{spec_artifact_name}"
    )


def plot_track(
    run_group: zarr.Group,
    track_group: zarr.Group,
    track_id: int,
    save_path: Optional[Path],
    bins: int,
    console: Console,
    run_name: Optional[str] = None,
    swim_bouts: Optional[List[Tuple[float, float]]] = None,
    swim_bout_label: Optional[str] = None,
    distance_series: Optional[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], str]] = None,
    render_png: bool = False,
    artifact_dpi: int = 150,
    show: bool = True,
    track_tables: Optional[TrackKinematicsTrackTables] = None,
) -> Optional[bytes]:
    if track_tables is not None and track_tables.time_seconds is not None:
        time_seconds = track_tables.time_seconds
    else:
        time_seconds = track_group["time_seconds"][:]

    def _speed(level: str, unit: str, fallback_name: str) -> Optional[np.ndarray]:
        if track_tables is not None:
            source = track_tables.speed_mm_by_level if unit == "mm" else track_tables.speed_px_by_level
            value = source.get(level)
            if value is not None:
                return value
        return track_group[fallback_name][:] if fallback_name in track_group else None

    speed_raw_px = _speed("raw", "px", "speed_raw_px")
    speed_raw_mm = _speed("raw", "mm", "speed_raw_mm")
    speed_filtered_px = _speed("filtered", "px", "speed_filtered_px")
    speed_filtered_mm = _speed("filtered", "mm", "speed_filtered_mm")
    speed_smoothed_px = _speed("smoothed", "px", "speed_smoothed_px")
    speed_smoothed_mm = _speed("smoothed", "mm", "speed_smoothed_mm")
    speed_averaged_px = _speed("averaged", "px", "speed_averaged_px")
    speed_averaged_mm = _speed("averaged", "mm", "speed_averaged_mm")
    if speed_raw_px is None or speed_raw_mm is None or speed_smoothed_px is None or speed_smoothed_mm is None:
        raise ValueError("Track is missing required raw/smoothed speed arrays for plotting.")

    if track_tables is not None and track_tables.smoothed_heading_degrees is not None:
        smoothed_heading_deg = track_tables.smoothed_heading_degrees
    else:
        smoothed_heading_deg = track_group["smoothed_heading_degrees"][:]
    if track_tables is not None:
        smoothed_accel_px = track_tables.smoothed_acceleration_px_by_level.get("smoothed")
        smoothed_accel_mm = track_tables.smoothed_acceleration_mm_by_level.get("smoothed")
    else:
        smoothed_accel_px = smoothed_accel_mm = None
    if smoothed_accel_px is None:
        smoothed_accel_px = track_group["smoothed_acceleration_px"][:]
    if smoothed_accel_mm is None:
        smoothed_accel_mm = track_group["smoothed_acceleration_mm"][:]

    unit_label, pos_x, pos_y = pick_units(run_group, track_group, track_tables)
    def _has_finite(values: Optional[np.ndarray]) -> bool:
        return values is not None and np.isfinite(values).any()

    def _pick_unit_series(
        mm_values: Optional[np.ndarray],
        px_values: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        if unit_label == "mm" and _has_finite(mm_values):
            return mm_values
        return px_values

    speed_smoothed = _pick_unit_series(speed_smoothed_mm, speed_smoothed_px)
    speed_raw = _pick_unit_series(speed_raw_mm, speed_raw_px)
    speed_filtered = _pick_unit_series(speed_filtered_mm, speed_filtered_px)
    speed_averaged = _pick_unit_series(speed_averaged_mm, speed_averaged_px)
    if speed_raw is None or speed_smoothed is None:
        raise ValueError("Track is missing required raw/smoothed speed arrays for plotting.")
    speed_label = f"Speed ({unit_label}/s)" if unit_label in {"mm", "px"} else "Speed"
    accel = (
        smoothed_accel_mm
        if unit_label == "mm" and np.isfinite(smoothed_accel_mm).any()
        else smoothed_accel_px
    )
    accel_label = f"Acceleration ({unit_label}/s^2)" if unit_label in {"mm", "px"} else "Acceleration"

    # Create figure with shared x-axes for time-series plots, but not for position heatmap
    num_timeseries = 4 + (1 if distance_series is not None else 0)  # speed, accel, [distance], heading, cumulative
    num_rows = num_timeseries + 1  # +1 for position heatmap

    fig = plt.figure(figsize=(10, 3.2 * num_rows))

    # Create time-series subplots with shared x-axis
    timeseries_axes = []
    for i in range(num_timeseries):
        if i == 0:
            # First subplot - this will be the shared x-axis reference
            ax = fig.add_subplot(num_rows, 1, i + 1)
        else:
            # Subsequent time-series subplots share x-axis with the first
            ax = fig.add_subplot(num_rows, 1, i + 1, sharex=timeseries_axes[0])
        timeseries_axes.append(ax)

    # Create position heatmap without shared x-axis (uses spatial coordinates, not time)
    heatmap_ax = fig.add_subplot(num_rows, 1, num_rows)

    # Combine all axes for indexing
    axes = timeseries_axes + [heatmap_ax]
    ax_idx = 0

    # Plot all available speed levels
    speed_ax = axes[ax_idx]
    ax_idx += 1
    speed_ax.plot(time_seconds, speed_raw, color="tab:gray", linewidth=0.8, alpha=0.4, label="Raw")
    if speed_filtered is not None:
        speed_ax.plot(time_seconds, speed_filtered, color="tab:green", linewidth=0.9, alpha=0.6, label="Filtered (hysteresis)")
    speed_ax.plot(time_seconds, speed_smoothed, color="tab:blue", linewidth=1.2, label="Smoothed")
    if speed_averaged is not None:
        speed_ax.plot(time_seconds, speed_averaged, color="tab:purple", linewidth=1.3, alpha=0.8, label="Averaged")
    speed_ax.set_xlabel("Time (s)")
    speed_ax.set_ylabel(speed_label)
    speed_ax.set_title(f"Track {track_id}: Speed over time")
    speed_ax.grid(alpha=0.3)
    if swim_bouts:
        label_added = False
        for start, stop in swim_bouts:
            if not np.isfinite(start) or not np.isfinite(stop):
                continue
            speed_ax.axvspan(
                start,
                stop,
                color="tab:orange",
                alpha=0.18,
                linewidth=0,
                label="Swim bout" if not label_added else None,
            )
            label_added = True
        if label_added and swim_bout_label:
            speed_ax.set_title(f"Track {track_id}: Speed over time (swim bouts: {swim_bout_label})")
    speed_ax.legend(loc="upper right")

    accel_ax = axes[ax_idx]
    ax_idx += 1
    accel_ax.plot(time_seconds, accel, color="tab:red", linewidth=1.0)
    accel_ax.set_xlabel("Time (s)")
    accel_ax.set_ylabel(accel_label)
    accel_ax.set_title("Smoothed acceleration over time")
    accel_ax.grid(alpha=0.3)

    if distance_series is not None:
        dist_time, dist_values, dist_smoothed, dist_unit = distance_series
        distance_ax = axes[ax_idx]
        ax_idx += 1
        distance_ax.plot(
            dist_time,
            dist_values,
            color="tab:purple",
            linewidth=1.0,
            alpha=0.6,
            label="Raw",
        )
        if dist_smoothed is not None and dist_smoothed.size == dist_time.size:
            distance_ax.plot(
                dist_time,
                dist_smoothed,
                color="tab:pink",
                linewidth=1.2,
                label="Smoothed",
            )
        distance_ax.set_xlabel("Time (s)")
        distance_ax.set_ylabel(f"Distance to target ({dist_unit})")
        distance_ax.set_title("Chaser → target distance")
        distance_ax.grid(alpha=0.3)
        # X-axis is automatically shared with other time-series plots
        if (dist_smoothed is not None and dist_smoothed.size == dist_time.size) or dist_values.size:
            distance_ax.legend(loc="upper right")

    heading_ax = axes[ax_idx]
    ax_idx += 1
    heading_ax.plot(time_seconds, smoothed_heading_deg, color="tab:orange", linewidth=1.0)
    heading_ax.set_xlabel("Time (s)")
    heading_ax.set_ylabel("Heading (deg)")
    heading_ax.set_title("Heading over time (smoothed)")
    heading_ax.grid(alpha=0.3)

    if track_tables is not None and track_tables.cumulative_path_distance_mm is not None:
        cumulative = track_tables.cumulative_path_distance_mm
    else:
        cumulative = track_group["cumulative_path_distance_mm"][:]
    if not (np.isfinite(cumulative).any() and unit_label == "mm"):
        if track_tables is not None and track_tables.cumulative_path_distance_px is not None:
            cumulative = track_tables.cumulative_path_distance_px
        else:
            cumulative = track_group["cumulative_path_distance_px"][:]
        cumulative_label = "Cumulative path distance (px)"
    else:
        cumulative_label = "Cumulative path distance (mm)"
    cumulative_ax = axes[ax_idx]
    ax_idx += 1
    cumulative_ax.plot(time_seconds, cumulative, color="tab:green", linewidth=1.2)
    cumulative_ax.set_xlabel("Time (s)")
    cumulative_ax.set_ylabel(cumulative_label)
    cumulative_ax.set_title("Cumulative path distance over time")
    cumulative_ax.grid(alpha=0.3)

    # Filter out NaN positions for histogram
    valid_pos = np.isfinite(pos_x) & np.isfinite(pos_y)
    if np.any(valid_pos):
        heatmap_ax = axes[ax_idx]
        heat = heatmap_ax.hist2d(pos_x[valid_pos], pos_y[valid_pos], bins=bins, cmap="inferno")
        heatmap_ax.set_xlabel(f"X ({unit_label})")
        heatmap_ax.set_ylabel(f"Y ({unit_label})")
        heatmap_ax.set_title(f"Position density ({valid_pos.sum()}/{len(pos_x)} valid)")
        cbar = fig.colorbar(heat[3], ax=heatmap_ax)
        cbar.set_label("Counts")
    else:
        heatmap_ax = axes[ax_idx]
        heatmap_ax.text(0.5, 0.5, "No valid positions", ha="center", va="center", transform=heatmap_ax.transAxes)
        heatmap_ax.set_xlabel(f"X ({unit_label})")
        heatmap_ax.set_ylabel(f"Y ({unit_label})")
        heatmap_ax.set_title("Position density (no valid data)")

    # Build title with run type (online/offline) if available
    title = f"Movement summary – track {track_id}"
    if run_name:
        # Extract online/offline prefix if present
        if run_name.startswith("online/"):
            title = f"Movement summary (online) – track {track_id}"
        elif run_name.startswith("offline/"):
            title = f"Movement summary (offline) – track {track_id}"
        else:
            title = f"Movement summary ({run_name}) – track {track_id}"

    fig.suptitle(title)

    # Add processing metadata at the bottom
    metadata_text = _format_processing_metadata(run_group)
    fig.text(
        0.5, 0.01,
        metadata_text,
        ha='center',
        va='bottom',
        fontsize=8,
        style='italic',
        color='gray',
        wrap=True
    )

    # Adjust layout to make room for metadata text at bottom and title at top
    fig.tight_layout(rect=[0, 0.025, 1, 0.97])

    if save_path:
        fig.savefig(save_path)
        console.print(f"[green]Saved plot to {save_path}[/green]")

    png_bytes: Optional[bytes] = None
    if render_png:
        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=int(artifact_dpi), bbox_inches="tight")
        buffer.seek(0)
        png_bytes = buffer.getvalue()

    if show:
        plt.show()

    plt.close(fig)
    return png_bytes


def main(argv: Optional[Iterable[str]] = None) -> None:
    argv_list = list(argv) if argv is not None else None
    parser = argparse.ArgumentParser(description="Plot track kinematics run metrics.")
    parser.add_argument("zarr_path", help="Path to the Palette Zarr archive.")
    parser.add_argument(
        "--track-kinematics-run",
        dest="track_kinematics_run",
        help="Track kinematics run name (defaults to latest).",
    )
    parser.add_argument("--track-id", type=int, help="Track ID to visualize.")
    parser.add_argument("--bins", type=int, default=200, help="Histogram bins for position heatmap (default: 200).")
    parser.add_argument("--save", help="Path to save the figure instead of showing interactively.")
    parser.add_argument(
        "--offline-only",
        action="store_true",
        help="Only plot track kinematics runs derived from offline metrics.",
    )
    parser.add_argument(
        "--online-only",
        action="store_true",
        help="Only plot detection-based track kinematics runs.",
    )
    parser.add_argument(
        "--swim-bout-run",
        default="latest",
        help="Swim-bout run under analysis/swim_bout_runs to overlay (default: 'latest'). Use empty string or 'none' to disable.",
    )
    parser.add_argument(
        "--speed-level",
        choices=["raw", "filtered", "smoothed", "averaged"],
        default="smoothed",
        help="Speed level to use for swim bout overlay in hierarchical runs (default: smoothed).",
    )
    parser.add_argument(
        "--write-zarr-artifacts",
        action="store_true",
        help=(
            "Persist a PNG snapshot and interactive plot spec under the selected "
            "track_kinematics run's visualizations group."
        ),
    )
    parser.add_argument(
        "--artifact-dpi",
        type=int,
        default=150,
        help="DPI for PNG artifacts written with --write-zarr-artifacts (default: 150).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display interactively even when --save or --write-zarr-artifacts is used.",
    )

    args = parser.parse_args(argv_list)
    command = (
        " ".join(sys.argv)
        if argv_list is None
        else " ".join(["fisheye.analysis.plot_track_kinematics", *map(str, argv_list)])
    )

    console = Console()
    zarr_path = Path(args.zarr_path)
    root = open_zarr_root(zarr_path, mode="a" if args.write_zarr_artifacts else "r")

    include_online = not args.offline_only
    include_offline = not args.online_only
    if args.offline_only and args.online_only:
        include_online = include_offline = True
    if not include_online and not include_offline:
        include_online = include_offline = True

    runs = resolve_track_kinematics_runs(
        root,
        args.track_kinematics_run,
        console,
        include_online=include_online,
        include_offline=include_offline,
    )
    if not runs:
        console.print("[yellow]No track kinematics runs matched the requested filters.[/yellow]")
        return

    save_path = Path(args.save) if args.save else None
    swim_spans: Optional[List[Tuple[float, float]]] = None
    swim_label: Optional[str] = None
    # Allow disabling swim bout overlay with empty string or 'none'
    if args.swim_bout_run and args.swim_bout_run.lower() not in ("", "none"):
        try:
            swim_spans, swim_label = resolve_swim_bout_spans(root, args.swim_bout_run, console, speed_level=args.speed_level)
            if swim_spans:
                console.print(
                    f"[dim]Overlaying {len(swim_spans)} swim bouts from swim_bout_runs/{swim_label}.[/dim]"
                )
            else:
                console.print(
                    f"[yellow]Swim bout run '{swim_label}' contains no bouts to overlay.[/yellow]"
                )
        except ValueError as exc:
            # Gracefully skip if swim bouts are not available
            console.print(f"[dim]Skipping swim bout overlay: {exc}[/dim]")
            swim_spans = None
            swim_label = None

    for run_name, run_group in runs:
        console.print(f"\n[bold]Plotting track kinematics run:[/bold] {run_name}")
        try:
            track_id, track_group = resolve_track(run_group, args.track_id, console)
        except ValueError as exc:
            console.print(f"[yellow]Warning:[/yellow] {exc}")
            continue
        run_scope, logical_run_name = run_name.split("/", 1)
        try:
            track_tables = load_track_kinematics_track(
                root,
                run_name=logical_run_name,
                scope=run_scope,
                track_id=track_id,
                required_speed_levels=("raw", "smoothed"),
            )
        except ValueError as exc:
            console.print(f"[yellow]Warning:[/yellow] Unable to load logical track tables: {exc}")
            continue
        dest = save_path
        if dest:
            # Replace slashes in run_name to avoid invalid filenames
            safe_run_name = run_name.replace("/", "_")
            dest = dest.with_name(f"{dest.stem}_{safe_run_name}{dest.suffix}")

        distance_series = None
        stim_run_name = resolve_stimulus_run(root, run_group)
        if stim_run_name:
            if run_name.startswith("online/"):
                distance_series = compute_online_chaser_distance(
                    zarr_path,
                    stim_run_name,
                    run_group,
                    track_group,
                    track_id,
                    console,
                )
            elif run_name.startswith("offline/"):
                distance_series = compute_offline_chaser_distance(
                    zarr_path,
                    stim_run_name,
                    run_group,
                    track_group,
                    track_id,
                    console,
                )
        else:
            console.print(
                "[yellow]Warning:[/yellow] Unable to resolve stimulus run for distance overlay."
            )

        show_plot = bool(args.show or (not dest and not args.write_zarr_artifacts))
        png_bytes = plot_track(
            run_group,
            track_group,
            track_id,
            dest,
            args.bins,
            console,
            run_name,
            swim_bouts=swim_spans,
            swim_bout_label=swim_label,
            distance_series=distance_series,
            render_png=bool(args.write_zarr_artifacts),
            artifact_dpi=int(args.artifact_dpi),
            show=show_plot,
            track_tables=track_tables,
        )
        if args.write_zarr_artifacts:
            if png_bytes is None:
                console.print("[yellow]Warning:[/yellow] No PNG bytes were rendered; skipping zarr artifacts.")
                continue
            write_track_kinematics_plot_artifacts(
                zarr_path=zarr_path,
                run_group=run_group,
                run_name=run_name,
                track_group=track_group,
                track_id=track_id,
                png_bytes=png_bytes,
                bins=int(args.bins),
                artifact_dpi=int(args.artifact_dpi),
                swim_bout_label=swim_label,
                swim_bout_requested=args.swim_bout_run,
                speed_level=str(args.speed_level),
                distance_series_present=distance_series is not None,
                stimulus_run=stim_run_name,
                console=console,
                command=command,
            )


if __name__ == "__main__":  # pragma: no cover
    main()
