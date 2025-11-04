"""Comprehensive movement run aggregation for Palette archives.

This module consolidates detections, ID assignments, keypoint headings, and
calibration metadata into an analysis-friendly layout under
``analysis/movement_runs``.

It prefers refined keypoints/detections when available, writes per-track
subgroups with rich movement metrics, and records provenance back to the
source runs so downstream tooling can trace inputs.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import zarr
from rich.console import Console
from rich.table import Table

from .compute_speed import (  # re-exported for compatibility
    TrackSpeeds,
    compute_track_speed,
    find_fps,
    load_detection_ids,
    map_refined_detection_ids,
    resolve_dimensions,
)
from .chaser_metrics_loader import load_chaser_metrics
from fisheye.utils.system import get_git_info, get_environment_info
from .chaser_state_interpolator import load_structured_dataset


@dataclass
class KeypointResolution:
    """Resolved keypoint run metadata."""

    group: zarr.Group
    run_name: str
    is_refined: bool
    base_run_name: str
    crop_run: str


@dataclass
class DetectionResolution:
    """Resolved detection group metadata."""

    group: zarr.Group
    path: str
    is_refined: bool
    run_name: str
    variant: str
    source_detect_run: Optional[str]
    parent_path: str


def resolve_keypoint_group(
    root: zarr.Group,
    requested: Optional[str],
    console: Console,
) -> KeypointResolution:
    """Resolve the keypoint group (preferring refined runs)."""

    refined_parent = root.get("refined_keypoints_runs")
    raw_parent = root.get("keypoints_runs")

    if raw_parent is None and refined_parent is None:
        raise ValueError("No keypoint runs found in archive.")

    def resolve_raw(name: str) -> KeypointResolution:
        if raw_parent is None or name not in raw_parent:
            raise ValueError(f"Keypoint run '{name}' not found in keypoints_runs.")
        group = raw_parent[name]
        crop_run = group.attrs.get("source_crop_run")
        if not crop_run:
            raise ValueError(
                f"Keypoint run '{name}' missing 'source_crop_run' attribute; cannot resolve detection source."
            )
        return KeypointResolution(group=group, run_name=name, is_refined=False, base_run_name=name, crop_run=crop_run)

    def resolve_refined(name: str) -> KeypointResolution:
        if refined_parent is None or name not in refined_parent:
            raise ValueError(f"Refined keypoint run '{name}' not found.")
        group = refined_parent[name]
        base_run = group.attrs.get("source_keypoints_run")
        if not base_run:
            raise ValueError(
                f"Refined keypoint run '{name}' missing 'source_keypoints_run' attribute; provenance is required."
            )
        base_resolution = resolve_raw(base_run)
        return KeypointResolution(
            group=group,
            run_name=name,
            is_refined=True,
            base_run_name=base_resolution.base_run_name,
            crop_run=base_resolution.crop_run,
        )

    if requested:
        if requested.startswith("refined/"):
            return resolve_refined(requested.split("/", 1)[1])
        if refined_parent is not None and requested in refined_parent:
            return resolve_refined(requested)
        return resolve_raw(requested)

    if refined_parent is not None:
        latest_refined = refined_parent.attrs.get("latest")
        if latest_refined:
            console.print(
                f"Using refined keypoints run: [cyan]{latest_refined}[/cyan]"
            )
            return resolve_refined(latest_refined)

    if raw_parent is not None:
        latest_raw = raw_parent.attrs.get("latest")
        if latest_raw:
            console.print(f"Using keypoints run: [cyan]{latest_raw}[/cyan]")
            return resolve_raw(latest_raw)

    raise ValueError("Unable to resolve a keypoint run; no runs detected.")


def resolve_detection_from_path(root: zarr.Group, path: str) -> DetectionResolution:
    """Resolve detection metadata from the crop-provided path."""

    if path not in root:
        # Handle legacy references written before refined_detect_runs rename.
        if path.startswith("refined_runs/"):
            tail = path[len("refined_runs/") :]
            legacy_candidates = [
                f"refined_detect_runs/{tail}",
            ]
            if tail.startswith("refined_"):
                # e.g. refined_runs/refined_2023-... -> refined_detect_runs/refined_detect_2023-...
                suffix = tail[len("refined_") :]
                legacy_candidates.append(f"refined_detect_runs/refined_detect_{suffix}")
            for candidate in legacy_candidates:
                if candidate in root:
                    return resolve_detection_from_path(root, candidate)
        raise ValueError(f"Detection group '{path}' referenced by crop run is missing.")

    group = root[path]
    parts = path.split("/")
    if not parts:
        raise ValueError(f"Invalid detection path '{path}'.")

    head = parts[0]
    if head == "refined_detect_runs":
        if len(parts) < 2:
            raise ValueError(f"Malformed refined detection path '{path}'.")
        run_name = parts[1]
        variant = parts[2] if len(parts) > 2 else "interpolated"
        parent_path = "/".join(parts[:2])
        parent_group = root[parent_path]
        source_detect_run = parent_group.attrs.get("source_detect_run")
        return DetectionResolution(
            group=group,
            path=path,
            is_refined=True,
            run_name=run_name,
            variant=variant,
            source_detect_run=source_detect_run,
            parent_path=parent_path,
        )
    if head == "refined_runs":  # legacy refined path fallback
        if len(parts) < 2:
            raise ValueError(f"Malformed legacy refined detection path '{path}'.")
        run_name = parts[1]
        variant = parts[2] if len(parts) > 2 else "interpolated"
        parent_path = "/".join(parts[:2])
        parent_group = root[parent_path]
        source_detect_run = parent_group.attrs.get("source_detect_run")
        return DetectionResolution(
            group=group,
            path=path,
            is_refined=True,
            run_name=run_name,
            variant=variant,
            source_detect_run=source_detect_run,
            parent_path=parent_path,
        )
    if head == "detect_runs":
        if len(parts) < 2:
            raise ValueError(f"Malformed detection path '{path}'.")
        run_name = parts[1]
        return DetectionResolution(
            group=group,
            path=path,
            is_refined=False,
            run_name=run_name,
            variant="raw",
            source_detect_run=run_name,
            parent_path="/".join(parts[:2]),
        )

    raise ValueError(
        "Unsupported detection path '{path}'. Expected detect_runs/ or refined_detect_runs/.".format(path=path)
    )


def _sorted_group_keys(group: Optional[zarr.Group]) -> List[str]:
    if group is None:
        return []
    keys_fn = getattr(group, "group_keys", None)
    try:
        keys = list(keys_fn()) if callable(keys_fn) else []
    except Exception:
        keys = []
    return sorted(key for key in keys if isinstance(key, str))


def prefer_refined_detection(
    root: zarr.Group, detection: DetectionResolution, console: Console
) -> DetectionResolution:
    """Prefer refined detection data when available for the same source run."""
    if detection.is_refined:
        return detection

    refined_parent = root.get("refined_detect_runs")
    if not isinstance(refined_parent, zarr.Group):
        return detection

    candidates: List[str] = []
    for run_name in _sorted_group_keys(refined_parent):
        run_group = refined_parent[run_name]
        if run_group.attrs.get("source_detect_run") == detection.run_name:
            candidates.append(run_name)

    if not candidates:
        return detection

    latest = refined_parent.attrs.get("latest")
    if latest in candidates:
        chosen = latest
    else:
        chosen = candidates[-1]

    target_group = refined_parent[chosen]
    variant_path = "interpolated" if "interpolated" in target_group else None
    refined_path = (
        f"refined_detect_runs/{chosen}/{variant_path}"
        if variant_path
        else f"refined_detect_runs/{chosen}"
    )

    console.print(
        f"[cyan]Preferring refined detections:[/cyan] {refined_path} "
        f"(source_detect_run={detection.run_name})"
    )

    return resolve_detection_from_path(root, refined_path)


def load_stimulus_run_frames(root: zarr.Group, stimulus_run: Optional[str] = None) -> Optional[np.ndarray]:
    """Load the set of camera frame IDs from the stimulus run (experimental period).

    Returns None if no stimulus run is available.
    """
    if "analysis" not in root or "stimulus_runs" not in root["analysis"]:
        return None

    stimulus_parent = root["analysis"]["stimulus_runs"]
    stim_run = stimulus_run
    if stim_run is None:
        latest = stimulus_parent.attrs.get("latest")
        if isinstance(latest, bytes):
            latest = latest.decode("utf-8", "ignore")
        if isinstance(latest, str) and latest in stimulus_parent:
            stim_run = latest

    if stim_run is None or stim_run not in stimulus_parent:
        return None

    stim_group = stimulus_parent[stim_run]
    if "video_metadata" not in stim_group or "frame_metadata" not in stim_group["video_metadata"]:
        return None

    frame_metadata, _ = load_structured_dataset(
        stim_group["video_metadata"], "frame_metadata"
    )
    dtype_names = frame_metadata.dtype.names or ()

    # Find camera frame ID field
    camera_field = None
    for candidate in ["triggering_camera_frame_id", "camera_frame_id"]:
        if candidate in dtype_names:
            camera_field = candidate
            break

    if camera_field is None:
        return None

    camera_frames = np.asarray(frame_metadata[camera_field], dtype=np.int64)
    return np.unique(camera_frames)


def resolve_calibration(root: zarr.Group) -> Tuple[Optional[float], Dict[str, Any]]:
    """Retrieve pixel-to-mm conversion if available."""

    calibration = root.get("calibration")
    if calibration is None:
        return None, {
            "has_calibration": False,
            "measured_fps": None,
            "stimulus_offset_x": None,
            "stimulus_offset_y": None,
            "primary_camera_id": None,
            "camera_offsets": {},
        }

    pixel_to_mm = calibration.attrs.get("pixel_to_mm")
    pixel_to_mm_val = float(pixel_to_mm) if pixel_to_mm is not None else None
    measured_fps = calibration.attrs.get("measured_fps")
    measured_fps_val = float(measured_fps) if measured_fps is not None else None
    stim_offset_x = calibration.attrs.get("stimulus_offset_x")
    stim_offset_y = calibration.attrs.get("stimulus_offset_y")

    stim_offset_x_val = float(stim_offset_x) if stim_offset_x is not None else None
    stim_offset_y_val = float(stim_offset_y) if stim_offset_y is not None else None

    primary_camera_id = calibration.attrs.get("primary_camera_id")
    if isinstance(primary_camera_id, bytes):
        primary_camera_id_val = primary_camera_id.decode("utf-8", "ignore")
    else:
        primary_camera_id_val = primary_camera_id if primary_camera_id is not None else None

    camera_offsets = {}
    if "cameras" in calibration:
        cameras_group = calibration["cameras"]
        if hasattr(cameras_group, "group_keys"):
            camera_ids = list(cameras_group.group_keys())
        else:
            camera_ids = list(cameras_group.keys())

        for cam_id in camera_ids:
            cam_group = cameras_group[cam_id]
            cam_offsets = {}
            for key in ("stimulus_offset_x", "stimulus_offset_y"):
                val = cam_group.attrs.get(key)
                if val is not None:
                    cam_offsets[key] = float(val)
            if cam_offsets:
                camera_offsets[str(cam_id)] = cam_offsets

    return pixel_to_mm_val, {
        "has_calibration": pixel_to_mm_val is not None,
        "measured_fps": measured_fps_val,
        "stimulus_offset_x": stim_offset_x_val,
        "stimulus_offset_y": stim_offset_y_val,
        "primary_camera_id": primary_camera_id_val,
        "camera_offsets": camera_offsets,
    }


def ensure_movement_run_group(
    root: zarr.Group,
    run_name: Optional[str],
    *,
    run_type: str = "online",
    overwrite: bool = False,
) -> Tuple[str, zarr.Group]:
    """Create /analysis/movement_runs/<type>/<run_name> (auto timestamp if needed)."""

    if run_type not in {"online", "offline"}:
        raise ValueError("run_type must be 'online' or 'offline'")

    analysis = root.require_group("analysis")
    movement_parent = analysis.require_group("movement_runs")
    type_parent = movement_parent.require_group(run_type)

    if run_name:
        if run_name in type_parent:
            if not overwrite:
                raise ValueError(f"Movement run '{run_name}' already exists under {run_type}.")
            del type_parent[run_name]
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        prefix = "movement" if run_type == "online" else "movement_offline"
        run_name = f"{prefix}_{timestamp}"

    run_group = type_parent.create_group(run_name)

    # Update convenience attributes
    movement_parent.attrs["latest"] = f"{run_type}/{run_name}"
    attr_key = "latest_online" if run_type == "online" else "latest_offline"
    type_parent.attrs["latest"] = run_name
    movement_parent.attrs[attr_key] = run_name

    return run_name, run_group


def _nan_array(shape: Tuple[int, ...], dtype: np.dtype = np.float32) -> np.ndarray:
    arr = np.empty(shape, dtype=dtype)
    arr.fill(np.nan)
    return arr


def _float32(data: np.ndarray) -> np.ndarray:
    return np.asarray(data, dtype=np.float32)


def _int64(data: np.ndarray) -> np.ndarray:
    return np.asarray(data, dtype=np.int64)


def _boolean(data: np.ndarray) -> np.ndarray:
    return np.asarray(data, dtype=bool)


def build_track_datasets(
    detection_ids: np.ndarray,
    frames: np.ndarray,
    positions_px: np.ndarray,
    headings_deg: np.ndarray,
    keypoint_success: np.ndarray,
    detection_source: Optional[np.ndarray],
    fps: float,
    smooth_seconds: float,
    pixel_to_mm: Optional[float],
) -> Tuple[Dict[int, Dict[str, np.ndarray]], List[Dict[str, float]]]:
    """Assemble per-track data arrays and summary statistics."""

    unique_ids = np.unique(detection_ids)
    tracks: Dict[int, Dict[str, np.ndarray]] = {}
    summaries: List[Dict[str, float]] = []

    pixel_to_mm_val = pixel_to_mm if (pixel_to_mm is not None and pixel_to_mm > 0) else None

    for track_id in unique_ids:
        mask = detection_ids == track_id
        if not np.any(mask):
            continue

        track_frames = frames[mask]
        coords_px = positions_px[mask]
        headings_track = headings_deg[mask]
        kp_success_track = keypoint_success[mask]
        detection_index = np.where(mask)[0]
        det_source_track = (
            detection_source[mask].astype(np.int8)
            if detection_source is not None
            else np.zeros(mask.sum(), dtype=np.int8)
        )

        order = np.argsort(track_frames, kind="stable")
        track_frames = track_frames[order]
        coords_px = coords_px[order]
        headings_track = headings_track[order]
        kp_success_track = kp_success_track[order]
        det_source_track = det_source_track[order]
        detection_indices_sorted = detection_index[order]

        speeds = compute_track_speed(track_frames.copy(), coords_px.copy(), fps=fps, smooth_seconds=smooth_seconds)

        instantaneous_px = speeds.instantaneous
        smoothed_px = speeds.smoothed
        distance_px = speeds.distance
        cumulative_px = speeds.cumulative_distance
        seconds = speeds.seconds
        speed_per_second_px = speeds.speed_per_second

        if pixel_to_mm_val is not None:
            coords_mm = coords_px * pixel_to_mm_val
            instantaneous_mm = instantaneous_px * pixel_to_mm_val
            smoothed_mm = smoothed_px * pixel_to_mm_val
            distance_mm = distance_px * pixel_to_mm_val
            cumulative_mm = cumulative_px * pixel_to_mm_val
            speed_per_second_mm = speed_per_second_px * pixel_to_mm_val
        else:
            coords_mm = _nan_array(coords_px.shape)
            instantaneous_mm = _nan_array(instantaneous_px.shape)
            smoothed_mm = _nan_array(smoothed_px.shape)
            distance_mm = _nan_array(distance_px.shape)
            cumulative_mm = _nan_array(cumulative_px.shape)
            speed_per_second_mm = _nan_array(speed_per_second_px.shape)

        heading_rad = np.deg2rad(headings_track)
        heading_valid = np.isfinite(heading_rad)
        time_seconds = track_frames.astype(np.float64) / fps
        seconds_per_frame = np.floor(time_seconds).astype(np.int64)

        delta_seconds_full = np.zeros(track_frames.shape[0], dtype=np.float64)
        if track_frames.size >= 2:
            delta_seconds_full[1:] = np.diff(track_frames) / fps

        # Acceleration from smoothed speed profile
        acceleration_px = np.full(smoothed_px.shape, np.nan, dtype=np.float64)
        acceleration_mm = np.full(smoothed_px.shape, np.nan, dtype=np.float64)

        if smoothed_px.size >= 2:
            delta_speed_px = smoothed_px[1:] - smoothed_px[:-1]
            delta_t = delta_seconds_full[1:]
            valid = (delta_t > 0) & np.isfinite(delta_speed_px)
            accel_vals = np.full(delta_speed_px.shape, np.nan, dtype=np.float64)
            accel_vals[valid] = delta_speed_px[valid] / delta_t[valid]
            acceleration_px[1:] = accel_vals
            if pixel_to_mm_val is not None and np.isfinite(pixel_to_mm_val):
                acceleration_mm[1:] = accel_vals * pixel_to_mm_val

    accel_window = max(1, int(round(fps * smooth_seconds)))
    if accel_window > 1 and acceleration_px.size > 0:
        kernel = np.ones(accel_window, dtype=np.float64)
        val_mask = np.isfinite(acceleration_px).astype(np.float64)
        accel_values = np.nan_to_num(acceleration_px, nan=0.0, copy=False)
        sum_values = np.convolve(accel_values, kernel, mode="same")
        count_values = np.convolve(val_mask, kernel, mode="same")
        smoothed_accel_px = np.full_like(acceleration_px, np.nan)
        valid = count_values > 0
        smoothed_accel_px[valid] = sum_values[valid] / count_values[valid]
    else:
        smoothed_accel_px = acceleration_px.copy()

    if pixel_to_mm_val is not None and np.isfinite(pixel_to_mm_val):
        smoothed_accel_mm = smoothed_accel_px * pixel_to_mm_val
        accel_mm = acceleration_mm
    else:
        smoothed_accel_mm = _nan_array(smoothed_accel_px.shape)
        accel_mm = _nan_array(acceleration_px.shape)

    heading_window = max(1, int(round(fps * smooth_seconds)))
    if heading_window > 1 and heading_rad.size > 0:
        kernel = np.ones(heading_window, dtype=np.float64)
        valid_weights = np.convolve(heading_valid.astype(np.float64), kernel, mode="same")
        cos_vals = np.cos(np.where(heading_valid, heading_rad, 0.0))
        sin_vals = np.sin(np.where(heading_valid, heading_rad, 0.0))
        cos_sum = np.convolve(cos_vals, kernel, mode="same")
        sin_sum = np.convolve(sin_vals, kernel, mode="same")
        with np.errstate(invalid="ignore"):
            cos_mean = np.where(valid_weights > 0, cos_sum / valid_weights, np.nan)
            sin_mean = np.where(valid_weights > 0, sin_sum / valid_weights, np.nan)
        smoothed_heading_rad = np.arctan2(sin_mean, cos_mean)
    else:
        smoothed_heading_rad = heading_rad.copy()

    smoothed_heading_deg = np.rad2deg(smoothed_heading_rad)

    unique_seconds = speeds.seconds.astype(np.int64)
    # fallback if TrackSpeeds.seconds is empty
    if unique_seconds.size == 0 and seconds_per_frame.size > 0:
        unique_seconds = np.unique(seconds_per_frame)
    heading_per_second_rad = np.full(unique_seconds.size, np.nan, dtype=np.float64)
    heading_per_second_resultant = np.zeros(unique_seconds.size, dtype=np.float32)
    for idx, sec in enumerate(unique_seconds):
        mask_sec = (seconds_per_frame == sec) & heading_valid
        valid_angles = heading_rad[mask_sec]
        if valid_angles.size:
            mean_vector = np.mean(np.exp(1j * valid_angles))
            heading_per_second_rad[idx] = math.atan2(mean_vector.imag, mean_vector.real)
            heading_per_second_resultant[idx] = np.float32(np.abs(mean_vector))
    heading_per_second_deg = np.rad2deg(heading_per_second_rad)

    tracks[int(track_id)] = {
        "frame_indices": track_frames.astype(np.int64),
        "time_seconds": _float32(time_seconds),
        "detection_indices": detection_indices_sorted.astype(np.int64),
        "positions_px": _float32(coords_px),
        "positions_mm": _float32(coords_mm),
        "heading_degrees": _float32(headings_track),
        "heading_radians": _float32(heading_rad),
        "smoothed_heading_degrees": _float32(smoothed_heading_deg),
        "smoothed_heading_radians": _float32(smoothed_heading_rad),
        "keypoint_success": _boolean(kp_success_track),
        "detection_source": det_source_track.astype(np.int8),
        "instantaneous_speed_px": _float32(instantaneous_px),
        "instantaneous_speed_mm": _float32(instantaneous_mm),
        "smoothed_speed_px": _float32(smoothed_px),
        "smoothed_speed_mm": _float32(smoothed_mm),
        "acceleration_px": _float32(acceleration_px),
        "acceleration_mm": _float32(accel_mm),
        "smoothed_acceleration_px": _float32(smoothed_accel_px),
        "smoothed_acceleration_mm": _float32(smoothed_accel_mm),
        "distance_per_frame_px": _float32(distance_px),
        "distance_per_frame_mm": _float32(distance_mm),
        "cumulative_distance_px": _float32(cumulative_px),
        "cumulative_distance_mm": _float32(cumulative_mm),
        "second_indices": seconds_per_frame,
        "speed_per_second_px": _float32(speed_per_second_px),
        "speed_per_second_mm": _float32(speed_per_second_mm),
        "heading_per_second_degrees": _float32(heading_per_second_deg),
        "heading_per_second_resultant": heading_per_second_resultant.astype(np.float32),
    }

    finite = instantaneous_px[np.isfinite(instantaneous_px)]
    mean_speed_px = float(np.mean(finite)) if finite.size else float("nan")
    median_speed_px = float(np.median(finite)) if finite.size else float("nan")
    max_speed_px = float(np.max(finite)) if finite.size else float("nan")

    total_distance_px = float(cumulative_px[-1]) if cumulative_px.size else 0.0
    total_distance_mm = (
        float(cumulative_mm[-1]) if cumulative_mm.size and pixel_to_mm_val is not None else float("nan")
    )

    mean_speed_mm = mean_speed_px * pixel_to_mm_val if pixel_to_mm_val is not None and np.isfinite(mean_speed_px) else float("nan")
    median_speed_mm = (
        median_speed_px * pixel_to_mm_val
        if pixel_to_mm_val is not None and np.isfinite(median_speed_px)
        else float("nan")
    )
    max_speed_mm = max_speed_px * pixel_to_mm_val if pixel_to_mm_val is not None and np.isfinite(max_speed_px) else float("nan")

    mean_speed_per_second_px = float(np.nanmean(speed_per_second_px)) if speed_per_second_px.size else float("nan")
    mean_speed_per_second_mm = (
        mean_speed_per_second_px * pixel_to_mm_val
        if pixel_to_mm_val is not None and np.isfinite(mean_speed_per_second_px)
        else float("nan")
    )

    valid_heading = heading_rad[np.isfinite(heading_rad)]
    if valid_heading.size:
        mean_vector = np.mean(np.exp(1j * valid_heading))
        heading_mean_deg = float(math.degrees(math.atan2(mean_vector.imag, mean_vector.real)))
        heading_consistency = float(np.abs(mean_vector))
    else:
        heading_mean_deg = float("nan")
        heading_consistency = float("nan")

    accel_finite = smoothed_accel_px[np.isfinite(smoothed_accel_px)]
    mean_accel_px = float(np.mean(accel_finite)) if accel_finite.size else float("nan")
    accel_std_px = float(np.std(accel_finite)) if accel_finite.size else float("nan")
    mean_accel_mm = mean_accel_px * pixel_to_mm_val if pixel_to_mm_val is not None and np.isfinite(mean_accel_px) else float("nan")
    accel_std_mm = accel_std_px * pixel_to_mm_val if pixel_to_mm_val is not None and np.isfinite(accel_std_px) else float("nan")

    summary = {
        "track_id": float(track_id),
        "samples": int(track_frames.size),
        "mean_speed_px": mean_speed_px,
        "median_speed_px": median_speed_px,
        "max_speed_px": max_speed_px,
        "mean_speed_mm": mean_speed_mm,
        "median_speed_mm": median_speed_mm,
        "max_speed_mm": max_speed_mm,
        "mean_speed_per_second_px": mean_speed_per_second_px,
        "mean_speed_per_second_mm": mean_speed_per_second_mm,
        "total_distance_px": total_distance_px,
        "total_distance_mm": total_distance_mm,
        "heading_mean_deg": heading_mean_deg,
        "heading_resultant": heading_consistency,
        "mean_acceleration_px": mean_accel_px,
        "mean_acceleration_mm": mean_accel_mm,
        "acceleration_std_px": accel_std_px,
        "acceleration_std_mm": accel_std_mm,
        "keypoint_success_rate": float(np.mean(kp_success_track)) if kp_success_track.size else float("nan"),
        "duration_seconds": float(time_seconds[-1] - time_seconds[0]) if time_seconds.size > 1 else 0.0,
    }
    summaries.append(summary)

    return tracks, summaries


def save_movement_tracks(
    run_group: zarr.Group,
    tracks: Dict[int, Dict[str, np.ndarray]],
    summaries: List[Dict[str, float]],
) -> List[int]:
    """Persist per-track data beneath the movement run group."""

    tracks_parent = run_group.create_group("tracks")
    ordered_ids = sorted(int(track_id) for track_id in tracks.keys())
    track_ids_array = np.asarray(ordered_ids, dtype=np.int32)
    chunks = (min(1024, len(ordered_ids)),) if ordered_ids else (1,)
    run_group.create_array("track_ids", data=track_ids_array, chunks=chunks, overwrite=True)

    manifest: List[Dict[str, float]] = []
    summary_by_id = {int(item["track_id"]): item for item in summaries}

    for track_id in ordered_ids:
        data = tracks[track_id]
        summary = summary_by_id.get(track_id, {})
        subgroup = tracks_parent.create_group(f"id_{track_id}")

        sample_count = int(data["frame_indices"].size)
        base_chunk = (min(1024, sample_count),) if sample_count else (1,)

        subgroup.create_array("frame_indices", data=data["frame_indices"], chunks=base_chunk, overwrite=True)
        subgroup.create_array("time_seconds", data=data["time_seconds"], chunks=base_chunk, overwrite=True)
        subgroup.create_array("detection_indices", data=data["detection_indices"], chunks=base_chunk, overwrite=True)
        subgroup.create_array("positions_px", data=data["positions_px"], chunks=(base_chunk[0], 2), overwrite=True)
        subgroup.create_array("positions_mm", data=data["positions_mm"], chunks=(base_chunk[0], 2), overwrite=True)
        subgroup.create_array("heading_degrees", data=data["heading_degrees"], chunks=base_chunk, overwrite=True)
        subgroup.create_array("heading_radians", data=data["heading_radians"], chunks=base_chunk, overwrite=True)
        subgroup.create_array("smoothed_heading_degrees", data=data["smoothed_heading_degrees"], chunks=base_chunk, overwrite=True)
        subgroup.create_array("smoothed_heading_radians", data=data["smoothed_heading_radians"], chunks=base_chunk, overwrite=True)
        subgroup.create_array("keypoint_success", data=data["keypoint_success"], chunks=base_chunk, overwrite=True)
        subgroup.create_array("detection_source", data=data["detection_source"], chunks=base_chunk, overwrite=True)
        subgroup.create_array("instantaneous_speed_px", data=data["instantaneous_speed_px"], chunks=base_chunk, overwrite=True)
        subgroup.create_array("instantaneous_speed_mm", data=data["instantaneous_speed_mm"], chunks=base_chunk, overwrite=True)
        subgroup.create_array("smoothed_speed_px", data=data["smoothed_speed_px"], chunks=base_chunk, overwrite=True)
        subgroup.create_array("smoothed_speed_mm", data=data["smoothed_speed_mm"], chunks=base_chunk, overwrite=True)
        subgroup.create_array("acceleration_px", data=data["acceleration_px"], chunks=base_chunk, overwrite=True)
        subgroup.create_array("acceleration_mm", data=data["acceleration_mm"], chunks=base_chunk, overwrite=True)
        subgroup.create_array("smoothed_acceleration_px", data=data["smoothed_acceleration_px"], chunks=base_chunk, overwrite=True)
        subgroup.create_array("smoothed_acceleration_mm", data=data["smoothed_acceleration_mm"], chunks=base_chunk, overwrite=True)
        subgroup.create_array("distance_per_frame_px", data=data["distance_per_frame_px"], chunks=base_chunk, overwrite=True)
        subgroup.create_array("distance_per_frame_mm", data=data["distance_per_frame_mm"], chunks=base_chunk, overwrite=True)
        subgroup.create_array("cumulative_distance_px", data=data["cumulative_distance_px"], chunks=base_chunk, overwrite=True)
        subgroup.create_array("cumulative_distance_mm", data=data["cumulative_distance_mm"], chunks=base_chunk, overwrite=True)

        seconds = data["second_indices"]
        sec_chunk = (min(512, seconds.size),) if seconds.size else (1,)
        subgroup.create_array("second_indices", data=seconds, chunks=sec_chunk, overwrite=True)
        subgroup.create_array("speed_per_second_px", data=data["speed_per_second_px"], chunks=sec_chunk, overwrite=True)
        subgroup.create_array("speed_per_second_mm", data=data["speed_per_second_mm"], chunks=sec_chunk, overwrite=True)
        subgroup.create_array("heading_per_second_degrees", data=data["heading_per_second_degrees"], chunks=sec_chunk, overwrite=True)
        subgroup.create_array("heading_per_second_resultant", data=data["heading_per_second_resultant"], chunks=sec_chunk, overwrite=True)

        subgroup.attrs.update(
            {
                "track_id": int(track_id),
                "num_samples": sample_count,
                "summary": summary,
            }
        )

        manifest.append(
            {
                "track_id": int(track_id),
                "group": f"tracks/id_{track_id}",
                "samples": sample_count,
                "mean_speed_px": float(summary.get("mean_speed_px", float("nan"))),
                "mean_speed_mm": float(summary.get("mean_speed_mm", float("nan"))),
                "total_distance_px": float(summary.get("total_distance_px", float("nan"))),
                "total_distance_mm": float(summary.get("total_distance_mm", float("nan"))),
                "heading_mean_deg": float(summary.get("heading_mean_deg", float("nan"))),
                "heading_resultant": float(summary.get("heading_resultant", float("nan"))),
                "mean_acceleration_px": float(summary.get("mean_acceleration_px", float("nan"))),
                "mean_acceleration_mm": float(summary.get("mean_acceleration_mm", float("nan"))),
            }
        )

    run_group.attrs["track_manifest"] = manifest
    return ordered_ids


def _write_run_array(group: zarr.Group, name: str, data: np.ndarray) -> None:
    """Create or overwrite an array under the movement run group."""

    array = np.asarray(data)
    if array.ndim == 0:
        chunks = None
    else:
        first_dim = array.shape[0] if array.shape[0] > 0 else 1
        chunk_first = max(1, min(4096, first_dim))
        chunks = (chunk_first,) + array.shape[1:]

    group.create_array(name, data=array, chunks=chunks, overwrite=True)


def _smooth_series(values: np.ndarray, window: int) -> np.ndarray:
    """Apply a centered moving average that ignores NaNs."""

    if window <= 1:
        return values.astype(np.float32, copy=True)

    series = np.asarray(values, dtype=np.float32)
    if series.size == 0:
        return series

    valid = np.isfinite(series)
    if not np.any(valid):
        return np.full(series.shape, np.nan, dtype=np.float32)

    kernel = np.ones(window, dtype=np.float32)
    filled = np.nan_to_num(series, nan=0.0, copy=False)
    counts = valid.astype(np.float32)
    sum_values = np.convolve(filled, kernel, mode="same")
    count_values = np.convolve(counts, kernel, mode="same")

    smoothed = np.full(series.shape, np.nan, dtype=np.float32)
    nonzero = count_values > 0
    smoothed[nonzero] = sum_values[nonzero] / count_values[nonzero]
    return smoothed


def _interpolate_gaps(values: np.ndarray, max_gap: int) -> np.ndarray:
    """Linearly interpolate NaN runs shorter than or equal to max_gap frames."""

    series = np.asarray(values, dtype=np.float32)
    if series.size == 0:
        return series
    if max_gap <= 0:
        return series.copy()

    result = series.copy()
    isnan = np.isnan(result)
    if not np.any(isnan):
        return result

    idx = 0
    length = result.shape[0]
    while idx < length:
        if not isnan[idx]:
            idx += 1
            continue
        start = idx
        while idx < length and isnan[idx]:
            idx += 1
        end = idx  # first finite after gap or len
        gap_size = end - start

        if gap_size == 0 or gap_size > max_gap:
            continue

        left_idx = start - 1
        right_idx = end
        if left_idx < 0 or right_idx >= length:
            continue

        left_val = result[left_idx]
        right_val = result[right_idx]
        if not np.isfinite(left_val) or not np.isfinite(right_val):
            continue

        step = (right_val - left_val) / (gap_size + 1)
        for offset in range(1, gap_size + 1):
            result[start + offset - 1] = left_val + step * offset

    return result


def _persist_chaser_metrics_to_run(
    run_group: zarr.Group,
    bundle: "ChaserMetricsBundle",
    *,
    fps: float,
    smooth_seconds: float,
    distance_interp_seconds: float,
) -> Dict[str, object]:
    """Write shared chaser metric arrays to the movement run root."""

    metadata: Dict[str, object] = {
        "metrics_run": bundle.provenance.get("metrics_run"),
        "stimulus_run": bundle.provenance.get("stimulus_run"),
        "chaser_index": int(bundle.provenance.get("chaser_index", 0)),
    }

    shared_arrays: Dict[str, np.ndarray] = {
        "camera_frame_ids": np.asarray(bundle.camera_frame_ids, dtype=np.int64),
        "stimulus_frame_nums": np.asarray(bundle.stimulus_frame_nums, dtype=np.int64),
        "timestamp_ns": np.asarray(bundle.timestamp_ns, dtype=np.int64),
        "trial_state": np.asarray(bundle.trial_state, dtype=np.int16),
    }
    if bundle.metadata_mask is not None:
        shared_arrays["metadata_mask"] = np.asarray(bundle.metadata_mask, dtype=bool)

    offline = bundle.offline
    interpolated_arrays: Dict[str, np.ndarray] = {}
    if offline:
        if "distance_px" in offline:
            distance_px = np.asarray(offline["distance_px"], dtype=np.float32)
            shared_arrays["distance_to_target_px"] = distance_px
        if "distance_mm" in offline:
            distance_mm = np.asarray(offline["distance_mm"], dtype=np.float32)
            shared_arrays["distance_to_target_mm"] = distance_mm
        if "angle_unsigned_deg" in offline:
            shared_arrays["angle_unsigned_deg"] = np.asarray(offline["angle_unsigned_deg"], dtype=np.float32)
        if "angle_signed_deg" in offline:
            shared_arrays["angle_signed_deg"] = np.asarray(offline["angle_signed_deg"], dtype=np.float32)
        if "heading_deg" in offline:
            shared_arrays["heading_deg"] = np.asarray(offline["heading_deg"], dtype=np.float32)
        if "fish_centroid_px" in offline:
            fish_centroids = np.asarray(offline["fish_centroid_px"], dtype=np.float32)
            shared_arrays["fish_centroid_px"] = fish_centroids
            shared_arrays.setdefault("fish_centroids_px", fish_centroids)
        if "chaser_position_px" in offline:
            chaser_positions = np.asarray(offline["chaser_position_px"], dtype=np.float32)
            shared_arrays["chaser_position_px"] = chaser_positions
            shared_arrays["chaser_positions_px"] = chaser_positions
        if "has_offline" in offline:
            shared_arrays["has_offline"] = np.asarray(offline["has_offline"], dtype=bool)

    # Persist raw arrays first; collect smoothed variants when applicable
    for name, array in shared_arrays.items():
        _write_run_array(run_group, name, array)

    window = 1
    max_gap = 0
    interp_seconds_val = 0.0
    try:
        fps_val = float(fps)
        smooth_val = float(smooth_seconds)
        if np.isfinite(fps_val) and fps_val > 0 and np.isfinite(smooth_val) and smooth_val > 0:
            window = max(1, int(round(fps_val * smooth_val)))
        interp_seconds = float(distance_interp_seconds)
        if np.isfinite(fps_val) and fps_val > 0 and np.isfinite(interp_seconds) and interp_seconds > 0:
            interp_seconds_val = interp_seconds
            max_gap = max(0, int(round(fps_val * interp_seconds)))
    except Exception:
        window = 1
        max_gap = 0
        interp_seconds_val = 0.0

    if max_gap > 0:
        if "distance_to_target_px" in shared_arrays:
            interpolated_px = _interpolate_gaps(shared_arrays["distance_to_target_px"], max_gap)
            interpolated_arrays["distance_to_target_interpolated_px"] = interpolated_px
        if "distance_to_target_mm" in shared_arrays:
            interpolated_mm = _interpolate_gaps(shared_arrays["distance_to_target_mm"], max_gap)
            interpolated_arrays["distance_to_target_interpolated_mm"] = interpolated_mm

    for name, array in interpolated_arrays.items():
        _write_run_array(run_group, name, array)

    if window > 1:
        source_px = interpolated_arrays.get("distance_to_target_interpolated_px")
        if source_px is None:
            source_px = shared_arrays.get("distance_to_target_px")
        source_mm = interpolated_arrays.get("distance_to_target_interpolated_mm")
        if source_mm is None:
            source_mm = shared_arrays.get("distance_to_target_mm")
        if source_px is not None:
            smoothed_px = _smooth_series(source_px, window)
            _write_run_array(run_group, "distance_to_target_smoothed_px", smoothed_px)
        if source_mm is not None:
            smoothed_mm = _smooth_series(source_mm, window)
            _write_run_array(run_group, "distance_to_target_smoothed_mm", smoothed_mm)

    metadata["distance_interpolation_seconds"] = float(interp_seconds_val)
    return metadata


def _columnar_bout_data(bouts: np.ndarray) -> Dict[str, np.ndarray]:
    """Convert structured bout array to columnar float32/int32 arrays."""

    columns: Dict[str, np.ndarray] = {}
    if bouts.size == 0 or bouts.dtype.names is None:
        return columns

    for name in bouts.dtype.names:
        data = bouts[name]
        kind = data.dtype.kind
        if kind in {"f", "c"}:  # floats (complex not expected but guard)
            columns[name] = np.asarray(data, dtype=np.float32)
        elif kind in {"i", "u"}:
            columns[name] = np.asarray(data, dtype=np.int32)
        else:
            # skip unsupported fields (e.g. strings)
            continue
    return columns


def _mirror_swim_bouts_to_tracks(
    root: zarr.Group,
    run_group: zarr.Group,
    track_ids: Iterable[int],
    swim_bout_run: Optional[str],
    console: Console,
) -> Optional[str]:
    analysis = root.get("analysis")
    if analysis is None or "swim_bout_runs" not in analysis:
        return None

    bouts_parent = analysis["swim_bout_runs"]
    run_name = swim_bout_run
    if not run_name or run_name.lower() == "latest":
        candidate = bouts_parent.attrs.get("latest")
        if isinstance(candidate, str) and candidate:
            run_name = candidate
    if not run_name:
        console.print(
            "[yellow]Warning:[/yellow] Unable to mirror swim bouts (no swim_bout_runs/latest attribute)."
        )
        return None
    if run_name not in bouts_parent:
        console.print(
            f"[yellow]Warning:[/yellow] Swim bout run '{run_name}' not found; skipping mirror."
        )
        return None

    bout_group = bouts_parent[run_name]
    if "bouts" not in bout_group:
        console.print(
            f"[yellow]Warning:[/yellow] Swim bout run '{run_name}' lacks a 'bouts' dataset."
        )
        return None

    bouts_struct = np.asarray(bout_group["bouts"])
    columns = _columnar_bout_data(bouts_struct)
    if not columns:
        console.print(
            f"[yellow]Warning:[/yellow] Swim bout run '{run_name}' contains no numeric bout fields to mirror."
        )
        return None

    tracks_parent = run_group["tracks"]
    for track_id in track_ids:
        subgroup = tracks_parent[f"id_{track_id}"].require_group("swim_bouts")
        # Clear existing arrays
        for name in list(subgroup.array_keys()):
            del subgroup[name]
        for name, array in columns.items():
            subgroup.create_array(
                name,
                data=array,
                chunks=(max(1, min(4096, array.shape[0])),),
                overwrite=True,
            )
        subgroup.attrs.update(
            {
                "source_swim_bout_run": run_name,
                "mirrored_fields": list(columns.keys()),
            }
        )

    console.print(
        f"[dim]Mirrored swim bouts from swim_bout_runs/{run_name} into movement tracks.[/dim]"
    )
    return run_name


def summarize_to_table(
    summaries: List[Dict[str, float]],
    pixel_to_mm: Optional[float],
    console: Console,
) -> Tuple[float, float]:
    """Render a Rich table summarizing track metrics."""

    table = Table(title="Movement summary", show_lines=False)
    table.add_column("Track ID", justify="right")
    table.add_column("Samples", justify="right")
    table.add_column("Mean px/s", justify="right")
    table.add_column("Mean mm/s", justify="right")
    table.add_column("Median px/s", justify="right")
    table.add_column("Max px/s", justify="right")
    table.add_column("Distance px", justify="right")
    table.add_column("Heading mean (deg)", justify="right")
    table.add_column("Heading resultant", justify="right")
    table.add_column("Mean accel (px/s²)", justify="right")
    table.add_column("Mean accel (mm/s²)", justify="right")
    table.add_column("Distance mm", justify="right")

    total_px = 0.0
    total_mm = 0.0

    for row in summaries:
        total_px += float(row.get("total_distance_px", 0.0))
        dist_mm = row.get("total_distance_mm", float("nan"))
        if not math.isnan(dist_mm):
            total_mm += float(dist_mm)
        table.add_row(
            str(int(row["track_id"])),
            str(int(row["samples"])),
            f"{row['mean_speed_px']:.2f}" if np.isfinite(row["mean_speed_px"]) else "nan",
            f"{row['mean_speed_mm']:.2f}" if np.isfinite(row["mean_speed_mm"]) else "nan",
            f"{row['median_speed_px']:.2f}" if np.isfinite(row["median_speed_px"]) else "nan",
            f"{row['max_speed_px']:.2f}" if np.isfinite(row["max_speed_px"]) else "nan",
            f"{row['total_distance_px']:.2f}",
            f"{row['heading_mean_deg']:.2f}" if np.isfinite(row["heading_mean_deg"]) else "nan",
            f"{row['heading_resultant']:.2f}" if np.isfinite(row["heading_resultant"]) else "nan",
            f"{row['mean_acceleration_px']:.2f}" if np.isfinite(row["mean_acceleration_px"]) else "nan",
            f"{row['mean_acceleration_mm']:.2f}" if np.isfinite(row["mean_acceleration_mm"]) else "nan",
            f"{row['total_distance_mm']:.2f}" if np.isfinite(row["total_distance_mm"]) else "nan",
        )

    console.print(table)
    if pixel_to_mm is not None:
        console.print(
            f"Total distance: {total_px:.2f} px ({total_mm:.2f} mm)"
        )
    else:
        console.print(f"Total distance: {total_px:.2f} px")
    return total_px, total_mm if pixel_to_mm is not None else float("nan")


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Create movement_runs entries consolidating detections, IDs, keypoints, and calibration.",
    )
    parser.add_argument("zarr_path", help="Path to the Palette Zarr archive.")
    parser.add_argument(
        "--keypoint-run",
        help="Keypoint run to use. Prefix with 'refined/' to target a refined run. Default: latest refined if available.",
    )
    parser.add_argument("--run-name", help="Optional name for the output movement run.")
    parser.add_argument("--smooth-seconds", type=float, default=1.0, help="Smoothing window in seconds (default: 1.0).")
    parser.add_argument(
        "--distance-interpolation-seconds",
        type=float,
        default=0.0,
        help="Maximum gap duration (seconds) to fill via linear interpolation for chaser distances (default: 0).",
    )
    parser.add_argument("--skip-unassigned", action="store_true", help="Ignore detections with ID < 0.")
    parser.add_argument("--fps", type=float, default=None, help="Override frames-per-second value.")
    parser.add_argument("--no-write", action="store_true", help="Do not write results back to the Zarr archive.")
    parser.add_argument(
        "--offline-only",
        action="store_true",
        help="Skip detection-based movement run; only compute offline metrics run.",
    )
    parser.add_argument(
        "--online-only",
        action="store_true",
        help="Only compute detection-based movement run, skipping offline metrics.",
    )
    parser.add_argument(
        "--metrics-run",
        help="Specific analysis/chaser_fish_metrics/<run> to use for offline movement analysis (default: latest).",
    )
    parser.add_argument(
        "--swim-bout-run",
        help="analysis/swim_bout_runs/<run> to mirror into the offline movement run (default: latest).",
    )
    parser.add_argument(
        "--chaser-index",
        type=int,
        default=0,
        help="Chaser index for offline metrics (default: 0).",
    )
    parser.add_argument(
        "--offline-run-name",
        help="Optional name for the offline movement run (auto-generated if omitted).",
    )
    parser.add_argument(
        "--stimulus-run",
        help="Stimulus run name to filter online movement data to experimental period (default: latest).",
    )
    parser.add_argument(
        "--refined-online-run",
        help="Use refined online positions from refined_online_runs/<run> instead of raw online data (default: None).",
    )

    args = parser.parse_args(argv)

    console = Console()
    mode = "r" if args.no_write else "a"
    root = zarr.open(args.zarr_path, mode=mode)

    render_online = not args.offline_only
    render_offline = not args.online_only
    if args.offline_only and args.online_only:
        render_online = render_offline = True
    if not render_online and not render_offline:
        render_online = render_offline = True

    fps = float(args.fps) if args.fps else find_fps(root, console)
    if fps <= 0:
        raise ValueError("FPS must be positive.")

    pixel_to_mm, calibration_info = resolve_calibration(root)

    if render_online:
        # Online movement: prefer refined positions, fall back to raw if unavailable
        # Try to use refined online positions by default
        use_refined_online = False
        refined_run_name = None

        # Check for refined online data (use by default if available)
        if "refined_online_runs" in root:
            refined_runs = root["refined_online_runs"]

            # Use specified run, or latest if not specified
            if args.refined_online_run is not None:
                refined_run_name = args.refined_online_run
            else:
                refined_run_name = refined_runs.attrs.get("latest")

            if refined_run_name and refined_run_name in refined_runs:
                console.print("[blue]Building online movement run from refined_online_runs (refined positions)...[/blue]")

                refined_group = refined_runs[refined_run_name]
                console.print(f"[cyan]Using refined online run:[/cyan] {refined_run_name}")

                # Load refined positions from interpolated group (final refined data)
                interp_grp = refined_group["interpolated"]
                frames_all = refined_group["camera_frame_ids"][:]
                positions_refined = interp_grp["positions_px"][:]
                valid_mask_refined = interp_grp["valid_mask"][:]

                # Get source stimulus run for provenance
                stimulus_run_name = refined_group.attrs.get("source_stimulus_run")
                texture_to_camera_scale = refined_group.attrs.get("texture_to_camera_scale", 1.0)

                # Get coordinate space and calibration
                coordinate_space = refined_group.attrs.get("coordinate_space", "camera")
                pixels_per_mm_projector = refined_group.attrs.get("pixels_per_mm_projector")

                # Use projector calibration for texture-space positions
                # (online positions are in texture space, so we need texture-space calibration)
                pixel_to_mm_online = None
                if pixels_per_mm_projector is not None and coordinate_space == "texture":
                    pixel_to_mm_online = float(pixels_per_mm_projector)
                    console.print(f"[cyan]Using projector calibration:[/cyan] {pixel_to_mm_online:.6f} pixels/mm (texture space)")
                else:
                    # Fall back to camera calibration if projector calibration not available
                    pixel_to_mm_online = pixel_to_mm
                    console.print(f"[yellow]Warning:[/yellow] Using camera calibration for online data (projector calibration not found)")

                # Use refined positions (in texture space for accurate distance calculations)
                positions_online = positions_refined
                use_refined_online = True

                console.print(f"  Source stimulus run: {stimulus_run_name}")
                console.print(f"  Coordinate space: {coordinate_space}")
                console.print(f"  Refined frames: {len(frames_all)}")
                console.print(f"  Valid frames: {valid_mask_refined.sum()} ({valid_mask_refined.sum()/len(frames_all)*100:.1f}%)")
            else:
                console.print(f"[yellow]Note:[/yellow] Refined run '{refined_run_name}' not found; using raw online data.")
        else:
            console.print("[yellow]Note:[/yellow] No refined_online_runs found; using raw online data.")

        if not use_refined_online:
            console.print("[blue]Building online movement run from stimulus_runs (H5-imported data)...[/blue]")

            # For raw online data, use camera calibration (positions will be transformed to camera space)
            pixel_to_mm_online = pixel_to_mm

            try:
                bundle = load_chaser_metrics(
                    args.zarr_path,
                    stimulus_run=args.stimulus_run,
                    metrics_run=None,  # Online uses chaser positions from stimulus_runs, not metrics
                    chaser_index=args.chaser_index,
                )
            except Exception as exc:
                console.print(f"[yellow]Warning:[/yellow] Unable to load stimulus run data ({exc}).")
                console.print("[yellow]Skipping online movement run.[/yellow]")
                render_online = False

        if render_online and not use_refined_online:
            frames_all = np.asarray(bundle.camera_frame_ids, dtype=np.int64)

            # Use online target positions (fish/target from H5) for movement tracking
            # These are logged for ALL trial states: PRE, TRAINING, and POST
            target_positions_x = bundle.online.get("target_pos_x")
            target_positions_y = bundle.online.get("target_pos_y")

            if target_positions_x is None or target_positions_y is None:
                console.print("[yellow]Warning:[/yellow] No target position data in stimulus run; skipping online movement.")
                render_online = False
            else:
                target_pos_x = np.asarray(target_positions_x, dtype=np.float64)
                target_pos_y = np.asarray(target_positions_y, dtype=np.float64)

                # Transform online positions from texture space to camera space
                # Online target positions are in texture/arena space (358x358) but need to be
                # in camera space (4512x4512) to match offline positions and get correct distances
                texture_to_camera_scale = 1.0  # default: no transformation
                stimulus_run_name = bundle.provenance.get("stimulus_run")
                if stimulus_run_name:
                    try:
                        import json
                        analysis_group = root.require_group("analysis")
                        stimulus_parent = analysis_group.require_group("stimulus_runs")
                        if stimulus_run_name in stimulus_parent:
                            stim_group = stimulus_parent[stimulus_run_name]
                            coord_transform_raw = stim_group.attrs.get("coordinate_transform")

                            # Parse JSON string to dict if needed
                            coord_transform = None
                            if isinstance(coord_transform_raw, str):
                                try:
                                    coord_transform = json.loads(coord_transform_raw)
                                except json.JSONDecodeError:
                                    console.print("[yellow]Warning:[/yellow] coordinate_transform is not valid JSON.")
                            elif isinstance(coord_transform_raw, dict):
                                coord_transform = coord_transform_raw

                            if coord_transform and "texture_to_camera_scale" in coord_transform:
                                texture_to_camera_scale = float(coord_transform["texture_to_camera_scale"])
                                console.print(f"[cyan]Applying coordinate transformation:[/cyan] texture_to_camera_scale = {texture_to_camera_scale:.6f}")
                            else:
                                console.print("[yellow]Warning:[/yellow] No coordinate_transform/texture_to_camera_scale found in stimulus run; using raw positions.")
                    except Exception as exc:
                        console.print(f"[yellow]Warning:[/yellow] Failed to load coordinate transformation: {exc}")

                # Apply transformation
                target_pos_x = target_pos_x * texture_to_camera_scale
                target_pos_y = target_pos_y * texture_to_camera_scale

                # Create position array - includes PRE, TRAINING, and POST periods
                positions_online = np.column_stack([target_pos_x, target_pos_y])

        if render_online:
            # Get heading from online fields if available, otherwise NaN
            if use_refined_online:
                # Refined data doesn't have heading, use NaN
                heading_online = np.full(frames_all.shape, np.nan, dtype=np.float64)
            else:
                heading_online = bundle.online.get("visual_angle_deg")
                if heading_online is not None:
                    heading_online = np.asarray(heading_online, dtype=np.float64)
                else:
                    heading_online = np.full(frames_all.shape, np.nan, dtype=np.float64)

            # Use ALL frames from stimulus run (PRE + TRAINING + POST periods)
            # No filtering needed since chaser positions are logged for all trial states
            frames_online = frames_all

            # Single track ID for online (chaser)
            detection_ids_online = np.zeros(frames_online.shape[0], dtype=np.int64)
            keypoint_success_online = np.ones(frames_online.shape[0], dtype=bool)

            console.print(f"[blue]Online frames:[/blue] {frames_online.shape[0]} (full experimental session: PRE + TRAINING + POST)")

            tracks_online, summaries_online = build_track_datasets(
                detection_ids=detection_ids_online,
                frames=frames_online,
                positions_px=positions_online,
                headings_deg=heading_online,
                keypoint_success=keypoint_success_online,
                detection_source=None,
                fps=fps,
                smooth_seconds=args.smooth_seconds,
                pixel_to_mm=pixel_to_mm_online,
            )

            if not summaries_online:
                console.print("[yellow]Warning:[/yellow] Online data produced no tracks.")
            else:
                total_px_online, total_mm_online = summarize_to_table(summaries_online, pixel_to_mm_online, console)

                if args.no_write:
                    console.print("[green]Skipping online write (--no-write).[/green]")
                else:
                    run_name, run_group = ensure_movement_run_group(root, args.run_name, run_type="online")
                    ordered_track_ids = save_movement_tracks(run_group, tracks_online, summaries_online)

                    created_at = datetime.now(timezone.utc).isoformat()

                    # Gather git and environment info for provenance
                    git_info = get_git_info()
                    env_info = get_environment_info()

                    if use_refined_online:
                        inputs = {
                            "refined_online_run": refined_run_name,
                            "stimulus_run": stimulus_run_name,
                            "chaser_index": args.chaser_index,
                        }
                        method = "movement_analysis_online_refined"
                        # For refined online data, save the coordinate space and calibration used
                        saved_coordinate_space = coordinate_space
                        saved_pixel_to_mm = pixel_to_mm_online
                    else:
                        inputs = {
                            "stimulus_run": bundle.provenance.get("stimulus_run"),
                            "chaser_index": int(bundle.provenance.get("chaser_index", args.chaser_index)),
                        }
                        method = "movement_analysis_online"
                        # For raw online data, positions are transformed to camera space
                        saved_coordinate_space = "camera" if texture_to_camera_scale != 1.0 else "texture"
                        saved_pixel_to_mm = pixel_to_mm

                    # Build comprehensive provenance record
                    provenance = {
                        "stage": "movement_analysis",
                        "method": method,
                        "command": " ".join(sys.argv),
                        "created_at_utc": created_at,
                        "git": {
                            "commit": git_info.get("commit_hash"),
                            "short": git_info.get("short_hash"),
                            "branch": git_info.get("branch"),
                            "is_dirty": git_info.get("is_dirty"),
                            "remote": git_info.get("remote_url"),
                        },
                        "environment": {
                            "hostname": env_info["platform"].get("hostname"),
                            "python_version": env_info["platform"].get("python_version"),
                            "system": env_info["platform"].get("system"),
                            "release": env_info["platform"].get("release"),
                        },
                        "parameters": {
                            "fps": fps,
                            "smoothing_seconds": args.smooth_seconds,
                            "coordinate_space": saved_coordinate_space,
                            "calibration_used": saved_pixel_to_mm,
                            "texture_to_camera_scale": texture_to_camera_scale,
                        },
                        "inputs": inputs,
                    }

                    run_group.attrs.update(
                        {
                            "method": method,
                            "created_at_utc": created_at,
                            "fps": fps,
                            "smoothing_seconds": args.smooth_seconds,
                            "pixel_to_mm": saved_pixel_to_mm,
                            "calibration": calibration_info,
                            "inputs": inputs,
                            "texture_to_camera_scale": texture_to_camera_scale,
                            "coordinate_space": saved_coordinate_space,
                            "summary": summaries_online,
                            "num_tracks": len(ordered_track_ids),
                            "total_distance_px": total_px_online,
                            "total_distance_mm": total_mm_online if pixel_to_mm_online is not None else float("nan"),
                            "provenance": provenance,
                        }
                    )

                    console.print(
                        f"[green]✓[/green] Saved movement run to [bold]analysis/movement_runs/online/{run_name}[/bold]"
                    )

    if render_offline:
        # Offline movement now uses ALL keypoint frames (entire video recording)
        # This provides a comprehensive movement profile from video start to end
        console.print("[blue]Building offline movement run from all keypoint frames...[/blue]")

        keypoints_offline = resolve_keypoint_group(root, args.keypoint_run, console)
        kp_parent_name = "refined_keypoints_runs" if keypoints_offline.is_refined else "keypoints_runs"

        crop_group_offline = root[f"crop_runs/{keypoints_offline.crop_run}"]
        detection_path_offline = crop_group_offline.attrs.get("source_coords_path")
        if not detection_path_offline:
            raise ValueError(
                f"Crop run '{keypoints_offline.crop_run}' missing 'source_coords_path'; cannot determine detection source."
            )

        detection_offline = resolve_detection_from_path(root, detection_path_offline)
        preferred_detection_offline = prefer_refined_detection(root, detection_offline, console)
        detection_offline = preferred_detection_offline

        detection_group_offline = detection_offline.group
        bbox_norm_offline = detection_group_offline["bbox_norm_coords"][:]
        frame_indices_offline = detection_group_offline["frame_indices"][:].astype(np.int64, copy=False)
        detection_source_offline = (
            detection_group_offline["detection_source"][:]
            if "detection_source" in detection_group_offline
            else None
        )

        heading_offline = keypoints_offline.group["heading"][:]
        if heading_offline.shape[0] != bbox_norm_offline.shape[0]:
            raise ValueError(
                "Offline: Heading array length does not match detection count."
            )

        keypoint_success_offline = (
            keypoints_offline.group["detection_success"][:]
            if "detection_success" in keypoints_offline.group
            else np.ones_like(heading_offline, dtype=bool)
        )

        width_offline, height_offline = resolve_dimensions(root)
        positions_offline = np.empty((bbox_norm_offline.shape[0], 2), dtype=np.float64)
        positions_offline[:, 0] = bbox_norm_offline[:, 0] * width_offline
        positions_offline[:, 1] = bbox_norm_offline[:, 1] * height_offline

        # For offline, use a single track ID (0) for all frames
        detection_ids_offline: Optional[np.ndarray]
        detection_ids_offline_metadata: Dict[str, Optional[str]] = {}

        expected_detect_run = (
            detection_offline.source_detect_run if detection_offline.source_detect_run else detection_offline.run_name
        )

        if detection_offline.is_refined:
            detection_ids_offline, detection_ids_offline_metadata = load_detection_ids(
                root,
                frame_indices_offline.shape[0],
                console,
                expected_detect_run=expected_detect_run,
                expected_refined_run=detection_offline.run_name,
                return_metadata=True,
            )

            assignments_match_refined = (
                detection_ids_offline is not None
                and detection_ids_offline_metadata.get("assignment_source") == "refined_interpolated"
                and detection_ids_offline_metadata.get("source_refined_run") == detection_offline.run_name
            )

            if not assignments_match_refined:
                if detection_ids_offline is not None:
                    console.print(
                        "[yellow]ID assignment metadata does not match refined detections; remapping from raw detections.[/yellow]"
                    )
                refined_parent_group = root[detection_offline.parent_path]
                detection_ids_offline = map_refined_detection_ids(
                    refined_parent_group,
                    frame_indices_offline,
                    bbox_norm_offline,
                    detection_source_offline,
                    root,
                    console,
                )
        else:
            detection_ids_offline, detection_ids_offline_metadata = load_detection_ids(
                root,
                frame_indices_offline.shape[0],
                console,
                expected_detect_run=expected_detect_run,
                return_metadata=True,
            )

        if detection_ids_offline is None:
            console.print(
                "[yellow]Warning:[/yellow] No ID assignments found; treating offline detections as a single track."
            )
            detection_ids_offline = np.zeros(frame_indices_offline.shape[0], dtype=np.int64)
        else:
            detection_ids_offline = detection_ids_offline.astype(np.int64, copy=False)

        console.print(f"[blue]Offline frames:[/blue] {frame_indices_offline.shape[0]} (all keypoint detections)")

        if frame_indices_offline.size == 0:
            console.print("[yellow]Warning:[/yellow] No offline frames available; skipping.")
        else:
            proceed_offline = True
            if args.skip_unassigned:
                valid_mask = detection_ids_offline >= 0
                if not np.any(valid_mask):
                    console.print(
                        "[yellow]Warning:[/yellow] All detections unassigned after filtering; skipping offline movement run."
                    )
                    proceed_offline = False
                else:
                    bbox_norm_offline = bbox_norm_offline[valid_mask]
                    frame_indices_offline = frame_indices_offline[valid_mask]
                    heading_offline = heading_offline[valid_mask]
                    keypoint_success_offline = keypoint_success_offline[valid_mask]
                    detection_ids_offline = detection_ids_offline[valid_mask]
                    positions_offline = positions_offline[valid_mask]
                    if detection_source_offline is not None:
                        detection_source_offline = detection_source_offline[valid_mask]

            if not proceed_offline or frame_indices_offline.size == 0:
                console.print("[yellow]Warning:[/yellow] No offline detections remaining after filtering; skipping.")
            else:
                tracks_offline, summaries_offline = build_track_datasets(
                    detection_ids=detection_ids_offline,
                    frames=frame_indices_offline,
                    positions_px=positions_offline,
                    headings_deg=heading_offline,
                    keypoint_success=keypoint_success_offline,
                    detection_source=detection_source_offline,
                    fps=fps,
                    smooth_seconds=args.smooth_seconds,
                    pixel_to_mm=pixel_to_mm,
                )

                if not summaries_offline:
                    console.print("[yellow]Warning:[/yellow] Offline metrics produced no tracks.")
                else:
                    total_px_offline, total_mm_offline = summarize_to_table(
                        summaries_offline, pixel_to_mm, console
                    )

                    if args.no_write:
                        console.print("[green]Skipping offline write (--no-write).[/green]")
                    else:
                        offline_run_name = args.offline_run_name
                        if not offline_run_name:
                            # Use keypoint run name as basis for offline run name
                            offline_run_name = f"{keypoints_offline.run_name}_movement"

                        offline_run_name, offline_group = ensure_movement_run_group(
                            root,
                            offline_run_name,
                            run_type="offline",
                            overwrite=True,
                        )
                        ordered_ids_offline = save_movement_tracks(
                            offline_group, tracks_offline, summaries_offline
                        )

                        metrics_metadata: Optional[Dict[str, object]] = None
                        swim_bout_mirror: Optional[str] = None
                        try:
                            chaser_bundle = load_chaser_metrics(
                                args.zarr_path,
                                stimulus_run=args.stimulus_run,
                                metrics_run=args.metrics_run,
                                chaser_index=args.chaser_index,
                            )
                        except Exception as exc:
                            console.print(
                                f"[yellow]Warning:[/yellow] Failed to load chaser metrics for offline run: {exc}"
                            )
                            chaser_bundle = None

                        if chaser_bundle is not None:
                            has_offline = chaser_bundle.offline.get("has_offline")
                            has_values = bool(has_offline is not None and np.any(has_offline))
                            if has_values:
                                metrics_metadata = _persist_chaser_metrics_to_run(
                                    offline_group,
                                    chaser_bundle,
                                    fps=fps,
                                    smooth_seconds=args.smooth_seconds,
                                    distance_interp_seconds=args.distance_interpolation_seconds,
                                )
                                run_id = metrics_metadata.get("metrics_run") or "latest"
                                console.print(
                                    f"[cyan]Stored chaser metrics arrays[/cyan] "
                                    f"(analysis/chaser_fish_metrics/{run_id})."
                                )
                            else:
                                console.print(
                                    "[yellow]Warning:[/yellow] Chaser metrics bundle contains no valid offline data; "
                                    "skipping shared metrics write."
                                )

                        try:
                            swim_bout_mirror = _mirror_swim_bouts_to_tracks(
                                root,
                                offline_group,
                                ordered_ids_offline,
                                args.swim_bout_run,
                                console,
                            )
                        except Exception as exc:
                            console.print(
                                f"[yellow]Warning:[/yellow] Failed to mirror swim bouts: {exc}"
                            )

                        created_at = datetime.now(timezone.utc).isoformat()

                        # Gather git and environment info for provenance
                        git_info = get_git_info()
                        env_info = get_environment_info()

                        offline_inputs = {
                            "detection_path": detection_offline.path,
                            "detection_run": detection_offline.run_name,
                            "detection_variant": detection_offline.variant,
                            "source_detect_run": detection_offline.source_detect_run,
                            "keypoint_run": keypoints_offline.run_name,
                            "keypoint_variant": "refined" if keypoints_offline.is_refined else "raw",
                            "base_keypoint_run": keypoints_offline.base_run_name,
                            "crop_run": keypoints_offline.crop_run,
                        }
                        if detection_ids_offline_metadata:
                            offline_inputs["id_assignment_metadata"] = detection_ids_offline_metadata
                        if metrics_metadata:
                            offline_inputs["chaser_metrics"] = metrics_metadata
                        if swim_bout_mirror:
                            offline_inputs["swim_bout_run"] = swim_bout_mirror

                        # Build comprehensive provenance record
                        offline_provenance = {
                            "stage": "movement_analysis",
                            "method": "movement_analysis_offline",
                            "command": " ".join(sys.argv),
                            "created_at_utc": created_at,
                            "git": {
                                "commit": git_info.get("commit_hash"),
                                "short": git_info.get("short_hash"),
                                "branch": git_info.get("branch"),
                                "is_dirty": git_info.get("is_dirty"),
                                "remote": git_info.get("remote_url"),
                            },
                            "environment": {
                                "hostname": env_info["platform"].get("hostname"),
                                "python_version": env_info["platform"].get("python_version"),
                                "system": env_info["platform"].get("system"),
                                "release": env_info["platform"].get("release"),
                            },
                            "parameters": {
                                "fps": fps,
                                "smoothing_seconds": args.smooth_seconds,
                                "distance_interpolation_seconds": args.distance_interpolation_seconds,
                                "coordinate_space": "camera",
                                "calibration_used": pixel_to_mm,
                            },
                            "inputs": offline_inputs,
                        }

                        offline_group.attrs.update(
                            {
                                "method": "movement_analysis_offline",
                                "created_at_utc": created_at,
                                "fps": fps,
                                "smoothing_seconds": args.smooth_seconds,
                                "distance_interpolation_seconds": args.distance_interpolation_seconds,
                                "pixel_to_mm": pixel_to_mm,
                                "calibration": calibration_info,
                                "inputs": offline_inputs,
                                "summary": summaries_offline,
                                "num_tracks": len(ordered_ids_offline),
                                "total_distance_px": total_px_offline,
                                "total_distance_mm": total_mm_offline if pixel_to_mm is not None else float("nan"),
                                "provenance": offline_provenance,
                            }
                        )

                        console.print(
                            f"[green]✓[/green] Saved offline movement run to [bold]analysis/movement_runs/offline/{offline_run_name}[/bold]"
                        )


__all__ = [
    "TrackSpeeds",
    "compute_track_speed",
    "find_fps",
    "load_detection_ids",
    "resolve_dimensions",
    "main",
]


if __name__ == "__main__":  # pragma: no cover
    main()
