"""Comprehensive track kinematics aggregation for Palette archives.

This module consolidates detections, arena assignments, keypoint headings, and
calibration metadata into an analysis-friendly layout under
``analysis/track_kinematics_runs``.

It prefers refined keypoints/detections when available, writes per-track
subgroups with rich kinematic metrics, and records provenance back to the
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
    load_arena_ids,
    resolve_dimensions,
)
from .chaser_metrics_loader import load_chaser_metrics
from fisheye.shared.stage_provenance import (
    build_stage_provenance,
    write_stage_provenance,
)
from fisheye.tracking.single_subject_per_arena import load_tracking_ids
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
            "measured_stimulus_fps": None,
            "stimulus_offset_x": None,
            "stimulus_offset_y": None,
            "primary_camera_id": None,
            "camera_offsets": {},
        }

    pixel_to_mm = calibration.attrs.get("pixel_to_mm")
    pixel_to_mm_val = float(pixel_to_mm) if pixel_to_mm is not None else None
    measured_stimulus_fps = calibration.attrs.get("measured_stimulus_fps")
    if measured_stimulus_fps is None:
        measured_stimulus_fps = calibration.attrs.get("measured_fps")
    measured_stimulus_fps_val = (
        float(measured_stimulus_fps) if measured_stimulus_fps is not None else None
    )
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
        "measured_fps": measured_stimulus_fps_val,
        "measured_stimulus_fps": measured_stimulus_fps_val,
        "stimulus_offset_x": stim_offset_x_val,
        "stimulus_offset_y": stim_offset_y_val,
        "primary_camera_id": primary_camera_id_val,
        "camera_offsets": camera_offsets,
    }


def ensure_track_kinematics_run_group(
    root: zarr.Group,
    run_name: Optional[str],
    *,
    run_type: str = "online",
    overwrite: bool = False,
) -> Tuple[str, zarr.Group]:
    """Create /analysis/track_kinematics_runs/<type>/<run_name>."""

    if run_type not in {"online", "offline"}:
        raise ValueError("run_type must be 'online' or 'offline'")

    analysis = root.require_group("analysis")
    track_parent = analysis.require_group("track_kinematics_runs")
    type_parent = track_parent.require_group(run_type)

    if run_name:
        if run_name in type_parent:
            if not overwrite:
                raise ValueError(
                    f"Track kinematics run '{run_name}' already exists under {run_type}."
                )
            del type_parent[run_name]
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        prefix = (
            "track_kinematics"
            if run_type == "online"
            else "track_kinematics_offline"
        )
        run_name = f"{prefix}_{timestamp}"

    run_group = type_parent.create_group(run_name)

    # Update convenience attributes
    track_parent.attrs["latest"] = f"{run_type}/{run_name}"
    attr_key = "latest_online" if run_type == "online" else "latest_offline"
    type_parent.attrs["latest"] = run_name
    track_parent.attrs[attr_key] = run_name

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


def _filter_public_track_rows(
    *,
    track_ids: np.ndarray,
    frames: np.ndarray,
    positions_px: np.ndarray,
    headings_deg: np.ndarray,
    keypoint_success: np.ndarray,
    detection_source: Optional[np.ndarray],
    include_unassigned: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Return row-aligned arrays suitable for public offline track outputs."""

    if include_unassigned:
        return (
            track_ids,
            frames,
            positions_px,
            headings_deg,
            keypoint_success,
            detection_source,
        )

    valid_mask = track_ids >= 0
    if not np.any(valid_mask):
        return (
            track_ids[valid_mask],
            frames[valid_mask],
            positions_px[valid_mask],
            headings_deg[valid_mask],
            keypoint_success[valid_mask],
            detection_source[valid_mask] if detection_source is not None else None,
        )

    return (
        track_ids[valid_mask],
        frames[valid_mask],
        positions_px[valid_mask],
        headings_deg[valid_mask],
        keypoint_success[valid_mask],
        detection_source[valid_mask] if detection_source is not None else None,
    )


def _ordered_track_arena_ids(
    ordered_ids: List[int],
    track_id_to_arena_id: Optional[Dict[int, int]],
) -> Optional[np.ndarray]:
    """Return arena IDs parallel to ordered track IDs for persisted outputs."""

    if not track_id_to_arena_id:
        return None

    unexpected_missing = [
        track_id
        for track_id in ordered_ids
        if track_id >= 0 and track_id not in track_id_to_arena_id
    ]
    if unexpected_missing:
        raise ValueError(
            "Missing arena mapping for persisted track IDs: "
            + ", ".join(str(track_id) for track_id in unexpected_missing)
        )

    return np.asarray(
        [int(track_id_to_arena_id.get(track_id, -1)) for track_id in ordered_ids],
        dtype=np.int32,
    )


def _wrap_heading_delta_degrees(delta_degrees: np.ndarray) -> np.ndarray:
    """Wrap heading deltas into the signed [-180, 180) range."""

    delta = np.asarray(delta_degrees, dtype=np.float64)
    return ((delta + 180.0) % 360.0) - 180.0


def build_track_datasets(
    track_ids: np.ndarray,
    frames: np.ndarray,
    positions_px: np.ndarray,
    headings_deg: np.ndarray,
    keypoint_success: np.ndarray,
    detection_source: Optional[np.ndarray],
    fps: float,
    smooth_seconds: float,
    pixel_to_mm: Optional[float],
    hysteresis_high_px: Optional[float] = None,
    hysteresis_low_px: Optional[float] = None,
    hysteresis_min_frames: Optional[int] = None,
    smoothing_method: str = "moving_average",
    savgol_polyorder: int = 3,
) -> Tuple[Dict[int, Dict[str, np.ndarray]], List[Dict[str, float]]]:
    """Assemble per-track data arrays and summary statistics.

    Optionally applies hysteresis filtering to remove micro-jitter during speed computation.
    Optionally applies Savitzky-Golay smoothing for shape-preserving filtering.
    """

    unique_ids = np.unique(track_ids)
    tracks: Dict[int, Dict[str, np.ndarray]] = {}
    summaries: List[Dict[str, float]] = []

    pixel_to_mm_val = pixel_to_mm if (pixel_to_mm is not None and pixel_to_mm > 0) else None

    for track_id in unique_ids:
        mask = track_ids == track_id
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

        speeds = compute_track_speed(
            track_frames.copy(),
            coords_px.copy(),
            fps=fps,
            smooth_seconds=smooth_seconds,
            hysteresis_high_px=hysteresis_high_px,
            hysteresis_low_px=hysteresis_low_px,
            hysteresis_min_frames=hysteresis_min_frames,
            smoothing_method=smoothing_method,
            savgol_polyorder=savgol_polyorder,
        )

        speed_raw_px = speeds.speed_raw
        speed_filtered_px = speeds.speed_filtered
        speed_smoothed_px = speeds.speed_smoothed
        speed_averaged_px = speeds.speed_averaged
        displacement_raw_px = speeds.displacement_raw
        displacement_filtered_px = speeds.displacement_filtered
        displacement_smoothed_px = speeds.displacement_smoothed
        cumulative_px = speeds.cumulative_distance
        seconds = speeds.seconds
        speed_per_second_px = speeds.speed_per_second

        if pixel_to_mm_val is not None:
            coords_mm = coords_px * pixel_to_mm_val
            speed_raw_mm = speed_raw_px * pixel_to_mm_val
            speed_filtered_mm = speed_filtered_px * pixel_to_mm_val
            speed_smoothed_mm = speed_smoothed_px * pixel_to_mm_val
            speed_averaged_mm = speed_averaged_px * pixel_to_mm_val
            displacement_raw_mm = displacement_raw_px * pixel_to_mm_val
            displacement_filtered_mm = displacement_filtered_px * pixel_to_mm_val
            displacement_smoothed_mm = displacement_smoothed_px * pixel_to_mm_val
            cumulative_mm = cumulative_px * pixel_to_mm_val
            speed_per_second_mm = speed_per_second_px * pixel_to_mm_val
        else:
            coords_mm = _nan_array(coords_px.shape)
            speed_raw_mm = _nan_array(speed_raw_px.shape)
            speed_filtered_mm = _nan_array(speed_filtered_px.shape)
            speed_smoothed_mm = _nan_array(speed_smoothed_px.shape)
            speed_averaged_mm = _nan_array(speed_averaged_px.shape)
            displacement_raw_mm = _nan_array(displacement_raw_px.shape)
            displacement_filtered_mm = _nan_array(displacement_filtered_px.shape)
            displacement_smoothed_mm = _nan_array(displacement_smoothed_px.shape)
            cumulative_mm = _nan_array(cumulative_px.shape)
            speed_per_second_mm = _nan_array(speed_per_second_px.shape)

        heading_rad = np.deg2rad(headings_track)
        heading_valid = np.isfinite(heading_rad)
        time_seconds = track_frames.astype(np.float64) / fps
        seconds_per_frame = np.floor(time_seconds).astype(np.int64)

        delta_seconds_full = np.zeros(track_frames.shape[0], dtype=np.float64)
        if track_frames.size >= 2:
            delta_seconds_full[1:] = np.diff(track_frames) / fps

        delta_heading_degrees = np.full(track_frames.shape[0], np.nan, dtype=np.float64)
        angular_velocity_deg_s = np.full(track_frames.shape[0], np.nan, dtype=np.float64)

        if headings_track.size >= 2:
            delta_heading_step = _wrap_heading_delta_degrees(
                headings_track[1:] - headings_track[:-1]
            )
            turning_valid = (
                np.isfinite(headings_track[1:])
                & np.isfinite(headings_track[:-1])
                & (delta_seconds_full[1:] > 0)
            )
            delta_heading_slice = delta_heading_degrees[1:]
            delta_heading_slice[turning_valid] = delta_heading_step[turning_valid]
            angular_velocity_slice = angular_velocity_deg_s[1:]
            angular_velocity_slice[turning_valid] = (
                delta_heading_step[turning_valid] / delta_seconds_full[1:][turning_valid]
            )

        # Acceleration from smoothed speed profile
        acceleration_px = np.full(speed_smoothed_px.shape, np.nan, dtype=np.float64)
        acceleration_mm = np.full(speed_smoothed_px.shape, np.nan, dtype=np.float64)

        if speed_smoothed_px.size >= 2:
            delta_speed_px = speed_smoothed_px[1:] - speed_smoothed_px[:-1]
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
            "delta_heading_degrees": _float32(delta_heading_degrees),
            "angular_velocity_deg_s": _float32(angular_velocity_deg_s),
            "smoothed_heading_degrees": _float32(smoothed_heading_deg),
            "smoothed_heading_radians": _float32(smoothed_heading_rad),
            "keypoint_success": _boolean(kp_success_track),
            "detection_source": det_source_track.astype(np.int8),
            "speed_raw_px": _float32(speed_raw_px),
            "speed_raw_mm": _float32(speed_raw_mm),
            "speed_filtered_px": _float32(speed_filtered_px),
            "speed_filtered_mm": _float32(speed_filtered_mm),
            "speed_smoothed_px": _float32(speed_smoothed_px),
            "speed_smoothed_mm": _float32(speed_smoothed_mm),
            "speed_averaged_px": _float32(speed_averaged_px),
            "speed_averaged_mm": _float32(speed_averaged_mm),
            "acceleration_px": _float32(acceleration_px),
            "acceleration_mm": _float32(accel_mm),
            "smoothed_acceleration_px": _float32(smoothed_accel_px),
            "smoothed_acceleration_mm": _float32(smoothed_accel_mm),
            "displacement_raw_px": _float32(displacement_raw_px),
            "displacement_raw_mm": _float32(displacement_raw_mm),
            "displacement_filtered_px": _float32(displacement_filtered_px),
            "displacement_filtered_mm": _float32(displacement_filtered_mm),
            "displacement_smoothed_px": _float32(displacement_smoothed_px),
            "displacement_smoothed_mm": _float32(displacement_smoothed_mm),
            "cumulative_distance_px": _float32(cumulative_px),
            "cumulative_distance_mm": _float32(cumulative_mm),
            "second_indices": seconds_per_frame,
            "speed_per_second_px": _float32(speed_per_second_px),
            "speed_per_second_mm": _float32(speed_per_second_mm),
            "heading_per_second_degrees": _float32(heading_per_second_deg),
            "heading_per_second_resultant": heading_per_second_resultant.astype(np.float32),
        }

        # Speed metrics for all processing levels
        # Raw speed (validity filtering only)
        finite_raw = speed_raw_px[np.isfinite(speed_raw_px)]
        mean_speed_raw_px = float(np.mean(finite_raw)) if finite_raw.size else float("nan")
        median_speed_raw_px = float(np.median(finite_raw)) if finite_raw.size else float("nan")
        max_speed_raw_px = float(np.max(finite_raw)) if finite_raw.size else float("nan")
        mean_speed_raw_mm = mean_speed_raw_px * pixel_to_mm_val if pixel_to_mm_val is not None and np.isfinite(mean_speed_raw_px) else float("nan")
        median_speed_raw_mm = median_speed_raw_px * pixel_to_mm_val if pixel_to_mm_val is not None and np.isfinite(median_speed_raw_px) else float("nan")
        max_speed_raw_mm = max_speed_raw_px * pixel_to_mm_val if pixel_to_mm_val is not None and np.isfinite(max_speed_raw_px) else float("nan")

        # Filtered speed (hysteresis applied)
        finite_filtered = speed_filtered_px[np.isfinite(speed_filtered_px)]
        mean_speed_filtered_px = float(np.mean(finite_filtered)) if finite_filtered.size else float("nan")
        median_speed_filtered_px = float(np.median(finite_filtered)) if finite_filtered.size else float("nan")
        max_speed_filtered_px = float(np.max(finite_filtered)) if finite_filtered.size else float("nan")
        mean_speed_filtered_mm = mean_speed_filtered_px * pixel_to_mm_val if pixel_to_mm_val is not None and np.isfinite(mean_speed_filtered_px) else float("nan")
        median_speed_filtered_mm = median_speed_filtered_px * pixel_to_mm_val if pixel_to_mm_val is not None and np.isfinite(median_speed_filtered_px) else float("nan")
        max_speed_filtered_mm = max_speed_filtered_px * pixel_to_mm_val if pixel_to_mm_val is not None and np.isfinite(max_speed_filtered_px) else float("nan")

        # Smoothed speed (temporal smoothing applied)
        finite_smoothed = speed_smoothed_px[np.isfinite(speed_smoothed_px)]
        mean_speed_smoothed_px = float(np.mean(finite_smoothed)) if finite_smoothed.size else float("nan")
        median_speed_smoothed_px = float(np.median(finite_smoothed)) if finite_smoothed.size else float("nan")
        max_speed_smoothed_px = float(np.max(finite_smoothed)) if finite_smoothed.size else float("nan")
        mean_speed_smoothed_mm = mean_speed_smoothed_px * pixel_to_mm_val if pixel_to_mm_val is not None and np.isfinite(mean_speed_smoothed_px) else float("nan")
        median_speed_smoothed_mm = median_speed_smoothed_px * pixel_to_mm_val if pixel_to_mm_val is not None and np.isfinite(median_speed_smoothed_px) else float("nan")
        max_speed_smoothed_mm = max_speed_smoothed_px * pixel_to_mm_val if pixel_to_mm_val is not None and np.isfinite(max_speed_smoothed_px) else float("nan")

        # Averaged speed (further temporal averaging)
        finite_averaged = speed_averaged_px[np.isfinite(speed_averaged_px)]
        mean_speed_averaged_px = float(np.mean(finite_averaged)) if finite_averaged.size else float("nan")
        median_speed_averaged_px = float(np.median(finite_averaged)) if finite_averaged.size else float("nan")
        max_speed_averaged_px = float(np.max(finite_averaged)) if finite_averaged.size else float("nan")
        mean_speed_averaged_mm = mean_speed_averaged_px * pixel_to_mm_val if pixel_to_mm_val is not None and np.isfinite(mean_speed_averaged_px) else float("nan")
        median_speed_averaged_mm = median_speed_averaged_px * pixel_to_mm_val if pixel_to_mm_val is not None and np.isfinite(median_speed_averaged_px) else float("nan")
        max_speed_averaged_mm = max_speed_averaged_px * pixel_to_mm_val if pixel_to_mm_val is not None and np.isfinite(max_speed_averaged_px) else float("nan")

        # Displacement totals for each processing level
        total_displacement_raw_px = float(np.sum(displacement_raw_px)) if displacement_raw_px.size else 0.0
        total_displacement_raw_mm = total_displacement_raw_px * pixel_to_mm_val if pixel_to_mm_val is not None else float("nan")

        total_displacement_filtered_px = float(np.sum(displacement_filtered_px)) if displacement_filtered_px.size else 0.0
        total_displacement_filtered_mm = total_displacement_filtered_px * pixel_to_mm_val if pixel_to_mm_val is not None else float("nan")

        total_displacement_smoothed_px = float(np.sum(displacement_smoothed_px)) if displacement_smoothed_px.size else 0.0
        total_displacement_smoothed_mm = total_displacement_smoothed_px * pixel_to_mm_val if pixel_to_mm_val is not None else float("nan")

        # Cumulative distance (from smoothed displacement)
        total_distance_px = float(cumulative_px[-1]) if cumulative_px.size else 0.0
        total_distance_mm = (
            float(cumulative_mm[-1]) if cumulative_mm.size and pixel_to_mm_val is not None else float("nan")
        )

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
            # Raw speed metrics (validity filtering only)
            "mean_speed_raw_px": mean_speed_raw_px,
            "median_speed_raw_px": median_speed_raw_px,
            "max_speed_raw_px": max_speed_raw_px,
            "mean_speed_raw_mm": mean_speed_raw_mm,
            "median_speed_raw_mm": median_speed_raw_mm,
            "max_speed_raw_mm": max_speed_raw_mm,
            # Filtered speed (hysteresis applied)
            "mean_speed_filtered_px": mean_speed_filtered_px,
            "median_speed_filtered_px": median_speed_filtered_px,
            "max_speed_filtered_px": max_speed_filtered_px,
            "mean_speed_filtered_mm": mean_speed_filtered_mm,
            "median_speed_filtered_mm": median_speed_filtered_mm,
            "max_speed_filtered_mm": max_speed_filtered_mm,
            # Smoothed speed (temporal smoothing applied)
            "mean_speed_smoothed_px": mean_speed_smoothed_px,
            "median_speed_smoothed_px": median_speed_smoothed_px,
            "max_speed_smoothed_px": max_speed_smoothed_px,
            "mean_speed_smoothed_mm": mean_speed_smoothed_mm,
            "median_speed_smoothed_mm": median_speed_smoothed_mm,
            "max_speed_smoothed_mm": max_speed_smoothed_mm,
            # Averaged speed (further temporal averaging)
            "mean_speed_averaged_px": mean_speed_averaged_px,
            "median_speed_averaged_px": median_speed_averaged_px,
            "max_speed_averaged_px": max_speed_averaged_px,
            "mean_speed_averaged_mm": mean_speed_averaged_mm,
            "median_speed_averaged_mm": median_speed_averaged_mm,
            "max_speed_averaged_mm": max_speed_averaged_mm,
            # Speed per second
            "mean_speed_per_second_px": mean_speed_per_second_px,
            "mean_speed_per_second_mm": mean_speed_per_second_mm,
            # Displacement totals
            "total_displacement_raw_px": total_displacement_raw_px,
            "total_displacement_raw_mm": total_displacement_raw_mm,
            "total_displacement_filtered_px": total_displacement_filtered_px,
            "total_displacement_filtered_mm": total_displacement_filtered_mm,
            "total_displacement_smoothed_px": total_displacement_smoothed_px,
            "total_displacement_smoothed_mm": total_displacement_smoothed_mm,
            # Cumulative distance
            "total_distance_px": total_distance_px,
            "total_distance_mm": total_distance_mm,
            # Heading
            "heading_mean_deg": heading_mean_deg,
            "heading_resultant": heading_consistency,
            # Acceleration
            "mean_acceleration_px": mean_accel_px,
            "mean_acceleration_mm": mean_accel_mm,
            "acceleration_std_px": accel_std_px,
            "acceleration_std_mm": accel_std_mm,
            # Other
            "keypoint_success_rate": float(np.mean(kp_success_track)) if kp_success_track.size else float("nan"),
            "duration_seconds": float(time_seconds[-1] - time_seconds[0]) if time_seconds.size > 1 else 0.0,
        }
        summaries.append(summary)

    return tracks, summaries


def save_track_kinematics_tracks(
    run_group: zarr.Group,
    tracks: Dict[int, Dict[str, np.ndarray]],
    summaries: List[Dict[str, float]],
    *,
    track_id_to_arena_id: Optional[Dict[int, int]] = None,
) -> List[int]:
    """Persist per-track data beneath the track kinematics run group."""

    tracks_parent = run_group.create_group("tracks")
    ordered_ids = sorted(int(track_id) for track_id in tracks.keys())
    track_ids_array = np.asarray(ordered_ids, dtype=np.int32)
    chunks = (min(1024, len(ordered_ids)),) if ordered_ids else (1,)
    run_group.create_array("track_ids", data=track_ids_array, chunks=chunks, overwrite=True)
    track_arena_ids = _ordered_track_arena_ids(ordered_ids, track_id_to_arena_id)
    if track_arena_ids is not None:
        run_group.create_array(
            "track_arena_ids",
            data=track_arena_ids,
            chunks=chunks,
            overwrite=True,
        )

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
        subgroup.create_array("delta_heading_degrees", data=data["delta_heading_degrees"], chunks=base_chunk, overwrite=True)
        subgroup.create_array("angular_velocity_deg_s", data=data["angular_velocity_deg_s"], chunks=base_chunk, overwrite=True)
        subgroup.create_array("smoothed_heading_degrees", data=data["smoothed_heading_degrees"], chunks=base_chunk, overwrite=True)
        subgroup.create_array("smoothed_heading_radians", data=data["smoothed_heading_radians"], chunks=base_chunk, overwrite=True)
        subgroup.create_array("keypoint_success", data=data["keypoint_success"], chunks=base_chunk, overwrite=True)
        subgroup.create_array("detection_source", data=data["detection_source"], chunks=base_chunk, overwrite=True)
        subgroup.create_array("speed_raw_px", data=data["speed_raw_px"], chunks=base_chunk, overwrite=True)
        subgroup.create_array("speed_raw_mm", data=data["speed_raw_mm"], chunks=base_chunk, overwrite=True)
        subgroup.create_array("speed_filtered_px", data=data["speed_filtered_px"], chunks=base_chunk, overwrite=True)
        subgroup.create_array("speed_filtered_mm", data=data["speed_filtered_mm"], chunks=base_chunk, overwrite=True)
        subgroup.create_array("speed_smoothed_px", data=data["speed_smoothed_px"], chunks=base_chunk, overwrite=True)
        subgroup.create_array("speed_smoothed_mm", data=data["speed_smoothed_mm"], chunks=base_chunk, overwrite=True)
        subgroup.create_array("speed_averaged_px", data=data["speed_averaged_px"], chunks=base_chunk, overwrite=True)
        subgroup.create_array("speed_averaged_mm", data=data["speed_averaged_mm"], chunks=base_chunk, overwrite=True)
        subgroup.create_array("acceleration_px", data=data["acceleration_px"], chunks=base_chunk, overwrite=True)
        subgroup.create_array("acceleration_mm", data=data["acceleration_mm"], chunks=base_chunk, overwrite=True)
        subgroup.create_array("smoothed_acceleration_px", data=data["smoothed_acceleration_px"], chunks=base_chunk, overwrite=True)
        subgroup.create_array("smoothed_acceleration_mm", data=data["smoothed_acceleration_mm"], chunks=base_chunk, overwrite=True)
        subgroup.create_array("displacement_raw_px", data=data["displacement_raw_px"], chunks=base_chunk, overwrite=True)
        subgroup.create_array("displacement_raw_mm", data=data["displacement_raw_mm"], chunks=base_chunk, overwrite=True)
        subgroup.create_array("displacement_filtered_px", data=data["displacement_filtered_px"], chunks=base_chunk, overwrite=True)
        subgroup.create_array("displacement_filtered_mm", data=data["displacement_filtered_mm"], chunks=base_chunk, overwrite=True)
        subgroup.create_array("displacement_smoothed_px", data=data["displacement_smoothed_px"], chunks=base_chunk, overwrite=True)
        subgroup.create_array("displacement_smoothed_mm", data=data["displacement_smoothed_mm"], chunks=base_chunk, overwrite=True)
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
                "arena_id": (
                    int(track_id_to_arena_id[track_id])
                    if track_id_to_arena_id and track_id in track_id_to_arena_id
                    else None
                ),
                "num_samples": sample_count,
                "summary": summary,
            }
        )

        manifest.append(
            {
                "track_id": int(track_id),
                "arena_id": (
                    int(track_id_to_arena_id[track_id])
                    if track_id_to_arena_id and track_id in track_id_to_arena_id
                    else float("nan")
                ),
                "group": f"tracks/id_{track_id}",
                "samples": sample_count,
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
    """Create or overwrite an array under the track kinematics run group."""

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
    """Write shared chaser metric arrays to the track kinematics run root."""

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

    # Detect hierarchical structure (multi-level bouts)
    speed_levels = ['speed_raw', 'speed_filtered', 'speed_smoothed', 'speed_averaged']
    is_hierarchical = all(level in bout_group for level in speed_levels)

    if is_hierarchical:
        # Mirror all 4 speed levels to separate subgroups
        tracks_parent = run_group["tracks"]
        default_level = bout_group.attrs.get('default_level', 'speed_smoothed')

        for track_id in track_ids:
            track_subgroup = tracks_parent[f"id_{track_id}"].require_group("swim_bouts")

            # Store metadata at track's swim_bouts level
            track_subgroup.attrs.update({
                "source_swim_bout_run": run_name,
                "default_level": default_level,
                "is_hierarchical": True,
            })

            # Mirror each speed level
            for level in speed_levels:
                level_group = bout_group[level]

                if "bouts" not in level_group:
                    console.print(
                        f"[yellow]Warning:[/yellow] Speed level '{level}' in run '{run_name}' lacks 'bouts' dataset."
                    )
                    continue

                bouts_struct = np.asarray(level_group["bouts"])
                columns = _columnar_bout_data(bouts_struct)

                if not columns:
                    continue

                # Create subgroup for this speed level
                level_subgroup = track_subgroup.require_group(level)

                # Clear existing arrays in this level
                for name in list(level_subgroup.array_keys()):
                    del level_subgroup[name]

                # Write bout data
                for name, array in columns.items():
                    level_subgroup.create_array(
                        name,
                        data=array,
                        chunks=(max(1, min(4096, array.shape[0])),),
                        overwrite=True,
                    )

                level_subgroup.attrs.update({
                    "speed_level": level,
                    "n_bouts": len(bouts_struct),
                    "mirrored_fields": list(columns.keys()),
                })

        console.print(
            f"[dim]Mirrored hierarchical swim bouts (4 levels) from swim_bout_runs/{run_name} into track kinematics tracks.[/dim]"
        )
        return run_name

    else:
        # Legacy flat structure - mirror as before
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
                    "is_hierarchical": False,
                }
            )

        console.print(
            f"[dim]Mirrored swim bouts from swim_bout_runs/{run_name} into track kinematics tracks.[/dim]"
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
    # Speed metrics for all processing levels (mm/s)
    table.add_column("Mean raw mm/s", justify="right")
    table.add_column("Mean filt mm/s", justify="right")
    table.add_column("Mean smooth mm/s", justify="right")
    table.add_column("Mean avg mm/s", justify="right")
    # Displacement totals (mm)
    table.add_column("Disp raw mm", justify="right")
    table.add_column("Disp filt mm", justify="right")
    table.add_column("Disp smooth mm", justify="right")
    table.add_column("Cumul dist mm", justify="right")
    # Other metrics
    table.add_column("Heading (deg)", justify="right")
    table.add_column("Head result", justify="right")
    table.add_column("Accel mm/s²", justify="right")

    total_px = 0.0
    total_mm = 0.0
    total_disp_raw_mm = 0.0
    total_disp_filt_mm = 0.0
    total_disp_smooth_mm = 0.0

    for row in summaries:
        total_px += float(row.get("total_distance_px", 0.0))
        dist_mm = row.get("total_distance_mm", float("nan"))
        if not math.isnan(dist_mm):
            total_mm += float(dist_mm)

        # Track displacement totals
        for key, var in [("total_displacement_raw_mm", "total_disp_raw_mm"),
                         ("total_displacement_filtered_mm", "total_disp_filt_mm"),
                         ("total_displacement_smoothed_mm", "total_disp_smooth_mm")]:
            val = row.get(key, float("nan"))
            if not math.isnan(val):
                if var == "total_disp_raw_mm":
                    total_disp_raw_mm += float(val)
                elif var == "total_disp_filt_mm":
                    total_disp_filt_mm += float(val)
                elif var == "total_disp_smooth_mm":
                    total_disp_smooth_mm += float(val)

        table.add_row(
            str(int(row["track_id"])),
            str(int(row["samples"])),
            f"{row['mean_speed_raw_mm']:.2f}" if np.isfinite(row["mean_speed_raw_mm"]) else "nan",
            f"{row['mean_speed_filtered_mm']:.2f}" if np.isfinite(row["mean_speed_filtered_mm"]) else "nan",
            f"{row['mean_speed_smoothed_mm']:.2f}" if np.isfinite(row["mean_speed_smoothed_mm"]) else "nan",
            f"{row['mean_speed_averaged_mm']:.2f}" if np.isfinite(row["mean_speed_averaged_mm"]) else "nan",
            f"{row['total_displacement_raw_mm']:.2f}" if np.isfinite(row["total_displacement_raw_mm"]) else "nan",
            f"{row['total_displacement_filtered_mm']:.2f}" if np.isfinite(row["total_displacement_filtered_mm"]) else "nan",
            f"{row['total_displacement_smoothed_mm']:.2f}" if np.isfinite(row["total_displacement_smoothed_mm"]) else "nan",
            f"{row['total_distance_mm']:.2f}" if np.isfinite(row["total_distance_mm"]) else "nan",
            f"{row['heading_mean_deg']:.2f}" if np.isfinite(row["heading_mean_deg"]) else "nan",
            f"{row['heading_resultant']:.2f}" if np.isfinite(row["heading_resultant"]) else "nan",
            f"{row['mean_acceleration_mm']:.2f}" if np.isfinite(row["mean_acceleration_mm"]) else "nan",
        )

    console.print(table)
    if pixel_to_mm is not None:
        console.print(f"Total cumulative distance: {total_px:.2f} px ({total_mm:.2f} mm)")
        console.print(f"Total displacement (raw): {total_disp_raw_mm:.2f} mm")
        console.print(f"Total displacement (filtered): {total_disp_filt_mm:.2f} mm")
        console.print(f"Total displacement (smoothed): {total_disp_smooth_mm:.2f} mm")
    else:
        console.print(f"Total cumulative distance: {total_px:.2f} px")
    return total_px, total_mm if pixel_to_mm is not None else float("nan")


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Create track_kinematics_runs entries consolidating detections, IDs, keypoints, and calibration.",
    )
    parser.add_argument("zarr_path", help="Path to the Palette Zarr archive.")
    parser.add_argument(
        "--keypoint-run",
        help="Keypoint run to use. Prefix with 'refined/' to target a refined run. Default: latest refined if available.",
    )
    parser.add_argument(
        "--run-name", help="Optional name for the output track kinematics run."
    )
    parser.add_argument("--smooth-seconds", type=float, default=1.0, help="Smoothing window in seconds (default: 1.0).")
    parser.add_argument(
        "--distance-interpolation-seconds",
        type=float,
        default=0.0,
        help="Maximum gap duration (seconds) to fill via linear interpolation for chaser distances (default: 0).",
    )
    parser.add_argument(
        "--include-unassigned",
        action="store_true",
        help="Include track_id < 0 rows in offline outputs for diagnostic use.",
    )
    parser.add_argument("--fps", type=float, default=None, help="Override frames-per-second value.")
    parser.add_argument("--no-write", action="store_true", help="Do not write results back to the Zarr archive.")
    parser.add_argument(
        "--offline-only",
        action="store_true",
        help="Skip detection-based track kinematics run; only compute offline metrics run.",
    )
    parser.add_argument(
        "--online-only",
        action="store_true",
        help="Only compute detection-based track kinematics run, skipping offline metrics.",
    )
    parser.add_argument(
        "--metrics-run",
        help="Specific analysis/chaser_fish_metrics/<run> to use for offline track kinematics (default: latest).",
    )
    parser.add_argument(
        "--swim-bout-run",
        help="analysis/swim_bout_runs/<run> to mirror into the offline track kinematics run (default: latest).",
    )
    parser.add_argument(
        "--chaser-index",
        type=int,
        default=0,
        help="Chaser index for offline metrics (default: 0).",
    )
    parser.add_argument(
        "--offline-run-name",
        help="Optional name for the offline track kinematics run (auto-generated if omitted).",
    )
    parser.add_argument(
        "--stimulus-run",
        help="Stimulus run name to filter online track kinematics data to the experimental period (default: latest).",
    )
    parser.add_argument(
        "--refined-online-run",
        help="Use refined online positions from refined_online_runs/<run> instead of raw online data (default: None).",
    )
    parser.add_argument(
        "--hysteresis-high-px",
        type=float,
        default=2.0,
        help="High threshold in pixels for hysteresis filter in offline analysis (enter 'moving' state, default: 2.0).",
    )
    parser.add_argument(
        "--hysteresis-low-px",
        type=float,
        default=1.0,
        help="Low threshold in pixels for hysteresis filter in offline analysis (exit 'moving' state, default: 1.0).",
    )
    parser.add_argument(
        "--hysteresis-min-frames",
        type=int,
        default=3,
        help="Minimum consecutive frames below low threshold to exit 'moving' state in offline analysis (default: 3).",
    )
    parser.add_argument(
        "--no-hysteresis",
        action="store_true",
        help="Disable hysteresis filter in offline analysis (allow all sub-pixel displacements).",
    )
    parser.add_argument(
        "--smoothing-method",
        type=str,
        choices=["moving_average", "savitzky_golay"],
        default="moving_average",
        help="Smoothing method for displacement in offline analysis: 'moving_average' (simple averaging) or 'savitzky_golay' (shape-preserving polynomial fit, better for derivatives) (default: moving_average)",
    )
    parser.add_argument(
        "--savgol-polyorder",
        type=int,
        default=3,
        help="Polynomial order for Savitzky-Golay filter in offline analysis (default: 3, typical for biomechanics). Auto-adjusted if window too small.",
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
        # Online track kinematics: prefer refined positions, fall back to raw if unavailable
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
                console.print("[blue]Building online track kinematics run from refined_online_runs (refined positions)...[/blue]")

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
            console.print("[blue]Building online track kinematics run from stimulus_runs (H5-imported data)...[/blue]")

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
                console.print("[yellow]Skipping online track kinematics run.[/yellow]")
                render_online = False

        if render_online and not use_refined_online:
            frames_all = np.asarray(bundle.camera_frame_ids, dtype=np.int64)

            # Use online target positions (fish/target from H5) for movement tracking
            # These are logged for ALL trial states: PRE, TRAINING, and POST
            target_positions_x = bundle.online.get("target_pos_x")
            target_positions_y = bundle.online.get("target_pos_y")

            if target_positions_x is None or target_positions_y is None:
                console.print("[yellow]Warning:[/yellow] No target position data in stimulus run; skipping online track kinematics.")
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
            track_ids_online = np.zeros(frames_online.shape[0], dtype=np.int64)
            keypoint_success_online = np.ones(frames_online.shape[0], dtype=bool)

            console.print(f"[blue]Online frames:[/blue] {frames_online.shape[0]} (full experimental session: PRE + TRAINING + POST)")

            tracks_online, summaries_online = build_track_datasets(
                track_ids=track_ids_online,
                frames=frames_online,
                positions_px=positions_online,
                headings_deg=heading_online,
                keypoint_success=keypoint_success_online,
                detection_source=None,
                fps=fps,
                smooth_seconds=args.smooth_seconds,
                pixel_to_mm=pixel_to_mm_online,
                smoothing_method=args.smoothing_method,
                savgol_polyorder=args.savgol_polyorder,
            )

            if not summaries_online:
                console.print("[yellow]Warning:[/yellow] Online data produced no tracks.")
            else:
                total_px_online, total_mm_online = summarize_to_table(summaries_online, pixel_to_mm_online, console)

                if args.no_write:
                    console.print("[green]Skipping online write (--no-write).[/green]")
                else:
                    run_name, run_group = ensure_track_kinematics_run_group(root, args.run_name, run_type="online")
                    ordered_track_ids = save_track_kinematics_tracks(run_group, tracks_online, summaries_online)

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
                        method = "track_kinematics_online_refined"
                        # For refined online data, save the coordinate space and calibration used
                        saved_coordinate_space = coordinate_space
                        saved_pixel_to_mm = pixel_to_mm_online
                    else:
                        inputs = {
                            "stimulus_run": bundle.provenance.get("stimulus_run"),
                            "chaser_index": int(bundle.provenance.get("chaser_index", args.chaser_index)),
                        }
                        method = "track_kinematics_online"
                        # For raw online data, positions are transformed to camera space
                        saved_coordinate_space = "camera" if texture_to_camera_scale != 1.0 else "texture"
                        saved_pixel_to_mm = pixel_to_mm

                    # Canonical stage provenance.
                    online_params = {
                        "fps": fps,
                        "smoothing_seconds": args.smooth_seconds,
                        "smoothing_method": args.smoothing_method,
                        "savgol_polyorder": int(args.savgol_polyorder) if args.smoothing_method == "savitzky_golay" else None,
                        "coordinate_space": saved_coordinate_space,
                        "calibration_used": saved_pixel_to_mm,
                        "texture_to_camera_scale": texture_to_camera_scale,
                    }
                    provenance = build_stage_provenance(
                        stage="track_kinematics",
                        created_at_utc=created_at,
                        parameters=online_params,
                        inputs=inputs,
                        command=" ".join(sys.argv),
                        git=git_info,
                        environment=env_info.get("platform"),
                    )
                    write_stage_provenance(run_group, provenance)

                    # Backward-compatible top-level attrs.
                    run_group.attrs.update(
                        {
                            "method": method,
                            "created_at_utc": created_at,
                            "fps": fps,
                            "smoothing_seconds": args.smooth_seconds,
                            "smoothing_method": args.smoothing_method,
                            "savgol_polyorder": int(args.savgol_polyorder) if args.smoothing_method == "savitzky_golay" else None,
                            "pixel_to_mm": saved_pixel_to_mm,
                            "calibration": calibration_info,
                            "inputs": inputs,
                            "texture_to_camera_scale": texture_to_camera_scale,
                            "coordinate_space": saved_coordinate_space,
                            "summary": summaries_online,
                            "num_tracks": len(ordered_track_ids),
                            "total_distance_px": total_px_online,
                            "total_distance_mm": total_mm_online if pixel_to_mm_online is not None else float("nan"),
                        }
                    )

                    console.print(
                        f"[green]✓[/green] Saved track kinematics run to [bold]analysis/track_kinematics_runs/online/{run_name}[/bold]"
                    )

    if render_offline:
        # Offline track kinematics now uses all keypoint frames across the video.
        console.print("[blue]Building offline track kinematics run from all keypoint frames...[/blue]")

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

        expected_detect_run = (
            detection_offline.source_detect_run if detection_offline.source_detect_run else detection_offline.run_name
        )
        track_ids_offline, tracking_metadata = load_tracking_ids(
            root,
            frame_indices_offline.shape[0],
            expected_detect_run=expected_detect_run,
            expected_refined_run=detection_offline.run_name if detection_offline.is_refined else None,
            return_metadata=True,
        )
        track_ids_offline = track_ids_offline.astype(np.int64, copy=False)
        track_id_to_arena_id = {
            int(track_id): int(arena_id)
            for track_id, arena_id in (
                tracking_metadata.get("track_id_to_arena_id", {}) or {}
            ).items()
        }

        console.print(f"[blue]Offline frames:[/blue] {frame_indices_offline.shape[0]} (all keypoint detections)")

        if frame_indices_offline.size == 0:
            console.print("[yellow]Warning:[/yellow] No offline frames available; skipping.")
        else:
            proceed_offline = True
            (
                track_ids_offline,
                frame_indices_offline,
                positions_offline,
                heading_offline,
                keypoint_success_offline,
                detection_source_offline,
            ) = _filter_public_track_rows(
                track_ids=track_ids_offline,
                frames=frame_indices_offline,
                positions_px=positions_offline,
                headings_deg=heading_offline,
                keypoint_success=keypoint_success_offline,
                detection_source=detection_source_offline,
                include_unassigned=args.include_unassigned,
            )
            if frame_indices_offline.size == 0:
                console.print(
                    "[yellow]Warning:[/yellow] All offline detections are unassigned; skipping public offline track kinematics run."
                )
                proceed_offline = False

            if not proceed_offline or frame_indices_offline.size == 0:
                console.print("[yellow]Warning:[/yellow] No offline detections remaining after filtering; skipping.")
            else:
                # Prepare hysteresis parameters for offline analysis
                hysteresis_high = None if args.no_hysteresis else args.hysteresis_high_px
                hysteresis_low = None if args.no_hysteresis else args.hysteresis_low_px
                hysteresis_min = None if args.no_hysteresis else args.hysteresis_min_frames

                tracks_offline, summaries_offline = build_track_datasets(
                    track_ids=track_ids_offline,
                    frames=frame_indices_offline,
                    positions_px=positions_offline,
                    headings_deg=heading_offline,
                    keypoint_success=keypoint_success_offline,
                    detection_source=detection_source_offline,
                    fps=fps,
                    smooth_seconds=args.smooth_seconds,
                    pixel_to_mm=pixel_to_mm,
                    hysteresis_high_px=hysteresis_high,
                    hysteresis_low_px=hysteresis_low,
                    hysteresis_min_frames=hysteresis_min,
                    smoothing_method=args.smoothing_method,
                    savgol_polyorder=args.savgol_polyorder,
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
                            # Use keypoint run name as basis for the offline run name.
                            offline_run_name = (
                                f"{keypoints_offline.run_name}_track_kinematics"
                            )

                        offline_run_name, offline_group = ensure_track_kinematics_run_group(
                            root,
                            offline_run_name,
                            run_type="offline",
                            overwrite=True,
                        )
                        ordered_ids_offline = save_track_kinematics_tracks(
                            offline_group,
                            tracks_offline,
                            summaries_offline,
                            track_id_to_arena_id=track_id_to_arena_id,
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
                            "source_tracking_run": tracking_metadata.get("track_run"),
                            "source_arena_assignment_run": tracking_metadata.get("source_arena_assignment_run"),
                        }
                        if tracking_metadata:
                            offline_inputs["tracking_metadata"] = tracking_metadata
                        if metrics_metadata:
                            offline_inputs["chaser_metrics"] = metrics_metadata
                        if swim_bout_mirror:
                            offline_inputs["swim_bout_run"] = swim_bout_mirror

                        # Canonical stage provenance.
                        offline_params = {
                            "fps": fps,
                            "smoothing_seconds": args.smooth_seconds,
                            "smoothing_method": args.smoothing_method,
                            "savgol_polyorder": int(args.savgol_polyorder) if args.smoothing_method == "savitzky_golay" else None,
                            "distance_interpolation_seconds": args.distance_interpolation_seconds,
                            "coordinate_space": "camera",
                            "calibration_used": pixel_to_mm,
                            "hysteresis_enabled": not args.no_hysteresis,
                            "hysteresis_high_px": float(args.hysteresis_high_px) if not args.no_hysteresis else None,
                            "hysteresis_low_px": float(args.hysteresis_low_px) if not args.no_hysteresis else None,
                            "hysteresis_min_frames": int(args.hysteresis_min_frames) if not args.no_hysteresis else None,
                        }
                        offline_provenance = build_stage_provenance(
                            stage="track_kinematics",
                            created_at_utc=created_at,
                            parameters=offline_params,
                            inputs=offline_inputs,
                            command=" ".join(sys.argv),
                            git=git_info,
                            environment=env_info.get("platform"),
                        )
                        write_stage_provenance(offline_group, offline_provenance)

                        # Backward-compatible top-level attrs.
                        offline_group.attrs.update(
                            {
                                "method": "track_kinematics_offline",
                                "created_at_utc": created_at,
                                "fps": fps,
                                "smoothing_seconds": args.smooth_seconds,
                                "smoothing_method": args.smoothing_method,
                                "savgol_polyorder": int(args.savgol_polyorder) if args.smoothing_method == "savitzky_golay" else None,
                                "distance_interpolation_seconds": args.distance_interpolation_seconds,
                                "pixel_to_mm": pixel_to_mm,
                                "calibration": calibration_info,
                                "hysteresis_enabled": not args.no_hysteresis,
                                "hysteresis_high_px": float(args.hysteresis_high_px) if not args.no_hysteresis else None,
                                "hysteresis_low_px": float(args.hysteresis_low_px) if not args.no_hysteresis else None,
                                "hysteresis_min_frames": int(args.hysteresis_min_frames) if not args.no_hysteresis else None,
                                "source_tracking_run": tracking_metadata.get("track_run"),
                                "source_arena_assignment_run": tracking_metadata.get("source_arena_assignment_run"),
                                "inputs": offline_inputs,
                                "summary": summaries_offline,
                                "num_tracks": len(ordered_ids_offline),
                                "total_distance_px": total_px_offline,
                                "total_distance_mm": total_mm_offline if pixel_to_mm is not None else float("nan"),
                            }
                        )

                        console.print(
                            f"[green]✓[/green] Saved offline track kinematics run to [bold]analysis/track_kinematics_runs/offline/{offline_run_name}[/bold]"
                        )


__all__ = [
    "TrackSpeeds",
    "_ordered_track_arena_ids",
    "compute_track_speed",
    "find_fps",
    "_filter_public_track_rows",
    "load_arena_ids",
    "resolve_dimensions",
    "main",
]


if __name__ == "__main__":  # pragma: no cover
    main()
