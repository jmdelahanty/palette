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
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import zarr
from rich.console import Console
from rich.table import Table

from .compute_speed import (  # re-exported for compatibility
    TrackSpeeds,
    compute_track_speed,
    find_fps,
    load_detection_ids,
    resolve_dimensions,
)
from .chaser_metrics_loader import load_chaser_metrics


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


def resolve_calibration(root: zarr.Group) -> Tuple[Optional[float], Dict[str, Optional[float]]]:
    """Retrieve pixel-to-mm conversion if available."""

    calibration = root.get("calibration")
    if calibration is None:
        return None, {"has_calibration": False, "measured_fps": None}

    pixel_to_mm = calibration.attrs.get("pixel_to_mm")
    pixel_to_mm_val = float(pixel_to_mm) if pixel_to_mm is not None else None
    measured_fps = calibration.attrs.get("measured_fps")
    measured_fps_val = float(measured_fps) if measured_fps is not None else None

    return pixel_to_mm_val, {
        "has_calibration": pixel_to_mm_val is not None,
        "measured_fps": measured_fps_val,
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
        "--chaser-index",
        type=int,
        default=0,
        help="Chaser index for offline metrics (default: 0).",
    )
    parser.add_argument(
        "--offline-run-name",
        help="Optional name for the offline movement run (auto-generated if omitted).",
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
        keypoints = resolve_keypoint_group(root, args.keypoint_run, console)
        kp_parent_name = "refined_keypoints_runs" if keypoints.is_refined else "keypoints_runs"
        kp_parent = root.get(kp_parent_name)
        latest_kp = kp_parent.attrs.get("latest") if kp_parent is not None else None
        console.print(
            f"[blue]Keypoint source:[/blue] {kp_parent_name}/{keypoints.run_name}"
            + (" [dim](latest)[/dim]" if latest_kp == keypoints.run_name else "")
        )
        if latest_kp and latest_kp != keypoints.run_name:
            console.print(f"[yellow]  Note:[/yellow] Latest {kp_parent_name} = {latest_kp}")

        crop_group = root[f"crop_runs/{keypoints.crop_run}"]
        detection_path = crop_group.attrs.get("source_coords_path")
        if not detection_path:
            raise ValueError(
                f"Crop run '{keypoints.crop_run}' missing 'source_coords_path'; cannot determine detection source."
            )

        detection = resolve_detection_from_path(root, detection_path)
        preferred_detection = prefer_refined_detection(root, detection, console)
        detection = preferred_detection
        detection_parent = root.get(detection.parent_path) if detection.parent_path in root else None
        latest_detection = detection_parent.attrs.get("latest") if detection_parent is not None else None
        console.print(
            f"[blue]Detection source:[/blue] {detection.path}"
            + (" [dim](latest)[/dim]" if latest_detection == detection.run_name else "")
        )
        if latest_detection and latest_detection != detection.run_name:
            console.print(f"[yellow]  Note:[/yellow] Latest {detection.parent_path} = {latest_detection}")

        detection_group = detection.group
        bbox_norm = detection_group["bbox_norm_coords"][:]
        frame_indices = detection_group["frame_indices"][:].astype(np.int64, copy=False)
        console.print(
            f"[blue]Detection rows:[/blue] {bbox_norm.shape[0]} | frame_indices len: {frame_indices.size}"
        )

        detection_source_arr = (
            detection_group["detection_source"][:].astype(np.int8, copy=False)
            if "detection_source" in detection_group
            else None
        )

        heading = keypoints.group["heading"][:]
        if heading.shape[0] != bbox_norm.shape[0]:
            raise ValueError(
                "Heading array length does not match detection count. Ensure keypoints run and detection source align."
            )
        console.print(f"[blue]Heading rows:[/blue] {heading.shape[0]}")

        keypoint_success = (
            keypoints.group["detection_success"][:]
            if "detection_success" in keypoints.group
            else np.ones_like(heading, dtype=bool)
        )

        if keypoint_success.shape[0] != heading.shape[0]:
            raise ValueError("Keypoint success array length mismatch with heading array.")
        console.print(f"[blue]Keypoint success rows:[/blue] {keypoint_success.shape[0]}")

        width, height = resolve_dimensions(root)
        positions_px = np.empty((bbox_norm.shape[0], 2), dtype=np.float64)
        positions_px[:, 0] = bbox_norm[:, 0] * width
        positions_px[:, 1] = bbox_norm[:, 1] * height

        detection_ids, id_metadata = load_detection_ids(
            root,
            detect_length=bbox_norm.shape[0],
            console=console,
            expected_detect_run=detection.source_detect_run,
            expected_refined_run=detection.run_name if detection.is_refined else None,
            return_metadata=True,
        )
        console.print(
            f"[blue]ID assignment rows:[/blue] {detection_ids.shape[0]} "
            f"(assign run: {id_metadata.get('assign_run')})"
        )

        if args.skip_unassigned:
            valid_mask = detection_ids >= 0
            detection_ids = detection_ids[valid_mask]
            frame_indices = frame_indices[valid_mask]
            positions_px = positions_px[valid_mask]
            heading = heading[valid_mask]
            keypoint_success = keypoint_success[valid_mask]
            if detection_source_arr is not None:
                detection_source_arr = detection_source_arr[valid_mask]

        if detection_ids.size == 0:
            raise ValueError("No valid detection IDs found after filtering.")

        tracks, summaries = build_track_datasets(
            detection_ids=detection_ids,
            frames=frame_indices,
            positions_px=positions_px,
            headings_deg=heading,
            keypoint_success=keypoint_success,
            detection_source=detection_source_arr,
            fps=fps,
            smooth_seconds=args.smooth_seconds,
            pixel_to_mm=pixel_to_mm,
        )

        if not summaries:
            raise ValueError("No tracks generated; check detection IDs and inputs.")

        total_px, total_mm = summarize_to_table(summaries, pixel_to_mm, console)

        if args.no_write:
            console.print("[green]Skipping write (--no-write).[/green]")
        else:
            run_name, run_group = ensure_movement_run_group(root, args.run_name, run_type="online")
            ordered_track_ids = save_movement_tracks(run_group, tracks, summaries)

            created_at = datetime.now(timezone.utc).isoformat()

            inputs = {
                "detection_path": detection.path,
                "detection_run": detection.run_name,
                "detection_variant": detection.variant,
                "source_detect_run": detection.source_detect_run,
                "keypoint_run": keypoints.run_name,
                "keypoint_variant": "refined" if keypoints.is_refined else "raw",
                "base_keypoint_run": keypoints.base_run_name,
                "crop_run": keypoints.crop_run,
                "id_assignment_run": id_metadata.get("assign_run"),
            }

            run_group.attrs.update(
                {
                    "method": "movement_analysis",
                    "created_at_utc": created_at,
                    "fps": fps,
                    "smoothing_seconds": args.smooth_seconds,
                    "pixel_to_mm": pixel_to_mm,
                    "calibration": calibration_info,
                    "inputs": inputs,
                    "summary": summaries,
                    "num_tracks": len(ordered_track_ids),
                    "total_distance_px": total_px,
                    "total_distance_mm": total_mm if pixel_to_mm is not None else float("nan"),
                }
            )

            console.print(
                f"[green]✓[/green] Saved movement run to [bold]analysis/movement_runs/online/{run_name}[/bold]"
            )

    if render_offline:
        try:
            bundle = load_chaser_metrics(
                args.zarr_path,
                stimulus_run=None,
                metrics_run=args.metrics_run,
                chaser_index=args.chaser_index,
            )
        except Exception as exc:
            console.print(f"[yellow]Warning:[/yellow] Unable to load offline metrics ({exc}).")
        else:
            frames_all = np.asarray(bundle.camera_frame_ids, dtype=np.int64)
            has_offline = np.asarray(bundle.offline.get("has_offline"), dtype=bool)
            fish_positions = np.asarray(bundle.offline.get("fish_centroid_px"), dtype=np.float64)
            heading_offline = bundle.offline.get("heading_deg")
            heading_offline = (
                np.asarray(heading_offline, dtype=np.float64)
                if heading_offline is not None
                else np.full(has_offline.shape, np.nan, dtype=np.float64)
            )

            finite_pos = np.all(np.isfinite(fish_positions), axis=1)
            valid_mask = has_offline & finite_pos

            frames_offline = frames_all[valid_mask]
            if frames_offline.size == 0:
                console.print("[yellow]Warning:[/yellow] Offline metrics contain no valid positions; skipping.")
            else:
                positions_offline = fish_positions[valid_mask]
                heading_values = heading_offline[valid_mask]
                detection_ids_offline = np.zeros(frames_offline.shape[0], dtype=np.int64)
                keypoint_success_offline = np.ones(frames_offline.shape[0], dtype=bool)

                tracks_offline, summaries_offline = build_track_datasets(
                    detection_ids=detection_ids_offline,
                    frames=frames_offline,
                    positions_px=positions_offline,
                    headings_deg=heading_values,
                    keypoint_success=keypoint_success_offline,
                    detection_source=None,
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
                            metrics_run_name = bundle.provenance.get("metrics_run")
                            if metrics_run_name:
                                offline_run_name = f"{metrics_run_name}_movement"

                        offline_run_name, offline_group = ensure_movement_run_group(
                            root,
                            offline_run_name,
                            run_type="offline",
                            overwrite=True,
                        )
                        ordered_ids_offline = save_movement_tracks(
                            offline_group, tracks_offline, summaries_offline
                        )

                        created_at = datetime.now(timezone.utc).isoformat()
                        offline_inputs = {
                            "metrics_run": bundle.provenance.get("metrics_run"),
                            "stimulus_run": bundle.provenance.get("stimulus_run"),
                            "source_keypoints_run": bundle.provenance.get("source_keypoints_run"),
                            "chaser_index": int(bundle.provenance.get("chaser_index", args.chaser_index)),
                        }

                        offline_group.attrs.update(
                            {
                                "method": "movement_analysis_offline",
                                "created_at_utc": created_at,
                                "fps": fps,
                                "smoothing_seconds": args.smooth_seconds,
                                "pixel_to_mm": pixel_to_mm,
                                "calibration": calibration_info,
                                "inputs": offline_inputs,
                                "summary": summaries_offline,
                                "num_tracks": len(ordered_ids_offline),
                                "total_distance_px": total_px_offline,
                                "total_distance_mm": total_mm_offline if pixel_to_mm is not None else float("nan"),
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
