# src/fisheye/analysis/compute_speed.py
"""Compute per-ID speeds from detection runs in a Palette Zarr archive.

Usage (basic)::

    python -m fisheye.analysis.compute_speed /path/to/archive.zarr

The script reads detections (optionally with assigned IDs), converts bounding-box
centres to pixel coordinates, and computes instantaneous and smoothed speeds for
each individual. Results are written to ``analysis/speed_runs/<run_name>`` unless
``--no-write`` is supplied.

Hysteresis Filtering
--------------------
By default, a hysteresis filter removes sub-pixel micro-jitter (detection noise) while
preserving real swims. The filter uses a two-level threshold:

- High threshold (default 2.0 px): displacement must exceed this to enter "moving" state
- Low threshold (default 1.0 px): displacement must stay below this for N consecutive
  frames to exit "moving" state
- Min frames (default 3): consecutive frames below low threshold required to exit

When not in "moving" state, displacements are zeroed and do not contribute to distance
or speed metrics. This prevents noise accumulation without clipping real swims.

To disable the filter, use ``--no-hysteresis``.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import zarr
from rich.console import Console
from rich.table import Table
from scipy.signal import savgol_filter


@dataclass
class TrackSpeeds:
    frames: np.ndarray
    speed_raw: np.ndarray  # Instantaneous speed from raw displacement
    speed_filtered: np.ndarray  # Instantaneous speed from hysteresis-filtered displacement
    speed_smoothed: np.ndarray  # Instantaneous speed from temporally smoothed displacement
    speed_averaged: np.ndarray  # Further temporal averaging of speed
    displacement_raw: np.ndarray  # Frame-to-frame displacement: validity filtering only
    displacement_filtered: np.ndarray  # Frame-to-frame displacement: hysteresis applied
    displacement_smoothed: np.ndarray  # Frame-to-frame displacement: temporal smoothing applied
    cumulative_distance: np.ndarray
    seconds: np.ndarray
    speed_per_second: np.ndarray


def find_fps(root: zarr.Group, console: Console, fallback: float = 60.0) -> float:
    """Extract FPS from root/raw_video attrs; fall back to provided default."""
    for key in ("fps", "video_fps"):
        if key in root.attrs:
            val = float(root.attrs[key])
            if val > 0:
                return val

    if "raw_video" in root and "fps" in root["raw_video"].attrs:
        val = float(root["raw_video"].attrs["fps"])
        if val > 0:
            return val

    console.print(
        f"[yellow]Warning:[/yellow] FPS metadata not found; defaulting to {fallback}."
    )
    return fallback


def resolve_dimensions(root: zarr.Group) -> Tuple[int, int]:
    """Return (width, height) using available metadata."""
    width = root.attrs.get("video_width") or root.attrs.get("width")
    height = root.attrs.get("video_height") or root.attrs.get("height")

    if width and height:
        return int(width), int(height)

    if "raw_video" in root and "images_ds" in root["raw_video"]:
        _, h, w = root["raw_video/images_ds"].shape
        return int(w), int(h)

    raise ValueError("Unable to determine video dimensions from Zarr metadata.")


def load_detection_ids(
    root: zarr.Group,
    detect_length: int,
    console: Console,
    expected_detect_run: Optional[str] = None,
    expected_refined_run: Optional[str] = None,
    return_metadata: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Optional[str]]]]:
    """
    Return detection IDs aligned with the detections.

    When ``return_metadata`` is True, the function returns a tuple of ``(ids, metadata)``
    so callers can inspect the provenance of the assignment.
    """
    def _default_response() -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Optional[str]]]]:
        default_ids = np.zeros(detect_length, dtype=np.int32)
        if return_metadata:
            return default_ids, {
                "assign_run": None,
                "source_detect_run": None,
                "source_refined_run": None,
                "assignment_source": None,
            }
        return default_ids

    if "id_assignment_runs" not in root:
        console.print("[yellow]No ID assignment runs found; treating detections as one track.[/yellow]")
        return _default_response()

    group = root["id_assignment_runs"]
    run = group.attrs.get("latest")
    if not run or f"{run}/detection_ids" not in group:
        console.print("[yellow]ID assignment missing detection_ids; treating detections as one track.[/yellow]")
        return _default_response()

    assign_grp = group[run]
    source_detect = assign_grp.attrs.get("source_detect_run")
    source_refined = assign_grp.attrs.get("source_refined_run")
    assignment_source = assign_grp.attrs.get("assignment_source")
    if expected_detect_run and source_detect and source_detect != expected_detect_run:
        console.print(
            f"[yellow]Warning:[/yellow] Detection run '{expected_detect_run}' differs from ID assignment source '{source_detect}'."
        )
    if expected_refined_run and source_refined and source_refined != expected_refined_run:
        console.print(
            f"[yellow]Warning:[/yellow] Refined run '{expected_refined_run}' differs from ID assignment source '{source_refined}'."
        )

    detection_ids = assign_grp["detection_ids"][:]
    if detection_ids.shape[0] != detect_length:
        raise ValueError(
            f"detection_ids length ({detection_ids.shape[0]}) does not match detection rows ({detect_length})."
        )
    if return_metadata:
        metadata = {
            "assign_run": run,
            "source_detect_run": source_detect,
            "source_refined_run": source_refined,
            "assignment_source": assignment_source,
        }
        return detection_ids, metadata
    return detection_ids


def get_detection_group(
    root: zarr.Group,
    requested: Optional[str],
    console: Console,
) -> Tuple[str, zarr.Group, bool, Optional[str]]:
    """
    Return (run_name, zarr_group, is_refined, source_detect_run).

    If refined detection runs exist, prefer the latest interpolated set.
    """
    refined_parent = root.get("refined_detect_runs")
    refined_latest = None
    if refined_parent is not None:
        refined_latest = refined_parent.attrs.get("latest")

    # Helper to resolve refined run names
    def resolve_refined(name: str) -> Tuple[str, zarr.Group, bool, str]:
        if name not in refined_parent:
            raise ValueError(f"Refined detection run '{name}' not found.")
        refined_group = refined_parent[name]
        if "interpolated" not in refined_group:
            raise ValueError(f"Refined detection run '{name}' missing 'interpolated' subgroup.")
        source_detect = refined_group.attrs.get("source_detect_run")
        return name, refined_group["interpolated"], True, source_detect

    if requested:
        # Allow users to specify refined run via "refined/<run>" or exact name
        if refined_parent is not None and requested in refined_parent:
            return resolve_refined(requested)
        if refined_parent is not None and requested.startswith("refined/"):
            refined_name = requested.split("/", 1)[1]
            return resolve_refined(refined_name)

        if "detect_runs" not in root:
            raise ValueError("Detection stage missing; run detection before computing speeds.")
        detect_parent = root["detect_runs"]
        if requested not in detect_parent:
            raise ValueError(f"Detection run '{requested}' not found.")
        return requested, detect_parent[requested], False, None

    # No run specified: prefer refined if available
    if refined_parent is not None and refined_latest:
        return resolve_refined(refined_latest)

    if "detect_runs" not in root:
        raise ValueError("Detection stage missing; run detection before computing speeds.")

    detect_parent = root["detect_runs"]
    run_name = detect_parent.attrs.get("latest")
    if not run_name:
        raise ValueError("No detection runs available (missing 'latest' attribute).")
    if run_name not in detect_parent:
        raise ValueError(f"Detection run '{run_name}' missing from detect_runs.")
    return run_name, detect_parent[run_name], False, None


def map_refined_detection_ids(
    refined_group: zarr.Group,
    refined_frames: np.ndarray,
    refined_bboxes: np.ndarray,
    detection_source: Optional[np.ndarray],
    root: zarr.Group,
    console: Console,
) -> np.ndarray:
    """
    Map refined detection rows back to detection IDs from the source detect run.

    Interpolated detections inherit IDs from neighbouring detections by forward/back fill.
    """
    source_detect = refined_group.attrs.get("source_detect_run")
    if not source_detect:
        raise ValueError("Refined detection run missing 'source_detect_run' attribute.")

    if "detect_runs" not in root or source_detect not in root["detect_runs"]:
        raise ValueError(f"Source detection run '{source_detect}' not found.")

    detect_group = root["detect_runs"][source_detect]
    orig_frames = detect_group["frame_indices"][:].astype(np.int64, copy=False)
    orig_bboxes = detect_group["bbox_norm_coords"][:]
    detection_ids = load_detection_ids(root, orig_frames.size, console, expected_detect_run=source_detect)

    frame_to_candidates: Dict[int, List[Dict[str, object]]] = {}
    for idx, frame in enumerate(orig_frames):
        key = int(frame)
        frame_to_candidates.setdefault(key, []).append(
            {"bbox": orig_bboxes[idx], "id": int(detection_ids[idx]), "used": False}
        )

    mapped_ids = np.full(refined_frames.shape[0], -1, dtype=np.int32)

    for i, frame in enumerate(refined_frames):
        frame_int = int(frame)
        candidates = frame_to_candidates.get(frame_int, [])
        assigned = False

        if detection_source is None or detection_source[i] == 0:
            for candidate in candidates:
                if candidate["used"]:
                    continue
                if np.allclose(candidate["bbox"], refined_bboxes[i], atol=1e-6):
                    mapped_ids[i] = candidate["id"]
                    candidate["used"] = True
                    assigned = True
                    break
            if not assigned and candidates:
                for candidate in candidates:
                    if not candidate["used"]:
                        mapped_ids[i] = candidate["id"]
                        candidate["used"] = True
                        assigned = True
                        break
        else:
            # interpolated entry; inherit previous ID if possible
            if i > 0 and mapped_ids[i - 1] >= 0:
                mapped_ids[i] = mapped_ids[i - 1]
                assigned = True

        if not assigned and i > 0 and mapped_ids[i] < 0:
            mapped_ids[i] = mapped_ids[i - 1]

    # Back fill any remaining gaps using next known ID
    for i in range(mapped_ids.size - 2, -1, -1):
        if mapped_ids[i] < 0 and mapped_ids[i + 1] >= 0:
            mapped_ids[i] = mapped_ids[i + 1]

    # Replace any still-unassigned entries with -1
    return mapped_ids


def compute_track_speed(
    frames: np.ndarray,
    positions: np.ndarray,
    fps: float,
    smooth_seconds: float,
    distance_smooth_seconds: Optional[float] = None,
    hysteresis_high_px: Optional[float] = None,
    hysteresis_low_px: Optional[float] = None,
    hysteresis_min_frames: Optional[int] = None,
    smoothing_method: str = "moving_average",
    savgol_polyorder: int = 3,
) -> TrackSpeeds:
    """Compute instantaneous and smoothed speeds for a single track.

    Optionally applies hysteresis filtering to remove micro-jitter (sub-pixel noise)
    while preserving real movement. When enabled, displacement must exceed
    hysteresis_high_px to enter "moving" state, and must remain below hysteresis_low_px
    for hysteresis_min_frames consecutive frames to exit "moving" state.

    Displacement smoothing can use either moving average (simple) or Savitzky-Golay
    (shape-preserving polynomial fit). Savitzky-Golay is better for preserving peak
    shapes and derivatives (e.g., acceleration) but is slightly more computationally
    expensive.

    Args:
        smoothing_method: "moving_average" (default) or "savitzky_golay"
        savgol_polyorder: Polynomial order for Savitzky-Golay (default: 3)
    """
    if frames.size == 0:
        empty = np.array([], dtype=np.float32)
        empty_int64 = np.array([], dtype=np.int64)
        return TrackSpeeds(frames, empty, empty, empty, empty, empty, empty, empty, empty, empty_int64, empty)

    order = np.argsort(frames)
    frames = frames[order].astype(np.int64, copy=False)
    positions = positions[order].astype(np.float64, copy=False)

    speed_raw = np.full(frames.shape[0], np.nan, dtype=np.float64)
    speed_filtered = np.full(frames.shape[0], np.nan, dtype=np.float64)
    speed_smoothed = np.full(frames.shape[0], np.nan, dtype=np.float64)
    displacement_raw = np.zeros(frames.shape[0], dtype=np.float64)
    displacement_filtered_data = np.zeros(frames.shape[0], dtype=np.float64)
    displacement_smoothed = np.zeros(frames.shape[0], dtype=np.float64)

    distance_window_seconds = (
        distance_smooth_seconds if distance_smooth_seconds is not None else smooth_seconds
    )
    distance_window = max(1, int(round(fps * distance_window_seconds)))
    speed_window = max(1, int(round(fps * smooth_seconds)))

    if frames.size >= 2:
        delta_frames = np.diff(frames)
        delta_seconds = delta_frames / fps
        displacement = np.linalg.norm(np.diff(positions, axis=0), axis=1)

        # Only consider valid consecutive frames (no gaps) for distance calculation
        # Also filter out extremely large displacements that likely indicate teleportation
        # or data artifacts (e.g., trial state changes, coordinate system glitches)
        consecutive_frames = delta_frames == 1

        # Flag displacements > 500px as suspicious (likely teleportation/reset)
        # At 60fps, a fish moving at 200mm/s (~8in/s, quite fast) would be ~3.3mm/frame
        # In camera pixels at ~51px/mm calibration, that's ~168px/frame maximum realistic
        # Setting threshold at 500px to be conservative
        reasonable_displacement = displacement < 500.0

        valid = (delta_seconds > 0) & np.isfinite(displacement) & consecutive_frames & reasonable_displacement

        # Retain only reasonable, consecutive displacements; others are treated as 0 to avoid
        # adding artificial distance or leaking into the smoothing window.
        displacement_pre_hysteresis = np.zeros_like(displacement)
        displacement_pre_hysteresis[valid] = displacement[valid]

        # Save pre-hysteresis displacement as displacement_raw
        displacement_post_hysteresis = displacement_pre_hysteresis.copy()

        # Apply hysteresis filter to remove micro-jitter (sub-pixel noise) while preserving real movement.
        # The state machine prevents toggling on every noisy sample by requiring sustained low displacement
        # to exit "moving" state, and any high displacement to enter it.
        if hysteresis_high_px is not None and hysteresis_low_px is not None:
            min_frames = hysteresis_min_frames if hysteresis_min_frames is not None else 3
            is_moving = False
            low_count = 0

            for i in range(displacement_post_hysteresis.size):
                if not valid[i]:
                    # Invalid frames don't affect state
                    continue

                current_displacement = displacement_post_hysteresis[i]

                # State transitions
                if current_displacement > hysteresis_high_px:
                    # Enter or maintain "moving" state
                    is_moving = True
                    low_count = 0
                elif current_displacement < hysteresis_low_px:
                    # Increment counter for potential exit from "moving" state
                    low_count += 1
                    if low_count >= min_frames:
                        is_moving = False
                else:
                    # Between low and high: maintain current state, reset counter
                    low_count = 0

                # Zero displacement when not moving
                if not is_moving:
                    displacement_post_hysteresis[i] = 0.0

        if distance_window > 1 and displacement_post_hysteresis.size > 0:
            if smoothing_method == "savitzky_golay":
                # Savitzky-Golay filter for shape-preserving smoothing
                # Ensure window length is odd
                window_length = distance_window if distance_window % 2 == 1 else distance_window + 1
                # Ensure polynomial order is less than window length
                polyorder = min(savgol_polyorder, window_length - 1)

                if window_length >= polyorder + 2:  # Minimum requirement for savgol_filter
                    displacement_post_smoothing = savgol_filter(
                        displacement_post_hysteresis,
                        window_length=window_length,
                        polyorder=polyorder,
                        mode='nearest'  # Handle edges by extending nearest value
                    )
                else:
                    # Window too small for Savitzky-Golay, fall back to moving average
                    kernel = np.ones(distance_window, dtype=np.float64)
                    sum_values = np.convolve(displacement_post_hysteresis, kernel, mode="same")
                    count_values = np.convolve(valid.astype(np.float64), kernel, mode="same")
                    displacement_post_smoothing = np.zeros_like(displacement_post_hysteresis)
                    nonzero = count_values > 0
                    displacement_post_smoothing[nonzero] = sum_values[nonzero] / count_values[nonzero]
            else:
                # Moving average (default method)
                kernel = np.ones(distance_window, dtype=np.float64)
                # Average only over windows that contain at least one valid displacement sample.
                sum_values = np.convolve(displacement_post_hysteresis, kernel, mode="same")
                count_values = np.convolve(valid.astype(np.float64), kernel, mode="same")

                displacement_post_smoothing = np.zeros_like(displacement_post_hysteresis)
                nonzero = count_values > 0
                displacement_post_smoothing[nonzero] = sum_values[nonzero] / count_values[nonzero]
        else:
            displacement_post_smoothing = displacement_post_hysteresis

        # Prevent gaps or invalid transitions from inheriting averaged motion.
        displacement_post_smoothing[~valid] = 0.0

        # Calculate speed_raw from raw displacement (validity filtered only)
        raw_slice = speed_raw[1:]
        raw_slice[valid] = displacement_pre_hysteresis[valid] / delta_seconds[valid]
        speed_raw[1:] = raw_slice

        # Calculate speed_filtered from hysteresis-filtered displacement
        filtered_slice = speed_filtered[1:]
        filtered_slice[valid] = displacement_post_hysteresis[valid] / delta_seconds[valid]
        speed_filtered[1:] = filtered_slice

        # Calculate speed_smoothed from temporally smoothed displacement
        smoothed_slice = speed_smoothed[1:]
        smoothed_slice[valid] = displacement_post_smoothing[valid] / delta_seconds[valid]
        speed_smoothed[1:] = smoothed_slice

        # Store displacement arrays
        displacement_raw[1:] = displacement_pre_hysteresis
        displacement_filtered_data[1:] = displacement_post_hysteresis
        displacement_smoothed[1:] = displacement_post_smoothing

    if speed_window <= 1:
        speed_averaged = speed_smoothed.copy()
    else:
        values = np.nan_to_num(speed_smoothed, nan=0.0, copy=False)
        counts = np.isfinite(speed_smoothed).astype(np.float32)
        kernel = np.ones(speed_window, dtype=np.float32)
        sum_values = np.convolve(values, kernel, mode="same")
        count_values = np.convolve(counts, kernel, mode="same")
        speed_averaged = np.full_like(speed_smoothed, np.nan)
        valid_avg = count_values > 0
        speed_averaged[valid_avg] = sum_values[valid_avg] / count_values[valid_avg]

    # Cumulative sum with NaN values already replaced with 0 in displacement_smoothed
    cumulative_distance = np.cumsum(displacement_smoothed)

    delta_seconds_full = np.zeros(frames.shape[0], dtype=np.float64)
    if frames.size >= 2:
        delta_seconds_full[1:] = np.diff(frames) / fps

    seconds = np.floor(frames / fps).astype(np.int64)
    unique_seconds = np.unique(seconds)
    speed_per_second = np.zeros(unique_seconds.size, dtype=np.float64)

    for idx, second_value in enumerate(unique_seconds):
        mask = seconds == second_value
        time_sum = delta_seconds_full[mask].sum()
        distance_sum = displacement_smoothed[mask].sum()
        if time_sum > 0:
            speed_per_second[idx] = distance_sum / time_sum
        else:
            speed_per_second[idx] = 0.0

    return TrackSpeeds(
        frames=frames,
        speed_raw=speed_raw.astype(np.float32),
        speed_filtered=speed_filtered.astype(np.float32),
        speed_smoothed=speed_smoothed.astype(np.float32),
        speed_averaged=speed_averaged.astype(np.float32),
        displacement_raw=displacement_raw.astype(np.float32),
        displacement_filtered=displacement_filtered_data.astype(np.float32),
        displacement_smoothed=displacement_smoothed.astype(np.float32),
        cumulative_distance=cumulative_distance.astype(np.float32),
        seconds=unique_seconds.astype(np.int64),
        speed_per_second=speed_per_second.astype(np.float32),
    )


def summarize_tracks(tracks: Dict[int, TrackSpeeds]) -> List[Dict[str, float]]:
    """Return per-track summary metrics."""
    summaries: List[Dict[str, float]] = []
    for track_id, data in tracks.items():
        inst = data.speed_raw
        finite = inst[np.isfinite(inst)]
        total_distance = float(data.cumulative_distance[-1]) if data.cumulative_distance.size else 0.0
        per_second = data.speed_per_second
        mean_speed_per_second = float(np.nanmean(per_second)) if per_second.size else float("nan")
        if finite.size == 0:
            summaries.append(
                {
                    "track_id": track_id,
                    "samples": int(data.frames.size),
                    "mean_speed": float("nan"),
                    "median_speed": float("nan"),
                    "max_speed": float("nan"),
                    "total_distance": total_distance,
                    "mean_speed_per_second": mean_speed_per_second,
                }
            )
            continue
        summaries.append(
            {
                "track_id": track_id,
                "samples": int(data.frames.size),
                "mean_speed": float(finite.mean()),
                "median_speed": float(np.median(finite)),
                "max_speed": float(finite.max()),
                "total_distance": total_distance,
                "mean_speed_per_second": mean_speed_per_second,
            }
        )
    return summaries


def ensure_speed_run_group(root: zarr.Group, run_name: Optional[str]) -> Tuple[str, zarr.Group]:
    """Create /analysis/speed_runs/<run_name> (with auto timestamp if needed)."""
    analysis = root.require_group("analysis")
    speed_parent = analysis.require_group("speed_runs")

    if run_name:
        if run_name in speed_parent:
            raise ValueError(f"Speed run '{run_name}' already exists.")
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_name = f"speed_{timestamp}"

    run_group = speed_parent.create_group(run_name)
    speed_parent.attrs["latest"] = run_name
    return run_name, run_group


def save_tracks(
    run_group: zarr.Group,
    tracks: Dict[int, TrackSpeeds],
    summaries: List[Dict[str, float]],
) -> List[int]:
    """Write per-track arrays organised under run_group/tracks and return sorted track IDs."""
    tracks_parent = run_group.create_group("tracks")
    summary_by_id = {int(item["track_id"]): item for item in summaries}

    ordered_ids = sorted(int(track_id) for track_id in tracks.keys())
    track_ids_array = np.asarray(ordered_ids, dtype=np.int32)
    track_ids_chunks = (min(1024, len(ordered_ids)),) if ordered_ids else (1,)
    run_group.create_array("track_ids", data=track_ids_array, chunks=track_ids_chunks, overwrite=True)

    manifest: List[Dict[str, float]] = []

    for track_id in ordered_ids:
        data = tracks[track_id]
        sub = tracks_parent.create_group(f"id_{track_id}")
        chunks = (min(1024, data.frames.size),)
        sub.create_array("frame_indices", data=data.frames, chunks=chunks, overwrite=True)
        sub.create_array("speed_raw", data=data.speed_raw, chunks=chunks, overwrite=True)
        sub.create_array("speed_filtered", data=data.speed_filtered, chunks=chunks, overwrite=True)
        sub.create_array("speed_smoothed", data=data.speed_smoothed, chunks=chunks, overwrite=True)
        sub.create_array("speed_averaged", data=data.speed_averaged, chunks=chunks, overwrite=True)
        sub.create_array("displacement_raw", data=data.displacement_raw, chunks=chunks, overwrite=True)
        sub.create_array("displacement_filtered", data=data.displacement_filtered, chunks=chunks, overwrite=True)
        sub.create_array("displacement_smoothed", data=data.displacement_smoothed, chunks=chunks, overwrite=True)
        sub.create_array(
            "cumulative_distance", data=data.cumulative_distance, chunks=chunks, overwrite=True
        )
        if data.seconds.size:
            seconds_chunks = (min(512, data.seconds.size),)
            sub.create_array("second_indices", data=data.seconds, chunks=seconds_chunks, overwrite=True)
            sub.create_array(
                "speed_per_second",
                data=data.speed_per_second,
                chunks=seconds_chunks,
                overwrite=True,
            )
        else:
            sub.create_array("second_indices", data=np.array([], dtype=np.int64), chunks=(1,), overwrite=True)
            sub.create_array("speed_per_second", data=np.array([], dtype=np.float32), chunks=(1,), overwrite=True)

        summary = summary_by_id.get(track_id)
        if summary is None:
            finite = data.speed_raw[np.isfinite(data.speed_raw)]
            summary = {
                "track_id": track_id,
                "samples": int(data.frames.size),
                "mean_speed": float(finite.mean()) if finite.size else float("nan"),
                "median_speed": float(np.median(finite)) if finite.size else float("nan"),
                "max_speed": float(finite.max()) if finite.size else float("nan"),
                "total_distance": float(data.cumulative_distance[-1]) if data.cumulative_distance.size else 0.0,
                "mean_speed_per_second": float(np.nanmean(data.speed_per_second)) if data.speed_per_second.size else float("nan"),
            }

        sub.attrs.update(
            {
                "num_samples": int(summary.get("samples", data.frames.size)),
                "mean_speed": float(summary.get("mean_speed", float("nan"))),
                "median_speed": float(summary.get("median_speed", float("nan"))),
                "max_speed": float(summary.get("max_speed", float("nan"))),
                "total_distance": float(summary.get("total_distance", 0.0)),
                "mean_speed_per_second": float(summary.get("mean_speed_per_second", float("nan"))),
            }
        )

        manifest.append(
            {
                "track_id": track_id,
                "group": f"tracks/id_{track_id}",
                "num_samples": int(summary.get("samples", data.frames.size)),
                "mean_speed": float(summary.get("mean_speed", float("nan"))),
                "median_speed": float(summary.get("median_speed", float("nan"))),
                "max_speed": float(summary.get("max_speed", float("nan"))),
                "total_distance": float(summary.get("total_distance", 0.0)),
                "mean_speed_per_second": float(summary.get("mean_speed_per_second", float("nan"))),
            }
        )

    run_group.attrs["track_manifest"] = manifest
    return ordered_ids


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Compute per-ID speeds from detections in a Palette Zarr archive.")
    parser.add_argument("zarr_path", help="Path to the Palette Zarr archive.")
    parser.add_argument("--detection-run", help="Detection run to use (defaults to latest, preferring refined runs).")
    parser.add_argument("--smooth-seconds", type=float, default=1.0, help="Speed smoothing window in seconds (moving average, default: 1.0).")
    parser.add_argument(
        "--distance-smooth-seconds",
        type=float,
        default=None,
        help="Displacement smoothing window in seconds before computing instantaneous speed (defaults to --smooth-seconds).",
    )
    parser.add_argument("--skip-unassigned", action="store_true", help="Ignore detections with ID < 0.")
    parser.add_argument("--run-name", help="Optional name for the output speed run.")
    parser.add_argument("--no-write", action="store_true", help="Do not write results back to the Zarr archive.")
    parser.add_argument("--fps", type=float, default=None, help="Override frames-per-second value.")
    parser.add_argument(
        "--hysteresis-high-px",
        type=float,
        default=2.0,
        help="High threshold in pixels for hysteresis filter (enter 'moving' state, default: 2.0).",
    )
    parser.add_argument(
        "--hysteresis-low-px",
        type=float,
        default=1.0,
        help="Low threshold in pixels for hysteresis filter (exit 'moving' state, default: 1.0).",
    )
    parser.add_argument(
        "--hysteresis-min-frames",
        type=int,
        default=3,
        help="Minimum consecutive frames below low threshold to exit 'moving' state (default: 3).",
    )
    parser.add_argument(
        "--no-hysteresis",
        action="store_true",
        help="Disable hysteresis filter (allow all sub-pixel displacements).",
    )
    parser.add_argument(
        "--smoothing-method",
        type=str,
        choices=["moving_average", "savitzky_golay"],
        default="moving_average",
        help="Smoothing method for displacement: 'moving_average' (simple averaging) or 'savitzky_golay' (shape-preserving polynomial fit, better for derivatives) (default: moving_average)",
    )
    parser.add_argument(
        "--savgol-polyorder",
        type=int,
        default=3,
        help="Polynomial order for Savitzky-Golay filter (default: 3, typical for biomechanics). Auto-adjusted if window too small.",
    )
    args = parser.parse_args(argv)

    console = Console()
    mode = "r" if args.no_write else "a"
    root = zarr.open(args.zarr_path, mode=mode)

    detect_run_name, detect_group, is_refined, source_detect_run = get_detection_group(
        root, args.detection_run, console
    )
    console.print(f"[bold]Detection run:[/bold] {detect_run_name}{' (refined)' if is_refined else ''}")

    if "bbox_norm_coords" not in detect_group or "frame_indices" not in detect_group:
        raise ValueError("Selected detection data missing required arrays ('bbox_norm_coords' and 'frame_indices').")

    bbox_norm = detect_group["bbox_norm_coords"][:]
    frame_indices = detect_group["frame_indices"][:].astype(np.int64, copy=False)

    if bbox_norm.size == 0:
        console.print("[yellow]No detections found; nothing to do.[/yellow]")
        return

    detection_source = detect_group["detection_source"][:] if "detection_source" in detect_group else None

    if is_refined:
        detection_ids = None
        id_metadata: Dict[str, Optional[str]] = {}
        try:
            detection_ids, id_metadata = load_detection_ids(
                root,
                bbox_norm.shape[0],
                console,
                expected_detect_run=source_detect_run,
                expected_refined_run=detect_run_name,
                return_metadata=True,
            )
        except ValueError:
            detection_ids = None
            id_metadata = {}

        assignments_match_refined = (
            detection_ids is not None
            and id_metadata.get("assignment_source") == "refined_interpolated"
            and id_metadata.get("source_refined_run") == detect_run_name
        )

        if assignments_match_refined and detection_ids is not None:
            console.print("[green]Using ID assignments stored with the refined detections.[/green]")
        else:
            if detection_ids is not None:
                console.print("[yellow]Stored IDs do not align with refined detections; remapping from raw detections.[/yellow]")
            else:
                console.print("[yellow]No stored refined IDs found; remapping from raw detections.[/yellow]")
            detection_ids = map_refined_detection_ids(
                root["refined_detect_runs"][detect_run_name],
                frame_indices,
                bbox_norm,
                detection_source,
                root,
                console,
            )
    else:
        detection_ids = load_detection_ids(root, bbox_norm.shape[0], console, expected_detect_run=detect_run_name)

    unique_ids = np.unique(detection_ids)
    if args.skip_unassigned:
        unique_ids = unique_ids[unique_ids >= 0]
    if unique_ids.size == 0:
        raise ValueError("No valid detection IDs found after filtering.")

    fps = float(args.fps) if args.fps else find_fps(root, console)
    if fps <= 0:
        raise ValueError("FPS must be positive.")

    width, height = resolve_dimensions(root)
    x = bbox_norm[:, 0] * width
    y = bbox_norm[:, 1] * height
    positions = np.stack([x, y], axis=1)

    tracks: Dict[int, TrackSpeeds] = {}
    distance_smooth_seconds = (
        float(args.distance_smooth_seconds) if args.distance_smooth_seconds is not None else args.smooth_seconds
    )

    # Prepare hysteresis parameters (None if disabled)
    hysteresis_high = None if args.no_hysteresis else args.hysteresis_high_px
    hysteresis_low = None if args.no_hysteresis else args.hysteresis_low_px
    hysteresis_min = None if args.no_hysteresis else args.hysteresis_min_frames

    for det_id in unique_ids:
        mask = detection_ids == det_id
        if not mask.any():
            continue
        frames = frame_indices[mask]
        coords = positions[mask]
        speeds = compute_track_speed(
            frames,
            coords,
            fps=fps,
            smooth_seconds=args.smooth_seconds,
            distance_smooth_seconds=distance_smooth_seconds,
            hysteresis_high_px=hysteresis_high,
            hysteresis_low_px=hysteresis_low,
            hysteresis_min_frames=hysteresis_min,
            smoothing_method=args.smoothing_method,
            savgol_polyorder=args.savgol_polyorder,
        )
        tracks[int(det_id)] = speeds

    summaries = summarize_tracks(tracks)
    total_distance_all = float(sum(row["total_distance"] for row in summaries))

    table = Table(title="Speed summary", show_lines=False)
    table.add_column("Track ID", justify="right")
    table.add_column("Samples", justify="right")
    table.add_column("Mean (px/s)", justify="right")
    table.add_column("Mean/sec (px/s)", justify="right")
    table.add_column("Median (px/s)", justify="right")
    table.add_column("Max (px/s)", justify="right")
    table.add_column("Total Distance (px)", justify="right")
    for row in summaries:
        table.add_row(
            str(row["track_id"]),
            str(row["samples"]),
            f"{row['mean_speed']:.2f}" if np.isfinite(row["mean_speed"]) else "nan",
            f"{row['mean_speed_per_second']:.2f}" if np.isfinite(row["mean_speed_per_second"]) else "nan",
            f"{row['median_speed']:.2f}" if np.isfinite(row["median_speed"]) else "nan",
            f"{row['max_speed']:.2f}" if np.isfinite(row["max_speed"]) else "nan",
            f"{row['total_distance']:.2f}",
        )
    console.print(table)
    console.print(f"Total distance across tracks: {total_distance_all:.2f} px")

    if args.no_write:
        console.print("[green]Skipping write (--no-write).[/green]")
        return

    run_name, run_group = ensure_speed_run_group(root, args.run_name)
    ordered_track_ids = save_tracks(run_group, tracks, summaries)
    run_group.attrs.update(
        {
            "method": "compute_speed",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "detection_run": detect_run_name,
            "detection_variant": "refined" if is_refined else "raw",
            "source_detect_run": source_detect_run if is_refined else detect_run_name,
            "fps": fps,
            "smoothing_seconds": args.smooth_seconds,
            "distance_smoothing_seconds": distance_smooth_seconds,
            "video_width": width,
            "video_height": height,
            "skip_unassigned": bool(args.skip_unassigned),
            "hysteresis_enabled": not args.no_hysteresis,
            "hysteresis_high_px": float(args.hysteresis_high_px) if not args.no_hysteresis else None,
            "hysteresis_low_px": float(args.hysteresis_low_px) if not args.no_hysteresis else None,
            "hysteresis_min_frames": int(args.hysteresis_min_frames) if not args.no_hysteresis else None,
            "smoothing_method": args.smoothing_method,
            "savgol_polyorder": int(args.savgol_polyorder) if args.smoothing_method == "savitzky_golay" else None,
            "command": " ".join(sys.argv),
            "summary": summaries,
            "total_distance_all_tracks": total_distance_all,
        }
    )
    run_group.attrs["num_tracks"] = len(ordered_track_ids)

    console.print(f"[green]✓[/green] Saved speed run to [bold]analysis/speed_runs/{run_name}[/bold]")


if __name__ == "__main__":  # pragma: no cover
    main()
