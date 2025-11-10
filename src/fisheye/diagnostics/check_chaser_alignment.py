#!/usr/bin/env python3
"""Check chaser position alignment against stimulus mappings."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import zarr


def _resolve_run(root: zarr.Group, name: Optional[str]) -> tuple[zarr.Group, str]:
    analysis = root.get("analysis")
    if analysis is None or "stimulus_runs" not in analysis:
        raise KeyError("analysis/stimulus_runs missing.")
    runs = analysis["stimulus_runs"]
    if name:
        if name not in runs:
            raise KeyError(f"Stimulus run '{name}' not found.")
        return runs[name], name
    latest = runs.attrs.get("latest")
    if latest and latest in runs:
        return runs[latest], latest
    keys = sorted(runs.group_keys())
    if not keys:
        raise KeyError("No stimulus runs present.")
    return runs[keys[-1]], keys[-1]


def _safe_column(group: zarr.Group, name: str) -> np.ndarray:
    if name not in group:
        raise KeyError(f"Dataset '{name}' missing from {group.path}.")
    return group[name][:]


def _project_timeline(
    camera_ids: np.ndarray,
    mapping: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, int]:
    times = np.full(camera_ids.shape, np.nan, dtype=np.float64)
    if mapping is None:
        return times, 0
    valid_camera = np.isfinite(camera_ids)
    cams = camera_ids[valid_camera].astype(np.int64, copy=False)
    hits = (cams >= 0) & (cams < mapping.shape[0])
    cams = cams[hits]
    if cams.size == 0:
        return times, 0
    meta_idx = mapping[cams]
    good = meta_idx >= 0
    if not np.any(good):
        return times, 0
    indices = np.where(valid_camera)[0][hits][good]
    times[indices] = values[meta_idx[good]]
    return times, indices.size


def _stats(label: str, camera_ids: np.ndarray, mapping: np.ndarray, values: np.ndarray, unit: str) -> None:
    valid_camera = np.isfinite(camera_ids)
    total_with_cams = int(valid_camera.sum())
    times, assigned = _project_timeline(camera_ids, mapping, values)
    print(f"\n[{label}]")
    print(f"  Total chaser samples: {camera_ids.size}")
    print(f"  Samples with camera IDs: {total_with_cams} ({total_with_cams/camera_ids.size:.2%})")
    print(f"  Samples mapped to metadata: {assigned} ({assigned/camera_ids.size:.2%})")
    print(f"  Samples with missing mapping: {camera_ids.size - assigned}")
    finite = np.isfinite(times)
    monotonic = np.all(np.diff(times[finite]) >= 0) if finite.sum() > 1 else True
    print(f"  Monotonic {unit}: {monotonic}")
    if finite.sum() > 1:
        diffs = np.diff(times[finite])
        print(
            f"  Δ{unit} stats: min {diffs.min():.6f} | median {np.median(diffs):.6f} | max {diffs.max():.6f}"
        )


def _build_camera_to_stimulus_map(
    camera_frames: np.ndarray,
    stimulus_frames: np.ndarray,
) -> np.ndarray:
    """Build a direct camera frame → stimulus frame mapping array."""
    max_cam = int(camera_frames.max()) if len(camera_frames) > 0 else 0
    cam_to_stim = np.full(max_cam + 1, -1, dtype=np.int64)
    for cam, stim in zip(camera_frames, stimulus_frames):
        if 0 <= cam <= max_cam:
            cam_to_stim[cam] = stim
    return cam_to_stim


def _analyze_drift(
    camera_frames: np.ndarray,
    stimulus_frames: np.ndarray,
    *,
    ratio: float,
    threshold: float,
) -> None:
    """Analyze drift between camera and stimulus frame numbering."""
    if ratio <= 0:
        print("\n[Stimulus drift]\n  Ratio must be > 0; skipping drift analysis.")
        return

    camera_to_stimulus = _build_camera_to_stimulus_map(camera_frames, stimulus_frames)
    mapping = camera_to_stimulus.astype(np.float64, copy=False)
    valid = mapping >= 0
    cam_indices = np.nonzero(valid)[0]

    if cam_indices.size < 2:
        print("\n[Stimulus drift]\n  Not enough valid camera frames to evaluate drift.")
        return

    first_cam = float(cam_indices[0])
    first_stim = mapping[cam_indices[0]]
    last_cam = float(cam_indices[-1])
    last_stim = mapping[cam_indices[-1]]
    observed_ratio = (last_stim - first_stim) / max(1.0, (last_cam - first_cam))
    expected = first_stim + (cam_indices - first_cam) * ratio
    drift = mapping[cam_indices] - expected
    abs_drift = np.abs(drift)

    max_idx = int(np.nanargmax(abs_drift))
    max_cam = cam_indices[max_idx]
    max_val = drift[max_idx]

    print("\n[Stimulus drift]")
    print(f"  Expected ratio (stimulus fps / camera fps): {ratio:.4f}")
    print(f"  First valid mapping: camera {int(first_cam)} → stimulus {first_stim:.0f}")
    print(f"  Last valid mapping: camera {int(last_cam)} → stimulus {last_stim:.0f}")
    print(f"  Observed average ratio: {observed_ratio:.4f}")
    print(f"  Checked {cam_indices.size} camera frames with valid mappings.")
    print(f"  Max drift: {max_val:+.3f} stimulus frames at camera frame {max_cam}")
    print(f"  Median drift: {np.median(drift):+.3f} frames")
    print(f"  95th percentile drift: {np.percentile(abs_drift, 95):.3f} frames (abs)")

    if abs(max_val) > threshold:
        print(
            f"  ⚠ Drift exceeds threshold ({threshold} frames). Investigate around camera frame {max_cam}."
        )
    else:
        print("  Drift stays within threshold.")

    deltas = np.diff(mapping[cam_indices])
    print(
        f"  Δstim per camera frame: min {deltas.min():.3f} | median {np.median(deltas):.3f} | max {deltas.max():.3f}"
    )


def analyze_chaser_alignment(
    zarr_path: Path,
    run_name: Optional[str],
    ratio: float,
    drift_threshold: float,
) -> None:
    root = zarr.open(zarr_path, mode="r")
    run, run_id = _resolve_run(root, run_name)
    print(f"Analyzing chaser alignment for run: {run_id}")

    meta_group = run["video_metadata"]["frame_metadata"]
    stim_frames = _safe_column(meta_group, "stimulus_frame_num").astype(np.int64, copy=False)
    camera_frames = _safe_column(meta_group, "triggering_camera_frame_id").astype(np.int64, copy=False)
    timestamps = _safe_column(meta_group, "timestamp_ns").astype(np.float64, copy=False) / 1e9

    chaser = run["tracking_data"]["chaser_states"]
    stim_chaser = _safe_column(chaser, "stimulus_frame_num").astype(np.int64, copy=False)
    frame_to_cam = dict(zip(stim_frames, camera_frames))
    cam_for_chaser = np.array([frame_to_cam.get(frame, np.nan) for frame in stim_chaser], dtype=np.float64)

    alignment = run["frame_alignment"]
    cam_to_meta = alignment["camera_to_metadata_index"][:]

    _stats("Frame alignment (timestamp)", cam_for_chaser, cam_to_meta, timestamps, "time (s)")
    _analyze_drift(camera_frames, stim_frames, ratio=ratio, threshold=drift_threshold)

    duplicates = len(stim_chaser) - np.unique(stim_chaser).size
    print(f"\nChaser stimulus frames: total {len(stim_chaser)}, unique {np.unique(stim_chaser).size}, duplicates {duplicates}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check chaser alignment and stimulus frame drift.")
    parser.add_argument("zarr_path", type=Path, help="Path to Palette Zarr archive.")
    parser.add_argument("--stimulus-run", type=str, help="Stimulus run name (defaults to latest).")
    parser.add_argument(
        "--ratio",
        type=float,
        default=2.0,
        help="Expected stimulus-frame-per-camera-frame ratio (default: 2.0 for 120fps stimulus vs 60fps camera).",
    )
    parser.add_argument(
        "--drift-threshold",
        type=float,
        default=4.0,
        help="Warn if absolute drift exceeds this many stimulus frames (default: 4).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyze_chaser_alignment(
        args.zarr_path,
        args.stimulus_run,
        ratio=args.ratio,
        drift_threshold=args.drift_threshold,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
