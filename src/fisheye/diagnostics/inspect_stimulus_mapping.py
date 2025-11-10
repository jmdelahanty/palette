#!/usr/bin/env python3
"""
Inspect camera→stimulus mapping arrays to verify alignment.

This script reports:
  * First/last real mappings and observed ratio
  * Drift from expected ratio
  * Distribution of per-camera stimulus deltas
  * Consistency between camera frames and stimulus frame numbers
"""

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


def _first_last(mapping: np.ndarray) -> tuple[int, float, int, float]:
    """Find first and last valid mappings."""
    valid = mapping >= 0
    indices = np.nonzero(valid)[0]
    if indices.size == 0:
        raise ValueError("No valid mappings found.")
    first_cam = int(indices[0])
    last_cam = int(indices[-1])
    return first_cam, float(mapping[first_cam]), last_cam, float(mapping[last_cam])


def _ratio(first_cam: float, first_stim: float, last_cam: float, last_stim: float) -> float:
    """Calculate observed stimulus:camera frame ratio."""
    span = max(1.0, last_cam - first_cam)
    return (last_stim - first_stim) / span


def _drift_report(mapping: np.ndarray, expected_ratio: float) -> None:
    """Report drift statistics from expected ratio."""
    valid = mapping >= 0
    valid_idx = np.nonzero(valid)[0]
    if valid_idx.size < 2:
        print("  Not enough valid frames for drift analysis.")
        return

    first_cam = float(valid_idx[0])
    first_stim = mapping[valid_idx[0]]
    expected = first_stim + (valid_idx - first_cam) * expected_ratio
    drift = mapping[valid_idx] - expected
    abs_drift = np.abs(drift)

    print("  Drift stats (frames):")
    print(f"    Median {np.median(drift):+.3f} | 95th {np.percentile(abs_drift, 95):.3f} | Max {drift[np.argmax(abs_drift)]:+.3f}")


def _inspect(zarr_path: Path, run_name: Optional[str], ratio: float) -> None:
    root = zarr.open(zarr_path, mode="r")
    run, run_id = _resolve_run(root, run_name)
    print(f"Inspecting stimulus mapping for run: {run_id}")

    meta_group = run["video_metadata"]["frame_metadata"]
    stim = meta_group["stimulus_frame_num"][:].astype(np.int64, copy=False)
    camera = meta_group["triggering_camera_frame_id"][:].astype(np.int64, copy=False)

    # Build camera → stimulus mapping from metadata
    cam_to_stim = _build_camera_to_stimulus_map(camera, stim)

    try:
        first_cam, first_stim, last_cam, last_stim = _first_last(cam_to_stim)
    except ValueError:
        raise RuntimeError("No valid camera→stimulus mappings found.")

    obs_ratio = _ratio(first_cam, first_stim, last_cam, last_stim)
    print(f"  First mapping: camera {first_cam} → stimulus {first_stim:.0f}")
    print(f"  Last mapping:  camera {last_cam} → stimulus {last_stim:.0f}")
    print(f"  Observed avg ratio: {obs_ratio:.4f} (expected {ratio:.4f})")

    _drift_report(cam_to_stim.astype(np.float64, copy=False), ratio)

    # Report on metadata consistency
    alignment = run["frame_alignment"]
    cam_to_meta = alignment["camera_to_metadata_index"][:]

    valid_cam_indices = np.nonzero(cam_to_stim >= 0)[0]
    lookup_matches = 0
    lookup_mismatches = 0

    for cam_idx in valid_cam_indices[:1000]:  # Sample first 1000 for speed
        if cam_idx < len(cam_to_meta):
            meta_idx = cam_to_meta[cam_idx]
            if 0 <= meta_idx < len(stim):
                if stim[meta_idx] == cam_to_stim[cam_idx]:
                    lookup_matches += 1
                else:
                    lookup_mismatches += 1

    total_checked = lookup_matches + lookup_mismatches
    if total_checked > 0:
        print(f"\n  Metadata consistency check (sample of {total_checked}):")
        print(f"    Matches: {lookup_matches} ({100*lookup_matches/total_checked:.1f}%)")
        if lookup_mismatches > 0:
            print(f"    Mismatches: {lookup_mismatches} ({100*lookup_mismatches/total_checked:.1f}%)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect camera→stimulus mapping arrays.")
    parser.add_argument("zarr_path", type=Path, help="Path to Palette Zarr archive.")
    parser.add_argument("--stimulus-run", type=str, help="Stimulus run name (defaults to latest).")
    parser.add_argument(
        "--ratio",
        type=float,
        default=2.0,
        help="Expected stimulus frames per camera frame (default: 2.0).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _inspect(args.zarr_path, args.stimulus_run, ratio=args.ratio)


if __name__ == "__main__":  # pragma: no cover
    main()
