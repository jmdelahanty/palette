#!/usr/bin/env python3
"""
Dry-run utility for diagnosing and repairing broken stimulus_frame_num sequences.

This tool inspects /video_metadata/frame_metadata inside a stimulus H5 file,
prints ratio statistics for the current camera→stimulus mapping, and simulates
the renumbering procedure that fixed the “stride-of-4” issue:

    1. Sort all metadata rows by (triggering_camera_frame_id, timestamp_ns)
    2. Assign new stimulus_frame_num values that increase by 1 each row

By default the script is read-only and only reports what would happen. Use
--apply to write the renumbered stimulus_frame_num values back into the H5.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np


def _compute_camera_stats(camera: np.ndarray) -> Tuple[int, int, int, int]:
    """Return (min_cam, max_cam, unique_count, missing_count)."""
    camera = camera.astype(np.int64, copy=False)
    unique = np.unique(camera)
    cam_min = int(unique[0])
    cam_max = int(unique[-1])
    missing = (cam_max - cam_min + 1) - unique.size
    return cam_min, cam_max, unique.size, missing


def _ratio_stats(camera: np.ndarray, stimulus: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (ratios, unique_camera, first_stimulus_per_camera).

    Each ratio is Δstimulus / Δcamera using the first entry per camera frame.
    """
    order = np.lexsort((stimulus, camera))
    cam_sorted = camera[order]
    stim_sorted = stimulus[order]
    unique_cam, idx_first = np.unique(cam_sorted, return_index=True)
    first_stim = stim_sorted[idx_first]

    if unique_cam.size <= 1:
        return np.zeros(0, dtype=np.float64), unique_cam, first_stim

    delta_cam = np.diff(unique_cam)
    delta_stim = np.diff(first_stim)
    valid = delta_cam != 0
    ratios = delta_stim[valid] / delta_cam[valid] if np.any(valid) else np.zeros(0, dtype=np.float64)
    return ratios, unique_cam, first_stim


def _print_ratios(label: str, camera: np.ndarray, stimulus: np.ndarray, preview: int) -> None:
    ratios, unique_cam, first_stim = _ratio_stats(camera, stimulus)
    cam_min, cam_max, unique_count, missing = _compute_camera_stats(camera)

    print(f"\n[{label}]")
    print(f"  Camera frames: {unique_count} (range {cam_min}–{cam_max}, missing {missing})")
    print(f"  Total rows: {camera.size}")
    if ratios.size:
        print(
            "  Δstim / Δcam ratios → "
            f"min {ratios.min():.3f} | median {np.median(ratios):.3f} "
            f"| mean {ratios.mean():.3f} | max {ratios.max():.3f}"
        )
        print(f"  Ratios outside ±0.5 of 2.0: {(np.abs(ratios - 2.0) > 0.5).sum()}")
        print(f"  Negative ratios: {(ratios < 0).sum()}")
    else:
        print("  Not enough unique camera frames to compute ratios.")

    preview_limit = min(preview, unique_cam.size)
    if preview_limit:
        print("  First camera→stimulus pairs:")
        for cam, stim in zip(unique_cam[:preview_limit], first_stim[:preview_limit]):
            print(f"    cam {int(cam):6d} → stim {int(stim):6d}")


def _generate_new_sequence(
    camera: np.ndarray,
    timestamps: Optional[np.ndarray],
    stimulus_dtype: np.dtype,
    start_value: Optional[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (new_stimulus_array, ordering_indices) for the renumbered data.

    Ordering uses timestamps when available to keep chronological order within
    each camera frame.
    """
    camera = camera.astype(np.int64, copy=False)
    if timestamps is not None:
        order = np.lexsort((timestamps, camera))
    else:
        order = np.argsort(camera, kind="mergesort")

    start = int(start_value) if start_value is not None else int(np.min(camera))

    new_values = np.arange(start, start + camera.size, dtype=np.int64)
    new_stimulus = np.empty_like(camera, dtype=np.int64)
    new_stimulus[order] = new_values
    return (
        new_stimulus.astype(stimulus_dtype, copy=False),
        order,
        new_values.astype(stimulus_dtype, copy=False),
    )


def dry_run_fix(h5_path: Path, preview: int, start_value: Optional[int], apply: bool, backup: bool) -> None:
    if not h5_path.exists():
        raise FileNotFoundError(h5_path)

    with h5py.File(h5_path, "r+") as handle:
        if "/video_metadata/frame_metadata" not in handle:
            raise KeyError("Dataset /video_metadata/frame_metadata not found.")
        ds = handle["/video_metadata/frame_metadata"]
        data = ds[:]

        cam = data["triggering_camera_frame_id"].astype(np.int64, copy=False)
        stim = data["stimulus_frame_num"].astype(np.int64, copy=False)
        timestamps = data["timestamp_ns"].astype(np.int64, copy=False) if "timestamp_ns" in ds.dtype.names else None

        _print_ratios("Current mapping", cam, stim, preview)

        new_stimulus, order, ordered_values = _generate_new_sequence(
            cam, timestamps, ds.dtype["stimulus_frame_num"], start_value
        )
        _print_ratios("Proposed mapping", cam, new_stimulus.astype(np.int64, copy=False), preview)

        diff = np.count_nonzero(stim != new_stimulus)
        print(f"\nRows needing change: {diff} / {stim.size}")

        if not apply:
            print("\nDry run complete. Re-run with --apply to write the new sequence.")
            return

        if backup:
            backup_path = h5_path.with_suffix(h5_path.suffix + ".bak")
            print(f"Creating backup at {backup_path} ...")
            shutil.copy2(h5_path, backup_path)

        print("Writing new stimulus_frame_num values ...")
        ordered_data = data[order].copy()
        ordered_data["stimulus_frame_num"] = ordered_values
        ds[...] = ordered_data

        print("Done. Verify with `python -m fisheye.diagnostics.check_stimulus_alignment --h5 ...`.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dry-run the stimulus_frame_num renumbering procedure used to fix stride errors.",
    )
    parser.add_argument("h5_path", type=Path, help="Path to the stimulus H5 file.")
    parser.add_argument("--preview", type=int, default=20, help="Number of camera→stimulus pairs to preview.")
    parser.add_argument(
        "--start-value",
        type=int,
        help="First stimulus_frame_num to assign (default: current minimum).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually write the renumbered values back into the H5 (default: dry run).",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create <h5>.bak before writing (only meaningful with --apply).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dry_run_fix(
        h5_path=args.h5_path,
        preview=max(1, args.preview),
        start_value=args.start_value,
        apply=args.apply,
        backup=args.backup,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
