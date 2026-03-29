#!/usr/bin/env python3
"""Analyze periodicity in chaser sample coverage across camera frames."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import zarr

from fisheye.shared.zarr_helpers import resolve_zarr_run


def analyze_periodicity(zarr_path: Path, run_name: Optional[str], max_period: int) -> None:
    root = zarr.open(zarr_path, mode="r")
    run, run_id = resolve_zarr_run(
        root,
        ("analysis", "stimulus_runs"),
        run_name,
        fallback_to_sorted="last",
        run_label="Stimulus run",
    )
    print(f"Checking chaser periodicity for run: {run_id}")

    meta = run["video_metadata"]["frame_metadata"]
    cam_frames = meta["triggering_camera_frame_id"][:].astype(np.int64, copy=False)
    stim_frames = meta["stimulus_frame_num"][:].astype(np.int64, copy=False)

    chaser = run["tracking_data"]["chaser_states"]
    stim_chaser = chaser["stimulus_frame_num"][:].astype(np.int64, copy=False)

    frame_to_cam = dict(zip(stim_frames, cam_frames))
    cam_samples = np.array([frame_to_cam.get(frame, np.nan) for frame in stim_chaser], dtype=np.float64)
    valid = np.isfinite(cam_samples)
    cam_samples = cam_samples[valid].astype(np.int64, copy=False)

    camera_min = int(cam_frames.min())
    camera_max = int(cam_frames.max())
    all_cams = np.arange(camera_min, camera_max + 1, dtype=np.int64)

    present = np.zeros(all_cams.size, dtype=bool)
    idx = cam_samples - camera_min
    idx = idx[(idx >= 0) & (idx < present.size)]
    present[idx] = True
    missing = all_cams[~present]

    print(f"Camera range: {camera_min}–{camera_max} ({all_cams.size} frames)")
    print(f"Chaser coverage: {present.sum()} frames ({present.sum()/all_cams.size:.2%})")
    print(f"Missing frames: {missing.size} ({missing.size/all_cams.size:.2%})")

    if missing.size == 0:
        print("No gaps detected; skipping periodicity analysis.")
        return

    deltas = np.diff(missing)
    if deltas.size == 0:
        print("Only isolated gaps; no periodic pattern detected.")
        return

    counter = Counter(deltas)
    print("\nGap spacing counts (top 10):")
    for spacing, count in counter.most_common(10):
        print(f"  Δcamera {spacing}: {count} occurrences")

    if deltas.size >= max_period:
        fft = np.fft.rfft(present.astype(float) - present.mean())
        freqs = np.fft.rfftfreq(present.size, d=1)
        power = np.abs(fft) ** 2
        top = np.argsort(power)[-5:][::-1]
        print("\nTop frequency components (approx periods):")
        for idx in top:
            if freqs[idx] == 0:
                continue
            period = 1.0 / freqs[idx]
            print(f"  Period ~{period:.2f} frames (power {power[idx]:.2e})")
    else:
        print("\nNot enough samples for FFT-based periodicity (set --max-period lower if needed).")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check for periodic gaps in chaser sampling.")
    parser.add_argument("zarr_path", type=Path, help="Path to Palette Zarr archive.")
    parser.add_argument("--stimulus-run", type=str, help="Stimulus run name (defaults to latest).")
    parser.add_argument(
        "--max-period",
        type=int,
        default=512,
        help="Minimum number of samples required to run FFT-based periodicity (default: 512).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyze_periodicity(args.zarr_path, args.stimulus_run, args.max_period)


if __name__ == "__main__":  # pragma: no cover
    main()
