#!/usr/bin/env python3
"""
Plot chaser X/Y positions against different timeline alignments:

1. Raw stimulus frame timeline (`stimulus_frame_num` or timestamp)
2. Camera frame alignment using `camera_to_metadata_index`
3. Interpolated chaser coverage
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import zarr

from fisheye.shared.zarr_helpers import resolve_zarr_run


def _load_column(group: zarr.Group, name: str) -> np.ndarray:
    if name not in group:
        raise KeyError(f"Dataset '{name}' missing from {group.path}.")
    return group[name][:]


def plot_chaser_alignment(zarr_path: Path, run_name: Optional[str], limit: Optional[int]) -> None:
    root = zarr.open(zarr_path, mode="r")
    run_group, run_id = resolve_zarr_run(
        root,
        ("analysis", "stimulus_runs"),
        run_name,
        fallback_to_sorted="last",
        run_label="Stimulus run",
    )

    meta_group = run_group["video_metadata"]["frame_metadata"]
    stim_frames = _load_column(meta_group, "stimulus_frame_num").astype(np.int64, copy=False)
    camera_frames = _load_column(meta_group, "triggering_camera_frame_id").astype(np.int64, copy=False)
    timestamps = _load_column(meta_group, "timestamp_ns").astype(np.float64, copy=False) / 1e9

    chaser_group = run_group["tracking_data"]["chaser_states"]
    stim_chaser = _load_column(chaser_group, "stimulus_frame_num").astype(np.int64, copy=False)
    pos_x = _load_column(chaser_group, "chaser_pos_x").astype(np.float64, copy=False)
    pos_y = _load_column(chaser_group, "chaser_pos_y").astype(np.float64, copy=False)

    if limit:
        idx = np.argsort(stim_chaser)
        idx = idx[:limit]
        stim_chaser = stim_chaser[idx]
        pos_x = pos_x[idx]
        pos_y = pos_y[idx]

    frame_to_time = dict(zip(stim_frames, timestamps))
    time_raw = np.array([frame_to_time.get(frame, np.nan) for frame in stim_chaser], dtype=np.float64)

    align_group = run_group["frame_alignment"]
    cam_to_meta_raw = align_group["camera_to_metadata_index"][:]

    stim_to_cam = dict(zip(stim_frames, camera_frames))
    cam_for_chaser = np.array([stim_to_cam.get(frame, np.nan) for frame in stim_chaser], dtype=np.float64)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)
    fig.suptitle(f"Chaser positions for {run_id}")

    axes[0].plot(time_raw, pos_x, label="X", color="tab:blue")
    axes[0].plot(time_raw, pos_y, label="Y", color="tab:orange")
    axes[0].set_xlabel("Stimulus time (s)")
    axes[0].set_ylabel("Position (px)")
    axes[0].set_title("Raw stimulus timeline")
    axes[0].legend()

    valid_cam = np.isfinite(cam_for_chaser)
    axes[1].plot(cam_for_chaser[valid_cam], pos_x[valid_cam], label="X", color="tab:blue")
    axes[1].plot(cam_for_chaser[valid_cam], pos_y[valid_cam], label="Y", color="tab:orange")

    # Calculate statistics for missing samples
    total_camera_frames = 0
    missing = np.array([])
    gap_info = ""

    if np.any(valid_cam):
        unique_cams = np.unique(cam_for_chaser[valid_cam].astype(int))
        all_cams = np.arange(int(unique_cams.min()), int(unique_cams.max()) + 1)
        missing = np.setdiff1d(all_cams, unique_cams)
        total_camera_frames = len(all_cams)

        if missing.size:
            axes[1].scatter(missing, np.zeros_like(missing), color="red", s=10, label="No chaser sample")

            # Analyze where gaps are (start, middle, end)
            start_gaps = np.sum(missing < unique_cams[len(unique_cams)//10])  # First 10%
            end_gaps = np.sum(missing > unique_cams[-(len(unique_cams)//10)])  # Last 10%
            middle_gaps = missing.size - start_gaps - end_gaps

            # Find largest gap
            if missing.size > 1:
                gap_sizes = np.diff(missing)
                max_gap_size = int(np.max(gap_sizes)) if len(gap_sizes) > 0 else 1
            else:
                max_gap_size = 1

            gap_info = (
                f"Missing: {missing.size:,} / {total_camera_frames:,} ({100*missing.size/total_camera_frames:.1f}%)\n"
                f"Coverage: {100*(1-missing.size/total_camera_frames):.1f}%\n"
                f"Start region: {start_gaps:,} | Middle: {middle_gaps:,} | End: {end_gaps:,}\n"
                f"Largest consecutive gap: {max_gap_size}"
            )
        else:
            gap_info = f"Coverage: 100% ({total_camera_frames:,} frames)"

    axes[1].set_xlabel("Camera frame")
    axes[1].set_ylabel("Position (px)")
    axes[1].set_title("Camera frame alignment")
    axes[1].legend(loc="upper left")

    # Add statistics text box
    if gap_info:
        axes[1].text(0.98, 0.02, gap_info, transform=axes[1].transAxes,
                    verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    fontsize=8, family='monospace')

    tracking_group = run_group["tracking_data"]
    if "chaser_states_interpolated" in tracking_group:
        interpolated = tracking_group["chaser_states_interpolated"]["stimulus_frame_num"][:]
        interp_cam = np.array([stim_to_cam.get(stim, np.nan) for stim in interpolated], dtype=np.float64)
        valid_interp = np.isfinite(interp_cam)
        axes[2].plot(interp_cam[valid_interp], annotate_positions := np.zeros_like(interp_cam[valid_interp]), '.', color="green", label="Interpolated sample")
        missing_interp = np.isnan(interp_cam)
        if np.any(missing_interp):
            axes[2].scatter(np.where(missing_interp)[0], np.zeros(np.sum(missing_interp)), color="red", s=10, label="Missing even after interpolation")
        axes[2].set_xlabel("Camera frame (interpolated dataset)")
        axes[2].set_ylabel("Indicator")
        axes[2].set_title("Interpolated chaser coverage")
        axes[2].legend()
    else:
        axes[2].text(0.5, 0.5, "No interpolated chaser dataset", ha="center", va="center", transform=axes[2].transAxes)
        axes[2].set_axis_off()

    plt.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot chaser positions vs different alignment timelines.")
    parser.add_argument("zarr_path", type=Path, help="Path to Palette Zarr archive.")
    parser.add_argument("--stimulus-run", type=str, help="Specific stimulus run name (defaults to latest).")
    parser.add_argument("--limit", type=int, help="Limit number of chaser samples (optional).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_chaser_alignment(args.zarr_path, args.stimulus_run, args.limit)


if __name__ == "__main__":  # pragma: no cover
    main()
