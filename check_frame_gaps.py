#!/usr/bin/env python3
"""Quick diagnostic to check for gaps in camera frame IDs."""

import sys
import numpy as np
import zarr

if len(sys.argv) < 2:
    print("Usage: python check_frame_gaps.py <zarr_path>")
    sys.exit(1)

zarr_path = sys.argv[1]
root = zarr.open(zarr_path, mode="r")

# Check refined online runs
if "refined_online_runs" in root:
    refined_runs = root["refined_online_runs"]
    latest = refined_runs.attrs.get("latest")

    if latest and latest in refined_runs:
        refined_group = refined_runs[latest]
        frames = refined_group["camera_frame_ids"][:]
        interp_grp = refined_group["interpolated"]
        valid_mask = interp_grp["valid_mask"][:]

        # Get valid frames only
        valid_frames = frames[valid_mask]

        print(f"Refined online run: {latest}")
        print(f"Total frames: {len(frames)}")
        print(f"Valid frames: {valid_mask.sum()}")
        print(f"\nFrame range: {frames.min()} to {frames.max()}")
        print(f"Valid frame range: {valid_frames.min()} to {valid_frames.max()}")

        # Check for gaps
        delta = np.diff(valid_frames)
        gaps = delta[delta > 1]

        print(f"\nFrame-to-frame deltas:")
        print(f"  Consecutive (delta=1): {(delta == 1).sum()}")
        print(f"  Gaps (delta>1): {len(gaps)}")

        if len(gaps) > 0:
            print(f"\nLargest gaps:")
            gap_indices = np.where(delta > 1)[0]
            gap_sizes = delta[gap_indices]
            sorted_idx = np.argsort(gap_sizes)[::-1][:10]  # Top 10

            for idx in sorted_idx:
                gap_idx = gap_indices[idx]
                gap_size = int(gap_sizes[idx])
                frame_before = int(valid_frames[gap_idx])
                frame_after = int(valid_frames[gap_idx + 1])
                print(f"  Gap of {gap_size} frames: {frame_before} -> {frame_after}")

        # Check all frames (including invalid)
        print(f"\nAll frames (including invalid):")
        all_delta = np.diff(frames)
        print(f"  Min delta: {all_delta.min()}")
        print(f"  Max delta: {all_delta.max()}")
        print(f"  Non-unit deltas: {(all_delta != 1).sum()}")

        if (all_delta != 1).any():
            print(f"\n  WARNING: Frame IDs are not consecutive!")
    else:
        print("No latest refined online run found")
else:
    print("No refined_online_runs in archive")
