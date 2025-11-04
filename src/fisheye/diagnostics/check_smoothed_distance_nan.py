#!/usr/bin/env python3
"""Check offline movement runs for NaNs in distance_to_target_smoothed_mm."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import zarr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Report NaN counts in distance_to_target_smoothed_mm for offline movement runs."
        )
    )
    parser.add_argument("store", type=Path, help="Path to Palette Zarr store")
    parser.add_argument(
        "--run",
        dest="run_name",
        help=(
            "Specific run under analysis/movement_runs/offline. Defaults to the group's 'latest'."
        ),
    )
    return parser.parse_args()


def resolve_offline_run(root: zarr.Group, name: str | None) -> tuple[zarr.Group, str]:
    try:
        offline_parent = root["analysis"]["movement_runs"]["offline"]
    except KeyError as exc:
        raise SystemExit("analysis/movement_runs/offline not found in store") from exc

    run_name = name or offline_parent.attrs.get("latest")
    if not run_name:
        raise SystemExit("No run specified and offline group has no 'latest' attribute")
    if run_name not in offline_parent:
        available = ", ".join(sorted(offline_parent.group_keys()))
        raise SystemExit(
            f"Run '{run_name}' not found under analysis/movement_runs/offline. Available: {available or '(none)'}"
        )

    return offline_parent[run_name], run_name


def inspect_run(group: zarr.Group) -> int:
    if "distance_to_target_smoothed_mm" not in group:
        print("distance_to_target_smoothed_mm missing; rerun movement_analysis with smoothing > 0")
        return -1

    array = np.asarray(group["distance_to_target_smoothed_mm"], dtype=np.float32)
    camera_frames = np.asarray(group.get("camera_frame_ids", []), dtype=np.int64)

    if camera_frames.size and camera_frames.size != array.size:
        raise SystemExit("camera_frame_ids length mismatches smoothed distance array")

    nan_mask = np.isnan(array)
    nan_count = int(nan_mask.sum())
    total = int(array.size)

    print(f"Samples: {total}")
    print(f"NaN entries: {nan_count}")
    if nan_count and camera_frames.size == array.size:
        frames = camera_frames[nan_mask]
        print("First NaN frames (up to 20):", frames[:20])
    return nan_count


def main() -> None:
    args = parse_args()
    if not args.store.exists():
        raise SystemExit(f"Store path '{args.store}' does not exist")

    root = zarr.open(str(args.store), mode="r")
    run_group, run_name = resolve_offline_run(root, args.run_name)
    print(f"Checking offline run: {run_name}")
    missing = inspect_run(run_group)
    if missing == -1:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
