"""Inspect coverage statistics for refined eye mask runs.

This script reports how many ROIs contain non-empty left/right masks, along with
basic area summaries. It is useful for spotting regressions where one channel
is consistently empty after refinement.

Usage:
    python -m fisheye.analysis.inspect_refined_eye_masks /path/to/archive.zarr \
        [--refined-run refined_eye_masks_YYYY-MM-DD_HH-MM-SS]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import zarr


def _iter_roi_chunks(array: zarr.Array, chunk_size: int) -> Iterable[Tuple[int, int, np.ndarray]]:
    """Yield (start, stop, data) slices without loading the whole array."""
    total = int(array.shape[0])
    if total == 0:
        return
    chunk_size = max(1, chunk_size)
    for start in range(0, total, chunk_size):
        stop = min(start + chunk_size, total)
        yield start, stop, np.asarray(array[start:stop])


def _summarize_run(run_group: zarr.Group, chunk_size: int) -> None:
    masks_arr = run_group["masks_roi"]
    total_rois, channel_count, _, _ = masks_arr.shape

    if channel_count < 2:
        print("  ! Run provides fewer than two mask channels; skipping detailed stats.")
        return

    eye_labels = list(run_group.attrs.get("eye_labels", []))
    if not eye_labels or len(eye_labels) < channel_count:
        eye_labels = [f"eye_{idx}" for idx in range(channel_count)]

    counts_present = np.zeros(channel_count, dtype=np.int64)
    area_sum = np.zeros(channel_count, dtype=np.float64)
    area_min = np.full(channel_count, np.inf, dtype=np.float64)
    area_max = np.zeros(channel_count, dtype=np.float64)
    both_present = 0
    left_only = 0
    right_only = 0

    for start, stop, chunk in _iter_roi_chunks(masks_arr, chunk_size):
        flat = chunk.reshape(chunk.shape[0], channel_count, -1)
        area = flat.sum(axis=-1)  # (chunk, channel)
        present = area > 0

        counts_present += present.sum(axis=0)
        area_sum += area.sum(axis=0)
        area_min = np.minimum(area_min, np.where(present, area, np.inf).min(axis=0))
        area_max = np.maximum(area_max, area.max(axis=0))

        both_present += int(np.all(present[:, :2], axis=1).sum())
        left_only += int(np.logical_and(present[:, 0], ~present[:, 1]).sum())
        right_only += int(np.logical_and(present[:, 1], ~present[:, 0]).sum())

    area_min = np.where(np.isfinite(area_min), area_min, 0.0)
    coverage = counts_present / total_rois if total_rois else np.zeros(channel_count, dtype=float)
    mean_area = area_sum / np.maximum(counts_present, 1)

    print(f"  Total ROIs: {total_rois}")
    denom = max(total_rois, 1)
    print(f"  Both channels present : {both_present:6d} ({both_present/denom:6.2%})")
    print(f"  Left only            : {left_only:6d} ({left_only/denom:6.2%})")
    print(f"  Right only           : {right_only:6d} ({right_only/denom:6.2%})")
    print()
    print("  Channel stats:")
    for idx in range(channel_count):
        label = eye_labels[idx] if idx < len(eye_labels) else f"channel_{idx}"
        print(
            f"    - {label:12s}: present={counts_present[idx]:6d} "
            f"coverage={coverage[idx]:6.2%} "
            f"mean_area={mean_area[idx]:8.1f} "
            f"min_area={area_min[idx]:6.1f} "
            f"max_area={area_max[idx]:8.1f}"
        )


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path, help="Path to the Palette Zarr archive.")
    parser.add_argument(
        "--refined-run",
        help="Specific refined eye-mask run to inspect (default: latest).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2048,
        help="Number of ROIs to load per chunk (default: 2048).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    root = zarr.open(str(args.zarr_path), mode="r")
    refined_parent = root.get("refined_eye_masks_runs")
    if refined_parent is None:
        raise SystemExit("Archive does not contain 'refined_eye_masks_runs'. Run refinement first.")

    if args.refined_run:
        if args.refined_run not in refined_parent:
            raise SystemExit(f"Refined run '{args.refined_run}' not found.")
        run_names = [args.refined_run]
    else:
        latest = refined_parent.attrs.get("latest")
        if latest and latest in refined_parent:
            run_names = [latest]
        else:
            run_names = [
                name
                for name in refined_parent.keys()
                if isinstance(refined_parent[name], zarr.Group)
            ]
            if not run_names:
                raise SystemExit("No refined runs found.")

    for run_name in run_names:
        run_group = refined_parent[run_name]
        print(f"\nRefined run: {run_name}")
        source = run_group.attrs.get("source_eye_masks_run", "<unknown>")
        method = run_group.attrs.get("method", "<unknown>")
        print(f"  Source run         : {source}")
        print(f"  Method             : {method}")
        _summarize_run(run_group, args.chunk_size)


if __name__ == "__main__":
    main()
