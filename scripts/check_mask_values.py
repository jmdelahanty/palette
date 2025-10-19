#!/usr/bin/env python3
"""Inspect stored eye-mask tensors for binary/probability ranges."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import zarr


def _get_run(root: zarr.Group, explicit: Optional[str]) -> str:
    if explicit:
        if explicit not in root:
            raise ValueError(f"Run '{explicit}' not found under eye_masks_runs")
        return explicit
    latest = root.attrs.get("latest")
    if not latest:
        raise ValueError("No runs recorded under eye_masks_runs and no --run provided.")
    return latest


def _summarize_array(arr: zarr.Array, name: str, chunk: int = 1024) -> None:
    total = arr.size
    min_val = float("inf")
    max_val = float("-inf")
    non_binary = 0
    sum_vals = 0.0
    sum_sq = 0.0

    n = arr.shape[0]
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        block = np.asarray(arr[start:stop])
        min_val = min(min_val, block.min())
        max_val = max(max_val, block.max())
        if block.dtype.kind in {"u", "i"}:
            non_binary += np.count_nonzero((block != 0) & (block != 1))
        sum_vals += block.sum()
        sum_sq += np.square(block, dtype=np.float64).sum()

    mean = sum_vals / total if total else float("nan")
    std = np.sqrt(sum_sq / total - mean * mean) if total else float("nan")
    binary_fraction = 1.0 - (non_binary / total) if total else float("nan")

    print(f"\n{name}:")
    print(f"  shape={arr.shape}, dtype={arr.dtype}")
    print(f"  min={min_val}, max={max_val}")
    print(f"  mean={mean:.6g}, std={std:.6g}")
    if arr.dtype.kind in {"u", "i"}:
        print(f"  binary_fraction={binary_fraction:.6f} (1.0 → fully binary)")
    else:
        print(f"  non-binary count check skipped (float dtype)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check stored binary/probability eye masks in a Zarr run.")
    parser.add_argument("zarr_path", type=Path, help="Path to Palette Zarr archive")
    parser.add_argument("--run", help="Specific eye mask run under eye_masks_runs/")
    parser.add_argument("--chunk", type=int, default=1024, help="Chunk size for iterating arrays (default: 1024)")
    args = parser.parse_args()

    if not args.zarr_path.exists():
        raise FileNotFoundError(args.zarr_path)

    root = zarr.open(str(args.zarr_path), mode="r")
    if "eye_masks_runs" not in root:
        raise ValueError("Zarr archive has no 'eye_masks_runs' group.")
    runs_group = root["eye_masks_runs"]
    run_name = _get_run(runs_group, args.run)
    run_group = runs_group[run_name]

    print(f"Inspecting eye_masks_runs/{run_name}")
    _summarize_array(run_group["masks_roi"], "masks_roi", chunk=args.chunk)
    if "mask_probs_roi" in run_group:
        _summarize_array(run_group["mask_probs_roi"], "mask_probs_roi", chunk=args.chunk)
    else:
        print("\nmask_probs_roi not present in this run.")


if __name__ == "__main__":
    main()

