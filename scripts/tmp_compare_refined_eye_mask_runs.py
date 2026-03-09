#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import zarr


def _same_array(lhs: np.ndarray, rhs: np.ndarray) -> bool:
    if lhs.shape != rhs.shape:
        return False
    if lhs.dtype.kind in "fc" or rhs.dtype.kind in "fc":
        return np.allclose(lhs, rhs, equal_nan=True, rtol=1e-6, atol=1e-6)
    return np.array_equal(lhs, rhs)


def _value_equal(lhs: Any, rhs: Any) -> bool:
    if isinstance(lhs, dict) and isinstance(rhs, dict):
        if set(lhs) != set(rhs):
            return False
        return all(_value_equal(lhs[key], rhs[key]) for key in lhs)
    if isinstance(lhs, (list, tuple)) and isinstance(rhs, (list, tuple)):
        if len(lhs) != len(rhs):
            return False
        return all(_value_equal(a, b) for a, b in zip(lhs, rhs))
    if isinstance(lhs, float) or isinstance(rhs, float):
        try:
            left = float(lhs)
            right = float(rhs)
        except (TypeError, ValueError):
            return False
        if math.isnan(left) and math.isnan(right):
            return True
        return math.isclose(left, right, rel_tol=1e-6, abs_tol=1e-6)
    return lhs == rhs


def _compare_array_groups(label: str, lhs: zarr.Group, rhs: zarr.Group) -> list[str]:
    mismatches: list[str] = []
    all_names = sorted(set(lhs.array_keys()) | set(rhs.array_keys()))
    for name in all_names:
        left_arr = lhs.get(name)
        right_arr = rhs.get(name)
        if not isinstance(left_arr, zarr.Array) or not isinstance(right_arr, zarr.Array):
            mismatches.append(f"{label}/{name}: missing in one run")
            continue
        left_np = np.asarray(left_arr[:])
        right_np = np.asarray(right_arr[:])
        if not _same_array(left_np, right_np):
            mismatches.append(
                f"{label}/{name}: differs shape_a={left_np.shape} shape_b={right_np.shape} "
                f"dtype_a={left_np.dtype} dtype_b={right_np.dtype}"
            )
    return mismatches


def main() -> int:
    parser = argparse.ArgumentParser(description="Temporary compare for two refined eye-mask runs.")
    parser.add_argument("zarr_path", type=Path, help="Path to Palette Zarr archive.")
    parser.add_argument("run_a", help="First run under refined_eye_masks_runs/.")
    parser.add_argument("run_b", help="Second run under refined_eye_masks_runs/.")
    args = parser.parse_args()

    root = zarr.open(str(args.zarr_path), mode="r", use_consolidated=False)
    refined_parent = root.get("refined_eye_masks_runs")
    if not isinstance(refined_parent, zarr.Group):
        print("FAIL refined_eye_masks_runs missing")
        return 1

    run_a = refined_parent.get(args.run_a)
    run_b = refined_parent.get(args.run_b)
    if not isinstance(run_a, zarr.Group) or not isinstance(run_b, zarr.Group):
        print(f"FAIL missing run(s): {args.run_a=} {args.run_b=}")
        return 1

    metrics_a = run_a.get("metrics")
    metrics_b = run_b.get("metrics")
    if not isinstance(metrics_a, zarr.Group) or not isinstance(metrics_b, zarr.Group):
        print("FAIL metrics group missing in one run")
        return 1

    mismatches: list[str] = []
    mismatches.extend(_compare_array_groups("top", run_a, run_b))
    mismatches.extend(_compare_array_groups("metrics", metrics_a, metrics_b))

    selected_attrs = [
        "total_rois",
        "successful_eyes",
        "successful_roi_pairs",
        "successful_roi_pair_rate",
        "reason_tag_counts",
        "metrics_summary",
    ]
    for key in selected_attrs:
        lhs = run_a.attrs.get(key)
        rhs = run_b.attrs.get(key)
        if not _value_equal(lhs, rhs):
            mismatches.append(f"attr/{key}: differs")

    if mismatches:
        print("FAIL refined run comparison")
        for item in mismatches:
            print(f"  - {item}")
        return 1

    print("OK refined run comparison")
    print(f"  run_a={args.run_a}")
    print(f"  run_b={args.run_b}")
    print("  compared: top-level arrays, metrics arrays, selected summary attrs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
