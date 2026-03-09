#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from fisheye.shared.detect_reason_codec import decode_reason_bytes


def _print_ok(label: str) -> None:
    print(f"OK   {label}")


def _print_fail(label: str, detail: str) -> None:
    print(f"FAIL {label}: {detail}")


def _require(group: zarr.Group, name: str) -> zarr.Array:
    arr = group.get(name)
    if not isinstance(arr, zarr.Array):
        raise KeyError(name)
    return arr


def _check_contours(run_group: zarr.Group) -> list[str]:
    failures: list[str] = []
    for side in ("left", "right"):
        ptr = np.asarray(_require(run_group, f"contour_{side}_ptr")[:], dtype=np.int64)
        length = np.asarray(_require(run_group, f"contour_{side}_len")[:], dtype=np.int32)
        store = np.asarray(_require(run_group, f"contours_{side}")[:], dtype=np.float32)

        has_data = ptr >= 0
        if not np.array_equal(has_data, length > 0):
            failures.append(f"contour_{side}: ptr/len presence mismatch")
            continue
        if np.any(ptr[has_data] + length[has_data] > store.shape[0]):
            failures.append(f"contour_{side}: pointer range exceeds contour store")
    return failures


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Temporary sanity checks for one refined eye-mask run.")
    parser.add_argument("zarr_path", type=Path, help="Path to Palette Zarr archive.")
    parser.add_argument("run_name", help="Run name under refined_eye_masks_runs/.")
    args = parser.parse_args()

    root = zarr.open(str(args.zarr_path), mode="r", use_consolidated=False)
    refined_parent = root.get("refined_eye_masks_runs")
    if not isinstance(refined_parent, zarr.Group):
        print("FAIL refined_eye_masks_runs missing")
        return 1
    run_group = refined_parent.get(args.run_name)
    if not isinstance(run_group, zarr.Group):
        print(f"FAIL refined_eye_masks_runs/{args.run_name} missing")
        return 1

    metrics_group = run_group.get("metrics")
    if not isinstance(metrics_group, zarr.Group):
        print("FAIL metrics group missing")
        return 1

    failed = False

    try:
        masks_roi = np.asarray(_require(run_group, "masks_roi")[:], dtype=np.uint8)
        ellipse_params = np.asarray(_require(run_group, "ellipse_params")[:], dtype=np.float32)
        ellipse_success = np.asarray(_require(run_group, "ellipse_success")[:], dtype=bool)
        eye_separation = np.asarray(_require(run_group, "eye_separation")[:], dtype=np.float32)
        reason = np.asarray(_require(metrics_group, "reason")[:], dtype=object)
        reason_bytes = np.asarray(_require(metrics_group, "reason_bytes")[:], dtype=np.uint8)
        area_refined = np.asarray(_require(metrics_group, "area_refined")[:], dtype=np.float32)
        filter_flags = np.asarray(_require(metrics_group, "filter_flags")[:], dtype=bool)
    except KeyError as exc:
        _print_fail("required datasets", f"missing {exc.args[0]}")
        return 1

    if "reason_local" in metrics_group:
        failed = True
        _print_fail("reason_local cleanup", "metrics/reason_local still exists")
    else:
        _print_ok("reason_local cleanup")

    decoded_reason = decode_reason_bytes(reason_bytes)
    if np.array_equal(reason.astype(str), decoded_reason.astype(str)):
        _print_ok("reason_bytes mirror")
    else:
        failed = True
        _print_fail("reason_bytes mirror", "metrics/reason does not match decoded metrics/reason_bytes")

    n_rows = int(masks_roi.shape[0])
    shape_ok = (
        masks_roi.ndim == 4
        and ellipse_params.shape[:2] == masks_roi.shape[:2]
        and ellipse_success.shape[:2] == masks_roi.shape[:2]
        and eye_separation.shape == (n_rows,)
        and reason.shape == (n_rows,)
        and area_refined.shape == (n_rows, 2)
        and filter_flags.shape == (n_rows, 2)
    )
    if shape_ok:
        _print_ok("row-aligned shapes")
    else:
        failed = True
        _print_fail(
            "row-aligned shapes",
            (
                f"masks={masks_roi.shape} ellipse_params={ellipse_params.shape} "
                f"ellipse_success={ellipse_success.shape} eye_separation={eye_separation.shape} "
                f"reason={reason.shape} area_refined={area_refined.shape} filter_flags={filter_flags.shape}"
            ),
        )

    contour_failures = _check_contours(run_group)
    if contour_failures:
        failed = True
        for item in contour_failures:
            _print_fail("contours", item)
    else:
        _print_ok("contours")

    successful_pairs_expected = int(np.all(ellipse_success, axis=1).sum())
    successful_pairs_attr = int(run_group.attrs.get("successful_roi_pairs", -1))
    if successful_pairs_expected == successful_pairs_attr:
        _print_ok("successful_roi_pairs attr")
    else:
        failed = True
        _print_fail(
            "successful_roi_pairs attr",
            f"attr={successful_pairs_attr} computed={successful_pairs_expected}",
        )

    reason_strings = reason.astype(str).tolist()
    reason_counts = dict(Counter(reason_strings))
    reason_tag_counts = dict(Counter(tag for item in reason_strings for tag in item.split("|") if tag))

    metrics_summary = run_group.attrs.get("metrics_summary", {})
    if not isinstance(metrics_summary, dict):
        failed = True
        _print_fail("metrics_summary", f"expected dict, got {type(metrics_summary).__name__}")
    else:
        summary_reason_counts = metrics_summary.get("reason_counts", {})
        summary_reason_tag_counts = metrics_summary.get("reason_tag_counts", {})
        if _value_equal(summary_reason_counts, reason_counts):
            _print_ok("metrics_summary.reason_counts")
        else:
            failed = True
            _print_fail(
                "metrics_summary.reason_counts",
                f"attr={summary_reason_counts} computed={reason_counts}",
            )
        if _value_equal(summary_reason_tag_counts, reason_tag_counts):
            _print_ok("metrics_summary.reason_tag_counts")
        else:
            failed = True
            _print_fail(
                "metrics_summary.reason_tag_counts",
                f"attr={summary_reason_tag_counts} computed={reason_tag_counts}",
            )

    total_rois_attr = int(run_group.attrs.get("total_rois", -1))
    if total_rois_attr == n_rows:
        _print_ok("total_rois attr")
    else:
        failed = True
        _print_fail("total_rois attr", f"attr={total_rois_attr} rows={n_rows}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
