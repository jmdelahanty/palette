#!/usr/bin/env python3
"""Check keypoint confidence completeness for keypoint and refined runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import zarr


def _decode(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", "ignore")
    text = str(value).strip()
    return text or None


def _resolve_run(parent: zarr.Group, requested: Optional[str]) -> str:
    if requested:
        if requested not in parent:
            raise RuntimeError(f"Run '{requested}' not found.")
        return requested
    latest = _decode(parent.attrs.get("latest"))
    if latest and latest in parent:
        return latest
    try:
        keys = list(parent.group_keys())  # type: ignore[attr-defined]
    except Exception:
        keys = [str(k) for k in parent.keys()]
    if not keys:
        raise RuntimeError("No runs found.")
    return sorted(keys)[-1]


def _run_conf_stats(group: zarr.Group) -> Dict[str, Any]:
    if "keypoint_confidences" not in group:
        return {"present": False}
    values = np.asarray(group["keypoint_confidences"][:], dtype=np.float64)
    finite = np.isfinite(values)
    row_all_finite = np.all(finite, axis=1) if values.ndim == 2 else np.zeros(0, dtype=bool)
    row_any_nan = np.any(~finite, axis=1) if values.ndim == 2 else np.zeros(0, dtype=bool)
    out: Dict[str, Any] = {
        "present": True,
        "shape": [int(x) for x in values.shape],
        "dtype": str(values.dtype),
        "finite_ratio": float(finite.mean()) if finite.size else None,
        "rows_all_finite": int(row_all_finite.sum()) if row_all_finite.size else 0,
        "rows_with_any_nan": int(row_any_nan.sum()) if row_any_nan.size else 0,
    }
    if finite.any():
        finite_vals = values[finite]
        out.update(
            {
                "min": float(np.min(finite_vals)),
                "p01": float(np.percentile(finite_vals, 1)),
                "p50": float(np.percentile(finite_vals, 50)),
                "p99": float(np.percentile(finite_vals, 99)),
                "max": float(np.max(finite_vals)),
            }
        )
    return out


def _as_int(value: object) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: object) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr", required=True, type=Path, help="Path to .zarr archive.")
    parser.add_argument("--keypoint-run", default=None, help="Explicit keypoints run (default: latest).")
    parser.add_argument("--refined-run", default=None, help="Explicit refined keypoints run (default: latest).")
    parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    args = parser.parse_args()

    zarr_path = args.zarr.expanduser()
    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)

    keypoints_parent = root.get("keypoints_runs")
    if keypoints_parent is None:
        raise SystemExit("No keypoints_runs group.")
    kp_run = _resolve_run(keypoints_parent, args.keypoint_run)
    kp_group = keypoints_parent[kp_run]

    refined_parent = root.get("refined_keypoints_runs") or root.get("keypoints_refined_runs")
    if refined_parent is None:
        raise SystemExit("No refined_keypoints_runs group.")
    ref_run = _resolve_run(refined_parent, args.refined_run)
    ref_group = refined_parent[ref_run]

    kp_summary = kp_group.attrs.get("summary_statistics", {})
    if not isinstance(kp_summary, dict):
        kp_summary = {}
    ref_summary_raw = ref_group.attrs.get("summary_statistics", {})
    if not isinstance(ref_summary_raw, dict):
        ref_summary_raw = {}
    ref_summary = ref_summary_raw.get("refine", ref_summary_raw) if isinstance(ref_summary_raw, dict) else {}
    if not isinstance(ref_summary, dict):
        ref_summary = {}

    result: Dict[str, Any] = {
        "zarr": str(zarr_path),
        "keypoints_run": kp_run,
        "refined_run": ref_run,
        "refined_source_keypoints_run": _decode(ref_group.attrs.get("source_keypoints_run")),
        "source_matches": _decode(ref_group.attrs.get("source_keypoints_run")) == kp_run,
        "keypoints_confidences": _run_conf_stats(kp_group),
        "refined_confidences": _run_conf_stats(ref_group),
        "keypoints_summary": {
            "total_rois": _as_int(kp_summary.get("total_rois")),
            "successful_detections": _as_int(kp_summary.get("successful_detections")),
            "failed_detections": _as_int(kp_summary.get("failed_detections")),
            "success_rate_percent": _as_float(kp_summary.get("success_rate_percent")),
            "mean_confidence": _as_float(kp_summary.get("mean_confidence")),
        },
        "refined_summary": {
            "total_rois": _as_int(ref_summary.get("total_rois")),
            "source_success": _as_int(ref_summary.get("source_success")),
            "source_failures": _as_int(ref_summary.get("source_failures")),
            "refined_success": _as_int(ref_summary.get("refined_success")),
            "confidence_missing": _as_int(ref_summary.get("confidence_missing")),
            "geometry_issues": _as_int(ref_summary.get("geometry_issues")),
            "usable_keypoints": _as_int(ref_summary.get("usable_keypoints")),
            "pass_rate_percent": _as_float(ref_summary.get("pass_rate_percent")),
        },
    }

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    print(f"Zarr: {result['zarr']}")
    print(f"Keypoints run: {kp_run}")
    print(f"Refined run: {ref_run}")
    print(
        "Refined source keypoints run:",
        f"{result['refined_source_keypoints_run']} (matches={result['source_matches']})",
    )
    for label, block in (
        ("Keypoints confidences", result["keypoints_confidences"]),
        ("Refined confidences", result["refined_confidences"]),
    ):
        print(f"\n{label}:")
        if not block.get("present"):
            print("  MISSING")
            continue
        print(f"  shape={block['shape']} dtype={block['dtype']}")
        print(
            f"  finite_ratio={block['finite_ratio']:.6f} "
            f"rows_all_finite={block['rows_all_finite']} rows_with_any_nan={block['rows_with_any_nan']}"
        )
        if "min" in block:
            print(
                f"  min={block['min']:.6f} p01={block['p01']:.6f} "
                f"p50={block['p50']:.6f} p99={block['p99']:.6f} max={block['max']:.6f}"
            )

    ks = result["keypoints_summary"]
    rs = result["refined_summary"]
    print("\nKeypoints summary:")
    print(
        f"  total_rois={ks['total_rois']} successful={ks['successful_detections']} "
        f"failed={ks['failed_detections']} success_rate_percent={ks['success_rate_percent']}"
    )
    print("\nRefined summary:")
    print(
        f"  total_rois={rs['total_rois']} source_success={rs['source_success']} "
        f"source_failures={rs['source_failures']} refined_success={rs['refined_success']}"
    )
    print(
        f"  confidence_missing={rs['confidence_missing']} geometry_issues={rs['geometry_issues']} "
        f"usable_keypoints={rs['usable_keypoints']} pass_rate_percent={rs['pass_rate_percent']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
