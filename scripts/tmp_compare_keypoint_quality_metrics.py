#!/usr/bin/env python3
"""Compare keypoint-run success_rate vs refined usable rate for one archive."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import zarr

from fisheye.utils import prepare_keypoint_training_from_registry as prep_pose


def _fmt(value: object) -> str:
    if value is None:
        return "None"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _count_tags(arr: zarr.Array | None) -> dict[str, int]:
    if arr is None:
        return {}
    raw = np.asarray(arr[:], dtype=object)
    counts: Counter[str] = Counter()
    for item in raw:
        text = str(item).strip() if item is not None else ""
        if not text:
            continue
        for tag in text.split("|"):
            tag = tag.strip()
            if tag:
                counts[tag] += 1
    return dict(counts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr", required=True, type=Path, help="Path to archive .zarr")
    parser.add_argument(
        "--selector",
        default="latest_traditional",
        help="Keypoint selector (latest, latest_traditional, latest_yolo, explicit run).",
    )
    args = parser.parse_args()

    zarr_path = args.zarr.expanduser()
    try:
        root = zarr.open_group(str(zarr_path), mode="r", consolidated=False)
    except TypeError:
        root = zarr.open_group(str(zarr_path), mode="r")

    resolved_run, selector_resolved = prep_pose._resolve_keypoint_run(root, args.selector)
    kp_group = root["keypoints_runs"][resolved_run]

    keypoints_total = int(kp_group["keypoints_roi"].shape[0]) if "keypoints_roi" in kp_group else None
    success_rate_attr = prep_pose._as_float(kp_group.attrs.get("success_rate"))
    keypoints_processed = prep_pose._as_int(kp_group.attrs.get("keypoints_processed"))

    success_rate_from_detection_success = None
    if "detection_success" in kp_group:
        success_arr = np.asarray(kp_group["detection_success"][:], dtype=bool)
        if success_arr.shape[0] > 0:
            success_rate_from_detection_success = float(success_arr.sum()) / float(success_arr.shape[0])

    quality = prep_pose._resolve_refined_keypoint_quality(root, resolved_run, zarr_path=zarr_path)
    refined_run = quality.get("refined_keypoint_run")
    usable_rate = prep_pose._as_float(quality.get("usable_keypoints_rate"))
    usable_total = prep_pose._as_int(quality.get("usable_keypoints_total"))
    review = quality.get("keypoint_review_status")
    if isinstance(review, dict):
        review_state = prep_pose._decode_attr(review.get("state"))
        review_use = prep_pose._decode_attr(review.get("intended_use"))
    else:
        review_state = None
        review_use = None

    refined_success_rate = None
    refined_success_total = None
    source_success_rate = None
    source_success_total = None
    manual_correction_count = None
    refined_parent = root.get("refined_keypoints_runs") or root.get("keypoints_refined_runs")
    if refined_run and refined_parent is not None and str(refined_run) in refined_parent:
        refined_group = refined_parent[str(refined_run)]
        if "refined_success" in refined_group:
            ref = np.asarray(refined_group["refined_success"][:], dtype=bool)
            refined_success_total = int(ref.sum())
            refined_success_rate = float(ref.mean()) if ref.size else None
        if "source_success" in refined_group:
            src = np.asarray(refined_group["source_success"][:], dtype=bool)
            source_success_total = int(src.sum())
            source_success_rate = float(src.mean()) if src.size else None
        reason_counts = _count_tags(refined_group.get("reason"))
        manual_correction_count = int(reason_counts.get("manual_correction", 0))

    print("Archive:", zarr_path)
    print("Selector requested:", args.selector)
    print("Selector resolved:", selector_resolved)
    print("Keypoint run resolved:", resolved_run)
    print("Keypoints total:", _fmt(keypoints_total))
    print("keypoints_processed attr:", _fmt(keypoints_processed))
    print("keypoints success_rate attr:", _fmt(success_rate_attr))
    print("keypoints detection_success-derived rate:", _fmt(success_rate_from_detection_success))
    print("Refined run:", _fmt(refined_run))
    print("Refined usable total:", _fmt(usable_total))
    print("Refined usable rate:", _fmt(usable_rate))
    print("Refined source_success total/rate:", f"{_fmt(source_success_total)}/{_fmt(source_success_rate)}")
    print("Refined refined_success total/rate:", f"{_fmt(refined_success_total)}/{_fmt(refined_success_rate)}")
    print("Refined manual_correction count:", _fmt(manual_correction_count))
    print("Review state/intended_use:", f"{_fmt(review_state)}/{_fmt(review_use)}")
    print("Review status divergence:", _fmt(quality.get("keypoint_review_status_divergence")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
