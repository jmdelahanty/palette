"""
Keypoint review entrypoint for retune, manual correction, and audit.

This keeps refined runs as the editable working copy and updates
postprocess summary statistics after corrections.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import zarr


def _get_latest_refined_run(root: zarr.Group) -> str:
    refined_parent = root.get("refined_keypoints_runs")
    if refined_parent is None:
        raise RuntimeError("No refined_keypoints_runs found in archive.")
    latest = refined_parent.attrs.get("latest")
    if not latest:
        raise RuntimeError("No refined keypoint runs recorded.")
    return latest


def _count_truthy(arr: Optional[zarr.Array]) -> int:
    if arr is None:
        return 0
    data = np.asarray(arr[:], dtype=bool)
    return int(np.sum(data))


def _count_reason_tags(reason_arr: Optional[zarr.Array]) -> Dict[str, int]:
    if reason_arr is None:
        return {}
    raw = reason_arr[:]
    counts: Counter[str] = Counter()
    for value in raw:
        if value is None:
            continue
        text = str(value)
        if not text:
            continue
        for tag in text.split("|"):
            tag = tag.strip()
            if tag:
                counts[tag] += 1
    return dict(counts)


def _count_ints(arr: Optional[zarr.Array]) -> Dict[str, int]:
    if arr is None:
        return {}
    data = np.asarray(arr[:])
    vals, counts = np.unique(data, return_counts=True)
    return {str(int(v)): int(c) for v, c in zip(vals.tolist(), counts.tolist())}


def _update_postprocess_summary(
    refined: zarr.Group,
    *,
    print_summary: bool = True,
) -> Dict[str, object]:
    total_rois = int(refined["keypoints_roi"].shape[0])
    refined_success = _count_truthy(refined.get("refined_success"))
    usable = _count_truthy(refined.get("usable_keypoints"))
    confidence_valid = _count_truthy(refined.get("confidence_valid"))
    geometry_valid = _count_truthy(refined.get("geometry_valid"))
    flip_corrected = _count_truthy(refined.get("flip_corrected"))
    heading_valid = _count_truthy(refined.get("heading_valid"))
    source_success = _count_truthy(refined.get("source_success"))

    remaining_failures = max(0, total_rois - refined_success)
    success_rate = (refined_success / total_rois * 100.0) if total_rois else 0.0

    detection_source_counts = _count_ints(refined.get("detection_source"))
    retune_id_counts = _count_ints(refined.get("retune_id"))
    retune_total = int(sum(count for key, count in retune_id_counts.items() if key != "-1"))
    reason_counts = _count_reason_tags(refined.get("reason"))
    manual_corrections = int(reason_counts.get("manual_correction", 0))

    post_stats: Dict[str, object] = {
        "total_rois": total_rois,
        "source_success": source_success,
        "refined_success": refined_success,
        "remaining_failures": remaining_failures,
        "success_rate_percent": round(success_rate, 2),
        "usable_keypoints": usable,
        "confidence_valid": confidence_valid,
        "geometry_valid": geometry_valid,
        "flip_corrected": flip_corrected,
        "heading_valid": heading_valid,
        "detection_source_counts": detection_source_counts,
        "reason_counts": reason_counts,
        "retune_id_counts": retune_id_counts,
        "retune_total": retune_total,
        "manual_corrections": manual_corrections,
    }

    summary_raw = refined.attrs.get("summary_statistics", {})
    if not isinstance(summary_raw, dict):
        summary_raw = {}

    if "refine" in summary_raw:
        summary_out = dict(summary_raw)
    else:
        summary_out = {"refine": summary_raw}

    summary_out["postprocess"] = post_stats
    summary_out["postprocess_updated_utc"] = datetime.now(timezone.utc).isoformat()
    refined.attrs["summary_statistics"] = summary_out

    if print_summary:
        print("\nPostprocess Summary")
        print(f"  Total ROIs: {total_rois}")
        print(f"  Refined success: {refined_success}")
        print(f"  Remaining failures: {remaining_failures}")
        print(f"  Success rate: {success_rate:.2f}%")
        print(f"  Usable keypoints: {usable}")
        print(f"  Retune total: {retune_total}")
        print(f"  Manual corrections: {manual_corrections}")

    return post_stats


def run_manual_review(
    zarr_path: str,
    refined_run: Optional[str] = None,
    crop_run: Optional[str] = None,
) -> Dict[str, object]:
    root = zarr.open_group(str(zarr_path), mode="a")
    refined_run = refined_run or _get_latest_refined_run(root)
    refined = root[f"refined_keypoints_runs/{refined_run}"]

    from .keypoint_failure_review import launch_review

    launch_review(str(zarr_path), refined_run=refined_run, crop_run=crop_run)
    return _update_postprocess_summary(refined, print_summary=True)


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Keypoint review: retune failures, manual corrections, or audit summary."
    )
    parser.add_argument("zarr_path", type=Path, help="Path to Palette Zarr directory.")
    parser.add_argument(
        "--refined-run",
        help="Refined keypoint run to review (defaults to latest).",
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--retune", action="store_true", help="Run failure retune UI.")
    mode.add_argument("--manual", action="store_true", help="Run manual correction UI.")
    mode.add_argument("--audit", action="store_true", help="Recompute postprocess summary only.")

    parser.add_argument(
        "--crop-run",
        help="Crop run to source ROI images from (manual mode only).",
    )
    parser.add_argument(
        "--apply-batch-size",
        type=int,
        default=128,
        help="Batch size for retune apply.",
    )
    parser.add_argument(
        "--apply-workers",
        type=int,
        default=4,
        help="Worker threads for retune apply.",
    )

    args = parser.parse_args(argv)

    root = zarr.open_group(str(args.zarr_path), mode="a")
    refined_run = args.refined_run or _get_latest_refined_run(root)
    refined = root[f"refined_keypoints_runs/{refined_run}"]

    if args.retune:
        from .keypoint_tuner import run_failure_tuner

        run_failure_tuner(
            str(args.zarr_path),
            refined_run,
            1,
            apply_batch_size=args.apply_batch_size,
            apply_workers=args.apply_workers,
        )
        _update_postprocess_summary(refined, print_summary=True)
    elif args.manual:
        run_manual_review(str(args.zarr_path), refined_run=refined_run, crop_run=args.crop_run)
    else:
        _update_postprocess_summary(refined, print_summary=True)


if __name__ == "__main__":  # pragma: no cover
    main()
