"""
Eye mask review entrypoint for retune, manual correction, and audit.

Workflow mirrors keypoint_review but operates on refined_eye_masks_runs.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import zarr


def _get_latest_refined_run(root: zarr.Group) -> str:
    refined_parent = root.get("refined_eye_masks_runs")
    if refined_parent is None:
        raise RuntimeError("No refined_eye_masks_runs found in archive.")
    latest = refined_parent.attrs.get("latest")
    if not latest:
        raise RuntimeError("No refined eye mask runs recorded.")
    return latest


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


def _get_reason_array(refined: zarr.Group) -> Optional[zarr.Array]:
    metrics = refined.get("metrics")
    if isinstance(metrics, zarr.Group) and "reason" in metrics:
        return metrics["reason"]
    return None


def _get_sep_limits(root: zarr.Group, refined: zarr.Group) -> tuple[Optional[float], Optional[float]]:
    analysis = root.get("analysis_metadata")
    if analysis is not None:
        tuning = analysis.attrs.get("eye_mask_tuning")
        if isinstance(tuning, dict):
            params = tuning.get("tuned_parameters", {})
            min_sep = params.get("min_eye_separation")
            max_sep = params.get("max_eye_separation")
            return (
                float(min_sep) if min_sep is not None else None,
                float(max_sep) if max_sep is not None else None,
            )

    source_run = refined.attrs.get("source_eye_masks_run")
    if source_run and "eye_masks_runs" in root and source_run in root["eye_masks_runs"]:
        src = root["eye_masks_runs"][source_run]
        min_sep = src.attrs.get("min_eye_separation")
        max_sep = src.attrs.get("max_eye_separation")
        return (
            float(min_sep) if min_sep is not None else None,
            float(max_sep) if max_sep is not None else None,
        )
    return None, None


def _compute_success_mask(
    ellipse_success: np.ndarray,
    eye_separation: np.ndarray,
    min_sep: Optional[float],
    max_sep: Optional[float],
) -> np.ndarray:
    pair_success = np.all(ellipse_success, axis=1)
    sep_ok = np.ones_like(pair_success, dtype=bool)
    if eye_separation.size:
        sep_ok = np.isfinite(eye_separation)
        if min_sep is not None:
            sep_ok &= eye_separation >= float(min_sep)
        if max_sep is not None:
            sep_ok &= eye_separation <= float(max_sep)
    return pair_success & sep_ok


def _update_postprocess_summary(
    root: zarr.Group,
    refined: zarr.Group,
    *,
    print_summary: bool = True,
) -> Dict[str, object]:
    ellipse_success = np.asarray(refined["ellipse_success"][:], dtype=bool)
    eye_separation = np.asarray(refined["eye_separation"][:], dtype=np.float32)
    min_sep, max_sep = _get_sep_limits(root, refined)
    success_mask = _compute_success_mask(ellipse_success, eye_separation, min_sep, max_sep)

    total_rois = int(success_mask.size)
    successful_pairs = int(np.sum(success_mask))
    remaining_failures = max(0, total_rois - successful_pairs)
    success_rate = (successful_pairs / total_rois * 100.0) if total_rois else 0.0

    reason_arr = _get_reason_array(refined)
    reason_counts = _count_reason_tags(reason_arr)
    manual_corrections = int(reason_counts.get("manual_correction", 0))

    retune_id_counts = _count_ints(refined.get("retune_id"))
    retune_total = int(sum(count for key, count in retune_id_counts.items() if key != "-1"))

    post_stats: Dict[str, object] = {
        "total_rois": total_rois,
        "successful_roi_pairs": successful_pairs,
        "remaining_failures": remaining_failures,
        "success_rate_percent": round(success_rate, 2),
        "manual_corrections": manual_corrections,
        "retune_id_counts": retune_id_counts,
        "retune_total": retune_total,
        "reason_counts": reason_counts,
        "min_eye_separation": min_sep,
        "max_eye_separation": max_sep,
    }

    summary_raw = refined.attrs.get("summary_statistics", {})
    if not isinstance(summary_raw, dict):
        summary_raw = {}

    if "refine" in summary_raw:
        summary_out = dict(summary_raw)
    else:
        summary_out = {"refine": refined.attrs.get("refine_stats", {})}

    summary_out["postprocess"] = post_stats
    summary_out["postprocess_updated_utc"] = datetime.now(timezone.utc).isoformat()
    refined.attrs["summary_statistics"] = summary_out

    if print_summary:
        print("\nPostprocess Summary")
        print(f"  Total ROIs: {total_rois}")
        print(f"  Successful ROI pairs: {successful_pairs}")
        print(f"  Remaining failures: {remaining_failures}")
        print(f"  Success rate: {success_rate:.2f}%")
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
    refined = root[f"refined_eye_masks_runs/{refined_run}"]

    from .eye_mask_failure_review import launch_review

    launch_review(str(zarr_path), refined_run=refined_run, crop_run=crop_run)
    return _update_postprocess_summary(root, refined, print_summary=True)


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Eye mask review: retune failures, manual corrections, or audit summary."
    )
    parser.add_argument("zarr_path", type=Path, help="Path to Palette Zarr directory.")
    parser.add_argument(
        "--refined-run",
        help="Refined eye mask run to review (defaults to latest).",
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
    refined = root[f"refined_eye_masks_runs/{refined_run}"]

    if args.retune:
        from .eye_mask_tuner import run_failure_tuner

        run_failure_tuner(
            str(args.zarr_path),
            refined_run,
            1,
            apply_batch_size=args.apply_batch_size,
            apply_workers=args.apply_workers,
        )
        _update_postprocess_summary(root, refined, print_summary=True)
    elif args.manual:
        run_manual_review(str(args.zarr_path), refined_run=refined_run, crop_run=args.crop_run)
    else:
        _update_postprocess_summary(root, refined, print_summary=True)


if __name__ == "__main__":  # pragma: no cover
    main()
