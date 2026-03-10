"""
Eye mask review entrypoint for retune, manual correction, and audit.

Workflow mirrors keypoint_review but operates on refined_eye_masks_runs.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Dict, Iterable, Optional

# Limit threading BEFORE importing numpy to avoid saturating all CPU cores.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

import numpy as np
import zarr

from ..shared.detect_reason_codec import read_reason_labels


_DEFAULT_SUCCESS_MIN_EYE_AREA_PX = 50.0


def _get_latest_refined_run(root: zarr.Group) -> str:
    refined_parent = root.get("refined_eye_masks_runs")
    if refined_parent is None:
        raise RuntimeError("No refined_eye_masks_runs found in archive.")
    latest = refined_parent.attrs.get("latest")
    if not latest:
        raise RuntimeError("No refined eye mask runs recorded.")
    return latest


def _read_reason_labels(refined: zarr.Group) -> Optional[np.ndarray]:
    metrics = refined.get("metrics")
    if isinstance(metrics, zarr.Group):
        labels = read_reason_labels(metrics)
        if labels is not None:
            return labels
    return read_reason_labels(refined)


def _count_reason_tags(refined: zarr.Group) -> Dict[str, int]:
    raw = _read_reason_labels(refined)
    if raw is None:
        return {}
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


def _positive_float(value: object) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric <= 0:
        return None
    return numeric


def _resolve_success_min_eye_area_px(refined: zarr.Group) -> Optional[float]:
    raw = refined.attrs.get("success_min_eye_area_px")
    if raw is None:
        raw = refined.attrs.get("min_eye_area_success_px")
    threshold = _positive_float(raw)
    if threshold is not None:
        return threshold
    return float(_DEFAULT_SUCCESS_MIN_EYE_AREA_PX)


def _load_area_refined(refined: zarr.Group) -> Optional[np.ndarray]:
    masks_arr = refined.get("masks_roi")
    shape = getattr(masks_arr, "shape", None) if masks_arr is not None else None
    if isinstance(shape, tuple) and len(shape) == 4 and int(shape[1]) >= 2:
        total = int(shape[0])
        if total <= 0:
            return np.zeros((0, 2), dtype=np.float32)
        chunks = getattr(masks_arr, "chunks", None)
        if isinstance(chunks, tuple) and chunks:
            step = max(1, int(chunks[0]))
        elif isinstance(chunks, int):
            step = max(1, int(chunks))
        else:
            step = 256
        out = np.zeros((total, 2), dtype=np.float32)
        for start in range(0, total, step):
            stop = min(total, start + step)
            block = np.asarray(masks_arr[start:stop], dtype=np.uint8)
            if block.ndim != 4 or int(block.shape[1]) < 2:
                return None
            area = np.sum((block[:, :2] > 0).reshape(block.shape[0], 2, -1), axis=2, dtype=np.int64)
            out[start:stop, :] = np.asarray(area, dtype=np.float32)
        return out

    metrics = refined.get("metrics")
    if not isinstance(metrics, zarr.Group):
        return None
    area_arr = metrics.get("area_refined")
    if area_arr is None:
        return None
    area_refined = np.asarray(area_arr[:], dtype=np.float32)
    if area_refined.ndim != 2 or area_refined.shape[1] < 2:
        return None
    return area_refined[:, :2]


def _load_eye_separation(refined: zarr.Group) -> np.ndarray:
    eye_separation_arr = refined.get("eye_separation")
    if eye_separation_arr is not None:
        return np.asarray(eye_separation_arr[:], dtype=np.float32)

    ellipse_success = np.asarray(refined["ellipse_success"][:], dtype=bool)
    ellipse_params = np.asarray(refined["ellipse_params"][:], dtype=np.float32)
    if ellipse_success.ndim != 2 or ellipse_success.shape[1] < 2:
        raise RuntimeError("ellipse_success must have shape (n, >=2) to derive eye_separation.")
    if ellipse_params.ndim != 3 or ellipse_params.shape[1] < 2 or ellipse_params.shape[2] < 2:
        raise RuntimeError("ellipse_params must have shape (n, >=2, >=2) to derive eye_separation.")

    separation = np.full((ellipse_success.shape[0],), np.nan, dtype=np.float32)
    finite = np.all(np.isfinite(ellipse_params[:, :2, :2]), axis=(1, 2))
    valid = np.all(ellipse_success[:, :2], axis=1) & finite
    if np.any(valid):
        delta = ellipse_params[valid, 0, :2] - ellipse_params[valid, 1, :2]
        separation[valid] = np.linalg.norm(delta, axis=1).astype(np.float32)
    return separation


def _compute_success_mask(
    ellipse_success: np.ndarray,
    eye_separation: np.ndarray,
    min_sep: Optional[float],
    max_sep: Optional[float],
    area_refined: Optional[np.ndarray],
    min_eye_area_px: Optional[float],
) -> np.ndarray:
    pair_success = np.all(ellipse_success, axis=1)
    sep_ok = np.ones_like(pair_success, dtype=bool)
    if eye_separation.size:
        sep_ok = np.isfinite(eye_separation)
        if min_sep is not None:
            sep_ok &= eye_separation >= float(min_sep)
        if max_sep is not None:
            sep_ok &= eye_separation <= float(max_sep)

    area_ok = np.ones_like(pair_success, dtype=bool)
    threshold = _positive_float(min_eye_area_px) if min_eye_area_px is not None else None
    if threshold is not None and area_refined is not None and area_refined.shape[0] == pair_success.shape[0]:
        finite = np.all(np.isfinite(area_refined[:, :2]), axis=1)
        area_ok = finite & np.all(area_refined[:, :2] >= float(threshold), axis=1)
    return pair_success & sep_ok & area_ok


def _update_postprocess_summary(
    root: zarr.Group,
    refined: zarr.Group,
    *,
    print_summary: bool = True,
) -> Dict[str, object]:
    group_path = str(getattr(refined, "path", "") or "")
    if group_path and not group_path.startswith("refined_eye_masks_runs/"):
        raise RuntimeError(
            "_update_postprocess_summary expects a refined_eye_masks_runs/<run> group; "
            f"got path={group_path!r}."
        )

    ellipse_success = np.asarray(refined["ellipse_success"][:], dtype=bool)
    eye_separation = _load_eye_separation(refined)
    min_sep, max_sep = _get_sep_limits(root, refined)
    area_refined = _load_area_refined(refined)
    min_eye_area_px = _resolve_success_min_eye_area_px(refined)
    success_mask = _compute_success_mask(
        ellipse_success,
        eye_separation,
        min_sep,
        max_sep,
        area_refined,
        min_eye_area_px,
    )

    total_rois = int(success_mask.size)
    successful_eyes = int(np.asarray(ellipse_success, dtype=bool).sum())
    successful_pairs = int(np.sum(success_mask))
    remaining_failures = max(0, total_rois - successful_pairs)
    successful_pair_rate = (float(successful_pairs) / float(total_rois)) if total_rois else 0.0
    success_rate = (successful_pairs / total_rois * 100.0) if total_rois else 0.0

    reason_counts = _count_reason_tags(refined)
    manual_corrections = int(reason_counts.get("manual_correction", 0))

    retune_id_counts = _count_ints(refined.get("retune_id"))
    retune_total = int(sum(count for key, count in retune_id_counts.items() if key != "-1"))

    post_stats: Dict[str, object] = {
        "total_rois": total_rois,
        "successful_eyes": successful_eyes,
        "successful_roi_pairs": successful_pairs,
        "successful_roi_pair_rate": successful_pair_rate,
        "remaining_failures": remaining_failures,
        "success_rate_percent": round(success_rate, 2),
        "manual_corrections": manual_corrections,
        "retune_id_counts": retune_id_counts,
        "retune_total": retune_total,
        "reason_counts": reason_counts,
        "min_eye_separation": min_sep,
        "max_eye_separation": max_sep,
        "min_eye_area_success_px": min_eye_area_px,
    }

    # Keep top-level refined run attrs aligned with postprocess metrics so registry
    # refresh/backfill and data-card aggregation see the latest reviewed values.
    refined.attrs["total_rois"] = total_rois
    refined.attrs["successful_eyes"] = successful_eyes
    refined.attrs["successful_roi_pairs"] = successful_pairs
    refined.attrs["successful_roi_pair_rate"] = successful_pair_rate
    refined.attrs["reason_counts"] = reason_counts

    summary_raw = refined.attrs.get("summary_statistics", {})
    if not isinstance(summary_raw, dict):
        summary_raw = {}

    if "refine" in summary_raw:
        summary_out = dict(summary_raw)
    else:
        summary_out = {"refine": refined.attrs.get("refine_stats", {})}

    previous_post = summary_out.get("postprocess")
    summary_out["postprocess"] = post_stats
    summary_out["reason_counts"] = reason_counts
    if previous_post != post_stats or "postprocess_updated_utc" not in summary_out:
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
    frame_flag_file: Optional[str] = None,
    review_state: str = "approved",
    review_method: str = "manual",
    review_intended_use: str = "training",
    reviewer: Optional[str] = None,
    review_notes: Optional[str] = None,
) -> Dict[str, object]:
    if not reviewer:
        reviewer = os.environ.get("USER")
    root = zarr.open_group(str(zarr_path), mode="a", use_consolidated=False)
    refined_run = refined_run or _get_latest_refined_run(root)
    refined = root[f"refined_eye_masks_runs/{refined_run}"]

    from .eye_mask_failure_review import launch_review

    launch_review(
        str(zarr_path),
        refined_run=refined_run,
        crop_run=crop_run,
        frame_flag_file=frame_flag_file,
        review_state=review_state,
        review_method=review_method,
        review_intended_use=review_intended_use,
        reviewer=reviewer,
        review_notes=review_notes,
    )
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
        "--frame-flag-file",
        default="eye_mask_frame_flags.json",
        help="JSON file with flagged eye-mask frames/ROIs to include in manual review.",
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
    parser.add_argument(
        "--review-state",
        default="approved",
        choices=["approved", "pending", "rejected", "needs_review"],
        help="Review state to set when approving in manual review.",
    )
    parser.add_argument(
        "--review-method",
        default="manual",
        choices=["manual", "algorithmic", "hybrid", "spotcheck"],
        help="Review method label (default: manual).",
    )
    parser.add_argument(
        "--review-intended-use",
        default="training",
        choices=["training", "full_recording"],
        help="Intended use label (default: training).",
    )
    parser.add_argument("--reviewer", help="Reviewer name (defaults to $USER).")
    parser.add_argument("--review-notes", help="Optional review notes.")

    args = parser.parse_args(argv)

    root = zarr.open_group(str(args.zarr_path), mode="a", use_consolidated=False)
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
        run_manual_review(
            str(args.zarr_path),
            refined_run=refined_run,
            crop_run=args.crop_run,
            frame_flag_file=args.frame_flag_file,
            review_state=args.review_state,
            review_method=args.review_method,
            review_intended_use=args.review_intended_use,
            reviewer=args.reviewer,
            review_notes=args.review_notes,
        )
    else:
        _update_postprocess_summary(root, refined, print_summary=True)


if __name__ == "__main__":  # pragma: no cover
    main()
