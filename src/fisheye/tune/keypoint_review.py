"""
Keypoint review entrypoint for retune, manual correction, and audit.

This keeps refined runs as the editable working copy and updates
postprocess summary statistics after corrections.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import zarr

from ..shared.detect_reason_codec import read_reason_labels


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


def _count_reason_tags(refined: zarr.Group) -> Dict[str, int]:
    reason_labels = read_reason_labels(refined)
    if reason_labels is None:
        return {}
    raw = np.asarray(reason_labels, dtype=object)
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


def _coerce_targets(value: object) -> tuple[Optional[list[int]], Optional[list[int]]]:
    frames: list[int] = []
    roi_indices: list[int] = []
    if isinstance(value, (list, tuple, np.ndarray)):
        items = value
    else:
        items = [value]
    for item in items:
        if isinstance(item, dict):
            frame_idx = item.get("frame_idx")
            if frame_idx is None:
                frame_idx = item.get("frame")
            if frame_idx is None:
                frame_idx = item.get("frame_index")
            roi_idx = item.get("roi_idx")
            if roi_idx is None:
                roi_idx = item.get("roi")
            if roi_idx is None:
                roi_idx = item.get("roi_index")
            if frame_idx is not None:
                try:
                    frames.append(int(frame_idx))
                except (TypeError, ValueError):
                    pass
            if roi_idx is not None:
                try:
                    roi_indices.append(int(roi_idx))
                except (TypeError, ValueError):
                    pass
            continue
        if isinstance(item, (int, np.integer)):
            frames.append(int(item))
            continue
        token = str(item).strip()
        if not token:
            continue
        try:
            frames.append(int(token))
        except ValueError:
            continue
    frames_out = sorted(set(frames)) if frames else None
    roi_out = sorted(set(roi_indices)) if roi_indices else None
    return frames_out, roi_out


def _parse_targets_arg(
    value: Optional[str],
    zarr_path: Optional[str],
) -> tuple[Optional[list[int]], Optional[list[int]]]:
    if value is None:
        return None, None
    text = value.strip()
    if not text:
        return None, None
    path = Path(text)
    if path.exists():
        raw = path.read_text(encoding="utf-8")
        try:
            data = json.loads(raw)
        except Exception:
            data = None
        if isinstance(data, dict):
            if not zarr_path:
                return None, None
            zarr_str = str(zarr_path)
            zarr_resolved = str(Path(zarr_path).resolve())
            frames_val = data.get(zarr_str) or data.get(zarr_resolved)
            if frames_val is None:
                for key, val in data.items():
                    try:
                        if Path(key).resolve() == Path(zarr_path).resolve():
                            frames_val = val
                            break
                    except Exception:
                        continue
            return _coerce_targets(frames_val)
        if isinstance(data, list):
            return _coerce_targets(data)
        items = re.split(r"[,\s]+", raw.strip())
        return _coerce_targets(items)
    items = re.split(r"[,\s]+", text)
    return _coerce_targets(items)


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
    heading_finite = _count_truthy(refined.get("heading_finite"))
    heading_usable = _count_truthy(refined.get("heading_usable"))
    source_success = _count_truthy(refined.get("source_success"))

    remaining_failures = max(0, total_rois - refined_success)
    success_rate = (refined_success / total_rois * 100.0) if total_rois else 0.0
    usable_rate = (usable / total_rois * 100.0) if total_rois else 0.0

    detection_source_counts = _count_ints(refined.get("detection_source"))
    retune_id_counts = _count_ints(refined.get("retune_id"))
    retune_total = int(sum(count for key, count in retune_id_counts.items() if key != "-1"))
    reason_counts = _count_reason_tags(refined)
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
        "heading_finite": heading_finite,
        "heading_usable": heading_usable,
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

    previous_post = summary_out.get("postprocess")
    summary_out["postprocess"] = post_stats
    if previous_post != post_stats or "postprocess_updated_utc" not in summary_out:
        summary_out["postprocess_updated_utc"] = datetime.now(timezone.utc).isoformat()
    refined.attrs["summary_statistics"] = summary_out

    if print_summary:
        print("\nPostprocess Summary")
        print(f"  Total ROIs: {total_rois}")
        print(f"  Refined success: {refined_success}/{total_rois} ({success_rate:.2f}%)")
        print(f"  Remaining failures: {remaining_failures}")
        print(f"  Trainable (QC): {usable}/{total_rois} ({usable_rate:.2f}%)")
        low_conf = int(reason_counts.get("low_confidence", 0))
        if low_conf:
            print(f"  Low confidence: {low_conf} (excluded from training)")
        print(f"  Retune total: {retune_total}")
        print(f"  Manual corrections: {manual_corrections}")

    return post_stats


def run_manual_review(
    zarr_path: str,
    refined_run: Optional[str] = None,
    crop_run: Optional[str] = None,
    include_all: bool = False,
    target_frames: Optional[Sequence[int]] = None,
    target_roi_indices: Optional[Sequence[int]] = None,
    review_state: str = "approved",
    review_method: str = "manual",
    review_intended_use: Optional[str] = None,
    reviewer: Optional[str] = None,
    review_notes: Optional[str] = None,
    frame_flag_file: Optional[str] = None,
    detect_flag_file: Optional[str] = None,
    detect_frame_flag_file: Optional[str] = None,
) -> Dict[str, object]:
    if not reviewer:
        reviewer = os.environ.get("USER")
    root = zarr.open_group(str(zarr_path), mode="a")
    refined_run = refined_run or _get_latest_refined_run(root)
    refined = root[f"refined_keypoints_runs/{refined_run}"]

    from .keypoint_failure_review import launch_review

    launch_review(
        str(zarr_path),
        refined_run=refined_run,
        crop_run=crop_run,
        include_all=include_all,
        target_frames=target_frames,
        target_roi_indices=target_roi_indices,
        review_state=review_state,
        review_method=review_method,
        review_intended_use=review_intended_use,
        reviewer=reviewer,
        review_notes=review_notes,
        frame_flag_file=frame_flag_file,
        detect_flag_file=detect_flag_file,
        detect_frame_flag_file=detect_frame_flag_file,
    )
    # Re-open to pick up review_status written by the UI before updating summary.
    root = zarr.open_group(str(zarr_path), mode="a")
    refined = root[f"refined_keypoints_runs/{refined_run}"]
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
        "--all",
        action="store_true",
        help="Review all keypoints (manual mode only; default is failed keypoints only).",
    )
    parser.add_argument(
        "--frames",
        type=str,
        help=(
            "Frame indices to review (manual/retune only). Accepts comma/space-separated "
            "indices or a path to a JSON/text list. JSON mapping values may be frame integers "
            "or objects with frame_idx/roi_idx for ROI-exact targeting in manual mode."
        ),
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
        default=None,
        choices=["training", "full_recording"],
        help="Intended use label (default: infer from existing status or zarr use).",
    )
    parser.add_argument("--reviewer", help="Reviewer name (defaults to $USER).")
    parser.add_argument("--review-notes", help="Optional review notes.")
    parser.add_argument(
        "--frame-flag-file",
        default="keypoint_frame_flags.json",
        help="JSON file to append flagged frames (manual/retune modes).",
    )
    parser.add_argument(
        "--detect-flag-file",
        default="retune_flags.txt",
        help="Text file to append recordings flagged for detection retune.",
    )
    parser.add_argument(
        "--detect-frame-flag-file",
        default="retune_frame_flags.json",
        help="JSON file to append detection frames flagged for retune.",
    )

    args = parser.parse_args(argv)

    root = zarr.open_group(str(args.zarr_path), mode="a")
    refined_run = args.refined_run or _get_latest_refined_run(root)
    refined = root[f"refined_keypoints_runs/{refined_run}"]
    target_frames, target_roi_indices = _parse_targets_arg(args.frames, str(args.zarr_path))
    if args.frames and not target_frames and not target_roi_indices:
        print("No target frames/ROIs found for this recording; skipping review.")
        return

    if args.retune:
        if target_roi_indices:
            print("Warning: ROI-index targets are ignored in retune mode; using frame targets only.")
        if args.all:
            print("Warning: --all is ignored in retune mode.")
        from .keypoint_tuner import run_failure_tuner

        run_failure_tuner(
            str(args.zarr_path),
            refined_run,
            1,
            apply_batch_size=args.apply_batch_size,
            apply_workers=args.apply_workers,
            target_frames=target_frames,
            frame_flag_file=args.frame_flag_file,
            detect_flag_file=args.detect_flag_file,
            detect_frame_flag_file=args.detect_frame_flag_file,
        )
        _update_postprocess_summary(refined, print_summary=True)
    elif args.manual:
        run_manual_review(
            str(args.zarr_path),
            refined_run=refined_run,
            crop_run=args.crop_run,
            include_all=bool(args.all),
            target_frames=target_frames,
            target_roi_indices=target_roi_indices,
            review_state=args.review_state,
            review_method=args.review_method,
            review_intended_use=args.review_intended_use,
            reviewer=args.reviewer,
            review_notes=args.review_notes,
            frame_flag_file=args.frame_flag_file,
            detect_flag_file=args.detect_flag_file,
            detect_frame_flag_file=args.detect_frame_flag_file,
        )
    else:
        _update_postprocess_summary(refined, print_summary=True)


if __name__ == "__main__":  # pragma: no cover
    main()
