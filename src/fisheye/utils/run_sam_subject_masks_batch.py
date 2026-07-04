#!/usr/bin/env python3
"""Batch inspect/apply utility for SAM-based subject-mask generation across recording zarrs."""

from __future__ import annotations

from fisheye.shared.zarr_helpers import infer_zarr_use as _infer_zarr_use
from fisheye.shared.zarr_discovery import iter_filesystem_zarrs as _iter_zarr
import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import zarr

from fisheye.utils.run_sam_subject_masks import (
    BOX_PROMPT_SOURCE_CHOICES,
    DEFAULT_BOX_PROMPT_SOURCE,
    DEFAULT_INFER_BATCH_SIZE,
    DEFAULT_NEGATIVE_POINT_MARGIN_FRACTION,
    DEFAULT_NEGATIVE_POINT_POLICY,
    DEFAULT_POSE_BOX_EXPAND_FRACTION,
    DEFAULT_ROI_INSET_FRACTION,
    KEYPOINT_GROUP_CHOICES,
    NEGATIVE_POINT_POLICY_CHOICES,
    inspect_sam_subject_archive_path,
    run_sam_subject_mask_inference,
)
from fisheye.utils.zarr_io import open_zarr_root


DEFAULT_RECORDINGS_ROOT = Path("/nvme1/recordings")
DEFAULT_BATCH_PREPARE_COUNT = 0


@dataclass(frozen=True)
class BatchOptions:
    apply: bool
    zarr_use_filter: str
    crop_run: Optional[str]
    keypoint_run: Optional[str]
    keypoint_group: str
    prepare_count: int
    output_run: Optional[str]
    sam3_root: Optional[Path]
    checkpoint: Optional[Path]
    batch_size: int
    device: Optional[str]
    multimask_output: bool
    use_box_prompt: bool
    box_prompt_source: str
    roi_inset_fraction: float
    pose_box_expand_fraction: float
    negative_point_policy: str
    negative_point_margin_fraction: float
    overwrite: bool
    no_hf_download: bool
    include_interpolated: bool
    positive_keypoint_labels: tuple[str, ...] | None
    allow_zero_eligible: bool
    roi_cache_policy: str
    roi_cache_dir: Optional[str]
    roi_live_acceleration: str
    roi_live_gpu_chunk_frames: int
    profile_timings: bool


@dataclass(frozen=True)
class BatchRow:
    zarr_path: Path
    status: str
    reason: Optional[str] = None
    observed_use: Optional[str] = None
    crop_run: Optional[str] = None
    keypoint_group: Optional[str] = None
    keypoint_run: Optional[str] = None
    output_run: Optional[str] = None
    eligible_rows: Optional[int] = None
    segmented_rows: Optional[int] = None
    nonempty_rows: Optional[int] = None


def _resolve_roots(paths: list[Path]) -> list[Path]:
    if paths:
        return paths
    env_root = os.environ.get("PALETTE_RECORDINGS_ROOT")
    if env_root:
        return [Path(env_root)]
    return [DEFAULT_RECORDINGS_ROOT]


def _output_run_exists(root: zarr.Group, output_run: str) -> bool:
    parent = root.get("subject_mask_runs")
    return parent is not None and output_run in parent


def _format_row(row: BatchRow) -> str:
    source = (
        f"{row.keypoint_group}/{row.keypoint_run}"
        if row.keypoint_group is not None and row.keypoint_run is not None
        else "-"
    )
    target = f"subject_mask_runs/{row.output_run}" if row.output_run is not None else "-"
    eligible = "-" if row.eligible_rows is None else str(row.eligible_rows)
    segmented = "-" if row.segmented_rows is None else str(row.segmented_rows)
    nonempty = "-" if row.nonempty_rows is None else str(row.nonempty_rows)
    use = row.observed_use or "unknown"
    reason = row.reason or "-"
    return (
        f"{row.status}\t{row.zarr_path}\tuse={use}\t{source}\t{target}\t"
        f"eligible={eligible}\tsegmented={segmented}\tnonempty={nonempty}\t{reason}"
    )


def _process_zarr_path(zarr_path: Path, options: BatchOptions) -> BatchRow:
    try:
        root = open_zarr_root(zarr_path, mode="r")
    except Exception as exc:
        return BatchRow(zarr_path=zarr_path, status="error", reason=f"failed to open archive ({exc})")

    observed_use = _infer_zarr_use(root, zarr_path)
    if options.zarr_use_filter != "any" and observed_use != options.zarr_use_filter:
        return BatchRow(
            zarr_path=zarr_path,
            status="filtered_zarr_use",
            reason=f"wanted={options.zarr_use_filter} found={observed_use or 'unknown'}",
            observed_use=observed_use,
        )

    try:
        inspect_summary = inspect_sam_subject_archive_path(
            zarr_path,
            crop_run=options.crop_run,
            keypoint_run=options.keypoint_run,
            keypoint_group=options.keypoint_group,
            skip_interpolated=not options.include_interpolated,
            prepare_count=int(options.prepare_count),
            output_run=options.output_run,
            sam3_root=options.sam3_root,
            inspect_runtime=False,
            use_box_prompt=options.use_box_prompt,
            box_prompt_source=options.box_prompt_source,
            roi_inset_fraction=float(options.roi_inset_fraction),
            pose_box_expand_fraction=float(options.pose_box_expand_fraction),
            negative_point_policy=options.negative_point_policy,
            negative_point_margin_fraction=float(options.negative_point_margin_fraction),
            positive_keypoint_labels=options.positive_keypoint_labels,
            roi_cache_policy=options.roi_cache_policy,
            roi_cache_dir=options.roi_cache_dir,
            roi_live_acceleration=options.roi_live_acceleration,
            roi_live_gpu_chunk_frames=int(options.roi_live_gpu_chunk_frames),
        )
    except Exception as exc:
        return BatchRow(
            zarr_path=zarr_path,
            status="error",
            reason=str(exc),
            observed_use=observed_use,
        )

    crop_run = str(inspect_summary.get("crop_run") or "")
    keypoint_group = str(inspect_summary.get("keypoint_group") or "")
    keypoint_run = str(inspect_summary.get("keypoint_run") or "")
    planned_output = inspect_summary.get("planned_output") or {}
    output_run = str(planned_output.get("output_run") or "")
    eligibility = inspect_summary.get("eligibility") or {}
    eligible_rows = int(eligibility.get("eligible_rows") or 0)

    if output_run and _output_run_exists(root, output_run) and not options.overwrite:
        return BatchRow(
            zarr_path=zarr_path,
            status="skipped_existing",
            reason=f"subject_mask_runs/{output_run} already exists",
            observed_use=observed_use,
            crop_run=crop_run or None,
            keypoint_group=keypoint_group or None,
            keypoint_run=keypoint_run or None,
            output_run=output_run or None,
            eligible_rows=eligible_rows,
        )

    if eligible_rows <= 0 and not options.allow_zero_eligible:
        return BatchRow(
            zarr_path=zarr_path,
            status="skipped_no_eligible",
            reason="inspect reported zero eligible rows",
            observed_use=observed_use,
            crop_run=crop_run or None,
            keypoint_group=keypoint_group or None,
            keypoint_run=keypoint_run or None,
            output_run=output_run or None,
            eligible_rows=eligible_rows,
        )

    if not options.apply:
        return BatchRow(
            zarr_path=zarr_path,
            status="planned",
            observed_use=observed_use,
            crop_run=crop_run or None,
            keypoint_group=keypoint_group or None,
            keypoint_run=keypoint_run or None,
            output_run=output_run or None,
            eligible_rows=eligible_rows,
        )

    try:
        apply_summary = run_sam_subject_mask_inference(
            zarr_path,
            crop_run=options.crop_run,
            keypoint_run=options.keypoint_run,
            keypoint_group=options.keypoint_group,
            output_run=options.output_run,
            sam3_root=options.sam3_root,
            checkpoint_path=options.checkpoint,
            batch_size=int(options.batch_size),
            skip_interpolated=not options.include_interpolated,
            multimask_output=options.multimask_output,
            use_box_prompt=options.use_box_prompt,
            box_prompt_source=options.box_prompt_source,
            roi_inset_fraction=float(options.roi_inset_fraction),
            pose_box_expand_fraction=float(options.pose_box_expand_fraction),
            negative_point_policy=options.negative_point_policy,
            negative_point_margin_fraction=float(options.negative_point_margin_fraction),
            overwrite=options.overwrite,
            device=options.device,
            no_hf_download=options.no_hf_download,
            positive_keypoint_labels=options.positive_keypoint_labels,
            roi_cache_policy=options.roi_cache_policy,
            roi_cache_dir=options.roi_cache_dir,
            roi_live_acceleration=options.roi_live_acceleration,
            roi_live_gpu_chunk_frames=int(options.roi_live_gpu_chunk_frames),
            profile_timings=bool(options.profile_timings),
        )
    except Exception as exc:
        return BatchRow(
            zarr_path=zarr_path,
            status="error",
            reason=str(exc),
            observed_use=observed_use,
            crop_run=crop_run or None,
            keypoint_group=keypoint_group or None,
            keypoint_run=keypoint_run or None,
            output_run=output_run or None,
            eligible_rows=eligible_rows,
        )

    return BatchRow(
        zarr_path=zarr_path,
        status="updated",
        observed_use=observed_use,
        crop_run=str(apply_summary.get("crop_run") or crop_run) or None,
        keypoint_group=str(apply_summary.get("keypoint_group") or keypoint_group) or None,
        keypoint_run=str(apply_summary.get("keypoint_run") or keypoint_run) or None,
        output_run=str(apply_summary.get("output_run") or output_run) or None,
        eligible_rows=int(apply_summary.get("rows_eligible") or eligible_rows),
        segmented_rows=int(apply_summary.get("rows_segmented") or 0),
        nonempty_rows=int(apply_summary.get("rows_with_nonempty_masks") or 0),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Batch inspect/apply SAM3 subject-mask generation across recording zarrs."
    )
    parser.add_argument("paths", nargs="*", type=Path, help="Recording roots or zarr paths.")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan for zarr archives.")
    parser.add_argument(
        "--zarr-use",
        choices=("analysis", "training", "any"),
        default="training",
        help="Filter zarr archives by purpose (default: training).",
    )
    parser.add_argument("--crop-run", type=str, help="Optional crop_runs/<run> override for every archive.")
    parser.add_argument("--keypoint-run", type=str, help="Optional keypoint run override for every archive.")
    parser.add_argument(
        "--keypoint-group",
        choices=list(KEYPOINT_GROUP_CHOICES),
        default="auto",
        help="Keypoint parent group preference (default: auto).",
    )
    parser.add_argument(
        "--prepare-count",
        type=int,
        default=DEFAULT_BATCH_PREPARE_COUNT,
        help=(
            "Preview row count for dry-run inspect summaries "
            f"(default: {DEFAULT_BATCH_PREPARE_COUNT}; use a positive value to materialize sample prompts)."
        ),
    )
    parser.add_argument(
        "--output-run",
        type=str,
        help="Optional explicit subject_mask_runs/<run> name for every archive.",
    )
    parser.add_argument("--sam3-root", type=Path, help="Optional SAM3 checkout root.")
    parser.add_argument("--checkpoint", type=Path, help="Optional local path to sam3.pt.")
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        help="Inference device when --apply is set (default: auto-detect).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_INFER_BATCH_SIZE,
        help=f"Eligible ROI batch size for SAM inference when --apply is used (default: {DEFAULT_INFER_BATCH_SIZE}).",
    )
    parser.add_argument("--single-mask", action="store_true", help="Use multimask_output=False during SAM inference.")
    parser.add_argument(
        "--box-prompt-source",
        choices=list(BOX_PROMPT_SOURCE_CHOICES),
        default=DEFAULT_BOX_PROMPT_SOURCE,
        help=f"Box prompt source when box prompting is enabled (default: {DEFAULT_BOX_PROMPT_SOURCE}).",
    )
    parser.add_argument(
        "--roi-inset-fraction",
        type=float,
        default=DEFAULT_ROI_INSET_FRACTION,
        help=f"Inset fraction for roi_inset box prompts (default: {DEFAULT_ROI_INSET_FRACTION}).",
    )
    parser.add_argument(
        "--pose-box-expand-fraction",
        type=float,
        default=DEFAULT_POSE_BOX_EXPAND_FRACTION,
        help=f"Expansion fraction for pose_roi box prompts (default: {DEFAULT_POSE_BOX_EXPAND_FRACTION}).",
    )
    parser.add_argument(
        "--negative-point-policy",
        choices=list(NEGATIVE_POINT_POLICY_CHOICES),
        default=DEFAULT_NEGATIVE_POINT_POLICY,
        help=f"Background negative-point prompt policy (default: {DEFAULT_NEGATIVE_POINT_POLICY}).",
    )
    parser.add_argument(
        "--negative-point-margin-fraction",
        type=float,
        default=DEFAULT_NEGATIVE_POINT_MARGIN_FRACTION,
        help=(
            "Border inset fraction for negative points, as a fraction of ROI width/height "
            f"(default: {DEFAULT_NEGATIVE_POINT_MARGIN_FRACTION})."
        ),
    )
    parser.add_argument(
        "--positive-keypoint-labels",
        nargs="+",
        help="Optional keypoint labels to use as positive SAM prompts for every archive.",
    )
    parser.add_argument("--no-box-prompt", action="store_true", help="Disable ROI-local box prompts.")
    parser.add_argument(
        "--no-hf-download",
        action="store_true",
        help="Do not allow SAM3 to download checkpoints from Hugging Face when --checkpoint is omitted.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output run when --apply is set.")
    parser.add_argument("--apply", action="store_true", help="Run SAM inference and write body-only subject-mask runs.")
    parser.add_argument(
        "--include-interpolated",
        action="store_true",
        help="Allow rows with detection_source != 0 to remain eligible.",
    )
    parser.add_argument(
        "--allow-zero-eligible",
        action="store_true",
        help="Do not skip archives whose inspect summary reports zero eligible rows.",
    )
    parser.add_argument(
        "--roi-cache-policy",
        choices=("never", "auto", "always"),
        default="auto",
        help="Temporary ROI cache policy for geometry-only crop runs (default: auto).",
    )
    parser.add_argument("--roi-cache-dir", type=str, help="Optional scratch directory for temporary ROI caches.")
    parser.add_argument(
        "--roi-live-acceleration",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="Live ROI read acceleration for geometry-only crop runs (default: auto).",
    )
    parser.add_argument(
        "--roi-live-gpu-chunk-frames",
        type=int,
        default=32,
        help="Frame batch size for GPU-accelerated live ROI reads (default: 32).",
    )
    parser.add_argument(
        "--profile-timings",
        action="store_true",
        help="Record and print per-stage timing diagnostics for SAM inference.",
    )
    args = parser.parse_args(argv)

    options = BatchOptions(
        apply=bool(args.apply),
        zarr_use_filter=str(args.zarr_use),
        crop_run=str(args.crop_run) if args.crop_run else None,
        keypoint_run=str(args.keypoint_run) if args.keypoint_run else None,
        keypoint_group=str(args.keypoint_group),
        prepare_count=int(args.prepare_count),
        output_run=str(args.output_run) if args.output_run else None,
        sam3_root=args.sam3_root,
        checkpoint=args.checkpoint,
        batch_size=int(args.batch_size),
        device=str(args.device) if args.device else None,
        multimask_output=not bool(args.single_mask),
        use_box_prompt=not bool(args.no_box_prompt),
        box_prompt_source=str(args.box_prompt_source),
        roi_inset_fraction=float(args.roi_inset_fraction),
        pose_box_expand_fraction=float(args.pose_box_expand_fraction),
        negative_point_policy=str(args.negative_point_policy),
        negative_point_margin_fraction=float(args.negative_point_margin_fraction),
        overwrite=bool(args.overwrite),
        no_hf_download=bool(args.no_hf_download),
        include_interpolated=bool(args.include_interpolated),
        positive_keypoint_labels=tuple(str(item) for item in args.positive_keypoint_labels)
        if args.positive_keypoint_labels
        else None,
        allow_zero_eligible=bool(args.allow_zero_eligible),
        roi_cache_policy=str(args.roi_cache_policy),
        roi_cache_dir=str(args.roi_cache_dir) if args.roi_cache_dir else None,
        roi_live_acceleration=str(args.roi_live_acceleration),
        roi_live_gpu_chunk_frames=int(args.roi_live_gpu_chunk_frames),
        profile_timings=bool(args.profile_timings),
    )

    roots = _resolve_roots(list(args.paths))
    rows: list[BatchRow] = []
    for zarr_path in _iter_zarr(roots, recursive=bool(args.recursive)):
        row = _process_zarr_path(zarr_path, options)
        rows.append(row)
        print(_format_row(row))

    if not rows:
        print("No zarr files found.")
        return 1

    counts = {
        "zarr_scanned": len(rows),
        "planned": 0,
        "updated": 0,
        "filtered_zarr_use": 0,
        "skipped_existing": 0,
        "skipped_no_eligible": 0,
        "error": 0,
    }
    for row in rows:
        counts[row.status] = counts.get(row.status, 0) + 1

    mode = "Applied" if options.apply else "Dry run"
    print(
        "Batch SAM subject masks: "
        f"mode={mode} scope={options.zarr_use_filter} zarr_scanned={counts['zarr_scanned']} "
        f"filtered_zarr_use={counts['filtered_zarr_use']} errors={counts['error']}"
    )
    print(
        f"Results: updated={counts['updated']} planned={counts['planned']} "
        f"skipped_existing={counts['skipped_existing']} skipped_no_eligible={counts['skipped_no_eligible']}"
    )
    if not options.apply:
        print("Dry run: add --apply to run SAM subject-mask inference.")
    return 0 if counts["error"] == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
