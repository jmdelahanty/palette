#!/usr/bin/env python3
"""Batch inspect/apply utility for traditional swim-bladder segmentation across recording zarrs."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

from fisheye.segmentation import swim_bladder_segmentation as swim_mod
from fisheye.utils.apply_tuning_by_camera import (
    _camera_id_for_zarr,
    _normalize_subject_mask_tuning_payload,
)
from fisheye.utils.zarr_io import open_zarr_root


DEFAULT_RECORDINGS_ROOT = Path("/nvme1/recordings")


@dataclass(frozen=True)
class BatchOptions:
    apply: bool
    zarr_use_filter: str
    output_run: Optional[str]
    overwrite: bool
    require_swim_tuning: bool
    scheduler: str
    num_workers: Optional[int]
    config_dict: Mapping[str, object]


@dataclass(frozen=True)
class BatchRow:
    zarr_path: Path
    status: str
    reason: Optional[str] = None
    observed_use: Optional[str] = None
    camera_id: Optional[str] = None
    output_run: Optional[str] = None
    matched_components: tuple[str, ...] = ()


def _resolve_roots(paths: list[Path]) -> list[Path]:
    if paths:
        return paths
    env_root = os.environ.get("PALETTE_RECORDINGS_ROOT")
    if env_root:
        return [Path(env_root)]
    return [DEFAULT_RECORDINGS_ROOT]


def _iter_zarr(roots: list[Path], recursive: bool) -> Iterable[Path]:
    seen: set[str] = set()
    for root in roots:
        root = root.expanduser()
        candidates: list[Path] = []
        if root.suffix == ".zarr" and (root.is_dir() or root.is_file()):
            candidates = [root]
        elif root.exists():
            if recursive:
                candidates = sorted(root.rglob("*.zarr"))
            else:
                candidates = sorted(root.glob("*.zarr")) + sorted(root.glob("*/zarr/*.zarr"))
        for candidate in candidates:
            try:
                key = str(candidate.resolve())
            except OSError:
                key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            yield candidate


def _infer_zarr_use(root: object, zarr_path: Path) -> Optional[str]:
    attrs = getattr(root, "attrs", {})
    for key in ("zarr_use", "zarr_purpose"):
        raw = attrs.get(key)
        if raw is None:
            continue
        value = str(raw).strip().lower()
        if value in {"analysis", "training"}:
            return value
    name = zarr_path.name.lower()
    if name.endswith("_analysis.zarr"):
        return "analysis"
    if name.endswith("_training.zarr"):
        return "training"
    return None


def _resolve_target_run_name(output_run: Optional[str]) -> str:
    return output_run or "(auto)"


def _output_run_exists(root: object, output_run: Optional[str]) -> bool:
    if not output_run:
        return False
    parent = root.get("subject_mask_runs")
    return parent is not None and output_run in parent


def _matched_tuning_components(root: object) -> tuple[str, ...]:
    analysis = root.get("analysis_metadata")
    if analysis is None:
        return ()
    attrs = dict(getattr(analysis, "attrs", {}) or {})
    subject_tuning = attrs.get("subject_mask_tuning")
    if not isinstance(subject_tuning, Mapping):
        return ()
    payload = _normalize_subject_mask_tuning_payload(subject_tuning)
    components = payload.get("components", {})
    if not isinstance(components, Mapping):
        return ()
    matched = [str(name) for name in sorted(components) if str(name) == "swim_bladder"]
    return tuple(matched)


def _format_row(row: BatchRow) -> str:
    use = row.observed_use or "unknown"
    camera_id = row.camera_id or "unknown"
    target = f"subject_mask_runs/{_resolve_target_run_name(row.output_run)}"
    matched = ",".join(row.matched_components) if row.matched_components else "-"
    reason = row.reason or "-"
    return (
        f"{row.status}\t{row.zarr_path}\tuse={use}\tcamera_id={camera_id}\t"
        f"target={target}\tmatched={matched}\t{reason}"
    )


def _process_zarr_path(zarr_path: Path, options: BatchOptions) -> BatchRow:
    try:
        root = open_zarr_root(zarr_path, mode="r")
    except Exception as exc:
        return BatchRow(zarr_path=zarr_path, status="error", reason=f"failed to open archive ({exc})")

    observed_use = _infer_zarr_use(root, zarr_path)
    camera_id = _camera_id_for_zarr(zarr_path, root)
    if options.zarr_use_filter != "any" and observed_use != options.zarr_use_filter:
        return BatchRow(
            zarr_path=zarr_path,
            status="filtered_zarr_use",
            reason=f"wanted={options.zarr_use_filter} found={observed_use or 'unknown'}",
            observed_use=observed_use,
            camera_id=camera_id,
            output_run=options.output_run,
        )

    matched_components = _matched_tuning_components(root)
    if options.require_swim_tuning and not matched_components:
        return BatchRow(
            zarr_path=zarr_path,
            status="skipped_missing_tuning",
            reason="analysis_metadata.subject_mask_tuning.components.swim_bladder missing",
            observed_use=observed_use,
            camera_id=camera_id,
            output_run=options.output_run,
        )

    if _output_run_exists(root, options.output_run) and not options.overwrite:
        return BatchRow(
            zarr_path=zarr_path,
            status="skipped_existing",
            reason=f"subject_mask_runs/{options.output_run} already exists",
            observed_use=observed_use,
            camera_id=camera_id,
            output_run=options.output_run,
            matched_components=matched_components,
        )

    if not options.apply:
        return BatchRow(
            zarr_path=zarr_path,
            status="planned",
            observed_use=observed_use,
            camera_id=camera_id,
            output_run=options.output_run,
            matched_components=matched_components,
        )

    try:
        run_name = swim_mod.segment_swim_bladder_masks(
            zarr_path,
            config_dict=options.config_dict,
            console=None,
            output_run=options.output_run,
            overwrite=options.overwrite,
            scheduler=options.scheduler,
            num_workers=options.num_workers,
        )
    except Exception as exc:
        return BatchRow(
            zarr_path=zarr_path,
            status="error",
            reason=str(exc),
            observed_use=observed_use,
            camera_id=camera_id,
            output_run=options.output_run,
            matched_components=matched_components,
        )

    return BatchRow(
        zarr_path=zarr_path,
        status="updated",
        observed_use=observed_use,
        camera_id=camera_id,
        output_run=run_name,
        matched_components=matched_components,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Batch inspect/apply traditional swim-bladder segmentation across recording zarrs."
    )
    parser.add_argument("paths", nargs="*", type=Path, help="Recording roots or zarr paths.")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan roots for zarr archives.")
    parser.add_argument(
        "--zarr-use",
        choices=("analysis", "training", "any"),
        default="training",
        help="Filter zarr archives by purpose (default: training).",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        required=True,
        help="Explicit subject_mask_runs/<run> output name for every archive.",
    )
    parser.add_argument(
        "--scheduler",
        type=swim_mod._parse_scheduler_arg,
        default="single-threaded",
        help=(
            "Scheduler to use for ROI processing. Accepted values: threads, processes, distributed, "
            "single-threaded, single-thread, single_thread. Default: single-threaded."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help="Optional number of worker threads/processes for scheduler-backed ROI processing.",
    )
    parser.add_argument("--crop-run", type=str, help="Optional crop_runs/<run> override for every archive.")
    parser.add_argument("--keypoint-run", type=str, help="Optional keypoint run override for every archive.")
    parser.add_argument(
        "--subject-method-family",
        type=str,
        help="Optional swim-bladder method family override for every archive.",
    )
    parser.add_argument("--roi-padding", type=int, help="Override swim-bladder patch padding.")
    parser.add_argument("--pre-threshold", type=int, help="Override patch pre-threshold.")
    parser.add_argument("--sobel-strength", type=float, help="Override Sobel strength.")
    parser.add_argument("--min-area", type=int, help="Override minimum connected-component area.")
    parser.add_argument("--max-area", type=int, help="Override maximum connected-component area.")
    parser.add_argument("--min-circularity", type=float, help="Override minimum circularity.")
    parser.add_argument("--closing-radius", type=int, help="Override morphological closing radius.")
    parser.add_argument("--opening-radius", type=int, help="Override morphological opening radius.")
    parser.add_argument("--angle-step-degrees", type=int, help="Override polar boundary angle step.")
    parser.add_argument("--min-radius-px", type=int, help="Override polar boundary minimum radius.")
    parser.add_argument("--max-radius-px", type=int, help="Override polar boundary maximum radius.")
    parser.add_argument("--smoothing-sigma", type=float, help="Override polar boundary smoothing sigma.")
    parser.add_argument("--response-threshold", type=float, help="Override polar boundary response threshold.")
    parser.add_argument("--gradient-mode", type=str, help="Override polar boundary gradient response.")
    parser.add_argument(
        "--max-missing-gap-degrees",
        type=int,
        help="Override maximum allowed missing-angle gap before rejecting a polar proposal.",
    )
    parser.add_argument(
        "--min-valid-ray-fraction",
        type=float,
        help="Override minimum valid-ray fraction required for a polar proposal.",
    )
    parser.add_argument("--prefilter-sigma", type=float, help="Override polar boundary prefilter sigma.")
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
        "--allow-missing-tuning",
        action="store_true",
        help="Allow segmentation even when subject_mask_tuning.components.swim_bladder is missing.",
    )
    parser.add_argument(
        "--show-skips",
        action="store_true",
        help="Print filtered and skipped rows in addition to planned/updated rows.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing subject_mask_runs/<run> when it already exists.",
    )
    parser.add_argument("--apply", action="store_true", help="Run segmentation and write swim-bladder mask runs.")
    args = parser.parse_args(argv)

    options = BatchOptions(
        apply=bool(args.apply),
        zarr_use_filter=str(args.zarr_use),
        output_run=str(args.run_name),
        overwrite=bool(args.overwrite),
        require_swim_tuning=not bool(args.allow_missing_tuning),
        scheduler=str(args.scheduler),
        num_workers=int(args.num_workers) if args.num_workers is not None else None,
        config_dict=swim_mod._build_cli_overrides(args),
    )

    roots = _resolve_roots(list(args.paths))
    rows: list[BatchRow] = []
    for zarr_path in _iter_zarr(roots, bool(args.recursive)):
        rows.append(_process_zarr_path(zarr_path, options))

    counts: dict[str, int] = {}
    for row in rows:
        counts[row.status] = counts.get(row.status, 0) + 1
        if row.status in {"planned", "updated"} or args.show_skips:
            print(_format_row(row))

    print(
        "Batch swim-bladder segmentation: "
        f"mode={'Applied' if args.apply else 'Dry run'} "
        f"scope={args.zarr_use} "
        f"zarr_scanned={len(rows)} "
        f"filtered_zarr_use={counts.get('filtered_zarr_use', 0)} "
        f"errors={counts.get('error', 0)}"
    )
    print(
        "Results: "
        f"updated={counts.get('updated', 0)} "
        f"planned={counts.get('planned', 0)} "
        f"skipped_existing={counts.get('skipped_existing', 0)} "
        f"skipped_missing_tuning={counts.get('skipped_missing_tuning', 0)}"
    )
    if not args.apply:
        print("Dry run: add --apply to write swim-bladder subject_mask_runs.")
    return 0 if counts.get("error", 0) == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
