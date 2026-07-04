#!/usr/bin/env python3
"""Resolve refined subject-mask stale markers after accepted source-drift review."""

from __future__ import annotations

from fisheye.shared.zarr_discovery import iter_filesystem_zarrs as _iter_zarr
import argparse
import os
from pathlib import Path
from typing import Iterable, List, Optional

import zarr

from ..shared.subject_mask_stale import resolve_downstream_subject_mask_runs_stale


def _load_paths_file(path: Path) -> List[Path]:
    lines = path.read_text(encoding="utf-8").splitlines()
    items: List[Path] = []
    for line in lines:
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        items.append(Path(value))
    return items


def _infer_zarr_use(root: zarr.Group, zarr_path: Path) -> str:
    for key in ("zarr_use", "zarr_purpose"):
        raw = root.attrs.get(key)
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
    return "unknown"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="Recording roots or zarr paths.")
    parser.add_argument("--file-list", type=Path, action="append", help="Path list file (one zarr per line).")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan for zarrs.")
    parser.add_argument(
        "--zarr-use",
        choices=["all", "analysis", "training"],
        default="all",
        help="Filter by zarr use (default: all).",
    )
    parser.add_argument("--refined-run", help="Only resolve stale on this refined_subject_masks run.")
    parser.add_argument(
        "--source-subject-mask-run",
        help="Only resolve stale for refined runs sourced from this subject_mask_runs/<run>.",
    )
    parser.add_argument(
        "--resolution",
        default="manual_accept_after_subject_mask_source_update_preserve_masks",
        help="Resolution label written into source_subject_mask_stale.resolution.",
    )
    parser.add_argument("--reviewer", default=os.environ.get("USER"), help="Resolver identity.")
    parser.add_argument("--notes", help="Optional resolution notes.")
    parser.add_argument("--apply", action="store_true", help="Apply changes (default: dry-run).")
    args = parser.parse_args(argv)

    roots: List[Path] = []
    if args.file_list:
        for path in args.file_list:
            roots.extend(_load_paths_file(path))
    roots.extend(args.paths)
    if not roots:
        roots = [Path("/nvme1/recordings")]

    summary = {"scanned": 0, "resolved": 0, "skipped": 0, "errors": 0, "filtered_zarr_use": 0}
    for zarr_path in _iter_zarr(roots, args.recursive):
        summary["scanned"] += 1
        try:
            root = zarr.open_group(str(zarr_path), mode="a", use_consolidated=False)
        except Exception as exc:  # pragma: no cover - defensive
            summary["errors"] += 1
            print(f"{zarr_path}: error opening zarr ({exc})")
            continue

        observed_use = _infer_zarr_use(root, zarr_path)
        if args.zarr_use != "all" and observed_use != args.zarr_use:
            summary["filtered_zarr_use"] += 1
            continue

        touched = resolve_downstream_subject_mask_runs_stale(
            root,
            refined_run=str(args.refined_run) if args.refined_run else None,
            source_subject_mask_run=str(args.source_subject_mask_run) if args.source_subject_mask_run else None,
            resolution=str(args.resolution),
            reviewer=args.reviewer,
            notes=args.notes,
            dry_run=not args.apply,
        )
        if touched == 0:
            summary["skipped"] += 1
            detail = str(args.refined_run or args.source_subject_mask_run or "selection")
            print(f"{zarr_path}: no stale runs to resolve for {detail}")
            continue

        summary["resolved"] += touched
        mode = "resolved" if args.apply else "would_resolve"
        selector = str(args.refined_run or args.source_subject_mask_run or "stale_subject_masks")
        print(f"{zarr_path}: {mode}={touched} selection={selector}")

    mode = "Apply" if args.apply else "Dry run"
    print(
        f"{mode} summary: scanned={summary['scanned']} resolved={summary['resolved']} "
        f"skipped={summary['skipped']} errors={summary['errors']} filtered_zarr_use={summary['filtered_zarr_use']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
