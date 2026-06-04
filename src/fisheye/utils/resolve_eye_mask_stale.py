#!/usr/bin/env python3
"""Resolve downstream subject-mask stale markers after intentional keypoint nudges.

Deprecated name: this CLI used to resolve legacy eye-mask stale markers. Phase 1
re-points keypoint edit invalidation to refined-subject mask runs while keeping
the old module path as an operator-compatible shim.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import zarr

from ..shared.subject_mask_stale import resolve_downstream_subject_mask_runs_stale


def _iter_zarr(paths: List[Path], recursive: bool) -> Iterable[Path]:
    for path in paths:
        path = path.expanduser()
        if path.is_dir() and path.suffix == ".zarr":
            yield path
            continue
        if not path.exists():
            continue
        if recursive:
            yield from path.rglob("*.zarr")
        else:
            yield from path.glob("*/zarr/*.zarr")
            yield from path.glob("*.zarr")


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


def _resolve_source_keypoint_lineage(
    root: zarr.Group,
    *,
    source_group: Optional[str],
    source_run: Optional[str],
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if source_run:
        if source_group:
            group = root.get(source_group)
            if not isinstance(group, zarr.Group):
                return None, None, f"missing keypoint group '{source_group}'"
            if source_run not in group:
                return None, None, f"missing run '{source_group}/{source_run}'"
            return source_group, source_run, None
        # No explicit group: prefer refined_keypoints_runs, then keypoints_runs.
        for candidate in ("refined_keypoints_runs", "keypoints_runs"):
            group = root.get(candidate)
            if isinstance(group, zarr.Group) and source_run in group:
                return candidate, source_run, None
        return None, None, f"source keypoint run '{source_run}' not found"

    # Group specified without explicit run: use latest from that group.
    if source_group:
        group = root.get(source_group)
        if not isinstance(group, zarr.Group):
            return None, None, f"missing keypoint group '{source_group}'"
        latest = group.attrs.get("latest")
        if latest and latest in group:
            return source_group, str(latest), None
        return None, None, f"no latest run in '{source_group}'"

    # No explicit run/group: resolve latest refined, fallback raw.
    for candidate in ("refined_keypoints_runs", "keypoints_runs"):
        group = root.get(candidate)
        if not isinstance(group, zarr.Group):
            continue
        latest = group.attrs.get("latest")
        if latest and latest in group:
            return candidate, str(latest), None
    return None, None, "no keypoint source run found"


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
    parser.add_argument("--source-keypoint-group", choices=["refined_keypoints_runs", "keypoints_runs"])
    parser.add_argument("--source-keypoints-run", help="Source keypoint run name (default: latest).")
    parser.add_argument(
        "--resolution",
        default="manual_accept_after_keypoint_nudge_preserve_masks",
        help="Resolution label written into source_keypoint_stale.resolution.",
    )
    parser.add_argument("--reviewer", default=os.environ.get("USER"), help="Resolver identity.")
    parser.add_argument("--notes", help="Optional resolution notes.")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes (default: dry-run).",
    )
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

        source_group, source_run, err = _resolve_source_keypoint_lineage(
            root,
            source_group=args.source_keypoint_group,
            source_run=args.source_keypoints_run,
        )
        if err:
            summary["skipped"] += 1
            print(f"{zarr_path}: skipped ({err})")
            continue

        touched = resolve_downstream_subject_mask_runs_stale(
            root,
            source_keypoint_group=str(source_group),
            source_keypoints_run=str(source_run),
            resolution=str(args.resolution),
            reviewer=args.reviewer,
            notes=args.notes,
            dry_run=not args.apply,
        )
        if touched == 0:
            summary["skipped"] += 1
            print(f"{zarr_path}: no subject-mask stale runs to resolve for {source_group}/{source_run}")
            continue

        summary["resolved"] += touched
        mode = "resolved" if args.apply else "would_resolve"
        print(f"{zarr_path}: {mode}={touched} source={source_group}/{source_run}")

    mode = "Apply" if args.apply else "Dry run"
    print(
        f"{mode} summary: scanned={summary['scanned']} resolved={summary['resolved']} "
        f"skipped={summary['skipped']} errors={summary['errors']} filtered_zarr_use={summary['filtered_zarr_use']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
