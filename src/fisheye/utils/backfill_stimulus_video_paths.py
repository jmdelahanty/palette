#!/usr/bin/env python3
"""Backfill source_stimulus_video_path on analysis stimulus runs.

This utility targets analysis archives (`*_analysis.zarr` or `zarr_purpose=analysis`).
Training archives are intentionally skipped because they are not expected to
carry stimulus-run provenance attrs.
"""

from __future__ import annotations

from fisheye.shared.zarr_discovery import iter_filesystem_zarrs as _iter_zarr
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import zarr


@dataclass
class BackfillResult:
    status: str
    reason: Optional[str] = None


def _infer_zarr_use(root: zarr.Group, zarr_path: Path) -> Optional[str]:
    purpose = root.attrs.get("zarr_purpose")
    if purpose is not None:
        value = str(purpose).strip().lower()
        if value in {"analysis", "training"}:
            return value
    name = zarr_path.name.lower()
    if name.endswith("_analysis.zarr"):
        return "analysis"
    if name.endswith("_training.zarr"):
        return "training"
    return None


def _resolve_rendered_video_from_source_h5(source_h5: object) -> Optional[Path]:
    if source_h5 is None:
        return None
    try:
        h5_path = Path(str(source_h5)).expanduser()
    except Exception:
        return None
    rendered = h5_path.with_suffix(".mp4")
    if rendered.exists() and rendered.is_file():
        return rendered.resolve()
    return None


def _select_stimulus_runs(root: zarr.Group, all_runs: bool) -> list[zarr.Group]:
    analysis = root.get("analysis")
    if analysis is None:
        return []
    runs_parent = analysis.get("stimulus_runs")
    if runs_parent is None:
        return []
    if all_runs:
        try:
            names = sorted(list(runs_parent.group_keys()))
        except Exception:
            names = sorted(list(runs_parent.keys()))
        return [runs_parent[name] for name in names if name in runs_parent]
    latest = runs_parent.attrs.get("latest")
    if latest and latest in runs_parent:
        return [runs_parent[str(latest)]]
    try:
        names = sorted(list(runs_parent.group_keys()))
    except Exception:
        names = sorted(list(runs_parent.keys()))
    if not names:
        return []
    return [runs_parent[names[-1]]]


def _backfill_run_group(
    run_group: zarr.Group,
    *,
    overwrite_existing: bool,
    apply: bool,
) -> BackfillResult:
    existing = run_group.attrs.get("source_stimulus_video_path")
    if existing and not overwrite_existing:
        return BackfillResult(status="skipped_existing")

    source_h5 = run_group.attrs.get("source_h5")
    if source_h5 in (None, ""):
        return BackfillResult(status="no_source_h5", reason="source_h5 missing")

    rendered = _resolve_rendered_video_from_source_h5(source_h5)
    if rendered is None:
        return BackfillResult(status="source_video_missing", reason=f"rendered video not found for {source_h5}")

    if apply:
        run_group.attrs["source_stimulus_video_path"] = str(rendered)
        return BackfillResult(status="updated")
    return BackfillResult(status="would_update")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="Recording roots or zarr paths.")
    parser.add_argument("--recursive", action="store_true", help="Recursively search for .zarr archives.")
    parser.add_argument(
        "--zarr-use",
        choices=["analysis", "any"],
        default="analysis",
        help="Archive scope (default: analysis).",
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Process all analysis/stimulus_runs (default: latest only).",
    )
    parser.add_argument("--overwrite-existing", action="store_true", help="Overwrite existing attr value.")
    parser.add_argument("--apply", action="store_true", help="Write updates (default: dry-run).")
    args = parser.parse_args(argv)

    roots = list(args.paths) if args.paths else [Path("/nvme1/recordings")]
    counts = {
        "zarr_scanned": 0,
        "runs_considered": 0,
        "filtered_zarr_use": 0,
        "training_not_expected": 0,
        "missing_stimulus_runs": 0,
        "updated": 0,
        "would_update": 0,
        "skipped_existing": 0,
        "no_source_h5": 0,
        "source_video_missing": 0,
        "errors": 0,
    }

    any_zarr = False
    for zarr_path in _iter_zarr(roots, recursive=bool(args.recursive)):
        any_zarr = True
        counts["zarr_scanned"] += 1
        try:
            root = zarr.open_group(str(zarr_path), mode="a" if args.apply else "r")
            observed_use = _infer_zarr_use(root, zarr_path)
            if args.zarr_use == "analysis" and observed_use != "analysis":
                counts["filtered_zarr_use"] += 1
                continue
            if args.zarr_use == "any" and observed_use == "training":
                counts["training_not_expected"] += 1
                continue

            run_groups = _select_stimulus_runs(root, all_runs=bool(args.all_runs))
            if not run_groups:
                counts["missing_stimulus_runs"] += 1
                continue

            for run_group in run_groups:
                counts["runs_considered"] += 1
                result = _backfill_run_group(
                    run_group,
                    overwrite_existing=bool(args.overwrite_existing),
                    apply=bool(args.apply),
                )
                counts[result.status] += 1
        except Exception as exc:
            counts["errors"] += 1
            print(f"error: {zarr_path}: {exc}")

    if not any_zarr:
        print("No zarr files found.")
        return 1

    mode = "Applied" if args.apply else "Dry run"
    print(
        "Stimulus video path backfill: "
        f"scope={args.zarr_use} zarr_scanned={counts['zarr_scanned']} "
        f"runs_considered={counts['runs_considered']} filtered_zarr_use={counts['filtered_zarr_use']} "
        f"training_not_expected={counts['training_not_expected']} "
        f"missing_stimulus_runs={counts['missing_stimulus_runs']} errors={counts['errors']}"
    )
    print(
        f"{mode}: updated={counts['updated']} would_update={counts['would_update']} "
        f"skipped_existing={counts['skipped_existing']} no_source_h5={counts['no_source_h5']} "
        f"source_video_missing={counts['source_video_missing']}"
    )
    return 0 if counts["errors"] == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
