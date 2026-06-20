#!/usr/bin/env python3
"""List Zarr archives with incomplete refined detect subgroups."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import zarr

# This audit targets legacy refined-detect subgroup layouts ("filtered",
# "interpolated", and manual review groups). The current refined-detect contract
# uses source_detections/instances, so keep this legacy minimum explicit instead
# of deriving it from the modern StageSpec.
REQUIRED_DETECT_ARRAYS = ("frame_indices", "bbox_norm_coords")


@dataclass
class RefinedGroupIssue:
    zarr_path: str
    zarr_use: str
    refined_run: str
    subgroup: str
    issue: str
    missing_arrays: Optional[str] = None
    detail: Optional[str] = None


def _resolve_roots(paths: Optional[List[Path]]) -> List[Path]:
    if paths:
        return paths
    env_root = os.environ.get("PALETTE_RECORDINGS_ROOT")
    if env_root:
        return [Path(env_root)]
    return [Path("/nvme1/recordings")]


def _iter_zarr(roots: List[Path], recursive: bool) -> Iterable[Path]:
    for root in roots:
        root = root.expanduser()
        if root.is_file() and root.suffix == ".zarr":
            yield root
            continue
        if not root.exists():
            continue
        if recursive:
            yield from root.rglob("zarr/*.zarr")
        else:
            yield from root.glob("*/zarr/*.zarr")


def _infer_zarr_use(zarr_path: Path, root: zarr.Group) -> str:
    purpose = root.attrs.get("zarr_purpose")
    if purpose is not None:
        norm = str(purpose).strip().lower()
        if norm in {"analysis", "training"}:
            return norm
    name = zarr_path.name.lower()
    if name.endswith("_analysis.zarr"):
        return "analysis"
    if name.endswith("_training.zarr"):
        return "training"
    return "unknown"


def _subgroups_to_validate(refined_run: zarr.Group) -> List[str]:
    keys: List[str] = []
    for key in ("filtered", "interpolated"):
        if key in refined_run:
            keys.append(key)

    manual_latest = refined_run.attrs.get("manual_review_latest")
    if manual_latest is not None:
        manual_label = str(manual_latest).strip()
        if manual_label and manual_label in refined_run and manual_label not in keys:
            keys.append(manual_label)
    if "manual" in refined_run and "manual" not in keys:
        keys.append("manual")
    return keys


def _validate_subgroup(
    *,
    zarr_path: Path,
    zarr_use: str,
    refined_run_name: str,
    subgroup_name: str,
    subgroup: zarr.Group,
) -> List[RefinedGroupIssue]:
    issues: List[RefinedGroupIssue] = []
    missing = [name for name in REQUIRED_DETECT_ARRAYS if name not in subgroup]
    if missing:
        issues.append(
            RefinedGroupIssue(
                zarr_path=str(zarr_path),
                zarr_use=zarr_use,
                refined_run=refined_run_name,
                subgroup=subgroup_name,
                issue="missing_required_arrays",
                missing_arrays=",".join(missing),
            )
        )
        return issues

    try:
        frame_count = int(subgroup["frame_indices"].shape[0])
        bbox_count = int(subgroup["bbox_norm_coords"].shape[0])
    except Exception as exc:
        issues.append(
            RefinedGroupIssue(
                zarr_path=str(zarr_path),
                zarr_use=zarr_use,
                refined_run=refined_run_name,
                subgroup=subgroup_name,
                issue="array_shape_read_failed",
                detail=str(exc),
            )
        )
        return issues

    if frame_count != bbox_count:
        issues.append(
            RefinedGroupIssue(
                zarr_path=str(zarr_path),
                zarr_use=zarr_use,
                refined_run=refined_run_name,
                subgroup=subgroup_name,
                issue="row_count_mismatch",
                detail=f"frame_indices={frame_count}, bbox_norm_coords={bbox_count}",
            )
        )

    if "detection_source" in subgroup:
        try:
            source_count = int(subgroup["detection_source"].shape[0])
            if source_count != frame_count:
                issues.append(
                    RefinedGroupIssue(
                        zarr_path=str(zarr_path),
                        zarr_use=zarr_use,
                        refined_run=refined_run_name,
                        subgroup=subgroup_name,
                        issue="detection_source_row_mismatch",
                        detail=f"frame_indices={frame_count}, detection_source={source_count}",
                    )
                )
        except Exception as exc:
            issues.append(
                RefinedGroupIssue(
                    zarr_path=str(zarr_path),
                    zarr_use=zarr_use,
                    refined_run=refined_run_name,
                    subgroup=subgroup_name,
                    issue="detection_source_shape_read_failed",
                    detail=str(exc),
                )
            )
    return issues


def _collect_issues(
    roots: List[Path],
    *,
    recursive: bool,
    zarr_use_filter: str,
    latest_only: bool,
) -> List[RefinedGroupIssue]:
    issues: List[RefinedGroupIssue] = []
    for zarr_path in _iter_zarr(roots, recursive):
        try:
            root = zarr.open_group(str(zarr_path), mode="r")
        except Exception as exc:
            issues.append(
                RefinedGroupIssue(
                    zarr_path=str(zarr_path),
                    zarr_use="unknown",
                    refined_run="<open>",
                    subgroup="<zarr>",
                    issue="zarr_open_failed",
                    detail=str(exc),
                )
            )
            continue

        zarr_use = _infer_zarr_use(zarr_path, root)
        if zarr_use_filter != "any" and zarr_use != zarr_use_filter:
            continue
        if "refined_detect_runs" not in root:
            continue

        refined_parent = root["refined_detect_runs"]
        if latest_only:
            latest = refined_parent.attrs.get("latest")
            run_names = [str(latest).strip()] if latest else []
        else:
            run_names = sorted(refined_parent.group_keys())

        for run_name in run_names:
            if not run_name:
                continue
            if run_name not in refined_parent:
                issues.append(
                    RefinedGroupIssue(
                        zarr_path=str(zarr_path),
                        zarr_use=zarr_use,
                        refined_run=run_name,
                        subgroup="<refined_run>",
                        issue="refined_run_missing",
                    )
                )
                continue
            run_group = refined_parent[run_name]
            for subgroup_name in _subgroups_to_validate(run_group):
                subgroup = run_group[subgroup_name]
                issues.extend(
                    _validate_subgroup(
                        zarr_path=zarr_path,
                        zarr_use=zarr_use,
                        refined_run_name=run_name,
                        subgroup_name=subgroup_name,
                        subgroup=subgroup,
                    )
                )

    issues.sort(key=lambda row: (row.zarr_path, row.refined_run, row.subgroup, row.issue))
    return issues


def _write_path_list(rows: List[RefinedGroupIssue], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    unique_paths = sorted({row.zarr_path for row in rows})
    payload = "\n".join(unique_paths)
    if payload:
        payload += "\n"
    output_path.write_text(payload, encoding="utf-8")


def _write_tsv(rows: List[RefinedGroupIssue], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "zarr_path\tzarr_use\trefined_run\tsubgroup\tissue\tmissing_arrays\tdetail",
    ]
    for row in rows:
        lines.append(
            "\t".join(
                [
                    row.zarr_path,
                    row.zarr_use,
                    row.refined_run,
                    row.subgroup,
                    row.issue,
                    row.missing_arrays or "",
                    row.detail or "",
                ]
            )
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Scan refined detect subgroups and list archives with incomplete or inconsistent "
            "subgroup arrays."
        )
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Recording roots or zarr paths (default: $PALETTE_RECORDINGS_ROOT or /nvme1/recordings).",
    )
    parser.add_argument("--recursive", action="store_true", help="Recursively scan for zarr archives.")
    parser.add_argument(
        "--zarr-use",
        choices=["analysis", "training", "any"],
        default="analysis",
        help="Filter by zarr use (default: analysis).",
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Check all refined runs (default: latest run only).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/incomplete_refined_detect_zarrs.txt"),
        help="Output file with one problematic zarr path per line.",
    )
    parser.add_argument(
        "--details",
        type=Path,
        default=Path("/tmp/incomplete_refined_detect_groups.tsv"),
        help="Detailed TSV output for each issue row.",
    )
    parser.add_argument("--json", action="store_true", help="Print issue rows as JSON.")
    args = parser.parse_args(argv)

    rows = _collect_issues(
        _resolve_roots(args.paths),
        recursive=bool(args.recursive),
        zarr_use_filter=str(args.zarr_use),
        latest_only=not bool(args.all_runs),
    )

    _write_path_list(rows, args.output)
    _write_tsv(rows, args.details)

    if args.json:
        print(json.dumps([asdict(row) for row in rows], indent=2))
    else:
        print(f"wrote: {args.output}")
        print(f"details: {args.details}")
        print(f"problem_zarrs: {len(set(row.zarr_path for row in rows))}")
        print(f"issues: {len(rows)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
