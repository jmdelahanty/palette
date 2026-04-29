#!/usr/bin/env python3
"""Backfill component contour caches for refined subject-mask runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import zarr

from fisheye.shared.refined_subject_component_contours import (
    ComponentContourSummary,
    build_component_contours_from_masks,
    summarize_existing_component_contours,
    write_refined_subject_component_contours,
)
from fisheye.utils.zarr_io import open_zarr_root


DEFAULT_COMPONENTS = ("subject_body", "swim_bladder")


def _resolve_refined_run(root: zarr.Group, refined_run: str | None) -> tuple[str, zarr.Group]:
    parent = root.get("refined_subject_masks_runs")
    if parent is None:
        raise ValueError("Archive has no refined_subject_masks_runs group.")
    run_name = str(refined_run or parent.attrs.get("latest") or "")
    if not run_name:
        candidates = sorted(str(name) for name in parent.keys())
        if not candidates:
            raise ValueError("refined_subject_masks_runs has no runs.")
        run_name = candidates[-1]
    if run_name not in parent:
        raise ValueError(f"refined_subject_masks_runs/{run_name} not found.")
    return run_name, parent[run_name]


def _parse_components(values: Optional[Sequence[Sequence[str]]], single_values: Optional[Sequence[str]]) -> list[str]:
    merged: list[str] = []
    for group in values or ():
        merged.extend(str(value) for value in group)
    for value in single_values or ():
        merged.append(str(value))
    if not merged:
        merged = list(DEFAULT_COMPONENTS)
    deduped: list[str] = []
    seen: set[str] = set()
    for component in merged:
        name = str(component).strip()
        if name and name not in seen:
            deduped.append(name)
            seen.add(name)
    return deduped


def _summary_to_dict(summary: ComponentContourSummary) -> dict[str, object]:
    return {
        "component": summary.component,
        "status": summary.status,
        "reason": summary.reason,
        "roi_count": int(summary.roi_count),
        "contour_count": int(summary.contour_count),
        "point_count": int(summary.point_count),
        "existing": bool(summary.existing),
    }


def _get_group_path(group: zarr.Group, path: str) -> zarr.Group | None:
    current: object = group
    for part in path.split("/"):
        if not isinstance(current, zarr.Group) or part not in current:
            return None
        current = current[part]
    return current if isinstance(current, zarr.Group) else None


def _dry_run_component(
    refined_group: zarr.Group,
    component: str,
    *,
    min_points: int,
) -> ComponentContourSummary:
    masks_roi = refined_group.get("masks_roi")
    roi_count = int(masks_roi.shape[0]) if masks_roi is not None and len(tuple(masks_roi.shape)) >= 1 else 0
    component_group = _get_group_path(refined_group, f"components/{component}")
    existing_summary = (
        summarize_existing_component_contours(component_group, component=component, roi_count=roi_count)
        if isinstance(component_group, zarr.Group)
        else None
    )
    if existing_summary is not None:
        return existing_summary
    _contours, summary = build_component_contours_from_masks(
        refined_group,
        component,
        min_points=min_points,
    )
    if summary.status == "computed":
        return ComponentContourSummary(
            component=component,
            status="would_write",
            roi_count=summary.roi_count,
            contour_count=summary.contour_count,
            point_count=summary.point_count,
        )
    return summary


def backfill_refined_subject_component_contours(
    zarr_path: Path,
    *,
    refined_run: str | None = None,
    components: Sequence[str] = DEFAULT_COMPONENTS,
    apply: bool = False,
    overwrite: bool = False,
    chunk_size: int = 256,
    min_points: int = 2,
) -> dict[str, object]:
    root = open_zarr_root(zarr_path, mode="a" if apply else "r")
    run_name, refined_group = _resolve_refined_run(root, refined_run)
    component_names = [str(component) for component in components]

    if apply:
        summaries = write_refined_subject_component_contours(
            refined_group,
            components=component_names,
            source_mask_run=run_name,
            chunk_rois=max(1, int(chunk_size)),
            min_points=max(1, int(min_points)),
            overwrite=bool(overwrite),
        )
    else:
        summaries = [
            _dry_run_component(refined_group, component, min_points=max(1, int(min_points)))
            for component in component_names
        ]

    return {
        "zarr_path": str(zarr_path),
        "refined_run": run_name,
        "apply": bool(apply),
        "overwrite": bool(overwrite),
        "components": [_summary_to_dict(summary) for summary in summaries],
        "written_count": int(sum(1 for summary in summaries if summary.status == "written")),
        "would_write_count": int(sum(1 for summary in summaries if summary.status == "would_write")),
        "existing_count": int(sum(1 for summary in summaries if summary.status == "existing")),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path, help="Palette analysis or training Zarr archive.")
    parser.add_argument("--refined-run", help="Target refined_subject_masks_runs/<run>. Defaults to latest.")
    parser.add_argument(
        "--components",
        nargs="+",
        action="append",
        help="Components to backfill. Defaults to subject_body and swim_bladder.",
    )
    parser.add_argument(
        "--component",
        action="append",
        dest="component_values",
        help="Single component selector. Repeat to add more components.",
    )
    parser.add_argument("--apply", action="store_true", help="Write missing contour caches. Default is dry-run.")
    parser.add_argument("--overwrite", action="store_true", help="Rewrite existing contour caches.")
    parser.add_argument("--chunk-size", type=int, default=256, help="Row chunk size for ptr/len arrays.")
    parser.add_argument("--min-points", type=int, default=2, help="Minimum contour points required per row.")
    parser.add_argument("--json", action="store_true", help="Emit compact JSON.")
    args = parser.parse_args(argv)

    summary = backfill_refined_subject_component_contours(
        args.zarr_path,
        refined_run=args.refined_run,
        components=_parse_components(args.components, args.component_values),
        apply=bool(args.apply),
        overwrite=bool(args.overwrite),
        chunk_size=int(args.chunk_size),
        min_points=int(args.min_points),
    )
    print(json.dumps(summary, indent=None if args.json else 2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
