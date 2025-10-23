#!/usr/bin/env python3
"""Diagnose provenance consistency across detect → crop → keypoint → ID stages."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import zarr
from rich.console import Console
from rich.table import Table


@dataclass
class ProvenanceRecord:
    detect_run: Optional[str]
    refined_run: Optional[str]
    crop_run: Optional[str]
    keypoint_run: Optional[str]
    id_run: Optional[str]
    detect_rows: Optional[int]
    refined_rows: Optional[int]
    crop_rois: Optional[int]
    keypoint_rows: Optional[int]
    id_rows: Optional[int]
    issues: list[str]


def _safe_len(arr: Optional[zarr.Array]) -> Optional[int]:
    if arr is None:
        return None
    try:
        return int(arr.shape[0])
    except Exception:
        return None


def _latest(group: Optional[zarr.Group]) -> Optional[str]:
    if group is None:
        return None
    latest = group.attrs.get("latest")
    return latest if isinstance(latest, str) and latest in group else None


def _first_matching_run(group: Optional[zarr.Group], name: Optional[str]) -> Optional[zarr.Group]:
    if group is None or not name:
        return None
    return group.get(name)


def _collect_provenance(root: zarr.Group) -> ProvenanceRecord:
    issues: list[str] = []

    detect_parent = root.get("detect_runs")
    detect_latest = _latest(detect_parent)
    detect_group = _first_matching_run(detect_parent, detect_latest)
    detect_rows = _safe_len(detect_group["bbox_norm_coords"]) if detect_group else None

    refined_parent = root.get("refined_detect_runs") or root.get("refined_runs")
    refined_latest = _latest(refined_parent)
    refined_group = _first_matching_run(refined_parent, refined_latest)
    refined_rows = None
    if refined_group is not None:
        interp = refined_group.get("interpolated")
        refined_rows = _safe_len(interp["bbox_norm_coords"]) if interp is not None else None
        source_detect = refined_group.attrs.get("source_detect_run")
        if detect_latest and source_detect and detect_latest != source_detect:
            issues.append(
                f"Refined detect run '{refined_latest}' references detect '{source_detect}', "
                f"but latest detect is '{detect_latest}'."
            )

    crop_parent = root.get("crop_runs")
    crop_latest = _latest(crop_parent)
    crop_group = _first_matching_run(crop_parent, crop_latest)
    crop_rois = _safe_len(crop_group["roi_images"]) if crop_group else None
    if crop_group is not None:
        crop_source = crop_group.attrs.get("detection_source_path")
        if refined_latest and crop_source and refined_group is not None:
            expected_path = refined_group.path + "/interpolated"
            if crop_source != expected_path:
                issues.append(
                    f"Crop run '{crop_latest}' sourced from '{crop_source}' but refined detection path is '{expected_path}'."
                )
        elif detect_latest and crop_source:
            expected = f"detect_runs/{detect_latest}"
            if crop_source != expected:
                issues.append(
                    f"Crop run '{crop_latest}' sourced from '{crop_source}' but latest detect run is '{expected}'."
                )

    keypoint_parent = root.get("refined_keypoints_runs") or root.get("keypoints_runs")
    keypoint_latest = _latest(keypoint_parent)
    keypoint_group = _first_matching_run(keypoint_parent, keypoint_latest)
    keypoint_rows = _safe_len(keypoint_group["heading"]) if keypoint_group else None
    if keypoint_group is not None:
        kp_source_crop = keypoint_group.attrs.get("source_crop_run")
        if crop_latest and kp_source_crop and crop_latest != kp_source_crop:
            issues.append(
                f"Keypoint run '{keypoint_latest}' references crop '{kp_source_crop}', but latest crop is '{crop_latest}'."
            )

    id_parent = root.get("id_assignment_runs")
    id_latest = _latest(id_parent)
    id_group = _first_matching_run(id_parent, id_latest)
    id_rows = _safe_len(id_group["detection_ids"]) if id_group else None
    if id_group is not None:
        id_source_detect = id_group.attrs.get("source_detect_run")
        if detect_latest and id_source_detect and detect_latest != id_source_detect:
            issues.append(
                f"ID run '{id_latest}' references detect '{id_source_detect}' but latest detect is '{detect_latest}'."
            )
        id_source_refined = id_group.attrs.get("source_refined_run")
        if refined_latest and id_source_refined and refined_latest != id_source_refined:
            issues.append(
                f"ID run '{id_latest}' references refined detect '{id_source_refined}' but latest refined detect is '{refined_latest}'."
            )

    if refined_rows is not None and keypoint_rows is not None and refined_rows != keypoint_rows:
        issues.append(
            f"Refined detection count ({refined_rows}) != keypoint heading count ({keypoint_rows})."
        )
    if refined_rows is not None and id_rows is not None and refined_rows != id_rows:
        issues.append(
            f"Refined detection count ({refined_rows}) != ID assignments ({id_rows})."
        )
    if crop_rois is not None and refined_rows is not None and crop_rois != refined_rows:
        issues.append(
            f"Crop ROI count ({crop_rois}) != refined detection count ({refined_rows})."
        )

    return ProvenanceRecord(
        detect_run=detect_latest,
        refined_run=refined_latest,
        crop_run=crop_latest,
        keypoint_run=keypoint_latest,
        id_run=id_latest,
        detect_rows=detect_rows,
        refined_rows=refined_rows,
        crop_rois=crop_rois,
        keypoint_rows=keypoint_rows,
        id_rows=id_rows,
        issues=issues,
    )


def show_provenance(zarr_path: Path) -> None:
    console = Console()
    root = zarr.open(str(zarr_path), mode="r")

    record = _collect_provenance(root)

    table = Table(title="Provenance Summary", show_lines=False, box=None)
    table.add_column("Stage", style="cyan")
    table.add_column("Run")
    table.add_column("Rows", justify="right")

    table.add_row("Detect", record.detect_run or "—", str(record.detect_rows or "—"))
    table.add_row("Refined Detect", record.refined_run or "—", str(record.refined_rows or "—"))
    table.add_row("Crop", record.crop_run or "—", str(record.crop_rois or "—"))
    table.add_row("Keypoints", record.keypoint_run or "—", str(record.keypoint_rows or "—"))
    table.add_row("ID Assignment", record.id_run or "—", str(record.id_rows or "—"))

    console.print(table)

    if record.issues:
        console.print("[bold red]Inconsistencies detected:[/bold red]")
        for issue in record.issues:
            console.print(f"  • {issue}")
    else:
        console.print("[green]No provenance issues detected.[/green]")


def parse_args(argv: Optional[sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check detect/crop/keypoint/ID provenance consistency."
    )
    parser.add_argument("zarr_path", type=Path, help="Path to Palette Zarr archive")
    return parser.parse_args(argv)


def main(argv: Optional[sequence[str]] = None) -> None:
    args = parse_args(argv)
    show_provenance(args.zarr_path)


if __name__ == "__main__":  # pragma: no cover
    main()
