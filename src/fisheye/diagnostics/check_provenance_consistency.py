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
    crop_source_path: Optional[str]
    crop_source_rows: Optional[int]
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


def _safe_get(group: Optional[zarr.Group], key: str) -> Optional[object]:
    if group is None:
        return None
    try:
        return group.get(key)
    except Exception:
        return None


def _group_for_path(root: zarr.Group, path: Optional[str]) -> Optional[zarr.Group]:
    if not path:
        return None
    normalized = path.strip("/")
    try:
        if normalized in root:
            return root[normalized]
    except Exception:
        return None
    return None


def _count_detections(group: Optional[zarr.Group]) -> Optional[int]:
    if group is None:
        return None
    if hasattr(group, "shape") and not hasattr(group, "get"):
        return _safe_len(group)  # type: ignore[arg-type]
    if isinstance(group, zarr.Array):
        return _safe_len(group)
    arr = _safe_get(group, "bbox_norm_coords") or _safe_get(group, "bbox_coords") or _safe_get(group, "bbox")
    return _safe_len(arr)


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
    detect_rows = _count_detections(detect_group)

    refined_parent_name = "refined_detect_runs"
    refined_parent = root.get(refined_parent_name)
    if refined_parent is None:
        refined_parent_name = "refined_runs"
        refined_parent = root.get(refined_parent_name)
    refined_latest = _latest(refined_parent)
    refined_group = _first_matching_run(refined_parent, refined_latest)
    refined_rows = None
    if refined_group is not None:
        interp = _safe_get(refined_group, "interpolated")
        if interp is not None and hasattr(interp, "get"):
            refined_rows = _count_detections(interp)
            if refined_rows is None:
                issues.append(
                    f"Refined detect run '{refined_latest}' is missing detection arrays under "
                    f"{refined_parent_name}/{refined_latest}/interpolated "
                    "(expected one of: bbox_norm_coords, bbox_coords, bbox)."
                )
        source_detect = refined_group.attrs.get("source_detect_run")
        if detect_latest and source_detect and detect_latest != source_detect:
            issues.append(
                f"Refined detect run '{refined_latest}' references detect '{source_detect}', "
                f"but latest detect is '{detect_latest}'."
            )

    crop_parent = root.get("crop_runs")
    crop_latest = _latest(crop_parent)
    crop_group = _first_matching_run(crop_parent, crop_latest)
    crop_rois = _safe_len(_safe_get(crop_group, "roi_images")) if crop_group else None
    crop_source_rows = None
    crop_source_path = None
    if crop_group is not None:
        crop_source = crop_group.attrs.get("detection_source_path")
        crop_source_path = crop_source

        def _expected_crop_path() -> Optional[str]:
            if refined_group is None:
                if detect_latest:
                    return f"detect_runs/{detect_latest}"
                return None

            manual_label = refined_group.attrs.get("manual_review_latest")
            if not manual_label and "manual" in refined_group:
                manual_label = "manual"

            review_status = refined_group.attrs.get("detect_review_status")
            resolved_group = None
            if isinstance(review_status, dict):
                resolved_group = review_status.get("target_group") or review_status.get("resolved_group")
            if resolved_group:
                resolved_group = str(resolved_group)
                if resolved_group in {"raw", "detect"}:
                    if detect_latest:
                        return f"detect_runs/{detect_latest}"
                    source_detect = refined_group.attrs.get("source_detect_run")
                    if source_detect:
                        return f"detect_runs/{source_detect}"
                    return None
                if resolved_group == "manual" and manual_label:
                    return refined_group.path + f"/{manual_label}"
                if resolved_group in refined_group:
                    return refined_group.path + f"/{resolved_group}"

            return refined_group.path + (f"/{manual_label}" if manual_label else "/interpolated")

        expected_path = _expected_crop_path()
        source_path = crop_source or expected_path
        if source_path:
            crop_source_path = source_path
        crop_source_rows = _count_detections(_group_for_path(root, source_path))
        if crop_source and expected_path and crop_source != expected_path:
            issues.append(
                f"Crop run '{crop_latest}' sourced from '{crop_source}' but expected '{expected_path}'."
            )

    keypoint_parent = root.get("refined_keypoints_runs") or root.get("keypoints_runs")
    keypoint_latest = _latest(keypoint_parent)
    keypoint_group = _first_matching_run(keypoint_parent, keypoint_latest)
    keypoint_rows = _safe_len(_safe_get(keypoint_group, "heading")) if keypoint_group else None
    if keypoint_group is not None:
        kp_source_crop = keypoint_group.attrs.get("source_crop_run")
        if crop_latest and kp_source_crop and crop_latest != kp_source_crop:
            issues.append(
                f"Keypoint run '{keypoint_latest}' references crop '{kp_source_crop}', but latest crop is '{crop_latest}'."
            )

    id_parent = root.get("id_assignment_runs")
    id_latest = _latest(id_parent)
    id_group = _first_matching_run(id_parent, id_latest)
    id_rows = _safe_len(_safe_get(id_group, "detection_ids")) if id_group else None
    id_source_rows = None
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
        if id_source_refined:
            id_source_rows = _count_detections(
                _group_for_path(root, f"refined_detect_runs/{id_source_refined}/interpolated")
                or _group_for_path(root, f"refined_runs/{id_source_refined}/interpolated")
            )
        if id_source_rows is None and id_source_detect:
            id_source_rows = _count_detections(_group_for_path(root, f"detect_runs/{id_source_detect}"))

    if crop_rois is not None and crop_source_rows is not None and crop_rois != crop_source_rows:
        issues.append(
            f"Crop ROI count ({crop_rois}) != detection source count ({crop_source_rows})"
            + (f" from '{crop_source_path}'." if crop_source_path else ".")
        )
    if crop_rois is not None and keypoint_rows is not None and crop_rois != keypoint_rows:
        issues.append(
            f"Crop ROI count ({crop_rois}) != keypoint heading count ({keypoint_rows})."
        )
    if id_rows is not None and id_source_rows is not None and id_rows != id_source_rows:
        issues.append(
            f"ID assignments ({id_rows}) != detection source count ({id_source_rows})."
        )

    return ProvenanceRecord(
        detect_run=detect_latest,
        refined_run=refined_latest,
        crop_run=crop_latest,
        crop_source_path=crop_source_path,
        crop_source_rows=crop_source_rows,
        keypoint_run=keypoint_latest,
        id_run=id_latest,
        detect_rows=detect_rows,
        refined_rows=refined_rows,
        crop_rois=crop_rois,
        keypoint_rows=keypoint_rows,
        id_rows=id_rows,
        issues=issues,
    )


def collect_provenance(root: zarr.Group) -> ProvenanceRecord:
    """Collect provenance consistency info without printing."""
    return _collect_provenance(root)


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
    source_label = record.crop_source_path or "—"
    det_count = record.crop_source_rows if record.crop_source_rows is not None else "—"
    crop_count = record.crop_rois if record.crop_rois is not None else "—"
    kp_count = record.keypoint_rows if record.keypoint_rows is not None else "—"
    console.print(f"[dim]Lineage counts:[/dim] source={source_label} det={det_count} crop={crop_count} kpt={kp_count}")

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
