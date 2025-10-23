#!/usr/bin/env python3
"""Check that regenerated interpolated crops are linked correctly."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import zarr
from rich.console import Console
from rich.table import Table


@dataclass
class RefinedLinkStatus:
    refined_run: str
    roi_path: Optional[str]
    roi_count_attr: Optional[int]
    roi_count_actual: Optional[int]
    roi_frames_attr: Optional[int]
    roi_frames_actual: Optional[int]
    crop_run: Optional[str]
    crop_path: Optional[str]
    crop_count_attr: Optional[int]
    crop_frames_attr: Optional[int]
    decoder: Optional[str]
    issues: List[str]


def _safe_int(value: Optional[object]) -> Optional[int]:
    if isinstance(value, (int, np.integer)):
        return int(value)
    return None


def _analyse_refined_run(root: zarr.Group, name: str, group: zarr.Group) -> RefinedLinkStatus:
    issues: List[str] = []

    roi_path = group.attrs.get("interpolated_roi_path")
    roi_count_attr = _safe_int(group.attrs.get("interpolated_roi_count"))
    roi_frames_attr = _safe_int(group.attrs.get("interpolated_roi_frames"))
    decoder = group.attrs.get("interpolated_roi_decoder")

    roi_count_actual: Optional[int] = None
    roi_frames_actual: Optional[int] = None

    if roi_path:
        if roi_path in root:
            roi_group = root[roi_path]
            if "roi_images" in roi_group:
                roi_count_actual = int(roi_group["roi_images"].shape[0])
            else:
                issues.append(
                    f"Refined ROI group '{roi_path}' missing 'roi_images'."
                )
            if "frame_indices" in roi_group:
                roi_frames_actual = int(roi_group["frame_indices"].shape[0])
            if roi_count_attr is not None and roi_count_actual is not None and roi_count_attr != roi_count_actual:
                issues.append(
                    f"Refined run '{name}' attr count {roi_count_attr} differs from actual {roi_count_actual}."
                )
            if roi_frames_attr is not None and roi_frames_actual is not None and roi_frames_attr != roi_frames_actual:
                issues.append(
                    f"Refined run '{name}' attr frames {roi_frames_attr} differs from actual {roi_frames_actual}."
                )
        else:
            issues.append(
                f"Refined ROI path '{roi_path}' for '{name}' does not exist in archive."
            )
    else:
        issues.append(f"Refined run '{name}' missing 'interpolated_roi_path' attribute.")

    crop_run = group.attrs.get("source_crop_run")
    crop_path = None
    crop_count_attr = None
    crop_frames_attr = None
    if crop_run:
        crop_parent = root.get("crop_runs")
        if isinstance(crop_parent, zarr.Group) and crop_run in crop_parent:
            crop_group = crop_parent[crop_run]
            crop_path = crop_group.attrs.get("refined_roi_path")
            crop_count_attr = _safe_int(crop_group.attrs.get("refined_roi_count"))
            crop_frames_attr = _safe_int(crop_group.attrs.get("refined_roi_frames"))
            crop_decoder = crop_group.attrs.get("refined_roi_decoder")
            crop_duration = crop_group.attrs.get("refined_roi_generation_duration_seconds")

            if roi_path and crop_path and roi_path != crop_path:
                issues.append(
                    f"Crop run '{crop_run}' links to '{crop_path}' but refined run '{name}' links to '{roi_path}'."
                )
            if crop_count_attr is not None and roi_count_actual is not None and crop_count_attr != roi_count_actual:
                issues.append(
                    f"Crop run '{crop_run}' count {crop_count_attr} differs from actual refined ROI count {roi_count_actual}."
                )
            if crop_frames_attr is not None and roi_frames_actual is not None and crop_frames_attr != roi_frames_actual:
                issues.append(
                    f"Crop run '{crop_run}' frame count {crop_frames_attr} differs from refined ROI frames {roi_frames_actual}."
                )
            if decoder is None and crop_decoder is not None:
                decoder = crop_decoder
            if decoder is None and crop_duration is not None:
                decoder = crop_group.attrs.get("refined_roi_decoder")
        else:
            issues.append(
                f"Refined run '{name}' references crop run '{crop_run}' which does not exist."
            )
    else:
        issues.append(f"Refined run '{name}' missing 'source_crop_run' attribute.")

    return RefinedLinkStatus(
        refined_run=name,
        roi_path=roi_path,
        roi_count_attr=roi_count_attr,
        roi_count_actual=roi_count_actual,
        roi_frames_attr=roi_frames_attr,
        roi_frames_actual=roi_frames_actual,
        crop_run=crop_run,
        crop_path=crop_path,
        crop_count_attr=crop_count_attr,
        crop_frames_attr=crop_frames_attr,
        decoder=decoder,
        issues=issues,
    )


def show_refined_roi_links(zarr_path: Path) -> None:
    console = Console()
    root = zarr.open(str(zarr_path), mode="r")

    refined_parent = root.get("refined_detect_runs") or root.get("refined_runs")
    if not isinstance(refined_parent, zarr.Group):
        raise SystemExit("Archive contains no refined detection runs.")

    statuses = []
    for name in sorted(refined_parent.group_keys()):
        status = _analyse_refined_run(root, name, refined_parent[name])
        statuses.append(status)

    table = Table(title="Refined ROI Link Status", show_lines=False, box=None)
    table.add_column("Refined Run", style="cyan")
    table.add_column("ROI Path")
    table.add_column("ROI Count", justify="right")
    table.add_column("Actual", justify="right")
    table.add_column("Crop Run")
    table.add_column("Crop Path")
    table.add_column("Decoder")

    for status in statuses:
        table.add_row(
            status.refined_run,
            status.roi_path or "—",
            str(status.roi_count_attr or "—"),
            str(status.roi_count_actual or "—"),
            status.crop_run or "—",
            status.crop_path or "—",
            status.decoder or "—",
        )

    console.print(table)

    total_issues = sum(len(s.issues) for s in statuses)
    if total_issues:
        console.print(f"[bold red]{total_issues} issue(s) detected:[/bold red]")
        for status in statuses:
            for issue in status.issues:
                console.print(f"  • [{status.refined_run}] {issue}")
    else:
        console.print("[green]All refined ROI links appear consistent.[/green]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check refined ROI linkage across crop runs.")
    parser.add_argument("zarr_path", type=Path, help="Path to Palette Zarr archive")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    show_refined_roi_links(args.zarr_path)


if __name__ == "__main__":  # pragma: no cover
    main()
