#!/usr/bin/env python3
"""
Diagnostic for keypoint run health.

Validates that each ``keypoints_runs/<run>`` (and ``refined_keypoints_runs`` if present)
contains the required mapping arrays, keypoint tensors, and source attrs that downstream
stages rely on. Highlights shape mismatches and missing provenance.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional

import zarr
from rich.console import Console
from rich.table import Table

KEYPOINT_ARRAYS = [
    ("keypoints_roi", "(n_rois, n_keypoints, 2) float32/float64"),
    ("keypoints_img", "(n_rois, n_keypoints, 2) float32/float64"),
    ("keypoints_norm", "(n_rois, n_keypoints, 2) float32/float64"),
    ("heading", "(n_rois,) float32/float64"),
    ("confidence", "(n_rois,) float32/float64"),
]

MAPPING_ARRAYS = [
    ("frame_indices", "(n_rois,) int32"),
    ("frame_counts", "(n_frames,) int32"),
    ("detection_indices", "(n_rois,) int32"),
]

REQUIRED_ATTRS = ["source_crop_run", "source_detect_run"]


def _check_stage(console: Console, parent: Optional[zarr.Group], stage_name: str) -> None:
    if parent is None:
        console.print(f"[yellow]Stage '{stage_name}' not found.[/yellow]")
        return

    run_names: List[str] = sorted(
        name for name in parent.group_keys() if isinstance(parent.get(name), zarr.Group)
    )
    if not run_names:
        console.print(f"[yellow]Stage '{stage_name}' has no runs.[/yellow]")
        return

    latest = parent.attrs.get("latest")
    console.print(f"\n[bold]{stage_name}[/bold] (latest: {latest or 'unknown'})")
    table = Table("Run", "Status", "Details", expand=True)

    for run_name in run_names:
        group = parent[run_name]
        status = "[green]healthy[/green]"
        details: List[str] = []

        n_rois = None
        if "keypoints_roi" in group:
            n_rois = group["keypoints_roi"].shape[0]
            details.append(f"keypoints_roi: {group['keypoints_roi'].shape}")
        else:
            status = "[red]missing data[/red]"
            details.append("missing 'keypoints_roi'")

        for array_name, desc in KEYPOINT_ARRAYS[1:]:
            if array_name not in group:
                status = "[red]missing data[/red]"
                details.append(f"missing '{array_name}' ({desc})")
                continue
            shape = group[array_name].shape
            details.append(f"{array_name}: {shape}")
            if n_rois is not None and shape[0] != n_rois:
                status = "[red]length mismatch[/red]"
                details.append(f"{array_name} len {shape[0]} ≠ keypoints_roi {n_rois}")

        for array_name, desc in MAPPING_ARRAYS:
            if array_name not in group:
                status = "[red]missing mapping[/red]"
                details.append(f"missing '{array_name}' ({desc})")
                continue
            shape = group[array_name].shape
            details.append(f"{array_name}: {shape}")
            if n_rois is not None and array_name != "frame_counts" and shape[0] != n_rois:
                status = "[red]length mismatch[/red]"
                details.append(f"{array_name} len {shape[0]} ≠ keypoints_roi {n_rois}")

        for attr in REQUIRED_ATTRS:
            if attr not in group.attrs:
                status = "[yellow]missing attr[/yellow]"
                details.append(f"missing attr '{attr}'")

        if "source_crop_run" in group.attrs:
            crop_run = group.attrs["source_crop_run"]
            details.append(f"source_crop_run: {crop_run}")

        table.add_row(run_name, status, "\n".join(details) if details else "ok")

    console.print(table)


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Diagnose keypoint run metadata.")
    parser.add_argument("zarr_path", type=Path, help="Path to Palette Zarr archive.")
    args = parser.parse_args()

    console = Console()
    console.print(f"[bold]Inspecting[/bold] {args.zarr_path}")

    root = zarr.open(str(args.zarr_path), mode="r")
    _check_stage(console, root.get("keypoints_runs"), "keypoints_runs")
    _check_stage(console, root.get("refined_keypoints_runs"), "refined_keypoints_runs")


if __name__ == "__main__":  # pragma: no cover
    main()
