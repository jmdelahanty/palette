"""
Diagnostic for crop run health.

Verifies that each `crop_runs/<run>` includes the arrays required to align ROIs
with detections (`roi_images`, `frame_indices`, `frame_counts`,
`detection_indices`, `roi_coordinates_full`, `roi_coordinates_ds`,
`bbox_norm_coords`) and reports missing items.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import zarr
import numpy as np
from rich.console import Console
from rich.table import Table

from fisheye.shared.zarr.stage_arrays import CROP_SPEC, array_specs_by_name, describe_array

_CROP_ARRAYS = array_specs_by_name(CROP_SPEC)

REQUIRED_ARRAYS = [
    (name, describe_array(_CROP_ARRAYS[name]))
    for name in (
        "roi_images",
        "frame_indices",
        "frame_counts",
        "detection_indices",
        "roi_coordinates_full",
        "roi_coordinates_ds",
        "bbox_norm_coords",
    )
]

REQUIRED_ATTRS = ["source_detect_run", "source_background_run"]


def _check_crop_runs(console: Console, parent: zarr.Group | None) -> None:
    if parent is None:
        console.print("[red]crop_runs group not found in Zarr archive.[/red]")
        return

    run_names = sorted(
        name for name in parent.keys() if isinstance(parent[name], zarr.Group)
    )
    if not run_names:
        console.print("[yellow]No crop runs found.[/yellow]")
        return

    latest = parent.attrs.get("latest")
    console.print(f"[bold]crop_runs[/bold] (latest: {latest or 'unknown'})")

    table = Table("Run", "Status", "Details", expand=True)
    for name in run_names:
        group = parent[name]
        status = "[green]healthy[/green]"
        details: List[str] = []

        roi_count = group["roi_images"].shape[0] if "roi_images" in group else None
        source_path = group.attrs.get("detection_source_path")
        if source_path:
            details.append(f"source_path: {source_path}")

        for arr_name, desc in REQUIRED_ARRAYS:
            if arr_name not in group:
                status = "[red]missing data[/red]"
                details.append(f"missing '{arr_name}' ({desc})")
                continue
            arr = group[arr_name]
            shapes = arr.shape
            details.append(f"{arr_name}: {shapes}")
            if roi_count is not None and arr_name != "frame_counts":
                if arr.shape[0] != roi_count:
                    status = "[red]mismatch[/red]"
                    details.append(
                        f"{arr_name} length {arr.shape[0]} ≠ roi_images {roi_count}"
                    )

        if "detection_source" in group:
            det_src = group["detection_source"][:]
            n_real = int(np.sum(det_src == 0))
            n_interp = int(np.sum(det_src == 1))
            details.append(f"detection_source: real={n_real}, interp={n_interp}")
            if n_interp == 0:
                details.append("[dim]no interpolated ROIs[/dim]")
        else:
            details.append("[yellow]detection_source array missing[/yellow]")

        for attr in REQUIRED_ATTRS:
            if attr not in group.attrs:
                status = "[yellow]missing attr[/yellow]"
                details.append(f"missing attr '{attr}'")

        table.add_row(name, status, "\n".join(details))

    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose crop run metadata.")
    parser.add_argument("zarr_path", type=str, help="Path to Palette Zarr archive.")
    args = parser.parse_args()

    console = Console()
    zarr_path = str(Path(args.zarr_path))
    console.print(f"[bold]Inspecting[/bold] {zarr_path}")

    root = zarr.open(zarr_path, mode="r")
    _check_crop_runs(console, root.get("crop_runs"))


if __name__ == "__main__":
    main()
