"""
Diagnostic for detection runs.

Verifies that each `detect_runs/<run>` contains the arrays required by downstream
stages (frame indices/counts, bounding boxes, scores, etc.) and that provenance
attributes are present.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import zarr
from rich.console import Console
from rich.table import Table

REQUIRED_ARRAYS = [
    ("frame_indices", "(n_detections,) int32"),
    ("frame_counts", "(n_frames,) int32"),
    ("n_detections", "(n_frames,) int32"),
    ("bbox_norm_coords", "(n_detections, 4) float32"),
    ("scores", "(n_detections,) float32"),
]

OPTIONAL_ARRAYS = [
    ("class_ids", "(n_detections,) int32"),
    ("centers_px", "(n_detections, 2) float32"),
]

REQUIRED_ATTRS = ["method"]


def _check_detection_runs(console: Console, parent: zarr.Group | None) -> None:
    if parent is None:
        console.print("[red]detect_runs group not found in Zarr archive.[/red]")
        return

    run_names = sorted(
        name for name in parent.keys() if isinstance(parent[name], zarr.Group)
    )
    if not run_names:
        console.print("[yellow]No detection runs found.[/yellow]")
        return

    latest = parent.attrs.get("latest")
    console.print(f"[bold]detect_runs[/bold] (latest: {latest or 'unknown'})")

    table = Table("Run", "Status", "Details", expand=True)
    for name in run_names:
        group = parent[name]
        status = "[green]healthy[/green]"
        details: List[str] = []

        det_count = group["frame_indices"].shape[0] if "frame_indices" in group else None

        for arr_name, desc in REQUIRED_ARRAYS:
            if arr_name not in group:
                status = "[red]missing data[/red]"
                details.append(f"missing '{arr_name}' ({desc})")
                continue
            arr = group[arr_name]
            details.append(f"{arr_name}: {arr.shape}")
            if det_count is not None and arr_name in {"bbox_norm_coords", "scores", "frame_indices"}:
                if arr.shape[0] != det_count:
                    status = "[red]mismatch[/red]"
                    details.append(f"{arr_name} len {arr.shape[0]} ≠ frame_indices {det_count}")

        for arr_name, desc in OPTIONAL_ARRAYS:
            if arr_name not in group:
                details.append(f"(optional) missing '{arr_name}'")
            else:
                details.append(f"{arr_name}: {group[arr_name].shape}")

        for attr in REQUIRED_ATTRS:
            if attr not in group.attrs:
                status = "[yellow]missing attr[/yellow]"
                details.append(f"missing attr '{attr}'")

        table.add_row(name, status, "\n".join(details))

    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose detection run metadata.")
    parser.add_argument("zarr_path", type=str, help="Path to Palette Zarr archive.")
    args = parser.parse_args()

    console = Console()
    zarr_path = str(Path(args.zarr_path))
    console.print(f"[bold]Inspecting[/bold] {zarr_path}")

    root = zarr.open(zarr_path, mode="r")
    _check_detection_runs(console, root.get("detect_runs"))


if __name__ == "__main__":
    main()
