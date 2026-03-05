"""
Diagnostic for raw-video stage health.

Validates `raw_video/` against the canonical stage array spec.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import zarr
from rich.console import Console
from rich.table import Table

from fisheye.shared.zarr.stage_arrays import (
    RAW_VIDEO_SPEC,
    array_specs_by_name,
    describe_array,
    validate_run,
)

RAW_ARRAYS = array_specs_by_name(RAW_VIDEO_SPEC)
REQUIRED_ARRAYS = [
    (name, describe_array(spec))
    for name, spec in RAW_ARRAYS.items()
    if spec.required
]
OPTIONAL_ARRAYS = [
    (name, describe_array(spec))
    for name, spec in RAW_ARRAYS.items()
    if not spec.required
]


def _check_raw_video(console: Console, root: zarr.Group) -> None:
    raw = root.get("raw_video")
    if raw is None:
        console.print("[red]raw_video group not found in Zarr archive.[/red]")
        return
    if not isinstance(raw, zarr.Group):
        console.print("[red]raw_video exists but is not a group.[/red]")
        return

    details: List[str] = []
    status = "[green]healthy[/green]"

    result = validate_run(raw, RAW_VIDEO_SPEC)
    if result.errors:
        status = "[red]schema error[/red]"
        details.extend(result.errors)
    for warning in result.warnings:
        details.append(f"[yellow]{warning}[/yellow]")

    for arr_name, desc in REQUIRED_ARRAYS:
        if arr_name in raw:
            details.append(f"{arr_name}: {raw[arr_name].shape}")
        elif status != "[red]schema error[/red]":
            status = "[red]missing data[/red]"
            details.append(f"missing '{arr_name}' ({desc})")
    for arr_name, _ in OPTIONAL_ARRAYS:
        if arr_name in raw:
            details.append(f"{arr_name}: {raw[arr_name].shape}")

    has_full = "images_full" in raw
    has_ds = "images_ds" in raw
    if not has_full and not has_ds:
        status = "[red]missing data[/red]"
        details.append("missing both 'images_full' and 'images_ds' (at least one is required)")

    frame_step = raw.attrs.get("frame_step")
    if frame_step is not None:
        try:
            if int(frame_step) > 1 and "original_frame_indices" not in raw:
                if status == "[green]healthy[/green]":
                    status = "[yellow]missing data[/yellow]"
                details.append("frame_step > 1 but 'original_frame_indices' is missing")
        except Exception:
            details.append("frame_step attr present but not an integer")

    table = Table("Stage", "Status", "Details", expand=True)
    table.add_row("raw_video", status, "\n".join(details))
    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose raw-video stage metadata.")
    parser.add_argument("zarr_path", type=str, help="Path to Palette Zarr archive.")
    args = parser.parse_args()

    console = Console()
    zarr_path = str(Path(args.zarr_path))
    console.print(f"[bold]Inspecting[/bold] {zarr_path}")

    root = zarr.open(zarr_path, mode="r")
    _check_raw_video(console, root)


if __name__ == "__main__":
    main()
