"""
Diagnostic for detect-quality run health.

Validates each `detect_runs/<run>/quality_reports/<qrun>` against the canonical
stage array spec.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import zarr
from rich.console import Console
from rich.table import Table

from fisheye.shared.zarr.stage_arrays import (
    DETECT_QUALITY_SPEC,
    array_specs_by_name,
    describe_array,
    validate_run,
)

QUALITY_ARRAYS = array_specs_by_name(DETECT_QUALITY_SPEC)
REQUIRED_ARRAYS = [
    (name, describe_array(spec))
    for name, spec in QUALITY_ARRAYS.items()
    if spec.required
]
OPTIONAL_ARRAYS = [
    (name, describe_array(spec))
    for name, spec in QUALITY_ARRAYS.items()
    if not spec.required
]
REQUIRED_ATTRS = ("analysis_timestamp",)


def _check_detect_quality_runs(console: Console, detect_parent: zarr.Group | None) -> None:
    if detect_parent is None:
        console.print("[red]detect_runs group not found in Zarr archive.[/red]")
        return

    detect_runs = sorted(name for name in detect_parent.keys() if isinstance(detect_parent[name], zarr.Group))
    if not detect_runs:
        console.print("[yellow]No detect runs found.[/yellow]")
        return

    console.print("[bold]detect_quality_runs[/bold] (under detect_runs/*/quality_reports)")
    table = Table("Detect Run", "Quality Run", "Status", "Details", expand=True)
    found_any_quality = False

    for detect_run in detect_runs:
        detect_group = detect_parent[detect_run]
        quality_parent = detect_group.get("quality_reports")
        if quality_parent is None:
            table.add_row(
                detect_run,
                "-",
                "[yellow]missing group[/yellow]",
                "missing 'quality_reports' subgroup",
            )
            continue

        quality_runs = sorted(
            name for name in quality_parent.keys() if isinstance(quality_parent[name], zarr.Group)
        )
        if not quality_runs:
            table.add_row(
                detect_run,
                "-",
                "[yellow]empty[/yellow]",
                "quality_reports exists but contains no runs",
            )
            continue

        found_any_quality = True
        latest = quality_parent.attrs.get("latest")
        for quality_run in quality_runs:
            group = quality_parent[quality_run]
            details: List[str] = []
            status = "[green]healthy[/green]"

            result = validate_run(group, DETECT_QUALITY_SPEC)
            if result.errors:
                status = "[red]schema error[/red]"
                details.extend(result.errors)
            for warning in result.warnings:
                details.append(f"[yellow]{warning}[/yellow]")

            for arr_name, desc in REQUIRED_ARRAYS:
                if arr_name in group:
                    details.append(f"{arr_name}: {group[arr_name].shape}")
                elif status != "[red]schema error[/red]":
                    status = "[red]missing data[/red]"
                    details.append(f"missing '{arr_name}' ({desc})")
            for arr_name, _ in OPTIONAL_ARRAYS:
                if arr_name in group:
                    details.append(f"{arr_name}: {group[arr_name].shape}")

            for attr in REQUIRED_ATTRS:
                if attr not in group.attrs:
                    if status == "[green]healthy[/green]":
                        status = "[yellow]missing attr[/yellow]"
                    details.append(f"missing attr '{attr}'")

            if latest and quality_run == latest:
                details.append("(latest quality run)")

            table.add_row(detect_run, quality_run, status, "\n".join(details))

    if not found_any_quality:
        console.print("[yellow]No detect quality runs found.[/yellow]")
    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose detect quality run metadata.")
    parser.add_argument("zarr_path", type=str, help="Path to Palette Zarr archive.")
    args = parser.parse_args()

    console = Console()
    zarr_path = str(Path(args.zarr_path))
    console.print(f"[bold]Inspecting[/bold] {zarr_path}")

    root = zarr.open(zarr_path, mode="r")
    _check_detect_quality_runs(console, root.get("detect_runs"))


if __name__ == "__main__":
    main()
