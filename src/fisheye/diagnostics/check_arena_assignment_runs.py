"""
Diagnostic for arena-assignment run health.

Validates each `arena_assignment_runs/<run>` against the canonical stage array
spec and reports provenance-attribute coverage.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import zarr
from rich.console import Console
from rich.table import Table

from fisheye.shared.zarr.stage_arrays import (
    ARENA_ASSIGNMENT_SPEC,
    array_specs_by_name,
    describe_array,
    validate_run,
)

ID_ARRAYS = array_specs_by_name(ARENA_ASSIGNMENT_SPEC)
REQUIRED_ARRAYS = [
    (name, describe_array(spec))
    for name, spec in ID_ARRAYS.items()
    if spec.required
]
OPTIONAL_ARRAYS = [
    (name, describe_array(spec))
    for name, spec in ID_ARRAYS.items()
    if not spec.required
]
REQUIRED_ATTRS = ("summary_statistics",)


def _check_arena_assignment_runs(console: Console, parent: zarr.Group | None) -> None:
    if parent is None:
        console.print("[red]arena_assignment_runs group not found in Zarr archive.[/red]")
        return

    run_names = sorted(name for name in parent.keys() if isinstance(parent[name], zarr.Group))
    if not run_names:
        console.print("[yellow]No arena-assignment runs found.[/yellow]")
        return

    latest = parent.attrs.get("latest")
    console.print(f"[bold]arena_assignment_runs[/bold] (latest: {latest or 'unknown'})")

    table = Table("Run", "Status", "Details", expand=True)
    for run_name in run_names:
        group = parent[run_name]
        details: List[str] = []
        status = "[green]healthy[/green]"

        result = validate_run(group, ARENA_ASSIGNMENT_SPEC)
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

        if "n_detections_per_arena" in group:
            details.append(f"n_detections_per_arena: {group['n_detections_per_arena'].shape}")

        for attr in REQUIRED_ATTRS:
            if attr not in group.attrs:
                if status == "[green]healthy[/green]":
                    status = "[yellow]missing attr[/yellow]"
                details.append(f"missing attr '{attr}'")

        table.add_row(run_name, status, "\n".join(details))

    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose arena-assignment run metadata.")
    parser.add_argument("zarr_path", type=str, help="Path to Palette Zarr archive.")
    args = parser.parse_args()

    console = Console()
    zarr_path = str(Path(args.zarr_path))
    console.print(f"[bold]Inspecting[/bold] {zarr_path}")

    root = zarr.open(zarr_path, mode="r")
    _check_arena_assignment_runs(console, root.get("arena_assignment_runs"))


if __name__ == "__main__":
    main()
