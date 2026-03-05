"""
Diagnostic for background run health.

Validates each `background_runs/<run>` against the canonical stage array spec.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import zarr
from rich.console import Console
from rich.table import Table

from fisheye.shared.zarr.stage_arrays import (
    BACKGROUND_SPEC,
    array_specs_by_name,
    describe_array,
    validate_run,
)

BACKGROUND_ARRAYS = array_specs_by_name(BACKGROUND_SPEC)
REQUIRED_ARRAYS = [
    (name, describe_array(spec))
    for name, spec in BACKGROUND_ARRAYS.items()
    if spec.required
]
OPTIONAL_ARRAYS = [
    (name, describe_array(spec))
    for name, spec in BACKGROUND_ARRAYS.items()
    if not spec.required
]
REQUIRED_ATTRS = ("parameters",)


def _check_background_runs(console: Console, parent: zarr.Group | None) -> None:
    if parent is None:
        console.print("[red]background_runs group not found in Zarr archive.[/red]")
        return

    run_names = sorted(name for name in parent.keys() if isinstance(parent[name], zarr.Group))
    if not run_names:
        console.print("[yellow]No background runs found.[/yellow]")
        return

    latest = parent.attrs.get("latest")
    console.print(f"[bold]background_runs[/bold] (latest: {latest or 'unknown'})")

    table = Table("Run", "Status", "Details", expand=True)
    for run_name in run_names:
        group = parent[run_name]
        details: List[str] = []
        status = "[green]healthy[/green]"

        result = validate_run(group, BACKGROUND_SPEC)
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

        has_full = "background_full" in group
        has_ds = "background_ds" in group
        if not has_full and not has_ds:
            status = "[red]missing data[/red]"
            details.append("missing both optional backgrounds ('background_full' and 'background_ds')")

        for arr_name, _ in OPTIONAL_ARRAYS:
            if arr_name in group:
                details.append(f"{arr_name}: {group[arr_name].shape}")

        for attr in REQUIRED_ATTRS:
            if attr not in group.attrs:
                if status == "[green]healthy[/green]":
                    status = "[yellow]missing attr[/yellow]"
                details.append(f"missing attr '{attr}'")

        table.add_row(run_name, status, "\n".join(details))

    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose background run metadata.")
    parser.add_argument("zarr_path", type=str, help="Path to Palette Zarr archive.")
    args = parser.parse_args()

    console = Console()
    zarr_path = str(Path(args.zarr_path))
    console.print(f"[bold]Inspecting[/bold] {zarr_path}")

    root = zarr.open(zarr_path, mode="r")
    _check_background_runs(console, root.get("background_runs"))


if __name__ == "__main__":
    main()
