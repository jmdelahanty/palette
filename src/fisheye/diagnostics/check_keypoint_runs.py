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

from fisheye.shared.zarr.stage_arrays import (
    KEYPOINTS_SPEC,
    REFINED_KEYPOINTS_SPEC,
    array_specs_by_name,
    describe_array,
    validate_run,
)

KEYPOINT_ARRAYS = array_specs_by_name(KEYPOINTS_SPEC)
REFINED_KEYPOINT_ARRAYS = array_specs_by_name(REFINED_KEYPOINTS_SPEC)

REQUIRED_ARRAYS = [
    (name, describe_array(spec))
    for name, spec in KEYPOINT_ARRAYS.items()
    if spec.required
]
OPTIONAL_ARRAYS = [
    (name, describe_array(spec))
    for name, spec in KEYPOINT_ARRAYS.items()
    if not spec.required
]
REFINED_REQUIRED_ARRAYS = [
    (name, describe_array(spec))
    for name, spec in REFINED_KEYPOINT_ARRAYS.items()
    if spec.required
]
REFINED_OPTIONAL_ARRAYS = [
    (name, describe_array(spec))
    for name, spec in REFINED_KEYPOINT_ARRAYS.items()
    if not spec.required
]

REQUIRED_ATTRS = {
    "keypoints_runs": ["source_crop_run", "source_detect_run"],
    "refined_keypoints_runs": ["source_crop_run", "source_detect_run", "source_keypoints_run"],
}

_STAGE_SPEC = {
    "keypoints_runs": KEYPOINTS_SPEC,
    "refined_keypoints_runs": REFINED_KEYPOINTS_SPEC,
}

_STAGE_REQUIRED_ARRAYS = {
    "keypoints_runs": REQUIRED_ARRAYS,
    "refined_keypoints_runs": REFINED_REQUIRED_ARRAYS,
}

_STAGE_OPTIONAL_ARRAYS = {
    "keypoints_runs": OPTIONAL_ARRAYS,
    "refined_keypoints_runs": REFINED_OPTIONAL_ARRAYS,
}


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

    stage_spec = _STAGE_SPEC[stage_name]
    required_arrays = _STAGE_REQUIRED_ARRAYS[stage_name]
    optional_arrays = _STAGE_OPTIONAL_ARRAYS[stage_name]

    for run_name in run_names:
        group = parent[run_name]
        status = "[green]healthy[/green]"
        details: List[str] = []

        result = validate_run(group, stage_spec)
        if result.errors:
            status = "[red]schema error[/red]"
            details.extend(result.errors)
        for warning in result.warnings:
            details.append(f"[yellow]{warning}[/yellow]")

        for array_name, desc in required_arrays:
            if array_name in group:
                details.append(f"{array_name}: {group[array_name].shape}")
            elif status != "[red]schema error[/red]":
                status = "[red]missing data[/red]"
                details.append(f"missing '{array_name}' ({desc})")
        for array_name, _ in optional_arrays:
            if array_name in group:
                details.append(f"{array_name}: {group[array_name].shape}")

        for attr in REQUIRED_ATTRS[stage_name]:
            if attr not in group.attrs:
                if status == "[green]healthy[/green]":
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
