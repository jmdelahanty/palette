"""
Quick diagnostic for eye mask runs.

Checks that each `eye_masks_runs` and `refined_eye_masks_runs` entry contains
the bookkeeping arrays downstream tooling relies on (e.g. `frame_indices`,
`frame_counts`, `detection_indices`) and that shapes match the mask tensors.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import zarr
import numpy as np
from rich.console import Console
from rich.table import Table

from ..shared.zarr.stage_arrays import (
    EYE_MASKS_SPEC,
    REFINED_EYE_MASKS_SPEC,
    array_specs_by_name,
    describe_array,
    validate_run,
)
from ..shared.provenance_attrs import (
    CANONICAL_SOURCE_KEYPOINTS_RUN_ATTR,
    LEGACY_SOURCE_KEYPOINT_RUN_ATTR,
    resolve_source_keypoints_run,
)

EYE_MASK_ARRAYS = array_specs_by_name(EYE_MASKS_SPEC)
REFINED_EYE_MASK_ARRAYS = array_specs_by_name(REFINED_EYE_MASKS_SPEC)
REFINED_EYE_MASK_METRICS = array_specs_by_name(REFINED_EYE_MASKS_SPEC, subgroup="metrics")

EYE_REQUIRED_ARRAYS = [
    (name, describe_array(spec))
    for name, spec in EYE_MASK_ARRAYS.items()
    if spec.required
]
EYE_OPTIONAL_ARRAYS = [
    (name, describe_array(spec))
    for name, spec in EYE_MASK_ARRAYS.items()
    if not spec.required
]
REFINED_REQUIRED_ARRAYS = [
    (name, describe_array(spec))
    for name, spec in REFINED_EYE_MASK_ARRAYS.items()
    if spec.required
]
REFINED_OPTIONAL_ARRAYS = [
    (name, describe_array(spec))
    for name, spec in REFINED_EYE_MASK_ARRAYS.items()
    if not spec.required
]
REFINED_METRICS_REQUIRED_ARRAYS = [
    (name, describe_array(spec))
    for name, spec in REFINED_EYE_MASK_METRICS.items()
    if spec.required
]

STAGE_SPEC = {
    "eye_masks_runs": EYE_MASKS_SPEC,
    "refined_eye_masks_runs": REFINED_EYE_MASKS_SPEC,
}

STAGE_REQUIRED_ARRAYS = {
    "eye_masks_runs": EYE_REQUIRED_ARRAYS,
    "refined_eye_masks_runs": REFINED_REQUIRED_ARRAYS,
}

STAGE_OPTIONAL_ARRAYS = {
    "eye_masks_runs": EYE_OPTIONAL_ARRAYS,
    "refined_eye_masks_runs": REFINED_OPTIONAL_ARRAYS,
}

STAGE_REQUIRED_ATTRS = {
    "eye_masks_runs": ("source_crop_run",),
    "refined_eye_masks_runs": ("source_eye_masks_run", "source_crop_run"),
}


def _check_keypoint_lineage_attrs(
    attrs: Dict[str, object],
    *,
    current_status: str,
    details: List[str],
) -> str:
    """Validate keypoint lineage attrs using canonical+legacy compatibility rules."""
    has_canonical_kp = CANONICAL_SOURCE_KEYPOINTS_RUN_ATTR in attrs
    has_legacy_kp = LEGACY_SOURCE_KEYPOINT_RUN_ATTR in attrs
    source_kp_run = resolve_source_keypoints_run(attrs)

    status = current_status
    if not has_canonical_kp and not has_legacy_kp:
        status = "[yellow]incomplete[/yellow]"
        details.append("missing attr 'source_keypoints_run' (legacy alias: 'source_keypoint_run')")
    elif source_kp_run is None:
        status = "[yellow]incomplete[/yellow]"
        details.append("source keypoint lineage attr present but empty")
    elif not has_canonical_kp and has_legacy_kp and status == "[green]healthy[/green]":
        status = "[yellow]legacy[/yellow]"
        details.append(
            "legacy attr 'source_keypoint_run' present; backfill canonical 'source_keypoints_run'"
        )
    return status


def _check_stage(
    console: Console,
    root: zarr.Group,
    parent: zarr.Group | None,
    parent_name: str,
) -> None:
    if parent is None:
        console.print(f"[yellow]Stage '{parent_name}' not found.[/yellow]")
        return

    run_names: List[str] = sorted(
        [name for name in parent.keys() if isinstance(parent[name], zarr.Group)]
    )
    if not run_names:
        console.print(f"[yellow]Stage '{parent_name}' has no runs.[/yellow]")
        return

    latest = parent.attrs.get("latest")
    console.print(f"\n[bold]{parent_name}[/bold] (latest: {latest or 'unknown'})")
    table = Table("Run", "Status", "Details", show_lines=False, expand=True)

    stage_spec = STAGE_SPEC[parent_name]
    required_arrays = STAGE_REQUIRED_ARRAYS[parent_name]
    optional_arrays = STAGE_OPTIONAL_ARRAYS[parent_name]

    for name in run_names:
        group = parent[name]
        details: List[str] = []
        status = "[green]healthy[/green]"

        result = validate_run(group, stage_spec)
        if result.errors:
            status = "[red]schema error[/red]"
            details.extend(result.errors)
        for warning in result.warnings:
            details.append(f"[yellow]{warning}[/yellow]")

        for arr_name, desc in required_arrays:
            if arr_name in group:
                details.append(f"{arr_name}: {group[arr_name].shape}")
            elif status != "[red]schema error[/red]":
                status = "[red]missing data[/red]"
                details.append(f"missing '{arr_name}' ({desc})")
        for arr_name, _ in optional_arrays:
            if arr_name in group:
                details.append(f"{arr_name}: {group[arr_name].shape}")

        if parent_name == "refined_eye_masks_runs":
            metrics = group.get("metrics")
            if isinstance(metrics, zarr.Group):
                details.append("metrics/: present")
                for arr_name, desc in REFINED_METRICS_REQUIRED_ARRAYS:
                    if arr_name in metrics:
                        details.append(f"metrics/{arr_name}: {metrics[arr_name].shape}")
                    elif status != "[red]schema error[/red]":
                        status = "[red]missing data[/red]"
                        details.append(f"missing 'metrics/{arr_name}' ({desc})")
            elif status != "[red]schema error[/red]":
                status = "[red]missing data[/red]"
                details.append("missing 'metrics/' subgroup")

        # Check provenance attrs (canonical with legacy compatibility)
        for attr in STAGE_REQUIRED_ATTRS[parent_name]:
            if attr not in group.attrs:
                if status == "[green]healthy[/green]":
                    status = "[yellow]incomplete[/yellow]"
                details.append(f"missing attr '{attr}'")

        status = _check_keypoint_lineage_attrs(dict(group.attrs), current_status=status, details=details)

        crop_run = group.attrs.get("source_crop_run")
        det_meta = ""
        if crop_run:
            crop_group_path = f"crop_runs/{crop_run}"
            crop_group = root.get(crop_group_path)
            if isinstance(crop_group, zarr.Group):
                if "detection_source" in crop_group:
                    det_src = crop_group["detection_source"][:]
                    n_real = int(np.sum(det_src == 0))
                    n_interp = int(np.sum(det_src == 1))
                    det_meta = f"detection_source: real={n_real}, interp={n_interp}"
                else:
                    det_meta = "detection_source: missing"

                if "frame_counts" in group and "frame_counts" in crop_group:
                    group_frames = int(group["frame_counts"].shape[0])
                    crop_frames = int(crop_group["frame_counts"].shape[0])
                    if group_frames != crop_frames:
                        status = "[red]length mismatch[/red]"
                        details.append(
                            f"frame_counts len {group_frames} ≠ source crop frame_counts {crop_frames}"
                        )
            else:
                det_meta = f"crop run '{crop_run}' missing"
        else:
            det_meta = "source_crop_run attr missing"
        if det_meta:
            details.append(det_meta)

        table.add_row(
            name,
            status,
            "\n".join(details) if details else "ok",
        )

    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose eye mask runs.")
    parser.add_argument("zarr_path", type=str, help="Path to Palette Zarr archive.")
    args = parser.parse_args()

    console = Console()
    zarr_path = str(Path(args.zarr_path))
    console.print(f"[bold]Inspecting[/bold] {zarr_path}")

    root = zarr.open(zarr_path, mode="r")
    _check_stage(console, root, root.get("eye_masks_runs"), "eye_masks_runs")
    _check_stage(console, root, root.get("refined_eye_masks_runs"), "refined_eye_masks_runs")


if __name__ == "__main__":
    main()
