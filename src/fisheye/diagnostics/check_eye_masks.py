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

from ..shared.provenance_attrs import (
    CANONICAL_SOURCE_KEYPOINTS_RUN_ATTR,
    LEGACY_SOURCE_KEYPOINT_RUN_ATTR,
    resolve_source_keypoints_run,
)

REQUIRED_MASK_ARRAYS = ("masks_roi",)
REQUIRED_MAPPING_ARRAYS = ("frame_indices", "frame_counts", "detection_indices")


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

    for name in run_names:
        group = parent[name]
        details: List[str] = []
        status = "[green]healthy[/green]"

        # Check required mask arrays
        for arr in REQUIRED_MASK_ARRAYS:
            if arr not in group:
                status = "[red]missing data[/red]"
                details.append(f"missing '{arr}'")
            else:
                shape = group[arr].shape
                details.append(f"{arr}: {shape[0]} ROIs")

        # Check mapping arrays
        roi_count = None
        if "masks_roi" in group:
            roi_count = group["masks_roi"].shape[0]

        for arr in REQUIRED_MAPPING_ARRAYS:
            if arr not in group:
                status = "[red]missing mapping[/red]"
                details.append(f"missing '{arr}'")
                continue
            arr_shape = group[arr].shape
            arr_len = arr_shape[0]
            details.append(f"{arr}: {arr_shape}")
            if arr == "frame_counts":
                expected_len = None
                source_crop = group.attrs.get("source_crop_run")
                if source_crop:
                    crop_path = f"crop_runs/{source_crop}"
                    crop_group = root.get(crop_path)
                    if isinstance(crop_group, zarr.Group) and "frame_counts" in crop_group:
                        expected_len = crop_group["frame_counts"].shape[0]
                if expected_len is not None and arr_len != expected_len:
                    status = "[red]length mismatch[/red]"
                    details.append(f"frame_counts len {arr_len} ≠ source crop frame_counts {expected_len}")
                continue
            if roi_count is not None and arr_len != roi_count:
                status = "[red]length mismatch[/red]"
                details.append(f"{arr} len {arr_len} ≠ masks {roi_count}")

        # Check provenance attrs (canonical with legacy compatibility)
        if "source_crop_run" not in group.attrs:
            status = "[yellow]incomplete[/yellow]"
            details.append("missing attr 'source_crop_run'")

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
