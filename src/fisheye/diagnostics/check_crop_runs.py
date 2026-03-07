"""Diagnostic for materialized and geometry-only crop run health."""

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

_MATERIALIZED_REQUIRED_ARRAYS = [
    (name, describe_array(_CROP_ARRAYS[name]))
    for name in (
        "roi_images",
        "frame_indices",
        "frame_counts",
        "detection_indices",
        "roi_coordinates_full",
        "bbox_norm_coords",
    )
]

_GEOMETRY_REQUIRED_ARRAYS = [
    (name, describe_array(_CROP_ARRAYS[name]))
    for name in (
        "frame_indices",
        "frame_counts",
        "detection_indices",
        "roi_coordinates_full",
        "bbox_norm_coords",
    )
]

_OPTIONAL_ARRAYS = [
    (name, describe_array(_CROP_ARRAYS[name]))
    for name in (
        "roi_coordinates_ds",
        "detection_source",
    )
]

_REQUIRED_ATTRS = ["source_detect_run"]
_OPTIONAL_ATTRS = ["source_background_run", "source_refined_run", "detect_review_status_ref", "detect_review_status"]
_NON_COMPLETED_STATUSES = {"failed", "running", "pending", "started", "created", "interrupted", "cancelled"}
_DEFAULT_ERROR_PREVIEW_CHARS = 160


def _normalize_status_value(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


def _summarize_present_arrays(group: zarr.Group) -> list[str]:
    details: list[str] = []
    for arr_name in (
        "roi_images",
        "frame_indices",
        "frame_counts",
        "detection_indices",
        "roi_coordinates_full",
        "bbox_norm_coords",
        "roi_coordinates_ds",
        "detection_source",
    ):
        if arr_name in group:
            details.append(f"{arr_name}: {group[arr_name].shape}")
    return details


def _summarize_error_text(value: object, *, verbose: bool) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if verbose:
        return text
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    first_line = lines[0]
    if len(lines) > 1:
        if len(first_line) >= _DEFAULT_ERROR_PREVIEW_CHARS:
            return first_line[: _DEFAULT_ERROR_PREVIEW_CHARS - 1].rstrip() + "…"
        return first_line + " …"
    if len(first_line) <= _DEFAULT_ERROR_PREVIEW_CHARS:
        return first_line
    if len(first_line) > _DEFAULT_ERROR_PREVIEW_CHARS:
        return first_line[: _DEFAULT_ERROR_PREVIEW_CHARS - 1].rstrip() + "…"
    return first_line


def _check_completed_run(
    group: zarr.Group,
    *,
    storage_mode: str,
    verbose: bool,
) -> tuple[str, list[str]]:
    severity = 0
    details: List[str] = [f"crop_storage_mode: {storage_mode}"]

    roi_count = group["roi_images"].shape[0] if "roi_images" in group else None
    source_path = group.attrs.get("detection_source_path")
    if source_path:
        details.append(f"source_path: {source_path}")

    required_arrays = (
        _MATERIALIZED_REQUIRED_ARRAYS if storage_mode == "materialized" else _GEOMETRY_REQUIRED_ARRAYS
    )
    for arr_name, desc in required_arrays:
        if arr_name not in group:
            severity = max(severity, 2)
            details.append(f"missing '{arr_name}' ({desc})")
            continue
        arr = group[arr_name]
        details.append(f"{arr_name}: {arr.shape}")
        if roi_count is not None and arr_name != "frame_counts" and arr.shape[0] != roi_count:
            severity = max(severity, 2)
            details.append(f"{arr_name} length {arr.shape[0]} ≠ roi_images {roi_count}")

    missing_optional_arrays: list[str] = []
    for arr_name, desc in _OPTIONAL_ARRAYS:
        if arr_name in group:
            details.append(f"{arr_name}: {group[arr_name].shape}")
        else:
            missing_optional_arrays.append(arr_name)
            if verbose:
                details.append(f"[dim]optional array missing '{arr_name}' ({desc})[/dim]")

    if "detection_source" in group:
        det_src = group["detection_source"][:]
        n_real = int(np.sum(det_src == 0))
        n_interp = int(np.sum(det_src == 1))
        details.append(f"detection_source: real={n_real}, interp={n_interp}")
        if n_interp == 0:
            details.append("[dim]no interpolated ROIs[/dim]")

    for attr in _REQUIRED_ATTRS:
        if attr not in group.attrs:
            severity = max(severity, 1)
            details.append(f"missing attr '{attr}'")
    missing_optional_attrs: list[str] = []
    for attr in _OPTIONAL_ATTRS:
        if attr not in group.attrs:
            missing_optional_attrs.append(attr)
            if verbose:
                details.append(f"[dim]optional attr missing '{attr}'[/dim]")

    if not verbose:
        if missing_optional_arrays:
            details.append(
                "[dim]optional arrays missing: " + ", ".join(missing_optional_arrays) + "[/dim]"
            )
        if missing_optional_attrs:
            details.append(
                "[dim]optional attrs missing: " + ", ".join(missing_optional_attrs) + "[/dim]"
            )

    if severity >= 2:
        return "[red]missing data[/red]", details
    if severity == 1:
        return "[yellow]missing provenance[/yellow]", details
    return "[green]healthy[/green]", details


def _check_non_completed_run(
    group: zarr.Group,
    *,
    storage_mode: str,
    run_status: str,
    verbose: bool,
) -> tuple[str, list[str]]:
    details: List[str] = [
        f"crop_storage_mode: {storage_mode}",
        f"pipeline_status: {run_status}",
    ]
    source_path = group.attrs.get("detection_source_path")
    if source_path:
        details.append(f"source_path: {source_path}")
    error_text = _summarize_error_text(group.attrs.get("error_message"), verbose=verbose)
    if error_text:
        details.append(f"error: {error_text}")
    failed_at = group.attrs.get("failed_at_utc") or group.attrs.get("ended_at_utc")
    if failed_at:
        details.append(f"ended_at_utc: {failed_at}")
    details.extend(_summarize_present_arrays(group))

    if run_status == "failed":
        return "[red]failed[/red]", details
    return "[yellow]incomplete[/yellow]", details


def _infer_crop_storage_mode(group: zarr.Group) -> str:
    explicit = group.attrs.get("crop_storage_mode")
    if explicit is not None:
        text = str(explicit).strip().lower()
        if text in {"materialized", "geometry_only"}:
            return text
    if "roi_images" in group:
        return "materialized"
    return "geometry_only"


def _check_crop_runs(console: Console, parent: zarr.Group | None, *, verbose: bool = False) -> None:
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
    latest_materialized = parent.attrs.get("latest_materialized")
    latest_any = parent.attrs.get("latest_any")
    console.print(
        "[bold]crop_runs[/bold] "
        f"(latest: {latest or 'unknown'}, "
        f"latest_materialized: {latest_materialized or 'unknown'}, "
        f"latest_any: {latest_any or 'unknown'})"
    )

    table = Table("Run", "Status", "Details", expand=True)
    for name in run_names:
        group = parent[name]
        storage_mode = _infer_crop_storage_mode(group)
        run_status = _normalize_status_value(group.attrs.get("status"))
        if run_status in _NON_COMPLETED_STATUSES:
            status, details = _check_non_completed_run(
                group,
                storage_mode=storage_mode,
                run_status=run_status,
                verbose=verbose,
            )
        else:
            status, details = _check_completed_run(group, storage_mode=storage_mode, verbose=verbose)

        table.add_row(name, status, "\n".join(details))

    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose crop run metadata.")
    parser.add_argument("zarr_path", type=str, help="Path to Palette Zarr archive.")
    parser.add_argument("--verbose", action="store_true", help="Show expanded optional-array and optional-attr details.")
    args = parser.parse_args()

    console = Console()
    zarr_path = str(Path(args.zarr_path))
    console.print(f"[bold]Inspecting[/bold] {zarr_path}")

    root = zarr.open(zarr_path, mode="r")
    _check_crop_runs(console, root.get("crop_runs"), verbose=bool(args.verbose))


if __name__ == "__main__":
    main()
