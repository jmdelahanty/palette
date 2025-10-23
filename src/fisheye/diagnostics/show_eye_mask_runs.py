#!/usr/bin/env python3
"""
Inspect eye mask inference runs and highlight configuration that impacts runtime.

For each run under ``eye_masks_runs`` the script reports device, batch size, crop
source, ROI counts, ROI dimensions, and recorded duration so you can quickly spot
changes that might explain performance shifts.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import zarr
from rich.console import Console
from rich.table import Table

DURATION_KEYS: Tuple[str, ...] = (
    "inference_duration_seconds",
    "duration_seconds",
)


def _sorted_group_keys(group: Optional[zarr.Group]) -> List[str]:
    if group is None:
        return []
    keys_fn = getattr(group, "group_keys", None)
    try:
        keys = list(keys_fn()) if callable(keys_fn) else []
    except Exception:
        keys = []
    return sorted(key for key in keys if isinstance(key, str))


def _format_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "—"
    if seconds < 1.0:
        return f"{seconds * 1000:.0f} ms"
    hours, rem = divmod(int(seconds), 3600)
    minutes, secs = divmod(rem, 60)
    remainder = seconds - int(seconds)
    frac = f"{remainder:.2f}"[1:] if remainder >= 0.005 else ""
    if hours:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s{frac}"
    if minutes:
        return f"{minutes:d}m {secs:02d}s{frac}"
    return f"{seconds:.2f}s"


def _extract_duration(attrs: zarr.AttrMapping) -> Optional[float]:
    for key in DURATION_KEYS:
        raw = attrs.get(key)
        if raw is None or isinstance(raw, bool):
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isnan(value) or math.isinf(value):
            continue
        return value
    return None


def _format_delta(current: Optional[int], previous: Optional[int]) -> str:
    if current is None or previous is None:
        return "—"
    delta = current - previous
    if delta == 0:
        return "0"
    sign = "+" if delta > 0 else ""
    return f"{sign}{delta}"


def _get_roi_shape(root: zarr.Group, crop_run: Optional[str]) -> str:
    if not crop_run:
        return "—"
    roi_path = f"crop_runs/{crop_run}/roi_images"
    if roi_path not in root:
        return "missing"
    array = root[roi_path]
    if array.ndim < 3:
        return "unknown"
    height = array.shape[-2]
    width = array.shape[-1]
    return f"{height}×{width}"


def _select_run_names(run_names: Sequence[str], latest: Optional[str], limit: Optional[int]) -> List[str]:
    if limit is not None and limit <= 0:
        return []
    selected = list(run_names)
    if limit is not None:
        selected = selected[-limit:]
    # Ensure latest is included and positioned last if present.
    if latest and latest in run_names and latest not in selected:
        selected.append(latest)
    if latest and latest in selected:
        selected = [name for name in selected if name != latest] + [latest]
    return selected


def show_eye_mask_runs(zarr_path: Path, limit: Optional[int]) -> None:
    console = Console()
    console.print(f"[bold]Inspecting eye mask runs in:[/bold] {zarr_path}")

    root = zarr.open(str(zarr_path), mode="r")
    if "eye_masks_runs" not in root:
        raise ValueError(f"No eye_masks_runs group found in archive '{zarr_path}'.")

    run_parent = root["eye_masks_runs"]
    run_names = _sorted_group_keys(run_parent)
    if not run_names:
        console.print("[yellow]No eye mask runs recorded.[/yellow]")
        return

    latest = run_parent.attrs.get("latest")
    selected_runs = _select_run_names(run_names, latest, limit)

    table = Table(title="Eye Mask Inference Runs", show_lines=False, box=None)
    table.add_column("Run", style="cyan")
    table.add_column("Device", style="green", no_wrap=True)
    table.add_column("GPU", style="green", no_wrap=True)
    table.add_column("Batch", justify="right")
    table.add_column("Total ROIs", justify="right")
    table.add_column("Δ ROIs", justify="right")
    table.add_column("ROI Size", style="yellow", no_wrap=True)
    table.add_column("Crop Run", style="magenta")
    table.add_column("Duration", style="bold", no_wrap=True)
    table.add_column("Latest", style="bold", no_wrap=True)

    previous_total: Optional[int] = None
    for run_name in selected_runs:
        run = run_parent.get(run_name)
        if not isinstance(run, zarr.Group):
            continue
        attrs = run.attrs
        device = attrs.get("inference_device", "unknown")
        batch = attrs.get("inference_batch_size")
        total_rois = attrs.get("total_rois")
        try:
            total_rois_int = int(total_rois) if total_rois is not None else None
        except (TypeError, ValueError):
            total_rois_int = None
        delta_rois = _format_delta(total_rois_int, previous_total)
        previous_total = total_rois_int
        crop_run = attrs.get("source_crop_run")
        duration = _extract_duration(attrs)
        roi_size = _get_roi_shape(root, crop_run)
        gpu_device = attrs.get("gpu_device", "—")
        latest_flag = "★" if latest and run_name == latest else ""
        table.add_row(
            run_name,
            str(device),
            str(gpu_device or "—"),
            str(batch) if batch is not None else "—",
            str(total_rois_int) if total_rois_int is not None else "—",
            delta_rois,
            roi_size,
            str(crop_run or "—"),
            _format_duration(duration),
            latest_flag,
        )

    console.print(table)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show device, ROI counts, and runtime metadata for eye mask inference runs."
    )
    parser.add_argument("zarr_path", type=Path, help="Path to Palette Zarr archive.")
    parser.add_argument(
        "--limit",
        type=int,
        help="Only display the most recent N runs (latest run is always included).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    show_eye_mask_runs(args.zarr_path, args.limit)


if __name__ == "__main__":  # pragma: no cover
    main()
