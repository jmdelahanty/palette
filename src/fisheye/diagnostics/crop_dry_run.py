#!/usr/bin/env python3
"""
Predict what the crop stage would do without actually writing ROIs.

This diagnostic reports the detection run and frame counts that source
resolution finds, then applies the future ordinary-writer admission rule. Only
an exact ``detect_runs/<run>`` source is currently eligible for a new normal
crop run.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import yaml
import zarr
from rich.console import Console
from rich.table import Table

from fisheye.tracking.crop import (
    _preflight_ordinary_crop_coordinates,
    infer_detection_source_type,
    get_detection_source_info,
    get_video_source,
)


def _load_config(path: Optional[Path]) -> Dict:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r") as f:
        return yaml.safe_load(f) or {}


def _normalize_source_path(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    normalized = str(path).strip().strip("/")
    return normalized or None


def simulate_crop(
    zarr_path: Path,
    config_path: Optional[Path],
    cli_source_type: Optional[str],
    cli_source_path: Optional[str],
    cli_selection_policy: Optional[str],
) -> None:
    console = Console()
    root = zarr.open(str(zarr_path), mode="r")

    config = _load_config(config_path) if config_path else {}
    crop_params = dict(config.get("crop", {}) or {})
    selection_policy = cli_selection_policy or crop_params.get("selection_policy")

    config_source_type = crop_params.get("source_type")
    config_source_path = crop_params.get("source_path")

    source_path = cli_source_path or config_source_path
    source_path = _normalize_source_path(source_path)
    raw_source_type = cli_source_type or config_source_type
    source_type = infer_detection_source_type(source_path, raw_source_type)

    console.print(f"[bold]Crop dry-run for[/bold] {zarr_path}")
    if config_path:
        console.print(f"[dim]Config:[/dim] {config_path}")
    console.print(f"[dim]Requested source_type:[/dim] {source_type}")
    if selection_policy:
        console.print(f"[dim]Selection policy:[/dim] {selection_policy}")
    if source_path:
        console.print(f"[dim]Requested source_path:[/dim] {source_path}")

    try:
        resolved_path, source_group, detection_source, resolved_type = get_detection_source_info(
            root=root,
            source_type=source_type,
            source_path_override=source_path,
            console=console,
            selection_policy=selection_policy,
        )
    except ValueError as exc:
        console.print(f"[red]Unable to resolve detection source: {exc}[/red]")
        return

    source_parts = str(resolved_path).strip().strip("/").split("/")
    if resolved_type != "detect" or len(source_parts) != 2 or source_parts[0] != "detect_runs":
        console.print(
            "[red]Unsupported future ordinary crop source:[/red] "
            f"{resolved_type} [{resolved_path}]. Select an exact detect_runs/<run> source."
        )
        return

    roi_sz = tuple(crop_params.get("roi_sz", [256, 256]))
    try:
        video_source_type, video_path = get_video_source(root, console=None)
        canonical_preflight = _preflight_ordinary_crop_coordinates(
            root,
            zarr_path=str(zarr_path),
            source_path=resolved_path,
            source_group=source_group,
            video_source_type=video_source_type,
            video_path=video_path,
            roi_size=roi_sz,
        )
    except Exception as exc:
        console.print(
            "[red]Canonical ordinary-crop preflight failed:[/red] "
            f"{type(exc).__name__}: {exc}"
        )
        return

    frame_indices = canonical_preflight.frame_indices
    total_detections = canonical_preflight.row_count
    if total_detections == 0:
        console.print(
            "[yellow]Selected source is canonically valid but contains zero "
            "detections; no crop run would be created.[/yellow]"
        )
        return

    total_frames = canonical_preflight.total_frames
    unique_frames = np.unique(frame_indices)
    frames_with_detections = int(unique_frames.size)
    coverage = (frames_with_detections / total_frames) * 100 if total_frames else 0.0

    # Detection source stats
    det_stats = None
    if detection_source is not None:
        det_stats = {
            "total": int(detection_source.size),
            "real": int(np.sum(detection_source == 0)),
            "interp": int(np.sum(detection_source == 1)),
        }

    scheduler = crop_params.get("scheduler", "processes")
    num_workers = crop_params.get("num_workers", "auto")

    table = Table(title="Crop Dry-Run Summary", show_lines=False)
    table.add_column("Field", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Resolved source type", resolved_type)
    table.add_row("Resolved source path", resolved_path)
    if selection_policy:
        table.add_row("Selection policy", selection_policy)
    table.add_row("Total detections", f"{total_detections:,}")
    table.add_row("Frames with detections", f"{frames_with_detections:,}")
    table.add_row("Total frames", f"{total_frames:,}")
    table.add_row("Coverage", f"{coverage:.2f}%")
    table.add_row("ROI size", f"{roi_sz[0]}×{roi_sz[1]}")
    table.add_row("Scheduler", str(scheduler))
    table.add_row("Workers", str(num_workers))

    if det_stats:
        table.add_row("Real detections", f"{det_stats['real']:,}")
        table.add_row("Interpolated detections", f"{det_stats['interp']:,}")

    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Show the detection source and stats that the crop stage would use."
        )
    )
    parser.add_argument("zarr_path", type=Path, help="Path to Palette Zarr archive.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/fisheye/default.yaml"),
        help="Pipeline config to read crop defaults from (default: configs/fisheye/default.yaml).",
    )
    parser.add_argument(
        "--crop-source",
        choices=["auto", "refined", "detect", "manual", "filtered", "interpolated"],
        help=(
            "Override crop source type. Future ordinary crop publication "
            "supports detect; legacy/refined resolutions are reported as unsupported."
        ),
    )
    parser.add_argument(
        "--crop-source-path",
        help=(
            "Explicit future-canonical detection source path: detect_runs/<run>."
        ),
    )
    parser.add_argument(
        "--selection-policy",
        choices=["training", "full_recording"],
        help="Policy for auto source selection (overrides config).",
    )

    args = parser.parse_args()
    simulate_crop(
        zarr_path=args.zarr_path,
        config_path=args.config,
        cli_source_type=args.crop_source,
        cli_source_path=args.crop_source_path,
        cli_selection_policy=args.selection_policy,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
