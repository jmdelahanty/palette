#!/usr/bin/env python3
"""
Predict what the crop stage would do without actually writing ROIs.

This diagnostic mirrors the crop stage's detection-source resolution logic and
reports the detection run, frame counts, and interpolated metadata that would
be used. It is helpful for confirming that the pipeline will consume the
canonical refined detect surface when available, with legacy sparse fallback
only on historical archives.
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
    infer_detection_source_type,
    get_detection_source_info,
)
from fisheye.utils.metadata import get_total_frames


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

    frame_indices = source_group["frame_indices"][:]
    total_detections = int(frame_indices.shape[0])
    if total_detections == 0:
        console.print("[yellow]Selected detection source has zero detections.[/yellow]")
        return

    total_frames = get_total_frames(root, source_group)
    if total_frames is None:
        total_frames = int(frame_indices.max(initial=0) + 1)
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

    roi_sz = tuple(crop_params.get("roi_sz", [256, 256]))
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
            "Override crop source type. 'auto' prefers the canonical current "
            "refined surface and falls back to raw detect; 'refined' requires "
            "the canonical curated refined surface. "
            "'manual'/'filtered'/'interpolated' are legacy sparse "
            "compatibility modes."
        ),
    )
    parser.add_argument(
        "--crop-source-path",
        help=(
            "Explicit detection source path (detect_runs/<run> or the "
            "canonical refined path "
            "refined_detect_runs/<run>/instances)."
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
