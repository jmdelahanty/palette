#!/usr/bin/env python3
"""
Inspect the spatial data that feeds the training heatmaps inside a Palette Zarr
archive. Reports frame coverage plus min/max X/Y camera coordinates for each
identified training period (pre/training/post).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import zarr
from rich.console import Console

from fisheye.analysis.plot_training_heatmaps_zarr import (
    _build_stim_to_camera_map,
    _collect_positions,
    _determine_periods,
    _load_events,
    _load_frame_metadata,
    _resolve_latest,
)


def _format_float(value: float) -> str:
    return f"{value:.2f}"


def analyze_heatmap_inputs(
    zarr_path: Path,
    movement_run_name: Optional[str],
    stimulus_run_name: Optional[str],
) -> None:
    console = Console()
    console.log(f"[bold]Opening Zarr archive:[/bold] {zarr_path}")
    root = zarr.open(zarr_path, mode="r")

    analysis = root.get("analysis")
    if analysis is None:
        raise ValueError("Archive missing 'analysis' group.")

    movement_parent = analysis.get("movement_runs")
    if movement_parent is None or not movement_parent:
        raise ValueError("No movement runs found under analysis/movement_runs.")
    movement_run_group, movement_run_name = _resolve_latest(
        movement_parent, movement_run_name, "movement_run", console
    )

    stimulus_parent = analysis.get("stimulus_runs")
    if stimulus_parent is None or not stimulus_parent:
        raise ValueError("No stimulus runs found under analysis/stimulus_runs.")
    stimulus_run_group, stimulus_run_name = _resolve_latest(
        stimulus_parent, stimulus_run_name, "stimulus_run", console
    )

    console.log("[bold]Loading stimulus metadata...[/bold]")
    events = _load_events(stimulus_run_group)
    frame_metadata = _load_frame_metadata(stimulus_run_group)
    stim_to_camera = _build_stim_to_camera_map(frame_metadata)
    camera_frames_all = frame_metadata.get("triggering_camera_frame_id")
    if camera_frames_all is None:
        raise ValueError("Stimulus metadata missing 'triggering_camera_frame_id'.")
    camera_frames_all = camera_frames_all.astype(np.int64, copy=False)

    console.log("[bold]Loading movement run positions...[/bold]")
    frames, positions = _collect_positions(movement_run_group)

    periods = _determine_periods(events, stim_to_camera, camera_frames_all, console)

    period_specs = [
        ("pre_training", periods.pre_start, periods.pre_end),
        ("training", periods.train_start, periods.train_end),
        ("post_training", periods.post_start, periods.post_end),
    ]

    console.log("[bold]Analyzing heatmap inputs...[/bold]")
    for label, start_frame, end_frame in period_specs:
        mask = (frames >= start_frame) & (frames <= end_frame)
        if not np.any(mask):
            console.log(f"[yellow]{label}: no detections within frame range {start_frame}–{end_frame}.[/yellow]")
            continue

        period_positions = positions[mask]
        valid = np.isfinite(period_positions).all(axis=1)
        period_positions = period_positions[valid]
        detections = period_positions.shape[0]
        unique_frames = np.unique(frames[mask][valid])
        total_span = max(end_frame - start_frame + 1, 1)
        coverage = unique_frames.size / total_span

        console.log(
            f"[cyan]{label}[/cyan] frames {start_frame}–{end_frame} "
            f"(detections={detections}, coverage={coverage*100:.1f}%)"
        )
        if detections == 0:
            console.log("  [yellow]All detections were invalid (NaN) in this period.[/yellow]")
            continue

        min_x = float(np.min(period_positions[:, 0]))
        max_x = float(np.max(period_positions[:, 0]))
        min_y = float(np.min(period_positions[:, 1]))
        max_y = float(np.max(period_positions[:, 1]))

        console.log(
            "  X range: "
            f"{_format_float(min_x)} – {_format_float(max_x)}  |  "
            "Y range: "
            f"{_format_float(min_y)} – {_format_float(max_y)}"
        )

    console.log(
        f"[green]Inspection complete for movement run '{movement_run_name}' "
        f"and stimulus run '{stimulus_run_name}'.[/green]"
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect the spatial inputs used when plotting training heatmaps from a Palette Zarr archive."
    )
    parser.add_argument("zarr_path", type=Path, help="Path to the Palette Zarr archive.")
    parser.add_argument("--movement-run", help="Specific movement run name (defaults to latest).")
    parser.add_argument("--stimulus-run", help="Specific stimulus run name under analysis/stimulus_runs.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    analyze_heatmap_inputs(
        zarr_path=args.zarr_path,
        movement_run_name=args.movement_run,
        stimulus_run_name=args.stimulus_run,
    )


if __name__ == "__main__":  # pragma: no cover
    main()

