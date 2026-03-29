#!/usr/bin/env python3
"""
Inspect the positional inputs that feed the 2D histogram in
plot_track_kinematics.py.

For a given track kinematics run/track, this reports how many samples are
present, how many contain finite coordinates, and the min/max X/Y values in
both pixels and millimetres (when those datasets exist). It also highlights
which unit plot_track_kinematics would use for the heatmap.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import zarr
from rich.console import Console

from fisheye.analysis.plot_track_kinematics import (
    pick_units,
    resolve_track,
    resolve_track_kinematics_run,
)


def _format_float(value: float) -> str:
    return f"{value:.3f}"


def _summarize_positions(
    console: Console,
    label: str,
    pos_x: Optional[np.ndarray],
    pos_y: Optional[np.ndarray],
) -> Tuple[int, int]:
    if pos_x is None or pos_y is None:
        console.log(f"[yellow]{label}: dataset missing.[/yellow]")
        return 0, 0

    total_samples = pos_x.shape[0]
    finite_mask = np.isfinite(pos_x) & np.isfinite(pos_y)
    valid_count = int(np.count_nonzero(finite_mask))
    coverage = valid_count / total_samples if total_samples else 0.0

    console.log(
        f"{label}: samples={total_samples}, valid={valid_count} "
        f"({coverage*100:.1f}% coverage)"
    )

    if valid_count == 0:
        console.log(f"  [red]{label}: no finite coordinates available.[/red]")
        return total_samples, valid_count

    valid_x = pos_x[finite_mask]
    valid_y = pos_y[finite_mask]
    min_x = float(np.min(valid_x))
    max_x = float(np.max(valid_x))
    min_y = float(np.min(valid_y))
    max_y = float(np.max(valid_y))

    console.log(
        f"  X range: {_format_float(min_x)} – {_format_float(max_x)}  |  "
        f"Y range: {_format_float(min_y)} – {_format_float(max_y)}"
    )

    return total_samples, valid_count


def analyze_track_positions(
    zarr_path: Path,
    track_kinematics_run_name: Optional[str],
    track_id: Optional[int],
) -> None:
    console = Console()
    console.log(f"[bold]Opening Zarr archive:[/bold] {zarr_path}")
    root = zarr.open(str(zarr_path), mode="r")

    run_name, run_group = resolve_track_kinematics_run(
        root, track_kinematics_run_name, console
    )
    track_id_resolved, track_group = resolve_track(run_group, track_id, console)

    unit_label, pos_x_default, pos_y_default = pick_units(run_group, track_group)
    console.log(f"[bold]plot_track_kinematics will use units: {unit_label}[/bold]")

    positions_px = track_group["positions_px"][:] if "positions_px" in track_group else None
    if positions_px is not None:
        pos_px_x = positions_px[:, 0]
        pos_px_y = positions_px[:, 1]
    else:
        pos_px_x = pos_px_y = None

    positions_mm = track_group["positions_mm"][:] if "positions_mm" in track_group else None
    if positions_mm is not None:
        pos_mm_x = positions_mm[:, 0]
        pos_mm_y = positions_mm[:, 1]
    else:
        pos_mm_x = pos_mm_y = None

    total_px, valid_px = _summarize_positions(console, "Pixels (positions_px)", pos_px_x, pos_px_y)
    total_mm, valid_mm = _summarize_positions(console, "Millimetres (positions_mm)", pos_mm_x, pos_mm_y)

    if valid_px == 0 and valid_mm == 0:
        console.log("[red]No usable positional data found for this track.[/red]")
        return

    if pos_x_default is not None and pos_y_default is not None:
        console.log(
            f"[cyan]plot_track_kinematics heatmap input coverage:[/cyan] "
            f"{valid_mm if unit_label == 'mm' else valid_px} samples "
            f"out of {total_mm if unit_label == 'mm' else total_px}"
        )

    if "time_seconds" in track_group:
        time_seconds = track_group["time_seconds"][:]
        duration = float(np.nanmax(time_seconds) - np.nanmin(time_seconds)) if time_seconds.size else 0.0
        console.log(f"Time span covered: {duration:.2f} s")

    console.log(
        f"[green]Inspection complete for track kinematics run '{run_name}', track {track_id_resolved}.[/green]"
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect the positional inputs used by plot_track_kinematics.py for a specific track kinematics track."
    )
    parser.add_argument("zarr_path", type=Path, help="Path to the Palette Zarr archive.")
    parser.add_argument(
        "--track-kinematics-run",
        dest="track_kinematics_run",
        help="Track kinematics run name (defaults to latest).",
    )
    parser.add_argument("--track-id", type=int, help="Track ID to inspect (defaults to the first track).")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    analyze_track_positions(
        zarr_path=args.zarr_path,
        track_kinematics_run_name=args.track_kinematics_run,
        track_id=args.track_id,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
