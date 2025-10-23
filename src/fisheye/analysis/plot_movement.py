"""Quick plotting helpers for movement_runs outputs.

Generates speed-vs-time, heading-vs-time, and position heatmaps for a
selected track within an analysis/movement_runs entry.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import zarr
from rich.console import Console


def resolve_movement_run(root: zarr.Group, requested: Optional[str], console: Console) -> tuple[str, zarr.Group]:
    if "analysis" not in root or "movement_runs" not in root["analysis"]:
        raise ValueError("No movement_runs group found under analysis/.")

    parent = root["analysis/movement_runs"]
    if requested:
        if requested not in parent:
            raise ValueError(f"Movement run '{requested}' not found.")
        return requested, parent[requested]

    run_name = parent.attrs.get("latest")
    if not run_name:
        raise ValueError("movement_runs does not have a 'latest' attribute; specify --movement-run.")
    if run_name not in parent:
        raise ValueError(f"Movement run '{run_name}' referenced by 'latest' is missing.")
    console.print(f"Using movement run: [cyan]{run_name}[/cyan]")
    return run_name, parent[run_name]


def resolve_track(group: zarr.Group, track_id: Optional[int], console: Console) -> tuple[int, zarr.Group]:
    track_ids = group["track_ids"][:]
    if track_ids.size == 0:
        raise ValueError("Movement run contains no tracks.")

    if track_id is None:
        track_id = int(track_ids[0])
        console.print(f"Using track id: [cyan]{track_id}[/cyan]")
    elif track_id not in track_ids:
        raise ValueError(f"Track id {track_id} not found in run. Available IDs: {track_ids.tolist()}")

    track_group_name = f"id_{int(track_id)}"
    if track_group_name not in group["tracks"]:
        raise ValueError(f"Track group '{track_group_name}' missing under movement run.")
    return int(track_id), group["tracks"][track_group_name]


def pick_units(run_group: zarr.Group, track_group: zarr.Group) -> tuple[str, np.ndarray, np.ndarray]:
    pixel_to_mm = run_group.attrs.get("pixel_to_mm")
    positions_mm = track_group["positions_mm"][:]
    if pixel_to_mm and np.isfinite(positions_mm).any():
        return "mm", positions_mm[:, 0], positions_mm[:, 1]
    positions_px = track_group["positions_px"][:]
    return "px", positions_px[:, 0], positions_px[:, 1]


def plot_track(
    run_group: zarr.Group,
    track_group: zarr.Group,
    track_id: int,
    save_path: Optional[Path],
    bins: int,
    console: Console,
) -> None:
    time_seconds = track_group["time_seconds"][:]
    smoothed_speed_px = track_group["smoothed_speed_px"][:]
    smoothed_speed_mm = track_group["smoothed_speed_mm"][:]
    smoothed_heading_deg = track_group["smoothed_heading_degrees"][:]
    smoothed_accel_px = track_group["smoothed_acceleration_px"][:]
    smoothed_accel_mm = track_group["smoothed_acceleration_mm"][:]

    unit_label, pos_x, pos_y = pick_units(run_group, track_group)
    speed = smoothed_speed_mm if unit_label == "mm" and np.isfinite(smoothed_speed_mm).any() else smoothed_speed_px
    speed_label = f"Speed ({unit_label}/s)" if unit_label in {"mm", "px"} else "Speed"
    accel = (
        smoothed_accel_mm
        if unit_label == "mm" and np.isfinite(smoothed_accel_mm).any()
        else smoothed_accel_px
    )
    accel_label = f"Acceleration ({unit_label}/s^2)" if unit_label in {"mm", "px"} else "Acceleration"

    fig, axes = plt.subplots(5, 1, figsize=(10, 16))

    axes[0].plot(time_seconds, speed, color="tab:blue", linewidth=1.2)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel(speed_label)
    axes[0].set_title(f"Track {track_id}: Speed over time")
    axes[0].grid(alpha=0.3)

    axes[1].plot(time_seconds, accel, color="tab:red", linewidth=1.0)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel(accel_label)
    axes[1].set_title("Smoothed acceleration over time")
    axes[1].grid(alpha=0.3)

    axes[2].plot(time_seconds, smoothed_heading_deg, color="tab:orange", linewidth=1.0)
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Heading (deg)")
    axes[2].set_title("Heading over time (smoothed)")
    axes[2].grid(alpha=0.3)

    cumulative = track_group["cumulative_distance_mm"][:]
    if not (np.isfinite(cumulative).any() and unit_label == "mm"):
        cumulative = track_group["cumulative_distance_px"][:]
        cumulative_label = "Cumulative distance (px)"
    else:
        cumulative_label = "Cumulative distance (mm)"
    axes[3].plot(time_seconds, cumulative, color="tab:green", linewidth=1.2)
    axes[3].set_xlabel("Time (s)")
    axes[3].set_ylabel(cumulative_label)
    axes[3].set_title("Cumulative distance over time")
    axes[3].grid(alpha=0.3)

    heat = axes[4].hist2d(pos_x, pos_y, bins=bins, cmap="inferno")
    axes[4].set_xlabel(f"X ({unit_label})")
    axes[4].set_ylabel(f"Y ({unit_label})")
    axes[4].set_title("Position density")
    cbar = fig.colorbar(heat[3], ax=axes[4])
    cbar.set_label("Counts")

    fig.suptitle(f"Movement summary – track {track_id}")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path:
        fig.savefig(save_path)
        console.print(f"[green]Saved plot to {save_path}[/green]")
    else:
        plt.show()

    plt.close(fig)


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Plot movement_run track metrics.")
    parser.add_argument("zarr_path", help="Path to the Palette Zarr archive.")
    parser.add_argument("--movement-run", help="Movement run name (defaults to latest).")
    parser.add_argument("--track-id", type=int, help="Track ID to visualize.")
    parser.add_argument("--bins", type=int, default=200, help="Histogram bins for position heatmap (default: 200).")
    parser.add_argument("--save", help="Path to save the figure instead of showing interactively.")

    args = parser.parse_args(argv)

    console = Console()
    root = zarr.open(args.zarr_path, mode="r")

    run_name, run_group = resolve_movement_run(root, args.movement_run, console)
    track_id, track_group = resolve_track(run_group, args.track_id, console)

    save_path = Path(args.save) if args.save else None
    plot_track(run_group, track_group, track_id, save_path, args.bins, console)


if __name__ == "__main__":  # pragma: no cover
    main()
