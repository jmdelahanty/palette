#!/usr/bin/env python3
"""Measure the noise floor for stationary fish from offline detection data.

This script analyzes movement data to empirically determine the noise floor -
the minimum speed observed during periods when the fish is stationary. This
helps set appropriate thresholds for distinguishing real movement from
centroid jitter and detection noise.

The script identifies stationary periods based on instantaneous speed and
computes statistics to recommend appropriate speed floor thresholds.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import zarr
from rich.console import Console
from rich.table import Table

from fisheye.shared.zarr.schema import get_run_group


def load_movement_run(
    zarr_path: str,
    movement_run: Optional[str] = None,
    console: Optional[Console] = None,
) -> Tuple[zarr.Group, Dict[str, zarr.Group]]:
    """Load movement run data from zarr archive.

    Args:
        zarr_path: Path to zarr archive
        movement_run: Movement run name (defaults to latest offline)
        console: Rich console for output

    Returns:
        Tuple of (run_group, tracks_dict)
    """
    if console is None:
        console = Console()

    root = zarr.open(str(zarr_path), mode="r")

    # Try to find offline movement run
    if movement_run is None:
        # Look for latest offline run
        if "analysis" in root and "movement_runs" in root["analysis"]:
            movement_parent = root["analysis"]["movement_runs"]
            if "offline" in movement_parent:
                offline_group = movement_parent["offline"]
                movement_run = offline_group.attrs.get("latest")
                if movement_run:
                    console.print(f"[cyan]Using latest offline run:[/cyan] {movement_run}")
                else:
                    raise ValueError("No offline movement runs found")
            else:
                raise ValueError("No offline movement runs found")
        else:
            raise ValueError("No movement_runs found in archive")

    # Load the run
    if "analysis" in root and "movement_runs" in root["analysis"]:
        movement_parent = root["analysis"]["movement_runs"]
        if "offline" in movement_parent:
            offline_group = movement_parent["offline"]
            if movement_run in offline_group:
                run_group = offline_group[movement_run]
            else:
                raise ValueError(f"Movement run '{movement_run}' not found in offline runs")
        else:
            raise ValueError("No offline movement runs found")
    else:
        raise ValueError("No movement_runs found in archive")

    # Load tracks
    tracks_dict = {}
    if "tracks" in run_group:
        tracks_group = run_group["tracks"]
        for track_id in tracks_group.keys():
            tracks_dict[track_id] = tracks_group[track_id]
        console.print(f"[cyan]Loaded {len(tracks_dict)} tracks[/cyan]")
    else:
        raise ValueError("No tracks found in movement run")

    return run_group, tracks_dict


def identify_stationary_periods(
    instantaneous_speed_mm: np.ndarray,
    frames: np.ndarray,
    max_stationary_speed: float = 0.5,
    min_period_frames: int = 30,
) -> List[Tuple[int, int]]:
    """Identify periods where the fish appears stationary.

    Args:
        instantaneous_speed_mm: Instantaneous speed array (mm/s)
        frames: Frame indices
        max_stationary_speed: Maximum speed to consider stationary (mm/s)
        min_period_frames: Minimum number of consecutive frames

    Returns:
        List of (start_idx, end_idx) tuples for stationary periods
    """
    # Find frames below threshold
    is_stationary = instantaneous_speed_mm < max_stationary_speed

    # Find consecutive runs
    periods = []
    start_idx = None

    for i in range(len(is_stationary)):
        if is_stationary[i] and start_idx is None:
            start_idx = i
        elif not is_stationary[i] and start_idx is not None:
            # End of period
            if i - start_idx >= min_period_frames:
                periods.append((start_idx, i))
            start_idx = None

    # Handle final period
    if start_idx is not None and len(is_stationary) - start_idx >= min_period_frames:
        periods.append((start_idx, len(is_stationary)))

    return periods


def compute_noise_statistics(
    tracks_dict: Dict[str, zarr.Group],
    max_stationary_speed: float = 0.5,
    min_period_frames: int = 30,
    console: Optional[Console] = None,
) -> Dict[str, any]:
    """Compute noise floor statistics from stationary periods.

    Args:
        tracks_dict: Dictionary of track groups
        max_stationary_speed: Maximum speed to consider stationary (mm/s)
        min_period_frames: Minimum stationary period length
        console: Rich console for output

    Returns:
        Dictionary with noise floor statistics
    """
    if console is None:
        console = Console()

    all_stationary_speeds = []
    all_stationary_displacements = []
    track_stats = []

    for track_id, track_group in tracks_dict.items():
        # Load data
        instantaneous_speed_mm = track_group["instantaneous_speed_mm"][:]
        distance_per_frame_mm = track_group["distance_per_frame_mm"][:]
        frame_indices = track_group["frame_indices"][:]

        # Skip if no valid data
        valid_mask = np.isfinite(instantaneous_speed_mm)
        if not valid_mask.any():
            continue

        # Identify stationary periods
        periods = identify_stationary_periods(
            instantaneous_speed_mm,
            frame_indices,
            max_stationary_speed,
            min_period_frames,
        )

        if not periods:
            continue

        # Collect statistics for this track
        track_stationary_speeds = []
        track_stationary_displacements = []

        for start_idx, end_idx in periods:
            period_speeds = instantaneous_speed_mm[start_idx:end_idx]
            period_displacements = distance_per_frame_mm[start_idx:end_idx]

            # Remove any NaN/inf values
            valid_speeds = period_speeds[np.isfinite(period_speeds)]
            valid_displacements = period_displacements[np.isfinite(period_displacements)]

            track_stationary_speeds.extend(valid_speeds)
            track_stationary_displacements.extend(valid_displacements)

        if track_stationary_speeds:
            track_stationary_speeds = np.array(track_stationary_speeds)
            track_stationary_displacements = np.array(track_stationary_displacements)

            all_stationary_speeds.extend(track_stationary_speeds)
            all_stationary_displacements.extend(track_stationary_displacements)

            track_stats.append({
                "track_id": track_id,
                "n_periods": len(periods),
                "total_stationary_frames": len(track_stationary_speeds),
                "mean_speed": np.mean(track_stationary_speeds),
                "median_speed": np.median(track_stationary_speeds),
                "p95_speed": np.percentile(track_stationary_speeds, 95),
                "max_speed": np.max(track_stationary_speeds),
                "mean_displacement": np.mean(track_stationary_displacements),
                "median_displacement": np.median(track_stationary_displacements),
                "p95_displacement": np.percentile(track_stationary_displacements, 95),
                "max_displacement": np.max(track_stationary_displacements),
            })

    # Compute aggregate statistics
    if not all_stationary_speeds:
        raise ValueError("No stationary periods found with current thresholds")

    all_stationary_speeds = np.array(all_stationary_speeds)
    all_stationary_displacements = np.array(all_stationary_displacements)

    statistics = {
        "n_tracks": len(track_stats),
        "total_stationary_frames": len(all_stationary_speeds),
        "speed_mean": np.mean(all_stationary_speeds),
        "speed_median": np.median(all_stationary_speeds),
        "speed_std": np.std(all_stationary_speeds),
        "speed_p50": np.percentile(all_stationary_speeds, 50),
        "speed_p75": np.percentile(all_stationary_speeds, 75),
        "speed_p90": np.percentile(all_stationary_speeds, 90),
        "speed_p95": np.percentile(all_stationary_speeds, 95),
        "speed_p99": np.percentile(all_stationary_speeds, 99),
        "speed_max": np.max(all_stationary_speeds),
        "displacement_mean": np.mean(all_stationary_displacements),
        "displacement_median": np.median(all_stationary_displacements),
        "displacement_std": np.std(all_stationary_displacements),
        "displacement_p95": np.percentile(all_stationary_displacements, 95),
        "displacement_max": np.max(all_stationary_displacements),
        "track_stats": track_stats,
    }

    return statistics


def print_statistics(
    statistics: Dict[str, any],
    console: Console,
) -> None:
    """Print noise floor statistics in a formatted table.

    Args:
        statistics: Statistics dictionary from compute_noise_statistics
        console: Rich console for output
    """
    console.rule("[bold]Noise Floor Analysis Results[/bold]")

    # Summary statistics
    console.print(f"\n[bold cyan]Dataset Summary:[/bold cyan]")
    console.print(f"  Tracks analyzed: {statistics['n_tracks']}")
    console.print(f"  Total stationary frames: {statistics['total_stationary_frames']}")

    # Speed statistics table
    speed_table = Table(title="Instantaneous Speed Statistics (mm/s)")
    speed_table.add_column("Metric", style="cyan")
    speed_table.add_column("Value", justify="right", style="yellow")

    speed_table.add_row("Mean", f"{statistics['speed_mean']:.4f}")
    speed_table.add_row("Median", f"{statistics['speed_median']:.4f}")
    speed_table.add_row("Std Dev", f"{statistics['speed_std']:.4f}")
    speed_table.add_row("50th percentile", f"{statistics['speed_p50']:.4f}")
    speed_table.add_row("75th percentile", f"{statistics['speed_p75']:.4f}")
    speed_table.add_row("90th percentile", f"{statistics['speed_p90']:.4f}")
    speed_table.add_row("95th percentile", f"{statistics['speed_p95']:.4f}")
    speed_table.add_row("99th percentile", f"{statistics['speed_p99']:.4f}")
    speed_table.add_row("Maximum", f"{statistics['speed_max']:.4f}")

    console.print(speed_table)

    # Displacement statistics table
    disp_table = Table(title="Frame-to-Frame Displacement Statistics (mm)")
    disp_table.add_column("Metric", style="cyan")
    disp_table.add_column("Value", justify="right", style="yellow")

    disp_table.add_row("Mean", f"{statistics['displacement_mean']:.4f}")
    disp_table.add_row("Median", f"{statistics['displacement_median']:.4f}")
    disp_table.add_row("Std Dev", f"{statistics['displacement_std']:.4f}")
    disp_table.add_row("95th percentile", f"{statistics['displacement_p95']:.4f}")
    disp_table.add_row("Maximum", f"{statistics['displacement_max']:.4f}")

    console.print(disp_table)

    # Recommendations
    console.print("\n[bold green]Recommended Speed Floor Thresholds:[/bold green]")
    console.print(f"  Conservative (95th percentile): [yellow]{statistics['speed_p95']:.4f} mm/s[/yellow]")
    console.print(f"  Moderate (99th percentile): [yellow]{statistics['speed_p99']:.4f} mm/s[/yellow]")
    console.print(f"  Permissive (mean + 2σ): [yellow]{statistics['speed_mean'] + 2 * statistics['speed_std']:.4f} mm/s[/yellow]")

    console.print("\n[bold]Usage:[/bold]")
    console.print("  Use the conservative threshold to ensure all noise is removed.")
    console.print("  Use the moderate threshold to balance noise removal with sensitivity.")
    console.print("  Apply with: --speed-floor-mm <threshold>")


def main():
    parser = argparse.ArgumentParser(
        description="Measure noise floor from stationary periods in movement data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "zarr_path",
        type=str,
        help="Path to zarr archive",
    )
    parser.add_argument(
        "--movement-run",
        type=str,
        help="Movement run name (defaults to latest offline)",
    )
    parser.add_argument(
        "--max-stationary-speed",
        type=float,
        default=0.5,
        help="Maximum speed to consider stationary in mm/s (default: 0.5)",
    )
    parser.add_argument(
        "--min-period-frames",
        type=int,
        default=30,
        help="Minimum consecutive frames for a stationary period (default: 30)",
    )

    args = parser.parse_args()

    console = Console()

    try:
        # Load movement data
        console.print(f"[bold]Loading movement data from:[/bold] {args.zarr_path}")
        run_group, tracks_dict = load_movement_run(
            args.zarr_path,
            args.movement_run,
            console,
        )

        # Display run info
        method = run_group.attrs.get("method", "unknown")
        fps = run_group.attrs.get("fps", "unknown")
        pixel_to_mm = run_group.attrs.get("pixel_to_mm", "unknown")

        console.print(f"[cyan]Method:[/cyan] {method}")
        console.print(f"[cyan]FPS:[/cyan] {fps}")
        console.print(f"[cyan]Calibration:[/cyan] {pixel_to_mm} pixels/mm")

        # Compute noise floor statistics
        console.print(f"\n[bold]Analyzing stationary periods...[/bold]")
        console.print(f"  Max stationary speed threshold: {args.max_stationary_speed} mm/s")
        console.print(f"  Min period length: {args.min_period_frames} frames")

        statistics = compute_noise_statistics(
            tracks_dict,
            args.max_stationary_speed,
            args.min_period_frames,
            console,
        )

        # Print results
        print_statistics(statistics, console)

        console.print("\n[bold green]✓ Noise floor analysis complete[/bold green]")

    except Exception as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
