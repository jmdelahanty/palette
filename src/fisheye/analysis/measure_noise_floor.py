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
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
from rich.console import Console
from rich.table import Table

from fisheye.analysis.track_kinematics_io import (
    TrackKinematicsTrackTables,
    list_track_ids,
    load_track_kinematics_track,
    resolve_track_kinematics_run,
)
from fisheye.shared.zarr_io import open_zarr_root


def load_track_kinematics_run(
    zarr_path: str,
    track_kinematics_run: Optional[str] = None,
    console: Optional[Console] = None,
) -> Tuple[Mapping[str, Any], Dict[str, TrackKinematicsTrackTables]]:
    """Load one verified canonical offline track-motion run.

    Args:
        zarr_path: Path to zarr archive
        track_kinematics_run: Track kinematics run name (defaults to latest offline)
        console: Rich console for output

    Returns:
        Tuple of (immutable run attrs, logical verified tracks)
    """
    if console is None:
        console = Console()

    root = open_zarr_root(zarr_path, mode="r")
    requested_name = str(track_kinematics_run or "latest")
    run_group, resolved_name, _run_path = resolve_track_kinematics_run(
        root,
        run_name=requested_name,
        scope="offline",
    )
    if requested_name == "latest":
        console.print(
            "[cyan]Using latest verified offline track kinematics run:[/cyan] "
            f"{resolved_name}"
        )
    track_ids = list_track_ids(run_group)
    if not track_ids:
        raise ValueError("Verified track-motion run has no tracks.")
    tracks: Dict[str, TrackKinematicsTrackTables] = {}
    manifest_sha256: str | None = None
    run_attrs: Mapping[str, Any] | None = None
    for track_id in track_ids:
        tables = load_track_kinematics_track(
            root,
            run_name=resolved_name,
            scope="offline",
            track_id=int(track_id),
            required_speed_levels=("raw",),
        )
        if manifest_sha256 is None:
            manifest_sha256 = tables.motion_manifest_sha256
            run_attrs = dict(tables.run_attrs)
        elif tables.motion_manifest_sha256 != manifest_sha256:
            raise ValueError(
                "Track-motion authority changed while loading noise-floor inputs."
            )
        tracks[f"id_{int(track_id)}"] = tables
    console.print(f"[cyan]Loaded {len(tracks)} verified tracks[/cyan]")
    assert run_attrs is not None
    return run_attrs, tracks


def identify_stationary_periods(
    raw_speed_mm: np.ndarray,
    frames: np.ndarray,
    max_stationary_speed: float = 0.5,
    min_period_frames: int = 30,
) -> List[Tuple[int, int]]:
    """Identify periods where the fish appears stationary.

    Args:
        raw_speed_mm: Raw speed array (mm/s)
        frames: Frame indices
        max_stationary_speed: Maximum speed to consider stationary (mm/s)
        min_period_frames: Minimum number of consecutive frames

    Returns:
        List of (start_idx, end_idx) tuples for stationary periods
    """
    speeds = np.asarray(raw_speed_mm, dtype=np.float64).reshape(-1)
    acquisition_frames = np.asarray(frames, dtype=np.int64).reshape(-1)
    if speeds.shape != acquisition_frames.shape:
        raise ValueError("raw_speed_mm and acquisition frame identities must align.")
    # Find valid rows below threshold. A gap in acquisition-frame identity ends a
    # stationary period even when the surrounding speed samples are both small.
    is_stationary = np.isfinite(speeds) & (speeds < max_stationary_speed)

    # Find consecutive runs
    periods = []
    start_idx = None

    for i in range(len(is_stationary)):
        if (
            i > 0
            and acquisition_frames[i] != acquisition_frames[i - 1] + 1
            and start_idx is not None
        ):
            if i - start_idx >= min_period_frames:
                periods.append((start_idx, i))
            start_idx = None
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
    tracks_dict: Mapping[str, TrackKinematicsTrackTables],
    max_stationary_speed: float = 0.5,
    min_period_frames: int = 30,
    console: Optional[Console] = None,
) -> Dict[str, Any]:
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
    all_stationary_path_distances = []
    track_stats = []

    for track_id, track_group in tracks_dict.items():
        # Load data
        if track_group.source_acquisition_frame_index is None:
            raise ValueError(
                f"Verified track motion {track_group.track_path} has no acquisition-frame identity."
            )
        if track_group.sample_valid is None or track_group.transition_valid is None:
            raise ValueError(
                f"Verified track motion {track_group.track_path} lacks validity surfaces."
            )
        try:
            raw_speed_mm = np.asarray(
                track_group.speed_mm_by_level["raw"], dtype=np.float64
            )
            frame_path_distance_mm = np.asarray(
                track_group.frame_path_distance_mm_by_level["raw"],
                dtype=np.float64,
            )
        except KeyError as exc:
            raise ValueError(
                f"Verified track motion {track_group.track_path} lacks raw physical motion surfaces."
            ) from exc
        frame_indices = np.asarray(
            track_group.source_acquisition_frame_index,
            dtype=np.int64,
        )
        sample_valid = np.asarray(track_group.sample_valid, dtype=bool)
        transition_valid = np.asarray(track_group.transition_valid, dtype=bool)
        if not (
            raw_speed_mm.shape
            == frame_path_distance_mm.shape
            == frame_indices.shape
            == sample_valid.shape
            == transition_valid.shape
        ):
            raise ValueError(
                f"Verified track motion {track_group.track_path} has inconsistent raw surface lengths."
            )
        valid_rows = sample_valid & transition_valid
        raw_speed_mm = np.where(valid_rows, raw_speed_mm, np.nan)
        frame_path_distance_mm = np.where(
            valid_rows,
            frame_path_distance_mm,
            np.nan,
        )

        # Skip if no valid data
        valid_mask = np.isfinite(raw_speed_mm) & np.isfinite(frame_path_distance_mm)
        if not valid_mask.any():
            continue

        # Identify stationary periods
        periods = identify_stationary_periods(
            raw_speed_mm,
            frame_indices,
            max_stationary_speed,
            min_period_frames,
        )

        if not periods:
            continue

        # Collect statistics for this track
        track_stationary_speeds = []
        track_stationary_path_distances = []

        for start_idx, end_idx in periods:
            period_speeds = raw_speed_mm[start_idx:end_idx]
            period_path_distances = frame_path_distance_mm[start_idx:end_idx]

            # Remove any NaN/inf values
            valid_speeds = period_speeds[np.isfinite(period_speeds)]
            valid_path_distances = period_path_distances[np.isfinite(period_path_distances)]

            track_stationary_speeds.extend(valid_speeds)
            track_stationary_path_distances.extend(valid_path_distances)

        if track_stationary_speeds:
            track_stationary_speeds = np.array(track_stationary_speeds)
            track_stationary_path_distances = np.array(track_stationary_path_distances)

            all_stationary_speeds.extend(track_stationary_speeds)
            all_stationary_path_distances.extend(track_stationary_path_distances)

            track_stats.append({
                "track_id": track_id,
                "n_periods": len(periods),
                "total_stationary_frames": len(track_stationary_speeds),
                "mean_speed": np.mean(track_stationary_speeds),
                "median_speed": np.median(track_stationary_speeds),
                "p95_speed": np.percentile(track_stationary_speeds, 95),
                "max_speed": np.max(track_stationary_speeds),
                "mean_path_distance": np.mean(track_stationary_path_distances),
                "median_path_distance": np.median(track_stationary_path_distances),
                "p95_path_distance": np.percentile(track_stationary_path_distances, 95),
                "max_path_distance": np.max(track_stationary_path_distances),
            })

    # Compute aggregate statistics
    if not all_stationary_speeds:
        raise ValueError("No stationary periods found with current thresholds")

    all_stationary_speeds = np.array(all_stationary_speeds)
    all_stationary_path_distances = np.array(all_stationary_path_distances)

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
        "path_distance_mean": np.mean(all_stationary_path_distances),
        "path_distance_median": np.median(all_stationary_path_distances),
        "path_distance_std": np.std(all_stationary_path_distances),
        "path_distance_p95": np.percentile(all_stationary_path_distances, 95),
        "path_distance_max": np.max(all_stationary_path_distances),
        "track_stats": track_stats,
    }

    return statistics


def print_statistics(
    statistics: Dict[str, Any],
    console: Console,
) -> None:
    """Print noise floor statistics in a formatted table.

    Args:
        statistics: Statistics dictionary from compute_noise_statistics
        console: Rich console for output
    """
    console.rule("[bold]Noise Floor Analysis Results[/bold]")

    # Summary statistics
    console.print("\n[bold cyan]Dataset Summary:[/bold cyan]")
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

    # Path-distance statistics table
    disp_table = Table(title="Frame Path-Distance Statistics (mm)")
    disp_table.add_column("Metric", style="cyan")
    disp_table.add_column("Value", justify="right", style="yellow")

    disp_table.add_row("Mean", f"{statistics['path_distance_mean']:.4f}")
    disp_table.add_row("Median", f"{statistics['path_distance_median']:.4f}")
    disp_table.add_row("Std Dev", f"{statistics['path_distance_std']:.4f}")
    disp_table.add_row("95th percentile", f"{statistics['path_distance_p95']:.4f}")
    disp_table.add_row("Maximum", f"{statistics['path_distance_max']:.4f}")

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
        "--track-kinematics-run",
        dest="track_kinematics_run",
        type=str,
        help="Track kinematics run name (defaults to latest offline)",
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
        run_attrs, tracks_dict = load_track_kinematics_run(
            args.zarr_path,
            args.track_kinematics_run,
            console,
        )

        # Display run info
        method = run_attrs.get("method", "unknown")
        fps = run_attrs.get("fps", "unknown")
        physical_outputs_available = run_attrs.get(
            "physical_outputs_available",
            "unknown",
        )

        console.print(f"[cyan]Method:[/cyan] {method}")
        console.print(f"[cyan]FPS:[/cyan] {fps}")
        console.print(
            "[cyan]Verified physical outputs available:[/cyan] "
            f"{physical_outputs_available}"
        )

        # Compute noise floor statistics
        console.print("\n[bold]Analyzing stationary periods...[/bold]")
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
