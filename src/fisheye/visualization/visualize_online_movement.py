#!/usr/bin/env python3
"""Visualize online movement tracking data from stimulus runs.

This script creates diagnostic visualizations for online (H5-imported) target position
data to help identify tracking artifacts, gaps, and movement patterns.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import zarr
from rich.console import Console

from fisheye.analysis.chaser_metrics_loader import load_chaser_metrics


def load_coordinate_transform(root: zarr.Group, stimulus_run: str, console: Console) -> float:
    """Load texture-to-camera coordinate transformation scale factor.

    Args:
        root: Root zarr group
        stimulus_run: Name of stimulus run
        console: Rich console for output

    Returns:
        Scale factor for transforming texture coordinates to camera coordinates
    """
    try:
        analysis_group = root.require_group("analysis")
        stimulus_parent = analysis_group.require_group("stimulus_runs")

        if stimulus_run not in stimulus_parent:
            console.print(f"[yellow]Warning:[/yellow] Stimulus run '{stimulus_run}' not found")
            return 1.0

        stim_group = stimulus_parent[stimulus_run]
        coord_transform_raw = stim_group.attrs.get("coordinate_transform")

        # Parse JSON string to dict if needed
        coord_transform = None
        if isinstance(coord_transform_raw, str):
            try:
                coord_transform = json.loads(coord_transform_raw)
            except json.JSONDecodeError:
                console.print("[yellow]Warning:[/yellow] coordinate_transform is not valid JSON")
                return 1.0
        elif isinstance(coord_transform_raw, dict):
            coord_transform = coord_transform_raw

        if coord_transform and "texture_to_camera_scale" in coord_transform:
            scale = float(coord_transform["texture_to_camera_scale"])
            console.print(f"[cyan]Loaded coordinate transform:[/cyan] texture_to_camera_scale = {scale:.6f}")
            return scale
        else:
            console.print("[yellow]Warning:[/yellow] No texture_to_camera_scale found; using raw positions")
            return 1.0

    except Exception as exc:
        console.print(f"[yellow]Warning:[/yellow] Failed to load coordinate transform: {exc}")
        return 1.0


def visualize_online_movement(
    zarr_path: str,
    stimulus_run: Optional[str] = None,
    chaser_index: int = 0,
    output_path: Optional[str] = None,
    jump_threshold: float = 500.0,
    console: Optional[Console] = None,
) -> None:
    """Create diagnostic visualization for online movement tracking.

    Args:
        zarr_path: Path to zarr archive
        stimulus_run: Stimulus run name (defaults to latest)
        chaser_index: Chaser index (default 0)
        output_path: Optional path to save figure
        jump_threshold: Displacement threshold (px) for flagging large jumps
        console: Rich console for output
    """
    if console is None:
        console = Console()

    console.print(f"\n[bold cyan]Online Movement Visualization[/bold cyan]")
    console.print(f"Zarr: {zarr_path}")

    # Load metrics bundle
    bundle = load_chaser_metrics(
        zarr_path,
        stimulus_run=stimulus_run,
        chaser_index=chaser_index,
    )

    stimulus_run_name = bundle.provenance.get("stimulus_run")
    console.print(f"Stimulus run: [cyan]{stimulus_run_name}[/cyan]")

    # Load coordinate transformation
    root = zarr.open(str(zarr_path), mode="r")
    texture_to_camera_scale = load_coordinate_transform(root, stimulus_run_name, console)

    # Extract online target positions
    target_pos_x_raw = bundle.online.get("target_pos_x")
    target_pos_y_raw = bundle.online.get("target_pos_y")

    if target_pos_x_raw is None or target_pos_y_raw is None:
        console.print("[red]Error:[/red] No target position data in online metrics")
        return

    target_pos_x = np.asarray(target_pos_x_raw, dtype=np.float64) * texture_to_camera_scale
    target_pos_y = np.asarray(target_pos_y_raw, dtype=np.float64) * texture_to_camera_scale
    camera_frames = bundle.camera_frame_ids

    # Identify valid (non-NaN) positions
    valid_mask = np.isfinite(target_pos_x) & np.isfinite(target_pos_y)
    valid_indices = np.where(valid_mask)[0]

    if valid_indices.size == 0:
        console.print("[red]Error:[/red] No valid positions found")
        return

    # Extract valid data
    valid_frames = camera_frames[valid_indices]
    valid_x = target_pos_x[valid_indices]
    valid_y = target_pos_y[valid_indices]

    # Calculate frame-to-frame displacement
    delta_frames = np.diff(valid_frames)
    displacement = np.sqrt(np.diff(valid_x)**2 + np.diff(valid_y)**2)

    # Identify consecutive frames and large jumps
    consecutive = delta_frames == 1
    large_jumps = displacement > jump_threshold
    artifacts = large_jumps | ~consecutive

    # Calculate statistics
    n_total_frames = len(camera_frames)
    n_valid_frames = valid_indices.size
    coverage_pct = (n_valid_frames / n_total_frames) * 100

    n_consecutive = consecutive.sum()
    n_large_jumps = large_jumps.sum()
    n_gaps = (~consecutive).sum()

    # Calculate distance (only consecutive, reasonable frames)
    valid_displacement = consecutive & ~large_jumps & np.isfinite(displacement)
    total_distance_px = displacement[valid_displacement].sum()

    # Get FPS for time conversion
    fps = root.attrs.get("fps", 60.0)
    duration_seconds = n_total_frames / fps

    # Estimate pixel-to-mm conversion if available
    pixel_to_mm = 1.0
    if "analysis" in root and "movement_runs" in root["analysis"]:
        movement_runs = root["analysis/movement_runs"]
        # Try to find pixel_to_mm from any recent run
        for run_type in ["online", "offline"]:
            if run_type in movement_runs:
                for run_name in movement_runs[run_type].group_keys():
                    run_group = movement_runs[run_type][run_name]
                    if "pixel_to_mm" in run_group.attrs:
                        pixel_to_mm = float(run_group.attrs["pixel_to_mm"])
                        break
                if pixel_to_mm != 1.0:
                    break

    total_distance_mm = total_distance_px * pixel_to_mm

    console.print(f"\n[bold]Statistics:[/bold]")
    console.print(f"  Total frames: {n_total_frames}")
    console.print(f"  Valid positions: {n_valid_frames} ({coverage_pct:.1f}%)")
    console.print(f"  Consecutive transitions: {n_consecutive}")
    console.print(f"  Frame gaps: {n_gaps}")
    console.print(f"  Large jumps (>{jump_threshold}px): {n_large_jumps}")
    console.print(f"  Total distance: {total_distance_px:.1f} px ({total_distance_mm:.1f} mm)")
    console.print(f"  Duration: {duration_seconds:.1f} seconds")
    console.print(f"  Coordinate space: camera (scale={texture_to_camera_scale:.6f})")

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Trajectory colored by time
    ax = axes[0, 0]
    scatter = ax.scatter(valid_x, valid_y, c=valid_frames, cmap='viridis', s=2, alpha=0.7)
    ax.set_xlabel('X Position (pixels, camera space)')
    ax.set_ylabel('Y Position (pixels, camera space)')
    ax.set_title('Target Trajectory (colored by frame number)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Camera Frame ID')

    # Add start and end markers
    ax.plot(valid_x[0], valid_y[0], 'go', markersize=10,
            markeredgecolor='white', markeredgewidth=2, label='Start', zorder=5)
    ax.plot(valid_x[-1], valid_y[-1], 'ro', markersize=10,
            markeredgecolor='white', markeredgewidth=2, label='End', zorder=5)
    ax.legend()

    # Plot 2: Position components over time
    ax = axes[0, 1]
    time_seconds = valid_frames / fps
    ax.plot(time_seconds, valid_x, 'b-', alpha=0.7, linewidth=0.5, label='X')
    ax.plot(time_seconds, valid_y, 'r-', alpha=0.7, linewidth=0.5, label='Y')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Position (pixels, camera space)')
    ax.set_title('Position Components Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Frame coverage
    ax = axes[1, 0]
    coverage = np.zeros(n_total_frames)
    coverage[valid_indices] = 1

    ax.imshow([coverage], aspect='auto', cmap='RdYlGn',
              extent=[0, n_total_frames, 0, 1],
              interpolation='nearest', vmin=0, vmax=1)

    # Highlight gaps in valid positions
    if valid_indices.size > 1:
        gap_indices = np.where(np.diff(valid_indices) > 1)[0]
        for idx in gap_indices:
            gap_start = valid_indices[idx]
            gap_end = valid_indices[idx + 1]
            if gap_end - gap_start > 10:
                ax.axvspan(gap_start, gap_end, alpha=0.3, color='blue',
                          ymin=0.2, ymax=0.8)

    ax.set_xlabel('Camera Frame ID')
    ax.set_yticks([])
    ax.set_title('Frame Coverage (Green=detected, Red=missing, Blue=large gaps)')
    ax.set_xlim(0, n_total_frames)

    ax.text(0.02, 0.5, f'Coverage: {coverage_pct:.1f}%',
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Plot 4: Frame-to-frame displacement
    ax = axes[1, 1]

    if displacement.size > 0:
        # Plot all displacements
        time_displacement = valid_frames[1:] / fps
        ax.plot(time_displacement, displacement, 'b-', alpha=0.5, linewidth=0.5,
                label='All transitions')

        # Highlight artifacts
        if artifacts.any():
            artifact_times = time_displacement[artifacts]
            artifact_displacements = displacement[artifacts]
            ax.scatter(artifact_times, artifact_displacements,
                      color='red', s=30, alpha=0.8, zorder=5,
                      label=f'Artifacts ({artifacts.sum()})')

        # Mark large jumps specifically
        if large_jumps.any():
            jump_times = time_displacement[large_jumps]
            jump_displacements = displacement[large_jumps]
            ax.scatter(jump_times, jump_displacements,
                      color='orange', s=50, marker='x', zorder=6,
                      label=f'Large jumps (>{jump_threshold}px)')

        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Displacement (pixels)')
        ax.set_title('Frame-to-Frame Displacement')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add mean displacement line (only for valid consecutive frames)
        if valid_displacement.any():
            mean_disp = displacement[valid_displacement].mean()
            ax.axhline(y=mean_disp, color='g', linestyle='--', alpha=0.5)
            ax.text(0.02, 0.95, f'Mean (valid): {mean_disp:.1f} px',
                   transform=ax.transAxes, fontsize=10, va='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Overall title
    fig.suptitle(f'Online Movement Visualization - {stimulus_run_name}',
                fontsize=14, fontweight='bold')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        console.print(f"[green]Saved plot to {output_path}[/green]")
    else:
        plt.show()

    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Visualize online movement tracking data from stimulus runs"
    )
    parser.add_argument("zarr_path", help="Path to Palette zarr archive")
    parser.add_argument("--stimulus-run", help="Stimulus run name (defaults to latest)")
    parser.add_argument("--chaser-index", type=int, default=0,
                       help="Chaser index to visualize (default: 0)")
    parser.add_argument("--output", help="Path to save figure")
    parser.add_argument("--jump-threshold", type=float, default=500.0,
                       help="Displacement threshold (px) for flagging large jumps (default: 500)")

    args = parser.parse_args(argv)

    console = Console()

    try:
        visualize_online_movement(
            zarr_path=args.zarr_path,
            stimulus_run=args.stimulus_run,
            chaser_index=args.chaser_index,
            output_path=args.output,
            jump_threshold=args.jump_threshold,
            console=console,
        )
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
