#!/usr/bin/env python3
"""Visualize online tracking data from stimulus runs.

This script creates diagnostic visualizations for online (H5-imported) target
position data to help identify tracking artifacts, gaps, and movement patterns.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console

from fisheye.analysis.chaser_metrics_loader import load_chaser_metrics


@dataclass(frozen=True)
class _VerifiedOnlineTrackingSurface:
    """Copied native online positions plus their still-live authority."""

    positions: np.ndarray
    camera_frame_ids: np.ndarray
    timestamp_ns: np.ndarray
    width_px: int
    height_px: int
    space_id: str
    profile_id: str
    handoff: Any = field(repr=False, compare=False)


def _verified_online_tracking_surface(bundle: Any) -> _VerifiedOnlineTrackingSurface:
    """Resolve only the canonical arena-relative target-position surface.

    This diagnostic deliberately stays in the array's native frame.  A scalar
    resolution ratio cannot stand in for the persisted directed transform to a
    source camera, and an unrelated track run cannot supply physical units.
    """

    handoff = getattr(bundle, "online_coordinate_handoff", None)
    if handoff is None:
        raise ValueError(
            "Online tracking visualization requires a canonical typed "
            "coordinate handoff."
        )
    handoff.assert_verified()
    descriptor = handoff.coordinate_descriptor.descriptor
    extent = descriptor.reference_extent
    width = extent.width
    height = extent.height
    if (
        descriptor.profile_id
        != "arena_relative_canvas_px.top_left_y_down.v1"
        or descriptor.space_id != "arena_relative_canvas_px"
        or descriptor.geometry_type != "point_xy"
        or descriptor.components != ("x", "y")
        or descriptor.component_units != ("px", "px")
        or descriptor.source_camera_overlay.status != "requires_transform"
        or not descriptor.source_camera_overlay.transform_refs
        or extent.units != "px"
        or isinstance(width, bool)
        or isinstance(height, bool)
        or not isinstance(width, (int, float))
        or not isinstance(height, (int, float))
        or not float(width).is_integer()
        or not float(height).is_integer()
        or int(width) <= 0
        or int(height) <= 0
    ):
        raise ValueError(
            "Online target positions do not have the supported canonical "
            "arena-relative canvas point semantics."
        )

    raw_positions = bundle.online.get("target_position_xy")
    if raw_positions is None:
        raise ValueError("Canonical online metrics omit target_position_xy.")
    positions = np.array(raw_positions, dtype=np.float64, copy=True, order="C")
    camera_frame_ids = np.array(
        bundle.camera_frame_ids,
        dtype=np.int64,
        copy=True,
        order="C",
    )
    timestamp_ns = np.array(
        bundle.timestamp_ns,
        dtype=np.int64,
        copy=True,
        order="C",
    )
    row_count = camera_frame_ids.shape[0]
    if (
        camera_frame_ids.ndim != 1
        or timestamp_ns.shape != (row_count,)
        or positions.shape != (row_count, 2)
    ):
        raise ValueError(
            "Canonical online positions, camera-frame identity, and timestamps "
            "must have one exact shared row count."
        )
    if row_count and (
        np.any(np.diff(camera_frame_ids) <= 0)
        or int(camera_frame_ids[0]) < 0
    ):
        raise ValueError(
            "Canonical online camera-frame identifiers must be strictly "
            "increasing nonnegative integers."
        )

    finite = np.isfinite(positions).all(axis=1)
    if np.any(
        (positions[finite, 0] < 0.0)
        | (positions[finite, 0] > float(width))
        | (positions[finite, 1] < 0.0)
        | (positions[finite, 1] > float(height))
    ):
        raise ValueError(
            "Canonical online target positions fall outside their exact "
            "reference extent."
        )

    # Recheck after every value/descriptor copy so a concurrent replacement
    # cannot return a mixed snapshot.
    handoff.assert_verified()
    return _VerifiedOnlineTrackingSurface(
        positions=positions,
        camera_frame_ids=camera_frame_ids,
        timestamp_ns=timestamp_ns,
        width_px=int(width),
        height_px=int(height),
        space_id=str(descriptor.space_id),
        profile_id=str(descriptor.profile_id),
        handoff=handoff,
    )


def _plot_axis(
    surface: _VerifiedOnlineTrackingSurface,
) -> tuple[np.ndarray, str, Optional[float]]:
    """Return a proved time axis, or camera-frame identity when time is absent."""

    timestamps = surface.timestamp_ns
    if (
        timestamps.size
        and np.all(timestamps >= 0)
        and np.all(np.diff(timestamps) >= 0)
    ):
        seconds = (timestamps - timestamps[0]).astype(np.float64) / 1e9
        return seconds, "Elapsed acquisition time (seconds)", float(seconds[-1])
    return (
        surface.camera_frame_ids.astype(np.float64),
        "Camera frame ID",
        None,
    )


def visualize_online_tracking(
    zarr_path: str,
    stimulus_run: Optional[str] = None,
    chaser_index: int = 0,
    output_path: Optional[str] = None,
    jump_threshold: float = 500.0,
    console: Optional[Console] = None,
) -> None:
    """Create diagnostic visualization for online tracking.

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

    console.print("\n[bold cyan]Online Tracking Visualization[/bold cyan]")
    console.print(f"Zarr: {zarr_path}")

    # Load metrics bundle
    bundle = load_chaser_metrics(
        zarr_path,
        stimulus_run=stimulus_run,
        chaser_index=chaser_index,
    )

    stimulus_run_name = bundle.provenance.get("stimulus_run")
    console.print(f"Stimulus run: [cyan]{stimulus_run_name}[/cyan]")

    surface = _verified_online_tracking_surface(bundle)
    target_pos_x = surface.positions[:, 0]
    target_pos_y = surface.positions[:, 1]
    camera_frames = surface.camera_frame_ids
    plot_axis, plot_axis_label, duration_seconds = _plot_axis(surface)

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

    console.print("\n[bold]Statistics:[/bold]")
    console.print(f"  Total frames: {n_total_frames}")
    console.print(f"  Valid positions: {n_valid_frames} ({coverage_pct:.1f}%)")
    console.print(f"  Consecutive transitions: {n_consecutive}")
    console.print(f"  Frame gaps: {n_gaps}")
    console.print(f"  Large jumps (>{jump_threshold}px): {n_large_jumps}")
    console.print(f"  Total distance: {total_distance_px:.1f} px")
    if duration_seconds is not None:
        console.print(f"  Duration: {duration_seconds:.3f} seconds")
    else:
        console.print("  Duration: unavailable (no complete timestamp authority)")
    console.print(
        "  Coordinate space: "
        f"{surface.space_id} ({surface.width_px}x{surface.height_px}px)"
    )

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Trajectory colored by time
    ax = axes[0, 0]
    scatter = ax.scatter(valid_x, valid_y, c=valid_frames, cmap='viridis', s=2, alpha=0.7)
    ax.set_xlabel('X position (px, arena-relative canvas)')
    ax.set_ylabel('Y position (px, arena-relative canvas)')
    ax.set_title('Target Trajectory (colored by frame number)')
    ax.set_aspect('equal')
    ax.set_xlim(0, surface.width_px)
    ax.set_ylim(surface.height_px, 0)
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
    valid_plot_axis = plot_axis[valid_indices]
    ax.plot(valid_plot_axis, valid_x, 'b-', alpha=0.7, linewidth=0.5, label='X')
    ax.plot(valid_plot_axis, valid_y, 'r-', alpha=0.7, linewidth=0.5, label='Y')
    ax.set_xlabel(plot_axis_label)
    ax.set_ylabel('Position (px, arena-relative canvas)')
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

    ax.set_xlabel('Aligned camera-metadata row')
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
        time_displacement = valid_plot_axis[1:]
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

        ax.set_xlabel(plot_axis_label)
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
    fig.suptitle(f'Online Tracking Visualization - {stimulus_run_name}',
                fontsize=14, fontweight='bold')

    plt.tight_layout()

    surface.handoff.assert_verified()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        console.print(f"[green]Saved plot to {output_path}[/green]")
    else:
        plt.show()

    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Visualize online tracking data from stimulus runs"
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
        visualize_online_tracking(
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
