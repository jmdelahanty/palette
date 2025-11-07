#!/usr/bin/env python3
"""
Plot stimulus frame number vs triggering camera frame ID from h5 file.

This script analyzes the frame_metadata dataset in a stimulus h5 file to verify
the correct mapping between stimulus frames (120 Hz) and camera frames (60 Hz).

Expected behavior with correct implementation:
- Camera frame IDs increment by 1: 353 → 354 → 355 → 356
- Each camera frame appears twice (120 Hz / 60 Hz = 2:1 ratio)
- Stride between first occurrences should be 2 stimulus frames

Optionally plots chaser X/Y positions over triggering_camera_frame_id.

Example
-------
python -m fisheye.diagnostics.plot_frame_id_mapping /path/to/run.h5
python -m fisheye.diagnostics.plot_frame_id_mapping /path/to/run.h5 --plot frame_mapping.png
python -m fisheye.diagnostics.plot_frame_id_mapping /path/to/run.h5 --plot frame_mapping.png --plot-chaser
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np


def _load_h5py():
    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "h5py is required for this diagnostic. Activate the Palette environment "
            "or install h5py and retry."
        ) from exc
    return h5py


def analyze_frame_mapping(
    h5_path: Path,
    dataset_path: str = "/video_metadata/frame_metadata",
    plot_path: Optional[Path] = None,
    plot_chaser: bool = False,
    chaser_dataset_path: str = "/tracking_data/chaser_states",
) -> None:
    """Analyze and optionally plot frame ID mapping from h5 file."""

    h5py = _load_h5py()

    # Try to use rich console for pretty output
    console: Optional["Console"] = None
    try:
        from rich.console import Console
        console = Console()
    except Exception:
        console = None

    def echo(message: str) -> None:
        if console is not None:
            console.print(message)
        else:
            # Strip rich markup for plain print
            import re
            plain = re.sub(r'\[/?[^\]]+\]', '', message)
            print(plain)

    if not h5_path.exists():
        raise FileNotFoundError(f"{h5_path} does not exist.")

    echo(f"[bold]Loading:[/bold] {h5_path}")
    echo(f"[bold]Dataset:[/bold] {dataset_path}")

    # Load the frame metadata
    with h5py.File(h5_path, "r") as handle:
        if dataset_path not in handle:
            raise KeyError(f"Dataset '{dataset_path}' not found in {h5_path}.")

        dataset = handle[dataset_path]
        if not isinstance(dataset, h5py.Dataset):
            raise TypeError(f"Object at '{dataset_path}' is not a dataset.")

        data = dataset[...]

    # Check available fields
    dtype_names = data.dtype.names or ()
    echo(f"[cyan]Available fields:[/cyan] {', '.join(dtype_names)}")

    # Determine field names (handle variations)
    stim_field = None
    camera_field = None

    for field in ['stimulus_frame_num', 'frame_number', 'stimulus_frame_id']:
        if field in dtype_names:
            stim_field = field
            break

    for field in ['triggering_camera_frame_id', 'camera_frame_id', 'triggering_frame_id']:
        if field in dtype_names:
            camera_field = field
            break

    if stim_field is None:
        raise ValueError(
            f"No stimulus frame field found. Available: {dtype_names}"
        )

    if camera_field is None:
        raise ValueError(
            f"No camera frame ID field found. Available: {dtype_names}"
        )

    echo(f"[green]Using stimulus field:[/green] {stim_field}")
    echo(f"[green]Using camera field:[/green] {camera_field}")

    # Extract data
    stimulus_frames = data[stim_field].astype(np.int64)
    camera_frames = data[camera_field].astype(np.int64)

    echo(f"\n[bold]Dataset info:[/bold]")
    echo(f"  Total records: {len(stimulus_frames):,}")
    echo(f"  Stimulus frames: {stimulus_frames.min()} to {stimulus_frames.max()}")
    echo(f"  Camera frames: {camera_frames.min()} to {camera_frames.max()}")

    # Analyze the mapping
    echo(f"\n[bold]Mapping analysis:[/bold]")

    # Find unique camera frames and their first occurrence
    unique_camera_frames = np.unique(camera_frames)
    echo(f"  Unique camera frames: {len(unique_camera_frames):,}")

    # Build mapping: camera_frame_id -> first stimulus_frame_num
    camera_to_first_stim = {}
    for cam_id in unique_camera_frames:
        first_idx = np.where(camera_frames == cam_id)[0][0]
        camera_to_first_stim[cam_id] = stimulus_frames[first_idx]

    # Calculate strides between consecutive camera frames
    sorted_camera_ids = sorted(camera_to_first_stim.keys())
    strides = []
    camera_id_diffs = []

    for i in range(len(sorted_camera_ids) - 1):
        cam_curr = sorted_camera_ids[i]
        cam_next = sorted_camera_ids[i + 1]

        stim_curr = camera_to_first_stim[cam_curr]
        stim_next = camera_to_first_stim[cam_next]

        stride = stim_next - stim_curr
        cam_diff = cam_next - cam_curr

        strides.append(stride)
        camera_id_diffs.append(cam_diff)

    strides = np.array(strides)
    camera_id_diffs = np.array(camera_id_diffs)

    # Analyze stride patterns
    unique_strides = np.unique(strides)
    echo(f"  Unique stride values: {unique_strides.tolist()}")

    for stride_val in unique_strides:
        count = np.sum(strides == stride_val)
        percentage = 100.0 * count / len(strides)
        echo(f"    Stride {stride_val}: {count:,} occurrences ({percentage:.1f}%)")

    # Check camera frame ID increments
    unique_cam_diffs = np.unique(camera_id_diffs)
    echo(f"  Camera frame ID increments: {unique_cam_diffs.tolist()}")

    # Expected vs actual
    echo(f"\n[bold]Diagnosis:[/bold]")

    if 2 in unique_strides and len(unique_strides) == 1:
        echo(f"  [green]✓ CORRECT:[/green] Stride is consistently 2 (as expected for 120Hz/60Hz = 2:1)")
    elif 4 in unique_strides:
        echo(f"  [red]✗ BUG DETECTED:[/red] Stride of 4 found (indicates double-increment bug)")
    else:
        echo(f"  [yellow]⚠ UNEXPECTED:[/yellow] Stride pattern is {unique_strides.tolist()}")

    # Check for duplicates (each camera frame should appear ~2 times for 120/60 ratio)
    counts_per_camera = []
    for cam_id in unique_camera_frames:
        count = np.sum(camera_frames == cam_id)
        counts_per_camera.append(count)

    counts_per_camera = np.array(counts_per_camera)
    avg_count = np.mean(counts_per_camera)
    echo(f"  Average stimulus frames per camera frame: {avg_count:.2f}")

    if 1.8 <= avg_count <= 2.2:
        echo(f"  [green]✓ Expected ~2 frames per camera frame (120Hz/60Hz)[/green]")
    else:
        echo(f"  [yellow]⚠ Unexpected average: {avg_count:.2f}[/yellow]")

    # Show first few mappings as example
    echo(f"\n[bold]First 10 camera frame mappings:[/bold]")
    for i, cam_id in enumerate(sorted_camera_ids[:10]):
        stim = camera_to_first_stim[cam_id]
        count = np.sum(camera_frames == cam_id)
        echo(f"  Camera {cam_id}: first appears at stimulus frame {stim} ({count} total occurrences)")

    # Load chaser positions if requested
    chaser_x = None
    chaser_y = None
    chaser_stim_frames = None
    chaser_camera_frames = None

    if plot_chaser:
        echo(f"\n[bold]Loading chaser positions:[/bold]")
        try:
            with h5py.File(h5_path, "r") as handle:
                if chaser_dataset_path not in handle:
                    echo(f"[yellow]Warning: Dataset '{chaser_dataset_path}' not found[/yellow]")
                else:
                    chaser_data = handle[chaser_dataset_path][...]
                    chaser_dtype_names = chaser_data.dtype.names or ()

                    # Find position fields
                    if 'chaser_pos_x' in chaser_dtype_names and 'chaser_pos_y' in chaser_dtype_names:
                        chaser_x = chaser_data['chaser_pos_x'].astype(np.float64)
                        chaser_y = chaser_data['chaser_pos_y'].astype(np.float64)

                        # Get frame numbers for chaser data
                        if 'stimulus_frame_num' in chaser_dtype_names:
                            chaser_stim_frames = chaser_data['stimulus_frame_num'].astype(np.int64)

                        # Try to get camera frame IDs if available
                        if 'triggering_camera_frame_id' in chaser_dtype_names:
                            chaser_camera_frames = chaser_data['triggering_camera_frame_id'].astype(np.int64)
                        elif 'camera_frame_id' in chaser_dtype_names:
                            chaser_camera_frames = chaser_data['camera_frame_id'].astype(np.int64)
                        else:
                            # Map stimulus frames to camera frames using frame_metadata
                            if chaser_stim_frames is not None:
                                stim_to_camera = dict(zip(stimulus_frames, camera_frames))
                                chaser_camera_frames = np.array([stim_to_camera.get(s, -1) for s in chaser_stim_frames])

                        echo(f"  Loaded {len(chaser_x):,} chaser position records")
                        echo(f"  X range: {np.nanmin(chaser_x):.2f} to {np.nanmax(chaser_x):.2f}")
                        echo(f"  Y range: {np.nanmin(chaser_y):.2f} to {np.nanmax(chaser_y):.2f}")
                    else:
                        echo(f"[yellow]Warning: chaser_pos_x/y fields not found in {chaser_dataset_path}[/yellow]")
        except Exception as e:
            echo(f"[yellow]Warning: Could not load chaser data: {e}[/yellow]")

    # Create plot if requested
    if plot_path:
        _create_plot(
            stimulus_frames,
            camera_frames,
            camera_to_first_stim,
            strides,
            plot_path,
            stim_field,
            camera_field,
            chaser_x,
            chaser_y,
            chaser_stim_frames,
            chaser_camera_frames,
        )
        echo(f"\n[bold]Plot saved:[/bold] {plot_path}")


def _create_plot(
    stimulus_frames: np.ndarray,
    camera_frames: np.ndarray,
    camera_to_first_stim: dict,
    strides: np.ndarray,
    output_path: Path,
    stim_field: str,
    camera_field: str,
    chaser_x: Optional[np.ndarray] = None,
    chaser_y: Optional[np.ndarray] = None,
    chaser_stim_frames: Optional[np.ndarray] = None,
    chaser_camera_frames: Optional[np.ndarray] = None,
) -> None:
    """Create visualization of frame mapping."""
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required for --plot output.") from exc

    # Determine number of subplots based on whether we have chaser data
    has_chaser = chaser_x is not None and chaser_camera_frames is not None
    n_rows = 5 if has_chaser else 3

    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 4 * n_rows))

    # Plot 1: Scatter plot of all mappings
    ax = axes[0]
    ax.scatter(stimulus_frames, camera_frames, s=1, alpha=0.3)
    ax.set_xlabel(f'{stim_field}', fontsize=11)
    ax.set_ylabel(f'{camera_field}', fontsize=11)
    ax.set_title('Camera Frame ID vs Stimulus Frame Number (All Points)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add expected 2:1 slope line
    stim_min, stim_max = stimulus_frames.min(), stimulus_frames.max()
    cam_min, cam_max = camera_frames.min(), camera_frames.max()

    # Expected: camera increases by 1 for every 2 stimulus frames
    # So: camera = camera_min + (stimulus - stim_min) / 2
    expected_cam = cam_min + (stimulus_frames - stim_min) / 2

    # Plot subset of expected line
    subsample = slice(0, len(stimulus_frames), max(1, len(stimulus_frames) // 1000))
    ax.plot(stimulus_frames[subsample], expected_cam[subsample],
            'r-', alpha=0.5, linewidth=2, label='Expected 2:1 ratio')
    ax.legend()

    # Plot 2: First occurrence mapping (cleaner view)
    ax = axes[1]
    sorted_camera_ids = sorted(camera_to_first_stim.keys())
    first_stim_frames = [camera_to_first_stim[c] for c in sorted_camera_ids]

    ax.plot(first_stim_frames, sorted_camera_ids, 'o-', markersize=3, linewidth=1, alpha=0.7)
    ax.set_xlabel(f'{stim_field} (first occurrence)', fontsize=11)
    ax.set_ylabel(f'{camera_field}', fontsize=11)
    ax.set_title('Camera Frame ID vs Stimulus Frame (First Occurrence Only)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Annotate expected vs actual slope
    if len(first_stim_frames) > 1:
        actual_slope = (sorted_camera_ids[-1] - sorted_camera_ids[0]) / (first_stim_frames[-1] - first_stim_frames[0])
        expected_slope = 0.5  # 1 camera frame per 2 stimulus frames
        ax.text(0.05, 0.95, f'Actual slope: {actual_slope:.3f}\nExpected slope: {expected_slope:.3f}',
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    # Plot 3: Stride histogram
    ax = axes[2]
    unique_strides, counts = np.unique(strides, return_counts=True)
    colors = ['green' if s == 2 else 'red' if s == 4 else 'orange' for s in unique_strides]

    bars = ax.bar(unique_strides.astype(str), counts, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Stride (stimulus frames between consecutive camera frames)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Distribution of Strides Between Camera Frame Changes', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Annotate bars with percentages
    total = len(strides)
    for i, (stride, count) in enumerate(zip(unique_strides, counts)):
        percentage = 100.0 * count / total
        ax.text(i, count, f'{count:,}\n({percentage:.1f}%)',
                ha='center', va='bottom', fontsize=9)

    # Add diagnosis text
    if len(unique_strides) == 1 and unique_strides[0] == 2:
        diagnosis = '✓ CORRECT: Consistent stride of 2 (120Hz/60Hz = 2:1)'
        color = 'green'
    elif 4 in unique_strides:
        diagnosis = '✗ BUG DETECTED: Stride of 4 indicates double-increment bug'
        color = 'red'
    else:
        diagnosis = f'⚠ UNEXPECTED: Stride pattern {unique_strides.tolist()}'
        color = 'orange'

    ax.text(0.5, 0.95, diagnosis, transform=ax.transAxes,
            fontsize=10, fontweight='bold', ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))

    # Plot 4 & 5: Chaser positions over triggering_camera_frame_id (if available)
    if has_chaser:
        # Plot 4: Chaser X position
        ax = axes[3]
        valid_mask = ~np.isnan(chaser_x) & (chaser_camera_frames >= 0)
        if np.any(valid_mask):
            ax.plot(chaser_camera_frames[valid_mask], chaser_x[valid_mask],
                   linewidth=0.8, alpha=0.7, color='blue')
            ax.scatter(chaser_camera_frames[valid_mask], chaser_x[valid_mask],
                      s=2, alpha=0.4, color='blue')
        ax.set_xlabel('Triggering Camera Frame ID', fontsize=11)
        ax.set_ylabel('Chaser X Position (pixels)', fontsize=11)
        ax.set_title('Chaser X Position Over Camera Frame ID', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add statistics
        if np.any(valid_mask):
            stats_text = f"Valid points: {np.sum(valid_mask):,}\n"
            stats_text += f"Range: {np.nanmin(chaser_x[valid_mask]):.1f} to {np.nanmax(chaser_x[valid_mask]):.1f}"
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Plot 5: Chaser Y position
        ax = axes[4]
        valid_mask = ~np.isnan(chaser_y) & (chaser_camera_frames >= 0)
        if np.any(valid_mask):
            ax.plot(chaser_camera_frames[valid_mask], chaser_y[valid_mask],
                   linewidth=0.8, alpha=0.7, color='green')
            ax.scatter(chaser_camera_frames[valid_mask], chaser_y[valid_mask],
                      s=2, alpha=0.4, color='green')
        ax.set_xlabel('Triggering Camera Frame ID', fontsize=11)
        ax.set_ylabel('Chaser Y Position (pixels)', fontsize=11)
        ax.set_title('Chaser Y Position Over Camera Frame ID', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add statistics
        if np.any(valid_mask):
            stats_text = f"Valid points: {np.sum(valid_mask):,}\n"
            stats_text += f"Range: {np.nanmin(chaser_y[valid_mask]):.1f} to {np.nanmax(chaser_y[valid_mask]):.1f}"
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "h5_path",
        type=Path,
        help="Path to the stimulus HDF5 file."
    )
    parser.add_argument(
        "--dataset",
        default="/video_metadata/frame_metadata",
        help="Path to frame_metadata dataset (default: /video_metadata/frame_metadata)."
    )
    parser.add_argument(
        "--plot",
        type=Path,
        help="Optional path to save a matplotlib plot."
    )
    parser.add_argument(
        "--plot-chaser",
        action="store_true",
        help="Include chaser position plots (requires --plot)."
    )
    parser.add_argument(
        "--chaser-dataset",
        default="/tracking_data/chaser_states",
        help="Path to chaser_states dataset (default: /tracking_data/chaser_states)."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyze_frame_mapping(
        h5_path=args.h5_path,
        dataset_path=args.dataset,
        plot_path=args.plot,
        plot_chaser=args.plot_chaser,
        chaser_dataset_path=args.chaser_dataset,
    )


if __name__ == "__main__":
    main()
