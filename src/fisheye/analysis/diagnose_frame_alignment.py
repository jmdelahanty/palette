"""
Diagnostic script for investigating frame alignment issues between stimulus and camera frames.

This script analyzes the mapping between stimulus frame numbers and camera frame IDs
to identify gaps, offsets, or interpolation issues that could cause misalignment.

Usage:
    python -m fisheye.analysis.diagnose_frame_alignment /path/to/archive.zarr --stimulus-run <run_name>

Example:
    python -m fisheye.analysis.diagnose_frame_alignment data/capture.zarr --stimulus-run latest
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import zarr
from rich.console import Console
from rich.table import Table

from fisheye.shared.zarr.columnar import load_structured_dataset


def load_frame_metadata(
    run_group: zarr.Group,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Load frame_metadata from stimulus run."""
    meta_group = run_group["video_metadata"]
    frame_metadata, meta_attrs = load_structured_dataset(meta_group, "frame_metadata")
    return frame_metadata, meta_attrs


def load_chaser_states(
    run_group: zarr.Group,
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, object]]]:
    """Load chaser_states from stimulus run if available."""
    if "tracking_data" not in run_group:
        return None, None

    track_group = run_group["tracking_data"]
    if "chaser_states" not in track_group:
        return None, None

    chaser_states, chaser_attrs = load_structured_dataset(track_group, "chaser_states")
    return chaser_states, chaser_attrs


def build_stim_to_camera_mapping(
    frame_metadata: np.ndarray,
) -> Dict[int, int]:
    """Build stimulus frame → camera frame mapping."""
    stim_to_camera: Dict[int, int] = {}

    # Handle field name variations
    stim_field = None
    camera_field = None

    for field in ["stimulus_frame_num", "frame_number"]:
        if field in frame_metadata.dtype.names:
            stim_field = field
            break

    for field in ["triggering_camera_frame_id", "camera_frame_id"]:
        if field in frame_metadata.dtype.names:
            camera_field = field
            break

    if stim_field is None or camera_field is None:
        raise ValueError(
            f"Could not find required fields. Available: {frame_metadata.dtype.names}"
        )

    for record in frame_metadata:
        camera_id = int(record[camera_field])
        stimulus_id = int(record[stim_field])
        stim_to_camera[stimulus_id] = camera_id

    return stim_to_camera


def analyze_mapping_gaps(
    stim_to_camera: Dict[int, int],
    console: Console,
) -> None:
    """Analyze gaps and inconsistencies in stimulus-to-camera mapping."""
    console.print("\n[bold]Mapping Analysis:[/bold]")

    stim_frames = sorted(stim_to_camera.keys())
    if not stim_frames:
        console.print("[yellow]No stimulus frames found in mapping[/yellow]")
        return

    # Basic statistics
    console.print(f"  Total mapped frames: {len(stim_frames)}")
    console.print(f"  Stimulus frame range: {stim_frames[0]} → {stim_frames[-1]}")

    camera_frames = [stim_to_camera[s] for s in stim_frames]
    console.print(f"  Camera frame range: {min(camera_frames)} → {max(camera_frames)}")

    # Check for gaps in stimulus frames
    stim_gaps = []
    for i in range(len(stim_frames) - 1):
        gap = stim_frames[i + 1] - stim_frames[i]
        if gap > 1:
            stim_gaps.append((stim_frames[i], stim_frames[i + 1], gap - 1))

    if stim_gaps:
        console.print(f"\n[yellow]Found {len(stim_gaps)} gaps in stimulus frames:[/yellow]")
        for start, end, size in stim_gaps[:5]:  # Show first 5
            console.print(f"    Gap of {size} frames: {start} → {end}")
        if len(stim_gaps) > 5:
            console.print(f"    ... and {len(stim_gaps) - 5} more gaps")
    else:
        console.print("  [green]No gaps in stimulus frame sequence[/green]")

    # Check for duplicates (multiple stimulus frames → same camera frame)
    camera_to_stim: Dict[int, List[int]] = {}
    for stim, cam in stim_to_camera.items():
        if cam not in camera_to_stim:
            camera_to_stim[cam] = []
        camera_to_stim[cam].append(stim)

    duplicates = {cam: stims for cam, stims in camera_to_stim.items() if len(stims) > 1}
    if duplicates:
        console.print(f"\n[yellow]Found {len(duplicates)} camera frames with multiple stimulus mappings:[/yellow]")
        for cam, stims in list(duplicates.items())[:5]:  # Show first 5
            console.print(f"    Camera frame {cam} ← stimulus frames {stims}")
        if len(duplicates) > 5:
            console.print(f"    ... and {len(duplicates) - 5} more duplicates")
    else:
        console.print("  [green]No duplicate camera frame mappings[/green]")


def investigate_specific_offset(
    stim_to_camera: Dict[int, int],
    chaser_states: Optional[np.ndarray],
    target_stim_frame: int,
    expected_camera_frame: int,
    console: Console,
) -> None:
    """Investigate a specific frame offset issue."""
    console.print(f"\n[bold]Investigating stimulus frame {target_stim_frame}:[/bold]")

    # Check the mapping
    if target_stim_frame in stim_to_camera:
        actual_camera_frame = stim_to_camera[target_stim_frame]
        console.print(f"  Mapped to camera frame: {actual_camera_frame}")
        console.print(f"  Expected camera frame: {expected_camera_frame}")

        offset = actual_camera_frame - expected_camera_frame
        if offset != 0:
            console.print(f"  [red]Offset: {offset} frames[/red]")
        else:
            console.print("  [green]No offset detected[/green]")
    else:
        console.print(f"  [yellow]Stimulus frame {target_stim_frame} not found in mapping[/yellow]")

    # Show context around target frame
    console.print(f"\n[bold]Mapping context around stimulus frame {target_stim_frame}:[/bold]")

    table = Table(show_header=True)
    table.add_column("Stimulus Frame", justify="right")
    table.add_column("Camera Frame", justify="right")
    table.add_column("Delta (stim)", justify="right")
    table.add_column("Delta (camera)", justify="right")

    # Find frames around target
    stim_frames = sorted(stim_to_camera.keys())
    try:
        target_idx = stim_frames.index(target_stim_frame)
        start_idx = max(0, target_idx - 5)
        end_idx = min(len(stim_frames), target_idx + 6)

        for i in range(start_idx, end_idx):
            stim = stim_frames[i]
            cam = stim_to_camera[stim]

            # Calculate deltas
            if i > start_idx:
                prev_stim = stim_frames[i - 1]
                prev_cam = stim_to_camera[prev_stim]
                delta_stim = stim - prev_stim
                delta_cam = cam - prev_cam
            else:
                delta_stim = "-"
                delta_cam = "-"

            # Highlight target row
            style = "bold yellow" if stim == target_stim_frame else ""
            table.add_row(
                str(stim),
                str(cam),
                str(delta_stim),
                str(delta_cam),
                style=style
            )

        console.print(table)
    except ValueError:
        console.print(f"  [yellow]Could not find stimulus frame {target_stim_frame} in sorted list[/yellow]")

    # Check chaser states if available
    if chaser_states is not None:
        console.print(f"\n[bold]Chaser states analysis:[/bold]")

        # Determine which field contains frame numbers
        stim_field = None
        for field in ["stimulus_frame_num", "frame_number"]:
            if field in chaser_states.dtype.names:
                stim_field = field
                break

        if stim_field:
            chaser_frames = chaser_states[stim_field]
            console.print(f"  Total chaser state records: {len(chaser_states)}")
            console.print(f"  Chaser frame range: {chaser_frames.min()} → {chaser_frames.max()}")

            # Find when chaser positions start changing
            if "chaser_pos_x" in chaser_states.dtype.names and "chaser_pos_y" in chaser_states.dtype.names:
                x_pos = chaser_states["chaser_pos_x"]
                y_pos = chaser_states["chaser_pos_y"]

                # Find first non-zero or changing position
                non_zero = (x_pos != 0) | (y_pos != 0)
                if non_zero.any():
                    first_active_idx = np.argmax(non_zero)
                    first_active_stim = chaser_frames[first_active_idx]

                    if first_active_stim in stim_to_camera:
                        first_active_camera = stim_to_camera[first_active_stim]
                        console.print(f"  First non-zero chaser position:")
                        console.print(f"    Stimulus frame: {first_active_stim}")
                        console.print(f"    Camera frame: {first_active_camera}")
                        console.print(f"    Position: ({x_pos[first_active_idx]:.2f}, {y_pos[first_active_idx]:.2f})")
                    else:
                        console.print(f"  First non-zero chaser at stimulus frame {first_active_stim} (not in mapping)")
                else:
                    console.print("  [yellow]All chaser positions are zero[/yellow]")
        else:
            console.print("  [yellow]Could not find frame number field in chaser_states[/yellow]")


def check_interpolation(
    run_group: zarr.Group,
    frame_metadata: np.ndarray,
    console: Console,
) -> None:
    """Check interpolation status and masks."""
    console.print("\n[bold]Interpolation Analysis:[/bold]")

    # Check metadata attributes
    meta_group = run_group["video_metadata/frame_metadata"]
    if isinstance(meta_group, zarr.Group):
        attrs = meta_group.attrs
        interpolated = attrs.get("interpolated", False)
        original_records = attrs.get("original_records", len(frame_metadata))
        total_records = attrs.get("total_records", len(frame_metadata))

        console.print(f"  Interpolated: {interpolated}")
        console.print(f"  Original records: {original_records}")
        console.print(f"  Total records: {total_records}")

        if interpolated:
            interpolated_count = total_records - original_records
            console.print(f"  [yellow]Interpolated {interpolated_count} frames ({100 * interpolated_count / total_records:.1f}%)[/yellow]")

    # Check interpolation mask if available
    if "interpolation_mask" in run_group:
        mask = run_group["interpolation_mask"][:]
        original_count = np.sum(mask)
        interpolated_count = len(mask) - original_count

        console.print(f"\n  Interpolation mask:")
        console.print(f"    Original frames: {original_count}")
        console.print(f"    Interpolated frames: {interpolated_count}")
        console.print(f"    Total: {len(mask)}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose frame alignment issues between stimulus and camera frames"
    )
    parser.add_argument("zarr_path", type=Path, help="Path to zarr archive")
    parser.add_argument(
        "--stimulus-run",
        type=str,
        default="latest",
        help="Stimulus run name (or 'latest', default: latest)",
    )
    parser.add_argument(
        "--target-stim-frame",
        type=int,
        default=72003,
        help="Target stimulus frame to investigate (default: 72003)",
    )
    parser.add_argument(
        "--expected-camera-frame",
        type=int,
        default=353,
        help="Expected camera frame for target stimulus frame (default: 353)",
    )

    args = parser.parse_args(argv)
    console = Console()

    # Open zarr
    console.print(f"[bold]Loading zarr archive:[/bold] {args.zarr_path}")
    root = zarr.open(args.zarr_path, mode="r")

    # Resolve stimulus run
    if "analysis" not in root or "stimulus_runs" not in root["analysis"]:
        console.print("[red]No stimulus runs found in zarr[/red]")
        return 1

    stim_runs = root["analysis/stimulus_runs"]

    if args.stimulus_run == "latest":
        # Find most recent run
        run_names = sorted(stim_runs.keys())
        if not run_names:
            console.print("[red]No stimulus runs available[/red]")
            return 1
        run_name = run_names[-1]
        console.print(f"[dim]Using latest stimulus run: {run_name}[/dim]")
    else:
        run_name = args.stimulus_run
        if run_name not in stim_runs:
            console.print(f"[red]Stimulus run '{run_name}' not found[/red]")
            console.print(f"Available runs: {', '.join(stim_runs.keys())}")
            return 1

    run_group = stim_runs[run_name]

    # Load data
    console.print("\n[bold]Loading frame metadata...[/bold]")
    try:
        frame_metadata, meta_attrs = load_frame_metadata(run_group)
        console.print(f"  Loaded {len(frame_metadata)} frame metadata records")
        console.print(f"  Fields: {', '.join(frame_metadata.dtype.names)}")
    except Exception as e:
        console.print(f"[red]Failed to load frame metadata: {e}[/red]")
        return 1

    console.print("\n[bold]Loading chaser states...[/bold]")
    chaser_states, chaser_attrs = load_chaser_states(run_group)
    if chaser_states is not None:
        console.print(f"  Loaded {len(chaser_states)} chaser state records")
        console.print(f"  Fields: {', '.join(chaser_states.dtype.names)}")
    else:
        console.print("  [dim]No chaser states available[/dim]")

    # Build mapping
    console.print("\n[bold]Building stimulus → camera mapping...[/bold]")
    try:
        stim_to_camera = build_stim_to_camera_mapping(frame_metadata)
        console.print(f"  Built mapping for {len(stim_to_camera)} frames")
    except Exception as e:
        console.print(f"[red]Failed to build mapping: {e}[/red]")
        return 1

    # Analyze
    analyze_mapping_gaps(stim_to_camera, console)
    check_interpolation(run_group, frame_metadata, console)
    investigate_specific_offset(
        stim_to_camera,
        chaser_states,
        args.target_stim_frame,
        args.expected_camera_frame,
        console,
    )

    console.print("\n[green]Diagnostic complete![/green]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
