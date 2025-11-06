"""
Diagnostic script for examining HDF5 events table and frame alignment.

This script reads the events table from an HDF5 file to understand the relationship
between stimulus frame numbers and camera frame IDs, particularly to diagnose
frame alignment issues.

Usage:
    python -m fisheye.analysis.diagnose_events_h5 /path/to/stimulus.h5

Example:
    python -m fisheye.analysis.diagnose_events_h5 /path/to/stimulus.h5 --target-stim-frame 72003
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import zarr
from rich.console import Console
from rich.table import Table


# Event type enum (from read_h5_data.py)
EXPERIMENT_EVENT_TYPE = {
    0: "PROTOCOL_START",
    1: "PROTOCOL_END",
    2: "PROTOCOL_PAUSE",
    3: "PROTOCOL_RESUME",
    10: "STEP_END",
    11: "STEP_START",
    20: "ACQUISITION_START",
    21: "ACQUISITION_END",
}

# Stimulus mode enum
STIMULUS_MODE_TYPE = {
    0: "BLANK",
    2: "COHERENT_DOTS",
    8: "STATIC_IMAGE",
    12: "CHASER",
}


def read_events_from_h5(h5_path: Path) -> Tuple[np.ndarray, Dict[str, object]]:
    """Read events table from HDF5 file."""
    with h5py.File(h5_path, "r") as h5:
        if "/events" not in h5:
            raise ValueError("No /events dataset found in HDF5 file")

        events = h5["/events"][:]
        attrs = dict(h5["/events"].attrs)

    return events, attrs


def read_frame_metadata_from_h5(h5_path: Path) -> Optional[np.ndarray]:
    """Read frame metadata from HDF5 file if available."""
    with h5py.File(h5_path, "r") as h5:
        if "/video_metadata/frame_metadata" not in h5:
            return None

        frame_metadata = h5["/video_metadata/frame_metadata"][:]

    return frame_metadata


def decode_event_name(events: np.ndarray, idx: int) -> str:
    """Decode event name from name_or_context field."""
    if "name_or_context" in events.dtype.names:
        name = events[idx]["name_or_context"]
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="ignore").rstrip("\x00")
        return str(name)
    return ""


def decode_event_details(events: np.ndarray, idx: int) -> Dict:
    """Decode details_json field."""
    if "details_json" in events.dtype.names:
        details_str = events[idx]["details_json"]
        if isinstance(details_str, bytes):
            details_str = details_str.decode("utf-8", errors="ignore").rstrip("\x00")
        if details_str:
            try:
                return json.loads(details_str)
            except json.JSONDecodeError:
                return {"raw": details_str}
    return {}


def analyze_key_events(events: np.ndarray, console: Console) -> None:
    """Find and display key events (protocol start, training start, etc.)."""
    console.print("\n[bold]Key Events Analysis:[/bold]")

    # Check required fields
    if "event_type_id" not in events.dtype.names:
        console.print("[yellow]No event_type_id field in events table[/yellow]")
        return

    has_stim_frame = "stimulus_frame_num" in events.dtype.names
    has_camera_frame = "camera_frame_id" in events.dtype.names

    if not has_stim_frame and not has_camera_frame:
        console.print("[yellow]No frame number fields in events table[/yellow]")
        return

    # Create table
    table = Table(show_header=True)
    table.add_column("Event", style="cyan")
    table.add_column("Event Type", justify="left")
    if has_stim_frame:
        table.add_column("Stimulus Frame", justify="right")
    if has_camera_frame:
        table.add_column("Camera Frame", justify="right")
    if has_stim_frame and has_camera_frame:
        table.add_column("Ratio (Stim/Cam)", justify="right")
    table.add_column("Details", justify="left")

    # Find key events
    protocol_starts = events["event_type_id"] == 0  # PROTOCOL_START
    step_starts = events["event_type_id"] == 11  # STEP_START

    event_count = 0

    # Show protocol starts
    for idx in np.where(protocol_starts)[0]:
        event_name = decode_event_name(events, idx)
        event_type = EXPERIMENT_EVENT_TYPE.get(int(events[idx]["event_type_id"]), "UNKNOWN")

        row = [f"#{event_count}", event_type]

        stim_frame = None
        camera_frame = None

        if has_stim_frame:
            stim_frame = int(events[idx]["stimulus_frame_num"])
            row.append(str(stim_frame))

        if has_camera_frame:
            camera_frame = int(events[idx]["camera_frame_id"])
            row.append(str(camera_frame))

        if has_stim_frame and has_camera_frame and camera_frame > 0:
            ratio = stim_frame / camera_frame
            row.append(f"{ratio:.4f}")
        elif has_stim_frame and has_camera_frame:
            row.append("-")

        details = decode_event_details(events, idx)
        details_str = event_name if event_name else ""
        if details and "protocol_name" in details:
            details_str = details["protocol_name"]

        row.append(details_str[:50])
        table.add_row(*row)
        event_count += 1

    # Show step starts (training phases)
    for idx in np.where(step_starts)[0][:10]:  # Limit to first 10 steps
        event_name = decode_event_name(events, idx)
        event_type = EXPERIMENT_EVENT_TYPE.get(int(events[idx]["event_type_id"]), "UNKNOWN")

        row = [f"#{event_count}", event_type]

        stim_frame = None
        camera_frame = None

        if has_stim_frame:
            stim_frame = int(events[idx]["stimulus_frame_num"])
            row.append(str(stim_frame))

        if has_camera_frame:
            camera_frame = int(events[idx]["camera_frame_id"])
            row.append(str(camera_frame))

        if has_stim_frame and has_camera_frame and camera_frame > 0:
            ratio = stim_frame / camera_frame
            row.append(f"{ratio:.4f}")
        elif has_stim_frame and has_camera_frame:
            row.append("-")

        details = decode_event_details(events, idx)
        details_str = event_name if event_name else ""
        if details:
            # Try to extract meaningful info
            if "step_name" in details:
                details_str = details["step_name"]
            elif "mode" in details:
                details_str = f"Mode: {details['mode']}"

        row.append(details_str[:50])
        table.add_row(*row)
        event_count += 1

    console.print(table)

    if event_count == 0:
        console.print("[yellow]No PROTOCOL_START or STEP_START events found[/yellow]")


def analyze_frame_rate_ratio(events: np.ndarray, console: Console) -> None:
    """Analyze the ratio between stimulus frames and camera frames."""
    console.print("\n[bold]Frame Rate Ratio Analysis:[/bold]")

    if "stimulus_frame_num" not in events.dtype.names or "camera_frame_id" not in events.dtype.names:
        console.print("[yellow]Cannot analyze ratio: missing frame fields[/yellow]")
        return

    stim_frames = events["stimulus_frame_num"]
    camera_frames = events["camera_frame_id"]

    # Filter to events with valid camera frames
    valid_mask = camera_frames > 0
    stim_valid = stim_frames[valid_mask]
    camera_valid = camera_frames[valid_mask]

    if len(stim_valid) == 0:
        console.print("[yellow]No events with valid camera frame IDs[/yellow]")
        return

    ratios = stim_valid.astype(float) / camera_valid.astype(float)

    console.print(f"  Events with valid camera frames: {len(stim_valid)}")
    console.print(f"  Mean ratio (stim/camera): {ratios.mean():.4f}")
    console.print(f"  Median ratio: {np.median(ratios):.4f}")
    console.print(f"  Min ratio: {ratios.min():.4f}")
    console.print(f"  Max ratio: {ratios.max():.4f}")
    console.print(f"  Std dev: {ratios.std():.4f}")

    # Check if ratio is consistently ~2.0
    if 1.9 < ratios.mean() < 2.1:
        console.print("\n  [green]✓ Ratio is consistently ~2.0 (stimulus runs at 2x camera rate)[/green]")
    else:
        console.print(f"\n  [yellow]⚠ Ratio is {ratios.mean():.4f}, not the expected 2.0[/yellow]")


def investigate_specific_frame(
    events: np.ndarray,
    frame_metadata: Optional[np.ndarray],
    target_stim_frame: int,
    console: Console,
) -> None:
    """Investigate a specific stimulus frame number."""
    console.print(f"\n[bold]Investigating Stimulus Frame {target_stim_frame}:[/bold]")

    if "stimulus_frame_num" not in events.dtype.names:
        console.print("[yellow]No stimulus_frame_num field in events[/yellow]")
        return

    # Find events at or near this frame
    stim_frames = events["stimulus_frame_num"]

    # Find closest events
    diffs = np.abs(stim_frames - target_stim_frame)
    closest_indices = np.argsort(diffs)[:5]

    console.print("\n  Closest events in events table:")
    table = Table(show_header=True)
    table.add_column("Stimulus Frame", justify="right")
    if "camera_frame_id" in events.dtype.names:
        table.add_column("Camera Frame", justify="right")
    table.add_column("Event Type", justify="left")
    table.add_column("Details", justify="left")

    for idx in closest_indices:
        stim = int(events[idx]["stimulus_frame_num"])
        row = [str(stim)]

        if "camera_frame_id" in events.dtype.names:
            cam = int(events[idx]["camera_frame_id"])
            row.append(str(cam))

        event_type = EXPERIMENT_EVENT_TYPE.get(int(events[idx]["event_type_id"]), "UNKNOWN")
        row.append(event_type)

        event_name = decode_event_name(events, idx)
        row.append(event_name[:40])

        style = "bold yellow" if stim == target_stim_frame else ""
        table.add_row(*row, style=style)

    console.print(table)

    # Check frame_metadata if available
    if frame_metadata is not None:
        console.print("\n  Frame metadata lookup:")

        if "stimulus_frame_num" in frame_metadata.dtype.names:
            stim_nums = frame_metadata["stimulus_frame_num"]
            matches = np.where(stim_nums == target_stim_frame)[0]

            if len(matches) > 0:
                idx = matches[0]
                if "triggering_camera_frame_id" in frame_metadata.dtype.names:
                    camera_frame = frame_metadata[idx]["triggering_camera_frame_id"]
                    console.print(f"    Stimulus frame {target_stim_frame} → Camera frame {camera_frame}")
                else:
                    console.print(f"    Found stimulus frame {target_stim_frame} but no camera frame mapping")
            else:
                console.print(f"    [yellow]Stimulus frame {target_stim_frame} not found in frame_metadata[/yellow]")
        else:
            console.print("    [yellow]No stimulus_frame_num field in frame_metadata[/yellow]")


def compare_with_zarr(
    h5_events: np.ndarray,
    zarr_path: Optional[Path],
    console: Console,
) -> None:
    """Compare H5 events with zarr events if zarr file exists."""
    if zarr_path is None or not zarr_path.exists():
        return

    console.print(f"\n[bold]Comparing with Zarr:[/bold] {zarr_path}")

    try:
        root = zarr.open(zarr_path, mode="r")

        # Find stimulus run
        if "analysis" not in root or "stimulus_runs" not in root["analysis"]:
            console.print("[yellow]No stimulus runs in zarr[/yellow]")
            return

        stim_runs = root["analysis/stimulus_runs"]
        run_names = sorted(stim_runs.keys())

        if not run_names:
            console.print("[yellow]No stimulus runs available[/yellow]")
            return

        run_name = run_names[-1]
        console.print(f"  Using stimulus run: {run_name}")

        run_group = stim_runs[run_name]

        if "events" not in run_group:
            console.print("[yellow]No events in zarr stimulus run[/yellow]")
            return

        events_group = run_group["events"]

        # Read columnar events
        zarr_stim_frames = None
        zarr_camera_frames = None

        if "stimulus_frame_num" in events_group:
            zarr_stim_frames = events_group["stimulus_frame_num"][:]

        if "camera_frame_id" in events_group:
            zarr_camera_frames = events_group["camera_frame_id"][:]

        # Compare
        h5_stim = h5_events["stimulus_frame_num"] if "stimulus_frame_num" in h5_events.dtype.names else None
        h5_camera = h5_events["camera_frame_id"] if "camera_frame_id" in h5_events.dtype.names else None

        if h5_stim is not None and zarr_stim_frames is not None:
            if np.array_equal(h5_stim, zarr_stim_frames):
                console.print("  [green]✓ Stimulus frame numbers match between H5 and Zarr[/green]")
            else:
                console.print("  [red]✗ Stimulus frame numbers differ between H5 and Zarr[/red]")
                console.print(f"    H5 range: {h5_stim.min()} - {h5_stim.max()}")
                console.print(f"    Zarr range: {zarr_stim_frames.min()} - {zarr_stim_frames.max()}")

        if h5_camera is not None and zarr_camera_frames is not None:
            if np.array_equal(h5_camera, zarr_camera_frames):
                console.print("  [green]✓ Camera frame IDs match between H5 and Zarr[/green]")
            else:
                console.print("  [red]✗ Camera frame IDs differ between H5 and Zarr[/red]")
                console.print(f"    H5 range: {h5_camera.min()} - {h5_camera.max()}")
                console.print(f"    Zarr range: {zarr_camera_frames.min()} - {zarr_camera_frames.max()}")

    except Exception as e:
        console.print(f"[yellow]Could not compare with zarr: {e}[/yellow]")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose frame alignment by examining HDF5 events table"
    )
    parser.add_argument("h5_path", type=Path, help="Path to HDF5 stimulus file")
    parser.add_argument(
        "--zarr-path",
        type=Path,
        default=None,
        help="Optional path to zarr file for comparison",
    )
    parser.add_argument(
        "--target-stim-frame",
        type=int,
        default=72003,
        help="Specific stimulus frame to investigate (default: 72003)",
    )

    args = parser.parse_args(argv)
    console = Console()

    if not args.h5_path.exists():
        console.print(f"[red]HDF5 file not found: {args.h5_path}[/red]")
        return 1

    console.print(f"[bold]Loading HDF5 file:[/bold] {args.h5_path}")

    # Read events
    try:
        events, attrs = read_events_from_h5(args.h5_path)
        console.print(f"  Loaded {len(events)} events")
        console.print(f"  Fields: {', '.join(events.dtype.names)}")
    except Exception as e:
        console.print(f"[red]Failed to read events: {e}[/red]")
        return 1

    # Read frame metadata
    console.print("\n[bold]Loading frame metadata...[/bold]")
    try:
        frame_metadata = read_frame_metadata_from_h5(args.h5_path)
        if frame_metadata is not None:
            console.print(f"  Loaded {len(frame_metadata)} frame metadata records")
            console.print(f"  Fields: {', '.join(frame_metadata.dtype.names)}")
        else:
            console.print("  [dim]No frame metadata available[/dim]")
    except Exception as e:
        console.print(f"[yellow]Could not read frame metadata: {e}[/yellow]")
        frame_metadata = None

    # Analyze
    analyze_key_events(events, console)
    analyze_frame_rate_ratio(events, console)
    investigate_specific_frame(events, frame_metadata, args.target_stim_frame, console)

    # Compare with zarr if provided
    if args.zarr_path:
        compare_with_zarr(events, args.zarr_path, console)

    console.print("\n[green]Diagnostic complete![/green]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
