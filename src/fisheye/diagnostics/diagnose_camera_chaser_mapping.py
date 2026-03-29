#!/usr/bin/env python3
"""Diagnose why camera frames fail to map to chaser positions.

This diagnostic investigates the 51% NaN issue where half of camera frames
don't get valid chaser position data during visualization.

The issue involves understanding the chain:
  camera_frame -> camera_to_metadata_index -> metadata[stimulus_frame_num] -> chaser_states

This script identifies where the chain breaks and why.

Example
-------
# Diagnose a specific run
python -m fisheye.diagnostics.diagnose_camera_chaser_mapping /path/to/archive.zarr

# Diagnose a specific run name
python -m fisheye.diagnostics.diagnose_camera_chaser_mapping /path/to/archive.zarr \\
    --run-name stimulus_20251107_171536

# Show detailed examples of failures
python -m fisheye.diagnostics.diagnose_camera_chaser_mapping /path/to/archive.zarr \\
    --show-examples 10
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np

from fisheye.shared.zarr_helpers import resolve_zarr_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "zarr_path",
        type=Path,
        help="Path to the Palette Zarr archive.",
    )
    parser.add_argument(
        "--run-name",
        help="Stimulus run name (default: latest).",
    )
    parser.add_argument(
        "--chaser-index",
        type=int,
        default=0,
        help="Chaser index to analyze (default: 0).",
    )
    parser.add_argument(
        "--show-examples",
        type=int,
        default=5,
        metavar="N",
        help="Show N examples of failed mappings (default: 5).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.zarr_path.exists():
        raise FileNotFoundError(f"{args.zarr_path} does not exist.")

    try:
        import zarr
    except ModuleNotFoundError:
        raise RuntimeError("zarr is required. Install it and retry.")

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
            import re
            plain = re.sub(r'\[/?[^\]]+\]', '', message)
            print(plain)

    # Open zarr
    root = zarr.open(str(args.zarr_path), mode="r")

    run, run_name = resolve_zarr_run(
        root,
        ("analysis", "stimulus_runs"),
        args.run_name,
        run_label="Stimulus run",
    )

    echo(f"[bold cyan]{'=' * 70}[/bold cyan]")
    echo(f"[bold]Diagnose Camera→Chaser Mapping[/bold]")
    echo(f"[bold cyan]{'=' * 70}[/bold cyan]")
    echo(f"[bold]Zarr:[/bold] {args.zarr_path.name}")
    echo(f"[bold]Run:[/bold] {run_name}")
    echo(f"[bold]Chaser Index:[/bold] {args.chaser_index}")
    echo("")

    # Load data
    echo("[bold cyan]Loading data...[/bold cyan]")

    if "video_metadata" not in run or "frame_metadata" not in run["video_metadata"]:
        raise ValueError("No frame_metadata found in run.")

    if "tracking_data" not in run or "chaser_states" not in run["tracking_data"]:
        raise ValueError("No chaser_states found in run.")

    if "frame_alignment" not in run or "camera_to_metadata_index" not in run["frame_alignment"]:
        raise ValueError("No camera_to_metadata_index found in run.")

    meta_group = run["video_metadata"]["frame_metadata"]
    chaser_group = run["tracking_data"]["chaser_states"]
    alignment_group = run["frame_alignment"]

    # Read metadata
    meta_field_names = meta_group.attrs.get("field_names", [])
    echo(f"  Metadata fields: {', '.join(meta_field_names)}")

    meta_stim_frame = None
    for field in ["stimulus_frame_num", "frame_num", "frame_id"]:
        if field in meta_field_names:
            meta_stim_frame = meta_group[field][:]
            meta_stim_field = field
            break

    if meta_stim_frame is None:
        raise ValueError("No stimulus_frame_num field in metadata.")

    meta_camera_frame = None
    for field in ["triggering_camera_frame_id", "camera_frame_id"]:
        if field in meta_field_names:
            meta_camera_frame = meta_group[field][:]
            meta_cam_field = field
            break

    if meta_camera_frame is None:
        raise ValueError("No camera_frame_id field in metadata.")

    # Read chaser states
    chaser_field_names = chaser_group.attrs.get("field_names", [])
    echo(f"  Chaser fields: {', '.join(chaser_field_names)}")

    chaser_stim_frame = None
    for field in ["stimulus_frame_num", "frame_num", "frame_id"]:
        if field in chaser_field_names:
            chaser_stim_frame = chaser_group[field][:]
            chaser_stim_field = field
            break

    if chaser_stim_frame is None:
        raise ValueError("No stimulus_frame_num field in chaser_states.")

    # Check if chaser has chaser_index field
    if "chaser_index" in chaser_field_names:
        chaser_indices = chaser_group["chaser_index"][:]
        mask = chaser_indices == args.chaser_index
        chaser_stim_frame = chaser_stim_frame[mask]
        echo(f"  Filtered to chaser_index={args.chaser_index}: {np.sum(mask):,} records")
    else:
        echo(f"  No chaser_index field; assuming single chaser")

    # Check if chaser has camera_frame_id
    chaser_camera_frame = None
    chaser_cam_field = None
    for field in ["triggering_camera_frame_id", "camera_frame_id"]:
        if field in chaser_field_names:
            chaser_camera_frame = chaser_group[field][:]
            if "chaser_index" in chaser_field_names:
                chaser_camera_frame = chaser_camera_frame[mask]
            chaser_cam_field = field
            break

    if chaser_camera_frame is not None:
        echo(f"  [green]✓ Chaser has camera_frame_id field[/green]")
    else:
        echo(f"  [yellow]⚠ Chaser does NOT have camera_frame_id field[/yellow]")

    # Read alignment arrays
    camera_to_meta_index = alignment_group["camera_to_metadata_index"][:]
    camera_offset = alignment_group.attrs.get("camera_frame_offset", 0)

    echo("")
    echo("[bold cyan]Data Summary:[/bold cyan]")
    echo(f"[bold cyan]{'-' * 70}[/bold cyan]")
    echo(f"  Metadata records: {len(meta_stim_frame):,}")
    echo(f"    Stimulus frame range: {int(meta_stim_frame.min()):,} to {int(meta_stim_frame.max()):,}")
    echo(f"    Camera frame range: {int(meta_camera_frame.min()):,} to {int(meta_camera_frame.max()):,}")
    echo(f"  Chaser records: {len(chaser_stim_frame):,}")
    echo(f"    Stimulus frame range: {int(chaser_stim_frame.min()):,} to {int(chaser_stim_frame.max()):,}")
    if chaser_camera_frame is not None:
        valid_chaser_cam = chaser_camera_frame[chaser_camera_frame >= 0]
        if len(valid_chaser_cam) > 0:
            echo(f"    Camera frame range: {int(valid_chaser_cam.min()):,} to {int(valid_chaser_cam.max()):,}")
            echo(f"    Valid camera IDs: {len(valid_chaser_cam):,} / {len(chaser_camera_frame):,} ({100*len(valid_chaser_cam)/len(chaser_camera_frame):.1f}%)")
        else:
            echo(f"    [red]No valid camera frame IDs in chaser_states![/red]")
    echo(f"  camera_to_metadata_index length: {len(camera_to_meta_index):,}")
    echo(f"  Camera offset: {camera_offset}")
    echo("")

    # Build lookup maps
    echo("[bold cyan]Building lookup maps...[/bold cyan]")

    # Map stimulus_frame_num -> metadata_index
    stim_to_meta_index = {}
    for idx, stim in enumerate(meta_stim_frame):
        if int(stim) not in stim_to_meta_index:  # First occurrence
            stim_to_meta_index[int(stim)] = idx

    # Map stimulus_frame_num -> camera_frame_id from metadata
    stim_to_camera = {}
    for stim, cam in zip(meta_stim_frame, meta_camera_frame):
        if int(stim) not in stim_to_camera:
            stim_to_camera[int(stim)] = int(cam)

    # Set of stimulus frames that have chaser data
    chaser_stim_set = set(chaser_stim_frame.tolist())

    echo(f"  Stimulus frames with metadata: {len(stim_to_meta_index):,}")
    echo(f"  Stimulus frames with chaser data: {len(chaser_stim_set):,}")
    echo("")

    # Analyze camera→chaser mapping
    echo("[bold cyan]Analyzing camera\u2192chaser mapping...[/bold cyan]")
    echo(f"[bold cyan]{'-' * 70}[/bold cyan]")

    camera_min = int(meta_camera_frame.min())
    camera_max = int(meta_camera_frame.max())
    camera_range = camera_max - camera_min + 1

    echo(f"  Camera frame range: {camera_min:,} to {camera_max:,} ({camera_range:,} frames)")
    echo("")

    # Test each camera frame
    success_count = 0
    fail_no_meta_index = 0
    fail_meta_out_of_bounds = 0
    fail_no_stim_in_chaser = 0
    fail_examples = []

    for camera_frame in range(camera_min, camera_max + 1):
        # Step 1: Get metadata index
        if camera_frame < len(camera_to_meta_index):
            meta_idx = camera_to_meta_index[camera_frame]
        else:
            meta_idx = -1

        if meta_idx < 0:
            fail_no_meta_index += 1
            if len(fail_examples) < args.show_examples:
                fail_examples.append({
                    "camera_frame": camera_frame,
                    "reason": "camera_to_metadata_index has -1 (no mapping)",
                })
            continue

        # Step 2: Check if meta_idx is in bounds
        if meta_idx >= len(meta_stim_frame):
            fail_meta_out_of_bounds += 1
            if len(fail_examples) < args.show_examples:
                fail_examples.append({
                    "camera_frame": camera_frame,
                    "meta_idx": meta_idx,
                    "reason": f"meta_idx {meta_idx} out of bounds (max {len(meta_stim_frame)-1})",
                })
            continue

        # Step 3: Get stimulus frame number
        stim_frame = int(meta_stim_frame[meta_idx])

        # Step 4: Check if chaser has data for this stimulus frame
        if stim_frame not in chaser_stim_set:
            fail_no_stim_in_chaser += 1
            if len(fail_examples) < args.show_examples:
                fail_examples.append({
                    "camera_frame": camera_frame,
                    "meta_idx": meta_idx,
                    "stim_frame": stim_frame,
                    "reason": f"stimulus frame {stim_frame} not in chaser_states",
                })
            continue

        # Success!
        success_count += 1

    total_tested = camera_range
    fail_count = fail_no_meta_index + fail_meta_out_of_bounds + fail_no_stim_in_chaser

    echo(f"[bold]Results:[/bold]")
    echo(f"  Camera frames tested: {total_tested:,}")
    echo(f"  [green]✓ Successful mappings: {success_count:,} ({100*success_count/total_tested:.1f}%)[/green]")
    echo(f"  [red]✗ Failed mappings: {fail_count:,} ({100*fail_count/total_tested:.1f}%)[/red]")
    echo("")

    if fail_count > 0:
        echo("[bold]Failure Breakdown:[/bold]")
        if fail_no_meta_index > 0:
            echo(f"  [yellow]• No metadata index: {fail_no_meta_index:,} ({100*fail_no_meta_index/fail_count:.1f}% of failures)[/yellow]")
        if fail_meta_out_of_bounds > 0:
            echo(f"  [yellow]• Metadata index out of bounds: {fail_meta_out_of_bounds:,} ({100*fail_meta_out_of_bounds/fail_count:.1f}% of failures)[/yellow]")
        if fail_no_stim_in_chaser > 0:
            echo(f"  [yellow]• Stimulus frame not in chaser_states: {fail_no_stim_in_chaser:,} ({100*fail_no_stim_in_chaser/fail_count:.1f}% of failures)[/yellow]")
        echo("")

    if fail_examples:
        echo(f"[bold]Example Failures (first {len(fail_examples)}):[/bold]")
        for i, example in enumerate(fail_examples, 1):
            echo(f"  {i}. Camera frame {example['camera_frame']:,}")
            for key, value in example.items():
                if key != "camera_frame":
                    echo(f"     {key}: {value}")
        echo("")

    # Summary
    echo("[bold cyan]Summary & Diagnosis:[/bold cyan]")
    echo(f"[bold cyan]{'-' * 70}[/bold cyan]")

    if success_count / total_tested > 0.95:
        echo("[green]✓ Camera\u2192chaser mapping is working well (>95% success)[/green]")
    elif fail_no_stim_in_chaser > fail_count * 0.8:
        echo("[yellow]⚠ Primary issue: Chaser states missing for stimulus frames in metadata[/yellow]")
        echo("  Likely causes:")
        echo("    • Chaser interpolation not filling all gaps in metadata")
        echo("    • Mismatch between metadata and chaser stimulus frame numbers")
        echo("  Recommendation: Check chaser interpolation logic")
    elif fail_no_meta_index > fail_count * 0.8:
        echo("[yellow]⚠ Primary issue: camera_to_metadata_index has many -1 entries[/yellow]")
        echo("  Likely causes:")
        echo("    • Metadata missing for many camera frames")
        echo("    • Metadata interpolation not covering all camera frames")
        echo("  Recommendation: Check metadata interpolation logic")
    else:
        echo("[yellow]⚠ Mixed issues detected[/yellow]")
        echo("  Multiple failure modes present. Review examples above.")

    echo("")


if __name__ == "__main__":
    main()
