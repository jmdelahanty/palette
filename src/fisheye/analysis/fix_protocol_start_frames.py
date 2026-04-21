"""
Script to fix incorrect frame values in PROTOCOL_START events.

PROTOCOL_START events often contain garbage values from stale shared memory
in the stimulus generation program. This script corrects those values.

The camera recording typically starts before the stimulus protocol begins, so:
- stimulus_frame_num should be 0 (first stimulus frame)
- camera_frame_id should be the actual camera frame when the protocol started
  (e.g., 353 if recording started 353 frames before protocol)

Usage:
    python -m fisheye.analysis.fix_protocol_start_frames /path/to/archive.zarr \\
        --protocol-start-camera-frame 353 --stimulus-run <run_name>

Example:
    python -m fisheye.analysis.fix_protocol_start_frames data/capture.zarr \\
        --protocol-start-camera-frame 353 --stimulus-run latest
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import zarr
from rich.console import Console
from rich.table import Table


# Legacy H5 event type enum — differs from the modern Citrus encoding.
# See ``fisheye.shared.citrus_enums`` for the canonical modern enum.
# This repair tool operates on H5-origin event_type IDs so the legacy
# mapping is intentionally preserved here.
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


def load_events_from_zarr(run_group: zarr.Group) -> Optional[zarr.Group]:
    """Load events group from stimulus run."""
    if "events" not in run_group:
        return None
    return run_group["events"]


def diagnose_events(
    events_group: zarr.Group,
    console: Console,
    protocol_start_camera_frame: int,
) -> dict:
    """Diagnose issues with PROTOCOL_START and STEP_START events."""
    console.print("\n[bold]Diagnosing events...[/bold]")

    if "event_type_id" not in events_group:
        console.print("[red]No event_type_id field in events[/red]")
        return {"has_issues": False}

    event_types = events_group["event_type_id"][:]
    stimulus_frames = events_group["stimulus_frame_num"][:] if "stimulus_frame_num" in events_group else None
    camera_frames = events_group["camera_frame_id"][:] if "camera_frame_id" in events_group else None

    if stimulus_frames is None or camera_frames is None:
        console.print("[red]Missing frame fields in events[/red]")
        return {"has_issues": False}

    # Find PROTOCOL_START events
    protocol_start_indices = np.where(event_types == 0)[0]
    step_start_indices = np.where(event_types == 11)[0]

    issues = {
        "has_issues": False,
        "protocol_start_fixes": [],
        "step_start_fixes": [],
    }

    # Check PROTOCOL_START events
    if len(protocol_start_indices) > 0:
        console.print(f"\n  Found {len(protocol_start_indices)} PROTOCOL_START event(s):")

        table = Table(show_header=True)
        table.add_column("Index", justify="right")
        table.add_column("Stimulus Frame", justify="right")
        table.add_column("Camera Frame", justify="right")
        table.add_column("Status", justify="left")

        for idx in protocol_start_indices:
            stim = int(stimulus_frames[idx])
            cam = int(camera_frames[idx])

            # PROTOCOL_START should have stim=0 and cam=protocol_start_camera_frame
            needs_fix = (stim != 0) or (cam != protocol_start_camera_frame)
            status = "[red]NEEDS FIX[/red]" if needs_fix else "[green]OK[/green]"

            # Build status details showing what will change
            if needs_fix:
                details = []
                if stim != 0:
                    details.append(f"stim: {stim}→0")
                if cam != protocol_start_camera_frame:
                    details.append(f"cam: {cam}→{protocol_start_camera_frame}")
                status_text = f"{status} ({', '.join(details)})" if details else status
            else:
                status_text = status

            table.add_row(str(idx), str(stim), str(cam), status_text)

            if needs_fix:
                issues["has_issues"] = True
                issues["protocol_start_fixes"].append({
                    "index": int(idx),
                    "old_stim": stim,
                    "old_cam": cam,
                    "new_stim": 0,
                    "new_cam": protocol_start_camera_frame,
                })

        console.print(table)

    # Check STEP_START events
    if len(step_start_indices) > 0:
        console.print(f"\n  Found {len(step_start_indices)} STEP_START event(s):")

        table = Table(show_header=True)
        table.add_column("Index", justify="right")
        table.add_column("Stimulus Frame", justify="right")
        table.add_column("Camera Frame", justify="right")
        table.add_column("Status", justify="left")

        for i, idx in enumerate(step_start_indices[:10]):  # Show first 10
            stim = int(stimulus_frames[idx])
            cam = int(camera_frames[idx])

            # First STEP_START should match PROTOCOL_START timing
            # Subsequent steps should have correct camera_frame_id already
            is_first_step = (i == 0)

            if is_first_step:
                # First step: should have stim=0 and cam=protocol_start_camera_frame
                needs_fix = (stim != 0) or (cam != protocol_start_camera_frame)
                new_cam = protocol_start_camera_frame
            else:
                # Subsequent steps: only fix stim if wrong, keep camera frame as-is
                needs_fix = (stim != 0)
                new_cam = cam

            status = "[red]NEEDS FIX[/red]" if needs_fix else "[green]OK[/green]"

            # Build status details showing what will change
            if needs_fix:
                details = []
                if stim != 0:
                    details.append(f"stim: {stim}→0")
                if is_first_step and cam != protocol_start_camera_frame:
                    details.append(f"cam: {cam}→{protocol_start_camera_frame}")
                status_text = f"{status} ({', '.join(details)})" if details else status
            else:
                status_text = status

            table.add_row(str(idx), str(stim), str(cam), status_text)

            if needs_fix:
                issues["has_issues"] = True
                issues["step_start_fixes"].append({
                    "index": int(idx),
                    "old_stim": stim,
                    "old_cam": cam,
                    "new_stim": 0,  # STEP_START resets stimulus frame to 0
                    "new_cam": new_cam,
                })

        if len(step_start_indices) > 10:
            console.print(f"  [dim]... and {len(step_start_indices) - 10} more[/dim]")

        console.print(table)

    return issues


def apply_fixes(events_group: zarr.Group, issues: dict, console: Console, dry_run: bool = False) -> bool:
    """Apply fixes to events."""
    if not issues["has_issues"]:
        console.print("\n[green]No fixes needed![/green]")
        return True

    total_fixes = len(issues["protocol_start_fixes"]) + len(issues["step_start_fixes"])
    console.print(f"\n[bold]{'[DRY RUN] Would apply' if dry_run else 'Applying'} {total_fixes} fix(es)...[/bold]")

    if dry_run:
        console.print("\n[yellow]DRY RUN - no changes will be made[/yellow]")
        return True

    try:
        # Fix PROTOCOL_START events
        if issues["protocol_start_fixes"]:
            stimulus_frames = events_group["stimulus_frame_num"]
            camera_frames = events_group["camera_frame_id"]

            for fix in issues["protocol_start_fixes"]:
                idx = fix["index"]
                console.print(f"  Fixing PROTOCOL_START at index {idx}:")
                console.print(f"    stimulus_frame_num: {fix['old_stim']} → {fix['new_stim']}")
                console.print(f"    camera_frame_id: {fix['old_cam']} → {fix['new_cam']}")

                stimulus_frames[idx] = fix["new_stim"]
                camera_frames[idx] = fix["new_cam"]

        # Fix STEP_START events
        if issues["step_start_fixes"]:
            stimulus_frames = events_group["stimulus_frame_num"]
            camera_frames = events_group["camera_frame_id"]

            for fix in issues["step_start_fixes"]:
                idx = fix["index"]
                console.print(f"  Fixing STEP_START at index {idx}:")
                console.print(f"    stimulus_frame_num: {fix['old_stim']} → {fix['new_stim']}")

                # Only show camera_frame_id change if it's actually changing
                if fix['old_cam'] != fix['new_cam']:
                    console.print(f"    camera_frame_id: {fix['old_cam']} → {fix['new_cam']}")

                stimulus_frames[idx] = fix["new_stim"]
                camera_frames[idx] = fix["new_cam"]

        # Add metadata noting correction was applied
        if "attrs" in dir(events_group):
            events_group.attrs["protocol_start_corrected"] = True
            events_group.attrs["correction_script"] = "fix_protocol_start_frames.py"

        console.print("\n[green]✓ Fixes applied successfully[/green]")
        return True

    except Exception as e:
        console.print(f"\n[red]✗ Failed to apply fixes: {e}[/red]")
        return False


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fix incorrect frame values in PROTOCOL_START events"
    )
    parser.add_argument("zarr_path", type=Path, help="Path to zarr archive")
    parser.add_argument(
        "--protocol-start-camera-frame",
        type=int,
        required=True,
        help="Camera frame ID when protocol started (e.g., 353 if recording started 353 frames before protocol)",
    )
    parser.add_argument(
        "--stimulus-run",
        type=str,
        default="latest",
        help="Stimulus run name (or 'latest', default: latest)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be fixed without making changes",
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Apply fixes to all stimulus runs",
    )

    args = parser.parse_args(argv)
    console = Console()

    console.print(f"[bold]Protocol start camera frame:[/bold] {args.protocol_start_camera_frame}")

    # Open zarr
    console.print(f"[bold]Loading zarr archive:[/bold] {args.zarr_path}")

    if not args.zarr_path.exists():
        console.print(f"[red]Zarr file not found: {args.zarr_path}[/red]")
        return 1

    root = zarr.open(args.zarr_path, mode="r+" if not args.dry_run else "r")

    # Resolve stimulus run(s)
    if "analysis" not in root or "stimulus_runs" not in root["analysis"]:
        console.print("[red]No stimulus runs found in zarr[/red]")
        return 1

    stim_runs = root["analysis/stimulus_runs"]

    # Determine which runs to process
    if args.all_runs:
        run_names = sorted(stim_runs.keys())
        console.print(f"[bold]Processing all {len(run_names)} stimulus run(s)[/bold]")
    else:
        if args.stimulus_run == "latest":
            run_names = sorted(stim_runs.keys())
            if not run_names:
                console.print("[red]No stimulus runs available[/red]")
                return 1
            run_names = [run_names[-1]]
            console.print(f"[dim]Using latest stimulus run: {run_names[0]}[/dim]")
        else:
            run_name = args.stimulus_run
            if run_name not in stim_runs:
                console.print(f"[red]Stimulus run '{run_name}' not found[/red]")
                console.print(f"Available runs: {', '.join(stim_runs.keys())}")
                return 1
            run_names = [run_name]

    # Process each run
    total_issues = 0
    total_fixed = 0

    for run_name in run_names:
        console.print(f"\n[bold cyan]Processing run: {run_name}[/bold cyan]")
        run_group = stim_runs[run_name]

        # Load events
        events_group = load_events_from_zarr(run_group)
        if events_group is None:
            console.print("[yellow]No events found in this run[/yellow]")
            continue

        # Diagnose
        issues = diagnose_events(events_group, console, args.protocol_start_camera_frame)

        if issues["has_issues"]:
            total_issues += len(issues["protocol_start_fixes"]) + len(issues["step_start_fixes"])

            # Apply fixes
            if apply_fixes(events_group, issues, console, dry_run=args.dry_run):
                total_fixed += len(issues["protocol_start_fixes"]) + len(issues["step_start_fixes"])

    # Summary
    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  Processed {len(run_names)} run(s)")
    console.print(f"  Found {total_issues} issue(s)")
    if args.dry_run:
        console.print(f"  [yellow]Would fix {total_fixed} issue(s) (dry run)[/yellow]")
    else:
        console.print(f"  [green]Fixed {total_fixed} issue(s)[/green]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
