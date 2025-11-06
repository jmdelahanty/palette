"""
Script to fix incorrect frame values in PROTOCOL_START events in HDF5 files.

PROTOCOL_START events often contain garbage values from stale shared memory
in the stimulus generation program. This script corrects those values directly
in the source HDF5 file.

The camera recording typically starts before the stimulus protocol begins, so:
- stimulus_frame_num should be 0 (first stimulus frame)
- camera_frame_id should be the actual camera frame when the protocol started
  (e.g., 353 if recording started 353 frames before protocol)

IMPORTANT: This script creates a backup of the H5 file before making any changes.
The backup is saved with a .backup suffix.

Usage:
    python -m fisheye.analysis.fix_protocol_start_frames_h5 /path/to/stimulus.h5 \\
        --protocol-start-camera-frame 353

Example:
    python -m fisheye.analysis.fix_protocol_start_frames_h5 /path/to/stimulus.h5 \\
        --protocol-start-camera-frame 353 --dry-run
"""

import argparse
import shutil
import sys
from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np
from rich.console import Console
from rich.table import Table


# Event type enum
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


def _update_event_fields(events_ds: h5py.Dataset, idx: int, **updates) -> None:
    """Write updated field values back into a compound /events dataset row."""
    record = events_ds[idx]
    for field, value in updates.items():
        record[field] = value
    events_ds[idx] = record


def create_backup(h5_path: Path, console: Console) -> Path:
    """Create a backup of the H5 file."""
    backup_path = h5_path.with_suffix(h5_path.suffix + ".backup")

    # If backup already exists, create numbered backup
    counter = 1
    while backup_path.exists():
        backup_path = h5_path.with_suffix(f"{h5_path.suffix}.backup{counter}")
        counter += 1

    console.print(f"[bold]Creating backup:[/bold] {backup_path.name}")
    shutil.copy2(h5_path, backup_path)
    console.print(f"[green]✓ Backup created[/green]")

    return backup_path


def diagnose_events(
    events: np.ndarray,
    console: Console,
    protocol_start_camera_frame: int,
) -> dict:
    """Diagnose issues with PROTOCOL_START and STEP_START events."""
    console.print("\n[bold]Diagnosing events...[/bold]")

    if "event_type_id" not in events.dtype.names:
        console.print("[red]No event_type_id field in events[/red]")
        return {"has_issues": False}

    event_types = events["event_type_id"]
    stimulus_frames = events["stimulus_frame_num"] if "stimulus_frame_num" in events.dtype.names else None
    camera_frames = events["camera_frame_id"] if "camera_frame_id" in events.dtype.names else None

    if stimulus_frames is None or camera_frames is None:
        console.print("[red]Missing frame fields in events[/red]")
        return {"has_issues": False}

    # Find PROTOCOL_START and STEP_START events
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


def apply_fixes(h5_path: Path, issues: dict, console: Console, dry_run: bool = False) -> bool:
    """Apply fixes to HDF5 events table."""
    if not issues["has_issues"]:
        console.print("\n[green]No fixes needed![/green]")
        return True

    total_fixes = len(issues["protocol_start_fixes"]) + len(issues["step_start_fixes"])
    console.print(f"\n[bold]{'[DRY RUN] Would apply' if dry_run else 'Applying'} {total_fixes} fix(es)...[/bold]")

    if dry_run:
        console.print("\n[yellow]DRY RUN - no changes will be made[/yellow]")
        return True

    try:
        # Open H5 file in read-write mode
        with h5py.File(h5_path, "r+") as h5:
            events = h5["/events"]

            # Fix PROTOCOL_START events
            if issues["protocol_start_fixes"]:
                for fix in issues["protocol_start_fixes"]:
                    idx = fix["index"]
                    console.print(f"  Fixing PROTOCOL_START at index {idx}:")
                    console.print(f"    stimulus_frame_num: {fix['old_stim']} → {fix['new_stim']}")
                    console.print(f"    camera_frame_id: {fix['old_cam']} → {fix['new_cam']}")

                    _update_event_fields(
                        events,
                        idx,
                        stimulus_frame_num=fix["new_stim"],
                        camera_frame_id=fix["new_cam"],
                    )

            # Fix STEP_START events
            if issues["step_start_fixes"]:
                for fix in issues["step_start_fixes"]:
                    idx = fix["index"]
                    console.print(f"  Fixing STEP_START at index {idx}:")
                    console.print(f"    stimulus_frame_num: {fix['old_stim']} → {fix['new_stim']}")

                    # Only show camera_frame_id change if it's actually changing
                    if fix['old_cam'] != fix['new_cam']:
                        console.print(f"    camera_frame_id: {fix['old_cam']} → {fix['new_cam']}")

                    _update_event_fields(
                        events,
                        idx,
                        stimulus_frame_num=fix["new_stim"],
                        camera_frame_id=fix["new_cam"],
                    )

            # Add attribute noting correction was applied
            events.attrs["protocol_start_corrected"] = True
            events.attrs["correction_script"] = "fix_protocol_start_frames_h5.py"

        console.print("\n[green]✓ Fixes applied successfully[/green]")
        return True

    except Exception as e:
        console.print(f"\n[red]✗ Failed to apply fixes: {e}[/red]")
        return False


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fix incorrect frame values in PROTOCOL_START events in HDF5 files"
    )
    parser.add_argument("h5_path", type=Path, help="Path to HDF5 stimulus file")
    parser.add_argument(
        "--protocol-start-camera-frame",
        type=int,
        required=True,
        help="Camera frame ID when protocol started (e.g., 353 if recording started 353 frames before protocol)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be fixed without making changes",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip creating backup file (not recommended)",
    )

    args = parser.parse_args(argv)
    console = Console()

    console.print(f"[bold]Protocol start camera frame:[/bold] {args.protocol_start_camera_frame}")

    # Check file exists
    console.print(f"[bold]Loading HDF5 file:[/bold] {args.h5_path}")

    if not args.h5_path.exists():
        console.print(f"[red]HDF5 file not found: {args.h5_path}[/red]")
        return 1

    # Create backup unless disabled or dry run
    if not args.dry_run and not args.no_backup:
        try:
            backup_path = create_backup(args.h5_path, console)
        except Exception as e:
            console.print(f"[red]Failed to create backup: {e}[/red]")
            console.print("[yellow]Aborting to prevent data loss[/yellow]")
            return 1

    # Read events
    try:
        with h5py.File(args.h5_path, "r") as h5:
            if "/events" not in h5:
                console.print("[red]No /events dataset found in HDF5 file[/red]")
                return 1

            events = h5["/events"][:]
            console.print(f"  Loaded {len(events)} events")
            console.print(f"  Fields: {', '.join(events.dtype.names)}")

    except Exception as e:
        console.print(f"[red]Failed to read events: {e}[/red]")
        return 1

    # Diagnose
    issues = diagnose_events(events, console, args.protocol_start_camera_frame)

    # Apply fixes
    if issues["has_issues"]:
        success = apply_fixes(args.h5_path, issues, console, dry_run=args.dry_run)
        if not success:
            return 1
    else:
        console.print("\n[green]No issues found![/green]")

    # Summary
    console.print(f"\n[bold]Summary:[/bold]")
    total_issues = len(issues.get("protocol_start_fixes", [])) + len(issues.get("step_start_fixes", []))
    console.print(f"  Found {total_issues} issue(s)")
    if args.dry_run:
        console.print(f"  [yellow]Would fix {total_issues} issue(s) (dry run)[/yellow]")
    else:
        if total_issues > 0:
            console.print(f"  [green]Fixed {total_issues} issue(s)[/green]")
            if not args.no_backup:
                console.print(f"  [dim]Backup saved: {backup_path}[/dim]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
