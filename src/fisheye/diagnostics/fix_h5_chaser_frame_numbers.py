#!/usr/bin/env python3
"""Fix chaser state frame numbering in H5 stimulus files.

This script corrects the issue where stimulus_frame_num increments by 2 instead of 1,
causing frame numbers like [1, 3, 5, 7...] instead of [0, 1, 2, 3...].

The root cause is a double-incrementing frame counter in the stimulus logging code.
This script renumbers the frames sequentially without data loss.

Safety features:
- Creates a backup (.h5.bak) before modification
- Dry-run mode to preview changes
- Validation of frame numbering after fix
- Only modifies chaser_states, leaves events/metadata untouched

Example
-------
# Dry run (preview only)
python -m fisheye.diagnostics.fix_h5_chaser_frame_numbers /path/to/stimulus.h5 --dry-run

# Apply fix (creates backup automatically)
python -m fisheye.diagnostics.fix_h5_chaser_frame_numbers /path/to/stimulus.h5

# Apply fix with custom backup location
python -m fisheye.diagnostics.fix_h5_chaser_frame_numbers /path/to/stimulus.h5 \\
    --backup /path/to/custom_backup.h5

# Skip backup (not recommended!)
python -m fisheye.diagnostics.fix_h5_chaser_frame_numbers /path/to/stimulus.h5 --no-backup
"""

from __future__ import annotations

import argparse
import shutil
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


def analyze_frame_numbering(h5_path: Path, dataset_path: str = "/tracking_data/chaser_states"):
    """Analyze the current frame numbering pattern."""
    h5py = _load_h5py()

    with h5py.File(h5_path, "r") as handle:
        if dataset_path not in handle:
            raise KeyError(f"Dataset '{dataset_path}' not found in {h5_path}.")

        dataset = handle[dataset_path]
        data = dataset[...]

    dtype_names = data.dtype.names or ()
    if "stimulus_frame_num" not in dtype_names:
        raise ValueError(f"Dataset does not contain 'stimulus_frame_num' field.")

    frame_nums = data["stimulus_frame_num"]

    return {
        "total_records": len(frame_nums),
        "min_frame": int(np.min(frame_nums)),
        "max_frame": int(np.max(frame_nums)),
        "unique_frames": len(np.unique(frame_nums)),
        "spacing": np.diff(np.sort(frame_nums)) if len(frame_nums) > 1 else np.array([]),
    }


def fix_frame_numbering(
    h5_path: Path,
    dataset_path: str = "/tracking_data/chaser_states",
    dry_run: bool = False,
    console: Optional["Console"] = None,
):
    """Fix the frame numbering in chaser_states."""
    h5py = _load_h5py()

    def echo(message: str) -> None:
        if console is not None:
            console.print(message)
        else:
            import re
            plain = re.sub(r'\[/?[^\]]+\]', '', message)
            print(plain)

    # Analyze before
    echo("[bold cyan]Analyzing current state...[/bold cyan]")
    before = analyze_frame_numbering(h5_path, dataset_path)

    echo(f"  Total records: {before['total_records']:,}")
    echo(f"  Frame range: {before['min_frame']:,} to {before['max_frame']:,}")
    echo(f"  Unique frames: {before['unique_frames']:,}")

    if len(before['spacing']) > 0:
        # Convert to int64 for bincount
        spacing_int = before['spacing'].astype(np.int64)
        spacing_counts = np.bincount(spacing_int)
        dominant_spacing = int(np.argmax(spacing_counts))
        echo(f"  Dominant spacing: {dominant_spacing}")

    # Check if fix is needed
    if before['min_frame'] == 0 and before['max_frame'] == before['total_records'] - 1:
        echo("[green]✓ Frame numbering is already sequential (0 to N-1). No fix needed![/green]")
        return

    if dry_run:
        echo("")
        echo("[bold yellow]DRY RUN MODE - No changes will be made[/bold yellow]")
        echo("")
        echo("[bold cyan]Proposed changes:[/bold cyan]")
        echo(f"  Current: frames {before['min_frame']:,} to {before['max_frame']:,}")
        echo(f"  After fix: frames 0 to {before['total_records'] - 1:,}")
        echo(f"  Records affected: {before['total_records']:,}")
        return

    # Apply fix
    echo("")
    echo("[bold cyan]Applying fix...[/bold cyan]")

    with h5py.File(h5_path, "r+") as handle:
        if dataset_path not in handle:
            raise KeyError(f"Dataset '{dataset_path}' not found in {h5_path}.")

        dataset = handle[dataset_path]

        # Read all data
        echo("  Reading data...")
        data = dataset[...]

        # Create sequential frame numbers
        echo("  Renumbering frames...")
        old_frames = data["stimulus_frame_num"].copy()
        data["stimulus_frame_num"] = np.arange(len(data))

        # Write back
        echo("  Writing updated data...")
        dataset[...] = data

        echo("[green]✓ Frame numbering updated successfully[/green]")

    # Validate after
    echo("")
    echo("[bold cyan]Validating fix...[/bold cyan]")
    after = analyze_frame_numbering(h5_path, dataset_path)

    echo(f"  Total records: {after['total_records']:,} (unchanged)")
    echo(f"  Frame range: {after['min_frame']:,} to {after['max_frame']:,}")
    echo(f"  Unique frames: {after['unique_frames']:,}")

    if after['min_frame'] == 0 and after['max_frame'] == after['total_records'] - 1:
        echo("[green]✓ Validation passed! Frames are now sequential.[/green]")
    else:
        echo("[red]✗ Validation failed! Frame numbering is still incorrect.[/red]")
        raise RuntimeError("Frame numbering fix validation failed")


def create_backup(h5_path: Path, backup_path: Optional[Path] = None) -> Path:
    """Create a backup of the H5 file."""
    if backup_path is None:
        backup_path = h5_path.with_suffix(h5_path.suffix + ".bak")

    if backup_path.exists():
        raise FileExistsError(
            f"Backup file already exists: {backup_path}\n"
            "Move or delete the existing backup before creating a new one."
        )

    shutil.copy2(h5_path, backup_path)
    return backup_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "h5_path",
        type=Path,
        help="Path to the stimulus HDF5 file to fix.",
    )
    parser.add_argument(
        "--dataset",
        default="/tracking_data/chaser_states",
        help="Path to the chaser_states dataset (default: /tracking_data/chaser_states).",
    )
    parser.add_argument(
        "--fix-metadata",
        action="store_true",
        help="Also fix /video_metadata/frame_metadata dataset.",
    )
    parser.add_argument(
        "--fix-all",
        action="store_true",
        help="Fix both chaser_states and frame_metadata datasets.",
    )
    parser.add_argument(
        "--backup",
        type=Path,
        help="Custom backup file path (default: <h5_path>.bak).",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip backup creation (not recommended!).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without modifying the file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.h5_path.exists():
        raise FileNotFoundError(f"{args.h5_path} does not exist.")

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

    # Determine which datasets to fix
    datasets_to_fix = []
    if args.fix_all:
        datasets_to_fix = ["/tracking_data/chaser_states", "/video_metadata/frame_metadata"]
    elif args.fix_metadata:
        datasets_to_fix = ["/video_metadata/frame_metadata"]
    else:
        datasets_to_fix = [args.dataset]

    echo(f"[bold cyan]{'=' * 70}[/bold cyan]")
    echo(f"[bold]Fix H5 Frame Numbering[/bold]")
    echo(f"[bold cyan]{'=' * 70}[/bold cyan]")
    echo(f"[bold]H5 File:[/bold] {args.h5_path}")
    echo(f"[bold]Datasets to fix:[/bold] {', '.join(datasets_to_fix)}")
    echo("")

    # Create backup (unless dry-run or --no-backup)
    if not args.dry_run and not args.no_backup:
        echo("[bold cyan]Creating backup...[/bold cyan]")
        try:
            backup_path = create_backup(args.h5_path, args.backup)
            echo(f"[green]✓ Backup created:[/green] {backup_path}")
            echo("")
        except FileExistsError as e:
            echo(f"[red]✗ {e}[/red]")
            return
    elif args.no_backup:
        echo("[yellow]⚠ Skipping backup (--no-backup specified)[/yellow]")
        echo("")

    # Fix frame numbering for each dataset
    for dataset_path in datasets_to_fix:
        echo(f"[bold cyan]Processing: {dataset_path}[/bold cyan]")
        echo(f"[bold cyan]{'-' * 70}[/bold cyan]")
        try:
            fix_frame_numbering(
                args.h5_path,
                dataset_path=dataset_path,
                dry_run=args.dry_run,
                console=console,
            )
        except Exception as e:
            echo(f"[red]✗ Error: {e}[/red]")
            raise
        echo("")

    echo("")
    if args.dry_run:
        echo("[bold]Next steps:[/bold]")
        echo("  Run without --dry-run to apply the fix:")
        echo(f"  python -m fisheye.diagnostics.fix_h5_chaser_frame_numbers {args.h5_path}")
    else:
        echo("[bold green]Fix completed successfully![/bold green]")
        echo("")
        echo("[bold]Next steps:[/bold]")
        echo("  1. Verify the fix with count_h5_chaser_states.py")
        echo("  2. Re-import to Zarr with the corrected H5 file")
        if not args.no_backup:
            echo(f"  3. If needed, restore from backup: {args.h5_path}.bak")


if __name__ == "__main__":
    main()
