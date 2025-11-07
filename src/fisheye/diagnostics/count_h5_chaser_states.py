#!/usr/bin/env python3
"""Count chaser states in stimulus H5 file and compare to video frame counts.

This diagnostic verifies that the stimulus program is logging chaser states correctly
by comparing:
- Total chaser state records in the H5 file
- Number of unique stimulus frames with chaser data
- Expected video frame counts (if provided)

Helps identify issues like:
- Missing chaser state logging
- Duplicate stimulus frames
- Mismatched frame counts between stimulus and camera video

Example
-------
# Basic count
python -m fisheye.diagnostics.count_h5_chaser_states /path/to/stimulus.h5

# Compare to video frame counts
python -m fisheye.diagnostics.count_h5_chaser_states /path/to/stimulus.h5 \\
    --stimulus-video-frames 89882 \\
    --camera-video-frames 45753

# Show per-chaser breakdown
python -m fisheye.diagnostics.count_h5_chaser_states /path/to/stimulus.h5 --per-chaser

# Check for duplicates
python -m fisheye.diagnostics.count_h5_chaser_states /path/to/stimulus.h5 --check-duplicates
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np


def _load_h5py():
    try:
        import h5py
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "h5py is required for this diagnostic. Activate the Palette environment "
            "or install h5py and retry."
        ) from exc
    return h5py


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "h5_path",
        type=Path,
        help="Path to the stimulus HDF5 file.",
    )
    parser.add_argument(
        "--dataset",
        default="/tracking_data/chaser_states",
        help="Path to the chaser_states dataset (default: /tracking_data/chaser_states).",
    )
    parser.add_argument(
        "--per-chaser",
        action="store_true",
        help="Show breakdown per chaser index.",
    )
    parser.add_argument(
        "--check-duplicates",
        action="store_true",
        help="Check for duplicate stimulus frames per chaser.",
    )
    parser.add_argument(
        "--stimulus-video-frames",
        type=int,
        help="Expected number of frames in the stimulus video (from ffprobe).",
    )
    parser.add_argument(
        "--camera-video-frames",
        type=int,
        help="Expected number of frames in the camera video (from ffprobe).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    h5py = _load_h5py()

    if not args.h5_path.exists():
        raise FileNotFoundError(f"{args.h5_path} does not exist.")

    console: Optional["Console"] = None
    try:
        from rich.console import Console

        console = Console()
    except Exception:  # pragma: no cover - rich optional
        console = None

    def echo(message: str) -> None:
        if console is not None:
            console.print(message)
        else:
            # Strip rich markup for plain output
            import re
            plain = re.sub(r'\[/?[^\]]+\]', '', message)
            print(plain)

    with h5py.File(args.h5_path, "r") as handle:
        if args.dataset not in handle:
            raise KeyError(f"Dataset '{args.dataset}' not found in {args.h5_path}.")
        dataset = handle[args.dataset]
        if not isinstance(dataset, h5py.Dataset):
            raise TypeError(f"Object at '{args.dataset}' is not a dataset.")

        data = dataset[...]

    dtype_names = data.dtype.names or ()

    # Basic counts
    total_records = data.shape[0]
    echo(f"[bold cyan]{'=' * 70}[/bold cyan]")
    echo(f"[bold]H5 File:[/bold] {args.h5_path.name}")
    echo(f"[bold]Dataset:[/bold] {args.dataset}")
    echo(f"[bold cyan]{'=' * 70}[/bold cyan]")
    echo("")
    echo(f"[bold green]Total chaser state records:[/bold green] {total_records:,}")

    # Check for stimulus_frame_num field
    if "stimulus_frame_num" in dtype_names:
        frame_nums = data["stimulus_frame_num"]
        unique_frames = np.unique(frame_nums)
        echo(f"[bold green]Unique stimulus frames:[/bold green] {len(unique_frames):,}")

        min_frame = int(np.min(frame_nums))
        max_frame = int(np.max(frame_nums))
        expected_sequential = max_frame - min_frame + 1
        echo(f"[bold]Frame range:[/bold] {min_frame:,} to {max_frame:,}")
        echo(f"[bold]Expected if sequential:[/bold] {expected_sequential:,}")

        if len(unique_frames) < expected_sequential:
            missing = expected_sequential - len(unique_frames)
            # Check if this is regular spacing (e.g., every 2nd frame)
            if len(unique_frames) > 1:
                spacing_ratio = expected_sequential / len(unique_frames)
                if abs(spacing_ratio - 2.0) < 0.01:
                    echo(f"[green]✓ Regular 2:1 spacing detected (every other frame logged)[/green]")
                    echo(f"  stimulus_frame_num appears to be a global counter")
                    echo(f"  but chaser states logged every 2 frames")
                else:
                    echo(f"[yellow]⚠ Missing {missing:,} frames (gaps in sequence)[/yellow]")
                    echo(f"  Spacing ratio: {spacing_ratio:.2f}:1")
            else:
                echo(f"[yellow]⚠ Missing {missing:,} frames (gaps in sequence)[/yellow]")
        elif len(unique_frames) == expected_sequential:
            echo(f"[green]✓ No gaps in stimulus frame sequence[/green]")
    else:
        echo("[yellow]⚠ No 'stimulus_frame_num' field found[/yellow]")
        unique_frames = None

    echo("")

    # Per-chaser breakdown
    if "chaser_index" in dtype_names:
        chaser_indices = np.unique(data["chaser_index"])
        echo(f"[bold]Chaser indices found:[/bold] {len(chaser_indices)} {list(chaser_indices)}")

        if args.per_chaser:
            echo("")
            echo("[bold cyan]Per-Chaser Breakdown:[/bold cyan]")
            echo(f"[bold cyan]{'-' * 70}[/bold cyan]")
            for idx in chaser_indices:
                mask = data["chaser_index"] == idx
                count = np.sum(mask)
                echo(f"  Chaser {idx}: {count:,} records")

                if "stimulus_frame_num" in dtype_names:
                    frames = data["stimulus_frame_num"][mask]
                    unique = np.unique(frames)
                    echo(f"    Unique frames: {len(unique):,}")

                    if args.check_duplicates:
                        counter = Counter(frames)
                        duplicates = {f: c for f, c in counter.items() if c > 1}
                        if duplicates:
                            echo(f"    [yellow]⚠ {len(duplicates):,} frames with duplicates:[/yellow]")
                            # Show first few examples
                            for frame, count in sorted(duplicates.items())[:5]:
                                echo(f"      Frame {frame}: {count} occurrences")
                            if len(duplicates) > 5:
                                echo(f"      ... and {len(duplicates) - 5} more")
                        else:
                            echo(f"    [green]✓ No duplicate frames[/green]")
    else:
        echo("[yellow]⚠ No 'chaser_index' field found[/yellow]")

    # Compare to video frame counts if provided
    if args.stimulus_video_frames or args.camera_video_frames:
        echo("")
        echo("[bold cyan]Video Frame Comparison:[/bold cyan]")
        echo(f"[bold cyan]{'-' * 70}[/bold cyan]")

        if args.stimulus_video_frames:
            echo(f"[bold]Stimulus video frames (ffprobe):[/bold] {args.stimulus_video_frames:,}")
            if unique_frames is not None:
                diff = len(unique_frames) - args.stimulus_video_frames
                ratio = len(unique_frames) / args.stimulus_video_frames
                echo(f"  Chaser unique frames: {len(unique_frames):,}")
                echo(f"  Difference: {diff:+,} ({ratio:.2%})")

                if abs(ratio - 1.0) < 0.01:
                    echo(f"  [green]✓ Chaser frames match stimulus video (within 1%)[/green]")
                elif ratio < 0.9:
                    echo(f"  [red]✗ Significantly fewer chaser frames than stimulus video[/red]")
                elif ratio > 1.1:
                    echo(f"  [yellow]⚠ More chaser frames than stimulus video (duplicates?)[/yellow]")

        if args.camera_video_frames:
            echo(f"[bold]Camera video frames (ffprobe):[/bold] {args.camera_video_frames:,}")
            if args.stimulus_video_frames:
                ratio = args.stimulus_video_frames / args.camera_video_frames
                echo(f"  Stimulus:Camera ratio: {ratio:.4f}:1")

                if abs(ratio - 2.0) < 0.05:
                    echo(f"  [green]✓ Expected 2:1 ratio (60fps stimulus, 30fps camera)[/green]")
                else:
                    echo(f"  [yellow]⚠ Unexpected ratio (expected ~2:1)[/yellow]")

    # Summary
    echo("")
    echo("[bold cyan]Summary:[/bold cyan]")
    echo(f"[bold cyan]{'-' * 70}[/bold cyan]")

    issues = []
    if unique_frames is not None and len(unique_frames) < expected_sequential:
        # Check if it's regular 2:1 spacing (not an issue)
        spacing_ratio = expected_sequential / len(unique_frames) if len(unique_frames) > 0 else 0
        if not (abs(spacing_ratio - 2.0) < 0.01):
            issues.append("Gaps in stimulus frame sequence")

    if args.check_duplicates and "chaser_index" in dtype_names and "stimulus_frame_num" in dtype_names:
        has_dupes = False
        for idx in chaser_indices:
            mask = data["chaser_index"] == idx
            frames = data["stimulus_frame_num"][mask]
            counter = Counter(frames)
            if any(c > 1 for c in counter.values()):
                has_dupes = True
                break
        if has_dupes:
            issues.append("Duplicate stimulus frames detected")

    if args.stimulus_video_frames and unique_frames is not None:
        ratio = len(unique_frames) / args.stimulus_video_frames
        if ratio < 0.9:
            issues.append("Significantly fewer chaser frames than stimulus video")

    if issues:
        echo("[yellow]Issues detected:[/yellow]")
        for issue in issues:
            echo(f"  • {issue}")
    else:
        echo("[green]✓ No obvious issues detected[/green]")

    echo("")


if __name__ == "__main__":
    main()
