#!/usr/bin/env python3
"""Analyze the spacing pattern of stimulus_frame_num in H5 chaser states.

This diagnostic checks whether chaser states are logged at regular intervals
by analyzing the frame number gaps. Helps identify issues like:
- Logging every Nth frame (e.g., every other frame)
- Irregular sampling patterns
- Frame counter mismatches

Example
-------
python -m fisheye.diagnostics.analyze_h5_frame_spacing /path/to/stimulus.h5
python -m fisheye.diagnostics.analyze_h5_frame_spacing /path/to/stimulus.h5 --plot spacing.png
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
    except ModuleNotFoundError as exc:
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
    parser.add_argument("h5_path", type=Path, help="Path to the stimulus HDF5 file.")
    parser.add_argument(
        "--dataset",
        default="/tracking_data/chaser_states",
        help="Path to the chaser_states dataset (default: /tracking_data/chaser_states).",
    )
    parser.add_argument(
        "--chaser-index",
        type=int,
        default=0,
        help="Chaser index to analyze (default: 0).",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        help="Optional path to save a matplotlib plot of frame spacing histogram.",
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
    except Exception:
        console = None

    def echo(message: str) -> None:
        if console is not None:
            console.print(message)
        else:
            import re
            plain = re.sub(r'\[/?[^\]]+\]', '', message)
            print(plain)

    with h5py.File(args.h5_path, "r") as handle:
        if args.dataset not in handle:
            raise KeyError(f"Dataset '{args.dataset}' not found in {args.h5_path}.")
        dataset = handle[args.dataset]
        data = dataset[...]

    dtype_names = data.dtype.names or ()
    if "stimulus_frame_num" not in dtype_names:
        raise ValueError("Dataset does not contain 'stimulus_frame_num' field.")

    # Filter to specific chaser
    if "chaser_index" in dtype_names:
        mask = data["chaser_index"] == args.chaser_index
        filtered = data[mask]
        if filtered.size == 0:
            raise ValueError(f"No rows found for chaser_index={args.chaser_index}.")
    else:
        filtered = data

    frame_nums = np.sort(filtered["stimulus_frame_num"])

    echo(f"[bold cyan]{'=' * 70}[/bold cyan]")
    echo(f"[bold]H5 File:[/bold] {args.h5_path.name}")
    echo(f"[bold]Chaser Index:[/bold] {args.chaser_index}")
    echo(f"[bold cyan]{'=' * 70}[/bold cyan]")
    echo("")
    echo(f"[bold]Total records:[/bold] {len(frame_nums):,}")
    echo(f"[bold]Frame range:[/bold] {int(frame_nums[0]):,} to {int(frame_nums[-1]):,}")
    echo("")

    # Compute spacing (frame number differences)
    if len(frame_nums) < 2:
        echo("[yellow]Not enough records to analyze spacing.[/yellow]")
        return

    spacing = np.diff(frame_nums)

    echo("[bold cyan]Frame Number Spacing Analysis:[/bold cyan]")
    echo(f"[bold cyan]{'-' * 70}[/bold cyan]")
    echo(f"[bold]Min spacing:[/bold] {int(np.min(spacing))}")
    echo(f"[bold]Max spacing:[/bold] {int(np.max(spacing))}")
    echo(f"[bold]Mean spacing:[/bold] {float(np.mean(spacing)):.2f}")
    echo(f"[bold]Median spacing:[/bold] {int(np.median(spacing))}")
    echo(f"[bold]Std dev:[/bold] {float(np.std(spacing)):.2f}")
    echo("")

    # Count frequency of each spacing value
    counter = Counter(spacing)
    most_common = counter.most_common(10)

    echo("[bold cyan]Most Common Spacings:[/bold cyan]")
    echo(f"[bold cyan]{'-' * 70}[/bold cyan]")
    total_gaps = len(spacing)
    for gap, count in most_common:
        pct = 100.0 * count / total_gaps
        echo(f"  Spacing {int(gap):4d}: {count:6,} occurrences ({pct:5.1f}%)")
    echo("")

    # Diagnosis
    echo("[bold cyan]Interpretation:[/bold cyan]")
    echo(f"[bold cyan]{'-' * 70}[/bold cyan]")

    dominant_spacing = most_common[0][0]
    dominant_pct = 100.0 * most_common[0][1] / total_gaps

    if dominant_pct > 95:
        echo(f"[green]✓ Very regular spacing: {int(dominant_spacing)} frames ({dominant_pct:.1f}% of gaps)[/green]")
        if dominant_spacing == 1:
            echo("  → Chaser states logged on consecutive stimulus frames")
        elif dominant_spacing == 2:
            echo("  → Chaser states logged every other frame (2:1 ratio)")
            echo("  → This matches camera:stimulus frame rate difference")
        else:
            echo(f"  → Chaser states logged every {int(dominant_spacing)} frames")
    elif dominant_pct > 80:
        echo(f"[yellow]⚠ Mostly regular spacing: {int(dominant_spacing)} frames ({dominant_pct:.1f}% of gaps)[/yellow]")
        echo(f"  → Some irregular gaps present ({100-dominant_pct:.1f}% of gaps)")
    else:
        echo(f"[red]✗ Irregular spacing detected[/red]")
        echo(f"  → Most common spacing is {int(dominant_spacing)} frames ({dominant_pct:.1f}%)")
        echo(f"  → But significant variation exists")

    # Check if spacing pattern matches expected 2:1 ratio
    if dominant_spacing == 2 and dominant_pct > 90:
        echo("")
        echo("[bold]Conclusion:[/bold]")
        echo("  Your stimulus_frame_num appears to be a GLOBAL frame counter")
        echo("  that increments every frame, but chaser states are logged every")
        echo("  OTHER frame. This explains:")
        echo("    • Why frame range is ~2x the record count")
        echo("    • Why you see 'missing' frames (they're just not logged)")
        echo("")
        echo("  [green]This is likely CORRECT behavior if your stimulus video is[/green]")
        echo("  [green]60fps but you're only logging chaser states at 30fps to match[/green]")
        echo("  [green]the camera frame rate.[/green]")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except ModuleNotFoundError:
            echo("[yellow]matplotlib not available, skipping plot.[/yellow]")
            return

        # Histogram of spacing
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        # Histogram
        axes[0].hist(spacing, bins=min(50, int(np.max(spacing) - np.min(spacing) + 1)),
                     edgecolor='black', alpha=0.7)
        axes[0].set_xlabel("Frame spacing")
        axes[0].set_ylabel("Frequency")
        axes[0].set_title(f"Distribution of frame_num spacing (chaser {args.chaser_index})")
        axes[0].grid(True, alpha=0.3)

        # Spacing over time
        axes[1].plot(spacing, lw=0.5, alpha=0.7)
        axes[1].axhline(dominant_spacing, color='red', linestyle='--',
                       label=f'Dominant spacing: {int(dominant_spacing)}')
        axes[1].set_xlabel("Gap index")
        axes[1].set_ylabel("Frame spacing")
        axes[1].set_title("Frame spacing over time")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot, dpi=150)
        plt.close(fig)
        echo(f"[bold]Saved plot:[/bold] {args.plot}")


if __name__ == "__main__":
    main()
