"""Inspect chaser positions stored in a raw stimulus HDF5 file.

This diagnostic is meant to verify that the `chaser_pos_x` and `chaser_pos_y`
fields inside ``/tracking_data/chaser_states`` vary over time.  It mirrors the
Zarr-side checker but operates directly on an imported *.h5 or *.h5.bak file so
you can compare the pre-import signal.

Example
-------
python -m fisheye.diagnostics.check_h5_chaser_positions /path/to/run.h5.bak
python -m fisheye.diagnostics.check_h5_chaser_positions /path/to/run.h5 --chaser-index 1 --plot h5_trace.png
"""

from __future__ import annotations

import argparse
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


def _describe_axis(values: np.ndarray, label: str, echo) -> None:
    if values.size == 0:
        echo(f"[red]{label}: no samples[/red]")
        return

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        echo(f"[yellow]{label}: all values are NaN[/yellow]")
        return

    val_min = float(np.nanmin(finite))
    val_max = float(np.nanmax(finite))
    val_range = val_max - val_min
    echo(
        f"[cyan]{label}[/cyan] samples: {values.size:,} • finite: {finite.size:,} • "
        f"min: {val_min:.3f} • max: {val_max:.3f} • range: {val_range:.3f}"
    )


def _movement_metrics(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(x) & np.isfinite(y)
    if not np.any(valid):
        return {
            "valid_samples": 0,
            "range_x": float("nan"),
            "range_y": float("nan"),
            "max_step": float("nan"),
            "nonzero_steps": 0,
        }

    x_valid = x[valid]
    y_valid = y[valid]
    range_x = float(np.nanmax(x_valid) - np.nanmin(x_valid))
    range_y = float(np.nanmax(y_valid) - np.nanmin(y_valid))

    if x_valid.size < 2:
        return {
            "valid_samples": int(x_valid.size),
            "range_x": range_x,
            "range_y": range_y,
            "max_step": 0.0,
            "nonzero_steps": 0,
        }

    dx = np.diff(x_valid)
    dy = np.diff(y_valid)
    step = np.sqrt(dx ** 2 + dy ** 2)
    max_step = float(np.nanmax(step))
    nonzero_steps = int(np.count_nonzero(step > 1e-6))

    return {
        "valid_samples": int(x_valid.size),
        "range_x": range_x,
        "range_y": range_y,
        "max_step": max_step,
        "nonzero_steps": nonzero_steps,
    }


def _save_plot(frames: np.ndarray, x: np.ndarray, y: np.ndarray, output: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("matplotlib is required for --plot output.") from exc

    valid = np.isfinite(x) & np.isfinite(y)
    frames_valid = frames[valid]
    x_valid = x[valid]
    y_valid = y[valid]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(frames_valid, x_valid, lw=1.0)
    axes[0].set_ylabel("Chaser X (px)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(frames_valid, y_valid, lw=1.0, color="tab:orange")
    axes[1].set_ylabel("Chaser Y (px)")
    axes[1].set_xlabel("Stimulus frame")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5_path", type=Path, help="Path to the stimulus HDF5 (.h5/.bak) file.")
    parser.add_argument(
        "--dataset",
        default="/tracking_data/chaser_states",
        help="Path to the chaser_states dataset (default: /tracking_data/chaser_states).",
    )
    parser.add_argument(
        "--chaser-index",
        type=int,
        default=0,
        help="Chaser index to inspect (default: 0).",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        help="Optional path to save a matplotlib plot of X/Y versus stimulus frame.",
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
            print(message)

    with h5py.File(args.h5_path, "r") as handle:
        if args.dataset not in handle:
            raise KeyError(f"Dataset '{args.dataset}' not found in {args.h5_path}.")
        dataset = handle[args.dataset]
        if not isinstance(dataset, h5py.Dataset):
            raise TypeError(f"Object at '{args.dataset}' is not a dataset.")

        data = dataset[...]

    dtype_names = data.dtype.names or ()
    required_fields = {"chaser_pos_x", "chaser_pos_y"}
    missing = required_fields - set(dtype_names)
    if missing:
        raise ValueError(
            f"Dataset is missing required fields: {', '.join(sorted(missing))}"
        )

    if "chaser_index" in dtype_names:
        mask = data["chaser_index"] == args.chaser_index
    else:
        mask = np.ones(data.shape[0], dtype=bool)

    filtered = data[mask]
    if filtered.size == 0:
        raise ValueError(
            f"No rows found for chaser_index={args.chaser_index}. "
            "Try a different index."
        )

    frames = (
        filtered["stimulus_frame_num"].astype(np.int64)
        if "stimulus_frame_num" in dtype_names
        else np.arange(filtered.size, dtype=np.int64)
    )
    x = filtered["chaser_pos_x"].astype(np.float64)
    y = filtered["chaser_pos_y"].astype(np.float64)

    echo(f"[bold]H5 file:[/bold] {args.h5_path}")
    echo(f"[bold]Dataset:[/bold] {args.dataset}")
    echo(f"[bold]Rows loaded:[/bold] {filtered.size:,} (from {data.size:,})")

    _describe_axis(x, "chaser_pos_x", echo)
    _describe_axis(y, "chaser_pos_y", echo)

    metrics = _movement_metrics(x, y)
    moving = (
        metrics["nonzero_steps"] > 0
        and (metrics["range_x"] > 1e-3 or metrics["range_y"] > 1e-3)
    )

    echo(
        f"[bold]Valid samples:[/bold] {metrics['valid_samples']:,} • "
        f"range_x: {metrics['range_x']:.3f} • range_y: {metrics['range_y']:.3f} • "
        f"max step: {metrics['max_step']:.3f} • non-zero steps: {metrics['nonzero_steps']:,}"
    )
    if moving:
        echo("[green]Chaser positions vary over time in the H5 data.[/green]")
    else:
        echo("[red]Chaser positions appear constant in the H5 data.[/red]")

    if args.plot:
        _save_plot(frames, x, y, args.plot)
        echo(f"[bold]Saved plot:[/bold] {args.plot}")


if __name__ == "__main__":
    main()
