"""Inspect chaser position traces from a stimulus run.

This utility loads the online chaser metrics for a Palette archive, reports
basic statistics for the chaser's X/Y trajectory, and optionally saves a plot
showing the position over time.  It is intended to confirm that the imported
stimulus data contains a moving trace rather than a constant position.

Example
-------
python -m fisheye.diagnostics.check_chaser_positions data.zarr
python -m fisheye.diagnostics.check_chaser_positions data.zarr --stimulus-run stimulus_20251101_212157
python -m fisheye.diagnostics.check_chaser_positions data.zarr --plot chaser_trace.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np

from fisheye.analysis.chaser_metrics_loader import load_chaser_metrics


def _describe_axis(values: np.ndarray, label: str, console) -> None:
    def _write(message: str) -> None:
        if console is not None:
            console.print(message)
        else:
            print(message)

    if values.size == 0:
        _write(f"[red]{label}: no samples after filtering[/red]")
        return

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        _write(f"[yellow]{label}: all values are NaN[/yellow]")
        return

    val_min = float(np.nanmin(finite))
    val_max = float(np.nanmax(finite))
    val_range = val_max - val_min
    _write(
        f"[cyan]{label}[/cyan] samples: {values.size:,} • finite: {finite.size:,} • "
        f"min: {val_min:.3f} • max: {val_max:.3f} • range: {val_range:.3f}"
    )


def _compute_motion_metrics(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
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


def _render_plot(
    frames: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    save_path: Optional[Path] = None,
    *,
    start_frame: Optional[int] = None,
    stimulus_frames: Optional[np.ndarray] = None,
    start_stimulus_frame: Optional[int] = None,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - plotting optional
        raise RuntimeError("matplotlib is required for --plot output") from exc

    valid = np.isfinite(x) & np.isfinite(y)
    frames_valid = frames[valid]
    x_valid = x[valid]
    y_valid = y[valid]

    has_stimulus = stimulus_frames is not None and stimulus_frames.size == frames.size
    subplot_rows = 3 if has_stimulus else 2
    fig, axes = plt.subplots(subplot_rows, 1, figsize=(10, 8 if has_stimulus else 6), sharex=False)
    axes = np.atleast_1d(axes)
    if axes.size >= 2:
        axes[1].sharex(axes[0])

    axes[0].plot(frames_valid, x_valid, lw=1.0)
    if start_frame is not None:
        axes[0].axvline(start_frame, color="red", linestyle="--", alpha=0.7, label="Start event")
        axes[0].legend(loc="upper left")
    axes[0].set_ylabel("Chaser X (px)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(frames_valid, y_valid, lw=1.0, color="tab:orange")
    if start_frame is not None:
        axes[1].axvline(start_frame, color="red", linestyle="--", alpha=0.7, label="Start event")
        axes[1].legend(loc="upper left")
    axes[1].set_ylabel("Chaser Y (px)")
    axes[1].set_xlabel("Camera frame")
    axes[1].grid(True, alpha=0.3)

    if has_stimulus:
        stim_axis = axes[2]
        stim_frames = stimulus_frames.copy()
        stim_frames = stim_frames.astype(np.int64, copy=False)
        stim_valid_mask = valid & (stim_frames >= 0)
        stim_axis.plot(stim_frames[stim_valid_mask], x[stim_valid_mask], lw=1.0, label="Chaser X")
        stim_axis.plot(stim_frames[stim_valid_mask], y[stim_valid_mask], lw=1.0, label="Chaser Y", color="tab:orange")
        if start_stimulus_frame is not None:
            stim_axis.axvline(
                start_stimulus_frame,
                color="red",
                linestyle="--",
                alpha=0.7,
                label="Start event",
            )
        stim_axis.set_ylabel("Chaser pos (px)")
        stim_axis.set_xlabel("Stimulus frame")
        stim_axis.legend(loc="upper left")
        stim_axis.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    # Always display the figure for quick visual checks.
    plt.show()
    plt.close(fig)


def _report_protocol_start(
    frames: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    start_cam_frame: Optional[int],
    echo,
) -> None:
    if start_cam_frame is None or frames.size == 0:
        return

    matches = np.where(frames == start_cam_frame)[0]
    if matches.size:
        idx = int(matches[0])
        pos_x = float(x[idx]) if idx < x.size else float("nan")
        pos_y = float(y[idx]) if idx < y.size else float("nan")
        echo(
            f"[bold]Protocol start camera frame:[/bold] {start_cam_frame} "
            f"(index {idx:,} of {frames.size:,})"
        )
        echo(f"[bold]Chaser at start:[/bold] x={pos_x:.3f}, y={pos_y:.3f}")
    else:
        insert_idx = int(np.searchsorted(frames, start_cam_frame))
        nearest_idx = max(0, min(frames.size - 1, insert_idx))
        nearest_frame = int(frames[nearest_idx])
        echo(
            f"[yellow]Start camera frame {start_cam_frame} not present; "
            f"nearest recorded frame {nearest_frame} at index {nearest_idx:,}[/yellow]"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path, help="Path to Palette Zarr archive.")
    parser.add_argument(
        "--stimulus-run",
        help="Specific analysis/stimulus_runs entry to inspect (default: latest).",
    )
    parser.add_argument(
        "--chaser-index",
        type=int,
        default=0,
        help="Chaser index to extract from chaser_states (default: 0).",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        help="Save the matplotlib plot to this path (plot is always shown).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    bundle = load_chaser_metrics(
        args.zarr_path,
        stimulus_run=args.stimulus_run,
        metrics_run=None,
        chaser_index=args.chaser_index,
    )

    console: Optional["Console"] = None
    try:
        from rich.console import Console

        console = Console()
    except Exception:  # pragma: no cover - rich is optional
        console = None

    def echo(message: str) -> None:
        if console is not None:
            console.print(message)
        else:
            print(message)

    x = bundle.online.get("chaser_pos_x")
    y = bundle.online.get("chaser_pos_y")

    if x is None or y is None:
        raise ValueError(
            "Stimulus run does not contain chaser_pos_x / chaser_pos_y fields. "
            "Verify that the run was imported with chaser tracking."
        )

    frames = bundle.camera_frame_ids.astype(np.int64, copy=False)

    echo(f"[bold]Stimulus run:[/bold] {bundle.provenance.get('stimulus_run', 'unknown')}")
    echo(f"[bold]Chaser index:[/bold] {args.chaser_index}")
    echo(f"[bold]Total frames:[/bold] {frames.size:,}")
    if frames.size:
        echo(f"[bold]Camera frame range:[/bold] {int(frames.min())} → {int(frames.max())}")

    _describe_axis(x, "chaser_pos_x", console=console)
    _describe_axis(y, "chaser_pos_y", console=console)

    start_cam_frame = bundle.provenance.get("start_event_camera_frame")
    start_stim_frame = bundle.provenance.get("start_event_stimulus_frame")

    metrics = _compute_motion_metrics(x, y)
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
        echo("[green]Chaser positions vary over time.[/green]")
    else:
        echo("[red]Chaser positions appear constant or missing; investigate import pipeline.[/red]")

    _report_protocol_start(frames, x, y, start_cam_frame, echo)

    if start_stim_frame is not None:
        echo(f"[bold]Start event (stimulus frame):[/bold] {start_stim_frame}")

    _render_plot(
        frames,
        x,
        y,
        save_path=args.plot,
        start_frame=start_cam_frame,
        stimulus_frames=bundle.stimulus_frame_nums,
        start_stimulus_frame=start_stim_frame,
    )
    if args.plot:
        echo(f"[bold]Saved plot:[/bold] {args.plot}")


if __name__ == "__main__":
    main()
