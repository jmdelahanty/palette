#!/usr/bin/env python3
"""Visualize refined online movement tracking data.

This script creates visualizations comparing original and refined
online target positions, showing the effects of smoothing, outlier removal,
and gap interpolation. Matches the style of visualize_refined_detections.py.
"""

from __future__ import annotations

import argparse
import copy
from typing import Any, Dict, Optional

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import zarr
from rich.console import Console

from ..refinement.refine_online_detect import (
    CanonicalOnlineRefinementError,
    REFINED_ONLINE_GROUP,
    load_bound_refined_online_coordinate_evidence,
)


def load_stage(
    group: zarr.Group,
    frames: np.ndarray,
    fps: float,
    stage_name: str,
) -> Dict:
    """Load data from a refinement stage (original, filtered, or interpolated).

    Args:
        group: Zarr group containing stage data
        frames: Camera frame IDs
        fps: Frames per second
        stage_name: Name of the stage

    Returns:
        Dictionary containing stage data
    """
    positions = group["positions_px"][:]
    valid_mask = group["valid_mask"][:]

    # Get valid positions
    valid_positions = positions[valid_mask]
    valid_frames = frames[valid_mask]

    # Calculate coverage
    total_frames = len(frames)
    frames_with_detections = valid_mask.sum()
    coverage_percent = (frames_with_detections / total_frames * 100) if total_frames > 0 else 0.0

    # Time array
    time_seconds = valid_frames / fps

    # Check for interpolation mask
    interpolation_mask = None
    if "interpolation_mask" in group:
        interpolation_mask = group["interpolation_mask"][:]

    return {
        "positions": positions,
        "valid_mask": valid_mask,
        "valid_positions": valid_positions,
        "valid_frames": valid_frames,
        "time_seconds": time_seconds,
        "interpolation_mask": interpolation_mask,
        "total_frames": total_frames,
        "frames_with_detections": int(frames_with_detections),
        "coverage_percent": float(coverage_percent),
        "total_detections": int(frames_with_detections),
        "stage": stage_name,
    }


def load_refined_online_visualization_inputs(
    root: zarr.Group,
    *,
    refined_run: Optional[str] = None,
) -> dict[str, Any]:
    """Copy one exact canonical publication, then revalidate its live proof."""

    refined_runs = root.get(REFINED_ONLINE_GROUP)
    if not isinstance(refined_runs, zarr.Group):
        raise CanonicalOnlineRefinementError(
            "No canonical refined_online_runs parent exists in this archive."
        )
    resolved = str(refined_run or "").strip()
    if not resolved:
        resolved = str(
            refined_runs.attrs.get("latest_complete")
            or refined_runs.attrs.get("latest")
            or ""
        ).strip()
    if not resolved or "/" in resolved or resolved not in refined_runs:
        raise CanonicalOnlineRefinementError(
            f"Canonical refined-online run {refined_run!r} is unavailable."
        )
    refined_group = refined_runs[resolved]
    if not isinstance(refined_group, zarr.Group):
        raise CanonicalOnlineRefinementError(
            f"refined_online_runs/{resolved} is not one run group."
        )
    evidence = load_bound_refined_online_coordinate_evidence(root, refined_group)
    fps_raw = root.attrs.get("fps", 60.0)
    try:
        fps = float(fps_raw)
    except (TypeError, ValueError) as exc:
        raise CanonicalOnlineRefinementError(
            f"Refined-online visualization fps is invalid: {exc}."
        ) from exc
    if not np.isfinite(fps) or fps <= 0.0:
        raise CanonicalOnlineRefinementError(
            "Refined-online visualization requires finite positive fps."
        )

    attrs = copy.deepcopy(dict(refined_group.attrs))
    frames = np.asarray(refined_group["camera_frame_ids"][:], dtype=np.int64)
    original_valid = np.asarray(
        refined_group["original_valid_mask"][:],
        dtype=bool,
    )
    original_positions = np.full((len(frames), 2), np.nan, dtype=np.float64)
    datasets: dict[str, dict[str, Any]] = {
        "original": {
            "positions": original_positions,
            "valid_mask": original_valid,
            "valid_positions": original_positions[original_valid],
            "valid_frames": frames[original_valid],
            "time_seconds": frames[original_valid] / fps,
            "interpolation_mask": None,
            "total_frames": len(frames),
            "frames_with_detections": int(original_valid.sum()),
            "coverage_percent": (
                float(original_valid.sum() / len(frames) * 100)
                if len(frames)
                else 0.0
            ),
            "total_detections": int(original_valid.sum()),
            "stage": "original",
        },
        "filtered": load_stage(refined_group["filtered"], frames, fps, "filtered"),
        "interpolated": load_stage(
            refined_group["interpolated"],
            frames,
            fps,
            "interpolated",
        ),
    }
    # Discard every copied value if the exact publication changed mid-read.
    evidence.assert_verified()
    return {
        "run_name": resolved,
        "fps": fps,
        "attrs": attrs,
        "frames": frames,
        "datasets": datasets,
    }


def visualize_refinement_pipeline(
    zarr_path: str,
    refined_run: Optional[str] = None,
    save_path: Optional[str] = None,
    console: Optional[Console] = None,
) -> None:
    """Create visualization comparing refinement stages.

    Args:
        zarr_path: Path to zarr archive
        refined_run: Refined run name (defaults to latest)
        save_path: Optional path to save figure
        console: Rich console for output
    """
    if console is None:
        console = Console()

    console.print(f"\n[bold cyan]{'=' * 70}[/bold cyan]")
    console.print("[bold cyan]ONLINE REFINEMENT PIPELINE VISUALIZATION[/bold cyan]")
    console.print(f"[bold cyan]{'=' * 70}[/bold cyan]")

    # Open zarr
    root = zarr.open(str(zarr_path), mode="r")
    inputs = load_refined_online_visualization_inputs(
        root,
        refined_run=refined_run,
    )
    refined_run = str(inputs["run_name"])
    refined_attrs = inputs["attrs"]
    frames = inputs["frames"]
    datasets = inputs["datasets"]
    console.print(f"\nRefined run: [cyan]{refined_run}[/cyan]")

    # Load metadata
    stimulus_run = refined_attrs.get("source_stimulus_run", "unknown")
    params = refined_attrs.get("parameters", {})

    console.print(f"Source stimulus run: [cyan]{stimulus_run}[/cyan]")
    console.print("\n[bold]Parameters:[/bold]")
    console.print(f"  Window length: {params.get('window_length', 'N/A')}")
    console.print(f"  Polynomial order: {params.get('polyorder', 'N/A')}")
    console.print(f"  Displacement threshold: {params.get('displacement_threshold', 'N/A')} px")
    console.print(f"  Max gap: {params.get('max_gap', 'N/A')} frames")

    # Create visualization
    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(
        4, 3, figure=fig, hspace=0.35, wspace=0.25, height_ratios=[1.2, 0.5, 0.8, 0.8]
    )

    stage_names = {
        "original": "Original Positions",
        "filtered": "Filtered (Smoothed + Outliers Removed)",
        "interpolated": "Interpolated (Gaps Filled)",
    }

    colors = {
        "original": "blue",
        "filtered": "green",
        "interpolated": "purple",
    }

    # Row 1: Trajectory plots
    for idx, (stage, data) in enumerate(datasets.items()):
        ax = fig.add_subplot(gs[0, idx])

        if len(data["valid_positions"]) > 0:
            if stage == "interpolated" and data["interpolation_mask"] is not None:
                # Show real vs interpolated
                interp_mask = data["interpolation_mask"]
                real_mask = data["valid_mask"] & ~interp_mask
                interp_only = data["valid_mask"] & interp_mask

                if real_mask.any():
                    real_pos = data["positions"][real_mask]
                    real_frames = frames[real_mask]
                    ax.scatter(
                        real_pos[:, 0],
                        real_pos[:, 1],
                        c=real_frames,
                        cmap="viridis",
                        s=2,
                        alpha=0.6,
                        label="Real",
                    )

                if interp_only.any():
                    interp_pos = data["positions"][interp_only]
                    ax.scatter(
                        interp_pos[:, 0],
                        interp_pos[:, 1],
                        c="red",
                        s=15,
                        marker="o",
                        alpha=0.7,
                        edgecolors="darkred",
                        linewidths=1,
                        label="Interpolated",
                    )
                ax.legend(loc="upper right", fontsize=8)
            else:
                # Standard scatter plot
                scatter = ax.scatter(
                    data["valid_positions"][:, 0],
                    data["valid_positions"][:, 1],
                    c=data["valid_frames"],
                    cmap="viridis",
                    s=2,
                    alpha=0.6,
                )
                plt.colorbar(scatter, ax=ax, label="Frame", pad=0.01)

        # Title with statistics
        stats = data
        total_frames = stats["total_frames"]
        title = f"{stage_names[stage]}\n"
        title += (
            f"Coverage: {stats['frames_with_detections']}/{total_frames} "
            f"frames ({stats['coverage_percent']:.2f}%)"
        )

        if stage == "filtered":
            removed = datasets["original"]["total_detections"] - stats["total_detections"]
            coverage_loss = datasets["original"]["coverage_percent"] - stats["coverage_percent"]
            title += f"\nRemoved: {removed} ({coverage_loss:.2f}%)"
        elif stage == "interpolated":
            added = stats["total_detections"] - datasets["filtered"]["total_detections"]
            coverage_gain = stats["coverage_percent"] - datasets["filtered"]["coverage_percent"]
            title += f"\nAdded: {added} ({coverage_gain:.2f}%)"

        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.set_xlabel("X Position (pixels)")
        ax.set_ylabel("Y Position (pixels)")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    # Row 2: Coverage barcode
    for idx, (stage, data) in enumerate(datasets.items()):
        ax = fig.add_subplot(gs[1, idx])

        valid_mask = data["valid_mask"]
        coverage_array = np.zeros((1, len(valid_mask)))
        coverage_array[0, valid_mask] = 1

        ax.imshow(
            coverage_array,
            aspect="auto",
            cmap="RdYlGn",
            extent=[0, len(valid_mask), 0, 1],
            interpolation="nearest",
            vmin=0,
            vmax=1,
        )

        ax.set_xlabel("Frame Index")
        ax.set_yticks([])
        ax.set_title(f"{stage.title()} Coverage", fontsize=9)

    # Row 3: Position over time
    for idx, (stage, data) in enumerate(datasets.items()):
        ax = fig.add_subplot(gs[2, idx])

        if len(data["valid_positions"]) > 0:
            ax.plot(
                data["time_seconds"],
                data["valid_positions"][:, 0],
                "b-",
                alpha=0.6,
                linewidth=0.5,
                label="X",
            )
            ax.plot(
                data["time_seconds"],
                data["valid_positions"][:, 1],
                "r-",
                alpha=0.6,
                linewidth=0.5,
                label="Y",
            )

        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Position (pixels)")
        ax.set_title(f"{stage.title()} Positions", fontsize=9)
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Row 4: Displacement analysis
    for idx, (stage, data) in enumerate(datasets.items()):
        ax = fig.add_subplot(gs[3, idx])

        if len(data["valid_positions"]) > 1:
            # Calculate frame-to-frame displacement
            displacement = np.sqrt(
                np.diff(data["valid_positions"][:, 0]) ** 2
                + np.diff(data["valid_positions"][:, 1]) ** 2
            )

            # Check for consecutive frames
            frame_diff = np.diff(data["valid_frames"])
            consecutive = frame_diff == 1

            if consecutive.any():
                time_disp = data["time_seconds"][1:]
                ax.plot(
                    time_disp[consecutive],
                    displacement[consecutive],
                    colors[stage],
                    alpha=0.6,
                    linewidth=0.5,
                )

                # Add threshold line if this is filtered or interpolated
                if stage in ["filtered", "interpolated"]:
                    threshold = params.get("displacement_threshold", 100.0)
                    ax.axhline(
                        y=threshold,
                        color="red",
                        linestyle="--",
                        alpha=0.5,
                        linewidth=1,
                        label=f"Threshold ({threshold} px)",
                    )
                    ax.legend()

                # Add mean
                mean_disp = displacement[consecutive].mean()
                ax.axhline(y=mean_disp, color="green", linestyle=":", alpha=0.5, linewidth=1)
                ax.text(
                    0.02,
                    0.95,
                    f"Mean: {mean_disp:.1f} px",
                    transform=ax.transAxes,
                    fontsize=9,
                    va="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )

        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Displacement (pixels)")
        ax.set_title(f"{stage.title()} Displacement", fontsize=9)
        ax.grid(True, alpha=0.3)

    # Overall title
    fig.suptitle(
        f"Online Refinement Pipeline - {refined_run}\nSource: {stimulus_run}",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        console.print(f"[green]Saved plot to {save_path}[/green]")
    else:
        plt.show()

    plt.close(fig)

    # Print summary
    console.print("\n[bold]Coverage Summary:[/bold]")
    for stage in ["original", "filtered", "interpolated"]:
        data = datasets[stage]
        console.print(
            f"  {stage.title():<13}: {data['coverage_percent']:.2f}% "
            f"({data['frames_with_detections']} frames, {data['total_detections']} detections)"
        )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Visualize refined online movement tracking data"
    )
    parser.add_argument("zarr_path", help="Path to Palette zarr archive")
    parser.add_argument("--refined-run", help="Refined run name (defaults to latest)")
    parser.add_argument("--output", help="Path to save figure")

    args = parser.parse_args(argv)

    console = Console()

    try:
        visualize_refinement_pipeline(
            zarr_path=args.zarr_path,
            refined_run=args.refined_run,
            save_path=args.output,
            console=console,
        )
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
