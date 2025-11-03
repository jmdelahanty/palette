#!/usr/bin/env python3
"""Overlay chaser and fish positions from offline metrics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import zarr
from matplotlib.patches import Circle
from rich.console import Console


DEFAULT_TITLE = "Chaser vs Fish Positions"


def _resolve_run(parent: zarr.Group, requested: Optional[str]) -> str:
    if requested:
        if requested not in parent:
            raise ValueError(f"Run '{requested}' not found in {parent.path}.")
        return requested
    latest = parent.attrs.get("latest")
    if isinstance(latest, bytes):
        latest = latest.decode("utf-8", "ignore")
    if isinstance(latest, str) and latest in parent:
        return latest
    keys_fn = getattr(parent, "group_keys", None)
    keys = list(keys_fn()) if callable(keys_fn) else list(parent.keys())
    if not keys:
        raise ValueError(f"No runs found in {parent.path}.")
    return sorted(keys)[-1]


def _load_metrics(run_group: zarr.Group) -> dict[str, np.ndarray]:
    def _maybe(name: str) -> Optional[np.ndarray]:
        return run_group[name][:] if name in run_group else None

    return {
        "frame_indices": run_group["frame_indices"][:],
        "valid_mask": run_group["valid_mask"][:],
        "fish_centroid_px": run_group["fish_centroid_px"][:],
        "chaser_position_px": run_group["chaser_position_px"][:],
        "distance_px": _maybe("distance_px"),
        "distance_mm": _maybe("distance_mm"),
    }


def _prepare_axes(title: str, size: float) -> tuple[plt.Figure, plt.Axes, plt.Axes, plt.Axes]:
    fig, axes = plt.subplots(3, 1, figsize=(12, 14), gridspec_kw={"height_ratios": [4, 1.5, 1.5]})
    ax_pos, ax_px, ax_mm = axes
    fig.suptitle(title)

    ax_pos.set_xlabel("X position (px)")
    ax_pos.set_ylabel("Y position (px)")
    if np.isfinite(size) and size > 0:
        ax_pos.set_xlim(0, size)
        ax_pos.set_ylim(0, size)
    ax_pos.set_aspect("equal")
    ax_pos.grid(True, linestyle="--", alpha=0.3)

    ax_px.set_xlabel("Frame index")
    ax_px.set_ylabel("Distance (px)")
    ax_px.grid(True, linestyle="--", alpha=0.3)

    ax_mm.set_xlabel("Frame index")
    ax_mm.set_ylabel("Distance (mm)")
    ax_mm.grid(True, linestyle="--", alpha=0.3)

    return fig, ax_pos, ax_px, ax_mm


def _plot_positions(
    ax: plt.Axes,
    fish: np.ndarray,
    chaser: np.ndarray,
    valid_mask: np.ndarray,
    *,
    sample_rate: int,
) -> None:
    fish_valid = fish[valid_mask]
    chaser_valid = chaser[valid_mask]

    ax.scatter(
        fish_valid[::sample_rate, 0],
        fish_valid[::sample_rate, 1],
        s=12,
        c="tab:blue",
        alpha=0.3,
        label="Fish",
    )
    ax.scatter(
        chaser_valid[::sample_rate, 0],
        chaser_valid[::sample_rate, 1],
        s=12,
        c="tab:red",
        alpha=0.3,
        label="Chaser",
    )

    ax.legend(loc="upper right")


def visualize_chaser_vs_fish(
    zarr_path: Path,
    metrics_run: Optional[str],
    *,
    sample_rate: int,
    arena_radius: Optional[float],
    show_distance: bool,
    title: str,
    output_path: Optional[Path],
    console: Console,
) -> None:
    root = zarr.open(str(zarr_path), mode="r")
    analysis = root.require_group("analysis")
    metrics_parent = analysis.require_group("chaser_fish_metrics")

    run_name = _resolve_run(metrics_parent, metrics_run)
    console.print(f"Using metrics run: [cyan]{run_name}[/cyan]")
    run_group = metrics_parent[run_name]

    data = _load_metrics(run_group)
    fish = data["fish_centroid_px"].astype(np.float64)
    chaser = data["chaser_position_px"].astype(np.float64)
    valid = data["valid_mask"].astype(bool)

    if fish.shape[0] != chaser.shape[0]:
        raise ValueError("fish and chaser arrays have different lengths")

    finite_max = np.nanmax(fish) if np.isfinite(fish).any() else np.nanmax(chaser)
    fig, ax_pos, ax_px, ax_mm = _prepare_axes(title or DEFAULT_TITLE, float(finite_max) if np.isfinite(finite_max) else 0.0)

    if arena_radius and arena_radius > 0:
        arena = Circle((arena_radius, arena_radius), radius=arena_radius, fill=False, linestyle="--", alpha=0.5)
        ax_pos.add_patch(arena)

    _plot_positions(ax_pos, fish, chaser, valid, sample_rate=sample_rate)

    distance_px = data.get("distance_px")
    distance_mm = data.get("distance_mm")
    if distance_px is not None:
        dist_px_series = distance_px[valid]
        frames = np.arange(dist_px_series.size)
        ax_px.plot(frames[::sample_rate], dist_px_series[::sample_rate], color="tab:green", alpha=0.7)
    else:
        ax_px.set_visible(False)

    if distance_mm is not None and np.isfinite(distance_mm).any():
        dist_mm_series = distance_mm[valid]
        frames_mm = np.arange(dist_mm_series.size)
        ax_mm.plot(frames_mm[::sample_rate], dist_mm_series[::sample_rate], color="tab:purple", alpha=0.7)
    else:
        ax_mm.set_visible(False)

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        console.print(f"[green]✓[/green] Figure saved to {output_path}")
    else:
        plt.show()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Overlay fish vs chaser positions from chaser_fish_metrics runs.",
    )
    parser.add_argument("zarr", type=Path, help="Path to Palette Zarr archive")
    parser.add_argument("--run", dest="metrics_run", help="analysis/chaser_fish_metrics/<run> (default: latest)")
    parser.add_argument("--sample-rate", type=int, default=5, help="Plot every Nth frame (default: 5)")
    parser.add_argument("--arena-radius", type=float, help="Optional arena radius in pixels for reference circle")
    parser.add_argument("--show-distance", action="store_true", help="Overlay distance over time on secondary axis")
    parser.add_argument("--title", help="Custom plot title")
    parser.add_argument("--output", type=Path, help="Save figure to disk instead of showing")
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    console = Console()

    visualize_chaser_vs_fish(
        zarr_path=args.zarr,
        metrics_run=args.metrics_run,
        sample_rate=max(1, args.sample_rate),
        arena_radius=args.arena_radius,
        show_distance=args.show_distance,
        title=args.title or DEFAULT_TITLE,
        output_path=args.output,
        console=console,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
