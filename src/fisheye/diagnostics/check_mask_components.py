"""Inspect YOLO eye-mask runs for disjoint connected components.

This utility loads a mask run (typically the raw YOLO output) and reports any
ROIs whose per-eye channels contain more than one substantial connected
component.  Use it to understand how often the segmentation contains stray
fragments that refinement may have to correct.

Example
-------
python -m fisheye.diagnostics.check_mask_components data.zarr \
    --eye-run eye_masks_2025-10-20_17-52-37 \
    --min-area 25 \
    --max-report 20
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import zarr
from rich.console import Console
from rich.table import Table
from skimage import measure


def _load_eye_run(root: zarr.Group, run_name: str | None) -> Tuple[str, zarr.Group]:
    parent = root["eye_masks_runs"]
    if run_name:
        if run_name not in parent:
            raise ValueError(f"eye_masks_runs/{run_name} not found in archive")
        return run_name, parent[run_name]
    latest = parent.attrs.get("latest")
    if latest is None:
        raise ValueError("eye_masks_runs has no 'latest' attribute; specify --eye-run explicitly.")
    return latest, parent[latest]


def _components(mask: np.ndarray, min_area: int) -> List[int]:
    """Return areas of connected components with area >= min_area."""
    labeled = measure.label(mask.astype(bool), connectivity=1)
    if labeled.max() == 0:
        return []
    labels, counts = np.unique(labeled, return_counts=True)
    areas = [int(count) for label, count in zip(labels, counts) if label != 0 and count >= min_area]
    areas.sort(reverse=True)
    return areas


def analyze_masks(
    masks: np.ndarray,
    min_area: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Find channel-level and union-level disjointness."""
    disjoint_channels: List[Dict[str, object]] = []
    disjoint_union: List[Dict[str, object]] = []
    total_rois, channels, _, _ = masks.shape

    for roi_idx in range(total_rois):
        union_mask = np.any(masks[roi_idx] > 0, axis=0)
        union_areas = _components(union_mask, min_area)
        if len(union_areas) > 1:
            disjoint_union.append(
                {"roi": roi_idx, "areas": union_areas, "total_area": int(np.sum(union_mask))}
            )

        for channel_idx in range(channels):
            channel_mask = masks[roi_idx, channel_idx] > 0
            channel_areas = _components(channel_mask, min_area)
            if len(channel_areas) > 1:
                disjoint_channels.append(
                    {
                        "roi": roi_idx,
                        "channel": channel_idx,
                        "areas": channel_areas,
                        "total_area": int(np.sum(channel_mask)),
                    }
                )

    return disjoint_channels, disjoint_union


def summarize(
    disjoint_channels: List[Dict[str, object]],
    disjoint_union: List[Dict[str, object]],
    total_rois: int,
    eye_run: str,
    min_area: int,
    max_report: int,
    console: Console,
) -> None:
    console.print(
        f"[bold]Disjoint-component analysis for eye_masks_runs/{eye_run}[/bold] "
        f"(channels checked against min_area={min_area})"
    )
    console.print(f"Total ROIs analysed: [cyan]{total_rois}[/cyan]")
    console.print(
        f"ROIs with disjoint channels: [red]{len(disjoint_channels)}[/red] "
        f"({len(disjoint_channels)/total_rois:.2%})"
    )
    console.print(
        f"ROIs with disjoint union mask: [red]{len(disjoint_union)}[/red] "
        f"({len(disjoint_union)/total_rois:.2%})"
    )

    if disjoint_channels:
        console.print("\n[bold]Examples – per-channel fragments[/bold]")
        table = Table("#", "ROI", "Channel", "Areas (px)", "Total")
        for idx, entry in enumerate(disjoint_channels[:max_report], start=1):
            areas_str = ", ".join(str(a) for a in entry["areas"])
            table.add_row(
                str(idx),
                str(entry["roi"]),
                str(entry["channel"]),
                areas_str,
                str(entry["total_area"]),
            )
        console.print(table)

    if disjoint_union:
        console.print("\n[bold]Examples – union mask fragments[/bold]")
        table = Table("#", "ROI", "Areas (px)", "Union Area")
        for idx, entry in enumerate(disjoint_union[:max_report], start=1):
            areas_str = ", ".join(str(a) for a in entry["areas"])
            table.add_row(
                str(idx),
                str(entry["roi"]),
                areas_str,
                str(entry["total_area"]),
            )
        console.print(table)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path, help="Path to Palette Zarr archive")
    parser.add_argument(
        "--eye-run",
        help="Specific eye mask run name to inspect (default: latest)",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=25,
        help="Minimum component area (pixels) to consider when counting fragments (default: 25)",
    )
    parser.add_argument(
        "--max-report",
        type=int,
        default=20,
        help="Maximum number of detailed rows to show for each table (default: 20)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    console = Console()

    if args.min_area <= 0:
        raise ValueError("min-area must be > 0")

    root = zarr.open(str(args.zarr_path), mode="r")
    eye_run, run_group = _load_eye_run(root, args.eye_run)

    masks = run_group["masks_roi"]
    total_rois = masks.shape[0]

    console.print(f"Loaded masks for run [cyan]{eye_run}[/cyan] ({total_rois} ROIs)")

    disjoint_channels, disjoint_union = analyze_masks(np.asarray(masks), args.min_area)

    summarize(
        disjoint_channels,
        disjoint_union,
        total_rois,
        eye_run,
        args.min_area,
        args.max_report,
        console,
    )


if __name__ == "__main__":
    main()
