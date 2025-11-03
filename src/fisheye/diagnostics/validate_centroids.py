#!/usr/bin/env python3
"""
Validate bounding-box centroid fields stored in Palette stimulus runs.

The script inspects ``analysis/stimulus_runs/<run>/tracking_data/bounding_boxes`` and
verifies that centroid columns are finite wherever the bounding box geometry is valid.
It surfaces rows where ``centroid_x``/``centroid_y`` are NaN despite finite ``width`` and
``height``, and optionally checks that the centroids match the expected midpoint within a
user-specified tolerance.

Example
-------
python -m fisheye.diagnostics.validate_centroids path/to/archive.zarr \\
    --run stimulus_20250101_120000 --tolerance 1e-3 --max-report 10 --strict
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import zarr
from rich.console import Console
from rich.table import Table

from fisheye.analysis.chaser_state_interpolator import load_structured_dataset


@dataclass(frozen=True)
class Issue:
    """Represents a single centroid anomaly for reporting."""

    index: int
    reason: str
    payload_frame_id: Optional[int]
    payload_camera_id: Optional[int]
    box_index: Optional[int]
    centroid_x: Optional[float]
    centroid_y: Optional[float]
    width: Optional[float]
    height: Optional[float]


def _pick_field(names: Sequence[str], candidates: Iterable[str]) -> Optional[str]:
    """Find the first matching field name from the candidate list."""
    for candidate in candidates:
        if candidate in names:
            return candidate
    return None


def _as_float(array: np.ndarray) -> np.ndarray:
    """Convert to float64 for stable math while preserving mask semantics."""
    return np.asarray(array, dtype=np.float64)


def _resolve_stimulus_run(root: zarr.Group, run_name: Optional[str]) -> Tuple[str, zarr.Group]:
    """Resolve the analysis/stimulus_runs entry to inspect."""
    if "analysis" not in root or "stimulus_runs" not in root["analysis"]:
        raise ValueError("analysis/stimulus_runs group not found in archive.")
    runs_group = root["analysis/stimulus_runs"]
    if run_name:
        if run_name not in runs_group:
            raise ValueError(f"Run '{run_name}' not found under analysis/stimulus_runs.")
        return run_name, runs_group[run_name]

    latest = runs_group.attrs.get("latest")
    if isinstance(latest, bytes):
        latest = latest.decode("utf-8", errors="ignore")
    if isinstance(latest, str) and latest in runs_group:
        return latest, runs_group[latest]

    # Fall back to alphabetical ordering if latest is unavailable.
    available = sorted(runs_group.group_keys())
    if not available:
        raise ValueError("analysis/stimulus_runs is empty; specify --run explicitly.")
    return available[-1], runs_group[available[-1]]


def _load_bounding_boxes(run_group: zarr.Group) -> np.ndarray:
    tracking = run_group.get("tracking_data")
    if tracking is None:
        raise ValueError("tracking_data group missing in selected run.")
    data, _ = load_structured_dataset(tracking, "bounding_boxes")
    if data.dtype.names is None:
        raise ValueError("bounding_boxes dataset is not structured; expected named fields.")
    return data


def analyze_centroids(
    bboxes: np.ndarray,
    tolerance: float,
) -> Tuple[dict, Sequence[Issue]]:
    """Inspect centroid columns and return summary statistics plus detailed issues."""
    fields = bboxes.dtype.names or ()
    required_centroid_fields = {"centroid_x", "centroid_y"}
    if not required_centroid_fields.issubset(fields):
        raise ValueError("bounding_boxes lacks centroid_x/centroid_y fields; rerun import.")

    width_name = _pick_field(fields, ("width", "width_px", "bbox_width", "w"))
    height_name = _pick_field(fields, ("height", "height_px", "bbox_height", "h"))
    x_min_name = _pick_field(fields, ("x_min", "left", "bbox_x_min"))
    y_min_name = _pick_field(fields, ("y_min", "top", "bbox_y_min"))

    if not all((width_name, height_name, x_min_name, y_min_name)):
        raise ValueError(
            "bounding_boxes missing one of the required geometry fields "
            f"(width, height, x_min, y_min). Found: {fields}"
        )

    centroid_x = _as_float(bboxes["centroid_x"])
    centroid_y = _as_float(bboxes["centroid_y"])
    width = _as_float(bboxes[width_name])
    height = _as_float(bboxes[height_name])
    x_min = _as_float(bboxes[x_min_name])
    y_min = _as_float(bboxes[y_min_name])

    total_rows = int(bboxes.shape[0])
    geom_finite = np.isfinite(width) & np.isfinite(height)
    geom_positive = (width > 0.0) & (height > 0.0)
    geom_valid = geom_finite & geom_positive

    centroid_finite = np.isfinite(centroid_x) & np.isfinite(centroid_y)
    unexpected_nan_mask = geom_valid & ~centroid_finite

    expected_x = x_min + (width * 0.5)
    expected_y = y_min + (height * 0.5)
    deviation_mask = centroid_finite & (
        (np.abs(centroid_x - expected_x) > tolerance)
        | (np.abs(centroid_y - expected_y) > tolerance)
    )

    issues: list[Issue] = []
    payload_frame_name = _pick_field(fields, ("payload_frame_id", "camera_frame_id"))
    payload_cam_name = _pick_field(fields, ("payload_camera_id", "camera_id"))
    box_index_name = _pick_field(fields, ("box_index_in_payload", "box_index", "index"))

    def _fetch_optional(name: Optional[str], idx: int) -> Optional[float]:
        if name is None:
            return None
        return float(bboxes[name][idx])

    for idx in np.flatnonzero(unexpected_nan_mask):
        issues.append(
            Issue(
                index=int(idx),
                reason="centroid is NaN with finite width/height",
                payload_frame_id=int(bboxes[payload_frame_name][idx]) if payload_frame_name else None,
                payload_camera_id=int(bboxes[payload_cam_name][idx]) if payload_cam_name else None,
                box_index=int(bboxes[box_index_name][idx]) if box_index_name else None,
                centroid_x=None,
                centroid_y=None,
                width=float(width[idx]),
                height=float(height[idx]),
            )
        )

    for idx in np.flatnonzero(deviation_mask):
        issues.append(
            Issue(
                index=int(idx),
                reason="centroid deviates from bbox midpoint beyond tolerance",
                payload_frame_id=int(bboxes[payload_frame_name][idx]) if payload_frame_name else None,
                payload_camera_id=int(bboxes[payload_cam_name][idx]) if payload_cam_name else None,
                box_index=int(bboxes[box_index_name][idx]) if box_index_name else None,
                centroid_x=float(centroid_x[idx]),
                centroid_y=float(centroid_y[idx]),
                width=float(width[idx]),
                height=float(height[idx]),
            )
        )

    summary = {
        "total_rows": total_rows,
        "geom_valid_rows": int(geom_valid.sum()),
        "unexpected_nan_rows": int(unexpected_nan_mask.sum()),
        "centroid_deviation_rows": int(deviation_mask.sum()),
        "tolerance": tolerance,
    }
    return summary, issues


def _build_table(console: Console, issues: Sequence[Issue], max_report: int) -> None:
    table = Table(
        "#",
        "Row",
        "Frame",
        "Camera",
        "BoxIdx",
        "Reason",
        "Centroid (x,y)",
        "Width",
        "Height",
    )
    for idx, issue in enumerate(issues[:max_report], start=1):
        centroid_str = (
            "NaN"
            if issue.centroid_x is None or issue.centroid_y is None
            else f"{issue.centroid_x:.3f}, {issue.centroid_y:.3f}"
        )
        table.add_row(
            str(idx),
            str(issue.index),
            str(issue.payload_frame_id) if issue.payload_frame_id is not None else "—",
            str(issue.payload_camera_id) if issue.payload_camera_id is not None else "—",
            str(issue.box_index) if issue.box_index is not None else "—",
            issue.reason,
            centroid_str,
            f"{issue.width:.3f}" if issue.width is not None else "—",
            f"{issue.height:.3f}" if issue.height is not None else "—",
        )
    console.print(table)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path, help="Path to the Palette Zarr archive.")
    parser.add_argument(
        "--run",
        dest="run_name",
        help="Name under analysis/stimulus_runs/. Defaults to the archive's 'latest' attribute if present.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-3,
        help="Maximum absolute difference between centroid and bbox midpoint before flagging (default: 1e-3).",
    )
    parser.add_argument(
        "--max-report",
        type=int,
        default=20,
        help="Maximum number of detailed issue rows to display (default: 20).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with status 1 if any anomalies are detected.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    console = Console()

    root = zarr.open(str(args.zarr_path), mode="r")
    run_name, run_group = _resolve_stimulus_run(root, args.run_name)
    bboxes = _load_bounding_boxes(run_group)
    if bboxes.size == 0:
        console.print(f"[yellow]bounding_boxes dataset is empty for run '{run_name}'.[/yellow]")
        raise SystemExit(0)

    summary, issues = analyze_centroids(bboxes, tolerance=args.tolerance)

    console.print(
        f"[bold]Centroid validation for analysis/stimulus_runs/{run_name}/tracking_data/bounding_boxes[/bold]"
    )
    console.print(f"Total rows: [cyan]{summary['total_rows']}[/cyan]")
    console.print(f"Rows with finite geometry: [cyan]{summary['geom_valid_rows']}[/cyan]")
    console.print(
        f"Unexpected centroid NaNs: "
        f"[{'red' if summary['unexpected_nan_rows'] else 'green'}]{summary['unexpected_nan_rows']}[/]"
    )
    console.print(
        f"Centroid midpoint deviations (> {summary['tolerance']:.2e}): "
        f"[{'red' if summary['centroid_deviation_rows'] else 'green'}]{summary['centroid_deviation_rows']}[/]"
    )

    if issues:
        console.print()
        console.print(f"[bold]Sample issues (showing up to {args.max_report})[/bold]")
        _build_table(console, issues, args.max_report)

    if args.strict and issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
