#!/usr/bin/env python3
"""
Summarize ID-assignment provenance and consistency within a Palette Zarr archive.

For each run in ``id_assignment_runs`` this script reports the detection sources,
array lengths, ROI counts, and highlights mismatches between detection rows and
stored `detection_ids`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import zarr
from rich.console import Console
from rich.table import Table


def _sorted_group_keys(group: Optional[zarr.Group]) -> List[str]:
    if group is None:
        return []
    keys_fn = getattr(group, "group_keys", None)
    try:
        keys = list(keys_fn()) if callable(keys_fn) else []
    except Exception:
        keys = []
    return sorted(key for key in keys if isinstance(key, str))


def _resolve_detection_group(
    root: zarr.Group, assign_group: zarr.Group
) -> Tuple[Optional[str], Optional[zarr.Group]]:
    assignment_source = assign_group.attrs.get("assignment_source")
    refined_run = assign_group.attrs.get("source_refined_run")
    detect_run = assign_group.attrs.get("source_detect_run")

    if assignment_source and assignment_source.startswith("refined") and refined_run:
        refined_parent = root.get("refined_detect_runs")
        if isinstance(refined_parent, zarr.Group) and refined_run in refined_parent:
            refined_group = refined_parent[refined_run]
            if "interpolated" in refined_group:
                return f"refined_detect_runs/{refined_run}/interpolated", refined_group["interpolated"]
            return f"refined_detect_runs/{refined_run}", refined_group

    if detect_run:
        detect_parent = root.get("detect_runs")
        if isinstance(detect_parent, zarr.Group) and detect_run in detect_parent:
            return f"detect_runs/{detect_run}", detect_parent[detect_run]

    return None, None


def _select_run_names(run_names: Sequence[str], latest: Optional[str], limit: Optional[int]) -> List[str]:
    if limit is not None and limit <= 0:
        return []
    selected = list(run_names)
    if limit is not None:
        selected = selected[-limit:]
    if latest and latest in run_names and latest not in selected:
        selected.append(latest)
    if latest and latest in selected:
        selected = [name for name in selected if name != latest] + [latest]
    return selected


def show_id_assignment_runs(zarr_path: Path, limit: Optional[int]) -> None:
    console = Console()
    console.print(f"[bold]Inspecting ID assignment runs in:[/bold] {zarr_path}")

    root = zarr.open(str(zarr_path), mode="r")
    if "id_assignment_runs" not in root:
        console.print("[yellow]Archive contains no id_assignment_runs group.[/yellow]")
        return

    parent = root["id_assignment_runs"]
    run_names = _sorted_group_keys(parent)
    if not run_names:
        console.print("[yellow]No ID assignment runs recorded.[/yellow]")
        return

    latest = parent.attrs.get("latest")
    selected_runs = _select_run_names(run_names, latest, limit)

    table = Table(title="ID Assignment Runs", show_lines=False, box=None)
    table.add_column("Run", style="cyan")
    table.add_column("Detect Run", style="green")
    table.add_column("Refined Run", style="green")
    table.add_column("Assignment", style="magenta")
    table.add_column("IDs", justify="right")
    table.add_column("Detections", justify="right")
    table.add_column("Δ", justify="right")
    table.add_column("Masks", justify="right")
    table.add_column("Source Group", overflow="fold")
    table.add_column("Latest", style="bold", no_wrap=True)
    table.add_column("Notes", style="yellow", overflow="fold")

    for run_name in selected_runs:
        run_group = parent.get(run_name)
        if not isinstance(run_group, zarr.Group):
            continue

        detect_run = run_group.attrs.get("source_detect_run")
        refined_run = run_group.attrs.get("source_refined_run")
        assignment_source = run_group.attrs.get("assignment_source", "unknown")
        num_masks = run_group.attrs.get("num_masks")

        detection_ids_len = 0
        if "detection_ids" in run_group:
            detection_ids_len = int(run_group["detection_ids"].shape[0])

        detect_path, detection_group = _resolve_detection_group(root, run_group)
        detection_rows: Optional[int] = None
        notes: List[str] = []

        if detection_group is None:
            notes.append("Detection source unavailable")
        else:
            if isinstance(detection_group, zarr.Group):
                if "bbox_norm_coords" in detection_group:
                    detection_rows = int(detection_group["bbox_norm_coords"].shape[0])
                else:
                    notes.append("bbox_norm_coords missing")
            elif isinstance(detection_group, zarr.Array):
                detection_rows = int(detection_group.shape[0])
            else:
                notes.append("Unexpected detection group type")

        delta_display = "—"
        if detection_rows is not None:
            delta = detection_ids_len - detection_rows
            delta_display = f"{delta:+d}"
            if delta != 0:
                notes.append("Length mismatch")

        latest_flag = "★" if latest and run_name == latest else ""

        table.add_row(
            run_name,
            detect_run or "—",
            refined_run or "—",
            str(assignment_source),
            str(detection_ids_len),
            str(detection_rows) if detection_rows is not None else "—",
            delta_display,
            str(num_masks) if num_masks is not None else "—",
            detect_path or "—",
            latest_flag,
            "; ".join(notes) if notes else "",
        )

    console.print(table)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show provenance and consistency of ID assignment runs."
    )
    parser.add_argument("zarr_path", type=Path, help="Path to Palette Zarr archive.")
    parser.add_argument(
        "--limit",
        type=int,
        help="Only display the most recent N runs (latest run is always included).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    show_id_assignment_runs(args.zarr_path, args.limit)


if __name__ == "__main__":  # pragma: no cover
    main()
