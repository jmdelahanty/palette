#!/usr/bin/env python3
"""
List pipeline run durations stored inside a Palette Zarr archive.

The script walks all groups named ``*_runs`` (recursively) and reports the
duration recorded for each run via ``duration_seconds`` or related attributes.
This makes it easy to spot unusually slow stages without opening the full
inspector.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import zarr
from rich.console import Console
from rich.table import Table

DURATION_KEYS: Tuple[str, ...] = (
    "duration_seconds",
    "inference_duration_seconds",
    "import_duration_seconds",
    "processing_duration_seconds",
    "training_duration_seconds",
)


def _sorted_group_keys(group: Optional[zarr.Group]) -> List[str]:
    if group is None:
        return []
    keys_fn = getattr(group, "group_keys", None)
    try:
        keys = list(keys_fn()) if callable(keys_fn) else []
    except Exception:
        keys = []
    return sorted(key for key in keys if isinstance(key, str))


def _find_run_groups(root: zarr.Group) -> List[Tuple[str, zarr.Group]]:
    results: List[Tuple[str, zarr.Group]] = []
    stack: List[Tuple[str, zarr.Group]] = [("", root)]
    while stack:
        prefix, group = stack.pop()
        for name in _sorted_group_keys(group):
            child = group[name]
            if not isinstance(child, zarr.Group):
                continue
            path = f"{prefix}/{name}".lstrip("/")
            if name.endswith("_runs"):
                results.append((path, child))
            stack.append((path, child))
    return sorted(results, key=lambda item: item[0])


def _extract_duration(attrs: zarr.AttrMapping) -> Tuple[Optional[float], Optional[str]]:
    for key in DURATION_KEYS:
        raw = attrs.get(key)
        if raw is None:
            continue
        if isinstance(raw, bool):
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isnan(value) or math.isinf(value):
            continue
        return value, key
    return None, None


def _format_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "—"
    if seconds < 1.0:
        return f"{seconds * 1000:.0f} ms"
    hours, rem = divmod(int(seconds), 3600)
    minutes, secs = divmod(rem, 60)
    remainder = seconds - int(seconds)
    frac = f"{remainder:.2f}"[1:] if remainder >= 0.005 else ""
    if hours:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s{frac}"
    if minutes:
        return f"{minutes:d}m {secs:02d}s{frac}"
    return f"{seconds:.2f}s"


def _select_runs(run_names: Sequence[str], latest: Optional[str], latest_only: bool) -> List[str]:
    if not run_names:
        return []
    if not latest_only:
        return list(run_names)
    if latest and latest in run_names:
        return [latest]
    return [run_names[-1]]


def show_run_durations(zarr_path: Path, latest_only: bool) -> None:
    console = Console()
    console.print(f"[bold]Inspecting runs in:[/bold] {zarr_path}")

    root = zarr.open(str(zarr_path), mode="r")
    run_groups = _find_run_groups(root)
    if not run_groups:
        console.print("[yellow]No *_runs groups found in archive.[/yellow]")
        return

    table = Table(title="Run Durations", show_lines=False, box=None)
    table.add_column("Stage", style="cyan", no_wrap=True)
    table.add_column("Run", style="green")
    table.add_column("Duration", style="yellow", justify="right")
    table.add_column("Source Attr", style="magenta", no_wrap=True)
    table.add_column("Latest", style="bold")

    for path, run_group in run_groups:
        run_names = _sorted_group_keys(run_group)
        latest = run_group.attrs.get("latest")
        for run_name in _select_runs(run_names, latest, latest_only):
            run = run_group.get(run_name)
            if not isinstance(run, zarr.Group):
                continue
            duration, key = _extract_duration(run.attrs)
            duration_display = _format_duration(duration)
            is_latest = "★" if latest and run_name == latest else ""
            table.add_row(path, run_name, duration_display, key or "—", is_latest)

    if table.row_count == 0:
        console.print("[yellow]No run durations recorded in archive.[/yellow]")
        return

    console.print(table)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize duration metadata for runs stored in a Palette Zarr archive."
    )
    parser.add_argument("zarr_path", type=Path, help="Path to Palette Zarr archive.")
    parser.add_argument(
        "--latest-only",
        action="store_true",
        help="Only show the latest run for each stage (if available).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    show_run_durations(args.zarr_path, latest_only=bool(args.latest_only))


if __name__ == "__main__":  # pragma: no cover
    main()
