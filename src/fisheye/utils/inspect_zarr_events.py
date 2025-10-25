#!/usr/bin/env python3
"""
Inspect stimulus events stored inside a Palette Zarr archive.

This utility loads `analysis/stimulus_runs/<run>/events` and prints each entry
with its attributes, making it easy to verify that `import_stimulus_to_zarr`
captured the expected event stream.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import zarr
from rich.console import Console
from rich.table import Table


def list_stimulus_runs(root: zarr.Group) -> list[str]:
    analysis = root.get("analysis")
    if analysis is None:
        return []
    stimulus_parent = analysis.get("stimulus_runs")
    if stimulus_parent is None:
        return []
    return sorted(
        name for name in stimulus_parent.group_keys() if isinstance(stimulus_parent.get(name), zarr.Group)
    )


def _reconstruct_structured(group: zarr.Group) -> np.ndarray:
    field_names = group.attrs.get("field_names")
    if not field_names:
        field_names = [name for name in group.array_keys()]
    columns = {}
    length = None
    for field in field_names:
        arr = group[field][:]
        columns[field] = arr
        if length is None:
            length = arr.shape[0]
    if length is None:
        return np.zeros(0, dtype=[])
    dtype_spec = []
    for field in field_names:
        col = columns[field]
        dtype_spec.append((field, col.dtype))
    structured = np.zeros(length, dtype=dtype_spec)
    for field in field_names:
        structured[field] = columns[field]
    return structured


def load_events(root: zarr.Group, run_name: str) -> tuple[np.ndarray, dict]:
    run_group = root[f"analysis/stimulus_runs/{run_name}"]
    if "events" not in run_group:
        raise ValueError(f"Run '{run_name}' does not contain an 'events' dataset.")
    node = run_group["events"]
    if isinstance(node, zarr.Array):
        events_array = node[:]
        events_attrs = dict(node.attrs)
    elif isinstance(node, zarr.Group):
        events_array = _reconstruct_structured(node)
        events_attrs = {
            key: value
            for key, value in node.attrs.items()
            if key not in {"field_names", "field_dtypes", "original_dtype"}
        }
    else:
        raise TypeError("Unsupported Zarr node type for events dataset.")
    return events_array, events_attrs


def inspect_events(zarr_path: Path, run_name: Optional[str], limit: Optional[int]) -> None:
    console = Console()
    console.print(f"[bold]Inspecting events in:[/bold] {zarr_path}")

    root = zarr.open(str(zarr_path), mode="r")
    available_runs = list_stimulus_runs(root)
    if not available_runs:
        raise ValueError("No stimulus runs found under analysis/stimulus_runs.")

    selected_run = run_name or available_runs[-1]
    if selected_run not in available_runs:
        raise ValueError(
            f"Run '{selected_run}' not found. Available runs: {', '.join(available_runs)}"
        )

    events, attrs = load_events(root, selected_run)
    console.print(
        f"[dim]Loaded {events.shape[0]} events from analysis/stimulus_runs/{selected_run}[/dim]"
    )
    if attrs:
        console.print(f"[dim]Event dataset attrs:[/dim] {attrs}")

    table = Table("Index", "Fields", show_lines=False, expand=True)
    max_rows = limit if limit is not None else events.shape[0]
    for idx in range(min(events.shape[0], max_rows)):
        record = events[idx]
        fields = ", ".join(f"{name}={record[name]}" for name in record.dtype.names or ())
        table.add_row(str(idx), fields or str(record))

    console.print(table)
    if events.shape[0] > max_rows:
        console.print(
            f"[dim]Showing first {max_rows} events out of {events.shape[0]}. Use --limit to adjust.[/dim]"
        )


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print stimulus events stored in analysis/stimulus_runs/<run>/events."
    )
    parser.add_argument("zarr_path", type=Path, help="Path to the Palette Zarr archive.")
    parser.add_argument(
        "--run-name",
        help="Specific stimulus run to inspect (default: latest).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of events to display (default: all).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    inspect_events(args.zarr_path, args.run_name, args.limit)


if __name__ == "__main__":  # pragma: no cover
    main()
