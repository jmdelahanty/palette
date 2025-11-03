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
import json


def _resolve_enums_group(root: zarr.Group) -> Optional[zarr.Group]:
    """Find the enums group, checking both new (analysis/enums) and legacy locations."""
    analysis = root.get("analysis")
    if analysis is not None:
        enums = analysis.get("enums")
        if enums is not None:
            return enums
    return root.get("enums")


def _load_enum_mapping(root: zarr.Group, name: str) -> dict[int, str]:
    """Load enum mapping, handling both columnar and legacy structured array formats.

    Supports:
    - New columnar format: enums/{name}/id and enums/{name}/name as separate arrays
    - Legacy structured format: enums/events/{name} as structured array [('id', 'i4'), ('name', 'S128')]
    """
    enums_group = _resolve_enums_group(root)
    if enums_group is None:
        return {}

    # Check for new columnar format: enums/{name}/ is a group
    if name in enums_group:
        node = enums_group[name]
        if isinstance(node, zarr.Group):
            # Columnar format: separate id and name arrays
            if 'id' in node and 'name' in node:
                ids = node['id'][:]
                names = node['name'][:]
                mapping = {}
                for enum_id, enum_name in zip(ids, names):
                    # Handle various string types
                    if isinstance(enum_name, bytes):
                        enum_name = enum_name.decode("utf-8", errors="ignore").rstrip("\x00")
                    elif not isinstance(enum_name, str):
                        enum_name = str(enum_name).rstrip("\x00")
                    mapping[int(enum_id)] = enum_name
                return mapping
        elif isinstance(node, zarr.Array):
            # Legacy structured array format
            data = node[:]
            mapping = {}
            for record in data:
                enum_id = int(record["id"])
                enum_name = record["name"]
                if isinstance(enum_name, bytes):
                    enum_name = enum_name.decode("utf-8", errors="ignore").rstrip("\x00")
                mapping[enum_id] = enum_name
            return mapping

    # Check legacy location: enums/events/{name}
    events_enums = enums_group.get("events")
    if events_enums is not None and name in events_enums:
        node = events_enums[name]
        if isinstance(node, zarr.Array):
            data = node[:]
            mapping = {}
            for record in data:
                enum_id = int(record["id"])
                enum_name = record["name"]
                if isinstance(enum_name, bytes):
                    enum_name = enum_name.decode("utf-8", errors="ignore").rstrip("\x00")
                mapping[enum_id] = enum_name
            return mapping

    return {}


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


def inspect_events(
    zarr_path: Path,
    run_name: Optional[str],
    limit: Optional[int],
    to_json: bool,
) -> None:
    console = Console()
    console.print(f"[bold]Inspecting events in:[/bold] {zarr_path}")

    root = zarr.open(str(zarr_path), mode="r")
    mode_mapping = _load_enum_mapping(root, "stimulus_modes")
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

    max_rows = limit if limit is not None else events.shape[0]
    records_to_show = []
    for idx in range(min(events.shape[0], max_rows)):
        record = events[idx]
        entry = {}
        if record.dtype.names:
            for name in record.dtype.names:
                value = record[name]
                if isinstance(value, bytes):
                    value = value.decode("utf-8", errors="ignore").rstrip("\x00")
                entry[name] = value
        else:
            entry = {"value": record}
        if mode_mapping and "stimulus_mode_id" in entry:
            entry["stimulus_mode_name"] = mode_mapping.get(
                int(entry["stimulus_mode_id"]), "UNKNOWN"
            )
        entry["index"] = idx
        records_to_show.append(entry)

    if to_json:
        print(json.dumps(records_to_show, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else o))
        return

    table = Table("Index", "Fields", show_lines=False, expand=True)
    for entry in records_to_show:
        idx = entry.pop("index")
        fields = ", ".join(f"{k}={v}" for k, v in entry.items())
        table.add_row(str(idx), fields)

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
    parser.add_argument(
        "--to-json",
        action="store_true",
        help="Print the selected events as JSON instead of a table.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    inspect_events(args.zarr_path, args.run_name, args.limit, args.to_json)


if __name__ == "__main__":  # pragma: no cover
    main()
