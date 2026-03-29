"""Inspect stimulus events stored inside a Palette zarr archive.

This diagnostic script is useful for correlating stimulus events (protocol start,
step transitions, etc.) with chaser activity during debugging.  It prints a table
of the events recorded under ``analysis/stimulus_runs/<run>/events`` and can be
filtered by event type or frame range.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import zarr
from rich.console import Console
from rich.table import Table

from fisheye.shared.zarr_helpers import resolve_zarr_run


def _decode_strings(values: np.ndarray) -> np.ndarray:
    """Decode byte arrays (or mixed types) to plain Python strings."""
    if values.dtype.kind == "S":
        return np.char.decode(values, "utf-8", errors="ignore").astype(str)
    if values.dtype.kind == "O":
        decoded: List[str] = []
        for item in values:
            if isinstance(item, bytes):
                decoded.append(item.decode("utf-8", "ignore"))
            elif item is None:
                decoded.append("")
            else:
                decoded.append(str(item))
        return np.asarray(decoded, dtype=str)
    return values.astype(str)


def _load_enum_mapping(analysis_group: zarr.Group, name: str) -> Dict[int, str]:
    enums_parent = analysis_group.get("enums")
    if enums_parent is None or name not in enums_parent:
        return {}

    enum_group = enums_parent[name]
    if "id" not in enum_group or "name" not in enum_group:
        return {}

    ids = np.asarray(enum_group["id"], dtype=np.int64)
    names = _decode_strings(np.asarray(enum_group["name"]))
    return {int(idx): str(label) for idx, label in zip(ids, names)}


def _load_events(run_group: zarr.Group) -> Dict[str, np.ndarray]:
    if "events" not in run_group:
        raise ValueError("Stimulus run does not contain an 'events' group")

    events_group = run_group["events"]
    field_names = events_group.attrs.get("field_names")
    if not field_names:
        field_names = list(events_group.keys())

    events: Dict[str, np.ndarray] = {}
    for field in field_names:
        if field not in events_group:
            continue
        data = np.asarray(events_group[field])
        if data.dtype.kind in ("S", "O"):
            data = _decode_strings(data)
        events[field] = data

    lengths = {len(column) for column in events.values()}
    if len(lengths) != 1:
        raise ValueError("Events columns have inconsistent lengths")
    return events


def _filter_events(
    events: Dict[str, np.ndarray],
    *,
    event_types: Optional[Sequence[str]],
    camera_range: Optional[Tuple[int, int]],
    stim_range: Optional[Tuple[int, int]],
    event_mapping: Dict[int, str],
) -> List[int]:
    count = len(next(iter(events.values()))) if events else 0
    if count == 0:
        return []

    mask = np.ones(count, dtype=bool)

    if event_types:
        event_type_ids = events.get("event_type_id")
        if event_type_ids is None:
            raise ValueError("Events dataset lacks 'event_type_id' field")

        desired_ids: List[int] = []
        for token in event_types:
            token = token.strip()
            if not token:
                continue
            if token.isdigit():
                desired_ids.append(int(token))
                continue
            matches = [eid for eid, name in event_mapping.items() if name == token]
            if not matches:
                raise ValueError(f"Event type '{token}' not recognized")
            desired_ids.extend(matches)

        mask &= np.isin(event_type_ids.astype(int), desired_ids)

    if camera_range and "camera_frame_id" in events:
        start, end = camera_range
        cam = events["camera_frame_id"].astype(int)
        mask &= (cam >= start) & (cam <= end)

    if stim_range and "stimulus_frame_num" in events:
        start, end = stim_range
        stim_vals = events["stimulus_frame_num"].astype(int)
        mask &= (stim_vals >= start) & (stim_vals <= end)

    return np.flatnonzero(mask).tolist()


def _print_event_table(
    console: Console,
    events: Dict[str, np.ndarray],
    indices: Sequence[int],
    event_mapping: Dict[int, str],
    *,
    limit: int,
) -> None:
    if not indices:
        console.print("[yellow]No events matched the given filters[/yellow]")
        return

    table = Table(show_header=True)
    cam = events.get("camera_frame_id")
    stim = events.get("stimulus_frame_num")

    columns = [
        ("Index", lambda i: str(i)),
        ("Camera", lambda i: str(int(cam[i])) if cam is not None else "-"),
        ("Stimulus", lambda i: str(int(stim[i])) if stim is not None else "-"),
    ]
    for title, _ in columns:
        table.add_column(title)

    for idx in indices[:limit]:
        table.add_row(*[render(idx) for _, render in columns])

    if len(indices) > limit:
        table.caption = f"Showing first {limit} of {len(indices)} matching events"

    console.print(table)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path, help="Path to Palette zarr archive")
    parser.add_argument("--stimulus-run", help="Stimulus run to inspect (default: latest)")
    parser.add_argument("--event-type", action="append", help="Filter by event type name or id")
    parser.add_argument("--camera-range", nargs=2, type=int, metavar=("START", "END"), help="Camera frame range filter")
    parser.add_argument("--stimulus-range", nargs=2, type=int, metavar=("START", "END"), help="Stimulus frame range filter")
    parser.add_argument("--limit", type=int, default=25, help="Maximum number of events to display")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    console = Console()

    root = zarr.open(str(args.zarr_path), mode="r")
    analysis_group = root.require_group("analysis")
    run_group, run_name = resolve_zarr_run(
        root,
        ("analysis", "stimulus_runs"),
        args.stimulus_run,
        fallback_to_sorted="last",
        latest_aliases=("latest",),
        run_label="Stimulus run",
    )
    console.print(f"[bold]Inspecting stimulus run:[/bold] {run_name}")
    event_mapping = _load_enum_mapping(analysis_group, "events")
    events = _load_events(run_group)
    total = len(next(iter(events.values()))) if events else 0
    console.print(f"Loaded {total} events with fields: {', '.join(events.keys())}")

    indices = _filter_events(
        events,
        event_types=args.event_type,
        camera_range=tuple(args.camera_range) if args.camera_range else None,
        stim_range=tuple(args.stimulus_range) if args.stimulus_range else None,
        event_mapping=event_mapping,
    )
    _print_event_table(console, events, indices, event_mapping, limit=args.limit)


if __name__ == "__main__":  # pragma: no cover
    main()
