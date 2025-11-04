#!/usr/bin/env python3
"""Report NaNs in smoothed heading/acceleration arrays for movement runs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import zarr


SECTIONS: Iterable[tuple[str, str]] = (
    ("heading_degrees", "heading_per_second_degrees"),
    ("smoothed_heading_degrees", "smoothed_heading_radians"),
    ("acceleration_mm", "smoothed_acceleration_mm"),
    ("acceleration_px", "smoothed_acceleration_px"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check movement run tracks for NaNs in smoothed heading/acceleration arrays."
    )
    parser.add_argument("store", type=Path, help="Path to Palette Zarr store")
    parser.add_argument(
        "--run",
        dest="run_name",
        help=(
            "Movement run name (accepts online/<run> or offline/<run>). Defaults to the latest offline run."
        ),
    )
    return parser.parse_args()


def resolve_run(root: zarr.Group, name: str | None) -> tuple[zarr.Group, str]:
    try:
        movement_parent = root["analysis"]["movement_runs"]
    except KeyError as exc:
        raise SystemExit("analysis/movement_runs missing from store") from exc

    def pick_latest(group: zarr.Group, prefix: str) -> tuple[zarr.Group, str]:
        latest = group.attrs.get("latest")
        if not latest:
            raise SystemExit(f"No '{prefix}' runs found and --run not supplied")
        if latest not in group:
            raise SystemExit(f"'{prefix}' latest run '{latest}' not found")
        return group[latest], f"{prefix}/{latest}"

    if name:
        if "/" in name:
            prefix, tail = name.split("/", 1)
            subgroup = movement_parent.get(prefix)
            if subgroup is None or tail not in subgroup:
                raise SystemExit(f"Run '{name}' not found under analysis/movement_runs")
            return subgroup[tail], name
        for prefix in ("offline", "online"):
            subgroup = movement_parent.get(prefix)
            if subgroup is not None and name in subgroup:
                return subgroup[name], f"{prefix}/{name}"
        raise SystemExit(f"Run '{name}' not found under analysis/movement_runs")

    offline_parent = movement_parent.get("offline")
    if offline_parent is not None:
        return pick_latest(offline_parent, "offline")

    online_parent = movement_parent.get("online")
    if online_parent is not None:
        return pick_latest(online_parent, "online")

    raise SystemExit("No movement runs available")


def report_nan(array: np.ndarray) -> tuple[int, int]:
    nan_mask = np.isnan(array)
    return int(nan_mask.sum()), int(array.size)


def inspect_run(group: zarr.Group) -> None:
    tracks = group.get("tracks")
    if tracks is None:
        raise SystemExit("Run has no tracks subgroup")

    for track_name in sorted(tracks.group_keys()):
        if not track_name.startswith("id_"):
            continue
        track = tracks[track_name]
        print(f"Track {track_name}:")
        for primary, secondary in SECTIONS:
            if primary not in track:
                print(f"  {primary}: missing")
                continue
            primary_arr = np.asarray(track[primary], dtype=np.float32)
            nan_primary, total = report_nan(primary_arr)
            message = f"  {primary}: {nan_primary}/{total} NaNs"
            if nan_primary and total:
                idx = np.where(np.isnan(primary_arr))[0]
                message += f" (first <=10 indices: {idx[:10]})"
            print(message)

            if secondary in track:
                secondary_arr = np.asarray(track[secondary], dtype=np.float32)
                nan_secondary, total_secondary = report_nan(secondary_arr)
                sec_msg = f"    {secondary}: {nan_secondary}/{total_secondary} NaNs"
                if nan_secondary and total_secondary:
                    idx2 = np.where(np.isnan(secondary_arr))[0]
                    sec_msg += f" (first <=10 indices: {idx2[:10]})"
                print(sec_msg)
        print()


def main() -> None:
    args = parse_args()
    if not args.store.exists():
        raise SystemExit(f"Store path '{args.store}' does not exist")

    root = zarr.open(str(args.store), mode="r")
    run_group, run_name = resolve_run(root, args.run_name)
    print(f"Checking run: {run_name}")
    inspect_run(run_group)


if __name__ == "__main__":
    main()
