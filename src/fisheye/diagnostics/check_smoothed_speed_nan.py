#!/usr/bin/env python3
"""Report NaNs in smoothed_speed_mm for movement runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import zarr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check movement run tracks for NaNs in smoothed_speed_mm."
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


def inspect_run(group: zarr.Group) -> None:
    tracks = group.get("tracks")
    if tracks is None:
        raise SystemExit("Run has no tracks subgroup")

    for track_name in sorted(tracks.group_keys()):
        if not track_name.startswith("id_"):
            continue
        track = tracks[track_name]
        if "smoothed_speed_mm" not in track:
            print(f"{track_name}: smoothed_speed_mm missing")
            continue
        values = np.asarray(track["smoothed_speed_mm"], dtype=np.float32)
        nan_mask = np.isnan(values)
        nan_count = int(nan_mask.sum())
        total = int(values.size)
        if nan_count:
            idx = np.where(nan_mask)[0]
            print(
                f"{track_name}: {nan_count}/{total} NaNs. First indices (<=20): {idx[:20]}"
            )
        else:
            print(f"{track_name}: 0/{total} NaNs")


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
