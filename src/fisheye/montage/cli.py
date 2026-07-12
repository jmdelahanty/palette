"""Command-line interface for registry-driven montage workflows."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from fisheye.registry.db import RegistryPaths

from .profiles import PLOT_PROFILES
from .workflow import build_registry_visualization_montages


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create visualization montages from a read-only Palette registry query.",
    )
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--protocol-name")
    parser.add_argument("--recording-id", action="append", default=[])
    parser.add_argument("--recording-id-contains")
    parser.add_argument("--path-contains")
    parser.add_argument("--arena-id", action="append", default=[])
    parser.add_argument(
        "--chaser-behavior",
        action="append",
        default=[],
        help="Require this configured behavior in the same stimulus run; repeatable.",
    )
    parser.add_argument("--chaser-count", type=int)
    parser.add_argument("--zarr-use", default="analysis")
    parser.add_argument("--status", default="active")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--all-recordings", action="store_true")
    parser.add_argument("--plot-type", action="append", choices=sorted(PLOT_PROFILES), default=[])
    parser.add_argument("--list-plot-types", action="store_true")
    parser.add_argument("--chaser-distance-run")
    parser.add_argument("--detection-occupancy-run")
    parser.add_argument("--egocentric-component")
    parser.add_argument("--escape-freeze-component")
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--tile-width", type=int, default=600)
    parser.add_argument("--max-image-height", type=int)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.list_plot_types:
        for profile in PLOT_PROFILES.values():
            required = ",".join(profile.required_parameters) or "none"
            print(f"{profile.profile_id}\t{profile.label}\trequires={required}")
        return 0
    if args.output_dir is None:
        raise SystemExit("--output-dir is required unless --list-plot-types is used.")
    registry_path = args.registry or RegistryPaths.from_env(Path.cwd()).path
    manifest = build_registry_visualization_montages(
        registry_path=registry_path,
        output_dir=args.output_dir,
        plot_types=args.plot_type,
        protocol_name=args.protocol_name,
        recording_ids=args.recording_id,
        recording_id_contains=args.recording_id_contains,
        path_contains=args.path_contains,
        arena_ids=args.arena_id,
        chaser_behaviors=args.chaser_behavior,
        chaser_count=args.chaser_count,
        zarr_use=str(args.zarr_use),
        status=str(args.status),
        limit=args.limit,
        all_recordings=bool(args.all_recordings),
        chaser_distance_run=args.chaser_distance_run,
        detection_occupancy_run=args.detection_occupancy_run,
        egocentric_component=args.egocentric_component,
        escape_freeze_component=args.escape_freeze_component,
        columns=int(args.columns),
        tile_width=int(args.tile_width),
        max_image_height=args.max_image_height,
        fail_on_missing=not bool(args.allow_missing),
        overwrite=bool(args.overwrite),
        dry_run=bool(args.dry_run),
    )
    print(f"recording_count\t{manifest['recording_count']}")
    print(f"missing_artifact_count\t{manifest['missing_artifact_count']}")
    for output in manifest["outputs"]:
        print(f"output\t{output['plot_type']}\t{output['path']}")
    if "manifest_path" in manifest:
        print(f"manifest_path\t{manifest['manifest_path']}")
    return 0
