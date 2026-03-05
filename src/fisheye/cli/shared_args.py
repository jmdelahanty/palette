from __future__ import annotations

import argparse
from pathlib import Path


def add_apply_dry_run_args(
    parser: argparse.ArgumentParser,
    *,
    apply_help: str,
    dry_run_help: str,
) -> argparse._MutuallyExclusiveGroup:
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--apply", action="store_true", help=apply_help)
    group.add_argument("--dry-run", action="store_true", help=dry_run_help)
    return group


def add_log_args(
    parser: argparse.ArgumentParser,
    *,
    log_dir_help: str,
    include_no_log: bool = True,
) -> None:
    parser.add_argument("--log-dir", type=Path, help=log_dir_help)
    if include_no_log:
        parser.add_argument("--no-log", action="store_true", help="Disable JSONL logging.")


def add_registry_discovery_args(
    parser: argparse.ArgumentParser,
    *,
    source_help: str = "Discovery source: filesystem (default) or registry.",
    registry_help: str = "Path to registry SQLite database (required when --source registry).",
    rig_help: str = "Filter by rig_id (registry mode).",
    arena_help: str = "Filter by arena_id (registry mode).",
    camera_help: str = "Filter by camera_id (registry mode).",
    path_contains_help: str = "Filter zarr_path by substring (registry mode).",
    camera_flag: str = "--camera-id",
    camera_dest: str | None = None,
) -> None:
    parser.add_argument(
        "--source",
        choices=["filesystem", "registry"],
        default="filesystem",
        help=source_help,
    )
    parser.add_argument(
        "--emit-paths",
        action="store_true",
        help="Print discovered zarr paths (one per line) and exit.",
    )
    parser.add_argument("--registry", type=Path, help=registry_help)
    parser.add_argument("--rig-id", type=str, help=rig_help)
    parser.add_argument("--arena-id", type=str, help=arena_help)
    if camera_dest is None:
        parser.add_argument(camera_flag, type=str, help=camera_help)
    else:
        parser.add_argument(camera_flag, dest=camera_dest, type=str, help=camera_help)
    parser.add_argument("--path-contains", type=str, help=path_contains_help)
