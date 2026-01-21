"""Scan Zarr archives and register them in a local SQLite registry."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from .db import Registry, RegistryPaths, scan_paths


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan Zarr datasets and update the Palette registry.",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Zarr root(s) or directories to scan.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively discover Zarr roots under the provided paths.",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        help="Optional path to the registry SQLite file.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    registry_path = args.registry or RegistryPaths.from_env(Path.cwd()).path
    registry = Registry(registry_path)
    dataset_ids = scan_paths(registry, args.paths, recursive=args.recursive)
    registry.close()
    if dataset_ids:
        print(f"Registered {len(dataset_ids)} dataset(s).")
        for dataset_id in dataset_ids:
            print(f" - {dataset_id}")
    else:
        print("No datasets registered.")


if __name__ == "__main__":
    main()
