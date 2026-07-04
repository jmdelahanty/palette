#!/usr/bin/env python3
"""Rescan Zarrs and update the registry with latest metadata."""

from __future__ import annotations

from fisheye.shared.zarr_discovery import iter_filesystem_zarrs as _iter_zarr
import argparse
from pathlib import Path
from typing import Iterable, List, Optional

from fisheye.registry.db import Registry, RegistryPaths


def _read_file_list(path: Path) -> List[Path]:
    items: List[Path] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        items.append(Path(value))
    return items


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="Recording roots or Zarr paths.")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan for Zarrs.")
    parser.add_argument(
        "--file-list",
        type=Path,
        action="append",
        help="Text file with one zarr path per line (comments with # allowed).",
    )
    parser.add_argument("--registry", type=Path, help="Optional registry SQLite path.")

    args = parser.parse_args(argv)

    roots: List[Path] = []
    if args.file_list:
        for path in args.file_list:
            roots.extend(_read_file_list(path))
    roots.extend(args.paths)
    if not roots:
        raise SystemExit("No paths provided.")

    registry_path = args.registry or RegistryPaths.from_env(Path.cwd()).path
    registry = Registry(registry_path)
    print(f"Registry: {registry_path}")

    count = 0
    for zarr_path in _iter_zarr(roots, args.recursive):
        try:
            dataset_id = registry.scan_zarr(zarr_path)
        except Exception as exc:
            print(f"{zarr_path}: error {exc}")
            continue
        if dataset_id:
            count += 1
            print(f"{zarr_path}: {dataset_id}")

    reconcile = registry.reconcile_missing_datasets(scope_paths=roots)

    registry.close()
    print(f"Updated {count} dataset(s).")
    print(
        "Marked {marked} missing dataset(s) (checked {checked} registry row(s)).".format(
            marked=reconcile.get("marked_missing", 0),
            checked=reconcile.get("checked", 0),
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
