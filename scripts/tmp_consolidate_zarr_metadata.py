#!/usr/bin/env python3
"""Temporary helper to reconsolidate Zarr v3 metadata for one or more archives."""

from __future__ import annotations

import argparse
from pathlib import Path

import zarr


def _iter_zarr(paths: list[Path], recursive: bool) -> list[Path]:
    discovered: set[Path] = set()
    for raw_path in paths:
        path = raw_path.expanduser()
        if path.is_dir() and path.suffix == ".zarr":
            discovered.add(path.resolve())
            continue
        if not path.exists():
            continue
        if recursive:
            for candidate in path.rglob("*.zarr"):
                discovered.add(candidate.resolve())
        else:
            for candidate in path.glob("*.zarr"):
                discovered.add(candidate.resolve())
            for candidate in path.glob("*/zarr/*.zarr"):
                discovered.add(candidate.resolve())
    return sorted(discovered)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="Zarr paths or parent directories.")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan for *.zarr.")
    parser.add_argument("--dry-run", action="store_true", help="Print candidate paths only.")
    parser.add_argument("--strict", action="store_true", help="Return non-zero when no archives are found.")
    args = parser.parse_args()

    zarr_paths = _iter_zarr(args.paths, args.recursive)
    if not zarr_paths:
        print("No .zarr archives found.")
        return 2 if args.strict else 0

    print(f"Found {len(zarr_paths)} archive(s).")
    if args.dry_run:
        for path in zarr_paths:
            print(path)
        return 0

    failures = 0
    for path in zarr_paths:
        try:
            zarr.consolidate_metadata(str(path))
            print(f"ok: {path}")
        except Exception as exc:  # pragma: no cover - manual diagnostic tool
            failures += 1
            print(f"error: {path}: {exc}")

    print(f"Summary: consolidated={len(zarr_paths) - failures} failed={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
