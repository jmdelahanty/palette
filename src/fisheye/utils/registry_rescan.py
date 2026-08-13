#!/usr/bin/env python3
"""Rescan Zarrs and update the registry with latest metadata."""

from __future__ import annotations

from fisheye.shared.zarr_discovery import iter_filesystem_zarrs as _iter_zarr
import argparse
from pathlib import Path
from typing import Any, Iterable, List, Optional

from fisheye.registry.db import Registry, RegistryPaths
from fisheye.shared.json_safety import write_json_atomic


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
    parser.add_argument(
        "paths", nargs="*", type=Path, help="Recording roots or Zarr paths."
    )
    parser.add_argument(
        "--recursive", action="store_true", help="Recursively scan for Zarrs."
    )
    parser.add_argument(
        "--file-list",
        type=Path,
        action="append",
        help="Text file with one zarr path per line (comments with # allowed).",
    )
    parser.add_argument("--registry", type=Path, help="Optional registry SQLite path.")
    parser.add_argument(
        "--result-json",
        type=Path,
        help="Optional machine-readable completion receipt.",
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Return nonzero when any requested Zarr cannot be scanned.",
    )

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
    scanned_count = 0
    errors: list[dict[str, Any]] = []
    updated: list[dict[str, str]] = []
    for zarr_path in _iter_zarr(roots, args.recursive):
        scanned_count += 1
        try:
            dataset_id = registry.scan_zarr(zarr_path)
        except Exception as exc:
            print(f"{zarr_path}: error {exc}")
            errors.append(
                {
                    "zarr_path": str(zarr_path.expanduser().resolve()),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            continue
        if dataset_id:
            count += 1
            updated.append(
                {
                    "zarr_path": str(zarr_path.expanduser().resolve()),
                    "dataset_id": str(dataset_id),
                }
            )
            print(f"{zarr_path}: {dataset_id}")
        elif args.fail_on_error:
            errors.append(
                {
                    "zarr_path": str(zarr_path.expanduser().resolve()),
                    "error_type": "DatasetNotRegistered",
                    "error": "Registry scan returned no dataset ID.",
                }
            )

    if args.fail_on_error and scanned_count == 0:
        errors.append(
            {
                "zarr_path": None,
                "error_type": "NoZarrDiscovered",
                "error": "No requested Zarr archive was discovered.",
            }
        )

    reconcile = registry.reconcile_missing_datasets(scope_paths=roots)

    registry.close()
    print(f"Updated {count} dataset(s).")
    print(
        "Marked {marked} missing dataset(s) (checked {checked} registry row(s)).".format(
            marked=reconcile.get("marked_missing", 0),
            checked=reconcile.get("checked", 0),
        )
    )
    result = {
        "schema_id": "palette.registry_rescan_result",
        "schema_version": 1,
        "status": "complete" if not errors else "completed_with_errors",
        "registry": str(registry_path.expanduser().resolve()),
        "requested_roots": [str(path.expanduser().resolve()) for path in roots],
        "updated_count": count,
        "scanned_count": scanned_count,
        "updated": updated,
        "errors": errors,
        "reconcile": reconcile,
    }
    if args.result_json is not None:
        write_json_atomic(args.result_json.expanduser().resolve(), result)
    return 1 if errors and args.fail_on_error else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
