#!/usr/bin/env python3
"""Rescan Zarrs and update the registry with latest metadata."""

from __future__ import annotations

from fisheye.shared.zarr_discovery import iter_filesystem_zarrs as _iter_zarr
import argparse
from pathlib import Path
from typing import Any, List, Optional

from fisheye.registry.db import Registry, RegistryPaths
from fisheye.registry.shadow_publish import publish_registry_shadow
from fisheye.shared.json_safety import write_json_atomic


def _read_file_list(path: Path) -> List[Path]:
    items: List[Path] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        items.append(Path(value))
    return items


def _rescan_registry(
    *,
    registry_path: Path,
    roots: List[Path],
    recursive: bool,
    fail_on_error: bool,
    reconcile_step_status: bool,
) -> tuple[int, dict[str, Any]]:
    registry = Registry(registry_path)
    print(f"Registry: {registry_path}")

    count = 0
    scanned_count = 0
    errors: list[dict[str, Any]] = []
    updated: list[dict[str, str]] = []
    try:
        for zarr_path in _iter_zarr(roots, recursive):
            scanned_count += 1
            try:
                dataset_id = (
                    registry.scan_zarr(zarr_path, include_step_status=True)
                    if reconcile_step_status
                    else registry.scan_zarr(zarr_path)
                )
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
            elif fail_on_error:
                errors.append(
                    {
                        "zarr_path": str(zarr_path.expanduser().resolve()),
                        "error_type": "DatasetNotRegistered",
                        "error": "Registry scan returned no dataset ID.",
                    }
                )

        if fail_on_error and scanned_count == 0:
            errors.append(
                {
                    "zarr_path": None,
                    "error_type": "NoZarrDiscovered",
                    "error": "No requested Zarr archive was discovered.",
                }
            )

        reconcile = registry.reconcile_missing_datasets(scope_paths=roots)
    finally:
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
        "recording_step_status_reconciled": bool(reconcile_step_status),
        "errors": errors,
        "reconcile": reconcile,
    }
    return (1 if errors and fail_on_error else 0), result


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
    parser.add_argument(
        "--reconcile-step-status",
        action="store_true",
        help=(
            "Refresh recording_step_status from each scanned Zarr in the same "
            "operation. Required after workflow stages publish new artifacts."
        ),
    )
    parser.add_argument(
        "--safe-shadow-publish",
        action="store_true",
        help=(
            "Mutate a node-local SQLite snapshot, run full integrity checks, and "
            "atomically publish it only if the canonical registry is unchanged."
        ),
    )
    parser.add_argument(
        "--backup-path",
        type=Path,
        help=(
            "Immutable pre-write SQLite backup. Required with "
            "--safe-shadow-publish."
        ),
    )
    parser.add_argument(
        "--local-temp-root",
        type=Path,
        help="Optional node-local parent for the shadow SQLite workspace.",
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
    if args.safe_shadow_publish and args.backup_path is None:
        parser.error("--backup-path is required with --safe-shadow-publish")
    if not args.safe_shadow_publish and args.backup_path is not None:
        parser.error("--backup-path requires --safe-shadow-publish")

    if args.safe_shadow_publish:
        operation_status = 0

        def mutate(local_registry: Path) -> dict[str, Any]:
            nonlocal operation_status
            operation_status, local_result = _rescan_registry(
                registry_path=local_registry,
                roots=roots,
                recursive=bool(args.recursive),
                fail_on_error=bool(args.fail_on_error),
                reconcile_step_status=bool(args.reconcile_step_status),
            )
            if operation_status != 0:
                raise RuntimeError(
                    "Registry rescan reported errors; refusing shadow publication."
                )
            return local_result

        publication = publish_registry_shadow(
            canonical_registry=registry_path,
            backup_path=args.backup_path,
            mutate=mutate,
            local_temp_root=args.local_temp_root,
        )
        result = dict(publication.mutation_result)
        result["registry"] = str(registry_path.expanduser().resolve())
        result["registry_publication"] = publication.to_json()
        result["registry_publication"]["mutation_result"] = {
            "status": result["status"],
            "updated_count": result["updated_count"],
            "scanned_count": result["scanned_count"],
        }
        status = operation_status
    else:
        status, result = _rescan_registry(
            registry_path=registry_path,
            roots=roots,
            recursive=bool(args.recursive),
            fail_on_error=bool(args.fail_on_error),
            reconcile_step_status=bool(args.reconcile_step_status),
        )
    if args.result_json is not None:
        write_json_atomic(args.result_json.expanduser().resolve(), result)
    return status


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
