"""Dry-run-first root sweep for registry reconciliation."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from fisheye.registry.db import (
    Registry,
    RegistryPaths,
    _is_empty_zarr_stub,
    _is_zarr_root,
    _normalize_fs_path,
    _open_zarr_group_non_consolidated,
    _path_matches_scope,
)
from fisheye.registry.prune_stale_datasets import connect_read_only


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _registry_path(registry: Registry | Path | str) -> Path:
    if isinstance(registry, Registry):
        return registry.path
    return Path(registry).expanduser()


def _normalize_roots(roots: Iterable[Path | str]) -> list[Path]:
    return [_normalize_fs_path(Path(root).expanduser()) for root in roots]


def _enumerate_zarr_roots(roots: Iterable[Path]) -> tuple[list[Path], list[dict[str, Any]]]:
    zarr_roots: list[Path] = []
    seen: set[Path] = set()
    root_reports: list[dict[str, Any]] = []

    for root in roots:
        entry: dict[str, Any] = {
            "path": str(root),
            "exists": root.exists(),
            "is_zarr_root": False,
            "discovered_count": 0,
        }
        if root.is_dir() and _is_zarr_root(root):
            resolved = _normalize_fs_path(root)
            entry["is_zarr_root"] = True
            if resolved not in seen:
                seen.add(resolved)
                zarr_roots.append(root)
                entry["discovered_count"] = 1
        elif root.is_dir():
            discovered_count = 0
            for current, dirnames, _filenames in os.walk(root):
                current_path = Path(current)
                for dirname in list(dirnames):
                    if not dirname.endswith(".zarr"):
                        continue
                    candidate = current_path / dirname
                    if not _is_zarr_root(candidate):
                        continue
                    dirnames.remove(dirname)
                    discovered_count += 1
                    resolved = _normalize_fs_path(candidate)
                    if resolved in seen:
                        continue
                    seen.add(resolved)
                    zarr_roots.append(candidate)
            entry["discovered_count"] = discovered_count
        root_reports.append(entry)

    zarr_roots.sort(key=lambda path: str(path))
    return zarr_roots, root_reports


def _dataset_rows(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT dataset_id, zarr_path, status
        FROM datasets
        ORDER BY zarr_path, dataset_id;
        """
    ).fetchall()


def _known_paths(conn: sqlite3.Connection) -> dict[Path, list[sqlite3.Row]]:
    by_path: dict[Path, list[sqlite3.Row]] = {}
    for row in _dataset_rows(conn):
        normalized = _normalize_fs_path(str(row["zarr_path"]))
        by_path.setdefault(normalized, []).append(row)
    return by_path


def _missing_candidates(conn: sqlite3.Connection, roots: Iterable[Path]) -> list[dict[str, Any]]:
    scope_roots = list(roots)
    missing: list[dict[str, Any]] = []
    for row in _dataset_rows(conn):
        if row["status"] == "missing":
            continue
        dataset_path = _normalize_fs_path(str(row["zarr_path"]))
        if scope_roots and not _path_matches_scope(dataset_path, scope_roots):
            continue
        path_exists = dataset_path.exists()
        zarr_root = _is_zarr_root(dataset_path)
        empty_stub = _is_empty_zarr_stub(dataset_path) if zarr_root else False
        if zarr_root and not empty_stub:
            continue
        missing.append(
            {
                "dataset_id": str(row["dataset_id"]),
                "zarr_path": str(row["zarr_path"]),
                "status": row["status"],
                "path_exists": path_exists,
                "is_zarr_root": zarr_root,
                "is_empty_zarr_stub": empty_stub,
            }
        )
    return missing


def _classification_counts(stores: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts = {"new": 0, "known": 0, "unreadable": 0}
    for store in stores:
        classification = str(store.get("classification", ""))
        if classification in counts:
            counts[classification] += 1
    return counts


def _open_store(path: Path):
    return _open_zarr_group_non_consolidated(path, mode="r")


def _store_row(path: Path, known_rows: list[sqlite3.Row]) -> dict[str, Any]:
    if known_rows:
        dataset_ids = [str(row["dataset_id"]) for row in known_rows]
        status_values = [row["status"] for row in known_rows]
        classification = "known"
    else:
        dataset_ids = []
        status_values = []
        classification = "new"
    return {
        "path": str(path),
        "classification": classification,
        "dataset_ids": dataset_ids,
        "registry_statuses": status_values,
        "is_empty_zarr_stub": _is_empty_zarr_stub(path),
        "would_reconcile": True,
    }


def reconcile_roots(
    registry: Registry | Path | str,
    roots: Iterable[Path | str],
    *,
    include_step_status: bool = True,
    apply: bool = False,
) -> dict[str, Any]:
    """Sweep roots and reconcile every readable Zarr store when ``apply`` is true.

    Dry-run mode opens the registry through SQLite ``mode=ro`` with
    ``PRAGMA query_only`` and never constructs ``Registry``.
    """

    registry_path = _registry_path(registry)
    normalized_roots = _normalize_roots(roots)
    zarr_roots, root_reports = _enumerate_zarr_roots(normalized_roots)

    if apply:
        writable_registry = registry if isinstance(registry, Registry) else Registry(registry_path)
        close_registry = not isinstance(registry, Registry)
        try:
            known = _known_paths(writable_registry.conn)
            stores: list[dict[str, Any]] = []
            for zarr_path in zarr_roots:
                normalized_path = _normalize_fs_path(zarr_path)
                row = _store_row(zarr_path, known.get(normalized_path, []))
                try:
                    root = _open_store(zarr_path)
                except Exception as exc:
                    row.update(
                        {
                            "classification": "unreadable",
                            "would_reconcile": False,
                            "applied": False,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    stores.append(row)
                    continue
                try:
                    result = writable_registry.reconcile_dataset_from_root(
                        root,
                        zarr_path,
                        include_step_status=include_step_status,
                    )
                except Exception as exc:
                    row.update(
                        {
                            "applied": False,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                else:
                    row.update(
                        {
                            "applied": True,
                            "dataset_id": result.get("dataset_id"),
                            "result": result,
                        }
                    )
                stores.append(row)
            missing_before = _missing_candidates(writable_registry.conn, normalized_roots)
            missing_result = writable_registry.reconcile_missing_datasets(scope_paths=normalized_roots)
            missing_after = _missing_candidates(writable_registry.conn, normalized_roots)
        finally:
            if close_registry:
                writable_registry.close()
    else:
        conn = connect_read_only(registry_path)
        try:
            known = _known_paths(conn)
            stores = []
            for zarr_path in zarr_roots:
                normalized_path = _normalize_fs_path(zarr_path)
                row = _store_row(zarr_path, known.get(normalized_path, []))
                try:
                    _open_store(zarr_path)
                except Exception as exc:
                    row.update(
                        {
                            "classification": "unreadable",
                            "would_reconcile": False,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                stores.append(row)
            missing_before = _missing_candidates(conn, normalized_roots)
            missing_result = {
                "checked": len(
                    [
                        row
                        for row in _dataset_rows(conn)
                        if row["status"] != "missing"
                        and _path_matches_scope(_normalize_fs_path(str(row["zarr_path"])), normalized_roots)
                    ]
                ),
                "marked_missing": len(missing_before),
            }
            missing_after = missing_before
        finally:
            conn.close()

    counts = _classification_counts(stores)
    return {
        "generated_at_utc": _utc_now(),
        "registry": str(registry_path),
        "mode": "apply" if apply else "dry-run",
        "read_only": not apply,
        "include_step_status": bool(include_step_status),
        "roots": root_reports,
        "summary": {
            "input_root_count": len(normalized_roots),
            "discovered_store_count": len(zarr_roots),
            "new_store_count": counts["new"],
            "known_store_count": counts["known"],
            "unreadable_store_count": counts["unreadable"],
            "would_reconcile_count": sum(1 for store in stores if store.get("would_reconcile")),
            "would_mark_missing_count": len(missing_before),
        },
        "stores": stores,
        "registered_but_vanished": {
            "would_mark_missing": missing_before,
            "result": missing_result,
            "remaining_after_apply": missing_after if apply else None,
        },
    }


def _write_json(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _print_report(report: Mapping[str, Any]) -> None:
    summary = report["summary"]
    print(f"Registry: {report['registry']}")
    print(f"Mode: {report['mode']}")
    print(
        "Stores: "
        f"discovered={summary['discovered_store_count']} "
        f"new={summary['new_store_count']} "
        f"known={summary['known_store_count']} "
        f"unreadable={summary['unreadable_store_count']}"
    )
    print(
        "Missing: "
        f"would_mark_missing={summary['would_mark_missing_count']}"
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dry-run-first registry reconcile sweep over one or more filesystem roots.",
    )
    parser.add_argument("--registry", type=Path, help="Path to the registry SQLite file.")
    parser.add_argument(
        "--reconcile-root",
        action="append",
        type=Path,
        required=True,
        help="Root path to sweep for Zarr stores. Repeat for multiple roots.",
    )
    parser.add_argument("--apply", action="store_true", help="Apply reconciliation writes.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview only; this is the default and opens the registry read-only.",
    )
    parser.add_argument("--json", type=Path, help="Write machine-readable report JSON to this path.")
    parser.add_argument(
        "--skip-step-status",
        action="store_true",
        help="In apply mode, skip recording_step_status refresh for reconciled datasets.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.apply and args.dry_run:
        raise SystemExit("--apply and --dry-run are mutually exclusive.")
    registry_path = args.registry or RegistryPaths.from_env(Path.cwd()).path
    report = reconcile_roots(
        registry_path,
        args.reconcile_root,
        include_step_status=not args.skip_step_status,
        apply=bool(args.apply),
    )
    if args.json is not None:
        _write_json(args.json, report)
    _print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
