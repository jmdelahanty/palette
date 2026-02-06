"""Maintenance CLI for cleaning stale/invalid registry rows."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from .db import Registry, RegistryPaths


@dataclass(frozen=True)
class InvalidDatasetCandidate:
    dataset_id: str
    zarr_path: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class FailedRunCandidate:
    run_id: str
    set_id: Optional[str]
    status: Optional[str]
    created_utc: Optional[str]


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Registry maintenance (reconcile, prune invalid rows, optional VACUUM).",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Optional root path(s) that scope reconcile/prune operations.",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        help="Optional path to the registry SQLite file.",
    )
    parser.add_argument(
        "--prune-invalid",
        action="store_true",
        help=(
            "Reconcile missing rows, then prune invalid datasets "
            "(status=missing or paths that point inside a Zarr store)."
        ),
    )
    parser.add_argument(
        "--prune-failed-runs",
        action="store_true",
        help="Prune training_runs rows with failed statuses.",
    )
    parser.add_argument(
        "--failed-status",
        action="append",
        help=(
            "Status value treated as failed when using --prune-failed-runs "
            "(repeatable or comma-separated, case-insensitive). Default: failed."
        ),
    )
    parser.add_argument(
        "--vacuum",
        action="store_true",
        help="Run SQLite VACUUM after maintenance actions.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without deleting rows or running VACUUM.",
    )
    parser.add_argument(
        "--list-limit",
        type=int,
        default=50,
        help="Number of candidate rows to print (0 = no limit).",
    )
    return parser.parse_args(argv)


def _normalize_scope_paths(scope_paths: Optional[Sequence[Path]]) -> List[Path]:
    if not scope_paths:
        return []
    normalized: List[Path] = []
    for path in scope_paths:
        candidate = Path(path).expanduser()
        try:
            normalized.append(candidate.resolve())
        except Exception:
            normalized.append(candidate.absolute())
    return normalized


def _matches_scope(path: str, scope_roots: Sequence[Path]) -> bool:
    if not scope_roots:
        return True
    candidate = Path(path).expanduser()
    try:
        candidate = candidate.resolve()
    except Exception:
        candidate = candidate.absolute()
    for root in scope_roots:
        if candidate == root:
            return True
        try:
            candidate.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _is_nested_zarr_subpath(path: str) -> bool:
    normalized = path.replace("\\", "/").rstrip("/").lower()
    if ".zarr/" not in normalized:
        return False
    return not normalized.endswith(".zarr")


def _is_zarr_root_path(path: Path) -> bool:
    return (path / "zarr.json").exists() or (path / ".zgroup").exists()


def _collect_invalid_dataset_candidates(
    registry: Registry,
    *,
    scope_paths: Optional[Sequence[Path]] = None,
    include_missing_scan: bool = False,
) -> List[InvalidDatasetCandidate]:
    scope_roots = _normalize_scope_paths(scope_paths)
    rows = registry.conn.execute(
        "SELECT dataset_id, zarr_path, status FROM datasets ORDER BY dataset_id;"
    ).fetchall()

    candidates: List[InvalidDatasetCandidate] = []
    for row in rows:
        dataset_id = str(row["dataset_id"])
        zarr_path = str(row["zarr_path"])
        if not _matches_scope(zarr_path, scope_roots):
            continue
        reasons: List[str] = []
        if row["status"] == "missing":
            reasons.append("status_missing")
        elif include_missing_scan:
            candidate = Path(zarr_path).expanduser()
            if not _is_zarr_root_path(candidate):
                reasons.append("status_missing")
        if _is_nested_zarr_subpath(zarr_path):
            reasons.append("nested_zarr_subpath")
        if reasons:
            candidates.append(
                InvalidDatasetCandidate(
                    dataset_id=dataset_id,
                    zarr_path=zarr_path,
                    reasons=tuple(sorted(reasons)),
                )
            )
    return candidates


def _delete_dataset_ids(registry: Registry, dataset_ids: Sequence[str], *, dry_run: bool) -> int:
    if dry_run or not dataset_ids:
        return 0
    with registry.conn:
        for dataset_id in dataset_ids:
            registry.conn.execute("DELETE FROM datasets WHERE dataset_id = ?;", (dataset_id,))
    return len(dataset_ids)


def _normalize_status_values(values: Optional[Sequence[str]]) -> tuple[str, ...]:
    normalized: set[str] = set()
    for value in values or ():
        for token in str(value).split(","):
            status = token.strip().lower()
            if status:
                normalized.add(status)
    if not normalized:
        normalized.add("failed")
    return tuple(sorted(normalized))


def _collect_failed_run_candidates(
    registry: Registry,
    *,
    status_values: Sequence[str],
) -> List[FailedRunCandidate]:
    target_statuses = {value.strip().lower() for value in status_values if value and value.strip()}
    if not target_statuses:
        target_statuses = {"failed"}
    rows = registry.conn.execute(
        """
        SELECT run_id, set_id, status, created_utc
        FROM training_runs
        ORDER BY created_utc DESC, run_id DESC;
        """
    ).fetchall()
    candidates: List[FailedRunCandidate] = []
    for row in rows:
        status_raw = row["status"]
        if status_raw is None:
            continue
        status = str(status_raw).strip().lower()
        if status not in target_statuses:
            continue
        candidates.append(
            FailedRunCandidate(
                run_id=str(row["run_id"]),
                set_id=row["set_id"],
                status=row["status"],
                created_utc=row["created_utc"],
            )
        )
    return candidates


def _delete_training_run_ids(registry: Registry, run_ids: Sequence[str], *, dry_run: bool) -> int:
    if dry_run or not run_ids:
        return 0
    with registry.conn:
        for run_id in run_ids:
            registry.conn.execute("DELETE FROM training_runs WHERE run_id = ?;", (run_id,))
    return len(run_ids)


def _print_candidates(candidates: Sequence[InvalidDatasetCandidate], *, list_limit: int) -> None:
    if not candidates:
        print("No invalid dataset rows found.")
        return

    print(f"Invalid dataset rows: {len(candidates)}")
    limit = len(candidates) if list_limit == 0 else min(len(candidates), list_limit)
    for candidate in candidates[:limit]:
        reasons = ",".join(candidate.reasons)
        print(f" - {candidate.dataset_id} [{reasons}]")
        print(f"   {candidate.zarr_path}")
    if limit < len(candidates):
        print(f" ... {len(candidates) - limit} more rows omitted (use --list-limit 0 to show all).")


def _print_failed_run_candidates(candidates: Sequence[FailedRunCandidate], *, list_limit: int) -> None:
    if not candidates:
        print("No failed training runs found.")
        return

    print(f"Failed training runs: {len(candidates)}")
    limit = len(candidates) if list_limit == 0 else min(len(candidates), list_limit)
    for candidate in candidates[:limit]:
        set_id = candidate.set_id or "—"
        status = candidate.status or "—"
        created_utc = candidate.created_utc or "—"
        print(f" - {candidate.run_id} [set={set_id} status={status} created={created_utc}]")
    if limit < len(candidates):
        print(f" ... {len(candidates) - limit} more rows omitted (use --list-limit 0 to show all).")


def _summarize_reconcile(stats: Dict[str, int]) -> None:
    checked = int(stats.get("checked", 0))
    marked_missing = int(stats.get("marked_missing", 0))
    print(f"Reconcile missing: checked={checked}, marked_missing={marked_missing}")


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = _parse_args(argv)
    if not args.prune_invalid and not args.prune_failed_runs and not args.vacuum:
        raise SystemExit(
            "No action selected. Use --prune-invalid, --prune-failed-runs, and/or --vacuum."
        )

    registry_path = args.registry or RegistryPaths.from_env(Path.cwd()).path
    scope_paths = [Path(path).expanduser() for path in args.paths]

    registry = Registry(registry_path)
    try:
        if args.prune_invalid:
            if args.dry_run:
                print("Dry run: reconcile step is simulated (no status fields are updated).")
                candidates = _collect_invalid_dataset_candidates(
                    registry,
                    scope_paths=scope_paths or None,
                    include_missing_scan=True,
                )
            else:
                stats = registry.reconcile_missing_datasets(scope_paths=scope_paths or None)
                _summarize_reconcile(stats)
                candidates = _collect_invalid_dataset_candidates(registry, scope_paths=scope_paths or None)
            _print_candidates(candidates, list_limit=args.list_limit)
            dataset_ids = [candidate.dataset_id for candidate in candidates]
            if args.dry_run:
                print(f"Dry run: would delete {len(dataset_ids)} dataset row(s).")
            else:
                deleted = _delete_dataset_ids(registry, dataset_ids, dry_run=False)
                print(f"Deleted {deleted} dataset row(s).")

        if args.prune_failed_runs:
            status_values = _normalize_status_values(args.failed_status)
            failed_run_candidates = _collect_failed_run_candidates(
                registry,
                status_values=status_values,
            )
            _print_failed_run_candidates(failed_run_candidates, list_limit=args.list_limit)
            run_ids = [candidate.run_id for candidate in failed_run_candidates]
            status_list = ", ".join(status_values)
            if args.dry_run:
                print(
                    f"Dry run: would delete {len(run_ids)} training run row(s) "
                    f"with status in [{status_list}]."
                )
            else:
                deleted = _delete_training_run_ids(registry, run_ids, dry_run=False)
                print(f"Deleted {deleted} training run row(s) with status in [{status_list}].")

        if args.vacuum:
            if args.dry_run:
                print("Dry run: would run VACUUM.")
            else:
                registry.conn.commit()
                registry.conn.execute("VACUUM;")
                print("VACUUM complete.")
    finally:
        registry.close()


if __name__ == "__main__":
    main()
