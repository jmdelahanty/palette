"""Read-only census and explicit backfill of normalized stimulus metadata."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import closing
import json
from pathlib import Path
import sqlite3
from typing import Any, Sequence
from urllib.parse import quote

import zarr

from fisheye.registry.db import Registry, RegistryPaths
from fisheye.registry.extractors.stimulus_metadata import extract_stimulus_metadata
from fisheye.registry.prune_stale_datasets import create_backup
from fisheye.shared.batch_logging import utc_now


SCHEMA_ID = "palette.registry_stimulus_metadata_census.v1"


def _connect_read_only(path: Path) -> sqlite3.Connection:
    resolved = path.expanduser().resolve(strict=True)
    conn = sqlite3.connect(f"file:{quote(str(resolved), safe='/')}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON;")
    conn.execute("PRAGMA busy_timeout = 5000;")
    return conn


def select_analysis_datasets(
    registry_path: Path,
    *,
    recording_ids: Sequence[str] = (),
    status: str = "active",
    limit: int | None = None,
    all_recordings: bool = False,
) -> list[dict[str, Any]]:
    """Return recording-owned analysis datasets without writing the registry."""

    if not all_recordings and not recording_ids:
        raise ValueError("Provide --recording-id or pass --all-recordings.")
    sql = [
        "SELECT dataset_id, recording_id, zarr_path, protocol_name",
        "FROM dataset_context_current",
        "WHERE zarr_use = 'analysis' AND dataset_status = ?",
        "AND NULLIF(TRIM(recording_id), '') IS NOT NULL",
    ]
    params: list[Any] = [status]
    if recording_ids:
        placeholders = ", ".join("?" for _ in recording_ids)
        sql.append(f"AND recording_id IN ({placeholders})")
        params.extend(str(value) for value in recording_ids)
    sql.append("ORDER BY recording_id, dataset_id")
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        sql.append("LIMIT ?")
        params.append(limit)
    with closing(_connect_read_only(registry_path)) as conn:
        rows = conn.execute("\n".join(sql), params).fetchall()
    return [
        {
            "dataset_id": str(row["dataset_id"]),
            "recording_id": str(row["recording_id"]),
            "zarr_path": str(
                Path(str(row["zarr_path"])).expanduser().resolve(strict=False)
            ),
            "protocol_name": (
                str(row["protocol_name"]) if row["protocol_name"] is not None else None
            ),
        }
        for row in rows
    ]


def _open_root(path: Path) -> zarr.Group:
    try:
        return zarr.open_group(str(path), mode="r", use_consolidated=False)
    except TypeError:
        return zarr.open_group(str(path), mode="r")


def _dataset_issue(
    dataset: dict[str, Any],
    *,
    reason: str,
    detail: str,
) -> dict[str, Any]:
    return {
        "dataset_id": dataset["dataset_id"],
        "recording_id": dataset["recording_id"],
        "zarr_path": dataset["zarr_path"],
        "reason": reason,
        "detail": detail,
    }


def build_stimulus_metadata_census(datasets: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Extract normalized rows from authoritative Zarr metadata without writes."""

    mode_counts: Counter[str] = Counter()
    latest_mode_counts: Counter[str] = Counter()
    latest_mode_datasets: defaultdict[str, set[str]] = defaultdict(set)
    protocol_counts: Counter[str] = Counter()
    issues: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    stimulus_run_count = 0
    latest_run_count = 0
    datasets_with_stimulus = 0

    for dataset in datasets:
        zarr_path = Path(str(dataset["zarr_path"]))
        try:
            extraction = extract_stimulus_metadata(
                _open_root(zarr_path),
                zarr_path=zarr_path,
                recording_id=str(dataset["recording_id"]),
            )
        except Exception as exc:
            issues.append(
                _dataset_issue(
                    dataset,
                    reason="archive_read_failed",
                    detail=f"{type(exc).__name__}: {exc}",
                )
            )
            dataset_rows.append(
                {
                    **dataset,
                    "read_status": "error",
                    "protocols": [],
                    "protocol_steps": [],
                    "recording_runs": [],
                    "recording_steps": [],
                    "recording_modes": [],
                }
            )
            continue

        protocols = [dict(row) for row in extraction.protocols]
        protocol_steps = [dict(row) for row in extraction.protocol_steps]
        recording_runs = [dict(row) for row in extraction.recording_runs]
        recording_steps = [dict(row) for row in extraction.recording_steps]
        recording_modes = [dict(row) for row in extraction.recording_modes]
        stimulus_run_count += len(recording_runs)
        latest_rows = [row for row in recording_runs if int(row.get("is_latest", 0)) == 1]
        latest_run_ids = {str(row["stimulus_run_id"]) for row in latest_rows}
        latest_run_count += len(latest_rows)
        if recording_runs:
            datasets_with_stimulus += 1
            if len(latest_rows) != 1:
                issues.append(
                    _dataset_issue(
                        dataset,
                        reason="latest_stimulus_run_count",
                        detail=f"expected 1 latest run, found {len(latest_rows)}",
                    )
                )
        for row in recording_modes:
            mode = str(row["stimulus_mode"])
            mode_counts[mode] += 1
            if str(row["stimulus_run_id"]) in latest_run_ids:
                latest_mode_counts[mode] += 1
                latest_mode_datasets[mode].add(str(dataset["dataset_id"]))
            if mode == "UNKNOWN":
                issues.append(
                    _dataset_issue(
                        dataset,
                        reason="unknown_stimulus_mode",
                        detail=f"run={row['stimulus_run_id']}",
                    )
                )
        for row in protocols:
            name = str(row.get("protocol_name") or "<unnamed>")
            protocol_counts[name] += 1
        dataset_rows.append(
            {
                **dataset,
                "read_status": "ok",
                "protocols": protocols,
                "protocol_steps": protocol_steps,
                "recording_runs": recording_runs,
                "recording_steps": recording_steps,
                "recording_modes": recording_modes,
            }
        )

    return {
        "schema_id": SCHEMA_ID,
        "created_utc": utc_now(),
        "dataset_count": len(datasets),
        "recording_count": len({str(row["recording_id"]) for row in datasets}),
        "physical_archive_count": len({str(row["zarr_path"]) for row in datasets}),
        "datasets_with_stimulus_count": datasets_with_stimulus,
        "stimulus_run_count": stimulus_run_count,
        "latest_stimulus_run_count": latest_run_count,
        "mode_run_counts": dict(sorted(mode_counts.items())),
        "latest_mode_run_counts": dict(sorted(latest_mode_counts.items())),
        "latest_mode_dataset_counts": {
            mode: len(dataset_ids)
            for mode, dataset_ids in sorted(latest_mode_datasets.items())
        },
        "protocol_dataset_counts": dict(sorted(protocol_counts.items())),
        "issue_count": len(issues),
        "issues": issues,
        "datasets": dataset_rows,
    }


def apply_stimulus_metadata_census(
    registry_path: Path,
    census: dict[str, Any],
    *,
    allow_issues: bool = False,
) -> dict[str, int]:
    """Replace only normalized stimulus tables for successfully read datasets."""

    issue_count = int(census.get("issue_count", 0))
    if issue_count and not allow_issues:
        raise ValueError(
            f"Census contains {issue_count} issue(s); inspect them or pass --allow-issues."
        )
    registry = Registry(registry_path)
    applied = 0
    skipped = 0
    try:
        with registry._transaction_context():
            for dataset in census.get("datasets", []):
                if dataset.get("read_status") != "ok":
                    skipped += 1
                    continue
                registry.replace_stimulus_metadata(
                    str(dataset["dataset_id"]),
                    protocols=dataset.get("protocols", []),
                    protocol_steps=dataset.get("protocol_steps", []),
                    recording_runs=dataset.get("recording_runs", []),
                    recording_steps=dataset.get("recording_steps", []),
                    recording_modes=dataset.get("recording_modes", []),
                )
                applied += 1
    finally:
        registry.close()
    return {"applied_dataset_count": applied, "skipped_dataset_count": skipped}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Census normalized stimulus metadata and optionally backfill only its "
            "registry tables."
        ),
    )
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--recording-id", action="append", default=[])
    parser.add_argument("--status", default="active")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--all-recordings", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--allow-issues", action="store_true")
    parser.add_argument(
        "--backup",
        type=Path,
        help="Required with --apply; created with the SQLite backup API before writes.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    registry_path = args.registry or RegistryPaths.from_env(Path.cwd()).path
    datasets = select_analysis_datasets(
        registry_path,
        recording_ids=args.recording_id,
        status=str(args.status),
        limit=args.limit,
        all_recordings=bool(args.all_recordings),
    )
    census = build_stimulus_metadata_census(datasets)
    if args.apply:
        if args.backup is None:
            raise ValueError("--apply requires --backup")
        if int(census.get("issue_count", 0)) and not args.allow_issues:
            raise ValueError(
                f"Census contains {census['issue_count']} issue(s); inspect them or "
                "pass --allow-issues."
            )
        if args.backup.exists():
            raise FileExistsError(f"Backup already exists: {args.backup}")
        create_backup(registry_path, args.backup)
        census["registry_backup"] = str(args.backup.expanduser().resolve(strict=False))
        census["apply_result"] = apply_stimulus_metadata_census(
            registry_path,
            census,
            allow_issues=bool(args.allow_issues),
        )
    payload = json.dumps(census, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SCHEMA_ID",
    "apply_stimulus_metadata_census",
    "build_stimulus_metadata_census",
    "select_analysis_datasets",
]
