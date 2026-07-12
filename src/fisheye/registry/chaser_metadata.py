"""Read-only census and explicit registry backfill for configured chasers."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import closing
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sqlite3
from typing import Any, Sequence
from urllib.parse import quote

import zarr

from fisheye.registry.db import Registry, RegistryPaths
from fisheye.registry.extractors.chaser_metadata import extract_recording_chaser_metadata
from fisheye.shared.batch_logging import utc_now


SCHEMA_ID = "palette.registry_chaser_metadata_census.v1"


@dataclass(frozen=True)
class CensusDataset:
    dataset_id: str
    recording_id: str
    zarr_path: Path
    protocol_name: str | None


def _connect_read_only(path: Path) -> sqlite3.Connection:
    resolved = path.expanduser().resolve(strict=True)
    conn = sqlite3.connect(f"file:{quote(str(resolved), safe='/')}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON;")
    return conn


def select_census_datasets(
    registry_path: Path,
    *,
    protocol_name: str | None = None,
    recording_ids: Sequence[str] = (),
    zarr_use: str = "analysis",
    status: str = "active",
    limit: int | None = None,
    all_recordings: bool = False,
) -> list[CensusDataset]:
    if not all_recordings and not protocol_name and not recording_ids:
        raise ValueError("Provide --protocol-name or --recording-id, or pass --all-recordings.")
    sql = [
        "SELECT dataset_id, recording_id, zarr_path, protocol_name",
        "FROM dataset_context_current",
        "WHERE zarr_use = ? AND dataset_status = ?",
    ]
    params: list[Any] = [zarr_use, status]
    if protocol_name:
        sql.append("AND protocol_name = ? COLLATE NOCASE")
        params.append(protocol_name)
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
        result = conn.execute("\n".join(sql), params).fetchall()
    return [
        CensusDataset(
            dataset_id=str(row["dataset_id"]),
            recording_id=str(row["recording_id"]),
            zarr_path=Path(str(row["zarr_path"])).expanduser().resolve(strict=False),
            protocol_name=str(row["protocol_name"]) if row["protocol_name"] is not None else None,
        )
        for row in result
    ]


def _open_root(path: Path) -> zarr.Group:
    try:
        return zarr.open_group(str(path), mode="r", use_consolidated=False)
    except TypeError:
        return zarr.open_group(str(path), mode="r")


def build_chaser_metadata_census(datasets: Sequence[CensusDataset]) -> dict[str, Any]:
    behavior_counts: Counter[str] = Counter()
    chaser_count_counts: Counter[int] = Counter()
    rows_by_dataset: dict[str, list[dict[str, Any]]] = {}
    issues: list[dict[str, Any]] = []
    stimulus_run_count = 0
    for dataset in datasets:
        try:
            extraction = extract_recording_chaser_metadata(
                _open_root(dataset.zarr_path),
                zarr_path=dataset.zarr_path,
                recording_id=dataset.recording_id,
            )
        except Exception as exc:
            issues.append(
                {
                    "dataset_id": dataset.dataset_id,
                    "recording_id": dataset.recording_id,
                    "source_path": str(dataset.zarr_path),
                    "reason": "archive_read_failed",
                    "detail": f"{type(exc).__name__}: {exc}",
                }
            )
            rows_by_dataset[dataset.dataset_id] = []
            continue
        stimulus_run_count += extraction.stimulus_run_count
        rows = [dict(row) for row in extraction.rows]
        rows_by_dataset[dataset.dataset_id] = rows
        behavior_counts.update(str(row["behavior_class"]) for row in rows)
        run_counts = Counter(str(row["stimulus_run_id"]) for row in rows)
        chaser_count_counts.update(run_counts.values())
        issues.extend(
            {
                "dataset_id": dataset.dataset_id,
                "recording_id": dataset.recording_id,
                **asdict(issue),
            }
            for issue in extraction.issues
        )

    return {
        "schema_id": SCHEMA_ID,
        "created_utc": utc_now(),
        "dataset_count": len(datasets),
        "recording_count": len({dataset.recording_id for dataset in datasets}),
        "physical_archive_count": len({str(dataset.zarr_path) for dataset in datasets}),
        "stimulus_run_count": stimulus_run_count,
        "chaser_row_count": sum(len(rows) for rows in rows_by_dataset.values()),
        "behavior_counts": dict(sorted(behavior_counts.items())),
        "stimulus_runs_by_chaser_count": {
            str(count): occurrences for count, occurrences in sorted(chaser_count_counts.items())
        },
        "issue_count": len(issues),
        "issues": issues,
        "datasets": [
            {
                "dataset_id": dataset.dataset_id,
                "recording_id": dataset.recording_id,
                "zarr_path": str(dataset.zarr_path),
                "protocol_name": dataset.protocol_name,
                "chaser_rows": rows_by_dataset.get(dataset.dataset_id, []),
            }
            for dataset in datasets
        ],
    }


def apply_chaser_metadata_census(
    registry_path: Path,
    census: dict[str, Any],
    *,
    allow_issues: bool = False,
) -> int:
    issue_count = int(census.get("issue_count", 0))
    if issue_count and not allow_issues:
        raise ValueError(
            f"Census contains {issue_count} issue(s); inspect them or pass --allow-issues explicitly."
        )
    registry = Registry(registry_path)
    applied = 0
    try:
        with registry._transaction_context():
            for dataset in census.get("datasets", []):
                registry.replace_recording_chasers(
                    str(dataset["dataset_id"]),
                    dataset.get("chaser_rows", []),
                )
                applied += 1
    finally:
        registry.close()
    return applied


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Census configured chaser metadata; optionally backfill normalized registry rows.",
    )
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--protocol-name")
    parser.add_argument("--recording-id", action="append", default=[])
    parser.add_argument("--zarr-use", default="analysis")
    parser.add_argument("--status", default="active")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--all-recordings", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--allow-issues", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    registry_path = args.registry or RegistryPaths.from_env(Path.cwd()).path
    datasets = select_census_datasets(
        registry_path,
        protocol_name=args.protocol_name,
        recording_ids=args.recording_id,
        zarr_use=str(args.zarr_use),
        status=str(args.status),
        limit=args.limit,
        all_recordings=bool(args.all_recordings),
    )
    census = build_chaser_metadata_census(datasets)
    if args.apply:
        census["applied_dataset_count"] = apply_chaser_metadata_census(
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
    "CensusDataset",
    "SCHEMA_ID",
    "apply_chaser_metadata_census",
    "build_chaser_metadata_census",
    "select_census_datasets",
]
