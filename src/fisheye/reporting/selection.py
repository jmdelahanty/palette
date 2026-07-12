"""Read-only registry selection for dataset report planning."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import quote

from .models import SelectedRecording


def _connect_read_only(registry_path: Path) -> sqlite3.Connection:
    resolved = registry_path.expanduser().resolve(strict=True)
    uri = f"file:{quote(str(resolved), safe='/')}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON;")
    return connection


def _view_columns(connection: sqlite3.Connection, view_name: str) -> set[str]:
    return {
        str(row[1])
        for row in connection.execute(f"PRAGMA table_info({view_name})").fetchall()
    }


def query_report_recordings(
    registry_path: Path,
    *,
    protocol_name: str | None = None,
    recording_ids: Sequence[str] = (),
    recording_id_contains: str | None = None,
    path_contains: str | None = None,
    zarr_use: str = "analysis",
    status: str = "active",
    limit: int | None = None,
    all_recordings: bool = False,
) -> list[SelectedRecording]:
    """Select one physical Zarr per recording from ``dataset_context_current``."""

    if not all_recordings and not any(
        (protocol_name, recording_ids, recording_id_contains, path_contains)
    ):
        raise ValueError(
            "Provide a cohort selector such as --protocol-name or pass "
            "--all-recordings explicitly."
        )
    if limit is not None and limit < 1:
        raise ValueError("limit must be >= 1")

    with closing(_connect_read_only(registry_path)) as connection:
        columns = _view_columns(connection, "dataset_context_current")
        if not columns:
            raise ValueError("Registry has no dataset_context_current view.")
        optional = {
            "protocol_hash": "NULL AS protocol_hash",
            "arena_id": "NULL AS arena_id",
            "recording_started_utc": "NULL AS recording_started_utc",
        }
        select_optional = [
            name if name in columns else fallback
            for name, fallback in optional.items()
        ]
        sql = [
            "SELECT dataset_id, recording_id, zarr_path, protocol_name,",
            "       " + ", ".join(select_optional),
            "FROM dataset_context_current",
            "WHERE zarr_use = ? AND dataset_status = ?",
        ]
        parameters: list[Any] = [str(zarr_use), str(status)]
        if protocol_name:
            sql.append("AND protocol_name = ? COLLATE NOCASE")
            parameters.append(str(protocol_name))
        if recording_ids:
            placeholders = ", ".join("?" for _ in recording_ids)
            sql.append(f"AND recording_id IN ({placeholders})")
            parameters.extend(str(value) for value in recording_ids)
        if recording_id_contains:
            sql.append("AND recording_id LIKE ?")
            parameters.append(f"%{recording_id_contains}%")
        if path_contains:
            sql.append("AND zarr_path LIKE ?")
            parameters.append(f"%{path_contains}%")
        sql.append(
            "ORDER BY COALESCE(recording_started_utc, recording_id), "
            "arena_id, recording_id, dataset_id"
        )
        if limit is not None:
            sql.append("LIMIT ?")
            parameters.append(int(limit))
        rows = connection.execute("\n".join(sql), parameters).fetchall()

    recordings: list[SelectedRecording] = []
    seen_paths: dict[str, Path] = {}
    for row in rows:
        recording_id = str(row["recording_id"] or "").strip()
        if not recording_id:
            raise ValueError(f"Dataset {row['dataset_id']} has no recording_id.")
        zarr_path = Path(str(row["zarr_path"])).expanduser().resolve(strict=False)
        existing = seen_paths.get(recording_id)
        if existing is not None and existing == zarr_path:
            continue
        if existing is not None:
            raise ValueError(
                f"Registry query selected multiple physical {zarr_use!r} datasets for "
                f"{recording_id!r}: {existing} and {zarr_path}."
            )
        seen_paths[recording_id] = zarr_path
        recordings.append(
            SelectedRecording(
                dataset_id=str(row["dataset_id"]),
                recording_id=recording_id,
                zarr_path=str(zarr_path),
                protocol_name=(
                    str(row["protocol_name"])
                    if row["protocol_name"] is not None
                    else None
                ),
                protocol_hash=(
                    str(row["protocol_hash"])
                    if row["protocol_hash"] is not None
                    else None
                ),
                arena_id=str(row["arena_id"]) if row["arena_id"] is not None else None,
                recording_started_utc=(
                    str(row["recording_started_utc"])
                    if row["recording_started_utc"] is not None
                    else None
                ),
            )
        )
    return recordings
