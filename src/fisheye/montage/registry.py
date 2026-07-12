"""Read-only registry selection for montage cohorts."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import quote

from fisheye.analysis.chaser_behavior import BEHAVIOR_CLASS_LABELS, canonical_behavior_label

from .models import RegistryRecording


def _connect_registry_read_only(registry_path: Path) -> sqlite3.Connection:
    resolved = registry_path.expanduser().resolve(strict=True)
    uri = f"file:{quote(str(resolved), safe='/')}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON;")
    return conn


def query_registry_recordings(
    registry_path: Path,
    *,
    protocol_name: str | None = None,
    recording_ids: Sequence[str] = (),
    recording_id_contains: str | None = None,
    path_contains: str | None = None,
    arena_ids: Sequence[str] = (),
    chaser_behaviors: Sequence[str] = (),
    chaser_count: int | None = None,
    zarr_use: str = "analysis",
    status: str = "active",
    limit: int | None = None,
    all_recordings: bool = False,
) -> list[RegistryRecording]:
    if not all_recordings and not any(
        (
            protocol_name,
            recording_ids,
            recording_id_contains,
            path_contains,
            arena_ids,
            chaser_behaviors,
            chaser_count,
        )
    ):
        raise ValueError(
            "Provide a cohort selector such as --protocol-name or pass --all-recordings explicitly."
        )
    if limit is not None and limit < 1:
        raise ValueError("limit must be >= 1")
    if chaser_count is not None and chaser_count < 1:
        raise ValueError("chaser_count must be >= 1")
    normalized_behaviors = tuple(
        sorted({canonical_behavior_label(value) for value in chaser_behaviors})
    )
    allowed_behaviors = set(BEHAVIOR_CLASS_LABELS.values())
    invalid_behaviors = sorted(set(normalized_behaviors) - allowed_behaviors)
    if invalid_behaviors:
        raise ValueError(f"Unknown chaser behavior(s): {', '.join(invalid_behaviors)}")
    with closing(_connect_registry_read_only(registry_path)) as conn:
        has_chaser_metadata = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'recording_chasers'"
        ).fetchone() is not None
    if (normalized_behaviors or chaser_count is not None) and not has_chaser_metadata:
        raise ValueError(
            "Registry has no recording_chasers index; run the chaser metadata census backfill first."
        )

    sql = [
        "SELECT dcc.dataset_id, dcc.recording_id, dcc.zarr_path, dcc.protocol_name,",
        "       dcc.arena_id, dcc.recording_started_utc,",
        (
            "       (SELECT GROUP_CONCAT(DISTINCT rc.behavior_class) "
            "FROM recording_chasers rc WHERE rc.dataset_id = dcc.dataset_id) AS chaser_behaviors,"
            if has_chaser_metadata
            else "       NULL AS chaser_behaviors,"
        ),
        (
            "       (SELECT MAX(rcr.chaser_count) FROM recording_chaser_runs rcr "
            "WHERE rcr.dataset_id = dcc.dataset_id) AS chaser_count"
            if has_chaser_metadata
            else "       NULL AS chaser_count"
        ),
        "FROM dataset_context_current dcc",
        "WHERE dcc.zarr_use = ? AND dcc.dataset_status = ?",
    ]
    parameters: list[Any] = [zarr_use, status]
    if protocol_name:
        sql.append("AND protocol_name = ? COLLATE NOCASE")
        parameters.append(protocol_name)
    if recording_ids:
        placeholders = ", ".join("?" for _ in recording_ids)
        sql.append(f"AND recording_id IN ({placeholders})")
        parameters.extend(recording_ids)
    if recording_id_contains:
        sql.append("AND recording_id LIKE ?")
        parameters.append(f"%{recording_id_contains}%")
    if path_contains:
        sql.append("AND zarr_path LIKE ?")
        parameters.append(f"%{path_contains}%")
    if arena_ids:
        placeholders = ", ".join("?" for _ in arena_ids)
        sql.append(f"AND arena_id IN ({placeholders})")
        parameters.extend(arena_ids)
    if normalized_behaviors or chaser_count is not None:
        having: list[str] = []
        if chaser_count is not None:
            having.append("COUNT(*) = ?")
        if normalized_behaviors:
            placeholders = ", ".join("?" for _ in normalized_behaviors)
            having.append(
                f"COUNT(DISTINCT CASE WHEN rc.behavior_class IN ({placeholders}) "
                "THEN rc.behavior_class END) = ?"
            )
        sql.append(
            "AND EXISTS (SELECT 1 FROM recording_chasers rc "
            "WHERE rc.dataset_id = dcc.dataset_id GROUP BY rc.stimulus_run_id HAVING "
            + " AND ".join(having)
            + ")"
        )
        if chaser_count is not None:
            parameters.append(int(chaser_count))
        if normalized_behaviors:
            parameters.extend(normalized_behaviors)
            parameters.append(len(normalized_behaviors))
    sql.append(
        "ORDER BY COALESCE(recording_started_utc, recording_id), arena_id, recording_id, dataset_id"
    )
    if limit is not None:
        sql.append("LIMIT ?")
        parameters.append(limit)

    with closing(_connect_registry_read_only(registry_path)) as conn:
        rows = conn.execute("\n".join(sql), parameters).fetchall()
    recordings: list[RegistryRecording] = []
    seen_recording_paths: dict[str, Path] = {}
    for row in rows:
        recording_id = str(row["recording_id"] or "").strip()
        if not recording_id:
            raise ValueError(f"Dataset {row['dataset_id']} has no recording_id.")
        zarr_path = Path(str(row["zarr_path"])).expanduser().resolve(strict=False)
        existing_path = seen_recording_paths.get(recording_id)
        if existing_path is not None and existing_path == zarr_path:
            continue
        if existing_path is not None:
            raise ValueError(
                f"Registry query selected more than one physical {zarr_use!r} dataset for "
                f"{recording_id!r}: {existing_path} and {zarr_path}."
            )
        seen_recording_paths[recording_id] = zarr_path
        recordings.append(
            RegistryRecording(
                recording_id=recording_id,
                zarr_path=zarr_path,
                dataset_id=str(row["dataset_id"]),
                protocol_name=str(row["protocol_name"]) if row["protocol_name"] is not None else None,
                arena_id=str(row["arena_id"]) if row["arena_id"] is not None else None,
                recording_started_utc=(
                    str(row["recording_started_utc"])
                    if row["recording_started_utc"] is not None
                    else None
                ),
                chaser_behaviors=tuple(
                    sorted(
                        value.strip()
                        for value in str(row["chaser_behaviors"] or "").split(",")
                        if value.strip()
                    )
                ),
                chaser_count=(int(row["chaser_count"]) if row["chaser_count"] is not None else None),
            )
        )
    return recordings
