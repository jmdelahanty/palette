"""Resolve indexed Palette analytics export table paths."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fisheye.registry.db import Registry, RegistryPaths


@dataclass(frozen=True)
class AnalyticsExportTableResolution:
    """Resolved Parquet table location from the analytics export registry."""

    registry_path: Path
    export_run_id: str
    table_name: str
    table_path: Path
    collection_id: str | None
    collection_manifest_sha256: str | None
    collection_name: str | None
    status: str | None
    output_root: Path | None
    export_manifest_path: Path | None
    created_at_utc: str | None
    row_count: int | None
    part_count: int | None
    part_files: tuple[str, ...]


def _registry_path(path: Path | None) -> Path:
    if path is not None:
        return path.expanduser().resolve()
    return RegistryPaths.from_env(Path.cwd()).path


def _optional_path(value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    return Path(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _part_files(value: Any) -> tuple[str, ...]:
    if not isinstance(value, str) or not value:
        return ()
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return ()
    if not isinstance(payload, list):
        return ()
    return tuple(str(item) for item in payload)


def resolve_latest_export_table(
    *,
    table_name: str,
    registry_path: Path | None = None,
    collection_id: str | None = None,
    collection_manifest_sha256: str | None = None,
    export_run_id: str | None = None,
    status: str = "active",
) -> AnalyticsExportTableResolution:
    """Resolve the latest matching indexed analytics export table.

    ``collection_id`` or ``collection_manifest_sha256`` should normally be
    provided so a broad registry does not accidentally resolve an unrelated
    table from a different cohort.
    """

    resolved_registry_path = _registry_path(registry_path)
    params: list[Any] = [str(table_name)]
    sql = [
        """
        SELECT
            ae.export_run_id,
            ae.status,
            ae.collection_id,
            ae.collection_manifest_sha256,
            ae.collection_name,
            ae.output_root,
            ae.export_manifest_path,
            ae.created_at_utc,
            aet.table_name,
            aet.table_path,
            aet.row_count,
            aet.part_count,
            aet.part_files_json
        FROM analytics_export_overview ae
        JOIN analytics_export_tables aet
          ON aet.export_run_id = ae.export_run_id
        WHERE aet.table_name = ?
        """
    ]
    if status != "any":
        sql.append("AND ae.status = ?")
        params.append(status)
    if collection_id:
        sql.append("AND ae.collection_id = ?")
        params.append(collection_id)
    if collection_manifest_sha256:
        sql.append("AND ae.collection_manifest_sha256 = ?")
        params.append(collection_manifest_sha256)
    if export_run_id:
        sql.append("AND ae.export_run_id = ?")
        params.append(export_run_id)
    sql.append("ORDER BY COALESCE(ae.created_at_utc, ae.indexed_utc) DESC, ae.export_run_id DESC")
    sql.append("LIMIT 1")

    registry = Registry(resolved_registry_path)
    try:
        row = registry.conn.execute("\n".join(sql), params).fetchone()
    finally:
        registry.close()

    if row is None:
        filters = [
            f"table_name={table_name!r}",
            f"status={status!r}",
            f"collection_id={collection_id!r}",
            f"collection_manifest_sha256={collection_manifest_sha256!r}",
            f"export_run_id={export_run_id!r}",
        ]
        raise LookupError(
            f"No indexed analytics export table matched {', '.join(filters)} "
            f"in registry {resolved_registry_path}"
        )
    if row["table_path"] is None:
        raise LookupError(
            f"Indexed analytics export {row['export_run_id']!r} table {table_name!r} "
            "does not have a table_path"
        )

    return AnalyticsExportTableResolution(
        registry_path=resolved_registry_path,
        export_run_id=str(row["export_run_id"]),
        table_name=str(row["table_name"]),
        table_path=Path(str(row["table_path"])),
        collection_id=row["collection_id"],
        collection_manifest_sha256=row["collection_manifest_sha256"],
        collection_name=row["collection_name"],
        status=row["status"],
        output_root=_optional_path(row["output_root"]),
        export_manifest_path=_optional_path(row["export_manifest_path"]),
        created_at_utc=row["created_at_utc"],
        row_count=_optional_int(row["row_count"]),
        part_count=_optional_int(row["part_count"]),
        part_files=_part_files(row["part_files_json"]),
    )
