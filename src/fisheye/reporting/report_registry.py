"""Bind immutable report bundles to indexed analytics exports."""

from __future__ import annotations

from contextlib import closing
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from fisheye.registry.db import Registry

from .catalog import VISUALIZATIONS
from .export import verify_report_manifest_sha256
from .selection import _connect_read_only


_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class AnalyticsExportBinding:
    """Verified registry and manifest identity for one analytics export."""

    export_run_id: str
    export_manifest_path: str
    export_manifest_content_sha256: str
    output_root: str
    collection_id: str | None
    collection_manifest_sha256: str | None
    available_tables: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["available_tables"] = list(self.available_tables)
        return payload


def validate_report_id(report_id: str) -> str:
    value = str(report_id).strip()
    if not _SAFE_ID.fullmatch(value):
        raise ValueError(
            "report_id must begin with an alphanumeric character and contain only "
            "letters, numbers, '.', '_', or '-'"
        )
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: JSON root must be an object")
    return payload


def resolve_analytics_export_binding(
    registry_path: Path,
    export_run_id: str,
    *,
    status: str = "active",
) -> AnalyticsExportBinding:
    """Resolve and verify an indexed analytics export without mutating the registry."""

    with closing(_connect_read_only(registry_path)) as connection:
        parameters: list[Any] = [str(export_run_id)]
        sql = [
            """
            SELECT export_run_id, status, collection_id,
                   collection_manifest_sha256, export_manifest_path, output_root
            FROM analytics_export_overview
            WHERE export_run_id = ?
            """
        ]
        if status != "any":
            sql.append("AND status = ?")
            parameters.append(str(status))
        row = connection.execute("\n".join(sql), parameters).fetchone()
        if row is None:
            raise LookupError(
                f"No indexed analytics export matched export_run_id={export_run_id!r}, "
                f"status={status!r}"
            )
        table_rows = connection.execute(
            """
            SELECT table_name
            FROM analytics_export_tables
            WHERE export_run_id = ?
            ORDER BY table_name
            """,
            (str(export_run_id),),
        ).fetchall()

    manifest_path = Path(str(row["export_manifest_path"])).expanduser().resolve()
    manifest = _load_json_object(manifest_path)
    manifest_export_run_id = str(manifest.get("export_run_id", ""))
    if manifest_export_run_id != str(export_run_id):
        raise ValueError(
            f"{manifest_path}: export_run_id is {manifest_export_run_id!r}, "
            f"expected {export_run_id!r}"
        )
    if row["output_root"] is None:
        raise ValueError(f"Indexed analytics export {export_run_id!r} has no output_root")
    output_root = Path(str(row["output_root"])).expanduser().resolve()
    return AnalyticsExportBinding(
        export_run_id=str(export_run_id),
        export_manifest_path=str(manifest_path),
        export_manifest_content_sha256=_file_sha256(manifest_path),
        output_root=str(output_root),
        collection_id=(str(row["collection_id"]) if row["collection_id"] is not None else None),
        collection_manifest_sha256=(
            str(row["collection_manifest_sha256"])
            if row["collection_manifest_sha256"] is not None
            else None
        ),
        available_tables=tuple(str(item["table_name"]) for item in table_rows),
    )


def report_output_dir(binding: AnalyticsExportBinding, report_id: str) -> Path:
    """Return the canonical immutable directory for a report/export pair."""

    safe_report_id = validate_report_id(report_id)
    return (
        Path(binding.output_root)
        / "v1"
        / "reports"
        / f"export_run_id={binding.export_run_id}"
        / f"report_id={safe_report_id}"
    )


def _one_or_none(values: Sequence[Any]) -> Any:
    unique = {value for value in values if value is not None}
    return next(iter(unique)) if len(unique) == 1 else None


def _visualization_summaries(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    artifacts = [item for item in manifest.get("artifacts", []) if isinstance(item, Mapping)]
    nonready = [item for item in manifest.get("nonready", []) if isinstance(item, Mapping)]
    ids = sorted(
        {str(value) for value in manifest.get("visualization_ids", [])}
        | {str(item.get("visualization_id")) for item in artifacts if item.get("visualization_id")}
        | {str(item.get("visualization_id")) for item in nonready if item.get("visualization_id")}
    )
    summaries: list[dict[str, Any]] = []
    for visualization_id in ids:
        selected_artifacts = [
            item for item in artifacts if item.get("visualization_id") == visualization_id
        ]
        selected_nonready = [
            item for item in nonready if item.get("visualization_id") == visualization_id
        ]
        spec = VISUALIZATIONS.get(visualization_id)
        paths: list[str] = []
        for item in selected_artifacts:
            materialized = item.get("materialized")
            if isinstance(materialized, Mapping):
                value = materialized.get("relative_path") or materialized.get("artifact_path")
                if value is not None:
                    paths.append(str(value))
        summaries.append(
            {
                "visualization_id": visualization_id,
                "provider_id": _one_or_none([item.get("provider_id") for item in selected_artifacts])
                or (spec.provider_id if spec else None),
                "label": spec.label if spec else None,
                "visualization_contract_id": _one_or_none(
                    [item.get("visualization_contract_id") for item in selected_artifacts]
                ),
                "renderer": _one_or_none([item.get("renderer") for item in selected_artifacts]),
                "renderer_version": _one_or_none(
                    [item.get("renderer_version") for item in selected_artifacts]
                ),
                "source_backends": sorted(
                    {
                        str(item.get("source_backend", "zarr"))
                        for item in selected_artifacts
                    }
                ),
                "artifact_count": len(selected_artifacts),
                "nonready_count": len(selected_nonready),
                "materialized_paths": paths,
            }
        )
    return summaries


def index_report_manifest(
    registry: Registry,
    manifest_path: Path,
    *,
    status: str = "active",
) -> tuple[str, str]:
    """Verify and index one report as a child of its analytics export."""

    resolved_path = manifest_path.expanduser().resolve()
    manifest = _load_json_object(resolved_path)
    if not verify_report_manifest_sha256(manifest):
        raise ValueError(f"{resolved_path}: manifest_sha256 mismatch")
    report_id = validate_report_id(str(manifest.get("report_id", "")))
    analytics_export = manifest.get("analytics_export")
    if not isinstance(analytics_export, Mapping):
        raise ValueError(f"{resolved_path}: analytics_export binding is required for indexing")
    export_run_id = str(analytics_export.get("export_run_id", ""))
    if not export_run_id:
        raise ValueError(f"{resolved_path}: analytics_export.export_run_id is required")
    parent = registry.conn.execute(
        """
        SELECT export_manifest_path, output_root, collection_id,
               collection_manifest_sha256
        FROM analytics_exports
        WHERE export_run_id = ?
        """,
        (export_run_id,),
    ).fetchone()
    if parent is None:
        raise LookupError(f"Analytics export {export_run_id!r} is not indexed")
    parent_manifest_path = Path(str(parent["export_manifest_path"])).expanduser().resolve()
    bound_manifest_path = Path(
        str(analytics_export.get("export_manifest_path", ""))
    ).expanduser().resolve()
    if bound_manifest_path != parent_manifest_path:
        raise ValueError(
            f"{resolved_path}: bound analytics manifest path does not match the registry"
        )
    actual_parent_sha256 = _file_sha256(parent_manifest_path)
    if analytics_export.get("export_manifest_content_sha256") != actual_parent_sha256:
        raise ValueError(
            f"{resolved_path}: bound analytics manifest content hash no longer matches"
        )
    for key in ("collection_id", "collection_manifest_sha256"):
        registered = parent[key]
        bound = analytics_export.get(key)
        if (str(registered) if registered is not None else None) != (
            str(bound) if bound is not None else None
        ):
            raise ValueError(f"{resolved_path}: analytics_export.{key} does not match the registry")
    available_rows = registry.conn.execute(
        """
        SELECT table_name FROM analytics_export_tables
        WHERE export_run_id = ? ORDER BY table_name
        """,
        (export_run_id,),
    ).fetchall()
    available_tables = {str(row["table_name"]) for row in available_rows}
    bound_tables = {str(value) for value in analytics_export.get("available_tables", [])}
    if bound_tables != available_tables:
        raise ValueError(
            f"{resolved_path}: bound available_tables no longer match the registry"
        )
    source_tables = {str(value) for value in manifest.get("source_tables", [])}
    if not source_tables.issubset(available_tables):
        raise ValueError(
            f"{resolved_path}: source_tables are not present in the bound analytics export"
        )
    registry.upsert_analytics_report(
        export_run_id=export_run_id,
        report_id=report_id,
        report_manifest_path=resolved_path,
        report_manifest_sha256=str(manifest["manifest_sha256"]),
        output_root=resolved_path.parent,
        schema_id=str(manifest.get("schema_id", "")),
        schema_version=int(manifest.get("schema_version", 0)),
        materialization_policy=str(manifest.get("materialization_policy", "")),
        source_backends=[str(v) for v in manifest.get("source_backends", [])],
        source_tables=[str(v) for v in manifest.get("source_tables", [])],
        visualization_summaries=_visualization_summaries(manifest),
        artifact_count=int(manifest.get("artifact_count", 0)),
        nonready_count=int(manifest.get("nonready_count", 0)),
        created_at_utc=(
            str(manifest["created_at_utc"])
            if manifest.get("created_at_utc") is not None
            else None
        ),
        status=status,
        metadata={
            "analytics_export_manifest_content_sha256": analytics_export.get(
                "export_manifest_content_sha256"
            ),
            "source_report_plan_sha256": manifest.get("source_report_plan_sha256"),
        },
    )
    return export_run_id, report_id


def query_indexed_reports(
    registry_path: Path,
    *,
    export_run_id: str | None = None,
    report_id: str | None = None,
    visualization_id: str | None = None,
    status: str = "active",
    latest: bool = False,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Read indexed report summaries without opening recording Zarrs."""

    parameters: list[Any] = []
    sql = ["SELECT DISTINCT aro.* FROM analytics_report_overview aro"]
    if visualization_id:
        sql.append(
            "JOIN analytics_report_visualizations arv "
            "ON arv.export_run_id = aro.export_run_id AND arv.report_id = aro.report_id"
        )
    sql.append("WHERE 1=1")
    for column, value in (("export_run_id", export_run_id), ("report_id", report_id)):
        if value:
            sql.append(f"AND aro.{column} = ?")
            parameters.append(str(value))
    if visualization_id:
        sql.append("AND arv.visualization_id = ?")
        parameters.append(str(visualization_id))
    if status != "any":
        sql.append("AND aro.status = ?")
        parameters.append(str(status))
    sql.append("ORDER BY COALESCE(aro.created_at_utc, aro.indexed_utc) DESC, aro.report_id DESC")
    effective_limit = 1 if latest else int(limit)
    if effective_limit > 0:
        sql.append("LIMIT ?")
        parameters.append(effective_limit)
    with closing(_connect_read_only(registry_path)) as connection:
        return [
            dict(row) for row in connection.execute("\n".join(sql), parameters).fetchall()
        ]


def check_report_manifest(
    manifest_path: Path,
    *,
    check_files: bool = False,
) -> dict[str, Any]:
    """Check report and parent-manifest hashes, optionally checking artifact files."""

    resolved_path = manifest_path.expanduser().resolve()
    errors: list[str] = []
    try:
        manifest = _load_json_object(resolved_path)
    except Exception as exc:
        return {
            "manifest_path": str(resolved_path),
            "ok": False,
            "errors": [str(exc)],
        }
    if not verify_report_manifest_sha256(manifest):
        errors.append("report manifest_sha256 mismatch")
    analytics_export = manifest.get("analytics_export")
    if isinstance(analytics_export, Mapping):
        parent_value = analytics_export.get("export_manifest_path")
        if not isinstance(parent_value, str) or not parent_value:
            errors.append("analytics export manifest path is missing")
        else:
            parent_path = Path(parent_value).expanduser().resolve()
            if not parent_path.is_file():
                errors.append(f"analytics export manifest is missing: {parent_path}")
            else:
                expected = analytics_export.get("export_manifest_content_sha256")
                actual = _file_sha256(parent_path)
                if expected != actual:
                    errors.append("analytics export manifest content hash mismatch")
    if check_files:
        for index, item in enumerate(manifest.get("artifacts", [])):
            if not isinstance(item, Mapping):
                errors.append(f"artifact {index} is not an object")
                continue
            materialized = item.get("materialized")
            if not isinstance(materialized, Mapping):
                errors.append(f"artifact {index} has no materialized reference")
                continue
            relative_path = materialized.get("relative_path")
            if isinstance(relative_path, str) and relative_path:
                artifact_path = resolved_path.parent / relative_path
                if not artifact_path.is_file():
                    errors.append(f"copied artifact is missing: {artifact_path}")
                    continue
                expected = materialized.get("content_sha256")
                if expected is not None and _file_sha256(artifact_path) != expected:
                    errors.append(f"copied artifact hash mismatch: {artifact_path}")
                continue
            zarr_value = materialized.get("zarr_path")
            if isinstance(zarr_value, str) and zarr_value:
                zarr_path = Path(zarr_value).expanduser().resolve()
                if not zarr_path.exists():
                    errors.append(f"referenced Zarr is missing: {zarr_path}")
            else:
                errors.append(f"artifact {index} has no copied path or Zarr path")
    return {
        "manifest_path": str(resolved_path),
        "report_id": manifest.get("report_id"),
        "export_run_id": (
            analytics_export.get("export_run_id")
            if isinstance(analytics_export, Mapping)
            else None
        ),
        "artifact_count": manifest.get("artifact_count"),
        "checked_files": bool(check_files),
        "ok": not errors,
        "errors": errors,
    }


__all__ = [
    "AnalyticsExportBinding",
    "check_report_manifest",
    "index_report_manifest",
    "query_indexed_reports",
    "report_output_dir",
    "resolve_analytics_export_binding",
    "validate_report_id",
]
