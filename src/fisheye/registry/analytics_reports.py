"""Registry persistence for immutable analytics report manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from fisheye.shared.batch_logging import utc_now


def _json_dumps(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, sort_keys=True)


class RegistryAnalyticsReportMixin:
    """Add normalized analytics-report persistence to a registry connection."""

    def _ensure_analytics_report_tables(self) -> None:
        """Create the analytics-report child index on older ad-hoc registries."""

        if not self._table_exists("analytics_reports"):
            self._migration_062_analytics_report_registry()

    def upsert_analytics_report(
        self,
        *,
        export_run_id: str,
        report_id: str,
        report_manifest_path: Path,
        report_manifest_sha256: str,
        output_root: Path,
        schema_id: str,
        schema_version: int,
        materialization_policy: str,
        source_backends: Sequence[str],
        source_tables: Sequence[str],
        visualization_summaries: Sequence[Mapping[str, Any]],
        artifact_count: int,
        nonready_count: int,
        created_at_utc: str | None = None,
        status: str = "active",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Index one immutable report manifest without expanding per-tile rows."""

        self._ensure_analytics_report_tables()
        indexed_utc = utc_now()
        report_payload = {
            "export_run_id": str(export_run_id),
            "report_id": str(report_id),
            "report_manifest_path": str(report_manifest_path),
            "report_manifest_sha256": str(report_manifest_sha256),
            "output_root": str(output_root),
            "schema_id": str(schema_id),
            "schema_version": int(schema_version),
            "materialization_policy": str(materialization_policy),
            "source_backends_json": _json_dumps(
                sorted({str(value) for value in source_backends})
            ),
            "source_tables_json": _json_dumps(
                sorted({str(value) for value in source_tables})
            ),
            "visualization_count": len(visualization_summaries),
            "artifact_count": int(artifact_count),
            "nonready_count": int(nonready_count),
            "created_at_utc": created_at_utc,
            "indexed_utc": indexed_utc,
            "status": str(status),
            "metadata_json": _json_dumps(metadata),
        }
        existing = self.conn.execute(
            """
            SELECT report_manifest_path, report_manifest_sha256
            FROM analytics_reports
            WHERE export_run_id = ? AND report_id = ?;
            """,
            (str(export_run_id), str(report_id)),
        ).fetchone()
        if existing is not None and (
            str(existing["report_manifest_sha256"]) != str(report_manifest_sha256)
            or Path(str(existing["report_manifest_path"])).expanduser().resolve()
            != Path(report_manifest_path).expanduser().resolve()
        ):
            raise ValueError(
                f"Immutable report identity {export_run_id!r}/{report_id!r} is "
                "already indexed with a different manifest path or hash"
            )
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO analytics_reports (
                    export_run_id, report_id, report_manifest_path,
                    report_manifest_sha256, output_root, schema_id, schema_version,
                    materialization_policy, source_backends_json, source_tables_json,
                    visualization_count, artifact_count, nonready_count,
                    created_at_utc, indexed_utc, status, metadata_json
                )
                VALUES (
                    :export_run_id, :report_id, :report_manifest_path,
                    :report_manifest_sha256, :output_root, :schema_id, :schema_version,
                    :materialization_policy, :source_backends_json, :source_tables_json,
                    :visualization_count, :artifact_count, :nonready_count,
                    :created_at_utc, :indexed_utc, :status, :metadata_json
                )
                ON CONFLICT(export_run_id, report_id) DO UPDATE SET
                    report_manifest_path=excluded.report_manifest_path,
                    report_manifest_sha256=excluded.report_manifest_sha256,
                    output_root=excluded.output_root,
                    schema_id=excluded.schema_id,
                    schema_version=excluded.schema_version,
                    materialization_policy=excluded.materialization_policy,
                    source_backends_json=excluded.source_backends_json,
                    source_tables_json=excluded.source_tables_json,
                    visualization_count=excluded.visualization_count,
                    artifact_count=excluded.artifact_count,
                    nonready_count=excluded.nonready_count,
                    created_at_utc=excluded.created_at_utc,
                    indexed_utc=excluded.indexed_utc,
                    status=excluded.status,
                    metadata_json=excluded.metadata_json;
                """,
                report_payload,
            )
            self.conn.execute(
                """
                DELETE FROM analytics_report_visualizations
                WHERE export_run_id = ? AND report_id = ?;
                """,
                (str(export_run_id), str(report_id)),
            )
            for summary in visualization_summaries:
                self.conn.execute(
                    """
                    INSERT INTO analytics_report_visualizations (
                        export_run_id, report_id, visualization_id, provider_id,
                        label, visualization_contract_id, renderer,
                        renderer_version, source_backends_json, artifact_count,
                        nonready_count, materialized_paths_json, indexed_utc
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        str(export_run_id),
                        str(report_id),
                        str(summary["visualization_id"]),
                        summary.get("provider_id"),
                        summary.get("label"),
                        summary.get("visualization_contract_id"),
                        summary.get("renderer"),
                        summary.get("renderer_version"),
                        _json_dumps(
                            sorted(
                                {
                                    str(value)
                                    for value in summary.get("source_backends", [])
                                }
                            )
                        ),
                        int(summary.get("artifact_count", 0)),
                        int(summary.get("nonready_count", 0)),
                        _json_dumps(
                            [str(value) for value in summary.get("materialized_paths", [])]
                        ),
                        indexed_utc,
                    ),
                )


__all__ = ["RegistryAnalyticsReportMixin"]
