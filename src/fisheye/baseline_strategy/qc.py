"""Lazy, read-only query surfaces for baseline-strategy QC applications."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import polars as pl

from fisheye.analytics_exports.contracts import BASELINE_KINEMATIC_SAMPLES_TABLE
from fisheye.analytics_exports.publication import (
    export_manifest_path,
    manifest_selected_part_files_from_payload,
)
from fisheye.utils.virtual_collection_manifest import verify_manifest_sha256

from .contracts import (
    BASELINE_STRATEGY_CLASSIFICATION_TABLE,
    BASELINE_STRATEGY_CLUSTERS_TABLE,
    BASELINE_STRATEGY_FEATURES_TABLE,
    IDENTITY_COLUMNS,
    SCHEMA_ID,
    SCHEMA_VERSION,
)
from .query import scan_strategy_table


@dataclass(frozen=True)
class StrategyRunEntry:
    analysis_run_id: str
    manifest_path: str
    created_at_utc: str | None
    source_export_run_id: str
    row_count: int
    missing_part_count: int

    @property
    def ready(self) -> bool:
        return self.missing_part_count == 0

    @property
    def label(self) -> str:
        state = "ready" if self.ready else f"{self.missing_part_count} missing part(s)"
        return f"{self.analysis_run_id} · {self.row_count} rows · {state}"


@dataclass(frozen=True)
class StrategyCatalogDiagnostic:
    manifest_path: str
    code: str
    message: str


@dataclass(frozen=True)
class StrategyCatalog:
    output_root: Path
    entries: tuple[StrategyRunEntry, ...]
    diagnostics: tuple[StrategyCatalogDiagnostic, ...]


@dataclass(frozen=True)
class SourceExportContext:
    export_root: Path
    export_run_id: str
    manifest_path: Path
    manifest: Mapping[str, Any]
    recording_protocols: Mapping[str, str]


def _safe_component(value: object, *, label: str) -> str:
    text = str(value or "").strip()
    if not text or Path(text).name != text or text in {".", ".."}:
        raise ValueError(f"invalid {label}: {value!r}")
    return text


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _timestamp(value: object) -> float:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return float("-inf")


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("manifest must contain an object")
    return payload


def _load_object_snapshot(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("manifest must contain an object")
    return payload, hashlib.sha256(raw).hexdigest()


def load_strategy_manifest(output_root: Path, analysis_run_id: str) -> dict[str, Any]:
    root = Path(output_root).expanduser().resolve()
    run_id = _safe_component(analysis_run_id, label="analysis run ID")
    path = (root / "v1" / "manifests" / f"analysis_run_id={run_id}.json").resolve()
    if not _within(path, root):
        raise PermissionError("strategy manifest resolves outside output root")
    payload = _load_object(path)
    if (
        payload.get("schema_id") != SCHEMA_ID
        or payload.get("schema_version") != SCHEMA_VERSION
    ):
        raise ValueError(f"unsupported strategy schema in {path}")
    if payload.get("analysis_run_id") != run_id:
        raise ValueError(f"analysis_run_id mismatch in {path}")
    return payload


def discover_strategy_catalog(output_root: Path) -> StrategyCatalog:
    """Discover selectable immutable runs without scanning Parquet contents."""

    root = Path(output_root).expanduser().resolve()
    manifest_dir = (root / "v1" / "manifests").resolve()
    diagnostics: list[StrategyCatalogDiagnostic] = []
    entries: list[StrategyRunEntry] = []
    if not _within(manifest_dir, root) or not manifest_dir.is_dir():
        return StrategyCatalog(
            root,
            (),
            (
                StrategyCatalogDiagnostic(
                    str(manifest_dir),
                    "manifest_directory_missing",
                    "No safe strategy manifest directory exists under the authorized root.",
                ),
            ),
        )
    for path in sorted(manifest_dir.glob("analysis_run_id=*.json")):
        try:
            payload = _load_object(path)
            run_id = _safe_component(payload.get("analysis_run_id"), label="analysis run ID")
            if path.name != f"analysis_run_id={run_id}.json":
                raise ValueError("manifest filename does not match analysis_run_id")
            if payload.get("schema_id") != SCHEMA_ID:
                raise ValueError(f"unsupported schema_id {payload.get('schema_id')!r}")
            if payload.get("schema_version") != SCHEMA_VERSION:
                raise ValueError(f"unsupported schema_version {payload.get('schema_version')!r}")
            source_run_id = _safe_component(
                payload.get("source_export_run_id"),
                label="source export run ID",
            )
            row_counts = payload.get("row_counts_by_table")
            parts_by_table = payload.get("part_files_by_table")
            if not isinstance(row_counts, Mapping) or not isinstance(parts_by_table, Mapping):
                raise ValueError("manifest table inventory is missing")
            missing_parts = 0
            for table_name, raw_parts in parts_by_table.items():
                table = _safe_component(table_name, label="table name")
                if not isinstance(raw_parts, list):
                    raise ValueError(f"{table}: part inventory is not an array")
                table_root = root / "v1" / table / f"analysis_run_id={run_id}"
                for raw_part in raw_parts:
                    candidate = (table_root / Path(str(raw_part)).name).resolve()
                    if not _within(candidate, root):
                        raise PermissionError(f"{table}: part resolves outside output root")
                    missing_parts += int(not candidate.is_file())
            entries.append(
                StrategyRunEntry(
                    analysis_run_id=run_id,
                    manifest_path=str(path.resolve()),
                    created_at_utc=(
                        str(payload["created_at_utc"])
                        if payload.get("created_at_utc") is not None
                        else None
                    ),
                    source_export_run_id=source_run_id,
                    row_count=sum(int(value) for value in row_counts.values()),
                    missing_part_count=missing_parts,
                )
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            diagnostics.append(
                StrategyCatalogDiagnostic(str(path), "invalid_manifest", str(exc))
            )
    entries.sort(
        key=lambda item: (_timestamp(item.created_at_utc), item.analysis_run_id),
        reverse=True,
    )
    return StrategyCatalog(root, tuple(entries), tuple(diagnostics))


def select_strategy_run_id(catalog: StrategyCatalog, requested: str) -> str:
    ready = [entry for entry in catalog.entries if entry.ready]
    if not ready:
        raise ValueError("no ready baseline-strategy runs are available")
    if str(requested).strip().lower() in {"", "latest"}:
        return ready[0].analysis_run_id
    run_id = _safe_component(requested, label="analysis run ID")
    if not any(entry.analysis_run_id == run_id and entry.ready for entry in ready):
        raise ValueError(f"baseline-strategy run is not selectable: {run_id}")
    return run_id


def _source_table_parts(
    export_root: Path,
    export_run_id: str,
    manifest: Mapping[str, Any],
    table_name: str,
) -> tuple[Path, ...]:
    table = _safe_component(table_name, label="table name")
    if manifest.get("export_run_id") != export_run_id:
        raise ValueError("source manifest run identity mismatch")
    parts = manifest_selected_part_files_from_payload(
        export_root,
        manifest,
        table,
    )
    for path in parts:
        if not path.is_file():
            raise FileNotFoundError(path)
    return parts


def source_export_context(
    strategy_root: Path,
    analysis_run_id: str,
    *,
    authorized_export_root: Path,
) -> SourceExportContext:
    """Resolve source lineage while enforcing the app's authorized export root."""

    strategy_manifest = load_strategy_manifest(strategy_root, analysis_run_id)
    export_root = Path(authorized_export_root).expanduser().resolve()
    declared_root = (
        Path(str(strategy_manifest.get("source_export_root") or ""))
        .expanduser()
        .resolve()
    )
    if declared_root != export_root:
        raise PermissionError(
            f"strategy source root {declared_root} is not the authorized export root {export_root}"
        )
    export_run_id = _safe_component(
        strategy_manifest.get("source_export_run_id"), label="source export run ID"
    )
    manifest_path = export_manifest_path(export_root, export_run_id)
    manifest, observed_export_sha256 = _load_object_snapshot(manifest_path)
    if manifest.get("export_run_id") != export_run_id:
        raise ValueError("source export manifest run ID mismatch")
    expected_export_sha256 = str(
        strategy_manifest.get("source_export_manifest_sha256") or ""
    ).strip()
    if expected_export_sha256:
        if observed_export_sha256 != expected_export_sha256:
            raise ValueError("source export manifest SHA-256 mismatch")

    protocols: dict[str, str] = {}
    collection = manifest.get("collection_manifest")
    if isinstance(collection, Mapping) and collection.get("path"):
        collection_path = (
            export_root
            / "v1"
            / "manifests"
            / "collections"
            / Path(str(collection["path"])).name
        ).resolve()
        if not _within(collection_path, export_root):
            raise PermissionError("collection manifest resolves outside export root")
        collection_payload = _load_object(collection_path)
        expected_collection_sha256 = str(
            collection.get("manifest_sha256") or ""
        ).strip()
        strategy_collection_sha256 = str(
            strategy_manifest.get("source_collection_manifest_sha256") or ""
        ).strip()
        if (
            strategy_collection_sha256
            and strategy_collection_sha256 != expected_collection_sha256
        ):
            raise ValueError("strategy/export collection manifest SHA-256 mismatch")
        if expected_collection_sha256:
            if (
                str(collection_payload.get("manifest_sha256") or "").strip()
                != expected_collection_sha256
                or not verify_manifest_sha256(collection_payload)
            ):
                raise ValueError("source collection manifest SHA-256 mismatch")
        for record in collection_payload.get("records", []):
            if not isinstance(record, Mapping):
                continue
            recording_id = str(record.get("recording_id") or "").strip()
            protocol = record.get("protocol")
            protocol_name = (
                str(protocol.get("protocol_name") or "").strip()
                if isinstance(protocol, Mapping)
                else ""
            )
            if recording_id:
                protocols[recording_id] = protocol_name or "unknown"
    return SourceExportContext(
        export_root=export_root,
        export_run_id=export_run_id,
        manifest_path=manifest_path,
        manifest=manifest,
        recording_protocols=protocols,
    )


def scan_source_export_table(
    context: SourceExportContext,
    table_name: str,
    *,
    columns: Sequence[str] | None = None,
) -> pl.LazyFrame:
    parts = _source_table_parts(
        context.export_root,
        context.export_run_id,
        context.manifest,
        table_name,
    )
    if not parts:
        return (
            pl.DataFrame({column: [] for column in columns}).lazy()
            if columns is not None
            else pl.DataFrame().lazy()
        )
    lazy = pl.scan_parquet([str(path) for path in parts], hive_partitioning=False)
    return lazy.select(list(columns)) if columns is not None else lazy


def scan_strategy_qc_rows(
    strategy_root: Path,
    analysis_run_id: str,
    *,
    recording_protocols: Mapping[str, str] | None = None,
) -> pl.LazyFrame:
    """Join small derived tables lazily; no source kinematic samples are read."""

    keys = list(IDENTITY_COLUMNS)
    features = scan_strategy_table(
        strategy_root, analysis_run_id, BASELINE_STRATEGY_FEATURES_TABLE
    )
    classifications = scan_strategy_table(
        strategy_root,
        analysis_run_id,
        BASELINE_STRATEGY_CLASSIFICATION_TABLE,
        columns=(
            *keys,
            "classification_status",
            "classification_reason",
            "activity_state",
            "boundary_strategy",
            "spatial_organization",
            "temporal_pattern",
            "primary_strategy",
            "classification_confidence_score",
        ),
    )
    clusters = scan_strategy_table(
        strategy_root,
        analysis_run_id,
        BASELINE_STRATEGY_CLUSTERS_TABLE,
        columns=(
            *keys,
            "cluster_status",
            "cluster_reason",
            "cluster_id",
            "cluster_probability",
        ),
    )
    joined = features.join(classifications, on=keys, how="left").join(
        clusters, on=keys, how="left"
    )
    protocols = recording_protocols or {}
    protocol_frame = pl.DataFrame(
        {
            "recording_id": list(protocols),
            "protocol_name": [protocols[key] for key in protocols],
        }
    ).lazy()
    if protocols:
        joined = joined.join(protocol_frame, on="recording_id", how="left")
    else:
        joined = joined.with_columns(pl.lit("unknown").alias("protocol_name"))
    return joined.with_columns(pl.col("protocol_name").fill_null("unknown"))


def scan_recording_baseline_samples(
    context: SourceExportContext, recording_id: str
) -> pl.LazyFrame:
    recording = str(recording_id).strip()
    if not recording:
        raise ValueError("recording_id is required")
    columns = (
        "recording_id",
        "relative_time_s",
        "source_frame",
        "x_arena_mm",
        "y_arena_mm",
        "speed_mm_s",
        "wall",
        "position_valid",
        "sample_valid",
    )
    return (
        scan_source_export_table(
            context, BASELINE_KINEMATIC_SAMPLES_TABLE, columns=columns
        )
        .filter(pl.col("recording_id") == recording)
        .sort(["relative_time_s", "source_frame"])
    )


__all__ = [
    "SourceExportContext",
    "StrategyCatalog",
    "StrategyCatalogDiagnostic",
    "StrategyRunEntry",
    "discover_strategy_catalog",
    "load_strategy_manifest",
    "scan_recording_baseline_samples",
    "scan_source_export_table",
    "scan_strategy_qc_rows",
    "select_strategy_run_id",
    "source_export_context",
]
