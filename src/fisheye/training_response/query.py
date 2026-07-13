"""Lazy queries for immutable whole-training response analytics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import polars as pl

from .contracts import (
    IDENTITY_COLUMNS,
    SCHEMA_ID,
    SCHEMA_VERSION,
    TRAINING_RESPONSE_CLASSIFICATION_TABLE,
    TRAINING_RESPONSE_CLUSTERS_TABLE,
    TRAINING_RESPONSE_FEATURES_TABLE,
    contract_fields,
)


@dataclass(frozen=True)
class TrainingResponseRunEntry:
    analysis_run_id: str
    manifest_path: str
    created_at_utc: str | None
    source_export_run_id: str
    source_export_manifest_sha256: str | None
    source_collection_manifest_sha256: str | None
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
class TrainingResponseCatalogDiagnostic:
    manifest_path: str
    code: str
    message: str


@dataclass(frozen=True)
class TrainingResponseCatalog:
    output_root: Path
    entries: tuple[TrainingResponseRunEntry, ...]
    diagnostics: tuple[TrainingResponseCatalogDiagnostic, ...]


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


def load_training_response_manifest(
    output_root: Path, analysis_run_id: str
) -> dict[str, Any]:
    root = Path(output_root).expanduser().resolve()
    run_id = _safe_component(analysis_run_id, label="analysis run ID")
    path = (root / "v1" / "manifests" / f"analysis_run_id={run_id}.json").resolve()
    if not _within(path, root):
        raise PermissionError("training-response manifest resolves outside output root")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("training-response manifest must contain an object")
    if (
        payload.get("schema_id") != SCHEMA_ID
        or payload.get("schema_version") != SCHEMA_VERSION
    ):
        raise ValueError(f"unsupported training-response schema in {path}")
    if payload.get("analysis_run_id") != run_id:
        raise ValueError(f"analysis_run_id mismatch in {path}")
    return payload


def discover_training_response_catalog(output_root: Path) -> TrainingResponseCatalog:
    """Discover immutable training-response runs from manifests only."""

    root = Path(output_root).expanduser().resolve()
    manifest_dir = (root / "v1" / "manifests").resolve()
    diagnostics: list[TrainingResponseCatalogDiagnostic] = []
    entries: list[TrainingResponseRunEntry] = []
    if not _within(manifest_dir, root) or not manifest_dir.is_dir():
        return TrainingResponseCatalog(
            root,
            (),
            (
                TrainingResponseCatalogDiagnostic(
                    str(manifest_dir),
                    "manifest_directory_missing",
                    "No safe training-response manifest directory exists under the authorized root.",
                ),
            ),
        )
    for path in sorted(manifest_dir.glob("analysis_run_id=*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, Mapping):
                raise ValueError("manifest must contain an object")
            run_id = _safe_component(payload.get("analysis_run_id"), label="analysis run ID")
            if path.name != f"analysis_run_id={run_id}.json":
                raise ValueError("manifest filename does not match analysis_run_id")
            if payload.get("schema_id") != SCHEMA_ID:
                raise ValueError(f"unsupported schema_id {payload.get('schema_id')!r}")
            if payload.get("schema_version") != SCHEMA_VERSION:
                raise ValueError(
                    f"unsupported schema_version {payload.get('schema_version')!r}"
                )
            source_run_id = _safe_component(
                payload.get("source_export_run_id"), label="source export run ID"
            )
            row_counts = payload.get("row_counts_by_table")
            parts_by_table = payload.get("part_files_by_table")
            if not isinstance(row_counts, Mapping) or not isinstance(parts_by_table, Mapping):
                raise ValueError("manifest table inventory is missing")
            missing_parts = 0
            for table_name, raw_parts in parts_by_table.items():
                table = _safe_component(table_name, label="table name")
                contract_fields(table)
                if not isinstance(raw_parts, list):
                    raise ValueError(f"{table}: part inventory is not an array")
                table_root = root / "v1" / table / f"analysis_run_id={run_id}"
                for raw_part in raw_parts:
                    candidate = (table_root / Path(str(raw_part)).name).resolve()
                    if not _within(candidate, root):
                        raise PermissionError(f"{table}: part resolves outside output root")
                    missing_parts += int(not candidate.is_file())
            entries.append(
                TrainingResponseRunEntry(
                    analysis_run_id=run_id,
                    manifest_path=str(path.resolve()),
                    created_at_utc=(
                        str(payload["created_at_utc"])
                        if payload.get("created_at_utc") is not None
                        else None
                    ),
                    source_export_run_id=source_run_id,
                    source_export_manifest_sha256=(
                        str(payload["source_export_manifest_sha256"])
                        if payload.get("source_export_manifest_sha256") is not None
                        else None
                    ),
                    source_collection_manifest_sha256=(
                        str(payload["source_collection_manifest_sha256"])
                        if payload.get("source_collection_manifest_sha256") is not None
                        else None
                    ),
                    row_count=sum(int(value) for value in row_counts.values()),
                    missing_part_count=missing_parts,
                )
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError, KeyError) as exc:
            diagnostics.append(
                TrainingResponseCatalogDiagnostic(str(path), "invalid_manifest", str(exc))
            )
    entries.sort(
        key=lambda item: (_timestamp(item.created_at_utc), item.analysis_run_id),
        reverse=True,
    )
    return TrainingResponseCatalog(root, tuple(entries), tuple(diagnostics))


def select_training_response_run_id(
    catalog: TrainingResponseCatalog,
    requested: str,
    *,
    source_export_run_id: str | None = None,
    source_export_manifest_sha256: str | None = None,
    source_collection_manifest_sha256: str | None = None,
) -> str:
    ready = [
        entry
        for entry in catalog.entries
        if entry.ready
        and (
            source_export_run_id is None
            or entry.source_export_run_id == source_export_run_id
        )
        and (
            source_export_manifest_sha256 is None
            or entry.source_export_manifest_sha256
            == source_export_manifest_sha256
        )
        and (
            source_collection_manifest_sha256 is None
            or entry.source_collection_manifest_sha256
            == source_collection_manifest_sha256
        )
    ]
    if not ready:
        suffix = (
            f" for source export {source_export_run_id!r}"
            if source_export_run_id is not None
            else ""
        )
        raise ValueError(f"no ready training-response runs are available{suffix}")
    if str(requested).strip().lower() in {"", "latest"}:
        return ready[0].analysis_run_id
    run_id = _safe_component(requested, label="analysis run ID")
    if not any(entry.analysis_run_id == run_id for entry in ready):
        raise ValueError(f"training-response run is not selectable: {run_id}")
    return run_id


def training_response_table_parts(
    output_root: Path, analysis_run_id: str, table_name: str
) -> tuple[Path, ...]:
    root = Path(output_root).expanduser().resolve()
    run_id = _safe_component(analysis_run_id, label="analysis run ID")
    table = _safe_component(table_name, label="table name")
    contract_fields(table)
    manifest_path = root / "v1" / "manifests" / f"analysis_run_id={run_id}.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_parts = payload.get("part_files_by_table", {}).get(table, [])
    if not isinstance(raw_parts, list):
        raise ValueError(f"manifest part list is not an array for {table}")
    table_root = root / "v1" / table / f"analysis_run_id={run_id}"
    parts: list[Path] = []
    for raw_part in raw_parts:
        path = (table_root / Path(str(raw_part)).name).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise PermissionError(f"part resolves outside output root: {raw_part}") from exc
        if not path.is_file():
            raise FileNotFoundError(path)
        parts.append(path)
    return tuple(parts)


def scan_training_response_table(
    output_root: Path,
    analysis_run_id: str,
    table_name: str,
    *,
    columns: Sequence[str] | None = None,
) -> pl.LazyFrame:
    parts = training_response_table_parts(output_root, analysis_run_id, table_name)
    if not parts:
        return (
            pl.DataFrame({column: [] for column in columns}).lazy()
            if columns is not None
            else pl.DataFrame().lazy()
        )
    lazy = pl.scan_parquet([str(path) for path in parts], hive_partitioning=False)
    return lazy.select(list(columns)) if columns is not None else lazy


def scan_training_response_qc_rows(
    output_root: Path, analysis_run_id: str
) -> pl.LazyFrame:
    """Join the three small derived tables lazily for interactive QC."""

    keys = list(IDENTITY_COLUMNS)
    features = scan_training_response_table(
        output_root, analysis_run_id, TRAINING_RESPONSE_FEATURES_TABLE
    )
    classifications = scan_training_response_table(
        output_root,
        analysis_run_id,
        TRAINING_RESPONSE_CLASSIFICATION_TABLE,
        columns=(
            *keys,
            "classification_status",
            "classification_reason",
            "locomotor_response",
            "boundary_response",
            "aggressive_proximity_state",
            "role_distance_selectivity",
            "close_contact_vigor",
            "primary_training_profile",
            "locomotor_response_score",
            "boundary_response_score",
            "aggressive_proximity_score",
            "role_distance_selectivity_score",
            "close_contact_vigor_score",
            "profile_separation_score",
        ),
    )
    clusters = scan_training_response_table(
        output_root,
        analysis_run_id,
        TRAINING_RESPONSE_CLUSTERS_TABLE,
        columns=(
            *keys,
            "cluster_status",
            "cluster_reason",
            "cluster_id",
            "cluster_probability",
        ),
    )
    return features.join(classifications, on=keys, how="left").join(
        clusters, on=keys, how="left"
    )


__all__ = [
    "TrainingResponseCatalog",
    "TrainingResponseCatalogDiagnostic",
    "TrainingResponseRunEntry",
    "discover_training_response_catalog",
    "load_training_response_manifest",
    "scan_training_response_qc_rows",
    "scan_training_response_table",
    "select_training_response_run_id",
    "training_response_table_parts",
]
