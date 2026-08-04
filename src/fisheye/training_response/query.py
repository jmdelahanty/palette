"""Lazy queries for immutable whole-training response analytics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import polars as pl

from fisheye.analytics_exports.derived_publication import (
    derived_manifest_path,
    derived_manifest_selected_parts,
    validate_derived_manifest_envelope,
)

from .contracts import (
    ARROW_CONTRACT_ENVELOPE_SCHEMA_ID,
    ARROW_CONTRACT_ENVELOPE_SCHEMA_VERSION,
    ARROW_TABLE_CONTRACTS,
    IDENTITY_COLUMNS,
    LEGACY_SCHEMA_VERSION,
    SCHEMA_ID,
    SCHEMA_VERSION,
    TRAINING_RESPONSE_CLASSIFICATION_TABLE,
    TRAINING_RESPONSE_CLUSTERS_TABLE,
    TRAINING_RESPONSE_FEATURES_TABLE,
    TRAINING_RESPONSE_TABLES,
    contract_fields,
)
from .validation import validate_training_response_run


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
    output_root: Path,
    analysis_run_id: str,
    *,
    allow_legacy_layout: bool = False,
) -> dict[str, Any]:
    root = Path(output_root).expanduser().resolve()
    run_id = _safe_component(analysis_run_id, label="analysis run ID")
    version = "v1" if allow_legacy_layout else "v2"
    path = (root / version / "manifests" / f"analysis_run_id={run_id}.json").resolve()
    if not _within(path, root):
        raise PermissionError("training-response manifest resolves outside output root")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("training-response manifest must contain an object")
    expected_version = LEGACY_SCHEMA_VERSION if allow_legacy_layout else SCHEMA_VERSION
    if payload.get("schema_id") != SCHEMA_ID or payload.get(
        "schema_version"
    ) != expected_version:
        raise ValueError(f"unsupported training-response schema in {path}")
    if payload.get("analysis_run_id") != run_id:
        raise ValueError(f"analysis_run_id mismatch in {path}")
    if not allow_legacy_layout:
        validate_training_response_run(root, run_id)
    return payload


def discover_training_response_catalog(output_root: Path) -> TrainingResponseCatalog:
    """Discover immutable training-response runs from manifests only."""

    root = Path(output_root).expanduser().resolve()
    manifest_dir = (root / "v2" / "manifests").resolve()
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
            validate_training_response_run(root, run_id)
            row_counts = payload.get("row_counts_by_table")
            if not isinstance(row_counts, Mapping):
                raise ValueError("manifest row-count inventory is missing")
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
                    missing_part_count=0,
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
    output_root: Path,
    analysis_run_id: str,
    table_name: str,
    *,
    allow_legacy_layout: bool = False,
) -> tuple[Path, ...]:
    root = Path(output_root).expanduser().resolve()
    run_id = _safe_component(analysis_run_id, label="analysis run ID")
    table = _safe_component(table_name, label="table name")
    contract_fields(table)
    if allow_legacy_layout:
        payload = load_training_response_manifest(
            root,
            run_id,
            allow_legacy_layout=True,
        )
        parts_by_table = payload.get("part_files_by_table")
        if not isinstance(parts_by_table, Mapping):
            raise ValueError("legacy manifest part inventory is missing")
        raw_parts = parts_by_table.get(table, [])
        if not isinstance(raw_parts, list):
            raise ValueError(f"legacy manifest part list is not an array for {table}")
        table_root = root / "v1" / table / f"analysis_run_id={run_id}"
        parts: list[Path] = []
        for raw_part in raw_parts:
            path = (table_root / Path(str(raw_part)).name).resolve()
            try:
                path.relative_to(root)
            except ValueError as exc:
                raise PermissionError(
                    f"legacy part resolves outside output root: {raw_part}"
                ) from exc
            if not path.is_file():
                raise FileNotFoundError(path)
            parts.append(path)
        return tuple(parts)
    manifest_path = derived_manifest_path(root, run_id)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("training-response manifest must contain an object")
    if (
        payload.get("schema_id") != SCHEMA_ID
        or payload.get("schema_version") != SCHEMA_VERSION
    ):
        raise ValueError("strict training-response reader requires schema v2")
    validate_derived_manifest_envelope(
        payload,
        analysis_run_id=run_id,
        table_names=TRAINING_RESPONSE_TABLES,
        contracts=ARROW_TABLE_CONTRACTS,
        arrow_envelope_schema_id=ARROW_CONTRACT_ENVELOPE_SCHEMA_ID,
        arrow_envelope_schema_version=ARROW_CONTRACT_ENVELOPE_SCHEMA_VERSION,
    )
    return derived_manifest_selected_parts(
        root,
        payload,
        table,
        table_names=TRAINING_RESPONSE_TABLES,
    )


def scan_training_response_table(
    output_root: Path,
    analysis_run_id: str,
    table_name: str,
    *,
    columns: Sequence[str] | None = None,
    allow_legacy_layout: bool = False,
) -> pl.LazyFrame:
    parts = training_response_table_parts(
        output_root,
        analysis_run_id,
        table_name,
        allow_legacy_layout=allow_legacy_layout,
    )
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
    cluster_columns = (
        *keys,
        "cluster_status",
        "cluster_reason",
        "cluster_id",
        "cluster_probability",
        "selected_component_count",
        "selected_bic",
        "bic_by_component_count_json",
        "cluster_stability_median_ari",
        "cluster_stability_threshold",
        "cluster_stability_resample_count",
        "cluster_min_rows_per_component",
        "cluster_axes",
        "cluster_semantics",
    )
    clusters = scan_training_response_table(
        output_root, analysis_run_id, TRAINING_RESPONSE_CLUSTERS_TABLE
    )
    available_cluster_columns = set(clusters.collect_schema().names())
    clusters = clusters.select(
        [
            (
                pl.col(column)
                if column in available_cluster_columns
                else pl.lit(None).alias(column)
            )
            for column in cluster_columns
        ]
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
