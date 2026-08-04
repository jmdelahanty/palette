"""Fail-closed validation for whole-training response v2 generations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from fisheye.analytics_exports.derived_publication import (
    derived_manifest_path,
    derived_manifest_selected_parts,
    validate_derived_manifest_envelope,
)

from .contracts import (
    ARROW_CONTRACT_ENVELOPE_SCHEMA_ID,
    ARROW_CONTRACT_ENVELOPE_SCHEMA_VERSION,
    ARROW_TABLE_CONTRACTS,
    SCHEMA_ID,
    SCHEMA_VERSION,
    TRAINING_RESPONSE_TABLES,
    TrainingResponseConfig,
    normalize_training_response_rows,
    validate_training_response_arrow_schema,
)


_MANIFEST_FIELDS = {
    "schema_id",
    "schema_version",
    "analysis_run_id",
    "created_at_utc",
    "source_export_root",
    "source_export_run_id",
    "source_export_manifest_sha256",
    "source_collection_manifest_sha256",
    "source_validation",
    "feature_config",
    "output_tables",
    "row_counts_by_table",
    "part_files_by_table",
    "primary_keys_by_table",
    "arrow_schema_contracts",
    "publication",
    "manifest_payload_sha256",
    "source_export_mutated",
    "interpretation_guardrail",
    "temporal_adaptation_status",
}


class TrainingResponseValidationError(ValueError):
    pass


def _component(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    text = value.strip()
    if text != value or not text or Path(text).name != text or text in {".", ".."}:
        raise ValueError(f"invalid {label}: {value!r}")
    return text


def _validate_payload(
    root: Path,
    run_id: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    import pyarrow.parquet as pq

    if set(payload) != _MANIFEST_FIELDS:
        raise ValueError("training-response manifest has an unexpected field set")
    if payload.get("schema_id") != SCHEMA_ID:
        raise ValueError(f"unsupported schema_id {payload.get('schema_id')!r}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported schema_version {payload.get('schema_version')!r}"
        )
    if payload.get("source_export_mutated") is not False:
        raise ValueError("manifest does not declare source_export_mutated=false")
    source_run_id = _component(
        payload.get("source_export_run_id"),
        label="source export run ID",
    )
    raw_config = payload.get("feature_config")
    if not isinstance(raw_config, Mapping):
        raise ValueError("training-response feature config is missing")
    expected_config_fields = set(TrainingResponseConfig().to_dict())
    if set(raw_config) != expected_config_fields:
        raise ValueError("training-response config field set is invalid")
    config = TrainingResponseConfig(**dict(raw_config))
    if config.to_dict() != dict(raw_config):
        raise ValueError("training-response config is not canonical")
    validate_derived_manifest_envelope(
        payload,
        analysis_run_id=run_id,
        table_names=TRAINING_RESPONSE_TABLES,
        contracts=ARROW_TABLE_CONTRACTS,
        arrow_envelope_schema_id=ARROW_CONTRACT_ENVELOPE_SCHEMA_ID,
        arrow_envelope_schema_version=ARROW_CONTRACT_ENVELOPE_SCHEMA_VERSION,
    )
    publication = payload["publication"]
    assert isinstance(publication, Mapping)
    if publication.get("selector_eligible") is not True:
        raise ValueError("training-response publication is not selector eligible")

    checked_parts = 0
    checked_rows = 0
    for table_name in TRAINING_RESPONSE_TABLES:
        parts = derived_manifest_selected_parts(
            root,
            payload,
            table_name,
            table_names=TRAINING_RESPONSE_TABLES,
        )
        if len(parts) != 1:
            raise ValueError(f"{table_name}: exactly one selected part is required")
        parquet_file = pq.ParquetFile(parts[0])
        validate_training_response_arrow_schema(
            table_name,
            parquet_file.schema_arrow,
        )
        metadata = parquet_file.schema_arrow.metadata or {}
        expected_metadata = {
            b"palette.schema_id": SCHEMA_ID.encode("utf-8"),
            b"palette.schema_version": str(SCHEMA_VERSION).encode("ascii"),
            b"palette.table_name": table_name.encode("utf-8"),
            b"palette.training_response_config": json.dumps(
                config.to_dict(), sort_keys=True, separators=(",", ":")
            ).encode("utf-8"),
        }
        for key, value in expected_metadata.items():
            if metadata.get(key) != value:
                raise ValueError(f"{table_name}: invalid Arrow footer metadata")
        rows = [dict(row) for row in parquet_file.read().to_pylist()]
        normalized = normalize_training_response_rows(
            table_name,
            rows,
            analysis_run_id=run_id,
            config=config,
        )
        if normalized != rows:
            raise ValueError(f"{table_name}: rows are not canonical v2 values")
        if any(row["source_export_run_id"] != source_run_id for row in rows):
            raise ValueError(f"{table_name}: source export identity differs")
        checked_parts += 1
        checked_rows += len(rows)
    return {
        "status": "valid",
        "output_root": str(root),
        "analysis_run_id": run_id,
        "manifest_path": str(derived_manifest_path(root, run_id)),
        "table_count": len(TRAINING_RESPONSE_TABLES),
        "part_count": checked_parts,
        "row_count": checked_rows,
    }


def validate_training_response_run(
    output_root: Path,
    analysis_run_id: str,
) -> dict[str, Any]:
    """Validate the exact manifest, part receipts, schemas, values, and keys."""

    root = Path(output_root).expanduser().resolve()
    try:
        run_id = _component(analysis_run_id, label="analysis run ID")
        manifest_path = derived_manifest_path(root, run_id)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("training-response manifest must contain an object")
        return _validate_payload(root, run_id, payload)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        if isinstance(exc, TrainingResponseValidationError):
            raise
        raise TrainingResponseValidationError(
            f"training-response run {analysis_run_id!r} failed validation: {exc}"
        ) from exc


__all__ = ["TrainingResponseValidationError", "validate_training_response_run"]
