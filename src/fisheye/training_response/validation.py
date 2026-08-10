"""Fail-closed validation for v3 and explicit legacy-v2 generations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from fisheye.analytics_exports.derived_publication import (
    derived_manifest_path,
    derived_manifest_selected_parts,
    validate_derived_manifest_envelope,
)
from fisheye.analytics_exports.contracts import (
    CHASER_DISTANCE_SUMMARY_TABLE,
    CHASER_EGOCENTRIC_SUMMARY_TABLE,
    CHASER_EPOCH_BEHAVIOR_TABLE,
    CHASER_SPEED_DISTANCE_TABLE,
)
from fisheye.analytics_exports.publication import (
    export_manifest_path,
    manifest_selected_part_files_from_payload,
)

from .contracts import (
    ARROW_CONTRACT_ENVELOPE_SCHEMA_ID,
    ARROW_CONTRACT_ENVELOPE_SCHEMA_VERSION,
    ARROW_TABLE_CONTRACTS,
    LEGACY_ARROW_CONTRACT_ENVELOPE_SCHEMA_VERSION,
    LEGACY_EXACT_SCHEMA_VERSION,
    LEGACY_V2_ARROW_TABLE_CONTRACTS,
    SCHEMA_ID,
    SCHEMA_VERSION,
    TRAINING_RESPONSE_TABLES,
    TrainingResponseConfig,
    normalize_training_response_rows,
    validate_training_response_arrow_schema,
    validate_training_response_primary_keys,
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
    "source_registry_identity_receipt",
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
_LEGACY_V2_MANIFEST_FIELDS = _MANIFEST_FIELDS - {"source_registry_identity_receipt"}


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
    *,
    schema_version: int = SCHEMA_VERSION,
) -> dict[str, Any]:
    import pyarrow.parquet as pq

    manifest_fields = (
        _MANIFEST_FIELDS
        if schema_version == SCHEMA_VERSION
        else _LEGACY_V2_MANIFEST_FIELDS
    )
    if set(payload) != manifest_fields:
        raise ValueError("training-response manifest has an unexpected field set")
    if payload.get("schema_id") != SCHEMA_ID:
        raise ValueError(f"unsupported schema_id {payload.get('schema_id')!r}")
    if payload.get("schema_version") != schema_version:
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
    if schema_version == SCHEMA_VERSION:
        contracts = ARROW_TABLE_CONTRACTS
        envelope_version = ARROW_CONTRACT_ENVELOPE_SCHEMA_VERSION
    elif schema_version == LEGACY_EXACT_SCHEMA_VERSION:
        contracts = LEGACY_V2_ARROW_TABLE_CONTRACTS
        envelope_version = LEGACY_ARROW_CONTRACT_ENVELOPE_SCHEMA_VERSION
    else:  # pragma: no cover - public entry points close this choice.
        raise ValueError("unsupported exact training-response schema version")
    validate_derived_manifest_envelope(
        payload,
        analysis_run_id=run_id,
        table_names=TRAINING_RESPONSE_TABLES,
        contracts=contracts,
        arrow_envelope_schema_id=ARROW_CONTRACT_ENVELOPE_SCHEMA_ID,
        arrow_envelope_schema_version=envelope_version,
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
            schema_version=schema_version,
        )
        metadata = parquet_file.schema_arrow.metadata or {}
        expected_metadata = {
            b"palette.schema_id": SCHEMA_ID.encode("utf-8"),
            b"palette.schema_version": str(schema_version).encode("ascii"),
            b"palette.table_name": table_name.encode("utf-8"),
            b"palette.training_response_config": json.dumps(
                config.to_dict(), sort_keys=True, separators=(",", ":")
            ).encode("utf-8"),
        }
        for key, value in expected_metadata.items():
            if metadata.get(key) != value:
                raise ValueError(f"{table_name}: invalid Arrow footer metadata")
        rows = [dict(row) for row in parquet_file.read().to_pylist()]
        if schema_version == SCHEMA_VERSION:
            normalized = normalize_training_response_rows(
                table_name,
                rows,
                analysis_run_id=run_id,
                config=config,
            )
            if normalized != rows:
                raise ValueError(f"{table_name}: rows are not canonical v3 values")
        else:
            validate_training_response_primary_keys(
                table_name,
                rows,
                schema_version=LEGACY_EXACT_SCHEMA_VERSION,
            )
            if any(
                row.get("schema_id") != SCHEMA_ID
                or row.get("schema_version") != LEGACY_EXACT_SCHEMA_VERSION
                or row.get("table_name") != table_name
                or row.get("method") != "whole_training_chaser_response"
                or row.get("method_version") != "1"
                or row.get("analysis_run_id") != run_id
                for row in rows
            ):
                raise ValueError(f"{table_name}: legacy v2 row identity is invalid")
        if any(row["source_export_run_id"] != source_run_id for row in rows):
            raise ValueError(f"{table_name}: source export identity differs")
        checked_parts += 1
        checked_rows += len(rows)
    if schema_version == SCHEMA_VERSION:
        identity_receipt = payload.get("source_registry_identity_receipt")
        if not isinstance(identity_receipt, Mapping):
            raise ValueError("source registry identity receipt is missing")
        source_root = (
            Path(str(payload.get("source_export_root") or "")).expanduser().resolve()
        )
        source_manifest_path = export_manifest_path(source_root, source_run_id)
        source_manifest_bytes = source_manifest_path.read_bytes()
        if hashlib.sha256(source_manifest_bytes).hexdigest() != payload.get(
            "source_export_manifest_sha256"
        ):
            raise ValueError(
                "source export manifest digest differs from derived binding"
            )
        source_manifest = json.loads(source_manifest_bytes)
        source_receipt = None
        if isinstance(source_manifest, Mapping):
            source_receipt = source_manifest.get("registry_identity_receipt")
            if source_receipt is None:
                source_receipt = source_manifest.get("registry_identity")
        if source_receipt != identity_receipt:
            raise ValueError(
                "source registry identity receipt differs from the digest-bound "
                "source manifest"
            )
        declared: set[tuple[str, str | None, str]] = set()
        behavior_identities: set[tuple[str, str | None, str]] = set()
        source_binding_by_recording: dict[str, tuple[str | None, str]] = {}
        for source_table in (
            CHASER_EPOCH_BEHAVIOR_TABLE,
            CHASER_DISTANCE_SUMMARY_TABLE,
            CHASER_EGOCENTRIC_SUMMARY_TABLE,
            CHASER_SPEED_DISTANCE_TABLE,
        ):
            for part in manifest_selected_part_files_from_payload(
                source_root,
                source_manifest,
                source_table,
            ):
                source_rows = (
                    pq.ParquetFile(part)
                    .read(
                        columns=["recording_id", "acquisition_batch_id", "subject_id"]
                    )
                    .to_pylist()
                )
                for row in source_rows:
                    key = (
                        row["recording_id"],
                        row["acquisition_batch_id"],
                        row["subject_id"],
                    )
                    if any(
                        type(value) is not str or not value.strip()
                        for value in (key[0], key[2])
                    ) or (
                        key[1] is not None
                        and (type(key[1]) is not str or not key[1].strip())
                    ):
                        raise ValueError(
                            "source export contains an invalid identity tuple"
                        )
                    previous = source_binding_by_recording.setdefault(
                        key[0],
                        (key[1], key[2]),
                    )
                    if previous != (key[1], key[2]):
                        raise ValueError(
                            "source export assigns conflicting batch/subject "
                            f"identities to recording {key[0]!r}"
                        )
                    declared.add(key)
                    if source_table == CHASER_EPOCH_BEHAVIOR_TABLE:
                        behavior_identities.add(key)
        output_identities: set[tuple[str, str | None, str]] | None = None
        for table_name in TRAINING_RESPONSE_TABLES:
            parts = derived_manifest_selected_parts(
                root,
                payload,
                table_name,
                table_names=TRAINING_RESPONSE_TABLES,
            )
            rows = (
                pq.ParquetFile(parts[0])
                .read(columns=["recording_id", "acquisition_batch_id", "subject_id"])
                .to_pylist()
            )
            observed = {
                (row["recording_id"], row["acquisition_batch_id"], row["subject_id"])
                for row in rows
            }
            if not observed <= declared:
                raise ValueError(
                    f"{table_name}: row identity is absent from the digest-bound "
                    "source export"
                )
            if observed != behavior_identities:
                raise ValueError(
                    f"{table_name}: row identities differ from the source behavior cohort"
                )
            if output_identities is None:
                output_identities = observed
            elif observed != output_identities:
                raise ValueError("training-response output identity sets differ")
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


def validate_training_response_v2_compatibility_run(
    output_root: Path,
    analysis_run_id: str,
) -> dict[str, Any]:
    """Validate a frozen v2 run only through the explicit compatibility path."""

    root = Path(output_root).expanduser().resolve()
    try:
        run_id = _component(analysis_run_id, label="analysis run ID")
        manifest_path = derived_manifest_path(root, run_id)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("training-response manifest must contain an object")
        return _validate_payload(
            root,
            run_id,
            payload,
            schema_version=LEGACY_EXACT_SCHEMA_VERSION,
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise TrainingResponseValidationError(
            f"legacy training-response v2 run {analysis_run_id!r} failed "
            f"compatibility validation: {exc}"
        ) from exc


__all__ = [
    "TrainingResponseValidationError",
    "validate_training_response_run",
    "validate_training_response_v2_compatibility_run",
]
