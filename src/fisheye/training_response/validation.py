"""Validation for published whole-training response analytics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .contracts import SCHEMA_ID, SCHEMA_VERSION, contract_fields


class TrainingResponseValidationError(ValueError):
    pass


def _safe_component(value: object, *, label: str) -> str:
    text = str(value or "").strip()
    if not text or Path(text).name != text or text in {".", ".."}:
        raise TrainingResponseValidationError(f"invalid {label}: {value!r}")
    return text


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def validate_training_response_run(
    output_root: Path, analysis_run_id: str
) -> dict[str, Any]:
    import pyarrow.parquet as pq

    root = Path(output_root).expanduser().resolve()
    run_id = _safe_component(analysis_run_id, label="analysis run ID")
    manifest_path = root / "v1" / "manifests" / f"analysis_run_id={run_id}.json"
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise TrainingResponseValidationError(
            f"cannot read training-response manifest {manifest_path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise TrainingResponseValidationError("manifest must contain an object")
    errors: list[str] = []
    if payload.get("schema_id") != SCHEMA_ID:
        errors.append(f"unsupported schema_id {payload.get('schema_id')!r}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"unsupported schema_version {payload.get('schema_version')!r}")
    if payload.get("analysis_run_id") != run_id:
        errors.append("manifest analysis_run_id does not match requested run")
    if payload.get("source_export_mutated") is not False:
        errors.append("manifest does not declare source_export_mutated=false")
    expected_source = str(payload.get("source_export_run_id") or "").strip()
    tables = payload.get("output_tables")
    if not isinstance(tables, list) or not tables:
        errors.append("manifest output_tables is missing or empty")
        tables = []
    row_counts = _mapping(payload.get("row_counts_by_table"))
    parts_by_table = _mapping(payload.get("part_files_by_table"))
    checked_parts = 0
    checked_rows = 0
    for raw_table in tables:
        try:
            table_name = _safe_component(raw_table, label="table name")
            required = {*contract_fields(table_name), "analysis_run_id"}
        except (ValueError, KeyError) as exc:
            errors.append(str(exc))
            continue
        declared_parts = parts_by_table.get(table_name)
        if not isinstance(declared_parts, list):
            errors.append(f"{table_name}: part list is not an array")
            continue
        observed_rows = 0
        for raw_part in declared_parts:
            part_name = Path(str(raw_part)).name
            path = (
                root
                / "v1"
                / table_name
                / f"analysis_run_id={run_id}"
                / part_name
            ).resolve()
            try:
                path.relative_to(root)
            except ValueError:
                errors.append(f"{table_name}: part resolves outside output root")
                continue
            if not path.is_file():
                errors.append(f"{table_name}: missing part {path}")
                continue
            try:
                parquet_file = pq.ParquetFile(path)
                schema = parquet_file.schema_arrow
                metadata = schema.metadata or {}
            except Exception as exc:
                errors.append(f"{table_name}/{part_name}: unreadable parquet: {exc}")
                continue
            if metadata.get(b"palette.schema_id", b"").decode() != SCHEMA_ID:
                errors.append(f"{table_name}/{part_name}: invalid footer schema ID")
            if metadata.get(b"palette.schema_version", b"").decode() != str(
                SCHEMA_VERSION
            ):
                errors.append(
                    f"{table_name}/{part_name}: invalid footer schema version"
                )
            if metadata.get(b"palette.table_name", b"").decode() != table_name:
                errors.append(f"{table_name}/{part_name}: invalid footer table name")
            missing = sorted(required - set(schema.names))
            if missing:
                errors.append(f"{table_name}/{part_name}: missing fields {missing}")
            if "source_export_run_id" in schema.names:
                source_ids = {
                    str(value)
                    for value in parquet_file.read(
                        columns=["source_export_run_id"]
                    ).column(0).to_pylist()
                    if value is not None
                }
                if source_ids != {expected_source}:
                    errors.append(
                        f"{table_name}/{part_name}: source export IDs {source_ids!r} "
                        f"do not match {expected_source!r}"
                    )
            if "analysis_run_id" in schema.names:
                analysis_ids = {
                    str(value)
                    for value in parquet_file.read(
                        columns=["analysis_run_id"]
                    ).column(0).to_pylist()
                    if value is not None
                }
                if analysis_ids != {run_id}:
                    errors.append(
                        f"{table_name}/{part_name}: analysis run IDs "
                        f"{analysis_ids!r} do not match {run_id!r}"
                    )
            rows = int(parquet_file.metadata.num_rows)
            observed_rows += rows
            checked_rows += rows
            checked_parts += 1
        if observed_rows != int(row_counts.get(table_name, -1)):
            errors.append(
                f"{table_name}: observed {observed_rows} rows, manifest declares "
                f"{row_counts.get(table_name)!r}"
            )
    if errors:
        raise TrainingResponseValidationError(
            f"training-response run {run_id!r} failed validation:\n- "
            + "\n- ".join(errors)
        )
    return {
        "status": "valid",
        "output_root": str(root),
        "analysis_run_id": run_id,
        "manifest_path": str(manifest_path),
        "table_count": len(tables),
        "part_count": checked_parts,
        "row_count": checked_rows,
    }


__all__ = ["TrainingResponseValidationError", "validate_training_response_run"]
