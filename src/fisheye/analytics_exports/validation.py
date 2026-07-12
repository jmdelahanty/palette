"""Integrity validation for published Palette analytics exports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .capabilities import resolve_capabilities
from .contracts import (
    EXPORT_SCHEMA_ID,
    EXPORT_SCHEMA_VERSION,
    TABLE_CONTRACTS,
    validate_table_columns,
)


class ExportValidationError(ValueError):
    """Raised when an analytics export does not satisfy its published contract."""


def _safe_component(value: str, *, label: str) -> str:
    text = str(value).strip()
    if not text or Path(text).name != text or text in {".", ".."}:
        raise ExportValidationError(f"Invalid {label}: {value!r}")
    return text


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ExportValidationError(f"Cannot read export manifest {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ExportValidationError(f"Export manifest must contain an object: {path}")
    return payload


def _declared_tables(payload: Mapping[str, Any]) -> tuple[str, ...]:
    tables: set[str] = set()
    for field in ("tables_requested", "output_tables"):
        values = payload.get(field)
        if isinstance(values, list):
            tables.update(str(value) for value in values)
    for field in ("row_counts_by_table", "part_files_by_table"):
        values = payload.get(field)
        if isinstance(values, Mapping):
            tables.update(str(value) for value in values)
    return tuple(sorted(tables))


def validate_export_run(export_root: Path, export_run_id: str) -> dict[str, Any]:
    """Validate every part of one immutable base or statistics export."""

    import pyarrow.parquet as pq

    root = Path(export_root).expanduser().resolve()
    run_id = _safe_component(export_run_id, label="export run ID")
    manifest_path = root / "v1" / "manifests" / f"export_run_id={run_id}.json"
    payload = _load_manifest(manifest_path)
    errors: list[str] = []

    if payload.get("export_run_id") != run_id:
        errors.append(
            f"manifest export_run_id {payload.get('export_run_id')!r} does not match {run_id!r}"
        )
    if payload.get("schema_id") != EXPORT_SCHEMA_ID:
        errors.append(f"unsupported schema_id {payload.get('schema_id')!r}")
    if payload.get("schema_version") != EXPORT_SCHEMA_VERSION:
        errors.append(f"unsupported schema_version {payload.get('schema_version')!r}")

    tables = _declared_tables(payload)
    unknown_tables = sorted(set(tables) - set(TABLE_CONTRACTS))
    if unknown_tables:
        errors.append(f"unknown table contracts: {unknown_tables}")

    declared_contracts = payload.get("table_contracts")
    if not isinstance(declared_contracts, Mapping):
        errors.append("manifest table_contracts is missing or is not an object")
        declared_contracts = {}
    for table in tables:
        if table in TABLE_CONTRACTS and declared_contracts.get(table) != TABLE_CONTRACTS[
            table
        ].to_dict():
            errors.append(f"{table}: manifest table contract does not match installed V2")

    parts_by_table = payload.get("part_files_by_table")
    row_counts_by_table = payload.get("row_counts_by_table")
    if not isinstance(parts_by_table, Mapping):
        errors.append("manifest part_files_by_table is missing or is not an object")
        parts_by_table = {}
    if not isinstance(row_counts_by_table, Mapping):
        errors.append("manifest row_counts_by_table is missing or is not an object")
        row_counts_by_table = {}

    checked_parts = 0
    checked_rows = 0
    columns_by_table: dict[str, tuple[str, ...]] = {}
    for raw_table in tables:
        try:
            table = _safe_component(raw_table, label="table name")
        except ExportValidationError as exc:
            errors.append(str(exc))
            continue
        if table not in TABLE_CONTRACTS:
            continue
        raw_parts = parts_by_table.get(table, [])
        if not isinstance(raw_parts, list):
            errors.append(f"{table}: part list is not an array")
            continue
        table_rows = 0
        reference_columns: tuple[str, ...] | None = None
        for raw_part in raw_parts:
            part_name = Path(str(raw_part)).name
            if part_name in {"", ".", ".."}:
                errors.append(f"{table}: invalid part path {raw_part!r}")
                continue
            part_path = root / "v1" / table / f"export_run_id={run_id}" / part_name
            resolved_part = part_path.resolve()
            try:
                resolved_part.relative_to(root)
            except ValueError:
                errors.append(f"{table}: part resolves outside export root: {raw_part}")
                continue
            if not resolved_part.is_file():
                errors.append(f"{table}: missing part {part_path}")
                continue
            try:
                parquet_file = pq.ParquetFile(resolved_part)
                schema = parquet_file.schema_arrow
                metadata = schema.metadata or {}
                footer_contract = json.loads(
                    metadata.get(b"palette.table_contract", b"null").decode("utf-8")
                )
            except Exception as exc:  # PyArrow exposes several format-specific errors.
                errors.append(f"{table}: cannot read {part_name}: {exc}")
                continue
            if metadata.get(b"palette.export_schema_id", b"").decode("utf-8") != EXPORT_SCHEMA_ID:
                errors.append(f"{table}/{part_name}: invalid footer schema ID")
            if metadata.get(b"palette.export_schema_version", b"").decode("utf-8") != str(
                EXPORT_SCHEMA_VERSION
            ):
                errors.append(f"{table}/{part_name}: invalid footer schema version")
            if footer_contract != TABLE_CONTRACTS[table].to_dict():
                errors.append(f"{table}/{part_name}: invalid footer table contract")
            columns = tuple(field.name for field in schema)
            missing = validate_table_columns(table, columns)
            if missing:
                errors.append(f"{table}/{part_name}: missing required columns {list(missing)}")
            legacy_columns = sorted(column for column in columns if "benign" in column.lower())
            if legacy_columns:
                errors.append(f"{table}/{part_name}: legacy columns {legacy_columns}")
            if reference_columns is None:
                reference_columns = columns
                columns_by_table[table] = columns
            elif columns != reference_columns:
                errors.append(f"{table}/{part_name}: schema differs from the first part")
            part_rows = int(parquet_file.metadata.num_rows)
            table_rows += part_rows
            checked_rows += part_rows
            checked_parts += 1
        try:
            expected_rows = int(row_counts_by_table.get(table, 0))
        except (TypeError, ValueError):
            errors.append(f"{table}: invalid manifest row count {row_counts_by_table.get(table)!r}")
        else:
            if table_rows != expected_rows:
                errors.append(
                    f"{table}: Parquet row count {table_rows} does not match manifest {expected_rows}"
                )

    resolved = resolve_capabilities(columns_by_table)
    resolved_capabilities = [
        status.capability_id for status in resolved if status.available
    ]
    if payload.get("capabilities") != resolved_capabilities:
        errors.append(
            "manifest capabilities do not match capabilities resolved from Parquet schemas"
        )

    if errors:
        raise ExportValidationError(
            f"Analytics export {run_id!r} failed validation:\n- " + "\n- ".join(errors)
        )
    return {
        "export_root": str(root),
        "export_run_id": run_id,
        "manifest_path": str(manifest_path),
        "schema_id": EXPORT_SCHEMA_ID,
        "schema_version": EXPORT_SCHEMA_VERSION,
        "table_count": len(tables),
        "part_count": checked_parts,
        "row_count": checked_rows,
        "capabilities": resolved_capabilities,
        "status": "valid",
    }


def validate_export_runs(
    export_root: Path,
    export_run_ids: Sequence[str],
) -> list[dict[str, Any]]:
    return [validate_export_run(export_root, run_id) for run_id in export_run_ids]


__all__ = [
    "ExportValidationError",
    "validate_export_run",
    "validate_export_runs",
]
