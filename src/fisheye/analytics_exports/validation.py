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
from .publication import (
    export_manifest_path,
    has_exclusive_inventory,
    manifest_selected_part_files_from_payload,
    publication_inventory,
    sha256_file,
    safe_component,
    validate_publication_envelope,
)


class ExportValidationError(ValueError):
    """Raised when an analytics export does not satisfy its published contract."""


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


def validate_export_payload(
    export_root: Path,
    export_run_id: str,
    payload: Mapping[str, Any],
    *,
    allow_legacy_layout: bool = False,
) -> dict[str, Any]:
    """Validate one already-loaded manifest snapshot and all bytes it selects."""

    import pyarrow.parquet as pq

    root = Path(export_root).expanduser().resolve()
    try:
        run_id = safe_component(export_run_id, label="export run ID")
        manifest_path = export_manifest_path(root, run_id)
    except ValueError as exc:
        raise ExportValidationError(str(exc)) from exc
    errors: list[str] = []

    if payload.get("export_run_id") != run_id:
        errors.append(
            f"manifest export_run_id {payload.get('export_run_id')!r} does not match {run_id!r}"
        )
    if payload.get("schema_id") != EXPORT_SCHEMA_ID:
        errors.append(f"unsupported schema_id {payload.get('schema_id')!r}")
    if payload.get("schema_version") != EXPORT_SCHEMA_VERSION:
        errors.append(f"unsupported schema_version {payload.get('schema_version')!r}")
    strict_publication = has_exclusive_inventory(payload)
    exact_publication = False
    publication_declared = "publication" in payload
    if not publication_declared and not allow_legacy_layout:
        errors.append(
            "missing manifest-exclusive publication-v1 inventory; use the explicit "
            "legacy compatibility option only for historical exports"
        )
    if publication_declared:
        try:
            validate_publication_envelope(payload)
        except ValueError as exc:
            errors.append(str(exc))
        else:
            strict_publication = True
            exact_publication = True

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
    inventory_by_table: Mapping[str, object] = {}
    if strict_publication:
        try:
            inventory_by_table = publication_inventory(payload)
        except ValueError as exc:
            errors.append(str(exc))
        if set(inventory_by_table) != set(tables):
            errors.append(
                "publication inventory table set does not match declared tables"
            )
    if exact_publication:
        publication = payload["publication"]
        assert isinstance(publication, Mapping)
        generation_relative = Path(str(publication["generation_path"]))
        generation_root = root / generation_relative
        current = root
        generation_has_symlink = False
        for component in generation_relative.parts:
            current /= component
            if current.is_symlink():
                generation_has_symlink = True
                break
        if generation_has_symlink:
            errors.append("generation path contains a symbolic-link alias")
        try:
            generation_root.resolve().relative_to(root)
        except ValueError:
            errors.append("generation path resolves outside the export root")
        if not generation_root.is_dir():
            errors.append("generation directory is missing")
        expected_generation_files = {
            str(entry["path"])
            for entries in inventory_by_table.values()
            if isinstance(entries, list)
            for entry in entries
            if isinstance(entry, Mapping)
        }
        actual_generation_files: set[str] = set()
        if generation_root.is_dir() and not generation_has_symlink:
            for path in generation_root.rglob("*"):
                if path.is_symlink():
                    errors.append("generation contains a symbolic-link alias")
                    continue
                if path.is_file():
                    actual_generation_files.add(path.relative_to(root).as_posix())
            if actual_generation_files != expected_generation_files:
                errors.append("generation contains files outside its exact inventory")
    for raw_table in tables:
        try:
            table = safe_component(raw_table, label="table name")
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if table not in TABLE_CONTRACTS:
            continue
        try:
            resolved_parts = manifest_selected_part_files_from_payload(
                root,
                payload,
                table,
                allow_legacy_layout=allow_legacy_layout,
            )
        except ValueError as exc:
            errors.append(str(exc))
            continue
        raw_parts = parts_by_table.get(table, [])
        if not isinstance(raw_parts, list):
            errors.append(f"{table}: part list is not an array")
            continue
        if not strict_publication and allow_legacy_layout and not raw_parts:
            raw_parts = [str(path) for path in resolved_parts]
        if len(set(map(str, raw_parts))) != len(raw_parts):
            errors.append(f"{table}: manifest contains duplicate part paths")
            continue
        if strict_publication:
            publication = payload["publication"]
            assert isinstance(publication, Mapping)
            table_root = (
                root
                / str(publication.get("generation_path"))
                / "tables"
                / table
            )
            actual_parts = {
                path.relative_to(root).as_posix()
                for path in table_root.glob("*.parquet")
                if path.is_file() and not path.is_symlink()
            }
            selected_parts = {
                path.relative_to(root).as_posix() for path in resolved_parts
            }
            if actual_parts != selected_parts:
                errors.append(f"{table}: generation contains unlisted files")
        raw_inventory = inventory_by_table.get(table, [])
        inventory_by_path = {
            str(entry.get("path")): entry
            for entry in raw_inventory
            if isinstance(entry, Mapping)
        } if isinstance(raw_inventory, list) else {}
        if strict_publication and len(inventory_by_path) != len(resolved_parts):
            errors.append(f"{table}: publication inventory is incomplete or duplicated")
        table_rows = 0
        reference_columns: tuple[str, ...] | None = None
        for raw_part, resolved_part in zip(raw_parts, resolved_parts, strict=True):
            part_name = resolved_part.name
            if not resolved_part.is_file():
                errors.append(f"{table}: missing part {resolved_part}")
                continue
            inventory_entry = inventory_by_path.get(str(raw_part))
            if strict_publication:
                if not isinstance(inventory_entry, Mapping):
                    errors.append(f"{table}/{part_name}: missing inventory entry")
                    continue
                if inventory_entry.get("sha256") != sha256_file(resolved_part):
                    errors.append(f"{table}/{part_name}: digest mismatch")
                    continue
                if inventory_entry.get("size_bytes") != resolved_part.stat().st_size:
                    errors.append(f"{table}/{part_name}: size mismatch")
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
            if strict_publication and inventory_entry.get("row_count") != part_rows:
                errors.append(f"{table}/{part_name}: row-count mismatch")
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


def validate_export_run(
    export_root: Path,
    export_run_id: str,
    *,
    allow_legacy_layout: bool = False,
) -> dict[str, Any]:
    """Load and validate every part of one immutable export."""

    root = Path(export_root).expanduser().resolve()
    try:
        run_id = safe_component(export_run_id, label="export run ID")
        manifest_path = export_manifest_path(root, run_id)
    except ValueError as exc:
        raise ExportValidationError(str(exc)) from exc
    return validate_export_payload(
        root,
        run_id,
        _load_manifest(manifest_path),
        allow_legacy_layout=allow_legacy_layout,
    )


def validate_export_runs(
    export_root: Path,
    export_run_ids: Sequence[str],
) -> list[dict[str, Any]]:
    return [validate_export_run(export_root, run_id) for run_id in export_run_ids]


__all__ = [
    "ExportValidationError",
    "validate_export_payload",
    "validate_export_run",
    "validate_export_runs",
]
