"""Integrity validation for published Palette analytics exports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .arrow_contracts import (
    validate_arrow_contract_envelope,
    validate_arrow_schema,
    validate_declared_v2_arrow_contract_envelope,
    validate_declared_v2_arrow_schema,
)
from .chaser_authority import validate_chaser_export_authority_receipt
from .capabilities import resolve_capabilities
from .contracts import (
    EXPORT_SCHEMA_ID,
    EXPORT_SCHEMA_VERSION,
    EXPORT_SCHEMA_VERSION_V2,
    REGISTRY_IDENTITY_TABLES,
    TABLE_CONTRACTS,
    validate_table_columns,
)
from .registry_identity import (
    registry_identity_sources_by_path,
    validate_registry_identity_receipt,
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
        raise ExportValidationError(
            f"Cannot read export manifest {path}: {exc}"
        ) from exc
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


def _validate_declared_v2_table_contract(
    table: str,
    value: object,
) -> dict[str, Any]:
    """Validate one historical self-contained table declaration."""

    fields = {
        "contract_id",
        "contract_version",
        "table_name",
        "grain",
        "primary_key",
        "required_columns",
        "units",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{table}: historical table contract field set is invalid")
    if (
        value.get("contract_id") != f"palette.analytics.table.{table}"
        or value.get("table_name") != table
        or type(value.get("contract_version")) is not int
        or int(value["contract_version"]) < 1
        or not isinstance(value.get("grain"), str)
        or not value["grain"]
        or not isinstance(value.get("units"), Mapping)
    ):
        raise ValueError(f"{table}: historical table contract identity is invalid")
    primary_key = value.get("primary_key")
    required = value.get("required_columns")
    if (
        not isinstance(primary_key, list)
        or not isinstance(required, list)
        or any(not isinstance(item, str) or not item for item in primary_key)
        or any(not isinstance(item, str) or not item for item in required)
        or len(primary_key) != len(set(primary_key))
        or len(required) != len(set(required))
        or not set(primary_key).issubset(required)
    ):
        raise ValueError(f"{table}: historical table contract columns are invalid")
    return dict(value)


def _part_identity_values(parquet_file: Any) -> tuple[set[tuple[Any, ...]], int]:
    names = ("zarr_path", "recording_id", "acquisition_batch_id", "subject_id")
    table = parquet_file.read(columns=list(names))
    values = set(zip(*(table.column(name).to_pylist() for name in names), strict=True))
    return values, int(table.num_rows)


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
    raw_schema_version = payload.get("schema_version")
    if raw_schema_version not in (EXPORT_SCHEMA_VERSION_V2, EXPORT_SCHEMA_VERSION):
        errors.append(f"unsupported schema_version {payload.get('schema_version')!r}")
    export_schema_version = (
        raw_schema_version
        if raw_schema_version in (EXPORT_SCHEMA_VERSION_V2, EXPORT_SCHEMA_VERSION)
        else EXPORT_SCHEMA_VERSION
    )
    current_schema = export_schema_version == EXPORT_SCHEMA_VERSION
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

    registry_sources: Mapping[str, Mapping[str, Any]] = {}
    if current_schema and set(tables) & REGISTRY_IDENTITY_TABLES:
        identity = payload.get("registry_identity")
        source_zarrs = payload.get("source_zarrs")
        if not isinstance(source_zarrs, list):
            errors.append("registry identity source_zarrs declaration is invalid")
        else:
            try:
                validated_identity = validate_registry_identity_receipt(
                    identity,
                    expected_zarr_paths=source_zarrs,
                )
                registry_sources = registry_identity_sources_by_path(validated_identity)
            except ValueError as exc:
                errors.append(str(exc))

    if current_schema and payload.get("chaser_export_authority") is not None:
        source_zarrs = payload.get("source_zarrs")
        if not isinstance(source_zarrs, list):
            errors.append("chaser authority source_zarrs declaration is invalid")
        else:
            try:
                validate_chaser_export_authority_receipt(
                    payload.get("chaser_export_authority"),
                    expected_zarr_paths=source_zarrs,
                )
            except ValueError as exc:
                errors.append(str(exc))

    arrow_contracts_valid = False
    arrow_contracts = payload.get("arrow_schema_contracts")
    if arrow_contracts is None:
        if not allow_legacy_layout:
            errors.append(
                "missing exact Arrow-contract envelope; use the explicit legacy "
                "compatibility option only for historical exports"
            )
    else:
        try:
            if current_schema:
                validate_arrow_contract_envelope(arrow_contracts, tables)
            else:
                validate_declared_v2_arrow_contract_envelope(
                    arrow_contracts,
                    tables,
                )
        except ValueError as exc:
            errors.append(str(exc))
        else:
            arrow_contracts_valid = True

    declared_contracts = payload.get("table_contracts")
    if not isinstance(declared_contracts, Mapping):
        errors.append("manifest table_contracts is missing or is not an object")
        declared_contracts = {}
    for table in tables:
        if table not in TABLE_CONTRACTS:
            continue
        if current_schema:
            if declared_contracts.get(table) != TABLE_CONTRACTS[table].to_dict():
                errors.append(
                    f"{table}: manifest table contract does not match installed V3"
                )
        else:
            try:
                _validate_declared_v2_table_contract(
                    table,
                    declared_contracts.get(table),
                )
            except ValueError as exc:
                errors.append(str(exc))

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
                root / str(publication.get("generation_path")) / "tables" / table
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
        inventory_by_path = (
            {
                str(entry.get("path")): entry
                for entry in raw_inventory
                if isinstance(entry, Mapping)
            }
            if isinstance(raw_inventory, list)
            else {}
        )
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
            if (
                metadata.get(b"palette.export_schema_id", b"").decode("utf-8")
                != EXPORT_SCHEMA_ID
            ):
                errors.append(f"{table}/{part_name}: invalid footer schema ID")
            if metadata.get(b"palette.export_schema_version", b"").decode(
                "utf-8"
            ) != str(export_schema_version):
                errors.append(f"{table}/{part_name}: invalid footer schema version")
            expected_table_contract = (
                TABLE_CONTRACTS[table].to_dict()
                if current_schema
                else declared_contracts.get(table)
            )
            if footer_contract != expected_table_contract:
                errors.append(f"{table}/{part_name}: invalid footer table contract")
            if arrow_contracts_valid:
                try:
                    if current_schema:
                        validate_arrow_schema(table, schema)
                    else:
                        assert isinstance(arrow_contracts, Mapping)
                        validate_declared_v2_arrow_schema(
                            table,
                            schema,
                            arrow_contracts,
                        )
                except ValueError as exc:
                    errors.append(f"{table}/{part_name}: {exc}")
            columns = tuple(field.name for field in schema)
            if current_schema:
                missing = validate_table_columns(table, columns)
            else:
                declared = declared_contracts.get(table)
                required_columns = (
                    declared.get("required_columns", [])
                    if isinstance(declared, Mapping)
                    else []
                )
                missing = tuple(sorted(set(required_columns) - set(columns)))
            if missing:
                errors.append(
                    f"{table}/{part_name}: missing required columns {list(missing)}"
                )
            legacy_columns = sorted(
                column for column in columns if "benign" in column.lower()
            )
            if legacy_columns:
                errors.append(f"{table}/{part_name}: legacy columns {legacy_columns}")
            if reference_columns is None:
                reference_columns = columns
                columns_by_table[table] = columns
            elif columns != reference_columns:
                errors.append(
                    f"{table}/{part_name}: schema differs from the first part"
                )
            part_rows = int(parquet_file.metadata.num_rows)
            if current_schema and table in REGISTRY_IDENTITY_TABLES and part_rows:
                try:
                    identity_values, identity_rows = _part_identity_values(parquet_file)
                except Exception as exc:
                    errors.append(
                        f"{table}/{part_name}: cannot read persisted registry identity: {exc}"
                    )
                else:
                    if identity_rows != part_rows or not identity_values:
                        errors.append(
                            f"{table}/{part_name}: part identity columns are incomplete"
                        )
                    else:
                        mismatched = []
                        for (
                            source_path,
                            recording_id,
                            acquisition_batch_id,
                            subject_id,
                        ) in sorted(
                            identity_values,
                            key=lambda item: tuple(map(str, item)),
                        ):
                            binding = registry_sources.get(str(source_path))
                            if binding is None or (
                                recording_id,
                                acquisition_batch_id,
                                subject_id,
                            ) != (
                                binding["recording_id"],
                                binding["acquisition_batch_id"],
                                binding["subject_id"],
                            ):
                                mismatched.append(str(source_path))
                        if mismatched:
                            errors.append(
                                f"{table}/{part_name}: persisted registry identity "
                                "differs from its receipt binding for "
                                f"{mismatched!r}"
                            )
            if strict_publication and inventory_entry.get("row_count") != part_rows:
                errors.append(f"{table}/{part_name}: row-count mismatch")
            table_rows += part_rows
            checked_rows += part_rows
            checked_parts += 1
        try:
            expected_rows = int(row_counts_by_table.get(table, 0))
        except (TypeError, ValueError):
            errors.append(
                f"{table}: invalid manifest row count {row_counts_by_table.get(table)!r}"
            )
        else:
            if table_rows != expected_rows:
                errors.append(
                    f"{table}: Parquet row count {table_rows} does not match manifest {expected_rows}"
                )

    if current_schema:
        resolved = resolve_capabilities(columns_by_table)
        resolved_capabilities = [
            status.capability_id for status in resolved if status.available
        ]
        if payload.get("capabilities") != resolved_capabilities:
            errors.append(
                "manifest capabilities do not match capabilities resolved from Parquet schemas"
            )
    else:
        raw_capabilities = payload.get("capabilities", [])
        resolved_capabilities = (
            [str(value) for value in raw_capabilities]
            if isinstance(raw_capabilities, list)
            else []
        )

    if current_schema and "eye_trace_samples" in tables:
        try:
            from .eye_trace_samples import validate_eye_trace_export_payload

            validate_eye_trace_export_payload(root, payload)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            errors.append(f"eye_trace_samples: {exc}")

    if current_schema and "kinematics_samples" in tables:
        try:
            from .kinematics_samples import (
                validate_kinematics_samples_export_payload,
            )

            validate_kinematics_samples_export_payload(root, payload)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            errors.append(f"kinematics_samples: {exc}")

    if current_schema and "activity_spatial_time_bins" in tables:
        try:
            from .activity_spatial_time_bins import (
                validate_activity_spatial_time_bins_export_payload,
            )

            validate_activity_spatial_time_bins_export_payload(root, payload)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            errors.append(f"activity_spatial_time_bins: {exc}")

    if current_schema and "tail_trace_samples" in tables:
        try:
            from .tail_trace_samples import validate_tail_trace_export_payload

            validate_tail_trace_export_payload(root, payload)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            errors.append(f"tail_trace_samples: {exc}")

    if errors:
        raise ExportValidationError(
            f"Analytics export {run_id!r} failed validation:\n- " + "\n- ".join(errors)
        )
    return {
        "export_root": str(root),
        "export_run_id": run_id,
        "manifest_path": str(manifest_path),
        "schema_id": EXPORT_SCHEMA_ID,
        "schema_version": export_schema_version,
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
