"""Explicit read-only validation for historical inferred group-statistics exports."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from fisheye.analytics_exports.arrow_contracts import (
    ARROW_CONTRACT_ENVELOPE_SCHEMA_ID,
    ARROW_CONTRACT_ENVELOPE_SCHEMA_VERSION,
)
from fisheye.analytics_exports.contracts import (
    DESCRIPTIVE_TABLE,
    EXPORT_SCHEMA_ID,
    EXPORT_SCHEMA_VERSION,
    STATISTICS_TABLE,
)
from fisheye.analytics_exports.publication import (
    manifest_selected_part_files_from_payload,
    sha256_file,
    validate_publication_envelope,
)


LEGACY_GROUP_STATISTICS_TABLES = (STATISTICS_TABLE, DESCRIPTIVE_TABLE)

_LEGACY_CONTRACTS: dict[str, dict[str, object]] = {
    STATISTICS_TABLE: {
        "contract_id": f"palette.analytics.table.{STATISTICS_TABLE}",
        "contract_version": 1,
        "table_name": STATISTICS_TABLE,
        "grain": "statistical_result",
        "primary_key": ["stat_result_id"],
        "required_columns": [
            "export_schema_version",
            "table_name",
            "stat_result_id",
            "source_export_run_id",
            "source_table",
            "metric_name",
            "status",
        ],
        "units": {},
    },
    DESCRIPTIVE_TABLE: {
        "contract_id": f"palette.analytics.table.{DESCRIPTIVE_TABLE}",
        "contract_version": 1,
        "table_name": DESCRIPTIVE_TABLE,
        "grain": "descriptive_result",
        "primary_key": ["descriptive_result_id"],
        "required_columns": [
            "export_schema_version",
            "table_name",
            "descriptive_result_id",
            "source_export_run_id",
            "source_table",
            "metric_name",
            "unit_count",
        ],
        "units": {},
    },
}


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def legacy_group_statistics_arrow_envelope(
    table_names: Sequence[str],
) -> dict[str, object]:
    names = tuple(table_names)
    if (
        len(set(names)) != len(names)
        or STATISTICS_TABLE not in names
        or not set(names) <= set(LEGACY_GROUP_STATISTICS_TABLES)
    ):
        raise ValueError("Legacy group-statistics table declaration is invalid")
    payload: dict[str, object] = {
        "schema_id": ARROW_CONTRACT_ENVELOPE_SCHEMA_ID,
        "schema_version": ARROW_CONTRACT_ENVELOPE_SCHEMA_VERSION,
        "exact_tables": {},
        "inferred_v2_compatibility_tables": sorted(names),
    }
    return {**payload, "payload_sha256": _canonical_sha256(payload)}


def legacy_group_statistics_contract_snapshot(
    table_names: Sequence[str],
) -> dict[str, dict[str, object]]:
    names = tuple(table_names)
    if (
        len(set(names)) != len(names)
        or STATISTICS_TABLE not in names
        or not set(names) <= set(LEGACY_GROUP_STATISTICS_TABLES)
    ):
        raise ValueError("Legacy group-statistics table declaration is invalid")
    return {name: dict(_LEGACY_CONTRACTS[name]) for name in names}


def validate_legacy_group_statistics_payload(
    export_root: Path,
    stats_run_id: str,
    payload: Mapping[str, Any],
) -> None:
    """Validate only the historical inferred-v2 group-statistics surface."""

    import pyarrow.parquet as pq

    if payload.get("export_run_id") != stats_run_id:
        raise ValueError("Legacy statistics run identity is invalid")
    if (
        payload.get("schema_id") != EXPORT_SCHEMA_ID
        or payload.get("schema_version") != EXPORT_SCHEMA_VERSION
    ):
        raise ValueError("Legacy statistics export schema is invalid")
    output_tables = payload.get("output_tables")
    if not isinstance(output_tables, list):
        raise ValueError("Legacy statistics output tables are invalid")
    tables = tuple(str(value) for value in output_tables)
    expected_contracts = legacy_group_statistics_contract_snapshot(tables)
    if payload.get("table_contracts") != expected_contracts:
        raise ValueError("Legacy statistics table contracts are invalid")
    if payload.get("arrow_schema_contracts") != (
        legacy_group_statistics_arrow_envelope(tables)
    ):
        raise ValueError("Legacy statistics Arrow envelope is invalid")

    inventory = validate_publication_envelope(payload)
    row_counts = payload.get("row_counts_by_table")
    part_files = payload.get("part_files_by_table")
    if not isinstance(row_counts, Mapping) or not isinstance(part_files, Mapping):
        raise ValueError("Legacy statistics logical inventory is invalid")
    if set(inventory) != set(tables):
        raise ValueError("Legacy statistics physical inventory is invalid")

    for table_name in tables:
        entries = inventory.get(table_name)
        if not isinstance(entries, list):
            raise ValueError(f"{table_name}: legacy inventory is invalid")
        files = manifest_selected_part_files_from_payload(
            Path(export_root).expanduser().resolve(),
            payload,
            table_name,
        )
        declared_paths = part_files.get(table_name)
        if not isinstance(declared_paths, list) or len(files) != len(entries):
            raise ValueError(f"{table_name}: legacy part declaration is invalid")
        table_rows = 0
        reference_fields: tuple[tuple[str, str], ...] | None = None
        required = set(expected_contracts[table_name]["required_columns"])
        for file_path, entry in zip(files, entries, strict=True):
            if not file_path.is_file():
                raise ValueError(f"{table_name}: legacy part is missing")
            if (
                entry.get("path") not in declared_paths
                or entry.get("sha256") != sha256_file(file_path)
                or entry.get("size_bytes") != file_path.stat().st_size
            ):
                raise ValueError(f"{table_name}: legacy part identity is invalid")
            parquet_file = pq.ParquetFile(file_path)
            if entry.get("row_count") != parquet_file.metadata.num_rows:
                raise ValueError(f"{table_name}: legacy part row count is invalid")
            schema = parquet_file.schema_arrow
            metadata = schema.metadata or {}
            if (
                metadata.get(b"palette.export_schema_id", b"").decode()
                != EXPORT_SCHEMA_ID
                or metadata.get(b"palette.export_schema_version", b"").decode()
                != str(EXPORT_SCHEMA_VERSION)
                or metadata.get(b"palette.arrow_schema_mode")
                != b"inferred_v2_compatibility"
            ):
                raise ValueError(f"{table_name}: legacy footer mode is invalid")
            try:
                footer_contract = json.loads(
                    metadata.get(b"palette.table_contract", b"null").decode()
                )
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"{table_name}: legacy footer contract is invalid"
                ) from exc
            if footer_contract != expected_contracts[table_name]:
                raise ValueError(f"{table_name}: legacy footer contract is invalid")
            names = tuple(field.name for field in schema)
            if not required <= set(names) or any(
                "benign" in name.lower() for name in names
            ):
                raise ValueError(f"{table_name}: legacy fields are invalid")
            fields = tuple((field.name, str(field.type)) for field in schema)
            if reference_fields is None:
                reference_fields = fields
            elif reference_fields != fields:
                raise ValueError(f"{table_name}: legacy part schemas differ")
            table_rows += int(parquet_file.metadata.num_rows)
        if table_rows != row_counts.get(table_name):
            raise ValueError(f"{table_name}: legacy manifest row count is invalid")


__all__ = [
    "LEGACY_GROUP_STATISTICS_TABLES",
    "legacy_group_statistics_arrow_envelope",
    "legacy_group_statistics_contract_snapshot",
    "validate_legacy_group_statistics_payload",
]
