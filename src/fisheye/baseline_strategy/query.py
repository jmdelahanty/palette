"""Lazy Parquet queries for published baseline-strategy analytics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import polars as pl

from fisheye.analytics_exports.derived_publication import (
    derived_manifest_path,
    derived_manifest_selected_parts,
    validate_derived_manifest_envelope,
)

from .contracts import (
    ARROW_CONTRACT_ENVELOPE_SCHEMA_ID,
    ARROW_CONTRACT_ENVELOPE_SCHEMA_VERSION,
    BASELINE_STRATEGY_ARROW_CONTRACTS,
    BASELINE_STRATEGY_TABLES,
    LEGACY_SCHEMA_VERSION,
    SCHEMA_ID,
    SCHEMA_VERSION,
    contract_fields,
)


def _safe_component(value: object, *, label: str) -> str:
    text = str(value or "").strip()
    if not text or Path(text).name != text or text in {".", ".."}:
        raise ValueError(f"invalid {label}: {value!r}")
    return text


def _legacy_table_parts(
    root: Path,
    run_id: str,
    table: str,
) -> tuple[Path, ...]:
    manifest_path = root / "v1" / "manifests" / f"analysis_run_id={run_id}.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("legacy strategy manifest must contain an object")
    if (
        payload.get("schema_id") != SCHEMA_ID
        or payload.get("schema_version") != LEGACY_SCHEMA_VERSION
    ):
        raise ValueError("explicit legacy strategy layout has an unsupported schema")
    raw_parts = payload.get("part_files_by_table", {}).get(table, [])
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
                f"legacy strategy part resolves outside output root: {raw_part}"
            ) from exc
        if not path.is_file():
            raise FileNotFoundError(path)
        parts.append(path)
    return tuple(parts)


def strategy_table_parts(
    output_root: Path,
    analysis_run_id: str,
    table_name: str,
    *,
    allow_legacy_layout: bool = False,
) -> tuple[Path, ...]:
    """Resolve only manifest-selected v2 parts, or explicit legacy-v1 parts."""

    root = Path(output_root).expanduser().resolve()
    run_id = _safe_component(analysis_run_id, label="analysis run ID")
    table = _safe_component(table_name, label="table name")
    contract_fields(table)
    if allow_legacy_layout:
        return _legacy_table_parts(root, run_id, table)
    manifest_path = derived_manifest_path(root, run_id)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("strategy manifest must contain an object")
    if (
        payload.get("schema_id") != SCHEMA_ID
        or payload.get("schema_version") != SCHEMA_VERSION
    ):
        raise ValueError("strict strategy reader requires schema v2")
    validate_derived_manifest_envelope(
        payload,
        analysis_run_id=run_id,
        table_names=BASELINE_STRATEGY_TABLES,
        contracts=BASELINE_STRATEGY_ARROW_CONTRACTS,
        arrow_envelope_schema_id=ARROW_CONTRACT_ENVELOPE_SCHEMA_ID,
        arrow_envelope_schema_version=ARROW_CONTRACT_ENVELOPE_SCHEMA_VERSION,
    )
    return derived_manifest_selected_parts(
        root,
        payload,
        table,
        table_names=BASELINE_STRATEGY_TABLES,
    )


def scan_strategy_table(
    output_root: Path,
    analysis_run_id: str,
    table_name: str,
    *,
    columns: Sequence[str] | None = None,
    allow_legacy_layout: bool = False,
) -> pl.LazyFrame:
    """Return a true lazy scan with exact empty schemas under strict v2."""

    parts = strategy_table_parts(
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


__all__ = ["scan_strategy_table", "strategy_table_parts"]
