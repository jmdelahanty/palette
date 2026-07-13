"""Lazy Parquet queries for published baseline strategy analytics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import polars as pl

from .contracts import contract_fields


def _safe_component(value: object, *, label: str) -> str:
    text = str(value or "").strip()
    if not text or Path(text).name != text or text in {".", ".."}:
        raise ValueError(f"invalid {label}: {value!r}")
    return text


def strategy_table_parts(
    output_root: Path, analysis_run_id: str, table_name: str
) -> tuple[Path, ...]:
    """Resolve manifest-declared parts without trusting historical absolute paths."""

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
    parts = []
    for raw_part in raw_parts:
        path = (table_root / Path(str(raw_part)).name).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise PermissionError(f"strategy part resolves outside output root: {raw_part}") from exc
        if not path.is_file():
            raise FileNotFoundError(path)
        parts.append(path)
    return tuple(parts)


def scan_strategy_table(
    output_root: Path,
    analysis_run_id: str,
    table_name: str,
    *,
    columns: Sequence[str] | None = None,
) -> pl.LazyFrame:
    """Return a true lazy Parquet scan for one derived strategy table."""

    parts = strategy_table_parts(output_root, analysis_run_id, table_name)
    if not parts:
        return (
            pl.DataFrame({column: [] for column in columns}).lazy()
            if columns is not None
            else pl.DataFrame().lazy()
        )
    lazy = pl.scan_parquet([str(path) for path in parts], hive_partitioning=False)
    return lazy.select(list(columns)) if columns is not None else lazy


__all__ = ["scan_strategy_table", "strategy_table_parts"]
