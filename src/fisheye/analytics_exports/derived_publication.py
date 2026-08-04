"""Exact immutable-generation publisher for derived analytics table families.

Canonical cross-recording exports and downstream derived families share the
same visibility rule: data bytes are immutable, a complete generation is
validated before publication, and one manifest rename is the only selection
boundary.  This module adapts that rule to small, closed Arrow table suites
without importing any family-specific scientific vocabulary.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence
import uuid

from .arrow_contract_core import (
    ArrowTableContract,
    canonical_bytes,
    exact_schema,
    payload_sha256,
    validate_contract_envelope,
    validate_exact_schema,
)
from .publication import (
    commit_validated_immutable_generation,
    manifest_identity,
    safe_component,
    sha256_file,
)


DERIVED_PUBLICATION_SCHEMA_ID = "palette.derived_analytics.publication"
DERIVED_PUBLICATION_SCHEMA_VERSION = 1
_PUBLICATION_FIELDS = {
    "schema_id",
    "schema_version",
    "state",
    "selector_eligible",
    "intended_use",
    "generation_id",
    "generation_path",
    "parts_by_table",
}
_PART_FIELDS = {"path", "sha256", "size_bytes", "row_count"}


def _component(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    return safe_component(value, label=label)


def _checked_relative_path(root: Path, relative: Path, *, label: str) -> Path:
    """Return one root-relative path after rejecting every symlink alias."""

    resolved_root = Path(root).expanduser().resolve()
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"{label} is not a safe relative path: {relative}")
    current = resolved_root
    for component in relative.parts:
        current /= component
        if current.is_symlink():
            raise ValueError(f"{label} contains a symbolic-link alias: {current}")
    try:
        current.resolve(strict=False).relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes its publication root: {relative}") from exc
    return current


def derived_manifest_path(output_root: Path, analysis_run_id: str) -> Path:
    root = Path(output_root).expanduser().resolve()
    run_id = _component(analysis_run_id, label="analysis run ID")
    return _checked_relative_path(
        root,
        Path("v2") / "manifests" / f"analysis_run_id={run_id}.json",
        label="Derived manifest path",
    )


def derived_generation_relative_path(
    analysis_run_id: str,
    generation_id: str,
) -> Path:
    run_id = _component(analysis_run_id, label="analysis run ID")
    generation = _component(generation_id, label="generation ID")
    return (
        Path("v2")
        / ".generations"
        / f"analysis_run_id={run_id}"
        / f"generation={generation}"
    )


def _staging_root(
    output_root: Path,
    analysis_run_id: str,
    generation_id: str,
) -> Path:
    root = Path(output_root).expanduser().resolve()
    run_id = _component(analysis_run_id, label="analysis run ID")
    generation = _component(generation_id, label="generation ID")
    return _checked_relative_path(
        root,
        Path("v2")
        / ".staging"
        / f"analysis_run_id={run_id}-generation={generation}",
        label="Derived staging path",
    )


def _validate_publication_envelope(
    value: object,
    *,
    analysis_run_id: str,
    expected_tables: Sequence[str],
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != _PUBLICATION_FIELDS:
        raise ValueError("Derived publication envelope has an unexpected field set")
    if value.get("schema_id") != DERIVED_PUBLICATION_SCHEMA_ID:
        raise ValueError("Derived publication schema ID is invalid")
    if value.get("schema_version") != DERIVED_PUBLICATION_SCHEMA_VERSION:
        raise ValueError("Derived publication schema version is invalid")
    if value.get("state") != "complete":
        raise ValueError("Derived publication state must be complete")
    if type(value.get("selector_eligible")) is not bool:
        raise ValueError("Derived publication selector eligibility must be boolean")
    if value.get("intended_use") != "analysis":
        raise ValueError("Derived publication intended use is invalid")
    generation_id = _component(value.get("generation_id"), label="generation ID")
    expected_generation = derived_generation_relative_path(
        analysis_run_id,
        generation_id,
    ).as_posix()
    if value.get("generation_path") != expected_generation:
        raise ValueError("Derived publication generation binding is invalid")
    inventory = value.get("parts_by_table")
    if not isinstance(inventory, Mapping) or set(inventory) != set(expected_tables):
        raise ValueError("Derived publication table inventory is incomplete")
    seen_paths: set[str] = set()
    for table_name in expected_tables:
        entries = inventory.get(table_name)
        if not isinstance(entries, list) or len(entries) != 1:
            raise ValueError(f"{table_name}: exactly one Parquet part is required")
        entry = entries[0]
        if not isinstance(entry, Mapping) or set(entry) != _PART_FIELDS:
            raise ValueError(f"{table_name}: invalid part receipt")
        expected_path = (
            Path(expected_generation) / "tables" / table_name / "part-00000.parquet"
        ).as_posix()
        if entry.get("path") != expected_path or expected_path in seen_paths:
            raise ValueError(f"{table_name}: invalid or duplicate part path")
        seen_paths.add(expected_path)
        digest = entry.get("sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"{table_name}: invalid part SHA-256")
        if type(entry.get("size_bytes")) is not int or entry["size_bytes"] < 0:
            raise ValueError(f"{table_name}: invalid part size")
        if type(entry.get("row_count")) is not int or entry["row_count"] < 0:
            raise ValueError(f"{table_name}: invalid part row count")
    return inventory


def validate_derived_manifest_envelope(
    payload: Mapping[str, Any],
    *,
    analysis_run_id: str,
    table_names: Sequence[str],
    contracts: Mapping[str, ArrowTableContract],
    arrow_envelope_schema_id: str,
    arrow_envelope_schema_version: int,
) -> None:
    """Validate the shared exact portions of one family manifest."""

    run_id = _component(analysis_run_id, label="analysis run ID")
    tables = tuple(table_names)
    if payload.get("analysis_run_id") != run_id:
        raise ValueError("Derived manifest analysis run identity is invalid")
    if payload.get("output_tables") != list(tables):
        raise ValueError("Derived manifest output table inventory is invalid")
    if set(contracts) != set(tables):
        raise ValueError("Installed derived Arrow contract inventory is incomplete")
    validate_contract_envelope(
        payload.get("arrow_schema_contracts"),
        tables,
        known_table_names=tables,
        contracts=contracts,
        schema_id=arrow_envelope_schema_id,
        schema_version=arrow_envelope_schema_version,
    )
    inventory = _validate_publication_envelope(
        payload.get("publication"),
        analysis_run_id=run_id,
        expected_tables=tables,
    )
    row_counts = payload.get("row_counts_by_table")
    part_files = payload.get("part_files_by_table")
    primary_keys = payload.get("primary_keys_by_table")
    if not isinstance(row_counts, Mapping) or set(row_counts) != set(tables):
        raise ValueError("Derived manifest row-count inventory is invalid")
    if not isinstance(part_files, Mapping) or set(part_files) != set(tables):
        raise ValueError("Derived manifest part-file inventory is invalid")
    if not isinstance(primary_keys, Mapping) or set(primary_keys) != set(tables):
        raise ValueError("Derived manifest primary-key inventory is invalid")
    for table_name in tables:
        entries = inventory[table_name]
        assert isinstance(entries, list)
        entry = entries[0]
        assert isinstance(entry, Mapping)
        if row_counts.get(table_name) != entry.get("row_count"):
            raise ValueError(f"{table_name}: row-count declarations differ")
        if part_files.get(table_name) != [entry.get("path")]:
            raise ValueError(f"{table_name}: part-file declarations differ")
        if primary_keys.get(table_name) != list(contracts[table_name].primary_key):
            raise ValueError(f"{table_name}: primary-key declaration is invalid")
    digest = payload.get("manifest_payload_sha256")
    unsigned = {key: value for key, value in payload.items() if key != "manifest_payload_sha256"}
    if digest != payload_sha256(unsigned):
        raise ValueError("Derived manifest payload digest is invalid")


def _validate_staged_generation(
    staging_root: Path,
    payload: Mapping[str, Any],
    *,
    analysis_run_id: str,
    table_names: Sequence[str],
    contracts: Mapping[str, ArrowTableContract],
    arrow_envelope_schema_id: str,
    arrow_envelope_schema_version: int,
) -> None:
    import pyarrow.parquet as pq

    stage = Path(staging_root).resolve()
    validate_derived_manifest_envelope(
        payload,
        analysis_run_id=analysis_run_id,
        table_names=table_names,
        contracts=contracts,
        arrow_envelope_schema_id=arrow_envelope_schema_id,
        arrow_envelope_schema_version=arrow_envelope_schema_version,
    )
    publication = payload["publication"]
    assert isinstance(publication, Mapping)
    inventory = publication["parts_by_table"]
    assert isinstance(inventory, Mapping)
    expected_files: set[str] = set()
    for table_name in table_names:
        entries = inventory[table_name]
        assert isinstance(entries, list)
        entry = entries[0]
        assert isinstance(entry, Mapping)
        staged_part = stage / "tables" / table_name / "part-00000.parquet"
        expected_files.add(staged_part.relative_to(stage).as_posix())
        if staged_part.is_symlink() or not staged_part.is_file():
            raise ValueError(f"{table_name}: staged part is missing or aliased")
        if staged_part.stat().st_size != entry["size_bytes"]:
            raise ValueError(f"{table_name}: staged part size mismatch")
        if sha256_file(staged_part) != entry["sha256"]:
            raise ValueError(f"{table_name}: staged part digest mismatch")
        parquet_file = pq.ParquetFile(staged_part)
        if int(parquet_file.metadata.num_rows) != entry["row_count"]:
            raise ValueError(f"{table_name}: staged part row-count mismatch")
        validate_exact_schema(contracts[table_name], parquet_file.schema_arrow)
        if "analysis_run_id" in parquet_file.schema_arrow.names:
            observed_ids = set(
                parquet_file.read(columns=["analysis_run_id"])
                .column(0)
                .to_pylist()
            )
            if observed_ids not in (set(), {analysis_run_id}):
                raise ValueError(f"{table_name}: in-row analysis run IDs differ")
    actual_files = {
        path.relative_to(stage).as_posix()
        for path in stage.rglob("*")
        if path.is_file()
    }
    if actual_files != expected_files:
        raise ValueError("Derived staged generation has files outside its inventory")


def publish_derived_table_generation(
    *,
    output_root: Path,
    analysis_run_id: str,
    rows_by_table: Mapping[str, Sequence[Mapping[str, Any]]],
    table_names: Sequence[str],
    contracts: Mapping[str, ArrowTableContract],
    arrow_contract_envelope: Mapping[str, Any],
    arrow_envelope_schema_id: str,
    arrow_envelope_schema_version: int,
    manifest_fields: Mapping[str, Any],
    footer_metadata: Mapping[bytes, bytes],
    selector_eligible: bool = True,
    generation_id: str | None = None,
) -> dict[str, Any]:
    """Write, validate, and atomically select one exact derived generation."""

    import pyarrow as pa
    import pyarrow.parquet as pq

    root = Path(output_root).expanduser().resolve()
    run_id = _component(analysis_run_id, label="analysis run ID")
    tables = tuple(table_names)
    if len(set(tables)) != len(tables) or set(rows_by_table) != set(tables):
        raise ValueError("Derived table rows must match the closed table inventory")
    if set(contracts) != set(tables):
        raise ValueError("Derived exact contracts must match the closed table inventory")
    generation = _component(
        generation_id or uuid.uuid4().hex,
        label="generation ID",
    )
    stage = _staging_root(root, run_id, generation)
    final_relative = derived_generation_relative_path(run_id, generation)
    final_generation = _checked_relative_path(
        root,
        final_relative,
        label="Derived generation path",
    )
    manifest_path = derived_manifest_path(root, run_id)
    if stage.exists() or final_generation.exists() or manifest_path.exists():
        raise FileExistsError(
            "Derived run, staging generation, or immutable generation already exists"
        )
    stage.mkdir(parents=True, exist_ok=False)
    try:
        inventory: dict[str, list[dict[str, object]]] = {}
        row_counts: dict[str, int] = {}
        part_files: dict[str, list[str]] = {}
        for table_name in tables:
            rows = [dict(row) for row in rows_by_table[table_name]]
            schema = exact_schema(
                contracts[table_name],
                metadata={
                    **footer_metadata,
                    b"palette.table_name": table_name.encode("utf-8"),
                },
            )
            table = pa.Table.from_pylist(rows, schema=schema)
            table_dir = stage / "tables" / table_name
            table_dir.mkdir(parents=True, exist_ok=False)
            temporary = table_dir / ".part-00000.parquet.tmp"
            part = table_dir / "part-00000.parquet"
            pq.write_table(table, temporary)
            os.replace(temporary, part)
            relative_part = (
                final_relative / "tables" / table_name / part.name
            ).as_posix()
            receipt = {
                "path": relative_part,
                "sha256": sha256_file(part),
                "size_bytes": part.stat().st_size,
                "row_count": table.num_rows,
            }
            inventory[table_name] = [receipt]
            row_counts[table_name] = table.num_rows
            part_files[table_name] = [relative_part]
        publication = {
            "schema_id": DERIVED_PUBLICATION_SCHEMA_ID,
            "schema_version": DERIVED_PUBLICATION_SCHEMA_VERSION,
            "state": "complete",
            "selector_eligible": bool(selector_eligible),
            "intended_use": "analysis",
            "generation_id": generation,
            "generation_path": final_relative.as_posix(),
            "parts_by_table": inventory,
        }
        reserved = {
            "analysis_run_id",
            "output_tables",
            "row_counts_by_table",
            "part_files_by_table",
            "primary_keys_by_table",
            "arrow_schema_contracts",
            "publication",
            "manifest_payload_sha256",
        }
        overlap = reserved & set(manifest_fields)
        if overlap:
            raise ValueError(f"Derived manifest fields use reserved names: {sorted(overlap)}")
        unsigned_payload: dict[str, Any] = {
            **dict(manifest_fields),
            "analysis_run_id": run_id,
            "output_tables": list(tables),
            "row_counts_by_table": row_counts,
            "part_files_by_table": part_files,
            "primary_keys_by_table": {
                table_name: list(contracts[table_name].primary_key)
                for table_name in tables
            },
            "arrow_schema_contracts": dict(arrow_contract_envelope),
            "publication": publication,
        }
        # Exercise the strict JSON boundary before touching the final namespace.
        canonical_bytes(unsigned_payload)
        payload = {
            **unsigned_payload,
            "manifest_payload_sha256": payload_sha256(unsigned_payload),
        }
        _validate_staged_generation(
            stage,
            payload,
            analysis_run_id=run_id,
            table_names=tables,
            contracts=contracts,
            arrow_envelope_schema_id=arrow_envelope_schema_id,
            arrow_envelope_schema_version=arrow_envelope_schema_version,
        )
        commit_validated_immutable_generation(
            root,
            stage,
            final_generation,
            manifest_path,
            payload,
            baseline_manifest_identity=manifest_identity(manifest_path),
            lock_directory=root / "v2" / ".locks",
            validate_staging=lambda: _validate_staged_generation(
                stage,
                payload,
                analysis_run_id=run_id,
                table_names=tables,
                contracts=contracts,
                arrow_envelope_schema_id=arrow_envelope_schema_id,
                arrow_envelope_schema_version=arrow_envelope_schema_version,
            ),
        )
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    return {**payload, "manifest_path": str(manifest_path)}


def derived_manifest_selected_parts(
    output_root: Path,
    payload: Mapping[str, Any],
    table_name: str,
    *,
    table_names: Sequence[str],
) -> tuple[Path, ...]:
    """Resolve only digest-declared parts from one selected v2 manifest."""

    root = Path(output_root).expanduser().resolve()
    run_id = _component(payload.get("analysis_run_id"), label="analysis run ID")
    table = _component(table_name, label="table name")
    inventory = _validate_publication_envelope(
        payload.get("publication"),
        analysis_run_id=run_id,
        expected_tables=table_names,
    )
    if table not in inventory:
        raise KeyError(f"Unknown derived table: {table}")
    paths: list[Path] = []
    entries = inventory[table]
    assert isinstance(entries, list)
    for raw_entry in entries:
        assert isinstance(raw_entry, Mapping)
        relative = Path(str(raw_entry["path"]))
        candidate = _checked_relative_path(
            root,
            relative,
            label=f"{table} selected part",
        )
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        if candidate.stat().st_size != raw_entry["size_bytes"]:
            raise ValueError(f"{table}: selected part size mismatch")
        if sha256_file(candidate) != raw_entry["sha256"]:
            raise ValueError(f"{table}: selected part digest mismatch")
        paths.append(candidate)
    return tuple(paths)


__all__ = [
    "DERIVED_PUBLICATION_SCHEMA_ID",
    "DERIVED_PUBLICATION_SCHEMA_VERSION",
    "derived_generation_relative_path",
    "derived_manifest_path",
    "derived_manifest_selected_parts",
    "publish_derived_table_generation",
    "validate_derived_manifest_envelope",
]
