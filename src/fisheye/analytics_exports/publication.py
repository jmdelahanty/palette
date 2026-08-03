"""Manifest-committed physical publication for analytics exports.

The logical analytics schema remains ``palette.analytics_export`` v2.  This
module owns the independent physical publication envelope: one immutable
generation directory, an exact part inventory, and one manifest rename as the
only visibility commit.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
import re
from typing import Any, Iterator, Mapping


PUBLICATION_SCHEMA_ID = "palette.analytics_export.publication"
PUBLICATION_SCHEMA_VERSION = 1
_PUBLICATION_FIELDS = {
    "schema_id",
    "schema_version",
    "state",
    "generation_id",
    "generation_path",
    "parts_by_table",
}
_INVENTORY_FIELDS = {"path", "sha256", "size_bytes", "row_count"}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PORTABLE_COMPONENT_RE = re.compile(
    r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$"
)


def safe_component(value: object, *, label: str) -> str:
    """Return a canonical portable identifier or reject it.

    Persisted identifiers use ASCII letters/digits with internal ``.``, ``_``,
    or ``-`` separators.  Requiring alphanumeric endpoints avoids hidden names,
    option-like names, and platform-specific trailing-character behavior.
    """

    raw_text = str(value)
    text = raw_text.strip()
    if (
        not text
        or raw_text != text
        or not _PORTABLE_COMPONENT_RE.fullmatch(text)
        or Path(text).name != text
        or text in {".", ".."}
    ):
        raise ValueError(f"Invalid {label}: {value!r}")
    return text


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_identity(path: Path) -> str | None:
    """Return the byte identity used by the publication compare-and-swap."""

    return sha256_file(path) if path.is_file() else None


def _checked_lifecycle_directory(export_root: Path, *parts: str) -> Path:
    """Resolve a lifecycle directory without following repository-local links."""

    root = Path(export_root).expanduser().resolve()
    candidate = root
    for part in parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise ValueError(
                f"Analytics export lifecycle path must not be a symlink: {candidate}"
            )
        if candidate.exists() and not candidate.is_dir():
            raise ValueError(
                f"Analytics export lifecycle path is not a directory: {candidate}"
            )
        try:
            candidate.resolve(strict=False).relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"Analytics export lifecycle path escapes its root: {candidate}"
            ) from exc
    return candidate


def export_manifest_directory(export_root: Path) -> Path:
    """Return the canonical manifest directory after fail-closed path checks."""

    return _checked_lifecycle_directory(export_root, "v1", "manifests")


def export_manifest_path(export_root: Path, export_run_id: str) -> Path:
    """Return one canonical manifest path without following a manifest link."""

    run_id = safe_component(export_run_id, label="export run ID")
    root = Path(export_root).expanduser().resolve()
    path = export_manifest_directory(root) / f"export_run_id={run_id}.json"
    if path.is_symlink():
        try:
            path.resolve(strict=False).relative_to(root)
        except ValueError as exc:
            raise PermissionError(
                f"Export manifest resolves outside the authorized root: {path}"
            ) from exc
        raise ValueError(f"Analytics export manifest must not be a symlink: {path}")
    try:
        path.resolve(strict=False).relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Analytics export manifest escapes its root: {path}") from exc
    return path


def publication_staging_root(
    export_root: Path,
    export_run_id: str,
    generation_id: str,
) -> Path:
    """Return the private staging root after checking every lifecycle component."""

    run_id = safe_component(export_run_id, label="export run ID")
    generation = safe_component(generation_id, label="generation ID")
    return _checked_lifecycle_directory(
        export_root,
        "v1",
        ".staging",
        f"export_run_id={run_id}-generation={generation}",
    )


def publication_generation_root(
    export_root: Path,
    export_run_id: str,
    generation_id: str,
) -> Path:
    """Return an immutable generation root after checking lifecycle components."""

    run_id = safe_component(export_run_id, label="export run ID")
    generation = safe_component(generation_id, label="generation ID")
    return _checked_lifecycle_directory(
        export_root,
        "v1",
        ".generations",
        f"export_run_id={run_id}",
        f"generation={generation}",
    )


@contextmanager
def manifest_commit_lock(manifest_path: Path) -> Iterator[None]:
    """Hold the short per-run advisory lock used only for manifest commit."""

    import fcntl

    manifest_path = Path(manifest_path)
    export_root = manifest_path.parent.parent.parent
    expected_manifest = export_manifest_path(
        export_root,
        manifest_path.name.removeprefix("export_run_id=").removesuffix(".json"),
    )
    if manifest_path != expected_manifest:
        raise ValueError("Manifest lock path is not the canonical export manifest path")
    lock_dir = _checked_lifecycle_directory(export_root, "v1", ".locks")
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f".{manifest_path.name}.lock"
    if lock_path.is_symlink():
        raise ValueError(f"Analytics export lock must not be a symlink: {lock_path}")
    if lock_path.exists() and not lock_path.is_file():
        raise ValueError(f"Analytics export lock is not a regular file: {lock_path}")
    try:
        lock_path.resolve(strict=False).relative_to(export_root.resolve())
    except ValueError as exc:
        raise ValueError(f"Analytics export lock escapes its root: {lock_path}") from exc
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def load_export_manifest(export_root: Path, export_run_id: str) -> dict[str, Any]:
    path = export_manifest_path(export_root, export_run_id)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Export manifest must contain an object: {path}")
    return payload


def has_exclusive_inventory(payload: Mapping[str, Any]) -> bool:
    publication = payload.get("publication")
    return bool(
        isinstance(publication, Mapping)
        and publication.get("schema_id") == PUBLICATION_SCHEMA_ID
        and publication.get("schema_version") == PUBLICATION_SCHEMA_VERSION
    )


def generation_relative_path(export_run_id: str, generation_id: str) -> Path:
    export_run_id = safe_component(export_run_id, label="export run ID")
    generation_id = safe_component(generation_id, label="generation ID")
    return (
        Path("v1")
        / ".generations"
        / f"export_run_id={export_run_id}"
        / f"generation={generation_id}"
    )


def _safe_relative_part(root: Path, value: object) -> Path:
    path_text = str(value)
    relative = Path(path_text)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or not relative.parts
        or relative.as_posix() != path_text
    ):
        raise ValueError(f"Invalid manifest-relative analytics part path: {value!r}")
    candidate = root / relative
    current = root
    for component in relative.parts:
        current /= component
        if current.is_symlink():
            raise ValueError(
                f"Analytics part path contains a symbolic-link alias: {value!r}"
            )
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"Analytics part resolves outside export root: {value!r}"
        ) from exc
    return candidate


def publication_inventory(
    payload: Mapping[str, Any],
) -> Mapping[str, object]:
    publication = payload.get("publication")
    if not isinstance(publication, Mapping):
        raise ValueError("Export publication envelope is missing or invalid")
    inventory = publication.get("parts_by_table")
    if not isinstance(inventory, Mapping):
        raise ValueError("Export publication part inventory is missing or invalid")
    return inventory


def validate_publication_envelope(
    payload: Mapping[str, Any],
) -> Mapping[str, object]:
    """Validate the exact fail-closed publication-v1 envelope and entries."""

    publication = payload.get("publication")
    if not isinstance(publication, Mapping):
        raise ValueError("Export publication envelope is missing or invalid")
    if set(publication) != _PUBLICATION_FIELDS:
        raise ValueError("Export publication envelope has an unexpected field set")
    if not isinstance(publication.get("schema_id"), str) or publication.get(
        "schema_id"
    ) != PUBLICATION_SCHEMA_ID:
        raise ValueError("Export publication schema ID is invalid")
    if type(publication.get("schema_version")) is not int or publication.get(
        "schema_version"
    ) != PUBLICATION_SCHEMA_VERSION:
        raise ValueError("Export publication schema version is invalid")
    if not isinstance(publication.get("state"), str) or publication.get("state") != "complete":
        raise ValueError("Export publication state must be 'complete'")
    raw_run_id = payload.get("export_run_id")
    if not isinstance(raw_run_id, str):
        raise ValueError("Export run ID must be a string")
    run_id = safe_component(raw_run_id, label="export run ID")
    raw_generation_id = publication.get("generation_id")
    if not isinstance(raw_generation_id, str):
        raise ValueError("Export generation ID must be a string")
    generation_id = safe_component(
        raw_generation_id, label="generation ID"
    )
    expected_path = generation_relative_path(run_id, generation_id)
    raw_generation_path = publication.get("generation_path")
    if not isinstance(raw_generation_path, str):
        raise ValueError("Export generation path must be a string")
    if raw_generation_path != expected_path.as_posix():
        raise ValueError("Export publication generation path/identity binding is invalid")
    inventory = publication.get("parts_by_table")
    if not isinstance(inventory, Mapping):
        raise ValueError("Export publication part inventory is missing or invalid")
    seen_paths: set[str] = set()
    for raw_table, raw_entries in inventory.items():
        if not isinstance(raw_table, str):
            raise ValueError("Export table name must be a string")
        table = safe_component(raw_table, label="table name")
        if not isinstance(raw_entries, list):
            raise ValueError(f"{table}: publication inventory is not an array")
        expected_parent = expected_path / "tables" / table
        for raw_entry in raw_entries:
            if not isinstance(raw_entry, Mapping) or set(raw_entry) != _INVENTORY_FIELDS:
                raise ValueError(f"{table}: invalid publication inventory entry fields")
            path_value = raw_entry.get("path")
            if not isinstance(path_value, str):
                raise ValueError(f"{table}: publication part path must be a string")
            relative = Path(path_value)
            if (
                relative.as_posix() != path_value
                or relative.parent != expected_parent
                or relative.suffix != ".parquet"
            ):
                raise ValueError(f"{table}: invalid publication part path")
            path_text = relative.as_posix()
            if path_text in seen_paths:
                raise ValueError(f"{table}: duplicate publication part path")
            seen_paths.add(path_text)
            digest = raw_entry.get("sha256")
            if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
                raise ValueError(f"{table}: invalid publication part digest")
            size = raw_entry.get("size_bytes")
            rows = raw_entry.get("row_count")
            if type(size) is not int or size < 0:
                raise ValueError(f"{table}: invalid publication part size")
            if type(rows) is not int or rows < 0:
                raise ValueError(f"{table}: invalid publication part row count")
    return inventory


def validate_staged_publication(
    staging_root: Path,
    payload: Mapping[str, Any],
) -> None:
    """Validate all staged bytes and declarations before the visibility commit."""

    import pyarrow.parquet as pq

    from .capabilities import resolve_capabilities
    from .arrow_contracts import (
        validate_arrow_contract_envelope,
        validate_arrow_schema,
    )
    from .contracts import (
        EXPORT_SCHEMA_ID,
        EXPORT_SCHEMA_VERSION,
        TABLE_CONTRACTS,
        validate_table_columns,
    )

    raw_staging_root = Path(staging_root)
    if raw_staging_root.is_symlink():
        raise ValueError("Staged generation directory must not be a symbolic link")
    staging_root = raw_staging_root.resolve()
    if not staging_root.is_dir():
        raise ValueError("Staged generation directory is missing")
    inventory = validate_publication_envelope(payload)
    declared_tables: set[str] = set()
    for field in ("tables_requested", "output_tables"):
        value = payload.get(field)
        if isinstance(value, list):
            declared_tables.update(str(item) for item in value)
    for field in ("row_counts_by_table", "part_files_by_table"):
        value = payload.get(field)
        if isinstance(value, Mapping):
            declared_tables.update(str(item) for item in value)
    if set(inventory) != declared_tables:
        raise ValueError("Staged publication inventory does not match declared tables")
    validate_arrow_contract_envelope(
        payload.get("arrow_schema_contracts"),
        tuple(sorted(declared_tables)),
    )

    part_files = payload.get("part_files_by_table")
    row_counts = payload.get("row_counts_by_table")
    contracts = payload.get("table_contracts")
    if not isinstance(part_files, Mapping) or not isinstance(row_counts, Mapping):
        raise ValueError("Staged publication lacks logical part/row declarations")
    if not isinstance(contracts, Mapping):
        raise ValueError("Staged publication lacks table contracts")

    publication = payload["publication"]
    assert isinstance(publication, Mapping)
    generation_path = Path(str(publication["generation_path"]))
    expected_files: set[str] = set()
    columns_by_table: dict[str, tuple[str, ...]] = {}
    for raw_table, raw_entries in inventory.items():
        table = str(raw_table)
        if table not in TABLE_CONTRACTS:
            raise ValueError(f"Unknown staged analytics table: {table}")
        if contracts.get(table) != TABLE_CONTRACTS[table].to_dict():
            raise ValueError(f"{table}: staged table contract is invalid")
        if not isinstance(raw_entries, list):
            raise ValueError(f"{table}: staged inventory is invalid")
        declared_paths = part_files.get(table)
        inventory_paths = [str(entry["path"]) for entry in raw_entries]
        if declared_paths != inventory_paths:
            raise ValueError(f"{table}: staged logical part list differs from inventory")
        table_rows = 0
        reference_columns: tuple[str, ...] | None = None
        for entry in raw_entries:
            final_path = Path(str(entry["path"]))
            try:
                relative_inside_generation = final_path.relative_to(generation_path)
            except ValueError as exc:
                raise ValueError(
                    f"{table}: staged part is outside its declared generation"
                ) from exc
            relative_text = relative_inside_generation.as_posix()
            staged_part = staging_root / relative_inside_generation
            current = staging_root
            for component in relative_inside_generation.parts:
                current /= component
                if current.is_symlink():
                    raise ValueError(
                        f"{table}: staged part path contains a symbolic-link alias"
                    )
            if not relative_text or relative_text in expected_files:
                raise ValueError(f"{table}: duplicate or invalid staged part path")
            expected_files.add(relative_text)
            if not staged_part.is_file():
                raise ValueError(f"{table}: staged part is missing: {staged_part}")
            if staged_part.stat().st_size != entry["size_bytes"]:
                raise ValueError(f"{table}: staged part size mismatch")
            if sha256_file(staged_part) != entry["sha256"]:
                raise ValueError(f"{table}: staged part digest mismatch")
            parquet_file = pq.ParquetFile(staged_part)
            if int(parquet_file.metadata.num_rows) != entry["row_count"]:
                raise ValueError(f"{table}: staged part row-count mismatch")
            schema = parquet_file.schema_arrow
            metadata = schema.metadata or {}
            if metadata.get(b"palette.export_schema_id", b"").decode() != EXPORT_SCHEMA_ID:
                raise ValueError(f"{table}: staged footer schema ID is invalid")
            if metadata.get(b"palette.export_schema_version", b"").decode() != str(
                EXPORT_SCHEMA_VERSION
            ):
                raise ValueError(f"{table}: staged footer schema version is invalid")
            footer_contract = json.loads(
                metadata.get(b"palette.table_contract", b"null").decode()
            )
            if footer_contract != TABLE_CONTRACTS[table].to_dict():
                raise ValueError(f"{table}: staged footer table contract is invalid")
            validate_arrow_schema(table, schema)
            columns = tuple(field.name for field in schema)
            missing = validate_table_columns(table, columns)
            if missing:
                raise ValueError(f"{table}: staged part is missing required columns")
            if reference_columns is None:
                reference_columns = columns
            elif columns != reference_columns:
                raise ValueError(f"{table}: staged part schemas differ")
            table_rows += int(parquet_file.metadata.num_rows)
        if table_rows != row_counts.get(table):
            raise ValueError(f"{table}: staged rows differ from manifest")
        if reference_columns is not None:
            columns_by_table[table] = reference_columns

    actual_files: set[str] = set()
    for path in staging_root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Staged generation contains a symbolic-link alias")
        if path.is_file():
            actual_files.add(path.relative_to(staging_root).as_posix())
    if actual_files != expected_files:
        raise ValueError("Staged generation contains files outside its exact inventory")
    capabilities = [
        item.capability_id
        for item in resolve_capabilities(columns_by_table)
        if item.available
    ]
    if payload.get("capabilities") != capabilities:
        raise ValueError("Staged publication capabilities are invalid")


def manifest_selected_part_files_from_payload(
    export_root: Path,
    payload: Mapping[str, Any],
    table_name: str,
    *,
    allow_legacy_layout: bool = False,
) -> tuple[Path, ...]:
    """Resolve only files selected by an exact publication-v1 manifest.

    Historical exports remain reachable only through the explicitly enabled
    compatibility path.  Strict callers never glob table directories.
    """

    root = Path(export_root).expanduser().resolve()
    raw_run_id = payload.get("export_run_id")
    if not isinstance(raw_run_id, str):
        raise ValueError("Export run ID must be a string")
    run_id = safe_component(raw_run_id, label="export run ID")
    table = safe_component(table_name, label="table name")
    if has_exclusive_inventory(payload):
        validate_publication_envelope(payload)
        publication = payload["publication"]
        assert isinstance(publication, Mapping)
        generation_id = str(publication.get("generation_id") or "")
        generation_path = Path(str(publication.get("generation_path") or ""))
        expected_generation = generation_relative_path(run_id, generation_id)
        if generation_path != expected_generation:
            raise ValueError(
                "Export publication generation path/identity binding is invalid"
            )
        publication_generation_root(root, run_id, generation_id)
        raw_entries = publication_inventory(payload).get(table, [])
        if not isinstance(raw_entries, list):
            raise ValueError(f"{table}: publication inventory is not an array")
        expected_parent = generation_path / "tables" / table
        resolved: list[Path] = []
        seen: set[str] = set()
        for raw_entry in raw_entries:
            if not isinstance(raw_entry, Mapping):
                raise ValueError(f"{table}: invalid publication inventory entry")
            entry_path = str(raw_entry.get("path") or "")
            if entry_path in seen:
                raise ValueError(f"{table}: duplicate publication part path")
            seen.add(entry_path)
            relative = Path(entry_path)
            if relative.parent != expected_parent:
                raise ValueError(
                    f"{table}: selected part is outside its bound generation table"
                )
            resolved.append(_safe_relative_part(root, entry_path))

        declared = payload.get("part_files_by_table")
        if not isinstance(declared, Mapping):
            raise ValueError("Export manifest part_files_by_table is missing or invalid")
        declared_parts = declared.get(table, [])
        if declared_parts != [str(entry.get("path")) for entry in raw_entries]:
            raise ValueError(
                f"{table}: logical part list differs from publication inventory"
            )
        return tuple(resolved)

    if not allow_legacy_layout:
        raise ValueError(
            "Export lacks the required manifest-exclusive publication-v1 inventory"
        )
    parts_by_table = payload.get("part_files_by_table")
    table_dir = root / "v1" / table / f"export_run_id={run_id}"
    if not isinstance(parts_by_table, Mapping):
        return tuple(sorted(table_dir.glob("*.parquet")))
    raw_parts = parts_by_table.get(table, [])
    if not isinstance(raw_parts, list):
        raise ValueError(f"{table}: legacy manifest part list is not an array")
    if len({Path(str(item)).name for item in raw_parts}) != len(raw_parts):
        raise ValueError(f"{table}: legacy manifest has duplicate part paths")
    return tuple((table_dir / Path(str(item)).name).resolve() for item in raw_parts)


def manifest_selected_part_files(
    export_root: Path,
    export_run_id: str,
    table_name: str,
    *,
    allow_legacy_layout: bool = False,
) -> tuple[Path, ...]:
    payload = load_export_manifest(export_root, export_run_id)
    return manifest_selected_part_files_from_payload(
        export_root,
        payload,
        table_name,
        allow_legacy_layout=allow_legacy_layout,
    )


__all__ = [
    "PUBLICATION_SCHEMA_ID",
    "PUBLICATION_SCHEMA_VERSION",
    "export_manifest_directory",
    "export_manifest_path",
    "generation_relative_path",
    "has_exclusive_inventory",
    "load_export_manifest",
    "manifest_commit_lock",
    "manifest_identity",
    "manifest_selected_part_files",
    "manifest_selected_part_files_from_payload",
    "publication_generation_root",
    "publication_staging_root",
    "publication_inventory",
    "safe_component",
    "sha256_file",
    "validate_publication_envelope",
    "validate_staged_publication",
]
