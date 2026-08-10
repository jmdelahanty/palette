"""Manifest-committed physical publication for analytics exports.

The current logical analytics schema is ``palette.analytics_export`` v3.  This
module owns the independent physical publication envelope: one immutable
generation directory, an exact part inventory, and one manifest rename as the
only visibility commit.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
import os
from pathlib import Path
import re
import shutil
from typing import Any, Callable, Iterator, Mapping
import uuid

from .runtime_telemetry import ExportRuntimePhaseRecorder


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
_PORTABLE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$")


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


def _within_root(path: Path, root: Path, *, label: str) -> Path:
    """Resolve a lifecycle path and prove that it remains beneath ``root``."""

    resolved_root = Path(root).expanduser().resolve()
    unresolved = Path(path).expanduser().absolute()
    try:
        relative = unresolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes its publication root: {path}") from exc
    current = resolved_root
    for component in relative.parts:
        current /= component
        if current.is_symlink():
            raise ValueError(f"{label} contains a symbolic-link alias: {path}")
    resolved = unresolved.resolve(strict=False)
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes its publication root: {path}") from exc
    return resolved


@contextmanager
def immutable_manifest_commit_lock(
    publication_root: Path,
    manifest_path: Path,
    *,
    lock_directory: Path,
) -> Iterator[None]:
    """Hold a short advisory lock for one generic immutable manifest commit.

    Family-specific publishers supply their own canonical manifest and lock
    paths.  This primitive owns only containment, regular-file, and advisory
    locking rules; it does not infer or reinterpret a scientific namespace.
    """

    import fcntl

    root = Path(publication_root).expanduser().resolve()
    manifest = _within_root(manifest_path, root, label="Manifest path")
    lock_dir = _within_root(lock_directory, root, label="Lock directory")
    if manifest.is_symlink():
        raise ValueError(f"Immutable manifest must not be a symlink: {manifest}")
    if lock_dir.is_symlink():
        raise ValueError(f"Immutable lock directory must not be a symlink: {lock_dir}")
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f".{manifest.name}.lock"
    if lock_path.is_symlink():
        raise ValueError(
            f"Immutable publication lock must not be a symlink: {lock_path}"
        )
    if lock_path.exists() and not lock_path.is_file():
        raise ValueError(f"Immutable publication lock is not a file: {lock_path}")
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def commit_validated_immutable_generation(
    publication_root: Path,
    staging_root: Path,
    generation_root: Path,
    manifest_path: Path,
    payload: Mapping[str, Any],
    *,
    baseline_manifest_identity: str | None,
    lock_directory: Path,
    validate_staging: Callable[[], None],
) -> Path:
    """Commit one predeclared immutable generation with manifest-last CAS.

    The caller owns exact schema and scientific validation.  Validation must
    examine the complete staged inventory and raise on any mismatch.  This
    function then makes the generation visible with one directory rename and
    makes it authoritative with one manifest rename.  Failed validation or a
    lost manifest race removes only this unpublished generation.
    """

    root = Path(publication_root).expanduser().resolve()
    stage = _within_root(staging_root, root, label="Staging generation")
    generation = _within_root(generation_root, root, label="Final generation")
    manifest = _within_root(manifest_path, root, label="Manifest path")
    lock_dir = _within_root(lock_directory, root, label="Lock directory")
    if stage.is_symlink() or generation.is_symlink() or manifest.is_symlink():
        raise ValueError("Immutable publication lifecycle paths must not be symlinks")
    if not stage.is_dir():
        raise ValueError(f"Staged generation is missing: {stage}")
    if generation.exists():
        raise FileExistsError(f"Immutable generation already exists: {generation}")
    try:
        validate_staging()
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise

    manifest.parent.mkdir(parents=True, exist_ok=True)
    generation.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest = manifest.parent / f".{manifest.name}.{uuid.uuid4().hex}.tmp"
    if temporary_manifest.exists() or temporary_manifest.is_symlink():
        raise FileExistsError(temporary_manifest)
    os.replace(stage, generation)
    try:
        temporary_manifest.write_text(
            json.dumps(
                dict(payload),
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        with immutable_manifest_commit_lock(
            root,
            manifest,
            lock_directory=lock_dir,
        ):
            if manifest_identity(manifest) != baseline_manifest_identity:
                raise RuntimeError(
                    "Immutable publication manifest changed during commit; "
                    "the generation was not selected"
                )
            os.replace(temporary_manifest, manifest)
    except Exception:
        temporary_manifest.unlink(missing_ok=True)
        if generation.exists():
            shutil.rmtree(generation)
        raise
    return manifest


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
        raise ValueError(
            f"Analytics export lock escapes its root: {lock_path}"
        ) from exc
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
    if (
        not isinstance(publication.get("schema_id"), str)
        or publication.get("schema_id") != PUBLICATION_SCHEMA_ID
    ):
        raise ValueError("Export publication schema ID is invalid")
    if (
        type(publication.get("schema_version")) is not int
        or publication.get("schema_version") != PUBLICATION_SCHEMA_VERSION
    ):
        raise ValueError("Export publication schema version is invalid")
    if (
        not isinstance(publication.get("state"), str)
        or publication.get("state") != "complete"
    ):
        raise ValueError("Export publication state must be 'complete'")
    raw_run_id = payload.get("export_run_id")
    if not isinstance(raw_run_id, str):
        raise ValueError("Export run ID must be a string")
    run_id = safe_component(raw_run_id, label="export run ID")
    raw_generation_id = publication.get("generation_id")
    if not isinstance(raw_generation_id, str):
        raise ValueError("Export generation ID must be a string")
    generation_id = safe_component(raw_generation_id, label="generation ID")
    expected_path = generation_relative_path(run_id, generation_id)
    raw_generation_path = publication.get("generation_path")
    if not isinstance(raw_generation_path, str):
        raise ValueError("Export generation path must be a string")
    if raw_generation_path != expected_path.as_posix():
        raise ValueError(
            "Export publication generation path/identity binding is invalid"
        )
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
            if (
                not isinstance(raw_entry, Mapping)
                or set(raw_entry) != _INVENTORY_FIELDS
            ):
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
        REGISTRY_IDENTITY_TABLES,
        TABLE_CONTRACTS,
        validate_table_columns,
    )
    from .chaser_authority import validate_chaser_export_authority_receipt
    from .registry_identity import (
        registry_identity_sources_by_path,
        validate_registry_identity_receipt,
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
    if (
        payload.get("schema_id") != EXPORT_SCHEMA_ID
        or payload.get("schema_version") != EXPORT_SCHEMA_VERSION
    ):
        raise ValueError("Staged publication must use the current export schema")
    validate_arrow_contract_envelope(
        payload.get("arrow_schema_contracts"),
        tuple(sorted(declared_tables)),
    )
    source_zarrs = payload.get("source_zarrs")
    if source_zarrs is None:
        source_zarrs = []
    if not isinstance(source_zarrs, list):
        raise ValueError("Staged publication source_zarrs declaration is invalid")
    registry_sources: Mapping[str, Mapping[str, Any]] = {}
    if declared_tables & REGISTRY_IDENTITY_TABLES:
        registry_receipt = validate_registry_identity_receipt(
            payload.get("registry_identity"),
            expected_zarr_paths=source_zarrs,
        )
        registry_sources = registry_identity_sources_by_path(registry_receipt)
    elif payload.get("registry_identity") is not None:
        raise ValueError(
            "Staged publication has registry identity evidence without identity tables"
        )
    if payload.get("chaser_export_authority") is not None:
        validate_chaser_export_authority_receipt(
            payload.get("chaser_export_authority"),
            expected_zarr_paths=source_zarrs,
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
            raise ValueError(
                f"{table}: staged logical part list differs from inventory"
            )
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
            if (
                metadata.get(b"palette.export_schema_id", b"").decode()
                != EXPORT_SCHEMA_ID
            ):
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
            part_rows = int(parquet_file.metadata.num_rows)
            if table in REGISTRY_IDENTITY_TABLES and part_rows:
                identity_table = parquet_file.read(
                    columns=[
                        "zarr_path",
                        "recording_id",
                        "acquisition_batch_id",
                        "subject_id",
                    ]
                )
                identity_values = set(
                    zip(
                        *(
                            identity_table.column(name).to_pylist()
                            for name in (
                                "zarr_path",
                                "recording_id",
                                "acquisition_batch_id",
                                "subject_id",
                            )
                        ),
                        strict=True,
                    )
                )
                if identity_table.num_rows != part_rows or not identity_values:
                    raise ValueError(
                        f"{table}: staged part identity columns are incomplete"
                    )
                for (
                    source_path,
                    recording_id,
                    acquisition_batch_id,
                    subject_id,
                ) in identity_values:
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
                        raise ValueError(
                            f"{table}: staged persisted registry identity differs "
                            f"from receipt for {source_path!r}"
                        )
            if reference_columns is None:
                reference_columns = columns
            elif columns != reference_columns:
                raise ValueError(f"{table}: staged part schemas differ")
            table_rows += part_rows
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


def commit_staged_publication(
    export_root: Path,
    staging_root: Path,
    payload: Mapping[str, Any],
    *,
    baseline_manifest_identity: str | None,
    runtime_recorder: ExportRuntimePhaseRecorder | None = None,
) -> Path:
    """Validate and atomically commit one immutable analytics generation.

    This is the shared visibility boundary for compact tables and streaming
    trace exports.  Generation bytes become visible first under their immutable
    identity; a short compare-and-swap manifest rename is the sole authority
    commit.  Any failure before that rename removes the unpublished generation.
    """

    root = Path(export_root).expanduser().resolve()
    stage = Path(staging_root).expanduser().resolve()
    raw_run_id = payload.get("export_run_id")
    if not isinstance(raw_run_id, str):
        raise ValueError("Export run ID must be a string")
    run_id = safe_component(raw_run_id, label="export run ID")
    publication = payload.get("publication")
    if not isinstance(publication, Mapping):
        raise ValueError("Export publication envelope is missing or invalid")
    generation_id = safe_component(
        publication.get("generation_id"),
        label="generation ID",
    )
    expected_stage = publication_staging_root(root, run_id, generation_id)
    if stage != expected_stage:
        raise ValueError(
            "Staged publication path does not match its generation identity"
        )
    manifest_path = export_manifest_path(root, run_id)
    final_generation = publication_generation_root(root, run_id, generation_id)

    def validate_stage() -> None:
        # Cleanup is authorized only after the supplied path has been proven to
        # be this payload's canonical hidden staging generation.
        validate_publication_envelope(payload)
        if final_generation.exists():
            raise FileExistsError(
                f"Analytics export generation already exists: {final_generation}"
            )
        validate_staged_publication(stage, payload)

    try:
        if runtime_recorder is None:
            validate_stage()
        else:
            with runtime_recorder.measure("publication_staged_validation"):
                validate_stage()
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    final_generation.parent.mkdir(parents=True, exist_ok=True)
    tmp_manifest = manifest_path.parent / (
        f".export_run_id={run_id}.generation={generation_id}.json.tmp"
    )
    if runtime_recorder is None:
        os.replace(stage, final_generation)
    else:
        with runtime_recorder.measure("publication_generation_rename"):
            os.replace(stage, final_generation)

    def commit_manifest() -> None:
        tmp_manifest.write_text(
            json.dumps(
                dict(payload),
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        with manifest_commit_lock(manifest_path):
            if manifest_identity(manifest_path) != baseline_manifest_identity:
                raise RuntimeError(
                    "Analytics export manifest changed during publication; "
                    "the staged generation was not committed"
                )
            os.replace(tmp_manifest, manifest_path)

    try:
        if runtime_recorder is None:
            commit_manifest()
        else:
            with runtime_recorder.measure("publication_manifest_commit"):
                commit_manifest()
    except Exception:
        tmp_manifest.unlink(missing_ok=True)
        if final_generation.exists():
            shutil.rmtree(final_generation)
        raise
    return manifest_path


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
            raise ValueError(
                "Export manifest part_files_by_table is missing or invalid"
            )
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
    "commit_validated_immutable_generation",
    "export_manifest_directory",
    "export_manifest_path",
    "commit_staged_publication",
    "generation_relative_path",
    "has_exclusive_inventory",
    "load_export_manifest",
    "manifest_commit_lock",
    "manifest_identity",
    "immutable_manifest_commit_lock",
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
