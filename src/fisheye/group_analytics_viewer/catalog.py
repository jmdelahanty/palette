"""Read-only discovery for analytics exports beneath an authorized root."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Mapping

from fisheye.analytics_exports.capabilities import resolve_capabilities
from fisheye.analytics_exports.contracts import (
    EXPORT_SCHEMA_ID,
    EXPORT_SCHEMA_VERSION,
    TABLE_CONTRACTS,
)
from fisheye.analytics_exports.publication import (
    has_exclusive_inventory,
    manifest_selected_part_files_from_payload,
    validate_publication_envelope,
)


@dataclass(frozen=True)
class ExportCatalogDiagnostic:
    """A manifest that could not safely become a selectable export."""

    manifest_path: str
    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {
            "manifest_path": self.manifest_path,
            "code": self.code,
            "message": self.message,
        }


@dataclass(frozen=True)
class ExportCatalogEntry:
    """Small manifest-derived summary used by dataset selectors."""

    export_run_id: str
    manifest_path: str
    created_at_utc: str | None
    source_recording_count: int | None
    collection_id: str | None
    collection_name: str | None
    table_names: tuple[str, ...]
    schema_id: str
    schema_version: int
    capabilities: tuple[str, ...]
    export_diagnostics_count: int
    missing_part_count: int

    @property
    def ready(self) -> bool:
        return self.missing_part_count == 0

    @property
    def label(self) -> str:
        collection = self.collection_name or self.collection_id
        prefix = f"{collection} · " if collection else ""
        recordings = (
            f"{self.source_recording_count} recording"
            f"{'s' if self.source_recording_count != 1 else ''}"
            if self.source_recording_count is not None
            else "recording count unknown"
        )
        state = "ready" if self.ready else f"{self.missing_part_count} missing part(s)"
        return f"{prefix}{self.export_run_id} · {recordings} · {state}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "export_run_id": self.export_run_id,
            "manifest_path": self.manifest_path,
            "created_at_utc": self.created_at_utc,
            "source_recording_count": self.source_recording_count,
            "collection_id": self.collection_id,
            "collection_name": self.collection_name,
            "table_names": list(self.table_names),
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "capabilities": list(self.capabilities),
            "export_diagnostics_count": self.export_diagnostics_count,
            "missing_part_count": self.missing_part_count,
            "ready": self.ready,
            "label": self.label,
        }


@dataclass(frozen=True)
class ExportCatalog:
    """Selectable exports and rejected-manifest diagnostics for one root."""

    export_root: Path
    entries: tuple[ExportCatalogEntry, ...]
    diagnostics: tuple[ExportCatalogDiagnostic, ...]

    def entry(self, export_run_id: str) -> ExportCatalogEntry | None:
        return next(
            (entry for entry in self.entries if entry.export_run_id == export_run_id),
            None,
        )


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _safe_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("manifest must contain a JSON object")
    return payload


def _referenced_parts(payload: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    parts_by_table = payload.get("part_files_by_table")
    if not isinstance(parts_by_table, Mapping):
        return ()
    parts: list[tuple[str, str]] = []
    for table_name, values in parts_by_table.items():
        if not isinstance(values, list):
            continue
        parts.extend(
            (str(table_name), str(value))
            for value in values
            if isinstance(value, (str, Path))
        )
    return tuple(parts)


def _table_names(payload: Mapping[str, Any]) -> tuple[str, ...]:
    names: set[str] = set()
    for field in ("tables_requested", "output_tables"):
        values = payload.get(field)
        if isinstance(values, list):
            names.update(str(value) for value in values)
    for field in ("row_counts_by_table", "part_files_by_table"):
        values = payload.get(field)
        if isinstance(values, Mapping):
            names.update(str(value) for value in values)
    return tuple(sorted(names))


def _timestamp_sort_value(value: str | None) -> float:
    if not value:
        return float("-inf")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return float("-inf")


def discover_export_catalog(export_root: Path) -> ExportCatalog:
    """Discover safe base-export manifests without reading Parquet contents.

    Paths in manifests are checked after symlink resolution. A manifest that
    references anything outside ``export_root`` is rejected rather than exposed
    to the viewer. Missing in-root parts are reported on the entry so the user
    can still inspect its provenance and health.
    """

    root = Path(export_root).expanduser().resolve()
    diagnostics: list[ExportCatalogDiagnostic] = []
    entries: list[ExportCatalogEntry] = []
    manifest_dir = root / "v1" / "manifests"
    resolved_manifest_dir = manifest_dir.resolve()
    if not _is_within(resolved_manifest_dir, root):
        diagnostics.append(
            ExportCatalogDiagnostic(
                manifest_path=str(manifest_dir),
                code="manifest_directory_outside_root",
                message="Manifest directory resolves outside the authorized export root.",
            )
        )
        return ExportCatalog(root, (), tuple(diagnostics))
    if not manifest_dir.is_dir():
        diagnostics.append(
            ExportCatalogDiagnostic(
                manifest_path=str(manifest_dir),
                code="manifest_directory_missing",
                message="No v1/manifests directory exists under the export root.",
            )
        )
        return ExportCatalog(root, (), tuple(diagnostics))

    for manifest_path in sorted(manifest_dir.glob("export_run_id=*.json")):
        resolved_manifest = manifest_path.resolve()
        if not _is_within(resolved_manifest, root):
            diagnostics.append(
                ExportCatalogDiagnostic(
                    manifest_path=str(manifest_path),
                    code="manifest_outside_root",
                    message="Manifest resolves outside the authorized export root.",
                )
            )
            continue
        try:
            payload = _load_manifest(resolved_manifest)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            diagnostics.append(
                ExportCatalogDiagnostic(
                    manifest_path=str(manifest_path),
                    code="invalid_manifest",
                    message=str(exc),
                )
            )
            continue

        # Statistical and other derived exports point back to a base export and
        # should not appear as independently selectable datasets.
        if payload.get("source_export_run_id"):
            continue

        if (
            payload.get("schema_id") != EXPORT_SCHEMA_ID
            or payload.get("schema_version") != EXPORT_SCHEMA_VERSION
        ):
            diagnostics.append(
                ExportCatalogDiagnostic(
                    manifest_path=str(manifest_path),
                    code="unsupported_export_schema",
                    message=(
                        f"Unsupported analytics export contract {payload.get('schema_id')!r} "
                        f"version {payload.get('schema_version')!r}; re-export with "
                        f"{EXPORT_SCHEMA_ID} version {EXPORT_SCHEMA_VERSION}."
                    ),
                )
            )
            continue

        filename_run_id = manifest_path.name.removeprefix("export_run_id=").removesuffix(".json")
        payload_run_id = payload.get("export_run_id")
        if not isinstance(payload_run_id, str) or payload_run_id != filename_run_id:
            diagnostics.append(
                ExportCatalogDiagnostic(
                    manifest_path=str(manifest_path),
                    code="export_run_id_mismatch",
                    message=(
                        "Manifest export_run_id does not match its filename: "
                        f"{payload_run_id!r} != {filename_run_id!r}."
                    ),
                )
            )
            continue

        if not has_exclusive_inventory(payload):
            diagnostics.append(
                ExportCatalogDiagnostic(
                    manifest_path=str(manifest_path),
                    code="legacy_publication_not_selectable",
                    message=(
                        "Manifest lacks the exact publication-v1 inventory and is "
                        "available only through an explicit legacy adapter."
                    ),
                )
            )
            continue
        try:
            inventory = validate_publication_envelope(payload)
        except ValueError as exc:
            diagnostics.append(
                ExportCatalogDiagnostic(
                    manifest_path=str(manifest_path),
                    code="invalid_publication",
                    message=str(exc),
                )
            )
            continue

        table_names = _table_names(payload)
        if set(inventory) != set(table_names):
            diagnostics.append(
                ExportCatalogDiagnostic(
                    manifest_path=str(manifest_path),
                    code="invalid_publication",
                    message=(
                        "Publication inventory table set does not match the "
                        "manifest's logical table declarations."
                    ),
                )
            )
            continue
        unknown_tables = tuple(sorted(set(table_names) - set(TABLE_CONTRACTS)))
        if unknown_tables:
            diagnostics.append(
                ExportCatalogDiagnostic(
                    manifest_path=str(manifest_path),
                    code="unknown_v2_table_contract",
                    message=f"Manifest declares unknown V2 tables: {list(unknown_tables)}",
                )
            )
            continue
        declared_contracts = payload.get("table_contracts")
        missing_contracts = (
            tuple(table_names)
            if not isinstance(declared_contracts, Mapping)
            else tuple(table for table in table_names if table not in declared_contracts)
        )
        if missing_contracts:
            diagnostics.append(
                ExportCatalogDiagnostic(
                    manifest_path=str(manifest_path),
                    code="missing_table_contract_snapshot",
                    message=f"Manifest lacks V2 contract snapshots for: {list(missing_contracts)}",
                )
            )
            continue
        mismatched_contracts = tuple(
            table
            for table in table_names
            if declared_contracts.get(table) != TABLE_CONTRACTS[table].to_dict()
        )
        if mismatched_contracts:
            diagnostics.append(
                ExportCatalogDiagnostic(
                    manifest_path=str(manifest_path),
                    code="mismatched_table_contract_snapshot",
                    message=(
                        "Manifest contract snapshots do not match the installed V2 contracts for: "
                        f"{list(mismatched_contracts)}"
                    ),
                )
            )
            continue

        missing_parts = 0
        unsafe_part: str | None = None
        columns_by_table: dict[str, tuple[str, ...]] = {}
        first_part_by_table: dict[str, Path] = {}
        for table_name in table_names:
            if Path(table_name).name != table_name or table_name in {"", ".", ".."}:
                unsafe_part = table_name
                break
            try:
                selected_parts = manifest_selected_part_files_from_payload(
                    root,
                    payload,
                    table_name,
                )
            except ValueError as exc:
                unsafe_part = str(exc)
                break
            for resolved_part in selected_parts:
                if not _is_within(resolved_part, root):
                    unsafe_part = str(resolved_part)
                    break
                if not resolved_part.is_file():
                    missing_parts += 1
                else:
                    first_part_by_table.setdefault(table_name, resolved_part)
            if unsafe_part is not None:
                break
        if unsafe_part is not None:
            diagnostics.append(
                ExportCatalogDiagnostic(
                    manifest_path=str(manifest_path),
                    code="part_file_outside_root",
                    message=f"Referenced part resolves outside the authorized root: {unsafe_part}",
                )
            )
            continue

        invalid_part_contract: tuple[str, str] | None = None
        if first_part_by_table:
            import polars as pl

            for table_name, part_path in first_part_by_table.items():
                try:
                    schema = pl.read_parquet_schema(part_path)
                    metadata = pl.read_parquet_metadata(part_path)
                    schema_id = metadata.get("palette.export_schema_id", "")
                    schema_version = metadata.get("palette.export_schema_version", "")
                    footer_contract = json.loads(
                        metadata.get("palette.table_contract", "null")
                    )
                except (
                    OSError,
                    ValueError,
                    UnicodeDecodeError,
                    json.JSONDecodeError,
                    pl.exceptions.PolarsError,
                ) as exc:
                    invalid_part_contract = (table_name, f"unreadable Parquet metadata: {exc}")
                    break
                if schema_id != EXPORT_SCHEMA_ID or schema_version != str(EXPORT_SCHEMA_VERSION):
                    invalid_part_contract = (
                        table_name,
                        f"footer declares {schema_id!r} version {schema_version!r}",
                    )
                    break
                if footer_contract != TABLE_CONTRACTS[table_name].to_dict():
                    invalid_part_contract = (table_name, "footer table contract does not match V2")
                    break
                columns_by_table[table_name] = tuple(schema.names())
        if invalid_part_contract is not None:
            table_name, reason = invalid_part_contract
            diagnostics.append(
                ExportCatalogDiagnostic(
                    manifest_path=str(manifest_path),
                    code="invalid_parquet_contract_metadata",
                    message=f"{table_name}: {reason}",
                )
            )
            continue
        capability_statuses = resolve_capabilities(columns_by_table)
        capabilities = tuple(
            status.capability_id for status in capability_statuses if status.available
        )

        collection = payload.get("collection_manifest")
        if not isinstance(collection, Mapping):
            collection = {}
        export_diagnostics = payload.get("diagnostics")
        entries.append(
            ExportCatalogEntry(
                export_run_id=payload_run_id,
                manifest_path=str(resolved_manifest),
                created_at_utc=(
                    str(payload["created_at_utc"])
                    if payload.get("created_at_utc") is not None
                    else None
                ),
                source_recording_count=_safe_int(payload.get("source_recording_count")),
                collection_id=(
                    str(collection["collection_id"])
                    if collection.get("collection_id") is not None
                    else None
                ),
                collection_name=(
                    str(collection["collection_name"])
                    if collection.get("collection_name") is not None
                    else None
                ),
                table_names=table_names,
                schema_id=EXPORT_SCHEMA_ID,
                schema_version=EXPORT_SCHEMA_VERSION,
                capabilities=capabilities,
                export_diagnostics_count=(
                    len(export_diagnostics) if isinstance(export_diagnostics, list) else 0
                ),
                missing_part_count=missing_parts,
            )
        )

    entries.sort(
        key=lambda entry: (_timestamp_sort_value(entry.created_at_utc), entry.export_run_id),
        reverse=True,
    )
    return ExportCatalog(root, tuple(entries), tuple(diagnostics))


def select_export_run_id(catalog: ExportCatalog, requested: str = "latest") -> str:
    """Resolve an initial selection strictly from discovered catalog entries."""

    if not catalog.entries:
        raise FileNotFoundError(f"No selectable analytics exports found under {catalog.export_root}")
    requested = str(requested).strip()
    if requested in {"", "auto", "latest"}:
        ready = next((entry for entry in catalog.entries if entry.ready), None)
        if ready is None:
            raise FileNotFoundError(
                f"No ready analytics exports found under {catalog.export_root}"
            )
        return ready.export_run_id
    selected = catalog.entry(requested)
    if selected is None:
        raise ValueError(
            f"Export run {requested!r} is not a selectable export under {catalog.export_root}"
        )
    if not selected.ready:
        raise ValueError(
            f"Export run {requested!r} is incomplete under {catalog.export_root}"
        )
    return requested


__all__ = [
    "ExportCatalog",
    "ExportCatalogDiagnostic",
    "ExportCatalogEntry",
    "discover_export_catalog",
    "select_export_run_id",
]
