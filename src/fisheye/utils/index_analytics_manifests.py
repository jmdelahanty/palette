"""Index analytics collection/export manifests in the Palette registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from fisheye.analytics_exports.publication import (
    has_exclusive_inventory,
    manifest_selected_part_files_from_payload,
)
from fisheye.analytics_exports.validation import validate_export_run
from fisheye.registry.db import Registry, RegistryPaths
from fisheye.utils.virtual_collection_manifest import (
    compute_manifest_sha256,
    load_manifest,
    verify_manifest_sha256,
)


def _load_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: JSON root must be an object")
    return payload


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _collection_counts(manifest: Mapping[str, Any]) -> tuple[int, int]:
    records = manifest.get("records")
    if not isinstance(records, list):
        return 0, 0
    included = 0
    for record in records:
        if not isinstance(record, Mapping):
            continue
        status = record.get("status")
        if isinstance(status, Mapping) and status.get("included") is True:
            included += 1
    return len(records), included


def index_collection_manifest(
    registry: Registry,
    manifest_path: Path,
    *,
    status: str = "active",
) -> tuple[str, str]:
    """Index one immutable collection manifest and return id/hash."""

    manifest_path = manifest_path.expanduser().resolve()
    manifest = load_manifest(manifest_path)
    if not verify_manifest_sha256(manifest):
        expected = manifest.get("manifest_sha256")
        actual = compute_manifest_sha256(manifest)
        raise ValueError(
            f"{manifest_path}: manifest_sha256 mismatch "
            f"(stored={expected!r}, computed={actual!r})"
        )
    record_count, included_count = _collection_counts(manifest)
    registry.upsert_analytics_collection(
        collection_id=str(manifest["collection_id"]),
        manifest_sha256=str(manifest["manifest_sha256"]),
        collection_name=(
            str(manifest["collection_name"])
            if manifest.get("collection_name") is not None
            else None
        ),
        manifest_path=manifest_path,
        schema_id=str(manifest.get("schema_id")) if manifest.get("schema_id") is not None else None,
        schema_version=_int_or_none(manifest.get("schema_version")),
        record_count=record_count,
        included_record_count=included_count,
        created_utc=str(manifest.get("created_utc")) if manifest.get("created_utc") is not None else None,
        status=status,
        metadata={
            "manifest_canonicalization": manifest.get("manifest_canonicalization"),
            "purpose": manifest.get("purpose"),
            "query": manifest.get("query"),
            "export_profiles": manifest.get("export_profiles"),
        },
    )
    return str(manifest["collection_id"]), str(manifest["manifest_sha256"])


def _export_output_root(manifest: Mapping[str, Any], manifest_path: Path) -> Path | None:
    explicit = manifest.get("output_root")
    if isinstance(explicit, str) and explicit:
        return Path(explicit)
    # Published manifests are authoritative at <root>/v1/manifests/<file>.
    # Prefer that stable location before interpreting legacy part paths.
    try:
        if manifest_path.parent.name == "manifests" and manifest_path.parent.parent.name == "v1":
            return manifest_path.parent.parent.parent
    except Exception:
        return None
    part_files = manifest.get("part_files_by_table")
    if isinstance(part_files, Mapping):
        for files in part_files.values():
            if isinstance(files, list) and files:
                try:
                    path = Path(str(files[0]))
                    # .../<output_root>/v1/<table>/export_run_id=<id>/part.parquet
                    return path.parents[3]
                except Exception:
                    return None
    return None


def index_export_manifest(
    registry: Registry,
    manifest_path: Path,
    *,
    status: str = "active",
    auto_index_collection: bool = True,
    allow_legacy_unvalidated: bool = False,
) -> str:
    """Index one immutable analytics export manifest and return export run id."""

    manifest_path = manifest_path.expanduser().resolve()
    manifest = _load_json_object(manifest_path)
    output_root = _export_output_root(manifest, manifest_path)
    if output_root is None:
        raise ValueError(f"Cannot resolve analytics export root from {manifest_path}")
    output_root = output_root.expanduser().resolve()
    export_run_id = str(manifest["export_run_id"])
    is_strict_publication = has_exclusive_inventory(manifest)
    if not is_strict_publication and not allow_legacy_unvalidated:
        raise ValueError(
            "Analytics export lacks publication-v1; use the explicit legacy "
            "compatibility option to index it as non-active history"
        )
    if is_strict_publication:
        validate_export_run(output_root, export_run_id)
    effective_status = status if is_strict_publication else "legacy_compatibility"
    raw_part_tables = manifest.get("part_files_by_table")
    if not isinstance(raw_part_tables, Mapping):
        raise ValueError("Analytics export part_files_by_table is missing or invalid")
    indexed_parts = {
        str(table_name): [
            str(path.resolve())
            for path in manifest_selected_part_files_from_payload(
                output_root,
                manifest,
                str(table_name),
                allow_legacy_layout=allow_legacy_unvalidated,
            )
        ]
        for table_name in raw_part_tables
    }
    collection = manifest.get("collection_manifest")
    collection_id = None
    collection_manifest_sha256 = None
    if isinstance(collection, Mapping):
        collection_id = (
            str(collection.get("collection_id"))
            if collection.get("collection_id") is not None
            else None
        )
        collection_manifest_sha256 = (
            str(collection.get("manifest_sha256"))
            if collection.get("manifest_sha256") is not None
            else None
        )
        collection_path = collection.get("path")
        if auto_index_collection and isinstance(collection_path, str) and collection_path:
            path = Path(collection_path).expanduser()
            if path.exists():
                index_collection_manifest(registry, path, status="active")

    diagnostics = manifest.get("diagnostics")
    registry.upsert_analytics_export(
        export_run_id=export_run_id,
        collection_id=collection_id,
        collection_manifest_sha256=collection_manifest_sha256,
        export_manifest_path=manifest_path,
        output_root=output_root,
        schema_version=_int_or_none(manifest.get("schema_version")),
        tool=str(manifest.get("tool")) if manifest.get("tool") is not None else None,
        palette_git_commit=(
            str(manifest.get("palette_git_commit"))
            if manifest.get("palette_git_commit") is not None
            else None
        ),
        palette_git_dirty=(
            bool(manifest.get("palette_git_dirty"))
            if manifest.get("palette_git_dirty") is not None
            else None
        ),
        source_recording_count=_int_or_none(manifest.get("source_recording_count")),
        row_counts_by_table=(
            manifest.get("row_counts_by_table")
            if isinstance(manifest.get("row_counts_by_table"), Mapping)
            else {}
        ),
        part_files_by_table=indexed_parts,
        created_at_utc=(
            str(manifest.get("created_at_utc"))
            if manifest.get("created_at_utc") is not None
            else None
        ),
        diagnostics_count=len(diagnostics) if isinstance(diagnostics, list) else None,
        status=effective_status,
        metadata={
            "tables_requested": manifest.get("tables_requested"),
            "export_parameters": manifest.get("export_parameters"),
            "source_zarrs": manifest.get("source_zarrs"),
            "publication_policy": (
                "manifest_exclusive_v1"
                if is_strict_publication
                else "explicit_legacy_compatibility"
            ),
        },
    )
    return export_run_id


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Index Palette analytics collection/export manifests in the registry.",
    )
    parser.add_argument("--registry", type=Path, help="Palette registry SQLite path.")
    parser.add_argument(
        "--collection-manifest",
        action="append",
        type=Path,
        help="Collection manifest JSON path. May be repeated.",
    )
    parser.add_argument(
        "--export-manifest",
        action="append",
        type=Path,
        help="Export manifest JSON path. May be repeated.",
    )
    parser.add_argument("--status", default="active", help="Registry status to assign to indexed manifests.")
    parser.add_argument(
        "--allow-legacy-unvalidated",
        action="store_true",
        help=(
            "Explicitly index a historical non-publication-v1 export as "
            "legacy_compatibility rather than active."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Print JSON summary.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    registry_path = (
        args.registry.expanduser().resolve()
        if args.registry is not None
        else RegistryPaths.from_env(Path.cwd()).path
    )
    collection_paths = list(args.collection_manifest or [])
    export_paths = list(args.export_manifest or [])
    if not collection_paths and not export_paths:
        raise SystemExit("at least one --collection-manifest or --export-manifest is required")

    registry = Registry(registry_path)
    indexed_collections: list[tuple[str, str]] = []
    indexed_exports: list[str] = []
    try:
        for path in collection_paths:
            indexed_collections.append(
                index_collection_manifest(registry, path, status=str(args.status))
            )
        for path in export_paths:
            indexed_exports.append(
                index_export_manifest(
                    registry,
                    path,
                    status=str(args.status),
                    allow_legacy_unvalidated=bool(args.allow_legacy_unvalidated),
                )
            )
    finally:
        registry.close()

    payload = {
        "registry": str(registry_path),
        "collections_indexed": [
            {"collection_id": collection_id, "manifest_sha256": manifest_sha256}
            for collection_id, manifest_sha256 in indexed_collections
        ],
        "exports_indexed": indexed_exports,
    }
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        print(f"registry\t{registry_path}")
        for item in payload["collections_indexed"]:
            print(f"collection\t{item['collection_id']}\t{item['manifest_sha256']}")
        for export_run_id in indexed_exports:
            print(f"export\t{export_run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
