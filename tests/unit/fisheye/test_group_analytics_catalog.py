from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fisheye.analytics_exports.arrow_contracts import (
    ARROW_TABLE_CONTRACTS,
    arrow_contract_envelope,
    exact_arrow_schema,
)
from fisheye.analytics_exports.capabilities import resolve_capabilities
from fisheye.analytics_exports.contracts import (
    EXPORT_SCHEMA_ID,
    EXPORT_SCHEMA_VERSION,
    POSITION_OCCUPANCY_HISTOGRAM_TABLE,
    RECORDING_SUMMARY_TABLE,
    contract_snapshot,
)
from fisheye.analytics_exports.publication import sha256_file
from fisheye.group_analytics_viewer.catalog import (
    discover_export_catalog,
    select_export_run_id,
)
from fisheye.group_analytics_viewer.query import (
    ViewerContext,
    build_context,
    parquet_files,
    resolve_statistics_run_id,
    table_dir,
)


TABLE = RECORDING_SUMMARY_TABLE


def _write_export_manifest(
    root: Path,
    export_run_id: str,
    *,
    created_at_utc: str,
    declared_part: str | None = None,
    write_part: bool = True,
    payload_run_id: str | None = None,
    source_export_run_id: str | None = None,
    publication_v1: bool = True,
) -> Path:
    generation_path = (
        Path("v1")
        / ".generations"
        / f"export_run_id={export_run_id}"
        / "generation=test"
    )
    part = root / generation_path / "tables" / TABLE / "part-00000.parquet"
    if write_part:
        part.parent.mkdir(parents=True, exist_ok=True)
        footer_metadata = {
            b"palette.export_schema_id": EXPORT_SCHEMA_ID.encode("utf-8"),
            b"palette.export_schema_version": str(EXPORT_SCHEMA_VERSION).encode(
                "utf-8"
            ),
            b"palette.table_contract": json.dumps(
                contract_snapshot([TABLE])[TABLE],
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8"),
        }
        schema = exact_arrow_schema(TABLE, metadata=footer_metadata)
        row = {field.name: None for field in schema}
        row.update(
            {
                "export_schema_version": EXPORT_SCHEMA_VERSION,
                "table_name": TABLE,
                "recording_id": "recording_1",
                "zarr_path": "/data/recording_1_analysis.zarr",
                "source_lineage_hash": "a" * 64,
                "stimulus_step_count": 0,
            }
        )
        table = pa.Table.from_pylist([row], schema=schema)
        pq.write_table(table, part)
    manifest_dir = root / "v1" / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "export_run_id": payload_run_id or export_run_id,
        "schema_id": EXPORT_SCHEMA_ID,
        "schema_version": EXPORT_SCHEMA_VERSION,
        "created_at_utc": created_at_utc,
        "source_recording_count": 2,
        "tables_requested": [TABLE],
        "table_contracts": contract_snapshot([TABLE]),
        "row_counts_by_table": {TABLE: 1},
        "part_files_by_table": {
            TABLE: [
                declared_part
                or (generation_path / "tables" / TABLE / part.name).as_posix()
            ]
        },
        "capabilities": [
            status.capability_id
            for status in resolve_capabilities(
                {
                    TABLE: tuple(
                        field.name for field in ARROW_TABLE_CONTRACTS[TABLE].fields
                    )
                }
            )
            if status.available
        ],
        "arrow_schema_contracts": arrow_contract_envelope([TABLE]),
        "diagnostics": [],
        "collection_manifest": {
            "collection_id": "protocol_alpha",
            "collection_name": "Protocol Alpha",
        },
    }
    if source_export_run_id is not None:
        payload["source_export_run_id"] = source_export_run_id
    if publication_v1:
        relative_part = (generation_path / "tables" / TABLE / part.name).as_posix()
        payload["part_files_by_table"] = {TABLE: [relative_part]}
        payload["publication"] = {
            "schema_id": "palette.analytics_export.publication",
            "schema_version": 1,
            "state": "complete",
            "generation_id": "test",
            "generation_path": generation_path.as_posix(),
            "parts_by_table": {
                TABLE: [
                    {
                        "path": relative_part,
                        "sha256": sha256_file(part) if write_part else "0" * 64,
                        "size_bytes": part.stat().st_size if write_part else 0,
                        "row_count": 1,
                    }
                ]
            },
        }
    manifest = manifest_dir / f"export_run_id={export_run_id}.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return manifest


def _write_empty_registry_identity_export(root: Path, export_run_id: str) -> Path:
    table_name = POSITION_OCCUPANCY_HISTOGRAM_TABLE
    generation_path = (
        Path("v1")
        / ".generations"
        / f"export_run_id={export_run_id}"
        / "generation=test"
    )
    part = root / generation_path / "tables" / table_name / "part-00000.parquet"
    part.parent.mkdir(parents=True, exist_ok=True)
    footer_metadata = {
        b"palette.export_schema_id": EXPORT_SCHEMA_ID.encode("utf-8"),
        b"palette.export_schema_version": str(EXPORT_SCHEMA_VERSION).encode("ascii"),
        b"palette.table_contract": json.dumps(
            contract_snapshot([table_name])[table_name],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8"),
    }
    schema = exact_arrow_schema(table_name, metadata=footer_metadata)
    pq.write_table(pa.Table.from_pylist([], schema=schema), part)
    relative_part = part.relative_to(root).as_posix()
    source_path = "/data/recording_1_analysis.zarr"
    payload = {
        "export_run_id": export_run_id,
        "schema_id": EXPORT_SCHEMA_ID,
        "schema_version": EXPORT_SCHEMA_VERSION,
        "created_at_utc": "2026-08-10T00:00:00+00:00",
        "source_recording_count": 1,
        "source_zarrs": [source_path],
        "tables_requested": [table_name],
        "output_tables": [table_name],
        "table_contracts": contract_snapshot([table_name]),
        "arrow_schema_contracts": arrow_contract_envelope([table_name]),
        "row_counts_by_table": {table_name: 0},
        "part_files_by_table": {table_name: [relative_part]},
        "capabilities": [
            status.capability_id
            for status in resolve_capabilities(
                {table_name: tuple(field.name for field in schema)}
            )
            if status.available
        ],
        "diagnostics": [],
        "registry_identity": {
            "schema_id": "palette.analytics_export.registry_identity",
            "schema_version": 1,
            "acquisition_batch_id_source": "dataset_context_current.recording_started_utc",
            "subject_id_source": (
                "coalesce(dataset_context_current.subject_id,"
                "dataset_context_current.legacy_fish_id)"
            ),
            "sources": {
                source_path: {
                    "recording_id": "recording_1",
                    "acquisition_batch_id": "session_1",
                    "subject_id": "subject_1",
                }
            },
        },
        "publication": {
            "schema_id": "palette.analytics_export.publication",
            "schema_version": 1,
            "state": "complete",
            "generation_id": "test",
            "generation_path": generation_path.as_posix(),
            "parts_by_table": {
                table_name: [
                    {
                        "path": relative_part,
                        "sha256": sha256_file(part),
                        "size_bytes": part.stat().st_size,
                        "row_count": 0,
                    }
                ]
            },
        },
    }
    manifest_dir = root / "v1" / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = manifest_dir / f"export_run_id={export_run_id}.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return manifest


def test_catalog_discovers_base_exports_and_selects_newest_created_export(
    tmp_path: Path,
) -> None:
    root = tmp_path / "analytics"
    _write_export_manifest(
        root,
        "export_z_name_but_older",
        created_at_utc="2025-01-01T00:00:00+00:00",
    )
    _write_export_manifest(
        root,
        "export_a_name_but_newer",
        created_at_utc="2025-02-01T00:00:00+00:00",
    )
    _write_export_manifest(
        root,
        "statistics_only",
        created_at_utc="2025-03-01T00:00:00+00:00",
        source_export_run_id="export_a_name_but_newer",
    )

    catalog = discover_export_catalog(root)

    assert [entry.export_run_id for entry in catalog.entries] == [
        "export_a_name_but_newer",
        "export_z_name_but_older",
    ]
    assert catalog.diagnostics == ()
    assert catalog.entries[0].collection_id == "protocol_alpha"
    assert catalog.entries[0].table_names == (TABLE,)
    assert catalog.entries[0].ready is True
    assert "2 recordings" in catalog.entries[0].label
    assert select_export_run_id(catalog, "latest") == "export_a_name_but_newer"
    assert (
        select_export_run_id(catalog, "export_z_name_but_older")
        == "export_z_name_but_older"
    )
    with pytest.raises(ValueError, match="not a selectable export"):
        select_export_run_id(catalog, "unknown")


def test_catalog_rebases_historical_absolute_part_paths_to_mounted_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "mounted" / "analytics"
    _write_export_manifest(
        root,
        "portable_export",
        created_at_utc="2025-02-01T00:00:00+00:00",
        declared_part="/old/workstation/exports/v1/recording_summary/"
        "export_run_id=portable_export/part-00000.parquet",
        publication_v1=False,
    )

    catalog = discover_export_catalog(root)

    assert catalog.entries == ()
    assert catalog.diagnostics[0].code == "legacy_publication_not_selectable"


def test_catalog_rejects_missing_in_root_parts(tmp_path: Path) -> None:
    root = tmp_path / "analytics"
    _write_export_manifest(
        root,
        "incomplete_export",
        created_at_utc="2025-02-01T00:00:00+00:00",
        write_part=False,
    )

    catalog = discover_export_catalog(root)

    assert catalog.entries == ()
    assert catalog.diagnostics[0].code == "invalid_export_payload"
    with pytest.raises(FileNotFoundError, match="No selectable analytics exports"):
        select_export_run_id(catalog, "latest")


def test_catalog_rejects_mismatched_manifest_identity(tmp_path: Path) -> None:
    root = tmp_path / "analytics"
    _write_export_manifest(
        root,
        "filename_id",
        payload_run_id="payload_id",
        created_at_utc="2025-02-01T00:00:00+00:00",
    )

    catalog = discover_export_catalog(root)

    assert catalog.entries == ()
    assert [diagnostic.code for diagnostic in catalog.diagnostics] == [
        "export_run_id_mismatch"
    ]


def test_catalog_rejects_version_1_export_with_reexport_message(tmp_path: Path) -> None:
    root = tmp_path / "analytics"
    manifest = _write_export_manifest(
        root,
        "legacy_export",
        created_at_utc="2025-02-01T00:00:00+00:00",
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload.pop("schema_id")
    payload["schema_version"] = 1
    payload.pop("table_contracts")
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    catalog = discover_export_catalog(root)

    assert catalog.entries == ()
    assert catalog.diagnostics[0].code == "unsupported_export_schema"
    assert "re-export" in catalog.diagnostics[0].message


def test_catalog_rejects_v2_manifest_with_mismatched_contract_snapshot(
    tmp_path: Path,
) -> None:
    root = tmp_path / "analytics"
    manifest = _write_export_manifest(
        root,
        "mismatched_contract",
        created_at_utc="2025-02-01T00:00:00+00:00",
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["table_contracts"][TABLE]["grain"] = "incorrect_grain"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    catalog = discover_export_catalog(root)

    assert catalog.entries == ()
    assert catalog.diagnostics[0].code == "mismatched_table_contract_snapshot"


def test_catalog_discovery_rejects_payload_digest_tampering(tmp_path: Path) -> None:
    root = tmp_path / "analytics"
    _write_export_manifest(
        root,
        "metadata_only",
        created_at_utc="2025-02-01T00:00:00+00:00",
    )
    part = next((root / "v1" / ".generations").rglob("*.parquet"))
    table = pq.read_table(part)
    recording_index = table.schema.get_field_index("recording_id")
    table = table.set_column(
        recording_index,
        table.schema.field(recording_index),
        pa.array(["recording_2"]),
    )
    pq.write_table(table, part)

    catalog = discover_export_catalog(root)

    assert catalog.entries == ()
    assert catalog.diagnostics[0].code == "invalid_export_payload"
    assert "digest mismatch" in catalog.diagnostics[0].message


@pytest.mark.parametrize("mutation", ["missing", "tampered"])
def test_catalog_rejects_missing_or_tampered_registry_identity(
    tmp_path: Path,
    mutation: str,
) -> None:
    root = tmp_path / "analytics"
    manifest = _write_empty_registry_identity_export(root, "identity_export")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if mutation == "missing":
        payload.pop("registry_identity")
    else:
        payload["registry_identity"]["acquisition_batch_id_source"] = (
            "manifest.session_name"
        )
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    catalog = discover_export_catalog(root)

    assert catalog.entries == ()
    assert catalog.diagnostics[0].code == "invalid_export_payload"
    assert "registry identity" in catalog.diagnostics[0].message


def test_catalog_and_query_reject_part_symlink_that_escapes_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "analytics"
    manifest = _write_export_manifest(
        root,
        "unsafe_export",
        created_at_utc="2025-02-01T00:00:00+00:00",
        write_part=False,
    )
    outside = tmp_path / "outside.parquet"
    outside.write_bytes(b"outside")
    part = (
        root
        / "v1"
        / ".generations"
        / "export_run_id=unsafe_export"
        / "generation=test"
        / "tables"
        / TABLE
        / "part-00000.parquet"
    )
    part.parent.mkdir(parents=True, exist_ok=True)
    part.symlink_to(outside)

    catalog = discover_export_catalog(root)

    assert catalog.entries == ()
    assert [diagnostic.code for diagnostic in catalog.diagnostics] == [
        "part_file_outside_root"
    ]
    context = ViewerContext(export_root=root.resolve(), export_run_id="unsafe_export")
    with pytest.raises(ValueError, match="symbolic-link alias"):
        parquet_files(context, TABLE)
    assert manifest.is_file()


def test_catalog_rejects_manifest_directory_symlink_that_escapes_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "analytics"
    outside = tmp_path / "outside-manifests"
    outside.mkdir()
    (root / "v1").mkdir(parents=True)
    (root / "v1" / "manifests").symlink_to(outside, target_is_directory=True)

    catalog = discover_export_catalog(root)

    assert catalog.entries == ()
    assert [diagnostic.code for diagnostic in catalog.diagnostics] == [
        "manifest_directory_outside_root"
    ]


def test_context_and_table_helpers_reject_path_component_traversal(
    tmp_path: Path,
) -> None:
    root = tmp_path / "analytics"
    _write_export_manifest(
        root,
        "safe_export",
        created_at_utc="2025-02-01T00:00:00+00:00",
    )

    with pytest.raises(ValueError, match="Invalid export run ID"):
        build_context(export_root=root, export_run_id="../outside")
    with pytest.raises(ValueError, match="Invalid statistics run ID"):
        build_context(
            export_root=root,
            export_run_id="safe_export",
            stats_run_id="../outside",
        )

    context = ViewerContext(export_root=root.resolve(), export_run_id="safe_export")
    with pytest.raises(ValueError, match="Invalid table name"):
        table_dir(context, "../outside")
    with pytest.raises(ValueError, match="Invalid export run ID"):
        table_dir(context, TABLE, export_run_id="../outside")


def test_explicit_statistics_manifest_cannot_escape_root_by_symlink(
    tmp_path: Path,
) -> None:
    root = tmp_path / "analytics"
    _write_export_manifest(
        root,
        "safe_export",
        created_at_utc="2025-02-01T00:00:00+00:00",
    )
    outside_manifest = tmp_path / "outside-statistics.json"
    outside_manifest.write_text(
        json.dumps(
            {
                "export_run_id": "unsafe_statistics",
                "source_export_run_id": "safe_export",
                "output_tables": ["group_statistical_summary"],
            }
        ),
        encoding="utf-8",
    )
    unsafe_manifest = root / "v1" / "manifests" / "export_run_id=unsafe_statistics.json"
    unsafe_manifest.symlink_to(outside_manifest)
    context = build_context(
        export_root=root,
        export_run_id="safe_export",
        stats_run_id="unsafe_statistics",
    )

    with pytest.raises(PermissionError, match="manifest resolves outside"):
        resolve_statistics_run_id(context)
