from __future__ import annotations

import json
from pathlib import Path

import pytest

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


TABLE = "example_behavior_summary"


def _write_export_manifest(
    root: Path,
    export_run_id: str,
    *,
    created_at_utc: str,
    declared_part: str | None = None,
    write_part: bool = True,
    payload_run_id: str | None = None,
    source_export_run_id: str | None = None,
) -> Path:
    part = root / "v1" / TABLE / f"export_run_id={export_run_id}" / "part-00000.parquet"
    if write_part:
        part.parent.mkdir(parents=True, exist_ok=True)
        part.write_bytes(b"catalog discovery does not read parquet contents")
    manifest_dir = root / "v1" / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "export_run_id": payload_run_id or export_run_id,
        "created_at_utc": created_at_utc,
        "source_recording_count": 2,
        "tables_requested": [TABLE],
        "row_counts_by_table": {TABLE: 4},
        "part_files_by_table": {TABLE: [declared_part or str(part)]},
        "diagnostics": [],
        "collection_manifest": {
            "collection_id": "protocol_alpha",
            "collection_name": "Protocol Alpha",
        },
    }
    if source_export_run_id is not None:
        payload["source_export_run_id"] = source_export_run_id
    manifest = manifest_dir / f"export_run_id={export_run_id}.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return manifest


def test_catalog_discovers_base_exports_and_selects_newest_created_export(tmp_path: Path) -> None:
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
    assert select_export_run_id(catalog, "export_z_name_but_older") == "export_z_name_but_older"
    with pytest.raises(ValueError, match="not a selectable export"):
        select_export_run_id(catalog, "unknown")


def test_catalog_rebases_historical_absolute_part_paths_to_mounted_root(tmp_path: Path) -> None:
    root = tmp_path / "mounted" / "analytics"
    _write_export_manifest(
        root,
        "portable_export",
        created_at_utc="2025-02-01T00:00:00+00:00",
        declared_part="/old/workstation/exports/v1/example_behavior_summary/"
        "export_run_id=portable_export/part-00000.parquet",
    )

    catalog = discover_export_catalog(root)

    assert [entry.export_run_id for entry in catalog.entries] == ["portable_export"]
    assert catalog.entries[0].ready is True


def test_catalog_reports_missing_in_root_parts_without_hiding_export(tmp_path: Path) -> None:
    root = tmp_path / "analytics"
    _write_export_manifest(
        root,
        "incomplete_export",
        created_at_utc="2025-02-01T00:00:00+00:00",
        write_part=False,
    )

    catalog = discover_export_catalog(root)

    assert catalog.entries[0].export_run_id == "incomplete_export"
    assert catalog.entries[0].ready is False
    assert catalog.entries[0].missing_part_count == 1


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


def test_catalog_and_query_reject_part_symlink_that_escapes_root(tmp_path: Path) -> None:
    root = tmp_path / "analytics"
    manifest = _write_export_manifest(
        root,
        "unsafe_export",
        created_at_utc="2025-02-01T00:00:00+00:00",
        write_part=False,
    )
    outside = tmp_path / "outside.parquet"
    outside.write_bytes(b"outside")
    part = root / "v1" / TABLE / "export_run_id=unsafe_export" / "part-00000.parquet"
    part.parent.mkdir(parents=True, exist_ok=True)
    part.symlink_to(outside)

    catalog = discover_export_catalog(root)

    assert catalog.entries == ()
    assert [diagnostic.code for diagnostic in catalog.diagnostics] == [
        "part_file_outside_root"
    ]
    context = ViewerContext(export_root=root.resolve(), export_run_id="unsafe_export")
    with pytest.raises(PermissionError, match="Parquet part resolves outside"):
        parquet_files(context, TABLE)
    assert manifest.is_file()


def test_catalog_rejects_manifest_directory_symlink_that_escapes_root(tmp_path: Path) -> None:
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


def test_context_and_table_helpers_reject_path_component_traversal(tmp_path: Path) -> None:
    root = tmp_path / "analytics"
    _write_export_manifest(
        root,
        "safe_export",
        created_at_utc="2025-02-01T00:00:00+00:00",
    )

    with pytest.raises(ValueError, match="not a selectable export"):
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


def test_explicit_statistics_manifest_cannot_escape_root_by_symlink(tmp_path: Path) -> None:
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
                "output_tables": ["goodcopbadcop_group_statistical_summary"],
            }
        ),
        encoding="utf-8",
    )
    unsafe_manifest = (
        root / "v1" / "manifests" / "export_run_id=unsafe_statistics.json"
    )
    unsafe_manifest.symlink_to(outside_manifest)
    context = build_context(
        export_root=root,
        export_run_id="safe_export",
        stats_run_id="unsafe_statistics",
    )

    with pytest.raises(PermissionError, match="manifest resolves outside"):
        resolve_statistics_run_id(context)
