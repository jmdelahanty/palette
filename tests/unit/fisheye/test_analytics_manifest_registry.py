from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fisheye.analytics_exports.arrow_contracts import arrow_contract_envelope
from fisheye.analytics_exports.contracts import (
    EXPORT_SCHEMA_ID,
    EXPORT_SCHEMA_VERSION,
    RECORDING_SUMMARY_TABLE,
    SWIM_BOUT_METRICS_TABLE,
    TABLE_CONTRACTS,
    canonicalize_export_row,
    contract_snapshot,
)
from fisheye.analytics_exports.publication import sha256_file
from fisheye.registry.db import Registry
from fisheye.utils.analytics_export_resolution import resolve_latest_export_table
from fisheye.utils.check_analytics_exports import main as check_main
from fisheye.utils.index_analytics_manifests import (
    index_collection_manifest,
    index_export_manifest,
    main as index_main,
)
from fisheye.utils.query_analytics_exports import main as query_main
from fisheye.utils.resolve_analytics_export import main as resolve_main
from fisheye.utils.virtual_collection_manifest import with_manifest_sha256


def _write_collection_manifest(path: Path) -> dict:
    manifest = with_manifest_sha256(
        {
            "schema_id": "palette.virtual_collection_manifest",
            "schema_version": 1,
            "collection_id": "movement_bouts_test_v001",
            "collection_name": "Movement Bouts Test",
            "created_utc": "2026-05-08T00:00:00Z",
            "purpose": "cross_recording_analytics_export",
            "manifest_canonicalization": "json_sorted_keys_no_hash_fields_v1",
            "selection_policy": {
                "latest_allowed_during_selection": True,
                "latest_resolved_before_export": True,
                "production_requires_explicit_runs": True,
            },
            "query": {
                "registry_path": None,
                "registry_snapshot_sha256": None,
                "registry_snapshot_status": "not_registry_derived",
                "filters": {"explicit_zarr_paths": ["/tmp/a.zarr", "/tmp/b.zarr"]},
                "ordering": ["input_order"],
            },
            "export_profiles": [
                {
                    "profile_id": "movement_bouts",
                    "required_run_families": [
                        "track_kinematics_run",
                        "swim_bout_run",
                        "bout_kinematics_run",
                    ],
                    "optional_run_families": [],
                }
            ],
            "records": [
                {
                    "recording_id": "rec_a",
                    "dataset_id": "analysis_rec_a",
                    "artifact_kind": "analysis_zarr",
                    "locator_at_selection": {
                        "uri": "/tmp/a.zarr",
                        "storage_tier": "hot_nvme",
                        "last_verified_utc": "2026-05-08T00:00:00Z",
                    },
                    "recording_attrs": {"recording_id": "rec_a"},
                    "protocol": {},
                    "source_runs": {
                        "track_kinematics_run": {
                            "present": True,
                            "required": True,
                            "run_id": "tk",
                            "path": "analysis/track_kinematics_runs/offline/tk",
                            "selection": "explicit",
                            "fingerprint_status": "best_effort",
                        }
                    },
                    "status": {"included": True, "warnings": [], "exclusions": []},
                },
                {
                    "recording_id": "rec_b",
                    "dataset_id": "analysis_rec_b",
                    "artifact_kind": "analysis_zarr",
                    "locator_at_selection": {
                        "uri": "/tmp/b.zarr",
                        "storage_tier": "hot_nvme",
                        "last_verified_utc": "2026-05-08T00:00:00Z",
                    },
                    "recording_attrs": {"recording_id": "rec_b"},
                    "protocol": {},
                    "source_runs": {},
                    "status": {"included": False, "warnings": [], "exclusions": ["missing run"]},
                },
            ],
        }
    )
    path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _write_export_manifest(path: Path, collection_manifest: dict, collection_path: Path) -> dict:
    export_root = path.parent.parent.parent
    generation_path = (
        Path("v1")
        / ".generations"
        / "export_run_id=run_test"
        / "generation=test"
    )
    parts: dict[str, Path] = {}
    rows_by_table = {
        RECORDING_SUMMARY_TABLE: [
            canonicalize_export_row(
                RECORDING_SUMMARY_TABLE,
                {"recording_id": f"recording_{index}"},
            )
            for index in range(2)
        ],
        SWIM_BOUT_METRICS_TABLE: [
            canonicalize_export_row(
                SWIM_BOUT_METRICS_TABLE,
                {"recording_id": "recording_0", "bout_id": index},
            )
            for index in range(42)
        ],
    }
    for table_name, rows in rows_by_table.items():
        part = export_root / generation_path / "tables" / table_name / "part-00000.parquet"
        part.parent.mkdir(parents=True, exist_ok=True)
        arrow = pa.Table.from_pylist(rows)
        arrow = arrow.replace_schema_metadata(
            {
                b"palette.export_schema_id": EXPORT_SCHEMA_ID.encode(),
                b"palette.export_schema_version": str(EXPORT_SCHEMA_VERSION).encode(),
                b"palette.table_contract": json.dumps(
                    TABLE_CONTRACTS[table_name].to_dict(),
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode(),
                b"palette.arrow_schema_mode": b"inferred_v2_compatibility",
            }
        )
        pq.write_table(arrow, part)
        parts[table_name] = part
    manifest = {
        "export_run_id": "run_test",
        "created_at_utc": "2026-05-08T00:10:00+00:00",
        "schema_id": EXPORT_SCHEMA_ID,
        "schema_version": EXPORT_SCHEMA_VERSION,
        "tool": "fisheye.utils.export_cross_recording_analytics",
        "palette_git_commit": "abc123",
        "palette_git_dirty": False,
        "source_recording_count": 2,
        "source_zarrs": ["/tmp/a.zarr", "/tmp/b.zarr"],
        "tables_requested": [RECORDING_SUMMARY_TABLE, SWIM_BOUT_METRICS_TABLE],
        "table_contracts": contract_snapshot(
            [RECORDING_SUMMARY_TABLE, SWIM_BOUT_METRICS_TABLE]
        ),
        "arrow_schema_contracts": arrow_contract_envelope(
            [RECORDING_SUMMARY_TABLE, SWIM_BOUT_METRICS_TABLE]
        ),
        "capabilities": [],
        "row_counts_by_table": {RECORDING_SUMMARY_TABLE: 2, SWIM_BOUT_METRICS_TABLE: 42},
        "part_files_by_table": {
            table_name: [
                (generation_path / "tables" / table_name / part.name).as_posix()
            ]
            for table_name, part in parts.items()
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
                        "path": (
                            generation_path / "tables" / table_name / part.name
                        ).as_posix(),
                        "sha256": sha256_file(part),
                        "size_bytes": part.stat().st_size,
                        "row_count": len(rows_by_table[table_name]),
                    }
                ]
                for table_name, part in parts.items()
            },
        },
        "diagnostics": [],
        "collection_manifest": {
            "path": str(collection_path),
            "collection_id": collection_manifest["collection_id"],
            "collection_name": collection_manifest["collection_name"],
            "manifest_sha256": collection_manifest["manifest_sha256"],
            "schema_id": collection_manifest["schema_id"],
            "schema_version": collection_manifest["schema_version"],
            "record_count": 2,
            "included_record_count": 1,
        },
        "export_parameters": {"jobs": 4},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _touch_indexed_parts(export_manifest: dict) -> None:
    part_files = export_manifest["part_files_by_table"]
    for files in part_files.values():
        for filename in files:
            path = Path(filename)
            if not path.is_absolute():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"PAR1")


def test_legacy_export_requires_explicit_non_active_index_policy(tmp_path: Path) -> None:
    collection_path = tmp_path / "collection.manifest.json"
    collection = _write_collection_manifest(collection_path)
    export_path = tmp_path / "exports" / "v1" / "manifests" / "export_run_id=run_test.json"
    payload = _write_export_manifest(export_path, collection, collection_path)
    payload.pop("publication")
    export_path.write_text(json.dumps(payload), encoding="utf-8")
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        with pytest.raises(ValueError, match="explicit legacy compatibility"):
            index_export_manifest(registry, export_path)
        assert index_export_manifest(
            registry,
            export_path,
            allow_legacy_unvalidated=True,
        ) == "run_test"
        row = registry.conn.execute(
            "SELECT status FROM analytics_exports WHERE export_run_id = ?",
            ("run_test",),
        ).fetchone()
        assert row["status"] == "legacy_compatibility"
    finally:
        registry.close()


def test_index_collection_and_export_manifest_tables(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    collection_path = tmp_path / "manifests" / "collections" / "collection.manifest.json"
    collection_path.parent.mkdir(parents=True)
    collection = _write_collection_manifest(collection_path)
    export_path = tmp_path / "exports" / "v1" / "manifests" / "export_run_id=run_test.json"
    _write_export_manifest(export_path, collection, collection_path)

    try:
        assert index_collection_manifest(registry, collection_path) == (
            "movement_bouts_test_v001",
            collection["manifest_sha256"],
        )
        assert index_export_manifest(registry, export_path) == "run_test"

        collection_row = registry.conn.execute(
            "SELECT * FROM analytics_collections WHERE collection_id = ?;",
            ("movement_bouts_test_v001",),
        ).fetchone()
        assert collection_row["record_count"] == 2
        assert collection_row["included_record_count"] == 1

        export_row = registry.conn.execute(
            "SELECT * FROM analytics_export_overview WHERE export_run_id = 'run_test';"
        ).fetchone()
        assert export_row["collection_manifest_sha256"] == collection["manifest_sha256"]
        assert export_row["source_recording_count"] == 2
        assert json.loads(export_row["row_counts_json"])["swim_bout_metrics"] == 42

        table_rows = registry.conn.execute(
            "SELECT table_name, row_count, part_count FROM analytics_export_tables ORDER BY table_name;"
        ).fetchall()
        assert [(row["table_name"], row["row_count"], row["part_count"]) for row in table_rows] == [
            ("recording_summary", 2, 1),
            ("swim_bout_metrics", 42, 1),
        ]

        resolution = resolve_latest_export_table(
            registry_path=tmp_path / "registry.sqlite",
            collection_id="movement_bouts_test_v001",
            table_name="swim_bout_metrics",
        )
        assert resolution.export_run_id == "run_test"
        assert resolution.table_name == "swim_bout_metrics"
        assert resolution.row_count == 42
        assert resolution.part_count == 1
        assert resolution.table_path == (
            tmp_path
            / "exports"
            / "v1"
            / ".generations"
            / "export_run_id=run_test"
            / "generation=test"
            / "tables"
            / "swim_bout_metrics"
        )
    finally:
        registry.close()


def test_active_registry_resolution_rechecks_publication_v1_manifest(
    tmp_path: Path,
) -> None:
    collection_path = tmp_path / "collection.manifest.json"
    collection = _write_collection_manifest(collection_path)
    export_path = (
        tmp_path
        / "exports"
        / "v1"
        / "manifests"
        / "export_run_id=run_test.json"
    )
    payload = _write_export_manifest(export_path, collection, collection_path)
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        assert index_export_manifest(registry, export_path) == "run_test"
    finally:
        registry.close()

    payload.pop("publication")
    export_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(LookupError, match="publication envelope is missing"):
        resolve_latest_export_table(
            registry_path=tmp_path / "registry.sqlite",
            table_name=SWIM_BOUT_METRICS_TABLE,
            export_run_id="run_test",
        )


def test_index_and_query_analytics_manifests_cli(tmp_path: Path, capsys) -> None:
    registry_path = tmp_path / "registry.sqlite"
    collection_path = tmp_path / "collection.manifest.json"
    collection = _write_collection_manifest(collection_path)
    export_path = tmp_path / "exports" / "v1" / "manifests" / "export_run_id=run_test.json"
    _write_export_manifest(export_path, collection, collection_path)

    assert (
        index_main(
            [
                "--registry",
                str(registry_path),
                "--export-manifest",
                str(export_path),
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert (
        query_main(
            [
                "--registry",
                str(registry_path),
                "--collection-id",
                "movement_bouts_test_v001",
                "--table",
                "swim_bout_metrics",
                "--format",
                "json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert len(payload) == 1
    assert payload[0]["export_run_id"] == "run_test"
    assert payload[0]["collection_manifest_sha256"] == collection["manifest_sha256"]

    assert (
        query_main(
            [
                "--registry",
                str(registry_path),
                "--collection-id",
                "movement_bouts_test_v001",
                "--table",
                "swim_bout_metrics",
                "--latest",
                "--format",
                "path",
            ]
        )
        == 0
    )
    path_lines = capsys.readouterr().out.strip().splitlines()
    assert path_lines == [
        str(
            tmp_path
            / "exports"
            / "v1"
            / ".generations"
            / "export_run_id=run_test"
            / "generation=test"
            / "tables"
            / "swim_bout_metrics"
        )
    ]

    assert (
        resolve_main(
            [
                "--registry",
                str(registry_path),
                "--collection-id",
                "movement_bouts_test_v001",
                "--table",
                "swim_bout_metrics",
                "--format",
                "json",
            ]
        )
        == 0
    )
    resolved = json.loads(capsys.readouterr().out)
    assert resolved["export_run_id"] == "run_test"
    assert resolved["table_name"] == "swim_bout_metrics"
    assert resolved["row_count"] == 42
    assert resolved["part_count"] == 1

    assert (
        resolve_main(
            [
                "--registry",
                str(registry_path),
                "--collection-id",
                "movement_bouts_test_v001",
                "--table",
                "swim_bout_metrics",
                "--format",
                "path",
            ]
        )
        == 0
    )
    assert capsys.readouterr().out.strip() == path_lines[0]


def test_check_analytics_exports_inventory_and_file_validation(tmp_path: Path, capsys) -> None:
    registry_path = tmp_path / "registry.sqlite"
    collection_path = tmp_path / "collection.manifest.json"
    collection = _write_collection_manifest(collection_path)
    export_path = tmp_path / "exports" / "v1" / "manifests" / "export_run_id=run_test.json"
    export_manifest = _write_export_manifest(export_path, collection, collection_path)

    assert index_main(
        [
            "--registry",
            str(registry_path),
            "--export-manifest",
            str(export_path),
        ]
    ) == 0
    capsys.readouterr()

    assert check_main(
        [
            "--registry",
            str(registry_path),
            "--collection-id",
            "movement_bouts_test_v001",
            "--format",
            "json",
        ]
    ) == 0
    inventory = json.loads(capsys.readouterr().out)
    assert {row["table_name"] for row in inventory} == {"recording_summary", "swim_bout_metrics"}
    assert {row["check_status"] for row in inventory} == {"not_checked"}

    export_root = export_path.parent.parent.parent
    moved_parts: list[tuple[Path, Path]] = []
    for files in export_manifest["part_files_by_table"].values():
        for relative in files:
            part = export_root / relative
            hidden = part.with_suffix(".missing")
            part.rename(hidden)
            moved_parts.append((part, hidden))
    assert check_main(
        [
            "--registry",
            str(registry_path),
            "--collection-id",
            "movement_bouts_test_v001",
            "--check-files",
            "--format",
            "json",
        ]
    ) == 1
    missing = json.loads(capsys.readouterr().out)
    assert {row["check_status"] for row in missing} == {"missing_part_files"}

    for part, hidden in moved_parts:
        hidden.rename(part)
    first_part = (
        export_root
        / export_manifest["part_files_by_table"][RECORDING_SUMMARY_TABLE][0]
    )
    extra_part = first_part.parent / "unlisted.parquet"
    extra_part.write_bytes(first_part.read_bytes())
    assert check_main(
        [
            "--registry",
            str(registry_path),
            "--collection-id",
            "movement_bouts_test_v001",
            "--table",
            RECORDING_SUMMARY_TABLE,
            "--check-files",
            "--format",
            "json",
        ]
    ) == 1
    unlisted = json.loads(capsys.readouterr().out)
    assert {row["check_status"] for row in unlisted} == {"unlisted_part_files"}
    extra_part.unlink()
    assert check_main(
        [
            "--registry",
            str(registry_path),
            "--collection-id",
            "movement_bouts_test_v001",
            "--check-files",
            "--format",
            "json",
        ]
    ) == 0
    checked = json.loads(capsys.readouterr().out)
    assert {row["check_status"] for row in checked} == {"ok"}
    assert {row["missing_part_count"] for row in checked} == {0}
