from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from fisheye.registry.db import Registry
from fisheye.registry.prune_stale_datasets import (
    assert_dependent_table_map_covers_schema,
    connect_read_only,
    main,
)
from fisheye.registry.status_ledger import upsert_recording_step_status


def _seed_dataset(
    registry: Registry,
    *,
    dataset_id: str,
    zarr_path: Path,
    recording_id: str,
) -> None:
    registry.upsert_recording(recording_id=recording_id, session_uuid=recording_id)
    registry.upsert_dataset(
        dataset_id,
        session_uuid=dataset_id,
        zarr_path=zarr_path,
        recording_id=recording_id,
        artifact_kind="source_recording",
        zarr_use="analysis",
    )


def _seed_dependents(registry: Registry, dataset_id: str, *, parent_dataset_id: str) -> None:
    registry.upsert_provenance(
        dataset_id,
        provenance={},
        context={},
        protocol_name=None,
        protocol_hash=None,
    )
    registry.replace_detect_quality(
        dataset_id,
        [
            {
                "refined_run": "refined_v1",
                "refined_created_utc": "2026-07-06T00:00:00Z",
                "source_detect_run": "detect_v1",
                "detect_method": "traditional",
                "review_state": "approved",
                "review_intended_use": "training",
                "review_reviewer": "agent",
                "review_timestamp_utc": "2026-07-06T00:00:00Z",
                "review_resolved_group": "refined_detect_runs/refined_v1",
                "total_detections": 1,
                "real_detections": 1,
                "interpolated_detections": 0,
                "interpolated_detections_rate": 0.0,
            }
        ],
    )
    upsert_recording_step_status(
        registry,
        dataset_id=dataset_id,
        step_name="detect",
        status="ok",
        updated_utc="2026-07-06T00:00:00Z",
    )
    registry.conn.execute(
        """
        INSERT INTO dataset_lineage (
            child_dataset_id, parent_dataset_id, relationship_type, source_set_id,
            metadata_json, created_utc, updated_utc
        )
        VALUES (?, ?, 'derived_from', NULL, NULL, '2026-07-06T00:00:00Z', '2026-07-06T00:00:00Z');
        """,
        (dataset_id, parent_dataset_id),
    )
    registry.conn.commit()


def _make_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    fake_tmp = Path("/palette-test-temp-root") / tmp_path.name
    monkeypatch.setattr("fisheye.registry.prune_stale_datasets.tempfile.gettempdir", lambda: str(fake_tmp))

    registry_path = tmp_path / "registry.sqlite"
    registry = Registry(registry_path)
    try:
        _seed_dataset(
            registry,
            dataset_id="temp_pytest",
            zarr_path=fake_tmp / "pytest-of-delahantyj" / "pytest-1" / "analysis.zarr",
            recording_id="rec_delete",
        )
        _seed_dataset(
            registry,
            dataset_id="temp_plain",
            zarr_path=fake_tmp / "plain" / "analysis.zarr",
            recording_id="rec_shared",
        )
        _seed_dataset(
            registry,
            dataset_id="archive_missing",
            zarr_path=Path("/mnt/archive/offline/analysis.zarr"),
            recording_id="rec_shared",
        )
        _seed_dataset(
            registry,
            dataset_id="home_review",
            zarr_path=Path("/home/delahantyj@hhmi.org/worktree/analysis.zarr"),
            recording_id="rec_home",
        )
        registry.upsert_dataset(
            "unowned_analysis",
            session_uuid=None,
            zarr_path=Path("/nvme1/dan_detect_infer.zarr"),
            recording_id=None,
            artifact_kind="derived_analysis",
            zarr_use="analysis",
        )
        _seed_dependents(registry, "temp_pytest", parent_dataset_id="archive_missing")
    finally:
        registry.close()
    return registry_path, fake_tmp


def _count(conn: sqlite3.Connection, table: str) -> int:
    return int(conn.execute(f"SELECT COUNT(*) FROM {table};").fetchone()[0])


def test_dry_run_counts_match_and_writes_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry_path, _fake_tmp = _make_registry(tmp_path, monkeypatch)
    report_path = tmp_path / "dryrun.json"

    assert main(["--registry", str(registry_path), "--json", str(report_path)]) == 0

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["mode"] == "dry-run"
    assert report["read_only"] is True
    assert report["summary"]["temp_root_dataset_count"] == 2
    assert report["classes"]["pytest-tmp"]["dataset_count"] == 1
    assert report["classes"]["tmp"]["dataset_count"] == 1
    assert report["needs_maintainer_review"]["dataset_count"] == 1
    assert report["unowned_analysis_review"]["dataset_count"] == 1
    assert report["unowned_analysis_review"]["datasets"][0]["dataset_id"] == "unowned_analysis"
    assert report["classes"]["pytest-tmp"]["dependent_row_counts"]["provenance"] == 1
    assert report["classes"]["pytest-tmp"]["dependent_row_counts"]["detect_quality"] == 1
    assert report["classes"]["pytest-tmp"]["dependent_row_counts"]["recording_step_status"] == 1
    assert report["classes"]["pytest-tmp"]["dependent_row_counts"]["recording_step_status_history"] == 1
    assert report["classes"]["pytest-tmp"]["dependent_row_counts"]["dataset_lineage"] == 1


def test_execute_deletes_classified_temp_rows_dependents_and_orphan_recording(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry_path, _fake_tmp = _make_registry(tmp_path, monkeypatch)
    backup_path = tmp_path / "backup.sqlite"

    with sqlite3.connect(registry_path) as conn:
        pre_dataset_count = _count(conn, "datasets")
        pre_provenance_count = _count(conn, "provenance")

    assert (
        main(
            [
                "--registry",
                str(registry_path),
                "--execute",
                "--include-temp-root",
                "all-temp",
                "--backup",
                str(backup_path),
            ]
        )
        == 0
    )

    with sqlite3.connect(registry_path) as conn:
        dataset_ids = {
            str(row[0])
            for row in conn.execute("SELECT dataset_id FROM datasets ORDER BY dataset_id;").fetchall()
        }
        recording_ids = {
            str(row[0])
            for row in conn.execute("SELECT recording_id FROM recordings ORDER BY recording_id;").fetchall()
        }
        assert dataset_ids == {"archive_missing", "home_review", "unowned_analysis"}
        assert "rec_delete" not in recording_ids
        assert "rec_shared" in recording_ids
        assert _count(conn, "provenance") == pre_provenance_count - 1
        assert _count(conn, "detect_quality") == 0
        assert _count(conn, "recording_step_status") == 0
        assert _count(conn, "recording_step_status_history") == 0
        assert _count(conn, "dataset_lineage") == 0
        assert conn.execute("PRAGMA integrity_check;").fetchone()[0] == "ok"
        assert conn.execute("PRAGMA foreign_key_check;").fetchall() == []

    with sqlite3.connect(backup_path) as backup:
        assert _count(backup, "datasets") == pre_dataset_count
        assert _count(backup, "provenance") == pre_provenance_count


def test_home_row_survives_unless_exactly_included(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry_path, _fake_tmp = _make_registry(tmp_path, monkeypatch)

    assert (
        main(
            [
                "--registry",
                str(registry_path),
                "--execute",
                "--include-temp-root",
                "all-temp",
                "--include-dataset-id",
                "home_review",
                "--backup",
                str(tmp_path / "backup.sqlite"),
            ]
        )
        == 0
    )

    with sqlite3.connect(registry_path) as conn:
        dataset_ids = {
            str(row[0])
            for row in conn.execute("SELECT dataset_id FROM datasets ORDER BY dataset_id;").fetchall()
        }
    assert dataset_ids == {"archive_missing", "unowned_analysis"}


def test_unowned_analysis_row_survives_unless_exactly_included(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry_path, _fake_tmp = _make_registry(tmp_path, monkeypatch)

    assert (
        main(
            [
                "--registry",
                str(registry_path),
                "--execute",
                "--include-temp-root",
                "all-temp",
                "--include-dataset-id",
                "unowned_analysis",
                "--backup",
                str(tmp_path / "backup.sqlite"),
            ]
        )
        == 0
    )

    with sqlite3.connect(registry_path) as conn:
        dataset_ids = {
            str(row[0])
            for row in conn.execute("SELECT dataset_id FROM datasets ORDER BY dataset_id;").fetchall()
        }
    assert dataset_ids == {"archive_missing", "home_review"}


def test_execute_rejects_non_home_include_dataset_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry_path, _fake_tmp = _make_registry(tmp_path, monkeypatch)

    with pytest.raises(SystemExit, match="/home or unowned-analysis review rows"):
        main(
            [
                "--registry",
                str(registry_path),
                "--execute",
                "--include-temp-root",
                "all-temp",
                "--include-dataset-id",
                "archive_missing",
                "--backup",
                str(tmp_path / "backup.sqlite"),
            ]
        )


def test_drift_guard_catches_uncovered_dataset_id_table(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry_path, _fake_tmp = _make_registry(tmp_path, monkeypatch)

    with sqlite3.connect(registry_path) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("CREATE TABLE future_dataset_metrics (dataset_id TEXT, metric REAL);")
        conn.commit()
        with pytest.raises(RuntimeError, match="future_dataset_metrics"):
            assert_dependent_table_map_covers_schema(conn)


def test_read_only_connection_sets_query_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry_path, _fake_tmp = _make_registry(tmp_path, monkeypatch)

    conn = connect_read_only(registry_path)
    try:
        assert int(conn.execute("PRAGMA query_only;").fetchone()[0]) == 1
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("DELETE FROM datasets;")
    finally:
        conn.close()
