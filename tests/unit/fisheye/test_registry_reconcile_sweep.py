from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
import sys

import zarr

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.registry.db import Registry
from fisheye.registry.reconcile_sweep import main, reconcile_roots


def _detection_summary() -> dict:
    return {
        "schema_name": "detection_dataset_profile",
        "schema_version": "v1",
        "created_at_utc": "2026-02-12T03:00:00+00:00",
        "dataset": {"recording_id": "rec_a", "zarr_use": "training"},
        "source": {"detection_path": "refined_detect_runs/r/manual", "detection_type": "manual"},
        "coverage": {"frames_total": 100, "frames_with_detections": 95, "coverage_percent": 95.0},
        "counts": {"detections_total": 950, "detections_per_frame": {"p50": 9.0, "p90": 10.0}},
        "geometry_norm": {
            "w": {"p10": 0.1, "p50": 0.2, "p90": 0.3},
            "h": {"p10": 0.1, "p50": 0.2, "p90": 0.3},
            "area": {"p10": 0.01, "p50": 0.04, "p90": 0.09},
            "aspect_ratio": {"p10": 0.8, "p50": 1.0, "p90": 1.2},
        },
        "spatial": {"edge_proximity_rate": 0.03},
        "composition": {"rig_id": "rig_a", "protocol_name": "DefaultScreen", "dpf_at_acquisition": 7},
    }


def _build_zarr(zarr_path: Path, *, recording_id: str, session_uuid: str) -> None:
    root = zarr.open_group(str(zarr_path), mode="w")
    root.attrs["session_uuid"] = session_uuid
    root.attrs["recording_id"] = recording_id
    root.attrs["zarr_use"] = "training"
    root.attrs["zarr_purpose"] = "training"

    analysis = root.require_group("analysis")
    det_parent = analysis.require_group("detection_profile_runs")
    det_run = det_parent.require_group("detection_profile_2026-02-12_03-00-00")
    det_run.attrs["profile_summary"] = _detection_summary()
    det_run.attrs["created_at_utc"] = "2026-02-12T03:00:00+00:00"
    det_parent.attrs["latest"] = "detection_profile_2026-02-12_03-00-00"


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dump_rows(registry_path: Path) -> dict[str, list[tuple]]:
    snapshot: dict[str, list[tuple]] = {}
    with sqlite3.connect(registry_path) as conn:
        conn.row_factory = sqlite3.Row
        table_rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
        ).fetchall()
        for table_row in table_rows:
            table = str(table_row["name"])
            if table.startswith("sqlite_"):
                continue
            cursor = conn.execute(f"SELECT * FROM {table};")
            columns = [desc[0] for desc in cursor.description]
            keep = [
                column
                for column in columns
                if column not in {"recorded_utc"} and not column.endswith(("updated_utc", "seen_utc"))
            ]
            snapshot[table] = sorted(
                tuple((column, row[column]) for column in keep)
                for row in cursor.fetchall()
            )
    return snapshot


def _seed_missing_row(registry_path: Path, missing_path: Path) -> None:
    registry = Registry(registry_path)
    try:
        registry.upsert_recording(recording_id="rec_missing", session_uuid="rec_missing")
        registry.upsert_dataset(
            "missing_dataset",
            session_uuid="missing_dataset",
            zarr_path=missing_path,
            recording_id="rec_missing",
            artifact_kind="source_recording",
            zarr_use="analysis",
        )
    finally:
        registry.close()


def test_dry_run_is_read_only_and_reports_new_known_unreadable_missing(tmp_path: Path) -> None:
    root_dir = tmp_path / "recordings"
    known_path = root_dir / "known.zarr"
    new_path = root_dir / "new.zarr"
    empty_stub_path = root_dir / "empty_stub.zarr"
    unreadable_path = root_dir / "bad.zarr"
    missing_path = root_dir / "vanished.zarr"
    _build_zarr(known_path, recording_id="rec_known", session_uuid="known_dataset")
    _build_zarr(new_path, recording_id="rec_new", session_uuid="new_dataset")
    empty_stub_path.mkdir(parents=True)
    (empty_stub_path / "zarr.json").write_text(
        json.dumps(
            {
                "attributes": {},
                "zarr_format": 3,
                "consolidated_metadata": None,
                "node_type": "group",
            }
        ),
        encoding="utf-8",
    )
    unreadable_path.mkdir(parents=True)
    (unreadable_path / "zarr.json").write_text("{not-json", encoding="utf-8")

    registry_path = tmp_path / "registry.sqlite"
    registry = Registry(registry_path)
    try:
        registry.register_from_root(zarr.open_group(str(known_path), mode="r"), known_path)
    finally:
        registry.close()
    _seed_missing_row(registry_path, missing_path)

    before = _hash_file(registry_path)
    report = reconcile_roots(registry_path, [root_dir], apply=False)
    after = _hash_file(registry_path)

    assert before == after
    assert report["read_only"] is True
    assert report["summary"]["new_store_count"] == 2
    assert report["summary"]["known_store_count"] == 1
    assert report["summary"]["unreadable_store_count"] == 1
    assert report["summary"]["would_mark_missing_count"] == 1
    assert {row["classification"] for row in report["stores"]} == {"new", "known", "unreadable"}
    empty_row = next(row for row in report["stores"] if row["path"] == str(empty_stub_path))
    assert empty_row["classification"] == "new"
    assert empty_row["is_empty_zarr_stub"] is True
    assert report["registered_but_vanished"]["would_mark_missing"][0]["dataset_id"] == "missing_dataset"


def test_apply_reconciles_new_and_known_marks_missing_and_is_idempotent(tmp_path: Path) -> None:
    root_dir = tmp_path / "recordings"
    known_path = root_dir / "known.zarr"
    new_path = root_dir / "new.zarr"
    missing_path = root_dir / "vanished.zarr"
    _build_zarr(known_path, recording_id="rec_known", session_uuid="known_dataset")
    _build_zarr(new_path, recording_id="rec_new", session_uuid="new_dataset")

    registry_path = tmp_path / "registry.sqlite"
    registry = Registry(registry_path)
    try:
        registry.register_from_root(zarr.open_group(str(known_path), mode="r"), known_path)
    finally:
        registry.close()
    _seed_missing_row(registry_path, missing_path)

    first_report = reconcile_roots(
        registry_path,
        [root_dir],
        include_step_status=False,
        apply=True,
    )
    first_snapshot = _dump_rows(registry_path)
    second_report = reconcile_roots(
        registry_path,
        [root_dir],
        include_step_status=False,
        apply=True,
    )
    second_snapshot = _dump_rows(registry_path)

    assert first_report["summary"]["new_store_count"] == 1
    assert first_report["summary"]["known_store_count"] == 1
    assert first_report["registered_but_vanished"]["result"]["marked_missing"] == 1
    assert second_report["registered_but_vanished"]["result"]["marked_missing"] == 0
    assert first_snapshot == second_snapshot

    with sqlite3.connect(registry_path) as conn:
        conn.row_factory = sqlite3.Row
        profile_count = conn.execute(
            "SELECT COUNT(*) AS n FROM detection_data_profile;"
        ).fetchone()["n"]
        missing_status = conn.execute(
            "SELECT status FROM datasets WHERE dataset_id = 'missing_dataset';"
        ).fetchone()["status"]
    assert profile_count == 2
    assert missing_status == "missing"


def test_apply_reports_unreadable_without_crashing(tmp_path: Path) -> None:
    root_dir = tmp_path / "recordings"
    unreadable_path = root_dir / "bad.zarr"
    unreadable_path.mkdir(parents=True)
    (unreadable_path / "zarr.json").write_text("{not-json", encoding="utf-8")

    registry_path = tmp_path / "registry.sqlite"
    Registry(registry_path).close()

    report = reconcile_roots(registry_path, [root_dir], include_step_status=False, apply=True)

    assert report["summary"]["unreadable_store_count"] == 1
    assert report["stores"][0]["classification"] == "unreadable"
    assert report["stores"][0]["applied"] is False
    assert "error" in report["stores"][0]


def test_cli_writes_json_and_defaults_to_dry_run(tmp_path: Path) -> None:
    root_dir = tmp_path / "recordings"
    new_path = root_dir / "new.zarr"
    _build_zarr(new_path, recording_id="rec_new", session_uuid="new_dataset")
    registry_path = tmp_path / "registry.sqlite"
    Registry(registry_path).close()
    report_path = tmp_path / "sweep.json"

    assert (
        main(
            [
                "--registry",
                str(registry_path),
                "--reconcile-root",
                str(root_dir),
                "--json",
                str(report_path),
            ]
        )
        == 0
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["mode"] == "dry-run"
    assert report["read_only"] is True
    assert report["summary"]["new_store_count"] == 1
