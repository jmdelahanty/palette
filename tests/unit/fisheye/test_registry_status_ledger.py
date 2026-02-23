"""Unit tests for the shared recording step status writer API."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.registry.db import Registry
from fisheye.registry.status_ledger import upsert_recording_step_status


def _create_registry(tmp_path: Path) -> Registry:
    registry = Registry(tmp_path / "registry.sqlite")
    registry.upsert_dataset(
        dataset_id="dataset_a",
        session_uuid="session_a",
        zarr_path=tmp_path / "recordings" / "recording_a" / "zarr" / "recording_a_analysis.zarr",
        recording_id="recording_a",
        artifact_kind="source_recording",
        zarr_use="analysis",
    )
    return registry


def test_upsert_recording_step_status_inserts_latest_row(tmp_path: Path) -> None:
    registry = _create_registry(tmp_path)

    payload = upsert_recording_step_status(
        registry,
        dataset_id="dataset_a",
        step_name="detect",
        status="OK",
        run_name="detect_2026-02-22_00-00-00",
        method="yolo",
        coverage_pct=99.5,
        review_status_json={"state": "approved", "reviewer": "pytest"},
        details_json={"frames_total": 1200, "frames_present": 1194},
        updated_utc="2026-02-22T00:00:00+00:00",
        source="unit_test",
    )

    row = registry.conn.execute(
        """
        SELECT
            dataset_id,
            recording_id,
            step_name,
            status,
            run_name,
            method,
            coverage_pct,
            review_status_json,
            details_json,
            updated_utc,
            source
        FROM recording_step_status
        WHERE dataset_id = ? AND step_name = ?;
        """,
        ("dataset_a", "detect"),
    ).fetchone()
    assert row is not None
    assert str(row["dataset_id"]) == "dataset_a"
    assert str(row["recording_id"]) == "recording_a"
    assert str(row["step_name"]) == "detect"
    assert str(row["status"]) == "ok"
    assert str(row["run_name"]) == "detect_2026-02-22_00-00-00"
    assert str(row["method"]) == "yolo"
    assert float(row["coverage_pct"]) == 99.5
    assert str(row["review_status_json"]) == '{"reviewer":"pytest","state":"approved"}'
    assert str(row["details_json"]) == '{"frames_present":1194,"frames_total":1200}'
    assert str(row["updated_utc"]) == "2026-02-22T00:00:00+00:00"
    assert str(row["source"]) == "unit_test"
    assert payload["recording_id"] == "recording_a"
    registry.close()


def test_upsert_recording_step_status_updates_existing_row_idempotently(tmp_path: Path) -> None:
    registry = _create_registry(tmp_path)

    upsert_recording_step_status(
        registry,
        dataset_id="dataset_a",
        step_name="keypoints",
        status="missing",
        run_name=None,
        method=None,
        coverage_pct=None,
        review_status_json={"state": "pending"},
        details_json={"reason": "no_source_crops"},
        updated_utc="2026-02-22T00:00:00+00:00",
        source="unit_test",
    )
    upsert_recording_step_status(
        registry,
        dataset_id="dataset_a",
        step_name="keypoints",
        status="ok",
        run_name="keypoints_2026-02-22_00-10-00",
        method="yolo_pose",
        coverage_pct=91.2,
        review_status_json={"state": "approved"},
        details_json={"reason": "resolved"},
        updated_utc="2026-02-22T00:10:00+00:00",
        source="unit_test",
    )

    count = registry.conn.execute(
        """
        SELECT COUNT(*) AS n
        FROM recording_step_status
        WHERE dataset_id = ? AND step_name = ?;
        """,
        ("dataset_a", "keypoints"),
    ).fetchone()
    assert count is not None
    assert int(count["n"]) == 1

    row = registry.conn.execute(
        """
        SELECT status, run_name, method, coverage_pct, updated_utc
        FROM recording_step_status
        WHERE dataset_id = ? AND step_name = ?;
        """,
        ("dataset_a", "keypoints"),
    ).fetchone()
    assert row is not None
    assert str(row["status"]) == "ok"
    assert str(row["run_name"]) == "keypoints_2026-02-22_00-10-00"
    assert str(row["method"]) == "yolo_pose"
    assert float(row["coverage_pct"]) == 91.2
    assert str(row["updated_utc"]) == "2026-02-22T00:10:00+00:00"
    registry.close()


def test_upsert_recording_step_status_appends_history_rows(tmp_path: Path) -> None:
    registry = _create_registry(tmp_path)

    upsert_recording_step_status(
        registry,
        dataset_id="dataset_a",
        step_name="crop",
        status="missing",
        updated_utc="2026-02-22T00:00:00+00:00",
        source="unit_test",
    )
    upsert_recording_step_status(
        registry,
        dataset_id="dataset_a",
        step_name="crop",
        status="ok",
        run_name="crop_2026-02-22_00-05-00",
        method="sam2",
        coverage_pct=100.0,
        updated_utc="2026-02-22T00:05:00+00:00",
        source="unit_test",
    )

    rows = registry.conn.execute(
        """
        SELECT status, run_name, recorded_utc
        FROM recording_step_status_history
        WHERE dataset_id = ? AND step_name = ?
        ORDER BY event_id;
        """,
        ("dataset_a", "crop"),
    ).fetchall()
    assert len(rows) == 2
    assert str(rows[0]["status"]) == "missing"
    assert rows[0]["run_name"] is None
    assert str(rows[1]["status"]) == "ok"
    assert str(rows[1]["run_name"]) == "crop_2026-02-22_00-05-00"
    assert str(rows[0]["recorded_utc"]).strip() != ""
    assert str(rows[1]["recorded_utc"]).strip() != ""
    registry.close()


def test_upsert_recording_step_status_rejects_invalid_status(tmp_path: Path) -> None:
    registry = _create_registry(tmp_path)

    with pytest.raises(ValueError, match="status must be one of"):
        upsert_recording_step_status(
            registry,
            dataset_id="dataset_a",
            step_name="detect",
            status="broken",
            source="unit_test",
        )

    count_latest = registry.conn.execute(
        "SELECT COUNT(*) AS n FROM recording_step_status WHERE dataset_id = ?;",
        ("dataset_a",),
    ).fetchone()
    count_history = registry.conn.execute(
        "SELECT COUNT(*) AS n FROM recording_step_status_history WHERE dataset_id = ?;",
        ("dataset_a",),
    ).fetchone()
    assert count_latest is not None
    assert count_history is not None
    assert int(count_latest["n"]) == 0
    assert int(count_history["n"]) == 0
    registry.close()
