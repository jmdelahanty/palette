from __future__ import annotations

from pathlib import Path
import sqlite3

import pytest

from fisheye.status_page.app import build_config
from fisheye.status_page.query import (
    build_health_report,
    query_dataset_status,
    query_status_heartbeat,
    query_status_history,
    query_status_summary,
    query_status_wide,
    validate_registry_path,
)


def _create_registry_fixture(path: Path, *, include_wide: bool = True) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            CREATE TABLE recording_step_status_latest (
                recording_id TEXT,
                dataset_id TEXT,
                camera_id TEXT,
                zarr_use TEXT,
                step_name TEXT,
                status TEXT,
                run_name TEXT,
                method TEXT,
                coverage_pct REAL,
                review_status_json TEXT,
                details_json TEXT,
                source TEXT,
                zarr_mtime_ns INTEGER,
                updated_utc TEXT
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE recording_step_status_history (
                event_id INTEGER PRIMARY KEY,
                dataset_id TEXT,
                recording_id TEXT,
                step_name TEXT,
                status TEXT,
                run_name TEXT,
                method TEXT,
                coverage_pct REAL,
                review_status_json TEXT,
                details_json TEXT,
                source TEXT,
                zarr_mtime_ns INTEGER,
                updated_utc TEXT,
                recorded_utc TEXT
            );
            """
        )
        if include_wide:
            conn.execute(
                """
                CREATE TABLE recording_step_status_wide (
                    "Recording" TEXT,
                    "Camera" TEXT,
                    "Use" TEXT,
                    "Detect" TEXT,
                    "Crop" TEXT,
                    "Keypoints" TEXT,
                    "Refined Keypoints (analysis/train)" TEXT,
                    "Eye Masks" TEXT,
                    "Refined Eye Masks" TEXT,
                    "Arena Assignment" TEXT,
                    "Track" TEXT
                );
                """
            )

        conn.executemany(
            """
            INSERT INTO recording_step_status_latest (
                recording_id, dataset_id, camera_id, zarr_use, step_name, status, run_name, method,
                coverage_pct, review_status_json, details_json, source,
                zarr_mtime_ns, updated_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            [
                (
                    "rec-a",
                    "ds-a",
                    "cam-1",
                    "analysis",
                    "keypoints",
                    "ok",
                    "kp_run_a",
                    "traditional",
                    100.0,
                    '{"state":"approved"}',
                    '{"note":"good"}',
                    "runtime",
                    1,
                    "2026-03-05T10:00:00+00:00",
                ),
                (
                    "rec-a",
                    "ds-a",
                    "cam-1",
                    "analysis",
                    "eye_masks",
                    "missing",
                    "eye_run_a",
                    "yolo",
                    80.0,
                    '{"state":"pending"}',
                    '{"note":"needs_eye"}',
                    "runtime",
                    2,
                    "2026-03-05T10:05:00+00:00",
                ),
                (
                    "rec-b",
                    "ds-b",
                    "cam-2",
                    "training",
                    "keypoints",
                    "error",
                    "kp_run_b",
                    "traditional",
                    0.0,
                    '{"state":"blocked"}',
                    '{"err":"bad"}',
                    "runtime",
                    3,
                    "2026-03-05T10:10:00+00:00",
                ),
                (
                    "rec-c",
                    "ds-c",
                    "cam-3",
                    "analysis",
                    "keypoints",
                    "ok",
                    "kp_run_c",
                    "traditional",
                    95.0,
                    '{"state":"approved"}',
                    '{"note":"ok"}',
                    "runtime",
                    4,
                    "2026-03-05T10:15:00+00:00",
                ),
            ],
        )

        conn.executemany(
            """
            INSERT INTO recording_step_status_history (
                event_id, dataset_id, recording_id, step_name, status, run_name, method,
                coverage_pct, review_status_json, details_json, source, zarr_mtime_ns,
                updated_utc, recorded_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            [
                (
                    1,
                    "ds-a",
                    "rec-a",
                    "keypoints",
                    "missing",
                    "kp_old",
                    "traditional",
                    50.0,
                    '{"state":"pending"}',
                    '{"note":"old"}',
                    "runtime",
                    10,
                    "2026-03-05T09:00:00+00:00",
                    "2026-03-05T09:01:00+00:00",
                ),
                (
                    2,
                    "ds-a",
                    "rec-a",
                    "keypoints",
                    "ok",
                    "kp_run_a",
                    "traditional",
                    100.0,
                    '{"state":"approved"}',
                    '{"note":"new"}',
                    "runtime",
                    11,
                    "2026-03-05T10:00:00+00:00",
                    "2026-03-05T10:01:00+00:00",
                ),
            ],
        )

        if include_wide:
            conn.executemany(
                """
                INSERT INTO recording_step_status_wide (
                    "Recording", "Camera", "Use", "Detect", "Crop", "Keypoints",
                    "Refined Keypoints (analysis/train)", "Eye Masks",
                    "Refined Eye Masks", "Arena Assignment", "Track"
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                [
                    (
                        "rec-a",
                        "cam-1",
                        "analysis",
                        "OK",
                        "OK",
                        "OK",
                        "OK",
                        "MISS",
                        "MISS",
                        "MISS",
                        "MISS",
                    ),
                    (
                        "rec-b",
                        "cam-2",
                        "training",
                        "OK",
                        "OK",
                        "MISS",
                        "MISS",
                        "N/A",
                        "N/A",
                        "N/A",
                        "N/A",
                    ),
                    (
                        "rec-c",
                        "cam-3",
                        "analysis",
                        "OK",
                        "OK",
                        "OK",
                        "OK",
                        "OK",
                        "OK",
                        "OK",
                        "OK",
                    ),
                ],
            )

        conn.commit()
    finally:
        conn.close()


def test_validate_registry_path_missing_raises(tmp_path: Path) -> None:
    missing = tmp_path / "missing_registry.sqlite"
    with pytest.raises(FileNotFoundError):
        validate_registry_path(missing)


def test_build_health_report_ok(tmp_path: Path) -> None:
    registry_path = tmp_path / "palette_registry.sqlite"
    _create_registry_fixture(registry_path)

    report = build_health_report(registry_path)

    assert report.ok is True
    assert report.details["sqlite_ok"] is True
    required = report.details["required_objects"]
    assert required["recording_step_status_latest"] is True
    assert required["recording_step_status_wide"] is True
    assert required["recording_step_status_history"] is True


def test_build_health_report_missing_required_object(tmp_path: Path) -> None:
    registry_path = tmp_path / "palette_registry.sqlite"
    _create_registry_fixture(registry_path, include_wide=False)

    report = build_health_report(registry_path)

    assert report.ok is False
    assert "missing_required_objects" in report.message
    required = report.details["required_objects"]
    assert required["recording_step_status_wide"] is False


def test_build_config_accepts_ready_registry(tmp_path: Path) -> None:
    registry_path = tmp_path / "palette_registry.sqlite"
    _create_registry_fixture(registry_path)

    config = build_config(
        registry=registry_path,
        host="127.0.0.1",
        port=8765,
        cwd=tmp_path,
    )

    assert config.registry_path == registry_path.resolve()
    assert config.host == "127.0.0.1"
    assert config.port == 8765
    assert config.static_dir.is_dir()


def test_query_status_summary_and_heartbeat(tmp_path: Path) -> None:
    registry_path = tmp_path / "palette_registry.sqlite"
    _create_registry_fixture(registry_path)

    summary = query_status_summary(registry_path)
    heartbeat = query_status_heartbeat(registry_path)

    assert summary["datasets_total"] == 3
    assert summary["recordings_total"] == 3
    assert summary["status_rows_total"] == 4
    assert summary["status_rows_missing"] == 1
    assert summary["status_rows_error"] == 1
    assert summary["wide_rows_total"] == 3
    assert summary["wide_rows_blocking"] == 2
    assert summary["wide_use_counts"]["analysis"] == 2
    assert summary["wide_use_counts"]["training"] == 1
    assert heartbeat["status_rows_total"] == 4
    assert heartbeat["wide_rows_total"] == 3


def test_query_status_wide_filters(tmp_path: Path) -> None:
    registry_path = tmp_path / "palette_registry.sqlite"
    _create_registry_fixture(registry_path)

    all_rows = query_status_wide(registry_path, limit=20, offset=0)
    assert all_rows["total_rows"] == 3
    assert all_rows["returned_rows"] == 3
    mapped = {row["Recording"]: row["_dataset_id"] for row in all_rows["rows"]}
    assert mapped["rec-a"] == "ds-a"
    assert mapped["rec-b"] == "ds-b"
    assert mapped["rec-c"] == "ds-c"

    filtered_q = query_status_wide(registry_path, q="rec-b", limit=20, offset=0)
    assert filtered_q["total_rows"] == 1
    assert filtered_q["rows"][0]["Recording"] == "rec-b"
    assert filtered_q["rows"][0]["_dataset_id"] == "ds-b"

    filtered_use = query_status_wide(registry_path, zarr_use="training", limit=20, offset=0)
    assert filtered_use["total_rows"] == 1
    assert filtered_use["rows"][0]["Use"] == "training"

    filtered_blocking = query_status_wide(registry_path, only_blocking=True, limit=20, offset=0)
    assert filtered_blocking["total_rows"] == 2
    assert {row["Recording"] for row in filtered_blocking["rows"]} == {"rec-a", "rec-b"}


def test_query_status_wide_does_not_treat_block_prefixed_cells_as_blocking(tmp_path: Path) -> None:
    registry_path = tmp_path / "palette_registry.sqlite"
    _create_registry_fixture(registry_path)

    conn = sqlite3.connect(registry_path)
    try:
        conn.execute(
            """
            UPDATE recording_step_status_wide
            SET "Track" = ?
            WHERE "Recording" = ?;
            """,
            ("BLOCK (1 unassigned, 25.0%)", "rec-c"),
        )
        conn.commit()
    finally:
        conn.close()

    summary = query_status_summary(registry_path)
    filtered_blocking = query_status_wide(registry_path, only_blocking=True, limit=20, offset=0)

    assert summary["wide_rows_blocking"] == 2
    assert filtered_blocking["total_rows"] == 2
    assert {row["Recording"] for row in filtered_blocking["rows"]} == {"rec-a", "rec-b"}


def test_query_dataset_status_and_history(tmp_path: Path) -> None:
    registry_path = tmp_path / "palette_registry.sqlite"
    _create_registry_fixture(registry_path)

    dataset = query_dataset_status(registry_path, dataset_id="ds-a")
    assert dataset["row_count"] == 2
    assert dataset["rows"][0]["step_name"] == "keypoints"
    assert dataset["rows"][0]["review_status"]["state"] == "approved"
    assert dataset["rows"][0]["details"]["note"] == "good"

    history = query_status_history(registry_path, dataset_id="ds-a", step_name="keypoints", limit=1)
    assert history["row_count"] == 1
    assert history["rows"][0]["run_name"] == "kp_run_a"
    assert history["rows"][0]["review_status"]["state"] == "approved"
    assert history["rows"][0]["details"]["note"] == "new"
