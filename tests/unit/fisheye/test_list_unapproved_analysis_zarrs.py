from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from fisheye.utils.list_unapproved_analysis_zarrs import (
    _collect_unapproved_rows,
    _collect_unapproved_rows_from_registry,
)


def _write_zarr_json(path: Path, attrs: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"zarr_format": 3, "node_type": "group", "attributes": attrs}),
        encoding="utf-8",
    )


def _make_archive(
    root: Path,
    name: str,
    *,
    zarr_use: str = "analysis",
    latest_run: str | None = "refined_detect_1",
    review_state: str | None = None,
) -> Path:
    zarr_path = root / name / "zarr" / f"{name}_{zarr_use}.zarr"
    _write_zarr_json(zarr_path / "zarr.json", {"zarr_purpose": zarr_use})
    if latest_run is not None:
        _write_zarr_json(zarr_path / "refined_detect_runs" / "zarr.json", {"latest": latest_run})
        status = {"state": review_state} if review_state is not None else None
        run_attrs = {"detect_review_status": status} if status is not None else {}
        _write_zarr_json(zarr_path / "refined_detect_runs" / latest_run / "zarr.json", run_attrs)
    return zarr_path


def test_collect_unapproved_rows_filters_analysis_and_detects_nonapproved(tmp_path: Path) -> None:
    _make_archive(tmp_path, "rec_approved", zarr_use="analysis", review_state="approved")
    needs_review = _make_archive(tmp_path, "rec_review", zarr_use="analysis", review_state="needs_review")
    _make_archive(tmp_path, "rec_training", zarr_use="training", review_state="approved")

    rows = _collect_unapproved_rows(
        [tmp_path],
        recursive=True,
        zarr_use_filter="analysis",
        approved_state="approved",
    )
    assert len(rows) == 1
    row = rows[0]
    assert row.zarr_path == str(needs_review)
    assert row.review_state == "needs_review"
    assert row.reason == "review_state_not_approved"


def test_collect_unapproved_rows_marks_missing_status(tmp_path: Path) -> None:
    missing_status = _make_archive(tmp_path, "rec_missing", zarr_use="analysis", review_state=None)

    rows = _collect_unapproved_rows(
        [tmp_path],
        recursive=True,
        zarr_use_filter="analysis",
        approved_state="approved",
    )
    assert len(rows) == 1
    row = rows[0]
    assert row.zarr_path == str(missing_status)
    assert row.review_state is None
    assert row.reason == "no_detect_review_status"


def test_collect_unapproved_rows_includes_no_latest_refined_run(tmp_path: Path) -> None:
    no_latest = _make_archive(tmp_path, "rec_nolatest", zarr_use="analysis", latest_run=None)

    rows = _collect_unapproved_rows(
        [tmp_path],
        recursive=True,
        zarr_use_filter="analysis",
        approved_state="approved",
    )
    assert len(rows) == 1
    row = rows[0]
    assert row.zarr_path == str(no_latest)
    assert row.latest_refined_run is None
    assert row.reason == "no_latest_refined_run"


def _write_registry_fixture(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(
            """
            CREATE TABLE datasets (
                dataset_id TEXT PRIMARY KEY,
                zarr_path TEXT,
                zarr_use TEXT,
                status TEXT
            );

            CREATE TABLE detect_quality_current (
                dataset_id TEXT,
                refined_run TEXT,
                refined_created_utc TEXT,
                review_state TEXT,
                review_intended_use TEXT,
                review_resolved_group TEXT,
                review_timestamp_utc TEXT,
                quality_updated_utc TEXT
            );

            CREATE VIEW refined_detect_review_current AS
            SELECT * FROM detect_quality_current;
            """
        )
        conn.executemany(
            "INSERT INTO datasets(dataset_id, zarr_path, zarr_use, status) VALUES (?, ?, ?, ?)",
            [
                ("d_approved", "/nvme1/recordings/a/zarr/a_analysis.zarr", "analysis", None),
                ("d_review", "/nvme1/recordings/b/zarr/b_analysis.zarr", "analysis", None),
                ("d_missing_state", "/nvme1/recordings/c/zarr/c_analysis.zarr", "analysis", None),
                ("d_no_quality", "/nvme1/recordings/d/zarr/d_analysis.zarr", "analysis", None),
                ("d_training", "/nvme1/recordings/e/zarr/e_training.zarr", "training", None),
                ("d_missing", "/nvme1/recordings/f/zarr/f_analysis.zarr", "analysis", "missing"),
                ("d_other_root", "/tmp/other_root/g_analysis.zarr", "analysis", None),
            ],
        )
        conn.executemany(
            """
            INSERT INTO detect_quality_current(
                dataset_id, refined_run, refined_created_utc, review_state, review_intended_use,
                review_resolved_group, review_timestamp_utc, quality_updated_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                ("d_approved", "refined_1", "2026-01-01T00:00:00Z", "approved", "training", "manual", None, None),
                ("d_review", "refined_2", "2026-01-02T00:00:00Z", "needs_review", "training", "interpolated", None, None),
                ("d_missing_state", "refined_3", "2026-01-03T00:00:00Z", None, None, "interpolated", None, None),
                ("d_training", "refined_4", "2026-01-04T00:00:00Z", "needs_review", "training", "interpolated", None, None),
                ("d_missing", "refined_5", "2026-01-05T00:00:00Z", "needs_review", "training", "interpolated", None, None),
                ("d_other_root", "refined_6", "2026-01-06T00:00:00Z", "needs_review", "training", "interpolated", None, None),
            ],
        )
        conn.commit()
    finally:
        conn.close()


def test_collect_unapproved_rows_from_registry_filters_and_classifies(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _write_registry_fixture(registry_path)

    rows = _collect_unapproved_rows_from_registry(
        registry_path,
        zarr_use_filter="analysis",
        approved_state="approved",
        path_contains="/nvme1/recordings",
    )

    by_path = {row.zarr_path: row for row in rows}
    assert "/nvme1/recordings/a/zarr/a_analysis.zarr" not in by_path
    assert "/nvme1/recordings/e/zarr/e_training.zarr" not in by_path
    assert "/nvme1/recordings/f/zarr/f_analysis.zarr" not in by_path
    assert "/tmp/other_root/g_analysis.zarr" not in by_path

    assert by_path["/nvme1/recordings/b/zarr/b_analysis.zarr"].reason == "review_state_not_approved"
    assert by_path["/nvme1/recordings/c/zarr/c_analysis.zarr"].reason == "review_state_missing"
    assert by_path["/nvme1/recordings/d/zarr/d_analysis.zarr"].reason == "no_detect_quality_row"
