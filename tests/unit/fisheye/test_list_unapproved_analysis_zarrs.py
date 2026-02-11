from __future__ import annotations

import json
from pathlib import Path

from fisheye.utils.list_unapproved_analysis_zarrs import _collect_unapproved_rows


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
