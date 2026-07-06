from __future__ import annotations

import json
from pathlib import Path

from fisheye.diagnostics.detect_review_pointer_census import (
    LEGACY_DETECT_REVIEW_AUTHORITY_ATTR,
    run_census,
    scan_store,
)
from fisheye.shared.zarr_run_completion import (
    COMPLETION_EPOCH_ATTR,
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
)


def _write_group(path: Path, attrs: dict[str, object] | None = None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    payload = {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": attrs or {},
    }
    (path / "zarr.json").write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_store(root: Path, name: str) -> Path:
    zarr_path = root / name / "zarr" / f"{name}_analysis.zarr"
    _write_group(zarr_path)
    return zarr_path


def test_scan_store_classifies_backfillable_legacy_pointer(tmp_path: Path) -> None:
    zarr_path = _write_store(tmp_path, "rec_a")
    parent = zarr_path / "refined_detect_runs"
    _write_group(
        parent,
        {
            LEGACY_DETECT_REVIEW_AUTHORITY_ATTR: "reviewed",
            "latest": "reviewed",
            COMPLETION_EPOCH_ATTR: 1,
        },
    )
    _write_group(parent / "reviewed", {RUN_COMPLETION_STATUS_ATTR: RUN_STATUS_COMPLETE})

    store = scan_store(zarr_path, ["filesystem"])

    assert store.bucket == "BACKFILLABLE"
    refined = next(parent for parent in store.parents if parent.parent_name == "refined_detect_runs")
    assert refined.legacy_fallback_would_fire is True
    assert refined.batch_winner_would_change is False
    assert refined.task_generation_winner_would_change is False


def test_scan_store_marks_conflicting_latest_ambiguous(tmp_path: Path) -> None:
    zarr_path = _write_store(tmp_path, "rec_b")
    parent = zarr_path / "refined_detect_runs"
    _write_group(
        parent,
        {
            LEGACY_DETECT_REVIEW_AUTHORITY_ATTR: "reviewed",
            "latest": "newer",
            COMPLETION_EPOCH_ATTR: 1,
        },
    )
    _write_group(parent / "reviewed", {RUN_COMPLETION_STATUS_ATTR: RUN_STATUS_COMPLETE})
    _write_group(parent / "newer", {RUN_COMPLETION_STATUS_ATTR: RUN_STATUS_COMPLETE})

    store = scan_store(zarr_path, ["registry"])

    assert store.bucket == "AMBIGUOUS"
    refined = next(parent for parent in store.parents if parent.parent_name == "refined_detect_runs")
    assert refined.legacy_fallback_would_fire is True
    assert refined.batch_winner_with_legacy == "newer"
    assert refined.task_generation_winner_with_legacy == "reviewed"
    assert refined.task_generation_winner_without_legacy == "newer"


def test_run_census_reports_filesystem_registry_diff_read_only(tmp_path: Path) -> None:
    recordings_root = tmp_path / "recordings"
    fs_zarr = _write_store(recordings_root, "rec_fs")
    _write_group(fs_zarr / "refined_detect_runs")
    registry_zarr = tmp_path / "elsewhere" / "registry_only.zarr"
    _write_group(registry_zarr)
    registry_path = tmp_path / "registry.sqlite"

    import sqlite3

    conn = sqlite3.connect(registry_path)
    try:
        conn.execute("CREATE TABLE datasets (zarr_path TEXT, status TEXT);")
        conn.execute("INSERT INTO datasets (zarr_path, status) VALUES (?, 'ok');", (str(registry_zarr),))
        conn.commit()
    finally:
        conn.close()

    census = run_census(recordings_root=recordings_root, registry_path=registry_path)

    assert census.enumeration.filesystem_count == 1
    assert census.enumeration.registry_count == 1
    assert census.enumeration.filesystem_only == (str(fs_zarr),)
    assert census.enumeration.registry_only == (str(registry_zarr),)
    assert census.bucket_counts == {"SAFE": 2, "BACKFILLABLE": 0, "AMBIGUOUS": 0}
