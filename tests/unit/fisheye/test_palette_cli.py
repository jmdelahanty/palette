from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import zarr

from fisheye.cli import palette
from fisheye.shared.zarr_run_completion import RUN_COMPLETED_AT_ATTR, mark_run_complete


def _run_json(capsys, *args: str) -> tuple[int, dict]:
    rc = palette.main([*args, "--json"])
    out = capsys.readouterr().out
    return rc, json.loads(out)


def _open_tmp_store(path: Path):
    return zarr.open_group(str(path), mode="w")


def _create_raw(root) -> None:
    raw = root.require_group("raw_video")
    if "images_full" not in raw:
        raw.create_array("images_full", shape=(1, 1, 1), dtype="uint8")


def _complete_run(root, parent_path: str, run_name: str, completed_at: str | None = None):
    parent = root
    for part in parent_path.split("/"):
        parent = parent.require_group(part)
    run = parent.require_group(run_name)
    mark_run_complete(run, parent_group=parent, run_name=run_name)
    if completed_at is not None:
        run.attrs[RUN_COMPLETED_AT_ATTR] = completed_at
    return run


def test_plan_fresh_store_recommends_import_frontier(tmp_path, capsys) -> None:
    zarr_path = tmp_path / "fresh_training.zarr"
    _open_tmp_store(zarr_path)

    rc, payload = _run_json(capsys, "plan", str(zarr_path))

    assert rc == palette.EXIT_OK
    assert payload["status"] == "ok"
    assert payload["metrics"]["complete"] == 0
    assert any(item["stage"] == "raw" and item["palette_verb"] == "import" for item in payload["next"])
    assert not any(item["stage"] in {"eye_masks", "refined_eye_masks", "eye_mask_tuning"} for item in payload["next"])
    eye_masks = next(stage for stage in payload["stages"] if stage["stage"] == "eye_masks")
    assert eye_masks["deprecated"] is True
    eye_mask_tuning = next(stage for stage in payload["stages"] if stage["stage"] == "eye_mask_tuning")
    assert eye_mask_tuning["deprecated"] is True
    detect = next(stage for stage in payload["stages"] if stage["stage"] == "detect")
    assert detect["state"] == "missing"


def test_plan_mid_pipeline_recommends_frontier(tmp_path, capsys) -> None:
    zarr_path = tmp_path / "mid_training.zarr"
    root = _open_tmp_store(zarr_path)
    _create_raw(root)

    rc, payload = _run_json(capsys, "plan", str(zarr_path))

    assert rc == palette.EXIT_OK
    next_stages = {item["stage"] for item in payload["next"]}
    assert "detect" in next_stages
    keypoints = next(stage for stage in payload["stages"] if stage["stage"] == "keypoints")
    assert keypoints["state"] == "blocked"
    assert "crop" in keypoints["blocked_by"]


def test_status_surfaces_legacy_assumed_completion(tmp_path, capsys) -> None:
    zarr_path = tmp_path / "legacy_training.zarr"
    root = _open_tmp_store(zarr_path)
    _create_raw(root)
    parent = root.require_group("detect_runs")
    parent.require_group("legacy_detect")
    parent.attrs["latest"] = "legacy_detect"

    rc, payload = _run_json(capsys, "status", str(zarr_path))

    assert rc == palette.EXIT_OK
    detect = next(stage for stage in payload["stages"] if stage["stage"] == "detect")
    assert detect["state"] == "complete"
    assert detect["run"] == "legacy_detect"
    assert detect["completion"] == "legacy_assumed"


def test_plan_reports_stale_downstream_stage(tmp_path, capsys) -> None:
    zarr_path = tmp_path / "stale_training.zarr"
    root = _open_tmp_store(zarr_path)
    _create_raw(root)
    _complete_run(root, "background", "background_001", "2026-01-01T00:00:00+00:00")
    _complete_run(root, "detect_runs", "detect_001", "2026-01-02T00:00:00+00:00")
    _complete_run(
        root,
        "detect_runs/detect_001/quality_reports",
        "quality_001",
        "2026-01-01T00:00:00+00:00",
    )

    rc, payload = _run_json(capsys, "plan", str(zarr_path))

    assert rc == palette.EXIT_OK
    assert any(
        item["stage"] == "detect_quality" and item["invalidated_by"] == "detect"
        for item in payload["stale"]
    )


def test_status_resolves_recording_from_readonly_registry(tmp_path, capsys) -> None:
    zarr_path = tmp_path / "registered_training.zarr"
    root = _open_tmp_store(zarr_path)
    _create_raw(root)
    registry = tmp_path / "palette_registry.sqlite"
    conn = sqlite3.connect(registry)
    conn.execute(
        """
        CREATE TABLE datasets (
            dataset_id TEXT PRIMARY KEY,
            session_uuid TEXT,
            zarr_path TEXT NOT NULL,
            recording_id TEXT,
            artifact_kind TEXT,
            zarr_origin TEXT,
            zarr_use TEXT,
            source_layout TEXT,
            source_frame_index_path TEXT,
            source_recording_frame_index_path TEXT,
            source_frame_index_schema TEXT,
            path_hash TEXT,
            created_utc TEXT,
            last_seen_utc TEXT,
            status TEXT
        );
        """
    )
    conn.execute(
        """
        INSERT INTO datasets(dataset_id, zarr_path, recording_id, zarr_use, status)
        VALUES (?, ?, ?, ?, ?);
        """,
        ("rec_001:zabc", str(zarr_path), "rec_001", "training", "active"),
    )
    conn.commit()
    conn.close()

    rc, payload = _run_json(capsys, "status", "rec_001", "--registry", str(registry))

    assert rc == palette.EXIT_OK
    assert payload["recording"] == "rec_001"
    assert payload["dataset_id"] == "rec_001:zabc"
    assert payload["zarr_path"] == str(zarr_path.resolve())


def test_missing_recording_returns_blocked_exit_code(tmp_path, capsys) -> None:
    registry = tmp_path / "palette_registry.sqlite"
    conn = sqlite3.connect(registry)
    conn.execute(
        """
        CREATE TABLE datasets (
            dataset_id TEXT PRIMARY KEY,
            session_uuid TEXT,
            zarr_path TEXT NOT NULL,
            recording_id TEXT,
            artifact_kind TEXT,
            zarr_origin TEXT,
            zarr_use TEXT,
            source_layout TEXT,
            source_frame_index_path TEXT,
            source_recording_frame_index_path TEXT,
            source_frame_index_schema TEXT,
            path_hash TEXT,
            created_utc TEXT,
            last_seen_utc TEXT,
            status TEXT
        );
        """
    )
    conn.commit()
    conn.close()

    rc, payload = _run_json(capsys, "status", "missing_rec", "--registry", str(registry))

    assert rc == palette.EXIT_BLOCKED
    assert payload["status"] == "blocked"
    assert payload["reason_code"] == "RECORDING_NOT_FOUND"
    assert payload["next_hints"]
