from __future__ import annotations

import json
import shlex
import sqlite3
from pathlib import Path

import zarr

from fisheye.cli import palette
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETED_AT_ATTR,
    mark_run_complete,
    set_authoritative_run,
)


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
    assert not any("<registry.sqlite>" in str(item.get("action") or "") for item in payload["next"])
    raw_action = next(item for item in payload["next"] if item["stage"] == "raw")
    assert raw_action["action_status"] == "needs_input"
    assert raw_action["missing_inputs"] == ["registry"]
    assert raw_action["action"] == ""
    assert not any("<registry.sqlite>" in hint for hint in payload["next_hints"])
    assert raw_action["action"] not in payload["next_hints"]
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


def test_plan_hint_for_path_with_space_is_shell_quoted(tmp_path, capsys) -> None:
    zarr_path = tmp_path / "recording with space" / "space training.zarr"
    zarr_path.parent.mkdir(parents=True)
    root = _open_tmp_store(zarr_path)
    _create_raw(root)

    rc, payload = _run_json(capsys, "plan", str(zarr_path))

    assert rc == palette.EXIT_OK
    detect = next(item for item in payload["next"] if item["stage"] == "detect")
    assert detect["action_status"] == "mapped"
    assert "<registry.sqlite>" not in detect["action"]
    assert shlex.split(detect["action"]) == ["palette", "detect", str(zarr_path.resolve()), "--apply"]
    assert detect["action"] in payload["next_hints"]


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


def test_status_surfaces_authoritative_run_distinct_from_latest(tmp_path, capsys) -> None:
    zarr_path = tmp_path / "authority_training.zarr"
    root = _open_tmp_store(zarr_path)
    _create_raw(root)
    approved = _complete_run(root, "subject_mask_runs", "subject_approved")
    _complete_run(root, "subject_mask_runs", "subject_smoke")
    set_authoritative_run(
        root["subject_mask_runs"],
        "subject_approved",
        approved_by="jeremy",
        approved_at="2026-07-02T12:00:00+00:00",
        git_sha="abc123",
        note="reviewed",
    )
    approved.attrs[RUN_COMPLETED_AT_ATTR] = "2026-07-02T12:00:00+00:00"

    rc, payload = _run_json(capsys, "status", str(zarr_path))

    assert rc == palette.EXIT_OK
    subject_masks = next(stage for stage in payload["stages"] if stage["stage"] == "subject_masks")
    assert subject_masks["state"] == "complete"
    assert subject_masks["run"] == "subject_approved"
    assert subject_masks["run_resolution"] == "authoritative"
    assert subject_masks["authoritative_run"] == "subject_approved"
    assert subject_masks["latest_complete_run"] == "subject_smoke"
    assert subject_masks["authoritative_run_provenance"]["approved_by"] == "jeremy"


def test_status_discovers_embedded_track_kinematics_visualization(
    tmp_path,
    capsys,
) -> None:
    zarr_path = tmp_path / "core_behavior.zarr"
    root = _open_tmp_store(zarr_path)
    parent = root.require_group("analysis").require_group("track_kinematics_runs")
    run = parent.require_group("offline").require_group("track_a")
    mark_run_complete(
        run,
        parent_group=parent,
        run_name="offline/track_a",
        completed_at_utc="2026-07-14T12:00:00+00:00",
    )
    artifact = run.require_group("visualizations").require_group(
        "track_kinematics_summary_track_0_interactive"
    )
    artifact.require_group("spec_json")
    artifact.attrs.update(
        {
            "renderer": "palette-track-kinematics-summary-v1",
            "created_at_utc": "2026-07-14T12:01:00+00:00",
        }
    )

    rc, payload = _run_json(capsys, "status", str(zarr_path))

    assert rc == palette.EXIT_OK
    visualization = next(
        stage
        for stage in payload["stages"]
        if stage["stage"] == "track_kinematics_visualization"
    )
    assert visualization["state"] == "complete"
    assert visualization["run"] == "offline/track_a"
    assert visualization["completion"] == "artifact_present"
    assert visualization["artifact"].endswith(
        "offline/track_a/visualizations/track_kinematics_summary_track_0_interactive"
    )


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


def test_plan_uses_authoritative_input_run_for_staleness(tmp_path, capsys) -> None:
    zarr_path = tmp_path / "authority_stale_training.zarr"
    root = _open_tmp_store(zarr_path)
    _create_raw(root)
    _complete_run(
        root,
        "subject_mask_runs",
        "subject_approved",
        "2026-01-01T00:00:00+00:00",
    )
    _complete_run(
        root,
        "refined_subject_masks_runs",
        "refined_approved",
        "2026-01-02T00:00:00+00:00",
    )
    _complete_run(
        root,
        "subject_mask_runs",
        "subject_smoke",
        "2026-01-03T00:00:00+00:00",
    )
    set_authoritative_run(
        root["subject_mask_runs"],
        "subject_approved",
        approved_by="jeremy",
        approved_at="2026-01-04T00:00:00+00:00",
        git_sha="abc123",
        note="reviewed full run",
    )

    rc, payload = _run_json(capsys, "plan", str(zarr_path))

    assert rc == palette.EXIT_OK
    subject_masks = next(stage for stage in payload["stages"] if stage["stage"] == "subject_masks")
    assert subject_masks["run"] == "subject_approved"
    assert subject_masks["latest_complete_run"] == "subject_smoke"
    assert subject_masks["run_resolution"] == "authoritative"
    assert not any(
        item["stage"] == "refined_subject_masks" and item["invalidated_by"] == "subject_masks"
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


def test_artifacts_verb_returns_inventory_envelope(tmp_path) -> None:
    zarr_path = tmp_path / "inventory_training.zarr"
    root = _open_tmp_store(zarr_path)
    _create_raw(root)
    _complete_run(root, "detect_runs", "detect_001")

    payload = palette.artifacts(palette.ArtifactsRequest(recording=zarr_path))

    assert payload["schema"] == palette.SCHEMA
    assert payload["command"] == "palette artifacts"
    assert payload["status"] == "ok"
    assert payload["reason_code"] == "OK"
    assert payload["zarr_path"] == str(zarr_path.resolve())
    assert payload["run"] is None
    assert payload["artifacts"] == [str(zarr_path.resolve())]
    assert payload["provenance"]["read_only"] is True
    assert payload["provenance"]["inventory_source"] == "fisheye.shared.recording_artifact_inventory"
    assert payload["next_hints"]
    inventory = payload["inventory"]
    assert inventory["schema_id"] == "palette.recording_artifact_inventory.v1"
    assert payload["metrics"]["run_count"] == inventory["run_count"] == 1
    detect_family = next(
        family for family in inventory["root_run_families"] if family["run_parent_path"] == "detect_runs"
    )
    assert detect_family["runs"][0]["name"] == "detect_001"


def test_artifacts_verb_missing_recording_returns_blocked_envelope(tmp_path) -> None:
    payload = palette.artifacts(
        palette.ArtifactsRequest(recording="missing_rec", registry=tmp_path / "absent_registry.sqlite")
    )

    assert payload["status"] == "blocked"
    assert payload["reason_code"] == "RECORDING_NOT_FOUND"
    assert payload["command"] == "palette artifacts"
    assert payload["recording"] == "missing_rec"
    assert payload["next_hints"]


def test_artifacts_cli_json_round_trips_inventory(tmp_path, capsys) -> None:
    zarr_path = tmp_path / "inventory_cli_training.zarr"
    root = _open_tmp_store(zarr_path)
    _create_raw(root)
    _complete_run(root, "detect_runs", "detect_001")

    rc, payload = _run_json(capsys, "artifacts", str(zarr_path))

    assert rc == palette.EXIT_OK
    assert payload["status"] == "ok"
    assert payload["inventory"]["zarr_path"] == str(zarr_path.resolve())
    assert payload["inventory"]["run_count"] == 1

    direct = palette.artifacts(palette.ArtifactsRequest(recording=zarr_path))
    direct["provenance"].pop("generated_at_utc")
    payload["provenance"].pop("generated_at_utc")
    assert payload == json.loads(json.dumps(direct, default=str))


def test_artifacts_cli_text_summary(tmp_path, capsys) -> None:
    zarr_path = tmp_path / "inventory_text_training.zarr"
    root = _open_tmp_store(zarr_path)
    _create_raw(root)
    _complete_run(root, "detect_runs", "detect_001")

    rc = palette.main(["artifacts", str(zarr_path)])

    out = capsys.readouterr().out
    assert rc == palette.EXIT_OK
    assert f"zarr_path: {zarr_path.resolve()}" in out
    assert "runs: 1" in out
    assert "detect_runs: 1 runs, resolved=detect_001" in out


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
