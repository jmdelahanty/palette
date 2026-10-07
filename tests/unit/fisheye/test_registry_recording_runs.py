"""Registry migration 075: run-grouping views over ``recordings.session_uuid``.

One Orange run records several cameras together, and every camera recording
carries the run's session id as ``recordings.session_uuid``. ``recording_runs``
summarizes each run and ``recording_run_members`` lists each run's recordings
with their current (non-missing) analysis dataset.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from fisheye.registry.db import Registry
from fisheye.registry.migrations import MIGRATION_METHODS
from tests.unit.fisheye.registry_test_fixtures import registry_from_empty_template

RUN_A = "orange-run-a"
RUN_B = "orange-run-b"
LEGACY_RUN = "2026-01-28T22-42-59Z_arena_1"
CAMERAS = ("cam3", "cam1", "cam4", "cam2")  # deliberately unsorted
RUN_VIEWS = ("recording_runs", "recording_run_members")


def _seed_run(
    registry: Registry,
    tmp_path: Path,
    session_uuid: str,
    *,
    started: str,
    intent: str,
    origin: str,
) -> None:
    for index, camera_id in enumerate(CAMERAS):
        recording_id = f"{session_uuid}-{camera_id}"
        registry.upsert_recording(
            recording_id=recording_id,
            session_uuid=session_uuid,
            recording_name=f"{session_uuid}_{camera_id}",
            started_utc=f"{started}:0{index}Z",
            recording_type="behavior",
            behavior_mode="free",
            camera_id=camera_id,
            recording_intent=intent,
            data_origin=origin,
        )
        registry.upsert_dataset(
            f"{recording_id}:analysis",
            session_uuid=session_uuid,
            zarr_path=tmp_path / "recordings" / recording_id / "analysis.zarr",
            recording_id=recording_id,
            zarr_use="analysis",
        )
        # A training artifact for the same recording is never an analysis dataset.
        registry.upsert_dataset(
            f"{recording_id}:training",
            session_uuid=session_uuid,
            zarr_path=tmp_path / "recordings" / recording_id / "training.zarr",
            recording_id=recording_id,
            zarr_use="training",
        )


def _seeded_registry(tmp_path: Path) -> Registry:
    registry = registry_from_empty_template(tmp_path / "registry.sqlite")
    _seed_run(
        registry,
        tmp_path,
        RUN_A,
        started="2026-10-01T10:00",
        intent="stimulus_experiment",
        origin="acquired",
    )
    _seed_run(
        registry,
        tmp_path,
        RUN_B,
        started="2026-10-02T11:00",
        intent="stimulus_experiment",
        origin="acquired",
    )
    # Historical per-arena recording: no camera, start time, intent or origin.
    registry.upsert_recording(recording_id="legacy-rec", session_uuid=LEGACY_RUN)
    # A recording without a session id belongs to no run.
    registry.upsert_recording(recording_id="orphan-rec", camera_id="cam9")
    registry.conn.commit()
    return registry


def _mark_missing(registry: Registry, dataset_id: str) -> None:
    registry.conn.execute(
        "UPDATE datasets SET status = 'missing' WHERE dataset_id = ?;", (dataset_id,)
    )
    registry.conn.commit()


def _runs(registry: Registry) -> dict[str, dict]:
    return {
        row["session_uuid"]: dict(row)
        for row in registry.conn.execute("SELECT * FROM recording_runs;").fetchall()
    }


def _members(registry: Registry, session_uuid: str) -> list[dict]:
    return [
        dict(row)
        for row in registry.conn.execute(
            "SELECT * FROM recording_run_members WHERE session_uuid = ? "
            "ORDER BY camera_id, recording_id;",
            (session_uuid,),
        ).fetchall()
    ]


def _view_names(conn) -> set[str]:
    return {
        str(row[0])
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='view';")
    }


def test_migration_075_is_registered_last() -> None:
    assert MIGRATION_METHODS[-1] == (
        75, "recording_run_views", "_migration_075_recording_run_views"
    )


def test_recording_runs_summarizes_each_run(tmp_path: Path) -> None:
    registry = _seeded_registry(tmp_path)
    try:
        runs = _runs(registry)
        assert set(runs) == {RUN_A, RUN_B, LEGACY_RUN}
        for session_uuid, day in ((RUN_A, "2026-10-01T10:00"), (RUN_B, "2026-10-02T11:00")):
            assert runs[session_uuid] == {
                "session_uuid": session_uuid,
                "camera_count": 4,
                "camera_ids": "cam1,cam2,cam3,cam4",
                "recording_count": 4,
                "first_started_utc": f"{day}:00Z",
                "last_started_utc": f"{day}:03Z",
                "recording_intents": "stimulus_experiment",
                "recording_types": "behavior",
                "data_origins": "acquired",
                "analysis_dataset_count": 4,
            }
        assert runs[LEGACY_RUN] == {
            "session_uuid": LEGACY_RUN,
            "camera_count": 0,
            "camera_ids": None,
            "recording_count": 1,
            "first_started_utc": None,
            "last_started_utc": None,
            "recording_intents": None,
            "recording_types": None,
            "data_origins": None,
            "analysis_dataset_count": 0,
        }
    finally:
        registry.close()


def test_recording_runs_joins_distinct_values_sorted(tmp_path: Path) -> None:
    registry = _seeded_registry(tmp_path)
    try:
        registry.upsert_recording(
            recording_id=f"{RUN_A}-cam2",
            recording_intent="calibration",
            data_origin="synthetic",
        )
        registry.conn.commit()
        run = _runs(registry)[RUN_A]
        assert run["recording_intents"] == "calibration,stimulus_experiment"
        assert run["data_origins"] == "acquired,synthetic"
        assert run["camera_ids"] == "cam1,cam2,cam3,cam4"
    finally:
        registry.close()


def test_recording_run_members_lists_each_camera_with_its_analysis_dataset(
    tmp_path: Path,
) -> None:
    registry = _seeded_registry(tmp_path)
    try:
        for session_uuid in (RUN_A, RUN_B):
            members = _members(registry, session_uuid)
            assert [m["camera_id"] for m in members] == ["cam1", "cam2", "cam3", "cam4"]
            for member in members:
                recording_id = f"{session_uuid}-{member['camera_id']}"
                assert member["recording_id"] == recording_id
                assert member["recording_name"] == f"{session_uuid}_{member['camera_id']}"
                assert member["recording_intent"] == "stimulus_experiment"
                assert member["data_origin"] == "acquired"
                assert member["started_utc"].startswith("2026-10-0")
                assert member["dataset_id"] == f"{recording_id}:analysis"
                assert member["zarr_path"] == str(
                    tmp_path / "recordings" / recording_id / "analysis.zarr"
                )
                assert member["analysis_dataset_count"] == 1

        (legacy,) = _members(registry, LEGACY_RUN)
        assert legacy["recording_id"] == "legacy-rec"
        assert legacy["camera_id"] is None
        assert legacy["dataset_id"] is None
        assert legacy["zarr_path"] is None
        assert legacy["analysis_dataset_count"] == 0

        member_ids = {
            row[0]
            for row in registry.conn.execute(
                "SELECT recording_id FROM recording_run_members;"
            )
        }
        assert "orphan-rec" not in member_ids
        assert len(member_ids) == 9
    finally:
        registry.close()


def test_missing_analysis_dataset_is_hidden_and_not_counted(tmp_path: Path) -> None:
    registry = _seeded_registry(tmp_path)
    try:
        _mark_missing(registry, f"{RUN_A}-cam1:analysis")
        assert _runs(registry)[RUN_A]["analysis_dataset_count"] == 3
        assert _runs(registry)[RUN_B]["analysis_dataset_count"] == 4
        cam1 = _members(registry, RUN_A)[0]
        assert cam1["camera_id"] == "cam1"
        assert cam1["dataset_id"] is None
        assert cam1["zarr_path"] is None
        assert cam1["analysis_dataset_count"] == 0
        # The recording itself remains a run member.
        assert _runs(registry)[RUN_A]["recording_count"] == 4
    finally:
        registry.close()


def test_current_analysis_dataset_skips_missing_rows_and_counts_live_ones(
    tmp_path: Path,
) -> None:
    registry = _seeded_registry(tmp_path)
    try:
        recording_id = f"{RUN_B}-cam2"
        registry.upsert_dataset(
            f"{recording_id}:analysis-rerun",
            session_uuid=RUN_B,
            zarr_path=tmp_path / "rerun" / recording_id / "analysis.zarr",
            recording_id=recording_id,
            zarr_use="analysis",
        )
        registry.conn.execute(
            "UPDATE datasets SET last_seen_utc = '2099-01-01T00:00:00Z' "
            "WHERE dataset_id = ?;",
            (f"{recording_id}:analysis-rerun",),
        )
        registry.conn.commit()
        cam2 = _members(registry, RUN_B)[1]
        assert cam2["dataset_id"] == f"{recording_id}:analysis-rerun"
        assert cam2["analysis_dataset_count"] == 2
        assert _runs(registry)[RUN_B]["analysis_dataset_count"] == 5

        _mark_missing(registry, f"{recording_id}:analysis-rerun")
        cam2 = _members(registry, RUN_B)[1]
        assert cam2["dataset_id"] == f"{recording_id}:analysis"
        assert cam2["analysis_dataset_count"] == 1
        assert _runs(registry)[RUN_B]["analysis_dataset_count"] == 4
    finally:
        registry.close()


def test_migration_075_upgrades_an_existing_v74_registry(tmp_path: Path) -> None:
    registry_path = tmp_path / "upgrade.sqlite"
    _seeded_registry(tmp_path / "seed").close()
    seeded = tmp_path / "seed" / "registry.sqlite"
    registry_path.write_bytes(seeded.read_bytes())
    with sqlite3.connect(registry_path) as conn:
        for view in reversed(RUN_VIEWS):
            conn.execute(f"DROP VIEW IF EXISTS {view};")
        conn.execute("DELETE FROM schema_version WHERE version >= 75;")
        conn.execute("PRAGMA user_version = 74;")
        conn.commit()
        assert not set(RUN_VIEWS) & _view_names(conn)

    upgraded = Registry(registry_path)
    try:
        assert upgraded._current_schema_version() == 75
        assert set(RUN_VIEWS) <= _view_names(upgraded.conn)
        assert _runs(upgraded)[RUN_A]["camera_count"] == 4
        assert upgraded.conn.execute("PRAGMA foreign_key_check;").fetchall() == []
        assert upgraded.conn.execute("PRAGMA integrity_check;").fetchone()[0] == "ok"
    finally:
        upgraded.close()


def test_migration_075_is_idempotent(tmp_path: Path) -> None:
    registry = _seeded_registry(tmp_path)
    try:
        before = _runs(registry)
        registry._migration_075_recording_run_views()
        registry._migration_075_recording_run_views()
        registry.conn.commit()
        assert _runs(registry) == before
        assert set(RUN_VIEWS) <= _view_names(registry.conn)
    finally:
        registry.close()

    reopened = Registry(tmp_path / "registry.sqlite")
    try:
        versions = [
            row[0]
            for row in reopened.conn.execute(
                "SELECT version FROM schema_version WHERE version = 75;"
            )
        ]
        assert versions == [75]
        assert _runs(reopened) == before
    finally:
        reopened.close()


def _browser_queries() -> dict[str, str]:
    import yaml

    metadata_path = (
        Path(__file__).resolve().parents[3]
        / "docs"
        / "registry_browser"
        / "datasette-metadata.yaml"
    )
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    queries = metadata["databases"]["palette_registry"]["queries"]
    return {name: entry["sql"] for name, entry in queries.items()}


def test_registry_browser_run_queries_execute(tmp_path: Path) -> None:
    queries = _browser_queries()
    registry = _seeded_registry(tmp_path)
    try:
        registry.upsert_recording(
            recording_id=f"{RUN_B}-cam5", session_uuid=RUN_B, camera_id="cam5"
        )
        registry.conn.commit()

        in_run = registry.conn.execute(
            queries["recordings_in_run"], {"session_uuid": RUN_A}
        ).fetchall()
        assert [row["camera_id"] for row in in_run] == ["cam1", "cam2", "cam3", "cam4"]
        assert in_run[0]["dataset_id"] == f"{RUN_A}-cam1:analysis"

        compared = registry.conn.execute(
            queries["compare_two_runs"], {"run_a": RUN_A, "run_b": RUN_B}
        ).fetchall()
        assert [row["camera_id"] for row in compared] == [
            "cam1", "cam2", "cam3", "cam4", "cam5"
        ]
        assert compared[0]["run_a_dataset_id"] == f"{RUN_A}-cam1:analysis"
        assert compared[0]["run_b_dataset_id"] == f"{RUN_B}-cam1:analysis"
        assert compared[-1]["run_a_recording_id"] is None
        assert compared[-1]["run_b_recording_id"] == f"{RUN_B}-cam5"
    finally:
        registry.close()
