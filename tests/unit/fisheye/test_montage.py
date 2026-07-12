from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from PIL import Image

from fisheye.montage import (
    MontageLayout,
    PLOT_PROFILES,
    RegistryRecording,
    compose_visualization_montage,
    query_registry_recordings,
)


def _write_registry(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE dataset_context_current (
                dataset_id TEXT,
                recording_id TEXT,
                zarr_path TEXT,
                protocol_name TEXT,
                arena_id TEXT,
                recording_started_utc TEXT,
                zarr_use TEXT,
                dataset_status TEXT
            )
            """
        )
        conn.executemany(
            "INSERT INTO dataset_context_current VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    "analysis_b",
                    "rec_b",
                    "/tmp/rec_b_analysis.zarr",
                    "RedScare",
                    "arena_2",
                    "2026-06-23T17:00:00Z",
                    "analysis",
                    "active",
                ),
                (
                    "analysis_a",
                    "rec_a",
                    "/tmp/rec_a_analysis.zarr",
                    "RedScare",
                    "arena_1",
                    "2026-06-23T16:00:00Z",
                    "analysis",
                    "active",
                ),
                (
                    "training_a",
                    "rec_a",
                    "/tmp/rec_a_training.zarr",
                    "RedScare",
                    "arena_1",
                    "2026-06-23T16:00:00Z",
                    "training",
                    "active",
                ),
                (
                    "analysis_other",
                    "rec_other",
                    "/tmp/rec_other_analysis.zarr",
                    "OtherProtocol",
                    "arena_1",
                    "2026-06-23T15:00:00Z",
                    "analysis",
                    "active",
                ),
            ],
        )
        conn.execute(
            """
            CREATE TABLE recording_chasers (
                dataset_id TEXT,
                recording_id TEXT,
                stimulus_run_id TEXT,
                chaser_index INTEGER,
                behavior_class TEXT
            )
            """
        )
        conn.executemany(
            "INSERT INTO recording_chasers VALUES (?, ?, ?, ?, ?)",
            [
                ("analysis_a", "rec_a", "stimulus_1", 0, "aggressive"),
                ("analysis_a", "rec_a", "stimulus_1", 1, "inert"),
                ("analysis_b", "rec_b", "stimulus_1", 0, "aggressive"),
            ],
        )
        conn.execute(
            """
            CREATE VIEW recording_chaser_runs AS
            SELECT dataset_id, recording_id, stimulus_run_id, COUNT(*) AS chaser_count
            FROM recording_chasers
            GROUP BY dataset_id, recording_id, stimulus_run_id
            """
        )


def test_query_registry_recordings_filters_protocol_and_orders(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _write_registry(registry_path)

    rows = query_registry_recordings(registry_path, protocol_name="redscare")

    assert [row.recording_id for row in rows] == ["rec_a", "rec_b"]
    assert all(row.protocol_name == "RedScare" for row in rows)
    assert all(row.zarr_path.name.endswith("_analysis.zarr") for row in rows)
    assert rows[0].chaser_behaviors == ("aggressive", "inert")
    assert rows[0].chaser_count == 2


def test_query_registry_recordings_collapses_dataset_aliases_for_same_path(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _write_registry(registry_path)
    with sqlite3.connect(registry_path) as conn:
        conn.execute(
            "INSERT INTO dataset_context_current VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "analysis_a_alias",
                "rec_a",
                "/tmp/rec_a_analysis.zarr",
                "RedScare",
                "arena_1",
                "2026-06-23T16:00:00Z",
                "analysis",
                "active",
            ),
        )
        conn.execute(
            "INSERT INTO recording_chasers VALUES (?, ?, ?, ?, ?)",
            ("analysis_a_alias", "rec_a", "stimulus_1", 0, "aggressive"),
        )

    rows = query_registry_recordings(registry_path, protocol_name="RedScare")

    assert [row.recording_id for row in rows] == ["rec_a", "rec_b"]


def test_query_registry_recordings_filters_variable_chaser_rows(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _write_registry(registry_path)

    pair = query_registry_recordings(
        registry_path,
        protocol_name="RedScare",
        chaser_behaviors=["aggressive", "inert"],
        chaser_count=2,
    )
    single = query_registry_recordings(
        registry_path,
        protocol_name="RedScare",
        chaser_count=1,
    )

    assert [row.recording_id for row in pair] == ["rec_a"]
    assert [row.recording_id for row in single] == ["rec_b"]


def test_query_registry_recordings_requires_explicit_scope(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _write_registry(registry_path)

    with pytest.raises(ValueError, match="cohort selector"):
        query_registry_recordings(registry_path)


def test_plot_profiles_resolve_required_run_paths() -> None:
    profile = PLOT_PROFILES["chaser-distance-distribution"]
    spec = profile.artifact_spec(
        {
            "chaser_distance_run": "distance_run",
            "detection_occupancy_run": None,
            "egocentric_component": None,
        }
    )

    assert spec.path == (
        "analysis/chaser_distance_runs/distance_run/"
        "visualizations/chaser_distance_epoch_distribution_png"
    )
    polar_profile = PLOT_PROFILES["egocentric-bearing-polar"]
    polar_spec = polar_profile.artifact_spec(
        {"chaser_distance_run": "distance_run", "egocentric_component": "bearing_run"}
    )
    assert polar_spec.visualization_contract_id == (
        "palette.chaser_egocentric_bearing.pre_post_polar_density.v2"
    )

    with pytest.raises(ValueError, match="--chaser-distance-run"):
        profile.artifact_spec({"chaser_distance_run": None})

    escape_profile = PLOT_PROFILES["fish-escape-outcome-timeline"]
    escape_spec = escape_profile.artifact_spec(
        {
            "chaser_distance_run": "distance_run",
            "escape_freeze_component": "escape_run",
        }
    )
    assert escape_spec.path == (
        "analysis/chaser_distance_runs/distance_run/chaser_escape_freeze/escape_run/"
        "visualizations/escape_freeze_trial_outcome_timeline_png"
    )


def test_compose_visualization_montage_dimensions() -> None:
    recordings = [
        RegistryRecording(
            recording_id=f"rec_{index}",
            zarr_path=Path(f"/tmp/{index}.zarr"),
            dataset_id=f"dataset_{index}",
            protocol_name="RedScare",
            arena_id=f"arena_{index}",
            recording_started_utc=None,
        )
        for index in range(3)
    ]
    layout = MontageLayout(
        columns=2,
        tile_width=120,
        max_image_height=100,
        margin=10,
        gutter=5,
        header_height=50,
        label_height=20,
    )

    montage = compose_visualization_montage(
        title="Example",
        query_label="protocol=RedScare",
        recordings=recordings,
        images=[Image.new("RGB", (100, 80)), None, Image.new("RGB", (80, 100))],
        errors=[None, "missing", None],
        layout=layout,
    )

    assert montage.size == (265, 305)
