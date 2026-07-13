from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from fisheye.registry.db import Registry
from fisheye.registry.extractors.stimulus_metadata import StimulusMetadataExtraction
from fisheye.registry import stimulus_metadata_backfill as backfill


def _extraction(*, mode: str = "CHASER", latest: int = 1) -> StimulusMetadataExtraction:
    return StimulusMetadataExtraction(
        protocols=(
            {
                "protocol_hash": "protocol_hash",
                "protocol_name": "Mixed chaser",
                "step_count": 1,
                "protocol_json": '{"steps":[]}',
                "definition_source": "stimulus_protocol_json",
                "extracted_utc": "2026-07-13T00:00:00Z",
            },
        ),
        protocol_steps=(
            {
                "protocol_hash": "protocol_hash",
                "step_index": 0,
                "step_name": "chase",
                "stimulus_mode": mode,
                "duration_s": 30.0,
                "parameters_json": None,
                "step_definition_json": "{}",
            },
        ),
        recording_runs=(
            {
                "recording_id": "recording_1",
                "stimulus_run_id": "stimulus_1",
                "protocol_hash": "protocol_hash",
                "protocol_name": "Mixed chaser",
                "is_latest": latest,
                "step_count": 1,
                "source_path": "analysis/stimulus_runs/stimulus_1",
                "source_metadata_sha256": "metadata_hash",
                "source_zarr_path": "/groups/recording_1_analysis.zarr",
                "extracted_utc": "2026-07-13T00:00:00Z",
            },
        ),
        recording_steps=(
            {
                "stimulus_run_id": "stimulus_1",
                "step_index": 0,
                "step_name": "chase",
                "stimulus_mode": mode,
                "start_camera_frame": 0,
                "end_camera_frame": 99,
                "duration_s": 30.0,
                "step_attrs_json": "{}",
            },
        ),
        recording_modes=(
            {
                "stimulus_run_id": "stimulus_1",
                "stimulus_mode": mode,
                "step_count": 1,
                "total_duration_s": 30.0,
            },
        ),
    )


def test_select_analysis_datasets_is_read_only_and_recording_owned(tmp_path: Path) -> None:
    registry = tmp_path / "registry.sqlite"
    with sqlite3.connect(registry) as conn:
        conn.execute(
            """
            CREATE TABLE dataset_context_current (
                dataset_id TEXT,
                recording_id TEXT,
                zarr_path TEXT,
                protocol_name TEXT,
                zarr_use TEXT,
                dataset_status TEXT
            )
            """
        )
        conn.executemany(
            "INSERT INTO dataset_context_current VALUES (?, ?, ?, ?, ?, ?)",
            [
                ("analysis_1", "recording_1", "/groups/one.zarr", "RedScare", "analysis", "active"),
                ("analysis_2", None, "/groups/unowned.zarr", None, "analysis", "active"),
                ("training_1", "recording_1", "/groups/train.zarr", None, "training", "active"),
            ],
        )

    rows = backfill.select_analysis_datasets(registry, all_recordings=True)

    assert [row["dataset_id"] for row in rows] == ["analysis_1"]
    with sqlite3.connect(registry) as conn:
        assert conn.execute("SELECT COUNT(*) FROM dataset_context_current").fetchone()[0] == 3


def test_build_census_reports_latest_normalized_modes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backfill, "_open_root", lambda _path: object())
    monkeypatch.setattr(
        backfill,
        "extract_stimulus_metadata",
        lambda _root, **_kwargs: _extraction(),
    )
    datasets = [
        {
            "dataset_id": "analysis_1",
            "recording_id": "recording_1",
            "zarr_path": "/groups/recording_1_analysis.zarr",
            "protocol_name": "RedScare",
        }
    ]

    census = backfill.build_stimulus_metadata_census(datasets)

    assert census["issue_count"] == 0
    assert census["datasets_with_stimulus_count"] == 1
    assert census["latest_mode_run_counts"] == {"CHASER": 1}
    assert census["latest_mode_dataset_counts"] == {"CHASER": 1}


def test_build_census_flags_missing_latest_and_unknown_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(backfill, "_open_root", lambda _path: object())
    monkeypatch.setattr(
        backfill,
        "extract_stimulus_metadata",
        lambda _root, **_kwargs: _extraction(mode="UNKNOWN", latest=0),
    )
    datasets = [
        {
            "dataset_id": "analysis_1",
            "recording_id": "recording_1",
            "zarr_path": "/groups/recording_1_analysis.zarr",
            "protocol_name": None,
        }
    ]

    census = backfill.build_stimulus_metadata_census(datasets)

    assert {issue["reason"] for issue in census["issues"]} == {
        "latest_stimulus_run_count",
        "unknown_stimulus_mode",
    }


def test_apply_census_replaces_only_stimulus_tables(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.sqlite"
    registry = Registry(registry_path)
    try:
        registry.upsert_dataset(
            "analysis_1",
            session_uuid="recording_1",
            recording_id="recording_1",
            zarr_path=tmp_path / "recording_1_analysis.zarr",
            zarr_use="analysis",
        )
    finally:
        registry.close()
    extraction = _extraction()
    census = {
        "issue_count": 0,
        "datasets": [
            {
                "dataset_id": "analysis_1",
                "read_status": "ok",
                "protocols": [dict(row) for row in extraction.protocols],
                "protocol_steps": [dict(row) for row in extraction.protocol_steps],
                "recording_runs": [dict(row) for row in extraction.recording_runs],
                "recording_steps": [dict(row) for row in extraction.recording_steps],
                "recording_modes": [dict(row) for row in extraction.recording_modes],
            }
        ],
    }

    result = backfill.apply_stimulus_metadata_census(registry_path, census)

    assert result == {"applied_dataset_count": 1, "skipped_dataset_count": 0}
    with sqlite3.connect(registry_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM recording_stimulus_runs").fetchone()[0] == 1
        assert conn.execute(
            "SELECT stimulus_mode FROM recording_stimulus_mode_counts WHERE is_latest = 1"
        ).fetchone()[0] == "CHASER"


def test_apply_refuses_issueful_census_without_override(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.sqlite"
    Registry(registry_path).close()

    with pytest.raises(ValueError, match="Census contains 1 issue"):
        backfill.apply_stimulus_metadata_census(
            registry_path,
            {"issue_count": 1, "datasets": []},
        )


def test_cli_apply_requires_backup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    registry_path = tmp_path / "registry.sqlite"
    Registry(registry_path).close()
    monkeypatch.setattr(backfill, "select_analysis_datasets", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        backfill,
        "build_stimulus_metadata_census",
        lambda _datasets: {"issue_count": 0, "datasets": []},
    )

    with pytest.raises(ValueError, match="--apply requires --backup"):
        backfill.main(
            [
                "--registry",
                str(registry_path),
                "--all-recordings",
                "--apply",
            ]
        )
