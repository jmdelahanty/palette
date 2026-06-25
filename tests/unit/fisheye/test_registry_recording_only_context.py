from __future__ import annotations

import json
from pathlib import Path

import zarr

from fisheye.registry.db import Registry


def _write_context_zarr(
    path: Path,
    *,
    zarr_purpose: str,
    recording_id: str,
    artifact_schema_id: str,
    source_layout: str | None = None,
) -> None:
    root = zarr.open_group(str(path), mode="w")
    root.attrs.update(
        {
            "zarr_purpose": zarr_purpose,
            "session_uuid": recording_id,
            "recording_id": recording_id,
            "recording_name": recording_id,
            "recording_path": str(path.parent.parent),
            "recording_type": "behavior",
            "recording_subtype": "free",
            "behavior_mode": "free",
            "artifact_schema_id": artifact_schema_id,
            "experiment_context_status": "absent",
            "experiment_context_source": "none",
            "stimulus_runs_available": False,
            "experiment_context_status_detail": "Synthetic recording-only fixture has no H5/protocol source.",
        }
    )
    if source_layout is not None:
        root.attrs["source_layout"] = source_layout
        root.attrs["source_frame_index_path"] = "source_frame_index.parquet"
        root.attrs["source_recording_frame_index_path"] = str(path.parent.parent / "recording_frame_index.parquet")
        root.attrs["source_frame_index_schema"] = "palette.training_source_frame_index.v1"


def test_registry_scan_indexes_recording_only_training_and_analysis_context(tmp_path: Path) -> None:
    recording_dir = tmp_path / "recordings" / "synthetic_recording_only"
    zarr_dir = recording_dir / "zarr"
    zarr_dir.mkdir(parents=True)
    training_zarr = zarr_dir / "synthetic_recording_only_training.zarr"
    analysis_zarr = zarr_dir / "synthetic_recording_only_analysis.zarr"

    _write_context_zarr(
        training_zarr,
        zarr_purpose="training",
        recording_id="synthetic_recording_only",
        artifact_schema_id="video_only_v1",
    )
    _write_context_zarr(
        analysis_zarr,
        zarr_purpose="analysis",
        recording_id="synthetic_recording_only",
        artifact_schema_id="recording_analysis_v1",
    )

    registry = Registry(tmp_path / "registry.sqlite")
    try:
        registry.scan_zarr(training_zarr)
        registry.scan_zarr(analysis_zarr)

        rows = registry.conn.execute(
            """
            SELECT
                dataset_id,
                zarr_use,
                recording_id,
                recording_type,
                experiment_context_status,
                experiment_context_source,
                stimulus_runs_available
            FROM dataset_context_current
            WHERE recording_id = ?
            ORDER BY zarr_use;
            """,
            ("synthetic_recording_only",),
        ).fetchall()
        query_rows = registry.query_datasets(
            experiment_context_status="absent",
            experiment_context_source="none",
            stimulus_runs_available=False,
            require_recording=True,
        )
    finally:
        registry.close()

    assert [row["zarr_use"] for row in rows] == ["analysis", "training"]
    assert {row["recording_type"] for row in rows} == {"behavior"}
    assert {row["experiment_context_status"] for row in rows} == {"absent"}
    assert {row["experiment_context_source"] for row in rows} == {"none"}
    assert {row["stimulus_runs_available"] for row in rows} == {0}
    assert {row["zarr_use"] for row in query_rows} == {"analysis", "training"}


def test_registry_scan_exposes_clipped_training_source_metadata(tmp_path: Path) -> None:
    recording_dir = tmp_path / "recordings" / "synthetic_rolling"
    zarr_dir = recording_dir / "zarr"
    zarr_dir.mkdir(parents=True)
    training_zarr = zarr_dir / "synthetic_rolling_clipped_training.zarr"

    _write_context_zarr(
        training_zarr,
        zarr_purpose="training",
        recording_id="synthetic_rolling",
        artifact_schema_id="video_only_v1",
        source_layout="rolling_clips",
    )

    registry = Registry(tmp_path / "registry.sqlite")
    try:
        registry.scan_zarr(training_zarr)
        rows = registry.query_datasets(source_layout="rolling_clips")
    finally:
        registry.close()

    assert len(rows) == 1
    row = rows[0]
    assert row["zarr_use"] == "training"
    assert row["source_layout"] == "rolling_clips"
    assert row["source_frame_index_path"] == "source_frame_index.parquet"
    assert row["source_frame_index_schema"] == "palette.training_source_frame_index.v1"
    assert str(row["source_recording_frame_index_path"]).endswith("recording_frame_index.parquet")


def test_registry_scan_uses_recording_manifest_when_training_root_lacks_context(tmp_path: Path) -> None:
    recording_dir = tmp_path / "recordings" / "2026-06-23T16-01-09Z_arena_1_RedScare"
    zarr_dir = recording_dir / "zarr"
    zarr_dir.mkdir(parents=True)
    training_zarr = zarr_dir / "2026-06-23T16-01-09Z_arena_1_RedScare_training.zarr"
    (recording_dir / "recording_manifest.json").write_text(
        json.dumps(
            {
                "recording_id": "2026-06-23T16-01-09Z_arena_1",
                "session_uuid": "2026-06-23T16-01-09Z_arena_1",
                "recording_name": "2026-06-23T16-01-09Z_arena_1_RedScare",
                "recording_type": "behavior",
                "recording_subtype": "free",
                "behavior_mode": "free",
                "protocol_name": "RedScare",
                "rig_id": "omnifin0",
                "arena_id": "1",
                "camera_id": "2010093",
                "canvas_name": "arena_1",
                "dish_design": "palm1",
            }
        ),
        encoding="utf-8",
    )

    root = zarr.open_group(str(training_zarr), mode="w", zarr_format=3)
    root.attrs.update(
        {
            "zarr_use": "training",
            "zarr_purpose": "training",
        }
    )
    root.create_group("raw_video")

    registry = Registry(tmp_path / "registry.sqlite")
    try:
        dataset_id = registry.scan_zarr(training_zarr)
        recording = registry.conn.execute(
            """
            SELECT
                recording_id,
                recording_name,
                recording_type,
                recording_subtype,
                behavior_mode,
                protocol_name,
                rig_id,
                arena_id,
                camera_id,
                dish_design
            FROM recordings
            WHERE recording_id = ?;
            """,
            ("2026-06-23T16-01-09Z_arena_1",),
        ).fetchone()
        dataset = registry.conn.execute(
            """
            SELECT recording_id, zarr_use, artifact_kind
            FROM datasets
            WHERE dataset_id = ?;
            """,
            (dataset_id,),
        ).fetchone()
    finally:
        registry.close()

    assert dataset_id.startswith("2026-06-23T16-01-09Z_arena_1:z")
    assert recording is not None
    assert recording["recording_name"] == "2026-06-23T16-01-09Z_arena_1_RedScare"
    assert recording["recording_type"] == "behavior"
    assert recording["recording_subtype"] == "free"
    assert recording["behavior_mode"] == "free"
    assert recording["protocol_name"] == "RedScare"
    assert recording["rig_id"] == "omnifin0"
    assert recording["arena_id"] == "1"
    assert recording["camera_id"] == "2010093"
    assert recording["dish_design"] == "palm1"
    assert dataset["recording_id"] == "2026-06-23T16-01-09Z_arena_1"
    assert dataset["zarr_use"] == "training"
    assert dataset["artifact_kind"] == "source_recording"


def test_registry_scan_ignores_empty_zarr_group_stubs(tmp_path: Path) -> None:
    stub_zarr = tmp_path / "recordings" / "synthetic" / "zarr" / "aborted_training.zarr"
    zarr.open_group(str(stub_zarr), mode="w", zarr_format=3)

    registry = Registry(tmp_path / "registry.sqlite")
    try:
        dataset_id = registry.scan_zarr(stub_zarr)
        rows = registry.conn.execute("SELECT dataset_id FROM datasets;").fetchall()
    finally:
        registry.close()

    assert dataset_id is None
    assert rows == []
