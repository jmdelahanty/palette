from __future__ import annotations

from pathlib import Path

import zarr

from fisheye.registry.db import Registry


def _write_context_zarr(
    path: Path,
    *,
    zarr_purpose: str,
    recording_id: str,
    artifact_schema_id: str,
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
