import json
import sqlite3
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from fisheye.utils.audit_clipped_collection_cache_readiness import (
    audit_clipped_collection_cache_readiness,
)


def _write_zarr_json(path: Path, attrs: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "zarr.json").write_text(
        json.dumps({"zarr_format": 3, "node_type": "group", "attributes": attrs}),
        encoding="utf-8",
    )


def _write_frame_index(path: Path, video_path: Path) -> None:
    pq.write_table(
        pa.table(
            {
                "camera_serial": ["2010095"],
                "clip_id": ["clip_000000"],
                "clip_local_frame_index": [0],
                "recording_frame_id": [1],
                "video_path": [str(video_path)],
            }
        ),
        path,
    )


def _write_registry(path: Path, zarr_path: Path, *, source_layout: str = "rolling_clips") -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            CREATE TABLE datasets (
                dataset_id TEXT,
                recording_id TEXT,
                zarr_use TEXT,
                source_layout TEXT,
                source_recording_frame_index_path TEXT,
                source_frame_index_schema TEXT,
                zarr_path TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO datasets (
                dataset_id, recording_id, zarr_use, source_layout,
                source_recording_frame_index_path, source_frame_index_schema, zarr_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "sleepyfish_cam2010095",
                "sleepyfish_cam2010095",
                "analysis",
                source_layout,
                str(zarr_path.parent.parent / "recording_frame_index.parquet"),
                "palette.recording_frame_index.v1",
                str(zarr_path.resolve()),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _make_ready_archive(tmp_path: Path) -> tuple[Path, Path]:
    recording_root = tmp_path / "sleepyfish_cam2010095"
    zarr_path = recording_root / "zarr" / "sleepyfish_cam2010095_analysis.zarr"
    video_path = recording_root / "clips" / "clip_000000" / "Cam2010095.mp4"
    video_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"fake")
    frame_index_path = recording_root / "recording_frame_index.parquet"
    _write_frame_index(frame_index_path, video_path)
    _write_zarr_json(
        zarr_path,
        {
            "recording_id": "sleepyfish_cam2010095",
            "zarr_use": "analysis",
            "source_layout": "rolling_clips",
            "source_recording_frame_index_path": str(frame_index_path),
            "source_frame_index_schema": "palette.recording_frame_index.v1",
        },
    )
    _write_zarr_json(
        zarr_path / "refined_detect_runs",
        {"latest_collection": "sleepyfish_collection_01"},
    )
    _write_zarr_json(
        zarr_path / "experiment_index" / "finalized_runs" / "sleepyfish_collection_01",
        {
            "selected_runs": [
                {
                    "work_unit_id": "clip_000000_cam2010095",
                    "camera_serial": "2010095",
                    "clip_id": "clip_000000",
                    "clip_index": 0,
                    "refined_detect_run": "refined_clip_000000",
                    "refined_group_path": "clips/clip_000000/cameras/2010095/refined_detect_runs/refined_clip_000000",
                    "source": {"video_path": str(video_path)},
                }
            ]
        },
    )
    return zarr_path, video_path


def test_audit_clipped_collection_cache_readiness_ok(tmp_path: Path):
    zarr_path, _video_path = _make_ready_archive(tmp_path)
    registry = tmp_path / "registry.sqlite"
    _write_registry(registry, zarr_path)

    payload = audit_clipped_collection_cache_readiness(
        zarr_path,
        registry=registry,
    )

    assert payload["status"] == "ok"
    assert payload["blocker_count"] == 0
    assert payload["warning_count"] == 0
    assert payload["collection"]["collection_id"] == "sleepyfish_collection_01"
    assert payload["collection"]["clip_count"] == 1
    assert payload["recording_frame_index"]["row_count"] == 1
    assert len(payload["registry"]["matched_dataset_rows"]) == 1


def test_audit_clipped_collection_cache_readiness_blocks_missing_selected_video(tmp_path: Path):
    zarr_path, video_path = _make_ready_archive(tmp_path)
    video_path.unlink()

    payload = audit_clipped_collection_cache_readiness(zarr_path)

    assert payload["status"] == "blocked"
    assert any(
        issue["code"] == "selected_runs_source_video_path_not_found"
        for issue in payload["issues"]
    )


def test_audit_clipped_collection_cache_readiness_warns_registry_not_clipped(tmp_path: Path):
    zarr_path, _video_path = _make_ready_archive(tmp_path)
    registry = tmp_path / "registry.sqlite"
    _write_registry(registry, zarr_path, source_layout="")

    payload = audit_clipped_collection_cache_readiness(
        zarr_path,
        registry=registry,
    )

    assert payload["status"] == "warning"
    assert any(
        issue["code"] == "registry_source_layout_not_clipped"
        for issue in payload["issues"]
    )
