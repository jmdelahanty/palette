import json
import sqlite3
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from fisheye.utils.backfill_clipped_analysis_metadata import (
    backfill_clipped_analysis_metadata,
)


def _write_zarr_json(path: Path, attrs: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "zarr.json").write_text(
        json.dumps({"zarr_format": 3, "node_type": "group", "attributes": attrs}),
        encoding="utf-8",
    )


def _read_zarr_attrs(path: Path) -> dict:
    return json.loads((path / "zarr.json").read_text(encoding="utf-8"))["attributes"]


def _write_frame_index(path: Path, *, old_root: Path) -> None:
    pq.write_table(
        pa.table(
            {
                "recording_folder": [str(old_root)],
                "camera_serial": ["2010095"],
                "clip_id": ["clip_000000"],
                "clip_local_frame_index": [0],
                "recording_frame_id": [1],
                "video_path": [str(old_root / "clips" / "clip_000000" / "Cam2010095.mp4")],
                "metadata_path": [str(old_root / "clips" / "clip_000000" / "Cam2010095_meta.csv")],
            }
        ),
        path,
    )


def _write_manifest(recording_root: Path, *, old_root: Path) -> None:
    (recording_root / "recording_frame_index_manifest.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "schema_version": "palette.recording_frame_index_manifest.v1",
                "frame_index_schema_version": "palette.recording_frame_index.v1",
                "recording_id": "sleepyfish_cam2010095",
                "session_id": "sleepyfish_cam2010095",
                "source_layout": "rolling_clips",
                "recording_frame_index_path": str(old_root / "recording_frame_index.parquet"),
                "row_count": 1,
                "recording_frame_id_min": 1,
                "recording_frame_id_max": 1,
            }
        ),
        encoding="utf-8",
    )


def _write_registry(path: Path, zarr_path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE datasets (
                dataset_id TEXT,
                recording_id TEXT,
                zarr_use TEXT,
                source_layout TEXT,
                source_recording_frame_index_path TEXT,
                source_frame_index_schema TEXT,
                zarr_path TEXT,
                last_seen_utc TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO datasets (
                dataset_id, recording_id, zarr_use, source_layout,
                source_recording_frame_index_path, source_frame_index_schema,
                zarr_path, last_seen_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "sleepyfish_cam2010095:ztest",
                None,
                "analysis",
                None,
                None,
                None,
                str(zarr_path.resolve()),
                "2026-01-01T00:00:00Z",
            ),
        )
        conn.commit()


def _make_archive(tmp_path: Path) -> tuple[Path, Path, Path]:
    recording_root = tmp_path / "sleepyfish_cam2010095"
    old_root = Path("/nvme1/recordings/sleepyfish_cam2010095")
    recording_root.mkdir()
    frame_index_path = recording_root / "recording_frame_index.parquet"
    _write_frame_index(frame_index_path, old_root=old_root)
    _write_manifest(recording_root, old_root=old_root)
    zarr_path = recording_root / "zarr" / "sleepyfish_cam2010095_analysis.zarr"
    _write_zarr_json(zarr_path, {"zarr_purpose": "production"})
    return zarr_path, recording_root, old_root


def test_backfill_clipped_analysis_metadata_plans_root_and_registry_changes(tmp_path: Path):
    zarr_path, _recording_root, _old_root = _make_archive(tmp_path)
    registry = tmp_path / "registry.sqlite"
    _write_registry(registry, zarr_path)

    result = backfill_clipped_analysis_metadata(zarr_path, registry_path=registry)

    assert result["status"] == "planned"
    assert result["attr_changes"]["zarr_purpose"]["wanted"] == "analysis"
    assert result["attr_changes"]["source_layout"]["wanted"] == "rolling_clips"
    assert result["registry"]["changes"]["source_layout"]["wanted"] == "rolling_clips"


def test_backfill_clipped_analysis_metadata_applies_root_registry_and_path_rewrite(tmp_path: Path):
    zarr_path, recording_root, old_root = _make_archive(tmp_path)
    registry = tmp_path / "registry.sqlite"
    _write_registry(registry, zarr_path)

    result = backfill_clipped_analysis_metadata(
        zarr_path,
        registry_path=registry,
        rewrite_frame_index_paths=True,
        old_root=str(old_root),
        new_root=str(recording_root),
        apply=True,
    )

    assert result["status"] == "applied"
    attrs = _read_zarr_attrs(zarr_path)
    assert attrs["zarr_purpose"] == "analysis"
    assert attrs["recording_id"] == "sleepyfish_cam2010095"
    assert attrs["source_layout"] == "rolling_clips"
    assert attrs["source_recording_frame_index_path"] == str(recording_root / "recording_frame_index.parquet")

    table = pq.read_table(recording_root / "recording_frame_index.parquet")
    assert table["video_path"][0].as_py().startswith(str(recording_root))
    assert table["metadata_path"][0].as_py().startswith(str(recording_root))

    with sqlite3.connect(registry) as conn:
        row = conn.execute(
            """
            SELECT recording_id, source_layout, source_recording_frame_index_path,
                   source_frame_index_schema
            FROM datasets
            WHERE zarr_path = ?
            """,
            (str(zarr_path.resolve()),),
        ).fetchone()
    assert row == (
        "sleepyfish_cam2010095",
        "rolling_clips",
        str(recording_root / "recording_frame_index.parquet"),
        "palette.recording_frame_index.v1",
    )
