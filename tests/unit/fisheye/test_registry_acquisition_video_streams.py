from __future__ import annotations

import json
from pathlib import Path

import zarr

from fisheye.registry.db import Registry
from fisheye.shared.acquisition_video_streams import write_acquisition_video_stream_inventory


def _write_recording_with_inventory(tmp_path: Path) -> Path:
    recording_dir = tmp_path / "2026-06-14T21-12-08Z_arena_1_GoodCopBadCop"
    (recording_dir / "cams").mkdir(parents=True)
    crop_dir = recording_dir / "derived" / "external_crop_recorder"
    crop_dir.mkdir(parents=True)
    (recording_dir / "cams" / "Cam2010093_sample.mp4").write_bytes(b"full")
    (crop_dir / "Cam2010093_sample_crop_external.mp4").write_bytes(b"crop")
    (crop_dir / "Cam2010093_sample_crop_meta.csv").write_text(
        "recording_frame_id,has_detection,blank_frame,crop_x,crop_y,crop_w,crop_h,"
        "detection_x,detection_y,detection_w,detection_h,crop_video_frame_index\n"
        "1,true,false,10,20,256,256,100,120,20,30,0\n"
        "2,false,true,0,0,0,0,0,0,0,0,1\n",
        encoding="utf-8",
    )
    (crop_dir / "Cam2010093_sample_crop_external_summary.json").write_text(
        json.dumps({"status": "completed", "frames_encoded": 2, "frames_dropped": 0}),
        encoding="utf-8",
    )
    manifest = {
        "recording_id": "2026-06-14T21-12-08Z_arena_1_GoodCopBadCop",
        "recording_name": "2026-06-14T21-12-08Z_arena_1_GoodCopBadCop",
        "session_uuid": "2026-06-14T21-12-08Z_arena_1_GoodCopBadCop",
        "camera_id": "2010093",
        "video_streams": {
            "schema_id": "orange_runtime_video_streams_v1",
            "frame_clock": "recording_frame_id",
            "streams": {
                "full": {
                    "role": "ingest_authoritative_full_frame",
                    "output_kind": "full",
                    "source": "orange_external_ipc",
                    "camera_id": "2010093",
                    "video": "cams/Cam2010093_sample.mp4",
                    "frame_clock": "recording_frame_id",
                    "frame_count": 2,
                    "width": 4512,
                    "height": 4512,
                },
                "crop": {
                    "role": "runtime_derived_acquisition_input",
                    "output_kind": "crop",
                    "source": "orange_external_ipc",
                    "camera_id": "2010093",
                    "stream_id": "2010093_crop",
                    "video": "derived/external_crop_recorder/Cam2010093_sample_crop_external.mp4",
                    "metadata": "derived/external_crop_recorder/Cam2010093_sample_crop_meta.csv",
                    "summary": (
                        "derived/external_crop_recorder/"
                        "Cam2010093_sample_crop_external_summary.json"
                    ),
                    "frame_clock": "recording_frame_id",
                    "video_pixel_coordinate_space": "crop_frame_pixels",
                    "source_geometry_coordinate_space": "full_frame_pixels",
                    "blank_frame_policy": "encode_black_frame_when_no_detection",
                    "selection_policy": "largest_detection_by_confidence",
                    "frame_count": 2,
                    "frame_rate": 100,
                    "width": 256,
                    "height": 256,
                    "codec": "hevc",
                    "encoded_format": "nv12",
                    "pixel_source_format": "mono8",
                },
            },
        },
    }
    (recording_dir / "recording_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    zarr_path = recording_dir / "zarr" / f"{recording_dir.name}_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    root.attrs.update(
        {
            "zarr_purpose": "analysis",
            "session_uuid": recording_dir.name,
            "recording_id": recording_dir.name,
            "recording_name": recording_dir.name,
            "recording_path": str(recording_dir),
            "camera_id": "2010093",
        }
    )
    write_acquisition_video_stream_inventory(root, recording_dir, manifest)
    return zarr_path


def test_schema_has_acquisition_video_streams_table_and_views(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        names = {
            str(row["name"])
            for row in registry.conn.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type IN ('table', 'view')
                  AND name IN (
                    'acquisition_video_streams',
                    'dataset_acquisition_video_streams_current',
                    'recording_acquisition_video_streams_current',
                    'recording_crop_video_available_current'
                  );
                """
            )
        }
    finally:
        registry.close()

    assert names == {
        "acquisition_video_streams",
        "dataset_acquisition_video_streams_current",
        "recording_acquisition_video_streams_current",
        "recording_crop_video_available_current",
    }


def test_scan_zarr_populates_acquisition_crop_video_view(tmp_path: Path) -> None:
    zarr_path = _write_recording_with_inventory(tmp_path)
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        dataset_id = registry.scan_zarr(zarr_path)
        rows = registry.conn.execute(
            """
            SELECT stream_key, output_kind, video_exists, metadata_exists, metadata_row_count
            FROM acquisition_video_streams
            WHERE dataset_id = ?
            ORDER BY stream_key;
            """,
            (dataset_id,),
        ).fetchall()
        crop = registry.conn.execute(
            """
            SELECT recording_id, crop_stream_available, crop_stream_consumer_ready,
                   availability_status,
                   video_pixel_coordinate_space, source_geometry_coordinate_space,
                   metadata_row_count, frames_encoded, frames_dropped,
                   canonical_ledger_status, canonical_ledger_row_count
            FROM recording_crop_video_available_current
            WHERE recording_id = ?;
            """,
            ("2026-06-14T21-12-08Z_arena_1_GoodCopBadCop",),
        ).fetchone()
    finally:
        registry.close()

    assert [(row["stream_key"], row["output_kind"]) for row in rows] == [
        ("crop", "crop"),
        ("full", "full"),
    ]
    assert int(rows[0]["video_exists"]) == 1
    assert int(rows[0]["metadata_exists"]) == 1
    assert int(rows[0]["metadata_row_count"]) == 2
    assert crop is not None
    assert int(crop["crop_stream_available"]) == 1
    assert int(crop["crop_stream_consumer_ready"]) == 1
    assert crop["availability_status"] == "ok"
    assert crop["video_pixel_coordinate_space"] == "crop_frame_pixels"
    assert crop["source_geometry_coordinate_space"] == "full_frame_pixels"
    assert int(crop["metadata_row_count"]) == 2
    assert int(crop["frames_encoded"]) == 2
    assert int(crop["frames_dropped"]) == 0
    assert crop["canonical_ledger_status"] == "complete"
    assert int(crop["canonical_ledger_row_count"]) == 2


def test_optional_inventory_warning_does_not_hide_complete_ledger(tmp_path: Path) -> None:
    zarr_path = _write_recording_with_inventory(tmp_path)
    root = zarr.open_group(str(zarr_path), mode="r+", use_consolidated=False)
    stream = root["analysis/acquisition_video_streams/streams/crop"]
    stream.attrs["availability_status"] = "warn"
    stream.attrs["warnings"] = ["summary_json_unreadable"]
    root["analysis/acquisition_video_streams"].attrs["inventory_status"] = "warn"

    registry = Registry(tmp_path / "registry.sqlite")
    try:
        registry.scan_zarr(zarr_path)
        crop = registry.conn.execute(
            """
            SELECT crop_stream_available, crop_stream_consumer_ready,
                   availability_status, canonical_ledger_status
            FROM recording_crop_video_available_current;
            """
        ).fetchone()
    finally:
        registry.close()

    assert crop is not None
    assert int(crop["crop_stream_available"]) == 0
    assert int(crop["crop_stream_consumer_ready"]) == 1
    assert crop["availability_status"] == "warn"
    assert crop["canonical_ledger_status"] == "complete"
