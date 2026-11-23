"""Registry migration 079: encoder settings and crop compression per stream."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from fisheye.registry.db import Registry
from fisheye.registry.extractors.acquisition_video_stream_encoders import (
    extract_acquisition_video_stream_encoder_rows,
    summary_relative_path,
)
from fisheye.registry.migrations import ADDITIVE_MIGRATIONS, MIGRATION_METHODS

from tests.unit.fisheye.test_orange_crop_cqp_gop1 import _cqp_summary
from tests.unit.fisheye.test_registry_acquisition_video_streams import (
    _write_recording_with_inventory,
)

CAMERA = "2010093"
ROLLING = {"source_profile": "rolling_clip_collection_v1", "camera_id": CAMERA}
CROP_SUMMARY = f"cams/acquisition/external_crop_recorder/Cam{CAMERA}_crop_external_summary.json"
FULL_SUMMARY = f"cams/acquisition/external_recorder/Cam{CAMERA}_external_summary.json"

FULL_VBR = {
    "tuning": "ll",
    "rate_control": {
        "resolved_strategy": "vbr", "const_qp": 0, "const_qp_active": False, "quality_value": 20,
    },
    "video_metadata": {
        "encoder": {
            "name": "nvenc", "codec": "hevc", "profile_name": "external_full_hevc_low_latency",
            "tuning": "ll", "preset": "p1", "rate_control_strategy": "vbr", "quality_value": 20,
            "target_bps": 45000000, "max_bps": 60000000, "resolved_gop_length": 25,
            "source_format": "mono8", "output_width": 4512, "output_height": 4512,
        }
    },
}
LOSSLESS_CROP = {
    "tuning": "lossless",
    "video_metadata": {"encoder": {"codec": "hevc", "tuning": "lossless",
                                   "profile_name": "external_crop_hevc_lossless_gop1"}},
}


def _stream(kind: str, **extra) -> dict:
    return {
        "stream_key": kind, "output_kind": kind, "camera_id": CAMERA, "recording_id": "rec-1",
        "contract_json": json.dumps({**ROLLING, "output_kind": kind}), **extra,
    }


def _write(recording: Path, relative: str, payload) -> bytes:
    path = recording / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    path.write_bytes(data)
    return data


def _rows(recording: Path, *streams: dict) -> dict:
    rows = extract_acquisition_video_stream_encoder_rows(recording, streams, dataset_id="d1")
    return {row["stream_key"]: row for row in rows}


def test_migration_079_is_registered_and_additive() -> None:
    assert (
        79,
        "acquisition_video_stream_encoders",
        "_migration_079_acquisition_video_stream_encoders",
    ) in MIGRATION_METHODS
    assert 79 in ADDITIVE_MIGRATIONS


def test_rolling_streams_locate_the_fixed_orange_layout() -> None:
    assert summary_relative_path(_stream("crop")) == CROP_SUMMARY
    assert summary_relative_path(_stream("full")) == FULL_SUMMARY
    assert summary_relative_path(_stream("crop", summary_path="derived/x_summary.json")) == (
        "derived/x_summary.json"
    )
    assert summary_relative_path({"stream_key": "crop", "output_kind": "crop"}) is None


def test_qp29_crop_and_vbr_full_are_projected(tmp_path: Path) -> None:
    crop_bytes = _write(tmp_path, CROP_SUMMARY, _cqp_summary(29))
    _write(tmp_path, FULL_SUMMARY, FULL_VBR)

    rows = _rows(tmp_path, _stream("crop"), _stream("full"))

    crop = rows["crop"]
    assert crop["summary_status"] == "read"
    assert crop["summary_sha256"] == hashlib.sha256(crop_bytes).hexdigest()
    assert crop["crop_compression_status"] == "admitted"
    assert crop["crop_compression_rule"] == "palette.orange_crop_cqp_gop1.v1"
    assert (crop["crop_compression_qp"], crop["crop_compression_lossless"]) == (29, 0)
    assert (crop["tuning"], crop["rate_control_strategy"], crop["const_qp"], crop["gop_length"]) == (
        "hq", "cqp", 29, 1,
    )
    full = rows["full"]
    assert full["crop_compression_status"] is None
    assert (full["encoder_profile_name"], full["tuning"], full["preset"]) == (
        "external_full_hevc_low_latency", "ll", "p1",
    )
    assert (full["rate_control_strategy"], full["const_qp"], full["quality_value"]) == ("vbr", None, 20)
    assert (full["target_bps"], full["max_bps"], full["gop_length"]) == (45000000, 60000000, 25)
    assert (full["output_width"], full["output_height"], full["source_format"]) == (4512, 4512, "mono8")


def test_lossless_crop_is_tagged_qp_zero(tmp_path: Path) -> None:
    _write(tmp_path, CROP_SUMMARY, LOSSLESS_CROP)

    crop = _rows(tmp_path, _stream("crop"))["crop"]

    assert crop["crop_compression_rule"] == "palette.orange_crop_lossless.v1"
    assert (crop["crop_compression_qp"], crop["crop_compression_lossless"]) == (0, 1)


def test_unadmitted_crop_records_the_refusal(tmp_path: Path) -> None:
    summary = _cqp_summary(29)
    summary["tuning"] = "ll"
    _write(tmp_path, CROP_SUMMARY, summary)

    crop = _rows(tmp_path, _stream("crop"))["crop"]

    assert crop["crop_compression_status"] == "refused"
    assert crop["crop_compression_rule"] is None
    assert "palette.orange_crop_cqp_gop1.v1" in crop["crop_compression_refusal"]


@pytest.mark.parametrize(
    ("content", "status"),
    [(None, "missing"), (b"{not json", "unreadable"), (b"[1]", "unreadable")],
)
def test_unread_summaries_are_explicit(tmp_path: Path, content, status) -> None:
    if content is not None:
        _write(tmp_path, CROP_SUMMARY, content)

    crop = _rows(tmp_path, _stream("crop"))["crop"]

    assert crop["summary_status"] == status
    assert crop["crop_compression_status"] is None
    assert crop["encoder_profile_name"] is None


def test_unlocated_stream_gets_a_row(tmp_path: Path) -> None:
    rows = _rows(tmp_path, {"stream_key": "odd", "output_kind": "crop"})
    assert rows["odd"]["summary_status"] == "unlocated"


def test_scan_projects_encoder_rows_for_a_legacy_stream_inventory(tmp_path: Path) -> None:
    zarr_path = _write_recording_with_inventory(tmp_path)
    recording = zarr_path.parent.parent
    _write(
        recording,
        "derived/external_crop_recorder/Cam2010093_sample_crop_external_summary.json",
        LOSSLESS_CROP,
    )
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        dataset_id = registry.scan_zarr(zarr_path)
        rows = {
            row["stream_key"]: row
            for row in registry.conn.execute(
                "SELECT * FROM acquisition_video_stream_encoders WHERE dataset_id = ?;",
                (dataset_id,),
            )
        }
        registry.scan_zarr(zarr_path)
        count = registry.conn.execute(
            "SELECT COUNT(*) FROM acquisition_video_stream_encoders WHERE dataset_id = ?;",
            (dataset_id,),
        ).fetchone()[0]
    finally:
        registry.close()

    assert sorted(rows) == ["crop", "full"]
    assert rows["crop"]["crop_compression_rule"] == "palette.orange_crop_lossless.v1"
    assert rows["crop"]["summary_relative_path"].startswith("derived/external_crop_recorder/")
    assert rows["full"]["summary_status"] == "unlocated"
    assert count == 2


def test_table_refuses_inconsistent_rows(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        registry.conn.execute("INSERT INTO datasets (dataset_id, zarr_path) VALUES ('d1', '/x');")
        with pytest.raises(sqlite3.IntegrityError):
            registry.replace_acquisition_video_stream_encoders(
                "d1",
                [{"stream_key": "crop", "summary_status": "read", "updated_utc": "t"}],
            )
        with pytest.raises(sqlite3.IntegrityError):
            registry.replace_acquisition_video_stream_encoders(
                "d1",
                [{"stream_key": "crop", "summary_status": "missing", "updated_utc": "t",
                  "crop_compression_status": "admitted"}],
            )
    finally:
        registry.close()
