from __future__ import annotations

import zarr

from fisheye.shared.import_source_fingerprint import (
    optional_source_stat_fingerprint_attrs,
    source_stat_fingerprint_attrs,
)
from fisheye.shared.import_video_metadata import write_video_metadata


def test_source_stat_fingerprint_attrs_are_stable_for_same_stat_and_metadata(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    extra = {"codec": "hevc", "frame_count": 123}

    first = source_stat_fingerprint_attrs(source, attr_prefix="source_video", extra=extra)
    second = source_stat_fingerprint_attrs(source, attr_prefix="source_video", extra=extra)

    assert first["source_video_fingerprint_strategy"] == "stat_v1"
    assert first["source_video_fingerprint"] == second["source_video_fingerprint"]
    assert first["source_video_fingerprint_payload"]["frame_count"] == 123
    assert first["source_video_size_bytes"] == len(b"video")


def test_optional_source_stat_fingerprint_records_missing_file_error(tmp_path):
    missing = tmp_path / "missing.h5"

    attrs = optional_source_stat_fingerprint_attrs(missing, attr_prefix="source_h5")

    assert attrs["source_h5_fingerprint_strategy"] == "stat_v1"
    assert "source_h5_fingerprint_error" in attrs
    assert "source_h5_fingerprint" not in attrs


def test_write_video_metadata_stamps_metadata_only_profile_and_source_fingerprint(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    root = zarr.open_group(str(tmp_path / "analysis.zarr"), mode="w", zarr_format=3)
    meta = {
        "source_video": source.name,
        "source_path": str(source),
        "width": 4512,
        "height": 4512,
        "total_frames": 1000,
        "fps": 100.0,
        "duration_seconds": 10.0,
        "codec": "hevc",
        "pix_fmt": "yuv420p",
    }

    updates = write_video_metadata(root, meta, overwrite=False, import_purpose="analysis")

    raw = root["raw_video"]
    assert root.attrs["import_profile"] == "metadata_only_analysis"
    assert raw.attrs["import_profile"] == "metadata_only_analysis"
    assert root.attrs["import_profile_schema_id"] == "palette.import_profile_contract.v1"
    assert raw.attrs["import_profile_schema_id"] == "palette.import_profile_contract.v1"
    assert raw.attrs["source_video_fingerprint_strategy"] == "stat_v1"
    assert raw.attrs["source_video_fingerprint"]
    assert raw.attrs["source_video_fingerprint_payload"]["frame_count"] == 1000
    assert raw.attrs["source_video_size_bytes"] == len(b"video")
    assert root.attrs["source_video_fingerprint"] == raw.attrs["source_video_fingerprint"]
    assert "source_video_fingerprint" in updates["raw_video"]
    assert "source_video_fingerprint" in updates["root"]
