from __future__ import annotations

import pytest
import zarr

from fisheye.shared import import_video_metadata as video_meta_mod
from fisheye.shared.import_source_fingerprint import (
    optional_source_stat_fingerprint_attrs,
    source_stat_fingerprint_attrs,
)
from fisheye.shared.import_video_metadata import probe_video_metadata, write_video_metadata
from fisheye.shared.source_video_metadata import SOURCE_VIDEO_METADATA_SCHEMA_ID


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
        "video_color_range": "tv",
        "video_color_space": "bt709",
        "video_color_transfer": "bt709",
        "video_color_primaries": "bt709",
        "source_video_colorimetry_source": "ffprobe_stream",
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
    assert root.attrs["source_video_metadata"]["schema_id"] == SOURCE_VIDEO_METADATA_SCHEMA_ID
    assert root.attrs["source_video_metadata"]["layout"] == "single_video"
    assert root.attrs["source_video_metadata"]["locator"] == {
        "kind": "absolute",
        "path": str(source.resolve()),
    }
    assert root.attrs["source_video_metadata"]["file_fingerprint"][
        "relocation_stable"
    ] is False
    assert raw.attrs["video_color_range"] == "tv"
    assert raw.attrs["video_color_space"] == "bt709"
    assert raw.attrs["source_video_colorimetry_source"] == "ffprobe_stream"
    assert root.attrs["video_color_range"] == "tv"
    assert "source_video_fingerprint" in updates["raw_video"]
    assert "source_video_fingerprint" in updates["root"]


def test_probe_video_metadata_parses_ffprobe_stream_colorimetry(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")

    class FakeCapture:
        def isOpened(self):
            return True

        def get(self, prop):
            if prop == video_meta_mod.cv2.CAP_PROP_FRAME_COUNT:
                return 100
            if prop == video_meta_mod.cv2.CAP_PROP_FPS:
                return 50.0
            if prop == video_meta_mod.cv2.CAP_PROP_FRAME_WIDTH:
                return 640
            if prop == video_meta_mod.cv2.CAP_PROP_FRAME_HEIGHT:
                return 480
            return 0

        def release(self):
            return None

    class FakeResult:
        returncode = 0
        stdout = (
            '{"format":{"tags":{"encoder":"orange"}},"streams":[{'
            '"codec_name":"hevc","pix_fmt":"yuv420p","color_range":"tv",'
            '"color_space":"bt709","color_transfer":"bt709","color_primaries":"bt709"'
            "}]} "
        )

    monkeypatch.setattr(video_meta_mod.cv2, "VideoCapture", lambda path: FakeCapture())
    monkeypatch.setattr(video_meta_mod.subprocess, "run", lambda *args, **kwargs: FakeResult())

    meta = probe_video_metadata(source)

    assert meta["video_color_range"] == "tv"
    assert meta["video_color_space"] == "bt709"
    assert meta["video_color_transfer"] == "bt709"
    assert meta["video_color_primaries"] == "bt709"
    assert meta["source_video_colorimetry_source"] == "ffprobe_stream"
    assert "imageio_metadata" not in meta


def test_probe_video_metadata_prefers_matching_producer_then_ffprobe(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")

    monkeypatch.setattr(
        video_meta_mod,
        "_probe_opencv",
        lambda _path: {
            "width": 640,
            "height": 480,
            "total_frames": 100,
            "fps": 50.0,
            "fourcc": "hvc1",
        },
    )
    monkeypatch.setattr(
        video_meta_mod,
        "_probe_ffprobe",
        lambda _path: {
            "width": 640,
            "height": 480,
            "total_frames": 100,
            "fps": 50.0,
            "codec": "hevc",
            "pix_fmt": "yuv420p",
        },
    )

    meta = probe_video_metadata(
        source,
        producer_metadata={
            "_source": "recording_manifest.video_streams.streams.full",
            "total_frames": 100,
            "fps": 50.0,
            "codec": "hevc",
        },
    )

    assert meta["total_frames"] == 100
    assert meta["fps"] == 50.0
    assert meta["width"] == 640
    assert meta["height"] == 480
    assert meta["codec"] == "hevc"
    assert meta["metadata_authority"]["field_sources"] == {
        "width": "ffprobe",
        "height": "ffprobe",
        "total_frames": "producer",
        "fps": "producer",
    }


def test_probe_video_metadata_rejects_producer_ffprobe_frame_conflict(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    monkeypatch.setattr(
        video_meta_mod,
        "_probe_opencv",
        lambda _path: {"width": 640, "height": 480, "total_frames": 100, "fps": 50.0},
    )
    monkeypatch.setattr(
        video_meta_mod,
        "_probe_ffprobe",
        lambda _path: {"width": 640, "height": 480, "total_frames": 100, "fps": 50.0},
    )

    with pytest.raises(ValueError, match="Producer and ffprobe disagree on total_frames"):
        probe_video_metadata(
            source,
            producer_metadata={"total_frames": 101, "fps": 50.0},
        )


def test_probe_video_metadata_does_not_require_opencv_when_ffprobe_is_complete(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")

    def _fail_opencv(_path):
        raise ValueError("opencv unavailable")

    monkeypatch.setattr(
        video_meta_mod,
        "_probe_opencv",
        _fail_opencv,
    )
    monkeypatch.setattr(
        video_meta_mod,
        "_probe_ffprobe",
        lambda _path: {
            "width": 640,
            "height": 480,
            "total_frames": 100,
            "fps": 50.0,
            "codec": "hevc",
            "pix_fmt": "yuv420p",
        },
    )

    meta = probe_video_metadata(source)

    assert set(meta["metadata_authority"]["field_sources"].values()) == {"ffprobe"}


def test_write_video_metadata_does_not_partially_upgrade_existing_legacy_object(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    root = zarr.open_group(str(tmp_path / "analysis.zarr"), mode="w", zarr_format=3)
    legacy = {"source_path": str(source), "width": 640, "height": 480}
    root.attrs["source_video_metadata"] = legacy

    write_video_metadata(
        root,
        {
            "source_video": source.name,
            "source_path": str(source),
            "width": 640,
            "height": 480,
            "total_frames": 10,
            "fps": 10.0,
            "codec": "hevc",
            "pix_fmt": "yuv420p",
        },
        overwrite=False,
        import_purpose="analysis",
    )

    assert root.attrs["source_video_metadata"] == legacy
    assert "source_video_metadata_schema_id" not in root.attrs
