import json
from pathlib import Path

import fisheye.utils.backfill_import_profile_metadata as backfill


def _write_group(path: Path, attrs: dict | None = None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "zarr.json").write_text(
        json.dumps(
            {
                "zarr_format": 3,
                "node_type": "group",
                "attributes": attrs or {},
            }
        ),
        encoding="utf-8",
    )


def _read_attrs(path: Path) -> dict:
    payload = json.loads((path / "zarr.json").read_text(encoding="utf-8"))
    return payload["attributes"]


def _make_recording(tmp_path: Path) -> tuple[Path, Path, Path]:
    recording = tmp_path / "2026-07-06T00-00-00Z_arena_1_Test"
    zarr_path = recording / "zarr" / "test_analysis.zarr"
    video_path = recording / "cams" / "Cam1.mp4"
    h5_path = recording / "raw" / "test.h5"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.write_bytes(b"not a real mp4")
    h5_path.write_bytes(b"not a real h5")
    _write_group(
        zarr_path,
        {
            "zarr_purpose": "analysis",
            "source_video_path": "cams/Cam1.mp4",
            "source_frame_count": 123,
            "source_video_width": 4512,
            "source_video_height": 4512,
        },
    )
    _write_group(zarr_path / "raw_video", {})
    return zarr_path, video_path, h5_path


def test_dry_run_plans_missing_import_profile_metadata_without_mutating(tmp_path, monkeypatch):
    zarr_path, _, _ = _make_recording(tmp_path)
    monkeypatch.setattr(
        backfill,
        "probe_video_colorimetry_attrs",
        lambda path: {
            "video_color_range": "pc",
            "video_color_space": "bt709",
            "source_video_colorimetry_source": "ffprobe_stream",
        },
    )

    rows = backfill.backfill_zarr_import_profile_metadata(
        zarr_path,
        apply=False,
        include_existing_colorimetry=False,
    )

    planned = {(row["action"], row["status"]) for row in rows}
    assert ("source_video_stat_fingerprint", "planned") in planned
    assert ("source_video_ffprobe_colorimetry", "planned") in planned
    assert ("source_h5_stat_fingerprint", "planned") in planned
    assert "source_video_fingerprint" not in _read_attrs(zarr_path)
    assert "video_color_range" not in _read_attrs(zarr_path)
    assert "source_h5_fingerprint" not in _read_attrs(zarr_path)


def test_apply_writes_v3_root_and_raw_video_attrs_without_legacy_color_names(tmp_path, monkeypatch):
    zarr_path, video_path, h5_path = _make_recording(tmp_path)
    monkeypatch.setattr(
        backfill,
        "probe_video_colorimetry_attrs",
        lambda path: {
            "video_color_range": "tv",
            "video_color_space": "bt709",
            "video_color_transfer": "bt709",
            "video_color_primaries": "bt709",
            "source_video_colorimetry_source": "ffprobe_stream",
        },
    )

    rows = backfill.backfill_zarr_import_profile_metadata(
        zarr_path,
        apply=True,
        include_existing_colorimetry=False,
    )

    assert all(row["status"] == "updated" for row in rows)
    for attrs in (_read_attrs(zarr_path), _read_attrs(zarr_path / "raw_video")):
        assert attrs["source_video_fingerprint_strategy"] == "stat_v1"
        assert attrs["source_video_fingerprint_payload"]["path"] == str(video_path)
        assert attrs["source_h5_fingerprint_strategy"] == "stat_v1"
        assert attrs["source_h5_fingerprint_payload"]["path"] == str(h5_path)
        assert attrs["source_h5_path"] == str(h5_path)
        assert attrs["source_h5"] == h5_path.name
        assert attrs["video_color_range"] == "tv"
        assert attrs["video_color_space"] == "bt709"
        assert "color_range" not in attrs
        assert "color_space" not in attrs


def test_include_existing_colorimetry_fills_missing_video_color_subfields(tmp_path, monkeypatch):
    zarr_path, _, _ = _make_recording(tmp_path)
    root_attrs = _read_attrs(zarr_path)
    root_attrs["video_color_range"] = "pc"
    (zarr_path / "zarr.json").write_text(
        json.dumps({"zarr_format": 3, "node_type": "group", "attributes": root_attrs}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        backfill,
        "probe_video_colorimetry_attrs",
        lambda path: {
            "video_color_range": "pc",
            "video_color_space": "bt709",
            "source_video_colorimetry_source": "ffprobe_stream",
        },
    )

    rows_without_flag = backfill.backfill_zarr_import_profile_metadata(
        zarr_path,
        apply=False,
        include_existing_colorimetry=False,
    )
    assert "source_video_ffprobe_colorimetry" not in {row["action"] for row in rows_without_flag}

    rows_with_flag = backfill.backfill_zarr_import_profile_metadata(
        zarr_path,
        apply=True,
        include_existing_colorimetry=True,
    )

    color_rows = [row for row in rows_with_flag if row["action"] == "source_video_ffprobe_colorimetry"]
    assert color_rows and color_rows[0]["status"] == "updated"
    attrs = _read_attrs(zarr_path)
    assert attrs["video_color_range"] == "pc"
    assert attrs["video_color_space"] == "bt709"


def test_skip_path_contains_prevents_backfill(tmp_path):
    zarr_path, _, _ = _make_recording(tmp_path / "sleepyfish")

    rows = backfill.backfill_zarr_import_profile_metadata(
        zarr_path,
        apply=True,
        include_existing_colorimetry=False,
        skip_if_path_contains=("sleepyfish",),
    )

    assert rows == [
        {
            "record_type": "import_profile_metadata_backfill_action",
            "zarr_path": str(zarr_path),
            "action": "skip_zarr",
            "status": "skipped",
            "reason": "path_excluded",
            "source_path": None,
            "source_resolution": None,
            "root": {},
            "raw_video": {},
            "values": {},
        }
    ]
    assert "source_video_fingerprint" not in _read_attrs(zarr_path)


def test_writer_is_zarr_v3_only_and_skips_legacy_zattrs(tmp_path):
    legacy = tmp_path / "legacy.zarr"
    legacy.mkdir()
    (legacy / ".zgroup").write_text("{}", encoding="utf-8")
    (legacy / ".zattrs").write_text("{}", encoding="utf-8")

    result = backfill._write_missing_node_attrs(legacy, {"video_color_range": "pc"}, apply=True)

    assert result["status"] == "skipped"
    assert result["reason"] == "missing_zarr_json"
    assert not (legacy / "zarr.json").exists()
