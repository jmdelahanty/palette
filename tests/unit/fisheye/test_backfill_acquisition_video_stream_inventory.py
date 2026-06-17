from __future__ import annotations

import json
from pathlib import Path

from fisheye.utils import backfill_acquisition_video_stream_inventory as mod


def _write_recording_with_crop_stream(tmp_path: Path) -> Path:
    recording_dir = tmp_path / "2026-06-14T21-12-08Z_arena_1_GoodCopBadCop"
    (recording_dir / "cams").mkdir(parents=True)
    (recording_dir / "zarr" / f"{recording_dir.name}_analysis.zarr").mkdir(parents=True)
    crop_dir = recording_dir / "derived" / "external_crop_recorder"
    crop_dir.mkdir(parents=True)
    (recording_dir / "cams" / "Cam2010093_sample.mp4").write_bytes(b"full")
    (crop_dir / "Cam2010093_sample_crop_external.mp4").write_bytes(b"crop")
    (crop_dir / "Cam2010093_sample_crop_meta.csv").write_text(
        "recording_frame_id,crop_x,crop_y,crop_w,crop_h,has_detection,blank_frame\n"
        "1,10,20,256,256,true,false\n"
        "2,10,20,256,256,false,true\n",
        encoding="utf-8",
    )
    (recording_dir / "recording_manifest.json").write_text(
        json.dumps(
            {
                "video_streams": {
                    "schema_id": "orange_runtime_video_streams_v1",
                    "frame_clock": "recording_frame_id",
                    "streams": {
                        "full": {
                            "role": "ingest_authoritative_full_frame",
                            "output_kind": "full",
                            "video": "cams/Cam2010093_sample.mp4",
                            "frame_clock": "recording_frame_id",
                            "frame_count": 2,
                        },
                        "crop": {
                            "role": "runtime_derived_acquisition_input",
                            "output_kind": "crop",
                            "video": (
                                "derived/external_crop_recorder/"
                                "Cam2010093_sample_crop_external.mp4"
                            ),
                            "metadata": (
                                "derived/external_crop_recorder/"
                                "Cam2010093_sample_crop_meta.csv"
                            ),
                            "frame_clock": "recording_frame_id",
                            "video_pixel_coordinate_space": "crop_frame_pixels",
                            "source_geometry_coordinate_space": "full_frame_pixels",
                            "frame_count": 2,
                        },
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    return recording_dir


def test_build_plan_detects_crop_stream(tmp_path: Path) -> None:
    recording_dir = _write_recording_with_crop_stream(tmp_path)

    plan = mod.build_plan(recording_dir / "recording_manifest.json")

    assert plan.status == "ok"
    assert plan.stream_count == 2
    assert plan.stream_keys == ("crop", "full")
    assert plan.crop_stream_available is True
    assert plan.inventory_status == "ok"


def test_main_apply_writes_inventory(monkeypatch, tmp_path: Path) -> None:
    class _FakeAttrs(dict):
        def put(self, payload):
            self.clear()
            self.update(payload)

    class _FakeGroup:
        def __init__(self) -> None:
            self.attrs = _FakeAttrs()
            self.groups: dict[str, _FakeGroup] = {}

        def require_group(self, name: str):
            group = self
            for part in name.split("/"):
                group = group.groups.setdefault(part, _FakeGroup())
            return group

    recording_dir = _write_recording_with_crop_stream(tmp_path)
    fake_root = _FakeGroup()
    monkeypatch.setattr(mod.zarr, "open_group", lambda *_args, **_kwargs: fake_root)

    rc = mod.main([str(recording_dir), "--apply"])

    assert rc == 0
    assert fake_root.attrs["acquisition_video_streams_available"] is True
    assert fake_root.attrs["acquisition_crop_video_available"] is True
    streams = fake_root.groups["analysis"].groups["acquisition_video_streams"].groups["streams"]
    assert streams.groups["crop"].attrs["files"]["metadata"]["data_row_count"] == 2
