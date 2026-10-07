from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.shared.acquisition_crop_stream_ledger import (
    ACQUISITION_CROP_SOURCE_PROFILE_COLLECTION,
    validate_current_acquisition_crop_stream_ledger,
)
from fisheye.shared.acquisition_video_streams import (
    write_acquisition_video_stream_inventory,
)


def _orange_crop_descriptor(size: tuple[int, int], *, frames: int) -> dict[str, object]:
    """The produced-side fields of an Orange crop output (orange.recording_output.v1)."""

    return {
        "schema_version": 2,
        "camera_serial": "123",
        "role": "runtime_derived_acquisition_input",
        "backend": "external_ipc",
        "status": "completed",
        "codec": "hevc",
        "container": "mp4",
        "frame_rate": 100,
        "packet_count": frames,
        "video_pixel_coordinate_space": "crop_frame_pixels",
        "source_geometry_coordinate_space": "full_frame_pixels",
        "width": size[0],
        "height": size[1],
    }


def _write_clip(
    recording_dir: Path,
    *,
    clip_index: int,
    first_frame_id: int,
    declared_size: tuple[int, int] | None = None,
    all_blank: bool = False,
) -> dict[str, object]:
    clip_id = f"clip_{clip_index:06d}"
    clip_dir = recording_dir / "clips" / clip_id
    crop_dir = clip_dir / "crop"
    crop_dir.mkdir(parents=True)
    full_video = clip_dir / "Cam123.mp4"
    crop_video = crop_dir / "Cam123_crop_external.mp4"
    crop_meta = crop_dir / "Cam123_crop_meta.csv"
    full_video.write_bytes(f"full-{clip_index}".encode())
    crop_video.write_bytes(f"crop-{clip_index}".encode())
    second_frame_id = first_frame_id + 1
    crop_meta.write_text(
        "recording_frame_id,has_detection,blank_frame,crop_x,crop_y,crop_w,crop_h,"
        "detection_x,detection_y,detection_w,detection_h,crop_video_frame_index,"
        "session_crop_video_frame_index\n"
        + (
            f"{first_frame_id},0,1,,,,,,,,,0,{first_frame_id - 1}\n"
            if all_blank
            else f"{first_frame_id},1,0,10,20,384,384,100,120,20,30,0,{first_frame_id - 1}\n"
        )
        +
        f"{second_frame_id},0,1,,,,,,,,,1,{second_frame_id - 1}\n",
        encoding="utf-8",
    )
    manifest_path = clip_dir / "clip_manifest.json"

    def relative(path: Path) -> str:
        return str(path.relative_to(recording_dir))

    manifest_path.write_text(
        json.dumps(
            {
                "clip_id": clip_id,
                "clip_index": clip_index,
                "recording_outputs": {
                    "123": {
                        "crop": {
                            "output_kind": "crop",
                            "first_recording_frame_id": first_frame_id,
                            "last_recording_frame_id": second_frame_id,
                            "frame_count": 2,
                            "video": relative(crop_video),
                            "metadata": relative(crop_meta),
                            **(
                                _orange_crop_descriptor(declared_size, frames=2)
                                if declared_size is not None
                                else {}
                            ),
                        },
                        "full": {
                            "output_kind": "full",
                            "first_recording_frame_id": first_frame_id,
                            "last_recording_frame_id": second_frame_id,
                            "frame_count": 2,
                            "video": relative(full_video),
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return {
        "camera_serial": "123",
        "clip_id": clip_id,
        "clip_index": clip_index,
        "clip_manifest_path": relative(manifest_path),
        "first_recording_frame_id": first_frame_id,
        "last_recording_frame_id": second_frame_id,
        "frame_count": 2,
    }


def test_inventory_publishes_one_collection_ledger_for_two_clips(
    tmp_path: Path,
) -> None:
    recording_dir = tmp_path / "recording"
    rows = [
        _write_clip(recording_dir, clip_index=0, first_frame_id=1),
        _write_clip(recording_dir, clip_index=1, first_frame_id=3),
    ]
    clip_index_path = recording_dir / "recording_clip_index.json"
    clip_index_path.write_text(
        json.dumps(
            {
                "schema_id": "palette.orange_external_ipc_recording_clip_index.v1",
                "mode": "rolling_clips",
                "rows": rows,
                "camera_ranges": {
                    "123": {
                        "clip_count": 2,
                        "total_frame_count": 4,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    manifest = {
        "camera_id": "123",
        "recording_clip_index": "recording_clip_index.json",
        "rolling_clip_streams": {
            "schema_id": "palette.orange_rolling_clip_streams.v1",
            "frame_clock": "recording_frame_id",
            "recording_clip_index": "recording_clip_index.json",
        },
    }
    root = zarr.open_group(str(tmp_path / "analysis.zarr"), mode="w")

    inventory = write_acquisition_video_stream_inventory(
        root,
        recording_dir,
        manifest,
        imported_at_utc="2026-08-21T12:00:00+00:00",
    )
    publication = validate_current_acquisition_crop_stream_ledger(root)
    run = root[
        "analysis/acquisition_video_streams/streams/crop/ledger_runs/"
        + publication.run_name
    ]

    assert inventory is not None
    assert inventory["source_profile"] == ACQUISITION_CROP_SOURCE_PROFILE_COLLECTION
    assert publication.source_profile == ACQUISITION_CROP_SOURCE_PROFILE_COLLECTION
    assert publication.row_count == 4
    assert len(run.attrs["source_media_members"]) == 2
    np.testing.assert_array_equal(
        run["source_recording_frame_ids"][:], np.array([1, 2, 3, 4])
    )
    np.testing.assert_array_equal(
        run["source_crop_video_frame_indices"][:], np.array([0, 1, 0, 1])
    )
    np.testing.assert_array_equal(
        run["source_crop_video_member_indices"][:], np.array([0, 0, 1, 1])
    )
    np.testing.assert_array_equal(
        run["source_full_video_member_indices"][:], np.array([0, 0, 1, 1])
    )


def _sealed_crop_output(side: int, *, runtime_size: int | None = None) -> dict:
    size = side if runtime_size is None else runtime_size
    return {
        "enabled": True,
        "mode": "yolo_centered_square",
        "schema_version": 1,
        "source": {"camera_config_path": "/tmp/123.json", "requires_recording": True,
                   "requires_yolo": True, "ui_selected": True},
        "runtime": {
            "worker": "CropAndEncodeWorker", "crop_size_px": side, "width": size, "height": size,
            "frame_rate": 100, "codec": "hevc", "container": "mp4", "tuning": "lossless",
            "coordinate_space": "full_frame_pixels",
            "selection_policy": "largest_detection_by_confidence",
            "blank_frame_policy": "encode_black_frame_when_no_detection",
            "files": {"video": "Cam123_crop.mp4", "metadata": "Cam123_crop_meta.csv",
                      "keyframes": "Cam123_crop_keyframe.json", "perf": "Cam123_crop_perf.csv",
                      "sidecar_perf": "Cam123_crop_sidecar_perf.csv"},
        },
    }


def _publish(tmp_path: Path, clips: list[dict], *, sealed: dict | None = None):
    recording_dir = tmp_path / "recording"
    rows = [
        _write_clip(recording_dir, clip_index=i, first_frame_id=1 + 2 * i, **clip)
        for i, clip in enumerate(clips)
    ]
    (recording_dir / "recording_clip_index.json").write_text(
        json.dumps(
            {
                "schema_id": "palette.orange_external_ipc_recording_clip_index.v1",
                "mode": "rolling_clips",
                "rows": rows,
                "camera_ranges": {"123": {"clip_count": len(rows), "total_frame_count": 2 * len(rows)}},
            }
        ),
        encoding="utf-8",
    )
    manifest = {
        "camera_id": "123",
        "recording_clip_index": "recording_clip_index.json",
        "rolling_clip_streams": {
            "schema_id": "palette.orange_rolling_clip_streams.v1",
            "frame_clock": "recording_frame_id",
            "recording_clip_index": "recording_clip_index.json",
        },
    }
    if sealed is not None:
        snapshot = recording_dir / "raw/acquisition/recording_snapshot_start.json"
        snapshot.parent.mkdir(parents=True)
        snapshot.write_text(json.dumps({"crop_outputs": {"123": sealed}}), encoding="utf-8")
        manifest["source_transfer"] = {
            "source_to_parent_files": [
                {"relative_path": "raw/acquisition/recording_snapshot_start.json",
                 "source": {"path": "recording_snapshot_start.json"}}
            ]
        }
    root = zarr.open_group(str(tmp_path / "analysis.zarr"), mode="w")
    write_acquisition_video_stream_inventory(
        root, recording_dir, manifest, imported_at_utc="2026-10-07T12:00:00+00:00"
    )
    return root, validate_current_acquisition_crop_stream_ledger(root)


def test_all_blank_stream_is_sized_from_the_declared_crop_video(tmp_path: Path) -> None:
    # A camera with no detection all session (Orange trial camera 2010095) has
    # no row to infer a size from; the clip manifest declares the encoded size.
    clips = [{"declared_size": (384, 384), "all_blank": True}] * 2
    root, publication = _publish(tmp_path, clips)
    assert publication.row_count == 4
    run = root["analysis/acquisition_video_streams/streams/crop/ledger_runs/" + publication.run_name]
    contract = run.attrs["source_stream_contract"]
    assert (contract["width"], contract["height"]) == (384, 384)


def test_all_blank_stream_without_a_declared_size_still_refuses(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no row with valid crop geometry"):
        _publish(tmp_path, [{"all_blank": True}] * 2)


def test_detected_row_must_match_the_declared_crop_video_size(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not match declared crop video size"):
        _publish(tmp_path, [{"declared_size": (256, 256)}] * 2)


@pytest.mark.parametrize(
    "sizes", [[(384, 384), (256, 256)], [(384, 384), None]], ids=["different", "partly_declared"]
)
def test_clips_must_declare_one_crop_video_size(tmp_path: Path, sizes) -> None:
    with pytest.raises(ValueError, match="inconsistent crop video sizes"):
        _publish(tmp_path, [{"declared_size": size} for size in sizes])


def test_declared_size_must_be_a_valid_orange_output_descriptor(tmp_path: Path, monkeypatch) -> None:
    import fisheye.shared.acquisition_crop_stream_ledger as ledger

    real = _orange_crop_descriptor
    monkeypatch.setitem(
        globals(), "_orange_crop_descriptor",
        lambda size, frames: {k: v for k, v in real(size, frames=frames).items() if k != "codec"},
    )
    with pytest.raises(ValueError, match="violates orange.recording_output.v1"):
        _publish(tmp_path, [{"declared_size": (384, 384)}] * 2)
    assert ledger._ORANGE_SCHEMAS


def test_produced_size_agreeing_with_the_sealed_declaration_publishes(tmp_path: Path) -> None:
    clips = [{"declared_size": (384, 384), "all_blank": True}] * 2
    _root, publication = _publish(tmp_path, clips, sealed=_sealed_crop_output(384))
    assert publication.row_count == 4


@pytest.mark.parametrize(
    ("sealed", "message"),
    [
        (_sealed_crop_output(256), "disagrees with the size sealed at record start"),
        (_sealed_crop_output(384, runtime_size=256), "disagrees with its runtime size"),
        ({"enabled": True}, "violates orange.recording_crop_output.v1"),
    ],
    ids=["other_size", "inconsistent_declaration", "invalid_declaration"],
)
def test_produced_size_must_agree_with_the_sealed_declaration(tmp_path: Path, sealed, message) -> None:
    with pytest.raises(ValueError, match=message):
        _publish(tmp_path, [{"declared_size": (384, 384)}] * 2, sealed=sealed)


def test_pinned_orange_schemas_detect_drift(monkeypatch) -> None:
    import fisheye.shared.acquisition_crop_stream_ledger as ledger

    ledger._orange_validator.cache_clear()
    monkeypatch.setitem(ledger._ORANGE_SCHEMAS, "recording_output", ("orange_recording_output_v1.schema.json", "0" * 64))
    try:
        with pytest.raises(ValueError, match="packaged_contract_drift"):
            ledger._orange_validator("recording_output")
    finally:
        ledger._orange_validator.cache_clear()
