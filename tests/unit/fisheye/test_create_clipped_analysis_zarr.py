from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import zarr

from fisheye.utils.build_recording_frame_index import build_recording_frame_index
from fisheye.utils.create_clipped_analysis_zarr import create_clipped_analysis_zarr


def _write_metadata(path: Path, frame_ids: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame_id", "timestamp", "timestamp_sys"])
        writer.writeheader()
        for idx, frame_id in enumerate(frame_ids):
            writer.writerow(
                {
                    "frame_id": frame_id,
                    "timestamp": 1000 + idx,
                    "timestamp_sys": 2000 + idx,
                }
            )


def _write_clip_bundle(root: Path, clip_id: str, frame_ids: list[int]) -> dict[str, object]:
    clip_index = int(clip_id.rsplit("_", 1)[1])
    clip_dir = root / "clips" / clip_id
    video = clip_dir / "Cam2010093_example.mp4"
    metadata = clip_dir / "Cam2010093_example_meta.csv"
    keyframe = clip_dir / "Cam2010093_example_keyframe.json"
    manifest = clip_dir / "clip_manifest.json"
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"video")
    _write_metadata(metadata, frame_ids)
    keyframe.write_text(
        json.dumps({"total_frames": len(frame_ids), "fps": 2, "keyframe_frames": [0]}),
        encoding="utf-8",
    )
    manifest.write_text(json.dumps({"clip_id": clip_id}), encoding="utf-8")
    return {
        "recording_id": "rec_a",
        "session_id": "rec_a",
        "producer": "test",
        "recording_backend_mode": "materialized_stream_copy",
        "camera_serial": "2010093",
        "clip_index": clip_index,
        "clip_id": clip_id,
        "clip_directory": f"clips/{clip_id}",
        "video_path": f"clips/{clip_id}/{video.name}",
        "metadata_path": f"clips/{clip_id}/{metadata.name}",
        "keyframe_path": f"clips/{clip_id}/{keyframe.name}",
        "clip_manifest_path": f"clips/{clip_id}/clip_manifest.json",
        "frame_count": len(frame_ids),
        "first_recording_frame_id": frame_ids[0],
        "last_recording_frame_id": frame_ids[-1],
        "first_clip_local_frame_index": 0,
        "last_clip_local_frame_index": len(frame_ids) - 1,
    }


def _write_clipped_recording(root: Path) -> None:
    first = _write_clip_bundle(root, "clip_000000", [1, 2, 3])
    second = _write_clip_bundle(root, "clip_000001", [4, 5])
    (root / "recording_clip_index.json").write_text(
        json.dumps(
            {
                "recording_id": "rec_a",
                "session_id": "rec_a",
                "producer": "test_index",
                "recording_backend_mode": "materialized_stream_copy",
                "clips": [first, second],
            }
        ),
        encoding="utf-8",
    )
    build_recording_frame_index(root)


def test_create_clipped_analysis_zarr_writes_shell_layout(tmp_path: Path) -> None:
    root_dir = tmp_path / "rec_a"
    _write_clipped_recording(root_dir)
    output_zarr = tmp_path / "rec_a_analysis.zarr"

    result = create_clipped_analysis_zarr(root_dir, output_zarr=output_zarr)

    assert result["status"] == "ok"
    assert result["wrote_zarr"] is True
    assert result["clip_count"] == 2
    assert result["clip_camera_row_count"] == 2
    assert result["recording_frame_index_row_count"] == 5
    assert Path(result["shell_manifest_path"]).exists()

    root = zarr.open_group(str(output_zarr), mode="r")
    assert root.attrs["analysis_layout"] == "clipped_recording_shell"
    assert root.attrs["source_layout"] == "rolling_clips"
    assert root.attrs["recording_frame_index_row_count"] == 5
    assert root.attrs["recording_frame_id_min"] == 1
    assert root.attrs["recording_frame_id_max"] == 5
    assert root["raw_video"].attrs["storage_mode"] == "external_clips"
    assert root["detect_runs"].attrs["scope"] == "parent_finalized_or_aggregated"
    assert root["analysis_metadata"].attrs["dish_mask_scope"] == "recording_camera"
    assert root["analysis_metadata"].attrs["dish_mask_required_per_clip"] is False
    assert root["analysis_metadata"].attrs["orange_fixed_dish_location_invariant"] is True

    clip_group = root["clips"]["clip_000000"]
    assert clip_group.attrs["granularity"] == "clip"
    camera_group = clip_group["cameras"]["2010093"]
    assert camera_group.attrs["granularity"] == "clip_camera"
    assert camera_group.attrs["frame_count"] == 3
    assert camera_group.attrs["dish_mask_scope"] == "recording_camera"
    assert camera_group.attrs["dish_mask_clip_policy"] == "single_camera_mask_applies_to_all_clips"
    assert camera_group["source"].attrs["video_path"].endswith("clips/clip_000000/Cam2010093_example.mp4")
    assert camera_group["source"]["frame_map"].attrs["recording_frame_id_semantics"] == (
        "session_continuous_recording_frame_id"
    )
    assert camera_group["detect_runs"].attrs["latest"] is None
    assert camera_group["detect_runs"].attrs["scope"] == "clip_camera"
    assert root["experiment_index"]["clip_table"].attrs["row_count"] == 2


def test_create_clipped_analysis_zarr_dry_run_does_not_write(tmp_path: Path) -> None:
    root_dir = tmp_path / "rec_a"
    _write_clipped_recording(root_dir)
    output_zarr = tmp_path / "rec_a_analysis.zarr"

    result = create_clipped_analysis_zarr(root_dir, output_zarr=output_zarr, dry_run=True)

    assert result["status"] == "ok"
    assert result["wrote_zarr"] is False
    assert result["wrote_manifest"] is False
    assert not output_zarr.exists()


def test_create_clipped_analysis_zarr_refuses_existing_output_without_overwrite(tmp_path: Path) -> None:
    root_dir = tmp_path / "rec_a"
    _write_clipped_recording(root_dir)
    output_zarr = tmp_path / "rec_a_analysis.zarr"
    create_clipped_analysis_zarr(root_dir, output_zarr=output_zarr)

    with pytest.raises(FileExistsError):
        create_clipped_analysis_zarr(root_dir, output_zarr=output_zarr)

    result = create_clipped_analysis_zarr(root_dir, output_zarr=output_zarr, overwrite=True)

    assert result["status"] == "ok"
    assert result["wrote_zarr"] is True
