from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import zarr

from fisheye.utils.build_recording_frame_index import build_recording_frame_index
from fisheye.utils.create_clipped_training_zarr import (
    ClippedTrainingOptions,
    _contiguous_sample_runs,
    create_clipped_training_zarr,
)


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


def _write_source_training_zarr(path: Path, original_frame_indices: list[int]) -> None:
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    raw = root.create_group("raw_video")
    raw.create_array(
        "original_frame_indices",
        data=np.asarray(original_frame_indices, dtype=np.int64),
        chunks=(max(1, len(original_frame_indices)),),
    )
    raw.create_array(
        "images_full",
        data=np.zeros((len(original_frame_indices), 4, 6), dtype=np.uint8),
        chunks=(max(1, len(original_frame_indices)), 4, 6),
    )
    analysis = root.create_group("analysis_metadata")
    analysis.attrs["dish_mask"] = {"type": "circle", "center": [3, 2], "radius": 2}
    detect = root.create_group("detect_runs")
    detect.attrs["latest"] = "detect_source"
    detect_run = detect.create_group("detect_source")
    detect_run.create_array("frame_indices", data=np.arange(len(original_frame_indices), dtype=np.int64))
    refined = root.create_group("refined_detect_runs")
    refined.attrs["latest"] = "refined_source"
    instances = refined.create_group("refined_source").create_group("instances")
    instances.create_array("frame_indices", data=np.arange(len(original_frame_indices), dtype=np.int64))
    instances.create_array("refined_row_ids", data=np.arange(len(original_frame_indices), dtype=np.int64))


def _fake_decode(_video_path: Path, frame_indices: list[int]) -> dict[int, np.ndarray]:
    return {
        int(frame_index): np.full((4, 6), int(frame_index) + 10, dtype=np.uint8)
        for frame_index in frame_indices
    }


def test_contiguous_sample_runs_groups_slice_writes() -> None:
    rows = [
        {"sample_index": 5},
        {"sample_index": 2},
        {"sample_index": 3},
        {"sample_index": 8},
        {"sample_index": 6},
    ]

    runs = _contiguous_sample_runs(rows)

    assert [[int(row["sample_index"]) for row in run] for run in runs] == [[2, 3], [5, 6], [8]]


def test_create_clipped_training_zarr_writes_source_map_and_copies_compatible_detections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recording_dir = tmp_path / "rec_a"
    _write_clipped_recording(recording_dir)
    source_zarr = tmp_path / "source_training.zarr"
    _write_source_training_zarr(source_zarr, [0, 2, 4])
    output_zarr = tmp_path / "rec_a_clipped_training.zarr"
    monkeypatch.setattr("fisheye.utils.create_clipped_training_zarr._decode_clip_frames", _fake_decode)

    result = create_clipped_training_zarr(
        recording_dir,
        output_zarr=output_zarr,
        options=ClippedTrainingOptions(
            frame_step=2,
            copy_existing_detections_from=source_zarr,
            require_dish_mask=True,
            write_manifest=True,
        ),
    )

    assert result["status"] == "ok"
    assert result["selected_frame_count"] == 3
    assert result["copied_detection_runs"]["copied_groups"] == ["detect_runs", "refined_detect_runs"]
    root = zarr.open_group(str(output_zarr), mode="r")
    assert root.attrs["zarr_purpose"] == "training"
    assert root["raw_video"]["images_full"].shape == (3, 4, 6)
    assert root["raw_video"]["original_frame_indices"][:].tolist() == [0, 2, 4]
    assert root["analysis_metadata"].attrs["dish_mask"]["type"] == "circle"
    assert root["analysis_metadata"].attrs["dish_mask_scope"] == "recording_camera"
    assert root["detect_runs"].attrs["latest"] == "detect_source"
    assert root["refined_detect_runs"].attrs["latest"] == "refined_source"

    source_index = pq.read_table(output_zarr / "source_frame_index.parquet").to_pylist()
    assert [row["sample_index"] for row in source_index] == [0, 1, 2]
    assert [row["parent_frame_index"] for row in source_index] == [0, 2, 4]
    assert [(row["clip_id"], row["clip_local_frame_index"]) for row in source_index] == [
        ("clip_000000", 0),
        ("clip_000000", 2),
        ("clip_000001", 1),
    ]


def test_create_clipped_training_zarr_refuses_to_copy_mismatched_detections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recording_dir = tmp_path / "rec_a"
    _write_clipped_recording(recording_dir)
    source_zarr = tmp_path / "source_training.zarr"
    _write_source_training_zarr(source_zarr, [0, 2])
    monkeypatch.setattr("fisheye.utils.create_clipped_training_zarr._decode_clip_frames", _fake_decode)

    with pytest.raises(ValueError, match="original_frame_indices does not exactly match"):
        create_clipped_training_zarr(
            recording_dir,
            output_zarr=tmp_path / "bad.zarr",
            options=ClippedTrainingOptions(frame_step=2, copy_existing_detections_from=source_zarr),
        )


def test_create_clipped_training_zarr_can_copy_metadata_without_detection_groups(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recording_dir = tmp_path / "rec_a"
    _write_clipped_recording(recording_dir)
    source_zarr = tmp_path / "source_training.zarr"
    _write_source_training_zarr(source_zarr, [999])
    output_zarr = tmp_path / "metadata_only.zarr"
    monkeypatch.setattr("fisheye.utils.create_clipped_training_zarr._decode_clip_frames", _fake_decode)

    result = create_clipped_training_zarr(
        recording_dir,
        output_zarr=output_zarr,
        options=ClippedTrainingOptions(
            frame_step=2,
            max_frames=2,
            copy_analysis_metadata_from=source_zarr,
            require_dish_mask=True,
        ),
    )

    assert result["status"] == "ok"
    assert result["analysis_metadata_preflight"]["has_dish_mask"] is True
    root = zarr.open_group(str(output_zarr), mode="r")
    assert root["analysis_metadata"].attrs["dish_mask"]["type"] == "circle"
    assert root["raw_video"]["original_frame_indices"][:].tolist() == [0, 2]


def test_create_clipped_training_zarr_require_dish_mask_fails_before_write_without_source(
    tmp_path: Path,
) -> None:
    recording_dir = tmp_path / "rec_a"
    _write_clipped_recording(recording_dir)
    output_zarr = tmp_path / "missing_mask.zarr"

    with pytest.raises(ValueError, match="requires --copy-analysis-metadata-from"):
        create_clipped_training_zarr(
            recording_dir,
            output_zarr=output_zarr,
            options=ClippedTrainingOptions(frame_step=2, require_dish_mask=True),
        )

    assert not output_zarr.exists()


def test_create_clipped_training_zarr_dry_run_does_not_write(tmp_path: Path) -> None:
    recording_dir = tmp_path / "rec_a"
    _write_clipped_recording(recording_dir)
    output_zarr = tmp_path / "dry.zarr"

    result = create_clipped_training_zarr(
        recording_dir,
        output_zarr=output_zarr,
        options=ClippedTrainingOptions(frame_step=2, dry_run=True),
    )

    assert result["status"] == "ok"
    assert result["wrote_zarr"] is False
    assert result["selected_frame_count"] == 3
    assert not output_zarr.exists()
