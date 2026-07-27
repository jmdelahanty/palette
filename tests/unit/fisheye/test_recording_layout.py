from __future__ import annotations

from pathlib import Path

import pytest

from fisheye.cluster.recording_layout import (
    FrameMappingMode,
    RecordingLayout,
    RecordingTarget,
    VideoFrameMapping,
    VideoWorkUnit,
    clipped_recording_target,
    whole_video_recording_target,
)


def _clip_rows(tmp_path: Path) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "work_unit_id": f"recording_clip_{index}",
            "clip_id": f"clip_{index:06d}",
            "clip_index": index,
            "camera_serial": "2010093",
            "frame_count": 54_000,
            "source": {
                "video_path": tmp_path / "clips" / f"clip_{index:06d}.mp4"
            },
        }
        for index in range(2)
    )


def test_clipped_adapter_preserves_partition_and_frame_authority(tmp_path: Path) -> None:
    frame_index = tmp_path / "recording" / "recording_frame_index.parquet"

    target = clipped_recording_target(
        target_id="sleepyfish_cam2010093",
        recording_id="sleepyfish_cam2010093:z1",
        recording_dir=tmp_path / "recording",
        analysis_zarr=tmp_path / "recording" / "analysis.zarr",
        work_units=_clip_rows(tmp_path),
        recording_frame_index=frame_index,
        expected_subject_count=1,
    )

    assert target.layout is RecordingLayout.CLIPPED_COLLECTION
    assert target.recording_frame_index == frame_index.resolve()
    assert [unit.source_partition_id for unit in target.work_units] == [
        "clip_000000",
        "clip_000001",
    ]
    assert [unit.source_partition_index for unit in target.work_units] == [0, 1]
    assert all(
        unit.frame_mapping.mode is FrameMappingMode.RECORDING_FRAME_INDEX
        and unit.frame_mapping.recording_frame_index == frame_index.resolve()
        for unit in target.work_units
    )
    assert target.to_json()["work_unit_count"] == 2


def test_whole_video_adapter_is_a_one_member_identity_collection(tmp_path: Path) -> None:
    target = whole_video_recording_target(
        target_id="batman_cam2010093",
        recording_id="batman_cam2010093:z1",
        recording_dir=tmp_path / "recording",
        analysis_zarr=tmp_path / "recording" / "analysis.zarr",
        video_path=tmp_path / "recording" / "cams" / "Cam2010093.mp4",
        camera_serial="2010093",
        frame_count=1_000,
        arena_id="arena_1",
    )

    assert target.layout is RecordingLayout.WHOLE_VIDEO
    assert len(target.work_units) == 1
    unit = target.work_units[0]
    assert unit.source_partition_id == "whole_video"
    assert unit.frame_mapping == VideoFrameMapping.identity()
    assert target.recording_frame_index is None


def test_clipped_adapter_allows_same_partition_index_for_distinct_cameras(
    tmp_path: Path,
) -> None:
    rows = tuple(
        {
            "work_unit_id": f"clip_000000_cam{camera}",
            "clip_id": "clip_000000",
            "clip_index": 0,
            "camera_serial": camera,
            "source": {"video_path": tmp_path / f"Cam{camera}.mp4"},
        }
        for camera in ("2010093", "2010094")
    )

    target = clipped_recording_target(
        target_id="two_camera_recording",
        recording_id="two_camera_recording:z1",
        recording_dir=tmp_path,
        analysis_zarr=tmp_path / "analysis.zarr",
        work_units=rows,
    )

    assert [unit.source_partition_index for unit in target.work_units] == [0, 0]
    assert [unit.camera_serial for unit in target.work_units] == [
        "2010093",
        "2010094",
    ]


def test_recording_target_rejects_duplicate_work_unit_identity(tmp_path: Path) -> None:
    mapping = VideoFrameMapping.indexed(tmp_path / "recording_frame_index.parquet")
    unit = VideoWorkUnit(
        work_unit_id="duplicate",
        source_partition_id="clip_000000",
        source_partition_index=0,
        video_path=tmp_path / "clip.mp4",
        camera_serial="2010093",
        frame_mapping=mapping,
    )

    with pytest.raises(ValueError, match="work_unit_id values must be unique"):
        RecordingTarget(
            target_id="target",
            recording_id="recording:z1",
            recording_dir=tmp_path,
            analysis_zarr=tmp_path / "analysis.zarr",
            layout=RecordingLayout.CLIPPED_COLLECTION,
            work_units=(unit, unit),
        )


def test_clipped_target_rejects_mixed_frame_index_authorities(tmp_path: Path) -> None:
    units = tuple(
        VideoWorkUnit(
            work_unit_id=f"unit_{index}",
            source_partition_id=f"clip_{index:06d}",
            source_partition_index=index,
            video_path=tmp_path / f"clip_{index}.mp4",
            camera_serial="2010093",
            frame_mapping=VideoFrameMapping.indexed(
                tmp_path / f"recording_frame_index_{index}.parquet"
            ),
        )
        for index in range(2)
    )

    with pytest.raises(ValueError, match="share one recording-frame index"):
        RecordingTarget(
            target_id="target",
            recording_id="recording:z1",
            recording_dir=tmp_path,
            analysis_zarr=tmp_path / "analysis.zarr",
            layout=RecordingLayout.CLIPPED_COLLECTION,
            work_units=units,
        )


def test_whole_video_target_rejects_indexed_mapping(tmp_path: Path) -> None:
    unit = VideoWorkUnit(
        work_unit_id="whole",
        source_partition_id="whole_video",
        source_partition_index=0,
        video_path=tmp_path / "video.mp4",
        camera_serial="2010093",
        frame_mapping=VideoFrameMapping.indexed(
            tmp_path / "recording_frame_index.parquet"
        ),
    )

    with pytest.raises(ValueError, match="identity frame mapping"):
        RecordingTarget(
            target_id="target",
            recording_id="recording:z1",
            recording_dir=tmp_path,
            analysis_zarr=tmp_path / "analysis.zarr",
            layout=RecordingLayout.WHOLE_VIDEO,
            work_units=(unit,),
        )
