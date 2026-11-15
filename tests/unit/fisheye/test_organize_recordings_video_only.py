from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.utils import organize_recordings
from fisheye.shared.source_recording_identity import (
    SOURCE_RECORDING_IDENTITY_PROFILE,
    SOURCE_RECORDING_IDENTITY_PROFILE_ATTR,
)


def _identity_fields(
    *,
    recording_id: str = "2026-03-09_colleague_set_001_cam2010093",
    session_uuid: str = "2026-03-09_colleague_set_001",
    camera_id: str = "2010093",
) -> dict[str, str]:
    return {
        SOURCE_RECORDING_IDENTITY_PROFILE_ATTR: SOURCE_RECORDING_IDENTITY_PROFILE,
        "recording_id": recording_id,
        "session_uuid": session_uuid,
        "camera_id": camera_id,
    }


def test_build_video_only_plan_defaults_and_renames_cam_file(tmp_path: Path) -> None:
    video_path = tmp_path / "Cam2010093.mp4"
    video_path.write_bytes(b"video")
    (tmp_path / "Cam2010093_keyframe.json").write_text("{}", encoding="utf-8")
    (tmp_path / "Cam2010093_pipeline_perf.csv").write_text("metric,value\n", encoding="utf-8")
    (tmp_path / "Cam2010093_acquisition_cadence_probe.csv").write_text("metric,value\n", encoding="utf-8")
    (tmp_path / "ptp_sync_summary.json").write_text("{}", encoding="utf-8")
    (tmp_path / "recording_snapshot.json").write_text("{}", encoding="utf-8")

    plan = organize_recordings._build_video_only_plan(
        {
            "source_video": str(video_path),
            **_identity_fields(),
            "recording_name": "Colleague Set 001",
            "dish_design": "cedar",
        },
        dest_root=tmp_path / "recordings",
        rename_cams=True,
    )

    assert plan.name == "Colleague_Set_001"
    assert plan.camera_id == "2010093"
    assert plan.meta["recording_type"] == "behavior"
    assert plan.meta["recording_subtype"] == "free"
    assert plan.meta["behavior_mode"] == "free"
    assert plan.meta["artifact_schema_id"] == "video_only_v1"
    assert plan.cam_files[0].dest_name == "Cam2010093_2026-03-09_colleague_set_001.mp4"
    assert plan.cam_files[1].dest_name == "Cam2010093_2026-03-09_colleague_set_001_keyframe.json"
    assert [file.dest_name for file in plan.raw_files] == [
        "ptp_sync_summary.json",
        "recording_snapshot_runtime.json",
    ]
    assert all(file.action == "copy" for file in plan.raw_files)
    assert [file.dest_name for file in plan.derived_files] == [
        "Cam2010093_2026-03-09_colleague_set_001_pipeline_perf.csv",
        "Cam2010093_2026-03-09_colleague_set_001_acquisition_cadence_probe.csv",
    ]






def test_video_only_camera_cross_check_ignores_unrelated_filename_digits(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "experiment_20260825.mp4"
    video_path.write_bytes(b"video")

    plan = organize_recordings._build_video_only_plan(
        {
            "source_video": str(video_path),
            **_identity_fields(camera_id="2010093"),
            "recording_name": "recording",
            "dish_design": "cedar",
        },
        dest_root=tmp_path / "recordings",
        rename_cams=True,
    )

    assert plan.camera_id == "2010093"
