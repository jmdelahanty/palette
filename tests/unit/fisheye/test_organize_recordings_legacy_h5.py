from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.utils import organize_recordings
from fisheye.shared.source_recording_identity import (
    SOURCE_RECORDING_IDENTITY_PROFILE,
    SOURCE_RECORDING_IDENTITY_PROFILE_ATTR,
)










def test_video_only_plan_finds_sidecars_in_organized_raw_directory(tmp_path: Path) -> None:
    recording_root = tmp_path / "recording"
    cams_root = recording_root / "cams"
    raw_root = recording_root / "raw"
    cams_root.mkdir(parents=True)
    raw_root.mkdir(parents=True)
    video = cams_root / "CamCAM-42_legacy.mp4"
    video.write_bytes(b"video")
    metadata = cams_root / "CamCAM-42_meta.csv"
    metadata.write_text("frame_id,timestamp,timestamp_sys\n", encoding="utf-8")
    (cams_root / "CamCAM-42_keyframe.json").write_text("{}", encoding="utf-8")
    (raw_root / "ptp_sync_summary.json").write_text("{}", encoding="utf-8")
    (raw_root / "recording_snapshot_runtime.json").write_text("{}", encoding="utf-8")

    plan = organize_recordings._build_video_only_plan(
        {
            "source_video": str(video),
            "source_camera_metadata_csv": str(metadata),
            SOURCE_RECORDING_IDENTITY_PROFILE_ATTR: SOURCE_RECORDING_IDENTITY_PROFILE,
            "camera_id": "2010093",
            "session_uuid": "legacy_video_only",
            "recording_id": "legacy_video_only_cam2010093",
            "recording_name": "legacy_video_only",
            "dish_design": "cedar",
        },
        dest_root=tmp_path / "recordings",
        rename_cams=True,
    )

    assert {planned.dest_name for planned in plan.raw_files} == {
        "ptp_sync_summary.json",
        "recording_snapshot_runtime.json",
    }
    assert all(planned.action == "copy" for planned in plan.raw_files)
