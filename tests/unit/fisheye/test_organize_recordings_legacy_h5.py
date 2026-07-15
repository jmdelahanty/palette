from pathlib import Path
import sys

import h5py

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.utils import organize_recordings


def _write_h5(path: Path, *, session_uuid: str, camera_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        h5.attrs["session_uuid"] = session_uuid
        h5.attrs["ipc_source_name"] = f"/shm_cam_{camera_id}"


def test_legacy_h5_plan_preserves_batch_root_sidecars(tmp_path: Path) -> None:
    batch_root = tmp_path / "2026_04_20_16_37_39"
    citrus_root = batch_root / "citrus"
    h5_path = citrus_root / "recording_arena_1.h5"
    _write_h5(h5_path, session_uuid="session_arena_1", camera_id="2010093")
    h5_path.with_suffix(".mp4").write_bytes(b"stimulus")
    h5_path.with_name(f"{h5_path.stem}_update_timing.csv").write_text(
        "frame,time\n", encoding="utf-8"
    )

    (batch_root / "Cam2010093.mp4").write_bytes(b"camera")
    (batch_root / "Cam2010093_meta.csv").write_text(
        "frame_id,timestamp,timestamp_sys\n", encoding="utf-8"
    )
    (batch_root / "Cam2010093_keyframe.json").write_text("{}", encoding="utf-8")
    snapshot = batch_root / "recording_snapshot.json"
    snapshot.write_text("{}", encoding="utf-8")
    ptp_summary = batch_root / "ptp_sync_summary.json"
    ptp_summary.write_text("{}", encoding="utf-8")

    plan = organize_recordings._build_plan(
        h5_path,
        dest_root=tmp_path / "recordings",
        cam_root=batch_root,
        rename_cams=True,
    )

    assert [planned.dest_name for planned in plan.cam_files] == [
        "Cam2010093_session_arena_1.mp4",
        "Cam2010093_session_arena_1_meta.csv",
        "Cam2010093_session_arena_1_keyframe.json",
    ]
    shared = {
        planned.dest_name: planned
        for planned in plan.raw_files
        if planned.dest_name in {"recording_snapshot_runtime.json", "ptp_sync_summary.json"}
    }
    assert shared["recording_snapshot_runtime.json"].source == snapshot
    assert shared["ptp_sync_summary.json"].source == ptp_summary
    assert all(planned.action == "copy" for planned in shared.values())

    organize_recordings._apply_plan_metadata_overrides(
        [plan], num_dishes=1, fish_per_dish=4
    )
    assert plan.meta["num_dishes"] == 1
    assert plan.meta["fish_per_dish"] == 4


def test_video_only_plan_finds_sidecars_in_organized_raw_directory(tmp_path: Path) -> None:
    recording_root = tmp_path / "recording"
    cams_root = recording_root / "cams"
    raw_root = recording_root / "raw"
    cams_root.mkdir(parents=True)
    raw_root.mkdir(parents=True)
    video = cams_root / "Cam2010093_legacy.mp4"
    video.write_bytes(b"video")
    metadata = cams_root / "Cam2010093_meta.csv"
    metadata.write_text("frame_id,timestamp,timestamp_sys\n", encoding="utf-8")
    (cams_root / "Cam2010093_keyframe.json").write_text("{}", encoding="utf-8")
    (raw_root / "ptp_sync_summary.json").write_text("{}", encoding="utf-8")
    (raw_root / "recording_snapshot_runtime.json").write_text("{}", encoding="utf-8")

    plan = organize_recordings._build_video_only_plan(
        {
            "source_video": str(video),
            "source_camera_metadata_csv": str(metadata),
            "camera_id": "2010093",
            "session_uuid": "legacy_video_only",
            "recording_id": "legacy_video_only",
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
