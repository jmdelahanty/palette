from pathlib import Path
import csv
import json
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


def test_main_video_only_apply_moves_file_and_writes_manifest(tmp_path: Path, monkeypatch) -> None:
    source_root = tmp_path / "staging"
    source_root.mkdir()
    video_path = source_root / "Cam2010093.mp4"
    video_path.write_bytes(b"video")
    keyframe_path = source_root / "Cam2010093_keyframe.json"
    keyframe_path.write_text("{}", encoding="utf-8")
    pipeline_perf_path = source_root / "Cam2010093_pipeline_perf.csv"
    pipeline_perf_path.write_text("metric,value\n", encoding="utf-8")
    cadence_probe_path = source_root / "Cam2010093_acquisition_cadence_probe.csv"
    cadence_probe_path.write_text("metric,value\n", encoding="utf-8")
    ptp_path = source_root / "ptp_sync_summary.json"
    ptp_path.write_text("{}", encoding="utf-8")
    snapshot_path = source_root / "recording_snapshot.json"
    snapshot_path.write_text("{}", encoding="utf-8")

    metadata_csv = tmp_path / "video_only_metadata.csv"
    with metadata_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_video",
                SOURCE_RECORDING_IDENTITY_PROFILE_ATTR,
                "recording_id",
                "session_uuid",
                "organizer_recording_id",
                "recording_name",
                "dish_design",
                "rig_id",
                "arena_id",
                "camera_id",
                "protocol_name",
                "num_dishes",
                "fish_per_dish",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "source_video": "Cam2010093.mp4",
                **_identity_fields(),
                "organizer_recording_id": "colleague_source_family",
                "recording_name": "Colleague Set 001",
                "dish_design": "cedar",
                "rig_id": "omnifin0",
                "arena_id": "arena_1",
                "camera_id": "2010093",
                "protocol_name": "ManualProtocol",
                "num_dishes": "1",
                "fish_per_dish": "1",
            }
        )

    monkeypatch.setattr(
        organize_recordings,
        "check_hevc_keyframe_flags",
        lambda _: {
            "codec": "hevc",
            "has_stss": True,
            "needs_fix": False,
            "message": "ok",
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "organize_recordings.py",
            str(source_root),
            "--video-only",
            "--metadata-csv",
            str(metadata_csv),
            "--dest-root",
            str(tmp_path / "recordings"),
            "--log-dir",
            str(tmp_path / "logs"),
            "--apply",
        ],
    )

    rc = organize_recordings.main()
    assert rc == 0

    dest_dir = tmp_path / "recordings" / "Colleague_Set_001"
    moved_video = dest_dir / "cams" / "Cam2010093_2026-03-09_colleague_set_001.mp4"
    assert moved_video.exists()
    assert not video_path.exists()
    assert (dest_dir / "raw" / "ptp_sync_summary.json").exists()
    assert (dest_dir / "raw" / "recording_snapshot_runtime.json").exists()
    assert ptp_path.exists()
    assert snapshot_path.exists()
    assert (dest_dir / "cams" / "Cam2010093_2026-03-09_colleague_set_001_keyframe.json").exists()
    assert (dest_dir / "derived" / "Cam2010093_2026-03-09_colleague_set_001_pipeline_perf.csv").exists()
    assert (
        dest_dir / "derived" / "Cam2010093_2026-03-09_colleague_set_001_acquisition_cadence_probe.csv"
    ).exists()
    assert not keyframe_path.exists()
    assert not pipeline_perf_path.exists()
    assert not cadence_probe_path.exists()

    manifest_path = dest_dir / "recording_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload[SOURCE_RECORDING_IDENTITY_PROFILE_ATTR] == (
        SOURCE_RECORDING_IDENTITY_PROFILE
    )
    assert payload["recording_id"] == "2026-03-09_colleague_set_001_cam2010093"
    assert payload["session_uuid"] == "2026-03-09_colleague_set_001"
    assert payload["camera_id"] == "2010093"
    assert payload["organizer_recording_id"] == "colleague_source_family"
    assert payload["artifact_schema_id"] == "video_only_v1"
    assert payload["recording_type"] == "behavior"
    assert payload["recording_subtype"] == "free"
    assert payload["behavior_mode"] == "free"
    assert payload["dish_design"] == "cedar"
    assert payload["protocol_name_from_definition"] == "ManualProtocol"
    assert payload["num_dishes"] == "1"
    assert payload["fish_per_dish"] == "1"
    assert payload["files"]["cams"] == [
        "cams/Cam2010093_2026-03-09_colleague_set_001.mp4",
        "cams/Cam2010093_2026-03-09_colleague_set_001_keyframe.json",
    ]
    assert payload["files"]["raw"] == [
        "raw/ptp_sync_summary.json",
        "raw/recording_snapshot_runtime.json",
    ]
    assert payload["files"]["derived"] == [
        "derived/Cam2010093_2026-03-09_colleague_set_001_pipeline_perf.csv",
        "derived/Cam2010093_2026-03-09_colleague_set_001_acquisition_cadence_probe.csv",
    ]


def test_video_only_apply_rejects_unmarked_identity_before_moving_files(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_root = tmp_path / "staging"
    source_root.mkdir()
    video_path = source_root / "Cam2010093.mp4"
    video_path.write_bytes(b"video")
    metadata_csv = tmp_path / "unmarked.csv"
    metadata_csv.write_text(
        "source_video,recording_id,session_uuid,camera_id\n"
        "Cam2010093.mp4,recording_cam2010093,session,2010093\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "organize_recordings.py",
            str(source_root),
            "--video-only",
            "--metadata-csv",
            str(metadata_csv),
            "--dest-root",
            str(tmp_path / "recordings"),
            "--log-dir",
            str(tmp_path / "logs"),
            "--apply",
            "--write-manifest",
        ],
    )

    assert organize_recordings.main() == 1
    assert video_path.exists()
    assert not (tmp_path / "recordings").exists()


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
