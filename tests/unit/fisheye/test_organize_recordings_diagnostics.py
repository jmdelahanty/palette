from pathlib import Path
import csv
import json
import sys

import h5py

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.utils import organize_recordings


def _write_video_only_metadata_csv(path: Path, source_video_name: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_video",
                "session_uuid",
                "recording_name",
                "dish_design",
                "camera_id",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "source_video": source_video_name,
                "session_uuid": "2026-03-09_colleague_set_001",
                "recording_name": "Colleague Set 001",
                "dish_design": "cedar",
                "camera_id": "2010093",
            }
        )


def test_main_video_only_apply_runs_video_diagnostics_hook(tmp_path: Path, monkeypatch) -> None:
    source_root = tmp_path / "staging"
    source_root.mkdir()
    video_path = source_root / "Cam2010093.mp4"
    video_path.write_bytes(b"video")
    metadata_csv = tmp_path / "video_only_metadata.csv"
    _write_video_only_metadata_csv(metadata_csv, video_path.name)

    seen: list[Path] = []
    monkeypatch.setattr(
        organize_recordings,
        "check_hevc_keyframe_flags",
        lambda _: {"codec": "hevc", "has_stss": True, "needs_fix": False, "message": "ok"},
    )
    monkeypatch.setattr(
        organize_recordings,
        "_run_video_diagnostics_for_plan",
        lambda plan, logger: seen.append(plan.dest_dir) or [],
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
            "--run-video-diagnostics",
        ],
    )

    rc = organize_recordings.main()

    assert rc == 0
    assert seen == [tmp_path / "recordings" / "Colleague_Set_001"]


def test_main_requires_apply_for_diagnostics_flags(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["organize_recordings.py", str(tmp_path), "--run-video-diagnostics"],
    )

    rc = organize_recordings.main()
    err = capsys.readouterr().err

    assert rc == 1
    assert "require --apply" in err


def test_main_rejects_h5_diagnostics_for_video_only(tmp_path: Path, monkeypatch, capsys) -> None:
    source_root = tmp_path / "staging"
    source_root.mkdir()
    metadata_csv = tmp_path / "video_only_metadata.csv"
    _write_video_only_metadata_csv(metadata_csv, "Cam2010093.mp4")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "organize_recordings.py",
            str(source_root),
            "--video-only",
            "--metadata-csv",
            str(metadata_csv),
            "--apply",
            "--run-h5-diagnostics",
        ],
    )

    rc = organize_recordings.main()
    err = capsys.readouterr().err

    assert rc == 1
    assert "not supported with --video-only" in err


def test_main_apply_runs_h5_diagnostics_hook(tmp_path: Path, monkeypatch) -> None:
    source_root = tmp_path / "staging"
    source_root.mkdir()
    h5_path = source_root / "recording_001.h5"
    with h5py.File(h5_path, "w") as handle:
        handle.attrs["camera_id"] = "2010093"
        handle.attrs["session_uuid"] = "session_1"
    cam_mp4 = source_root / "Cam2010093.mp4"
    cam_meta = source_root / "Cam2010093_meta.csv"
    cam_mp4.write_bytes(b"video")
    cam_meta.write_text("frame_id,timestamp,timestamp_sys\n1,1,1\n", encoding="utf-8")

    seen: list[Path] = []
    monkeypatch.setattr(
        organize_recordings,
        "check_hevc_keyframe_flags",
        lambda _: {"codec": "hevc", "has_stss": True, "needs_fix": False, "message": "ok"},
    )
    monkeypatch.setattr(
        organize_recordings,
        "_run_h5_diagnostics_for_plan",
        lambda plan, logger: seen.append(plan.dest_dir) or [],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "organize_recordings.py",
            str(source_root),
            "--dest-root",
            str(tmp_path / "recordings"),
            "--log-dir",
            str(tmp_path / "logs"),
            "--apply",
            "--run-h5-diagnostics",
        ],
    )

    rc = organize_recordings.main()

    assert rc == 0
    assert seen == [tmp_path / "recordings" / "recording_001"]


def test_main_apply_persists_h5_preflight_manifest(tmp_path: Path, monkeypatch) -> None:
    source_root = tmp_path / "staging"
    source_root.mkdir()
    h5_path = source_root / "recording_002.h5"
    with h5py.File(h5_path, "w") as handle:
        handle.attrs["camera_id"] = "2010093"
        handle.attrs["session_uuid"] = "session_2"
    cam_mp4 = source_root / "Cam2010093.mp4"
    cam_meta = source_root / "Cam2010093_meta.csv"
    cam_mp4.write_bytes(b"video")
    cam_meta.write_text("frame_id,timestamp,timestamp_sys\n1,1,1\n", encoding="utf-8")

    monkeypatch.setattr(
        organize_recordings,
        "check_hevc_keyframe_flags",
        lambda _: {"codec": "hevc", "has_stss": True, "needs_fix": False, "message": "ok"},
    )
    monkeypatch.setattr(
        organize_recordings,
        "_run_h5_diagnostics_for_plan",
        lambda plan, logger: organize_recordings.H5DiagnosticsHookResult(
            manifest_payload=organize_recordings.build_h5_preflight_payload(
                status="pass",
                core_status="pass",
                optional_status="pass",
                tooling_status="pass",
                finding_codes=[],
            ),
            warnings=[],
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "organize_recordings.py",
            str(source_root),
            "--dest-root",
            str(tmp_path / "recordings"),
            "--log-dir",
            str(tmp_path / "logs"),
            "--apply",
            "--run-h5-diagnostics",
        ],
    )

    rc = organize_recordings.main()

    assert rc == 0
    manifest_path = tmp_path / "recordings" / "recording_002" / "recording_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["preflight"]["status"] == "pass"
    assert payload["preflight"]["h5"]["core_status"] == "pass"
    assert payload["preflight"]["checked_at_utc"]
