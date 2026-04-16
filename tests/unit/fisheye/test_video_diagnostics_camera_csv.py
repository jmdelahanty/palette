from __future__ import annotations

from pathlib import Path

from fisheye.diagnostics import video as video_mod
from fisheye.diagnostics.video import camera_csv as mod
from fisheye.diagnostics.video.models import StreamInfo


def test_inspect_camera_csv_passes_on_matching_metadata(tmp_path: Path) -> None:
    cams_dir = tmp_path / "session" / "cams"
    cams_dir.mkdir(parents=True)
    video_path = cams_dir / "Cam1.mp4"
    csv_path = cams_dir / "Cam1_meta.csv"
    video_path.write_bytes(b"video")
    csv_path.write_text(
        "\n".join(
            [
                "frame_id,timestamp,timestamp_sys",
                "1,100,1100",
                "2,200,1200",
                "3,300,1300",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    info, findings = mod.inspect_camera_csv(video_path, expected_frame_count=3)

    assert findings == []
    assert info.status == "pass"
    assert info.rows == 3
    assert info.frame_id_first == 1
    assert info.frame_id_last == 3
    assert info.frame_id_monotonic is True
    assert info.frame_id_contiguous is True
    assert info.timestamp_monotonic is True
    assert info.timestamp_sys_monotonic is True
    assert info.row_count_matches_video is True
    assert info.median_timestamp_step_ns == 100
    assert info.median_timestamp_sys_step_ns == 100
    assert info.timestamp_offset_drift_ns == 0


def test_inspect_camera_csv_warns_when_missing(tmp_path: Path) -> None:
    cams_dir = tmp_path / "session" / "cams"
    cams_dir.mkdir(parents=True)
    video_path = cams_dir / "Cam1.mp4"
    video_path.write_bytes(b"video")

    info, findings = mod.inspect_camera_csv(video_path, expected_frame_count=3)

    assert info.status == "warn"
    assert any(f.code == "video.camera_csv_missing" for f in findings)


def test_inspect_camera_csv_fails_on_schema_error(tmp_path: Path) -> None:
    cams_dir = tmp_path / "session" / "cams"
    cams_dir.mkdir(parents=True)
    video_path = cams_dir / "Cam1.mp4"
    csv_path = cams_dir / "Cam1_meta.csv"
    video_path.write_bytes(b"video")
    csv_path.write_text("frame_id,timestamp\n1,100\n", encoding="utf-8")

    info, findings = mod.inspect_camera_csv(video_path, expected_frame_count=1)

    assert info.status == "fail"
    assert info.schema_ok is False
    assert info.missing_columns == ["timestamp_sys"]
    assert any(f.code == "video.camera_csv_schema" for f in findings)


def test_build_video_report_fails_media_when_camera_csv_row_count_mismatches(monkeypatch, tmp_path: Path) -> None:
    cams_dir = tmp_path / "session" / "cams"
    cams_dir.mkdir(parents=True)
    video_path = cams_dir / "Cam1.mp4"
    csv_path = cams_dir / "Cam1_meta.csv"
    video_path.write_bytes(b"video")
    csv_path.write_text(
        "\n".join(
            [
                "frame_id,timestamp,timestamp_sys",
                "1,100,1100",
                "2,200,1200",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(video_mod, "inspect_stream", lambda _: (StreamInfo(status="pass", codec="h264", nb_frames=3), []))

    report = video_mod.build_video_report(
        video_path,
        include_timing=False,
        include_gop=False,
        include_decode=False,
    )

    assert report.camera_csv.status == "fail"
    assert report.media_status == "fail"
    assert report.overall_status == "fail"
    assert any(f.code == "video.camera_csv_row_count_mismatch" for f in report.findings)
