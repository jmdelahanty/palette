from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.diagnostics.video.models import (
    BatchDiagnosticsReport,
    BatchSummary,
    FileInfo,
    RecordingDiagnosticsSummary,
    StreamInfo,
    VideoDiagnosticsReport,
)
from fisheye.utils import repair_trimmed_video_sidecars as mod


def _write_csv(path: Path, rows: int) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame_id", "timestamp", "timestamp_sys"])
        writer.writeheader()
        for idx in range(1, rows + 1):
            writer.writerow(
                {
                    "frame_id": idx,
                    "timestamp": 1000 + idx,
                    "timestamp_sys": 2000 + idx,
                }
            )


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _make_recording(tmp_path: Path) -> tuple[Path, Path, Path]:
    recording_dir = tmp_path / "recordings" / "sleepyfish_2026_05_05_17_45_30_cam2010093"
    cams_dir = recording_dir / "cams"
    derived_dir = recording_dir / "derived"
    cams_dir.mkdir(parents=True)
    derived_dir.mkdir()
    video = cams_dir / "Cam2010093_sleepyfish_2026_05_05_17_45_30_cam2010093.mp4"
    video.write_bytes(b"video")
    camera_csv = cams_dir / "Cam2010093_sleepyfish_2026_05_05_17_45_30_cam2010093_meta.csv"
    _write_csv(camera_csv, rows=5)
    keyframe = cams_dir / "Cam2010093_sleepyfish_2026_05_05_17_45_30_cam2010093_keyframe.json"
    keyframe.write_text(
        json.dumps(
            {
                "codec": "hevc",
                "fps": 30,
                "total_frames": 5,
                "keyframe_frames": [0, 2, 4],
            }
        ),
        encoding="utf-8",
    )
    (recording_dir / "recording_manifest.json").write_text(
        json.dumps(
            {
                "recording_name": recording_dir.name,
                "files": {
                    "cams": [
                        f"cams/{video.name}",
                        f"cams/{camera_csv.name}",
                        f"cams/{keyframe.name}",
                    ],
                    "derived": [],
                },
                "preflight": {
                    "status": "fail",
                    "video": {"status": "fail", "media_status": "fail"},
                    "h5": None,
                },
            }
        ),
        encoding="utf-8",
    )
    return recording_dir, camera_csv, keyframe


def test_dry_run_does_not_mutate_trimmed_video_sidecars(tmp_path: Path, monkeypatch) -> None:
    recording_dir, camera_csv, keyframe = _make_recording(tmp_path)
    monkeypatch.setattr(
        mod,
        "inspect_stream",
        lambda _: (StreamInfo(status="pass", nb_frames=3, avg_fps=30.0, duration_seconds=0.1), []),
    )

    rc = mod.main([str(recording_dir), "--dry-run", "--repair-id", "testrepair"])

    assert rc == 0
    assert len(_csv_rows(camera_csv)) == 5
    assert json.loads(keyframe.read_text(encoding="utf-8"))["total_frames"] == 5
    assert not (recording_dir / "derived" / "original_sidecars").exists()


def test_apply_trims_csv_and_keyframe_json_with_backups(tmp_path: Path, monkeypatch) -> None:
    recording_dir, camera_csv, keyframe = _make_recording(tmp_path)
    monkeypatch.setattr(
        mod,
        "inspect_stream",
        lambda _: (StreamInfo(status="pass", nb_frames=3, avg_fps=30.0, duration_seconds=0.1), []),
    )

    rc = mod.main(
        [
            str(recording_dir),
            "--apply",
            "--repair-id",
            "testrepair",
            "--no-run-video-preflight",
        ]
    )

    assert rc == 0
    rows = _csv_rows(camera_csv)
    assert len(rows) == 3
    assert rows[-1]["frame_id"] == "3"

    keyframe_payload = json.loads(keyframe.read_text(encoding="utf-8"))
    assert keyframe_payload["total_frames"] == 3
    assert keyframe_payload["keyframe_frames"] == [0, 2]
    assert keyframe_payload["palette_trim_repair"]["original_total_frames"] == 5

    backup_dir = recording_dir / "derived" / "original_sidecars"
    assert (backup_dir / f"{camera_csv.name[:-4]}.pre_trim_testrepair.csv").exists()
    assert (backup_dir / f"{keyframe.name[:-5]}.pre_trim_testrepair.json").exists()

    manifest = json.loads((recording_dir / "recording_manifest.json").read_text(encoding="utf-8"))
    assert "derived/original_sidecars/Cam2010093_sleepyfish_2026_05_05_17_45_30_cam2010093_meta.pre_trim_testrepair.csv" in manifest["files"]["derived"]
    assert "derived/original_sidecars/Cam2010093_sleepyfish_2026_05_05_17_45_30_cam2010093_keyframe.pre_trim_testrepair.json" in manifest["files"]["derived"]
    assert manifest["metadata_repairs"][0]["video_frame_count"] == 3
    assert manifest["metadata_repairs"][0]["camera_metadata_csv"]["original_rows"] == 5


def test_repair_falls_back_to_legacy_derived_keyframe_location(tmp_path: Path, monkeypatch) -> None:
    recording_dir, _, keyframe = _make_recording(tmp_path)
    legacy_keyframe = recording_dir / "derived" / keyframe.name
    legacy_keyframe.write_text(keyframe.read_text(encoding="utf-8"), encoding="utf-8")
    keyframe.unlink()
    monkeypatch.setattr(
        mod,
        "inspect_stream",
        lambda _: (StreamInfo(status="pass", nb_frames=3, avg_fps=30.0, duration_seconds=0.1), []),
    )

    rc = mod.main(
        [
            str(recording_dir),
            "--apply",
            "--repair-id",
            "legacyrepair",
            "--no-run-video-preflight",
        ]
    )

    assert rc == 0
    assert json.loads(legacy_keyframe.read_text(encoding="utf-8"))["total_frames"] == 3


def test_apply_can_refresh_video_preflight(tmp_path: Path, monkeypatch) -> None:
    recording_dir, _, _ = _make_recording(tmp_path)
    monkeypatch.setattr(
        mod,
        "inspect_stream",
        lambda _: (StreamInfo(status="pass", nb_frames=3, avg_fps=30.0, duration_seconds=0.1), []),
    )

    def fake_batch_report(paths: list[Path], **_: object) -> BatchDiagnosticsReport:
        item = VideoDiagnosticsReport(
            overall_status="pass",
            file_info=FileInfo(
                path=str(recording_dir / "cams" / "Cam2010093_sleepyfish_2026_05_05_17_45_30_cam2010093.mp4"),
                exists=True,
                source_kind="cams",
                recording_root=str(recording_dir),
            ),
            media_status="pass",
            tooling_status="pass",
        )
        return BatchDiagnosticsReport(
            overall_status="pass",
            roots=[str(path) for path in paths],
            recursive=True,
            items=[item],
            summary=BatchSummary(scanned=1, passed=1),
            recordings=[
                RecordingDiagnosticsSummary(
                    recording_root=str(recording_dir),
                    overall_status="pass",
                    media_status="pass",
                    tooling_status="pass",
                    item_count=1,
                )
            ],
        )

    monkeypatch.setattr(mod, "build_batch_report", fake_batch_report)

    rc = mod.main([str(recording_dir), "--apply", "--repair-id", "testrepair"])

    assert rc == 0
    manifest = json.loads((recording_dir / "recording_manifest.json").read_text(encoding="utf-8"))
    assert manifest["preflight"]["status"] == "pass"
    assert manifest["preflight"]["video"]["media_status"] == "pass"
