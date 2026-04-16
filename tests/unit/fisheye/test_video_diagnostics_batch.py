from __future__ import annotations

from pathlib import Path

from fisheye.diagnostics.video import batch as mod
from fisheye.diagnostics.video.models import (
    BatchDiagnosticsReport,
    BatchSummary,
    FileInfo,
    RecordingDiagnosticsSummary,
    VideoDiagnosticsReport,
)
from fisheye.diagnostics.video.render import render_batch_jsonl, render_batch_report


def test_iter_video_paths_recurses_and_deduplicates(tmp_path: Path) -> None:
    root = tmp_path / "recordings"
    nested = root / "session" / "cams"
    nested.mkdir(parents=True)
    a = nested / "a.mp4"
    b = nested / "b.mov"
    c = root / "top.mkv"
    fixed = nested / "a_fixed.mp4"
    a.write_bytes(b"a")
    b.write_bytes(b"b")
    c.write_bytes(b"c")
    fixed.write_bytes(b"fixed")

    paths = list(mod.iter_video_paths([root, a, fixed], recursive=True, source="all"))

    assert paths == sorted([a.resolve(), b.resolve(), c.resolve()])


def test_iter_video_paths_filters_by_source_kind(tmp_path: Path) -> None:
    root = tmp_path / "recordings"
    cams = root / "session" / "cams"
    raw = root / "session" / "raw"
    other = root / "session" / "exports"
    cams.mkdir(parents=True)
    raw.mkdir(parents=True)
    other.mkdir(parents=True)
    cams_file = cams / "cam.mp4"
    raw_file = raw / "raw.mp4"
    other_file = other / "other.mp4"
    cams_file.write_bytes(b"cams")
    raw_file.write_bytes(b"raw")
    other_file.write_bytes(b"other")

    assert list(mod.iter_video_paths([root], recursive=True, source="cams")) == [cams_file.resolve()]
    assert list(mod.iter_video_paths([root], recursive=True, source="raw")) == [raw_file.resolve()]
    assert list(mod.iter_video_paths([root], recursive=True, source="other")) == [other_file.resolve()]


def test_build_batch_report_summarizes_statuses(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "recordings"
    cams = root / "cams"
    raw = root / "raw"
    cams.mkdir(parents=True)
    raw.mkdir(parents=True)
    a = cams / "a.mp4"
    b = raw / "b.mp4"
    a.write_bytes(b"a")
    b.write_bytes(b"b")

    def _fake_build_video_report(video_path: Path, **_: object) -> VideoDiagnosticsReport:
        status = "pass" if video_path.name == "a.mp4" else "warn"
        return VideoDiagnosticsReport(
            overall_status=status,
            file_info=FileInfo(
                path=str(video_path),
                exists=True,
                source_kind="cams" if "cams" in video_path.parts else "raw",
                recording_root=str(root),
            ),
        )

    monkeypatch.setattr(mod, "build_video_report", _fake_build_video_report)

    report = mod.build_batch_report([root], recursive=True, source="all")

    assert report.overall_status == "warn"
    assert report.summary.scanned == 2
    assert report.summary.passed == 1
    assert report.summary.warned == 1
    assert report.summary.source_counts == {"cams": 1, "raw": 1, "other": 0}
    assert report.summary.recording_counts == {"pass": 0, "warn": 1, "fail": 0, "error": 0, "skip": 0}
    assert report.summary.tooling_counts == {"pass": 2, "warn": 0, "fail": 0, "error": 0, "skip": 0}
    assert report.summary.tooling_recording_counts == {"pass": 1, "warn": 0, "fail": 0, "error": 0, "skip": 0}
    assert len(report.recordings) == 1
    assert report.recordings[0].recording_root == str(root)
    assert report.recordings[0].overall_status == "warn"
    assert report.recordings[0].media_status == "warn"
    assert report.recordings[0].tooling_status == "pass"
    assert report.recordings[0].item_count == 2
    assert report.recordings[0].source_counts == {"cams": 1, "raw": 1, "other": 0}


def test_render_batch_report_groups_items_by_recording_root() -> None:
    recording_root = "/tmp/recording_a"
    report = BatchDiagnosticsReport(
        overall_status="warn",
        roots=[recording_root],
        recursive=True,
        summary=BatchSummary(
            scanned=2,
            warned=2,
            source_counts={"cams": 1, "raw": 1, "other": 0},
            recording_counts={"pass": 0, "warn": 1, "fail": 0, "error": 0, "skip": 0},
            tooling_counts={"pass": 0, "warn": 0, "fail": 0, "error": 2, "skip": 0},
            tooling_recording_counts={"pass": 0, "warn": 0, "fail": 0, "error": 1, "skip": 0},
        ),
        recordings=[
            RecordingDiagnosticsSummary(
                recording_root=recording_root,
                overall_status="warn",
                media_status="warn",
                tooling_status="error",
                item_count=2,
                source_counts={"cams": 1, "raw": 1, "other": 0},
            )
        ],
        items=[
            VideoDiagnosticsReport(
                overall_status="warn",
                media_status="warn",
                tooling_status="error",
                file_info=FileInfo(
                    path=f"{recording_root}/cams/cam.mp4",
                    exists=True,
                    source_kind="cams",
                    recording_root=recording_root,
                ),
            ),
            VideoDiagnosticsReport(
                overall_status="warn",
                media_status="warn",
                tooling_status="error",
                file_info=FileInfo(
                    path=f"{recording_root}/raw/raw.mp4",
                    exists=True,
                    source_kind="raw",
                    recording_root=recording_root,
                ),
            ),
        ],
    )

    text = render_batch_report(report)

    assert "  media_files: pass=0, warn=2, fail=0, error=0, skip=0" in text
    assert "  tooling_files: pass=0, warn=0, fail=0, error=2, skip=0" in text
    assert "  media_recordings: pass=0, warn=1, fail=0, error=0, skip=0" in text
    assert "  tooling_recordings: pass=0, warn=0, fail=0, error=1, skip=0" in text
    assert (
        f"  Recording [media=warn; tooling=error]: {recording_root} (items=2; sources: cams=1, raw=1, other=0)"
        in text
    )
    assert "    [media=warn; tooling=error] [cams] cams/cam.mp4" in text
    assert "    [media=warn; tooling=error] [raw] raw/raw.mp4" in text


def test_render_batch_jsonl_emits_one_item_per_line() -> None:
    recording_root = "/tmp/recording_a"
    report = BatchDiagnosticsReport(
        overall_status="warn",
        roots=[recording_root],
        recursive=True,
        items=[
            VideoDiagnosticsReport(
                overall_status="pass",
                media_status="pass",
                tooling_status="pass",
                file_info=FileInfo(
                    path=f"{recording_root}/cams/cam.mp4",
                    exists=True,
                    source_kind="cams",
                    recording_root=recording_root,
                ),
            ),
            VideoDiagnosticsReport(
                overall_status="pass",
                media_status="pass",
                tooling_status="error",
                file_info=FileInfo(
                    path=f"{recording_root}/raw/raw.mp4",
                    exists=True,
                    source_kind="raw",
                    recording_root=recording_root,
                ),
            ),
        ],
    )

    text = render_batch_jsonl(report)

    lines = [line for line in text.splitlines() if line]
    assert len(lines) == 2
    assert '"overall_status": "pass"' in lines[0]
    assert '"media_status": "pass"' in lines[0]
    assert '"tooling_status": "pass"' in lines[0]
    assert '"source_kind": "cams"' in lines[0]
    assert '"overall_status": "pass"' in lines[1]
    assert '"media_status": "pass"' in lines[1]
    assert '"tooling_status": "error"' in lines[1]
    assert '"source_kind": "raw"' in lines[1]
