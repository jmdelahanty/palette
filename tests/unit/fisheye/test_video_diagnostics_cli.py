from __future__ import annotations

from pathlib import Path

from fisheye.diagnostics.video import cli
from fisheye.diagnostics.video.models import BatchDiagnosticsReport, BatchSummary, FileInfo, VideoDiagnosticsReport


def test_cli_report_json(monkeypatch, tmp_path: Path, capsys) -> None:
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"video")

    def _fake_build_report(path: Path, **_: object) -> VideoDiagnosticsReport:
        return VideoDiagnosticsReport(overall_status="pass", file_info=FileInfo(path=str(path), exists=True))

    monkeypatch.setattr(cli, "build_video_report", _fake_build_report)

    rc = cli.main(["report", str(video_path), "--json"])
    out = capsys.readouterr().out

    assert rc == 0
    assert '"overall_status": "pass"' in out


def test_cli_report_uses_quick_defaults(monkeypatch, tmp_path: Path) -> None:
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"video")
    seen: dict[str, object] = {}

    def _fake_build_report(path: Path, **kwargs: object) -> VideoDiagnosticsReport:
        seen["path"] = path
        seen.update(kwargs)
        return VideoDiagnosticsReport(overall_status="pass", file_info=FileInfo(path=str(path), exists=True))

    monkeypatch.setattr(cli, "build_video_report", _fake_build_report)

    rc = cli.main(["report", str(video_path)])

    assert rc == 0
    assert seen["path"] == video_path
    assert seen["full_scan"] is False
    assert seen["sample_frames"] == cli.DEFAULT_SAMPLE_FRAMES
    assert seen["decode_frames"] == cli.DEFAULT_DECODE_FRAMES
    assert seen["seek_samples"] == cli.DEFAULT_SEEK_SAMPLES


def test_cli_batch_uses_recursive_defaults(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "recordings"
    root.mkdir()
    seen: dict[str, object] = {}

    def _fake_build_batch_report(paths, **kwargs: object) -> BatchDiagnosticsReport:
        seen["paths"] = list(paths)
        seen.update(kwargs)
        return BatchDiagnosticsReport(
            overall_status="pass",
            roots=[str(root)],
            recursive=True,
            summary=BatchSummary(scanned=1, passed=1),
        )

    monkeypatch.setattr(cli, "build_batch_report", _fake_build_batch_report)

    rc = cli.main(["batch", str(root)])

    assert rc == 0
    assert seen["paths"] == [root]
    assert seen["recursive"] is True
    assert seen["source"] == "all"
    assert seen["sample_frames"] == cli.DEFAULT_SAMPLE_FRAMES


def test_cli_batch_writes_jsonl_output(monkeypatch, tmp_path: Path, capsys) -> None:
    root = tmp_path / "recordings"
    root.mkdir()
    output_path = tmp_path / "reports" / "batch.jsonl"

    def _fake_build_batch_report(paths, **kwargs: object) -> BatchDiagnosticsReport:
        del paths, kwargs
        return BatchDiagnosticsReport(
            overall_status="pass",
            roots=[str(root)],
            recursive=True,
            summary=BatchSummary(
                scanned=1,
                passed=1,
                recording_counts={"pass": 1, "warn": 0, "fail": 0, "error": 0, "skip": 0},
                tooling_counts={"pass": 0, "warn": 0, "fail": 0, "error": 1, "skip": 0},
                tooling_recording_counts={"pass": 0, "warn": 0, "fail": 0, "error": 1, "skip": 0},
            ),
            items=[
                VideoDiagnosticsReport(
                    overall_status="pass",
                    media_status="pass",
                    tooling_status="error",
                    file_info=FileInfo(
                        path=str(root / "cams" / "cam.mp4"),
                        exists=True,
                        source_kind="cams",
                        recording_root=str(root),
                    ),
                )
            ],
        )

    monkeypatch.setattr(cli, "build_batch_report", _fake_build_batch_report)

    rc = cli.main(["batch", str(root), "--jsonl", str(output_path)])
    out = capsys.readouterr().out

    assert rc == 0
    assert "Overall: pass" in out
    assert output_path.exists()
    jsonl_text = output_path.read_text(encoding="utf-8")
    assert '"overall_status": "pass"' in jsonl_text
    assert '"media_status": "pass"' in jsonl_text
    assert '"tooling_status": "error"' in jsonl_text
    assert '"source_kind": "cams"' in jsonl_text
