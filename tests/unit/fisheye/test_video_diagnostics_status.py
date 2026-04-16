from __future__ import annotations

from pathlib import Path

from fisheye.diagnostics import video as mod
from fisheye.diagnostics.video.models import BackendDecodeReport, Finding, GOPInfo, StreamInfo, TimingInfo


def test_build_video_report_keeps_media_status_clean_when_only_tooling_fails(monkeypatch, tmp_path: Path) -> None:
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"video")

    monkeypatch.setattr(mod, "inspect_stream", lambda _: (StreamInfo(status="pass", codec="h264"), []))
    monkeypatch.setattr(mod, "probe_frame_payload", lambda *_args, **_kwargs: [{"pts_time": "0.0", "pict_type": "I", "key_frame": 1}])
    monkeypatch.setattr(mod, "analyze_timing_frames", lambda *_args, **_kwargs: (TimingInfo(status="pass"), []))
    monkeypatch.setattr(mod, "analyze_gop_frames", lambda *_args, **_kwargs: (GOPInfo(status="pass"), []))
    monkeypatch.setattr(
        mod,
        "inspect_decode",
        lambda *_args, **_kwargs: (
            [
                BackendDecodeReport(backend="opencv", status="pass", available=True, open_ok=True),
                BackendDecodeReport(backend="decord", status="error", available=False, error="decord missing"),
            ],
            [
                Finding(
                    severity="error",
                    code="video.decord_unavailable",
                    summary="Decord is unavailable in this environment for decode diagnostics.",
                    component="decode",
                    kind="tooling",
                )
            ],
        ),
    )

    report = mod.build_video_report(video_path)

    assert report.media_status == "pass"
    assert report.tooling_status == "error"
    assert report.overall_status == "pass"


def test_build_video_report_uses_media_failure_for_default_verdict(monkeypatch, tmp_path: Path) -> None:
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"video")

    monkeypatch.setattr(mod, "inspect_stream", lambda _: (StreamInfo(status="pass", codec="h264"), []))
    monkeypatch.setattr(mod, "probe_frame_payload", lambda *_args, **_kwargs: [{"pts_time": "0.0", "pict_type": "I", "key_frame": 1}])
    monkeypatch.setattr(
        mod,
        "analyze_timing_frames",
        lambda *_args, **_kwargs: (
            TimingInfo(status="fail"),
            [Finding(severity="fail", code="video.pts_non_monotonic", summary="PTS values are not monotonic.")],
        ),
    )
    monkeypatch.setattr(mod, "analyze_gop_frames", lambda *_args, **_kwargs: (GOPInfo(status="pass"), []))
    monkeypatch.setattr(
        mod,
        "inspect_decode",
        lambda *_args, **_kwargs: ([BackendDecodeReport(backend="opencv", status="pass", available=True, open_ok=True)], []),
    )

    report = mod.build_video_report(video_path)

    assert report.media_status == "fail"
    assert report.tooling_status == "pass"
    assert report.overall_status == "fail"
