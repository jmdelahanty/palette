from __future__ import annotations

from fisheye.diagnostics.video import timing as mod


def test_analyze_timing_frames_passes_monotonic_pts() -> None:
    frames = [
        {"pkt_pts_time": "0.000"},
        {"pkt_pts_time": "0.016"},
        {"pkt_pts_time": "0.032"},
    ]

    info, findings = mod.analyze_timing_frames(frames, scope="sampled")

    assert info.status == "pass"
    assert info.pts_monotonic is True
    assert info.gap_count == 0
    assert findings == []


def test_analyze_timing_frames_fails_non_monotonic_pts() -> None:
    frames = [
        {"pkt_pts_time": "0.000"},
        {"pkt_pts_time": "0.032"},
        {"pkt_pts_time": "0.016"},
    ]

    info, findings = mod.analyze_timing_frames(frames, scope="sampled")

    assert info.status == "fail"
    assert info.pts_monotonic is False
    assert any(f.code == "video.pts_non_monotonic" for f in findings)


def test_analyze_timing_frames_warns_on_non_monotonic_dts_when_pts_ok() -> None:
    frames = [
        {"pkt_pts_time": "0.000", "pkt_dts_time": "0.000"},
        {"pkt_pts_time": "0.016", "pkt_dts_time": "0.032"},
        {"pkt_pts_time": "0.032", "pkt_dts_time": "0.016"},
    ]

    info, findings = mod.analyze_timing_frames(frames, scope="sampled")

    assert info.status == "warn"
    assert info.pts_monotonic is True
    assert info.dts_monotonic is False
    assert any(f.code == "video.dts_non_monotonic" for f in findings)

