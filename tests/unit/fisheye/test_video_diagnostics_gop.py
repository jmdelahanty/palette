from __future__ import annotations

from fisheye.diagnostics.video import gop as mod


def test_analyze_gop_frames_warns_for_sparse_hevc_keyframes() -> None:
    frames = [
        {"key_frame": 1, "pict_type": "I", "pkt_pts_time": "0.0"},
        {"key_frame": 0, "pict_type": "B", "pkt_pts_time": "1.0"},
        {"key_frame": 0, "pict_type": "P", "pkt_pts_time": "2.0"},
        {"key_frame": 1, "pict_type": "I", "pkt_pts_time": "11.5"},
    ]

    info, findings = mod.analyze_gop_frames(frames, scope="sampled", codec="hevc", avg_fps=60.0)

    assert info.status == "warn"
    assert info.b_frames_present is True
    assert info.max_keyframe_interval_s == 11.5
    assert any(f.code == "video.sparse_keyframes" for f in findings)
    assert any(f.code == "video.hevc_b_frames" for f in findings)

