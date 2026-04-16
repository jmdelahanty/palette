from __future__ import annotations

from pathlib import Path

from fisheye.diagnostics.video import probe as mod


def test_inspect_stream_parses_video_fields(monkeypatch) -> None:
    def _fake_probe(_: Path) -> dict[str, object]:
        return {
            "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "duration": "10.5", "tags": {"encoder": "nvenc"}},
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "hevc",
                    "profile": "Main",
                    "level": 150,
                    "width": 1920,
                    "height": 1080,
                    "pix_fmt": "yuv420p",
                    "avg_frame_rate": "60/1",
                    "nb_frames": "630",
                    "has_b_frames": "0",
                }
            ],
        }

    monkeypatch.setattr(mod, "probe_stream_payload", _fake_probe)
    info, findings = mod.inspect_stream(Path("/tmp/video.mp4"))

    assert findings == []
    assert info.status == "pass"
    assert info.codec == "hevc"
    assert info.width == 1920
    assert info.height == 1080
    assert info.avg_fps == 60.0
    assert info.has_b_frames is False
    assert info.format_tags["encoder"] == "nvenc"


def test_build_file_info_sets_source_and_recording_root(tmp_path: Path) -> None:
    video_path = tmp_path / "session" / "cams" / "cam.mp4"
    video_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"video")

    info = mod.build_file_info(video_path)

    assert info.exists is True
    assert info.source_kind == "cams"
    assert info.recording_root == str(tmp_path / "session")
