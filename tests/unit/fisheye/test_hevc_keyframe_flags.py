from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.diagnostics.video import container as hevc_keyframe_flags


def _atom(atom_type: bytes, payload: bytes = b"") -> bytes:
    size = 8 + len(payload)
    return size.to_bytes(4, "big") + atom_type + payload


def test_check_hevc_keyframe_flags_detects_stss_for_hevc(tmp_path: Path, monkeypatch) -> None:
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(
        _atom(b"ftyp", b"isom0000")
        + _atom(b"mdat", b"12345678")
        + _atom(b"moov", b"trakxxxxstssyyyy")
    )

    monkeypatch.setattr(hevc_keyframe_flags, "_probe_codec_name", lambda _: "hevc")
    result = hevc_keyframe_flags.check_hevc_keyframe_flags(video_path)

    assert result["codec"] == "hevc"
    assert result["has_stss"] is True
    assert result["needs_fix"] is False


def test_check_hevc_keyframe_flags_flags_missing_stss_for_hevc(tmp_path: Path, monkeypatch) -> None:
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(
        _atom(b"ftyp", b"isom0000")
        + _atom(b"mdat", b"12345678")
        + _atom(b"moov", b"trakxxxxsttsyyyy")
    )

    monkeypatch.setattr(hevc_keyframe_flags, "_probe_codec_name", lambda _: "hevc")
    result = hevc_keyframe_flags.check_hevc_keyframe_flags(video_path)

    assert result["codec"] == "hevc"
    assert result["has_stss"] is False
    assert result["needs_fix"] is True


def test_check_hevc_keyframe_flags_normalizes_hevc_tag_codec(tmp_path: Path, monkeypatch) -> None:
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(
        _atom(b"ftyp", b"isom0000")
        + _atom(b"moov", b"trakxxxxstssyyyy")
    )

    monkeypatch.setattr(hevc_keyframe_flags, "_probe_codec_name", lambda _: "hvc1")
    result = hevc_keyframe_flags.check_hevc_keyframe_flags(video_path)

    assert result["codec"] == "hevc"
    assert result["needs_fix"] is False


def test_check_hevc_keyframe_flags_non_hevc_does_not_require_fix(tmp_path: Path, monkeypatch) -> None:
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(
        _atom(b"ftyp", b"isom0000")
        + _atom(b"mdat", b"12345678")
        + _atom(b"moov", b"trakxxxx")
    )

    monkeypatch.setattr(hevc_keyframe_flags, "_probe_codec_name", lambda _: "h264")
    result = hevc_keyframe_flags.check_hevc_keyframe_flags(video_path)

    assert result["codec"] == "h264"
    assert result["needs_fix"] is False


def test_check_hevc_keyframe_flags_reports_missing_moov_for_hevc(tmp_path: Path, monkeypatch) -> None:
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(_atom(b"ftyp", b"isom0000") + _atom(b"mdat", b"12345678"))

    monkeypatch.setattr(hevc_keyframe_flags, "_probe_codec_name", lambda _: "hevc")
    result = hevc_keyframe_flags.check_hevc_keyframe_flags(video_path)

    assert result["codec"] == "hevc"
    assert result["has_stss"] is False
    assert result["needs_fix"] is True
    assert "moov" in result["message"]
