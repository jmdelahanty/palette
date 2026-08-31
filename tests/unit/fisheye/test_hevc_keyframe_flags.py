from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.diagnostics.video import container as hevc_keyframe_flags


def _atom(atom_type: bytes, payload: bytes = b"") -> bytes:
    size = 8 + len(payload)
    return size.to_bytes(4, "big") + atom_type + payload


def _mp4(*, has_stss: bool) -> bytes:
    hdlr = _atom(b"hdlr", b"\x00" * 8 + b"vide" + b"\x00" * 12)
    stbl = _atom(b"stbl", _atom(b"stss") if has_stss else _atom(b"stts"))
    mdia = _atom(b"mdia", hdlr + _atom(b"minf", stbl))
    return _atom(b"ftyp", b"isom0000") + _atom(b"moov", _atom(b"trak", mdia))


def test_check_hevc_keyframe_flags_detects_stss_for_hevc(tmp_path: Path, monkeypatch) -> None:
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(_mp4(has_stss=True))

    monkeypatch.setattr(hevc_keyframe_flags, "_probe_codec_name", lambda _: "hevc")
    result = hevc_keyframe_flags.check_hevc_keyframe_flags(video_path)

    assert result["codec"] == "hevc"
    assert result["has_stss"] is True
    assert result["sync_sample_semantics"] == "indexed_sync_samples"
    assert result["sync_sample_proof"] == "container_declared"
    assert result["needs_fix"] is False


def test_check_hevc_keyframe_flags_treats_absent_stss_as_all_samples_sync(
    tmp_path: Path, monkeypatch
) -> None:
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(_mp4(has_stss=False))

    monkeypatch.setattr(hevc_keyframe_flags, "_probe_codec_name", lambda _: "hevc")
    result = hevc_keyframe_flags.check_hevc_keyframe_flags(video_path)

    assert result["codec"] == "hevc"
    assert result["has_stss"] is False
    assert result["sync_sample_semantics"] == "all_samples_sync"
    assert result["needs_fix"] is False
    assert "declares every sample" in result["message"]


def test_check_hevc_keyframe_flags_normalizes_hevc_tag_codec(tmp_path: Path, monkeypatch) -> None:
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(_mp4(has_stss=True))

    monkeypatch.setattr(hevc_keyframe_flags, "_probe_codec_name", lambda _: "hvc1")
    result = hevc_keyframe_flags.check_hevc_keyframe_flags(video_path)

    assert result["codec"] == "hevc"
    assert result["needs_fix"] is False


def test_check_hevc_keyframe_flags_non_hevc_does_not_require_fix(tmp_path: Path, monkeypatch) -> None:
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(_mp4(has_stss=False))

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
    assert result["has_stss"] is None
    assert result["sync_sample_semantics"] == "unreadable"
    assert result["container_inspection_status"] == "missing_moov"
    assert result["needs_fix"] is False
    assert "moov" in result["message"]
