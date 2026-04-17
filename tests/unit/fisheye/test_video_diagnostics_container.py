from __future__ import annotations

from pathlib import Path

from fisheye.diagnostics.video import container as mod


def _atom(atom_type: bytes, payload: bytes = b"") -> bytes:
    size = 8 + len(payload)
    return size.to_bytes(4, "big") + atom_type + payload


def test_inspect_container_warns_when_hevc_is_missing_stss(tmp_path: Path, monkeypatch) -> None:
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(
        _atom(b"ftyp", b"isom0000")
        + _atom(b"mdat", b"12345678")
        + _atom(b"moov", b"trakxxxxsttsyyyy")
    )

    monkeypatch.setattr(mod, "_probe_codec_name", lambda _: "hevc")

    info, findings = mod.inspect_container(video_path)

    assert info.status == "warn"
    assert info.codec == "hevc"
    assert info.has_stss is False
    assert info.needs_fix is True
    assert any(f.code == "video.hevc_missing_stss" for f in findings)


def test_inspect_container_passes_non_hevc_without_stss_requirement(tmp_path: Path, monkeypatch) -> None:
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(
        _atom(b"ftyp", b"isom0000")
        + _atom(b"mdat", b"12345678")
        + _atom(b"moov", b"trakxxxx")
    )

    monkeypatch.setattr(mod, "_probe_codec_name", lambda _: "h264")

    info, findings = mod.inspect_container(video_path)

    assert info.status == "pass"
    assert info.codec == "h264"
    assert info.needs_fix is False
    assert findings == []
