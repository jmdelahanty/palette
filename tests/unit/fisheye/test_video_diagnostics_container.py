from __future__ import annotations

from pathlib import Path

from fisheye.diagnostics.video import container as mod
from fisheye.diagnostics.video.models import ContainerInfo, FileInfo, VideoDiagnosticsReport
from fisheye.diagnostics.video.remediation import build_remediation


def _atom(atom_type: bytes, payload: bytes = b"") -> bytes:
    size = 8 + len(payload)
    return size.to_bytes(4, "big") + atom_type + payload


def _mp4(*, has_stss: bool) -> bytes:
    hdlr = _atom(b"hdlr", b"\x00" * 8 + b"vide" + b"\x00" * 12)
    stbl = _atom(b"stbl", _atom(b"stss") if has_stss else _atom(b"stts"))
    mdia = _atom(b"mdia", hdlr + _atom(b"minf", stbl))
    return _atom(b"ftyp", b"isom0000") + _atom(b"moov", _atom(b"trak", mdia))


def _track(*, handler: bytes, has_stss: bool) -> bytes:
    hdlr = _atom(b"hdlr", b"\x00" * 8 + handler + b"\x00" * 12)
    stbl = _atom(b"stbl", _atom(b"stss") if has_stss else _atom(b"stts"))
    return _atom(b"trak", _atom(b"mdia", hdlr + _atom(b"minf", stbl)))


def test_inspect_container_accepts_hevc_all_samples_sync(tmp_path: Path, monkeypatch) -> None:
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(_mp4(has_stss=False))

    monkeypatch.setattr(mod, "_probe_codec_name", lambda _: "hevc")

    info, findings = mod.inspect_container(video_path)

    assert info.status == "pass"
    assert info.codec == "hevc"
    assert info.has_stss is False
    assert info.sync_sample_semantics == "all_samples_sync"
    assert info.sync_sample_proof == "container_declared"
    assert info.needs_fix is False
    assert findings == []


def test_stss_in_audio_track_does_not_override_video_all_sync(
    tmp_path: Path, monkeypatch
) -> None:
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(
        _atom(b"ftyp", b"isom0000")
        + _atom(
            b"moov",
            _track(handler=b"soun", has_stss=True)
            + _track(handler=b"vide", has_stss=False),
        )
    )
    monkeypatch.setattr(mod, "_probe_codec_name", lambda _: "hevc")

    info, findings = mod.inspect_container(video_path)

    assert info.sync_sample_semantics == "all_samples_sync"
    assert info.has_stss is False
    assert findings == []


def test_inspect_container_passes_non_hevc_without_stss_requirement(tmp_path: Path, monkeypatch) -> None:
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(_mp4(has_stss=False))

    monkeypatch.setattr(mod, "_probe_codec_name", lambda _: "h264")

    info, findings = mod.inspect_container(video_path)

    assert info.status == "pass"
    assert info.codec == "h264"
    assert info.needs_fix is False
    assert findings == []


def test_all_samples_sync_does_not_offer_reencode_remediation(tmp_path: Path) -> None:
    report = VideoDiagnosticsReport(
        overall_status="warn",
        file_info=FileInfo(path=str(tmp_path / "valid_all_i.mp4"), exists=True),
        container=ContainerInfo(
            status="pass",
            codec="hevc",
            has_stss=False,
            sync_sample_semantics="all_samples_sync",
            sync_sample_proof="container_declared",
            container_inspection_status="ok",
            needs_fix=False,
        ),
    )

    assert build_remediation(report) == []


def test_orange_contradiction_remediation_never_rewrites_recording(
    tmp_path: Path,
) -> None:
    report = VideoDiagnosticsReport(
        overall_status="fail",
        file_info=FileInfo(path=str(tmp_path / "crop.mp4"), exists=True),
        container=ContainerInfo(
            status="fail",
            codec="hevc",
            has_stss=False,
            sync_sample_semantics="all_samples_sync",
            sync_sample_proof="orange_idr_sidecar_contradiction",
            container_inspection_status="ok",
            needs_fix=True,
        ),
    )

    actions = build_remediation(report)

    assert len(actions) == 1
    assert actions[0].command is None
    assert "Do not rewrite" in actions[0].description
