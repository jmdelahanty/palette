from __future__ import annotations

import json
from pathlib import Path

import pytest

from fisheye.diagnostics.video import container
from fisheye.diagnostics.video.orange_sync import OrangeCropSyncEvidence


def _atom(atom_type: bytes, payload: bytes = b"") -> bytes:
    return (8 + len(payload)).to_bytes(4, "big") + atom_type + payload


def _write_mp4(path: Path, *, has_stss: bool) -> None:
    hdlr = _atom(b"hdlr", b"\x00" * 8 + b"vide" + b"\x00" * 12)
    stbl = _atom(b"stbl", _atom(b"stss") if has_stss else _atom(b"stts"))
    mdia = _atom(b"mdia", hdlr + _atom(b"minf", stbl))
    path.write_bytes(_atom(b"ftyp", b"isom0000") + _atom(b"moov", _atom(b"trak", mdia)))


def _write_evidence(
    tmp_path: Path,
    *,
    frames_encoded: int = 4,
    total_frames: int = 4,
    keyframe_frames: list[int] | None = None,
    resolved_gop_length: int = 1,
) -> OrangeCropSyncEvidence:
    summary = tmp_path / "crop_summary.json"
    keyframes = tmp_path / "crop_keyframe.json"
    summary.write_text(
        json.dumps(
            {
                "output_kind": "crop",
                "stream_kind": "crop",
                "tuning": "lossless",
                "resolved_gop_length": resolved_gop_length,
                "frames_encoded": frames_encoded,
                "outputs": {"mp4_keyframe": str(keyframes)},
            }
        ),
        encoding="utf-8",
    )
    keyframes.write_text(
        json.dumps(
            {
                "total_frames": total_frames,
                "keyframe_frames": (
                    list(range(total_frames))
                    if keyframe_frames is None
                    else keyframe_frames
                ),
            }
        ),
        encoding="utf-8",
    )
    return OrangeCropSyncEvidence(
        summary_path=summary,
        keyframe_path=keyframes,
        declared_output_kind="crop",
        declared_stream_kind="crop",
        declared_tuning="lossless",
        declared_frame_count=frames_encoded,
        declared_packet_count=frames_encoded,
    )


def test_orange_gop1_all_i_crop_verifies_without_stss(
    tmp_path: Path, monkeypatch
) -> None:
    video = tmp_path / "crop.mp4"
    _write_mp4(video, has_stss=False)
    evidence = _write_evidence(tmp_path)
    monkeypatch.setattr(container, "_probe_codec_name", lambda _: "hevc")

    result = container.check_hevc_keyframe_flags(
        video, orange_crop_evidence=evidence
    )

    assert result["sync_sample_semantics"] == "all_samples_sync"
    assert result["sync_sample_proof"] == "orange_idr_sidecar_verified"
    assert result["needs_fix"] is False
    assert result["orange_evidence"]["frames_encoded"] == 4
    assert result["orange_evidence"]["keyframe_count"] == 4


def test_orange_crop_sidecars_are_discovered_by_exact_sibling_names(
    tmp_path: Path, monkeypatch
) -> None:
    video = tmp_path / "Cam1_crop_external.mp4"
    _write_mp4(video, has_stss=False)
    evidence = _write_evidence(tmp_path)
    Path(evidence.summary_path).rename(
        tmp_path / "Cam1_crop_external_summary.json"
    )
    Path(evidence.keyframe_path).rename(
        tmp_path / "Cam1_crop_external_keyframe.json"
    )
    monkeypatch.setattr(container, "_probe_codec_name", lambda _: "hevc")

    result = container.check_hevc_keyframe_flags(video)

    assert result["sync_sample_proof"] == "orange_idr_sidecar_verified"
    assert result["needs_fix"] is False


@pytest.mark.parametrize(
    ("frames_encoded", "total_frames", "indices", "error_fragment"),
    [
        (4, 3, [0, 1, 2], "total_frames contradicts"),
        (4, 4, [0, 1, 3, 4], "not contiguous"),
        (4, 4, [0, 1, 2], "does not cover every"),
    ],
)
def test_orange_gop1_keyframe_contradictions_fail(
    tmp_path: Path,
    monkeypatch,
    frames_encoded: int,
    total_frames: int,
    indices: list[int],
    error_fragment: str,
) -> None:
    video = tmp_path / "crop.mp4"
    _write_mp4(video, has_stss=False)
    evidence = _write_evidence(
        tmp_path,
        frames_encoded=frames_encoded,
        total_frames=total_frames,
        keyframe_frames=indices,
    )
    monkeypatch.setattr(container, "_probe_codec_name", lambda _: "hevc")

    result = container.check_hevc_keyframe_flags(
        video, orange_crop_evidence=evidence
    )

    assert result["sync_sample_semantics"] == "all_samples_sync"
    assert result["sync_sample_proof"] == "orange_idr_sidecar_contradiction"
    assert result["needs_fix"] is True
    assert error_fragment in result["orange_evidence"]["error"]


def test_orange_interframe_stream_cannot_claim_all_samples_sync(
    tmp_path: Path, monkeypatch
) -> None:
    video = tmp_path / "crop.mp4"
    _write_mp4(video, has_stss=False)
    evidence = _write_evidence(tmp_path, resolved_gop_length=30)
    monkeypatch.setattr(container, "_probe_codec_name", lambda _: "hevc")

    result = container.check_hevc_keyframe_flags(
        video, orange_crop_evidence=evidence
    )

    assert result["sync_sample_proof"] == "orange_idr_sidecar_contradiction"
    assert "inter-frame GOP length" in result["orange_evidence"]["error"]


def test_large_keyframe_sidecar_is_validated_incrementally(
    tmp_path: Path, monkeypatch
) -> None:
    frame_count = 100_000
    video = tmp_path / "crop.mp4"
    _write_mp4(video, has_stss=False)
    evidence = _write_evidence(
        tmp_path,
        frames_encoded=frame_count,
        total_frames=frame_count,
    )
    monkeypatch.setattr(container, "_probe_codec_name", lambda _: "hevc")

    result = container.check_hevc_keyframe_flags(
        video, orange_crop_evidence=evidence
    )

    assert result["sync_sample_proof"] == "orange_idr_sidecar_verified"
    assert result["orange_evidence"]["keyframe_count"] == frame_count


def test_historical_all_sync_mp4_without_orange_summary_is_inspectable(
    tmp_path: Path, monkeypatch
) -> None:
    video = tmp_path / "historical.mp4"
    _write_mp4(video, has_stss=False)
    monkeypatch.setattr(container, "_probe_codec_name", lambda _: "hevc")

    result = container.check_hevc_keyframe_flags(video)

    assert result["sync_sample_semantics"] == "all_samples_sync"
    assert result["sync_sample_proof"] == "container_declared"
    assert result["orange_evidence"] is None
    assert result["needs_fix"] is False


def test_malformed_mp4_atom_layout_remains_inspection_error(
    tmp_path: Path, monkeypatch
) -> None:
    video = tmp_path / "broken.mp4"
    video.write_bytes((100).to_bytes(4, "big") + b"moov" + b"short")
    monkeypatch.setattr(container, "_probe_codec_name", lambda _: "hevc")

    info, findings = container.inspect_container(video)

    assert info.status == "error"
    assert info.sync_sample_semantics == "unreadable"
    assert info.container_inspection_status == "malformed_atom_layout"
    assert [finding.code for finding in findings] == [
        "video.container_inspection_error"
    ]
