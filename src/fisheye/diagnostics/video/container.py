from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

from .models import ContainerInfo, Finding
from .orange_sync import (
    OrangeCropSyncEvidence,
    assess_orange_crop_sync_evidence,
)

_HEVC_CODEC_ALIASES = {"hevc", "hvc1", "hev1"}
_H264_CODEC_ALIASES = {"h264", "avc1", "avc"}

CodecProbe = Callable[[Path], str]
StssScan = Callable[[Path], Tuple[bool, Optional[str]]]


@dataclass(frozen=True)
class _Atom:
    atom_type: bytes
    payload_offset: int
    payload_size: int


@dataclass(frozen=True)
class _SyncSampleInspection:
    semantics: str
    has_stss: Optional[bool]
    status: str
    error: Optional[str] = None


class _AtomLayoutError(ValueError):
    pass


def _normalize_codec_name(codec_name: Optional[str]) -> str:
    if codec_name is None:
        return "unknown"
    normalized = codec_name.strip().lower()
    if not normalized:
        return "unknown"
    if normalized in _HEVC_CODEC_ALIASES:
        return "hevc"
    if normalized in _H264_CODEC_ALIASES:
        return "h264"
    return normalized


def _probe_codec_name(video_path: Path) -> str:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,codec_tag_string",
                "-of",
                "json",
                str(video_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"

    if result.returncode != 0 or not result.stdout:
        return "unknown"

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return "unknown"

    streams = payload.get("streams")
    if not isinstance(streams, list) or not streams:
        return "unknown"
    stream = streams[0]
    if not isinstance(stream, dict):
        return "unknown"

    codec_name = stream.get("codec_name")
    codec_tag = stream.get("codec_tag_string")
    return _normalize_codec_name(
        str(codec_name) if codec_name else (str(codec_tag) if codec_tag else None)
    )


def _read_atom_header(fh: Any, offset: int, region_end: int) -> Tuple[int, bytes, int]:
    if offset + 8 > region_end:
        raise _AtomLayoutError(f"truncated MP4 atom header at byte {offset}")

    fh.seek(offset)
    header = fh.read(8)
    if len(header) != 8:
        raise _AtomLayoutError(f"unreadable MP4 atom header at byte {offset}")

    atom_size_32 = int.from_bytes(header[:4], "big")
    atom_type = header[4:8]
    header_size = 8

    if atom_size_32 == 1:
        extended = fh.read(8)
        if len(extended) != 8:
            raise _AtomLayoutError(f"truncated extended MP4 atom at byte {offset}")
        atom_size = int.from_bytes(extended, "big")
        header_size = 16
    elif atom_size_32 == 0:
        atom_size = region_end - offset
    else:
        atom_size = atom_size_32

    if atom_size < header_size:
        raise _AtomLayoutError(f"invalid MP4 atom size at byte {offset}")
    if offset + atom_size > region_end:
        raise _AtomLayoutError(f"MP4 atom at byte {offset} exceeds its parent")

    return atom_size, atom_type, header_size


def _atoms(fh: Any, *, offset: int, size: int) -> list[_Atom]:
    region_end = offset + size
    atoms: list[_Atom] = []
    while offset < region_end:
        atom_size, atom_type, header_size = _read_atom_header(fh, offset, region_end)
        atoms.append(
            _Atom(
                atom_type=atom_type,
                payload_offset=offset + header_size,
                payload_size=atom_size - header_size,
            )
        )
        offset += atom_size
    return atoms


def _children(fh: Any, parent: _Atom, atom_type: bytes) -> list[_Atom]:
    return [
        atom
        for atom in _atoms(
            fh, offset=parent.payload_offset, size=parent.payload_size
        )
        if atom.atom_type == atom_type
    ]


def _one_child(fh: Any, parent: _Atom, atom_type: bytes) -> _Atom:
    matches = _children(fh, parent, atom_type)
    if len(matches) != 1:
        raise _AtomLayoutError(
            f"video track requires exactly one {atom_type.decode('ascii')} atom"
        )
    return matches[0]


def _handler_type(fh: Any, mdia: _Atom) -> bytes:
    hdlr = _one_child(fh, mdia, b"hdlr")
    if hdlr.payload_size < 12:
        raise _AtomLayoutError("hdlr atom is too small")
    fh.seek(hdlr.payload_offset + 8)
    value = fh.read(4)
    if len(value) != 4:
        raise _AtomLayoutError("hdlr atom is unreadable")
    return value


def _scan_video_track_sync_samples(video_path: Path) -> _SyncSampleInspection:
    try:
        file_size = video_path.stat().st_size
        if file_size <= 0:
            return _SyncSampleInspection(
                semantics="unreadable",
                has_stss=None,
                status="malformed_atom_layout",
                error="MP4 file is empty",
            )
        with video_path.open("rb") as fh:
            top_level = _atoms(fh, offset=0, size=file_size)
            moov_atoms = [atom for atom in top_level if atom.atom_type == b"moov"]
            if not moov_atoms:
                return _SyncSampleInspection(
                    semantics="unreadable",
                    has_stss=None,
                    status="missing_moov",
                    error="moov atom not found",
                )
            if len(moov_atoms) != 1:
                raise _AtomLayoutError("MP4 contains multiple moov atoms")
            moov = moov_atoms[0]
            for trak in _children(fh, moov, b"trak"):
                mdia = _one_child(fh, trak, b"mdia")
                if _handler_type(fh, mdia) != b"vide":
                    continue
                minf = _one_child(fh, mdia, b"minf")
                stbl = _one_child(fh, minf, b"stbl")
                has_stss = bool(_children(fh, stbl, b"stss"))
                return _SyncSampleInspection(
                    semantics=(
                        "indexed_sync_samples" if has_stss else "all_samples_sync"
                    ),
                    has_stss=has_stss,
                    status="ok",
                )
            return _SyncSampleInspection(
                semantics="unreadable",
                has_stss=None,
                status="video_track_missing",
                error="moov atom does not contain a video track",
            )
    except _AtomLayoutError as exc:
        return _SyncSampleInspection(
            semantics="unreadable",
            has_stss=None,
            status="malformed_atom_layout",
            error=str(exc),
        )
    except OSError as exc:
        return _SyncSampleInspection(
            semantics="unreadable",
            has_stss=None,
            status="unreadable",
            error=f"failed to inspect MP4 atoms ({exc})",
        )


def _discover_orange_crop_evidence(
    video_path: Path,
) -> Optional[OrangeCropSyncEvidence]:
    if not video_path.stem.endswith("_crop_external"):
        return None
    summary = video_path.with_name(f"{video_path.stem}_summary.json")
    keyframes = video_path.with_name(f"{video_path.stem}_keyframe.json")
    if not summary.exists() and not keyframes.exists():
        return None
    return OrangeCropSyncEvidence(summary_path=summary, keyframe_path=keyframes)


def evaluate_keyframe_flags(
    video_path: Path,
    *,
    codec_name: Optional[str] = None,
    codec_probe: Optional[CodecProbe] = None,
    stss_scan: Optional[StssScan] = None,
    orange_crop_evidence: Optional[OrangeCropSyncEvidence] = None,
) -> tuple[dict[str, Any], Optional[str]]:
    video_path = Path(video_path)
    if not video_path.exists() or not video_path.is_file():
        return (
            {
                "schema_id": "palette.video.sync_sample_assessment.v1",
                "codec": "unknown",
                "has_stss": None,
                "sync_sample_semantics": "unreadable",
                "sync_sample_proof": None,
                "container_inspection_status": "file_missing",
                "orange_evidence": None,
                "needs_fix": False,
                "message": f"Video file not found: {video_path}",
            },
            None,
        )

    normalized_codec = (
        _normalize_codec_name(codec_name)
        if codec_name is not None
        else _normalize_codec_name((codec_probe or _probe_codec_name)(video_path))
    )
    if stss_scan is None:
        inspection = _scan_video_track_sync_samples(video_path)
    else:
        has_stss, scan_error = stss_scan(video_path)
        inspection = _SyncSampleInspection(
            semantics=(
                "unreadable"
                if scan_error
                else ("indexed_sync_samples" if has_stss else "all_samples_sync")
            ),
            has_stss=None if scan_error else bool(has_stss),
            status="unreadable" if scan_error else "ok",
            error=scan_error,
        )

    evidence = orange_crop_evidence or _discover_orange_crop_evidence(video_path)
    orange_assessment = None
    if normalized_codec == "hevc" and evidence is not None:
        orange_assessment = assess_orange_crop_sync_evidence(
            evidence,
            all_samples_sync_declared=(
                inspection.semantics == "all_samples_sync"
            ),
        )
    proof = (
        orange_assessment.status
        if orange_assessment is not None
        else ("container_declared" if inspection.status == "ok" else None)
    )
    needs_fix = bool(
        normalized_codec == "hevc"
        and proof == "orange_idr_sidecar_contradiction"
    )

    if normalized_codec != "hevc":
        if normalized_codec == "unknown":
            message = "Codec unknown; HEVC keyframe flag check not applicable."
        else:
            message = f"Codec '{normalized_codec}' is not HEVC; keyframe-flag fix not required."
    elif inspection.semantics == "unreadable":
        message = f"HEVC MP4 sync-sample inspection failed: {inspection.error}."
    elif inspection.semantics == "indexed_sync_samples":
        message = "HEVC MP4 video track uses an stss sync-sample index."
    else:
        message = (
            "HEVC MP4 video track omits stss; ISO BMFF therefore declares "
            "every sample to be a sync sample."
        )
    if orange_assessment is not None:
        if orange_assessment.status == "orange_idr_sidecar_verified":
            message = f"{message} Orange GOP=1 IDR sidecar proof verified."
        elif orange_assessment.status == "orange_idr_sidecar_unavailable":
            message = f"{message} Independent Orange IDR proof is unavailable."
        elif orange_assessment.status == "orange_idr_sidecar_contradiction":
            message = (
                f"{message} Orange producer evidence contradicts the sync-sample "
                f"claim: {orange_assessment.error}."
            )

    return (
        {
            "schema_id": "palette.video.sync_sample_assessment.v1",
            "codec": normalized_codec,
            "has_stss": inspection.has_stss,
            "sync_sample_semantics": inspection.semantics,
            "sync_sample_proof": proof,
            "container_inspection_status": inspection.status,
            "orange_evidence": (
                orange_assessment.as_dict()
                if orange_assessment is not None
                else None
            ),
            "needs_fix": bool(needs_fix),
            "message": message,
        },
        inspection.error,
    )


def check_hevc_keyframe_flags(
    video_path: Path,
    *,
    codec_name: Optional[str] = None,
    orange_crop_evidence: Optional[OrangeCropSyncEvidence] = None,
) -> dict[str, Any]:
    payload, _ = evaluate_keyframe_flags(
        Path(video_path),
        codec_name=codec_name,
        orange_crop_evidence=orange_crop_evidence,
    )
    return payload


def inspect_container(
    video_path: Path | str,
    *,
    codec_hint: Optional[str] = None,
    orange_crop_evidence: Optional[OrangeCropSyncEvidence] = None,
) -> tuple[ContainerInfo, list[Finding]]:
    path = Path(video_path).expanduser()
    payload, scan_error = evaluate_keyframe_flags(
        path,
        codec_name=codec_hint,
        orange_crop_evidence=orange_crop_evidence,
    )
    info = ContainerInfo(
        status="pass",
        codec=str(payload.get("codec") or "") or None,
        has_stss=bool(payload.get("has_stss")) if payload.get("has_stss") is not None else None,
        sync_sample_semantics=payload.get("sync_sample_semantics"),
        sync_sample_proof=payload.get("sync_sample_proof"),
        container_inspection_status=str(
            payload.get("container_inspection_status") or ""
        )
        or None,
        orange_evidence=(
            dict(payload["orange_evidence"])
            if isinstance(payload.get("orange_evidence"), dict)
            else None
        ),
        needs_fix=bool(payload.get("needs_fix")) if payload.get("needs_fix") is not None else None,
        message=str(payload.get("message") or "") or None,
        scan_error=scan_error,
    )
    findings: list[Finding] = []

    if not path.exists() or not path.is_file():
        info.status = "fail"
        findings.append(
            Finding(
                severity="fail",
                code="video.file_missing",
                summary="Video file not found for container inspection.",
                details=info.message,
                component="container",
            )
        )
        return info, findings

    if info.codec == "hevc" and info.sync_sample_semantics == "unreadable":
        info.status = "error"
        findings.append(
            Finding(
                severity="error",
                code="video.container_inspection_error",
                summary="HEVC MP4 sync-sample semantics could not be inspected.",
                details=info.message,
                component="container",
            )
        )
    elif info.codec == "hevc" and (
        info.sync_sample_proof == "orange_idr_sidecar_contradiction"
    ):
        info.status = "fail"
        findings.append(
            Finding(
                severity="fail",
                code="video.orange_sync_evidence_contradiction",
                summary="Orange producer evidence contradicts MP4 sync-sample semantics.",
                details=info.message,
                component="container",
            )
        )

    return info, findings
