from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

from .models import ContainerInfo, Finding

_HEVC_CODEC_ALIASES = {"hevc", "hvc1", "hev1"}
_H264_CODEC_ALIASES = {"h264", "avc1", "avc"}

CodecProbe = Callable[[Path], str]
StssScan = Callable[[Path], Tuple[bool, Optional[str]]]


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


def _read_atom_header(
    fh: Any,
    offset: int,
    file_size: int,
) -> Optional[Tuple[int, bytes, int]]:
    if offset + 8 > file_size:
        return None

    fh.seek(offset)
    header = fh.read(8)
    if len(header) != 8:
        return None

    atom_size_32 = int.from_bytes(header[:4], "big")
    atom_type = header[4:8]
    header_size = 8

    if atom_size_32 == 1:
        extended = fh.read(8)
        if len(extended) != 8:
            return None
        atom_size = int.from_bytes(extended, "big")
        header_size = 16
    elif atom_size_32 == 0:
        atom_size = file_size - offset
    else:
        atom_size = atom_size_32

    if atom_size < header_size:
        return None
    if offset + atom_size > file_size:
        return None

    return atom_size, atom_type, header_size


def _find_moov_atom(video_path: Path) -> Optional[Tuple[int, int]]:
    file_size = video_path.stat().st_size
    if file_size <= 0:
        return None

    offset = 0
    with video_path.open("rb") as fh:
        while offset < file_size:
            header = _read_atom_header(fh, offset, file_size)
            if header is None:
                return None
            atom_size, atom_type, header_size = header

            if atom_type == b"moov":
                return offset + header_size, atom_size - header_size

            next_offset = offset + atom_size
            if next_offset <= offset:
                return None
            offset = next_offset

    return None


def _region_contains_token(video_path: Path, offset: int, size: int, token: bytes) -> bool:
    chunk_size = 1024 * 1024
    overlap = max(len(token) - 1, 0)

    remaining = size
    carry = b""
    with video_path.open("rb") as fh:
        fh.seek(offset)
        while remaining > 0:
            data = fh.read(min(chunk_size, remaining))
            if not data:
                break
            haystack = carry + data
            if token in haystack:
                return True
            carry = haystack[-overlap:] if overlap else b""
            remaining -= len(data)

    return False


def _scan_stss_presence(video_path: Path) -> Tuple[bool, Optional[str]]:
    try:
        moov = _find_moov_atom(video_path)
    except OSError as exc:
        return False, f"failed to inspect MP4 atoms ({exc})"

    if moov is None:
        return False, "moov atom not found"

    moov_offset, moov_size = moov
    if moov_size <= 0:
        return False, "moov atom is empty"

    try:
        return _region_contains_token(video_path, moov_offset, moov_size, b"stss"), None
    except OSError as exc:
        return False, f"failed to read moov atom ({exc})"


def evaluate_keyframe_flags(
    video_path: Path,
    *,
    codec_name: Optional[str] = None,
    codec_probe: Optional[CodecProbe] = None,
    stss_scan: Optional[StssScan] = None,
) -> tuple[dict[str, Any], Optional[str]]:
    video_path = Path(video_path)
    if not video_path.exists() or not video_path.is_file():
        return (
            {
                "codec": "unknown",
                "has_stss": False,
                "needs_fix": False,
                "message": f"Video file not found: {video_path}",
            },
            None,
        )

    normalized_codec = _normalize_codec_name(codec_name) if codec_name is not None else _normalize_codec_name((codec_probe or _probe_codec_name)(video_path))
    has_stss, scan_error = (stss_scan or _scan_stss_presence)(video_path)
    needs_fix = normalized_codec == "hevc" and not has_stss

    if normalized_codec != "hevc":
        if normalized_codec == "unknown":
            message = "Codec unknown; HEVC keyframe flag check not applicable."
        else:
            message = f"Codec '{normalized_codec}' is not HEVC; keyframe-flag fix not required."
    elif has_stss:
        message = "HEVC stream has sync sample table (stss) in moov."
    else:
        message = "HEVC stream is missing sync sample table (stss); seeking may be unreliable."
        if scan_error:
            message = f"{message} Container scan issue: {scan_error}."

    return (
        {
            "codec": normalized_codec,
            "has_stss": bool(has_stss),
            "needs_fix": bool(needs_fix),
            "message": message,
        },
        scan_error,
    )


def check_hevc_keyframe_flags(video_path: Path, *, codec_name: Optional[str] = None) -> dict[str, Any]:
    payload, _ = evaluate_keyframe_flags(Path(video_path), codec_name=codec_name)
    return payload


def inspect_container(video_path: Path | str, *, codec_hint: Optional[str] = None) -> tuple[ContainerInfo, list[Finding]]:
    path = Path(video_path).expanduser()
    payload, scan_error = evaluate_keyframe_flags(path, codec_name=codec_hint)
    info = ContainerInfo(
        status="pass",
        codec=str(payload.get("codec") or "") or None,
        has_stss=bool(payload.get("has_stss")) if payload.get("has_stss") is not None else None,
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

    if info.codec == "hevc" and info.needs_fix:
        info.status = "warn"
        findings.append(
            Finding(
                severity="warn",
                code="video.hevc_missing_stss",
                summary="HEVC stream is missing sync sample table (stss) entries.",
                details=info.message,
                component="container",
            )
        )

    return info, findings
