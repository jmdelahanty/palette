from __future__ import annotations

from pathlib import Path
from typing import Any

from .ffprobe import is_ffprobe_tooling_issue, probe_stream_payload
from .models import FileInfo, Finding, StreamInfo


def classify_video_source(video_path: Path) -> str:
    for part in reversed(video_path.parts[:-1]):
        lowered = part.lower()
        if lowered == "cams":
            return "cams"
        if lowered == "raw":
            return "raw"
    return "other"


def infer_recording_root(video_path: Path) -> str:
    for ancestor in video_path.parents:
        if ancestor.name.lower() in {"cams", "raw"}:
            return str(ancestor.parent)
    return str(video_path.parent)


def build_file_info(video_path: Path) -> FileInfo:
    exists = video_path.exists()
    if not exists:
        return FileInfo(
            path=str(video_path),
            exists=False,
            source_kind=classify_video_source(video_path),
            recording_root=infer_recording_root(video_path),
        )
    stat = video_path.stat()
    return FileInfo(
        path=str(video_path),
        exists=True,
        size_bytes=int(stat.st_size),
        modified_time=float(stat.st_mtime),
        source_kind=classify_video_source(video_path),
        recording_root=infer_recording_root(video_path),
    )


def _parse_fraction(value: object) -> float | None:
    text = str(value).strip()
    if not text:
        return None
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        try:
            num = float(numerator)
            den = float(denominator)
        except ValueError:
            return None
        if den == 0:
            return None
        return num / den
    try:
        return float(text)
    except ValueError:
        return None


def _coerce_int(value: object) -> int | None:
    if value in (None, "", "N/A"):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: object) -> float | None:
    if value in (None, "", "N/A"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_bool(value: object) -> bool | None:
    if value in (None, "", "N/A"):
        return None
    if isinstance(value, bool):
        return value
    try:
        return bool(int(value))
    except (TypeError, ValueError):
        text = str(value).strip().lower()
        if text in {"true", "yes"}:
            return True
        if text in {"false", "no"}:
            return False
        return None


def inspect_stream(video_path: Path) -> tuple[StreamInfo, list[Finding]]:
    try:
        payload = probe_stream_payload(video_path)
    except Exception as exc:
        message = str(exc)
        tooling_issue = is_ffprobe_tooling_issue(message)
        return (
            StreamInfo(status="error" if tooling_issue else "fail", error=message),
            [
                Finding(
                    severity="error" if tooling_issue else "fail",
                    code="video.ffprobe_unavailable" if tooling_issue else "video.ffprobe_stream_error",
                    summary=(
                        "ffprobe is unavailable in this environment."
                        if tooling_issue
                        else "Could not inspect stream metadata."
                    ),
                    details=message,
                    component="probe",
                    kind="tooling" if tooling_issue else "media",
                )
            ],
        )

    streams = payload.get("streams", [])
    if not isinstance(streams, list):
        streams = []
    video_stream: dict[str, Any] | None = None
    for stream in streams:
        if isinstance(stream, dict) and stream.get("codec_type") == "video":
            video_stream = stream
            break
    if video_stream is None:
        return (
            StreamInfo(status="fail", error="No video stream found."),
            [
                Finding(
                    severity="fail",
                    code="video.no_video_stream",
                    summary="ffprobe did not report a video stream.",
                    component="probe",
                )
            ],
        )

    format_payload = payload.get("format", {})
    if not isinstance(format_payload, dict):
        format_payload = {}
    tags = format_payload.get("tags", {})
    if not isinstance(tags, dict):
        tags = {}

    fps = _parse_fraction(video_stream.get("avg_frame_rate")) or _parse_fraction(video_stream.get("r_frame_rate"))
    info = StreamInfo(
        status="pass",
        container_format=str(format_payload.get("format_name") or "") or None,
        codec=str(video_stream.get("codec_name") or "") or None,
        profile=str(video_stream.get("profile") or "") or None,
        level=str(video_stream.get("level") or "") or None,
        width=_coerce_int(video_stream.get("width")),
        height=_coerce_int(video_stream.get("height")),
        pix_fmt=str(video_stream.get("pix_fmt") or "") or None,
        avg_fps=fps,
        duration_seconds=_coerce_float(video_stream.get("duration")) or _coerce_float(format_payload.get("duration")),
        nb_frames=_coerce_int(video_stream.get("nb_frames")),
        has_b_frames=_coerce_bool(video_stream.get("has_b_frames")),
        format_tags=dict(tags),
    )
    return info, []
