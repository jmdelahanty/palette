from __future__ import annotations

from pathlib import Path
from typing import Optional

from .camera_csv import inspect_camera_csv
from .container import inspect_container
from .decode import inspect_decode
from .ffprobe import is_ffprobe_tooling_issue, probe_frame_payload
from .gop import analyze_gop_frames
from .models import (
    CameraCsvInfo,
    compute_media_status,
    ContainerInfo,
    Finding,
    GOPInfo,
    StreamInfo,
    TimingInfo,
    VideoDiagnosticsReport,
    compute_overall_status,
    compute_tooling_status,
)
from .probe import build_file_info, inspect_stream
from .remediation import build_remediation
from .timing import analyze_timing_frames

__all__ = ["build_video_report"]


def _timing_error(scope: str, message: str) -> tuple[TimingInfo, list[Finding]]:
    tooling_issue = is_ffprobe_tooling_issue(message)
    info = TimingInfo(status="error" if tooling_issue else "fail", scope=scope, error=message)
    findings = [
        Finding(
            severity="error" if tooling_issue else "fail",
            code="video.ffprobe_unavailable" if tooling_issue else "video.frame_probe_error",
            summary=(
                "ffprobe is unavailable in this environment."
                if tooling_issue
                else "Could not collect frame-level timing metadata."
            ),
            details=message,
            component="timing",
            kind="tooling" if tooling_issue else "media",
        )
    ]
    return info, findings


def _gop_error(scope: str, message: str) -> tuple[GOPInfo, list[Finding]]:
    tooling_issue = is_ffprobe_tooling_issue(message)
    info = GOPInfo(status="error" if tooling_issue else "fail", scope=scope, error=message)
    findings = [
        Finding(
            severity="error" if tooling_issue else "fail",
            code="video.ffprobe_unavailable" if tooling_issue else "video.frame_probe_error",
            summary=(
                "ffprobe is unavailable in this environment."
                if tooling_issue
                else "Could not collect frame-level GOP metadata."
            ),
            details=message,
            component="gop",
            kind="tooling" if tooling_issue else "media",
        )
    ]
    return info, findings


def build_video_report(
    video_path: Path | str,
    *,
    include_container: bool = True,
    include_probe: bool = True,
    include_timing: bool = True,
    include_gop: bool = True,
    include_decode: bool = True,
    include_camera_csv: bool = True,
    full_scan: bool = False,
    sample_frames: int = 1000,
    decode_backend: str = "opencv",
    decode_frames: int = 100,
    seek_samples: int = 20,
) -> VideoDiagnosticsReport:
    path = Path(video_path).expanduser()
    report = VideoDiagnosticsReport(overall_status="skip", file_info=build_file_info(path))

    findings: list[Finding] = []
    stream_info = StreamInfo()
    if include_probe:
        stream_info, section_findings = inspect_stream(path)
        findings.extend(section_findings)
    report.stream_info = stream_info

    container_info = ContainerInfo()
    if include_container:
        codec_hint = stream_info.codec if stream_info.status != "skip" else None
        container_info, section_findings = inspect_container(path, codec_hint=codec_hint)
        findings.extend(section_findings)
    report.container = container_info

    scope = "full" if full_scan else "sampled"
    frames = None
    frame_error: Optional[str] = None
    if include_timing or include_gop:
        try:
            frames = probe_frame_payload(path, max_frames=None if full_scan else int(sample_frames))
        except Exception as exc:
            frame_error = str(exc)

    if include_timing:
        if frame_error is not None:
            timing_info, section_findings = _timing_error(scope, frame_error)
        else:
            timing_info, section_findings = analyze_timing_frames(frames or [], scope=scope)
        report.timing = timing_info
        findings.extend(section_findings)

    if include_gop:
        if frame_error is not None:
            gop_info, section_findings = _gop_error(scope, frame_error)
        else:
            gop_info, section_findings = analyze_gop_frames(
                frames or [],
                scope=scope,
                codec=stream_info.codec,
                avg_fps=stream_info.avg_fps,
            )
        report.gop = gop_info
        findings.extend(section_findings)

    if include_decode:
        decode_reports, section_findings = inspect_decode(
            path,
            backend=decode_backend,
            frames_to_check=int(decode_frames),
            seek_samples=int(seek_samples),
        )
        report.decode = decode_reports
        findings.extend(section_findings)

    camera_csv_info = CameraCsvInfo()
    if include_camera_csv:
        camera_csv_info, section_findings = inspect_camera_csv(
            path,
            expected_frame_count=stream_info.nb_frames,
        )
        findings.extend(section_findings)
    report.camera_csv = camera_csv_info

    report.findings = findings
    report.media_status = compute_media_status(report)
    report.tooling_status = compute_tooling_status(report)
    report.overall_status = compute_overall_status(report)
    report.remediation = build_remediation(report)
    return report
