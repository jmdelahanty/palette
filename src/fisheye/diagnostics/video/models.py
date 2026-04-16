from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional

CheckStatus = Literal["pass", "warn", "fail", "error", "skip"]
VideoSourceKind = Literal["cams", "raw", "other"]
FindingKind = Literal["media", "tooling"]


def _default_source_counts() -> dict[str, int]:
    return {"cams": 0, "raw": 0, "other": 0}


def _default_status_counts() -> dict[str, int]:
    return {"pass": 0, "warn": 0, "fail": 0, "error": 0, "skip": 0}


@dataclass
class FileInfo:
    path: str
    exists: bool
    size_bytes: Optional[int] = None
    modified_time: Optional[float] = None
    source_kind: Optional[VideoSourceKind] = None
    recording_root: Optional[str] = None


@dataclass
class StreamInfo:
    status: CheckStatus = "skip"
    container_format: Optional[str] = None
    codec: Optional[str] = None
    profile: Optional[str] = None
    level: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    pix_fmt: Optional[str] = None
    avg_fps: Optional[float] = None
    duration_seconds: Optional[float] = None
    nb_frames: Optional[int] = None
    has_b_frames: Optional[bool] = None
    format_tags: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class TimingGap:
    position: int
    time_seconds: float
    gap_duration_seconds: float
    estimated_missing_frames: int


@dataclass
class TimingInfo:
    status: CheckStatus = "skip"
    scope: str = "skip"
    frames_analyzed: int = 0
    timing_basis: Optional[str] = None
    pts_present: bool = False
    dts_present: bool = False
    pts_monotonic: Optional[bool] = None
    dts_monotonic: Optional[bool] = None
    median_interval_ms: Optional[float] = None
    mean_interval_ms: Optional[float] = None
    std_interval_ms: Optional[float] = None
    max_gap_ms: Optional[float] = None
    estimated_missing_frames: int = 0
    gap_count: int = 0
    gaps: list[TimingGap] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class GOPInfo:
    status: CheckStatus = "skip"
    scope: str = "skip"
    frames_analyzed: int = 0
    keyframe_count: int = 0
    avg_keyframe_interval_s: Optional[float] = None
    min_keyframe_interval_s: Optional[float] = None
    max_keyframe_interval_s: Optional[float] = None
    b_frames_present: Optional[bool] = None
    b_frame_count: int = 0
    max_gop_frames: Optional[int] = None
    frame_type_counts: dict[str, int] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class CameraCsvInfo:
    status: CheckStatus = "skip"
    path: Optional[str] = None
    exists: Optional[bool] = None
    rows: int = 0
    schema_ok: Optional[bool] = None
    missing_columns: list[str] = field(default_factory=list)
    frame_id_first: Optional[int] = None
    frame_id_last: Optional[int] = None
    frame_id_monotonic: Optional[bool] = None
    frame_id_contiguous: Optional[bool] = None
    timestamp_monotonic: Optional[bool] = None
    timestamp_sys_monotonic: Optional[bool] = None
    median_timestamp_step_ns: Optional[int] = None
    median_timestamp_sys_step_ns: Optional[int] = None
    video_frame_count: Optional[int] = None
    row_count_matches_video: Optional[bool] = None
    timestamp_offset_first_ns: Optional[int] = None
    timestamp_offset_last_ns: Optional[int] = None
    timestamp_offset_drift_ns: Optional[int] = None
    error: Optional[str] = None


@dataclass
class SeekInaccuracy:
    requested_frame: int
    observed_frame: int


@dataclass
class BackendDecodeReport:
    backend: str
    status: CheckStatus = "skip"
    available: Optional[bool] = None
    open_ok: Optional[bool] = None
    total_frames: Optional[int] = None
    frames_checked: int = 0
    sequential_failed_frames: list[int] = field(default_factory=list)
    seek_positions_checked: int = 0
    seek_failures: list[int] = field(default_factory=list)
    seek_inaccuracies: list[SeekInaccuracy] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class Finding:
    severity: Literal["warn", "fail", "error"]
    code: str
    summary: str
    details: Optional[str] = None
    component: Optional[str] = None
    kind: FindingKind = "media"


@dataclass
class Remediation:
    issue: str
    description: str
    command: Optional[str] = None


@dataclass
class VideoDiagnosticsReport:
    overall_status: CheckStatus
    file_info: FileInfo
    media_status: CheckStatus = "skip"
    tooling_status: CheckStatus = "skip"
    stream_info: StreamInfo = field(default_factory=StreamInfo)
    timing: TimingInfo = field(default_factory=TimingInfo)
    gop: GOPInfo = field(default_factory=GOPInfo)
    camera_csv: CameraCsvInfo = field(default_factory=CameraCsvInfo)
    decode: list[BackendDecodeReport] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    remediation: list[Remediation] = field(default_factory=list)


@dataclass
class BatchSummary:
    scanned: int = 0
    passed: int = 0
    warned: int = 0
    failed: int = 0
    errored: int = 0
    skipped: int = 0
    source_counts: dict[str, int] = field(default_factory=_default_source_counts)
    recording_counts: dict[str, int] = field(default_factory=_default_status_counts)
    tooling_counts: dict[str, int] = field(default_factory=_default_status_counts)
    tooling_recording_counts: dict[str, int] = field(default_factory=_default_status_counts)


@dataclass
class RecordingDiagnosticsSummary:
    recording_root: str
    overall_status: CheckStatus
    media_status: CheckStatus = "skip"
    tooling_status: CheckStatus = "skip"
    item_count: int = 0
    source_counts: dict[str, int] = field(default_factory=_default_source_counts)


@dataclass
class BatchDiagnosticsReport:
    overall_status: CheckStatus
    roots: list[str]
    recursive: bool
    items: list[VideoDiagnosticsReport] = field(default_factory=list)
    summary: BatchSummary = field(default_factory=BatchSummary)
    recordings: list[RecordingDiagnosticsSummary] = field(default_factory=list)


def report_to_dict(report: VideoDiagnosticsReport) -> dict[str, Any]:
    return asdict(report)


def _has_media_checks(report: VideoDiagnosticsReport) -> bool:
    if any(finding.kind == "media" for finding in report.findings):
        return True
    statuses = [report.stream_info.status, report.timing.status, report.gop.status, report.camera_csv.status]
    if any(status in {"pass", "warn", "fail"} for status in statuses):
        return True
    return any(item.status in {"pass", "warn", "fail"} for item in report.decode)


def _has_tooling_checks(report: VideoDiagnosticsReport) -> bool:
    statuses = [report.stream_info.status, report.timing.status, report.gop.status]
    return any(status != "skip" for status in statuses) or bool(report.decode)


def resolved_media_status(report: VideoDiagnosticsReport) -> CheckStatus:
    if report.media_status != "skip" or report.overall_status == "skip":
        return report.media_status
    return report.overall_status


def resolved_tooling_status(report: VideoDiagnosticsReport) -> CheckStatus:
    if report.tooling_status != "skip":
        return report.tooling_status
    if report.overall_status != "skip":
        return "pass"
    return "skip"


def compute_media_status(report: VideoDiagnosticsReport) -> CheckStatus:
    media_findings = [finding for finding in report.findings if finding.kind == "media"]
    severities = {finding.severity for finding in media_findings}
    if "fail" in severities:
        return "fail"
    if "warn" in severities or "error" in severities:
        return "warn"
    if _has_media_checks(report):
        return "pass"
    return "skip"


def compute_tooling_status(report: VideoDiagnosticsReport) -> CheckStatus:
    tooling_findings = [finding for finding in report.findings if finding.kind == "tooling"]
    severities = {finding.severity for finding in tooling_findings}
    if "fail" in severities:
        return "fail"
    if "error" in severities:
        return "error"
    if "warn" in severities:
        return "warn"
    if _has_tooling_checks(report):
        return "pass"
    return "skip"


def compute_overall_status(report: VideoDiagnosticsReport) -> CheckStatus:
    media_status = report.media_status if report.media_status != "skip" or report.overall_status == "skip" else report.overall_status
    if media_status != "skip":
        return media_status
    return "skip"
