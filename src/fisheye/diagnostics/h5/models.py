from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional

CheckStatus = Literal["pass", "warn", "fail", "error", "skip"]
H5SourceKind = Literal["raw", "other"]
FindingKind = Literal["core", "optional", "tooling"]


def _default_status_counts() -> dict[str, int]:
    return {"pass": 0, "warn": 0, "fail": 0, "error": 0, "skip": 0}


def _default_source_counts() -> dict[str, int]:
    return {"raw": 0, "other": 0}


@dataclass
class FileInfo:
    input_path: str
    path: str
    exists: bool
    source_kind: Optional[H5SourceKind] = None
    recording_root: Optional[str] = None


@dataclass
class CoreInfo:
    status: CheckStatus = "skip"
    profile: str = "palette-import"
    h5_opened: bool = False
    root_keys: list[str] = field(default_factory=list)
    events_present: Optional[bool] = None
    frame_metadata_present: Optional[bool] = None
    events_rows: Optional[int] = None
    frame_metadata_rows: Optional[int] = None
    missing_event_fields: list[str] = field(default_factory=list)
    missing_frame_metadata_fields: list[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class EventsInfo:
    status: CheckStatus = "skip"
    rows: int = 0
    fields: list[str] = field(default_factory=list)
    timestamp_monotonic: Optional[bool] = None
    has_camera_frame_id: Optional[bool] = None
    event_type_counts: dict[str, int] = field(default_factory=dict)
    details_json_nonempty: int = 0
    details_json_parse_failures: int = 0
    error: Optional[str] = None


@dataclass
class FrameMetadataInfo:
    status: CheckStatus = "skip"
    rows: int = 0
    fields: list[str] = field(default_factory=list)
    stimulus_monotonic: Optional[bool] = None
    camera_nondecreasing: Optional[bool] = None
    unique_camera_frames: int = 0
    camera_min: Optional[int] = None
    camera_max: Optional[int] = None
    missing_camera_frames: int = 0
    mean_stimulus_per_camera: Optional[float] = None
    median_stimulus_per_camera: Optional[float] = None
    expected_ratio: float = 2.0
    ratio_warn_count: int = 0
    ratio_warn_fraction: Optional[float] = None
    max_ratio_warn_run_length: int = 0
    max_abs_cumulative_drift: Optional[float] = None
    error: Optional[str] = None


@dataclass
class DatasetSummary:
    name: str
    status: CheckStatus = "skip"
    present: bool = False
    rows: int = 0
    fields: list[str] = field(default_factory=list)
    missing_fields: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class TrackingInfo:
    status: CheckStatus = "skip"
    tracking_group_present: Optional[bool] = None
    datasets: dict[str, DatasetSummary] = field(default_factory=dict)
    chaser_position_varies: Optional[bool] = None
    error: Optional[str] = None


@dataclass
class SnapshotsInfo:
    status: CheckStatus = "skip"
    protocol_snapshot_present: Optional[bool] = None
    protocol_json_parseable: Optional[bool] = None
    calibration_snapshot_present: Optional[bool] = None
    calibration_json_parseable: Optional[bool] = None
    recording_snapshot_present: Optional[bool] = None
    recording_json_parseable: Optional[bool] = None
    subject_metadata_present: Optional[bool] = None
    session_metadata_present: Optional[bool] = None
    stimulus_coordinates_present: Optional[bool] = None
    subject_metadata_keys: list[str] = field(default_factory=list)
    session_metadata_keys: list[str] = field(default_factory=list)
    stimulus_coordinate_arenas: list[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class EnumsInfo:
    status: CheckStatus = "skip"
    dataset_counts: dict[str, int] = field(default_factory=dict)
    malformed_datasets: list[str] = field(default_factory=list)
    missing_expected: list[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class Finding:
    severity: Literal["warn", "fail", "error"]
    code: str
    summary: str
    details: Optional[str] = None
    component: Optional[str] = None
    kind: FindingKind = "core"


@dataclass
class H5DiagnosticsReport:
    overall_status: CheckStatus
    file_info: FileInfo
    core_status: CheckStatus = "skip"
    optional_status: CheckStatus = "skip"
    tooling_status: CheckStatus = "skip"
    core: CoreInfo = field(default_factory=CoreInfo)
    events: EventsInfo = field(default_factory=EventsInfo)
    frame_metadata: FrameMetadataInfo = field(default_factory=FrameMetadataInfo)
    tracking: TrackingInfo = field(default_factory=TrackingInfo)
    snapshots: SnapshotsInfo = field(default_factory=SnapshotsInfo)
    enums: EnumsInfo = field(default_factory=EnumsInfo)
    findings: list[Finding] = field(default_factory=list)


@dataclass
class BatchSummary:
    scanned: int = 0
    passed: int = 0
    warned: int = 0
    failed: int = 0
    errored: int = 0
    skipped: int = 0
    source_counts: dict[str, int] = field(default_factory=_default_source_counts)
    optional_counts: dict[str, int] = field(default_factory=_default_status_counts)
    tooling_counts: dict[str, int] = field(default_factory=_default_status_counts)
    recording_counts: dict[str, int] = field(default_factory=_default_status_counts)
    optional_recording_counts: dict[str, int] = field(default_factory=_default_status_counts)
    tooling_recording_counts: dict[str, int] = field(default_factory=_default_status_counts)


@dataclass
class RecordingDiagnosticsSummary:
    recording_root: str
    overall_status: CheckStatus
    core_status: CheckStatus = "skip"
    optional_status: CheckStatus = "skip"
    tooling_status: CheckStatus = "skip"
    item_count: int = 0
    source_counts: dict[str, int] = field(default_factory=_default_source_counts)


@dataclass
class BatchDiagnosticsReport:
    overall_status: CheckStatus
    roots: list[str]
    recursive: bool
    items: list[H5DiagnosticsReport] = field(default_factory=list)
    summary: BatchSummary = field(default_factory=BatchSummary)
    recordings: list[RecordingDiagnosticsSummary] = field(default_factory=list)


def report_to_dict(report: H5DiagnosticsReport | BatchDiagnosticsReport) -> dict[str, Any]:
    return asdict(report)


def resolved_core_status(report: H5DiagnosticsReport | RecordingDiagnosticsSummary) -> CheckStatus:
    if hasattr(report, "core_status"):
        return getattr(report, "core_status")
    return getattr(report, "overall_status")


def resolved_optional_status(report: H5DiagnosticsReport | RecordingDiagnosticsSummary) -> CheckStatus:
    return getattr(report, "optional_status", "skip")


def resolved_tooling_status(report: H5DiagnosticsReport | RecordingDiagnosticsSummary) -> CheckStatus:
    return getattr(report, "tooling_status", "skip")


def _section_statuses(report: H5DiagnosticsReport, *, optional: bool) -> list[CheckStatus]:
    if optional:
        return [report.tracking.status, report.snapshots.status, report.enums.status]
    return [report.core.status, report.events.status, report.frame_metadata.status]


def compute_core_status(report: H5DiagnosticsReport) -> CheckStatus:
    statuses = set(_section_statuses(report, optional=False))
    if "error" in statuses:
        return "error"
    if "fail" in statuses:
        return "fail"
    if "warn" in statuses:
        return "warn"
    core_findings = [finding for finding in report.findings if finding.kind == "core"]
    severities = {finding.severity for finding in core_findings}
    if "fail" in severities:
        return "fail"
    if "warn" in severities or "error" in severities:
        return "warn"
    if "pass" in statuses:
        return "pass"
    return "skip"


def compute_optional_status(report: H5DiagnosticsReport) -> CheckStatus:
    statuses = set(_section_statuses(report, optional=True))
    if "error" in statuses:
        return "error"
    if "fail" in statuses:
        return "fail"
    if "warn" in statuses:
        return "warn"
    optional_findings = [finding for finding in report.findings if finding.kind == "optional"]
    severities = {finding.severity for finding in optional_findings}
    if "fail" in severities:
        return "fail"
    if "warn" in severities or "error" in severities:
        return "warn"
    if "pass" in statuses:
        return "pass"
    return "skip"


def compute_tooling_status(report: H5DiagnosticsReport) -> CheckStatus:
    tooling_findings = [finding for finding in report.findings if finding.kind == "tooling"]
    severities = {finding.severity for finding in tooling_findings}
    if "fail" in severities:
        return "fail"
    if "error" in severities:
        return "error"
    if "warn" in severities:
        return "warn"
    if report.core.status != "skip":
        return "pass"
    return "skip"


def compute_overall_status(report: H5DiagnosticsReport) -> CheckStatus:
    if report.core_status != "skip":
        return report.core_status
    return "skip"
