from __future__ import annotations

from pathlib import Path

import h5py

from .core import inspect_core
from .enums import inspect_enums
from .events import inspect_events
from .frame_metadata import inspect_frame_metadata
from .models import (
    CoreInfo,
    Finding,
    H5DiagnosticsReport,
    compute_core_status,
    compute_optional_status,
    compute_overall_status,
    compute_tooling_status,
)
from .reader import build_file_info
from .snapshots import inspect_snapshots
from .tracking import inspect_tracking

__all__ = ["build_h5_report"]


def _open_error_report(path: Path | str, *, profile: str, message: str) -> H5DiagnosticsReport:
    report = H5DiagnosticsReport(overall_status="error", file_info=build_file_info(path))
    report.core = CoreInfo(status="error", profile=profile, error=message)
    report.findings = [
        Finding(
            severity="error",
            code="h5.open_error",
            summary="Could not open the H5 file.",
            details=message,
            component="core",
            kind="tooling",
        )
    ]
    report.core_status = compute_core_status(report)
    report.optional_status = compute_optional_status(report)
    report.tooling_status = compute_tooling_status(report)
    report.overall_status = compute_overall_status(report)
    return report


def build_h5_report(
    path: Path | str,
    *,
    profile: str = "palette-import",
    include_core: bool = True,
    include_events: bool = True,
    include_frame_metadata: bool = True,
    include_tracking: bool = True,
    include_snapshots: bool = True,
    include_enums: bool = True,
) -> H5DiagnosticsReport:
    report = H5DiagnosticsReport(overall_status="skip", file_info=build_file_info(path))
    if not report.file_info.exists:
        report.core = CoreInfo(status="fail", profile=profile, error="No H5 file found for input path")
        report.findings = [
            Finding(
                severity="fail",
                code="h5.file_missing",
                summary="No H5 file could be resolved from the input path.",
                component="core",
                kind="core",
            )
        ]
        report.core_status = compute_core_status(report)
        report.optional_status = compute_optional_status(report)
        report.tooling_status = compute_tooling_status(report)
        report.overall_status = compute_overall_status(report)
        return report

    try:
        with h5py.File(report.file_info.path, "r") as handle:
            findings: list[Finding] = []
            if include_core:
                report.core, section_findings = inspect_core(handle, profile=profile)
                findings.extend(section_findings)
            if include_events:
                report.events, section_findings = inspect_events(handle)
                findings.extend(section_findings)
            if include_frame_metadata:
                report.frame_metadata, section_findings = inspect_frame_metadata(
                    handle,
                    required=(profile == "palette-import"),
                )
                findings.extend(section_findings)
            if include_tracking:
                report.tracking, section_findings = inspect_tracking(handle)
                findings.extend(section_findings)
            if include_snapshots:
                report.snapshots, section_findings = inspect_snapshots(handle)
                findings.extend(section_findings)
            if include_enums:
                report.enums, section_findings = inspect_enums(handle)
                findings.extend(section_findings)
            report.findings = findings
    except Exception as exc:
        return _open_error_report(path, profile=profile, message=str(exc))

    report.core_status = compute_core_status(report)
    report.optional_status = compute_optional_status(report)
    report.tooling_status = compute_tooling_status(report)
    report.overall_status = compute_overall_status(report)
    return report
