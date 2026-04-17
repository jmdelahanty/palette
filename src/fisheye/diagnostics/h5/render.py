from __future__ import annotations

import json
from pathlib import Path

from .models import (
    BatchDiagnosticsReport,
    H5DiagnosticsReport,
    RecordingDiagnosticsSummary,
    report_to_dict,
    resolved_core_status,
    resolved_optional_status,
    resolved_tooling_status,
)


def _group_items_by_recording(items: list[H5DiagnosticsReport]) -> dict[str, list[H5DiagnosticsReport]]:
    grouped: dict[str, list[H5DiagnosticsReport]] = {}
    for item in items:
        recording_root = item.file_info.recording_root or str(Path(item.file_info.path).parent)
        grouped.setdefault(recording_root, []).append(item)
    return grouped


def _display_batch_item_path(item: H5DiagnosticsReport, recording_root: str) -> str:
    file_path = Path(item.file_info.path)
    try:
        return str(file_path.relative_to(Path(recording_root)))
    except ValueError:
        return item.file_info.path


def render_report(report: H5DiagnosticsReport, *, as_json: bool = False) -> str:
    if as_json:
        return json.dumps(report_to_dict(report), indent=2, sort_keys=True)

    lines = [
        f"Overall: {report.overall_status}",
        f"Core: {resolved_core_status(report)}",
        f"Optional: {resolved_optional_status(report)}",
        f"Tooling: {resolved_tooling_status(report)}",
        f"File: {report.file_info.path}",
    ]
    if report.file_info.source_kind:
        lines.append(f"Source: {report.file_info.source_kind}")
    if report.file_info.recording_root:
        lines.append(f"Recording: {report.file_info.recording_root}")

    if report.core.status != "skip":
        lines.extend(
            [
                "",
                f"Core [{report.core.status}]",
                f"  profile: {report.core.profile}",
                f"  events_present: {report.core.events_present}",
                f"  frame_metadata_present: {report.core.frame_metadata_present}",
                f"  events_rows: {report.core.events_rows if report.core.events_rows is not None else '—'}",
                f"  frame_metadata_rows: {report.core.frame_metadata_rows if report.core.frame_metadata_rows is not None else '—'}",
            ]
        )

    if report.events.status != "skip":
        lines.extend(
            [
                "",
                f"Events [{report.events.status}]",
                f"  rows: {report.events.rows}",
                f"  timestamp_monotonic: {report.events.timestamp_monotonic if report.events.timestamp_monotonic is not None else '—'}",
                f"  details_json_parse_failures: {report.events.details_json_parse_failures}",
            ]
        )

    if report.frame_metadata.status != "skip":
        lines.extend(
            [
                "",
                f"Frame Metadata [{report.frame_metadata.status}]",
                f"  rows: {report.frame_metadata.rows}",
                f"  stimulus_monotonic: {report.frame_metadata.stimulus_monotonic if report.frame_metadata.stimulus_monotonic is not None else '—'}",
                f"  camera_nondecreasing: {report.frame_metadata.camera_nondecreasing if report.frame_metadata.camera_nondecreasing is not None else '—'}",
                f"  missing_camera_frames: {report.frame_metadata.missing_camera_frames}",
                f"  ratio_warn_count: {report.frame_metadata.ratio_warn_count}",
                f"  ratio_warn_fraction: {report.frame_metadata.ratio_warn_fraction:.6f}" if report.frame_metadata.ratio_warn_fraction is not None else "  ratio_warn_fraction: —",
                f"  max_ratio_warn_run_length: {report.frame_metadata.max_ratio_warn_run_length}",
                f"  max_abs_cumulative_drift: {report.frame_metadata.max_abs_cumulative_drift:.3f}" if report.frame_metadata.max_abs_cumulative_drift is not None else "  max_abs_cumulative_drift: —",
            ]
        )

    if report.tracking.status != "skip":
        lines.extend(["", f"Tracking [{report.tracking.status}]", f"  tracking_group_present: {report.tracking.tracking_group_present}"])
        for name, dataset in sorted(report.tracking.datasets.items()):
            lines.append(f"  {name}: status={dataset.status} present={dataset.present} rows={dataset.rows}")

    if report.snapshots.status != "skip":
        lines.extend(["", f"Snapshots [{report.snapshots.status}]", f"  protocol_snapshot_present: {report.snapshots.protocol_snapshot_present}"])

    if report.enums.status != "skip":
        dataset_summary = ", ".join(f"{name}={count}" for name, count in sorted(report.enums.dataset_counts.items())) or "—"
        lines.extend(["", f"Enums [{report.enums.status}]", f"  datasets: {dataset_summary}"])

    if report.findings:
        lines.append("")
        lines.append("Findings")
        for finding in report.findings:
            lines.append(f"  [{finding.severity}] {finding.code}: {finding.summary}")
            if finding.details:
                lines.append(f"    {finding.details}")

    return "\n".join(lines)


def render_batch_report(report: BatchDiagnosticsReport, *, as_json: bool = False) -> str:
    if as_json:
        return json.dumps(report_to_dict(report), indent=2, sort_keys=True)

    summary = report.summary
    lines = [
        f"Overall: {report.overall_status}",
        f"Roots: {', '.join(report.roots)}",
        f"Recursive: {report.recursive}",
        "",
        "Summary",
        f"  scanned: {summary.scanned}",
        "  core_files: " + ", ".join(
            (
                f"pass={summary.passed}",
                f"warn={summary.warned}",
                f"fail={summary.failed}",
                f"error={summary.errored}",
                f"skip={summary.skipped}",
            )
        ),
        "  optional_files: " + ", ".join(
            f"{name}={summary.optional_counts.get(name, 0)}" for name in ("pass", "warn", "fail", "error", "skip")
        ),
        "  tooling_files: " + ", ".join(
            f"{name}={summary.tooling_counts.get(name, 0)}" for name in ("pass", "warn", "fail", "error", "skip")
        ),
    ]

    if report.items:
        lines.append("")
        lines.append("Items")
        grouped_items = _group_items_by_recording(report.items)
        recordings = report.recordings or [
            RecordingDiagnosticsSummary(recording_root=recording_root, overall_status="skip", item_count=len(items))
            for recording_root, items in grouped_items.items()
        ]
        for recording in recordings:
            lines.append(
                f"  Recording [core={resolved_core_status(recording)}; optional={resolved_optional_status(recording)}; tooling={resolved_tooling_status(recording)}]: {recording.recording_root}"
            )
            for item in grouped_items.get(recording.recording_root, []):
                finding_codes = ", ".join(f.code for f in item.findings[:3])
                suffix = f" ({finding_codes})" if finding_codes else ""
                source_kind = item.file_info.source_kind or "other"
                display_path = _display_batch_item_path(item, recording.recording_root)
                lines.append(
                    f"    [core={resolved_core_status(item)}; optional={resolved_optional_status(item)}; tooling={resolved_tooling_status(item)}] "
                    f"[{source_kind}] {display_path}{suffix}"
                )

    return "\n".join(lines)


def render_batch_jsonl(report: BatchDiagnosticsReport) -> str:
    lines = [json.dumps(report_to_dict(item), sort_keys=True) for item in report.items]
    if not lines:
        return ""
    return "\n".join(lines) + "\n"
