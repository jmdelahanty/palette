from __future__ import annotations

import json
from pathlib import Path

from .models import (
    BatchDiagnosticsReport,
    RecordingDiagnosticsSummary,
    VideoDiagnosticsReport,
    report_to_dict,
    resolved_media_status,
    resolved_tooling_status,
)


def _group_items_by_recording(items: list[VideoDiagnosticsReport]) -> dict[str, list[VideoDiagnosticsReport]]:
    grouped: dict[str, list[VideoDiagnosticsReport]] = {}
    for item in items:
        recording_root = item.file_info.recording_root or str(Path(item.file_info.path).parent)
        grouped.setdefault(recording_root, []).append(item)
    return grouped


def _display_batch_item_path(item: VideoDiagnosticsReport, recording_root: str) -> str:
    file_path = Path(item.file_info.path)
    try:
        return str(file_path.relative_to(Path(recording_root)))
    except ValueError:
        return item.file_info.path


def render_report(report: VideoDiagnosticsReport, *, as_json: bool = False) -> str:
    if as_json:
        return json.dumps(report_to_dict(report), indent=2, sort_keys=True)

    lines: list[str] = [
        f"Overall: {report.overall_status}",
        f"Media: {resolved_media_status(report)}",
        f"Tooling: {resolved_tooling_status(report)}",
        f"File: {report.file_info.path}",
    ]
    if report.file_info.source_kind:
        lines.append(f"Source: {report.file_info.source_kind}")
    if report.file_info.recording_root:
        lines.append(f"Recording: {report.file_info.recording_root}")

    stream = report.stream_info
    if stream.status != "skip":
        lines.extend(
            [
                "",
                f"Stream [{stream.status}]",
                f"  codec: {stream.codec or '—'}",
                f"  resolution: {stream.width or '—'}x{stream.height or '—'}",
                f"  fps: {stream.avg_fps:.3f}" if stream.avg_fps is not None else "  fps: —",
                f"  duration_s: {stream.duration_seconds:.3f}" if stream.duration_seconds is not None else "  duration_s: —",
                f"  pix_fmt: {stream.pix_fmt or '—'}",
            ]
        )
        if stream.error:
            lines.append(f"  error: {stream.error}")

    timing = report.timing
    if timing.status != "skip":
        lines.extend(
            [
                "",
                f"Timing [{timing.status}]",
                f"  scope: {timing.scope}",
                f"  frames_analyzed: {timing.frames_analyzed}",
                f"  basis: {timing.timing_basis or '—'}",
                f"  pts_monotonic: {timing.pts_monotonic if timing.pts_monotonic is not None else '—'}",
                f"  dts_monotonic: {timing.dts_monotonic if timing.dts_monotonic is not None else '—'}",
                f"  gap_count: {timing.gap_count}",
                f"  estimated_missing_frames: {timing.estimated_missing_frames}",
            ]
        )
        if timing.error:
            lines.append(f"  error: {timing.error}")

    gop = report.gop
    if gop.status != "skip":
        lines.extend(
            [
                "",
                f"GOP [{gop.status}]",
                f"  scope: {gop.scope}",
                f"  frames_analyzed: {gop.frames_analyzed}",
                f"  keyframes: {gop.keyframe_count}",
                f"  max_gop_frames: {gop.max_gop_frames if gop.max_gop_frames is not None else '—'}",
                f"  b_frames_present: {gop.b_frames_present if gop.b_frames_present is not None else '—'}",
            ]
        )
        if gop.max_keyframe_interval_s is not None:
            lines.append(f"  max_keyframe_interval_s: {gop.max_keyframe_interval_s:.3f}")
        if gop.error:
            lines.append(f"  error: {gop.error}")

    camera_csv = report.camera_csv
    if camera_csv.status != "skip":
        lines.extend(
            [
                "",
                f"Camera CSV [{camera_csv.status}]",
                f"  path: {camera_csv.path or '—'}",
                f"  rows: {camera_csv.rows}",
                f"  frame_ids: "
                f"{camera_csv.frame_id_first if camera_csv.frame_id_first is not None else '—'}"
                f"..{camera_csv.frame_id_last if camera_csv.frame_id_last is not None else '—'}",
                f"  row_count_matches_video: "
                f"{camera_csv.row_count_matches_video if camera_csv.row_count_matches_video is not None else '—'}",
                f"  frame_id_monotonic: "
                f"{camera_csv.frame_id_monotonic if camera_csv.frame_id_monotonic is not None else '—'}",
                f"  frame_id_contiguous: "
                f"{camera_csv.frame_id_contiguous if camera_csv.frame_id_contiguous is not None else '—'}",
                f"  timestamp_monotonic: "
                f"{camera_csv.timestamp_monotonic if camera_csv.timestamp_monotonic is not None else '—'}",
                f"  timestamp_sys_monotonic: "
                f"{camera_csv.timestamp_sys_monotonic if camera_csv.timestamp_sys_monotonic is not None else '—'}",
            ]
        )
        if camera_csv.median_timestamp_step_ns is not None:
            lines.append(f"  median_timestamp_step_ns: {camera_csv.median_timestamp_step_ns}")
        if camera_csv.median_timestamp_sys_step_ns is not None:
            lines.append(f"  median_timestamp_sys_step_ns: {camera_csv.median_timestamp_sys_step_ns}")
        if camera_csv.timestamp_offset_drift_ns is not None:
            lines.append(f"  timestamp_offset_drift_ns: {camera_csv.timestamp_offset_drift_ns}")
        if camera_csv.missing_columns:
            lines.append(f"  missing_columns: {', '.join(camera_csv.missing_columns)}")
        if camera_csv.error:
            lines.append(f"  error: {camera_csv.error}")

    if report.decode:
        lines.append("")
        lines.append("Decode")
        for item in report.decode:
            lines.append(f"  {item.backend} [{item.status}]")
            if item.error:
                lines.append(f"    error: {item.error}")
                if item.status == "error" and item.available is False:
                    lines.append("    note: backend/tooling issue; this does not by itself indicate broken media")
                continue
            lines.append(f"    available: {item.available}")
            lines.append(f"    open_ok: {item.open_ok}")
            lines.append(f"    total_frames: {item.total_frames if item.total_frames is not None else '—'}")
            lines.append(f"    frames_checked: {item.frames_checked}")
            lines.append(f"    sequential_failures: {len(item.sequential_failed_frames)}")
            lines.append(f"    seek_failures: {len(item.seek_failures)}")
            lines.append(f"    seek_inaccuracies: {len(item.seek_inaccuracies)}")

    if report.findings:
        lines.append("")
        lines.append("Findings")
        for finding in report.findings:
            lines.append(f"  [{finding.severity}] {finding.code}: {finding.summary}")
            if finding.details:
                lines.append(f"    {finding.details}")

    if report.remediation:
        lines.append("")
        lines.append("Remediation")
        for item in report.remediation:
            lines.append(f"  {item.issue}: {item.description}")
            if item.command:
                lines.append(f"    {item.command}")

    return "\n".join(lines)


def render_batch_report(report: BatchDiagnosticsReport, *, as_json: bool = False) -> str:
    if as_json:
        return json.dumps(report_to_dict(report), indent=2, sort_keys=True)

    summary = report.summary
    lines: list[str] = [
        f"Overall: {report.overall_status}",
        f"Roots: {', '.join(report.roots)}",
        f"Recursive: {report.recursive}",
        "",
        "Summary",
        f"  scanned: {summary.scanned}",
        "  media_files: "
        + ", ".join(
            (
                f"pass={summary.passed}",
                f"warn={summary.warned}",
                f"fail={summary.failed}",
                f"error={summary.errored}",
                f"skip={summary.skipped}",
            )
        ),
        "  tooling_files: "
        + ", ".join(
            f"{name}={summary.tooling_counts.get(name, 0)}"
            for name in ("pass", "warn", "fail", "error", "skip")
        ),
        "  sources: "
        + ", ".join(
            f"{name}={summary.source_counts.get(name, 0)}"
            for name in ("cams", "raw", "other")
        ),
        "  media_recordings: "
        + ", ".join(
            f"{name}={summary.recording_counts.get(name, 0)}"
            for name in ("pass", "warn", "fail", "error", "skip")
        ),
        "  tooling_recordings: "
        + ", ".join(
            f"{name}={summary.tooling_recording_counts.get(name, 0)}"
            for name in ("pass", "warn", "fail", "error", "skip")
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
            media_status = resolved_media_status(recording)
            tooling_status = resolved_tooling_status(recording)
            source_summary = ", ".join(
                f"{name}={recording.source_counts.get(name, 0)}"
                for name in ("cams", "raw", "other")
            )
            lines.append(
                f"  Recording [media={media_status}; tooling={tooling_status}]: {recording.recording_root}"
                f" (items={recording.item_count}; sources: {source_summary})"
            )
            items = grouped_items.get(recording.recording_root, [])
            for item in items:
                finding_codes = ", ".join(f.code for f in item.findings[:3])
                suffix = f" ({finding_codes})" if finding_codes else ""
                source_kind = item.file_info.source_kind or "other"
                display_path = _display_batch_item_path(item, recording.recording_root)
                lines.append(
                    f"    [media={resolved_media_status(item)}; tooling={resolved_tooling_status(item)}] "
                    f"[{source_kind}] {display_path}{suffix}"
                )

    return "\n".join(lines)


def render_batch_jsonl(report: BatchDiagnosticsReport) -> str:
    lines = [json.dumps(report_to_dict(item), sort_keys=True) for item in report.items]
    if not lines:
        return ""
    return "\n".join(lines) + "\n"
