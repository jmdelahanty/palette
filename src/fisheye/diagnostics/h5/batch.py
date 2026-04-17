from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from . import build_h5_report
from .models import (
    BatchDiagnosticsReport,
    BatchSummary,
    H5DiagnosticsReport,
    RecordingDiagnosticsSummary,
    resolved_core_status,
    resolved_optional_status,
    resolved_tooling_status,
)
from .reader import iter_h5_paths


def _summarize(items: list[H5DiagnosticsReport]) -> BatchSummary:
    summary = BatchSummary(scanned=len(items))
    for item in items:
        core_status = resolved_core_status(item)
        optional_status = resolved_optional_status(item)
        tooling_status = resolved_tooling_status(item)
        source_kind = item.file_info.source_kind or "other"
        summary.source_counts[source_kind] = summary.source_counts.get(source_kind, 0) + 1
        if core_status == "pass":
            summary.passed += 1
        elif core_status == "warn":
            summary.warned += 1
        elif core_status == "fail":
            summary.failed += 1
        elif core_status == "error":
            summary.errored += 1
        else:
            summary.skipped += 1
        summary.optional_counts[optional_status] = summary.optional_counts.get(optional_status, 0) + 1
        summary.tooling_counts[tooling_status] = summary.tooling_counts.get(tooling_status, 0) + 1
    return summary


def _compute_status(statuses: set[str]) -> str:
    if "fail" in statuses:
        return "fail"
    if "error" in statuses:
        return "error"
    if "warn" in statuses:
        return "warn"
    if "pass" in statuses:
        return "pass"
    return "skip"


def _summarize_recordings(items: list[H5DiagnosticsReport]) -> list[RecordingDiagnosticsSummary]:
    grouped: dict[str, list[H5DiagnosticsReport]] = {}
    for item in items:
        recording_root = item.file_info.recording_root or str(Path(item.file_info.path).parent)
        grouped.setdefault(recording_root, []).append(item)

    summaries: list[RecordingDiagnosticsSummary] = []
    for recording_root in sorted(grouped):
        recording_items = grouped[recording_root]
        source_counts = {"raw": 0, "other": 0}
        for item in recording_items:
            source_kind = item.file_info.source_kind or "other"
            source_counts[source_kind] = source_counts.get(source_kind, 0) + 1
        core_status = _compute_status({resolved_core_status(item) for item in recording_items})
        optional_status = _compute_status({resolved_optional_status(item) for item in recording_items})
        tooling_status = _compute_status({resolved_tooling_status(item) for item in recording_items})
        summaries.append(
            RecordingDiagnosticsSummary(
                recording_root=recording_root,
                overall_status=core_status,
                core_status=core_status,
                optional_status=optional_status,
                tooling_status=tooling_status,
                item_count=len(recording_items),
                source_counts=source_counts,
            )
        )
    return summaries


def _add_recording_counts(summary: BatchSummary, recordings: list[RecordingDiagnosticsSummary]) -> None:
    for recording in recordings:
        summary.recording_counts[recording.core_status] = summary.recording_counts.get(recording.core_status, 0) + 1
        summary.optional_recording_counts[recording.optional_status] = (
            summary.optional_recording_counts.get(recording.optional_status, 0) + 1
        )
        summary.tooling_recording_counts[recording.tooling_status] = (
            summary.tooling_recording_counts.get(recording.tooling_status, 0) + 1
        )


def build_batch_report(
    paths: Iterable[Path],
    *,
    recursive: bool,
    limit: Optional[int] = None,
    **kwargs: object,
) -> BatchDiagnosticsReport:
    selected: list[Path] = []
    for h5_path in iter_h5_paths(paths, recursive=recursive):
        selected.append(h5_path)
        if limit is not None and limit > 0 and len(selected) >= limit:
            break

    items = [build_h5_report(path, **kwargs) for path in selected]
    recordings = _summarize_recordings(items)
    summary = _summarize(items)
    _add_recording_counts(summary, recordings)
    overall_status = _compute_status({resolved_core_status(item) for item in items}) if items else "skip"
    return BatchDiagnosticsReport(
        overall_status=overall_status,
        roots=[str(Path(path).expanduser()) for path in paths],
        recursive=bool(recursive),
        items=items,
        summary=summary,
        recordings=recordings,
    )
