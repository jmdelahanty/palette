from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from . import build_video_report
from .models import (
    BatchDiagnosticsReport,
    BatchSummary,
    RecordingDiagnosticsSummary,
    VideoDiagnosticsReport,
    resolved_media_status,
    resolved_tooling_status,
)
from .probe import classify_video_source

VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".avi")


def _is_video_path(path: Path) -> bool:
    return (
        path.is_file()
        and path.suffix.lower() in VIDEO_EXTENSIONS
        and not path.name.lower().endswith("_fixed.mp4")
    )


def _matches_source_filter(path: Path, source: str) -> bool:
    if source == "all":
        return True
    return classify_video_source(path) == source


def iter_video_paths(paths: Iterable[Path], *, recursive: bool, source: str = "all") -> Iterable[Path]:
    seen: set[Path] = set()
    for raw_path in paths:
        path = raw_path.expanduser()
        if _is_video_path(path) and _matches_source_filter(path, source):
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                yield resolved
            continue
        if not path.exists() or not path.is_dir():
            continue
        candidates: list[Path] = []
        for extension in VIDEO_EXTENSIONS:
            pattern = f"*{extension}"
            if recursive:
                candidates.extend(path.rglob(pattern))
            else:
                candidates.extend(path.glob(pattern))
        for candidate in sorted(candidates):
            if not _is_video_path(candidate) or not _matches_source_filter(candidate, source):
                continue
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            yield resolved


def _summarize(items: list[VideoDiagnosticsReport]) -> BatchSummary:
    summary = BatchSummary(scanned=len(items))
    for item in items:
        media_status = resolved_media_status(item)
        tooling_status = resolved_tooling_status(item)
        source_kind = item.file_info.source_kind or "other"
        summary.source_counts[source_kind] = summary.source_counts.get(source_kind, 0) + 1
        if media_status == "pass":
            summary.passed += 1
        elif media_status == "warn":
            summary.warned += 1
        elif media_status == "fail":
            summary.failed += 1
        elif media_status == "error":
            summary.errored += 1
        else:
            summary.skipped += 1
        summary.tooling_counts[tooling_status] = summary.tooling_counts.get(tooling_status, 0) + 1
    return summary


def _add_recording_counts(summary: BatchSummary, recordings: list[RecordingDiagnosticsSummary]) -> None:
    for recording in recordings:
        summary.recording_counts[recording.media_status] = (
            summary.recording_counts.get(recording.media_status, 0) + 1
        )
        summary.tooling_recording_counts[recording.tooling_status] = (
            summary.tooling_recording_counts.get(recording.tooling_status, 0) + 1
        )


def _compute_batch_status(summary: BatchSummary) -> str:
    if summary.failed > 0:
        return "fail"
    if summary.warned > 0 or summary.errored > 0:
        return "warn"
    if summary.passed > 0:
        return "pass"
    return "skip"


def _compute_media_status(items: list[VideoDiagnosticsReport]) -> str:
    statuses = {resolved_media_status(item) for item in items}
    if "fail" in statuses:
        return "fail"
    if "warn" in statuses or "error" in statuses:
        return "warn"
    if "pass" in statuses:
        return "pass"
    return "skip"


def _compute_tooling_status(items: list[VideoDiagnosticsReport]) -> str:
    statuses = {resolved_tooling_status(item) for item in items}
    if "fail" in statuses:
        return "fail"
    if "error" in statuses:
        return "error"
    if "warn" in statuses:
        return "warn"
    if "pass" in statuses:
        return "pass"
    return "skip"


def _summarize_recordings(items: list[VideoDiagnosticsReport]) -> list[RecordingDiagnosticsSummary]:
    grouped: dict[str, list[VideoDiagnosticsReport]] = {}
    for item in items:
        recording_root = item.file_info.recording_root or str(Path(item.file_info.path).parent)
        grouped.setdefault(recording_root, []).append(item)

    summaries: list[RecordingDiagnosticsSummary] = []
    for recording_root in sorted(grouped):
        recording_items = grouped[recording_root]
        source_counts = {"cams": 0, "raw": 0, "other": 0}
        for item in recording_items:
            source_kind = item.file_info.source_kind or "other"
            source_counts[source_kind] = source_counts.get(source_kind, 0) + 1
        media_status = _compute_media_status(recording_items)
        tooling_status = _compute_tooling_status(recording_items)
        summaries.append(
            RecordingDiagnosticsSummary(
                recording_root=recording_root,
                overall_status=media_status,
                media_status=media_status,
                tooling_status=tooling_status,
                item_count=len(recording_items),
                source_counts=source_counts,
            )
        )
    return summaries


def build_batch_report(
    paths: Iterable[Path],
    *,
    recursive: bool,
    limit: Optional[int] = None,
    source: str = "all",
    **kwargs: object,
) -> BatchDiagnosticsReport:
    selected: list[Path] = []
    for video_path in iter_video_paths(paths, recursive=recursive, source=source):
        selected.append(video_path)
        if limit is not None and limit > 0 and len(selected) >= limit:
            break

    items = [build_video_report(video_path, **kwargs) for video_path in selected]
    recordings = _summarize_recordings(items)
    summary = _summarize(items)
    _add_recording_counts(summary, recordings)
    return BatchDiagnosticsReport(
        overall_status=_compute_batch_status(summary),
        roots=[str(Path(path).expanduser()) for path in paths],
        recursive=bool(recursive),
        items=items,
        summary=summary,
        recordings=recordings,
    )
