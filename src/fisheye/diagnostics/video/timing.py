from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from .models import Finding, TimingGap, TimingInfo


def _extract_timestamp(frame: dict[str, Any], keys: Iterable[str]) -> float | None:
    for key in keys:
        value = frame.get(key)
        if value in (None, "", "N/A"):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _is_monotonic(values: list[float]) -> bool | None:
    if len(values) < 2:
        return None
    diffs = np.diff(np.asarray(values, dtype=np.float64))
    return bool(np.all(diffs >= 0))


def analyze_timing_frames(frames: list[dict[str, Any]], *, scope: str) -> tuple[TimingInfo, list[Finding]]:
    info = TimingInfo(scope=scope, frames_analyzed=len(frames))
    findings: list[Finding] = []
    if not frames:
        info.status = "fail"
        info.error = "No frame metadata returned by ffprobe."
        findings.append(
            Finding(
                severity="fail",
                code="video.no_frame_metadata",
                summary="ffprobe did not return frame-level timing metadata.",
                component="timing",
            )
        )
        return info, findings

    pts_times: list[float] = []
    dts_times: list[float] = []
    for frame in frames:
        pts = _extract_timestamp(frame, ("pkt_pts_time", "pts_time", "best_effort_timestamp_time"))
        dts = _extract_timestamp(frame, ("pkt_dts_time", "dts_time"))
        if pts is not None:
            pts_times.append(pts)
        if dts is not None:
            dts_times.append(dts)

    info.pts_present = bool(pts_times)
    info.dts_present = bool(dts_times)
    info.pts_monotonic = _is_monotonic(pts_times)
    info.dts_monotonic = _is_monotonic(dts_times)

    series = pts_times if pts_times else dts_times
    info.timing_basis = "pts" if pts_times else ("dts" if dts_times else None)
    if len(series) > 1:
        diffs = np.diff(np.asarray(series, dtype=np.float64))
        positive_diffs = diffs[diffs > 0]
        if positive_diffs.size > 0:
            expected_interval = float(np.median(positive_diffs))
            info.median_interval_ms = expected_interval * 1000.0
            info.mean_interval_ms = float(np.mean(diffs)) * 1000.0
            info.std_interval_ms = float(np.std(diffs)) * 1000.0
            gap_threshold = expected_interval * 1.5
            gaps: list[TimingGap] = []
            for idx, diff in enumerate(diffs):
                if diff > gap_threshold:
                    estimated_missing = max(int(round(diff / expected_interval)) - 1, 1)
                    gaps.append(
                        TimingGap(
                            position=int(idx),
                            time_seconds=float(series[idx]),
                            gap_duration_seconds=float(diff),
                            estimated_missing_frames=int(estimated_missing),
                        )
                    )
            info.gaps = gaps
            info.gap_count = len(gaps)
            info.estimated_missing_frames = int(sum(gap.estimated_missing_frames for gap in gaps))
            info.max_gap_ms = max((gap.gap_duration_seconds for gap in gaps), default=0.0) * 1000.0

    info.status = "pass"
    if not info.pts_present and not info.dts_present:
        info.status = "warn"
        findings.append(
            Finding(
                severity="warn",
                code="video.timestamps_missing",
                summary="No usable PTS or DTS timestamps were found.",
                component="timing",
            )
        )
        return info, findings

    if info.pts_monotonic is False:
        info.status = "fail"
        findings.append(
            Finding(
                severity="fail",
                code="video.pts_non_monotonic",
                summary="PTS values are not monotonic in the inspected frames.",
                component="timing",
            )
        )
    if info.gap_count > 0:
        info.status = "fail"
        findings.append(
            Finding(
                severity="fail",
                code="video.timestamp_gaps",
                summary=f"Detected {info.gap_count} suspicious timestamp gap(s).",
                details=f"Estimated missing frames: {info.estimated_missing_frames}",
                component="timing",
            )
        )
    if info.dts_monotonic is False and info.status != "fail":
        info.status = "warn"
        findings.append(
            Finding(
                severity="warn",
                code="video.dts_non_monotonic",
                summary="DTS values are not monotonic in the inspected frames.",
                component="timing",
            )
        )
    return info, findings
