from __future__ import annotations

from collections import Counter
from typing import Any, Optional

import numpy as np

from .models import Finding, GOPInfo
from .timing import _extract_timestamp


def analyze_gop_frames(
    frames: list[dict[str, Any]],
    *,
    scope: str,
    codec: Optional[str] = None,
    avg_fps: Optional[float] = None,
) -> tuple[GOPInfo, list[Finding]]:
    info = GOPInfo(scope=scope, frames_analyzed=len(frames))
    findings: list[Finding] = []
    if not frames:
        info.status = "fail"
        info.error = "No frame metadata returned by ffprobe."
        findings.append(
            Finding(
                severity="fail",
                code="video.no_gop_metadata",
                summary="ffprobe did not return frame-level GOP metadata.",
                component="gop",
            )
        )
        return info, findings

    frame_types = Counter(str(frame.get("pict_type", "U")) for frame in frames)
    keyframe_positions: list[int] = []
    keyframe_times: list[float] = []
    current_gop_start = 0
    gop_sizes: list[int] = []
    for index, frame in enumerate(frames):
        is_keyframe = frame.get("key_frame") == 1 or frame.get("pict_type") == "I"
        if not is_keyframe:
            continue
        if index > 0:
            gop_sizes.append(index - current_gop_start)
        current_gop_start = index
        keyframe_positions.append(index)
        timestamp = _extract_timestamp(frame, ("pkt_pts_time", "pts_time", "best_effort_timestamp_time"))
        if timestamp is not None:
            keyframe_times.append(timestamp)

    info.status = "pass"
    info.frame_type_counts = dict(frame_types)
    info.keyframe_count = len(keyframe_positions)
    info.b_frame_count = int(frame_types.get("B", 0))
    info.b_frames_present = info.b_frame_count > 0
    info.max_gop_frames = max(gop_sizes) if gop_sizes else None

    intervals_s: list[float] = []
    if len(keyframe_times) == len(keyframe_positions) and len(keyframe_times) > 1:
        intervals_s = np.diff(np.asarray(keyframe_times, dtype=np.float64)).tolist()
    elif avg_fps is not None and avg_fps > 0 and len(keyframe_positions) > 1:
        intervals_s = (np.diff(np.asarray(keyframe_positions, dtype=np.float64)) / float(avg_fps)).tolist()
    if intervals_s:
        info.avg_keyframe_interval_s = float(np.mean(intervals_s))
        info.min_keyframe_interval_s = float(np.min(intervals_s))
        info.max_keyframe_interval_s = float(np.max(intervals_s))

    if info.max_keyframe_interval_s is not None and info.max_keyframe_interval_s > 10.0:
        info.status = "warn"
        findings.append(
            Finding(
                severity="warn",
                code="video.sparse_keyframes",
                summary="Keyframes are spaced far apart in the inspected frames.",
                details=f"Maximum keyframe interval: {info.max_keyframe_interval_s:.2f}s",
                component="gop",
            )
        )
    if info.max_gop_frames is not None and info.max_gop_frames > 300:
        info.status = "warn"
        findings.append(
            Finding(
                severity="warn",
                code="video.large_gop",
                summary="A large GOP was detected in the inspected frames.",
                details=f"Maximum GOP size: {info.max_gop_frames} frames",
                component="gop",
            )
        )
    if info.b_frames_present and codec == "hevc":
        info.status = "warn"
        findings.append(
            Finding(
                severity="warn",
                code="video.hevc_b_frames",
                summary="HEVC stream contains B-frames, which can make seeking more fragile.",
                component="gop",
            )
        )
    return info, findings
