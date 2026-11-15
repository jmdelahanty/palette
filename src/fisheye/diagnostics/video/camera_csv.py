from __future__ import annotations

import csv
from pathlib import Path
from statistics import median

import numpy as np

from fisheye.shared.acquisition_frame_clock import (
    CLOCK_CSV_TIMESTAMP_FIELDS,
    clock_csv_frame_field,
    parse_acquisition_frame_clock_csv,
)

from .models import CameraCsvInfo, Finding
from .probe import classify_video_source

REQUIRED_CAMERA_CSV_COLUMNS = ("recording_frame_id", *CLOCK_CSV_TIMESTAMP_FIELDS)


def expected_camera_csv_path(video_path: Path) -> Path:
    return video_path.with_name(f"{video_path.stem}_meta.csv")


def _is_monotonic(values: list[int]) -> bool | None:
    if len(values) < 2:
        return None
    return all(curr >= prev for prev, curr in zip(values, values[1:]))


def _is_contiguous(values: list[int]) -> bool | None:
    if len(values) < 2:
        return None
    return all((curr - prev) == 1 for prev, curr in zip(values, values[1:]))


def _median_step(values: list[int]) -> int | None:
    if len(values) < 2:
        return None
    diffs = [curr - prev for prev, curr in zip(values, values[1:])]
    if not diffs:
        return None
    return int(median(diffs))


def inspect_camera_csv(
    video_path: Path,
    *,
    expected_frame_count: int | None = None,
) -> tuple[CameraCsvInfo, list[Finding]]:
    csv_path = expected_camera_csv_path(video_path)
    info = CameraCsvInfo(
        path=str(csv_path),
        exists=csv_path.exists(),
        video_frame_count=expected_frame_count,
    )
    if classify_video_source(video_path) != "cams":
        return info, []

    findings: list[Finding] = []
    if not csv_path.exists():
        info.status = "warn"
        findings.append(
            Finding(
                severity="warn",
                code="video.camera_csv_missing",
                summary="Expected camera metadata CSV is missing next to the cams video.",
                details=str(csv_path),
                component="camera_csv",
            )
        )
        return info, findings

    try:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            fieldnames = [
                str(name).strip() for name in (csv.DictReader(handle).fieldnames or [])
            ]
        missing_columns = [
            name
            for name in REQUIRED_CAMERA_CSV_COLUMNS
            if (
                clock_csv_frame_field(fieldnames) is None
                if name == "recording_frame_id"
                else name not in fieldnames
            )
        ]
        info.schema_ok = not missing_columns
        info.missing_columns = missing_columns
        if missing_columns:
            info.status = "fail"
            findings.append(
                Finding(
                    severity="fail",
                    code="video.camera_csv_schema",
                    summary="Camera metadata CSV is missing required columns.",
                    details=f"Missing columns: {', '.join(missing_columns)}",
                    component="camera_csv",
                )
            )
            return info, findings
        # The clock authority's own parser, so this report reads rows exactly as
        # clock publication does (empty timestamps are invalid, not parse errors).
        parsed = parse_acquisition_frame_clock_csv(csv_path)
    except ValueError as exc:
        info.status = "fail"
        info.error = str(exc)
        findings.append(
            Finding(
                severity="fail",
                code="video.camera_csv_parse_error",
                summary="Camera metadata CSV contains invalid row data.",
                details=str(exc),
                component="camera_csv",
            )
        )
        return info, findings
    except OSError as exc:
        info.status = "fail"
        info.error = str(exc)
        findings.append(
            Finding(
                severity="fail",
                code="video.camera_csv_read_error",
                summary="Could not read the camera metadata CSV.",
                details=str(exc),
                component="camera_csv",
            )
        )
        return info, findings

    frame_ids = [int(value) for value in parsed.recording_frame_id]
    timestamps = [
        int(value)
        for value in parsed.camera_timestamp_ns[parsed.camera_timestamp_valid]
    ]
    timestamps_sys = [
        int(value)
        for value in parsed.system_timestamp_ns[parsed.system_timestamp_valid]
    ]
    both_valid = parsed.camera_timestamp_valid & parsed.system_timestamp_valid
    offsets = [
        int(value)
        for value in (
            parsed.system_timestamp_ns[both_valid] - parsed.camera_timestamp_ns[both_valid]
        )
    ]
    invalid_timestamp_rows = int(np.count_nonzero(~both_valid))
    info.rows = len(frame_ids)
    if not frame_ids:
        info.status = "fail"
        findings.append(
            Finding(
                severity="fail",
                code="video.camera_csv_empty",
                summary="Camera metadata CSV has no data rows.",
                component="camera_csv",
            )
        )
        return info, findings

    info.frame_id_first = frame_ids[0]
    info.frame_id_last = frame_ids[-1]
    info.frame_id_monotonic = _is_monotonic(frame_ids)
    info.frame_id_contiguous = _is_contiguous(frame_ids)
    info.timestamp_monotonic = _is_monotonic(timestamps)
    info.timestamp_sys_monotonic = _is_monotonic(timestamps_sys)
    info.median_timestamp_step_ns = _median_step(timestamps)
    info.median_timestamp_sys_step_ns = _median_step(timestamps_sys)
    if offsets:
        info.timestamp_offset_first_ns = offsets[0]
        info.timestamp_offset_last_ns = offsets[-1]
        info.timestamp_offset_drift_ns = offsets[-1] - offsets[0]
    if expected_frame_count is not None:
        info.row_count_matches_video = info.rows == int(expected_frame_count)

    info.status = "pass"
    if invalid_timestamp_rows:
        info.status = "warn"
        findings.append(
            Finding(
                severity="warn",
                code="video.camera_csv_invalid_timestamps",
                summary="Camera metadata rows have empty timestamps; the clock marks them invalid.",
                details=f"rows={invalid_timestamp_rows}",
                component="camera_csv",
            )
        )
    if info.frame_id_monotonic is False:
        info.status = "fail"
        findings.append(
            Finding(
                severity="fail",
                code="video.camera_csv_frame_id_non_monotonic",
                summary="Camera metadata frame IDs are not monotonic.",
                component="camera_csv",
            )
        )
    if info.frame_id_contiguous is False:
        info.status = "fail"
        findings.append(
            Finding(
                severity="fail",
                code="video.camera_csv_frame_id_non_contiguous",
                summary="Camera metadata frame IDs are not contiguous.",
                component="camera_csv",
            )
        )
    if info.timestamp_monotonic is False:
        info.status = "fail"
        findings.append(
            Finding(
                severity="fail",
                code="video.camera_csv_timestamp_non_monotonic",
                summary="Camera metadata timestamps are not monotonic.",
                component="camera_csv",
            )
        )
    if info.timestamp_sys_monotonic is False:
        info.status = "fail"
        findings.append(
            Finding(
                severity="fail",
                code="video.camera_csv_timestamp_sys_non_monotonic",
                summary="Camera metadata system timestamps are not monotonic.",
                component="camera_csv",
            )
        )
    if info.row_count_matches_video is False:
        info.status = "fail"
        findings.append(
            Finding(
                severity="fail",
                code="video.camera_csv_row_count_mismatch",
                summary="Camera metadata row count does not match the video frame count.",
                details=f"csv_rows={info.rows}, video_frames={expected_frame_count}",
                component="camera_csv",
            )
        )
    return info, findings
