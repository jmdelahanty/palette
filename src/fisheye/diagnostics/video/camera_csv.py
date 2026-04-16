from __future__ import annotations

import csv
from pathlib import Path
from statistics import median

from .models import CameraCsvInfo, Finding
from .probe import classify_video_source

REQUIRED_CAMERA_CSV_COLUMNS = ("frame_id", "timestamp", "timestamp_sys")


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


def _parse_int(value: object, *, field_name: str, row_number: int) -> int:
    text = str(value).strip()
    if not text:
        raise ValueError(f"Row {row_number}: missing value for {field_name}")
    try:
        return int(text)
    except ValueError as exc:
        raise ValueError(f"Row {row_number}: invalid integer for {field_name}: {text}") from exc


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
            reader = csv.DictReader(handle)
            fieldnames = [str(name).strip() for name in (reader.fieldnames or [])]
            missing_columns = [name for name in REQUIRED_CAMERA_CSV_COLUMNS if name not in fieldnames]
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

            frame_ids: list[int] = []
            timestamps: list[int] = []
            timestamps_sys: list[int] = []
            for row_number, row in enumerate(reader, start=2):
                frame_ids.append(_parse_int(row.get("frame_id"), field_name="frame_id", row_number=row_number))
                timestamps.append(_parse_int(row.get("timestamp"), field_name="timestamp", row_number=row_number))
                timestamps_sys.append(_parse_int(row.get("timestamp_sys"), field_name="timestamp_sys", row_number=row_number))
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
    info.timestamp_offset_first_ns = timestamps_sys[0] - timestamps[0]
    info.timestamp_offset_last_ns = timestamps_sys[-1] - timestamps[-1]
    info.timestamp_offset_drift_ns = info.timestamp_offset_last_ns - info.timestamp_offset_first_ns
    if expected_frame_count is not None:
        info.row_count_matches_video = info.rows == int(expected_frame_count)

    info.status = "pass"
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
