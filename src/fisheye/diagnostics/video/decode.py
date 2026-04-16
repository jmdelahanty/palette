from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np

from .models import BackendDecodeReport, Finding, SeekInaccuracy


def _load_cv2():
    try:
        return importlib.import_module("cv2"), None
    except Exception as exc:  # pragma: no cover - dependency specific
        return None, str(exc)


def _load_decord():
    try:
        decord = importlib.import_module("decord")
        return decord, None
    except Exception as exc:  # pragma: no cover - dependency specific
        return None, str(exc)


def _build_positions(total_frames: int, samples: int) -> list[int]:
    if total_frames <= 0 or samples <= 0:
        return []
    if total_frames <= samples:
        return list(range(total_frames))
    return [int(pos) for pos in np.linspace(0, total_frames - 1, samples, dtype=int)]


def inspect_opencv(video_path: Path, *, frames_to_check: int, seek_samples: int) -> tuple[BackendDecodeReport, list[Finding]]:
    report = BackendDecodeReport(backend="opencv")
    findings: list[Finding] = []
    cv2, error = _load_cv2()
    if cv2 is None:
        report.status = "error"
        report.available = False
        report.error = error
        findings.append(
            Finding(
                severity="error",
                code="video.opencv_unavailable",
                summary="OpenCV is unavailable in this environment for decode diagnostics.",
                details=(
                    f"{error}. This is a backend/environment issue and does not by itself mean "
                    "the video file is broken."
                ),
                component="decode",
                kind="tooling",
            )
        )
        return report, findings

    report.available = True
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        report.status = "fail"
        report.open_ok = False
        findings.append(
            Finding(
                severity="fail",
                code="video.opencv_open_failed",
                summary="OpenCV could not open the video file.",
                component="decode",
            )
        )
        return report, findings

    report.open_ok = True
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    report.total_frames = total_frames

    sequential_positions = range(min(frames_to_check, total_frames if total_frames > 0 else frames_to_check))
    for index in sequential_positions:
        report.frames_checked += 1
        ret, frame = cap.read()
        if not ret or frame is None:
            report.sequential_failed_frames.append(int(index))
            cap.set(cv2.CAP_PROP_POS_FRAMES, index + 1)

    seek_positions = _build_positions(total_frames, seek_samples)
    report.seek_positions_checked = len(seek_positions)
    for position in seek_positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(position))
        observed = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        ret, frame = cap.read()
        if not ret or frame is None:
            report.seek_failures.append(int(position))
        elif abs(observed - int(position)) > 1:
            report.seek_inaccuracies.append(
                SeekInaccuracy(requested_frame=int(position), observed_frame=int(observed))
            )
    cap.release()

    report.status = "pass"
    if report.sequential_failed_frames or report.seek_failures:
        report.status = "fail"
    elif report.seek_inaccuracies:
        report.status = "warn"

    if report.sequential_failed_frames:
        findings.append(
            Finding(
                severity="fail",
                code="video.opencv_sequential_failures",
                summary="OpenCV failed to decode one or more sequential frames.",
                details=f"Failed frames: {report.sequential_failed_frames[:10]}",
                component="decode",
            )
        )
    if report.seek_failures:
        findings.append(
            Finding(
                severity="fail",
                code="video.opencv_seek_failures",
                summary="OpenCV failed random-access reads for one or more positions.",
                details=f"Seek failures: {report.seek_failures[:10]}",
                component="decode",
            )
        )
    if report.seek_inaccuracies and report.status != "fail":
        findings.append(
            Finding(
                severity="warn",
                code="video.opencv_seek_inaccuracy",
                summary="OpenCV seek positions were inaccurate for one or more samples.",
                details=f"Count: {len(report.seek_inaccuracies)}",
                component="decode",
            )
        )
    return report, findings


def inspect_decord(video_path: Path, *, frames_to_check: int, seek_samples: int) -> tuple[BackendDecodeReport, list[Finding]]:
    report = BackendDecodeReport(backend="decord")
    findings: list[Finding] = []
    decord, error = _load_decord()
    if decord is None:
        report.status = "error"
        report.available = False
        report.error = error
        findings.append(
            Finding(
                severity="error",
                code="video.decord_unavailable",
                summary="Decord is unavailable in this environment for decode diagnostics.",
                details=(
                    f"{error}. This is a backend/environment issue and does not by itself mean "
                    "the video file is broken."
                ),
                component="decode",
                kind="tooling",
            )
        )
        return report, findings

    report.available = True
    try:
        vr = decord.VideoReader(str(video_path), ctx=decord.cpu(0))
    except Exception as exc:
        report.status = "fail"
        report.open_ok = False
        report.error = str(exc)
        findings.append(
            Finding(
                severity="fail",
                code="video.decord_open_failed",
                summary="Decord could not open the video file.",
                details=str(exc),
                component="decode",
            )
        )
        return report, findings

    report.open_ok = True
    total_frames = len(vr)
    report.total_frames = total_frames
    sequential_positions = range(min(frames_to_check, total_frames if total_frames > 0 else frames_to_check))
    for index in sequential_positions:
        report.frames_checked += 1
        try:
            _ = vr[int(index)]
        except Exception:
            report.sequential_failed_frames.append(int(index))

    seek_positions = _build_positions(total_frames, seek_samples)
    report.seek_positions_checked = len(seek_positions)
    for position in seek_positions:
        try:
            _ = vr[int(position)]
        except Exception:
            report.seek_failures.append(int(position))

    report.status = "pass"
    if report.sequential_failed_frames or report.seek_failures:
        report.status = "fail"
    if report.sequential_failed_frames:
        findings.append(
            Finding(
                severity="fail",
                code="video.decord_sequential_failures",
                summary="Decord failed to decode one or more sequential frames.",
                details=f"Failed frames: {report.sequential_failed_frames[:10]}",
                component="decode",
            )
        )
    if report.seek_failures:
        findings.append(
            Finding(
                severity="fail",
                code="video.decord_seek_failures",
                summary="Decord failed random-access reads for one or more positions.",
                details=f"Seek failures: {report.seek_failures[:10]}",
                component="decode",
            )
        )
    return report, findings


def inspect_decode(
    video_path: Path,
    *,
    backend: str,
    frames_to_check: int,
    seek_samples: int,
) -> tuple[list[BackendDecodeReport], list[Finding]]:
    reports: list[BackendDecodeReport] = []
    findings: list[Finding] = []
    selections = ["opencv", "decord"] if backend == "all" else [backend]
    for name in selections:
        if name == "opencv":
            report, section_findings = inspect_opencv(
                video_path,
                frames_to_check=frames_to_check,
                seek_samples=seek_samples,
            )
        elif name == "decord":
            report, section_findings = inspect_decord(
                video_path,
                frames_to_check=frames_to_check,
                seek_samples=seek_samples,
            )
        else:
            report = BackendDecodeReport(
                backend=name,
                status="error",
                available=False,
                error=f"Unknown backend: {name}",
            )
            section_findings = [
                Finding(
                    severity="error",
                    code="video.unknown_backend",
                    summary=f"Unknown decode backend: {name}",
                    component="decode",
                    kind="tooling",
                )
            ]
        reports.append(report)
        findings.extend(section_findings)
    return reports, findings
