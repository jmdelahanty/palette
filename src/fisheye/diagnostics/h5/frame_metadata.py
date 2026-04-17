from __future__ import annotations

import h5py
import numpy as np

from .core import REQUIRED_FRAME_METADATA_FIELDS
from .models import Finding, FrameMetadataInfo
from .reader import dataset_fields, dataset_row_count

EXPECTED_RATIO = 2.0
RATIO_WARN_THRESHOLD = 0.5
MAX_CUMULATIVE_DRIFT_WARN_THRESHOLD = 3.0


def _max_true_run_length(mask: np.ndarray) -> int:
    if mask.size == 0 or not np.any(mask):
        return 0
    indices = np.flatnonzero(mask)
    max_run = 1
    current_run = 1
    for prev, current in zip(indices, indices[1:]):
        if current == prev + 1:
            current_run += 1
        else:
            max_run = max(max_run, current_run)
            current_run = 1
    return max(max_run, current_run)


def inspect_frame_metadata(
    handle: h5py.File,
    *,
    required: bool = True,
    expected_ratio: float = EXPECTED_RATIO,
    warn_threshold: float = RATIO_WARN_THRESHOLD,
    max_cumulative_drift_warn_threshold: float = MAX_CUMULATIVE_DRIFT_WARN_THRESHOLD,
) -> tuple[FrameMetadataInfo, list[Finding]]:
    dataset = handle.get("/video_metadata/frame_metadata")
    if dataset is None:
        status = "fail" if required else "skip"
        findings: list[Finding] = []
        if required:
            findings.append(
                Finding(
                    severity="fail",
                    code="h5.frame_metadata_missing",
                    summary="/video_metadata/frame_metadata dataset is missing.",
                    component="frame_metadata",
                    kind="core",
                )
            )
        return FrameMetadataInfo(status=status, error="frame_metadata missing"), findings

    fields = dataset_fields(dataset)
    rows = dataset_row_count(dataset)
    info = FrameMetadataInfo(status="pass", rows=rows, fields=fields, expected_ratio=expected_ratio)
    findings: list[Finding] = []

    missing_fields = [field for field in REQUIRED_FRAME_METADATA_FIELDS if field not in fields]
    if missing_fields:
        info.status = "fail"
        findings.append(
            Finding(
                severity="fail",
                code="h5.frame_metadata_fields_missing",
                summary="Required frame_metadata fields are missing.",
                details=", ".join(missing_fields),
                component="frame_metadata",
                kind="core",
            )
        )
        return info, findings

    if rows == 0:
        info.status = "fail"
        findings.append(
            Finding(
                severity="fail",
                code="h5.frame_metadata_empty",
                summary="frame_metadata dataset is empty.",
                component="frame_metadata",
                kind="core",
            )
        )
        return info, findings

    data = dataset[:]
    stimulus_frames = data["stimulus_frame_num"].astype(np.int64, copy=False)
    camera_frames = data["triggering_camera_frame_id"].astype(np.int64, copy=False)

    info.stimulus_monotonic = bool(np.all(np.diff(stimulus_frames) >= 0)) if stimulus_frames.size > 1 else True
    if info.stimulus_monotonic is False:
        info.status = "fail"
        findings.append(
            Finding(
                severity="fail",
                code="h5.frame_metadata_stimulus_nonmonotonic",
                summary="stimulus_frame_num is not monotonic.",
                component="frame_metadata",
                kind="core",
            )
        )

    valid_camera = camera_frames[camera_frames > 0]
    if valid_camera.size == 0:
        info.status = "fail"
        findings.append(
            Finding(
                severity="fail",
                code="h5.frame_metadata_no_camera_frames",
                summary="frame_metadata has no positive triggering_camera_frame_id values.",
                component="frame_metadata",
                kind="core",
            )
        )
        return info, findings

    info.camera_nondecreasing = bool(np.all(np.diff(valid_camera) >= 0)) if valid_camera.size > 1 else True
    if info.camera_nondecreasing is False:
        info.status = "fail"
        findings.append(
            Finding(
                severity="fail",
                code="h5.frame_metadata_camera_nonmonotonic",
                summary="triggering_camera_frame_id is not nondecreasing.",
                component="frame_metadata",
                kind="core",
            )
        )

    unique_camera, counts = np.unique(valid_camera, return_counts=True)
    info.unique_camera_frames = int(unique_camera.size)
    info.camera_min = int(unique_camera[0])
    info.camera_max = int(unique_camera[-1])
    info.missing_camera_frames = int((info.camera_max - info.camera_min + 1) - unique_camera.size)
    info.mean_stimulus_per_camera = float(np.mean(counts))
    info.median_stimulus_per_camera = float(np.median(counts))

    irregular_mask = np.abs(counts.astype(float) - expected_ratio) > warn_threshold
    info.ratio_warn_count = int(np.count_nonzero(irregular_mask))
    info.ratio_warn_fraction = float(info.ratio_warn_count / unique_camera.size)
    info.max_ratio_warn_run_length = _max_true_run_length(irregular_mask)

    cumulative_drift = np.cumsum(counts.astype(float) - expected_ratio)
    info.max_abs_cumulative_drift = float(np.max(np.abs(cumulative_drift))) if cumulative_drift.size else 0.0

    warn_reasons: list[str] = []
    if info.missing_camera_frames > 0:
        warn_reasons.append(f"missing_camera_frames={info.missing_camera_frames}")
    if info.max_abs_cumulative_drift is not None and info.max_abs_cumulative_drift > max_cumulative_drift_warn_threshold:
        warn_reasons.append(
            f"max_abs_cumulative_drift={info.max_abs_cumulative_drift:.3f}"
        )

    if info.status == "pass" and warn_reasons:
        info.status = "warn"
        findings.append(
            Finding(
                severity="warn",
                code="h5.frame_metadata_alignment_irregular",
                summary="frame_metadata camera/stimulus alignment looks irregular.",
                details=(
                    ", ".join(warn_reasons)
                    + f"; ratio_warn_count={info.ratio_warn_count}"
                    + f"; max_ratio_warn_run_length={info.max_ratio_warn_run_length}"
                ),
                component="frame_metadata",
                kind="core",
            )
        )

    return info, findings
