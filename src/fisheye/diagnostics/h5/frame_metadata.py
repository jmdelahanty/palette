from __future__ import annotations

import json
import math

import h5py
import numpy as np

from .core import missing_required_frame_metadata_fields
from .models import Finding, FrameMetadataInfo
from .reader import dataset_fields, dataset_row_count

EXPECTED_RATIO = 2.0
RATIO_WARN_THRESHOLD = 0.5
MAX_CUMULATIVE_DRIFT_WARN_THRESHOLD = 3.0
MISSING_CAMERA_FRAME_FRACTION_WARN_THRESHOLD = 0.001
INFERRED_RATIO_RELATIVE_WARN_THRESHOLD = 0.05
INFERRED_RATIO_ABSOLUTE_WARN_THRESHOLD = 0.05
TIMESTAMP_FIELDS = ("timestamp_ns_epoch", "timestamp_ns", "timestamp_ns_session", "relative_timestamp_ns")


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


def _coerce_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result) or result <= 0:
        return None
    return result


def _read_json_dataset(handle: h5py.File, path: str) -> object | None:
    dataset = handle.get(path)
    if dataset is None:
        return None
    try:
        value = dataset[()]
    except Exception:
        return None
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    elif isinstance(value, np.bytes_):
        text = bytes(value).decode("utf-8", errors="replace")
    else:
        text = str(value)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _collect_json_rates(value: object) -> list[float]:
    rates: list[float] = []
    if isinstance(value, dict):
        for key, nested_value in value.items():
            if str(key) in {"frame_rate", "camera_frame_rate_configured_hz", "fps", "camera_fps"}:
                rate = _coerce_float(nested_value)
                if rate is not None:
                    rates.append(rate)
            rates.extend(_collect_json_rates(nested_value))
    elif isinstance(value, list):
        for item in value:
            rates.extend(_collect_json_rates(item))
    return rates


def _collapse_consistent_rates(rates: list[float]) -> float | None:
    if not rates:
        return None
    values = np.asarray(rates, dtype=np.float64)
    median_value = float(np.median(values))
    if not math.isfinite(median_value) or median_value <= 0:
        return None
    if np.max(np.abs(values - median_value)) <= max(1e-6, median_value * 0.01):
        return median_value
    return None


def _infer_camera_fps(handle: h5py.File) -> float | None:
    rates: list[float] = []
    camera_metadata = handle.get("/associated_cameras/camera_metadata")
    if isinstance(camera_metadata, h5py.Group):
        for camera_group in camera_metadata.values():
            if not isinstance(camera_group, h5py.Group):
                continue
            rate = _coerce_float(camera_group.attrs.get("camera_frame_rate_configured_hz"))
            if rate is not None:
                rates.append(rate)
    collapsed = _collapse_consistent_rates(rates)
    if collapsed is not None:
        return collapsed

    snapshot = _read_json_dataset(handle, "/recording_snapshot/recording_snapshot_json")
    if snapshot is not None:
        collapsed = _collapse_consistent_rates(_collect_json_rates(snapshot))
        if collapsed is not None:
            return collapsed
    return None


def _resolve_timestamp_field(fields: list[str]) -> str | None:
    for field in TIMESTAMP_FIELDS:
        if field in fields:
            return field
    return None


def _infer_stimulus_fps(data: np.ndarray, fields: list[str]) -> float | None:
    timestamp_field = _resolve_timestamp_field(fields)
    if timestamp_field is None:
        return None
    timestamps = data[timestamp_field].astype(np.int64, copy=False)
    if timestamps.size < 2:
        return None
    deltas = np.diff(timestamps)
    positive_deltas = deltas[deltas > 0]
    if positive_deltas.size == 0:
        return None
    median_delta_ns = float(np.median(positive_deltas.astype(np.float64)))
    if not math.isfinite(median_delta_ns) or median_delta_ns <= 0:
        return None
    return 1_000_000_000.0 / median_delta_ns


def _resolve_expected_ratio(
    handle: h5py.File,
    data: np.ndarray,
    fields: list[str],
    expected_ratio: float | None,
) -> tuple[float, str, float | None, float | None]:
    if expected_ratio is not None:
        return float(expected_ratio), "explicit", None, None
    stimulus_fps = _infer_stimulus_fps(data, fields)
    camera_fps = _infer_camera_fps(handle)
    if stimulus_fps is not None and camera_fps is not None:
        return float(stimulus_fps / camera_fps), "inferred_from_timestamps_and_acquisition_metadata", camera_fps, stimulus_fps
    return EXPECTED_RATIO, "fallback_default", camera_fps, stimulus_fps


def _ratio_tolerance(expected_ratio: float) -> float:
    return max(INFERRED_RATIO_ABSOLUTE_WARN_THRESHOLD, abs(expected_ratio) * INFERRED_RATIO_RELATIVE_WARN_THRESHOLD)


def _expected_count_bounds(expected_ratio: float) -> tuple[int, int]:
    if expected_ratio < 1.0:
        return 1, 1
    nearest = round(expected_ratio)
    if math.isclose(expected_ratio, float(nearest), rel_tol=0.0, abs_tol=1e-6):
        return int(nearest), int(nearest)
    return max(1, int(math.floor(expected_ratio))), max(1, int(math.ceil(expected_ratio)))


def inspect_frame_metadata(
    handle: h5py.File,
    *,
    required: bool = True,
    expected_ratio: float | None = None,
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
    initial_expected_ratio = EXPECTED_RATIO if expected_ratio is None else float(expected_ratio)
    info = FrameMetadataInfo(status="pass", rows=rows, fields=fields, expected_ratio=initial_expected_ratio)
    findings: list[Finding] = []

    missing_fields = missing_required_frame_metadata_fields(fields)
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
    resolved_expected_ratio, expected_ratio_source, camera_fps, stimulus_fps = _resolve_expected_ratio(
        handle,
        data,
        fields,
        expected_ratio,
    )
    info.expected_ratio = resolved_expected_ratio
    info.expected_ratio_source = expected_ratio_source
    info.inferred_camera_fps = camera_fps
    info.inferred_stimulus_fps = stimulus_fps

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
    if unique_camera.size > 1:
        info.mean_camera_frame_step = float(np.mean(np.diff(unique_camera).astype(np.float64)))
    if resolved_expected_ratio > 0:
        info.expected_camera_frame_step = float(1.0 / resolved_expected_ratio)

    lower_count, upper_count = _expected_count_bounds(resolved_expected_ratio)
    irregular_mask = (counts < lower_count) | (counts > upper_count)
    info.ratio_warn_count = int(np.count_nonzero(irregular_mask))
    info.ratio_warn_fraction = float(info.ratio_warn_count / unique_camera.size)
    info.max_ratio_warn_run_length = _max_true_run_length(irregular_mask)

    cumulative_drift = np.cumsum(counts.astype(float) - resolved_expected_ratio)
    info.max_abs_cumulative_drift = float(np.max(np.abs(cumulative_drift))) if cumulative_drift.size else 0.0

    warn_reasons: list[str] = []
    observed_camera_span = max(1, int(info.camera_max - info.camera_min + 1))
    missing_camera_fraction = float(info.missing_camera_frames / observed_camera_span)
    if resolved_expected_ratio >= 1.0:
        if missing_camera_fraction > MISSING_CAMERA_FRAME_FRACTION_WARN_THRESHOLD:
            warn_reasons.append(
                f"missing_camera_frames={info.missing_camera_frames}"
                f" ({missing_camera_fraction:.6f})"
            )
        if expected_ratio_source == "inferred_from_timestamps_and_acquisition_metadata" and info.ratio_warn_count > 0:
            warn_reasons.append(
                f"row_counts_outside_expected_bounds={info.ratio_warn_count}"
                f" (expected {lower_count}-{upper_count})"
            )
        observed_ratio = info.mean_stimulus_per_camera
        if observed_ratio is not None and abs(observed_ratio - resolved_expected_ratio) > _ratio_tolerance(resolved_expected_ratio):
            warn_reasons.append(
                f"mean_stimulus_per_camera={observed_ratio:.6f}"
                f" expected={resolved_expected_ratio:.6f}"
            )
        if expected_ratio_source in {"explicit", "fallback_default"} and info.max_abs_cumulative_drift is not None and info.max_abs_cumulative_drift > max_cumulative_drift_warn_threshold:
            warn_reasons.append(
                f"max_abs_cumulative_drift={info.max_abs_cumulative_drift:.3f}"
            )
    else:
        if info.ratio_warn_count > 0:
            warn_reasons.append(
                f"multiple_metadata_rows_on_camera_frame={info.ratio_warn_count}"
            )
        if info.mean_camera_frame_step is not None and info.expected_camera_frame_step is not None:
            if abs(info.mean_camera_frame_step - info.expected_camera_frame_step) > _ratio_tolerance(info.expected_camera_frame_step):
                warn_reasons.append(
                    f"mean_camera_frame_step={info.mean_camera_frame_step:.6f}"
                    f" expected={info.expected_camera_frame_step:.6f}"
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
