"""Generic helpers for event-aligned epoch/segment summaries."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
import zarr

from fisheye.shared.json_safety import decode_null_terminated_text


@dataclass(frozen=True)
class EpochSegment:
    segment_id: int
    label: str
    start_frame: int
    end_frame: int
    start_time_s: float
    end_time_s: float
    duration_s: float
    source_start_event_name: str | None = None
    source_end_event_name: str | None = None
    source_start_event_frame: int | None = None
    source_end_event_frame: int | None = None
    source_policy: str | None = None


@dataclass(frozen=True)
class HistogramMetricSpec:
    metric_name: str
    units: str
    bin_policy: str
    bin_width: float | None = None
    bin_edges: tuple[float, ...] | None = None
    range_min: float | None = None
    range_max: float | None = None
    include_overflow_bins: bool = False


def _decode_text_column(data: np.ndarray) -> list[str]:
    values = np.asarray(data)
    if values.ndim == 2 and values.dtype.kind in ("u", "i"):
        return [decode_null_terminated_text(row).strip() for row in values]
    return [decode_null_terminated_text(value).strip() for value in values.reshape(-1)]


def read_epoch_segments(epoch_group: zarr.Group) -> tuple[EpochSegment, ...]:
    """Read `analysis/stimulus_epoch_runs/<run>/windows` as generic segments."""

    windows = epoch_group.get("windows")
    if windows is None:
        raise ValueError("Epoch run has no windows group.")
    required = ("window_id", "label_bytes", "start_frame", "end_frame", "start_time_s", "end_time_s", "duration_s")
    missing = [name for name in required if name not in windows]
    if missing:
        raise ValueError(f"Epoch windows missing arrays: {missing}")

    ids = np.asarray(windows["window_id"][:], dtype=np.int32).reshape(-1)
    labels = _decode_text_column(np.asarray(windows["label_bytes"][:]))
    starts = np.asarray(windows["start_frame"][:], dtype=np.int64).reshape(-1)
    ends = np.asarray(windows["end_frame"][:], dtype=np.int64).reshape(-1)
    start_times = np.asarray(windows["start_time_s"][:], dtype=np.float64).reshape(-1)
    end_times = np.asarray(windows["end_time_s"][:], dtype=np.float64).reshape(-1)
    durations = np.asarray(windows["duration_s"][:], dtype=np.float64).reshape(-1)
    start_event_names = (
        _decode_text_column(np.asarray(windows["source_start_event_name_bytes"][:]))
        if "source_start_event_name_bytes" in windows
        else [""] * ids.shape[0]
    )
    end_event_names = (
        _decode_text_column(np.asarray(windows["source_end_event_name_bytes"][:]))
        if "source_end_event_name_bytes" in windows
        else [""] * ids.shape[0]
    )
    policies = (
        _decode_text_column(np.asarray(windows["source_policy_bytes"][:]))
        if "source_policy_bytes" in windows
        else [""] * ids.shape[0]
    )
    source_start_event_frames = (
        np.asarray(windows["source_start_event_frame"][:], dtype=np.int64).reshape(-1)
        if "source_start_event_frame" in windows
        else np.full(ids.shape[0], -1, dtype=np.int64)
    )
    source_end_event_frames = (
        np.asarray(windows["source_end_event_frame"][:], dtype=np.int64).reshape(-1)
        if "source_end_event_frame" in windows
        else np.full(ids.shape[0], -1, dtype=np.int64)
    )

    n = int(ids.shape[0])
    arrays = (labels, starts, ends, start_times, end_times, durations)
    if any(len(item) != n for item in arrays):
        raise ValueError("Epoch window arrays disagree on length.")
    return tuple(
        EpochSegment(
            segment_id=int(ids[i]),
            label=str(labels[i]),
            start_frame=int(starts[i]),
            end_frame=int(ends[i]),
            start_time_s=float(start_times[i]),
            end_time_s=float(end_times[i]),
            duration_s=float(durations[i]),
            source_start_event_name=str(start_event_names[i]) if i < len(start_event_names) else None,
            source_end_event_name=str(end_event_names[i]) if i < len(end_event_names) else None,
            source_start_event_frame=(
                int(source_start_event_frames[i])
                if i < len(source_start_event_frames)
                and int(source_start_event_frames[i]) >= 0
                else None
            ),
            source_end_event_frame=(
                int(source_end_event_frames[i])
                if i < len(source_end_event_frames)
                and int(source_end_event_frames[i]) >= 0
                else None
            ),
            source_policy=str(policies[i]) if i < len(policies) else None,
        )
        for i in range(n)
    )


def segments_from_window_objects(windows: Sequence[object]) -> tuple[EpochSegment, ...]:
    """Convert local window dataclasses into generic segment records."""

    out: list[EpochSegment] = []
    for index, window in enumerate(windows):
        out.append(
            EpochSegment(
                segment_id=int(getattr(window, "window_id", index)),
                label=str(getattr(window, "label", f"window_{index}")),
                start_frame=int(getattr(window, "start_frame")),
                end_frame=int(getattr(window, "end_frame")),
                start_time_s=float(getattr(window, "start_time_s", math.nan)),
                end_time_s=float(getattr(window, "end_time_s", math.nan)),
                duration_s=float(getattr(window, "duration_s", math.nan)),
                source_start_event_name=getattr(window, "source_start_event_name", None),
                source_end_event_name=getattr(window, "source_end_event_name", None),
                source_start_event_frame=getattr(window, "source_start_event_frame", None),
                source_end_event_frame=getattr(window, "source_end_event_frame", None),
                source_policy=getattr(window, "source_policy", None),
            )
        )
    return tuple(out)


def assign_frames_to_segments(total_frames: int, segments: Sequence[EpochSegment]) -> np.ndarray:
    out = np.full(int(total_frames), -1, dtype=np.int32)
    for index, segment in enumerate(segments):
        start = max(0, int(segment.start_frame))
        end = min(int(total_frames) - 1, int(segment.end_frame))
        if end >= start:
            out[start : end + 1] = int(index)
    return out


def assign_point_events_to_segments(frames: np.ndarray, segments: Sequence[EpochSegment]) -> np.ndarray:
    values = np.asarray(frames, dtype=np.int64).reshape(-1)
    out = np.full(values.shape[0], -1, dtype=np.int32)
    for index, segment in enumerate(segments):
        mask = (values >= int(segment.start_frame)) & (values <= int(segment.end_frame))
        out[mask] = int(index)
    return out


def assign_intervals_to_segments(
    start_frames: np.ndarray,
    end_frames: np.ndarray,
    segments: Sequence[EpochSegment],
    *,
    rule: str = "contained",
) -> np.ndarray:
    starts = np.asarray(start_frames, dtype=np.int64).reshape(-1)
    ends = np.asarray(end_frames, dtype=np.int64).reshape(-1)
    n = min(starts.shape[0], ends.shape[0])
    out = np.full(n, -1, dtype=np.int32)
    for index, segment in enumerate(segments):
        seg_start = int(segment.start_frame)
        seg_end = int(segment.end_frame)
        if rule == "start":
            mask = (starts[:n] >= seg_start) & (starts[:n] <= seg_end)
        elif rule == "end":
            mask = (ends[:n] >= seg_start) & (ends[:n] <= seg_end)
        elif rule == "midpoint":
            midpoint = np.floor((starts[:n] + ends[:n]) / 2.0).astype(np.int64)
            mask = (midpoint >= seg_start) & (midpoint <= seg_end)
        elif rule == "overlap":
            mask = (starts[:n] <= seg_end) & (ends[:n] >= seg_start)
        elif rule == "contained":
            mask = (starts[:n] >= seg_start) & (ends[:n] <= seg_end)
        else:
            raise ValueError(f"Unsupported interval assignment rule: {rule!r}")
        out[mask] = int(index)
    return out


def finite_summary(values: np.ndarray, percentiles: Sequence[float] = (5.0, 50.0, 95.0)) -> dict[str, float | int]:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = data[np.isfinite(data)]
    summary: dict[str, float | int] = {"count": int(finite.size)}
    if finite.size == 0:
        summary.update({"mean": math.nan, "min": math.nan, "max": math.nan})
        for percentile in percentiles:
            summary[f"p{int(percentile):02d}"] = math.nan
        return summary
    summary.update(
        {
            "mean": float(np.mean(finite)),
            "min": float(np.min(finite)),
            "max": float(np.max(finite)),
        }
    )
    for percentile in percentiles:
        summary[f"p{int(percentile):02d}"] = float(np.percentile(finite, float(percentile)))
    return summary


def rate_per_minute(count: int, duration_s: float) -> float:
    duration = float(duration_s)
    return float(count) / (duration / 60.0) if duration > 0 else math.nan


def _fixed_string(value: object, width: int) -> bytes:
    return str(value).encode("utf-8", "ignore")[: max(0, int(width) - 1)]


def _histogram_edges(values_by_segment: Sequence[np.ndarray], spec: HistogramMetricSpec) -> np.ndarray:
    if spec.bin_edges is not None:
        edges = np.asarray(spec.bin_edges, dtype=np.float64).reshape(-1)
        if edges.shape[0] < 2 or not np.all(np.diff(edges) > 0):
            raise ValueError(f"{spec.metric_name}: bin_edges must contain at least two increasing values.")
        return edges
    width = float(spec.bin_width if spec.bin_width is not None else math.nan)
    if not math.isfinite(width) or width <= 0:
        raise ValueError(f"{spec.metric_name}: bin_width must be positive when bin_edges are not supplied.")
    finite_parts = []
    for values in values_by_segment:
        data = np.asarray(values, dtype=np.float64).reshape(-1)
        finite = data[np.isfinite(data)]
        if finite.size:
            finite_parts.append(finite)
    finite_values = np.concatenate(finite_parts) if finite_parts else np.asarray([], dtype=np.float64)
    if spec.range_min is not None:
        left = float(spec.range_min)
    elif finite_values.size and np.nanmin(finite_values) < 0:
        left = math.floor(float(np.nanmin(finite_values)) / width) * width
    else:
        left = 0.0
    if spec.range_max is not None:
        right = float(spec.range_max)
    elif finite_values.size:
        right = math.ceil(float(np.nanmax(finite_values)) / width) * width
    else:
        right = left + width
    if not math.isfinite(left) or not math.isfinite(right):
        raise ValueError(f"{spec.metric_name}: non-finite histogram range.")
    if right <= left:
        right = left + width
    return np.arange(left, right + width * 0.5, width, dtype=np.float64)


def histogram_table(
    *,
    segments: Sequence[EpochSegment],
    values_by_segment: Sequence[np.ndarray],
    metric_spec: HistogramMetricSpec,
) -> np.ndarray:
    """Build a persisted table with shared bins across all segments."""

    segment_count = len(segments)
    if len(values_by_segment) != segment_count:
        raise ValueError("values_by_segment length must match segments length.")
    edges = _histogram_edges(values_by_segment, metric_spec)
    bin_count = int(edges.shape[0] - 1)
    dtype = np.dtype(
        [
            ("metric_name", "S96"),
            ("units", "S32"),
            ("window_id", np.int32),
            ("window_index", np.int32),
            ("window_label", "S96"),
            ("start_frame", np.int64),
            ("end_frame", np.int64),
            ("start_time_s", np.float64),
            ("end_time_s", np.float64),
            ("duration_s", np.float64),
            ("bin_index", np.int32),
            ("bin_left", np.float64),
            ("bin_right", np.float64),
            ("bin_center", np.float64),
            ("bin_width", np.float64),
            ("hist_count", np.int64),
            ("hist_fraction", np.float64),
            ("hist_density", np.float64),
            ("source_sample_count", np.int64),
            ("finite_sample_count", np.int64),
            ("bin_policy", "S96"),
        ]
    )
    rows = np.zeros(segment_count * bin_count, dtype=dtype)
    for name in rows.dtype.names or ():
        if rows.dtype[name].kind == "f":
            rows[name] = np.nan
    row_idx = 0
    for segment_index, segment in enumerate(segments):
        raw_values = np.asarray(values_by_segment[segment_index], dtype=np.float64).reshape(-1)
        finite_values = raw_values[np.isfinite(raw_values)]
        counts, _ = np.histogram(finite_values, bins=edges)
        total = int(np.sum(counts))
        for bin_index in range(bin_count):
            left = float(edges[bin_index])
            right = float(edges[bin_index + 1])
            width = float(right - left)
            count = int(counts[bin_index])
            fraction = float(count) / float(total) if total > 0 else math.nan
            rows[row_idx] = (
                _fixed_string(metric_spec.metric_name, 96),
                _fixed_string(metric_spec.units, 32),
                int(segment.segment_id),
                int(segment_index),
                _fixed_string(segment.label, 96),
                int(segment.start_frame),
                int(segment.end_frame),
                float(segment.start_time_s),
                float(segment.end_time_s),
                float(segment.duration_s),
                int(bin_index),
                left,
                right,
                (left + right) / 2.0,
                width,
                count,
                fraction,
                fraction / width if total > 0 and width > 0 else math.nan,
                int(raw_values.size),
                int(finite_values.size),
                _fixed_string(metric_spec.bin_policy, 96),
            )
            row_idx += 1
    return rows
