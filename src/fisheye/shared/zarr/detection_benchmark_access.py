"""Consumer-shaped read workloads for canonical sparse detections."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Mapping

import numpy as np

from fisheye.shared.zarr.benchmark_runtime import sha256_array
from fisheye.shared.zarr.canonical_detection_benchmark_input import (
    CanonicalDetectionBenchmarkInput,
)
from fisheye.shared.zarr.detection_schema import CANONICAL_DETECTION_SCHEMA_V1


DETECTION_CONSUMER_READ_SCHEMA_ID = (
    "palette.canonical_detection_consumer_read_workloads"
)
DETECTION_CONSUMER_READ_SCHEMA_VERSION = 1
FRAME_ROW_OFFSETS_PATH = "instances/frame_row_offsets"
INSTANCE_PATHS = tuple(
    path
    for path in CANONICAL_DETECTION_SCHEMA_V1.binding_paths
    if path != FRAME_ROW_OFFSETS_PATH
)


@dataclass(frozen=True)
class DetectionReadWorkloadConfig:
    """Exact selections and pass count shared by every physical candidate."""

    seed: int = 20_260_724
    pass_count: int = 2
    random_frame_count: int = 128
    random_row_range_count: int = 64
    random_row_range_rows: int = 32
    sequential_frame_window: int = 700
    target_frames_per_second: int = 700

    def __post_init__(self) -> None:
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative exact integer.")
        for field in (
            "pass_count",
            "random_frame_count",
            "random_row_range_count",
            "random_row_range_rows",
            "sequential_frame_window",
            "target_frames_per_second",
        ):
            value = getattr(self, field)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{field} must be a positive exact integer.")
        if self.pass_count < 2:
            raise ValueError("Read workloads require a first and warm pass.")

    def as_manifest(self) -> dict[str, int]:
        return {
            "seed": self.seed,
            "pass_count": self.pass_count,
            "random_frame_count": self.random_frame_count,
            "random_row_range_count": self.random_row_range_count,
            "random_row_range_rows": self.random_row_range_rows,
            "sequential_frame_window": self.sequential_frame_window,
            "target_frames_per_second": self.target_frames_per_second,
        }


def _sample_indices(*, population: int, count: int, seed: int) -> np.ndarray:
    selected = min(int(population), int(count))
    if selected <= 0:
        return np.empty((0,), dtype=np.int64)
    return np.random.default_rng(seed).choice(
        int(population),
        size=selected,
        replace=False,
    ).astype(np.int64, copy=False)


def _cache_condition(pass_index: int) -> str:
    return (
        "process_first_pass_os_cache_uncontrolled"
        if pass_index == 0
        else f"same_process_warm_pass_{pass_index}"
    )


def _percentile_seconds(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _pass_result(
    *,
    pass_index: int,
    started: float,
    durations: list[float],
    validation_seconds: float,
    logical_bytes: int,
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    wall_seconds = float(time.perf_counter() - started)
    consumer_seconds = max(0.0, wall_seconds - float(validation_seconds))
    result: dict[str, object] = {
        "pass_index": int(pass_index),
        "cache_condition": _cache_condition(pass_index),
        "read_seconds": float(sum(durations)),
        "consumer_seconds": consumer_seconds,
        "dispatch_seconds": max(0.0, consumer_seconds - float(sum(durations))),
        "validation_seconds": float(validation_seconds),
        "operation_count": len(durations),
        "logical_bytes": int(logical_bytes),
        "p50_operation_seconds": _percentile_seconds(durations, 50),
        "p95_operation_seconds": _percentile_seconds(durations, 95),
        "max_operation_seconds": max(durations, default=0.0),
        "exact": True,
    }
    result.update(dict(extra or {}))
    return result


def _timed_read(array: Any, selection: Any) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    values = np.asarray(array[selection])
    return values, float(time.perf_counter() - started)


def _require_equal(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    label: str,
) -> float:
    started = time.perf_counter()
    exact = bool(np.array_equal(actual, expected))
    seconds = float(time.perf_counter() - started)
    if not exact:
        raise RuntimeError(f"Detection read workload mismatch: {label}.")
    return seconds


def _benchmark_eager_offsets(
    benchmark_input: CanonicalDetectionBenchmarkInput,
    group: Any,
    config: DetectionReadWorkloadConfig,
) -> dict[str, object]:
    expected = benchmark_input.arrays[FRAME_ROW_OFFSETS_PATH]
    passes: list[dict[str, object]] = []
    for pass_index in range(config.pass_count):
        started = time.perf_counter()
        actual, duration = _timed_read(group[FRAME_ROW_OFFSETS_PATH], slice(None))
        validation = _require_equal(
            actual,
            expected,
            label=f"eager offsets pass {pass_index}",
        )
        passes.append(
            _pass_result(
                pass_index=pass_index,
                started=started,
                durations=[duration],
                validation_seconds=validation,
                logical_bytes=int(actual.nbytes),
                extra={"requested_rows": int(actual.shape[0])},
            )
        )
    return {
        "workload_id": "palette.detection_read.eager_frame_row_offsets.v1",
        "passes": passes,
    }


def _benchmark_random_frames(
    benchmark_input: CanonicalDetectionBenchmarkInput,
    group: Any,
    config: DetectionReadWorkloadConfig,
) -> dict[str, object]:
    frames = _sample_indices(
        population=benchmark_input.dimensions.n_frames,
        count=config.random_frame_count,
        seed=config.seed,
    )
    expected_offsets = benchmark_input.arrays[FRAME_ROW_OFFSETS_PATH]
    passes: list[dict[str, object]] = []
    for pass_index in range(config.pass_count):
        started = time.perf_counter()
        durations: list[float] = []
        frame_durations: list[float] = []
        validation_seconds = 0.0
        logical_bytes = 0
        selected_rows = 0
        for frame in frames.tolist():
            frame_duration = 0.0
            actual_offsets, duration = _timed_read(
                group[FRAME_ROW_OFFSETS_PATH],
                slice(frame, frame + 2),
            )
            durations.append(duration)
            frame_duration += duration
            expected_pair = expected_offsets[frame : frame + 2]
            validation_seconds += _require_equal(
                actual_offsets,
                expected_pair,
                label=f"random frame {frame} offsets pass {pass_index}",
            )
            logical_bytes += int(actual_offsets.nbytes)
            start_row, stop_row = map(int, expected_pair)
            selected_rows += stop_row - start_row
            for path in INSTANCE_PATHS:
                selection = (slice(start_row, stop_row),) + (
                    (slice(None),) * (benchmark_input.arrays[path].ndim - 1)
                )
                actual, duration = _timed_read(group[path], selection)
                durations.append(duration)
                frame_duration += duration
                validation_seconds += _require_equal(
                    actual,
                    benchmark_input.arrays[path][selection],
                    label=f"random frame {frame} {path} pass {pass_index}",
                )
                logical_bytes += int(actual.nbytes)
            frame_durations.append(frame_duration)
        passes.append(
            _pass_result(
                pass_index=pass_index,
                started=started,
                durations=durations,
                validation_seconds=validation_seconds,
                logical_bytes=logical_bytes,
                extra={
                    "requested_frames": int(frames.shape[0]),
                    "selected_instance_rows": selected_rows,
                    "p50_frame_seconds": _percentile_seconds(frame_durations, 50),
                    "p95_frame_seconds": _percentile_seconds(frame_durations, 95),
                    "max_frame_seconds": max(frame_durations, default=0.0),
                },
            )
        )
    return {
        "workload_id": "palette.detection_read.random_frame_slices.v1",
        "frame_indices": [int(value) for value in frames],
        "frame_indices_sha256": sha256_array(frames),
        "passes": passes,
    }


def _benchmark_random_row_ranges(
    benchmark_input: CanonicalDetectionBenchmarkInput,
    group: Any,
    config: DetectionReadWorkloadConfig,
) -> dict[str, object]:
    row_count = benchmark_input.dimensions.n_instances
    range_rows = min(config.random_row_range_rows, row_count)
    start_population = row_count - range_rows + 1 if range_rows else 0
    starts = _sample_indices(
        population=start_population,
        count=config.random_row_range_count,
        seed=config.seed + 1,
    )
    passes: list[dict[str, object]] = []
    for pass_index in range(config.pass_count):
        started = time.perf_counter()
        durations: list[float] = []
        range_durations: list[float] = []
        validation_seconds = 0.0
        logical_bytes = 0
        for start_row in starts.tolist():
            stop_row = start_row + range_rows
            range_duration = 0.0
            for path in INSTANCE_PATHS:
                selection = (slice(start_row, stop_row),) + (
                    (slice(None),) * (benchmark_input.arrays[path].ndim - 1)
                )
                actual, duration = _timed_read(group[path], selection)
                durations.append(duration)
                range_duration += duration
                validation_seconds += _require_equal(
                    actual,
                    benchmark_input.arrays[path][selection],
                    label=(
                        f"random rows {start_row}:{stop_row} {path} "
                        f"pass {pass_index}"
                    ),
                )
                logical_bytes += int(actual.nbytes)
            range_durations.append(range_duration)
        passes.append(
            _pass_result(
                pass_index=pass_index,
                started=started,
                durations=durations,
                validation_seconds=validation_seconds,
                logical_bytes=logical_bytes,
                extra={
                    "requested_ranges": int(starts.shape[0]),
                    "rows_per_range": range_rows,
                    "p50_range_seconds": _percentile_seconds(range_durations, 50),
                    "p95_range_seconds": _percentile_seconds(range_durations, 95),
                    "max_range_seconds": max(range_durations, default=0.0),
                },
            )
        )
    return {
        "workload_id": "palette.detection_read.random_observation_ranges.v1",
        "row_starts": [int(value) for value in starts],
        "row_starts_sha256": sha256_array(starts),
        "passes": passes,
    }


def _benchmark_sequential_windows(
    benchmark_input: CanonicalDetectionBenchmarkInput,
    group: Any,
    config: DetectionReadWorkloadConfig,
) -> dict[str, object]:
    n_frames = benchmark_input.dimensions.n_frames
    expected_offsets = benchmark_input.arrays[FRAME_ROW_OFFSETS_PATH]
    window_count = int(math.ceil(n_frames / config.sequential_frame_window))
    passes: list[dict[str, object]] = []
    for pass_index in range(config.pass_count):
        started = time.perf_counter()
        durations: list[float] = []
        window_durations: list[float] = []
        validation_seconds = 0.0
        logical_bytes = 0
        actual_offsets, duration = _timed_read(
            group[FRAME_ROW_OFFSETS_PATH],
            slice(None),
        )
        durations.append(duration)
        validation_seconds += _require_equal(
            actual_offsets,
            expected_offsets,
            label=f"sequential offsets pass {pass_index}",
        )
        logical_bytes += int(actual_offsets.nbytes)
        for start_frame in range(0, n_frames, config.sequential_frame_window):
            stop_frame = min(start_frame + config.sequential_frame_window, n_frames)
            start_row = int(actual_offsets[start_frame])
            stop_row = int(actual_offsets[stop_frame])
            window_duration = 0.0
            for path in INSTANCE_PATHS:
                selection = (slice(start_row, stop_row),) + (
                    (slice(None),) * (benchmark_input.arrays[path].ndim - 1)
                )
                actual, duration = _timed_read(group[path], selection)
                durations.append(duration)
                window_duration += duration
                validation_seconds += _require_equal(
                    actual,
                    benchmark_input.arrays[path][selection],
                    label=(
                        f"sequential frames {start_frame}:{stop_frame} {path} "
                        f"pass {pass_index}"
                    ),
                )
                logical_bytes += int(actual.nbytes)
            window_durations.append(window_duration)
        result = _pass_result(
            pass_index=pass_index,
            started=started,
            durations=durations,
            validation_seconds=validation_seconds,
            logical_bytes=logical_bytes,
            extra={
                "requested_frames": n_frames,
                "frame_window": config.sequential_frame_window,
                "window_count": window_count,
                "p50_window_seconds": _percentile_seconds(window_durations, 50),
                "p95_window_seconds": _percentile_seconds(window_durations, 95),
                "max_window_seconds": max(window_durations, default=0.0),
            },
        )
        consumer_seconds = float(result["consumer_seconds"])
        frames_per_second = (
            float(n_frames / consumer_seconds) if consumer_seconds > 0 else None
        )
        result.update(
            {
                "frames_per_second": frames_per_second,
                "target_frames_per_second": config.target_frames_per_second,
                "meets_target_frames_per_second": (
                    frames_per_second is None
                    or frames_per_second >= config.target_frames_per_second
                ),
            }
        )
        passes.append(result)
    return {
        "workload_id": "palette.detection_read.sequential_frame_windows.v1",
        "passes": passes,
    }


def benchmark_detection_consumer_workloads(
    benchmark_input: CanonicalDetectionBenchmarkInput,
    *,
    group: Any,
    config: DetectionReadWorkloadConfig,
) -> dict[str, object]:
    """Run exact seeded consumer workloads against one already-open group."""

    missing = [
        path
        for path in CANONICAL_DETECTION_SCHEMA_V1.binding_paths
        if path not in group
    ]
    if missing:
        raise ValueError(f"Detection benchmark candidate lacks arrays: {missing!r}")
    return {
        "schema_id": DETECTION_CONSUMER_READ_SCHEMA_ID,
        "schema_version": DETECTION_CONSUMER_READ_SCHEMA_VERSION,
        "logical_schema": {
            "id": CANONICAL_DETECTION_SCHEMA_V1.schema_id,
            "version": CANONICAL_DETECTION_SCHEMA_V1.schema_version,
        },
        "config": config.as_manifest(),
        "instance_paths": list(INSTANCE_PATHS),
        "execution_order": [
            "eager_frame_row_offsets",
            "random_frame_slices",
            "random_observation_ranges",
            "sequential_frame_windows",
        ],
        "eager_frame_row_offsets": _benchmark_eager_offsets(
            benchmark_input,
            group,
            config,
        ),
        "random_frame_slices": _benchmark_random_frames(
            benchmark_input,
            group,
            config,
        ),
        "random_observation_ranges": _benchmark_random_row_ranges(
            benchmark_input,
            group,
            config,
        ),
        "sequential_frame_windows": _benchmark_sequential_windows(
            benchmark_input,
            group,
            config,
        ),
    }


def require_detection_consumer_workloads(value: Mapping[str, object]) -> None:
    """Reject incomplete, inexact, or single-pass composite read evidence."""

    if value.get("schema_id") != DETECTION_CONSUMER_READ_SCHEMA_ID:
        raise ValueError("Unsupported detection consumer-read workload schema.")
    if value.get("schema_version") != DETECTION_CONSUMER_READ_SCHEMA_VERSION:
        raise ValueError("Unsupported detection consumer-read schema version.")
    for field in (
        "eager_frame_row_offsets",
        "random_frame_slices",
        "random_observation_ranges",
        "sequential_frame_windows",
    ):
        workload = value.get(field)
        if not isinstance(workload, Mapping):
            raise ValueError(f"Detection consumer-read evidence lacks {field}.")
        passes = workload.get("passes")
        if not isinstance(passes, list) or len(passes) < 2:
            raise ValueError(f"Detection consumer-read {field} lacks warm passes.")
        if any(
            not isinstance(item, Mapping) or item.get("exact") is not True
            for item in passes
        ):
            raise ValueError(f"Detection consumer-read {field} is not exact.")


__all__ = [
    "DETECTION_CONSUMER_READ_SCHEMA_ID",
    "DETECTION_CONSUMER_READ_SCHEMA_VERSION",
    "DetectionReadWorkloadConfig",
    "benchmark_detection_consumer_workloads",
    "require_detection_consumer_workloads",
]
