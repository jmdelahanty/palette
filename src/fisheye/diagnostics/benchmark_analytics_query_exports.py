"""Benchmark exact immutable analytics query exports in fresh processes.

The controller publishes one manifest-selected Parquet generation through the
real maintained exporter, then launches fresh-process read trials.  It never
changes Zarr selectors, registries, storage profiles, or source metadata.

The benchmark intentionally keeps three kinds of evidence separate:

* publisher-owned, process-local phase telemetry;
* process-tree CPU/RSS telemetry for publication and each read trial; and
* Parquet access workloads and filesystem allocation statistics.

Filesystem reads do not provide compressed network-transfer telemetry.  The
result therefore records those fields as unavailable instead of inventing
them.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time
from typing import Any

import pyarrow.dataset as ds
import pyarrow.parquet as pq

from fisheye.analytics_exports.activity_spatial_time_bins import (
    export_activity_spatial_time_bins,
)
from fisheye.analytics_exports.eye_trace_samples import export_eye_trace_samples
from fisheye.analytics_exports.kinematics_samples import export_kinematics_samples
from fisheye.analytics_exports.publication import (
    export_manifest_path,
    safe_component,
    sha256_file,
)
from fisheye.analytics_exports.runtime_telemetry import (
    validate_export_runtime_telemetry,
)
from fisheye.analytics_exports.tail_trace_samples import export_tail_trace_samples
from fisheye.analytics_exports.validation import validate_export_run
from fisheye.diagnostics.run_with_resource_telemetry import (
    run_with_resource_telemetry,
)
from fisheye.shared.system_metadata import get_git_info
from fisheye.shared.zarr.benchmark_runtime import peak_rss_bytes, utc_now
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


BENCHMARK_ID = "analytics_query_export_benchmark_v1"
REQUEST_SCHEMA_ID = "palette.analytics_query_export_benchmark.request"
REQUEST_SCHEMA_VERSION = 1
PUBLICATION_RESULT_SCHEMA_ID = (
    "palette.analytics_query_export_benchmark.publication_result"
)
PUBLICATION_RESULT_SCHEMA_VERSION = 1
READ_TRIAL_SCHEMA_ID = "palette.analytics_query_export_benchmark.read_trial"
READ_TRIAL_SCHEMA_VERSION = 1
MATRIX_RESULT_SCHEMA_ID = "palette.analytics_query_export_benchmark.matrix_result"
MATRIX_RESULT_SCHEMA_VERSION = 1

DEFAULT_SEED = 17
DEFAULT_REPETITIONS = 5
DEFAULT_RANDOM_FRAME_READS = 32
DEFAULT_WINDOW_COUNT = 8
DEFAULT_WINDOW_FRAMES = 4_096
_SCALE_IDS = frozenset({"representative_short", "full_duration"})
_TRACK_SCOPES = frozenset({"online", "offline"})


@dataclass(frozen=True)
class _Family:
    family_id: str
    table_name: str
    source_fields: tuple[str, ...]
    publisher_fields: tuple[str, ...]
    axis_column: str
    hot_columns: tuple[str, ...]
    publisher: Callable[..., dict[str, Any]]


_COMMON_HOT = (
    "recording_id",
    "source_acquisition_frame_index",
)

_FAMILIES = {
    "eye_trace_samples": _Family(
        family_id="eye_trace_samples",
        table_name="eye_trace_samples",
        source_fields=("eye_angle_run",),
        publisher_fields=("row_group_rows",),
        axis_column="source_acquisition_frame_index",
        hot_columns=(
            *_COMMON_HOT,
            "left_eye_angle_deg",
            "right_eye_angle_deg",
            "vergence_eye_angle_deg",
            "valid_frame",
        ),
        publisher=export_eye_trace_samples,
    ),
    "kinematics_samples": _Family(
        family_id="kinematics_samples",
        table_name="kinematics_samples",
        source_fields=("track_kinematics_run", "track_scope"),
        publisher_fields=(
            "requested_sample_rate_hz",
            "source_window_rows",
            "row_group_rows",
        ),
        axis_column="source_acquisition_frame_index",
        hot_columns=(
            *_COMMON_HOT,
            "track_id",
            "position_x_mm",
            "position_y_mm",
            "speed_mm_s",
            "motion_heading_degrees",
            "sample_valid",
        ),
        publisher=export_kinematics_samples,
    ),
    "activity_spatial_time_bins": _Family(
        family_id="activity_spatial_time_bins",
        table_name="activity_spatial_time_bins",
        source_fields=(
            "track_kinematics_run",
            "track_scope",
            "swim_bout_runs_by_track",
        ),
        publisher_fields=("requested_bin_size_s", "row_group_rows"),
        axis_column="start_acquisition_frame_index",
        hot_columns=(
            "recording_id",
            "track_id",
            "time_bin_index",
            "start_acquisition_frame_index",
            "end_acquisition_frame_index_exclusive",
            "mean_position_x_mm",
            "mean_position_y_mm",
            "mean_speed_mm_s",
            "bout_count_started",
            "bin_valid",
        ),
        publisher=export_activity_spatial_time_bins,
    ),
    "tail_trace_samples": _Family(
        family_id="tail_trace_samples",
        table_name="tail_trace_samples",
        source_fields=(
            "tail_kinematics_run",
            "subject_shape_run",
            "track_kinematics_run",
            "track_scope",
        ),
        publisher_fields=(
            "source_window_rows",
            "source_rows_per_part",
            "row_group_rows",
        ),
        axis_column="source_acquisition_frame_index",
        hot_columns=(
            *_COMMON_HOT,
            "instance_key",
            "tail_sample_index",
            "normalized_tail_position",
            "body_longitudinal_fraction",
            "body_lateral_fraction",
            "tangent_angle_rad",
            "sample_valid",
        ),
        publisher=export_tail_trace_samples,
    ),
}


def _strict_envelope(
    schema_id: str,
    schema_version: int,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    body = dict(payload)
    json.dumps(body, allow_nan=False)
    return {
        "schema_id": schema_id,
        "schema_version": schema_version,
        "payload": body,
        "payload_digest": canonical_json_sha256(body),
    }


def _require_envelope(
    value: Mapping[str, Any],
    *,
    schema_id: str,
    schema_version: int,
) -> Mapping[str, Any]:
    if set(value) != {
        "schema_id",
        "schema_version",
        "payload",
        "payload_digest",
    }:
        raise ValueError("Benchmark envelope has an unexpected field set.")
    if (
        value.get("schema_id") != schema_id
        or value.get("schema_version") != schema_version
    ):
        raise ValueError("Benchmark envelope schema identity is unsupported.")
    payload = value.get("payload")
    if not isinstance(payload, Mapping):
        raise ValueError("Benchmark envelope payload must be one object.")
    if value.get("payload_digest") != canonical_json_sha256(payload):
        raise ValueError("Benchmark envelope payload digest mismatch.")
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Benchmark evidence is not strict JSON: {exc}") from exc
    return payload


def _family(family_id: object) -> _Family:
    try:
        return _FAMILIES[str(family_id)]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported query-export family {family_id!r}; "
            f"expected {sorted(_FAMILIES)!r}."
        ) from exc


def _safe_positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive exact integer.")
    return value


def _safe_positive_float(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a positive finite number.")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{label} must be a positive finite number.")
    return result


def _safe_existing_archive(value: object) -> Path:
    archive = Path(str(value)).expanduser().resolve()
    if not archive.is_dir() or not (archive / "zarr.json").is_file():
        raise FileNotFoundError(f"Analysis Zarr archive not found: {archive}")
    return archive


def _benchmark_path(value: object, *, label: str) -> Path:
    path = Path(str(value)).expanduser().resolve()
    if not any("benchmark" in component.lower() for component in path.parts):
        raise ValueError(f"{label} must be explicitly benchmark-namespaced.")
    if path in {Path("/"), Path.home().resolve()}:
        raise ValueError(f"{label} is too broad.")
    return path


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def build_request(
    *,
    family_id: str,
    scale_id: str,
    zarr_path: str | Path,
    export_root: str | Path,
    scratch_root: str | Path,
    benchmark_output_dir: str | Path,
    export_run_id: str,
    source_runs: Mapping[str, Any],
    publisher_parameters: Mapping[str, Any],
    seed: int = DEFAULT_SEED,
    repetitions: int = DEFAULT_REPETITIONS,
    random_frame_reads: int = DEFAULT_RANDOM_FRAME_READS,
    window_count: int = DEFAULT_WINDOW_COUNT,
    window_frames: int = DEFAULT_WINDOW_FRAMES,
    requested_workers: int = 1,
    allocated_slots: int = 1,
    sample_interval_seconds: float = 0.25,
    cache_state: str = "uncontrolled_fresh_process",
) -> dict[str, Any]:
    """Build and validate one closed benchmark request."""

    payload = {
        "benchmark_id": BENCHMARK_ID,
        "family_id": family_id,
        "scale_id": scale_id,
        "zarr_path": str(Path(zarr_path).expanduser().resolve()),
        "export_root": str(Path(export_root).expanduser().resolve()),
        "scratch_root": str(Path(scratch_root).expanduser().resolve()),
        "benchmark_output_dir": str(Path(benchmark_output_dir).expanduser().resolve()),
        "export_run_id": export_run_id,
        "source_runs": dict(source_runs),
        "publisher_parameters": dict(publisher_parameters),
        "workload": {
            "seed": seed,
            "repetitions": repetitions,
            "random_frame_reads": random_frame_reads,
            "window_count": window_count,
            "window_frames": window_frames,
        },
        "resources": {
            "requested_workers": requested_workers,
            "allocated_slots": allocated_slots,
            "sample_interval_seconds": sample_interval_seconds,
            "cache_state": cache_state,
        },
    }
    request = _strict_envelope(REQUEST_SCHEMA_ID, REQUEST_SCHEMA_VERSION, payload)
    require_request(request, require_paths=False)
    return request


def require_request(
    request: Mapping[str, Any],
    *,
    require_paths: bool = True,
) -> Mapping[str, Any]:
    payload = _require_envelope(
        request,
        schema_id=REQUEST_SCHEMA_ID,
        schema_version=REQUEST_SCHEMA_VERSION,
    )
    expected = {
        "benchmark_id",
        "family_id",
        "scale_id",
        "zarr_path",
        "export_root",
        "scratch_root",
        "benchmark_output_dir",
        "export_run_id",
        "source_runs",
        "publisher_parameters",
        "workload",
        "resources",
    }
    if set(payload) != expected or payload.get("benchmark_id") != BENCHMARK_ID:
        raise ValueError("Query-export benchmark request fields or ID are invalid.")
    family = _family(payload["family_id"])
    if payload.get("scale_id") not in _SCALE_IDS:
        raise ValueError(
            "Benchmark scale must be representative_short or full_duration."
        )
    safe_component(payload["export_run_id"], label="export run ID")
    source_runs = payload.get("source_runs")
    parameters = payload.get("publisher_parameters")
    if not isinstance(source_runs, Mapping) or set(source_runs) != set(
        family.source_fields
    ):
        raise ValueError(
            "Benchmark source-run declaration is not exact for its family."
        )
    if not isinstance(parameters, Mapping) or set(parameters) != set(
        family.publisher_fields
    ):
        raise ValueError("Benchmark publisher parameters are not exact for its family.")
    for name, value in source_runs.items():
        if name == "track_scope":
            if value not in _TRACK_SCOPES:
                raise ValueError("track_scope must be online or offline.")
        elif name == "swim_bout_runs_by_track":
            if not isinstance(value, Mapping) or not value:
                raise ValueError("swim_bout_runs_by_track must be a nonempty object.")
            for raw_track_id, run_name in value.items():
                if not str(raw_track_id).isdigit() or str(int(raw_track_id)) != str(
                    raw_track_id
                ):
                    raise ValueError("Swim-bout track IDs must be canonical integers.")
                safe_component(run_name, label="swim-bout run ID")
        else:
            safe_component(value, label=name.replace("_", " "))
    for name, value in parameters.items():
        if name in {
            "row_group_rows",
            "source_window_rows",
            "source_rows_per_part",
        }:
            _safe_positive_int(value, label=name)
        else:
            _safe_positive_float(value, label=name)
    workload = payload.get("workload")
    if not isinstance(workload, Mapping) or set(workload) != {
        "seed",
        "repetitions",
        "random_frame_reads",
        "window_count",
        "window_frames",
    }:
        raise ValueError("Benchmark workload declaration is not closed.")
    if isinstance(workload["seed"], bool) or type(workload["seed"]) is not int:
        raise ValueError("Benchmark seed must be an exact integer.")
    for name in (
        "repetitions",
        "random_frame_reads",
        "window_count",
        "window_frames",
    ):
        _safe_positive_int(workload[name], label=name)
    resources = payload.get("resources")
    if not isinstance(resources, Mapping) or set(resources) != {
        "requested_workers",
        "allocated_slots",
        "sample_interval_seconds",
        "cache_state",
    }:
        raise ValueError("Benchmark resource declaration is not closed.")
    _safe_positive_int(resources["requested_workers"], label="requested_workers")
    _safe_positive_int(resources["allocated_slots"], label="allocated_slots")
    _safe_positive_float(
        resources["sample_interval_seconds"], label="sample_interval_seconds"
    )
    if (
        not isinstance(resources["cache_state"], str)
        or not resources["cache_state"].strip()
    ):
        raise ValueError("Benchmark cache_state must be explicit.")
    if require_paths:
        archive = _safe_existing_archive(payload["zarr_path"])
    else:
        archive = Path(str(payload["zarr_path"])).expanduser().resolve()
    export_root = _benchmark_path(payload["export_root"], label="export_root")
    scratch_root = _benchmark_path(payload["scratch_root"], label="scratch_root")
    output = _benchmark_path(
        payload["benchmark_output_dir"], label="benchmark_output_dir"
    )
    if any(
        _paths_overlap(left, right)
        for left, right in (
            (archive, export_root),
            (archive, scratch_root),
            (archive, output),
            (export_root, scratch_root),
            (output, scratch_root),
        )
    ):
        raise ValueError(
            "Source, export, scratch, and evidence paths must not overlap."
        )
    if (
        require_paths
        and export_manifest_path(export_root, str(payload["export_run_id"])).exists()
    ):
        raise FileExistsError("Benchmark export manifest already exists.")
    return payload


def _write_strict_json(path: Path, value: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    if path.exists():
        raise FileExistsError(f"Refusing to replace immutable evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"Temporary evidence path already exists: {temporary}")
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    )
    temporary.write_text(encoded + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _read_strict_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda raw: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON token {raw}")
        ),
    )
    if not isinstance(value, Mapping):
        raise ValueError(f"Strict JSON document is not one object: {path}")
    return value


def _publisher_kwargs(payload: Mapping[str, Any]) -> dict[str, Any]:
    family = _family(payload["family_id"])
    kwargs = {
        **dict(payload["source_runs"]),
        **dict(payload["publisher_parameters"]),
        "output_root": payload["export_root"],
        "export_run_id": payload["export_run_id"],
        "scratch_root": payload["scratch_root"],
        "overwrite": False,
    }
    if family.family_id == "activity_spatial_time_bins":
        kwargs["swim_bout_runs_by_track"] = {
            int(track_id): run_name
            for track_id, run_name in kwargs["swim_bout_runs_by_track"].items()
        }
    return kwargs


def run_publication(request: Mapping[str, Any]) -> dict[str, Any]:
    payload = require_request(request)
    family = _family(payload["family_id"])
    started_at = utc_now()
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    result = family.publisher(payload["zarr_path"], **_publisher_kwargs(payload))
    telemetry = result.get("runtime_telemetry")
    if not isinstance(telemetry, Mapping):
        raise ValueError("Maintained exporter omitted process-local runtime telemetry.")
    validate_export_runtime_telemetry(telemetry)
    manifest_path = Path(str(result["manifest_path"])).resolve()
    expected_manifest = export_manifest_path(
        Path(str(payload["export_root"])), str(payload["export_run_id"])
    )
    if manifest_path != expected_manifest or not manifest_path.is_file():
        raise ValueError("Exporter committed an unexpected manifest path.")
    manifest = _read_strict_json(manifest_path)
    if "runtime_telemetry" in manifest:
        raise ValueError("Runtime telemetry leaked into the immutable manifest.")
    publication = manifest.get("publication")
    if not isinstance(publication, Mapping):
        raise ValueError("Published analytics manifest lacks publication metadata.")
    parts = publication.get("parts_by_table")
    if not isinstance(parts, Mapping) or set(parts) != {family.table_name}:
        raise ValueError("Published analytics manifest has an unexpected table set.")
    validation_key = {
        "eye_trace_samples": "eye_trace_validation",
        "kinematics_samples": "kinematics_samples_validation",
        "activity_spatial_time_bins": "activity_spatial_time_bins_validation",
        "tail_trace_samples": "tail_trace_validation",
    }[family.family_id]
    validation = result.get(validation_key)
    if not isinstance(validation, Mapping) or validation.get("valid") is not True:
        raise ValueError("Maintained exporter did not return exact final validation.")
    body = {
        "benchmark_id": BENCHMARK_ID,
        "family_id": family.family_id,
        "request_payload_digest": request["payload_digest"],
        "started_at_utc": started_at,
        "finished_at_utc": utc_now(),
        "manifest_path": str(manifest_path),
        "manifest_file_sha256": sha256_file(manifest_path),
        "manifest_payload_sha256": canonical_json_sha256(manifest),
        "row_count": int(manifest["row_counts_by_table"][family.table_name]),
        "part_count": len(parts[family.table_name]),
        "exact_export_validation": dict(validation),
        "publisher_runtime_telemetry": dict(telemetry),
        "process_runtime": {
            "wall_seconds": float(time.perf_counter() - wall_started),
            "cpu_seconds": float(time.process_time() - cpu_started),
            "peak_rss_bytes": peak_rss_bytes(),
        },
    }
    result_envelope = _strict_envelope(
        PUBLICATION_RESULT_SCHEMA_ID,
        PUBLICATION_RESULT_SCHEMA_VERSION,
        body,
    )
    require_publication_result(result_envelope, request=request)
    return result_envelope


def require_publication_result(
    value: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
) -> Mapping[str, Any]:
    request_payload = require_request(request, require_paths=False)
    payload = _require_envelope(
        value,
        schema_id=PUBLICATION_RESULT_SCHEMA_ID,
        schema_version=PUBLICATION_RESULT_SCHEMA_VERSION,
    )
    expected = {
        "benchmark_id",
        "family_id",
        "request_payload_digest",
        "started_at_utc",
        "finished_at_utc",
        "manifest_path",
        "manifest_file_sha256",
        "manifest_payload_sha256",
        "row_count",
        "part_count",
        "exact_export_validation",
        "publisher_runtime_telemetry",
        "process_runtime",
    }
    if set(payload) != expected:
        raise ValueError("Publication result has an unexpected field set.")
    if (
        payload["benchmark_id"] != BENCHMARK_ID
        or payload["family_id"] != request_payload["family_id"]
        or payload["request_payload_digest"] != request["payload_digest"]
    ):
        raise ValueError("Publication result/request identity mismatch.")
    if payload["exact_export_validation"].get("valid") is not True:
        raise ValueError("Publication result exact validation did not pass.")
    validate_export_runtime_telemetry(payload["publisher_runtime_telemetry"])
    _safe_positive_int(payload["part_count"], label="publication part_count")
    if type(payload["row_count"]) is not int or payload["row_count"] < 0:
        raise ValueError("Publication row_count is invalid.")
    return payload


def _part_paths(
    *,
    export_root: Path,
    manifest: Mapping[str, Any],
    table_name: str,
) -> tuple[Path, ...]:
    publication = manifest.get("publication")
    if not isinstance(publication, Mapping):
        raise ValueError("Export manifest lacks publication metadata.")
    parts = publication.get("parts_by_table")
    if not isinstance(parts, Mapping) or set(parts) != {table_name}:
        raise ValueError(
            "Export manifest table inventory differs from benchmark family."
        )
    records = parts[table_name]
    if not isinstance(records, list) or not records:
        raise ValueError("Export manifest has no selected Parquet parts.")
    paths: list[Path] = []
    for record in records:
        if not isinstance(record, Mapping) or set(record) != {
            "path",
            "sha256",
            "size_bytes",
            "row_count",
        }:
            raise ValueError("Export manifest part record is not closed.")
        path = (export_root / str(record["path"])).resolve()
        if (
            not path.is_relative_to(export_root)
            or path.is_symlink()
            or not path.is_file()
        ):
            raise ValueError("Manifest-selected Parquet part is unsafe or missing.")
        paths.append(path)
    if len(paths) != len(set(paths)):
        raise ValueError("Export manifest repeats a Parquet part.")
    return tuple(paths)


def _measure(call: Callable[[], Any]) -> tuple[Any, dict[str, float]]:
    wall = time.perf_counter()
    cpu = time.process_time()
    value = call()
    return value, {
        "wall_seconds": float(time.perf_counter() - wall),
        "cpu_seconds": float(time.process_time() - cpu),
    }


def _axis_extent(parts: Sequence[Path], axis_column: str) -> tuple[int, int] | None:
    minimum: int | None = None
    maximum: int | None = None
    for path in parts:
        parquet = pq.ParquetFile(path)
        try:
            column_index = parquet.schema_arrow.names.index(axis_column)
        except ValueError as exc:
            raise ValueError(
                f"Parquet part lacks benchmark axis {axis_column!r}: {path}"
            ) from exc
        for row_group_index in range(parquet.metadata.num_row_groups):
            statistics_value = (
                parquet.metadata.row_group(row_group_index)
                .column(column_index)
                .statistics
            )
            if statistics_value is None or not statistics_value.has_min_max:
                raise ValueError(
                    f"Parquet benchmark axis lacks row-group min/max statistics: {path}"
                )
            local_min = int(statistics_value.min)
            local_max = int(statistics_value.max)
            minimum = local_min if minimum is None else min(minimum, local_min)
            maximum = local_max if maximum is None else max(maximum, local_max)
    return None if minimum is None or maximum is None else (minimum, maximum)


def _distribution(samples: Sequence[float]) -> dict[str, Any]:
    if not samples:
        return {
            "count": 0,
            "samples_seconds": [],
            "median_seconds": None,
            "p95_seconds": None,
        }
    ordered = sorted(float(value) for value in samples)
    p95_index = min(len(ordered) - 1, max(0, math.ceil(0.95 * len(ordered)) - 1))
    return {
        "count": len(ordered),
        "samples_seconds": [float(value) for value in samples],
        "median_seconds": float(statistics.median(ordered)),
        "p95_seconds": float(ordered[p95_index]),
    }


def _consume_table(table: Any) -> dict[str, int]:
    return {"rows": int(table.num_rows), "decoded_bytes": int(table.nbytes)}


def _read_workloads(
    *,
    parts: Sequence[Path],
    family: _Family,
    seed: int,
    random_frame_reads: int,
    window_count: int,
    window_frames: int,
) -> dict[str, Any]:
    footer_values, footer_timing = _measure(
        lambda: [
            {
                "path": str(path),
                "row_groups": pq.ParquetFile(path).metadata.num_row_groups,
                "rows": pq.ParquetFile(path).metadata.num_rows,
            }
            for path in parts
        ]
    )
    dataset = ds.dataset([str(path) for path in parts], format="parquet")
    if set(family.hot_columns) - set(dataset.schema.names):
        raise ValueError("Query-export benchmark hot-column contract is unavailable.")
    extent = _axis_extent(parts, family.axis_column)
    random_timings: list[float] = []
    random_rows = 0
    random_bytes = 0
    window_timings: list[float] = []
    window_rows = 0
    window_bytes = 0
    random_frames: list[int] = []
    windows: list[list[int]] = []
    if extent is not None:
        first, last = extent
        rng = random.Random(seed)
        random_frames = [rng.randint(first, last) for _ in range(random_frame_reads)]
        max_start = max(first, last - window_frames + 1)
        windows = [
            [
                first + ((index + 1) * max(0, max_start - first)) // (window_count + 1),
                0,
            ]
            for index in range(window_count)
        ]
        for window in windows:
            window[1] = min(last + 1, window[0] + window_frames)
        for frame in random_frames:
            started = time.perf_counter()
            table = dataset.to_table(
                columns=list(family.hot_columns),
                filter=ds.field(family.axis_column) == frame,
            )
            random_timings.append(float(time.perf_counter() - started))
            consumed = _consume_table(table)
            random_rows += consumed["rows"]
            random_bytes += consumed["decoded_bytes"]
        for start, stop in windows:
            started = time.perf_counter()
            table = dataset.to_table(
                columns=list(family.hot_columns),
                filter=(ds.field(family.axis_column) >= start)
                & (ds.field(family.axis_column) < stop),
            )
            window_timings.append(float(time.perf_counter() - started))
            consumed = _consume_table(table)
            window_rows += consumed["rows"]
            window_bytes += consumed["decoded_bytes"]

    digest = hashlib.sha256()
    full_rows = 0
    full_decoded_bytes = 0

    def full_scan() -> None:
        nonlocal full_rows, full_decoded_bytes
        for batch in dataset.to_batches(batch_size=65_536):
            digest.update(batch.serialize().to_pybytes())
            full_rows += int(batch.num_rows)
            full_decoded_bytes += int(batch.nbytes)

    _, full_timing = _measure(full_scan)
    return {
        "footer_open": {
            "parts": footer_values,
            **footer_timing,
        },
        "axis": {
            "column": family.axis_column,
            "minimum": None if extent is None else extent[0],
            "maximum": None if extent is None else extent[1],
            "statistics_source": "parquet_row_group_min_max",
        },
        "random_frame_hot_columns": {
            "frames": random_frames,
            "columns": list(family.hot_columns),
            "rows": random_rows,
            "decoded_bytes": random_bytes,
            "latency": _distribution(random_timings),
        },
        "windowed_frame_hot_columns": {
            "windows": windows,
            "columns": list(family.hot_columns),
            "rows": window_rows,
            "decoded_bytes": window_bytes,
            "latency": _distribution(window_timings),
        },
        "full_scan": {
            "rows": full_rows,
            "decoded_bytes": full_decoded_bytes,
            "logical_stream_sha256": digest.hexdigest(),
            **full_timing,
            "rows_per_second": (
                float(full_rows / full_timing["wall_seconds"])
                if full_timing["wall_seconds"] > 0
                else None
            ),
            "decoded_bytes_per_second": (
                float(full_decoded_bytes / full_timing["wall_seconds"])
                if full_timing["wall_seconds"] > 0
                else None
            ),
        },
    }


def _storage_stats(paths: Sequence[Path], manifest_path: Path) -> dict[str, int]:
    files = [manifest_path, *paths]
    apparent = 0
    allocated = 0
    for path in files:
        stat = path.stat()
        apparent += int(stat.st_size)
        allocated += int(stat.st_blocks * 512)
    return {
        "object_count": len(files),
        "manifest_object_count": 1,
        "parquet_object_count": len(paths),
        "apparent_bytes": apparent,
        "allocated_bytes": allocated,
    }


def run_read_trial(
    request: Mapping[str, Any],
    *,
    repetition_index: int,
) -> dict[str, Any]:
    payload = require_request(request, require_paths=False)
    family = _family(payload["family_id"])
    workload = payload["workload"]
    if (
        isinstance(repetition_index, bool)
        or type(repetition_index) is not int
        or not 0 <= repetition_index < workload["repetitions"]
    ):
        raise ValueError("Read-trial repetition index is invalid.")
    export_root = Path(str(payload["export_root"])).resolve()
    manifest_path = export_manifest_path(export_root, str(payload["export_run_id"]))
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Published export manifest not found: {manifest_path}")
    manifest = _read_strict_json(manifest_path)
    parts = _part_paths(
        export_root=export_root,
        manifest=manifest,
        table_name=family.table_name,
    )
    started_at = utc_now()
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    validation, validation_timing = _measure(
        lambda: validate_export_run(
            export_root,
            str(payload["export_run_id"]),
            allow_legacy_layout=False,
        )
    )
    workloads = _read_workloads(
        parts=parts,
        family=family,
        seed=int(workload["seed"]) + repetition_index,
        random_frame_reads=int(workload["random_frame_reads"]),
        window_count=int(workload["window_count"]),
        window_frames=int(workload["window_frames"]),
    )
    row_count = int(manifest["row_counts_by_table"][family.table_name])
    if validation.get("status") != "valid" or int(validation["row_count"]) != row_count:
        raise ValueError("Independent validation result differs from the manifest.")
    if int(workloads["full_scan"]["rows"]) != row_count:
        raise ValueError("Full-scan row count differs from the immutable manifest.")
    body = {
        "benchmark_id": BENCHMARK_ID,
        "family_id": family.family_id,
        "request_payload_digest": request["payload_digest"],
        "repetition_index": repetition_index,
        "cache_state": payload["resources"]["cache_state"],
        "started_at_utc": started_at,
        "finished_at_utc": utc_now(),
        "manifest_path": str(manifest_path),
        "manifest_file_sha256": sha256_file(manifest_path),
        "manifest_payload_sha256": canonical_json_sha256(manifest),
        "validation": {"result": validation, **validation_timing},
        "workloads": workloads,
        "storage": _storage_stats(parts, manifest_path),
        "process_runtime": {
            "wall_seconds": float(time.perf_counter() - wall_started),
            "cpu_seconds": float(time.process_time() - cpu_started),
            "peak_rss_bytes": peak_rss_bytes(),
        },
        "physical_io": {
            "request_count": None,
            "transferred_bytes": None,
            "availability": (
                "unavailable_from_process_local_parquet_reader; use the Linux "
                "file-I/O tracer or Crimson mounted-reader telemetry separately"
            ),
        },
    }
    result = _strict_envelope(
        READ_TRIAL_SCHEMA_ID,
        READ_TRIAL_SCHEMA_VERSION,
        body,
    )
    require_read_trial(result, request=request)
    return result


def require_read_trial(
    value: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
) -> Mapping[str, Any]:
    request_payload = require_request(request, require_paths=False)
    payload = _require_envelope(
        value,
        schema_id=READ_TRIAL_SCHEMA_ID,
        schema_version=READ_TRIAL_SCHEMA_VERSION,
    )
    expected = {
        "benchmark_id",
        "family_id",
        "request_payload_digest",
        "repetition_index",
        "cache_state",
        "started_at_utc",
        "finished_at_utc",
        "manifest_path",
        "manifest_file_sha256",
        "manifest_payload_sha256",
        "validation",
        "workloads",
        "storage",
        "process_runtime",
        "physical_io",
    }
    if set(payload) != expected:
        raise ValueError("Read-trial result has an unexpected field set.")
    if (
        payload["benchmark_id"] != BENCHMARK_ID
        or payload["family_id"] != request_payload["family_id"]
        or payload["request_payload_digest"] != request["payload_digest"]
        or payload["cache_state"] != request_payload["resources"]["cache_state"]
    ):
        raise ValueError("Read-trial result/request identity mismatch.")
    validation = payload["validation"]
    if (
        not isinstance(validation, Mapping)
        or validation.get("result", {}).get("status") != "valid"
    ):
        raise ValueError("Read-trial exact validation did not pass.")
    physical = payload["physical_io"]
    if (
        not isinstance(physical, Mapping)
        or physical.get("request_count") is not None
        or physical.get("transferred_bytes") is not None
    ):
        raise ValueError("Read trial fabricated unavailable physical I/O telemetry.")
    return payload


def _source_metadata_paths(payload: Mapping[str, Any]) -> tuple[Path, ...]:
    archive = Path(str(payload["zarr_path"])).resolve()
    source = payload["source_runs"]
    family = str(payload["family_id"])
    roots: list[Path] = [archive / "zarr.json"]

    def add_run(parent: str, run_name: str, *, scope: str | None = None) -> None:
        parent_root = archive / "analysis" / parent
        roots.append(parent_root / "zarr.json")
        if scope is not None:
            roots.append(parent_root / scope / "zarr.json")
            run_root = parent_root / scope / run_name
        else:
            run_root = parent_root / run_name
        if not run_root.is_dir():
            raise FileNotFoundError(f"Benchmark source run does not exist: {run_root}")
        roots.extend(sorted(run_root.rglob("zarr.json")))

    if family == "eye_trace_samples":
        add_run("eye_angle_runs", str(source["eye_angle_run"]))
    elif family == "kinematics_samples":
        add_run(
            "track_kinematics_runs",
            str(source["track_kinematics_run"]),
            scope=str(source["track_scope"]),
        )
    elif family == "activity_spatial_time_bins":
        add_run(
            "track_kinematics_runs",
            str(source["track_kinematics_run"]),
            scope=str(source["track_scope"]),
        )
        for run_name in source["swim_bout_runs_by_track"].values():
            add_run("swim_bout_runs", str(run_name))
    elif family == "tail_trace_samples":
        add_run("tail_kinematics_runs", str(source["tail_kinematics_run"]))
        add_run("subject_shape_runs", str(source["subject_shape_run"]))
        add_run(
            "track_kinematics_runs",
            str(source["track_kinematics_run"]),
            scope=str(source["track_scope"]),
        )
    else:  # pragma: no cover - family was validated above
        raise ValueError(f"Unsupported family {family!r}")
    unique = sorted({path.resolve() for path in roots})
    for path in unique:
        if not path.is_file() or not path.is_relative_to(archive):
            raise FileNotFoundError(f"Source metadata guard path is missing: {path}")
    return tuple(unique)


def _metadata_guard(payload: Mapping[str, Any]) -> dict[str, str]:
    archive = Path(str(payload["zarr_path"])).resolve()
    return {
        str(path.relative_to(archive)): sha256_file(path)
        for path in _source_metadata_paths(payload)
    }


def _run_resource_command(
    command: Sequence[str],
    *,
    root: Path,
    stem: str,
    resources: Mapping[str, Any],
) -> dict[str, object]:
    return run_with_resource_telemetry(
        command,
        summary_json=root / f"{stem}.resources.json",
        samples_jsonl=root / f"{stem}.resources.jsonl",
        stdout_log=root / f"{stem}.stdout.log",
        requested_workers=int(resources["requested_workers"]),
        allocated_slots=int(resources["allocated_slots"]),
        sample_interval_seconds=float(resources["sample_interval_seconds"]),
    )


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(float(value) for value in values))


def run_matrix(request: Mapping[str, Any]) -> dict[str, Any]:
    payload = require_request(request)
    output = Path(str(payload["benchmark_output_dir"])).resolve()
    if output.exists():
        raise FileExistsError(f"Benchmark evidence directory already exists: {output}")
    output.mkdir(parents=True, exist_ok=False)
    request_copy = output / "request.json"
    _write_strict_json(request_copy, request)
    guard_before = _metadata_guard(payload)
    started_at = utc_now()
    resources = payload["resources"]
    publication_result_path = output / "publication_result.json"
    publication_command = [
        sys.executable,
        "-m",
        "fisheye.diagnostics.benchmark_analytics_query_exports",
        "publish",
        "--request",
        str(request_copy),
        "--output",
        str(publication_result_path),
    ]
    publication_resources = _run_resource_command(
        publication_command,
        root=output,
        stem="publication",
        resources=resources,
    )
    if publication_resources["status"] != "ok":
        raise RuntimeError("Fresh-process analytics export publication failed.")
    publication_result = _read_strict_json(publication_result_path)
    publication = require_publication_result(publication_result, request=request)

    read_trials: list[Mapping[str, Any]] = []
    read_resources: list[dict[str, object]] = []
    trial_files: list[str] = []
    for repetition in range(int(payload["workload"]["repetitions"])):
        trial_path = output / f"read_trial_{repetition:02d}.json"
        command = [
            sys.executable,
            "-m",
            "fisheye.diagnostics.benchmark_analytics_query_exports",
            "read",
            "--request",
            str(request_copy),
            "--repetition-index",
            str(repetition),
            "--output",
            str(trial_path),
        ]
        resource = _run_resource_command(
            command,
            root=output,
            stem=f"read_trial_{repetition:02d}",
            resources=resources,
        )
        if resource["status"] != "ok":
            raise RuntimeError(f"Fresh-process read trial {repetition} failed.")
        trial = _read_strict_json(trial_path)
        require_read_trial(trial, request=request)
        read_trials.append(trial)
        read_resources.append(resource)
        trial_files.append(trial_path.name)
    guard_after = _metadata_guard(payload)
    if guard_after != guard_before:
        raise RuntimeError("Source Zarr metadata changed during the benchmark.")
    trial_payloads = [trial["payload"] for trial in read_trials]
    full_scan_seconds = [
        float(trial["workloads"]["full_scan"]["wall_seconds"])
        for trial in trial_payloads
    ]
    random_p95 = [
        float(trial["workloads"]["random_frame_hot_columns"]["latency"]["p95_seconds"])
        for trial in trial_payloads
    ]
    window_p95 = [
        float(
            trial["workloads"]["windowed_frame_hot_columns"]["latency"]["p95_seconds"]
        )
        for trial in trial_payloads
    ]
    validation_seconds = [
        float(trial["validation"]["wall_seconds"]) for trial in trial_payloads
    ]
    full_digests = {
        str(trial["workloads"]["full_scan"]["logical_stream_sha256"])
        for trial in trial_payloads
    }
    manifest_digests = {str(trial["manifest_file_sha256"]) for trial in trial_payloads}
    if len(full_digests) != 1 or len(manifest_digests) != 1:
        raise RuntimeError("Fresh-process read trials disagreed on immutable bytes.")
    git = get_git_info(Path(__file__).resolve().parents[3])
    body = {
        "benchmark_id": BENCHMARK_ID,
        "family_id": payload["family_id"],
        "scale_id": payload["scale_id"],
        "request_payload_digest": request["payload_digest"],
        "started_at_utc": started_at,
        "finished_at_utc": utc_now(),
        "palette_git": git,
        "publication_result_file": publication_result_path.name,
        "publication_result_payload_digest": publication_result["payload_digest"],
        "publication_process_resources": publication_resources,
        "read_trial_files": trial_files,
        "read_trial_payload_digests": [
            trial["payload_digest"] for trial in read_trials
        ],
        "read_trial_process_resources": read_resources,
        "storage": trial_payloads[0]["storage"],
        "performance_summary": {
            "publication_wall_seconds": publication["process_runtime"]["wall_seconds"],
            "publisher_phase_seconds": publication["publisher_runtime_telemetry"][
                "phases_seconds"
            ],
            "median_validation_seconds": _median(validation_seconds),
            "median_random_frame_p95_seconds": _median(random_p95),
            "median_window_p95_seconds": _median(window_p95),
            "median_full_scan_seconds": _median(full_scan_seconds),
            "full_scan_logical_stream_sha256": next(iter(full_digests)),
        },
        "source_metadata_guard": {
            "file_count": len(guard_before),
            "before_sha256_by_relative_path": guard_before,
            "after_sha256_by_relative_path": guard_after,
            "unchanged": True,
        },
        "physical_io": {
            "network_request_count": None,
            "network_transferred_bytes": None,
            "availability": (
                "not measured by this runner; Linux process-requested file bytes "
                "and Crimson mounted-reader telemetry are separate evidence"
            ),
        },
        "promotion_authorized": False,
    }
    result = _strict_envelope(
        MATRIX_RESULT_SCHEMA_ID,
        MATRIX_RESULT_SCHEMA_VERSION,
        body,
    )
    require_matrix_result(result, request=request)
    _write_strict_json(output / "matrix_result.json", result)
    return result


def require_matrix_result(
    value: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
) -> Mapping[str, Any]:
    request_payload = require_request(request, require_paths=False)
    payload = _require_envelope(
        value,
        schema_id=MATRIX_RESULT_SCHEMA_ID,
        schema_version=MATRIX_RESULT_SCHEMA_VERSION,
    )
    expected = {
        "benchmark_id",
        "family_id",
        "scale_id",
        "request_payload_digest",
        "started_at_utc",
        "finished_at_utc",
        "palette_git",
        "publication_result_file",
        "publication_result_payload_digest",
        "publication_process_resources",
        "read_trial_files",
        "read_trial_payload_digests",
        "read_trial_process_resources",
        "storage",
        "performance_summary",
        "source_metadata_guard",
        "physical_io",
        "promotion_authorized",
    }
    if set(payload) != expected:
        raise ValueError("Benchmark matrix result has an unexpected field set.")
    if (
        payload["benchmark_id"] != BENCHMARK_ID
        or payload["family_id"] != request_payload["family_id"]
        or payload["scale_id"] != request_payload["scale_id"]
        or payload["request_payload_digest"] != request["payload_digest"]
        or payload["promotion_authorized"] is not False
    ):
        raise ValueError("Benchmark matrix identity or promotion boundary is invalid.")
    repetitions = int(request_payload["workload"]["repetitions"])
    if (
        len(payload["read_trial_files"]) != repetitions
        or len(payload["read_trial_payload_digests"]) != repetitions
        or len(payload["read_trial_process_resources"]) != repetitions
    ):
        raise ValueError("Benchmark matrix read-trial inventory is incomplete.")
    guard = payload["source_metadata_guard"]
    if not isinstance(guard, Mapping) or guard.get("unchanged") is not True:
        raise ValueError("Benchmark matrix source nonmutation guard did not pass.")
    physical = payload["physical_io"]
    if (
        physical.get("network_request_count") is not None
        or physical.get("network_transferred_bytes") is not None
    ):
        raise ValueError("Benchmark matrix fabricated network-transfer telemetry.")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-request")
    build.add_argument("--family", choices=tuple(sorted(_FAMILIES)), required=True)
    build.add_argument("--scale", choices=tuple(sorted(_SCALE_IDS)), required=True)
    build.add_argument("--zarr", type=Path, required=True)
    build.add_argument("--export-root", type=Path, required=True)
    build.add_argument("--scratch-root", type=Path, required=True)
    build.add_argument("--benchmark-output-dir", type=Path, required=True)
    build.add_argument("--export-run-id", required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--eye-angle-run")
    build.add_argument("--track-kinematics-run")
    build.add_argument("--track-scope", choices=tuple(sorted(_TRACK_SCOPES)))
    build.add_argument("--tail-kinematics-run")
    build.add_argument("--subject-shape-run")
    build.add_argument(
        "--track-swim-bout-run",
        action="append",
        default=[],
        metavar="TRACK_ID=RUN",
    )
    build.add_argument("--requested-sample-rate-hz", type=float)
    build.add_argument("--requested-bin-size-s", type=float)
    build.add_argument("--source-window-rows", type=int)
    build.add_argument("--source-rows-per-part", type=int)
    build.add_argument("--row-group-rows", type=int, default=65_536)
    build.add_argument("--seed", type=int, default=DEFAULT_SEED)
    build.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    build.add_argument(
        "--random-frame-reads", type=int, default=DEFAULT_RANDOM_FRAME_READS
    )
    build.add_argument("--window-count", type=int, default=DEFAULT_WINDOW_COUNT)
    build.add_argument("--window-frames", type=int, default=DEFAULT_WINDOW_FRAMES)
    build.add_argument("--requested-workers", type=int, default=1)
    build.add_argument("--allocated-slots", type=int, default=1)
    build.add_argument("--sample-interval-seconds", type=float, default=0.25)
    build.add_argument("--cache-state", default="uncontrolled_fresh_process")
    for name in ("matrix", "publish", "read"):
        child = subparsers.add_parser(name)
        child.add_argument("--request", type=Path, required=True)
        if name in {"publish", "read"}:
            child.add_argument("--output", type=Path, required=True)
        if name == "read":
            child.add_argument("--repetition-index", type=int, required=True)
    return parser


def _required_cli_value(value: object, *, option: str) -> object:
    if value is None:
        raise ValueError(f"{option} is required for the selected benchmark family.")
    return value


def _request_from_args(args: argparse.Namespace) -> dict[str, Any]:
    family = str(args.family)
    if family == "eye_trace_samples":
        source_runs = {
            "eye_angle_run": _required_cli_value(
                args.eye_angle_run, option="--eye-angle-run"
            )
        }
        parameters = {"row_group_rows": args.row_group_rows}
    elif family == "kinematics_samples":
        source_runs = {
            "track_kinematics_run": _required_cli_value(
                args.track_kinematics_run, option="--track-kinematics-run"
            ),
            "track_scope": _required_cli_value(
                args.track_scope, option="--track-scope"
            ),
        }
        parameters = {
            "requested_sample_rate_hz": _required_cli_value(
                args.requested_sample_rate_hz, option="--requested-sample-rate-hz"
            ),
            "source_window_rows": _required_cli_value(
                args.source_window_rows, option="--source-window-rows"
            ),
            "row_group_rows": args.row_group_rows,
        }
    elif family == "activity_spatial_time_bins":
        run_map: dict[str, str] = {}
        for raw in args.track_swim_bout_run:
            track_id, separator, run_name = str(raw).partition("=")
            if not separator or not track_id or not run_name or track_id in run_map:
                raise ValueError(
                    "--track-swim-bout-run requires one unique TRACK_ID=RUN value."
                )
            run_map[track_id] = run_name
        source_runs = {
            "track_kinematics_run": _required_cli_value(
                args.track_kinematics_run, option="--track-kinematics-run"
            ),
            "track_scope": _required_cli_value(
                args.track_scope, option="--track-scope"
            ),
            "swim_bout_runs_by_track": run_map,
        }
        parameters = {
            "requested_bin_size_s": _required_cli_value(
                args.requested_bin_size_s, option="--requested-bin-size-s"
            ),
            "row_group_rows": args.row_group_rows,
        }
    else:
        source_runs = {
            "tail_kinematics_run": _required_cli_value(
                args.tail_kinematics_run, option="--tail-kinematics-run"
            ),
            "subject_shape_run": _required_cli_value(
                args.subject_shape_run, option="--subject-shape-run"
            ),
            "track_kinematics_run": _required_cli_value(
                args.track_kinematics_run, option="--track-kinematics-run"
            ),
            "track_scope": _required_cli_value(
                args.track_scope, option="--track-scope"
            ),
        }
        parameters = {
            "source_window_rows": _required_cli_value(
                args.source_window_rows, option="--source-window-rows"
            ),
            "source_rows_per_part": _required_cli_value(
                args.source_rows_per_part, option="--source-rows-per-part"
            ),
            "row_group_rows": args.row_group_rows,
        }
    return build_request(
        family_id=family,
        scale_id=args.scale,
        zarr_path=args.zarr,
        export_root=args.export_root,
        scratch_root=args.scratch_root,
        benchmark_output_dir=args.benchmark_output_dir,
        export_run_id=args.export_run_id,
        source_runs=source_runs,
        publisher_parameters=parameters,
        seed=args.seed,
        repetitions=args.repetitions,
        random_frame_reads=args.random_frame_reads,
        window_count=args.window_count,
        window_frames=args.window_frames,
        requested_workers=args.requested_workers,
        allocated_slots=args.allocated_slots,
        sample_interval_seconds=args.sample_interval_seconds,
        cache_state=args.cache_state,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build-request":
        request = _request_from_args(args)
        require_request(request)
        _write_strict_json(args.output, request)
        print(
            json.dumps(
                {
                    "status": "complete",
                    "request": str(args.output.expanduser().resolve()),
                    "payload_digest": request["payload_digest"],
                },
                sort_keys=True,
            )
        )
        return 0
    request_path = args.request.expanduser().resolve()
    request = _read_strict_json(request_path)
    if args.command == "publish":
        result = run_publication(request)
        _write_strict_json(args.output, result)
        return 0
    if args.command == "read":
        result = run_read_trial(request, repetition_index=args.repetition_index)
        _write_strict_json(args.output, result)
        return 0
    result = run_matrix(request)
    print(
        json.dumps(
            {
                "status": "complete",
                "matrix_result": str(
                    Path(request["payload"]["benchmark_output_dir"])
                    / "matrix_result.json"
                ),
                "payload_digest": result["payload_digest"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "BENCHMARK_ID",
    "DEFAULT_REPETITIONS",
    "MATRIX_RESULT_SCHEMA_ID",
    "READ_TRIAL_SCHEMA_ID",
    "REQUEST_SCHEMA_ID",
    "build_request",
    "require_matrix_result",
    "require_publication_result",
    "require_read_trial",
    "require_request",
    "run_matrix",
    "run_publication",
    "run_read_trial",
]
