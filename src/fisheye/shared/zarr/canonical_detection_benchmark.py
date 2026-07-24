"""Reusable kernel for disposable canonical detection Zarr benchmarks.

The source group is opened read-only. Destinations must be fresh paths below
``/tmp/palette-zarr-benchmarks``; this utility cannot write into a Palette
recording archive or update selectors.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Mapping, Sequence
import warnings

import numpy as np
import zarr

from fisheye.shared.zarr.array_factory import create_array_from_plan
from fisheye.shared.zarr.benchmark_contracts import (
    BenchmarkPhase,
    EAGER_FULL_READ_V1,
    StorageBenchmarkCase,
    WINDOWED_ROWS_READ_V1,
    WRITE_MATERIALIZATION_V1,
    benchmark_result_envelope,
    require_benchmark_result_envelope,
)
from fisheye.shared.zarr.benchmark_runtime import (
    local_environment_manifest,
    peak_rss_bytes,
    sha256_array,
    storage_stats,
    utc_now,
)
from fisheye.shared.zarr.canonical_detection_benchmark_input import (
    CanonicalDetectionBenchmarkInput,
    build_canonical_detection_benchmark_input,
    load_detection_benchmark_input,
)
from fisheye.shared.zarr.detection_schema import CANONICAL_DETECTION_SCHEMA_V1
from fisheye.shared.zarr.detection_storage import CanonicalDetectionStoragePlanSet


BENCHMARK_OUTPUT_ROOT = Path("/tmp/palette-zarr-benchmarks")
REPORT_SCHEMA_ID = "palette.canonical_detection_storage_benchmark"
REPORT_SCHEMA_VERSION = 1


def require_safe_benchmark_destination(
    destination: Path,
    *,
    benchmark_root: Path = BENCHMARK_OUTPUT_ROOT,
) -> Path:
    path = destination.expanduser().resolve()
    root = benchmark_root.expanduser().resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError(f"Benchmark destination must be below {root}.")
    if path.exists():
        raise FileExistsError(f"Benchmark destination already exists: {path}")
    return path


def _write_array_by_physical_units(
    destination: Any,
    values: np.ndarray,
    *,
    plan: Any,
) -> None:
    unit_rows = int(
        plan.shard_shape[0] if plan.shard_shape is not None else plan.chunk_shape[0]
    )
    trailing = (slice(None),) * (values.ndim - 1)
    for start in range(0, int(values.shape[0]), unit_rows):
        stop = min(start + unit_rows, int(values.shape[0]))
        selection = (slice(start, stop), *trailing)
        destination[selection] = values[selection]


@dataclass(frozen=True)
class DetectionCandidateMaterialization:
    """Fresh candidate arrays and per-array write measurements."""

    output_path: Path
    destination_arrays: Mapping[str, Any]
    write_results: tuple[Mapping[str, object], ...]


@dataclass(frozen=True)
class DetectionCandidateValidation:
    """Exact decoded-value validation for a materialized candidate."""

    seconds: float
    digests: Mapping[str, Mapping[str, object]]


@dataclass(frozen=True)
class DetectionCandidateMetadataOpen:
    """Immutable metadata publication and direct/consolidated open timings."""

    consolidated_group: Any
    consolidation_seconds: float
    consolidation_warnings: tuple[str, ...]
    direct_open_seconds: float
    consolidated_open_seconds: float


def materialize_detection_benchmark_candidate(
    benchmark_input: CanonicalDetectionBenchmarkInput,
    *,
    destination: Path,
    plans: CanonicalDetectionStoragePlanSet,
    benchmark_root: Path = BENCHMARK_OUTPUT_ROOT,
) -> DetectionCandidateMaterialization:
    """Create and completely write one exclusive benchmark destination."""

    output_path = require_safe_benchmark_destination(
        destination,
        benchmark_root=benchmark_root,
    )
    if plans.dimensions != benchmark_input.dimensions:
        raise ValueError("Storage plan dimensions do not match benchmark input.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(output_path), mode="w-", zarr_format=3)
    root.attrs.update(
        {
            "benchmark_only": True,
            "canonical": False,
            "registry_registered": False,
            "selector_eligible": False,
            "schema_id": REPORT_SCHEMA_ID,
            "schema_version": REPORT_SCHEMA_VERSION,
            "created_at_utc": utc_now(),
            "logical_schema": CANONICAL_DETECTION_SCHEMA_V1.as_manifest(
                dimensions=benchmark_input.dimensions
            ),
            "storage_plan": plans.as_manifest(),
        }
    )
    instances = root.create_group("instances")
    write_results: list[Mapping[str, object]] = []
    destination_arrays: dict[str, Any] = {}
    for entry in plans.entries:
        path = entry.rule.path
        leaf = path.rsplit("/", 1)[-1]
        values = benchmark_input.arrays[path]
        binding = next(
            item for item in CANONICAL_DETECTION_SCHEMA_V1.bindings if item.path == path
        )
        contract = CANONICAL_DETECTION_SCHEMA_V1.contracts.resolve(
            binding.contract_id,
            binding.contract_version,
        )
        array_started = time.perf_counter()
        destination_array = create_array_from_plan(
            instances,
            name=leaf,
            contract=contract,
            plan=entry.plan,
            fill_value=0,
        )
        _write_array_by_physical_units(
            destination_array,
            values,
            plan=entry.plan,
        )
        write_seconds = float(time.perf_counter() - array_started)
        destination_arrays[path] = destination_array
        write_results.append(
            {
                "path": path,
                "write_seconds": write_seconds,
                "logical_bytes": int(values.nbytes),
                "peak_rss_bytes": peak_rss_bytes(),
                "plan": entry.plan.as_dict(),
            }
        )
    return DetectionCandidateMaterialization(
        output_path=output_path,
        destination_arrays=destination_arrays,
        write_results=tuple(write_results),
    )


def validate_detection_benchmark_candidate(
    benchmark_input: CanonicalDetectionBenchmarkInput,
    materialization: DetectionCandidateMaterialization,
) -> DetectionCandidateValidation:
    """Validate exact schema, dtype, shape, and decoded array digests."""

    validation_started = time.perf_counter()
    CANONICAL_DETECTION_SCHEMA_V1.require(
        materialization.destination_arrays,
        dimensions=benchmark_input.dimensions,
    )
    digest_validation: dict[str, dict[str, object]] = {}
    for path, source_values in benchmark_input.arrays.items():
        destination_values = np.asarray(materialization.destination_arrays[path][:])
        source_digest = sha256_array(source_values)
        destination_digest = sha256_array(destination_values)
        digest_validation[path] = {
            "source_sha256": source_digest,
            "destination_sha256": destination_digest,
            "exact": source_digest == destination_digest,
        }
    validation_seconds = float(time.perf_counter() - validation_started)
    if not all(bool(item["exact"]) for item in digest_validation.values()):
        raise RuntimeError("Canonical detection candidate digest mismatch.")
    return DetectionCandidateValidation(
        seconds=validation_seconds,
        digests=digest_validation,
    )


def consolidate_and_open_detection_benchmark_candidate(
    materialization: DetectionCandidateMaterialization,
) -> DetectionCandidateMetadataOpen:
    """Consolidate immutable metadata and measure both supported open modes."""

    output_path = materialization.output_path
    consolidation_started = time.perf_counter()
    with warnings.catch_warnings(record=True) as consolidation_warning_records:
        warnings.simplefilter("always")
        zarr.consolidate_metadata(str(output_path))
    consolidation_seconds = float(time.perf_counter() - consolidation_started)
    consolidation_warnings = [
        str(item.message) for item in consolidation_warning_records
    ]
    direct_started = time.perf_counter()
    zarr.open_group(str(output_path), mode="r", use_consolidated=False)
    direct_open_seconds = float(time.perf_counter() - direct_started)
    consolidated_started = time.perf_counter()
    consolidated = zarr.open_group(
        str(output_path),
        mode="r",
        use_consolidated=True,
    )
    consolidated_open_seconds = float(time.perf_counter() - consolidated_started)
    return DetectionCandidateMetadataOpen(
        consolidated_group=consolidated,
        consolidation_seconds=consolidation_seconds,
        consolidation_warnings=tuple(consolidation_warnings),
        direct_open_seconds=direct_open_seconds,
        consolidated_open_seconds=consolidated_open_seconds,
    )


def run_detection_benchmark_read_workloads(
    metadata_open: DetectionCandidateMetadataOpen,
    *,
    plans: CanonicalDetectionStoragePlanSet,
) -> tuple[Mapping[str, object], ...]:
    """Run deterministic representative window and full-array reads."""

    read_results: list[dict[str, object]] = []
    for entry in plans.entries:
        path = entry.rule.path
        array = metadata_open.consolidated_group[path]
        rows = int(array.shape[0])
        window_rows = min(1024, rows)
        window_start = max(0, (rows - window_rows) // 2)
        read_started = time.perf_counter()
        window = np.asarray(array[window_start : window_start + window_rows, ...])
        window_seconds = float(time.perf_counter() - read_started)
        read_started = time.perf_counter()
        full = np.asarray(array[:])
        full_seconds = float(time.perf_counter() - read_started)
        read_results.append(
            {
                "path": path,
                "window_rows": window_rows,
                "window_seconds": window_seconds,
                "window_sha256": sha256_array(window),
                "full_seconds": full_seconds,
                "full_sha256": sha256_array(full),
            }
        )
    return tuple(read_results)


def build_detection_benchmark_report(
    benchmark_input: CanonicalDetectionBenchmarkInput,
    *,
    plans: CanonicalDetectionStoragePlanSet,
    materialization: DetectionCandidateMaterialization,
    validation: DetectionCandidateValidation,
    metadata_open: DetectionCandidateMetadataOpen,
    read_results: Sequence[Mapping[str, object]],
    total_seconds: float,
) -> dict[str, object]:
    """Build and validate dataset-specific plus common benchmark envelopes."""

    output_path = materialization.output_path
    physical = storage_stats(output_path)
    source_manifest = benchmark_input.as_manifest()
    environment = local_environment_manifest()
    common_envelopes: list[dict[str, object]] = []
    write_by_path = {
        str(item["path"]): item for item in materialization.write_results
    }
    read_by_path = {str(item["path"]): item for item in read_results}
    for entry in plans.entries:
        path = entry.rule.path
        binding = next(
            item for item in CANONICAL_DETECTION_SCHEMA_V1.bindings if item.path == path
        )
        contract = CANONICAL_DETECTION_SCHEMA_V1.contracts.resolve(
            binding.contract_id,
            binding.contract_version,
        )
        write_case = StorageBenchmarkCase(
            case_id=f"{path}__{plans.profile.profile_id}__write",
            phase=BenchmarkPhase.WRITE,
            array_contract=contract,
            storage_plan=entry.plan,
            workload=WRITE_MATERIALIZATION_V1,
        )
        array_stats = storage_stats(output_path / path)
        write_trial = {
            **write_by_path[path],
            "physical_bytes": array_stats["apparent_bytes"],
            "payload_object_count": array_stats["payload_file_count"],
        }
        write_envelope = benchmark_result_envelope(
            write_case,
            source_identity=source_manifest["source_identity"],
            environment=environment,
            trials=[write_trial],
            summary=write_trial,
            validation=validation.digests[path],
        )
        require_benchmark_result_envelope(write_envelope)
        common_envelopes.append(write_envelope)
        read_workload = (
            EAGER_FULL_READ_V1
            if entry.plan.access_pattern == "eager"
            else WINDOWED_ROWS_READ_V1
        )
        read_case = StorageBenchmarkCase(
            case_id=f"{path}__{plans.profile.profile_id}__read",
            phase=BenchmarkPhase.READ,
            array_contract=contract,
            storage_plan=entry.plan,
            workload=read_workload,
        )
        read_result = read_by_path[path]
        eager_read = entry.plan.access_pattern == "eager"
        read_trial = {
            **read_result,
            "logical_bytes": int(benchmark_input.arrays[path].nbytes),
            "requested_rows": (
                int(benchmark_input.arrays[path].shape[0])
                if eager_read
                else int(read_result["window_rows"])
            ),
            "decoded_bytes": None,
            "transferred_bytes": None,
            "request_count": None,
            "read_seconds": (
                read_result["full_seconds"]
                if eager_read
                else read_result["window_seconds"]
            ),
        }
        read_envelope = benchmark_result_envelope(
            read_case,
            source_identity=source_manifest["source_identity"],
            environment=environment,
            trials=[read_trial],
            summary=read_trial,
            validation=validation.digests[path],
        )
        require_benchmark_result_envelope(read_envelope)
        common_envelopes.append(read_envelope)

    return {
        "schema_id": REPORT_SCHEMA_ID,
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "complete_exploratory_smoke",
        "destination": str(output_path),
        "source": source_manifest,
        "storage_plan": plans.as_manifest(),
        "timing": {
            "total_seconds": total_seconds,
            "validation_seconds": validation.seconds,
            "consolidation_seconds": metadata_open.consolidation_seconds,
            "consolidation_warnings": list(
                metadata_open.consolidation_warnings
            ),
            "direct_open_seconds": metadata_open.direct_open_seconds,
            "consolidated_open_seconds": metadata_open.consolidated_open_seconds,
        },
        "physical": physical,
        "digest_validation": dict(validation.digests),
        "common_benchmark_envelopes": common_envelopes,
        "environment": environment,
    }


def write_detection_benchmark_candidate(
    benchmark_input: CanonicalDetectionBenchmarkInput,
    *,
    destination: Path,
    plans: CanonicalDetectionStoragePlanSet,
    benchmark_root: Path = BENCHMARK_OUTPUT_ROOT,
) -> dict[str, object]:
    """Write, validate, consolidate, read, and report one fresh candidate."""

    started = time.perf_counter()
    materialization = materialize_detection_benchmark_candidate(
        benchmark_input,
        destination=destination,
        plans=plans,
        benchmark_root=benchmark_root,
    )
    validation = validate_detection_benchmark_candidate(
        benchmark_input,
        materialization,
    )
    metadata_open = consolidate_and_open_detection_benchmark_candidate(
        materialization
    )
    read_results = run_detection_benchmark_read_workloads(
        metadata_open,
        plans=plans,
    )
    return build_detection_benchmark_report(
        benchmark_input,
        plans=plans,
        materialization=materialization,
        validation=validation,
        metadata_open=metadata_open,
        read_results=read_results,
        total_seconds=float(time.perf_counter() - started),
    )


__all__ = [
    "BENCHMARK_OUTPUT_ROOT",
    "REPORT_SCHEMA_ID",
    "REPORT_SCHEMA_VERSION",
    "CanonicalDetectionBenchmarkInput",
    "DetectionCandidateMaterialization",
    "DetectionCandidateMetadataOpen",
    "DetectionCandidateValidation",
    "build_detection_benchmark_report",
    "build_canonical_detection_benchmark_input",
    "consolidate_and_open_detection_benchmark_candidate",
    "load_detection_benchmark_input",
    "materialize_detection_benchmark_candidate",
    "require_safe_benchmark_destination",
    "run_detection_benchmark_read_workloads",
    "validate_detection_benchmark_candidate",
    "write_detection_benchmark_candidate",
]
