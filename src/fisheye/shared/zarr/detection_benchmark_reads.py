"""Read-only destination-tier workloads for canonical detection candidates."""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import zarr

from fisheye.shared.zarr.benchmark_contracts import (
    BenchmarkPhase,
    EAGER_FULL_READ_V1,
    StorageBenchmarkCase,
    WINDOWED_ROWS_READ_V1,
    benchmark_result_envelope,
    require_benchmark_result_envelope,
)
from fisheye.shared.zarr.benchmark_runtime import (
    local_environment_manifest,
    sha256_array,
)
from fisheye.shared.zarr.canonical_detection_benchmark_input import (
    CanonicalDetectionBenchmarkInput,
)
from fisheye.shared.zarr.detection_benchmark_access import (
    DetectionReadWorkloadConfig,
    benchmark_detection_consumer_workloads,
)
from fisheye.shared.zarr.detection_schema import CANONICAL_DETECTION_SCHEMA_V1
from fisheye.shared.zarr.detection_storage import CanonicalDetectionStoragePlanSet


def benchmark_detection_candidate_reads(
    benchmark_input: CanonicalDetectionBenchmarkInput,
    *,
    candidate: Path,
    plans: CanonicalDetectionStoragePlanSet,
    storage_tier: str,
    workload_config: DetectionReadWorkloadConfig | None = None,
) -> dict[str, object]:
    """Measure exact first/warm array and composite consumer reads."""

    candidate_path = candidate.expanduser().resolve()
    if plans.dimensions != benchmark_input.dimensions:
        raise ValueError("Read benchmark plan dimensions do not match source.")
    config = workload_config or DetectionReadWorkloadConfig()
    direct_open_trials: list[dict[str, object]] = []
    consolidated_open_trials: list[dict[str, object]] = []
    group = None
    for pass_index in range(config.pass_count):
        direct_started = time.perf_counter()
        zarr.open_group(str(candidate_path), mode="r", use_consolidated=False)
        direct_open_trials.append(
            {
                "pass_index": pass_index,
                "cache_condition": (
                    "process_first_pass_os_cache_uncontrolled"
                    if pass_index == 0
                    else f"same_process_warm_pass_{pass_index}"
                ),
                "seconds": float(time.perf_counter() - direct_started),
            }
        )
        consolidated_started = time.perf_counter()
        group = zarr.open_group(
            str(candidate_path),
            mode="r",
            use_consolidated=True,
        )
        consolidated_open_trials.append(
            {
                "pass_index": pass_index,
                "cache_condition": (
                    "process_first_pass_os_cache_uncontrolled"
                    if pass_index == 0
                    else f"same_process_warm_pass_{pass_index}"
                ),
                "seconds": float(time.perf_counter() - consolidated_started),
            }
        )
    if group is None:  # pragma: no cover - config validation requires passes
        raise RuntimeError("Detection read benchmark did not open the candidate.")
    direct_open_seconds = float(direct_open_trials[0]["seconds"])
    consolidated_open_seconds = float(consolidated_open_trials[0]["seconds"])

    environment = local_environment_manifest()
    environment.update(
        {
            "storage_tier": str(storage_tier),
            "cache_state": (
                "process_first_then_same_process_warm_os_cache_uncontrolled"
            ),
        }
    )
    consumer_workloads = benchmark_detection_consumer_workloads(
        benchmark_input,
        group=group,
        config=config,
    )
    results: list[dict[str, object]] = []
    envelopes: list[dict[str, object]] = []
    for entry in plans.entries:
        path = entry.rule.path
        array = group[path]
        expected = benchmark_input.arrays[path]
        rows = int(array.shape[0])
        window_rows = min(1024, rows)
        window_start = max(0, (rows - window_rows) // 2)
        expected_digest = sha256_array(expected)
        window_trials: list[dict[str, object]] = []
        full_trials: list[dict[str, object]] = []
        for pass_index in range(config.pass_count):
            cache_condition = (
                "process_first_pass_os_cache_uncontrolled"
                if pass_index == 0
                else f"same_process_warm_pass_{pass_index}"
            )
            read_started = time.perf_counter()
            window = np.asarray(
                array[window_start : window_start + window_rows, ...]
            )
            window_seconds = float(time.perf_counter() - read_started)
            expected_window = expected[
                window_start : window_start + window_rows, ...
            ]
            if not np.array_equal(window, expected_window):
                raise RuntimeError(f"Published window mismatch at {path}.")
            window_trials.append(
                {
                    "pass_index": pass_index,
                    "cache_condition": cache_condition,
                    "read_seconds": window_seconds,
                    "window_start": window_start,
                    "window_rows": window_rows,
                    "requested_rows": window_rows,
                    "logical_bytes": int(window.nbytes),
                    "window_sha256": sha256_array(window),
                    "decoded_bytes": None,
                    "transferred_bytes": None,
                    "request_count": None,
                    "exact": True,
                }
            )
            read_started = time.perf_counter()
            full = np.asarray(array[:])
            full_seconds = float(time.perf_counter() - read_started)
            full_digest = sha256_array(full)
            if full_digest != expected_digest:
                raise RuntimeError(f"Published read digest mismatch at {path}.")
            full_trials.append(
                {
                    "pass_index": pass_index,
                    "cache_condition": cache_condition,
                    "read_seconds": full_seconds,
                    "requested_rows": int(expected.shape[0]),
                    "logical_bytes": int(expected.nbytes),
                    "full_sha256": full_digest,
                    "decoded_bytes": None,
                    "transferred_bytes": None,
                    "request_count": None,
                    "exact": True,
                }
            )
        result = {
            "path": path,
            "window_start": window_start,
            "window_rows": window_rows,
            "window_seconds": window_trials[0]["read_seconds"],
            "warm_window_seconds": window_trials[1]["read_seconds"],
            "window_sha256": window_trials[0]["window_sha256"],
            "full_seconds": full_trials[0]["read_seconds"],
            "warm_full_seconds": full_trials[1]["read_seconds"],
            "full_sha256": full_trials[0]["full_sha256"],
            "expected_sha256": expected_digest,
            "window_trials": window_trials,
            "full_trials": full_trials,
            "exact": True,
        }
        results.append(result)

        binding = next(
            item for item in CANONICAL_DETECTION_SCHEMA_V1.bindings if item.path == path
        )
        contract = CANONICAL_DETECTION_SCHEMA_V1.contracts.resolve(
            binding.contract_id,
            binding.contract_version,
        )
        workload = (
            EAGER_FULL_READ_V1
            if entry.plan.access_pattern == "eager"
            else WINDOWED_ROWS_READ_V1
        )
        case = StorageBenchmarkCase(
            case_id=f"{path}__{plans.profile.profile_id}__{storage_tier}__read",
            phase=BenchmarkPhase.READ,
            array_contract=contract,
            storage_plan=entry.plan,
            workload=workload,
        )
        eager = entry.plan.access_pattern == "eager"
        trials = full_trials if eager else window_trials
        envelope = benchmark_result_envelope(
            case,
            source_identity=benchmark_input.source_identity,
            environment=environment,
            trials=trials,
            summary=trials[0],
            validation={"exact": True, "expected_sha256": expected_digest},
        )
        require_benchmark_result_envelope(envelope)
        envelopes.append(envelope)

    return {
        "schema_id": "palette.canonical_detection_destination_reads",
        "schema_version": 2,
        "status": "complete",
        "candidate": str(candidate_path),
        "storage_tier": str(storage_tier),
        "execution_order": [
            "direct_and_consolidated_metadata_open_trials",
            "consumer_workloads",
            "per_array_window_and_full_scan_trials",
        ],
        "direct_open_seconds": direct_open_seconds,
        "consolidated_open_seconds": consolidated_open_seconds,
        "direct_open_trials": direct_open_trials,
        "consolidated_open_trials": consolidated_open_trials,
        "arrays": results,
        "consumer_workloads": consumer_workloads,
        "common_benchmark_envelopes": envelopes,
        "environment": environment,
    }


__all__ = ["benchmark_detection_candidate_reads"]
