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
from fisheye.shared.zarr.detection_schema import CANONICAL_DETECTION_SCHEMA_V1
from fisheye.shared.zarr.detection_storage import CanonicalDetectionStoragePlanSet


def benchmark_detection_candidate_reads(
    benchmark_input: CanonicalDetectionBenchmarkInput,
    *,
    candidate: Path,
    plans: CanonicalDetectionStoragePlanSet,
    storage_tier: str,
) -> dict[str, object]:
    """Measure direct/consolidated opens, windows, and full exact reads."""

    candidate_path = candidate.expanduser().resolve()
    if plans.dimensions != benchmark_input.dimensions:
        raise ValueError("Read benchmark plan dimensions do not match source.")
    direct_started = time.perf_counter()
    zarr.open_group(str(candidate_path), mode="r", use_consolidated=False)
    direct_open_seconds = float(time.perf_counter() - direct_started)
    consolidated_started = time.perf_counter()
    group = zarr.open_group(
        str(candidate_path),
        mode="r",
        use_consolidated=True,
    )
    consolidated_open_seconds = float(time.perf_counter() - consolidated_started)

    environment = local_environment_manifest()
    environment.update(
        {
            "storage_tier": str(storage_tier),
            "cache_state": "uncontrolled_first_pass_after_publication",
        }
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
        read_started = time.perf_counter()
        window = np.asarray(array[window_start : window_start + window_rows, ...])
        window_seconds = float(time.perf_counter() - read_started)
        read_started = time.perf_counter()
        full = np.asarray(array[:])
        full_seconds = float(time.perf_counter() - read_started)
        expected_digest = sha256_array(expected)
        full_digest = sha256_array(full)
        if full_digest != expected_digest:
            raise RuntimeError(f"Published read digest mismatch at {path}.")
        result = {
            "path": path,
            "window_start": window_start,
            "window_rows": window_rows,
            "window_seconds": window_seconds,
            "window_sha256": sha256_array(window),
            "full_seconds": full_seconds,
            "full_sha256": full_digest,
            "expected_sha256": expected_digest,
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
        trial = {
            **result,
            "read_seconds": full_seconds if eager else window_seconds,
            "logical_bytes": int(expected.nbytes),
            "requested_rows": int(expected.shape[0]) if eager else window_rows,
            "decoded_bytes": None,
            "transferred_bytes": None,
            "request_count": None,
        }
        envelope = benchmark_result_envelope(
            case,
            source_identity=benchmark_input.source_identity,
            environment=environment,
            trials=[trial],
            summary=trial,
            validation={"exact": True, "expected_sha256": expected_digest},
        )
        require_benchmark_result_envelope(envelope)
        envelopes.append(envelope)

    return {
        "schema_id": "palette.canonical_detection_destination_reads",
        "schema_version": 1,
        "status": "complete",
        "candidate": str(candidate_path),
        "storage_tier": str(storage_tier),
        "direct_open_seconds": direct_open_seconds,
        "consolidated_open_seconds": consolidated_open_seconds,
        "arrays": results,
        "common_benchmark_envelopes": envelopes,
        "environment": environment,
    }


__all__ = ["benchmark_detection_candidate_reads"]
