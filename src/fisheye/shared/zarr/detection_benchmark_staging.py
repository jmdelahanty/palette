"""Fixed local canonical staging for detection storage benchmarks."""

from __future__ import annotations

from pathlib import Path
import time

import zarr

from fisheye.shared.zarr.canonical_detection_benchmark import (
    CanonicalDetectionBenchmarkInput,
    consolidate_and_open_detection_benchmark_candidate,
    materialize_detection_benchmark_candidate,
    validate_detection_benchmark_candidate,
)
from fisheye.shared.zarr.detection_storage import plan_canonical_detection_storage
from fisheye.shared.zarr.storage_profiles import SCRATCH_COMPUTE_V1


def prepare_canonical_detection_benchmark_staging(
    benchmark_input: CanonicalDetectionBenchmarkInput,
    *,
    destination: Path,
    scratch_root: Path,
) -> dict[str, object]:
    """Materialize and validate the one canonical source shared by candidates."""

    plans = plan_canonical_detection_storage(
        benchmark_input.dimensions,
        profile=SCRATCH_COMPUTE_V1,
    )
    started = time.perf_counter()
    materialization = materialize_detection_benchmark_candidate(
        benchmark_input,
        destination=destination,
        plans=plans,
        benchmark_root=scratch_root,
    )
    mutable = zarr.open_group(
        str(materialization.output_path),
        mode="r+",
        use_consolidated=False,
    )
    mutable.attrs["benchmark_input_source_identity"] = dict(
        benchmark_input.source_identity
    )
    validation = validate_detection_benchmark_candidate(
        benchmark_input,
        materialization,
    )
    metadata = consolidate_and_open_detection_benchmark_candidate(materialization)
    return {
        "schema_id": "palette.canonical_detection_benchmark_staging",
        "schema_version": 1,
        "status": "complete",
        "destination": str(materialization.output_path),
        "dimensions": benchmark_input.dimensions.as_manifest(),
        "source": benchmark_input.as_manifest(),
        "storage_plan": plans.as_manifest(),
        "digest_validation": dict(validation.digests),
        "timing": {
            "total_seconds": float(time.perf_counter() - started),
            "validation_seconds": validation.seconds,
            "consolidation_seconds": metadata.consolidation_seconds,
            "direct_open_seconds": metadata.direct_open_seconds,
            "consolidated_open_seconds": metadata.consolidated_open_seconds,
            "consolidation_warnings": list(metadata.consolidation_warnings),
        },
    }


__all__ = ["prepare_canonical_detection_benchmark_staging"]
