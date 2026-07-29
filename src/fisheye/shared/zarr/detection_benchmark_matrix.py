"""Canonical-detection adapter for the shared storage benchmark matrix."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

from fisheye.shared.zarr.benchmark_contracts import (
    EAGER_FULL_READ_V1,
    FULL_SCAN_READ_V1,
    PUBLISH_VALIDATE_V1,
    WINDOWED_ROWS_READ_V1,
    WRITE_MATERIALIZATION_V1,
)
from fisheye.shared.zarr.benchmark_matrix import (
    BenchmarkLayout,
    BenchmarkScale,
    MatrixWorkload,
    StorageBenchmarkMatrix,
    StorageCandidateRequest,
    plan_storage_benchmark_matrix,
)
from fisheye.shared.zarr.detection_schema import CanonicalDetectionDimensions
from fisheye.shared.zarr.detection_benchmark_planning import (
    plan_detection_benchmark_candidate,
)
from fisheye.shared.zarr.storage_profiles import (
    KIB,
    MIB,
)


INITIAL_CHUNK_TARGETS = (128 * KIB, 512 * KIB, 1 * MIB, 2 * MIB)
INITIAL_SHARD_TARGETS = (8 * MIB, 32 * MIB, 128 * MIB, 512 * MIB)

ACCESS_AWARE_HYBRID_REQUEST = StorageCandidateRequest(
    layout=BenchmarkLayout.SHARDED,
    target_chunk_bytes=128 * KIB,
    target_shard_bytes=8 * MIB,
    target_chunk_bytes_by_access=(("eager", 1 * MIB),),
)

INITIAL_DETECTION_MATRIX_WORKLOADS = tuple(
    MatrixWorkload.from_contract(contract)
    for contract in (
        WRITE_MATERIALIZATION_V1,
        PUBLISH_VALIDATE_V1,
        EAGER_FULL_READ_V1,
        WINDOWED_ROWS_READ_V1,
        FULL_SCAN_READ_V1,
    )
) + (
    MatrixWorkload(
        workload_id="palette.detection_read.random_frame_slices.v1",
        phases=("read",),
        access_patterns=("eager", "windowed"),
    ),
    MatrixWorkload(
        workload_id="palette.detection_read.random_observation_ranges.v1",
        phases=("read",),
        access_patterns=("windowed",),
    ),
    MatrixWorkload(
        workload_id="palette.detection_read.sequential_frame_windows.v1",
        phases=("read",),
        access_patterns=("eager", "windowed"),
    ),
)

INITIAL_DETECTION_CORRECTNESS_GATES = {
    "exact_decoded_array_digests": True,
    "exact_logical_schema_and_dtypes": True,
    "frame_row_offsets_contract": True,
    "source_tree_unchanged": True,
    "exclusive_fresh_destinations": True,
    "whole_nonoverlapping_physical_unit_writes": True,
    "direct_metadata_open": True,
    "consolidated_metadata_open": True,
    "crimson_codec_compatibility_before_promotion": True,
}

INITIAL_DETECTION_PERFORMANCE_TOLERANCES = {
    "control": "regular__chunk_1048576",
    "comparison": "same_scale_repetition_host_and_storage_tier",
    "minimum_balanced_repetitions_for_reduction": 5,
    "max_median_write_time_ratio_to_control": 1.25,
    "max_median_publish_time_ratio_to_control": 1.25,
    "max_median_required_read_latency_ratio_to_control": 1.10,
    "max_p95_required_read_latency_ratio_to_control": 1.20,
    "max_peak_rss_ratio_to_control": 1.25,
    "selection_objective": (
        "fewest_payload_objects_among_candidates_passing_all_gates"
    ),
    "http_and_crimson_evidence_required_for_profile_promotion": True,
}


def initial_detection_candidate_requests() -> tuple[StorageCandidateRequest, ...]:
    """Return the declared byte-only regular and sharded sweep."""

    regular = tuple(
        StorageCandidateRequest(
            layout=BenchmarkLayout.REGULAR,
            target_chunk_bytes=chunk_bytes,
        )
        for chunk_bytes in INITIAL_CHUNK_TARGETS
    )
    sharded = tuple(
        StorageCandidateRequest(
            layout=BenchmarkLayout.SHARDED,
            target_chunk_bytes=chunk_bytes,
            target_shard_bytes=shard_bytes,
        )
        for chunk_bytes in INITIAL_CHUNK_TARGETS
        for shard_bytes in INITIAL_SHARD_TARGETS
        if shard_bytes >= chunk_bytes
    )
    return regular + sharded


def selectable_detection_candidate_requests() -> tuple[StorageCandidateRequest, ...]:
    """Return the original sweep plus reviewed access-aware candidates."""

    return initial_detection_candidate_requests() + (ACCESS_AWARE_HYBRID_REQUEST,)


def _canonical_dimensions(scale: BenchmarkScale) -> CanonicalDetectionDimensions:
    dimensions = scale.dimension_map
    expected = ("n_frames", "n_instances", "source_width", "source_height")
    if tuple(dimensions) != expected:
        raise ValueError(
            "Canonical detection benchmark dimensions must be ordered exactly "
            f"as {expected!r}; got {tuple(dimensions)!r}."
        )
    return CanonicalDetectionDimensions(**dimensions)


def _resolve_detection_stage_plan(
    scale: BenchmarkScale,
    request: StorageCandidateRequest,
) -> Mapping[str, object]:
    return plan_detection_benchmark_candidate(
        _canonical_dimensions(scale),
        target_chunk_bytes=request.target_chunk_bytes,
        target_shard_bytes=request.target_shard_bytes,
        layout=request.layout.value,
        target_chunk_bytes_by_access=dict(
            request.target_chunk_bytes_by_access
        ),
    ).as_manifest()


def plan_canonical_detection_benchmark_matrix(
    *,
    matrix_id: str,
    scales: Sequence[BenchmarkScale],
    destination_root: Path,
    repetitions: int = 5,
    repetition_start: int = 0,
    seed: int = 20_260_724,
    occupied_destinations: Iterable[Path] = (),
    candidate_requests: Sequence[StorageCandidateRequest] | None = None,
) -> StorageBenchmarkMatrix:
    """Plan the canonical detection matrix through production storage policy."""

    return plan_storage_benchmark_matrix(
        matrix_id=matrix_id,
        scales=scales,
        candidate_requests=(
            tuple(candidate_requests)
            if candidate_requests is not None
            else initial_detection_candidate_requests()
        ),
        workloads=INITIAL_DETECTION_MATRIX_WORKLOADS,
        repetitions=repetitions,
        repetition_start=repetition_start,
        seed=seed,
        destination_root=destination_root,
        resolve_stage_plan=_resolve_detection_stage_plan,
        occupied_destinations=occupied_destinations,
        correctness_gates=INITIAL_DETECTION_CORRECTNESS_GATES,
        performance_tolerances=INITIAL_DETECTION_PERFORMANCE_TOLERANCES,
    )


__all__ = [
    "ACCESS_AWARE_HYBRID_REQUEST",
    "INITIAL_CHUNK_TARGETS",
    "INITIAL_DETECTION_CORRECTNESS_GATES",
    "INITIAL_DETECTION_MATRIX_WORKLOADS",
    "INITIAL_DETECTION_PERFORMANCE_TOLERANCES",
    "INITIAL_SHARD_TARGETS",
    "initial_detection_candidate_requests",
    "plan_canonical_detection_benchmark_matrix",
    "selectable_detection_candidate_requests",
]
