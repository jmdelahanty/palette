"""One byte-only planner adapter shared by detection benchmark runtimes."""

from __future__ import annotations

from typing import Mapping

from fisheye.shared.zarr.storage_intent import AccessPattern

from fisheye.shared.zarr.detection_schema import CanonicalDetectionDimensions
from fisheye.shared.zarr.detection_storage import (
    CanonicalDetectionStoragePlanSet,
    plan_canonical_detection_storage,
)
from fisheye.shared.zarr.storage_profiles import (
    PUBLISHED_HTTP_V1,
    make_benchmark_storage_profile,
)


def plan_detection_benchmark_candidate(
    dimensions: CanonicalDetectionDimensions,
    *,
    target_chunk_bytes: int,
    target_shard_bytes: int | None,
    layout: str,
    target_chunk_bytes_by_access: Mapping[AccessPattern | str, int] | None = None,
) -> CanonicalDetectionStoragePlanSet:
    """Resolve one regular or sharded candidate without row overrides."""

    resolved_layout = str(layout)
    if resolved_layout not in {"regular", "sharded"}:
        raise ValueError("Detection benchmark layout must be regular or sharded.")
    chunk_bytes = int(target_chunk_bytes)
    if chunk_bytes <= 0:
        raise ValueError("Detection benchmark chunk target must be positive.")
    if resolved_layout == "sharded" and target_shard_bytes is None:
        raise ValueError("Sharded detection benchmark candidates require a shard target.")
    shard_bytes = (
        int(target_shard_bytes)
        if target_shard_bytes is not None
        else max(PUBLISHED_HTTP_V1.target_shard_bytes, chunk_bytes)
    )
    profile = make_benchmark_storage_profile(
        target_chunk_bytes=chunk_bytes,
        target_shard_bytes=shard_bytes,
        shard_immutable=resolved_layout == "sharded",
        target_chunk_bytes_by_access=target_chunk_bytes_by_access,
    )
    return plan_canonical_detection_storage(dimensions, profile=profile)


__all__ = ["plan_detection_benchmark_candidate"]
