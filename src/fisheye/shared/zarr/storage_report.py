"""Read-only comparisons between observed and proposed Zarr array layouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from fisheye.shared.zarr.storage_intent import ArrayIntent, StoragePlan
from fisheye.shared.zarr.storage_planner import plan_storage
from fisheye.shared.zarr.storage_profiles import StorageProfile


def _normalize_shape(shape: Any | None) -> tuple[int, ...] | None:
    if shape is None:
        return None
    return tuple(int(value) for value in shape)


@dataclass(frozen=True)
class StorageLayoutComparison:
    """Observed physical layout paired with a proposed storage plan."""

    array_name: str | None
    actual_chunk_shape: tuple[int, ...] | None
    actual_shard_shape: tuple[int, ...] | None
    proposed: StoragePlan

    @property
    def chunk_shape_changes(self) -> bool:
        return self.actual_chunk_shape != self.proposed.chunk_shape

    @property
    def shard_shape_changes(self) -> bool:
        return self.actual_shard_shape != self.proposed.shard_shape

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-safe report record."""

        return {
            "array_name": self.array_name,
            "actual_chunk_shape": (
                list(self.actual_chunk_shape) if self.actual_chunk_shape else None
            ),
            "actual_shard_shape": (
                list(self.actual_shard_shape) if self.actual_shard_shape else None
            ),
            "chunk_shape_changes": self.chunk_shape_changes,
            "shard_shape_changes": self.shard_shape_changes,
            "proposed": self.proposed.as_dict(),
        }


def compare_storage_layout(
    *,
    intent: ArrayIntent,
    profile: StorageProfile,
    actual_chunk_shape: Any | None,
    actual_shard_shape: Any | None,
) -> StorageLayoutComparison:
    """Compare explicit observed shapes with the shared planner's proposal."""

    return StorageLayoutComparison(
        array_name=intent.name,
        actual_chunk_shape=_normalize_shape(actual_chunk_shape),
        actual_shard_shape=_normalize_shape(actual_shard_shape),
        proposed=plan_storage(intent, profile),
    )


def compare_array_storage(
    array: Any,
    *,
    intent: ArrayIntent,
    profile: StorageProfile,
) -> StorageLayoutComparison:
    """Read layout attributes from an array-like object without mutating it."""

    observed_shape = tuple(int(value) for value in array.shape)
    if observed_shape != intent.shape:
        raise ValueError(
            f"Observed shape {observed_shape!r} does not match intent "
            f"shape {intent.shape!r}."
        )
    observed_dtype = np.dtype(array.dtype)
    if observed_dtype != intent.dtype:
        raise ValueError(
            f"Observed dtype {observed_dtype!r} does not match intent "
            f"dtype {intent.dtype!r}."
        )
    return compare_storage_layout(
        intent=intent,
        profile=profile,
        actual_chunk_shape=getattr(array, "chunks", None),
        actual_shard_shape=getattr(array, "shards", None),
    )
