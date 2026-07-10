"""Method-neutral observation and result contracts for Palette tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


TRACKING_IDENTITY_MODE_KEYED = "instance_key"
TRACKING_IDENTITY_MODE_LEGACY_POSITIONAL = "legacy_positional"


def _optional_vector(
    value: np.ndarray | None,
    *,
    dtype: np.dtype,
    name: str,
    row_count: int,
) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=dtype).reshape(-1)
    if int(array.shape[0]) != int(row_count):
        raise ValueError(
            f"{name} row count ({int(array.shape[0])}) does not match observations ({int(row_count)})."
        )
    return array


@dataclass(frozen=True)
class TrackingObservations:
    """Exact row-aligned observations consumed by one tracking method."""

    arena_ids: np.ndarray
    frame_indices: np.ndarray
    instance_key: np.ndarray | None = None
    source_refined_row_ids: np.ndarray | None = None
    source_detect_row_index: np.ndarray | None = None

    @classmethod
    def from_arrays(
        cls,
        *,
        arena_ids: np.ndarray,
        frame_indices: np.ndarray,
        instance_key: np.ndarray | None = None,
        source_refined_row_ids: np.ndarray | None = None,
        source_detect_row_index: np.ndarray | None = None,
    ) -> "TrackingObservations":
        arenas = np.asarray(arena_ids, dtype=np.int32).reshape(-1)
        frames = np.asarray(frame_indices, dtype=np.int64).reshape(-1)
        if int(arenas.shape[0]) != int(frames.shape[0]):
            raise ValueError("arena_ids and frame_indices must have the same length.")
        row_count = int(frames.shape[0])
        keys = _optional_vector(
            instance_key,
            dtype=np.dtype(np.uint64),
            name="instance_key",
            row_count=row_count,
        )
        if keys is not None and int(np.unique(keys).shape[0]) != row_count:
            raise ValueError("instance_key values must be unique within tracking observations.")
        return cls(
            arena_ids=arenas,
            frame_indices=frames,
            instance_key=keys,
            source_refined_row_ids=_optional_vector(
                source_refined_row_ids,
                dtype=np.dtype(np.int64),
                name="source_refined_row_ids",
                row_count=row_count,
            ),
            source_detect_row_index=_optional_vector(
                source_detect_row_index,
                dtype=np.dtype(np.int32),
                name="source_detect_row_index",
                row_count=row_count,
            ),
        )

    @property
    def row_count(self) -> int:
        return int(self.frame_indices.shape[0])

    @property
    def identity_mode(self) -> str:
        if self.instance_key is not None:
            return TRACKING_IDENTITY_MODE_KEYED
        return TRACKING_IDENTITY_MODE_LEGACY_POSITIONAL


@dataclass(frozen=True)
class TrackingResult:
    """Shared output contract written by every Palette tracking method."""

    method: str
    track_ids: np.ndarray
    arena_ids: np.ndarray
    frame_indices: np.ndarray
    source_row_indices: np.ndarray
    track_ids_present: np.ndarray
    track_arena_ids: np.ndarray
    n_unassigned_rows: int
    tracking_confidence: np.ndarray | None = None
    tracking_status: np.ndarray | None = None
    association_cost: np.ndarray | None = None
    summary: Mapping[str, object] | None = None

    def validate_against(self, observations: TrackingObservations) -> None:
        """Validate common row and track axes independently of method."""

        row_count = observations.row_count
        row_arrays = {
            "track_ids": self.track_ids,
            "arena_ids": self.arena_ids,
            "frame_indices": self.frame_indices,
            "source_row_indices": self.source_row_indices,
            "tracking_confidence": self.tracking_confidence,
            "tracking_status": self.tracking_status,
            "association_cost": self.association_cost,
        }
        for name, value in row_arrays.items():
            if value is None:
                continue
            if np.asarray(value).ndim != 1 or int(np.asarray(value).shape[0]) != row_count:
                raise ValueError(f"Tracking result {name} must have shape ({row_count},).")
        track_ids_present = np.asarray(self.track_ids_present).reshape(-1)
        track_arena_ids = np.asarray(self.track_arena_ids).reshape(-1)
        if track_ids_present.shape != track_arena_ids.shape:
            raise ValueError("track_ids_present and track_arena_ids must have the same shape.")
        if not np.array_equal(
            np.asarray(self.source_row_indices, dtype=np.int64),
            np.arange(row_count, dtype=np.int64),
        ):
            raise ValueError("source_row_indices must address the exact observation rowset in order.")
        if not np.array_equal(np.asarray(self.arena_ids, dtype=np.int32), observations.arena_ids):
            raise ValueError("Tracking result arena_ids do not match the input observations.")
        if not np.array_equal(np.asarray(self.frame_indices, dtype=np.int64), observations.frame_indices):
            raise ValueError("Tracking result frame_indices do not match the input observations.")


__all__ = [
    "TRACKING_IDENTITY_MODE_KEYED",
    "TRACKING_IDENTITY_MODE_LEGACY_POSITIONAL",
    "TrackingObservations",
    "TrackingResult",
]
