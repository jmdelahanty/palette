"""Logical storage intent and resolved physical-plan types.

These types describe how an array is read and written without importing Zarr.
The physical planner can therefore be tested independently of filesystem and
codec behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import prod
from typing import Any

import numpy as np


STORAGE_POLICY_VERSION = "palette.storage_planner.v1"


class AccessPattern(str, Enum):
    """How consumers normally access an array."""

    EAGER = "eager"
    WINDOWED = "windowed"
    PER_ROW = "per_row"
    INDEXED = "indexed"


class WriteMode(str, Enum):
    """How an array may change during its lifecycle."""

    RANDOM_UPDATE = "random_update"
    APPEND_ONLY = "append_only"
    IMMUTABLE = "immutable"


@dataclass(frozen=True)
class ArrayIntent:
    """Logical array facts needed to derive a physical storage layout.

    ``access_unit_shape`` is the smallest record that should remain intact in
    one inner chunk. The planner scales that unit along ``growth_axis``. For
    example, ``(1, 5, 2)`` preserves a full keypoint record while
    ``(1, 1, 512, 512)`` preserves one mask component plane.

    ``shard_axes`` controls which chunk-grid axes an immutable outer shard may
    combine. It defaults to every axis, starting with ``growth_axis``.
    """

    shape: tuple[int, ...]
    dtype: Any
    access: AccessPattern
    write_mode: WriteMode
    logical_schema_id: str | None = None
    logical_schema_version: int | None = None
    access_unit_shape: tuple[int, ...] | None = None
    growth_axis: int = 0
    shard_axes: tuple[int, ...] | None = None
    logical_itemsize_bytes: int | None = None
    whole_shard_writes: bool = False
    name: str | None = None

    def __post_init__(self) -> None:
        shape = tuple(int(value) for value in self.shape)
        if any(value < 0 for value in shape):
            raise ValueError(
                f"Array shape cannot contain negative dimensions: {shape!r}."
            )
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "access", AccessPattern(self.access))
        object.__setattr__(self, "write_mode", WriteMode(self.write_mode))

        if (self.logical_schema_id is None) != (self.logical_schema_version is None):
            raise ValueError(
                "logical_schema_id and logical_schema_version must be provided together."
            )
        if self.logical_schema_id is not None:
            schema_id = str(self.logical_schema_id).strip()
            if not schema_id:
                raise ValueError("logical_schema_id cannot be empty.")
            if (
                type(self.logical_schema_version) is not int
                or self.logical_schema_version <= 0
            ):
                raise ValueError(
                    "logical_schema_version must be a positive exact integer."
                )
            object.__setattr__(self, "logical_schema_id", schema_id)

        dtype = np.dtype(self.dtype)
        object.__setattr__(self, "dtype", dtype)

        if self.logical_itemsize_bytes is None:
            if dtype.hasobject:
                raise ValueError(
                    "Object and variable-width representations require "
                    "logical_itemsize_bytes."
                )
            itemsize = int(dtype.itemsize)
        else:
            itemsize = int(self.logical_itemsize_bytes)
        if itemsize <= 0:
            raise ValueError("logical_itemsize_bytes must be positive.")
        object.__setattr__(self, "logical_itemsize_bytes", itemsize)

        rank = len(shape)
        if rank == 0:
            if self.access_unit_shape not in (None, ()):
                raise ValueError("Scalar arrays must use an empty access-unit shape.")
            if self.shard_axes not in (None, ()):
                raise ValueError("Scalar arrays cannot declare shard axes.")
            object.__setattr__(self, "access_unit_shape", ())
            object.__setattr__(self, "shard_axes", ())
            return

        growth_axis = int(self.growth_axis)
        if not 0 <= growth_axis < rank:
            raise ValueError(
                f"growth_axis must address shape rank {rank}; got {growth_axis}."
            )
        object.__setattr__(self, "growth_axis", growth_axis)

        if self.access_unit_shape is None:
            access_unit = tuple(
                1 if axis == growth_axis else max(1, dimension)
                for axis, dimension in enumerate(shape)
            )
        else:
            access_unit = tuple(int(value) for value in self.access_unit_shape)
        if len(access_unit) != rank:
            raise ValueError(
                "access_unit_shape must have the same rank as shape; "
                f"got shape={shape!r}, access_unit_shape={access_unit!r}."
            )
        if any(value <= 0 for value in access_unit):
            raise ValueError("access_unit_shape dimensions must be positive.")
        for axis, (unit, dimension) in enumerate(zip(access_unit, shape)):
            if unit > max(1, dimension):
                raise ValueError(
                    f"Access unit {unit} exceeds dimension {dimension} on axis {axis}."
                )
        object.__setattr__(self, "access_unit_shape", access_unit)

        if self.shard_axes is None:
            shard_axes = (growth_axis,) + tuple(
                axis for axis in range(rank) if axis != growth_axis
            )
        else:
            shard_axes = tuple(int(axis) for axis in self.shard_axes)
        if len(set(shard_axes)) != len(shard_axes):
            raise ValueError(f"shard_axes cannot contain duplicates: {shard_axes!r}.")
        if any(axis < 0 or axis >= rank for axis in shard_axes):
            raise ValueError(
                f"shard_axes must address shape rank {rank}; got {shard_axes!r}."
            )
        object.__setattr__(self, "shard_axes", shard_axes)

        if self.whole_shard_writes and self.write_mode is WriteMode.RANDOM_UPDATE:
            raise ValueError(
                "Random-update arrays cannot opt into whole-shard append ownership."
            )

    @property
    def itemsize_bytes(self) -> int:
        """Return the encoded logical bytes per scalar element."""

        return int(self.logical_itemsize_bytes)

    @property
    def logical_nbytes(self) -> int:
        """Return total uncompressed logical bytes."""

        return self.itemsize_bytes * prod(self.shape)

    @property
    def access_unit_nbytes(self) -> int:
        """Return uncompressed bytes in one declared access unit."""

        return self.itemsize_bytes * prod(self.access_unit_shape)


@dataclass(frozen=True)
class StoragePlan:
    """Resolved chunks, shards, estimates, and rationale for one array."""

    policy_version: str
    profile_id: str
    codec_profile_id: str
    array_name: str | None
    logical_schema_id: str | None
    logical_schema_version: int | None
    logical_shape: tuple[int, ...]
    logical_dtype: str
    access_pattern: str
    write_mode: str
    chunk_shape: tuple[int, ...] | None
    shard_shape: tuple[int, ...] | None
    logical_nbytes: int
    access_unit_nbytes: int
    chunk_nbytes: int
    shard_nbytes: int | None
    chunk_grid_shape: tuple[int, ...]
    estimated_chunk_count: int
    estimated_payload_objects: int
    object_budget_satisfied: bool
    shard_byte_budget_satisfied: bool
    write_ownership: str
    rationale: tuple[str, ...]

    @property
    def is_sharded(self) -> bool:
        """Return whether indexed outer sharding is planned."""

        return self.shard_shape is not None

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-safe resolved storage contract."""

        return {
            "schema_id": "palette.array_storage_contract",
            "schema_version": 1,
            "policy_version": self.policy_version,
            "profile_id": self.profile_id,
            "codec_profile_id": self.codec_profile_id,
            "array_name": self.array_name,
            "logical_schema_id": self.logical_schema_id,
            "logical_schema_version": self.logical_schema_version,
            "logical_shape": list(self.logical_shape),
            "logical_dtype": self.logical_dtype,
            "access_pattern": self.access_pattern,
            "write_mode": self.write_mode,
            "chunk_shape": list(self.chunk_shape) if self.chunk_shape else None,
            "shard_shape": list(self.shard_shape) if self.shard_shape else None,
            "logical_nbytes": self.logical_nbytes,
            "access_unit_nbytes": self.access_unit_nbytes,
            "chunk_nbytes": self.chunk_nbytes,
            "shard_nbytes": self.shard_nbytes,
            "chunk_grid_shape": list(self.chunk_grid_shape),
            "estimated_chunk_count": self.estimated_chunk_count,
            "estimated_payload_objects": self.estimated_payload_objects,
            "object_budget_satisfied": self.object_budget_satisfied,
            "shard_byte_budget_satisfied": self.shard_byte_budget_satisfied,
            "write_ownership": self.write_ownership,
            "rationale": list(self.rationale),
        }
