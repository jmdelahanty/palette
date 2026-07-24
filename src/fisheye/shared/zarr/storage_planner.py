"""Pure byte-based chunk and shard planning for Palette Zarr arrays."""

from __future__ import annotations

from math import prod

from fisheye.shared.zarr.storage_intent import (
    STORAGE_POLICY_VERSION,
    AccessPattern,
    ArrayIntent,
    StoragePlan,
    WriteMode,
)
from fisheye.shared.zarr.storage_profiles import StorageProfile


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def _candidate_unit_counts(maximum_units: int) -> tuple[int, ...]:
    candidates: set[int] = {1, maximum_units}
    power = 1
    while power <= maximum_units:
        candidates.add(power)
        power *= 2
    return tuple(sorted(candidates))


def _choose_growth_units(
    *,
    maximum_units: int,
    unit_nbytes: int,
    profile: StorageProfile,
) -> int:
    candidates = _candidate_unit_counts(maximum_units)
    bounded = tuple(
        units
        for units in candidates
        if profile.min_chunk_bytes
        <= units * unit_nbytes
        <= profile.max_chunk_bytes
    )
    choices = bounded or candidates
    return min(
        choices,
        key=lambda units: (
            abs((units * unit_nbytes) - profile.target_chunk_bytes),
            units,
        ),
    )


def _chunk_grid_shape(
    shape: tuple[int, ...], chunk_shape: tuple[int, ...]
) -> tuple[int, ...]:
    return tuple(
        0 if dimension == 0 else _ceil_div(dimension, chunk)
        for dimension, chunk in zip(shape, chunk_shape)
    )


def _plan_chunk_shape(
    intent: ArrayIntent, profile: StorageProfile
) -> tuple[tuple[int, ...] | None, tuple[str, ...]]:
    if not intent.shape:
        return None, ("scalar arrays use one regular payload object",)

    physical_shape = tuple(max(1, dimension) for dimension in intent.shape)
    if (
        intent.access is AccessPattern.EAGER
        and intent.logical_nbytes <= profile.eager_max_bytes
    ):
        return physical_shape, (
            "eager array fits the full-array byte cap and uses one inner chunk",
        )

    growth_axis = intent.growth_axis
    access_unit = intent.access_unit_shape
    maximum_units = max(
        1,
        _ceil_div(physical_shape[growth_axis], access_unit[growth_axis]),
    )
    growth_units = _choose_growth_units(
        maximum_units=maximum_units,
        unit_nbytes=intent.access_unit_nbytes,
        profile=profile,
    )
    chunk_shape = list(access_unit)
    chunk_shape[growth_axis] *= growth_units
    chunk_nbytes = intent.itemsize_bytes * prod(chunk_shape)
    return tuple(chunk_shape), (
        "inner chunk derived from uncompressed access-unit bytes",
        (
            f"selected {growth_units} access units at "
            f"{intent.access_unit_nbytes} bytes each for a "
            f"{chunk_nbytes}-byte inner chunk"
        ),
    )


def _should_shard(intent: ArrayIntent, profile: StorageProfile, chunk_count: int) -> bool:
    if chunk_count <= 1:
        return False
    if intent.write_mode is WriteMode.IMMUTABLE:
        return profile.shard_immutable
    if intent.write_mode is WriteMode.APPEND_ONLY:
        return profile.shard_owned_appends and intent.whole_shard_writes
    return False


def _allocate_shard_multipliers(
    *,
    grid_shape: tuple[int, ...],
    shard_axes: tuple[int, ...],
    desired_chunks: int,
) -> tuple[int, ...]:
    multipliers = [1] * len(grid_shape)
    remaining = max(1, desired_chunks)
    for axis in shard_axes:
        if remaining <= 1:
            break
        capacity = max(1, grid_shape[axis])
        multiplier = min(capacity, remaining)
        multipliers[axis] = multiplier
        remaining = _ceil_div(remaining, multiplier)
    return tuple(multipliers)


def _write_ownership(intent: ArrayIntent, sharded: bool) -> str:
    if sharded:
        return "whole_shard_single_writer"
    if intent.write_mode is WriteMode.RANDOM_UPDATE:
        return "serialized_partial_chunk_updates"
    if intent.write_mode is WriteMode.APPEND_ONLY:
        return "whole_chunk_append_single_writer"
    return "single_writer_immutable_materialization"


def plan_storage(intent: ArrayIntent, profile: StorageProfile) -> StoragePlan:
    """Resolve a deterministic physical layout from logical array facts.

    Calculations use uncompressed encoded bytes. Compression ratios are
    intentionally excluded because content-dependent compressed size is not
    known when an array is created.
    """

    chunk_shape, chunk_rationale = _plan_chunk_shape(intent, profile)
    if chunk_shape is None:
        chunk_nbytes = intent.itemsize_bytes
        grid_shape: tuple[int, ...] = ()
        chunk_count = 1
    else:
        chunk_nbytes = intent.itemsize_bytes * prod(chunk_shape)
        grid_shape = _chunk_grid_shape(intent.shape, chunk_shape)
        chunk_count = prod(grid_shape)

    shard_shape: tuple[int, ...] | None = None
    shard_nbytes: int | None = None
    shard_byte_budget_satisfied = True
    rationale = list(chunk_rationale)

    if _should_shard(intent, profile, chunk_count):
        target_shard_bytes = profile.shard_target_bytes(intent.access)
        chunks_for_byte_target = max(
            2,
            _ceil_div(target_shard_bytes, chunk_nbytes),
        )
        chunks_for_object_target = max(
            2,
            _ceil_div(chunk_count, profile.max_payload_objects),
        )
        desired_chunks = max(chunks_for_byte_target, chunks_for_object_target)
        maximum_chunks_by_bytes = max(2, profile.max_shard_bytes // chunk_nbytes)
        desired_chunks = min(desired_chunks, maximum_chunks_by_bytes, chunk_count)
        multipliers = _allocate_shard_multipliers(
            grid_shape=grid_shape,
            shard_axes=intent.shard_axes,
            desired_chunks=desired_chunks,
        )
        chunks_per_shard = prod(multipliers)
        if chunks_per_shard > 1:
            shard_shape = tuple(
                chunk * multiplier
                for chunk, multiplier in zip(chunk_shape, multipliers)
            )
            shard_nbytes = chunk_nbytes * chunks_per_shard
            shard_byte_budget_satisfied = shard_nbytes <= profile.max_shard_bytes
            rationale.append(
                "immutable or explicitly shard-owned multi-chunk array uses "
                "indexed outer sharding"
            )
            rationale.append(
                f"outer shard combines {chunks_per_shard} inner chunks into "
                f"{shard_nbytes} uncompressed bytes"
            )
        else:
            rationale.append(
                "sharding was eligible but declared shard axes cannot combine chunks"
            )
    elif chunk_count <= 1:
        rationale.append("single-chunk array does not benefit from outer sharding")
    elif intent.write_mode is WriteMode.RANDOM_UPDATE:
        rationale.append("random-update authority remains regular-chunked")
    elif intent.write_mode is WriteMode.APPEND_ONLY and not intent.whole_shard_writes:
        rationale.append("append-only writer lacks whole-shard ownership")
    else:
        rationale.append(f"profile {profile.profile_id} does not require sharding")

    if shard_shape is None:
        estimated_payload_objects = chunk_count
    else:
        shard_grid = tuple(
            _ceil_div(dimension, shard)
            for dimension, shard in zip(intent.shape, shard_shape)
        )
        estimated_payload_objects = prod(shard_grid)

    object_budget_satisfied = (
        estimated_payload_objects <= profile.max_payload_objects
    )
    if not object_budget_satisfied:
        rationale.append(
            f"estimated payload objects exceed profile budget of "
            f"{profile.max_payload_objects}"
        )

    return StoragePlan(
        policy_version=STORAGE_POLICY_VERSION,
        profile_id=profile.profile_id,
        codec_profile_id=profile.codec_profile_id,
        array_name=intent.name,
        logical_schema_id=intent.logical_schema_id,
        logical_schema_version=intent.logical_schema_version,
        logical_shape=intent.shape,
        logical_dtype=str(intent.dtype),
        access_unit_shape=intent.access_unit_shape,
        growth_axis=intent.growth_axis if intent.shape else None,
        shard_axes=intent.shard_axes,
        access_pattern=intent.access.value,
        write_mode=intent.write_mode.value,
        chunk_shape=chunk_shape,
        shard_shape=shard_shape,
        logical_nbytes=intent.logical_nbytes,
        access_unit_nbytes=intent.access_unit_nbytes,
        chunk_nbytes=chunk_nbytes,
        shard_nbytes=shard_nbytes,
        chunk_grid_shape=grid_shape,
        estimated_chunk_count=chunk_count,
        estimated_payload_objects=estimated_payload_objects,
        object_budget_satisfied=object_budget_satisfied,
        shard_byte_budget_satisfied=shard_byte_budget_satisfied,
        write_ownership=_write_ownership(intent, shard_shape is not None),
        rationale=tuple(rationale),
    )
