"""Byte-derived immutable storage plans for body-frame snapshots."""

from __future__ import annotations

from dataclasses import dataclass

from fisheye.shared.zarr.array_contracts import ArrayContract
from fisheye.shared.zarr.body_frame_schema import (
    BODY_FRAME_SCHEMA_V1,
    BodyFrameDimensions,
)
from fisheye.shared.zarr.storage_intent import AccessPattern, StoragePlan, WriteMode
from fisheye.shared.zarr.storage_planner import plan_storage
from fisheye.shared.zarr.storage_profiles import PUBLISHED_HTTP_V1, StorageProfile


BODY_FRAME_STORAGE_SCHEMA_ID = "palette.stage_storage.body_frame"
BODY_FRAME_STORAGE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class BodyFrameStorageRule:
    path: str
    access: AccessPattern
    access_unit_semantics: str


BODY_FRAME_STORAGE_RULES = tuple(
    BodyFrameStorageRule(
        path=path,
        access=AccessPattern.EAGER
        if path == "frame_row_offsets"
        else AccessPattern.WINDOWED,
        access_unit_semantics=(
            "complete_retained_frame_boundary_index"
            if path == "frame_row_offsets"
            else "complete_body_frame_observation_row"
        ),
    )
    for path in BODY_FRAME_SCHEMA_V1.binding_paths
)


@dataclass(frozen=True)
class BodyFrameStoragePlanEntry:
    rule: BodyFrameStorageRule
    plan: StoragePlan

    def as_manifest(self) -> dict[str, object]:
        return {
            "path": self.rule.path,
            "access_unit_semantics": self.rule.access_unit_semantics,
            "metadata_object_count": 1,
            "estimated_total_objects": 1 + self.plan.estimated_payload_objects,
            "plan": self.plan.as_dict(),
        }


@dataclass(frozen=True)
class BodyFrameStoragePlanSet:
    dimensions: BodyFrameDimensions
    profile: StorageProfile
    entries: tuple[BodyFrameStoragePlanEntry, ...]

    def __post_init__(self) -> None:
        if (
            tuple(entry.rule.path for entry in self.entries)
            != BODY_FRAME_SCHEMA_V1.binding_paths
        ):
            raise ValueError(
                "Body-frame storage plans must match schema order exactly."
            )
        for entry in self.entries:
            plan = entry.plan
            if (
                plan.array_name != entry.rule.path
                or plan.profile_id != self.profile.profile_id
            ):
                raise ValueError("Body-frame storage plan identity mismatch.")
            if plan.access_pattern != entry.rule.access.value:
                raise ValueError("Body-frame access classification mismatch.")
            if plan.write_mode != WriteMode.IMMUTABLE.value or plan.shard_axes != (0,):
                raise ValueError(
                    "Body-frame publication must be immutable and row-sharded."
                )
            if plan.chunk_shape is None:
                raise ValueError("Body-frame arrays cannot be scalar.")
            trailing = tuple(max(1, value) for value in plan.logical_shape[1:])
            if plan.chunk_shape[1:] != trailing:
                raise ValueError("Body-frame chunks must preserve complete rows.")
            if plan.shard_shape is not None:
                if any(
                    shard % chunk
                    for shard, chunk in zip(
                        plan.shard_shape, plan.chunk_shape, strict=True
                    )
                ):
                    raise ValueError("Body-frame shards must contain whole chunks.")
                if plan.write_ownership != "whole_shard_single_writer":
                    raise ValueError(
                        "Sharded body-frame writes require whole-shard ownership."
                    )
            elif plan.write_ownership != "single_writer_immutable_materialization":
                raise ValueError(
                    "Unsharded body-frame publication requires one writer."
                )

    @property
    def estimated_logical_nbytes(self) -> int:
        return sum(entry.plan.logical_nbytes for entry in self.entries)

    @property
    def estimated_payload_objects(self) -> int:
        return sum(entry.plan.estimated_payload_objects for entry in self.entries)

    @property
    def estimated_stage_objects(self) -> int:
        return self.estimated_payload_objects + len(self.entries) + 2

    def as_manifest(self) -> dict[str, object]:
        return {
            "schema_id": BODY_FRAME_STORAGE_SCHEMA_ID,
            "schema_version": BODY_FRAME_STORAGE_SCHEMA_VERSION,
            "logical_stage_schema": {
                "id": BODY_FRAME_SCHEMA_V1.schema_id,
                "version": BODY_FRAME_SCHEMA_V1.schema_version,
            },
            "dimensions": self.dimensions.as_manifest(),
            "storage_profile": self.profile.as_manifest(),
            "object_estimate": {
                "logical_nbytes": self.estimated_logical_nbytes,
                "inner_chunk_count": sum(
                    entry.plan.estimated_chunk_count for entry in self.entries
                ),
                "payload_objects": self.estimated_payload_objects,
                "sharded_arrays": sum(entry.plan.is_sharded for entry in self.entries),
                "array_metadata_objects": len(self.entries),
                "group_metadata_objects": 2,
                "stage_objects": self.estimated_stage_objects,
                "fill_elision_note": "payload count is a conservative populated-object estimate",
            },
            "metadata_open_contract": {
                "published": "validated_consolidated_root",
                "mutable_or_in_progress": "direct_metadata",
                "archive_root_request_count_target": 1,
            },
            "write_partition_contract": {
                "axis": 0,
                "sharded": "one writer owns each complete outer shard",
                "unsharded": "one writer owns the complete inner chunk grid",
                "partial_physical_unit_writes": "forbidden",
            },
            "arrays": [entry.as_manifest() for entry in self.entries],
        }


def _concrete_shape(
    contract: ArrayContract, dimensions: BodyFrameDimensions
) -> tuple[int, ...]:
    values = dimensions.contract_dimensions
    return tuple(
        axis if isinstance(axis, int) else values[axis]
        for axis in contract.shape_template
    )


def plan_body_frame_storage(
    dimensions: BodyFrameDimensions,
    *,
    profile: StorageProfile = PUBLISHED_HTTP_V1,
) -> BodyFrameStoragePlanSet:
    rules = {rule.path: rule for rule in BODY_FRAME_STORAGE_RULES}
    if tuple(rules) != BODY_FRAME_SCHEMA_V1.binding_paths:
        raise ValueError("Body-frame storage rules do not match the schema.")
    entries: list[BodyFrameStoragePlanEntry] = []
    for binding in BODY_FRAME_SCHEMA_V1.bindings:
        rule = rules[binding.path]
        contract = BODY_FRAME_SCHEMA_V1.contracts.resolve(
            binding.contract_id, binding.contract_version
        )
        shape = _concrete_shape(contract, dimensions)
        intent = contract.storage_intent(
            shape=shape,
            access=rule.access,
            write_mode=WriteMode.IMMUTABLE,
            access_unit_shape=(1, *shape[1:]),
            growth_axis=0,
            shard_axes=(0,),
            whole_shard_writes=True,
            name=binding.path,
            dimensions=dimensions.contract_dimensions,
        )
        entries.append(
            BodyFrameStoragePlanEntry(rule=rule, plan=plan_storage(intent, profile))
        )
    return BodyFrameStoragePlanSet(
        dimensions=dimensions, profile=profile, entries=tuple(entries)
    )


__all__ = [
    "BODY_FRAME_STORAGE_RULES",
    "BODY_FRAME_STORAGE_SCHEMA_ID",
    "BODY_FRAME_STORAGE_SCHEMA_VERSION",
    "BodyFrameStoragePlanEntry",
    "BodyFrameStoragePlanSet",
    "BodyFrameStorageRule",
    "plan_body_frame_storage",
]
