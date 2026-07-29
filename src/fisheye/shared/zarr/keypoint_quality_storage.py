"""Byte-derived immutable storage plans for keypoint-quality snapshots."""

from __future__ import annotations

from dataclasses import dataclass

from fisheye.shared.zarr.array_contracts import ArrayContract
from fisheye.shared.zarr.keypoint_quality_schema import (
    KEYPOINT_QUALITY_SCHEMA_V1,
    KeypointQualityDimensions,
)
from fisheye.shared.zarr.storage_intent import (
    AccessPattern,
    StoragePlan,
    WriteMode,
)
from fisheye.shared.zarr.storage_planner import plan_storage
from fisheye.shared.zarr.storage_profiles import PUBLISHED_HTTP_V1, StorageProfile


KEYPOINT_QUALITY_STORAGE_SCHEMA_ID = "palette.stage_storage.keypoint_quality"
KEYPOINT_QUALITY_STORAGE_SCHEMA_VERSION = 1
_ARRAY_METADATA_OBJECTS_PER_ARRAY = 1
_STAGE_GROUP_METADATA_OBJECTS = 2


@dataclass(frozen=True)
class KeypointQualityStorageRule:
    path: str
    access: AccessPattern
    access_unit_semantics: str

    def __post_init__(self) -> None:
        path = str(self.path).strip().strip("/")
        if not path:
            raise ValueError("Keypoint-quality storage path cannot be empty.")
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "access", AccessPattern(self.access))
        if not str(self.access_unit_semantics).strip():
            raise ValueError("access_unit_semantics cannot be empty.")


KEYPOINT_QUALITY_STORAGE_RULES = tuple(
    KeypointQualityStorageRule(
        path=path,
        access=(
            AccessPattern.EAGER
            if path == "frame_row_offsets"
            else AccessPattern.WINDOWED
        ),
        access_unit_semantics=(
            "complete_retained_frame_boundary_index"
            if path == "frame_row_offsets"
            else "complete_quality_observation_row"
        ),
    )
    for path in KEYPOINT_QUALITY_SCHEMA_V1.binding_paths
)


@dataclass(frozen=True)
class KeypointQualityStoragePlanEntry:
    rule: KeypointQualityStorageRule
    plan: StoragePlan

    def as_manifest(self) -> dict[str, object]:
        return {
            "path": self.rule.path,
            "access_unit_semantics": self.rule.access_unit_semantics,
            "metadata_object_count": _ARRAY_METADATA_OBJECTS_PER_ARRAY,
            "estimated_total_objects": (
                _ARRAY_METADATA_OBJECTS_PER_ARRAY
                + self.plan.estimated_payload_objects
            ),
            "plan": self.plan.as_dict(),
        }


@dataclass(frozen=True)
class KeypointQualityStoragePlanSet:
    dimensions: KeypointQualityDimensions
    profile: StorageProfile
    entries: tuple[KeypointQualityStoragePlanEntry, ...]

    def __post_init__(self) -> None:
        expected = KEYPOINT_QUALITY_SCHEMA_V1.binding_paths
        observed = tuple(entry.rule.path for entry in self.entries)
        if observed != expected:
            raise ValueError(
                "Keypoint-quality storage plans must match schema order exactly."
            )
        for entry in self.entries:
            plan = entry.plan
            if plan.array_name != entry.rule.path:
                raise ValueError("Storage plan array name mismatch.")
            if plan.profile_id != self.profile.profile_id:
                raise ValueError("Storage plan profile mismatch.")
            if plan.access_pattern != entry.rule.access.value:
                raise ValueError("Storage plan access classification mismatch.")
            if plan.write_mode != WriteMode.IMMUTABLE.value:
                raise ValueError("Published keypoint quality must be immutable.")
            if plan.shard_axes != (0,):
                raise ValueError("Keypoint quality may shard only along rows.")
            self._require_complete_rows(entry)

    @staticmethod
    def _require_complete_rows(entry: KeypointQualityStoragePlanEntry) -> None:
        plan = entry.plan
        chunk = plan.chunk_shape
        if chunk is None:
            raise ValueError("Keypoint-quality arrays cannot be scalar.")
        expected_trailing = tuple(max(1, value) for value in plan.logical_shape[1:])
        if chunk[1:] != expected_trailing:
            raise ValueError(
                f"Chunks must preserve complete rows at {entry.rule.path!r}."
            )
        if plan.shard_shape is not None:
            if any(
                shard % inner
                for shard, inner in zip(plan.shard_shape, chunk, strict=True)
            ):
                raise ValueError("Shard shapes must contain whole inner chunks.")
            if plan.write_ownership != "whole_shard_single_writer":
                raise ValueError("Sharded quality writes require whole-shard ownership.")
        elif plan.write_ownership != "single_writer_immutable_materialization":
            raise ValueError("Unsharded quality publication requires one writer.")

    @property
    def estimated_logical_nbytes(self) -> int:
        return sum(entry.plan.logical_nbytes for entry in self.entries)

    @property
    def estimated_inner_chunk_count(self) -> int:
        return sum(entry.plan.estimated_chunk_count for entry in self.entries)

    @property
    def estimated_payload_objects(self) -> int:
        return sum(entry.plan.estimated_payload_objects for entry in self.entries)

    @property
    def estimated_stage_objects(self) -> int:
        return (
            self.estimated_payload_objects
            + len(self.entries) * _ARRAY_METADATA_OBJECTS_PER_ARRAY
            + _STAGE_GROUP_METADATA_OBJECTS
        )

    @property
    def sharded_array_count(self) -> int:
        return sum(entry.plan.is_sharded for entry in self.entries)

    def as_manifest(self) -> dict[str, object]:
        return {
            "schema_id": KEYPOINT_QUALITY_STORAGE_SCHEMA_ID,
            "schema_version": KEYPOINT_QUALITY_STORAGE_SCHEMA_VERSION,
            "logical_stage_schema": {
                "id": KEYPOINT_QUALITY_SCHEMA_V1.schema_id,
                "version": KEYPOINT_QUALITY_SCHEMA_V1.schema_version,
            },
            "dimensions": self.dimensions.as_manifest(),
            "storage_profile": self.profile.as_manifest(),
            "object_estimate": {
                "logical_nbytes": self.estimated_logical_nbytes,
                "inner_chunk_count": self.estimated_inner_chunk_count,
                "payload_objects": self.estimated_payload_objects,
                "sharded_arrays": self.sharded_array_count,
                "array_metadata_objects": len(self.entries),
                "group_metadata_objects": _STAGE_GROUP_METADATA_OBJECTS,
                "stage_objects": self.estimated_stage_objects,
                "fill_elision_note": (
                    "payload count is a conservative populated-object estimate"
                ),
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
    contract: ArrayContract,
    dimensions: KeypointQualityDimensions,
) -> tuple[int, ...]:
    values = dimensions.contract_dimensions
    return tuple(
        dimension if isinstance(dimension, int) else values[dimension]
        for dimension in contract.shape_template
    )


def plan_keypoint_quality_storage(
    dimensions: KeypointQualityDimensions,
    *,
    profile: StorageProfile = PUBLISHED_HTTP_V1,
) -> KeypointQualityStoragePlanSet:
    """Plan every quality array from exact dtype, shape, and access class."""

    rules = {rule.path: rule for rule in KEYPOINT_QUALITY_STORAGE_RULES}
    if tuple(rules) != KEYPOINT_QUALITY_SCHEMA_V1.binding_paths:
        raise ValueError("Keypoint-quality storage rules do not match the schema.")
    entries: list[KeypointQualityStoragePlanEntry] = []
    for binding in KEYPOINT_QUALITY_SCHEMA_V1.bindings:
        rule = rules[binding.path]
        contract = KEYPOINT_QUALITY_SCHEMA_V1.contracts.resolve(
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
            KeypointQualityStoragePlanEntry(
                rule=rule,
                plan=plan_storage(intent, profile),
            )
        )
    return KeypointQualityStoragePlanSet(
        dimensions=dimensions,
        profile=profile,
        entries=tuple(entries),
    )


__all__ = [
    "KEYPOINT_QUALITY_STORAGE_RULES",
    "KEYPOINT_QUALITY_STORAGE_SCHEMA_ID",
    "KEYPOINT_QUALITY_STORAGE_SCHEMA_VERSION",
    "KeypointQualityStoragePlanEntry",
    "KeypointQualityStoragePlanSet",
    "KeypointQualityStorageRule",
    "plan_keypoint_quality_storage",
]
