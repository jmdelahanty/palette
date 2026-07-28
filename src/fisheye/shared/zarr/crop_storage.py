"""Byte-derived immutable storage plans for crop geometry observations.

The module classifies access and write ownership only.  It does not import
Zarr, create arrays, or change crop selectors.
"""

from __future__ import annotations

from dataclasses import dataclass

from fisheye.shared.zarr.array_contracts import ArrayContract
from fisheye.shared.zarr.crop_schema import (
    CROP_GEOMETRY_SCHEMA_V1,
    CropDimensions,
)
from fisheye.shared.zarr.storage_intent import (
    AccessPattern,
    StoragePlan,
    WriteMode,
)
from fisheye.shared.zarr.storage_planner import plan_storage
from fisheye.shared.zarr.storage_profiles import (
    PUBLISHED_HTTP_V1,
    StorageProfile,
)


CROP_GEOMETRY_STORAGE_SCHEMA_ID = "palette.stage_storage.crop_geometry"
CROP_GEOMETRY_STORAGE_SCHEMA_VERSION = 1

_ARRAY_METADATA_OBJECTS_PER_ARRAY = 1
_STAGE_GROUP_METADATA_OBJECTS = 2


@dataclass(frozen=True)
class CropStorageRule:
    """Stable access classification for one crop geometry path."""

    path: str
    access: AccessPattern
    access_unit_semantics: str
    representative_request: str

    def __post_init__(self) -> None:
        path = str(self.path).strip().strip("/")
        if not path:
            raise ValueError("Crop storage rule path cannot be empty.")
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "access", AccessPattern(self.access))
        if not self.access_unit_semantics.strip():
            raise ValueError("access_unit_semantics cannot be empty.")
        if not self.representative_request.strip():
            raise ValueError("representative_request cannot be empty.")


def _row_rule(path: str) -> CropStorageRule:
    return CropStorageRule(
        path=path,
        access=AccessPattern.WINDOWED,
        access_unit_semantics="one_complete_crop_observation_row",
        representative_request="contiguous_frame_or_downstream_batch_row_range",
    )


CROP_GEOMETRY_STORAGE_RULES = tuple(
    CropStorageRule(
        path=path,
        access=AccessPattern.EAGER,
        access_unit_semantics="one_frame_boundary_offset",
        representative_request="whole_retained_frame_index",
    )
    if path == "frame_row_offsets"
    else _row_rule(path)
    for path in CROP_GEOMETRY_SCHEMA_V1.binding_paths
)


@dataclass(frozen=True)
class CropStoragePlanEntry:
    rule: CropStorageRule
    plan: StoragePlan

    def as_manifest(self) -> dict[str, object]:
        return {
            "path": self.rule.path,
            "access_unit_semantics": self.rule.access_unit_semantics,
            "representative_request": self.rule.representative_request,
            "metadata_object_count": _ARRAY_METADATA_OBJECTS_PER_ARRAY,
            "estimated_total_objects": (
                _ARRAY_METADATA_OBJECTS_PER_ARRAY
                + self.plan.estimated_payload_objects
            ),
            "plan": self.plan.as_dict(),
        }


@dataclass(frozen=True)
class CropGeometryStoragePlanSet:
    """Schema-complete storage plans for one concrete crop snapshot."""

    dimensions: CropDimensions
    profile: StorageProfile
    entries: tuple[CropStoragePlanEntry, ...]

    def __post_init__(self) -> None:
        expected_paths = CROP_GEOMETRY_SCHEMA_V1.binding_paths
        actual_paths = tuple(entry.rule.path for entry in self.entries)
        if actual_paths != expected_paths:
            raise ValueError(
                "Crop storage plans must match schema binding order exactly; "
                f"expected {expected_paths!r}, got {actual_paths!r}."
            )
        for entry in self.entries:
            plan = entry.plan
            if plan.array_name != entry.rule.path:
                raise ValueError(
                    f"Storage plan name does not match {entry.rule.path!r}."
                )
            if plan.profile_id != self.profile.profile_id:
                raise ValueError(
                    f"Storage profile mismatch at {entry.rule.path!r}."
                )
            if plan.access_pattern != entry.rule.access.value:
                raise ValueError(
                    f"Storage access mismatch at {entry.rule.path!r}."
                )
            if plan.write_mode != WriteMode.IMMUTABLE.value:
                raise ValueError(
                    f"Published crop geometry must be immutable at {entry.rule.path!r}."
                )
            if plan.shard_axes != (0,):
                raise ValueError(
                    f"Crop geometry may shard only along rows at {entry.rule.path!r}."
                )
            self._require_complete_rows(entry)

    @staticmethod
    def _require_complete_rows(entry: CropStoragePlanEntry) -> None:
        plan = entry.plan
        chunk = plan.chunk_shape
        if chunk is None:
            raise ValueError(f"Crop geometry cannot be scalar at {entry.rule.path!r}.")
        expected_trailing = tuple(max(1, value) for value in plan.logical_shape[1:])
        if chunk[1:] != expected_trailing:
            raise ValueError(
                f"Crop chunks must preserve complete rows at {entry.rule.path!r}."
            )
        if plan.shard_shape is not None:
            if any(
                shard_axis % chunk_axis
                for shard_axis, chunk_axis in zip(plan.shard_shape, chunk)
            ):
                raise ValueError(
                    f"Shard shape must contain whole chunks at {entry.rule.path!r}."
                )
            if plan.write_ownership != "whole_shard_single_writer":
                raise ValueError(
                    f"Sharded crop publication lacks whole-shard ownership at "
                    f"{entry.rule.path!r}."
                )
        elif plan.write_ownership != "single_writer_immutable_materialization":
            raise ValueError(
                f"Unsharded crop publication must have one writer at "
                f"{entry.rule.path!r}."
            )

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
    def estimated_array_metadata_objects(self) -> int:
        return len(self.entries) * _ARRAY_METADATA_OBJECTS_PER_ARRAY

    @property
    def estimated_stage_objects(self) -> int:
        return (
            self.estimated_payload_objects
            + self.estimated_array_metadata_objects
            + _STAGE_GROUP_METADATA_OBJECTS
        )

    @property
    def sharded_array_count(self) -> int:
        return sum(entry.plan.is_sharded for entry in self.entries)

    def as_manifest(self) -> dict[str, object]:
        return {
            "schema_id": CROP_GEOMETRY_STORAGE_SCHEMA_ID,
            "schema_version": CROP_GEOMETRY_STORAGE_SCHEMA_VERSION,
            "logical_stage_schema": {
                "id": CROP_GEOMETRY_SCHEMA_V1.schema_id,
                "version": CROP_GEOMETRY_SCHEMA_V1.schema_version,
            },
            "dimensions": self.dimensions.as_manifest(),
            "storage_profile": self.profile.as_manifest(),
            "object_estimate": {
                "logical_nbytes": self.estimated_logical_nbytes,
                "inner_chunk_count": self.estimated_inner_chunk_count,
                "payload_objects": self.estimated_payload_objects,
                "sharded_arrays": self.sharded_array_count,
                "array_metadata_objects": self.estimated_array_metadata_objects,
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
    dimensions: CropDimensions,
) -> tuple[int, ...]:
    concrete: list[int] = []
    values = dimensions.contract_dimensions
    for dimension in contract.shape_template:
        if isinstance(dimension, int):
            concrete.append(dimension)
        else:
            try:
                concrete.append(values[dimension])
            except KeyError as exc:
                raise ValueError(f"Unknown crop dimension {dimension!r}.") from exc
    return tuple(concrete)


def plan_crop_geometry_storage(
    dimensions: CropDimensions,
    *,
    profile: StorageProfile = PUBLISHED_HTTP_V1,
) -> CropGeometryStoragePlanSet:
    """Resolve every crop path from access shape and uncompressed bytes."""

    rules = {rule.path: rule for rule in CROP_GEOMETRY_STORAGE_RULES}
    if len(rules) != len(CROP_GEOMETRY_STORAGE_RULES):
        raise ValueError("Crop storage rule paths must be unique.")
    if tuple(rules) != CROP_GEOMETRY_SCHEMA_V1.binding_paths:
        raise ValueError("Crop storage rules must match schema bindings exactly.")

    entries: list[CropStoragePlanEntry] = []
    for binding in CROP_GEOMETRY_SCHEMA_V1.bindings:
        rule = rules[binding.path]
        contract = CROP_GEOMETRY_SCHEMA_V1.contracts.resolve(
            binding.contract_id,
            binding.contract_version,
        )
        shape = _concrete_shape(contract, dimensions)
        intent = contract.storage_intent(
            shape=shape,
            access=rule.access,
            write_mode=WriteMode.IMMUTABLE,
            access_unit_shape=(1, *shape[1:]),
            growth_axis=0,
            shard_axes=(0,),
            name=binding.path,
            dimensions=dimensions.contract_dimensions,
        )
        entries.append(
            CropStoragePlanEntry(
                rule=rule,
                plan=plan_storage(intent, profile),
            )
        )

    return CropGeometryStoragePlanSet(
        dimensions=dimensions,
        profile=profile,
        entries=tuple(entries),
    )


__all__ = [
    "CROP_GEOMETRY_STORAGE_RULES",
    "CROP_GEOMETRY_STORAGE_SCHEMA_ID",
    "CROP_GEOMETRY_STORAGE_SCHEMA_VERSION",
    "CropGeometryStoragePlanSet",
    "CropStoragePlanEntry",
    "CropStorageRule",
    "plan_crop_geometry_storage",
]
