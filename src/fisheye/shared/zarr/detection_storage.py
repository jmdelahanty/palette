"""Physical storage intents for the canonical detection stage.

This module classifies consumer access and lifecycle once, then delegates every
row-depth, chunk, and shard decision to the shared byte-based planner. It does
not import Zarr or create arrays.
"""

from __future__ import annotations

from dataclasses import dataclass

from fisheye.shared.zarr.array_contracts import ArrayContract
from fisheye.shared.zarr.detection_schema import (
    CANONICAL_DETECTION_SCHEMA_V1,
    CanonicalDetectionDimensions,
)
from fisheye.shared.zarr.storage_intent import (
    AccessPattern,
    StoragePlan,
    WriteMode,
)
from fisheye.shared.zarr.storage_planner import plan_storage
from fisheye.shared.zarr.storage_profiles import (
    DETECTION_PUBLISHED_ACCESS_AWARE_V1,
    StorageProfile,
)


CANONICAL_DETECTION_STORAGE_SCHEMA_ID = "palette.stage_storage.canonical_detection"
CANONICAL_DETECTION_STORAGE_SCHEMA_VERSION = 1

_ARRAY_METADATA_OBJECTS_PER_ARRAY = 1
_STAGE_GROUP_METADATA_OBJECTS = 2


@dataclass(frozen=True)
class DetectionStorageRule:
    """Stable access classification for one canonical detection path."""

    path: str
    access: AccessPattern
    access_unit_semantics: str
    representative_request: str

    def __post_init__(self) -> None:
        path = str(self.path).strip().strip("/")
        if not path:
            raise ValueError("Detection storage rule path cannot be empty.")
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "access", AccessPattern(self.access))
        if not self.access_unit_semantics.strip():
            raise ValueError("access_unit_semantics cannot be empty.")
        if not self.representative_request.strip():
            raise ValueError("representative_request cannot be empty.")


def _instance_rule(
    name: str,
    *,
    representative_request: str = "contiguous_frame_or_join_row_range",
) -> DetectionStorageRule:
    return DetectionStorageRule(
        path=f"instances/{name}",
        access=AccessPattern.WINDOWED,
        access_unit_semantics="one_complete_detection_instance_row",
        representative_request=representative_request,
    )


CANONICAL_DETECTION_STORAGE_RULES = (
    _instance_rule("frame_indices"),
    _instance_rule("source_acquisition_frame_index"),
    _instance_rule("instance_key"),
    _instance_rule("bbox_norm_coords"),
    _instance_rule("bbox_img_xyxy"),
    _instance_rule("centers_img_xy"),
    _instance_rule("scores"),
    _instance_rule("class_ids"),
    DetectionStorageRule(
        path="instances/frame_row_offsets",
        access=AccessPattern.EAGER,
        access_unit_semantics="one_frame_boundary_offset",
        representative_request="whole_index_or_two_adjacent_frame_boundaries",
    ),
)


@dataclass(frozen=True)
class DetectionStoragePlanEntry:
    """One canonical path's access rule and resolved physical plan."""

    rule: DetectionStorageRule
    plan: StoragePlan

    def as_manifest(self) -> dict[str, object]:
        return {
            "path": self.rule.path,
            "access_unit_semantics": self.rule.access_unit_semantics,
            "representative_request": self.rule.representative_request,
            "metadata_object_count": _ARRAY_METADATA_OBJECTS_PER_ARRAY,
            "estimated_total_objects": (
                _ARRAY_METADATA_OBJECTS_PER_ARRAY + self.plan.estimated_payload_objects
            ),
            "plan": self.plan.as_dict(),
        }


@dataclass(frozen=True)
class CanonicalDetectionStoragePlanSet:
    """Schema-complete storage plan set for one concrete detection run."""

    dimensions: CanonicalDetectionDimensions
    profile: StorageProfile
    entries: tuple[DetectionStoragePlanEntry, ...]

    def __post_init__(self) -> None:
        expected_paths = CANONICAL_DETECTION_SCHEMA_V1.binding_paths
        actual_paths = tuple(entry.rule.path for entry in self.entries)
        if actual_paths != expected_paths:
            raise ValueError(
                "Detection storage plans must match canonical schema binding "
                f"order exactly; expected {expected_paths!r}, got {actual_paths!r}."
            )

        for entry in self.entries:
            plan = entry.plan
            if plan.array_name != entry.rule.path:
                raise ValueError(
                    f"Storage plan name does not match path {entry.rule.path!r}."
                )
            if plan.profile_id != self.profile.profile_id:
                raise ValueError(
                    f"Storage plan profile mismatch at {entry.rule.path!r}."
                )
            if plan.access_pattern != entry.rule.access.value:
                raise ValueError(
                    f"Storage plan access mismatch at {entry.rule.path!r}."
                )
            if plan.write_mode != WriteMode.IMMUTABLE.value:
                raise ValueError(
                    f"Canonical detection publication must be immutable at "
                    f"{entry.rule.path!r}."
                )
            if plan.shard_axes != (0,):
                raise ValueError(
                    f"Detection storage may shard only along the row axis at "
                    f"{entry.rule.path!r}."
                )
            self._require_whole_physical_units(entry)

    @staticmethod
    def _require_whole_physical_units(entry: DetectionStoragePlanEntry) -> None:
        plan = entry.plan
        chunk = plan.chunk_shape
        if chunk is None:
            raise ValueError(
                f"Canonical detection arrays cannot be scalars: {entry.rule.path!r}."
            )
        if chunk[1:] != tuple(max(1, value) for value in plan.logical_shape[1:]):
            raise ValueError(
                "Detection chunks must preserve complete trailing row axes at "
                f"{entry.rule.path!r}."
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
                    f"Sharded publication lacks whole-shard ownership at "
                    f"{entry.rule.path!r}."
                )
        elif plan.write_ownership != "single_writer_immutable_materialization":
            raise ValueError(
                f"Unsharded immutable publication must use one materializing "
                f"writer at {entry.rule.path!r}."
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
        """Return a JSON-safe stage storage planning report."""

        return {
            "schema_id": CANONICAL_DETECTION_STORAGE_SCHEMA_ID,
            "schema_version": CANONICAL_DETECTION_STORAGE_SCHEMA_VERSION,
            "logical_stage_schema": {
                "id": CANONICAL_DETECTION_SCHEMA_V1.schema_id,
                "version": CANONICAL_DETECTION_SCHEMA_V1.schema_version,
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
    dimensions: CanonicalDetectionDimensions,
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
                raise ValueError(
                    f"Unknown canonical detection dimension {dimension!r}."
                ) from exc
    return tuple(concrete)


def plan_canonical_detection_storage(
    dimensions: CanonicalDetectionDimensions,
    *,
    profile: StorageProfile = DETECTION_PUBLISHED_ACCESS_AWARE_V1,
) -> CanonicalDetectionStoragePlanSet:
    """Resolve every canonical detection path from byte-based policy."""

    rules = {rule.path: rule for rule in CANONICAL_DETECTION_STORAGE_RULES}
    if len(rules) != len(CANONICAL_DETECTION_STORAGE_RULES):
        raise ValueError("Canonical detection storage rule paths must be unique.")
    if tuple(rules) != CANONICAL_DETECTION_SCHEMA_V1.binding_paths:
        raise ValueError(
            "Canonical detection storage rules must match stage bindings exactly."
        )

    entries: list[DetectionStoragePlanEntry] = []
    for binding in CANONICAL_DETECTION_SCHEMA_V1.bindings:
        try:
            rule = rules[binding.path]
        except KeyError as exc:
            raise ValueError(
                f"Missing storage rule for canonical path {binding.path!r}."
            ) from exc
        contract = CANONICAL_DETECTION_SCHEMA_V1.contracts.resolve(
            binding.contract_id,
            binding.contract_version,
        )
        shape = _concrete_shape(contract, dimensions)
        access_unit_shape = (1, *shape[1:])
        intent = contract.storage_intent(
            shape=shape,
            access=rule.access,
            write_mode=WriteMode.IMMUTABLE,
            access_unit_shape=access_unit_shape,
            growth_axis=0,
            shard_axes=(0,),
            name=binding.path,
            dimensions=dimensions.contract_dimensions,
        )
        entries.append(
            DetectionStoragePlanEntry(
                rule=rule,
                plan=plan_storage(intent, profile),
            )
        )

    return CanonicalDetectionStoragePlanSet(
        dimensions=dimensions,
        profile=profile,
        entries=tuple(entries),
    )


__all__ = [
    "CANONICAL_DETECTION_STORAGE_RULES",
    "CANONICAL_DETECTION_STORAGE_SCHEMA_ID",
    "CANONICAL_DETECTION_STORAGE_SCHEMA_VERSION",
    "CanonicalDetectionStoragePlanSet",
    "DetectionStoragePlanEntry",
    "DetectionStorageRule",
    "plan_canonical_detection_storage",
]
