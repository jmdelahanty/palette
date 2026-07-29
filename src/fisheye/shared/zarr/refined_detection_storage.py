"""Byte-budgeted physical storage contract for refined-detection snapshots.

The module classifies every frozen logical binding by consumer access pattern
and delegates chunk/shard row depths to the shared byte planner.  It does not
create arrays or route any production writer.
"""

from __future__ import annotations

from dataclasses import dataclass

from fisheye.shared.zarr.array_contracts import ArrayContract
from fisheye.shared.zarr.codec_profiles import get_codec_profile
from fisheye.shared.zarr.refined_detection_schema import (
    REFINED_DETECTION_INSTANCE_GROUP,
    REFINED_DETECTION_SCHEMA_V1,
    REFINED_DETECTION_SOURCE_GROUP,
    RefinedDetectionDimensions,
)
from fisheye.shared.zarr.storage_intent import (
    AccessPattern,
    StoragePlan,
    WriteMode,
)
from fisheye.shared.zarr.storage_planner import plan_storage
from fisheye.shared.zarr.storage_profiles import (
    DETECTION_PUBLISHED_ACCESS_AWARE_V1,
    DETECTION_REGULAR_ROLLBACK_V1,
    KIB,
    MIB,
    PUBLISHED_HTTP_V1,
    StorageProfile,
    make_benchmark_storage_profile,
)


REFINED_DETECTION_STORAGE_SCHEMA_ID = "palette.stage_storage.refined_detection"
REFINED_DETECTION_STORAGE_SCHEMA_VERSION = 1

_ARRAY_METADATA_OBJECTS_PER_ARRAY = 1
_STAGE_GROUP_METADATA_OBJECTS = 3


# Frozen evidence candidate, not the default passed to the planner and not a
# production-writer promotion.  It reproduces the physical profile that passed
# the canonical-detection compatibility and residency work: narrow windowed /
# indexed payload chunks, a separately eager offset index, and 8 MiB shards.
REFINED_DETECTION_ACCESS_AWARE_CANDIDATE_V1 = make_benchmark_storage_profile(
    base=PUBLISHED_HTTP_V1,
    target_chunk_bytes=128 * KIB,
    target_shard_bytes=8 * MIB,
    shard_immutable=True,
    target_chunk_bytes_by_access={AccessPattern.EAGER: 1 * MIB},
)

REFINED_DETECTION_REGULAR_CONTROL_V1 = make_benchmark_storage_profile(
    base=PUBLISHED_HTTP_V1,
    target_chunk_bytes=1 * MIB,
    target_shard_bytes=32 * MIB,
    shard_immutable=False,
)


def _profile_role(profile: StorageProfile) -> str:
    if profile == DETECTION_PUBLISHED_ACCESS_AWARE_V1:
        return "promoted_detection_snapshot_default"
    if profile == DETECTION_REGULAR_ROLLBACK_V1:
        return "explicit_detection_snapshot_rollback"
    if profile == REFINED_DETECTION_ACCESS_AWARE_CANDIDATE_V1:
        return "unpromoted_access_aware_candidate"
    if profile == REFINED_DETECTION_REGULAR_CONTROL_V1:
        return "paired_unsharded_control"
    if profile == PUBLISHED_HTTP_V1:
        return "generic_shared_sharded_baseline"
    return "caller_supplied_unpromoted_profile"


@dataclass(frozen=True)
class RefinedDetectionStorageRule:
    path: str
    access: AccessPattern
    access_unit_semantics: str
    representative_request: str

    def __post_init__(self) -> None:
        path = str(self.path).strip().strip("/")
        if not path:
            raise ValueError("Refined detection storage path cannot be empty.")
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "access", AccessPattern(self.access))
        if not self.access_unit_semantics.strip():
            raise ValueError("access_unit_semantics cannot be empty.")
        if not self.representative_request.strip():
            raise ValueError("representative_request cannot be empty.")


def _rule_for_path(path: str) -> RefinedDetectionStorageRule:
    if path.endswith("/frame_row_offsets"):
        return RefinedDetectionStorageRule(
            path=path,
            access=AccessPattern.EAGER,
            access_unit_semantics="one_frame_boundary_offset",
            representative_request=(
                "whole_selected_table_index_or_two_adjacent_frame_boundaries"
            ),
        )
    if path.startswith(f"{REFINED_DETECTION_SOURCE_GROUP}/"):
        return RefinedDetectionStorageRule(
            path=path,
            access=AccessPattern.INDEXED,
            access_unit_semantics="one_complete_source_candidate_row",
            representative_request="source_audit_row_or_frame_resolved_row_range",
        )
    if path.startswith(f"{REFINED_DETECTION_INSTANCE_GROUP}/"):
        return RefinedDetectionStorageRule(
            path=path,
            access=AccessPattern.WINDOWED,
            access_unit_semantics="one_complete_refined_instance_row",
            representative_request="frame_page_or_contiguous_instance_row_range",
        )
    raise ValueError(f"Unknown refined detection binding path: {path!r}.")


@dataclass(frozen=True)
class RefinedDetectionStoragePlanEntry:
    rule: RefinedDetectionStorageRule
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
class RefinedDetectionStoragePlanSet:
    dimensions: RefinedDetectionDimensions
    profile: StorageProfile
    entries: tuple[RefinedDetectionStoragePlanEntry, ...]

    def __post_init__(self) -> None:
        expected = REFINED_DETECTION_SCHEMA_V1.binding_paths_for(self.dimensions)
        actual = tuple(entry.rule.path for entry in self.entries)
        if actual != expected:
            raise ValueError(
                "Refined detection storage plans must match the active logical "
                f"binding order exactly; expected {expected!r}, got {actual!r}."
            )
        for entry in self.entries:
            plan = entry.plan
            if plan.array_name != entry.rule.path:
                raise ValueError(f"Storage plan name mismatch at {entry.rule.path!r}.")
            if plan.profile_id != self.profile.profile_id:
                raise ValueError(
                    f"Storage profile mismatch at {entry.rule.path!r}."
                )
            if plan.access_pattern != entry.rule.access.value:
                raise ValueError(f"Storage access mismatch at {entry.rule.path!r}.")
            if plan.write_mode != WriteMode.IMMUTABLE.value:
                raise ValueError(
                    "Canonical refined-detection snapshots must be immutable."
                )
            if plan.shard_axes != (0,):
                raise ValueError(
                    "Refined-detection arrays may shard only along their row axis."
                )
            self._require_physical_units(entry)

    @staticmethod
    def _require_physical_units(entry: RefinedDetectionStoragePlanEntry) -> None:
        plan = entry.plan
        chunk = plan.chunk_shape
        if chunk is None:
            raise ValueError("Refined-detection snapshot arrays cannot be scalars.")
        if chunk[1:] != tuple(max(1, value) for value in plan.logical_shape[1:]):
            raise ValueError(
                "Chunks must preserve complete trailing row axes at "
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
                f"Unsharded publication must have one materializing writer at "
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
    def sharded_array_count(self) -> int:
        return sum(entry.plan.is_sharded for entry in self.entries)

    @property
    def estimated_stage_objects(self) -> int:
        return (
            self.estimated_payload_objects
            + self.estimated_array_metadata_objects
            + _STAGE_GROUP_METADATA_OBJECTS
        )

    def as_manifest(self) -> dict[str, object]:
        codec = get_codec_profile(self.profile.codec_profile_id)
        profile_role = _profile_role(self.profile)
        profile_status = (
            "promoted_production_default"
            if profile_role == "promoted_detection_snapshot_default"
            else (
                "available_only_by_explicit_rollback"
                if profile_role == "explicit_detection_snapshot_rollback"
                else "resolved_plan_evidence_not_a_production_default_promotion"
            )
        )
        return {
            "schema_id": REFINED_DETECTION_STORAGE_SCHEMA_ID,
            "schema_version": REFINED_DETECTION_STORAGE_SCHEMA_VERSION,
            "logical_stage_schema": {
                "id": REFINED_DETECTION_SCHEMA_V1.schema_id,
                "version": REFINED_DETECTION_SCHEMA_V1.schema_version,
            },
            "dimensions": self.dimensions.as_manifest(),
            "storage_profile": self.profile.as_manifest(),
            "storage_profile_role": profile_role,
            "codec_profile": codec.as_manifest(),
            "profile_status": profile_status,
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
                "direct_consolidated_equivalence": "required_before_visibility",
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
    dimensions: RefinedDetectionDimensions,
) -> tuple[int, ...]:
    values = dimensions.contract_dimensions
    concrete: list[int] = []
    for dimension in contract.shape_template:
        if isinstance(dimension, int):
            concrete.append(dimension)
        else:
            try:
                concrete.append(values[dimension])
            except KeyError as exc:
                raise ValueError(
                    f"Unknown refined detection dimension {dimension!r}."
                ) from exc
    return tuple(concrete)


def plan_refined_detection_storage(
    dimensions: RefinedDetectionDimensions,
    *,
    profile: StorageProfile = DETECTION_PUBLISHED_ACCESS_AWARE_V1,
) -> RefinedDetectionStoragePlanSet:
    """Resolve every exact logical binding through byte-based policy."""

    entries: list[RefinedDetectionStoragePlanEntry] = []
    for binding in REFINED_DETECTION_SCHEMA_V1.bindings_for(dimensions):
        rule = _rule_for_path(binding.path)
        contract = REFINED_DETECTION_SCHEMA_V1.contracts.resolve(
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
            RefinedDetectionStoragePlanEntry(
                rule=rule,
                plan=plan_storage(intent, profile),
            )
        )
    return RefinedDetectionStoragePlanSet(
        dimensions=dimensions,
        profile=profile,
        entries=tuple(entries),
    )


__all__ = [
    "REFINED_DETECTION_ACCESS_AWARE_CANDIDATE_V1",
    "REFINED_DETECTION_REGULAR_CONTROL_V1",
    "REFINED_DETECTION_STORAGE_SCHEMA_ID",
    "REFINED_DETECTION_STORAGE_SCHEMA_VERSION",
    "RefinedDetectionStoragePlanEntry",
    "RefinedDetectionStoragePlanSet",
    "RefinedDetectionStorageRule",
    "plan_refined_detection_storage",
]
