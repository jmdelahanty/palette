"""Byte-derived immutable storage plans for subject-mask core arrays.

The module consumes exact logical schemas and emits storage manifests only.  It
does not import Zarr, create arrays, activate selectors, or modify production
writers.  Editable-draft and variable-length cache plans are separate future
contracts because their write ownership and access units differ.
"""

from __future__ import annotations

from dataclasses import dataclass

from fisheye.shared.zarr.array_contracts import (
    ArrayContract,
    ArrayContractBinding,
    ArrayContractCatalog,
)
from fisheye.shared.zarr.storage_intent import AccessPattern, StoragePlan, WriteMode
from fisheye.shared.zarr.storage_planner import plan_storage
from fisheye.shared.zarr.storage_profiles import PUBLISHED_HTTP_V1, StorageProfile
from fisheye.shared.zarr.subject_mask_schema import (
    RAW_SUBJECT_MASK_FLOAT16_SCHEMA_V1,
    RAW_SUBJECT_MASK_UINT8_SCHEMA_V1,
    REFINED_SUBJECT_MASK_CORE_SCHEMA_V1,
    RawSubjectMaskSchema,
    RefinedSubjectMaskCoreSchema,
    SubjectMaskDimensions,
    SubjectMaskProbabilityEncoding,
)


SUBJECT_MASK_STORAGE_SCHEMA_ID = "palette.stage_storage.subject_mask_core"
SUBJECT_MASK_STORAGE_SCHEMA_VERSION = 1

_MASK_PAYLOAD_PATHS = {"mask_probs_roi", "masks_roi"}
_EAGER_PATHS = {"frame_row_offsets", "available_channels"}


@dataclass(frozen=True)
class SubjectMaskStorageRule:
    path: str
    access: AccessPattern
    access_unit_semantics: str
    representative_request: str

    def __post_init__(self) -> None:
        path = str(self.path).strip().strip("/")
        if not path:
            raise ValueError("Subject-mask storage path cannot be empty.")
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "access", AccessPattern(self.access))
        if not self.access_unit_semantics.strip():
            raise ValueError("Access-unit semantics cannot be empty.")
        if not self.representative_request.strip():
            raise ValueError("Representative request cannot be empty.")


def _rule(path: str) -> SubjectMaskStorageRule:
    if path in _MASK_PAYLOAD_PATHS:
        return SubjectMaskStorageRule(
            path=path,
            access=AccessPattern.PER_ROW,
            access_unit_semantics="one_observation_one_component_roi_plane",
            representative_request="one_frame_component_or_short_forward_window",
        )
    if path == "frame_row_offsets":
        return SubjectMaskStorageRule(
            path=path,
            access=AccessPattern.EAGER,
            access_unit_semantics="one_frame_boundary_offset",
            representative_request="whole_retained_frame_index",
        )
    if path == "available_channels":
        return SubjectMaskStorageRule(
            path=path,
            access=AccessPattern.EAGER,
            access_unit_semantics="one_component_availability_value",
            representative_request="whole_component_registry_at_open",
        )
    return SubjectMaskStorageRule(
        path=path,
        access=AccessPattern.WINDOWED,
        access_unit_semantics="one_complete_observation_row",
        representative_request="contiguous_frame_or_analysis_row_window",
    )


@dataclass(frozen=True)
class SubjectMaskStoragePlanEntry:
    rule: SubjectMaskStorageRule
    plan: StoragePlan

    def as_manifest(self) -> dict[str, object]:
        return {
            "path": self.rule.path,
            "access_unit_semantics": self.rule.access_unit_semantics,
            "representative_request": self.rule.representative_request,
            "metadata_object_count": 1,
            "estimated_total_objects": 1 + self.plan.estimated_payload_objects,
            "plan": self.plan.as_dict(),
        }


@dataclass(frozen=True)
class SubjectMaskStoragePlanSet:
    stage_kind: str
    logical_schema_id: str
    logical_schema_version: int
    dimensions: SubjectMaskDimensions
    profile: StorageProfile
    entries: tuple[SubjectMaskStoragePlanEntry, ...]

    def __post_init__(self) -> None:
        if self.stage_kind not in {
            "raw_probability_uint8",
            "raw_probability_float16",
            "refined_dense_publication_core",
        }:
            raise ValueError(
                f"Unsupported subject-mask stage kind {self.stage_kind!r}."
            )
        paths = tuple(entry.rule.path for entry in self.entries)
        if len(paths) != len(set(paths)):
            raise ValueError("Subject-mask storage plan paths must be unique.")
        for entry in self.entries:
            plan = entry.plan
            if plan.array_name != entry.rule.path:
                raise ValueError("Subject-mask plan name differs from its rule path.")
            if plan.profile_id != self.profile.profile_id:
                raise ValueError("Subject-mask storage profile identity mismatch.")
            if plan.access_pattern != entry.rule.access.value:
                raise ValueError("Subject-mask access classification mismatch.")
            if plan.write_mode != WriteMode.IMMUTABLE.value:
                raise ValueError("This plan set is only for immutable publications.")
            if plan.shard_axes != (0,):
                raise ValueError("Subject-mask arrays may shard only along rows.")
            if not plan.shard_byte_budget_satisfied:
                raise ValueError("Subject-mask shard exceeds the profile byte ceiling.")
            self._require_access_unit(entry)
            if plan.shard_shape is not None:
                if any(
                    shard % chunk
                    for shard, chunk in zip(
                        plan.shard_shape,
                        plan.chunk_shape or (),
                        strict=True,
                    )
                ):
                    raise ValueError("Subject-mask shard must contain whole chunks.")
                if plan.write_ownership != "whole_shard_single_writer":
                    raise ValueError("Sharded masks require whole-shard ownership.")
            elif plan.write_ownership != "single_writer_immutable_materialization":
                raise ValueError("Unsharded masks require one immutable writer.")

    @staticmethod
    def _require_access_unit(entry: SubjectMaskStoragePlanEntry) -> None:
        plan = entry.plan
        chunk = plan.chunk_shape
        if chunk is None:
            raise ValueError("Subject-mask arrays cannot be scalar.")
        if entry.rule.path in _MASK_PAYLOAD_PATHS:
            expected = (1, *plan.logical_shape[2:])
            if chunk[1:] != expected:
                raise ValueError(
                    "Mask payload chunks must preserve one complete component plane."
                )
            return
        expected = tuple(max(1, value) for value in plan.logical_shape[1:])
        if chunk[1:] != expected:
            raise ValueError("Tabular mask chunks must preserve complete rows.")

    @property
    def estimated_logical_nbytes(self) -> int:
        return sum(entry.plan.logical_nbytes for entry in self.entries)

    @property
    def estimated_payload_objects(self) -> int:
        return sum(entry.plan.estimated_payload_objects for entry in self.entries)

    @property
    def estimated_inner_chunks(self) -> int:
        return sum(entry.plan.estimated_chunk_count for entry in self.entries)

    @property
    def estimated_stage_objects(self) -> int:
        return self.estimated_payload_objects + len(self.entries) + 2

    @property
    def arrays_over_object_budget(self) -> tuple[str, ...]:
        return tuple(
            entry.rule.path
            for entry in self.entries
            if not entry.plan.object_budget_satisfied
        )

    def as_manifest(self) -> dict[str, object]:
        return {
            "schema_id": SUBJECT_MASK_STORAGE_SCHEMA_ID,
            "schema_version": SUBJECT_MASK_STORAGE_SCHEMA_VERSION,
            "stage_kind": self.stage_kind,
            "logical_stage_schema": {
                "id": self.logical_schema_id,
                "version": self.logical_schema_version,
            },
            "dimensions": self.dimensions.as_manifest(),
            "storage_profile": self.profile.as_manifest(),
            "object_estimate": {
                "logical_nbytes": self.estimated_logical_nbytes,
                "inner_chunk_count": self.estimated_inner_chunks,
                "payload_objects": self.estimated_payload_objects,
                "sharded_arrays": sum(entry.plan.is_sharded for entry in self.entries),
                "array_metadata_objects": len(self.entries),
                "group_metadata_objects": 2,
                "stage_objects": self.estimated_stage_objects,
                "budget_scope": "each_array_plan; stage_total_reported_separately",
                "arrays_over_object_budget": list(self.arrays_over_object_budget),
                "all_array_object_budgets_satisfied": not self.arrays_over_object_budget,
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
    dimensions: SubjectMaskDimensions,
) -> tuple[int, ...]:
    values = dimensions.contract_dimensions
    return tuple(
        axis if isinstance(axis, int) else values[axis]
        for axis in contract.shape_template
    )


def _access_unit(path: str, shape: tuple[int, ...]) -> tuple[int, ...]:
    if path in _MASK_PAYLOAD_PATHS:
        return (1, 1, *shape[2:])
    return (1, *shape[1:])


def _plan_bindings(
    *,
    bindings: tuple[ArrayContractBinding, ...],
    contracts: ArrayContractCatalog,
    dimensions: SubjectMaskDimensions,
    profile: StorageProfile,
) -> tuple[SubjectMaskStoragePlanEntry, ...]:
    entries: list[SubjectMaskStoragePlanEntry] = []
    for binding in bindings:
        contract = contracts.resolve(binding.contract_id, binding.contract_version)
        shape = _concrete_shape(contract, dimensions)
        rule = _rule(binding.path)
        intent = contract.storage_intent(
            shape=shape,
            access=rule.access,
            write_mode=WriteMode.IMMUTABLE,
            access_unit_shape=_access_unit(binding.path, shape),
            growth_axis=0,
            shard_axes=(0,),
            whole_shard_writes=True,
            name=binding.path,
            dimensions=dimensions.contract_dimensions,
        )
        entries.append(
            SubjectMaskStoragePlanEntry(
                rule=rule,
                plan=plan_storage(intent, profile),
            )
        )
    return tuple(entries)


def _raw_schema(
    encoding: SubjectMaskProbabilityEncoding,
) -> tuple[RawSubjectMaskSchema, str]:
    if encoding is SubjectMaskProbabilityEncoding.LINEAR_UINT8_0_255:
        return RAW_SUBJECT_MASK_UINT8_SCHEMA_V1, "raw_probability_uint8"
    if encoding is SubjectMaskProbabilityEncoding.UNIT_FLOAT16:
        return RAW_SUBJECT_MASK_FLOAT16_SCHEMA_V1, "raw_probability_float16"
    raise ValueError(f"Unsupported subject-mask probability encoding {encoding!r}.")


def plan_raw_subject_mask_storage(
    dimensions: SubjectMaskDimensions,
    *,
    encoding: SubjectMaskProbabilityEncoding,
    include_threshold_cache: bool = False,
    profile: StorageProfile = PUBLISHED_HTTP_V1,
) -> SubjectMaskStoragePlanSet:
    """Plan a complete immutable raw core without creating any arrays."""

    encoding = SubjectMaskProbabilityEncoding(encoding)
    schema, stage_kind = _raw_schema(encoding)
    bindings = tuple(
        binding
        for binding in schema.bindings
        if binding.required or include_threshold_cache
    )
    return SubjectMaskStoragePlanSet(
        stage_kind=stage_kind,
        logical_schema_id=schema.schema_id,
        logical_schema_version=schema.schema_version,
        dimensions=dimensions,
        profile=profile,
        entries=_plan_bindings(
            bindings=bindings,
            contracts=schema.contracts,
            dimensions=dimensions,
            profile=profile,
        ),
    )


def plan_refined_subject_mask_publication_storage(
    dimensions: SubjectMaskDimensions,
    *,
    profile: StorageProfile = PUBLISHED_HTTP_V1,
) -> SubjectMaskStoragePlanSet:
    """Plan the immutable refined dense scientific core only."""

    schema: RefinedSubjectMaskCoreSchema = REFINED_SUBJECT_MASK_CORE_SCHEMA_V1
    return SubjectMaskStoragePlanSet(
        stage_kind="refined_dense_publication_core",
        logical_schema_id=schema.schema_id,
        logical_schema_version=schema.schema_version,
        dimensions=dimensions,
        profile=profile,
        entries=_plan_bindings(
            bindings=schema.bindings,
            contracts=schema.contracts,
            dimensions=dimensions,
            profile=profile,
        ),
    )


__all__ = [
    "SUBJECT_MASK_STORAGE_SCHEMA_ID",
    "SUBJECT_MASK_STORAGE_SCHEMA_VERSION",
    "SubjectMaskStoragePlanEntry",
    "SubjectMaskStoragePlanSet",
    "SubjectMaskStorageRule",
    "plan_raw_subject_mask_storage",
    "plan_refined_subject_mask_publication_storage",
]
