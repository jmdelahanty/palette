"""Byte-derived immutable storage plans for subject-mask presentation caches.

Dense ``masks_roi`` remains the scientific, editable, and training authority.
This module plans independently regenerable fixed-count sampled contours that
are optimized for row-addressed viewer reads.
"""

from __future__ import annotations

from dataclasses import dataclass

from fisheye.shared.zarr.array_contracts import (
    ArrayContract,
    SUBJECT_MASK_SAMPLED_CONTOUR_POINTS_XY_V1,
    SUBJECT_MASK_SAMPLED_CONTOUR_SOURCE_POINT_COUNT_V1,
    SUBJECT_MASK_SAMPLED_CONTOUR_VALID_V1,
)
from fisheye.shared.zarr.refined_subject_mask_extensions import (
    SubjectMaskSampledContourProfile,
    default_subject_mask_sampled_contour_profile,
)
from fisheye.shared.zarr.storage_intent import AccessPattern, StoragePlan, WriteMode
from fisheye.shared.zarr.storage_planner import plan_storage
from fisheye.shared.zarr.storage_profiles import (
    SUBJECT_MASK_PRESENTATION_CANDIDATE_V1,
    StorageProfile,
)
from fisheye.shared.zarr.subject_mask_schema import (
    SubjectMaskComponentRegistry,
    SubjectMaskDimensions,
)


SUBJECT_MASK_CACHE_STORAGE_SCHEMA_ID = (
    "palette.stage_storage.subject_mask_sampled_contours"
)
SUBJECT_MASK_CACHE_STORAGE_SCHEMA_VERSION = 1
SUBJECT_MASK_SAMPLED_CONTOUR_STAGE_KIND = "sampled_contour_display_cache"

_CONTRACTS: dict[str, ArrayContract] = {
    "points_xy": SUBJECT_MASK_SAMPLED_CONTOUR_POINTS_XY_V1,
    "valid": SUBJECT_MASK_SAMPLED_CONTOUR_VALID_V1,
    "source_point_count": SUBJECT_MASK_SAMPLED_CONTOUR_SOURCE_POINT_COUNT_V1,
}


@dataclass(frozen=True)
class SubjectMaskCacheStorageRule:
    path: str
    component: str
    field: str
    sample_count: int
    access: AccessPattern
    access_unit_semantics: str
    representative_request: str

    def __post_init__(self) -> None:
        component = str(self.component).strip()
        field = str(self.field).strip()
        expected = f"components/{component}/sampled_contours/{field}"
        if not component or field not in _CONTRACTS or self.path != expected:
            raise ValueError("Sampled-contour storage rule path is not canonical.")
        if type(self.sample_count) is not int or self.sample_count <= 0:
            raise ValueError("Sampled-contour sample_count must be positive.")
        object.__setattr__(self, "access", AccessPattern(self.access))

    @property
    def contract(self) -> ArrayContract:
        return _CONTRACTS[self.field]


@dataclass(frozen=True)
class SubjectMaskCacheStoragePlanEntry:
    rule: SubjectMaskCacheStorageRule
    plan: StoragePlan

    def as_manifest(self) -> dict[str, object]:
        return {
            "path": self.rule.path,
            "component": self.rule.component,
            "field": self.rule.field,
            "sample_count": self.rule.sample_count,
            "access_unit_semantics": self.rule.access_unit_semantics,
            "representative_request": self.rule.representative_request,
            "metadata_object_count": 1,
            "estimated_total_objects": 1 + self.plan.estimated_payload_objects,
            "plan": self.plan.as_dict(),
        }


@dataclass(frozen=True)
class SubjectMaskSampledContourStoragePlanSet:
    dimensions: SubjectMaskDimensions
    components: SubjectMaskComponentRegistry
    contour_profile: SubjectMaskSampledContourProfile
    profile: StorageProfile
    entries: tuple[SubjectMaskCacheStoragePlanEntry, ...]

    def __post_init__(self) -> None:
        self.components.require_dimensions(self.dimensions)
        self.contour_profile.require_components(self.components)
        expected_paths = {
            f"components/{component}/sampled_contours/{field}"
            for component in self.components.labels
            for field in _CONTRACTS
        }
        observed_paths = {entry.rule.path for entry in self.entries}
        if observed_paths != expected_paths or len(observed_paths) != len(self.entries):
            raise ValueError("Sampled-contour storage plan paths are not exact.")
        for entry in self.entries:
            plan = entry.plan
            if plan.array_name != entry.rule.path:
                raise ValueError("Sampled-contour plan name differs from its path.")
            if plan.profile_id != self.profile.profile_id:
                raise ValueError("Sampled-contour storage profile identity mismatch.")
            if plan.access_pattern != entry.rule.access.value:
                raise ValueError("Sampled-contour access classification mismatch.")
            if plan.write_mode != WriteMode.IMMUTABLE.value:
                raise ValueError("Published sampled contours must be immutable.")
            if plan.shard_axes != (0,):
                raise ValueError("Sampled contours may shard only along observations.")
            if not plan.shard_byte_budget_satisfied:
                raise ValueError("Sampled-contour shard exceeds the byte ceiling.")
            if plan.chunk_shape is None:
                raise ValueError("Sampled-contour arrays cannot be scalar.")
            expected_trailing = tuple(max(1, value) for value in plan.logical_shape[1:])
            if plan.chunk_shape[1:] != expected_trailing:
                raise ValueError("Sampled-contour chunks must preserve complete rows.")
            if plan.shard_shape is not None:
                if any(
                    shard % chunk
                    for shard, chunk in zip(
                        plan.shard_shape, plan.chunk_shape, strict=True
                    )
                ):
                    raise ValueError("Sampled-contour shards must contain whole chunks.")
                if plan.write_ownership != "whole_shard_single_writer":
                    raise ValueError("Sharded contour caches require whole-shard ownership.")

    @property
    def estimated_logical_nbytes(self) -> int:
        return sum(entry.plan.logical_nbytes for entry in self.entries)

    @property
    def estimated_payload_objects(self) -> int:
        return sum(entry.plan.estimated_payload_objects for entry in self.entries)

    @property
    def arrays_over_object_budget(self) -> tuple[str, ...]:
        return tuple(
            entry.rule.path
            for entry in self.entries
            if not entry.plan.object_budget_satisfied
        )

    def as_manifest(self) -> dict[str, object]:
        return {
            "schema_id": SUBJECT_MASK_CACHE_STORAGE_SCHEMA_ID,
            "schema_version": SUBJECT_MASK_CACHE_STORAGE_SCHEMA_VERSION,
            "stage_kind": SUBJECT_MASK_SAMPLED_CONTOUR_STAGE_KIND,
            "dimensions": self.dimensions.as_manifest(),
            "components": self.components.as_manifest(),
            "contour_profile": self.contour_profile.as_manifest(
                components=self.components
            ),
            "storage_profile": self.profile.as_manifest(),
            "object_estimate": {
                "logical_nbytes": self.estimated_logical_nbytes,
                "inner_chunk_count": sum(
                    entry.plan.estimated_chunk_count for entry in self.entries
                ),
                "payload_objects": self.estimated_payload_objects,
                "sharded_arrays": sum(entry.plan.is_sharded for entry in self.entries),
                "array_metadata_objects": len(self.entries),
                "arrays_over_object_budget": list(self.arrays_over_object_budget),
                "all_array_object_budgets_satisfied": not self.arrays_over_object_budget,
                "budget_scope": "each_array_plan; stage_total_reported_separately",
            },
            "write_partition_contract": {
                "axis": 0,
                "ownership": "one_process_owns_every_complete_output_shard",
                "partial_cross_worker_physical_unit_writes": "forbidden",
            },
            "arrays": [entry.as_manifest() for entry in self.entries],
        }

    def by_path(self) -> dict[str, SubjectMaskCacheStoragePlanEntry]:
        return {entry.rule.path: entry for entry in self.entries}


def _rule(
    component: str, field: str, sample_count: int
) -> SubjectMaskCacheStorageRule:
    points = field == "points_xy"
    return SubjectMaskCacheStorageRule(
        path=f"components/{component}/sampled_contours/{field}",
        component=component,
        field=field,
        sample_count=sample_count,
        access=AccessPattern.PER_ROW if points else AccessPattern.WINDOWED,
        access_unit_semantics=(
            "one_complete_fixed_count_contour_row"
            if points
            else "one_observation_companion_value"
        ),
        representative_request=(
            "one_frame_component_or_short_forward_window"
            if points
            else "same_row_window_as_contour_points"
        ),
    )


def plan_subject_mask_sampled_contour_storage(
    dimensions: SubjectMaskDimensions,
    *,
    components: SubjectMaskComponentRegistry,
    contour_profile: SubjectMaskSampledContourProfile | None = None,
    profile: StorageProfile = SUBJECT_MASK_PRESENTATION_CANDIDATE_V1,
) -> SubjectMaskSampledContourStoragePlanSet:
    """Plan every component's fixed-K display cache from uncompressed bytes."""

    components.require_dimensions(dimensions)
    contour_profile = contour_profile or default_subject_mask_sampled_contour_profile(
        components
    )
    contour_profile.require_components(components)
    entries: list[SubjectMaskCacheStoragePlanEntry] = []
    for component in components.labels:
        sample_count = int(contour_profile.sample_counts[component])
        dimensions_with_samples = {
            **dimensions.contract_dimensions,
            "n_samples": sample_count,
        }
        shapes = {
            "points_xy": (dimensions.n_rois, sample_count, 2),
            "valid": (dimensions.n_rois,),
            "source_point_count": (dimensions.n_rois,),
        }
        for field in ("points_xy", "valid", "source_point_count"):
            rule = _rule(component, field, sample_count)
            contract = rule.contract
            shape = shapes[field]
            intent = contract.storage_intent(
                shape=shape,
                access=rule.access,
                write_mode=WriteMode.IMMUTABLE,
                access_unit_shape=(1, *shape[1:]),
                growth_axis=0,
                shard_axes=(0,),
                whole_shard_writes=True,
                name=rule.path,
                dimensions=dimensions_with_samples,
            )
            entries.append(
                SubjectMaskCacheStoragePlanEntry(
                    rule=rule,
                    plan=plan_storage(intent, profile),
                )
            )
    return SubjectMaskSampledContourStoragePlanSet(
        dimensions=dimensions,
        components=components,
        contour_profile=contour_profile,
        profile=profile,
        entries=tuple(entries),
    )


__all__ = [
    "SUBJECT_MASK_CACHE_STORAGE_SCHEMA_ID",
    "SUBJECT_MASK_CACHE_STORAGE_SCHEMA_VERSION",
    "SUBJECT_MASK_SAMPLED_CONTOUR_STAGE_KIND",
    "SubjectMaskCacheStoragePlanEntry",
    "SubjectMaskCacheStorageRule",
    "SubjectMaskSampledContourStoragePlanSet",
    "plan_subject_mask_sampled_contour_storage",
]
