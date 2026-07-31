"""Exact logical contract for immutable subject-mask quality snapshots.

Quality runs are row-for-row, observation-local diagnostics bound to one
immutable refined dense subject-mask snapshot.  They may propose usability,
but never own pixels, accepted review state, or longitudinal identity.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from fisheye.shared.zarr.array_contracts import (
    DETECTION_INSTANCE_KEY_V1,
    DETECTION_SOURCE_ACQUISITION_FRAME_INDEX_V1,
    FRAME_ROW_OFFSETS_V1,
    SUBJECT_MASK_QUALITY_ARRAY_CONTRACTS,
    SUBJECT_MASK_QUALITY_COMPONENT_FLAGS_V1,
    SUBJECT_MASK_QUALITY_COMPONENT_METRIC_VALID_V1,
    SUBJECT_MASK_QUALITY_COMPONENT_METRIC_VALUES_V1,
    SUBJECT_MASK_QUALITY_OBSERVATION_FLAGS_V1,
    SUBJECT_MASK_QUALITY_OBSERVATION_METRIC_VALID_V1,
    SUBJECT_MASK_QUALITY_OBSERVATION_METRIC_VALUES_V1,
    SUBJECT_MASK_QUALITY_PROPOSED_COMPONENT_USABLE_V1,
    SUBJECT_MASK_QUALITY_PROPOSED_OBSERVATION_USABLE_V1,
    SUBJECT_MASK_QUALITY_SOURCE_MASK_ROW_IDS_V1,
    ArrayContract,
    ArrayContractBinding,
    ArrayContractCatalog,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.subject_mask_schema import (
    REFINED_SUBJECT_MASK_CORE_SCHEMA_ID,
    REFINED_SUBJECT_MASK_CORE_SCHEMA_VERSION,
    SubjectMaskComponentRegistry,
    SubjectMaskSchemaError,
    SubjectMaskSchemaIssue,
    derive_subject_mask_frame_row_offsets,
)

SUBJECT_MASK_QUALITY_SCHEMA_ID = "palette.stage.subject_mask_quality"
SUBJECT_MASK_QUALITY_SCHEMA_VERSION = 1
SUBJECT_MASK_QUALITY_LAYOUT = "immutable_source_bound_observation_diagnostics_v1"
SUBJECT_MASK_QUALITY_BASE_PATH = "subject_mask_quality_runs/<run>"

_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_METRIC_TOKENS = frozenset(
    {"heading", "temporal", "track", "trajectory", "velocity"}
)

FORBIDDEN_SUBJECT_MASK_QUALITY_ARRAYS = (
    "contours",
    "heading",
    "mask_bitpacked",
    "mask_probs_roi",
    "mask_rle",
    "masks_roi",
    "review_state_codes",
    "row_revision",
    "track_id",
)


@dataclass(frozen=True)
class SubjectMaskQualityMetricDefinition:
    metric_id: str
    metric_version: int
    units: str
    higher_is_worse: bool
    description: str

    def __post_init__(self) -> None:
        metric_id = str(self.metric_id).strip()
        if not _IDENTIFIER.fullmatch(metric_id):
            raise ValueError("metric_id must be lowercase snake_case.")
        if frozenset(metric_id.split("_")).intersection(_FORBIDDEN_METRIC_TOKENS):
            raise ValueError(
                "subject-mask quality v1 metrics must be observation-local and "
                "cannot contain temporal, track, trajectory, velocity, or heading terms."
            )
        if type(self.metric_version) is not int or self.metric_version <= 0:
            raise ValueError("metric_version must be a positive exact integer.")
        units = str(self.units).strip()
        description = str(self.description).strip()
        if not units or "\n" in units:
            raise ValueError("units must be one nonempty line.")
        if not description:
            raise ValueError("description cannot be empty.")
        if type(self.higher_is_worse) is not bool:
            raise TypeError("higher_is_worse must be an exact bool.")
        object.__setattr__(self, "metric_id", metric_id)
        object.__setattr__(self, "units", units)
        object.__setattr__(self, "description", description)

    def as_manifest(self) -> dict[str, object]:
        return {
            "metric_id": self.metric_id,
            "metric_version": self.metric_version,
            "units": self.units,
            "higher_is_worse": self.higher_is_worse,
            "description": self.description,
        }


def _normalize_flag_map(
    value: Mapping[int, str], *, field_name: str
) -> Mapping[int, str]:
    normalized: dict[int, str] = {}
    for raw_bit, raw_label in value.items():
        if type(raw_bit) is not int or not (1 <= raw_bit <= 0x8000):
            raise ValueError(f"{field_name} keys must be positive uint16 bits.")
        if raw_bit & (raw_bit - 1):
            raise ValueError(f"{field_name} keys must each contain exactly one bit.")
        label = str(raw_label).strip()
        if not _IDENTIFIER.fullmatch(label):
            raise ValueError(f"{field_name} labels must be lowercase snake_case.")
        normalized[raw_bit] = label
    if len(normalized) != len(value) or len(set(normalized.values())) != len(
        normalized
    ):
        raise ValueError(f"{field_name} keys and labels must be unique.")
    return MappingProxyType(dict(sorted(normalized.items())))


@dataclass(frozen=True)
class SubjectMaskQualityProfile:
    """Digest-bound metric axes, flag registries, and proposal policy."""

    profile_id: str
    profile_version: int
    policy_digest: str
    component_metrics: tuple[SubjectMaskQualityMetricDefinition, ...]
    observation_metrics: tuple[SubjectMaskQualityMetricDefinition, ...]
    component_flag_map: Mapping[int, str]
    observation_flag_map: Mapping[int, str]

    def __post_init__(self) -> None:
        profile_id = str(self.profile_id).strip()
        if not _IDENTIFIER.fullmatch(profile_id):
            raise ValueError("profile_id must be lowercase snake_case.")
        if type(self.profile_version) is not int or self.profile_version <= 0:
            raise ValueError("profile_version must be a positive exact integer.")
        policy_digest = str(self.policy_digest).strip()
        if not _SHA256.fullmatch(policy_digest):
            raise ValueError("policy_digest must be lowercase hexadecimal SHA-256.")
        component_metrics = tuple(self.component_metrics)
        observation_metrics = tuple(self.observation_metrics)
        if not component_metrics and not observation_metrics:
            raise ValueError("A quality profile must declare at least one metric.")
        metric_ids = [
            metric.metric_id for metric in (*component_metrics, *observation_metrics)
        ]
        if len(metric_ids) != len(set(metric_ids)):
            raise ValueError("Metric IDs must be unique across both metric axes.")
        object.__setattr__(self, "profile_id", profile_id)
        object.__setattr__(self, "policy_digest", policy_digest)
        object.__setattr__(self, "component_metrics", component_metrics)
        object.__setattr__(self, "observation_metrics", observation_metrics)
        object.__setattr__(
            self,
            "component_flag_map",
            _normalize_flag_map(
                self.component_flag_map, field_name="component_flag_map"
            ),
        )
        object.__setattr__(
            self,
            "observation_flag_map",
            _normalize_flag_map(
                self.observation_flag_map, field_name="observation_flag_map"
            ),
        )

    @property
    def declared_component_flag_mask(self) -> int:
        return sum(self.component_flag_map)

    @property
    def declared_observation_flag_mask(self) -> int:
        return sum(self.observation_flag_map)

    def _payload(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "profile_version": self.profile_version,
            "policy_digest": self.policy_digest,
            "component_metrics": [
                metric.as_manifest() for metric in self.component_metrics
            ],
            "observation_metrics": [
                metric.as_manifest() for metric in self.observation_metrics
            ],
            "component_flag_map": {
                str(bit): label for bit, label in self.component_flag_map.items()
            },
            "observation_flag_map": {
                str(bit): label for bit, label in self.observation_flag_map.items()
            },
            "zero_flag_semantics": "no_quality_finding",
        }

    @property
    def profile_digest(self) -> str:
        return canonical_json_sha256(self._payload())

    def as_manifest(self) -> dict[str, object]:
        return {**self._payload(), "profile_digest": self.profile_digest}


@dataclass(frozen=True)
class SubjectMaskQualitySourceReference:
    """One immutable refined dense-mask authority bound by a quality run."""

    run_name: str
    manifest_digest: str
    dense_array_values_sha256: str
    component_registry_digest: str

    def __post_init__(self) -> None:
        run_name = str(self.run_name).strip()
        if not run_name or "/" in run_name:
            raise ValueError("run_name must be one nonempty archive group name.")
        object.__setattr__(self, "run_name", run_name)
        for name in (
            "manifest_digest",
            "dense_array_values_sha256",
            "component_registry_digest",
        ):
            value = str(getattr(self, name)).strip()
            if not _SHA256.fullmatch(value):
                raise ValueError(f"{name} must be lowercase hexadecimal SHA-256.")
            object.__setattr__(self, name, value)

    def as_manifest(self) -> dict[str, object]:
        return {
            "stage": "refined_subject_mask",
            "run_name": self.run_name,
            "run_path": f"refined_subject_masks_runs/{self.run_name}",
            "schema_id": REFINED_SUBJECT_MASK_CORE_SCHEMA_ID,
            "schema_version": REFINED_SUBJECT_MASK_CORE_SCHEMA_VERSION,
            "manifest_digest": self.manifest_digest,
            "dense_array_values_sha256": self.dense_array_values_sha256,
            "component_registry_digest": self.component_registry_digest,
            "coverage": "every_source_row_exactly_once_in_source_order",
        }


@dataclass(frozen=True)
class SubjectMaskQualityDimensions:
    n_frames: int
    n_rois: int
    n_channels: int
    roi_height: int
    roi_width: int
    n_component_metrics: int
    n_observation_metrics: int

    def __post_init__(self) -> None:
        for name in (
            "n_frames",
            "n_rois",
            "n_channels",
            "roi_height",
            "roi_width",
            "n_component_metrics",
            "n_observation_metrics",
        ):
            if type(getattr(self, name)) is not int:
                raise TypeError(f"{name} must be an exact integer.")
        if (
            min(
                self.n_frames,
                self.n_rois,
                self.n_channels,
                self.roi_height,
                self.roi_width,
            )
            <= 0
        ):
            raise ValueError(
                "Frame, row, component, and ROI dimensions must be positive."
            )
        if self.n_component_metrics < 0 or self.n_observation_metrics < 0:
            raise ValueError("Metric dimensions cannot be negative.")
        if self.n_component_metrics + self.n_observation_metrics == 0:
            raise ValueError("At least one metric dimension must be positive.")

    @property
    def contract_dimensions(self) -> dict[str, int]:
        return {
            "n_frames": self.n_frames,
            "n_frame_boundaries": self.n_frames + 1,
            "n_instances": self.n_rois,
            "n_rois": self.n_rois,
            "n_channels": self.n_channels,
            "H": self.roi_height,
            "W": self.roi_width,
            "n_component_metrics": self.n_component_metrics,
            "n_observation_metrics": self.n_observation_metrics,
        }

    def as_manifest(self) -> dict[str, int]:
        return {
            **self.contract_dimensions,
            "roi_height": self.roi_height,
            "roi_width": self.roi_width,
        }


def _binding(path: str, contract: ArrayContract) -> ArrayContractBinding:
    return ArrayContractBinding(
        path=path,
        contract_id=contract.schema_id,
        contract_version=contract.schema_version,
        required=True,
    )


SUBJECT_MASK_QUALITY_BINDINGS = (
    _binding("instance_key", DETECTION_INSTANCE_KEY_V1),
    _binding("source_mask_row_ids", SUBJECT_MASK_QUALITY_SOURCE_MASK_ROW_IDS_V1),
    _binding(
        "source_acquisition_frame_index",
        DETECTION_SOURCE_ACQUISITION_FRAME_INDEX_V1,
    ),
    _binding("frame_row_offsets", FRAME_ROW_OFFSETS_V1),
    _binding(
        "component_metric_values", SUBJECT_MASK_QUALITY_COMPONENT_METRIC_VALUES_V1
    ),
    _binding("component_metric_valid", SUBJECT_MASK_QUALITY_COMPONENT_METRIC_VALID_V1),
    _binding(
        "observation_metric_values",
        SUBJECT_MASK_QUALITY_OBSERVATION_METRIC_VALUES_V1,
    ),
    _binding(
        "observation_metric_valid",
        SUBJECT_MASK_QUALITY_OBSERVATION_METRIC_VALID_V1,
    ),
    _binding("component_quality_flags", SUBJECT_MASK_QUALITY_COMPONENT_FLAGS_V1),
    _binding("observation_quality_flags", SUBJECT_MASK_QUALITY_OBSERVATION_FLAGS_V1),
    _binding(
        "proposed_component_usable",
        SUBJECT_MASK_QUALITY_PROPOSED_COMPONENT_USABLE_V1,
    ),
    _binding(
        "proposed_observation_usable",
        SUBJECT_MASK_QUALITY_PROPOSED_OBSERVATION_USABLE_V1,
    ),
)


def _materialize(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    try:
        return np.asarray(value[...])
    except (IndexError, KeyError, TypeError):
        return np.asarray(value)


def _issue(code: str, path: str, message: str) -> SubjectMaskSchemaIssue:
    return SubjectMaskSchemaIssue(code=code, path=path, message=message)


@dataclass(frozen=True)
class SubjectMaskQualitySchema:
    schema_id: str = SUBJECT_MASK_QUALITY_SCHEMA_ID
    schema_version: int = SUBJECT_MASK_QUALITY_SCHEMA_VERSION
    stage: str = "subject_mask_quality"
    layout: str = SUBJECT_MASK_QUALITY_LAYOUT
    base_path: str = SUBJECT_MASK_QUALITY_BASE_PATH
    bindings: tuple[ArrayContractBinding, ...] = SUBJECT_MASK_QUALITY_BINDINGS
    contracts: ArrayContractCatalog = SUBJECT_MASK_QUALITY_ARRAY_CONTRACTS

    @property
    def binding_paths(self) -> tuple[str, ...]:
        return tuple(binding.path for binding in self.bindings)

    def validate(
        self,
        arrays: Mapping[str, Any],
        *,
        dimensions: SubjectMaskQualityDimensions,
        components: SubjectMaskComponentRegistry,
        profile: SubjectMaskQualityProfile,
        source_mask_arrays: Mapping[str, Any] | None,
    ) -> tuple[SubjectMaskSchemaIssue, ...]:
        issues: list[SubjectMaskSchemaIssue] = []
        if len(components.labels) != dimensions.n_channels:
            raise ValueError("Component count does not match quality dimensions.")
        if len(profile.component_metrics) != dimensions.n_component_metrics:
            issues.append(
                _issue(
                    "component_metric_catalog_size_mismatch",
                    "component_metric_values",
                    "The component metric axis must equal the ordered profile catalog.",
                )
            )
        if len(profile.observation_metrics) != dimensions.n_observation_metrics:
            issues.append(
                _issue(
                    "observation_metric_catalog_size_mismatch",
                    "observation_metric_values",
                    "The observation metric axis must equal the ordered profile catalog.",
                )
            )

        expected = set(self.binding_paths)
        invalid: set[str] = set()
        for path in sorted(set(arrays) - expected):
            code = (
                "authority_or_longitudinal_payload_forbidden"
                if path in FORBIDDEN_SUBJECT_MASK_QUALITY_ARRAYS
                else "unexpected_array"
            )
            issues.append(_issue(code, path, "Array is outside quality v1."))
        for binding in self.bindings:
            path = binding.path
            if path not in arrays:
                invalid.add(path)
                issues.append(
                    _issue("missing_required_array", path, "Array is absent.")
                )
                continue
            contract = self.contracts.resolve(
                binding.contract_id, binding.contract_version
            )
            try:
                errors = contract.validate_observation(
                    arrays[path], dimensions=dimensions.contract_dimensions
                )
            except Exception as exc:
                errors = (f"array metadata is unreadable: {exc}",)
            if errors:
                invalid.add(path)
                issues.extend(
                    _issue("array_contract_violation", path, error) for error in errors
                )

        values: dict[str, np.ndarray] = {}
        for path in self.binding_paths:
            if path in arrays and path not in invalid:
                try:
                    values[path] = _materialize(arrays[path])
                except Exception as exc:
                    issues.append(_issue("array_read_failure", path, str(exc)))

        frames = values.get("source_acquisition_frame_index")
        offsets = values.get("frame_row_offsets")
        if frames is not None:
            try:
                expected_offsets = derive_subject_mask_frame_row_offsets(
                    frames, n_frames=dimensions.n_frames
                )
            except ValueError as exc:
                issues.append(
                    _issue(
                        "invalid_frame_order",
                        "source_acquisition_frame_index",
                        str(exc),
                    )
                )
            else:
                if offsets is not None and not np.array_equal(
                    offsets, expected_offsets
                ):
                    issues.append(
                        _issue(
                            "frame_row_offsets_mismatch",
                            "frame_row_offsets",
                            "Offsets must exactly index the sorted quality rows.",
                        )
                    )
        keys = values.get("instance_key")
        if keys is not None and np.unique(keys).size != keys.size:
            issues.append(
                _issue("duplicate_instance_key", "instance_key", "Keys must be unique.")
            )
        rows = values.get("source_mask_row_ids")
        if rows is not None and not np.array_equal(
            rows, np.arange(dimensions.n_rois, dtype=np.int64)
        ):
            issues.append(
                _issue(
                    "incomplete_source_row_coverage",
                    "source_mask_row_ids",
                    "Quality v1 must cover every source row once and in order.",
                )
            )

        for value_path, valid_path in (
            ("component_metric_values", "component_metric_valid"),
            ("observation_metric_values", "observation_metric_valid"),
        ):
            metric_values = values.get(value_path)
            metric_valid = values.get(valid_path)
            if metric_values is None or metric_valid is None:
                continue
            if not np.array_equal(metric_valid, np.isfinite(metric_values)):
                issues.append(
                    _issue(
                        "metric_validity_mismatch",
                        valid_path,
                        "Validity must exactly mark finite values.",
                    )
                )
            if np.any(~metric_valid & ~np.isnan(metric_values)):
                issues.append(
                    _issue(
                        "invalid_metric_not_nan",
                        value_path,
                        "Invalid metric values require canonical NaN.",
                    )
                )

        for path, declared_mask in (
            ("component_quality_flags", profile.declared_component_flag_mask),
            ("observation_quality_flags", profile.declared_observation_flag_mask),
        ):
            flags = values.get(path)
            if flags is not None:
                unknown = np.bitwise_and(
                    flags.astype(np.uint32), np.uint32(0xFFFF ^ declared_mask)
                )
                if np.any(unknown):
                    issues.append(
                        _issue(
                            "undeclared_quality_flag",
                            path,
                            "Flags contain undeclared bits.",
                        )
                    )

        proposed_components = values.get("proposed_component_usable")
        proposed_observation = values.get("proposed_observation_usable")
        if proposed_components is not None and proposed_observation is not None:
            if np.any(proposed_observation & ~np.any(proposed_components, axis=1)):
                issues.append(
                    _issue(
                        "usable_observation_without_component",
                        "proposed_observation_usable",
                        "A usable observation must retain at least one usable component.",
                    )
                )

        self._validate_source_binding(
            values,
            source_mask_arrays=source_mask_arrays,
            dimensions=dimensions,
            issues=issues,
        )
        return tuple(issues)

    @staticmethod
    def _validate_source_binding(
        values: Mapping[str, np.ndarray],
        *,
        source_mask_arrays: Mapping[str, Any] | None,
        dimensions: SubjectMaskQualityDimensions,
        issues: list[SubjectMaskSchemaIssue],
    ) -> None:
        if source_mask_arrays is None:
            issues.append(
                _issue(
                    "missing_source_mask_evidence",
                    "source_mask_row_ids",
                    "Bound refined mask evidence is required.",
                )
            )
            return
        required = {
            "instance_key": ((dimensions.n_rois,), np.dtype(np.uint64)),
            "source_acquisition_frame_index": (
                (dimensions.n_rois,),
                np.dtype(np.int64),
            ),
            "available_channels": ((dimensions.n_channels,), np.dtype(bool)),
        }
        if not set(required) <= set(source_mask_arrays):
            issues.append(
                _issue(
                    "incomplete_source_mask_evidence",
                    "source_mask_row_ids",
                    "Source evidence lacks identity or channel availability arrays.",
                )
            )
            return
        source: dict[str, np.ndarray] = {}
        for path, (shape, dtype) in required.items():
            source[path] = _materialize(source_mask_arrays[path])
            if source[path].shape != shape or source[path].dtype != dtype:
                issues.append(
                    _issue(
                        "source_mask_contract_violation",
                        path,
                        f"Expected {shape!r} {dtype}.",
                    )
                )
                return
        for path in ("instance_key", "source_acquisition_frame_index"):
            if path in values and not np.array_equal(values[path], source[path]):
                issues.append(
                    _issue(
                        "source_mask_binding_mismatch",
                        path,
                        "Values differ from the bound source snapshot.",
                    )
                )
        proposed = values.get("proposed_component_usable")
        if proposed is not None and np.any(proposed[:, ~source["available_channels"]]):
            issues.append(
                _issue(
                    "unavailable_component_proposed_usable",
                    "proposed_component_usable",
                    "Unavailable source channels cannot be proposed usable.",
                )
            )

    def require(self, arrays: Mapping[str, Any], **kwargs: Any) -> None:
        issues = self.validate(arrays, **kwargs)
        if issues:
            raise SubjectMaskSchemaError(issues)

    def as_manifest(
        self,
        *,
        dimensions: SubjectMaskQualityDimensions,
        components: SubjectMaskComponentRegistry,
        profile: SubjectMaskQualityProfile,
        source: SubjectMaskQualitySourceReference,
    ) -> dict[str, object]:
        if len(components.labels) != dimensions.n_channels:
            raise ValueError("Component count does not match dimensions.")
        if len(profile.component_metrics) != dimensions.n_component_metrics:
            raise ValueError("Component metric count does not match dimensions.")
        if len(profile.observation_metrics) != dimensions.n_observation_metrics:
            raise ValueError("Observation metric count does not match dimensions.")
        if source.component_registry_digest != canonical_json_sha256(
            components.as_manifest()
        ):
            raise ValueError(
                "Source component-registry digest does not match the declared registry."
            )
        return {
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "stage": self.stage,
            "layout": self.layout,
            "base_path": self.base_path,
            "dimensions": dimensions.as_manifest(),
            "components": components.as_manifest(),
            "source": source.as_manifest(),
            "profile": profile.as_manifest(),
            "bindings": [binding.as_manifest() for binding in self.bindings],
            "forbidden_arrays": list(FORBIDDEN_SUBJECT_MASK_QUALITY_ARRAYS),
            "array_contracts": self.contracts.as_manifest(),
            "invariants": {
                "row_identity": "source_instance_key",
                "row_order": "exact_refined_subject_mask_row_order",
                "frame_lookup": "frame_row_offsets_csr",
                "instances_per_frame": "zero_one_or_many",
                "metric_invalid_value": "canonical_nan",
                "quality_output": "diagnostic_and_proposed_not_accepted_review",
                "pixels": "forbidden_reference_bound_dense_snapshot",
                "derived_caches": "forbidden_reference_bound_cache_receipts",
                "longitudinal_metrics": "forbidden_without_track_lineage_new_schema",
            },
        }


SUBJECT_MASK_QUALITY_SCHEMA_V1 = SubjectMaskQualitySchema()


__all__ = [
    "FORBIDDEN_SUBJECT_MASK_QUALITY_ARRAYS",
    "SUBJECT_MASK_QUALITY_BASE_PATH",
    "SUBJECT_MASK_QUALITY_BINDINGS",
    "SUBJECT_MASK_QUALITY_LAYOUT",
    "SUBJECT_MASK_QUALITY_SCHEMA_ID",
    "SUBJECT_MASK_QUALITY_SCHEMA_V1",
    "SUBJECT_MASK_QUALITY_SCHEMA_VERSION",
    "SubjectMaskQualityDimensions",
    "SubjectMaskQualityMetricDefinition",
    "SubjectMaskQualityProfile",
    "SubjectMaskQualitySchema",
    "SubjectMaskQualitySourceReference",
]
