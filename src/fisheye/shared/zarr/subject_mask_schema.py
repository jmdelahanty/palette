"""Exact logical array schemas for raw and refined subject-mask snapshots.

This module is independent of Zarr creation and physical storage planning.  It
defines the future-facing, recording-bound core arrays and validates large mask
payloads in bounded row blocks.  Mutable draft state, derived compact caches,
publication manifests, and selector activation remain separate contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Any, Mapping

import numpy as np

from fisheye.shared.zarr.array_contracts import (
    CROP_SOURCE_CROP_XYWH_V1,
    DENSE_SUBJECT_MASKS_ROI_V1,
    DETECTION_INSTANCE_KEY_V1,
    DETECTION_SOURCE_ACQUISITION_FRAME_INDEX_V1,
    FRAME_ROW_OFFSETS_V1,
    RAW_SUBJECT_MASK_FLOAT16_ARRAY_CONTRACTS,
    RAW_SUBJECT_MASK_UINT8_ARRAY_CONTRACTS,
    REFINED_SUBJECT_MASK_CORE_ARRAY_CONTRACTS,
    SUBJECT_MASK_AREA_PX_V1,
    SUBJECT_MASK_AVAILABLE_CHANNELS_V1,
    SUBJECT_MASK_BBOX_VALID_V1,
    SUBJECT_MASK_BBOX_XYXY_V1,
    SUBJECT_MASK_CENTROID_VALID_V1,
    SUBJECT_MASK_CENTROID_XY_V1,
    SUBJECT_MASK_PRESENT_V1,
    SUBJECT_MASK_PROBABILITIES_FLOAT16_V1,
    SUBJECT_MASK_PROBABILITIES_UINT8_V1,
    SUBJECT_MASK_PROB_MAX_V1,
    SUBJECT_MASK_SOURCE_CROP_ROW_IDS_V1,
    ArrayContract,
    ArrayContractBinding,
    ArrayContractCatalog,
)

RAW_SUBJECT_MASK_UINT8_SCHEMA_ID = "palette.stage.subject_mask_probabilities_uint8"
RAW_SUBJECT_MASK_FLOAT16_SCHEMA_ID = "palette.stage.subject_mask_probabilities_float16"
RAW_SUBJECT_MASK_SCHEMA_VERSION = 1
REFINED_SUBJECT_MASK_CORE_SCHEMA_ID = "palette.stage.refined_subject_mask_dense_core"
REFINED_SUBJECT_MASK_CORE_SCHEMA_VERSION = 1
SUBJECT_MASK_LAYOUT = "recording_observations_with_frame_row_offsets_v1"

_CONTENT_SCAN_TARGET_BYTES = 8 * 1024 * 1024
_COMPONENT_LABEL = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class SubjectMaskComponentSchema:
    """Named component membership and requiredness for one mask schema."""

    schema_id: str
    schema_version: int
    labels: tuple[str, ...]
    required_labels: tuple[str, ...]

    def __post_init__(self) -> None:
        schema_id = str(self.schema_id).strip()
        labels = tuple(str(label).strip() for label in self.labels)
        required = tuple(str(label).strip() for label in self.required_labels)
        if not schema_id:
            raise ValueError("Subject-mask component schema_id is required.")
        if type(self.schema_version) is not int or self.schema_version <= 0:
            raise ValueError("Subject-mask component schema_version must be positive.")
        if not labels:
            raise ValueError("At least one subject-mask component is required.")
        if len(labels) != len(set(labels)):
            raise ValueError("Subject-mask component labels must be unique.")
        for label in labels:
            if not _COMPONENT_LABEL.fullmatch(label):
                raise ValueError(
                    "Subject-mask component labels must match ^[a-z][a-z0-9_]*$."
                )
        if len(required) != len(set(required)):
            raise ValueError("Required subject-mask component labels must be unique.")
        unknown = set(required).difference(labels)
        if unknown:
            raise ValueError(
                "Required subject-mask components are absent from the schema: "
                + ", ".join(sorted(unknown))
            )
        object.__setattr__(self, "schema_id", schema_id)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "required_labels", required)

    @property
    def optional_labels(self) -> tuple[str, ...]:
        required = set(self.required_labels)
        return tuple(label for label in self.labels if label not in required)

    def as_manifest(self) -> dict[str, object]:
        return {
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "labels": list(self.labels),
            "required_labels": list(self.required_labels),
            "optional_labels": list(self.optional_labels),
            "requiredness_policy": "declared_component_schema_v1",
        }


SUBJECT_V1_UNION_COMPONENT_SCHEMA = SubjectMaskComponentSchema(
    schema_id="subject_v1_union",
    schema_version=1,
    labels=("subject_body", "eyes_union", "swim_bladder"),
    required_labels=("subject_body", "eyes_union", "swim_bladder"),
)
SUBJECT_V1_LR_COMPONENT_SCHEMA = SubjectMaskComponentSchema(
    schema_id="subject_v1_lr",
    schema_version=1,
    labels=("subject_body", "eye_left", "eye_right", "swim_bladder"),
    required_labels=("subject_body", "eye_left", "eye_right", "swim_bladder"),
)
SUBJECT_MASK_COMPONENT_SCHEMAS = {
    schema.schema_id: schema
    for schema in (
        SUBJECT_V1_UNION_COMPONENT_SCHEMA,
        SUBJECT_V1_LR_COMPONENT_SCHEMA,
    )
}


def resolve_subject_mask_component_schema(
    *,
    schema_id: str | None,
    labels: tuple[str, ...],
) -> SubjectMaskComponentSchema:
    """Resolve exact persisted labels to one named requiredness contract."""

    normalized_labels = tuple(str(label).strip() for label in labels)
    normalized_id = str(schema_id or "").strip()
    if normalized_id:
        schema = SUBJECT_MASK_COMPONENT_SCHEMAS.get(normalized_id)
        if schema is None:
            raise ValueError(
                f"Unsupported subject-mask component schema_id {normalized_id!r}."
            )
        if normalized_labels != schema.labels:
            raise ValueError(
                f"Subject-mask labels {normalized_labels!r} do not match "
                f"{normalized_id!r}: {schema.labels!r}."
            )
        return schema
    matches = [
        schema
        for schema in SUBJECT_MASK_COMPONENT_SCHEMAS.values()
        if normalized_labels == schema.labels
    ]
    if len(matches) != 1:
        raise ValueError(
            "Persisted subject-mask labels do not resolve to exactly one component "
            f"schema: {normalized_labels!r}."
        )
    return matches[0]


class SubjectMaskProbabilityEncoding(str, Enum):
    LINEAR_UINT8_0_255 = "linear_uint8_0_255"
    UNIT_FLOAT16 = "unit_float16"


@dataclass(frozen=True)
class SubjectMaskDimensions:
    """Concrete recording, row, component, and ROI dimensions."""

    n_frames: int
    n_rois: int
    n_channels: int
    roi_height: int
    roi_width: int

    def __post_init__(self) -> None:
        for name in (
            "n_frames",
            "n_rois",
            "n_channels",
            "roi_height",
            "roi_width",
        ):
            if type(getattr(self, name)) is not int:
                raise TypeError(f"{name} must be an exact integer.")
        if self.n_frames <= 0:
            raise ValueError("n_frames must be positive.")
        if self.n_rois <= 0:
            raise ValueError("Canonical subject-mask snapshots cannot be empty.")
        if self.n_channels <= 0:
            raise ValueError("n_channels must be positive.")
        if self.roi_height <= 0 or self.roi_width <= 0:
            raise ValueError("ROI height and width must be positive.")

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
        }

    def as_manifest(self) -> dict[str, int]:
        return {
            **self.contract_dimensions,
            "roi_height": self.roi_height,
            "roi_width": self.roi_width,
        }


@dataclass(frozen=True)
class SubjectMaskComponentRegistry:
    """Ordered, exact component labels for one mask snapshot."""

    labels: tuple[str, ...]

    def __post_init__(self) -> None:
        labels = tuple(str(label).strip() for label in self.labels)
        if not labels:
            raise ValueError("At least one subject-mask component is required.")
        if len(labels) != len(set(labels)):
            raise ValueError("Subject-mask component labels must be unique.")
        for label in labels:
            if not _COMPONENT_LABEL.fullmatch(label):
                raise ValueError(
                    "Subject-mask component labels must match ^[a-z][a-z0-9_]*$."
                )
        object.__setattr__(self, "labels", labels)

    def require_dimensions(self, dimensions: SubjectMaskDimensions) -> None:
        if len(self.labels) != dimensions.n_channels:
            raise ValueError(
                "Component label count must equal the declared channel dimension."
            )

    def as_manifest(self) -> dict[str, object]:
        return {
            "schema_id": "palette.subject_mask.component_registry",
            "schema_version": 1,
            "labels": list(self.labels),
            "channel_axis": 1,
            "ordering": "persisted_exact_order",
        }


@dataclass(frozen=True)
class SubjectMaskSchemaIssue:
    code: str
    path: str
    message: str

    def as_manifest(self) -> dict[str, str]:
        return {"code": self.code, "path": self.path, "message": self.message}


class SubjectMaskSchemaError(ValueError):
    def __init__(self, issues: tuple[SubjectMaskSchemaIssue, ...]) -> None:
        self.issues = issues
        detail = "; ".join(
            f"{issue.code} at {issue.path}: {issue.message}" for issue in issues
        )
        super().__init__(
            f"Subject-mask schema validation failed with {len(issues)} "
            f"issue(s): {detail}"
        )


def _issue(code: str, path: str, message: str) -> SubjectMaskSchemaIssue:
    return SubjectMaskSchemaIssue(code=code, path=path, message=message)


def _binding(
    path: str, contract: ArrayContract, *, required: bool = True
) -> ArrayContractBinding:
    return ArrayContractBinding(
        path=path,
        contract_id=contract.schema_id,
        contract_version=contract.schema_version,
        required=required,
    )


_IDENTITY_BINDINGS = (
    _binding("source_crop_row_ids", SUBJECT_MASK_SOURCE_CROP_ROW_IDS_V1),
    _binding("instance_key", DETECTION_INSTANCE_KEY_V1),
    _binding(
        "source_acquisition_frame_index",
        DETECTION_SOURCE_ACQUISITION_FRAME_INDEX_V1,
    ),
    _binding("frame_row_offsets", FRAME_ROW_OFFSETS_V1),
    _binding("source_crop_xywh", CROP_SOURCE_CROP_XYWH_V1),
)

_DERIVED_BINDINGS = (
    _binding("metrics/mask_present", SUBJECT_MASK_PRESENT_V1),
    _binding("metrics/area_px", SUBJECT_MASK_AREA_PX_V1),
    _binding("metrics/centroid_xy", SUBJECT_MASK_CENTROID_XY_V1),
    _binding("metrics/centroid_valid", SUBJECT_MASK_CENTROID_VALID_V1),
    _binding("metrics/bbox_xyxy", SUBJECT_MASK_BBOX_XYXY_V1),
    _binding("metrics/bbox_valid", SUBJECT_MASK_BBOX_VALID_V1),
)

RAW_SUBJECT_MASK_UINT8_BINDINGS = (
    *_IDENTITY_BINDINGS,
    _binding("mask_probs_roi", SUBJECT_MASK_PROBABILITIES_UINT8_V1),
    _binding("masks_roi", DENSE_SUBJECT_MASKS_ROI_V1, required=False),
    _binding("available_channels", SUBJECT_MASK_AVAILABLE_CHANNELS_V1),
    _binding("metrics/prob_max", SUBJECT_MASK_PROB_MAX_V1),
    *_DERIVED_BINDINGS,
)

RAW_SUBJECT_MASK_FLOAT16_BINDINGS = (
    *_IDENTITY_BINDINGS,
    _binding("mask_probs_roi", SUBJECT_MASK_PROBABILITIES_FLOAT16_V1),
    _binding("masks_roi", DENSE_SUBJECT_MASKS_ROI_V1, required=False),
    _binding("available_channels", SUBJECT_MASK_AVAILABLE_CHANNELS_V1),
    _binding("metrics/prob_max", SUBJECT_MASK_PROB_MAX_V1),
    *_DERIVED_BINDINGS,
)

REFINED_SUBJECT_MASK_CORE_BINDINGS = (
    *_IDENTITY_BINDINGS,
    _binding("masks_roi", DENSE_SUBJECT_MASKS_ROI_V1),
    _binding("available_channels", SUBJECT_MASK_AVAILABLE_CHANNELS_V1),
    *_DERIVED_BINDINGS,
)

FORBIDDEN_SUBJECT_MASK_ARRAYS = (
    "detection_indices",
    "detection_source",
    "frame_counts",
    "frame_indices",
    "n_rois",
    "source_detect_row_index",
    "source_frame_indices",
    "source_refined_row_ids",
)


def derive_subject_mask_frame_row_offsets(
    source_acquisition_frame_index: np.ndarray,
    *,
    n_frames: int,
) -> np.ndarray:
    frames = np.asarray(source_acquisition_frame_index)
    if frames.dtype != np.dtype(np.int64) or frames.ndim != 1:
        raise ValueError(
            "source_acquisition_frame_index must be one exact int64 vector."
        )
    if type(n_frames) is not int or n_frames <= 0:
        raise ValueError("n_frames must be a positive exact integer.")
    if np.any(frames < 0) or np.any(frames >= n_frames):
        raise ValueError("source_acquisition_frame_index is outside the frame domain.")
    if frames.size > 1 and np.any(np.diff(frames) < 0):
        raise ValueError("source_acquisition_frame_index must be nondecreasing.")
    counts = np.bincount(frames, minlength=n_frames)
    offsets = np.zeros(n_frames + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts, dtype=np.int64)
    return offsets


def derive_subject_mask_metrics(
    dense_masks: np.ndarray,
) -> dict[str, np.ndarray]:
    """Derive exact binary-mask metrics for a bounded row block."""

    masks = np.asarray(dense_masks)
    if masks.dtype != np.dtype(np.uint8) or masks.ndim != 4:
        raise ValueError("dense_masks must have exact uint8 shape (R,C,H,W).")
    if not np.all((masks == 0) | (masks == 1)):
        raise ValueError("dense_masks must contain only binary 0/1 values.")
    rows, components, _height, _width = masks.shape
    area = np.sum(masks, axis=(2, 3), dtype=np.float32)
    present = area > 0
    centroid = np.zeros((rows, components, 2), dtype=np.float32)
    bbox = np.zeros((rows, components, 4), dtype=np.float32)
    for row in range(rows):
        for component in range(components):
            if not present[row, component]:
                continue
            yy, xx = np.nonzero(masks[row, component])
            centroid[row, component, 0] = np.mean(xx, dtype=np.float32)
            centroid[row, component, 1] = np.mean(yy, dtype=np.float32)
            bbox[row, component] = np.asarray(
                (xx.min(), yy.min(), xx.max() + 1, yy.max() + 1),
                dtype=np.float32,
            )
    return {
        "mask_present": present.astype(bool, copy=False),
        "area_px": area.astype(np.float32, copy=False),
        "centroid_xy": centroid,
        "centroid_valid": present.astype(bool, copy=True),
        "bbox_xyxy": bbox,
        "bbox_valid": present.astype(bool, copy=True),
    }


def _materialize(array: Any) -> np.ndarray:
    if isinstance(array, np.ndarray):
        return array
    try:
        return np.asarray(array[...])
    except (IndexError, KeyError, TypeError):
        return np.asarray(array)


def _rows(array: Any, start: int, stop: int) -> np.ndarray:
    return np.asarray(array[start:stop])


def _row_block_size(array: Any) -> int:
    row_elements = int(np.prod(tuple(int(value) for value in array.shape[1:])))
    row_bytes = max(1, row_elements * int(np.dtype(array.dtype).itemsize))
    return max(1, _CONTENT_SCAN_TARGET_BYTES // row_bytes)


def _validate_contracts(
    *,
    arrays: Mapping[str, Any],
    bindings: tuple[ArrayContractBinding, ...],
    contracts: ArrayContractCatalog,
    dimensions: SubjectMaskDimensions,
) -> tuple[list[SubjectMaskSchemaIssue], set[str]]:
    issues: list[SubjectMaskSchemaIssue] = []
    invalid: set[str] = set()
    allowed = {binding.path for binding in bindings}
    for path in sorted(set(arrays) - allowed):
        issues.append(
            _issue(
                "unexpected_array",
                path,
                "The exact subject-mask schema does not declare this array.",
            )
        )
    for path in FORBIDDEN_SUBJECT_MASK_ARRAYS:
        if path in arrays:
            issues.append(
                _issue(
                    "forbidden_legacy_array",
                    path,
                    "Legacy row/count aliases are outside the canonical schema.",
                )
            )
    for binding in bindings:
        if binding.path not in arrays:
            if binding.required:
                invalid.add(binding.path)
                issues.append(
                    _issue(
                        "missing_required_array",
                        binding.path,
                        "Required subject-mask array is absent.",
                    )
                )
            continue
        contract = contracts.resolve(
            binding.contract_id,
            binding.contract_version,
        )
        try:
            errors = contract.validate_observation(
                arrays[binding.path],
                dimensions=dimensions.contract_dimensions,
            )
        except Exception as exc:
            errors = (f"array metadata is unreadable: {exc}",)
        if errors:
            invalid.add(binding.path)
            issues.extend(
                _issue("array_contract_violation", binding.path, error)
                for error in errors
            )
    return issues, invalid


def _validate_identity(
    *,
    arrays: Mapping[str, Any],
    dimensions: SubjectMaskDimensions,
    invalid: set[str],
    source_crop_arrays: Mapping[str, Any] | None,
) -> list[SubjectMaskSchemaIssue]:
    issues: list[SubjectMaskSchemaIssue] = []
    required = {
        "source_crop_row_ids",
        "instance_key",
        "source_acquisition_frame_index",
        "frame_row_offsets",
        "source_crop_xywh",
    }
    if required & invalid or not required <= set(arrays):
        return issues
    crop_rows = _materialize(arrays["source_crop_row_ids"])
    keys = _materialize(arrays["instance_key"])
    frames = _materialize(arrays["source_acquisition_frame_index"])
    offsets = _materialize(arrays["frame_row_offsets"])
    if np.unique(crop_rows).size != crop_rows.size or np.any(crop_rows < 0):
        issues.append(
            _issue(
                "invalid_source_crop_row_ids",
                "source_crop_row_ids",
                "Crop row IDs must be unique and nonnegative.",
            )
        )
    if np.unique(keys).size != keys.size:
        issues.append(
            _issue(
                "duplicate_instance_key",
                "instance_key",
                "Observation instance keys must be unique within the snapshot.",
            )
        )
    try:
        expected_offsets = derive_subject_mask_frame_row_offsets(
            frames,
            n_frames=dimensions.n_frames,
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
        if not np.array_equal(offsets, expected_offsets):
            issues.append(
                _issue(
                    "frame_row_offsets_mismatch",
                    "frame_row_offsets",
                    "Offsets must exactly index the sorted acquisition-frame rows.",
                )
            )
    if source_crop_arrays is None:
        issues.append(
            _issue(
                "missing_source_crop_evidence",
                "source_crop_row_ids",
                "Canonical masks require the bound crop-v2 rowset as evidence.",
            )
        )
        return issues
    evidence_names = (
        "instance_key",
        "source_acquisition_frame_index",
        "source_crop_xywh",
    )
    missing = [name for name in evidence_names if name not in source_crop_arrays]
    if missing:
        issues.append(
            _issue(
                "incomplete_source_crop_evidence",
                "source_crop_row_ids",
                f"Bound crop evidence is missing {missing!r}.",
            )
        )
        return issues
    source_count = int(source_crop_arrays["instance_key"].shape[0])
    if crop_rows.size and int(crop_rows.max()) >= source_count:
        issues.append(
            _issue(
                "source_crop_row_out_of_bounds",
                "source_crop_row_ids",
                "A selected crop row exceeds the bound crop-v2 row count.",
            )
        )
        return issues
    for name in evidence_names:
        selected = np.asarray(source_crop_arrays[name][crop_rows])
        actual = _materialize(arrays[name])
        if actual.dtype != selected.dtype or not np.array_equal(actual, selected):
            issues.append(
                _issue(
                    "source_crop_selection_mismatch",
                    name,
                    "Array must exactly dtype-preserve the selected crop-v2 rows.",
                )
            )
    if np.dtype(arrays["source_crop_xywh"].dtype) != np.dtype(np.float32):
        issues.append(
            _issue(
                "noncanonical_crop_placement_dtype",
                "source_crop_xywh",
                "Canonical subject masks require crop-v2 float32 placement.",
            )
        )
    return issues


def _compare_metric(
    *,
    actual: np.ndarray,
    expected: np.ndarray,
    path: str,
) -> SubjectMaskSchemaIssue | None:
    if path == "metrics/centroid_xy":
        equal = np.allclose(actual, expected, rtol=0.0, atol=2e-5)
    else:
        equal = np.array_equal(actual, expected)
    if equal:
        return None
    return _issue(
        "derived_metric_mismatch",
        path,
        "Persisted values differ from the authoritative mask surface.",
    )


@dataclass(frozen=True)
class RawSubjectMaskSchema:
    schema_id: str
    schema_version: int
    encoding: SubjectMaskProbabilityEncoding
    probability_contract: ArrayContract
    bindings: tuple[ArrayContractBinding, ...]
    contracts: ArrayContractCatalog

    def __post_init__(self) -> None:
        if not self.schema_id.strip():
            raise ValueError("Raw subject-mask schema ID cannot be empty.")
        if type(self.schema_version) is not int or self.schema_version <= 0:
            raise ValueError("Raw subject-mask schema version must be positive.")
        paths = tuple(binding.path for binding in self.bindings)
        if len(paths) != len(set(paths)):
            raise ValueError("Raw subject-mask binding paths must be unique.")
        probability_bindings = [
            binding for binding in self.bindings if binding.path == "mask_probs_roi"
        ]
        if len(probability_bindings) != 1:
            raise ValueError("Raw schema requires one probability authority binding.")
        probability_binding = probability_bindings[0]
        if (
            probability_binding.contract_id,
            probability_binding.contract_version,
        ) != self.probability_contract.key:
            raise ValueError("Probability binding and contract identity differ.")
        for binding in self.bindings:
            self.contracts.resolve(
                binding.contract_id,
                binding.contract_version,
            )

    @property
    def binding_paths(self) -> tuple[str, ...]:
        return tuple(binding.path for binding in self.bindings)

    def coordinate_contract_manifest(self) -> dict[str, object]:
        """Return the exact coordinate catalog for raw mask surfaces."""

        from fisheye.shared.zarr.coordinate_contracts import (
            array_coordinate_catalog_manifest,
        )

        return array_coordinate_catalog_manifest(self.contracts)

    def validate(
        self,
        arrays: Mapping[str, Any],
        *,
        dimensions: SubjectMaskDimensions,
        components: SubjectMaskComponentRegistry,
        threshold: float,
        source_crop_arrays: Mapping[str, Any] | None,
    ) -> tuple[SubjectMaskSchemaIssue, ...]:
        issues, invalid = _validate_contracts(
            arrays=arrays,
            bindings=self.bindings,
            contracts=self.contracts,
            dimensions=dimensions,
        )
        try:
            components.require_dimensions(dimensions)
        except ValueError as exc:
            issues.append(_issue("component_registry_mismatch", "components", str(exc)))
        if not np.isfinite(threshold) or not 0.0 <= float(threshold) <= 1.0:
            issues.append(
                _issue(
                    "invalid_threshold",
                    "threshold",
                    "Threshold must be finite and within [0,1].",
                )
            )
        issues.extend(
            _validate_identity(
                arrays=arrays,
                dimensions=dimensions,
                invalid=invalid,
                source_crop_arrays=source_crop_arrays,
            )
        )
        if "available_channels" not in invalid and "available_channels" in arrays:
            available = _materialize(arrays["available_channels"])
            if not bool(np.all(available)):
                issues.append(
                    _issue(
                        "unavailable_canonical_component",
                        "available_channels",
                        "Every persisted canonical raw component must be available.",
                    )
                )
        payload_paths = {
            "mask_probs_roi",
            "metrics/prob_max",
            *(binding.path for binding in _DERIVED_BINDINGS),
        }
        if payload_paths & invalid or not payload_paths <= set(arrays):
            return tuple(issues)
        probabilities = arrays["mask_probs_roi"]
        block_rows = _row_block_size(probabilities)
        threshold32 = np.float32(threshold)
        for start in range(0, dimensions.n_rois, block_rows):
            stop = min(dimensions.n_rois, start + block_rows)
            encoded = _rows(probabilities, start, stop)
            if self.encoding is SubjectMaskProbabilityEncoding.LINEAR_UINT8_0_255:
                decoded = encoded.astype(np.float32) / np.float32(255.0)
                expected_prob_max = np.max(encoded, axis=(2, 3)).astype(
                    np.float32, copy=False
                ) / np.float32(255.0)
            else:
                decoded = encoded.astype(np.float32)
                expected_prob_max = np.max(decoded, axis=(2, 3)).astype(
                    np.float32,
                    copy=False,
                )
                if (
                    not np.all(np.isfinite(decoded))
                    or np.any(decoded < 0.0)
                    or np.any(decoded > 1.0)
                ):
                    issues.append(
                        _issue(
                            "invalid_probability_range",
                            "mask_probs_roi",
                            "Float16 probabilities must be finite within [0,1].",
                        )
                    )
                    break
            binary = (decoded >= threshold32).astype(np.uint8)
            expected = derive_subject_mask_metrics(binary)
            expected["prob_max"] = expected_prob_max
            for name in (
                "prob_max",
                "mask_present",
                "area_px",
                "centroid_xy",
                "centroid_valid",
                "bbox_xyxy",
                "bbox_valid",
            ):
                path = f"metrics/{name}"
                mismatch = _compare_metric(
                    actual=_rows(arrays[path], start, stop),
                    expected=expected[name],
                    path=path,
                )
                if mismatch is not None:
                    issues.append(mismatch)
            if "masks_roi" in arrays and "masks_roi" not in invalid:
                if not np.array_equal(_rows(arrays["masks_roi"], start, stop), binary):
                    issues.append(
                        _issue(
                            "threshold_cache_mismatch",
                            "masks_roi",
                            "Optional dense cache differs from thresholded probabilities.",
                        )
                    )
        return tuple(issues)

    def require(self, arrays: Mapping[str, Any], **kwargs: Any) -> None:
        issues = self.validate(arrays, **kwargs)
        if issues:
            raise SubjectMaskSchemaError(issues)

    def as_manifest(
        self,
        *,
        dimensions: SubjectMaskDimensions,
        components: SubjectMaskComponentRegistry,
        threshold: float,
    ) -> dict[str, object]:
        components.require_dimensions(dimensions)
        return {
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "layout": SUBJECT_MASK_LAYOUT,
            "dimensions": dimensions.as_manifest(),
            "components": components.as_manifest(),
            "probability_encoding": self.encoding.value,
            "probability_semantics": "sigmoid_multilabel_logits",
            "output_semantics": "multilabel",
            "overlap_policy": "independent_sigmoid",
            "threshold": float(threshold),
            "bindings": [binding.as_manifest() for binding in self.bindings],
            "invariants": {
                "instances_per_frame": "zero_one_or_many",
                "frame_index": "retained_int64_f_plus_one_offsets",
                "row_order": "nondecreasing_source_acquisition_frame_index",
                "crop_contract": "palette.stage.crop_geometry_v1_float32_placement",
                "legacy_aliases": "forbidden",
            },
        }


@dataclass(frozen=True)
class RefinedSubjectMaskCoreSchema:
    schema_id: str
    schema_version: int
    bindings: tuple[ArrayContractBinding, ...]
    contracts: ArrayContractCatalog

    def __post_init__(self) -> None:
        if not self.schema_id.strip():
            raise ValueError("Refined subject-mask schema ID cannot be empty.")
        if type(self.schema_version) is not int or self.schema_version <= 0:
            raise ValueError("Refined subject-mask schema version must be positive.")
        paths = tuple(binding.path for binding in self.bindings)
        if len(paths) != len(set(paths)):
            raise ValueError("Refined subject-mask binding paths must be unique.")
        if any(not binding.required for binding in self.bindings):
            raise ValueError("Every refined scientific-core binding must be required.")
        for binding in self.bindings:
            self.contracts.resolve(
                binding.contract_id,
                binding.contract_version,
            )

    @property
    def binding_paths(self) -> tuple[str, ...]:
        return tuple(binding.path for binding in self.bindings)

    def coordinate_contract_manifest(self) -> dict[str, object]:
        """Return the exact coordinate catalog for refined mask surfaces."""

        from fisheye.shared.zarr.coordinate_contracts import (
            array_coordinate_catalog_manifest,
        )

        return array_coordinate_catalog_manifest(self.contracts)

    def validate(
        self,
        arrays: Mapping[str, Any],
        *,
        dimensions: SubjectMaskDimensions,
        components: SubjectMaskComponentRegistry,
        source_crop_arrays: Mapping[str, Any] | None,
    ) -> tuple[SubjectMaskSchemaIssue, ...]:
        issues, invalid = _validate_contracts(
            arrays=arrays,
            bindings=self.bindings,
            contracts=self.contracts,
            dimensions=dimensions,
        )
        try:
            components.require_dimensions(dimensions)
        except ValueError as exc:
            issues.append(_issue("component_registry_mismatch", "components", str(exc)))
        issues.extend(
            _validate_identity(
                arrays=arrays,
                dimensions=dimensions,
                invalid=invalid,
                source_crop_arrays=source_crop_arrays,
            )
        )
        if "available_channels" not in invalid and "available_channels" in arrays:
            if not bool(np.all(_materialize(arrays["available_channels"]))):
                issues.append(
                    _issue(
                        "unavailable_canonical_component",
                        "available_channels",
                        "Every published refined component must be available.",
                    )
                )
        payload_paths = {"masks_roi", *(binding.path for binding in _DERIVED_BINDINGS)}
        if payload_paths & invalid or not payload_paths <= set(arrays):
            return tuple(issues)
        masks = arrays["masks_roi"]
        for start in range(0, dimensions.n_rois, _row_block_size(masks)):
            stop = min(
                dimensions.n_rois,
                start + _row_block_size(masks),
            )
            values = _rows(masks, start, stop)
            try:
                expected = derive_subject_mask_metrics(values)
            except ValueError as exc:
                issues.append(_issue("invalid_dense_authority", "masks_roi", str(exc)))
                break
            for name in (
                "mask_present",
                "area_px",
                "centroid_xy",
                "centroid_valid",
                "bbox_xyxy",
                "bbox_valid",
            ):
                path = f"metrics/{name}"
                mismatch = _compare_metric(
                    actual=_rows(arrays[path], start, stop),
                    expected=expected[name],
                    path=path,
                )
                if mismatch is not None:
                    issues.append(mismatch)
        return tuple(issues)

    def require(self, arrays: Mapping[str, Any], **kwargs: Any) -> None:
        issues = self.validate(arrays, **kwargs)
        if issues:
            raise SubjectMaskSchemaError(issues)

    def as_manifest(
        self,
        *,
        dimensions: SubjectMaskDimensions,
        components: SubjectMaskComponentRegistry,
    ) -> dict[str, object]:
        components.require_dimensions(dimensions)
        return {
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "layout": SUBJECT_MASK_LAYOUT,
            "dimensions": dimensions.as_manifest(),
            "components": components.as_manifest(),
            "authority": "dense_binary_masks_roi",
            "bindings": [binding.as_manifest() for binding in self.bindings],
            "invariants": {
                "instances_per_frame": "zero_one_or_many",
                "frame_index": "retained_int64_f_plus_one_offsets",
                "row_order": "nondecreasing_source_acquisition_frame_index",
                "crop_contract": "palette.stage.crop_geometry_v1_float32_placement",
                "derived_surfaces": "must_exactly_match_dense_authority",
                "legacy_aliases": "forbidden",
            },
        }


RAW_SUBJECT_MASK_UINT8_SCHEMA_V1 = RawSubjectMaskSchema(
    schema_id=RAW_SUBJECT_MASK_UINT8_SCHEMA_ID,
    schema_version=RAW_SUBJECT_MASK_SCHEMA_VERSION,
    encoding=SubjectMaskProbabilityEncoding.LINEAR_UINT8_0_255,
    probability_contract=SUBJECT_MASK_PROBABILITIES_UINT8_V1,
    bindings=RAW_SUBJECT_MASK_UINT8_BINDINGS,
    contracts=RAW_SUBJECT_MASK_UINT8_ARRAY_CONTRACTS,
)

RAW_SUBJECT_MASK_FLOAT16_SCHEMA_V1 = RawSubjectMaskSchema(
    schema_id=RAW_SUBJECT_MASK_FLOAT16_SCHEMA_ID,
    schema_version=RAW_SUBJECT_MASK_SCHEMA_VERSION,
    encoding=SubjectMaskProbabilityEncoding.UNIT_FLOAT16,
    probability_contract=SUBJECT_MASK_PROBABILITIES_FLOAT16_V1,
    bindings=RAW_SUBJECT_MASK_FLOAT16_BINDINGS,
    contracts=RAW_SUBJECT_MASK_FLOAT16_ARRAY_CONTRACTS,
)

REFINED_SUBJECT_MASK_CORE_SCHEMA_V1 = RefinedSubjectMaskCoreSchema(
    schema_id=REFINED_SUBJECT_MASK_CORE_SCHEMA_ID,
    schema_version=REFINED_SUBJECT_MASK_CORE_SCHEMA_VERSION,
    bindings=REFINED_SUBJECT_MASK_CORE_BINDINGS,
    contracts=REFINED_SUBJECT_MASK_CORE_ARRAY_CONTRACTS,
)


__all__ = [
    "FORBIDDEN_SUBJECT_MASK_ARRAYS",
    "RAW_SUBJECT_MASK_FLOAT16_SCHEMA_ID",
    "RAW_SUBJECT_MASK_FLOAT16_SCHEMA_V1",
    "RAW_SUBJECT_MASK_SCHEMA_VERSION",
    "RAW_SUBJECT_MASK_UINT8_SCHEMA_ID",
    "RAW_SUBJECT_MASK_UINT8_SCHEMA_V1",
    "REFINED_SUBJECT_MASK_CORE_SCHEMA_ID",
    "REFINED_SUBJECT_MASK_CORE_SCHEMA_V1",
    "REFINED_SUBJECT_MASK_CORE_SCHEMA_VERSION",
    "SUBJECT_MASK_COMPONENT_SCHEMAS",
    "SUBJECT_V1_LR_COMPONENT_SCHEMA",
    "SUBJECT_V1_UNION_COMPONENT_SCHEMA",
    "RawSubjectMaskSchema",
    "RefinedSubjectMaskCoreSchema",
    "SubjectMaskComponentSchema",
    "SubjectMaskComponentRegistry",
    "SubjectMaskDimensions",
    "SubjectMaskProbabilityEncoding",
    "SubjectMaskSchemaError",
    "SubjectMaskSchemaIssue",
    "derive_subject_mask_frame_row_offsets",
    "derive_subject_mask_metrics",
    "resolve_subject_mask_component_schema",
]
