"""Versioned logical schema for canonical sparse detection instances.

This module is deliberately independent of Zarr I/O and physical storage
planning. It validates NumPy-compatible arrays, binds their relative archive
paths to exact :class:`ArrayContract` versions, and emits JSON-safe manifests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from fisheye.shared.zarr.array_contracts import (
    DETECTION_ARRAY_CONTRACTS,
    DETECTION_BBOX_IMG_XYXY_V1,
    DETECTION_BBOX_NORM_COORDS_V1,
    DETECTION_CENTERS_IMG_XY_V1,
    DETECTION_CLASS_IDS_V1,
    DETECTION_FRAME_INDICES_V1,
    DETECTION_INSTANCE_KEY_V1,
    DETECTION_SCORES_V1,
    DETECTION_SOURCE_ACQUISITION_FRAME_INDEX_V1,
    FRAME_ROW_OFFSETS_V1,
    ArrayContract,
    ArrayContractBinding,
    ArrayContractCatalog,
)


CANONICAL_DETECTION_SCHEMA_ID = "palette.stage.canonical_detection"
CANONICAL_DETECTION_SCHEMA_VERSION = 1
CANONICAL_DETECTION_LAYOUT = "sparse_instances_with_frame_row_offsets_v1"
CANONICAL_DETECTION_INSTANCE_GROUP = "instances"

MAX_CANONICAL_CLASS_ID = int(np.iinfo(np.uint16).max)


def _instance_path(name: str) -> str:
    return f"{CANONICAL_DETECTION_INSTANCE_GROUP}/{name}"


_CONTRACT_BY_NAME: tuple[tuple[str, ArrayContract], ...] = (
    ("frame_indices", DETECTION_FRAME_INDICES_V1),
    (
        "source_acquisition_frame_index",
        DETECTION_SOURCE_ACQUISITION_FRAME_INDEX_V1,
    ),
    ("instance_key", DETECTION_INSTANCE_KEY_V1),
    ("bbox_norm_coords", DETECTION_BBOX_NORM_COORDS_V1),
    ("bbox_img_xyxy", DETECTION_BBOX_IMG_XYXY_V1),
    ("centers_img_xy", DETECTION_CENTERS_IMG_XY_V1),
    ("scores", DETECTION_SCORES_V1),
    ("class_ids", DETECTION_CLASS_IDS_V1),
    ("frame_row_offsets", FRAME_ROW_OFFSETS_V1),
)

CANONICAL_DETECTION_BINDINGS = tuple(
    ArrayContractBinding(
        path=_instance_path(name),
        contract_id=contract.schema_id,
        contract_version=contract.schema_version,
        required=True,
    )
    for name, contract in _CONTRACT_BY_NAME
)

FORBIDDEN_CANONICAL_DETECTION_COUNT_PATHS = (
    "frame_counts",
    "n_detections",
    _instance_path("frame_counts"),
    _instance_path("n_detections"),
)


@dataclass(frozen=True)
class CanonicalDetectionDimensions:
    """Concrete logical dimensions and source-camera extent for one run."""

    n_frames: int
    n_instances: int
    source_width: int
    source_height: int

    def __post_init__(self) -> None:
        for name in ("n_frames", "n_instances", "source_width", "source_height"):
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"{name} must be an exact integer.")
        if self.n_frames < 0:
            raise ValueError("n_frames cannot be negative.")
        if self.n_instances < 0:
            raise ValueError("n_instances cannot be negative.")
        if self.source_width <= 0 or self.source_height <= 0:
            raise ValueError("Source-camera width and height must be positive.")
        if self.n_frames > int(np.iinfo(np.int32).max):
            raise ValueError("n_frames exceeds the canonical int32 frame domain.")

    @property
    def contract_dimensions(self) -> dict[str, int]:
        return {
            "n_frames": self.n_frames,
            "n_instances": self.n_instances,
            "n_frame_boundaries": self.n_frames + 1,
        }

    def as_manifest(self) -> dict[str, int]:
        return {
            **self.contract_dimensions,
            "source_width": self.source_width,
            "source_height": self.source_height,
        }


@dataclass(frozen=True)
class DetectionSchemaIssue:
    """One structured stage-schema validation failure."""

    code: str
    path: str
    message: str

    def as_manifest(self) -> dict[str, str]:
        return {
            "code": self.code,
            "path": self.path,
            "message": self.message,
        }


class CanonicalDetectionSchemaError(ValueError):
    """Raised when arrays fail the canonical detection schema."""

    def __init__(self, issues: tuple[DetectionSchemaIssue, ...]) -> None:
        self.issues = issues
        detail = "; ".join(
            f"{issue.code} at {issue.path}: {issue.message}" for issue in issues
        )
        super().__init__(
            f"Canonical detection schema validation failed with "
            f"{len(issues)} issue(s): {detail}"
        )


def _issue(code: str, path: str, message: str) -> DetectionSchemaIssue:
    return DetectionSchemaIssue(code=code, path=path, message=message)


def _materialize(array: Any) -> np.ndarray:
    if isinstance(array, np.ndarray):
        return array
    try:
        return np.asarray(array[...])
    except (IndexError, KeyError, TypeError):
        return np.asarray(array)


def _derive_source_camera_geometry(
    bbox_norm_coords: np.ndarray,
    *,
    source_width: int,
    source_height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact float32 source-camera pixel boxes and centers."""

    half = np.asarray(0.5, dtype=np.float32)
    width_px = np.asarray(source_width, dtype=np.float32)
    height_px = np.asarray(source_height, dtype=np.float32)
    cx = bbox_norm_coords[:, 0]
    cy = bbox_norm_coords[:, 1]
    width = bbox_norm_coords[:, 2]
    height = bbox_norm_coords[:, 3]
    bbox_img = np.column_stack(
        (
            (cx - width * half) * width_px,
            (cy - height * half) * height_px,
            (cx + width * half) * width_px,
            (cy + height * half) * height_px,
        )
    ).astype(np.float32, copy=False)
    centers = np.column_stack(
        (
            (bbox_img[:, 0] + bbox_img[:, 2]) * half,
            (bbox_img[:, 1] + bbox_img[:, 3]) * half,
        )
    ).astype(np.float32, copy=False)
    return (
        np.array(bbox_img, copy=True, order="C"),
        np.array(centers, copy=True, order="C"),
    )


@dataclass(frozen=True)
class CanonicalDetectionSchema:
    """Exact sparse-instance schema and its cross-array invariants."""

    schema_id: str
    schema_version: int
    bindings: tuple[ArrayContractBinding, ...]
    contracts: ArrayContractCatalog

    def __post_init__(self) -> None:
        if not self.schema_id.strip():
            raise ValueError("schema_id cannot be empty.")
        if type(self.schema_version) is not int or self.schema_version <= 0:
            raise ValueError("schema_version must be a positive exact integer.")
        paths = [binding.path for binding in self.bindings]
        if len(paths) != len(set(paths)):
            raise ValueError("Canonical detection binding paths must be unique.")
        for binding in self.bindings:
            if not binding.required:
                raise ValueError("Every canonical detection binding must be required.")
            self.contracts.resolve(
                binding.contract_id,
                binding.contract_version,
            )

    @property
    def binding_paths(self) -> tuple[str, ...]:
        return tuple(binding.path for binding in self.bindings)

    def validate(
        self,
        arrays: Mapping[str, Any],
        *,
        dimensions: CanonicalDetectionDimensions,
    ) -> tuple[DetectionSchemaIssue, ...]:
        """Return every logical and cross-array validation issue."""

        issues: list[DetectionSchemaIssue] = []
        invalid_paths: set[str] = set()

        for path in FORBIDDEN_CANONICAL_DETECTION_COUNT_PATHS:
            if path in arrays:
                issues.append(
                    _issue(
                        "forbidden_count_binding",
                        path,
                        "Count vectors are compatibility-only and cannot be bound "
                        "by the canonical sparse-instance schema.",
                    )
                )

        for binding in self.bindings:
            path = binding.path
            if path not in arrays:
                invalid_paths.add(path)
                issues.append(
                    _issue(
                        "missing_required_array",
                        path,
                        "Required canonical detection array is absent.",
                    )
                )
                continue
            contract = self.contracts.resolve(
                binding.contract_id,
                binding.contract_version,
            )
            try:
                contract_errors = contract.validate_observation(
                    arrays[path],
                    dimensions=dimensions.contract_dimensions,
                )
            except Exception as exc:
                contract_errors = (f"array metadata is unreadable: {exc}",)
            if contract_errors:
                invalid_paths.add(path)
                issues.extend(
                    _issue("array_contract_violation", path, error)
                    for error in contract_errors
                )

        values: dict[str, np.ndarray] = {}
        for path in self.binding_paths:
            if path in invalid_paths:
                continue
            try:
                values[path] = _materialize(arrays[path])
            except Exception as exc:
                invalid_paths.add(path)
                issues.append(
                    _issue(
                        "array_read_failure",
                        path,
                        f"Array values could not be materialized: {exc}",
                    )
                )

        frame_path = _instance_path("frame_indices")
        source_frame_path = _instance_path("source_acquisition_frame_index")
        key_path = _instance_path("instance_key")
        bbox_norm_path = _instance_path("bbox_norm_coords")
        bbox_img_path = _instance_path("bbox_img_xyxy")
        centers_path = _instance_path("centers_img_xy")
        scores_path = _instance_path("scores")
        class_path = _instance_path("class_ids")
        offsets_path = _instance_path("frame_row_offsets")

        frames = values.get(frame_path)
        frames_in_bounds = False
        if frames is not None:
            frames_in_bounds = bool(
                np.all(frames >= 0) and np.all(frames < dimensions.n_frames)
            )
            if not frames_in_bounds:
                issues.append(
                    _issue(
                        "frame_index_out_of_bounds",
                        frame_path,
                        "Every frame index must be in [0, n_frames).",
                    )
                )
            if frames.size > 1 and np.any(np.diff(frames) < 0):
                issues.append(
                    _issue(
                        "frame_indices_not_sorted",
                        frame_path,
                        "Instance rows must be contiguous in nondecreasing frame order.",
                    )
                )

        source_frames = values.get(source_frame_path)
        if frames is not None and source_frames is not None:
            if not np.array_equal(frames.astype(np.int64), source_frames):
                issues.append(
                    _issue(
                        "source_frame_identity_mismatch",
                        source_frame_path,
                        "Full-acquisition canonical detections require the exact "
                        "widened frame_indices identity mapping.",
                    )
                )

        keys = values.get(key_path)
        if keys is not None and np.unique(keys).shape[0] != keys.shape[0]:
            issues.append(
                _issue(
                    "duplicate_instance_key",
                    key_path,
                    "Every canonical detection instance_key must be unique.",
                )
            )

        bbox_norm = values.get(bbox_norm_path)
        valid_bbox_norm = False
        if bbox_norm is not None:
            half = np.asarray(0.5, dtype=np.float32)
            one = np.asarray(1.0, dtype=np.float32)
            x_min = bbox_norm[:, 0] - bbox_norm[:, 2] * half
            y_min = bbox_norm[:, 1] - bbox_norm[:, 3] * half
            x_max = bbox_norm[:, 0] + bbox_norm[:, 2] * half
            y_max = bbox_norm[:, 1] + bbox_norm[:, 3] * half
            valid_bbox_norm = bool(
                np.isfinite(bbox_norm).all()
                and np.all(bbox_norm[:, 2] > 0)
                and np.all(bbox_norm[:, 3] > 0)
                and np.all(x_min >= 0)
                and np.all(y_min >= 0)
                and np.all(x_max <= one)
                and np.all(y_max <= one)
            )
            if not valid_bbox_norm:
                issues.append(
                    _issue(
                        "invalid_bbox_norm_coords",
                        bbox_norm_path,
                        "Boxes must be finite positive-area cx,cy,w,h rows "
                        "contained in the normalized source-camera extent.",
                    )
                )

        bbox_img = values.get(bbox_img_path)
        centers = values.get(centers_path)
        if bbox_img is not None and not np.isfinite(bbox_img).all():
            issues.append(
                _issue(
                    "nonfinite_bbox_img_xyxy",
                    bbox_img_path,
                    "Pixel bbox projections must be finite.",
                )
            )
        if centers is not None and not np.isfinite(centers).all():
            issues.append(
                _issue(
                    "nonfinite_centers_img_xy",
                    centers_path,
                    "Pixel center projections must be finite.",
                )
            )
        if valid_bbox_norm and bbox_img is not None and centers is not None:
            expected_bbox, expected_centers = _derive_source_camera_geometry(
                bbox_norm,
                source_width=dimensions.source_width,
                source_height=dimensions.source_height,
            )
            if not np.array_equal(bbox_img, expected_bbox):
                issues.append(
                    _issue(
                        "bbox_img_projection_mismatch",
                        bbox_img_path,
                        "bbox_img_xyxy must be the exact float32 source-camera "
                        "projection of bbox_norm_coords.",
                    )
                )
            if not np.array_equal(centers, expected_centers):
                issues.append(
                    _issue(
                        "center_projection_mismatch",
                        centers_path,
                        "centers_img_xy must be the exact float32 midpoint of "
                        "bbox_img_xyxy.",
                    )
                )

        scores = values.get(scores_path)
        if scores is not None and not bool(
            np.isfinite(scores).all()
            and np.all(scores >= np.float32(0.0))
            and np.all(scores <= np.float32(1.0))
        ):
            issues.append(
                _issue(
                    "invalid_score",
                    scores_path,
                    "Scores must be finite float32 values in [0, 1].",
                )
            )

        class_ids = values.get(class_path)
        if class_ids is not None and not bool(
            np.all(class_ids >= 0) and np.all(class_ids <= MAX_CANONICAL_CLASS_ID)
        ):
            issues.append(
                _issue(
                    "invalid_class_id",
                    class_path,
                    f"Class IDs must be in [0, {MAX_CANONICAL_CLASS_ID}] and "
                    "cannot use a missing-value sentinel.",
                )
            )

        offsets = values.get(offsets_path)
        if offsets is not None:
            starts_at_zero = bool(offsets.size and int(offsets[0]) == 0)
            monotonic = bool(offsets.size and np.all(np.diff(offsets) >= 0))
            ends_at_instances = bool(
                offsets.size and int(offsets[-1]) == dimensions.n_instances
            )
            if not starts_at_zero:
                issues.append(
                    _issue(
                        "offset_start_mismatch",
                        offsets_path,
                        "frame_row_offsets must start at zero.",
                    )
                )
            if not monotonic:
                issues.append(
                    _issue(
                        "offsets_not_monotonic",
                        offsets_path,
                        "frame_row_offsets must be nondecreasing.",
                    )
                )
            if not ends_at_instances:
                issues.append(
                    _issue(
                        "offset_end_mismatch",
                        offsets_path,
                        "The final frame_row_offsets value must equal n_instances.",
                    )
                )
        if frames is not None and frames_in_bounds and offsets is not None:
            counts = np.bincount(
                frames.astype(np.int64, copy=False),
                minlength=dimensions.n_frames,
            )
            expected_offsets = np.zeros(dimensions.n_frames + 1, dtype=np.int64)
            if dimensions.n_frames:
                expected_offsets[1:] = np.cumsum(counts, dtype=np.int64)
            if not np.array_equal(offsets, expected_offsets):
                issues.append(
                    _issue(
                        "frame_row_offsets_mismatch",
                        offsets_path,
                        "Offsets must exactly describe the contiguous frame_indices "
                        "instance ranges.",
                    )
                )

        return tuple(issues)

    def require(
        self,
        arrays: Mapping[str, Any],
        *,
        dimensions: CanonicalDetectionDimensions,
    ) -> None:
        """Raise a structured error unless all stage invariants hold."""

        issues = self.validate(arrays, dimensions=dimensions)
        if issues:
            raise CanonicalDetectionSchemaError(issues)

    def as_manifest(
        self,
        *,
        dimensions: CanonicalDetectionDimensions,
    ) -> dict[str, object]:
        """Return the exact JSON-safe schema and concrete run dimensions."""

        return {
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "stage": "detect",
            "layout": CANONICAL_DETECTION_LAYOUT,
            "base_path": "detect_runs/<run>",
            "instance_group": CANONICAL_DETECTION_INSTANCE_GROUP,
            "dimensions": dimensions.as_manifest(),
            "bindings": [binding.as_manifest() for binding in self.bindings],
            "forbidden_count_bindings": list(FORBIDDEN_CANONICAL_DETECTION_COUNT_PATHS),
            "array_contracts": self.contracts.as_manifest(),
            "invariants": {
                "row_order": "frame_indices_nondecreasing",
                "within_frame_order": "persisted_producer_order_no_rank_semantics",
                "row_identity": "instance_key",
                "frame_lookup": "frame_row_offsets_csr",
                "instances_per_frame": "zero_one_or_many",
                "missing_instance_representation": "absent_row",
                "nullability": "forbidden",
                "semantic_sentinels": "forbidden",
                "physical_fill_semantics": "initialization_only_not_missing",
                "publication_completeness": (
                    "all_values_written_and_validated_before_visibility"
                ),
                "class_id_range": [0, MAX_CANONICAL_CLASS_ID],
                "geometry_authority": "bbox_norm_coords",
                "geometry_projections": ["bbox_img_xyxy", "centers_img_xy"],
                "source_frame_authority": "source_acquisition_frame_index",
                "source_frame_mapping": "full_acquisition_identity",
                "instance_key_derivation": (
                    "validated_by_recording_bound_publication_contract"
                ),
            },
        }


CANONICAL_DETECTION_SCHEMA_V1 = CanonicalDetectionSchema(
    schema_id=CANONICAL_DETECTION_SCHEMA_ID,
    schema_version=CANONICAL_DETECTION_SCHEMA_VERSION,
    bindings=CANONICAL_DETECTION_BINDINGS,
    contracts=DETECTION_ARRAY_CONTRACTS,
)


__all__ = [
    "CANONICAL_DETECTION_BINDINGS",
    "CANONICAL_DETECTION_INSTANCE_GROUP",
    "CANONICAL_DETECTION_LAYOUT",
    "CANONICAL_DETECTION_SCHEMA_ID",
    "CANONICAL_DETECTION_SCHEMA_V1",
    "CANONICAL_DETECTION_SCHEMA_VERSION",
    "FORBIDDEN_CANONICAL_DETECTION_COUNT_PATHS",
    "MAX_CANONICAL_CLASS_ID",
    "CanonicalDetectionDimensions",
    "CanonicalDetectionSchema",
    "CanonicalDetectionSchemaError",
    "DetectionSchemaIssue",
]
