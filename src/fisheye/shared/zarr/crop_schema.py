"""Exact logical schema for immutable geometry-only crop observations.

The schema deliberately contains no crop pixels.  It binds one complete
refined-detection rowset to an explicit crop policy and freezes the exact
source-camera extraction geometry that downstream materializers must use.
Physical Zarr layout and publication are separate concerns.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Any, Mapping

import numpy as np

from fisheye.shared.zarr.array_contracts import (
    CROP_ARRAY_CONTRACTS,
    CROP_BBOX_ROI_XYXY_V1,
    CROP_FRAME_INDICES_V1,
    CROP_ROI_COORDINATES_FULL_V1,
    CROP_ROI_SIZES_FULL_V1,
    CROP_SOURCE_CROP_XYWH_V1,
    CROP_SOURCE_REFINED_ROW_IDS_V1,
    CROP_SOURCE_ROW_SIGNATURE_V1,
    DETECTION_BBOX_IMG_XYXY_V1,
    DETECTION_BBOX_NORM_COORDS_V1,
    DETECTION_CENTERS_IMG_XY_V1,
    DETECTION_INSTANCE_KEY_V1,
    DETECTION_SOURCE_ACQUISITION_FRAME_INDEX_V1,
    FRAME_ROW_OFFSETS_V1,
    ArrayContract,
    ArrayContractBinding,
    ArrayContractCatalog,
)
from fisheye.shared.zarr.detection_schema import (
    derive_canonical_detection_geometry,
)
from fisheye.shared.zarr.manifest_digest import (
    CANONICAL_JSON_DIGEST_ALGORITHM,
    canonical_json_sha256,
)


CROP_GEOMETRY_SCHEMA_ID = "palette.stage.crop_geometry"
CROP_GEOMETRY_SCHEMA_VERSION = 1
CROP_GEOMETRY_LAYOUT = "geometry_only_sparse_rows_with_frame_row_offsets_v1"
CROP_GEOMETRY_POLICY_SCHEMA_ID = "palette.crop_geometry_policy"
CROP_GEOMETRY_POLICY_SCHEMA_VERSION = 1

_PURPOSE_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


class CropSizeMode(str, Enum):
    """How extraction-window sizes are declared within one crop run."""

    FIXED_PER_RUN = "fixed_per_run"
    VARIABLE_PER_ROW = "variable_per_row"


class CropPaddingMode(str, Enum):
    """Source-frame behavior for extraction windows crossing an image edge."""

    REQUIRE_FULLY_CONTAINED = "require_fully_contained"
    ZERO_OUTSIDE_SOURCE_FRAME = "zero_outside_source_frame"


@dataclass(frozen=True)
class CropGeometryPolicy:
    """Versioned placement and size policy, independent of detection identity."""

    purpose: str
    size_mode: CropSizeMode
    fixed_size_wh: tuple[int, int] | None = None
    padding_mode: CropPaddingMode = CropPaddingMode.ZERO_OUTSIDE_SOURCE_FRAME

    def __post_init__(self) -> None:
        purpose = str(self.purpose).strip()
        if not _PURPOSE_PATTERN.fullmatch(purpose):
            raise ValueError(
                "purpose must match ^[a-z][a-z0-9_]*$ for stable identity."
            )
        object.__setattr__(self, "purpose", purpose)
        object.__setattr__(self, "size_mode", CropSizeMode(self.size_mode))
        object.__setattr__(self, "padding_mode", CropPaddingMode(self.padding_mode))

        size = self.fixed_size_wh
        if self.size_mode is CropSizeMode.FIXED_PER_RUN:
            if not isinstance(size, tuple) or len(size) != 2:
                raise ValueError(
                    "fixed_per_run requires fixed_size_wh=(width, height)."
                )
            if any(type(value) is not int or value <= 0 for value in size):
                raise ValueError(
                    "fixed_size_wh values must be positive exact integers."
                )
        elif size is not None:
            raise ValueError(
                "variable_per_row derives sizes from roi_sizes_full and cannot "
                "declare fixed_size_wh."
            )

    @property
    def payload(self) -> dict[str, object]:
        return {
            "schema_id": CROP_GEOMETRY_POLICY_SCHEMA_ID,
            "schema_version": CROP_GEOMETRY_POLICY_SCHEMA_VERSION,
            "purpose": self.purpose,
            "placement": {
                "center_source": "persisted_centers_img_xy",
                "center_rounding": "numpy_round_ties_to_even_v1",
                "top_left_rule": "rounded_center_minus_floor_size_over_two",
                "size_mode": self.size_mode.value,
                "fixed_size_wh": (
                    None
                    if self.fixed_size_wh is None
                    else list(self.fixed_size_wh)
                ),
                "padding_mode": self.padding_mode.value,
            },
        }

    @property
    def payload_digest(self) -> str:
        return canonical_json_sha256(self.payload)

    def as_manifest(self) -> dict[str, object]:
        return {
            "payload": self.payload,
            "payload_digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
            "payload_digest": self.payload_digest,
        }


def crop_geometry_policy_from_manifest(
    value: Mapping[str, Any],
) -> CropGeometryPolicy:
    """Parse and require the exact canonical crop-policy envelope."""

    if set(value) != {
        "payload",
        "payload_digest_algorithm",
        "payload_digest",
    }:
        raise ValueError("Crop policy envelope has an unexpected field set.")
    if value.get("payload_digest_algorithm") != CANONICAL_JSON_DIGEST_ALGORITHM:
        raise ValueError("Crop policy digest algorithm mismatch.")
    payload = value.get("payload")
    if not isinstance(payload, Mapping):
        raise TypeError("Crop policy payload must be an object.")
    if value.get("payload_digest") != canonical_json_sha256(payload):
        raise ValueError("Crop policy payload digest mismatch.")
    if set(payload) != {"schema_id", "schema_version", "purpose", "placement"}:
        raise ValueError("Crop policy payload has an unexpected field set.")
    if (
        payload.get("schema_id") != CROP_GEOMETRY_POLICY_SCHEMA_ID
        or payload.get("schema_version") != CROP_GEOMETRY_POLICY_SCHEMA_VERSION
    ):
        raise ValueError("Crop policy schema identity mismatch.")
    placement = payload.get("placement")
    if not isinstance(placement, Mapping) or set(placement) != {
        "center_source",
        "center_rounding",
        "top_left_rule",
        "size_mode",
        "fixed_size_wh",
        "padding_mode",
    }:
        raise ValueError("Crop policy placement has an unexpected field set.")
    if placement.get("center_source") != "persisted_centers_img_xy":
        raise ValueError("Crop policy center source mismatch.")
    if placement.get("center_rounding") != "numpy_round_ties_to_even_v1":
        raise ValueError("Crop policy center rounding mismatch.")
    if (
        placement.get("top_left_rule")
        != "rounded_center_minus_floor_size_over_two"
    ):
        raise ValueError("Crop policy top-left rule mismatch.")
    raw_size = placement.get("fixed_size_wh")
    fixed_size = None
    if raw_size is not None:
        if not isinstance(raw_size, list) or len(raw_size) != 2:
            raise ValueError("Crop policy fixed_size_wh must be null or [width,height].")
        if any(type(item) is not int for item in raw_size):
            raise TypeError("Crop policy fixed_size_wh values must be exact integers.")
        fixed_size = (raw_size[0], raw_size[1])
    policy = CropGeometryPolicy(
        purpose=payload.get("purpose"),
        size_mode=placement.get("size_mode"),
        fixed_size_wh=fixed_size,
        padding_mode=placement.get("padding_mode"),
    )
    if dict(value) != policy.as_manifest():
        raise ValueError("Crop policy is not in canonical persisted form.")
    return policy


_CONTRACT_BY_PATH: tuple[tuple[str, ArrayContract], ...] = (
    ("instance_key", DETECTION_INSTANCE_KEY_V1),
    ("source_refined_row_ids", CROP_SOURCE_REFINED_ROW_IDS_V1),
    ("frame_indices", CROP_FRAME_INDICES_V1),
    (
        "source_acquisition_frame_index",
        DETECTION_SOURCE_ACQUISITION_FRAME_INDEX_V1,
    ),
    ("frame_row_offsets", FRAME_ROW_OFFSETS_V1),
    ("bbox_norm_coords", DETECTION_BBOX_NORM_COORDS_V1),
    ("bbox_img_xyxy", DETECTION_BBOX_IMG_XYXY_V1),
    ("centers_img_xy", DETECTION_CENTERS_IMG_XY_V1),
    ("roi_coordinates_full", CROP_ROI_COORDINATES_FULL_V1),
    ("roi_sizes_full", CROP_ROI_SIZES_FULL_V1),
    ("source_crop_xywh", CROP_SOURCE_CROP_XYWH_V1),
    ("bbox_roi_xyxy", CROP_BBOX_ROI_XYXY_V1),
    ("source_row_signature", CROP_SOURCE_ROW_SIGNATURE_V1),
)

CROP_GEOMETRY_BINDINGS = tuple(
    ArrayContractBinding(
        path=path,
        contract_id=contract.schema_id,
        contract_version=contract.schema_version,
        required=True,
    )
    for path, contract in _CONTRACT_BY_PATH
)

FORBIDDEN_CROP_GEOMETRY_ARRAY_PATHS = (
    "roi_images",
    "roi_images_delta",
    "frame_counts",
    "n_detections",
    "detection_indices",
    "source_frame_indices",
)


@dataclass(frozen=True)
class CropDimensions:
    """Concrete row/frame/source-camera dimensions for one crop snapshot."""

    n_frames: int
    n_instances: int
    source_width: int
    source_height: int

    def __post_init__(self) -> None:
        for name in ("n_frames", "n_instances", "source_width", "source_height"):
            if type(getattr(self, name)) is not int:
                raise TypeError(f"{name} must be an exact integer.")
        if self.n_frames <= 0:
            raise ValueError("n_frames must be positive for a crop snapshot.")
        if self.n_instances < 0:
            raise ValueError("n_instances cannot be negative.")
        if self.source_width <= 0 or self.source_height <= 0:
            raise ValueError("Source-camera width and height must be positive.")

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
class CropSchemaIssue:
    code: str
    path: str
    message: str

    def as_manifest(self) -> dict[str, str]:
        return {"code": self.code, "path": self.path, "message": self.message}


class CropGeometrySchemaError(ValueError):
    """Raised when a row table violates the crop geometry v1 contract."""

    def __init__(self, issues: tuple[CropSchemaIssue, ...]) -> None:
        self.issues = issues
        detail = "; ".join(
            f"{issue.code} at {issue.path}: {issue.message}" for issue in issues
        )
        super().__init__(
            f"Crop geometry schema validation failed with {len(issues)} "
            f"issue(s): {detail}"
        )


def _issue(code: str, path: str, message: str) -> CropSchemaIssue:
    return CropSchemaIssue(code=code, path=path, message=message)


def _materialize(array: Any) -> np.ndarray:
    if isinstance(array, np.ndarray):
        return array
    try:
        return np.asarray(array[...])
    except (IndexError, KeyError, TypeError):
        return np.asarray(array)


def derive_frame_row_offsets(
    frame_indices: np.ndarray,
    *,
    n_frames: int,
) -> np.ndarray:
    """Return the exact F+1 CSR index for sorted acquisition-frame rows."""

    frames = np.asarray(frame_indices)
    if frames.ndim != 1 or frames.dtype != np.dtype(np.int64):
        raise ValueError("frame_indices must be one exact int64 vector.")
    if type(n_frames) is not int or n_frames <= 0:
        raise ValueError("n_frames must be a positive exact integer.")
    if np.any(frames < 0) or np.any(frames >= n_frames):
        raise ValueError("frame_indices contains an out-of-bounds frame.")
    if frames.size > 1 and np.any(np.diff(frames) < 0):
        raise ValueError("frame_indices must be nondecreasing.")
    counts = np.bincount(frames, minlength=n_frames)
    offsets = np.zeros(n_frames + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts, dtype=np.int64)
    return offsets


def derive_crop_placement_geometry(
    centers_img_xy: np.ndarray,
    bbox_img_xyxy: np.ndarray,
    roi_sizes_full: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Derive exact top-left, source xywh, and crop-local bbox arrays."""

    centers = np.asarray(centers_img_xy)
    bbox_img = np.asarray(bbox_img_xyxy)
    sizes = np.asarray(roi_sizes_full)
    if centers.dtype != np.dtype(np.float32) or centers.ndim != 2 or centers.shape[1:] != (2,):
        raise ValueError("centers_img_xy must have exact float32 shape (N, 2).")
    if bbox_img.dtype != np.dtype(np.float32) or bbox_img.shape != (centers.shape[0], 4):
        raise ValueError("bbox_img_xyxy must have exact float32 shape (N, 4).")
    if sizes.dtype != np.dtype(np.int32) or sizes.shape != centers.shape:
        raise ValueError("roi_sizes_full must have exact int32 shape (N, 2).")
    if not np.isfinite(centers).all() or not np.isfinite(bbox_img).all():
        raise ValueError("Crop placement inputs must be finite.")
    if np.any(sizes <= 0):
        raise ValueError("Every crop width and height must be positive.")

    rounded = np.rint(centers).astype(np.int64)
    sizes_i64 = sizes.astype(np.int64, copy=False)
    top_left_i64 = rounded - np.floor_divide(sizes_i64, 2)
    int32 = np.iinfo(np.int32)
    if np.any(top_left_i64 < int32.min) or np.any(top_left_i64 > int32.max):
        raise ValueError("Derived crop top-left exceeds the int32 coordinate domain.")
    top_left = top_left_i64.astype(np.int32)
    source_crop = np.column_stack((top_left_i64, sizes_i64)).astype(
        np.float32,
        copy=False,
    )
    offsets = np.column_stack(
        (
            source_crop[:, 0],
            source_crop[:, 1],
            source_crop[:, 0],
            source_crop[:, 1],
        )
    )
    bbox_roi = np.asarray(bbox_img - offsets, dtype=np.float32)
    return (
        np.array(top_left, copy=True, order="C"),
        np.array(source_crop, copy=True, order="C"),
        np.array(bbox_roi, copy=True, order="C"),
    )


@dataclass(frozen=True)
class CropGeometrySchema:
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
            raise ValueError("Crop geometry binding paths must be unique.")
        for binding in self.bindings:
            if not binding.required:
                raise ValueError("Every crop geometry binding must be required.")
            self.contracts.resolve(binding.contract_id, binding.contract_version)

    @property
    def binding_paths(self) -> tuple[str, ...]:
        return tuple(binding.path for binding in self.bindings)

    def validate(
        self,
        arrays: Mapping[str, Any],
        *,
        dimensions: CropDimensions,
        policy: CropGeometryPolicy,
    ) -> tuple[CropSchemaIssue, ...]:
        issues: list[CropSchemaIssue] = []
        invalid_paths: set[str] = set()
        expected = set(self.binding_paths)

        for path in sorted(set(arrays) - expected):
            code = (
                "forbidden_pixel_or_compatibility_array"
                if path in FORBIDDEN_CROP_GEOMETRY_ARRAY_PATHS
                else "unexpected_array"
            )
            issues.append(
                _issue(
                    code,
                    path,
                    "The exact geometry-only crop schema does not declare this array.",
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
                        "Required geometry-only crop array is absent.",
                    )
                )
                continue
            contract = self.contracts.resolve(
                binding.contract_id,
                binding.contract_version,
            )
            try:
                errors = contract.validate_observation(
                    arrays[path],
                    dimensions=dimensions.contract_dimensions,
                )
            except Exception as exc:
                errors = (f"array metadata is unreadable: {exc}",)
            if errors:
                invalid_paths.add(path)
                issues.extend(
                    _issue("array_contract_violation", path, error)
                    for error in errors
                )

        values: dict[str, np.ndarray] = {}
        for path in self.binding_paths:
            if path in invalid_paths:
                continue
            try:
                values[path] = _materialize(arrays[path])
            except Exception as exc:
                issues.append(
                    _issue(
                        "array_read_failure",
                        path,
                        f"Array values could not be materialized: {exc}",
                    )
                )

        frames = values.get("frame_indices")
        frames_valid = False
        if frames is not None:
            frames_valid = bool(
                np.all(frames >= 0) and np.all(frames < dimensions.n_frames)
            )
            if not frames_valid:
                issues.append(
                    _issue(
                        "frame_index_out_of_bounds",
                        "frame_indices",
                        "Every crop row frame must be in [0, n_frames).",
                    )
                )
            if frames.size > 1 and np.any(np.diff(frames) < 0):
                frames_valid = False
                issues.append(
                    _issue(
                        "frame_indices_not_sorted",
                        "frame_indices",
                        "Crop rows must be contiguous in nondecreasing frame order.",
                    )
                )

        source_frames = values.get("source_acquisition_frame_index")
        if frames is not None and source_frames is not None:
            if not np.array_equal(frames, source_frames):
                issues.append(
                    _issue(
                        "source_frame_identity_mismatch",
                        "source_acquisition_frame_index",
                        "Full-acquisition crop v1 requires exact frame identity.",
                    )
                )

        keys = values.get("instance_key")
        if keys is not None and np.unique(keys).shape[0] != keys.shape[0]:
            issues.append(
                _issue(
                    "duplicate_instance_key",
                    "instance_key",
                    "Every crop observation must preserve one unique instance_key.",
                )
            )

        refined_ids = values.get("source_refined_row_ids")
        if refined_ids is not None:
            if np.any(refined_ids < 0):
                issues.append(
                    _issue(
                        "invalid_source_refined_row_id",
                        "source_refined_row_ids",
                        "Every source refined row identity must be nonnegative.",
                    )
                )
            if np.unique(refined_ids).shape[0] != refined_ids.shape[0]:
                issues.append(
                    _issue(
                        "duplicate_source_refined_row_id",
                        "source_refined_row_ids",
                        "The complete refined source rowset cannot reuse a row identity.",
                    )
                )

        bbox_norm = values.get("bbox_norm_coords")
        bbox_norm_valid = False
        if bbox_norm is not None:
            half = np.float32(0.5)
            x_min = bbox_norm[:, 0] - bbox_norm[:, 2] * half
            y_min = bbox_norm[:, 1] - bbox_norm[:, 3] * half
            x_max = bbox_norm[:, 0] + bbox_norm[:, 2] * half
            y_max = bbox_norm[:, 1] + bbox_norm[:, 3] * half
            bbox_norm_valid = bool(
                np.isfinite(bbox_norm).all()
                and np.all(bbox_norm[:, 2:] > 0)
                and np.all(x_min >= 0)
                and np.all(y_min >= 0)
                and np.all(x_max <= np.float32(1.0))
                and np.all(y_max <= np.float32(1.0))
            )
            if not bbox_norm_valid:
                issues.append(
                    _issue(
                        "invalid_bbox_norm_coords",
                        "bbox_norm_coords",
                        "Normalized source boxes must be finite, positive, and contained.",
                    )
                )

        bbox_img = values.get("bbox_img_xyxy")
        centers = values.get("centers_img_xy")
        if bbox_norm_valid and bbox_img is not None and centers is not None:
            expected_bbox, expected_centers = derive_canonical_detection_geometry(
                bbox_norm,
                source_width=dimensions.source_width,
                source_height=dimensions.source_height,
            )
            if not np.array_equal(bbox_img, expected_bbox):
                issues.append(
                    _issue(
                        "bbox_img_projection_mismatch",
                        "bbox_img_xyxy",
                        "Pixel boxes must exactly project bbox_norm_coords as float32.",
                    )
                )
            if not np.array_equal(centers, expected_centers):
                issues.append(
                    _issue(
                        "center_projection_mismatch",
                        "centers_img_xy",
                        "Pixel centers must exactly project bbox_norm_coords as float32.",
                    )
                )

        offsets = values.get("frame_row_offsets")
        if offsets is not None:
            if int(offsets[0]) != 0:
                issues.append(
                    _issue(
                        "offset_start_mismatch",
                        "frame_row_offsets",
                        "frame_row_offsets must start at zero.",
                    )
                )
            if np.any(np.diff(offsets) < 0):
                issues.append(
                    _issue(
                        "offsets_not_monotonic",
                        "frame_row_offsets",
                        "frame_row_offsets must be nondecreasing.",
                    )
                )
            if int(offsets[-1]) != dimensions.n_instances:
                issues.append(
                    _issue(
                        "offset_end_mismatch",
                        "frame_row_offsets",
                        "The final offset must equal n_instances.",
                    )
                )
        if frames is not None and frames_valid and offsets is not None:
            expected_offsets = derive_frame_row_offsets(
                frames,
                n_frames=dimensions.n_frames,
            )
            if not np.array_equal(offsets, expected_offsets):
                issues.append(
                    _issue(
                        "frame_row_offsets_mismatch",
                        "frame_row_offsets",
                        "Offsets must exactly index the contiguous crop rows.",
                    )
                )

        sizes = values.get("roi_sizes_full")
        sizes_valid = False
        if sizes is not None:
            sizes_valid = bool(np.all(sizes > 0))
            if not sizes_valid:
                issues.append(
                    _issue(
                        "invalid_roi_size",
                        "roi_sizes_full",
                        "Every crop width and height must be positive.",
                    )
                )
            if (
                policy.size_mode is CropSizeMode.FIXED_PER_RUN
                and not np.all(
                    sizes
                    == np.asarray(policy.fixed_size_wh, dtype=np.int32).reshape(1, 2)
                )
            ):
                issues.append(
                    _issue(
                        "fixed_roi_size_mismatch",
                        "roi_sizes_full",
                        "Every row must equal the policy fixed_size_wh.",
                    )
                )

        coordinates = values.get("roi_coordinates_full")
        source_crop = values.get("source_crop_xywh")
        bbox_roi = values.get("bbox_roi_xyxy")
        if (
            centers is not None
            and bbox_img is not None
            and sizes is not None
            and sizes_valid
            and coordinates is not None
            and source_crop is not None
            and bbox_roi is not None
        ):
            try:
                expected_coordinates, expected_crop, expected_bbox_roi = (
                    derive_crop_placement_geometry(centers, bbox_img, sizes)
                )
            except ValueError as exc:
                issues.append(
                    _issue(
                        "crop_geometry_derivation_failure",
                        "roi_coordinates_full",
                        str(exc),
                    )
                )
            else:
                if not np.array_equal(coordinates, expected_coordinates):
                    issues.append(
                        _issue(
                            "crop_origin_mismatch",
                            "roi_coordinates_full",
                            "Top-left must exactly follow the versioned center rule.",
                        )
                    )
                if not np.array_equal(source_crop, expected_crop):
                    issues.append(
                        _issue(
                            "source_crop_projection_mismatch",
                            "source_crop_xywh",
                            "source_crop_xywh must exactly project integer origin and size.",
                        )
                    )
                if not np.array_equal(bbox_roi, expected_bbox_roi):
                    issues.append(
                        _issue(
                            "bbox_roi_projection_mismatch",
                            "bbox_roi_xyxy",
                            "bbox_roi_xyxy must exactly translate bbox_img_xyxy.",
                        )
                    )

        if (
            policy.padding_mode is CropPaddingMode.REQUIRE_FULLY_CONTAINED
            and coordinates is not None
            and sizes is not None
            and sizes_valid
        ):
            ends = coordinates.astype(np.int64) + sizes.astype(np.int64)
            if bool(
                np.any(coordinates < 0)
                or np.any(ends[:, 0] > dimensions.source_width)
                or np.any(ends[:, 1] > dimensions.source_height)
            ):
                issues.append(
                    _issue(
                        "crop_not_fully_contained",
                        "roi_coordinates_full",
                        "The policy forbids extraction outside source-frame bounds.",
                    )
                )

        return tuple(issues)

    def require(
        self,
        arrays: Mapping[str, Any],
        *,
        dimensions: CropDimensions,
        policy: CropGeometryPolicy,
    ) -> None:
        issues = self.validate(arrays, dimensions=dimensions, policy=policy)
        if issues:
            raise CropGeometrySchemaError(issues)

    def as_manifest(
        self,
        *,
        dimensions: CropDimensions,
        policy: CropGeometryPolicy,
    ) -> dict[str, object]:
        return {
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "stage": "crop",
            "layout": CROP_GEOMETRY_LAYOUT,
            "base_path": "crop_runs/<run>",
            "artifact_profile": "geometry_only_analysis",
            "source_profile": "refined_detection_v1_full_acquisition_complete_rowset",
            "dimensions": dimensions.as_manifest(),
            "crop_policy": policy.as_manifest(),
            "bindings": [binding.as_manifest() for binding in self.bindings],
            "forbidden_arrays": list(FORBIDDEN_CROP_GEOMETRY_ARRAY_PATHS),
            "array_contracts": self.contracts.as_manifest(),
            "invariants": {
                "row_order": "frame_indices_nondecreasing",
                "within_frame_order": "exact_bound_refined_snapshot_order",
                "row_identity": "instance_key",
                "source_row_identity": "source_refined_row_ids",
                "frame_lookup": "frame_row_offsets_csr",
                "instances_per_frame": "zero_one_or_many",
                "crop_size": policy.size_mode.value,
                "geometry_authority": "bound_refined_detection_plus_crop_policy",
                "pixel_payload": "absent",
                "source_pixel_padding": policy.padding_mode.value,
                "instance_key_semantics": "observation_identity_not_subject_identity",
                "nullability": "forbidden",
                "physical_fill_semantics": "initialization_only_not_missing",
            },
        }


CROP_GEOMETRY_SCHEMA_V1 = CropGeometrySchema(
    schema_id=CROP_GEOMETRY_SCHEMA_ID,
    schema_version=CROP_GEOMETRY_SCHEMA_VERSION,
    bindings=CROP_GEOMETRY_BINDINGS,
    contracts=CROP_ARRAY_CONTRACTS,
)


__all__ = [
    "CROP_GEOMETRY_BINDINGS",
    "CROP_GEOMETRY_LAYOUT",
    "CROP_GEOMETRY_POLICY_SCHEMA_ID",
    "CROP_GEOMETRY_POLICY_SCHEMA_VERSION",
    "CROP_GEOMETRY_SCHEMA_ID",
    "CROP_GEOMETRY_SCHEMA_V1",
    "CROP_GEOMETRY_SCHEMA_VERSION",
    "FORBIDDEN_CROP_GEOMETRY_ARRAY_PATHS",
    "CropDimensions",
    "CropGeometryPolicy",
    "CropGeometrySchema",
    "CropGeometrySchemaError",
    "CropPaddingMode",
    "CropSchemaIssue",
    "CropSizeMode",
    "crop_geometry_policy_from_manifest",
    "derive_crop_placement_geometry",
    "derive_frame_row_offsets",
]
