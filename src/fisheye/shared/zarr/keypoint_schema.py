"""Exact logical schemas for canonical and refined keypoint snapshots.

The module is independent of Zarr I/O and physical storage planning.  It binds
archive-relative array paths to exact logical contracts and validates the
row/coordinate/QC invariants that writers and consumers share.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import re
from typing import Any, Mapping

import numpy as np

from fisheye.shared.zarr.array_contracts import (
    FRAME_ROW_OFFSETS_V1,
    KEYPOINT_ARRAY_CONTRACTS,
    KEYPOINT_CONFIDENCES_V1,
    KEYPOINT_FRAME_INDICES_V1,
    KEYPOINT_INSTANCE_KEY_V1,
    KEYPOINT_POSE_BBOX_XYXY_IMG_V1,
    KEYPOINT_POSE_BBOX_XYXY_ROI_V1,
    KEYPOINT_POSE_CONFIDENCE_V1,
    KEYPOINT_POSE_SUCCESS_V1,
    KEYPOINT_ROW_SIGNATURE_V1,
    KEYPOINT_SOURCE_ACQUISITION_FRAME_INDEX_V1,
    KEYPOINT_SOURCE_CROP_ROW_IDS_V1,
    KEYPOINT_SOURCE_CROP_ROW_SIGNATURE_V1,
    KEYPOINT_VALID_V1,
    KEYPOINTS_IMG_V2,
    KEYPOINTS_ROI_V2,
    REFINED_KEYPOINT_ARRAY_CONTRACTS,
    REFINED_KEYPOINT_CONFIDENCE_VALID_V1,
    REFINED_KEYPOINT_EDIT_FLAGS_V1,
    REFINED_KEYPOINT_FLIP_CORRECTED_V1,
    REFINED_KEYPOINT_GEOMETRY_VALID_V1,
    REFINED_KEYPOINT_REASON_CODES_V1,
    REFINED_KEYPOINT_REFINED_SUCCESS_V1,
    REFINED_KEYPOINT_REVIEW_STATE_CODES_V1,
    REFINED_KEYPOINT_SOURCE_SUCCESS_V1,
    REFINED_KEYPOINT_USABLE_V1,
    ArrayContract,
    ArrayContractBinding,
    ArrayContractCatalog,
)

KEYPOINT_SCHEMA_ID = "palette.stage.keypoint_observations"
KEYPOINT_SCHEMA_VERSION = 2
REFINED_KEYPOINT_SCHEMA_ID = "palette.stage.refined_keypoints"
REFINED_KEYPOINT_SCHEMA_VERSION = 2
KEYPOINT_LAYOUT = "sparse_observations_with_frame_row_offsets_v2"
REFINED_KEYPOINT_LAYOUT = "immutable_reviewed_snapshot_v2"
KEYPOINT_ROW_SIGNATURE_DOMAIN = b"palette.keypoint_row_signature.v1\0"

_CODE_LABEL = re.compile(r"^[a-z][a-z0-9_]*$")


_SHARED_CONTRACTS: tuple[tuple[str, ArrayContract], ...] = (
    ("instance_key", KEYPOINT_INSTANCE_KEY_V1),
    ("source_crop_row_ids", KEYPOINT_SOURCE_CROP_ROW_IDS_V1),
    (
        "source_acquisition_frame_index",
        KEYPOINT_SOURCE_ACQUISITION_FRAME_INDEX_V1,
    ),
    ("frame_indices", KEYPOINT_FRAME_INDICES_V1),
    ("frame_row_offsets", FRAME_ROW_OFFSETS_V1),
    ("source_crop_row_signature", KEYPOINT_SOURCE_CROP_ROW_SIGNATURE_V1),
    ("keypoint_row_signature", KEYPOINT_ROW_SIGNATURE_V1),
    ("keypoints_roi", KEYPOINTS_ROI_V2),
    ("keypoints_img", KEYPOINTS_IMG_V2),
    ("keypoint_confidences", KEYPOINT_CONFIDENCES_V1),
    ("keypoint_valid", KEYPOINT_VALID_V1),
    ("pose_confidence", KEYPOINT_POSE_CONFIDENCE_V1),
    ("pose_bbox_xyxy_roi", KEYPOINT_POSE_BBOX_XYXY_ROI_V1),
    ("pose_bbox_xyxy_img", KEYPOINT_POSE_BBOX_XYXY_IMG_V1),
)

_RAW_CONTRACTS = (*_SHARED_CONTRACTS, ("pose_success", KEYPOINT_POSE_SUCCESS_V1))

_REFINED_CONTRACTS = (
    *_SHARED_CONTRACTS,
    ("source_success", REFINED_KEYPOINT_SOURCE_SUCCESS_V1),
    ("refined_success", REFINED_KEYPOINT_REFINED_SUCCESS_V1),
    ("keypoint_edit_flags", REFINED_KEYPOINT_EDIT_FLAGS_V1),
    ("flip_corrected", REFINED_KEYPOINT_FLIP_CORRECTED_V1),
    ("confidence_valid", REFINED_KEYPOINT_CONFIDENCE_VALID_V1),
    ("geometry_valid", REFINED_KEYPOINT_GEOMETRY_VALID_V1),
    ("usable_keypoints", REFINED_KEYPOINT_USABLE_V1),
    ("review_state_codes", REFINED_KEYPOINT_REVIEW_STATE_CODES_V1),
    ("reason_codes", REFINED_KEYPOINT_REASON_CODES_V1),
)

KEYPOINT_BINDINGS = tuple(
    ArrayContractBinding(
        path=name,
        contract_id=contract.schema_id,
        contract_version=contract.schema_version,
        required=True,
    )
    for name, contract in _RAW_CONTRACTS
)

REFINED_KEYPOINT_BINDINGS = tuple(
    ArrayContractBinding(
        path=name,
        contract_id=contract.schema_id,
        contract_version=contract.schema_version,
        required=True,
    )
    for name, contract in _REFINED_CONTRACTS
)

FORBIDDEN_KEYPOINT_V2_ARRAYS = (
    "confidence",
    "detection_indices",
    "detection_success",
    "frame_counts",
    "heading",
    "heading_delta_next_deg",
    "heading_delta_prev_deg",
    "heading_finite",
    "heading_temporal_outlier",
    "heading_usable",
    "keypoints_norm",
    "n_keypoints",
    "n_rois",
    "quality_labels",
    "triangle_angles",
    "triangle_angles_raw",
    "triangle_area",
)


@dataclass(frozen=True)
class KeypointDimensions:
    """Concrete dimensions for one recording-level keypoint snapshot."""

    n_frames: int
    n_instances: int
    n_keypoints: int
    source_width: int
    source_height: int

    def __post_init__(self) -> None:
        for name in (
            "n_frames",
            "n_instances",
            "n_keypoints",
            "source_width",
            "source_height",
        ):
            if type(getattr(self, name)) is not int:
                raise TypeError(f"{name} must be an exact integer.")
        if self.n_frames <= 0:
            raise ValueError("n_frames must be positive.")
        if self.n_instances < 0:
            raise ValueError("n_instances cannot be negative.")
        if self.n_keypoints <= 0:
            raise ValueError("n_keypoints must be positive.")
        if self.source_width <= 0 or self.source_height <= 0:
            raise ValueError("Source-camera width and height must be positive.")

    @property
    def contract_dimensions(self) -> dict[str, int]:
        return {
            "n_frames": self.n_frames,
            "n_frame_boundaries": self.n_frames + 1,
            "n_instances": self.n_instances,
            "n_keypoints": self.n_keypoints,
        }

    def as_manifest(self) -> dict[str, int]:
        return {
            **self.contract_dimensions,
            "source_width": self.source_width,
            "source_height": self.source_height,
        }


@dataclass(frozen=True)
class KeypointSchemaIssue:
    code: str
    path: str
    message: str

    def as_manifest(self) -> dict[str, str]:
        return {"code": self.code, "path": self.path, "message": self.message}


class KeypointSchemaError(ValueError):
    def __init__(self, issues: tuple[KeypointSchemaIssue, ...]) -> None:
        self.issues = issues
        detail = "; ".join(
            f"{issue.code} at {issue.path}: {issue.message}" for issue in issues
        )
        super().__init__(
            f"Keypoint schema validation failed with {len(issues)} issue(s): {detail}"
        )


def _issue(code: str, path: str, message: str) -> KeypointSchemaIssue:
    return KeypointSchemaIssue(code=code, path=path, message=message)


def _materialize(array: Any) -> np.ndarray:
    if isinstance(array, np.ndarray):
        return array
    try:
        return np.asarray(array[...])
    except (IndexError, KeyError, TypeError):
        return np.asarray(array)


def derive_frame_row_offsets(frame_indices: np.ndarray, *, n_frames: int) -> np.ndarray:
    frames = np.asarray(frame_indices)
    if frames.dtype != np.dtype(np.int64) or frames.ndim != 1:
        raise ValueError("frame_indices must be one exact int64 vector.")
    if type(n_frames) is not int or n_frames <= 0:
        raise ValueError("n_frames must be a positive exact integer.")
    if np.any(frames < 0) or np.any(frames >= n_frames):
        raise ValueError("frame_indices contains an out-of-bounds value.")
    if frames.size > 1 and np.any(np.diff(frames) < 0):
        raise ValueError("frame_indices must be nondecreasing.")
    counts = np.bincount(frames, minlength=n_frames)
    offsets = np.zeros(n_frames + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts, dtype=np.int64)
    return offsets


def derive_keypoint_row_signatures(
    *,
    instance_key: np.ndarray,
    source_crop_row_signature: np.ndarray,
    keypoints_roi: np.ndarray,
    keypoint_valid: np.ndarray,
    skeleton_digest: str,
) -> np.ndarray:
    """Derive canonical row signatures for incremental compatibility."""

    keys = np.asarray(instance_key)
    crop_signatures = np.asarray(source_crop_row_signature)
    points = np.asarray(keypoints_roi)
    valid = np.asarray(keypoint_valid)
    if keys.dtype != np.dtype(np.uint64) or keys.ndim != 1:
        raise ValueError("instance_key must be one exact uint64 vector.")
    n_rows = int(keys.shape[0])
    if crop_signatures.dtype != np.dtype(np.uint8) or crop_signatures.shape != (
        n_rows,
        32,
    ):
        raise ValueError("source_crop_row_signature must have uint8 shape (N, 32).")
    if (
        points.dtype != np.dtype(np.float32)
        or points.ndim != 3
        or points.shape[0] != n_rows
        or points.shape[2] != 2
    ):
        raise ValueError("keypoints_roi must have exact float32 shape (N, K, 2).")
    if valid.dtype != np.dtype(bool) or valid.shape != points.shape[:2]:
        raise ValueError("keypoint_valid must have exact bool shape (N, K).")
    try:
        skeleton_bytes = bytes.fromhex(str(skeleton_digest))
    except ValueError as exc:
        raise ValueError(
            "skeleton_digest must be lowercase hexadecimal SHA-256."
        ) from exc
    if (
        len(skeleton_bytes) != 32
        or str(skeleton_digest) != str(skeleton_digest).lower()
    ):
        raise ValueError("skeleton_digest must be lowercase hexadecimal SHA-256.")

    canonical_points = np.array(points, dtype="<f4", order="C", copy=True)
    canonical_points[~valid] = np.float32(np.nan)
    result = np.empty((n_rows, 32), dtype=np.uint8)
    for row in range(n_rows):
        digest = sha256()
        digest.update(KEYPOINT_ROW_SIGNATURE_DOMAIN)
        digest.update(skeleton_bytes)
        digest.update(np.asarray(keys[row], dtype="<u8").tobytes())
        digest.update(crop_signatures[row].tobytes(order="C"))
        digest.update(canonical_points[row].tobytes(order="C"))
        digest.update(valid[row].astype(np.uint8, copy=False).tobytes(order="C"))
        result[row] = np.frombuffer(digest.digest(), dtype=np.uint8)
    return result


def _validate_code_registry(
    *,
    values: np.ndarray,
    registry: Mapping[int, str] | None,
    path: str,
    max_code: int,
) -> tuple[KeypointSchemaIssue, ...]:
    issues: list[KeypointSchemaIssue] = []
    if registry is None:
        return (
            _issue(
                "missing_code_registry",
                path,
                "A complete controlled code registry is required.",
            ),
        )
    normalized: dict[int, str] = {}
    for raw_code, raw_label in registry.items():
        if type(raw_code) is not int or not (0 <= raw_code <= max_code):
            issues.append(
                _issue(
                    "invalid_code_registry",
                    path,
                    f"Registry code {raw_code!r} is outside [0, {max_code}].",
                )
            )
            continue
        label = str(raw_label)
        if not _CODE_LABEL.fullmatch(label):
            issues.append(
                _issue(
                    "invalid_code_registry",
                    path,
                    f"Registry label {label!r} is not lowercase snake_case.",
                )
            )
            continue
        normalized[raw_code] = label
    if 0 not in normalized:
        issues.append(
            _issue(
                "missing_zero_code",
                path,
                "Every refined keypoint code registry must define code zero.",
            )
        )
    unknown = sorted(set(int(value) for value in np.unique(values)) - set(normalized))
    if unknown:
        issues.append(
            _issue(
                "unknown_persisted_code",
                path,
                f"Persisted codes are absent from the registry: {unknown!r}.",
            )
        )
    return tuple(issues)


@dataclass(frozen=True)
class KeypointSnapshotSchema:
    schema_id: str
    schema_version: int
    stage: str
    layout: str
    base_path: str
    bindings: tuple[ArrayContractBinding, ...]
    contracts: ArrayContractCatalog
    refined: bool = False

    def __post_init__(self) -> None:
        if not self.schema_id.strip() or not self.stage.strip():
            raise ValueError("schema_id and stage cannot be empty.")
        if type(self.schema_version) is not int or self.schema_version <= 0:
            raise ValueError("schema_version must be a positive exact integer.")
        paths = [binding.path for binding in self.bindings]
        if len(paths) != len(set(paths)):
            raise ValueError("Keypoint binding paths must be unique.")
        for binding in self.bindings:
            if not binding.required:
                raise ValueError("Every keypoint-v2 binding must be required.")
            self.contracts.resolve(binding.contract_id, binding.contract_version)

    @property
    def binding_paths(self) -> tuple[str, ...]:
        return tuple(binding.path for binding in self.bindings)

    def coordinate_contract_manifest(self) -> dict[str, object]:
        from fisheye.shared.zarr.coordinate_contracts import (
            array_coordinate_catalog_manifest,
        )

        return array_coordinate_catalog_manifest(self.contracts)

    def validate(
        self,
        arrays: Mapping[str, Any],
        *,
        dimensions: KeypointDimensions,
        source_crop_arrays: Mapping[str, Any] | None,
        skeleton_digest: str,
        review_state_map: Mapping[int, str] | None = None,
        reason_code_map: Mapping[int, str] | None = None,
    ) -> tuple[KeypointSchemaIssue, ...]:
        issues: list[KeypointSchemaIssue] = []
        invalid_paths: set[str] = set()
        expected = set(self.binding_paths)

        for path in sorted(set(arrays) - expected):
            code = (
                "forbidden_v1_array"
                if path in FORBIDDEN_KEYPOINT_V2_ARRAYS
                else "unexpected_array"
            )
            issues.append(
                _issue(
                    code,
                    path,
                    "The exact keypoint-v2 schema does not declare this array.",
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
                        "Required keypoint array is absent.",
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
                    _issue("array_contract_violation", path, error) for error in errors
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
                        "array_read_failure", path, f"Array could not be read: {exc}"
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
                        "Every row frame must be in [0, n_frames).",
                    )
                )
            if frames.size > 1 and np.any(np.diff(frames) < 0):
                frames_valid = False
                issues.append(
                    _issue(
                        "frame_indices_not_sorted",
                        "frame_indices",
                        "Rows must be contiguous in nondecreasing frame order.",
                    )
                )

        offsets = values.get("frame_row_offsets")
        if offsets is not None:
            if int(offsets[0]) != 0:
                issues.append(
                    _issue(
                        "offset_start_mismatch",
                        "frame_row_offsets",
                        "Offsets must start at zero.",
                    )
                )
            if np.any(np.diff(offsets) < 0):
                issues.append(
                    _issue(
                        "offsets_not_monotonic",
                        "frame_row_offsets",
                        "Offsets must be nondecreasing.",
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
                        "Offsets must exactly index the contiguous keypoint rows.",
                    )
                )

        keys = values.get("instance_key")
        if keys is not None and np.unique(keys).shape[0] != keys.shape[0]:
            issues.append(
                _issue(
                    "duplicate_instance_key",
                    "instance_key",
                    "instance_key values must be unique.",
                )
            )
        source_rows = values.get("source_crop_row_ids")
        if source_rows is not None:
            if np.any(source_rows < 0):
                issues.append(
                    _issue(
                        "invalid_source_crop_row_id",
                        "source_crop_row_ids",
                        "Source crop row IDs must be nonnegative.",
                    )
                )
            if np.unique(source_rows).shape[0] != source_rows.shape[0]:
                issues.append(
                    _issue(
                        "duplicate_source_crop_row_id",
                        "source_crop_row_ids",
                        "One keypoint snapshot cannot reuse a crop row.",
                    )
                )

        points_roi = values.get("keypoints_roi")
        points_img = values.get("keypoints_img")
        confidence = values.get("keypoint_confidences")
        valid = values.get("keypoint_valid")
        if points_roi is not None:
            finite_x = np.isfinite(points_roi[..., 0])
            finite_y = np.isfinite(points_roi[..., 1])
            if not np.array_equal(finite_x, finite_y):
                issues.append(
                    _issue(
                        "partial_keypoint_coordinate",
                        "keypoints_roi",
                        "Each ROI keypoint must be wholly finite or wholly NaN.",
                    )
                )
        if points_img is not None:
            finite_x = np.isfinite(points_img[..., 0])
            finite_y = np.isfinite(points_img[..., 1])
            if not np.array_equal(finite_x, finite_y):
                issues.append(
                    _issue(
                        "partial_keypoint_coordinate",
                        "keypoints_img",
                        "Each image keypoint must be wholly finite or wholly NaN.",
                    )
                )
        if points_roi is not None and valid is not None:
            expected_valid = np.all(np.isfinite(points_roi), axis=2)
            if not np.array_equal(valid, expected_valid):
                issues.append(
                    _issue(
                        "keypoint_valid_mismatch",
                        "keypoint_valid",
                        "keypoint_valid must exactly mark finite ROI coordinate pairs.",
                    )
                )
        if confidence is not None:
            finite_confidence = np.isfinite(confidence)
            if np.any(
                finite_confidence
                & ((confidence < np.float32(0.0)) | (confidence > np.float32(1.0)))
            ):
                issues.append(
                    _issue(
                        "invalid_keypoint_confidence",
                        "keypoint_confidences",
                        "Finite keypoint confidences must lie in [0, 1].",
                    )
                )
            if valid is not None and np.any(~valid & finite_confidence):
                issues.append(
                    _issue(
                        "invalid_keypoint_has_confidence",
                        "keypoint_confidences",
                        "Invalid landmarks must use NaN confidence.",
                    )
                )

        pose_confidence = values.get("pose_confidence")
        if pose_confidence is not None:
            finite_pose = np.isfinite(pose_confidence)
            if np.any(
                finite_pose
                & (
                    (pose_confidence < np.float32(0.0))
                    | (pose_confidence > np.float32(1.0))
                )
            ):
                issues.append(
                    _issue(
                        "invalid_pose_confidence",
                        "pose_confidence",
                        "Finite pose confidence must lie in [0, 1].",
                    )
                )

        bbox_roi = values.get("pose_bbox_xyxy_roi")
        bbox_img = values.get("pose_bbox_xyxy_img")
        for path, boxes in (
            ("pose_bbox_xyxy_roi", bbox_roi),
            ("pose_bbox_xyxy_img", bbox_img),
        ):
            if boxes is None:
                continue
            finite = np.isfinite(boxes)
            whole = np.all(finite, axis=1) | np.all(~finite, axis=1)
            if not np.all(whole):
                issues.append(
                    _issue(
                        "partial_pose_bbox",
                        path,
                        "Each pose bbox must be wholly finite or wholly NaN.",
                    )
                )
            finite_rows = np.all(finite, axis=1)
            if np.any(
                finite_rows
                & ((boxes[:, 2] <= boxes[:, 0]) | (boxes[:, 3] <= boxes[:, 1]))
            ):
                issues.append(
                    _issue(
                        "invalid_pose_bbox",
                        path,
                        "Finite pose boxes must have positive half-open extent.",
                    )
                )

        pose_success = values.get("pose_success")
        if pose_success is not None:
            finite_pose = (
                np.isfinite(pose_confidence)
                if pose_confidence is not None
                else np.zeros(dimensions.n_instances, dtype=bool)
            )
            finite_bbox = (
                np.all(np.isfinite(bbox_roi), axis=1)
                if bbox_roi is not None
                else np.zeros(dimensions.n_instances, dtype=bool)
            )
            if not np.array_equal(pose_success, finite_pose & finite_bbox):
                issues.append(
                    _issue(
                        "pose_success_mismatch",
                        "pose_success",
                        "Raw pose_success must exactly mark finite pose confidence and bbox.",
                    )
                )
            if valid is not None and np.any(~pose_success & np.any(valid, axis=1)):
                issues.append(
                    _issue(
                        "failed_pose_has_valid_keypoints",
                        "keypoint_valid",
                        "A failed raw pose cannot contain valid landmarks.",
                    )
                )

        if source_crop_arrays is None:
            issues.append(
                _issue(
                    "missing_source_crop_evidence",
                    "source_crop_row_ids",
                    "Exact bound crop arrays are required for lineage and projection validation.",
                )
            )
        elif source_rows is not None and np.all(source_rows >= 0):
            issues.extend(
                self._validate_source_crop_binding(
                    values,
                    source_crop_arrays=source_crop_arrays,
                    source_rows=source_rows,
                )
            )

        row_signature = values.get("keypoint_row_signature")
        crop_signature = values.get("source_crop_row_signature")
        if (
            row_signature is not None
            and crop_signature is not None
            and keys is not None
            and points_roi is not None
            and valid is not None
        ):
            try:
                expected_signatures = derive_keypoint_row_signatures(
                    instance_key=keys,
                    source_crop_row_signature=crop_signature,
                    keypoints_roi=points_roi,
                    keypoint_valid=valid,
                    skeleton_digest=skeleton_digest,
                )
            except ValueError as exc:
                issues.append(
                    _issue(
                        "invalid_skeleton_digest", "keypoint_row_signature", str(exc)
                    )
                )
            else:
                if not np.array_equal(row_signature, expected_signatures):
                    issues.append(
                        _issue(
                            "keypoint_row_signature_mismatch",
                            "keypoint_row_signature",
                            "Row signatures must exactly bind identity, crop, skeleton, coordinates, and validity.",
                        )
                    )

        if self.refined:
            issues.extend(
                self._validate_refined_qc(
                    values,
                    review_state_map=review_state_map,
                    reason_code_map=reason_code_map,
                )
            )
        return tuple(issues)

    def _validate_source_crop_binding(
        self,
        values: Mapping[str, np.ndarray],
        *,
        source_crop_arrays: Mapping[str, Any],
        source_rows: np.ndarray,
    ) -> tuple[KeypointSchemaIssue, ...]:
        issues: list[KeypointSchemaIssue] = []
        required = (
            "instance_key",
            "frame_indices",
            "source_acquisition_frame_index",
            "source_row_signature",
            "roi_coordinates_full",
            "roi_sizes_full",
        )
        missing = [path for path in required if path not in source_crop_arrays]
        if missing:
            return (
                _issue(
                    "incomplete_source_crop_evidence",
                    "source_crop_row_ids",
                    f"Bound crop evidence is missing {missing!r}.",
                ),
            )
        try:
            source = {path: _materialize(source_crop_arrays[path]) for path in required}
        except Exception as exc:
            return (
                _issue(
                    "source_crop_read_failure",
                    "source_crop_row_ids",
                    f"Bound crop evidence could not be read: {exc}",
                ),
            )
        source_count = int(source["instance_key"].shape[0])
        if np.any(source_rows >= source_count):
            return (
                _issue(
                    "source_crop_row_out_of_bounds",
                    "source_crop_row_ids",
                    "A source crop row ID is outside the bound crop table.",
                ),
            )
        selected = source_rows.astype(np.int64, copy=False)
        comparisons = (
            ("instance_key", "instance_key"),
            ("frame_indices", "frame_indices"),
            (
                "source_acquisition_frame_index",
                "source_acquisition_frame_index",
            ),
            ("source_crop_row_signature", "source_row_signature"),
        )
        for target_path, source_path in comparisons:
            target = values.get(target_path)
            if target is not None and not np.array_equal(
                target, source[source_path][selected]
            ):
                issues.append(
                    _issue(
                        "source_crop_binding_mismatch",
                        target_path,
                        f"Values do not equal bound crop rows from {source_path}.",
                    )
                )

        origins = source["roi_coordinates_full"][selected]
        sizes = source["roi_sizes_full"][selected]
        points_roi = values.get("keypoints_roi")
        points_img = values.get("keypoints_img")
        if points_roi is not None and points_img is not None:
            finite = np.all(np.isfinite(points_roi), axis=2)
            if np.any(
                finite
                & (
                    (points_roi[..., 0] < np.float32(0.0))
                    | (points_roi[..., 1] < np.float32(0.0))
                    | (points_roi[..., 0] >= sizes[:, None, 0])
                    | (points_roi[..., 1] >= sizes[:, None, 1])
                )
            ):
                issues.append(
                    _issue(
                        "keypoint_outside_roi",
                        "keypoints_roi",
                        "Finite keypoints must lie in their row-specific ROI extent.",
                    )
                )
            expected_img = points_roi + origins.astype(np.float32)[:, None, :]
            if not np.array_equal(points_img, expected_img, equal_nan=True):
                issues.append(
                    _issue(
                        "keypoints_img_projection_mismatch",
                        "keypoints_img",
                        "Image keypoints must exactly translate ROI points by crop origin.",
                    )
                )

        bbox_roi = values.get("pose_bbox_xyxy_roi")
        bbox_img = values.get("pose_bbox_xyxy_img")
        if bbox_roi is not None and bbox_img is not None:
            finite = np.all(np.isfinite(bbox_roi), axis=1)
            if np.any(
                finite
                & (
                    (bbox_roi[:, 0] < np.float32(0.0))
                    | (bbox_roi[:, 1] < np.float32(0.0))
                    | (bbox_roi[:, 2] > sizes[:, 0])
                    | (bbox_roi[:, 3] > sizes[:, 1])
                )
            ):
                issues.append(
                    _issue(
                        "pose_bbox_outside_roi",
                        "pose_bbox_xyxy_roi",
                        "Finite pose boxes must lie in their row-specific ROI extent.",
                    )
                )
            offsets = np.column_stack((origins, origins)).astype(np.float32, copy=False)
            expected_img = bbox_roi + offsets
            if not np.array_equal(bbox_img, expected_img, equal_nan=True):
                issues.append(
                    _issue(
                        "pose_bbox_img_projection_mismatch",
                        "pose_bbox_xyxy_img",
                        "Image pose boxes must exactly translate ROI boxes by crop origin.",
                    )
                )
        return tuple(issues)

    def _validate_refined_qc(
        self,
        values: Mapping[str, np.ndarray],
        *,
        review_state_map: Mapping[int, str] | None,
        reason_code_map: Mapping[int, str] | None,
    ) -> tuple[KeypointSchemaIssue, ...]:
        issues: list[KeypointSchemaIssue] = []
        usable = values.get("usable_keypoints")
        refined_success = values.get("refined_success")
        confidence_valid = values.get("confidence_valid")
        geometry_valid = values.get("geometry_valid")
        if (
            usable is not None
            and refined_success is not None
            and confidence_valid is not None
            and geometry_valid is not None
        ):
            expected_usable = refined_success & confidence_valid & geometry_valid
            if not np.array_equal(usable, expected_usable):
                issues.append(
                    _issue(
                        "usable_keypoints_policy_mismatch",
                        "usable_keypoints",
                        "Usable rows must exactly equal the refined-success, confidence, and geometry gates.",
                    )
                )
        valid = values.get("keypoint_valid")
        if refined_success is not None and valid is not None:
            any_valid = np.any(valid, axis=1)
            if not np.array_equal(refined_success, any_valid):
                issues.append(
                    _issue(
                        "refined_success_mismatch",
                        "refined_success",
                        "refined_success must exactly mark rows with at least one valid landmark.",
                    )
                )
        flips = values.get("flip_corrected")
        edits = values.get("keypoint_edit_flags")
        if (
            flips is not None
            and edits is not None
            and np.any(flips & ~np.any(edits, axis=1))
        ):
            issues.append(
                _issue(
                    "flip_without_landmark_edit",
                    "flip_corrected",
                    "A flip correction must change at least one landmark identity.",
                )
            )
        review_codes = values.get("review_state_codes")
        if review_codes is not None:
            issues.extend(
                _validate_code_registry(
                    values=review_codes,
                    registry=review_state_map,
                    path="review_state_codes",
                    max_code=int(np.iinfo(np.uint8).max),
                )
            )
        reason_codes = values.get("reason_codes")
        if reason_codes is not None:
            issues.extend(
                _validate_code_registry(
                    values=reason_codes,
                    registry=reason_code_map,
                    path="reason_codes",
                    max_code=int(np.iinfo(np.uint16).max),
                )
            )
        return tuple(issues)

    def require(
        self,
        arrays: Mapping[str, Any],
        *,
        dimensions: KeypointDimensions,
        source_crop_arrays: Mapping[str, Any] | None,
        skeleton_digest: str,
        review_state_map: Mapping[int, str] | None = None,
        reason_code_map: Mapping[int, str] | None = None,
    ) -> None:
        issues = self.validate(
            arrays,
            dimensions=dimensions,
            source_crop_arrays=source_crop_arrays,
            skeleton_digest=skeleton_digest,
            review_state_map=review_state_map,
            reason_code_map=reason_code_map,
        )
        if issues:
            raise KeypointSchemaError(issues)

    def as_manifest(self, *, dimensions: KeypointDimensions) -> dict[str, object]:
        return {
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "stage": self.stage,
            "layout": self.layout,
            "base_path": self.base_path,
            "dimensions": dimensions.as_manifest(),
            "bindings": [binding.as_manifest() for binding in self.bindings],
            "forbidden_v1_arrays": list(FORBIDDEN_KEYPOINT_V2_ARRAYS),
            "array_contracts": self.contracts.as_manifest(),
            "invariants": {
                "row_order": "frame_indices_nondecreasing",
                "row_identity": "instance_key",
                "landmark_identity": "digest_bound_ordered_skeleton_label",
                "source_row_identity": "source_crop_row_ids",
                "frame_lookup": "frame_row_offsets_csr",
                "instances_per_frame": "zero_one_or_many",
                "coordinate_authority": "keypoints_roi",
                "image_projection": "exact_float32_crop_origin_translation",
                "invalid_landmark": "nan_xy_keypoint_valid_false",
                "heading_payload": "forbidden_use_bound_body_frame_run",
                "instance_key_semantics": "observation_identity_not_subject_identity",
            },
        }


KEYPOINT_SCHEMA_V2 = KeypointSnapshotSchema(
    schema_id=KEYPOINT_SCHEMA_ID,
    schema_version=KEYPOINT_SCHEMA_VERSION,
    stage="keypoints",
    layout=KEYPOINT_LAYOUT,
    base_path="keypoints_runs/<run>",
    bindings=KEYPOINT_BINDINGS,
    contracts=KEYPOINT_ARRAY_CONTRACTS,
    refined=False,
)

REFINED_KEYPOINT_SCHEMA_V2 = KeypointSnapshotSchema(
    schema_id=REFINED_KEYPOINT_SCHEMA_ID,
    schema_version=REFINED_KEYPOINT_SCHEMA_VERSION,
    stage="refined_keypoints",
    layout=REFINED_KEYPOINT_LAYOUT,
    base_path="refined_keypoints_runs/<run>",
    bindings=REFINED_KEYPOINT_BINDINGS,
    contracts=REFINED_KEYPOINT_ARRAY_CONTRACTS,
    refined=True,
)


__all__ = [
    "FORBIDDEN_KEYPOINT_V2_ARRAYS",
    "KEYPOINT_BINDINGS",
    "KEYPOINT_LAYOUT",
    "KEYPOINT_ROW_SIGNATURE_DOMAIN",
    "KEYPOINT_SCHEMA_ID",
    "KEYPOINT_SCHEMA_V2",
    "KEYPOINT_SCHEMA_VERSION",
    "REFINED_KEYPOINT_BINDINGS",
    "REFINED_KEYPOINT_LAYOUT",
    "REFINED_KEYPOINT_SCHEMA_ID",
    "REFINED_KEYPOINT_SCHEMA_V2",
    "REFINED_KEYPOINT_SCHEMA_VERSION",
    "KeypointDimensions",
    "KeypointSchemaError",
    "KeypointSchemaIssue",
    "KeypointSnapshotSchema",
    "derive_frame_row_offsets",
    "derive_keypoint_row_signatures",
]
