from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union

import numpy as np
import zarr

ShapeDim = Union[str, int]


@dataclass(frozen=True)
class ArraySpec:
    name: str
    dtype: str
    shape_template: Tuple[ShapeDim, ...]
    required: bool = True
    description: str = ""


@dataclass(frozen=True)
class StageSpec:
    stage_name: str
    zarr_group: str
    specs: Tuple[ArraySpec, ...]
    subgroups: Dict[str, Tuple[ArraySpec, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    errors: List[str]
    warnings: List[str]


def _shape_to_text(shape_template: Tuple[ShapeDim, ...]) -> str:
    if not shape_template:
        return "()"
    parts = ", ".join(str(dim) for dim in shape_template)
    if len(shape_template) == 1:
        parts += ","
    return f"({parts})"


def describe_array(spec: ArraySpec) -> str:
    return f"{_shape_to_text(spec.shape_template)} {spec.dtype}"


def array_specs_by_name(
    stage_spec: StageSpec,
    *,
    subgroup: str | None = None,
) -> Dict[str, ArraySpec]:
    specs = stage_spec.subgroups[subgroup] if subgroup else stage_spec.specs
    return {item.name: item for item in specs}


def required_array_specs(
    stage_spec: StageSpec,
    *,
    subgroup: str | None = None,
) -> Tuple[ArraySpec, ...]:
    specs = stage_spec.subgroups[subgroup] if subgroup else stage_spec.specs
    return tuple(item for item in specs if item.required)


def required_array_names(
    stage_spec: StageSpec,
    *,
    subgroup: str | None = None,
) -> Tuple[str, ...]:
    return tuple(item.name for item in required_array_specs(stage_spec, subgroup=subgroup))


def _infer_expected_kinds(dtype: str) -> set[str]:
    expected: set[str] = set()
    for part in dtype.lower().split("/"):
        token = part.strip()
        if token.startswith("uint"):
            expected.add("uint")
        elif token.startswith("int"):
            expected.add("int")
        elif token.startswith("float"):
            expected.add("float")
        elif token == "bool":
            expected.add("bool")
        elif token == "string":
            expected.add("string")
    return expected


def _kind_from_numpy(dtype: np.dtype) -> str:
    kind = dtype.kind
    if kind == "u":
        return "uint"
    if kind == "i":
        return "int"
    if kind == "f":
        return "float"
    if kind == "b":
        return "bool"
    if kind in {"U", "S", "O"}:
        return "string"
    return kind


def _kind_from_any_dtype(dtype_obj: object) -> str:
    try:
        return _kind_from_numpy(np.dtype(dtype_obj))
    except TypeError:
        dtype_name = type(dtype_obj).__name__.lower()
        dtype_text = str(dtype_obj).lower()
        if "variablelengthutf8" in dtype_name or "variablelengthutf8" in dtype_text:
            return "string"
        kind_attr = getattr(dtype_obj, "kind", None)
        if isinstance(kind_attr, str):
            if kind_attr == "u":
                return "uint"
            if kind_attr == "i":
                return "int"
            if kind_attr == "f":
                return "float"
            if kind_attr == "b":
                return "bool"
            if kind_attr in {"U", "S", "O"}:
                return "string"
        if "string" in dtype_name or "utf" in dtype_name:
            return "string"
        return dtype_name


def _validate_specs(
    group: zarr.Group,
    specs: Tuple[ArraySpec, ...],
    *,
    group_label: str,
    errors: List[str],
    warnings: List[str],
) -> None:
    leading_dims: Dict[str, int] = {}

    for spec in specs:
        if spec.name not in group:
            message = f"{group_label}: missing {'required' if spec.required else 'optional'} array '{spec.name}'"
            if spec.required:
                errors.append(message)
            else:
                warnings.append(message)
            continue

        array_obj = group[spec.name]
        if not hasattr(array_obj, "shape") or not hasattr(array_obj, "dtype"):
            errors.append(f"{group_label}/{spec.name}: expected array, found non-array object")
            continue

        shape = tuple(int(dim) for dim in array_obj.shape)
        raw_dtype = array_obj.dtype

        expected_kinds = _infer_expected_kinds(spec.dtype)
        if expected_kinds:
            actual_kind = _kind_from_any_dtype(raw_dtype)
            if actual_kind not in expected_kinds:
                expected_text = "/".join(sorted(expected_kinds))
                errors.append(
                    f"{group_label}/{spec.name}: dtype kind mismatch (expected {expected_text} from '{spec.dtype}', got {actual_kind} [{raw_dtype}])"
                )

        template = spec.shape_template
        if len(shape) != len(template):
            errors.append(
                f"{group_label}/{spec.name}: rank mismatch (expected ndim={len(template)} from {_shape_to_text(template)}, got ndim={len(shape)} with shape={shape})"
            )
            continue

        for idx, expected_dim in enumerate(template):
            if isinstance(expected_dim, int) and shape[idx] != expected_dim:
                errors.append(
                    f"{group_label}/{spec.name}: dimension {idx} mismatch (expected {expected_dim}, got {shape[idx]})"
                )

        if shape and isinstance(template[0], str):
            leading_key = template[0]
            current = shape[0]
            previous = leading_dims.get(leading_key)
            if previous is None:
                leading_dims[leading_key] = current
            elif previous != current:
                errors.append(
                    f"{group_label}: leading dimension mismatch for '{leading_key}' ({spec.name} has {current}, expected {previous})"
                )


def validate_run(group: zarr.Group, spec: StageSpec) -> ValidationResult:
    errors: List[str] = []
    warnings: List[str] = []

    if spec.specs:
        _validate_specs(
            group,
            spec.specs,
            group_label=spec.stage_name,
            errors=errors,
            warnings=warnings,
        )

    if spec.subgroups:
        for subgroup_name, subgroup_specs in spec.subgroups.items():
            if subgroup_name not in group:
                errors.append(f"{spec.stage_name}: missing required subgroup '{subgroup_name}'")
                continue
            subgroup = group[subgroup_name]
            if not isinstance(subgroup, zarr.Group):
                errors.append(f"{spec.stage_name}: '{subgroup_name}' exists but is not a group")
                continue

            _validate_specs(
                subgroup,
                subgroup_specs,
                group_label=f"{spec.stage_name}/{subgroup_name}",
                errors=errors,
                warnings=warnings,
            )

    return ValidationResult(valid=not errors, errors=errors, warnings=warnings)


RAW_VIDEO_SPEC = StageSpec(
    stage_name="raw_video",
    zarr_group="raw_video/",
    specs=(
        ArraySpec("images_full", "uint8", ("n_frames", "H", "W"), required=False),
        ArraySpec(
            "images_ds",
            "uint8",
            ("n_frames", "H_ds", "W_ds"),
            required=False,
            description="Optional downsampled import; at least one of images_full/images_ds should exist.",
        ),
        ArraySpec("images_ds_rgb", "uint8", ("n_frames", "H_ds", "W_ds", 3), required=False),
        ArraySpec(
            "original_frame_indices",
            "int32",
            ("n_import_frames",),
            required=False,
            description="Present when frame_step > 1.",
        ),
    ),
)

BACKGROUND_SPEC = StageSpec(
    stage_name="background",
    zarr_group="background_runs/<run>/",
    specs=(
        ArraySpec("background_full", "uint8", ("H", "W"), required=False),
        ArraySpec("background_ds", "uint8", ("H_ds", "W_ds"), required=False),
        ArraySpec("frame_indices", "int32", ("n_samples",)),
    ),
)

DETECT_SPEC = StageSpec(
    stage_name="detect",
    zarr_group="detect_runs/<run>/",
    specs=(
        ArraySpec("frame_indices", "int32", ("n_detections",)),
        ArraySpec("bbox_norm_coords", "float32", ("n_detections", 4)),
        ArraySpec("scores", "float32", ("n_detections",)),
        ArraySpec("class_ids", "int32", ("n_detections",)),
        ArraySpec("frame_counts", "int32", ("n_frames",)),
        ArraySpec(
            "n_detections",
            "int32",
            ("n_frames",),
            description="Legacy alias of frame_counts.",
        ),
        ArraySpec(
            "centers_px",
            "float32",
            ("n_detections", 2),
            required=False,
            description="Written by blob method.",
        ),
    ),
)

DETECT_QUALITY_SPEC = StageSpec(
    stage_name="detect_quality",
    zarr_group="detect_runs/<run>/quality_reports/<qrun>/",
    specs=(
        ArraySpec("quality_flags", "int8", ("n_frames",)),
        ArraySpec("detection_quality_labels", "int8", ("n_detections",)),
    ),
)

_REFINED_DETECT_ARRAYS: Tuple[ArraySpec, ...] = (
    ArraySpec("bbox_norm_coords", "float32", ("n_refined", 4)),
    ArraySpec("scores", "float32", ("n_refined",)),
    ArraySpec("frame_indices", "int32", ("n_refined",)),
    ArraySpec("class_ids", "int32", ("n_refined",)),
    ArraySpec("frame_counts", "int32", ("n_frames",)),
    ArraySpec(
        "n_detections",
        "int32",
        ("n_frames",),
        description="Legacy alias of frame_counts.",
    ),
    ArraySpec(
        "detection_source",
        "int8",
        ("n_refined",),
        description="0=real, 1=interpolated.",
    ),
    ArraySpec("reason_bytes", "uint8", ("n_refined", "width")),
    ArraySpec("reason", "string", ("n_refined",)),
    ArraySpec(
        "frame_mapping",
        "int32",
        ("n_refined",),
        description="Legacy alias of frame_indices.",
    ),
)

_REFINED_DETECT_SOURCE_DETECTIONS_ARRAYS: Tuple[ArraySpec, ...] = (
    ArraySpec("source_detect_row_index", "int32", ("n_source_detections",)),
    ArraySpec("frame_indices", "int32", ("n_source_detections",)),
    ArraySpec("bbox_img_xyxy", "float64", ("n_source_detections", 4)),
    ArraySpec("bbox_norm_coords", "float64", ("n_source_detections", 4)),
    ArraySpec("decision_codes", "int8", ("n_source_detections",)),
    ArraySpec("resolved_refined_row_id", "int64", ("n_source_detections",)),
    ArraySpec("reason_bytes", "uint8", ("n_source_detections", "width")),
    ArraySpec("reason", "string", ("n_source_detections",)),
    ArraySpec("confidence_scores", "float32", ("n_source_detections",), required=False),
    ArraySpec("class_ids", "int32", ("n_source_detections",), required=False),
    ArraySpec("review_notes", "string", ("n_source_detections",), required=False),
)

_REFINED_DETECT_INSTANCES_ARRAYS: Tuple[ArraySpec, ...] = (
    ArraySpec("refined_row_ids", "int64", ("n_instances",)),
    ArraySpec("frame_indices", "int32", ("n_instances",)),
    ArraySpec("frame_offsets", "int64", ("n_frame_offsets",)),
    ArraySpec("bbox_img_xyxy", "float64", ("n_instances", 4)),
    ArraySpec("bbox_norm_coords", "float64", ("n_instances", 4)),
    ArraySpec("source_kind_codes", "int8", ("n_instances",)),
    ArraySpec("manual_edit_flags", "bool", ("n_instances",)),
    ArraySpec("source_detect_row_index", "int32", ("n_instances",)),
    ArraySpec("frame_counts", "int32", ("n_frames",)),
    ArraySpec("reason_bytes", "uint8", ("n_instances", "width")),
    ArraySpec("reason", "string", ("n_instances",)),
    ArraySpec("confidence_scores", "float32", ("n_instances",), required=False),
    ArraySpec("class_ids", "int32", ("n_instances",), required=False),
    ArraySpec("review_notes", "string", ("n_instances",), required=False),
)

REFINED_DETECT_SPEC = StageSpec(
    stage_name="refined_detect",
    zarr_group="refined_detect_runs/<run>/",
    specs=(),
    subgroups={
        "source_detections": _REFINED_DETECT_SOURCE_DETECTIONS_ARRAYS,
        "instances": _REFINED_DETECT_INSTANCES_ARRAYS,
    },
)

CROP_SPEC = StageSpec(
    stage_name="crop",
    zarr_group="crop_runs/<run>/",
    specs=(
        ArraySpec("roi_images", "uint8", ("n_rois", "h", "w"), required=False),
        ArraySpec("roi_coordinates_full", "int32", ("n_rois", 2)),
        ArraySpec("roi_coordinates_ds", "int32", ("n_rois", 2), required=False),
        ArraySpec("bbox_norm_coords", "float32", ("n_rois", 4)),
        ArraySpec("frame_indices", "int32", ("n_rois",)),
        ArraySpec("frame_counts", "int32", ("n_frames",)),
        ArraySpec("detection_indices", "int32", ("n_rois",)),
        ArraySpec("detection_source", "int8", ("n_rois",), required=False),
    ),
)

KEYPOINTS_SPEC = StageSpec(
    stage_name="keypoints",
    zarr_group="keypoints_runs/<run>/",
    specs=(
        ArraySpec("frame_indices", "int32", ("n_rois",)),
        ArraySpec("frame_counts", "int32", ("n_frames",)),
        ArraySpec("n_rois", "int32", ("n_frames",), description="Legacy alias of frame_counts."),
        ArraySpec("detection_indices", "int32", ("n_rois",)),
        ArraySpec("keypoints_roi", "float64", ("n_rois", "n_keypoints", 2)),
        ArraySpec("keypoints_img", "float64", ("n_rois", "n_keypoints", 2)),
        ArraySpec("keypoints_norm", "float64", ("n_rois", "n_keypoints", 2)),
        ArraySpec("heading", "float64", ("n_rois",)),
        ArraySpec("confidence", "float64", ("n_rois",)),
        ArraySpec("keypoint_confidences", "float64", ("n_rois", "n_keypoints")),
        ArraySpec("effective_threshold", "float64", ("n_rois",)),
        ArraySpec("effective_se2_radius", "float64", ("n_rois",)),
        ArraySpec("detection_success", "bool", ("n_rois",)),
        ArraySpec("detection_source", "int8", ("n_rois",)),
        ArraySpec("heading_finite", "bool", ("n_rois",)),
        ArraySpec("heading_usable", "bool", ("n_rois",)),
        ArraySpec(
            "heading_delta_prev_deg",
            "float32",
            ("n_rois",),
            required=False,
            description="Circular absolute delta to the previous usable temporal heading neighbor; omitted when temporal-heading review is disabled (for example sampled imports).",
        ),
        ArraySpec(
            "heading_delta_next_deg",
            "float32",
            ("n_rois",),
            required=False,
            description="Circular absolute delta to the next usable temporal heading neighbor; omitted when temporal-heading review is disabled (for example sampled imports).",
        ),
        ArraySpec(
            "heading_temporal_outlier",
            "bool",
            ("n_rois",),
            required=False,
            description="True when both temporal heading deltas exceed the configured outlier threshold; omitted when temporal-heading review is disabled (for example sampled imports).",
        ),
        ArraySpec("n_keypoints", "int32", ("n_frames",)),
        ArraySpec("triangle_angles", "float64", ("n_rois", 3)),
        ArraySpec("triangle_angles_raw", "float64", ("n_rois", 3)),
        ArraySpec("triangle_area", "float64", ("n_rois",)),
    ),
)

REFINED_KEYPOINTS_SPEC = StageSpec(
    stage_name="refined_keypoints",
    zarr_group="refined_keypoints_runs/<run>/",
    specs=(
        ArraySpec("frame_indices", "int32", ("n_rois",)),
        ArraySpec("frame_counts", "int32", ("n_frames",)),
        ArraySpec("n_rois", "int32", ("n_frames",), description="Legacy alias of frame_counts."),
        ArraySpec("detection_indices", "int32", ("n_rois",)),
        ArraySpec("detection_source", "int8", ("n_rois",)),
        ArraySpec("retune_id", "int32", ("n_rois",), description="-1 indicates none."),
        ArraySpec("keypoints_roi", "float64", ("n_rois", "n_keypoints", 2)),
        ArraySpec("keypoints_img", "float64", ("n_rois", "n_keypoints", 2)),
        ArraySpec("keypoints_norm", "float64", ("n_rois", "n_keypoints", 2)),
        ArraySpec("heading", "float64", ("n_rois",)),
        ArraySpec("confidence", "float64", ("n_rois",)),
        ArraySpec("keypoint_confidences", "float64", ("n_rois", "n_keypoints")),
        ArraySpec(
            "effective_threshold",
            "float64",
            ("n_rois",),
            required=False,
            description="Present when available in the source run.",
        ),
        ArraySpec(
            "effective_se2_radius",
            "float64",
            ("n_rois",),
            required=False,
            description="Present when available in the source run.",
        ),
        ArraySpec("triangle_area", "float64", ("n_rois",)),
        ArraySpec("min_angle", "float64", ("n_rois",)),
        ArraySpec("triangle_angles", "float64", ("n_rois", 3)),
        ArraySpec("quality_labels", "int8", ("n_rois",)),
        ArraySpec("refined_success", "bool", ("n_rois",)),
        ArraySpec("source_success", "bool", ("n_rois",)),
        ArraySpec("flip_corrected", "bool", ("n_rois",)),
        ArraySpec("heading_finite", "bool", ("n_rois",)),
        ArraySpec("heading_usable", "bool", ("n_rois",)),
        ArraySpec("confidence_valid", "bool", ("n_rois",)),
        ArraySpec("geometry_valid", "bool", ("n_rois",)),
        ArraySpec("usable_keypoints", "bool", ("n_rois",)),
        ArraySpec(
            "edge_pairs",
            "int16/int32",
            ("n_edges", 2),
            required=False,
            description="Optional skeleton edge index pairs for per-ROI distance auditing.",
        ),
        ArraySpec(
            "edge_distances",
            "float32",
            ("n_rois", "n_edges"),
            required=False,
            description="Optional per-ROI Euclidean edge distances in ROI pixels.",
        ),
        ArraySpec(
            "edge_distances_norm",
            "float32",
            ("n_rois", "n_edges"),
            required=False,
            description="Optional per-ROI edge distances normalized by ROI diagonal.",
        ),
        ArraySpec(
            "edge_distance_valid",
            "bool",
            ("n_rois", "n_edges"),
            required=False,
            description="Optional validity mask for per-ROI edge distances.",
        ),
        ArraySpec(
            "derived_metric_values",
            "float32",
            ("n_rois", "n_metrics"),
            required=False,
            description="Optional schema-driven named derived keypoint metrics in ROI pixels.",
        ),
        ArraySpec(
            "derived_metric_values_norm",
            "float32",
            ("n_rois", "n_metrics"),
            required=False,
            description="Optional derived metrics normalized by the configured metric schema normalization.",
        ),
        ArraySpec(
            "derived_metric_valid",
            "bool",
            ("n_rois", "n_metrics"),
            required=False,
            description="Optional validity mask for schema-driven derived keypoint metrics.",
        ),
        ArraySpec("edit_applied", "bool", ("n_rois",)),
        ArraySpec("reason_bytes", "uint8", ("n_rois", "width")),
        ArraySpec("reason", "string", ("n_rois",)),
        ArraySpec("failure_indices", "int32", ("n_failures",)),
    ),
)

EYE_MASKS_SPEC = StageSpec(
    stage_name="eye_masks",
    zarr_group="eye_masks_runs/<run>/",
    specs=(
        ArraySpec("masks_roi", "uint8", ("n_rois", 2, "H", "W")),
        ArraySpec("mask_probs_roi", "float16/float32/uint8", ("n_rois", 2, "H", "W"), required=False),
        ArraySpec("mask_scores", "float32", ("n_rois",), required=False),
        ArraySpec("ellipse_params", "float32", ("n_rois", 2, 5)),
        ArraySpec("ellipse_success", "bool", ("n_rois", 2)),
        ArraySpec("eye_separation", "float32", ("n_rois",)),
        ArraySpec("detection_source", "int8", ("n_rois",)),
        ArraySpec("contour_left_ptr", "int32", ("n_rois",)),
        ArraySpec("contour_left_len", "int32", ("n_rois",)),
        ArraySpec("contour_right_ptr", "int32", ("n_rois",)),
        ArraySpec("contour_right_len", "int32", ("n_rois",)),
        ArraySpec("contours_left", "float32", ("n_points", 2)),
        ArraySpec("contours_right", "float32", ("n_points", 2)),
        ArraySpec("reason", "string", ("n_rois",)),
    ),
)

_REFINED_EYE_MASK_METRICS: Tuple[ArraySpec, ...] = (
    ArraySpec("area_refined", "float32", ("n_rois", 2)),
    ArraySpec("area_source", "float32", ("n_rois", 2)),
    ArraySpec("area_zscore", "float32", ("n_rois", 2)),
    ArraySpec("area_delta_vs_source", "float32", ("n_rois", 2)),
    ArraySpec("centroid_error", "float32", ("n_rois", 2)),
    ArraySpec("symmetry_offsets", "float32", ("n_rois", 2)),
    ArraySpec("separation_refined", "float32", ("n_rois",)),
    ArraySpec("axis_ratio", "float32", ("n_rois", 2)),
    ArraySpec("circularity", "float32", ("n_rois", 2)),
    ArraySpec("connectivity_flags", "uint8", ("n_rois",)),
    ArraySpec("smoothing_flags", "uint8", ("n_rois", 2)),
    ArraySpec("pixels_reassigned", "int32", ("n_rois",)),
    ArraySpec("probabilities_used", "bool", ("n_rois",)),
    ArraySpec("filter_flags", "uint8", ("n_rois", 2)),
    ArraySpec("reason", "string", ("n_rois",)),
)

REFINED_EYE_MASKS_SPEC = StageSpec(
    stage_name="refined_eye_masks",
    zarr_group="refined_eye_masks_runs/<run>/",
    specs=(
        ArraySpec("masks_roi", "uint8", ("n_rois", 2, "H", "W")),
        ArraySpec("edit_applied", "bool", ("n_rois", 2)),
        ArraySpec("ellipse_params", "float32", ("n_rois", 2, 5)),
        ArraySpec("ellipse_success", "bool", ("n_rois", 2)),
        ArraySpec("eye_separation", "float32", ("n_rois",)),
        ArraySpec("retune_id", "int32", ("n_rois",), required=False),
        ArraySpec("mask_probs_roi_refined", "float16", ("n_rois", 2, "H", "W"), required=False),
        ArraySpec("contour_left_ptr", "int32", ("n_rois",)),
        ArraySpec("contour_left_len", "int32", ("n_rois",)),
        ArraySpec("contour_right_ptr", "int32", ("n_rois",)),
        ArraySpec("contour_right_len", "int32", ("n_rois",)),
        ArraySpec("contours_left", "float32", ("n_points", 2)),
        ArraySpec("contours_right", "float32", ("n_points", 2)),
    ),
    subgroups={"metrics": _REFINED_EYE_MASK_METRICS},
)

_SUBJECT_MASK_METRICS: Tuple[ArraySpec, ...] = (
    ArraySpec("prob_max", "float32", ("n_rois", "n_channels")),
    ArraySpec("mask_present", "bool", ("n_rois", "n_channels")),
    ArraySpec("area_px", "float32", ("n_rois", "n_channels"), required=False),
    ArraySpec("centroid_xy", "float32", ("n_rois", "n_channels", 2), required=False),
    ArraySpec("centroid_valid", "bool", ("n_rois", "n_channels"), required=False),
    ArraySpec("bbox_xyxy", "float32", ("n_rois", "n_channels", 4), required=False),
    ArraySpec("bbox_valid", "bool", ("n_rois", "n_channels"), required=False),
)

SUBJECT_MASKS_SPEC = StageSpec(
    stage_name="subject_masks",
    zarr_group="subject_mask_runs/<run>/",
    specs=(
        ArraySpec("frame_indices", "int32", ("n_rois",), required=False),
        ArraySpec("frame_counts", "int32", ("n_frames",), required=False),
        ArraySpec("detection_indices", "int32", ("n_rois",), required=False),
        ArraySpec("detection_source", "int8", ("n_rois",)),
        ArraySpec("masks_roi", "uint8", ("n_rois", "n_channels", "H", "W")),
        ArraySpec("mask_probs_roi", "float16/float32/uint8", ("n_rois", "n_channels", "H", "W")),
        ArraySpec("available_channels", "bool", ("n_channels",)),
    ),
    subgroups={"metrics": _SUBJECT_MASK_METRICS},
)

_REFINED_SUBJECT_MASK_METRICS: Tuple[ArraySpec, ...] = (
    ArraySpec("mask_present", "bool", ("n_rois", "n_channels")),
    ArraySpec("area_px", "float32", ("n_rois", "n_channels")),
)

REFINED_SUBJECT_MASKS_SPEC = StageSpec(
    stage_name="refined_subject_masks",
    zarr_group="refined_subject_masks_runs/<run>/",
    specs=(
        ArraySpec("frame_indices", "int32", ("n_rois",), required=False),
        ArraySpec("frame_counts", "int32", ("n_frames",), required=False),
        ArraySpec("detection_indices", "int32", ("n_rois",), required=False),
        ArraySpec("detection_source", "int8", ("n_rois",)),
        ArraySpec("masks_roi", "uint8", ("n_rois", "n_channels", "H", "W")),
        ArraySpec("available_channels", "bool", ("n_channels",)),
        ArraySpec("edit_applied", "bool", ("n_rois", "n_channels")),
        ArraySpec("reason_bytes", "uint8", ("n_rois", "width"), required=False),
        ArraySpec("reason", "string", ("n_rois",), required=False),
    ),
    subgroups={"metrics": _REFINED_SUBJECT_MASK_METRICS},
)

ARENA_ASSIGNMENT_SPEC = StageSpec(
    stage_name="arena_assignment",
    zarr_group="arena_assignment_runs/<run>/",
    specs=(
        ArraySpec("arena_ids", "int32", ("n_detections",)),
        ArraySpec("confidence", "float32", ("n_detections",)),
    ),
)

TRACKING_SPEC = StageSpec(
    stage_name="tracking",
    zarr_group="tracking_runs/<run>/",
    specs=(
        ArraySpec("track_ids", "int32", ("n_detections",)),
        ArraySpec("arena_ids", "int32", ("n_detections",)),
        ArraySpec("frame_indices", "int32", ("n_detections",)),
        ArraySpec("source_row_indices", "int32", ("n_detections",)),
        ArraySpec("track_ids_present", "int32", ("n_tracks",)),
        ArraySpec("track_arena_ids", "int32", ("n_tracks",)),
    ),
)

STAGES: Dict[str, StageSpec] = {
    RAW_VIDEO_SPEC.stage_name: RAW_VIDEO_SPEC,
    BACKGROUND_SPEC.stage_name: BACKGROUND_SPEC,
    DETECT_SPEC.stage_name: DETECT_SPEC,
    DETECT_QUALITY_SPEC.stage_name: DETECT_QUALITY_SPEC,
    REFINED_DETECT_SPEC.stage_name: REFINED_DETECT_SPEC,
    CROP_SPEC.stage_name: CROP_SPEC,
    KEYPOINTS_SPEC.stage_name: KEYPOINTS_SPEC,
    REFINED_KEYPOINTS_SPEC.stage_name: REFINED_KEYPOINTS_SPEC,
    EYE_MASKS_SPEC.stage_name: EYE_MASKS_SPEC,
    REFINED_EYE_MASKS_SPEC.stage_name: REFINED_EYE_MASKS_SPEC,
    SUBJECT_MASKS_SPEC.stage_name: SUBJECT_MASKS_SPEC,
    REFINED_SUBJECT_MASKS_SPEC.stage_name: REFINED_SUBJECT_MASKS_SPEC,
    ARENA_ASSIGNMENT_SPEC.stage_name: ARENA_ASSIGNMENT_SPEC,
    TRACKING_SPEC.stage_name: TRACKING_SPEC,
}

__all__ = [
    "ArraySpec",
    "StageSpec",
    "ValidationResult",
    "RAW_VIDEO_SPEC",
    "BACKGROUND_SPEC",
    "DETECT_SPEC",
    "DETECT_QUALITY_SPEC",
    "REFINED_DETECT_SPEC",
    "CROP_SPEC",
    "KEYPOINTS_SPEC",
    "REFINED_KEYPOINTS_SPEC",
    "EYE_MASKS_SPEC",
    "REFINED_EYE_MASKS_SPEC",
    "SUBJECT_MASKS_SPEC",
    "REFINED_SUBJECT_MASKS_SPEC",
    "ARENA_ASSIGNMENT_SPEC",
    "TRACKING_SPEC",
    "STAGES",
    "array_specs_by_name",
    "describe_array",
    "required_array_names",
    "required_array_specs",
    "validate_run",
]
