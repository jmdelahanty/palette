"""Fail-closed lifecycle for unbound detection-artifact producers.

This module is deliberately narrower than the central run-completion helpers.
It protects artifact attempts from leaking selector changes and provides the
one supported destination for detection geometry that is not backed by the
canonical acquisition/coordinate publication contract.  Canonical detection
publication has a stricter staged-validation and eligibility-last lifecycle;
this helper intentionally cannot publish those runs.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Mapping
from uuid import uuid4

import numpy as np

from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
    RUN_STATUS_FAILED,
    mark_run_complete,
    mark_run_failed,
    mark_run_started,
    require_runs_parent,
)


DETECTION_ARTIFACT_RUN_FAMILY = "detection_artifact_runs"
DETECTION_ARTIFACT_FAMILY_CONTRACT = "palette.detection_artifact_family.v1"
UNBOUND_DETECTION_ARTIFACT_COORDINATE_CONTRACT = "unbound_detection_artifact_v1"
EMPTY_ARTIFACT_OBSERVATION_PROOF_ATTR = "empty_artifact_observation_proof"
EMPTY_ARTIFACT_OBSERVATION_PROOF_SCHEMA = (
    "palette.unbound_detection_artifact_zero_observation_proof.v1"
)
UNBOUND_ARTIFACT_NUMERIC_SEMANTICS_ATTR = "unbound_artifact_numeric_semantics"
UNBOUND_ARTIFACT_NUMERIC_SEMANTICS_SCHEMA = (
    "palette.unbound_artifact_numeric_semantics.v2"
)
ARTIFACT_PAYLOAD_INVENTORY_SEAL_ATTR = "artifact_payload_inventory_seal"
ARTIFACT_PAYLOAD_INVENTORY_SEAL_SCHEMA = (
    "palette.detection_artifact_payload_inventory_seal.v2"
)
STRICT_ARTIFACT_INTEGRITY_CONTRACT = "palette.detection_artifact_strict_integrity.v2"
UNBOUND_ARTIFACT_RUN_BINDING_KEY = "unbound_numeric_binding"
UNBOUND_ARTIFACT_RUN_BINDING_SCHEMA = (
    "palette.unbound_detection_artifact_run_binding.v1"
)


@dataclass(frozen=True)
class UnboundNumericProfile:
    """Immutable fixed semantics for one unbound artifact array role."""

    numeric_space_id: str
    geometry_type: str
    components: tuple[str, ...]
    component_units: tuple[str, ...]
    origin: str
    positive_x_direction: str
    positive_y_direction: str
    pixel_convention: str
    axis_0_domain: str
    row_frame_binding_kind: str
    temporal_domain_id: str
    reference_kind: str
    source_sha256_kind: str
    source_mapping_sha256_policy: str
    derivation_operation_id: str
    dtype: str
    rank: int
    trailing_shape: tuple[int, ...]


_NONSPATIAL_AXES = {
    "origin": "not_applicable",
    "positive_x_direction": "not_applicable",
    "positive_y_direction": "not_applicable",
    "pixel_convention": "not_applicable",
}
_IMAGE_AXES_CONTINUOUS = {
    "origin": "top_left",
    "positive_x_direction": "right",
    "positive_y_direction": "down",
    "pixel_convention": "continuous",
}
_IMAGE_AXES_SOURCE_EDGE_UNDECLARED = {
    "origin": "top_left",
    "positive_x_direction": "right",
    "positive_y_direction": "down",
    "pixel_convention": "source_edge_convention_undeclared",
}
_OBSERVATION_VALUE_BINDING = "axis_0_aligned_to_artifact_row_id_and_frame_indices"
_DENSE_FRAME_BINDING = "axis_0_index_equals_temporal_frame_index"


def _profile(
    *,
    numeric_space_id: str,
    geometry_type: str,
    components: tuple[str, ...],
    component_units: tuple[str, ...],
    axis_0_domain: str,
    row_frame_binding_kind: str,
    temporal_domain_id: str,
    reference_kind: str,
    source_sha256_kind: str,
    derivation_operation_id: str,
    axes: Mapping[str, str],
    dtype: str,
    trailing_shape: tuple[int, ...],
    source_mapping_sha256_policy: str = "forbidden",
) -> UnboundNumericProfile:
    return UnboundNumericProfile(
        numeric_space_id=numeric_space_id,
        geometry_type=geometry_type,
        components=components,
        component_units=component_units,
        origin=axes["origin"],
        positive_x_direction=axes["positive_x_direction"],
        positive_y_direction=axes["positive_y_direction"],
        pixel_convention=axes["pixel_convention"],
        axis_0_domain=axis_0_domain,
        row_frame_binding_kind=row_frame_binding_kind,
        temporal_domain_id=temporal_domain_id,
        reference_kind=reference_kind,
        source_sha256_kind=source_sha256_kind,
        source_mapping_sha256_policy=source_mapping_sha256_policy,
        derivation_operation_id=derivation_operation_id,
        dtype=np.dtype(dtype).str,
        rank=1 + len(trailing_shape),
        trailing_shape=trailing_shape,
    )


_TRAINING_TEMPORAL_DOMAIN = "training_selected_frame_row_v1"
_TRAINING_REFERENCE_KIND = "selected_training_frame_array"
_TRAINING_SOURCE_DIGEST_KIND = "canonical_json_artifact_frame_source_lineage_v1"
_TRADITIONAL_TEMPORAL_DOMAIN = "raw_video_images_ds_frame_row_v1"
_TRADITIONAL_REFERENCE_KIND = "raw_video_images_ds_array"
_TRADITIONAL_SOURCE_DIGEST_KIND = (
    "canonical_json_traditional_detection_source_lineage_v1"
)
_IMPORT_TEMPORAL_DOMAIN = "palette_frame_row_from_recording_frame_id_v1"
_IMPORT_REFERENCE_KIND = "manifest_declared_full_frame"
_IMPORT_SOURCE_DIGEST_KIND = (
    "canonical_json_external_detection_source_frame_evidence_v1"
)


UNBOUND_NUMERIC_PROFILES: Mapping[str, UnboundNumericProfile] = MappingProxyType(
    {
        "training.artifact_row_id.v1": _profile(
            numeric_space_id="run_local_artifact_row",
            geometry_type="row_identifier",
            components=("artifact_row_id",),
            component_units=("artifact_row_index",),
            axis_0_domain="observation_rows",
            row_frame_binding_kind="axis_0_index_equals_artifact_row_id",
            temporal_domain_id=_TRAINING_TEMPORAL_DOMAIN,
            reference_kind=_TRAINING_REFERENCE_KIND,
            source_sha256_kind=_TRAINING_SOURCE_DIGEST_KIND,
            derivation_operation_id=(
                "dense_zero_based_uint64_arange_from_output_row_count_v1"
            ),
            axes=_NONSPATIAL_AXES,
            dtype="uint64",
            trailing_shape=(),
        ),
        "training.frame_indices.v1": _profile(
            numeric_space_id="selected_training_frame_row",
            geometry_type="frame_index",
            components=("frame_index",),
            component_units=("frame_index",),
            axis_0_domain="observation_rows",
            row_frame_binding_kind=("values_bind_observation_rows_to_temporal_domain"),
            temporal_domain_id=_TRAINING_TEMPORAL_DOMAIN,
            reference_kind=_TRAINING_REFERENCE_KIND,
            source_sha256_kind=_TRAINING_SOURCE_DIGEST_KIND,
            derivation_operation_id=(
                "copy_inference_batch_training_frame_row_index_v1"
            ),
            axes=_NONSPATIAL_AXES,
            dtype="int32",
            trailing_shape=(),
        ),
        "training.bbox_norm_cxcywh.v1": _profile(
            numeric_space_id="selected_training_frame_normalized_xy",
            geometry_type="bbox_cxcywh",
            components=("center_x", "center_y", "width", "height"),
            component_units=("normalized",) * 4,
            axis_0_domain="observation_rows",
            row_frame_binding_kind=_OBSERVATION_VALUE_BINDING,
            temporal_domain_id=_TRAINING_TEMPORAL_DOMAIN,
            reference_kind=_TRAINING_REFERENCE_KIND,
            source_sha256_kind=_TRAINING_SOURCE_DIGEST_KIND,
            derivation_operation_id=("ultralytics_xyxy_to_normalized_cxcywh_v1"),
            axes=_IMAGE_AXES_CONTINUOUS,
            dtype="float64",
            trailing_shape=(4,),
        ),
        "training.scores.v1": _profile(
            numeric_space_id="model_confidence",
            geometry_type="score",
            components=("confidence",),
            component_units=("unitless",),
            axis_0_domain="observation_rows",
            row_frame_binding_kind=_OBSERVATION_VALUE_BINDING,
            temporal_domain_id=_TRAINING_TEMPORAL_DOMAIN,
            reference_kind=_TRAINING_REFERENCE_KIND,
            source_sha256_kind=_TRAINING_SOURCE_DIGEST_KIND,
            derivation_operation_id="copy_ultralytics_boxes_conf_v1",
            axes=_NONSPATIAL_AXES,
            dtype="float32",
            trailing_shape=(),
        ),
        "training.class_ids.v1": _profile(
            numeric_space_id="model_class_index",
            geometry_type="class_index",
            components=("class_id",),
            component_units=("class_index",),
            axis_0_domain="observation_rows",
            row_frame_binding_kind=_OBSERVATION_VALUE_BINDING,
            temporal_domain_id=_TRAINING_TEMPORAL_DOMAIN,
            reference_kind=_TRAINING_REFERENCE_KIND,
            source_sha256_kind=_TRAINING_SOURCE_DIGEST_KIND,
            derivation_operation_id=("cast_ultralytics_boxes_cls_to_int32_v1"),
            axes=_NONSPATIAL_AXES,
            dtype="int32",
            trailing_shape=(),
        ),
        "training.frame_counts.v1": _profile(
            numeric_space_id="selected_training_frame_row_count",
            geometry_type="frame_observation_count",
            components=("artifact_row_count",),
            component_units=("count",),
            axis_0_domain="dense_frame_rows",
            row_frame_binding_kind=_DENSE_FRAME_BINDING,
            temporal_domain_id=_TRAINING_TEMPORAL_DOMAIN,
            reference_kind=_TRAINING_REFERENCE_KIND,
            source_sha256_kind=_TRAINING_SOURCE_DIGEST_KIND,
            derivation_operation_id="bincount_frame_indices_full_domain_v1",
            axes=_NONSPATIAL_AXES,
            dtype="int32",
            trailing_shape=(),
        ),
        "training.n_detections.v1": _profile(
            numeric_space_id="selected_training_frame_row_count",
            geometry_type="frame_observation_count",
            components=("artifact_row_count",),
            component_units=("count",),
            axis_0_domain="dense_frame_rows",
            row_frame_binding_kind=_DENSE_FRAME_BINDING,
            temporal_domain_id=_TRAINING_TEMPORAL_DOMAIN,
            reference_kind=_TRAINING_REFERENCE_KIND,
            source_sha256_kind=_TRAINING_SOURCE_DIGEST_KIND,
            derivation_operation_id="exact_alias_of_frame_counts_v1",
            axes=_NONSPATIAL_AXES,
            dtype="int32",
            trailing_shape=(),
        ),
        "training.source_frame_indices.v1": _profile(
            numeric_space_id="unbound_recording_source_frame_index",
            geometry_type="frame_index",
            components=("source_frame_index",),
            component_units=("frame_index",),
            axis_0_domain="observation_rows",
            row_frame_binding_kind=_OBSERVATION_VALUE_BINDING,
            temporal_domain_id=_TRAINING_TEMPORAL_DOMAIN,
            reference_kind=_TRAINING_REFERENCE_KIND,
            source_sha256_kind=_TRAINING_SOURCE_DIGEST_KIND,
            derivation_operation_id=(
                "index_original_frame_indices_by_training_frame_row_v1"
            ),
            axes=_NONSPATIAL_AXES,
            dtype="int64",
            trailing_shape=(),
            source_mapping_sha256_policy="required",
        ),
        "traditional.artifact_row_id.v1": _profile(
            numeric_space_id="run_local_artifact_row",
            geometry_type="row_identifier",
            components=("artifact_row_id",),
            component_units=("artifact_row_index",),
            axis_0_domain="observation_rows",
            row_frame_binding_kind="axis_0_index_equals_artifact_row_id",
            temporal_domain_id=_TRADITIONAL_TEMPORAL_DOMAIN,
            reference_kind=_TRADITIONAL_REFERENCE_KIND,
            source_sha256_kind=_TRADITIONAL_SOURCE_DIGEST_KIND,
            derivation_operation_id=(
                "dense_zero_based_uint64_arange_from_output_row_count_v1"
            ),
            axes=_NONSPATIAL_AXES,
            dtype="uint64",
            trailing_shape=(),
        ),
        "traditional.frame_indices.v1": _profile(
            numeric_space_id="raw_video_images_ds_frame_row",
            geometry_type="frame_index",
            components=("frame_index",),
            component_units=("frame_index",),
            axis_0_domain="observation_rows",
            row_frame_binding_kind=("values_bind_observation_rows_to_temporal_domain"),
            temporal_domain_id=_TRADITIONAL_TEMPORAL_DOMAIN,
            reference_kind=_TRADITIONAL_REFERENCE_KIND,
            source_sha256_kind=_TRADITIONAL_SOURCE_DIGEST_KIND,
            derivation_operation_id=(
                "copy_validated_traditional_detection_frame_index_v1"
            ),
            axes=_NONSPATIAL_AXES,
            dtype="int32",
            trailing_shape=(),
        ),
        "traditional.bbox_norm_cxcywh.v1": _profile(
            numeric_space_id="raw_video_images_ds_normalized_xy",
            geometry_type="bbox_cxcywh",
            components=("center_x", "center_y", "width", "height"),
            component_units=("normalized",) * 4,
            axis_0_domain="observation_rows",
            row_frame_binding_kind=_OBSERVATION_VALUE_BINDING,
            temporal_domain_id=_TRADITIONAL_TEMPORAL_DOMAIN,
            reference_kind=_TRADITIONAL_REFERENCE_KIND,
            source_sha256_kind=_TRADITIONAL_SOURCE_DIGEST_KIND,
            derivation_operation_id=(
                "skimage_regionprops_max_exclusive_bbox_to_normalized_cxcywh_v1"
            ),
            axes=_IMAGE_AXES_CONTINUOUS,
            dtype="float64",
            trailing_shape=(4,),
        ),
        "traditional.scores.v1": _profile(
            numeric_space_id="traditional_detection_score",
            geometry_type="score",
            components=("score",),
            component_units=("unitless",),
            axis_0_domain="observation_rows",
            row_frame_binding_kind=_OBSERVATION_VALUE_BINDING,
            temporal_domain_id=_TRADITIONAL_TEMPORAL_DOMAIN,
            reference_kind=_TRADITIONAL_REFERENCE_KIND,
            source_sha256_kind=_TRADITIONAL_SOURCE_DIGEST_KIND,
            derivation_operation_id="fill_constant_one_float32_v1",
            axes=_NONSPATIAL_AXES,
            dtype="float32",
            trailing_shape=(),
        ),
        "traditional.class_ids.v1": _profile(
            numeric_space_id="traditional_detection_class_index",
            geometry_type="class_index",
            components=("class_id",),
            component_units=("class_index",),
            axis_0_domain="observation_rows",
            row_frame_binding_kind=_OBSERVATION_VALUE_BINDING,
            temporal_domain_id=_TRADITIONAL_TEMPORAL_DOMAIN,
            reference_kind=_TRADITIONAL_REFERENCE_KIND,
            source_sha256_kind=_TRADITIONAL_SOURCE_DIGEST_KIND,
            derivation_operation_id="fill_constant_zero_int32_v1",
            axes=_NONSPATIAL_AXES,
            dtype="int32",
            trailing_shape=(),
        ),
        "traditional.frame_counts.v1": _profile(
            numeric_space_id="raw_video_images_ds_frame_row_count",
            geometry_type="frame_observation_count",
            components=("artifact_row_count",),
            component_units=("count",),
            axis_0_domain="dense_frame_rows",
            row_frame_binding_kind=_DENSE_FRAME_BINDING,
            temporal_domain_id=_TRADITIONAL_TEMPORAL_DOMAIN,
            reference_kind=_TRADITIONAL_REFERENCE_KIND,
            source_sha256_kind=_TRADITIONAL_SOURCE_DIGEST_KIND,
            derivation_operation_id="bincount_frame_indices_full_domain_v1",
            axes=_NONSPATIAL_AXES,
            dtype="int32",
            trailing_shape=(),
        ),
        "traditional.n_detections.v1": _profile(
            numeric_space_id="raw_video_images_ds_frame_row_count",
            geometry_type="frame_observation_count",
            components=("artifact_row_count",),
            component_units=("count",),
            axis_0_domain="dense_frame_rows",
            row_frame_binding_kind=_DENSE_FRAME_BINDING,
            temporal_domain_id=_TRADITIONAL_TEMPORAL_DOMAIN,
            reference_kind=_TRADITIONAL_REFERENCE_KIND,
            source_sha256_kind=_TRADITIONAL_SOURCE_DIGEST_KIND,
            derivation_operation_id="exact_alias_of_frame_counts_v1",
            axes=_NONSPATIAL_AXES,
            dtype="int32",
            trailing_shape=(),
        ),
        "import.artifact_row_id.v1": _profile(
            numeric_space_id="run_local_artifact_row",
            geometry_type="row_identifier",
            components=("artifact_row_id",),
            component_units=("artifact_row_index",),
            axis_0_domain="observation_rows",
            row_frame_binding_kind="axis_0_index_equals_artifact_row_id",
            temporal_domain_id=_IMPORT_TEMPORAL_DOMAIN,
            reference_kind=_IMPORT_REFERENCE_KIND,
            source_sha256_kind=_IMPORT_SOURCE_DIGEST_KIND,
            derivation_operation_id=(
                "dense_zero_based_uint64_arange_from_output_row_count_v1"
            ),
            axes=_NONSPATIAL_AXES,
            dtype="uint64",
            trailing_shape=(),
        ),
        "import.frame_indices.v1": _profile(
            numeric_space_id="palette_zero_based_frame_row",
            geometry_type="frame_index",
            components=("frame_index",),
            component_units=("frame_index",),
            axis_0_domain="observation_rows",
            row_frame_binding_kind=("values_bind_observation_rows_to_temporal_domain"),
            temporal_domain_id=_IMPORT_TEMPORAL_DOMAIN,
            reference_kind=_IMPORT_REFERENCE_KIND,
            source_sha256_kind=_IMPORT_SOURCE_DIGEST_KIND,
            derivation_operation_id=(
                "subtract_one_from_validated_positive_recording_frame_id_v1"
            ),
            axes=_NONSPATIAL_AXES,
            dtype="int32",
            trailing_shape=(),
            source_mapping_sha256_policy="required",
        ),
        "import.bbox_norm_cxcywh.v1": _profile(
            numeric_space_id="manifest_full_frame_normalized_xy",
            geometry_type="bbox_cxcywh",
            components=("center_x", "center_y", "width", "height"),
            component_units=("normalized",) * 4,
            axis_0_domain="observation_rows",
            row_frame_binding_kind=_OBSERVATION_VALUE_BINDING,
            temporal_domain_id=_IMPORT_TEMPORAL_DOMAIN,
            reference_kind=_IMPORT_REFERENCE_KIND,
            source_sha256_kind=_IMPORT_SOURCE_DIGEST_KIND,
            derivation_operation_id=(
                "manifest_full_frame_xyxy_to_normalized_cxcywh_v1"
            ),
            axes=_IMAGE_AXES_CONTINUOUS,
            dtype="float32",
            trailing_shape=(4,),
        ),
        "import.bbox_img_xyxy.v1": _profile(
            numeric_space_id="manifest_full_frame_image_px",
            geometry_type="bbox_xyxy",
            components=("x_min", "y_min", "x_max", "y_max"),
            component_units=("px",) * 4,
            axis_0_domain="observation_rows",
            row_frame_binding_kind=_OBSERVATION_VALUE_BINDING,
            temporal_domain_id=_IMPORT_TEMPORAL_DOMAIN,
            reference_kind=_IMPORT_REFERENCE_KIND,
            source_sha256_kind=_IMPORT_SOURCE_DIGEST_KIND,
            derivation_operation_id=(
                "raw_detection_xywh_to_xyxy_float64_source_edges_undeclared_v1"
            ),
            axes=_IMAGE_AXES_SOURCE_EDGE_UNDECLARED,
            dtype="float64",
            trailing_shape=(4,),
        ),
        "import.centers_img_xy.v1": _profile(
            numeric_space_id="manifest_full_frame_image_px",
            geometry_type="point_xy",
            components=("x", "y"),
            component_units=("px", "px"),
            axis_0_domain="observation_rows",
            row_frame_binding_kind=_OBSERVATION_VALUE_BINDING,
            temporal_domain_id=_IMPORT_TEMPORAL_DOMAIN,
            reference_kind=_IMPORT_REFERENCE_KIND,
            source_sha256_kind=_IMPORT_SOURCE_DIGEST_KIND,
            derivation_operation_id="bbox_xyxy_midpoint_float64_v1",
            axes=_IMAGE_AXES_CONTINUOUS,
            dtype="float64",
            trailing_shape=(2,),
        ),
        "import.scores.v1": _profile(
            numeric_space_id="external_detection_confidence",
            geometry_type="score",
            components=("confidence",),
            component_units=("unitless",),
            axis_0_domain="observation_rows",
            row_frame_binding_kind=_OBSERVATION_VALUE_BINDING,
            temporal_domain_id=_IMPORT_TEMPORAL_DOMAIN,
            reference_kind=_IMPORT_REFERENCE_KIND,
            source_sha256_kind=_IMPORT_SOURCE_DIGEST_KIND,
            derivation_operation_id=(
                "copy_validated_crop_meta_detection_confidence_float32_v1"
            ),
            axes=_NONSPATIAL_AXES,
            dtype="float32",
            trailing_shape=(),
        ),
        "import.class_ids.v1": _profile(
            numeric_space_id="external_detection_class_index",
            geometry_type="class_index",
            components=("class_id",),
            component_units=("class_index",),
            axis_0_domain="observation_rows",
            row_frame_binding_kind=_OBSERVATION_VALUE_BINDING,
            temporal_domain_id=_IMPORT_TEMPORAL_DOMAIN,
            reference_kind=_IMPORT_REFERENCE_KIND,
            source_sha256_kind=_IMPORT_SOURCE_DIGEST_KIND,
            derivation_operation_id="fill_requested_class_id_int32_v1",
            axes=_NONSPATIAL_AXES,
            dtype="int32",
            trailing_shape=(),
        ),
        "import.frame_counts.v1": _profile(
            numeric_space_id="palette_zero_based_frame_row_count",
            geometry_type="frame_observation_count",
            components=("artifact_row_count",),
            component_units=("count",),
            axis_0_domain="dense_frame_rows",
            row_frame_binding_kind=_DENSE_FRAME_BINDING,
            temporal_domain_id=_IMPORT_TEMPORAL_DOMAIN,
            reference_kind=_IMPORT_REFERENCE_KIND,
            source_sha256_kind=_IMPORT_SOURCE_DIGEST_KIND,
            derivation_operation_id="bincount_frame_indices_full_domain_v1",
            axes=_NONSPATIAL_AXES,
            dtype="int32",
            trailing_shape=(),
        ),
        "import.n_detections.v1": _profile(
            numeric_space_id="palette_zero_based_frame_row_count",
            geometry_type="frame_observation_count",
            components=("artifact_row_count",),
            component_units=("count",),
            axis_0_domain="dense_frame_rows",
            row_frame_binding_kind=_DENSE_FRAME_BINDING,
            temporal_domain_id=_IMPORT_TEMPORAL_DOMAIN,
            reference_kind=_IMPORT_REFERENCE_KIND,
            source_sha256_kind=_IMPORT_SOURCE_DIGEST_KIND,
            derivation_operation_id="exact_alias_of_frame_counts_v1",
            axes=_NONSPATIAL_AXES,
            dtype="int32",
            trailing_shape=(),
        ),
        "import.source_crop_xywh.v1": _profile(
            numeric_space_id="manifest_full_frame_image_px",
            geometry_type="crop_xywh",
            components=("x", "y", "width", "height"),
            component_units=("px",) * 4,
            axis_0_domain="observation_rows",
            row_frame_binding_kind=_OBSERVATION_VALUE_BINDING,
            temporal_domain_id=_IMPORT_TEMPORAL_DOMAIN,
            reference_kind=_IMPORT_REFERENCE_KIND,
            source_sha256_kind=_IMPORT_SOURCE_DIGEST_KIND,
            derivation_operation_id=(
                "select_raw_crop_xywh_by_detection_row_source_edges_undeclared_v1"
            ),
            axes=_IMAGE_AXES_SOURCE_EDGE_UNDECLARED,
            dtype="float64",
            trailing_shape=(4,),
        ),
        "import.source_crop_meta_row_indices.v1": _profile(
            numeric_space_id="external_crop_meta_csv_row",
            geometry_type="source_row_index",
            components=("source_row_index",),
            component_units=("source_row_index",),
            axis_0_domain="observation_rows",
            row_frame_binding_kind=_OBSERVATION_VALUE_BINDING,
            temporal_domain_id=_IMPORT_TEMPORAL_DOMAIN,
            reference_kind=_IMPORT_REFERENCE_KIND,
            source_sha256_kind=_IMPORT_SOURCE_DIGEST_KIND,
            derivation_operation_id=(
                "copy_validated_crop_meta_source_row_index_int64_v1"
            ),
            axes=_NONSPATIAL_AXES,
            dtype="int64",
            trailing_shape=(),
        ),
        "import.source_recording_frame_ids.v1": _profile(
            numeric_space_id="recording_positive_frame_id",
            geometry_type="recording_frame_id",
            components=("recording_frame_id",),
            component_units=("recording_frame_id",),
            axis_0_domain="observation_rows",
            row_frame_binding_kind=_OBSERVATION_VALUE_BINDING,
            temporal_domain_id=_IMPORT_TEMPORAL_DOMAIN,
            reference_kind=_IMPORT_REFERENCE_KIND,
            source_sha256_kind=_IMPORT_SOURCE_DIGEST_KIND,
            derivation_operation_id=("add_one_to_zero_based_palette_frame_index_v1"),
            axes=_NONSPATIAL_AXES,
            dtype="int64",
            trailing_shape=(),
        ),
    }
)
UNBOUND_NUMERIC_MANIFEST_ATTR = "unbound_numeric_manifest_id"
UNBOUND_NUMERIC_MANIFEST_DIGEST_ATTR = "unbound_numeric_manifest_sha256"
UNBOUND_NUMERIC_MANIFEST_SCHEMA = "palette.unbound_numeric_manifest.v1"


@dataclass(frozen=True)
class UnboundProducerManifest:
    """Immutable producer-family binding from live arrays to semantic profiles."""

    producer_family_id: str
    array_profiles: tuple[tuple[str, str], ...]
    row_array_names: tuple[str, ...]
    count_array_names: tuple[str, ...]
    source_mapping_array_names: tuple[str, ...]
    source_evidence_attr: str
    source_evidence_schema_id: str
    source_mapping_direction: str | None


_TRAINING_ARRAY_PROFILES = (
    ("artifact_row_id", "training.artifact_row_id.v1"),
    ("frame_indices", "training.frame_indices.v1"),
    ("bbox_norm_coords", "training.bbox_norm_cxcywh.v1"),
    ("scores", "training.scores.v1"),
    ("class_ids", "training.class_ids.v1"),
    ("frame_counts", "training.frame_counts.v1"),
    ("n_detections", "training.n_detections.v1"),
)
_TRAINING_ROW_ARRAYS = (
    "artifact_row_id",
    "frame_indices",
    "bbox_norm_coords",
    "scores",
    "class_ids",
)
_COUNT_ARRAYS = ("frame_counts", "n_detections")


UNBOUND_PRODUCER_MANIFESTS: Mapping[str, UnboundProducerManifest] = MappingProxyType(
    {
        "training_detection_without_source_mapping.v1": UnboundProducerManifest(
            producer_family_id="training_detection_prediction",
            array_profiles=_TRAINING_ARRAY_PROFILES,
            row_array_names=_TRAINING_ROW_ARRAYS,
            count_array_names=_COUNT_ARRAYS,
            source_mapping_array_names=(),
            source_evidence_attr="artifact_frame_source_lineage",
            source_evidence_schema_id=(
                "palette.training_detection_artifact_frame_source_lineage.v1"
            ),
            source_mapping_direction=None,
        ),
        "training_detection_with_source_mapping.v1": UnboundProducerManifest(
            producer_family_id="training_detection_prediction",
            array_profiles=(
                *_TRAINING_ARRAY_PROFILES,
                (
                    "source_frame_indices",
                    "training.source_frame_indices.v1",
                ),
            ),
            row_array_names=(*_TRAINING_ROW_ARRAYS, "source_frame_indices"),
            count_array_names=_COUNT_ARRAYS,
            source_mapping_array_names=("source_frame_indices",),
            source_evidence_attr="artifact_frame_source_lineage",
            source_evidence_schema_id=(
                "palette.training_detection_artifact_frame_source_lineage.v1"
            ),
            source_mapping_direction=(
                "training_frame_row_to_recording_source_frame_index"
            ),
        ),
        "traditional_detection.v1": UnboundProducerManifest(
            producer_family_id="traditional_blob_detection",
            array_profiles=(
                ("artifact_row_id", "traditional.artifact_row_id.v1"),
                ("frame_indices", "traditional.frame_indices.v1"),
                (
                    "bbox_norm_coords",
                    "traditional.bbox_norm_cxcywh.v1",
                ),
                ("scores", "traditional.scores.v1"),
                ("class_ids", "traditional.class_ids.v1"),
                ("frame_counts", "traditional.frame_counts.v1"),
                ("n_detections", "traditional.n_detections.v1"),
            ),
            row_array_names=(
                "artifact_row_id",
                "frame_indices",
                "bbox_norm_coords",
                "scores",
                "class_ids",
            ),
            count_array_names=_COUNT_ARRAYS,
            source_mapping_array_names=(),
            source_evidence_attr="artifact_source_lineage",
            source_evidence_schema_id=(
                "palette.traditional_detection_artifact_source_lineage.v1"
            ),
            source_mapping_direction=None,
        ),
        "acquisition_detection_import.v1": UnboundProducerManifest(
            producer_family_id="acquisition_crop_meta_detection_import",
            array_profiles=(
                ("artifact_row_id", "import.artifact_row_id.v1"),
                ("frame_indices", "import.frame_indices.v1"),
                ("bbox_norm_coords", "import.bbox_norm_cxcywh.v1"),
                ("bbox_img_xyxy", "import.bbox_img_xyxy.v1"),
                ("centers_img_xy", "import.centers_img_xy.v1"),
                ("scores", "import.scores.v1"),
                ("class_ids", "import.class_ids.v1"),
                ("frame_counts", "import.frame_counts.v1"),
                ("n_detections", "import.n_detections.v1"),
                ("source_crop_xywh", "import.source_crop_xywh.v1"),
                (
                    "source_crop_meta_row_indices",
                    "import.source_crop_meta_row_indices.v1",
                ),
                (
                    "source_recording_frame_ids",
                    "import.source_recording_frame_ids.v1",
                ),
            ),
            row_array_names=(
                "artifact_row_id",
                "frame_indices",
                "bbox_norm_coords",
                "bbox_img_xyxy",
                "centers_img_xy",
                "scores",
                "class_ids",
                "source_crop_xywh",
                "source_crop_meta_row_indices",
                "source_recording_frame_ids",
            ),
            count_array_names=_COUNT_ARRAYS,
            source_mapping_array_names=("frame_indices",),
            source_evidence_attr="external_source_frame_evidence",
            source_evidence_schema_id=(
                "palette.external_detection_source_frame_evidence.v1"
            ),
            source_mapping_direction=(
                "recording_frame_id_to_palette_frame_index"
            ),
        ),
    }
)

_SELECTOR_ATTRS = (
    "latest",
    "latest_complete",
    "latest_pending",
    "authoritative_run",
    "authoritative_run_provenance",
)
_ARTIFACT_FAMILY_CONTRACT_ATTR = "artifact_family_contract"
_ARTIFACT_FAMILY_ELIGIBILITY_ATTR = "stage_selector_eligible"
_ARTIFACT_INTEGRITY_CONTRACT_ATTR = "artifact_integrity_contract"
_PUBLICATION_OWNER_ATTR = "detection_artifact_publication_owner"
_TRUSTED_RUN_ATTRS = (
    "acquisition_frame_mapping",
    "detection_acquisition_frame_mapping",
    "row_identity_contract",
    "source_row_temporal_authority",
)
_TRUSTED_NODE_ATTRS = frozenset(
    {
        "coordinate_descriptor",
        "coordinate_descriptor_sha256",
        "coordinate_descriptor_v1",
        "coordinate_descriptor_v1_sha256",
        "directed_transform",
        "directed_transform_sha256",
        "directed_transform_v2",
        "directed_transform_v2_sha256",
        "pixel_frame_authority",
        "pixel_frame_authority_sha256",
        "physical_frame_calibration",
        "physical_frame_calibration_sha256",
        "fish_anatomical_body_frame",
        "fish_anatomical_body_frame_sha256",
        "selected_camera_frame_evidence",
        "selected_camera_frame_evidence_sha256",
        "row_identity_contract",
        "row_identity_contract_sha256",
        "row_identity_key",
        "row_identity_key_sha256",
        "source_row_temporal_authority",
        "source_row_temporal_authority_sha256",
        "track_sample_time_lineage",
        "track_sample_time_lineage_sha256",
        "transform_authority",
        "transform_authority_sha256",
        "unbound_coordinate_semantics",
        "unbound_temporal_semantics",
    }
)
_IDENTITY_ARRAY_NAMES = frozenset(
    {
        "instance_key",
        "source_instance_key",
        "track_sample_key",
        "source_acquisition_frame_index",
        "source_row_index",
    }
)
_MISSING = object()


def _snapshot(attrs: Any) -> dict[str, Any]:
    return {
        name: deepcopy(attrs[name]) if name in attrs else _MISSING
        for name in _SELECTOR_ATTRS
    }


def _restore(attrs: Any, snapshot: Mapping[str, Any]) -> None:
    failures: list[str] = []
    for name, value in snapshot.items():
        try:
            if value is _MISSING:
                if name in attrs:
                    del attrs[name]
            else:
                attrs[name] = deepcopy(value)
        except BaseException as exc:  # pragma: no cover - hostile persistent mapping
            failures.append(f"{name}: {exc}")
    try:
        matches = _matches(attrs, snapshot)
    except BaseException as exc:  # pragma: no cover - hostile persistent mapping
        failures.append(f"verification: {exc}")
        matches = False
    if not matches and not failures:
        failures.append("persisted selector state differs from the exact snapshot")
    if failures:
        raise RuntimeError(
            f"Detection artifact selector restoration was incomplete: {failures!r}."
        )


def _matches(attrs: Any, snapshot: Mapping[str, Any]) -> bool:
    for name, expected in snapshot.items():
        if expected is _MISSING:
            if name in attrs:
                return False
        elif name not in attrs or attrs[name] != expected:
            return False
    return True


def _missing_selector_snapshot() -> dict[str, Any]:
    return {name: _MISSING for name in _SELECTOR_ATTRS}


def _child(node: Any, name: str) -> Any | None:
    try:
        if name in node:
            return node[name]
    except BaseException:
        pass
    try:
        return node.get(name)
    except BaseException:
        return None


def _child_names(node: Any) -> tuple[str, ...] | None:
    try:
        return tuple(str(name) for name in node.keys())
    except BaseException:
        return None


def _require_selector_free_artifact_parent(
    parent: Any,
    *,
    stamp_missing: bool,
) -> None:
    attrs = parent.attrs
    forbidden = tuple(name for name in _SELECTOR_ATTRS if name in attrs)
    if forbidden:
        raise ValueError(
            "detection_artifact_runs is a selector-free namespace and cannot "
            f"carry selector attrs: {forbidden!r}."
        )
    contract = attrs.get(_ARTIFACT_FAMILY_CONTRACT_ATTR)
    if contract is None and stamp_missing:
        attrs[_ARTIFACT_FAMILY_CONTRACT_ATTR] = DETECTION_ARTIFACT_FAMILY_CONTRACT
        contract = DETECTION_ARTIFACT_FAMILY_CONTRACT
    if contract != DETECTION_ARTIFACT_FAMILY_CONTRACT:
        raise ValueError(
            "detection_artifact_runs has a missing or unsupported artifact family "
            f"contract: {contract!r}."
        )
    eligibility = attrs.get(_ARTIFACT_FAMILY_ELIGIBILITY_ATTR)
    if eligibility is None and stamp_missing:
        attrs[_ARTIFACT_FAMILY_ELIGIBILITY_ATTR] = False
        eligibility = False
    if eligibility is not False:
        raise ValueError(
            "detection_artifact_runs must declare stage_selector_eligible=false."
        )


def _owned_child(
    parent: Any,
    run_name: str,
    owner_token: str,
) -> Any | None:
    child = _child(parent, run_name)
    if child is None:
        return None
    try:
        if child.attrs.get(_PUBLICATION_OWNER_ATTR) != owner_token:
            return None
    except BaseException:
        return None
    return child


def _rollback_failed_begin(
    *,
    root: Any,
    family: str,
    run_name: str,
    parent: Any | None,
    run: Any | None,
    parent_preexisting: bool,
    creation_started: bool,
    owner_token: str,
    selector_snapshot: Mapping[str, Any],
    cause: BaseException,
) -> BaseException | None:
    """Best-effort cleanup with a final fail-closed state verification."""

    current_parent = parent if parent is not None else _child(root, family)
    if current_parent is None:
        return (
            None
            if not parent_preexisting
            else RuntimeError(
                "Pre-existing detection parent disappeared during failed setup."
            )
        )

    owned_run = None
    if creation_started:
        owned_run = _owned_child(current_parent, run_name, owner_token)
        if owned_run is not None:
            try:
                mark_run_failed(
                    owned_run,
                    parent_group=current_parent,
                    run_name=run_name,
                    error=f"detection producer setup failed: {cause}",
                )
            except BaseException:
                pass
            try:
                del current_parent[run_name]
            except BaseException:
                # A store that cannot delete must retain an explicitly failed
                # child. Final verification below rejects every other state.
                pass

    try:
        _restore(current_parent.attrs, selector_snapshot)
    except BaseException as exc:
        return exc

    current_parent = _child(root, family)
    if not parent_preexisting and current_parent is not None:
        names = _child_names(current_parent)
        if names == ():
            try:
                del root[family]
            except BaseException:
                pass
            current_parent = _child(root, family)

    if current_parent is None:
        return (
            None
            if not parent_preexisting
            else RuntimeError(
                "Pre-existing detection parent disappeared during failed setup."
            )
        )
    try:
        selectors_match = _matches(current_parent.attrs, selector_snapshot)
    except BaseException:
        return RuntimeError(
            "Detection setup rollback could not verify selector restoration."
        )
    if not selectors_match:
        return RuntimeError(
            "Detection setup rollback did not restore exact selector state."
        )

    current_run = _owned_child(current_parent, run_name, owner_token)
    if current_run is not None:
        try:
            failed = (
                current_run.attrs.get(RUN_COMPLETION_STATUS_ATTR) == RUN_STATUS_FAILED
            )
        except BaseException:
            failed = False
        if not failed:
            return RuntimeError(
                "Failed detection setup left a child that is neither deleted nor failed."
            )
    if not parent_preexisting and current_run is None:
        names = _child_names(current_parent)
        if names == ():
            return RuntimeError(
                "Failed detection setup could not remove its newly-created empty parent."
            )
    return None


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_evidence(array: Any) -> dict[str, Any]:
    shape = tuple(int(value) for value in array.shape)
    dtype = np.dtype(array.dtype)
    if dtype.hasobject:
        raise ValueError("Detection artifact proof arrays cannot use object dtype.")
    values = np.asarray(array[...])
    if values.shape != shape or values.dtype != dtype:
        raise ValueError(
            "Detection artifact proof array changed shape or dtype while reading."
        )
    digest = hashlib.sha256()
    digest.update(b"palette.ndarray_payload.v1\x00")
    digest.update(dtype.str.encode("ascii"))
    digest.update(b"\x00")
    digest.update(np.asarray(shape, dtype="<i8").tobytes())
    digest.update(np.ascontiguousarray(values).tobytes(order="C"))
    return {
        "dtype": dtype.str,
        "shape": list(shape),
        "sha256": digest.hexdigest(),
    }


def _exact_text(value: Any, *, field_name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{field_name} must be an exact unpadded nonempty string.")
    return value


def _require_sha256(value: Any, *, field_name: str) -> str:
    text = _exact_text(value, field_name=field_name)
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest.")
    return text


_PROFILE_FIXED_FIELDS = (
    "numeric_space_id",
    "geometry_type",
    "components",
    "component_units",
    "origin",
    "positive_x_direction",
    "positive_y_direction",
    "pixel_convention",
    "axis_0_domain",
    "row_frame_binding_kind",
    "temporal_domain_id",
    "source_sha256_kind",
    "source_mapping_sha256_policy",
    "dtype",
    "rank",
    "trailing_shape",
)
_CONTROLLED_COMPONENT_UNITS = frozenset(
    {
        "px",
        "normalized",
        "unitless",
        "artifact_row_index",
        "frame_index",
        "source_row_index",
        "recording_frame_id",
        "class_index",
        "count",
    }
)
_CONTROLLED_PIXEL_CONVENTIONS = frozenset(
    {
        "continuous",
        "pixel_edge_half_open",
        "source_edge_convention_undeclared",
        "not_applicable",
    }
)
_CONTROLLED_REFERENCE_KINDS = frozenset(
    {
        "selected_training_frame_array",
        "raw_video_images_ds_array",
        "manifest_declared_full_frame",
    }
)
_CONTROLLED_SOURCE_EVIDENCE_SCHEMAS = frozenset(
    {
        "palette.training_detection_artifact_frame_source_lineage.v1",
        "palette.traditional_detection_artifact_source_lineage.v1",
        "palette.external_detection_source_frame_evidence.v1",
    }
)


def _profile_fixed_record(profile: UnboundNumericProfile) -> dict[str, Any]:
    return {
        "numeric_space_id": profile.numeric_space_id,
        "geometry_type": profile.geometry_type,
        "components": list(profile.components),
        "component_units": list(profile.component_units),
        "origin": profile.origin,
        "positive_x_direction": profile.positive_x_direction,
        "positive_y_direction": profile.positive_y_direction,
        "pixel_convention": profile.pixel_convention,
        "axis_0_domain": profile.axis_0_domain,
        "row_frame_binding_kind": profile.row_frame_binding_kind,
        "temporal_domain_id": profile.temporal_domain_id,
        "source_sha256_kind": profile.source_sha256_kind,
        "source_mapping_sha256_policy": profile.source_mapping_sha256_policy,
        "dtype": profile.dtype,
        "rank": profile.rank,
        "trailing_shape": list(profile.trailing_shape),
    }


def _manifest_record(
    manifest_id: str,
    manifest: UnboundProducerManifest,
) -> dict[str, Any]:
    return {
        "schema_id": UNBOUND_NUMERIC_MANIFEST_SCHEMA,
        "manifest_id": manifest_id,
        "producer_family_id": manifest.producer_family_id,
        "array_profiles": [list(item) for item in manifest.array_profiles],
        "row_array_names": list(manifest.row_array_names),
        "count_array_names": list(manifest.count_array_names),
        "source_mapping_array_names": list(manifest.source_mapping_array_names),
        "source_evidence_attr": manifest.source_evidence_attr,
        "source_evidence_schema_id": manifest.source_evidence_schema_id,
        "source_mapping_direction": manifest.source_mapping_direction,
    }


def _resolve_manifest(manifest_id: Any) -> UnboundProducerManifest:
    resolved = _exact_text(manifest_id, field_name="unbound_numeric_manifest_id")
    manifest = UNBOUND_PRODUCER_MANIFESTS.get(resolved)
    if manifest is None:
        raise ValueError(f"Unbound numeric manifest is not registered: {resolved!r}.")
    return manifest


def _require_run_manifest(
    run: Any,
    *,
    expected_manifest_id: str | None = None,
) -> tuple[str, UnboundProducerManifest]:
    manifest_id = _exact_text(
        run.attrs.get(UNBOUND_NUMERIC_MANIFEST_ATTR),
        field_name=UNBOUND_NUMERIC_MANIFEST_ATTR,
    )
    if expected_manifest_id is not None and manifest_id != expected_manifest_id:
        raise ValueError(
            "Detection artifact numeric manifest changed from its producer binding."
        )
    manifest = _resolve_manifest(manifest_id)
    expected_digest = _canonical_sha256(_manifest_record(manifest_id, manifest))
    if run.attrs.get(UNBOUND_NUMERIC_MANIFEST_DIGEST_ATTR) != expected_digest:
        raise ValueError("Detection artifact numeric manifest digest is invalid.")
    return manifest_id, manifest


def _validate_profile_registry() -> None:
    for profile_id, profile in UNBOUND_NUMERIC_PROFILES.items():
        _exact_text(profile_id, field_name="semantic_profile_id")
        fixed = _profile_fixed_record(profile)
        for name in (
            "numeric_space_id",
            "geometry_type",
            "origin",
            "positive_x_direction",
            "positive_y_direction",
            "pixel_convention",
            "axis_0_domain",
            "row_frame_binding_kind",
            "temporal_domain_id",
            "source_sha256_kind",
            "source_mapping_sha256_policy",
            "dtype",
        ):
            _exact_text(fixed[name], field_name=f"profile.{name}")
        if (
            not profile.components
            or len(profile.components) != len(profile.component_units)
            or any(
                type(component) is not str or not component
                for component in profile.components
            )
            or any(
                unit not in _CONTROLLED_COMPONENT_UNITS
                for unit in profile.component_units
            )
        ):
            raise RuntimeError(
                f"Unbound numeric profile {profile_id!r} has invalid components."
            )
        if profile.origin not in {"top_left", "not_applicable"}:
            raise RuntimeError(
                f"Unbound numeric profile {profile_id!r} has invalid origin."
            )
        if profile.positive_x_direction not in {"right", "not_applicable"}:
            raise RuntimeError(
                f"Unbound numeric profile {profile_id!r} has invalid +X direction."
            )
        if profile.positive_y_direction not in {"down", "not_applicable"}:
            raise RuntimeError(
                f"Unbound numeric profile {profile_id!r} has invalid +Y direction."
            )
        if profile.pixel_convention not in _CONTROLLED_PIXEL_CONVENTIONS:
            raise RuntimeError(
                f"Unbound numeric profile {profile_id!r} has invalid pixel convention."
            )
        if profile.axis_0_domain not in {"observation_rows", "dense_frame_rows"}:
            raise RuntimeError(
                f"Unbound numeric profile {profile_id!r} has invalid axis-0 domain."
            )
        if profile.reference_kind not in _CONTROLLED_REFERENCE_KINDS:
            raise RuntimeError(
                f"Unbound numeric profile {profile_id!r} has invalid reference kind."
            )
        if profile.source_mapping_sha256_policy not in {"required", "forbidden"}:
            raise RuntimeError(
                f"Unbound numeric profile {profile_id!r} has invalid mapping policy."
            )
        try:
            resolved_dtype = np.dtype(profile.dtype)
        except TypeError as exc:
            raise RuntimeError(
                f"Unbound numeric profile {profile_id!r} has invalid dtype."
            ) from exc
        if (
            resolved_dtype.hasobject
            or resolved_dtype.str != profile.dtype
            or type(profile.rank) is not int
            or profile.rank <= 0
            or type(profile.trailing_shape) is not tuple
            or profile.rank != 1 + len(profile.trailing_shape)
            or any(type(value) is not int or value <= 0 for value in profile.trailing_shape)
            or (
                len(profile.components) == 1
                and profile.trailing_shape != ()
            )
            or (
                len(profile.components) > 1
                and profile.trailing_shape != (len(profile.components),)
            )
        ):
            raise RuntimeError(
                f"Unbound numeric profile {profile_id!r} has invalid structural shape."
            )
        _exact_text(
            profile.derivation_operation_id,
            field_name="profile.derivation_operation_id",
        )


def _validate_manifest_registry() -> None:
    for manifest_id, manifest in UNBOUND_PRODUCER_MANIFESTS.items():
        _exact_text(manifest_id, field_name="manifest_id")
        _exact_text(manifest.producer_family_id, field_name="producer_family_id")
        _exact_text(
            manifest.source_evidence_attr,
            field_name="source_evidence_attr",
        )
        _exact_text(
            manifest.source_evidence_schema_id,
            field_name="source_evidence_schema_id",
        )
        if manifest.source_evidence_schema_id not in _CONTROLLED_SOURCE_EVIDENCE_SCHEMAS:
            raise RuntimeError(
                f"Unbound numeric manifest {manifest_id!r} has an unsupported "
                "source evidence schema."
            )
        array_profiles = dict(manifest.array_profiles)
        if len(array_profiles) != len(manifest.array_profiles):
            raise RuntimeError(
                f"Unbound numeric manifest {manifest_id!r} repeats array names."
            )
        row_names = set(manifest.row_array_names)
        count_names = set(manifest.count_array_names)
        mapping_names = set(manifest.source_mapping_array_names)
        if (
            not row_names
            or "artifact_row_id" not in row_names
            or "frame_indices" not in row_names
            or manifest.count_array_names != _COUNT_ARRAYS
            or row_names & count_names
            or set(array_profiles) != row_names | count_names
            or len(mapping_names) != len(manifest.source_mapping_array_names)
            or not mapping_names <= row_names
        ):
            raise RuntimeError(
                f"Unbound numeric manifest {manifest_id!r} has invalid inventory."
            )
        profiles: list[UnboundNumericProfile] = []
        for array_name, profile_id in manifest.array_profiles:
            _exact_text(array_name, field_name="manifest array name")
            profile = UNBOUND_NUMERIC_PROFILES.get(profile_id)
            if profile is None:
                raise RuntimeError(
                    f"Manifest {manifest_id!r} references unknown profile {profile_id!r}."
                )
            expected_axis = (
                "dense_frame_rows" if array_name in count_names else "observation_rows"
            )
            if profile.axis_0_domain != expected_axis:
                raise RuntimeError(
                    f"Manifest {manifest_id!r} profile {profile_id!r} has wrong axis-0 domain."
                )
            profiles.append(profile)
        required_mapping_names = {
            array_name
            for array_name, profile_id in manifest.array_profiles
            if UNBOUND_NUMERIC_PROFILES[profile_id].source_mapping_sha256_policy
            == "required"
        }
        if mapping_names != required_mapping_names:
            raise RuntimeError(
                f"Unbound numeric manifest {manifest_id!r} mapping anchors do not "
                "match its profile policies."
            )
        if mapping_names:
            _exact_text(
                manifest.source_mapping_direction,
                field_name="source_mapping_direction",
            )
        elif manifest.source_mapping_direction is not None:
            raise RuntimeError(
                f"Unbound numeric manifest {manifest_id!r} declares a mapping "
                "direction without mapping arrays."
            )
        for field_name in (
            "reference_kind",
            "temporal_domain_id",
            "source_sha256_kind",
        ):
            if len({getattr(profile, field_name) for profile in profiles}) != 1:
                raise RuntimeError(
                    f"Manifest {manifest_id!r} mixes profile {field_name}."
                )


def _require_json_evidence(value: Any, *, field_name: str) -> Any:
    if value is None or type(value) in {bool, int, str}:
        return deepcopy(value)
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{field_name} cannot contain non-finite numbers.")
        return value
    if isinstance(value, (list, tuple)):
        return [
            _require_json_evidence(item, field_name=f"{field_name}[]") for item in value
        ]
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            name = _exact_text(key, field_name=f"{field_name} key")
            normalized[name] = _require_json_evidence(
                item,
                field_name=f"{field_name}.{name}",
            )
        return normalized
    raise ValueError(f"{field_name} must contain only exact JSON evidence values.")


def build_unbound_artifact_run_binding(
    *,
    manifest_id: str,
    reference_node_path: str,
    reference_width: int,
    reference_height: int,
    source_frame_count: int,
    source_mapping_sha256: str | None = None,
) -> dict[str, Any]:
    """Build the exact manifest-owned binding embedded in source evidence."""

    resolved_manifest_id = _exact_text(manifest_id, field_name="manifest_id")
    manifest = _resolve_manifest(resolved_manifest_id)
    profiles = [
        UNBOUND_NUMERIC_PROFILES[profile_id]
        for _array_name, profile_id in manifest.array_profiles
    ]
    reference_kinds = {profile.reference_kind for profile in profiles}
    temporal_domains = {profile.temporal_domain_id for profile in profiles}
    if len(reference_kinds) != 1 or len(temporal_domains) != 1:
        raise RuntimeError(
            f"Manifest {resolved_manifest_id!r} has inconsistent binding profiles."
        )
    if (
        type(reference_width) is not int
        or type(reference_height) is not int
        or reference_width <= 0
        or reference_height <= 0
        or type(source_frame_count) is not int
        or source_frame_count <= 0
    ):
        raise ValueError(
            "Artifact run binding requires positive exact reference extents and "
            "source frame count."
        )
    if manifest.source_mapping_array_names:
        if source_mapping_sha256 is None:
            raise ValueError(
                "Mapped artifact manifest requires exact source mapping evidence."
            )
        mapping: dict[str, Any] | None = {
            "direction": manifest.source_mapping_direction,
            "sha256": _require_sha256(
                source_mapping_sha256,
                field_name="source_mapping_sha256",
            ),
        }
    else:
        if source_mapping_sha256 is not None:
            raise ValueError(
                "Unmapped artifact manifest forbids source mapping evidence."
            )
        mapping = None
    return {
        "schema_id": UNBOUND_ARTIFACT_RUN_BINDING_SCHEMA,
        "manifest_id": resolved_manifest_id,
        "reference": {
            "kind": next(iter(reference_kinds)),
            "node_path": _exact_text(
                reference_node_path,
                field_name="reference_node_path",
            ),
            "width": reference_width,
            "height": reference_height,
        },
        "temporal": {
            "domain_id": next(iter(temporal_domains)),
            "source_frame_count": source_frame_count,
        },
        "mapping": mapping,
    }


def _validate_run_binding_record(
    binding: Any,
    *,
    manifest_id: str,
    manifest: UnboundProducerManifest,
) -> dict[str, Any]:
    if not isinstance(binding, Mapping):
        raise ValueError("Detection artifact source evidence lacks a run binding.")
    record = deepcopy(dict(binding))
    if set(record) != {
        "schema_id",
        "manifest_id",
        "reference",
        "temporal",
        "mapping",
    }:
        raise ValueError("Detection artifact run-binding fields are not exact.")
    if (
        record["schema_id"] != UNBOUND_ARTIFACT_RUN_BINDING_SCHEMA
        or record["manifest_id"] != manifest_id
    ):
        raise ValueError("Detection artifact run-binding header is invalid.")
    profiles = [
        UNBOUND_NUMERIC_PROFILES[profile_id]
        for _array_name, profile_id in manifest.array_profiles
    ]
    reference = record["reference"]
    temporal = record["temporal"]
    if not isinstance(reference, Mapping) or set(reference) != {
        "kind",
        "node_path",
        "width",
        "height",
    }:
        raise ValueError("Detection artifact run reference evidence is invalid.")
    if (
        reference["kind"] != profiles[0].reference_kind
        or any(profile.reference_kind != reference["kind"] for profile in profiles)
    ):
        raise ValueError("Detection artifact run reference kind is inconsistent.")
    _exact_text(reference["node_path"], field_name="reference.node_path")
    if (
        type(reference["width"]) is not int
        or type(reference["height"]) is not int
        or reference["width"] <= 0
        or reference["height"] <= 0
    ):
        raise ValueError("Detection artifact run reference extent is invalid.")
    if not isinstance(temporal, Mapping) or set(temporal) != {
        "domain_id",
        "source_frame_count",
    }:
        raise ValueError("Detection artifact run temporal evidence is invalid.")
    if (
        temporal["domain_id"] != profiles[0].temporal_domain_id
        or any(
            profile.temporal_domain_id != temporal["domain_id"]
            for profile in profiles
        )
        or type(temporal["source_frame_count"]) is not int
        or temporal["source_frame_count"] <= 0
    ):
        raise ValueError("Detection artifact run temporal authority is inconsistent.")
    mapping = record["mapping"]
    if manifest.source_mapping_array_names:
        if not isinstance(mapping, Mapping) or set(mapping) != {
            "direction",
            "sha256",
        }:
            raise ValueError("Detection artifact run mapping evidence is invalid.")
        if mapping["direction"] != manifest.source_mapping_direction:
            raise ValueError("Detection artifact run mapping direction is invalid.")
        _require_sha256(mapping["sha256"], field_name="mapping.sha256")
    elif mapping is not None:
        raise ValueError("Unmapped artifact run carries mapping evidence.")
    return record


def _validate_source_evidence_internal_consistency(
    evidence: Mapping[str, Any],
    *,
    manifest: UnboundProducerManifest,
    binding: Mapping[str, Any],
) -> None:
    """Cross-check each registered producer evidence schema with its binding."""

    reference = binding["reference"]
    temporal = binding["temporal"]
    mapping = binding["mapping"]
    schema_id = manifest.source_evidence_schema_id
    if schema_id == "palette.traditional_detection_artifact_source_lineage.v1":
        frame_source = evidence.get("frame_source")
        if not isinstance(frame_source, Mapping):
            raise ValueError("Traditional artifact source evidence is incomplete.")
        shape = frame_source.get("shape")
        if (
            frame_source.get("node_path") != reference["node_path"]
            or not isinstance(shape, list)
            or len(shape) != 3
            or shape[0] != temporal["source_frame_count"]
            or shape[1] != reference["height"]
            or shape[2] != reference["width"]
        ):
            raise ValueError("Traditional artifact source/reference evidence disagrees.")
        _require_sha256(
            frame_source.get("content_sha256"),
            field_name="frame_source.content_sha256",
        )
    elif schema_id == "palette.training_detection_artifact_frame_source_lineage.v1":
        extent = evidence.get("frame_source_extent")
        shape = evidence.get("selected_array_shape")
        original_mapping = evidence.get("original_frame_mapping")
        if (
            evidence.get("selected_array_path") != reference["node_path"]
            or evidence.get("frame_row_count") != temporal["source_frame_count"]
            or not isinstance(extent, Mapping)
            or extent.get("width") != reference["width"]
            or extent.get("height") != reference["height"]
            or not isinstance(shape, list)
            or len(shape) not in {3, 4}
            or shape[0] != temporal["source_frame_count"]
            or shape[1] != reference["height"]
            or shape[2] != reference["width"]
        ):
            raise ValueError("Training artifact source/reference evidence disagrees.")
        if not isinstance(original_mapping, Mapping):
            raise ValueError("Training artifact mapping evidence is missing.")
        if mapping is None:
            if original_mapping.get("status") != "absent":
                raise ValueError("Unmapped training artifact carries source mapping.")
        elif (
            original_mapping.get("status") != "present_unbound_source_evidence"
            or original_mapping.get("direction") != mapping["direction"]
            or original_mapping.get("source_payload_sha256") != mapping["sha256"]
        ):
            raise ValueError("Training artifact source mapping evidence disagrees.")
    elif schema_id == "palette.external_detection_source_frame_evidence.v1":
        if (
            evidence.get("manifest_full_stream_ref") != reference["node_path"]
            or evidence.get("reference_width") != reference["width"]
            or evidence.get("reference_height") != reference["height"]
            or evidence.get("frame_count") != temporal["source_frame_count"]
            or mapping is None
            or evidence.get("direction") != mapping["direction"]
            or evidence.get("recording_frame_ids_sha256") != mapping["sha256"]
        ):
            raise ValueError(
                "Acquisition-import source/reference or mapping evidence disagrees."
            )
        for name in ("crop_meta_sha256", "recording_manifest_sha256"):
            _require_sha256(evidence.get(name), field_name=name)
    else:  # pragma: no cover - registry validation makes this unreachable
        raise RuntimeError(f"Unsupported artifact source evidence schema: {schema_id!r}.")


def _resolve_run_owned_source_evidence(
    run: Any,
    *,
    manifest_id: str,
    manifest: UnboundProducerManifest,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    attr_name = manifest.source_evidence_attr
    digest_name = f"{attr_name}_sha256"
    value = run.attrs.get(attr_name)
    digest = run.attrs.get(digest_name)
    if not isinstance(value, Mapping) or type(digest) is not str:
        raise ValueError(
            "Detection artifact lacks its manifest-declared source evidence/digest."
        )
    evidence = deepcopy(dict(value))
    if evidence.get("schema_id") != manifest.source_evidence_schema_id:
        raise ValueError("Detection artifact source evidence schema is invalid.")
    normalized = _require_json_evidence(evidence, field_name=attr_name)
    if not isinstance(normalized, dict):
        raise ValueError("Detection artifact source evidence is not canonical JSON.")
    evidence = normalized
    observed_digest = _canonical_sha256(evidence)
    if digest != observed_digest:
        raise ValueError("Detection artifact source evidence digest does not match.")
    binding = _validate_run_binding_record(
        evidence.get(UNBOUND_ARTIFACT_RUN_BINDING_KEY),
        manifest_id=manifest_id,
        manifest=manifest,
    )
    _validate_source_evidence_internal_consistency(
        evidence,
        manifest=manifest,
        binding=binding,
    )
    return evidence, observed_digest, binding


_validate_profile_registry()
_validate_manifest_registry()


def stamp_unbound_artifact_numeric_semantics(
    array: Any,
    *,
    semantic_profile_id: str,
    reference_node_path: str,
    reference_width: int,
    reference_height: int,
    source_frame_count: int,
    source_sha256: str,
    source_mapping_sha256: str | None = None,
    derivation_parameters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Stamp one registered unbound profile plus schema-checked live evidence."""

    resolved_profile_id = _exact_text(
        semantic_profile_id,
        field_name="semantic_profile_id",
    )
    profile = UNBOUND_NUMERIC_PROFILES.get(resolved_profile_id)
    if profile is None:
        raise ValueError(
            "Unbound artifact semantic_profile_id is not registered: "
            f"{resolved_profile_id!r}."
        )
    if (
        type(reference_width) is not int
        or type(reference_height) is not int
        or reference_width <= 0
        or reference_height <= 0
    ):
        raise ValueError(
            "Unbound artifact reference width and height must be positive exact integers."
        )
    if type(source_frame_count) is not int or source_frame_count <= 0:
        raise ValueError(
            "Unbound artifact source_frame_count must be a positive exact integer."
        )
    mapping_policy = profile.source_mapping_sha256_policy
    if mapping_policy == "required" and source_mapping_sha256 is None:
        raise ValueError(
            "Registered semantic profile requires source mapping evidence."
        )
    if mapping_policy == "forbidden" and source_mapping_sha256 is not None:
        raise ValueError("Registered semantic profile forbids source mapping evidence.")
    if source_mapping_sha256 is not None:
        resolved_mapping_sha256: str | None = _require_sha256(
            source_mapping_sha256,
            field_name="source_mapping_sha256",
        )
    else:
        resolved_mapping_sha256 = None
    parameters = _require_json_evidence(
        {} if derivation_parameters is None else derivation_parameters,
        field_name="derivation_parameters",
    )
    if not isinstance(parameters, dict):
        raise ValueError("derivation_parameters must be a JSON object.")
    evidence = _array_evidence(array)
    shape = tuple(evidence["shape"])
    if (
        evidence["dtype"] != profile.dtype
        or len(shape) != profile.rank
        or shape[1:] != profile.trailing_shape
    ):
        raise ValueError(
            "Unbound artifact dtype, rank, or trailing shape does not match its "
            "registered semantic profile."
        )
    if profile.axis_0_domain == "dense_frame_rows" and shape[0] != source_frame_count:
        raise ValueError(
            "Dense-frame semantic profile shape must match source_frame_count."
        )
    record = {
        "schema_id": UNBOUND_ARTIFACT_NUMERIC_SEMANTICS_SCHEMA,
        "semantic_profile_id": resolved_profile_id,
        "canonical_binding_status": "unbound",
        **_profile_fixed_record(profile),
        "reference": {
            "kind": profile.reference_kind,
            "node_path": _exact_text(
                reference_node_path,
                field_name="reference_node_path",
            ),
            "width": reference_width,
            "height": reference_height,
        },
        "temporal_evidence": {
            "source_frame_count": source_frame_count,
            "source_mapping_sha256": resolved_mapping_sha256,
        },
        "source_sha256": _require_sha256(
            source_sha256,
            field_name="source_sha256",
        ),
        "derivation": {
            "operation_id": profile.derivation_operation_id,
            "parameters": parameters,
        },
        "shape": evidence["shape"],
        "source_camera_overlay_suitability": "unsupported",
        "canonical_promotion_suitability": "unsupported",
    }
    attr_name = UNBOUND_ARTIFACT_NUMERIC_SEMANTICS_ATTR
    digest_name = f"{attr_name}_sha256"
    snapshot = {
        attr_name: deepcopy(array.attrs[attr_name])
        if attr_name in array.attrs
        else _MISSING,
        digest_name: deepcopy(array.attrs[digest_name])
        if digest_name in array.attrs
        else _MISSING,
    }
    try:
        array.attrs[attr_name] = record
        array.attrs[digest_name] = _canonical_sha256(record)
        validate_unbound_artifact_numeric_semantics(array)
    except BaseException:
        _restore(array.attrs, snapshot)
        raise
    return record


def validate_unbound_artifact_numeric_semantics(array: Any) -> dict[str, Any]:
    """Revalidate one array-owned unbound numeric-semantics record."""

    attr_name = UNBOUND_ARTIFACT_NUMERIC_SEMANTICS_ATTR
    digest_name = f"{attr_name}_sha256"
    value = array.attrs.get(attr_name)
    digest = array.attrs.get(digest_name)
    if not isinstance(value, Mapping) or type(digest) is not str:
        raise ValueError("Artifact array lacks numeric semantics and digest.")
    record = deepcopy(dict(value))
    if _canonical_sha256(record) != digest:
        raise ValueError("Artifact array numeric-semantics digest does not match.")
    required = {
        "schema_id",
        "semantic_profile_id",
        "canonical_binding_status",
        "numeric_space_id",
        "components",
        "component_units",
        "origin",
        "positive_x_direction",
        "positive_y_direction",
        "pixel_convention",
        "geometry_type",
        "axis_0_domain",
        "row_frame_binding_kind",
        "temporal_domain_id",
        "reference",
        "temporal_evidence",
        "source_sha256",
        "source_sha256_kind",
        "source_mapping_sha256_policy",
        "dtype",
        "rank",
        "trailing_shape",
        "derivation",
        "shape",
        "source_camera_overlay_suitability",
        "canonical_promotion_suitability",
    }
    if set(record) != required:
        raise ValueError("Artifact array numeric-semantics fields are not exact.")
    if (
        record["schema_id"] != UNBOUND_ARTIFACT_NUMERIC_SEMANTICS_SCHEMA
        or record["canonical_binding_status"] != "unbound"
        or record["source_camera_overlay_suitability"] != "unsupported"
        or record["canonical_promotion_suitability"] != "unsupported"
    ):
        raise ValueError("Artifact array numeric-semantics contract is invalid.")
    profile_id = _exact_text(
        record["semantic_profile_id"],
        field_name="semantic_profile_id",
    )
    profile = UNBOUND_NUMERIC_PROFILES.get(profile_id)
    if profile is None:
        raise ValueError("Artifact array semantic_profile_id is not registered.")
    expected_fixed = _profile_fixed_record(profile)
    changed_fixed = tuple(
        name
        for name in _PROFILE_FIXED_FIELDS
        if record.get(name) != expected_fixed[name]
    )
    if changed_fixed:
        raise ValueError(
            "Artifact array fixed semantics do not match its registered profile: "
            f"{changed_fixed!r}."
        )
    reference = record["reference"]
    if not isinstance(reference, Mapping) or set(reference) != {
        "kind",
        "node_path",
        "width",
        "height",
    }:
        raise ValueError("Artifact array reference record is invalid.")
    if reference["kind"] != profile.reference_kind:
        raise ValueError(
            "Artifact array reference kind does not match its registered profile."
        )
    _exact_text(reference["node_path"], field_name="reference.node_path")
    if (
        type(reference["width"]) is not int
        or type(reference["height"]) is not int
        or reference["width"] <= 0
        or reference["height"] <= 0
    ):
        raise ValueError("Artifact array reference extent is invalid.")
    temporal = record["temporal_evidence"]
    derivation = record["derivation"]
    if (
        not isinstance(temporal, Mapping)
        or set(temporal) != {"source_frame_count", "source_mapping_sha256"}
        or not isinstance(derivation, Mapping)
        or set(derivation) != {"operation_id", "parameters"}
    ):
        raise ValueError("Artifact temporal basis or derivation is invalid.")
    source_frame_count = temporal["source_frame_count"]
    if type(source_frame_count) is not int or source_frame_count <= 0:
        raise ValueError("Artifact temporal source_frame_count is invalid.")
    mapping_sha256 = temporal["source_mapping_sha256"]
    if profile.source_mapping_sha256_policy == "required":
        if mapping_sha256 is None:
            raise ValueError(
                "Artifact semantic profile requires source mapping evidence."
            )
        _require_sha256(mapping_sha256, field_name="source_mapping_sha256")
    elif mapping_sha256 is not None:
        raise ValueError("Artifact semantic profile forbids source mapping evidence.")
    if derivation["operation_id"] != profile.derivation_operation_id:
        raise ValueError(
            "Artifact derivation operation does not match its registered profile."
        )
    parameters = _require_json_evidence(
        derivation["parameters"],
        field_name="derivation.parameters",
    )
    if not isinstance(parameters, dict) or parameters != derivation["parameters"]:
        raise ValueError("Artifact derivation parameters are not canonical JSON.")
    _require_sha256(record["source_sha256"], field_name="source_sha256")
    evidence = _array_evidence(array)
    if record["dtype"] != evidence["dtype"] or record["shape"] != evidence["shape"]:
        raise ValueError("Artifact array changed after numeric semantics were stamped.")
    shape = tuple(evidence["shape"])
    if (
        len(shape) != profile.rank
        or shape[1:] != profile.trailing_shape
        or record["rank"] != profile.rank
        or record["trailing_shape"] != list(profile.trailing_shape)
    ):
        raise ValueError("Artifact array structural shape is invalid.")
    if profile.axis_0_domain == "dense_frame_rows" and shape[0] != source_frame_count:
        raise ValueError(
            "Artifact dense-frame array does not match its temporal frame count."
        )
    return record


def publish_empty_artifact_observation_proof(
    run: Any,
    *,
    source_frame_count: int,
    row_array_names: tuple[str, ...],
    full_domain_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Seal one auditable full-domain proof for a genuine zero-row artifact."""

    _manifest_id, manifest = _require_run_manifest(run)
    if type(source_frame_count) is not int or source_frame_count <= 0:
        raise ValueError(
            "Zero-observation proof requires a positive exact frame count."
        )
    if (
        type(row_array_names) is not tuple
        or not row_array_names
        or any(type(name) is not str or not name for name in row_array_names)
        or len(set(row_array_names)) != len(row_array_names)
        or "frame_indices" not in row_array_names
    ):
        raise ValueError(
            "Zero-observation proof requires unique exact row-array names."
        )
    if row_array_names != manifest.row_array_names:
        raise ValueError(
            "Zero-observation row arrays do not match the registered producer manifest."
        )
    evidence = deepcopy(dict(full_domain_evidence))
    if evidence.get("coverage_status") != "full_source_domain_validated":
        raise ValueError(
            "Zero-observation proof requires full_source_domain_validated evidence."
        )
    row_arrays: dict[str, dict[str, Any]] = {}
    for name in row_array_names:
        if name not in run:
            raise ValueError(f"Zero-observation row array {name!r} is missing.")
        array_evidence = _array_evidence(run[name])
        shape = tuple(array_evidence["shape"])
        if not shape or shape[0] != 0:
            raise ValueError(
                f"Zero-observation row array {name!r} does not have zero rows."
            )
        row_arrays[name] = array_evidence
    count_arrays: dict[str, dict[str, Any]] = {}
    for name in manifest.count_array_names:
        if name not in run:
            raise ValueError(f"Zero-observation count array {name!r} is missing.")
        array = run[name]
        values = np.asarray(array[...])
        if (
            values.dtype != np.dtype("int32")
            or values.shape != (source_frame_count,)
            or np.any(values != 0)
        ):
            raise ValueError(
                f"Zero-observation count array {name!r} must be exact int32 zeros "
                "over the full source frame domain."
            )
        count_arrays[name] = _array_evidence(array)
    if count_arrays["frame_counts"] != count_arrays["n_detections"]:
        raise ValueError(
            "Zero-observation frame_counts and n_detections evidence must agree."
        )
    expected_inventory = set(row_array_names) | set(manifest.count_array_names)
    live_names = _child_names(run)
    if live_names is None or set(live_names) != expected_inventory:
        raise ValueError(
            "Zero-observation proof must enumerate the exact live array inventory."
        )
    array_inventory = {
        name: _array_evidence(run[name]) for name in sorted(expected_inventory)
    }
    record = {
        "schema_id": EMPTY_ARTIFACT_OBSERVATION_PROOF_SCHEMA,
        "status": "verified_no_observations",
        "source_frame_count": source_frame_count,
        "observation_row_count": 0,
        "row_arrays": row_arrays,
        "count_arrays": count_arrays,
        "array_inventory": array_inventory,
        "full_domain_evidence": evidence,
    }
    attr_name = EMPTY_ARTIFACT_OBSERVATION_PROOF_ATTR
    digest_name = f"{attr_name}_sha256"
    snapshot = {
        attr_name: deepcopy(run.attrs[attr_name])
        if attr_name in run.attrs
        else _MISSING,
        digest_name: deepcopy(run.attrs[digest_name])
        if digest_name in run.attrs
        else _MISSING,
    }
    try:
        run.attrs[attr_name] = record
        run.attrs[digest_name] = _canonical_sha256(record)
        validate_empty_artifact_observation_proof(run)
    except BaseException:
        _restore(run.attrs, snapshot)
        raise
    return record


def validate_empty_artifact_observation_proof(run: Any) -> dict[str, Any]:
    """Validate the persisted zero-row proof against every live bound array."""

    _manifest_id, manifest = _require_run_manifest(run)
    attr_name = EMPTY_ARTIFACT_OBSERVATION_PROOF_ATTR
    digest_name = f"{attr_name}_sha256"
    record = run.attrs.get(attr_name)
    digest = run.attrs.get(digest_name)
    if not isinstance(record, Mapping) or type(digest) is not str:
        raise ValueError(
            "Zero-row detection artifact lacks its persisted proof/digest."
        )
    normalized = deepcopy(dict(record))
    if _canonical_sha256(normalized) != digest:
        raise ValueError("Zero-row detection artifact proof digest does not match.")
    if (
        normalized.get("schema_id") != EMPTY_ARTIFACT_OBSERVATION_PROOF_SCHEMA
        or normalized.get("status") != "verified_no_observations"
        or normalized.get("observation_row_count") != 0
        or type(normalized.get("source_frame_count")) is not int
        or normalized["source_frame_count"] <= 0
    ):
        raise ValueError("Zero-row detection artifact proof header is invalid.")
    evidence = normalized.get("full_domain_evidence")
    if not isinstance(evidence, Mapping) or evidence.get("coverage_status") != (
        "full_source_domain_validated"
    ):
        raise ValueError("Zero-row detection artifact lacks full-domain evidence.")
    row_arrays = normalized.get("row_arrays")
    if not isinstance(row_arrays, Mapping) or set(row_arrays) != set(
        manifest.row_array_names
    ):
        raise ValueError("Zero-row detection artifact proof has no row-array evidence.")
    for name, expected in row_arrays.items():
        if (
            type(name) is not str
            or name not in run
            or not isinstance(expected, Mapping)
        ):
            raise ValueError(
                "Zero-row detection artifact row-array evidence is invalid."
            )
        observed = _array_evidence(run[name])
        if (
            observed != dict(expected)
            or not observed["shape"]
            or observed["shape"][0] != 0
        ):
            raise ValueError(
                f"Zero-row detection artifact row array {name!r} changed after proof."
            )
    source_frame_count = normalized["source_frame_count"]
    count_arrays = normalized.get("count_arrays")
    if not isinstance(count_arrays, Mapping) or set(count_arrays) != set(
        manifest.count_array_names
    ):
        raise ValueError("Zero-row detection artifact count evidence is incomplete.")
    for name, expected in count_arrays.items():
        if name not in run or not isinstance(expected, Mapping):
            raise ValueError("Zero-row detection artifact count evidence is invalid.")
        values = np.asarray(run[name][...])
        observed = _array_evidence(run[name])
        if (
            observed != dict(expected)
            or values.dtype != np.dtype("int32")
            or values.shape != (source_frame_count,)
            or np.any(values != 0)
        ):
            raise ValueError(
                f"Zero-row detection artifact count array {name!r} changed after proof."
            )
    inventory = normalized.get("array_inventory")
    expected_names = set(row_arrays) | set(count_arrays)
    live_names = _child_names(run)
    if (
        not isinstance(inventory, Mapping)
        or set(inventory) != expected_names
        or live_names is None
        or set(live_names) != expected_names
    ):
        raise ValueError(
            "Zero-row detection artifact live array inventory is not exact."
        )
    for name, expected in inventory.items():
        if not isinstance(expected, Mapping) or _array_evidence(run[name]) != dict(
            expected
        ):
            raise ValueError(
                f"Zero-row detection artifact inventory array {name!r} changed."
            )
    return normalized


def _validate_manifest_bound_semantics(
    run: Any,
    *,
    source_frame_count: int,
    expected_manifest_id: str | None = None,
) -> tuple[
    str,
    UnboundProducerManifest,
    dict[str, dict[str, Any]],
    str | None,
    dict[str, Any],
]:
    manifest_id, manifest = _require_run_manifest(
        run,
        expected_manifest_id=expected_manifest_id,
    )
    _source_evidence, source_evidence_sha256, run_binding = (
        _resolve_run_owned_source_evidence(
            run,
            manifest_id=manifest_id,
            manifest=manifest,
        )
    )
    if run_binding["temporal"]["source_frame_count"] != source_frame_count:
        raise ValueError(
            "Detection artifact source frame count disagrees with run-owned evidence."
        )
    run_mapping = run_binding["mapping"]
    source_mapping_sha256 = (
        run_mapping["sha256"] if isinstance(run_mapping, Mapping) else None
    )
    source_sha256_kinds = {
        UNBOUND_NUMERIC_PROFILES[profile_id].source_sha256_kind
        for _array_name, profile_id in manifest.array_profiles
    }
    if len(source_sha256_kinds) != 1:
        raise RuntimeError("Registered artifact manifest mixes source digest kinds.")
    expected_shared = (
        run_binding["reference"]["kind"],
        run_binding["reference"]["node_path"],
        run_binding["reference"]["width"],
        run_binding["reference"]["height"],
        run_binding["temporal"]["domain_id"],
        run_binding["temporal"]["source_frame_count"],
        source_evidence_sha256,
        next(iter(source_sha256_kinds)),
    )
    profile_by_array = dict(manifest.array_profiles)
    live_names = _child_names(run)
    if live_names is None or set(live_names) != set(profile_by_array):
        raise ValueError(
            "Detection artifact arrays do not match its registered producer manifest."
        )
    semantics_by_array: dict[str, dict[str, Any]] = {}
    mapping_by_array: dict[str, str] = {}
    for name, expected_profile_id in manifest.array_profiles:
        semantics = validate_unbound_artifact_numeric_semantics(run[name])
        if semantics["semantic_profile_id"] != expected_profile_id:
            raise ValueError(
                f"Detection artifact array {name!r} uses profile "
                f"{semantics['semantic_profile_id']!r}, expected "
                f"{expected_profile_id!r} from manifest {manifest_id!r}."
            )
        temporal = semantics["temporal_evidence"]
        if temporal["source_frame_count"] != source_frame_count:
            raise ValueError(
                f"Detection artifact array {name!r} has inconsistent frame count."
            )
        current_shared = (
            semantics["reference"]["kind"],
            semantics["reference"]["node_path"],
            semantics["reference"]["width"],
            semantics["reference"]["height"],
            semantics["temporal_domain_id"],
            temporal["source_frame_count"],
            semantics["source_sha256"],
            semantics["source_sha256_kind"],
        )
        if current_shared != expected_shared:
            raise ValueError(
                "Detection artifact array semantics do not resolve to the exact "
                "run-owned reference, source, or temporal evidence."
            )
        mapping_sha256 = temporal["source_mapping_sha256"]
        if mapping_sha256 is not None:
            mapping_by_array[name] = mapping_sha256
        semantics_by_array[name] = semantics
    expected_mapping_names = set(manifest.source_mapping_array_names)
    if set(mapping_by_array) != expected_mapping_names:
        raise ValueError(
            "Detection artifact source mapping evidence does not match its "
            "manifest-designated arrays."
        )
    mapping_digests = set(mapping_by_array.values())
    expected_mapping_digests = (
        {source_mapping_sha256} if source_mapping_sha256 is not None else set()
    )
    if mapping_digests != expected_mapping_digests:
        raise ValueError(
            "Detection artifact mapping arrays do not resolve to the exact "
            "run-owned source mapping evidence."
        )
    run_evidence = {
        "attr_name": manifest.source_evidence_attr,
        "schema_id": manifest.source_evidence_schema_id,
        "sha256": source_evidence_sha256,
        "binding": run_binding,
    }
    return (
        manifest_id,
        manifest,
        semantics_by_array,
        source_mapping_sha256,
        run_evidence,
    )


def publish_artifact_payload_inventory_seal(
    run: Any,
    *,
    source_frame_count: int,
) -> dict[str, Any]:
    """Seal exact arrays, row cardinality, counts, and unbound semantics."""

    if type(source_frame_count) is not int or source_frame_count <= 0:
        raise ValueError("Artifact inventory requires a positive exact frame count.")
    (
        manifest_id,
        manifest,
        semantics_by_array,
        source_mapping_sha256,
        run_evidence,
    ) = _validate_manifest_bound_semantics(
        run,
        source_frame_count=source_frame_count,
    )
    row_array_names = manifest.row_array_names
    count_array_names = manifest.count_array_names
    expected_names = set(dict(manifest.array_profiles))

    artifact_row_id = np.asarray(run["artifact_row_id"][...])
    if artifact_row_id.ndim != 1 or artifact_row_id.dtype != np.dtype("uint64"):
        raise ValueError("artifact_row_id must be an exact rank-1 uint64 array.")
    row_count = int(artifact_row_id.shape[0])
    if not np.array_equal(
        artifact_row_id,
        np.arange(row_count, dtype=np.uint64),
    ):
        raise ValueError("artifact_row_id must be dense zero-based run-local identity.")

    frame_indices = np.asarray(run["frame_indices"][...])
    if (
        frame_indices.dtype != np.dtype("int32")
        or frame_indices.shape != (row_count,)
        or np.any(frame_indices < 0)
        or np.any(frame_indices >= source_frame_count)
    ):
        raise ValueError(
            "frame_indices must be exact int32 rows inside the source frame domain."
        )

    arrays: dict[str, dict[str, Any]] = {}
    for name in sorted(expected_names):
        node = run[name]
        evidence = _array_evidence(node)
        semantics = semantics_by_array[name]
        semantics_digest = node.attrs[
            f"{UNBOUND_ARTIFACT_NUMERIC_SEMANTICS_ATTR}_sha256"
        ]
        arrays[name] = {
            "payload": evidence,
            "numeric_semantics_sha256": semantics_digest,
            "semantic_profile_id": semantics["semantic_profile_id"],
            "numeric_space_id": semantics["numeric_space_id"],
        }

    for name in row_array_names:
        shape = tuple(arrays[name]["payload"]["shape"])
        if not shape or shape[0] != row_count:
            raise ValueError(
                f"Artifact row array {name!r} does not match row cardinality."
            )

    count_values: dict[str, np.ndarray] = {}
    for name in count_array_names:
        values = np.asarray(run[name][...])
        if values.dtype != np.dtype("int32") or values.shape != (source_frame_count,):
            raise ValueError(
                f"Artifact count array {name!r} must be exact full-domain int32."
            )
        if np.any(values < 0):
            raise ValueError(f"Artifact count array {name!r} cannot be negative.")
        count_values[name] = values
    expected_counts = np.bincount(
        frame_indices.astype(np.int64, copy=False),
        minlength=source_frame_count,
    )
    if (
        not np.array_equal(count_values["frame_counts"], count_values["n_detections"])
        or not np.array_equal(
            count_values["frame_counts"].astype(np.int64, copy=False),
            expected_counts,
        )
        or int(np.sum(count_values["frame_counts"], dtype=np.int64)) != row_count
    ):
        raise ValueError(
            "Artifact count arrays do not exactly match live frame_indices."
        )

    if row_count == 0:
        zero_proof = validate_empty_artifact_observation_proof(run)
        zero_inventory = zero_proof["array_inventory"]
        for name, value in arrays.items():
            if value["payload"] != dict(zero_inventory[name]):
                raise ValueError(
                    "Artifact inventory disagrees with the zero-observation proof."
                )
    elif (
        EMPTY_ARTIFACT_OBSERVATION_PROOF_ATTR in run.attrs
        or f"{EMPTY_ARTIFACT_OBSERVATION_PROOF_ATTR}_sha256" in run.attrs
    ):
        raise ValueError("Nonempty artifact cannot carry a zero-observation proof.")

    record = {
        "schema_id": ARTIFACT_PAYLOAD_INVENTORY_SEAL_SCHEMA,
        "status": "exact_live_payload_validated",
        "unbound_numeric_manifest_id": manifest_id,
        "unbound_numeric_manifest_sha256": run.attrs[
            UNBOUND_NUMERIC_MANIFEST_DIGEST_ATTR
        ],
        "source_frame_count": source_frame_count,
        "source_mapping_sha256": source_mapping_sha256,
        "run_evidence": run_evidence,
        "row_count": row_count,
        "row_array_names": list(row_array_names),
        "count_array_names": list(count_array_names),
        "arrays": arrays,
    }
    attr_name = ARTIFACT_PAYLOAD_INVENTORY_SEAL_ATTR
    digest_name = f"{attr_name}_sha256"
    snapshot = {
        attr_name: deepcopy(run.attrs[attr_name])
        if attr_name in run.attrs
        else _MISSING,
        digest_name: deepcopy(run.attrs[digest_name])
        if digest_name in run.attrs
        else _MISSING,
    }
    try:
        run.attrs[attr_name] = record
        run.attrs[digest_name] = _canonical_sha256(record)
        validate_artifact_payload_inventory_seal(run)
    except BaseException:
        _restore(run.attrs, snapshot)
        raise
    return record


def validate_artifact_payload_inventory_seal(run: Any) -> dict[str, Any]:
    """Freshly validate a strict artifact seal against every live array."""

    attr_name = ARTIFACT_PAYLOAD_INVENTORY_SEAL_ATTR
    digest_name = f"{attr_name}_sha256"
    value = run.attrs.get(attr_name)
    digest = run.attrs.get(digest_name)
    if not isinstance(value, Mapping) or type(digest) is not str:
        raise ValueError("Strict detection artifact lacks its payload inventory seal.")
    record = deepcopy(dict(value))
    if _canonical_sha256(record) != digest:
        raise ValueError("Detection artifact payload inventory digest does not match.")
    if set(record) != {
        "schema_id",
        "status",
        "unbound_numeric_manifest_id",
        "unbound_numeric_manifest_sha256",
        "source_frame_count",
        "source_mapping_sha256",
        "run_evidence",
        "row_count",
        "row_array_names",
        "count_array_names",
        "arrays",
    }:
        raise ValueError("Detection artifact payload inventory fields are not exact.")
    if (
        record["schema_id"] != ARTIFACT_PAYLOAD_INVENTORY_SEAL_SCHEMA
        or record["status"] != "exact_live_payload_validated"
        or type(record["source_frame_count"]) is not int
        or record["source_frame_count"] <= 0
        or type(record["row_count"]) is not int
        or record["row_count"] < 0
    ):
        raise ValueError("Detection artifact payload inventory header is invalid.")
    row_names = record["row_array_names"]
    count_names = record["count_array_names"]
    arrays = record["arrays"]
    (
        _manifest_id,
        manifest,
        semantics_by_array,
        source_mapping_sha256,
        run_evidence,
    ) = _validate_manifest_bound_semantics(
        run,
        source_frame_count=record["source_frame_count"],
        expected_manifest_id=record["unbound_numeric_manifest_id"],
    )
    if (
        record["unbound_numeric_manifest_sha256"]
        != run.attrs.get(UNBOUND_NUMERIC_MANIFEST_DIGEST_ATTR)
        or record["source_mapping_sha256"] != source_mapping_sha256
        or record["run_evidence"] != run_evidence
        or not isinstance(row_names, list)
        or len(set(row_names)) != len(row_names)
        or row_names != list(manifest.row_array_names)
        or count_names != list(manifest.count_array_names)
        or not isinstance(arrays, Mapping)
    ):
        raise ValueError("Detection artifact payload inventory names are invalid.")
    expected_names = set(dict(manifest.array_profiles))
    live_names = _child_names(run)
    if (
        set(arrays) != expected_names
        or live_names is None
        or set(live_names) != expected_names
    ):
        raise ValueError("Detection artifact live inventory changed after sealing.")

    for name, expected in arrays.items():
        if not isinstance(expected, Mapping) or set(expected) != {
            "payload",
            "numeric_semantics_sha256",
            "semantic_profile_id",
            "numeric_space_id",
        }:
            raise ValueError("Detection artifact array evidence is invalid.")
        node = run[name]
        payload = expected["payload"]
        if not isinstance(payload, Mapping) or _array_evidence(node) != dict(payload):
            raise ValueError(
                f"Detection artifact array {name!r} changed after sealing."
            )
        semantics = semantics_by_array[name]
        semantics_digest = node.attrs.get(
            f"{UNBOUND_ARTIFACT_NUMERIC_SEMANTICS_ATTR}_sha256"
        )
        if (
            semantics_digest != expected["numeric_semantics_sha256"]
            or semantics["semantic_profile_id"] != expected["semantic_profile_id"]
            or semantics["semantic_profile_id"] != dict(manifest.array_profiles)[name]
            or semantics["numeric_space_id"] != expected["numeric_space_id"]
        ):
            raise ValueError(
                f"Detection artifact array {name!r} semantics changed after sealing."
            )

    source_frame_count = record["source_frame_count"]
    row_count = record["row_count"]
    artifact_row_id = np.asarray(run["artifact_row_id"][...])
    if (
        artifact_row_id.dtype != np.dtype("uint64")
        or artifact_row_id.shape != (row_count,)
        or not np.array_equal(
            artifact_row_id,
            np.arange(row_count, dtype=np.uint64),
        )
    ):
        raise ValueError("Detection artifact row identity is not dense uint64.")
    frame_indices = np.asarray(run["frame_indices"][...])
    if (
        frame_indices.dtype != np.dtype("int32")
        or frame_indices.shape != (row_count,)
        or np.any(frame_indices < 0)
        or np.any(frame_indices >= source_frame_count)
    ):
        raise ValueError("Detection artifact frame_indices are invalid.")
    for name in row_names:
        if tuple(run[name].shape)[0] != row_count:
            raise ValueError(
                f"Detection artifact row array {name!r} changed cardinality."
            )
    frame_counts = np.asarray(run["frame_counts"][...])
    n_detections = np.asarray(run["n_detections"][...])
    expected_counts = np.bincount(
        frame_indices.astype(np.int64, copy=False),
        minlength=source_frame_count,
    )
    if (
        frame_counts.dtype != np.dtype("int32")
        or n_detections.dtype != np.dtype("int32")
        or frame_counts.shape != (source_frame_count,)
        or n_detections.shape != (source_frame_count,)
        or not np.array_equal(frame_counts, n_detections)
        or not np.array_equal(
            frame_counts.astype(np.int64, copy=False),
            expected_counts,
        )
        or int(np.sum(frame_counts, dtype=np.int64)) != row_count
    ):
        raise ValueError("Detection artifact count arrays are invalid.")
    if row_count == 0:
        zero_proof = validate_empty_artifact_observation_proof(run)
        for name, expected in zero_proof["array_inventory"].items():
            if dict(expected) != arrays[name]["payload"]:
                raise ValueError(
                    "Detection artifact seal disagrees with zero-observation proof."
                )
    elif (
        EMPTY_ARTIFACT_OBSERVATION_PROOF_ATTR in run.attrs
        or f"{EMPTY_ARTIFACT_OBSERVATION_PROOF_ATTR}_sha256" in run.attrs
    ):
        raise ValueError("Nonempty artifact carries a zero-observation proof.")
    return record


def _iter_nodes(node: Any, *, path: str = ""):
    yield path, node
    names = _child_names(node)
    if names is None:
        return
    for name in names:
        child = node[name]
        child_path = f"{path}/{name}" if path else name
        if hasattr(child, "keys"):
            yield from _iter_nodes(child, path=child_path)
        else:
            yield child_path, child


def _require_no_identity_or_coordinate_claims(run: Any) -> None:
    violations: list[str] = []
    for path, node in _iter_nodes(run):
        basename = path.rsplit("/", 1)[-1]
        if basename in _IDENTITY_ARRAY_NAMES:
            violations.append(f"identity array {path!r}")
        attrs = getattr(node, "attrs", None)
        if attrs is None:
            continue
        for name in tuple(attrs.keys()):
            if (
                name in _TRUSTED_NODE_ATTRS
                or name in _TRUSTED_RUN_ATTRS
                or name.startswith("instance_key_")
                or name.startswith("acquisition_frame_mapping")
            ):
                violations.append(f"attribute {name!r} on {path or '<run>'}")
    if violations:
        raise ValueError(
            "Unbound detection artifact carries identity or coordinate claims: "
            f"{violations!r}."
        )


@dataclass
class DetectionProducerAttempt:
    """One unbound-artifact attempt with exact selector rollback."""

    parent: Any
    run: Any
    run_name: str
    owner_token: str
    selector_snapshot: Mapping[str, Any]
    semantic_manifest_id: str
    strict_integrity_required: bool
    _finalized: bool = field(default=False, init=False)

    @classmethod
    def begin(
        cls,
        root: Any,
        *,
        run_name: str,
        output_parent: str,
        selector_eligible: bool,
        coordinate_contract: str,
        stage: str,
        semantic_manifest_id: str,
        strict_integrity_required: bool = True,
    ) -> "DetectionProducerAttempt":
        if type(run_name) is not str or type(output_parent) is not str:
            raise ValueError(
                "Detection run_name and output_parent must be exact strings."
            )
        name = run_name.strip()
        family = output_parent.strip().strip("/")
        if not name or "/" in name or name in {".", ".."}:
            raise ValueError("Detection run_name must be one non-empty path segment.")
        if not family or "/" in family:
            raise ValueError("Detection output_parent must be one path segment.")
        if selector_eligible is not False:
            raise ValueError(
                "DetectionProducerAttempt is artifact-only and rejects every "
                "selector-eligible output; canonical detection publication must "
                "use the staged canonical publisher."
            )
        if strict_integrity_required is not True:
            raise ValueError(
                "Every detection artifact requires strict_integrity_required=True."
            )
        if family != DETECTION_ARTIFACT_RUN_FAMILY:
            raise ValueError(
                "DetectionProducerAttempt may write only to detection_artifact_runs."
            )
        if coordinate_contract != UNBOUND_DETECTION_ARTIFACT_COORDINATE_CONTRACT:
            raise ValueError(
                "DetectionProducerAttempt requires the explicit unbound detection-"
                "artifact coordinate contract."
            )
        resolved_manifest_id = _exact_text(
            semantic_manifest_id,
            field_name="semantic_manifest_id",
        )
        manifest = _resolve_manifest(resolved_manifest_id)
        manifest_digest = _canonical_sha256(
            _manifest_record(resolved_manifest_id, manifest)
        )
        existing_parent = _child(root, family)
        if existing_parent is not None:
            existing_attrs = existing_parent.attrs
            forbidden_selectors = tuple(
                selector for selector in _SELECTOR_ATTRS if selector in existing_attrs
            )
            if forbidden_selectors:
                raise ValueError(
                    "detection_artifact_runs is selector-free; remove or migrate "
                    f"invalid selector attrs before writing: {forbidden_selectors!r}."
                )
            existing_contract = existing_attrs.get(_ARTIFACT_FAMILY_CONTRACT_ATTR)
            if existing_contract not in (None, DETECTION_ARTIFACT_FAMILY_CONTRACT):
                raise ValueError(
                    "Existing detection_artifact_runs has an unsupported family "
                    f"contract: {existing_contract!r}."
                )
            existing_eligibility = existing_attrs.get(_ARTIFACT_FAMILY_ELIGIBILITY_ATTR)
            if existing_eligibility is not None and existing_eligibility is not False:
                raise ValueError(
                    "Existing detection_artifact_runs is not explicitly nonselector."
                )
        if existing_parent is not None and name in existing_parent:
            raise ValueError(
                f"Refusing to replace existing detection output {family}/{name}; "
                "publish a new immutable run name."
            )
        parent_preexisting = existing_parent is not None
        snapshot = (
            _snapshot(existing_parent.attrs)
            if existing_parent is not None
            else _missing_selector_snapshot()
        )
        parent = None
        run = None
        creation_started = False
        owner_token = uuid4().hex
        try:
            parent = require_runs_parent(root, family)
            _require_selector_free_artifact_parent(parent, stamp_missing=True)
            if name in parent:
                raise ValueError(
                    f"Refusing to replace existing detection output {family}/{name}; "
                    "publish a new immutable run name."
                )
            creation_started = True
            sentinel_attrs = {
                _PUBLICATION_OWNER_ATTR: owner_token,
                "stage_selector_eligible": False,
                "coordinate_contract_mode": "setup_incomplete_fail_closed",
                UNBOUND_NUMERIC_MANIFEST_ATTR: resolved_manifest_id,
                UNBOUND_NUMERIC_MANIFEST_DIGEST_ATTR: manifest_digest,
            }
            # Zarr v3 persists group attributes with creation. Rollback may only
            # reopen/delete a child carrying this exact unguessable owner token.
            run = parent.create_group(name, attributes=sentinel_attrs)
            if _owned_child(parent, name, owner_token) is None:
                raise RuntimeError(
                    "Detection artifact child did not persist its atomic ownership "
                    "sentinel."
                )
            mark_run_started(run, run_name=name, stage=stage)
            run_attrs = {
                "output_parent": family,
                "run_group_parent": family,
                "stage_selector_eligible": False,
                "coordinate_contract": coordinate_contract,
                "coordinate_contract_mode": "artifact_unbound",
                "is_detection_artifact": True,
                "artifact_publication_intent": "explicit_artifact_only",
                "detection_artifact_family_contract": (
                    DETECTION_ARTIFACT_FAMILY_CONTRACT
                ),
                "detection_artifact_layout": "detection_artifact_sparse_v1",
                "selector_policy": "never_select_or_promote_v1",
                UNBOUND_NUMERIC_MANIFEST_ATTR: resolved_manifest_id,
                UNBOUND_NUMERIC_MANIFEST_DIGEST_ATTR: manifest_digest,
            }
            run_attrs[_ARTIFACT_INTEGRITY_CONTRACT_ATTR] = (
                STRICT_ARTIFACT_INTEGRITY_CONTRACT
            )
            run.attrs.update(run_attrs)
        except BaseException as exc:
            cleanup_error = _rollback_failed_begin(
                root=root,
                family=family,
                run_name=name,
                parent=parent,
                run=run,
                parent_preexisting=parent_preexisting,
                creation_started=creation_started,
                owner_token=owner_token,
                selector_snapshot=snapshot,
                cause=exc,
            )
            if cleanup_error is not None:
                raise RuntimeError(
                    "Detection producer setup failed and could not be rolled back "
                    f"safely: {cleanup_error}"
                ) from exc
            raise
        assert parent is not None and run is not None
        return cls(
            parent=parent,
            run=run,
            run_name=name,
            owner_token=owner_token,
            selector_snapshot=snapshot,
            semantic_manifest_id=resolved_manifest_id,
            strict_integrity_required=strict_integrity_required,
        )

    @classmethod
    def begin_unbound_artifact(
        cls,
        root: Any,
        *,
        run_name: str,
        semantic_manifest_id: str,
        stage: str = "detection_artifact",
        strict_integrity_required: bool = True,
    ) -> "DetectionProducerAttempt":
        return cls.begin(
            root,
            run_name=run_name,
            output_parent=DETECTION_ARTIFACT_RUN_FAMILY,
            selector_eligible=False,
            coordinate_contract=UNBOUND_DETECTION_ARTIFACT_COORDINATE_CONTRACT,
            stage=stage,
            semantic_manifest_id=semantic_manifest_id,
            strict_integrity_required=strict_integrity_required,
        )

    def complete(
        self,
        *,
        run_provenance: Mapping[str, Any],
    ) -> None:
        if self._finalized:
            return
        owned = _owned_child(self.parent, self.run_name, self.owner_token)
        if owned is None:
            raise RuntimeError(
                "Detection artifact publication lost exact child ownership."
            )
        self.run = owned
        _require_selector_free_artifact_parent(self.parent, stamp_missing=False)
        expected_manifest = _resolve_manifest(self.semantic_manifest_id)
        expected_manifest_digest = _canonical_sha256(
            _manifest_record(self.semantic_manifest_id, expected_manifest)
        )
        expected_attrs = {
            "output_parent": DETECTION_ARTIFACT_RUN_FAMILY,
            "run_group_parent": DETECTION_ARTIFACT_RUN_FAMILY,
            "stage_selector_eligible": False,
            "coordinate_contract": UNBOUND_DETECTION_ARTIFACT_COORDINATE_CONTRACT,
            "coordinate_contract_mode": "artifact_unbound",
            "is_detection_artifact": True,
            "artifact_publication_intent": "explicit_artifact_only",
            "detection_artifact_family_contract": DETECTION_ARTIFACT_FAMILY_CONTRACT,
            "detection_artifact_layout": "detection_artifact_sparse_v1",
            "selector_policy": "never_select_or_promote_v1",
            UNBOUND_NUMERIC_MANIFEST_ATTR: self.semantic_manifest_id,
            UNBOUND_NUMERIC_MANIFEST_DIGEST_ATTR: expected_manifest_digest,
        }
        invalid = tuple(
            name
            for name, expected in expected_attrs.items()
            if self.run.attrs.get(name) != expected
        )
        if invalid:
            raise ValueError(
                "Detection artifact publication invariants changed before completion: "
                f"{invalid!r}."
            )
        integrity_contract = self.run.attrs.get(_ARTIFACT_INTEGRITY_CONTRACT_ATTR)
        if integrity_contract != STRICT_ARTIFACT_INTEGRITY_CONTRACT:
            raise ValueError(
                "Strict detection artifact integrity contract changed before "
                "completion."
            )
        _require_no_identity_or_coordinate_claims(self.run)
        if "frame_indices" not in self.run:
            raise ValueError("Detection artifact is missing frame_indices.")
        row_count = int(self.run["frame_indices"].shape[0])
        if row_count == 0:
            validate_empty_artifact_observation_proof(self.run)
        elif (
            EMPTY_ARTIFACT_OBSERVATION_PROOF_ATTR in self.run.attrs
            or f"{EMPTY_ARTIFACT_OBSERVATION_PROOF_ATTR}_sha256" in self.run.attrs
        ):
            raise ValueError(
                "Nonempty detection artifact cannot carry a zero-observation proof."
            )
        seal = validate_artifact_payload_inventory_seal(self.run)
        if seal["row_count"] != row_count:
            raise ValueError(
                "Detection artifact seal row count disagrees with frame_indices."
            )
        mark_run_complete(
            self.run,
            parent_group=self.parent,
            run_name=self.run_name,
            run_provenance=run_provenance,
        )
        completed = _owned_child(self.parent, self.run_name, self.owner_token)
        if completed is None:
            raise RuntimeError(
                "Completed detection artifact could not be freshly resolved with "
                "its exact publication ownership."
            )
        self.run = completed
        if self.run.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE:
            raise RuntimeError(
                "Freshly resolved detection artifact is not marked complete."
            )
        completed_invalid = tuple(
            name
            for name, expected in expected_attrs.items()
            if self.run.attrs.get(name) != expected
        )
        if completed_invalid:
            raise ValueError(
                "Freshly resolved completed detection artifact changed publication "
                f"invariants: {completed_invalid!r}."
            )
        if (
            self.run.attrs.get(_ARTIFACT_INTEGRITY_CONTRACT_ATTR)
            != STRICT_ARTIFACT_INTEGRITY_CONTRACT
        ):
            raise ValueError(
                "Freshly resolved completed detection artifact changed its strict "
                "integrity contract."
            )
        _require_no_identity_or_coordinate_claims(self.run)
        completed_seal = validate_artifact_payload_inventory_seal(self.run)
        if completed_seal != seal or completed_seal["row_count"] != row_count:
            raise ValueError(
                "Freshly resolved completed detection artifact disagrees with its "
                "pre-completion payload seal."
            )
        if not _matches(self.parent.attrs, self.selector_snapshot):
            _restore(self.parent.attrs, self.selector_snapshot)
            raise RuntimeError(
                "A nonselector detection artifact attempted to mutate stage selectors."
            )
        _require_selector_free_artifact_parent(self.parent, stamp_missing=False)
        if _owned_child(self.parent, self.run_name, self.owner_token) is None:
            raise RuntimeError(
                "Detection artifact publication lost ownership during completion."
            )
        self._finalized = True

    def fail(self, cause: BaseException) -> None:
        if self._finalized:
            return
        owned = _owned_child(self.parent, self.run_name, self.owner_token)
        if owned is None:
            raise RuntimeError(
                "Refusing to roll back a detection artifact child not owned by this "
                "publication attempt."
            ) from cause
        self.run = owned
        rollback_errors: list[BaseException] = []
        try:
            self.run.attrs["stage_selector_eligible"] = False
        except BaseException as exc:  # pragma: no cover - hostile store
            rollback_errors.append(exc)
        try:
            mark_run_failed(
                self.run,
                parent_group=self.parent,
                run_name=self.run_name,
                error=str(cause),
            )
        except BaseException as exc:  # pragma: no cover - hostile store
            rollback_errors.append(exc)
        finally:
            try:
                _restore(self.parent.attrs, self.selector_snapshot)
            except BaseException as exc:  # pragma: no cover - hostile store
                rollback_errors.append(exc)
        try:
            _require_selector_free_artifact_parent(
                self.parent,
                stamp_missing=False,
            )
        except BaseException as exc:  # pragma: no cover - hostile store
            rollback_errors.append(exc)
        if rollback_errors:
            raise RuntimeError(
                "Detection attempt rollback could not restore its exact selector state."
            ) from rollback_errors[0]
        self._finalized = True

    def __enter__(self) -> "DetectionProducerAttempt":
        return self

    def __exit__(self, exc_type: Any, exc: BaseException | None, tb: Any) -> bool:
        if exc is not None:
            self.fail(exc)
        elif not self._finalized:
            error = RuntimeError(
                "Detection attempt exited without explicit completion."
            )
            self.fail(error)
            raise error
        return False


__all__ = [
    "ARTIFACT_PAYLOAD_INVENTORY_SEAL_ATTR",
    "ARTIFACT_PAYLOAD_INVENTORY_SEAL_SCHEMA",
    "DETECTION_ARTIFACT_FAMILY_CONTRACT",
    "DETECTION_ARTIFACT_RUN_FAMILY",
    "DetectionProducerAttempt",
    "EMPTY_ARTIFACT_OBSERVATION_PROOF_ATTR",
    "EMPTY_ARTIFACT_OBSERVATION_PROOF_SCHEMA",
    "STRICT_ARTIFACT_INTEGRITY_CONTRACT",
    "UNBOUND_DETECTION_ARTIFACT_COORDINATE_CONTRACT",
    "UNBOUND_ARTIFACT_RUN_BINDING_KEY",
    "UNBOUND_ARTIFACT_RUN_BINDING_SCHEMA",
    "UNBOUND_ARTIFACT_NUMERIC_SEMANTICS_ATTR",
    "UNBOUND_ARTIFACT_NUMERIC_SEMANTICS_SCHEMA",
    "UNBOUND_NUMERIC_PROFILES",
    "UNBOUND_NUMERIC_MANIFEST_ATTR",
    "UNBOUND_NUMERIC_MANIFEST_DIGEST_ATTR",
    "UNBOUND_NUMERIC_MANIFEST_SCHEMA",
    "UNBOUND_PRODUCER_MANIFESTS",
    "UnboundNumericProfile",
    "UnboundProducerManifest",
    "build_unbound_artifact_run_binding",
    "publish_artifact_payload_inventory_seal",
    "publish_empty_artifact_observation_proof",
    "stamp_unbound_artifact_numeric_semantics",
    "validate_artifact_payload_inventory_seal",
    "validate_empty_artifact_observation_proof",
    "validate_unbound_artifact_numeric_semantics",
]
