#!/usr/bin/env python3
"""
Frame-wise eye angle computation for Palette archives.

This module derives head-relative eye angles, per-eye kinematics, and quality
flags from canonical subject-mask eye geometry and its source keypoint headings.
The results are stored under
``analysis/eye_angle_runs/<run>`` with full provenance metadata so downstream
tools can consume clean, frame-aligned measurements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import dask
from dask import delayed
import numpy as np
import zarr
from rich.console import Console

try:
    from dask.distributed import Client, LocalCluster

    HAVE_DISTRIBUTED = True
except ImportError:  # pragma: no cover - depends on optional dependency
    Client = None  # type: ignore
    LocalCluster = None  # type: ignore
    HAVE_DISTRIBUTED = False

from fisheye.shared.provenance_attrs import (
    build_source_keypoints_attrs,
    resolve_source_keypoints_run,
)
from fisheye.shared.coordinate_frame_record import array_values_sha256
from fisheye.shared.coordinate_identity import (
    INSTANCE_KEY_ARRAY_REF,
    INSTANCE_KEY_MODE,
    OBSERVATION_INSTANCE_DOMAIN,
)
from fisheye.shared.coordinate_record import bind_persisted_coordinate_record
from fisheye.registry.derived_analysis_status import (
    emit_eye_angle_stage_completion,
)
from fisheye.shared.run_lineage_fingerprint import write_best_effort_run_lineage_attrs
from fisheye.shared.run_provenance import build_writer_run_provenance
from fisheye.shared.zarr_run_completion import mark_run_complete, mark_run_started, require_runs_parent
from fisheye.shared.detect_reason_codec import REASON_BYTES_ENCODING, REASON_BYTES_MIN_WIDTH
from fisheye.shared.eye_geometry_source import (
    EYE_GEOMETRY_STAGE_REFINED_SUBJECT,
    EYE_GEOMETRY_STAGE_SUBJECT_SHAPE,
    resolve_eye_geometry_source,
)
from fisheye.shared.keypoint_coordinate_publication import (
    KEYPOINT_LABEL_AUTHORITY_ATTR,
    KEYPOINT_LABEL_AUTHORITY_SCHEMA_ID,
    KEYPOINT_LABEL_AUTHORITY_SCHEMA_VERSION,
    load_persisted_keypoint_coordinate_surfaces,
)
from fisheye.pose.body_frame import (
    BODY_FRAME_COORDINATE_SPACE_ROI,
    BODY_FRAME_SCHEMA_ID,
    BODY_FRAME_SCHEMA_VERSION,
    build_keypoint_body_frame_contract_attrs,
    compute_keypoint_body_frame,
)
from fisheye.pose.schema import (
    resolve_keypoint_labels_from_attrs,
    resolve_required_keypoint_indices_from_attrs,
)
from fisheye.shared.metadata import get_fps
from fisheye.shared.system_metadata import get_git_info
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.shared.eye_angle_schema import (
    CANONICAL_FRAME_ANGLE_CHANNELS,
    CANONICAL_ROI_ANGLE_CHANNELS,
    FRAME_QA_CHANNELS,
    ROI_QA_CHANNELS,
    ROI_VECTOR_CHANNELS,
    EYE_ANGLE_ARRAY_SCHEMA_ATTR,
    EYE_ANGLE_COLUMN_ORDER_PROFILE,
    EYE_ANGLE_COLUMN_ORDER_SCHEMA_ID,
    EYE_ANGLE_LAYOUT_CHOICES,
    EYE_ANGLE_LAYOUT_COMPACT_DENSE_V2,
    EYE_ANGLE_LAYOUT_DEFAULT,
    EYE_ANGLE_LAYOUT_HIERARCHICAL_V1 as _EYE_ANGLE_LAYOUT_HIERARCHICAL_V1,
    EYE_ANGLE_LEGACY_RUN_SCHEMA_VERSION,
    EYE_ANGLE_RUN_SCHEMA_ID,
    EYE_ANGLE_RUN_SCHEMA_VERSION,
    EyeAngleDimensions,
    collect_eye_angle_arrays,
    collect_eye_angle_channel_index_attrs,
    eye_angle_array_schema_manifest,
    eye_angle_channel_index_attrs,
    eye_angle_channel_metadata,
    eye_qa_channel_metadata,
    eye_vector_channel_metadata,
    _formula_for_angle_channel as _schema_formula_for_angle_channel,
    semantic_angle_channel_order as _schema_semantic_angle_channel_order,
    validate_eye_angle_compact_arrays,
    validate_eye_angle_value_aliases,
    canonical_exact_json_bytes,
)
from fisheye.analysis.eye_angle_schema import validate_eye_angle_compact_run
from fisheye.analysis.eye_angle_storage import (
    EYE_ANGLE_ACCESS_AWARE_CANDIDATE_PROFILE_ID,
    EYE_ANGLE_LEGACY_EXPLICIT_STORAGE,
    EYE_ANGLE_STORAGE_CANDIDATE_ATTR,
    EYE_ANGLE_STORAGE_PLAN_ATTR,
    EYE_ANGLE_STORAGE_PROFILE_CHOICES,
    build_eye_angle_candidate_storage_plan,
    create_eye_angle_array_from_entry,
    eye_angle_storage_entries_by_path,
    is_eye_angle_storage_candidate,
)
from fisheye.shared.zarr.analysis_storage_planning import (
    AnalysisArrayStoragePlanReceipt,
    AnalysisStoragePlanReceipt,
)

# Preserve the long-standing module import surface while the implementation
# and exact declarations live in ``eye_angle_schema``.
EYE_ANGLE_LAYOUT_HIERARCHICAL_V1 = _EYE_ANGLE_LAYOUT_HIERARCHICAL_V1
_formula_for_angle_channel = _schema_formula_for_angle_channel

# Reason-code bitmask values (shared across detection- and frame-level QA)
REASON_NONE = np.uint16(0)
REASON_DETECTION_FAILURE = np.uint16(1 << 0)
REASON_HEADING_INVALID = np.uint16(1 << 1)
REASON_LEFT_ELLIPSE_INVALID = np.uint16(1 << 2)
REASON_RIGHT_ELLIPSE_INVALID = np.uint16(1 << 3)
REASON_MULTI_DETECTION = np.uint16(1 << 4)
REASON_NO_DETECTION = np.uint16(1 << 5)

REASON_CODE_MAP = {
    int(REASON_DETECTION_FAILURE): "detection_failure",
    int(REASON_HEADING_INVALID): "heading_invalid",
    int(REASON_LEFT_ELLIPSE_INVALID): "left_ellipse_invalid",
    int(REASON_RIGHT_ELLIPSE_INVALID): "right_ellipse_invalid",
    int(REASON_MULTI_DETECTION): "multiple_detections",
    int(REASON_NO_DETECTION): "no_detection",
}

ELLIPSE_CIRCULARITY_THRESHOLD = 0.95  # reject nearly circular fits that lack a stable major axis
DERIVATIVE_MAX_DT = 0.25  # seconds; ignore large gaps when computing discrete derivatives
ANGLE_SMOOTHING_WINDOW = 7  # frames; moving-average window for smoothed angle outputs
_HEAD_KEYPOINT_LABELS = ("swim_bladder", "eye_left", "eye_right")
SUPPORTED_SCHEDULERS = ("single-threaded", "threads", "processes", "distributed")
EXECUTION_BACKENDS = ("serial_driver", "dask_worker_chunks")
SERIAL_EXECUTION_BACKEND = "serial_driver"
DASK_WORKER_EXECUTION_BACKEND = "dask_worker_chunks"
EYE_ANGLE_OUTPUT_SCHEMA_ID = "analysis.eye_angle_output_schema"
EYE_ANGLE_OUTPUT_SCHEMA_VERSION = 9
EYE_ANGLE_VARIANT_SCHEMA_ID = "analysis.eye_angle_variant_schema"
EYE_ANGLE_VARIANT_SCHEMA_VERSION = 1
EYE_ANGLE_ALGORITHM_CONTRACT_SCHEMA_ID = "analysis.eye_angle_algorithm_contract"
EYE_ANGLE_ALGORITHM_CONTRACT_SCHEMA_VERSION = 1
EYE_ANGLE_METHOD = "ellipse_and_centroid_eye_angles"
EYE_ANGLE_METHOD_VERSION = "eye_angle_analysis.v5"
EYE_ANGLE_ROW_AXIS = "keypoint_detection_rows"
EYE_ANGLE_DENSE_CHUNK_ROWS = 4_096
EYE_ANGLE_DENSE_CHUNK_COLUMNS = 16
MAJOR_AXIS_MARGINAL_DOT_THRESHOLD = 0.1
EYE_ANGLE_STAGED_INPUT_INTEGRITY_SCHEMA_ID = (
    "palette.eye_angle_staged_input_integrity_receipt"
)
EYE_ANGLE_STAGED_INPUT_INTEGRITY_SCHEMA_VERSION = 1
EYE_ANGLE_STAGED_INPUT_INTEGRITY_SCOPE = (
    "materializer_private_exact_worker_input_snapshot_v1"
)
EYE_ANGLE_STAGED_INPUT_CHUNK_SCHEMA_ID = (
    "palette.eye_angle_staged_input_chunk_integrity_receipt"
)
EYE_ANGLE_STAGED_INPUT_CHUNK_SCHEMA_VERSION = 1
EYE_ANGLE_INPUT_PAYLOAD_CANONICALIZATION = (
    "numpy_dtype_shape_c_order_bytes_v1"
)
_EYE_ANGLE_WORKER_LOGICAL_INPUTS = (
    "ellipse_params",
    "ellipse_success",
    "keypoints_roi",
    "detection_success",
    "instance_key",
    "source_acquisition_frame_index",
)
EYE_ANGLE_KEYPOINT_SOURCE_CANONICAL = "canonical_subject_shape_assignment"
EYE_ANGLE_KEYPOINT_SOURCE_REFINED_DIAGNOSTIC = (
    "legacy_unverified_refined_keypoint_diagnostic"
)
EYE_ANGLE_REFINED_DIAGNOSTIC_COORDINATE_CONTRACT = (
    "palette.eye_angles.legacy_refined_keypoint_diagnostic_nonselector.v1"
)
EYE_ANGLE_STAGED_KEYPOINT_AUTHORITY_SCHEMA_ID = (
    "palette.eye_angle_staged_canonical_keypoint_authority"
)
EYE_ANGLE_STAGED_KEYPOINT_AUTHORITY_SCHEMA_VERSION = 1
EYE_ANGLE_STAGED_KEYPOINT_AUTHORITY_SCOPE = (
    "materializer_private_subject_shape_assignment_keypoints_subset_v1"
)

_BASE_ROI_RESULT_FIELDS: tuple[tuple[str, str], ...] = (
    ("left_deg", "left_deg"),
    ("right_deg", "right_deg"),
    ("left_signed_deg", "left_signed_deg"),
    ("right_signed_deg", "right_signed_deg"),
    ("left_major_signed_deg", "left_major_signed_deg"),
    ("right_major_signed_deg", "right_major_signed_deg"),
    ("left_eye_angle_deg", "left_eye_angle_deg"),
    ("right_eye_angle_deg", "right_eye_angle_deg"),
    ("vergence_eye_angle_deg", "vergence_eye_angle_deg"),
    ("vergence_deg", "vergence_deg"),
    ("vergence_signed_deg", "vergence_signed_deg"),
    ("vergence_major_signed_deg", "vergence_major_signed_deg"),
    ("version_deg", "version_deg"),
    ("version_major_deg", "version_major_deg"),
    ("left_minor_signed_deg", "left_minor_signed_deg"),
    ("right_minor_signed_deg", "right_minor_signed_deg"),
    ("vergence_minor_signed_deg", "vergence_minor_signed_deg"),
    ("version_minor_deg", "version_minor_deg"),
    ("left_gaze_deg", "left_gaze_deg"),
    ("right_gaze_deg", "right_gaze_deg"),
    ("left_gaze_signed_deg", "left_gaze_signed_deg"),
    ("right_gaze_signed_deg", "right_gaze_signed_deg"),
    ("vergence_gaze_deg", "vergence_gaze_deg"),
    ("vergence_gaze_signed_deg", "vergence_gaze_signed_deg"),
    ("left_nasal_gaze_deg", "left_nasal_gaze_deg"),
    ("right_nasal_gaze_deg", "right_nasal_gaze_deg"),
    ("mean_eye_vergence_gaze_deg", "mean_eye_vergence_gaze_deg"),
    ("version_gaze_deg", "version_gaze_deg"),
    ("heading_deg", "heading_deg"),
    ("left_centroid_deg", "left_centroid_deg"),
    ("right_centroid_deg", "right_centroid_deg"),
    ("vergence_centroid_deg", "vergence_centroid_deg"),
)

_BASE_QA_RESULT_FIELDS: tuple[tuple[str, str], ...] = (
    ("valid_left", "valid_left"),
    ("valid_right", "valid_right"),
    ("valid_frame", "valid_frame"),
    ("reason_codes", "reason_codes"),
    ("left_major_axis_marginal", "left_major_axis_marginal"),
    ("right_major_axis_marginal", "right_major_axis_marginal"),
    ("major_axis_marginal", "major_axis_marginal"),
)

_BASE_SUPPORT_RESULT_FIELDS: tuple[tuple[str, str], ...] = (
    ("ellipse_major", "ellipse_major"),
    ("ellipse_minor", "ellipse_minor"),
    ("ellipse_ratio", "ellipse_ratio"),
)


def _eye_angle_definition_attrs() -> Dict[str, object]:
    """Return stable metadata definitions for eye-angle output arrays."""
    return {
        "signed_angles": True,
        "signed_angle_convention": "per-eye signed angles are body-frame anatomical-left-positive",
        "canonical_eye_orientation_axis": "ellipse_major",
        "canonical_eye_orientation_arrays": ["left_major_signed_deg", "right_major_signed_deg"],
        "angle_zero": "major axis aligned with body forward axis (0 deg = AP-aligned at rest)",
        "angle_sign_convention": "positive = rotation toward anatomical left",
        "axis_ambiguity_resolution": (
            "ellipse major axis is resolved into the body-frame forward half-plane; "
            "gaze/minor direction is derived by eye-specific 90 degree rotation from that resolved major axis"
        ),
        "major_axis_marginal_definition": (
            f"abs(dot(resolved_major_axis_xy, forward_axis_xy)) < {MAJOR_AXIS_MARGINAL_DOT_THRESHOLD}"
        ),
        "vergence_definition": "undirected_axis_separation(left_signed_deg, right_signed_deg)",
        "vergence_signed_definition": "same as vergence_deg for directionless ellipse axes",
        "version_definition": "0.5*(left_signed_deg + right_signed_deg)",
        "major_signed_angles": True,
        "major_signed_angle_convention": "major-axis signed angles are body-frame anatomical-left-positive",
        "major_vergence_definition": "undirected_axis_separation(left_major_signed_deg, right_major_signed_deg)",
        "major_vergence_signed_definition": "same as vergence_major_signed_deg for directionless ellipse axes",
        "major_version_definition": "0.5*(left_major_signed_deg + right_major_signed_deg)",
        "eye_frame_angles": True,
        "eye_frame_angle_convention": (
            "left/right_eye_angle_deg are eye-frame nasal-positive angles "
            "(Bianco/Engert-lab convention); left_eye_angle_deg = -left_major_signed_deg, "
            "right_eye_angle_deg = +right_major_signed_deg"
        ),
        "vergence_eye_angle_definition": (
            "vergence_eye_angle_deg = left_eye_angle_deg + right_eye_angle_deg "
            "(signed sum of per-eye nasal rotations); positive = converged, "
            "negative = diverged, zero = at rest"
        ),
        "minor_signed_angles": True,
        "minor_signed_angle_convention": (
            "per-eye minor/gaze signed angles are body-frame anatomical-left-positive "
            "and derived from resolved major axes"
        ),
        "minor_vergence_definition": "undirected_axis_separation(left_minor_signed_deg, right_minor_signed_deg)",
        "minor_vergence_signed_definition": "same as vergence_minor_deg for directionless ellipse axes",
        "minor_version_definition": "0.5*(left_minor_signed_deg + right_minor_signed_deg)",
        "preferred_eye_axis": "ellipse_major",
        "preferred_angle_family": "gaze",
        "gaze_angle_source": "ellipse_minor_derived_from_resolved_major_axis",
        "gaze_angle_definition": (
            "left/right_gaze_signed_deg are signed gaze/minor-axis angles derived from the "
            "forward-half-plane resolved major axis; left eye = major + 90 deg, right eye = major - 90 deg"
        ),
        "gaze_vector_definition": (
            "left/right_gaze_xy are ROI/image-space unit vectors for the same derived gaze directions"
        ),
        "gaze_vergence_definition": "undirected_axis_separation(left_gaze_signed_deg, right_gaze_signed_deg)",
        "gaze_vergence_signed_definition": "same as vergence_gaze_deg for directionless ellipse axes",
        "gaze_total_vergence_definition": (
            "vergence_gaze_deg retains the v3-compatible undirected axis separation; "
            "under expected outward anatomical eye-axis polarity it equals "
            "left_nasal_gaze_deg + right_nasal_gaze_deg"
        ),
        "mean_eye_vergence_gaze_definition": "0.5 * (left_nasal_gaze_deg + right_nasal_gaze_deg)",
        "nasal_gaze_definition": "90 - abs(outward_from_midline_gaze_axis_angle_deg)",
        "beast_comparable_eye_vergence": "mean_eye_vergence_gaze_deg",
        "gaze_version_definition": "0.5*(left_gaze_signed_deg + right_gaze_signed_deg)",
        "body_frame_schema_id": BODY_FRAME_SCHEMA_ID,
        "body_frame_schema_version": BODY_FRAME_SCHEMA_VERSION,
        "body_frame_estimator": "keypoint_head_axis",
    }


def _source_geometry_kind(stage_group: str) -> str:
    return {
        EYE_GEOMETRY_STAGE_SUBJECT_SHAPE: "subject_shape_eye_geometry",
        EYE_GEOMETRY_STAGE_REFINED_SUBJECT: "refined_subject_eye_geometry",
    }.get(str(stage_group), "unknown_eye_geometry")


def _eye_angle_variant_schema() -> Dict[str, object]:
    """Return a UI-friendly registry of eye-angle representations.

    This is intentionally descriptive rather than executable. UIs can use the
    representation metadata to group selectable traces without hardcoding every
    field-name convention.
    """

    representations: Dict[str, Dict[str, object]] = {
        "major": {
            "display_name": "Canonical major-axis orientation",
            "role": "canonical_geometry",
            "axis": "ellipse_major",
            "coordinate_frame": "fish_body_frame",
            "units": "deg",
            "sign_convention": "positive_anatomical_left",
            "derived_from": "resolved_ellipse_major_axis",
            "preferred_for": ["geometry", "provenance", "derived_conventions"],
            "primary_roi_fields": ["left_major_signed_deg", "right_major_signed_deg"],
            "aggregate_roi_fields": ["vergence_major_signed_deg", "version_major_deg"],
            "default_plot_fields": ["left_major_signed_deg", "right_major_signed_deg"],
            "frame_fields": ["vergence_major_signed_deg", "version_major_deg"],
        },
        "eye_frame": {
            "display_name": "Bianco/Engert eye-frame angles",
            "role": "biological_presentation",
            "axis": "ellipse_major",
            "coordinate_frame": "per_eye_nasal_positive",
            "units": "deg",
            "sign_convention": "positive_nasal_for_each_eye",
            "derived_from": "major",
            "left_transform": "-left_major_signed_deg",
            "right_transform": "right_major_signed_deg",
            "preferred_for": ["paper_style_per_eye_angles", "signed_convergence"],
            "primary_roi_fields": ["left_eye_angle_deg", "right_eye_angle_deg"],
            "aggregate_roi_fields": ["vergence_eye_angle_deg"],
            "default_plot_fields": [
                "left_eye_angle_deg_smoothed",
                "right_eye_angle_deg_smoothed",
                "vergence_eye_angle_deg_smoothed",
            ],
            "frame_fields": ["left_eye_angle_deg", "right_eye_angle_deg", "vergence_eye_angle_deg"],
        },
        "gaze": {
            "display_name": "Gaze direction",
            "role": "gaze_direction",
            "axis": "ellipse_minor_derived_from_resolved_major_axis",
            "coordinate_frame": "fish_body_frame",
            "units": "deg",
            "sign_convention": "positive_anatomical_left",
            "derived_from": "major",
            "left_transform": "wrap(left_major_signed_deg + 90 deg)",
            "right_transform": "wrap(right_major_signed_deg - 90 deg)",
            "preferred_for": ["ray_drawing", "body_frame_gaze_direction", "gaze_qc"],
            "primary_roi_fields": ["left_gaze_signed_deg", "right_gaze_signed_deg"],
            "unsigned_roi_fields": ["left_gaze_deg", "right_gaze_deg"],
            "vector_roi_fields": ["left_gaze_xy", "right_gaze_xy"],
            "aggregate_roi_fields": ["vergence_gaze_deg", "version_gaze_deg"],
            "default_plot_fields": ["left_gaze_signed_deg_smoothed", "right_gaze_signed_deg_smoothed"],
            "frame_fields": ["left_gaze_signed_deg", "right_gaze_signed_deg", "vergence_gaze_deg"],
        },
        "nasal_gaze": {
            "display_name": "BEAST/Johnson nasal-gaze convergence",
            "role": "compatibility_analysis",
            "axis": "gaze_nasal_transform",
            "coordinate_frame": "per_eye_nasal_positive",
            "units": "deg",
            "sign_convention": "larger_is_more_nasal",
            "derived_from": "gaze",
            "left_transform": "90 - abs(left_gaze_signed_deg)",
            "right_transform": "90 - abs(right_gaze_signed_deg)",
            "preferred_for": ["beast_johnson_mean_eye_vergence_plots"],
            "primary_roi_fields": ["left_nasal_gaze_deg", "right_nasal_gaze_deg"],
            "aggregate_roi_fields": ["mean_eye_vergence_gaze_deg"],
            "default_plot_fields": [
                "left_nasal_gaze_deg_smoothed",
                "right_nasal_gaze_deg_smoothed",
                "mean_eye_vergence_gaze_deg_smoothed",
            ],
            "frame_fields": ["left_nasal_gaze_deg", "right_nasal_gaze_deg", "mean_eye_vergence_gaze_deg"],
        },
        "centroid": {
            "display_name": "Centroid-position diagnostics",
            "role": "diagnostic_pose_context",
            "axis": "eye_centroid_position",
            "coordinate_frame": "fish_body_frame",
            "units": "deg",
            "sign_convention": "positive_anatomical_left",
            "derived_from": "eye_centroid_positions",
            "preferred_for": ["pose_diagnostics", "covariates"],
            "primary_roi_fields": ["left_centroid_deg", "right_centroid_deg"],
            "aggregate_roi_fields": ["vergence_centroid_deg"],
            "default_plot_fields": ["left_centroid_deg_smoothed", "right_centroid_deg_smoothed"],
            "frame_fields": ["left_centroid_deg", "right_centroid_deg", "vergence_centroid_deg"],
        },
        "legacy": {
            "display_name": "Legacy compatibility aliases",
            "role": "compatibility_alias",
            "axis": "mixed_legacy",
            "coordinate_frame": "fish_body_frame",
            "units": "deg",
            "sign_convention": "see_alias_target",
            "derived_from": "major_or_gaze",
            "preferred_for": ["old_readers_only"],
            "primary_roi_fields": [
                "left_deg",
                "right_deg",
                "left_signed_deg",
                "right_signed_deg",
                "left_minor_signed_deg",
                "right_minor_signed_deg",
            ],
            "aggregate_roi_fields": [
                "vergence_deg",
                "vergence_signed_deg",
                "version_deg",
                "vergence_minor_signed_deg",
                "version_minor_deg",
            ],
            "alias_targets": {
                "left_signed_deg": "left_major_signed_deg",
                "right_signed_deg": "right_major_signed_deg",
                "left_minor_signed_deg": "left_gaze_signed_deg",
                "right_minor_signed_deg": "right_gaze_signed_deg",
                "vergence_minor_signed_deg": "vergence_gaze_deg",
                "version_minor_deg": "version_gaze_deg",
            },
        },
    }

    fields: Dict[str, Dict[str, object]] = {}
    for key, representation in representations.items():
        for group_name in (
            "primary_roi_fields",
            "aggregate_roi_fields",
            "unsigned_roi_fields",
            "vector_roi_fields",
        ):
            for field in representation.get(group_name, []):
                fields[str(field)] = {
                    "representation": key,
                    "field_role": group_name.removesuffix("_fields"),
                    "display_name": representation["display_name"],
                    "units": representation.get("units"),
                }
        for field in representation.get("default_plot_fields", []):
            fields.setdefault(str(field), {
                "representation": key,
                "field_role": "default_plot",
                "display_name": representation["display_name"],
                "units": representation.get("units"),
            })
            fields[str(field)]["default_plot"] = True

    return {
        "schema_id": EYE_ANGLE_VARIANT_SCHEMA_ID,
        "schema_version": EYE_ANGLE_VARIANT_SCHEMA_VERSION,
        "purpose": "classify eye-angle arrays into UI-selectable representations",
        "default_representation": "eye_frame",
        "representation_order": ["eye_frame", "gaze", "nasal_gaze", "major", "centroid", "legacy"],
        "representations": representations,
        "fields": fields,
    }


def _eye_angle_output_schema() -> Dict[str, object]:
    """Return a machine-readable schema for eye-angle output arrays."""

    roi_angle_outputs = [
        "left_deg",
        "right_deg",
        "left_signed_deg",
        "right_signed_deg",
        "left_major_signed_deg",
        "right_major_signed_deg",
        "left_eye_angle_deg",
        "right_eye_angle_deg",
        "vergence_eye_angle_deg",
        "vergence_deg",
        "vergence_signed_deg",
        "vergence_major_signed_deg",
        "version_deg",
        "version_major_deg",
        "left_minor_signed_deg",
        "right_minor_signed_deg",
        "vergence_minor_signed_deg",
        "version_minor_deg",
        "left_gaze_deg",
        "right_gaze_deg",
        "left_gaze_signed_deg",
        "right_gaze_signed_deg",
        "vergence_gaze_deg",
        "vergence_gaze_signed_deg",
        "left_nasal_gaze_deg",
        "right_nasal_gaze_deg",
        "mean_eye_vergence_gaze_deg",
        "version_gaze_deg",
        "heading_deg",
        "left_centroid_deg",
        "right_centroid_deg",
        "vergence_centroid_deg",
    ]
    derivative_outputs = [
        "left_speed_deg_s",
        "right_speed_deg_s",
        "vergence_speed_deg_s",
        "vergence_signed_speed_deg_s",
        "version_speed_deg_s",
        "left_gaze_speed_deg_s",
        "right_gaze_speed_deg_s",
        "vergence_gaze_speed_deg_s",
        "vergence_gaze_signed_speed_deg_s",
        "version_gaze_speed_deg_s",
        "mean_eye_vergence_gaze_speed_deg_s",
        "left_accel_deg_s2",
        "right_accel_deg_s2",
        "vergence_accel_deg_s2",
        "vergence_signed_accel_deg_s2",
        "version_accel_deg_s2",
        "left_gaze_accel_deg_s2",
        "right_gaze_accel_deg_s2",
        "vergence_gaze_accel_deg_s2",
        "vergence_gaze_signed_accel_deg_s2",
        "version_gaze_accel_deg_s2",
        "mean_eye_vergence_gaze_accel_deg_s2",
    ]
    support_outputs = [
        {"name": "instance_key", "row_axis": "roi", "value_kind": "observation_instance_key"},
        {
            "name": "source_acquisition_frame_index",
            "row_axis": "roi",
            "value_kind": "source_acquisition_frame_index",
        },
        {
            "name": "frame_indices",
            "row_axis": "roi",
            "value_kind": "source_acquisition_frame_index",
            "compatibility_alias_of": "support/source_acquisition_frame_index",
            "values_must_equal_canonical": True,
        },
        {"name": "time_seconds", "row_axis": "roi", "units": "s"},
        {"name": "ellipse_major", "row_axis": "roi", "units": "px"},
        {"name": "ellipse_minor", "row_axis": "roi", "units": "px"},
        {"name": "ellipse_ratio", "row_axis": "roi", "value_kind": "ratio"},
        {"name": "body_frame/origin_xy", "row_axis": "roi", "units": "px"},
        {"name": "body_frame/forward_axis_xy", "row_axis": "roi", "value_kind": "unit_vector_xy"},
        {"name": "body_frame/left_axis_xy", "row_axis": "roi", "value_kind": "unit_vector_xy"},
        {"name": "body_frame/heading_deg", "row_axis": "roi", "units": "deg"},
        {"name": "body_frame/valid", "row_axis": "roi", "value_kind": "bool"},
        {"name": "body_frame/failure_reason_bytes", "row_axis": "roi", "value_kind": "reason_tag"},
        {"name": "frame_time_seconds", "row_axis": "frame", "units": "s", "optional": False},
    ]
    qa_outputs = [
        "valid_left",
        "valid_right",
        "valid_frame",
        "reason_codes",
        "left_major_axis_marginal",
        "right_major_axis_marginal",
        "major_axis_marginal",
    ]
    frame_outputs = [
        "left_deg",
        "right_deg",
        "vergence_deg",
        "vergence_signed_deg",
        "vergence_major_signed_deg",
        "left_eye_angle_deg",
        "right_eye_angle_deg",
        "vergence_eye_angle_deg",
        "version_deg",
        "version_major_deg",
        "vergence_minor_signed_deg",
        "version_minor_deg",
        "left_gaze_deg",
        "right_gaze_deg",
        "left_gaze_signed_deg",
        "right_gaze_signed_deg",
        "vergence_gaze_deg",
        "vergence_gaze_signed_deg",
        "left_nasal_gaze_deg",
        "right_nasal_gaze_deg",
        "mean_eye_vergence_gaze_deg",
        "version_gaze_deg",
        "left_centroid_deg",
        "right_centroid_deg",
        "vergence_centroid_deg",
    ]
    return {
        "schema_id": EYE_ANGLE_OUTPUT_SCHEMA_ID,
        "schema_version": EYE_ANGLE_OUTPUT_SCHEMA_VERSION,
        "algorithm_contract": {
            "schema_id": EYE_ANGLE_ALGORITHM_CONTRACT_SCHEMA_ID,
            "schema_version": EYE_ANGLE_ALGORITHM_CONTRACT_SCHEMA_VERSION,
            "run_attr": "eye_angle_algorithm_contract",
        },
        "variant_schema": _eye_angle_variant_schema(),
        "row_axes": {
            "roi": EYE_ANGLE_ROW_AXIS,
            "frame": "video_frame_rows",
        },
        "groups": {
            "angles/roi": {
                "row_axis": "roi",
                "units": "deg",
                "base_outputs": roi_angle_outputs,
                "vector_outputs": [
                    {"name": "left_gaze_xy", "shape": ["N", 2], "value_kind": "unit_vector_xy_roi"},
                    {"name": "right_gaze_xy", "shape": ["N", 2], "value_kind": "unit_vector_xy_roi"},
                ],
                "smoothed_suffix": "_smoothed",
                "delta_suffix": "_delta_deg",
                "delta_smoothed_suffix": "_delta_deg_smoothed",
                "derivative_outputs": derivative_outputs,
            },
            "angles/frame": {
                "row_axis": "frame",
                "units": "deg",
                "base_outputs": frame_outputs,
                "smoothed_suffix": "_smoothed",
                "delta_suffix": "_delta_deg",
                "delta_smoothed_suffix": "_delta_deg_smoothed",
            },
            "qa/roi": {
                "row_axis": "roi",
                "outputs": qa_outputs,
            },
            "qa/frame": {
                "row_axis": "frame",
                "outputs": ["valid_frame", "reason_codes", "major_axis_marginal"],
            },
            "support": {
                "row_axis": "mixed",
                "outputs": support_outputs,
            },
        },
        "angle_units": "degrees",
        "time_units": "seconds",
        "temporal_operators": {
            "smoothing": "nan_aware_centered_boxcar_finite_count_normalized",
            "delta": "absolute_adjacent_finite_difference",
            "derivative": "backward_difference_to_previous_valid_sample",
        },
        "signed_angle_convention": "per-eye signed angles are body-frame anatomical-left-positive",
        "canonical_eye_orientation_axis": "ellipse_major",
        "canonical_eye_orientation_arrays": ["left_major_signed_deg", "right_major_signed_deg"],
        "angle_zero": "major axis aligned with body forward axis (0 deg = AP-aligned at rest)",
        "angle_sign_convention": "positive = rotation toward anatomical left",
        "axis_ambiguity_resolution": (
            "ellipse major axis is resolved into the body-frame forward half-plane; "
            "gaze/minor direction is derived by eye-specific 90 degree rotation from that resolved major axis"
        ),
        "vergence_signed_definition": "same as vergence_deg for directionless ellipse axes",
        "version_definition": "0.5*(left_signed_deg + right_signed_deg)",
        "major_vergence_definition": "undirected_axis_separation(left_major_signed_deg, right_major_signed_deg)",
        "major_version_definition": "0.5*(left_major_signed_deg + right_major_signed_deg)",
        "eye_frame_angles": True,
        "eye_frame_angle_convention": (
            "left/right_eye_angle_deg are eye-frame nasal-positive angles "
            "(Bianco/Engert-lab convention); left_eye_angle_deg = -left_major_signed_deg, "
            "right_eye_angle_deg = +right_major_signed_deg"
        ),
        "vergence_eye_angle_definition": (
            "vergence_eye_angle_deg = left_eye_angle_deg + right_eye_angle_deg "
            "(signed sum of per-eye nasal rotations); positive = converged, "
            "negative = diverged, zero = at rest"
        ),
        "preferred_angle_family": "gaze",
        "preferred_eye_axis": "ellipse_major",
        "gaze_angle_source": "ellipse_minor_derived_from_resolved_major_axis",
        "gaze_angle_definition": (
            "left/right_gaze_signed_deg are signed gaze/minor-axis angles derived from the "
            "forward-half-plane resolved major axis; left eye = major + 90 deg, right eye = major - 90 deg"
        ),
        "gaze_vector_definition": "left/right_gaze_xy are ROI/image-space unit vectors",
        "gaze_vergence_signed_definition": "same as vergence_gaze_deg for directionless ellipse axes",
        "gaze_vergence_definition": "undirected_axis_separation(left_gaze_signed_deg, right_gaze_signed_deg)",
        "gaze_total_vergence_definition": (
            "vergence_gaze_deg retains the v3-compatible undirected axis separation; "
            "under expected outward anatomical eye-axis polarity it equals "
            "left_nasal_gaze_deg + right_nasal_gaze_deg"
        ),
        "mean_eye_vergence_gaze_definition": "0.5 * (left_nasal_gaze_deg + right_nasal_gaze_deg)",
        "nasal_gaze_definition": "90 - abs(outward_from_midline_gaze_axis_angle_deg)",
        "beast_comparable_eye_vergence": "mean_eye_vergence_gaze_deg",
        "gaze_version_definition": "0.5*(left_gaze_signed_deg + right_gaze_signed_deg)",
        "centroid_angle_definition": "atan2(rotated_eye_vector_y, rotated_eye_vector_x) in fish frame",
        "centroid_vergence_definition": "abs(left_centroid_deg) + abs(right_centroid_deg)",
        "body_frame_schema_id": BODY_FRAME_SCHEMA_ID,
        "body_frame_schema_version": BODY_FRAME_SCHEMA_VERSION,
        "body_frame_estimator": "keypoint_head_axis",
        "body_frame_group": "support/body_frame",
        "compatibility_aliases": {
            "angles/roi/heading_deg": {
                "canonical_path": "support/body_frame/heading_deg",
                "values_must_equal_canonical": True,
                "authority": "compatibility_alias_only",
            }
        },
        "qa_reason_codes_attr": "reason_code_map",
    }


@dataclass
class EyeAngleResults:
    """Container for detection-level outputs."""

    left_deg: np.ndarray
    right_deg: np.ndarray
    left_signed_deg: np.ndarray
    right_signed_deg: np.ndarray
    left_major_signed_deg: np.ndarray
    right_major_signed_deg: np.ndarray
    left_eye_angle_deg: np.ndarray
    right_eye_angle_deg: np.ndarray
    vergence_eye_angle_deg: np.ndarray
    left_minor_signed_deg: np.ndarray
    right_minor_signed_deg: np.ndarray
    left_gaze_xy: np.ndarray
    right_gaze_xy: np.ndarray
    left_gaze_deg: np.ndarray
    right_gaze_deg: np.ndarray
    left_gaze_signed_deg: np.ndarray
    right_gaze_signed_deg: np.ndarray
    left_nasal_gaze_deg: np.ndarray
    right_nasal_gaze_deg: np.ndarray
    mean_eye_vergence_gaze_deg: np.ndarray
    vergence_deg: np.ndarray
    vergence_signed_deg: np.ndarray
    vergence_major_signed_deg: np.ndarray
    vergence_minor_signed_deg: np.ndarray
    vergence_gaze_deg: np.ndarray
    vergence_gaze_signed_deg: np.ndarray
    version_deg: np.ndarray
    version_major_deg: np.ndarray
    version_minor_deg: np.ndarray
    version_gaze_deg: np.ndarray
    ellipse_major: np.ndarray
    ellipse_minor: np.ndarray
    ellipse_ratio: np.ndarray
    valid_left: np.ndarray
    valid_right: np.ndarray
    valid_frame: np.ndarray
    reason_codes: np.ndarray
    left_major_axis_marginal: np.ndarray
    right_major_axis_marginal: np.ndarray
    major_axis_marginal: np.ndarray
    heading_deg: np.ndarray
    body_frame_origin_xy: np.ndarray
    body_frame_forward_axis_xy: np.ndarray
    body_frame_left_axis_xy: np.ndarray
    body_frame_valid: np.ndarray
    body_frame_failure_reason_bytes: np.ndarray
    # Centroid-based eye-position angles (auxiliary pose context)
    left_centroid_deg: np.ndarray
    right_centroid_deg: np.ndarray
    vergence_centroid_deg: np.ndarray


@dataclass
class EyeAngleInputContext:
    """Resolved zarr inputs for one eye-angle run."""

    eye_geometry: Any
    kp_group: zarr.Group
    kp_group_path: str
    source_kp_group: Optional[zarr.Group]
    source_kp_run_name: Optional[str]
    source_kp_group_path: Optional[str]
    detection_success_source: zarr.Group
    detection_success_key: str
    detection_success_path: str
    frame_indices_source: zarr.Group
    frame_indices_key: str
    frame_indices_path: str
    instance_key_source: Optional[zarr.Group]
    instance_key_key: Optional[str]
    instance_key_path: Optional[str]
    keypoint_run_name: str
    keypoint_indices: Dict[str, int]
    keypoint_labels: tuple[str, ...]
    keypoint_source_mode: str
    source_total_frames: Optional[int]
    canonical_keypoint_surfaces: Any = None
    canonical_keypoint_authority: Optional[Mapping[str, Any]] = None


@dataclass(frozen=True)
class _EyeAngleChunkInputSnapshot:
    """Owned worker inputs that are verified before any scientific computation."""

    ellipse_params: np.ndarray
    ellipse_success: np.ndarray
    keypoints_roi: np.ndarray
    detection_success: np.ndarray
    instance_key: np.ndarray
    source_acquisition_frame_index: np.ndarray


_SOURCE_CONTRACT_ATTRS = (
    "schema_id",
    "schema_version",
    "method",
    "method_version",
    "palette_run_completion_status",
    "source_fingerprint",
    "source_lineage_hash",
    "lineage_hash",
    "fingerprint_status",
    "git_commit",
    "git_dirty",
    "created_at_utc",
    "completed_at_utc",
)


def _source_group_contract(group: Optional[zarr.Group], *, path: Optional[str]) -> Dict[str, object]:
    """Capture stable identity, implementation, and lineage fields for one input run."""

    if group is None or not path:
        return {"path": path, "available": False}
    attrs = group.attrs
    contract: Dict[str, object] = {
        "path": str(path),
        "available": True,
    }
    for name in _SOURCE_CONTRACT_ATTRS:
        if name in attrs:
            contract[name] = attrs[name]
    provenance = attrs.get("provenance")
    if isinstance(provenance, str):
        try:
            provenance = json.loads(provenance)
        except json.JSONDecodeError:
            provenance = None
    if isinstance(provenance, Mapping):
        git = provenance.get("git")
        if isinstance(git, Mapping):
            contract["provenance_git"] = dict(git)
        for name in ("script", "stage", "command"):
            if name in provenance:
                contract[f"provenance_{name}"] = provenance[name]
    return contract


def _eye_geometry_component_contracts(context: EyeAngleInputContext) -> list[Dict[str, object]]:
    """Describe the exact arrays and upstream ellipse estimator for each eye."""

    geometry = context.eye_geometry
    components: list[Dict[str, object]] = []
    for component in ("eye_left", "eye_right"):
        if geometry.stage_group == EYE_GEOMETRY_STAGE_SUBJECT_SHAPE:
            relative_group = f"components/{component}"
        else:
            relative_group = f"components/{component}/geometry"
        component_group = geometry.group.get(relative_group)
        attrs = dict(component_group.attrs) if isinstance(component_group, zarr.Group) else {}
        components.append(
            {
                "component": component,
                "group_path": f"{geometry.group_path}/{relative_group}",
                "ellipse_params_path": (
                    f"{geometry.group_path}/{relative_group}/ellipse_params"
                ),
                "ellipse_success_path": (
                    f"{geometry.group_path}/{relative_group}/ellipse_success"
                ),
                "ellipse_source_contract": {
                    key: attrs[key]
                    for key in (
                        "ellipse_method",
                        "geometry_schema_id",
                        "geometry_method",
                        "source_mask_component",
                    )
                    if key in attrs
                },
            }
        )
    return components


def _eye_angle_source_contracts(context: EyeAngleInputContext) -> Dict[str, object]:
    """Return the complete resolved source identity used by one eye-angle run."""

    geometry = context.eye_geometry
    source_authority = getattr(geometry, "source_authority", None)
    return {
        "eye_geometry": {
            **_source_group_contract(geometry.group, path=geometry.group_path),
            "stage_group": geometry.stage_group,
            "run_name": geometry.run_name,
            "geometry_kind": _source_geometry_kind(geometry.stage_group),
            "source_subject_shape_run": geometry.source_subject_shape_run,
            "source_refined_subject_masks_run": geometry.source_refined_subject_run,
            "source_refined_eye_run": geometry.source_refined_eye_run,
            # This location-independent, self-digested receipt is identical for
            # the authoritative publication and its explicitly staged subset.
            # Keep execution authority mode out of the scientific source
            # contract so planning and staged-compute hashes remain identical.
            **(
                {"source_authority": _canonical_json_copy(source_authority)}
                if isinstance(source_authority, Mapping)
                else {}
            ),
            "components": _eye_geometry_component_contracts(context),
        },
        "keypoints": {
            **_source_group_contract(
                context.kp_group,
                path=context.kp_group_path,
            ),
            "run_name": context.keypoint_run_name,
            "source_mode": context.keypoint_source_mode,
            **(
                {
                    "canonical_keypoint_authority": _canonical_json_copy(
                        context.canonical_keypoint_authority
                    )
                }
                if isinstance(context.canonical_keypoint_authority, Mapping)
                else {}
            ),
        },
        "diagnostic_base_keypoints": {
            **_source_group_contract(
                context.source_kp_group,
                path=context.source_kp_group_path,
            ),
            "run_name": context.source_kp_run_name,
        },
        "resolved_arrays": {
            "keypoints_roi": f"{context.kp_group_path}/keypoints_roi",
            "detection_success": context.detection_success_path,
            "instance_key": context.instance_key_path,
            "source_acquisition_frame_index": context.frame_indices_path,
        },
    }


def _build_eye_angle_algorithm_contract(
    *,
    component_sources: Sequence[Mapping[str, object]],
    keypoint_indices: Mapping[str, int],
    fps: Optional[float],
    fps_source: str,
    smoothing_window_requested: int,
    smoothing_window_source: str,
    detection_smoothing_window: int,
    frame_smoothing_window: int,
) -> Dict[str, object]:
    """Describe the exact scientific transformations used for eye-angle outputs."""

    return {
        "schema_id": EYE_ANGLE_ALGORITHM_CONTRACT_SCHEMA_ID,
        "schema_version": EYE_ANGLE_ALGORITHM_CONTRACT_SCHEMA_VERSION,
        "method": EYE_ANGLE_METHOD,
        "method_version": EYE_ANGLE_METHOD_VERSION,
        "source_contracts_attr": "eye_angle_source_contracts",
        "ellipse_input": {
            "parameter_order": [
                "center_x_px",
                "center_y_px",
                "major_axis_length_px",
                "minor_axis_length_px",
                "major_axis_angle_deg",
            ],
            "parameter_normalization": (
                "cv2.fitEllipse axes reordered so major >= minor and major-axis angle "
                "normalized to [0, 180) degrees"
            ),
            "validity_array": "ellipse_success",
            "finite_positive_axis_requirement": True,
            "circularity_ratio_formula": "minor_axis_length_px / major_axis_length_px",
            "circularity_reject_condition": (
                f"ellipse_ratio > {ELLIPSE_CIRCULARITY_THRESHOLD}"
            ),
            "circularity_threshold": float(ELLIPSE_CIRCULARITY_THRESHOLD),
            "component_sources": _canonical_json_copy(list(component_sources)),
        },
        "body_frame": {
            "schema_id": BODY_FRAME_SCHEMA_ID,
            "schema_version": BODY_FRAME_SCHEMA_VERSION,
            "estimator": "keypoint_head_axis",
            "coordinate_space": BODY_FRAME_COORDINATE_SPACE_ROI,
            "required_keypoint_labels": list(_HEAD_KEYPOINT_LABELS),
            "resolved_keypoint_indices": {
                key: int(value) for key, value in keypoint_indices.items()
            },
            "origin_formula": "0.5 * (eye_left_xy + eye_right_xy)",
            "forward_axis_formula": (
                "unit(origin_xy - swim_bladder_xy)"
            ),
            "left_axis_resolution": (
                "choose the forward-axis perpendicular whose dot product with "
                "unit(eye_left_xy - eye_right_xy) is nonnegative"
            ),
            "signed_angle_formula": (
                "degrees(atan2(dot(vector_xy, left_axis_xy), "
                "dot(vector_xy, forward_axis_xy)))"
            ),
            "signed_angle_convention": "positive_anatomical_left",
        },
        "major_axis_resolution": {
            "input_axis_is_directionless": True,
            "half_turn_normalization": "major_axis_angle_rad modulo pi",
            "forward_half_plane_rule": (
                "multiply ellipse major unit vector by -1 when its dot product "
                "with body forward_axis_xy is negative"
            ),
            "marginal_condition": (
                f"abs(dot(resolved_major_axis_xy, forward_axis_xy)) < "
                f"{MAJOR_AXIS_MARGINAL_DOT_THRESHOLD}"
            ),
            "marginal_dot_threshold": float(MAJOR_AXIS_MARGINAL_DOT_THRESHOLD),
        },
        "angle_families": {
            "canonical_major": (
                "left/right_major_signed_deg = body-frame signed angle of the "
                "forward-half-plane resolved ellipse major axis"
            ),
            "eye_frame": {
                "left_eye_angle_deg": "wrap_signed(-left_major_signed_deg)",
                "right_eye_angle_deg": "wrap_signed(right_major_signed_deg)",
                "vergence_eye_angle_deg": (
                    "left_eye_angle_deg + right_eye_angle_deg"
                ),
                "convention": "per-eye nasal-positive Bianco/Engert",
            },
            "gaze": {
                "left_gaze_axis": "resolved_left_major_axis rotated +90 deg in body frame",
                "right_gaze_axis": "resolved_right_major_axis rotated -90 deg in body frame",
                "signed_angle_wrap_range_deg": "[-180, 180)",
                "directional_assumption": (
                    "gaze direction inherits the forward-half-plane major-axis "
                    "resolution and eye-specific 90-degree rotation"
                ),
            },
            "undirected_vergence": (
                "minimum separation of the two directionless axes after wrapping "
                "the absolute signed-angle difference through 360 and 180 degrees"
            ),
            "nasal_gaze": "90 - abs(gaze_signed_deg) per eye",
            "mean_eye_vergence_gaze_deg": (
                "0.5 * (left_nasal_gaze_deg + right_nasal_gaze_deg)"
            ),
            "version": "0.5 * (left_signed_deg + right_signed_deg)",
        },
        "temporal_sampling": {
            "fps": float(fps) if fps else None,
            "fps_source": fps_source,
            "time_seconds_formula": (
                "source_acquisition_frame_index / fps when fps is available"
            ),
        },
        "smoothing": {
            "method": "nan_aware_centered_boxcar_finite_count_normalized",
            "implementation": "numpy.convolve(mode='same')",
            "edge_policy": "partial centered window normalized by finite sample count",
            "missing_value_policy": "ignore NaN; output NaN only when finite count is zero",
            "requested_window": int(smoothing_window_requested),
            "requested_window_source": smoothing_window_source,
            "window_resolution": (
                "cap at axis length, decrement even windows to odd, disable below 3"
            ),
            "effective_detection_row_window": (
                int(detection_smoothing_window) if detection_smoothing_window else None
            ),
            "effective_frame_window": (
                int(frame_smoothing_window) if frame_smoothing_window else None
            ),
        },
        "delta": {
            "method": "absolute_adjacent_finite_difference",
            "formula": "abs(value[row] - value[row - 1])",
            "first_row": "NaN",
            "missing_value_policy": "NaN unless both adjacent values are finite",
            "time_normalized": False,
        },
        "derivative": {
            "method": "backward_difference_to_previous_valid_sample",
            "formula": "(value[current] - value[previous_valid]) / dt",
            "maximum_dt_seconds": float(DERIVATIVE_MAX_DT),
            "gap_policy": "NaN when dt <= 0 or dt exceeds maximum_dt_seconds",
            "acceleration": "apply the same derivative operator to angular speed",
        },
        "frame_projection": {
            "source_axis": EYE_ANGLE_ROW_AXIS,
            "target_axis": "video_frame_rows",
            "unique_detection_rule": (
                "copy a detection row only when exactly one row maps to the frame"
            ),
            "zero_detection_rule": "leave values NaN and set no_detection QA bit",
            "multiple_detection_rule": (
                "leave values NaN and set multiple_detections QA bit"
            ),
        },
    }


def _eye_angle_algorithm_contract(
    context: EyeAngleInputContext,
    *,
    fps: Optional[float],
    fps_source: str,
    smoothing_window_requested: int,
    smoothing_window_source: str,
    detection_smoothing_window: int,
    frame_smoothing_window: int,
) -> Dict[str, object]:
    """Describe the exact scientific transformations used for eye-angle outputs."""

    return _build_eye_angle_algorithm_contract(
        component_sources=_eye_geometry_component_contracts(context),
        keypoint_indices=context.keypoint_indices,
        fps=fps,
        fps_source=fps_source,
        smoothing_window_requested=smoothing_window_requested,
        smoothing_window_source=smoothing_window_source,
        detection_smoothing_window=detection_smoothing_window,
        frame_smoothing_window=frame_smoothing_window,
    )


def expected_eye_angle_algorithm_contract_from_run_attrs(
    attrs: Mapping[str, Any],
) -> Dict[str, object]:
    """Reconstruct the only valid algorithm contract for one compact-v7 run."""

    source_contracts = attrs.get("eye_angle_source_contracts")
    if type(source_contracts) is not dict:
        raise ValueError("eye_angle_source_contracts must be one exact JSON object.")
    eye_geometry = source_contracts.get("eye_geometry")
    if type(eye_geometry) is not dict:
        raise ValueError("eye_angle_source_contracts.eye_geometry must be an object.")
    component_sources = eye_geometry.get("components")
    if type(component_sources) is not list or any(
        type(component) is not dict for component in component_sources
    ):
        raise ValueError(
            "eye_angle_source_contracts.eye_geometry.components must be a JSON object list."
        )
    keypoint_indices = attrs.get("resolved_head_keypoint_indices")
    if (
        type(keypoint_indices) is not dict
        or set(keypoint_indices) != set(_HEAD_KEYPOINT_LABELS)
        or any(type(value) is not int or value < 0 for value in keypoint_indices.values())
    ):
        raise ValueError(
            "resolved_head_keypoint_indices must exactly map the three semantic labels "
            "to nonnegative integers."
        )
    fps = attrs.get("fps")
    if type(fps) is not float or not np.isfinite(fps) or fps <= 0.0:
        raise ValueError("fps must be one exact positive finite JSON float.")
    fps_source = attrs.get("fps_source")
    smoothing_source = attrs.get("angle_smoothing_window_source")
    if type(fps_source) is not str or not fps_source:
        raise ValueError("fps_source must be one exact nonempty string.")
    if type(smoothing_source) is not str or not smoothing_source:
        raise ValueError(
            "angle_smoothing_window_source must be one exact nonempty string."
        )

    def exact_window(name: str, *, required: bool) -> int:
        value = attrs.get(name)
        if not required and value is None:
            return 0
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be one exact integer in its valid range.")
        return value

    return _build_eye_angle_algorithm_contract(
        component_sources=component_sources,
        keypoint_indices=keypoint_indices,
        fps=fps,
        fps_source=fps_source,
        smoothing_window_requested=exact_window(
            "angle_smoothing_window_requested",
            required=True,
        ),
        smoothing_window_source=smoothing_source,
        detection_smoothing_window=exact_window(
            "angle_smoothing_window_detections",
            required=False,
        ),
        frame_smoothing_window=exact_window(
            "angle_smoothing_window_frames",
            required=False,
        ),
    )


def validate_eye_angle_persisted_contract_manifests(
    attrs: Mapping[str, Any],
) -> tuple[str, ...]:
    """Deeply validate static and reconstructed compact-v7 contract manifests."""

    errors: list[str] = []
    expected_static = {
        "eye_angle_output_schema": _eye_angle_output_schema(),
        "eye_angle_variant_schema": _eye_angle_variant_schema(),
    }
    for attr_name, expected in expected_static.items():
        try:
            matches = canonical_exact_json_bytes(
                attrs.get(attr_name),
                path=f"$.{attr_name}",
            ) == canonical_exact_json_bytes(expected)
        except (TypeError, ValueError):
            matches = False
        if not matches:
            errors.append(f"{attr_name} must exactly equal its executable contract")
    try:
        expected_algorithm = expected_eye_angle_algorithm_contract_from_run_attrs(attrs)
        algorithm_matches = canonical_exact_json_bytes(
            attrs.get("eye_angle_algorithm_contract"),
            path="$.eye_angle_algorithm_contract",
        ) == canonical_exact_json_bytes(expected_algorithm)
    except (TypeError, ValueError) as exc:
        errors.append(f"cannot reconstruct eye_angle_algorithm_contract: {exc}")
    else:
        if not algorithm_matches:
            errors.append(
                "eye_angle_algorithm_contract must exactly equal the reconstructed executable contract"
            )
    exact_scalars = {
        "method": EYE_ANGLE_METHOD,
        "method_version": EYE_ANGLE_METHOD_VERSION,
        "row_axis": EYE_ANGLE_ROW_AXIS,
    }
    for attr_name, expected in exact_scalars.items():
        observed = attrs.get(attr_name)
        if type(observed) is not str or observed != expected:
            errors.append(f"{attr_name} must be exact string {expected!r}")
    return tuple(errors)


def _normalize_scheduler(value: str) -> str:
    scheduler = str(value).strip().lower().replace("_", "-")
    aliases = {
        "single": "single-threaded",
        "single_threaded": "single-threaded",
        "thread": "threads",
        "process": "processes",
        "local-cluster": "distributed",
        "local_cluster": "distributed",
    }
    scheduler = aliases.get(scheduler, scheduler)
    if scheduler not in SUPPORTED_SCHEDULERS:
        raise argparse.ArgumentTypeError(
            f"scheduler must be one of {', '.join(SUPPORTED_SCHEDULERS)}; got {value!r}."
        )
    return scheduler


def _normalize_execution_backend(value: str) -> str:
    backend = str(value).strip().lower().replace("-", "_")
    aliases = {
        "serial": SERIAL_EXECUTION_BACKEND,
        "driver": SERIAL_EXECUTION_BACKEND,
        "dask": DASK_WORKER_EXECUTION_BACKEND,
        "dask_chunks": DASK_WORKER_EXECUTION_BACKEND,
    }
    backend = aliases.get(backend, backend)
    if backend not in EXECUTION_BACKENDS:
        raise argparse.ArgumentTypeError(f"execution_backend must be one of {EXECUTION_BACKENDS}; got {value!r}.")
    return backend


def _row_chunks(total_rows: int, chunk_size: int) -> list[tuple[int, int]]:
    total = max(0, int(total_rows))
    chunk = max(1, int(chunk_size))
    return [(start, min(total, start + chunk)) for start in range(0, total, chunk)]


def _to_half_turn(angle_rad: np.ndarray) -> np.ndarray:
    """Map angles into [0, π) so 180° flips of the major axis are treated identically."""
    return np.mod(angle_rad, np.pi)


def _signed_angle_from_body_axes(
    vectors_xy: np.ndarray,
    forward_axis_xy: np.ndarray,
    left_axis_xy: np.ndarray,
) -> np.ndarray:
    """Return anatomical-left-positive signed angles against body axes."""
    forward = np.einsum("ij,ij->i", vectors_xy, forward_axis_xy)
    left = np.einsum("ij,ij->i", vectors_xy, left_axis_xy)
    return np.rad2deg(np.arctan2(left, forward)).astype(np.float32, copy=False)


def _wrap_signed_degrees(values: np.ndarray) -> np.ndarray:
    """Wrap degrees into [-180, 180)."""
    wrapped = (np.asarray(values, dtype=np.float64) + 180.0) % 360.0 - 180.0
    return wrapped.astype(np.float32, copy=False)


def _rotate_body_frame_90(
    vectors_xy: np.ndarray,
    forward_axis_xy: np.ndarray,
    left_axis_xy: np.ndarray,
    *,
    direction: int,
) -> np.ndarray:
    """Rotate ROI/image-space vectors by +/-90 degrees in body-frame coordinates."""
    forward_component = np.einsum("ij,ij->i", vectors_xy, forward_axis_xy)
    left_component = np.einsum("ij,ij->i", vectors_xy, left_axis_xy)
    if int(direction) >= 0:
        rotated_forward = -left_component
        rotated_left = forward_component
    else:
        rotated_forward = left_component
        rotated_left = -forward_component
    rotated = rotated_forward[:, None] * forward_axis_xy + rotated_left[:, None] * left_axis_xy
    return rotated.astype(np.float64, copy=False)


def _undirected_axis_separation_deg(left_signed_deg: np.ndarray, right_signed_deg: np.ndarray) -> np.ndarray:
    """Return the smaller angle between two directionless eye axes.

    OpenCV ellipse axes have a 180-degree ambiguity. After each eye axis is
    oriented into the forward half-plane, the biologically useful vergence is
    the smaller separation between the two undirected axis lines, not the raw
    directed left/right difference.
    """

    left = np.asarray(left_signed_deg, dtype=np.float64)
    right = np.asarray(right_signed_deg, dtype=np.float64)
    diff = np.abs(left - right)
    diff = np.mod(diff, 360.0)
    diff = np.where(diff > 180.0, 360.0 - diff, diff)
    return np.minimum(diff, 180.0 - diff).astype(np.float32, copy=False)


def _resolve_smoothing_window(length: int, desired: int) -> int:
    """Return an odd window length that fits within the sequence, else 0 for no smoothing."""
    if length <= 0:
        return 0
    window = min(desired, length)
    if window < 3:
        return 0
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return 0
    return window


def _smooth_signal(values: np.ndarray, window: int) -> np.ndarray:
    """Apply a NaN-aware moving average to 1D data."""
    if window <= 2:
        return np.array(values, copy=True)
    kernel = np.ones(window, dtype=np.float32)
    finite_mask = np.isfinite(values)
    if not np.any(finite_mask):
        return np.full_like(values, np.nan)
    sums = np.convolve(np.nan_to_num(values, nan=0.0), kernel, mode="same")
    counts = np.convolve(finite_mask.astype(np.float32), kernel, mode="same")
    smoothed = np.full_like(values, np.nan)
    valid = counts > 0
    smoothed[valid] = sums[valid] / counts[valid]
    return smoothed


def _compute_delta(values: np.ndarray) -> np.ndarray:
    """Compute absolute frame-to-frame differences, preserving NaNs."""
    delta = np.full_like(values, np.nan)
    if values.size > 1:
        prev = values[:-1]
        curr = values[1:]
        mask = np.isfinite(prev) & np.isfinite(curr)
        diffs = np.abs(curr - prev)
        out_slice = delta[1:]
        out_slice[mask] = diffs[mask]
        delta[1:] = out_slice
    return delta


def _process_chunk(
    ellipse_params: np.ndarray,
    ellipse_success: np.ndarray,
    keypoints_roi: np.ndarray,
    detection_success: np.ndarray,
    *,
    keypoint_indices: Dict[str, int],
) -> EyeAngleResults:
    """Process a chunk of detections into eye angles and QA flags."""
    chunk_len = ellipse_params.shape[0]

    left_angles = np.full(chunk_len, np.nan, dtype=np.float32)
    right_angles = np.full(chunk_len, np.nan, dtype=np.float32)
    vergence = np.full(chunk_len, np.nan, dtype=np.float32)
    left_signed = np.full(chunk_len, np.nan, dtype=np.float32)
    right_signed = np.full(chunk_len, np.nan, dtype=np.float32)
    left_major_signed = np.full(chunk_len, np.nan, dtype=np.float32)
    right_major_signed = np.full(chunk_len, np.nan, dtype=np.float32)
    left_eye_angle = np.full(chunk_len, np.nan, dtype=np.float32)
    right_eye_angle = np.full(chunk_len, np.nan, dtype=np.float32)
    vergence_eye_angle = np.full(chunk_len, np.nan, dtype=np.float32)
    vergence_signed = np.full(chunk_len, np.nan, dtype=np.float32)
    vergence_major_signed = np.full(chunk_len, np.nan, dtype=np.float32)
    version = np.full(chunk_len, np.nan, dtype=np.float32)
    version_major = np.full(chunk_len, np.nan, dtype=np.float32)
    left_minor_signed = np.full(chunk_len, np.nan, dtype=np.float32)
    right_minor_signed = np.full(chunk_len, np.nan, dtype=np.float32)
    left_gaze_xy = np.full((chunk_len, 2), np.nan, dtype=np.float32)
    right_gaze_xy = np.full((chunk_len, 2), np.nan, dtype=np.float32)
    vergence_minor_signed = np.full(chunk_len, np.nan, dtype=np.float32)
    version_minor = np.full(chunk_len, np.nan, dtype=np.float32)
    ellipse_major = np.full(chunk_len, np.nan, dtype=np.float32)
    ellipse_minor = np.full(chunk_len, np.nan, dtype=np.float32)
    ellipse_ratio = np.full(chunk_len, np.nan, dtype=np.float32)
    valid_left = np.zeros(chunk_len, dtype=bool)
    valid_right = np.zeros(chunk_len, dtype=bool)
    valid_frame = np.zeros(chunk_len, dtype=bool)
    reason_codes = np.zeros(chunk_len, dtype=np.uint16)
    left_major_axis_marginal = np.zeros(chunk_len, dtype=bool)
    right_major_axis_marginal = np.zeros(chunk_len, dtype=bool)

    bladder = keypoints_roi[:, int(keypoint_indices["swim_bladder"]), :]
    eye_left_kp = keypoints_roi[:, int(keypoint_indices["eye_left"]), :]
    eye_right_kp = keypoints_roi[:, int(keypoint_indices["eye_right"]), :]
    body_frame = compute_keypoint_body_frame(
        keypoints_roi,
        keypoint_indices=keypoint_indices,
        detection_success=detection_success,
    )
    heading_out = body_frame.heading_deg.astype(np.float64, copy=True)
    body_frame_valid = body_frame.valid

    reason_codes[~detection_success] |= REASON_DETECTION_FAILURE
    reason_codes[~body_frame_valid] |= REASON_HEADING_INVALID

    # ---------- Centroid-based eye-position angles (auxiliary pose context) ----------
    # Measures eye position angle in fish-frame coordinates.
    # Paper method: vergence = |theta_L| + |theta_R|
    left_centroid = np.full(chunk_len, np.nan, dtype=np.float32)
    right_centroid = np.full(chunk_len, np.nan, dtype=np.float32)
    vergence_centroid = np.full(chunk_len, np.nan, dtype=np.float32)

    # Paper head center: mean of the 3 ROI keypoints
    head_center = (bladder + eye_left_kp + eye_right_kp) / 3.0

    centroid_mask = (
        body_frame_valid
        & np.all(np.isfinite(head_center), axis=1)
        & np.all(np.isfinite(eye_left_kp), axis=1)
        & np.all(np.isfinite(eye_right_kp), axis=1)
    )

    if np.any(centroid_mask):
        cidxs = np.where(centroid_mask)[0]

        # Vectors from head center to each eye in image coords
        vL = eye_left_kp[cidxs] - head_center[cidxs]
        vR = eye_right_kp[cidxs] - head_center[cidxs]

        theta_L = _signed_angle_from_body_axes(
            vL,
            body_frame.forward_axis_xy[cidxs].astype(np.float64, copy=False),
            body_frame.left_axis_xy[cidxs].astype(np.float64, copy=False),
        )
        theta_R = _signed_angle_from_body_axes(
            vR,
            body_frame.forward_axis_xy[cidxs].astype(np.float64, copy=False),
            body_frame.left_axis_xy[cidxs].astype(np.float64, copy=False),
        )

        left_centroid[cidxs] = theta_L
        right_centroid[cidxs] = theta_R
        vergence_centroid[cidxs] = np.abs(theta_L) + np.abs(theta_R)

    for (
        eye_idx,
        target_array,
        valid_array,
        signed_array,
        major_array,
        gaze_vector_array,
        minor_array,
        marginal_array,
        fail_bit,
    ) in (
        (
            0,
            left_angles,
            valid_left,
            left_signed,
            left_major_signed,
            left_gaze_xy,
            left_minor_signed,
            left_major_axis_marginal,
            REASON_LEFT_ELLIPSE_INVALID,
        ),
        (
            1,
            right_angles,
            valid_right,
            right_signed,
            right_major_signed,
            right_gaze_xy,
            right_minor_signed,
            right_major_axis_marginal,
            REASON_RIGHT_ELLIPSE_INVALID,
        ),
    ):
        ellipse_ok = ellipse_success[:, eye_idx]
        eye_params = ellipse_params[:, eye_idx, :]
        angle_deg = eye_params[:, 4]
        major = eye_params[:, 2]
        minor = eye_params[:, 3]

        finite_mask = (
            ellipse_ok
            & np.isfinite(angle_deg)
            & np.isfinite(major)
            & np.isfinite(minor)
            & (major > 0)
            & (minor > 0)
        )

        ratio = np.zeros_like(major, dtype=np.float64)
        ratio_mask = finite_mask & (major > 0)
        ratio[ratio_mask] = minor[ratio_mask] / major[ratio_mask]
        circular_mask = ratio_mask & (ratio > ELLIPSE_CIRCULARITY_THRESHOLD)
        finite_mask &= ~circular_mask
        if np.any(circular_mask):
            reason_codes[circular_mask] |= fail_bit

        combined_mask = finite_mask & body_frame_valid
        reason_codes[~finite_mask] |= fail_bit

        if np.any(combined_mask):
            idxs = np.where(combined_mask)[0]
            # alpha_eye: major-axis angle (radians) in image coordinates; 0 rad along +x, CCW positive
            alpha_eye = np.deg2rad(angle_deg[idxs]).astype(np.float64)
            alpha_eye = _to_half_turn(alpha_eye)

            forward_axis = body_frame.forward_axis_xy[idxs].astype(np.float64, copy=False)
            left_axis = body_frame.left_axis_xy[idxs].astype(np.float64, copy=False)

            axis_major = np.stack([np.cos(alpha_eye), np.sin(alpha_eye)], axis=1)
            sign_major = np.where(np.einsum("ij,ij->i", axis_major, forward_axis) >= 0.0, 1.0, -1.0)
            axis_major_aligned = axis_major * sign_major[:, None]
            major_forward_dot = np.einsum("ij,ij->i", axis_major_aligned, forward_axis)
            signed_major = _signed_angle_from_body_axes(
                axis_major_aligned,
                forward_axis,
                left_axis,
            )

            target_array[idxs] = np.abs(signed_major)
            signed_array[idxs] = signed_major
            major_array[idxs] = signed_major
            marginal_array[idxs] = np.abs(major_forward_dot) < MAJOR_AXIS_MARGINAL_DOT_THRESHOLD

            gaze_xy = _rotate_body_frame_90(
                axis_major_aligned,
                forward_axis,
                left_axis,
                direction=1 if eye_idx == 0 else -1,
            )
            signed_gaze = _signed_angle_from_body_axes(gaze_xy, forward_axis, left_axis)
            signed_gaze = _wrap_signed_degrees(signed_gaze)

            gaze_vector_array[idxs] = gaze_xy.astype(np.float32, copy=False)
            minor_array[idxs] = signed_gaze

            ellipse_major[idxs] = major[idxs].astype(np.float32, copy=False)
            ellipse_minor[idxs] = minor[idxs].astype(np.float32, copy=False)
            # ratio stored post-thresholding to allow post-hoc tuning
            ellipse_ratio[idxs] = (minor[idxs] / major[idxs]).astype(np.float32, copy=False)
            valid_array[combined_mask] = True

        # Ensure invalid entries remain NaN
        target_array[~valid_array] = np.nan
        signed_array[~valid_array] = np.nan
        major_array[~valid_array] = np.nan
        gaze_vector_array[~valid_array, :] = np.nan
        minor_array[~valid_array] = np.nan
        marginal_array[~valid_array] = False

    left_signed[~valid_left] = np.nan
    right_signed[~valid_right] = np.nan
    left_major_signed[~valid_left] = np.nan
    right_major_signed[~valid_right] = np.nan
    left_minor_signed[~valid_left] = np.nan
    right_minor_signed[~valid_right] = np.nan
    left_gaze_xy[~valid_left, :] = np.nan
    right_gaze_xy[~valid_right, :] = np.nan

    valid_frame[:] = valid_left & valid_right & detection_success
    major_axis_marginal = (left_major_axis_marginal | right_major_axis_marginal) & (valid_left | valid_right)

    if np.any(valid_left):
        left_eye_angle[valid_left] = _wrap_signed_degrees(-left_major_signed[valid_left])
    if np.any(valid_right):
        right_eye_angle[valid_right] = _wrap_signed_degrees(right_major_signed[valid_right])
    # Bianco/Engert-style nasal-positive angles match nasal_gaze in the usual
    # regime, but are derived directly from the canonical major-axis AP angle.
    if np.any(valid_frame):
        vergence_eye_angle[valid_frame] = (
            left_eye_angle[valid_frame] + right_eye_angle[valid_frame]
        ).astype(np.float32, copy=False)

    mask = valid_frame
    if np.any(mask):
        # Body-frame signed eye angles are anatomical-left-positive. Ellipse
        # axes are directionless, so vergence is the smaller angle between the
        # two undirected eye-axis lines rather than the raw directed delta.
        left_body = left_signed[mask]
        right_body = right_signed[mask]
        left_major_body = left_major_signed[mask]
        right_major_body = right_major_signed[mask]
        left_minor_body = left_minor_signed[mask]
        right_minor_body = right_minor_signed[mask]

        vergence_signed_vals = _undirected_axis_separation_deg(left_body, right_body)
        vergence[mask] = vergence_signed_vals
        vergence_signed[mask] = vergence_signed_vals
        version[mask] = 0.5 * (left_body + right_body)

        vergence_major_signed_vals = _undirected_axis_separation_deg(left_major_body, right_major_body)
        vergence_major_signed[mask] = vergence_major_signed_vals
        version_major[mask] = 0.5 * (left_major_body + right_major_body)

        vergence_minor_signed_vals = _undirected_axis_separation_deg(left_minor_body, right_minor_body)
        vergence_minor_signed[mask] = vergence_minor_signed_vals
        version_minor[mask] = 0.5 * (left_minor_body + right_minor_body)

    left_nasal_gaze = 90.0 - np.abs(left_minor_signed)
    right_nasal_gaze = 90.0 - np.abs(right_minor_signed)
    left_nasal_gaze[~valid_left] = np.nan
    right_nasal_gaze[~valid_right] = np.nan
    mean_eye_vergence_gaze = np.full(chunk_len, np.nan, dtype=np.float32)
    if np.any(valid_frame):
        mean_eye_vergence_gaze[valid_frame] = (
            0.5 * (left_nasal_gaze[valid_frame] + right_nasal_gaze[valid_frame])
        ).astype(np.float32, copy=False)

    return EyeAngleResults(
        left_deg=left_angles,
        right_deg=right_angles,
        left_signed_deg=left_signed,
        right_signed_deg=right_signed,
        left_major_signed_deg=left_major_signed,
        right_major_signed_deg=right_major_signed,
        left_eye_angle_deg=left_eye_angle,
        right_eye_angle_deg=right_eye_angle,
        vergence_eye_angle_deg=vergence_eye_angle,
        left_minor_signed_deg=left_minor_signed,
        right_minor_signed_deg=right_minor_signed,
        left_gaze_xy=left_gaze_xy,
        right_gaze_xy=right_gaze_xy,
        left_gaze_deg=np.abs(left_minor_signed),
        right_gaze_deg=np.abs(right_minor_signed),
        left_gaze_signed_deg=left_minor_signed,
        right_gaze_signed_deg=right_minor_signed,
        left_nasal_gaze_deg=left_nasal_gaze.astype(np.float32, copy=False),
        right_nasal_gaze_deg=right_nasal_gaze.astype(np.float32, copy=False),
        mean_eye_vergence_gaze_deg=mean_eye_vergence_gaze,
        vergence_deg=vergence,
        vergence_signed_deg=vergence_signed,
        vergence_major_signed_deg=vergence_major_signed,
        vergence_minor_signed_deg=vergence_minor_signed,
        vergence_gaze_deg=vergence_minor_signed,
        vergence_gaze_signed_deg=vergence_minor_signed,
        version_deg=version,
        version_major_deg=version_major,
        version_minor_deg=version_minor,
        version_gaze_deg=version_minor,
        ellipse_major=ellipse_major,
        ellipse_minor=ellipse_minor,
        ellipse_ratio=ellipse_ratio,
        valid_left=valid_left,
        valid_right=valid_right,
        valid_frame=valid_frame,
        reason_codes=reason_codes,
        left_major_axis_marginal=left_major_axis_marginal,
        right_major_axis_marginal=right_major_axis_marginal,
        major_axis_marginal=major_axis_marginal,
        heading_deg=heading_out.astype(np.float32, copy=False),
        body_frame_origin_xy=body_frame.origin_xy,
        body_frame_forward_axis_xy=body_frame.forward_axis_xy,
        body_frame_left_axis_xy=body_frame.left_axis_xy,
        body_frame_valid=body_frame.valid,
        body_frame_failure_reason_bytes=body_frame.failure_reason_bytes,
        left_centroid_deg=left_centroid,
        right_centroid_deg=right_centroid,
        vergence_centroid_deg=vergence_centroid,
    )


def _compute_derivative(
    values: np.ndarray,
    time_seconds: np.ndarray,
    valid_mask: np.ndarray,
    max_dt: Optional[float] = None,
) -> np.ndarray:
    """Backward difference using the previous valid sample."""
    derivative = np.full(values.shape, np.nan, dtype=np.float32)
    valid_indices = np.where(valid_mask & np.isfinite(values) & np.isfinite(time_seconds))[0]
    if valid_indices.size < 2:
        return derivative

    prev_idx = valid_indices[0]
    for idx in valid_indices[1:]:
        dt = time_seconds[idx] - time_seconds[prev_idx]
        if dt > 0 and (max_dt is None or dt <= max_dt):
            derivative[idx] = (values[idx] - values[prev_idx]) / dt
        prev_idx = idx
    return derivative


def _to_serializable(value):
    """Convert numpy/python types to plain JSON-serialisable values."""
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, (datetime,)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_serializable(v) for v in value]
    return value


def _canonical_json_copy(value: Any) -> Any:
    """Return a deterministic, process-safe copy of one JSON value."""

    return json.loads(
        json.dumps(
            value,
            default=_to_serializable,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    )


def _canonical_json_sha256(value: Any) -> str:
    canonical = _canonical_json_copy(value)
    payload = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _count_reason_bits(reason_codes: np.ndarray) -> Dict[str, int]:
    """Aggregate counts for each reason bit."""
    counts: Dict[str, int] = {}
    for code, name in REASON_CODE_MAP.items():
        mask = (reason_codes & code) > 0
        counts[name] = int(mask.sum())
    return counts


def _prepare_output_arrays(
    group: zarr.Group,
    dataset_specs: List[Tuple[str, Tuple[int, ...], Tuple[int, ...], str]],
    fill_value: Optional[float] = None,
    *,
    storage_entries: Mapping[str, AnalysisArrayStoragePlanReceipt] | None = None,
    path_prefix: str = "",
) -> None:
    """Create (or overwrite) output arrays according to specs."""
    for name, shape, chunks, dtype in dataset_specs:
        path = f"{path_prefix}/{name}" if path_prefix else name
        entry = (storage_entries or {}).get(path)
        if name in group:
            existing = group[name]
            if tuple(existing.shape) == tuple(shape) and np.dtype(existing.dtype) == np.dtype(dtype):
                continue
            del group[name]
        if entry is not None:
            if entry.plan.logical_shape != tuple(shape) or np.dtype(
                entry.plan.logical_dtype
            ) != np.dtype(dtype):
                raise ValueError(
                    f"{path}: resolved storage plan differs from writer array spec."
                )
            create_eye_angle_array_from_entry(
                group,
                name=name,
                entry=entry,
            )
            continue
        kwargs = {"dtype": dtype, "chunks": chunks, "overwrite": True}
        if fill_value is not None:
            kwargs["fill_value"] = fill_value
        group.create_array(name, shape=shape, **kwargs)


def _fixed_width_text_array(values: Sequence[object], *, width: int = 256) -> np.ndarray:
    """Encode text metadata as uint8 fixed-width rows for Zarr-v3 stability."""

    out = np.zeros((len(values), width), dtype=np.uint8)
    for idx, value in enumerate(values):
        encoded = str(value or "").encode("utf-8")[: max(0, width - 1)]
        out[idx, : len(encoded)] = np.frombuffer(encoded, dtype=np.uint8)
    return out


def _write_text_index_array(
    group: zarr.Group,
    name: str,
    values: Sequence[object],
    *,
    width: int = 256,
    storage_entries: Mapping[str, AnalysisArrayStoragePlanReceipt] | None = None,
    path_prefix: str = "",
) -> None:
    data = _fixed_width_text_array(values, width=width)
    if name in group:
        del group[name]
    path = f"{path_prefix}/{name}" if path_prefix else name
    entry = (storage_entries or {}).get(path)
    if entry is not None:
        create_eye_angle_array_from_entry(
            group,
            name=name,
            entry=entry,
            data=data,
        )
        return
    group.create_array(name, data=data, chunks=(max(1, int(data.shape[0])), int(data.shape[1])), overwrite=True)


def _write_bool_index_array(
    group: zarr.Group,
    name: str,
    values: Sequence[bool],
    *,
    storage_entries: Mapping[str, AnalysisArrayStoragePlanReceipt] | None = None,
    path_prefix: str = "",
) -> None:
    data = np.asarray(values, dtype=bool)
    if name in group:
        del group[name]
    path = f"{path_prefix}/{name}" if path_prefix else name
    entry = (storage_entries or {}).get(path)
    if entry is not None:
        create_eye_angle_array_from_entry(
            group,
            name=name,
            entry=entry,
            data=data,
        )
        return
    group.create_array(name, data=data, chunks=(max(1, int(data.shape[0])),), overwrite=True)


def _delete_child(group: zarr.Group, name: str) -> None:
    if name in group:
        del group[name]


def _array_keys(group: zarr.Group) -> list[str]:
    try:
        return sorted(str(key) for key in group.keys())
    except Exception:
        return []


def _scalar_channel_names(group: zarr.Group, *, dtype_kinds: str) -> list[str]:
    names: list[str] = []
    for name in _array_keys(group):
        try:
            array = group[name]
            if len(array.shape) == 1 and np.dtype(array.dtype).kind in dtype_kinds:
                names.append(name)
        except Exception:
            continue
    return names


def _vector_channel_names(group: zarr.Group) -> list[str]:
    names: list[str] = []
    for name in _array_keys(group):
        try:
            array = group[name]
            if len(array.shape) == 2 and int(array.shape[1]) == 2 and np.dtype(array.dtype).kind == "f":
                names.append(name)
        except Exception:
            continue
    return names


def _ordered_union(*name_lists: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for names in name_lists:
        for name in names:
            if name in seen:
                continue
            seen.add(name)
            ordered.append(str(name))
    return ordered


_SEMANTIC_ANGLE_BASE_BUNDLES: tuple[tuple[str, ...], ...] = (
    ("left_eye_angle_deg", "right_eye_angle_deg", "vergence_eye_angle_deg"),
    ("left_gaze_signed_deg", "right_gaze_signed_deg", "vergence_gaze_deg"),
    (
        "left_nasal_gaze_deg",
        "right_nasal_gaze_deg",
        "mean_eye_vergence_gaze_deg",
    ),
    ("left_major_signed_deg", "right_major_signed_deg", "vergence_major_signed_deg"),
    ("left_centroid_deg", "right_centroid_deg", "vergence_centroid_deg"),
    ("left_deg", "right_deg", "vergence_deg"),
    ("left_signed_deg", "right_signed_deg", "vergence_signed_deg"),
    (
        "left_minor_signed_deg",
        "right_minor_signed_deg",
        "vergence_minor_signed_deg",
    ),
    ("left_gaze_deg", "right_gaze_deg", "vergence_gaze_signed_deg"),
)

EYE_ANGLE_PRIMARY_INTERACTIVE_CHANNELS: tuple[str, ...] = (
    "left_eye_angle_deg",
    "right_eye_angle_deg",
    "vergence_eye_angle_deg",
    "left_eye_angle_deg_smoothed",
    "right_eye_angle_deg_smoothed",
    "vergence_eye_angle_deg_smoothed",
    "left_gaze_signed_deg",
    "right_gaze_signed_deg",
    "vergence_gaze_deg",
    "left_gaze_signed_deg_smoothed",
    "right_gaze_signed_deg_smoothed",
    "vergence_gaze_deg_smoothed",
    "left_nasal_gaze_deg",
    "right_nasal_gaze_deg",
    "mean_eye_vergence_gaze_deg",
    "mean_eye_vergence_gaze_deg_smoothed",
)

_SEMANTIC_ANGLE_KINEMATIC_BUNDLES: tuple[tuple[str, ...], ...] = (
    ("left_speed_deg_s", "right_speed_deg_s", "vergence_speed_deg_s"),
    (
        "left_gaze_speed_deg_s",
        "right_gaze_speed_deg_s",
        "vergence_gaze_speed_deg_s",
    ),
    ("left_accel_deg_s2", "right_accel_deg_s2", "vergence_accel_deg_s2"),
    (
        "left_gaze_accel_deg_s2",
        "right_gaze_accel_deg_s2",
        "vergence_gaze_accel_deg_s2",
    ),
)


def _angle_variant_name(base_name: str, variant: str) -> str:
    if variant == "raw":
        return base_name
    if variant == "smoothed":
        return f"{base_name}_smoothed"
    stem = base_name[: -len("_deg")] if base_name.endswith("_deg") else base_name
    if variant == "delta":
        return f"{stem}_delta_deg"
    if variant == "delta_smoothed":
        return f"{stem}_delta_deg_smoothed"
    raise ValueError(f"Unknown eye-angle channel variant: {variant!r}")


def semantic_angle_channel_order(
    channel_names: Sequence[str],
    *,
    block_width: int = EYE_ANGLE_DENSE_CHUNK_COLUMNS,
) -> list[str]:
    """Place commonly selected left/right/binocular channels together.

    Channel names remain the sole logical contract.  This ordering is a
    physical locality hint that keeps each available semantic bundle inside a
    column chunk when enough non-bundle channels are available as padding.
    """

    return list(
        _schema_semantic_angle_channel_order(
            channel_names,
            block_width=block_width,
        )
    )


def _stack_scalar_channels(
    group: zarr.Group,
    channel_names: Sequence[str],
    *,
    row_count: int,
    dtype: np.dtype | str,
    fill_value: float | int,
) -> np.ndarray:
    data = np.full((int(row_count), len(channel_names)), fill_value, dtype=dtype)
    for channel_idx, name in enumerate(channel_names):
        if name not in group:
            continue
        values = np.asarray(group[name][:])
        if values.ndim != 1:
            raise ValueError(f"Expected scalar eye-angle channel '{name}' to be 1D, got shape {values.shape}.")
        if int(values.shape[0]) != int(row_count):
            raise ValueError(
                f"Expected scalar eye-angle channel '{name}' length {row_count}, got {values.shape[0]}."
            )
        data[:, channel_idx] = values.astype(dtype, copy=False)
    return data


def _stack_vector_channels(
    group: zarr.Group,
    channel_names: Sequence[str],
    *,
    row_count: int,
    fill_value: float = np.nan,
) -> np.ndarray:
    data = np.full((int(row_count), len(channel_names), 2), fill_value, dtype=np.float32)
    for channel_idx, name in enumerate(channel_names):
        if name not in group:
            continue
        values = np.asarray(group[name][:], dtype=np.float32)
        if values.ndim != 2 or int(values.shape[1]) != 2:
            raise ValueError(f"Expected vector eye-angle channel '{name}' to have shape (N, 2), got {values.shape}.")
        if int(values.shape[0]) != int(row_count):
            raise ValueError(
                f"Expected vector eye-angle channel '{name}' length {row_count}, got {values.shape[0]}."
            )
        data[:, channel_idx, :] = values
    return data


def _replace_array(
    group: zarr.Group,
    name: str,
    data: np.ndarray,
    *,
    chunks: tuple[int, ...],
    storage_entries: Mapping[str, AnalysisArrayStoragePlanReceipt] | None = None,
    path: str | None = None,
) -> None:
    if name in group:
        del group[name]
    entry = (storage_entries or {}).get(path or name)
    if entry is not None:
        create_eye_angle_array_from_entry(
            group,
            name=name,
            entry=entry,
            data=data,
        )
        return
    group.create_array(name, data=data, chunks=chunks, overwrite=True)


def _write_angle_channel_index(
    run_group: zarr.Group,
    channel_names: Sequence[str],
    *,
    roi_available: Sequence[bool],
    frame_available: Sequence[bool],
    storage_entries: Mapping[str, AnalysisArrayStoragePlanReceipt] | None = None,
) -> None:
    group = run_group.require_group("angle_channel_index")
    prefix = "angle_channel_index"
    for name in _array_keys(group):
        del group[name]
    metadata = eye_angle_channel_metadata(channel_names)
    _write_text_index_array(group, "name", metadata["name"], storage_entries=storage_entries, path_prefix=prefix)
    _write_bool_index_array(group, "roi_available", roi_available, storage_entries=storage_entries, path_prefix=prefix)
    _write_bool_index_array(group, "frame_available", frame_available, storage_entries=storage_entries, path_prefix=prefix)
    _write_text_index_array(group, "representation", metadata["representation"], storage_entries=storage_entries, path_prefix=prefix)
    _write_text_index_array(group, "eye", metadata["eye"], width=64, storage_entries=storage_entries, path_prefix=prefix)
    _write_text_index_array(group, "value_kind", metadata["value_kind"], width=64, storage_entries=storage_entries, path_prefix=prefix)
    _write_text_index_array(group, "units", metadata["units"], width=64, storage_entries=storage_entries, path_prefix=prefix)
    _write_text_index_array(group, "source_channel", metadata["source_channel"], storage_entries=storage_entries, path_prefix=prefix)
    _write_text_index_array(group, "formula", metadata["formula"], width=512, storage_entries=storage_entries, path_prefix=prefix)
    _write_text_index_array(
        group,
        "compatibility_alias_of",
        metadata["compatibility_alias_of"],
        storage_entries=storage_entries,
        path_prefix=prefix,
    )
    group.attrs.update(
        eye_angle_channel_index_attrs(
            "angle_channel_index",
            channel_count=len(channel_names),
        )
    )


def _write_vector_channel_index(
    run_group: zarr.Group,
    channel_names: Sequence[str],
    *,
    roi_available: Sequence[bool],
    frame_available: Sequence[bool],
    storage_entries: Mapping[str, AnalysisArrayStoragePlanReceipt] | None = None,
) -> None:
    group = run_group.require_group("vector_channel_index")
    prefix = "vector_channel_index"
    for name in _array_keys(group):
        del group[name]
    metadata = eye_vector_channel_metadata(channel_names)
    _write_text_index_array(group, "name", metadata["name"], storage_entries=storage_entries, path_prefix=prefix)
    _write_bool_index_array(group, "roi_available", roi_available, storage_entries=storage_entries, path_prefix=prefix)
    _write_bool_index_array(group, "frame_available", frame_available, storage_entries=storage_entries, path_prefix=prefix)
    _write_text_index_array(group, "representation", metadata["representation"], storage_entries=storage_entries, path_prefix=prefix)
    _write_text_index_array(group, "eye", metadata["eye"], width=64, storage_entries=storage_entries, path_prefix=prefix)
    _write_text_index_array(group, "value_kind", metadata["value_kind"], width=64, storage_entries=storage_entries, path_prefix=prefix)
    _write_text_index_array(group, "units", metadata["units"], width=64, storage_entries=storage_entries, path_prefix=prefix)
    group.attrs.update(
        eye_angle_channel_index_attrs(
            "vector_channel_index",
            channel_count=len(channel_names),
        )
    )


def _write_qa_channel_index(
    run_group: zarr.Group,
    channel_names: Sequence[str],
    dtype_by_name: Mapping[str, str],
    *,
    roi_available: Sequence[bool],
    frame_available: Sequence[bool],
    storage_entries: Mapping[str, AnalysisArrayStoragePlanReceipt] | None = None,
) -> None:
    group = run_group.require_group("qa_channel_index")
    prefix = "qa_channel_index"
    for name in _array_keys(group):
        del group[name]
    metadata = eye_qa_channel_metadata(
        channel_names,
        dtype_by_name=dtype_by_name,
    )
    _write_text_index_array(group, "name", metadata["name"], storage_entries=storage_entries, path_prefix=prefix)
    _write_bool_index_array(group, "roi_available", roi_available, storage_entries=storage_entries, path_prefix=prefix)
    _write_bool_index_array(group, "frame_available", frame_available, storage_entries=storage_entries, path_prefix=prefix)
    _write_text_index_array(group, "value_kind", metadata["value_kind"], storage_entries=storage_entries, path_prefix=prefix)
    _write_text_index_array(group, "dtype", metadata["dtype"], width=64, storage_entries=storage_entries, path_prefix=prefix)
    group.attrs.update(
        eye_angle_channel_index_attrs(
            "qa_channel_index",
            channel_count=len(channel_names),
        )
    )


def _write_compact_dense_layout(
    run_group: zarr.Group,
    *,
    total_detections: int,
    num_frames: int,
    chunk_len: int,
    frame_chunk: int,
    dense_chunk_rows: int = EYE_ANGLE_DENSE_CHUNK_ROWS,
    dense_chunk_columns: int = EYE_ANGLE_DENSE_CHUNK_COLUMNS,
    enforce_current_schema: bool = False,
    storage_plan: AnalysisStoragePlanReceipt | None = None,
) -> None:
    """Pack completed hierarchical eye-angle outputs into compact dense arrays."""

    if int(dense_chunk_rows) <= 0:
        raise ValueError("Compact eye-angle chunk rows must be positive.")
    if int(dense_chunk_columns) < 3:
        raise ValueError(
            "Compact eye-angle chunks require at least three columns so a "
            "left/right/binocular semantic bundle remains indivisible."
        )

    angles_group = run_group["angles"]
    roi_group = angles_group["roi"]
    frame_group = angles_group["frame"]
    qa_group = run_group["qa"]
    qa_roi = qa_group["roi"]
    qa_frame = qa_group["frame"]
    storage_entries = eye_angle_storage_entries_by_path(storage_plan)

    roi_angle_names = _scalar_channel_names(roi_group, dtype_kinds="f")
    frame_angle_names = _scalar_channel_names(frame_group, dtype_kinds="f")
    if enforce_current_schema:
        if tuple(roi_angle_names) != tuple(sorted(CANONICAL_ROI_ANGLE_CHANNELS)):
            raise ValueError(
                "ROI angle channels differ from compact eye-angle v7: "
                f"missing={sorted(set(CANONICAL_ROI_ANGLE_CHANNELS) - set(roi_angle_names))!r}, "
                f"unexpected={sorted(set(roi_angle_names) - set(CANONICAL_ROI_ANGLE_CHANNELS))!r}."
            )
        if tuple(frame_angle_names) != tuple(sorted(CANONICAL_FRAME_ANGLE_CHANNELS)):
            raise ValueError("Frame angle channels differ from compact eye-angle v7.")
    angle_names = semantic_angle_channel_order(
        _ordered_union(roi_angle_names, frame_angle_names),
        block_width=dense_chunk_columns,
    )
    roi_angle_name_set = set(roi_angle_names)
    frame_angle_name_set = set(frame_angle_names)
    _write_angle_channel_index(
        run_group,
        angle_names,
        roi_available=[name in roi_angle_name_set for name in angle_names],
        frame_available=[name in frame_angle_name_set for name in angle_names],
        storage_entries=storage_entries,
    )
    _replace_array(
        run_group,
        "roi_angles",
        _stack_scalar_channels(
            roi_group,
            angle_names,
            row_count=total_detections,
            dtype=np.float32,
            fill_value=np.nan,
        ),
        chunks=(
            max(1, min(int(dense_chunk_rows), max(1, int(total_detections)))),
            max(1, min(int(dense_chunk_columns), max(1, len(angle_names)))),
        ),
        storage_entries=storage_entries,
        path="roi_angles",
    )
    _replace_array(
        run_group,
        "frame_angles",
        _stack_scalar_channels(
            frame_group,
            angle_names,
            row_count=num_frames,
            dtype=np.float32,
            fill_value=np.nan,
        ),
        chunks=(
            max(1, min(int(dense_chunk_rows), max(1, int(num_frames)))),
            max(1, min(int(dense_chunk_columns), max(1, len(angle_names)))),
        ),
        storage_entries=storage_entries,
        path="frame_angles",
    )

    roi_vector_names = _vector_channel_names(roi_group)
    frame_vector_names = _vector_channel_names(frame_group)
    if enforce_current_schema and (
        tuple(roi_vector_names) != ROI_VECTOR_CHANNELS or frame_vector_names
    ):
        raise ValueError("Vector channels differ from compact eye-angle v7.")
    vector_names = _ordered_union(roi_vector_names, frame_vector_names)
    if vector_names:
        roi_vector_name_set = set(roi_vector_names)
        frame_vector_name_set = set(frame_vector_names)
        _write_vector_channel_index(
            run_group,
            vector_names,
            roi_available=[name in roi_vector_name_set for name in vector_names],
            frame_available=[name in frame_vector_name_set for name in vector_names],
            storage_entries=storage_entries,
        )
        _replace_array(
            run_group,
            "roi_vectors",
            _stack_vector_channels(roi_group, vector_names, row_count=total_detections),
            chunks=(max(1, min(int(chunk_len), max(1, int(total_detections)))), max(1, len(vector_names)), 2),
            storage_entries=storage_entries,
            path="roi_vectors",
        )
        if frame_vector_names:
            _replace_array(
                run_group,
                "frame_vectors",
                _stack_vector_channels(frame_group, vector_names, row_count=num_frames),
                chunks=(max(1, min(int(frame_chunk), max(1, int(num_frames)))), max(1, len(vector_names)), 2),
                storage_entries=storage_entries,
                path="frame_vectors",
            )

    roi_qa_names = _scalar_channel_names(qa_roi, dtype_kinds="bui")
    frame_qa_names = _scalar_channel_names(qa_frame, dtype_kinds="bui")
    if enforce_current_schema and (
        tuple(roi_qa_names) != ROI_QA_CHANNELS
        or tuple(frame_qa_names) != FRAME_QA_CHANNELS
    ):
        raise ValueError("QA channels differ from compact eye-angle v7.")
    qa_names = _ordered_union(roi_qa_names, frame_qa_names)
    dtype_by_name: dict[str, str] = {}
    for source_group in (qa_roi, qa_frame):
        for name in qa_names:
            if name in dtype_by_name or name not in source_group:
                continue
            dtype_by_name[name] = str(np.dtype(source_group[name].dtype))
    roi_qa_name_set = set(roi_qa_names)
    frame_qa_name_set = set(frame_qa_names)
    _write_qa_channel_index(
        run_group,
        qa_names,
        dtype_by_name,
        roi_available=[name in roi_qa_name_set for name in qa_names],
        frame_available=[name in frame_qa_name_set for name in qa_names],
        storage_entries=storage_entries,
    )
    _replace_array(
        run_group,
        "roi_qa",
        _stack_scalar_channels(
            qa_roi,
            qa_names,
            row_count=total_detections,
            dtype=np.uint16,
            fill_value=0,
        ),
        chunks=(max(1, min(int(chunk_len), max(1, int(total_detections)))), max(1, len(qa_names))),
        storage_entries=storage_entries,
        path="roi_qa",
    )
    _replace_array(
        run_group,
        "frame_qa",
        _stack_scalar_channels(
            qa_frame,
            qa_names,
            row_count=num_frames,
            dtype=np.uint16,
            fill_value=0,
        ),
        chunks=(max(1, min(int(frame_chunk), max(1, int(num_frames)))), max(1, len(qa_names))),
        storage_entries=storage_entries,
        path="frame_qa",
    )

    _delete_child(run_group, "angles")
    _delete_child(run_group, "qa")
    run_group.attrs.update(
        {
            "layout": EYE_ANGLE_LAYOUT_COMPACT_DENSE_V2,
            "storage_layout": EYE_ANGLE_LAYOUT_COMPACT_DENSE_V2,
            "compact_dense_v2": True,
            "compact_dense_v2_angle_channel_count": int(len(angle_names)),
            "compact_dense_v2_vector_channel_count": int(len(vector_names)),
            "compact_dense_v2_qa_channel_count": int(len(qa_names)),
            "angle_column_order_contract": {
                "schema_id": EYE_ANGLE_COLUMN_ORDER_SCHEMA_ID,
                "profile": EYE_ANGLE_COLUMN_ORDER_PROFILE,
                "logical_lookup": "angle_channel_index/name",
                "physical_index_semantics": False,
                "semantic_bundle_width": int(dense_chunk_columns),
                "requested_dense_inner_chunks": [
                    int(dense_chunk_rows),
                    int(dense_chunk_columns),
                ],
                "effective_roi_chunks": [
                    int(value) for value in run_group["roi_angles"].chunks
                ],
                "effective_frame_chunks": [
                    int(value) for value in run_group["frame_angles"].chunks
                ],
                "first_angle_chunk_channels": angle_names[
                    : int(dense_chunk_columns)
                ],
            },
            "compact_dense_v2_note": (
                "Eye-angle scalar channels are stored in roi_angles/frame_angles and resolved "
                "by angle_channel_index names; physical column indexes are not semantic. Logical "
                "hierarchical paths remain available through eye_angle_io."
            ),
        }
    )
    if enforce_current_schema:
        dimensions = EyeAngleDimensions(
            n_roi_rows=int(total_detections),
            n_frames=int(num_frames),
            angle_block_width=int(dense_chunk_columns),
        )
        byte_planner_adopted = storage_plan is not None
        manifest = eye_angle_array_schema_manifest(
            dimensions,
            byte_planner_adopted=byte_planner_adopted,
        )
        run_group.attrs[EYE_ANGLE_ARRAY_SCHEMA_ATTR] = manifest
        if storage_plan is not None:
            run_group.attrs[EYE_ANGLE_STORAGE_PLAN_ATTR] = storage_plan.as_manifest()
        issues = validate_eye_angle_compact_arrays(
            collect_eye_angle_arrays(run_group),
            dimensions=dimensions,
            persisted_manifest=manifest,
            channel_index_attrs=collect_eye_angle_channel_index_attrs(run_group),
        )
        if issues:
            raise ValueError(
                "Compact eye-angle v7 array validation failed: "
                + "; ".join(f"{item.code}:{item.path}" for item in issues)
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute head-relative eye angles and QA flags from subject-shape or "
            "refined-subject eye geometry."
        )
    )
    parser.add_argument("zarr_path", type=Path, help="Path to the Palette Zarr archive.")
    parser.add_argument(
        "--subject-shape-run",
        type=str,
        help=(
            "analysis/subject_shape_runs/<run> providing preferred eye geometry "
            "(default: latest subject-shape run with LR eye geometry when available)."
        ),
    )
    parser.add_argument(
        "--refined-subject-run",
        type=str,
        help="Canonical refined_subject_masks_runs/<run> providing eye geometry (default: latest with LR eyes).",
    )
    parser.add_argument(
        "--keypoint-run",
        type=str,
        help=(
            "Optional exact base keypoints_runs child name. This is an assertion "
            "against the canonical keypoint dependency sealed by subject shape; "
            "there is no latest or refined-keypoint fallback."
        ),
    )
    parser.add_argument(
        "--diagnostic-refined-keypoint-run",
        type=str,
        help=(
            "Explicit historical refined_keypoints_runs child for a permanently "
            "nonselector diagnostic eye-angle output. Not accepted by materialized "
            "future-normal publication."
        ),
    )
    parser.add_argument(
        "--run-name",
        type=str,
        help="Optional name for the output run (default: timestamp-based).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=8192,
        help="Number of detections to process per chunk (default: 8192).",
    )
    parser.add_argument(
        "--execution-backend",
        type=_normalize_execution_backend,
        choices=EXECUTION_BACKENDS,
        default=SERIAL_EXECUTION_BACKEND,
        help="Use dask_worker_chunks to process and write independent ROI chunks from workers.",
    )
    parser.add_argument(
        "--scheduler",
        type=_normalize_scheduler,
        choices=SUPPORTED_SCHEDULERS,
        default="single-threaded",
        help="Dask scheduler used when --execution-backend=dask_worker_chunks.",
    )
    parser.add_argument("--num-workers", type=int, help="Dask worker count for --execution-backend=dask_worker_chunks.")
    parser.add_argument(
        "--include-chunk-timings",
        action="store_true",
        help="Store per-chunk timing metadata and include detailed timings in run attributes.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        help="Override frames-per-second when computing derivatives (default: infer from archive).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output.",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=None,
        help=f"Override the moving-average window for angle smoothing (default: {ANGLE_SMOOTHING_WINDOW}).",
    )
    parser.add_argument(
        "--layout",
        choices=EYE_ANGLE_LAYOUT_CHOICES,
        default=EYE_ANGLE_LAYOUT_DEFAULT,
        help=(
            f"Output storage layout (default: {EYE_ANGLE_LAYOUT_DEFAULT}). "
            "compact_dense_v2 packs completed angle/QA outputs into dense channel tables; "
            "hierarchical_v1 writes one array per logical field for compatibility/debug runs."
        ),
    )
    parser.add_argument(
        "--storage-profile",
        choices=EYE_ANGLE_STORAGE_PROFILE_CHOICES,
        default=EYE_ANGLE_LEGACY_EXPLICIT_STORAGE,
        help=(
            "Physical storage policy. The default preserves established explicit "
            "chunks and normal activation. "
            f"{EYE_ANGLE_ACCESS_AWARE_CANDIDATE_PROFILE_ID} is an opt-in, "
            "selector-ineligible byte-planned benchmark candidate."
        ),
    )
    parser.add_argument(
        "--dense-chunk-rows",
        type=int,
        default=EYE_ANGLE_DENSE_CHUNK_ROWS,
        help=(
            "Row dimension of compact roi_angles/frame_angles chunks "
            f"(default: {EYE_ANGLE_DENSE_CHUNK_ROWS})."
        ),
    )
    parser.add_argument(
        "--dense-chunk-columns",
        type=int,
        default=EYE_ANGLE_DENSE_CHUNK_COLUMNS,
        help=(
            "Column dimension of compact roi_angles/frame_angles chunks "
            f"(default: {EYE_ANGLE_DENSE_CHUNK_COLUMNS})."
        ),
    )
    return parser


def _exact_child_run_name(value: Optional[str], *, label: str) -> Optional[str]:
    if value is None:
        return None
    name = str(value).strip()
    if not name or "/" in name or name in {".", "..", "latest"}:
        raise ValueError(f"{label} must be one exact child run name, not a selector or path.")
    return name


def _open_archive_for_eye_angle(zarr_path: Path) -> zarr.Group:
    """Open mutable Palette zarrs with the repository's non-consolidated fallback policy."""
    return open_zarr_root(zarr_path, mode="a")


def _resolve_head_keypoint_indices(
    kp_group: zarr.Group,
    *,
    labels: Optional[Sequence[str]] = None,
) -> Dict[str, int]:
    keypoint_count = int(kp_group["keypoints_roi"].shape[1])
    attrs: Mapping[str, Any] = (
        {"keypoint_labels": [str(value) for value in labels]}
        if labels is not None
        else kp_group.attrs
    )
    try:
        return resolve_required_keypoint_indices_from_attrs(
            attrs,
            _HEAD_KEYPOINT_LABELS,
            keypoint_count=keypoint_count,
        )
    except ValueError as exc:
        raise ValueError(
            "Keypoint run is missing canonical head labels required for eye-angle analysis "
            f"({_HEAD_KEYPOINT_LABELS}): {exc}"
        ) from exc


def _row_identity_evidence(identity: Any) -> dict[str, Any]:
    contract = identity.contract
    key = contract.key_array
    return {
        "record_ref": str(identity.record_ref),
        "record_sha256": str(identity.record_sha256),
        "domain": str(contract.domain),
        "mode": str(contract.mode),
        "components": [str(value) for value in key.components],
        "dtype": str(key.dtype),
        "shape": [int(value) for value in key.shape],
        "leading_dimension": int(contract.leading_dimension),
        "content_sha256": str(key.content_sha256),
    }


def _temporal_authority_evidence(authority: Any) -> dict[str, Any]:
    record = authority.record
    frame = record.source_acquisition_frame_index
    return {
        "record_ref": str(authority.record_ref),
        "record_sha256": str(authority.record_sha256),
        "recording_id": str(record.recording_id),
        "camera_id": str(record.camera_id),
        "source_total_frames": int(record.source_total_frames),
        "source_identity_domain": str(record.source_identity_domain),
        "source_identity_mode": str(record.source_identity_mode),
        "source_leading_dimension": int(record.source_leading_dimension),
        "frame_index_dtype": str(frame.dtype),
        "frame_index_shape": [int(value) for value in frame.shape],
        "frame_index_content_sha256": str(frame.content_sha256),
    }


def _require_ordered_eye_row_alignment(
    subject_identity: Any,
    keypoint_identity: Any,
    subject_temporal: Any,
    keypoint_temporal: Any,
) -> dict[str, Any]:
    subject = _row_identity_evidence(subject_identity)
    keypoint = _row_identity_evidence(keypoint_identity)
    expected_identity_vocabulary = {
        "domain": OBSERVATION_INSTANCE_DOMAIN,
        "mode": INSTANCE_KEY_MODE,
        "components": [INSTANCE_KEY_ARRAY_REF],
    }
    if any(
        subject[name] != value or keypoint[name] != value
        for name, value in expected_identity_vocabulary.items()
    ):
        raise ValueError(
            "Canonical eye inputs must use the observation_instance / "
            "instance_key row-identity vocabulary."
        )
    comparable_identity_fields = (
        "domain",
        "mode",
        "components",
        "dtype",
        "shape",
        "leading_dimension",
        "content_sha256",
    )
    if any(subject[name] != keypoint[name] for name in comparable_identity_fields):
        raise ValueError(
            "Canonical eye keypoints do not have the exact ordered instance_key "
            "identity of the selected subject-shape rows."
        )
    subject_time = _temporal_authority_evidence(subject_temporal)
    keypoint_time = _temporal_authority_evidence(keypoint_temporal)
    comparable_time_fields = (
        "recording_id",
        "camera_id",
        "source_total_frames",
        "source_identity_domain",
        "source_identity_mode",
        "source_leading_dimension",
        "frame_index_dtype",
        "frame_index_shape",
        "frame_index_content_sha256",
    )
    if any(subject_time[name] != keypoint_time[name] for name in comparable_time_fields):
        raise ValueError(
            "Canonical eye keypoints do not have the exact acquisition-frame "
            "mapping of the selected subject-shape rows."
        )
    return {
        "policy": "same_ordered_observation_instance_and_acquisition_time_v1",
        "subject_shape_row_identity": subject,
        "keypoint_row_identity": keypoint,
        "shared_instance_key_content_sha256": subject["content_sha256"],
        "subject_shape_temporal_authority": subject_time,
        "keypoint_temporal_authority": keypoint_time,
        "shared_frame_index_content_sha256": subject_time[
            "frame_index_content_sha256"
        ],
    }


def _array_authority_entry(node: Any, *, array_ref: str) -> dict[str, Any]:
    return {
        "array_ref": str(array_ref),
        "dtype": np.dtype(node.dtype).str,
        "shape": [int(value) for value in node.shape],
        "content_sha256": array_values_sha256(node),
        "canonicalization": EYE_ANGLE_INPUT_PAYLOAD_CANONICALIZATION,
    }


def _build_staged_canonical_keypoint_authority(
    *,
    eye_geometry: Any,
    surfaces: Any,
    assignment_authority: Any,
    alignment: Mapping[str, Any],
) -> dict[str, Any]:
    context = surfaces.context
    run_path = str(context.run_path)
    run_name = run_path.split("/", 1)[1]
    group = context._run_group
    arrays = {
        name: _array_authority_entry(
            group[name],
            array_ref=f"/{run_path}/{name}",
        )
        for name in (
            "keypoints_roi",
            "detection_success",
            "instance_key",
            "source_acquisition_frame_index",
        )
    }
    body = {
        "schema_id": EYE_ANGLE_STAGED_KEYPOINT_AUTHORITY_SCHEMA_ID,
        "schema_version": EYE_ANGLE_STAGED_KEYPOINT_AUTHORITY_SCHEMA_VERSION,
        "authority_scope": EYE_ANGLE_STAGED_KEYPOINT_AUTHORITY_SCOPE,
        "keypoint_run_name": run_name,
        "keypoint_run_path": run_path,
        "publication": {
            "coordinate_context_ref": context.context_record.record_ref,
            "coordinate_context_sha256": context.context_record.record_sha256,
            "coordinate_derivation_ref": surfaces.derivation.record_ref,
            "coordinate_derivation_sha256": surfaces.derivation.record_sha256,
            "keypoints_roi_descriptor_sha256": (
                surfaces.keypoints_roi.descriptor.digest()
            ),
            "row_identity_ref": context.row_identity.record_ref,
            "row_identity_sha256": context.row_identity.record_sha256,
            "temporal_authority_ref": context.temporal_authority.record_ref,
            "temporal_authority_sha256": context.temporal_authority.record_sha256,
            "keypoint_label_authority_ref": (
                context.keypoint_label_authority.record_ref
            ),
            "keypoint_label_authority_sha256": (
                context.keypoint_label_authority.record_sha256
            ),
        },
        "assignment_authority": {
            "record_ref": assignment_authority.record_ref,
            "record_sha256": assignment_authority.record_sha256,
        },
        "subject_shape_run_path": str(
            eye_geometry.subject_shape_coordinate_publication.run_path
        ),
        "ordered_row_alignment": _canonical_json_copy(alignment),
        "keypoint_labels": [str(value) for value in context.keypoint_labels],
        "source_total_frames": int(context.temporal_authority.record.source_total_frames),
        "arrays": arrays,
        "closed_array_inventory": True,
        "normal_reader_authority": False,
    }
    return {**body, "record_sha256": _canonical_json_sha256(body)}


def _canonical_staged_keypoint_authority(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("Staged canonical keypoint authority must be a mapping.")
    canonical = _canonical_json_copy(value)
    digest = canonical.pop("record_sha256", None)
    expected = {
        "schema_id",
        "schema_version",
        "authority_scope",
        "keypoint_run_name",
        "keypoint_run_path",
        "publication",
        "assignment_authority",
        "subject_shape_run_path",
        "ordered_row_alignment",
        "keypoint_labels",
        "source_total_frames",
        "arrays",
        "closed_array_inventory",
        "normal_reader_authority",
    }
    if set(canonical) != expected:
        raise ValueError("Staged canonical keypoint authority fields are not exact.")
    if (
        canonical.get("schema_id") != EYE_ANGLE_STAGED_KEYPOINT_AUTHORITY_SCHEMA_ID
        or canonical.get("schema_version")
        != EYE_ANGLE_STAGED_KEYPOINT_AUTHORITY_SCHEMA_VERSION
        or canonical.get("authority_scope")
        != EYE_ANGLE_STAGED_KEYPOINT_AUTHORITY_SCOPE
        or canonical.get("closed_array_inventory") is not True
        or canonical.get("normal_reader_authority") is not False
        or not _is_sha256(digest)
        or digest != _canonical_json_sha256(canonical)
    ):
        raise ValueError("Staged canonical keypoint authority is unsupported or stale.")
    name = _exact_child_run_name(
        canonical.get("keypoint_run_name"),
        label="Staged canonical keypoint run",
    )
    if canonical.get("keypoint_run_path") != f"keypoints_runs/{name}":
        raise ValueError("Staged canonical keypoint authority names an invalid run path.")
    run_path = str(canonical["keypoint_run_path"])
    publication = canonical.get("publication")
    publication_fields = {
        "coordinate_context_ref",
        "coordinate_context_sha256",
        "coordinate_derivation_ref",
        "coordinate_derivation_sha256",
        "keypoints_roi_descriptor_sha256",
        "row_identity_ref",
        "row_identity_sha256",
        "temporal_authority_ref",
        "temporal_authority_sha256",
        "keypoint_label_authority_ref",
        "keypoint_label_authority_sha256",
    }
    if not isinstance(publication, Mapping) or set(publication) != publication_fields:
        raise ValueError("Staged canonical keypoint publication proof is incomplete.")
    for field in publication_fields:
        value = publication.get(field)
        if field.endswith("sha256"):
            if not _is_sha256(value):
                raise ValueError("Staged canonical keypoint publication digest is invalid.")
        elif not isinstance(value, str) or not value.startswith(f"/{run_path}"):
            raise ValueError("Staged canonical keypoint publication reference escaped its run.")
    assignment = canonical.get("assignment_authority")
    if (
        not isinstance(assignment, Mapping)
        or set(assignment) != {"record_ref", "record_sha256"}
        or not isinstance(assignment.get("record_ref"), str)
        or not assignment["record_ref"].startswith("/")
        or not _is_sha256(assignment.get("record_sha256"))
    ):
        raise ValueError("Staged canonical assignment authority pointer is invalid.")
    subject_shape_path = canonical.get("subject_shape_run_path")
    if (
        not isinstance(subject_shape_path, str)
        or not subject_shape_path.startswith("analysis/subject_shape_runs/")
        or subject_shape_path.count("/") != 2
    ):
        raise ValueError("Staged canonical keypoints name an invalid subject-shape path.")
    labels = canonical.get("keypoint_labels")
    total_frames = canonical.get("source_total_frames")
    if (
        not isinstance(labels, list)
        or not labels
        or any(not isinstance(label, str) or not label for label in labels)
        or type(total_frames) is not int
        or total_frames <= 0
    ):
        raise ValueError("Staged canonical keypoint labels or frame extent are invalid.")
    arrays = canonical.get("arrays")
    required_arrays = {
        "keypoints_roi",
        "detection_success",
        "instance_key",
        "source_acquisition_frame_index",
    }
    if not isinstance(arrays, Mapping) or set(arrays) != required_arrays:
        raise ValueError("Staged canonical keypoint array inventory is not closed.")
    for array_name, entry in arrays.items():
        expected_fields = {
            "array_ref",
            "dtype",
            "shape",
            "content_sha256",
            "canonicalization",
        }
        if (
            not isinstance(entry, Mapping)
            or set(entry) != expected_fields
            or entry.get("array_ref")
            != f"/{canonical['keypoint_run_path']}/{array_name}"
            or not isinstance(entry.get("shape"), list)
            or not entry["shape"]
            or any(type(item) is not int or item < 0 for item in entry["shape"])
            or np.dtype(entry.get("dtype")).str != entry.get("dtype")
            or not _is_sha256(entry.get("content_sha256"))
            or entry.get("canonicalization")
            != EYE_ANGLE_INPUT_PAYLOAD_CANONICALIZATION
        ):
            raise ValueError(
                f"Staged canonical keypoint array authority for {array_name!r} is invalid."
            )
    alignment = canonical.get("ordered_row_alignment")
    alignment_fields = {
        "policy",
        "subject_shape_row_identity",
        "keypoint_row_identity",
        "shared_instance_key_content_sha256",
        "subject_shape_temporal_authority",
        "keypoint_temporal_authority",
        "shared_frame_index_content_sha256",
    }
    if (
        not isinstance(alignment, Mapping)
        or set(alignment) != alignment_fields
        or alignment.get("policy")
        != "same_ordered_observation_instance_and_acquisition_time_v1"
        or arrays["instance_key"]["content_sha256"]
        != alignment.get("shared_instance_key_content_sha256")
        or arrays["source_acquisition_frame_index"]["content_sha256"]
        != alignment.get("shared_frame_index_content_sha256")
    ):
        raise ValueError(
            "Staged canonical keypoint identity arrays disagree with their ordered alignment proof."
        )

    keypoint_shape = arrays["keypoints_roi"]["shape"]
    instance_shape = arrays["instance_key"]["shape"]
    if (
        len(keypoint_shape) != 3
        or keypoint_shape[2] != 2
        or len(instance_shape) != 1
        or arrays["detection_success"]["shape"] != instance_shape
        or arrays["source_acquisition_frame_index"]["shape"] != instance_shape
        or keypoint_shape[0] != instance_shape[0]
        or len(labels) != keypoint_shape[1]
        or len(set(labels)) != len(labels)
        or arrays["detection_success"]["dtype"] != np.dtype(bool).str
        or arrays["instance_key"]["dtype"] != np.dtype("<u8").str
        or arrays["source_acquisition_frame_index"]["dtype"]
        != np.dtype("<i8").str
    ):
        raise ValueError(
            "Staged canonical keypoint arrays do not form one exact row-aligned axis."
        )
    row_count = int(instance_shape[0])

    identity_fields = {
        "record_ref",
        "record_sha256",
        "domain",
        "mode",
        "components",
        "dtype",
        "shape",
        "leading_dimension",
        "content_sha256",
    }
    subject_identity = alignment.get("subject_shape_row_identity")
    keypoint_identity = alignment.get("keypoint_row_identity")
    for evidence in (subject_identity, keypoint_identity):
        if (
            not isinstance(evidence, Mapping)
            or set(evidence) != identity_fields
            or not isinstance(evidence.get("record_ref"), str)
            or not evidence["record_ref"].startswith("/")
            or not _is_sha256(evidence.get("record_sha256"))
            or not _is_sha256(evidence.get("content_sha256"))
            or evidence.get("domain") != OBSERVATION_INSTANCE_DOMAIN
            or evidence.get("mode") != INSTANCE_KEY_MODE
            or evidence.get("components") != [INSTANCE_KEY_ARRAY_REF]
            or evidence.get("dtype") != np.dtype("<u8").str
            or evidence.get("shape") != [row_count]
            or evidence.get("leading_dimension") != row_count
        ):
            raise ValueError("Staged canonical row-identity evidence is invalid.")
    identity_semantic_fields = identity_fields - {"record_ref", "record_sha256"}
    if (
        any(
            subject_identity[field] != keypoint_identity[field]
            for field in identity_semantic_fields
        )
        or keypoint_identity["content_sha256"]
        != arrays["instance_key"]["content_sha256"]
        or publication["row_identity_ref"] != keypoint_identity["record_ref"]
        or publication["row_identity_sha256"]
        != keypoint_identity["record_sha256"]
    ):
        raise ValueError(
            "Staged canonical keypoint row identity is not closed over its publication and array."
        )

    temporal_fields = {
        "record_ref",
        "record_sha256",
        "recording_id",
        "camera_id",
        "source_total_frames",
        "source_identity_domain",
        "source_identity_mode",
        "source_leading_dimension",
        "frame_index_dtype",
        "frame_index_shape",
        "frame_index_content_sha256",
    }
    subject_temporal = alignment.get("subject_shape_temporal_authority")
    keypoint_temporal = alignment.get("keypoint_temporal_authority")
    for temporal, identity in (
        (subject_temporal, subject_identity),
        (keypoint_temporal, keypoint_identity),
    ):
        if (
            not isinstance(temporal, Mapping)
            or set(temporal) != temporal_fields
            or not isinstance(temporal.get("record_ref"), str)
            or not temporal["record_ref"].startswith("/")
            or not _is_sha256(temporal.get("record_sha256"))
            or not isinstance(temporal.get("recording_id"), str)
            or not temporal["recording_id"]
            or not isinstance(temporal.get("camera_id"), str)
            or not temporal["camera_id"]
            or type(temporal.get("source_total_frames")) is not int
            or temporal["source_total_frames"] <= 0
            or temporal.get("source_identity_domain") != identity["domain"]
            or temporal.get("source_identity_mode") != identity["mode"]
            or temporal.get("source_leading_dimension") != row_count
            or temporal.get("frame_index_dtype") != np.dtype("<i8").str
            or temporal.get("frame_index_shape") != [row_count]
            or not _is_sha256(temporal.get("frame_index_content_sha256"))
        ):
            raise ValueError("Staged canonical temporal-authority evidence is invalid.")
    temporal_semantic_fields = temporal_fields - {"record_ref", "record_sha256"}
    if (
        any(
            subject_temporal[field] != keypoint_temporal[field]
            for field in temporal_semantic_fields
        )
        or total_frames != keypoint_temporal["source_total_frames"]
        or keypoint_temporal["frame_index_content_sha256"]
        != arrays["source_acquisition_frame_index"]["content_sha256"]
        or publication["temporal_authority_ref"] != keypoint_temporal["record_ref"]
        or publication["temporal_authority_sha256"]
        != keypoint_temporal["record_sha256"]
    ):
        raise ValueError(
            "Staged canonical temporal authority is not closed over its publication, frame extent, and array."
        )
    return {**canonical, "record_sha256": str(digest)}


def _validate_staged_canonical_keypoint_source(
    root: zarr.Group,
    *,
    authority: Mapping[str, Any],
    subject_shape_authority: Mapping[str, Any],
    expected_keypoint_run: Optional[str],
    verify_payload: bool,
) -> tuple[zarr.Group, str, tuple[str, ...], dict[str, Any]]:
    canonical = _canonical_staged_keypoint_authority(authority)
    run_name = str(canonical["keypoint_run_name"])
    if expected_keypoint_run is not None and expected_keypoint_run != run_name:
        raise ValueError(
            "--keypoint-run differs from the exact base keypoint run sealed by "
            "the selected subject-shape publication."
        )
    subject_publication = subject_shape_authority.get("canonical_publication")
    subject_assignment = (
        subject_publication.get("assignment_keypoint_authority")
        if isinstance(subject_publication, Mapping)
        else None
    )
    alignment = canonical.get("ordered_row_alignment")
    subject_identity = (
        alignment.get("subject_shape_row_identity")
        if isinstance(alignment, Mapping)
        else None
    )
    if (
        not isinstance(subject_publication, Mapping)
        or not isinstance(subject_identity, Mapping)
        or subject_shape_authority.get("source_subject_shape_run_ref")
        != f"/{canonical['subject_shape_run_path']}"
        or subject_identity.get("record_ref")
        != subject_publication.get("row_identity_ref")
        or subject_identity.get("record_sha256")
        != subject_publication.get("row_identity_sha256")
        or subject_assignment != canonical.get("assignment_authority")
    ):
        raise ValueError(
            "Staged canonical keypoint authority is not bound to this subject-shape dependency."
        )
    path = str(canonical["keypoint_run_path"])
    group = root.get(path)
    if group is None:
        raise ValueError(f"Staged canonical keypoint source {path!r} is missing.")
    keypoint_entry = canonical["arrays"]["keypoints_roi"]
    keypoint_count = int(keypoint_entry["shape"][1])
    sealed_labels = tuple(canonical["keypoint_labels"])
    raw_live_labels = group.attrs.get("keypoint_labels")
    if type(raw_live_labels) is not list or raw_live_labels != list(sealed_labels):
        raise ValueError(
            "Staged canonical keypoint labels differ from the exact source-group "
            "label order."
        )
    # Confirm that the exact live labels remain resolvable for anatomical
    # selection without using alias normalization as the equality authority.
    resolved_live_labels = resolve_keypoint_labels_from_attrs(
        group.attrs,
        keypoint_count=keypoint_count,
    )
    if not resolved_live_labels:
        raise ValueError(
            "Staged canonical keypoint labels cannot resolve an anatomical axis."
        )
    try:
        label_authority = bind_persisted_coordinate_record(
            group,
            attr_name=KEYPOINT_LABEL_AUTHORITY_ATTR,
        )
    except ValueError as exc:
        raise ValueError(
            "Staged canonical keypoint source lacks its exact persisted label authority."
        ) from exc
    label_record = label_authority.record
    label_row_axis = label_record.get("axis0")
    label_axis = label_record.get("axis1")
    label_coordinate_axis = label_record.get("coordinate_component_axis")
    label_arrays = label_record.get("arrays")
    label_keypoints = (
        label_arrays.get("keypoints_roi")
        if isinstance(label_arrays, Mapping)
        else None
    )
    alignment = canonical["ordered_row_alignment"]
    keypoint_identity = alignment["keypoint_row_identity"]
    publication = canonical["publication"]
    if (
        label_authority.record_ref
        != publication["keypoint_label_authority_ref"]
        or label_authority.record_sha256
        != publication["keypoint_label_authority_sha256"]
        or label_record.get("schema_id") != KEYPOINT_LABEL_AUTHORITY_SCHEMA_ID
        or type(label_record.get("schema_version")) is not int
        or label_record.get("schema_version")
        != KEYPOINT_LABEL_AUTHORITY_SCHEMA_VERSION
        or not isinstance(label_row_axis, Mapping)
        or label_row_axis.get("role") != OBSERVATION_INSTANCE_DOMAIN
        or label_row_axis.get("row_identity_ref")
        != keypoint_identity["record_ref"]
        or label_row_axis.get("row_identity_sha256")
        != keypoint_identity["record_sha256"]
        or not isinstance(label_axis, Mapping)
        or label_axis.get("role") != "keypoint"
        or type(label_axis.get("cardinality")) is not int
        or label_axis.get("cardinality") != keypoint_count
        or label_axis.get("labels") != list(sealed_labels)
        or not isinstance(label_coordinate_axis, Mapping)
        or set(label_coordinate_axis) != {"axis", "components"}
        or type(label_coordinate_axis.get("axis")) is not int
        or label_coordinate_axis.get("axis") != 2
        or label_coordinate_axis.get("components") != ["x", "y"]
        or not isinstance(label_keypoints, Mapping)
        or set(label_keypoints)
        != {"array_ref", "shape", "dtype", "keypoint_axis"}
        or label_keypoints.get("array_ref")
        != f"/{canonical['keypoint_run_path']}/keypoints_roi"
        or label_keypoints.get("shape") != keypoint_entry["shape"]
        or label_keypoints.get("dtype") != keypoint_entry["dtype"]
        or type(label_keypoints.get("keypoint_axis")) is not int
        or label_keypoints.get("keypoint_axis") != 1
    ):
        raise ValueError(
            "Staged canonical keypoint label authority differs from its sealed "
            "publication or ordered labels."
        )
    for name, entry in canonical["arrays"].items():
        node = group.get(name)
        if (
            node is None
            or np.dtype(node.dtype).str != entry["dtype"]
            or [int(value) for value in node.shape] != entry["shape"]
            or (
                verify_payload
                and array_values_sha256(node) != entry["content_sha256"]
            )
        ):
            raise ValueError(
                f"Staged canonical keypoint array {path}/{name} differs from its authority."
            )
    return group, run_name, tuple(canonical["keypoint_labels"]), canonical


def _resolve_canonical_eye_keypoints(
    root: zarr.Group,
    *,
    eye_geometry: Any,
    expected_keypoint_run: Optional[str],
    staged_keypoint_authority: Optional[Mapping[str, Any]],
    staged_subject_shape_authority: Optional[Mapping[str, Any]],
    verify_staged_payload: bool,
) -> tuple[Any, ...]:
    if staged_keypoint_authority is not None:
        if not isinstance(staged_subject_shape_authority, Mapping):
            raise ValueError(
                "Staged canonical keypoints require the matching subject-shape authority."
            )
        group, run_name, labels, authority = _validate_staged_canonical_keypoint_source(
            root,
            authority=staged_keypoint_authority,
            subject_shape_authority=staged_subject_shape_authority,
            expected_keypoint_run=expected_keypoint_run,
            verify_payload=verify_staged_payload,
        )
        path = f"keypoints_runs/{run_name}"
        return (
            group,
            path,
            run_name,
            labels,
            None,
            authority,
            int(authority["source_total_frames"]),
        )

    publication = getattr(
        eye_geometry,
        "subject_shape_coordinate_publication",
        None,
    )
    if publication is None:
        raise ValueError(
            "Future-normal eye analysis requires a canonical subject-shape publication. "
            "Refined-subject geometry is available only with an explicit refined-keypoint "
            "diagnostic source."
        )
    refined_context = publication.source.context
    nested = refined_context.assignment_keypoint_surfaces
    assignment_authority = refined_context.assignment_keypoint_authority
    if (
        nested is None
        or assignment_authority.record.get("status") != "used"
    ):
        raise ValueError(
            "The selected subject-shape publication does not seal canonical assignment "
            "keypoints; future-normal eye analysis must fail closed."
        )
    path = str(nested.context.run_path)
    if not path.startswith("keypoints_runs/") or path.count("/") != 1:
        raise ValueError("Subject-shape assignment names a noncanonical keypoint path.")
    run_name = path.split("/", 1)[1]
    if expected_keypoint_run is not None and expected_keypoint_run != run_name:
        raise ValueError(
            "--keypoint-run differs from the exact base keypoint run sealed by "
            "the selected subject-shape publication."
        )
    surfaces = load_persisted_keypoint_coordinate_surfaces(root, path)
    if (
        surfaces.context.run_path != path
        or surfaces.derivation.record_sha256 != nested.derivation.record_sha256
        or surfaces.context.row_identity.record_sha256
        != nested.context.row_identity.record_sha256
        or surfaces.keypoints_roi.descriptor.digest()
        != nested.keypoints_roi.descriptor.digest()
    ):
        raise ValueError(
            "Fresh canonical keypoint proof differs from the subject-shape assignment proof."
        )
    group = root.get(path)
    if group is None or "detection_success" not in group:
        raise ValueError("Canonical assignment keypoint run lacks detection_success.")
    success = group["detection_success"]
    expected_success = assignment_authority.record.get("success", {}).get("payload")
    expected_keypoints = assignment_authority.record.get("keypoints_roi", {}).get(
        "payload"
    )
    if (
        not isinstance(expected_success, Mapping)
        or not isinstance(expected_keypoints, Mapping)
        or array_values_sha256(success)
        != expected_success.get("array_values_sha256")
        or array_values_sha256(group["keypoints_roi"])
        != expected_keypoints.get("array_values_sha256")
    ):
        raise ValueError(
            "Canonical eye keypoint values differ from the subject-shape assignment authority."
        )
    alignment = _require_ordered_eye_row_alignment(
        publication.row_identity,
        surfaces.context.row_identity,
        publication.temporal_authority,
        surfaces.context.temporal_authority,
    )
    authority = _build_staged_canonical_keypoint_authority(
        eye_geometry=eye_geometry,
        surfaces=surfaces,
        assignment_authority=assignment_authority,
        alignment=alignment,
    )
    authority_arrays = authority["arrays"]
    if (
        authority_arrays["keypoints_roi"]["content_sha256"]
        != expected_keypoints.get("array_values_sha256")
        or authority_arrays["detection_success"]["content_sha256"]
        != expected_success.get("array_values_sha256")
        or authority_arrays["instance_key"]["content_sha256"]
        != alignment["shared_instance_key_content_sha256"]
        or authority_arrays["source_acquisition_frame_index"]["content_sha256"]
        != alignment["shared_frame_index_content_sha256"]
    ):
        raise ValueError(
            "Canonical eye keypoint payload changed while its detached authority "
            "was being sealed."
        )
    return (
        group,
        path,
        run_name,
        tuple(surfaces.context.keypoint_labels),
        surfaces,
        authority,
        int(surfaces.context.temporal_authority.record.source_total_frames),
    )


def _resolve_refined_keypoint_diagnostic(
    root: zarr.Group,
    *,
    run_name: str,
) -> tuple[zarr.Group, str, tuple[str, ...], Optional[zarr.Group], Optional[str]]:
    parent = root.get("refined_keypoints_runs")
    if parent is None or run_name not in parent:
        raise ValueError(f"Explicit refined-keypoint diagnostic run {run_name!r} is missing.")
    group = parent[run_name]
    if (
        group.attrs.get("palette_run_completion_status") != "complete"
        or group.attrs.get("stage_selector_eligible") is not False
        or group.attrs.get("coordinate_contract")
        != "palette.refined_keypoints.legacy_unverified_nonselector.v1"
        or group.attrs.get("legacy_unverified_diagnostic_output") is not True
        or group.attrs.get("publication_scope") != "historical_diagnostic_only"
    ):
        raise ValueError(
            "Refined keypoints may enter eye analysis only as an explicitly declared, "
            "permanently nonselector legacy diagnostic."
        )
    for name in (
        "keypoints_roi",
        "refined_success",
        "instance_key",
        "frame_indices",
    ):
        if name not in group:
            raise ValueError(
                f"Refined-keypoint diagnostic {run_name!r} lacks exact local {name!r}."
            )
    keypoint_count = int(group["keypoints_roi"].shape[1])
    labels = resolve_keypoint_labels_from_attrs(
        group.attrs,
        keypoint_count=keypoint_count,
    )
    if not labels:
        raise ValueError("Refined-keypoint diagnostic lacks exact keypoint labels.")
    source_name = resolve_source_keypoints_run(group.attrs)
    source_group = None
    source_path = None
    if isinstance(source_name, str) and source_name:
        source_group = root.get(f"keypoints_runs/{source_name}")
        source_path = f"keypoints_runs/{source_name}" if source_group is not None else None
    return group, f"refined_keypoints_runs/{run_name}", tuple(labels), source_group, source_path


def _resolve_eye_angle_inputs(
    root: zarr.Group,
    *,
    subject_shape_run: Optional[str],
    refined_subject_run: Optional[str],
    keypoint_run: Optional[str],
    diagnostic_refined_keypoint_run: Optional[str] = None,
    _staged_subject_shape_authority: Optional[Mapping[str, Any]] = None,
    _staged_keypoint_authority: Optional[Mapping[str, Any]] = None,
    _verify_staged_payload: bool = True,
) -> EyeAngleInputContext:
    expected_keypoint_run = _exact_child_run_name(
        keypoint_run,
        label="Canonical base keypoint run",
    )
    diagnostic_run = _exact_child_run_name(
        diagnostic_refined_keypoint_run,
        label="Diagnostic refined-keypoint run",
    )
    if expected_keypoint_run is not None and diagnostic_run is not None:
        raise ValueError(
            "Canonical --keypoint-run and --diagnostic-refined-keypoint-run are mutually exclusive."
        )
    if diagnostic_run is not None and (
        _staged_subject_shape_authority is not None
        or _staged_keypoint_authority is not None
    ):
        raise ValueError("Staged/materialized eye analysis cannot use refined diagnostics.")
    eye_geometry = resolve_eye_geometry_source(
        root,
        subject_shape_run=subject_shape_run,
        refined_subject_run=refined_subject_run,
        prefer_subject_shape=True,
        prefer_subject=True,
        _staged_subject_shape_authority=_staged_subject_shape_authority,
        _verify_staged_payload=_verify_staged_payload,
    )
    if diagnostic_run is None:
        (
            kp_group,
            kp_group_path,
            keypoint_run_name,
            keypoint_labels,
            canonical_surfaces,
            canonical_authority,
            source_total_frames,
        ) = _resolve_canonical_eye_keypoints(
            root,
            eye_geometry=eye_geometry,
            expected_keypoint_run=expected_keypoint_run,
            staged_keypoint_authority=_staged_keypoint_authority,
            staged_subject_shape_authority=_staged_subject_shape_authority,
            verify_staged_payload=_verify_staged_payload,
        )
        source_kp_group = None
        source_kp_run_name = None
        source_kp_group_path = None
        detection_success_key = "detection_success"
        frame_indices_key = "source_acquisition_frame_index"
        instance_key_source: Optional[zarr.Group] = kp_group
        instance_key_key: Optional[str] = "instance_key"
        instance_key_path: Optional[str] = f"{kp_group_path}/instance_key"
        keypoint_source_mode = EYE_ANGLE_KEYPOINT_SOURCE_CANONICAL
    else:
        (
            kp_group,
            kp_group_path,
            keypoint_labels,
            source_kp_group,
            source_kp_group_path,
        ) = _resolve_refined_keypoint_diagnostic(root, run_name=diagnostic_run)
        keypoint_run_name = diagnostic_run
        source_kp_run_name = resolve_source_keypoints_run(kp_group.attrs)
        detection_success_key = "refined_success"
        frame_indices_key = "frame_indices"
        instance_key_source = kp_group if "instance_key" in kp_group else None
        instance_key_key = "instance_key" if instance_key_source is not None else None
        instance_key_path = (
            f"{kp_group_path}/instance_key" if instance_key_source is not None else None
        )
        canonical_surfaces = None
        canonical_authority = None
        source_total_frames = None
        keypoint_source_mode = EYE_ANGLE_KEYPOINT_SOURCE_REFINED_DIAGNOSTIC

    detection_success_source = kp_group
    detection_success_path = f"{kp_group_path}/{detection_success_key}"
    frame_indices_source = kp_group
    frame_indices_path = f"{kp_group_path}/{frame_indices_key}"

    total_detections = eye_geometry.ellipse_params.shape[0]
    if kp_group["keypoints_roi"].shape[0] != total_detections:
        raise ValueError("Mismatch between eye geometry source and keypoint detections.")
    row_count = int(total_detections)
    if (
        tuple(int(value) for value in kp_group[detection_success_key].shape)
        != (row_count,)
        or np.dtype(kp_group[detection_success_key].dtype) != np.dtype(bool)
        or tuple(int(value) for value in kp_group[frame_indices_key].shape)
        != (row_count,)
    ):
        raise ValueError("Eye keypoint success or acquisition-frame axis is not row aligned.")
    if (
        instance_key_source is None
        or tuple(int(value) for value in instance_key_source["instance_key"].shape)
        != (row_count,)
        or np.dtype(instance_key_source["instance_key"].dtype) != np.dtype("<u8")
    ):
        raise ValueError("Eye keypoint inputs require exact uint64 instance_key rows.")

    return EyeAngleInputContext(
        eye_geometry=eye_geometry,
        kp_group=kp_group,
        kp_group_path=kp_group_path,
        source_kp_group=source_kp_group,
        source_kp_run_name=source_kp_run_name,
        source_kp_group_path=source_kp_group_path,
        detection_success_source=detection_success_source,
        detection_success_key=detection_success_key,
        detection_success_path=detection_success_path,
        frame_indices_source=frame_indices_source,
        frame_indices_key=frame_indices_key,
        frame_indices_path=frame_indices_path,
        instance_key_source=instance_key_source,
        instance_key_key=instance_key_key,
        instance_key_path=instance_key_path,
        keypoint_run_name=keypoint_run_name,
        keypoint_indices=_resolve_head_keypoint_indices(
            kp_group,
            labels=keypoint_labels,
        ),
        keypoint_labels=tuple(keypoint_labels),
        keypoint_source_mode=keypoint_source_mode,
        source_total_frames=source_total_frames,
        canonical_keypoint_surfaces=canonical_surfaces,
        canonical_keypoint_authority=canonical_authority,
    )


def _resolved_eye_angle_input_identity(
    context: EyeAngleInputContext,
) -> Dict[str, object]:
    """Capture exact run/path selection plus its stable scientific contract."""

    geometry = context.eye_geometry
    return _canonical_json_copy(
        {
            "eye_geometry_stage": geometry.stage_group,
            "eye_geometry_run": geometry.run_name,
            "eye_geometry_path": geometry.group_path,
            "source_authority_mode": getattr(
                geometry,
                "source_authority_mode",
                "canonical_publication",
            )
            or "canonical_publication",
            "keypoint_source_mode": context.keypoint_source_mode,
            "keypoints_run": context.keypoint_run_name,
            "keypoints_path": context.kp_group_path,
            "diagnostic_base_keypoints_run": context.source_kp_run_name,
            "diagnostic_base_keypoints_path": context.source_kp_group_path,
            "detection_success_path": context.detection_success_path,
            "instance_key_path": context.instance_key_path,
            "source_acquisition_frame_index_path": context.frame_indices_path,
            "resolved_head_keypoint_indices": {
                key: int(value) for key, value in context.keypoint_indices.items()
            },
            "source_contracts": _eye_angle_source_contracts(context),
        }
    )


def _staged_input_source_identity(context: EyeAngleInputContext) -> dict[str, Any]:
    geometry = context.eye_geometry
    return {
        "eye_geometry_stage": str(geometry.stage_group),
        "eye_geometry_run": str(geometry.run_name),
        "eye_geometry_path": str(geometry.group_path),
        "keypoint_source_mode": str(context.keypoint_source_mode),
        "keypoints_run": str(context.keypoint_run_name),
        "keypoints_path": str(context.kp_group_path),
        "diagnostic_base_keypoints_run": context.source_kp_run_name,
        "diagnostic_base_keypoints_path": context.source_kp_group_path,
        "detection_success_path": str(context.detection_success_path),
        "instance_key_path": context.instance_key_path,
        "source_acquisition_frame_index_path": str(context.frame_indices_path),
    }


def _logical_input_source_specs(
    context: EyeAngleInputContext,
) -> dict[str, dict[str, Any]]:
    geometry = context.eye_geometry
    geometry_path = str(geometry.group_path)
    ellipse_param_nodes = (
        geometry.group["components/eye_left/ellipse_params"],
        geometry.group["components/eye_right/ellipse_params"],
    )
    ellipse_success_nodes = (
        geometry.group["components/eye_left/ellipse_success"],
        geometry.group["components/eye_right/ellipse_success"],
    )
    direct_nodes = {
        "keypoints_roi": context.kp_group["keypoints_roi"],
        "detection_success": context.detection_success_source[
            context.detection_success_key
        ],
        "instance_key": context.instance_key_source[context.instance_key_key],
        "source_acquisition_frame_index": context.frame_indices_source[
            context.frame_indices_key
        ],
    }

    def source_metadata(nodes: Sequence[Any]) -> tuple[list[str], list[list[int]]]:
        return (
            [np.dtype(node.dtype).str for node in nodes],
            [[int(value) for value in node.shape] for node in nodes],
        )

    param_dtypes, param_shapes = source_metadata(ellipse_param_nodes)
    success_dtypes, success_shapes = source_metadata(ellipse_success_nodes)
    specs: dict[str, dict[str, Any]] = {
        "ellipse_params": {
            "source_array_refs": [
                f"{geometry_path}/components/eye_left/ellipse_params",
                f"{geometry_path}/components/eye_right/ellipse_params",
            ],
            "source_dtypes": param_dtypes,
            "source_shapes": param_shapes,
            "assembly": "stack_axis_1_eye_left_then_eye_right",
            "snapshot_dtype": np.dtype(geometry.ellipse_params.dtype).str,
            "snapshot_shape": [int(value) for value in geometry.ellipse_params.shape],
            "canonicalization": EYE_ANGLE_INPUT_PAYLOAD_CANONICALIZATION,
        },
        "ellipse_success": {
            "source_array_refs": [
                f"{geometry_path}/components/eye_left/ellipse_success",
                f"{geometry_path}/components/eye_right/ellipse_success",
            ],
            "source_dtypes": success_dtypes,
            "source_shapes": success_shapes,
            "assembly": "stack_axis_1_eye_left_then_eye_right",
            "snapshot_dtype": np.dtype(geometry.ellipse_success.dtype).str,
            "snapshot_shape": [int(value) for value in geometry.ellipse_success.shape],
            "canonicalization": EYE_ANGLE_INPUT_PAYLOAD_CANONICALIZATION,
        },
    }
    direct_paths = {
        "keypoints_roi": f"{context.kp_group_path}/keypoints_roi",
        "detection_success": context.detection_success_path,
        "instance_key": context.instance_key_path,
        "source_acquisition_frame_index": context.frame_indices_path,
    }
    normalized_dtypes = {
        "detection_success": np.dtype(bool).str,
        "instance_key": np.dtype("<u8").str,
        "source_acquisition_frame_index": np.dtype(np.int64).str,
    }
    normalized_assemblies = {
        "detection_success": "owned_c_order_astype_bool",
        "instance_key": "owned_c_order_astype_uint64",
        "source_acquisition_frame_index": "owned_c_order_astype_int64",
    }
    for name, node in direct_nodes.items():
        snapshot_dtype = normalized_dtypes.get(name, np.dtype(node.dtype).str)
        specs[name] = {
            "source_array_refs": [str(direct_paths[name])],
            "source_dtypes": [np.dtype(node.dtype).str],
            "source_shapes": [[int(value) for value in node.shape]],
            "assembly": normalized_assemblies.get(name, "owned_c_order_identity"),
            "snapshot_dtype": snapshot_dtype,
            "snapshot_shape": [int(value) for value in node.shape],
            "canonicalization": EYE_ANGLE_INPUT_PAYLOAD_CANONICALIZATION,
        }
    return {name: specs[name] for name in _EYE_ANGLE_WORKER_LOGICAL_INPUTS}


def _owned_c_array(values: Any, *, dtype: Any = None) -> np.ndarray:
    array = np.array(values, dtype=dtype, copy=True, order="C")
    if array.dtype.hasobject:
        raise ValueError("Eye-angle worker inputs cannot use object-reference dtype.")
    array.setflags(write=False)
    return array


def _load_eye_angle_chunk_input_snapshot(
    context: EyeAngleInputContext,
    *,
    start_row: int,
    stop_row: int,
) -> _EyeAngleChunkInputSnapshot:
    row_count = int(context.eye_geometry.ellipse_params.shape[0])
    if (
        type(start_row) is not int
        or type(stop_row) is not int
        or not (0 <= start_row < stop_row <= row_count)
    ):
        raise ValueError("Eye-angle input snapshot row bounds are invalid.")
    row_slice = slice(start_row, stop_row)
    return _EyeAngleChunkInputSnapshot(
        ellipse_params=_owned_c_array(
            context.eye_geometry.ellipse_params[row_slice]
        ),
        ellipse_success=_owned_c_array(
            context.eye_geometry.ellipse_success[row_slice]
        ),
        keypoints_roi=_owned_c_array(context.kp_group["keypoints_roi"][row_slice]),
        detection_success=_owned_c_array(
            context.detection_success_source[context.detection_success_key][row_slice],
            dtype=bool,
        ),
        instance_key=_owned_c_array(
            context.instance_key_source[context.instance_key_key][row_slice],
            dtype=np.uint64,
        ),
        source_acquisition_frame_index=_owned_c_array(
            context.frame_indices_source[context.frame_indices_key][row_slice],
            dtype=np.int64,
        ),
    )


def _chunk_snapshot_arrays(
    snapshot: _EyeAngleChunkInputSnapshot,
) -> dict[str, np.ndarray]:
    return {
        name: getattr(snapshot, name)
        for name in _EYE_ANGLE_WORKER_LOGICAL_INPUTS
    }


def _snapshot_payload_record(values: np.ndarray) -> dict[str, Any]:
    return {
        "dtype": np.dtype(values.dtype).str,
        "shape": [int(value) for value in values.shape],
        "canonicalization": EYE_ANGLE_INPUT_PAYLOAD_CANONICALIZATION,
        "content_sha256": array_values_sha256(values),
    }


def _keypoint_axis_receipt(context: EyeAngleInputContext) -> dict[str, Any]:
    labels = tuple(context.keypoint_labels)
    if not labels:
        raise ValueError(
            "Staged eye-angle integrity requires exact resolved keypoint labels."
        )
    return {
        "resolved_labels": [str(label) for label in labels],
        "resolved_head_keypoint_indices": {
            key: int(value) for key, value in context.keypoint_indices.items()
        },
    }


def _chunk_integrity_record(
    snapshot: _EyeAngleChunkInputSnapshot,
    *,
    chunk_index: int,
    start_row: int,
    stop_row: int,
) -> dict[str, Any]:
    body = {
        "schema_id": EYE_ANGLE_STAGED_INPUT_CHUNK_SCHEMA_ID,
        "schema_version": EYE_ANGLE_STAGED_INPUT_CHUNK_SCHEMA_VERSION,
        "chunk_index": int(chunk_index),
        "start_row": int(start_row),
        "stop_row": int(stop_row),
        "logical_inputs": {
            name: _snapshot_payload_record(values)
            for name, values in _chunk_snapshot_arrays(snapshot).items()
        },
    }
    return {**body, "record_sha256": _canonical_json_sha256(body)}


def _verify_receipt_geometry_payloads(
    context: EyeAngleInputContext,
    parts: Mapping[str, Sequence[np.ndarray]],
) -> None:
    authority = getattr(context.eye_geometry, "source_authority", None)
    allowed = authority.get("allowed_arrays") if isinstance(authority, Mapping) else None
    if not isinstance(allowed, Mapping):
        raise ValueError(
            "Staged eye-angle integrity lacks canonical subject-shape payload authority."
        )
    relative_paths = {
        "left_params": "components/eye_left/ellipse_params",
        "right_params": "components/eye_right/ellipse_params",
        "left_success": "components/eye_left/ellipse_success",
        "right_success": "components/eye_right/ellipse_success",
    }
    for key, relative_path in relative_paths.items():
        node = context.eye_geometry.group[relative_path]
        values = (
            np.concatenate(tuple(parts[key]), axis=0)
            if parts[key]
            else np.empty(tuple(int(value) for value in node.shape), dtype=node.dtype)
        )
        declared = allowed.get(relative_path)
        if (
            not isinstance(declared, Mapping)
            or array_values_sha256(values) != declared.get("content_sha256")
        ):
            raise ValueError(
                f"Staged eye-angle snapshot for {relative_path!r} differs from "
                "canonical subject-shape payload authority."
            )


def _build_staged_eye_angle_input_integrity_receipt(
    context: EyeAngleInputContext,
    *,
    chunk_rows: int,
    fps: Optional[float],
    fps_source: str,
) -> dict[str, Any]:
    """Seal exact staged worker inputs without granting coordinate authority."""

    if type(chunk_rows) is not int or chunk_rows <= 0:
        raise ValueError("Staged eye-angle integrity chunk_rows must be positive.")
    if fps_source not in {
        "cli_override",
        "authoritative_recording_metadata",
        "unavailable",
    }:
        raise ValueError("Staged eye-angle integrity fps_source is unsupported.")
    canonical_fps = None if fps is None else float(fps)
    if (canonical_fps is None) != (fps_source == "unavailable") or (
        canonical_fps is not None and canonical_fps <= 0.0
    ):
        raise ValueError("Staged eye-angle integrity FPS value and source disagree.")
    geometry = context.eye_geometry
    authority = getattr(geometry, "source_authority", None)
    if (
        getattr(geometry, "source_authority_mode", None)
        not in {"canonical_publication", "digest_bound_staged_subset"}
        or not isinstance(authority, Mapping)
        or not _is_sha256(authority.get("record_sha256"))
    ):
        raise ValueError(
            "Staged eye-angle input integrity requires canonical subject-shape "
            "authority or its exact digest-bound staged subset."
        )
    keypoint_authority = context.canonical_keypoint_authority
    if (
        context.keypoint_source_mode != EYE_ANGLE_KEYPOINT_SOURCE_CANONICAL
        or not isinstance(keypoint_authority, Mapping)
        or not _is_sha256(keypoint_authority.get("record_sha256"))
    ):
        raise ValueError(
            "Staged eye-angle input integrity requires the canonical base "
            "keypoints sealed by subject shape."
        )

    row_count = int(geometry.ellipse_params.shape[0])
    chunk_records: list[dict[str, Any]] = []
    geometry_parts: dict[str, list[np.ndarray]] = {
        "left_params": [],
        "right_params": [],
        "left_success": [],
        "right_success": [],
    }
    for chunk_index, (start_row, stop_row) in enumerate(
        _row_chunks(row_count, chunk_rows)
    ):
        snapshot = _load_eye_angle_chunk_input_snapshot(
            context,
            start_row=start_row,
            stop_row=stop_row,
        )
        chunk_records.append(
            _chunk_integrity_record(
                snapshot,
                chunk_index=chunk_index,
                start_row=start_row,
                stop_row=stop_row,
            )
        )
        geometry_parts["left_params"].append(snapshot.ellipse_params[:, 0, ...])
        geometry_parts["right_params"].append(snapshot.ellipse_params[:, 1, ...])
        geometry_parts["left_success"].append(snapshot.ellipse_success[:, 0, ...])
        geometry_parts["right_success"].append(snapshot.ellipse_success[:, 1, ...])
    _verify_receipt_geometry_payloads(context, geometry_parts)

    body = {
        "schema_id": EYE_ANGLE_STAGED_INPUT_INTEGRITY_SCHEMA_ID,
        "schema_version": EYE_ANGLE_STAGED_INPUT_INTEGRITY_SCHEMA_VERSION,
        "integrity_scope": EYE_ANGLE_STAGED_INPUT_INTEGRITY_SCOPE,
        "receipt_role": "materializer_private_integrity_not_coordinate_authority",
        "source_identity": _staged_input_source_identity(context),
        "source_contract_sha256": _canonical_json_sha256(
            _eye_angle_source_contracts(context)
        ),
        "subject_shape_authority_sha256": str(authority["record_sha256"]),
        "subject_shape_authority": _canonical_json_copy(authority),
        "canonical_keypoint_authority_sha256": str(
            keypoint_authority["record_sha256"]
        ),
        "canonical_keypoint_authority": _canonical_json_copy(
            keypoint_authority
        ),
        "keypoint_axis": _keypoint_axis_receipt(context),
        "scientific_parameters": {
            "fps": canonical_fps,
            "fps_source": fps_source,
        },
        "row_count": row_count,
        "requested_chunk_rows": int(chunk_rows),
        "logical_inputs": _logical_input_source_specs(context),
        "chunks": chunk_records,
        "closed_logical_input_inventory": True,
        "normal_reader_authority": False,
        "coordinate_authority": False,
    }
    receipt = {**body, "record_sha256": _canonical_json_sha256(body)}
    # Re-read every logical input and compare it with the just-captured
    # chunks.  This is the keypoint-side counterpart to the canonical
    # subject-shape loader's two-read payload binding.
    return _validate_staged_eye_angle_input_integrity_receipt(
        context,
        receipt,
        verify_payload=True,
    )


def _canonical_chunk_integrity_record(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("Staged eye-angle chunk receipt must be a mapping.")
    canonical = _canonical_json_copy(value)
    digest = canonical.pop("record_sha256", None)
    expected_fields = {
        "schema_id",
        "schema_version",
        "chunk_index",
        "start_row",
        "stop_row",
        "logical_inputs",
    }
    if set(canonical) != expected_fields:
        raise ValueError("Staged eye-angle chunk receipt fields are not exact.")
    if (
        canonical.get("schema_id") != EYE_ANGLE_STAGED_INPUT_CHUNK_SCHEMA_ID
        or type(canonical.get("schema_version")) is not int
        or canonical.get("schema_version")
        != EYE_ANGLE_STAGED_INPUT_CHUNK_SCHEMA_VERSION
        or not _is_sha256(digest)
        or digest != _canonical_json_sha256(canonical)
    ):
        raise ValueError("Staged eye-angle chunk receipt is unsupported or stale.")
    start_row = canonical.get("start_row")
    stop_row = canonical.get("stop_row")
    chunk_index = canonical.get("chunk_index")
    if (
        type(chunk_index) is not int
        or chunk_index < 0
        or type(start_row) is not int
        or type(stop_row) is not int
        or not (0 <= start_row < stop_row)
    ):
        raise ValueError("Staged eye-angle chunk receipt bounds are invalid.")
    logical = canonical.get("logical_inputs")
    if not isinstance(logical, Mapping) or set(logical) != set(
        _EYE_ANGLE_WORKER_LOGICAL_INPUTS
    ):
        raise ValueError("Staged eye-angle chunk logical input inventory is not closed.")
    row_count = stop_row - start_row
    for name in _EYE_ANGLE_WORKER_LOGICAL_INPUTS:
        record = logical.get(name)
        if not isinstance(record, Mapping) or set(record) != {
            "dtype",
            "shape",
            "canonicalization",
            "content_sha256",
        }:
            raise ValueError(
                f"Staged eye-angle chunk payload record for {name!r} is not exact."
            )
        shape = record.get("shape")
        dtype = record.get("dtype")
        try:
            canonical_dtype = np.dtype(dtype).str
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Staged eye-angle chunk dtype for {name!r} is invalid."
            ) from exc
        if (
            canonical_dtype != dtype
            or not isinstance(shape, list)
            or not shape
            or any(type(item) is not int or item < 0 for item in shape)
            or shape[0] != row_count
            or record.get("canonicalization")
            != EYE_ANGLE_INPUT_PAYLOAD_CANONICALIZATION
            or not _is_sha256(record.get("content_sha256"))
        ):
            raise ValueError(
                f"Staged eye-angle chunk payload record for {name!r} is invalid."
            )
    return {**canonical, "record_sha256": str(digest)}


def _canonical_staged_input_integrity_receipt(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("Staged eye-angle input integrity receipt must be a mapping.")
    canonical = _canonical_json_copy(value)
    digest = canonical.pop("record_sha256", None)
    expected_fields = {
        "schema_id",
        "schema_version",
        "integrity_scope",
        "receipt_role",
        "source_identity",
        "source_contract_sha256",
        "subject_shape_authority_sha256",
        "subject_shape_authority",
        "canonical_keypoint_authority_sha256",
        "canonical_keypoint_authority",
        "keypoint_axis",
        "scientific_parameters",
        "row_count",
        "requested_chunk_rows",
        "logical_inputs",
        "chunks",
        "closed_logical_input_inventory",
        "normal_reader_authority",
        "coordinate_authority",
    }
    if set(canonical) != expected_fields:
        raise ValueError("Staged eye-angle input integrity receipt fields are not exact.")
    if (
        canonical.get("schema_id") != EYE_ANGLE_STAGED_INPUT_INTEGRITY_SCHEMA_ID
        or type(canonical.get("schema_version")) is not int
        or canonical.get("schema_version")
        != EYE_ANGLE_STAGED_INPUT_INTEGRITY_SCHEMA_VERSION
        or canonical.get("integrity_scope")
        != EYE_ANGLE_STAGED_INPUT_INTEGRITY_SCOPE
        or canonical.get("receipt_role")
        != "materializer_private_integrity_not_coordinate_authority"
        or canonical.get("closed_logical_input_inventory") is not True
        or canonical.get("normal_reader_authority") is not False
        or canonical.get("coordinate_authority") is not False
        or not _is_sha256(digest)
        or digest != _canonical_json_sha256(canonical)
    ):
        raise ValueError("Staged eye-angle input integrity receipt is unsupported or stale.")
    if not _is_sha256(canonical.get("source_contract_sha256")):
        raise ValueError("Staged eye-angle source-contract digest is invalid.")
    scientific_parameters = canonical.get("scientific_parameters")
    if (
        not isinstance(scientific_parameters, Mapping)
        or set(scientific_parameters) != {"fps", "fps_source"}
        or scientific_parameters.get("fps_source")
        not in {
            "cli_override",
            "authoritative_recording_metadata",
            "unavailable",
        }
    ):
        raise ValueError("Staged eye-angle scientific parameters are invalid.")
    fps_value = scientific_parameters.get("fps")
    if (
        fps_value is not None
        and (type(fps_value) not in {int, float} or float(fps_value) <= 0.0)
    ) or ((fps_value is None) != (scientific_parameters.get("fps_source") == "unavailable")):
        raise ValueError("Staged eye-angle FPS value and source disagree.")

    authority = canonical.get("subject_shape_authority")
    authority_sha = canonical.get("subject_shape_authority_sha256")
    if not isinstance(authority, Mapping) or not _is_sha256(authority_sha):
        raise ValueError("Staged eye-angle receipt lacks subject-shape authority.")
    authority_body = _canonical_json_copy(authority)
    authority_digest = authority_body.pop("record_sha256", None)
    if (
        authority_digest != authority_sha
        or not _is_sha256(authority_digest)
        or authority_digest != _canonical_json_sha256(authority_body)
        or authority.get("normal_reader_authority") is not False
    ):
        raise ValueError("Nested subject-shape authority is missing or stale.")

    keypoint_authority = _canonical_staged_keypoint_authority(
        canonical.get("canonical_keypoint_authority")
    )
    if (
        canonical.get("canonical_keypoint_authority_sha256")
        != keypoint_authority["record_sha256"]
    ):
        raise ValueError("Nested canonical keypoint authority is missing or stale.")
    canonical["canonical_keypoint_authority"] = keypoint_authority

    row_count = canonical.get("row_count")
    chunk_rows = canonical.get("requested_chunk_rows")
    if (
        type(row_count) is not int
        or row_count < 0
        or type(chunk_rows) is not int
        or chunk_rows <= 0
    ):
        raise ValueError("Staged eye-angle receipt row or chunk count is invalid.")
    logical_specs = canonical.get("logical_inputs")
    if not isinstance(logical_specs, Mapping) or set(logical_specs) != set(
        _EYE_ANGLE_WORKER_LOGICAL_INPUTS
    ):
        raise ValueError("Staged eye-angle logical input specification is not closed.")
    spec_fields = {
        "source_array_refs",
        "source_dtypes",
        "source_shapes",
        "assembly",
        "snapshot_dtype",
        "snapshot_shape",
        "canonicalization",
    }
    for name in _EYE_ANGLE_WORKER_LOGICAL_INPUTS:
        spec = logical_specs.get(name)
        if not isinstance(spec, Mapping) or set(spec) != spec_fields:
            raise ValueError(
                f"Staged eye-angle logical input specification for {name!r} is not exact."
            )
        refs = spec.get("source_array_refs")
        source_dtypes = spec.get("source_dtypes")
        source_shapes = spec.get("source_shapes")
        snapshot_shape = spec.get("snapshot_shape")
        if (
            not isinstance(refs, list)
            or not refs
            or any(not isinstance(item, str) or not item for item in refs)
            or not isinstance(source_dtypes, list)
            or len(source_dtypes) != len(refs)
            or not isinstance(source_shapes, list)
            or len(source_shapes) != len(refs)
            or not isinstance(snapshot_shape, list)
            or not snapshot_shape
            or snapshot_shape[0] != row_count
            or spec.get("canonicalization")
            != EYE_ANGLE_INPUT_PAYLOAD_CANONICALIZATION
        ):
            raise ValueError(
                f"Staged eye-angle logical input specification for {name!r} is invalid."
            )
        try:
            if np.dtype(spec.get("snapshot_dtype")).str != spec.get(
                "snapshot_dtype"
            ):
                raise ValueError
            if any(np.dtype(item).str != item for item in source_dtypes):
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Staged eye-angle logical input dtype specification for {name!r} is invalid."
            ) from exc
        if any(
            not isinstance(shape, list)
            or not shape
            or any(type(item) is not int or item < 0 for item in shape)
            for shape in source_shapes
        ) or any(type(item) is not int or item < 0 for item in snapshot_shape):
            raise ValueError(
                f"Staged eye-angle logical input shape specification for {name!r} is invalid."
            )

    chunks = canonical.get("chunks")
    if not isinstance(chunks, list):
        raise ValueError("Staged eye-angle receipt chunk inventory must be a list.")
    canonical_chunks: list[dict[str, Any]] = []
    cursor = 0
    for expected_index, raw_chunk in enumerate(chunks):
        chunk = _canonical_chunk_integrity_record(raw_chunk)
        if (
            chunk["chunk_index"] != expected_index
            or chunk["start_row"] != cursor
            or chunk["stop_row"] > row_count
            or chunk["stop_row"] - chunk["start_row"] > chunk_rows
            or (
                chunk["stop_row"] < row_count
                and chunk["stop_row"] - chunk["start_row"] != chunk_rows
            )
        ):
            raise ValueError(
                "Staged eye-angle chunk receipts have a gap, overlap, or wrong order."
            )
        for name in _EYE_ANGLE_WORKER_LOGICAL_INPUTS:
            payload = chunk["logical_inputs"][name]
            spec = logical_specs[name]
            expected_shape = [
                chunk["stop_row"] - chunk["start_row"],
                *spec["snapshot_shape"][1:],
            ]
            if (
                payload["dtype"] != spec["snapshot_dtype"]
                or payload["shape"] != expected_shape
            ):
                raise ValueError(
                    f"Staged eye-angle chunk payload for {name!r} differs from its "
                    "logical input specification."
                )
        cursor = int(chunk["stop_row"])
        canonical_chunks.append(chunk)
    if cursor != row_count or (row_count == 0 and canonical_chunks):
        raise ValueError("Staged eye-angle chunk receipts do not cover every row exactly once.")
    canonical["chunks"] = canonical_chunks
    return {**canonical, "record_sha256": str(digest)}


def _staged_subject_shape_authority_from_input_receipt(
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Extract nested coordinate authority only after integrity-envelope validation."""

    canonical = _canonical_staged_input_integrity_receipt(receipt)
    return _canonical_json_copy(canonical["subject_shape_authority"])


def _staged_keypoint_authority_from_input_receipt(
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Extract the materializer-private canonical keypoint subset receipt."""

    canonical = _canonical_staged_input_integrity_receipt(receipt)
    return _canonical_json_copy(canonical["canonical_keypoint_authority"])


def _validate_chunk_snapshot_against_receipt(
    snapshot: _EyeAngleChunkInputSnapshot,
    chunk_receipt: Mapping[str, Any],
) -> str:
    chunk = _canonical_chunk_integrity_record(chunk_receipt)
    observed = _chunk_snapshot_arrays(snapshot)
    errors: list[str] = []
    for name in _EYE_ANGLE_WORKER_LOGICAL_INPUTS:
        values = observed[name]
        expected = chunk["logical_inputs"][name]
        if np.dtype(values.dtype).str != expected["dtype"]:
            errors.append(f"{name}: dtype changed")
        elif [int(value) for value in values.shape] != expected["shape"]:
            errors.append(f"{name}: shape changed")
        elif array_values_sha256(values) != expected["content_sha256"]:
            errors.append(f"{name}: payload changed")
        elif (
            values.flags.writeable
            or not values.flags.c_contiguous
            or not values.flags.owndata
        ):
            errors.append(f"{name}: snapshot is not immutable owned C-order data")
    if errors:
        raise ValueError(
            "Staged eye-angle worker input differs from its exact integrity receipt: "
            + "; ".join(errors)
        )
    return str(chunk["record_sha256"])


def _validate_staged_eye_angle_input_integrity_receipt(
    context: EyeAngleInputContext,
    receipt: Mapping[str, Any],
    *,
    verify_payload: bool,
) -> dict[str, Any]:
    """Validate a private integrity receipt without minting reader authority."""

    if type(verify_payload) is not bool:
        raise ValueError("Staged eye-angle payload verification flag must be an exact bool.")
    canonical = _canonical_staged_input_integrity_receipt(receipt)
    geometry = context.eye_geometry
    authority = getattr(geometry, "source_authority", None)
    if (
        getattr(geometry, "source_authority_mode", None)
        not in {"canonical_publication", "digest_bound_staged_subset"}
        or not isinstance(authority, Mapping)
        or _canonical_json_copy(authority) != canonical["subject_shape_authority"]
        or authority.get("record_sha256")
        != canonical["subject_shape_authority_sha256"]
    ):
        raise ValueError(
            "Staged eye-angle integrity receipt is not bound to this exact "
            "subject-shape source authority."
        )
    if (
        context.keypoint_source_mode != EYE_ANGLE_KEYPOINT_SOURCE_CANONICAL
        or not isinstance(context.canonical_keypoint_authority, Mapping)
        or _canonical_json_copy(context.canonical_keypoint_authority)
        != canonical["canonical_keypoint_authority"]
        or context.canonical_keypoint_authority.get("record_sha256")
        != canonical["canonical_keypoint_authority_sha256"]
    ):
        raise ValueError(
            "Staged eye-angle integrity receipt is not bound to this exact "
            "canonical keypoint source authority."
        )
    if canonical["source_identity"] != _staged_input_source_identity(context):
        raise ValueError("Staged eye-angle receipt names different source runs or paths.")
    if canonical["source_contract_sha256"] != _canonical_json_sha256(
        _eye_angle_source_contracts(context)
    ):
        raise ValueError("Staged eye-angle source contract differs from its receipt.")
    if canonical["keypoint_axis"] != _keypoint_axis_receipt(context):
        raise ValueError("Staged eye-angle keypoint labels or head indices changed.")
    if canonical["row_count"] != int(geometry.ellipse_params.shape[0]):
        raise ValueError("Staged eye-angle row count differs from its receipt.")
    if canonical["logical_inputs"] != _logical_input_source_specs(context):
        raise ValueError("Staged eye-angle logical input paths or metadata changed.")

    if verify_payload:
        geometry_parts: dict[str, list[np.ndarray]] = {
            "left_params": [],
            "right_params": [],
            "left_success": [],
            "right_success": [],
        }
        for chunk in canonical["chunks"]:
            snapshot = _load_eye_angle_chunk_input_snapshot(
                context,
                start_row=int(chunk["start_row"]),
                stop_row=int(chunk["stop_row"]),
            )
            _validate_chunk_snapshot_against_receipt(snapshot, chunk)
            geometry_parts["left_params"].append(snapshot.ellipse_params[:, 0, ...])
            geometry_parts["right_params"].append(snapshot.ellipse_params[:, 1, ...])
            geometry_parts["left_success"].append(snapshot.ellipse_success[:, 0, ...])
            geometry_parts["right_success"].append(snapshot.ellipse_success[:, 1, ...])
        _verify_receipt_geometry_payloads(context, geometry_parts)
    return canonical


def _load_validated_staged_frame_indices(
    context: EyeAngleInputContext,
    receipt: Mapping[str, Any],
) -> np.ndarray:
    canonical = _validate_staged_eye_angle_input_integrity_receipt(
        context,
        receipt,
        verify_payload=False,
    )
    values = _owned_c_array(
        context.frame_indices_source[context.frame_indices_key][:],
        dtype=np.int64,
    )
    if values.shape != (int(canonical["row_count"]),):
        raise ValueError(
            "Staged full source_acquisition_frame_index snapshot has the wrong shape."
        )
    for chunk in canonical["chunks"]:
        start_row = int(chunk["start_row"])
        stop_row = int(chunk["stop_row"])
        expected = chunk["logical_inputs"]["source_acquisition_frame_index"]
        observed = _owned_c_array(values[start_row:stop_row], dtype=np.int64)
        if (
            [int(value) for value in observed.shape] != expected["shape"]
            or np.dtype(observed.dtype).str != expected["dtype"]
            or array_values_sha256(observed) != expected["content_sha256"]
        ):
            raise ValueError(
                "Staged full source_acquisition_frame_index snapshot differs from "
                "its chunk integrity receipt."
            )
    return values


def _prepare_base_output_arrays(
    run_group: zarr.Group,
    *,
    total_detections: int,
    chunk_len: int,
    storage_plan: AnalysisStoragePlanReceipt | None = None,
) -> None:
    angles_group = run_group.require_group("angles")
    roi_group = angles_group.require_group("roi")
    qa_group = run_group.require_group("qa")
    qa_roi = qa_group.require_group("roi")
    support_group = run_group.require_group("support")
    body_frame_group = support_group.require_group("body_frame")
    storage_entries = eye_angle_storage_entries_by_path(storage_plan)

    _prepare_output_arrays(
        roi_group,
        [(name, (total_detections,), (chunk_len,), "f4") for name, _field in _BASE_ROI_RESULT_FIELDS],
    )
    _prepare_output_arrays(
        roi_group,
        [
            ("left_gaze_xy", (total_detections, 2), (chunk_len, 2), "f4"),
            ("right_gaze_xy", (total_detections, 2), (chunk_len, 2), "f4"),
        ],
    )
    _prepare_output_arrays(
        qa_roi,
        [
            ("valid_left", (total_detections,), (chunk_len,), "bool"),
            ("valid_right", (total_detections,), (chunk_len,), "bool"),
            ("valid_frame", (total_detections,), (chunk_len,), "bool"),
            ("reason_codes", (total_detections,), (chunk_len,), "u2"),
            ("left_major_axis_marginal", (total_detections,), (chunk_len,), "bool"),
            ("right_major_axis_marginal", (total_detections,), (chunk_len,), "bool"),
            ("major_axis_marginal", (total_detections,), (chunk_len,), "bool"),
        ],
    )
    _prepare_output_arrays(
        support_group,
        [
            ("instance_key", (total_detections,), (chunk_len,), "u8"),
            (
                "source_acquisition_frame_index",
                (total_detections,),
                (chunk_len,),
                "i8",
            ),
            ("frame_indices", (total_detections,), (chunk_len,), "i8"),
            ("time_seconds", (total_detections,), (chunk_len,), "f4"),
            ("ellipse_major", (total_detections,), (chunk_len,), "f4"),
            ("ellipse_minor", (total_detections,), (chunk_len,), "f4"),
            ("ellipse_ratio", (total_detections,), (chunk_len,), "f4"),
        ],
        storage_entries=storage_entries,
        path_prefix="support",
    )
    _prepare_output_arrays(
        body_frame_group,
        [
            ("origin_xy", (total_detections, 2), (chunk_len, 2), "f4"),
            ("forward_axis_xy", (total_detections, 2), (chunk_len, 2), "f4"),
            ("left_axis_xy", (total_detections, 2), (chunk_len, 2), "f4"),
            ("heading_deg", (total_detections,), (chunk_len,), "f4"),
            ("valid", (total_detections,), (chunk_len,), "bool"),
            ("failure_reason_bytes", (total_detections, 64), (chunk_len, 64), "u1"),
        ],
        storage_entries=storage_entries,
        path_prefix="support/body_frame",
    )
    body_frame_group.attrs.update(
        build_keypoint_body_frame_contract_attrs(
            source_refined_keypoints_run=None,
            coordinate_space=BODY_FRAME_COORDINATE_SPACE_ROI,
        )
    )
    body_frame_group.attrs["reason_encoding"] = REASON_BYTES_ENCODING
    body_frame_group.attrs["reason_bytes_width"] = REASON_BYTES_MIN_WIDTH
    body_frame_group.attrs["reason_bytes_null_terminated"] = True
    support_group["instance_key"].attrs.update(
        {
            "identity_domain": "observation_instance",
            "identity_mode": "instance_key",
            "row_axis": EYE_ANGLE_ROW_AXIS,
        }
    )
    support_group["source_acquisition_frame_index"].attrs.update(
        {
            "value_kind": "source_acquisition_frame_index",
            "row_axis": EYE_ANGLE_ROW_AXIS,
        }
    )
    support_group["frame_indices"].attrs.update(
        {
            "compatibility_alias_of": "support/source_acquisition_frame_index",
            "values_must_equal_canonical": True,
            "value_kind": "source_acquisition_frame_index",
            "row_axis": EYE_ANGLE_ROW_AXIS,
        }
    )


def _write_base_eye_angle_result(
    run_group: zarr.Group,
    row_slice: slice,
    result: EyeAngleResults,
    *,
    frame_indices: np.ndarray,
    instance_key: np.ndarray,
    time_seconds: np.ndarray,
) -> None:
    roi_group = run_group["angles"]["roi"]
    qa_roi = run_group["qa"]["roi"]
    support_group = run_group["support"]
    body_frame_group = support_group["body_frame"]

    for dataset_name, field_name in _BASE_ROI_RESULT_FIELDS:
        roi_group[dataset_name][row_slice] = getattr(result, field_name)
    roi_group["left_gaze_xy"][row_slice, :] = result.left_gaze_xy
    roi_group["right_gaze_xy"][row_slice, :] = result.right_gaze_xy
    for dataset_name, field_name in _BASE_QA_RESULT_FIELDS:
        qa_roi[dataset_name][row_slice] = getattr(result, field_name)
    support_group["instance_key"][row_slice] = instance_key
    support_group["source_acquisition_frame_index"][row_slice] = frame_indices
    support_group["frame_indices"][row_slice] = frame_indices
    support_group["time_seconds"][row_slice] = time_seconds
    for dataset_name, field_name in _BASE_SUPPORT_RESULT_FIELDS:
        support_group[dataset_name][row_slice] = getattr(result, field_name)
    body_frame_group["origin_xy"][row_slice] = result.body_frame_origin_xy
    body_frame_group["forward_axis_xy"][row_slice] = result.body_frame_forward_axis_xy
    body_frame_group["left_axis_xy"][row_slice] = result.body_frame_left_axis_xy
    body_frame_group["heading_deg"][row_slice] = result.heading_deg
    body_frame_group["valid"][row_slice] = result.body_frame_valid
    body_frame_group["failure_reason_bytes"][row_slice, :] = result.body_frame_failure_reason_bytes


def _process_and_write_eye_angle_chunk_groups(
    context: EyeAngleInputContext,
    run_group: zarr.Group,
    *,
    start_row: int,
    stop_row: int,
    chunk_index: int,
    fps: Optional[float],
    execution_backend: str,
    _staged_input_integrity_chunk: Optional[Mapping[str, Any]] = None,
) -> dict[str, object]:
    chunk_start = time.perf_counter()
    row_slice = slice(int(start_row), int(stop_row))
    timing: dict[str, object] = {
        "chunk_index": int(chunk_index),
        "start_row": int(start_row),
        "stop_row": int(stop_row),
        "row_count": int(stop_row - start_row),
        "execution_backend": execution_backend,
    }

    phase_start = time.perf_counter()
    integrity_chunk = (
        _canonical_chunk_integrity_record(_staged_input_integrity_chunk)
        if _staged_input_integrity_chunk is not None
        else None
    )
    if integrity_chunk is not None and (
        integrity_chunk["chunk_index"] != int(chunk_index)
        or integrity_chunk["start_row"] != int(start_row)
        or integrity_chunk["stop_row"] != int(stop_row)
    ):
        raise ValueError(
            "Staged eye-angle worker received an integrity receipt for another chunk."
        )
    snapshot = _load_eye_angle_chunk_input_snapshot(
        context,
        start_row=int(start_row),
        stop_row=int(stop_row),
    )
    chunk_receipt_sha256 = (
        _validate_chunk_snapshot_against_receipt(snapshot, integrity_chunk)
        if integrity_chunk is not None
        else None
    )
    timing["read_seconds"] = float(time.perf_counter() - phase_start)

    phase_start = time.perf_counter()
    chunk_result = _process_chunk(
        ellipse_params=snapshot.ellipse_params,
        ellipse_success=snapshot.ellipse_success,
        keypoints_roi=snapshot.keypoints_roi,
        detection_success=snapshot.detection_success,
        keypoint_indices=context.keypoint_indices,
    )
    timing["compute_seconds"] = float(time.perf_counter() - phase_start)

    if fps:
        chunk_time_seconds = (
            snapshot.source_acquisition_frame_index.astype(np.float64) / float(fps)
        ).astype(np.float32, copy=False)
    else:
        chunk_time_seconds = np.full(
            snapshot.source_acquisition_frame_index.shape,
            np.nan,
            dtype=np.float32,
        )

    phase_start = time.perf_counter()
    _write_base_eye_angle_result(
        run_group,
        row_slice,
        chunk_result,
        frame_indices=snapshot.source_acquisition_frame_index,
        instance_key=snapshot.instance_key,
        time_seconds=chunk_time_seconds,
    )
    timing["write_seconds"] = float(time.perf_counter() - phase_start)
    timing["valid_frame_count"] = int(chunk_result.valid_frame.sum())
    timing["total_seconds"] = float(time.perf_counter() - chunk_start)
    return {
        "chunk_timing": timing,
        "valid_frame_count": int(chunk_result.valid_frame.sum()),
        "staged_input_chunk_receipt_sha256": chunk_receipt_sha256,
    }


def _process_and_write_eye_angle_chunk(
    zarr_path: str,
    *,
    subject_shape_run: Optional[str],
    refined_subject_run: Optional[str],
    keypoint_run: Optional[str],
    diagnostic_refined_keypoint_run: Optional[str],
    eye_angle_run: str,
    start_row: int,
    stop_row: int,
    chunk_index: int,
    fps: Optional[float],
    _staged_input_integrity_receipt: Optional[Mapping[str, Any]] = None,
) -> dict[str, object]:
    staged_subject_shape_authority = (
        _staged_subject_shape_authority_from_input_receipt(
            _staged_input_integrity_receipt
        )
        if _staged_input_integrity_receipt is not None
        else None
    )
    staged_keypoint_authority = (
        _staged_keypoint_authority_from_input_receipt(
            _staged_input_integrity_receipt
        )
        if _staged_input_integrity_receipt is not None
        else None
    )
    root = open_zarr_root(zarr_path, mode="a")
    context = _resolve_eye_angle_inputs(
        root,
        subject_shape_run=subject_shape_run,
        refined_subject_run=refined_subject_run,
        keypoint_run=keypoint_run,
        diagnostic_refined_keypoint_run=diagnostic_refined_keypoint_run,
        _staged_subject_shape_authority=staged_subject_shape_authority,
        _staged_keypoint_authority=staged_keypoint_authority,
        _verify_staged_payload=(staged_subject_shape_authority is None),
    )
    staged_receipt = (
        _validate_staged_eye_angle_input_integrity_receipt(
            context,
            _staged_input_integrity_receipt,
            verify_payload=False,
        )
        if _staged_input_integrity_receipt is not None
        else None
    )
    integrity_chunk = (
        staged_receipt["chunks"][int(chunk_index)]
        if staged_receipt is not None
        and 0 <= int(chunk_index) < len(staged_receipt["chunks"])
        else None
    )
    if staged_receipt is not None and integrity_chunk is None:
        raise ValueError("Staged eye-angle worker chunk is absent from its receipt.")
    run_group = root["analysis"]["eye_angle_runs"][eye_angle_run]
    return _process_and_write_eye_angle_chunk_groups(
        context,
        run_group,
        start_row=start_row,
        stop_row=stop_row,
        chunk_index=chunk_index,
        fps=fps,
        execution_backend=DASK_WORKER_EXECUTION_BACKEND,
        _staged_input_integrity_chunk=integrity_chunk,
    )


def _compute_dask_tasks(
    tasks: Sequence[object],
    *,
    scheduler_key: str,
    num_workers: Optional[int],
) -> list[dict[str, object]]:
    if not tasks:
        return []
    cluster = None
    client = None
    try:
        if scheduler_key == "distributed":
            if not HAVE_DISTRIBUTED:
                raise RuntimeError(
                    "Dask distributed is not available. Install dask[distributed] or choose a different scheduler."
                )
            cluster_kwargs: dict[str, object] = {}
            if num_workers is not None:
                cluster_kwargs["n_workers"] = int(num_workers)
            cluster = LocalCluster(**cluster_kwargs)
            client = Client(cluster)
            results = list(client.gather(client.compute(list(tasks))))
        else:
            compute_kwargs: dict[str, object] = {"scheduler": scheduler_key}
            if num_workers is not None and scheduler_key != "single-threaded":
                compute_kwargs["num_workers"] = int(num_workers)
            results = list(dask.compute(*tasks, **compute_kwargs))
    finally:
        if client is not None:
            client.close()
        if cluster is not None:
            cluster.close()
    return [dict(result) for result in results]


def _project_detection_arrays_to_frames(
    frame_indices: np.ndarray,
    *,
    num_frames: int,
    valid_frame: np.ndarray,
    reason_codes: np.ndarray,
    arrays: Dict[str, np.ndarray],
) -> tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    frame_arrays = {name: np.full(num_frames, np.nan, dtype=np.float32) for name in arrays}
    frame_valid = np.zeros(num_frames, dtype=bool)
    frame_reason = np.zeros(num_frames, dtype=np.uint16)
    if num_frames <= 0:
        return frame_arrays, frame_valid, frame_reason

    valid_index_mask = (frame_indices >= 0) & (frame_indices < num_frames)
    valid_indices = frame_indices[valid_index_mask]
    counts = np.bincount(valid_indices, minlength=num_frames) if valid_indices.size else np.zeros(num_frames, dtype=np.int64)
    frame_reason[counts == 0] |= REASON_NO_DETECTION
    frame_reason[counts > 1] |= REASON_MULTI_DETECTION

    detection_indices = np.nonzero(valid_index_mask)[0]
    unique_detection_indices = detection_indices[counts[frame_indices[detection_indices]] == 1]
    unique_frames = frame_indices[unique_detection_indices]
    if unique_detection_indices.size:
        for name, values in arrays.items():
            frame_arrays[name][unique_frames] = values[unique_detection_indices]
        frame_valid[unique_frames] = valid_frame[unique_detection_indices]
        frame_reason[unique_frames] |= reason_codes[unique_detection_indices]
    return frame_arrays, frame_valid, frame_reason


def _project_detection_bool_to_frames(
    frame_indices: np.ndarray,
    *,
    num_frames: int,
    values: np.ndarray,
) -> np.ndarray:
    """Project row-aligned booleans to frames only when a frame has one detection row."""
    frame_values = np.zeros(num_frames, dtype=bool)
    if num_frames <= 0:
        return frame_values
    valid_index_mask = (frame_indices >= 0) & (frame_indices < num_frames)
    valid_indices = frame_indices[valid_index_mask]
    if not valid_indices.size:
        return frame_values
    counts = np.bincount(valid_indices, minlength=num_frames)
    detection_indices = np.nonzero(valid_index_mask)[0]
    unique_detection_indices = detection_indices[counts[frame_indices[detection_indices]] == 1]
    unique_frames = frame_indices[unique_detection_indices]
    if unique_detection_indices.size:
        frame_values[unique_frames] = np.asarray(values, dtype=bool)[unique_detection_indices]
    return frame_values


def _is_selector_eligible_eye_angle_output(
    *,
    diagnostic_output: bool,
    staged_input_integrity_receipt: Optional[Mapping[str, Any]],
    output_layout: str,
    storage_candidate: bool = False,
) -> bool:
    """Allow canonical activation only for the maintained compact-v7 layout."""

    return (
        not diagnostic_output
        and staged_input_integrity_receipt is None
        and output_layout == EYE_ANGLE_LAYOUT_COMPACT_DENSE_V2
        and not storage_candidate
    )


def run(
    args: argparse.Namespace,
    *,
    _staged_input_integrity_receipt: Optional[Mapping[str, Any]] = None,
) -> None:
    console = Console()
    root = _open_archive_for_eye_angle(args.zarr_path)

    backend = _normalize_execution_backend(args.execution_backend)
    scheduler_key = _normalize_scheduler(args.scheduler)
    staged_subject_shape_authority = (
        _staged_subject_shape_authority_from_input_receipt(
            _staged_input_integrity_receipt
        )
        if _staged_input_integrity_receipt is not None
        else None
    )
    staged_keypoint_authority = (
        _staged_keypoint_authority_from_input_receipt(
            _staged_input_integrity_receipt
        )
        if _staged_input_integrity_receipt is not None
        else None
    )
    context = _resolve_eye_angle_inputs(
        root,
        subject_shape_run=args.subject_shape_run,
        refined_subject_run=args.refined_subject_run,
        keypoint_run=args.keypoint_run,
        diagnostic_refined_keypoint_run=args.diagnostic_refined_keypoint_run,
        _staged_subject_shape_authority=staged_subject_shape_authority,
        _staged_keypoint_authority=staged_keypoint_authority,
        _verify_staged_payload=True,
    )
    staged_input_integrity_receipt = (
        _validate_staged_eye_angle_input_integrity_receipt(
            context,
            _staged_input_integrity_receipt,
            verify_payload=True,
        )
        if _staged_input_integrity_receipt is not None
        else None
    )
    eye_geometry = context.eye_geometry
    initial_input_identity = _resolved_eye_angle_input_identity(context)
    source_authority_mode = str(
        getattr(eye_geometry, "source_authority_mode", None)
        or "canonical_publication"
    )
    keypoint_run_name = context.keypoint_run_name
    total_detections = int(eye_geometry.ellipse_params.shape[0])
    chunk_size = max(1, int(args.chunk_size))
    if total_detections and chunk_size > total_detections:
        chunk_size = total_detections

    chunks = _row_chunks(total_detections, chunk_size)
    if staged_input_integrity_receipt is not None:
        receipt_chunks = [
            (int(chunk["start_row"]), int(chunk["stop_row"]))
            for chunk in staged_input_integrity_receipt["chunks"]
        ]
        if receipt_chunks != chunks:
            raise ValueError(
                "Staged eye-angle integrity receipt chunking differs from writer chunking."
            )
        frame_indices = _load_validated_staged_frame_indices(
            context,
            staged_input_integrity_receipt,
        )
    else:
        frame_indices = context.frame_indices_source[context.frame_indices_key][
            :
        ].astype(
            np.int64,
            copy=False,
        )
    if frame_indices.shape[0] != total_detections:
        raise ValueError("Mismatch between frame_indices and detection count.")

    if staged_input_integrity_receipt is not None:
        staged_parameters = staged_input_integrity_receipt["scientific_parameters"]
        expected_fps = staged_parameters["fps"]
        supplied_fps = (
            None if args.fps is None else float(args.fps)
        )
        if supplied_fps != expected_fps:
            raise ValueError(
                "Staged eye-angle FPS differs from its sealed materialization plan."
            )
        fps = None if expected_fps is None else float(expected_fps)
    else:
        fps = args.fps or get_fps(root)
        if fps is None or fps <= 0:
            fps = None
    smoothing_window_param = args.smoothing_window
    valid_frame_index_mask = frame_indices >= 0
    if context.source_total_frames is not None:
        num_frames = int(context.source_total_frames)
        if np.any(frame_indices[valid_frame_index_mask] >= num_frames):
            raise ValueError(
                "Canonical acquisition-frame indices exceed their sealed full-video extent."
            )
    else:
        num_frames = (
            int(frame_indices[valid_frame_index_mask].max() + 1)
            if np.any(valid_frame_index_mask)
            else 0
        )
    chunk_len = min(chunk_size, total_detections) if total_detections else 1
    frame_chunk = min(chunk_size, num_frames) if num_frames else 1

    if args.run_name:
        resolved_run_name = args.run_name
    else:
        resolved_run_name = datetime.now(timezone.utc).strftime("eye_angle_%Y%m%d_%H%M%S")

    output_layout = str(args.layout)
    storage_profile_id = str(
        getattr(args, "storage_profile", EYE_ANGLE_LEGACY_EXPLICIT_STORAGE)
    )
    storage_candidate = is_eye_angle_storage_candidate(storage_profile_id)
    storage_plan: AnalysisStoragePlanReceipt | None = None
    if storage_candidate:
        if output_layout != EYE_ANGLE_LAYOUT_COMPACT_DENSE_V2:
            raise ValueError(
                "The eye-angle byte-planned candidate requires compact_dense_v2."
            )
        if backend != SERIAL_EXECUTION_BACKEND:
            raise ValueError(
                "The eye-angle byte-planned candidate requires serial_driver so "
                "one writer owns every complete physical shard."
            )
        storage_plan = build_eye_angle_candidate_storage_plan(
            EyeAngleDimensions(
                n_roi_rows=int(total_detections),
                n_frames=int(num_frames),
                angle_block_width=int(args.dense_chunk_columns),
            )
        )

    analysis_group = root.require_group("analysis")
    parent_group = require_runs_parent(analysis_group, "eye_angle_runs")
    if resolved_run_name in parent_group:
        raise ValueError(f"Run '{resolved_run_name}' already exists in analysis/eye_angle_runs.")

    diagnostic_output = (
        context.keypoint_source_mode
        == EYE_ANGLE_KEYPOINT_SOURCE_REFINED_DIAGNOSTIC
    )
    initial_publication_attrs: dict[str, Any] = {
        # A run becomes selector eligible only after every source recheck,
        # output/provenance write, and completion marker has persisted.  This
        # remains false forever for diagnostic and staged-local outputs.
        "stage_selector_eligible": False,
        "keypoint_source_mode": context.keypoint_source_mode,
    }
    if storage_candidate:
        initial_publication_attrs.update(
            {
                EYE_ANGLE_STORAGE_CANDIDATE_ATTR: {
                    "profile_id": EYE_ANGLE_ACCESS_AWARE_CANDIDATE_PROFILE_ID,
                    "status": "selector_ineligible_candidate",
                    "activation_allowed": False,
                    "whole_shard_write_ownership": "single_serial_writer",
                },
                "publication_scope": "storage_benchmark_candidate_only",
            }
        )
    if diagnostic_output:
        initial_publication_attrs.update(
            {
                "coordinate_contract": (
                    EYE_ANGLE_REFINED_DIAGNOSTIC_COORDINATE_CONTRACT
                ),
                "legacy_unverified_diagnostic_output": True,
                "publication_scope": "historical_diagnostic_only",
            }
        )
    run_group = parent_group.create_group(
        resolved_run_name,
        attributes=initial_publication_attrs,
    )
    mark_run_started(run_group, run_name=resolved_run_name, stage="eye_angle")
    run_group.attrs["status"] = "running"
    run_group.attrs["layout"] = output_layout
    if output_layout != EYE_ANGLE_LAYOUT_COMPACT_DENSE_V2:
        run_group.attrs.update(
            {
                "publication_scope": "legacy_compatibility_only",
                "legacy_storage_layout": True,
            }
        )
    run_group.attrs["execution_backend"] = backend
    run_group.attrs["keypoint_source_mode"] = context.keypoint_source_mode
    run_group.attrs["source_eye_geometry_authority_mode"] = source_authority_mode
    if staged_input_integrity_receipt is not None:
        run_group.attrs["staged_input_integrity_receipt_sha256"] = (
            staged_input_integrity_receipt["record_sha256"]
        )
        run_group.attrs["staged_input_integrity_scope"] = (
            staged_input_integrity_receipt["integrity_scope"]
        )
    run_group.attrs["dask_scheduler"] = scheduler_key
    run_group.attrs["dask_num_workers"] = int(args.num_workers) if args.num_workers is not None else None
    if not args.quiet:
        console.print(f"Created run group: [cyan]analysis/eye_angle_runs/{resolved_run_name}[/cyan]")

    _prepare_base_output_arrays(
        run_group,
        total_detections=total_detections,
        chunk_len=chunk_len,
        storage_plan=storage_plan,
    )
    run_group["support"]["body_frame"].attrs.update(
        build_keypoint_body_frame_contract_attrs(
            source_keypoints_run=(
                keypoint_run_name if not diagnostic_output else None
            ),
            source_refined_keypoints_run=(
                keypoint_run_name if diagnostic_output else None
            ),
            coordinate_space=BODY_FRAME_COORDINATE_SPACE_ROI,
        )
    )
    run_group["support"]["body_frame"].attrs.update(
        {
            "resolved_keypoint_indices": {
                key: int(value) for key, value in context.keypoint_indices.items()
            },
            "source_keypoints_roi_path": f"{context.kp_group_path}/keypoints_roi",
            "source_detection_success_path": context.detection_success_path,
        }
    )
    chunk_timings: list[dict[str, object]] = []
    stage_start = time.perf_counter()

    if backend == DASK_WORKER_EXECUTION_BACKEND:
        worker_zarr_path = str(args.zarr_path.expanduser().resolve())
        worker_refined_subject_run = (
            eye_geometry.run_name if eye_geometry.stage_group == EYE_GEOMETRY_STAGE_REFINED_SUBJECT else None
        )
        worker_subject_shape_run = (
            eye_geometry.run_name if eye_geometry.stage_group == EYE_GEOMETRY_STAGE_SUBJECT_SHAPE else None
        )
        worker_staged_input_integrity_receipt = (
            _canonical_json_copy(staged_input_integrity_receipt)
            if staged_input_integrity_receipt is not None
            else None
        )
        tasks = [
            delayed(_process_and_write_eye_angle_chunk)(
                worker_zarr_path,
                subject_shape_run=worker_subject_shape_run,
                refined_subject_run=worker_refined_subject_run,
                keypoint_run=(
                    keypoint_run_name
                    if context.keypoint_source_mode
                    == EYE_ANGLE_KEYPOINT_SOURCE_CANONICAL
                    else None
                ),
                diagnostic_refined_keypoint_run=(
                    keypoint_run_name
                    if context.keypoint_source_mode
                    == EYE_ANGLE_KEYPOINT_SOURCE_REFINED_DIAGNOSTIC
                    else None
                ),
                eye_angle_run=resolved_run_name,
                start_row=start_row,
                stop_row=stop_row,
                chunk_index=chunk_index,
                fps=fps,
                _staged_input_integrity_receipt=(
                    worker_staged_input_integrity_receipt
                ),
            )
            for chunk_index, (start_row, stop_row) in enumerate(chunks)
        ]
        results = _compute_dask_tasks(tasks, scheduler_key=scheduler_key, num_workers=args.num_workers)
    else:
        results = [
            _process_and_write_eye_angle_chunk_groups(
                context,
                run_group,
                start_row=start_row,
                stop_row=stop_row,
                chunk_index=chunk_index,
                fps=fps,
                execution_backend=SERIAL_EXECUTION_BACKEND,
                _staged_input_integrity_chunk=(
                    staged_input_integrity_receipt["chunks"][chunk_index]
                    if staged_input_integrity_receipt is not None
                    else None
                ),
            )
            for chunk_index, (start_row, stop_row) in enumerate(chunks)
        ]
    if staged_input_integrity_receipt is not None:
        expected_chunk_receipts = [
            str(chunk["record_sha256"])
            for chunk in staged_input_integrity_receipt["chunks"]
        ]
        observed_chunk_receipts = [
            result.get("staged_input_chunk_receipt_sha256") for result in results
        ]
        if (
            len(observed_chunk_receipts) != len(expected_chunk_receipts)
            or len(set(observed_chunk_receipts)) != len(observed_chunk_receipts)
            or sorted(str(value) for value in observed_chunk_receipts)
            != sorted(expected_chunk_receipts)
        ):
            raise RuntimeError(
                "Staged eye-angle workers did not attest the exact complete chunk "
                "integrity receipt set."
            )
    for result in sorted(results, key=lambda item: int(dict(item["chunk_timing"]).get("chunk_index") or 0)):
        chunk_timings.append(dict(result["chunk_timing"]))
    phase_seconds: dict[str, float] = {
        "parallel_chunk_compute_and_base_write": float(
            time.perf_counter() - stage_start
        )
    }
    derived_materialization_started = time.perf_counter()

    roi_group = run_group["angles"]["roi"]
    qa_roi = run_group["qa"]["roi"]
    support_group = run_group["support"]
    left_angles = roi_group["left_deg"][:]
    right_angles = roi_group["right_deg"][:]
    left_signed = roi_group["left_signed_deg"][:]
    right_signed = roi_group["right_signed_deg"][:]
    left_major_signed = roi_group["left_major_signed_deg"][:]
    right_major_signed = roi_group["right_major_signed_deg"][:]
    left_eye_angle = roi_group["left_eye_angle_deg"][:]
    right_eye_angle = roi_group["right_eye_angle_deg"][:]
    vergence_eye_angle = roi_group["vergence_eye_angle_deg"][:]
    left_minor_signed = roi_group["left_minor_signed_deg"][:]
    right_minor_signed = roi_group["right_minor_signed_deg"][:]
    vergence = roi_group["vergence_deg"][:]
    vergence_signed = roi_group["vergence_signed_deg"][:]
    vergence_major_signed = roi_group["vergence_major_signed_deg"][:]
    vergence_minor_signed = roi_group["vergence_minor_signed_deg"][:]
    version = roi_group["version_deg"][:]
    version_major = roi_group["version_major_deg"][:]
    version_minor = roi_group["version_minor_deg"][:]
    left_gaze = roi_group["left_gaze_deg"][:]
    right_gaze = roi_group["right_gaze_deg"][:]
    left_gaze_signed = roi_group["left_gaze_signed_deg"][:]
    right_gaze_signed = roi_group["right_gaze_signed_deg"][:]
    vergence_gaze = roi_group["vergence_gaze_deg"][:]
    vergence_gaze_signed = roi_group["vergence_gaze_signed_deg"][:]
    left_nasal_gaze = roi_group["left_nasal_gaze_deg"][:]
    right_nasal_gaze = roi_group["right_nasal_gaze_deg"][:]
    mean_eye_vergence_gaze = roi_group["mean_eye_vergence_gaze_deg"][:]
    version_gaze = roi_group["version_gaze_deg"][:]
    heading_deg_out = roi_group["heading_deg"][:]
    left_centroid = roi_group["left_centroid_deg"][:]
    right_centroid = roi_group["right_centroid_deg"][:]
    vergence_centroid = roi_group["vergence_centroid_deg"][:]
    valid_left = qa_roi["valid_left"][:]
    valid_right = qa_roi["valid_right"][:]
    valid_frame = qa_roi["valid_frame"][:]
    reason_codes = qa_roi["reason_codes"][:]
    major_axis_marginal = qa_roi["major_axis_marginal"][:]
    time_seconds = support_group["time_seconds"][:]
    ellipse_major = support_group["ellipse_major"][:]
    ellipse_minor = support_group["ellipse_minor"][:]
    ellipse_ratio = support_group["ellipse_ratio"][:]

    left_speed = (
        _compute_derivative(left_angles, time_seconds, valid_left, max_dt=DERIVATIVE_MAX_DT)
        if fps
        else np.full_like(left_angles, np.nan)
    )
    right_speed = (
        _compute_derivative(right_angles, time_seconds, valid_right, max_dt=DERIVATIVE_MAX_DT)
        if fps
        else np.full_like(right_angles, np.nan)
    )
    vergence_speed = (
        _compute_derivative(vergence, time_seconds, valid_frame, max_dt=DERIVATIVE_MAX_DT)
        if fps
        else np.full_like(vergence, np.nan)
    )
    vergence_signed_speed = (
        _compute_derivative(vergence_signed, time_seconds, valid_frame, max_dt=DERIVATIVE_MAX_DT)
        if fps
        else np.full_like(vergence_signed, np.nan)
    )
    version_speed = (
        _compute_derivative(version, time_seconds, valid_frame, max_dt=DERIVATIVE_MAX_DT)
        if fps
        else np.full_like(version, np.nan)
    )
    left_gaze_speed = (
        _compute_derivative(left_gaze, time_seconds, valid_left, max_dt=DERIVATIVE_MAX_DT)
        if fps
        else np.full_like(left_gaze, np.nan)
    )
    right_gaze_speed = (
        _compute_derivative(right_gaze, time_seconds, valid_right, max_dt=DERIVATIVE_MAX_DT)
        if fps
        else np.full_like(right_gaze, np.nan)
    )
    vergence_gaze_speed = (
        _compute_derivative(vergence_gaze, time_seconds, valid_frame, max_dt=DERIVATIVE_MAX_DT)
        if fps
        else np.full_like(vergence_gaze, np.nan)
    )
    vergence_gaze_signed_speed = (
        _compute_derivative(vergence_gaze_signed, time_seconds, valid_frame, max_dt=DERIVATIVE_MAX_DT)
        if fps
        else np.full_like(vergence_gaze_signed, np.nan)
    )
    version_gaze_speed = (
        _compute_derivative(version_gaze, time_seconds, valid_frame, max_dt=DERIVATIVE_MAX_DT)
        if fps
        else np.full_like(version_gaze, np.nan)
    )
    mean_eye_vergence_gaze_speed = (
        _compute_derivative(mean_eye_vergence_gaze, time_seconds, valid_frame, max_dt=DERIVATIVE_MAX_DT)
        if fps
        else np.full_like(mean_eye_vergence_gaze, np.nan)
    )

    left_accel = (
        _compute_derivative(left_speed, time_seconds, np.isfinite(left_speed), max_dt=DERIVATIVE_MAX_DT)
        if fps
        else np.full_like(left_angles, np.nan)
    )
    right_accel = (
        _compute_derivative(right_speed, time_seconds, np.isfinite(right_speed), max_dt=DERIVATIVE_MAX_DT)
        if fps
        else np.full_like(right_angles, np.nan)
    )
    vergence_accel = (
        _compute_derivative(vergence_speed, time_seconds, np.isfinite(vergence_speed), max_dt=DERIVATIVE_MAX_DT)
        if fps
        else np.full_like(vergence, np.nan)
    )
    vergence_signed_accel = (
        _compute_derivative(vergence_signed_speed, time_seconds, np.isfinite(vergence_signed_speed), max_dt=DERIVATIVE_MAX_DT)
        if fps
        else np.full_like(vergence_signed, np.nan)
    )
    version_accel = (
        _compute_derivative(version_speed, time_seconds, np.isfinite(version_speed), max_dt=DERIVATIVE_MAX_DT)
        if fps
        else np.full_like(version, np.nan)
    )
    left_gaze_accel = (
        _compute_derivative(left_gaze_speed, time_seconds, np.isfinite(left_gaze_speed), max_dt=DERIVATIVE_MAX_DT)
        if fps
        else np.full_like(left_gaze, np.nan)
    )
    right_gaze_accel = (
        _compute_derivative(right_gaze_speed, time_seconds, np.isfinite(right_gaze_speed), max_dt=DERIVATIVE_MAX_DT)
        if fps
        else np.full_like(right_gaze, np.nan)
    )
    vergence_gaze_accel = (
        _compute_derivative(vergence_gaze_speed, time_seconds, np.isfinite(vergence_gaze_speed), max_dt=DERIVATIVE_MAX_DT)
        if fps
        else np.full_like(vergence_gaze, np.nan)
    )
    vergence_gaze_signed_accel = (
        _compute_derivative(
            vergence_gaze_signed_speed,
            time_seconds,
            np.isfinite(vergence_gaze_signed_speed),
            max_dt=DERIVATIVE_MAX_DT,
        )
        if fps
        else np.full_like(vergence_gaze_signed, np.nan)
    )
    version_gaze_accel = (
        _compute_derivative(version_gaze_speed, time_seconds, np.isfinite(version_gaze_speed), max_dt=DERIVATIVE_MAX_DT)
        if fps
        else np.full_like(version_gaze, np.nan)
    )
    mean_eye_vergence_gaze_accel = (
        _compute_derivative(
            mean_eye_vergence_gaze_speed,
            time_seconds,
            np.isfinite(mean_eye_vergence_gaze_speed),
            max_dt=DERIVATIVE_MAX_DT,
        )
        if fps
        else np.full_like(mean_eye_vergence_gaze, np.nan)
    )

    window_setting = smoothing_window_param if smoothing_window_param is not None else ANGLE_SMOOTHING_WINDOW
    detection_smooth_window = _resolve_smoothing_window(total_detections, window_setting)
    if detection_smooth_window:
        left_smoothed = _smooth_signal(left_angles, detection_smooth_window).astype(np.float32, copy=False)
        right_smoothed = _smooth_signal(right_angles, detection_smooth_window).astype(np.float32, copy=False)
        vergence_smoothed = _smooth_signal(vergence, detection_smooth_window).astype(np.float32, copy=False)
        left_signed_smoothed = _smooth_signal(left_signed, detection_smooth_window).astype(np.float32, copy=False)
        right_signed_smoothed = _smooth_signal(right_signed, detection_smooth_window).astype(np.float32, copy=False)
        vergence_signed_smoothed = _smooth_signal(vergence_signed, detection_smooth_window).astype(np.float32, copy=False)
        version_smoothed = _smooth_signal(version, detection_smooth_window).astype(np.float32, copy=False)
        left_eye_angle_smoothed = _smooth_signal(left_eye_angle, detection_smooth_window).astype(np.float32, copy=False)
        right_eye_angle_smoothed = _smooth_signal(right_eye_angle, detection_smooth_window).astype(np.float32, copy=False)
        vergence_eye_angle_smoothed = _smooth_signal(vergence_eye_angle, detection_smooth_window).astype(np.float32, copy=False)
        left_minor_signed_smoothed = _smooth_signal(left_minor_signed, detection_smooth_window).astype(np.float32, copy=False)
        right_minor_signed_smoothed = _smooth_signal(right_minor_signed, detection_smooth_window).astype(np.float32, copy=False)
        vergence_minor_signed_smoothed = _smooth_signal(vergence_minor_signed, detection_smooth_window).astype(np.float32, copy=False)
        version_minor_smoothed = _smooth_signal(version_minor, detection_smooth_window).astype(np.float32, copy=False)
        left_gaze_smoothed = _smooth_signal(left_gaze, detection_smooth_window).astype(np.float32, copy=False)
        right_gaze_smoothed = _smooth_signal(right_gaze, detection_smooth_window).astype(np.float32, copy=False)
        left_gaze_signed_smoothed = _smooth_signal(left_gaze_signed, detection_smooth_window).astype(np.float32, copy=False)
        right_gaze_signed_smoothed = _smooth_signal(right_gaze_signed, detection_smooth_window).astype(np.float32, copy=False)
        vergence_gaze_smoothed = _smooth_signal(vergence_gaze, detection_smooth_window).astype(np.float32, copy=False)
        vergence_gaze_signed_smoothed = _smooth_signal(vergence_gaze_signed, detection_smooth_window).astype(np.float32, copy=False)
        left_nasal_gaze_smoothed = _smooth_signal(left_nasal_gaze, detection_smooth_window).astype(np.float32, copy=False)
        right_nasal_gaze_smoothed = _smooth_signal(right_nasal_gaze, detection_smooth_window).astype(np.float32, copy=False)
        mean_eye_vergence_gaze_smoothed = _smooth_signal(mean_eye_vergence_gaze, detection_smooth_window).astype(np.float32, copy=False)
        version_gaze_smoothed = _smooth_signal(version_gaze, detection_smooth_window).astype(np.float32, copy=False)
        left_centroid_smoothed = _smooth_signal(left_centroid, detection_smooth_window).astype(np.float32, copy=False)
        right_centroid_smoothed = _smooth_signal(right_centroid, detection_smooth_window).astype(np.float32, copy=False)
        vergence_centroid_smoothed = _smooth_signal(vergence_centroid, detection_smooth_window).astype(np.float32, copy=False)
    else:
        left_smoothed = np.array(left_angles, copy=True)
        right_smoothed = np.array(right_angles, copy=True)
        vergence_smoothed = np.array(vergence, copy=True)
        left_signed_smoothed = np.array(left_signed, copy=True)
        right_signed_smoothed = np.array(right_signed, copy=True)
        vergence_signed_smoothed = np.array(vergence_signed, copy=True)
        version_smoothed = np.array(version, copy=True)
        left_eye_angle_smoothed = np.array(left_eye_angle, copy=True)
        right_eye_angle_smoothed = np.array(right_eye_angle, copy=True)
        vergence_eye_angle_smoothed = np.array(vergence_eye_angle, copy=True)
        left_minor_signed_smoothed = np.array(left_minor_signed, copy=True)
        right_minor_signed_smoothed = np.array(right_minor_signed, copy=True)
        vergence_minor_signed_smoothed = np.array(vergence_minor_signed, copy=True)
        version_minor_smoothed = np.array(version_minor, copy=True)
        left_gaze_smoothed = np.array(left_gaze, copy=True)
        right_gaze_smoothed = np.array(right_gaze, copy=True)
        left_gaze_signed_smoothed = np.array(left_gaze_signed, copy=True)
        right_gaze_signed_smoothed = np.array(right_gaze_signed, copy=True)
        vergence_gaze_smoothed = np.array(vergence_gaze, copy=True)
        vergence_gaze_signed_smoothed = np.array(vergence_gaze_signed, copy=True)
        left_nasal_gaze_smoothed = np.array(left_nasal_gaze, copy=True)
        right_nasal_gaze_smoothed = np.array(right_nasal_gaze, copy=True)
        mean_eye_vergence_gaze_smoothed = np.array(mean_eye_vergence_gaze, copy=True)
        version_gaze_smoothed = np.array(version_gaze, copy=True)
        left_centroid_smoothed = np.array(left_centroid, copy=True)
        right_centroid_smoothed = np.array(right_centroid, copy=True)
        vergence_centroid_smoothed = np.array(vergence_centroid, copy=True)

    left_delta = _compute_delta(left_angles)
    right_delta = _compute_delta(right_angles)
    vergence_delta = _compute_delta(vergence)
    left_signed_delta = _compute_delta(left_signed)
    right_signed_delta = _compute_delta(right_signed)
    vergence_signed_delta = _compute_delta(vergence_signed)
    version_delta = _compute_delta(version)
    left_eye_angle_delta = _compute_delta(left_eye_angle)
    right_eye_angle_delta = _compute_delta(right_eye_angle)
    vergence_eye_angle_delta = _compute_delta(vergence_eye_angle)
    left_minor_delta = _compute_delta(left_minor_signed)
    right_minor_delta = _compute_delta(right_minor_signed)
    vergence_minor_delta = _compute_delta(vergence_minor_signed)
    version_minor_delta = _compute_delta(version_minor)
    left_gaze_delta = _compute_delta(left_gaze)
    right_gaze_delta = _compute_delta(right_gaze)
    left_gaze_signed_delta = _compute_delta(left_gaze_signed)
    right_gaze_signed_delta = _compute_delta(right_gaze_signed)
    vergence_gaze_delta = _compute_delta(vergence_gaze)
    vergence_gaze_signed_delta = _compute_delta(vergence_gaze_signed)
    left_nasal_gaze_delta = _compute_delta(left_nasal_gaze)
    right_nasal_gaze_delta = _compute_delta(right_nasal_gaze)
    mean_eye_vergence_gaze_delta = _compute_delta(mean_eye_vergence_gaze)
    version_gaze_delta = _compute_delta(version_gaze)
    left_centroid_delta = _compute_delta(left_centroid)
    right_centroid_delta = _compute_delta(right_centroid)
    vergence_centroid_delta = _compute_delta(vergence_centroid)

    left_delta_smoothed = _compute_delta(left_smoothed)
    right_delta_smoothed = _compute_delta(right_smoothed)
    vergence_delta_smoothed = _compute_delta(vergence_smoothed)
    left_signed_delta_smoothed = _compute_delta(left_signed_smoothed)
    right_signed_delta_smoothed = _compute_delta(right_signed_smoothed)
    vergence_signed_delta_smoothed = _compute_delta(vergence_signed_smoothed)
    version_delta_smoothed = _compute_delta(version_smoothed)
    left_eye_angle_delta_smoothed = _compute_delta(left_eye_angle_smoothed)
    right_eye_angle_delta_smoothed = _compute_delta(right_eye_angle_smoothed)
    vergence_eye_angle_delta_smoothed = _compute_delta(vergence_eye_angle_smoothed)
    left_minor_delta_smoothed = _compute_delta(left_minor_signed_smoothed)
    right_minor_delta_smoothed = _compute_delta(right_minor_signed_smoothed)
    vergence_minor_delta_smoothed = _compute_delta(vergence_minor_signed_smoothed)
    version_minor_delta_smoothed = _compute_delta(version_minor_smoothed)
    left_gaze_delta_smoothed = _compute_delta(left_gaze_smoothed)
    right_gaze_delta_smoothed = _compute_delta(right_gaze_smoothed)
    left_gaze_signed_delta_smoothed = _compute_delta(left_gaze_signed_smoothed)
    right_gaze_signed_delta_smoothed = _compute_delta(right_gaze_signed_smoothed)
    vergence_gaze_delta_smoothed = _compute_delta(vergence_gaze_smoothed)
    vergence_gaze_signed_delta_smoothed = _compute_delta(vergence_gaze_signed_smoothed)
    left_nasal_gaze_delta_smoothed = _compute_delta(left_nasal_gaze_smoothed)
    right_nasal_gaze_delta_smoothed = _compute_delta(right_nasal_gaze_smoothed)
    mean_eye_vergence_gaze_delta_smoothed = _compute_delta(mean_eye_vergence_gaze_smoothed)
    version_gaze_delta_smoothed = _compute_delta(version_gaze_smoothed)
    left_centroid_delta_smoothed = _compute_delta(left_centroid_smoothed)
    right_centroid_delta_smoothed = _compute_delta(right_centroid_smoothed)
    vergence_centroid_delta_smoothed = _compute_delta(vergence_centroid_smoothed)

    frame_arrays, frame_valid, frame_reason = _project_detection_arrays_to_frames(
        frame_indices,
        num_frames=num_frames,
        valid_frame=valid_frame,
        reason_codes=reason_codes,
        arrays={
            "left": left_angles,
            "right": right_angles,
            "vergence": vergence,
            "vergence_signed": vergence_signed,
            "vergence_major_signed": vergence_major_signed,
            "left_eye_angle": left_eye_angle,
            "right_eye_angle": right_eye_angle,
            "vergence_eye_angle": vergence_eye_angle,
            "vergence_signed_minor": vergence_minor_signed,
            "version": version,
            "version_major": version_major,
            "version_minor": version_minor,
            "left_gaze": left_gaze,
            "right_gaze": right_gaze,
            "left_gaze_signed": left_gaze_signed,
            "right_gaze_signed": right_gaze_signed,
            "vergence_gaze": vergence_gaze,
            "vergence_gaze_signed": vergence_gaze_signed,
            "left_nasal_gaze": left_nasal_gaze,
            "right_nasal_gaze": right_nasal_gaze,
            "mean_eye_vergence_gaze": mean_eye_vergence_gaze,
            "version_gaze": version_gaze,
            "left_centroid": left_centroid,
            "right_centroid": right_centroid,
            "vergence_centroid": vergence_centroid,
        },
    )
    frame_left = frame_arrays["left"]
    frame_right = frame_arrays["right"]
    frame_vergence = frame_arrays["vergence"]
    frame_vergence_signed = frame_arrays["vergence_signed"]
    frame_vergence_major_signed = frame_arrays["vergence_major_signed"]
    frame_left_eye_angle = frame_arrays["left_eye_angle"]
    frame_right_eye_angle = frame_arrays["right_eye_angle"]
    frame_vergence_eye_angle = frame_arrays["vergence_eye_angle"]
    frame_vergence_signed_minor = frame_arrays["vergence_signed_minor"]
    frame_version = frame_arrays["version"]
    frame_version_major = frame_arrays["version_major"]
    frame_version_minor = frame_arrays["version_minor"]
    frame_left_gaze = frame_arrays["left_gaze"]
    frame_right_gaze = frame_arrays["right_gaze"]
    frame_left_gaze_signed = frame_arrays["left_gaze_signed"]
    frame_right_gaze_signed = frame_arrays["right_gaze_signed"]
    frame_vergence_gaze = frame_arrays["vergence_gaze"]
    frame_vergence_gaze_signed = frame_arrays["vergence_gaze_signed"]
    frame_left_nasal_gaze = frame_arrays["left_nasal_gaze"]
    frame_right_nasal_gaze = frame_arrays["right_nasal_gaze"]
    frame_mean_eye_vergence_gaze = frame_arrays["mean_eye_vergence_gaze"]
    frame_version_gaze = frame_arrays["version_gaze"]
    frame_left_centroid = frame_arrays["left_centroid"]
    frame_right_centroid = frame_arrays["right_centroid"]
    frame_vergence_centroid = frame_arrays["vergence_centroid"]
    frame_major_axis_marginal = _project_detection_bool_to_frames(
        frame_indices,
        num_frames=num_frames,
        values=major_axis_marginal,
    )

    frame_smooth_window = _resolve_smoothing_window(num_frames, window_setting)
    if frame_smooth_window:
        frame_left_smoothed = _smooth_signal(frame_left, frame_smooth_window).astype(np.float32, copy=False)
        frame_right_smoothed = _smooth_signal(frame_right, frame_smooth_window).astype(np.float32, copy=False)
        frame_vergence_smoothed = _smooth_signal(frame_vergence, frame_smooth_window).astype(np.float32, copy=False)
        frame_vergence_signed_smoothed = _smooth_signal(frame_vergence_signed, frame_smooth_window).astype(np.float32, copy=False)
        frame_vergence_major_signed_smoothed = _smooth_signal(frame_vergence_major_signed, frame_smooth_window).astype(np.float32, copy=False)
        frame_left_eye_angle_smoothed = _smooth_signal(frame_left_eye_angle, frame_smooth_window).astype(np.float32, copy=False)
        frame_right_eye_angle_smoothed = _smooth_signal(frame_right_eye_angle, frame_smooth_window).astype(np.float32, copy=False)
        frame_vergence_eye_angle_smoothed = _smooth_signal(frame_vergence_eye_angle, frame_smooth_window).astype(np.float32, copy=False)
        frame_version_smoothed = _smooth_signal(frame_version, frame_smooth_window).astype(np.float32, copy=False)
        frame_version_major_smoothed = _smooth_signal(frame_version_major, frame_smooth_window).astype(np.float32, copy=False)
        frame_vergence_minor_signed_smoothed = _smooth_signal(frame_vergence_signed_minor, frame_smooth_window).astype(np.float32, copy=False)
        frame_version_minor_smoothed = _smooth_signal(frame_version_minor, frame_smooth_window).astype(np.float32, copy=False)
        frame_left_gaze_smoothed = _smooth_signal(frame_left_gaze, frame_smooth_window).astype(np.float32, copy=False)
        frame_right_gaze_smoothed = _smooth_signal(frame_right_gaze, frame_smooth_window).astype(np.float32, copy=False)
        frame_left_gaze_signed_smoothed = _smooth_signal(frame_left_gaze_signed, frame_smooth_window).astype(np.float32, copy=False)
        frame_right_gaze_signed_smoothed = _smooth_signal(frame_right_gaze_signed, frame_smooth_window).astype(np.float32, copy=False)
        frame_vergence_gaze_smoothed = _smooth_signal(frame_vergence_gaze, frame_smooth_window).astype(np.float32, copy=False)
        frame_vergence_gaze_signed_smoothed = _smooth_signal(frame_vergence_gaze_signed, frame_smooth_window).astype(np.float32, copy=False)
        frame_left_nasal_gaze_smoothed = _smooth_signal(frame_left_nasal_gaze, frame_smooth_window).astype(np.float32, copy=False)
        frame_right_nasal_gaze_smoothed = _smooth_signal(frame_right_nasal_gaze, frame_smooth_window).astype(np.float32, copy=False)
        frame_mean_eye_vergence_gaze_smoothed = _smooth_signal(frame_mean_eye_vergence_gaze, frame_smooth_window).astype(np.float32, copy=False)
        frame_version_gaze_smoothed = _smooth_signal(frame_version_gaze, frame_smooth_window).astype(np.float32, copy=False)
        frame_left_centroid_smoothed = _smooth_signal(frame_left_centroid, frame_smooth_window).astype(np.float32, copy=False)
        frame_right_centroid_smoothed = _smooth_signal(frame_right_centroid, frame_smooth_window).astype(np.float32, copy=False)
        frame_vergence_centroid_smoothed = _smooth_signal(frame_vergence_centroid, frame_smooth_window).astype(np.float32, copy=False)
    else:
        frame_left_smoothed = np.array(frame_left, copy=True)
        frame_right_smoothed = np.array(frame_right, copy=True)
        frame_vergence_smoothed = np.array(frame_vergence, copy=True)
        frame_vergence_signed_smoothed = np.array(frame_vergence_signed, copy=True)
        frame_vergence_major_signed_smoothed = np.array(frame_vergence_major_signed, copy=True)
        frame_left_eye_angle_smoothed = np.array(frame_left_eye_angle, copy=True)
        frame_right_eye_angle_smoothed = np.array(frame_right_eye_angle, copy=True)
        frame_vergence_eye_angle_smoothed = np.array(frame_vergence_eye_angle, copy=True)
        frame_version_smoothed = np.array(frame_version, copy=True)
        frame_version_major_smoothed = np.array(frame_version_major, copy=True)
        frame_vergence_minor_signed_smoothed = np.array(frame_vergence_signed_minor, copy=True)
        frame_version_minor_smoothed = np.array(frame_version_minor, copy=True)
        frame_left_gaze_smoothed = np.array(frame_left_gaze, copy=True)
        frame_right_gaze_smoothed = np.array(frame_right_gaze, copy=True)
        frame_left_gaze_signed_smoothed = np.array(frame_left_gaze_signed, copy=True)
        frame_right_gaze_signed_smoothed = np.array(frame_right_gaze_signed, copy=True)
        frame_vergence_gaze_smoothed = np.array(frame_vergence_gaze, copy=True)
        frame_vergence_gaze_signed_smoothed = np.array(frame_vergence_gaze_signed, copy=True)
        frame_left_nasal_gaze_smoothed = np.array(frame_left_nasal_gaze, copy=True)
        frame_right_nasal_gaze_smoothed = np.array(frame_right_nasal_gaze, copy=True)
        frame_mean_eye_vergence_gaze_smoothed = np.array(frame_mean_eye_vergence_gaze, copy=True)
        frame_version_gaze_smoothed = np.array(frame_version_gaze, copy=True)
        frame_left_centroid_smoothed = np.array(frame_left_centroid, copy=True)
        frame_right_centroid_smoothed = np.array(frame_right_centroid, copy=True)
        frame_vergence_centroid_smoothed = np.array(frame_vergence_centroid, copy=True)

    frame_left_delta = _compute_delta(frame_left)
    frame_right_delta = _compute_delta(frame_right)
    frame_vergence_delta = _compute_delta(frame_vergence)
    frame_vergence_signed_delta = _compute_delta(frame_vergence_signed)
    frame_vergence_major_delta = _compute_delta(frame_vergence_major_signed)
    frame_left_eye_angle_delta = _compute_delta(frame_left_eye_angle)
    frame_right_eye_angle_delta = _compute_delta(frame_right_eye_angle)
    frame_vergence_eye_angle_delta = _compute_delta(frame_vergence_eye_angle)
    frame_vergence_minor_delta = _compute_delta(frame_vergence_signed_minor)
    frame_version_delta = _compute_delta(frame_version)
    frame_version_major_delta = _compute_delta(frame_version_major)
    frame_version_minor_delta = _compute_delta(frame_version_minor)
    frame_left_gaze_delta = _compute_delta(frame_left_gaze)
    frame_right_gaze_delta = _compute_delta(frame_right_gaze)
    frame_left_gaze_signed_delta = _compute_delta(frame_left_gaze_signed)
    frame_right_gaze_signed_delta = _compute_delta(frame_right_gaze_signed)
    frame_vergence_gaze_delta = _compute_delta(frame_vergence_gaze)
    frame_vergence_gaze_signed_delta = _compute_delta(frame_vergence_gaze_signed)
    frame_left_nasal_gaze_delta = _compute_delta(frame_left_nasal_gaze)
    frame_right_nasal_gaze_delta = _compute_delta(frame_right_nasal_gaze)
    frame_mean_eye_vergence_gaze_delta = _compute_delta(frame_mean_eye_vergence_gaze)
    frame_version_gaze_delta = _compute_delta(frame_version_gaze)
    frame_left_centroid_delta = _compute_delta(frame_left_centroid)
    frame_right_centroid_delta = _compute_delta(frame_right_centroid)
    frame_vergence_centroid_delta = _compute_delta(frame_vergence_centroid)

    frame_left_delta_smoothed = _compute_delta(frame_left_smoothed)
    frame_right_delta_smoothed = _compute_delta(frame_right_smoothed)
    frame_vergence_delta_smoothed = _compute_delta(frame_vergence_smoothed)
    frame_vergence_signed_delta_smoothed = _compute_delta(frame_vergence_signed_smoothed)
    frame_vergence_major_delta_smoothed = _compute_delta(frame_vergence_major_signed_smoothed)
    frame_left_eye_angle_delta_smoothed = _compute_delta(frame_left_eye_angle_smoothed)
    frame_right_eye_angle_delta_smoothed = _compute_delta(frame_right_eye_angle_smoothed)
    frame_vergence_eye_angle_delta_smoothed = _compute_delta(frame_vergence_eye_angle_smoothed)
    frame_vergence_minor_delta_smoothed = _compute_delta(frame_vergence_minor_signed_smoothed)
    frame_version_delta_smoothed = _compute_delta(frame_version_smoothed)
    frame_version_major_delta_smoothed = _compute_delta(frame_version_major_smoothed)
    frame_version_minor_delta_smoothed = _compute_delta(frame_version_minor_smoothed)
    frame_left_gaze_delta_smoothed = _compute_delta(frame_left_gaze_smoothed)
    frame_right_gaze_delta_smoothed = _compute_delta(frame_right_gaze_smoothed)
    frame_left_gaze_signed_delta_smoothed = _compute_delta(frame_left_gaze_signed_smoothed)
    frame_right_gaze_signed_delta_smoothed = _compute_delta(frame_right_gaze_signed_smoothed)
    frame_vergence_gaze_delta_smoothed = _compute_delta(frame_vergence_gaze_smoothed)
    frame_vergence_gaze_signed_delta_smoothed = _compute_delta(frame_vergence_gaze_signed_smoothed)
    frame_left_nasal_gaze_delta_smoothed = _compute_delta(frame_left_nasal_gaze_smoothed)
    frame_right_nasal_gaze_delta_smoothed = _compute_delta(frame_right_nasal_gaze_smoothed)
    frame_mean_eye_vergence_gaze_delta_smoothed = _compute_delta(frame_mean_eye_vergence_gaze_smoothed)
    frame_version_gaze_delta_smoothed = _compute_delta(frame_version_gaze_smoothed)
    frame_left_centroid_delta_smoothed = _compute_delta(frame_left_centroid_smoothed)
    frame_right_centroid_delta_smoothed = _compute_delta(frame_right_centroid_smoothed)
    frame_vergence_centroid_delta_smoothed = _compute_delta(frame_vergence_centroid_smoothed)

    angles_group = run_group.require_group("angles")
    roi_group = angles_group.require_group("roi")
    frame_group = angles_group.require_group("frame")

    _prepare_output_arrays(
        roi_group,
        [
            ("left_deg", (total_detections,), (chunk_len,), "f4"),
            ("left_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
            ("left_delta_deg", (total_detections,), (chunk_len,), "f4"),
            ("left_delta_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
            ("right_deg", (total_detections,), (chunk_len,), "f4"),
            ("right_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
            ("right_delta_deg", (total_detections,), (chunk_len,), "f4"),
            ("right_delta_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
            ("vergence_deg", (total_detections,), (chunk_len,), "f4"),
            ("vergence_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
            ("vergence_delta_deg", (total_detections,), (chunk_len,), "f4"),
            ("vergence_delta_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
            ("left_signed_deg", (total_detections,), (chunk_len,), "f4"),
            ("left_signed_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
            ("left_signed_delta_deg", (total_detections,), (chunk_len,), "f4"),
            ("left_signed_delta_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
            ("right_signed_deg", (total_detections,), (chunk_len,), "f4"),
            ("right_signed_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
            ("right_signed_delta_deg", (total_detections,), (chunk_len,), "f4"),
            ("right_signed_delta_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
            ("left_eye_angle_deg", (total_detections,), (chunk_len,), "f4"),
            ("left_eye_angle_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
            ("left_eye_angle_delta_deg", (total_detections,), (chunk_len,), "f4"),
            ("left_eye_angle_delta_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
            ("right_eye_angle_deg", (total_detections,), (chunk_len,), "f4"),
            ("right_eye_angle_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
            ("right_eye_angle_delta_deg", (total_detections,), (chunk_len,), "f4"),
            ("right_eye_angle_delta_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
            ("vergence_eye_angle_deg", (total_detections,), (chunk_len,), "f4"),
            ("vergence_eye_angle_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
            ("vergence_eye_angle_delta_deg", (total_detections,), (chunk_len,), "f4"),
            ("vergence_eye_angle_delta_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
            ("vergence_signed_deg", (total_detections,), (chunk_len,), "f4"),
            ("vergence_signed_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
            ("vergence_signed_delta_deg", (total_detections,), (chunk_len,), "f4"),
            ("vergence_signed_delta_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
            ("version_deg", (total_detections,), (chunk_len,), "f4"),
            ("version_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
            ("version_delta_deg", (total_detections,), (chunk_len,), "f4"),
            ("version_delta_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
            ("left_minor_signed_deg", (total_detections,), (chunk_len,), "f4"),
            ("left_minor_signed_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
            ("left_minor_signed_delta_deg", (total_detections,), (chunk_len,), "f4"),
            ("left_minor_signed_delta_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
            ("right_minor_signed_deg", (total_detections,), (chunk_len,), "f4"),
            ("right_minor_signed_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
            ("right_minor_signed_delta_deg", (total_detections,), (chunk_len,), "f4"),
            ("right_minor_signed_delta_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
            ("vergence_minor_signed_deg", (total_detections,), (chunk_len,), "f4"),
            ("vergence_minor_signed_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
            ("vergence_minor_signed_delta_deg", (total_detections,), (chunk_len,), "f4"),
            ("vergence_minor_signed_delta_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
            ("version_minor_deg", (total_detections,), (chunk_len,), "f4"),
            ("version_minor_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
            ("version_minor_delta_deg", (total_detections,), (chunk_len,), "f4"),
            ("version_minor_delta_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
            ("left_gaze_deg", (total_detections,), (chunk_len,), "f4"),
            ("left_gaze_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
            ("left_gaze_delta_deg", (total_detections,), (chunk_len,), "f4"),
            ("left_gaze_delta_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
            ("right_gaze_deg", (total_detections,), (chunk_len,), "f4"),
            ("right_gaze_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
            ("right_gaze_delta_deg", (total_detections,), (chunk_len,), "f4"),
            ("right_gaze_delta_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
            ("left_gaze_signed_deg", (total_detections,), (chunk_len,), "f4"),
            ("left_gaze_signed_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
            ("left_gaze_signed_delta_deg", (total_detections,), (chunk_len,), "f4"),
            ("left_gaze_signed_delta_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
            ("right_gaze_signed_deg", (total_detections,), (chunk_len,), "f4"),
            ("right_gaze_signed_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
            ("right_gaze_signed_delta_deg", (total_detections,), (chunk_len,), "f4"),
            ("right_gaze_signed_delta_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
            ("vergence_gaze_deg", (total_detections,), (chunk_len,), "f4"),
            ("vergence_gaze_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
            ("vergence_gaze_delta_deg", (total_detections,), (chunk_len,), "f4"),
            ("vergence_gaze_delta_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
            ("vergence_gaze_signed_deg", (total_detections,), (chunk_len,), "f4"),
            ("vergence_gaze_signed_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
            ("vergence_gaze_signed_delta_deg", (total_detections,), (chunk_len,), "f4"),
            ("vergence_gaze_signed_delta_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
            ("left_nasal_gaze_deg", (total_detections,), (chunk_len,), "f4"),
            ("left_nasal_gaze_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
            ("left_nasal_gaze_delta_deg", (total_detections,), (chunk_len,), "f4"),
            ("left_nasal_gaze_delta_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
            ("right_nasal_gaze_deg", (total_detections,), (chunk_len,), "f4"),
            ("right_nasal_gaze_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
            ("right_nasal_gaze_delta_deg", (total_detections,), (chunk_len,), "f4"),
            ("right_nasal_gaze_delta_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
            ("mean_eye_vergence_gaze_deg", (total_detections,), (chunk_len,), "f4"),
            ("mean_eye_vergence_gaze_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
            ("mean_eye_vergence_gaze_delta_deg", (total_detections,), (chunk_len,), "f4"),
            ("mean_eye_vergence_gaze_delta_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
            ("version_gaze_deg", (total_detections,), (chunk_len,), "f4"),
            ("version_gaze_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
            ("version_gaze_delta_deg", (total_detections,), (chunk_len,), "f4"),
            ("version_gaze_delta_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
            ("left_speed_deg_s", (total_detections,), (chunk_len,), "f4"),
            ("right_speed_deg_s", (total_detections,), (chunk_len,), "f4"),
            ("vergence_speed_deg_s", (total_detections,), (chunk_len,), "f4"),
            ("vergence_signed_speed_deg_s", (total_detections,), (chunk_len,), "f4"),
            ("version_speed_deg_s", (total_detections,), (chunk_len,), "f4"),
            ("left_gaze_speed_deg_s", (total_detections,), (chunk_len,), "f4"),
            ("right_gaze_speed_deg_s", (total_detections,), (chunk_len,), "f4"),
            ("vergence_gaze_speed_deg_s", (total_detections,), (chunk_len,), "f4"),
            ("vergence_gaze_signed_speed_deg_s", (total_detections,), (chunk_len,), "f4"),
            ("version_gaze_speed_deg_s", (total_detections,), (chunk_len,), "f4"),
            ("mean_eye_vergence_gaze_speed_deg_s", (total_detections,), (chunk_len,), "f4"),
            ("left_accel_deg_s2", (total_detections,), (chunk_len,), "f4"),
            ("right_accel_deg_s2", (total_detections,), (chunk_len,), "f4"),
            ("vergence_accel_deg_s2", (total_detections,), (chunk_len,), "f4"),
            ("vergence_signed_accel_deg_s2", (total_detections,), (chunk_len,), "f4"),
            ("version_accel_deg_s2", (total_detections,), (chunk_len,), "f4"),
            ("left_gaze_accel_deg_s2", (total_detections,), (chunk_len,), "f4"),
            ("right_gaze_accel_deg_s2", (total_detections,), (chunk_len,), "f4"),
            ("vergence_gaze_accel_deg_s2", (total_detections,), (chunk_len,), "f4"),
            ("vergence_gaze_signed_accel_deg_s2", (total_detections,), (chunk_len,), "f4"),
            ("version_gaze_accel_deg_s2", (total_detections,), (chunk_len,), "f4"),
            ("mean_eye_vergence_gaze_accel_deg_s2", (total_detections,), (chunk_len,), "f4"),
            ("heading_deg", (total_detections,), (chunk_len,), "f4"),
            # Centroid-based eye-position angles
            ("left_centroid_deg", (total_detections,), (chunk_len,), "f4"),
            ("left_centroid_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
            ("left_centroid_delta_deg", (total_detections,), (chunk_len,), "f4"),
            ("left_centroid_delta_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
            ("right_centroid_deg", (total_detections,), (chunk_len,), "f4"),
            ("right_centroid_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
            ("right_centroid_delta_deg", (total_detections,), (chunk_len,), "f4"),
            ("right_centroid_delta_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
            ("vergence_centroid_deg", (total_detections,), (chunk_len,), "f4"),
            ("vergence_centroid_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
            ("vergence_centroid_delta_deg", (total_detections,), (chunk_len,), "f4"),
            ("vergence_centroid_delta_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
        ],
    )
    roi_group["left_deg_smoothed"][:] = left_smoothed
    roi_group["left_delta_deg"][:] = left_delta
    roi_group["left_delta_deg_smoothed"][:] = left_delta_smoothed
    roi_group["right_deg_smoothed"][:] = right_smoothed
    roi_group["right_delta_deg"][:] = right_delta
    roi_group["right_delta_deg_smoothed"][:] = right_delta_smoothed
    roi_group["vergence_deg_smoothed"][:] = vergence_smoothed
    roi_group["vergence_delta_deg"][:] = vergence_delta
    roi_group["vergence_delta_deg_smoothed"][:] = vergence_delta_smoothed
    roi_group["left_signed_deg_smoothed"][:] = left_signed_smoothed
    roi_group["left_signed_delta_deg"][:] = left_signed_delta
    roi_group["left_signed_delta_deg_smoothed"][:] = left_signed_delta_smoothed
    roi_group["right_signed_deg_smoothed"][:] = right_signed_smoothed
    roi_group["right_signed_delta_deg"][:] = right_signed_delta
    roi_group["right_signed_delta_deg_smoothed"][:] = right_signed_delta_smoothed
    roi_group["left_eye_angle_deg_smoothed"][:] = left_eye_angle_smoothed
    roi_group["left_eye_angle_delta_deg"][:] = left_eye_angle_delta
    roi_group["left_eye_angle_delta_deg_smoothed"][:] = left_eye_angle_delta_smoothed
    roi_group["right_eye_angle_deg_smoothed"][:] = right_eye_angle_smoothed
    roi_group["right_eye_angle_delta_deg"][:] = right_eye_angle_delta
    roi_group["right_eye_angle_delta_deg_smoothed"][:] = right_eye_angle_delta_smoothed
    roi_group["vergence_eye_angle_deg_smoothed"][:] = vergence_eye_angle_smoothed
    roi_group["vergence_eye_angle_delta_deg"][:] = vergence_eye_angle_delta
    roi_group["vergence_eye_angle_delta_deg_smoothed"][:] = vergence_eye_angle_delta_smoothed
    roi_group["vergence_signed_deg_smoothed"][:] = vergence_signed_smoothed
    roi_group["vergence_signed_delta_deg"][:] = vergence_signed_delta
    roi_group["vergence_signed_delta_deg_smoothed"][:] = vergence_signed_delta_smoothed
    roi_group["version_deg_smoothed"][:] = version_smoothed
    roi_group["version_delta_deg"][:] = version_delta
    roi_group["version_delta_deg_smoothed"][:] = version_delta_smoothed
    roi_group["left_minor_signed_deg_smoothed"][:] = left_minor_signed_smoothed
    roi_group["left_minor_signed_delta_deg"][:] = left_minor_delta
    roi_group["left_minor_signed_delta_deg_smoothed"][:] = left_minor_delta_smoothed
    roi_group["right_minor_signed_deg_smoothed"][:] = right_minor_signed_smoothed
    roi_group["right_minor_signed_delta_deg"][:] = right_minor_delta
    roi_group["right_minor_signed_delta_deg_smoothed"][:] = right_minor_delta_smoothed
    roi_group["vergence_minor_signed_deg_smoothed"][:] = vergence_minor_signed_smoothed
    roi_group["vergence_minor_signed_delta_deg"][:] = vergence_minor_delta
    roi_group["vergence_minor_signed_delta_deg_smoothed"][:] = vergence_minor_delta_smoothed
    roi_group["version_minor_deg_smoothed"][:] = version_minor_smoothed
    roi_group["version_minor_delta_deg"][:] = version_minor_delta
    roi_group["version_minor_delta_deg_smoothed"][:] = version_minor_delta_smoothed
    roi_group["left_gaze_deg_smoothed"][:] = left_gaze_smoothed
    roi_group["left_gaze_delta_deg"][:] = left_gaze_delta
    roi_group["left_gaze_delta_deg_smoothed"][:] = left_gaze_delta_smoothed
    roi_group["right_gaze_deg_smoothed"][:] = right_gaze_smoothed
    roi_group["right_gaze_delta_deg"][:] = right_gaze_delta
    roi_group["right_gaze_delta_deg_smoothed"][:] = right_gaze_delta_smoothed
    roi_group["left_gaze_signed_deg_smoothed"][:] = left_gaze_signed_smoothed
    roi_group["left_gaze_signed_delta_deg"][:] = left_gaze_signed_delta
    roi_group["left_gaze_signed_delta_deg_smoothed"][:] = left_gaze_signed_delta_smoothed
    roi_group["right_gaze_signed_deg_smoothed"][:] = right_gaze_signed_smoothed
    roi_group["right_gaze_signed_delta_deg"][:] = right_gaze_signed_delta
    roi_group["right_gaze_signed_delta_deg_smoothed"][:] = right_gaze_signed_delta_smoothed
    roi_group["vergence_gaze_deg_smoothed"][:] = vergence_gaze_smoothed
    roi_group["vergence_gaze_delta_deg"][:] = vergence_gaze_delta
    roi_group["vergence_gaze_delta_deg_smoothed"][:] = vergence_gaze_delta_smoothed
    roi_group["vergence_gaze_signed_deg_smoothed"][:] = vergence_gaze_signed_smoothed
    roi_group["vergence_gaze_signed_delta_deg"][:] = vergence_gaze_signed_delta
    roi_group["vergence_gaze_signed_delta_deg_smoothed"][:] = vergence_gaze_signed_delta_smoothed
    roi_group["left_nasal_gaze_deg_smoothed"][:] = left_nasal_gaze_smoothed
    roi_group["left_nasal_gaze_delta_deg"][:] = left_nasal_gaze_delta
    roi_group["left_nasal_gaze_delta_deg_smoothed"][:] = left_nasal_gaze_delta_smoothed
    roi_group["right_nasal_gaze_deg_smoothed"][:] = right_nasal_gaze_smoothed
    roi_group["right_nasal_gaze_delta_deg"][:] = right_nasal_gaze_delta
    roi_group["right_nasal_gaze_delta_deg_smoothed"][:] = right_nasal_gaze_delta_smoothed
    roi_group["mean_eye_vergence_gaze_deg_smoothed"][:] = mean_eye_vergence_gaze_smoothed
    roi_group["mean_eye_vergence_gaze_delta_deg"][:] = mean_eye_vergence_gaze_delta
    roi_group["mean_eye_vergence_gaze_delta_deg_smoothed"][:] = mean_eye_vergence_gaze_delta_smoothed
    roi_group["version_gaze_deg_smoothed"][:] = version_gaze_smoothed
    roi_group["version_gaze_delta_deg"][:] = version_gaze_delta
    roi_group["version_gaze_delta_deg_smoothed"][:] = version_gaze_delta_smoothed
    roi_group["left_speed_deg_s"][:] = left_speed
    roi_group["right_speed_deg_s"][:] = right_speed
    roi_group["vergence_speed_deg_s"][:] = vergence_speed
    roi_group["vergence_signed_speed_deg_s"][:] = vergence_signed_speed
    roi_group["version_speed_deg_s"][:] = version_speed
    roi_group["left_gaze_speed_deg_s"][:] = left_gaze_speed
    roi_group["right_gaze_speed_deg_s"][:] = right_gaze_speed
    roi_group["vergence_gaze_speed_deg_s"][:] = vergence_gaze_speed
    roi_group["vergence_gaze_signed_speed_deg_s"][:] = vergence_gaze_signed_speed
    roi_group["version_gaze_speed_deg_s"][:] = version_gaze_speed
    roi_group["mean_eye_vergence_gaze_speed_deg_s"][:] = mean_eye_vergence_gaze_speed
    roi_group["left_accel_deg_s2"][:] = left_accel
    roi_group["right_accel_deg_s2"][:] = right_accel
    roi_group["vergence_accel_deg_s2"][:] = vergence_accel
    roi_group["vergence_signed_accel_deg_s2"][:] = vergence_signed_accel
    roi_group["version_accel_deg_s2"][:] = version_accel
    roi_group["left_gaze_accel_deg_s2"][:] = left_gaze_accel
    roi_group["right_gaze_accel_deg_s2"][:] = right_gaze_accel
    roi_group["vergence_gaze_accel_deg_s2"][:] = vergence_gaze_accel
    roi_group["vergence_gaze_signed_accel_deg_s2"][:] = vergence_gaze_signed_accel
    roi_group["version_gaze_accel_deg_s2"][:] = version_gaze_accel
    roi_group["mean_eye_vergence_gaze_accel_deg_s2"][:] = mean_eye_vergence_gaze_accel
    # Centroid-based angles
    roi_group["left_centroid_deg_smoothed"][:] = left_centroid_smoothed
    roi_group["left_centroid_delta_deg"][:] = left_centroid_delta
    roi_group["left_centroid_delta_deg_smoothed"][:] = left_centroid_delta_smoothed
    roi_group["right_centroid_deg_smoothed"][:] = right_centroid_smoothed
    roi_group["right_centroid_delta_deg"][:] = right_centroid_delta
    roi_group["right_centroid_delta_deg_smoothed"][:] = right_centroid_delta_smoothed
    roi_group["vergence_centroid_deg_smoothed"][:] = vergence_centroid_smoothed
    roi_group["vergence_centroid_delta_deg"][:] = vergence_centroid_delta
    roi_group["vergence_centroid_delta_deg_smoothed"][:] = vergence_centroid_delta_smoothed

    _prepare_output_arrays(
        frame_group,
        [
            ("left_deg", (num_frames,), (frame_chunk,), "f4"),
            ("left_deg_smoothed", (num_frames,), (frame_chunk,), "f4"),
            ("left_delta_deg", (num_frames,), (frame_chunk,), "f4"),
            ("left_delta_deg_smoothed", (num_frames,), (frame_chunk,), "f4"),
            ("right_deg", (num_frames,), (frame_chunk,), "f4"),
            ("right_deg_smoothed", (num_frames,), (frame_chunk,), "f4"),
            ("right_delta_deg", (num_frames,), (frame_chunk,), "f4"),
            ("right_delta_deg_smoothed", (num_frames,), (frame_chunk,), "f4"),
            ("vergence_deg", (num_frames,), (frame_chunk,), "f4"),
            ("vergence_deg_smoothed", (num_frames,), (frame_chunk,), "f4"),
            ("vergence_delta_deg", (num_frames,), (frame_chunk,), "f4"),
            ("vergence_delta_deg_smoothed", (num_frames,), (frame_chunk,), "f4"),
            ("vergence_signed_deg", (num_frames,), (frame_chunk,), "f4"),
            ("vergence_signed_deg_smoothed", (num_frames,), (frame_chunk,), "f4"),
            ("vergence_signed_delta_deg", (num_frames,), (frame_chunk,), "f4"),
            ("vergence_signed_delta_deg_smoothed", (num_frames,), (frame_chunk,), "f4"),
            ("vergence_major_signed_deg", (num_frames,), (frame_chunk,), "f4"),
            ("vergence_major_signed_deg_smoothed", (num_frames,), (frame_chunk,), "f4"),
            ("vergence_major_signed_delta_deg", (num_frames,), (frame_chunk,), "f4"),
            ("vergence_major_signed_delta_deg_smoothed", (num_frames,), (frame_chunk,), "f4"),
            ("left_eye_angle_deg", (num_frames,), (frame_chunk,), "f4"),
            ("left_eye_angle_deg_smoothed", (num_frames,), (frame_chunk,), "f4"),
            ("left_eye_angle_delta_deg", (num_frames,), (frame_chunk,), "f4"),
            ("left_eye_angle_delta_deg_smoothed", (num_frames,), (frame_chunk,), "f4"),
            ("right_eye_angle_deg", (num_frames,), (frame_chunk,), "f4"),
            ("right_eye_angle_deg_smoothed", (num_frames,), (frame_chunk,), "f4"),
            ("right_eye_angle_delta_deg", (num_frames,), (frame_chunk,), "f4"),
            ("right_eye_angle_delta_deg_smoothed", (num_frames,), (frame_chunk,), "f4"),
            ("vergence_eye_angle_deg", (num_frames,), (frame_chunk,), "f4"),
            ("vergence_eye_angle_deg_smoothed", (num_frames,), (frame_chunk,), "f4"),
            ("vergence_eye_angle_delta_deg", (num_frames,), (frame_chunk,), "f4"),
            ("vergence_eye_angle_delta_deg_smoothed", (num_frames,), (frame_chunk,), "f4"),
            ("vergence_minor_signed_deg", (num_frames,), (frame_chunk,), "f4"),
            ("vergence_minor_signed_deg_smoothed", (num_frames,), (frame_chunk,), "f4"),
            ("vergence_minor_signed_delta_deg", (num_frames,), (frame_chunk,), "f4"),
            ("vergence_minor_signed_delta_deg_smoothed", (num_frames,), (frame_chunk,), "f4"),
            ("version_deg", (num_frames,), (frame_chunk,), "f4"),
            ("version_deg_smoothed", (num_frames,), (frame_chunk,), "f4"),
            ("version_delta_deg", (num_frames,), (frame_chunk,), "f4"),
            ("version_delta_deg_smoothed", (num_frames,), (frame_chunk,), "f4"),
            ("version_major_deg", (num_frames,), (frame_chunk,), "f4"),
            ("version_major_deg_smoothed", (num_frames,), (frame_chunk,), "f4"),
            ("version_major_delta_deg", (num_frames,), (frame_chunk,), "f4"),
            ("version_major_delta_deg_smoothed", (num_frames,), (frame_chunk,), "f4"),
            ("version_minor_deg", (num_frames,), (frame_chunk,), "f4"),
            ("version_minor_deg_smoothed", (num_frames,), (frame_chunk,), "f4"),
            ("version_minor_delta_deg", (num_frames,), (frame_chunk,), "f4"),
            ("version_minor_delta_deg_smoothed", (num_frames,), (frame_chunk,), "f4"),
            ("left_gaze_deg", (num_frames,), (frame_chunk,), "f4"),
            ("left_gaze_deg_smoothed", (num_frames,), (frame_chunk,), "f4"),
            ("left_gaze_delta_deg", (num_frames,), (frame_chunk,), "f4"),
            ("left_gaze_delta_deg_smoothed", (num_frames,), (frame_chunk,), "f4"),
            ("right_gaze_deg", (num_frames,), (frame_chunk,), "f4"),
            ("right_gaze_deg_smoothed", (num_frames,), (frame_chunk,), "f4"),
            ("right_gaze_delta_deg", (num_frames,), (frame_chunk,), "f4"),
            ("right_gaze_delta_deg_smoothed", (num_frames,), (frame_chunk,), "f4"),
            ("left_gaze_signed_deg", (num_frames,), (frame_chunk,), "f4"),
            ("left_gaze_signed_deg_smoothed", (num_frames,), (frame_chunk,), "f4"),
            ("left_gaze_signed_delta_deg", (num_frames,), (frame_chunk,), "f4"),
            ("left_gaze_signed_delta_deg_smoothed", (num_frames,), (frame_chunk,), "f4"),
            ("right_gaze_signed_deg", (num_frames,), (frame_chunk,), "f4"),
            ("right_gaze_signed_deg_smoothed", (num_frames,), (frame_chunk,), "f4"),
            ("right_gaze_signed_delta_deg", (num_frames,), (frame_chunk,), "f4"),
            ("right_gaze_signed_delta_deg_smoothed", (num_frames,), (frame_chunk,), "f4"),
            ("vergence_gaze_deg", (num_frames,), (frame_chunk,), "f4"),
            ("vergence_gaze_deg_smoothed", (num_frames,), (frame_chunk,), "f4"),
            ("vergence_gaze_delta_deg", (num_frames,), (frame_chunk,), "f4"),
            ("vergence_gaze_delta_deg_smoothed", (num_frames,), (frame_chunk,), "f4"),
            ("vergence_gaze_signed_deg", (num_frames,), (frame_chunk,), "f4"),
            ("vergence_gaze_signed_deg_smoothed", (num_frames,), (frame_chunk,), "f4"),
            ("vergence_gaze_signed_delta_deg", (num_frames,), (frame_chunk,), "f4"),
            ("vergence_gaze_signed_delta_deg_smoothed", (num_frames,), (frame_chunk,), "f4"),
            ("left_nasal_gaze_deg", (num_frames,), (frame_chunk,), "f4"),
            ("left_nasal_gaze_deg_smoothed", (num_frames,), (frame_chunk,), "f4"),
            ("left_nasal_gaze_delta_deg", (num_frames,), (frame_chunk,), "f4"),
            ("left_nasal_gaze_delta_deg_smoothed", (num_frames,), (frame_chunk,), "f4"),
            ("right_nasal_gaze_deg", (num_frames,), (frame_chunk,), "f4"),
            ("right_nasal_gaze_deg_smoothed", (num_frames,), (frame_chunk,), "f4"),
            ("right_nasal_gaze_delta_deg", (num_frames,), (frame_chunk,), "f4"),
            ("right_nasal_gaze_delta_deg_smoothed", (num_frames,), (frame_chunk,), "f4"),
            ("mean_eye_vergence_gaze_deg", (num_frames,), (frame_chunk,), "f4"),
            ("mean_eye_vergence_gaze_deg_smoothed", (num_frames,), (frame_chunk,), "f4"),
            ("mean_eye_vergence_gaze_delta_deg", (num_frames,), (frame_chunk,), "f4"),
            ("mean_eye_vergence_gaze_delta_deg_smoothed", (num_frames,), (frame_chunk,), "f4"),
            ("version_gaze_deg", (num_frames,), (frame_chunk,), "f4"),
            ("version_gaze_deg_smoothed", (num_frames,), (frame_chunk,), "f4"),
            ("version_gaze_delta_deg", (num_frames,), (frame_chunk,), "f4"),
            ("version_gaze_delta_deg_smoothed", (num_frames,), (frame_chunk,), "f4"),
            # Centroid-based eye-position angles
            ("left_centroid_deg", (num_frames,), (frame_chunk,), "f4"),
            ("left_centroid_deg_smoothed", (num_frames,), (frame_chunk,), "f4"),
            ("left_centroid_delta_deg", (num_frames,), (frame_chunk,), "f4"),
            ("left_centroid_delta_deg_smoothed", (num_frames,), (frame_chunk,), "f4"),
            ("right_centroid_deg", (num_frames,), (frame_chunk,), "f4"),
            ("right_centroid_deg_smoothed", (num_frames,), (frame_chunk,), "f4"),
            ("right_centroid_delta_deg", (num_frames,), (frame_chunk,), "f4"),
            ("right_centroid_delta_deg_smoothed", (num_frames,), (frame_chunk,), "f4"),
            ("vergence_centroid_deg", (num_frames,), (frame_chunk,), "f4"),
            ("vergence_centroid_deg_smoothed", (num_frames,), (frame_chunk,), "f4"),
            ("vergence_centroid_delta_deg", (num_frames,), (frame_chunk,), "f4"),
            ("vergence_centroid_delta_deg_smoothed", (num_frames,), (frame_chunk,), "f4"),
        ],
    )
    if num_frames > 0:
        frame_group["left_deg"][:] = frame_left
        frame_group["left_deg_smoothed"][:] = frame_left_smoothed
        frame_group["left_delta_deg"][:] = frame_left_delta
        frame_group["left_delta_deg_smoothed"][:] = frame_left_delta_smoothed
        frame_group["right_deg"][:] = frame_right
        frame_group["right_deg_smoothed"][:] = frame_right_smoothed
        frame_group["right_delta_deg"][:] = frame_right_delta
        frame_group["right_delta_deg_smoothed"][:] = frame_right_delta_smoothed
        frame_group["vergence_deg"][:] = frame_vergence
        frame_group["vergence_deg_smoothed"][:] = frame_vergence_smoothed
        frame_group["vergence_delta_deg"][:] = frame_vergence_delta
        frame_group["vergence_delta_deg_smoothed"][:] = frame_vergence_delta_smoothed
        frame_group["vergence_signed_deg"][:] = frame_vergence_signed
        frame_group["vergence_signed_deg_smoothed"][:] = frame_vergence_signed_smoothed
        frame_group["vergence_signed_delta_deg"][:] = frame_vergence_signed_delta
        frame_group["vergence_signed_delta_deg_smoothed"][:] = frame_vergence_signed_delta_smoothed
        frame_group["vergence_major_signed_deg"][:] = frame_vergence_major_signed
        frame_group["vergence_major_signed_deg_smoothed"][:] = frame_vergence_major_signed_smoothed
        frame_group["vergence_major_signed_delta_deg"][:] = frame_vergence_major_delta
        frame_group["vergence_major_signed_delta_deg_smoothed"][:] = frame_vergence_major_delta_smoothed
        frame_group["left_eye_angle_deg"][:] = frame_left_eye_angle
        frame_group["left_eye_angle_deg_smoothed"][:] = frame_left_eye_angle_smoothed
        frame_group["left_eye_angle_delta_deg"][:] = frame_left_eye_angle_delta
        frame_group["left_eye_angle_delta_deg_smoothed"][:] = frame_left_eye_angle_delta_smoothed
        frame_group["right_eye_angle_deg"][:] = frame_right_eye_angle
        frame_group["right_eye_angle_deg_smoothed"][:] = frame_right_eye_angle_smoothed
        frame_group["right_eye_angle_delta_deg"][:] = frame_right_eye_angle_delta
        frame_group["right_eye_angle_delta_deg_smoothed"][:] = frame_right_eye_angle_delta_smoothed
        frame_group["vergence_eye_angle_deg"][:] = frame_vergence_eye_angle
        frame_group["vergence_eye_angle_deg_smoothed"][:] = frame_vergence_eye_angle_smoothed
        frame_group["vergence_eye_angle_delta_deg"][:] = frame_vergence_eye_angle_delta
        frame_group["vergence_eye_angle_delta_deg_smoothed"][:] = frame_vergence_eye_angle_delta_smoothed
        frame_group["vergence_minor_signed_deg"][:] = frame_vergence_signed_minor
        frame_group["vergence_minor_signed_deg_smoothed"][:] = frame_vergence_minor_signed_smoothed
        frame_group["vergence_minor_signed_delta_deg"][:] = frame_vergence_minor_delta
        frame_group["vergence_minor_signed_delta_deg_smoothed"][:] = frame_vergence_minor_delta_smoothed
        frame_group["version_deg"][:] = frame_version
        frame_group["version_deg_smoothed"][:] = frame_version_smoothed
        frame_group["version_delta_deg"][:] = frame_version_delta
        frame_group["version_delta_deg_smoothed"][:] = frame_version_delta_smoothed
        frame_group["version_major_deg"][:] = frame_version_major
        frame_group["version_major_deg_smoothed"][:] = frame_version_major_smoothed
        frame_group["version_major_delta_deg"][:] = frame_version_major_delta
        frame_group["version_major_delta_deg_smoothed"][:] = frame_version_major_delta_smoothed
        frame_group["version_minor_deg"][:] = frame_version_minor
        frame_group["version_minor_deg_smoothed"][:] = frame_version_minor_smoothed
        frame_group["version_minor_delta_deg"][:] = frame_version_minor_delta
        frame_group["version_minor_delta_deg_smoothed"][:] = frame_version_minor_delta_smoothed
        frame_group["left_gaze_deg"][:] = frame_left_gaze
        frame_group["left_gaze_deg_smoothed"][:] = frame_left_gaze_smoothed
        frame_group["left_gaze_delta_deg"][:] = frame_left_gaze_delta
        frame_group["left_gaze_delta_deg_smoothed"][:] = frame_left_gaze_delta_smoothed
        frame_group["right_gaze_deg"][:] = frame_right_gaze
        frame_group["right_gaze_deg_smoothed"][:] = frame_right_gaze_smoothed
        frame_group["right_gaze_delta_deg"][:] = frame_right_gaze_delta
        frame_group["right_gaze_delta_deg_smoothed"][:] = frame_right_gaze_delta_smoothed
        frame_group["left_gaze_signed_deg"][:] = frame_left_gaze_signed
        frame_group["left_gaze_signed_deg_smoothed"][:] = frame_left_gaze_signed_smoothed
        frame_group["left_gaze_signed_delta_deg"][:] = frame_left_gaze_signed_delta
        frame_group["left_gaze_signed_delta_deg_smoothed"][:] = frame_left_gaze_signed_delta_smoothed
        frame_group["right_gaze_signed_deg"][:] = frame_right_gaze_signed
        frame_group["right_gaze_signed_deg_smoothed"][:] = frame_right_gaze_signed_smoothed
        frame_group["right_gaze_signed_delta_deg"][:] = frame_right_gaze_signed_delta
        frame_group["right_gaze_signed_delta_deg_smoothed"][:] = frame_right_gaze_signed_delta_smoothed
        frame_group["vergence_gaze_deg"][:] = frame_vergence_gaze
        frame_group["vergence_gaze_deg_smoothed"][:] = frame_vergence_gaze_smoothed
        frame_group["vergence_gaze_delta_deg"][:] = frame_vergence_gaze_delta
        frame_group["vergence_gaze_delta_deg_smoothed"][:] = frame_vergence_gaze_delta_smoothed
        frame_group["vergence_gaze_signed_deg"][:] = frame_vergence_gaze_signed
        frame_group["vergence_gaze_signed_deg_smoothed"][:] = frame_vergence_gaze_signed_smoothed
        frame_group["vergence_gaze_signed_delta_deg"][:] = frame_vergence_gaze_signed_delta
        frame_group["vergence_gaze_signed_delta_deg_smoothed"][:] = frame_vergence_gaze_signed_delta_smoothed
        frame_group["left_nasal_gaze_deg"][:] = frame_left_nasal_gaze
        frame_group["left_nasal_gaze_deg_smoothed"][:] = frame_left_nasal_gaze_smoothed
        frame_group["left_nasal_gaze_delta_deg"][:] = frame_left_nasal_gaze_delta
        frame_group["left_nasal_gaze_delta_deg_smoothed"][:] = frame_left_nasal_gaze_delta_smoothed
        frame_group["right_nasal_gaze_deg"][:] = frame_right_nasal_gaze
        frame_group["right_nasal_gaze_deg_smoothed"][:] = frame_right_nasal_gaze_smoothed
        frame_group["right_nasal_gaze_delta_deg"][:] = frame_right_nasal_gaze_delta
        frame_group["right_nasal_gaze_delta_deg_smoothed"][:] = frame_right_nasal_gaze_delta_smoothed
        frame_group["mean_eye_vergence_gaze_deg"][:] = frame_mean_eye_vergence_gaze
        frame_group["mean_eye_vergence_gaze_deg_smoothed"][:] = frame_mean_eye_vergence_gaze_smoothed
        frame_group["mean_eye_vergence_gaze_delta_deg"][:] = frame_mean_eye_vergence_gaze_delta
        frame_group["mean_eye_vergence_gaze_delta_deg_smoothed"][:] = frame_mean_eye_vergence_gaze_delta_smoothed
        frame_group["version_gaze_deg"][:] = frame_version_gaze
        frame_group["version_gaze_deg_smoothed"][:] = frame_version_gaze_smoothed
        frame_group["version_gaze_delta_deg"][:] = frame_version_gaze_delta
        frame_group["version_gaze_delta_deg_smoothed"][:] = frame_version_gaze_delta_smoothed
        # Centroid-based angles
        frame_group["left_centroid_deg"][:] = frame_left_centroid
        frame_group["left_centroid_deg_smoothed"][:] = frame_left_centroid_smoothed
        frame_group["left_centroid_delta_deg"][:] = frame_left_centroid_delta
        frame_group["left_centroid_delta_deg_smoothed"][:] = frame_left_centroid_delta_smoothed
        frame_group["right_centroid_deg"][:] = frame_right_centroid
        frame_group["right_centroid_deg_smoothed"][:] = frame_right_centroid_smoothed
        frame_group["right_centroid_delta_deg"][:] = frame_right_centroid_delta
        frame_group["right_centroid_delta_deg_smoothed"][:] = frame_right_centroid_delta_smoothed
        frame_group["vergence_centroid_deg"][:] = frame_vergence_centroid
        frame_group["vergence_centroid_deg_smoothed"][:] = frame_vergence_centroid_smoothed
        frame_group["vergence_centroid_delta_deg"][:] = frame_vergence_centroid_delta
        frame_group["vergence_centroid_delta_deg_smoothed"][:] = frame_vergence_centroid_delta_smoothed

    qa_group = run_group.require_group("qa")
    qa_roi = qa_group.require_group("roi")
    qa_frame = qa_group.require_group("frame")

    _prepare_output_arrays(
        qa_roi,
        [
            ("valid_left", (total_detections,), (chunk_len,), "bool"),
            ("valid_right", (total_detections,), (chunk_len,), "bool"),
            ("valid_frame", (total_detections,), (chunk_len,), "bool"),
            ("reason_codes", (total_detections,), (chunk_len,), "u2"),
            ("left_major_axis_marginal", (total_detections,), (chunk_len,), "bool"),
            ("right_major_axis_marginal", (total_detections,), (chunk_len,), "bool"),
            ("major_axis_marginal", (total_detections,), (chunk_len,), "bool"),
        ],
    )

    _prepare_output_arrays(
        qa_frame,
        [
            ("valid_frame", (num_frames,), (frame_chunk,), "bool"),
            ("reason_codes", (num_frames,), (frame_chunk,), "u2"),
            ("major_axis_marginal", (num_frames,), (frame_chunk,), "bool"),
        ],
    )
    if num_frames > 0:
        qa_frame["valid_frame"][:] = frame_valid
        qa_frame["reason_codes"][:] = frame_reason
        qa_frame["major_axis_marginal"][:] = frame_major_axis_marginal

    support_group = run_group.require_group("support")
    storage_entries = eye_angle_storage_entries_by_path(storage_plan)
    _prepare_output_arrays(
        support_group,
        [
            ("instance_key", (total_detections,), (chunk_len,), "u8"),
            (
                "source_acquisition_frame_index",
                (total_detections,),
                (chunk_len,),
                "i8",
            ),
            ("frame_indices", (total_detections,), (chunk_len,), "i8"),
            ("time_seconds", (total_detections,), (chunk_len,), "f4"),
            ("ellipse_major", (total_detections,), (chunk_len,), "f4"),
            ("ellipse_minor", (total_detections,), (chunk_len,), "f4"),
            ("ellipse_ratio", (total_detections,), (chunk_len,), "f4"),
        ],
        storage_entries=storage_entries,
        path_prefix="support",
    )

    if num_frames > 0 and fps:
        frame_time = np.arange(num_frames, dtype=np.float32) / float(fps)
        if "frame_time_seconds" in support_group:
            del support_group["frame_time_seconds"]
        frame_time_entry = storage_entries.get("support/frame_time_seconds")
        if frame_time_entry is not None:
            create_eye_angle_array_from_entry(
                support_group,
                name="frame_time_seconds",
                entry=frame_time_entry,
                data=frame_time,
            )
        else:
            support_group.create_array(
                "frame_time_seconds",
                data=frame_time,
                chunks=(frame_chunk,),
                overwrite=True,
            )

    phase_seconds["derived_trace_and_frame_materialization"] = float(
        time.perf_counter() - derived_materialization_started
    )
    if output_layout == EYE_ANGLE_LAYOUT_COMPACT_DENSE_V2:
        if fps is None or not np.isfinite(float(fps)) or float(fps) <= 0.0:
            raise ValueError(
                "Maintained compact eye-angle v7 requires a positive finite fps."
            )
        compact_packing_started = time.perf_counter()
        _write_compact_dense_layout(
            run_group,
            total_detections=total_detections,
            num_frames=num_frames,
            chunk_len=chunk_len,
            frame_chunk=frame_chunk,
            dense_chunk_rows=int(args.dense_chunk_rows),
            dense_chunk_columns=int(args.dense_chunk_columns),
            enforce_current_schema=True,
            storage_plan=storage_plan,
        )
        phase_seconds["compact_dense_packing"] = float(
            time.perf_counter() - compact_packing_started
        )

    worker_chunk_summed_seconds = {
        key: float(sum(float(item.get(key, 0.0)) for item in chunk_timings))
        for key in ("read_seconds", "compute_seconds", "write_seconds", "total_seconds")
    }

    duration_seconds = float(time.perf_counter() - stage_start)
    rows_per_second = float(total_detections / duration_seconds) if duration_seconds > 0.0 else float("inf")
    timing_summary = {
        "total_detections": int(total_detections),
        "duration_seconds": duration_seconds,
        "rows_per_second": rows_per_second,
        "execution_backend": backend,
        "dask_scheduler": scheduler_key,
        "dask_num_workers": int(args.num_workers) if args.num_workers is not None else None,
        "dask_chunk_size": int(chunk_size),
        "dask_version": getattr(dask, "__version__", "unknown"),
        "chunk_count": len(chunks),
        "chunk_timing_count": len(chunk_timings),
        "phase_seconds": phase_seconds,
        "worker_chunk_summed_seconds": worker_chunk_summed_seconds,
    }
    fps_source = (
        str(staged_input_integrity_receipt["scientific_parameters"]["fps_source"])
        if staged_input_integrity_receipt is not None
        else "cli_override"
        if args.fps is not None and float(args.fps) > 0.0
        else "recording_metadata"
        if fps
        else "unavailable"
    )
    smoothing_window_source = (
        "cli_override" if smoothing_window_param is not None else "module_default"
    )

    # Close the staged-source TOCTOU window after every source read and before
    # any completion/provenance publication.  A fresh open avoids trusting
    # cached group/array metadata. Exact worker snapshots are the computation
    # authority; this independent full scan remains defense in depth.
    verified_root = _open_archive_for_eye_angle(args.zarr_path)
    verified_context = _resolve_eye_angle_inputs(
        verified_root,
        subject_shape_run=(
            eye_geometry.run_name
            if eye_geometry.stage_group == EYE_GEOMETRY_STAGE_SUBJECT_SHAPE
            else None
        ),
        refined_subject_run=(
            eye_geometry.run_name
            if eye_geometry.stage_group == EYE_GEOMETRY_STAGE_REFINED_SUBJECT
            else None
        ),
        keypoint_run=(
            keypoint_run_name
            if context.keypoint_source_mode == EYE_ANGLE_KEYPOINT_SOURCE_CANONICAL
            else None
        ),
        diagnostic_refined_keypoint_run=(
            keypoint_run_name
            if context.keypoint_source_mode
            == EYE_ANGLE_KEYPOINT_SOURCE_REFINED_DIAGNOSTIC
            else None
        ),
        _staged_subject_shape_authority=(
            _canonical_json_copy(staged_subject_shape_authority)
            if staged_input_integrity_receipt is not None
            else None
        ),
        _staged_keypoint_authority=(
            _canonical_json_copy(staged_keypoint_authority)
            if staged_input_integrity_receipt is not None
            else None
        ),
        _verify_staged_payload=True,
    )
    if staged_input_integrity_receipt is not None:
        verified_staged_receipt = (
            _validate_staged_eye_angle_input_integrity_receipt(
                verified_context,
                staged_input_integrity_receipt,
                verify_payload=True,
            )
        )
        if (
            verified_staged_receipt["record_sha256"]
            != staged_input_integrity_receipt["record_sha256"]
        ):
            raise RuntimeError(
                "Staged eye-angle input integrity receipt changed after source reads."
            )
    verified_input_identity = _resolved_eye_angle_input_identity(verified_context)
    if verified_input_identity != initial_input_identity:
        raise RuntimeError(
            "Eye-angle source selection or contract changed after source reads; "
            "refusing to complete the run."
        )
    context = verified_context
    eye_geometry = context.eye_geometry

    source_contracts = json.loads(
        json.dumps(
            _eye_angle_source_contracts(verified_context),
            default=_to_serializable,
        )
    )
    algorithm_contract = json.loads(
        json.dumps(
            _eye_angle_algorithm_contract(
                verified_context,
                fps=fps,
                fps_source=fps_source,
                smoothing_window_requested=int(window_setting),
                smoothing_window_source=smoothing_window_source,
                detection_smoothing_window=int(detection_smooth_window),
                frame_smoothing_window=int(frame_smooth_window),
            ),
            default=_to_serializable,
        )
    )
    run_group.attrs.update(
        {
            "status": "complete",
            "schema_id": EYE_ANGLE_RUN_SCHEMA_ID,
            "schema_version": (
                EYE_ANGLE_RUN_SCHEMA_VERSION
                if output_layout == EYE_ANGLE_LAYOUT_COMPACT_DENSE_V2
                else EYE_ANGLE_LEGACY_RUN_SCHEMA_VERSION
            ),
            "layout": output_layout,
            **(
                {"storage_profile_request": storage_profile_id}
                if storage_candidate
                else {}
            ),
            "method": EYE_ANGLE_METHOD,
            "method_version": EYE_ANGLE_METHOD_VERSION,
            "row_axis": EYE_ANGLE_ROW_AXIS,
            "report_version": "2.0",
            "reason_code_map": REASON_CODE_MAP,
            "source_eye_geometry_stage": eye_geometry.stage_group,
            "source_eye_geometry_run": eye_geometry.run_name,
            "source_geometry_kind": _source_geometry_kind(eye_geometry.stage_group),
            "source_subject_shape_run": eye_geometry.source_subject_shape_run,
            "source_refined_eye_run": eye_geometry.source_refined_eye_run,
            "source_refined_subject_masks_run": eye_geometry.source_refined_subject_run,
            "keypoint_source_mode": context.keypoint_source_mode,
            "source_base_keypoints_run": (
                context.keypoint_run_name
                if context.keypoint_source_mode
                == EYE_ANGLE_KEYPOINT_SOURCE_CANONICAL
                else context.source_kp_run_name
            ),
            "source_refined_keypoints_diagnostic_run": (
                context.keypoint_run_name
                if context.keypoint_source_mode
                == EYE_ANGLE_KEYPOINT_SOURCE_REFINED_DIAGNOSTIC
                else None
            ),
            "source_detection_success_path": context.detection_success_path,
            "source_instance_key_path": context.instance_key_path,
            "source_acquisition_frame_index_path": context.frame_indices_path,
            "source_frame_indices_path": context.frame_indices_path,
            "source_eye_geometry_authority_mode": source_authority_mode,
            "staged_input_integrity_receipt_sha256": (
                staged_input_integrity_receipt["record_sha256"]
                if staged_input_integrity_receipt is not None
                else None
            ),
            "resolved_head_keypoint_indices": {
                key: int(value) for key, value in context.keypoint_indices.items()
            },
            "eye_angle_source_contracts": source_contracts,
            "eye_angle_algorithm_contract": algorithm_contract,
            "eye_angle_output_schema": _eye_angle_output_schema(),
            "eye_angle_variant_schema": _eye_angle_variant_schema(),
            **build_source_keypoints_attrs(
                (
                    context.keypoint_run_name
                    if context.keypoint_source_mode
                    == EYE_ANGLE_KEYPOINT_SOURCE_CANONICAL
                    else context.source_kp_run_name
                ),
                include_legacy_alias=False,
            ),
            **build_keypoint_body_frame_contract_attrs(
                source_keypoints_run=(
                    keypoint_run_name
                    if context.keypoint_source_mode
                    == EYE_ANGLE_KEYPOINT_SOURCE_CANONICAL
                    else None
                ),
                source_refined_keypoints_run=(
                    keypoint_run_name
                    if context.keypoint_source_mode
                    == EYE_ANGLE_KEYPOINT_SOURCE_REFINED_DIAGNOSTIC
                    else None
                ),
                coordinate_space=BODY_FRAME_COORDINATE_SPACE_ROI,
            ),
            "fps": float(fps) if fps else None,
            "fps_source": fps_source,
            "num_detections": int(total_detections),
            "num_frames": int(num_frames),
            "duration_seconds": duration_seconds,
            "rows_per_second": rows_per_second,
            "execution_backend": backend,
            "dask_scheduler": scheduler_key,
            "dask_num_workers": int(args.num_workers) if args.num_workers is not None else None,
            "dask_chunk_size": int(chunk_size),
            "dask_version": getattr(dask, "__version__", "unknown"),
            "eye_angle_timing_summary": json.loads(json.dumps(timing_summary, default=_to_serializable)),
            "valid_detection_fraction": float(valid_frame.sum() / total_detections) if total_detections else 0.0,
            "valid_frame_fraction": float(frame_valid.sum() / num_frames) if num_frames else 0.0,
            "circularity_reject_ratio": float(ELLIPSE_CIRCULARITY_THRESHOLD),
            "major_axis_marginal_dot_threshold": float(MAJOR_AXIS_MARGINAL_DOT_THRESHOLD),
            **_eye_angle_definition_attrs(),
            "schema_migration_note": (
                "v5 makes the resolved ellipse major axis the canonical eye-orientation "
                "representation and derives gaze/minor direction from that major axis without clipping."
            ),
            "angle_smoothing_method": "moving_average",
            "angle_smoothing_algorithm": (
                "nan_aware_centered_boxcar_finite_count_normalized"
            ),
            "angle_smoothing_window_detections": int(detection_smooth_window) if detection_smooth_window else None,
            "angle_smoothing_window_frames": int(frame_smooth_window) if frame_smooth_window else None,
            "angle_smoothing_window_requested": int(window_setting),
            "angle_smoothing_window_source": smoothing_window_source,
            "angle_delta_method": "absolute_adjacent_finite_difference",
            "angle_derivative_method": "backward_difference_to_previous_valid_sample",
            "angle_derivative_max_dt_seconds": float(DERIVATIVE_MAX_DT),
            # Centroid-based eye-position angles are auxiliary pose context.
            "centroid_angles": True,
            "centroid_angle_definition": "atan2(rotated_eye_vector_y, rotated_eye_vector_x) in fish frame",
            "centroid_vergence_definition": "abs(left_centroid_deg) + abs(right_centroid_deg)",
        }
    )
    if args.include_chunk_timings:
        run_group.attrs["eye_angle_chunk_timings"] = json.loads(json.dumps(chunk_timings, default=_to_serializable))
    provenance = {
        "script": "fisheye.analysis.eye_angle_analysis",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git": get_git_info(),
        "algorithm_contract": algorithm_contract,
        "source_contracts": source_contracts,
        "execution": {
            "source_eye_geometry_authority_mode": source_authority_mode,
            "staged_input_integrity_receipt_sha256": (
                staged_input_integrity_receipt["record_sha256"]
                if staged_input_integrity_receipt is not None
                else None
            ),
        },
        "arguments": {
            "zarr_path": str(args.zarr_path),
            "eye_geometry_stage": eye_geometry.stage_group,
            "eye_geometry_run": eye_geometry.run_name,
            "subject_shape_run": eye_geometry.source_subject_shape_run,
            "refined_eye_run": eye_geometry.source_refined_eye_run,
            "refined_subject_run": eye_geometry.source_refined_subject_run,
            "keypoint_source_mode": context.keypoint_source_mode,
            "keypoint_run": keypoint_run_name,
            "base_keypoint_run": (
                keypoint_run_name
                if context.keypoint_source_mode
                == EYE_ANGLE_KEYPOINT_SOURCE_CANONICAL
                else context.source_kp_run_name
            ),
            "keypoints_roi_path": f"{context.kp_group_path}/keypoints_roi",
            "detection_success_path": context.detection_success_path,
            "instance_key_path": context.instance_key_path,
            "source_acquisition_frame_index_path": context.frame_indices_path,
            "resolved_head_keypoint_indices": {
                key: int(value) for key, value in context.keypoint_indices.items()
            },
            "run_name": args.run_name,
            "chunk_size": chunk_size,
            "execution_backend": backend,
            "dask_scheduler": scheduler_key,
            "dask_num_workers": int(args.num_workers) if args.num_workers is not None else None,
            "fps_override": args.fps,
            "smoothing_window": smoothing_window_param,
            "layout": output_layout,
            **(
                {"storage_profile": storage_profile_id}
                if storage_candidate
                else {}
            ),
        },
        "outputs": {
            "left_signed_deg": True,
            "right_signed_deg": True,
            "left_major_signed_deg": True,
            "right_major_signed_deg": True,
            "left_eye_angle_deg": True,
            "right_eye_angle_deg": True,
            "vergence_eye_angle_deg": True,
            "vergence_signed_deg": True,
            "vergence_major_signed_deg": True,
            "version_deg": True,
            "version_major_deg": True,
            "left_deg_smoothed": True,
            "right_deg_smoothed": True,
            "vergence_deg_smoothed": True,
            "left_signed_deg_smoothed": True,
            "right_signed_deg_smoothed": True,
            "vergence_signed_deg_smoothed": True,
            "version_deg_smoothed": True,
            "vergence_signed_speed_deg_s": bool(fps),
            "version_speed_deg_s": bool(fps),
            "vergence_signed_accel_deg_s2": bool(fps),
            "version_accel_deg_s2": bool(fps),
            "ellipse_major": True,
            "ellipse_minor": True,
            "ellipse_ratio": True,
            "support_body_frame": True,
            "support_body_frame_origin_xy": True,
            "support_body_frame_forward_axis_xy": True,
            "support_body_frame_left_axis_xy": True,
            "support_body_frame_heading_deg": True,
            "support_body_frame_valid": True,
            "support_body_frame_failure_reason_bytes": True,
            "left_minor_signed_deg": True,
            "right_minor_signed_deg": True,
            "vergence_minor_signed_deg": True,
            "version_minor_deg": True,
            "left_minor_signed_deg_smoothed": True,
            "right_minor_signed_deg_smoothed": True,
            "vergence_minor_signed_deg_smoothed": True,
            "version_minor_deg_smoothed": True,
            "left_delta_deg": True,
            "right_delta_deg": True,
            "vergence_delta_deg": True,
            "left_signed_delta_deg": True,
            "right_signed_delta_deg": True,
            "vergence_signed_delta_deg": True,
            "version_delta_deg": True,
            "left_eye_angle_deg_smoothed": True,
            "right_eye_angle_deg_smoothed": True,
            "vergence_eye_angle_deg_smoothed": True,
            "left_eye_angle_delta_deg": True,
            "right_eye_angle_delta_deg": True,
            "vergence_eye_angle_delta_deg": True,
            "left_eye_angle_delta_deg_smoothed": True,
            "right_eye_angle_delta_deg_smoothed": True,
            "vergence_eye_angle_delta_deg_smoothed": True,
            "left_delta_deg_smoothed": True,
            "right_delta_deg_smoothed": True,
            "vergence_delta_deg_smoothed": True,
            "left_signed_delta_deg_smoothed": True,
            "right_signed_delta_deg_smoothed": True,
            "vergence_signed_delta_deg_smoothed": True,
            "version_delta_deg_smoothed": True,
            "left_minor_signed_delta_deg": True,
            "right_minor_signed_delta_deg": True,
            "vergence_minor_signed_delta_deg": True,
            "version_minor_delta_deg": True,
            "left_minor_signed_delta_deg_smoothed": True,
            "right_minor_signed_delta_deg_smoothed": True,
            "vergence_minor_signed_delta_deg_smoothed": True,
            "version_minor_delta_deg_smoothed": True,
            "left_gaze_deg": True,
            "right_gaze_deg": True,
            "left_gaze_signed_deg": True,
            "right_gaze_signed_deg": True,
            "left_gaze_xy": True,
            "right_gaze_xy": True,
            "vergence_gaze_deg": True,
            "vergence_gaze_signed_deg": True,
            "left_nasal_gaze_deg": True,
            "right_nasal_gaze_deg": True,
            "mean_eye_vergence_gaze_deg": True,
            "version_gaze_deg": True,
            "left_gaze_deg_smoothed": True,
            "right_gaze_deg_smoothed": True,
            "left_gaze_signed_deg_smoothed": True,
            "right_gaze_signed_deg_smoothed": True,
            "vergence_gaze_deg_smoothed": True,
            "vergence_gaze_signed_deg_smoothed": True,
            "left_nasal_gaze_deg_smoothed": True,
            "right_nasal_gaze_deg_smoothed": True,
            "mean_eye_vergence_gaze_deg_smoothed": True,
            "version_gaze_deg_smoothed": True,
            "left_gaze_delta_deg": True,
            "right_gaze_delta_deg": True,
            "left_gaze_signed_delta_deg": True,
            "right_gaze_signed_delta_deg": True,
            "vergence_gaze_delta_deg": True,
            "vergence_gaze_signed_delta_deg": True,
            "left_nasal_gaze_delta_deg": True,
            "right_nasal_gaze_delta_deg": True,
            "mean_eye_vergence_gaze_delta_deg": True,
            "version_gaze_delta_deg": True,
            "left_gaze_delta_deg_smoothed": True,
            "right_gaze_delta_deg_smoothed": True,
            "left_gaze_signed_delta_deg_smoothed": True,
            "right_gaze_signed_delta_deg_smoothed": True,
            "vergence_gaze_delta_deg_smoothed": True,
            "vergence_gaze_signed_delta_deg_smoothed": True,
            "left_nasal_gaze_delta_deg_smoothed": True,
            "right_nasal_gaze_delta_deg_smoothed": True,
            "mean_eye_vergence_gaze_delta_deg_smoothed": True,
            "version_gaze_delta_deg_smoothed": True,
            "left_gaze_speed_deg_s": bool(fps),
            "right_gaze_speed_deg_s": bool(fps),
            "vergence_gaze_speed_deg_s": bool(fps),
            "vergence_gaze_signed_speed_deg_s": bool(fps),
            "mean_eye_vergence_gaze_speed_deg_s": bool(fps),
            "version_gaze_speed_deg_s": bool(fps),
            "left_gaze_accel_deg_s2": bool(fps),
            "right_gaze_accel_deg_s2": bool(fps),
            "vergence_gaze_accel_deg_s2": bool(fps),
            "vergence_gaze_signed_accel_deg_s2": bool(fps),
            "mean_eye_vergence_gaze_accel_deg_s2": bool(fps),
            "version_gaze_accel_deg_s2": bool(fps),
            "frame_left_gaze_deg": True,
            "frame_right_gaze_deg": True,
            "frame_left_gaze_signed_deg": True,
            "frame_right_gaze_signed_deg": True,
            "frame_left_eye_angle_deg": True,
            "frame_right_eye_angle_deg": True,
            "frame_vergence_eye_angle_deg": True,
            "frame_vergence_gaze_deg": True,
            "frame_vergence_gaze_signed_deg": True,
            "frame_left_nasal_gaze_deg": True,
            "frame_right_nasal_gaze_deg": True,
            "frame_mean_eye_vergence_gaze_deg": True,
            "frame_version_gaze_deg": True,
            "frame_left_gaze_deg_smoothed": True,
            "frame_right_gaze_deg_smoothed": True,
            "frame_left_gaze_signed_deg_smoothed": True,
            "frame_right_gaze_signed_deg_smoothed": True,
            "frame_left_eye_angle_deg_smoothed": True,
            "frame_right_eye_angle_deg_smoothed": True,
            "frame_vergence_eye_angle_deg_smoothed": True,
            "frame_vergence_gaze_deg_smoothed": True,
            "frame_vergence_gaze_signed_deg_smoothed": True,
            "frame_left_nasal_gaze_deg_smoothed": True,
            "frame_right_nasal_gaze_deg_smoothed": True,
            "frame_mean_eye_vergence_gaze_deg_smoothed": True,
            "frame_version_gaze_deg_smoothed": True,
            "frame_left_gaze_delta_deg": True,
            "frame_right_gaze_delta_deg": True,
            "frame_left_gaze_signed_delta_deg": True,
            "frame_right_gaze_signed_delta_deg": True,
            "frame_left_eye_angle_delta_deg": True,
            "frame_right_eye_angle_delta_deg": True,
            "frame_vergence_eye_angle_delta_deg": True,
            "frame_vergence_gaze_delta_deg": True,
            "frame_vergence_gaze_signed_delta_deg": True,
            "frame_left_nasal_gaze_delta_deg": True,
            "frame_right_nasal_gaze_delta_deg": True,
            "frame_mean_eye_vergence_gaze_delta_deg": True,
            "frame_version_gaze_delta_deg": True,
            "frame_left_gaze_delta_deg_smoothed": True,
            "frame_right_gaze_delta_deg_smoothed": True,
            "frame_left_gaze_signed_delta_deg_smoothed": True,
            "frame_right_gaze_signed_delta_deg_smoothed": True,
            "frame_left_eye_angle_delta_deg_smoothed": True,
            "frame_right_eye_angle_delta_deg_smoothed": True,
            "frame_vergence_eye_angle_delta_deg_smoothed": True,
            "frame_vergence_gaze_delta_deg_smoothed": True,
            "frame_vergence_gaze_signed_delta_deg_smoothed": True,
            "frame_left_nasal_gaze_delta_deg_smoothed": True,
            "frame_right_nasal_gaze_delta_deg_smoothed": True,
            "frame_mean_eye_vergence_gaze_delta_deg_smoothed": True,
            "frame_version_gaze_delta_deg_smoothed": True,
            "frame_left_deg_smoothed": True,
            "frame_right_deg_smoothed": True,
            "frame_vergence_deg_smoothed": True,
            "frame_vergence_signed_deg_smoothed": True,
            "frame_version_deg_smoothed": True,
            "frame_vergence_minor_signed_deg": True,
            "frame_vergence_minor_signed_deg_smoothed": True,
            "frame_vergence_major_signed_deg": True,
            "frame_vergence_major_signed_deg_smoothed": True,
            "frame_left_delta_deg": True,
            "frame_right_delta_deg": True,
            "frame_vergence_delta_deg": True,
            "frame_vergence_signed_delta_deg": True,
            "frame_version_delta_deg": True,
            "frame_left_delta_deg_smoothed": True,
            "frame_right_delta_deg_smoothed": True,
            "frame_vergence_delta_deg_smoothed": True,
            "frame_vergence_signed_delta_deg_smoothed": True,
            "frame_version_delta_deg_smoothed": True,
            "frame_vergence_minor_signed_delta_deg": True,
            "frame_vergence_minor_signed_delta_deg_smoothed": True,
            "frame_vergence_major_signed_delta_deg": True,
            "frame_vergence_major_signed_delta_deg_smoothed": True,
            "frame_version_minor_delta_deg": True,
            "frame_version_minor_delta_deg_smoothed": True,
            "frame_version_major_delta_deg": True,
            "frame_version_major_delta_deg_smoothed": True,
            "frame_version_minor_deg": True,
            "frame_version_minor_deg_smoothed": True,
            "frame_version_major_deg": True,
            "frame_version_major_deg_smoothed": True,
            "qa_major_axis_marginal": True,
            # Centroid-based eye-position angles
            "left_centroid_deg": True,
            "right_centroid_deg": True,
            "vergence_centroid_deg": True,
            "left_centroid_deg_smoothed": True,
            "right_centroid_deg_smoothed": True,
            "vergence_centroid_deg_smoothed": True,
            "left_centroid_delta_deg": True,
            "right_centroid_delta_deg": True,
            "vergence_centroid_delta_deg": True,
            "left_centroid_delta_deg_smoothed": True,
            "right_centroid_delta_deg_smoothed": True,
            "vergence_centroid_delta_deg_smoothed": True,
            "frame_left_centroid_deg": True,
            "frame_right_centroid_deg": True,
            "frame_vergence_centroid_deg": True,
            "frame_left_centroid_deg_smoothed": True,
            "frame_right_centroid_deg_smoothed": True,
            "frame_vergence_centroid_deg_smoothed": True,
            "frame_left_centroid_delta_deg": True,
            "frame_right_centroid_delta_deg": True,
            "frame_vergence_centroid_delta_deg": True,
            "frame_left_centroid_delta_deg_smoothed": True,
            "frame_right_centroid_delta_deg_smoothed": True,
            "frame_vergence_centroid_delta_deg_smoothed": True,
        },
        "valid_reason_counts": _count_reason_bits(reason_codes),
        "frame_reason_counts": _count_reason_bits(frame_reason) if num_frames else {},
    }
    run_group.attrs["provenance"] = json.loads(json.dumps(provenance, default=_to_serializable))
    if output_layout == EYE_ANGLE_LAYOUT_COMPACT_DENSE_V2:
        array_issues = validate_eye_angle_compact_run(run_group)
        alias_issues = validate_eye_angle_value_aliases(run_group)
        manifest_issues = validate_eye_angle_persisted_contract_manifests(
            run_group.attrs
        )
        if array_issues or alias_issues or manifest_issues:
            raise RuntimeError(
                "Refusing to complete an invalid compact eye-angle v7 run: "
                + "; ".join(
                    [
                        *(
                            f"{item.code}:{item.path}:{item.message}"
                            for item in array_issues
                        ),
                        *(
                            f"{item.code}:{item.path}:{item.message}"
                            for item in alias_issues
                        ),
                        *(f"persisted_manifest:{item}" for item in manifest_issues),
                    ]
                )
            )
    write_best_effort_run_lineage_attrs(run_group, run_family="eye_angle_run")
    mark_run_complete(
        run_group,
        parent_group=parent_group,
        run_name=resolved_run_name,
        run_provenance=build_writer_run_provenance(
            command="fisheye.analysis.eye_angle_analysis",
            params=provenance.get("arguments", {}),
            input_run_ids={
                "eye_geometry_stage": eye_geometry.stage_group,
                "eye_geometry_run": eye_geometry.run_name,
                "subject_shape_run": eye_geometry.source_subject_shape_run,
                "refined_eye_run": eye_geometry.source_refined_eye_run,
                "refined_subject_run": eye_geometry.source_refined_subject_run,
                "keypoint_source_mode": context.keypoint_source_mode,
                "keypoint_run": keypoint_run_name,
                "base_keypoint_run": (
                    keypoint_run_name
                    if context.keypoint_source_mode
                    == EYE_ANGLE_KEYPOINT_SOURCE_CANONICAL
                    else context.source_kp_run_name
                ),
                "eye_geometry_path": eye_geometry.group_path,
                "keypoints_path": context.kp_group_path,
                "detection_success_path": context.detection_success_path,
                "instance_key_path": context.instance_key_path,
                "source_acquisition_frame_index_path": context.frame_indices_path,
            },
        ),
    )
    if _is_selector_eligible_eye_angle_output(
        diagnostic_output=diagnostic_output,
        staged_input_integrity_receipt=staged_input_integrity_receipt,
        output_layout=output_layout,
        storage_candidate=storage_candidate,
    ):
        # Publish selectors while the completed candidate still fails closed,
        # then make eligibility the final canonical activation write. Strict
        # readers therefore cannot observe a partially written eye run.
        parent_group.attrs["latest_complete"] = resolved_run_name
        parent_group.attrs["latest"] = resolved_run_name
        run_group.attrs["stage_selector_eligible"] = True
        registry_root = open_zarr_root(args.zarr_path, mode="r")
        emit_eye_angle_stage_completion(
            registry_root,
            args.zarr_path,
            run_group=registry_root[
                f"analysis/eye_angle_runs/{resolved_run_name}"
            ],
            run_name=resolved_run_name,
            source="runtime_eye_angle_analysis",
            console=console,
        )

    if not args.quiet:
        console.print(
            f"[green]✓[/green] Eye angle analysis saved to "
            f"[cyan]analysis/eye_angle_runs/{resolved_run_name}[/cyan]"
        )
        console.print(
            f"Valid detections: {valid_frame.sum()} / {total_detections} "
            f"({(valid_frame.sum() / total_detections * 100.0) if total_detections else 0:.1f}%)"
        )


def main(
    argv: Optional[Iterable[str]] = None,
    *,
    _staged_input_integrity_receipt: Optional[Mapping[str, Any]] = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run(
        args,
        _staged_input_integrity_receipt=_staged_input_integrity_receipt,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
