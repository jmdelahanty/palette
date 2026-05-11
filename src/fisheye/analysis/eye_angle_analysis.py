#!/usr/bin/env python3
"""
Frame-wise eye angle computation for Palette archives.

This module derives head-relative eye angles, per-eye kinematics, and quality
flags from canonical refined-subject eye geometry, legacy refined-eye geometry
fallbacks, and their source keypoint headings. The results are stored under
``analysis/eye_angle_runs/<run>`` with full provenance metadata so downstream
tools can consume clean, frame-aligned measurements.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
from fisheye.shared.run_lineage_fingerprint import write_best_effort_run_lineage_attrs
from fisheye.shared.detect_reason_codec import REASON_BYTES_ENCODING, REASON_BYTES_MIN_WIDTH
from fisheye.shared.eye_geometry_source import (
    EYE_GEOMETRY_STAGE_REFINED_EYE,
    EYE_GEOMETRY_STAGE_REFINED_SUBJECT,
    EYE_GEOMETRY_STAGE_SUBJECT_SHAPE,
    resolve_eye_geometry_source,
)
from fisheye.pose.body_frame import (
    BODY_FRAME_COORDINATE_SPACE_ROI,
    BODY_FRAME_SCHEMA_ID,
    BODY_FRAME_SCHEMA_VERSION,
    build_keypoint_body_frame_contract_attrs,
    compute_keypoint_body_frame,
)
from fisheye.pose.schema import resolve_required_keypoint_indices_from_attrs
from fisheye.utils.metadata import get_fps
from fisheye.utils.system import get_git_info
from fisheye.utils.zarr_io import open_zarr_root

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
EYE_ANGLE_RUN_SCHEMA_ID = "analysis.eye_angle_runs"
EYE_ANGLE_RUN_SCHEMA_VERSION = 5
EYE_ANGLE_OUTPUT_SCHEMA_ID = "analysis.eye_angle_output_schema"
EYE_ANGLE_OUTPUT_SCHEMA_VERSION = 7
EYE_ANGLE_VARIANT_SCHEMA_ID = "analysis.eye_angle_variant_schema"
EYE_ANGLE_VARIANT_SCHEMA_VERSION = 1
EYE_ANGLE_METHOD = "ellipse_and_centroid_eye_angles"
EYE_ANGLE_METHOD_VERSION = "eye_angle_analysis.v5"
EYE_ANGLE_ROW_AXIS = "keypoint_detection_rows"
EYE_ANGLE_LAYOUT_HIERARCHICAL_V1 = "hierarchical_v1"
EYE_ANGLE_LAYOUT_COMPACT_DENSE_V2 = "compact_dense_v2"
EYE_ANGLE_LAYOUT_CHOICES = (EYE_ANGLE_LAYOUT_HIERARCHICAL_V1, EYE_ANGLE_LAYOUT_COMPACT_DENSE_V2)
MAJOR_AXIS_MARGINAL_DOT_THRESHOLD = 0.1

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
        EYE_GEOMETRY_STAGE_REFINED_EYE: "legacy_refined_eye_geometry",
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
        {"name": "frame_indices", "row_axis": "roi", "value_kind": "frame_index"},
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
        {"name": "frame_time_seconds", "row_axis": "frame", "units": "s", "optional": True},
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
    detection_success_source: zarr.Group
    detection_success_key: str
    frame_indices_source: zarr.Group
    keypoint_run_name: str
    keypoint_indices: Dict[str, int]


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
    heading_deg: np.ndarray,
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
) -> None:
    """Create (or overwrite) output arrays according to specs."""
    for name, shape, chunks, dtype in dataset_specs:
        if name in group:
            existing = group[name]
            if tuple(existing.shape) == tuple(shape) and np.dtype(existing.dtype) == np.dtype(dtype):
                continue
            del group[name]
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


def _write_text_index_array(group: zarr.Group, name: str, values: Sequence[object], *, width: int = 256) -> None:
    data = _fixed_width_text_array(values, width=width)
    if name in group:
        del group[name]
    group.create_array(name, data=data, chunks=(max(1, int(data.shape[0])), int(data.shape[1])), overwrite=True)


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


def _replace_array(group: zarr.Group, name: str, data: np.ndarray, *, chunks: tuple[int, ...]) -> None:
    if name in group:
        del group[name]
    group.create_array(name, data=data, chunks=chunks, overwrite=True)


def _eye_for_channel(name: str) -> str:
    if name.startswith("left_"):
        return "left"
    if name.startswith("right_"):
        return "right"
    if name.startswith(("vergence_", "version_", "mean_eye_vergence_")):
        return "binocular"
    return "none"


def _value_kind_for_angle_channel(name: str) -> str:
    if name.endswith("_accel_deg_s2"):
        return "acceleration"
    if name.endswith("_speed_deg_s"):
        return "speed"
    if "delta_deg" in name:
        return "delta"
    if name.startswith("vergence_") or name.startswith("mean_eye_vergence_"):
        return "vergence"
    if name.startswith("version_"):
        return "version"
    if name == "heading_deg":
        return "heading"
    return "angle"


def _units_for_angle_channel(name: str) -> str:
    if name.endswith("_accel_deg_s2"):
        return "deg/s2"
    if name.endswith("_speed_deg_s"):
        return "deg/s"
    return "deg"


def _representation_for_angle_channel(name: str) -> str:
    if "centroid" in name:
        return "centroid"
    if "nasal_gaze" in name or name.startswith("mean_eye_vergence_gaze"):
        return "nasal_gaze"
    if "eye_angle" in name:
        return "eye_frame"
    if "gaze" in name:
        return "gaze"
    if "major" in name:
        return "major"
    if "minor" in name:
        return "legacy_minor"
    if name in {"left_deg", "right_deg", "left_signed_deg", "right_signed_deg", "vergence_deg", "vergence_signed_deg"}:
        return "legacy"
    return "major" if name in {"version_deg", "heading_deg"} else "legacy"


def _alias_target_for_angle_channel(name: str) -> str:
    aliases = {
        "left_signed_deg": "left_major_signed_deg",
        "right_signed_deg": "right_major_signed_deg",
        "left_minor_signed_deg": "left_gaze_signed_deg",
        "right_minor_signed_deg": "right_gaze_signed_deg",
        "vergence_minor_signed_deg": "vergence_gaze_deg",
        "version_minor_deg": "version_gaze_deg",
        "vergence_deg": "vergence_major_signed_deg",
        "vergence_signed_deg": "vergence_major_signed_deg",
        "version_deg": "version_major_deg",
    }
    return aliases.get(name, "")


def _angle_channel_from_stem(stem: str) -> str:
    return stem if stem.endswith("_deg") else f"{stem}_deg"


def _source_channel_for_angle_channel(name: str) -> str:
    if name.endswith("_delta_deg_smoothed"):
        return _angle_channel_from_stem(name[: -len("_delta_deg_smoothed")])
    if name.endswith("_delta_deg"):
        return _angle_channel_from_stem(name[: -len("_delta_deg")])
    if name.endswith("_smoothed"):
        return name[: -len("_smoothed")]
    if name.endswith("_speed_deg_s"):
        return _angle_channel_from_stem(name[: -len("_speed_deg_s")])
    if name.endswith("_accel_deg_s2"):
        return f"{name[: -len('_accel_deg_s2')]}_speed_deg_s"
    return _alias_target_for_angle_channel(name)


def _formula_for_angle_channel(name: str) -> str:
    if name.endswith("_smoothed"):
        return "moving_average(source_channel)"
    if name.endswith("_delta_deg"):
        return "framewise_delta(source_channel)"
    if name.endswith("_delta_deg_smoothed"):
        return "framewise_delta(smoothed_source_channel)"
    if name.endswith("_speed_deg_s"):
        return "time_derivative(source_channel)"
    if name.endswith("_accel_deg_s2"):
        return "time_derivative(speed_channel)"
    formulas = {
        "left_eye_angle_deg": "-left_major_signed_deg",
        "right_eye_angle_deg": "right_major_signed_deg",
        "vergence_eye_angle_deg": "left_eye_angle_deg + right_eye_angle_deg",
        "mean_eye_vergence_gaze_deg": "0.5 * (left_nasal_gaze_deg + right_nasal_gaze_deg)",
    }
    return formulas.get(name, "")


def _write_angle_channel_index(run_group: zarr.Group, channel_names: Sequence[str]) -> None:
    group = run_group.require_group("angle_channel_index")
    for name in _array_keys(group):
        del group[name]
    _write_text_index_array(group, "name", channel_names)
    _write_text_index_array(group, "representation", [_representation_for_angle_channel(name) for name in channel_names])
    _write_text_index_array(group, "eye", [_eye_for_channel(name) for name in channel_names], width=64)
    _write_text_index_array(group, "value_kind", [_value_kind_for_angle_channel(name) for name in channel_names], width=64)
    _write_text_index_array(group, "units", [_units_for_angle_channel(name) for name in channel_names], width=64)
    _write_text_index_array(group, "source_channel", [_source_channel_for_angle_channel(name) for name in channel_names])
    _write_text_index_array(group, "formula", [_formula_for_angle_channel(name) for name in channel_names], width=512)
    _write_text_index_array(
        group,
        "compatibility_alias_of",
        [_alias_target_for_angle_channel(name) for name in channel_names],
    )
    group.attrs.update(
        {
            "channel_count": int(len(channel_names)),
            "encoding": "uint8_fixed_width_null_terminated_utf8",
            "axis": 1,
        }
    )


def _write_vector_channel_index(run_group: zarr.Group, channel_names: Sequence[str]) -> None:
    group = run_group.require_group("vector_channel_index")
    for name in _array_keys(group):
        del group[name]
    _write_text_index_array(group, "name", channel_names)
    _write_text_index_array(group, "representation", ["gaze" if "gaze" in name else "support" for name in channel_names])
    _write_text_index_array(group, "eye", [_eye_for_channel(name) for name in channel_names], width=64)
    _write_text_index_array(group, "value_kind", ["unit_vector_xy" for _name in channel_names], width=64)
    _write_text_index_array(group, "units", ["unitless" for _name in channel_names], width=64)
    group.attrs.update(
        {
            "channel_count": int(len(channel_names)),
            "encoding": "uint8_fixed_width_null_terminated_utf8",
            "axis": 1,
            "component_axis": 2,
        }
    )


def _write_qa_channel_index(run_group: zarr.Group, channel_names: Sequence[str], dtype_by_name: Mapping[str, str]) -> None:
    group = run_group.require_group("qa_channel_index")
    for name in _array_keys(group):
        del group[name]
    _write_text_index_array(group, "name", channel_names)
    _write_text_index_array(
        group,
        "value_kind",
        ["reason_code" if name == "reason_codes" else "warning_flag" if "marginal" in name else "validity_flag" for name in channel_names],
    )
    _write_text_index_array(group, "dtype", [dtype_by_name.get(name, "uint16") for name in channel_names], width=64)
    group.attrs.update(
        {
            "channel_count": int(len(channel_names)),
            "encoding": "uint8_fixed_width_null_terminated_utf8",
            "axis": 1,
        }
    )


def _write_compact_dense_layout(
    run_group: zarr.Group,
    *,
    total_detections: int,
    num_frames: int,
    chunk_len: int,
    frame_chunk: int,
) -> None:
    """Pack completed hierarchical eye-angle outputs into compact dense arrays."""

    angles_group = run_group["angles"]
    roi_group = angles_group["roi"]
    frame_group = angles_group["frame"]
    qa_group = run_group["qa"]
    qa_roi = qa_group["roi"]
    qa_frame = qa_group["frame"]

    roi_angle_names = _scalar_channel_names(roi_group, dtype_kinds="f")
    frame_angle_names = _scalar_channel_names(frame_group, dtype_kinds="f")
    angle_names = _ordered_union(roi_angle_names, frame_angle_names)
    _write_angle_channel_index(run_group, angle_names)
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
        chunks=(max(1, min(int(chunk_len), max(1, int(total_detections)))), max(1, len(angle_names))),
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
        chunks=(max(1, min(int(frame_chunk), max(1, int(num_frames)))), max(1, len(angle_names))),
    )

    roi_vector_names = _vector_channel_names(roi_group)
    frame_vector_names = _vector_channel_names(frame_group)
    vector_names = _ordered_union(roi_vector_names, frame_vector_names)
    if vector_names:
        _write_vector_channel_index(run_group, vector_names)
        _replace_array(
            run_group,
            "roi_vectors",
            _stack_vector_channels(roi_group, vector_names, row_count=total_detections),
            chunks=(max(1, min(int(chunk_len), max(1, int(total_detections)))), max(1, len(vector_names)), 2),
        )
        if frame_vector_names:
            _replace_array(
                run_group,
                "frame_vectors",
                _stack_vector_channels(frame_group, vector_names, row_count=num_frames),
                chunks=(max(1, min(int(frame_chunk), max(1, int(num_frames)))), max(1, len(vector_names)), 2),
            )

    roi_qa_names = _scalar_channel_names(qa_roi, dtype_kinds="bui")
    frame_qa_names = _scalar_channel_names(qa_frame, dtype_kinds="bui")
    qa_names = _ordered_union(roi_qa_names, frame_qa_names)
    dtype_by_name: dict[str, str] = {}
    for source_group in (qa_roi, qa_frame):
        for name in qa_names:
            if name in dtype_by_name or name not in source_group:
                continue
            dtype_by_name[name] = str(np.dtype(source_group[name].dtype))
    _write_qa_channel_index(run_group, qa_names, dtype_by_name)
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
            "compact_dense_v2_note": (
                "Eye-angle scalar channels are stored in roi_angles/frame_angles and resolved "
                "by angle_channel_index; logical hierarchical paths remain available through eye_angle_io."
            ),
        }
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute head-relative eye angles and QA flags from subject-shape or "
            "refined-subject eye geometry, with refined-eye compatibility fallback."
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
        "--refined-eye-run",
        type=str,
        help=(
            "Compatibility refined eye mask run under refined_eye_masks_runs. "
            "When it maps to refined_subject_masks_runs, canonical subject-eye geometry is used."
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
        help="Refined keypoint run providing heading and ROI coordinates (default: inferred from refined eye run or latest).",
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
        default=EYE_ANGLE_LAYOUT_HIERARCHICAL_V1,
        help=(
            "Output storage layout. hierarchical_v1 writes one array per logical field; "
            "compact_dense_v2 packs completed angle/QA outputs into dense channel tables."
        ),
    )
    return parser


def _resolve_keypoint_run_name(
    *,
    explicit_keypoint_run: Optional[str],
    refined_attrs: Dict[str, object],
    parent_latest: Optional[str],
) -> Optional[str]:
    """Resolve refined-keypoints run from explicit, canonical, legacy, then latest."""
    return (
        explicit_keypoint_run
        or resolve_source_keypoints_run(refined_attrs)
        or parent_latest
    )


def _open_archive_for_eye_angle(zarr_path: Path) -> zarr.Group:
    """Open mutable Palette zarrs with the repository's non-consolidated fallback policy."""
    return open_zarr_root(zarr_path, mode="a")


def _resolve_head_keypoint_indices(kp_group: zarr.Group) -> Dict[str, int]:
    keypoint_count = int(kp_group["keypoints_roi"].shape[1])
    try:
        return resolve_required_keypoint_indices_from_attrs(
            kp_group.attrs,
            _HEAD_KEYPOINT_LABELS,
            keypoint_count=keypoint_count,
        )
    except ValueError as exc:
        raise ValueError(
            "Keypoint run is missing canonical head labels required for eye-angle analysis "
            f"({_HEAD_KEYPOINT_LABELS}): {exc}"
        ) from exc


def _resolve_eye_angle_inputs(
    root: zarr.Group,
    *,
    subject_shape_run: Optional[str],
    refined_subject_run: Optional[str],
    refined_eye_run: Optional[str],
    keypoint_run: Optional[str],
) -> EyeAngleInputContext:
    eye_geometry = resolve_eye_geometry_source(
        root,
        subject_shape_run=subject_shape_run,
        refined_subject_run=refined_subject_run,
        refined_eye_run=refined_eye_run,
        prefer_subject_shape=True,
        prefer_subject=True,
    )

    kp_parent = root.require_group("refined_keypoints_runs")
    keypoint_run_name = _resolve_keypoint_run_name(
        explicit_keypoint_run=keypoint_run,
        refined_attrs=dict(eye_geometry.lineage_attrs),
        parent_latest=kp_parent.attrs.get("latest"),
    )
    if not keypoint_run_name or keypoint_run_name not in kp_parent:
        raise ValueError("Refined keypoint run not found; specify --keypoint-run.")
    kp_group = kp_parent[keypoint_run_name]

    source_kp_run_name = resolve_source_keypoints_run(kp_group.attrs)
    source_kp_group = None
    if source_kp_run_name:
        source_kp_parent = root.get("keypoints_runs")
        if source_kp_parent and source_kp_run_name in source_kp_parent:
            source_kp_group = source_kp_parent[source_kp_run_name]

    required_kp = ["keypoints_roi", "heading"]
    for dataset in required_kp:
        if dataset not in kp_group:
            raise ValueError(f"Keypoint run '{keypoint_run_name}' missing dataset '{dataset}'.")

    if "refined_success" in kp_group:
        detection_success_key = "refined_success"
        detection_success_source = kp_group
    elif "detection_success" in kp_group:
        detection_success_key = "detection_success"
        detection_success_source = kp_group
    elif source_kp_group is not None and "detection_success" in source_kp_group:
        detection_success_key = "detection_success"
        detection_success_source = source_kp_group
    else:
        raise ValueError(
            f"Keypoint run '{keypoint_run_name}' missing detection success data "
            "(no 'refined_success' or 'detection_success' in refined or source keypoints run)."
        )

    frame_indices_source = kp_group if "frame_indices" in kp_group else source_kp_group
    if frame_indices_source is None or "frame_indices" not in frame_indices_source:
        raise ValueError(
            f"Keypoint run '{keypoint_run_name}' missing 'frame_indices' "
            "(not in refined or source keypoints run)."
        )

    total_detections = eye_geometry.ellipse_params.shape[0]
    if kp_group["keypoints_roi"].shape[0] != total_detections:
        raise ValueError("Mismatch between eye geometry source and keypoint detections.")

    return EyeAngleInputContext(
        eye_geometry=eye_geometry,
        kp_group=kp_group,
        detection_success_source=detection_success_source,
        detection_success_key=detection_success_key,
        frame_indices_source=frame_indices_source,
        keypoint_run_name=keypoint_run_name,
        keypoint_indices=_resolve_head_keypoint_indices(kp_group),
    )


def _prepare_base_output_arrays(
    run_group: zarr.Group,
    *,
    total_detections: int,
    chunk_len: int,
) -> None:
    angles_group = run_group.require_group("angles")
    roi_group = angles_group.require_group("roi")
    qa_group = run_group.require_group("qa")
    qa_roi = qa_group.require_group("roi")
    support_group = run_group.require_group("support")
    body_frame_group = support_group.require_group("body_frame")

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
            ("frame_indices", (total_detections,), (chunk_len,), "i8"),
            ("time_seconds", (total_detections,), (chunk_len,), "f4"),
            ("ellipse_major", (total_detections,), (chunk_len,), "f4"),
            ("ellipse_minor", (total_detections,), (chunk_len,), "f4"),
            ("ellipse_ratio", (total_detections,), (chunk_len,), "f4"),
        ],
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


def _write_base_eye_angle_result(
    run_group: zarr.Group,
    row_slice: slice,
    result: EyeAngleResults,
    *,
    frame_indices: np.ndarray,
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
    ellipse_params = context.eye_geometry.ellipse_params[row_slice]
    ellipse_success = context.eye_geometry.ellipse_success[row_slice]
    keypoints_roi = context.kp_group["keypoints_roi"][row_slice]
    heading_deg = context.kp_group["heading"][row_slice]
    detection_success = context.detection_success_source[context.detection_success_key][row_slice].astype(bool, copy=False)
    frame_indices = context.frame_indices_source["frame_indices"][row_slice].astype(np.int64, copy=False)
    timing["read_seconds"] = float(time.perf_counter() - phase_start)

    phase_start = time.perf_counter()
    chunk_result = _process_chunk(
        ellipse_params=ellipse_params,
        ellipse_success=ellipse_success,
        keypoints_roi=keypoints_roi,
        heading_deg=heading_deg,
        detection_success=detection_success,
        keypoint_indices=context.keypoint_indices,
    )
    timing["compute_seconds"] = float(time.perf_counter() - phase_start)

    if fps:
        chunk_time_seconds = (frame_indices.astype(np.float64) / float(fps)).astype(np.float32, copy=False)
    else:
        chunk_time_seconds = np.full(frame_indices.shape, np.nan, dtype=np.float32)

    phase_start = time.perf_counter()
    _write_base_eye_angle_result(
        run_group,
        row_slice,
        chunk_result,
        frame_indices=frame_indices,
        time_seconds=chunk_time_seconds,
    )
    timing["write_seconds"] = float(time.perf_counter() - phase_start)
    timing["valid_frame_count"] = int(chunk_result.valid_frame.sum())
    timing["total_seconds"] = float(time.perf_counter() - chunk_start)
    return {"chunk_timing": timing, "valid_frame_count": int(chunk_result.valid_frame.sum())}


def _process_and_write_eye_angle_chunk(
    zarr_path: str,
    *,
    subject_shape_run: Optional[str],
    refined_subject_run: Optional[str],
    refined_eye_run: Optional[str],
    keypoint_run: Optional[str],
    eye_angle_run: str,
    start_row: int,
    stop_row: int,
    chunk_index: int,
    fps: Optional[float],
) -> dict[str, object]:
    root = open_zarr_root(zarr_path, mode="a")
    context = _resolve_eye_angle_inputs(
        root,
        subject_shape_run=subject_shape_run,
        refined_subject_run=refined_subject_run,
        refined_eye_run=refined_eye_run,
        keypoint_run=keypoint_run,
    )
    run_group = root["analysis"]["eye_angle_runs"][eye_angle_run]
    return _process_and_write_eye_angle_chunk_groups(
        context,
        run_group,
        start_row=start_row,
        stop_row=stop_row,
        chunk_index=chunk_index,
        fps=fps,
        execution_backend=DASK_WORKER_EXECUTION_BACKEND,
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


def run(args: argparse.Namespace) -> None:
    console = Console()
    root = _open_archive_for_eye_angle(args.zarr_path)

    analysis_group = root.require_group("analysis")
    parent_group = analysis_group.require_group("eye_angle_runs")

    backend = _normalize_execution_backend(args.execution_backend)
    scheduler_key = _normalize_scheduler(args.scheduler)
    context = _resolve_eye_angle_inputs(
        root,
        subject_shape_run=args.subject_shape_run,
        refined_subject_run=args.refined_subject_run,
        refined_eye_run=args.refined_eye_run,
        keypoint_run=args.keypoint_run,
    )
    eye_geometry = context.eye_geometry
    keypoint_run_name = context.keypoint_run_name
    total_detections = int(eye_geometry.ellipse_params.shape[0])
    chunk_size = max(1, int(args.chunk_size))
    if total_detections and chunk_size > total_detections:
        chunk_size = total_detections

    frame_indices = context.frame_indices_source["frame_indices"][:].astype(np.int64, copy=False)
    if frame_indices.shape[0] != total_detections:
        raise ValueError("Mismatch between frame_indices and detection count.")

    fps = args.fps or get_fps(root)
    if fps is None or fps <= 0:
        fps = None
    smoothing_window_param = args.smoothing_window
    valid_frame_index_mask = frame_indices >= 0
    num_frames = int(frame_indices[valid_frame_index_mask].max() + 1) if np.any(valid_frame_index_mask) else 0
    chunk_len = min(chunk_size, total_detections) if total_detections else 1
    frame_chunk = min(chunk_size, num_frames) if num_frames else 1

    if args.run_name:
        resolved_run_name = args.run_name
    else:
        resolved_run_name = datetime.now(timezone.utc).strftime("eye_angle_%Y%m%d_%H%M%S")

    if resolved_run_name in parent_group:
        raise ValueError(f"Run '{resolved_run_name}' already exists in analysis/eye_angle_runs.")

    run_group = parent_group.create_group(resolved_run_name)
    output_layout = str(args.layout)
    run_group.attrs["status"] = "running"
    run_group.attrs["layout"] = output_layout
    run_group.attrs["execution_backend"] = backend
    run_group.attrs["dask_scheduler"] = scheduler_key
    run_group.attrs["dask_num_workers"] = int(args.num_workers) if args.num_workers is not None else None
    if not args.quiet:
        console.print(f"Created run group: [cyan]analysis/eye_angle_runs/{resolved_run_name}[/cyan]")

    _prepare_base_output_arrays(run_group, total_detections=total_detections, chunk_len=chunk_len)
    run_group["support"]["body_frame"].attrs.update(
        build_keypoint_body_frame_contract_attrs(
            source_refined_keypoints_run=keypoint_run_name,
            coordinate_space=BODY_FRAME_COORDINATE_SPACE_ROI,
        )
    )
    chunks = _row_chunks(total_detections, chunk_size)
    chunk_timings: list[dict[str, object]] = []
    stage_start = time.perf_counter()

    if backend == DASK_WORKER_EXECUTION_BACKEND:
        worker_zarr_path = str(args.zarr_path.expanduser().resolve())
        worker_refined_subject_run = (
            eye_geometry.run_name if eye_geometry.stage_group == EYE_GEOMETRY_STAGE_REFINED_SUBJECT else None
        )
        worker_refined_eye_run = (
            eye_geometry.run_name if eye_geometry.stage_group == EYE_GEOMETRY_STAGE_REFINED_EYE else None
        )
        worker_subject_shape_run = (
            eye_geometry.run_name if eye_geometry.stage_group == EYE_GEOMETRY_STAGE_SUBJECT_SHAPE else None
        )
        tasks = [
            delayed(_process_and_write_eye_angle_chunk)(
                worker_zarr_path,
                subject_shape_run=worker_subject_shape_run,
                refined_subject_run=worker_refined_subject_run,
                refined_eye_run=worker_refined_eye_run,
                keypoint_run=keypoint_run_name,
                eye_angle_run=resolved_run_name,
                start_row=start_row,
                stop_row=stop_row,
                chunk_index=chunk_index,
                fps=fps,
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
            )
            for chunk_index, (start_row, stop_row) in enumerate(chunks)
        ]
    for result in sorted(results, key=lambda item: int(dict(item["chunk_timing"]).get("chunk_index") or 0)):
        chunk_timings.append(dict(result["chunk_timing"]))

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
    _prepare_output_arrays(
        support_group,
        [
            ("frame_indices", (total_detections,), (chunk_len,), "i8"),
            ("time_seconds", (total_detections,), (chunk_len,), "f4"),
            ("ellipse_major", (total_detections,), (chunk_len,), "f4"),
            ("ellipse_minor", (total_detections,), (chunk_len,), "f4"),
            ("ellipse_ratio", (total_detections,), (chunk_len,), "f4"),
        ],
    )

    if num_frames > 0 and fps:
        frame_time = np.arange(num_frames, dtype=np.float32) / float(fps)
        if "frame_time_seconds" in support_group:
            del support_group["frame_time_seconds"]
        support_group.create_array(
            "frame_time_seconds",
            data=frame_time,
            chunks=(frame_chunk,),
            overwrite=True,
        )

    if output_layout == EYE_ANGLE_LAYOUT_COMPACT_DENSE_V2:
        _write_compact_dense_layout(
            run_group,
            total_detections=total_detections,
            num_frames=num_frames,
            chunk_len=chunk_len,
            frame_chunk=frame_chunk,
        )

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
    }
    run_group.attrs.update(
        {
            "status": "complete",
            "schema_id": EYE_ANGLE_RUN_SCHEMA_ID,
            "schema_version": EYE_ANGLE_RUN_SCHEMA_VERSION,
            "layout": output_layout,
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
            "eye_angle_output_schema": _eye_angle_output_schema(),
            "eye_angle_variant_schema": _eye_angle_variant_schema(),
            **build_source_keypoints_attrs(keypoint_run_name, include_legacy_alias=True),
            **build_keypoint_body_frame_contract_attrs(
                source_refined_keypoints_run=keypoint_run_name,
                coordinate_space=BODY_FRAME_COORDINATE_SPACE_ROI,
            ),
            "fps": float(fps) if fps else None,
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
            "angle_smoothing_window_detections": int(detection_smooth_window) if detection_smooth_window else None,
            "angle_smoothing_window_frames": int(frame_smooth_window) if frame_smooth_window else None,
            "angle_smoothing_window_requested": int(smoothing_window_param) if smoothing_window_param else None,
            # Centroid-based eye-position angles are auxiliary pose context.
            "centroid_angles": True,
            "centroid_angle_definition": "atan2(rotated_eye_vector_y, rotated_eye_vector_x) in fish frame",
            "centroid_vergence_definition": "abs(left_centroid_deg) + abs(right_centroid_deg)",
        }
    )
    if args.include_chunk_timings:
        run_group.attrs["eye_angle_chunk_timings"] = json.loads(json.dumps(chunk_timings, default=_to_serializable))
    parent_group.attrs["latest"] = resolved_run_name

    provenance = {
        "script": "fisheye.analysis.eye_angle_analysis",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git": get_git_info(),
        "arguments": {
            "zarr_path": str(args.zarr_path),
            "eye_geometry_stage": eye_geometry.stage_group,
            "eye_geometry_run": eye_geometry.run_name,
            "subject_shape_run": eye_geometry.source_subject_shape_run,
            "refined_eye_run": eye_geometry.source_refined_eye_run,
            "refined_subject_run": eye_geometry.source_refined_subject_run,
            "keypoint_run": keypoint_run_name,
            "run_name": args.run_name,
            "chunk_size": chunk_size,
            "execution_backend": backend,
            "dask_scheduler": scheduler_key,
            "dask_num_workers": int(args.num_workers) if args.num_workers is not None else None,
            "fps_override": args.fps,
            "smoothing_window": smoothing_window_param,
            "layout": output_layout,
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
    write_best_effort_run_lineage_attrs(run_group, run_family="eye_angle_run")

    if not args.quiet:
        console.print(
            f"[green]✓[/green] Eye angle analysis saved to "
            f"[cyan]analysis/eye_angle_runs/{resolved_run_name}[/cyan]"
        )
        console.print(
            f"Valid detections: {valid_frame.sum()} / {total_detections} "
            f"({(valid_frame.sum() / total_detections * 100.0) if total_detections else 0:.1f}%)"
        )


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
