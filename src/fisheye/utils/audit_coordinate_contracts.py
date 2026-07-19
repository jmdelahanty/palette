"""Inventory persisted coordinate contracts without opening or mutating Zarr.

The scanner deliberately has no apply mode.  Registry access uses SQLite's
``mode=ro`` URI together with ``PRAGMA query_only`` and Zarr inspection reads
only ``zarr.json`` / Zarr-v2 metadata files.  Array payloads, consolidated
metadata, registry models, and processing code are never opened by this module.

Without recording filters, the JSONL output contains one ``coordinate_dataset``
record for every row in ``datasets`` (including rows whose path is missing),
followed by zero or more ``coordinate_surface`` records.  Filtered scans retain
selected-versus-total counts and every ``recordings``/``datasets`` row in the
normalized registry snapshot and coverage artifacts, so partial scans cannot
be mistaken for full-registry reconciliation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import math
import os
import posixpath
import re
import sqlite3
import stat
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from fisheye.shared.acquisition_publication_status import (
    ACQUISITION_AUTHORITY_NOT_PUBLISHED as _ACQUISITION_AUTHORITY_NOT_PUBLISHED,
    ACQUISITION_AUTHORITY_PENDING as _ACQUISITION_AUTHORITY_PENDING,
    ACQUISITION_AUTHORITY_PUBLISHED as _ACQUISITION_AUTHORITY_PUBLISHED,
    ACQUISITION_AUTHORITY_STATUS_ATTR as _ACQUISITION_AUTHORITY_STATUS_ATTR,
    ACQUISITION_AUTHORITY_STATUS_SCHEMA_ID as _ACQUISITION_AUTHORITY_STATUS_SCHEMA_ID,
    EXTERNAL_ACQUISITION_AUTHORITY_MODE,
    MATERIALIZED_ACQUISITION_AUTHORITY_MODE,
    AcquisitionPublicationStatusError,
    parse_acquisition_authority_publication_status,
)
from fisheye.shared.coordinate_descriptor import (
    CANONICAL_COORDINATE_DESCRIPTOR_SCHEMA_VERSION,
    PIXEL_FRAME_AUTHORITY_RECORD_KIND,
    CoordinateDescriptorError,
    CoordinateRecordRef,
    LegacySpaceContext,
    load_canonical_coordinate_descriptor_attrs,
    load_historical_coordinate_descriptor_v1_attrs,
    parse_canonical_coordinate_descriptor,
    resolve_legacy_space_id,
    validate_canonical_coordinate_descriptor,
    validate_historical_coordinate_descriptor_v1,
)
from fisheye.shared.coordinate_frame_record import (
    FRAME_RECORD_DIGEST_SUFFIX,
    PHYSICAL_FRAME_CALIBRATION_ATTR,
    PHYSICAL_FRAME_CALIBRATION_KIND,
    PHYSICAL_SOURCE_CAMERA_PROFILE_ID,
    SELECTED_CAMERA_FRAME_EVIDENCE_ATTR,
    CoordinateFrameRecordError,
    parse_physical_frame_calibration_record,
    parse_selected_camera_frame_evidence_record,
)
from fisheye.shared.coordinate_identity import (
    INSTANCE_KEY_ARRAY_REF,
    INSTANCE_KEY_MODE,
    OBSERVATION_INSTANCE_DOMAIN,
    ROW_IDENTITY_CONTRACT_ATTR,
    ROW_IDENTITY_KEY_CONTENT_CANONICALIZATION,
    STIMULUS_STATE_DOMAIN,
    STIMULUS_STATE_KEY_ARRAY_REF,
    SOURCE_ROW_ACQUISITION_FRAME_INDEX_REF,
    SOURCE_ROW_TEMPORAL_AUTHORITY_ATTR,
    SOURCE_ROW_TEMPORAL_AUTHORITY_DIGEST_ATTR,
    SOURCE_ROW_TEMPORAL_AUTHORITY_SCHEMA_ID,
    SOURCE_ROW_TEMPORAL_AUTHORITY_SCHEMA_VERSION,
    TRACK_SAMPLE_DOMAIN,
    TRACK_SAMPLE_INTERPOLATION_REF,
    TRACK_SAMPLE_SOURCE_ROW_INDEX_REF,
    TRACK_SAMPLE_SOURCE_FRAME_INDEX_REF,
    TRACK_SAMPLE_SOURCE_INSTANCE_KEY_REF,
    TRACK_SAMPLE_TIME_LINEAGE_ATTR,
    TRACK_SAMPLE_TIME_LINEAGE_DIGEST_ATTR,
    TRACK_SAMPLE_TIME_LINEAGE_SCHEMA_ID,
    TRACK_SAMPLE_TIME_LINEAGE_SCHEMA_VERSION,
    TRACK_SAMPLE_KEY_ARRAY_REF,
    RowIdentityContractError,
    load_row_identity_contract_attrs,
    load_row_identity_key_attrs,
    parse_row_identity_contract,
    validate_row_identity_contract,
)
from fisheye.shared.directed_transform import (
    CAMERA_BOUND_SPACE_IDS,
    DirectedTransformError,
    load_directed_homography_attrs,
)
from fisheye.shared.directed_transform_v2 import (
    AFFINE_2D_CONSTANT_KIND,
    AFFINE_2D_ROWWISE_KIND,
    DIRECTED_TRANSFORM_V2_ATTR,
    DIRECTED_TRANSFORM_V2_DIGEST_ATTR,
    HOMOGRAPHY_KIND,
    DirectedTransformV2Error,
    parse_directed_transform_v2,
)
from fisheye.shared.coordinate_reference import (
    ARRAY_REFERENCE_EXTENT_SCHEMA_ID,
    ATTRS_REFERENCE_EXTENT_SCHEMA_ID,
    REFERENCE_EXTENT_CANONICALIZATION,
    REFERENCE_EXTENT_SCHEMA_VERSION,
)
from fisheye.shared.pixel_frame_authority import (
    ACQUISITION_CAMERA_FRAME_SCHEMA_ID,
    ACQUISITION_CAMERA_FRAME_ATTR,
    ACQUISITION_CAMERA_FRAME_DIGEST_ATTR,
    ACQUISITION_CHUNK_MANIFEST_SCOPE,
    ACQUISITION_CHUNK_CONTENT_EVIDENCE_SCOPE,
    ACQUISITION_CHUNK_ENTRY_CANONICALIZATION,
    ACQUISITION_IMPORT_PRODUCER,
    ACQUISITION_IMPORT_OWNERSHIP_ATTR,
    ACQUISITION_IMPORT_OWNERSHIP_DIGEST_ATTR,
    ACQUISITION_IMPORT_OWNERSHIP_SCHEMA_ID,
    ACQUISITION_MATERIALIZATION_MANIFEST_ATTR,
    ACQUISITION_MATERIALIZATION_MANIFEST_DIGEST_ATTR,
    ACQUISITION_MATERIALIZATION_MANIFEST_PATH,
    ACQUISITION_MATERIALIZATION_MANIFEST_SCHEMA_ID,
    ACQUISITION_MATERIALIZATION_MANIFEST_SCHEMA_VERSION,
    ACQUISITION_MATERIALIZATION_WRITE_POLICY,
    ACQUISITION_METADATA_ONLY_VERIFICATION_SCOPE,
    ACQUISITION_PHYSICAL_CHUNK_MANIFEST_ATTR,
    ACQUISITION_PHYSICAL_CHUNK_MANIFEST_DIGEST_ATTR,
    ACQUISITION_PHYSICAL_CHUNK_MANIFEST_SCHEMA_ID,
    ACQUISITION_PHYSICAL_CHUNK_MANIFEST_SCHEMA_VERSION,
    ARENA_RELATIVE_CANVAS_FRAME_KIND,
    CROP_PLACEMENT_WINDOW_POLICY,
    CROP_PLACEMENT_OWNERSHIP_ATTR,
    CROP_PLACEMENT_OWNERSHIP_DIGEST_ATTR,
    DETECTOR_NORMALIZED_FRAME_KIND,
    MODEL_INPUT_FRAME_KIND,
    NORMALIZED_TO_PIXEL_CENTER_INDEX_V1,
    NORMALIZED_TO_PIXEL_EDGE_EXTENT_V1,
    PIXEL_FRAME_AUTHORITY_ATTR,
    PIXEL_FRAME_AUTHORITY_CANONICALIZATION,
    PIXEL_FRAME_AUTHORITY_DIGEST_ATTR,
    ROI_FRAME_KIND,
    SELECTED_CANVAS_FRAME_KIND,
    SOURCE_CAMERA_IMAGE_SPACE_ID,
    SOURCE_CAMERA_FRAME_KIND,
    SOURCE_CAMERA_NORMALIZED_FRAME_KIND,
    PixelFrameAuthorityError,
    parse_acquisition_camera_frame,
    parse_acquisition_import_ownership,
    parse_crop_placement_ownership,
    parse_pixel_frame_record,
)
from fisheye.shared.observation_coordinate_publication import (
    BBOX_CENTER_DERIVATION_ATTR,
    BBOX_CENTER_DERIVATION_OPERATION,
    BBOX_CENTER_DERIVATION_SCHEMA_ID,
    BBOX_CENTER_DERIVATION_SCHEMA_VERSION,
    CROP_GEOMETRY_SELECTION_ATTR,
    CROP_GEOMETRY_SELECTION_OPERATION,
    CROP_GEOMETRY_SELECTION_SCHEMA_ID,
    CROP_GEOMETRY_SELECTION_SCHEMA_VERSION,
    CROP_ROI_GEOMETRY_DERIVATION_ATTR,
    CROP_ROI_GEOMETRY_DERIVATION_OPERATION,
    CROP_ROI_GEOMETRY_DERIVATION_SCHEMA_ID,
    CROP_ROI_GEOMETRY_DERIVATION_SCHEMA_VERSION,
    DETECTION_ACQUISITION_MAPPING_ATTR,
    DETECTION_ACQUISITION_MAPPING_SCHEMA_ID,
    DETECTION_ACQUISITION_MAPPING_SCHEMA_VERSION,
    DETECTION_BBOX_PROJECTION_ATTR,
    DETECTION_BBOX_PROJECTION_OPERATION,
    DETECTION_BBOX_PROJECTION_SCHEMA_ID,
    DETECTION_BBOX_PROJECTION_SCHEMA_VERSION,
)
from fisheye.shared.selected_calibration import (
    CAMERA_CALIBRATION_SCHEMA_ID,
    CAMERA_CALIBRATION_SCHEMA_VERSION,
    SELECTED_CALIBRATION_SCHEMA_ID,
    SELECTED_CALIBRATION_SCHEMA_VERSION,
    SELECTED_CALIBRATION_MANIFEST_ATTR,
    SOURCE_DISPLAY_EVIDENCE_ATTR,
    SelectedCalibrationError,
    load_selected_calibration_manifest_attrs,
    load_selected_display_evidence_attrs,
    load_selected_homography_evidence_attrs,
    parse_selected_calibration_manifest,
    parse_selected_display_source_evidence,
)
from fisheye.shared.refined_subject_component_contours import (
    COMPONENT_CONTOUR_SCHEMA_ID,
)
from fisheye.shared.transform_authority import (
    ARENA_CANVAS_PLACEMENT_AUTHORITY_KIND,
    CROP_PLACEMENT_AUTHORITY_KIND,
    MODEL_INPUT_PREPROCESSING_AUTHORITY_KIND,
    NORMALIZED_TO_PIXEL_AUTHORITY_KIND,
    SELECTED_CALIBRATION_AUTHORITY_KIND,
    TRANSFORM_AUTHORITY_ATTR,
    TRANSFORM_AUTHORITY_DIGEST_ATTR,
    TransformAuthorityError,
    parse_transform_authority,
)


AUDIT_SCHEMA_ID = "palette.coordinate_contract_inventory"
AUDIT_SCHEMA_VERSION = 11
CHECKPOINT_SCHEMA_ID = "palette.coordinate_contract_inventory.dataset_checkpoint"
CHECKPOINT_SCHEMA_VERSION = 11
ARTIFACT_SCHEMA_VERSION = 11
AUDIT_RULESET_ID = "palette.coordinate_contract_inventory.rules"
AUDIT_RULESET_VERSION = 11

_MAX_METADATA_PHYSICAL_CHUNK_GRID_ENTRIES = 1_000_000

_REGISTERED_OBSERVATION_COORDINATE_RECORDS = {
    DETECTION_ACQUISITION_MAPPING_SCHEMA_ID: {
        "attribute": DETECTION_ACQUISITION_MAPPING_ATTR,
        "schema_version": DETECTION_ACQUISITION_MAPPING_SCHEMA_VERSION,
        "owner_family": "detect_runs",
    },
    DETECTION_BBOX_PROJECTION_SCHEMA_ID: {
        "attribute": DETECTION_BBOX_PROJECTION_ATTR,
        "schema_version": DETECTION_BBOX_PROJECTION_SCHEMA_VERSION,
        "owner_family": "detect_runs",
    },
    BBOX_CENTER_DERIVATION_SCHEMA_ID: {
        "attribute": BBOX_CENTER_DERIVATION_ATTR,
        "schema_version": BBOX_CENTER_DERIVATION_SCHEMA_VERSION,
        "owner_family": "detect_runs",
    },
    CROP_GEOMETRY_SELECTION_SCHEMA_ID: {
        "attribute": CROP_GEOMETRY_SELECTION_ATTR,
        "schema_version": CROP_GEOMETRY_SELECTION_SCHEMA_VERSION,
        "owner_family": "crop_runs",
    },
    CROP_ROI_GEOMETRY_DERIVATION_SCHEMA_ID: {
        "attribute": CROP_ROI_GEOMETRY_DERIVATION_ATTR,
        "schema_version": CROP_ROI_GEOMETRY_DERIVATION_SCHEMA_VERSION,
        "owner_family": "crop_runs",
    },
    SOURCE_ROW_TEMPORAL_AUTHORITY_SCHEMA_ID: {
        "attribute": SOURCE_ROW_TEMPORAL_AUTHORITY_ATTR,
        "schema_version": SOURCE_ROW_TEMPORAL_AUTHORITY_SCHEMA_VERSION,
        "owner_family": "observation_rowset",
    },
}
_REGISTERED_OBSERVATION_RECORDS_BY_ATTR = {
    str(rule["attribute"]): (schema_id, rule)
    for schema_id, rule in _REGISTERED_OBSERVATION_COORDINATE_RECORDS.items()
}

_KNOWN_ACQUISITION_SCHEMA_ATTRS = {
    ACQUISITION_CAMERA_FRAME_SCHEMA_ID: ACQUISITION_CAMERA_FRAME_ATTR,
    ACQUISITION_IMPORT_OWNERSHIP_SCHEMA_ID: ACQUISITION_IMPORT_OWNERSHIP_ATTR,
    ACQUISITION_PHYSICAL_CHUNK_MANIFEST_SCHEMA_ID: (
        ACQUISITION_PHYSICAL_CHUNK_MANIFEST_ATTR
    ),
    ACQUISITION_MATERIALIZATION_MANIFEST_SCHEMA_ID: (
        ACQUISITION_MATERIALIZATION_MANIFEST_ATTR
    ),
    _ACQUISITION_AUTHORITY_STATUS_SCHEMA_ID: _ACQUISITION_AUTHORITY_STATUS_ATTR,
}

NORMALIZED_ARTIFACT_FILENAMES = (
    "registry_snapshot.json",
    "targets.jsonl",
    "issues.jsonl",
    "issue_summary.csv",
    "archive_summary.csv",
    "coverage.json",
    "migration_manifest.jsonl",
    "artifact_manifest.json",
)

STATUSES = (
    "compatible",
    "compatible_via_explicit_legacy_rule",
    "metadata_backfill_candidate",
    "numerical_validation_required",
    "recompute_required",
    "ambiguous_fail_closed",
    "missing_or_unreadable",
    "not_applicable_unscanned",
)

# Worst status wins when surface results are rolled up to a registry row.
_STATUS_PRIORITY = {
    "compatible": 0,
    "compatible_via_explicit_legacy_rule": 1,
    "metadata_backfill_candidate": 2,
    "numerical_validation_required": 3,
    "ambiguous_fail_closed": 4,
    "recompute_required": 5,
    "missing_or_unreadable": 6,
    "not_applicable_unscanned": -1,
}

_PIXEL_OR_NORMALIZED_SURFACES = {
    "track_positions_px",
    "refined_online_positions_px",
    "detect_bbox",
    "refined_detect_bbox",
    "crop_geometry",
    "keypoint_roi",
    "keypoint_source_image",
    "keypoint_normalized",
    "stimulus_chaser_position",
    "stimulus_target_position",
    "stimulus_target_clamped_position",
    "stimulus_bbox_component",
    "occupancy_zone_bounds",
    "subject_shape_geometry",
    "subject_mask_raster",
    "subject_mask_seed_raster",
    "subject_mask_metric_geometry",
    "subject_mask_component_geometry",
    "subject_mask_contour",
    "subject_mask_compact_encoding",
    "body_frame_origin_geometry",
    "body_frame_axis_geometry",
}

_DIRECTED_TRANSFORM_SURFACE_BY_KIND = {
    HOMOGRAPHY_KIND: "directed_projective_homography",
    AFFINE_2D_CONSTANT_KIND: "directed_affine_2d_constant",
    AFFINE_2D_ROWWISE_KIND: "directed_affine_2d_rowwise",
}
_DIRECTED_TRANSFORM_SURFACES = frozenset(
    {
        *_DIRECTED_TRANSFORM_SURFACE_BY_KIND.values(),
        "directed_transform_v2_invalid",
    }
)

_DESCRIPTOR_ATTRS = (
    "coordinate_descriptor",
    "coordinate_contract",
)

_SPACE_KEYS = (
    "space_id",
    "coordinate_space_id",
    "coordinate_space",
    "coordinate_frame",
    "reference_space",
)
_UNITS_KEYS = ("units", "coordinate_units")
_ORIGIN_KEYS = ("origin", "coordinate_origin")
_X_AXIS_KEYS = ("positive_x_direction", "x_axis_direction", "x_direction")
_Y_AXIS_KEYS = ("positive_y_direction", "y_axis_direction", "y_direction")
_WIDTH_KEYS = (
    "reference_width",
    "coordinate_reference_width",
    "source_width",
    "image_width",
    "video_width",
    "texture_width",
    "canvas_width",
    "roi_width",
)
_HEIGHT_KEYS = (
    "reference_height",
    "coordinate_reference_height",
    "source_height",
    "image_height",
    "video_height",
    "texture_height",
    "canvas_height",
    "roi_height",
)
_REFERENCE_AUTHORITY_KEYS = (
    "reference_authority",
    "coordinate_reference_authority",
    "reference_extent_authority",
)
_PIXEL_CONVENTION_KEYS = (
    "pixel_convention",
    "pixel_center_convention",
    "pixel_coordinate_convention",
)
_GEOMETRY_CONVENTION_KEYS = (
    "geometry_convention",
    "coordinate_format",
    "bbox_format",
    "component_order",
)
_ROW_IDENTITY_KEYS = (
    "row_identity_ref",
    "row_identity",
    "row_axis",
    "frame_index_path",
    "frame_indices_path",
    "source_row_ids_path",
    "row_identity",
)
_SOURCE_REF_KEYS = (
    "source_ref",
    "source_coordinate_ref",
    "source_coordinate_descriptor_ref",
    "source_path",
    "position_source_path",
    "source_rowset_path",
    "source_crop_run",
    "source_detect_run",
    "source_keypoints_run",
    "source_keypoint_run",
    "lineage_refs",
)
_TRANSFORM_REF_KEYS = (
    "transform_ref",
    "coordinate_transform_ref",
    "transform_lineage_ref",
    "calibration_ref",
    "calibration_path",
    "transform_refs",
)
_TRANSFORM_DIRECTION_KEYS = (
    "transform_direction",
    "homography_direction",
    "coordinate_transform_direction",
)
_TRANSFORM_FROM_KEYS = ("from_space_id", "source_space_id", "transform_from_space")
_TRANSFORM_TO_KEYS = ("to_space_id", "target_space_id", "transform_to_space")
_OVERLAY_KEYS = (
    "source_camera_overlay_suitable",
    "suitable_for_source_camera_overlay",
    "camera_overlay_compatible",
    "source_camera_overlay",
)


@dataclass(frozen=True)
class MetadataNode:
    """One node discovered from an on-disk Zarr metadata file."""

    relative_path: str
    node_type: str | None
    metadata_format: str
    shape: Any
    data_type: Any
    chunk_shape: Any
    storage_metadata: Any
    attributes: dict[str, Any]
    metadata_error: str | None = None


class MetadataTraversalError(RuntimeError):
    """Raised when a metadata-only archive walk cannot be completed."""


@dataclass(frozen=True)
class DescriptorMatch:
    descriptor: dict[str, Any]
    source: str
    array_specific: bool
    owner_path: str
    attr_name: str | None


@dataclass(frozen=True)
class SurfaceContractProfile:
    """Explicit consumer contract for one persisted coordinate surface family."""

    profile_id: str
    geometry_types: frozenset[str]
    space_ids: frozenset[str]
    row_identity_domains: frozenset[str]
    overlay_statuses: frozenset[str]
    requires_lineage: bool = True
    requires_transform: bool = False
    rowless: bool = False


_BBOX_GEOMETRIES = frozenset(
    {"bbox_xyxy", "bbox_xywh", "bbox_cxcywh"}
)
_POINT_GEOMETRIES = frozenset(
    {"point_xy", "points_xy", "coordinate_component"}
)
_OBSERVATION_IDENTITY = frozenset({OBSERVATION_INSTANCE_DOMAIN})
_TRACK_IDENTITY = frozenset({TRACK_SAMPLE_DOMAIN})
_STIMULUS_IDENTITY = frozenset({STIMULUS_STATE_DOMAIN})

_SURFACE_PROFILES: dict[str, SurfaceContractProfile] = {
    "detect_bbox": SurfaceContractProfile(
        "detect_bbox",
        _BBOX_GEOMETRIES | _POINT_GEOMETRIES,
        frozenset(
            {
                "detector_normalized_xy",
                "source_camera_normalized_xy",
                "source_camera_image_px",
            }
        ),
        _OBSERVATION_IDENTITY,
        frozenset({"direct", "requires_transform", "not_suitable"}),
    ),
    "refined_detect_bbox": SurfaceContractProfile(
        "refined_detect_bbox",
        _BBOX_GEOMETRIES,
        frozenset({"source_camera_image_px", "source_camera_normalized_xy"}),
        _OBSERVATION_IDENTITY,
        frozenset({"direct", "requires_transform"}),
    ),
    "crop_geometry": SurfaceContractProfile(
        "crop_geometry",
        _BBOX_GEOMETRIES | _POINT_GEOMETRIES | frozenset({"raster_yx"}),
        frozenset(
            {
                "source_camera_image_px",
                "source_camera_normalized_xy",
                "roi_local_px",
            }
        ),
        _OBSERVATION_IDENTITY,
        frozenset({"direct", "requires_transform"}),
    ),
    "keypoint_roi": SurfaceContractProfile(
        "keypoint_roi",
        _POINT_GEOMETRIES,
        frozenset({"roi_local_px"}),
        _OBSERVATION_IDENTITY,
        frozenset({"requires_transform", "not_suitable"}),
    ),
    "keypoint_source_image": SurfaceContractProfile(
        "keypoint_source_image",
        _POINT_GEOMETRIES,
        frozenset({"source_camera_image_px"}),
        _OBSERVATION_IDENTITY,
        frozenset({"direct"}),
    ),
    "keypoint_normalized": SurfaceContractProfile(
        "keypoint_normalized",
        _POINT_GEOMETRIES,
        frozenset({"source_camera_normalized_xy", "detector_normalized_xy"}),
        _OBSERVATION_IDENTITY,
        frozenset({"requires_transform", "not_suitable"}),
    ),
    "keypoint_pose_bbox": SurfaceContractProfile(
        "keypoint_pose_bbox",
        _BBOX_GEOMETRIES,
        frozenset({"roi_local_px", "source_camera_image_px"}),
        _OBSERVATION_IDENTITY,
        frozenset({"direct", "requires_transform", "not_suitable"}),
    ),
    "track_positions_px": SurfaceContractProfile(
        "track_positions_px",
        _POINT_GEOMETRIES,
        frozenset(
            {
                "source_camera_image_px",
                "roi_local_px",
                "stimulus_texture_px",
                "stimulus_canvas_px",
                "arena_relative_canvas_px",
            }
        ),
        _TRACK_IDENTITY,
        frozenset({"direct", "requires_transform", "not_suitable"}),
    ),
    "track_positions_mm": SurfaceContractProfile(
        "track_positions_mm",
        _POINT_GEOMETRIES,
        frozenset({"physical_mm"}),
        _TRACK_IDENTITY,
        frozenset({"not_suitable"}),
    ),
    "refined_online_positions_px": SurfaceContractProfile(
        "refined_online_positions_px",
        _POINT_GEOMETRIES,
        frozenset(
            {"stimulus_texture_px", "stimulus_canvas_px", "arena_relative_canvas_px"}
        ),
        _STIMULUS_IDENTITY,
        frozenset({"requires_transform", "not_suitable"}),
    ),
    "refined_online_positions_mm": SurfaceContractProfile(
        "refined_online_positions_mm",
        _POINT_GEOMETRIES,
        frozenset({"physical_mm"}),
        _STIMULUS_IDENTITY,
        frozenset({"not_suitable"}),
    ),
    "stimulus_chaser_position": SurfaceContractProfile(
        "stimulus_chaser_position",
        _POINT_GEOMETRIES,
        frozenset(
            {"arena_relative_canvas_px", "stimulus_canvas_px", "stimulus_texture_px"}
        ),
        _STIMULUS_IDENTITY,
        frozenset({"not_suitable", "requires_transform"}),
    ),
    "stimulus_target_position": SurfaceContractProfile(
        "stimulus_target_position",
        _POINT_GEOMETRIES,
        frozenset(
            {"arena_relative_canvas_px", "stimulus_canvas_px", "stimulus_texture_px"}
        ),
        _STIMULUS_IDENTITY,
        frozenset({"not_suitable", "requires_transform"}),
    ),
    "stimulus_target_clamped_position": SurfaceContractProfile(
        "stimulus_target_clamped_position",
        _POINT_GEOMETRIES,
        frozenset(
            {"arena_relative_canvas_px", "stimulus_canvas_px", "stimulus_texture_px"}
        ),
        _STIMULUS_IDENTITY,
        frozenset({"not_suitable", "requires_transform"}),
    ),
    "stimulus_bbox": SurfaceContractProfile(
        "stimulus_bbox",
        _BBOX_GEOMETRIES,
        frozenset(
            {"arena_relative_canvas_px", "stimulus_canvas_px", "stimulus_texture_px"}
        ),
        _STIMULUS_IDENTITY,
        frozenset({"not_suitable", "requires_transform"}),
    ),
    "stimulus_bbox_component": SurfaceContractProfile(
        "stimulus_bbox_component",
        _POINT_GEOMETRIES,
        frozenset(
            {"arena_relative_canvas_px", "stimulus_canvas_px", "stimulus_texture_px"}
        ),
        _STIMULUS_IDENTITY,
        frozenset({"not_suitable", "requires_transform"}),
    ),
    "occupancy_zone_bounds": SurfaceContractProfile(
        "occupancy_zone_bounds",
        _BBOX_GEOMETRIES,
        frozenset({"source_camera_image_px", "arena_relative_canvas_px"}),
        frozenset({"occupancy_zone"}),
        frozenset({"direct", "not_suitable", "requires_transform"}),
    ),
    "chaser_distance_image_position": SurfaceContractProfile(
        "chaser_distance_image_position",
        _POINT_GEOMETRIES,
        frozenset({"source_camera_image_px"}),
        _TRACK_IDENTITY,
        frozenset({"direct"}),
    ),
    "chaser_distance_arena_position": SurfaceContractProfile(
        "chaser_distance_arena_position",
        _POINT_GEOMETRIES,
        frozenset({"arena_relative_canvas_px"}),
        _TRACK_IDENTITY,
        frozenset({"not_suitable", "requires_transform"}),
    ),
    "chaser_distance_px": SurfaceContractProfile(
        "chaser_distance_px",
        frozenset({"distance"}),
        frozenset({"arena_relative_canvas_px"}),
        _TRACK_IDENTITY,
        frozenset({"not_suitable"}),
    ),
    "chaser_distance_mm": SurfaceContractProfile(
        "chaser_distance_mm",
        frozenset({"distance"}),
        frozenset({"physical_mm"}),
        _TRACK_IDENTITY,
        frozenset({"not_suitable"}),
    ),
    "tracking_geometry": SurfaceContractProfile(
        "tracking_geometry",
        _BBOX_GEOMETRIES | _POINT_GEOMETRIES | frozenset({"polygon_xy"}),
        frozenset({"source_camera_image_px", "arena_relative_canvas_px"}),
        _OBSERVATION_IDENTITY,
        frozenset({"direct", "requires_transform", "not_suitable"}),
    ),
    "subject_shape_geometry": SurfaceContractProfile(
        "subject_shape_geometry",
        _POINT_GEOMETRIES
        | frozenset(
            {
                "polyline_xy",
                "polygon_xy",
                "vector_xy",
                "ellipse_cxcy_wh_angle",
            }
        ),
        frozenset(
            {"roi_local_px", "source_camera_image_px", "fish_anatomical_body_frame"}
        ),
        _OBSERVATION_IDENTITY,
        frozenset({"direct", "requires_transform", "not_suitable"}),
    ),
    "body_frame_origin_geometry": SurfaceContractProfile(
        "body_frame_origin_geometry",
        _POINT_GEOMETRIES,
        frozenset({"roi_local_px", "source_camera_image_px"}),
        _OBSERVATION_IDENTITY,
        frozenset({"direct", "requires_transform", "not_suitable"}),
    ),
    "body_frame_axis_geometry": SurfaceContractProfile(
        "body_frame_axis_geometry",
        frozenset({"vector_xy", "points_xy"}),
        frozenset({"roi_local_px", "source_camera_image_px", "fish_anatomical_body_frame"}),
        _OBSERVATION_IDENTITY,
        frozenset({"direct", "requires_transform", "not_suitable"}),
    ),
    "subject_mask_raster": SurfaceContractProfile(
        "subject_mask_raster",
        frozenset({"raster_yx"}),
        frozenset({"roi_local_px", "source_camera_image_px"}),
        _OBSERVATION_IDENTITY,
        frozenset({"direct", "requires_transform", "not_suitable"}),
    ),
    "subject_mask_seed_raster": SurfaceContractProfile(
        "subject_mask_seed_raster",
        frozenset({"raster_yx"}),
        frozenset({"roi_local_px", "source_camera_image_px"}),
        _OBSERVATION_IDENTITY,
        frozenset({"direct", "requires_transform", "not_suitable"}),
    ),
    "subject_mask_metric_geometry": SurfaceContractProfile(
        "subject_mask_metric_geometry",
        _BBOX_GEOMETRIES | _POINT_GEOMETRIES,
        frozenset({"roi_local_px", "source_camera_image_px"}),
        _OBSERVATION_IDENTITY,
        frozenset({"direct", "requires_transform", "not_suitable"}),
    ),
    "subject_mask_component_geometry": SurfaceContractProfile(
        "subject_mask_component_geometry",
        frozenset({"ellipse_cxcy_wh_angle"}),
        frozenset({"roi_local_px", "source_camera_image_px"}),
        _OBSERVATION_IDENTITY,
        frozenset({"direct", "requires_transform", "not_suitable"}),
    ),
    "subject_mask_contour": SurfaceContractProfile(
        "subject_mask_contour",
        frozenset({"polygon_xy", "polyline_xy", "points_xy"}),
        frozenset({"roi_local_px", "source_camera_image_px"}),
        _OBSERVATION_IDENTITY,
        frozenset({"direct", "requires_transform", "not_suitable"}),
    ),
    "subject_mask_compact_encoding": SurfaceContractProfile(
        "subject_mask_compact_encoding",
        frozenset({"raster_yx"}),
        frozenset({"roi_local_px", "source_camera_image_px"}),
        _OBSERVATION_IDENTITY,
        frozenset({"direct", "requires_transform", "not_suitable"}),
    ),
    "calibration_homography": SurfaceContractProfile(
        "calibration_homography",
        frozenset(),
        frozenset(),
        frozenset({"not_applicable", "singleton"}),
        frozenset(),
        requires_lineage=True,
        requires_transform=True,
        rowless=True,
    ),
    "directed_projective_homography": SurfaceContractProfile(
        "directed_projective_homography",
        frozenset(),
        frozenset(),
        frozenset({"not_applicable", "singleton"}),
        frozenset(),
        requires_lineage=True,
        requires_transform=True,
        rowless=True,
    ),
    "directed_affine_2d_constant": SurfaceContractProfile(
        "directed_affine_2d_constant",
        frozenset(),
        frozenset(),
        frozenset({"not_applicable", "singleton"}),
        frozenset(),
        requires_lineage=True,
        requires_transform=True,
        rowless=True,
    ),
    "directed_affine_2d_rowwise": SurfaceContractProfile(
        "directed_affine_2d_rowwise",
        frozenset(),
        frozenset(),
        frozenset({OBSERVATION_INSTANCE_DOMAIN}),
        frozenset(),
        requires_lineage=True,
        requires_transform=True,
    ),
    "directed_transform_v2_invalid": SurfaceContractProfile(
        "directed_transform_v2_invalid",
        frozenset(),
        frozenset(),
        frozenset(),
        frozenset(),
        requires_lineage=True,
        requires_transform=True,
        rowless=True,
    ),
}


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _reject_duplicate_json_object_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key is forbidden: {key}")
        result[key] = value
    return result


def _parse_finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite JSON float is forbidden: {value}")
    return parsed


def _require_json_unicode_scalars(value: Any) -> None:
    if isinstance(value, str):
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError("JSON strings must contain valid Unicode scalars") from exc
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _require_json_unicode_scalars(key)
            _require_json_unicode_scalars(item)
        return
    if isinstance(value, list):
        for item in value:
            _require_json_unicode_scalars(item)


def _strict_json_loads(value: str) -> Any:
    """Parse only finite, unambiguous JSON accepted by canonical writers."""

    parsed = json.loads(
        value,
        parse_constant=_reject_json_constant,
        parse_float=_parse_finite_json_float,
        object_pairs_hook=_reject_duplicate_json_object_keys,
    )
    _require_json_unicode_scalars(parsed)
    return parsed


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    if isinstance(value, str):
        try:
            parsed = _strict_json_loads(value)
        except (TypeError, ValueError):
            return {}
        if isinstance(parsed, Mapping):
            return {str(key): item for key, item in parsed.items()}
    return {}


def _canonical_json(value: Any) -> str:
    try:
        canonical = json.dumps(
            _json_safe(value),
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError):
        raise
    try:
        canonical.encode("utf-8")
    except UnicodeError as exc:
        raise ValueError(
            "Canonical JSON requires valid Unicode scalar strings."
        ) from exc
    return canonical


def _exact_json_equal(left: Any, right: Any) -> bool:
    """Compare persisted JSON without Python's bool/int or int/float coercions."""

    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        return set(left) == set(right) and all(
            _exact_json_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _exact_json_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _surface_profile_payload() -> dict[str, Any]:
    return {
        name: {
            "profile_id": profile.profile_id,
            "geometry_types": sorted(profile.geometry_types),
            "space_ids": sorted(profile.space_ids),
            "row_identity_domains": sorted(profile.row_identity_domains),
            "overlay_statuses": sorted(profile.overlay_statuses),
            "requires_lineage": profile.requires_lineage,
            "requires_transform": profile.requires_transform,
            "rowless": profile.rowless,
        }
        for name, profile in sorted(_SURFACE_PROFILES.items())
    }


def _ruleset_content_sha256() -> str:
    """Digest the declarative rules independently from a version integer."""

    return _fingerprint(
        {
            "ruleset_id": AUDIT_RULESET_ID,
            "ruleset_version": AUDIT_RULESET_VERSION,
            "surface_profiles": _surface_profile_payload(),
            "pixel_or_normalized_surfaces": sorted(_PIXEL_OR_NORMALIZED_SURFACES),
            "directed_transform_surface_by_kind": dict(
                sorted(_DIRECTED_TRANSFORM_SURFACE_BY_KIND.items())
            ),
            "mask_direct_raster_leaves_by_stage": {
                key: sorted(value)
                for key, value in sorted(
                    _MASK_DIRECT_RASTER_LEAVES_BY_STAGE.items()
                )
            }
            if "_MASK_DIRECT_RASTER_LEAVES_BY_STAGE" in globals()
            else {},
            "controlled_zarr_uses": sorted(_CONTROLLED_ZARR_USES)
            if "_CONTROLLED_ZARR_USES" in globals()
            else [],
            "controlled_zarr_origins": sorted(_CONTROLLED_ZARR_ORIGINS)
            if "_CONTROLLED_ZARR_ORIGINS" in globals()
            else [],
            "controlled_dataset_artifact_kinds": sorted(
                _CONTROLLED_DATASET_ARTIFACT_KINDS
            )
            if "_CONTROLLED_DATASET_ARTIFACT_KINDS" in globals()
            else [],
            "artifact_required_zarr_use": dict(
                sorted(_ARTIFACT_REQUIRED_ZARR_USE.items())
            )
            if "_ARTIFACT_REQUIRED_ZARR_USE" in globals()
            else {},
            "artifact_allowed_zarr_origins": {
                key: sorted(value)
                for key, value in sorted(
                    _ARTIFACT_ALLOWED_ZARR_ORIGINS.items()
                )
            }
            if "_ARTIFACT_ALLOWED_ZARR_ORIGINS" in globals()
            else {},
            "explicit_run_partitions": {
                key: sorted(value)
                for key, value in sorted(_EXPLICIT_RUN_PARTITIONS.items())
            }
            if "_EXPLICIT_RUN_PARTITIONS" in globals()
            else {},
            "registered_observation_coordinate_records": {
                key: dict(value)
                for key, value in sorted(
                    _REGISTERED_OBSERVATION_COORDINATE_RECORDS.items()
                )
            },
            "known_acquisition_schema_attrs": dict(
                sorted(_KNOWN_ACQUISITION_SCHEMA_ATTRS.items())
            ),
            "max_metadata_physical_chunk_grid_entries": (
                _MAX_METADATA_PHYSICAL_CHUNK_GRID_ENTRIES
            ),
            "status_priority": _STATUS_PRIORITY,
        }
    )


def _contract_dependency_source_sha256() -> str:
    """Bind resume data to the strict shared parsers used by this ruleset."""

    source_paths = {
        Path(callable_obj.__code__.co_filename).resolve()
        for callable_obj in (
            parse_canonical_coordinate_descriptor,
            parse_directed_transform_v2,
            parse_pixel_frame_record,
            parse_acquisition_camera_frame,
            parse_acquisition_import_ownership,
            parse_crop_placement_ownership,
            parse_physical_frame_calibration_record,
            parse_selected_camera_frame_evidence_record,
            parse_selected_calibration_manifest,
            parse_selected_display_source_evidence,
            parse_transform_authority,
            parse_row_identity_contract,
        )
    }
    records: list[dict[str, Any]] = []
    for path in sorted(source_paths):
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            digest = None
        records.append({"name": path.name, "sha256": digest})
    return _fingerprint(records)


def _scanner_source_binding() -> dict[str, Any]:
    source_path = Path(__file__).resolve()
    try:
        source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    except OSError:
        source_sha256 = None
    repository_root: Path | None = None
    for parent in source_path.parents:
        if (parent / ".git").exists():
            repository_root = parent
            break
    dirty: bool | None = None
    relative_source: str | None = None
    if repository_root is not None:
        try:
            relative_source = source_path.relative_to(repository_root).as_posix()
            result = subprocess.run(
                [
                    "git",
                    "-C",
                    str(repository_root),
                    "status",
                    "--porcelain",
                    "--untracked-files=no",
                    "--",
                    relative_source,
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            dirty = result.returncode == 0 and bool(result.stdout.strip())
            if result.returncode != 0:
                dirty = None
        except (OSError, subprocess.SubprocessError, ValueError):
            dirty = None
    payload = {
        "scanner_source_path": relative_source or source_path.name,
        "scanner_source_sha256": source_sha256,
        "scanner_source_dirty": dirty,
        "ruleset_content_sha256": _ruleset_content_sha256(),
        "contract_dependency_source_sha256": (
            _contract_dependency_source_sha256()
        ),
        "repository_commit": _repository_commit(),
    }
    payload["scanner_binding_sha256"] = _fingerprint(payload)
    return payload


def _record_bundle_digest(records: Sequence[Mapping[str, Any]]) -> str:
    unsigned: list[dict[str, Any]] = []
    for raw in records:
        record = dict(_json_safe(raw))
        record.pop("record_bundle_sha256", None)
        unsigned.append(record)
    return _fingerprint(unsigned)


def _stamp_record_bundle(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    digest = _record_bundle_digest(records)
    for record in records:
        record["record_bundle_sha256"] = digest
    return list(records)


def _atomic_write_text(path: Path, text: str) -> None:
    """Atomically replace *path* with UTF-8 *text* in its own directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def open_registry_readonly(registry_path: Path) -> sqlite3.Connection:
    """Open *registry_path* through SQLite's immutable read-only boundary.

    ``query_only`` is intentionally redundant with ``mode=ro``.  It protects
    callers if SQLite URI handling changes and is straightforward to assert in
    focused tests.
    """

    resolved = registry_path.expanduser().resolve()
    uri = f"{resolved.as_uri()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    return conn


def _table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    try:
        rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    except sqlite3.Error:
        return set()
    return {str(row[1]) for row in rows}


def _query_registry_table_rows(
    conn: sqlite3.Connection,
    table_name: str,
    *,
    registry_path: Path,
) -> list[dict[str, Any]]:
    if table_name not in {"datasets", "recordings"}:
        raise ValueError(f"unsupported registry snapshot table: {table_name}")
    columns = _table_columns(conn, table_name)
    if not columns:
        raise ValueError(f"registry has no {table_name} table: {registry_path}")
    try:
        raw_rows = conn.execute(
            f"SELECT rowid AS _registry_rowid, t.* FROM {table_name} t"
        ).fetchall()
    except sqlite3.Error:
        raw_rows = conn.execute(f"SELECT t.* FROM {table_name} t").fetchall()

    rows = [{str(key): _json_safe(row[key]) for key in row.keys()} for row in raw_rows]
    primary_key = "dataset_id" if table_name == "datasets" else "recording_id"
    path_key = "zarr_path" if table_name == "datasets" else "recording_path"
    return sorted(
        rows,
        key=lambda row: (
            str(row.get(primary_key) or ""),
            str(row.get(path_key) or ""),
            str(row.get("_registry_rowid") or ""),
            _canonical_json(row),
        ),
    )


def _read_registry_table_rows(
    registry_path: Path,
    table_name: str,
) -> list[dict[str, Any]]:
    conn = open_registry_readonly(registry_path)
    try:
        return _query_registry_table_rows(
            conn,
            table_name,
            registry_path=registry_path,
        )
    finally:
        conn.close()


def read_registry_snapshot_rows(
    registry_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read recordings and datasets from one SQLite read transaction."""

    conn = open_registry_readonly(registry_path)
    try:
        conn.execute("BEGIN")
        recording_rows = _query_registry_table_rows(
            conn,
            "recordings",
            registry_path=registry_path,
        )
        dataset_rows = _query_registry_table_rows(
            conn,
            "datasets",
            registry_path=registry_path,
        )
        return recording_rows, dataset_rows
    finally:
        conn.close()


def read_registry_dataset_rows(registry_path: Path) -> list[dict[str, Any]]:
    """Return every ``datasets`` row in deterministic order."""

    return _read_registry_table_rows(registry_path, "datasets")


def read_registry_recording_rows(registry_path: Path) -> list[dict[str, Any]]:
    """Return every ``recordings`` row in deterministic order."""

    return _read_registry_table_rows(registry_path, "recordings")


def _registry_integrity_issues(registry_path: Path) -> list[dict[str, Any]]:
    """Run SQLite integrity checks through the same read-only boundary."""

    issues: list[dict[str, Any]] = []
    conn = open_registry_readonly(registry_path)
    try:
        try:
            quick_rows = [str(row[0]) for row in conn.execute("PRAGMA quick_check")]
        except sqlite3.Error as exc:
            quick_rows = []
            issues.append(
                _issue(
                    "REGISTRY_QUICK_CHECK_FAILED",
                    "critical",
                    "SQLite quick_check could not be completed.",
                    error=str(exc),
                )
            )
        if quick_rows != ["ok"]:
            issues.append(
                _issue(
                    "REGISTRY_INTEGRITY_INVALID",
                    "critical",
                    "SQLite quick_check reported registry corruption or structural defects.",
                    quick_check=quick_rows,
                )
            )
        try:
            foreign_key_rows = [list(row) for row in conn.execute("PRAGMA foreign_key_check")]
        except sqlite3.Error as exc:
            foreign_key_rows = []
            issues.append(
                _issue(
                    "REGISTRY_FOREIGN_KEY_CHECK_FAILED",
                    "critical",
                    "SQLite foreign_key_check could not be completed.",
                    error=str(exc),
                )
            )
        if foreign_key_rows:
            issues.append(
                _issue(
                    "REGISTRY_FOREIGN_KEY_INVALID",
                    "critical",
                    "Registry rows violate declared foreign-key relationships.",
                    foreign_key_rows=foreign_key_rows,
                )
            )
    finally:
        conn.close()
    return issues


def _read_json_object(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        payload = _strict_json_loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # metadata corruption and filesystem errors are audit data
        return None, f"{type(exc).__name__}: {exc}"
    if not isinstance(payload, dict):
        return None, "metadata JSON is not an object"
    return payload, None


def _read_metadata_node(path: Path, relative_path: str) -> MetadataNode | None:
    metadata_files: dict[str, bool] = {}
    for name in ("zarr.json", ".zgroup", ".zarray", ".zattrs"):
        candidate = path / name
        try:
            mode = candidate.lstat().st_mode
        except FileNotFoundError:
            metadata_files[name] = False
            continue
        except OSError as exc:
            raise MetadataTraversalError(
                f"unable to stat metadata file {candidate}: {type(exc).__name__}: {exc}"
            ) from exc
        if stat.S_ISLNK(mode):
            raise MetadataTraversalError(f"symlinked metadata file is forbidden: {candidate}")
        if not stat.S_ISREG(mode):
            raise MetadataTraversalError(f"metadata path is not a regular file: {candidate}")
        metadata_files[name] = True

    zarr_json = path / "zarr.json"
    if metadata_files["zarr.json"]:
        payload, error = _read_json_object(zarr_json)
        payload = payload or {}
        attrs = payload.get("attributes")
        errors = [error] if error else []
        if any(metadata_files[name] for name in (".zgroup", ".zarray", ".zattrs")):
            errors.append("node mixes zarr.json with Zarr-v2 metadata files")
        if payload and (
            type(payload.get("zarr_format")) is not int
            or payload.get("zarr_format") != 3
        ):
            errors.append("zarr.json: zarr_format is not 3")
        if payload and payload.get("node_type") not in {"array", "group"}:
            errors.append("zarr.json: node_type is not 'array' or 'group'")
        if payload and not isinstance(attrs, Mapping):
            errors.append("zarr.json: attributes is not an object")
        if payload.get("node_type") == "array":
            shape = payload.get("shape")
            if not isinstance(shape, (list, tuple)) or any(
                isinstance(item, bool) or not isinstance(item, int) or item < 0
                for item in (shape or ())
            ):
                errors.append("zarr.json: array shape is invalid")
            if payload.get("data_type") in (None, ""):
                errors.append("zarr.json: array data_type is missing")
            chunk_grid = _as_mapping(payload.get("chunk_grid"))
            chunk_configuration = _as_mapping(
                chunk_grid.get("configuration")
            )
            chunk_shape = chunk_configuration.get("chunk_shape")
            if (
                chunk_grid.get("name") != "regular"
                or not isinstance(chunk_shape, (list, tuple))
                or len(chunk_shape) != len(shape or ())
                or any(
                    isinstance(item, bool)
                    or not isinstance(item, int)
                    or item <= 0
                    for item in (chunk_shape or ())
                )
            ):
                errors.append("zarr.json: regular array chunk shape is invalid")
        else:
            chunk_shape = None
        storage_metadata = dict(payload) if payload.get("node_type") == "array" else None
        if storage_metadata is not None:
            storage_metadata.pop("attributes", None)
        return MetadataNode(
            relative_path=relative_path,
            node_type=str(payload.get("node_type")) if payload.get("node_type") is not None else None,
            metadata_format="zarr.json",
            shape=payload.get("shape"),
            data_type=payload.get("data_type"),
            chunk_shape=chunk_shape,
            storage_metadata=storage_metadata,
            attributes=dict(attrs) if isinstance(attrs, Mapping) else {},
            metadata_error="; ".join(str(item) for item in errors if item) or None,
        )

    zgroup = path / ".zgroup"
    zarray = path / ".zarray"
    zattrs = path / ".zattrs"
    if any(metadata_files[name] for name in (".zgroup", ".zarray", ".zattrs")):
        attrs_payload: dict[str, Any] = {}
        errors: list[str] = []
        if metadata_files[".zgroup"]:
            group_payload, error = _read_json_object(zgroup)
            if error:
                errors.append(f".zgroup: {error}")
            elif group_payload is not None and (
                type(group_payload.get("zarr_format")) is not int
                or group_payload.get("zarr_format") != 2
            ):
                errors.append(".zgroup: zarr_format is not 2")
        if metadata_files[".zattrs"]:
            attrs, error = _read_json_object(zattrs)
            attrs_payload = attrs or {}
            if error:
                errors.append(f".zattrs: {error}")
        array_payload: dict[str, Any] = {}
        if metadata_files[".zarray"]:
            array, error = _read_json_object(zarray)
            array_payload = array or {}
            if error:
                errors.append(f".zarray: {error}")
            elif (
                type(array_payload.get("zarr_format")) is not int
                or array_payload.get("zarr_format") != 2
            ):
                errors.append(".zarray: zarr_format is not 2")
            shape = array_payload.get("shape")
            if not isinstance(shape, (list, tuple)) or any(
                isinstance(item, bool) or not isinstance(item, int) or item < 0
                for item in (shape or ())
            ):
                errors.append(".zarray: array shape is invalid")
            raw_dtype = array_payload.get("dtype")
            try:
                np.dtype(raw_dtype)
            except (TypeError, ValueError):
                errors.append(".zarray: array dtype is invalid")
            chunks = array_payload.get("chunks")
            if (
                not isinstance(chunks, (list, tuple))
                or len(chunks) != len(shape or ())
                or any(
                    isinstance(item, bool)
                    or not isinstance(item, int)
                    or item <= 0
                    for item in (chunks or ())
                )
            ):
                errors.append(".zarray: array chunks are invalid")
        if metadata_files[".zgroup"] and metadata_files[".zarray"]:
            errors.append("node contains both .zgroup and .zarray")
        if metadata_files[".zattrs"] and not (
            metadata_files[".zgroup"] or metadata_files[".zarray"]
        ):
            errors.append("node contains .zattrs without .zgroup or .zarray")
        return MetadataNode(
            relative_path=relative_path,
            node_type="array" if metadata_files[".zarray"] else "group",
            metadata_format="zarr_v2",
            shape=array_payload.get("shape"),
            data_type=array_payload.get("dtype"),
            chunk_shape=array_payload.get("chunks"),
            storage_metadata=(dict(array_payload) if array_payload else None),
            attributes=attrs_payload,
            metadata_error="; ".join(errors) or None,
        )
    return None


def _root_metadata_fingerprint(zarr_path: Path) -> str | None:
    """Digest root metadata so resume cannot rely on registry identity alone."""

    digest = hashlib.sha256()
    found = False
    for name in (".zarray", ".zattrs", ".zgroup", "zarr.json"):
        path = zarr_path / name
        try:
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                return None
            if not stat.S_ISREG(mode):
                continue
            payload = path.read_bytes()
        except FileNotFoundError:
            continue
        except OSError:
            return None
        found = True
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")
    return digest.hexdigest() if found else None


def _metadata_inventory_fingerprint(nodes: Sequence[MetadataNode]) -> str:
    """Digest every metadata node actually used for classification."""

    digest = hashlib.sha256()
    for node in sorted(nodes, key=lambda item: item.relative_path):
        payload = {
            "relative_path": node.relative_path,
            "node_type": node.node_type,
            "metadata_format": node.metadata_format,
            "shape": _json_safe(node.shape),
            "data_type": _json_safe(node.data_type),
            "chunk_shape": _json_safe(node.chunk_shape),
            "storage_metadata": _json_safe(node.storage_metadata),
            "attributes": _json_safe(node.attributes),
            "metadata_error": node.metadata_error,
        }
        digest.update(_canonical_json(payload).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _prove_non_zarr_sidecar(path: Path) -> None:
    """Prove a permitted sidecar subtree contains no nested Zarr metadata."""

    stack = [path]
    metadata_names = {"zarr.json", ".zgroup", ".zarray", ".zattrs"}
    while stack:
        current = stack.pop()
        try:
            entries = list(current.iterdir())
        except OSError as exc:
            raise MetadataTraversalError(
                f"unable to prove sidecar is non-Zarr at {current}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        for entry in entries:
            try:
                mode = entry.lstat().st_mode
            except OSError as exc:
                raise MetadataTraversalError(
                    f"unable to stat sidecar entry {entry}: {type(exc).__name__}: {exc}"
                ) from exc
            if stat.S_ISLNK(mode):
                raise MetadataTraversalError(
                    f"symlinked sidecar entry is forbidden: {entry}"
                )
            if entry.name in metadata_names:
                raise MetadataTraversalError(
                    "a directory excluded as a non-Zarr sidecar contains nested "
                    f"Zarr metadata: {entry}"
                )
            if stat.S_ISDIR(mode):
                stack.append(entry)


def iter_metadata_nodes(zarr_path: Path) -> Iterable[MetadataNode]:
    """Yield nodes found by directory metadata only, never by a Zarr API."""

    try:
        root_mode = zarr_path.lstat().st_mode
        if stat.S_ISLNK(root_mode):
            raise MetadataTraversalError(
                f"symlinked archive root is forbidden: {zarr_path}"
            )
        if not stat.S_ISDIR(root_mode):
            raise MetadataTraversalError(f"archive root is not a directory: {zarr_path}")
        archive_root = zarr_path.resolve(strict=True)
    except MetadataTraversalError:
        raise
    except OSError as exc:
        raise MetadataTraversalError(
            f"unable to resolve archive root {zarr_path}: {type(exc).__name__}: {exc}"
        ) from exc
    try:
        if not stat.S_ISDIR(archive_root.stat().st_mode):
            raise MetadataTraversalError(f"archive root is not a directory: {archive_root}")
    except OSError as exc:
        raise MetadataTraversalError(
            f"unable to stat archive root {archive_root}: {type(exc).__name__}: {exc}"
        ) from exc

    stack: list[tuple[Path, str]] = [(zarr_path, ".")]
    visited_directories: set[tuple[int, int]] = set()
    while stack:
        path, relative_path = stack.pop()
        try:
            resolved_path = path.resolve(strict=True)
            path_stat = resolved_path.stat()
        except OSError as exc:
            raise MetadataTraversalError(
                f"unable to resolve metadata directory {path}: {type(exc).__name__}: {exc}"
            ) from exc
        if not _is_same_or_descendant(resolved_path, archive_root):
            raise MetadataTraversalError(
                f"metadata directory escapes archive root: {path} -> {resolved_path}"
            )
        directory_identity = (int(path_stat.st_dev), int(path_stat.st_ino))
        if directory_identity in visited_directories:
            raise MetadataTraversalError(
                f"metadata directory cycle or duplicate directory identity at {path}"
            )
        visited_directories.add(directory_identity)
        node = _read_metadata_node(path, relative_path)
        if node is None:
            raise MetadataTraversalError(
                f"metadata directory has no zarr.json/.zgroup/.zarray metadata: {path}"
            )
        yield node
        # Zarr arrays are metadata leaves.  Their directories may contain very
        # large chunk trees; enumerating those paths is both unnecessary and
        # capable of turning a metadata-only audit into an unbounded payload
        # filesystem walk.
        if node.node_type == "array":
            continue
        try:
            entries = list(path.iterdir())
        except OSError as exc:
            raise MetadataTraversalError(
                f"unable to enumerate metadata children at {path}: {type(exc).__name__}: {exc}"
            ) from exc
        children: list[Path] = []
        for child in entries:
            try:
                child_mode = child.lstat().st_mode
            except OSError as exc:
                raise MetadataTraversalError(
                    f"unable to stat metadata child {child}: {type(exc).__name__}: {exc}"
                ) from exc
            if stat.S_ISLNK(child_mode):
                raise MetadataTraversalError(
                    f"symlinked metadata child is forbidden: {child}"
                )
            if stat.S_ISDIR(child_mode):
                children.append(child)
        discovered: list[tuple[Path, str]] = []
        for child in children:
            metadata_file_found = False
            for name in ("zarr.json", ".zgroup", ".zarray", ".zattrs"):
                candidate = child / name
                try:
                    candidate_mode = candidate.lstat().st_mode
                except FileNotFoundError:
                    continue
                except OSError as exc:
                    raise MetadataTraversalError(
                        f"unable to stat metadata file {candidate}: {type(exc).__name__}: {exc}"
                    ) from exc
                if stat.S_ISLNK(candidate_mode):
                    raise MetadataTraversalError(
                        f"symlinked metadata file is forbidden: {candidate}"
                    )
                if stat.S_ISREG(candidate_mode):
                    metadata_file_found = True
                    break
            if not metadata_file_found:
                # Palette archives may carry a root-level ``logs`` sidecar and
                # hidden run-control directories such as ``.failed``.  They are
                # explicitly outside the Zarr hierarchy and cannot contain
                # coordinate authority.  A directory with real node metadata
                # was already admitted above; all other visible implicit
                # directories still fail the complete inventory closed.
                if (
                    relative_path == "." and child.name == "logs"
                ) or child.name.startswith("."):
                    _prove_non_zarr_sidecar(child)
                    continue
                raise MetadataTraversalError(
                    "group child directory lacks explicit Zarr node metadata; "
                    f"implicit or hidden subtree cannot be audited safely: {child}"
                )
            child_relative = child.name if relative_path == "." else f"{relative_path}/{child.name}"
            discovered.append((child, child_relative))
        # Reverse push order so iteration itself is lexical.
        stack.extend(reversed(sorted(discovered, key=lambda item: item[1])))


def _path_parts(relative_path: str) -> tuple[str, ...]:
    if relative_path in ("", "."):
        return ()
    return tuple(part.lower() for part in PurePosixPath(relative_path).parts)


_COORDINATE_DECLARATION_KEYS = frozenset(
    {
        *_SPACE_KEYS,
        *_UNITS_KEYS,
        *_ORIGIN_KEYS,
        *_X_AXIS_KEYS,
        *_Y_AXIS_KEYS,
        *_WIDTH_KEYS,
        *_HEIGHT_KEYS,
        *_REFERENCE_AUTHORITY_KEYS,
        *_PIXEL_CONVENTION_KEYS,
        *_GEOMETRY_CONVENTION_KEYS,
        *_ROW_IDENTITY_KEYS,
        *_SOURCE_REF_KEYS,
        *_TRANSFORM_REF_KEYS,
        *_TRANSFORM_DIRECTION_KEYS,
        *_TRANSFORM_FROM_KEYS,
        *_TRANSFORM_TO_KEYS,
        *_OVERLAY_KEYS,
        "physical_frame",
        "directed_transform",
    }
)


def _mapping_has_coordinate_declaration(
    mapping: Mapping[str, Any],
    *,
    depth: int = 0,
) -> bool:
    if depth > 8:
        return False
    for raw_key, value in mapping.items():
        key = str(raw_key).lower()
        if (
            key in _COORDINATE_DECLARATION_KEYS
            or key in _DESCRIPTOR_ATTRS
            or key == "coordinate_descriptors"
            or key.endswith("_coordinate_descriptor")
        ):
            return True
        nested = _as_mapping(value)
        if nested and _mapping_has_coordinate_declaration(nested, depth=depth + 1):
            return True
    return False


def _node_is_coordinate_bearing(
    node: MetadataNode,
    nodes: Mapping[str, MetadataNode] | None = None,
) -> bool:
    """Return whether *node itself* carries controlled coordinate authority.

    Ancestor provenance and geometry-looking names are deliberately excluded:
    both produced large false-positive inventories (validity arrays, scalar
    signals, PNG byte stores, and entire run groups).  Unknown producers remain
    visible only when the array itself carries a descriptor schema or a
    controlled semantic role.
    """

    if node.node_type != "array":
        return False
    attrs = node.attributes
    if _node_has_direct_coordinate_descriptor(node):
        return True
    return attrs.get("semantic_role") in {
        "chaser_position",
        "target_position",
        "target_clamped_position",
        "occupancy_zone_bounds",
        "pose_bbox",
        "coordinate_geometry",
    }


def _node_has_direct_coordinate_descriptor(node: MetadataNode) -> bool:
    """Recognize an array-owned descriptor declaration, including malformed ones."""

    if node.node_type != "array":
        return False
    attrs = node.attributes
    leaf = PurePosixPath(node.relative_path).name
    if any(name in attrs for name in (*_DESCRIPTOR_ATTRS, f"{leaf}_coordinate_descriptor")):
        return True
    if "coordinate_descriptors" in attrs:
        # The container itself is a coordinate declaration.  Inventory it even
        # when it is malformed or keyed for another array so a producer cannot
        # hide an ambiguous surface merely by misspelling its array key.
        return True
    return any(str(key).endswith("_coordinate_descriptor") for key in attrs)


_NON_GEOMETRY_ARRAY_NAMES = frozenset(
    {
        "valid",
        "validity",
        "status",
        "success",
        "heading",
        "heading_deg",
        "confidence",
        "score",
        "failure_reason_bytes",
        "reason_bytes",
        "detection_occupancy_overview_png",
        "session_occupancy_overview_png",
    }
)
_NON_GEOMETRY_SUFFIXES = (
    "_valid",
    "_validity",
    "_status",
    "_success",
    "_finite",
    "_usable",
    "_reason",
    "_reason_bytes",
    "_count",
    "_signal",
    "_png",
    "_png_buffer",
    "_png_bytes",
)

_HOMOGRAPHY_ARRAY_NAMES = frozenset(
    {
        "homography",
        "homography_matrix",
        "homography_matrix_yml",
        "camera_to_canvas_homography",
        "canvas_to_camera_homography",
        "projector_to_camera_homography",
        "camera_to_projector_homography",
    }
)

_MASK_STAGE_NAMES = frozenset(
    {
        "subject_mask_runs",
        "refined_subject_masks_runs",
        "eye_masks_runs",
        "refined_eye_masks_runs",
    }
)
_LEGACY_EYE_MASK_STAGES = frozenset(
    {"eye_masks_runs", "refined_eye_masks_runs"}
)
_MASK_DIRECT_RASTER_LEAVES_BY_STAGE = {
    "subject_mask_runs": frozenset({"masks_roi", "mask_probs_roi"}),
    "refined_subject_masks_runs": frozenset(
        {"masks_roi", "mask_probs_roi"}
    ),
    "eye_masks_runs": frozenset({"masks_roi", "mask_probs_roi"}),
    "refined_eye_masks_runs": frozenset(
        {"masks_roi", "mask_probs_roi_refined"}
    ),
}


def _classify_mask_stage_surface(
    parts: tuple[str, ...],
    node: MetadataNode,
) -> str | None:
    """Classify exact writer-layout leaves, never producer-looking groups."""

    if node.node_type != "array":
        return None
    stage_index = next(
        (index for index, part in enumerate(parts) if part in _MASK_STAGE_NAMES),
        None,
    )
    if stage_index is None or len(parts) <= stage_index + 2:
        return None
    stage = parts[stage_index]
    # One run-id component is part of every stage layout.  Rules below are
    # relative to that exact run root instead of broad ancestor membership.
    relative = parts[stage_index + 2 :]
    if not relative:
        return None
    leaf = relative[-1]

    if len(relative) == 1:
        if leaf in _MASK_DIRECT_RASTER_LEAVES_BY_STAGE[stage]:
            return "subject_mask_raster"
        if stage in _LEGACY_EYE_MASK_STAGES and leaf == "ellipse_params":
            return "subject_mask_component_geometry"
        if stage in _LEGACY_EYE_MASK_STAGES and leaf in {
            "contours_left",
            "contours_right",
        }:
            return "subject_mask_contour"

    if stage not in {"subject_mask_runs", "refined_subject_masks_runs"}:
        return None

    if (
        len(relative) == 3
        and relative[0] == "components"
        and leaf == "source_seed_masks_roi"
    ):
        return "subject_mask_seed_raster"
    if relative == ("mask_bitpacked", "masks_packed"):
        return "subject_mask_compact_encoding"
    if (
        len(relative) >= 2
        and relative[0] == "metrics"
        and leaf in {"bbox_xyxy", "centroid_xy"}
    ):
        return "subject_mask_metric_geometry"
    if (
        len(relative) == 4
        and relative[0] == "components"
        and relative[2] == "geometry"
        and leaf == "ellipse_params"
    ):
        return "subject_mask_component_geometry"
    if (
        len(relative) == 4
        and relative[0] == "components"
        and relative[2] in {"contours", "sampled_contours"}
        and leaf == "points_xy"
    ):
        return "subject_mask_contour"
    if (
        len(relative) == 4
        and relative[0] == "mask_rle"
        and relative[1] == "components"
        and leaf in {"counts", "indptr"}
    ):
        return "subject_mask_compact_encoding"
    if (
        len(relative) == 4
        and relative[0] == "mask_rle"
        and relative[1] == "components"
        and leaf == "bbox_xyxy"
    ):
        return "subject_mask_metric_geometry"
    # Early subject-mask writers nested the component-local RLE cache beneath
    # components/<name>/mask_rle.  It remains an explicit historical layout.
    if (
        len(relative) == 4
        and relative[0] == "components"
        and relative[2] == "mask_rle"
        and leaf == "bbox_xyxy"
    ):
        return "subject_mask_metric_geometry"
    return None


def _is_explicitly_non_geometry_array(leaf: str, node: MetadataNode) -> bool:
    if node.node_type != "array":
        return True
    if leaf in _NON_GEOMETRY_ARRAY_NAMES or leaf.endswith(_NON_GEOMETRY_SUFFIXES):
        return True
    attrs = node.attributes
    if attrs.get("artifact_schema_id") == "palette.visualization.png_bytes.v1":
        return True
    media_type = str(attrs.get("media_type") or attrs.get("mime") or "").lower()
    if media_type.startswith("image/"):
        return True
    if attrs.get("storage_encoding") == "png_bytes_uint8":
        return True
    semantic_kind = str(
        attrs.get("semantic_kind")
        or attrs.get("surface_role")
        or attrs.get("field_classification")
        or ""
    ).lower()
    return semantic_kind in {
        "validity",
        "status",
        "scalar_metric",
        "signal",
        "visualization",
        "non_spatial",
        "row_identity",
    }


def classify_surface(
    relative_path: str,
    node: MetadataNode,
    nodes: Mapping[str, MetadataNode] | None = None,
) -> str | None:
    """Map an important persisted geometry node to a stable surface family."""

    parts = _path_parts(relative_path)
    if not parts:
        return None
    leaf = parts[-1]
    part_set = set(parts)

    # Canonical crop placement arrays deliberately host their rowwise
    # ROI-to-camera transform record.  They remain geometry payloads first;
    # the transform is validated through the descriptor lineage below.
    if "crop_runs" in part_set and leaf == "source_crop_xywh":
        return "crop_geometry"

    if DIRECTED_TRANSFORM_V2_ATTR in node.attributes:
        raw_transform = _as_mapping(
            node.attributes.get(DIRECTED_TRANSFORM_V2_ATTR)
        )
        transform_kind = raw_transform.get("kind")
        if type(transform_kind) is str:
            surface_type = _DIRECTED_TRANSFORM_SURFACE_BY_KIND.get(
                transform_kind
            )
            if surface_type is not None:
                return surface_type
        # Persisted v2-looking metadata must still be inventoried when its
        # controlled kind is absent or invalid; the strict parser later fails
        # it closed.
        return "directed_transform_v2_invalid"

    if (
        _is_explicitly_non_geometry_array(leaf, node)
        and not _node_has_direct_coordinate_descriptor(node)
    ):
        return None

    if "track_kinematics_runs" in part_set and leaf == "positions_px":
        return "track_positions_px"
    if "track_kinematics_runs" in part_set and leaf == "positions_mm":
        return "track_positions_mm"
    if any(
        part in {
            "refined_online_runs",
            "refined_online_detect_runs",
            "refined_online_detection_runs",
        }
        for part in parts
    ):
        if leaf == "positions_px":
            return "refined_online_positions_px"
        if leaf == "positions_mm":
            return "refined_online_positions_mm"

    if "tracking_data" in part_set and "chaser_states" in part_set:
        controlled_role = node.attributes.get("semantic_role")
        if leaf == "target_position_xy" and controlled_role in {
            None,
            "target_position",
        }:
            return "stimulus_target_position"
        if leaf in {"chaser_position_xy", "target_clamped_position_xy"}:
            expected_role = {
                "chaser_position_xy": "chaser_position",
                "target_clamped_position_xy": "target_clamped_position",
            }[leaf]
            if controlled_role in {None, expected_role}:
                return {
                    "chaser_position_xy": "stimulus_chaser_position",
                    "target_clamped_position_xy": (
                        "stimulus_target_clamped_position"
                    ),
                }[leaf]
    if (
        "tracking_data" in part_set
        and "chaser_states" in part_set
        and leaf
        in {
            "chaser_pos_x",
            "chaser_pos_y",
            "target_pos_x",
            "target_pos_y",
            "target_clamped_pos_x",
            "target_clamped_pos_y",
        }
    ):
        if leaf.startswith("chaser_"):
            return "stimulus_chaser_position"
        if leaf.startswith("target_clamped_"):
            return "stimulus_target_clamped_position"
        return "stimulus_target_position"

    if leaf in _HOMOGRAPHY_ARRAY_NAMES and (
        "calibration" in part_set or "calibration_runs" in part_set
    ):
        return "calibration_homography"
    if (
        node.node_type == "array"
        and list(node.shape or ()) == [3, 3]
        and "directed_transform" in node.attributes
    ):
        return "calibration_homography"

    bbox_like = leaf in {
        "bbox_norm_coords",
        "bbox_img_xyxy",
        "bbox_xyxy",
        "bboxes",
        "boxes",
        "roi_bounds",
        "crop_bounds",
        "bounds_xyxy",
        "pose_bbox_xyxy_roi",
    }
    if bbox_like and "refined_detect_runs" in part_set:
        return "refined_detect_bbox"
    if (
        (bbox_like or leaf == "centers_img_xy")
        and any(part in {"detect_runs", "detection_runs"} for part in parts)
    ):
        return "detect_bbox"

    if "crop_runs" in part_set:
        if bbox_like or leaf in {
            "bbox_roi_xyxy",
            "centers_img_xy",
            "roi_coordinates_full",
            "roi_coordinates_ds",
            "roi_centers",
            "crop_centers",
            "source_crop_xywh",
            "source_centers_px",
        }:
            return "crop_geometry"

    if any(part in {"keypoints_runs", "refined_keypoints_runs"} for part in parts):
        if leaf in {"keypoints_roi", "keypoint_roi"}:
            return "keypoint_roi"
        if leaf in {"keypoints_img", "keypoints_image", "keypoints_source_image"}:
            return "keypoint_source_image"
        if leaf in {"keypoints_norm", "keypoints_normalized"}:
            return "keypoint_normalized"
        if leaf in {"pose_bbox_xyxy_roi", "bbox_xyxy", "bbox_img_xyxy"}:
            return "keypoint_pose_bbox"

    if "chaser_distance_runs" in part_set:
        if leaf == "fish_centroid_img_xy":
            return "chaser_distance_image_position"
        if leaf in {"fish_centroid_arena_xy", "chaser_arena_xy"}:
            return "chaser_distance_arena_position"
        if leaf == "distance_px":
            return "chaser_distance_px"
        if leaf in {"distance_mm", "nearest_distance_mm"}:
            return "chaser_distance_mm"

    if "tracking_data" in part_set and "bounding_boxes" in part_set:
        if leaf in {"bbox_xyxy", "bounds_xyxy"}:
            return "stimulus_bbox"
        if leaf in {
            "x_min",
            "y_min",
            "width",
            "height",
            "centroid_x",
            "centroid_y",
        }:
            return "stimulus_bbox_component"

    if (
        any(part in {"detection_occupancy_runs", "session_occupancy_runs"} for part in parts)
        and "spatial_occupancy" in part_set
        and "zone_spec" in part_set
        and leaf == "bounds_xyxy"
    ):
        return "occupancy_zone_bounds"

    if any(
        part in {"tracking_runs", "arena_assignment_runs", "arena_assignments"}
        for part in parts
    ) and leaf in {
        "bbox_xyxy",
        "bbox_img_xyxy",
        "centroid_xy",
        "position_xy",
        "polygon_xy",
    }:
        return "tracking_geometry"

    if any(part in {"body_frame_runs", "body_frame", "body_frames"} for part in parts):
        if leaf == "origin_xy":
            return "body_frame_origin_geometry"
        if leaf in {"forward_axis_xy", "left_axis_xy"}:
            return "body_frame_axis_geometry"

    subject_shape = any(part in {"subject_shape_runs", "subject_shapes_runs"} for part in parts)
    subject_shape_geometry_leaves = {
        "centroid_xy",
        "bbox_xyxy",
        "principal_axis_xy",
        "centerline_xy",
        "centerline_spline_xy",
        "bspline_control_points_xy",
        "bspline_sample_xy",
        "tail_sample_xy",
        "tail_tangent_xy",
        "tail_normal_xy",
        "snout_tip_xy",
        "head_endpoint_xy",
        "tail_tip_xy",
        "tail_base_xy",
        "caudal_contour_point_xy",
        "midpoint_xy",
        "left_eye_offset_xy",
        "right_eye_offset_xy",
        "ellipse_params",
    }
    if subject_shape and leaf in subject_shape_geometry_leaves:
        return "subject_shape_geometry"

    mask_surface = _classify_mask_stage_surface(parts, node)
    if mask_surface is not None:
        return mask_surface

    # Unknown producers are inventoried only through direct controlled schema
    # or semantic-role evidence.  Names and ancestor attrs are not authority.
    if node.node_type == "array" and _node_is_coordinate_bearing(node, nodes):
        return "unclassified_geometry_candidate"
    return None


def _ancestor_paths(relative_path: str) -> list[str]:
    if relative_path in ("", "."):
        return ["."]
    path = PurePosixPath(relative_path)
    result: list[str] = []
    parent = path.parent
    while str(parent) not in ("", "."):
        result.append(parent.as_posix())
        parent = parent.parent
    result.append(".")
    return result


def _deep_find(mapping: Mapping[str, Any], keys: Sequence[str], *, prefix: str = "", depth: int = 0) -> tuple[Any, str] | None:
    if depth > 8:
        return None
    wanted = {str(key).lower() for key in keys}
    for key in sorted(mapping, key=lambda item: str(item)):
        value = mapping[key]
        key_text = str(key)
        location = f"{prefix}.{key_text}" if prefix else key_text
        if key_text.lower() in wanted and value not in (None, ""):
            return value, location
    for key in sorted(mapping, key=lambda item: str(item)):
        nested = _as_mapping(mapping[key])
        if not nested:
            continue
        key_text = str(key)
        location = f"{prefix}.{key_text}" if prefix else key_text
        found = _deep_find(nested, keys, prefix=location, depth=depth + 1)
        if found:
            return found
    return None


def _find_declared(
    node: MetadataNode,
    nodes: Mapping[str, MetadataNode],
    keys: Sequence[str],
    *,
    include_nested: bool = True,
) -> tuple[Any, str] | None:
    paths = [node.relative_path, *_ancestor_paths(node.relative_path)]
    seen: set[str] = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        candidate = nodes.get(path)
        if candidate is None:
            continue
        direct = _deep_find(candidate.attributes, keys) if include_nested else None
        if not include_nested:
            for key in keys:
                value = candidate.attributes.get(key)
                if value not in (None, ""):
                    direct = (value, key)
                    break
        if direct:
            value, attr_path = direct
            return value, f"{path}:{attr_path}"
    return None


def _find_producer_declared(
    node: MetadataNode,
    nodes: Mapping[str, MetadataNode],
    keys: Sequence[str],
) -> tuple[Any, str] | None:
    """Read producer/run declarations without consulting the output descriptor."""

    for path in _ancestor_paths(node.relative_path):
        candidate = nodes.get(path)
        if candidate is None:
            continue
        attrs = {
            str(name): value
            for name, value in candidate.attributes.items()
            if name not in {*_DESCRIPTOR_ATTRS, "coordinate_descriptors"}
            and not str(name).endswith("_coordinate_descriptor")
        }
        direct = _deep_find(attrs, keys)
        if direct:
            value, attr_path = direct
            return value, f"{path}:{attr_path}"
    return None


def _surface_prefixed_keys(node: MetadataNode, keys: Sequence[str]) -> tuple[str, ...]:
    leaf = PurePosixPath(node.relative_path).name
    prefixed = [f"{leaf}_{key}" for key in keys]
    # Persisted bbox contracts commonly use a shorter stem.
    if leaf.startswith("bbox_norm"):
        prefixed.extend(f"bbox_norm_{key.removeprefix('coordinate_')}" for key in keys)
    if leaf.startswith("bbox_img"):
        prefixed.extend(f"bbox_img_xyxy_{key.removeprefix('coordinate_')}" for key in keys)
    return tuple(dict.fromkeys([*prefixed, *keys]))


def _all_descriptor_matches(
    node: MetadataNode,
    nodes: Mapping[str, MetadataNode],
) -> list[DescriptorMatch]:
    leaf = PurePosixPath(node.relative_path).name
    matches: list[DescriptorMatch] = []
    paths = [node.relative_path, *_ancestor_paths(node.relative_path)]
    for path_index, owner_path in enumerate(paths):
        owner = nodes.get(owner_path)
        if owner is None:
            continue
        attrs = owner.attributes
        array_specific = path_index == 0
        for name in (f"{leaf}_coordinate_descriptor", *_DESCRIPTOR_ATTRS):
            descriptor = _as_mapping(attrs.get(name))
            if descriptor:
                matches.append(
                    DescriptorMatch(
                        descriptor=descriptor,
                        source=f"{owner_path}:{name}",
                        array_specific=array_specific,
                        owner_path=owner_path,
                        attr_name=name,
                    )
                )
        descriptors = _as_mapping(attrs.get("coordinate_descriptors"))
        for key in dict.fromkeys((leaf, node.relative_path)):
            descriptor = _as_mapping(descriptors.get(key))
            if descriptor:
                matches.append(
                    DescriptorMatch(
                        descriptor=descriptor,
                        source=f"{owner_path}:coordinate_descriptors.{key}",
                        array_specific=array_specific,
                        owner_path=owner_path,
                        attr_name=None,
                    )
                )
    # Keep deterministic declaration order while eliminating an exact duplicate
    # source that can arise when leaf == relative_path at the archive root.
    deduplicated: dict[str, DescriptorMatch] = {}
    for match in matches:
        deduplicated.setdefault(match.source, match)
    return list(deduplicated.values())


def _find_descriptor(
    node: MetadataNode,
    nodes: Mapping[str, MetadataNode],
) -> DescriptorMatch | None:
    matches = _all_descriptor_matches(node, nodes)
    return matches[0] if matches else None


def _descriptor_declaration_issues(
    node: MetadataNode,
    nodes: Mapping[str, MetadataNode],
) -> list[dict[str, Any]]:
    matches = _all_descriptor_matches(node, nodes)
    issues: list[dict[str, Any]] = []
    if "coordinate_descriptors" in node.attributes:
        raw_descriptors = node.attributes.get("coordinate_descriptors")
        descriptors = _as_mapping(raw_descriptors)
        leaf = PurePosixPath(node.relative_path).name
        applicable_keys = tuple(dict.fromkeys((leaf, node.relative_path)))
        declared_keys = set(descriptors)
        unexpected_keys = sorted(declared_keys.difference(applicable_keys))
        applicable_values = [
            descriptors.get(key) for key in applicable_keys if key in descriptors
        ]
        if (
            type(raw_descriptors) is not dict
            or not applicable_values
            or any(type(value) is not dict for value in applicable_values)
            or unexpected_keys
        ):
            issues.append(
                _issue(
                    "ARRAY_COORDINATE_DESCRIPTORS_CONTAINER_INVALID",
                    "critical",
                    "An array-owned coordinate_descriptors container must contain only the exact array leaf and/or archive-relative path keys, with object-valued descriptors; unrelated entries are contradictory declarations and fail closed.",
                    surface_path=node.relative_path,
                    expected_keys=list(applicable_keys),
                    declared_keys=sorted(descriptors),
                    unexpected_keys=unexpected_keys,
                    container_type=type(raw_descriptors).__name__,
                )
            )
    leaf = PurePosixPath(node.relative_path).name
    expected_array_specific_attr = f"{leaf}_coordinate_descriptor"
    miskeyed_descriptor_attrs = sorted(
        str(name)
        for name in node.attributes
        if str(name).endswith("_coordinate_descriptor")
        and str(name) not in {*_DESCRIPTOR_ATTRS, expected_array_specific_attr}
    )
    if miskeyed_descriptor_attrs:
        issues.append(
            _issue(
                "ARRAY_COORDINATE_DESCRIPTOR_ATTR_MISKEYED",
                "critical",
                "An array-owned descriptor attribute is keyed for a different surface; mis-keyed coordinate declarations cannot be ignored.",
                surface_path=node.relative_path,
                expected_attr=expected_array_specific_attr,
                declared_attrs=miskeyed_descriptor_attrs,
            )
        )
    if not matches:
        return issues
    descriptor_digests = {
        _fingerprint(match.descriptor): match.descriptor for match in matches
    }
    if len(descriptor_digests) > 1:
        issues.append(
            _issue(
                "MULTIPLE_COORDINATE_DESCRIPTORS_CONFLICT",
                "critical",
                "Multiple applicable coordinate descriptors disagree; declaration precedence is not coordinate authority.",
                declarations=[
                    {
                        "source": match.source,
                        "descriptor_sha256": _fingerprint(match.descriptor),
                    }
                    for match in matches
                ],
            )
        )
    contaminated = [
        match.source
        for match in matches
        if not match.array_specific and match.attr_name in _DESCRIPTOR_ATTRS
    ]
    if contaminated:
        issues.append(
            _issue(
                "GENERIC_ANCESTOR_DESCRIPTOR_CONTAMINATION",
                "error",
                "A generic ancestor descriptor cannot define every descendant array; use an exact keyed or array-specific descriptor.",
                declarations=contaminated,
            )
        )
    return issues


def _descriptor_value(descriptor: Mapping[str, Any], keys: Sequence[str]) -> Any:
    found = _deep_find(descriptor, keys)
    return found[0] if found else None


def _descriptor_extent(descriptor: Mapping[str, Any]) -> tuple[Any, Any]:
    extent = _as_mapping(descriptor.get("reference_extent"))
    width = _descriptor_value(extent, ("width", "reference_width"))
    height = _descriptor_value(extent, ("height", "reference_height"))
    if width is None:
        width = _descriptor_value(descriptor, _WIDTH_KEYS)
    if height is None:
        height = _descriptor_value(descriptor, _HEIGHT_KEYS)
    return width, height


def _issue(code: str, severity: str, message: str, **evidence: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"code": code, "severity": severity, "message": message}
    if evidence:
        result["evidence"] = _json_safe(evidence)
    return result


def _has_row_identity(
    surface_type: str,
    node: MetadataNode,
    nodes: Mapping[str, MetadataNode],
) -> tuple[bool, str | None]:
    resolved, issues = _legacy_row_identity_resolution(
        surface_type=surface_type,
        node=node,
        nodes=nodes,
    )
    return (
        resolved is not None
        and not any(issue["severity"] in {"error", "critical"} for issue in issues),
        resolved,
    )


def _surface_evidence(
    surface_type: str,
    node: MetadataNode,
    nodes: Mapping[str, MetadataNode],
) -> tuple[dict[str, Any], DescriptorMatch | None]:
    descriptor_match = _find_descriptor(node, nodes)
    descriptor = descriptor_match.descriptor if descriptor_match else None
    descriptor_source = descriptor_match.source if descriptor_match else None

    evidence: dict[str, Any] = {}
    field_specs = {
        "space_id": _SPACE_KEYS,
        "units": _UNITS_KEYS,
        "origin": _ORIGIN_KEYS,
        "x_axis_direction": _X_AXIS_KEYS,
        "y_axis_direction": _Y_AXIS_KEYS,
        "reference_width": _WIDTH_KEYS,
        "reference_height": _HEIGHT_KEYS,
        "reference_authority": _REFERENCE_AUTHORITY_KEYS,
        "pixel_convention": _PIXEL_CONVENTION_KEYS,
        "geometry_convention": _GEOMETRY_CONVENTION_KEYS,
        "source_ref": _SOURCE_REF_KEYS,
        "transform_ref": _TRANSFORM_REF_KEYS,
        "transform_direction": _TRANSFORM_DIRECTION_KEYS,
        "transform_from_space": _TRANSFORM_FROM_KEYS,
        "transform_to_space": _TRANSFORM_TO_KEYS,
        "source_camera_overlay_suitable": _OVERLAY_KEYS,
    }
    for field, keys in field_specs.items():
        declared = _find_declared(node, nodes, _surface_prefixed_keys(node, keys))
        if declared:
            evidence[field] = {"value": _json_safe(declared[0]), "source": declared[1]}

    if descriptor is None:
        row_identity, row_identity_source = _has_row_identity(
            surface_type, node, nodes
        )
        if row_identity:
            evidence["row_identity"] = {
                "value": True,
                "source": row_identity_source,
            }

    if descriptor:
        descriptor_fields = {
            "space_id": ("space_id",),
            "origin": ("origin",),
            "pixel_convention": ("pixel_convention",),
            "geometry_convention": ("geometry_type",),
        }
        for field, keys in descriptor_fields.items():
            value = _descriptor_value(descriptor, keys)
            if value is not None:
                evidence[field] = {"value": _json_safe(value), "source": descriptor_source}
        component_units = descriptor.get("component_units")
        if isinstance(component_units, (list, tuple)) and component_units:
            distinct_units = tuple(dict.fromkeys(str(unit) for unit in component_units))
            evidence["units"] = {
                "value": distinct_units[0] if len(distinct_units) == 1 else list(distinct_units),
                "source": descriptor_source,
            }
        directions = _as_mapping(descriptor.get("positive_directions"))
        if directions.get("x") not in (None, ""):
            evidence["x_axis_direction"] = {
                "value": _json_safe(directions["x"]),
                "source": descriptor_source,
            }
        if directions.get("y") not in (None, ""):
            evidence["y_axis_direction"] = {
                "value": _json_safe(directions["y"]),
                "source": descriptor_source,
            }
        width, height = _descriptor_extent(descriptor)
        reference_extent = _as_mapping(descriptor.get("reference_extent"))
        if width is not None:
            evidence["reference_width"] = {"value": _json_safe(width), "source": descriptor_source}
        if height is not None:
            evidence["reference_height"] = {"value": _json_safe(height), "source": descriptor_source}
        if reference_extent.get("authority") not in (None, ""):
            evidence["reference_authority"] = {
                "value": _json_safe(reference_extent["authority"]),
                "source": descriptor_source,
            }
        row_identity = _as_mapping(descriptor.get("row_identity"))
        if row_identity:
            evidence["row_identity"] = {
                "value": True,
                "source": f"{descriptor_source}:row_identity",
                "descriptor_value": _json_safe(row_identity),
            }
        if descriptor.get("source_camera_overlay") not in (None, ""):
            evidence["source_camera_overlay_suitable"] = {
                "value": _json_safe(descriptor["source_camera_overlay"]),
                "source": f"{descriptor_source}:source_camera_overlay",
            }
        lineage_refs = descriptor.get("lineage_refs")
        if isinstance(lineage_refs, (list, tuple)) and lineage_refs:
            evidence["source_ref"] = {
                "value": _json_safe(lineage_refs),
                "source": f"{descriptor_source}:lineage_refs",
            }
        transform_refs = descriptor.get("transform_refs")
        if not isinstance(transform_refs, (list, tuple)):
            transform_refs = _as_mapping(
                descriptor.get("source_camera_overlay")
            ).get("transform_refs")
        if isinstance(transform_refs, (list, tuple)) and transform_refs:
            evidence["transform_ref"] = {
                "value": _json_safe(transform_refs),
                "source": f"{descriptor_source}:source_camera_overlay.transform_refs",
            }
    return evidence, descriptor_match


def _collect_attr_declarations(
    mapping: Mapping[str, Any],
    keys: Sequence[str],
    *,
    prefix: str = "",
    depth: int = 0,
) -> list[tuple[Any, str]]:
    if depth > 8:
        return []
    wanted = {str(key).lower() for key in keys}
    found: list[tuple[Any, str]] = []
    for raw_key, value in mapping.items():
        key = str(raw_key)
        lowered = key.lower()
        if (
            lowered in _DESCRIPTOR_ATTRS
            or lowered == "coordinate_descriptors"
            or lowered.endswith("_coordinate_descriptor")
            or lowered.endswith("_coordinate_descriptor_sha256")
            or lowered in _REGISTERED_OBSERVATION_RECORDS_BY_ATTR
            or lowered.removesuffix("_sha256")
            in _REGISTERED_OBSERVATION_RECORDS_BY_ATTR
            or lowered
            in {
                DIRECTED_TRANSFORM_V2_ATTR,
                DIRECTED_TRANSFORM_V2_DIGEST_ATTR,
                TRANSFORM_AUTHORITY_ATTR,
                TRANSFORM_AUTHORITY_DIGEST_ATTR,
            }
        ):
            continue
        location = f"{prefix}.{key}" if prefix else key
        if lowered in wanted and value not in (None, ""):
            found.append((value, location))
        nested = _as_mapping(value)
        if nested:
            found.extend(
                _collect_attr_declarations(
                    nested,
                    keys,
                    prefix=location,
                    depth=depth + 1,
                )
            )
    return found


def _normalized_conflict_value(field: str, value: Any) -> Any:
    if field == "space_id" and isinstance(value, str):
        return {
            "camera": "source_camera_image_px",
            "source_image_px": "source_camera_image_px",
            "source_camera": "source_camera_image_px",
            "texture": "stimulus_texture_px",
        }.get(value.strip().lower(), value.strip())
    if field == "origin" and isinstance(value, str):
        return {
            "top_left_of_active_arena": "arena_top_left",
        }.get(value.strip().lower(), value.strip())
    if field == "units":
        if isinstance(value, (list, tuple)):
            distinct = tuple(dict.fromkeys(str(item) for item in value))
            return distinct[0] if len(distinct) == 1 else distinct
        return str(value)
    if field == "source_camera_overlay":
        if value is True:
            return "direct"
        if value is False:
            return "not_suitable"
        if isinstance(value, Mapping):
            return value.get("status")
    if field in {"reference_width", "reference_height"}:
        try:
            return float(value)
        except (TypeError, ValueError):
            return value
    return _json_safe(value)


def _descriptor_conflict_issues(
    *,
    descriptor: Mapping[str, Any],
    node: MetadataNode,
    nodes: Mapping[str, MetadataNode],
) -> list[dict[str, Any]]:
    extent = _as_mapping(descriptor.get("reference_extent"))
    directions = _as_mapping(descriptor.get("positive_directions"))
    component_units = descriptor.get("component_units")
    expected: dict[str, Any] = {
        "space_id": descriptor.get("space_id"),
        "units": component_units,
        "origin": descriptor.get("origin"),
        "x_axis_direction": directions.get("x"),
        "y_axis_direction": directions.get("y"),
        "reference_width": extent.get("width"),
        "reference_height": extent.get("height"),
        "reference_authority": extent.get("authority"),
        "pixel_convention": descriptor.get("pixel_convention"),
        "geometry_convention": descriptor.get("geometry_type"),
        "source_camera_overlay": _overlay_status(descriptor),
    }
    specs = {
        "space_id": _SPACE_KEYS,
        "units": _UNITS_KEYS,
        "origin": _ORIGIN_KEYS,
        "x_axis_direction": _X_AXIS_KEYS,
        "y_axis_direction": _Y_AXIS_KEYS,
        "reference_width": _WIDTH_KEYS,
        "reference_height": _HEIGHT_KEYS,
        "reference_authority": _REFERENCE_AUTHORITY_KEYS,
        "pixel_convention": _PIXEL_CONVENTION_KEYS,
        "geometry_convention": _GEOMETRY_CONVENTION_KEYS,
        "source_camera_overlay": _OVERLAY_KEYS,
    }
    issues: list[dict[str, Any]] = []
    for path in [node.relative_path, *_ancestor_paths(node.relative_path)]:
        owner = nodes.get(path)
        if owner is None:
            continue
        for field, keys in specs.items():
            expected_value = expected[field]
            if expected_value is None:
                continue
            declarations = _collect_attr_declarations(
                owner.attributes,
                _surface_prefixed_keys(node, keys),
            )
            for declared, attr_path in declarations:
                if _normalized_conflict_value(
                    field, declared
                ) == _normalized_conflict_value(field, expected_value):
                    continue
                issues.append(
                    _issue(
                        "DESCRIPTOR_DECLARATION_CONFLICT",
                        "error",
                        "Canonical array descriptor conflicts with direct attrs or ancestor/run provenance.",
                        field=field,
                        descriptor_value=expected_value,
                        declared_value=declared,
                        declaration_source=f"{path}:{attr_path}",
                    )
                )

        descriptor_transform_refs = descriptor.get("transform_refs")
        if not isinstance(descriptor_transform_refs, (list, tuple)):
            descriptor_transform_refs = _as_mapping(
                descriptor.get("source_camera_overlay")
            ).get("transform_refs")
        expected_ref_sets = {
            "lineage_refs": {
                str(item.get("ref") or item.get("record_ref"))
                for item in (descriptor.get("lineage_refs") or ())
                if isinstance(item, Mapping)
                and isinstance(item.get("ref") or item.get("record_ref"), str)
            },
            "transform_refs": {
                str(item.get("ref") or item.get("record_ref"))
                for item in (descriptor_transform_refs or ())
                if isinstance(item, Mapping)
                and isinstance(item.get("ref") or item.get("record_ref"), str)
            },
        }
        for field, keys in (
            ("lineage_refs", _SOURCE_REF_KEYS),
            ("transform_refs", _TRANSFORM_REF_KEYS),
        ):
            expected_refs = expected_ref_sets[field]
            for declared, attr_path in _collect_attr_declarations(
                owner.attributes,
                _surface_prefixed_keys(node, keys),
            ):
                values = declared if isinstance(declared, (list, tuple)) else [declared]
                declared_refs = {
                    str(value.get("ref") or value.get("record_ref"))
                    if isinstance(value, Mapping)
                    and isinstance(value.get("ref") or value.get("record_ref"), str)
                    else str(value)
                    for value in values
                    if value not in (None, "")
                }
                if declared_refs and declared_refs <= expected_refs:
                    continue
                issues.append(
                    _issue(
                        "DESCRIPTOR_LINEAGE_CONFLICT",
                        "error",
                        "Canonical descriptor lineage conflicts with direct attrs or ancestor/run provenance.",
                        field=field,
                        descriptor_refs=sorted(expected_refs),
                        declared_refs=sorted(declared_refs),
                        declaration_source=f"{path}:{attr_path}",
                    )
                )

        descriptor_row = _as_mapping(descriptor.get("row_identity"))
        descriptor_row_refs = {
            ref for _name, ref in _row_identity_refs(descriptor_row)
        }
        if isinstance(descriptor_row.get("record_ref"), str):
            descriptor_row_refs.add(str(descriptor_row["record_ref"]))
        for declared, attr_path in _collect_attr_declarations(
            owner.attributes,
            _surface_prefixed_keys(node, _ROW_IDENTITY_KEYS),
        ):
            declared_mapping = _as_mapping(declared)
            if declared_mapping:
                declared_mode = declared_mapping.get("mode")
                declared_refs = {
                    ref for _name, ref in _row_identity_refs(declared_mapping)
                }
                if isinstance(declared_mapping.get("record_ref"), str):
                    declared_refs.add(str(declared_mapping["record_ref"]))
            else:
                declared_mode = None
                declared_refs = {str(declared)} if isinstance(declared, str) else set()
            if (
                (declared_mode in (None, descriptor_row.get("mode")))
                and declared_refs
                and declared_refs <= descriptor_row_refs
            ):
                continue
            issues.append(
                _issue(
                    "DESCRIPTOR_ROW_IDENTITY_CONFLICT",
                    "error",
                    "Canonical descriptor row identity conflicts with direct attrs or ancestor/run provenance.",
                    descriptor_row_identity=descriptor_row,
                    declared_value=declared,
                    declaration_source=f"{path}:{attr_path}",
                )
            )
    return issues


def _value(evidence: Mapping[str, Any], field: str) -> Any:
    item = evidence.get(field)
    return item.get("value") if isinstance(item, Mapping) else None


def _is_direct_source(node: MetadataNode, evidence: Mapping[str, Any], field: str) -> bool:
    item = evidence.get(field)
    if not isinstance(item, Mapping):
        return False
    return str(item.get("source") or "").startswith(f"{node.relative_path}:")


_COORDINATE_VALUE_VALIDATION_ATTR = "coordinate_value_validation"
_COORDINATE_VALUE_VALIDATION_DIGEST_ATTR = (
    "coordinate_value_validation_sha256"
)
_COORDINATE_VALUE_VALIDATION_REF_ATTR = "coordinate_value_validation_ref"
_COORDINATE_VALUE_VALIDATION_SCHEMA_ID = "palette.coordinate_value_validation"
_COORDINATE_VALUE_VALIDATION_FIELDS = frozenset(
    {
        "schema_id",
        "schema_version",
        "validation_kind",
        "producer_signature",
        "surface_ref",
        "values_sha256",
        "checks",
        "result",
        "validator_commit",
        "canonicalization",
    }
)
_COORDINATE_VALUE_VALIDATION_CHECKS = {
    "online_mm_conversion_values_v1": {
        "camera_scale_reciprocal_verified": True,
        "source_pixel_values_recomputed": True,
        "output_values_match_recomputation": True,
        "row_identity_verified": True,
    },
    "offline_crop_camera_values_v1": {
        "crop_placement_direction_verified": True,
        "source_reference_extent_verified": True,
        "output_values_match_source_overlay": True,
        "row_identity_verified": True,
    },
}


def _independent_value_validation_issues(
    *,
    node: MetadataNode,
    nodes: Mapping[str, MetadataNode],
    validation_kind: str,
    producer_signature: str,
) -> tuple[bool, list[dict[str, Any]]]:
    pointer = _as_mapping(
        node.attributes.get(_COORDINATE_VALUE_VALIDATION_REF_ATTR)
    )
    record_ref = pointer.get("record_ref")
    record_sha256 = pointer.get("record_sha256")
    target_path, attr_name = _canonical_v2_record_target(record_ref)
    target = nodes.get(target_path or "")
    if (
        set(pointer) != {"record_ref", "record_sha256"}
        or target is None
        or target.relative_path == node.relative_path
        or attr_name != _COORDINATE_VALUE_VALIDATION_ATTR
    ):
        return False, []
    raw = target.attributes.get(_COORDINATE_VALUE_VALIDATION_ATTR)
    record = _as_mapping(raw)
    expected_checks = _COORDINATE_VALUE_VALIDATION_CHECKS[validation_kind]
    valid = (
        set(record) == _COORDINATE_VALUE_VALIDATION_FIELDS
        and record.get("schema_id") == _COORDINATE_VALUE_VALIDATION_SCHEMA_ID
        and record.get("schema_version") == 1
        and record.get("validation_kind") == validation_kind
        and record.get("producer_signature") == producer_signature
        and record.get("surface_ref") == f"/{node.relative_path}@array_values"
        and isinstance(record.get("values_sha256"), str)
        and _SHA256_HEX_RE.fullmatch(str(record.get("values_sha256"))) is not None
        and record.get("checks") == expected_checks
        and record.get("result") == "pass"
        and isinstance(record.get("validator_commit"), str)
        and bool(str(record.get("validator_commit")).strip())
        and record.get("canonicalization") == "canonical_json_sort_keys_v1"
        and record_sha256 == _fingerprint(record)
        and target.attributes.get(_COORDINATE_VALUE_VALIDATION_DIGEST_ATTR)
        == _fingerprint(record)
    )
    if not valid:
        return False, [
            _issue(
                "COORDINATE_VALUE_VALIDATION_RECORD_INVALID",
                "error",
                "A correction exception requires one independent, exact, digest-bound value-validation record.",
                validation_kind=validation_kind,
                record_ref=record_ref,
            )
        ]
    return True, [
        _issue(
            "COORDINATE_VALUE_VALIDATION_PAYLOAD_CHECK_REQUIRED",
            "warning",
            "Independent validation metadata is sealed, but this metadata-only audit does not rehash the coordinate payload.",
            validation_kind=validation_kind,
            record_ref=record_ref,
            expected_values_sha256=record.get("values_sha256"),
        )
    ]


def _sealed_online_mm_correction(
    *,
    node: MetadataNode,
    nodes: Mapping[str, MetadataNode],
    descriptor: Mapping[str, Any] | None,
) -> tuple[bool, list[dict[str, Any]]]:
    if not descriptor or descriptor.get("schema_version") != 2:
        return False, []
    frame = _as_mapping(descriptor.get("frame_record"))
    if frame.get("kind") != PHYSICAL_FRAME_CALIBRATION_KIND:
        return False, []
    authority_issues = _canonical_physical_frame_record_issues(
        descriptor,
        nodes=nodes,
    )
    if any(
        issue["severity"] in {"error", "critical"}
        for issue in authority_issues
    ):
        return False, authority_issues
    validation_ok, validation_issues = _independent_value_validation_issues(
        node=node,
        nodes=nodes,
        validation_kind="online_mm_conversion_values_v1",
        producer_signature="track_kinematics_online_refined.corrected.v1",
    )
    return validation_ok, [*authority_issues, *validation_issues]


def _sealed_offline_crop_correction(
    *,
    node: MetadataNode,
    nodes: Mapping[str, MetadataNode],
    descriptor: Mapping[str, Any] | None,
) -> tuple[bool, list[dict[str, Any]]]:
    if not descriptor or descriptor.get("schema_version") != 2:
        return False, []
    transform_proved = False
    transform_issues: list[dict[str, Any]] = []
    for raw in descriptor.get("lineage_refs") or ():
        pointer = _as_mapping(raw)
        target_path, attr_name = _canonical_v2_record_target(
            pointer.get("record_ref")
        )
        target = nodes.get(target_path or "")
        if target is None or attr_name != DIRECTED_TRANSFORM_V2_ATTR:
            continue
        transform, parsed_issues = _parse_directed_transform_v2_node(
            target,
            record_ref=str(pointer.get("record_ref") or ""),
            nodes=nodes,
        )
        transform_issues.extend(parsed_issues)
        if (
            transform is not None
            and not any(
                issue["severity"] in {"error", "critical"}
                for issue in parsed_issues
            )
            and transform.kind == AFFINE_2D_ROWWISE_KIND
            and transform.source.space_id == "roi_local_px"
            and transform.target.space_id == SOURCE_CAMERA_IMAGE_SPACE_ID
            and transform.transform_authority.kind
            == CROP_PLACEMENT_AUTHORITY_KIND
        ):
            transform_proved = True
    if not transform_proved:
        return False, transform_issues
    validation_ok, validation_issues = _independent_value_validation_issues(
        node=node,
        nodes=nodes,
        validation_kind="offline_crop_camera_values_v1",
        producer_signature="track_kinematics_offline.corrected.v1",
    )
    return validation_ok, [*transform_issues, *validation_issues]


def _legacy_online_mm_requires_recompute(
    surface_type: str,
    node: MetadataNode,
    nodes: Mapping[str, MetadataNode],
    descriptor: Mapping[str, Any] | None,
) -> tuple[bool, dict[str, Any], list[dict[str, Any]]]:
    if surface_type not in {"track_positions_mm", "refined_online_positions_mm"}:
        return False, {}, []
    method = _find_producer_declared(node, nodes, ("method",))
    space = _find_producer_declared(node, nodes, _SPACE_KEYS)
    if (
        method is None
        or str(method[0]).strip().lower() != "track_kinematics_online_refined"
        or space is None
        or str(space[0]).strip().lower() != "texture"
    ):
        return False, {}, []
    pixel_to_mm = _find_producer_declared(
        node, nodes, ("pixel_to_mm", "calibration_used")
    )
    ppm = _find_producer_declared(
        node, nodes, ("pixels_per_mm_projector", "pixels_per_mm")
    )
    same_declared_value = False
    if pixel_to_mm and ppm:
        try:
            same_declared_value = float(pixel_to_mm[0]) == float(ppm[0])
        except (TypeError, ValueError):
            same_declared_value = False
    mm_per_pixel = _find_producer_declared(node, nodes, ("mm_per_pixel",))
    if not (same_declared_value or mm_per_pixel is None):
        return False, {}, []
    corrected, correction_issues = _sealed_online_mm_correction(
        node=node,
        nodes=nodes,
        descriptor=descriptor,
    )
    evidence = {
        "method_source": method[1],
        "coordinate_space_source": space[1],
        "pixel_to_mm_source": pixel_to_mm[1] if pixel_to_mm else None,
        "pixels_per_mm_source": ppm[1] if ppm else None,
        "same_pixels_per_mm_value": same_declared_value,
        "mm_per_pixel_missing": mm_per_pixel is None,
        "sealed_correction_exception": corrected,
    }
    return not corrected, evidence, correction_issues


def _offline_crop_reconstruction_requires_recompute(
    surface_type: str,
    node: MetadataNode,
    nodes: Mapping[str, MetadataNode],
    descriptor: Mapping[str, Any] | None,
) -> tuple[bool, dict[str, Any], list[dict[str, Any]]]:
    if surface_type != "track_positions_px":
        return False, {}, []
    method = _find_producer_declared(node, nodes, ("method",))
    position_kind = _find_producer_declared(
        node, nodes, ("position_source_kind",)
    )
    position_path = _find_producer_declared(
        node, nodes, ("position_source_path",)
    )
    geometry_path = _find_producer_declared(
        node, nodes, ("position_geometry_path",)
    )
    space = _find_producer_declared(node, nodes, _SPACE_KEYS)
    if (
        method is None
        or str(method[0]).strip().lower() != "track_kinematics_offline"
        or not position_kind
        or str(position_kind[0]).strip() != "crop_rows"
        or not position_path
        or "crop_runs" not in PurePosixPath(
            str(position_path[0]).strip().strip("/")
        ).parts
    ):
        return False, {}, []
    if not space or str(space[0]).lower() not in {
        "camera",
        "source_camera",
        "source_camera_px",
        "source_camera_image_px",
    }:
        return False, {}, []
    geometry_text = (
        str(geometry_path[0]).strip().lower() if geometry_path else ""
    )
    if geometry_text.endswith("bbox_img_xyxy"):
        return False, {}, []
    corrected, correction_issues = _sealed_offline_crop_correction(
        node=node,
        nodes=nodes,
        descriptor=descriptor,
    )
    return not corrected, {
        "method_source": method[1],
        "position_source_kind_source": position_kind[1],
        "position_source_path": position_path[0],
        "position_source_path_source": position_path[1],
        "position_geometry_path": geometry_path[0] if geometry_path else None,
        "declared_space_source": space[1],
        "sealed_correction_exception": corrected,
    }, correction_issues


def _normalize_archive_ref(
    ref: str,
    *,
    owner_path: str | None = None,
    owner_is_array: bool = False,
) -> str | None:
    text = str(ref).strip()
    if not text:
        return None
    if text.startswith("zarr:/"):
        text = text.removeprefix("zarr:")
    elif re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", text):
        # External HDF5/file/URI references are legitimate provenance but are
        # not resolvable as nodes in the archive currently being inventoried.
        return None
    if text.startswith("/"):
        candidate = text.strip("/")
    elif owner_path is None:
        candidate = text.strip("/")
    else:
        base = (
            PurePosixPath(owner_path).parent.as_posix()
            if owner_is_array
            else PurePosixPath(owner_path).as_posix()
        )
        candidate = posixpath.join("" if base == "." else base, text)
    normalized = posixpath.normpath(candidate).strip("/")
    if normalized in {"", ".", ".."} or normalized.startswith("../"):
        return None
    return normalized


def _record_ref_target(ref: str) -> tuple[str, str | None, str | None]:
    """Return archive node path plus optional selector kind/value."""

    text = str(ref).strip()
    if ".attrs[" in text and text.endswith("]"):
        node_ref, selector = text.rsplit(".attrs[", 1)
        return node_ref, "attr", selector[:-1]
    if "#" in text:
        node_ref, selector = text.split("#", 1)
        return node_ref, "fragment", selector or None
    return text, None, None


_SHAPE_AUTHORITY_RE = re.compile(r"^(?P<node>.+)\.shape\[-2:\]$")
_ATTR_FIELD_AUTHORITY_RE = re.compile(
    r"^(?P<node>.+)\.attrs\[(?P<attr>[^\[\],]+)\]\.(?P<field>[A-Za-z0-9_]+)$"
)
_ATTR_SELECTOR_AUTHORITY_RE = re.compile(
    r"^(?P<node>.+)\.attrs\[(?P<selectors>[^\[\]]+)\]$"
)
_ATTRS_AUTHORITY_RE = re.compile(r"^(?P<node>.+)\.attrs$")
_AT_AUTHORITY_RE = re.compile(r"^(?P<node>.+)@(?P<selectors>[^@]+)$")
_OUTPUT_GEOMETRY_RE = re.compile(
    r"^(?P<width>[1-9][0-9]*)x(?P<height>[1-9][0-9]*)"
    r"(?P<x>[+-](?:0|[1-9][0-9]*))(?P<y>[+-](?:0|[1-9][0-9]*))$"
)
_PHYSICAL_FRAME_RECORD_FIELDS = {
    "schema_id",
    "schema_version",
    "physical_frame_id",
    "units",
    "origin",
    "positive_directions",
    "source_space_id",
    "source_reference_authority",
    "calibration_ref",
    "calibration_sha256",
    "pixels_per_mm",
    "mm_per_pixel",
    "reciprocal_derivation",
}
_REFERENCE_EXTENT_RECORD_FIELDS = {
    "schema_id",
    "schema_version",
    "space_id",
    "width",
    "height",
    "units",
}
_CAMERA_EXTENT_SELECTORS = ("native_width_px", "native_height_px")
_ARENA_EXTENT_SELECTORS = (
    "arena_region_width_px",
    "arena_region_height_px",
)
_DISPLAY_AUTHORITY_REQUIRED_ATTRS = {
    "selected_output_name",
    "selected_output_geometry",
    "selected_output_transform_token",
    "source_display_dataset_path",
    "source_display_dataset_sha256",
    "source_display_dataset_digest_canonicalization",
}
_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ResolvedReferenceAuthority:
    """One exact, controlled metadata record that proves a reference extent."""

    width: Any
    height: Any
    units: str
    authority_kind: str
    node_path: str
    camera_id: str | None = None
    physical_record: Mapping[str, Any] | None = None


def _load_selected_calibration_metadata(
    *,
    camera_path: str,
    nodes: Mapping[str, MetadataNode],
) -> tuple[Any, Any]:
    """Validate one future selected-calibration snapshot without matrix reads."""

    camera_node = nodes.get(camera_path)
    if camera_node is None:
        raise SelectedCalibrationError("Selected camera-calibration node is missing.")
    camera_parts = PurePosixPath(camera_path).parts
    if (
        len(camera_parts) < 5
        or camera_parts[0:2] != ("analysis", "stimulus_runs")
        or camera_parts[-2] != "calibration"
    ):
        raise SelectedCalibrationError(
            "Selected camera-calibration ref is not in a concrete stimulus-run calibration path."
        )
    stimulus_run = camera_parts[2]
    camera_id = camera_parts[-1]
    calibration_path = PurePosixPath(camera_path).parent.as_posix()
    run_path = PurePosixPath(calibration_path).parent.as_posix()
    display_path = f"{run_path}/display_snapshot"
    calibration_node = nodes.get(calibration_path)
    display_node = nodes.get(display_path)
    if calibration_node is None or display_node is None:
        raise SelectedCalibrationError(
            "Selected calibration or display snapshot node is missing."
        )
    calibration_attrs = calibration_node.attributes
    if (
        calibration_attrs.get("schema_id") != SELECTED_CALIBRATION_SCHEMA_ID
        or calibration_attrs.get("schema_version")
        != SELECTED_CALIBRATION_SCHEMA_VERSION
        or calibration_attrs.get("stimulus_run") != stimulus_run
        or calibration_attrs.get("active_camera_id") != camera_id
        or _normalize_archive_ref(
            str(calibration_attrs.get("active_camera_calibration_ref") or "")
        )
        != camera_path
    ):
        raise SelectedCalibrationError(
            "Selected-calibration pointer attrs are incomplete or inconsistent."
        )
    manifest = load_selected_calibration_manifest_attrs(calibration_attrs)
    if (
        manifest.stimulus_run != stimulus_run
        or manifest.camera_id != camera_id
        or _normalize_archive_ref(manifest.camera_calibration_ref) != camera_path
    ):
        raise SelectedCalibrationError(
            "Selected-calibration manifest does not bind the exact run and camera."
        )
    transform_path = _normalize_archive_ref(manifest.transform_ref)
    if transform_path is None:
        raise SelectedCalibrationError("Manifest transform_ref is not archive-relative.")
    if _normalize_archive_ref(
        str(calibration_attrs.get("active_camera_transform_ref") or "")
    ) != transform_path or calibration_attrs.get(
        "active_camera_transform_sha256"
    ) != manifest.transform_sha256:
        raise SelectedCalibrationError(
            "Selected-calibration transform pointer disagrees with its manifest."
        )

    expected_camera = manifest.camera_calibration.to_dict()
    camera_attrs = camera_node.attributes
    if (
        camera_attrs.get("schema_id") != CAMERA_CALIBRATION_SCHEMA_ID
        or camera_attrs.get("schema_version") != CAMERA_CALIBRATION_SCHEMA_VERSION
        or camera_attrs.get("camera_id") != camera_id
        or any(camera_attrs.get(name) != value for name, value in expected_camera.items())
    ):
        raise SelectedCalibrationError(
            "Persisted camera calibration attrs do not match the selected manifest."
        )

    display_evidence = load_selected_display_evidence_attrs(display_node.attributes)
    display_manifest = manifest.display_snapshot
    if (
        display_evidence != manifest.source_display
        or display_node.attributes.get("selected_output_name")
        != display_manifest.selected_output_name
        or display_node.attributes.get("selected_output_geometry")
        != display_manifest.selected_output_geometry
        or display_node.attributes.get("selected_output_transform_token")
        != display_manifest.selected_output_transform_token
        or display_node.attributes.get("source_display_dataset_path")
        != display_manifest.source_h5_dataset_path
        or display_node.attributes.get("source_display_dataset_sha256")
        != display_manifest.source_h5_dataset_sha256
        or display_node.attributes.get(
            "source_display_dataset_digest_canonicalization"
        )
        != display_manifest.source_h5_dataset_digest_canonicalization
    ):
        raise SelectedCalibrationError(
            "Persisted display evidence does not match the selected manifest."
        )
    if _normalize_archive_ref(manifest.display_snapshot.ref) != display_path:
        raise SelectedCalibrationError(
            "Selected display snapshot path does not match the manifest."
        )

    transform_node = nodes.get(transform_path)
    if (
        transform_node is None
        or transform_node.node_type != "array"
        or list(transform_node.shape or []) != [3, 3]
    ):
        raise SelectedCalibrationError(
            "Selected homography must resolve to one exact 3x3 array."
        )
    transform = load_directed_homography_attrs(transform_node.attributes)
    homography_evidence = load_selected_homography_evidence_attrs(
        transform_node.attributes
    )
    if (
        homography_evidence != manifest.source_homography
        or transform.digest() != manifest.transform_sha256
        or transform.matrix_sha256 != manifest.matrix_sha256
        or _normalize_archive_ref(transform.calibration_ref) != camera_path
        or transform.camera_id != camera_id
    ):
        raise SelectedCalibrationError(
            "Selected homography metadata/evidence does not match the manifest."
        )
    return manifest, transform


def _metadata_node_record(node: MetadataNode) -> dict[str, Any]:
    """Return the exact metadata-only record bound by a node reference digest."""

    return {
        "relative_path": node.relative_path,
        "node_type": node.node_type,
        "metadata_format": node.metadata_format,
        "shape": _json_safe(node.shape),
        "data_type": _json_safe(node.data_type),
        "chunk_shape": _json_safe(node.chunk_shape),
        "storage_metadata": _json_safe(node.storage_metadata),
        "attributes": _json_safe(node.attributes),
        "metadata_error": node.metadata_error,
    }


def _metadata_node_record_digest(node: MetadataNode) -> str:
    return _fingerprint(_metadata_node_record(node))


def _extent_values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return False
    try:
        return float(left) == float(right)
    except (TypeError, ValueError):
        return False


def _positive_extent_value(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return float(value) > 0 and float(value) != float("inf")
    except (TypeError, ValueError, OverflowError):
        return False


def _finite_extent_value(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return numeric == numeric and numeric not in {float("inf"), float("-inf")}


def _authority_issue(
    code: str,
    message: str,
    *,
    authority: str,
    **evidence: Any,
) -> list[dict[str, Any]]:
    return [_issue(code, "error", message, authority=authority, **evidence)]


def _physical_frame_record_issues(
    record: Mapping[str, Any],
    *,
    authority: str,
    physical_frame: Any,
    component_units: Sequence[Any],
    origin: Any,
    positive_directions: Mapping[str, Any] | None,
    reference_units: Any,
    nodes: Mapping[str, MetadataNode],
) -> list[dict[str, Any]]:
    if set(record) != _PHYSICAL_FRAME_RECORD_FIELDS:
        return [
            _issue(
                "PHYSICAL_FRAME_RECORD_INVALID",
                "error",
                "Physical authority must resolve to the exact controlled physical-frame record schema.",
                authority=authority,
                fields=sorted(str(key) for key in record),
            )
        ]
    expected_directions = dict(positive_directions or {})
    mismatches: dict[str, Any] = {}
    if record.get("schema_id") != "palette.physical_coordinate_frame":
        mismatches["schema_id"] = record.get("schema_id")
    if record.get("schema_version") != 1:
        mismatches["schema_version"] = record.get("schema_version")
    if record.get("physical_frame_id") != physical_frame:
        mismatches["physical_frame_id"] = {
            "declared": physical_frame,
            "authority": record.get("physical_frame_id"),
        }
    if record.get("units") != "mm" or any(str(unit) != "mm" for unit in component_units):
        mismatches["units"] = {
            "component_units": list(component_units),
            "authority": record.get("units"),
        }
    if reference_units != "not_applicable":
        mismatches["reference_units"] = reference_units
    if record.get("origin") != origin:
        mismatches["origin"] = {
            "declared": origin,
            "authority": record.get("origin"),
        }
    if _as_mapping(record.get("positive_directions")) != expected_directions:
        mismatches["positive_directions"] = {
            "declared": expected_directions,
            "authority": record.get("positive_directions"),
        }
    source_space_id = record.get("source_space_id")
    if source_space_id not in {
        "source_camera_image_px",
        "stimulus_texture_px",
        "stimulus_canvas_px",
        "arena_relative_canvas_px",
    }:
        mismatches["source_space_id"] = source_space_id
    if not isinstance(record.get("source_reference_authority"), str):
        mismatches["source_reference_authority"] = record.get(
            "source_reference_authority"
        )
    pixels_per_mm = record.get("pixels_per_mm")
    mm_per_pixel = record.get("mm_per_pixel")
    if (
        not _positive_extent_value(pixels_per_mm)
        or not _positive_extent_value(mm_per_pixel)
        or not _extent_values_equal(float(pixels_per_mm) * float(mm_per_pixel), 1.0)
        or record.get("reciprocal_derivation")
        != "mm_per_pixel_reciprocal_of_pixels_per_mm_v1"
    ):
        mismatches["reciprocal_calibration"] = {
            "pixels_per_mm": pixels_per_mm,
            "mm_per_pixel": mm_per_pixel,
            "reciprocal_derivation": record.get("reciprocal_derivation"),
        }
    calibration_ref = record.get("calibration_ref")
    calibration_path = (
        _normalize_archive_ref(calibration_ref)
        if isinstance(calibration_ref, str)
        else None
    )
    calibration_node = nodes.get(calibration_path or "")
    calibration_digest = record.get("calibration_sha256")
    if (
        calibration_node is None
        or not isinstance(calibration_digest, str)
        or calibration_digest != _metadata_node_record_digest(calibration_node)
    ):
        mismatches["calibration_binding"] = {
            "calibration_ref": calibration_ref,
            "declared_sha256": calibration_digest,
            "actual_sha256": (
                _metadata_node_record_digest(calibration_node)
                if calibration_node is not None
                else None
            ),
        }
    if mismatches:
        return [
            _issue(
                "PHYSICAL_FRAME_RECORD_MISMATCH",
                "error",
                "Physical descriptor disagrees with its exact controlled physical-frame authority.",
                authority=authority,
                mismatches=mismatches,
            )
        ]
    return []


def _canonical_physical_frame_record_issues(
    descriptor: Mapping[str, Any],
    *,
    nodes: Mapping[str, MetadataNode],
) -> list[dict[str, Any]]:
    """Validate the sealed source-camera-to-mm record used by schema v2."""

    frame_pointer = _as_mapping(descriptor.get("frame_record"))
    record_ref = frame_pointer.get("record_ref")
    node_path, attr_name = _canonical_v2_record_target(record_ref)
    target = nodes.get(node_path or "")
    if target is None or attr_name != PHYSICAL_FRAME_CALIBRATION_ATTR:
        return [
            _issue(
                "PHYSICAL_FRAME_AUTHORITY_REQUIRED",
                "error",
                "Canonical physical coordinates require an exact @physical_frame_calibration record.",
                record_ref=record_ref,
            )
        ]
    raw = target.attributes.get(PHYSICAL_FRAME_CALIBRATION_ATTR)
    try:
        record = parse_physical_frame_calibration_record(raw)
    except CoordinateFrameRecordError as exc:
        return [
            _issue(
                "PHYSICAL_FRAME_RECORD_INVALID",
                "error",
                "Physical-frame authority fails the shared strict parser.",
                record_ref=record_ref,
                error=str(exc),
            )
        ]
    issues: list[dict[str, Any]] = []
    digest_attr = f"{PHYSICAL_FRAME_CALIBRATION_ATTR}{FRAME_RECORD_DIGEST_SUFFIX}"
    if (
        not _exact_json_equal(raw, record.to_dict())
        or target.attributes.get(digest_attr) != record.digest()
        or frame_pointer.get("record_sha256") != record.digest()
    ):
        issues.append(
            _issue(
                "PHYSICAL_FRAME_RECORD_DIGEST_MISMATCH",
                "error",
                "Physical-frame pointer and stored digest must bind the exact canonical record.",
                record_ref=record_ref,
            )
        )

    profile_id = descriptor.get("profile_id")
    if profile_id == PHYSICAL_SOURCE_CAMERA_PROFILE_ID:
        if profile_id not in record.compatible_profile_ids:
            issues.append(
                _issue(
                    "PHYSICAL_FRAME_PROFILE_MISMATCH",
                    "error",
                    "Physical descriptor profile is not authorized by its exact physical-frame record.",
                    profile_id=profile_id,
                    compatible_profile_ids=record.compatible_profile_ids,
                )
            )
    elif profile_id == "physical_mm.arena_y_down.v1":
        issues.append(
            _issue(
                "PHYSICAL_ARENA_TRANSFORM_AUTHORITY_REQUIRED",
                "error",
                "The arena-y-down profile is not equivalent to source-camera-preserving millimetres; no controlled direction-labelled physical arena transform is bound.",
                profile_id=profile_id,
                source_profile_id=PHYSICAL_SOURCE_CAMERA_PROFILE_ID,
            )
        )
    else:
        issues.append(
            _issue(
                "PHYSICAL_FRAME_PROFILE_UNSUPPORTED",
                "error",
                "This physical profile requires an explicit controlled direction-labelled transform not represented by the source-camera calibration record.",
                profile_id=profile_id,
                compatible_profile_ids=record.compatible_profile_ids,
            )
        )

    directions = _as_mapping(descriptor.get("positive_directions"))
    extent = _as_mapping(descriptor.get("reference_extent"))
    physical_extent = record.physical_extent
    if (
        tuple(descriptor.get("component_units") or ()) != ("mm", "mm")
        or descriptor.get("origin") != record.origin
        or directions != {"x": record.positive_x, "y": record.positive_y}
        or extent.get("width") != physical_extent.width
        or extent.get("height") != physical_extent.height
        or extent.get("units") != physical_extent.units
    ):
        issues.append(
            _issue(
                "PHYSICAL_FRAME_RECORD_MISMATCH",
                "error",
                "Physical descriptor units, axes, origin, or extent disagree with its exact frame record.",
                profile_id=profile_id,
            )
        )

    source_pointer = record.source_camera_pixels.to_dict()
    source_frame, _source_node, source_issues = _pixel_frame_record_metadata(
        record_ref=source_pointer.get("record_ref"),
        record_sha256=source_pointer.get("record_sha256"),
        role="physical_frame.source_camera_pixels",
        nodes=nodes,
    )
    issues.extend(source_issues)
    if (
        source_frame is None
        or source_frame.kind != SOURCE_CAMERA_FRAME_KIND
        or source_frame.lineage.get("camera_id") != record.camera_id
    ):
        issues.append(
            _issue(
                "PHYSICAL_FRAME_SOURCE_CAMERA_MISMATCH",
                "error",
                "Physical calibration must bind the exact selected source-camera pixel frame.",
                record_ref=source_pointer.get("record_ref"),
            )
        )

    selected_pointer = record.selected_camera_evidence.to_dict()
    selected_path, selected_attr = _canonical_v2_record_target(
        selected_pointer.get("record_ref")
    )
    selected_node = nodes.get(selected_path or "")
    if (
        selected_node is None
        or selected_attr != SELECTED_CAMERA_FRAME_EVIDENCE_ATTR
    ):
        issues.append(
            _issue(
                "PHYSICAL_FRAME_SELECTED_CAMERA_UNRESOLVED",
                "error",
                "Physical calibration selected-camera evidence does not resolve.",
                record_ref=selected_pointer.get("record_ref"),
            )
        )
    else:
        raw_selected = selected_node.attributes.get(
            SELECTED_CAMERA_FRAME_EVIDENCE_ATTR
        )
        try:
            selected = parse_selected_camera_frame_evidence_record(raw_selected)
        except CoordinateFrameRecordError as exc:
            issues.append(
                _issue(
                    "PHYSICAL_FRAME_SELECTED_CAMERA_INVALID",
                    "error",
                    "Selected-camera frame evidence fails the shared strict parser.",
                    error=str(exc),
                )
            )
        else:
            selected_digest_attr = (
                f"{SELECTED_CAMERA_FRAME_EVIDENCE_ATTR}{FRAME_RECORD_DIGEST_SUFFIX}"
            )
            if (
                not _exact_json_equal(raw_selected, selected.to_dict())
                or selected_node.attributes.get(selected_digest_attr)
                != selected.digest()
                or selected_pointer.get("record_sha256") != selected.digest()
                or selected.camera_id != record.camera_id
                or selected.pixels_per_mm_camera != record.pixels_per_mm_camera
            ):
                issues.append(
                    _issue(
                        "PHYSICAL_FRAME_SELECTED_CAMERA_MISMATCH",
                        "error",
                        "Physical calibration does not bind the exact selected-camera evidence and scale.",
                    )
                )
    return issues


def _resolve_reference_authority(
    *,
    authority: Any,
    space_id: Any,
    nodes: Mapping[str, MetadataNode],
) -> tuple[ResolvedReferenceAuthority | None, list[dict[str, Any]]]:
    """Resolve only explicitly controlled authority syntaxes and records."""

    if not isinstance(authority, str) or not authority.strip():
        return None, [
            _issue(
                "REFERENCE_AUTHORITY_MISSING",
                "error",
                "Reference extent authority is missing.",
            )
        ]
    text = authority.strip()

    match = _SHAPE_AUTHORITY_RE.fullmatch(text)
    if match:
        normalized = _normalize_archive_ref(match.group("node"))
        target = nodes.get(normalized or "")
        if target is None or target.node_type != "array":
            return None, _authority_issue(
                "REFERENCE_AUTHORITY_UNRESOLVED",
                "Shape authority does not resolve to a persisted array.",
                authority=text,
                resolved_ref=normalized,
            )
        if (
            not isinstance(target.shape, (list, tuple))
            or len(target.shape) < 2
            or not _positive_extent_value(target.shape[-1])
            or not _positive_extent_value(target.shape[-2])
        ):
            return None, _authority_issue(
                "REFERENCE_AUTHORITY_TARGET_INVALID",
                "Shape authority target has no positive two-dimensional trailing extent.",
                authority=text,
                target_shape=target.shape,
            )
        return (
            ResolvedReferenceAuthority(
                width=target.shape[-1],
                height=target.shape[-2],
                units="px",
                authority_kind="array_shape_trailing_yx",
                node_path=str(normalized),
            ),
            [],
        )

    match = _AT_AUTHORITY_RE.fullmatch(text)
    if match:
        normalized = _normalize_archive_ref(match.group("node"))
        target = nodes.get(normalized or "")
        selectors = tuple(item.strip() for item in match.group("selectors").split(","))
        if target is None:
            return None, _authority_issue(
                "REFERENCE_AUTHORITY_UNRESOLVED",
                "Selected-calibration authority does not resolve to persisted metadata.",
                authority=text,
                resolved_ref=normalized,
            )
        if selectors == _CAMERA_EXTENT_SELECTORS:
            attrs = target.attributes
            camera_id = attrs.get("camera_id")
            try:
                manifest, _transform = _load_selected_calibration_metadata(
                    camera_path=str(normalized),
                    nodes=nodes,
                )
            except SelectedCalibrationError as exc:
                return None, _authority_issue(
                    "REFERENCE_AUTHORITY_TARGET_INVALID",
                    "Camera extent authority must resolve through the strict selected-calibration manifest and source-evidence chain.",
                    authority=text,
                    resolved_ref=normalized,
                    error=str(exc),
                )
            return (
                ResolvedReferenceAuthority(
                    width=manifest.camera_calibration.native_width_px,
                    height=manifest.camera_calibration.native_height_px,
                    units="px",
                    authority_kind="selected_camera_calibration",
                    node_path=str(normalized),
                    camera_id=str(camera_id),
                ),
                [],
            )
        if selectors == ("selected_output_geometry",):
            attrs = target.attributes
            geometry = attrs.get("selected_output_geometry")
            geometry_match = (
                _OUTPUT_GEOMETRY_RE.fullmatch(geometry)
                if isinstance(geometry, str)
                else None
            )
            valid_display = (
                PurePosixPath(normalized or "").name == "display_snapshot"
                and _DISPLAY_AUTHORITY_REQUIRED_ATTRS <= set(attrs)
                and isinstance(attrs.get("selected_output_name"), str)
                and bool(str(attrs.get("selected_output_name")))
                and geometry_match is not None
                and attrs.get("selected_output_transform_token") == "normal"
                and isinstance(attrs.get("source_display_dataset_path"), str)
                and str(attrs.get("source_display_dataset_path")).startswith("/")
                and isinstance(attrs.get("source_display_dataset_sha256"), str)
                and _SHA256_HEX_RE.fullmatch(
                    str(attrs.get("source_display_dataset_sha256"))
                )
                is not None
                and attrs.get("source_display_dataset_digest_canonicalization")
                == "utf8_bytes_v1"
            )
            if not valid_display:
                return None, _authority_issue(
                    "REFERENCE_AUTHORITY_TARGET_INVALID",
                    "Display extent authority must resolve to the exact controlled selected-output snapshot schema.",
                    authority=text,
                    resolved_ref=normalized,
                )
            assert geometry_match is not None
            return (
                ResolvedReferenceAuthority(
                    width=int(geometry_match.group("width")),
                    height=int(geometry_match.group("height")),
                    units="px",
                    authority_kind="selected_display_snapshot",
                    node_path=str(normalized),
                ),
                [],
            )
        return None, _authority_issue(
            "REFERENCE_AUTHORITY_SYNTAX_UNSUPPORTED",
            "At-sign authorities support only exact selected-camera or selected-output selectors.",
            authority=text,
            selectors=selectors,
        )

    match = _ATTR_FIELD_AUTHORITY_RE.fullmatch(text)
    if match:
        normalized = _normalize_archive_ref(match.group("node"))
        target = nodes.get(normalized or "")
        attr_name = match.group("attr")
        field_name = match.group("field")
        if target is None or attr_name not in target.attributes:
            return None, _authority_issue(
                "REFERENCE_AUTHORITY_UNRESOLVED",
                "Nested attr authority does not resolve to a persisted attr.",
                authority=text,
                resolved_ref=normalized,
                attr=attr_name,
            )
        expected_scope = {
            "coordinate_transform": "run_level_legacy_texture_space",
            "legacy_texture_to_camera_transform": "legacy_texture_space_fallback",
        }.get(attr_name)
        expected_space = {
            "texture_dimensions": "stimulus_texture_px",
            "camera_dimensions": "source_camera_image_px",
        }.get(field_name)
        record = _as_mapping(target.attributes[attr_name])
        texture = record.get("texture_dimensions")
        camera = record.get("camera_dimensions")
        scale = record.get("texture_to_camera_scale")
        valid_legacy = (
            expected_scope is not None
            and expected_space is not None
            and str(space_id) == expected_space
            and record.get("scope") == expected_scope
            and isinstance(texture, (list, tuple))
            and len(texture) == 2
            and all(_positive_extent_value(value) for value in texture)
            and isinstance(camera, (list, tuple))
            and len(camera) == 2
            and all(_positive_extent_value(value) for value in camera)
            and _positive_extent_value(scale)
            and _extent_values_equal(float(camera[0]) / float(texture[0]), scale)
            and _extent_values_equal(float(camera[1]) / float(texture[1]), scale)
        )
        if not valid_legacy:
            return None, _authority_issue(
                "REFERENCE_AUTHORITY_TARGET_INVALID",
                "Nested attr authority is not an exact controlled legacy texture/camera extent record.",
                authority=text,
                attr=attr_name,
                field=field_name,
            )
        selected = texture if field_name == "texture_dimensions" else camera
        return (
            ResolvedReferenceAuthority(
                width=selected[0],
                height=selected[1],
                units="px",
                authority_kind="controlled_legacy_texture_camera_extent",
                node_path=str(normalized),
            ),
            [],
        )

    match = _ATTR_SELECTOR_AUTHORITY_RE.fullmatch(text)
    if match:
        normalized = _normalize_archive_ref(match.group("node"))
        target = nodes.get(normalized or "")
        selectors = tuple(item.strip() for item in match.group("selectors").split(","))
        if target is None or any(
            not selector or selector not in target.attributes for selector in selectors
        ):
            return None, _authority_issue(
                "REFERENCE_AUTHORITY_UNRESOLVED",
                "Attr authority does not resolve every selected persisted attr.",
                authority=text,
                resolved_ref=normalized,
                selectors=selectors,
            )
        if selectors == _ARENA_EXTENT_SELECTORS:
            attrs = target.attributes
            if (
                str(space_id) != "arena_relative_canvas_px"
                or PurePosixPath(normalized or "").name != "arena_geometry"
                or not all(_positive_extent_value(attrs.get(name)) for name in selectors)
            ):
                return None, _authority_issue(
                    "REFERENCE_AUTHORITY_TARGET_INVALID",
                    "Arena extent selectors require the canonical arena_geometry record and space.",
                    authority=text,
                )
            return (
                ResolvedReferenceAuthority(
                    width=attrs[selectors[0]],
                    height=attrs[selectors[1]],
                    units="px",
                    authority_kind="canonical_arena_geometry_attrs",
                    node_path=str(normalized),
                ),
                [],
            )
        if len(selectors) != 1:
            return None, _authority_issue(
                "REFERENCE_AUTHORITY_SYNTAX_UNSUPPORTED",
                "Generic width/height attr-name suffixes are not coordinate authority proof.",
                authority=text,
                selectors=selectors,
            )
        selected = _as_mapping(target.attributes[selectors[0]])
        if set(selected) == _PHYSICAL_FRAME_RECORD_FIELDS:
            return (
                ResolvedReferenceAuthority(
                    width=None,
                    height=None,
                    units="not_applicable",
                    authority_kind="physical_frame_record_attr",
                    node_path=str(normalized),
                    physical_record=selected,
                ),
                [],
            )
        if set(selected) == _REFERENCE_EXTENT_RECORD_FIELDS:
            valid_extent_record = (
                selected.get("schema_id") == "palette.coordinate_reference_extent"
                and selected.get("schema_version") == 1
                and selected.get("space_id") == space_id
                and selected.get("units") in {"px", "mm"}
                and _positive_extent_value(selected.get("width"))
                and _positive_extent_value(selected.get("height"))
            )
            if valid_extent_record:
                return (
                    ResolvedReferenceAuthority(
                        width=selected["width"],
                        height=selected["height"],
                        units=str(selected["units"]),
                        authority_kind="coordinate_reference_extent_record",
                        node_path=str(normalized),
                    ),
                    [],
                )
        return None, _authority_issue(
            "REFERENCE_AUTHORITY_SYNTAX_UNSUPPORTED",
            "Single-attr authority must select an exact controlled extent or physical-frame record.",
            authority=text,
        )

    match = _ATTRS_AUTHORITY_RE.fullmatch(text)
    if match:
        normalized = _normalize_archive_ref(match.group("node"))
        target = nodes.get(normalized or "")
        if target is None:
            return None, _authority_issue(
                "REFERENCE_AUTHORITY_UNRESOLVED",
                "Attrs authority does not resolve to persisted metadata.",
                authority=text,
                resolved_ref=normalized,
            )
        attrs = target.attributes
        arena_fields_valid = (
            str(space_id) == "arena_relative_canvas_px"
            and PurePosixPath(normalized or "").name == "arena_geometry"
            and all(_positive_extent_value(attrs.get(name)) for name in _ARENA_EXTENT_SELECTORS)
            and _finite_extent_value(attrs.get("arena_origin_in_canvas_x_px"))
            and _finite_extent_value(attrs.get("arena_origin_in_canvas_y_px"))
        )
        if not arena_fields_valid:
            return None, _authority_issue(
                "REFERENCE_AUTHORITY_SYNTAX_UNSUPPORTED",
                "Whole-attrs authority is supported only for the exact canonical arena_geometry record.",
                authority=text,
            )
        return (
            ResolvedReferenceAuthority(
                width=attrs[_ARENA_EXTENT_SELECTORS[0]],
                height=attrs[_ARENA_EXTENT_SELECTORS[1]],
                units="px",
                authority_kind="canonical_arena_geometry_attrs",
                node_path=str(normalized),
            ),
            [],
        )

    normalized = _normalize_archive_ref(text)
    target = nodes.get(normalized or "")
    if target is None:
        return None, _authority_issue(
            "REFERENCE_AUTHORITY_UNRESOLVED",
            "Reference authority does not resolve to persisted metadata.",
            authority=text,
            resolved_ref=normalized,
        )
    if set(target.attributes) == _PHYSICAL_FRAME_RECORD_FIELDS:
        return (
            ResolvedReferenceAuthority(
                width=None,
                height=None,
                units="not_applicable",
                authority_kind="physical_frame_record_node",
                node_path=str(normalized),
                physical_record=dict(target.attributes),
            ),
            [],
        )
    return None, _authority_issue(
        "REFERENCE_AUTHORITY_SYNTAX_UNSUPPORTED",
        "Reference authority must use an exact controlled shape, attr, selected-calibration, or physical-frame record.",
        authority=text,
        resolved_ref=normalized,
    )


def _reference_authority_issues(
    *,
    authority: Any,
    reference_width: Any,
    reference_height: Any,
    reference_units: Any,
    space_id: Any,
    nodes: Mapping[str, MetadataNode],
    physical_frame: Any = None,
    component_units: Sequence[Any] = (),
    origin: Any = None,
    positive_directions: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Verify descriptor values against one exact controlled authority record."""

    resolved, issues = _resolve_reference_authority(
        authority=authority,
        space_id=space_id,
        nodes=nodes,
    )
    if resolved is None:
        return issues
    text = str(authority).strip()
    if str(space_id) == "physical_mm":
        if resolved.physical_record is None:
            return _authority_issue(
                "PHYSICAL_FRAME_AUTHORITY_REQUIRED",
                "physical_mm requires an exact controlled physical-frame record, not a generic extent target.",
                authority=text,
            )
        physical_issues = _physical_frame_record_issues(
            resolved.physical_record,
            authority=text,
            physical_frame=physical_frame,
            component_units=component_units,
            origin=origin,
            positive_directions=positive_directions,
            reference_units=reference_units,
            nodes=nodes,
        )
        if reference_width is not None or reference_height is not None:
            physical_issues.append(
                _issue(
                    "PHYSICAL_FRAME_EXTENT_UNPROVEN",
                    "error",
                    "The physical-frame schema identifies the frame but does not prove a finite millimetre extent.",
                    authority=text,
                    declared_width=reference_width,
                    declared_height=reference_height,
                )
            )
        return physical_issues
    if resolved.physical_record is not None:
        return _authority_issue(
            "PHYSICAL_FRAME_AUTHORITY_SPACE_MISMATCH",
            "A physical-frame authority cannot prove a non-physical coordinate space.",
            authority=text,
            space_id=space_id,
        )

    if str(reference_units) != resolved.units:
        issues.append(
            _issue(
                "REFERENCE_AUTHORITY_UNITS_MISMATCH",
                "error",
                "Descriptor reference units disagree with the selected authority.",
                authority=text,
                declared_units=reference_units,
                authority_units=resolved.units,
            )
        )
    if not _extent_values_equal(reference_width, resolved.width) or not _extent_values_equal(
        reference_height, resolved.height
    ):
        issues.append(
            _issue(
                "REFERENCE_AUTHORITY_EXTENT_MISMATCH",
                "error",
                "Descriptor width/height disagree with the exact selected authority.",
                authority=text,
                declared_width=reference_width,
                declared_height=reference_height,
                authority_width=resolved.width,
                authority_height=resolved.height,
            )
        )
    return issues


def _validate_record_refs(
    descriptor: Mapping[str, Any],
    *,
    field_name: str,
    nodes: Mapping[str, MetadataNode],
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    raw_refs = descriptor.get(field_name)
    if not isinstance(raw_refs, (list, tuple)):
        return issues
    for index, raw_ref in enumerate(raw_refs):
        if not isinstance(raw_ref, Mapping):
            continue  # canonical descriptor validation reports this separately
        ref = raw_ref.get("ref")
        if not isinstance(ref, str) or not ref.strip():
            continue
        node_ref, selector_kind, selector = _record_ref_target(ref)
        normalized = _normalize_archive_ref(node_ref)
        target = nodes.get(normalized or "")
        declared_digest = raw_ref.get("sha256")
        if not isinstance(declared_digest, str):
            issues.append(
                _issue(
                    "COORDINATE_RECORD_DIGEST_MISSING",
                    "error",
                    "Canonical coordinate lineage and transform refs must be digest-bound.",
                    field=field_name,
                    index=index,
                    ref=ref,
                )
            )
        if target is None:
            issues.append(
                _issue(
                    "COORDINATE_RECORD_REF_UNRESOLVED",
                    "error",
                    "Coordinate lineage/transform reference does not resolve to persisted archive metadata.",
                    field=field_name,
                    index=index,
                    ref=ref,
                )
            )
            continue

        selected_value: Any = None
        selected = False
        if selector_kind == "attr":
            if selector not in target.attributes:
                issues.append(
                    _issue(
                        "COORDINATE_RECORD_ATTR_UNRESOLVED",
                        "error",
                        "Coordinate record reference names a missing persisted attr.",
                        field=field_name,
                        ref=ref,
                        attr=selector,
                    )
                )
                continue
            selected_value = target.attributes[selector]
            selected = True
        elif selector_kind == "fragment":
            if selector not in target.attributes:
                issues.append(
                    _issue(
                        "COORDINATE_RECORD_SELECTOR_UNRESOLVED",
                        "error",
                        "Coordinate record fragment does not name an exact persisted attr record.",
                        field=field_name,
                        ref=ref,
                        attr=selector,
                    )
                )
                continue
            selected_value = target.attributes[selector]
            selected = True

        if field_name == "transform_refs":
            if selector_kind in {"attr", "fragment"}:
                issues.append(
                    _issue(
                        "TRANSFORM_REF_NOT_DIRECTION_EXPLICIT",
                        "error",
                        "Canonical transform refs must target a complete persisted directed-transform node, not an arbitrary attr.",
                        ref=ref,
                    )
                )
                continue
            try:
                transform = load_directed_homography_attrs(target.attributes)
            except DirectedTransformError as exc:
                issues.append(
                    _issue(
                        "DIRECTED_TRANSFORM_REF_INVALID",
                        "error",
                        "Referenced transform metadata or its digest is invalid.",
                        ref=ref,
                        error=str(exc),
                    )
                )
                continue
            if target.node_type != "array" or list(target.shape or []) != [3, 3]:
                issues.append(
                    _issue(
                        "DIRECTED_TRANSFORM_TARGET_INVALID",
                        "error",
                        "Directed homography ref must resolve to a persisted 3x3 array node.",
                        ref=ref,
                        node_type=target.node_type,
                        shape=target.shape,
                    )
                )
            actual_digest = transform.digest()
        else:
            actual_digest = (
                _fingerprint(selected_value)
                if selected
                else _metadata_node_record_digest(target)
            )

        if isinstance(declared_digest, str) and declared_digest != actual_digest:
            issues.append(
                _issue(
                    "COORDINATE_RECORD_DIGEST_MISMATCH",
                    "error",
                    "Coordinate record digest does not match the exact referenced persisted record.",
                    field=field_name,
                    ref=ref,
                    declared_sha256=declared_digest,
                    actual_sha256=actual_digest,
                )
            )
    return issues


def _valid_directed_transform_refs(
    descriptor: Mapping[str, Any],
    *,
    nodes: Mapping[str, MetadataNode],
) -> list[tuple[str, Any]]:
    resolved: list[tuple[str, Any]] = []
    raw_refs = descriptor.get("transform_refs")
    if not isinstance(raw_refs, (list, tuple)):
        return resolved
    for raw_ref in raw_refs:
        if not isinstance(raw_ref, Mapping):
            continue
        ref = raw_ref.get("ref")
        digest = raw_ref.get("sha256")
        if not isinstance(ref, str) or not isinstance(digest, str):
            continue
        node_ref, selector_kind, _ = _record_ref_target(ref)
        if selector_kind is not None:
            continue
        target = nodes.get(_normalize_archive_ref(node_ref) or "")
        if target is None or target.node_type != "array" or list(target.shape or []) != [3, 3]:
            continue
        try:
            transform = load_directed_homography_attrs(target.attributes)
        except DirectedTransformError:
            continue
        if transform.digest() != digest:
            continue
        resolved.append((ref, transform))
    return resolved


def _transform_descriptor_issues(
    descriptor: Mapping[str, Any],
    *,
    nodes: Mapping[str, MetadataNode],
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    raw_refs = descriptor.get("transform_refs")
    has_refs = isinstance(raw_refs, (list, tuple)) and bool(raw_refs)
    overlay = descriptor.get("source_camera_overlay")
    if overlay == "requires_transform" and not has_refs:
        return [
            _issue(
                "REQUIRED_TRANSFORM_REF_MISSING",
                "error",
                "source_camera_overlay='requires_transform' requires a digest-bound directed transform chain.",
            )
        ]
    transforms = _valid_directed_transform_refs(descriptor, nodes=nodes)
    if not transforms:
        return issues

    raw_ref_count = len(raw_refs) if isinstance(raw_refs, (list, tuple)) else 0
    if len(transforms) != raw_ref_count:
        issues.append(
            _issue(
                "TRANSFORM_CHAIN_UNRESOLVED",
                "error",
                "Every transform_refs entry must resolve to one digest-bound directed transform before a chain can be trusted.",
                declared_ref_count=raw_ref_count,
                resolved_ref_count=len(transforms),
            )
        )

    surface_space = descriptor.get("space_id")
    if not isinstance(surface_space, str) or not surface_space:
        return issues
    directed_edges = [
        {
            "ref": ref,
            "transform_id": transform.transform_id,
            "from_space_id": transform.from_space_id,
            "to_space_id": transform.to_space_id,
        }
        for ref, transform in transforms
    ]

    chain_defects: list[dict[str, Any]] = []
    for index, ((left_ref, left), (right_ref, right)) in enumerate(
        zip(transforms, transforms[1:], strict=False)
    ):
        if left.to_space_id != right.from_space_id:
            chain_defects.append(
                {
                    "kind": "adjacency",
                    "index": index,
                    "left_ref": left_ref,
                    "left_to_space_id": left.to_space_id,
                    "right_ref": right_ref,
                    "right_from_space_id": right.from_space_id,
                }
            )
        if _canonical_json(left.target_reference_extent.to_dict()) != _canonical_json(
            right.source_reference_extent.to_dict()
        ):
            issues.append(
                _issue(
                    "TRANSFORM_CHAIN_EXTENT_MISMATCH",
                    "error",
                    "Adjacent directed transforms disagree on the exact shared-space extent and authority.",
                    left_ref=left_ref,
                    right_ref=right_ref,
                    space_id=left.to_space_id,
                    left_target_extent=left.target_reference_extent.to_dict(),
                    right_source_extent=right.source_reference_extent.to_dict(),
                )
            )

    space_sequence = [transforms[0][1].from_space_id] + [
        transform.to_space_id for _, transform in transforms
    ]
    repeated_spaces = sorted(
        space for space, count in Counter(space_sequence).items() if count > 1
    )
    if repeated_spaces:
        chain_defects.append(
            {
                "kind": "cycle_or_repeated_space",
                "spaces": repeated_spaces,
            }
        )
    refs = [ref for ref, _ in transforms]
    transform_ids = [transform.transform_id for _, transform in transforms]
    if len(set(refs)) != len(refs) or len(set(transform_ids)) != len(transform_ids):
        chain_defects.append(
            {
                "kind": "duplicate_ref_or_transform_id",
                "refs": refs,
                "transform_ids": transform_ids,
            }
        )
    if chain_defects:
        issues.append(
            _issue(
                "TRANSFORM_CHAIN_NOT_LINEAR",
                "error",
                "Canonical transform_refs must be one ordered, acyclic, unbranched linear chain.",
                defects=chain_defects,
                directed_edges=directed_edges,
            )
        )
        issues.append(
            _issue(
                "TRANSFORM_CHAIN_DISCONNECTED_OR_REVERSED",
                "error",
                "The declared transform order contains a disconnected, competing, cyclic, or reversed edge.",
                directed_edges=directed_edges,
            )
        )

    last_ref, last_transform = transforms[-1]
    if last_transform.to_space_id != surface_space:
        issues.append(
            _issue(
                "TRANSFORM_DIRECTION_INCOMPATIBLE_WITH_SURFACE",
                "error",
                "The final referenced directed transform does not terminate in the persisted surface space.",
                surface_space_id=surface_space,
                directed_edges=directed_edges,
            )
        )
        if not chain_defects:
            issues.append(
                _issue(
                    "TRANSFORM_CHAIN_DISCONNECTED_OR_REVERSED",
                    "error",
                    "The directed chain endpoint is disconnected from or reversed relative to the persisted surface.",
                    directed_edges=directed_edges,
                )
            )

    first_transform = transforms[0][1]
    if (
        overlay == "requires_transform"
        and first_transform.from_space_id != "source_camera_image_px"
    ):
        issues.append(
            _issue(
                "SOURCE_CAMERA_TRANSFORM_ROUTE_MISSING",
                "error",
                "A source-camera overlay route must begin at source_camera_image_px and follow transform_refs order.",
                surface_space_id=surface_space,
                chain_start_space_id=first_transform.from_space_id,
            )
        )

    extent = _as_mapping(descriptor.get("reference_extent"))
    target_extent = last_transform.target_reference_extent
    if not (
        _extent_values_equal(extent.get("width"), target_extent.width)
        and _extent_values_equal(extent.get("height"), target_extent.height)
        and extent.get("units") == target_extent.units
        and extent.get("authority") == target_extent.authority
    ):
        issues.append(
            _issue(
                "TRANSFORM_TARGET_EXTENT_MISMATCH",
                "error",
                "Final directed-transform target extent does not exactly match the persisted surface descriptor.",
                ref=last_ref,
                descriptor_extent=extent,
                transform_target_extent=target_extent.to_dict(),
            )
        )

    camera_bindings: list[dict[str, str]] = []
    for ref, transform in transforms:
        calibration_path = _normalize_archive_ref(transform.calibration_ref)
        if calibration_path not in nodes:
            issues.append(
                _issue(
                    "TRANSFORM_CALIBRATION_REF_UNRESOLVED",
                    "error",
                    "Directed transform calibration_ref does not resolve to persisted metadata.",
                    ref=ref,
                    calibration_ref=transform.calibration_ref,
                )
            )
        for endpoint, endpoint_space, endpoint_extent in (
            ("source", transform.from_space_id, transform.source_reference_extent),
            ("target", transform.to_space_id, transform.target_reference_extent),
        ):
            issues.extend(
                _reference_authority_issues(
                    authority=endpoint_extent.authority,
                    reference_width=endpoint_extent.width,
                    reference_height=endpoint_extent.height,
                    reference_units=endpoint_extent.units,
                    space_id=endpoint_space,
                    nodes=nodes,
                )
            )
            if endpoint_space not in CAMERA_BOUND_SPACE_IDS:
                continue
            resolved, _authority_issues = _resolve_reference_authority(
                authority=endpoint_extent.authority,
                space_id=endpoint_space,
                nodes=nodes,
            )
            if resolved is None or resolved.camera_id is None:
                issues.append(
                    _issue(
                        "TRANSFORM_CAMERA_AUTHORITY_IDENTITY_MISSING",
                        "error",
                        "A camera-bound transform endpoint must use an exact selected-camera calibration authority.",
                        ref=ref,
                        endpoint=endpoint,
                        authority=endpoint_extent.authority,
                        transform_camera_id=transform.camera_id,
                    )
                )
                continue
            camera_bindings.append(
                {
                    "ref": ref,
                    "camera_id": resolved.camera_id,
                    "calibration_path": resolved.node_path,
                }
            )
            if transform.camera_id != resolved.camera_id:
                issues.append(
                    _issue(
                        "TRANSFORM_CAMERA_ID_MISMATCH",
                        "error",
                        "Directed-transform camera_id disagrees with its selected-camera extent authority.",
                        ref=ref,
                        endpoint=endpoint,
                        transform_camera_id=transform.camera_id,
                        authority_camera_id=resolved.camera_id,
                    )
                )
            if calibration_path != resolved.node_path:
                issues.append(
                    _issue(
                        "TRANSFORM_CAMERA_CALIBRATION_MISMATCH",
                        "error",
                        "Camera-bound transform calibration_ref must name the same selected-camera snapshot as its extent authority.",
                        ref=ref,
                        transform_calibration_ref=transform.calibration_ref,
                        authority_calibration_ref=resolved.node_path,
                    )
                )

    camera_ids = {binding["camera_id"] for binding in camera_bindings}
    camera_paths = {binding["calibration_path"] for binding in camera_bindings}
    declared_camera_ids = {
        str(transform.camera_id)
        for _ref, transform in transforms
        if transform.camera_id is not None
    }
    declared_camera_paths = {
        str(_normalize_archive_ref(transform.calibration_ref))
        for _ref, transform in transforms
        if transform.camera_id is not None
    }
    if (
        len(camera_ids) > 1
        or len(camera_paths) > 1
        or len(declared_camera_ids) > 1
        or len(declared_camera_paths) > 1
    ):
        issues.append(
            _issue(
                "TRANSFORM_CHAIN_CAMERA_IDENTITY_CONFLICT",
                "error",
                "One transform chain may bind only one exact selected-camera identity and calibration snapshot.",
                camera_bindings=camera_bindings,
                declared_camera_ids=sorted(declared_camera_ids),
                declared_camera_paths=sorted(declared_camera_paths),
            )
        )
    # Metadata binds the expected matrix digest but this metadata-only scanner
    # deliberately does not read array payloads.  A valid-looking transform is
    # therefore a numerical-validation target, never proof of compatibility.
    for ref, transform in transforms:
        issues.append(
            _issue(
                "TRANSFORM_MATRIX_PAYLOAD_VALIDATION_REQUIRED",
                "warning",
                "Directed-transform metadata is valid, but its 3x3 payload must be hashed and numerically validated before use.",
                ref=ref,
                expected_matrix_sha256=transform.matrix_sha256,
            )
        )
    return issues


def _record_ref_identity(ref: str) -> tuple[str | None, str | None, str | None]:
    node_ref, selector_kind, selector = _record_ref_target(ref)
    return _normalize_archive_ref(node_ref), selector_kind, selector


def _physical_authority_lineage_issues(
    descriptor: Mapping[str, Any],
    *,
    nodes: Mapping[str, MetadataNode],
) -> list[dict[str, Any]]:
    if descriptor.get("space_id") != "physical_mm":
        return []
    extent = _as_mapping(descriptor.get("reference_extent"))
    authority = extent.get("authority")
    if not isinstance(authority, str) or not authority.strip():
        return []  # authority validation reports the missing value
    authority_identity = _record_ref_identity(authority)
    raw_refs = descriptor.get("lineage_refs")
    candidates = raw_refs if isinstance(raw_refs, (list, tuple)) else ()
    matching = [
        item
        for item in candidates
        if isinstance(item, Mapping)
        and isinstance(item.get("ref"), str)
        and _record_ref_identity(str(item["ref"])) == authority_identity
    ]
    if len(matching) != 1:
        return [
            _issue(
                "PHYSICAL_FRAME_LINEAGE_MISSING",
                "error",
                "physical_mm must link its exact physical-frame authority exactly once through lineage_refs.",
                authority=authority,
                matching_lineage_ref_count=len(matching),
            )
        ]

    node_path, selector_kind, selector = authority_identity
    target = nodes.get(node_path or "")
    if target is None:
        return []  # authority/ref validation reports the unresolved target
    if selector_kind == "attr" and selector in target.attributes:
        expected_digest = _fingerprint(target.attributes[selector])
    elif selector_kind is None:
        expected_digest = _metadata_node_record_digest(target)
    else:
        return []
    declared_digest = matching[0].get("sha256")
    if not isinstance(declared_digest, str):
        return [
            _issue(
                "PHYSICAL_FRAME_LINEAGE_DIGEST_MISSING",
                "error",
                "The exact physical-frame lineage reference must be digest-bound.",
                authority=authority,
            )
        ]
    if declared_digest != expected_digest:
        return [
            _issue(
                "PHYSICAL_FRAME_LINEAGE_DIGEST_MISMATCH",
                "error",
                "The physical-frame lineage digest does not match the exact authority record.",
                authority=authority,
                declared_sha256=declared_digest,
                actual_sha256=expected_digest,
            )
        ]
    return []


def _authority_digest_binding(
    authority: str,
    *,
    resolved: ResolvedReferenceAuthority,
    nodes: Mapping[str, MetadataNode],
) -> tuple[tuple[str | None, str | None, str | None], str | None]:
    """Return the exact lineage identity and digest for a resolved authority."""

    shape_match = _SHAPE_AUTHORITY_RE.fullmatch(authority)
    if shape_match:
        identity = (_normalize_archive_ref(shape_match.group("node")), None, None)
    else:
        node_ref, selector_kind, selector = _record_ref_target(authority)
        # Strip selector syntaxes that are part of the authority grammar rather
        # than the coordinate-record grammar.  A controlled record selected
        # from one attr is itself the authority, so its selector must remain in
        # the lineage identity and its digest must bind that exact record.
        for pattern in (
            _AT_AUTHORITY_RE,
            _ATTR_FIELD_AUTHORITY_RE,
            _ATTR_SELECTOR_AUTHORITY_RE,
            _ATTRS_AUTHORITY_RE,
        ):
            match = pattern.fullmatch(authority)
            if match:
                if (
                    pattern is _ATTR_SELECTOR_AUTHORITY_RE
                    and selector_kind == "attr"
                    and resolved.authority_kind
                    in {
                        "coordinate_reference_extent_record",
                        "physical_frame_record_attr",
                    }
                ):
                    break
                node_ref = match.group("node")
                selector_kind = None
                selector = None
                break
        identity = (_normalize_archive_ref(node_ref), selector_kind, selector)
    node_path, selector_kind, selector = identity
    target = nodes.get(node_path or resolved.node_path)
    if target is None:
        return identity, None
    if selector_kind in {"attr", "fragment"} and selector in target.attributes:
        return identity, _fingerprint(target.attributes[selector])
    return (resolved.node_path, None, None), _metadata_node_record_digest(target)


def _lineage_record_schema(
    raw_ref: Mapping[str, Any],
    *,
    nodes: Mapping[str, MetadataNode],
) -> str | None:
    ref = raw_ref.get("ref")
    if not isinstance(ref, str):
        return None
    node_ref, selector_kind, selector = _record_ref_target(ref)
    target = nodes.get(_normalize_archive_ref(node_ref) or "")
    if target is None:
        return None
    if selector_kind in {"attr", "fragment"}:
        record = _as_mapping(target.attributes.get(str(selector)))
    else:
        record = target.attributes
    schema_id = record.get("schema_id")
    return str(schema_id) if isinstance(schema_id, str) else None


def _reference_authority_lineage_issues(
    descriptor: Mapping[str, Any],
    *,
    nodes: Mapping[str, MetadataNode],
) -> list[dict[str, Any]]:
    """Require a role-controlled, digest-bound authority lineage edge."""

    space_id = descriptor.get("space_id")
    extent = _as_mapping(descriptor.get("reference_extent"))
    authority = extent.get("authority")
    if not isinstance(authority, str) or not authority.strip():
        return []
    resolved, authority_issues = _resolve_reference_authority(
        authority=authority,
        space_id=space_id,
        nodes=nodes,
    )
    if resolved is None:
        return []  # the extent resolver reports the primary defect

    issues: list[dict[str, Any]] = []
    allowed_kinds_by_space = {
        "source_camera_image_px": {"selected_camera_calibration"},
        "source_camera_normalized_xy": {"selected_camera_calibration"},
        "detector_model_input_px": {"coordinate_reference_extent_record"},
        "detector_normalized_xy": {"coordinate_reference_extent_record"},
        "roi_local_px": {"array_shape_trailing_yx", "coordinate_reference_extent_record"},
        "stimulus_texture_px": {"selected_display_snapshot", "coordinate_reference_extent_record"},
        "stimulus_canvas_px": {"selected_display_snapshot", "coordinate_reference_extent_record"},
        "arena_relative_canvas_px": {"canonical_arena_geometry_attrs"},
        "physical_mm": {"physical_frame_record_attr", "physical_frame_record_node"},
        "fish_anatomical_body_frame": {"coordinate_reference_extent_record"},
    }
    allowed_kinds = allowed_kinds_by_space.get(str(space_id), set())
    if resolved.authority_kind not in allowed_kinds:
        issues.append(
            _issue(
                "REFERENCE_AUTHORITY_ROLE_INVALID",
                "error",
                "Reference authority kind is not controlled for the declared coordinate space.",
                space_id=space_id,
                authority=authority,
                authority_kind=resolved.authority_kind,
                allowed_authority_kinds=sorted(allowed_kinds),
            )
        )

    if resolved.authority_kind == "array_shape_trailing_yx":
        target = nodes.get(resolved.node_path)
        expected_role = {
            "roi_local_px": "crop_roi_raster",
            "detector_model_input_px": "detector_model_input_raster",
            "stimulus_texture_px": "stimulus_texture_raster",
            "stimulus_canvas_px": "stimulus_canvas_raster",
        }.get(str(space_id))
        actual_role = (
            target.attributes.get("coordinate_authority_role")
            if target is not None
            else None
        )
        if expected_role is None or actual_role != expected_role:
            issues.append(
                _issue(
                    "REFERENCE_AUTHORITY_ROLE_INVALID",
                    "error",
                    "An arbitrary same-shape array is not coordinate authority; the target needs the controlled role for this space.",
                    authority=authority,
                    expected_role=expected_role,
                    actual_role=actual_role,
                    target_path=resolved.node_path,
                )
            )

    identity, expected_digest = _authority_digest_binding(
        authority,
        resolved=resolved,
        nodes=nodes,
    )
    lineage = descriptor.get("lineage_refs")
    refs = lineage if isinstance(lineage, (list, tuple)) else ()
    matching = [
        raw
        for raw in refs
        if isinstance(raw, Mapping)
        and isinstance(raw.get("ref"), str)
        and _record_ref_identity(str(raw["ref"])) == identity
    ]
    if len(matching) != 1:
        issues.append(
            _issue(
                "REFERENCE_AUTHORITY_LINEAGE_MISSING",
                "error",
                "The exact reference authority must appear once in digest-bound descriptor lineage.",
                authority=authority,
                resolved_authority_identity=identity,
                matching_lineage_ref_count=len(matching),
            )
        )
    elif expected_digest is None or matching[0].get("sha256") != expected_digest:
        issues.append(
            _issue(
                "REFERENCE_AUTHORITY_LINEAGE_DIGEST_MISMATCH",
                "error",
                "Reference-authority lineage digest does not bind the exact controlled metadata record.",
                authority=authority,
                declared_sha256=matching[0].get("sha256"),
                actual_sha256=expected_digest,
            )
        )

    if space_id == "roi_local_px":
        placement_refs = [
            raw
            for raw in refs
            if isinstance(raw, Mapping)
            and _lineage_record_schema(raw, nodes=nodes)
            in {"palette.crop_placement", "palette.crop_placement_lineage"}
        ]
        if len(placement_refs) != 1 or not isinstance(
            placement_refs[0].get("sha256") if placement_refs else None, str
        ):
            issues.append(
                _issue(
                    "ROI_CROP_PLACEMENT_LINEAGE_MISSING",
                    "error",
                    "ROI-local coordinates require one exact digest-bound crop-placement record.",
                    crop_placement_ref_count=len(placement_refs),
                )
            )
    return issues


def _metadata_dtype(value: Any) -> np.dtype[Any] | None:
    try:
        return np.dtype(value)
    except (TypeError, ValueError):
        return None


def _row_identity_refs(row_identity: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Return controlled identity component names and archive refs.

    Version 1 descriptors expose one ``array_ref``.  Version 2 descriptors may
    expose named ``key_arrays``/``array_refs`` so composite keys are explicit.
    Keeping this extraction narrow prevents arbitrary nested strings from being
    treated as identity proof.
    """

    refs: list[tuple[str, str]] = []
    array_ref = row_identity.get("array_ref")
    if isinstance(array_ref, str) and array_ref.strip():
        refs.append((str(row_identity.get("key_name") or "identity"), array_ref))
    for field in ("key_arrays", "array_refs"):
        raw = row_identity.get(field)
        if isinstance(raw, Mapping):
            for name, ref in raw.items():
                if isinstance(ref, str) and ref.strip():
                    refs.append((str(name), ref))
        elif isinstance(raw, (list, tuple)):
            for index, item in enumerate(raw):
                if isinstance(item, Mapping):
                    name = item.get("name") or item.get("component") or index
                    ref = item.get("ref") or item.get("array_ref")
                    if isinstance(ref, str) and ref.strip():
                        refs.append((str(name), ref))
                elif isinstance(item, str) and item.strip():
                    refs.append((str(index), item))
    return list(dict.fromkeys(refs))


def _surface_leading_dimension(
    surface_node: MetadataNode,
    *,
    nodes: Mapping[str, MetadataNode],
    excluded_paths: set[str],
) -> tuple[int | None, list[int]]:
    shape = surface_node.shape
    leaf = PurePosixPath(surface_node.relative_path).name
    if (
        surface_node.node_type == "array"
        and leaf in {"contours_left", "contours_right"}
        and isinstance(shape, (list, tuple))
        and len(shape) == 2
    ):
        side = "left" if leaf == "contours_left" else "right"
        parent = PurePosixPath(surface_node.relative_path).parent.as_posix()
        ptr = nodes.get(f"{parent}/contour_{side}_ptr")
        length = nodes.get(f"{parent}/contour_{side}_len")
        if (
            ptr is not None
            and length is not None
            and ptr.node_type == length.node_type == "array"
            and isinstance(ptr.shape, (list, tuple))
            and isinstance(length.shape, (list, tuple))
            and len(ptr.shape) == len(length.shape) == 1
            and tuple(ptr.shape) == tuple(length.shape)
        ):
            return int(ptr.shape[0]), []
        return None, []
    if (
        surface_node.node_type == "array"
        and leaf in {"counts", "indptr"}
        and "mask_rle" in PurePosixPath(surface_node.relative_path).parts
    ):
        component = PurePosixPath(surface_node.relative_path).parent.as_posix()
        indptr = nodes.get(f"{component}/indptr")
        if (
            indptr is not None
            and indptr.node_type == "array"
            and isinstance(indptr.shape, (list, tuple))
            and len(indptr.shape) == 1
            and int(indptr.shape[0]) >= 1
        ):
            return int(indptr.shape[0]) - 1, []
        return None, []
    if (
        surface_node.node_type == "array"
        and leaf == "points_xy"
        and PurePosixPath(surface_node.relative_path).parent.name == "contours"
        and isinstance(shape, (list, tuple))
        and len(shape) == 2
    ):
        # ``contours/points_xy`` is a flattened point store.  Its observation
        # axis is defined by the sibling ptr/len arrays, never by point count.
        parent = PurePosixPath(surface_node.relative_path).parent.as_posix()
        ptr = nodes.get(f"{parent}/ptr")
        length = nodes.get(f"{parent}/len")
        if (
            ptr is not None
            and length is not None
            and ptr.node_type == length.node_type == "array"
            and isinstance(ptr.shape, (list, tuple))
            and isinstance(length.shape, (list, tuple))
            and len(ptr.shape) == len(length.shape) == 1
            and tuple(ptr.shape) == tuple(length.shape)
        ):
            return int(ptr.shape[0]), []
        return None, []
    if isinstance(shape, (list, tuple)) and shape:
        return int(shape[0]), []
    if surface_node.node_type != "group":
        return None, []
    child_counts = [
        int(candidate.shape[0])
        for path, candidate in nodes.items()
        if candidate.node_type == "array"
        and str(PurePosixPath(path).parent) == surface_node.relative_path
        and path not in excluded_paths
        and isinstance(candidate.shape, (list, tuple))
        and candidate.shape
    ]
    distinct = sorted(set(child_counts))
    return (distinct[0] if len(distinct) == 1 else None), distinct


def _flattened_contour_lineage_issues(
    *,
    surface_type: str,
    node: MetadataNode,
    nodes: Mapping[str, MetadataNode],
) -> list[dict[str, Any]]:
    """Validate row offsets for the variable-length component contour store."""

    leaf = PurePosixPath(node.relative_path).name
    if (
        surface_type == "subject_mask_contour"
        and node.node_type == "array"
        and leaf in {"contours_left", "contours_right"}
        and isinstance(node.shape, (list, tuple))
        and len(node.shape) == 2
    ):
        side = "left" if leaf == "contours_left" else "right"
        parent_path = PurePosixPath(node.relative_path).parent.as_posix()
        ptr_path = f"{parent_path}/contour_{side}_ptr"
        len_path = f"{parent_path}/contour_{side}_len"
        ptr = nodes.get(ptr_path)
        length = nodes.get(len_path)
        if ptr is None or length is None:
            return [
                _issue(
                    "LEGACY_FLATTENED_CONTOUR_INDEX_LINEAGE_MISSING",
                    "error",
                    "Historical eye contours require exact side-specific ptr and len siblings.",
                    contour_path=node.relative_path,
                    ptr_path=ptr_path,
                    len_path=len_path,
                )
            ]
        ptr_dtype = _metadata_dtype(ptr.data_type)
        len_dtype = _metadata_dtype(length.data_type)
        if not (
            ptr.node_type == length.node_type == "array"
            and isinstance(ptr.shape, (list, tuple))
            and isinstance(length.shape, (list, tuple))
            and len(ptr.shape) == len(length.shape) == 1
            and tuple(ptr.shape) == tuple(length.shape)
            and ptr_dtype is not None
            and len_dtype is not None
            and ptr_dtype.kind in "iu"
            and len_dtype.kind in "iu"
            and int(node.shape[1]) == 2
        ):
            return [
                _issue(
                    "LEGACY_FLATTENED_CONTOUR_INDEX_LINEAGE_INVALID",
                    "error",
                    "Historical contour ptr/len must be equal-length rank-1 integer arrays and points must have shape (M,2).",
                    contour_path=node.relative_path,
                    ptr_shape=(ptr.shape if ptr else None),
                    len_shape=(length.shape if length else None),
                )
            ]
        return [
            _issue(
                "FLATTENED_CONTOUR_INDEX_PAYLOAD_VALIDATION_REQUIRED",
                "warning",
                "Contour index metadata is present, but ptr/len ranges and point payloads require live validation.",
                contour_path=node.relative_path,
                ptr_path=ptr_path,
                len_path=len_path,
            )
        ]

    if (
        surface_type != "subject_mask_contour"
        or node.node_type != "array"
        or leaf != "points_xy"
        or PurePosixPath(node.relative_path).parent.name != "contours"
        or not isinstance(node.shape, (list, tuple))
        or len(node.shape) != 2
    ):
        return []

    parent_path = PurePosixPath(node.relative_path).parent.as_posix()
    parent = nodes.get(parent_path)
    issues: list[dict[str, Any]] = []
    if (
        parent is None
        or parent.node_type != "group"
        or parent.attributes.get("schema_id") != COMPONENT_CONTOUR_SCHEMA_ID
        or parent.attributes.get("contour_schema_id")
        != COMPONENT_CONTOUR_SCHEMA_ID
    ):
        issues.append(
            _issue(
                "FLATTENED_CONTOUR_SCHEMA_MISSING",
                "error",
                "Flattened contour points require the exact component_contours_v1 parent schema.",
                contour_group=parent_path,
                schema_id=(parent.attributes.get("schema_id") if parent else None),
                contour_schema_id=(
                    parent.attributes.get("contour_schema_id") if parent else None
                ),
            )
        )

    ptr_path = f"{parent_path}/ptr"
    len_path = f"{parent_path}/len"
    ptr = nodes.get(ptr_path)
    length = nodes.get(len_path)
    if ptr is None or length is None:
        issues.append(
            _issue(
                "FLATTENED_CONTOUR_INDEX_LINEAGE_MISSING",
                "error",
                "Flattened contour points require sibling ptr and len row-index arrays.",
                ptr_path=ptr_path,
                len_path=len_path,
                ptr_present=ptr is not None,
                len_present=length is not None,
            )
        )
        return issues

    ptr_dtype = _metadata_dtype(ptr.data_type)
    len_dtype = _metadata_dtype(length.data_type)
    valid_index_metadata = (
        ptr.node_type == length.node_type == "array"
        and isinstance(ptr.shape, (list, tuple))
        and isinstance(length.shape, (list, tuple))
        and len(ptr.shape) == len(length.shape) == 1
        and tuple(ptr.shape) == tuple(length.shape)
        and ptr_dtype is not None
        and len_dtype is not None
        and ptr_dtype.kind in "iu"
        and len_dtype.kind in "iu"
        and len(node.shape) == 2
        and int(node.shape[1]) == 2
    )
    if not valid_index_metadata:
        issues.append(
            _issue(
                "FLATTENED_CONTOUR_INDEX_LINEAGE_INVALID",
                "error",
                "Contour ptr/len must be equal-length rank-1 integer arrays and points_xy must have shape (M,2).",
                points_shape=node.shape,
                ptr_shape=ptr.shape,
                ptr_dtype=ptr.data_type,
                len_shape=length.shape,
                len_dtype=length.data_type,
            )
        )
        return issues

    issues.append(
        _issue(
            "FLATTENED_CONTOUR_INDEX_PAYLOAD_VALIDATION_REQUIRED",
            "warning",
            "Contour row-index metadata is present, but ptr/len ranges and point coverage require payload validation.",
            ptr_path=ptr_path,
            len_path=len_path,
            row_count=int(ptr.shape[0]),
            point_count=int(node.shape[0]),
        )
    )
    return issues


def _subject_mask_rle_lineage_issues(
    *,
    surface_type: str,
    node: MetadataNode,
    nodes: Mapping[str, MetadataNode],
) -> list[dict[str, Any]]:
    """Validate exact RLE metadata topology; encoded values remain unchecked."""

    if surface_type != "subject_mask_compact_encoding":
        return []
    parts = PurePosixPath(node.relative_path).parts
    if node.node_type != "array" or not parts or parts[-1] not in {
        "counts",
        "indptr",
    }:
        return []
    try:
        rle_index = parts.index("mask_rle")
    except ValueError:
        return []
    if (
        len(parts) != rle_index + 4
        or parts[rle_index + 1] != "components"
    ):
        return [
            _issue(
                "SUBJECT_MASK_RLE_LAYOUT_INVALID",
                "error",
                "RLE arrays must use mask_rle/components/<component>/{counts,indptr}.",
                surface_path=node.relative_path,
            )
        ]
    rle_path = PurePosixPath(*parts[: rle_index + 1]).as_posix()
    component_path = PurePosixPath(*parts[: rle_index + 3]).as_posix()
    rle = nodes.get(rle_path)
    counts = nodes.get(f"{component_path}/counts")
    indptr = nodes.get(f"{component_path}/indptr")
    present = nodes.get(f"{component_path}/present")
    bbox = nodes.get(f"{component_path}/bbox_xyxy")
    encoded_shape = rle.attributes.get("encoded_shape_hw") if rle else None
    counts_dtype = _metadata_dtype(counts.data_type) if counts else None
    indptr_dtype = _metadata_dtype(indptr.data_type) if indptr else None
    present_dtype = _metadata_dtype(present.data_type) if present else None
    bbox_dtype = _metadata_dtype(bbox.data_type) if bbox else None
    n_rows = (
        int(indptr.shape[0]) - 1
        if indptr is not None
        and isinstance(indptr.shape, (list, tuple))
        and len(indptr.shape) == 1
        and int(indptr.shape[0]) >= 1
        else None
    )
    valid = (
        rle is not None
        and rle.node_type == "group"
        and rle.attributes.get("schema_id") == "palette_mask_rle_binary_v1"
        and rle.attributes.get("mask_encoding") == "coco_rle_fortran_v1"
        and rle.attributes.get("layout") == "component_groups"
        and isinstance(encoded_shape, (list, tuple))
        and len(encoded_shape) == 2
        and all(type(value) is int and value > 0 for value in encoded_shape)
        and counts is not None
        and counts.node_type == "array"
        and isinstance(counts.shape, (list, tuple))
        and len(counts.shape) == 1
        and counts_dtype == np.dtype("<u4")
        and indptr is not None
        and indptr.node_type == "array"
        and isinstance(indptr.shape, (list, tuple))
        and len(indptr.shape) == 1
        and indptr_dtype == np.dtype("<i8")
        and n_rows is not None
        and present is not None
        and present.node_type == "array"
        and tuple(present.shape or ()) == (n_rows,)
        and present_dtype == np.dtype("bool")
        and bbox is not None
        and bbox.node_type == "array"
        and tuple(bbox.shape or ()) == (n_rows, 4)
        and bbox_dtype is not None
        and bbox_dtype.kind in "iu"
    )
    if not valid:
        return [
            _issue(
                "SUBJECT_MASK_RLE_LINEAGE_INVALID",
                "error",
                "RLE counts require exact root encoding/shape metadata and sibling indptr, present, and bbox row lineage.",
                rle_path=rle_path,
                component_path=component_path,
                encoded_shape_hw=encoded_shape,
                counts_shape=(counts.shape if counts else None),
                counts_dtype=(counts.data_type if counts else None),
                indptr_shape=(indptr.shape if indptr else None),
                indptr_dtype=(indptr.data_type if indptr else None),
                present_shape=(present.shape if present else None),
                bbox_shape=(bbox.shape if bbox else None),
            )
        ]
    return [
        _issue(
            "SUBJECT_MASK_RLE_PAYLOAD_VALIDATION_REQUIRED",
            "warning",
            "RLE topology is exact, but indptr monotonicity, terminal counts length, and decoded shape require payload validation.",
            rle_path=rle_path,
            component_path=component_path,
        )
    ]


_ROW_IDENTITY_COMMON_REQUIRED_ATTRS = {
    "schema_id",
    "schema_version",
    "leading_dimension",
    "unique",
    "content_sha256",
    "digest_canonicalization",
}


def _row_identity_schema_issues(
    *,
    mode: Any,
    component_name: str,
    resolved: str,
    row_node: MetadataNode,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    attrs = row_node.attributes
    schema_id = attrs.get("schema_id")
    expected_schemas = {
        "instance_key": {"palette.instance_key_row_identity"},
        "stimulus_state_key": {"palette.coordinate_row_identity"},
        "explicit_array": {"palette.coordinate_row_identity"},
        "explicit_key": {"palette.coordinate_row_identity"},
        "track_frame_indices": {"palette.track_row_identity"},
        "track_key": {"palette.track_row_identity"},
        "frame_indices": {"palette.frame_row_identity"},
    }.get(str(mode), set())
    if (
        not expected_schemas
        or schema_id not in expected_schemas
        or attrs.get("schema_version") != 1
        or not _ROW_IDENTITY_COMMON_REQUIRED_ATTRS <= set(attrs)
    ):
        issues.append(
            _issue(
                "ROW_IDENTITY_SCHEMA_INVALID",
                "error",
                "Row identity arrays require a controlled schema, version, uniqueness declaration, and content digest.",
                mode=mode,
                component=component_name,
                row_identity_path=resolved,
                schema_id=schema_id,
                required_common_attrs=sorted(_ROW_IDENTITY_COMMON_REQUIRED_ATTRS),
                allowed_schema_ids=sorted(expected_schemas),
            )
        )
        return issues
    row_count = row_node.shape[0] if isinstance(row_node.shape, (list, tuple)) and row_node.shape else None
    if attrs.get("leading_dimension") != row_count or attrs.get("unique") is not True:
        issues.append(
            _issue(
                "ROW_IDENTITY_SCHEMA_MISMATCH",
                "error",
                "Row identity schema metadata disagrees with its array shape or uniqueness requirement.",
                row_identity_path=resolved,
                declared_leading_dimension=attrs.get("leading_dimension"),
                actual_leading_dimension=row_count,
                declared_unique=attrs.get("unique"),
            )
        )
    content_digest = attrs.get("content_sha256")
    if not isinstance(content_digest, str) or _SHA256_HEX_RE.fullmatch(content_digest) is None:
        issues.append(
            _issue(
                "ROW_IDENTITY_CONTENT_DIGEST_INVALID",
                "error",
                "Row identity content_sha256 must be an exact SHA-256 digest.",
                row_identity_path=resolved,
            )
        )
    canonicalization = attrs.get("digest_canonicalization")
    if canonicalization not in {
        "numpy_dtype_shape_c_order_bytes_v1",
        "little_endian_uint64_c_order_v1",
        "canonical_integer_array_v1",
    }:
        issues.append(
            _issue(
                "ROW_IDENTITY_DIGEST_CANONICALIZATION_INVALID",
                "error",
                "Row identity digest canonicalization is missing or unsupported.",
                row_identity_path=resolved,
                digest_canonicalization=canonicalization,
            )
        )
    return issues


def _legacy_row_identity_integrity_issues(
    *,
    descriptor: Mapping[str, Any],
    match: DescriptorMatch,
    surface_node: MetadataNode,
    nodes: Mapping[str, MetadataNode],
) -> list[dict[str, Any]]:
    row_identity = _as_mapping(descriptor.get("row_identity"))
    mode = row_identity.get("mode")
    if mode == "not_applicable":
        return []
    refs = _row_identity_refs(row_identity)
    if not refs:
        return [
            _issue(
                "ROW_IDENTITY_REF_MISSING",
                "error",
                "Canonical row identity must name one or more persisted key arrays.",
                mode=mode,
            )
        ]
    owner_node = nodes.get(match.owner_path)
    resolved_rows: list[tuple[str, str, MetadataNode]] = []
    issues: list[dict[str, Any]] = []
    for component_name, raw_ref in refs:
        resolved = _normalize_archive_ref(
            raw_ref,
            owner_path=match.owner_path,
            owner_is_array=bool(owner_node and owner_node.node_type == "array"),
        )
        row_node = nodes.get(resolved or "")
        if row_node is None or row_node.node_type != "array":
            issues.append(
                _issue(
                    "ROW_IDENTITY_REF_UNRESOLVED",
                    "error",
                    "Canonical row identity does not resolve to a persisted array.",
                    component=component_name,
                    array_ref=raw_ref,
                    resolved_ref=resolved,
                )
            )
            continue
        assert resolved is not None
        resolved_rows.append((component_name, resolved, row_node))

    excluded = {resolved for _, resolved, _ in resolved_rows}
    surface_count, component_counts = _surface_leading_dimension(
        surface_node,
        nodes=nodes,
        excluded_paths=excluded,
    )
    row_counts: list[int] = []
    for component_name, resolved, row_node in resolved_rows:
        shape = row_node.shape
        dtype = _metadata_dtype(row_node.data_type)
        rank = len(shape) if isinstance(shape, (list, tuple)) else None
        if rank is None or rank < 1:
            issues.append(
                _issue(
                    "ROW_IDENTITY_RANK_INVALID",
                    "error",
                    "Row identity arrays must have an explicit leading row dimension.",
                    row_identity_path=resolved,
                    shape=shape,
                )
            )
            continue
        row_counts.append(int(shape[0]))
        if mode == "instance_key":
            if rank != 1 or dtype != np.dtype("uint64") or PurePosixPath(resolved).name != "instance_key":
                issues.append(
                    _issue(
                        "INSTANCE_KEY_CONTRACT_INVALID",
                        "error",
                        "instance_key identity must resolve to a rank-1 uint64 array named instance_key.",
                        row_identity_path=resolved,
                        shape=shape,
                        dtype=str(dtype) if dtype is not None else row_node.data_type,
                    )
                )
        elif mode in {"stimulus_state_key", "explicit_array", "explicit_key"}:
            if (
                rank not in {1, 2}
                or dtype != np.dtype("int64")
                or PurePosixPath(resolved).name != "coordinate_row_identity"
            ):
                issues.append(
                    _issue(
                        "STIMULUS_ROW_IDENTITY_CONTRACT_INVALID",
                        "error",
                        "Stimulus state identity must resolve to a rank-1/2 signed-int64 coordinate_row_identity array.",
                        row_identity_path=resolved,
                        shape=shape,
                        dtype=str(dtype) if dtype is not None else row_node.data_type,
                    )
                )
        elif mode in {"track_frame_indices", "track_key", "frame_indices"}:
            if rank != 1 or dtype is None or dtype.kind not in "iu":
                issues.append(
                    _issue(
                        "TRACK_ROW_IDENTITY_CONTRACT_INVALID",
                        "error",
                        "Track/frame identity components must be rank-1 integer arrays.",
                        row_identity_path=resolved,
                        shape=shape,
                        dtype=str(dtype) if dtype is not None else row_node.data_type,
                    )
                )
        issues.extend(
            _row_identity_schema_issues(
                mode=mode,
                component_name=component_name,
                resolved=resolved,
                row_node=row_node,
            )
        )
    if len(set(row_counts)) > 1:
        issues.append(
            _issue(
                "ROW_IDENTITY_COMPONENT_LENGTH_MISMATCH",
                "error",
                "Composite identity component arrays disagree on row count.",
                row_counts=row_counts,
            )
        )
    if row_counts and surface_count is not None and any(count != surface_count for count in row_counts):
        issues.append(
            _issue(
                "ROW_IDENTITY_LENGTH_MISMATCH",
                "error",
                "Row identity leading dimension does not match the coordinate surface.",
                row_identity_counts=row_counts,
                surface_count=surface_count,
            )
        )
    if component_counts and len(component_counts) > 1:
        issues.append(
            _issue(
                "ROW_IDENTITY_LENGTH_MISMATCH",
                "error",
                "Coordinate component arrays disagree on their leading row dimension.",
                component_counts=component_counts,
            )
        )
    return issues


_LEGACY_ROW_IDENTITY_SCHEMA_IDS = frozenset(
    {
        "palette.coordinate_row_identity",
        "palette.frame_row_identity",
        "palette.instance_key_row_identity",
        "palette.track_row_identity",
    }
)
_CANONICAL_DESCRIPTOR_IDENTITY_BY_DOMAIN = {
    OBSERVATION_INSTANCE_DOMAIN: (INSTANCE_KEY_MODE, INSTANCE_KEY_ARRAY_REF),
    TRACK_SAMPLE_DOMAIN: ("explicit_array", TRACK_SAMPLE_KEY_ARRAY_REF),
    STIMULUS_STATE_DOMAIN: ("explicit_array", STIMULUS_STATE_KEY_ARRAY_REF),
}


def _identity_dtype_signature(value: Any) -> Any:
    """Normalize JSON dtype metadata without reading an identity payload."""

    candidate = value
    if isinstance(value, list):
        try:
            candidate = [tuple(item) for item in value]
        except TypeError:
            return None
    try:
        dtype = np.dtype(candidate)
    except (TypeError, ValueError):
        return None
    if dtype.fields is None:
        return dtype.str
    return tuple((name, dtype.fields[name][0].str) for name in dtype.names or ())


def _temporal_array_record_issues(
    value: Any,
    *,
    role: str,
    expected_ref: str,
    expected_shape: tuple[int, ...],
    expected_dtype: Any,
    nodes: Mapping[str, MetadataNode],
) -> tuple[MetadataNode | None, list[dict[str, Any]]]:
    """Resolve one digest-bearing temporal array pointer metadata-first."""

    record = _as_mapping(value)
    path = _normalize_archive_ref(str(record.get("ref") or ""))
    node = nodes.get(path or "")
    valid = (
        set(record)
        == {
            "ref",
            "dtype",
            "shape",
            "content_sha256",
            "canonicalization",
        }
        and record.get("ref") == expected_ref
        and path == expected_ref.lstrip("/")
        and node is not None
        and node.node_type == "array"
        and record.get("shape") == list(expected_shape)
        and tuple(node.shape or ()) == expected_shape
        and _identity_dtype_signature(record.get("dtype"))
        == expected_dtype
        and _identity_dtype_signature(node.data_type) == expected_dtype
        and isinstance(record.get("content_sha256"), str)
        and _SHA256_HEX_RE.fullmatch(str(record.get("content_sha256")))
        is not None
        and record.get("canonicalization")
        == ROW_IDENTITY_KEY_CONTENT_CANONICALIZATION
    )
    if valid:
        return node, []
    return node, [
        _issue(
            "TRACK_TIME_LINEAGE_ARRAY_INVALID",
            "error",
            "Temporal-lineage arrays require exact paths, dtypes, shapes, and content digests.",
            role=role,
            expected_ref=expected_ref,
            expected_shape=list(expected_shape),
            expected_dtype=expected_dtype,
            array_record=record,
            metadata_shape=(node.shape if node else None),
            metadata_dtype=(node.data_type if node else None),
        )
    ]


def _track_sample_time_lineage_issues(
    *,
    contract: Any,
    contract_owner_path: str,
    nodes: Mapping[str, MetadataNode],
) -> list[dict[str, Any]]:
    """Validate track time through a sealed immediate-source row authority."""

    if contract.domain != TRACK_SAMPLE_DOMAIN:
        return []
    lineage_ref = contract.time_lineage
    expected_ref = (
        f"/{contract_owner_path}@{TRACK_SAMPLE_TIME_LINEAGE_ATTR}"
        if contract_owner_path
        else f"/@{TRACK_SAMPLE_TIME_LINEAGE_ATTR}"
    )
    if lineage_ref is None:
        return [
            _issue(
                "TRACK_TIME_LINEAGE_REF_INVALID",
                "error",
                "Canonical track identity requires an exact track-sample time-lineage record.",
                contract_owner=contract_owner_path,
            )
        ]
    target_path, attr_name = _canonical_v2_record_target(
        lineage_ref.record_ref
    )
    target = nodes.get(target_path or "")
    if (
        lineage_ref.record_ref != expected_ref
        or target_path != contract_owner_path
        or attr_name != TRACK_SAMPLE_TIME_LINEAGE_ATTR
        or target is None
    ):
        return [
            _issue(
                "TRACK_TIME_LINEAGE_REF_INVALID",
                "error",
                "Track time lineage must be the exact immediate-parent rowset attr.",
                expected_record_ref=expected_ref,
                actual_record_ref=lineage_ref.record_ref,
            )
        ]

    raw = target.attributes.get(TRACK_SAMPLE_TIME_LINEAGE_ATTR)
    record = _as_mapping(raw)
    expected_fields = {
        "schema_id",
        "schema_version",
        "source_row_temporal_authority",
        "recording_id",
        "camera_id",
        "source_total_frames",
        "source_rowset_ref",
        "source_identity_domain",
        "source_identity_mode",
        "source_leading_dimension",
        "leading_dimension",
        "track_sample_key_content_sha256",
        "source_row_index",
        "source_frame_index",
        "interpolation",
        "source_instance_key",
    }
    if set(record) != expected_fields or (
        record.get("schema_id") != TRACK_SAMPLE_TIME_LINEAGE_SCHEMA_ID
        or record.get("schema_version")
        != TRACK_SAMPLE_TIME_LINEAGE_SCHEMA_VERSION
    ):
        codes = [
            _issue(
                "TRACK_TIME_LINEAGE_RECORD_INVALID",
                "error",
                "Track time lineage does not match the exact immediate-source schema.",
                record_fields=sorted(str(name) for name in record),
            )
        ]
        if "acquisition_camera_frame" in record:
            codes.append(
                _issue(
                    "TRACK_TIME_LINEAGE_RETIRED_DIRECT_ACQUISITION",
                    "error",
                    "A track may no longer self-certify acquisition time directly; it must bind its selected immediate source rowset.",
                )
            )
        return codes

    issues: list[dict[str, Any]] = []
    digest = _fingerprint(record)
    if (
        not _exact_json_equal(raw, record)
        or lineage_ref.record_sha256 != digest
        or target.attributes.get(TRACK_SAMPLE_TIME_LINEAGE_DIGEST_ATTR)
        != digest
    ):
        issues.append(
            _issue(
                "TRACK_TIME_LINEAGE_DIGEST_MISMATCH",
                "error",
                "Track time-lineage pointer and stored digest must bind the exact record.",
                expected_sha256=digest,
                pointer_sha256=lineage_ref.record_sha256,
                stored_sha256=target.attributes.get(
                    TRACK_SAMPLE_TIME_LINEAGE_DIGEST_ATTR
                ),
            )
        )
    if (
        record.get("leading_dimension") != contract.leading_dimension
        or record.get("track_sample_key_content_sha256")
        != contract.key_array.content_sha256
    ):
        issues.append(
            _issue(
                "TRACK_TIME_LINEAGE_IDENTITY_MISMATCH",
                "error",
                "Track time lineage must bind the exact track key digest and row count.",
                contract_row_count=contract.leading_dimension,
                lineage_row_count=record.get("leading_dimension"),
            )
        )

    authority_pointer = _as_mapping(
        record.get("source_row_temporal_authority")
    )
    authority_path, authority_attr = _canonical_v2_record_target(
        authority_pointer.get("record_ref")
    )
    authority_node = nodes.get(authority_path or "")
    authority_raw = (
        authority_node.attributes.get(SOURCE_ROW_TEMPORAL_AUTHORITY_ATTR)
        if authority_node is not None
        else None
    )
    authority = _as_mapping(authority_raw)
    authority_fields = {
        "schema_id",
        "schema_version",
        "acquisition_camera_frame",
        "recording_id",
        "camera_id",
        "source_total_frames",
        "source_rowset_ref",
        "source_row_identity",
        "source_identity_domain",
        "source_identity_mode",
        "source_leading_dimension",
        "source_acquisition_frame_index",
        "observation_instance_key",
    }
    self_certified_source = authority_path == contract_owner_path
    authority_valid = (
        set(authority_pointer) == {"record_ref", "record_sha256"}
        and authority_node is not None
        and authority_attr == SOURCE_ROW_TEMPORAL_AUTHORITY_ATTR
        and set(authority) == authority_fields
        and authority.get("schema_id")
        == SOURCE_ROW_TEMPORAL_AUTHORITY_SCHEMA_ID
        and authority.get("schema_version")
        == SOURCE_ROW_TEMPORAL_AUTHORITY_SCHEMA_VERSION
        and authority.get("source_rowset_ref") == f"/{authority_path}"
        and not self_certified_source
    )
    authority_digest = _fingerprint(authority) if authority else None
    if not authority_valid or (
        not _exact_json_equal(authority_raw, authority)
        or authority_pointer.get("record_sha256") != authority_digest
        or (
            authority_node is not None
            and authority_node.attributes.get(
                SOURCE_ROW_TEMPORAL_AUTHORITY_DIGEST_ATTR
            )
            != authority_digest
        )
    ):
        issues.append(
            _issue(
                "TRACK_TIME_LINEAGE_SOURCE_AUTHORITY_INVALID",
                "error",
                "Track lineage must bind one exact digest-sealed source-row temporal authority on the selected source rowset.",
                authority_record_ref=authority_pointer.get("record_ref"),
                authority_fields=sorted(str(name) for name in authority),
            )
        )
    if self_certified_source:
        issues.append(
            _issue(
                "TRACK_TIME_LINEAGE_SELF_CERTIFIED_SOURCE",
                "error",
                "A track rowset cannot act as its own immediate-source temporal authority.",
                track_rowset=contract_owner_path,
                authority_record_ref=authority_pointer.get("record_ref"),
            )
        )

    source_contract = None
    if authority_valid and authority_path is not None:
        source_identity_pointer = _as_mapping(
            authority.get("source_row_identity")
        )
        source_identity_path, source_identity_attr = (
            _canonical_v2_record_target(
                source_identity_pointer.get("record_ref")
            )
        )
        try:
            if (
                set(source_identity_pointer)
                != {"record_ref", "record_sha256"}
                or source_identity_path != authority_path
                or source_identity_attr != ROW_IDENTITY_CONTRACT_ATTR
            ):
                raise RowIdentityContractError(())
            source_contract = load_row_identity_contract_attrs(
                authority_node.attributes
            )
        except RowIdentityContractError as exc:
            issues.append(
                _issue(
                    "TRACK_TIME_LINEAGE_SOURCE_IDENTITY_INVALID",
                    "error",
                    "Source temporal authority must bind the exact selected source row-identity contract.",
                    source_identity_ref=source_identity_pointer.get(
                        "record_ref"
                    ),
                    error=str(exc),
                )
            )
        else:
            if (
                source_identity_pointer.get("record_sha256")
                != source_contract.digest()
                or authority.get("source_identity_domain")
                != source_contract.domain
                or authority.get("source_identity_mode")
                != source_contract.mode
                or authority.get("source_leading_dimension")
                != source_contract.leading_dimension
            ):
                issues.append(
                    _issue(
                        "TRACK_TIME_LINEAGE_SOURCE_IDENTITY_INVALID",
                        "error",
                        "Source authority identity fields differ from its sealed row contract.",
                    )
                )
            source_key_path = (
                f"{authority_path}/{source_contract.key_array.ref}"
            )
            source_key_node = nodes.get(source_key_path)
            try:
                if (
                    source_key_node is None
                    or source_key_node.node_type != "array"
                    or tuple(source_key_node.shape or ())
                    != tuple(source_contract.key_array.shape)
                    or _identity_dtype_signature(source_key_node.data_type)
                    != _identity_dtype_signature(
                        source_contract.key_array.dtype
                    )
                ):
                    raise RowIdentityContractError(())
                load_row_identity_key_attrs(
                    source_key_node.attributes,
                    contract=source_contract,
                )
            except RowIdentityContractError as exc:
                issues.append(
                    _issue(
                        "TRACK_TIME_LINEAGE_SOURCE_IDENTITY_INVALID",
                        "error",
                        "Source temporal authority row identity does not bind its exact persisted key array.",
                        source_key_path=source_key_path,
                        error=str(exc),
                    )
                )

        source_leading = authority.get("source_leading_dimension")
        if type(source_leading) is int and source_leading >= 0:
            source_frame_ref = (
                f"/{authority_path}/{SOURCE_ROW_ACQUISITION_FRAME_INDEX_REF}"
            )
            _source_frame_node, source_frame_issues = (
                _temporal_array_record_issues(
                    authority.get("source_acquisition_frame_index"),
                    role="source_acquisition_frame_index",
                    expected_ref=source_frame_ref,
                    expected_shape=(source_leading,),
                    expected_dtype="<i8",
                    nodes=nodes,
                )
            )
            for item in source_frame_issues:
                item = dict(item)
                item["code"] = "TRACK_TIME_LINEAGE_SOURCE_ARRAY_INVALID"
                issues.append(item)
        else:
            issues.append(
                _issue(
                    "TRACK_TIME_LINEAGE_SOURCE_ARRAY_INVALID",
                    "error",
                    "Source temporal authority has an invalid leading dimension.",
                )
            )

        observation_record = authority.get("observation_instance_key")
        if source_contract is not None and (
            source_contract.domain == OBSERVATION_INSTANCE_DOMAIN
        ):
            instance_ref = (
                f"/{authority_path}/{source_contract.key_array.ref}"
            )
            _instance_node, instance_issues = _temporal_array_record_issues(
                observation_record,
                role="observation_instance_key",
                expected_ref=instance_ref,
                expected_shape=(source_contract.leading_dimension,),
                expected_dtype="<u8",
                nodes=nodes,
            )
            issues.extend(
                {
                    **item,
                    "code": "TRACK_TIME_LINEAGE_SOURCE_ARRAY_INVALID",
                }
                for item in instance_issues
            )
            if (
                not instance_issues
                and _as_mapping(observation_record).get("content_sha256")
                != source_contract.key_array.content_sha256
            ):
                issues.append(
                    _issue(
                        "TRACK_TIME_LINEAGE_SOURCE_IDENTITY_INVALID",
                        "error",
                        "Source observation payload digest differs from its row identity key digest.",
                    )
                )
        elif observation_record is not None:
            issues.append(
                _issue(
                    "TRACK_TIME_LINEAGE_SOURCE_IDENTITY_INVALID",
                    "error",
                    "Only an observation-instance source may bind observation_instance_key lineage.",
                )
            )

        acquisition_pointer = _as_mapping(
            authority.get("acquisition_camera_frame")
        )
        acquisition_path, acquisition_attr = _canonical_v2_record_target(
            acquisition_pointer.get("record_ref")
        )
        acquisition_node = nodes.get(acquisition_path or "")
        try:
            if (
                set(acquisition_pointer)
                != {"record_ref", "record_sha256"}
                or acquisition_node is None
                or acquisition_attr != ACQUISITION_CAMERA_FRAME_ATTR
            ):
                raise PixelFrameAuthorityError("invalid acquisition pointer")
            acquisition_raw = acquisition_node.attributes.get(
                ACQUISITION_CAMERA_FRAME_ATTR
            )
            acquisition = parse_acquisition_camera_frame(acquisition_raw)
        except PixelFrameAuthorityError as exc:
            issues.append(
                _issue(
                    "TRACK_TIME_LINEAGE_ACQUISITION_INVALID",
                    "error",
                    "Source temporal authority acquisition record is invalid.",
                    error=str(exc),
                )
            )
        else:
            _acquisition_target, acquisition_binding_issues = (
                _reference_extent_binding_issues(
                    {
                        **acquisition_pointer,
                        "selector": ACQUISITION_CAMERA_FRAME_ATTR,
                        "width": acquisition.width_px,
                        "height": acquisition.height_px,
                        "units": "px",
                    },
                    role="track_time_lineage.source_acquisition",
                    nodes=nodes,
                )
            )
            if any(
                issue["severity"] in {"error", "critical"}
                for issue in acquisition_binding_issues
            ):
                issues.append(
                    _issue(
                        "TRACK_TIME_LINEAGE_ACQUISITION_INVALID",
                        "error",
                        "Source acquisition record lacks exact sealed import ownership.",
                        validation_issues=acquisition_binding_issues,
                    )
                )
            if (
                not _exact_json_equal(acquisition_raw, acquisition.to_dict())
                or acquisition_pointer.get("record_sha256")
                != acquisition.digest()
                or acquisition_node.attributes.get(
                    ACQUISITION_CAMERA_FRAME_DIGEST_ATTR
                )
                != acquisition.digest()
                or authority.get("recording_id")
                != acquisition.recording_id
                or authority.get("camera_id") != acquisition.camera_id
                or authority.get("source_total_frames")
                != acquisition.source_total_frames
            ):
                issues.append(
                    _issue(
                        "TRACK_TIME_LINEAGE_ACQUISITION_MISMATCH",
                        "error",
                        "Source authority identifiers disagree with acquisition authority.",
                    )
                )

        copied_fields = (
            "recording_id",
            "camera_id",
            "source_total_frames",
            "source_rowset_ref",
            "source_identity_domain",
            "source_identity_mode",
            "source_leading_dimension",
        )
        if any(record.get(name) != authority.get(name) for name in copied_fields):
            issues.append(
                _issue(
                    "TRACK_TIME_LINEAGE_SOURCE_AUTHORITY_MISMATCH",
                    "error",
                    "Track lineage source fields must be exact copies of its source-row authority.",
                    mismatched_fields=[
                        name
                        for name in copied_fields
                        if record.get(name) != authority.get(name)
                    ],
                )
            )

    local_prefix = f"/{contract_owner_path}" if contract_owner_path else ""
    expected_arrays = {
        "source_row_index": (
            TRACK_SAMPLE_SOURCE_ROW_INDEX_REF,
            "<i8",
        ),
        "source_frame_index": (
            TRACK_SAMPLE_SOURCE_FRAME_INDEX_REF,
            "<i8",
        ),
        "interpolation": (
            TRACK_SAMPLE_INTERPOLATION_REF,
            (
                ("left_source_frame_index", "<i8"),
                ("right_source_frame_index", "<i8"),
                ("right_weight", "<f8"),
            ),
        ),
        "source_instance_key": (
            TRACK_SAMPLE_SOURCE_INSTANCE_KEY_REF,
            (("valid", "|b1"), ("instance_key", "<u8")),
        ),
    }
    for role, (leaf, expected_dtype) in expected_arrays.items():
        _array_node, array_issues = _temporal_array_record_issues(
            record.get(role),
            role=role,
            expected_ref=f"{local_prefix}/{leaf}",
            expected_shape=(contract.leading_dimension,),
            expected_dtype=expected_dtype,
            nodes=nodes,
        )
        issues.extend(array_issues)

    if not any(
        issue["severity"] in {"error", "critical"} for issue in issues
    ):
        issues.append(
            _issue(
                "TRACK_TIME_LINEAGE_PAYLOAD_VALIDATION_REQUIRED",
                "warning",
                "Metadata binds exact source rows and arrays, but payload hashes and source_row_index/frame equality require a live validation pass.",
                record_ref=lineage_ref.record_ref,
            )
        )
    return issues


def _row_identity_contract_issues(
    *,
    descriptor: Mapping[str, Any],
    match: DescriptorMatch,
    surface_node: MetadataNode,
    nodes: Mapping[str, MetadataNode],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Validate the canonical row-set contract using metadata only.

    Historical identity arrays remain discoverable as migration evidence, but
    they never make a future-write surface canonical.  The shared
    ``coordinate_identity`` parser is the authority for domains, modes, key
    names, ranks, dtypes, and digest-bearing record schemas.
    """

    row_identity = _as_mapping(descriptor.get("row_identity"))
    mode = row_identity.get("mode")
    if mode == "not_applicable":
        return None, []
    refs = _row_identity_refs(row_identity)
    if len(refs) != 1:
        return None, [
            _issue(
                "ROW_IDENTITY_REF_COUNT_INVALID",
                "error",
                "A coordinate descriptor must resolve through exactly one canonical row-identity key array.",
                mode=mode,
                row_identity_refs=refs,
            )
        ]

    component_name, raw_ref = refs[0]
    owner_node = nodes.get(match.owner_path)
    resolved_key_path = _normalize_archive_ref(
        raw_ref,
        owner_path=match.owner_path,
        owner_is_array=bool(owner_node and owner_node.node_type == "array"),
    )
    key_node = nodes.get(resolved_key_path or "")
    if key_node is None or key_node.node_type != "array":
        return None, [
            _issue(
                "ROW_IDENTITY_REF_UNRESOLVED",
                "error",
                "Canonical row identity does not resolve to a persisted key array.",
                component=component_name,
                array_ref=raw_ref,
                resolved_ref=resolved_key_path,
            )
        ]
    assert resolved_key_path is not None

    contract_owner_path = PurePosixPath(resolved_key_path).parent.as_posix()
    if contract_owner_path == ".":
        contract_owner_path = ""
    contract_owner = nodes.get(contract_owner_path)
    contract_candidates = [
        path
        for path, candidate in nodes.items()
        if ROW_IDENTITY_CONTRACT_ATTR in candidate.attributes
        and (
            path == contract_owner_path
            or contract_owner_path.startswith(f"{path}/")
            or (path == "" and contract_owner_path)
        )
    ]
    contract_candidates.sort()
    if (
        contract_owner is None
        or ROW_IDENTITY_CONTRACT_ATTR not in contract_owner.attributes
    ):
        legacy_schema = key_node.attributes.get("schema_id")
        if legacy_schema in _LEGACY_ROW_IDENTITY_SCHEMA_IDS:
            issues = _legacy_row_identity_integrity_issues(
                descriptor=descriptor,
                match=match,
                surface_node=surface_node,
                nodes=nodes,
            )
            issues.extend(
                [
                    _issue(
                        "CANONICAL_ROW_IDENTITY_CONTRACT_MISSING",
                        "warning",
                        "Historical row identity is present, but the row set lacks the canonical digest-bound identity contract.",
                        row_identity_path=resolved_key_path,
                        legacy_schema_id=legacy_schema,
                    ),
                    _issue(
                        "LEGACY_ROW_IDENTITY_REQUIRES_MIGRATION",
                        "warning",
                        "Historical identity metadata is migration evidence, not a canonical future-write identity.",
                        row_identity_path=resolved_key_path,
                        legacy_schema_id=legacy_schema,
                    ),
                ]
            )
            if PurePosixPath(resolved_key_path).name == "coordinate_row_identity":
                issues.append(
                    _issue(
                        "LEGACY_COORDINATE_ROW_IDENTITY_REQUIRES_MIGRATION",
                        "warning",
                        "coordinate_row_identity is retained only as historical migration evidence; future stimulus rows require stimulus_state_key.",
                        row_identity_path=resolved_key_path,
                    )
                )
            return None, issues
        return None, [
            _issue(
                "CANONICAL_ROW_IDENTITY_CONTRACT_MISSING",
                "error",
                "The key-array owner must persist a canonical row_identity_contract and digest.",
                row_identity_path=resolved_key_path,
                expected_contract_owner=contract_owner_path,
            )
        ]

    issues: list[dict[str, Any]] = []
    if contract_candidates != [contract_owner_path]:
        issues.append(
            _issue(
                "ROW_IDENTITY_CONTRACT_OWNER_AMBIGUOUS",
                "error",
                "Exactly one canonical identity contract must own the key array at its immediate parent row set.",
                expected_contract_owner=contract_owner_path,
                contract_candidates=contract_candidates,
            )
        )

    raw_contract = contract_owner.attributes.get(ROW_IDENTITY_CONTRACT_ATTR)
    validation_issues = validate_row_identity_contract(raw_contract)
    if validation_issues:
        issues.append(
            _issue(
                "ROW_IDENTITY_CONTRACT_INVALID",
                "error",
                "The row-set identity contract fails the shared canonical schema.",
                contract_owner=contract_owner_path,
                validation_issues=[
                    {
                        "code": item.code,
                        "path": item.path,
                        "message": item.message,
                    }
                    for item in validation_issues
                ],
            )
        )
        return None, issues
    try:
        parsed_contract = parse_row_identity_contract(raw_contract)
        loaded_contract = load_row_identity_contract_attrs(contract_owner.attributes)
    except RowIdentityContractError as exc:
        issues.append(
            _issue(
                "ROW_IDENTITY_CONTRACT_DIGEST_INVALID",
                "error",
                "The persisted row-set identity contract or its digest is invalid.",
                contract_owner=contract_owner_path,
                validation_issues=[
                    {
                        "code": item.code,
                        "path": item.path,
                        "message": item.message,
                    }
                    for item in exc.issues
                ],
            )
        )
        return None, issues
    if loaded_contract != parsed_contract:
        issues.append(
            _issue(
                "ROW_IDENTITY_CONTRACT_DIGEST_INVALID",
                "error",
                "Parsed and digest-validated row identity contracts disagree.",
                contract_owner=contract_owner_path,
            )
        )
        return None, issues
    issues.extend(
        _track_sample_time_lineage_issues(
            contract=parsed_contract,
            contract_owner_path=contract_owner_path,
            nodes=nodes,
        )
    )

    contract_key = parsed_contract.key_array
    canonical_key_path = _normalize_archive_ref(
        contract_key.ref,
        owner_path=contract_owner_path,
        owner_is_array=False,
    )
    if canonical_key_path != resolved_key_path:
        issues.append(
            _issue(
                "ROW_IDENTITY_DESCRIPTOR_CONTRACT_MISMATCH",
                "error",
                "The descriptor does not resolve to the exact key array named by its owning identity contract.",
                descriptor_key_path=resolved_key_path,
                contract_key_path=canonical_key_path,
            )
        )

    expected_descriptor = _CANONICAL_DESCRIPTOR_IDENTITY_BY_DOMAIN.get(
        parsed_contract.domain
    )
    if expected_descriptor != (mode, PurePosixPath(resolved_key_path).name):
        issues.append(
            _issue(
                "ROW_IDENTITY_DESCRIPTOR_DOMAIN_MISMATCH",
                "error",
                "Descriptor identity mode/key do not represent the canonical contract domain.",
                contract_domain=parsed_contract.domain,
                contract_mode=parsed_contract.mode,
                descriptor_mode=mode,
                descriptor_key_name=PurePosixPath(resolved_key_path).name,
                expected_descriptor_identity=expected_descriptor,
            )
        )

    try:
        loaded_key = load_row_identity_key_attrs(
            key_node.attributes,
            contract=parsed_contract,
        )
    except RowIdentityContractError as exc:
        issues.append(
            _issue(
                "ROW_IDENTITY_KEY_RECORD_INVALID",
                "error",
                "The key array does not carry the exact digest-bound record required by its row-set identity contract.",
                row_identity_path=resolved_key_path,
                validation_issues=[
                    {
                        "code": item.code,
                        "path": item.path,
                        "message": item.message,
                    }
                    for item in exc.issues
                ],
            )
        )
    else:
        if loaded_key != contract_key:
            issues.append(
                _issue(
                    "ROW_IDENTITY_KEY_RECORD_INVALID",
                    "error",
                    "Digest-validated key metadata disagrees with the owning identity contract.",
                    row_identity_path=resolved_key_path,
                )
            )

    metadata_dtype = _metadata_dtype(key_node.data_type)
    metadata_shape = tuple(key_node.shape or ())
    if (
        metadata_dtype is None
        or metadata_dtype.str != contract_key.dtype
        or metadata_shape != contract_key.shape
    ):
        issues.append(
            _issue(
                "ROW_IDENTITY_KEY_METADATA_MISMATCH",
                "error",
                "Persisted key-array dtype/shape disagree with the canonical identity contract.",
                row_identity_path=resolved_key_path,
                metadata_dtype=(
                    metadata_dtype.str
                    if metadata_dtype is not None
                    else key_node.data_type
                ),
                metadata_shape=list(metadata_shape),
                contract_dtype=contract_key.dtype,
                contract_shape=list(contract_key.shape),
            )
        )

    surface_count, component_counts = _surface_leading_dimension(
        surface_node,
        nodes=nodes,
        excluded_paths={resolved_key_path},
    )
    if (
        surface_count is not None
        and parsed_contract.leading_dimension != surface_count
    ):
        issues.append(
            _issue(
                "ROW_IDENTITY_LENGTH_MISMATCH",
                "error",
                "Canonical identity leading dimension does not match the coordinate surface.",
                row_identity_count=parsed_contract.leading_dimension,
                surface_count=surface_count,
            )
        )
    if component_counts and len(component_counts) > 1:
        issues.append(
            _issue(
                "ROW_IDENTITY_LENGTH_MISMATCH",
                "error",
                "Coordinate component arrays disagree on their leading row dimension.",
                component_counts=component_counts,
            )
        )

    issues.append(
        _issue(
            "ROW_IDENTITY_KEY_PAYLOAD_VALIDATION_REQUIRED",
            "warning",
            "Identity metadata is canonical and digest-bound, but this metadata-only scan does not hash or validate key-array payload values.",
            row_identity_path=resolved_key_path,
            expected_content_sha256=contract_key.content_sha256,
        )
    )
    return parsed_contract.domain, issues


def _historical_descriptor_integrity_issues(
    *,
    match: DescriptorMatch,
    surface_node: MetadataNode,
    nodes: Mapping[str, MetadataNode],
) -> tuple[list[dict[str, Any]], str | None]:
    issues: list[dict[str, Any]] = [
        _issue(
            "HISTORICAL_COORDINATE_DESCRIPTOR_V1_REQUIRES_MIGRATION",
            "warning",
            "Schema-v1 coordinate descriptors remain readable only as explicit historical migration surfaces; future writers require canonical schema v2.",
            descriptor_source=match.source,
        )
    ]
    owner = nodes.get(match.owner_path)
    if owner is None or match.attr_name is None:
        issues.append(
            _issue(
                "COORDINATE_DESCRIPTOR_DIGEST_MISSING",
                "error",
                "Canonical descriptor must be stored as a direct attr with its content digest.",
                descriptor_source=match.source,
            )
        )
    else:
        try:
            load_historical_coordinate_descriptor_v1_attrs(
                owner.attributes,
                attr_name=match.attr_name,
            )
        except CoordinateDescriptorError as exc:
            validation_codes = {item.code for item in exc.issues}
            if "descriptor_digest_mismatch" in validation_codes:
                issue_code = "COORDINATE_DESCRIPTOR_DIGEST_MISMATCH"
            elif "descriptor_digest_missing" in validation_codes:
                issue_code = "COORDINATE_DESCRIPTOR_DIGEST_MISSING"
            else:
                issue_code = "COORDINATE_DESCRIPTOR_INTEGRITY_INVALID"
            issues.append(
                _issue(
                    issue_code,
                    "error",
                    "Descriptor attr or its digest failed closed validation.",
                    descriptor_source=match.source,
                    validation_issues=[
                        {"code": item.code, "path": item.path, "message": item.message}
                        for item in exc.issues
                    ],
                )
            )

    row_identity_domain, row_identity_issues = _row_identity_contract_issues(
        descriptor=match.descriptor,
        match=match,
        surface_node=surface_node,
        nodes=nodes,
    )
    issues.extend(row_identity_issues)

    reference_extent = _as_mapping(match.descriptor.get("reference_extent"))
    issues.extend(
        _reference_authority_issues(
            authority=reference_extent.get("authority"),
            reference_width=reference_extent.get("width"),
            reference_height=reference_extent.get("height"),
            reference_units=reference_extent.get("units"),
            space_id=match.descriptor.get("space_id"),
            nodes=nodes,
            physical_frame=match.descriptor.get("physical_frame"),
            component_units=tuple(match.descriptor.get("component_units") or ()),
            origin=match.descriptor.get("origin"),
            positive_directions=_as_mapping(
                match.descriptor.get("positive_directions")
            ),
        )
    )
    issues.extend(
        _physical_authority_lineage_issues(match.descriptor, nodes=nodes)
    )
    issues.extend(
        _reference_authority_lineage_issues(match.descriptor, nodes=nodes)
    )
    issues.extend(
        _validate_record_refs(match.descriptor, field_name="lineage_refs", nodes=nodes)
    )
    issues.extend(
        _validate_record_refs(match.descriptor, field_name="transform_refs", nodes=nodes)
    )
    issues.extend(_transform_descriptor_issues(match.descriptor, nodes=nodes))
    return issues, row_identity_domain


def _canonical_v2_record_target(ref: Any) -> tuple[str | None, str | None]:
    if not isinstance(ref, str):
        return None, None
    node_ref, separator, attr_name = ref.partition("@")
    normalized = _normalize_archive_ref(node_ref)
    return normalized, attr_name if separator else None


def _canonical_v2_record_digest(
    *,
    record_ref: Any,
    nodes: Mapping[str, MetadataNode],
) -> tuple[str | None, MetadataNode | None, list[dict[str, Any]]]:
    node_path, attr_name = _canonical_v2_record_target(record_ref)
    target = nodes.get(node_path or "")
    if target is None:
        return None, None, [
            _issue(
                "CANONICAL_RECORD_REF_UNRESOLVED",
                "error",
                "Canonical descriptor record_ref does not resolve to persisted archive metadata.",
                record_ref=record_ref,
                resolved_node_path=node_path,
            )
        ]
    if attr_name in (None, "zarr_metadata"):
        return _metadata_node_record_digest(target), target, []
    if attr_name not in target.attributes:
        return None, target, [
            _issue(
                "CANONICAL_RECORD_ATTR_UNRESOLVED",
                "error",
                "Canonical descriptor record_ref names a missing persisted attr.",
                record_ref=record_ref,
                resolved_node_path=node_path,
                attr_name=attr_name,
            )
        ]
    return _fingerprint(target.attributes[attr_name]), target, []


def _canonical_v2_record_value(
    *,
    record_ref: Any,
    nodes: Mapping[str, MetadataNode],
) -> Mapping[str, Any]:
    node_path, attr_name = _canonical_v2_record_target(record_ref)
    target = nodes.get(node_path or "")
    if target is None:
        return {}
    if attr_name not in (None, "zarr_metadata"):
        return _as_mapping(target.attributes.get(attr_name))
    return target.attributes


def _registered_observation_record_integrity_issues(
    *,
    record_ref: Any,
    declared_digest: Any,
    target: MetadataNode | None,
) -> list[dict[str, Any]]:
    """Validate attr/schema/digest registration for a known observation record."""

    node_path, attr_name = _canonical_v2_record_target(record_ref)
    if target is None or attr_name is None:
        return []
    raw_value = target.attributes.get(attr_name)
    raw = _as_mapping(raw_value)
    schema_id = raw.get("schema_id")
    by_schema = _REGISTERED_OBSERVATION_COORDINATE_RECORDS.get(schema_id)
    by_attr = _REGISTERED_OBSERVATION_RECORDS_BY_ATTR.get(attr_name)
    if by_schema is None and by_attr is None:
        return []

    issues: list[dict[str, Any]] = []
    expected_schema_id = schema_id
    expected_rule = by_schema
    if expected_rule is None and by_attr is not None:
        expected_schema_id, expected_rule = by_attr
    assert expected_rule is not None
    expected_attr = str(expected_rule["attribute"])
    expected_version = expected_rule["schema_version"]
    if attr_name != expected_attr:
        issues.append(
            _issue(
                "REGISTERED_COORDINATE_RECORD_ATTR_INVALID",
                "error",
                "A registered observation coordinate schema is stored under the wrong attribute name.",
                record_ref=record_ref,
                schema_id=schema_id,
                expected_attribute=expected_attr,
                actual_attribute=attr_name,
            )
        )
    if (
        type(raw_value) is not dict
        or schema_id != expected_schema_id
        or type(raw.get("schema_version")) is not int
        or raw.get("schema_version") != expected_version
    ):
        issues.append(
            _issue(
                "REGISTERED_COORDINATE_RECORD_SCHEMA_INVALID",
                "error",
                "A registered observation coordinate record must use its exact schema id/version and built-in JSON mapping.",
                record_ref=record_ref,
                expected_schema_id=expected_schema_id,
                expected_schema_version=expected_version,
                actual_schema_id=schema_id,
                actual_schema_version=raw.get("schema_version"),
                value_type=type(raw_value).__name__,
            )
        )
    actual_digest = _fingerprint(raw) if raw else None
    digest_attr = f"{expected_attr}_sha256"
    if (
        actual_digest is None
        or declared_digest != actual_digest
        or target.attributes.get(digest_attr) != actual_digest
    ):
        issues.append(
            _issue(
                "REGISTERED_COORDINATE_RECORD_DIGEST_MISMATCH",
                "error",
                "A registered observation coordinate record requires its exact sibling digest and matching pointer digest.",
                record_ref=record_ref,
                digest_attribute=digest_attr,
                pointer_sha256=declared_digest,
                stored_sha256=target.attributes.get(digest_attr),
                actual_sha256=actual_digest,
            )
        )
    if target.node_type != "group" or target.relative_path != node_path:
        issues.append(
            _issue(
                "REGISTERED_COORDINATE_RECORD_OWNER_INVALID",
                "error",
                "Observation coordinate records must be owned by the exact referenced rowset group.",
                record_ref=record_ref,
                node_type=target.node_type,
                node_path=target.relative_path,
            )
        )
    return issues


def _canonical_v2_bound_record_issues(
    raw_record: Mapping[str, Any],
    *,
    role: str,
    nodes: Mapping[str, MetadataNode],
) -> tuple[MetadataNode | None, list[dict[str, Any]]]:
    record_ref = raw_record.get("record_ref")
    declared_digest = raw_record.get("record_sha256")
    actual_digest, target, issues = _canonical_v2_record_digest(
        record_ref=record_ref,
        nodes=nodes,
    )
    if not issues and declared_digest != actual_digest:
        issues.append(
            _issue(
                "CANONICAL_RECORD_DIGEST_MISMATCH",
                "error",
                "Canonical descriptor record digest does not bind the exact persisted metadata record.",
                role=role,
                record_ref=record_ref,
                declared_sha256=declared_digest,
                actual_sha256=actual_digest,
            )
        )
    issues.extend(
        _registered_observation_record_integrity_issues(
            record_ref=record_ref,
            declared_digest=declared_digest,
            target=target,
        )
    )
    return target, issues


def _metadata_array_extent_pointer(
    node: MetadataNode,
    *,
    units: str,
) -> dict[str, Any] | None:
    """Rebuild the shared array-extent pointer from metadata only."""

    dtype = _metadata_dtype(node.data_type)
    shape = node.shape
    if (
        node.node_type != "array"
        or dtype is None
        or not isinstance(shape, (list, tuple))
        or len(shape) < 2
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in shape
        )
        or int(shape[-2]) <= 0
        or int(shape[-1]) <= 0
    ):
        return None
    normalized_shape = [int(item) for item in shape]
    record = {
        "schema_id": ARRAY_REFERENCE_EXTENT_SCHEMA_ID,
        "schema_version": REFERENCE_EXTENT_SCHEMA_VERSION,
        "array_path": f"/{node.relative_path}",
        "shape": normalized_shape,
        "dtype": dtype.str,
        "selector": "shape[-2:]",
        "width": normalized_shape[-1],
        "height": normalized_shape[-2],
        "units": units,
        "canonicalization": REFERENCE_EXTENT_CANONICALIZATION,
    }
    return {
        "record_ref": f"/{node.relative_path}@zarr_metadata",
        "record_sha256": _fingerprint(record),
        "selector": "shape[-2:]",
        "width": normalized_shape[-1],
        "height": normalized_shape[-2],
        "units": units,
    }


def _metadata_array_storage_identity(
    node: MetadataNode,
) -> dict[str, Any] | None:
    """Rebuild importer storage identity without reading array payloads."""

    dtype = _metadata_dtype(node.data_type)
    shape = node.shape
    storage_metadata = node.storage_metadata
    if (
        node.node_type != "array"
        or dtype is None
        or not isinstance(shape, (list, tuple))
        or not shape
        or type(storage_metadata) is not dict
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item <= 0
            for item in shape
        )
    ):
        return None
    try:
        canonical_storage = _strict_json_loads(_canonical_json(storage_metadata))
    except (TypeError, ValueError):
        return None
    if type(canonical_storage) is not dict:
        return None
    metadata_shape = canonical_storage.get("shape")
    try:
        metadata_dtype = np.dtype(canonical_storage.get("data_type"))
    except (TypeError, ValueError):
        return None
    if (
        type(canonical_storage.get("zarr_format")) is not int
        or canonical_storage.get("zarr_format") != 3
        or canonical_storage.get("node_type") != "array"
        or type(metadata_shape) is not list
        or len(metadata_shape) != len(shape)
        or any(type(item) is not int or item <= 0 for item in metadata_shape)
        or tuple(metadata_shape) != tuple(shape)
        or metadata_dtype != dtype
    ):
        return None
    chunk_shapes = _metadata_array_chunk_shapes(
        node,
        dimensions=len(shape),
    )
    if chunk_shapes is None:
        return None
    logical_chunks, physical_chunks = chunk_shapes
    record = {
        "record_ref": f"/{node.relative_path}@array_storage_metadata",
        "selector": "array_storage_metadata",
        "shape": [int(item) for item in shape],
        "dtype": _json_safe(np.lib.format.dtype_to_descr(dtype)),
        "logical_chunk_shape": list(logical_chunks),
        "physical_chunk_shape": list(physical_chunks),
        "zarr_metadata_without_attrs": canonical_storage,
    }
    return {**record, "record_sha256": _fingerprint(record)}


def _metadata_array_chunk_shapes(
    node: MetadataNode,
    *,
    dimensions: int,
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    """Return exact ``(logical, physical)`` Zarr-v3 chunk shapes.

    ``chunk_grid`` identifies encoded outer storage objects.  With one outer
    ``sharding_indexed`` codec, its configured chunk shape identifies logical
    chunks inside each physical shard; unsharded arrays use the outer shape for
    both.  This mirrors the acquisition writer without opening the array.
    """

    metadata = node.storage_metadata
    if type(metadata) is not dict:
        return None
    chunk_grid = metadata.get("chunk_grid")
    if type(chunk_grid) is not dict or chunk_grid.get("name") != "regular":
        return None
    configuration = chunk_grid.get("configuration")
    if type(configuration) is not dict:
        return None

    def exact_shape(value: Any) -> tuple[int, ...] | None:
        if (
            type(value) is not list
            or len(value) != dimensions
            or any(type(item) is not int or item <= 0 for item in value)
        ):
            return None
        return tuple(value)

    physical = exact_shape(configuration.get("chunk_shape"))
    declared_physical = exact_shape(node.chunk_shape)
    codecs = metadata.get("codecs")
    if physical is None or declared_physical != physical or type(codecs) is not list:
        return None
    sharding_codecs = [
        codec
        for codec in codecs
        if type(codec) is dict and codec.get("name") == "sharding_indexed"
    ]
    if not sharding_codecs:
        return physical, physical
    if len(sharding_codecs) != 1:
        return None
    shard_configuration = sharding_codecs[0].get("configuration")
    if type(shard_configuration) is not dict:
        return None
    logical = exact_shape(shard_configuration.get("chunk_shape"))
    if logical is None or any(
        outer < inner or outer % inner != 0
        for inner, outer in zip(logical, physical, strict=True)
    ):
        return None
    return logical, physical


def _metadata_physical_chunk_indices(
    shape: Sequence[Any],
    storage_identity: Mapping[str, Any] | None,
) -> list[list[int]] | None:
    """Enumerate a bounded outer physical grid, never logical shard chunks.

    ``None`` means that otherwise valid metadata declares more storage objects
    than this metadata-only ruleset will materialize in memory.  Invalid grid
    metadata continues to return ``[]``.  Callers must fail the oversized case
    closed instead of attempting an attacker-controlled Cartesian product.
    """

    physical_chunks = tuple(
        _as_mapping(storage_identity).get("physical_chunk_shape") or ()
    )
    if (
        not shape
        or any(type(size) is not int or size <= 0 for size in shape)
        or len(physical_chunks) != len(shape)
        or any(
            type(chunk) is not int or chunk <= 0
            for chunk in physical_chunks
        )
    ):
        return []
    counts = [
        (size + chunk - 1) // chunk
        for size, chunk in zip(shape, physical_chunks, strict=True)
    ]
    entry_count = math.prod(counts)
    if entry_count > _MAX_METADATA_PHYSICAL_CHUNK_GRID_ENTRIES:
        return None
    ranges = [range(count) for count in counts]
    return [list(index) for index in itertools.product(*ranges)]


def _acquisition_live_metadata_issues(
    *,
    acquisition: Any,
    ownership: Any,
    authority_node: MetadataNode,
    role: str,
    nodes: Mapping[str, MetadataNode],
) -> list[dict[str, Any]]:
    """Cross-validate acquisition records against their actual metadata nodes.

    This deliberately proves only metadata topology and cross-record equality.
    Array values, encoded chunks, and external video contents remain outside a
    metadata-only audit and continue to require a live validation pass.
    """

    issues: list[dict[str, Any]] = []

    def add(code: str, message: str, **evidence: Any) -> None:
        issues.append(_issue(code, "error", message, role=role, **evidence))

    expected_authority_path = (
        f"analysis/acquisition_camera_frames/{acquisition.camera_id}"
    )
    if (
        authority_node.node_type != "group"
        or authority_node.relative_path != expected_authority_path
    ):
        add(
            "ACQUISITION_AUTHORITY_PATH_MISMATCH",
            "Acquisition authority is not persisted at its camera-selected canonical path.",
            expected_path=expected_authority_path,
            actual_path=authority_node.relative_path,
            node_type=authority_node.node_type,
        )

    root = nodes.get(".") or nodes.get("")
    root_metadata = _as_mapping(
        root.attributes.get("source_video_metadata") if root is not None else None
    )
    if (
        root is None
        or root.node_type != "group"
        or root.relative_path not in {"", "."}
        or root.attributes.get("recording_id") != acquisition.recording_id
        or not _exact_json_equal(
            root_metadata,
            dict(acquisition.source_video_metadata),
        )
        or _fingerprint(root_metadata)
        != acquisition.source_video_metadata_sha256
    ):
        add(
            "ACQUISITION_ROOT_METADATA_MISMATCH",
            "Acquisition authority does not equal the live archive-root recording/source-video metadata.",
            root_present=root is not None,
            root_recording_id=(
                root.attributes.get("recording_id") if root is not None else None
            ),
            acquisition_recording_id=acquisition.recording_id,
        )

    if (
        ownership.recording_id != acquisition.recording_id
        or ownership.camera_id != acquisition.camera_id
        or ownership.source_video_metadata_sha256
        != acquisition.source_video_metadata_sha256
    ):
        add(
            "ACQUISITION_IMPORT_OWNERSHIP_IDENTITY_MISMATCH",
            "Acquisition ownership identity and source digest disagree with the camera-frame record.",
            ownership_recording_id=ownership.recording_id,
            ownership_camera_id=ownership.camera_id,
            acquisition_recording_id=acquisition.recording_id,
            acquisition_camera_id=acquisition.camera_id,
        )

    acquisition_materialized = (
        acquisition.frame_array is not None
        or acquisition.frame_index is not None
    )
    ownership_materialized = ownership.mode == "materialized_source_frames_v1"
    if acquisition_materialized != ownership_materialized:
        add(
            "ACQUISITION_MODE_MISMATCH",
            "Acquisition frame storage mode disagrees with its exact import-ownership mode.",
            ownership_mode=ownership.mode,
            acquisition_has_frame_array=acquisition.frame_array is not None,
            acquisition_has_frame_index=acquisition.frame_index is not None,
        )
        return issues

    if not acquisition_materialized:
        if (
            ownership.mode != "external_video_v1"
            or ownership.frame_array is not None
            or ownership.frame_index is not None
            or ownership.import_operation is not None
        ):
            add(
                "ACQUISITION_MODE_MISMATCH",
                "External-video acquisition records must not claim materialized storage lineage.",
                ownership_mode=ownership.mode,
            )
        return issues

    frame_path = "raw_video/images_full"
    index_path = (
        "raw_video/frame_domain_maps/"
        "stored_zarr_frame_to_acquisition_frame"
    )
    frame_node = nodes.get(frame_path)
    index_node = nodes.get(index_path)
    if (
        frame_node is None
        or index_node is None
        or frame_node.relative_path != frame_path
        or index_node.relative_path != index_path
    ):
        add(
            "ACQUISITION_MATERIALIZED_NODE_UNRESOLVED",
            "Materialized acquisition authority requires the exact canonical image and frame-map metadata nodes.",
            frame_path=frame_path,
            frame_present=frame_node is not None,
            index_path=index_path,
            index_present=index_node is not None,
        )
        return issues

    frame_dtype = _metadata_dtype(frame_node.data_type)
    index_dtype = _metadata_dtype(index_node.data_type)
    frame_shape = tuple(frame_node.shape or ())
    index_shape = tuple(index_node.shape or ())
    if (
        frame_node.node_type != "array"
        or frame_dtype != np.dtype("uint8")
        or len(frame_shape) != 3
        or frame_shape[0] != acquisition.frame_count
        or frame_shape[1:] != (acquisition.height_px, acquisition.width_px)
        or index_node.node_type != "array"
        or index_dtype != np.dtype("<i8")
        or index_shape != (acquisition.frame_count,)
        or index_node.attributes.get("source_domain") != "stored_zarr_frame"
        or index_node.attributes.get("target_domain") != "acquisition_frame"
        or index_node.attributes.get("semantics")
        not in {
            "identity_map_zero_based_full_import",
            "explicit_stored_zarr_to_acquisition_frame_v1",
        }
    ):
        add(
            "ACQUISITION_MATERIALIZED_NODE_METADATA_MISMATCH",
            "Materialized image/frame-map metadata disagree with the acquisition dimensions, row count, dtype, or map semantics.",
            frame_shape=list(frame_shape),
            frame_dtype=frame_node.data_type,
            index_shape=list(index_shape),
            index_dtype=index_node.data_type,
            index_attrs=index_node.attributes,
        )

    extent_pointer = _metadata_array_extent_pointer(frame_node, units="px")
    storage_identity = _metadata_array_storage_identity(frame_node)
    raw_frame_index = _as_mapping(acquisition.frame_index)
    ownership_frame_index = _as_mapping(ownership.frame_index)
    expected_index_ref = f"/{index_path}@array_values"
    index_pointer_valid = (
        set(raw_frame_index) == {"record_ref", "record_sha256", "selector"}
        and raw_frame_index.get("record_ref") == expected_index_ref
        and raw_frame_index.get("selector") == "array_values"
        and isinstance(raw_frame_index.get("record_sha256"), str)
        and _SHA256_HEX_RE.fullmatch(raw_frame_index["record_sha256"])
        is not None
    )
    expected_ownership_frame = (
        {**extent_pointer, "storage_identity": storage_identity}
        if extent_pointer is not None and storage_identity is not None
        else None
    )
    expected_domain = {
        "mode": "explicit_stored_zarr_to_acquisition_frame_map_v1",
        "index_record_ref": raw_frame_index.get("record_ref"),
        "index_record_sha256": raw_frame_index.get("record_sha256"),
    }
    if (
        extent_pointer is None
        or storage_identity is None
        or not _exact_json_equal(acquisition.frame_array, extent_pointer)
        or not _exact_json_equal(ownership.frame_array, expected_ownership_frame)
        or not index_pointer_valid
        or not _exact_json_equal(ownership_frame_index, raw_frame_index)
        or not _exact_json_equal(acquisition.frame_domain, expected_domain)
    ):
        add(
            "ACQUISITION_MATERIALIZED_POINTER_MISMATCH",
            "Materialized acquisition records do not bind the exact canonical image metadata and frame-map pointer.",
            expected_frame_array=extent_pointer,
            expected_storage_identity=storage_identity,
            expected_index_ref=expected_index_ref,
        )

    manifest_path = ACQUISITION_MATERIALIZATION_MANIFEST_PATH
    manifest_node = nodes.get(manifest_path)
    if (
        manifest_node is None
        or manifest_node.node_type != "group"
        or manifest_node.relative_path != manifest_path
    ):
        add(
            "ACQUISITION_MATERIALIZATION_MANIFEST_NODE_UNRESOLVED",
            "Materialized acquisition authority requires its exact canonical manifest metadata node.",
            expected_manifest_path=manifest_path,
            manifest_present=manifest_node is not None,
        )
        return issues

    raw_chunk_value = manifest_node.attributes.get(
        ACQUISITION_PHYSICAL_CHUNK_MANIFEST_ATTR
    )
    raw_chunk = _as_mapping(raw_chunk_value)
    chunk_fields = {
        "schema_id",
        "schema_version",
        "producer",
        "array_ref",
        "array_storage",
        "scope",
        "content_evidence_scope",
        "digest_algorithm",
        "entries",
        "entry_count",
        "entries_sha256",
        "entries_canonicalization",
        "canonicalization",
    }
    entries = raw_chunk.get("entries")
    expected_indices = _metadata_physical_chunk_indices(
        frame_shape,
        storage_identity,
    )
    if expected_indices is None:
        physical_chunks = tuple(
            _as_mapping(storage_identity).get("physical_chunk_shape") or ()
        )
        expected_entry_count = math.prod(
            (size + chunk - 1) // chunk
            for size, chunk in zip(
                frame_shape,
                physical_chunks,
                strict=True,
            )
        )
        add(
            "ACQUISITION_PHYSICAL_CHUNK_GRID_EXCEEDS_AUDIT_BOUND",
            "The physical storage-object grid exceeds the bounded metadata-only validation limit.",
            manifest_path=manifest_path,
            expected_entry_count=expected_entry_count,
            maximum_entry_count=_MAX_METADATA_PHYSICAL_CHUNK_GRID_ENTRIES,
        )
        return issues
    entry_storage_keys: set[str] = set()
    storage_metadata = _as_mapping(
        _as_mapping(storage_identity).get("zarr_metadata_without_attrs")
    )
    chunk_key_encoding_value = storage_metadata.get("chunk_key_encoding")
    chunk_key_encoding = _as_mapping(chunk_key_encoding_value)
    chunk_key_configuration_value = chunk_key_encoding.get("configuration")
    chunk_key_configuration = _as_mapping(chunk_key_configuration_value)
    chunk_key_name = chunk_key_encoding.get("name")
    chunk_key_separator = chunk_key_configuration.get("separator")
    chunk_key_encoding_valid = (
        type(chunk_key_encoding_value) is dict
        and type(chunk_key_configuration_value) is dict
        and chunk_key_name in {"default", "v2"}
        and chunk_key_separator in {"/", "."}
    )
    entries_valid = (
        storage_identity is not None
        and bool(expected_indices)
        and chunk_key_encoding_valid
        and type(entries) is list
        and len(entries) == len(expected_indices)
    )
    if entries_valid:
        for entry, expected_indices_row in zip(
            entries,
            expected_indices,
            strict=True,
        ):
            entry_map = _as_mapping(entry)
            storage_key = entry_map.get("storage_key")
            encoded_digest = entry_map.get("encoded_payload_sha256")
            encoded_size = entry_map.get("encoded_size_bytes")
            suffix = str(chunk_key_separator).join(
                str(item) for item in expected_indices_row
            )
            expected_storage_key = (
                f"c{chunk_key_separator}{suffix}"
                if chunk_key_name == "default"
                else suffix
            )
            if not (
                type(entry) is dict
                and set(entry_map)
                == {
                    "chunk_indices",
                    "storage_key",
                    "encoded_payload_sha256",
                    "encoded_size_bytes",
                }
                and type(entry_map.get("chunk_indices")) is list
                and len(entry_map["chunk_indices"])
                == len(expected_indices_row)
                and all(
                    type(item) is int
                    for item in entry_map["chunk_indices"]
                )
                and _exact_json_equal(
                    entry_map["chunk_indices"],
                    expected_indices_row,
                )
                and isinstance(storage_key, str)
                and bool(storage_key)
                and storage_key == storage_key.strip()
                and storage_key not in entry_storage_keys
                and storage_key == expected_storage_key
                and isinstance(encoded_digest, str)
                and _SHA256_HEX_RE.fullmatch(encoded_digest) is not None
                and type(encoded_size) is int
                and encoded_size > 0
            ):
                entries_valid = False
                break
            entry_storage_keys.add(storage_key)
    chunk_digest = _fingerprint(raw_chunk)
    physical_chunk_valid = (
        type(raw_chunk_value) is dict
        and set(raw_chunk) == chunk_fields
        and raw_chunk.get("schema_id")
        == ACQUISITION_PHYSICAL_CHUNK_MANIFEST_SCHEMA_ID
        and type(raw_chunk.get("schema_version")) is int
        and raw_chunk.get("schema_version")
        == ACQUISITION_PHYSICAL_CHUNK_MANIFEST_SCHEMA_VERSION
        and raw_chunk.get("producer") == ACQUISITION_IMPORT_PRODUCER
        and raw_chunk.get("array_ref") == f"/{frame_path}@array_values"
        and _exact_json_equal(
            raw_chunk.get("array_storage"), storage_identity
        )
        and raw_chunk.get("scope") == ACQUISITION_CHUNK_MANIFEST_SCOPE
        and raw_chunk.get("content_evidence_scope")
        == ACQUISITION_CHUNK_CONTENT_EVIDENCE_SCOPE
        and raw_chunk.get("digest_algorithm") == "sha256"
        and entries_valid
        and type(raw_chunk.get("entry_count")) is int
        and raw_chunk.get("entry_count") == len(expected_indices)
        and raw_chunk.get("entries_sha256") == _fingerprint(entries)
        and raw_chunk.get("entries_canonicalization")
        == ACQUISITION_CHUNK_ENTRY_CANONICALIZATION
        and raw_chunk.get("canonicalization")
        == PIXEL_FRAME_AUTHORITY_CANONICALIZATION
        and manifest_node.attributes.get(
            ACQUISITION_PHYSICAL_CHUNK_MANIFEST_DIGEST_ATTR
        )
        == chunk_digest
    )
    if not physical_chunk_valid:
        add(
            "ACQUISITION_PHYSICAL_CHUNK_MANIFEST_INVALID",
            "The canonical physical-chunk manifest does not exactly bind the complete importer entry list and live array storage metadata.",
            manifest_path=manifest_path,
            expected_entry_count=len(expected_indices),
        )
        return issues

    expected_chunk_pointer = {
        "record_ref": (
            f"/{manifest_path}@{ACQUISITION_PHYSICAL_CHUNK_MANIFEST_ATTR}"
        ),
        "record_sha256": chunk_digest,
        "entry_count": len(expected_indices),
        "scope": ACQUISITION_CHUNK_MANIFEST_SCOPE,
        "content_evidence_scope": ACQUISITION_CHUNK_CONTENT_EVIDENCE_SCOPE,
        "metadata_only_verification_scope": (
            ACQUISITION_METADATA_ONLY_VERIFICATION_SCOPE
        ),
    }
    raw_manifest_value = manifest_node.attributes.get(
        ACQUISITION_MATERIALIZATION_MANIFEST_ATTR
    )
    raw_manifest = _as_mapping(raw_manifest_value)
    manifest_fields = {
        "schema_id",
        "schema_version",
        "producer",
        "recording_id",
        "camera_id",
        "materialization_id",
        "write_policy",
        "completed",
        "source_video_metadata_sha256",
        "decode",
        "images_full_storage",
        "frame_map",
        "physical_chunk_manifest",
        "canonicalization",
    }
    if (
        type(raw_manifest_value) is not dict
        or set(raw_manifest) != manifest_fields
        or raw_manifest.get("schema_id")
        != ACQUISITION_MATERIALIZATION_MANIFEST_SCHEMA_ID
        or type(raw_manifest.get("schema_version")) is not int
        or raw_manifest.get("schema_version")
        != ACQUISITION_MATERIALIZATION_MANIFEST_SCHEMA_VERSION
        or raw_manifest.get("producer") != ACQUISITION_IMPORT_PRODUCER
        or raw_manifest.get("write_policy")
        != ACQUISITION_MATERIALIZATION_WRITE_POLICY
        or raw_manifest.get("completed") is not True
        or raw_manifest.get("canonicalization")
        != PIXEL_FRAME_AUTHORITY_CANONICALIZATION
        or manifest_node.attributes.get(
            ACQUISITION_MATERIALIZATION_MANIFEST_DIGEST_ATTR
        )
        != _fingerprint(raw_manifest)
    ):
        add(
            "ACQUISITION_MATERIALIZATION_MANIFEST_INVALID",
            "Materialized acquisition pixels lack the exact canonical immutable importer manifest and digest.",
            manifest_present=bool(raw_manifest),
        )
        return issues

    raw_decode = raw_manifest.get("decode")
    decode = _as_mapping(raw_decode)
    raw_chunk_pointer = raw_manifest.get("physical_chunk_manifest")
    chunk_pointer = _as_mapping(raw_chunk_pointer)
    identity_basis = {
        "recording_id": acquisition.recording_id,
        "camera_id": acquisition.camera_id,
        "source_video_metadata_sha256": (
            acquisition.source_video_metadata_sha256
        ),
        "decode": decode,
        "images_full_storage": storage_identity,
        "frame_map": raw_frame_index,
        "physical_chunk_manifest": expected_chunk_pointer,
    }
    manifest_cross_valid = (
        type(raw_decode) is dict
        and type(raw_chunk_pointer) is dict
        and set(decode)
        == {
            "import_method",
            "import_stage",
            "import_mode",
            "decode_backend",
            "source_decode_surface",
        }
        and all(type(value) is str and value for value in decode.values())
        and decode.get("import_mode") == "full"
        and decode.get("import_stage") in {"complete", "full_resolution"}
        and raw_manifest.get("recording_id") == acquisition.recording_id
        and raw_manifest.get("camera_id") == acquisition.camera_id
        and raw_manifest.get("source_video_metadata_sha256")
        == acquisition.source_video_metadata_sha256
        and _exact_json_equal(
            raw_manifest.get("images_full_storage"), storage_identity
        )
        and _exact_json_equal(raw_manifest.get("frame_map"), raw_frame_index)
        and _exact_json_equal(chunk_pointer, expected_chunk_pointer)
        and raw_manifest.get("materialization_id")
        == _fingerprint(identity_basis)
    )

    source_metadata = dict(acquisition.source_video_metadata)
    expected_operation = {
        "schema_id": "palette.acquisition_materialization_receipt",
        "schema_version": 1,
        "producer": ACQUISITION_IMPORT_PRODUCER,
        "recording_id": acquisition.recording_id,
        "camera_id": acquisition.camera_id,
        "source_locator": source_metadata.get("locator"),
        "source_fingerprint": source_metadata.get("file_fingerprint"),
        "decode": decode,
        "materialization_manifest": {
            "record_ref": (
                f"/{manifest_path}@{ACQUISITION_MATERIALIZATION_MANIFEST_ATTR}"
            ),
            "record_sha256": _fingerprint(raw_manifest),
            "materialization_id": raw_manifest.get("materialization_id"),
            "physical_chunk_manifest": dict(expected_chunk_pointer),
            "verification_scope": (
                ACQUISITION_METADATA_ONLY_VERIFICATION_SCOPE
            ),
        },
        "canonicalization": PIXEL_FRAME_AUTHORITY_CANONICALIZATION,
    }
    if (
        not manifest_cross_valid
        or not _exact_json_equal(
            ownership.import_operation,
            expected_operation,
        )
    ):
        add(
            "ACQUISITION_MATERIALIZATION_MANIFEST_MISMATCH",
            "Importer manifest, receipt, storage metadata, and frame-map lineage do not agree exactly.",
            expected_manifest_ref=expected_operation[
                "materialization_manifest"
            ]["record_ref"],
            expected_chunk_ref=expected_chunk_pointer["record_ref"],
        )
    return issues


def _reference_extent_binding_issues(
    pointer: Mapping[str, Any],
    *,
    role: str,
    nodes: Mapping[str, MetadataNode],
) -> tuple[MetadataNode | None, list[dict[str, Any]]]:
    """Reconstruct one stable coordinate-reference binding from metadata."""

    record_ref = pointer.get("record_ref")
    declared_digest = pointer.get("record_sha256")
    selector = pointer.get("selector")
    width = pointer.get("width")
    height = pointer.get("height")
    units = pointer.get("units")
    node_path, attr_name = _canonical_v2_record_target(record_ref)
    target = nodes.get(node_path or "")
    if target is None:
        return None, [
            _issue(
                "REFERENCE_EXTENT_RECORD_UNRESOLVED",
                "error",
                "Reference extent record_ref does not resolve to archive metadata.",
                role=role,
                record_ref=record_ref,
            )
        ]
    issues: list[dict[str, Any]] = []
    expected_digest: str | None = None
    resolved_width: Any = None
    resolved_height: Any = None
    resolved_units: Any = units

    if attr_name == ACQUISITION_CAMERA_FRAME_ATTR:
        if selector != ACQUISITION_CAMERA_FRAME_ATTR:
            issues.append(
                _issue(
                    "REFERENCE_EXTENT_SELECTOR_INVALID",
                    "error",
                    "Acquisition-camera extents require selector='acquisition_camera_frame'.",
                    role=role,
                    selector=selector,
                )
            )
        raw = target.attributes.get(ACQUISITION_CAMERA_FRAME_ATTR)
        try:
            acquisition = parse_acquisition_camera_frame(raw)
        except PixelFrameAuthorityError as exc:
            issues.append(
                _issue(
                    "ACQUISITION_CAMERA_FRAME_RECORD_INVALID",
                    "error",
                    "Acquisition-camera extent record fails the shared strict parser.",
                    role=role,
                    record_ref=record_ref,
                    error=str(exc),
                )
            )
            return target, issues
        expected_digest = acquisition.digest()
        resolved_width = acquisition.width_px
        resolved_height = acquisition.height_px
        resolved_units = "px"
        if (
            not _exact_json_equal(raw, acquisition.to_dict())
            or target.attributes.get(ACQUISITION_CAMERA_FRAME_DIGEST_ATTR)
            != expected_digest
        ):
            issues.append(
                _issue(
                    "ACQUISITION_CAMERA_FRAME_DIGEST_MISMATCH",
                    "error",
                    "Acquisition-camera extent record is noncanonical or has a stale stored digest.",
                    role=role,
                    record_ref=record_ref,
                )
            )
        ownership_pointer = _as_mapping(acquisition.import_ownership)
        ownership_path, ownership_attr = _canonical_v2_record_target(
            ownership_pointer.get("record_ref")
        )
        ownership_node = nodes.get(ownership_path or "")
        if (
            ownership_node is None
            or ownership_node is not target
            or ownership_attr != ACQUISITION_IMPORT_OWNERSHIP_ATTR
        ):
            issues.append(
                _issue(
                    "ACQUISITION_IMPORT_OWNERSHIP_UNRESOLVED",
                    "error",
                    "Acquisition-camera authority must bind exact importer-owned acquisition evidence on the same node.",
                    role=role,
                    ownership_record_ref=ownership_pointer.get("record_ref"),
                )
            )
        else:
            raw_ownership = ownership_node.attributes.get(
                ACQUISITION_IMPORT_OWNERSHIP_ATTR
            )
            try:
                ownership = parse_acquisition_import_ownership(raw_ownership)
            except PixelFrameAuthorityError as exc:
                issues.append(
                    _issue(
                        "ACQUISITION_IMPORT_OWNERSHIP_INVALID",
                        "error",
                        "Acquisition import ownership fails the shared strict parser.",
                        role=role,
                        error=str(exc),
                    )
                )
            else:
                if (
                    not _exact_json_equal(raw_ownership, ownership.to_dict())
                    or ownership_node.attributes.get(
                        ACQUISITION_IMPORT_OWNERSHIP_DIGEST_ATTR
                    )
                    != ownership.digest()
                    or ownership_pointer.get("record_sha256")
                    != ownership.digest()
                    or ownership.recording_id != acquisition.recording_id
                    or ownership.camera_id != acquisition.camera_id
                ):
                    issues.append(
                        _issue(
                            "ACQUISITION_IMPORT_OWNERSHIP_MISMATCH",
                            "error",
                            "Acquisition camera frame does not bind the exact canonical import ownership record.",
                            role=role,
                        )
                    )
                issues.extend(
                    _acquisition_live_metadata_issues(
                        acquisition=acquisition,
                        ownership=ownership,
                        authority_node=target,
                        role=role,
                        nodes=nodes,
                    )
                )
    elif attr_name == "zarr_metadata":
        dtype = _metadata_dtype(target.data_type)
        if (
            selector != "shape[-2:]"
            or target.node_type != "array"
            or not isinstance(target.shape, (list, tuple))
            or len(target.shape) < 2
            or dtype is None
        ):
            issues.append(
                _issue(
                    "REFERENCE_EXTENT_ARRAY_INVALID",
                    "error",
                    "Array extent authority requires @zarr_metadata, selector shape[-2:], a valid dtype, and rank at least two.",
                    role=role,
                    record_ref=record_ref,
                    selector=selector,
                    shape=target.shape,
                    data_type=target.data_type,
                )
            )
        else:
            resolved_width = int(target.shape[-1])
            resolved_height = int(target.shape[-2])
            record = {
                "schema_id": ARRAY_REFERENCE_EXTENT_SCHEMA_ID,
                "schema_version": REFERENCE_EXTENT_SCHEMA_VERSION,
                "array_path": f"/{target.relative_path}",
                "shape": [int(item) for item in target.shape],
                "dtype": dtype.str,
                "selector": "shape[-2:]",
                "width": resolved_width,
                "height": resolved_height,
                "units": units,
                "canonicalization": REFERENCE_EXTENT_CANONICALIZATION,
            }
            expected_digest = _fingerprint(record)
    elif attr_name is None:
        match = (
            re.fullmatch(r"attrs\[([a-z][a-z0-9_]*),([a-z][a-z0-9_]*)\]", selector)
            if isinstance(selector, str)
            else None
        )
        if match is None or any(
            name not in target.attributes for name in (match.group(1), match.group(2))
        ):
            issues.append(
                _issue(
                    "REFERENCE_EXTENT_ATTRS_INVALID",
                    "error",
                    "Attrs extent authority requires a canonical attrs[width,height] selector with both direct attrs present.",
                    role=role,
                    record_ref=record_ref,
                    selector=selector,
                )
            )
        else:
            resolved_width = target.attributes[match.group(1)]
            resolved_height = target.attributes[match.group(2)]
            record = {
                "schema_id": ATTRS_REFERENCE_EXTENT_SCHEMA_ID,
                "schema_version": REFERENCE_EXTENT_SCHEMA_VERSION,
                "node_path": f"/{target.relative_path}",
                "selector": selector,
                "width": resolved_width,
                "height": resolved_height,
                "units": units,
                "canonicalization": REFERENCE_EXTENT_CANONICALIZATION,
            }
            expected_digest = _fingerprint(record)
    else:
        match = (
            re.fullmatch(r"attrs\[([a-z][a-z0-9_]*),([a-z][a-z0-9_]*)\]", selector)
            if isinstance(selector, str)
            else None
        )
        raw_record = target.attributes.get(attr_name)
        if match is None or not isinstance(raw_record, Mapping):
            issues.append(
                _issue(
                    "REFERENCE_EXTENT_PERSISTED_RECORD_INVALID",
                    "error",
                    "Persisted-record extent authority requires an exact mapping attr and attrs[width,height] selector.",
                    role=role,
                    record_ref=record_ref,
                    selector=selector,
                )
            )
        else:
            width_field, height_field = match.groups()
            expected_digest = _fingerprint(raw_record)
            resolved_width = raw_record.get(width_field)
            resolved_height = raw_record.get(height_field)
            resolved_units = raw_record.get("units")
            matching_digest_attrs = sorted(
                key
                for key, value in target.attributes.items()
                if key != attr_name
                and key.endswith("_sha256")
                and value == expected_digest
            )
            if (
                width_field not in raw_record
                or height_field not in raw_record
                or target.attributes.get(width_field) != resolved_width
                or target.attributes.get(height_field) != resolved_height
                or resolved_units not in {"px", "mm"}
                or len(matching_digest_attrs) != 1
            ):
                issues.append(
                    _issue(
                        "REFERENCE_EXTENT_PERSISTED_RECORD_INVALID",
                        "error",
                        "Persisted-record extent must have one matching stored digest and agree with direct width/height attrs and canonical units.",
                        role=role,
                        record_ref=record_ref,
                        matching_digest_attrs=matching_digest_attrs,
                    )
                )

    if expected_digest is not None and declared_digest != expected_digest:
        issues.append(
            _issue(
                "REFERENCE_EXTENT_RECORD_DIGEST_MISMATCH",
                "error",
                "Reference extent digest does not match the exact stable authority record.",
                role=role,
                record_ref=record_ref,
                declared_sha256=declared_digest,
                actual_sha256=expected_digest,
            )
        )
    if not (
        _extent_values_equal(width, resolved_width)
        and _extent_values_equal(height, resolved_height)
        and units == resolved_units
    ):
        issues.append(
            _issue(
                "REFERENCE_EXTENT_RECORD_VALUE_MISMATCH",
                "error",
                "Reference extent width, height, or units disagree with the selected stable authority.",
                role=role,
                declared_width=width,
                declared_height=height,
                declared_units=units,
                resolved_width=resolved_width,
                resolved_height=resolved_height,
                resolved_units=resolved_units,
            )
        )
    return target, issues


def _array_values_pointer_issues(
    pointer: Any,
    *,
    role: str,
    nodes: Mapping[str, MetadataNode],
    expected_shape: tuple[int | None, ...] | None = None,
) -> tuple[MetadataNode | None, list[dict[str, Any]]]:
    raw = _as_mapping(pointer)
    node_path, attr_name = _canonical_v2_record_target(raw.get("record_ref"))
    target = nodes.get(node_path or "")
    issues: list[dict[str, Any]] = []
    if (
        set(raw) != {"record_ref", "record_sha256", "selector"}
        or attr_name != "array_values"
        or raw.get("selector") != "array_values"
        or target is None
        or target.node_type != "array"
        or not isinstance(raw.get("record_sha256"), str)
        or _SHA256_HEX_RE.fullmatch(str(raw.get("record_sha256"))) is None
    ):
        issues.append(
            _issue(
                "PIXEL_FRAME_ARRAY_LINEAGE_INVALID",
                "error",
                "Pixel-frame array lineage requires an exact digest-bearing @array_values pointer.",
                role=role,
                pointer=raw,
            )
        )
        return target, issues
    if expected_shape is not None:
        actual_shape = tuple(target.shape or ())
        shape_matches = len(actual_shape) == len(expected_shape) and all(
            expected is None or actual == expected
            for actual, expected in zip(actual_shape, expected_shape, strict=True)
        )
        if not shape_matches:
            issues.append(
                _issue(
                    "PIXEL_FRAME_ARRAY_LINEAGE_SHAPE_MISMATCH",
                    "error",
                    "Pixel-frame lineage array shape does not match its typed role.",
                    role=role,
                    expected_shape=expected_shape,
                    actual_shape=actual_shape,
                )
            )
    return target, issues


def _pixel_frame_record_metadata(
    *,
    record_ref: Any,
    record_sha256: Any,
    role: str,
    nodes: Mapping[str, MetadataNode],
    seen: frozenset[str] = frozenset(),
) -> tuple[Any | None, MetadataNode | None, list[dict[str, Any]]]:
    """Parse a pixel frame and validate its kind-specific persisted lineage."""

    node_path, attr_name = _canonical_v2_record_target(record_ref)
    target = nodes.get(node_path or "")
    if target is None or attr_name != PIXEL_FRAME_AUTHORITY_ATTR:
        return None, target, [
            _issue(
                "PIXEL_FRAME_AUTHORITY_REF_INVALID",
                "error",
                "Pixel-frame references must resolve to an exact @pixel_frame_authority attr.",
                role=role,
                record_ref=record_ref,
            )
        ]
    normalized_ref = str(record_ref)
    if normalized_ref in seen:
        return None, target, [
            _issue(
                "PIXEL_FRAME_LINEAGE_CYCLE",
                "error",
                "Pixel-frame lineage contains a cycle.",
                role=role,
                record_ref=record_ref,
            )
        ]
    raw = target.attributes.get(PIXEL_FRAME_AUTHORITY_ATTR)
    try:
        frame = parse_pixel_frame_record(raw)
    except PixelFrameAuthorityError as exc:
        return None, target, [
            _issue(
                "PIXEL_FRAME_AUTHORITY_RECORD_INVALID",
                "error",
                "Referenced pixel-frame authority fails the shared strict schema.",
                role=role,
                record_ref=record_ref,
                error=str(exc),
            )
        ]
    issues: list[dict[str, Any]] = []
    if not _exact_json_equal(raw, frame.to_dict()):
        issues.append(
            _issue(
                "PIXEL_FRAME_AUTHORITY_RECORD_NONCANONICAL",
                "error",
                "Persisted pixel-frame authority is not its exact canonical mapping.",
                role=role,
                record_ref=record_ref,
            )
        )
    if (
        target.attributes.get(PIXEL_FRAME_AUTHORITY_DIGEST_ATTR) != frame.digest()
        or record_sha256 != frame.digest()
    ):
        issues.append(
            _issue(
                "PIXEL_FRAME_AUTHORITY_DIGEST_MISMATCH",
                "error",
                "Pixel-frame pointer and stored digest must bind the exact parsed frame.",
                role=role,
                record_ref=record_ref,
                pointer_sha256=record_sha256,
                stored_sha256=target.attributes.get(
                    PIXEL_FRAME_AUTHORITY_DIGEST_ATTR
                ),
                actual_sha256=frame.digest(),
            )
        )
    _extent_target, extent_issues = _reference_extent_binding_issues(
        frame.reference_extent,
        role=f"{role}.reference_extent",
        nodes=nodes,
    )
    issues.extend(extent_issues)

    lineage = _as_mapping(frame.lineage)
    next_seen = seen | {normalized_ref}
    if frame.kind == SOURCE_CAMERA_FRAME_KIND:
        expected_fields = {
            "acquisition_camera_frame",
            "acquisition_import_ownership",
            "recording_id",
            "camera_id",
        }
        acquisition_pointer = _as_mapping(
            lineage.get("acquisition_camera_frame")
        )
        ownership_pointer = _as_mapping(
            lineage.get("acquisition_import_ownership")
        )
        if set(lineage) != expected_fields or set(acquisition_pointer) != {
            "record_ref",
            "record_sha256",
            "selector",
        } or set(ownership_pointer) != {"record_ref", "record_sha256"}:
            issues.append(
                _issue(
                    "PIXEL_FRAME_LINEAGE_SCHEMA_INVALID",
                    "error",
                    "Source-camera frames require exact acquisition-camera lineage fields.",
                    role=role,
                    frame_kind=frame.kind,
                    lineage=lineage,
                )
            )
        else:
            acquisition_target, acquisition_issues = (
                _reference_extent_binding_issues(
                    {
                        **acquisition_pointer,
                        "width": frame.reference_extent["width"],
                        "height": frame.reference_extent["height"],
                        "units": "px",
                    },
                    role=f"{role}.acquisition_camera_frame",
                    nodes=nodes,
                )
            )
            issues.extend(acquisition_issues)
            acquisition_raw = (
                acquisition_target.attributes.get(ACQUISITION_CAMERA_FRAME_ATTR)
                if acquisition_target is not None
                else None
            )
            try:
                acquisition = parse_acquisition_camera_frame(acquisition_raw)
            except PixelFrameAuthorityError:
                acquisition = None
            if acquisition is not None and (
                lineage.get("recording_id") != acquisition.recording_id
                or lineage.get("camera_id") != acquisition.camera_id
                or ownership_pointer != acquisition.import_ownership
                or dict(frame.reference_extent)
                != {
                    "record_ref": acquisition_pointer.get("record_ref"),
                    "record_sha256": acquisition_pointer.get("record_sha256"),
                    "selector": acquisition_pointer.get("selector"),
                    "width": acquisition.width_px,
                    "height": acquisition.height_px,
                    "units": "px",
                }
            ):
                issues.append(
                    _issue(
                        "PIXEL_FRAME_SOURCE_ACQUISITION_MISMATCH",
                        "error",
                        "Source-camera frame extent and identifiers must equal its exact acquisition authority.",
                        role=role,
                    )
                )
    elif frame.kind == SELECTED_CANVAS_FRAME_KIND:
        expected_fields = {
            "selected_calibration_manifest",
            "source_display",
            "stimulus_run",
            "camera_id",
            "external_h5_freshness",
        }
        manifest_pointer = _as_mapping(
            lineage.get("selected_calibration_manifest")
        )
        display_pointer = _as_mapping(lineage.get("source_display"))
        if (
            set(lineage) != expected_fields
            or set(manifest_pointer) != {"record_ref", "record_sha256"}
            or set(display_pointer) != {"record_ref", "record_sha256"}
        ):
            issues.append(
                _issue(
                    "PIXEL_FRAME_LINEAGE_SCHEMA_INVALID",
                    "error",
                    "Selected-canvas frames require exact selected-display lineage fields.",
                    role=role,
                    frame_kind=frame.kind,
                    lineage=lineage,
                )
            )
        else:
            manifest_path, manifest_attr = _canonical_v2_record_target(
                manifest_pointer.get("record_ref")
            )
            manifest_node = nodes.get(manifest_path or "")
            display_path, display_attr = _canonical_v2_record_target(
                display_pointer.get("record_ref")
            )
            display_node = nodes.get(display_path or "")
            try:
                if (
                    manifest_node is None
                    or manifest_attr != SELECTED_CALIBRATION_MANIFEST_ATTR
                ):
                    raise SelectedCalibrationError(
                        "selected-calibration manifest pointer is unresolved"
                    )
                manifest = load_selected_calibration_manifest_attrs(
                    manifest_node.attributes
                )
                if (
                    display_node is None
                    or display_attr != SOURCE_DISPLAY_EVIDENCE_ATTR
                ):
                    raise SelectedCalibrationError(
                        "selected-display evidence pointer is unresolved"
                    )
                display = load_selected_display_evidence_attrs(
                    display_node.attributes
                )
            except SelectedCalibrationError as exc:
                issues.append(
                    _issue(
                        "PIXEL_FRAME_SELECTED_DISPLAY_INVALID",
                        "error",
                        "Selected-canvas source display evidence fails the shared strict parser.",
                        role=role,
                        error=str(exc),
                    )
                )
            else:
                if (
                    manifest_pointer.get("record_sha256") != manifest.digest()
                    or display_pointer.get("record_sha256") != display.digest()
                    or manifest.source_display != display
                    or lineage.get("stimulus_run") != manifest.stimulus_run
                    or lineage.get("camera_id") != manifest.camera_id
                    or lineage.get("external_h5_freshness")
                    != "persisted_import_snapshot"
                    or frame.reference_extent.get("width")
                    != manifest.display_snapshot.width_px
                    or frame.reference_extent.get("height")
                    != manifest.display_snapshot.height_px
                ):
                    issues.append(
                        _issue(
                            "PIXEL_FRAME_SELECTED_DISPLAY_MISMATCH",
                            "error",
                            "Selected-canvas frame does not bind the exact selected display and dimensions.",
                            role=role,
                        )
                    )
    elif frame.kind == ARENA_RELATIVE_CANVAS_FRAME_KIND:
        expected_fields = {
            "arena_geometry",
            "layout",
            "origin",
            "origin_in_selected_canvas_px",
            "selected_canvas_frame",
        }
        if set(lineage) != expected_fields:
            issues.append(
                _issue(
                    "PIXEL_FRAME_LINEAGE_SCHEMA_INVALID",
                    "error",
                    "Arena-relative frames require exact placement and selected-canvas lineage fields.",
                    role=role,
                    frame_kind=frame.kind,
                    lineage=lineage,
                )
            )
        else:
            geometry, geometry_issues = _array_values_pointer_issues(
                lineage.get("arena_geometry"),
                role=f"{role}.arena_geometry",
                nodes=nodes,
                expected_shape=(4,),
            )
            issues.extend(geometry_issues)
            selected_pointer = _as_mapping(lineage.get("selected_canvas_frame"))
            selected, _selected_target, selected_issues = _pixel_frame_record_metadata(
                record_ref=selected_pointer.get("record_ref"),
                record_sha256=selected_pointer.get("record_sha256"),
                role=f"{role}.selected_canvas_frame",
                nodes=nodes,
                seen=next_seen,
            )
            issues.extend(selected_issues)
            origin = _as_mapping(lineage.get("origin_in_selected_canvas_px"))
            geometry_dtype = _metadata_dtype(geometry.data_type) if geometry else None
            if (
                set(selected_pointer) != {"record_ref", "record_sha256"}
                or lineage.get("layout") != "selected_canvas_xywh_px"
                or lineage.get("origin") != "arena_top_left"
                or set(origin) != {"x", "y"}
                or any(type(origin.get(axis)) is not int or origin.get(axis) < 0 for axis in ("x", "y"))
                or geometry_dtype is None
                or geometry_dtype.kind not in "iu"
                or selected is None
                or selected.kind != SELECTED_CANVAS_FRAME_KIND
                or selected.pixel_convention != frame.pixel_convention
            ):
                issues.append(
                    _issue(
                        "PIXEL_FRAME_ARENA_LINEAGE_MISMATCH",
                        "error",
                        "Arena-relative frame placement, selected canvas, axes, or conventions are inconsistent.",
                        role=role,
                    )
                )
    elif frame.kind == ROI_FRAME_KIND:
        expected_fields = {
            "crop_placement_ownership",
            "crop_placement",
            "layout",
            "window_policy",
            "row_identity",
            "source_camera_frame",
            "camera_id",
        }
        if set(lineage) != expected_fields:
            issues.append(
                _issue(
                    "PIXEL_FRAME_LINEAGE_SCHEMA_INVALID",
                    "error",
                    "ROI frames require exact crop placement, observation identity, and source-camera lineage fields.",
                    role=role,
                    frame_kind=frame.kind,
                    lineage=lineage,
                )
            )
        else:
            ownership_pointer = _as_mapping(
                lineage.get("crop_placement_ownership")
            )
            ownership_path, ownership_attr = _canonical_v2_record_target(
                ownership_pointer.get("record_ref")
            )
            ownership_node = nodes.get(ownership_path or "")
            ownership = None
            if (
                ownership_node is not None
                and ownership_attr == CROP_PLACEMENT_OWNERSHIP_ATTR
            ):
                raw_ownership = ownership_node.attributes.get(
                    CROP_PLACEMENT_OWNERSHIP_ATTR
                )
                try:
                    ownership = parse_crop_placement_ownership(raw_ownership)
                except PixelFrameAuthorityError:
                    ownership = None
                else:
                    if (
                        not _exact_json_equal(raw_ownership, ownership.to_dict())
                        or ownership_node.attributes.get(
                            CROP_PLACEMENT_OWNERSHIP_DIGEST_ATTR
                        )
                        != ownership.digest()
                        or ownership_pointer.get("record_sha256")
                        != ownership.digest()
                    ):
                        ownership = None
            placement, placement_issues = _array_values_pointer_issues(
                lineage.get("crop_placement"),
                role=f"{role}.crop_placement",
                nodes=nodes,
                expected_shape=(None, 4),
            )
            issues.extend(placement_issues)
            identity = _as_mapping(lineage.get("row_identity"))
            identity_path, identity_attr = _canonical_v2_record_target(
                identity.get("record_ref")
            )
            identity_node = nodes.get(identity_path or "")
            contract = None
            if identity_node is not None and identity_attr == ROW_IDENTITY_CONTRACT_ATTR:
                try:
                    contract = load_row_identity_contract_attrs(identity_node.attributes)
                except RowIdentityContractError:
                    contract = None
            source_pointer = _as_mapping(lineage.get("source_camera_frame"))
            source, _source_target, source_issues = _pixel_frame_record_metadata(
                record_ref=source_pointer.get("record_ref"),
                record_sha256=source_pointer.get("record_sha256"),
                role=f"{role}.source_camera_frame",
                nodes=nodes,
                seen=next_seen,
            )
            issues.extend(source_issues)
            placement_dtype = _metadata_dtype(placement.data_type) if placement else None
            if (
                set(ownership_pointer) != {"record_ref", "record_sha256"}
                or ownership is None
                or ownership.crop_placement != lineage.get("crop_placement")
                or ownership.row_identity != lineage.get("row_identity")
                or ownership.source_camera_frame
                != lineage.get("source_camera_frame")
                or ownership.camera_id != lineage.get("camera_id")
                or lineage.get("layout") != "xywh"
                or lineage.get("window_policy")
                != CROP_PLACEMENT_WINDOW_POLICY
                or set(identity) != {
                    "record_ref",
                    "record_sha256",
                    "leading_dimension",
                }
                or contract is None
                or contract.domain != OBSERVATION_INSTANCE_DOMAIN
                or contract.mode != INSTANCE_KEY_MODE
                or contract.key_array.ref != INSTANCE_KEY_ARRAY_REF
                or identity.get("record_sha256") != contract.digest()
                or identity.get("leading_dimension") != contract.leading_dimension
                or placement is None
                or not isinstance(placement.shape, (list, tuple))
                or placement.shape[0] != contract.leading_dimension
                or placement_dtype is None
                or placement_dtype.kind not in "iuf"
                or source is None
                or source.kind != SOURCE_CAMERA_FRAME_KIND
                or lineage.get("camera_id") != source.lineage.get("camera_id")
            ):
                issues.append(
                    _issue(
                        "PIXEL_FRAME_ROI_LINEAGE_MISMATCH",
                        "error",
                        "ROI frame does not bind exact crop placement, observation identity, and source-camera authority.",
                        role=role,
                    )
                )
    elif frame.kind == MODEL_INPUT_FRAME_KIND:
        expected_fields = {"preprocessing_payload", "preprocessing", "roi_frame"}
        if set(lineage) != expected_fields:
            issues.append(
                _issue(
                    "PIXEL_FRAME_LINEAGE_SCHEMA_INVALID",
                    "error",
                    "Model-input frames require exact preprocessing payload, semantics, and ROI lineage fields.",
                    role=role,
                    frame_kind=frame.kind,
                    lineage=lineage,
                )
            )
        else:
            payload, payload_issues = _array_values_pointer_issues(
                lineage.get("preprocessing_payload"),
                role=f"{role}.preprocessing_payload",
                nodes=nodes,
                expected_shape=(3, 3),
            )
            issues.extend(payload_issues)
            preprocessing = _as_mapping(lineage.get("preprocessing"))
            expected_pre_fields = {
                "name",
                "native_height",
                "native_width",
                "model_height",
                "model_width",
                "pad_top",
                "pad_bottom",
                "pad_left",
                "pad_right",
            }
            roi_pointer = _as_mapping(lineage.get("roi_frame"))
            roi, _roi_target, roi_issues = _pixel_frame_record_metadata(
                record_ref=roi_pointer.get("record_ref"),
                record_sha256=roi_pointer.get("record_sha256"),
                role=f"{role}.roi_frame",
                nodes=nodes,
                seen=next_seen,
            )
            issues.extend(roi_issues)
            integer_fields = expected_pre_fields - {"name"}
            payload_dtype = _metadata_dtype(payload.data_type) if payload else None
            if (
                set(preprocessing) != expected_pre_fields
                or preprocessing.get("name") not in {"identity", "pad_to_size"}
                or any(
                    type(preprocessing.get(name)) is not int
                    or preprocessing.get(name) < 0
                    for name in integer_fields
                )
                or preprocessing.get("model_height")
                != preprocessing.get("pad_top")
                + preprocessing.get("native_height")
                + preprocessing.get("pad_bottom")
                or preprocessing.get("model_width")
                != preprocessing.get("pad_left")
                + preprocessing.get("native_width")
                + preprocessing.get("pad_right")
                or payload_dtype != np.dtype("float64")
                or roi is None
                or roi.kind != ROI_FRAME_KIND
                or frame.reference_extent.get("width")
                != preprocessing.get("model_width")
                or frame.reference_extent.get("height")
                != preprocessing.get("model_height")
                or roi.reference_extent.get("width")
                != preprocessing.get("native_width")
                or roi.reference_extent.get("height")
                != preprocessing.get("native_height")
                or roi.pixel_convention != frame.pixel_convention
            ):
                issues.append(
                    _issue(
                        "PIXEL_FRAME_MODEL_INPUT_LINEAGE_MISMATCH",
                        "error",
                        "Model-input frame preprocessing metadata or ROI binding is inconsistent.",
                        role=role,
                    )
                )
    elif frame.kind in {
        SOURCE_CAMERA_NORMALIZED_FRAME_KIND,
        DETECTOR_NORMALIZED_FRAME_KIND,
    }:
        expected_fields = {
            "pixel_frame",
            "normalization_formula",
            "reference_width_px",
            "reference_height_px",
            "target_pixel_convention",
        }
        pixel_pointer = _as_mapping(lineage.get("pixel_frame"))
        pixel, _pixel_target, pixel_issues = _pixel_frame_record_metadata(
            record_ref=pixel_pointer.get("record_ref"),
            record_sha256=pixel_pointer.get("record_sha256"),
            role=f"{role}.pixel_frame",
            nodes=nodes,
            seen=next_seen,
        )
        issues.extend(pixel_issues)
        expected_pixel_kind = (
            SOURCE_CAMERA_FRAME_KIND
            if frame.kind == SOURCE_CAMERA_NORMALIZED_FRAME_KIND
            else MODEL_INPUT_FRAME_KIND
        )
        expected_formula = (
            NORMALIZED_TO_PIXEL_CENTER_INDEX_V1
            if pixel is not None and pixel.pixel_convention == "pixel_center"
            else NORMALIZED_TO_PIXEL_EDGE_EXTENT_V1
        )
        if (
            set(lineage) != expected_fields
            or set(pixel_pointer) != {"record_ref", "record_sha256"}
            or pixel is None
            or pixel.kind != expected_pixel_kind
            or frame.coordinate_units != "normalized"
            or frame.pixel_convention != "continuous"
            or lineage.get("target_pixel_convention")
            != pixel.pixel_convention
            or lineage.get("normalization_formula") != expected_formula
            or lineage.get("reference_width_px")
            != pixel.reference_extent.get("width")
            or lineage.get("reference_height_px")
            != pixel.reference_extent.get("height")
            or frame.reference_extent != pixel.reference_extent
        ):
            issues.append(
                _issue(
                    "PIXEL_FRAME_NORMALIZED_LINEAGE_MISMATCH",
                    "error",
                    "Normalized frame must bind the exact pixel frame, reference dimensions, and controlled sampling formula.",
                    role=role,
                )
            )
    return frame, target, issues


def _canonical_v2_pixel_frame_issues(
    descriptor: Mapping[str, Any],
    *,
    nodes: Mapping[str, MetadataNode],
) -> list[dict[str, Any]]:
    """Validate the exact typed pixel-frame record required by v2 profiles."""

    frame_pointer = _as_mapping(descriptor.get("frame_record"))
    if frame_pointer.get("kind") != PIXEL_FRAME_AUTHORITY_RECORD_KIND:
        return []
    extent = _as_mapping(descriptor.get("reference_extent"))
    authority = _as_mapping(extent.get("authority"))
    issues: list[dict[str, Any]] = []
    if (
        authority.get("selector") != "record"
        or frame_pointer.get("record_ref") != authority.get("record_ref")
        or frame_pointer.get("record_sha256")
        != authority.get("record_sha256")
    ):
        issues.append(
            _issue(
                "PIXEL_FRAME_REFERENCE_AUTHORITY_MISMATCH",
                "error",
                "Canonical pixel/normalized coordinates must use the exact typed frame record as their reference authority with selector='record'.",
                frame_record=frame_pointer,
                reference_authority=authority,
            )
        )
        return issues
    frame, _target, frame_issues = _pixel_frame_record_metadata(
        record_ref=frame_pointer.get("record_ref"),
        record_sha256=frame_pointer.get("record_sha256"),
        role="descriptor.frame_record",
        nodes=nodes,
    )
    issues.extend(frame_issues)
    if frame is None:
        return issues

    expected_frame_space = descriptor.get("space_id")
    if frame.space_id != expected_frame_space:
        issues.append(
            _issue(
                "PIXEL_FRAME_AUTHORITY_SPACE_MISMATCH",
                "error",
                "Descriptor space is not backed by the corresponding typed pixel frame.",
                descriptor_space_id=descriptor.get("space_id"),
                expected_frame_space_id=expected_frame_space,
                frame_space_id=frame.space_id,
            )
        )
    if (
        not _extent_values_equal(extent.get("width"), frame.reference_extent.get("width"))
        or not _extent_values_equal(
            extent.get("height"), frame.reference_extent.get("height")
        )
    ):
        issues.append(
            _issue(
                "PIXEL_FRAME_AUTHORITY_EXTENT_MISMATCH",
                "error",
                "Descriptor reference dimensions disagree with its typed pixel frame.",
                descriptor_width=extent.get("width"),
                descriptor_height=extent.get("height"),
                frame_width=frame.reference_extent.get("width"),
                frame_height=frame.reference_extent.get("height"),
            )
        )
    if descriptor.get("pixel_convention") != frame.pixel_convention:
        issues.append(
            _issue(
                "PIXEL_FRAME_AUTHORITY_CONVENTION_MISMATCH",
                "error",
                "Descriptor pixel convention disagrees with its typed pixel frame.",
                descriptor_pixel_convention=descriptor.get("pixel_convention"),
                frame_pixel_convention=frame.pixel_convention,
            )
        )

    issues.append(
        _issue(
            "PIXEL_FRAME_AUTHORITY_LIVE_VALIDATION_REQUIRED",
            "warning",
            "Pixel-frame schema and metadata bindings are valid, but the metadata-only scanner cannot revalidate the producer-sealed authority and payload lineage.",
            record_ref=frame_pointer.get("record_ref"),
        )
    )
    return issues


def _canonical_v2_reference_authority_issues(
    descriptor: Mapping[str, Any],
    *,
    nodes: Mapping[str, MetadataNode],
) -> list[dict[str, Any]]:
    extent = _as_mapping(descriptor.get("reference_extent"))
    authority = _as_mapping(extent.get("authority"))
    selector = authority.get("selector")
    if selector == "record":
        target, issues = _canonical_v2_bound_record_issues(
            authority,
            role="reference_extent.authority",
            nodes=nodes,
        )
    else:
        target, issues = _reference_extent_binding_issues(
            {
                **authority,
                "width": extent.get("width"),
                "height": extent.get("height"),
                "units": extent.get("units"),
            },
            role="reference_extent.authority",
            nodes=nodes,
        )
    if target is None:
        return issues

    if descriptor.get("space_id") == "roi_local_px" and selector == "shape[-2:]":
        if target.attributes.get("coordinate_authority_role") != "crop_roi_raster":
            issues.append(
                _issue(
                    "REFERENCE_AUTHORITY_ROLE_INVALID",
                    "error",
                    "ROI shape authority must carry the controlled crop_roi_raster role.",
                    target_path=target.relative_path,
                    actual_role=target.attributes.get("coordinate_authority_role"),
                )
            )
    return issues


def _canonical_v2_row_identity_issues(
    *,
    descriptor: Mapping[str, Any],
    match: DescriptorMatch,
    surface_node: MetadataNode,
    nodes: Mapping[str, MetadataNode],
) -> tuple[Any | None, str | None, list[dict[str, Any]]]:
    raw_identity = _as_mapping(descriptor.get("row_identity"))
    record_ref = raw_identity.get("record_ref")
    contract_path, attr_name = _canonical_v2_record_target(record_ref)
    issues: list[dict[str, Any]] = []
    if attr_name != ROW_IDENTITY_CONTRACT_ATTR:
        return None, None, [
            _issue(
                "ROW_IDENTITY_RECORD_REF_INVALID",
                "error",
                "Canonical descriptor row identity must name @row_identity_contract.",
                record_ref=record_ref,
            )
        ]
    contract_node = nodes.get(contract_path or "")
    if contract_node is None:
        return None, None, [
            _issue(
                "ROW_IDENTITY_CONTRACT_UNRESOLVED",
                "error",
                "Canonical descriptor row-identity record does not resolve.",
                record_ref=record_ref,
                contract_path=contract_path,
            )
        ]
    validation_issues = validate_row_identity_contract(
        contract_node.attributes.get(ROW_IDENTITY_CONTRACT_ATTR)
    )
    if validation_issues:
        return None, None, [
            _issue(
                "ROW_IDENTITY_CONTRACT_INVALID",
                "error",
                "Canonical descriptor row-identity record fails the shared schema.",
                contract_path=contract_path,
                validation_issues=[
                    {
                        "code": item.code,
                        "path": item.path,
                        "message": item.message,
                    }
                    for item in validation_issues
                ],
            )
        ]
    try:
        parsed_contract = parse_row_identity_contract(
            contract_node.attributes[ROW_IDENTITY_CONTRACT_ATTR]
        )
        loaded_contract = load_row_identity_contract_attrs(contract_node.attributes)
    except RowIdentityContractError as exc:
        return None, None, [
            _issue(
                "ROW_IDENTITY_CONTRACT_DIGEST_INVALID",
                "error",
                "Canonical row-identity contract digest is missing or invalid.",
                contract_path=contract_path,
                validation_issues=[
                    {
                        "code": item.code,
                        "path": item.path,
                        "message": item.message,
                    }
                    for item in exc.issues
                ],
            )
        ]
    if loaded_contract != parsed_contract:
        issues.append(
            _issue(
                "ROW_IDENTITY_CONTRACT_DIGEST_INVALID",
                "error",
                "Parsed and digest-validated row identity contracts disagree.",
                contract_path=contract_path,
            )
        )
    if raw_identity.get("record_sha256") != parsed_contract.digest():
        issues.append(
            _issue(
                "ROW_IDENTITY_RECORD_DIGEST_MISMATCH",
                "error",
                "Descriptor row-identity digest does not bind the exact external contract.",
                record_ref=record_ref,
                declared_sha256=raw_identity.get("record_sha256"),
                actual_sha256=parsed_contract.digest(),
            )
        )
    issues.extend(
        _track_sample_time_lineage_issues(
            contract=parsed_contract,
            contract_owner_path=contract_path or "",
            nodes=nodes,
        )
    )

    key_path = f"{contract_path}/{parsed_contract.key_array.ref}"
    key_node = nodes.get(key_path)
    if key_node is None or key_node.node_type != "array":
        issues.append(
            _issue(
                "ROW_IDENTITY_KEY_UNRESOLVED",
                "error",
                "Canonical identity contract key array does not resolve beside its owner.",
                key_path=key_path,
            )
        )
        return parsed_contract, parsed_contract.domain, issues
    try:
        load_row_identity_key_attrs(key_node.attributes, contract=parsed_contract)
    except RowIdentityContractError as exc:
        issues.append(
            _issue(
                "ROW_IDENTITY_KEY_RECORD_INVALID",
                "error",
                "Canonical key array metadata does not bind its owning identity contract.",
                key_path=key_path,
                validation_issues=[
                    {
                        "code": item.code,
                        "path": item.path,
                        "message": item.message,
                    }
                    for item in exc.issues
                ],
            )
        )
    key_dtype = _metadata_dtype(key_node.data_type)
    if (
        key_dtype is None
        or key_dtype.str != parsed_contract.key_array.dtype
        or tuple(key_node.shape or ()) != parsed_contract.key_array.shape
    ):
        issues.append(
            _issue(
                "ROW_IDENTITY_KEY_METADATA_MISMATCH",
                "error",
                "Canonical key-array shape/dtype disagree with the row-identity contract.",
                key_path=key_path,
                metadata_shape=key_node.shape,
                metadata_dtype=(key_dtype.str if key_dtype is not None else key_node.data_type),
                contract_shape=list(parsed_contract.key_array.shape),
                contract_dtype=parsed_contract.key_array.dtype,
            )
        )
    surface_count, component_counts = _surface_leading_dimension(
        surface_node,
        nodes=nodes,
        excluded_paths={key_path},
    )
    if surface_count is not None and surface_count != parsed_contract.leading_dimension:
        issues.append(
            _issue(
                "ROW_IDENTITY_LENGTH_MISMATCH",
                "error",
                "Canonical descriptor row count disagrees with its external identity contract.",
                surface_count=surface_count,
                identity_count=parsed_contract.leading_dimension,
            )
        )
    if component_counts and len(component_counts) > 1:
        issues.append(
            _issue(
                "ROW_IDENTITY_LENGTH_MISMATCH",
                "error",
                "Coordinate component arrays disagree on leading row count.",
                component_counts=component_counts,
            )
        )
    issues.append(
        _issue(
            "ROW_IDENTITY_KEY_PAYLOAD_VALIDATION_REQUIRED",
            "warning",
            "Canonical identity metadata is bound, but this metadata-only scan does not hash key-array payload values.",
            key_path=key_path,
            expected_content_sha256=parsed_contract.key_array.content_sha256,
        )
    )
    return parsed_contract, parsed_contract.domain, issues


def _directed_transform_v2_endpoint_issues(
    endpoint: Any,
    *,
    role: str,
    nodes: Mapping[str, MetadataNode],
) -> list[dict[str, Any]]:
    """Resolve one v2 endpoint to its exact typed pixel-frame record."""

    frame, _target, issues = _pixel_frame_record_metadata(
        record_ref=endpoint.record_ref,
        record_sha256=endpoint.record_sha256,
        role=role,
        nodes=nodes,
    )
    if frame is None:
        return issues
    if (
        endpoint.selector != PIXEL_FRAME_AUTHORITY_ATTR
        or endpoint.space_id != frame.space_id
        or endpoint.width != frame.reference_extent.get("width")
        or endpoint.height != frame.reference_extent.get("height")
        or endpoint.units != frame.coordinate_units
        or endpoint.pixel_convention != frame.pixel_convention
    ):
        issues.append(
            _issue(
                "DIRECTED_TRANSFORM_V2_ENDPOINT_FRAME_MISMATCH",
                "error",
                "Directed-transform endpoint fields disagree with the exact typed pixel frame.",
                role=role,
                endpoint=endpoint.to_dict(),
                frame_record=frame.to_dict(),
            )
        )
    return issues


def _directed_transform_v2_identity_issues(
    transform: Any,
    *,
    nodes: Mapping[str, MetadataNode],
) -> list[dict[str, Any]]:
    identity = transform.row_identity
    if identity is None:
        return []
    issues: list[dict[str, Any]] = []
    contract_path, attr_name = _canonical_v2_record_target(identity.record_ref)
    contract_node = nodes.get(contract_path or "")
    if (
        attr_name != ROW_IDENTITY_CONTRACT_ATTR
        or contract_node is None
        or contract_node.relative_path != contract_path
    ):
        return [
            _issue(
                "DIRECTED_TRANSFORM_V2_ROW_IDENTITY_UNRESOLVED",
                "error",
                "Rowwise transform identity must resolve to an exact @row_identity_contract record.",
                record_ref=identity.record_ref,
            )
        ]
    try:
        contract = load_row_identity_contract_attrs(contract_node.attributes)
    except RowIdentityContractError as exc:
        return [
            _issue(
                "DIRECTED_TRANSFORM_V2_ROW_IDENTITY_INVALID",
                "error",
                "Rowwise transform identity contract or digest is invalid.",
                record_ref=identity.record_ref,
                validation_issues=[
                    {"code": item.code, "path": item.path, "message": item.message}
                    for item in exc.issues
                ],
            )
        ]
    if (
        contract.digest() != identity.record_sha256
        or contract.leading_dimension != identity.leading_dimension
        or contract.domain != OBSERVATION_INSTANCE_DOMAIN
        or contract.mode != INSTANCE_KEY_MODE
        or contract.key_array.ref != INSTANCE_KEY_ARRAY_REF
    ):
        issues.append(
            _issue(
                "DIRECTED_TRANSFORM_V2_ROW_IDENTITY_MISMATCH",
                "error",
                "Rowwise transform identity differs from the exact observation-instance rowset contract.",
                record_ref=identity.record_ref,
                transform_identity=identity.to_dict(),
                contract=contract.to_dict(),
            )
        )
    key_path = f"{contract_path}/{contract.key_array.ref}"
    key_node = nodes.get(key_path)
    if key_node is None or key_node.node_type != "array":
        issues.append(
            _issue(
                "DIRECTED_TRANSFORM_V2_ROW_IDENTITY_KEY_UNRESOLVED",
                "error",
                "Rowwise transform identity key array is absent from its exact rowset.",
                key_path=key_path,
            )
        )
    else:
        try:
            load_row_identity_key_attrs(key_node.attributes, contract=contract)
        except RowIdentityContractError as exc:
            issues.append(
                _issue(
                    "DIRECTED_TRANSFORM_V2_ROW_IDENTITY_KEY_INVALID",
                    "error",
                    "Rowwise transform key metadata does not bind its rowset contract.",
                    key_path=key_path,
                    validation_issues=[
                        {
                            "code": item.code,
                            "path": item.path,
                            "message": item.message,
                        }
                        for item in exc.issues
                    ],
                )
            )
        key_dtype = _metadata_dtype(key_node.data_type)
        if (
            key_node.relative_path != key_path
            or key_dtype is None
            or key_dtype.str != contract.key_array.dtype
            or tuple(key_node.shape or ()) != contract.key_array.shape
        ):
            issues.append(
                _issue(
                    "DIRECTED_TRANSFORM_V2_ROW_IDENTITY_KEY_METADATA_MISMATCH",
                    "error",
                    "Rowwise transform key path, shape, and dtype must equal the exact sealed rowset key declaration.",
                    expected_key_path=key_path,
                    actual_key_path=key_node.relative_path,
                    metadata_shape=key_node.shape,
                    metadata_dtype=(
                        key_dtype.str
                        if key_dtype is not None
                        else key_node.data_type
                    ),
                    contract_shape=list(contract.key_array.shape),
                    contract_dtype=contract.key_array.dtype,
                )
            )
    return issues


def _parse_directed_transform_v2_node(
    target: MetadataNode,
    *,
    record_ref: str,
    nodes: Mapping[str, MetadataNode],
) -> tuple[Any | None, list[dict[str, Any]]]:
    """Validate all v2 metadata that can be proven without array payload reads."""

    issues: list[dict[str, Any]] = []
    _node_path, attr_name = _canonical_v2_record_target(record_ref)
    if attr_name != DIRECTED_TRANSFORM_V2_ATTR:
        return None, [
            _issue(
                "DIRECTED_TRANSFORM_V2_REF_INVALID",
                "error",
                "Canonical transform refs must name an exact @directed_transform_v2 record.",
                record_ref=record_ref,
            )
        ]
    raw = target.attributes.get(DIRECTED_TRANSFORM_V2_ATTR)
    try:
        transform = parse_directed_transform_v2(raw)
    except DirectedTransformV2Error as exc:
        return None, [
            _issue(
                "DIRECTED_TRANSFORM_V2_METADATA_INVALID",
                "error",
                "Directed-transform-v2 metadata fails the shared strict parser.",
                record_ref=record_ref,
                error=str(exc),
            )
        ]
    if not _exact_json_equal(raw, transform.to_dict()):
        issues.append(
            _issue(
                "DIRECTED_TRANSFORM_V2_NONCANONICAL",
                "error",
                "Persisted directed transform is not its exact type-strict canonical mapping.",
                record_ref=record_ref,
            )
        )
    stored = target.attributes.get(DIRECTED_TRANSFORM_V2_DIGEST_ATTR)
    if stored != transform.digest():
        issues.append(
            _issue(
                "DIRECTED_TRANSFORM_V2_DIGEST_MISMATCH",
                "error",
                "Persisted directed-transform-v2 digest is missing or stale.",
                record_ref=record_ref,
                declared_sha256=stored,
                actual_sha256=transform.digest(),
            )
        )
    expected_shape = (
        [int(transform.row_identity.leading_dimension), 4]
        if transform.kind == AFFINE_2D_ROWWISE_KIND
        and transform.row_identity is not None
        else [3, 3]
    )
    dtype = _metadata_dtype(target.data_type)
    dtype_valid = (
        dtype is not None
        and dtype.kind in "iuf"
        and (
            transform.kind == AFFINE_2D_ROWWISE_KIND
            or dtype.str == np.dtype("<f8").str
        )
    )
    if target.node_type != "array" or list(target.shape or ()) != expected_shape or not dtype_valid:
        issues.append(
            _issue(
                "DIRECTED_TRANSFORM_V2_PAYLOAD_METADATA_INVALID",
                "error",
                "Transform array shape/dtype disagree with the strict v2 transform kind.",
                record_ref=record_ref,
                transform_kind=transform.kind,
                expected_shape=expected_shape,
                actual_shape=target.shape,
                actual_dtype=target.data_type,
            )
        )
    expected_payload_ref = f"/{target.relative_path}@array_values"
    if (
        transform.payload.record_ref != expected_payload_ref
        or transform.payload.selector != "array_values"
    ):
        issues.append(
            _issue(
                "DIRECTED_TRANSFORM_V2_PAYLOAD_REF_MISMATCH",
                "error",
                "Transform payload pointer does not identify this exact array's values.",
                record_ref=record_ref,
                declared_payload_ref=transform.payload.record_ref,
                expected_payload_ref=expected_payload_ref,
            )
        )
    issues.extend(
        _directed_transform_v2_endpoint_issues(
            transform.source,
            role=f"{record_ref}.source",
            nodes=nodes,
        )
    )
    issues.extend(
        _directed_transform_v2_endpoint_issues(
            transform.target,
            role=f"{record_ref}.target",
            nodes=nodes,
        )
    )
    issues.extend(_directed_transform_v2_identity_issues(transform, nodes=nodes))

    authority_pointer = {
        "record_ref": transform.transform_authority.record_ref,
        "record_sha256": transform.transform_authority.record_sha256,
    }
    authority_node, authority_issues = _canonical_v2_bound_record_issues(
        authority_pointer,
        role=f"{record_ref}.transform_authority",
        nodes=nodes,
    )
    issues.extend(authority_issues)
    _authority_path, authority_attr = _canonical_v2_record_target(
        transform.transform_authority.record_ref
    )
    authority = None
    if authority_node is not None and authority_attr != TRANSFORM_AUTHORITY_ATTR:
        issues.append(
            _issue(
                "DIRECTED_TRANSFORM_V2_AUTHORITY_REF_INVALID",
                "error",
                "Transform authority must name an exact @transform_authority record.",
                record_ref=transform.transform_authority.record_ref,
            )
        )
    elif authority_node is not None:
        raw_authority = authority_node.attributes.get(TRANSFORM_AUTHORITY_ATTR)
        try:
            authority = parse_transform_authority(raw_authority)
        except TransformAuthorityError as exc:
            issues.append(
                _issue(
                    "DIRECTED_TRANSFORM_V2_AUTHORITY_INVALID",
                    "error",
                    "Transform authority fails the shared strict parser.",
                    record_ref=transform.transform_authority.record_ref,
                    error=str(exc),
                )
            )
        else:
            if not _exact_json_equal(raw_authority, authority.to_dict()):
                issues.append(
                    _issue(
                        "DIRECTED_TRANSFORM_V2_AUTHORITY_NONCANONICAL",
                        "error",
                        "Transform authority is not its exact type-strict canonical mapping.",
                        record_ref=transform.transform_authority.record_ref,
                    )
                )
            authority_digest = authority_node.attributes.get(
                TRANSFORM_AUTHORITY_DIGEST_ATTR
            )
            if (
                authority_digest != authority.digest()
                or transform.transform_authority.record_sha256
                != authority.digest()
                or transform.transform_authority.kind != authority.kind
            ):
                issues.append(
                    _issue(
                        "DIRECTED_TRANSFORM_V2_AUTHORITY_DIGEST_MISMATCH",
                        "error",
                        "Transform authority pointer, kind, or stored digest is stale.",
                        record_ref=transform.transform_authority.record_ref,
                        pointer_kind=transform.transform_authority.kind,
                        authority_kind=authority.kind,
                        pointer_sha256=transform.transform_authority.record_sha256,
                        stored_sha256=authority_digest,
                        actual_sha256=authority.digest(),
                    )
                )

    expected_authority_kinds = {
        HOMOGRAPHY_KIND: {SELECTED_CALIBRATION_AUTHORITY_KIND},
        AFFINE_2D_CONSTANT_KIND: {
            MODEL_INPUT_PREPROCESSING_AUTHORITY_KIND,
            ARENA_CANVAS_PLACEMENT_AUTHORITY_KIND,
            NORMALIZED_TO_PIXEL_AUTHORITY_KIND,
        },
        AFFINE_2D_ROWWISE_KIND: {CROP_PLACEMENT_AUTHORITY_KIND},
    }[transform.kind]
    if authority is not None:
        forward = transform.inverse_of is None
        endpoint_match = (
            authority.source == transform.source
            and authority.target == transform.target
            if forward
            else authority.source == transform.target
            and authority.target == transform.source
        )
        record_match = (
            authority.kind in expected_authority_kinds
            and endpoint_match
            and authority.sampling_formula == transform.sampling_formula
            and authority.camera_id == transform.camera_id
            and (
                (forward and authority.row_identity == transform.row_identity)
                or (not forward and transform.row_identity is None)
            )
            and (not forward or authority.payload == transform.payload)
        )
        if not record_match:
            issues.append(
                _issue(
                    "DIRECTED_TRANSFORM_V2_AUTHORITY_MISMATCH",
                    "error",
                    "Directed transform differs from its exact typed transform authority.",
                    record_ref=record_ref,
                    authority_ref=transform.transform_authority.record_ref,
                )
            )

    if transform.inverse_of is not None:
        inverse_target, inverse_issues = _canonical_v2_bound_record_issues(
            {
                "record_ref": transform.inverse_of.record_ref,
                "record_sha256": transform.inverse_of.record_sha256,
            },
            role=f"{record_ref}.inverse_of",
            nodes=nodes,
        )
        issues.extend(inverse_issues)
        inverse_attr = _canonical_v2_record_target(
            transform.inverse_of.record_ref
        )[1]
        if inverse_target is None or inverse_attr != DIRECTED_TRANSFORM_V2_ATTR:
            issues.append(
                _issue(
                    "DIRECTED_TRANSFORM_V2_INVERSE_REF_INVALID",
                    "error",
                    "inverse_of must bind an exact persisted directed-transform-v2 record.",
                    record_ref=transform.inverse_of.record_ref,
                )
            )
        else:
            raw_forward_transform = inverse_target.attributes.get(
                DIRECTED_TRANSFORM_V2_ATTR
            )
            try:
                forward_transform = parse_directed_transform_v2(
                    raw_forward_transform
                )
            except DirectedTransformV2Error as exc:
                issues.append(
                    _issue(
                        "DIRECTED_TRANSFORM_V2_INVERSE_TARGET_INVALID",
                        "error",
                        "inverse_of target fails the strict v2 parser.",
                        record_ref=transform.inverse_of.record_ref,
                        error=str(exc),
                    )
                )
            else:
                if (
                    forward_transform.digest() != transform.inverse_of.record_sha256
                    or inverse_target.attributes.get(
                        DIRECTED_TRANSFORM_V2_DIGEST_ATTR
                    )
                    != forward_transform.digest()
                    or not _exact_json_equal(
                        raw_forward_transform, forward_transform.to_dict()
                    )
                    or forward_transform.inverse_of is not None
                    or forward_transform.kind != HOMOGRAPHY_KIND
                    or forward_transform.source != transform.target
                    or forward_transform.target != transform.source
                    or forward_transform.transform_authority
                    != transform.transform_authority
                ):
                    issues.append(
                        _issue(
                            "DIRECTED_TRANSFORM_V2_INVERSE_MISMATCH",
                            "error",
                            "Explicit inverse does not swap and bind the exact forward transform.",
                            record_ref=record_ref,
                            inverse_of_ref=transform.inverse_of.record_ref,
                        )
                    )

    issues.append(
        _issue(
            "DIRECTED_TRANSFORM_V2_LIVE_VALIDATION_REQUIRED",
            "warning",
            "V2 transform metadata is internally bound, but array payload hashes, numerical validity, and producer-sealed authorities require a live Zarr validation pass.",
            record_ref=record_ref,
            expected_payload_sha256=transform.payload.record_sha256,
        )
    )
    return transform, issues


def _canonical_v2_transform_issues(
    descriptor: Mapping[str, Any],
    *,
    nodes: Mapping[str, MetadataNode],
) -> list[dict[str, Any]]:
    overlay = _as_mapping(descriptor.get("source_camera_overlay"))
    raw_refs = overlay.get("transform_refs")
    refs = raw_refs if isinstance(raw_refs, (list, tuple)) else ()
    if overlay.get("status") != "requires_transform":
        return []
    issues: list[dict[str, Any]] = []
    transforms: list[tuple[str, Any]] = []
    for raw in refs:
        record = _as_mapping(raw)
        target, record_issues = _canonical_v2_bound_record_issues(
            record,
            role="source_camera_overlay.transform",
            nodes=nodes,
        )
        issues.extend(record_issues)
        if target is None:
            continue
        transform, transform_issues = _parse_directed_transform_v2_node(
            target,
            record_ref=str(record.get("record_ref") or ""),
            nodes=nodes,
        )
        issues.extend(transform_issues)
        if transform is not None:
            transforms.append((str(record.get("record_ref")), transform))
    if len(transforms) != len(refs):
        issues.append(
            _issue(
                "DIRECTED_TRANSFORM_V2_CHAIN_UNRESOLVED",
                "error",
                "Every canonical overlay transform must resolve before its direction can be trusted.",
                declared_ref_count=len(refs),
                resolved_ref_count=len(transforms),
            )
        )
    if not transforms:
        return issues

    expected_source_space = str(descriptor.get("space_id"))
    seen_spaces = {expected_source_space}
    seen_records: set[str] = set()
    for index, (record_ref, transform) in enumerate(transforms):
        if record_ref in seen_records:
            issues.append(
                _issue(
                    "DIRECTED_TRANSFORM_V2_CHAIN_CYCLE",
                    "error",
                    "Canonical overlay chain repeats a transform record.",
                    record_ref=record_ref,
                    chain_index=index,
                )
            )
        seen_records.add(record_ref)
        if index == 0:
            connected = transform.source.space_id == expected_source_space
        else:
            connected = transforms[index - 1][1].target == transform.source
        if not connected:
            issues.append(
                _issue(
                    "TRANSFORM_CHAIN_DISCONNECTED_OR_REVERSED",
                    "error",
                    "Canonical overlay chain must match exact typed endpoints in descriptor-to-camera order.",
                    record_ref=record_ref,
                    expected_source_space_id=expected_source_space,
                    actual_source=transform.source.to_dict(),
                    chain_index=index,
                )
            )
        if transform.source.space_id == SOURCE_CAMERA_IMAGE_SPACE_ID:
            issues.append(
                _issue(
                    "DIRECTED_TRANSFORM_V2_CAMERA_NOT_TERMINAL",
                    "error",
                    "source_camera_image_px may occur only as the final chain endpoint.",
                    record_ref=record_ref,
                    chain_index=index,
                )
            )
        target_space = transform.target.space_id
        if target_space in seen_spaces:
            issues.append(
                _issue(
                    "DIRECTED_TRANSFORM_V2_CHAIN_CYCLE",
                    "error",
                    "Canonical overlay chain repeats a coordinate space and is cyclic or ambiguous.",
                    repeated_space_id=target_space,
                    chain_index=index,
                )
            )
        seen_spaces.add(target_space)
        if (
            target_space == SOURCE_CAMERA_IMAGE_SPACE_ID
            and index != len(transforms) - 1
        ):
            issues.append(
                _issue(
                    "DIRECTED_TRANSFORM_V2_CAMERA_NOT_TERMINAL",
                    "error",
                    "source_camera_image_px may occur only as the final chain endpoint.",
                    record_ref=record_ref,
                    chain_index=index,
                )
            )
        expected_source_space = target_space
    if transforms[-1][1].target.space_id != SOURCE_CAMERA_IMAGE_SPACE_ID:
        issues.append(
            _issue(
                "SOURCE_CAMERA_TRANSFORM_ROUTE_MISSING",
                "error",
                "Canonical overlay chain does not terminate in source_camera_image_px.",
                terminal_space_id=transforms[-1][1].target.space_id,
            )
        )
    return issues


def _canonical_observation_rowset_family(path: str) -> str | None:
    parts = PurePosixPath(path).parts
    if len(parts) == 2 and parts[0] in {"detect_runs", "crop_runs"}:
        return parts[0]
    return None


def _observation_array_payload_metadata(
    value: Any,
    *,
    role: str,
    nodes: Mapping[str, MetadataNode],
    expected_path: str | None = None,
    expected_shape: tuple[int | None, ...] | None = None,
    expected_dtype: str | None = None,
    dtype_kind: str | None = None,
) -> tuple[MetadataNode | None, list[dict[str, Any]]]:
    """Validate the metadata portion of one observation array-payload record."""

    raw = _as_mapping(value)
    array_ref = raw.get("array_ref")
    path = _normalize_archive_ref(array_ref) if isinstance(array_ref, str) else None
    target = nodes.get(path or "")
    dtype = _metadata_dtype(target.data_type) if target is not None else None
    declared_dtype = _metadata_dtype(raw.get("dtype"))
    shape = tuple(target.shape or ()) if target is not None else ()
    declared_shape = raw.get("shape")
    shape_matches = (
        expected_shape is None
        or len(shape) == len(expected_shape)
        and all(
            expected is None or actual == expected
            for actual, expected in zip(shape, expected_shape, strict=True)
        )
    )
    ref_valid = path is not None and array_ref == f"/{path}"
    valid = (
        type(value) is dict
        and set(raw) == {"array_ref", "dtype", "shape", "content_sha256"}
        and isinstance(array_ref, str)
        and ref_valid
    )
    valid = bool(
        valid
        and (expected_path is None or path == expected_path)
        and target is not None
        and target.node_type == "array"
        and type(declared_shape) is list
        and all(type(item) is int and item >= 0 for item in declared_shape)
        and tuple(declared_shape) == shape
        and shape_matches
        and type(raw.get("dtype")) is str
        and declared_dtype is not None
        and dtype is not None
        and declared_dtype.str == dtype.str
        and raw.get("dtype") == dtype.str
        and (expected_dtype is None or dtype.str == expected_dtype)
        and (dtype_kind is None or dtype.kind in dtype_kind)
        and isinstance(raw.get("content_sha256"), str)
        and _SHA256_HEX_RE.fullmatch(str(raw.get("content_sha256")))
        is not None
    )
    if valid:
        return target, []
    return target, [
        _issue(
            "OBSERVATION_ARRAY_PAYLOAD_METADATA_INVALID",
            "error",
            "Observation payload metadata requires an exact canonical child path, dtype, shape, and syntactic content digest.",
            role=role,
            expected_path=expected_path,
            expected_shape=(
                list(expected_shape) if expected_shape is not None else None
            ),
            expected_dtype=expected_dtype,
            payload=raw,
            resolved_path=path,
            metadata_shape=(target.shape if target is not None else None),
            metadata_dtype=(target.data_type if target is not None else None),
        )
    ]


def _observation_pointer_matches(
    value: Any,
    *,
    expected_ref: str,
    expected_sha256: str,
) -> bool:
    return (
        type(value) is dict
        and set(value) == {"record_ref", "record_sha256"}
        and value.get("record_ref") == expected_ref
        and value.get("record_sha256") == expected_sha256
    )


def _observation_source_temporal_authority_issues(
    *,
    rowset_path: str,
    contract: Any,
    nodes: Mapping[str, MetadataNode],
) -> list[dict[str, Any]]:
    """Validate observation row time without reading its array payload bytes."""

    rowset = nodes.get(rowset_path)
    raw_value = (
        rowset.attributes.get(SOURCE_ROW_TEMPORAL_AUTHORITY_ATTR)
        if rowset is not None
        else None
    )
    raw = _as_mapping(raw_value)
    digest = _fingerprint(raw) if raw else None
    pointer = {
        "record_ref": f"/{rowset_path}@{SOURCE_ROW_TEMPORAL_AUTHORITY_ATTR}",
        "record_sha256": digest,
    }
    _target, issues = _canonical_v2_bound_record_issues(
        pointer,
        role="observation.source_row_temporal_authority",
        nodes=nodes,
    )
    expected_fields = {
        "schema_id",
        "schema_version",
        "acquisition_camera_frame",
        "recording_id",
        "camera_id",
        "source_total_frames",
        "source_rowset_ref",
        "source_row_identity",
        "source_identity_domain",
        "source_identity_mode",
        "source_leading_dimension",
        "source_acquisition_frame_index",
        "observation_instance_key",
    }
    expected_identity_ref = f"/{rowset_path}@{ROW_IDENTITY_CONTRACT_ATTR}"
    if (
        rowset is None
        or rowset.node_type != "group"
        or type(raw_value) is not dict
        or set(raw) != expected_fields
        or raw.get("schema_id") != SOURCE_ROW_TEMPORAL_AUTHORITY_SCHEMA_ID
        or type(raw.get("schema_version")) is not int
        or raw.get("schema_version")
        != SOURCE_ROW_TEMPORAL_AUTHORITY_SCHEMA_VERSION
        or raw.get("source_rowset_ref") != f"/{rowset_path}"
        or not _observation_pointer_matches(
            raw.get("source_row_identity"),
            expected_ref=expected_identity_ref,
            expected_sha256=contract.digest(),
        )
        or raw.get("source_identity_domain") != OBSERVATION_INSTANCE_DOMAIN
        or raw.get("source_identity_domain") != contract.domain
        or raw.get("source_identity_mode") != INSTANCE_KEY_MODE
        or raw.get("source_identity_mode") != contract.mode
        or raw.get("source_leading_dimension") != contract.leading_dimension
    ):
        issues.append(
            _issue(
                "OBSERVATION_TEMPORAL_AUTHORITY_INVALID",
                "error",
                "Observation temporal authority must use the exact rowset path, instance identity, schema, and row count.",
                rowset_path=rowset_path,
                record_fields=sorted(str(name) for name in raw),
            )
        )

    source_frame_value = raw.get("source_acquisition_frame_index")
    source_frame = _as_mapping(source_frame_value)
    source_frame_path = f"{rowset_path}/{SOURCE_ROW_ACQUISITION_FRAME_INDEX_REF}"
    source_frame_node = nodes.get(source_frame_path)
    if not (
        type(source_frame_value) is dict
        and set(source_frame)
        == {"ref", "dtype", "shape", "content_sha256", "canonicalization"}
        and source_frame.get("ref") == f"/{source_frame_path}"
        and source_frame.get("dtype") == "<i8"
        and source_frame.get("shape") == [contract.leading_dimension]
        and source_frame.get("canonicalization")
        == ROW_IDENTITY_KEY_CONTENT_CANONICALIZATION
        and isinstance(source_frame.get("content_sha256"), str)
        and _SHA256_HEX_RE.fullmatch(str(source_frame.get("content_sha256")))
        is not None
        and source_frame_node is not None
        and source_frame_node.node_type == "array"
        and tuple(source_frame_node.shape or ()) == (contract.leading_dimension,)
        and _identity_dtype_signature(source_frame_node.data_type) == "<i8"
    ):
        issues.append(
            _issue(
                "OBSERVATION_TEMPORAL_ARRAY_INVALID",
                "error",
                "Observation source_acquisition_frame_index metadata must bind the exact int64 row-aligned child array.",
                rowset_path=rowset_path,
                payload=source_frame,
            )
        )

    instance_value = raw.get("observation_instance_key")
    instance = _as_mapping(instance_value)
    instance_path = f"{rowset_path}/{INSTANCE_KEY_ARRAY_REF}"
    instance_node = nodes.get(instance_path)
    if not (
        type(instance_value) is dict
        and set(instance)
        == {"ref", "dtype", "shape", "content_sha256", "canonicalization"}
        and instance.get("ref") == f"/{instance_path}"
        and instance.get("dtype") == "<u8"
        and instance.get("shape") == [contract.leading_dimension]
        and instance.get("content_sha256")
        == contract.key_array.content_sha256
        and instance.get("canonicalization")
        == ROW_IDENTITY_KEY_CONTENT_CANONICALIZATION
        and instance_node is not None
        and instance_node.node_type == "array"
        and tuple(instance_node.shape or ()) == (contract.leading_dimension,)
        and _identity_dtype_signature(instance_node.data_type) == "<u8"
    ):
        issues.append(
            _issue(
                "OBSERVATION_TEMPORAL_IDENTITY_ARRAY_INVALID",
                "error",
                "Observation temporal authority must bind the exact canonical instance_key payload metadata and digest.",
                rowset_path=rowset_path,
                payload=instance,
            )
        )

    acquisition_pointer = _as_mapping(raw.get("acquisition_camera_frame"))
    acquisition_path, acquisition_attr = _canonical_v2_record_target(
        acquisition_pointer.get("record_ref")
    )
    acquisition_node = nodes.get(acquisition_path or "")
    acquisition = None
    try:
        if (
            type(raw.get("acquisition_camera_frame")) is not dict
            or set(acquisition_pointer) != {"record_ref", "record_sha256"}
            or acquisition_node is None
            or acquisition_attr != ACQUISITION_CAMERA_FRAME_ATTR
        ):
            raise PixelFrameAuthorityError("invalid acquisition pointer")
        acquisition_raw = acquisition_node.attributes.get(
            ACQUISITION_CAMERA_FRAME_ATTR
        )
        acquisition = parse_acquisition_camera_frame(acquisition_raw)
    except PixelFrameAuthorityError as exc:
        issues.append(
            _issue(
                "OBSERVATION_TEMPORAL_ACQUISITION_INVALID",
                "error",
                "Observation temporal authority must bind one exact acquisition-camera record.",
                rowset_path=rowset_path,
                error=str(exc),
            )
        )
    else:
        _acquisition_target, acquisition_issues = _reference_extent_binding_issues(
            {
                **acquisition_pointer,
                "selector": ACQUISITION_CAMERA_FRAME_ATTR,
                "width": acquisition.width_px,
                "height": acquisition.height_px,
                "units": "px",
            },
            role="observation.source_row_temporal_authority.acquisition",
            nodes=nodes,
        )
        issues.extend(acquisition_issues)
        if (
            acquisition_pointer.get("record_sha256") != acquisition.digest()
            or raw.get("recording_id") != acquisition.recording_id
            or raw.get("camera_id") != acquisition.camera_id
            or raw.get("source_total_frames")
            != acquisition.source_total_frames
        ):
            issues.append(
                _issue(
                    "OBSERVATION_TEMPORAL_ACQUISITION_MISMATCH",
                    "error",
                    "Observation temporal identifiers must equal the exact acquisition authority.",
                    rowset_path=rowset_path,
                )
            )
    return issues


def _registered_lineage_records(
    descriptor: Mapping[str, Any],
    *,
    nodes: Mapping[str, MetadataNode],
) -> dict[str, list[tuple[dict[str, Any], MetadataNode, dict[str, Any]]]]:
    result: dict[
        str,
        list[tuple[dict[str, Any], MetadataNode, dict[str, Any]]],
    ] = {}
    for value in descriptor.get("lineage_refs") or ():
        pointer = _as_mapping(value)
        path, attr = _canonical_v2_record_target(pointer.get("record_ref"))
        target = nodes.get(path or "")
        if target is None or attr is None:
            continue
        record = _as_mapping(target.attributes.get(attr))
        schema_id = record.get("schema_id")
        if schema_id not in _REGISTERED_OBSERVATION_COORDINATE_RECORDS:
            continue
        result.setdefault(str(schema_id), []).append(
            (pointer, target, record)
        )
    return result


def _detection_acquisition_mapping_issues(
    *,
    target: MetadataNode,
    record: Mapping[str, Any],
    nodes: Mapping[str, MetadataNode],
) -> list[dict[str, Any]]:
    rowset_path = target.relative_path
    expected_fields = {
        "schema_id",
        "schema_version",
        "operation",
        "direction",
        "decode_frame_index",
        "source_acquisition_frame_index",
        "acquisition_camera_frame",
        "source_video_locator",
        "source_video_fingerprint",
        "source_total_frames",
        "proof",
    }
    issues: list[dict[str, Any]] = []
    decode_node, decode_issues = _observation_array_payload_metadata(
        record.get("decode_frame_index"),
        role="detection_acquisition_frame_mapping.decode_frame_index",
        nodes=nodes,
        expected_path=f"{rowset_path}/frame_indices",
        expected_shape=(None,),
        dtype_kind="iu",
    )
    source_node, source_issues = _observation_array_payload_metadata(
        record.get("source_acquisition_frame_index"),
        role="detection_acquisition_frame_mapping.source_acquisition_frame_index",
        nodes=nodes,
        expected_path=f"{rowset_path}/source_acquisition_frame_index",
        expected_shape=(None,),
        expected_dtype="<i8",
    )
    issues.extend(decode_issues)
    issues.extend(source_issues)
    acquisition_pointer = _as_mapping(record.get("acquisition_camera_frame"))
    acquisition_path, acquisition_attr = _canonical_v2_record_target(
        acquisition_pointer.get("record_ref")
    )
    acquisition_node = nodes.get(acquisition_path or "")
    acquisition = None
    try:
        if (
            type(record.get("acquisition_camera_frame")) is not dict
            or set(acquisition_pointer) != {"record_ref", "record_sha256"}
            or acquisition_node is None
            or acquisition_attr != ACQUISITION_CAMERA_FRAME_ATTR
        ):
            raise PixelFrameAuthorityError("invalid acquisition pointer")
        acquisition = parse_acquisition_camera_frame(
            acquisition_node.attributes.get(ACQUISITION_CAMERA_FRAME_ATTR)
        )
    except PixelFrameAuthorityError as exc:
        issues.append(
            _issue(
                "DETECTION_ACQUISITION_MAPPING_INVALID",
                "error",
                "Detection acquisition mapping has no exact acquisition-camera authority.",
                rowset_path=rowset_path,
                error=str(exc),
            )
        )
    if (
        set(record) != expected_fields
        or record.get("operation")
        != "full_untrimmed_video_decode_identity_to_acquisition_v1"
        or record.get("direction")
        != "decode_frame_index_to_source_acquisition_frame_index"
        or record.get("proof")
        != "exact_locator_and_stat_fingerprint_revalidated_after_full_decode"
        or decode_node is None
        or source_node is None
        or tuple(decode_node.shape or ()) != tuple(source_node.shape or ())
        or acquisition is None
        or acquisition_pointer.get("record_sha256") != acquisition.digest()
        or record.get("source_total_frames") != acquisition.source_total_frames
        or not _exact_json_equal(
            record.get("source_video_locator"),
            acquisition.source_video_metadata.get("locator"),
        )
        or not _exact_json_equal(
            record.get("source_video_fingerprint"),
            acquisition.source_video_metadata.get("file_fingerprint"),
        )
    ):
        issues.append(
            _issue(
                "DETECTION_ACQUISITION_MAPPING_INVALID",
                "error",
                "Detection decode mapping must bind exact full-video source/destination arrays, direction, and acquisition source evidence.",
                rowset_path=rowset_path,
            )
        )
    return issues


def _detection_bbox_projection_issues(
    *,
    target: MetadataNode,
    record: Mapping[str, Any],
    contract: Any,
    temporal_digest: str | None,
    nodes: Mapping[str, MetadataNode],
) -> list[dict[str, Any]]:
    rowset_path = target.relative_path
    expected_fields = {
        "schema_id",
        "schema_version",
        "operation",
        "source_bbox",
        "source_frame",
        "destination_bbox",
        "destination_frame",
        "direction",
        "transform_chain",
        "reference_width_px",
        "reference_height_px",
        "formula",
        "row_identity",
        "temporal_authority",
        "source_lineage",
    }
    issues: list[dict[str, Any]] = []
    source_node, source_issues = _observation_array_payload_metadata(
        record.get("source_bbox"),
        role="detection_bbox_projection.source_bbox",
        nodes=nodes,
        expected_path=f"{rowset_path}/bbox_norm_coords",
        expected_shape=(contract.leading_dimension, 4),
        dtype_kind="f",
    )
    destination_node, destination_issues = _observation_array_payload_metadata(
        record.get("destination_bbox"),
        role="detection_bbox_projection.destination_bbox",
        nodes=nodes,
        expected_path=f"{rowset_path}/bbox_img_xyxy",
        expected_shape=(contract.leading_dimension, 4),
        dtype_kind="f",
    )
    issues.extend(source_issues)
    issues.extend(destination_issues)
    source_frame_pointer = _as_mapping(record.get("source_frame"))
    source_frame, _source_frame_node, source_frame_issues = (
        _pixel_frame_record_metadata(
            record_ref=source_frame_pointer.get("record_ref"),
            record_sha256=source_frame_pointer.get("record_sha256"),
            role="detection_bbox_projection.source_frame",
            nodes=nodes,
        )
    )
    destination_frame_pointer = _as_mapping(record.get("destination_frame"))
    destination_frame, _destination_frame_node, destination_frame_issues = (
        _pixel_frame_record_metadata(
            record_ref=destination_frame_pointer.get("record_ref"),
            record_sha256=destination_frame_pointer.get("record_sha256"),
            role="detection_bbox_projection.destination_frame",
            nodes=nodes,
        )
    )
    issues.extend(source_frame_issues)
    issues.extend(destination_frame_issues)

    transforms: list[Any] = []
    raw_chain = record.get("transform_chain")
    if type(raw_chain) is list:
        for index, raw_pointer in enumerate(raw_chain):
            pointer = _as_mapping(raw_pointer)
            transform_path, transform_attr = _canonical_v2_record_target(
                pointer.get("record_ref")
            )
            transform_node = nodes.get(transform_path or "")
            if (
                type(raw_pointer) is not dict
                or set(pointer) != {"record_ref", "record_sha256"}
                or transform_node is None
                or transform_attr != DIRECTED_TRANSFORM_V2_ATTR
            ):
                issues.append(
                    _issue(
                        "DETECTION_BBOX_PROJECTION_INVALID",
                        "error",
                        "Detection projection transform_chain contains an unresolved or noncanonical pointer.",
                        chain_index=index,
                    )
                )
                continue
            transform, transform_issues = _parse_directed_transform_v2_node(
                transform_node,
                record_ref=str(pointer.get("record_ref")),
                nodes=nodes,
            )
            issues.extend(transform_issues)
            if transform is not None:
                if pointer.get("record_sha256") != transform.digest():
                    issues.append(
                        _issue(
                            "DETECTION_BBOX_PROJECTION_INVALID",
                            "error",
                            "Detection projection transform pointer digest is stale.",
                            chain_index=index,
                        )
                    )
                transforms.append(transform)

    identity_ref = f"/{rowset_path}@{ROW_IDENTITY_CONTRACT_ATTR}"
    temporal_ref = f"/{rowset_path}@{SOURCE_ROW_TEMPORAL_AUTHORITY_ATTR}"
    transform_connected = bool(transforms)
    if transforms and source_frame is not None and destination_frame is not None:
        transform_connected = (
            transforms[0].source.record_ref
            == source_frame_pointer.get("record_ref")
            and transforms[-1].target.record_ref
            == destination_frame_pointer.get("record_ref")
            and all(item.row_identity is None for item in transforms)
            and all(
                left.target == right.source
                for left, right in zip(transforms, transforms[1:], strict=False)
            )
        )
    if (
        set(record) != expected_fields
        or record.get("operation") != DETECTION_BBOX_PROJECTION_OPERATION
        or record.get("direction")
        != "source_camera_normalized_xy_to_source_camera_image_px"
        or record.get("formula")
        != "cxcywh_normalized_to_xyxy_edges_using_exact_reference_extent_v1"
        or not _observation_pointer_matches(
            record.get("row_identity"),
            expected_ref=identity_ref,
            expected_sha256=contract.digest(),
        )
        or temporal_digest is None
        or not _observation_pointer_matches(
            record.get("temporal_authority"),
            expected_ref=temporal_ref,
            expected_sha256=temporal_digest,
        )
        or source_node is None
        or destination_node is None
        or _metadata_dtype(source_node.data_type)
        != _metadata_dtype(destination_node.data_type)
        or source_frame is None
        or source_frame.kind != SOURCE_CAMERA_NORMALIZED_FRAME_KIND
        or destination_frame is None
        or destination_frame.kind != SOURCE_CAMERA_FRAME_KIND
        or record.get("reference_width_px")
        != destination_frame.reference_extent.get("width")
        or record.get("reference_height_px")
        != destination_frame.reference_extent.get("height")
        or not transform_connected
    ):
        issues.append(
            _issue(
                "DETECTION_BBOX_PROJECTION_INVALID",
                "error",
                "Detection bbox projection must bind exact normalized/image payloads, frames, direction, dimensions, identity, time, and transform chain.",
                rowset_path=rowset_path,
            )
        )
    return issues


def _bbox_center_derivation_issues(
    *,
    target: MetadataNode,
    record: Mapping[str, Any],
    contract: Any,
    nodes: Mapping[str, MetadataNode],
) -> list[dict[str, Any]]:
    rowset_path = target.relative_path
    issues: list[dict[str, Any]] = []
    source, source_issues = _observation_array_payload_metadata(
        record.get("source_bbox"),
        role="bbox_center_derivation.source_bbox",
        nodes=nodes,
        expected_path=f"{rowset_path}/bbox_img_xyxy",
        expected_shape=(contract.leading_dimension, 4),
        dtype_kind="f",
    )
    output, output_issues = _observation_array_payload_metadata(
        record.get("output_centers"),
        role="bbox_center_derivation.output_centers",
        nodes=nodes,
        expected_path=f"{rowset_path}/centers_img_xy",
        expected_shape=(contract.leading_dimension, 2),
        dtype_kind="f",
    )
    issues.extend(source_issues)
    issues.extend(output_issues)
    frame_pointer = _as_mapping(record.get("coordinate_frame"))
    frame, _frame_node, frame_issues = _pixel_frame_record_metadata(
        record_ref=frame_pointer.get("record_ref"),
        record_sha256=frame_pointer.get("record_sha256"),
        role="bbox_center_derivation.coordinate_frame",
        nodes=nodes,
    )
    issues.extend(frame_issues)
    if (
        set(record)
        != {
            "schema_id",
            "schema_version",
            "operation",
            "source_bbox",
            "output_centers",
            "coordinate_frame",
            "formula",
            "row_identity",
        }
        or record.get("operation") != BBOX_CENTER_DERIVATION_OPERATION
        or record.get("formula")
        != "center_x=(x_min+x_max)/2;center_y=(y_min+y_max)/2"
        or not _observation_pointer_matches(
            record.get("row_identity"),
            expected_ref=f"/{rowset_path}@{ROW_IDENTITY_CONTRACT_ATTR}",
            expected_sha256=contract.digest(),
        )
        or source is None
        or output is None
        or _metadata_dtype(source.data_type) != _metadata_dtype(output.data_type)
        or frame is None
        or frame.kind != SOURCE_CAMERA_FRAME_KIND
    ):
        issues.append(
            _issue(
                "BBOX_CENTER_DERIVATION_INVALID",
                "error",
                "BBox center derivation must bind exact image bbox/center payloads, source-camera frame, formula, and row identity.",
                rowset_path=rowset_path,
            )
        )
    return issues


def _crop_geometry_selection_issues(
    *,
    target: MetadataNode,
    record: Mapping[str, Any],
    contract: Any,
    temporal_digest: str | None,
    nodes: Mapping[str, MetadataNode],
) -> list[dict[str, Any]]:
    rowset_path = target.relative_path
    issues: list[dict[str, Any]] = []
    source = _as_mapping(record.get("source_rowset"))
    output = _as_mapping(record.get("output_rowset"))
    source_identity_ref = source.get("row_identity_ref")
    source_path, source_identity_attr = _canonical_v2_record_target(
        source_identity_ref
    )
    source_rowset = nodes.get(source_path or "")
    source_contract = None
    if source_rowset is not None and source_identity_attr == ROW_IDENTITY_CONTRACT_ATTR:
        try:
            source_contract = load_row_identity_contract_attrs(
                source_rowset.attributes
            )
        except RowIdentityContractError as exc:
            issues.append(
                _issue(
                    "CROP_GEOMETRY_SELECTION_INVALID",
                    "error",
                    "Crop selection source row identity is invalid.",
                    error=str(exc),
                )
            )
    selection_node, selection_issues = _observation_array_payload_metadata(
        record.get("selection"),
        role="crop_geometry_selection.selection",
        nodes=nodes,
        expected_path=f"{rowset_path}/detection_indices",
        expected_shape=(contract.leading_dimension,),
        expected_dtype="<i8",
    )
    issues.extend(selection_issues)

    output_specs = {
        "instance_key": ("<u8", (contract.leading_dimension,)),
        "source_acquisition_frame_index": (
            "<i8",
            (contract.leading_dimension,),
        ),
        "bbox_norm_coords": (None, (contract.leading_dimension, 4)),
        "bbox_img_xyxy": (None, (contract.leading_dimension, 4)),
        "centers_img_xy": (None, (contract.leading_dimension, 2)),
    }
    for name, (dtype, shape) in output_specs.items():
        _node, payload_issues = _observation_array_payload_metadata(
            output.get(name),
            role=f"crop_geometry_selection.output_rowset.{name}",
            nodes=nodes,
            expected_path=f"{rowset_path}/{name}",
            expected_shape=shape,
            expected_dtype=dtype,
            dtype_kind=("f" if name.startswith("bbox") or name == "centers_img_xy" else None),
        )
        issues.extend(payload_issues)

    if source_contract is not None and source_path is not None:
        source_specs = {
            "instance_key": ("<u8", (source_contract.leading_dimension,)),
            "source_acquisition_frame_index": (
                "<i8",
                (source_contract.leading_dimension,),
            ),
            "bbox_norm_coords": (None, (source_contract.leading_dimension, 4)),
            "bbox_img_xyxy": (None, (source_contract.leading_dimension, 4)),
            "centers_img_xy": (None, (source_contract.leading_dimension, 2)),
        }
        for name, (dtype, shape) in source_specs.items():
            _node, payload_issues = _observation_array_payload_metadata(
                source.get(name),
                role=f"crop_geometry_selection.source_rowset.{name}",
                nodes=nodes,
                expected_path=f"{source_path}/{name}",
                expected_shape=shape,
                expected_dtype=dtype,
                dtype_kind=("f" if name.startswith("bbox") or name == "centers_img_xy" else None),
            )
            issues.extend(payload_issues)
        issues.extend(
            _observation_source_temporal_authority_issues(
                rowset_path=source_path,
                contract=source_contract,
                nodes=nodes,
            )
        )

    expected_source_temporal_ref = (
        f"/{source_path}@{SOURCE_ROW_TEMPORAL_AUTHORITY_ATTR}"
        if source_path is not None
        else ""
    )
    source_temporal_digest = (
        source_rowset.attributes.get(SOURCE_ROW_TEMPORAL_AUTHORITY_DIGEST_ATTR)
        if source_rowset is not None
        else None
    )
    if (
        set(record)
        != {
            "schema_id",
            "schema_version",
            "operation",
            "source_rowset",
            "selection",
            "output_rowset",
        }
        or record.get("operation") != CROP_GEOMETRY_SELECTION_OPERATION
        or set(source)
        != {
            "row_identity_ref",
            "row_identity_sha256",
            "temporal_authority_ref",
            "temporal_authority_sha256",
            "instance_key",
            "source_acquisition_frame_index",
            "bbox_norm_coords",
            "bbox_img_xyxy",
            "centers_img_xy",
        }
        or set(output)
        != {
            "row_identity_ref",
            "row_identity_sha256",
            "temporal_authority_ref",
            "temporal_authority_sha256",
            "instance_key",
            "source_acquisition_frame_index",
            "bbox_norm_coords",
            "bbox_img_xyxy",
            "centers_img_xy",
        }
        or selection_node is None
        or source_contract is None
        or _canonical_observation_rowset_family(str(source_path or ""))
        != "detect_runs"
        or source.get("row_identity_sha256") != source_contract.digest()
        or source.get("temporal_authority_ref")
        != expected_source_temporal_ref
        or source.get("temporal_authority_sha256")
        != source_temporal_digest
        or output.get("row_identity_ref")
        != f"/{rowset_path}@{ROW_IDENTITY_CONTRACT_ATTR}"
        or output.get("row_identity_sha256") != contract.digest()
        or output.get("temporal_authority_ref")
        != f"/{rowset_path}@{SOURCE_ROW_TEMPORAL_AUTHORITY_ATTR}"
        or temporal_digest is None
        or output.get("temporal_authority_sha256") != temporal_digest
    ):
        issues.append(
            _issue(
                "CROP_GEOMETRY_SELECTION_INVALID",
                "error",
                "Crop selection must bind exact source/output row identity, time, canonical payload paths, and selection metadata.",
                rowset_path=rowset_path,
                source_rowset_path=source_path,
            )
        )
    return issues


def _crop_roi_geometry_derivation_issues(
    *,
    target: MetadataNode,
    record: Mapping[str, Any],
    contract: Any,
    nodes: Mapping[str, MetadataNode],
) -> list[dict[str, Any]]:
    rowset_path = target.relative_path
    issues: list[dict[str, Any]] = []
    placement, placement_issues = _observation_array_payload_metadata(
        record.get("source_crop_xywh"),
        role="crop_roi_geometry_derivation.source_crop_xywh",
        nodes=nodes,
        expected_path=f"{rowset_path}/source_crop_xywh",
        expected_shape=(contract.leading_dimension, 4),
        dtype_kind="iuf",
    )
    bbox_roi, bbox_roi_issues = _observation_array_payload_metadata(
        record.get("bbox_roi_xyxy"),
        role="crop_roi_geometry_derivation.bbox_roi_xyxy",
        nodes=nodes,
        expected_path=f"{rowset_path}/bbox_roi_xyxy",
        expected_shape=(contract.leading_dimension, 4),
        dtype_kind="f",
    )
    bbox_img, bbox_img_issues = _observation_array_payload_metadata(
        record.get("bbox_img_xyxy"),
        role="crop_roi_geometry_derivation.bbox_img_xyxy",
        nodes=nodes,
        expected_path=f"{rowset_path}/bbox_img_xyxy",
        expected_shape=(contract.leading_dimension, 4),
        dtype_kind="f",
    )
    issues.extend(placement_issues)
    issues.extend(bbox_roi_issues)
    issues.extend(bbox_img_issues)

    ownership_pointer = _as_mapping(record.get("crop_placement_ownership"))
    ownership_target, ownership_issues = _canonical_v2_bound_record_issues(
        ownership_pointer,
        role="crop_roi_geometry_derivation.crop_placement_ownership",
        nodes=nodes,
    )
    issues.extend(ownership_issues)
    ownership = None
    _ownership_path, ownership_attr = _canonical_v2_record_target(
        ownership_pointer.get("record_ref")
    )
    if ownership_target is not None and ownership_attr == CROP_PLACEMENT_OWNERSHIP_ATTR:
        try:
            ownership = parse_crop_placement_ownership(
                ownership_target.attributes.get(CROP_PLACEMENT_OWNERSHIP_ATTR)
            )
        except PixelFrameAuthorityError as exc:
            issues.append(
                _issue(
                    "CROP_ROI_GEOMETRY_DERIVATION_INVALID",
                    "error",
                    "Crop placement ownership fails its strict schema.",
                    error=str(exc),
                )
            )

    roi_pointer = _as_mapping(record.get("roi_frame"))
    roi_frame, _roi_node, roi_issues = _pixel_frame_record_metadata(
        record_ref=roi_pointer.get("record_ref"),
        record_sha256=roi_pointer.get("record_sha256"),
        role="crop_roi_geometry_derivation.roi_frame",
        nodes=nodes,
    )
    issues.extend(roi_issues)
    transforms: list[Any] = []
    raw_chain = record.get("transform_chain")
    if type(raw_chain) is list:
        for raw_pointer in raw_chain:
            pointer = _as_mapping(raw_pointer)
            path, attr = _canonical_v2_record_target(pointer.get("record_ref"))
            transform_node = nodes.get(path or "")
            if transform_node is None or attr != DIRECTED_TRANSFORM_V2_ATTR:
                continue
            transform, transform_issues = _parse_directed_transform_v2_node(
                transform_node,
                record_ref=str(pointer.get("record_ref")),
                nodes=nodes,
            )
            issues.extend(transform_issues)
            if transform is not None and pointer.get("record_sha256") == transform.digest():
                transforms.append(transform)
    transform_valid = (
        len(transforms) == 1
        and transforms[0].kind == AFFINE_2D_ROWWISE_KIND
        and transforms[0].row_identity is not None
        and transforms[0].row_identity.record_ref
        == f"/{rowset_path}@{ROW_IDENTITY_CONTRACT_ATTR}"
        and transforms[0].row_identity.record_sha256 == contract.digest()
        and roi_frame is not None
        and transforms[0].source.record_ref == roi_pointer.get("record_ref")
        and transforms[0].target.space_id == SOURCE_CAMERA_IMAGE_SPACE_ID
    )
    if (
        set(record)
        != {
            "schema_id",
            "schema_version",
            "operation",
            "source_crop_xywh",
            "crop_placement_ownership",
            "roi_frame",
            "direction",
            "transform_chain",
            "bbox_roi_xyxy",
            "bbox_img_xyxy",
            "row_identity",
            "formula",
        }
        or record.get("operation") != CROP_ROI_GEOMETRY_DERIVATION_OPERATION
        or record.get("direction")
        != "roi_local_px_to_source_camera_image_px"
        or record.get("formula")
        != "apply_exact_rowwise_crop_placement_to_each_xyxy_corner_v1"
        or not _observation_pointer_matches(
            record.get("row_identity"),
            expected_ref=f"/{rowset_path}@{ROW_IDENTITY_CONTRACT_ATTR}",
            expected_sha256=contract.digest(),
        )
        or placement is None
        or bbox_roi is None
        or bbox_img is None
        or ownership is None
        or ownership_pointer.get("record_sha256") != ownership.digest()
        or ownership_pointer.get("record_ref")
        != f"/{rowset_path}/source_crop_xywh@{CROP_PLACEMENT_OWNERSHIP_ATTR}"
        or roi_frame is None
        or roi_frame.kind != ROI_FRAME_KIND
        or not transform_valid
    ):
        issues.append(
            _issue(
                "CROP_ROI_GEOMETRY_DERIVATION_INVALID",
                "error",
                "Crop ROI derivation must bind exact placement/ROI/image payloads, ownership, frame, row identity, and ROI-to-camera transform direction.",
                rowset_path=rowset_path,
            )
        )
    return issues


def _observation_coordinate_record_semantic_issues(
    *,
    descriptor: Mapping[str, Any],
    surface_node: MetadataNode,
    contract: Any,
    nodes: Mapping[str, MetadataNode],
) -> list[dict[str, Any]]:
    """Validate the canonical detection/crop record graph metadata-first."""

    if contract is None or contract.domain != OBSERVATION_INSTANCE_DOMAIN:
        return []
    identity_pointer = _as_mapping(descriptor.get("row_identity"))
    rowset_path, identity_attr = _canonical_v2_record_target(
        identity_pointer.get("record_ref")
    )
    if rowset_path is None:
        return []
    family = _canonical_observation_rowset_family(rowset_path)
    records = _registered_lineage_records(descriptor, nodes=nodes)
    has_registered_lineage = bool(records)
    if family is None and not has_registered_lineage:
        return []

    issues: list[dict[str, Any]] = []
    expected_parent = PurePosixPath(surface_node.relative_path).parent.as_posix()
    if (
        family is None
        or identity_attr != ROW_IDENTITY_CONTRACT_ATTR
        or expected_parent != rowset_path
    ):
        issues.append(
            _issue(
                "OBSERVATION_COORDINATE_ROWSET_PATH_INVALID",
                "error",
                "Canonical detection/crop arrays and observation identity must share one exact top-level run rowset.",
                surface_path=surface_node.relative_path,
                rowset_path=rowset_path,
                rowset_family=family,
            )
        )

    temporal_issues = _observation_source_temporal_authority_issues(
        rowset_path=rowset_path,
        contract=contract,
        nodes=nodes,
    )
    issues.extend(temporal_issues)
    rowset = nodes.get(rowset_path)
    temporal_digest = (
        rowset.attributes.get(SOURCE_ROW_TEMPORAL_AUTHORITY_DIGEST_ATTR)
        if rowset is not None
        else None
    )

    for schema_id, occurrences in records.items():
        rule = _REGISTERED_OBSERVATION_COORDINATE_RECORDS[schema_id]
        for pointer, target, record in occurrences:
            expected_family = str(rule["owner_family"])
            actual_family = _canonical_observation_rowset_family(
                target.relative_path
            )
            if expected_family != "observation_rowset" and actual_family != expected_family:
                issues.append(
                    _issue(
                        "REGISTERED_COORDINATE_RECORD_OWNER_INVALID",
                        "error",
                        "Registered observation coordinate record is stored outside its canonical run family.",
                        record_ref=pointer.get("record_ref"),
                        expected_owner_family=expected_family,
                        actual_owner_family=actual_family,
                    )
                )
            if schema_id == DETECTION_ACQUISITION_MAPPING_SCHEMA_ID:
                issues.extend(
                    _detection_acquisition_mapping_issues(
                        target=target,
                        record=record,
                        nodes=nodes,
                    )
                )
            elif schema_id == DETECTION_BBOX_PROJECTION_SCHEMA_ID:
                try:
                    target_contract = (
                        contract
                        if target.relative_path == rowset_path
                        else load_row_identity_contract_attrs(target.attributes)
                    )
                except RowIdentityContractError as exc:
                    issues.append(
                        _issue(
                            "DETECTION_BBOX_PROJECTION_INVALID",
                            "error",
                            "Detection projection owner row identity is invalid.",
                            record_ref=pointer.get("record_ref"),
                            error=str(exc),
                        )
                    )
                    continue
                issues.extend(
                    _detection_bbox_projection_issues(
                        target=target,
                        record=record,
                        contract=target_contract,
                        temporal_digest=(
                            temporal_digest
                            if target.relative_path == rowset_path
                            else target.attributes.get(
                                SOURCE_ROW_TEMPORAL_AUTHORITY_DIGEST_ATTR
                            )
                        ),
                        nodes=nodes,
                    )
                )
            elif schema_id == BBOX_CENTER_DERIVATION_SCHEMA_ID:
                try:
                    target_contract = (
                        contract
                        if target.relative_path == rowset_path
                        else load_row_identity_contract_attrs(target.attributes)
                    )
                except RowIdentityContractError as exc:
                    issues.append(
                        _issue(
                            "BBOX_CENTER_DERIVATION_INVALID",
                            "error",
                            "BBox-center owner row identity is invalid.",
                            record_ref=pointer.get("record_ref"),
                            error=str(exc),
                        )
                    )
                    continue
                issues.extend(
                    _bbox_center_derivation_issues(
                        target=target,
                        record=record,
                        contract=target_contract,
                        nodes=nodes,
                    )
                )
            elif schema_id == CROP_GEOMETRY_SELECTION_SCHEMA_ID:
                issues.extend(
                    _crop_geometry_selection_issues(
                        target=target,
                        record=record,
                        contract=contract,
                        temporal_digest=temporal_digest,
                        nodes=nodes,
                    )
                )
            elif schema_id == CROP_ROI_GEOMETRY_DERIVATION_SCHEMA_ID:
                issues.extend(
                    _crop_roi_geometry_derivation_issues(
                        target=target,
                        record=record,
                        contract=contract,
                        nodes=nodes,
                    )
                )

    leaf = PurePosixPath(surface_node.relative_path).name
    required: set[str] = set()
    if family == "detect_runs":
        required = {
            DETECTION_ACQUISITION_MAPPING_SCHEMA_ID,
            DETECTION_BBOX_PROJECTION_SCHEMA_ID,
        }
        if leaf == "centers_img_xy":
            required.add(BBOX_CENTER_DERIVATION_SCHEMA_ID)
    elif family == "crop_runs":
        required = {CROP_GEOMETRY_SELECTION_SCHEMA_ID}
        if leaf in {"source_crop_xywh", "bbox_roi_xyxy"}:
            required.add(CROP_ROI_GEOMETRY_DERIVATION_SCHEMA_ID)
        else:
            required.update(
                {
                    DETECTION_ACQUISITION_MAPPING_SCHEMA_ID,
                    DETECTION_BBOX_PROJECTION_SCHEMA_ID,
                }
            )
            if leaf == "centers_img_xy":
                required.add(BBOX_CENTER_DERIVATION_SCHEMA_ID)
    invalid_counts = {
        schema_id: len(records.get(schema_id, ()))
        for schema_id in sorted(required)
        if len(records.get(schema_id, ())) != 1
    }
    if invalid_counts:
        issues.append(
            _issue(
                "OBSERVATION_COORDINATE_LINEAGE_REQUIRED",
                "error",
                "Canonical detection/crop surfaces must bind each required registered lineage record exactly once.",
                surface_path=surface_node.relative_path,
                invalid_schema_counts=invalid_counts,
            )
        )

    if not any(
        issue["severity"] in {"error", "critical"} for issue in issues
    ):
        payloads = sorted(
            {
                str(payload.get("array_ref") or payload.get("ref"))
                for occurrences in records.values()
                for _pointer, _target, record in occurrences
                for payload in (
                    _as_mapping(value)
                    for value in record.values()
                    if isinstance(value, Mapping)
                )
                if payload.get("content_sha256") is not None
            }
        )
        issues.append(
            _issue(
                "OBSERVATION_COORDINATE_PAYLOAD_VALIDATION_REQUIRED",
                "warning",
                "Observation record metadata is exact, but array payload hashes and numerical projection/selection equalities require live validation.",
                surface_path=surface_node.relative_path,
                payload_refs=payloads,
            )
        )
    return issues


def _canonical_v2_descriptor_integrity_issues(
    *,
    match: DescriptorMatch,
    surface_node: MetadataNode,
    nodes: Mapping[str, MetadataNode],
) -> tuple[list[dict[str, Any]], str | None]:
    issues: list[dict[str, Any]] = []
    try:
        descriptor = parse_canonical_coordinate_descriptor(match.descriptor).to_dict()
    except CoordinateDescriptorError as exc:
        return (
            [
                _issue(
                    "COORDINATE_DESCRIPTOR_INTEGRITY_INVALID",
                    "error",
                    "Canonical v2 descriptor could not be parsed by the shared strict parser.",
                    descriptor_source=match.source,
                    validation_issues=[
                        {
                            "code": item.code,
                            "path": item.path,
                            "message": item.message,
                        }
                        for item in exc.issues
                    ],
                )
            ],
            None,
        )
    contract, row_identity_domain, identity_issues = (
        _canonical_v2_row_identity_issues(
            descriptor=descriptor,
            match=match,
            surface_node=surface_node,
            nodes=nodes,
        )
    )
    issues.extend(identity_issues)
    owner = nodes.get(match.owner_path)
    if owner is None or match.attr_name is None or contract is None:
        if owner is None or match.attr_name is None:
            issues.append(
                _issue(
                    "COORDINATE_DESCRIPTOR_DIGEST_MISSING",
                    "error",
                    "Canonical v2 descriptor must be an array-specific attr with its digest.",
                    descriptor_source=match.source,
                )
            )
    else:
        record_ref = _as_mapping(descriptor.get("row_identity")).get(
            "record_ref"
        )
        owner_shape = tuple(surface_node.shape or ())
        try:
            load_canonical_coordinate_descriptor_attrs(
                owner.attributes,
                row_identity_contract=contract,
                expected_row_identity_record_ref=str(record_ref),
                owner_shape=owner_shape,
                attr_name=match.attr_name,
            )
        except CoordinateDescriptorError as exc:
            codes = {item.code for item in exc.issues}
            issue_code = (
                "COORDINATE_DESCRIPTOR_DIGEST_MISMATCH"
                if "descriptor_digest_mismatch" in codes
                else "COORDINATE_DESCRIPTOR_DIGEST_MISSING"
                if "descriptor_digest_missing" in codes
                else "COORDINATE_DESCRIPTOR_INTEGRITY_INVALID"
            )
            issues.append(
                _issue(
                    issue_code,
                    "error",
                    "Canonical v2 descriptor failed strict digest, identity, or row-count loading.",
                    descriptor_source=match.source,
                    validation_issues=[
                        {
                            "code": item.code,
                            "path": item.path,
                            "message": item.message,
                        }
                        for item in exc.issues
                    ],
                )
            )

    issues.extend(
        _canonical_v2_reference_authority_issues(
            descriptor,
            nodes=nodes,
        )
    )
    issues.extend(_canonical_v2_pixel_frame_issues(descriptor, nodes=nodes))
    bound_records: list[tuple[str, Mapping[str, Any]]] = []
    for raw in descriptor.get("lineage_refs") or ():
        if isinstance(raw, Mapping):
            bound_records.append(("lineage", raw))
    frame_record = _as_mapping(descriptor.get("frame_record"))
    if frame_record:
        bound_records.append(("frame_record", frame_record))
    for role, record in bound_records:
        _target, record_issues = _canonical_v2_bound_record_issues(
            record,
            role=role,
            nodes=nodes,
        )
        issues.extend(record_issues)
    if descriptor.get("space_id") == "physical_mm" and frame_record:
        issues.extend(
            _canonical_physical_frame_record_issues(
                descriptor,
                nodes=nodes,
            )
        )
    if descriptor.get("space_id") == "fish_anatomical_body_frame" and frame_record:
        body_record = _canonical_v2_record_value(
            record_ref=frame_record.get("record_ref"),
            nodes=nodes,
        )
        schema_id = body_record.get("schema_id") or body_record.get(
            "body_frame_schema_id"
        )
        schema_version = body_record.get("schema_version") or body_record.get(
            "body_frame_schema_version"
        )
        if schema_id != "fish_anatomical_body_frame" or schema_version != 1:
            issues.append(
                _issue(
                    "BODY_FRAME_RECORD_SCHEMA_INVALID",
                    "error",
                    "Fish-anatomical coordinates require a digest-bound fish_anatomical_body_frame schema-v1 record.",
                    record_ref=frame_record.get("record_ref"),
                    schema_id=schema_id,
                    schema_version=schema_version,
                )
            )
    issues.extend(
        _observation_coordinate_record_semantic_issues(
            descriptor=descriptor,
            surface_node=surface_node,
            contract=contract,
            nodes=nodes,
        )
    )
    issues.extend(_canonical_v2_transform_issues(descriptor, nodes=nodes))
    return issues, row_identity_domain


def _descriptor_integrity_issues(
    *,
    match: DescriptorMatch,
    surface_node: MetadataNode,
    nodes: Mapping[str, MetadataNode],
) -> tuple[list[dict[str, Any]], str | None]:
    if (
        match.descriptor.get("schema_version")
        == CANONICAL_COORDINATE_DESCRIPTOR_SCHEMA_VERSION
    ):
        return _canonical_v2_descriptor_integrity_issues(
            match=match,
            surface_node=surface_node,
            nodes=nodes,
        )
    return _historical_descriptor_integrity_issues(
        match=match,
        surface_node=surface_node,
        nodes=nodes,
    )


def _legacy_context_is_valid(
    *,
    legacy_label: str,
    evidence: Mapping[str, Any],
    nodes: Mapping[str, MetadataNode],
) -> tuple[bool, str | None]:
    label = legacy_label.strip().lower()
    expected = {
        "camera": "source_camera_image_px",
        "texture": "stimulus_texture_px",
    }.get(label)
    if expected is None:
        return False, f"unsupported legacy label {legacy_label!r}"
    width = _value(evidence, "reference_width")
    height = _value(evidence, "reference_height")
    units = _value(evidence, "units")
    origin = _value(evidence, "origin")
    x_axis = _value(evidence, "x_axis_direction")
    y_axis = _value(evidence, "y_axis_direction")
    pixel_convention = _value(evidence, "pixel_convention")
    geometry_convention = _value(evidence, "geometry_convention")
    overlay = _normalized_conflict_value(
        "source_camera_overlay",
        _value(evidence, "source_camera_overlay_suitable"),
    )
    allowed_pixel_conventions = {
        "camera": {"pixel_center"},
        "texture": {"pixel_center", "continuous"},
    }[label]
    allowed_overlay = {
        "camera": {"direct"},
        "texture": {"requires_transform", "not_suitable"},
    }[label]
    controlled_values_valid = (
        str(units) == "px"
        and origin == "top_left"
        and x_axis == "right"
        and y_axis == "down"
        and pixel_convention in allowed_pixel_conventions
        and geometry_convention
        in {
            "point_xy",
            "points_xy",
            "xy_point",
            "bbox_xyxy",
            "bbox_xywh",
            "coordinate_component",
            "raster_yx",
        }
        and overlay in allowed_overlay
    )
    if not controlled_values_valid:
        return (
            False,
            "legacy origin/axes/units/pixel convention/geometry/overlay values "
            "do not match the controlled compatibility vocabulary",
        )
    authority = _value(evidence, "reference_authority")
    raw_evidence = _value(evidence, "source_ref") or _value(evidence, "transform_ref")
    refs: list[CoordinateRecordRef] = []
    candidates = raw_evidence if isinstance(raw_evidence, (list, tuple)) else [raw_evidence]
    for candidate in candidates:
        if isinstance(candidate, Mapping) and isinstance(candidate.get("ref"), str):
            refs.append(
                CoordinateRecordRef(
                    ref=str(candidate["ref"]),
                    sha256=(
                        str(candidate["sha256"])
                        if isinstance(candidate.get("sha256"), str)
                        else None
                    ),
                )
            )
        elif isinstance(candidate, str) and candidate.strip():
            refs.append(CoordinateRecordRef(ref=candidate.strip()))
    authority_issues = _reference_authority_issues(
        authority=authority,
        reference_width=width,
        reference_height=height,
        reference_units=units,
        space_id=expected,
        nodes=nodes,
    )
    if authority_issues:
        return False, "; ".join(
            f"{issue['code']}: {issue['message']}" for issue in authority_issues
        )
    ref_payloads = [ref.to_dict() for ref in refs]
    ref_issues = _validate_record_refs(
        {"lineage_refs": ref_payloads},
        field_name="lineage_refs",
        nodes=nodes,
    )
    if ref_issues:
        return False, "; ".join(
            f"{issue['code']}: {issue['message']}" for issue in ref_issues
        )
    try:
        resolve_legacy_space_id(
            label,
            context=LegacySpaceContext(
                canonical_space_id=expected,
                reference_width=width,
                reference_height=height,
                reference_units=str(units or ""),
                reference_authority=str(authority or ""),
                evidence_refs=tuple(refs),
            ),
        )
    except (CoordinateDescriptorError, TypeError, ValueError) as exc:
        return False, str(exc)
    return True, None


_LEGACY_COMPATIBILITY_EVIDENCE_FIELDS = {
    "schema_id",
    "schema_version",
    "legacy_label",
    "canonical_space_id",
    "surface_path",
    "row_count",
    "values_sha256",
    "validation_tool_commit",
    "numerical_invariants",
    "values_changed",
}


def _legacy_numerical_invariant_proof(
    *,
    legacy_label: str,
    node: MetadataNode,
    evidence: Mapping[str, Any],
    nodes: Mapping[str, MetadataNode],
) -> tuple[bool, str | None, Mapping[str, Any] | None]:
    expected_space = {
        "camera": "source_camera_image_px",
        "texture": "stimulus_texture_px",
    }.get(legacy_label.strip().lower())
    raw_refs = _value(evidence, "source_ref")
    refs = raw_refs if isinstance(raw_refs, (list, tuple)) else [raw_refs]
    candidates: list[Mapping[str, Any]] = []
    for raw in refs:
        if not isinstance(raw, Mapping) or not isinstance(raw.get("ref"), str):
            continue
        node_ref, selector_kind, selector = _record_ref_target(str(raw["ref"]))
        target = nodes.get(_normalize_archive_ref(node_ref) or "")
        if target is None or selector_kind not in {"attr", "fragment"}:
            continue
        record = _as_mapping(target.attributes.get(str(selector)))
        if record.get("schema_id") == "palette.legacy_coordinate_compatibility_evidence":
            candidates.append(record)
    if len(candidates) != 1:
        return (
            False,
            "exactly one digest-bound legacy compatibility evidence record is required",
            None,
        )
    record = candidates[0]
    if set(record) != _LEGACY_COMPATIBILITY_EVIDENCE_FIELDS:
        return False, "legacy compatibility evidence fields are not exact", record
    invariants = _as_mapping(record.get("numerical_invariants"))
    required_invariants = {
        "reference_extent_verified": True,
        "row_identity_verified": True,
        "values_finite_or_declared_missing": True,
        "values_preserved": True,
    }
    row_count = node.shape[0] if isinstance(node.shape, (list, tuple)) and node.shape else None
    valid = (
        record.get("schema_version") == 1
        and record.get("legacy_label") == legacy_label.strip().lower()
        and record.get("canonical_space_id") == expected_space
        and _normalize_archive_ref(str(record.get("surface_path") or ""))
        == _normalize_archive_ref(node.relative_path)
        and record.get("row_count") == row_count
        and isinstance(record.get("values_sha256"), str)
        and _SHA256_HEX_RE.fullmatch(str(record.get("values_sha256"))) is not None
        and isinstance(record.get("validation_tool_commit"), str)
        and bool(str(record.get("validation_tool_commit")).strip())
        and invariants == required_invariants
        and record.get("values_changed") is False
    )
    if not valid:
        return (
            False,
            "legacy compatibility evidence does not prove exact numerical invariants",
            record,
        )
    return True, None, record


def _legacy_reference_issues(
    *,
    evidence: Mapping[str, Any],
    nodes: Mapping[str, MetadataNode],
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for evidence_field, descriptor_field in (
        ("source_ref", "lineage_refs"),
        ("transform_ref", "transform_refs"),
    ):
        raw = _value(evidence, evidence_field)
        if raw in (None, ""):
            continue
        values = raw if isinstance(raw, (list, tuple)) else [raw]
        refs: list[dict[str, Any]] = []
        for value in values:
            if isinstance(value, Mapping) and isinstance(value.get("ref"), str):
                refs.append(dict(value))
            elif isinstance(value, str) and value.strip():
                refs.append({"ref": value.strip()})
        if refs:
            issues.extend(
                _validate_record_refs(
                    {descriptor_field: refs},
                    field_name=descriptor_field,
                    nodes=nodes,
                )
            )
    return issues


_LEGACY_IDENTITY_RULES_BY_DOMAIN = {
    OBSERVATION_INSTANCE_DOMAIN: {
        "canonical_name": INSTANCE_KEY_ARRAY_REF,
        "historical": {
            INSTANCE_KEY_ARRAY_REF: (
                "instance_key",
                "palette.instance_key_row_identity",
            )
        },
    },
    TRACK_SAMPLE_DOMAIN: {
        "canonical_name": TRACK_SAMPLE_KEY_ARRAY_REF,
        "historical": {
            "frame_indices": ("track_frame_indices", "palette.track_row_identity")
        },
    },
    STIMULUS_STATE_DOMAIN: {
        "canonical_name": STIMULUS_STATE_KEY_ARRAY_REF,
        "historical": {
            "coordinate_row_identity": (
                "explicit_array",
                "palette.coordinate_row_identity",
            )
        },
    },
}


def _legacy_identity_expected_domain(surface_type: str) -> str | None:
    profile = _SURFACE_PROFILES.get(surface_type)
    if profile is None or profile.rowless or len(profile.row_identity_domains) != 1:
        return None
    domain = next(iter(profile.row_identity_domains))
    return domain if domain in _LEGACY_IDENTITY_RULES_BY_DOMAIN else None


def _canonical_identity_candidate_issues(
    *,
    expected_domain: str,
    candidate_path: str,
    candidate: MetadataNode,
    surface_node: MetadataNode,
    nodes: Mapping[str, MetadataNode],
) -> list[dict[str, Any]]:
    parent_path = PurePosixPath(candidate_path).parent.as_posix()
    if parent_path == ".":
        parent_path = ""
    parent = nodes.get(parent_path)
    if parent is None or ROW_IDENTITY_CONTRACT_ATTR not in parent.attributes:
        return [
            _issue(
                "LEGACY_ROW_IDENTITY_UNTYPED",
                "error",
                "A canonical identity key is not typed without an exact immediate-parent row_identity_contract.",
                row_identity_path=candidate_path,
                expected_domain=expected_domain,
                expected_contract_owner=parent_path,
            )
        ]
    try:
        contract = parse_row_identity_contract(
            parent.attributes[ROW_IDENTITY_CONTRACT_ATTR]
        )
        loaded = load_row_identity_contract_attrs(parent.attributes)
    except RowIdentityContractError as exc:
        return [
            _issue(
                "LEGACY_ROW_IDENTITY_UNTYPED",
                "error",
                "The immediate-parent canonical identity contract is invalid or not digest-bound.",
                row_identity_path=candidate_path,
                expected_domain=expected_domain,
                validation_issues=[
                    {
                        "code": item.code,
                        "path": item.path,
                        "message": item.message,
                    }
                    for item in exc.issues
                ],
            )
        ]
    expected_name = str(
        _LEGACY_IDENTITY_RULES_BY_DOMAIN[expected_domain]["canonical_name"]
    )
    issues: list[dict[str, Any]] = []
    if (
        loaded != contract
        or contract.domain != expected_domain
        or contract.key_array.ref != expected_name
        or PurePosixPath(candidate_path).name != expected_name
    ):
        issues.append(
            _issue(
                "LEGACY_ROW_IDENTITY_UNTYPED",
                "error",
                "Canonical key name, owning contract domain, and contract key reference must agree exactly.",
                row_identity_path=candidate_path,
                expected_domain=expected_domain,
                actual_domain=contract.domain,
                expected_key_name=expected_name,
                contract_key_ref=contract.key_array.ref,
            )
        )
        return issues
    issues.extend(
        _track_sample_time_lineage_issues(
            contract=contract,
            contract_owner_path=parent_path,
            nodes=nodes,
        )
    )
    try:
        load_row_identity_key_attrs(candidate.attributes, contract=contract)
    except RowIdentityContractError as exc:
        issues.append(
            _issue(
                "LEGACY_ROW_IDENTITY_UNTYPED",
                "error",
                "Canonical key metadata is not the exact digest-bound record required by its owner.",
                row_identity_path=candidate_path,
                validation_issues=[
                    {
                        "code": item.code,
                        "path": item.path,
                        "message": item.message,
                    }
                    for item in exc.issues
                ],
            )
        )
    dtype = _metadata_dtype(candidate.data_type)
    if (
        dtype is None
        or dtype.str != contract.key_array.dtype
        or tuple(candidate.shape or ()) != contract.key_array.shape
    ):
        issues.append(
            _issue(
                "LEGACY_ROW_IDENTITY_UNTYPED",
                "error",
                "Canonical key array metadata disagrees with its exact owning contract.",
                row_identity_path=candidate_path,
                metadata_shape=candidate.shape,
                metadata_dtype=(dtype.str if dtype is not None else candidate.data_type),
                contract_shape=list(contract.key_array.shape),
                contract_dtype=contract.key_array.dtype,
            )
        )
    surface_count, _component_counts = _surface_leading_dimension(
        surface_node,
        nodes=nodes,
        excluded_paths={candidate_path},
    )
    if surface_count is not None and surface_count != contract.leading_dimension:
        issues.append(
            _issue(
                "ROW_IDENTITY_LENGTH_MISMATCH",
                "error",
                "Canonical identity leading dimension does not match the coordinate surface.",
                row_identity_path=candidate_path,
                row_count=contract.leading_dimension,
                surface_count=surface_count,
            )
        )
    return issues


def _historical_identity_candidate_issues(
    *,
    expected_domain: str,
    candidate_path: str,
    candidate: MetadataNode,
    surface_node: MetadataNode,
    nodes: Mapping[str, MetadataNode],
) -> list[dict[str, Any]]:
    leaf = PurePosixPath(candidate_path).name
    historical = _LEGACY_IDENTITY_RULES_BY_DOMAIN[expected_domain]["historical"]
    mode, schema_id = historical[leaf]
    issues = _row_identity_schema_issues(
        mode=mode,
        component_name=leaf,
        resolved=candidate_path,
        row_node=candidate,
    )
    dtype = _metadata_dtype(candidate.data_type)
    rank = len(candidate.shape) if isinstance(candidate.shape, (list, tuple)) else None
    shape_valid = (
        rank == 1 and dtype == np.dtype("uint64")
        if expected_domain == OBSERVATION_INSTANCE_DOMAIN
        else rank == 1 and dtype is not None and dtype.kind in "iu"
        if expected_domain == TRACK_SAMPLE_DOMAIN
        else rank in {1, 2} and dtype == np.dtype("int64")
    )
    if candidate.attributes.get("schema_id") != schema_id or not shape_valid:
        issues.append(
            _issue(
                "LEGACY_ROW_IDENTITY_UNTYPED",
                "error",
                "Historical row identity must match the exact family name, schema, rank, and dtype.",
                row_identity_path=candidate_path,
                expected_domain=expected_domain,
                expected_schema_id=schema_id,
                actual_schema_id=candidate.attributes.get("schema_id"),
                shape=candidate.shape,
                dtype=candidate.data_type,
            )
        )
    surface_count, _component_counts = _surface_leading_dimension(
        surface_node,
        nodes=nodes,
        excluded_paths={candidate_path},
    )
    row_count = (
        int(candidate.shape[0])
        if isinstance(candidate.shape, (list, tuple)) and candidate.shape
        else None
    )
    if surface_count is not None and row_count != surface_count:
        issues.append(
            _issue(
                "ROW_IDENTITY_LENGTH_MISMATCH",
                "error",
                "Historical row identity leading dimension does not match the coordinate surface.",
                row_identity_path=candidate_path,
                row_count=row_count,
                surface_count=surface_count,
            )
        )
    return issues


def _legacy_row_identity_resolution(
    *,
    surface_type: str,
    node: MetadataNode,
    nodes: Mapping[str, MetadataNode],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Resolve descriptor-free identity without sibling order or shape guessing."""

    profile = _SURFACE_PROFILES.get(surface_type)
    if profile is not None and profile.rowless:
        return None, []
    expected_domain = _legacy_identity_expected_domain(surface_type)
    if expected_domain is None:
        return None, [
            _issue(
                "ROW_IDENTITY_REF_UNRESOLVED",
                "error",
                "This surface family has no exact descriptor-free identity compatibility rule.",
                surface_type=surface_type,
            )
        ]
    rules = _LEGACY_IDENTITY_RULES_BY_DOMAIN[expected_domain]
    canonical_name = str(rules["canonical_name"])
    allowed_names = {canonical_name, *rules["historical"]}
    parent = (
        PurePosixPath(node.relative_path).parent.as_posix()
        if node.node_type == "array"
        else node.relative_path
    )
    if parent == ".":
        parent = ""
    candidate_paths = {
        f"{parent}/{name}" if parent else name
        for name in allowed_names
        if (f"{parent}/{name}" if parent else name) in nodes
    }
    explicit = _find_declared(node, nodes, _ROW_IDENTITY_KEYS)
    explicit_resolved: str | None = None
    if explicit:
        raw_ref = explicit[0]
        if isinstance(raw_ref, Mapping):
            raw_ref = raw_ref.get("array_ref") or raw_ref.get("ref")
        if isinstance(raw_ref, str):
            explicit_resolved = _normalize_archive_ref(
                raw_ref,
                owner_path=node.relative_path,
                owner_is_array=node.node_type == "array",
            )
            if explicit_resolved is not None:
                candidate_paths.add(explicit_resolved)

    issues: list[dict[str, Any]] = []
    if expected_domain == TRACK_SAMPLE_DOMAIN:
        source_instance_path = f"{parent}/instance_key" if parent else "instance_key"
        if source_instance_path in nodes:
            issues.append(
                _issue(
                    "TRACK_SOURCE_INSTANCE_KEY_IS_LINEAGE_ONLY",
                    "warning",
                    "A track's source instance_key is observation lineage, not track-sample row identity.",
                    instance_key_path=source_instance_path,
                    required_identity_names=sorted(allowed_names),
                )
            )

    valid: list[str] = []
    untyped: list[dict[str, Any]] = []
    candidate_warnings: list[dict[str, Any]] = []
    for candidate_path in sorted(candidate_paths):
        candidate = nodes.get(candidate_path)
        leaf = PurePosixPath(candidate_path).name
        if (
            candidate is None
            or candidate.node_type != "array"
            or leaf not in allowed_names
            or PurePosixPath(candidate_path).parent.as_posix()
            != (parent if parent else ".")
        ):
            untyped.append(
                _issue(
                    "LEGACY_ROW_IDENTITY_UNTYPED",
                    "error",
                    "A legacy identity declaration must resolve to an exact allowed sibling array for this surface family.",
                    row_identity_path=candidate_path,
                    expected_domain=expected_domain,
                    allowed_names=sorted(allowed_names),
                    explicit_declaration=(candidate_path == explicit_resolved),
                )
            )
            continue
        parent_path = PurePosixPath(candidate_path).parent.as_posix()
        if parent_path == ".":
            parent_path = ""
        owner = nodes.get(parent_path)
        has_contract = bool(
            owner and ROW_IDENTITY_CONTRACT_ATTR in owner.attributes
        )
        if leaf == canonical_name and has_contract:
            candidate_issues = _canonical_identity_candidate_issues(
                expected_domain=expected_domain,
                candidate_path=candidate_path,
                candidate=candidate,
                surface_node=node,
                nodes=nodes,
            )
        elif leaf in rules["historical"]:
            candidate_issues = _historical_identity_candidate_issues(
                expected_domain=expected_domain,
                candidate_path=candidate_path,
                candidate=candidate,
                surface_node=node,
                nodes=nodes,
            )
        else:
            candidate_issues = [
                _issue(
                    "LEGACY_ROW_IDENTITY_UNTYPED",
                    "error",
                    "A canonical identity key without its immediate-parent contract is untyped.",
                    row_identity_path=candidate_path,
                    expected_domain=expected_domain,
                )
            ]
        if any(
            item["severity"] in {"error", "critical"}
            for item in candidate_issues
        ):
            untyped.extend(candidate_issues)
        else:
            valid.append(candidate_path)
            candidate_warnings.extend(candidate_issues)

    issues.extend(candidate_warnings)
    issues.extend(untyped)
    if len(valid) > 1:
        issues.append(
            _issue(
                "LEGACY_ROW_IDENTITY_AMBIGUOUS",
                "error",
                "Multiple typed row-identity siblings are eligible and the surface does not bind one explicitly enough to choose safely.",
                expected_domain=expected_domain,
                candidate_paths=valid,
            )
        )
        return None, issues
    if len(valid) == 1 and not untyped:
        return valid[0], issues
    if not valid and not untyped:
        issues.append(
            _issue(
                "ROW_IDENTITY_REF_UNRESOLVED",
                "error",
                "No exact typed row identity is persisted for this surface family.",
                expected_domain=expected_domain,
                allowed_names=sorted(allowed_names),
                explicit_ref=explicit_resolved,
            )
        )
    return None, issues


def _legacy_row_identity_issues(
    *,
    surface_type: str,
    node: MetadataNode,
    nodes: Mapping[str, MetadataNode],
) -> list[dict[str, Any]]:
    _resolved, issues = _legacy_row_identity_resolution(
        surface_type=surface_type,
        node=node,
        nodes=nodes,
    )
    return issues


def _overlay_status(descriptor: Mapping[str, Any]) -> Any:
    raw = descriptor.get("source_camera_overlay")
    if isinstance(raw, Mapping):
        return raw.get("status")
    if raw is None:
        raw = descriptor.get("source_camera_overlay_status")
    return raw


def _row_identity_mode(descriptor: Mapping[str, Any]) -> Any:
    return _as_mapping(descriptor.get("row_identity")).get("mode")


def _surface_profile_issues(
    *,
    surface_type: str,
    descriptor: Mapping[str, Any],
    row_identity_domain: str | None,
) -> list[dict[str, Any]]:
    profile = _SURFACE_PROFILES.get(surface_type)
    if profile is None:
        return [
            _issue(
                "SURFACE_PROFILE_UNCLASSIFIED",
                "error",
                "Coordinate-bearing surfaces require an explicit producer/consumer profile.",
                surface_type=surface_type,
            )
        ]
    issues: list[dict[str, Any]] = []
    geometry = descriptor.get("geometry_type")
    space = descriptor.get("space_id")
    row_mode = _row_identity_mode(descriptor)
    overlay = _overlay_status(descriptor)
    if profile.geometry_types and geometry not in profile.geometry_types:
        issues.append(
            _issue(
                "SURFACE_PROFILE_GEOMETRY_UNSUPPORTED",
                "error",
                "Descriptor geometry_type is not allowed by the explicit surface profile.",
                profile_id=profile.profile_id,
                declared_geometry_type=geometry,
                allowed_geometry_types=sorted(profile.geometry_types),
            )
        )
    if profile.space_ids and space not in profile.space_ids:
        issues.append(
            _issue(
                "SURFACE_PROFILE_SPACE_UNSUPPORTED",
                "error",
                "Descriptor space_id is not allowed by the explicit surface profile.",
                profile_id=profile.profile_id,
                declared_space_id=space,
                allowed_space_ids=sorted(profile.space_ids),
            )
        )
    if row_mode == "not_applicable" and not profile.rowless:
        issues.append(
            _issue(
                "ROW_IDENTITY_NOT_APPLICABLE_FORBIDDEN",
                "error",
                "not_applicable row identity is reserved for truly rowless calibration surfaces.",
                profile_id=profile.profile_id,
            )
        )
    elif (
        row_identity_domain is not None
        and row_identity_domain not in profile.row_identity_domains
    ):
        issues.append(
            _issue(
                "SURFACE_PROFILE_ROW_IDENTITY_UNSUPPORTED",
                "error",
                "Canonical row-identity domain is not allowed by the explicit surface profile.",
                profile_id=profile.profile_id,
                declared_row_identity_mode=row_mode,
                declared_row_identity_domain=row_identity_domain,
                allowed_row_identity_domains=sorted(profile.row_identity_domains),
            )
        )
    if profile.overlay_statuses and overlay not in profile.overlay_statuses:
        issues.append(
            _issue(
                "SURFACE_PROFILE_OVERLAY_UNSUPPORTED",
                "error",
                "Descriptor camera-overlay declaration is not allowed by the explicit surface profile.",
                profile_id=profile.profile_id,
                declared_overlay_status=overlay,
                allowed_overlay_statuses=sorted(profile.overlay_statuses),
            )
        )
    lineage_refs = descriptor.get("lineage_refs")
    if profile.requires_lineage and not (
        isinstance(lineage_refs, (list, tuple)) and lineage_refs
    ):
        issues.append(
            _issue(
                "SURFACE_PROFILE_LINEAGE_REQUIRED",
                "error",
                "The explicit surface profile requires digest-bound source lineage.",
                profile_id=profile.profile_id,
            )
        )
    transform_refs = descriptor.get("transform_refs")
    if not isinstance(transform_refs, (list, tuple)):
        transform_refs = _as_mapping(
            descriptor.get("source_camera_overlay")
        ).get("transform_refs")
    if profile.requires_transform and not (
        isinstance(transform_refs, (list, tuple)) and transform_refs
    ):
        issues.append(
            _issue(
                "SURFACE_PROFILE_TRANSFORM_REQUIRED",
                "error",
                "The explicit surface profile requires a directed transform reference.",
                profile_id=profile.profile_id,
            )
        )
    return issues


def classify_surface_contract(
    *,
    surface_type: str,
    node: MetadataNode,
    nodes: Mapping[str, MetadataNode],
) -> dict[str, Any]:
    """Classify a surface from declarations and linked metadata only."""

    evidence, descriptor_match = _surface_evidence(surface_type, node, nodes)
    descriptor = descriptor_match.descriptor if descriptor_match else None
    descriptor_source = descriptor_match.source if descriptor_match else None
    descriptor_is_array_specific = bool(
        descriptor_match and descriptor_match.array_specific
    )
    issues: list[dict[str, Any]] = _descriptor_declaration_issues(node, nodes)

    if surface_type == "unclassified_geometry_candidate":
        if _node_has_direct_coordinate_descriptor(node):
            issues.append(
                _issue(
                    "UNSUPPORTED_DECLARED_COORDINATE_SURFACE",
                    "error",
                    "An array-specific coordinate descriptor is persisted on a numeric surface that has no controlled producer/consumer profile.",
                    surface_path=node.relative_path,
                )
            )
        else:
            issues.append(
                _issue(
                    "UNCLASSIFIED_GEOMETRY_CANDIDATE",
                    "error",
                    "A geometry-like persisted array is not assigned to a controlled audit surface family.",
                    surface_path=node.relative_path,
                )
            )

    if node.metadata_error:
        issues.append(
            _issue(
                "INVALID_ZARR_METADATA",
                "error",
                "The surface metadata file could not be parsed completely.",
                error=node.metadata_error,
            )
        )
        return {
            "status": "ambiguous_fail_closed",
            "issues": issues,
            "evidence": evidence,
            "coordinate_descriptor": descriptor,
            "descriptor_source": descriptor_source,
            "descriptor_is_array_specific": descriptor_is_array_specific,
        }

    if node.node_type == "array":
        dtype = _metadata_dtype(node.data_type)
        if dtype is None or dtype.kind not in "iuf":
            issues.append(
                _issue(
                    "COORDINATE_ARRAY_DTYPE_NONNUMERIC",
                    "critical",
                    "A persisted coordinate/geometry surface must use a real numeric dtype.",
                    data_type=node.data_type,
                )
            )
    if surface_type in _DIRECTED_TRANSFORM_SURFACES:
        transform, transform_issues = _parse_directed_transform_v2_node(
            node,
            record_ref=(
                f"/{node.relative_path}@{DIRECTED_TRANSFORM_V2_ATTR}"
            ),
            nodes=nodes,
        )
        issues.extend(transform_issues)
        expected_surface = (
            _DIRECTED_TRANSFORM_SURFACE_BY_KIND.get(transform.kind)
            if transform is not None
            else None
        )
        if transform is not None and expected_surface != surface_type:
            issues.append(
                _issue(
                    "DIRECTED_TRANSFORM_V2_SURFACE_KIND_MISMATCH",
                    "critical",
                    "The inventoried transform role must equal its exact parsed v2 kind.",
                    transform_kind=transform.kind,
                    expected_surface_type=expected_surface,
                    actual_surface_type=surface_type,
                )
            )
        has_failure = any(
            issue["severity"] in {"error", "critical"}
            for issue in issues
        )
        return {
            "status": (
                "ambiguous_fail_closed"
                if has_failure
                else "numerical_validation_required"
            ),
            "issues": issues,
            "evidence": {
                **evidence,
                "directed_transform_v2_kind": (
                    transform.kind if transform is not None else None
                ),
            },
            "coordinate_descriptor": descriptor,
            "descriptor_source": descriptor_source,
            "descriptor_is_array_specific": descriptor_is_array_specific,
        }
    issues.extend(
        _flattened_contour_lineage_issues(
            surface_type=surface_type,
            node=node,
            nodes=nodes,
        )
    )
    issues.extend(
        _subject_mask_rle_lineage_issues(
            surface_type=surface_type,
            node=node,
            nodes=nodes,
        )
    )

    online_bad, online_bad_evidence, online_correction_issues = (
        _legacy_online_mm_requires_recompute(
            surface_type, node, nodes, descriptor
        )
    )
    issues.extend(online_correction_issues)
    if surface_type == "unclassified_geometry_candidate":
        status = "ambiguous_fail_closed"
    elif online_bad:
        issues.append(
            _issue(
                "ONLINE_MM_CONVERSION_RECOMPUTATION_REQUIRED",
                "critical",
                "Historical online millimetre positions may have multiplied by pixels/mm; metadata cannot repair values.",
                **online_bad_evidence,
            )
        )

    offline_crop_bad, offline_crop_evidence, offline_correction_issues = (
        _offline_crop_reconstruction_requires_recompute(
            surface_type, node, nodes, descriptor
        )
    )
    issues.extend(offline_correction_issues)
    if offline_crop_bad:
        issues.append(
            _issue(
                "OFFLINE_CROP_SOURCE_RECONSTRUCTION_NUMERICAL_VALIDATION_REQUIRED",
                "error",
                "Crop-row positions declared as camera pixels need targeted validation against exact source reference dimensions.",
                **offline_crop_evidence,
            )
        )

    space = _value(evidence, "space_id")
    units = _value(evidence, "units")
    width = _value(evidence, "reference_width")
    height = _value(evidence, "reference_height")
    reference_authority = _value(evidence, "reference_authority")
    origin = _value(evidence, "origin")
    x_axis = _value(evidence, "x_axis_direction")
    y_axis = _value(evidence, "y_axis_direction")
    row_identity = _value(evidence, "row_identity")
    source_ref = _value(evidence, "source_ref")
    transform_ref = _value(evidence, "transform_ref")
    overlay = _value(evidence, "source_camera_overlay_suitable")
    pixel_convention = _value(evidence, "pixel_convention")
    geometry_convention = _value(evidence, "geometry_convention")

    if descriptor is None:
        issues.append(
            _issue(
                "ARRAY_COORDINATE_DESCRIPTOR_MISSING",
                "warning",
                "No compact array-specific coordinate descriptor is persisted.",
            )
        )
        issues.extend(_legacy_reference_issues(evidence=evidence, nodes=nodes))
        issues.extend(
            _legacy_row_identity_issues(
                surface_type=surface_type,
                node=node,
                nodes=nodes,
            )
        )
    elif not descriptor_is_array_specific:
        issues.append(
            _issue(
                "COORDINATE_DESCRIPTOR_INHERITED",
                (
                    "error"
                    if descriptor.get("schema_version")
                    == CANONICAL_COORDINATE_DESCRIPTOR_SCHEMA_VERSION
                    else "warning"
                ),
                "Canonical v2 descriptors must be array-specific; historical descriptors inherited from an ancestor remain migration evidence only.",
                descriptor_source=descriptor_source,
            )
        )

    descriptor_validation_issues: list[dict[str, str]] = []
    row_identity_domain: str | None = None
    if descriptor is not None:
        descriptor_validator = (
            validate_canonical_coordinate_descriptor
            if descriptor.get("schema_version")
            == CANONICAL_COORDINATE_DESCRIPTOR_SCHEMA_VERSION
            else validate_historical_coordinate_descriptor_v1
        )
        descriptor_validation_issues = [
            {"code": issue.code, "path": issue.path, "message": issue.message}
            for issue in descriptor_validator(descriptor)
        ]
        if descriptor_validation_issues:
            issues.append(
                _issue(
                    "COORDINATE_DESCRIPTOR_INVALID",
                    "error",
                    "The compact coordinate descriptor does not satisfy the canonical schema.",
                    validation_issues=descriptor_validation_issues,
                    descriptor_source=descriptor_source,
                )
            )
        if descriptor_match is not None:
            descriptor_integrity_issues, row_identity_domain = (
                _descriptor_integrity_issues(
                    match=descriptor_match,
                    surface_node=node,
                    nodes=nodes,
                )
            )
            issues.extend(descriptor_integrity_issues)
        issues.extend(
            _surface_profile_issues(
                surface_type=surface_type,
                descriptor=descriptor,
                row_identity_domain=row_identity_domain,
            )
        )
        issues.extend(
            _descriptor_conflict_issues(
                descriptor=descriptor,
                node=node,
                nodes=nodes,
            )
        )
        if online_bad_evidence.get("sealed_correction_exception") is True:
            producer_space_source = online_bad_evidence.get(
                "coordinate_space_source"
            )
            superseded = [
                issue
                for issue in issues
                if issue.get("code") == "DESCRIPTOR_DECLARATION_CONFLICT"
                and _as_mapping(issue.get("evidence")).get("field")
                == "space_id"
                and _as_mapping(issue.get("evidence")).get("declared_value")
                == "texture"
                and _as_mapping(issue.get("evidence")).get("descriptor_value")
                == "physical_mm"
                and _as_mapping(issue.get("evidence")).get(
                    "declaration_source"
                )
                == producer_space_source
            ]
            if superseded:
                issues = [issue for issue in issues if issue not in superseded]
                issues.append(
                    _issue(
                        "LEGACY_PRODUCER_DECLARATION_SUPERSEDED_BY_SEALED_CORRECTION",
                        "warning",
                        "The exact historical texture-space producer signature is retained as provenance, while sealed physical authority and independent value validation govern the corrected array.",
                        declaration_source=producer_space_source,
                    )
                )

    if space in (None, ""):
        issues.append(_issue("COORDINATE_SPACE_MISSING", "error", "Coordinate space is not declared."))
    legacy_resolution_valid = False
    legacy_invariants_valid = False
    legacy_compatibility_proof: Mapping[str, Any] | None = None
    if isinstance(space, str) and space.lower() in {"camera", "texture"}:
        legacy_resolution_valid, legacy_error = _legacy_context_is_valid(
            legacy_label=space,
            evidence=evidence,
            nodes=nodes,
        )
        issues.append(
            _issue(
                "LEGACY_SPACE_LABEL_REQUIRES_COMPATIBILITY_RULE",
                "warning",
                "The legacy camera/texture label needs an explicit compatibility mapping.",
                declared_space=space,
            )
        )
        if not legacy_resolution_valid:
            issues.append(
                _issue(
                    "LEGACY_SPACE_CONTEXT_INVALID",
                    "error",
                    "Legacy camera/texture label lacks the explicit extent authority and evidence required by the compatibility resolver.",
                    declared_space=space,
                    error=legacy_error,
                )
            )
        else:
            (
                legacy_invariants_valid,
                invariant_error,
                legacy_compatibility_proof,
            ) = (
                _legacy_numerical_invariant_proof(
                    legacy_label=space,
                    node=node,
                    evidence=evidence,
                    nodes=nodes,
                )
            )
            if not legacy_invariants_valid:
                issues.append(
                    _issue(
                        "LEGACY_NUMERICAL_INVARIANTS_MISSING",
                        "error",
                        "Legacy compatibility is not a safe metadata backfill without an independent value-invariant record.",
                        declared_space=space,
                        error=invariant_error,
                    )
                )
            else:
                issues.append(
                    _issue(
                        "LEGACY_COMPATIBILITY_PROOF_NOT_CANONICAL_AUTHORITY",
                        "warning",
                        "An archive-local compatibility assertion is migration evidence, not authority for a canonical descriptor or a safe backfill.",
                        declared_space=space,
                    )
                )
    elif descriptor is None and space not in (None, ""):
        issues.append(
            _issue(
                "LEGACY_SPACE_UNCONTROLLED",
                "error",
                "Descriptor-free coordinates may use only the explicit controlled camera/texture compatibility resolver.",
                declared_space=space,
            )
        )
    if units in (None, ""):
        issues.append(_issue("COORDINATE_UNITS_MISSING", "error", "Coordinate units are not declared."))
    if origin in (None, "") or x_axis in (None, "") or y_axis in (None, ""):
        issues.append(
            _issue(
                "ORIGIN_OR_AXES_MISSING",
                "error",
                "Origin and positive X/Y directions are not fully declared.",
                origin=origin,
                x_axis_direction=x_axis,
                y_axis_direction=y_axis,
            )
        )
    if surface_type in _PIXEL_OR_NORMALIZED_SURFACES and (width in (None, "") or height in (None, "")):
        issues.append(
            _issue(
                "REFERENCE_EXTENT_MISSING",
                "error",
                "Pixel or normalized coordinates lack exact reference width/height.",
                reference_width=width,
                reference_height=height,
            )
        )
    if surface_type in _PIXEL_OR_NORMALIZED_SURFACES and reference_authority in (None, ""):
        issues.append(
            _issue(
                "REFERENCE_AUTHORITY_MISSING",
                "error",
                "Pixel or normalized coordinates lack an exact reference-extent authority.",
            )
        )
    if surface_type in _PIXEL_OR_NORMALIZED_SURFACES and pixel_convention in (None, ""):
        issues.append(
            _issue(
                "PIXEL_CONVENTION_MISSING",
                "error",
                "Pixel-center, pixel-edge, or continuous-coordinate convention is not declared.",
            )
        )
    if geometry_convention in (None, ""):
        issues.append(
            _issue(
                "GEOMETRY_CONVENTION_MISSING",
                "error",
                "Component order and geometry convention are not declared.",
            )
        )
    if not row_identity and surface_type != "calibration_homography":
        issues.append(_issue("ROW_IDENTITY_MISSING", "error", "Frame/row identity is not linked."))

    if surface_type == "calibration_homography":
        if node.node_type != "array" or list(node.shape or []) != [3, 3]:
            issues.append(
                _issue(
                    "HOMOGRAPHY_ARRAY_INVALID",
                    "critical",
                    "A persisted homography surface must be exactly one 3x3 array.",
                    node_type=node.node_type,
                    shape=node.shape,
                )
            )
        homography_dtype = _metadata_dtype(node.data_type)
        if homography_dtype is None or homography_dtype.kind not in "f":
            issues.append(
                _issue(
                    "HOMOGRAPHY_DTYPE_NONNUMERIC",
                    "critical",
                    "A persisted homography matrix must use a real floating-point dtype.",
                    data_type=node.data_type,
                )
            )
        has_directed_transform_v2 = DIRECTED_TRANSFORM_V2_ATTR in node.attributes
        has_historical_directed_transform = "directed_transform" in node.attributes
        if has_directed_transform_v2 and has_historical_directed_transform:
            issues.append(
                _issue(
                    "DIRECTED_TRANSFORM_SCHEMA_CONFLICT",
                    "critical",
                    "A homography cannot publish canonical v2 and historical directed-transform records together.",
                )
            )
        if has_directed_transform_v2:
            homography_transform_v2, transform_v2_issues = (
                _parse_directed_transform_v2_node(
                    node,
                    record_ref=(
                        f"/{node.relative_path}@{DIRECTED_TRANSFORM_V2_ATTR}"
                    ),
                    nodes=nodes,
                )
            )
            issues.extend(transform_v2_issues)
            if (
                homography_transform_v2 is not None
                and homography_transform_v2.kind != HOMOGRAPHY_KIND
            ):
                issues.append(
                    _issue(
                        "HOMOGRAPHY_TRANSFORM_KIND_INVALID",
                        "critical",
                        "A calibration homography surface must publish a v2 homography transform kind.",
                        transform_kind=homography_transform_v2.kind,
                    )
                )
        elif has_historical_directed_transform:
            try:
                homography_transform = load_directed_homography_attrs(node.attributes)
            except DirectedTransformError as exc:
                issues.append(
                    _issue(
                        "DIRECTED_TRANSFORM_METADATA_INVALID",
                        "critical",
                        "Homography directed-transform metadata or digest is invalid.",
                        error=str(exc),
                    )
                )
            else:
                calibration_path = _normalize_archive_ref(
                    homography_transform.calibration_ref
                )
                try:
                    manifest, selected_transform = (
                        _load_selected_calibration_metadata(
                            camera_path=str(calibration_path or ""),
                            nodes=nodes,
                        )
                    )
                except SelectedCalibrationError as exc:
                    issues.append(
                        _issue(
                            "HOMOGRAPHY_CALIBRATION_SCHEMA_INVALID",
                            "critical",
                            "Homography calibration_ref must bind the strict selected-calibration manifest and source-evidence chain.",
                            calibration_ref=homography_transform.calibration_ref,
                            error=str(exc),
                        )
                    )
                else:
                    if (
                        _normalize_archive_ref(manifest.transform_ref)
                        != node.relative_path
                        or selected_transform.digest()
                        != homography_transform.digest()
                    ):
                        issues.append(
                            _issue(
                                "HOMOGRAPHY_SELECTED_TRANSFORM_MISMATCH",
                                "critical",
                                "Homography surface is not the exact selected transform named by its calibration manifest.",
                                surface_path=node.relative_path,
                                selected_transform_ref=manifest.transform_ref,
                            )
                        )
                issues.append(
                    _issue(
                        "HOMOGRAPHY_MATRIX_PAYLOAD_VALIDATION_REQUIRED",
                        "warning",
                        "Direction and metadata digest are valid, but the 3x3 matrix payload still requires hash and numerical validation.",
                        expected_matrix_sha256=homography_transform.matrix_sha256,
                    )
                )
        else:
            issues.append(
                _issue(
                    "DIRECTED_TRANSFORM_METADATA_MISSING",
                    "critical",
                    "A homography array must carry digest-bound palette.directed_transform metadata.",
                )
            )
        direction = _value(evidence, "transform_direction")
        from_space = _value(evidence, "transform_from_space")
        to_space = _value(evidence, "transform_to_space")
        if (
            not has_directed_transform_v2
            and direction in (None, "")
            and (from_space in (None, "") or to_space in (None, ""))
        ):
            issues.append(
                _issue(
                    "HOMOGRAPHY_DIRECTION_MISSING",
                    "critical",
                    "Homography direction is not explicitly labelled; its historical name is not evidence.",
                )
            )
        if (
            not has_directed_transform_v2
            and transform_ref in (None, "")
            and not descriptor_is_array_specific
        ):
            issues.append(
                _issue(
                    "CALIBRATION_LINEAGE_MISSING",
                    "error",
                    "Homography calibration lineage is not linked from this surface.",
                )
            )
    else:
        if source_ref in (None, "") and transform_ref in (None, ""):
            issues.append(
                _issue(
                    "SOURCE_OR_TRANSFORM_LINEAGE_MISSING",
                    "error",
                    "The selected source or transform lineage is not linked.",
                )
            )
        if overlay in (None, ""):
            issues.append(
                _issue(
                    "SOURCE_CAMERA_OVERLAY_SUITABILITY_UNDECLARED",
                    "warning",
                    "Suitability for source-camera overlay is not declared.",
                )
            )

    fail_closed_issue_codes = {
            "ARRAY_COORDINATE_DESCRIPTORS_CONTAINER_INVALID",
            "ARRAY_COORDINATE_DESCRIPTOR_ATTR_MISKEYED",
            "COORDINATE_ARRAY_DTYPE_NONNUMERIC",
            "HOMOGRAPHY_DTYPE_NONNUMERIC",
            "MULTIPLE_COORDINATE_DESCRIPTORS_CONFLICT",
            "GENERIC_ANCESTOR_DESCRIPTOR_CONTAMINATION",
            "COORDINATE_DESCRIPTOR_DIGEST_MISMATCH",
            "COORDINATE_DESCRIPTOR_INTEGRITY_INVALID",
            "ROW_IDENTITY_REF_UNRESOLVED",
            "ROW_IDENTITY_LENGTH_MISMATCH",
            "COORDINATE_RECORD_REF_UNRESOLVED",
            "COORDINATE_RECORD_ATTR_UNRESOLVED",
            "COORDINATE_RECORD_SELECTOR_UNRESOLVED",
            "COORDINATE_RECORD_DIGEST_MISSING",
            "COORDINATE_RECORD_DIGEST_MISMATCH",
            "TRANSFORM_REF_NOT_DIRECTION_EXPLICIT",
            "DIRECTED_TRANSFORM_REF_INVALID",
            "DIRECTED_TRANSFORM_TARGET_INVALID",
            "DIRECTED_TRANSFORM_METADATA_INVALID",
            "REQUIRED_TRANSFORM_REF_MISSING",
            "TRANSFORM_CHAIN_UNRESOLVED",
            "TRANSFORM_CHAIN_NOT_LINEAR",
            "TRANSFORM_DIRECTION_INCOMPATIBLE_WITH_SURFACE",
            "TRANSFORM_TARGET_EXTENT_MISMATCH",
            "TRANSFORM_CHAIN_DISCONNECTED_OR_REVERSED",
            "TRANSFORM_CHAIN_EXTENT_MISMATCH",
            "SOURCE_CAMERA_TRANSFORM_ROUTE_MISSING",
            "TRANSFORM_CALIBRATION_REF_UNRESOLVED",
            "TRANSFORM_CAMERA_AUTHORITY_IDENTITY_MISSING",
            "TRANSFORM_CAMERA_ID_MISMATCH",
            "TRANSFORM_CAMERA_CALIBRATION_MISMATCH",
            "TRANSFORM_CHAIN_CAMERA_IDENTITY_CONFLICT",
            "REFERENCE_AUTHORITY_MISSING",
            "REFERENCE_AUTHORITY_UNRESOLVED",
            "REFERENCE_AUTHORITY_TARGET_INVALID",
            "REFERENCE_AUTHORITY_SYNTAX_UNSUPPORTED",
            "REFERENCE_AUTHORITY_UNITS_UNRESOLVED",
            "REFERENCE_AUTHORITY_UNITS_MISMATCH",
            "REFERENCE_AUTHORITY_EXTENT_MISMATCH",
            "ACQUISITION_AUTHORITY_PATH_MISMATCH",
            "ACQUISITION_ROOT_METADATA_MISMATCH",
            "ACQUISITION_IMPORT_OWNERSHIP_IDENTITY_MISMATCH",
            "ACQUISITION_MODE_MISMATCH",
            "ACQUISITION_MATERIALIZED_NODE_UNRESOLVED",
            "ACQUISITION_MATERIALIZED_NODE_METADATA_MISMATCH",
            "ACQUISITION_MATERIALIZED_POINTER_MISMATCH",
            "ACQUISITION_MATERIALIZATION_MANIFEST_NODE_UNRESOLVED",
            "ACQUISITION_PHYSICAL_CHUNK_MANIFEST_INVALID",
            "ACQUISITION_MATERIALIZATION_MANIFEST_INVALID",
            "ACQUISITION_MATERIALIZATION_MANIFEST_MISMATCH",
            "PHYSICAL_FRAME_AUTHORITY_REQUIRED",
            "PHYSICAL_FRAME_RECORD_INVALID",
            "PHYSICAL_FRAME_RECORD_MISMATCH",
            "PHYSICAL_FRAME_EXTENT_UNPROVEN",
            "PHYSICAL_FRAME_AUTHORITY_SPACE_MISMATCH",
            "PHYSICAL_FRAME_LINEAGE_MISSING",
            "PHYSICAL_FRAME_LINEAGE_DIGEST_MISSING",
            "PHYSICAL_FRAME_LINEAGE_DIGEST_MISMATCH",
            "FLATTENED_CONTOUR_SCHEMA_MISSING",
            "FLATTENED_CONTOUR_INDEX_LINEAGE_MISSING",
            "FLATTENED_CONTOUR_INDEX_LINEAGE_INVALID",
            "LEGACY_FLATTENED_CONTOUR_INDEX_LINEAGE_MISSING",
            "LEGACY_FLATTENED_CONTOUR_INDEX_LINEAGE_INVALID",
            "SUBJECT_MASK_RLE_LAYOUT_INVALID",
            "SUBJECT_MASK_RLE_LINEAGE_INVALID",
            "LEGACY_ROW_IDENTITY_AMBIGUOUS",
            "LEGACY_ROW_IDENTITY_UNTYPED",
            "TRACK_TIME_LINEAGE_REF_INVALID",
            "TRACK_TIME_LINEAGE_RECORD_INVALID",
            "TRACK_TIME_LINEAGE_RETIRED_DIRECT_ACQUISITION",
            "TRACK_TIME_LINEAGE_DIGEST_MISMATCH",
            "TRACK_TIME_LINEAGE_IDENTITY_MISMATCH",
            "TRACK_TIME_LINEAGE_ACQUISITION_INVALID",
            "TRACK_TIME_LINEAGE_ACQUISITION_MISMATCH",
            "TRACK_TIME_LINEAGE_ARRAY_INVALID",
            "TRACK_TIME_LINEAGE_SOURCE_AUTHORITY_INVALID",
            "TRACK_TIME_LINEAGE_SOURCE_AUTHORITY_MISMATCH",
            "TRACK_TIME_LINEAGE_SELF_CERTIFIED_SOURCE",
            "TRACK_TIME_LINEAGE_SOURCE_IDENTITY_INVALID",
            "TRACK_TIME_LINEAGE_SOURCE_ARRAY_INVALID",
            "REGISTERED_COORDINATE_RECORD_ATTR_INVALID",
            "REGISTERED_COORDINATE_RECORD_SCHEMA_INVALID",
            "REGISTERED_COORDINATE_RECORD_DIGEST_MISMATCH",
            "REGISTERED_COORDINATE_RECORD_OWNER_INVALID",
            "OBSERVATION_COORDINATE_ROWSET_PATH_INVALID",
            "OBSERVATION_COORDINATE_LINEAGE_REQUIRED",
            "OBSERVATION_ARRAY_PAYLOAD_METADATA_INVALID",
            "OBSERVATION_TEMPORAL_AUTHORITY_INVALID",
            "OBSERVATION_TEMPORAL_ARRAY_INVALID",
            "OBSERVATION_TEMPORAL_IDENTITY_ARRAY_INVALID",
            "OBSERVATION_TEMPORAL_ACQUISITION_INVALID",
            "OBSERVATION_TEMPORAL_ACQUISITION_MISMATCH",
            "DETECTION_ACQUISITION_MAPPING_INVALID",
            "DETECTION_BBOX_PROJECTION_INVALID",
            "BBOX_CENTER_DERIVATION_INVALID",
            "CROP_GEOMETRY_SELECTION_INVALID",
            "CROP_ROI_GEOMETRY_DERIVATION_INVALID",
            "LEGACY_SPACE_CONTEXT_INVALID",
            "LEGACY_SPACE_UNCONTROLLED",
    }
    if any(issue["code"] in fail_closed_issue_codes for issue in issues) or (
        surface_type == "calibration_homography"
        and any(issue["severity"] in {"error", "critical"} for issue in issues)
    ):
        status = "ambiguous_fail_closed"
    elif descriptor is not None and any(
        issue["severity"] in {"error", "critical"} for issue in issues
    ):
        status = "ambiguous_fail_closed"
    elif online_bad:
        status = "recompute_required"
    elif offline_crop_bad:
        status = "numerical_validation_required"
    elif surface_type == "calibration_homography" and any(
        issue["code"] == "HOMOGRAPHY_DIRECTION_MISSING" for issue in issues
    ):
        status = "ambiguous_fail_closed"
    elif any(
        issue["code"]
        in {
            "HOMOGRAPHY_MATRIX_PAYLOAD_VALIDATION_REQUIRED",
            "HISTORICAL_COORDINATE_DESCRIPTOR_V1_REQUIRES_MIGRATION",
            "LEGACY_COORDINATE_ROW_IDENTITY_REQUIRES_MIGRATION",
            "LEGACY_ROW_IDENTITY_REQUIRES_MIGRATION",
            "ROW_IDENTITY_KEY_PAYLOAD_VALIDATION_REQUIRED",
            "TRANSFORM_MATRIX_PAYLOAD_VALIDATION_REQUIRED",
            "FLATTENED_CONTOUR_INDEX_PAYLOAD_VALIDATION_REQUIRED",
            "SUBJECT_MASK_RLE_PAYLOAD_VALIDATION_REQUIRED",
            "COORDINATE_VALUE_VALIDATION_PAYLOAD_CHECK_REQUIRED",
            "TRACK_TIME_LINEAGE_PAYLOAD_VALIDATION_REQUIRED",
            "PIXEL_FRAME_AUTHORITY_LIVE_VALIDATION_REQUIRED",
            "DIRECTED_TRANSFORM_V2_LIVE_VALIDATION_REQUIRED",
            "OBSERVATION_COORDINATE_PAYLOAD_VALIDATION_REQUIRED",
        }
        for issue in issues
    ):
        status = "numerical_validation_required"
    elif descriptor_is_array_specific and not descriptor_validation_issues and not any(
        issue["severity"] in {"error", "critical"} for issue in issues
    ):
        status = "compatible"
    else:
        critical_missing = {
            "COORDINATE_SPACE_MISSING",
            "REFERENCE_EXTENT_MISSING",
            "ORIGIN_OR_AXES_MISSING",
            "SOURCE_OR_TRANSFORM_LINEAGE_MISSING",
        }
        issue_codes = {str(issue["code"]) for issue in issues}
        has_direct_legacy_core = all(
            _is_direct_source(node, evidence, field)
            for field in ("space_id", "units", "origin", "x_axis_direction", "y_axis_direction")
        ) and (
            isinstance(space, str)
            and space.lower() in {"camera", "texture"}
            and legacy_resolution_valid
            and legacy_invariants_valid
        )
        exact_link_available = source_ref not in (None, "") or transform_ref not in (None, "")
        if issue_codes & critical_missing and not exact_link_available:
            status = "ambiguous_fail_closed"
        elif descriptor is not None or exact_link_available or space not in (None, ""):
            status = "metadata_backfill_candidate"
        elif has_direct_legacy_core:
            status = "compatible_via_explicit_legacy_rule"
        else:
            status = "ambiguous_fail_closed"

        # A fully explicit direct legacy declaration remains readable under a
        # testable compatibility rule even though a canonical descriptor is
        # absent.  Reference extent and row identity must still be present.
        if descriptor is None and has_direct_legacy_core:
            # Legacy keys and archive-local proof records can prioritize a
            # migration, but they never prove a canonical identity/descriptor
            # contract or a safe metadata-only rewrite.
            status = "numerical_validation_required"

        if (
            descriptor is None
            and isinstance(space, str)
            and space.lower() in {"camera", "texture"}
            and legacy_resolution_valid
            and not legacy_invariants_valid
            and status != "ambiguous_fail_closed"
        ):
            status = "numerical_validation_required"

        # These historical surfaces need value-level confirmation even when
        # their legacy declarations are sufficient to construct metadata.  The
        # scanner never samples array payloads, so it records that work rather
        # than guessing correctness from names or ranges.
        if (
            status in {
                "metadata_backfill_candidate",
                "compatible_via_explicit_legacy_rule",
            }
            and surface_type in {
                "refined_online_positions_px",
                "refined_online_positions_mm",
            }
        ):
            status = "numerical_validation_required"

    if surface_type == "unclassified_geometry_candidate":
        status = "ambiguous_fail_closed"

    return {
        "status": status,
        "issues": issues,
        "evidence": evidence,
        "coordinate_descriptor": descriptor,
        "descriptor_source": descriptor_source,
        "descriptor_is_array_specific": descriptor_is_array_specific,
        "legacy_compatibility_proof": _json_safe(legacy_compatibility_proof),
    }


def _track_owner_path(relative_path: str) -> str | None:
    parts = PurePosixPath(relative_path).parts
    try:
        tracks_index = parts.index("tracks")
    except ValueError:
        return None
    if len(parts) <= tracks_index + 1:
        return None
    return PurePosixPath(*parts[: tracks_index + 2]).as_posix()


def _is_position_derived_track_array(relative_path: str) -> bool:
    leaf = PurePosixPath(relative_path).name.lower()
    path = relative_path.lower()
    if leaf in {"positions_px", "positions_mm"}:
        return False
    if any(token in leaf for token in ("speed", "acceleration", "path_distance")):
        return True
    return any(
        token in path
        for token in ("/movement/speed/", "/speed_derivatives/")
    ) and leaf in {"px", "mm"}


def _append_unique_issue(record: dict[str, Any], issue: Mapping[str, Any]) -> None:
    code = str(issue.get("code") or "")
    if code not in {str(item.get("code")) for item in record.get("issues", []) if isinstance(item, Mapping)}:
        record.setdefault("issues", []).append(_json_safe(issue))
    record["issue_codes"] = sorted(
        {str(item.get("code")) for item in record.get("issues", []) if isinstance(item, Mapping)}
    )


def _propagate_track_dependency_risk(
    surface_records: list[dict[str, Any]],
    nodes: Mapping[str, MetadataNode],
) -> None:
    by_owner: dict[str, list[dict[str, Any]]] = {}
    for record in surface_records:
        owner = _track_owner_path(str(record.get("surface_path") or ""))
        if owner is not None:
            by_owner.setdefault(owner, []).append(record)

    for owner, records in by_owner.items():
        positions_px = next(
            (record for record in records if record.get("surface_type") == "track_positions_px"),
            None,
        )
        positions_mm = next(
            (record for record in records if record.get("surface_type") == "track_positions_mm"),
            None,
        )
        dependency_paths = sorted(
            path
            for path, candidate in nodes.items()
            if candidate.node_type == "array"
            and path.startswith(f"{owner}/")
            and _is_position_derived_track_array(path)
        )
        if positions_px is not None:
            positions_px["dependent_surface_paths"] = dependency_paths
        if positions_mm is not None:
            positions_mm["dependent_surface_paths"] = [
                path
                for path in dependency_paths
                if PurePosixPath(path).name.lower().endswith("mm")
                or PurePosixPath(path).name.lower() == "mm"
            ]
        if positions_px is None:
            continue
        source_status = str(positions_px.get("status"))
        if positions_mm is not None:
            prior_status = str(positions_mm.get("status"))
            effective_status = max(
                (prior_status, source_status),
                key=lambda status: _STATUS_PRIORITY[status],
            )
            positions_mm["status"] = effective_status
            positions_mm["dependency_source_paths"] = [positions_px["surface_path"]]
            _append_unique_issue(
                positions_mm,
                _issue(
                    "UPSTREAM_POSITION_RISK_PROPAGATED",
                    "error",
                    "Derived millimetre positions retain the worst status across their own contract and positions_px.",
                    upstream_path=positions_px["surface_path"],
                    upstream_status=source_status,
                    prior_status=prior_status,
                    effective_status=effective_status,
                ),
            )


def _descriptor_row_identity_signature(
    record: Mapping[str, Any],
) -> tuple[Any, ...] | None:
    descriptor = _as_mapping(record.get("coordinate_descriptor"))
    row_identity = _as_mapping(descriptor.get("row_identity"))
    if descriptor.get("schema_version") == CANONICAL_COORDINATE_DESCRIPTOR_SCHEMA_VERSION:
        record_ref = row_identity.get("record_ref")
        record_digest = row_identity.get("record_sha256")
        if not isinstance(record_ref, str) or not isinstance(record_digest, str):
            return None
        return ("canonical_v2", record_ref, record_digest)
    mode = row_identity.get("mode")
    if not isinstance(mode, str):
        return None
    owner = str(record.get("surface_path") or "")
    refs = tuple(
        sorted(
            (
                name,
                _normalize_archive_ref(
                    ref,
                    owner_path=owner,
                    owner_is_array=record.get("node_type") == "array",
                ),
            )
            for name, ref in _row_identity_refs(row_identity)
        )
    )
    return mode, refs


def _physical_record_from_descriptor(
    descriptor: Mapping[str, Any],
    *,
    nodes: Mapping[str, MetadataNode],
) -> Mapping[str, Any] | None:
    authority = _as_mapping(descriptor.get("reference_extent")).get("authority")
    if isinstance(authority, Mapping):
        record_ref = authority.get("record_ref")
        node_path, attr_name = _canonical_v2_record_target(record_ref)
        target = nodes.get(node_path or "")
        if target is None:
            return None
        if attr_name not in (None, "zarr_metadata"):
            return _as_mapping(target.attributes.get(attr_name)) or None
        return target.attributes
    if not isinstance(authority, str):
        return None
    node_ref, selector_kind, selector = _record_ref_target(authority)
    target = nodes.get(_normalize_archive_ref(node_ref) or "")
    if target is None:
        return None
    if selector_kind in {"attr", "fragment"}:
        return _as_mapping(target.attributes.get(str(selector))) or None
    return target.attributes


def _validate_track_px_mm_coherence(
    surface_records: Sequence[dict[str, Any]],
    nodes: Mapping[str, MetadataNode],
) -> None:
    by_owner: dict[str, dict[str, dict[str, Any]]] = {}
    for record in surface_records:
        surface_type = str(record.get("surface_type") or "")
        if surface_type not in {
            "track_positions_px",
            "track_positions_mm",
            "refined_online_positions_px",
            "refined_online_positions_mm",
        }:
            continue
        owner = _track_owner_path(str(record.get("surface_path") or ""))
        if owner is None:
            owner = str(PurePosixPath(str(record.get("surface_path") or "")).parent)
        by_owner.setdefault(owner, {})[surface_type] = record

    for pairs in by_owner.values():
        for px_type, mm_type in (
            ("track_positions_px", "track_positions_mm"),
            ("refined_online_positions_px", "refined_online_positions_mm"),
        ):
            px_record = pairs.get(px_type)
            mm_record = pairs.get(mm_type)
            if px_record is None or mm_record is None:
                continue
            px_identity = _descriptor_row_identity_signature(px_record)
            mm_identity = _descriptor_row_identity_signature(mm_record)
            defects: dict[str, Any] = {}
            if px_identity is None or mm_identity is None or px_identity != mm_identity:
                defects["row_identity"] = {
                    "positions_px": px_identity,
                    "positions_mm": mm_identity,
                }
            px_descriptor = _as_mapping(px_record.get("coordinate_descriptor"))
            mm_descriptor = _as_mapping(mm_record.get("coordinate_descriptor"))
            physical = _physical_record_from_descriptor(mm_descriptor, nodes=nodes)
            px_authority = _as_mapping(px_descriptor.get("reference_extent")).get(
                "authority"
            )
            if isinstance(px_authority, Mapping):
                px_authority = px_authority.get("record_ref")
            if (
                physical is None
                or physical.get("source_space_id") != px_descriptor.get("space_id")
                or physical.get("source_reference_authority") != px_authority
            ):
                defects["physical_source_relationship"] = {
                    "positions_px_space_id": px_descriptor.get("space_id"),
                    "positions_px_reference_authority": px_authority,
                    "physical_frame_source_space_id": (
                        physical.get("source_space_id") if physical else None
                    ),
                    "physical_frame_source_reference_authority": (
                        physical.get("source_reference_authority")
                        if physical
                        else None
                    ),
                }
            if not defects:
                continue
            for record in (px_record, mm_record):
                _append_unique_issue(
                    record,
                    _issue(
                        "TRACK_PX_MM_COORDINATE_CONTRACT_MISMATCH",
                        "critical",
                        "positions_px and positions_mm must share one exact row identity and a reciprocal, source-bound calibration relationship.",
                        defects=defects,
                    ),
                )
                if _STATUS_PRIORITY.get(str(record.get("status")), 99) < _STATUS_PRIORITY[
                    "ambiguous_fail_closed"
                ]:
                    record["status"] = "ambiguous_fail_closed"


def _invalidate_surface_records(
    surface_records: Sequence[dict[str, Any]],
    *,
    reason_code: str,
    message: str,
) -> None:
    for record in surface_records:
        record["status"] = "missing_or_unreadable"
        record["scan_snapshot_valid"] = False
        _append_unique_issue(
            record,
            _issue(reason_code, "critical", message),
        )


def _dataset_key(row: Mapping[str, Any], ordinal: int) -> str:
    dataset_id = row.get("dataset_id")
    if dataset_id not in (None, ""):
        return str(dataset_id)
    registry_rowid = row.get("_registry_rowid")
    if registry_rowid not in (None, ""):
        return f"registry_rowid:{registry_rowid}"
    return f"row:{ordinal:08d}:{_fingerprint(row)[:16]}"


def _recording_key(row: Mapping[str, Any], ordinal: int) -> str:
    recording_id = row.get("recording_id")
    if recording_id not in (None, ""):
        return str(recording_id)
    registry_rowid = row.get("_registry_rowid")
    if registry_rowid not in (None, ""):
        return f"registry_rowid:{registry_rowid}"
    return f"recording_row:{ordinal:08d}:{_fingerprint(row)[:16]}"


def _registry_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _json_safe(value) for key, value in sorted(row.items())}


_CONTROLLED_ZARR_USES = frozenset(
    {"analysis", "training", "inference", "export", "archive"}
)
_CONTROLLED_ZARR_ORIGINS = frozenset({"source", "derived", "imported"})
_CONTROLLED_DATASET_ARTIFACT_KINDS = frozenset(
    {
        "source_recording",
        "derived_analysis",
        "derived_training_merge",
        "model_input_export",
    }
)
_ARTIFACT_REQUIRED_ZARR_USE = {
    "derived_analysis": "analysis",
    "derived_training_merge": "training",
    "model_input_export": "export",
}
_ARTIFACT_ALLOWED_ZARR_ORIGINS = {
    # Imported external recordings remain source artifacts; ``imported`` is
    # provenance, while artifact_kind continues to describe their role.
    "source_recording": frozenset({"source", "imported"}),
    "derived_analysis": frozenset({"derived"}),
    "derived_training_merge": frozenset({"derived"}),
    "model_input_export": frozenset({"derived"}),
}
_OPTIONAL_RECORDING_ARTIFACT_KINDS = frozenset(
    {"derived_analysis", "derived_training_merge", "model_input_export"}
)


def _dataset_role_issues(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Validate exact registry vocabulary; substrings carry no semantics."""

    zarr_use = row.get("zarr_use")
    zarr_origin = row.get("zarr_origin")
    artifact_kind = row.get("artifact_kind")
    issues: list[dict[str, Any]] = []
    if type(zarr_use) is not str or zarr_use not in _CONTROLLED_ZARR_USES:
        issues.append(
            _issue(
                "REGISTRY_ZARR_USE_UNCONTROLLED",
                "critical",
                "datasets.zarr_use must use the exact controlled registry vocabulary.",
                zarr_use=zarr_use,
                allowed=sorted(_CONTROLLED_ZARR_USES),
            )
        )
    if (
        type(zarr_origin) is not str
        or zarr_origin not in _CONTROLLED_ZARR_ORIGINS
    ):
        issues.append(
            _issue(
                "REGISTRY_ZARR_ORIGIN_UNCONTROLLED",
                "critical",
                "datasets.zarr_origin must use the exact controlled registry vocabulary.",
                zarr_origin=zarr_origin,
                allowed=sorted(_CONTROLLED_ZARR_ORIGINS),
            )
        )
    if (
        type(artifact_kind) is not str
        or artifact_kind not in _CONTROLLED_DATASET_ARTIFACT_KINDS
    ):
        issues.append(
            _issue(
                "REGISTRY_ARTIFACT_KIND_UNCONTROLLED",
                "critical",
                "datasets.artifact_kind must use the exact controlled registry vocabulary.",
                artifact_kind=artifact_kind,
                allowed=sorted(_CONTROLLED_DATASET_ARTIFACT_KINDS),
            )
        )
        return issues
    required_use = _ARTIFACT_REQUIRED_ZARR_USE.get(artifact_kind)
    if required_use is not None and zarr_use != required_use:
        issues.append(
            _issue(
                "REGISTRY_DATASET_ROLE_CONFLICT",
                "critical",
                "The controlled artifact kind conflicts with datasets.zarr_use.",
                artifact_kind=artifact_kind,
                zarr_use=zarr_use,
                required_zarr_use=required_use,
            )
        )
    allowed_origins = _ARTIFACT_ALLOWED_ZARR_ORIGINS[artifact_kind]
    if zarr_origin not in allowed_origins:
        issues.append(
            _issue(
                "REGISTRY_DATASET_ORIGIN_CONFLICT",
                "critical",
                "The controlled artifact kind conflicts with datasets.zarr_origin.",
                artifact_kind=artifact_kind,
                zarr_origin=zarr_origin,
                allowed_zarr_origins=sorted(allowed_origins),
            )
        )
    return issues


def _dataset_requires_recording_binding(row: Mapping[str, Any]) -> bool:
    artifact_kind = row.get("artifact_kind")
    return artifact_kind not in _OPTIONAL_RECORDING_ARTIFACT_KINDS


def _normalized_filters(values: Sequence[str] | None) -> tuple[str, ...]:
    return tuple(sorted({str(value).strip() for value in (values or ()) if str(value).strip()}))


def _surface_matches_run_families(
    relative_path: str,
    run_families: Sequence[str],
) -> bool:
    if not run_families:
        return True
    normalized_path = PurePosixPath(relative_path).as_posix().strip("/")
    parts = PurePosixPath(normalized_path).parts
    for raw_family in run_families:
        family = PurePosixPath(raw_family.strip("/")).as_posix()
        if not family:
            continue
        if "/" not in family and family in parts:
            return True
        if (
            normalized_path == family
            or normalized_path.startswith(f"{family}/")
            or f"/{family}/" in f"/{normalized_path}/"
        ):
            return True
    return False


def _expected_surface_identities(
    nodes: Sequence[MetadataNode],
    *,
    run_families: Sequence[str] = (),
) -> list[dict[str, str]]:
    node_map = {node.relative_path: node for node in nodes}
    identities: list[dict[str, str]] = []
    for node in nodes:
        surface_type = classify_surface(node.relative_path, node, node_map)
        if surface_type is None or not _surface_matches_run_families(
            node.relative_path,
            run_families,
        ):
            continue
        identities.append(
            {
                "surface_path": node.relative_path,
                "surface_type": surface_type,
            }
        )
    return sorted(
        identities,
        key=lambda item: (item["surface_path"], item["surface_type"]),
    )


_EXPLICIT_RUN_PARTITIONS: dict[str, frozenset[str]] = {
    "track_kinematics_runs": frozenset({"online", "offline"}),
}


def _run_pointer_names(attrs: Mapping[str, Any]) -> list[str]:
    return sorted(
        str(key)
        for key in attrs
        if str(key) in {"authoritative_run", "selected_run"}
        or str(key).startswith("latest")
    )


def _pointer_requires_complete(pointer: str) -> bool:
    return pointer in {"authoritative_run", "selected_run", "latest_complete"} or (
        pointer.startswith("latest_complete_")
    )


def _resolve_run_pointer_path(
    *,
    family_path: str,
    family_name: str,
    pointer: str,
    value: Any,
) -> str | None:
    if not isinstance(value, str) or not value or value.startswith("/"):
        return None
    candidate = PurePosixPath(value)
    if any(part in {"", ".", ".."} for part in candidate.parts):
        return None
    if "/" in value:
        return f"{family_path}/{candidate.as_posix()}"
    suffix = pointer.removeprefix("latest_")
    partitions = _EXPLICIT_RUN_PARTITIONS.get(family_name, frozenset())
    if suffix in partitions:
        return f"{family_path}/{suffix}/{value}"
    return f"{family_path}/{value}"


def _run_context_for_surface(
    relative_path: str,
    nodes: Mapping[str, MetadataNode],
) -> dict[str, Any]:
    """Return a normalized, non-authoritative run context for every surface."""

    parts = PurePosixPath(relative_path).parts
    family_index = next(
        (index for index, part in enumerate(parts) if part.endswith("_runs")),
        None,
    )
    empty = {
        "family": None,
        "family_path": None,
        "partition": None,
        "run_path": None,
        "run_name": None,
        "run_leaf_name": None,
        "completion_contract": None,
        "completion_status": None,
        "publication_status": None,
        "pointer_set": {
            "selected": [],
            "latest": [],
            "authoritative": [],
            "all_matching": [],
        },
    }
    if family_index is None:
        return empty
    family_name = parts[family_index]
    family_path = PurePosixPath(*parts[: family_index + 1]).as_posix()
    tail = parts[family_index + 1 :]
    if not tail:
        return {**empty, "family": family_name, "family_path": family_path}
    partitions = _EXPLICIT_RUN_PARTITIONS.get(family_name, frozenset())
    partition = tail[0] if tail[0] in partitions else None
    if partition is not None:
        if len(tail) < 2:
            return {
                **empty,
                "family": family_name,
                "family_path": family_path,
                "partition": partition,
            }
        run_tail = (partition, tail[1])
    else:
        run_tail = (tail[0],)
    run_name = PurePosixPath(*run_tail).as_posix()
    run_path = f"{family_path}/{run_name}"
    run_node = nodes.get(run_path)
    run_attrs = run_node.attributes if run_node is not None else {}
    family_node = nodes.get(family_path)
    pointer_records: list[dict[str, Any]] = []
    if family_node is not None:
        for pointer in _run_pointer_names(family_node.attributes):
            value = family_node.attributes.get(pointer)
            target_path = _resolve_run_pointer_path(
                family_path=family_path,
                family_name=family_name,
                pointer=pointer,
                value=value,
            )
            if target_path == run_path:
                pointer_records.append(
                    {
                        "pointer": pointer,
                        "value": value,
                        "target_path": target_path,
                        "completion_required": _pointer_requires_complete(pointer),
                    }
                )
    pointer_names = [record["pointer"] for record in pointer_records]
    return {
        "family": family_name,
        "family_path": family_path,
        "partition": partition,
        "run_path": run_path,
        "run_name": run_name,
        "run_leaf_name": run_tail[-1],
        "completion_contract": run_attrs.get("palette_run_completion_contract"),
        "completion_status": run_attrs.get("palette_run_completion_status"),
        "publication_status": (
            run_attrs.get("palette_run_publication_status")
            or run_attrs.get("publication_status")
            or run_attrs.get("publish_status")
        ),
        "pointer_set": {
            "selected": sorted(
                pointer for pointer in pointer_names if pointer.startswith("selected")
            ),
            "latest": sorted(
                pointer for pointer in pointer_names if pointer.startswith("latest")
            ),
            "authoritative": sorted(
                pointer
                for pointer in pointer_names
                if pointer.startswith("authoritative")
            ),
            "all_matching": sorted(pointer_names),
            "records": pointer_records,
        },
    }


def _run_pointer_contract_issues(
    nodes: Mapping[str, MetadataNode],
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for path, parent in nodes.items():
        if parent.node_type != "group" or not PurePosixPath(path).name.endswith(
            "_runs"
        ):
            continue
        attrs = parent.attributes
        epoch = attrs.get("palette_completion_epoch")
        strict_epoch = isinstance(epoch, int) and not isinstance(epoch, bool) and epoch >= 1
        family_name = PurePosixPath(path).name
        pointer_names = _run_pointer_names(attrs)
        resolved_pointer_targets: set[str] = set()
        partition_paths: set[str] = {
            f"{path}/{partition}"
            for partition in _EXPLICIT_RUN_PARTITIONS.get(
                family_name, frozenset()
            )
            if f"{path}/{partition}" in nodes
            and nodes[f"{path}/{partition}"].node_type == "group"
        }
        for pointer in pointer_names:
            if pointer not in attrs:
                continue
            value = attrs.get(pointer)
            target_path = _resolve_run_pointer_path(
                family_path=path,
                family_name=family_name,
                pointer=pointer,
                value=value,
            )
            if isinstance(value, str) and "/" in value:
                partition = value.split("/", 1)[0]
                if partition in _EXPLICIT_RUN_PARTITIONS.get(
                    family_name, frozenset()
                ):
                    partition_paths.add(f"{path}/{partition}")
            suffix = pointer.removeprefix("latest_")
            if suffix in _EXPLICIT_RUN_PARTITIONS.get(family_name, frozenset()):
                partition_paths.add(f"{path}/{suffix}")
            target = nodes.get(target_path or "")
            if target is None or target.node_type != "group":
                issues.append(
                    _issue(
                        "RUN_POINTER_UNRESOLVED",
                        "critical",
                        "Runs-parent pointer does not resolve to one exact child run group.",
                        runs_parent=path,
                        pointer=pointer,
                        value=value,
                    )
                )
                continue
            if target_path in partition_paths:
                issues.append(
                    _issue(
                        "RUN_POINTER_TARGET_IS_PARTITION",
                        "critical",
                        "A run pointer resolves to an organizational partition group instead of a concrete run.",
                        runs_parent=path,
                        pointer=pointer,
                        target_path=target_path,
                    )
                )
                continue
            resolved_pointer_targets.add(target_path)
            status = target.attributes.get("palette_run_completion_status")
            if _pointer_requires_complete(pointer) and status != "complete":
                issues.append(
                    _issue(
                        "RUN_POINTER_COMPLETION_MISMATCH",
                        "critical",
                        "A selected/latest-complete run pointer targets a run without a complete marker.",
                        runs_parent=path,
                        pointer=pointer,
                        target_path=target_path,
                        completion_status=status,
                    )
                )
            run_name = target.attributes.get("palette_run_name")
            if run_name not in (None, PurePosixPath(target_path).name):
                issues.append(
                    _issue(
                        "RUN_POINTER_NAME_MISMATCH",
                        "critical",
                        "Persisted run identity disagrees with its selected child path.",
                        target_path=target_path,
                        declared_run_name=run_name,
                    )
                )
        if strict_epoch:
            candidate_paths = {
                child_path
                for child_path, child in nodes.items()
                if child.node_type == "group"
                and str(PurePosixPath(child_path).parent) == path
                and child_path not in partition_paths
            }
            candidate_paths.update(
                child_path
                for child_path, child in nodes.items()
                if child.node_type == "group"
                and str(PurePosixPath(child_path).parent) in partition_paths
            )
            candidate_paths.update(resolved_pointer_targets)
            for child_path in sorted(candidate_paths):
                child = nodes[child_path]
                child_attrs = child.attributes
                if child_attrs.get("palette_run_completion_contract") != (
                    "palette.zarr_run_completion.v1"
                ) or child_attrs.get("palette_run_completion_status") not in {
                    "running",
                    "complete",
                    "failed",
                }:
                    issues.append(
                        _issue(
                            "RUN_COMPLETION_SCHEMA_INVALID",
                            "critical",
                            "A completion-epoch runs parent contains a child without the controlled run-completion schema.",
                            runs_parent=path,
                            run_path=child.relative_path,
                        )
                    )
    return issues


_ACQUISITION_AUTHORITY_SIGNAL_ATTRS = frozenset(
    {
        ACQUISITION_CAMERA_FRAME_ATTR,
        ACQUISITION_CAMERA_FRAME_DIGEST_ATTR,
        ACQUISITION_IMPORT_OWNERSHIP_ATTR,
        ACQUISITION_IMPORT_OWNERSHIP_DIGEST_ATTR,
    }
)
_ACQUISITION_MANIFEST_SIGNAL_ATTRS = frozenset(
    {
        ACQUISITION_PHYSICAL_CHUNK_MANIFEST_ATTR,
        ACQUISITION_PHYSICAL_CHUNK_MANIFEST_DIGEST_ATTR,
        ACQUISITION_MATERIALIZATION_MANIFEST_ATTR,
        ACQUISITION_MATERIALIZATION_MANIFEST_DIGEST_ATTR,
    }
)


def _acquisition_signal_summary(
    *,
    node: MetadataNode,
    attribute: str,
) -> dict[str, Any]:
    """Return bounded evidence for one acquisition declaration occurrence."""

    value = node.attributes.get(attribute)
    try:
        value_sha256 = _fingerprint(value)
    except (TypeError, ValueError):
        value_sha256 = None
    summary: dict[str, Any] = {
        "node_path": node.relative_path,
        "node_type": node.node_type,
        "attribute": attribute,
        "value_type": type(value).__name__,
        "value_sha256": value_sha256,
    }
    if attribute == _ACQUISITION_AUTHORITY_STATUS_ATTR:
        summary["value"] = _json_safe(value)
    elif isinstance(value, str):
        summary["value"] = value
    elif isinstance(value, Mapping):
        summary["record_identity"] = {
            key: _json_safe(value.get(key))
            for key in (
                "schema_id",
                "schema_version",
                "recording_id",
                "camera_id",
                "producer",
            )
            if key in value
        }
    return summary


def _known_acquisition_schema_attr_mismatches(
    nodes: Mapping[str, MetadataNode],
) -> list[dict[str, Any]]:
    """Find direct acquisition records hidden under noncanonical attr names.

    Attribute names are not authority, but neither may a producer evade the
    acquisition inventory by persisting a known acquisition ``schema_id``
    beneath an arbitrary name.  Only direct mapping attrs are inspected here;
    nested receipt records retain their documented containing schema.
    """

    mismatches: list[dict[str, Any]] = []
    for path, node in sorted(nodes.items()):
        for attribute, value in sorted(
            node.attributes.items(),
            key=lambda item: str(item[0]),
        ):
            if type(value) is not dict:
                continue
            schema_id = value.get("schema_id")
            expected_attribute = _KNOWN_ACQUISITION_SCHEMA_ATTRS.get(schema_id)
            if expected_attribute is None or attribute == expected_attribute:
                continue
            mismatches.append(
                {
                    **_acquisition_signal_summary(
                        node=node,
                        attribute=str(attribute),
                    ),
                    "schema_id": schema_id,
                    "expected_attribute": expected_attribute,
                }
            )
    return mismatches


def _dataset_acquisition_authority_inventory(
    nodes: Mapping[str, MetadataNode],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Audit acquisition publication independently of geometry references.

    Acquisition records are archive-level coordinate authority.  Looking at
    them only while resolving a geometry descriptor allowed corrupt, orphaned,
    or half-published authority to disappear from datasets with no downstream
    coordinate array.  This inventory enumerates every declaration first, then
    delegates the sole candidate's strict content validation to
    :func:`_reference_extent_binding_issues`.
    """

    signal_attributes = {
        *_ACQUISITION_AUTHORITY_SIGNAL_ATTRS,
        *_ACQUISITION_MANIFEST_SIGNAL_ATTRS,
        _ACQUISITION_AUTHORITY_STATUS_ATTR,
    }
    occurrences = [
        _acquisition_signal_summary(node=node, attribute=attribute)
        for path, node in sorted(nodes.items())
        for attribute in sorted(signal_attributes)
        if attribute in node.attributes
    ]
    schema_attr_mismatches = _known_acquisition_schema_attr_mismatches(nodes)
    occurrence_by_attr = {
        attribute: [
            occurrence
            for occurrence in occurrences
            if occurrence["attribute"] == attribute
        ]
        for attribute in signal_attributes
    }
    authority_container_path = "analysis/acquisition_camera_frames"
    authority_container_present = authority_container_path in nodes
    authority_nodes = sorted(
        path
        for path in nodes
        if tuple(PurePosixPath(path).parts[:2]) == (
            "analysis",
            "acquisition_camera_frames",
        )
        and len(PurePosixPath(path).parts) == 3
    )
    manifest_node_present = ACQUISITION_MATERIALIZATION_MANIFEST_PATH in nodes
    root_node = nodes.get(".") or nodes.get("")
    raw_video_node = nodes.get("raw_video")
    applicability_evidence = {
        "root_source_video_metadata": bool(
            root_node is not None
            and "source_video_metadata" in root_node.attributes
        ),
        "raw_video_source_locator": bool(
            raw_video_node is not None
            and any(
                name in raw_video_node.attributes
                for name in ("source_path", "source_video", "source_video_path")
            )
        ),
        "materialized_images_full": "raw_video/images_full" in nodes,
    }
    acquisition_applicable = any(applicability_evidence.values())
    has_any_signal = bool(
        occurrences
        or schema_attr_mismatches
        or authority_container_present
        or authority_nodes
        or manifest_node_present
    )
    inventory: dict[str, Any] = {
        "inventory_status": "not_applicable_unscanned",
        "publication_state": "absent",
        "signal_count": len(occurrences),
        "signals": occurrences,
        "schema_attr_mismatches": schema_attr_mismatches,
        "authority_container_present": authority_container_present,
        "authority_node_paths": authority_nodes,
        "materialization_manifest_node_present": manifest_node_present,
        "validated_authority_path": None,
        "authority_mode": None,
        "applicable": acquisition_applicable,
        "applicability_evidence": applicability_evidence,
        "validation_issue_codes": [],
    }
    if not has_any_signal:
        if not acquisition_applicable:
            return inventory, []
        issue = _issue(
            "ACQUISITION_AUTHORITY_MISSING",
            "critical",
            "Source-video/materialized acquisition evidence exists without an explicit publication status and acquisition authority.",
            applicability_evidence=applicability_evidence,
        )
        inventory["inventory_status"] = "ambiguous_fail_closed"
        inventory["validation_issue_codes"] = [issue["code"]]
        return inventory, [issue]

    issues: list[dict[str, Any]] = []

    def add(code: str, message: str, **evidence: Any) -> None:
        issues.append(_issue(code, "critical", message, **evidence))

    def add_warning(code: str, message: str, **evidence: Any) -> None:
        issues.append(_issue(code, "warning", message, **evidence))

    if schema_attr_mismatches:
        add(
            "ACQUISITION_SCHEMA_ATTR_MISMATCH",
            "Known acquisition schemas must be stored only under their exact controlled attribute names.",
            mismatches=schema_attr_mismatches,
        )

    authority_misplaced = [
        occurrence
        for attribute in _ACQUISITION_AUTHORITY_SIGNAL_ATTRS
        for occurrence in occurrence_by_attr[attribute]
        if occurrence["node_path"] not in authority_nodes
    ]
    manifest_misplaced = [
        occurrence
        for attribute in _ACQUISITION_MANIFEST_SIGNAL_ATTRS
        for occurrence in occurrence_by_attr[attribute]
        if occurrence["node_path"] != ACQUISITION_MATERIALIZATION_MANIFEST_PATH
    ]
    status_misplaced = [
        occurrence
        for occurrence in occurrence_by_attr[_ACQUISITION_AUTHORITY_STATUS_ATTR]
        if occurrence["node_path"] not in {".", "raw_video"}
    ]
    if authority_misplaced or manifest_misplaced or status_misplaced:
        add(
            "ACQUISITION_AUTHORITY_SIGNAL_MISPLACED",
            "Acquisition authority, publication, and manifest attrs are valid only at their exact canonical archive paths.",
            authority_signals=authority_misplaced,
            manifest_signals=manifest_misplaced,
            publication_signals=status_misplaced,
        )

    frame_occurrences = occurrence_by_attr[ACQUISITION_CAMERA_FRAME_ATTR]
    ownership_occurrences = occurrence_by_attr[ACQUISITION_IMPORT_OWNERSHIP_ATTR]
    authority_occurrence_paths = {
        str(occurrence["node_path"])
        for attribute in _ACQUISITION_AUTHORITY_SIGNAL_ATTRS
        for occurrence in occurrence_by_attr[attribute]
    }
    if (
        len(frame_occurrences) > 1
        or len(ownership_occurrences) > 1
        or len(authority_nodes) > 1
    ):
        add(
            "ACQUISITION_AUTHORITY_MULTIPLE",
            "A source-recording dataset must not advertise multiple acquisition-camera authorities.",
            authority_node_paths=authority_nodes,
            acquisition_record_paths=[
                occurrence["node_path"] for occurrence in frame_occurrences
            ],
            ownership_record_paths=[
                occurrence["node_path"] for occurrence in ownership_occurrences
            ],
        )
    if len(authority_occurrence_paths) > 1:
        add(
            "ACQUISITION_AUTHORITY_RECORD_SPLIT",
            "Acquisition frame, ownership, and their digests must be co-located on one canonical node.",
            declaration_paths=sorted(authority_occurrence_paths),
        )
    if authority_container_present and not authority_nodes:
        add(
            "ACQUISITION_AUTHORITY_INCOMPLETE",
            "The acquisition-camera authority container has no exact camera child authority.",
            authority_container_path=authority_container_path,
        )
    for path in sorted(set(authority_nodes) | authority_occurrence_paths):
        node = nodes.get(path)
        present = (
            set(node.attributes).intersection(_ACQUISITION_AUTHORITY_SIGNAL_ATTRS)
            if node is not None
            else set()
        )
        if (
            node is None
            or node.node_type != "group"
            or present != _ACQUISITION_AUTHORITY_SIGNAL_ATTRS
        ):
            add(
                "ACQUISITION_AUTHORITY_INCOMPLETE",
                "Each acquisition authority node requires one co-located frame record, ownership record, and both exact digests.",
                node_path=path,
                node_type=node.node_type if node is not None else None,
                present_attributes=sorted(present),
                missing_attributes=sorted(
                    _ACQUISITION_AUTHORITY_SIGNAL_ATTRS - present
                ),
            )
    if not frame_occurrences and (
        authority_container_present or authority_occurrence_paths
    ):
        add(
            "ACQUISITION_AUTHORITY_INCOMPLETE",
            "Acquisition authority signals exist without one acquisition-camera frame record.",
        )

    manifest_occurrence_paths = {
        str(occurrence["node_path"])
        for attribute in _ACQUISITION_MANIFEST_SIGNAL_ATTRS
        for occurrence in occurrence_by_attr[attribute]
    }
    if len(manifest_occurrence_paths) > 1 or any(
        len(occurrence_by_attr[attribute]) > 1
        for attribute in _ACQUISITION_MANIFEST_SIGNAL_ATTRS
    ):
        add(
            "ACQUISITION_MATERIALIZATION_MANIFEST_SPLIT",
            "Materialization and physical-object manifest records and digests must be unique and co-located.",
            declaration_paths=sorted(manifest_occurrence_paths),
        )
    if manifest_node_present or manifest_occurrence_paths:
        manifest_node = nodes.get(ACQUISITION_MATERIALIZATION_MANIFEST_PATH)
        present = (
            set(manifest_node.attributes).intersection(
                _ACQUISITION_MANIFEST_SIGNAL_ATTRS
            )
            if manifest_node is not None
            else set()
        )
        if (
            manifest_node is None
            or manifest_node.node_type != "group"
            or present != _ACQUISITION_MANIFEST_SIGNAL_ATTRS
        ):
            add(
                "ACQUISITION_MATERIALIZATION_MANIFEST_INCOMPLETE",
                "The canonical materialization node requires both manifest records and their exact digests.",
                node_type=(
                    manifest_node.node_type if manifest_node is not None else None
                ),
                present_attributes=sorted(present),
                missing_attributes=sorted(
                    _ACQUISITION_MANIFEST_SIGNAL_ATTRS - present
                ),
            )
        if not frame_occurrences:
            add(
                "ACQUISITION_MATERIALIZATION_MANIFEST_ORPHAN",
                "A materialization manifest exists without one acquisition-camera authority.",
                manifest_path=ACQUISITION_MATERIALIZATION_MANIFEST_PATH,
            )

    # One candidate is validated through the existing strict reference resolver;
    # this is the only place that parses its acquisition/ownership/manifest
    # content, avoiding a second subtly different dataset-level schema.
    if len(frame_occurrences) == 1:
        authority_path = str(frame_occurrences[0]["node_path"])
        authority_node = nodes.get(authority_path)
        raw_frame = _as_mapping(
            authority_node.attributes.get(ACQUISITION_CAMERA_FRAME_ATTR)
            if authority_node is not None
            else None
        )
        pointer = {
            "record_ref": f"/{authority_path}@{ACQUISITION_CAMERA_FRAME_ATTR}",
            "record_sha256": (
                authority_node.attributes.get(
                    ACQUISITION_CAMERA_FRAME_DIGEST_ATTR
                )
                if authority_node is not None
                else None
            ),
            "selector": ACQUISITION_CAMERA_FRAME_ATTR,
            "width": raw_frame.get("width_px"),
            "height": raw_frame.get("height_px"),
            "units": "px",
        }
        _target, validation_issues = _reference_extent_binding_issues(
            pointer,
            role="dataset.acquisition_camera_frame",
            nodes=nodes,
        )
        issues.extend(validation_issues)
        inventory["validated_authority_path"] = authority_path

    resolved_authority_mode: str | None = None
    if len(ownership_occurrences) == 1:
        ownership_path = str(ownership_occurrences[0]["node_path"])
        ownership_node = nodes.get(ownership_path)
        try:
            parsed_ownership = parse_acquisition_import_ownership(
                ownership_node.attributes.get(ACQUISITION_IMPORT_OWNERSHIP_ATTR)
                if ownership_node is not None
                else None
            )
        except PixelFrameAuthorityError:
            pass
        else:
            resolved_authority_mode = parsed_ownership.mode
            inventory["authority_mode"] = resolved_authority_mode

    status_occurrences = occurrence_by_attr[_ACQUISITION_AUTHORITY_STATUS_ATTR]
    status_by_path = {
        path: nodes[path].attributes.get(_ACQUISITION_AUTHORITY_STATUS_ATTR)
        for path in (".", "raw_video")
        if path in nodes
        and _ACQUISITION_AUTHORITY_STATUS_ATTR in nodes[path].attributes
    }
    status_required = bool(
        authority_container_present
        or authority_occurrence_paths
        or manifest_node_present
        or manifest_occurrence_paths
    )
    if len(status_occurrences) != 2 or set(status_by_path) != {".", "raw_video"}:
        if status_required or status_occurrences:
            add(
                "ACQUISITION_PUBLICATION_STATUS_MISSING",
                "Acquisition publication state must be persisted exactly and equally at archive root and raw_video.",
                status_paths=sorted(status_by_path),
                occurrence_count=len(status_occurrences),
            )

    status_record: Any = None
    if set(status_by_path) == {".", "raw_video"}:
        if not _exact_json_equal(status_by_path["."], status_by_path["raw_video"]):
            add(
                "ACQUISITION_PUBLICATION_STATUS_CONFLICT",
                "Root and raw_video acquisition publication records disagree.",
                root_status=status_by_path["."],
                raw_video_status=status_by_path["raw_video"],
            )
            inventory["publication_state"] = "conflicting"
        else:
            status_record = status_by_path["."]

    parsed_status = None
    if status_record is not None:
        try:
            parsed_status = parse_acquisition_authority_publication_status(
                status_record
            )
        except AcquisitionPublicationStatusError as exc:
            add(
                "ACQUISITION_PUBLICATION_STATUS_INVALID",
                "Acquisition publication state fails the shared exact controlled schema.",
                status=status_record,
                error=str(exc),
            )
            inventory["publication_state"] = "invalid"

    if parsed_status is not None:
        publication_state = parsed_status.status
        inventory["publication_state"] = publication_state
        inventory["declared_authority_mode"] = parsed_status.authority_mode
        sole_authority_path = (
            str(frame_occurrences[0]["node_path"])
            if len(frame_occurrences) == 1
            else None
        )
        if publication_state == _ACQUISITION_AUTHORITY_PUBLISHED:
            mode_evidence_valid = (
                parsed_status.authority_mode == resolved_authority_mode
                and parsed_status.authority_path == sole_authority_path
                and len(frame_occurrences) == 1
            )
            materialization_evidence_valid = (
                parsed_status.authority_mode
                == MATERIALIZED_ACQUISITION_AUTHORITY_MODE
                and manifest_node_present
            ) or (
                parsed_status.authority_mode
                == EXTERNAL_ACQUISITION_AUTHORITY_MODE
                and not manifest_node_present
                and not manifest_occurrence_paths
            )
            if not mode_evidence_valid or not materialization_evidence_valid:
                add(
                    "ACQUISITION_PUBLICATION_STATUS_CONFLICT",
                    "Published acquisition state conflicts with the sole authority mode/path or its mode-specific completion evidence.",
                    declared_authority_path=parsed_status.authority_path,
                    resolved_authority_path=sole_authority_path,
                    declared_authority_mode=parsed_status.authority_mode,
                    resolved_authority_mode=resolved_authority_mode,
                    manifest_present=manifest_node_present,
                )
        elif publication_state == _ACQUISITION_AUTHORITY_PENDING:
            if (
                resolved_authority_mode is not None
                and parsed_status.authority_mode != resolved_authority_mode
            ) or (
                parsed_status.authority_mode
                == EXTERNAL_ACQUISITION_AUTHORITY_MODE
                and (manifest_node_present or manifest_occurrence_paths)
            ):
                add(
                    "ACQUISITION_PUBLICATION_STATUS_CONFLICT",
                    "Pending acquisition state conflicts with persisted mode-specific evidence.",
                    declared_authority_mode=parsed_status.authority_mode,
                    resolved_authority_mode=resolved_authority_mode,
                    manifest_present=manifest_node_present,
                )
            add(
                "ACQUISITION_PUBLICATION_STATUS_PENDING",
                "Acquisition authority publication is incomplete and must fail closed until an exact retry finishes.",
                authority_path=parsed_status.authority_path,
                authority_mode=parsed_status.authority_mode,
            )
        elif publication_state == _ACQUISITION_AUTHORITY_NOT_PUBLISHED:
            if (
                authority_container_present
                or authority_occurrence_paths
                or manifest_node_present
                or manifest_occurrence_paths
            ):
                add(
                    "ACQUISITION_PUBLICATION_STATUS_CONFLICT",
                    "Noncanonical publication state cannot coexist with acquisition authority or materialization declarations.",
                    status=parsed_status.to_dict(),
                    authority_node_paths=authority_nodes,
                    manifest_present=manifest_node_present,
                )
            elif parsed_status.reason_code == "organized_recording_identity_absent":
                add(
                    "ACQUISITION_AUTHORITY_NOT_PUBLISHED",
                    "Acquisition authority was explicitly withheld because canonical recording identity is absent.",
                    reason_code=parsed_status.reason_code,
                )
            else:
                add_warning(
                    "ACQUISITION_AUTHORITY_NOT_PUBLISHED",
                    "Acquisition authority was explicitly withheld for a controlled noncanonical import condition.",
                    reason_code=parsed_status.reason_code,
                )

    validation_issue_codes = sorted({str(issue["code"]) for issue in issues})
    inventory["validation_issue_codes"] = validation_issue_codes
    if any(issue["severity"] in {"error", "critical"} for issue in issues):
        inventory["inventory_status"] = "ambiguous_fail_closed"
    elif inventory["publication_state"] == _ACQUISITION_AUTHORITY_PUBLISHED:
        inventory["inventory_status"] = "compatible"
    elif inventory["publication_state"] == _ACQUISITION_AUTHORITY_NOT_PUBLISHED:
        inventory["inventory_status"] = "recompute_required"
    return inventory, issues


def audit_dataset_row(
    row: Mapping[str, Any],
    *,
    ordinal: int = 0,
    _preloaded_nodes: Sequence[MetadataNode] | None = None,
    run_families: Sequence[str] | None = None,
    scan_scope: Mapping[str, Any] | None = None,
    registry_snapshot_fingerprint: str | None = None,
    scanner_binding: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Audit one registry dataset row and always return a dataset record."""

    registry = _registry_projection(row)
    normalized_run_families = _normalized_filters(run_families)
    normalized_scan_scope = _json_safe(scan_scope or {})
    normalized_scanner_binding = dict(
        _json_safe(scanner_binding or _scanner_source_binding())
    )
    key = _dataset_key(registry, ordinal)
    raw_path = registry.get("zarr_path")
    root_metadata_fingerprint = (
        _root_metadata_fingerprint(Path(str(raw_path)).expanduser())
        if raw_path not in (None, "")
        else None
    )
    base = {
        "audit_schema_id": AUDIT_SCHEMA_ID,
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "audit_ruleset_id": AUDIT_RULESET_ID,
        "audit_ruleset_version": AUDIT_RULESET_VERSION,
        "dataset_key": key,
        "dataset_id": registry.get("dataset_id"),
        "recording_id": registry.get("recording_id"),
        "zarr_path": raw_path,
        "registry_status": registry.get("status"),
        "zarr_origin": registry.get("zarr_origin"),
        "zarr_use": registry.get("zarr_use"),
        "artifact_kind": registry.get("artifact_kind"),
        "registry": registry,
        "registry_fingerprint": _fingerprint(registry),
        "root_metadata_fingerprint": root_metadata_fingerprint,
        "metadata_inventory_fingerprint": None,
        "scan_scope": normalized_scan_scope,
        "scan_scope_fingerprint": _fingerprint(normalized_scan_scope),
        "run_family_filters": list(normalized_run_families),
        "registry_snapshot_fingerprint": registry_snapshot_fingerprint,
        **normalized_scanner_binding,
    }

    dataset_issues: list[dict[str, Any]] = []
    if raw_path in (None, ""):
        dataset_issues.append(
            _issue("DATASET_ZARR_PATH_MISSING", "critical", "Registry row has no zarr_path.")
        )
        return _stamp_record_bundle([
            {
                **base,
                "record_type": "coordinate_dataset",
                "status": "missing_or_unreadable",
                "issues": dataset_issues,
                "issue_codes": [issue["code"] for issue in dataset_issues],
                "surface_count": 0,
                "scan_complete": True,
                "generation_complete": True,
                "expected_surface_identities": [],
                "expected_surface_identities_sha256": _fingerprint([]),
            }
        ])

    zarr_path = Path(str(raw_path)).expanduser()
    registry_status = str(registry.get("status") or "").lower()
    if registry_status == "missing":
        dataset_issues.append(
            _issue("REGISTRY_STATUS_MISSING", "critical", "Registry marks this dataset missing.")
        )
    try:
        path_exists = zarr_path.exists()
        path_is_dir = zarr_path.is_dir() if path_exists else False
    except OSError as exc:
        path_exists = False
        path_is_dir = False
        dataset_issues.append(
            _issue("DATASET_PATH_STAT_FAILED", "critical", "Dataset path could not be inspected.", error=str(exc))
        )
    if not path_exists or not path_is_dir:
        dataset_issues.append(
            _issue("DATASET_PATH_UNREACHABLE", "critical", "Dataset path is missing or not a directory.")
        )
        return _stamp_record_bundle([
            {
                **base,
                "record_type": "coordinate_dataset",
                "status": "missing_or_unreadable",
                "issues": dataset_issues,
                "issue_codes": sorted({issue["code"] for issue in dataset_issues}),
                "surface_count": 0,
                "scan_complete": True,
                "generation_complete": True,
                "expected_surface_identities": [],
                "expected_surface_identities_sha256": _fingerprint([]),
            }
        ])

    try:
        nodes_list = (
            list(_preloaded_nodes)
            if _preloaded_nodes is not None
            else list(iter_metadata_nodes(zarr_path))
        )
    except MetadataTraversalError as exc:
        dataset_issues.append(
            _issue(
                "ZARR_METADATA_TRAVERSAL_FAILED",
                "critical",
                "The metadata-only archive walk could not be completed.",
                error=str(exc),
            )
        )
        return _stamp_record_bundle([
            {
                **base,
                "record_type": "coordinate_dataset",
                "status": "missing_or_unreadable",
                "issues": dataset_issues,
                "issue_codes": sorted({issue["code"] for issue in dataset_issues}),
                "surface_count": 0,
                "discovered_surface_count": 0,
                "metadata_node_count": 0,
                "scan_complete": False,
                "generation_complete": False,
                "expected_surface_identities": [],
                "expected_surface_identities_sha256": _fingerprint([]),
            }
        ])
    base["metadata_inventory_fingerprint"] = _metadata_inventory_fingerprint(nodes_list)
    nodes = {node.relative_path: node for node in nodes_list}
    if "." not in nodes:
        dataset_issues.append(
            _issue(
                "ZARR_ROOT_METADATA_MISSING",
                "critical",
                "Dataset directory has no root zarr.json/.zgroup/.zarray metadata.",
            )
        )
        return _stamp_record_bundle([
            {
                **base,
                "record_type": "coordinate_dataset",
                "status": "missing_or_unreadable",
                "issues": dataset_issues,
                "issue_codes": sorted({issue["code"] for issue in dataset_issues}),
                "surface_count": 0,
                "scan_complete": True,
                "generation_complete": True,
                "expected_surface_identities": [],
                "expected_surface_identities_sha256": _fingerprint([]),
            }
        ])

    root_is_array = nodes["."].node_type == "array"
    if root_is_array:
        dataset_issues.append(
            _issue(
                "ZARR_ROOT_ARRAY_FORBIDDEN",
                "critical",
                "A registry dataset archive must be a Zarr group; a root array cannot provide an auditable analysis hierarchy.",
            )
        )

    root_attrs = nodes["."].attributes
    archive_registry_identity_issues: list[dict[str, Any]] = []
    for field in ("dataset_id", "recording_id"):
        expected = registry.get(field)
        declared = root_attrs.get(field)
        if expected in (None, ""):
            continue
        if declared in (None, ""):
            archive_registry_identity_issues.append(
                _issue(
                    "ARCHIVE_REGISTRY_IDENTITY_MISSING",
                    "critical",
                    "Archive root metadata does not persist the registry identity needed to prove this dataset/recording binding.",
                    field=field,
                    expected=expected,
                )
            )
        elif str(declared) != str(expected):
            archive_registry_identity_issues.append(
                _issue(
                    "ARCHIVE_REGISTRY_IDENTITY_MISMATCH",
                    "critical",
                    "Archive root identity conflicts with the selected registry row.",
                    field=field,
                    expected=expected,
                    declared=declared,
                )
            )
    dataset_issues.extend(archive_registry_identity_issues)
    run_pointer_issues = _run_pointer_contract_issues(nodes)
    dataset_issues.extend(run_pointer_issues)
    acquisition_inventory, acquisition_inventory_issues = (
        _dataset_acquisition_authority_inventory(nodes)
    )
    dataset_issues.extend(acquisition_inventory_issues)

    malformed_nodes = [node for node in nodes_list if node.metadata_error]
    if malformed_nodes:
        dataset_issues.append(
            _issue(
                "INVALID_ZARR_METADATA_INVENTORY",
                "critical",
                "One or more Zarr metadata nodes are malformed; the archive cannot be classified safely.",
                nodes=[
                    {"path": node.relative_path, "error": node.metadata_error}
                    for node in malformed_nodes
                ],
            )
        )

    surface_records: list[dict[str, Any]] = []
    discovered_surface_count = 0
    expected_surface_identities = _expected_surface_identities(
        nodes_list,
        run_families=normalized_run_families,
    )
    expected_surface_identities_sha256 = _fingerprint(expected_surface_identities)
    coordinate_bearing_node_count = sum(
        _node_is_coordinate_bearing(node, nodes) for node in nodes_list
    )
    for node in nodes_list:
        surface_type = classify_surface(node.relative_path, node, nodes)
        if surface_type is None:
            continue
        discovered_surface_count += 1
        if not _surface_matches_run_families(
            node.relative_path,
            normalized_run_families,
        ):
            continue
        result = classify_surface_contract(surface_type=surface_type, node=node, nodes=nodes)
        run_context = _run_context_for_surface(node.relative_path, nodes)
        surface_records.append(
            {
                **base,
                "record_type": "coordinate_surface",
                "surface_type": surface_type,
                "surface_path": node.relative_path,
                "run_context": run_context,
                "node_type": node.node_type,
                "metadata_format": node.metadata_format,
                "shape": _json_safe(node.shape),
                "data_type": _json_safe(node.data_type),
                "status": result["status"],
                "issues": result["issues"],
                "issue_codes": sorted({str(issue["code"]) for issue in result["issues"]}),
                "evidence": result["evidence"],
                "coordinate_descriptor": _json_safe(result["coordinate_descriptor"]),
                "descriptor_source": result["descriptor_source"],
                "descriptor_is_array_specific": result["descriptor_is_array_specific"],
                "legacy_compatibility_proof": result.get(
                    "legacy_compatibility_proof"
                ),
            }
        )
    surface_records.sort(key=lambda item: (str(item["surface_path"]), str(item["surface_type"])))
    _propagate_track_dependency_risk(surface_records, nodes)
    _validate_track_px_mm_coherence(surface_records, nodes)
    archive_registry_mismatch_issues = [
        issue
        for issue in archive_registry_identity_issues
        if issue["code"] == "ARCHIVE_REGISTRY_IDENTITY_MISMATCH"
    ]
    if archive_registry_mismatch_issues:
        for surface_record in surface_records:
            for identity_issue in archive_registry_mismatch_issues:
                _append_unique_issue(surface_record, identity_issue)
            if _STATUS_PRIORITY.get(
                str(surface_record.get("status")), 99
            ) < _STATUS_PRIORITY["ambiguous_fail_closed"]:
                surface_record["status"] = "ambiguous_fail_closed"
    # A second metadata-only walk closes the time-of-check/time-of-use gap.  A
    # mixed snapshot is never advertised as resumable or complete.
    post_scan_error: str | None = None
    try:
        post_scan_nodes = list(iter_metadata_nodes(zarr_path))
        post_scan_fingerprint = _metadata_inventory_fingerprint(post_scan_nodes)
    except MetadataTraversalError as exc:
        post_scan_nodes = []
        post_scan_fingerprint = None
        post_scan_error = str(exc)
    source_changed_during_scan = (
        post_scan_fingerprint != base["metadata_inventory_fingerprint"]
    )
    if post_scan_error is not None:
        dataset_issues.append(
            _issue(
                "POST_SCAN_METADATA_TRAVERSAL_FAILED",
                "critical",
                "Post-scan metadata verification could not be completed.",
                error=post_scan_error,
            )
        )
    if source_changed_during_scan:
        dataset_issues.append(
            _issue(
                "SOURCE_CHANGED_DURING_SCAN",
                "critical",
                "Zarr metadata changed between the audit snapshot and post-scan verification.",
                initial_metadata_inventory_fingerprint=base["metadata_inventory_fingerprint"],
                post_scan_metadata_inventory_fingerprint=post_scan_fingerprint,
            )
        )
    for surface_record in surface_records:
        surface_record["scan_snapshot_valid"] = not source_changed_during_scan and not malformed_nodes

    if malformed_nodes:
        _invalidate_surface_records(
            surface_records,
            reason_code="DATASET_METADATA_INVALIDATES_SURFACE",
            message="Malformed archive metadata invalidates every surface classification in this dataset snapshot.",
        )
    if root_is_array:
        _invalidate_surface_records(
            surface_records,
            reason_code="ROOT_ARRAY_INVALIDATES_SURFACE",
            message="A root-array archive cannot establish an auditable coordinate hierarchy.",
        )
    if source_changed_during_scan:
        _invalidate_surface_records(
            surface_records,
            reason_code="DATASET_SCAN_INVALIDATES_SURFACE",
            message="An incomplete or changing dataset scan invalidates every surface migration classification.",
        )

    if not surface_records and not normalized_run_families:
        if coordinate_bearing_node_count:
            dataset_issues.append(
                _issue(
                    "COORDINATE_BEARING_ARCHIVE_HAS_NO_AUDITED_SURFACES",
                    "critical",
                    "Coordinate declarations exist but exhaustive discovery produced no audited surface.",
                    coordinate_bearing_node_count=coordinate_bearing_node_count,
                )
            )
        else:
            dataset_issues.append(
                _issue(
                    "NO_COORDINATE_SURFACES_DETECTED",
                    "info",
                    "The archive contains no persisted coordinate declaration or descriptor.",
                )
            )
    elif not surface_records:
        dataset_issues.append(
            _issue(
                "RUN_FAMILY_FILTER_NO_MATCH",
                "info",
                "No covered coordinate surface matched the selected run-family filters.",
                run_family_filters=list(normalized_run_families),
                discovered_surface_count=discovered_surface_count,
            )
        )
    statuses = [str(record["status"]) for record in surface_records]
    acquisition_inventory_status = str(
        acquisition_inventory.get("inventory_status")
        or "not_applicable_unscanned"
    )
    if acquisition_inventory_status != "not_applicable_unscanned":
        statuses.append(acquisition_inventory_status)
    if statuses:
        dataset_status = max(statuses, key=lambda status: _STATUS_PRIORITY[status])
    elif normalized_run_families:
        dataset_status = "not_applicable_unscanned"
    elif coordinate_bearing_node_count:
        dataset_status = "ambiguous_fail_closed"
    else:
        dataset_status = "not_applicable_unscanned"
    if registry_status == "missing":
        dataset_status = "missing_or_unreadable"
        _invalidate_surface_records(
            surface_records,
            reason_code="REGISTRY_STATUS_INVALIDATES_SURFACE",
            message="Registry missing state invalidates every surface migration classification.",
        )
    if malformed_nodes:
        dataset_status = "missing_or_unreadable"
    if source_changed_during_scan:
        dataset_status = "missing_or_unreadable"
    if root_is_array:
        dataset_status = "missing_or_unreadable"
    archive_identity_issue_codes = {
        str(issue["code"]) for issue in archive_registry_identity_issues
    }
    if (
        "ARCHIVE_REGISTRY_IDENTITY_MISMATCH" in archive_identity_issue_codes
        and _STATUS_PRIORITY.get(dataset_status, 99)
        < _STATUS_PRIORITY["ambiguous_fail_closed"]
    ):
        dataset_status = "ambiguous_fail_closed"
    elif (
        "ARCHIVE_REGISTRY_IDENTITY_MISSING" in archive_identity_issue_codes
        and _STATUS_PRIORITY.get(dataset_status, 99)
        < _STATUS_PRIORITY["metadata_backfill_candidate"]
    ):
        dataset_status = "metadata_backfill_candidate"
    if (
        run_pointer_issues
        and _STATUS_PRIORITY.get(dataset_status, 99)
        < _STATUS_PRIORITY["ambiguous_fail_closed"]
    ):
        dataset_status = "ambiguous_fail_closed"
    for surface_record in surface_records:
        surface_record["archive_issues"] = _json_safe(dataset_issues)
        surface_record["archive_issue_codes"] = sorted(
            {str(issue["code"]) for issue in dataset_issues}
        )
    dataset_record = {
        **base,
        "record_type": "coordinate_dataset",
        "status": dataset_status,
        "issues": dataset_issues,
        "issue_codes": sorted({issue["code"] for issue in dataset_issues}),
        "surface_count": len(surface_records),
        "discovered_surface_count": discovered_surface_count,
        "coordinate_bearing_node_count": coordinate_bearing_node_count,
        "acquisition_authority_inventory": _json_safe(acquisition_inventory),
        "metadata_node_count": len(nodes_list),
        "scan_complete": (
            not source_changed_during_scan
            and not malformed_nodes
            and not root_is_array
            and registry_status != "missing"
        ),
        "generation_complete": (
            not source_changed_during_scan
            and not malformed_nodes
            and not root_is_array
            and registry_status != "missing"
        ),
        "expected_surface_identities": expected_surface_identities,
        "expected_surface_identities_sha256": expected_surface_identities_sha256,
        "post_scan_metadata_inventory_fingerprint": post_scan_fingerprint,
    }
    return _stamp_record_bundle([dataset_record, *surface_records])


def _load_resume_records(path: Path | None) -> dict[str, list[dict[str, Any]]]:
    if path is None or not path.is_file():
        return {}
    grouped: dict[str, list[dict[str, Any]]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    for line in lines:
        if not line.strip():
            continue
        try:
            record = _strict_json_loads(line)
        except (TypeError, ValueError):
            continue
        if not isinstance(record, dict) or record.get("audit_schema_id") != AUDIT_SCHEMA_ID:
            continue
        key = record.get("dataset_key")
        if key not in (None, ""):
            grouped.setdefault(str(key), []).append(record)
    reusable: dict[str, list[dict[str, Any]]] = {}
    for key, records in grouped.items():
        if _record_bundle_is_complete(key, records):
            reusable[key] = records
    return reusable


def _record_bundle_is_complete(
    dataset_key: str,
    records: Sequence[Mapping[str, Any]],
) -> bool:
    if not records:
        return False
    if any(
        record.get("audit_schema_id") != AUDIT_SCHEMA_ID
        or record.get("audit_schema_version") != AUDIT_SCHEMA_VERSION
        or record.get("audit_ruleset_id") != AUDIT_RULESET_ID
        or record.get("audit_ruleset_version") != AUDIT_RULESET_VERSION
        or str(record.get("dataset_key")) != str(dataset_key)
        for record in records
    ):
        return False
    binding_fields = (
        "scanner_source_sha256",
        "scanner_source_dirty",
        "ruleset_content_sha256",
        "contract_dependency_source_sha256",
        "repository_commit",
        "scanner_binding_sha256",
    )
    first = records[0]
    if any(field not in first for field in binding_fields):
        return False
    if any(
        any(record.get(field) != first.get(field) for field in binding_fields)
        for record in records[1:]
    ):
        return False
    declared_bundle_digests = {
        record.get("record_bundle_sha256") for record in records
    }
    if declared_bundle_digests != {_record_bundle_digest(records)}:
        return False
    dataset_records = [
        record for record in records if record.get("record_type") == "coordinate_dataset"
    ]
    surface_records = [
        record for record in records if record.get("record_type") == "coordinate_surface"
    ]
    if len(dataset_records) != 1:
        return False
    dataset = dataset_records[0]
    if dataset.get("scan_complete") is not True:
        return False
    if dataset.get("generation_complete") is not True:
        return False
    if dataset.get("surface_count") != len(surface_records):
        return False
    identities = [
        (str(record.get("surface_path") or ""), str(record.get("surface_type") or ""))
        for record in surface_records
    ]
    if len(identities) != len(set(identities)):
        return False
    expected = dataset.get("expected_surface_identities")
    if not isinstance(expected, list):
        return False
    normalized_expected = [
        (
            str(item.get("surface_path") or ""),
            str(item.get("surface_type") or ""),
        )
        for item in expected
        if isinstance(item, Mapping)
    ]
    if len(normalized_expected) != len(expected):
        return False
    if sorted(identities) != sorted(normalized_expected):
        return False
    if dataset.get("expected_surface_identities_sha256") != _fingerprint(expected):
        return False
    if any(record.get("scan_snapshot_valid") is False for record in surface_records):
        return False
    return True


def _checkpoint_path(checkpoint_dir: Path, dataset_key: str) -> Path:
    filename = f"dataset-{_fingerprint({'dataset_key': dataset_key})}.json"
    return checkpoint_dir / filename


def _write_dataset_checkpoint(
    checkpoint_dir: Path,
    dataset_key: str,
    records: Sequence[Mapping[str, Any]],
) -> None:
    normalized_records = [_json_safe(record) for record in records]
    dataset_record = next(
        record
        for record in normalized_records
        if record.get("record_type") == "coordinate_dataset"
    )
    payload = {
        "checkpoint_schema_id": CHECKPOINT_SCHEMA_ID,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "dataset_key": dataset_key,
        "audit_schema_id": AUDIT_SCHEMA_ID,
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "audit_ruleset_id": AUDIT_RULESET_ID,
        "audit_ruleset_version": AUDIT_RULESET_VERSION,
        "scanner_source_sha256": dataset_record.get("scanner_source_sha256"),
        "scanner_source_dirty": dataset_record.get("scanner_source_dirty"),
        "ruleset_content_sha256": dataset_record.get("ruleset_content_sha256"),
        "repository_commit": dataset_record.get("repository_commit"),
        "scanner_binding_sha256": dataset_record.get("scanner_binding_sha256"),
        "registry_fingerprint": dataset_record.get("registry_fingerprint"),
        "registry_snapshot_fingerprint": dataset_record.get(
            "registry_snapshot_fingerprint"
        ),
        "scan_scope_fingerprint": dataset_record.get("scan_scope_fingerprint"),
        "record_count": len(normalized_records),
        "surface_count": dataset_record.get("surface_count"),
        "expected_surface_identities_sha256": dataset_record.get(
            "expected_surface_identities_sha256"
        ),
        "generation_complete": dataset_record.get("generation_complete"),
        "records_sha256": _fingerprint(normalized_records),
        "record_bundle_sha256": dataset_record.get("record_bundle_sha256"),
        "records": normalized_records,
    }
    _atomic_write_text(
        _checkpoint_path(checkpoint_dir, dataset_key),
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )


def _load_checkpoint_records(checkpoint_dir: Path | None) -> dict[str, list[dict[str, Any]]]:
    if checkpoint_dir is None or not checkpoint_dir.is_dir():
        return {}
    reusable: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(checkpoint_dir.glob("dataset-*.json")):
        try:
            payload = _strict_json_loads(path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError):
            continue
        if not isinstance(payload, Mapping):
            continue
        if payload.get("checkpoint_schema_id") != CHECKPOINT_SCHEMA_ID:
            continue
        if payload.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
            continue
        if payload.get("audit_schema_id") != AUDIT_SCHEMA_ID:
            continue
        if payload.get("audit_schema_version") != AUDIT_SCHEMA_VERSION:
            continue
        if payload.get("audit_ruleset_id") != AUDIT_RULESET_ID:
            continue
        if payload.get("audit_ruleset_version") != AUDIT_RULESET_VERSION:
            continue
        key = payload.get("dataset_key")
        raw_records = payload.get("records")
        if key in (None, "") or not isinstance(raw_records, list):
            continue
        records = [record for record in raw_records if isinstance(record, dict)]
        if len(records) != len(raw_records):
            continue
        if payload.get("record_count") != len(records):
            continue
        if payload.get("records_sha256") != _fingerprint(records):
            continue
        if not _record_bundle_is_complete(str(key), records):
            continue
        dataset_record = next(
            record for record in records if record.get("record_type") == "coordinate_dataset"
        )
        if payload.get("surface_count") != dataset_record.get("surface_count"):
            continue
        if (
            payload.get("expected_surface_identities_sha256")
            != dataset_record.get("expected_surface_identities_sha256")
            or payload.get("generation_complete") is not True
        ):
            continue
        if payload.get("record_bundle_sha256") != dataset_record.get(
            "record_bundle_sha256"
        ):
            continue
        for field in (
            "registry_fingerprint",
            "registry_snapshot_fingerprint",
            "scan_scope_fingerprint",
            "scanner_source_sha256",
            "scanner_source_dirty",
            "ruleset_content_sha256",
            "repository_commit",
            "scanner_binding_sha256",
        ):
            if payload.get(field) != dataset_record.get(field):
                break
        else:
            reusable[str(key)] = records
    return reusable


def _dataset_matches_recording_filters(
    row: Mapping[str, Any],
    *,
    recording_ids: Sequence[str],
    recording_path_contains: Sequence[str],
    recordings_by_id: Mapping[str, Sequence[Mapping[str, Any]]],
) -> bool:
    recording_id = str(row.get("recording_id") or "")
    if recording_ids and recording_id not in set(recording_ids):
        return False
    if recording_path_contains:
        recording_paths = [
            str(record.get("recording_path") or "")
            for record in recordings_by_id.get(recording_id, ())
        ]
        if not any(
            needle in recording_path
            for needle in recording_path_contains
            for recording_path in recording_paths
        ):
            return False
    return True


def _registry_snapshot_fingerprint(
    recording_rows: Sequence[Mapping[str, Any]],
    dataset_rows: Sequence[Mapping[str, Any]],
) -> str:
    return _fingerprint(
        {
            "recordings": [_registry_projection(row) for row in recording_rows],
            "datasets": [_registry_projection(row) for row in dataset_rows],
        }
    )


def _duplicate_values(values: Sequence[str]) -> list[str]:
    counts = Counter(values)
    return sorted(value for value, count in counts.items() if count > 1)


def _resolved_path(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _is_same_or_descendant(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _validate_write_locations(
    *,
    registry_path: Path,
    dataset_rows: Sequence[Mapping[str, Any]],
    file_paths: Sequence[Path] = (),
    directory_paths: Sequence[Path] = (),
) -> None:
    registry_resolved = _resolved_path(registry_path)
    zarr_roots = [
        _resolved_path(Path(str(row["zarr_path"])))
        for row in dataset_rows
        if row.get("zarr_path") not in (None, "")
    ]
    resolved_files = [_resolved_path(path) for path in file_paths]
    resolved_dirs = [_resolved_path(path) for path in directory_paths]
    duplicate_outputs = _duplicate_values(
        [str(path) for path in [*resolved_files, *resolved_dirs]]
    )
    if duplicate_outputs:
        raise ValueError(f"audit output paths collide: {duplicate_outputs}")
    for output in resolved_files:
        if output == registry_resolved:
            raise ValueError("audit output file must not replace the source registry")
        if any(_is_same_or_descendant(output, root) for root in zarr_roots):
            raise ValueError(
                f"audit output file must not be inside a scanned Zarr archive: {output}"
            )
    for output_dir in resolved_dirs:
        if output_dir == registry_resolved:
            raise ValueError("audit output directory must not replace the source registry")
        if any(_is_same_or_descendant(output_dir, root) for root in zarr_roots):
            raise ValueError(
                f"audit output directory must not be inside a scanned Zarr archive: {output_dir}"
            )


def _registry_archive_alias_keys(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[str]]:
    identities: dict[tuple[Any, ...], list[str]] = {}
    labels: dict[tuple[Any, ...], str] = {}
    for ordinal, row in enumerate(rows):
        raw_path = row.get("zarr_path")
        if raw_path in (None, ""):
            continue
        path = Path(str(raw_path)).expanduser()
        try:
            resolved = path.resolve(strict=True)
            metadata = resolved.stat()
            identity: tuple[Any, ...] = (
                "inode",
                int(metadata.st_dev),
                int(metadata.st_ino),
            )
            labels[identity] = str(resolved)
        except OSError:
            identity = ("lexical", str(path.resolve(strict=False)))
            labels[identity] = str(path.resolve(strict=False))
        identities.setdefault(identity, []).append(_dataset_key(row, ordinal))
    result: dict[str, list[str]] = {}
    for identity, keys in identities.items():
        if len(keys) < 2:
            continue
        label = labels[identity]
        for key in keys:
            result[key] = sorted(other for other in keys if other != key) + [label]
    return result


def _partition_registry_integrity_issues(
    issues: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], dict[str, list[Mapping[str, Any]]]]:
    """Localize dataset foreign-key defects; keep structural defects global."""

    global_issues: list[Mapping[str, Any]] = []
    by_dataset: dict[str, list[Mapping[str, Any]]] = {}
    rowid_to_key = {
        str(row.get("_registry_rowid")): _dataset_key(row, ordinal)
        for ordinal, row in enumerate(rows)
        if row.get("_registry_rowid") not in (None, "")
    }
    for issue in issues:
        if issue.get("code") != "REGISTRY_FOREIGN_KEY_INVALID":
            global_issues.append(issue)
            continue
        evidence = _as_mapping(issue.get("evidence"))
        raw_defects = evidence.get("foreign_key_rows")
        defects = raw_defects if isinstance(raw_defects, list) else []
        unlocalized: list[Any] = []
        for defect in defects:
            if (
                isinstance(defect, list)
                and len(defect) >= 2
                and str(defect[0]) == "datasets"
                and str(defect[1]) in rowid_to_key
            ):
                key = rowid_to_key[str(defect[1])]
                by_dataset.setdefault(key, []).append(
                    _issue(
                        "REGISTRY_DATASET_FOREIGN_KEY_INVALID",
                        "critical",
                        "This dataset row violates a declared registry foreign-key relationship.",
                        foreign_key_row=defect,
                    )
                )
            else:
                unlocalized.append(defect)
        if unlocalized:
            global_issues.append(
                _issue(
                    "REGISTRY_FOREIGN_KEY_INVALID",
                    "critical",
                    "Registry foreign-key defects could not all be localized to one dataset row.",
                    foreign_key_rows=unlocalized,
                )
            )
    return global_issues, by_dataset


def _apply_registry_contract_issues(
    records: Sequence[dict[str, Any]],
    *,
    global_issues: Sequence[Mapping[str, Any]],
    row_issues: Sequence[Mapping[str, Any]],
) -> None:
    issues = [*global_issues, *row_issues]
    if not issues:
        return
    global_failure = bool(global_issues)
    for record in records:
        for issue in issues:
            _append_unique_issue(record, issue)
        if global_failure:
            record["status"] = "missing_or_unreadable"
            if record.get("record_type") == "coordinate_dataset":
                record["scan_complete"] = False
                record["generation_complete"] = False
            else:
                record["scan_snapshot_valid"] = False
        elif _STATUS_PRIORITY.get(str(record.get("status")), 99) < _STATUS_PRIORITY[
            "ambiguous_fail_closed"
        ]:
            record["status"] = "ambiguous_fail_closed"
        # Row-local registry findings are part of a complete, reusable scan
        # bundle.  They change migration status, not whether the read-only
        # inventory generation completed.


def audit_registry(
    registry_path: Path,
    *,
    resume_jsonl: Path | None = None,
    checkpoint_dir: Path | None = None,
    recording_ids: Sequence[str] | None = None,
    recording_path_contains: Sequence[str] | None = None,
    run_families: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Audit every registry row with optional validated resume/checkpoints.

    When *checkpoint_dir* is supplied, each completed dataset is durably
    replaced as one atomic JSON document before the next registry row begins.
    Checkpoints use the same registry and complete metadata fingerprint checks
    as ``resume_jsonl``; they are never trusted only because a filename exists.
    """

    recording_rows, rows = read_registry_snapshot_rows(registry_path)
    raw_registry_integrity_issues = _registry_integrity_issues(registry_path)
    registry_integrity_issues, registry_row_integrity_issues = (
        _partition_registry_integrity_issues(raw_registry_integrity_issues, rows)
    )
    scanner_binding = _scanner_source_binding()
    registry_snapshot_fingerprint = _registry_snapshot_fingerprint(
        recording_rows,
        rows,
    )
    dataset_keys = [_dataset_key(row, ordinal) for ordinal, row in enumerate(rows)]
    duplicate_dataset_keys = _duplicate_values(dataset_keys)
    if duplicate_dataset_keys:
        raise ValueError(
            "registry dataset rows do not have unique audit keys: "
            f"{duplicate_dataset_keys}"
        )
    alias_keys = _registry_archive_alias_keys(rows)
    if checkpoint_dir is not None:
        _validate_write_locations(
            registry_path=registry_path,
            dataset_rows=rows,
            file_paths=tuple(
                _checkpoint_path(checkpoint_dir, key) for key in dataset_keys
            ),
            directory_paths=(checkpoint_dir,),
        )
    normalized_recording_ids = _normalized_filters(recording_ids)
    normalized_path_filters = _normalized_filters(recording_path_contains)
    normalized_run_families = _normalized_filters(run_families)
    recordings_by_id: dict[str, list[Mapping[str, Any]]] = {}
    recording_paths: dict[str, list[Mapping[str, Any]]] = {}
    for recording in recording_rows:
        recordings_by_id.setdefault(str(recording.get("recording_id") or ""), []).append(
            recording
        )
        recording_path = recording.get("recording_path")
        if recording_path not in (None, ""):
            recording_paths.setdefault(str(recording_path), []).append(recording)
    duplicate_recording_paths_by_id: dict[str, list[str]] = {}
    for recording_path, matching_rows in recording_paths.items():
        if len(matching_rows) < 2:
            continue
        for matching_row in matching_rows:
            matching_id = matching_row.get("recording_id")
            if matching_id not in (None, ""):
                duplicate_recording_paths_by_id.setdefault(
                    str(matching_id), []
                ).append(recording_path)
    scan_scope = {
        "recording_ids": list(normalized_recording_ids),
        "recording_path_contains": list(normalized_path_filters),
        "run_families": list(normalized_run_families),
    }
    scan_scope_fingerprint = _fingerprint(scan_scope)
    resumed = _load_resume_records(resume_jsonl)
    resumed.update(_load_checkpoint_records(checkpoint_dir))
    records: list[dict[str, Any]] = []
    for ordinal, row in enumerate(rows):
        if not _dataset_matches_recording_filters(
            row,
            recording_ids=normalized_recording_ids,
            recording_path_contains=normalized_path_filters,
            recordings_by_id=recordings_by_id,
        ):
            continue
        key = _dataset_key(row, ordinal)
        registry_fingerprint = _fingerprint(_registry_projection(row))
        raw_path = row.get("zarr_path")
        root_metadata_fingerprint = (
            _root_metadata_fingerprint(Path(str(raw_path)).expanduser())
            if raw_path not in (None, "")
            else None
        )
        prior = resumed.get(key, [])
        prior_dataset = next(
            (record for record in prior if record.get("record_type") == "coordinate_dataset"), None
        )
        preloaded_nodes: list[MetadataNode] | None = None
        metadata_inventory_fingerprint: str | None = None
        if prior_dataset and raw_path not in (None, ""):
            candidate_path = Path(str(raw_path)).expanduser()
            try:
                candidate_is_dir = candidate_path.is_dir()
            except OSError:
                candidate_is_dir = False
            if candidate_is_dir:
                try:
                    preloaded_nodes = list(iter_metadata_nodes(candidate_path))
                except MetadataTraversalError:
                    preloaded_nodes = None
                else:
                    metadata_inventory_fingerprint = _metadata_inventory_fingerprint(preloaded_nodes)
        resume_matches = bool(
            prior_dataset
            and prior_dataset.get("registry_fingerprint") == registry_fingerprint
            and prior_dataset.get("root_metadata_fingerprint")
            == root_metadata_fingerprint
            and metadata_inventory_fingerprint is not None
            and prior_dataset.get("metadata_inventory_fingerprint")
            == metadata_inventory_fingerprint
            and prior_dataset.get("scan_scope_fingerprint")
            == scan_scope_fingerprint
            and prior_dataset.get("registry_snapshot_fingerprint")
            == registry_snapshot_fingerprint
            and all(
                prior_dataset.get(field) == scanner_binding.get(field)
                for field in (
                    "scanner_source_sha256",
                    "scanner_source_dirty",
                    "ruleset_content_sha256",
                    "contract_dependency_source_sha256",
                    "repository_commit",
                    "scanner_binding_sha256",
                )
            )
            and preloaded_nodes is not None
            and prior_dataset.get("expected_surface_identities_sha256")
            == _fingerprint(
                _expected_surface_identities(
                    preloaded_nodes,
                    run_families=normalized_run_families,
                )
            )
        )
        if resume_matches:
            # Verify the prospective reuse snapshot just as strictly as a new
            # scan.  If it moved, fall through to a fresh classification using
            # the most recent complete metadata snapshot.
            candidate_path = Path(str(raw_path)).expanduser()
            try:
                verified_nodes = list(iter_metadata_nodes(candidate_path))
            except MetadataTraversalError:
                verified_nodes = None
                verified_fingerprint = None
                verified_root_fingerprint = None
            else:
                verified_fingerprint = _metadata_inventory_fingerprint(verified_nodes)
                verified_root_fingerprint = _root_metadata_fingerprint(candidate_path)
            if (
                verified_fingerprint != metadata_inventory_fingerprint
                or verified_root_fingerprint != root_metadata_fingerprint
                or verified_nodes is None
                or prior_dataset.get("expected_surface_identities_sha256")
                != _fingerprint(
                    _expected_surface_identities(
                        verified_nodes or (),
                        run_families=normalized_run_families,
                    )
                )
            ):
                resume_matches = False
                preloaded_nodes = verified_nodes
        if resume_matches:
            dataset_records = sorted(
                prior,
                key=lambda record: (
                    0 if record.get("record_type") == "coordinate_dataset" else 1,
                    str(record.get("surface_path") or ""),
                ),
            )
            row_contract_issues: list[dict[str, Any]] = _dataset_role_issues(
                row
            )
            recording_id = row.get("recording_id")
            matching_recordings = recordings_by_id.get(str(recording_id or ""), [])
            if _dataset_requires_recording_binding(row) and (
                recording_id in (None, "") or len(matching_recordings) != 1
            ):
                row_contract_issues.append(
                    _issue(
                        "REGISTRY_DATASET_RECORDING_IDENTITY_INVALID",
                        "critical",
                        "Each dataset row must bind exactly one recordings row.",
                        recording_id=recording_id,
                        matching_recording_row_count=len(matching_recordings),
                    )
                )
            elif (
                not _dataset_requires_recording_binding(row)
                and recording_id not in (None, "")
                and len(matching_recordings) != 1
            ):
                row_contract_issues.append(
                    _issue(
                        "REGISTRY_OPTIONAL_RECORDING_IDENTITY_INVALID",
                        "critical",
                        "An optional derived/training recording_id was supplied but does not bind exactly one recordings row.",
                        recording_id=recording_id,
                        matching_recording_row_count=len(matching_recordings),
                    )
                )
            row_contract_issues.extend(registry_row_integrity_issues.get(key, ()))
            duplicate_paths = duplicate_recording_paths_by_id.get(
                str(recording_id or ""), []
            )
            if duplicate_paths:
                row_contract_issues.append(
                    _issue(
                        "REGISTRY_DUPLICATE_RECORDING_PATH",
                        "critical",
                        "The dataset's recordings row shares its recording_path with another recordings row.",
                        recording_id=recording_id,
                        recording_paths=sorted(duplicate_paths),
                    )
                )
            if key in alias_keys:
                row_contract_issues.append(
                    _issue(
                        "REGISTRY_ZARR_ARCHIVE_ALIAS",
                        "critical",
                        "Multiple dataset rows resolve to the same archive identity.",
                        alias_evidence=alias_keys[key],
                    )
                )
            _apply_registry_contract_issues(
                dataset_records,
                global_issues=registry_integrity_issues,
                row_issues=row_contract_issues,
            )
            _stamp_record_bundle(dataset_records)
            if checkpoint_dir is not None and _record_bundle_is_complete(
                key, dataset_records
            ):
                _write_dataset_checkpoint(checkpoint_dir, key, dataset_records)
            records.extend(dataset_records)
            continue
        dataset_records = audit_dataset_row(
            row,
            ordinal=ordinal,
            _preloaded_nodes=preloaded_nodes,
            run_families=normalized_run_families,
            scan_scope=scan_scope,
            registry_snapshot_fingerprint=registry_snapshot_fingerprint,
            scanner_binding=scanner_binding,
        )
        row_contract_issues = _dataset_role_issues(row)
        recording_id = row.get("recording_id")
        matching_recordings = recordings_by_id.get(str(recording_id or ""), [])
        if _dataset_requires_recording_binding(row) and (
            recording_id in (None, "") or len(matching_recordings) != 1
        ):
            row_contract_issues.append(
                _issue(
                    "REGISTRY_DATASET_RECORDING_IDENTITY_INVALID",
                    "critical",
                    "Each dataset row must bind exactly one recordings row.",
                    recording_id=recording_id,
                    matching_recording_row_count=len(matching_recordings),
                )
            )
        elif (
            not _dataset_requires_recording_binding(row)
            and recording_id not in (None, "")
            and len(matching_recordings) != 1
        ):
            row_contract_issues.append(
                _issue(
                    "REGISTRY_OPTIONAL_RECORDING_IDENTITY_INVALID",
                    "critical",
                    "An optional derived/training recording_id was supplied but does not bind exactly one recordings row.",
                    recording_id=recording_id,
                    matching_recording_row_count=len(matching_recordings),
                )
            )
        row_contract_issues.extend(registry_row_integrity_issues.get(key, ()))
        duplicate_paths = duplicate_recording_paths_by_id.get(
            str(recording_id or ""), []
        )
        if duplicate_paths:
            row_contract_issues.append(
                _issue(
                    "REGISTRY_DUPLICATE_RECORDING_PATH",
                    "critical",
                    "The dataset's recordings row shares its recording_path with another recordings row.",
                    recording_id=recording_id,
                    recording_paths=sorted(duplicate_paths),
                )
            )
        if key in alias_keys:
            row_contract_issues.append(
                _issue(
                    "REGISTRY_ZARR_ARCHIVE_ALIAS",
                    "critical",
                    "Multiple dataset rows resolve to the same archive identity.",
                    alias_evidence=alias_keys[key],
                )
            )
        _apply_registry_contract_issues(
            dataset_records,
            global_issues=registry_integrity_issues,
            row_issues=row_contract_issues,
        )
        _stamp_record_bundle(dataset_records)
        if checkpoint_dir is not None and _record_bundle_is_complete(key, dataset_records):
            _write_dataset_checkpoint(checkpoint_dir, key, dataset_records)
        records.extend(dataset_records)
    return records


def summarize(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    datasets = [record for record in records if record.get("record_type") == "coordinate_dataset"]
    surfaces = [record for record in records if record.get("record_type") == "coordinate_surface"]
    return {
        "audit_schema_id": AUDIT_SCHEMA_ID,
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "record_type": "coordinate_inventory_summary",
        "dataset_row_count": len(datasets),
        "distinct_recording_count": len(
            {str(record.get("recording_id")) for record in datasets if record.get("recording_id") not in (None, "")}
        ),
        "surface_count": len(surfaces),
        "dataset_status_counts": dict(sorted(Counter(str(record.get("status")) for record in datasets).items())),
        "surface_status_counts": dict(sorted(Counter(str(record.get("status")) for record in surfaces).items())),
        "surface_type_counts": dict(sorted(Counter(str(record.get("surface_type")) for record in surfaces).items())),
        "issue_code_counts": dict(
            sorted(
                Counter(
                    str(code)
                    for record in records
                    for code in (record.get("issue_codes") or [])
                ).items()
            )
        ),
    }


def build_registry_snapshot(
    registry_path: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    scan_scope: Mapping[str, Any] | None = None,
    registry_rows: tuple[
        Sequence[Mapping[str, Any]],
        Sequence[Mapping[str, Any]],
    ]
    | None = None,
) -> dict[str, Any]:
    """Build a full recordings/datasets snapshot plus explicit scan selection."""

    if registry_rows is None:
        recording_rows, dataset_rows = read_registry_snapshot_rows(registry_path)
    else:
        raw_recordings, raw_datasets = registry_rows
        recording_rows = [dict(row) for row in raw_recordings]
        dataset_rows = [dict(row) for row in raw_datasets]
    scanned_datasets = [
        record for record in records if record.get("record_type") == "coordinate_dataset"
    ]
    selected_dataset_keys = [
        str(record.get("dataset_key")) for record in scanned_datasets
    ]
    selected_dataset_key_set = set(selected_dataset_keys)
    all_dataset_keys = [
        _dataset_key(row, ordinal) for ordinal, row in enumerate(dataset_rows)
    ]
    dataset_recording_ids = {
        str(row.get("recording_id"))
        for row in dataset_rows
        if row.get("recording_id") not in (None, "")
    }
    recording_ids = [
        str(row.get("recording_id"))
        for row in recording_rows
        if row.get("recording_id") not in (None, "")
    ]
    recording_row_keys = [
        _recording_key(row, ordinal) for ordinal, row in enumerate(recording_rows)
    ]
    recording_rows_without_datasets = [
        row
        for row in recording_rows
        if str(row.get("recording_id") or "") not in dataset_recording_ids
    ]
    dataset_rows_without_recording_id = [
        row for row in dataset_rows if row.get("recording_id") in (None, "")
    ]
    dataset_keys_without_recording_id = [
        _dataset_key(row, ordinal)
        for ordinal, row in enumerate(dataset_rows)
        if row.get("recording_id") in (None, "")
    ]
    known_recording_ids = set(recording_ids)
    dataset_rows_with_unknown_recording_id = [
        row
        for row in dataset_rows
        if row.get("recording_id") not in (None, "")
        and str(row.get("recording_id")) not in known_recording_ids
    ]
    recording_path_groups: dict[str, list[dict[str, Any]]] = {}
    for row in recording_rows:
        recording_path = row.get("recording_path")
        if recording_path not in (None, ""):
            recording_path_groups.setdefault(str(recording_path), []).append(row)
    duplicate_recording_paths = [
        {
            "recording_path": path,
            "recording_row_count": len(matches),
            "recording_ids": sorted(
                str(match.get("recording_id"))
                for match in matches
                if match.get("recording_id") not in (None, "")
            ),
        }
        for path, matches in sorted(recording_path_groups.items())
        if len(matches) > 1
    ]
    inferred_scope = next(
        (
            record.get("scan_scope")
            for record in scanned_datasets
            if isinstance(record.get("scan_scope"), Mapping)
        ),
        {},
    )
    normalized_scope = _json_safe(scan_scope if scan_scope is not None else inferred_scope)
    scope_mapping = normalized_scope if isinstance(normalized_scope, Mapping) else {}
    recordings_by_id: dict[str, list[Mapping[str, Any]]] = {}
    for recording in recording_rows:
        recordings_by_id.setdefault(str(recording.get("recording_id") or ""), []).append(
            recording
        )
    expected_selected_dataset_keys = [
        _dataset_key(row, ordinal)
        for ordinal, row in enumerate(dataset_rows)
        if _dataset_matches_recording_filters(
            row,
            recording_ids=_normalized_filters(scope_mapping.get("recording_ids")),
            recording_path_contains=_normalized_filters(
                scope_mapping.get("recording_path_contains")
            ),
            recordings_by_id=recordings_by_id,
        )
    ]
    expected_selected_key_set = set(expected_selected_dataset_keys)
    missing_expected_dataset_keys = sorted(
        expected_selected_key_set - selected_dataset_key_set
    )
    unexpected_selected_dataset_keys = sorted(
        selected_dataset_key_set - expected_selected_key_set
    )
    current_fingerprints = {
        _dataset_key(row, ordinal): _fingerprint(_registry_projection(row))
        for ordinal, row in enumerate(dataset_rows)
    }
    changed_selected_dataset_keys = sorted(
        str(record.get("dataset_key"))
        for record in scanned_datasets
        if current_fingerprints.get(str(record.get("dataset_key")))
        != record.get("registry_fingerprint")
    )
    current_snapshot_fingerprint = _registry_snapshot_fingerprint(
        recording_rows,
        dataset_rows,
    )
    initial_snapshot_fingerprints = sorted(
        {
            str(record.get("registry_snapshot_fingerprint"))
            for record in scanned_datasets
            if record.get("registry_snapshot_fingerprint") not in (None, "")
        }
    )
    initial_snapshot_fingerprint = (
        initial_snapshot_fingerprints[0]
        if len(initial_snapshot_fingerprints) == 1
        else None
    )
    registry_snapshot_mixed = len(initial_snapshot_fingerprints) > 1
    registry_changed_after_scan = (
        registry_snapshot_mixed
        or initial_snapshot_fingerprint is None
        or initial_snapshot_fingerprint != current_snapshot_fingerprint
        or bool(changed_selected_dataset_keys)
    )
    zarr_path_counts = Counter(
        str(row.get("zarr_path"))
        for row in dataset_rows
        if row.get("zarr_path") not in (None, "")
    )
    duplicate_zarr_paths = [
        {"zarr_path": path, "dataset_row_count": count}
        for path, count in sorted(zarr_path_counts.items())
        if count > 1
    ]
    duplicate_dataset_keys = _duplicate_values(all_dataset_keys)
    duplicate_recording_keys = _duplicate_values(recording_row_keys)
    return {
        "schema_id": "palette.coordinate_contract_audit.registry_snapshot",
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "source_registry_path": str(registry_path.expanduser().resolve()),
        "scan_scope": normalized_scope,
        "scan_scope_sha256": _fingerprint(normalized_scope),
        "recording_row_count": len(recording_rows),
        "recording_rows_sha256": _fingerprint(recording_rows),
        "recording_row_ids": recording_ids,
        "recording_row_keys": recording_row_keys,
        "duplicate_recording_key_count": len(duplicate_recording_keys),
        "duplicate_recording_keys": duplicate_recording_keys,
        "recording_snapshot_complete": (
            len(recording_row_keys) == len(recording_rows)
            and not duplicate_recording_keys
        ),
        "recording_rows": recording_rows,
        "dataset_row_count": len(dataset_rows),
        "dataset_rows_sha256": _fingerprint(dataset_rows),
        "dataset_row_keys": all_dataset_keys,
        "duplicate_dataset_key_count": len(duplicate_dataset_keys),
        "duplicate_dataset_keys": duplicate_dataset_keys,
        "duplicate_zarr_path_count": len(duplicate_zarr_paths),
        "duplicate_zarr_paths": duplicate_zarr_paths,
        "dataset_rows": dataset_rows,
        "selected_dataset_row_count": len(selected_dataset_keys),
        "selected_dataset_keys": selected_dataset_keys,
        "expected_selected_dataset_row_count": len(expected_selected_dataset_keys),
        "expected_selected_dataset_keys": expected_selected_dataset_keys,
        "missing_expected_dataset_keys": missing_expected_dataset_keys,
        "unexpected_selected_dataset_keys": unexpected_selected_dataset_keys,
        "unselected_dataset_keys": [
            key for key in all_dataset_keys if key not in selected_dataset_key_set
        ],
        "selected_recording_ids": sorted(
            {
                str(record.get("recording_id"))
                for record in scanned_datasets
                if record.get("recording_id") not in (None, "")
            }
        ),
        "recordings_without_dataset_count": len(recording_rows_without_datasets),
        "recording_ids_without_dataset": [
            str(row.get("recording_id"))
            for row in recording_rows_without_datasets
            if row.get("recording_id") not in (None, "")
        ],
        "recording_row_keys_without_dataset": [
            _recording_key(row, ordinal)
            for ordinal, row in enumerate(recording_rows)
            if str(row.get("recording_id") or "") not in dataset_recording_ids
        ],
        "dataset_rows_without_recording_id_count": len(
            dataset_rows_without_recording_id
        ),
        "dataset_ids_without_recording_id": [
            str(row.get("dataset_id"))
            for row in dataset_rows_without_recording_id
            if row.get("dataset_id") not in (None, "")
        ],
        "dataset_keys_without_recording_id": dataset_keys_without_recording_id,
        "dataset_rows_with_unknown_recording_id_count": len(
            dataset_rows_with_unknown_recording_id
        ),
        "dataset_ids_with_unknown_recording_id": [
            str(row.get("dataset_id"))
            for row in dataset_rows_with_unknown_recording_id
            if row.get("dataset_id") not in (None, "")
        ],
        "duplicate_recording_path_count": len(duplicate_recording_paths),
        "duplicate_recording_paths": duplicate_recording_paths,
        "initial_registry_snapshot_fingerprint": initial_snapshot_fingerprint,
        "current_registry_snapshot_fingerprint": current_snapshot_fingerprint,
        "registry_snapshot_mixed": registry_snapshot_mixed,
        "registry_changed_after_scan": registry_changed_after_scan,
        "changed_selected_dataset_keys": changed_selected_dataset_keys,
    }


def _records_by_dataset(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        key = record.get("dataset_key")
        if key not in (None, ""):
            grouped.setdefault(str(key), []).append(record)
    return grouped


def build_targets(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Normalize one scan target row for every registry dataset row."""

    grouped = _records_by_dataset(records)
    targets: list[dict[str, Any]] = []
    for record in records:
        if record.get("record_type") != "coordinate_dataset":
            continue
        key = str(record.get("dataset_key"))
        related = grouped.get(key, [])
        surfaces = [item for item in related if item.get("record_type") == "coordinate_surface"]
        issue_codes = sorted(
            {
                str(code)
                for item in related
                for code in (item.get("issue_codes") or [])
            }
        )
        targets.append(
            {
                "schema_id": "palette.coordinate_contract_audit.target",
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "target_id": key,
                "dataset_key": key,
                "dataset_id": record.get("dataset_id"),
                "recording_id": record.get("recording_id"),
                "zarr_path": record.get("zarr_path"),
                "registry_status": record.get("registry_status"),
                "zarr_origin": record.get("zarr_origin"),
                "zarr_use": record.get("zarr_use"),
                "artifact_kind": record.get("artifact_kind"),
                "scan_status": record.get("status"),
                "scan_complete": record.get("scan_complete") is True,
                "surface_count": len(surfaces),
                "surface_type_counts": dict(
                    sorted(Counter(str(item.get("surface_type")) for item in surfaces).items())
                ),
                "surface_status_counts": dict(
                    sorted(Counter(str(item.get("status")) for item in surfaces).items())
                ),
                "issue_codes": issue_codes,
                "registry_fingerprint": record.get("registry_fingerprint"),
                "root_metadata_fingerprint": record.get("root_metadata_fingerprint"),
                "metadata_inventory_fingerprint": record.get("metadata_inventory_fingerprint"),
            }
        )
    return targets


def build_issues(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Flatten dataset and surface issues into stable, addressable records."""

    issues: list[dict[str, Any]] = []
    for record in records:
        raw_issues = record.get("issues")
        if not isinstance(raw_issues, list):
            continue
        for issue_ordinal, raw_issue in enumerate(raw_issues):
            if not isinstance(raw_issue, Mapping):
                continue
            identity = {
                "dataset_key": record.get("dataset_key"),
                "record_type": record.get("record_type"),
                "surface_path": record.get("surface_path"),
                "surface_type": record.get("surface_type"),
                "issue_ordinal": issue_ordinal,
                "issue": _json_safe(raw_issue),
            }
            issues.append(
                {
                    "schema_id": "palette.coordinate_contract_audit.issue",
                    "schema_version": ARTIFACT_SCHEMA_VERSION,
                    "issue_id": _fingerprint(identity),
                    "dataset_key": record.get("dataset_key"),
                    "dataset_id": record.get("dataset_id"),
                    "recording_id": record.get("recording_id"),
                    "zarr_path": record.get("zarr_path"),
                    "record_type": record.get("record_type"),
                    "surface_type": record.get("surface_type"),
                    "surface_path": record.get("surface_path"),
                    "scan_status": record.get("status"),
                    "issue_code": raw_issue.get("code"),
                    "severity": raw_issue.get("severity"),
                    "message": raw_issue.get("message"),
                    "evidence": _json_safe(raw_issue.get("evidence")),
                }
            )
    return issues


def build_issue_summary(issues: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate normalized issues while retaining exact affected identities."""

    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for issue in issues:
        key = (str(issue.get("issue_code") or ""), str(issue.get("severity") or ""))
        grouped.setdefault(key, []).append(issue)
    rows: list[dict[str, Any]] = []
    for (issue_code, severity), matches in sorted(grouped.items()):
        rows.append(
            {
                "issue_code": issue_code,
                "severity": severity,
                "occurrence_count": len(matches),
                "affected_dataset_count": len(
                    {str(item.get("dataset_key")) for item in matches}
                ),
                "affected_recording_count": len(
                    {
                        str(item.get("recording_id"))
                        for item in matches
                        if item.get("recording_id") not in (None, "")
                    }
                ),
                "affected_archive_count": len(
                    {
                        str(item.get("zarr_path"))
                        for item in matches
                        if item.get("zarr_path") not in (None, "")
                    }
                ),
                "affected_dataset_keys": sorted(
                    {str(item.get("dataset_key")) for item in matches}
                ),
                "affected_recording_ids": sorted(
                    {
                        str(item.get("recording_id"))
                        for item in matches
                        if item.get("recording_id") not in (None, "")
                    }
                ),
                "affected_zarr_paths": sorted(
                    {
                        str(item.get("zarr_path"))
                        for item in matches
                        if item.get("zarr_path") not in (None, "")
                    }
                ),
            }
        )
    return rows


_MIGRATION_CLASS_BY_STATUS = {
    "compatible": "no_change",
    "compatible_via_explicit_legacy_rule": "safe_metadata_only_backfill",
    "metadata_backfill_candidate": "metadata_backfill_requires_review",
    "numerical_validation_required": "numerical_validation_required",
    "recompute_required": "recomputation_required",
    "ambiguous_fail_closed": "ambiguous_fail_closed",
    "missing_or_unreadable": "missing_or_unreadable_fail_closed",
    "not_applicable_unscanned": "not_applicable_unscanned",
}


def _migration_class(record: Mapping[str, Any]) -> str:
    status = str(record.get("status"))
    migration_class = _MIGRATION_CLASS_BY_STATUS.get(
        status, "ambiguous_fail_closed"
    )
    if migration_class != "safe_metadata_only_backfill":
        return migration_class
    # A legacy surface is only a safe metadata-only backfill when every issue
    # is exactly the expected descriptor/label compatibility debt.  Missing
    # overlay, lineage, dimensions, axes, or row identity still needs review.
    safe_legacy_issue_codes = {
        "ARRAY_COORDINATE_DESCRIPTOR_MISSING",
        "LEGACY_SPACE_LABEL_REQUIRES_COMPATIBILITY_RULE",
    }
    issue_codes = {str(code) for code in (record.get("issue_codes") or [])}
    if not issue_codes <= safe_legacy_issue_codes:
        return "metadata_backfill_requires_review"
    proof = _as_mapping(record.get("legacy_compatibility_proof"))
    if (
        set(proof) != _LEGACY_COMPATIBILITY_EVIDENCE_FIELDS
        or proof.get("values_changed") is not False
        or not isinstance(proof.get("values_sha256"), str)
        or _SHA256_HEX_RE.fullmatch(str(proof.get("values_sha256"))) is None
        or not isinstance(proof.get("validation_tool_commit"), str)
        or not str(proof.get("validation_tool_commit")).strip()
    ):
        return "metadata_backfill_requires_review"
    return migration_class


def _repository_commit() -> str | None:
    repository = Path(__file__).resolve()
    for parent in repository.parents:
        git_path = parent / ".git"
        if not git_path.exists():
            continue
        try:
            if git_path.is_file():
                pointer = git_path.read_text(encoding="utf-8").strip()
                if not pointer.startswith("gitdir: "):
                    return None
                git_path = (parent / pointer.removeprefix("gitdir: ")).resolve()
            head = (git_path / "HEAD").read_text(encoding="utf-8").strip()
            if head.startswith("ref: "):
                ref_name = head.removeprefix("ref: ")
                common_path = git_path
                common_pointer = git_path / "commondir"
                if common_pointer.is_file():
                    common_path = (
                        git_path
                        / common_pointer.read_text(encoding="utf-8").strip()
                    ).resolve()
                for ref_root in dict.fromkeys((git_path, common_path)):
                    loose_ref = ref_root / ref_name
                    if loose_ref.is_file():
                        head = loose_ref.read_text(encoding="utf-8").strip()
                        break
                else:
                    head = ""
                    for ref_root in dict.fromkeys((git_path, common_path)):
                        packed_refs = ref_root / "packed-refs"
                        if not packed_refs.is_file():
                            continue
                        for line in packed_refs.read_text(
                            encoding="utf-8"
                        ).splitlines():
                            if not line or line.startswith(("#", "^")):
                                continue
                            commit, separator, packed_name = line.partition(" ")
                            if separator and packed_name == ref_name:
                                head = commit
                                break
                        if head:
                            break
            return (
                head.lower()
                if re.fullmatch(r"[0-9a-fA-F]{40,64}", head)
                else None
            )
        except OSError:
            return None
    return None


def _migration_evidence(record: Mapping[str, Any]) -> dict[str, Any]:
    descriptor = _as_mapping(record.get("coordinate_descriptor"))
    refs: list[Mapping[str, Any]] = []
    for field in ("lineage_refs", "transform_refs"):
        raw = descriptor.get(field)
        if isinstance(raw, (list, tuple)):
            refs.extend(item for item in raw if isinstance(item, Mapping))
    overlay_refs = _as_mapping(descriptor.get("source_camera_overlay")).get(
        "transform_refs"
    )
    if isinstance(overlay_refs, (list, tuple)):
        refs.extend(item for item in overlay_refs if isinstance(item, Mapping))
    authority = _as_mapping(descriptor.get("reference_extent")).get("authority")
    authority_record = _as_mapping(authority)
    if authority_record:
        refs.append(authority_record)
    frame_record = _as_mapping(descriptor.get("frame_record"))
    if frame_record:
        refs.append(frame_record)
    row_identity = _as_mapping(descriptor.get("row_identity"))
    if isinstance(row_identity.get("record_ref"), str):
        refs.append(row_identity)
    paths = sorted(
        {
            *(
                str(item.get("ref") or item.get("record_ref"))
                for item in refs
                if isinstance(item.get("ref") or item.get("record_ref"), str)
            ),
            *(
                ref
                for _name, ref in _row_identity_refs(row_identity)
                if isinstance(ref, str)
            ),
            *([str(authority)] if isinstance(authority, str) else []),
        }
    )
    hashes: dict[str, str] = {}
    for item in refs:
        ref = item.get("ref") or item.get("record_ref")
        digest = item.get("sha256") or item.get("record_sha256")
        if isinstance(ref, str) and isinstance(digest, str):
            hashes[ref] = digest
    proof = _as_mapping(record.get("legacy_compatibility_proof"))
    if isinstance(proof.get("surface_path"), str) and isinstance(
        proof.get("values_sha256"), str
    ):
        hashes[f"values:{proof['surface_path']}"] = str(proof["values_sha256"])
    return {
        "evidence_paths": paths,
        "evidence_sha256": dict(sorted(hashes.items())),
        "validation_tool_commit": (
            proof.get("validation_tool_commit") or _repository_commit()
        ),
    }


def _hierarchy_migration_target(
    *,
    target_kind: str,
    identity: Mapping[str, Any],
    scan_status: str,
    migration_class: str,
    issue_codes: Sequence[str] = (),
    dataset_key: Any = None,
    dataset_id: Any = None,
    recording_id: Any = None,
    zarr_path: Any = None,
    run_context: Mapping[str, Any] | None = None,
    metadata_inventory_fingerprint: Any = None,
) -> dict[str, Any]:
    previous_state = {
        "scan_status": scan_status,
        "target_identity": _json_safe(identity),
        "metadata_inventory_fingerprint": metadata_inventory_fingerprint,
    }
    result_state = {
        "migration_class": migration_class,
        "apply_status": "not_applied_read_only_plan",
        "coordinate_descriptor": "not_applicable_hierarchy_target",
    }
    return {
        "schema_id": "palette.coordinate_contract_audit.migration_target",
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "migration_target_id": _fingerprint(
            {"target_kind": target_kind, **dict(_json_safe(identity))}
        ),
        "dataset_key": dataset_key,
        "dataset_id": dataset_id,
        "recording_id": recording_id,
        "zarr_path": zarr_path,
        "target_kind": target_kind,
        "surface_type": None,
        "surface_path": None,
        "run_context": _json_safe(run_context),
        "scan_status": scan_status,
        "migration_class": migration_class,
        "safe_metadata_only_backfill": False,
        "requires_numerical_validation": False,
        "requires_recomputation": False,
        "must_fail_closed": migration_class
        in {"ambiguous_fail_closed", "missing_or_unreadable_fail_closed"},
        "automatic_apply_allowed": False,
        "issue_codes": sorted({str(code) for code in issue_codes}),
        "descriptor_source": None,
        "metadata_inventory_fingerprint": metadata_inventory_fingerprint,
        "dependent_surface_paths": [],
        "evidence_paths": [],
        "evidence_sha256": {},
        "validation_tool_commit": _repository_commit(),
        "validation_result": "pending_or_failed",
        "values_changed": None,
        "previous_state": previous_state,
        "previous_state_sha256": _fingerprint(previous_state),
        "result_state": result_state,
        "result_state_sha256": _fingerprint(result_state),
    }


def build_migration_manifest(
    records: Sequence[Mapping[str, Any]],
    *,
    registry_snapshot: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Map every important surface to an explicit dry-run migration class."""

    grouped = _records_by_dataset(records)
    manifest: list[dict[str, Any]] = []
    snapshot = registry_snapshot or {}
    if snapshot:
        registry_issue_codes: list[str] = []
        if snapshot.get("registry_changed_after_scan"):
            registry_issue_codes.append("REGISTRY_CHANGED_AFTER_SCAN")
        if snapshot.get("duplicate_dataset_keys"):
            registry_issue_codes.append("REGISTRY_DUPLICATE_DATASET_KEY")
        if snapshot.get("duplicate_recording_keys"):
            registry_issue_codes.append("REGISTRY_DUPLICATE_RECORDING_KEY")
        if snapshot.get("duplicate_zarr_paths"):
            registry_issue_codes.append("REGISTRY_DUPLICATE_ZARR_PATH")
        if snapshot.get("duplicate_recording_paths"):
            registry_issue_codes.append("REGISTRY_DUPLICATE_RECORDING_PATH")
        registry_fail_closed = any(
            code != "REGISTRY_DUPLICATE_RECORDING_PATH"
            for code in registry_issue_codes
        )
        manifest.append(
            _hierarchy_migration_target(
                target_kind="registry",
                identity={
                    "source_registry_path": snapshot.get("source_registry_path"),
                    "current_registry_snapshot_fingerprint": snapshot.get(
                        "current_registry_snapshot_fingerprint"
                    ),
                },
                scan_status=(
                    "ambiguous_fail_closed" if registry_fail_closed else "compatible"
                ),
                migration_class=(
                    "ambiguous_fail_closed"
                    if registry_fail_closed
                    else "registry_reconciliation_required"
                    if registry_issue_codes
                    else "no_change"
                ),
                issue_codes=registry_issue_codes,
            )
        )
        duplicate_paths_by_recording: dict[str, list[str]] = {}
        for duplicate in snapshot.get("duplicate_recording_paths") or []:
            if not isinstance(duplicate, Mapping):
                continue
            path = str(duplicate.get("recording_path") or "")
            for recording_id in duplicate.get("recording_ids") or []:
                duplicate_paths_by_recording.setdefault(
                    str(recording_id), []
                ).append(path)
        for ordinal, recording in enumerate(snapshot.get("recording_rows") or []):
            if not isinstance(recording, Mapping):
                continue
            recording_id = recording.get("recording_id")
            duplicate_paths = duplicate_paths_by_recording.get(
                str(recording_id), []
            )
            manifest.append(
                _hierarchy_migration_target(
                    target_kind="recording",
                    identity={
                        "recording_key": _recording_key(recording, ordinal),
                        "recording_path": recording.get("recording_path"),
                    },
                    recording_id=recording_id,
                    scan_status=(
                        "metadata_backfill_candidate"
                        if duplicate_paths
                        else "compatible"
                    ),
                    migration_class=(
                        "registry_reconciliation_required"
                        if duplicate_paths
                        else "no_change"
                    ),
                    issue_codes=(
                        ["REGISTRY_DUPLICATE_RECORDING_PATH"]
                        if duplicate_paths
                        else []
                    ),
                )
            )
    for dataset_record in records:
        if dataset_record.get("record_type") != "coordinate_dataset":
            continue
        key = str(dataset_record.get("dataset_key"))
        surfaces = [
            record
            for record in grouped.get(key, [])
            if record.get("record_type") == "coordinate_surface"
        ]
        targets = [dataset_record, *surfaces]
        for record in targets:
            status = str(record.get("status"))
            migration_class = _migration_class(record)
            migration_evidence = _migration_evidence(record)
            previous_state = {
                "scan_status": status,
                "coordinate_descriptor": _json_safe(
                    record.get("coordinate_descriptor")
                ),
                "metadata_inventory_fingerprint": record.get(
                    "metadata_inventory_fingerprint"
                ),
            }
            result_state = {
                "migration_class": migration_class,
                "apply_status": "not_applied_read_only_plan",
                "coordinate_descriptor": (
                    "preserve"
                    if migration_class == "no_change"
                    else "backfill_only_after_evidence_gate"
                    if migration_class == "safe_metadata_only_backfill"
                    else "undefined_until_validation_or_recomputation"
                ),
            }
            target_kind = (
                "coordinate_surface"
                if record.get("record_type") == "coordinate_surface"
                else "archive"
            )
            identity = {
                "dataset_key": key,
                "target_kind": target_kind,
                "surface_path": record.get("surface_path"),
                "surface_type": record.get("surface_type"),
            }
            manifest.append(
                {
                    "schema_id": "palette.coordinate_contract_audit.migration_target",
                    "schema_version": ARTIFACT_SCHEMA_VERSION,
                    "migration_target_id": _fingerprint(identity),
                    "dataset_key": key,
                    "dataset_id": record.get("dataset_id"),
                    "recording_id": record.get("recording_id"),
                    "zarr_path": record.get("zarr_path"),
                    "target_kind": target_kind,
                    "surface_type": record.get("surface_type"),
                    "surface_path": record.get("surface_path"),
                    "run_context": _json_safe(record.get("run_context")),
                    "scan_status": status,
                    "migration_class": migration_class,
                    "safe_metadata_only_backfill": migration_class
                    == "safe_metadata_only_backfill",
                    "requires_numerical_validation": migration_class
                    == "numerical_validation_required",
                    "requires_recomputation": migration_class == "recomputation_required",
                    "must_fail_closed": migration_class
                    in {
                        "ambiguous_fail_closed",
                        "missing_or_unreadable_fail_closed",
                    },
                    # This module deliberately has no mutation/apply path.
                    "automatic_apply_allowed": False,
                    "issue_codes": sorted(
                        {
                            *(str(code) for code in (record.get("issue_codes") or [])),
                            *(
                                str(code)
                                for code in (record.get("archive_issue_codes") or [])
                            ),
                        }
                    ),
                    "archive_issue_codes": sorted(
                        str(code)
                        for code in (record.get("archive_issue_codes") or [])
                    ),
                    "descriptor_source": record.get("descriptor_source"),
                    "metadata_inventory_fingerprint": record.get(
                        "metadata_inventory_fingerprint"
                    ),
                    "dependent_surface_paths": sorted(
                        str(path) for path in (record.get("dependent_surface_paths") or [])
                    ),
                    **migration_evidence,
                    "validation_result": (
                        "metadata_and_value_invariants_verified"
                        if migration_class == "safe_metadata_only_backfill"
                        else "canonical_metadata_verified"
                        if migration_class == "no_change"
                        else "pending_or_failed"
                    ),
                    "values_changed": (
                        False
                        if migration_class
                        in {"no_change", "safe_metadata_only_backfill"}
                        else None
                    ),
                    "previous_state": previous_state,
                    "previous_state_sha256": _fingerprint(previous_state),
                    "result_state": result_state,
                    "result_state_sha256": _fingerprint(result_state),
                }
            )
        run_groups: dict[str, list[Mapping[str, Any]]] = {}
        for surface in surfaces:
            context = _as_mapping(surface.get("run_context"))
            run_path = context.get("run_path")
            if isinstance(run_path, str) and run_path:
                run_groups.setdefault(run_path, []).append(surface)
        for run_path, run_surfaces in sorted(run_groups.items()):
            source = max(
                run_surfaces,
                key=lambda item: _STATUS_PRIORITY.get(str(item.get("status")), 99),
            )
            context = _as_mapping(source.get("run_context"))
            status = str(source.get("status"))
            migration_class = _migration_class(source)
            if migration_class == "safe_metadata_only_backfill":
                migration_class = "metadata_backfill_requires_review"
            manifest.append(
                _hierarchy_migration_target(
                    target_kind="run",
                    identity={
                        "dataset_key": key,
                        "run_path": run_path,
                        "run_name": context.get("run_name"),
                    },
                    scan_status=status,
                    migration_class=migration_class,
                    issue_codes=sorted(
                        {
                            *(
                                str(code)
                                for item in run_surfaces
                                for code in (item.get("issue_codes") or [])
                            ),
                            *(
                                str(code)
                                for code in (source.get("archive_issue_codes") or [])
                            ),
                        }
                    ),
                    dataset_key=key,
                    dataset_id=dataset_record.get("dataset_id"),
                    recording_id=dataset_record.get("recording_id"),
                    zarr_path=dataset_record.get("zarr_path"),
                    run_context=context,
                    metadata_inventory_fingerprint=dataset_record.get(
                        "metadata_inventory_fingerprint"
                    ),
                )
            )
        actual_surface_paths = {
            str(record.get("surface_path"))
            for record in surfaces
            if record.get("surface_path") not in (None, "")
        }
        dependent_sources: dict[str, list[Mapping[str, Any]]] = {}
        for source in surfaces:
            for path in source.get("dependent_surface_paths") or []:
                path_text = str(path)
                if path_text not in actual_surface_paths:
                    dependent_sources.setdefault(path_text, []).append(source)
        for path, sources in sorted(dependent_sources.items()):
            source = max(
                sources,
                key=lambda item: _STATUS_PRIORITY.get(str(item.get("status")), 99),
            )
            status = str(source.get("status"))
            migration_class = _migration_class(source)
            if migration_class == "safe_metadata_only_backfill":
                migration_class = "metadata_backfill_requires_review"
            source_paths = sorted(str(item.get("surface_path")) for item in sources)
            migration_evidence = _migration_evidence(source)
            previous_state = {
                "scan_status": status,
                "dependency_source_paths": source_paths,
                "metadata_inventory_fingerprint": dataset_record.get(
                    "metadata_inventory_fingerprint"
                ),
            }
            result_state = {
                "migration_class": migration_class,
                "apply_status": "not_applied_read_only_plan",
                "coordinate_descriptor": "derived_from_revalidated_source_only",
            }
            identity = {
                "dataset_key": key,
                "target_kind": "derived_surface",
                "surface_path": path,
                "surface_type": "track_kinematics_derived",
            }
            manifest.append(
                {
                    "schema_id": "palette.coordinate_contract_audit.migration_target",
                    "schema_version": ARTIFACT_SCHEMA_VERSION,
                    "migration_target_id": _fingerprint(identity),
                    "dataset_key": key,
                    "dataset_id": dataset_record.get("dataset_id"),
                    "recording_id": dataset_record.get("recording_id"),
                    "zarr_path": dataset_record.get("zarr_path"),
                    "target_kind": "derived_surface",
                    "surface_type": "track_kinematics_derived",
                    "surface_path": path,
                    "scan_status": status,
                    "migration_class": migration_class,
                    "safe_metadata_only_backfill": False,
                    "requires_numerical_validation": migration_class
                    == "numerical_validation_required",
                    "requires_recomputation": migration_class
                    == "recomputation_required",
                    "must_fail_closed": migration_class
                    in {"ambiguous_fail_closed", "missing_or_unreadable_fail_closed"},
                    "automatic_apply_allowed": False,
                    "issue_codes": ["UPSTREAM_POSITION_RISK_PROPAGATED"],
                    "dependency_source_paths": source_paths,
                    "descriptor_source": None,
                    "metadata_inventory_fingerprint": dataset_record.get(
                        "metadata_inventory_fingerprint"
                    ),
                    "dependent_surface_paths": [],
                    **migration_evidence,
                    "validation_result": "pending_or_failed",
                    "values_changed": None,
                    "previous_state": previous_state,
                    "previous_state_sha256": _fingerprint(previous_state),
                    "result_state": result_state,
                    "result_state_sha256": _fingerprint(result_state),
                }
            )
    registry_snapshot_invalid = bool(
        snapshot.get("registry_changed_after_scan")
        or snapshot.get("registry_snapshot_mixed")
        or snapshot.get("missing_expected_dataset_keys")
        or snapshot.get("unexpected_selected_dataset_keys")
        or snapshot.get("duplicate_dataset_keys")
        or snapshot.get("duplicate_recording_keys")
    )
    if registry_snapshot_invalid:
        for target in manifest:
            target["scan_status"] = "ambiguous_fail_closed"
            target["migration_class"] = "ambiguous_fail_closed"
            target["safe_metadata_only_backfill"] = False
            target["requires_numerical_validation"] = False
            target["requires_recomputation"] = False
            target["must_fail_closed"] = True
            target["validation_result"] = "registry_snapshot_invalid"
            target["values_changed"] = None
            target["result_state"] = {
                "migration_class": "ambiguous_fail_closed",
                "apply_status": "forbidden_registry_snapshot_invalid",
                "coordinate_descriptor": "undefined",
            }
            target["result_state_sha256"] = _fingerprint(target["result_state"])
            target["issue_codes"] = sorted(
                {
                    *(str(code) for code in (target.get("issue_codes") or [])),
                    "REGISTRY_SNAPSHOT_INVALIDATES_MIGRATION",
                }
            )
    return manifest


def build_archive_summary(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return one deterministic reconciliation row per registry archive."""

    grouped = _records_by_dataset(records)
    rows: list[dict[str, Any]] = []
    for dataset in records:
        if dataset.get("record_type") != "coordinate_dataset":
            continue
        key = str(dataset.get("dataset_key"))
        related = grouped.get(key, [])
        surfaces = [item for item in related if item.get("record_type") == "coordinate_surface"]
        issue_codes = sorted(
            {
                str(code)
                for item in related
                for code in (item.get("issue_codes") or [])
            }
        )
        migration_classes = [
            _migration_class(item)
            for item in (surfaces or [dataset])
        ]
        rows.append(
            {
                "dataset_key": key,
                "dataset_id": dataset.get("dataset_id"),
                "recording_id": dataset.get("recording_id"),
                "zarr_path": dataset.get("zarr_path"),
                "registry_status": dataset.get("registry_status"),
                "zarr_origin": dataset.get("zarr_origin"),
                "zarr_use": dataset.get("zarr_use"),
                "artifact_kind": dataset.get("artifact_kind"),
                "scan_status": dataset.get("status"),
                "scan_complete": dataset.get("scan_complete") is True,
                "metadata_node_count": dataset.get("metadata_node_count"),
                "surface_count": len(surfaces),
                "surface_type_counts": dict(
                    sorted(Counter(str(item.get("surface_type")) for item in surfaces).items())
                ),
                "surface_status_counts": dict(
                    sorted(Counter(str(item.get("status")) for item in surfaces).items())
                ),
                "migration_classes": sorted(set(migration_classes)),
                "issue_codes": issue_codes,
                "registry_fingerprint": dataset.get("registry_fingerprint"),
                "metadata_inventory_fingerprint": dataset.get(
                    "metadata_inventory_fingerprint"
                ),
            }
        )
    return rows


def build_coverage(
    records: Sequence[Mapping[str, Any]],
    *,
    registry_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe registry-row and important-surface audit coverage explicitly."""

    datasets = [
        record for record in records if record.get("record_type") == "coordinate_dataset"
    ]
    surfaces = [
        record for record in records if record.get("record_type") == "coordinate_surface"
    ]
    dataset_keys = [str(record.get("dataset_key")) for record in datasets]
    complete_count = sum(record.get("scan_complete") is True for record in datasets)
    missing_count = sum(
        record.get("status") == "missing_or_unreadable" for record in datasets
    )
    snapshot = registry_snapshot or {}
    total_dataset_rows = int(snapshot.get("dataset_row_count", len(datasets)))
    total_recording_rows = int(
        snapshot.get(
            "recording_row_count",
            len(
                {
                    str(record.get("recording_id"))
                    for record in datasets
                    if record.get("recording_id") not in (None, "")
                }
            ),
        )
    )
    selected_dataset_rows = int(
        snapshot.get("selected_dataset_row_count", len(datasets))
    )
    expected_selected_keys = [
        str(key) for key in (snapshot.get("expected_selected_dataset_keys") or dataset_keys)
    ]
    expected_selected_key_set = set(expected_selected_keys)
    represented_key_set = set(dataset_keys)
    duplicate_dataset_key_count = len(dataset_keys) - len(represented_key_set)
    all_expected_represented = (
        duplicate_dataset_key_count == 0
        and represented_key_set == expected_selected_key_set
        and not bool(snapshot.get("registry_changed_after_scan", False))
    )
    recording_snapshot_complete = bool(
        snapshot.get("recording_snapshot_complete", True)
    )
    return {
        "schema_id": "palette.coordinate_contract_audit.coverage",
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "scan_scope": _json_safe(snapshot.get("scan_scope") or {}),
        "registry_recording_row_count": total_recording_rows,
        "registry_recording_ids": _json_safe(snapshot.get("recording_row_ids") or []),
        "represented_recording_row_count": (
            total_recording_rows if recording_snapshot_complete else 0
        ),
        "duplicate_recording_key_count": int(
            snapshot.get("duplicate_recording_key_count", 0)
        ),
        "duplicate_recording_keys": _json_safe(
            snapshot.get("duplicate_recording_keys") or []
        ),
        "all_registry_recording_rows_represented": recording_snapshot_complete,
        "registry_dataset_row_count": total_dataset_rows,
        "registry_dataset_keys": _json_safe(snapshot.get("dataset_row_keys") or dataset_keys),
        "selected_dataset_row_count": selected_dataset_rows,
        "selected_dataset_keys": _json_safe(
            snapshot.get("selected_dataset_keys") or dataset_keys
        ),
        "expected_selected_dataset_row_count": len(expected_selected_keys),
        "expected_selected_dataset_keys": expected_selected_keys,
        "missing_expected_dataset_keys": sorted(
            expected_selected_key_set - represented_key_set
        ),
        "unexpected_selected_dataset_keys": sorted(
            represented_key_set - expected_selected_key_set
        ),
        "unselected_dataset_row_count": total_dataset_rows - selected_dataset_rows,
        "unselected_dataset_keys": _json_safe(snapshot.get("unselected_dataset_keys") or []),
        "selected_recording_ids": _json_safe(snapshot.get("selected_recording_ids") or []),
        "represented_dataset_row_count": len(dataset_keys),
        "distinct_dataset_key_count": len(set(dataset_keys)),
        "duplicate_dataset_key_count": duplicate_dataset_key_count,
        "all_selected_dataset_rows_represented": all_expected_represented,
        "all_registry_dataset_rows_selected": selected_dataset_rows
        == total_dataset_rows,
        "all_registry_rows_represented": (
            all_expected_represented
            and recording_snapshot_complete
            and represented_key_set
            == set(str(key) for key in (snapshot.get("dataset_row_keys") or dataset_keys))
        ),
        "completed_dataset_scan_count": complete_count,
        "all_selected_dataset_scans_complete": (
            all_expected_represented and complete_count == len(expected_selected_keys)
        ),
        "all_dataset_scans_complete": (
            all_expected_represented
            and len(expected_selected_keys) == total_dataset_rows
            and complete_count == total_dataset_rows
        ),
        "missing_or_unreadable_dataset_count": missing_count,
        "inspectable_dataset_count": len(datasets) - missing_count,
        "recordings_without_dataset_count": int(
            snapshot.get("recordings_without_dataset_count", 0)
        ),
        "recording_ids_without_dataset": _json_safe(
            snapshot.get("recording_ids_without_dataset") or []
        ),
        "dataset_rows_without_recording_id_count": int(
            snapshot.get("dataset_rows_without_recording_id_count", 0)
        ),
        "dataset_ids_without_recording_id": _json_safe(
            snapshot.get("dataset_ids_without_recording_id") or []
        ),
        "dataset_rows_with_unknown_recording_id_count": int(
            snapshot.get("dataset_rows_with_unknown_recording_id_count", 0)
        ),
        "dataset_ids_with_unknown_recording_id": _json_safe(
            snapshot.get("dataset_ids_with_unknown_recording_id") or []
        ),
        "duplicate_recording_path_count": int(
            snapshot.get("duplicate_recording_path_count", 0)
        ),
        "duplicate_recording_paths": _json_safe(
            snapshot.get("duplicate_recording_paths") or []
        ),
        "registry_changed_after_scan": bool(
            snapshot.get("registry_changed_after_scan", False)
        ),
        "changed_selected_dataset_keys": _json_safe(
            snapshot.get("changed_selected_dataset_keys") or []
        ),
        "important_coordinate_surface_count": len(surfaces),
        "surface_type_counts": dict(
            sorted(Counter(str(record.get("surface_type")) for record in surfaces).items())
        ),
        "surface_status_counts": dict(
            sorted(Counter(str(record.get("status")) for record in surfaces).items())
        ),
        "unclassified_geometry_candidate_count": sum(
            record.get("surface_type") == "unclassified_geometry_candidate"
            for record in surfaces
        ),
        "inventory_records_sha256": _fingerprint([_json_safe(record) for record in records]),
    }


def write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    text = "".join(json.dumps(_json_safe(record), sort_keys=True) + "\n" for record in records)
    _atomic_write_text(path, text)


_CSV_COLUMNS = (
    "record_type",
    "dataset_key",
    "dataset_id",
    "recording_id",
    "zarr_path",
    "registry_status",
    "zarr_origin",
    "zarr_use",
    "artifact_kind",
    "surface_type",
    "surface_path",
    "node_type",
    "metadata_format",
    "shape",
    "data_type",
    "status",
    "issue_codes",
    "run_context",
    "descriptor_source",
    "descriptor_is_array_specific",
    "evidence",
    "coordinate_descriptor",
)


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (Mapping, list, tuple, set)):
        return _canonical_json(value)
    return value


def write_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=_CSV_COLUMNS, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for record in records:
        writer.writerow({column: _csv_value(record.get(column)) for column in _CSV_COLUMNS})
    _atomic_write_text(path, stream.getvalue())


def _markdown_escape(value: Any) -> str:
    return str(value if value is not None else "").replace("|", "\\|").replace("\n", " ")


def write_markdown(path: Path, records: Sequence[Mapping[str, Any]], summary: Mapping[str, Any]) -> None:
    rows = [
        record
        for record in records
        if record.get("record_type") == "coordinate_surface"
        or record.get("status") == "missing_or_unreadable"
    ]
    lines = [
        "# Coordinate contract inventory",
        "",
        f"- Registry dataset rows: {summary.get('dataset_row_count', 0)}",
        f"- Distinct recordings: {summary.get('distinct_recording_count', 0)}",
        f"- Important geometry surfaces: {summary.get('surface_count', 0)}",
        "",
        "## Status counts",
        "",
        "```json",
        json.dumps(_json_safe(summary.get("surface_status_counts", {})), indent=2, sort_keys=True),
        "```",
        "",
        "## Inventory",
        "",
        "| Dataset | Recording | Surface | Path | Status | Issues |",
        "|---|---|---|---|---|---|",
    ]
    for record in rows:
        lines.append(
            "| "
            + " | ".join(
                _markdown_escape(value)
                for value in (
                    record.get("dataset_id") or record.get("dataset_key"),
                    record.get("recording_id"),
                    record.get("surface_type") or "dataset",
                    record.get("surface_path") or record.get("zarr_path"),
                    record.get("status"),
                    ", ".join(str(code) for code in (record.get("issue_codes") or [])),
                )
            )
            + " |"
        )
    _atomic_write_text(path, "\n".join(lines) + "\n")


def write_summary(path: Path, summary: Mapping[str, Any]) -> None:
    _atomic_write_text(
        path,
        json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n",
    )


_ISSUE_SUMMARY_COLUMNS = (
    "issue_code",
    "severity",
    "occurrence_count",
    "affected_dataset_count",
    "affected_recording_count",
    "affected_archive_count",
    "affected_dataset_keys",
    "affected_recording_ids",
    "affected_zarr_paths",
)

_ARCHIVE_SUMMARY_COLUMNS = (
    "dataset_key",
    "dataset_id",
    "recording_id",
    "zarr_path",
    "registry_status",
    "zarr_use",
    "artifact_kind",
    "scan_status",
    "scan_complete",
    "metadata_node_count",
    "surface_count",
    "surface_type_counts",
    "surface_status_counts",
    "migration_classes",
    "issue_codes",
    "registry_fingerprint",
    "metadata_inventory_fingerprint",
)


def _write_normalized_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
) -> None:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=columns,
        extrasaction="ignore",
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({column: _csv_value(row.get(column)) for column in columns})
    _atomic_write_text(path, stream.getvalue())


def write_normalized_artifacts(
    output_dir: Path,
    registry_path: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    scan_scope: Mapping[str, Any] | None = None,
    external_outputs: Mapping[str, Path] | None = None,
) -> dict[str, Path]:
    """Emit the deterministic registry reconciliation artifact set."""

    # One final transaction is authoritative for both output-path validation
    # and registry_snapshot.json.  build_registry_snapshot compares its digest
    # with the initial snapshot embedded in every scan record; any drift makes
    # the migration manifest fail closed.
    recording_rows, dataset_rows = read_registry_snapshot_rows(registry_path)
    normalized_external_outputs = {
        str(role): Path(path).expanduser().resolve(strict=False)
        for role, path in sorted((external_outputs or {}).items())
    }
    _validate_write_locations(
        registry_path=registry_path,
        dataset_rows=dataset_rows,
        file_paths=(
            *tuple(output_dir / name for name in NORMALIZED_ARTIFACT_FILENAMES),
            *tuple(normalized_external_outputs.values()),
        ),
        directory_paths=(output_dir,),
    )
    paths = {name: output_dir / name for name in NORMALIZED_ARTIFACT_FILENAMES}
    registry_snapshot = build_registry_snapshot(
        registry_path,
        records,
        scan_scope=scan_scope,
        registry_rows=(recording_rows, dataset_rows),
    )
    targets = build_targets(records)
    issues = build_issues(records)
    issue_summary = build_issue_summary(issues)
    archive_summary = build_archive_summary(records)
    coverage = build_coverage(records, registry_snapshot=registry_snapshot)
    migration_manifest = build_migration_manifest(
        records,
        registry_snapshot=registry_snapshot,
    )

    write_summary(paths["registry_snapshot.json"], registry_snapshot)
    write_jsonl(paths["targets.jsonl"], targets)
    write_jsonl(paths["issues.jsonl"], issues)
    _write_normalized_csv(
        paths["issue_summary.csv"],
        issue_summary,
        _ISSUE_SUMMARY_COLUMNS,
    )
    _write_normalized_csv(
        paths["archive_summary.csv"],
        archive_summary,
        _ARCHIVE_SUMMARY_COLUMNS,
    )
    write_summary(paths["coverage.json"], coverage)
    write_jsonl(paths["migration_manifest.jsonl"], migration_manifest)
    content_names = [
        name for name in NORMALIZED_ARTIFACT_FILENAMES if name != "artifact_manifest.json"
    ]
    file_records = {
        f"artifact:{name}": {
            "path": name,
            "path_kind": "relative_to_manifest",
            "sha256": hashlib.sha256(paths[name].read_bytes()).hexdigest(),
            "size_bytes": paths[name].stat().st_size,
        }
        for name in content_names
    }
    for role, path in normalized_external_outputs.items():
        try:
            content = path.read_bytes()
        except OSError as exc:
            raise ValueError(
                f"requested external audit output is unreadable: {role}: {path}"
            ) from exc
        file_records[f"external:{role}"] = {
            "path": str(path),
            "path_kind": "absolute",
            "sha256": hashlib.sha256(content).hexdigest(),
            "size_bytes": len(content),
        }
    generation_complete = bool(
        coverage.get("all_registry_rows_represented")
        and not registry_snapshot.get("registry_changed_after_scan")
        and not any(
            (_as_mapping(registry_snapshot.get("scan_scope"))).get(name)
            for name in ("recording_ids", "recording_path_contains", "run_families")
        )
        and all(
            record.get("generation_complete") is True
            or record.get("status") == "missing_or_unreadable"
            for record in records
            if record.get("record_type") == "coordinate_dataset"
        )
    )
    generation_payload = {
        "schema_id": "palette.coordinate_contract_audit.artifact_generation",
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "complete": generation_complete,
        "integrity_manifest_complete": True,
        "declared_output_files": sorted(
            {
                *NORMALIZED_ARTIFACT_FILENAMES,
                *(str(path) for path in normalized_external_outputs.values()),
            }
        ),
        "manifest_file": "artifact_manifest.json",
        "manifest_self_digest_policy": (
            "canonical_json_payload_excluding_generation_sha256_v1"
        ),
        "files": file_records,
        "inventory_records_sha256": _fingerprint(
            [_json_safe(record) for record in records]
        ),
        "registry_snapshot_sha256": hashlib.sha256(
            paths["registry_snapshot.json"].read_bytes()
        ).hexdigest(),
    }
    generation_payload["generation_sha256"] = _fingerprint(generation_payload)
    write_summary(paths["artifact_manifest.json"], generation_payload)
    return paths


def verify_normalized_artifact_generation(output_dir: Path) -> dict[str, Any]:
    """Load and verify the final marker for one normalized artifact generation."""

    manifest_path = output_dir / "artifact_manifest.json"
    payload, error = _read_json_object(manifest_path)
    if error or payload is None:
        raise ValueError(f"artifact generation manifest is unreadable: {error}")
    if payload.get("schema_id") != "palette.coordinate_contract_audit.artifact_generation":
        raise ValueError("artifact generation manifest has an unsupported schema_id")
    if payload.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("artifact generation manifest has an unsupported schema_version")
    if payload.get("integrity_manifest_complete") is not True:
        raise ValueError("artifact integrity manifest is not marked complete")
    declared_outputs = payload.get("declared_output_files")
    if not isinstance(declared_outputs, list) or len(declared_outputs) != len(
        set(str(path) for path in declared_outputs)
    ):
        raise ValueError("artifact generation declared output file set is invalid")
    if payload.get("manifest_file") != "artifact_manifest.json":
        raise ValueError("artifact generation manifest filename is invalid")
    if payload.get("manifest_self_digest_policy") != (
        "canonical_json_payload_excluding_generation_sha256_v1"
    ):
        raise ValueError("artifact generation manifest self-digest policy is invalid")
    stored_generation_digest = payload.get("generation_sha256")
    unsigned = dict(payload)
    unsigned.pop("generation_sha256", None)
    if stored_generation_digest != _fingerprint(unsigned):
        raise ValueError("artifact generation manifest digest mismatch")
    file_records = payload.get("files")
    if not isinstance(file_records, Mapping):
        raise ValueError("artifact generation manifest has no file records")
    expected_normalized_keys = {
        f"artifact:{name}"
        for name in NORMALIZED_ARTIFACT_FILENAMES
        if name != "artifact_manifest.json"
    }
    if not expected_normalized_keys <= set(file_records):
        raise ValueError("artifact generation normalized file set is incomplete")
    declared_record_paths: set[str] = set()
    for name in sorted(file_records):
        record = file_records[name]
        if not isinstance(record, Mapping):
            raise ValueError(f"artifact generation file record is invalid: {name}")
        path_kind = record.get("path_kind")
        raw_path = record.get("path")
        if not isinstance(raw_path, str):
            raise ValueError(f"artifact generation file path is invalid: {name}")
        if path_kind == "relative_to_manifest":
            path = output_dir / raw_path
            declared_record_paths.add(raw_path)
        elif path_kind == "absolute":
            path = Path(raw_path)
            declared_record_paths.add(str(path.resolve(strict=False)))
        else:
            raise ValueError(f"artifact generation path kind is invalid: {name}")
        try:
            content = path.read_bytes()
        except OSError as exc:
            raise ValueError(f"artifact generation file is unreadable: {name}") from exc
        if record.get("size_bytes") != len(content):
            raise ValueError(f"artifact generation file size mismatch: {name}")
        if record.get("sha256") != hashlib.sha256(content).hexdigest():
            raise ValueError(f"artifact generation file digest mismatch: {name}")
    expected_declared_paths = {
        *declared_record_paths,
        "artifact_manifest.json",
    }
    if set(str(path) for path in declared_outputs) != expected_declared_paths:
        raise ValueError("artifact generation declared output paths do not match file records")
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only registry-wide audit of persisted coordinate contracts."
    )
    parser.add_argument("--registry", type=Path, required=True, help="Palette SQLite registry (opened mode=ro).")
    parser.add_argument("--output-jsonl", type=Path, help="Deterministic detailed inventory JSONL.")
    parser.add_argument("--output-csv", type=Path, help="Deterministic flattened inventory CSV.")
    parser.add_argument("--output-markdown", type=Path, help="Deterministic human-readable report.")
    parser.add_argument("--summary-json", type=Path, help="Deterministic summary JSON.")
    parser.add_argument(
        "--resume-jsonl",
        type=Path,
        help="Reuse complete rows whose registry fingerprint matches this prior JSONL (for immutable archives).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help=(
            "Atomically checkpoint each completed dataset and resume matching "
            "checkpoints on later invocations."
        ),
    )
    parser.add_argument(
        "--artifact-dir",
        "--artifacts-dir",
        dest="artifact_dir",
        type=Path,
        help=(
            "Emit registry_snapshot.json, targets.jsonl, issues.jsonl, "
            "issue_summary.csv, archive_summary.csv, coverage.json, and "
            "migration_manifest.jsonl."
        ),
    )
    parser.add_argument(
        "--recording-id",
        action="append",
        default=[],
        help="Select an exact registry recording_id; may be repeated.",
    )
    parser.add_argument(
        "--recording-path-contains",
        action="append",
        default=[],
        help="Select recording_path values containing this literal text; may be repeated.",
    )
    parser.add_argument(
        "--run-family",
        action="append",
        default=[],
        help=(
            "Limit reported coordinate surfaces to this exact path/segment "
            "(for example track_kinematics_runs); may be repeated."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _, dataset_rows = read_registry_snapshot_rows(args.registry)
    output_files = [
        path
        for path in (
            args.output_jsonl,
            args.output_csv,
            args.output_markdown,
            args.summary_json,
        )
        if path is not None
    ]
    if args.artifact_dir is not None:
        output_files.extend(
            args.artifact_dir / name for name in NORMALIZED_ARTIFACT_FILENAMES
        )
    output_dirs = [
        path for path in (args.checkpoint_dir, args.artifact_dir) if path is not None
    ]
    _validate_write_locations(
        registry_path=args.registry,
        dataset_rows=dataset_rows,
        file_paths=output_files,
        directory_paths=output_dirs,
    )
    records = audit_registry(
        args.registry,
        resume_jsonl=args.resume_jsonl,
        checkpoint_dir=args.checkpoint_dir,
        recording_ids=args.recording_id,
        recording_path_contains=args.recording_path_contains,
        run_families=args.run_family,
    )
    summary = summarize(records)
    if args.output_jsonl:
        write_jsonl(args.output_jsonl, records)
    else:
        for record in records:
            print(json.dumps(_json_safe(record), sort_keys=True))
    if args.output_csv:
        write_csv(args.output_csv, records)
    if args.output_markdown:
        write_markdown(args.output_markdown, records, summary)
    if args.summary_json:
        write_summary(args.summary_json, summary)
    if args.artifact_dir:
        external_outputs = {
            role: path
            for role, path in {
                "inventory_jsonl": args.output_jsonl,
                "inventory_csv": args.output_csv,
                "report_markdown": args.output_markdown,
                "summary_json": args.summary_json,
            }.items()
            if path is not None
        }
        write_normalized_artifacts(
            args.artifact_dir,
            args.registry,
            records,
            scan_scope={
                "recording_ids": list(_normalized_filters(args.recording_id)),
                "recording_path_contains": list(
                    _normalized_filters(args.recording_path_contains)
                ),
                "run_families": list(_normalized_filters(args.run_family)),
            },
            external_outputs=external_outputs,
        )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
