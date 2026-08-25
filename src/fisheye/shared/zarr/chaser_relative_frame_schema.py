"""Logical storage contracts for frame-by-chaser relative observations.

This module intentionally contains no Zarr I/O.  It describes the exact arrays
that a chaser-relative analysis may publish and validates concrete in-memory
array mappings before a writer is introduced.

The base table has one row per fish/recording frame/chaser pairing.  The body
extension is optional: a relative frame can be published from positional
providers alone, while body-frame coordinates are only published when an
anatomical provider is available and valid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from fisheye.shared.zarr.array_contracts import (
    BOOL,
    FLOAT32,
    INT64,
    UINT8,
    UINT16,
    ArrayContract,
    ArrayContractBinding,
    ArrayContractCatalog,
)


CHASER_RELATIVE_FRAME_SCHEMA_ID = "palette.analysis.chaser_relative_frame"
CHASER_RELATIVE_FRAME_SCHEMA_VERSION = 1
CHASER_RELATIVE_FRAME_LAYOUT = "frame_x_chaser_sparse_rows_v1"

CHASER_RELATIVE_FRAME_BODY_EXTENSION_SCHEMA_ID = (
    "palette.analysis.chaser_relative_frame.body_extension"
)
CHASER_RELATIVE_FRAME_BODY_EXTENSION_SCHEMA_VERSION = 1
CHASER_RELATIVE_FRAME_BODY_EXTENSION_LAYOUT = (
    "frame_x_chaser_body_frame_extension_v1"
)

CHASER_RELATIVE_FRAME_ANGLE_CONVENTION = (
    "atan2_left_over_forward_degrees_source_camera_y_down"
)
CHASER_RELATIVE_FRAME_AXIS_HANDEDNESS = "forward_cross_left_negative_source_camera_xy"

# Code zero is deliberately reserved for ``valid``.  Nonzero values are
# versioned producer reason codes; the values are stored in arrays, while this
# registry mirrors the controlled reason vocabulary emitted by the pure
# in-memory computation, including its transition/censoring states.
CHASER_RELATIVE_FRAME_REASON_CODES = {
    0: "valid",
    1: "selection_excluded",
    2: "occurrence_excluded",
    # Read compatibility only. Current writers cannot emit this category: a
    # controller's active state is orthogonal to measurable position geometry.
    3: "chaser_inactive",
    4: "fish_invalid",
    5: "chaser_invalid",
    6: "nonfinite_coordinate",
    7: "no_chaser_axis",
    8: "no_valid_chaser",
    9: "body_frame_unavailable",
    10: "body_frame_invalid",
    11: "body_frame_nonfinite",
    12: "zero_relative_vector",
    13: "no_predecessor",
    14: "nonconsecutive_acquisition_frame",
    15: "timestamp_unavailable",
    16: "nonpositive_timestamp_delta",
    17: "selection_boundary",
    18: "occurrence_boundary",
    19: "trial_boundary",
    20: "invalid_current_or_previous_position",
    21: "invalid_current_or_previous_body_frame",
    22: "source_row_unavailable",
    23: "trial_unavailable",
    24: "active_state_unavailable",
    25: "behavior_role_unavailable",
}


def _contract(
    name: str,
    *,
    dtype: Any,
    shape: tuple[str | int, ...],
    axes: tuple[str, ...],
    description: str,
    units: str | None = None,
    coordinate_space: str | None = None,
) -> ArrayContract:
    return ArrayContract(
        schema_id=f"palette.array.chaser_relative_frame.{name}",
        schema_version=1,
        dtype=dtype,
        shape_template=shape,
        axis_names=axes,
        description=description,
        units=units,
        coordinate_space=coordinate_space,
    )


_N = ("n_rows",)
_XY = ("n_rows", 2)
_ROW_AXIS = ("relative_frame_row",)
_XY_AXIS = ("relative_frame_row", "xy")


_BASE_CONTRACTS: tuple[tuple[str, ArrayContract], ...] = (
    (
        "acquisition_frame_id",
        _contract(
            "acquisition_frame_id",
            dtype=INT64,
            shape=_N,
            axes=_ROW_AXIS,
            description="Exact acquisition-camera frame identity for this row.",
            units="acquisition_frame_index",
        ),
    ),
    (
        "track_sample_id",
        _contract(
            "track_sample_id",
            dtype=INT64,
            shape=_N,
            axes=_ROW_AXIS,
            description="Exact fish tracking sample identity joined to this row.",
            units="track_sample_index",
        ),
    ),
    (
        "timestamp_ns",
        _contract(
            "timestamp_ns",
            dtype=INT64,
            shape=_N,
            axes=_ROW_AXIS,
            description="Camera acquisition timestamp for this frame row.",
            units="nanoseconds",
        ),
    ),
    (
        "timestamp_valid",
        _contract(
            "timestamp_valid",
            dtype=BOOL,
            shape=_N,
            axes=_ROW_AXIS,
            description="Whether timestamp_ns is semantically valid.",
        ),
    ),
    (
        "timestamp_reason_code",
        _contract(
            "timestamp_reason_code",
            dtype=UINT16,
            shape=_N,
            axes=_ROW_AXIS,
            description="Versioned reason code paired with timestamp_valid.",
        ),
    ),
    (
        "fish_source_row_id",
        _contract(
            "fish_source_row_id",
            dtype=INT64,
            shape=_N,
            axes=_ROW_AXIS,
            description="Exact row identity in the bound fish-position source.",
            units="source_row_index",
        ),
    ),
    (
        "fish_source_row_valid",
        _contract(
            "fish_source_row_valid",
            dtype=BOOL,
            shape=_N,
            axes=_ROW_AXIS,
            description="Whether fish_source_row_id identifies a source row.",
        ),
    ),
    (
        "fish_source_row_reason_code",
        _contract(
            "fish_source_row_reason_code",
            dtype=UINT16,
            shape=_N,
            axes=_ROW_AXIS,
            description="Versioned reason code paired with fish source-row validity.",
        ),
    ),
    (
        "chaser_source_row_id",
        _contract(
            "chaser_source_row_id",
            dtype=INT64,
            shape=_N,
            axes=_ROW_AXIS,
            description="Exact row identity in the bound chaser-position source.",
            units="source_row_index",
        ),
    ),
    (
        "chaser_source_row_valid",
        _contract(
            "chaser_source_row_valid",
            dtype=BOOL,
            shape=_N,
            axes=_ROW_AXIS,
            description="Whether chaser_source_row_id identifies a source row.",
        ),
    ),
    (
        "chaser_source_row_reason_code",
        _contract(
            "chaser_source_row_reason_code",
            dtype=UINT16,
            shape=_N,
            axes=_ROW_AXIS,
            description=(
                "Versioned reason code paired with chaser source-row validity."
            ),
        ),
    ),
    (
        "fish_position_xy_px",
        _contract(
            "fish_position_xy_px",
            dtype=FLOAT32,
            shape=_XY,
            axes=_XY_AXIS,
            description="Fish position from the selected provider in pixels.",
            units="pixels",
            coordinate_space="source_camera_continuous_pixel_xy",
        ),
    ),
    (
        "fish_position_valid",
        _contract(
            "fish_position_valid",
            dtype=BOOL,
            shape=_N,
            axes=_ROW_AXIS,
            description="Whether fish_position_xy_px is valid.",
        ),
    ),
    (
        "fish_position_reason_code",
        _contract(
            "fish_position_reason_code",
            dtype=UINT16,
            shape=_N,
            axes=_ROW_AXIS,
            description="Versioned reason code paired with fish position validity.",
        ),
    ),
    (
        "chaser_position_xy_px",
        _contract(
            "chaser_position_xy_px",
            dtype=FLOAT32,
            shape=_XY,
            axes=_XY_AXIS,
            description="Chaser position from the selected provider in pixels.",
            units="pixels",
            coordinate_space="source_camera_continuous_pixel_xy",
        ),
    ),
    (
        "chaser_position_valid",
        _contract(
            "chaser_position_valid",
            dtype=BOOL,
            shape=_N,
            axes=_ROW_AXIS,
            description="Whether chaser_position_xy_px is valid.",
        ),
    ),
    (
        "chaser_position_reason_code",
        _contract(
            "chaser_position_reason_code",
            dtype=UINT16,
            shape=_N,
            axes=_ROW_AXIS,
            description="Versioned reason code paired with chaser position validity.",
        ),
    ),
    (
        "fish_identity_code",
        _contract(
            "fish_identity_code",
            dtype=UINT16,
            shape=_N,
            axes=_ROW_AXIS,
            description="Stable bounded identity code for the fish track.",
            units="identity_code",
        ),
    ),
    (
        "chaser_identity_code",
        _contract(
            "chaser_identity_code",
            dtype=UINT16,
            shape=_N,
            axes=_ROW_AXIS,
            description="Stable bounded identity code for the chaser.",
            units="identity_code",
        ),
    ),
    (
        "chaser_behavior_role_code",
        _contract(
            "chaser_behavior_role_code",
            dtype=UINT8,
            shape=_N,
            axes=_ROW_AXIS,
            description="Time-varying versioned behavior-role code for this chaser.",
            units="behavior_role_code",
        ),
    ),
    (
        "chaser_behavior_role_valid",
        _contract(
            "chaser_behavior_role_valid",
            dtype=BOOL,
            shape=_N,
            axes=_ROW_AXIS,
            description="Whether the time-varying behavior-role code is valid.",
        ),
    ),
    (
        "chaser_behavior_role_reason_code",
        _contract(
            "chaser_behavior_role_reason_code",
            dtype=UINT16,
            shape=_N,
            axes=_ROW_AXIS,
            description="Versioned reason code paired with behavior-role validity.",
        ),
    ),
    (
        "selection_member",
        _contract(
            "selection_member",
            dtype=BOOL,
            shape=_N,
            axes=_ROW_AXIS,
            description="Whether this row belongs to the temporal protocol selection.",
        ),
    ),
    (
        "chaser_occurrence_member",
        _contract(
            "chaser_occurrence_member",
            dtype=BOOL,
            shape=_N,
            axes=_ROW_AXIS,
            description="Whether this row belongs to the retained chaser occurrence.",
        ),
    ),
    (
        "trial_id",
        _contract(
            "trial_id",
            dtype=INT64,
            shape=_N,
            axes=_ROW_AXIS,
            description="Optional trial identity; paired with trial_valid.",
            units="trial_index",
        ),
    ),
    (
        "trial_valid",
        _contract(
            "trial_valid",
            dtype=BOOL,
            shape=_N,
            axes=_ROW_AXIS,
            description="Explicit validity for optional trial_id.",
        ),
    ),
    (
        "trial_reason_code",
        _contract(
            "trial_reason_code",
            dtype=UINT16,
            shape=_N,
            axes=_ROW_AXIS,
            description="Versioned reason code paired with trial_valid.",
        ),
    ),
    (
        "active_state_code",
        _contract(
            "active_state_code",
            dtype=UINT8,
            shape=_N,
            axes=_ROW_AXIS,
            description="Optional time-varying active-state code.",
            units="active_state_code",
        ),
    ),
    (
        "active_state_valid",
        _contract(
            "active_state_valid",
            dtype=BOOL,
            shape=_N,
            axes=_ROW_AXIS,
            description="Explicit validity for optional active_state_code.",
        ),
    ),
    (
        "active_state_reason_code",
        _contract(
            "active_state_reason_code",
            dtype=UINT16,
            shape=_N,
            axes=_ROW_AXIS,
            description="Versioned reason code paired with active_state_valid.",
        ),
    ),
    (
        "row_valid",
        _contract(
            "row_valid",
            dtype=BOOL,
            shape=_N,
            axes=_ROW_AXIS,
            description="Aggregate validity of the frame-by-chaser relation row.",
        ),
    ),
    (
        "row_reason_code",
        _contract(
            "row_reason_code",
            dtype=UINT16,
            shape=_N,
            axes=_ROW_AXIS,
            description="Versioned aggregate relation-row reason code.",
        ),
    ),
    (
        "acquisition_frame_delta",
        _contract(
            "acquisition_frame_delta",
            dtype=INT64,
            shape=_N,
            axes=_ROW_AXIS,
            description=(
                "Acquisition-frame delta from the previous frame, repeated for each "
                "flattened chaser row."
            ),
            units="acquisition_frame_index_delta",
        ),
    ),
    (
        "timestamp_delta_ns",
        _contract(
            "timestamp_delta_ns",
            dtype=INT64,
            shape=_N,
            axes=_ROW_AXIS,
            description=(
                "Camera timestamp delta from the previous frame, repeated for each "
                "flattened chaser row."
            ),
            units="nanoseconds",
        ),
    ),
    (
        "fish_transition_valid",
        _contract(
            "fish_transition_valid",
            dtype=BOOL,
            shape=_N,
            axes=_ROW_AXIS,
            description=(
                "Validity of the fish transition from the previous frame, repeated "
                "for each flattened chaser row."
            ),
        ),
    ),
    (
        "fish_transition_reason_code",
        _contract(
            "fish_transition_reason_code",
            dtype=UINT16,
            shape=_N,
            axes=_ROW_AXIS,
            description="Reason code for the repeated fish transition evidence.",
        ),
    ),
    (
        "relative_transition_valid",
        _contract(
            "relative_transition_valid",
            dtype=BOOL,
            shape=_N,
            axes=_ROW_AXIS,
            description="Validity of this fish/chaser transition.",
        ),
    ),
    (
        "relative_transition_reason_code",
        _contract(
            "relative_transition_reason_code",
            dtype=UINT16,
            shape=_N,
            axes=_ROW_AXIS,
            description="Versioned reason code for this fish/chaser transition.",
        ),
    ),
    (
        "relative_vector_px_xy",
        _contract(
            "relative_vector_px_xy",
            dtype=FLOAT32,
            shape=_XY,
            axes=_XY_AXIS,
            description="Fish-to-chaser relative vector in source-camera pixels.",
            units="pixels",
            coordinate_space="source_camera_continuous_pixel_xy",
        ),
    ),
    (
        "relative_distance_px",
        _contract(
            "relative_distance_px",
            dtype=FLOAT32,
            shape=_N,
            axes=_ROW_AXIS,
            description="Euclidean fish-to-chaser distance in source-camera pixels.",
            units="pixels",
            coordinate_space="source_camera_continuous_pixel_xy",
        ),
    ),
    (
        "relative_px_valid",
        _contract(
            "relative_px_valid",
            dtype=BOOL,
            shape=_N,
            axes=_ROW_AXIS,
            description="Whether the pixel relative vector and distance are valid.",
        ),
    ),
    (
        "relative_px_reason_code",
        _contract(
            "relative_px_reason_code",
            dtype=UINT16,
            shape=_N,
            axes=_ROW_AXIS,
            description="Versioned reason code paired with pixel-relative validity.",
        ),
    ),
    (
        "relative_vector_physical_xy",
        _contract(
            "relative_vector_physical_xy",
            dtype=FLOAT32,
            shape=_XY,
            axes=_XY_AXIS,
            description="Fish-to-chaser relative vector in calibrated physical units.",
            units="calibrated_length",
            coordinate_space="source_camera_calibrated_xy",
        ),
    ),
    (
        "relative_distance_physical",
        _contract(
            "relative_distance_physical",
            dtype=FLOAT32,
            shape=_N,
            axes=_ROW_AXIS,
            description="Euclidean fish-to-chaser distance in calibrated units.",
            units="calibrated_length",
            coordinate_space="source_camera_calibrated_xy",
        ),
    ),
    (
        "relative_physical_valid",
        _contract(
            "relative_physical_valid",
            dtype=BOOL,
            shape=_N,
            axes=_ROW_AXIS,
            description="Whether calibrated relative vector and distance are valid.",
        ),
    ),
    (
        "relative_physical_reason_code",
        _contract(
            "relative_physical_reason_code",
            dtype=UINT16,
            shape=_N,
            axes=_ROW_AXIS,
            description=(
                "Versioned reason code paired with calibrated-relative validity."
            ),
        ),
    ),
    (
        "nearest_chaser_member",
        _contract(
            "nearest_chaser_member",
            dtype=BOOL,
            shape=_N,
            axes=_ROW_AXIS,
            description="Whether this row is the selected nearest-chaser projection.",
        ),
    ),
    (
        "nearest_chaser_identity_code",
        _contract(
            "nearest_chaser_identity_code",
            dtype=UINT16,
            shape=_N,
            axes=_ROW_AXIS,
            description="Identity code projected from the nearest chaser per frame.",
            units="identity_code",
        ),
    ),
    (
        "nearest_chaser_source_row_id",
        _contract(
            "nearest_chaser_source_row_id",
            dtype=INT64,
            shape=_N,
            axes=_ROW_AXIS,
            description="Source-row identity projected from the nearest chaser.",
            units="source_row_index",
        ),
    ),
    (
        "nearest_chaser_distance_px",
        _contract(
            "nearest_chaser_distance_px",
            dtype=FLOAT32,
            shape=_N,
            axes=_ROW_AXIS,
            description="Nearest-chaser distance projected in source-camera pixels.",
            units="pixels",
            coordinate_space="source_camera_continuous_pixel_xy",
        ),
    ),
    (
        "nearest_chaser_distance_physical",
        _contract(
            "nearest_chaser_distance_physical",
            dtype=FLOAT32,
            shape=_N,
            axes=_ROW_AXIS,
            description="Nearest-chaser distance projected in calibrated units.",
            units="calibrated_length",
            coordinate_space="source_camera_calibrated_xy",
        ),
    ),
    (
        "nearest_chaser_valid",
        _contract(
            "nearest_chaser_valid",
            dtype=BOOL,
            shape=_N,
            axes=_ROW_AXIS,
            description="Whether nearest-chaser projection is valid for this frame.",
        ),
    ),
    (
        "nearest_chaser_reason_code",
        _contract(
            "nearest_chaser_reason_code",
            dtype=UINT16,
            shape=_N,
            axes=_ROW_AXIS,
            description="Versioned reason code paired with nearest projection validity.",
        ),
    ),
)


_BODY_CONTRACTS: tuple[tuple[str, ArrayContract], ...] = (
    (
        "body_source_row_id",
        _contract(
            "body_source_row_id",
            dtype=INT64,
            shape=_N,
            axes=_ROW_AXIS,
            description="Exact source row identity for the body-frame provider.",
            units="source_row_index",
        ),
    ),
    (
        "body_source_row_valid",
        _contract(
            "body_source_row_valid",
            dtype=BOOL,
            shape=_N,
            axes=_ROW_AXIS,
            description="Whether body_source_row_id identifies a source row.",
        ),
    ),
    (
        "body_source_row_reason_code",
        _contract(
            "body_source_row_reason_code",
            dtype=UINT16,
            shape=_N,
            axes=_ROW_AXIS,
            description="Versioned reason code paired with body source-row validity.",
        ),
    ),
    (
        "body_origin_xy_px",
        _contract(
            "body_origin_xy_px",
            dtype=FLOAT32,
            shape=_XY,
            axes=_XY_AXIS,
            description="Body-frame origin in source-camera continuous pixels.",
            units="pixels",
            coordinate_space="source_camera_continuous_pixel_xy",
        ),
    ),
    (
        "body_forward_axis_xy",
        _contract(
            "body_forward_axis_xy",
            dtype=FLOAT32,
            shape=_XY,
            axes=_XY_AXIS,
            description="Unit anatomical-forward axis in source-camera XY.",
            coordinate_space="source_camera_unit_vector_xy",
        ),
    ),
    (
        "body_left_axis_xy",
        _contract(
            "body_left_axis_xy",
            dtype=FLOAT32,
            shape=_XY,
            axes=_XY_AXIS,
            description="Unit anatomical-left axis in source-camera XY.",
            coordinate_space="source_camera_unit_vector_xy",
        ),
    ),
    (
        "body_origin_valid",
        _contract(
            "body_origin_valid",
            dtype=BOOL,
            shape=_N,
            axes=_ROW_AXIS,
            description="Whether the body-frame origin is valid.",
        ),
    ),
    (
        "body_origin_reason_code",
        _contract(
            "body_origin_reason_code",
            dtype=UINT16,
            shape=_N,
            axes=_ROW_AXIS,
            description="Versioned reason code paired with body-origin validity.",
        ),
    ),
    (
        "body_axes_valid",
        _contract(
            "body_axes_valid",
            dtype=BOOL,
            shape=_N,
            axes=_ROW_AXIS,
            description="Whether forward and left form a valid body frame.",
        ),
    ),
    (
        "body_axes_reason_code",
        _contract(
            "body_axes_reason_code",
            dtype=UINT16,
            shape=_N,
            axes=_ROW_AXIS,
            description="Versioned reason code paired with body-axis validity.",
        ),
    ),
    (
        "body_relative_vector_px_xy",
        _contract(
            "body_relative_vector_px_xy",
            dtype=FLOAT32,
            shape=_XY,
            axes=_XY_AXIS,
            description=(
                "Chaser position minus the body-frame origin in source-camera pixels; "
                "this is independent of the fish position provider."
            ),
            units="pixels",
            coordinate_space="source_camera_continuous_pixel_xy",
        ),
    ),
    (
        "body_relative_px_valid",
        _contract(
            "body_relative_px_valid",
            dtype=BOOL,
            shape=_N,
            axes=_ROW_AXIS,
            description="Whether body_relative_vector_px_xy is valid.",
        ),
    ),
    (
        "body_relative_px_reason_code",
        _contract(
            "body_relative_px_reason_code",
            dtype=UINT16,
            shape=_N,
            axes=_ROW_AXIS,
            description="Versioned reason code paired with body-relative pixel validity.",
        ),
    ),
    (
        "body_relative_vector_physical_xy",
        _contract(
            "body_relative_vector_physical_xy",
            dtype=FLOAT32,
            shape=_XY,
            axes=_XY_AXIS,
            description="Chaser position minus body origin in calibrated units.",
            units="calibrated_length",
            coordinate_space="source_camera_calibrated_xy",
        ),
    ),
    (
        "body_relative_physical_valid",
        _contract(
            "body_relative_physical_valid",
            dtype=BOOL,
            shape=_N,
            axes=_ROW_AXIS,
            description="Whether body_relative_vector_physical_xy is valid.",
        ),
    ),
    (
        "body_relative_physical_reason_code",
        _contract(
            "body_relative_physical_reason_code",
            dtype=UINT16,
            shape=_N,
            axes=_ROW_AXIS,
            description=(
                "Versioned reason code paired with body-relative physical validity."
            ),
        ),
    ),
    (
        "body_heading_deg",
        _contract(
            "body_heading_deg",
            dtype=FLOAT32,
            shape=_N,
            axes=_ROW_AXIS,
            description="Heading derived from the anatomical-forward axis.",
            units="degrees",
            coordinate_space="source_camera_angle",
        ),
    ),
    (
        "body_heading_valid",
        _contract(
            "body_heading_valid",
            dtype=BOOL,
            shape=_N,
            axes=_ROW_AXIS,
            description="Whether body_heading_deg is valid.",
        ),
    ),
    (
        "body_heading_reason_code",
        _contract(
            "body_heading_reason_code",
            dtype=UINT16,
            shape=_N,
            axes=_ROW_AXIS,
            description="Versioned reason code paired with heading validity.",
        ),
    ),
    (
        "body_heading_transition_valid",
        _contract(
            "body_heading_transition_valid",
            dtype=BOOL,
            shape=_N,
            axes=_ROW_AXIS,
            description=(
                "Validity of the body-heading transition from the previous frame, "
                "repeated for each flattened chaser row."
            ),
        ),
    ),
    (
        "body_heading_transition_reason_code",
        _contract(
            "body_heading_transition_reason_code",
            dtype=UINT16,
            shape=_N,
            axes=_ROW_AXIS,
            description="Reason code for the repeated body-heading transition evidence.",
        ),
    ),
    (
        "body_forward_coordinate_px",
        _contract(
            "body_forward_coordinate_px",
            dtype=FLOAT32,
            shape=_N,
            axes=_ROW_AXIS,
            description="Relative vector projection onto body-forward in pixels.",
            units="pixels",
            coordinate_space="body_frame_pixel_xy",
        ),
    ),
    (
        "body_left_coordinate_px",
        _contract(
            "body_left_coordinate_px",
            dtype=FLOAT32,
            shape=_N,
            axes=_ROW_AXIS,
            description="Relative vector projection onto body-left in pixels.",
            units="pixels",
            coordinate_space="body_frame_pixel_xy",
        ),
    ),
    (
        "body_coordinates_px_valid",
        _contract(
            "body_coordinates_px_valid",
            dtype=BOOL,
            shape=_N,
            axes=_ROW_AXIS,
            description="Whether body-frame pixel coordinates are valid.",
        ),
    ),
    (
        "body_coordinates_px_reason_code",
        _contract(
            "body_coordinates_px_reason_code",
            dtype=UINT16,
            shape=_N,
            axes=_ROW_AXIS,
            description="Versioned reason code paired with body pixel-coordinate validity.",
        ),
    ),
    (
        "body_forward_coordinate_physical",
        _contract(
            "body_forward_coordinate_physical",
            dtype=FLOAT32,
            shape=_N,
            axes=_ROW_AXIS,
            description="Relative vector projection onto body-forward in calibrated units.",
            units="calibrated_length",
            coordinate_space="body_frame_calibrated_xy",
        ),
    ),
    (
        "body_left_coordinate_physical",
        _contract(
            "body_left_coordinate_physical",
            dtype=FLOAT32,
            shape=_N,
            axes=_ROW_AXIS,
            description="Relative vector projection onto body-left in calibrated units.",
            units="calibrated_length",
            coordinate_space="body_frame_calibrated_xy",
        ),
    ),
    (
        "body_coordinates_physical_valid",
        _contract(
            "body_coordinates_physical_valid",
            dtype=BOOL,
            shape=_N,
            axes=_ROW_AXIS,
            description="Whether body-frame calibrated coordinates are valid.",
        ),
    ),
    (
        "body_coordinates_physical_reason_code",
        _contract(
            "body_coordinates_physical_reason_code",
            dtype=UINT16,
            shape=_N,
            axes=_ROW_AXIS,
            description=(
                "Versioned reason code paired with body calibrated-coordinate validity."
            ),
        ),
    ),
    (
        "body_bearing_deg",
        _contract(
            "body_bearing_deg",
            dtype=FLOAT32,
            shape=_N,
            axes=_ROW_AXIS,
            description="Signed bearing from body-forward toward anatomical-left.",
            units="degrees",
            coordinate_space="body_frame_angle",
        ),
    ),
    (
        "body_bearing_valid",
        _contract(
            "body_bearing_valid",
            dtype=BOOL,
            shape=_N,
            axes=_ROW_AXIS,
            description="Whether body_bearing_deg is valid.",
        ),
    ),
    (
        "body_bearing_reason_code",
        _contract(
            "body_bearing_reason_code",
            dtype=UINT16,
            shape=_N,
            axes=_ROW_AXIS,
            description="Versioned reason code paired with body-bearing validity.",
        ),
    ),
    (
        "body_valid",
        _contract(
            "body_valid",
            dtype=BOOL,
            shape=_N,
            axes=_ROW_AXIS,
            description="Aggregate validity of the body-frame extension row.",
        ),
    ),
    (
        "body_reason_code",
        _contract(
            "body_reason_code",
            dtype=UINT16,
            shape=_N,
            axes=_ROW_AXIS,
            description="Versioned aggregate body-frame reason code.",
        ),
    ),
)


def _bindings(
    contracts: tuple[tuple[str, ArrayContract], ...],
    *,
    required: frozenset[str],
) -> tuple[ArrayContractBinding, ...]:
    return tuple(
        ArrayContractBinding(
            path=name,
            contract_id=contract.schema_id,
            contract_version=contract.schema_version,
            required=name in required,
        )
        for name, contract in contracts
    )


_OPTIONAL_BASE = frozenset(
    {
        "trial_id",
        "trial_valid",
        "trial_reason_code",
        "active_state_code",
        "active_state_valid",
        "active_state_reason_code",
    }
)
_BASE_NAMES = frozenset(name for name, _ in _BASE_CONTRACTS)
_BODY_NAMES = frozenset(name for name, _ in _BODY_CONTRACTS)

CHASER_RELATIVE_FRAME_ARRAY_CONTRACTS = ArrayContractCatalog(
    contract for _, contract in _BASE_CONTRACTS
)
CHASER_RELATIVE_FRAME_BODY_ARRAY_CONTRACTS = ArrayContractCatalog(
    contract for _, contract in _BODY_CONTRACTS
)
CHASER_RELATIVE_FRAME_BINDINGS = _bindings(
    _BASE_CONTRACTS,
    required=_BASE_NAMES - _OPTIONAL_BASE,
)
CHASER_RELATIVE_FRAME_BODY_BINDINGS = _bindings(
    _BODY_CONTRACTS,
    required=_BODY_NAMES,
)


@dataclass(frozen=True)
class ChaserRelativeFrameDimensions:
    """Concrete row dimensions for a frame-by-chaser sidecar."""

    n_rows: int

    def __post_init__(self) -> None:
        if type(self.n_rows) is not int:
            raise TypeError("n_rows must be an exact integer.")
        if self.n_rows < 0:
            raise ValueError("n_rows cannot be negative.")

    @property
    def contract_dimensions(self) -> dict[str, int]:
        return {"n_rows": self.n_rows}

    def as_manifest(self) -> dict[str, int]:
        return dict(self.contract_dimensions)


@dataclass(frozen=True)
class ChaserRelativeFrameSchemaIssue:
    code: str
    path: str
    message: str

    def as_manifest(self) -> dict[str, str]:
        return {"code": self.code, "path": self.path, "message": self.message}


class ChaserRelativeFrameSchemaError(ValueError):
    """Raised when a base or body-extension mapping violates this contract."""

    def __init__(self, issues: tuple[ChaserRelativeFrameSchemaIssue, ...]) -> None:
        self.issues = issues
        detail = "; ".join(
            f"{issue.code} at {issue.path}: {issue.message}" for issue in issues
        )
        super().__init__(
            "Chaser-relative frame schema validation failed with "
            f"{len(issues)} issue(s): {detail}"
        )


def _issue(code: str, path: str, message: str) -> ChaserRelativeFrameSchemaIssue:
    return ChaserRelativeFrameSchemaIssue(code=code, path=path, message=message)


def _materialize(array: Any) -> np.ndarray:
    if isinstance(array, np.ndarray):
        return array
    try:
        return np.asarray(array[...])
    except (IndexError, KeyError, TypeError):
        return np.asarray(array)


def _finite_rows(array: np.ndarray) -> np.ndarray:
    if array.ndim == 1:
        return np.isfinite(array)
    return np.all(np.isfinite(array), axis=tuple(range(1, array.ndim)))


def _validate_validity_reason(
    values: Mapping[str, np.ndarray],
    *,
    valid_path: str,
    reason_path: str,
    issues: list[ChaserRelativeFrameSchemaIssue],
) -> None:
    valid = values.get(valid_path)
    reason = values.get(reason_path)
    if valid is None or reason is None:
        return
    valid_with_reason = valid & (reason != 0)
    invalid_without_reason = (~valid) & (reason == 0)
    if np.any(valid_with_reason):
        issues.append(
            _issue(
                "valid_reason_mismatch",
                reason_path,
                f"{reason_path} must be zero for valid {valid_path} rows.",
            )
        )
    if np.any(invalid_without_reason):
        issues.append(
            _issue(
                "missing_invalid_reason",
                reason_path,
                f"Invalid {valid_path} rows require a nonzero reason code.",
            )
        )


def _validate_float_validity(
    values: Mapping[str, np.ndarray],
    *,
    value_paths: tuple[str, ...],
    valid_path: str,
    issues: list[ChaserRelativeFrameSchemaIssue],
) -> None:
    valid = values.get(valid_path)
    if valid is None:
        return
    for path in value_paths:
        value = values.get(path)
        if value is None:
            continue
        finite = _finite_rows(value)
        if np.any(valid & ~finite):
            issues.append(
                _issue(
                    "valid_value_not_finite",
                    path,
                    f"{path} must be finite wherever {valid_path} is true.",
                )
            )
        if np.any(~valid & finite):
            issues.append(
                _issue(
                    "invalid_value_not_nan",
                    path,
                    f"{path} must be NaN wherever {valid_path} is false.",
                )
            )


def _validate_source_rows(
    values: Mapping[str, np.ndarray],
    *,
    row_path: str,
    valid_path: str,
    reason_path: str,
    issues: list[ChaserRelativeFrameSchemaIssue],
) -> None:
    rows = values.get(row_path)
    valid = values.get(valid_path)
    if rows is None or valid is None:
        return
    if np.any(valid & (rows < 0)):
        issues.append(
            _issue(
                "valid_source_row_negative",
                row_path,
                f"Valid {row_path} values must be nonnegative.",
            )
        )
    if np.any(~valid & (rows != -1)):
        issues.append(
            _issue(
                "invalid_source_row_sentinel_mismatch",
                row_path,
                f"Invalid {row_path} values must use the int64 sentinel -1.",
            )
        )
    _validate_validity_reason(
        values,
        valid_path=valid_path,
        reason_path=reason_path,
        issues=issues,
    )


def _validate_bindings(
    arrays: Mapping[str, Any],
    *,
    bindings: tuple[ArrayContractBinding, ...],
    contracts: ArrayContractCatalog,
    dimensions: ChaserRelativeFrameDimensions,
    issues: list[ChaserRelativeFrameSchemaIssue],
) -> tuple[dict[str, np.ndarray], set[str]]:
    expected = {binding.path for binding in bindings}
    invalid_paths: set[str] = set()
    for path in sorted(set(arrays) - expected):
        issues.append(
            _issue(
                "unexpected_array",
                path,
                "The exact schema does not declare this array.",
            )
        )
    values: dict[str, np.ndarray] = {}
    for binding in bindings:
        path = binding.path
        if path not in arrays:
            if binding.required:
                invalid_paths.add(path)
                issues.append(
                    _issue("missing_required_array", path, "Required array is absent.")
                )
            continue
        contract = contracts.resolve(binding.contract_id, binding.contract_version)
        try:
            errors = contract.validate_observation(
                arrays[path], dimensions=dimensions.contract_dimensions
            )
        except Exception as exc:
            errors = (f"array metadata is unreadable: {exc}",)
        if errors:
            invalid_paths.add(path)
            issues.extend(
                _issue("array_contract_violation", path, error) for error in errors
            )
            continue
        try:
            values[path] = _materialize(arrays[path])
        except Exception as exc:
            invalid_paths.add(path)
            issues.append(_issue("array_read_failure", path, str(exc)))
    return values, invalid_paths


def _validate_optional_pairs(
    arrays: Mapping[str, Any],
    *,
    left: str,
    right: str,
    issues: list[ChaserRelativeFrameSchemaIssue],
) -> None:
    if (left in arrays) != (right in arrays):
        missing = right if left in arrays else left
        issues.append(
            _issue(
                "missing_optional_validity_pair",
                missing,
                f"{left} and {right} must be supplied together.",
            )
        )


def _validate_base_invariants(
    values: Mapping[str, np.ndarray],
    *,
    issues: list[ChaserRelativeFrameSchemaIssue],
) -> None:
    for path in ("acquisition_frame_id", "track_sample_id"):
        array = values.get(path)
        if array is not None and np.any(array < 0):
            issues.append(
                _issue(
                    "negative_index",
                    path,
                    "Frame and track sample indices must be nonnegative.",
                )
            )

    timestamp = values.get("timestamp_ns")
    timestamp_valid = values.get("timestamp_valid")
    if timestamp is not None and timestamp_valid is not None:
        if np.any(timestamp_valid & (timestamp < 0)):
            issues.append(
                _issue(
                    "negative_timestamp",
                    "timestamp_ns",
                    "Valid timestamps must be nonnegative nanosecond values.",
                )
            )
        if np.any(~timestamp_valid & (timestamp != -1)):
            issues.append(
                _issue(
                    "invalid_timestamp_sentinel_mismatch",
                    "timestamp_ns",
                    "Invalid timestamps must use the int64 sentinel -1.",
                )
            )

    for prefix in ("timestamp", "fish_source_row", "chaser_source_row"):
        _validate_validity_reason(
            values,
            valid_path=f"{prefix}_valid",
            reason_path=f"{prefix}_reason_code",
            issues=issues,
        )
    _validate_source_rows(
        values,
        row_path="fish_source_row_id",
        valid_path="fish_source_row_valid",
        reason_path="fish_source_row_reason_code",
        issues=issues,
    )
    _validate_source_rows(
        values,
        row_path="chaser_source_row_id",
        valid_path="chaser_source_row_valid",
        reason_path="chaser_source_row_reason_code",
        issues=issues,
    )
    _validate_validity_reason(
        values,
        valid_path="chaser_behavior_role_valid",
        reason_path="chaser_behavior_role_reason_code",
        issues=issues,
    )
    for prefix in ("fish_position", "chaser_position"):
        _validate_validity_reason(
            values,
            valid_path=f"{prefix}_valid",
            reason_path=f"{prefix}_reason_code",
            issues=issues,
        )
        _validate_float_validity(
            values,
            value_paths=(f"{prefix}_xy_px",),
            valid_path=f"{prefix}_valid",
            issues=issues,
        )
    for prefix in ("trial", "active_state"):
        _validate_validity_reason(
            values,
            valid_path=f"{prefix}_valid",
            reason_path=f"{prefix}_reason_code",
            issues=issues,
        )
    _validate_validity_reason(
        values,
        valid_path="row_valid",
        reason_path="row_reason_code",
        issues=issues,
    )
    for valid_path, reason_path in (
        ("fish_transition_valid", "fish_transition_reason_code"),
        ("relative_transition_valid", "relative_transition_reason_code"),
    ):
        _validate_validity_reason(
            values,
            valid_path=valid_path,
            reason_path=reason_path,
            issues=issues,
        )

    for valid_path, reason_path, value_paths in (
        (
            "relative_px_valid",
            "relative_px_reason_code",
            ("relative_vector_px_xy", "relative_distance_px"),
        ),
        (
            "relative_physical_valid",
            "relative_physical_reason_code",
            ("relative_vector_physical_xy", "relative_distance_physical"),
        ),
        (
            "nearest_chaser_valid",
            "nearest_chaser_reason_code",
            ("nearest_chaser_distance_px", "nearest_chaser_distance_physical"),
        ),
    ):
        _validate_validity_reason(
            values,
            valid_path=valid_path,
            reason_path=reason_path,
            issues=issues,
        )
        _validate_float_validity(
            values,
            value_paths=value_paths,
            valid_path=valid_path,
            issues=issues,
        )

    fish_position = values.get("fish_position_xy_px")
    chaser_position = values.get("chaser_position_xy_px")
    fish_position_valid = values.get("fish_position_valid")
    chaser_position_valid = values.get("chaser_position_valid")
    relative_px = values.get("relative_vector_px_xy")
    relative_px_valid = values.get("relative_px_valid")
    if (
        fish_position is not None
        and chaser_position is not None
        and fish_position_valid is not None
        and chaser_position_valid is not None
        and relative_px is not None
        and relative_px_valid is not None
    ):
        comparable = relative_px_valid & fish_position_valid & chaser_position_valid
        expected_relative = (
            chaser_position.astype(np.float64) - fish_position.astype(np.float64)
        )
        if np.any(comparable) and not np.allclose(
            relative_px[comparable], expected_relative[comparable], atol=5e-4, rtol=0.0
        ):
            issues.append(
                _issue(
                    "relative_pixel_derivation_mismatch",
                    "relative_vector_px_xy",
                    "Relative pixels must equal chaser position minus fish position.",
                )
            )

    nearest_valid = values.get("nearest_chaser_valid")
    nearest_source = values.get("nearest_chaser_source_row_id")
    if nearest_valid is not None and nearest_source is not None:
        if np.any(nearest_valid & (nearest_source < 0)):
            issues.append(
                _issue(
                    "nearest_source_row_negative",
                    "nearest_chaser_source_row_id",
                    "Valid nearest projections require a nonnegative source row.",
                )
            )
        if np.any(~nearest_valid & (nearest_source != -1)):
            issues.append(
                _issue(
                    "nearest_source_row_sentinel_mismatch",
                    "nearest_chaser_source_row_id",
                    "Invalid nearest projections require the int64 sentinel -1.",
                )
            )


def _validate_body_invariants(
    values: Mapping[str, np.ndarray],
    *,
    base_values: Mapping[str, np.ndarray] | None,
    issues: list[ChaserRelativeFrameSchemaIssue],
) -> None:
    _validate_validity_reason(
        values,
        valid_path="body_source_row_valid",
        reason_path="body_source_row_reason_code",
        issues=issues,
    )
    _validate_source_rows(
        values,
        row_path="body_source_row_id",
        valid_path="body_source_row_valid",
        reason_path="body_source_row_reason_code",
        issues=issues,
    )
    for prefix in (
        "body_origin",
        "body_axes",
        "body_relative_px",
        "body_relative_physical",
        "body_heading",
        "body_heading_transition",
        "body_coordinates_px",
        "body_coordinates_physical",
        "body_bearing",
        "body",
    ):
        _validate_validity_reason(
            values,
            valid_path=f"{prefix}_valid",
            reason_path=f"{prefix}_reason_code",
            issues=issues,
        )

    _validate_float_validity(
        values,
        value_paths=("body_origin_xy_px",),
        valid_path="body_origin_valid",
        issues=issues,
    )
    _validate_float_validity(
        values,
        value_paths=("body_forward_axis_xy", "body_left_axis_xy"),
        valid_path="body_axes_valid",
        issues=issues,
    )
    _validate_float_validity(
        values,
        value_paths=("body_relative_vector_px_xy",),
        valid_path="body_relative_px_valid",
        issues=issues,
    )
    _validate_float_validity(
        values,
        value_paths=("body_relative_vector_physical_xy",),
        valid_path="body_relative_physical_valid",
        issues=issues,
    )
    _validate_float_validity(
        values,
        value_paths=("body_heading_deg",),
        valid_path="body_heading_valid",
        issues=issues,
    )
    _validate_float_validity(
        values,
        value_paths=("body_forward_coordinate_px", "body_left_coordinate_px"),
        valid_path="body_coordinates_px_valid",
        issues=issues,
    )
    _validate_float_validity(
        values,
        value_paths=(
            "body_forward_coordinate_physical",
            "body_left_coordinate_physical",
        ),
        valid_path="body_coordinates_physical_valid",
        issues=issues,
    )
    _validate_float_validity(
        values,
        value_paths=("body_bearing_deg",),
        valid_path="body_bearing_valid",
        issues=issues,
    )

    axes_valid = values.get("body_axes_valid")
    forward = values.get("body_forward_axis_xy")
    left = values.get("body_left_axis_xy")
    if axes_valid is not None and forward is not None and left is not None:
        selected_forward = forward[axes_valid].astype(np.float64, copy=False)
        selected_left = left[axes_valid].astype(np.float64, copy=False)
        if selected_forward.size:
            forward_norm = np.linalg.norm(selected_forward, axis=1)
            left_norm = np.linalg.norm(selected_left, axis=1)
            dot = np.einsum("ij,ij->i", selected_forward, selected_left)
            determinant = (
                selected_forward[:, 0] * selected_left[:, 1]
                - selected_forward[:, 1] * selected_left[:, 0]
            )
            if not (
                np.allclose(forward_norm, 1.0, atol=5e-5, rtol=0.0)
                and np.allclose(left_norm, 1.0, atol=5e-5, rtol=0.0)
                and np.allclose(dot, 0.0, atol=5e-5, rtol=0.0)
                and np.allclose(determinant, -1.0, atol=5e-5, rtol=0.0)
            ):
                issues.append(
                    _issue(
                        "invalid_body_axes",
                        "body_forward_axis_xy",
                        "Valid body axes must be orthonormal with determinant -1.",
                    )
                )

    heading_valid = values.get("body_heading_valid")
    heading = values.get("body_heading_deg")
    if heading_valid is not None and heading is not None and forward is not None:
        expected_heading = np.rad2deg(
            np.arctan2(-forward[:, 1], forward[:, 0])
        ).astype(np.float32)
        if np.any(
            heading_valid
            & ~np.isclose(heading, expected_heading, atol=5e-5, rtol=0.0)
        ):
            issues.append(
                _issue(
                    "heading_derivation_mismatch",
                    "body_heading_deg",
                    "Heading must equal atan2(-forward_y, forward_x) in degrees.",
                )
            )

    px_valid = values.get("body_coordinates_px_valid")
    body_relative_px = values.get("body_relative_vector_px_xy")
    body_relative_px_valid = values.get("body_relative_px_valid")
    physical_valid = values.get("body_coordinates_physical_valid")
    if px_valid is not None and axes_valid is not None:
        forward_coord = values.get("body_forward_coordinate_px")
        left_coord = values.get("body_left_coordinate_px")
        if (
            body_relative_px is not None
            and body_relative_px_valid is not None
            and forward_coord is not None
            and left_coord is not None
        ):
            comparable = px_valid & body_relative_px_valid & axes_valid
            expected_forward = np.einsum(
                "ij,ij->i",
                body_relative_px.astype(np.float64),
                forward.astype(np.float64),
            )
            expected_left = np.einsum(
                "ij,ij->i",
                body_relative_px.astype(np.float64),
                left.astype(np.float64),
            )
            if np.any(comparable) and not (
                np.allclose(
                    forward_coord[comparable], expected_forward[comparable], atol=5e-4
                )
                and np.allclose(
                    left_coord[comparable], expected_left[comparable], atol=5e-4
                )
            ):
                issues.append(
                    _issue(
                        "body_pixel_projection_mismatch",
                        "body_forward_coordinate_px",
                        "Body pixel coordinates must be dot products with the "
                        "body-origin-relative vector and axes.",
                    )
                )

    if base_values is not None:
        chaser_position = base_values.get("chaser_position_xy_px")
        chaser_position_valid = base_values.get("chaser_position_valid")
        origin = values.get("body_origin_xy_px")
        origin_valid = values.get("body_origin_valid")
        if (
            body_relative_px is not None
            and body_relative_px_valid is not None
            and chaser_position is not None
            and chaser_position_valid is not None
            and origin is not None
            and origin_valid is not None
        ):
            comparable = body_relative_px_valid & chaser_position_valid & origin_valid
            expected = chaser_position.astype(np.float64) - origin.astype(np.float64)
            if np.any(comparable) and not np.allclose(
                body_relative_px[comparable], expected[comparable], atol=5e-4, rtol=0.0
            ):
                issues.append(
                    _issue(
                        "body_relative_pixel_derivation_mismatch",
                        "body_relative_vector_px_xy",
                        "Body-relative pixels must equal chaser position minus body origin.",
                    )
                )

    bearing_valid = values.get("body_bearing_valid")
    bearing = values.get("body_bearing_deg")
    forward_coord = values.get("body_forward_coordinate_px")
    left_coord = values.get("body_left_coordinate_px")
    if (
        bearing_valid is not None
        and bearing is not None
        and forward_coord is not None
        and left_coord is not None
    ):
        comparable = bearing_valid & (px_valid if px_valid is not None else False)
        expected_bearing = np.rad2deg(
            np.arctan2(left_coord, forward_coord)
        ).astype(np.float32)
        if np.any(comparable) and not np.allclose(
            bearing[comparable], expected_bearing[comparable], atol=5e-4, rtol=0.0
        ):
            issues.append(
                _issue(
                    "bearing_derivation_mismatch",
                    "body_bearing_deg",
                    "Bearing must equal atan2(left_coordinate, forward_coordinate).",
                )
            )

    body_relative_physical = values.get("body_relative_vector_physical_xy")
    body_relative_physical_valid = values.get("body_relative_physical_valid")
    if physical_valid is not None and axes_valid is not None:
        forward_physical = values.get("body_forward_coordinate_physical")
        left_physical = values.get("body_left_coordinate_physical")
        if (
            body_relative_physical is not None
            and body_relative_physical_valid is not None
            and forward_physical is not None
            and left_physical is not None
        ):
            comparable = physical_valid & body_relative_physical_valid & axes_valid
            expected_forward = np.einsum(
                "ij,ij->i",
                body_relative_physical.astype(np.float64),
                forward.astype(np.float64),
            )
            expected_left = np.einsum(
                "ij,ij->i",
                body_relative_physical.astype(np.float64),
                left.astype(np.float64),
            )
            if np.any(comparable) and not (
                np.allclose(
                    forward_physical[comparable], expected_forward[comparable], atol=5e-4
                )
                and np.allclose(
                    left_physical[comparable], expected_left[comparable], atol=5e-4
                )
            ):
                issues.append(
                    _issue(
                        "body_physical_projection_mismatch",
                        "body_forward_coordinate_physical",
                        "Body physical coordinates must be dot products with the "
                        "body-origin-relative vector and axes.",
                    )
                )


@dataclass(frozen=True)
class ChaserRelativeFrameBodyExtensionSchema:
    schema_id: str
    schema_version: int
    bindings: tuple[ArrayContractBinding, ...]
    contracts: ArrayContractCatalog

    @property
    def binding_paths(self) -> tuple[str, ...]:
        return tuple(binding.path for binding in self.bindings)

    def validate(
        self,
        arrays: Mapping[str, Any],
        *,
        dimensions: ChaserRelativeFrameDimensions,
        base_arrays: Mapping[str, Any] | None = None,
    ) -> tuple[ChaserRelativeFrameSchemaIssue, ...]:
        issues: list[ChaserRelativeFrameSchemaIssue] = []
        values, _ = _validate_bindings(
            arrays,
            bindings=self.bindings,
            contracts=self.contracts,
            dimensions=dimensions,
            issues=issues,
        )
        base_values = None
        if base_arrays is not None:
            base_values = dict(base_arrays)
        _validate_body_invariants(values, base_values=base_values, issues=issues)
        return tuple(issues)

    def require(
        self,
        arrays: Mapping[str, Any],
        *,
        dimensions: ChaserRelativeFrameDimensions,
        base_arrays: Mapping[str, Any] | None = None,
    ) -> None:
        issues = self.validate(
            arrays, dimensions=dimensions, base_arrays=base_arrays
        )
        if issues:
            raise ChaserRelativeFrameSchemaError(issues)

    def as_manifest(
        self, *, dimensions: ChaserRelativeFrameDimensions
    ) -> dict[str, object]:
        return {
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "stage": "chaser_relative_frame_body_extension",
            "layout": CHASER_RELATIVE_FRAME_BODY_EXTENSION_LAYOUT,
            "dimensions": dimensions.as_manifest(),
            "bindings": [binding.as_manifest() for binding in self.bindings],
            "array_contracts": self.contracts.as_manifest(),
            "invariants": {
                "coordinate_space": "source_camera_continuous_pixel_xy",
                "axis_handedness": CHASER_RELATIVE_FRAME_AXIS_HANDEDNESS,
                "heading_derivation": CHASER_RELATIVE_FRAME_ANGLE_CONVENTION,
                "bearing_derivation": "atan2(left_coordinate, forward_coordinate)",
                "body_relative_vector": "chaser_position_minus_body_origin",
                "invalid_float_values": "NaN",
            },
        }


@dataclass(frozen=True)
class ChaserRelativeFrameSchema:
    schema_id: str
    schema_version: int
    bindings: tuple[ArrayContractBinding, ...]
    contracts: ArrayContractCatalog
    body_extension: ChaserRelativeFrameBodyExtensionSchema

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
        dimensions: ChaserRelativeFrameDimensions,
        body_arrays: Mapping[str, Any] | None = None,
    ) -> tuple[ChaserRelativeFrameSchemaIssue, ...]:
        issues: list[ChaserRelativeFrameSchemaIssue] = []
        values, _ = _validate_bindings(
            arrays,
            bindings=self.bindings,
            contracts=self.contracts,
            dimensions=dimensions,
            issues=issues,
        )
        for left, right in (
            ("trial_id", "trial_valid"),
            ("trial_valid", "trial_reason_code"),
            ("active_state_code", "active_state_valid"),
            ("active_state_valid", "active_state_reason_code"),
        ):
            _validate_optional_pairs(arrays, left=left, right=right, issues=issues)
        _validate_base_invariants(values, issues=issues)
        if body_arrays is not None:
            issues.extend(
                self.body_extension.validate(
                    body_arrays,
                    dimensions=dimensions,
                    base_arrays=values,
                )
            )
        return tuple(issues)

    def require(
        self,
        arrays: Mapping[str, Any],
        *,
        dimensions: ChaserRelativeFrameDimensions,
        body_arrays: Mapping[str, Any] | None = None,
    ) -> None:
        issues = self.validate(
            arrays, dimensions=dimensions, body_arrays=body_arrays
        )
        if issues:
            raise ChaserRelativeFrameSchemaError(issues)

    def as_manifest(
        self, *, dimensions: ChaserRelativeFrameDimensions
    ) -> dict[str, object]:
        return {
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "stage": "chaser_relative_frame",
            "layout": CHASER_RELATIVE_FRAME_LAYOUT,
            "dimensions": dimensions.as_manifest(),
            "bindings": [binding.as_manifest() for binding in self.bindings],
            "array_contracts": self.contracts.as_manifest(),
            "body_extension": self.body_extension.as_manifest(dimensions=dimensions),
            "reason_codes": {str(k): v for k, v in CHASER_RELATIVE_FRAME_REASON_CODES.items()},
            "invariants": {
                "row_axis": "frame_x_chaser",
                "row_evidence": "source_row_ids_are_persisted_arrays",
                "selection": "selection_member_is_temporal_membership_not_chaser_identity",
                "coordinate_space": "source_camera_continuous_pixel_xy",
                "physical_space": "calibrated_length_xy",
                "invalid_float_values": "NaN",
                "optional_state": "trial_and_active_state_require_explicit_validity_arrays",
                "nearest_projection": (
                    "nearest_chaser_fields_are_repeated_on_each_frame_x_chaser_row"
                ),
                "frame_only_evidence": (
                    "acquisition_frame_delta, timestamp_delta_ns, "
                    "fish_transition_valid, and fish_transition_reason_code are "
                    "repeated for each chaser row"
                ),
                "relative_transition": (
                    "relative_transition_valid_and_reason_code_are_per_chaser_row"
                ),
                "body_transition": (
                    "body_heading_transition_valid and "
                    "body_heading_transition_reason_code are repeated for each "
                    "chaser row"
                ),
            },
        }


CHASER_RELATIVE_FRAME_BODY_EXTENSION_SCHEMA_V1 = (
    ChaserRelativeFrameBodyExtensionSchema(
        schema_id=CHASER_RELATIVE_FRAME_BODY_EXTENSION_SCHEMA_ID,
        schema_version=CHASER_RELATIVE_FRAME_BODY_EXTENSION_SCHEMA_VERSION,
        bindings=CHASER_RELATIVE_FRAME_BODY_BINDINGS,
        contracts=CHASER_RELATIVE_FRAME_BODY_ARRAY_CONTRACTS,
    )
)

CHASER_RELATIVE_FRAME_SCHEMA_V1 = ChaserRelativeFrameSchema(
    schema_id=CHASER_RELATIVE_FRAME_SCHEMA_ID,
    schema_version=CHASER_RELATIVE_FRAME_SCHEMA_VERSION,
    bindings=CHASER_RELATIVE_FRAME_BINDINGS,
    contracts=CHASER_RELATIVE_FRAME_ARRAY_CONTRACTS,
    body_extension=CHASER_RELATIVE_FRAME_BODY_EXTENSION_SCHEMA_V1,
)


__all__ = [
    "CHASER_RELATIVE_FRAME_ANGLE_CONVENTION",
    "CHASER_RELATIVE_FRAME_ARRAY_CONTRACTS",
    "CHASER_RELATIVE_FRAME_BINDINGS",
    "CHASER_RELATIVE_FRAME_BODY_ARRAY_CONTRACTS",
    "CHASER_RELATIVE_FRAME_BODY_BINDINGS",
    "CHASER_RELATIVE_FRAME_BODY_EXTENSION_LAYOUT",
    "CHASER_RELATIVE_FRAME_BODY_EXTENSION_SCHEMA_ID",
    "CHASER_RELATIVE_FRAME_BODY_EXTENSION_SCHEMA_V1",
    "CHASER_RELATIVE_FRAME_BODY_EXTENSION_SCHEMA_VERSION",
    "CHASER_RELATIVE_FRAME_LAYOUT",
    "CHASER_RELATIVE_FRAME_REASON_CODES",
    "CHASER_RELATIVE_FRAME_SCHEMA_ID",
    "CHASER_RELATIVE_FRAME_SCHEMA_V1",
    "CHASER_RELATIVE_FRAME_SCHEMA_VERSION",
    "ChaserRelativeFrameBodyExtensionSchema",
    "ChaserRelativeFrameDimensions",
    "ChaserRelativeFrameSchema",
    "ChaserRelativeFrameSchemaError",
    "ChaserRelativeFrameSchemaIssue",
]
