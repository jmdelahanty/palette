"""Comprehensive track kinematics aggregation for Palette archives.

This module consolidates detections, arena assignments, keypoint headings, and
calibration metadata into an analysis-friendly layout under
``analysis/track_kinematics_runs``.

It prefers refined keypoints/detections when available, writes per-track
subgroups with rich kinematic metrics, and records provenance back to the
source runs so downstream tooling can trace inputs.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import math
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import zarr
from rich.console import Console
from rich.table import Table

from .compute_speed import (  # re-exported for compatibility
    HYSTERESIS_BAND_POLICIES,
    SMOOTHING_ALIGNMENTS,
    TRANSITION_REASON_CODES,
    TrackSpeeds,
    compute_track_speed,
    find_fps,
    load_arena_ids,
    resolve_dimensions,
)
from .chaser_metrics_loader import (
    CanonicalOnlineCoordinateHandoff,
    ChaserMetricsBundle,
    load_canonical_online_coordinate_surface,
    load_chaser_metrics,
)
from fisheye.shared.archive_identity import ArchiveIdentity, archive_identity
from fisheye.shared.canonical_coordinate_publication import (
    BoundCanonicalCoordinateDescriptor,
    require_bound_canonical_coordinate_descriptor,
)
from fisheye.shared.coordinate_frame_record import (
    BoundPhysicalFrameCalibration,
    array_payload_sha256,
)
from fisheye.shared.coordinate_record import bind_persisted_coordinate_record
from fisheye.shared.coordinate_descriptor import COORDINATE_DESCRIPTOR_ATTR
from fisheye.shared.coordinate_identity import (
    TRACK_SAMPLE_INTERPOLATION_DTYPE,
    TRACK_SAMPLE_SOURCE_INSTANCE_KEY_DTYPE,
    TRACK_SAMPLE_DOMAIN,
    BoundSourceRowTemporalAuthority,
    build_row_identity_contract,
    build_track_sample_key,
    derive_track_source_instance_values,
    identity_array_content_sha256,
    load_bound_row_identity_contract,
    load_bound_source_row_temporal_authority,
    load_bound_track_sample_time_lineage,
    load_row_identity_contract_attrs,
    require_bound_source_row_temporal_authority,
    resolve_source_acquisition_frame_indices,
    stamp_and_bind_row_identity_contract,
    stamp_track_sample_time_lineage,
)
from fisheye.shared.pixel_frame_authority import (
    PixelFrameAuthorityError,
    load_persisted_acquisition_camera_authority,
)
from fisheye.shared.json_safety import json_attr_safe
from fisheye.shared.observation_coordinate_publication import (
    BoundSourceCameraPositionSurface,
    CROP_GEOMETRY_SELECTION_OPERATION,
    CROP_GEOMETRY_SELECTION_SCHEMA_ID,
    CROP_GEOMETRY_SELECTION_SCHEMA_VERSION,
    load_collection_proxy_successor_source_rowset,
    load_persisted_source_camera_position_surface,
    require_bound_source_camera_position_surface,
)
from fisheye.shared.stage_provenance import (
    build_stage_provenance,
    write_stage_provenance,
)
from fisheye.shared.run_provenance import (
    build_run_provenance_from_stage_record,
    sha256_payload,
    validate_run_provenance,
)
from fisheye.shared.run_lineage_fingerprint import write_best_effort_run_lineage_attrs
from fisheye.shared.rowset_fingerprint import (
    RowsetFingerprint,
    build_rowset_fingerprint,
    build_group_rowset_fingerprint,
    resolve_rowset_edit_revision,
)
from fisheye.shared.zarr.chunk_profiles import (
    geometry_preload_attrs,
    geometry_preload_chunks_for_shape,
    stamp_geometry_preload_attrs,
)
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
    RUN_STATUS_FAILED,
    mark_run_complete,
    mark_run_failed,
    mark_run_started,
    require_runs_parent,
)
from fisheye.tracking.single_subject_per_arena import load_tracking_ids
from fisheye.shared.system_metadata import get_git_info, get_environment_info
from fisheye.shared.track_coordinate_publication import (
    TRACK_POSITION_DERIVATION_ATTR,
    TrackPositionPublicationResult,
    load_track_position_coordinates,
    publish_track_position_coordinates,
)
from fisheye.shared.stimulus_physical_coordinate import (
    STIMULUS_PHYSICAL_COORDINATE_REASON_CODE_ATTR,
    STIMULUS_PHYSICAL_COORDINATE_STATUS_ATTR,
    BoundStimulusPhysicalCoordinateAuthority,
    StimulusPhysicalCoordinateUnavailableError,
    load_stimulus_physical_coordinate_authority,
    require_bound_stimulus_physical_coordinate_authority,
)
from fisheye.shared.source_camera_physical_authority import (
    BoundSourceCameraPhysicalAuthority,
    load_source_camera_physical_authority,
    require_bound_source_camera_physical_authority,
)
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.shared.zarr.columnar import load_structured_dataset
from .swim_bout_io import SwimBoutIOError, load_default_swim_bout_tables, load_swim_bout_tables


TrackPhysicalAuthority = (
    BoundStimulusPhysicalCoordinateAuthority
    | BoundSourceCameraPhysicalAuthority
)


SAMPLE_REASON_OK = 0
SAMPLE_REASON_UNASSIGNED = 1
SAMPLE_REASON_SOURCE_INTERPOLATED = 2
SAMPLE_REASON_SOURCE_MISSING = 3
SAMPLE_REASON_KEYPOINT_FAILED = 4
SAMPLE_REASON_HEADING_UNUSABLE = 5
SAMPLE_REASON_POSITION_NAN = 6
SAMPLE_REASON_MANUAL_REJECT = 7

SAMPLE_REASON_CODES = {
    str(SAMPLE_REASON_OK): "ok",
    str(SAMPLE_REASON_UNASSIGNED): "unassigned",
    str(SAMPLE_REASON_SOURCE_INTERPOLATED): "source_interpolated",
    str(SAMPLE_REASON_SOURCE_MISSING): "source_missing",
    str(SAMPLE_REASON_KEYPOINT_FAILED): "keypoint_failed",
    str(SAMPLE_REASON_HEADING_UNUSABLE): "heading_unusable",
    str(SAMPLE_REASON_POSITION_NAN): "position_nan",
    str(SAMPLE_REASON_MANUAL_REJECT): "manual_reject",
}

_UNKNOWN_SOURCE_VALUES = {"", "unknown", "none", "null"}

KEYPOINT_USABILITY_DATASET_CANDIDATES = (
    "heading_usable",
    "refined_success",
    "detection_success",
    "source_success",
)

DEFAULT_SMOOTH_SECONDS = 0.05
DEFAULT_HYSTERESIS_HIGH_PX = 4.0
DEFAULT_HYSTERESIS_LOW_PX = 2.0
DEFAULT_HYSTERESIS_MIN_FRAMES = 3
DEFAULT_HYSTERESIS_BAND_POLICY = "latch"
DEFAULT_SMOOTHING_ALIGNMENT = "causal"

TRACK_KINEMATICS_RUN_SCHEMA_ID = "analysis.track_kinematics_runs"
TRACK_KINEMATICS_RUN_SCHEMA_VERSION = 1
TRACK_KINEMATICS_METHOD_VERSION = "track_kinematics.v1"
TRACK_KINEMATICS_ROW_AXIS = "track_samples"
CANONICAL_OFFLINE_POSITION_SOURCE_KIND = (
    "canonical_crop_rows_source_camera_centers"
)
_REQUIRED_CANONICAL_OFFLINE_INPUT_KEYS = frozenset(
    {
        "detection_path",
        "position_source_path",
        "position_source_rowset_path",
        "position_source_kind",
        "crop_run",
        "keypoint_path",
        "tracking_path",
    }
)
_OPTIONAL_CANONICAL_OFFLINE_INPUT_KEYS = frozenset(
    {
        "chaser_metrics",
        "keypoint_source_crop_run",
        "swim_bout_run",
        "tracking_source_rowset_path",
    }
)
_CANONICAL_ONLINE_RAW_INPUT_KEYS = frozenset(
    {
        "stimulus_run",
        "chaser_index",
        "positions_px_source_path",
        "positions_px_coordinate_descriptor_sha256",
    }
)
_CANONICAL_ONLINE_REFINED_INPUT_KEYS = (
    _CANONICAL_ONLINE_RAW_INPUT_KEYS | {"refined_online_run"}
)
_CANONICAL_TRACK_BASE_PARAMETER_KEYS = frozenset(
    {
        "fps",
        "smoothing_seconds",
        "smoothing_method",
        "smoothing_alignment",
        "savgol_polyorder",
        "coordinate_space",
        "hysteresis_enabled",
        "hysteresis_high_px",
        "hysteresis_low_px",
        "hysteresis_min_frames",
        "hysteresis_band_policy",
    }
)
_CANONICAL_OFFLINE_PARAMETER_KEYS = (
    _CANONICAL_TRACK_BASE_PARAMETER_KEYS | {"distance_interpolation_seconds"}
)
_CANONICAL_ONLINE_PARAMETER_KEYS = _CANONICAL_TRACK_BASE_PARAMETER_KEYS
_CANONICAL_CHASER_METRICS_INPUT_KEYS = frozenset(
    {
        "metrics_run",
        "stimulus_run",
        "chaser_index",
        "distance_interpolation_seconds",
        "coordinate_geometry_status",
        "coordinate_geometry_reason_code",
        "omitted_coordinate_fields",
    }
)
_CANONICAL_CHASER_METRICS_REQUIRED_ARRAYS = frozenset(
    {
        "camera_frame_ids",
        "stimulus_frame_nums",
        "timestamp_ns",
        "trial_state",
        "has_offline",
    }
)
_CANONICAL_CHASER_METRICS_ARRAYS = (
    _CANONICAL_CHASER_METRICS_REQUIRED_ARRAYS
    | {
        "metadata_mask",
        "angle_unsigned_deg",
        "angle_signed_deg",
        "heading_deg",
    }
)
_CANONICAL_CHASER_OMITTED_COORDINATE_FIELDS = frozenset(
    {
        "distance_px",
        "distance_mm",
        "fish_centroid_px",
        "chaser_position_px",
    }
)

CAMERA_PIXEL_COORDINATE_SPACES = frozenset(
    {
        "camera",
        "source_camera_image_px",
    }
)
PROJECTOR_PIXEL_COORDINATE_SPACES = frozenset(
    {
        "texture",
        "stimulus_texture_px",
        "stimulus_canvas_px",
        "projector_px",
        "arena_relative_canvas_px",
    }
)

SPEED_DERIVATIVE_LEVELS = (
    "speed_raw",
    "speed_filtered",
    "speed_smoothed",
    "speed_averaged",
)
SPEED_DERIVATIVES_SCHEMA_ID = "palette.track_speed_derivatives.v1"
SPEED_DERIVATIVE_SCHEMA_ID = "palette.track_speed_derivative.v1"
DEFAULT_ACCELERATION_SOURCE_SPEED_LEVEL = "speed_smoothed"
TRACK_KINEMATICS_STAGING_MANIFEST_ATTR = "track_kinematics_staging_manifest"
TRACK_KINEMATICS_STAGING_MANIFEST_DIGEST_ATTR = (
    "track_kinematics_staging_manifest_sha256"
)
TRACK_KINEMATICS_STAGING_MANIFEST_SCHEMA_ID = (
    "palette.track_kinematics_staging_manifest.v3"
)
TRACK_KINEMATICS_STAGING_MANIFEST_SCHEMA_VERSION = 3
TRACK_KINEMATICS_COORDINATE_BINDING_STATUS_ATTR = "coordinate_binding_status"
TRACK_KINEMATICS_UNBOUND_STAGE_STATUS = "unbound_numeric_stage_complete_v1"
TRACK_KINEMATICS_PUBLISHING_BINDING_STATUS = "publishing_canonical_binding_v1"
TRACK_KINEMATICS_BOUND_CANONICAL_STATUS = "bound_canonical_v2"
TRACK_MOTION_PUBLICATION_MANIFEST_ATTR = (
    "track_motion_publication_manifest"
)
TRACK_MOTION_PUBLICATION_MANIFEST_DIGEST_ATTR = (
    "track_motion_publication_manifest_sha256"
)
TRACK_MOTION_PUBLICATION_MANIFEST_SCHEMA_ID = (
    "palette.track_motion_publication_manifest"
)
TRACK_MOTION_PUBLICATION_MANIFEST_SCHEMA_VERSION = 1
TRACK_MOTION_PUBLICATION_COMMIT_ATTR = "track_motion_publication_commit"
TRACK_MOTION_PUBLICATION_COMMIT_SCHEMA_ID = (
    "palette.track_motion_publication_commit"
)
TRACK_MOTION_PUBLICATION_COMMIT_SCHEMA_VERSION = 1
TRACK_MOTION_INPUT_AUTHORITY_ATTR = "track_motion_input_authority"
TRACK_MOTION_INPUT_AUTHORITY_SCHEMA_ID = "palette.track_motion_input_authority"
TRACK_MOTION_INPUT_AUTHORITY_SCHEMA_VERSION = 1
TRACK_MOTION_AXIS_TRACK_SAMPLE = "track_sample"
TRACK_MOTION_AXIS_TRACK_TRANSITION = "track_transition_destination_sample"
TRACK_MOTION_AXIS_TRACK_SECOND = "track_second_bin"
TRACK_MOTION_AXIS_TRACK_BOUT = "track_bout_event"
TRACK_MOTION_AXIS_RUN_TRACK = "run_track"
TRACK_MOTION_AXIS_RUN_CAMERA_SAMPLE = "run_camera_sample"
TRACK_KINEMATICS_PUBLICATION_OWNER_ATTR = (
    "track_kinematics_publication_owner_uuid"
)
TRACK_KINEMATICS_SELECTOR_OWNER_ATTR = (
    "track_kinematics_selector_publication_owner"
)
TRACK_KINEMATICS_PUBLICATION_TOMBSTONE_ATTR = (
    "track_kinematics_publication_tombstone"
)
_BOUND_TRACK_MOTION_SEAL = object()
_BOUND_TRACK_MOTION_INPUT_AUTHORITY_SEAL = object()


@dataclass(frozen=True)
class BoundTrackPositionBindings:
    """Fresh typed position bindings for one public track run.

    This object authorizes the row-bound ``positions_px`` and optional
    ``positions_mm`` surfaces only.  It deliberately does not authorize the
    run's derived speed, path, heading, acceleration, or time arrays; those
    require a separate exact payload/derivation seal before a scientific
    reader may expose them as canonical.
    """

    archive_identity: ArchiveIdentity
    run_type: str
    run_name: str
    source_positions: BoundCanonicalCoordinateDescriptor
    source_temporal_authority: BoundSourceRowTemporalAuthority
    physical_authority: TrackPhysicalAuthority | None
    track_positions: tuple[tuple[int, TrackPositionPublicationResult], ...]
    run_group: zarr.Group

    def position_for_track(self, track_id: int) -> TrackPositionPublicationResult:
        for candidate_id, binding in self.track_positions:
            if candidate_id == int(track_id):
                return binding
        raise KeyError(f"Track {track_id} is not present in /{self.run_group.path}.")


@dataclass(frozen=True, init=False)
class BoundTrackMotionInputAuthority:
    """Exact live-source evidence accepted by the future track writer."""

    archive_identity: ArchiveIdentity
    record: Mapping[str, Any]
    _seal: object

    def __init__(
        self,
        *,
        archive: ArchiveIdentity,
        record: Mapping[str, Any],
        _verification_seal: object | None = None,
    ) -> None:
        if _verification_seal is not _BOUND_TRACK_MOTION_INPUT_AUTHORITY_SEAL:
            raise ValueError(
                "Track-motion input authority must be minted from exact live arrays."
            )
        object.__setattr__(self, "archive_identity", archive)
        object.__setattr__(self, "record", _freeze_motion_manifest(record))
        object.__setattr__(self, "_seal", _verification_seal)


@dataclass(frozen=True, init=False)
class BoundTrackMotionSurface:
    """One exact live surface authorized by the full-motion manifest."""

    relative_path: str
    axis0_domain: str
    units: str
    semantic_profile: str
    operation_id: str
    input_refs: tuple[Mapping[str, Any], ...]
    alias_of: str | None
    dtype: str
    shape: tuple[int, ...]
    content_sha256: str
    node: Any
    _seal: object

    def __init__(
        self,
        *,
        relative_path: str,
        axis0_domain: str,
        units: str,
        semantic_profile: str,
        operation_id: str,
        input_refs: tuple[Mapping[str, Any], ...],
        alias_of: str | None,
        dtype: str,
        shape: tuple[int, ...],
        content_sha256: str,
        node: Any,
        _verification_seal: object | None = None,
    ) -> None:
        if _verification_seal is not _BOUND_TRACK_MOTION_SEAL:
            raise ValueError(
                "Bound track-motion surfaces must be minted by the live loader."
            )
        object.__setattr__(self, "relative_path", relative_path)
        object.__setattr__(self, "axis0_domain", axis0_domain)
        object.__setattr__(self, "units", units)
        object.__setattr__(self, "semantic_profile", semantic_profile)
        object.__setattr__(self, "operation_id", operation_id)
        object.__setattr__(
            self,
            "input_refs",
            tuple(_freeze_motion_manifest(value) for value in input_refs),
        )
        object.__setattr__(self, "alias_of", alias_of)
        object.__setattr__(self, "dtype", str(dtype))
        object.__setattr__(self, "shape", tuple(int(value) for value in shape))
        object.__setattr__(self, "content_sha256", str(content_sha256))
        object.__setattr__(self, "node", node)
        object.__setattr__(self, "_seal", _verification_seal)


@dataclass(frozen=True, init=False)
class BoundTrackMotionTrack:
    """Typed full-motion bindings for one track identity."""

    track_id: int
    position_binding: TrackPositionPublicationResult
    surfaces: tuple[BoundTrackMotionSurface, ...]
    track_group: zarr.Group
    _seal: object

    def __init__(
        self,
        *,
        track_id: int,
        position_binding: TrackPositionPublicationResult,
        surfaces: tuple[BoundTrackMotionSurface, ...],
        track_group: zarr.Group,
        _verification_seal: object | None = None,
    ) -> None:
        if _verification_seal is not _BOUND_TRACK_MOTION_SEAL:
            raise ValueError(
                "Bound track-motion tracks must be minted by the live loader."
            )
        object.__setattr__(self, "track_id", int(track_id))
        object.__setattr__(self, "position_binding", position_binding)
        object.__setattr__(self, "surfaces", tuple(surfaces))
        object.__setattr__(self, "track_group", track_group)
        object.__setattr__(self, "_seal", _verification_seal)

    def surface(self, relative_path: str) -> BoundTrackMotionSurface:
        for candidate in self.surfaces:
            if candidate.relative_path == str(relative_path):
                return candidate
        raise KeyError(
            f"Motion surface {relative_path!r} is not present in "
            f"/{self.track_group.path}."
        )


@dataclass(frozen=True, init=False)
class BoundTrackMotionRun:
    """Fresh authority for every sealed public derived-motion surface."""

    position_bindings: BoundTrackPositionBindings
    manifest_sha256: str
    manifest: Mapping[str, Any]
    tracks: tuple[BoundTrackMotionTrack, ...]
    run_group: zarr.Group
    _authoritative_root: zarr.Group
    _expected_selector_eligible: bool
    _seal: object

    def __init__(
        self,
        *,
        position_bindings: BoundTrackPositionBindings,
        manifest_sha256: str,
        manifest: Mapping[str, Any],
        tracks: tuple[BoundTrackMotionTrack, ...],
        run_group: zarr.Group,
        authoritative_root: zarr.Group,
        expected_selector_eligible: bool,
        _verification_seal: object | None = None,
    ) -> None:
        if _verification_seal is not _BOUND_TRACK_MOTION_SEAL:
            raise ValueError(
                "Bound track-motion runs must be minted by the live loader."
            )
        object.__setattr__(self, "position_bindings", position_bindings)
        object.__setattr__(self, "manifest_sha256", str(manifest_sha256))
        object.__setattr__(self, "manifest", _freeze_motion_manifest(manifest))
        object.__setattr__(self, "tracks", tuple(tracks))
        object.__setattr__(self, "run_group", run_group)
        object.__setattr__(self, "_authoritative_root", authoritative_root)
        object.__setattr__(
            self,
            "_expected_selector_eligible",
            bool(expected_selector_eligible),
        )
        object.__setattr__(self, "_seal", _verification_seal)

    def track(self, track_id: int) -> BoundTrackMotionTrack:
        for candidate in self.tracks:
            if candidate.track_id == int(track_id):
                return candidate
        raise KeyError(
            f"Track {track_id} is not present in /{self.run_group.path}."
        )

    def assert_verified(self) -> None:
        if getattr(self, "_seal", None) is not _BOUND_TRACK_MOTION_SEAL:
            raise ValueError("Track full-motion authority is not loader-sealed.")
        current = _load_bound_track_motion_run_impl(
            self._authoritative_root,
            self.run_group,
            expected_selector_eligible=self._expected_selector_eligible,
        )
        if (
            current.manifest_sha256 != self.manifest_sha256
            or current.run_group.path != self.run_group.path
            or current.position_bindings.archive_identity
            != self.position_bindings.archive_identity
        ):
            raise ValueError(
                "Track full-motion authority changed after it was loaded."
            )


def resolve_mm_per_pixel_for_coordinate_space(
    coordinate_space: object,
    *,
    camera_mm_per_pixel: object = None,
    pixels_per_mm_projector: object = None,
) -> float:
    """Legacy compatibility calculation; never track-publication authority.

    Canonical writers must instead use a freshly rebound
    ``BoundStimulusPhysicalCoordinateAuthority``.  This calculation remains
    isolated for explicit legacy readers/tests and is not called by the writer.
    """

    if not isinstance(coordinate_space, str) or not coordinate_space.strip():
        raise ValueError("A declared coordinate_space is required for px-to-mm conversion.")
    space = coordinate_space.strip()

    if space in CAMERA_PIXEL_COORDINATE_SPACES:
        scale_name = "camera_mm_per_pixel"
        scale_value = camera_mm_per_pixel
        invert = False
    elif space in PROJECTOR_PIXEL_COORDINATE_SPACES:
        scale_name = "pixels_per_mm_projector"
        scale_value = pixels_per_mm_projector
        invert = True
    else:
        raise ValueError(
            f"Unsupported coordinate_space {space!r} for px-to-mm conversion."
        )

    try:
        scale = float(scale_value)
    except (TypeError, ValueError):
        scale = float("nan")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(
            f"coordinate_space {space!r} requires a positive finite {scale_name}."
        )
    return 1.0 / scale if invert else scale


def _track_preload_chunks(shape: Tuple[int, ...] | Iterable[int]) -> Tuple[int, ...] | None:
    return geometry_preload_chunks_for_shape(tuple(int(dim) for dim in shape))


def _stamp_geometry_preload_tree(group: zarr.Group) -> None:
    stamp_geometry_preload_attrs(group)
    for name in list(group.array_keys()):
        stamp_geometry_preload_attrs(group[name])
    for name in list(group.group_keys()):
        _stamp_geometry_preload_tree(group[name])


MOVEMENT_SCHEMA_ID = "palette.track_movement.v2"
MOVEMENT_SPEED_SCHEMA_ID = "palette.track_movement_speed.v2"
MOVEMENT_SPEED_LEVEL_SCHEMA_ID = "palette.track_movement_speed_level.v2"
MOVEMENT_SPEED_LEVEL_NAMES = {
    "speed_raw": "raw",
    "speed_filtered": "filtered",
    "speed_smoothed": "smoothed",
    "speed_averaged": "averaged",
}


def _controlled_two_component_run_path(
    value: Any,
    *,
    families: frozenset[str],
    label: str,
) -> str:
    if not isinstance(value, str) or value != value.strip().strip("/"):
        raise ValueError(f"{label} must be one canonical archive-relative path.")
    parts = value.split("/")
    if (
        len(parts) != 2
        or parts[0] not in families
        or parts[1] in {"", ".", ".."}
        or parts[1].strip() != parts[1]
    ):
        raise ValueError(
            f"{label} must have one controlled run-family prefix and one leaf."
        )
    return value


def _controlled_run_leaf(value: Any, *, label: str) -> str:
    """Return one canonical run-name leaf without path interpretation."""

    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "/" in value
        or value in {".", ".."}
    ):
        raise ValueError(f"{label} must be one canonical run-name leaf.")
    return value


def _sha256_text(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{label} must be one lowercase SHA-256 digest.")
    return value


def _track_kinematics_source_refs(
    *,
    run_type: str,
    inputs: Mapping[str, Any],
) -> Dict[str, Any]:
    """Normalize exact archive-relative dependencies for one run."""

    refs: Dict[str, Any] = {}

    if run_type == "online":
        refined_run = inputs.get("refined_online_run")
        if refined_run is not None:
            refs["source_refined_online_path"] = (
                "refined_online_runs/"
                + _controlled_run_leaf(
                    refined_run,
                    label="refined_online_run",
                )
            )
        stimulus_run = inputs.get("stimulus_run")
        if stimulus_run is not None:
            refs["source_stimulus_path"] = (
                "analysis/stimulus_runs/"
                + _controlled_run_leaf(stimulus_run, label="stimulus_run")
            )
        positions_path = inputs.get("positions_px_source_path")
        if positions_path not in (None, ""):
            if (
                not isinstance(positions_path, str)
                or positions_path != positions_path.strip().strip("/")
                or any(
                    part in {"", ".", ".."}
                    for part in positions_path.split("/")
                )
            ):
                raise ValueError(
                    "positions_px_source_path must be one canonical archive-relative path."
                )
            refs["source_positions_px_path"] = positions_path
        descriptor_digest = inputs.get(
            "positions_px_coordinate_descriptor_sha256"
        )
        if descriptor_digest not in (None, ""):
            refs["source_positions_px_coordinate_descriptor_sha256"] = (
                _sha256_text(
                    descriptor_digest,
                    label="positions_px_coordinate_descriptor_sha256",
                )
            )
        if inputs.get("chaser_index") is not None:
            chaser_index = inputs["chaser_index"]
            if (
                isinstance(chaser_index, (bool, np.bool_))
                or not isinstance(chaser_index, (int, np.integer))
                or int(chaser_index) < 0
            ):
                raise ValueError("chaser_index must be one nonnegative integer.")
            refs["source_chaser_index"] = int(chaser_index)
        return refs

    for source_key in (
        "detection_path",
        "position_source_path",
        "position_source_rowset_path",
    ):
        value = inputs.get(source_key)
        if value not in (None, ""):
            refs[f"source_{source_key}"] = str(value)
    position_source_kind = inputs.get("position_source_kind")
    if position_source_kind not in (None, ""):
        refs["source_position_source_kind"] = str(position_source_kind)

    keypoint_path = inputs.get("keypoint_path")
    if keypoint_path not in (None, ""):
        refs["source_keypoint_path"] = _controlled_two_component_run_path(
            keypoint_path,
            families=frozenset({"keypoints_runs", "refined_keypoints_runs"}),
            label="keypoint_path",
        )
    crop_run = inputs.get("crop_run")
    if crop_run is not None:
        refs["source_crop_path"] = (
            "crop_runs/" + _controlled_run_leaf(crop_run, label="crop_run")
        )
    keypoint_source_crop_run = inputs.get("keypoint_source_crop_run")
    tracking_source_rowset_path = inputs.get("tracking_source_rowset_path")
    if (keypoint_source_crop_run is None) != (
        tracking_source_rowset_path is None
    ):
        raise ValueError(
            "keypoint_source_crop_run and tracking_source_rowset_path must be "
            "persisted together."
        )
    if keypoint_source_crop_run is not None:
        keypoint_source_path = "crop_runs/" + _controlled_run_leaf(
            keypoint_source_crop_run,
            label="keypoint_source_crop_run",
        )
        tracking_source_path = _controlled_two_component_run_path(
            tracking_source_rowset_path,
            families=frozenset({"crop_runs"}),
            label="tracking_source_rowset_path",
        )
        if tracking_source_path != keypoint_source_path:
            raise ValueError(
                "Keypoint and tracking lineage must identify the same exact source "
                "rowset."
            )
        refs["source_keypoint_crop_path"] = keypoint_source_path
        refs["source_tracking_rowset_path"] = tracking_source_path
    tracking_path = inputs.get("tracking_path")
    if tracking_path not in (None, ""):
        refs["source_tracking_path"] = _controlled_two_component_run_path(
            tracking_path,
            families=frozenset({"tracking_runs"}),
            label="tracking_path",
        )
    swim_bout_run = inputs.get("swim_bout_run")
    if swim_bout_run is not None:
        refs["source_swim_bout_path"] = (
            "analysis/swim_bout_runs/"
            + _controlled_run_leaf(swim_bout_run, label="swim_bout_run")
        )
    return refs


def _track_kinematics_contract_attrs(
    *,
    run_type: str,
    method: str,
    parameters: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> Dict[str, Any]:
    """Return the shared derived-run contract attrs for track kinematics."""

    return {
        "schema_id": TRACK_KINEMATICS_RUN_SCHEMA_ID,
        "schema_version": TRACK_KINEMATICS_RUN_SCHEMA_VERSION,
        "method": method,
        "method_version": TRACK_KINEMATICS_METHOD_VERSION,
        "row_axis": TRACK_KINEMATICS_ROW_AXIS,
        "parameters": dict(parameters),
        "source_refs": _track_kinematics_source_refs(
            run_type=run_type,
            inputs=inputs,
        ),
    }


@dataclass
class KeypointResolution:
    """Resolved keypoint run metadata."""

    group: zarr.Group
    run_name: str
    is_refined: bool
    base_run_name: str
    crop_run: str


@dataclass(frozen=True)
class CollectionProxySuccessorTrackingResolution:
    """Original tracking authority for an exact current-coordinate successor."""

    position_crop_run: str
    historical_source_rowset_path: str
    expected_detect_run: str
    expected_source_rowset_fingerprint: RowsetFingerprint


@dataclass
class DetectionResolution:
    """Resolved detection group metadata."""

    group: zarr.Group
    path: str
    is_refined: bool
    run_name: str
    variant: str
    source_detect_run: Optional[str]
    parent_path: str


@dataclass
class OfflinePositionSource:
    """Row-aligned position arrays for offline track kinematics."""

    positions_px: np.ndarray
    frame_indices: np.ndarray
    detection_source: Optional[np.ndarray]
    path: str
    kind: str
    geometry_path: str
    instance_key: Optional[np.ndarray]
    rowset_fingerprint: RowsetFingerprint
    rowset_group: Any
    position_surface: BoundSourceCameraPositionSurface | None = None


def resolve_keypoint_group(
    root: zarr.Group,
    requested: Optional[str],
    console: Console,
) -> KeypointResolution:
    """Resolve the keypoint group (preferring refined runs)."""

    refined_parent = root.get("refined_keypoints_runs")
    raw_parent = root.get("keypoints_runs")

    if raw_parent is None and refined_parent is None:
        raise ValueError("No keypoint runs found in archive.")

    def resolve_raw(name: str) -> KeypointResolution:
        if raw_parent is None or name not in raw_parent:
            raise ValueError(f"Keypoint run '{name}' not found in keypoints_runs.")
        group = raw_parent[name]
        crop_run = group.attrs.get("source_crop_run")
        if not crop_run:
            raise ValueError(
                f"Keypoint run '{name}' missing 'source_crop_run' attribute; cannot resolve detection source."
            )
        return KeypointResolution(group=group, run_name=name, is_refined=False, base_run_name=name, crop_run=crop_run)

    def resolve_refined(name: str) -> KeypointResolution:
        if refined_parent is None or name not in refined_parent:
            raise ValueError(f"Refined keypoint run '{name}' not found.")
        group = refined_parent[name]
        base_run = group.attrs.get("source_keypoints_run")
        if not base_run:
            raise ValueError(
                f"Refined keypoint run '{name}' missing 'source_keypoints_run' attribute; provenance is required."
            )
        base_resolution = resolve_raw(base_run)
        return KeypointResolution(
            group=group,
            run_name=name,
            is_refined=True,
            base_run_name=base_resolution.base_run_name,
            crop_run=base_resolution.crop_run,
        )

    if requested:
        if requested.startswith("refined/"):
            return resolve_refined(requested.split("/", 1)[1])
        if refined_parent is not None and requested in refined_parent:
            return resolve_refined(requested)
        return resolve_raw(requested)

    if refined_parent is not None:
        latest_refined = refined_parent.attrs.get("latest")
        if latest_refined:
            console.print(
                f"Using refined keypoints run: [cyan]{latest_refined}[/cyan]"
            )
            return resolve_refined(latest_refined)

    if raw_parent is not None:
        latest_raw = raw_parent.attrs.get("latest")
        if latest_raw:
            console.print(f"Using keypoints run: [cyan]{latest_raw}[/cyan]")
            return resolve_raw(latest_raw)

    raise ValueError("Unable to resolve a keypoint run; no runs detected.")


def resolve_collection_proxy_successor_tracking(
    root: zarr.Group,
    *,
    keypoints: KeypointResolution,
    position_crop_run: str,
) -> CollectionProxySuccessorTrackingResolution:
    """Bind successor positions to their exact historical tracking authority.

    Keypoints and tracking remain immutable on the historical merged-proxy
    rowset. A successor may reuse those arrays only when its sealed mapping
    proves an exact all-row copy of that same rowset. Tracking IDs are still
    aligned by the persisted unique ``instance_key`` set downstream.
    """

    successor_name = _controlled_run_leaf(
        position_crop_run,
        label="position_source_run",
    )
    successor_path = f"crop_runs/{successor_name}"
    historical_path = load_collection_proxy_successor_source_rowset(
        root,
        successor_path,
    )
    expected_historical_path = f"crop_runs/{keypoints.crop_run}"
    if historical_path != expected_historical_path:
        raise ValueError(
            "Selected coordinate successor does not prove exact identity with "
            "the keypoint source rowset: "
            f"successor_source={historical_path!r}, "
            f"keypoint_source={expected_historical_path!r}."
        )
    historical_group = root[historical_path]
    revision = resolve_rowset_edit_revision(historical_group.attrs)
    fingerprint = build_group_rowset_fingerprint(
        historical_group,
        source_rowset_path=historical_path,
        source_edit_revision=revision,
    )
    base_group = root[f"keypoints_runs/{keypoints.base_run_name}"]
    candidates = {
        str(value).strip()
        for value in (
            keypoints.group.attrs.get("source_detect_run"),
            base_group.attrs.get("source_detect_run"),
        )
        if isinstance(value, str) and str(value).strip()
    }
    if len(candidates) != 1:
        raise ValueError(
            "Keypoint lineage does not identify one exact source_detect_run for "
            "successor tracking resolution."
        )
    return CollectionProxySuccessorTrackingResolution(
        position_crop_run=successor_name,
        historical_source_rowset_path=historical_path,
        expected_detect_run=next(iter(candidates)),
        expected_source_rowset_fingerprint=fingerprint,
    )


def load_keypoint_usability_array(
    group: zarr.Group,
    expected_length: int,
) -> Tuple[np.ndarray, str]:
    """Return the best row-level keypoint/heading usability array available."""

    for dataset_name in KEYPOINT_USABILITY_DATASET_CANDIDATES:
        if dataset_name not in group:
            continue
        values = np.asarray(group[dataset_name][:], dtype=bool)
        if values.shape[0] != expected_length:
            raise ValueError(
                f"Keypoint usability dataset '{dataset_name}' length "
                f"{values.shape[0]} does not match expected row count {expected_length}."
            )
        return values, dataset_name

    return np.ones(expected_length, dtype=bool), "implicit_all_true"


def resolve_detection_from_path(root: zarr.Group, path: str) -> DetectionResolution:
    """Resolve detection metadata from the crop-provided path."""

    if path not in root:
        # Handle legacy references written before refined_detect_runs rename.
        if path.startswith("refined_runs/"):
            tail = path[len("refined_runs/") :]
            legacy_candidates = [
                f"refined_detect_runs/{tail}",
            ]
            if tail.startswith("refined_"):
                # e.g. refined_runs/refined_2023-... -> refined_detect_runs/refined_detect_2023-...
                suffix = tail[len("refined_") :]
                legacy_candidates.append(f"refined_detect_runs/refined_detect_{suffix}")
            for candidate in legacy_candidates:
                if candidate in root:
                    return resolve_detection_from_path(root, candidate)
        raise ValueError(f"Detection group '{path}' referenced by crop run is missing.")

    group = root[path]
    parts = path.split("/")
    if not parts:
        raise ValueError(f"Invalid detection path '{path}'.")

    head = parts[0]
    if head == "refined_detect_runs":
        if len(parts) < 2:
            raise ValueError(f"Malformed refined detection path '{path}'.")
        run_name = parts[1]
        variant = parts[2] if len(parts) > 2 else "interpolated"
        parent_path = "/".join(parts[:2])
        parent_group = root[parent_path]
        source_detect_run = parent_group.attrs.get("source_detect_run")
        return DetectionResolution(
            group=group,
            path=path,
            is_refined=True,
            run_name=run_name,
            variant=variant,
            source_detect_run=source_detect_run,
            parent_path=parent_path,
        )
    if head == "refined_runs":  # legacy refined path fallback
        if len(parts) < 2:
            raise ValueError(f"Malformed legacy refined detection path '{path}'.")
        run_name = parts[1]
        variant = parts[2] if len(parts) > 2 else "interpolated"
        parent_path = "/".join(parts[:2])
        parent_group = root[parent_path]
        source_detect_run = parent_group.attrs.get("source_detect_run")
        return DetectionResolution(
            group=group,
            path=path,
            is_refined=True,
            run_name=run_name,
            variant=variant,
            source_detect_run=source_detect_run,
            parent_path=parent_path,
        )
    if head == "detect_runs":
        if len(parts) < 2:
            raise ValueError(f"Malformed detection path '{path}'.")
        run_name = parts[1]
        return DetectionResolution(
            group=group,
            path=path,
            is_refined=False,
            run_name=run_name,
            variant="raw",
            source_detect_run=run_name,
            parent_path="/".join(parts[:2]),
        )

    raise ValueError(
        "Unsupported detection path '{path}'. Expected detect_runs/ or refined_detect_runs/.".format(path=path)
    )


def _clean_source_text(value: Any) -> Optional[str]:
    if isinstance(value, bytes):
        value = value.decode("utf-8", "ignore")
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in _UNKNOWN_SOURCE_VALUES:
        return None
    return text


def _crop_row_source_label(attrs: Mapping[str, Any]) -> Optional[str]:
    """Return the source label used by tracking for row-aligned crop metadata."""

    direct = _clean_source_text(attrs.get("source_detect_run"))
    if direct:
        return direct

    source_type = _clean_source_text(
        attrs.get("detection_source_type") or attrs.get("source_type")
    )
    if source_type and (
        source_type.startswith("external_crop_recorder")
        or source_type.startswith("finalized_clipped_refined_detect_collection")
    ):
        return source_type
    return None


def _sorted_group_keys(group: Optional[zarr.Group]) -> List[str]:
    if group is None:
        return []
    keys_fn = getattr(group, "group_keys", None)
    try:
        keys = list(keys_fn()) if callable(keys_fn) else []
    except Exception:
        keys = []
    return sorted(key for key in keys if isinstance(key, str))


def prefer_refined_detection(
    root: zarr.Group, detection: DetectionResolution, console: Console
) -> DetectionResolution:
    """Prefer refined detection data when available for the same source run."""
    if detection.is_refined:
        return detection

    refined_parent = root.get("refined_detect_runs")
    if not isinstance(refined_parent, zarr.Group):
        return detection

    candidates: List[str] = []
    for run_name in _sorted_group_keys(refined_parent):
        run_group = refined_parent[run_name]
        if run_group.attrs.get("source_detect_run") == detection.run_name:
            candidates.append(run_name)

    if not candidates:
        return detection

    latest = refined_parent.attrs.get("latest")
    if latest in candidates:
        chosen = latest
    else:
        chosen = candidates[-1]

    target_group = refined_parent[chosen]
    variant_path = "interpolated" if "interpolated" in target_group else None
    refined_path = (
        f"refined_detect_runs/{chosen}/{variant_path}"
        if variant_path
        else f"refined_detect_runs/{chosen}"
    )

    console.print(
        f"[cyan]Preferring refined detections:[/cyan] {refined_path} "
        f"(source_detect_run={detection.run_name})"
    )

    return resolve_detection_from_path(root, refined_path)


def load_offline_position_source(
    crop_group: zarr.Group,
    *,
    crop_run_name: str,
    detection: Optional[DetectionResolution],
    root: Optional[zarr.Group] = None,
) -> OfflinePositionSource:
    """Load exact source-image centres for the selected offline rowset.

    Normalized boxes plus a root-level width/height are not an authority.  A
    canonical track writer accepts only source-image ``bbox_img_xyxy`` values
    already aligned to the crop rows, or an exact ``instance_key`` join to the
    selected detection rowset that owns those boxes.  Historical normalized-
    only archives must be handled by explicit audit/migration tooling.
    """

    def _bbox_centres(
        group: zarr.Group,
        *,
        group_path: str,
    ) -> tuple[np.ndarray, str]:
        node = group.get("bbox_img_xyxy")
        if node is None:
            raise ValueError(
                f"{group_path} lacks authoritative bbox_img_xyxy; refusing to "
                "reconstruct camera coordinates from normalized values or root "
                "dimensions. Run the coordinate audit/migration workflow."
            )
        boxes = np.asarray(node[:], dtype=np.float64)
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError(
                f"{group_path}/bbox_img_xyxy must have shape (N, 4); got "
                f"{tuple(int(value) for value in boxes.shape)}."
            )
        centres = np.column_stack(
            ((boxes[:, 0] + boxes[:, 2]) * 0.5, (boxes[:, 1] + boxes[:, 3]) * 0.5)
        )
        return centres, f"{group_path}/bbox_img_xyxy"

    if "frame_indices" in crop_group and (
        "bbox_img_xyxy" in crop_group or "bbox_norm_coords" in crop_group
    ):
        frame_indices = np.asarray(crop_group["frame_indices"][:], dtype=np.int64)
        path = f"crop_runs/{crop_run_name}"
        crop_instance_key = (
            np.asarray(crop_group["instance_key"][:], dtype=np.uint64).reshape(-1)
            if "instance_key" in crop_group
            else None
        )

        if "bbox_img_xyxy" in crop_group:
            positions_px, geometry_path = _bbox_centres(
                crop_group,
                group_path=path,
            )
        else:
            if detection is None or crop_instance_key is None:
                raise ValueError(
                    f"{path} is normalized-only and lacks the exact detection "
                    "instance-key join needed to resolve bbox_img_xyxy; refusing "
                    "root-dimension reconstruction."
                )
            detection_group = detection.group
            if "instance_key" not in detection_group:
                raise ValueError(
                    f"{detection.path} lacks instance_key required to align its "
                    f"bbox_img_xyxy to {path}."
                )
            detection_centres, geometry_path = _bbox_centres(
                detection_group,
                group_path=detection.path,
            )
            detection_keys = np.asarray(
                detection_group["instance_key"][:],
                dtype=np.uint64,
            ).reshape(-1)
            if detection_keys.shape[0] != detection_centres.shape[0]:
                raise ValueError(
                    f"{detection.path} instance_key/bbox_img_xyxy row count mismatch."
                )
            if np.unique(detection_keys).shape[0] != detection_keys.shape[0]:
                raise ValueError(f"{detection.path} contains duplicate instance_key values.")
            row_by_key = {int(key): index for index, key in enumerate(detection_keys)}
            missing = sorted(
                int(key) for key in crop_instance_key if int(key) not in row_by_key
            )
            if missing:
                raise ValueError(
                    f"{path} instance_key values are absent from {detection.path}: "
                    f"{missing!r}."
                )
            positions_px = detection_centres[
                np.asarray([row_by_key[int(key)] for key in crop_instance_key])
            ]

        if int(frame_indices.shape[0]) != int(positions_px.shape[0]):
            raise ValueError(
                f"crop_runs/{crop_run_name} row count mismatch: frame_indices has "
                f"{int(frame_indices.shape[0])} rows but authoritative source-image "
                f"positions have {int(positions_px.shape[0])} rows."
            )
        if crop_instance_key is not None and crop_instance_key.shape[0] != frame_indices.shape[0]:
            raise ValueError(
                f"{path} instance_key length does not match frame_indices."
            )
        revision = resolve_rowset_edit_revision(crop_group.attrs)
        fingerprint = build_group_rowset_fingerprint(
            crop_group,
            source_rowset_path=path,
            source_edit_revision=revision,
        )
        return OfflinePositionSource(
            positions_px=positions_px,
            frame_indices=frame_indices,
            detection_source=(
                crop_group["detection_source"][:]
                if "detection_source" in crop_group
                else None
            ),
            path=path,
            kind="crop_rows_source_image_bbox",
            geometry_path=geometry_path,
            instance_key=crop_instance_key,
            rowset_fingerprint=fingerprint,
            rowset_group=crop_group,
        )

    if detection is None:
        raise ValueError(
            f"Crop run '{crop_run_name}' missing row-aligned geometry/frame_indices "
            "and no source_coords_path was available."
        )

    detection_group = detection.group
    revision_sources: list[Mapping[str, Any]] = [detection_group.attrs]
    if root is not None and detection.is_refined and detection.parent_path in root:
        revision_sources.append(root[detection.parent_path].attrs)
    revision = resolve_rowset_edit_revision(*revision_sources)
    fingerprint = build_group_rowset_fingerprint(
        detection_group,
        source_rowset_path=detection.path,
        source_edit_revision=revision,
    )
    positions_px, geometry_path = _bbox_centres(
        detection_group,
        group_path=detection.path,
    )
    frame_indices = np.asarray(
        detection_group["frame_indices"][:],
        dtype=np.int64,
    )
    if frame_indices.shape[0] != positions_px.shape[0]:
        raise ValueError(
            f"{detection.path} frame_indices/bbox_img_xyxy row count mismatch."
        )
    return OfflinePositionSource(
        positions_px=positions_px,
        frame_indices=frame_indices,
        detection_source=(
            detection_group["detection_source"][:]
            if "detection_source" in detection_group
            else None
        ),
        path=detection.path,
        kind="detection_rows_source_image_bbox",
        geometry_path=geometry_path,
        instance_key=(
            np.asarray(detection_group["instance_key"][:], dtype=np.uint64).reshape(-1)
            if "instance_key" in detection_group
            else None
        ),
        rowset_fingerprint=fingerprint,
        rowset_group=detection_group,
    )


def load_canonical_offline_position_source(
    root: zarr.Group,
    crop_group: zarr.Group,
    *,
    crop_run_name: str,
) -> OfflinePositionSource:
    """Load the exact persisted crop-center surface used by future track writers.

    This path has no legacy inference branch.  The crop producer must already
    have published canonical source-camera centers, observation identity,
    acquisition-frame mapping, and exact detection/crop lineage.  Historical
    bbox-only or normalized-only rows remain audit/migration inputs and cannot
    enter normal track publication.
    """

    path = f"crop_runs/{crop_run_name}"
    surface = require_bound_source_camera_position_surface(
        load_persisted_source_camera_position_surface(root, path)
    )
    coordinate_node = surface.coordinates.coordinate_node
    if (
        archive_identity(root) != archive_identity(coordinate_node)
        or archive_identity(crop_group) != archive_identity(coordinate_node)
        or getattr(crop_group, "path", None) != path
        or getattr(coordinate_node, "path", None) != f"{path}/centers_img_xy"
        or surface.coordinates.row_identity.rowset_path != path
        or surface.temporal_authority.record.source_rowset_ref != f"/{path}"
    ):
        raise ValueError(
            "Canonical offline position evidence does not bind the exact selected "
            f"crop rowset /{path}."
        )
    positions_px = np.array(coordinate_node[:], copy=True, order="C")
    if (
        positions_px.shape != coordinate_node.shape
        or positions_px.dtype != np.dtype(coordinate_node.dtype)
        or positions_px.ndim != 2
        or positions_px.shape[1] != 2
        or not np.issubdtype(positions_px.dtype, np.number)
    ):
        raise ValueError(
            f"/{coordinate_node.path} must be one exact numeric (N, 2) canonical "
            "source-camera center payload."
        )
    frame_node = crop_group["source_acquisition_frame_index"]
    frame_indices = np.array(frame_node[:], copy=True, order="C")
    instance_node = crop_group["instance_key"]
    instance_key = np.array(instance_node[:], copy=True, order="C")
    row_count = int(positions_px.shape[0])
    if (
        archive_identity(frame_node) != archive_identity(coordinate_node)
        or archive_identity(instance_node) != archive_identity(coordinate_node)
        or getattr(frame_node, "path", None)
        != f"{path}/source_acquisition_frame_index"
        or getattr(instance_node, "path", None) != f"{path}/instance_key"
        or identity_array_content_sha256(frame_indices)
        != surface.temporal_authority.record.source_acquisition_frame_index.content_sha256
        or identity_array_content_sha256(instance_key)
        != surface.coordinates.row_identity.contract.key_array.content_sha256
        or frame_indices.dtype != np.dtype("<i8")
        or frame_indices.shape != (row_count,)
        or instance_key.dtype != np.dtype("<u8")
        or instance_key.shape != (row_count,)
    ):
        raise ValueError(
            f"/{path} canonical position, acquisition-frame, and instance-key "
            "surfaces are not exactly row aligned."
        )
    revision = resolve_rowset_edit_revision(crop_group.attrs)
    fingerprint = build_rowset_fingerprint(
        source_rowset_path=path,
        row_count=row_count,
        instance_keys=instance_key,
        source_edit_revision=revision,
    )
    get_child = getattr(crop_group, "get", None)
    detection_source_node = (
        get_child("detection_source") if callable(get_child) else None
    )
    return OfflinePositionSource(
        positions_px=positions_px,
        frame_indices=frame_indices,
        detection_source=(
            np.array(detection_source_node[:], copy=True, order="C")
            if detection_source_node is not None
            else None
        ),
        path=path,
        kind="canonical_crop_rows_source_camera_centers",
        geometry_path=coordinate_node.path,
        instance_key=instance_key,
        rowset_fingerprint=fingerprint,
        rowset_group=crop_group,
        position_surface=surface,
    )


def _canonical_crop_detection_rowset_path(
    coordinates: BoundCanonicalCoordinateDescriptor,
) -> str:
    """Return the exact detection rowset selected by one canonical crop.

    The crop publication record is already digest-bound by the canonical
    coordinate descriptor.  Future track publication projects that exact
    authority instead of reconstructing detection lineage from crop attrs or
    independently selected detection runs.
    """

    coordinates = require_bound_canonical_coordinate_descriptor(coordinates)
    matches: list[str] = []
    expected_leaves = (
        "instance_key",
        "source_acquisition_frame_index",
        "bbox_norm_coords",
        "bbox_img_xyxy",
        "centers_img_xy",
    )
    for bound_record in coordinates.lineage_records:
        bound_record.assert_verified()
        record = bound_record.record
        if record.get("schema_id") != CROP_GEOMETRY_SELECTION_SCHEMA_ID:
            continue
        if (
            record.get("schema_version")
            != CROP_GEOMETRY_SELECTION_SCHEMA_VERSION
            or record.get("operation") != CROP_GEOMETRY_SELECTION_OPERATION
        ):
            raise ValueError(
                "Canonical crop position lineage uses an unsupported selection "
                "record version or operation."
            )
        source_rowset = record.get("source_rowset")
        if type(source_rowset) is not dict:
            raise ValueError(
                "Canonical crop selection lacks one exact source rowset record."
            )
        rowset_path: str | None = None
        for leaf in expected_leaves:
            payload = source_rowset.get(leaf)
            array_ref = payload.get("array_ref") if type(payload) is dict else None
            suffix = f"/{leaf}"
            if (
                not isinstance(array_ref, str)
                or not array_ref.startswith("/")
                or not array_ref.endswith(suffix)
            ):
                raise ValueError(
                    "Canonical crop selection source-rowset array references are "
                    "missing or malformed."
                )
            candidate = array_ref[1 : -len(suffix)]
            if not candidate or (rowset_path is not None and candidate != rowset_path):
                raise ValueError(
                    "Canonical crop selection source arrays do not share one exact "
                    "detection rowset."
                )
            rowset_path = candidate
        assert rowset_path is not None
        if not (
            rowset_path.startswith("detect_runs/")
            or rowset_path.startswith("refined_detect_runs/")
        ):
            raise ValueError(
                "Canonical crop selection does not identify a supported future "
                "detection rowset."
            )
        matches.append(rowset_path)
    if len(matches) != 1:
        raise ValueError(
            "Canonical crop position lineage must contain exactly one bound crop "
            "geometry selection record."
        )
    return matches[0]


def _offline_position_source_inputs(
    source: OfflinePositionSource,
) -> dict[str, Any]:
    """Persist the exact coordinate array separately from its owning rowset."""

    surface = require_bound_source_camera_position_surface(source.position_surface)
    coordinate_path = str(surface.coordinates.coordinate_node.path).strip("/")
    rowset_path = coordinate_path.rsplit("/", 1)[0]
    if (
        source.geometry_path != coordinate_path
        or source.path != rowset_path
        or source.kind != CANONICAL_OFFLINE_POSITION_SOURCE_KIND
        or str(getattr(source.rowset_group, "path", "")).strip("/")
        != rowset_path
        or archive_identity(source.rowset_group)
        != archive_identity(surface.coordinates.coordinate_node)
    ):
        raise ValueError(
            "Offline position-source metadata conflicts with the exact canonical "
            "coordinate array or its owning rowset."
        )
    return {
        "position_source_path": coordinate_path,
        "position_source_rowset_path": rowset_path,
        "position_source_kind": CANONICAL_OFFLINE_POSITION_SOURCE_KIND,
    }


def load_track_source_temporal_authority(
    source_rowset: Any,
    *,
    acquisition_frame: Any,
) -> Any:
    """Load exact immediate-source identity/time authority for track input.

    Normal track writing never upgrades historical ``frame_indices`` or
    positional rows.  The selected source producer must already have
    published its canonical row identity and ``source_acquisition_frame_index``
    authority; audit/migration tooling handles older archives.
    """

    attrs = getattr(source_rowset, "attrs", None)
    if attrs is None:
        raise ValueError("Selected track source rowset has no persisted attrs.")
    try:
        contract = load_row_identity_contract_attrs(attrs)
        key_node = source_rowset[contract.key_array.ref]
        source_identity = load_bound_row_identity_contract(
            source_rowset,
            key_node,
        )
        source_frame_node = source_rowset["source_acquisition_frame_index"]
        return load_bound_source_row_temporal_authority(
            source_rowset,
            source_frame_node,
            source_row_identity=source_identity,
            acquisition_frame=acquisition_frame,
        )
    except Exception as exc:
        source_path = getattr(source_rowset, "path", "<unknown>")
        raise ValueError(
            f"Selected track source /{source_path} lacks an exact canonical "
            "row/time authority. Historical frame_indices, equal row counts, "
            "and nested provenance are not sufficient for future publication."
        ) from exc


def select_canonical_online_track_rows(
    handoff: CanonicalOnlineCoordinateHandoff,
    *,
    chaser_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select one chaser as an exact source-row subset for track publication.

    The bundle-level camera arrays are presentation conveniences and may contain
    rows without a stimulus state.  Track coordinates instead select directly
    from the descriptor-owning point surface and carry those exact source row
    indices into the track derivation record.
    """

    if isinstance(chaser_index, bool) or not isinstance(chaser_index, int):
        raise ValueError("chaser_index must be a nonnegative integer.")
    if chaser_index < 0:
        raise ValueError("chaser_index must be a nonnegative integer.")
    handoff.assert_verified()
    source = require_bound_canonical_coordinate_descriptor(
        handoff.coordinate_descriptor
    )
    keys = np.asarray(handoff.stimulus_state_key)
    components = handoff.row_identity.contract.key_array.components
    row_count = handoff.row_identity.leading_dimension
    if keys.shape[0] != row_count:
        raise ValueError("Stimulus-state key length differs from its sealed identity.")
    if "chaser_index" in components:
        component_index = components.index("chaser_index")
        if keys.ndim == 1:
            if len(components) != 1 or component_index != 0:
                raise ValueError(
                    "Rank-one stimulus-state keys do not match their components."
                )
            chaser_values = keys
        elif keys.ndim == 2 and keys.shape[1] == len(components):
            chaser_values = keys[:, component_index]
        else:
            raise ValueError(
                "Stimulus-state key shape does not match its declared components."
            )
        selected_rows = np.flatnonzero(chaser_values == chaser_index).astype(
            np.int64,
            copy=False,
        )
    else:
        if chaser_index != 0:
            raise ValueError(
                "Stimulus-state identity has no chaser_index component, so only "
                "chaser_index=0 is addressable."
            )
        selected_rows = np.arange(row_count, dtype=np.int64)
    if selected_rows.size == 0:
        raise ValueError(
            f"Stimulus-state source contains no rows for chaser_index={chaser_index}."
        )

    acquisition_frames = np.asarray(handoff.source_acquisition_frame_index)
    if acquisition_frames.dtype != np.dtype("<i8") or acquisition_frames.shape != (
        row_count,
    ):
        raise ValueError(
            "Stimulus source acquisition-frame mapping is not exact int64 row data."
        )
    selected_frames = acquisition_frames[selected_rows]
    if np.unique(selected_frames).shape[0] != selected_frames.shape[0]:
        raise ValueError(
            "Selected stimulus rows map ambiguously to duplicate acquisition frames."
        )
    order = np.argsort(selected_frames, kind="stable")
    selected_rows = selected_rows[order]
    selected_frames = selected_frames[order]

    source_values = np.array(source.coordinate_node[:], copy=True, order="C")
    if (
        source_values.dtype != np.dtype(source.coordinate_node.dtype)
        or source_values.shape != source.coordinate_node.shape
        or source_values.shape != (row_count, 2)
    ):
        raise ValueError(
            "Canonical stimulus position surface is not exact row-aligned (N, 2) data."
        )
    return selected_rows, selected_frames, source_values[selected_rows]


def load_stimulus_run_frames(root: zarr.Group, stimulus_run: Optional[str] = None) -> Optional[np.ndarray]:
    """Load the set of camera frame IDs from the stimulus run (experimental period).

    Returns None if no stimulus run is available.
    """
    if "analysis" not in root or "stimulus_runs" not in root["analysis"]:
        return None

    stimulus_parent = root["analysis"]["stimulus_runs"]
    stim_run = stimulus_run
    if stim_run is None:
        latest = stimulus_parent.attrs.get("latest")
        if isinstance(latest, bytes):
            latest = latest.decode("utf-8", "ignore")
        if isinstance(latest, str) and latest in stimulus_parent:
            stim_run = latest

    if stim_run is None or stim_run not in stimulus_parent:
        return None

    stim_group = stimulus_parent[stim_run]
    if "video_metadata" not in stim_group or "frame_metadata" not in stim_group["video_metadata"]:
        return None

    frame_metadata, _ = load_structured_dataset(
        stim_group["video_metadata"], "frame_metadata"
    )
    dtype_names = frame_metadata.dtype.names or ()

    # Find camera frame ID field
    camera_field = None
    for candidate in ["triggering_camera_frame_id", "camera_frame_id"]:
        if candidate in dtype_names:
            camera_field = candidate
            break

    if camera_field is None:
        return None

    camera_frames = np.asarray(frame_metadata[camera_field], dtype=np.int64)
    return np.unique(camera_frames)


def _resolve_calibration_group(root: zarr.Group) -> Tuple[Optional[zarr.Group], Optional[str]]:
    """Return the canonical calibration group and its logical path."""

    analysis = root.get("analysis")
    if analysis is not None:
        analysis_calibration = analysis.get("calibration")
        if analysis_calibration is not None:
            return analysis_calibration, "analysis/calibration"

    calibration = root.get("calibration")
    if calibration is not None:
        return calibration, "calibration"

    return None, None


def resolve_calibration(root: zarr.Group) -> Tuple[Optional[float], Dict[str, Any]]:
    """Retrieve pixel-to-mm conversion if available."""

    calibration, calibration_path = _resolve_calibration_group(root)
    if calibration is None:
        return None, {
            "has_calibration": False,
            "calibration_path": None,
            "measured_fps": None,
            "measured_stimulus_fps": None,
            "stimulus_offset_x": None,
            "stimulus_offset_y": None,
            "primary_camera_id": None,
            "camera_offsets": {},
        }

    pixel_to_mm = calibration.attrs.get("pixel_to_mm")
    pixel_to_mm_val = float(pixel_to_mm) if pixel_to_mm is not None else None
    measured_stimulus_fps = calibration.attrs.get("measured_stimulus_fps")
    if measured_stimulus_fps is None:
        measured_stimulus_fps = calibration.attrs.get("measured_fps")
    measured_stimulus_fps_val = (
        float(measured_stimulus_fps) if measured_stimulus_fps is not None else None
    )
    stim_offset_x = calibration.attrs.get("stimulus_offset_x")
    stim_offset_y = calibration.attrs.get("stimulus_offset_y")

    stim_offset_x_val = float(stim_offset_x) if stim_offset_x is not None else None
    stim_offset_y_val = float(stim_offset_y) if stim_offset_y is not None else None

    primary_camera_id = calibration.attrs.get("primary_camera_id")
    if isinstance(primary_camera_id, bytes):
        primary_camera_id_val = primary_camera_id.decode("utf-8", "ignore")
    else:
        primary_camera_id_val = primary_camera_id if primary_camera_id is not None else None

    camera_offsets = {}
    if "cameras" in calibration:
        cameras_group = calibration["cameras"]
        if hasattr(cameras_group, "group_keys"):
            camera_ids = list(cameras_group.group_keys())
        else:
            camera_ids = list(cameras_group.keys())

        for cam_id in camera_ids:
            cam_group = cameras_group[cam_id]
            cam_offsets = {}
            for key in ("stimulus_offset_x", "stimulus_offset_y"):
                val = cam_group.attrs.get(key)
                if val is not None:
                    cam_offsets[key] = float(val)
            if cam_offsets:
                camera_offsets[str(cam_id)] = cam_offsets

    return pixel_to_mm_val, {
        "has_calibration": pixel_to_mm_val is not None,
        "calibration_path": calibration_path,
        "measured_fps": measured_stimulus_fps_val,
        "measured_stimulus_fps": measured_stimulus_fps_val,
        "stimulus_offset_x": stim_offset_x_val,
        "stimulus_offset_y": stim_offset_y_val,
        "primary_camera_id": primary_camera_id_val,
        "camera_offsets": camera_offsets,
    }


def resolve_canonical_track_physical_authority(
    root: zarr.Group,
    *,
    stimulus_run: str | None,
) -> tuple[BoundStimulusPhysicalCoordinateAuthority | None, Dict[str, Any]]:
    """Resolve physical track authority only through one exact stimulus run."""

    analysis = root.get("analysis")
    stimulus_runs = analysis.get("stimulus_runs") if analysis is not None else None
    if stimulus_runs is None:
        return None, {
            "status": "omitted",
            "reason_code": "NO_STIMULUS_RUNS_CONTAINER",
            "stimulus_run": None,
        }
    selected_run = stimulus_run
    if selected_run is None:
        candidate = stimulus_runs.attrs.get("latest_complete")
        selected_run = candidate if isinstance(candidate, str) else None
    if selected_run is None:
        return None, {
            "status": "omitted",
            "reason_code": "NO_COMPLETE_STIMULUS_RUN_SELECTED",
            "stimulus_run": None,
        }
    selected_group = (
        stimulus_runs.get(selected_run)
        if isinstance(selected_run, str)
        and selected_run
        and callable(getattr(stimulus_runs, "get", None))
        else None
    )
    if selected_group is None:
        raise ValueError(
            f"Selected stimulus run {selected_run!r} does not exist exactly."
        )
    try:
        authority = load_stimulus_physical_coordinate_authority(
            root,
            stimulus_run=selected_run,
        )
    except StimulusPhysicalCoordinateUnavailableError:
        return None, {
            "status": "omitted",
            "reason_code": "STIMULUS_PHYSICAL_AUTHORITY_NOT_PUBLISHED",
            "stimulus_run": selected_run,
        }
    run_group = selected_group
    if authority is None:
        return None, {
            "status": str(
                run_group.attrs.get(
                    STIMULUS_PHYSICAL_COORDINATE_STATUS_ATTR,
                    "omitted",
                )
            ),
            "reason_code": str(
                run_group.attrs.get(
                    STIMULUS_PHYSICAL_COORDINATE_REASON_CODE_ATTR,
                    "STIMULUS_PHYSICAL_AUTHORITY_UNAVAILABLE",
                )
            ),
            "stimulus_run": selected_run,
        }
    bound = require_bound_stimulus_physical_coordinate_authority(authority)
    return bound, {
        "status": "bound_typed_source_camera_mm_v1",
        "reason_code": "NONE",
        "stimulus_run": selected_run,
        "camera_id": bound.camera_id,
        "mm_per_pixel": bound.mm_per_pixel,
        "authority_manifest_ref": bound.manifest.record_ref,
        "authority_manifest_sha256": bound.manifest.record_sha256,
        "physical_frame_ref": bound.physical_frame.record_ref,
        "physical_frame_sha256": bound.physical_frame.record_sha256,
    }


_TRACK_SELECTOR_MISSING = object()


@dataclass(frozen=True)
class _TrackSelectorMutation:
    """One exact parent-attribute mutation owned by a track publication."""

    parent_path: str
    attr_name: str
    previous: Any = dataclasses.field(repr=False, compare=False)
    written: Any = dataclasses.field(repr=False, compare=False)


@dataclass(frozen=True)
class DeferredTrackKinematicsSelectorActivation:
    """Process-local receipt for selected but still-ineligible track output."""

    root: Any = dataclasses.field(repr=False, compare=False)
    expected_archive: ArchiveIdentity
    expected_owner: Mapping[str, Any]
    mutations: tuple[_TrackSelectorMutation, ...] = dataclasses.field(
        repr=False,
        compare=False,
    )
    _commit: Callable[..., None] = dataclasses.field(repr=False, compare=False)

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        self._commit(*args, **kwargs)


def _track_publication_owner_uuid(run_group: Any) -> str:
    raw = run_group.attrs.get(TRACK_KINEMATICS_PUBLICATION_OWNER_ATTR)
    if not isinstance(raw, str):
        raise RuntimeError(
            f"/{run_group.path} lacks its atomic track-publication owner UUID."
        )
    try:
        parsed = uuid.UUID(raw)
    except (ValueError, AttributeError) as exc:
        raise RuntimeError(
            f"/{run_group.path} has an invalid track-publication owner UUID."
        ) from exc
    if str(parsed) != raw or parsed.version != 4:
        raise RuntimeError(
            f"/{run_group.path} has a noncanonical track-publication owner UUID."
        )
    return raw


def _resolve_owned_track_run_child(
    root: Any,
    *,
    run_name: str,
    run_type: str,
    owner_uuid: str,
    required: bool = True,
) -> Any | None:
    """Freshly resolve one run child and prove exact attempt ownership."""

    try:
        current = root["analysis"]["track_kinematics_runs"][run_type][run_name]
    except (KeyError, TypeError):
        if not required:
            return None
        raise RuntimeError(
            f"Track publication child {run_type}/{run_name!s} disappeared."
        )
    expected_path = f"analysis/track_kinematics_runs/{run_type}/{run_name}"
    if (
        str(current.path) != expected_path
        or current.attrs.get(TRACK_KINEMATICS_PUBLICATION_OWNER_ATTR)
        != owner_uuid
    ):
        if not required:
            return None
        raise RuntimeError(
            f"Track publication child {run_type}/{run_name!s} was replaced or "
            "lost exact attempt ownership."
        )
    return current


def _track_selector_owner_record(
    *,
    owner_uuid: str,
    qualified_name: str,
) -> dict[str, Any]:
    return {
        "schema_id": "palette.track_kinematics_selector_publication_owner",
        "schema_version": 1,
        "owner_uuid": owner_uuid,
        "qualified_run_name": qualified_name,
    }


def _selector_value(attrs: Any, name: str) -> Any:
    if name not in attrs:
        return _TRACK_SELECTOR_MISSING
    return copy.deepcopy(attrs[name])


def _selector_value_matches(current: Any, expected: Any) -> bool:
    if expected is _TRACK_SELECTOR_MISSING:
        return current is _TRACK_SELECTOR_MISSING
    return current is not _TRACK_SELECTOR_MISSING and _track_attr_values_equal(
        current,
        expected,
    )


def _resolve_track_selector_parent(
    root: Any,
    parent_path: str,
    *,
    expected_archive: ArchiveIdentity,
) -> Any:
    try:
        parent = root[parent_path]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(
            f"Track selector parent /{parent_path} disappeared during rollback."
        ) from exc
    if (
        str(parent.path).strip("/") != parent_path.strip("/")
        or archive_identity(parent) != expected_archive
    ):
        raise RuntimeError(
            f"Track selector parent /{parent_path} changed path or archive."
        )
    return parent


def _fresh_owned_track_selector_parent(
    root: Any,
    *,
    expected_archive: ArchiveIdentity,
    expected_owner: Mapping[str, Any],
) -> Any | None:
    parent = _resolve_track_selector_parent(
        root,
        "analysis/track_kinematics_runs",
        expected_archive=expected_archive,
    )
    if not _track_attr_values_equal(
        parent.attrs.get(TRACK_KINEMATICS_SELECTOR_OWNER_ATTR),
        dict(expected_owner),
    ):
        return None
    return parent


def _restore_track_selector_value(attrs: Any, name: str, previous: Any) -> None:
    """Perform one injectable selector rollback mutation."""

    if previous is _TRACK_SELECTOR_MISSING:
        if name in attrs:
            del attrs[name]
    else:
        attrs[name] = copy.deepcopy(previous)


def _restore_owned_selector_mutations(
    root: Any,
    mutations: Iterable[_TrackSelectorMutation],
    *,
    expected_archive: ArchiveIdentity,
    expected_owner: Mapping[str, Any],
) -> None:
    """Restore exact mutations, rechecking ownership before every write."""

    errors: list[str] = []
    for mutation in reversed(tuple(mutations)):
        # A one-time lease check is insufficient: another publisher can take
        # over between rollback writes. Re-resolve both the lease parent and
        # mutation parent for every attribute and stop on any ownership loss.
        if (
            _fresh_owned_track_selector_parent(
                root,
                expected_archive=expected_archive,
                expected_owner=expected_owner,
            )
            is None
        ):
            return
        parent = _resolve_track_selector_parent(
            root,
            mutation.parent_path,
            expected_archive=expected_archive,
        )
        current = _selector_value(parent.attrs, mutation.attr_name)
        if _selector_value_matches(current, mutation.previous):
            continue
        if not _selector_value_matches(current, mutation.written):
            # An unleased or partially concurrent mutation is not ours to
            # repair. Stop instead of composing a mixed selector state.
            return

        # Reading the attempted value may itself yield control to a hostile or
        # remote store. Recheck the exact lease immediately before mutation,
        # then re-resolve and compare the attempted value one final time.
        if (
            _fresh_owned_track_selector_parent(
                root,
                expected_archive=expected_archive,
                expected_owner=expected_owner,
            )
            is None
        ):
            return
        parent = _resolve_track_selector_parent(
            root,
            mutation.parent_path,
            expected_archive=expected_archive,
        )
        current = _selector_value(parent.attrs, mutation.attr_name)
        if _selector_value_matches(current, mutation.previous):
            continue
        if not _selector_value_matches(current, mutation.written):
            return
        try:
            _restore_track_selector_value(
                parent.attrs,
                mutation.attr_name,
                mutation.previous,
            )
            restoring_lease = (
                mutation.parent_path == "analysis/track_kinematics_runs"
                and mutation.attr_name == TRACK_KINEMATICS_SELECTOR_OWNER_ATTR
            )
            if not restoring_lease and (
                _fresh_owned_track_selector_parent(
                    root,
                    expected_archive=expected_archive,
                    expected_owner=expected_owner,
                )
                is None
            ):
                return
            verified_parent = _resolve_track_selector_parent(
                root,
                mutation.parent_path,
                expected_archive=expected_archive,
            )
            if not _selector_value_matches(
                _selector_value(verified_parent.attrs, mutation.attr_name),
                mutation.previous,
            ):
                errors.append(
                    f"verify selector {mutation.attr_name!r}: persisted value differs"
                )
                break
        except BaseException as exc:  # pragma: no cover - hostile store
            errors.append(f"restore selector {mutation.attr_name!r}: {exc}")
            break
    if errors:
        raise RuntimeError(
            f"Owner-aware track selector rollback was incomplete: {errors!r}."
        )


def rollback_deferred_track_kinematics_selector_activation(
    activation: DeferredTrackKinematicsSelectorActivation,
    *,
    root: Any | None = None,
) -> None:
    """Cancel only the exact selector mutations recorded by one receipt."""

    if type(activation) is not DeferredTrackKinematicsSelectorActivation:
        raise TypeError("Deferred track selector activation receipt is invalid.")
    rollback_root = activation.root if root is None else root
    if archive_identity(rollback_root) != activation.expected_archive:
        raise RuntimeError("Deferred track selector rollback changed archives/stores.")
    _restore_owned_selector_mutations(
        rollback_root,
        activation.mutations,
        expected_archive=activation.expected_archive,
        expected_owner=activation.expected_owner,
    )

def resolve_track_physical_authority(
    root: zarr.Group,
    *,
    stimulus_run: str | None,
) -> tuple[TrackPhysicalAuthority | None, Dict[str, Any]]:
    """Resolve stimulus or recording calibration to one downstream contract."""

    stimulus, info = resolve_canonical_track_physical_authority(
        root,
        stimulus_run=stimulus_run,
    )
    if stimulus is not None:
        return stimulus, info
    try:
        recording = load_source_camera_physical_authority(root)
    except KeyError:
        return None, info
    recording = require_bound_source_camera_physical_authority(recording)
    return recording, {
        "status": "bound_typed_source_camera_mm_v1",
        "reason_code": "NONE",
        "authority_kind": "recording_calibration",
        "camera_id": recording.camera_id,
        "mm_per_pixel": recording.mm_per_pixel,
        "authority_manifest_ref": recording.manifest.record_ref,
        "authority_manifest_sha256": recording.manifest.record_sha256,
        "physical_frame_ref": recording.physical_frame.record_ref,
        "physical_frame_sha256": recording.physical_frame.record_sha256,
    }
def ensure_track_kinematics_run_group(
    root: zarr.Group,
    run_name: Optional[str],
    *,
    run_type: str = "online",
    overwrite: bool = False,
) -> Tuple[str, zarr.Group]:
    """Create one immutable canonical track run child.

    ``overwrite`` is retained only for call-site compatibility.  Occupied
    canonical names, including failed public tombstones, are never reused.
    """

    if run_type not in {"online", "offline"}:
        raise ValueError("run_type must be 'online' or 'offline'")

    analysis = root.require_group("analysis")
    track_parent = require_runs_parent(analysis, "track_kinematics_runs")
    type_parent = track_parent.require_group(run_type)

    if run_name:
        if run_name in type_parent:
            qualified_name = f"{run_type}/{run_name}"
            existing = type_parent[run_name]
            selected = (
                track_parent.attrs.get("latest") == qualified_name
                or track_parent.attrs.get("latest_complete") == qualified_name
                or track_parent.attrs.get(f"latest_{run_type}") == run_name
                or type_parent.attrs.get("latest") == run_name
                or type_parent.attrs.get("latest_complete") == run_name
                or existing.attrs.get("stage_selector_eligible") is True
            )
            status = existing.attrs.get(RUN_COMPLETION_STATUS_ATTR)
            if not overwrite:
                raise ValueError(
                    f"Track kinematics run '{run_name}' already exists under {run_type}."
                )
            if selected or status == RUN_STATUS_COMPLETE:
                raise ValueError(
                    f"Refusing to overwrite complete or selected track kinematics "
                    f"run {qualified_name!r}; publish a new immutable run name."
                )
            if status != RUN_STATUS_FAILED:
                raise ValueError(
                    f"Refusing to overwrite non-failed track kinematics run "
                    f"{qualified_name!r} (status={status!r}); publish a new immutable "
                    "run name."
                )
            raise ValueError(
                f"Refusing to overwrite failed track kinematics run "
                f"{qualified_name!r}; failed public children are immutable "
                "tombstones. Publish a new immutable run name."
            )
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        prefix = (
            "track_kinematics"
            if run_type == "online"
            else "track_kinematics_offline"
        )
        run_name = f"{prefix}_{timestamp}"

    owner_uuid = str(uuid.uuid4())
    try:
        qualified_name = f"{run_type}/{run_name}"
        run_group = type_parent.create_group(
            run_name,
            attributes={
                "stage_selector_eligible": False,
                TRACK_KINEMATICS_PUBLICATION_OWNER_ATTR: owner_uuid,
                "palette_run_completion_contract": (
                    "palette.zarr_run_completion.v1"
                ),
                RUN_COMPLETION_STATUS_ATTR: "running",
                "palette_run_started_at_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
                "palette_run_name": qualified_name,
                "palette_run_stage": "track_kinematics",
            },
        )
        mark_run_started(
            run_group,
            run_name=qualified_name,
            stage="track_kinematics",
        )
        run_group = _resolve_owned_track_run_child(
            root,
            run_name=run_name,
            run_type=run_type,
            owner_uuid=owner_uuid,
        )
        if run_group.attrs.get("stage_selector_eligible") is not False:
            raise RuntimeError(
                "Track run did not persist fail-closed selector eligibility at "
                "creation."
            )
    except BaseException as exc:
        rollback_errors: list[str] = []

        def fresh_owned_run() -> Optional[zarr.Group]:
            return _resolve_owned_track_run_child(
                root,
                run_name=run_name,
                run_type=run_type,
                owner_uuid=owner_uuid,
                required=False,
            )

        owned_run = fresh_owned_run()
        if owned_run is not None:
            try:
                owned_run.attrs["stage_selector_eligible"] = False
            except BaseException as rollback_exc:  # pragma: no cover - hostile store
                rollback_errors.append(f"disarm eligibility: {rollback_exc}")
            try:
                owned_run = fresh_owned_run()
                if owned_run is not None:
                    if "palette_run_completed_at_utc" in owned_run.attrs:
                        del owned_run.attrs["palette_run_completed_at_utc"]
            except BaseException as rollback_exc:  # pragma: no cover - hostile store
                rollback_errors.append(f"remove completion timestamp: {rollback_exc}")
            try:
                owned_run = fresh_owned_run()
                if owned_run is not None:
                    mark_run_failed(
                        owned_run,
                        parent_group=track_parent,
                        run_name=f"{run_type}/{run_name}",
                        error=str(exc),
                    )
            except BaseException as rollback_exc:  # pragma: no cover - hostile store
                rollback_errors.append(f"mark failed: {rollback_exc}")
            try:
                owned_run = fresh_owned_run()
                if owned_run is not None:
                    owned_run.attrs[
                        TRACK_KINEMATICS_PUBLICATION_TOMBSTONE_ATTR
                    ] = json_attr_safe(
                        {
                            "schema_id": (
                                "palette.track_kinematics_publication_tombstone"
                            ),
                            "schema_version": 1,
                            "publication_owner_uuid": owner_uuid,
                            "qualified_run_name": f"{run_type}/{run_name}",
                            "public_path_retained": True,
                            "selector_eligible": False,
                            "retry_policy": "new_immutable_run_name_required",
                        }
                    )
            except BaseException as rollback_exc:  # pragma: no cover - hostile store
                rollback_errors.append(f"persist failed tombstone: {rollback_exc}")
        if rollback_errors:
            raise RuntimeError(
                "Track run creation failed and could not be left explicitly failed "
                f"and selector-ineligible: {rollback_errors!r}."
            ) from exc
        raise

    return run_name, run_group


def mark_track_kinematics_run_complete(
    root: zarr.Group,
    run_group: zarr.Group,
    *,
    run_name: str,
    run_type: str,
    publication_owner_uuid: str,
    validate_complete_run: Callable[[zarr.Group], Mapping[str, Any]],
    defer_selector_eligibility: bool = False,
    deferred_activation_sink: (
        Callable[[DeferredTrackKinematicsSelectorActivation], None] | None
    ) = None,
) -> Optional[DeferredTrackKinematicsSelectorActivation]:
    """Validate a complete ineligible run, prepare pointers, and expose it last."""

    if deferred_activation_sink is not None and not defer_selector_eligibility:
        raise ValueError(
            "A deferred track activation sink requires deferred selector eligibility."
        )
    track_parent = root["analysis"]["track_kinematics_runs"]
    type_parent = track_parent[run_type]
    expected_archive = archive_identity(root)
    qualified_name = f"{run_type}/{run_name}"
    expected_path = f"analysis/track_kinematics_runs/{qualified_name}"
    if str(run_group.path) != expected_path:
        raise ValueError(
            f"Track run path /{run_group.path} differs from /{expected_path}."
        )
    try:
        parsed_owner = uuid.UUID(str(publication_owner_uuid))
    except (ValueError, AttributeError) as exc:
        raise ValueError("Track completion requires one canonical owner UUID.") from exc
    owner_uuid = str(parsed_owner)
    if (
        owner_uuid != publication_owner_uuid
        or parsed_owner.version != 4
    ):
        raise ValueError("Track completion requires one canonical UUIDv4 owner.")
    resolved_run = _resolve_owned_track_run_child(
        root,
        run_name=run_name,
        run_type=run_type,
        owner_uuid=owner_uuid,
    )
    assert resolved_run is not None
    run_group = resolved_run
    if run_group.attrs.get("stage_selector_eligible") is not False:
        raise ValueError(
            "Track completion requires literal stage_selector_eligible=false."
        )
    selector_mutations: list[_TrackSelectorMutation] = []

    def selector_parent(parent_path: str, *, require_owned: bool) -> Any:
        if require_owned and (
            _fresh_owned_track_selector_parent(
                root,
                expected_archive=expected_archive,
                expected_owner=selector_owner,
            )
            is None
        ):
            raise RuntimeError("Track selector ownership changed during publication.")
        return _resolve_track_selector_parent(
            root,
            parent_path,
            expected_archive=expected_archive,
        )

    def write_selector(
        parent_path: str,
        name: str,
        value: Any,
        *,
        require_owned: bool = True,
    ) -> None:
        parent = selector_parent(parent_path, require_owned=require_owned)
        previous = _selector_value(parent.attrs, name)
        written = copy.deepcopy(value)
        selector_mutations.append(
            _TrackSelectorMutation(
                parent_path=parent_path,
                attr_name=name,
                previous=previous,
                written=written,
            )
        )
        parent.attrs[name] = copy.deepcopy(written)

    def delete_selector(parent_path: str, name: str) -> None:
        parent = selector_parent(parent_path, require_owned=True)
        if name not in parent.attrs:
            return
        previous = _selector_value(parent.attrs, name)
        selector_mutations.append(
            _TrackSelectorMutation(
                parent_path=parent_path,
                attr_name=name,
                previous=previous,
                written=_TRACK_SELECTOR_MISSING,
            )
        )
        del parent.attrs[name]

    selector_owner = _track_selector_owner_record(
        owner_uuid=owner_uuid,
        qualified_name=qualified_name,
    )
    try:
        provenance = build_run_provenance_from_stage_record(
            run_group.attrs.get("provenance", {}),
            fallback_command="track_kinematics",
        )
        provenance_validation = validate_run_provenance(provenance)
        if not provenance_validation.valid:
            raise RuntimeError(
                "Track completion provenance is invalid: "
                f"{provenance_validation.errors!r}."
            )
        run_group = _resolve_owned_track_run_child(
            root,
            run_name=run_name,
            run_type=run_type,
            owner_uuid=owner_uuid,
        )
        assert run_group is not None
        mark_run_complete(
            run_group,
            # Parent selectors are handled below under an explicit attempt
            # owner.  Passing the parent here would let the generic lifecycle
            # helper clear latest_pending without ownership evidence.
            parent_group=None,
            run_name=qualified_name,
            run_provenance=(
                provenance_validation.normalized
                or dict(provenance)
            ),
        )
        if (
            run_group.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE
            or run_group.attrs.get("stage_selector_eligible") is not False
        ):
            raise RuntimeError(
                "Track run did not remain complete and selector-ineligible for final "
                "validation."
            )
        sealed_motion = _seal_and_load_track_motion_run_before_selection(
            root,
            run_group,
            expected_publication_owner_uuid=owner_uuid,
        )
        if not sealed_motion.tracks:
            raise RuntimeError(
                "Complete track run produced no sealed full-motion tracks."
            )
        resolved_run = _resolve_owned_track_run_child(
            root,
            run_name=run_name,
            run_type=run_type,
            owner_uuid=owner_uuid,
        )
        assert resolved_run is not None
        validation = validate_complete_run(resolved_run)
        if not isinstance(validation, Mapping) or validation.get("valid") is not True:
            raise RuntimeError(
                "Complete track pre-selection validation did not report valid=true: "
                f"{validation!r}."
            )
        resolved_run = _resolve_owned_track_run_child(
            root,
            run_name=run_name,
            run_type=run_type,
            owner_uuid=owner_uuid,
        )
        assert resolved_run is not None
        sealed_motion.assert_verified()

        write_selector(
            "analysis/track_kinematics_runs",
            TRACK_KINEMATICS_SELECTOR_OWNER_ATTR,
            selector_owner,
            require_owned=False,
        )
        if (
            _fresh_owned_track_selector_parent(
                root,
                expected_archive=expected_archive,
                expected_owner=selector_owner,
            )
            is None
        ):
            raise RuntimeError("Track selector ownership was lost before publication.")
        write_selector(
            "analysis/track_kinematics_runs",
            "latest_complete",
            qualified_name,
        )
        write_selector(
            "analysis/track_kinematics_runs",
            "latest",
            qualified_name,
        )
        write_selector(
            f"analysis/track_kinematics_runs/{run_type}",
            "latest",
            run_name,
        )
        attr_key = "latest_online" if run_type == "online" else "latest_offline"
        write_selector(
            "analysis/track_kinematics_runs",
            attr_key,
            run_name,
        )
        pending_parent = _fresh_owned_track_selector_parent(
            root,
            expected_archive=expected_archive,
            expected_owner=selector_owner,
        )
        if pending_parent is None:
            raise RuntimeError("Track selector ownership changed during publication.")
        if pending_parent.attrs.get("latest_pending") == qualified_name:
            delete_selector(
                "analysis/track_kinematics_runs",
                "latest_pending",
            )
        fresh_track_parent = root["analysis"]["track_kinematics_runs"]
        if (
            fresh_track_parent.attrs.get(TRACK_KINEMATICS_SELECTOR_OWNER_ATTR)
            != selector_owner
        ):
            raise RuntimeError("Track selector ownership changed during publication.")
        resolved_run = _resolve_owned_track_run_child(
            root,
            run_name=run_name,
            run_type=run_type,
            owner_uuid=owner_uuid,
        )
        assert resolved_run is not None
        sealed_motion.assert_verified()
        baseline_motion_manifest_sha256 = getattr(
            sealed_motion,
            "manifest_sha256",
            None,
        )

        def commit_selector_eligibility(
            commit_root: zarr.Group,
            commit_type_parent: zarr.Group,
            commit_run_group: zarr.Group,
            *,
            validate_fresh_complete_run: Callable[
                [zarr.Group], Mapping[str, Any]
            ],
            expected_cluster_output_staging: Any = None,
        ) -> None:
            """Rebind the receipt to fresh handles and expose eligibility last."""

            if (
                archive_identity(commit_root) != expected_archive
                or archive_identity(commit_type_parent) != expected_archive
                or archive_identity(commit_run_group) != expected_archive
                or str(commit_type_parent.path)
                != f"analysis/track_kinematics_runs/{run_type}"
                or str(commit_run_group.path) != expected_path
                or commit_run_group.attrs.get(
                    TRACK_KINEMATICS_PUBLICATION_OWNER_ATTR
                )
                != owner_uuid
            ):
                raise RuntimeError(
                    "Track eligibility commit received invalid fresh archive, "
                    "parent, run, or owner binding."
                )
            fresh_track_parent = commit_root["analysis"][
                "track_kinematics_runs"
            ]
            fresh_type_parent = fresh_track_parent[run_type]
            fresh_run = _resolve_owned_track_run_child(
                commit_root,
                run_name=run_name,
                run_type=run_type,
                owner_uuid=owner_uuid,
            )
            assert fresh_run is not None
            if (
                fresh_track_parent.attrs.get(
                    TRACK_KINEMATICS_SELECTOR_OWNER_ATTR
                )
                != selector_owner
                or fresh_track_parent.attrs.get("latest_complete")
                != qualified_name
                or fresh_track_parent.attrs.get("latest") != qualified_name
                or fresh_track_parent.attrs.get(attr_key) != run_name
                or fresh_type_parent.attrs.get("latest") != run_name
                or fresh_run.attrs.get(RUN_COMPLETION_STATUS_ATTR)
                != RUN_STATUS_COMPLETE
                or fresh_run.attrs.get("stage_selector_eligible") is not False
            ):
                raise RuntimeError(
                    "Track eligibility commit lost its exact completed child or "
                    "owned selector state."
                )
            if (
                expected_cluster_output_staging is not None
                and not _track_attr_values_equal(
                    fresh_run.attrs.get("cluster_output_staging"),
                    expected_cluster_output_staging,
                )
            ):
                raise RuntimeError(
                    "Track eligibility commit observed different final publisher "
                    "metadata."
                )
            validation = validate_fresh_complete_run(fresh_run)
            if (
                not isinstance(validation, Mapping)
                or validation.get("valid") is not True
            ):
                raise RuntimeError(
                    "Track eligibility commit fresh validation did not report "
                    f"valid=true: {validation!r}."
                )
            fresh_motion = _seal_and_load_track_motion_run_before_selection(
                commit_root,
                fresh_run,
                expected_publication_owner_uuid=owner_uuid,
            )
            if not fresh_motion.tracks or (
                baseline_motion_manifest_sha256 is not None
                and getattr(fresh_motion, "manifest_sha256", None)
                != baseline_motion_manifest_sha256
            ):
                raise RuntimeError(
                    "Track eligibility commit observed different sealed motion "
                    "payload."
                )
            fresh_motion.assert_verified()
            fresh_run = _resolve_owned_track_run_child(
                commit_root,
                run_name=run_name,
                run_type=run_type,
                owner_uuid=owner_uuid,
            )
            assert fresh_run is not None
            if (
                expected_cluster_output_staging is not None
                and not _track_attr_values_equal(
                    fresh_run.attrs.get("cluster_output_staging"),
                    expected_cluster_output_staging,
                )
            ):
                raise RuntimeError(
                    "Track motion revalidation changed final publisher metadata."
                )
            try:
                # Persistent publication commit point: no fallible store
                # mutation follows on the ordinary success path.
                fresh_run.attrs["stage_selector_eligible"] = True
            except BaseException:
                committed = _resolve_owned_track_run_child(
                    commit_root,
                    run_name=run_name,
                    run_type=run_type,
                    owner_uuid=owner_uuid,
                    required=False,
                )
                if (
                    committed is not None
                    and committed.attrs.get("stage_selector_eligible") is True
                ):
                    return
                raise

        if defer_selector_eligibility:
            activation = DeferredTrackKinematicsSelectorActivation(
                root=root,
                expected_archive=expected_archive,
                expected_owner=copy.deepcopy(selector_owner),
                mutations=tuple(selector_mutations),
                _commit=commit_selector_eligibility,
            )
            if deferred_activation_sink is not None:
                deferred_activation_sink(activation)
            return activation
        commit_selector_eligibility(
            root,
            type_parent,
            resolved_run,
            validate_fresh_complete_run=validate_complete_run,
        )
        return None
    except BaseException as exc:
        rollback_errors: list[str] = []
        owned_run = _resolve_owned_track_run_child(
            root,
            run_name=run_name,
            run_type=run_type,
            owner_uuid=owner_uuid,
            required=False,
        )
        if owned_run is not None:
            try:
                owned_run.attrs["stage_selector_eligible"] = False
            except BaseException as rollback_exc:  # pragma: no cover - hostile store
                rollback_errors.append(
                    f"disarm owned selector eligibility: {rollback_exc}"
                )
            try:
                mark_run_failed(
                    owned_run,
                    parent_group=None,
                    run_name=qualified_name,
                    error=str(exc),
                )
            except BaseException as rollback_exc:  # pragma: no cover - hostile store
                rollback_errors.append(f"mark owned run failed: {rollback_exc}")
        try:
            _restore_owned_selector_mutations(
                root,
                selector_mutations,
                expected_archive=expected_archive,
                expected_owner=selector_owner,
            )
        except BaseException as rollback_exc:  # pragma: no cover - hostile store
            rollback_errors.append(str(rollback_exc))
        if rollback_errors:
            raise RuntimeError(
                "Track completion failed and exact publication rollback was "
                f"incomplete: {rollback_errors!r}."
            ) from exc
        raise


def _nan_array(shape: Tuple[int, ...], dtype: np.dtype = np.float32) -> np.ndarray:
    arr = np.empty(shape, dtype=dtype)
    arr.fill(np.nan)
    return arr


def _float32(data: np.ndarray) -> np.ndarray:
    return np.asarray(data, dtype=np.float32)


def _int64(data: np.ndarray) -> np.ndarray:
    return np.asarray(data, dtype=np.int64)


def _boolean(data: np.ndarray) -> np.ndarray:
    return np.asarray(data, dtype=bool)


def _build_sample_validity_arrays(
    *,
    track_id: int,
    positions_px: np.ndarray,
    headings_deg: np.ndarray,
    keypoint_success: np.ndarray,
    detection_source: Optional[np.ndarray],
) -> Dict[str, np.ndarray]:
    """Project upstream row/source/keypoint state into track-aligned validity arrays."""

    n_rows = int(positions_px.shape[0])
    sample_observed = np.full(n_rows, track_id >= 0, dtype=bool)
    position_finite = np.all(np.isfinite(positions_px), axis=1)
    heading_usable = np.asarray(keypoint_success, dtype=bool) & np.isfinite(headings_deg)
    keypoint_usable = heading_usable.copy()
    if detection_source is None:
        source_observed = np.ones(n_rows, dtype=bool)
    else:
        source_observed = np.asarray(detection_source, dtype=np.int8) == 0

    sample_valid = (
        sample_observed
        & source_observed
        & keypoint_usable
        & position_finite
    )

    reason = np.full(n_rows, SAMPLE_REASON_OK, dtype=np.int16)
    reason[~position_finite] = SAMPLE_REASON_POSITION_NAN
    reason[np.asarray(keypoint_success, dtype=bool) & ~np.isfinite(headings_deg)] = (
        SAMPLE_REASON_HEADING_UNUSABLE
    )
    reason[~np.asarray(keypoint_success, dtype=bool)] = SAMPLE_REASON_KEYPOINT_FAILED
    reason[~source_observed] = SAMPLE_REASON_SOURCE_INTERPOLATED
    reason[~sample_observed] = SAMPLE_REASON_UNASSIGNED
    reason[sample_valid] = SAMPLE_REASON_OK

    return {
        "sample_observed": sample_observed,
        "sample_valid": sample_valid,
        "source_observed": source_observed,
        "keypoint_usable": keypoint_usable,
        "position_finite": position_finite,
        "heading_usable": heading_usable,
        "sample_reason_code": reason,
    }


def _filter_public_track_rows(
    *,
    track_ids: np.ndarray,
    frames: np.ndarray,
    positions_px: np.ndarray,
    headings_deg: np.ndarray,
    keypoint_success: np.ndarray,
    detection_source: Optional[np.ndarray],
    include_unassigned: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Return row-aligned arrays suitable for public offline track outputs."""

    if include_unassigned:
        return (
            track_ids,
            frames,
            positions_px,
            headings_deg,
            keypoint_success,
            detection_source,
        )

    valid_mask = track_ids >= 0
    if not np.any(valid_mask):
        return (
            track_ids[valid_mask],
            frames[valid_mask],
            positions_px[valid_mask],
            headings_deg[valid_mask],
            keypoint_success[valid_mask],
            detection_source[valid_mask] if detection_source is not None else None,
        )

    return (
        track_ids[valid_mask],
        frames[valid_mask],
        positions_px[valid_mask],
        headings_deg[valid_mask],
        keypoint_success[valid_mask],
        detection_source[valid_mask] if detection_source is not None else None,
    )


def _ordered_track_arena_ids(
    ordered_ids: List[int],
    track_id_to_arena_id: Optional[Dict[int, int]],
) -> Optional[np.ndarray]:
    """Return arena IDs parallel to ordered track IDs for persisted outputs."""

    if not track_id_to_arena_id:
        return None

    unexpected_missing = [
        track_id
        for track_id in ordered_ids
        if track_id >= 0 and track_id not in track_id_to_arena_id
    ]
    if unexpected_missing:
        raise ValueError(
            "Missing arena mapping for persisted track IDs: "
            + ", ".join(str(track_id) for track_id in unexpected_missing)
        )

    return np.asarray(
        [int(track_id_to_arena_id.get(track_id, -1)) for track_id in ordered_ids],
        dtype=np.int32,
    )


def _wrap_heading_delta_degrees(delta_degrees: np.ndarray) -> np.ndarray:
    """Wrap heading deltas into the signed [-180, 180) range."""

    delta = np.asarray(delta_degrees, dtype=np.float64)
    return ((delta + 180.0) % 360.0) - 180.0


def _compute_heading_turning(
    headings_deg: np.ndarray,
    delta_seconds_full: np.ndarray,
    *,
    transition_valid: Optional[np.ndarray] = None,
    sample_valid: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return gap-aware heading delta, angular velocity, and angular speed."""

    headings = np.asarray(headings_deg, dtype=np.float64)
    delta_seconds = np.asarray(delta_seconds_full, dtype=np.float64)
    delta_heading = np.full(headings.shape[0], np.nan, dtype=np.float64)
    angular_velocity = np.full(headings.shape[0], np.nan, dtype=np.float64)
    angular_speed = np.full(headings.shape[0], np.nan, dtype=np.float64)

    if headings.size < 2:
        return delta_heading, angular_velocity, angular_speed

    step_delta = _wrap_heading_delta_degrees(headings[1:] - headings[:-1])
    valid = (
        np.isfinite(headings[1:])
        & np.isfinite(headings[:-1])
        & np.isfinite(delta_seconds[1:])
        & (delta_seconds[1:] > 0)
    )
    if transition_valid is not None:
        transition = np.asarray(transition_valid, dtype=bool)
        if transition.shape[0] == headings.shape[0]:
            valid &= transition[1:]
    if sample_valid is not None:
        samples = np.asarray(sample_valid, dtype=bool)
        if samples.shape[0] == headings.shape[0]:
            valid &= samples[1:] & samples[:-1]

    delta_heading_step_values = delta_heading[1:]
    delta_heading_step_values[valid] = step_delta[valid]
    angular_velocity_values = np.full(step_delta.shape, np.nan, dtype=np.float64)
    angular_velocity_values[valid] = step_delta[valid] / delta_seconds[1:][valid]
    angular_velocity[1:] = angular_velocity_values
    angular_speed[1:] = np.abs(angular_velocity_values)
    return delta_heading, angular_velocity, angular_speed


def _bounded_smoothing_window(requested: int, sample_count: int) -> int:
    """Return the exact smoothing window usable by one array domain."""

    if sample_count <= 0:
        return 0
    return min(max(1, int(requested)), int(sample_count))


def _smooth_acceleration_trace(acceleration_px: np.ndarray, window: int) -> np.ndarray:
    """Return a centered moving average of acceleration, ignoring NaNs."""

    acceleration = np.asarray(acceleration_px, dtype=np.float64)
    effective_window = _bounded_smoothing_window(window, int(acceleration.size))
    if effective_window <= 1:
        return acceleration.copy()

    kernel = np.ones(effective_window, dtype=np.float64)
    val_mask = np.isfinite(acceleration).astype(np.float64)
    accel_values = np.nan_to_num(acceleration, nan=0.0, copy=True)
    sum_values = np.convolve(accel_values, kernel, mode="same")
    count_values = np.convolve(val_mask, kernel, mode="same")
    smoothed = np.full_like(acceleration, np.nan)
    valid = count_values > 0
    smoothed[valid] = sum_values[valid] / count_values[valid]
    return smoothed


def _smooth_heading_radians(
    headings_deg: np.ndarray,
    requested_window: int,
) -> tuple[np.ndarray, int]:
    """Circularly smooth persisted float32 headings with zero-masked gaps."""

    headings = np.asarray(headings_deg, dtype=np.float32)
    heading_radians = np.deg2rad(headings)
    effective_window = _bounded_smoothing_window(
        requested_window,
        int(heading_radians.size),
    )
    if effective_window <= 1:
        return np.array(heading_radians, copy=True), effective_window

    finite = np.isfinite(heading_radians)
    kernel = np.ones(effective_window, dtype=np.float64)
    valid_weights = np.convolve(finite.astype(np.float64), kernel, mode="same")
    # Invalid headings contribute zero weight and zero numerator.  Substituting
    # angle zero before cosine would instead inject a spurious +X unit vector.
    cos_values = np.where(finite, np.cos(heading_radians), 0.0)
    sin_values = np.where(finite, np.sin(heading_radians), 0.0)
    cos_sum = np.convolve(cos_values, kernel, mode="same")
    sin_sum = np.convolve(sin_values, kernel, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_mean = np.where(valid_weights > 0, cos_sum / valid_weights, np.nan)
        sin_mean = np.where(valid_weights > 0, sin_sum / valid_weights, np.nan)
    return np.arctan2(sin_mean, cos_mean), effective_window


def _physical_values_from_pixel_peer(
    pixel_values: np.ndarray,
    mm_per_pixel: Optional[float],
) -> np.ndarray:
    """Scale one finalized pixel payload in its own persisted dtype domain."""

    pixel = np.asarray(pixel_values)
    if pixel.dtype.kind != "f":
        raise ValueError("Physical track peers require floating-point pixels.")
    if mm_per_pixel is None or not math.isfinite(mm_per_pixel):
        return _nan_array(pixel.shape, dtype=pixel.dtype)
    scale = np.asarray(mm_per_pixel, dtype=pixel.dtype)
    with np.errstate(over="ignore", invalid="ignore"):
        return np.asarray(pixel * scale, dtype=pixel.dtype)


def _compute_speed_derivative(
    speed_px: np.ndarray,
    delta_seconds_full: np.ndarray,
    *,
    pixel_to_mm: Optional[float],
    smooth_seconds: float,
    fps: float,
) -> Dict[str, np.ndarray | int | float | str]:
    """Differentiate one named speed trace and return its acceleration arrays."""

    speed = np.asarray(speed_px, dtype=np.float64)
    delta_seconds = np.asarray(delta_seconds_full, dtype=np.float64)
    acceleration_px = np.full(speed.shape, np.nan, dtype=np.float64)

    if speed.size >= 2:
        delta_speed_px = speed[1:] - speed[:-1]
        delta_t = delta_seconds[1:]
        valid = (delta_t > 0) & np.isfinite(delta_speed_px)
        accel_vals = np.full(delta_speed_px.shape, np.nan, dtype=np.float64)
        accel_vals[valid] = delta_speed_px[valid] / delta_t[valid]
        acceleration_px[1:] = accel_vals

    requested_post_window = max(1, int(round(fps * smooth_seconds)))
    effective_post_window = _bounded_smoothing_window(
        requested_post_window,
        int(acceleration_px.size),
    )
    smoothed_acceleration_px = _smooth_acceleration_trace(
        acceleration_px,
        effective_post_window,
    )
    # Float32 is the public acceleration precision.  Freeze it before deriving
    # any physical peer so writer and validator share one rounding path.
    persisted_acceleration_px = _float32(acceleration_px)
    persisted_smoothed_acceleration_px = _float32(smoothed_acceleration_px)
    acceleration_mm = _physical_values_from_pixel_peer(
        persisted_acceleration_px,
        pixel_to_mm,
    )
    smoothed_acceleration_mm = _physical_values_from_pixel_peer(
        persisted_smoothed_acceleration_px,
        pixel_to_mm,
    )

    return {
        "acceleration_px": persisted_acceleration_px,
        "acceleration_mm": acceleration_mm,
        "smoothed_acceleration_px": persisted_smoothed_acceleration_px,
        "smoothed_acceleration_mm": smoothed_acceleration_mm,
        "derivative_method": "first_difference",
        "post_smoothing_method": "moving_average",
        "post_smoothing_alignment": "centered",
        "post_smoothing_window_frames": int(effective_post_window),
        "post_smoothing_window_frames_requested": int(requested_post_window),
        "post_smoothing_window_frames_effective": int(effective_post_window),
        "post_smoothing_window_s": float(smooth_seconds),
    }


def _compute_speed_derivatives(
    speed_by_level_px: Dict[str, np.ndarray],
    delta_seconds_full: np.ndarray,
    *,
    pixel_to_mm: Optional[float],
    smooth_seconds: float,
    fps: float,
) -> Dict[str, Dict[str, np.ndarray | int | float | str]]:
    """Return acceleration derivatives for every persisted speed level."""

    return {
        level: _compute_speed_derivative(
            speed_by_level_px[level],
            delta_seconds_full,
            pixel_to_mm=pixel_to_mm,
            smooth_seconds=smooth_seconds,
            fps=fps,
        )
        for level in SPEED_DERIVATIVE_LEVELS
        if level in speed_by_level_px
    }


def build_track_datasets(
    track_ids: np.ndarray,
    frames: np.ndarray,
    positions_px: np.ndarray,
    headings_deg: np.ndarray,
    keypoint_success: np.ndarray,
    detection_source: Optional[np.ndarray],
    fps: float,
    smooth_seconds: float,
    pixel_to_mm: Optional[float],
    hysteresis_high_px: Optional[float] = None,
    hysteresis_low_px: Optional[float] = None,
    hysteresis_min_frames: Optional[int] = None,
    hysteresis_band_policy: str = DEFAULT_HYSTERESIS_BAND_POLICY,
    smoothing_method: str = "moving_average",
    smoothing_alignment: str = DEFAULT_SMOOTHING_ALIGNMENT,
    savgol_polyorder: int = 3,
    source_row_index: Optional[np.ndarray] = None,
    source_temporal_authority: Any = None,
) -> Tuple[Dict[int, Dict[str, Any]], List[Dict[str, float]]]:
    """Assemble per-track data arrays and summary statistics.

    Optionally applies hysteresis filtering to remove micro-jitter during speed computation.
    Optionally applies Savitzky-Golay smoothing for shape-preserving filtering.
    """

    source_row_indices = None
    source_instance_rows = None
    if source_row_index is not None:
        raw_source_rows = np.asarray(source_row_index)
        if (
            raw_source_rows.dtype.kind not in "iu"
            or raw_source_rows.ndim != 1
            or raw_source_rows.shape[0] != track_ids.shape[0]
        ):
            raise ValueError(
                "source_row_index must be a row-aligned one-dimensional integer array."
            )
        if raw_source_rows.size and (
            int(raw_source_rows.min()) < 0
            or int(raw_source_rows.max()) > np.iinfo(np.int64).max
        ):
            raise ValueError("source_row_index values must fit nonnegative int64.")
        source_row_indices = raw_source_rows.astype(np.int64, copy=False)
        if source_temporal_authority is None:
            raise ValueError(
                "source_row_index requires a sealed immediate-source temporal authority."
            )
        source_temporal_authority = require_bound_source_row_temporal_authority(
            source_temporal_authority
        )
        resolved_frames = resolve_source_acquisition_frame_indices(
            source_temporal_authority,
            source_row_indices,
        )
        provided_frames = np.asarray(frames)
        if provided_frames.dtype.kind not in "iu" or not np.array_equal(
            provided_frames.astype(np.int64, copy=False),
            resolved_frames,
        ):
            raise ValueError(
                "frames must exactly equal the immediate source acquisition-frame "
                "mapping selected by source_row_index."
            )
        source_instance_rows = derive_track_source_instance_values(
            source_temporal_authority,
            source_row_indices,
        )
    elif source_temporal_authority is not None:
        raise ValueError(
            "A source temporal authority cannot be supplied without source_row_index."
        )

    unique_ids = np.unique(track_ids)
    tracks: Dict[int, Dict[str, np.ndarray]] = {}
    summaries: List[Dict[str, float]] = []

    pixel_to_mm_val = pixel_to_mm if (pixel_to_mm is not None and pixel_to_mm > 0) else None

    for track_id in unique_ids:
        mask = track_ids == track_id
        if not np.any(mask):
            continue

        track_frames = frames[mask]
        coords_px = positions_px[mask]
        # Heading-derived public surfaces use float32.  Normalize once before
        # validity, radians, smoothing, turning, and summaries so the writer
        # seals against the same precision that it persists.
        headings_track = np.asarray(headings_deg[mask], dtype=np.float32)
        kp_success_track = keypoint_success[mask]
        det_source_track = (
            detection_source[mask].astype(np.int8)
            if detection_source is not None
            else np.zeros(mask.sum(), dtype=np.int8)
        )
        source_instance_track = (
            source_instance_rows[mask]
            if source_instance_rows is not None
            else None
        )
        source_rows_track = (
            source_row_indices[mask]
            if source_row_indices is not None
            else None
        )
        sample_validity = _build_sample_validity_arrays(
            track_id=int(track_id),
            positions_px=coords_px,
            headings_deg=headings_track,
            keypoint_success=kp_success_track,
            detection_source=det_source_track if detection_source is not None else None,
        )

        order = np.argsort(track_frames, kind="stable")
        track_frames = track_frames[order]
        coords_px = coords_px[order]
        headings_track = headings_track[order]
        kp_success_track = kp_success_track[order]
        det_source_track = det_source_track[order]
        if source_instance_track is not None:
            source_instance_track = source_instance_track[order]
        if source_rows_track is not None:
            source_rows_track = source_rows_track[order]
        sample_validity = {
            name: values[order]
            for name, values in sample_validity.items()
        }

        speeds = compute_track_speed(
            track_frames.copy(),
            coords_px.copy(),
            fps=fps,
            smooth_seconds=smooth_seconds,
            hysteresis_high_px=hysteresis_high_px,
            hysteresis_low_px=hysteresis_low_px,
            hysteresis_min_frames=hysteresis_min_frames,
            hysteresis_band_policy=hysteresis_band_policy,
            smoothing_method=smoothing_method,
            smoothing_alignment=smoothing_alignment,
            savgol_polyorder=savgol_polyorder,
        )

        # Freeze every public pixel payload before producing a physical peer.
        # This prevents independent float64 calculations from landing on a
        # different float32 ULP than persisted_px * persisted_scale.
        coords_px = np.array(coords_px, copy=True, order="C")
        speed_raw_px = _float32(speeds.speed_raw)
        speed_filtered_px = _float32(speeds.speed_filtered)
        speed_smoothed_px = _float32(speeds.speed_smoothed)
        speed_averaged_px = _float32(speeds.speed_averaged)
        frame_path_distance_raw_px = _float32(speeds.frame_path_distance_raw)
        frame_path_distance_filtered_px = _float32(
            speeds.frame_path_distance_filtered
        )
        frame_path_distance_smoothed_px = _float32(
            speeds.frame_path_distance_smoothed
        )
        cumulative_path_px = _float32(speeds.cumulative_path_distance)
        speed_per_second_px = _float32(speeds.speed_per_second)
        delta_frames = speeds.delta_frames
        delta_seconds = speeds.delta_seconds
        transition_valid = speeds.transition_valid
        transition_reason_code = speeds.transition_reason_code

        coords_mm = _physical_values_from_pixel_peer(coords_px, pixel_to_mm_val)
        speed_raw_mm = _physical_values_from_pixel_peer(
            speed_raw_px,
            pixel_to_mm_val,
        )
        speed_filtered_mm = _physical_values_from_pixel_peer(
            speed_filtered_px,
            pixel_to_mm_val,
        )
        speed_smoothed_mm = _physical_values_from_pixel_peer(
            speed_smoothed_px,
            pixel_to_mm_val,
        )
        speed_averaged_mm = _physical_values_from_pixel_peer(
            speed_averaged_px,
            pixel_to_mm_val,
        )
        frame_path_distance_raw_mm = _physical_values_from_pixel_peer(
            frame_path_distance_raw_px,
            pixel_to_mm_val,
        )
        frame_path_distance_filtered_mm = _physical_values_from_pixel_peer(
            frame_path_distance_filtered_px,
            pixel_to_mm_val,
        )
        frame_path_distance_smoothed_mm = _physical_values_from_pixel_peer(
            frame_path_distance_smoothed_px,
            pixel_to_mm_val,
        )
        cumulative_path_mm = _physical_values_from_pixel_peer(
            cumulative_path_px,
            pixel_to_mm_val,
        )
        speed_per_second_mm = _physical_values_from_pixel_peer(
            speed_per_second_px,
            pixel_to_mm_val,
        )

        heading_rad = np.deg2rad(headings_track)
        heading_valid = np.isfinite(heading_rad)
        time_seconds = track_frames.astype(np.float64) / fps
        seconds_per_frame = np.floor(time_seconds).astype(np.int64)

        delta_seconds_full = np.zeros(track_frames.shape[0], dtype=np.float64)
        if track_frames.size >= 2:
            delta_seconds_full[1:] = np.diff(track_frames) / fps

        speed_derivatives = _compute_speed_derivatives(
            {
                "speed_raw": speed_raw_px,
                "speed_filtered": speed_filtered_px,
                "speed_smoothed": speed_smoothed_px,
                "speed_averaged": speed_averaged_px,
            },
            delta_seconds_full,
            pixel_to_mm=pixel_to_mm_val,
            smooth_seconds=smooth_seconds,
            fps=fps,
        )
        default_speed_derivative = speed_derivatives[DEFAULT_ACCELERATION_SOURCE_SPEED_LEVEL]
        acceleration_px = _float32(default_speed_derivative["acceleration_px"])
        accel_mm = _float32(default_speed_derivative["acceleration_mm"])
        smoothed_accel_px = _float32(
            default_speed_derivative["smoothed_acceleration_px"]
        )
        smoothed_accel_mm = _float32(
            default_speed_derivative["smoothed_acceleration_mm"]
        )

        heading_window_requested = max(1, int(round(fps * smooth_seconds)))
        smoothed_heading_rad, heading_window_effective = (
            _smooth_heading_radians(headings_track, heading_window_requested)
        )

        smoothed_heading_deg = np.rad2deg(smoothed_heading_rad)

        delta_heading_degrees, angular_velocity_raw_deg_s, angular_speed_raw_deg_s = (
            _compute_heading_turning(
                headings_track,
                delta_seconds_full,
                transition_valid=transition_valid,
                sample_valid=sample_validity["sample_valid"],
            )
        )
        delta_heading_smoothed_degrees, angular_velocity_smoothed_deg_s, angular_speed_smoothed_deg_s = (
            _compute_heading_turning(
                smoothed_heading_deg,
                delta_seconds_full,
                transition_valid=transition_valid,
                sample_valid=sample_validity["sample_valid"],
            )
        )
        angular_velocity_deg_s = angular_velocity_raw_deg_s

        unique_seconds = speeds.seconds.astype(np.int64)
        # fallback if TrackSpeeds.seconds is empty
        if unique_seconds.size == 0 and seconds_per_frame.size > 0:
            unique_seconds = np.unique(seconds_per_frame)
        heading_per_second_rad = np.full(unique_seconds.size, np.nan, dtype=np.float64)
        heading_per_second_resultant = np.zeros(unique_seconds.size, dtype=np.float32)
        for idx, sec in enumerate(unique_seconds):
            mask_sec = (seconds_per_frame == sec) & heading_valid
            valid_angles = heading_rad[mask_sec]
            if valid_angles.size:
                mean_vector = np.mean(np.exp(1j * valid_angles))
                heading_per_second_rad[idx] = math.atan2(mean_vector.imag, mean_vector.real)
                heading_per_second_resultant[idx] = np.float32(np.abs(mean_vector))
        heading_per_second_deg = np.rad2deg(heading_per_second_rad)

        track_sample_key = build_track_sample_key(
            np.full(track_frames.shape, int(track_id), dtype=np.int64),
            track_frames,
        )
        if np.unique(track_sample_key, axis=0).shape[0] != track_sample_key.shape[0]:
            raise ValueError(
                "Track samples contain duplicate "
                "(track_id, acquisition_frame_index) identities."
            )

        if source_rows_track is not None:
            assert source_instance_track is not None
            source_instance_lineage = np.array(source_instance_track, copy=True)
            persisted_interpolation = np.zeros(
                track_frames.shape,
                dtype=TRACK_SAMPLE_INTERPOLATION_DTYPE,
            )
            persisted_interpolation["left_source_frame_index"] = track_frames
            persisted_interpolation["right_source_frame_index"] = track_frames
            persisted_interpolation["right_weight"] = 0.0
        else:
            source_instance_lineage = None
            persisted_interpolation = None

        tracks[int(track_id)] = {
            "frame_indices": track_frames.astype(np.int64),
            "track_sample_key": track_sample_key,
            "source_acquisition_frame_index": track_frames.astype(np.int64),
            **(
                {"source_frame_interpolation": persisted_interpolation}
                if persisted_interpolation is not None
                else {}
            ),
            **(
                {"source_instance_key": source_instance_lineage}
                if source_instance_lineage is not None
                else {}
            ),
            "time_seconds": _float32(time_seconds),
            **(
                {"source_row_index": source_rows_track}
                if source_rows_track is not None
                else {}
            ),
            # Coordinate publication proves an exact dtype-preserving
            # subset/reorder. Kinematic derivatives may use compact float32,
            # but the authoritative positions must retain source precision.
            "positions_px": np.array(coords_px, copy=True, order="C"),
            "positions_mm": np.array(coords_mm, copy=True, order="C"),
            "heading_degrees": _float32(headings_track),
            "heading_radians": _float32(heading_rad),
            "delta_heading_degrees": _float32(delta_heading_degrees),
            "angular_velocity_deg_s": _float32(angular_velocity_deg_s),
            "angular_velocity_raw_deg_s": _float32(angular_velocity_raw_deg_s),
            "angular_speed_raw_deg_s": _float32(angular_speed_raw_deg_s),
            "delta_heading_smoothed_degrees": _float32(delta_heading_smoothed_degrees),
            "angular_velocity_smoothed_deg_s": _float32(angular_velocity_smoothed_deg_s),
            "angular_speed_smoothed_deg_s": _float32(angular_speed_smoothed_deg_s),
            "smoothed_heading_degrees": _float32(smoothed_heading_deg),
            "smoothed_heading_radians": _float32(smoothed_heading_rad),
            "keypoint_success": _boolean(kp_success_track),
            "detection_source": det_source_track.astype(np.int8),
            "sample_observed": _boolean(sample_validity["sample_observed"]),
            "sample_valid": _boolean(sample_validity["sample_valid"]),
            "source_observed": _boolean(sample_validity["source_observed"]),
            "keypoint_usable": _boolean(sample_validity["keypoint_usable"]),
            "position_finite": _boolean(sample_validity["position_finite"]),
            "heading_usable": _boolean(sample_validity["heading_usable"]),
            "sample_reason_code": sample_validity["sample_reason_code"].astype(np.int16),
            "delta_frames": delta_frames.astype(np.int32),
            "delta_seconds": _float32(delta_seconds),
            "transition_valid": _boolean(transition_valid),
            "transition_reason_code": transition_reason_code.astype(np.int16),
            "speed_raw_px": _float32(speed_raw_px),
            "speed_raw_mm": _float32(speed_raw_mm),
            "speed_filtered_px": _float32(speed_filtered_px),
            "speed_filtered_mm": _float32(speed_filtered_mm),
            "speed_smoothed_px": _float32(speed_smoothed_px),
            "speed_smoothed_mm": _float32(speed_smoothed_mm),
            "speed_averaged_px": _float32(speed_averaged_px),
            "speed_averaged_mm": _float32(speed_averaged_mm),
            "acceleration_px": _float32(acceleration_px),
            "acceleration_mm": _float32(accel_mm),
            "smoothed_acceleration_px": _float32(smoothed_accel_px),
            "smoothed_acceleration_mm": _float32(smoothed_accel_mm),
            "speed_derivatives": speed_derivatives,
            "frame_path_distance_raw_px": _float32(frame_path_distance_raw_px),
            "frame_path_distance_raw_mm": _float32(frame_path_distance_raw_mm),
            "frame_path_distance_filtered_px": _float32(frame_path_distance_filtered_px),
            "frame_path_distance_filtered_mm": _float32(frame_path_distance_filtered_mm),
            "frame_path_distance_smoothed_px": _float32(frame_path_distance_smoothed_px),
            "frame_path_distance_smoothed_mm": _float32(frame_path_distance_smoothed_mm),
            "cumulative_path_distance_px": _float32(cumulative_path_px),
            "cumulative_path_distance_mm": _float32(cumulative_path_mm),
            # The second-domain arrays below have one row per unique elapsed
            # second, not one row per track sample.  Persist their exact key
            # vector rather than the per-sample floor(time_seconds) mapping.
            "second_indices": unique_seconds,
            "speed_per_second_px": _float32(speed_per_second_px),
            "speed_per_second_mm": _float32(speed_per_second_mm),
            "heading_per_second_degrees": _float32(heading_per_second_deg),
            "heading_per_second_resultant": heading_per_second_resultant.astype(np.float32),
            "motion_smoothing_windows": {
                "schema_id": "palette.track_motion_smoothing_windows",
                "schema_version": 1,
                "distance_transition": {
                    "alignment": smoothing_alignment,
                    "requested_frames": int(
                        speeds.distance_smoothing_window_frames_requested
                    ),
                    "effective_frames": int(
                        speeds.distance_smoothing_window_frames_effective
                    ),
                },
                "speed_sample": {
                    "alignment": smoothing_alignment,
                    "requested_frames": int(
                        speeds.speed_smoothing_window_frames_requested
                    ),
                    "effective_frames": int(
                        speeds.speed_smoothing_window_frames_effective
                    ),
                },
                "acceleration_sample": {
                    "alignment": "centered",
                    "requested_frames": int(
                        default_speed_derivative[
                            "post_smoothing_window_frames_requested"
                        ]
                    ),
                    "effective_frames": int(
                        default_speed_derivative[
                            "post_smoothing_window_frames_effective"
                        ]
                    ),
                },
                "heading_sample": {
                    "alignment": "centered",
                    "requested_frames": int(heading_window_requested),
                    "effective_frames": int(heading_window_effective),
                },
            },
        }

        # Speed metrics for all processing levels
        # Raw speed (validity filtering only)
        finite_raw = speed_raw_px[np.isfinite(speed_raw_px)]
        mean_speed_raw_px = float(np.mean(finite_raw)) if finite_raw.size else float("nan")
        median_speed_raw_px = float(np.median(finite_raw)) if finite_raw.size else float("nan")
        max_speed_raw_px = float(np.max(finite_raw)) if finite_raw.size else float("nan")
        mean_speed_raw_mm = mean_speed_raw_px * pixel_to_mm_val if pixel_to_mm_val is not None and np.isfinite(mean_speed_raw_px) else float("nan")
        median_speed_raw_mm = median_speed_raw_px * pixel_to_mm_val if pixel_to_mm_val is not None and np.isfinite(median_speed_raw_px) else float("nan")
        max_speed_raw_mm = max_speed_raw_px * pixel_to_mm_val if pixel_to_mm_val is not None and np.isfinite(max_speed_raw_px) else float("nan")

        # Filtered speed (hysteresis applied)
        finite_filtered = speed_filtered_px[np.isfinite(speed_filtered_px)]
        mean_speed_filtered_px = float(np.mean(finite_filtered)) if finite_filtered.size else float("nan")
        median_speed_filtered_px = float(np.median(finite_filtered)) if finite_filtered.size else float("nan")
        max_speed_filtered_px = float(np.max(finite_filtered)) if finite_filtered.size else float("nan")
        mean_speed_filtered_mm = mean_speed_filtered_px * pixel_to_mm_val if pixel_to_mm_val is not None and np.isfinite(mean_speed_filtered_px) else float("nan")
        median_speed_filtered_mm = median_speed_filtered_px * pixel_to_mm_val if pixel_to_mm_val is not None and np.isfinite(median_speed_filtered_px) else float("nan")
        max_speed_filtered_mm = max_speed_filtered_px * pixel_to_mm_val if pixel_to_mm_val is not None and np.isfinite(max_speed_filtered_px) else float("nan")

        # Smoothed speed (temporal smoothing applied)
        finite_smoothed = speed_smoothed_px[np.isfinite(speed_smoothed_px)]
        mean_speed_smoothed_px = float(np.mean(finite_smoothed)) if finite_smoothed.size else float("nan")
        median_speed_smoothed_px = float(np.median(finite_smoothed)) if finite_smoothed.size else float("nan")
        max_speed_smoothed_px = float(np.max(finite_smoothed)) if finite_smoothed.size else float("nan")
        mean_speed_smoothed_mm = mean_speed_smoothed_px * pixel_to_mm_val if pixel_to_mm_val is not None and np.isfinite(mean_speed_smoothed_px) else float("nan")
        median_speed_smoothed_mm = median_speed_smoothed_px * pixel_to_mm_val if pixel_to_mm_val is not None and np.isfinite(median_speed_smoothed_px) else float("nan")
        max_speed_smoothed_mm = max_speed_smoothed_px * pixel_to_mm_val if pixel_to_mm_val is not None and np.isfinite(max_speed_smoothed_px) else float("nan")

        # Averaged speed (further temporal averaging)
        finite_averaged = speed_averaged_px[np.isfinite(speed_averaged_px)]
        mean_speed_averaged_px = float(np.mean(finite_averaged)) if finite_averaged.size else float("nan")
        median_speed_averaged_px = float(np.median(finite_averaged)) if finite_averaged.size else float("nan")
        max_speed_averaged_px = float(np.max(finite_averaged)) if finite_averaged.size else float("nan")
        mean_speed_averaged_mm = mean_speed_averaged_px * pixel_to_mm_val if pixel_to_mm_val is not None and np.isfinite(mean_speed_averaged_px) else float("nan")
        median_speed_averaged_mm = median_speed_averaged_px * pixel_to_mm_val if pixel_to_mm_val is not None and np.isfinite(median_speed_averaged_px) else float("nan")
        max_speed_averaged_mm = max_speed_averaged_px * pixel_to_mm_val if pixel_to_mm_val is not None and np.isfinite(max_speed_averaged_px) else float("nan")

        # Frame path-distance totals for each processing level.
        total_path_distance_raw_px = (
            float(np.sum(frame_path_distance_raw_px)) if frame_path_distance_raw_px.size else 0.0
        )
        total_path_distance_raw_mm = (
            total_path_distance_raw_px * pixel_to_mm_val
            if pixel_to_mm_val is not None
            else float("nan")
        )

        total_path_distance_filtered_px = (
            float(np.sum(frame_path_distance_filtered_px)) if frame_path_distance_filtered_px.size else 0.0
        )
        total_path_distance_filtered_mm = (
            total_path_distance_filtered_px * pixel_to_mm_val
            if pixel_to_mm_val is not None
            else float("nan")
        )

        total_path_distance_smoothed_px = (
            float(np.sum(frame_path_distance_smoothed_px)) if frame_path_distance_smoothed_px.size else 0.0
        )
        total_path_distance_smoothed_mm = (
            total_path_distance_smoothed_px * pixel_to_mm_val
            if pixel_to_mm_val is not None
            else float("nan")
        )

        # Cumulative path distance (from smoothed frame path-distance).
        total_distance_px = float(cumulative_path_px[-1]) if cumulative_path_px.size else 0.0
        total_distance_mm = (
            total_distance_px * pixel_to_mm_val
            if pixel_to_mm_val is not None
            else float("nan")
        )

        mean_speed_per_second_px = float(np.nanmean(speed_per_second_px)) if speed_per_second_px.size else float("nan")
        mean_speed_per_second_mm = (
            mean_speed_per_second_px * pixel_to_mm_val
            if pixel_to_mm_val is not None and np.isfinite(mean_speed_per_second_px)
            else float("nan")
        )

        valid_heading = heading_rad[np.isfinite(heading_rad)]
        if valid_heading.size:
            mean_vector = np.mean(np.exp(1j * valid_heading))
            heading_mean_deg = float(math.degrees(math.atan2(mean_vector.imag, mean_vector.real)))
            heading_consistency = float(np.abs(mean_vector))
        else:
            heading_mean_deg = float("nan")
            heading_consistency = float("nan")

        accel_finite = smoothed_accel_px[np.isfinite(smoothed_accel_px)]
        mean_accel_px = float(np.mean(accel_finite)) if accel_finite.size else float("nan")
        accel_std_px = float(np.std(accel_finite)) if accel_finite.size else float("nan")
        mean_accel_mm = mean_accel_px * pixel_to_mm_val if pixel_to_mm_val is not None and np.isfinite(mean_accel_px) else float("nan")
        accel_std_mm = accel_std_px * pixel_to_mm_val if pixel_to_mm_val is not None and np.isfinite(accel_std_px) else float("nan")

        summary = {
            "track_id": float(track_id),
            "samples": int(track_frames.size),
            # Raw speed metrics (validity filtering only)
            "mean_speed_raw_px": mean_speed_raw_px,
            "median_speed_raw_px": median_speed_raw_px,
            "max_speed_raw_px": max_speed_raw_px,
            "mean_speed_raw_mm": mean_speed_raw_mm,
            "median_speed_raw_mm": median_speed_raw_mm,
            "max_speed_raw_mm": max_speed_raw_mm,
            # Filtered speed (hysteresis applied)
            "mean_speed_filtered_px": mean_speed_filtered_px,
            "median_speed_filtered_px": median_speed_filtered_px,
            "max_speed_filtered_px": max_speed_filtered_px,
            "mean_speed_filtered_mm": mean_speed_filtered_mm,
            "median_speed_filtered_mm": median_speed_filtered_mm,
            "max_speed_filtered_mm": max_speed_filtered_mm,
            # Smoothed speed (temporal smoothing applied)
            "mean_speed_smoothed_px": mean_speed_smoothed_px,
            "median_speed_smoothed_px": median_speed_smoothed_px,
            "max_speed_smoothed_px": max_speed_smoothed_px,
            "mean_speed_smoothed_mm": mean_speed_smoothed_mm,
            "median_speed_smoothed_mm": median_speed_smoothed_mm,
            "max_speed_smoothed_mm": max_speed_smoothed_mm,
            # Averaged speed (further temporal averaging)
            "mean_speed_averaged_px": mean_speed_averaged_px,
            "median_speed_averaged_px": median_speed_averaged_px,
            "max_speed_averaged_px": max_speed_averaged_px,
            "mean_speed_averaged_mm": mean_speed_averaged_mm,
            "median_speed_averaged_mm": median_speed_averaged_mm,
            "max_speed_averaged_mm": max_speed_averaged_mm,
            # Speed per second
            "mean_speed_per_second_px": mean_speed_per_second_px,
            "mean_speed_per_second_mm": mean_speed_per_second_mm,
            # Path-distance totals
            "total_path_distance_raw_px": total_path_distance_raw_px,
            "total_path_distance_raw_mm": total_path_distance_raw_mm,
            "total_path_distance_filtered_px": total_path_distance_filtered_px,
            "total_path_distance_filtered_mm": total_path_distance_filtered_mm,
            "total_path_distance_smoothed_px": total_path_distance_smoothed_px,
            "total_path_distance_smoothed_mm": total_path_distance_smoothed_mm,
            # Cumulative path distance
            "total_distance_px": total_distance_px,
            "total_distance_mm": total_distance_mm,
            # Heading
            "heading_mean_deg": heading_mean_deg,
            "heading_resultant": heading_consistency,
            # Acceleration
            "mean_acceleration_px": mean_accel_px,
            "mean_acceleration_mm": mean_accel_mm,
            "acceleration_std_px": accel_std_px,
            "acceleration_std_mm": accel_std_mm,
            # Other
            "keypoint_success_rate": float(np.mean(kp_success_track)) if kp_success_track.size else float("nan"),
            "duration_seconds": float(time_seconds[-1] - time_seconds[0]) if time_seconds.size > 1 else 0.0,
        }
        summaries.append(summary)

    return tracks, summaries


_TRACK_STAGING_CRITICAL_ARRAYS = (
    "frame_indices",
    "track_sample_key",
    "source_acquisition_frame_index",
    "source_frame_interpolation",
    "source_instance_key",
    "source_row_index",
    "positions_px",
)


def _iter_track_array_nodes(
    group: Any,
    *,
    prefix: str = "",
) -> Iterable[tuple[str, Any]]:
    array_keys = getattr(group, "array_keys", None)
    group_keys = getattr(group, "group_keys", None)
    if not callable(array_keys) or not callable(group_keys):
        raise ValueError(f"/{group.path} is not a traversable persisted group.")
    for name in sorted(str(value) for value in array_keys()):
        relative = f"{prefix}/{name}" if prefix else name
        yield relative, group[name]
    for name in sorted(str(value) for value in group_keys()):
        relative = f"{prefix}/{name}" if prefix else name
        yield from _iter_track_array_nodes(group[name], prefix=relative)


def _is_physical_track_array_path(relative_path: str) -> bool:
    leaf = relative_path.rsplit("/", 1)[-1]
    return leaf == "mm" or leaf.endswith("_mm")


def _track_physical_array_nodes(group: Any) -> dict[str, Any]:
    return {
        path: node
        for path, node in _iter_track_array_nodes(group)
        if _is_physical_track_array_path(path)
    }


def _iter_mapping_physical_arrays(
    value: Mapping[str, Any],
    *,
    prefix: str = "",
) -> Iterable[tuple[str, Mapping[str, Any], str, Any]]:
    """Yield every in-memory array whose leaf declares physical mm units."""

    for raw_name, child in value.items():
        name = str(raw_name)
        relative_path = f"{prefix}/{name}" if prefix else name
        if isinstance(child, Mapping):
            yield from _iter_mapping_physical_arrays(
                child,
                prefix=relative_path,
            )
        elif _is_physical_track_array_path(relative_path):
            yield relative_path, value, name, child


def _physical_to_pixel_leaf(name: str) -> str:
    if name == "mm":
        return "px"
    if name.endswith("_mm"):
        return f"{name[:-3]}_px"
    raise ValueError(f"Physical field {name!r} has no exact pixel-pair rule.")


def _validate_in_memory_track_physical_arrays(
    data: Mapping[str, Any],
    *,
    track_id: int,
    physical_authority: TrackPhysicalAuthority,
) -> None:
    """Validate every prospective mm array before mutating the destination."""

    records = list(_iter_mapping_physical_arrays(data))
    paths = {path for path, _, _, _ in records}
    expected_paths: set[str] = set()
    stack: list[tuple[str, Mapping[str, Any]]] = [("", data)]
    while stack:
        prefix, parent = stack.pop()
        for raw_name, child in parent.items():
            name = str(raw_name)
            relative_path = f"{prefix}/{name}" if prefix else name
            if isinstance(child, Mapping):
                stack.append((relative_path, child))
            elif name == "px" or name.endswith("_px"):
                physical_name = "mm" if name == "px" else f"{name[:-3]}_mm"
                expected_paths.add(
                    f"{prefix}/{physical_name}" if prefix else physical_name
                )
    if paths != expected_paths or "positions_mm" not in paths:
        raise ValueError(
            f"Track {track_id} physical array inventory is incomplete or "
            f"unexpected (expected={sorted(expected_paths)!r}, "
            f"found={sorted(paths)!r})."
        )
    for relative_path, parent, name, physical_value in records:
        pixel_name = _physical_to_pixel_leaf(name)
        if pixel_name not in parent:
            raise ValueError(
                f"Track {track_id} physical array {relative_path!r} lacks exact "
                f"pixel pair {pixel_name!r}."
            )
        physical_values = np.asarray(physical_value)
        pixel_values = np.asarray(parent[pixel_name])
        if (
            physical_values.dtype.hasobject
            or pixel_values.dtype.hasobject
            or physical_values.dtype != pixel_values.dtype
            or physical_values.shape != pixel_values.shape
        ):
            raise ValueError(
                f"Track {track_id} physical array {relative_path!r} differs in "
                "dtype/shape from its exact pixel pair."
            )
        scale = np.asarray(
            physical_authority.mm_per_pixel,
            dtype=pixel_values.dtype,
        )
        expected = np.asarray(pixel_values * scale, dtype=pixel_values.dtype)
        if not np.array_equal(physical_values, expected, equal_nan=True):
            raise ValueError(
                f"Track {track_id} physical array {relative_path!r} does not use "
                "the exact source-camera mm_per_pixel authority."
            )


def _is_mm_summary_field(name: object) -> bool:
    return str(name).endswith("_mm")


def _summary_number(value: Any, *, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be numeric or null, not boolean.")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric or null.") from exc


def _validate_scaled_scalar_pair(
    pixel_value: Any,
    physical_value: Any,
    *,
    mm_per_pixel: float,
    label: str,
) -> None:
    pixel = _summary_number(pixel_value, label=f"{label} pixel value")
    physical = _summary_number(
        physical_value,
        label=f"{label} physical value",
    )
    if pixel is None or not math.isfinite(pixel):
        if physical is not None and math.isfinite(physical):
            raise ValueError(
                f"{label} physical value is finite while its pixel value is not."
            )
        return
    if physical is None or not math.isfinite(physical):
        raise ValueError(
            f"{label} physical value is missing/nonfinite for a finite pixel value."
        )
    if physical != pixel * float(mm_per_pixel):
        raise ValueError(
            f"{label} does not use the exact source-camera mm_per_pixel authority."
        )


def _validate_track_summary_physical_fields(
    summary: Mapping[str, Any],
    *,
    physical_authority: TrackPhysicalAuthority | None,
    label: str,
) -> None:
    mm_fields = {str(name) for name in summary if _is_mm_summary_field(name)}
    if physical_authority is None:
        if mm_fields:
            raise ValueError(
                f"{label} retains omitted physical summary fields: "
                f"{sorted(mm_fields)!r}."
            )
        return
    expected_mm_fields = {
        f"{str(name)[:-3]}_mm"
        for name in summary
        if str(name).endswith("_px")
    }
    if mm_fields != expected_mm_fields or "total_distance_mm" not in mm_fields:
        raise ValueError(
            f"{label} physical summary inventory is incomplete or unexpected "
            f"(expected={sorted(expected_mm_fields)!r}, "
            f"found={sorted(mm_fields)!r})."
        )
    for mm_name in sorted(mm_fields):
        px_name = f"{mm_name[:-3]}_px"
        _validate_scaled_scalar_pair(
            summary[px_name],
            summary[mm_name],
            mm_per_pixel=physical_authority.mm_per_pixel,
            label=f"{label}.{mm_name}",
        )


def _persisted_track_summary(
    summary: Mapping[str, Any],
    *,
    include_physical: bool,
) -> dict[str, Any]:
    payload = (
        dict(summary)
        if include_physical
        else {
            str(name): value
            for name, value in summary.items()
            if not _is_mm_summary_field(name)
        }
    )
    normalized = json_attr_safe(payload)
    if not isinstance(normalized, dict):
        raise ValueError("Track summary did not normalize to a JSON object.")
    return normalized


def _finite_summary_total(
    summaries: Iterable[Mapping[str, Any]],
    field: str,
) -> float:
    total = 0.0
    for index, summary in enumerate(summaries):
        value = _summary_number(
            summary.get(field),
            label=f"run summary[{index}].{field}",
        )
        if value is None or not math.isfinite(value):
            continue
        total += value
    return float(total)


def _run_physical_surface_record(run_group: Any) -> dict[str, Any]:
    attrs = run_group.attrs
    return {
        "summary": copy.deepcopy(attrs.get("summary")),
        "track_manifest": copy.deepcopy(attrs.get("track_manifest")),
        "total_distance_px": copy.deepcopy(attrs.get("total_distance_px")),
        "total_distance_mm": {
            "present": "total_distance_mm" in attrs,
            "value": copy.deepcopy(attrs.get("total_distance_mm")),
        },
    }


def _validate_no_run_root_coordinate_arrays(run_group: Any) -> None:
    """Reject untyped legacy geometry/distance mirrors at the run root."""

    array_keys = getattr(run_group, "array_keys", None)
    if not callable(array_keys):
        raise ValueError("Track run does not expose a persisted root-array inventory.")
    forbidden = sorted(
        str(name)
        for name in array_keys()
        if str(name) in {"px", "mm"}
        or str(name).endswith("_px")
        or str(name).endswith("_mm")
    )
    if forbidden:
        raise ValueError(
            "Track run root contains unsupported untyped coordinate/distance arrays: "
            f"{forbidden!r}. Coordinate-bearing data must live in a typed, "
            "row-identified surface."
        )


def _physical_authority_manifest_record(
    value: TrackPhysicalAuthority | None,
) -> dict[str, Any] | None:
    if value is None:
        return None
    if type(value) is BoundStimulusPhysicalCoordinateAuthority:
        authority = require_bound_stimulus_physical_coordinate_authority(value)
        selector = {"stimulus_run": authority.stimulus_run}
    else:
        authority = require_bound_source_camera_physical_authority(value)
        selector = {
            "authority_kind": "recording_calibration",
            "recording_calibration": True,
        }
    physical = authority.physical_frame
    return {
        **selector,
        "camera_id": authority.camera_id,
        "authority_manifest_ref": authority.manifest.record_ref,
        "authority_manifest_sha256": authority.manifest.record_sha256,
        "physical_frame_ref": physical.record_ref,
        "physical_frame_sha256": physical.record_sha256,
        "selected_camera_evidence_ref": (
            physical.selected_camera_evidence.record_ref
        ),
        "selected_camera_evidence_sha256": (
            physical.selected_camera_evidence.record_sha256
        ),
        "source_camera_frame_ref": physical.source_camera_pixels.record_ref,
        "source_camera_frame_sha256": (
            physical.source_camera_pixels.record_sha256
        ),
        "mm_per_pixel": float(physical.record.mm_per_pixel),
    }


def _canonical_json_sha256(value: Mapping[str, Any]) -> str:
    try:
        payload = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Track staging manifest is not strict canonical JSON: {exc}.") from exc
    return hashlib.sha256(payload).hexdigest()


def _archive_identity_manifest_record(node: Any) -> dict[str, Any]:
    identity = archive_identity(node)
    key: list[str | int] = []
    for item in identity.key:
        if type(item) not in {str, int}:
            raise ValueError(
                "Track staging supports only stable string/integer archive identity keys."
            )
        key.append(item)
    return {"kind": identity.kind, "key": key}


def _stage_array_payload_record(
    node: Any,
    *,
    relative_ref: str,
    include_attrs: bool = False,
) -> dict[str, Any]:
    dtype = np.dtype(getattr(node, "dtype"))
    shape = tuple(int(item) for item in getattr(node, "shape"))
    if dtype.hasobject:
        raise ValueError(f"/{node.path} uses an unsupported object dtype.")
    dtype_fields = None
    if dtype.fields is not None:
        dtype_fields = [
            {
                "name": str(name),
                "dtype": np.dtype(field[0]).str,
                "offset": int(field[1]),
            }
            for name, field in dtype.fields.items()
        ]
    record = {
        "relative_ref": relative_ref,
        "dtype": dtype.str,
        "dtype_fields": dtype_fields,
        "itemsize": int(dtype.itemsize),
        "shape": [int(item) for item in shape],
        "content_sha256": array_payload_sha256(node),
    }
    if include_attrs:
        attrs = _motion_json_object(
            dict(getattr(node, "attrs", {})),
            label=f"/{node.path} array attrs",
        )
        record["attrs"] = attrs
        record["attrs_sha256"] = _canonical_json_sha256(attrs)
    return record


_MOTION_RUN_DERIVATION_ATTR_NAMES = (
    "schema_id",
    "schema_version",
    "method",
    "method_version",
    "row_axis",
    "parameters",
    "source_refs",
    "inputs",
    TRACK_MOTION_INPUT_AUTHORITY_ATTR,
    "provenance",
    "run_provenance",
    "fps",
    "smoothing_seconds",
    "smoothing_method",
    "smoothing_alignment",
    "savgol_polyorder",
    "hysteresis_enabled",
    "hysteresis_high_px",
    "hysteresis_low_px",
    "hysteresis_min_frames",
    "hysteresis_band_policy",
    "distance_interpolation_seconds",
    "physical_outputs_status",
    "physical_outputs_reason_code",
    "physical_coordinate_authority",
    "summary",
    "track_manifest",
    "total_distance_px",
    "total_distance_mm",
    "source_fingerprint",
    "source_lineage_hash",
    "lineage_hash",
    "fingerprint_status",
    "lineage_fingerprint_schema_id",
    "lineage_fingerprint_schema_version",
    "lineage_fingerprint_canonicalization",
    "lineage_payload_json",
)

_MOTION_RUN_LEGACY_COMPATIBILITY_ATTR_NAMES = frozenset(
    {
        "created_at_utc",
        "coordinate_space",
        "positions_px_source_path",
        "positions_px_source_coordinate_descriptor_sha256",
        "num_tracks",
        "source_zarr",
        "output_zarr",
    }
)
_MOTION_RUN_STORAGE_ATTR_NAMES = frozenset(
    {
        "chunk_policy_version",
        "storage_profile_class",
        "storage_profile_id",
        "storage_profile_reason",
        "storage_profile_row_chunk",
        "node_local_materialization",
        "physical_storage_layout",
    }
)
_MOTION_RUN_LIFECYCLE_ATTR_NAMES = frozenset(
    {
        "palette_run_completion_contract",
        "palette_run_completion_status",
        "palette_run_started_at_utc",
        "palette_run_completed_at_utc",
        "palette_run_name",
        "palette_run_stage",
        "run_provenance",
        TRACK_KINEMATICS_PUBLICATION_OWNER_ATTR,
    }
)
_MOTION_RUN_PUBLICATION_DYNAMIC_ATTR_NAMES = frozenset(
    {
        "stage_selector_eligible",
        TRACK_MOTION_PUBLICATION_MANIFEST_ATTR,
        TRACK_MOTION_PUBLICATION_MANIFEST_DIGEST_ATTR,
        TRACK_MOTION_PUBLICATION_COMMIT_ATTR,
        # The atomic materializer updates this operational receipt after the
        # scientific publication has been sealed and validated.  It is not a
        # scientific or coordinate authority and therefore must remain outside
        # the immutable motion manifest.
        "cluster_output_staging",
    }
)
_MOTION_RUN_ALLOWED_ATTR_NAMES = frozenset(_MOTION_RUN_DERIVATION_ATTR_NAMES) | (
    _MOTION_RUN_LEGACY_COMPATIBILITY_ATTR_NAMES
    | _MOTION_RUN_STORAGE_ATTR_NAMES
    | _MOTION_RUN_LIFECYCLE_ATTR_NAMES
    | frozenset(
        {
            TRACK_KINEMATICS_STAGING_MANIFEST_ATTR,
            TRACK_KINEMATICS_STAGING_MANIFEST_DIGEST_ATTR,
            TRACK_KINEMATICS_COORDINATE_BINDING_STATUS_ATTR,
        }
    )
)

_MOTION_ARRAY_DUPLICATE_SEMANTIC_ATTR_NAMES = frozenset(
    {
        "authority_scope",
        "units",
        "axis0_domain",
        "semantic_profile",
        "operation_id",
        "input_refs",
        "alias_of",
        "transition_anchor",
        "pixel_source_ref",
        "physical_authority_sha256",
        "physical_value_comparison",
    }
)
_MOTION_ARRAY_FORBIDDEN_COORDINATE_ATTR_NAMES = frozenset(
    {
        "coordinate_space",
        "space_id",
        "origin",
        "positive_x_direction",
        "positive_y_direction",
        "reference_width",
        "reference_height",
        "pixel_convention",
        "transform_direction",
    }
)
_MOTION_ARRAY_STORAGE_ATTR_NAMES = frozenset(
    {
        "chunk_policy_version",
        "storage_profile_class",
        "storage_profile_id",
        "storage_profile_reason",
        "storage_profile_row_chunk",
    }
)
_MOTION_ARRAY_POSITION_ATTR_NAMES = frozenset(
    {
        "coordinate_descriptor",
        "coordinate_descriptor_owner_dtype",
        "coordinate_descriptor_sha256",
    }
)
_MOTION_ARRAY_IDENTITY_ATTR_NAMES = frozenset(
    {
        "row_identity_contract_ref",
        "row_identity_contract_sha256",
        "row_identity_key",
        "row_identity_key_sha256",
        "semantic_role",
        "authoritative_array_ref",
        "canonical_consumers_must_use",
        "source_identity_domain",
        "nullable_target_domain",
        "primary_row_identity",
        "null_encoding",
    }
)
_MOTION_ARRAY_ALLOWED_ATTR_NAMES = (
    _MOTION_ARRAY_STORAGE_ATTR_NAMES
    | _MOTION_ARRAY_POSITION_ATTR_NAMES
    | _MOTION_ARRAY_IDENTITY_ATTR_NAMES
    | _MOTION_ARRAY_DUPLICATE_SEMANTIC_ATTR_NAMES
)

_MOTION_ARRAY_EXTRA_ATTR_NAMES_BY_PATH = {
    "frame_indices": frozenset(
        {
            "semantic_role",
            "authoritative_array_ref",
            "canonical_consumers_must_use",
        }
    ),
    "source_instance_key": frozenset(
        {
            "semantic_role",
            "source_identity_domain",
            "nullable_target_domain",
            "primary_row_identity",
            "null_encoding",
        }
    ),
    "track_sample_key": frozenset(
        {
            "row_identity_contract_ref",
            "row_identity_contract_sha256",
            "row_identity_key",
            "row_identity_key_sha256",
        }
    ),
    "positions_px": _MOTION_ARRAY_POSITION_ATTR_NAMES,
    "positions_mm": _MOTION_ARRAY_POSITION_ATTR_NAMES,
}

_MOTION_TRACK_GROUP_STORAGE_ATTR_NAMES = _MOTION_ARRAY_STORAGE_ATTR_NAMES
_MOTION_TRACK_ROOT_GROUP_ATTR_NAMES = frozenset(
    {
        "track_id",
        "arena_id",
        "num_samples",
        "sample_validity_schema_id",
        "sample_reason_codes",
        "transition_validity_schema_id",
        "transition_reason_codes",
        "motion_smoothing_windows",
        "summary",
        "physical_outputs_status",
        "physical_outputs_reason_code",
        "physical_coordinate_authority",
        "legacy_acceleration_source_speed_level",
        "speed_derivatives_schema_id",
        "row_identity_contract",
        "row_identity_contract_sha256",
        "track_sample_time_lineage",
        "track_sample_time_lineage_sha256",
        "track_position_derivation",
        "track_position_derivation_sha256",
    }
)
_MOTION_SWIM_BOUT_ROOT_ATTR_NAMES = frozenset(
    {
        "source_swim_bout_run",
        "source_track_kinematics_run",
        "source_swim_bout_track_id",
        "source_swim_bout_candidate_id",
        "source_swim_bout_default_signal_id",
        "mirror_scope",
        "default_level",
        "layout",
        "is_hierarchical",
        "mirrored_fields",
    }
)
_MOTION_SWIM_BOUT_LEVEL_ATTR_NAMES = frozenset(
    {
        "speed_level",
        "signal_id",
        "signal_name",
        "signal_role",
        "signal_source_level",
        "source_swim_bout_path",
        "n_bouts",
        "mirrored_fields",
    }
)


def _freeze_motion_manifest(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                str(name): _freeze_motion_manifest(child)
                for name, child in value.items()
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_motion_manifest(child) for child in value)
    return copy.deepcopy(value)


def _thaw_motion_manifest(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(name): _thaw_motion_manifest(child)
            for name, child in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_thaw_motion_manifest(child) for child in value]
    return copy.deepcopy(value)


def _motion_json_object(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    normalized = json_attr_safe(copy.deepcopy(dict(value)))
    if type(normalized) is not dict:
        raise ValueError(f"{label} did not normalize to one strict JSON object.")
    _canonical_json_sha256(normalized)
    return normalized


def _motion_group_attrs_sha256(group: Any) -> str:
    return str(_motion_group_attrs_record(group)["attrs_sha256"])


def _motion_group_attrs_record(group: Any) -> dict[str, Any]:
    attrs = _motion_json_object(
        dict(group.attrs),
        label=f"/{group.path} attrs",
    )
    return {
        "attrs": attrs,
        "attrs_sha256": _canonical_json_sha256(attrs),
    }


def _iter_motion_group_nodes(
    group: Any,
    *,
    prefix: str = "",
) -> Iterable[tuple[str, Any]]:
    yield prefix, group
    group_keys = getattr(group, "group_keys", None)
    if not callable(group_keys):
        raise ValueError(f"/{group.path} is not a traversable persisted group.")
    for name in sorted(str(value) for value in group_keys()):
        relative = f"{prefix}/{name}" if prefix else name
        yield from _iter_motion_group_nodes(group[name], prefix=relative)


def _motion_track_ref(track_group: Any, relative_path: str) -> str:
    return f"/{track_group.path}/{relative_path}"


def _motion_alias_target(relative_path: str) -> str | None:
    if relative_path == "frame_indices":
        return "source_acquisition_frame_index"
    if relative_path == "angular_velocity_deg_s":
        return "angular_velocity_raw_deg_s"
    for source_level, group_level in MOVEMENT_SPEED_LEVEL_NAMES.items():
        for suffix in ("px", "mm"):
            if relative_path == f"{source_level}_{suffix}":
                return f"movement/speed/{group_level}/{suffix}"
    for level in ("raw", "filtered", "smoothed"):
        for suffix in ("px", "mm"):
            if relative_path == f"frame_path_distance_{level}_{suffix}":
                return (
                    f"movement/speed/{level}/frame_path_distance_{suffix}"
                )
    for suffix in ("px", "mm"):
        if relative_path == f"acceleration_{suffix}":
            return (
                "speed_derivatives/"
                f"{DEFAULT_ACCELERATION_SOURCE_SPEED_LEVEL}/acceleration_{suffix}"
            )
        if relative_path == f"smoothed_acceleration_{suffix}":
            return (
                "speed_derivatives/"
                f"{DEFAULT_ACCELERATION_SOURCE_SPEED_LEVEL}/"
                f"smoothed_acceleration_{suffix}"
            )
    parts = relative_path.split("/")
    if (
        len(parts) == 4
        and parts[:2] == ["movement", "speed"]
        and parts[2] in MOVEMENT_SPEED_LEVEL_NAMES.values()
        and parts[3]
        in {
            "acceleration_px",
            "acceleration_mm",
            "smoothed_acceleration_px",
            "smoothed_acceleration_mm",
        }
    ):
        source_level = next(
            name
            for name, group_name in MOVEMENT_SPEED_LEVEL_NAMES.items()
            if group_name == parts[2]
        )
        return f"speed_derivatives/{source_level}/{parts[3]}"
    return None


def _motion_pixel_peer(relative_path: str) -> str | None:
    parts = relative_path.split("/")
    leaf = parts[-1]
    if leaf == "mm":
        parts[-1] = "px"
    elif leaf.endswith("_mm"):
        parts[-1] = f"{leaf[:-3]}_px"
    else:
        return None
    return "/".join(parts)


def _motion_physical_peer(relative_path: str) -> str | None:
    parts = relative_path.split("/")
    leaf = parts[-1]
    if leaf == "px":
        parts[-1] = "mm"
    elif leaf.endswith("_px"):
        parts[-1] = f"{leaf[:-3]}_mm"
    else:
        return None
    return "/".join(parts)


def _motion_units_for_physical(pixel_units: str) -> str:
    mapping = {
        "px": "mm",
        "px/s": "mm/s",
        "px/s^2": "mm/s^2",
    }
    if pixel_units not in mapping:
        raise ValueError(
            f"Pixel units {pixel_units!r} have no controlled physical mapping."
        )
    return mapping[pixel_units]


def _motion_track_surface_contract(
    track_group: Any,
    relative_path: str,
    *,
    physical_authority_sha256: str | None,
) -> dict[str, Any]:
    """Return controlled domain/units/derivation semantics for one array."""

    run_derivation_ref = "#/run_derivation"
    input_authority_ref = "#/input_authority/fields"
    source_coordinate_ref = "#/source_authority/position"
    source_time_ref = "#/source_authority/temporal"

    pixel_peer = _motion_pixel_peer(relative_path)
    if pixel_peer is not None:
        if physical_authority_sha256 is None:
            raise ValueError(
                f"/{track_group.path}/{relative_path} is physical but the run "
                "has no exact physical authority."
            )
        pixel = _motion_track_surface_contract(
            track_group,
            pixel_peer,
            physical_authority_sha256=physical_authority_sha256,
        )
        alias_of = _motion_alias_target(relative_path)
        return {
            "authority_scope": "public_derived_motion",
            "axis0_domain": pixel["axis0_domain"],
            "units": _motion_units_for_physical(str(pixel["units"])),
            "semantic_profile": str(pixel["semantic_profile"]).replace(
                ".pixel.", ".physical."
            ),
            "operation_id": (
                "exact_alias_v1"
                if alias_of is not None
                else "scale_by_exact_physical_mm_per_pixel_v1"
            ),
            "input_refs": (
                [_motion_track_ref(track_group, alias_of)]
                if alias_of is not None
                else [_motion_track_ref(track_group, pixel_peer)]
            )
            + ["#/physical_authority"],
            "alias_of": (
                _motion_track_ref(track_group, alias_of)
                if alias_of is not None
                else None
            ),
            "pixel_source_ref": _motion_track_ref(track_group, pixel_peer),
            "physical_authority_sha256": physical_authority_sha256,
            "physical_value_comparison": {
                "mode": "dtype_exact_after_multiply_mm_per_pixel_v1",
                "rtol": 0.0,
                "atol": 0.0,
                "nan_policy": "same_mask",
                "infinity_policy": "same_sign_mask",
            },
        }

    def ref(name: str) -> str:
        return _motion_track_ref(track_group, name)

    track_id_ref = f"/{track_group.path}@track_id"
    sample_specs: dict[str, tuple[str, str, str, list[str]]] = {
        "track_sample_key": (
            "identifier_pair",
            "palette.track_motion.track_sample_key.v1",
            "build_track_sample_key_v1",
            [ref("source_acquisition_frame_index"), f"/{track_group.path}@track_id"],
        ),
        "source_acquisition_frame_index": (
            "frame_index",
            "palette.track_motion.source_acquisition_frame.v1",
            "exact_source_row_temporal_subset_v1",
            [ref("source_row_index"), source_time_ref],
        ),
        "source_frame_interpolation": (
            "interpolation_record",
            "palette.track_motion.source_frame_interpolation.v1",
            "exact_source_row_temporal_subset_v1",
            [ref("source_row_index"), source_time_ref],
        ),
        "source_instance_key": (
            "instance_key",
            "palette.track_motion.source_instance_lineage.v1",
            "exact_nullable_source_instance_subset_v1",
            [ref("source_row_index"), source_time_ref],
        ),
        "source_row_index": (
            "row_index",
            "palette.track_motion.source_row_index.v1",
            "selected_source_row_index_v1",
            [source_coordinate_ref, source_time_ref],
        ),
        "time_seconds": (
            "s",
            "palette.track_motion.sample_time.v1",
            "acquisition_frame_divide_fps_v1",
            [ref("source_acquisition_frame_index"), run_derivation_ref],
        ),
        "positions_px": (
            "px",
            "palette.track_motion.pixel.position_xy.v1",
            "exact_subset_reorder_v1",
            [source_coordinate_ref, ref("source_row_index")],
        ),
        "heading_degrees": (
            "deg",
            "palette.track_motion.heading_degrees.v1",
            "float32_source_heading_subset_reorder_v1",
            [
                ref("source_row_index"),
                f"{input_authority_ref}/heading_degrees",
            ],
        ),
        "heading_radians": (
            "rad",
            "palette.track_motion.heading_radians.v1",
            "degrees_to_radians_v1",
            [ref("heading_degrees")],
        ),
        "smoothed_heading_degrees": (
            "deg",
            "palette.track_motion.smoothed_heading_degrees.v1",
            "radians_to_degrees_v1",
            [ref("smoothed_heading_radians")],
        ),
        "smoothed_heading_radians": (
            "rad",
            "palette.track_motion.smoothed_heading_radians.v1",
            "circular_temporal_smoothing_v1",
            [ref("heading_radians"), run_derivation_ref],
        ),
        "keypoint_success": (
            "bool",
            "palette.track_motion.keypoint_success.v1",
            "exact_source_validity_subset_reorder_v1",
            [
                ref("source_row_index"),
                f"{input_authority_ref}/keypoint_success",
            ],
        ),
        "detection_source": (
            "source_code",
            "palette.track_motion.detection_source.v1",
            "exact_source_or_default_code_subset_v1",
            [
                ref("source_row_index"),
                f"{input_authority_ref}/detection_source",
            ],
        ),
        "sample_observed": (
            "bool",
            "palette.track_motion.sample_observed.v1",
            "assigned_track_identity_test_v1",
            [track_id_ref, f"{input_authority_ref}/track_id"],
        ),
        "sample_valid": (
            "bool",
            "palette.track_motion.sample_valid.v1",
            "track_sample_validity_v1",
            [
                ref("sample_observed"),
                ref("source_observed"),
                ref("keypoint_usable"),
                ref("position_finite"),
            ],
        ),
        "source_observed": (
            "bool",
            "palette.track_motion.source_observed.v1",
            "detection_source_observed_test_v1",
            [ref("detection_source")],
        ),
        "keypoint_usable": (
            "bool",
            "palette.track_motion.keypoint_usable.v1",
            "track_sample_validity_v1",
            [ref("keypoint_success"), ref("heading_degrees")],
        ),
        "position_finite": (
            "bool",
            "palette.track_motion.position_finite.v1",
            "finite_xy_test_v1",
            [ref("positions_px")],
        ),
        "heading_usable": (
            "bool",
            "palette.track_motion.heading_usable.v1",
            "finite_heading_test_v1",
            [ref("heading_degrees"), ref("keypoint_success")],
        ),
        "sample_reason_code": (
            "reason_code",
            "palette.track_motion.sample_reason_code.v1",
            "track_sample_validity_reason_v1",
            [
                ref("sample_valid"),
                ref("sample_observed"),
                ref("source_observed"),
                ref("position_finite"),
                ref("heading_usable"),
                ref("keypoint_success"),
            ],
        ),
        "cumulative_path_distance_px": (
            "px",
            "palette.track_motion.pixel.cumulative_path_distance.v1",
            "cumulative_valid_path_distance_v1",
            [ref("movement/speed/smoothed/frame_path_distance_px")],
        ),
    }
    if relative_path == "frame_indices":
        return {
            "authority_scope": "public_derived_motion",
            "axis0_domain": TRACK_MOTION_AXIS_TRACK_SAMPLE,
            "units": "frame_index",
            "semantic_profile": "palette.track_motion.frame_indices_compatibility.v1",
            "operation_id": "exact_alias_v1",
            "input_refs": [ref("source_acquisition_frame_index")],
            "alias_of": ref("source_acquisition_frame_index"),
        }
    if relative_path in sample_specs:
        units, profile, operation, inputs = sample_specs[relative_path]
        return {
            "authority_scope": "public_derived_motion",
            "axis0_domain": TRACK_MOTION_AXIS_TRACK_SAMPLE,
            "units": units,
            "semantic_profile": profile,
            "operation_id": operation,
            "input_refs": inputs,
            "alias_of": None,
        }

    transition_specs: dict[str, tuple[str, str, str, list[str]]] = {
        "delta_heading_degrees": (
            "deg",
            "palette.track_motion.delta_heading.v1",
            "wrapped_heading_difference_v1",
            [
                ref("heading_degrees"),
                ref("delta_seconds"),
                ref("transition_valid"),
                ref("sample_valid"),
            ],
        ),
        "angular_velocity_raw_deg_s": (
            "deg/s",
            "palette.track_motion.angular_velocity_raw.v1",
            "heading_delta_divide_delta_seconds_v1",
            [ref("delta_heading_degrees"), ref("delta_seconds")],
        ),
        "angular_speed_raw_deg_s": (
            "deg/s",
            "palette.track_motion.angular_speed_raw.v1",
            "absolute_value_v1",
            [ref("angular_velocity_raw_deg_s")],
        ),
        "delta_heading_smoothed_degrees": (
            "deg",
            "palette.track_motion.delta_heading_smoothed.v1",
            "wrapped_heading_difference_v1",
            [
                ref("smoothed_heading_degrees"),
                ref("delta_seconds"),
                ref("transition_valid"),
                ref("sample_valid"),
            ],
        ),
        "angular_velocity_smoothed_deg_s": (
            "deg/s",
            "palette.track_motion.angular_velocity_smoothed.v1",
            "heading_delta_divide_delta_seconds_v1",
            [ref("delta_heading_smoothed_degrees"), ref("delta_seconds")],
        ),
        "angular_speed_smoothed_deg_s": (
            "deg/s",
            "palette.track_motion.angular_speed_smoothed.v1",
            "absolute_value_v1",
            [ref("angular_velocity_smoothed_deg_s")],
        ),
        "delta_frames": (
            "frame",
            "palette.track_motion.transition_delta_frames.v1",
            "acquisition_frame_difference_v1",
            [ref("source_acquisition_frame_index")],
        ),
        "delta_seconds": (
            "s",
            "palette.track_motion.transition_delta_seconds.v1",
            "frame_delta_divide_fps_v1",
            [ref("delta_frames"), run_derivation_ref],
        ),
        "transition_valid": (
            "bool",
            "palette.track_motion.transition_valid.v1",
            "track_transition_validity_v1",
            [ref("delta_frames"), ref("delta_seconds"), ref("positions_px")],
        ),
        "transition_reason_code": (
            "reason_code",
            "palette.track_motion.transition_reason_code.v1",
            "track_transition_validity_reason_v1",
            [
                ref("transition_valid"),
                ref("delta_frames"),
                ref("delta_seconds"),
                ref("positions_px"),
            ],
        ),
    }
    if relative_path == "angular_velocity_deg_s":
        return {
            "authority_scope": "public_derived_motion",
            "axis0_domain": TRACK_MOTION_AXIS_TRACK_TRANSITION,
            "units": "deg/s",
            "semantic_profile": "palette.track_motion.angular_velocity_compatibility.v1",
            "operation_id": "exact_alias_v1",
            "input_refs": [ref("angular_velocity_raw_deg_s")],
            "alias_of": ref("angular_velocity_raw_deg_s"),
        }
    if relative_path in transition_specs:
        units, profile, operation, inputs = transition_specs[relative_path]
        return {
            "authority_scope": "public_derived_motion",
            "axis0_domain": TRACK_MOTION_AXIS_TRACK_TRANSITION,
            "units": units,
            "semantic_profile": profile,
            "operation_id": operation,
            "input_refs": inputs,
            "alias_of": None,
            "transition_anchor": "destination_track_sample",
        }

    for source_level, group_level in MOVEMENT_SPEED_LEVEL_NAMES.items():
        grouped_px = f"movement/speed/{group_level}/px"
        if relative_path == grouped_px:
            upstream = {
                "speed_raw": [
                    ref("movement/speed/raw/frame_path_distance_px"),
                    ref("delta_seconds"),
                ],
                "speed_filtered": [
                    ref("movement/speed/filtered/frame_path_distance_px"),
                    ref("delta_seconds"),
                ],
                "speed_smoothed": [
                    ref("movement/speed/smoothed/frame_path_distance_px"),
                    ref("delta_seconds"),
                ],
                "speed_averaged": [ref("movement/speed/smoothed/px"), run_derivation_ref],
            }[source_level]
            operation = {
                "speed_raw": "euclidean_displacement_over_delta_seconds_v1",
                "speed_filtered": "hysteresis_filter_v1",
                "speed_smoothed": "temporal_smoothing_v1",
                "speed_averaged": "temporal_average_v1",
            }[source_level]
            return {
                "authority_scope": "public_derived_motion",
                "axis0_domain": TRACK_MOTION_AXIS_TRACK_TRANSITION,
                "units": "px/s",
                "semantic_profile": f"palette.track_motion.pixel.{source_level}.v1",
                "operation_id": operation,
                "input_refs": upstream,
                "alias_of": None,
                "transition_anchor": "destination_track_sample",
            }
        flat_px = f"{source_level}_px"
        if relative_path == flat_px:
            target = ref(grouped_px)
            return {
                "authority_scope": "public_derived_motion",
                "axis0_domain": TRACK_MOTION_AXIS_TRACK_TRANSITION,
                "units": "px/s",
                "semantic_profile": f"palette.track_motion.pixel.{source_level}_compatibility.v1",
                "operation_id": "exact_alias_v1",
                "input_refs": [target],
                "alias_of": target,
                "transition_anchor": "destination_track_sample",
            }

    for level in ("raw", "filtered", "smoothed"):
        grouped_path = f"movement/speed/{level}/frame_path_distance_px"
        if relative_path == grouped_path:
            inputs = {
                "raw": [ref("positions_px"), ref("transition_valid")],
                "filtered": [
                    ref("movement/speed/raw/frame_path_distance_px"),
                    run_derivation_ref,
                ],
                "smoothed": [
                    ref("movement/speed/filtered/frame_path_distance_px"),
                    ref("transition_valid"),
                    run_derivation_ref,
                ],
            }[level]
            return {
                "authority_scope": "public_derived_motion",
                "axis0_domain": TRACK_MOTION_AXIS_TRACK_TRANSITION,
                "units": "px",
                "semantic_profile": f"palette.track_motion.pixel.frame_path_distance_{level}.v1",
                "operation_id": {
                    "raw": "valid_euclidean_displacement_v1",
                    "filtered": "hysteresis_filter_v1",
                    "smoothed": "temporal_smoothing_v1",
                }[level],
                "input_refs": inputs,
                "alias_of": None,
                "transition_anchor": "destination_track_sample",
            }
        flat_path = f"frame_path_distance_{level}_px"
        if relative_path == flat_path:
            target = ref(grouped_path)
            return {
                "authority_scope": "public_derived_motion",
                "axis0_domain": TRACK_MOTION_AXIS_TRACK_TRANSITION,
                "units": "px",
                "semantic_profile": f"palette.track_motion.pixel.frame_path_distance_{level}_compatibility.v1",
                "operation_id": "exact_alias_v1",
                "input_refs": [target],
                "alias_of": target,
                "transition_anchor": "destination_track_sample",
            }

    parts = relative_path.split("/")
    if (
        len(parts) == 3
        and parts[0] == "speed_derivatives"
        and parts[1] in SPEED_DERIVATIVE_LEVELS
        and parts[2]
        in {"acceleration_px", "smoothed_acceleration_px"}
    ):
        smoothed = parts[2].startswith("smoothed_")
        acceleration_ref = ref(
            f"speed_derivatives/{parts[1]}/acceleration_px"
        )
        return {
            "authority_scope": "public_derived_motion",
            "axis0_domain": TRACK_MOTION_AXIS_TRACK_TRANSITION,
            "units": "px/s^2",
            "semantic_profile": (
                "palette.track_motion.pixel.smoothed_acceleration.v1"
                if smoothed
                else "palette.track_motion.pixel.acceleration.v1"
            ),
            "operation_id": (
                "temporal_smoothing_v1"
                if smoothed
                else "speed_difference_over_delta_seconds_v1"
            ),
            "input_refs": (
                [acceleration_ref, run_derivation_ref]
                if smoothed
                else [ref(f"{parts[1]}_px"), ref("delta_seconds")]
            ),
            "alias_of": None,
            "transition_anchor": "destination_track_sample",
        }
    if (
        len(parts) == 4
        and parts[:2] == ["movement", "speed"]
        and parts[2] in MOVEMENT_SPEED_LEVEL_NAMES.values()
        and parts[3]
        in {"acceleration_px", "smoothed_acceleration_px"}
    ):
        source_level = next(
            name
            for name, group_name in MOVEMENT_SPEED_LEVEL_NAMES.items()
            if group_name == parts[2]
        )
        target = ref(f"speed_derivatives/{source_level}/{parts[3]}")
        return {
            "authority_scope": "public_derived_motion",
            "axis0_domain": TRACK_MOTION_AXIS_TRACK_TRANSITION,
            "units": "px/s^2",
            "semantic_profile": "palette.track_motion.pixel.acceleration_group_alias.v1",
            "operation_id": "exact_alias_v1",
            "input_refs": [target],
            "alias_of": target,
            "transition_anchor": "destination_track_sample",
        }
    if relative_path in {"acceleration_px", "smoothed_acceleration_px"}:
        target_relative = _motion_alias_target(relative_path)
        assert target_relative is not None
        target = ref(target_relative)
        return {
            "authority_scope": "public_derived_motion",
            "axis0_domain": TRACK_MOTION_AXIS_TRACK_TRANSITION,
            "units": "px/s^2",
            "semantic_profile": "palette.track_motion.pixel.acceleration_flat_alias.v1",
            "operation_id": "exact_alias_v1",
            "input_refs": [target],
            "alias_of": target,
            "transition_anchor": "destination_track_sample",
        }

    second_specs: dict[str, tuple[str, str, str, list[str]]] = {
        "second_indices": (
            "second_index",
            "palette.track_motion.second_bin_identity.v1",
            "unique_floor_time_second_bins_v1",
            [ref("source_acquisition_frame_index"), run_derivation_ref],
        ),
        "speed_per_second_px": (
            "px/s",
            "palette.track_motion.pixel.speed_per_second.v1",
            "aggregate_speed_by_second_v1",
            [
                ref("movement/speed/smoothed/frame_path_distance_px"),
                ref("delta_seconds"),
                ref("second_indices"),
                ref("source_acquisition_frame_index"),
                run_derivation_ref,
            ],
        ),
        "heading_per_second_degrees": (
            "deg",
            "palette.track_motion.heading_per_second.v1",
            "circular_heading_mean_by_second_v1",
            [
                ref("heading_radians"),
                ref("second_indices"),
                ref("source_acquisition_frame_index"),
                run_derivation_ref,
            ],
        ),
        "heading_per_second_resultant": (
            "dimensionless",
            "palette.track_motion.heading_resultant_per_second.v1",
            "circular_heading_resultant_by_second_v1",
            [
                ref("heading_radians"),
                ref("second_indices"),
                ref("source_acquisition_frame_index"),
                run_derivation_ref,
            ],
        ),
    }
    if relative_path in second_specs:
        units, profile, operation, inputs = second_specs[relative_path]
        return {
            "authority_scope": "public_derived_motion",
            "axis0_domain": TRACK_MOTION_AXIS_TRACK_SECOND,
            "units": units,
            "semantic_profile": profile,
            "operation_id": operation,
            "input_refs": inputs,
            "alias_of": None,
        }

    if relative_path.startswith("swim_bouts/"):
        leaf = relative_path.rsplit("/", 1)[-1]
        units = "field_native"
        if leaf.endswith("_px"):
            units = "px"
        elif leaf.endswith("_mm"):
            units = "mm"
        elif leaf.endswith("_s") or "time" in leaf:
            units = "s"
        return {
            "authority_scope": "sealed_auxiliary_not_motion_public",
            "axis0_domain": TRACK_MOTION_AXIS_TRACK_BOUT,
            "units": units,
            "semantic_profile": "palette.track_motion.mirrored_swim_bout_field.v1",
            "operation_id": "exact_external_swim_bout_mirror_v1",
            "input_refs": [
                f"/{track_group.path}/swim_bouts@source_swim_bout_run"
            ],
            "alias_of": None,
        }

    raise ValueError(
        f"/{track_group.path}/{relative_path} is not in the controlled full-motion "
        "surface vocabulary."
    )


def _expected_motion_track_surface_paths(*, include_physical: bool) -> set[str]:
    paths = {
        "frame_indices",
        "track_sample_key",
        "source_acquisition_frame_index",
        "source_frame_interpolation",
        "source_instance_key",
        "source_row_index",
        "time_seconds",
        "positions_px",
        "heading_degrees",
        "heading_radians",
        "delta_heading_degrees",
        "angular_velocity_deg_s",
        "angular_velocity_raw_deg_s",
        "angular_speed_raw_deg_s",
        "delta_heading_smoothed_degrees",
        "angular_velocity_smoothed_deg_s",
        "angular_speed_smoothed_deg_s",
        "smoothed_heading_degrees",
        "smoothed_heading_radians",
        "keypoint_success",
        "detection_source",
        "sample_observed",
        "sample_valid",
        "source_observed",
        "keypoint_usable",
        "position_finite",
        "heading_usable",
        "sample_reason_code",
        "delta_frames",
        "delta_seconds",
        "transition_valid",
        "transition_reason_code",
        "speed_raw_px",
        "speed_filtered_px",
        "speed_smoothed_px",
        "speed_averaged_px",
        "acceleration_px",
        "smoothed_acceleration_px",
        "frame_path_distance_raw_px",
        "frame_path_distance_filtered_px",
        "frame_path_distance_smoothed_px",
        "cumulative_path_distance_px",
        "second_indices",
        "speed_per_second_px",
        "heading_per_second_degrees",
        "heading_per_second_resultant",
    }
    for source_level, group_level in MOVEMENT_SPEED_LEVEL_NAMES.items():
        paths.add(f"speed_derivatives/{source_level}/acceleration_px")
        paths.add(
            f"speed_derivatives/{source_level}/smoothed_acceleration_px"
        )
        paths.add(f"movement/speed/{group_level}/px")
        paths.add(f"movement/speed/{group_level}/acceleration_px")
        paths.add(
            f"movement/speed/{group_level}/smoothed_acceleration_px"
        )
        if group_level != "averaged":
            paths.add(
                f"movement/speed/{group_level}/frame_path_distance_px"
            )
    if include_physical:
        physical = {
            _motion_physical_peer(path)
            for path in paths
            if _motion_physical_peer(path) is not None
        }
        paths.update(str(path) for path in physical if path is not None)
    return paths


def _motion_run_array_contract(
    run_group: Any,
    relative_path: str,
) -> dict[str, Any]:
    def ref(name: str) -> str:
        return f"/{run_group.path}/{name}"
    if relative_path == "track_ids":
        return {
            "authority_scope": "run_track_inventory",
            "axis0_domain": TRACK_MOTION_AXIS_RUN_TRACK,
            "units": "track_id",
            "semantic_profile": "palette.track_motion.track_inventory.v1",
            "operation_id": "sorted_unique_track_inventory_v1",
            "input_refs": ["#/input_authority/fields/track_id", "#/tracks"],
        }
    if relative_path == "track_arena_ids":
        return {
            "authority_scope": "run_track_inventory",
            "axis0_domain": TRACK_MOTION_AXIS_RUN_TRACK,
            "units": "arena_id",
            "semantic_profile": "palette.track_motion.track_arena_identity.v1",
            "operation_id": "exact_track_to_arena_mapping_v1",
            "input_refs": [
                ref("track_ids"),
                "#/input_authority/fields/arena_id",
                "#/input_authority/arena_inventory",
            ],
        }
    camera_specs = {
        "camera_frame_ids": ("frame_index", "camera_frame_identity"),
        "stimulus_frame_nums": ("frame_index", "stimulus_frame_mapping"),
        "timestamp_ns": ("ns", "camera_timestamp"),
        "trial_state": ("state_code", "stimulus_trial_state"),
        "metadata_mask": ("bool", "metadata_interpolation_mask"),
        "angle_unsigned_deg": ("deg", "chaser_angle_unsigned"),
        "angle_signed_deg": ("deg", "chaser_angle_signed"),
        "heading_deg": ("deg", "chaser_heading"),
        "has_offline": ("bool", "chaser_offline_presence"),
    }
    if relative_path in camera_specs:
        units, semantic = camera_specs[relative_path]
        return {
            "authority_scope": "sealed_run_auxiliary_not_track_motion_public",
            "axis0_domain": TRACK_MOTION_AXIS_RUN_CAMERA_SAMPLE,
            "units": units,
            "semantic_profile": f"palette.track_motion.{semantic}.v1",
            "operation_id": "exact_chaser_metrics_projection_v1",
            "input_refs": [
                "#/run_derivation/record/inputs/chaser_metrics"
            ],
        }
    raise ValueError(
        f"Run-root array {relative_path!r} is outside the controlled motion-run "
        "inventory."
    )


def _validate_track_parameter_inventory(
    *,
    run_type: str,
    parameters: Mapping[str, Any],
) -> None:
    expected = (
        _CANONICAL_OFFLINE_PARAMETER_KEYS
        if run_type == "offline"
        else _CANONICAL_ONLINE_PARAMETER_KEYS
    )
    missing = sorted(set(expected) - set(parameters))
    unsupported = sorted(set(parameters) - set(expected))
    if missing or unsupported:
        raise ValueError(
            f"Canonical {run_type} track parameter inventory is not closed "
            f"(missing={missing!r}, unsupported={unsupported!r})."
        )


def _validate_chaser_metrics_input(
    run_group: Any,
    value: Any,
) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("chaser_metrics must be one closed persisted object.")
    missing = sorted(_CANONICAL_CHASER_METRICS_INPUT_KEYS - set(value))
    unsupported = sorted(set(value) - _CANONICAL_CHASER_METRICS_INPUT_KEYS)
    if missing or unsupported:
        raise ValueError(
            "Canonical chaser_metrics input inventory is not closed "
            f"(missing={missing!r}, unsupported={unsupported!r})."
        )
    _controlled_run_leaf(value["metrics_run"], label="chaser_metrics.metrics_run")
    _controlled_run_leaf(
        value["stimulus_run"],
        label="chaser_metrics.stimulus_run",
    )
    chaser_index = value["chaser_index"]
    if (
        isinstance(chaser_index, (bool, np.bool_))
        or not isinstance(chaser_index, (int, np.integer))
        or int(chaser_index) < 0
    ):
        raise ValueError("chaser_metrics.chaser_index must be nonnegative int.")
    try:
        interpolation_seconds = float(value["distance_interpolation_seconds"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "chaser_metrics.distance_interpolation_seconds must be finite and nonnegative."
        ) from exc
    if not np.isfinite(interpolation_seconds) or interpolation_seconds < 0:
        raise ValueError(
            "chaser_metrics.distance_interpolation_seconds must be finite and nonnegative."
        )
    omitted = value["omitted_coordinate_fields"]
    if (
        not isinstance(omitted, list)
        or omitted != sorted(set(omitted))
        or any(
            not isinstance(name, str)
            or name not in _CANONICAL_CHASER_OMITTED_COORDINATE_FIELDS
            for name in omitted
        )
    ):
        raise ValueError(
            "chaser_metrics.omitted_coordinate_fields must be one sorted controlled list."
        )
    status = value["coordinate_geometry_status"]
    reason = value["coordinate_geometry_reason_code"]
    if status == "omitted_untyped_legacy_chaser_metrics_v1":
        if (
            not omitted
            or reason != "LEGACY_METRICS_LACK_SEALED_COORDINATE_AUTHORITY"
        ):
            raise ValueError(
                "Omitted legacy chaser geometry requires its exact reason and field list."
            )
    elif status == "not_present":
        if omitted or reason != "NONE":
            raise ValueError(
                "Absent chaser geometry must use an empty field list and reason NONE."
            )
    else:
        raise ValueError("chaser_metrics.coordinate_geometry_status is unsupported.")
    live_arrays = {str(name) for name in run_group.array_keys()}
    missing_arrays = sorted(_CANONICAL_CHASER_METRICS_REQUIRED_ARRAYS - live_arrays)
    if missing_arrays:
        raise ValueError(
            "chaser_metrics is claimed without its exact sealed auxiliary arrays "
            f"(missing={missing_arrays!r})."
        )


def _validate_offline_auxiliary_inputs(
    run_group: Any,
    inputs: Mapping[str, Any],
) -> None:
    chaser_metrics = inputs.get("chaser_metrics")
    live_arrays = {str(name) for name in run_group.array_keys()}
    live_chaser_arrays = live_arrays & set(_CANONICAL_CHASER_METRICS_ARRAYS)
    if chaser_metrics is None:
        if live_chaser_arrays:
            raise ValueError(
                "Canonical run publishes chaser auxiliary arrays without one closed "
                "chaser_metrics input."
            )
    else:
        _validate_chaser_metrics_input(run_group, chaser_metrics)

    swim_bout_run = inputs.get("swim_bout_run")
    if swim_bout_run is not None:
        _controlled_run_leaf(swim_bout_run, label="swim_bout_run")
    tracks_parent = run_group["tracks"] if "tracks" in run_group else None
    if tracks_parent is None:
        raise ValueError("Track run lacks its exact tracks group.")
    has_mirrors = any(
        "swim_bouts" in tracks_parent[name]
        for name in tracks_parent.group_keys()
    )
    if (swim_bout_run is None) != (not has_mirrors):
        raise ValueError(
            "swim_bout_run and persisted mirrored swim-bout surfaces must be present together."
        )


def _validate_online_input_inventory(
    *,
    method: Any,
    inputs: Mapping[str, Any],
    source_path: str,
) -> None:
    if method == "track_kinematics_online":
        expected = _CANONICAL_ONLINE_RAW_INPUT_KEYS
        refined = False
    elif method == "track_kinematics_online_refined":
        expected = _CANONICAL_ONLINE_REFINED_INPUT_KEYS
        refined = True
    else:
        raise ValueError("Canonical online track method is unsupported.")
    missing = sorted(set(expected) - set(inputs))
    unsupported = sorted(set(inputs) - set(expected))
    if missing or unsupported:
        raise ValueError(
            "Canonical online track input inventory is not closed "
            f"(missing={missing!r}, unsupported={unsupported!r})."
        )
    stimulus_run = _controlled_run_leaf(
        inputs["stimulus_run"],
        label="stimulus_run",
    )
    if refined:
        refined_run = _controlled_run_leaf(
            inputs["refined_online_run"],
            label="refined_online_run",
        )
        expected_source_prefix = f"refined_online_runs/{refined_run}/"
    else:
        expected_source_prefix = f"analysis/stimulus_runs/{stimulus_run}/"
    if not source_path.startswith(expected_source_prefix):
        raise ValueError(
            "Online declared run inputs do not own the exact sealed position source."
        )
    # Reuse the source-ref builder for strict chaser-index, path, and digest grammar.
    _track_kinematics_source_refs(run_type="online", inputs=inputs)


def _motion_run_derivation_record(
    run_group: Any,
    positions: BoundTrackPositionBindings,
) -> dict[str, Any]:
    record = {
        name: copy.deepcopy(run_group.attrs[name])
        for name in _MOTION_RUN_DERIVATION_ATTR_NAMES
        if name in run_group.attrs
    }
    required = {
        "schema_id",
        "schema_version",
        "method",
        "method_version",
        "row_axis",
        "parameters",
        "source_refs",
        "inputs",
        "provenance",
        "run_provenance",
    }
    missing = sorted(required - set(record))
    if missing:
        raise ValueError(
            "Track run lacks exact derivation attrs required for full-motion "
            f"publication: {missing!r}."
        )
    if not isinstance(record["parameters"], Mapping) or not isinstance(
        record["source_refs"], Mapping
    ) or not isinstance(record["inputs"], Mapping):
        raise ValueError(
            "Track motion parameters/source_refs/inputs must be persisted objects."
        )
    parameters = record["parameters"]
    inputs = record["inputs"]
    if "physical_calibration" in parameters:
        raise ValueError(
            "Canonical track parameters must not duplicate the typed physical "
            "coordinate authority."
        )
    provenance = record["provenance"]
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("stage") != "track_kinematics"
        or not _track_attr_values_equal(provenance.get("parameters"), parameters)
        or not _track_attr_values_equal(provenance.get("inputs"), inputs)
    ):
        raise ValueError(
            "Track motion stage provenance parameters or inputs conflict with the "
            "canonical derivation mappings."
        )
    run_provenance = record["run_provenance"]
    run_provenance_validation = validate_run_provenance(
        run_provenance if isinstance(run_provenance, Mapping) else None
    )
    if (
        not run_provenance_validation.valid
        or not isinstance(run_provenance, Mapping)
        or not _track_attr_values_equal(run_provenance.get("params"), parameters)
        or not _track_attr_values_equal(
            run_provenance.get("input_run_ids"),
            inputs,
        )
        or run_provenance.get("config_hash") != sha256_payload(parameters)
    ):
        raise ValueError(
            "Track motion finalization provenance conflicts with canonical "
            "parameters or inputs."
        )
    for duplicate_name in (
        "fps",
        "smoothing_seconds",
        "smoothing_method",
        "smoothing_alignment",
        "savgol_polyorder",
        "hysteresis_enabled",
        "hysteresis_high_px",
        "hysteresis_low_px",
        "hysteresis_min_frames",
        "hysteresis_band_policy",
        "distance_interpolation_seconds",
    ):
        if duplicate_name in record and (
            duplicate_name not in parameters
            or not _track_attr_values_equal(
                record[duplicate_name],
                parameters[duplicate_name],
            )
        ):
            raise ValueError(
                f"Track motion root {duplicate_name} conflicts with canonical "
                "parameters."
            )
    path_parts = str(getattr(run_group, "path", "")).strip("/").split("/")
    if (
        len(path_parts) != 4
        or path_parts[:2] != ["analysis", "track_kinematics_runs"]
        or path_parts[2] not in {"offline", "online"}
    ):
        raise ValueError("Track motion run path does not identify one canonical scope.")
    run_type = path_parts[2]
    _validate_track_parameter_inventory(
        run_type=run_type,
        parameters=parameters,
    )
    if run_type == "offline":
        if record["method"] != "track_kinematics_offline":
            raise ValueError("Canonical offline track method is unsupported.")
        required_inputs = set(_REQUIRED_CANONICAL_OFFLINE_INPUT_KEYS)
        allowed_inputs = required_inputs | set(
            _OPTIONAL_CANONICAL_OFFLINE_INPUT_KEYS
        )
        missing_inputs = sorted(required_inputs - set(inputs))
        unsupported_inputs = sorted(set(inputs) - allowed_inputs)
        if missing_inputs or unsupported_inputs:
            raise ValueError(
                "Canonical offline track input inventory is not closed "
                f"(missing={missing_inputs!r}, unsupported={unsupported_inputs!r})."
            )
        _validate_offline_auxiliary_inputs(run_group, inputs)
    expected_refs = _track_kinematics_source_refs(
        run_type=run_type,
        inputs=inputs,
    )
    if not _track_attr_values_equal(record["source_refs"], expected_refs):
        raise ValueError(
            "Track motion source_refs are not the exact mechanical projection of inputs."
        )
    source = positions.source_positions
    source_path = str(source.coordinate_node.path).strip("/")
    if run_type == "offline":
        if (
            record["source_refs"].get("source_position_source_kind")
            != CANONICAL_OFFLINE_POSITION_SOURCE_KIND
        ):
            raise ValueError(
                "Offline position source kind must identify canonical crop-row "
                "source-camera centers."
            )
        if record["source_refs"].get("source_position_source_path") != source_path:
            raise ValueError(
                "Offline declared position source is not the exact sealed position "
                "coordinate array."
            )
        declared_rowset = record["source_refs"].get(
            "source_position_source_rowset_path"
        )
        expected_rowset = str(source.row_identity.rowset_path).strip("/")
        if (
            not isinstance(declared_rowset, str)
            or not declared_rowset
            or declared_rowset != expected_rowset
            or source_path.rsplit("/", 1)[0] != expected_rowset
        ):
            raise ValueError(
                "Offline declared position rowset is required and must be the "
                "exact row-identity owner of the sealed position coordinate array."
            )
        if record["source_refs"].get("source_crop_path") != expected_rowset:
            raise ValueError(
                "Offline crop_run does not identify the exact sealed position "
                "rowset."
            )
        exact_detection_path = _canonical_crop_detection_rowset_path(source)
        if (
            record["source_refs"].get("source_detection_path")
            != exact_detection_path
        ):
            raise ValueError(
                "Offline detection_path does not identify the exact detection "
                "rowset bound by the canonical crop selection."
            )
    else:
        _validate_online_input_inventory(
            method=record["method"],
            inputs=inputs,
            source_path=source_path,
        )
        if (
            record["source_refs"].get("source_positions_px_path") != source_path
            or record["source_refs"].get(
                "source_positions_px_coordinate_descriptor_sha256"
            )
            != source.descriptor.digest()
        ):
            raise ValueError(
                "Online declared position path or descriptor digest differs from "
                "the exact sealed position coordinate array."
            )
    normalized = _motion_json_object(
        record,
        label=f"/{run_group.path} motion derivation",
    )
    return {
        "record": normalized,
        "record_sha256": _canonical_json_sha256(normalized),
    }


def _motion_source_authority_record(
    positions: BoundTrackPositionBindings,
) -> dict[str, Any]:
    source = positions.source_positions
    temporal = positions.source_temporal_authority
    node = source.coordinate_node
    return {
        "position": {
            "array_ref": f"/{node.path}",
            "dtype": np.dtype(node.dtype).str,
            "shape": [int(value) for value in node.shape],
            "content_sha256": array_payload_sha256(node),
            "coordinate_descriptor_sha256": source.descriptor.digest(),
            "row_identity_ref": source.row_identity.record_ref,
            "row_identity_sha256": source.row_identity.record_sha256,
        },
        "temporal": {
            "record_ref": temporal.record_ref,
            "record_sha256": temporal.record_sha256,
        },
    }


def _motion_input_array_record(node: Any) -> dict[str, Any]:
    path = str(getattr(node, "path", "")).strip("/")
    if not path or any(part in {"", ".", ".."} for part in path.split("/")):
        raise ValueError("Track-motion input arrays require canonical archive paths.")
    dtype = np.dtype(getattr(node, "dtype"))
    shape = tuple(int(value) for value in getattr(node, "shape"))
    if dtype.hasobject:
        raise ValueError(f"/{path} uses an unsupported object dtype.")
    return {
        "array_ref": f"/{path}",
        "dtype": dtype.str,
        "shape": [int(value) for value in shape],
        "content_sha256": array_payload_sha256(node),
    }


def _motion_input_child(group: Any, name: str) -> Any | None:
    try:
        return group[name]
    except (KeyError, TypeError, AttributeError):
        return None


def _motion_parent_path(node: Any) -> str:
    path = str(getattr(node, "path", "")).strip("/")
    return path.rsplit("/", 1)[0] if "/" in path else ""


def _motion_exact_sibling_ref(node: Any, leaf_name: str) -> str:
    """Return the only accepted path for a controlled sibling input leaf."""

    if (
        not isinstance(leaf_name, str)
        or not leaf_name
        or "/" in leaf_name
        or leaf_name in {".", ".."}
    ):
        raise ValueError("Track-motion sibling leaf name is invalid.")
    parent = _motion_parent_path(node)
    return f"/{parent}/{leaf_name}" if parent else f"/{leaf_name}"


def _motion_optional_input_node(
    authoritative_root: Any,
    expected_ref: str,
) -> Any | None:
    """Resolve one optional canonical input path and distinguish absence from aliasing."""

    if (
        not isinstance(expected_ref, str)
        or not expected_ref.startswith("/")
        or any(part in {"", ".", ".."} for part in expected_ref[1:].split("/"))
    ):
        raise ValueError(f"Track-motion optional input ref {expected_ref!r} is invalid.")
    node = authoritative_root
    try:
        for part in expected_ref[1:].split("/"):
            node = node[part]
    except (KeyError, TypeError, AttributeError):
        return None
    live_ref = f"/{str(getattr(node, 'path', '')).strip('/')}"
    if live_ref != expected_ref or archive_identity(node) != archive_identity(
        authoritative_root
    ):
        raise ValueError(
            f"Track-motion sibling {expected_ref!r} resolved to another node or archive."
        )
    return node


def _motion_optional_exact_sibling(
    authoritative_root: Any,
    source_node: Any,
    leaf_name: str,
) -> Any | None:
    """Resolve one exact sibling without treating a detached lookalike as absence."""

    return _motion_optional_input_node(
        authoritative_root,
        _motion_exact_sibling_ref(source_node, leaf_name),
    )


def _motion_selected_keypoint_usability_node(
    authoritative_root: Any,
    keypoint_sibling: Any,
) -> tuple[str | None, Any | None]:
    """Resolve the first live usability leaf under the controlled writer priority."""

    for name in KEYPOINT_USABILITY_DATASET_CANDIDATES:
        node = _motion_optional_exact_sibling(
            authoritative_root,
            keypoint_sibling,
            name,
        )
        if node is not None:
            return name, node
    return None, None


def _motion_input_array_field(
    node: Any,
    *,
    row_alignment: str,
    output_dtype: str,
) -> dict[str, Any]:
    return {
        "source_kind": "array",
        "row_alignment": row_alignment,
        "output_dtype": np.dtype(output_dtype).str,
        "array": _motion_input_array_record(node),
    }


def _motion_generated_field(
    *,
    generator_id: str,
    row_count: int,
    output_dtype: str,
    value: Any | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "source_kind": "generated",
        "generator_id": generator_id,
        "output_dtype": np.dtype(output_dtype).str,
        "shape": [int(row_count)],
    }
    if value is not None:
        record["value"] = value
    return record


def build_track_motion_input_authority(
    authoritative_root: Any,
    *,
    source_positions: BoundCanonicalCoordinateDescriptor,
    mode: str,
    heading_node: Any | None = None,
    keypoint_usability_node: Any | None = None,
    keypoint_row_key_node: Any | None = None,
    tracking_group: Any | None = None,
    detection_source_node: Any | None = None,
    generated_track_id: int | None = None,
) -> BoundTrackMotionInputAuthority:
    """Bind exact row-aligned inputs used to derive one future motion run.

    Offline inputs must prove keypoint and tracking row alignment through exact
    instance-key payloads.  Online inputs may use explicitly named deterministic
    generators for values that have no upstream array (single track id,
    keypoint usability, and missing headings/detection codes).
    """

    if mode not in {"offline_exact_sources_v1", "online_exact_or_generated_v1"}:
        raise ValueError(f"Unsupported track-motion input authority mode {mode!r}.")
    source_positions = require_bound_canonical_coordinate_descriptor(
        source_positions
    )
    root_archive = archive_identity(authoritative_root)
    if root_archive != archive_identity(source_positions.coordinate_node):
        raise ValueError(
            "Track-motion input authority and selected positions must belong to "
            "the same exact archive."
        )
    source_row_count = int(source_positions.row_identity.leading_dimension)
    if tuple(int(value) for value in source_positions.coordinate_node.shape) != (
        source_row_count,
        2,
    ):
        raise ValueError(
            "Track-motion source positions must contain one point_xy per source row."
        )
    position_key_node = source_positions.row_identity._key_array_node
    if archive_identity(position_key_node) != root_archive:
        raise ValueError("Track-motion source row key belongs to another archive.")
    position_key_values = np.array(position_key_node[:], copy=True, order="C")
    if position_key_values.shape != (source_row_count,):
        raise ValueError("Track-motion source row key length is inconsistent.")

    fields: dict[str, Any] = {}
    keypoint_alignment: dict[str, Any]
    if mode == "offline_exact_sources_v1":
        if heading_node is None or keypoint_row_key_node is None:
            raise ValueError(
                "Offline track motion requires exact heading and keypoint row-key arrays."
            )
        for node in (heading_node, keypoint_row_key_node):
            if archive_identity(node) != root_archive:
                raise ValueError("Offline keypoint evidence belongs to another archive.")
        keypoint_key_values = np.array(
            keypoint_row_key_node[:], copy=True, order="C"
        )
        if (
            keypoint_key_values.dtype != position_key_values.dtype
            or keypoint_key_values.shape != position_key_values.shape
            or not np.array_equal(keypoint_key_values, position_key_values)
        ):
            raise ValueError(
                "Offline keypoint row identity does not exactly equal the selected "
                "position row identity."
            )
        keypoint_alignment = {
            "mode": "exact_row_key_equality_v1",
            "position_row_key": _motion_input_array_record(position_key_node),
            "keypoint_row_key": _motion_input_array_record(
                keypoint_row_key_node
            ),
        }
        if tuple(int(value) for value in heading_node.shape) != (
            source_row_count,
        ):
            raise ValueError("Offline heading array is not source-row aligned.")
        fields["heading_degrees"] = _motion_input_array_field(
            heading_node,
            row_alignment="keypoint_exact_row_key_equality_v1",
            output_dtype="<f4",
        )
        selected_usability_name, selected_usability_node = (
            _motion_selected_keypoint_usability_node(
                authoritative_root,
                heading_node,
            )
        )
        if keypoint_usability_node is not None:
            if archive_identity(keypoint_usability_node) != root_archive:
                raise ValueError(
                    "Offline keypoint-usability evidence belongs to another archive."
                )
            expected_usability_ref = (
                _motion_exact_sibling_ref(heading_node, selected_usability_name)
                if selected_usability_name is not None
                else None
            )
            observed_usability_ref = (
                f"/{str(getattr(keypoint_usability_node, 'path', '')).strip('/')}"
            )
            if (
                selected_usability_node is None
                or observed_usability_ref != expected_usability_ref
            ):
                raise ValueError(
                    "Offline keypoint usability must be the exact first available "
                    "controlled usability leaf from the selected keypoint run."
                )
            if tuple(int(value) for value in keypoint_usability_node.shape) != (
                source_row_count,
            ):
                raise ValueError(
                    "Offline keypoint-usability array is not source-row aligned."
                )
            fields["keypoint_success"] = _motion_input_array_field(
                keypoint_usability_node,
                row_alignment="keypoint_exact_row_key_equality_v1",
                output_dtype="|b1",
            )
        else:
            if selected_usability_node is not None:
                raise ValueError(
                    f"Offline {selected_usability_name} exists on the selected "
                    "keypoint run and cannot be replaced by implicit_all_true."
                )
            fields["keypoint_success"] = _motion_generated_field(
                generator_id="all_true_v1",
                row_count=source_row_count,
                output_dtype="|b1",
                value=True,
            )
    else:
        if keypoint_row_key_node is not None or keypoint_usability_node is not None:
            raise ValueError(
                "Online track motion uses its selected position rowset and explicit "
                "generators, not offline keypoint evidence."
            )
        keypoint_alignment = {"mode": "selected_position_rowset_v1"}
        if heading_node is None:
            if (
                _motion_optional_exact_sibling(
                    authoritative_root,
                    source_positions.coordinate_node,
                    "visual_angle_deg",
                )
                is not None
            ):
                raise ValueError(
                    "Online visual_angle_deg exists on the selected position rowset "
                    "and cannot be replaced by a generated heading."
                )
            fields["heading_degrees"] = _motion_generated_field(
                generator_id="all_nan_float32_v1",
                row_count=source_row_count,
                output_dtype="<f4",
            )
        else:
            if archive_identity(heading_node) != root_archive:
                raise ValueError("Online heading evidence belongs to another archive.")
            if (
                _motion_parent_path(heading_node)
                != _motion_parent_path(source_positions.coordinate_node)
                or f"/{str(getattr(heading_node, 'path', '')).strip('/')}"
                != _motion_exact_sibling_ref(
                    source_positions.coordinate_node,
                    "visual_angle_deg",
                )
                or tuple(int(value) for value in heading_node.shape)
                != (source_row_count,)
            ):
                raise ValueError(
                    "Online heading must be the exact visual_angle_deg sibling of "
                    "the selected position surface with the same row count."
                )
            fields["heading_degrees"] = _motion_input_array_field(
                heading_node,
                row_alignment="selected_position_rowset_sibling_v1",
                output_dtype="<f4",
            )
        fields["keypoint_success"] = _motion_generated_field(
            generator_id="all_true_v1",
            row_count=source_row_count,
            output_dtype="|b1",
            value=True,
        )

    tracking_alignment: dict[str, Any]
    arena_inventory: dict[str, Any] | None = None
    if tracking_group is not None:
        if mode != "offline_exact_sources_v1" or generated_track_id is not None:
            raise ValueError("Exact tracking arrays are valid only for offline authority.")
        if archive_identity(tracking_group) != root_archive:
            raise ValueError("Tracking evidence belongs to another archive.")
        tracking_track_ids = _motion_input_child(tracking_group, "track_ids")
        tracking_keys = _motion_input_child(tracking_group, "instance_key")
        if tracking_track_ids is None or tracking_keys is None:
            raise ValueError(
                "Future offline track motion requires tracking track_ids and "
                "instance_key arrays."
            )
        raw_tracking_keys = np.array(tracking_keys[:], copy=True, order="C")
        if (
            raw_tracking_keys.dtype != position_key_values.dtype
            or raw_tracking_keys.shape != position_key_values.shape
            or np.unique(raw_tracking_keys).shape[0] != source_row_count
            or np.unique(position_key_values).shape[0] != source_row_count
        ):
            raise ValueError(
                "Tracking and position instance-key arrays must be equal-length, "
                "unique, and dtype-identical."
            )
        tracking_order = np.argsort(raw_tracking_keys, kind="stable")
        position_order = np.argsort(position_key_values, kind="stable")
        if not np.array_equal(
            raw_tracking_keys[tracking_order],
            position_key_values[position_order],
        ):
            raise ValueError(
                "Tracking instance-key set differs from the selected position rowset."
            )
        position_to_tracking = np.empty(source_row_count, dtype=np.int64)
        position_to_tracking[position_order] = tracking_order.astype(
            np.int64, copy=False
        )
        tracking_alignment = {
            "mode": "instance_key_exact_set_reorder_v1",
            "position_instance_key": _motion_input_array_record(
                position_key_node
            ),
            "tracking_instance_key": _motion_input_array_record(tracking_keys),
            "position_to_tracking_row_dtype": position_to_tracking.dtype.str,
            "position_to_tracking_row_shape": [source_row_count],
            "position_to_tracking_row_sha256": identity_array_content_sha256(
                position_to_tracking
            ),
        }
        if tuple(int(value) for value in tracking_track_ids.shape) != (
            source_row_count,
        ):
            raise ValueError("Tracking track_ids is not aligned to tracking rows.")
        fields["track_id"] = _motion_input_array_field(
            tracking_track_ids,
            row_alignment="tracking_instance_key_reorder_v1",
            output_dtype="<i8",
        )
        tracking_arena_ids = _motion_input_child(tracking_group, "arena_ids")
        if tracking_arena_ids is not None:
            if tuple(int(value) for value in tracking_arena_ids.shape) != (
                source_row_count,
            ):
                raise ValueError("Tracking arena_ids is not aligned to tracking rows.")
            fields["arena_id"] = _motion_input_array_field(
                tracking_arena_ids,
                row_alignment="tracking_instance_key_reorder_v1",
                output_dtype="<i8",
            )
        else:
            fields["arena_id"] = _motion_generated_field(
                generator_id="unavailable_v1",
                row_count=source_row_count,
                output_dtype="<i8",
            )
        present = _motion_input_child(tracking_group, "track_ids_present")
        track_arenas = _motion_input_child(tracking_group, "track_arena_ids")
        if (present is None) != (track_arenas is None):
            raise ValueError(
                "Tracking arena inventory requires both track_ids_present and "
                "track_arena_ids."
            )
        if present is not None and track_arenas is not None:
            if tuple(present.shape) != tuple(track_arenas.shape):
                raise ValueError("Tracking arena inventory arrays disagree in shape.")
            arena_inventory = {
                "track_ids_present": _motion_input_array_record(present),
                "track_arena_ids": _motion_input_array_record(track_arenas),
            }
    else:
        if mode != "online_exact_or_generated_v1" or generated_track_id is None:
            raise ValueError(
                "Online track motion requires one explicit generated track id."
            )
        if isinstance(generated_track_id, (bool, np.bool_)) or int(
            generated_track_id
        ) < 0:
            raise ValueError("Generated online track id must be a nonnegative integer.")
        tracking_alignment = {"mode": "generated_single_track_v1"}
        fields["track_id"] = _motion_generated_field(
            generator_id="constant_int_v1",
            row_count=source_row_count,
            output_dtype="<i8",
            value=int(generated_track_id),
        )
        fields["arena_id"] = _motion_generated_field(
            generator_id="unavailable_v1",
            row_count=source_row_count,
            output_dtype="<i8",
        )

    if detection_source_node is None:
        if (
            _motion_optional_exact_sibling(
                authoritative_root,
                source_positions.coordinate_node,
                "detection_source",
            )
            is not None
        ):
            raise ValueError(
                "detection_source exists on the selected position rowset and "
                "cannot be replaced by a generated value."
            )
        fields["detection_source"] = _motion_generated_field(
            generator_id="constant_int_v1",
            row_count=source_row_count,
            output_dtype="|i1",
            value=0,
        )
    else:
        if (
            archive_identity(detection_source_node) != root_archive
            or _motion_parent_path(detection_source_node)
            != _motion_parent_path(source_positions.coordinate_node)
            or f"/{str(getattr(detection_source_node, 'path', '')).strip('/')}"
            != _motion_exact_sibling_ref(
                source_positions.coordinate_node,
                "detection_source",
            )
            or tuple(int(value) for value in detection_source_node.shape)
            != (source_row_count,)
        ):
            raise ValueError(
                "Detection-source evidence must be the exact detection_source "
                "sibling of the selected position surface."
            )
        fields["detection_source"] = _motion_input_array_field(
            detection_source_node,
            row_alignment="selected_position_rowset_sibling_v1",
            output_dtype="|i1",
        )

    record = _motion_json_object(
        {
            "schema_id": TRACK_MOTION_INPUT_AUTHORITY_SCHEMA_ID,
            "schema_version": TRACK_MOTION_INPUT_AUTHORITY_SCHEMA_VERSION,
            "mode": mode,
            "source_row_count": source_row_count,
            "position_row_identity": {
                "record_ref": source_positions.row_identity.record_ref,
                "record_sha256": source_positions.row_identity.record_sha256,
                "key_array": _motion_input_array_record(position_key_node),
            },
            "keypoint_alignment": keypoint_alignment,
            "tracking_alignment": tracking_alignment,
            "fields": fields,
            "arena_inventory": arena_inventory,
        },
        label="track-motion input authority",
    )
    return BoundTrackMotionInputAuthority(
        archive=root_archive,
        record=record,
        _verification_seal=_BOUND_TRACK_MOTION_INPUT_AUTHORITY_SEAL,
    )


def _resolve_motion_input_node(authoritative_root: Any, array_ref: Any) -> Any:
    if (
        not isinstance(array_ref, str)
        or not array_ref.startswith("/")
        or any(part in {"", ".", ".."} for part in array_ref[1:].split("/"))
    ):
        raise ValueError(f"Track-motion input array ref {array_ref!r} is invalid.")
    node = authoritative_root
    try:
        for part in array_ref[1:].split("/"):
            node = node[part]
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(
            f"Track-motion input array {array_ref!r} is unavailable."
        ) from exc
    if f"/{str(getattr(node, 'path', '')).strip('/')}" != array_ref:
        raise ValueError(
            f"Track-motion input array {array_ref!r} resolved to another node."
        )
    if archive_identity(node) != archive_identity(authoritative_root):
        raise ValueError(
            f"Track-motion input array {array_ref!r} belongs to another archive."
        )
    return node


def _validate_motion_input_array_record(
    authoritative_root: Any,
    value: Any,
    *,
    label: str,
) -> tuple[Any, np.ndarray]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be one exact input-array record.")
    record = dict(value)
    if set(record) != {"array_ref", "dtype", "shape", "content_sha256"}:
        raise ValueError(f"{label} has an unsupported input-array record shape.")
    node = _resolve_motion_input_node(
        authoritative_root,
        record.get("array_ref"),
    )
    live = _motion_input_array_record(node)
    if not _track_attr_values_equal(record, live):
        raise ValueError(
            f"{label} dtype, shape, path, or payload changed after publication."
        )
    return node, np.array(node[:], copy=True, order="C")


def _motion_input_field_values(
    authoritative_root: Any,
    field: Any,
    *,
    label: str,
    source_row_count: int,
    position_to_tracking: np.ndarray | None,
) -> np.ndarray | None:
    if not isinstance(field, Mapping):
        raise ValueError(f"{label} must be one controlled input field.")
    source_kind = field.get("source_kind")
    try:
        output_dtype = np.dtype(field.get("output_dtype"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} output dtype is invalid.") from exc
    if output_dtype.hasobject:
        raise ValueError(f"{label} output dtype is unsupported.")
    if source_kind == "array":
        expected_names = {
            "source_kind",
            "row_alignment",
            "output_dtype",
            "array",
        }
        if set(field) != expected_names:
            raise ValueError(f"{label} array field inventory is not closed.")
        _node, values = _validate_motion_input_array_record(
            authoritative_root,
            field["array"],
            label=f"{label}.array",
        )
        alignment = field.get("row_alignment")
        if alignment == "tracking_instance_key_reorder_v1":
            if position_to_tracking is None:
                raise ValueError(f"{label} lacks its exact tracking row mapping.")
            values = values[position_to_tracking]
        elif alignment not in {
            "keypoint_exact_row_key_equality_v1",
            "selected_position_rowset_sibling_v1",
        }:
            raise ValueError(f"{label} uses an unsupported row alignment.")
        converted = np.asarray(values, dtype=output_dtype)
    elif source_kind == "generated":
        allowed = {
            "source_kind",
            "generator_id",
            "output_dtype",
            "shape",
            "value",
        }
        if set(field) - allowed:
            raise ValueError(f"{label} generated field inventory is not closed.")
        if field.get("shape") != [int(source_row_count)]:
            raise ValueError(f"{label} generated shape is not source-row aligned.")
        generator = field.get("generator_id")
        if generator == "unavailable_v1":
            if "value" in field:
                raise ValueError(f"{label} unavailable generator cannot carry a value.")
            return None
        if generator == "all_nan_float32_v1":
            if output_dtype != np.dtype("<f4") or "value" in field:
                raise ValueError(f"{label} all-NaN generator is malformed.")
            converted = np.full(source_row_count, np.nan, dtype=np.float32)
        elif generator == "all_true_v1":
            if field.get("value") is not True or output_dtype != np.dtype("|b1"):
                raise ValueError(f"{label} all-true generator is malformed.")
            converted = np.ones(source_row_count, dtype=bool)
        elif generator == "constant_int_v1":
            value = field.get("value")
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)
            ):
                raise ValueError(f"{label} constant integer is malformed.")
            converted = np.full(source_row_count, int(value), dtype=output_dtype)
        else:
            raise ValueError(f"{label} uses an unsupported generator {generator!r}.")
    else:
        raise ValueError(f"{label} uses an unsupported source kind {source_kind!r}.")
    if converted.shape != (source_row_count,):
        raise ValueError(f"{label} does not resolve to one value per source row.")
    return np.array(converted, copy=True, order="C")


def _motion_array_ref_from_field(field: Any) -> str | None:
    if not isinstance(field, Mapping) or field.get("source_kind") != "array":
        return None
    array = field.get("array")
    return str(array.get("array_ref")) if isinstance(array, Mapping) else None


def _validate_track_motion_input_authority(
    authoritative_root: Any,
    run_group: Any,
    positions: BoundTrackPositionBindings,
    groups: list[tuple[int, Any]],
) -> tuple[dict[str, Any], dict[str, np.ndarray | None]]:
    raw = run_group.attrs.get(TRACK_MOTION_INPUT_AUTHORITY_ATTR)
    if not isinstance(raw, Mapping):
        raise ValueError(
            "Future track-motion publication requires exact persisted input authority."
        )
    record = _motion_json_object(
        raw,
        label=f"/{run_group.path} track-motion input authority",
    )
    expected_top_level = {
        "schema_id",
        "schema_version",
        "mode",
        "source_row_count",
        "position_row_identity",
        "keypoint_alignment",
        "tracking_alignment",
        "fields",
        "arena_inventory",
    }
    if set(record) != expected_top_level:
        raise ValueError("Track-motion input-authority field inventory is not closed.")
    if (
        record.get("schema_id") != TRACK_MOTION_INPUT_AUTHORITY_SCHEMA_ID
        or record.get("schema_version")
        != TRACK_MOTION_INPUT_AUTHORITY_SCHEMA_VERSION
    ):
        raise ValueError("Track-motion input-authority schema is unsupported.")
    source_row_count = int(positions.source_positions.row_identity.leading_dimension)
    if record.get("source_row_count") != source_row_count:
        raise ValueError(
            "Track-motion input authority and selected position row count disagree."
        )
    position_identity = record.get("position_row_identity")
    if not isinstance(position_identity, Mapping) or set(position_identity) != {
        "record_ref",
        "record_sha256",
        "key_array",
    }:
        raise ValueError("Track-motion position-row authority is malformed.")
    if (
        position_identity.get("record_ref")
        != positions.source_positions.row_identity.record_ref
        or position_identity.get("record_sha256")
        != positions.source_positions.row_identity.record_sha256
    ):
        raise ValueError("Track-motion position row identity changed.")
    _position_key_node, position_key_values = _validate_motion_input_array_record(
        authoritative_root,
        position_identity["key_array"],
        label="track-motion position row key",
    )
    expected_position_key_record = _motion_input_array_record(
        positions.source_positions.row_identity._key_array_node
    )
    if not _track_attr_values_equal(
        position_identity["key_array"], expected_position_key_record
    ):
        raise ValueError(
            "Track-motion position-row authority is not the exact selected "
            "position row-key array."
        )
    if position_key_values.shape != (source_row_count,):
        raise ValueError("Track-motion position row-key length changed.")

    mode = record.get("mode")
    keypoint_alignment = record.get("keypoint_alignment")
    if not isinstance(keypoint_alignment, Mapping):
        raise ValueError("Track-motion keypoint alignment is malformed.")
    if mode == "offline_exact_sources_v1":
        if set(keypoint_alignment) != {
            "mode",
            "position_row_key",
            "keypoint_row_key",
        } or keypoint_alignment.get("mode") != "exact_row_key_equality_v1":
            raise ValueError("Offline keypoint alignment authority is incomplete.")
        _kp_position_node, kp_position_values = (
            _validate_motion_input_array_record(
                authoritative_root,
                keypoint_alignment["position_row_key"],
                label="offline keypoint position row key",
            )
        )
        _kp_node, keypoint_key_values = _validate_motion_input_array_record(
            authoritative_root,
            keypoint_alignment["keypoint_row_key"],
            label="offline keypoint row key",
        )
        if (
            not _track_attr_values_equal(
                keypoint_alignment["position_row_key"],
                expected_position_key_record,
            )
            or kp_position_values.dtype != position_key_values.dtype
            or not np.array_equal(kp_position_values, position_key_values)
            or keypoint_key_values.dtype != position_key_values.dtype
            or not np.array_equal(keypoint_key_values, position_key_values)
        ):
            raise ValueError("Offline keypoint row mapping changed after publication.")
    elif mode == "online_exact_or_generated_v1":
        if dict(keypoint_alignment) != {"mode": "selected_position_rowset_v1"}:
            raise ValueError("Online keypoint alignment authority is malformed.")
    else:
        raise ValueError(f"Unsupported track-motion authority mode {mode!r}.")

    tracking_alignment = record.get("tracking_alignment")
    if not isinstance(tracking_alignment, Mapping):
        raise ValueError("Track-motion tracking alignment is malformed.")
    position_to_tracking: np.ndarray | None = None
    if tracking_alignment.get("mode") == "instance_key_exact_set_reorder_v1":
        if mode != "offline_exact_sources_v1":
            raise ValueError(
                "Exact tracking instance-key alignment is valid only offline."
            )
        expected_names = {
            "mode",
            "position_instance_key",
            "tracking_instance_key",
            "position_to_tracking_row_dtype",
            "position_to_tracking_row_shape",
            "position_to_tracking_row_sha256",
        }
        if set(tracking_alignment) != expected_names:
            raise ValueError("Tracking instance-key alignment inventory is not closed.")
        _position_node, aligned_position_keys = _validate_motion_input_array_record(
            authoritative_root,
            tracking_alignment["position_instance_key"],
            label="tracking position instance key",
        )
        _tracking_node, tracking_keys = _validate_motion_input_array_record(
            authoritative_root,
            tracking_alignment["tracking_instance_key"],
            label="tracking instance key",
        )
        if (
            not _track_attr_values_equal(
                tracking_alignment["position_instance_key"],
                expected_position_key_record,
            )
            or aligned_position_keys.dtype != position_key_values.dtype
            or not np.array_equal(aligned_position_keys, position_key_values)
            or tracking_keys.dtype != position_key_values.dtype
            or tracking_keys.shape != position_key_values.shape
            or np.unique(tracking_keys).shape[0] != source_row_count
        ):
            raise ValueError("Tracking instance-key evidence changed.")
        tracking_order = np.argsort(tracking_keys, kind="stable")
        position_order = np.argsort(position_key_values, kind="stable")
        if not np.array_equal(
            tracking_keys[tracking_order], position_key_values[position_order]
        ):
            raise ValueError("Tracking instance-key set changed.")
        position_to_tracking = np.empty(source_row_count, dtype=np.int64)
        position_to_tracking[position_order] = tracking_order.astype(
            np.int64, copy=False
        )
        if (
            tracking_alignment.get("position_to_tracking_row_dtype")
            != position_to_tracking.dtype.str
            or tracking_alignment.get("position_to_tracking_row_shape")
            != [source_row_count]
            or tracking_alignment.get("position_to_tracking_row_sha256")
            != identity_array_content_sha256(position_to_tracking)
        ):
            raise ValueError("Tracking row-reorder mapping changed.")
    else:
        if dict(tracking_alignment) != {"mode": "generated_single_track_v1"}:
            raise ValueError("Track-motion tracking alignment mode is unsupported.")
        if mode != "online_exact_or_generated_v1":
            raise ValueError(
                "Generated single-track alignment is valid only online."
            )

    fields = record.get("fields")
    expected_fields = {
        "heading_degrees",
        "keypoint_success",
        "track_id",
        "arena_id",
        "detection_source",
    }
    if not isinstance(fields, Mapping) or set(fields) != expected_fields:
        raise ValueError("Track-motion input field inventory is not closed.")
    expected_output_dtypes = {
        "heading_degrees": np.dtype("<f4"),
        "keypoint_success": np.dtype("|b1"),
        "track_id": np.dtype("<i8"),
        "arena_id": np.dtype("<i8"),
        "detection_source": np.dtype("|i1"),
    }
    for name, expected_dtype in expected_output_dtypes.items():
        try:
            observed_dtype = np.dtype(fields[name].get("output_dtype"))
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Track-motion input field {name} output dtype is invalid."
            ) from exc
        if observed_dtype != expected_dtype:
            raise ValueError(
                f"Track-motion input field {name} output dtype is not canonical."
            )
    values = {
        name: _motion_input_field_values(
            authoritative_root,
            fields[name],
            label=f"track-motion input field {name}",
            source_row_count=source_row_count,
            position_to_tracking=position_to_tracking,
        )
        for name in sorted(expected_fields)
    }

    source_refs = run_group.attrs.get("source_refs")
    inputs = run_group.attrs.get("inputs")
    if not isinstance(source_refs, Mapping) or not isinstance(inputs, Mapping):
        raise ValueError("Track-motion run lacks source refs for input authority.")
    if mode == "offline_exact_sources_v1":
        keypoint_prefix = source_refs.get("source_keypoint_path")
        tracking_prefix = source_refs.get("source_tracking_path")
        keypoint_prefix = _controlled_two_component_run_path(
            keypoint_prefix,
            families=frozenset({"keypoints_runs", "refined_keypoints_runs"}),
            label="source_keypoint_path",
        )
        tracking_prefix = _controlled_two_component_run_path(
            tracking_prefix,
            families=frozenset({"tracking_runs"}),
            label="source_tracking_path",
        )
        heading_ref = _motion_array_ref_from_field(fields["heading_degrees"])
        if heading_ref != f"/{keypoint_prefix}/heading":
            raise ValueError("Offline heading authority is not the selected keypoint run.")
        if fields["heading_degrees"].get("row_alignment") != (
            "keypoint_exact_row_key_equality_v1"
        ):
            raise ValueError("Offline heading row alignment is not exact keypoint order.")
        keypoint_key_ref = str(
            keypoint_alignment["keypoint_row_key"].get("array_ref")
        )
        if keypoint_key_ref != f"/{keypoint_prefix}/instance_key":
            raise ValueError(
                "Offline row-key authority is not the selected keypoint run."
            )
        usability_ref = _motion_array_ref_from_field(fields["keypoint_success"])
        selected_usability_name, selected_usability_node = (
            _motion_selected_keypoint_usability_node(
                authoritative_root,
                _resolve_motion_input_node(
                    authoritative_root,
                    f"/{keypoint_prefix}/heading",
                ),
            )
        )
        if usability_ref is None:
            if (
                selected_usability_node is not None
                or selected_usability_name is not None
                or fields["keypoint_success"].get("source_kind") != "generated"
                or fields["keypoint_success"].get("generator_id") != "all_true_v1"
            ):
                raise ValueError(
                    "Offline generated keypoint usability either omits a controlled "
                    "selected-run leaf or lacks the controlled all-true generator."
                )
        elif (
            selected_usability_name is None
            or selected_usability_node is None
            or usability_ref != f"/{keypoint_prefix}/{selected_usability_name}"
        ):
            raise ValueError(
                "Offline keypoint-usability authority is not the exact first "
                "available controlled dataset from the selected keypoint run."
            )
        elif fields["keypoint_success"].get("row_alignment") != (
            "keypoint_exact_row_key_equality_v1"
        ):
            raise ValueError(
                "Offline keypoint-usability row alignment is not exact keypoint order."
            )
        track_ref = _motion_array_ref_from_field(fields["track_id"])
        if track_ref != f"/{tracking_prefix}/track_ids":
            raise ValueError("Offline track-id authority is not the selected tracking run.")
        if fields["track_id"].get("row_alignment") != (
            "tracking_instance_key_reorder_v1"
        ):
            raise ValueError("Offline track-id row alignment is not exact tracking order.")
        tracking_key_ref = str(
            tracking_alignment["tracking_instance_key"].get("array_ref")
        )
        if tracking_key_ref != f"/{tracking_prefix}/instance_key":
            raise ValueError(
                "Tracking row-key authority is not the selected tracking run."
            )
        arena_ref = _motion_array_ref_from_field(fields["arena_id"])
        if arena_ref is not None and arena_ref != f"/{tracking_prefix}/arena_ids":
            raise ValueError(
                "Offline arena-id authority is not the selected tracking run."
            )
        if arena_ref is None:
            if (
                _motion_optional_input_node(
                    authoritative_root,
                    f"/{tracking_prefix}/arena_ids",
                )
                is not None
                or fields["arena_id"].get("source_kind") != "generated"
                or fields["arena_id"].get("generator_id") != "unavailable_v1"
            ):
                raise ValueError(
                    "Offline missing arena-id authority is malformed or omits the "
                    "selected tracking arena_ids leaf."
                )
        elif fields["arena_id"].get("row_alignment") != (
            "tracking_instance_key_reorder_v1"
        ):
            raise ValueError("Offline arena-id row alignment is not exact tracking order.")
    else:
        if (
            fields["keypoint_success"].get("source_kind") != "generated"
            or fields["keypoint_success"].get("generator_id") != "all_true_v1"
            or fields["track_id"].get("source_kind") != "generated"
            or fields["track_id"].get("generator_id") != "constant_int_v1"
            or fields["arena_id"].get("source_kind") != "generated"
            or fields["arena_id"].get("generator_id") != "unavailable_v1"
        ):
            raise ValueError("Online generated input authority is malformed.")
        heading_ref = _motion_array_ref_from_field(fields["heading_degrees"])
        if heading_ref is None:
            if (
                _motion_optional_exact_sibling(
                    authoritative_root,
                    positions.source_positions.coordinate_node,
                    "visual_angle_deg",
                )
                is not None
            ):
                raise ValueError(
                    "Generated online heading omits the selected visual_angle_deg leaf."
                )
            if (
                fields["heading_degrees"].get("source_kind") != "generated"
                or fields["heading_degrees"].get("generator_id")
                != "all_nan_float32_v1"
            ):
                raise ValueError("Online generated heading authority is malformed.")
        elif heading_ref != _motion_exact_sibling_ref(
            positions.source_positions.coordinate_node,
            "visual_angle_deg",
        ):
            raise ValueError(
                "Online heading is not the exact selected visual_angle_deg leaf."
            )
        elif fields["heading_degrees"].get("row_alignment") != (
            "selected_position_rowset_sibling_v1"
        ):
            raise ValueError("Online heading row alignment is not the selected rowset.")

    detection_ref = _motion_array_ref_from_field(fields["detection_source"])
    if detection_ref is None:
        if (
            _motion_optional_exact_sibling(
                authoritative_root,
                positions.source_positions.coordinate_node,
                "detection_source",
            )
            is not None
        ):
            raise ValueError(
                "Generated detection-source authority omits the selected "
                "detection_source leaf."
            )
        if (
            fields["detection_source"].get("source_kind") != "generated"
            or fields["detection_source"].get("generator_id") != "constant_int_v1"
            or fields["detection_source"].get("value") != 0
        ):
            raise ValueError("Generated detection-source authority is malformed.")
    elif detection_ref != _motion_exact_sibling_ref(
        positions.source_positions.coordinate_node,
        "detection_source",
    ):
        raise ValueError(
            "Detection-source authority is not the exact selected detection_source leaf."
        )
    elif fields["detection_source"].get("row_alignment") != (
        "selected_position_rowset_sibling_v1"
    ):
        raise ValueError("Detection-source row alignment is not the selected rowset.")

    inventory_record = record.get("arena_inventory")
    inventory_map: dict[int, int] | None = None
    live_inventory_ids: Any | None = None
    live_inventory_arenas: Any | None = None
    if mode == "offline_exact_sources_v1":
        tracking_prefix = source_refs["source_tracking_path"]
        live_inventory_ids = _motion_optional_input_node(
            authoritative_root,
            f"/{tracking_prefix}/track_ids_present",
        )
        live_inventory_arenas = _motion_optional_input_node(
            authoritative_root,
            f"/{tracking_prefix}/track_arena_ids",
        )
        if (live_inventory_ids is None) != (live_inventory_arenas is None):
            raise ValueError(
                "Selected tracking run has an incomplete live arena inventory."
            )
        if inventory_record is None and live_inventory_ids is not None:
            raise ValueError(
                "Track-motion authority omits the selected tracking arena inventory."
            )
    if inventory_record is not None:
        if not isinstance(inventory_record, Mapping) or set(inventory_record) != {
            "track_ids_present",
            "track_arena_ids",
        }:
            raise ValueError("Track arena inventory authority is malformed.")
        if mode != "offline_exact_sources_v1":
            raise ValueError("Track arena inventory is valid only offline.")
        tracking_prefix = source_refs["source_tracking_path"]
        if (
            inventory_record["track_ids_present"].get("array_ref")
            != f"/{tracking_prefix}/track_ids_present"
            or inventory_record["track_arena_ids"].get("array_ref")
            != f"/{tracking_prefix}/track_arena_ids"
        ):
            raise ValueError(
                "Track arena inventory is not from the selected tracking run."
            )
        _ids_node, inventory_ids = _validate_motion_input_array_record(
            authoritative_root,
            inventory_record["track_ids_present"],
            label="tracking track_ids_present",
        )
        _arenas_node, inventory_arenas = _validate_motion_input_array_record(
            authoritative_root,
            inventory_record["track_arena_ids"],
            label="tracking track_arena_ids",
        )
        if (
            inventory_ids.ndim != 1
            or inventory_arenas.ndim != 1
            or inventory_ids.shape != inventory_arenas.shape
            or np.unique(inventory_ids).shape[0] != inventory_ids.shape[0]
        ):
            raise ValueError("Tracking arena inventory payload is invalid.")
        inventory_map = {
            int(track_id): int(arena_id)
            for track_id, arena_id in zip(
                inventory_ids.tolist(), inventory_arenas.tolist(), strict=True
            )
        }
    arena_values = values["arena_id"]
    track_values = values["track_id"]
    if arena_values is not None and inventory_map is not None:
        for track_id in np.unique(track_values):
            selected = arena_values[track_values == track_id]
            unique_arenas = np.unique(selected)
            if unique_arenas.shape != (1,) or inventory_map.get(int(track_id)) != int(
                unique_arenas[0]
            ):
                raise ValueError(
                    "Tracking row-level arena IDs disagree with its exact track inventory."
                )

    track_manifest = run_group.attrs.get("track_manifest")
    if not isinstance(track_manifest, list) or len(track_manifest) != len(groups):
        raise ValueError("Track manifest is unavailable for input-authority checks.")
    expected_track_arenas: list[int | None] = []
    for index, (track_id, subgroup) in enumerate(groups):
        source_rows = np.array(
            subgroup["source_row_index"][:], dtype=np.int64, copy=True
        )
        if (
            source_rows.ndim != 1
            or np.any(source_rows < 0)
            or np.any(source_rows >= source_row_count)
        ):
            raise ValueError(f"Track {track_id} source-row selection is invalid.")
        expected_track_ids = track_values[source_rows]
        if np.any(expected_track_ids != int(track_id)):
            raise ValueError(
                f"Track {track_id} rows are not assigned by the exact tracking authority."
            )
        expected_heading = np.asarray(
            values["heading_degrees"][source_rows], dtype=np.float32
        )
        observed_heading = np.array(
            subgroup["heading_degrees"][:], dtype=np.float32, copy=True
        )
        if not np.array_equal(observed_heading, expected_heading, equal_nan=True):
            raise ValueError(
                f"Track {track_id} headings differ from the exact selected source rows."
            )
        for name, dtype in (
            ("keypoint_success", np.dtype(bool)),
            ("detection_source", np.dtype(np.int8)),
        ):
            expected = np.asarray(values[name][source_rows], dtype=dtype)
            observed = np.array(subgroup[name][:], dtype=dtype, copy=True)
            if not np.array_equal(observed, expected):
                raise ValueError(
                    f"Track {track_id} {name} differs from its exact source rows."
                )
        expected_arena: int | None = None
        if arena_values is not None:
            selected_arenas = np.unique(arena_values[source_rows])
            if selected_arenas.shape != (1,):
                raise ValueError(
                    f"Track {track_id} contains multiple authoritative arena IDs."
                )
            candidate = int(selected_arenas[0])
            if inventory_map is not None and inventory_map.get(track_id) == candidate:
                expected_arena = candidate
        elif inventory_map is not None:
            expected_arena = inventory_map.get(track_id)
        manifest_entry = track_manifest[index]
        if not isinstance(manifest_entry, Mapping):
            raise ValueError(f"Track {track_id} manifest entry is malformed.")
        if (
            subgroup.attrs.get("arena_id") != expected_arena
            or manifest_entry.get("arena_id") != expected_arena
        ):
            raise ValueError(
                f"Track {track_id} arena identity differs from exact tracking authority."
            )
        expected_track_arenas.append(expected_arena)

    has_run_arenas = "track_arena_ids" in run_group
    if all(value is not None for value in expected_track_arenas):
        if not has_run_arenas:
            raise ValueError("Run omits exact track_arena_ids despite complete authority.")
        observed = np.array(run_group["track_arena_ids"][:], copy=True)
        expected = np.asarray(expected_track_arenas, dtype=observed.dtype)
        if observed.ndim != 1 or not np.array_equal(observed, expected):
            raise ValueError("Run track_arena_ids differs from exact tracking authority.")
    elif has_run_arenas:
        raise ValueError("Run publishes track_arena_ids without complete authority.")

    manifest_record = {
        "record": record,
        "record_sha256": _canonical_json_sha256(record),
    }
    return manifest_record, values


def _canonical_track_space_from_label(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be one controlled coordinate-space label.")
    cleaned = value.strip()
    if cleaned not in {
        "source_camera_image_px",
        "stimulus_texture_px",
        "stimulus_canvas_px",
        "projector_px",
        "arena_relative_canvas_px",
    }:
        raise ValueError(
            f"{label} uses unsupported track coordinate space {cleaned!r}."
        )
    return cleaned


def _motion_run_root_attrs_record(
    run_group: Any,
    positions: BoundTrackPositionBindings,
) -> dict[str, Any]:
    """Return one closed, conflict-checked run-root attribute partition."""

    live_names = set(str(name) for name in run_group.attrs)
    unknown = sorted(
        live_names
        - set(_MOTION_RUN_ALLOWED_ATTR_NAMES)
        - set(_MOTION_RUN_PUBLICATION_DYNAMIC_ATTR_NAMES)
    )
    if unknown:
        raise ValueError(
            f"/{run_group.path} has unsupported run-root attrs: {unknown!r}."
        )

    expected_space = positions.source_positions.descriptor.space_id
    parameters = run_group.attrs.get("parameters")
    if not isinstance(parameters, Mapping) or "coordinate_space" not in parameters:
        raise ValueError(
            "Track run parameters lack one direction-explicit coordinate space."
        )
    parameter_space = _canonical_track_space_from_label(
        parameters["coordinate_space"],
        label="track parameters.coordinate_space",
    )
    if parameter_space != expected_space:
        raise ValueError(
            "Track parameter coordinate space conflicts with the exact selected "
            f"position descriptor ({parameter_space!r} != {expected_space!r})."
        )
    if "coordinate_space" in run_group.attrs:
        root_space = _canonical_track_space_from_label(
            run_group.attrs["coordinate_space"],
            label="legacy root coordinate_space",
        )
        if root_space != expected_space:
            raise ValueError(
                "Legacy root coordinate_space conflicts with the authoritative "
                "position descriptor."
            )
    exact_source_path = str(positions.source_positions.coordinate_node.path).strip("/")
    exact_source_digest = positions.source_positions.descriptor.digest()
    if (
        "positions_px_source_path" in run_group.attrs
        and run_group.attrs["positions_px_source_path"] != exact_source_path
    ):
        raise ValueError(
            "Legacy root positions_px_source_path conflicts with the exact sealed "
            "position source."
        )
    if (
        "positions_px_source_coordinate_descriptor_sha256" in run_group.attrs
        and run_group.attrs["positions_px_source_coordinate_descriptor_sha256"]
        != exact_source_digest
    ):
        raise ValueError(
            "Legacy root position descriptor digest conflicts with the exact sealed "
            "position source."
        )
    inputs = run_group.attrs.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("Track run root lacks exact persisted inputs.")

    immutable = {
        name: copy.deepcopy(run_group.attrs[name])
        for name in sorted(live_names - _MOTION_RUN_LEGACY_COMPATIBILITY_ATTR_NAMES)
        if name not in _MOTION_RUN_PUBLICATION_DYNAMIC_ATTR_NAMES
    }
    legacy = {
        name: copy.deepcopy(run_group.attrs[name])
        for name in sorted(live_names & _MOTION_RUN_LEGACY_COMPATIBILITY_ATTR_NAMES)
    }
    record = {
        "immutable_attrs": immutable,
        "legacy_compatibility": {
            "authority_scope": "sealed_non_authoritative_legacy_metadata",
            "attrs": legacy,
        },
    }
    normalized = _motion_json_object(
        record,
        label=f"/{run_group.path} closed root attrs",
    )
    return {
        "record": normalized,
        "record_sha256": _canonical_json_sha256(normalized),
    }


def _validate_motion_storage_attrs(
    attrs: Mapping[str, Any],
    *,
    label: str,
    required: bool,
) -> None:
    expected = geometry_preload_attrs()
    present = set(attrs) & _MOTION_ARRAY_STORAGE_ATTR_NAMES
    if not present and not required:
        return
    if present != set(expected):
        raise ValueError(
            f"{label} storage attrs are not one exact geometry-preload profile."
        )
    for name, value in expected.items():
        if not _track_attr_values_equal(attrs[name], value):
            raise ValueError(
                f"{label} storage attr {name!r} differs from the controlled "
                "geometry-preload profile."
            )


def _motion_nested_group_expected_attrs(
    run_group: Any,
    relative_path: str,
    *,
    include_physical: bool,
    sample_count: int,
) -> dict[str, Any]:
    parameters = run_group.attrs.get("parameters")
    if not isinstance(parameters, Mapping):
        raise ValueError("Track motion group validation requires parameters.")
    try:
        fps = float(parameters["fps"])
        smooth_seconds = float(parameters["smoothing_seconds"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Track motion group validation requires numeric fps/smoothing_seconds."
        ) from exc
    requested_post_window = max(1, int(round(fps * smooth_seconds)))
    effective_post_window = _bounded_smoothing_window(
        requested_post_window,
        sample_count,
    )

    if relative_path == "movement":
        return {
            "schema_id": MOVEMENT_SCHEMA_ID,
            "layout": "movement/speed/<level>",
            "compatibility_flat_arrays": True,
            "compatibility_speed_derivatives": True,
        }
    if relative_path == "movement/speed":
        return {
            "schema_id": MOVEMENT_SPEED_SCHEMA_ID,
            "levels": list(MOVEMENT_SPEED_LEVEL_NAMES.values()),
            "source_level_names": dict(MOVEMENT_SPEED_LEVEL_NAMES),
            "preferred_read_contract": "movement/speed/<level>",
        }
    if relative_path.startswith("movement/speed/"):
        group_level = relative_path.rsplit("/", 1)[-1]
        source_level = next(
            (
                source
                for source, group in MOVEMENT_SPEED_LEVEL_NAMES.items()
                if group == group_level
            ),
            None,
        )
        if source_level is None:
            raise ValueError(
                f"Unsupported movement speed group {relative_path!r}."
            )
        expected: dict[str, Any] = {
            "schema_id": MOVEMENT_SPEED_LEVEL_SCHEMA_ID,
            "source_speed_level": source_level,
            "level": group_level,
            "units_px": "px/s",
            "flat_speed_px_array": f"../../../{source_level}_px",
            "time_delta_array": "../../../delta_seconds",
            "derivative_method": "first_difference",
            "post_smoothing_method": "moving_average",
            "post_smoothing_alignment": "centered",
            "post_smoothing_window_frames": effective_post_window,
            "post_smoothing_window_frames_requested": requested_post_window,
            "post_smoothing_window_frames_effective": effective_post_window,
            "post_smoothing_window_s": smooth_seconds,
        }
        if include_physical:
            expected.update(
                {
                    "units_mm": "mm/s",
                    "flat_speed_mm_array": f"../../../{source_level}_mm",
                }
            )
        if source_level != "speed_averaged":
            path_stem = source_level.removeprefix("speed_")
            expected["flat_frame_path_distance_px_array"] = (
                f"../../../frame_path_distance_{path_stem}_px"
            )
            if include_physical:
                expected["flat_frame_path_distance_mm_array"] = (
                    f"../../../frame_path_distance_{path_stem}_mm"
                )
        return expected
    if relative_path == "speed_derivatives":
        aliases = {
            "acceleration_px": (
                "speed_derivatives/"
                f"{DEFAULT_ACCELERATION_SOURCE_SPEED_LEVEL}/acceleration_px"
            ),
            "smoothed_acceleration_px": (
                "speed_derivatives/"
                f"{DEFAULT_ACCELERATION_SOURCE_SPEED_LEVEL}/"
                "smoothed_acceleration_px"
            ),
        }
        if include_physical:
            aliases.update(
                {
                    "acceleration_mm": (
                        "speed_derivatives/"
                        f"{DEFAULT_ACCELERATION_SOURCE_SPEED_LEVEL}/"
                        "acceleration_mm"
                    ),
                    "smoothed_acceleration_mm": (
                        "speed_derivatives/"
                        f"{DEFAULT_ACCELERATION_SOURCE_SPEED_LEVEL}/"
                        "smoothed_acceleration_mm"
                    ),
                }
            )
        return {
            "schema_id": SPEED_DERIVATIVES_SCHEMA_ID,
            "speed_levels": list(SPEED_DERIVATIVE_LEVELS),
            "default_source_speed_level": (
                DEFAULT_ACCELERATION_SOURCE_SPEED_LEVEL
            ),
            "compatibility_alias_arrays": aliases,
        }
    if relative_path.startswith("speed_derivatives/"):
        source_level = relative_path.rsplit("/", 1)[-1]
        if source_level not in SPEED_DERIVATIVE_LEVELS:
            raise ValueError(
                f"Unsupported speed-derivative group {relative_path!r}."
            )
        expected = {
            "schema_id": SPEED_DERIVATIVE_SCHEMA_ID,
            "source_speed_level": source_level,
            "source_speed_px_array": f"../../{source_level}_px",
            "time_delta_array": "../../delta_seconds",
            "derivative_method": "first_difference",
            "post_smoothing_method": "moving_average",
            "post_smoothing_alignment": "centered",
            "post_smoothing_window_frames": effective_post_window,
            "post_smoothing_window_frames_requested": requested_post_window,
            "post_smoothing_window_frames_effective": effective_post_window,
            "post_smoothing_window_s": smooth_seconds,
            "interpretation": (
                "Framewise time derivative of the named source speed trace. "
                "Use this group, not the legacy flat acceleration arrays, when "
                "the source speed semantics matter."
            ),
        }
        if include_physical:
            expected["source_speed_mm_array"] = f"../../{source_level}_mm"
        return expected
    raise ValueError(
        f"/{relative_path} is outside the controlled track group metadata schema."
    )


def _expected_motion_smoothing_windows(
    run_group: Any,
    *,
    sample_count: int,
) -> dict[str, Any]:
    parameters = run_group.attrs.get("parameters")
    if not isinstance(parameters, Mapping):
        raise ValueError("Track motion smoothing validation requires parameters.")
    try:
        requested = max(
            1,
            int(
                round(
                    float(parameters["fps"])
                    * float(parameters["smoothing_seconds"])
                )
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Track motion smoothing validation requires numeric fps and "
            "smoothing_seconds."
        ) from exc
    alignment = str(parameters.get("smoothing_alignment", "centered"))
    distance_effective = _bounded_smoothing_window(
        requested,
        max(0, int(sample_count) - 1),
    )
    smoothing_method = str(parameters.get("smoothing_method", "moving_average"))
    if smoothing_method == "savitzky_golay" and distance_effective > 1:
        savgol_window = (
            distance_effective
            if distance_effective % 2 == 1
            else distance_effective - 1
        )
        savgol_window = max(1, savgol_window)
        raw_polyorder = parameters.get("savgol_polyorder")
        polyorder = 3 if raw_polyorder is None else int(raw_polyorder)
        polyorder = min(polyorder, savgol_window - 1)
        if savgol_window >= polyorder + 2:
            distance_effective = savgol_window
    return {
        "schema_id": "palette.track_motion_smoothing_windows",
        "schema_version": 1,
        "distance_transition": {
            "alignment": alignment,
            "requested_frames": requested,
            "effective_frames": distance_effective,
        },
        "speed_sample": {
            "alignment": alignment,
            "requested_frames": requested,
            "effective_frames": _bounded_smoothing_window(
                requested,
                int(sample_count),
            ),
        },
        "acceleration_sample": {
            "alignment": "centered",
            "requested_frames": requested,
            "effective_frames": _bounded_smoothing_window(
                requested,
                int(sample_count),
            ),
        },
        "heading_sample": {
            "alignment": "centered",
            "requested_frames": requested,
            "effective_frames": _bounded_smoothing_window(
                requested,
                int(sample_count),
            ),
        },
    }


def _validate_motion_group_semantic_attrs(
    run_group: Any,
    track_group: Any,
    relative_path: str,
    group: Any,
    *,
    track_id: int,
    include_physical: bool,
) -> None:
    attrs = dict(group.attrs)
    label = f"/{group.path}"
    if relative_path == "swim_bouts" or relative_path.startswith("swim_bouts/"):
        allowed = (
            _MOTION_SWIM_BOUT_ROOT_ATTR_NAMES
            if relative_path == "swim_bouts"
            else _MOTION_SWIM_BOUT_LEVEL_ATTR_NAMES
        ) | _MOTION_TRACK_GROUP_STORAGE_ATTR_NAMES
        unknown = sorted(set(attrs) - allowed)
        if unknown:
            raise ValueError(
                f"{label} has unsupported auxiliary group attrs: {unknown!r}."
            )
        _validate_motion_storage_attrs(
            attrs,
            label=label,
            required=False,
        )
        if relative_path == "swim_bouts":
            inputs = run_group.attrs.get("inputs")
            expected_run = (
                inputs.get("swim_bout_run")
                if isinstance(inputs, Mapping)
                else None
            )
            if expected_run is not None and attrs.get("source_swim_bout_run") != expected_run:
                raise ValueError(
                    f"{label} source swim-bout run conflicts with run inputs."
                )
        return

    _validate_motion_storage_attrs(attrs, label=label, required=True)
    storage_names = set(_MOTION_TRACK_GROUP_STORAGE_ATTR_NAMES)
    if relative_path == "":
        expected_names = storage_names | set(_MOTION_TRACK_ROOT_GROUP_ATTR_NAMES)
        if set(attrs) != expected_names:
            raise ValueError(
                f"{label} track-root group attr inventory is not closed "
                f"(expected={sorted(expected_names)!r}, "
                f"found={sorted(attrs)!r})."
            )
        fixed = {
            "track_id": int(track_id),
            "num_samples": int(track_group["track_sample_key"].shape[0]),
            "sample_validity_schema_id": "palette.track_sample_validity.v1",
            "sample_reason_codes": dict(SAMPLE_REASON_CODES),
            "transition_validity_schema_id": (
                "palette.track_transition_validity.v1"
            ),
            "transition_reason_codes": dict(TRANSITION_REASON_CODES),
            "motion_smoothing_windows": _expected_motion_smoothing_windows(
                run_group,
                sample_count=int(track_group["track_sample_key"].shape[0]),
            ),
            "legacy_acceleration_source_speed_level": (
                DEFAULT_ACCELERATION_SOURCE_SPEED_LEVEL
            ),
            "speed_derivatives_schema_id": SPEED_DERIVATIVES_SCHEMA_ID,
            "physical_outputs_status": run_group.attrs.get(
                "physical_outputs_status"
            ),
            "physical_outputs_reason_code": run_group.attrs.get(
                "physical_outputs_reason_code"
            ),
            "physical_coordinate_authority": run_group.attrs.get(
                "physical_coordinate_authority"
            ),
        }
        for name, expected in fixed.items():
            if not _track_attr_values_equal(attrs[name], expected):
                raise ValueError(
                    f"{label} group attr {name!r} conflicts with the controlled "
                    "track-motion metadata contract."
                )
        arena_id = attrs["arena_id"]
        if arena_id is not None and (
            isinstance(arena_id, (bool, np.bool_))
            or not isinstance(arena_id, (int, np.integer))
        ):
            raise ValueError(f"{label} arena_id must be one integer or null.")
        return

    expected = _motion_nested_group_expected_attrs(
        run_group,
        relative_path,
        include_physical=include_physical,
        sample_count=int(track_group["track_sample_key"].shape[0]),
    )
    expected_names = storage_names | set(expected)
    if set(attrs) != expected_names:
        raise ValueError(
            f"{label} group attr inventory is not closed "
            f"(expected={sorted(expected_names)!r}, found={sorted(attrs)!r})."
        )
    for name, value in expected.items():
        if not _track_attr_values_equal(attrs[name], value):
            raise ValueError(
                f"{label} group attr {name!r} conflicts with the controlled "
                "track-motion metadata contract."
            )


def _validate_motion_array_semantic_attrs(
    node: Any,
    contract: Mapping[str, Any],
    *,
    relative_path: str,
    position_surface: bool,
    source_identity_domain: str | None = None,
) -> None:
    attrs = dict(getattr(node, "attrs", {}))
    extras = _MOTION_ARRAY_EXTRA_ATTR_NAMES_BY_PATH.get(
        relative_path,
        frozenset(),
    )
    expected_names = set(_MOTION_ARRAY_STORAGE_ATTR_NAMES) | set(extras)
    if set(attrs) != expected_names:
        raise ValueError(
            f"/{node.path} array attr inventory is not closed "
            f"(expected={sorted(expected_names)!r}, found={sorted(attrs)!r})."
        )
    _validate_motion_storage_attrs(
        attrs,
        label=f"/{node.path}",
        required=True,
    )
    for name in _MOTION_ARRAY_DUPLICATE_SEMANTIC_ATTR_NAMES:
        if name not in attrs:
            continue
        expected = contract.get(name)
        if not _track_attr_values_equal(attrs[name], expected):
            raise ValueError(
                f"/{node.path} array attr {name!r} conflicts with its controlled "
                "track-motion surface contract."
            )
    forbidden = sorted(
        name
        for name in attrs
        if name in _MOTION_ARRAY_FORBIDDEN_COORDINATE_ATTR_NAMES
    )
    if forbidden:
        raise ValueError(
            f"/{node.path} carries forbidden duplicate coordinate attrs: "
            f"{forbidden!r}."
        )
    coordinate_attrs = sorted(
        name for name in attrs if str(name).startswith("coordinate_descriptor")
    )
    if coordinate_attrs and not position_surface:
        raise ValueError(
            f"/{node.path} is not a position surface but carries coordinate "
            f"descriptor attrs: {coordinate_attrs!r}."
        )
    if position_surface and relative_path not in {"positions_px", "positions_mm"}:
        raise ValueError("Internal position-surface classification is inconsistent.")
    if relative_path == "frame_indices":
        expected_identity = {
            "semantic_role": (
                "compatibility_alias_of_source_acquisition_frame_index"
            ),
            "authoritative_array_ref": (
                f"/{node.path.rsplit('/', 1)[0]}/"
                "source_acquisition_frame_index"
            ),
            "canonical_consumers_must_use": (
                "track_sample_key_and_source_acquisition_frame_index"
            ),
        }
        for name, expected in expected_identity.items():
            if not _track_attr_values_equal(attrs[name], expected):
                raise ValueError(
                    f"/{node.path} identity attr {name!r} conflicts with its "
                    "controlled compatibility role."
                )
    elif relative_path == "source_instance_key":
        if not isinstance(source_identity_domain, str) or not source_identity_domain:
            raise ValueError(
                "Source-instance array validation lacks its exact identity domain."
            )
        expected_identity = {
            "semantic_role": "nullable_source_observation_identity_lineage",
            "source_identity_domain": source_identity_domain,
            "nullable_target_domain": "observation_instance",
            "primary_row_identity": False,
            "null_encoding": "valid_false_instance_key_zero",
        }
        for name, expected in expected_identity.items():
            if not _track_attr_values_equal(attrs[name], expected):
                raise ValueError(
                    f"/{node.path} identity attr {name!r} conflicts with its "
                    "controlled nullable-source role."
                )


def _resolve_motion_manifest_pointer(
    context: Mapping[str, Any],
    pointer: str,
) -> Any:
    if not pointer.startswith("#/"):
        raise ValueError(f"Motion manifest pointer {pointer!r} is not canonical.")
    value: Any = context
    for encoded in pointer[2:].split("/"):
        name = encoded.replace("~1", "/").replace("~0", "~")
        if not isinstance(value, Mapping) or name not in value:
            raise ValueError(
                f"Motion derivation manifest pointer {pointer!r} does not resolve."
            )
        value = value[name]
    return value


def _bind_motion_input_refs(
    raw_refs: Any,
    *,
    run_group: Any,
    local_array_records: Mapping[str, Mapping[str, Any]],
    manifest_context: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not isinstance(raw_refs, list) or not all(
        isinstance(value, str) and value for value in raw_refs
    ):
        raise ValueError("Motion derivation input refs must be nonempty strings.")
    run_prefix = f"/{run_group.path}/"
    bound: list[dict[str, Any]] = []
    for raw_ref in raw_refs:
        if raw_ref.startswith("#/"):
            value = _resolve_motion_manifest_pointer(manifest_context, raw_ref)
            normalized = json_attr_safe(copy.deepcopy(value))
            record = _motion_json_object(
                {"value": normalized},
                label=f"motion manifest input {raw_ref}",
            )
            bound.append(
                {
                    "kind": (
                        "external_lineage"
                        if raw_ref.startswith(
                            "#/run_derivation/record/source_refs"
                        )
                        or raw_ref.startswith(
                            "#/run_derivation/record/inputs/"
                        )
                        else "manifest_record"
                    ),
                    "ref": raw_ref,
                    "record_sha256": _canonical_json_sha256(record),
                }
            )
            continue
        if not raw_ref.startswith(run_prefix):
            raise ValueError(
                f"Motion derivation input ref {raw_ref!r} escapes /{run_group.path}."
            )
        if "@" in raw_ref:
            group_ref, attr_name = raw_ref.rsplit("@", 1)
            if not attr_name:
                raise ValueError(
                    f"Motion group-attr input ref {raw_ref!r} lacks an attr name."
                )
            relative_group = group_ref[len(run_prefix) :]
            group = (
                run_group
                if not relative_group
                else _relative_child(run_group, relative_group)
            )
            if attr_name not in group.attrs:
                raise ValueError(
                    f"Motion group-attr input ref {raw_ref!r} does not resolve."
                )
            value_record = _motion_json_object(
                {"value": copy.deepcopy(group.attrs[attr_name])},
                label=f"motion group-attr input {raw_ref}",
            )
            bound.append(
                {
                    "kind": "group_attr",
                    "ref": raw_ref,
                    "value_sha256": _canonical_json_sha256(value_record),
                }
            )
            continue
        target = local_array_records.get(raw_ref)
        if target is None:
            raise ValueError(
                f"Motion local-array input ref {raw_ref!r} does not resolve."
            )
        bound.append(
            {
                "kind": "array",
                "ref": raw_ref,
                "dtype": target.get("dtype"),
                "shape": copy.deepcopy(target.get("shape")),
                "content_sha256": target.get("content_sha256"),
            }
        )
    return bound


def _bind_track_motion_surface_inputs(
    track_group: Any,
    records: dict[str, dict[str, Any]],
    *,
    run_group: Any,
    manifest_context: Mapping[str, Any],
) -> None:
    absolute = {
        f"/{track_group.path}/{relative_path}": record
        for relative_path, record in records.items()
    }
    for record in records.values():
        record["input_refs"] = _bind_motion_input_refs(
            record.get("input_refs"),
            run_group=run_group,
            local_array_records=absolute,
            manifest_context=manifest_context,
        )


def _motion_surface_record(
    track_group: Any,
    relative_path: str,
    node: Any,
    *,
    sample_count: int,
    second_count: int,
    row_identity_ref: str,
    row_identity_sha256: str,
    track_time_lineage_ref: str,
    track_time_lineage_sha256: str,
    second_identity_sha256: str,
    physical_authority_sha256: str | None,
    source_identity_domain: str,
) -> dict[str, Any]:
    contract = _motion_track_surface_contract(
        track_group,
        relative_path,
        physical_authority_sha256=physical_authority_sha256,
    )
    _validate_motion_array_semantic_attrs(
        node,
        contract,
        relative_path=relative_path,
        position_surface=relative_path in {"positions_px", "positions_mm"},
        source_identity_domain=source_identity_domain,
    )
    record = _stage_array_payload_record(
        node,
        relative_ref=relative_path,
        include_attrs=True,
    )
    axis = contract["axis0_domain"]
    shape = record["shape"]
    if not shape:
        raise ValueError(
            f"/{track_group.path}/{relative_path} must expose an axis-0 domain."
        )
    if axis in {
        TRACK_MOTION_AXIS_TRACK_SAMPLE,
        TRACK_MOTION_AXIS_TRACK_TRANSITION,
    }:
        if shape[0] != sample_count:
            raise ValueError(
                f"/{track_group.path}/{relative_path} axis-0 length {shape[0]} "
                f"does not match track-sample count {sample_count}."
            )
        contract["axis0_identity"] = {
            "domain": "track_sample",
            "key_array_ref": f"/{track_group.path}/track_sample_key",
            "row_identity_ref": row_identity_ref,
            "row_identity_sha256": row_identity_sha256,
            "track_time_lineage_ref": track_time_lineage_ref,
            "track_time_lineage_sha256": track_time_lineage_sha256,
            **(
                {"transition_anchor": "destination_track_sample"}
                if axis == TRACK_MOTION_AXIS_TRACK_TRANSITION
                else {}
            ),
        }
    elif axis == TRACK_MOTION_AXIS_TRACK_SECOND:
        if shape[0] != second_count:
            raise ValueError(
                f"/{track_group.path}/{relative_path} axis-0 length {shape[0]} "
                f"does not match second-bin count {second_count}."
            )
        contract["axis0_identity"] = {
            "domain": TRACK_MOTION_AXIS_TRACK_SECOND,
            "key_array_ref": f"/{track_group.path}/second_indices",
            "key_content_sha256": second_identity_sha256,
        }
    elif axis == TRACK_MOTION_AXIS_TRACK_BOUT:
        contract["axis0_identity"] = {
            "domain": TRACK_MOTION_AXIS_TRACK_BOUT,
            "group_ref": (
                f"/{track_group.path}/{relative_path.rsplit('/', 1)[0]}"
            ),
        }
    else:  # pragma: no cover - controlled contract bug
        raise ValueError(f"Unsupported track motion axis domain {axis!r}.")
    record.update(contract)
    return record


def _validate_motion_alias_records(
    track_group: Any,
    records: Mapping[str, Mapping[str, Any]],
) -> None:
    prefix = f"/{track_group.path}/"
    for relative_path, record in records.items():
        alias_ref = record.get("alias_of")
        if alias_ref is None:
            continue
        if not isinstance(alias_ref, str) or not alias_ref.startswith(prefix):
            raise ValueError(
                f"/{track_group.path}/{relative_path} has a nonlocal alias target."
            )
        target_path = alias_ref[len(prefix) :]
        target = records.get(target_path)
        if target is None:
            raise ValueError(
                f"/{track_group.path}/{relative_path} alias target "
                f"{target_path!r} is missing."
            )
        for field in ("dtype", "dtype_fields", "itemsize", "shape", "content_sha256"):
            if record.get(field) != target.get(field):
                raise ValueError(
                    f"/{track_group.path}/{relative_path} disagrees with exact "
                    f"alias target {target_path!r} in {field}."
                )


def _validate_motion_physical_values(
    track_group: Any,
    records: Mapping[str, Mapping[str, Any]],
    *,
    physical_authority: BoundStimulusPhysicalCoordinateAuthority | None,
) -> None:
    physical_records = {
        path: record
        for path, record in records.items()
        if record.get("physical_authority_sha256") is not None
    }
    if not physical_records:
        if physical_authority is not None:
            raise ValueError(
                f"/{track_group.path} has physical authority but no physical surfaces."
            )
        return
    if physical_authority is None:
        raise ValueError(
            f"/{track_group.path} has physical surfaces without exact authority."
        )
    scale_value = float(physical_authority.mm_per_pixel)
    if not math.isfinite(scale_value) or scale_value <= 0:
        raise ValueError("Track physical authority has invalid mm_per_pixel.")
    prefix = f"/{track_group.path}/"
    for relative_path, record in physical_records.items():
        pixel_ref = record.get("pixel_source_ref")
        if not isinstance(pixel_ref, str) or not pixel_ref.startswith(prefix):
            raise ValueError(
                f"/{track_group.path}/{relative_path} has invalid physical pixel peer."
            )
        pixel_path = pixel_ref[len(prefix) :]
        if pixel_path not in records:
            raise ValueError(
                f"/{track_group.path}/{relative_path} physical pixel peer is missing."
            )
        physical_values = np.array(
            _relative_child(track_group, relative_path)[:],
            copy=True,
            order="C",
        )
        pixel_values = np.array(
            _relative_child(track_group, pixel_path)[:],
            copy=True,
            order="C",
        )
        if (
            physical_values.dtype != pixel_values.dtype
            or physical_values.shape != pixel_values.shape
            or physical_values.dtype.kind != "f"
        ):
            raise ValueError(
                f"/{track_group.path}/{relative_path} physical dtype/shape differs "
                "from its exact pixel peer."
            )
        scale = np.asarray(scale_value, dtype=pixel_values.dtype)
        with np.errstate(over="ignore", invalid="ignore"):
            expected = np.asarray(pixel_values * scale, dtype=pixel_values.dtype)
        if (
            not np.array_equal(np.isnan(physical_values), np.isnan(expected))
            or not np.array_equal(np.isposinf(physical_values), np.isposinf(expected))
            or not np.array_equal(np.isneginf(physical_values), np.isneginf(expected))
        ):
            raise ValueError(
                f"/{track_group.path}/{relative_path} physical NaN/Inf mask differs "
                "from pixel * mm_per_pixel."
            )
        finite = np.isfinite(expected)
        if not np.array_equal(physical_values[finite], expected[finite]):
            raise ValueError(
                f"/{track_group.path}/{relative_path} does not exactly equal its "
                "pixel peer multiplied by authoritative mm_per_pixel."
            )


def _exact_motion_array(
    track_group: Any,
    relative_path: str,
    expected: Any,
) -> None:
    observed = np.array(
        _relative_child(track_group, relative_path)[:],
        copy=True,
        order="C",
    )
    expected_array = np.asarray(expected)
    if (
        observed.dtype != expected_array.dtype
        or observed.shape != expected_array.shape
        or not np.array_equal(observed, expected_array, equal_nan=True)
    ):
        raise ValueError(
            f"/{track_group.path}/{relative_path} violates its exact controlled "
            "numeric derivation invariant."
        )


def _float32_angle_pair_matches(
    left: np.ndarray,
    right: np.ndarray,
    *,
    left_to_right: Callable[[np.ndarray], np.ndarray],
) -> bool:
    if (
        left.dtype != np.dtype("<f4")
        or right.dtype != np.dtype("<f4")
        or left.shape != right.shape
    ):
        return False
    expected = np.asarray(
        left_to_right(left.astype(np.float64)),
        dtype=np.float32,
    )
    if (
        not np.array_equal(np.isnan(right), np.isnan(expected))
        or not np.array_equal(np.isposinf(right), np.isposinf(expected))
        or not np.array_equal(np.isneginf(right), np.isneginf(expected))
    ):
        return False
    finite = np.isfinite(expected)
    epsilon = np.finfo(np.float32).eps
    return bool(
        np.allclose(
            right[finite],
            expected[finite],
            rtol=4.0 * epsilon,
            atol=4.0 * epsilon,
        )
    )


def _validate_motion_core_numeric_invariants(
    run_group: Any,
    track_group: Any,
    *,
    track_id: int,
) -> None:
    """Re-evaluate bounded, controlled invariants using existing kernels."""

    parameters = run_group.attrs.get("parameters")
    if not isinstance(parameters, Mapping):
        raise ValueError("Track motion numeric validation requires parameters.")

    def numeric_parameter(name: str, *, positive: bool = False) -> float:
        raw = parameters.get(name, run_group.attrs.get(name))
        if isinstance(raw, (bool, np.bool_)):
            raise ValueError(f"Track parameter {name!r} is not numeric.")
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Track parameter {name!r} is not numeric.") from exc
        if not math.isfinite(value) or (positive and value <= 0):
            raise ValueError(f"Track parameter {name!r} is invalid.")
        if name in run_group.attrs and float(run_group.attrs[name]) != value:
            raise ValueError(
                f"Track root attr {name!r} conflicts with parameters.{name}."
            )
        return value

    fps = numeric_parameter("fps", positive=True)
    smooth_seconds = numeric_parameter("smoothing_seconds", positive=True)
    smoothing_method = str(
        parameters.get(
            "smoothing_method",
            run_group.attrs.get("smoothing_method", "moving_average"),
        )
    )
    smoothing_alignment = str(
        parameters.get(
            "smoothing_alignment",
            run_group.attrs.get(
                "smoothing_alignment",
                DEFAULT_SMOOTHING_ALIGNMENT,
            ),
        )
    )
    savgol_raw = parameters.get(
        "savgol_polyorder",
        run_group.attrs.get("savgol_polyorder"),
    )
    savgol_polyorder = 3 if savgol_raw is None else int(savgol_raw)
    hysteresis_enabled = bool(
        parameters.get(
            "hysteresis_enabled",
            run_group.attrs.get("hysteresis_enabled", False),
        )
    )

    def optional_float(name: str) -> float | None:
        if not hysteresis_enabled:
            return None
        raw = parameters.get(name, run_group.attrs.get(name))
        if raw is None:
            raise ValueError(
                f"Enabled track hysteresis lacks parameter {name!r}."
            )
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError(f"Track hysteresis parameter {name!r} is invalid.")
        return value

    hysteresis_high = optional_float("hysteresis_high_px")
    hysteresis_low = optional_float("hysteresis_low_px")
    hysteresis_min_raw = parameters.get(
        "hysteresis_min_frames",
        run_group.attrs.get("hysteresis_min_frames"),
    )
    hysteresis_min = (
        int(hysteresis_min_raw)
        if hysteresis_enabled and hysteresis_min_raw is not None
        else None
    )
    hysteresis_policy = str(
        parameters.get(
            "hysteresis_band_policy",
            run_group.attrs.get(
                "hysteresis_band_policy",
                DEFAULT_HYSTERESIS_BAND_POLICY,
            ),
        )
    )
    if hysteresis_policy not in HYSTERESIS_BAND_POLICIES:
        raise ValueError(
            "Track hysteresis_band_policy must remain one valid effective "
            f"policy even when hysteresis is disabled; got {hysteresis_policy!r}."
        )

    frames = np.array(
        track_group["source_acquisition_frame_index"][:],
        copy=True,
        order="C",
    )
    positions = np.array(
        track_group["positions_px"][:],
        copy=True,
        order="C",
    )
    headings = np.array(
        track_group["heading_degrees"][:],
        copy=True,
        order="C",
    )
    keypoint_success = np.array(
        track_group["keypoint_success"][:],
        copy=True,
        order="C",
    )
    detection_source = np.array(
        track_group["detection_source"][:],
        copy=True,
        order="C",
    )
    if frames.dtype != np.dtype("<i8"):
        raise ValueError("Track acquisition-frame identity must be exact int64.")

    _exact_motion_array(track_group, "frame_indices", frames)
    expected_key = build_track_sample_key(
        np.full(frames.shape, int(track_id), dtype=np.int64),
        frames,
    )
    _exact_motion_array(track_group, "track_sample_key", expected_key)
    expected_time = np.asarray(
        frames.astype(np.float64) / fps,
        dtype=np.float32,
    )
    _exact_motion_array(track_group, "time_seconds", expected_time)

    validity = _build_sample_validity_arrays(
        track_id=int(track_id),
        positions_px=positions,
        headings_deg=headings,
        keypoint_success=keypoint_success,
        detection_source=detection_source,
    )
    for name in (
        "sample_observed",
        "sample_valid",
        "source_observed",
        "keypoint_usable",
        "position_finite",
        "heading_usable",
        "sample_reason_code",
    ):
        expected = (
            validity[name].astype(np.int16)
            if name == "sample_reason_code"
            else validity[name].astype(bool)
        )
        _exact_motion_array(track_group, name, expected)

    recomputed = compute_track_speed(
        frames.copy(),
        positions.copy(),
        fps=fps,
        smooth_seconds=smooth_seconds,
        hysteresis_high_px=hysteresis_high,
        hysteresis_low_px=hysteresis_low,
        hysteresis_min_frames=hysteresis_min,
        hysteresis_band_policy=hysteresis_policy,
        smoothing_method=smoothing_method,
        smoothing_alignment=smoothing_alignment,
        savgol_polyorder=savgol_polyorder,
    )
    exact_speed_surfaces = {
        "delta_frames": recomputed.delta_frames,
        "delta_seconds": recomputed.delta_seconds,
        "transition_valid": recomputed.transition_valid,
        "transition_reason_code": recomputed.transition_reason_code,
        "speed_raw_px": recomputed.speed_raw,
        "speed_filtered_px": recomputed.speed_filtered,
        "speed_smoothed_px": recomputed.speed_smoothed,
        "speed_averaged_px": recomputed.speed_averaged,
        "frame_path_distance_raw_px": recomputed.frame_path_distance_raw,
        "frame_path_distance_filtered_px": recomputed.frame_path_distance_filtered,
        "frame_path_distance_smoothed_px": recomputed.frame_path_distance_smoothed,
        "cumulative_path_distance_px": recomputed.cumulative_path_distance,
        "second_indices": recomputed.seconds,
        "speed_per_second_px": recomputed.speed_per_second,
    }
    for path, expected in exact_speed_surfaces.items():
        _exact_motion_array(track_group, path, expected)

    heading_radians = np.array(track_group["heading_radians"][:], copy=True)
    smoothed_degrees = np.array(
        track_group["smoothed_heading_degrees"][:], copy=True
    )
    smoothed_radians = np.array(
        track_group["smoothed_heading_radians"][:], copy=True
    )
    if not _float32_angle_pair_matches(
        headings,
        heading_radians,
        left_to_right=np.deg2rad,
    ):
        raise ValueError("Track heading degree/radian surfaces disagree.")
    if not _float32_angle_pair_matches(
        smoothed_degrees,
        smoothed_radians,
        left_to_right=np.deg2rad,
    ):
        raise ValueError("Track smoothed-heading degree/radian surfaces disagree.")

    # Reuse the writer's bounded circular-heading kernel in the exact persisted
    # float32 heading domain.
    heading_radians_kernel = np.deg2rad(headings)
    heading_finite = np.isfinite(heading_radians_kernel)
    heading_window_requested = max(1, int(round(fps * smooth_seconds)))
    expected_smoothed_radians, _heading_window_effective = (
        _smooth_heading_radians(headings, heading_window_requested)
    )
    expected_smoothed_degrees = np.rad2deg(expected_smoothed_radians)
    _exact_motion_array(
        track_group,
        "smoothed_heading_radians",
        np.asarray(expected_smoothed_radians, dtype=np.float32),
    )
    _exact_motion_array(
        track_group,
        "smoothed_heading_degrees",
        np.asarray(expected_smoothed_degrees, dtype=np.float32),
    )

    delta_seconds_kernel = np.zeros(frames.shape[0], dtype=np.float64)
    if frames.size >= 2:
        delta_seconds_kernel[1:] = np.diff(frames) / fps
    raw_turning = _compute_heading_turning(
        headings,
        delta_seconds_kernel,
        transition_valid=recomputed.transition_valid,
        sample_valid=validity["sample_valid"],
    )
    smoothed_turning = _compute_heading_turning(
        expected_smoothed_degrees,
        delta_seconds_kernel,
        transition_valid=recomputed.transition_valid,
        sample_valid=validity["sample_valid"],
    )
    turning_surfaces = {
        "delta_heading_degrees": raw_turning[0],
        "angular_velocity_deg_s": raw_turning[1],
        "angular_velocity_raw_deg_s": raw_turning[1],
        "angular_speed_raw_deg_s": raw_turning[2],
        "delta_heading_smoothed_degrees": smoothed_turning[0],
        "angular_velocity_smoothed_deg_s": smoothed_turning[1],
        "angular_speed_smoothed_deg_s": smoothed_turning[2],
    }
    for path, expected in turning_surfaces.items():
        _exact_motion_array(
            track_group,
            path,
            np.asarray(expected, dtype=np.float32),
        )

    derivatives = _compute_speed_derivatives(
        {
            "speed_raw": recomputed.speed_raw,
            "speed_filtered": recomputed.speed_filtered,
            "speed_smoothed": recomputed.speed_smoothed,
            "speed_averaged": recomputed.speed_averaged,
        },
        delta_seconds_kernel,
        pixel_to_mm=None,
        smooth_seconds=smooth_seconds,
        fps=fps,
    )
    for level, derivative in derivatives.items():
        for leaf in ("acceleration_px", "smoothed_acceleration_px"):
            _exact_motion_array(
                track_group,
                f"speed_derivatives/{level}/{leaf}",
                np.asarray(derivative[leaf], dtype=np.float32),
            )

    seconds = recomputed.seconds
    for path in (
        "speed_per_second_px",
        "heading_per_second_degrees",
        "heading_per_second_resultant",
    ):
        if int(track_group[path].shape[0]) != int(seconds.shape[0]):
            raise ValueError(
                f"/{track_group.path}/{path} is not aligned to unique second bins."
            )
    seconds_per_sample = np.floor(frames.astype(np.float64) / fps).astype(
        np.int64
    )
    expected_heading_by_second = np.full(
        seconds.size,
        np.nan,
        dtype=np.float64,
    )
    expected_resultant = np.zeros(seconds.size, dtype=np.float32)
    for index, second in enumerate(seconds):
        mask = (seconds_per_sample == second) & heading_finite
        valid_angles = heading_radians_kernel[mask]
        if valid_angles.size:
            mean_vector = np.mean(np.exp(1j * valid_angles))
            expected_heading_by_second[index] = math.atan2(
                mean_vector.imag,
                mean_vector.real,
            )
            expected_resultant[index] = np.float32(np.abs(mean_vector))
    _exact_motion_array(
        track_group,
        "heading_per_second_degrees",
        np.asarray(
            np.rad2deg(expected_heading_by_second),
            dtype=np.float32,
        ),
    )
    _exact_motion_array(
        track_group,
        "heading_per_second_resultant",
        expected_resultant,
    )

    # Track summaries are public numeric surfaces too.  Recompute every
    # pixel/non-physical field from the same bounded kernels above instead of
    # treating a self-consistent, re-minted attrs payload as authoritative.
    summary = track_group.attrs.get("summary")
    if type(summary) is not dict:
        raise ValueError(f"/{track_group.path} summary attrs are invalid.")

    def speed_statistics(values: np.ndarray) -> tuple[float, float, float]:
        finite = values[np.isfinite(values)]
        if not finite.size:
            nan = float("nan")
            return nan, nan, nan
        return (
            float(np.mean(finite)),
            float(np.median(finite)),
            float(np.max(finite)),
        )

    expected_summary: dict[str, Any] = {
        "track_id": float(track_id),
        "samples": int(frames.size),
    }
    for level_name, values in (
        ("raw", recomputed.speed_raw),
        ("filtered", recomputed.speed_filtered),
        ("smoothed", recomputed.speed_smoothed),
        ("averaged", recomputed.speed_averaged),
    ):
        mean_value, median_value, max_value = speed_statistics(values)
        expected_summary.update(
            {
                f"mean_speed_{level_name}_px": mean_value,
                f"median_speed_{level_name}_px": median_value,
                f"max_speed_{level_name}_px": max_value,
            }
        )

    speed_per_second = recomputed.speed_per_second
    expected_summary["mean_speed_per_second_px"] = (
        float(np.nanmean(speed_per_second))
        if speed_per_second.size and np.any(~np.isnan(speed_per_second))
        else float("nan")
    )
    expected_summary.update(
        {
            "total_path_distance_raw_px": (
                float(np.sum(recomputed.frame_path_distance_raw))
                if recomputed.frame_path_distance_raw.size
                else 0.0
            ),
            "total_path_distance_filtered_px": (
                float(np.sum(recomputed.frame_path_distance_filtered))
                if recomputed.frame_path_distance_filtered.size
                else 0.0
            ),
            "total_path_distance_smoothed_px": (
                float(np.sum(recomputed.frame_path_distance_smoothed))
                if recomputed.frame_path_distance_smoothed.size
                else 0.0
            ),
            "total_distance_px": (
                float(recomputed.cumulative_path_distance[-1])
                if recomputed.cumulative_path_distance.size
                else 0.0
            ),
        }
    )

    valid_headings = heading_radians_kernel[heading_finite]
    if valid_headings.size:
        mean_vector = np.mean(np.exp(1j * valid_headings))
        expected_summary["heading_mean_deg"] = float(
            math.degrees(math.atan2(mean_vector.imag, mean_vector.real))
        )
        expected_summary["heading_resultant"] = float(np.abs(mean_vector))
    else:
        expected_summary["heading_mean_deg"] = float("nan")
        expected_summary["heading_resultant"] = float("nan")

    default_derivative = derivatives[DEFAULT_ACCELERATION_SOURCE_SPEED_LEVEL]
    summary_acceleration = np.asarray(
        default_derivative["smoothed_acceleration_px"],
        dtype=np.float64,
    )
    finite_acceleration = summary_acceleration[np.isfinite(summary_acceleration)]
    expected_summary["mean_acceleration_px"] = (
        float(np.mean(finite_acceleration))
        if finite_acceleration.size
        else float("nan")
    )
    expected_summary["acceleration_std_px"] = (
        float(np.std(finite_acceleration))
        if finite_acceleration.size
        else float("nan")
    )
    expected_summary["keypoint_success_rate"] = (
        float(np.mean(keypoint_success))
        if keypoint_success.size
        else float("nan")
    )
    summary_times = frames.astype(np.float64) / fps
    expected_summary["duration_seconds"] = (
        float(summary_times[-1] - summary_times[0])
        if summary_times.size > 1
        else 0.0
    )

    normalized_expected_summary = json_attr_safe(expected_summary)
    if type(normalized_expected_summary) is not dict:  # pragma: no cover
        raise ValueError("Recomputed track summary is not a JSON object.")
    expected_summary_names = set(normalized_expected_summary)
    if run_group.attrs.get("physical_coordinate_authority") is not None:
        expected_summary_names.update(
            f"{name[:-3]}_mm"
            for name in normalized_expected_summary
            if name.endswith("_px")
        )
    if set(summary) != expected_summary_names:
        raise ValueError(
            f"/{track_group.path} summary inventory differs from the controlled "
            f"numeric schema (expected={sorted(expected_summary_names)!r}, "
            f"found={sorted(summary)!r})."
        )
    for name, expected in normalized_expected_summary.items():
        if not _track_attr_values_equal(summary[name], expected):
            raise ValueError(
                f"/{track_group.path} summary field {name!r} violates its exact "
                "controlled numeric derivation invariant."
            )

    resultant = np.array(track_group["heading_per_second_resultant"][:], copy=True)
    finite_resultant = resultant[np.isfinite(resultant)]
    if np.any(finite_resultant < 0.0) or np.any(finite_resultant > 1.0):
        raise ValueError("Track heading resultants must remain within [0, 1].")


def _validate_motion_bout_domains(
    track_group: Any,
    records: Mapping[str, Mapping[str, Any]],
) -> None:
    counts: dict[str, int] = {}
    for relative_path, record in records.items():
        if record.get("axis0_domain") != TRACK_MOTION_AXIS_TRACK_BOUT:
            continue
        parent = relative_path.rsplit("/", 1)[0]
        count = int(record["shape"][0])
        prior = counts.setdefault(parent, count)
        if prior != count:
            raise ValueError(
                f"/{track_group.path}/{parent} mirrored bout fields disagree in "
                "axis-0 length."
            )


def _allowed_motion_track_group(relative_path: str) -> bool:
    if relative_path in {
        "",
        "movement",
        "movement/speed",
        "speed_derivatives",
    }:
        return True
    if relative_path.startswith("movement/speed/"):
        return relative_path.count("/") == 2 and relative_path.rsplit("/", 1)[-1] in (
            set(MOVEMENT_SPEED_LEVEL_NAMES.values())
        )
    if relative_path.startswith("speed_derivatives/"):
        return relative_path.count("/") == 1 and relative_path.rsplit("/", 1)[-1] in (
            set(SPEED_DERIVATIVE_LEVELS)
        )
    return relative_path == "swim_bouts" or (
        relative_path.startswith("swim_bouts/")
        and relative_path.count("/") == 1
    )


def _build_track_motion_publication_manifest(
    authoritative_root: Any,
    run_group: Any,
    positions: BoundTrackPositionBindings,
) -> dict[str, Any]:
    """Build and validate the exact live full-motion inventory."""

    if positions.run_group.path != run_group.path:
        raise ValueError("Position bindings and motion run paths disagree.")
    root_group_names = sorted(str(value) for value in run_group.group_keys())
    if root_group_names != ["tracks"]:
        raise ValueError(
            f"/{run_group.path} full-motion root group inventory is not closed "
            f"(expected=['tracks'], found={root_group_names!r})."
        )
    groups = _live_track_groups(run_group)
    if [track_id for track_id, _ in groups] != [
        track_id for track_id, _ in positions.track_positions
    ]:
        raise ValueError("Position bindings and live track inventory disagree.")
    input_authority, _input_values = _validate_track_motion_input_authority(
        authoritative_root,
        run_group,
        positions,
        groups,
    )
    physical_record = _physical_authority_manifest_record(
        positions.physical_authority
    )
    physical_sha256 = (
        _canonical_json_sha256(physical_record)
        if physical_record is not None
        else None
    )
    include_physical = physical_record is not None
    source_authority = _motion_source_authority_record(positions)
    run_derivation = _motion_run_derivation_record(run_group, positions)
    run_root_attrs = _motion_run_root_attrs_record(run_group, positions)
    manifest_context: dict[str, Any] = {
        "source_authority": source_authority,
        "physical_authority": physical_record,
        "run_derivation": run_derivation,
        "input_authority": input_authority["record"],
    }
    expected_paths = _expected_motion_track_surface_paths(
        include_physical=include_physical
    )
    position_by_id = dict(positions.track_positions)

    tracks_parent = run_group["tracks"]
    tracks_parent_attrs = dict(tracks_parent.attrs)
    expected_tracks_parent_attrs = geometry_preload_attrs()
    if set(tracks_parent_attrs) != set(expected_tracks_parent_attrs):
        raise ValueError(
            f"/{tracks_parent.path} attr inventory is not closed "
            f"(expected={sorted(expected_tracks_parent_attrs)!r}, "
            f"found={sorted(tracks_parent_attrs)!r})."
        )
    _validate_motion_storage_attrs(
        tracks_parent_attrs,
        label=f"/{tracks_parent.path}",
        required=True,
    )
    tracks_parent_attr_record = _motion_group_attrs_record(tracks_parent)
    track_records: dict[str, Any] = {}
    for track_id, track_group in groups:
        position = position_by_id[track_id]
        track_key = np.array(
            track_group["track_sample_key"][:], copy=True, order="C"
        )
        second_key = np.array(
            track_group["second_indices"][:], copy=True, order="C"
        )
        if track_key.dtype != np.dtype("<i8") or track_key.ndim != 2 or track_key.shape[1] != 2:
            raise ValueError(
                f"/{track_group.path}/track_sample_key is not exact int64 (N, 2)."
            )
        if (
            second_key.dtype != np.dtype("<i8")
            or second_key.ndim != 1
            or (
                second_key.size > 1
                and np.any(np.diff(second_key) <= 0)
            )
        ):
            raise ValueError(
                f"/{track_group.path}/second_indices is not a strictly increasing "
                "int64 second-bin identity."
            )
        sample_count = int(track_key.shape[0])
        second_count = int(second_key.shape[0])
        row_identity = position.positions_px.row_identity
        if row_identity.leading_dimension != sample_count:
            raise ValueError(
                f"/{track_group.path} row identity and track sample count disagree."
            )
        track_time_lineage = load_bound_track_sample_time_lineage(
            track_group,
            track_group["track_sample_key"],
            track_group["source_row_index"],
            track_group["source_acquisition_frame_index"],
            track_group["source_frame_interpolation"],
            track_group["source_instance_key"],
            source_temporal_authority=positions.source_temporal_authority,
        )

        group_records: dict[str, Any] = {}
        for relative_group, group in _iter_motion_group_nodes(track_group):
            if not _allowed_motion_track_group(relative_group):
                raise ValueError(
                    f"/{track_group.path}/{relative_group} is an unexpected "
                    "full-motion group."
                )
            _validate_motion_group_semantic_attrs(
                run_group,
                track_group,
                relative_group,
                group,
                track_id=track_id,
                include_physical=include_physical,
            )
            group_records[relative_group or "."] = {
                "relative_ref": relative_group or ".",
                **_motion_group_attrs_record(group),
                "array_names": sorted(str(name) for name in group.array_keys()),
                "group_names": sorted(str(name) for name in group.group_keys()),
            }

        surface_nodes = dict(_iter_track_array_nodes(track_group))
        public_paths = {
            path
            for path in surface_nodes
            if not path.startswith("swim_bouts/")
        }
        if public_paths != expected_paths:
            raise ValueError(
                f"/{track_group.path} full-motion array inventory differs from "
                f"the controlled schema (missing={sorted(expected_paths - public_paths)!r}, "
                f"extra={sorted(public_paths - expected_paths)!r})."
            )
        second_digest = array_payload_sha256(track_group["second_indices"])
        surfaces = {
            path: _motion_surface_record(
                track_group,
                path,
                node,
                sample_count=sample_count,
                second_count=second_count,
                row_identity_ref=row_identity.record_ref,
                row_identity_sha256=row_identity.record_sha256,
                track_time_lineage_ref=track_time_lineage.record_ref,
                track_time_lineage_sha256=track_time_lineage.record_sha256,
                second_identity_sha256=second_digest,
                physical_authority_sha256=physical_sha256,
                source_identity_domain=(
                    positions.source_temporal_authority.record.source_identity_domain
                ),
            )
            for path, node in sorted(surface_nodes.items())
        }
        _bind_track_motion_surface_inputs(
            track_group,
            surfaces,
            run_group=run_group,
            manifest_context=manifest_context,
        )
        _validate_motion_core_numeric_invariants(
            run_group,
            track_group,
            track_id=track_id,
        )
        _validate_motion_physical_values(
            track_group,
            surfaces,
            physical_authority=positions.physical_authority,
        )
        _validate_motion_alias_records(track_group, surfaces)
        _validate_motion_bout_domains(track_group, surfaces)
        track_records[f"id_{track_id}"] = {
            "track_id": int(track_id),
            "track_ref": f"/{track_group.path}",
            "track_sample_count": sample_count,
            "second_bin_count": second_count,
            "row_identity_ref": row_identity.record_ref,
            "row_identity_sha256": row_identity.record_sha256,
            "track_time_lineage_ref": track_time_lineage.record_ref,
            "track_time_lineage_sha256": track_time_lineage.record_sha256,
            "position_derivation_ref": position.derivation.record_ref,
            "position_derivation_sha256": position.derivation.record_sha256,
            "groups": group_records,
            "surfaces": surfaces,
        }

    run_arrays: dict[str, Any] = {}
    track_count = len(groups)
    root_array_names = sorted(str(value) for value in run_group.array_keys())
    camera_count = (
        int(run_group["camera_frame_ids"].shape[0])
        if "camera_frame_ids" in root_array_names
        else None
    )
    for name in root_array_names:
        node = run_group[name]
        contract = _motion_run_array_contract(run_group, name)
        _validate_motion_array_semantic_attrs(
            node,
            contract,
            relative_path=name,
            position_surface=False,
        )
        record = _stage_array_payload_record(
            node,
            relative_ref=name,
            include_attrs=True,
        )
        record.update(contract)
        if not record["shape"]:
            raise ValueError(f"/{node.path} must expose an axis-0 domain.")
        if contract["axis0_domain"] == TRACK_MOTION_AXIS_RUN_TRACK:
            if record["shape"][0] != track_count:
                raise ValueError(
                    f"/{node.path} axis-0 length differs from track inventory."
                )
            record["axis0_identity"] = {
                "domain": TRACK_MOTION_AXIS_RUN_TRACK,
                "key_array_ref": f"/{run_group.path}/track_ids",
            }
        else:
            if camera_count is None:
                raise ValueError(
                    "Run camera-sample auxiliary arrays require camera_frame_ids "
                    "as their exact row identity."
                )
            if record["shape"][0] != camera_count:
                raise ValueError(
                    f"/{node.path} axis-0 length differs from camera_frame_ids."
                )
            record["axis0_identity"] = {
                "domain": TRACK_MOTION_AXIS_RUN_CAMERA_SAMPLE,
                "key_array_ref": f"/{run_group.path}/camera_frame_ids",
            }
        run_arrays[name] = record
    if "track_ids" not in run_arrays:
        raise ValueError("Full-motion run lacks exact track_ids inventory.")
    track_ids_digest = str(run_arrays["track_ids"]["content_sha256"])
    camera_ids_digest = (
        str(run_arrays["camera_frame_ids"]["content_sha256"])
        if "camera_frame_ids" in run_arrays
        else None
    )
    for record in run_arrays.values():
        identity = record.get("axis0_identity")
        if not isinstance(identity, dict):
            raise ValueError("Run motion array lacks exact axis-0 identity.")
        if record.get("axis0_domain") == TRACK_MOTION_AXIS_RUN_TRACK:
            identity["key_content_sha256"] = track_ids_digest
        elif camera_ids_digest is not None:
            identity["key_content_sha256"] = camera_ids_digest
    manifest_context["tracks"] = track_records
    absolute_run_arrays = {
        f"/{run_group.path}/{name}": record
        for name, record in run_arrays.items()
    }
    for record in run_arrays.values():
        record["input_refs"] = _bind_motion_input_refs(
            record.get("input_refs"),
            run_group=run_group,
            local_array_records=absolute_run_arrays,
            manifest_context=manifest_context,
        )

    manifest = {
        "schema_id": TRACK_MOTION_PUBLICATION_MANIFEST_SCHEMA_ID,
        "schema_version": TRACK_MOTION_PUBLICATION_MANIFEST_SCHEMA_VERSION,
        "run_ref": f"/{run_group.path}",
        "run_type": positions.run_type,
        "run_name": positions.run_name,
        "coordinate_binding_status": run_group.attrs.get(
            TRACK_KINEMATICS_COORDINATE_BINDING_STATUS_ATTR
        ),
        "source_authority": source_authority,
        "input_authority": input_authority,
        "physical_authority": physical_record,
        "run_derivation": run_derivation,
        "run_root_attrs": run_root_attrs,
        "run_group_inventory": {
            "array_names": root_array_names,
            "group_names": root_group_names,
        },
        "tracks_group_inventory": {
            **tracks_parent_attr_record,
            "array_names": sorted(
                str(name) for name in tracks_parent.array_keys()
            ),
            "group_names": sorted(
                str(name) for name in tracks_parent.group_keys()
            ),
        },
        "run_arrays": run_arrays,
        "track_count": track_count,
        "tracks": track_records,
    }
    return _motion_json_object(
        manifest,
        label=f"/{run_group.path} full-motion publication manifest",
    )


def _track_motion_publication_commit(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    tracks = manifest.get("tracks")
    source = manifest.get("source_authority")
    derivation = manifest.get("run_derivation")
    input_authority = manifest.get("input_authority")
    if not isinstance(tracks, Mapping) or not isinstance(
        source, Mapping
    ) or not isinstance(derivation, Mapping) or not isinstance(
        input_authority, Mapping
    ):
        raise ValueError("Full-motion manifest cannot mint a publication commit.")
    position_derivations = {
        str(name): {
            "record_ref": record.get("position_derivation_ref"),
            "record_sha256": record.get("position_derivation_sha256"),
        }
        for name, record in tracks.items()
        if isinstance(record, Mapping)
    }
    commit = {
        "schema_id": TRACK_MOTION_PUBLICATION_COMMIT_SCHEMA_ID,
        "schema_version": TRACK_MOTION_PUBLICATION_COMMIT_SCHEMA_VERSION,
        "run_ref": manifest.get("run_ref"),
        "manifest_sha256": _canonical_json_sha256(manifest),
        "source_authority_sha256": _canonical_json_sha256(source),
        "input_authority_sha256": input_authority.get("record_sha256"),
        "run_derivation_sha256": derivation.get("record_sha256"),
        "position_derivations": position_derivations,
    }
    return _motion_json_object(commit, label="track motion publication commit")


def _build_track_staging_manifest(
    run_group: zarr.Group,
    *,
    ordered_ids: List[int],
    source_positions: BoundCanonicalCoordinateDescriptor,
    source_temporal_authority: Any,
    physical_authority: TrackPhysicalAuthority | None,
    physical_omission_reason_code: str,
    keypoint_run: str,
    run_name: str,
) -> dict[str, Any]:
    source = require_bound_canonical_coordinate_descriptor(source_positions)
    temporal = require_bound_source_row_temporal_authority(
        source_temporal_authority
    )
    source_coordinate_path = str(getattr(source.coordinate_node, "path", ""))
    if not source_coordinate_path:
        raise ValueError("Canonical source positions have no persisted array path.")
    tracks_group = run_group["tracks"]
    track_records: dict[str, Any] = {}
    for track_id in ordered_ids:
        name = f"id_{int(track_id)}"
        subgroup = tracks_group[name]
        arrays: dict[str, Any] = {}
        for array_name in _TRACK_STAGING_CRITICAL_ARRAYS:
            node = subgroup[array_name]
            relative_ref = f"tracks/{name}/{array_name}"
            if str(getattr(node, "path", "")) != f"{run_group.path}/{relative_ref}":
                raise ValueError(
                    f"Staged track array /{node.path} is outside its exact run path."
                )
            arrays[array_name] = _stage_array_payload_record(
                node,
                relative_ref=relative_ref,
            )
        physical_arrays = {
            relative_path: _stage_array_payload_record(
                node,
                relative_ref=f"tracks/{name}/{relative_path}",
            )
            for relative_path, node in _track_physical_array_nodes(
                subgroup
            ).items()
        }
        if (physical_authority is None) != (not physical_arrays):
            raise ValueError(
                f"{name} physical arrays must exist exactly when the staged run "
                "binds an exact stimulus physical authority."
            )
        if physical_arrays and "positions_mm" not in physical_arrays:
            raise ValueError(
                f"{name} staged physical payload lacks authoritative positions_mm."
            )
        track_records[name] = {
            "track_id": int(track_id),
            "row_count": int(subgroup.attrs["num_samples"]),
            "arrays": arrays,
            "physical_arrays": physical_arrays,
            "summary": copy.deepcopy(subgroup.attrs.get("summary")),
        }
    track_ids_node = run_group["track_ids"]
    manifest = {
        "schema_id": TRACK_KINEMATICS_STAGING_MANIFEST_SCHEMA_ID,
        "schema_version": TRACK_KINEMATICS_STAGING_MANIFEST_SCHEMA_VERSION,
        "run_name": str(run_name),
        "keypoint_run": str(keypoint_run),
        "run_ref": f"/{run_group.path}",
        "source_archive": _archive_identity_manifest_record(
            source.coordinate_node
        ),
        "source_positions": {
            "array_ref": f"/{source_coordinate_path}",
            "dtype": np.dtype(source.coordinate_node.dtype).str,
            "shape": [int(item) for item in source.coordinate_node.shape],
            "content_sha256": array_payload_sha256(source.coordinate_node),
            "coordinate_descriptor_sha256": source.descriptor.digest(),
            "row_identity_ref": source.row_identity.record_ref,
            "row_identity_sha256": source.row_identity.record_sha256,
        },
        "source_temporal_authority": {
            "record_ref": temporal.record_ref,
            "record_sha256": temporal.record_sha256,
        },
        "physical_authority": _physical_authority_manifest_record(
            physical_authority
        ),
        "physical_omission_reason_code": (
            "NONE"
            if physical_authority is not None
            else str(physical_omission_reason_code)
        ),
        "track_ids": _stage_array_payload_record(
            track_ids_node,
            relative_ref="track_ids",
        ),
        "track_count": len(ordered_ids),
        "tracks": track_records,
        "run_physical_surfaces": _run_physical_surface_record(run_group),
    }
    _canonical_json_sha256(manifest)
    return manifest


def save_track_kinematics_tracks(
    run_group: zarr.Group,
    tracks: Dict[int, Dict[str, Any]],
    summaries: List[Dict[str, float]],
    *,
    source_temporal_authority: Any,
    positions_px_source: BoundCanonicalCoordinateDescriptor,
    input_authority: BoundTrackMotionInputAuthority | None = None,
    physical_frame: BoundPhysicalFrameCalibration | None = None,
    physical_authority: TrackPhysicalAuthority | None = None,
    physical_omission_reason_code: str = "NO_COMPATIBLE_TYPED_PHYSICAL_FRAME",
    track_id_to_arena_id: Optional[Dict[int, int]] = None,
    defer_coordinate_binding: bool = False,
    staging_keypoint_run: str | None = None,
    staging_run_name: str | None = None,
) -> List[int]:
    """Persist per-track data beneath the track kinematics run group.

    Every normally published track row is bound to the exact
    acquisition-camera frame domain.  The explicit deferred mode writes only
    a numerically validated, fail-closed staging artifact; it is reserved for
    the final-path materializer and contains no canonical coordinate claims.
    ``source_instance_key`` remains nullable observation lineage and can never
    substitute for the primary ``track_sample_key``.
    """

    if physical_frame is not None:
        raise ValueError(
            "Detached physical_frame values cannot authorize canonical track mm "
            "outputs; supply a sealed source-camera physical authority."
        )
    if type(physical_authority) is BoundStimulusPhysicalCoordinateAuthority:
        physical_authority = require_bound_stimulus_physical_coordinate_authority(
            physical_authority
        )
        physical_frame = physical_authority.physical_frame
    elif physical_authority is not None:
        physical_authority = require_bound_source_camera_physical_authority(
            physical_authority
        )
        physical_frame = physical_authority.physical_frame
    if defer_coordinate_binding:
        if not staging_keypoint_run or not staging_run_name:
            raise ValueError(
                "Deferred track staging requires exact keypoint and run names."
            )

    source_temporal_authority = require_bound_source_row_temporal_authority(
        source_temporal_authority
    )
    positions_px_source = require_bound_canonical_coordinate_descriptor(
        positions_px_source
    )
    if input_authority is not None:
        if (
            type(input_authority) is not BoundTrackMotionInputAuthority
            or getattr(input_authority, "_seal", None)
            is not _BOUND_TRACK_MOTION_INPUT_AUTHORITY_SEAL
        ):
            raise ValueError(
                "Track input authority was not minted from exact live arrays."
            )
        if input_authority.archive_identity != archive_identity(
            positions_px_source.coordinate_node
        ):
            raise ValueError(
                "Track input authority and selected positions belong to different "
                "archives."
            )
        authority_record = _motion_json_object(
            _thaw_motion_manifest(input_authority.record),
            label="track-motion writer input authority",
        )
        position_identity = authority_record.get("position_row_identity")
        if (
            not isinstance(position_identity, Mapping)
            or position_identity.get("record_ref")
            != positions_px_source.row_identity.record_ref
            or position_identity.get("record_sha256")
            != positions_px_source.row_identity.record_sha256
        ):
            raise ValueError(
                "Track input authority does not bind the selected position row identity."
            )
        run_group.attrs[TRACK_MOTION_INPUT_AUTHORITY_ATTR] = authority_record
    include_physical = physical_authority is not None
    if physical_authority is not None:
        if physical_authority.archive_identity != archive_identity(
            positions_px_source.coordinate_node
        ):
            raise ValueError(
                "Track physical authority and selected positions must belong to "
                "the same exact archive."
            )
    if include_physical:
        physical_omission_reason_code = "NONE"
    elif (
        not isinstance(physical_omission_reason_code, str)
        or not physical_omission_reason_code
        or physical_omission_reason_code != physical_omission_reason_code.strip()
    ):
        raise ValueError(
            "Omitted physical outputs require one stable nonempty reason code."
        )
    if (
        positions_px_source.row_identity.record_ref
        != source_temporal_authority.source_row_identity.record_ref
        or positions_px_source.row_identity.record_sha256
        != source_temporal_authority.source_row_identity.record_sha256
    ):
        raise ValueError(
            "Selected positions and immediate-source temporal authority do not "
            "bind the same exact row identity."
        )
    source_positions_values = np.array(
        positions_px_source.coordinate_node[:],
        copy=True,
        order="C",
    )
    if (
        source_positions_values.dtype
        != np.dtype(positions_px_source.coordinate_node.dtype)
        or source_positions_values.shape
        != positions_px_source.coordinate_node.shape
        or source_positions_values.ndim != 2
        or source_positions_values.shape[1] != 2
    ):
        raise ValueError(
            "Selected source positions must expose one exact numeric (N, 2) payload."
        )
    ordered_ids = sorted(int(track_id) for track_id in tracks.keys())
    summary_ids: list[int] = []
    for index, summary in enumerate(summaries):
        if not isinstance(summary, Mapping) or "track_id" not in summary:
            raise ValueError(f"Track summary {index} lacks one track_id.")
        try:
            summary_id = int(summary["track_id"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Track summary {index} has an invalid track_id.") from exc
        if float(summary["track_id"]) != float(summary_id):
            raise ValueError(f"Track summary {index} has a nonintegral track_id.")
        summary_ids.append(summary_id)
    if len(set(summary_ids)) != len(summary_ids) or set(summary_ids) != set(
        ordered_ids
    ):
        raise ValueError(
            "Track summaries must name every track exactly once."
        )
    summary_by_id = {
        track_id: summaries[summary_ids.index(track_id)]
        for track_id in ordered_ids
    }
    persisted_summary_by_id: dict[int, dict[str, Any]] = {}
    for track_id in ordered_ids:
        summary = summary_by_id[track_id]
        if physical_authority is not None:
            _validate_track_summary_physical_fields(
                summary,
                physical_authority=physical_authority,
                label=f"Track {track_id} input summary",
            )
        persisted_summary_by_id[track_id] = _persisted_track_summary(
            summary,
            include_physical=include_physical,
        )
    selected_source_rows: list[np.ndarray] = []
    for track_id in ordered_ids:
        data = tracks[track_id]
        required = {
            "frame_indices",
            "track_sample_key",
            "source_acquisition_frame_index",
            "source_frame_interpolation",
            "source_instance_key",
            "source_row_index",
        }
        missing = sorted(required.difference(data))
        if missing:
            raise ValueError(
                f"Track {track_id} lacks canonical identity/time-lineage arrays: "
                f"{missing!r}."
            )
        frame_indices = np.asarray(data["frame_indices"])
        key = np.asarray(data["track_sample_key"])
        source_frames = np.asarray(data["source_acquisition_frame_index"])
        interpolation = np.asarray(data["source_frame_interpolation"])
        source_instances = np.asarray(data["source_instance_key"])
        source_rows = np.asarray(data["source_row_index"])
        row_count = int(frame_indices.shape[0]) if frame_indices.ndim == 1 else -1
        if (
            frame_indices.dtype != np.dtype("<i8")
            or key.dtype != np.dtype("<i8")
            or key.shape != (row_count, 2)
            or source_frames.dtype != np.dtype("<i8")
            or source_frames.shape != (row_count,)
            or interpolation.dtype != TRACK_SAMPLE_INTERPOLATION_DTYPE
            or interpolation.shape != (row_count,)
            or source_instances.dtype != TRACK_SAMPLE_SOURCE_INSTANCE_KEY_DTYPE
            or source_instances.shape != (row_count,)
            or source_rows.dtype != np.dtype("<i8")
            or source_rows.shape != (row_count,)
        ):
            raise ValueError(
                f"Track {track_id} identity/time-lineage arrays do not use the "
                "canonical dtypes and shapes."
            )
        if (
            not np.array_equal(frame_indices, source_frames)
            or not np.array_equal(key[:, 1], source_frames)
            or np.any(key[:, 0] != track_id)
        ):
            raise ValueError(
                f"Track {track_id} keys do not exactly equal its acquisition-frame mapping."
            )
        expected_frames = resolve_source_acquisition_frame_indices(
            source_temporal_authority,
            source_rows,
        )
        expected_instances = derive_track_source_instance_values(
            source_temporal_authority,
            source_rows,
        )
        if not np.array_equal(source_frames, expected_frames):
            raise ValueError(
                f"Track {track_id} acquisition frames are not the exact selected "
                "immediate-source rows."
            )
        if not np.array_equal(source_instances, expected_instances):
            raise ValueError(
                f"Track {track_id} source instance lineage was not mechanically "
                "derived from its immediate source row identity."
            )
        if np.any(~source_instances["valid"] & (source_instances["instance_key"] != 0)):
            raise ValueError(
                f"Track {track_id} uses a noncanonical nullable source-instance value."
            )
        positions_px = np.asarray(data["positions_px"])
        if (
            positions_px.dtype != source_positions_values.dtype
            or positions_px.shape != (row_count, 2)
            or not np.array_equal(
                positions_px,
                source_positions_values[source_rows],
                equal_nan=True,
            )
        ):
            raise ValueError(
                f"Track {track_id} positions_px is not an exact dtype-preserving "
                "subset/reorder of the selected source coordinate surface."
            )
        if physical_frame is not None:
            assert physical_authority is not None
            _validate_in_memory_track_physical_arrays(
                data,
                track_id=track_id,
                physical_authority=physical_authority,
            )
        selected_source_rows.append(source_rows)
    if selected_source_rows:
        all_source_rows = np.concatenate(selected_source_rows)
        if np.unique(all_source_rows).shape[0] != all_source_rows.shape[0]:
            raise ValueError(
                "Canonical track publication requires a unique subset/reorder of "
                "the immediate source rowset; source_row_index values repeat."
            )

    tracks_parent = run_group.create_group("tracks")
    stamp_geometry_preload_attrs(run_group)
    stamp_geometry_preload_attrs(tracks_parent)
    track_ids_array = np.asarray(ordered_ids, dtype=np.int32)
    chunks = _track_preload_chunks(track_ids_array.shape) or (1,)
    run_group.create_array("track_ids", data=track_ids_array, chunks=chunks, overwrite=True)
    stamp_geometry_preload_attrs(run_group["track_ids"])
    track_arena_ids = _ordered_track_arena_ids(ordered_ids, track_id_to_arena_id)
    if track_arena_ids is not None:
        run_group.create_array(
            "track_arena_ids",
            data=track_arena_ids,
            chunks=chunks,
            overwrite=True,
        )
        stamp_geometry_preload_attrs(run_group["track_arena_ids"])

    manifest: List[Dict[str, float]] = []
    for track_id in ordered_ids:
        data = tracks[track_id]
        summary = summary_by_id[track_id]
        persisted_summary = persisted_summary_by_id[track_id]
        subgroup = tracks_parent.create_group(f"id_{track_id}")

        sample_count = int(data["frame_indices"].size)
        base_chunk = _track_preload_chunks(data["frame_indices"].shape) or (1,)

        frame_indices_array = subgroup.create_array(
            "frame_indices",
            data=data["frame_indices"],
            chunks=base_chunk,
            overwrite=True,
        )
        frame_indices_array.attrs.update(
            {
                "semantic_role": (
                    "compatibility_alias_of_source_acquisition_frame_index"
                ),
                "authoritative_array_ref": (
                    f"/{subgroup.path}/source_acquisition_frame_index"
                ),
                "canonical_consumers_must_use": (
                    "track_sample_key_and_source_acquisition_frame_index"
                ),
            }
        )
        track_sample_key_array = subgroup.create_array(
            "track_sample_key",
            data=data["track_sample_key"],
            chunks=(base_chunk[0], 2),
            overwrite=True,
        )
        if "source_frame_interpolation" not in data:
            raise ValueError(
                "Canonical track publication requires explicit acquisition-frame "
                "interpolation lineage for every sample."
            )
        source_acquisition_frame_array = subgroup.create_array(
            "source_acquisition_frame_index",
            data=data["source_acquisition_frame_index"],
            chunks=base_chunk,
            overwrite=True,
        )
        source_frame_interpolation_array = subgroup.create_array(
            "source_frame_interpolation",
            data=data["source_frame_interpolation"],
            chunks=base_chunk,
            overwrite=True,
        )
        source_instance_array = subgroup.create_array(
            "source_instance_key",
            data=data["source_instance_key"],
            chunks=base_chunk,
            overwrite=True,
        )
        source_row_index_array = subgroup.create_array(
            "source_row_index",
            data=data["source_row_index"],
            chunks=base_chunk,
            overwrite=True,
        )
        source_instance_array.attrs.update(
            {
                "semantic_role": "nullable_source_observation_identity_lineage",
                "source_identity_domain": (
                    source_temporal_authority.record.source_identity_domain
                ),
                "nullable_target_domain": "observation_instance",
                "primary_row_identity": False,
                "null_encoding": "valid_false_instance_key_zero",
            }
        )
        bound_track_identity = None
        if not defer_coordinate_binding:
            track_time_lineage = stamp_track_sample_time_lineage(
                subgroup,
                track_sample_key_array,
                source_row_index_array,
                source_acquisition_frame_array,
                source_frame_interpolation_array,
                source_instance_array,
                source_temporal_authority=source_temporal_authority,
            )
            track_identity = build_row_identity_contract(
                domain=TRACK_SAMPLE_DOMAIN,
                values=data["track_sample_key"],
                track_time_lineage=track_time_lineage,
            )
            bound_track_identity = stamp_and_bind_row_identity_contract(
                subgroup,
                track_sample_key_array,
                contract=track_identity,
                track_time_lineage=track_time_lineage,
            )
        subgroup.create_array("time_seconds", data=data["time_seconds"], chunks=base_chunk, overwrite=True)
        positions_px_array = subgroup.create_array(
            "positions_px",
            data=data["positions_px"],
            chunks=(base_chunk[0], 2),
            overwrite=True,
        )
        positions_mm_array = None
        if physical_frame is not None:
            positions_mm_array = subgroup.create_array(
                "positions_mm",
                data=data["positions_mm"],
                chunks=(base_chunk[0], 2),
                overwrite=True,
            )
        if not defer_coordinate_binding:
            assert bound_track_identity is not None
            publish_track_position_coordinates(
                subgroup,
                positions_px_array,
                source_row_index_array,
                track_row_identity=bound_track_identity,
                source_positions=positions_px_source,
                source_temporal_authority=source_temporal_authority,
                positions_mm_node=positions_mm_array,
                physical_frame=physical_frame,
            )
        subgroup.create_array("heading_degrees", data=data["heading_degrees"], chunks=base_chunk, overwrite=True)
        subgroup.create_array("heading_radians", data=data["heading_radians"], chunks=base_chunk, overwrite=True)
        subgroup.create_array("delta_heading_degrees", data=data["delta_heading_degrees"], chunks=base_chunk, overwrite=True)
        subgroup.create_array("angular_velocity_deg_s", data=data["angular_velocity_deg_s"], chunks=base_chunk, overwrite=True)
        subgroup.create_array(
            "angular_velocity_raw_deg_s",
            data=data["angular_velocity_raw_deg_s"],
            chunks=base_chunk,
            overwrite=True,
        )
        subgroup.create_array(
            "angular_speed_raw_deg_s",
            data=data["angular_speed_raw_deg_s"],
            chunks=base_chunk,
            overwrite=True,
        )
        subgroup.create_array(
            "delta_heading_smoothed_degrees",
            data=data["delta_heading_smoothed_degrees"],
            chunks=base_chunk,
            overwrite=True,
        )
        subgroup.create_array(
            "angular_velocity_smoothed_deg_s",
            data=data["angular_velocity_smoothed_deg_s"],
            chunks=base_chunk,
            overwrite=True,
        )
        subgroup.create_array(
            "angular_speed_smoothed_deg_s",
            data=data["angular_speed_smoothed_deg_s"],
            chunks=base_chunk,
            overwrite=True,
        )
        subgroup.create_array("smoothed_heading_degrees", data=data["smoothed_heading_degrees"], chunks=base_chunk, overwrite=True)
        subgroup.create_array("smoothed_heading_radians", data=data["smoothed_heading_radians"], chunks=base_chunk, overwrite=True)
        subgroup.create_array("keypoint_success", data=data["keypoint_success"], chunks=base_chunk, overwrite=True)
        subgroup.create_array("detection_source", data=data["detection_source"], chunks=base_chunk, overwrite=True)
        subgroup.create_array("sample_observed", data=data["sample_observed"], chunks=base_chunk, overwrite=True)
        subgroup.create_array("sample_valid", data=data["sample_valid"], chunks=base_chunk, overwrite=True)
        subgroup.create_array("source_observed", data=data["source_observed"], chunks=base_chunk, overwrite=True)
        subgroup.create_array("keypoint_usable", data=data["keypoint_usable"], chunks=base_chunk, overwrite=True)
        subgroup.create_array("position_finite", data=data["position_finite"], chunks=base_chunk, overwrite=True)
        subgroup.create_array("heading_usable", data=data["heading_usable"], chunks=base_chunk, overwrite=True)
        subgroup.create_array("sample_reason_code", data=data["sample_reason_code"], chunks=base_chunk, overwrite=True)
        subgroup.create_array("delta_frames", data=data["delta_frames"], chunks=base_chunk, overwrite=True)
        subgroup.create_array("delta_seconds", data=data["delta_seconds"], chunks=base_chunk, overwrite=True)
        subgroup.create_array("transition_valid", data=data["transition_valid"], chunks=base_chunk, overwrite=True)
        subgroup.create_array(
            "transition_reason_code",
            data=data["transition_reason_code"],
            chunks=base_chunk,
            overwrite=True,
        )
        subgroup.create_array("speed_raw_px", data=data["speed_raw_px"], chunks=base_chunk, overwrite=True)
        subgroup.create_array("speed_filtered_px", data=data["speed_filtered_px"], chunks=base_chunk, overwrite=True)
        subgroup.create_array("speed_smoothed_px", data=data["speed_smoothed_px"], chunks=base_chunk, overwrite=True)
        subgroup.create_array("speed_averaged_px", data=data["speed_averaged_px"], chunks=base_chunk, overwrite=True)
        subgroup.create_array("acceleration_px", data=data["acceleration_px"], chunks=base_chunk, overwrite=True)
        subgroup.create_array("smoothed_acceleration_px", data=data["smoothed_acceleration_px"], chunks=base_chunk, overwrite=True)
        if include_physical:
            for name in (
                "speed_raw_mm",
                "speed_filtered_mm",
                "speed_smoothed_mm",
                "speed_averaged_mm",
                "acceleration_mm",
                "smoothed_acceleration_mm",
            ):
                subgroup.create_array(
                    name,
                    data=data[name],
                    chunks=base_chunk,
                    overwrite=True,
                )
        subgroup.attrs["legacy_acceleration_source_speed_level"] = DEFAULT_ACCELERATION_SOURCE_SPEED_LEVEL
        subgroup.attrs["speed_derivatives_schema_id"] = SPEED_DERIVATIVES_SCHEMA_ID
        _write_speed_derivative_groups(
            subgroup,
            data["speed_derivatives"],
            chunks=base_chunk,
            include_physical=include_physical,
        )
        _write_movement_speed_groups(
            subgroup,
            data,
            chunks=base_chunk,
            include_physical=include_physical,
        )
        subgroup.create_array(
            "frame_path_distance_raw_px",
            data=data["frame_path_distance_raw_px"],
            chunks=base_chunk,
            overwrite=True,
        )
        subgroup.create_array(
            "frame_path_distance_filtered_px",
            data=data["frame_path_distance_filtered_px"],
            chunks=base_chunk,
            overwrite=True,
        )
        subgroup.create_array(
            "frame_path_distance_smoothed_px",
            data=data["frame_path_distance_smoothed_px"],
            chunks=base_chunk,
            overwrite=True,
        )
        subgroup.create_array(
            "cumulative_path_distance_px",
            data=data["cumulative_path_distance_px"],
            chunks=base_chunk,
            overwrite=True,
        )
        if include_physical:
            for name in (
                "frame_path_distance_raw_mm",
                "frame_path_distance_filtered_mm",
                "frame_path_distance_smoothed_mm",
                "cumulative_path_distance_mm",
            ):
                subgroup.create_array(
                    name,
                    data=data[name],
                    chunks=base_chunk,
                    overwrite=True,
                )

        seconds = data["second_indices"]
        sec_chunk = _track_preload_chunks(seconds.shape) or (1,)
        subgroup.create_array("second_indices", data=seconds, chunks=sec_chunk, overwrite=True)
        subgroup.create_array("speed_per_second_px", data=data["speed_per_second_px"], chunks=sec_chunk, overwrite=True)
        if include_physical:
            subgroup.create_array(
                "speed_per_second_mm",
                data=data["speed_per_second_mm"],
                chunks=sec_chunk,
                overwrite=True,
            )
        subgroup.create_array("heading_per_second_degrees", data=data["heading_per_second_degrees"], chunks=sec_chunk, overwrite=True)
        subgroup.create_array("heading_per_second_resultant", data=data["heading_per_second_resultant"], chunks=sec_chunk, overwrite=True)

        subgroup.attrs.update(
            {
                "track_id": int(track_id),
                "arena_id": (
                    int(track_id_to_arena_id[track_id])
                    if track_id_to_arena_id and track_id in track_id_to_arena_id
                    else None
                ),
                "num_samples": sample_count,
                "sample_validity_schema_id": "palette.track_sample_validity.v1",
                "sample_reason_codes": dict(SAMPLE_REASON_CODES),
                "transition_validity_schema_id": "palette.track_transition_validity.v1",
                "transition_reason_codes": dict(TRANSITION_REASON_CODES),
                "motion_smoothing_windows": copy.deepcopy(
                    data["motion_smoothing_windows"]
                ),
                "summary": persisted_summary,
                "physical_outputs_status": (
                    "available_typed_source_camera_frame"
                    if include_physical
                    else "omitted_no_compatible_typed_physical_frame"
                ),
                "physical_outputs_reason_code": (
                    "NONE"
                    if include_physical
                    else physical_omission_reason_code
                ),
                "physical_coordinate_authority": (
                    _physical_authority_manifest_record(physical_authority)
                ),
            }
        )
        _stamp_geometry_preload_tree(subgroup)

        manifest_entry = json_attr_safe({
                "track_id": int(track_id),
                "arena_id": (
                    int(track_id_to_arena_id[track_id])
                    if track_id_to_arena_id and track_id in track_id_to_arena_id
                    else None
                ),
                "group": f"tracks/id_{track_id}",
                "samples": sample_count,
                "total_distance_px": persisted_summary.get("total_distance_px"),
                "heading_mean_deg": persisted_summary.get("heading_mean_deg"),
                "heading_resultant": persisted_summary.get("heading_resultant"),
                "mean_acceleration_px": persisted_summary.get("mean_acceleration_px"),
        })
        if not isinstance(manifest_entry, dict):
            raise ValueError("Track manifest entry did not normalize to an object.")
        if include_physical:
            manifest_entry.update(
                {
                    "total_distance_mm": persisted_summary.get(
                        "total_distance_mm"
                    ),
                    "mean_acceleration_mm": persisted_summary.get(
                        "mean_acceleration_mm"
                    ),
                }
            )
        manifest.append(manifest_entry)

    persisted_run_summary = [
        copy.deepcopy(persisted_summary_by_id[track_id])
        for track_id in ordered_ids
    ]
    run_group.attrs["summary"] = persisted_run_summary
    run_group.attrs["track_manifest"] = manifest
    run_group.attrs.update(
        {
            "physical_outputs_status": (
                "available_typed_source_camera_frame"
                if include_physical
                else "omitted_no_compatible_typed_physical_frame"
            ),
            "physical_outputs_reason_code": (
                "NONE"
                if include_physical
                else physical_omission_reason_code
            ),
            "physical_coordinate_authority": (
                _physical_authority_manifest_record(physical_authority)
            ),
            "total_distance_px": _finite_summary_total(
                persisted_run_summary,
                "total_distance_px",
            ),
        }
    )
    if include_physical:
        run_group.attrs["total_distance_mm"] = (
            float(run_group.attrs["total_distance_px"])
            * float(physical_authority.mm_per_pixel)
        )
    elif "total_distance_mm" in run_group.attrs:
        del run_group.attrs["total_distance_mm"]
    groups = [
        (track_id, run_group[f"tracks/id_{track_id}"])
        for track_id in ordered_ids
    ]
    _validate_run_track_physical_surfaces(
        run_group,
        groups=groups,
        physical_authority=physical_authority,
        physical_omission_reason_code=physical_omission_reason_code,
        binding_status=(
            TRACK_KINEMATICS_UNBOUND_STAGE_STATUS
            if defer_coordinate_binding
            else TRACK_KINEMATICS_BOUND_CANONICAL_STATUS
        ),
    )
    if defer_coordinate_binding:
        assert staging_keypoint_run is not None
        assert staging_run_name is not None
        staging_manifest = _build_track_staging_manifest(
            run_group,
            ordered_ids=ordered_ids,
            source_positions=positions_px_source,
            source_temporal_authority=source_temporal_authority,
            physical_authority=physical_authority,
            physical_omission_reason_code=physical_omission_reason_code,
            keypoint_run=staging_keypoint_run,
            run_name=staging_run_name,
        )
        run_group.attrs.update(
            {
                TRACK_KINEMATICS_STAGING_MANIFEST_ATTR: staging_manifest,
                TRACK_KINEMATICS_STAGING_MANIFEST_DIGEST_ATTR: (
                    _canonical_json_sha256(staging_manifest)
                ),
                TRACK_KINEMATICS_COORDINATE_BINDING_STATUS_ATTR: (
                    TRACK_KINEMATICS_UNBOUND_STAGE_STATUS
                ),
            }
        )
    else:
        run_group.attrs[TRACK_KINEMATICS_COORDINATE_BINDING_STATUS_ATTR] = (
            TRACK_KINEMATICS_BOUND_CANONICAL_STATUS
        )
    return ordered_ids


def _track_attr_values_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return set(left) == set(right) and all(
            _track_attr_values_equal(left[name], right[name]) for name in left
        )
    if type(left) in {list, tuple}:
        return len(left) == len(right) and all(
            _track_attr_values_equal(a, b)
            for a, b in zip(left, right, strict=True)
        )
    if type(left) is float and math.isnan(left) and math.isnan(right):
        return True
    return bool(left == right)


def _validate_run_track_physical_surfaces(
    run_group: Any,
    *,
    groups: list[tuple[int, Any]],
    physical_authority: TrackPhysicalAuthority | None,
    physical_omission_reason_code: str,
    binding_status: str,
    expected_track_records: Mapping[str, Any] | None = None,
    expected_run_record: Mapping[str, Any] | None = None,
) -> None:
    """Validate every persisted mm-bearing track/run surface as one unit."""

    _validate_no_run_root_coordinate_arrays(run_group)
    include_physical = physical_authority is not None
    expected_authority = _physical_authority_manifest_record(physical_authority)
    expected_status = (
        "available_typed_source_camera_frame"
        if include_physical
        else "omitted_no_compatible_typed_physical_frame"
    )
    expected_reason = "NONE" if include_physical else physical_omission_reason_code
    if (
        run_group.attrs.get("physical_coordinate_authority")
        != expected_authority
        or run_group.attrs.get("physical_outputs_status") != expected_status
        or run_group.attrs.get("physical_outputs_reason_code") != expected_reason
    ):
        raise ValueError(
            "Track run physical status/reason/authority is inconsistent."
        )
    if not include_physical:
        forbidden = sorted(
            str(name)
            for name in run_group.attrs
            if _is_mm_summary_field(name)
        )
        if forbidden:
            raise ValueError(
                "Omitted track run retains physical aggregate attrs: "
                f"{forbidden!r}."
            )

    run_summary = run_group.attrs.get("summary")
    track_manifest = run_group.attrs.get("track_manifest")
    if (
        type(run_summary) is not list
        or type(track_manifest) is not list
        or len(run_summary) != len(groups)
        or len(track_manifest) != len(groups)
    ):
        raise ValueError(
            "Track run summary/manifest does not match its exact track inventory."
        )

    run_track_arenas: np.ndarray | None = None
    if "track_arena_ids" in run_group:
        run_track_arenas = np.array(
            run_group["track_arena_ids"][:],
            copy=True,
            order="C",
        )
        if (
            run_track_arenas.dtype != np.dtype("<i4")
            or run_track_arenas.shape != (len(groups),)
        ):
            raise ValueError(
                "Run track_arena_ids must be exact int32 and align with track_ids."
            )

    for index, (track_id, subgroup) in enumerate(groups):
        name = f"id_{track_id}"
        if (
            subgroup.attrs.get("physical_coordinate_authority")
            != expected_authority
            or subgroup.attrs.get("physical_outputs_status") != expected_status
            or subgroup.attrs.get("physical_outputs_reason_code") != expected_reason
        ):
            raise ValueError(
                f"{name} physical status/reason/authority is inconsistent."
            )
        summary = subgroup.attrs.get("summary")
        if not isinstance(summary, Mapping):
            raise ValueError(f"{name} summary attrs are invalid.")
        expected_track_record = (
            expected_track_records.get(name)
            if expected_track_records is not None
            else None
        )
        if expected_track_records is not None:
            if type(expected_track_record) is not dict or not _track_attr_values_equal(
                dict(summary),
                expected_track_record.get("summary"),
            ):
                raise ValueError(
                    f"{name} summary changed after numerical staging."
                )
        _validate_track_summary_physical_fields(
            summary,
            physical_authority=physical_authority,
            label=f"{name} persisted summary",
        )
        if not _track_attr_values_equal(run_summary[index], dict(summary)):
            raise ValueError(
                f"Run summary entry {index} differs from {name} summary attrs."
            )

        manifest_entry = track_manifest[index]
        if not isinstance(manifest_entry, Mapping):
            raise ValueError(f"{name} track_manifest entry is invalid.")
        if (
            manifest_entry.get("track_id") != track_id
            or manifest_entry.get("group") != f"tracks/{name}"
            or manifest_entry.get("samples") != subgroup.attrs.get("num_samples")
        ):
            raise ValueError(f"{name} track_manifest identity is invalid.")
        subgroup_arena = subgroup.attrs.get("arena_id")
        if manifest_entry.get("arena_id") != subgroup_arena:
            raise ValueError(
                f"{name} track_manifest arena_id differs from subgroup authority."
            )
        if run_track_arenas is not None and (
            subgroup_arena is None
            or int(run_track_arenas[index]) != int(subgroup_arena)
        ):
            raise ValueError(
                f"{name} run, subgroup, and manifest arena identities disagree."
            )
        _validate_track_summary_physical_fields(
            manifest_entry,
            physical_authority=physical_authority,
            label=f"{name} track_manifest",
        )
        for field in (
            "total_distance_px",
            "mean_acceleration_px",
            "total_distance_mm",
            "mean_acceleration_mm",
        ):
            if field in manifest_entry or field in summary:
                if not _track_attr_values_equal(
                    manifest_entry.get(field),
                    summary.get(field),
                ):
                    raise ValueError(
                        f"{name} track_manifest {field} differs from its summary."
                    )
        physical_records = (
            expected_track_record.get("physical_arrays")
            if expected_track_record is not None
            else None
        )
        _validate_track_physical_arrays(
            subgroup,
            expected_records=physical_records,
            physical_authority=physical_authority,
            binding_status=binding_status,
        )

    expected_total_px = _finite_summary_total(run_summary, "total_distance_px")
    observed_total_px = _summary_number(
        run_group.attrs.get("total_distance_px"),
        label="track run total_distance_px",
    )
    if observed_total_px != expected_total_px:
        raise ValueError(
            "Track run total_distance_px differs from its sealed run summary."
        )
    if include_physical:
        if "total_distance_mm" not in run_group.attrs:
            raise ValueError("Physical track run lacks total_distance_mm.")
        _validate_scaled_scalar_pair(
            observed_total_px,
            run_group.attrs.get("total_distance_mm"),
            mm_per_pixel=physical_authority.mm_per_pixel,
            label="track run total_distance_mm",
        )
    elif "total_distance_mm" in run_group.attrs:
        raise ValueError("Omitted physical track run retains total_distance_mm.")

    live_run_record = _run_physical_surface_record(run_group)
    if expected_run_record is not None and not _track_attr_values_equal(
        live_run_record,
        dict(expected_run_record),
    ):
        raise ValueError(
            "Track run summary, manifest, or aggregate attrs changed after staging."
        )


def _restore_track_attrs(attrs: Any, snapshot: Mapping[str, Any]) -> None:
    for name in tuple(attrs.keys()):
        del attrs[name]
    attrs.update(copy.deepcopy(dict(snapshot)))
    if not _track_attr_values_equal(dict(attrs), dict(snapshot)):
        raise RuntimeError("Track coordinate attrs rollback was not exact.")


def _manifest_source_rowset_ref(manifest: Mapping[str, Any]) -> str:
    source = manifest.get("source_positions")
    if type(source) is not dict:
        raise ValueError("Track staging manifest lacks exact source_positions.")
    array_ref = source.get("array_ref")
    if type(array_ref) is not str or not array_ref.startswith("/"):
        raise ValueError("Track staging source coordinate ref is not canonical.")
    parts = array_ref[1:].split("/")
    if (
        len(parts) < 2
        or any(not part or part in {".", ".."} for part in parts)
        or parts[-1] != "centers_img_xy"
    ):
        raise ValueError("Track staging source coordinate ref is not canonical.")
    return "/".join(parts[:-1])


def _load_track_staging_manifest(
    run_group: zarr.Group,
    *,
    expected_keypoint_run: str,
    expected_run_name: str,
) -> tuple[dict[str, Any], str]:
    raw = run_group.attrs.get(TRACK_KINEMATICS_STAGING_MANIFEST_ATTR)
    if type(raw) is not dict:
        raise ValueError("Track run lacks one exact typed staging manifest.")
    manifest = copy.deepcopy(raw)
    expected_digest = _canonical_json_sha256(manifest)
    if (
        manifest.get("schema_id")
        != TRACK_KINEMATICS_STAGING_MANIFEST_SCHEMA_ID
        or type(manifest.get("schema_version")) is not int
        or manifest.get("schema_version")
        != TRACK_KINEMATICS_STAGING_MANIFEST_SCHEMA_VERSION
        or manifest.get("run_name") != expected_run_name
        or manifest.get("keypoint_run") != expected_keypoint_run
        or manifest.get("run_ref") != f"/{run_group.path}"
        or run_group.attrs.get(TRACK_KINEMATICS_STAGING_MANIFEST_DIGEST_ATTR)
        != expected_digest
    ):
        raise ValueError(
            "Track staging manifest schema, identity, path, or digest is invalid."
        )
    return manifest, expected_digest


def _require_stage_source_surface(
    authoritative_root: zarr.Group,
    manifest: Mapping[str, Any],
) -> BoundSourceCameraPositionSurface:
    rowset_ref = _manifest_source_rowset_ref(manifest)
    surface = require_bound_source_camera_position_surface(
        load_persisted_source_camera_position_surface(
            authoritative_root,
            rowset_ref,
        )
    )
    source = manifest["source_positions"]
    descriptor = surface.coordinates
    expected = {
        "array_ref": f"/{descriptor.coordinate_node.path}",
        "dtype": np.dtype(descriptor.coordinate_node.dtype).str,
        "shape": [int(item) for item in descriptor.coordinate_node.shape],
        "content_sha256": array_payload_sha256(descriptor.coordinate_node),
        "coordinate_descriptor_sha256": descriptor.descriptor.digest(),
        "row_identity_ref": descriptor.row_identity.record_ref,
        "row_identity_sha256": descriptor.row_identity.record_sha256,
    }
    temporal = manifest.get("source_temporal_authority")
    expected_temporal = {
        "record_ref": surface.temporal_authority.record_ref,
        "record_sha256": surface.temporal_authority.record_sha256,
    }
    if (
        source != expected
        or temporal != expected_temporal
        or manifest.get("source_archive")
        != _archive_identity_manifest_record(descriptor.coordinate_node)
    ):
        raise ValueError(
            "Authoritative source positions, identity, time, descriptor, or "
            "archive identity changed after numerical staging."
        )
    return surface


def _require_stage_physical_authority(
    authoritative_root: zarr.Group,
    manifest: Mapping[str, Any],
) -> TrackPhysicalAuthority | None:
    record = manifest.get("physical_authority")
    reason = manifest.get("physical_omission_reason_code")
    if record is None:
        if (
            not isinstance(reason, str)
            or not reason
            or reason == "NONE"
            or reason != reason.strip()
        ):
            raise ValueError(
                "Track staging omission requires one stable physical reason code."
            )
        return None
    common_fields = {
        "camera_id",
        "authority_manifest_ref",
        "authority_manifest_sha256",
        "physical_frame_ref",
        "physical_frame_sha256",
        "selected_camera_evidence_ref",
        "selected_camera_evidence_sha256",
        "source_camera_frame_ref",
        "source_camera_frame_sha256",
        "mm_per_pixel",
    }
    authority_kind = record.get("authority_kind") if type(record) is dict else None
    selector_fields = (
        {"stimulus_run"}
        if authority_kind is None and type(record) is dict and "stimulus_run" in record
        else {"authority_kind", "recording_calibration"}
        if authority_kind == "recording_calibration"
        else set()
    )
    if type(record) is not dict or set(record) != common_fields | selector_fields:
        raise ValueError(
            "Track staging physical authority record is absent or not closed."
        )
    if reason != "NONE":
        raise ValueError(
            "A staged physical authority cannot carry an omission reason."
        )
    if authority_kind is None:
        stimulus_run = record.get("stimulus_run")
        if not isinstance(stimulus_run, str) or not stimulus_run:
            raise ValueError("Track staging physical stimulus run is invalid.")
        authority = load_stimulus_physical_coordinate_authority(
            authoritative_root,
            stimulus_run=stimulus_run,
        )
        if authority is None:
            raise ValueError(
                "Selected stimulus no longer provides the staged physical authority."
            )
        authority = require_bound_stimulus_physical_coordinate_authority(authority)
    else:
        if record.get("recording_calibration") is not True:
            raise ValueError("Track staging recording calibration selector is invalid.")
        authority = require_bound_source_camera_physical_authority(
            load_source_camera_physical_authority(authoritative_root)
        )
    if _physical_authority_manifest_record(authority) != record:
        raise ValueError(
            "Source-camera physical authority, frame, calibration, scale, or "
            "digest changed after numerical staging."
        )
    return authority


def _relative_child(group: Any, relative_path: str) -> Any:
    current = group
    for part in relative_path.split("/"):
        if not part or part in {".", ".."}:
            raise ValueError("Track physical array path is noncanonical.")
        current = current[part]
    return current


def _physical_to_pixel_array_path(relative_path: str) -> str:
    parent, separator, leaf = relative_path.rpartition("/")
    if leaf == "mm":
        pixel_leaf = "px"
    elif leaf.endswith("_mm"):
        pixel_leaf = f"{leaf[:-3]}_px"
    else:
        raise ValueError(
            f"Physical array {relative_path!r} has no exact pixel-pair rule."
        )
    return f"{parent}{separator}{pixel_leaf}" if parent else pixel_leaf


def _validate_track_physical_arrays(
    subgroup: Any,
    *,
    expected_records: Mapping[str, Any] | None,
    physical_authority: TrackPhysicalAuthority | None,
    binding_status: str,
) -> None:
    live = _track_physical_array_nodes(subgroup)
    if expected_records is not None and set(live) != set(expected_records):
        raise ValueError(
            f"/{subgroup.path} physical array inventory changed after staging."
        )
    if physical_authority is None:
        if live:
            raise ValueError(
                f"/{subgroup.path} carries mm arrays without exact physical authority."
            )
        for relative_path, node in _iter_track_array_nodes(subgroup):
            descriptor = getattr(node, "attrs", {}).get(
                COORDINATE_DESCRIPTOR_ATTR
            )
            if isinstance(descriptor, Mapping) and str(
                descriptor.get("profile_id", "")
            ).startswith("physical_mm."):
                raise ValueError(
                    f"/{subgroup.path}/{relative_path} carries a stale physical "
                    "coordinate descriptor while physical outputs are omitted."
                )
        return
    if "positions_mm" not in live:
        raise ValueError(
            f"/{subgroup.path} physical publication lacks positions_mm."
        )
    scale_value = physical_authority.mm_per_pixel
    for relative_path, node in live.items():
        if expected_records is not None:
            expected_record = expected_records.get(relative_path)
            live_record = _stage_array_payload_record(
                node,
                relative_ref=(
                    f"tracks/{subgroup.path.rsplit('/', 1)[-1]}/{relative_path}"
                ),
            )
            if live_record != expected_record:
                raise ValueError(
                    f"/{subgroup.path}/{relative_path} changed after physical staging."
                )
        pixel_path = _physical_to_pixel_array_path(relative_path)
        try:
            pixel_node = _relative_child(subgroup, pixel_path)
        except Exception as exc:
            raise ValueError(
                f"/{subgroup.path}/{relative_path} lacks exact pixel pair "
                f"{pixel_path!r}."
            ) from exc
        physical_values = np.array(node[:], copy=True, order="C")
        pixel_values = np.array(pixel_node[:], copy=True, order="C")
        if (
            physical_values.dtype != pixel_values.dtype
            or physical_values.shape != pixel_values.shape
        ):
            raise ValueError(
                f"/{subgroup.path}/{relative_path} dtype/shape differs from "
                "its exact pixel pair."
            )
        scale = np.asarray(scale_value, dtype=pixel_values.dtype)
        expected_values = np.asarray(
            pixel_values * scale,
            dtype=pixel_values.dtype,
        )
        if not np.array_equal(
            physical_values,
            expected_values,
            equal_nan=True,
        ):
            raise ValueError(
                f"/{subgroup.path}/{relative_path} does not use the exact staged "
                "physical mm_per_pixel authority."
            )
        has_descriptor = COORDINATE_DESCRIPTOR_ATTR in getattr(node, "attrs", {})
        if relative_path == "positions_mm":
            expected_bound = binding_status == TRACK_KINEMATICS_BOUND_CANONICAL_STATUS
            if has_descriptor != expected_bound:
                raise ValueError(
                    "positions_mm descriptor must be absent while staged and "
                    "present only after authoritative final-path binding."
                )
        elif has_descriptor:
            raise ValueError(
                f"Non-position physical array {relative_path!r} carries an "
                "unsupported coordinate descriptor."
            )


def _stage_track_groups(
    run_group: zarr.Group,
    manifest: Mapping[str, Any],
) -> list[tuple[int, zarr.Group]]:
    tracks_manifest = manifest.get("tracks")
    track_count = manifest.get("track_count")
    if type(tracks_manifest) is not dict or type(track_count) is not int:
        raise ValueError("Track staging manifest has invalid track inventory.")
    try:
        tracks_group = run_group["tracks"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Track staging run has no tracks group.") from exc
    if not callable(getattr(tracks_group, "group_keys", None)):
        raise ValueError("Track staging run has no tracks group.")
    live_names = sorted(str(name) for name in tracks_group.group_keys())
    if live_names != sorted(tracks_manifest) or len(live_names) != track_count:
        raise ValueError("Track staging manifest and live track groups disagree.")
    track_ids = np.array(run_group["track_ids"][:], copy=True, order="C")
    if (
        track_ids.dtype != np.dtype("<i4")
        or track_ids.shape != (track_count,)
        or _stage_array_payload_record(
            run_group["track_ids"],
            relative_ref="track_ids",
        )
        != manifest.get("track_ids")
    ):
        raise ValueError("Track staging track_ids payload is invalid.")
    result: list[tuple[int, zarr.Group]] = []
    for index, track_id_value in enumerate(track_ids):
        track_id = int(track_id_value)
        name = f"id_{track_id}"
        if index and track_id <= int(track_ids[index - 1]):
            raise ValueError("Track staging track_ids must be strictly increasing.")
        if name not in tracks_manifest:
            raise ValueError("Track staging track_ids do not name exact track groups.")
        result.append((track_id, tracks_group[name]))
    return result


def _validate_unbound_track_payloads(
    authoritative_root: zarr.Group,
    run_group: zarr.Group,
    *,
    expected_keypoint_run: str,
    expected_run_name: str,
    expected_binding_status: str,
    require_complete: bool,
    expected_selector_eligible: bool,
) -> tuple[
    dict[str, Any],
    str,
    BoundSourceCameraPositionSurface,
    BoundStimulusPhysicalCoordinateAuthority | None,
    list[tuple[int, zarr.Group]],
    int,
]:
    if str(run_group.path) != (
        f"analysis/track_kinematics_runs/offline/{expected_run_name}"
    ):
        raise ValueError("Track staging run is not at its exact declared path.")
    if run_group.attrs.get(TRACK_KINEMATICS_COORDINATE_BINDING_STATUS_ATTR) != (
        expected_binding_status
    ):
        raise ValueError("Track run has the wrong coordinate binding status.")
    expected_completion = "complete" if require_complete else "running"
    if run_group.attrs.get("palette_run_completion_status") != expected_completion:
        raise ValueError("Track run has the wrong generic completion status.")
    if (
        run_group.attrs.get("stage_selector_eligible")
        is not expected_selector_eligible
    ):
        raise ValueError(
            "Track run has the wrong literal stage-selector eligibility state."
        )
    manifest, manifest_digest = _load_track_staging_manifest(
        run_group,
        expected_keypoint_run=expected_keypoint_run,
        expected_run_name=expected_run_name,
    )
    surface = _require_stage_source_surface(authoritative_root, manifest)
    physical_authority = _require_stage_physical_authority(
        authoritative_root,
        manifest,
    )
    expected_physical_record = manifest.get("physical_authority")
    expected_reason = manifest.get("physical_omission_reason_code")
    expected_physical_status = (
        "available_typed_source_camera_frame"
        if physical_authority is not None
        else "omitted_no_compatible_typed_physical_frame"
    )
    if (
        run_group.attrs.get("physical_coordinate_authority")
        != expected_physical_record
        or run_group.attrs.get("physical_outputs_status")
        != expected_physical_status
        or run_group.attrs.get("physical_outputs_reason_code")
        != expected_reason
    ):
        raise ValueError(
            "Track run physical status/reason/authority differs from its sealed "
            "staging manifest."
        )
    source_values = np.array(
        surface.coordinates.coordinate_node[:],
        copy=True,
        order="C",
    )
    groups = _stage_track_groups(run_group, manifest)
    all_source_rows: list[np.ndarray] = []
    total_rows = 0
    for track_id, subgroup in groups:
        name = f"id_{track_id}"
        record = manifest["tracks"].get(name)
        if type(record) is not dict or record.get("track_id") != track_id:
            raise ValueError(f"{name} staging manifest identity is invalid.")
        row_count = record.get("row_count")
        if (
            type(row_count) is not int
            or row_count < 1
            or subgroup.attrs.get("num_samples") != row_count
        ):
            raise ValueError(f"{name} staging row count is invalid.")
        if (
            subgroup.attrs.get("physical_outputs_status")
            != expected_physical_status
            or subgroup.attrs.get("physical_outputs_reason_code")
            != expected_reason
            or subgroup.attrs.get("physical_coordinate_authority")
            != expected_physical_record
        ):
            raise ValueError(
                f"{name} physical status/reason differs from its staging manifest."
            )
        summary = subgroup.attrs.get("summary")
        if not isinstance(summary, Mapping):
            raise ValueError(f"{name} summary attrs are invalid.")
        if physical_authority is None and any(
            "_mm" in str(field_name) for field_name in summary
        ):
            raise ValueError(
                f"{name} omitted physical output retains mm-derived summary fields."
            )
        arrays_record = record.get("arrays")
        if type(arrays_record) is not dict or set(arrays_record) != set(
            _TRACK_STAGING_CRITICAL_ARRAYS
        ):
            raise ValueError(f"{name} staging critical-array inventory is invalid.")
        physical_arrays_record = record.get("physical_arrays")
        if type(physical_arrays_record) is not dict:
            raise ValueError(f"{name} staging physical-array inventory is invalid.")
        nodes = {array_name: subgroup[array_name] for array_name in arrays_record}
        for array_name, node in nodes.items():
            live_record = _stage_array_payload_record(
                node,
                relative_ref=f"tracks/{name}/{array_name}",
            )
            if live_record != arrays_record[array_name]:
                raise ValueError(
                    f"{name}/{array_name} changed after numerical staging."
                )
        frame_indices = np.array(nodes["frame_indices"][:], copy=True, order="C")
        track_key = np.array(nodes["track_sample_key"][:], copy=True, order="C")
        source_frames = np.array(
            nodes["source_acquisition_frame_index"][:], copy=True, order="C"
        )
        interpolation = np.array(
            nodes["source_frame_interpolation"][:], copy=True, order="C"
        )
        source_instances = np.array(
            nodes["source_instance_key"][:], copy=True, order="C"
        )
        source_rows = np.array(
            nodes["source_row_index"][:], copy=True, order="C"
        )
        positions = np.array(nodes["positions_px"][:], copy=True, order="C")
        if (
            frame_indices.dtype != np.dtype("<i8")
            or frame_indices.shape != (row_count,)
            or track_key.dtype != np.dtype("<i8")
            or track_key.shape != (row_count, 2)
            or source_frames.dtype != np.dtype("<i8")
            or source_frames.shape != (row_count,)
            or interpolation.dtype != TRACK_SAMPLE_INTERPOLATION_DTYPE
            or interpolation.shape != (row_count,)
            or source_instances.dtype != TRACK_SAMPLE_SOURCE_INSTANCE_KEY_DTYPE
            or source_instances.shape != (row_count,)
            or source_rows.dtype != np.dtype("<i8")
            or source_rows.shape != (row_count,)
            or positions.dtype != source_values.dtype
            or positions.shape != (row_count, 2)
            or np.any(source_rows < 0)
            or np.any(source_rows >= source_values.shape[0])
            or not np.array_equal(frame_indices, source_frames)
            or not np.array_equal(track_key[:, 1], source_frames)
            or np.any(track_key[:, 0] != track_id)
            or not np.array_equal(
                source_frames,
                resolve_source_acquisition_frame_indices(
                    surface.temporal_authority,
                    source_rows,
                ),
            )
            or not np.array_equal(
                source_instances,
                derive_track_source_instance_values(
                    surface.temporal_authority,
                    source_rows,
                ),
            )
            or not np.array_equal(
                positions,
                source_values[source_rows],
                equal_nan=True,
            )
        ):
            raise ValueError(
                f"{name} is not an exact identity/time/position subset of its "
                "authoritative source."
            )
        _validate_track_physical_arrays(
            subgroup,
            expected_records=physical_arrays_record,
            physical_authority=physical_authority,
            binding_status=expected_binding_status,
        )
        all_source_rows.append(source_rows)
        total_rows += row_count
    if all_source_rows:
        combined = np.concatenate(all_source_rows)
        if np.unique(combined).shape[0] != combined.shape[0]:
            raise ValueError(
                "Staged track source_row_index values repeat across track groups."
            )
    expected_run_record = manifest.get("run_physical_surfaces")
    if type(expected_run_record) is not dict:
        raise ValueError(
            "Track staging manifest lacks sealed run physical surfaces."
        )
    _validate_run_track_physical_surfaces(
        run_group,
        groups=groups,
        physical_authority=physical_authority,
        physical_omission_reason_code=str(expected_reason),
        binding_status=expected_binding_status,
        expected_track_records=manifest["tracks"],
        expected_run_record=expected_run_record,
    )
    return (
        manifest,
        manifest_digest,
        surface,
        physical_authority,
        groups,
        total_rows,
    )


def _load_bound_track_publication(
    subgroup: zarr.Group,
    *,
    surface: BoundSourceCameraPositionSurface,
    physical_authority: TrackPhysicalAuthority | None,
) -> TrackPositionPublicationResult:
    return _load_bound_track_publication_from_source(
        subgroup,
        source_positions=surface.coordinates,
        source_temporal_authority=surface.temporal_authority,
        physical_authority=physical_authority,
    )


def _load_bound_track_publication_from_source(
    subgroup: zarr.Group,
    *,
    source_positions: BoundCanonicalCoordinateDescriptor,
    source_temporal_authority: BoundSourceRowTemporalAuthority,
    physical_authority: TrackPhysicalAuthority | None,
) -> TrackPositionPublicationResult:
    track_sample_key_node = subgroup["track_sample_key"]
    source_row_index_node = subgroup["source_row_index"]
    source_acquisition_frame_node = subgroup[
        "source_acquisition_frame_index"
    ]
    source_frame_interpolation_node = subgroup["source_frame_interpolation"]
    source_instance_key_node = subgroup["source_instance_key"]
    time_lineage = load_bound_track_sample_time_lineage(
        subgroup,
        track_sample_key_node,
        source_row_index_node,
        source_acquisition_frame_node,
        source_frame_interpolation_node,
        source_instance_key_node,
        source_temporal_authority=source_temporal_authority,
    )
    identity = load_bound_row_identity_contract(
        subgroup,
        track_sample_key_node,
        track_time_lineage=time_lineage,
    )
    return load_track_position_coordinates(
        subgroup,
        subgroup["positions_px"],
        source_row_index_node,
        track_row_identity=identity,
        source_positions=source_positions,
        source_temporal_authority=source_temporal_authority,
        positions_mm_node=(
            subgroup["positions_mm"]
            if physical_authority is not None
            else None
        ),
        physical_frame=(
            physical_authority.physical_frame
            if physical_authority is not None
            else None
        ),
    )


def _fresh_track_run_group(
    authoritative_root: Any,
    run_group: Any,
) -> Any:
    """Resolve a caller-supplied track path from the authoritative root."""

    path = str(getattr(run_group, "path", "")).strip("/")
    parts = path.split("/")
    if (
        len(parts) != 4
        or parts[:2] != ["analysis", "track_kinematics_runs"]
        or parts[2] not in {"online", "offline"}
        or not parts[3]
    ):
        raise ValueError("Track run is not at one canonical typed run path.")
    try:
        fresh = authoritative_root
        for part in parts:
            fresh = fresh[part]
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(
            f"Authoritative track run /{path} is unavailable."
        ) from exc
    if (
        str(getattr(fresh, "path", "")).strip("/") != path
        or not callable(getattr(fresh, "group_keys", None))
        or not callable(getattr(fresh, "array_keys", None))
    ):
        raise ValueError(
            f"Authoritative track run /{path} resolved to a different node."
        )
    return fresh


def _live_track_groups(run_group: zarr.Group) -> list[tuple[int, zarr.Group]]:
    try:
        track_ids_node = run_group["track_ids"]
        tracks_group = run_group["tracks"]
    except (KeyError, TypeError):
        raise ValueError("Track run lacks its exact track_ids/tracks inventory.")
    track_ids = np.array(track_ids_node[:], copy=True, order="C")
    if track_ids.dtype != np.dtype("<i4") or track_ids.ndim != 1:
        raise ValueError("Track run track_ids must be one exact signed int32 vector.")
    ids = [int(value) for value in track_ids]
    if ids != sorted(set(ids)):
        raise ValueError("Track run track_ids must be strictly increasing and unique.")
    expected_names = [f"id_{track_id}" for track_id in ids]
    group_keys = getattr(tracks_group, "group_keys", None)
    array_keys = getattr(tracks_group, "array_keys", None)
    if not callable(group_keys) or not callable(array_keys):
        raise ValueError("Track run tracks node is not a persisted group.")
    unexpected_arrays = sorted(str(name) for name in array_keys())
    if unexpected_arrays:
        raise ValueError(
            "Track run /tracks array inventory is not closed "
            f"(unexpected={unexpected_arrays!r})."
        )
    live_names = sorted(str(name) for name in group_keys())
    if live_names != sorted(expected_names):
        raise ValueError("Track run track_ids and track subgroup inventory disagree.")
    return [(track_id, tracks_group[f"id_{track_id}"]) for track_id in ids]


def _resolve_track_source_authority(
    authoritative_root: zarr.Group,
    groups: list[tuple[int, zarr.Group]],
) -> tuple[
    BoundCanonicalCoordinateDescriptor,
    BoundSourceRowTemporalAuthority,
]:
    if not groups:
        raise ValueError("A public track run must contain at least one track.")
    first_derivation = bind_persisted_coordinate_record(
        groups[0][1],
        attr_name=TRACK_POSITION_DERIVATION_ATTR,
    ).record
    source_record = first_derivation.get("source_coordinate")
    temporal_record = first_derivation.get("source_temporal_authority")
    if not isinstance(source_record, Mapping) or not isinstance(
        temporal_record,
        Mapping,
    ):
        raise ValueError("Track derivation lacks sealed source coordinate/time refs.")
    source_ref = source_record.get("array_ref")
    if (
        not isinstance(source_ref, str)
        or not source_ref.startswith("/")
        or any(part in {"", ".", ".."} for part in source_ref[1:].split("/"))
    ):
        raise ValueError("Track derivation source coordinate ref is noncanonical.")
    parts = source_ref[1:].split("/")

    source: BoundCanonicalCoordinateDescriptor
    temporal: BoundSourceRowTemporalAuthority
    if (
        len(parts) == 6
        and parts[:2] == ["analysis", "stimulus_runs"]
        and parts[3:] == [
            "tracking_data",
            "chaser_states",
            "target_position_xy",
        ]
    ):
        stimulus_group = authoritative_root[
            f"analysis/stimulus_runs/{parts[2]}"
        ]
        chaser_group = stimulus_group["tracking_data/chaser_states"]
        _, _, handoff = load_canonical_online_coordinate_surface(
            authoritative_root,
            stimulus_group,
            chaser_group,
        )
        source = require_bound_canonical_coordinate_descriptor(
            handoff.coordinate_descriptor
        )
        temporal = require_bound_source_row_temporal_authority(
            handoff.source_temporal_authority
        )
    elif (
        len(parts) == 4
        and parts[0] == "refined_online_runs"
        and parts[2:] == ["interpolated", "positions_px"]
    ):
        from fisheye.refinement.refine_online_detect import (
            load_bound_refined_online_coordinate_evidence,
        )

        refined_group = authoritative_root[
            f"refined_online_runs/{parts[1]}"
        ]
        refined = load_bound_refined_online_coordinate_evidence(
            authoritative_root,
            refined_group,
        )
        refined.assert_verified()
        source = require_bound_canonical_coordinate_descriptor(
            refined.descriptor_for("interpolated")
        )
        temporal = require_bound_source_row_temporal_authority(
            refined.source_temporal_authority
        )
    elif parts[-1] == "centers_img_xy":
        rowset_ref = "/".join(parts[:-1])
        surface = require_bound_source_camera_position_surface(
            load_persisted_source_camera_position_surface(
                authoritative_root,
                rowset_ref,
            )
        )
        source = require_bound_canonical_coordinate_descriptor(
            surface.coordinates
        )
        temporal = require_bound_source_row_temporal_authority(
            surface.temporal_authority
        )
    else:
        raise ValueError(
            "Track derivation names an unsupported sealed source-coordinate "
            f"resolver: {source_ref!r}."
        )

    if f"/{source.coordinate_node.path}" != source_ref:
        raise ValueError("Resolved source coordinate path differs from its derivation.")
    source_values = np.array(source.coordinate_node[:], copy=True, order="C")
    if (
        source_record.get("dtype") != source_values.dtype.str
        or source_record.get("shape")
        != [int(value) for value in source_values.shape]
        or source_record.get("content_sha256")
        != array_payload_sha256(source.coordinate_node)
        or source_record.get("descriptor_sha256") != source.descriptor.digest()
        or source_record.get("row_identity_ref")
        != source.row_identity.record_ref
        or source_record.get("row_identity_sha256")
        != source.row_identity.record_sha256
        or temporal_record.get("record_ref") != temporal.record_ref
        or temporal_record.get("record_sha256") != temporal.record_sha256
    ):
        raise ValueError(
            "Resolved source coordinate, descriptor, identity, time, or payload "
            "differs from the sealed track derivation."
        )
    return source, temporal


def _resolve_track_physical_authority(
    authoritative_root: zarr.Group,
    run_group: zarr.Group,
) -> TrackPhysicalAuthority | None:
    expected = run_group.attrs.get("physical_coordinate_authority")
    if expected is None:
        return None
    if not isinstance(expected, Mapping):
        raise ValueError("Track run physical authority record is not an object.")
    authority_kind = expected.get("authority_kind")
    if authority_kind is None and "stimulus_run" in expected:
        stimulus_run = expected.get("stimulus_run")
        if not isinstance(stimulus_run, str) or not stimulus_run:
            raise ValueError("Track run physical authority lacks one stimulus run.")
        authority = load_stimulus_physical_coordinate_authority(
            authoritative_root,
            stimulus_run=stimulus_run,
        )
        if authority is None:
            raise ValueError("Track run declares unavailable stimulus physical authority.")
        authority = require_bound_stimulus_physical_coordinate_authority(authority)
    elif authority_kind == "recording_calibration":
        authority = require_bound_source_camera_physical_authority(
            load_source_camera_physical_authority(authoritative_root)
        )
    else:
        raise ValueError("Track run physical authority kind is unsupported.")
    if _physical_authority_manifest_record(authority) != dict(expected):
        raise ValueError("Track run physical authority changed after publication.")
    return authority


def _load_bound_track_position_bindings_impl(
    authoritative_root: zarr.Group,
    run_group: zarr.Group,
    *,
    require_complete: bool,
    expected_selector_eligible: bool,
) -> BoundTrackPositionBindings:
    run_group = _fresh_track_run_group(authoritative_root, run_group)
    parts = str(run_group.path).split("/")
    expected_status = RUN_STATUS_COMPLETE if require_complete else "running"
    if (
        run_group.attrs.get(RUN_COMPLETION_STATUS_ATTR) != expected_status
        or run_group.attrs.get("stage_selector_eligible")
        is not expected_selector_eligible
        or run_group.attrs.get(TRACK_KINEMATICS_COORDINATE_BINDING_STATUS_ATTR)
        != TRACK_KINEMATICS_BOUND_CANONICAL_STATUS
    ):
        raise ValueError(
            "Track run completion, selector eligibility, or coordinate-binding "
            "status is not public-contract compatible."
        )
    groups = _live_track_groups(run_group)
    source, temporal = _resolve_track_source_authority(
        authoritative_root,
        groups,
    )
    physical = _resolve_track_physical_authority(
        authoritative_root,
        run_group,
    )
    omission_reason = str(run_group.attrs.get("physical_outputs_reason_code", ""))
    _validate_run_track_physical_surfaces(
        run_group,
        groups=groups,
        physical_authority=physical,
        physical_omission_reason_code=omission_reason,
        binding_status=TRACK_KINEMATICS_BOUND_CANONICAL_STATUS,
    )
    bindings = tuple(
        (
            track_id,
            _load_bound_track_publication_from_source(
                subgroup,
                source_positions=source,
                source_temporal_authority=temporal,
                physical_authority=physical,
            ),
        )
        for track_id, subgroup in groups
    )
    return BoundTrackPositionBindings(
        archive_identity=archive_identity(run_group),
        run_type=parts[2],
        run_name=parts[3],
        source_positions=source,
        source_temporal_authority=temporal,
        physical_authority=physical,
        track_positions=bindings,
        run_group=run_group,
    )


def _load_bound_track_position_bindings_before_selection(
    authoritative_root: zarr.Group,
    run_group: zarr.Group,
    *,
    require_complete: bool,
) -> BoundTrackPositionBindings:
    """Internal loader for a running or complete-but-ineligible bound run."""

    return _load_bound_track_position_bindings_impl(
        authoritative_root,
        run_group,
        require_complete=require_complete,
        expected_selector_eligible=False,
    )


def load_bound_track_position_bindings(
    authoritative_root: zarr.Group,
    run_group: zarr.Group,
) -> BoundTrackPositionBindings:
    """Load typed positions only from a complete, literally eligible track run.

    This coordinate boundary does not validate or authorize derived kinematic
    payload arrays.
    """

    return _load_bound_track_position_bindings_impl(
        authoritative_root,
        run_group,
        require_complete=True,
        expected_selector_eligible=True,
    )


def validate_bound_track_position_bindings(
    authoritative_root: zarr.Group,
    run_group: zarr.Group,
) -> BoundTrackPositionBindings:
    """Validate and return the same typed bindings as the canonical loader."""

    return load_bound_track_position_bindings(authoritative_root, run_group)


def _load_bound_track_motion_run_impl(
    authoritative_root: zarr.Group,
    run_group: zarr.Group,
    *,
    expected_selector_eligible: bool,
) -> BoundTrackMotionRun:
    """Freshly bind the exact sealed motion payload from live arrays."""

    if expected_selector_eligible:
        positions = load_bound_track_position_bindings(
            authoritative_root,
            run_group,
        )
    else:
        positions = _load_bound_track_position_bindings_before_selection(
            authoritative_root,
            run_group,
            require_complete=True,
        )
    # Position binding resolves the child from the authoritative root.  Every
    # subsequent payload and attribute read must use that same fresh node, not
    # the caller's potentially detached or replaced handle.
    run_group = positions.run_group
    raw_manifest = run_group.attrs.get(TRACK_MOTION_PUBLICATION_MANIFEST_ATTR)
    raw_digest = run_group.attrs.get(
        TRACK_MOTION_PUBLICATION_MANIFEST_DIGEST_ATTR
    )
    raw_commit = run_group.attrs.get(TRACK_MOTION_PUBLICATION_COMMIT_ATTR)
    if (
        not isinstance(raw_manifest, Mapping)
        or not isinstance(raw_digest, str)
        or not isinstance(raw_commit, Mapping)
    ):
        raise ValueError(
            "Track run lacks one versioned digest-bound full-motion manifest "
            "and publication commit."
        )
    manifest = _motion_json_object(
        raw_manifest,
        label=f"/{run_group.path} persisted full-motion manifest",
    )
    expected_digest = _canonical_json_sha256(manifest)
    if (
        manifest.get("schema_id") != TRACK_MOTION_PUBLICATION_MANIFEST_SCHEMA_ID
        or manifest.get("schema_version")
        != TRACK_MOTION_PUBLICATION_MANIFEST_SCHEMA_VERSION
        or manifest.get("run_ref") != f"/{run_group.path}"
        or manifest.get("run_type") != positions.run_type
        or manifest.get("run_name") != positions.run_name
        or raw_digest != expected_digest
    ):
        raise ValueError(
            "Track full-motion manifest schema, run identity, or digest is invalid."
        )
    commit = _motion_json_object(
        raw_commit,
        label=f"/{run_group.path} persisted full-motion publication commit",
    )
    if commit != _track_motion_publication_commit(manifest):
        raise ValueError(
            "Track full-motion publication commit does not bind the exact manifest."
        )
    live_manifest = _build_track_motion_publication_manifest(
        authoritative_root,
        run_group,
        positions,
    )
    if not _track_attr_values_equal(manifest, live_manifest):
        raise ValueError(
            "Track full-motion manifest differs from the exact live payload, "
            "domains, aliases, derivations, or authorities."
        )

    position_by_id = dict(positions.track_positions)
    track_bindings: list[BoundTrackMotionTrack] = []
    tracks_record = manifest.get("tracks")
    if not isinstance(tracks_record, Mapping):  # pragma: no cover - schema guard
        raise ValueError("Track full-motion manifest track inventory is invalid.")
    for track_id, track_group in _live_track_groups(run_group):
        record = tracks_record.get(f"id_{track_id}")
        if not isinstance(record, Mapping) or not isinstance(
            record.get("surfaces"), Mapping
        ):
            raise ValueError(
                f"Track {track_id} full-motion manifest record is invalid."
            )
        surfaces: list[BoundTrackMotionSurface] = []
        for relative_path, surface_record in sorted(
            record["surfaces"].items()
        ):
            if not isinstance(surface_record, Mapping):
                raise ValueError(
                    f"Track {track_id} motion surface record is invalid."
                )
            if surface_record.get("authority_scope") != "public_derived_motion":
                continue
            input_refs = surface_record.get("input_refs")
            if not isinstance(input_refs, list) or not all(
                isinstance(value, Mapping)
                and value.get("kind")
                in {
                    "array",
                    "group_attr",
                    "manifest_record",
                    "external_lineage",
                }
                and isinstance(value.get("ref"), str)
                for value in input_refs
            ):
                raise ValueError(
                    f"Track {track_id} motion derivation input refs are invalid."
                )
            surface_shape = surface_record.get("shape")
            surface_dtype = surface_record.get("dtype")
            surface_digest = surface_record.get("content_sha256")
            if (
                not isinstance(surface_shape, list)
                or not all(
                    isinstance(value, int) and not isinstance(value, bool)
                    and value >= 0
                    for value in surface_shape
                )
                or not isinstance(surface_dtype, str)
                or not isinstance(surface_digest, str)
            ):
                raise ValueError(
                    f"Track {track_id} motion surface payload binding is invalid."
                )
            _sha256_text(
                surface_digest,
                label=f"track {track_id} {relative_path} content_sha256",
            )
            surfaces.append(
                BoundTrackMotionSurface(
                    relative_path=str(relative_path),
                    axis0_domain=str(surface_record["axis0_domain"]),
                    units=str(surface_record["units"]),
                    semantic_profile=str(surface_record["semantic_profile"]),
                    operation_id=str(surface_record["operation_id"]),
                    input_refs=tuple(dict(value) for value in input_refs),
                    alias_of=(
                        str(surface_record["alias_of"])
                        if surface_record.get("alias_of") is not None
                        else None
                    ),
                    dtype=surface_dtype,
                    shape=tuple(surface_shape),
                    content_sha256=surface_digest,
                    node=_relative_child(track_group, str(relative_path)),
                    _verification_seal=_BOUND_TRACK_MOTION_SEAL,
                )
            )
        track_bindings.append(
            BoundTrackMotionTrack(
                track_id=track_id,
                position_binding=position_by_id[track_id],
                surfaces=tuple(surfaces),
                track_group=track_group,
                _verification_seal=_BOUND_TRACK_MOTION_SEAL,
            )
        )
    return BoundTrackMotionRun(
        position_bindings=positions,
        manifest_sha256=expected_digest,
        manifest=manifest,
        tracks=tuple(track_bindings),
        run_group=run_group,
        authoritative_root=authoritative_root,
        expected_selector_eligible=expected_selector_eligible,
        _verification_seal=_BOUND_TRACK_MOTION_SEAL,
    )


def _seal_and_load_track_motion_run_before_selection(
    authoritative_root: zarr.Group,
    run_group: zarr.Group,
    *,
    expected_publication_owner_uuid: str | None = None,
) -> BoundTrackMotionRun:
    """Persist then freshly reload a complete, selector-ineligible motion seal."""

    run_group = _fresh_track_run_group(authoritative_root, run_group)
    owner_uuid = (
        _track_publication_owner_uuid(run_group)
        if expected_publication_owner_uuid is None
        else str(expected_publication_owner_uuid)
    )
    if _track_publication_owner_uuid(run_group) != owner_uuid:
        raise RuntimeError(
            "Full-motion sealing observed a different publication owner."
        )
    if (
        run_group.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE
        or run_group.attrs.get("stage_selector_eligible") is not False
    ):
        raise ValueError(
            "Full-motion sealing requires a complete, literally selector-ineligible run."
        )
    positions = _load_bound_track_position_bindings_before_selection(
        authoritative_root,
        run_group,
        require_complete=True,
    )
    manifest = _build_track_motion_publication_manifest(
        authoritative_root,
        run_group,
        positions,
    )
    run_group.attrs[TRACK_MOTION_PUBLICATION_MANIFEST_ATTR] = manifest
    run_group = _resolve_owned_track_run_child(
        authoritative_root,
        run_name=positions.run_name,
        run_type=positions.run_type,
        owner_uuid=owner_uuid,
    )
    assert run_group is not None
    run_group.attrs[TRACK_MOTION_PUBLICATION_MANIFEST_DIGEST_ATTR] = (
        _canonical_json_sha256(manifest)
    )
    run_group = _resolve_owned_track_run_child(
        authoritative_root,
        run_name=positions.run_name,
        run_type=positions.run_type,
        owner_uuid=owner_uuid,
    )
    assert run_group is not None
    run_group.attrs[TRACK_MOTION_PUBLICATION_COMMIT_ATTR] = (
        _track_motion_publication_commit(manifest)
    )
    return _load_bound_track_motion_run_impl(
        authoritative_root,
        run_group,
        expected_selector_eligible=False,
    )


def load_bound_track_motion_run(
    authoritative_root: zarr.Group,
    run_group: zarr.Group,
) -> BoundTrackMotionRun:
    """Load a complete public run with an exact live full-motion seal."""

    return _load_bound_track_motion_run_impl(
        authoritative_root,
        run_group,
        expected_selector_eligible=True,
    )


def validate_bound_track_motion_run(
    authoritative_root: zarr.Group,
    run_group: zarr.Group,
) -> BoundTrackMotionRun:
    """Validate and return the same typed full-motion authority."""

    return load_bound_track_motion_run(authoritative_root, run_group)


def _validate_direct_track_kinematics_run_before_selection(
    authoritative_root: zarr.Group,
    run_group: zarr.Group,
    *,
    run_name: str,
    run_type: str,
    source_positions: BoundCanonicalCoordinateDescriptor,
    source_temporal_authority: Any,
    physical_authority: TrackPhysicalAuthority | None,
) -> Mapping[str, Any]:
    """Freshly validate one direct writer output while it remains ineligible."""

    bound = _load_bound_track_position_bindings_before_selection(
        authoritative_root,
        run_group,
        require_complete=True,
    )
    expected_source = require_bound_canonical_coordinate_descriptor(
        source_positions
    )
    expected_temporal = require_bound_source_row_temporal_authority(
        source_temporal_authority
    )
    expected_physical = (
        require_bound_stimulus_physical_coordinate_authority(physical_authority)
        if physical_authority is not None
        else None
    )
    if (
        bound.run_name != run_name
        or bound.run_type != run_type
        or bound.source_positions.coordinate_node.path
        != expected_source.coordinate_node.path
        or bound.source_positions.descriptor.digest()
        != expected_source.descriptor.digest()
        or bound.source_temporal_authority.record_ref
        != expected_temporal.record_ref
        or bound.source_temporal_authority.record_sha256
        != expected_temporal.record_sha256
        or _physical_authority_manifest_record(bound.physical_authority)
        != _physical_authority_manifest_record(expected_physical)
    ):
        raise ValueError(
            "Freshly resolved track authority differs from direct-writer evidence."
        )
    return {
        "valid": True,
        "status": TRACK_KINEMATICS_BOUND_CANONICAL_STATUS,
        "run_name": str(run_name),
        "run_type": str(run_type),
        "track_count": len(bound.track_positions),
        "row_count": sum(
            int(run_group[f"tracks/id_{track_id}"].attrs["num_samples"])
            for track_id, _ in bound.track_positions
        ),
    }


def stage_offline_track_kinematics_run(
    source_zarr: str | Path,
    staging_zarr: str | Path,
    *,
    keypoint_run: str,
    run_name: str,
    writer_arguments: Iterable[str] = (),
) -> Mapping[str, Any]:
    """Compute one fail-closed numeric stage with no coordinate publication."""

    source_path = Path(source_zarr).expanduser().resolve()
    staging_path = Path(staging_zarr).expanduser().resolve()
    if not source_path.is_dir():
        raise FileNotFoundError(f"Source analysis Zarr not found: {source_path}")
    if staging_path.exists():
        raise FileExistsError(f"Refusing existing staging Zarr: {staging_path}")
    if source_path == staging_path:
        raise ValueError("Track staging output must differ from the source archive.")
    try:
        staging_path.relative_to(source_path)
    except ValueError:
        pass
    else:
        raise ValueError("Track staging output must not be inside the source archive.")
    run_value = str(run_name).strip()
    keypoint_value = str(keypoint_run).strip()
    if (
        not run_value
        or run_value in {".", ".."}
        or "/" in run_value
        or "\\" in run_value
        or not keypoint_value
    ):
        raise ValueError("Track staging requires safe nonempty run/keypoint names.")
    forwarded = tuple(str(item) for item in writer_arguments)
    managed = {
        "--output-zarr-path",
        "--offline-only",
        "--online-only",
        "--no-write",
        "--keypoint-run",
        "--offline-run-name",
        "--_unbound-coordinate-stage",
    }
    forbidden = sorted(
        item.split("=", 1)[0]
        for item in forwarded
        if item.split("=", 1)[0] in managed
    )
    if forbidden:
        raise ValueError(
            "Track staging owns these writer arguments: " + ", ".join(forbidden)
        )
    stimulus_selector: str | None = None
    for index, item in enumerate(forwarded):
        if item.startswith("--stimulus-run="):
            stimulus_selector = item.split("=", 1)[1]
        elif item == "--stimulus-run":
            if index + 1 >= len(forwarded) or forwarded[index + 1].startswith("--"):
                raise ValueError("--stimulus-run requires one nonempty run name.")
            stimulus_selector = forwarded[index + 1]
    source_root = open_zarr_root(source_path, mode="r")
    stage_physical_authority, stage_physical_info = resolve_track_physical_authority(
        source_root,
        stimulus_run=stimulus_selector,
    )
    if stage_physical_authority is None:
        raise ValueError(
            "Offline track-kinematics publication requires sealed source-camera "
            "physical calibration authority; none was available "
            f"(reason_code={stage_physical_info.get('reason_code')!r})."
        )
    main(
        [
            str(source_path),
            "--output-zarr-path",
            str(staging_path),
            "--offline-only",
            "--keypoint-run",
            keypoint_value,
            "--offline-run-name",
            run_value,
            "--_unbound-coordinate-stage",
            *forwarded,
        ]
    )
    authoritative_root = open_zarr_root(source_path, mode="r")
    staging_root = open_zarr_root(staging_path, mode="r")
    run_group = staging_root[
        f"analysis/track_kinematics_runs/offline/{run_value}"
    ]
    _, digest, _, _, groups, total_rows = _validate_unbound_track_payloads(
        authoritative_root,
        run_group,
        expected_keypoint_run=keypoint_value,
        expected_run_name=run_value,
        expected_binding_status=TRACK_KINEMATICS_UNBOUND_STAGE_STATUS,
        require_complete=True,
        expected_selector_eligible=False,
    )
    return {
        "valid": True,
        "status": TRACK_KINEMATICS_UNBOUND_STAGE_STATUS,
        "run_name": run_value,
        "track_count": len(groups),
        "row_count": total_rows,
        "staging_manifest_sha256": digest,
    }


def bind_staged_offline_track_kinematics_run(
    authoritative_root: zarr.Group,
    final_run_group: zarr.Group,
    *,
    expected_keypoint_run: str,
    expected_run_name: str,
) -> Mapping[str, Any]:
    """Bind a validated stage only at its authoritative final archive path."""

    if archive_identity(authoritative_root) != archive_identity(final_run_group):
        raise ValueError(
            "Final track run is not inside the supplied authoritative archive."
        )
    (
        _,
        digest,
        surface,
        physical_authority,
        groups,
        total_rows,
    ) = _validate_unbound_track_payloads(
        authoritative_root,
        final_run_group,
        expected_keypoint_run=expected_keypoint_run,
        expected_run_name=expected_run_name,
        expected_binding_status=TRACK_KINEMATICS_PUBLISHING_BINDING_STATUS,
        require_complete=False,
        expected_selector_eligible=False,
    )
    attrs_targets: list[Any] = [final_run_group.attrs]
    for _, subgroup in groups:
        attrs_targets.extend(
            (
                subgroup.attrs,
                subgroup["track_sample_key"].attrs,
                subgroup["positions_px"].attrs,
            )
        )
        if physical_authority is not None:
            attrs_targets.append(subgroup["positions_mm"].attrs)
    snapshots = [copy.deepcopy(dict(attrs)) for attrs in attrs_targets]
    try:
        for track_id, subgroup in groups:
            track_sample_key_node = subgroup["track_sample_key"]
            source_row_index_node = subgroup["source_row_index"]
            source_acquisition_frame_node = subgroup[
                "source_acquisition_frame_index"
            ]
            source_frame_interpolation_node = subgroup[
                "source_frame_interpolation"
            ]
            source_instance_key_node = subgroup["source_instance_key"]
            time_lineage = stamp_track_sample_time_lineage(
                subgroup,
                track_sample_key_node,
                source_row_index_node,
                source_acquisition_frame_node,
                source_frame_interpolation_node,
                source_instance_key_node,
                source_temporal_authority=surface.temporal_authority,
            )
            key_values = np.array(
                track_sample_key_node[:],
                copy=True,
                order="C",
            )
            identity_contract = build_row_identity_contract(
                domain=TRACK_SAMPLE_DOMAIN,
                values=key_values,
                track_time_lineage=time_lineage,
            )
            identity = stamp_and_bind_row_identity_contract(
                subgroup,
                track_sample_key_node,
                contract=identity_contract,
                track_time_lineage=time_lineage,
            )
            if np.any(key_values[:, 0] != track_id):
                raise ValueError("Track identity changed during final-path binding.")
            publish_track_position_coordinates(
                subgroup,
                subgroup["positions_px"],
                subgroup["source_row_index"],
                track_row_identity=identity,
                source_positions=surface.coordinates,
                source_temporal_authority=surface.temporal_authority,
                positions_mm_node=(
                    subgroup["positions_mm"]
                    if physical_authority is not None
                    else None
                ),
                physical_frame=(
                    physical_authority.physical_frame
                    if physical_authority is not None
                    else None
                ),
            )
        for _, subgroup in groups:
            _load_bound_track_publication(
                subgroup,
                surface=surface,
                physical_authority=physical_authority,
            )
        final_run_group.attrs[TRACK_KINEMATICS_COORDINATE_BINDING_STATUS_ATTR] = (
            TRACK_KINEMATICS_BOUND_CANONICAL_STATUS
        )
        validated = _validate_bound_offline_track_kinematics_run_or_raise(
            authoritative_root,
            final_run_group,
            expected_keypoint_run=expected_keypoint_run,
            expected_run_name=expected_run_name,
            require_complete=False,
            expected_selector_eligible=False,
        )
        return {
            **validated,
            "status": TRACK_KINEMATICS_BOUND_CANONICAL_STATUS,
            "track_count": len(groups),
            "row_count": total_rows,
            "binding_manifest_sha256": digest,
        }
    except BaseException as exc:
        rollback_errors: list[str] = []
        for attrs, snapshot in zip(attrs_targets, snapshots, strict=True):
            try:
                _restore_track_attrs(attrs, snapshot)
            except BaseException as rollback_exc:  # pragma: no cover - hostile store
                rollback_errors.append(str(rollback_exc))
        if rollback_errors:
            raise RuntimeError(
                "Final-path track binding failed and attrs rollback was incomplete: "
                f"{rollback_errors!r}."
            ) from exc
        raise


def _validate_bound_offline_track_kinematics_run_or_raise(
    authoritative_root: zarr.Group,
    final_run_group: zarr.Group,
    *,
    expected_keypoint_run: str,
    expected_run_name: str,
    require_complete: bool,
    expected_selector_eligible: bool,
) -> dict[str, Any]:
    if archive_identity(authoritative_root) != archive_identity(final_run_group):
        raise ValueError(
            "Bound track run is not inside the supplied authoritative archive."
        )
    (
        _,
        digest,
        surface,
        physical_authority,
        groups,
        total_rows,
    ) = _validate_unbound_track_payloads(
        authoritative_root,
        final_run_group,
        expected_keypoint_run=expected_keypoint_run,
        expected_run_name=expected_run_name,
        expected_binding_status=TRACK_KINEMATICS_BOUND_CANONICAL_STATUS,
        require_complete=require_complete,
        expected_selector_eligible=expected_selector_eligible,
    )
    for _, subgroup in groups:
        _load_bound_track_publication(
            subgroup,
            surface=surface,
            physical_authority=physical_authority,
        )
    return {
        "valid": True,
        "status": TRACK_KINEMATICS_BOUND_CANONICAL_STATUS,
        "run_name": expected_run_name,
        "track_count": len(groups),
        "row_count": total_rows,
        "binding_manifest_sha256": digest,
    }


def _validate_bound_offline_track_kinematics_run_before_selection(
    authoritative_root: zarr.Group,
    final_run_group: zarr.Group,
    *,
    expected_keypoint_run: str,
    expected_run_name: str,
    require_complete: bool,
) -> Mapping[str, Any]:
    """Internal validator for a running or complete ineligible bound run."""

    try:
        if final_run_group.attrs.get("stage_selector_eligible") is not False:
            raise ValueError(
                "Pre-selection track validation requires literal "
                "stage_selector_eligible=false."
            )
        return _validate_bound_offline_track_kinematics_run_or_raise(
            authoritative_root,
            final_run_group,
            expected_keypoint_run=expected_keypoint_run,
            expected_run_name=expected_run_name,
            require_complete=bool(require_complete),
            expected_selector_eligible=False,
        )
    except Exception as exc:
        return {
            "valid": False,
            "status": "invalid",
            "run_name": str(expected_run_name),
            "errors": [str(exc)],
        }


def validate_bound_offline_track_kinematics_run(
    authoritative_root: zarr.Group,
    final_run_group: zarr.Group,
    *,
    expected_keypoint_run: str,
    expected_run_name: str,
) -> Mapping[str, Any]:
    """Compatibility validator for a completed deferred-staging manifest.

    This does not authorize every derived kinematic payload and is intentionally
    excluded from the module's public export surface.  New consumers should use
    :func:`validate_bound_track_position_bindings` for position coordinates and
    must wait for a separate exact derived-payload contract before treating
    speed/path/heading arrays as canonical.
    """

    try:
        return _validate_bound_offline_track_kinematics_run_or_raise(
            authoritative_root,
            final_run_group,
            expected_keypoint_run=expected_keypoint_run,
            expected_run_name=expected_run_name,
            require_complete=True,
            expected_selector_eligible=True,
        )
    except Exception as exc:
        return {
            "valid": False,
            "status": "invalid",
            "run_name": str(expected_run_name),
            "errors": [str(exc)],
        }


def _write_movement_speed_groups(
    track_group: zarr.Group,
    data: Dict[str, Any],
    *,
    chunks: Tuple[int, ...],
    include_physical: bool,
) -> None:
    """Write the grouped v2 movement/speed layout alongside flat v1 arrays."""

    movement = track_group.create_group("movement")
    movement.attrs.update(
        {
            "schema_id": MOVEMENT_SCHEMA_ID,
            "layout": "movement/speed/<level>",
            "compatibility_flat_arrays": True,
            "compatibility_speed_derivatives": True,
        }
    )
    speed_parent = movement.create_group("speed")
    speed_parent.attrs.update(
        {
            "schema_id": MOVEMENT_SPEED_SCHEMA_ID,
            "levels": list(MOVEMENT_SPEED_LEVEL_NAMES.values()),
            "source_level_names": dict(MOVEMENT_SPEED_LEVEL_NAMES),
            "preferred_read_contract": "movement/speed/<level>",
        }
    )

    path_distance_by_level = {
        "speed_raw": ("frame_path_distance_raw_px", "frame_path_distance_raw_mm"),
        "speed_filtered": ("frame_path_distance_filtered_px", "frame_path_distance_filtered_mm"),
        "speed_smoothed": ("frame_path_distance_smoothed_px", "frame_path_distance_smoothed_mm"),
    }
    derivatives = data["speed_derivatives"]

    for source_level, group_name in MOVEMENT_SPEED_LEVEL_NAMES.items():
        level_group = speed_parent.create_group(group_name)
        derivative = derivatives[source_level]
        flat_px_key = f"{source_level}_px"
        flat_mm_key = f"{source_level}_mm"
        level_attrs: Dict[str, Any] = {
                "schema_id": MOVEMENT_SPEED_LEVEL_SCHEMA_ID,
                "source_speed_level": source_level,
                "level": group_name,
                "units_px": "px/s",
                "flat_speed_px_array": f"../../../{flat_px_key}",
                "time_delta_array": "../../../delta_seconds",
                "derivative_method": str(derivative.get("derivative_method", "first_difference")),
                "post_smoothing_method": str(derivative.get("post_smoothing_method", "moving_average")),
                "post_smoothing_alignment": str(derivative.get("post_smoothing_alignment", "centered")),
                "post_smoothing_window_frames": int(derivative.get("post_smoothing_window_frames", 1)),
                "post_smoothing_window_frames_requested": int(
                    derivative.get(
                        "post_smoothing_window_frames_requested",
                        derivative.get("post_smoothing_window_frames", 1),
                    )
                ),
                "post_smoothing_window_frames_effective": int(
                    derivative.get(
                        "post_smoothing_window_frames_effective",
                        derivative.get("post_smoothing_window_frames", 1),
                    )
                ),
                "post_smoothing_window_s": float(derivative.get("post_smoothing_window_s", 0.0)),
        }
        if include_physical:
            level_attrs.update(
                {
                    "units_mm": "mm/s",
                    "flat_speed_mm_array": f"../../../{flat_mm_key}",
                }
            )
        level_group.attrs.update(level_attrs)
        level_group.create_array("px", data=data[flat_px_key], chunks=chunks, overwrite=True)
        if include_physical:
            level_group.create_array(
                "mm",
                data=data[flat_mm_key],
                chunks=chunks,
                overwrite=True,
            )
        level_group.create_array(
            "acceleration_px",
            data=_float32(np.asarray(derivative["acceleration_px"])),
            chunks=chunks,
            overwrite=True,
        )
        level_group.create_array(
            "smoothed_acceleration_px",
            data=_float32(np.asarray(derivative["smoothed_acceleration_px"])),
            chunks=chunks,
            overwrite=True,
        )
        if include_physical:
            level_group.create_array(
                "acceleration_mm",
                data=_float32(np.asarray(derivative["acceleration_mm"])),
                chunks=chunks,
                overwrite=True,
            )
            level_group.create_array(
                "smoothed_acceleration_mm",
                data=_float32(np.asarray(derivative["smoothed_acceleration_mm"])),
                chunks=chunks,
                overwrite=True,
            )

        path_keys = path_distance_by_level.get(source_level)
        if path_keys is not None:
            path_px_key, path_mm_key = path_keys
            level_group.attrs["flat_frame_path_distance_px_array"] = f"../../../{path_px_key}"
            level_group.create_array(
                "frame_path_distance_px",
                data=data[path_px_key],
                chunks=chunks,
                overwrite=True,
            )
            if include_physical:
                level_group.attrs["flat_frame_path_distance_mm_array"] = (
                    f"../../../{path_mm_key}"
                )
                level_group.create_array(
                    "frame_path_distance_mm",
                    data=data[path_mm_key],
                    chunks=chunks,
                    overwrite=True,
                )


def _write_speed_derivative_groups(
    track_group: zarr.Group,
    derivatives: Dict[str, Dict[str, Any]],
    *,
    chunks: Tuple[int, ...],
    include_physical: bool,
) -> None:
    """Write speed-derived acceleration arrays grouped by source speed level."""

    parent = track_group.create_group("speed_derivatives")
    compatibility_aliases = {
        "acceleration_px": f"speed_derivatives/{DEFAULT_ACCELERATION_SOURCE_SPEED_LEVEL}/acceleration_px",
        "smoothed_acceleration_px": f"speed_derivatives/{DEFAULT_ACCELERATION_SOURCE_SPEED_LEVEL}/smoothed_acceleration_px",
    }
    if include_physical:
        compatibility_aliases.update(
            {
                "acceleration_mm": f"speed_derivatives/{DEFAULT_ACCELERATION_SOURCE_SPEED_LEVEL}/acceleration_mm",
                "smoothed_acceleration_mm": f"speed_derivatives/{DEFAULT_ACCELERATION_SOURCE_SPEED_LEVEL}/smoothed_acceleration_mm",
            }
        )
    parent.attrs.update(
        {
            "schema_id": SPEED_DERIVATIVES_SCHEMA_ID,
            "speed_levels": list(SPEED_DERIVATIVE_LEVELS),
            "default_source_speed_level": DEFAULT_ACCELERATION_SOURCE_SPEED_LEVEL,
            "compatibility_alias_arrays": compatibility_aliases,
        }
    )

    for level in SPEED_DERIVATIVE_LEVELS:
        if level not in derivatives:
            continue

        level_group = parent.create_group(level)
        item = derivatives[level]
        level_attrs: Dict[str, Any] = {
                "schema_id": SPEED_DERIVATIVE_SCHEMA_ID,
                "source_speed_level": level,
                "source_speed_px_array": f"../../{level}_px",
                "time_delta_array": "../../delta_seconds",
                "derivative_method": str(item.get("derivative_method", "first_difference")),
                "post_smoothing_method": str(item.get("post_smoothing_method", "moving_average")),
                "post_smoothing_alignment": str(item.get("post_smoothing_alignment", "centered")),
                "post_smoothing_window_frames": int(item.get("post_smoothing_window_frames", 1)),
                "post_smoothing_window_frames_requested": int(
                    item.get(
                        "post_smoothing_window_frames_requested",
                        item.get("post_smoothing_window_frames", 1),
                    )
                ),
                "post_smoothing_window_frames_effective": int(
                    item.get(
                        "post_smoothing_window_frames_effective",
                        item.get("post_smoothing_window_frames", 1),
                    )
                ),
                "post_smoothing_window_s": float(item.get("post_smoothing_window_s", 0.0)),
                "interpretation": (
                    "Framewise time derivative of the named source speed trace. "
                    "Use this group, not the legacy flat acceleration arrays, when "
                    "the source speed semantics matter."
                ),
        }
        if include_physical:
            level_attrs["source_speed_mm_array"] = f"../../{level}_mm"
        level_group.attrs.update(level_attrs)
        level_group.create_array(
            "acceleration_px",
            data=_float32(np.asarray(item["acceleration_px"])),
            chunks=chunks,
            overwrite=True,
        )
        level_group.create_array(
            "smoothed_acceleration_px",
            data=_float32(np.asarray(item["smoothed_acceleration_px"])),
            chunks=chunks,
            overwrite=True,
        )
        if include_physical:
            level_group.create_array(
                "acceleration_mm",
                data=_float32(np.asarray(item["acceleration_mm"])),
                chunks=chunks,
                overwrite=True,
            )
            level_group.create_array(
                "smoothed_acceleration_mm",
                data=_float32(np.asarray(item["smoothed_acceleration_mm"])),
                chunks=chunks,
                overwrite=True,
            )


def _write_run_array(group: zarr.Group, name: str, data: np.ndarray) -> None:
    """Create or overwrite an array under the track kinematics run group."""

    array = np.asarray(data)
    chunks = _track_preload_chunks(array.shape)
    kwargs: Dict[str, Any] = {"data": array, "overwrite": True}
    if chunks is not None:
        kwargs["chunks"] = chunks
    group.create_array(name, **kwargs)
    stamp_geometry_preload_attrs(group[name])


def _smooth_series(values: np.ndarray, window: int) -> np.ndarray:
    """Apply a centered moving average that ignores NaNs."""

    if window <= 1:
        return values.astype(np.float32, copy=True)

    series = np.asarray(values, dtype=np.float32)
    if series.size == 0:
        return series

    valid = np.isfinite(series)
    if not np.any(valid):
        return np.full(series.shape, np.nan, dtype=np.float32)

    kernel = np.ones(window, dtype=np.float32)
    filled = np.nan_to_num(series, nan=0.0, copy=False)
    counts = valid.astype(np.float32)
    sum_values = np.convolve(filled, kernel, mode="same")
    count_values = np.convolve(counts, kernel, mode="same")

    smoothed = np.full(series.shape, np.nan, dtype=np.float32)
    nonzero = count_values > 0
    smoothed[nonzero] = sum_values[nonzero] / count_values[nonzero]
    return smoothed


def _interpolate_gaps(values: np.ndarray, max_gap: int) -> np.ndarray:
    """Linearly interpolate NaN runs shorter than or equal to max_gap frames."""

    series = np.asarray(values, dtype=np.float32)
    if series.size == 0:
        return series
    if max_gap <= 0:
        return series.copy()

    result = series.copy()
    isnan = np.isnan(result)
    if not np.any(isnan):
        return result

    idx = 0
    length = result.shape[0]
    while idx < length:
        if not isnan[idx]:
            idx += 1
            continue
        start = idx
        while idx < length and isnan[idx]:
            idx += 1
        end = idx  # first finite after gap or len
        gap_size = end - start

        if gap_size == 0 or gap_size > max_gap:
            continue

        left_idx = start - 1
        right_idx = end
        if left_idx < 0 or right_idx >= length:
            continue

        left_val = result[left_idx]
        right_val = result[right_idx]
        if not np.isfinite(left_val) or not np.isfinite(right_val):
            continue

        step = (right_val - left_val) / (gap_size + 1)
        for offset in range(1, gap_size + 1):
            result[start + offset - 1] = left_val + step * offset

    return result


def _persist_chaser_metrics_to_run(
    run_group: zarr.Group,
    bundle: "ChaserMetricsBundle",
    *,
    fps: float,
    smooth_seconds: float,
    distance_interp_seconds: float,
) -> Dict[str, object]:
    """Write only non-coordinate legacy chaser fields at the run root.

    The legacy metrics bundle has no sealed row/frame/calibration authority for
    its pixel and millimetre fields.  Re-emitting those values under a canonical
    track run would turn an upstream ambiguity into a new publication claim, so
    coordinate/distance arrays are deliberately omitted.
    """

    metadata: Dict[str, object] = {
        "metrics_run": bundle.provenance.get("metrics_run"),
        "stimulus_run": bundle.provenance.get("stimulus_run"),
        "chaser_index": int(bundle.provenance.get("chaser_index", 0)),
    }

    shared_arrays: Dict[str, np.ndarray] = {
        "camera_frame_ids": np.asarray(bundle.camera_frame_ids, dtype=np.int64),
        "stimulus_frame_nums": np.asarray(bundle.stimulus_frame_nums, dtype=np.int64),
        "timestamp_ns": np.asarray(bundle.timestamp_ns, dtype=np.int64),
        "trial_state": np.asarray(bundle.trial_state, dtype=np.int16),
    }
    if bundle.metadata_mask is not None:
        shared_arrays["metadata_mask"] = np.asarray(bundle.metadata_mask, dtype=bool)

    offline = bundle.offline
    omitted_coordinate_fields: list[str] = []
    if offline:
        omitted_coordinate_fields.extend(
            name
            for name in (
                "distance_px",
                "distance_mm",
                "fish_centroid_px",
                "chaser_position_px",
            )
            if name in offline
        )
        if "angle_unsigned_deg" in offline:
            shared_arrays["angle_unsigned_deg"] = np.asarray(offline["angle_unsigned_deg"], dtype=np.float32)
        if "angle_signed_deg" in offline:
            shared_arrays["angle_signed_deg"] = np.asarray(offline["angle_signed_deg"], dtype=np.float32)
        if "heading_deg" in offline:
            shared_arrays["heading_deg"] = np.asarray(offline["heading_deg"], dtype=np.float32)
        if "has_offline" in offline:
            shared_arrays["has_offline"] = np.asarray(offline["has_offline"], dtype=bool)

    for name, array in shared_arrays.items():
        _write_run_array(run_group, name, array)

    interp_seconds_val = 0.0
    try:
        interp_seconds = float(distance_interp_seconds)
        if np.isfinite(interp_seconds) and interp_seconds > 0:
            interp_seconds_val = interp_seconds
    except (TypeError, ValueError):
        interp_seconds_val = 0.0
    metadata["distance_interpolation_seconds"] = float(interp_seconds_val)
    metadata["coordinate_geometry_status"] = (
        "omitted_untyped_legacy_chaser_metrics_v1"
        if omitted_coordinate_fields
        else "not_present"
    )
    metadata["coordinate_geometry_reason_code"] = (
        "LEGACY_METRICS_LACK_SEALED_COORDINATE_AUTHORITY"
        if omitted_coordinate_fields
        else "NONE"
    )
    metadata["omitted_coordinate_fields"] = sorted(omitted_coordinate_fields)
    return metadata


def _columnar_bout_data(bouts: np.ndarray) -> Dict[str, np.ndarray]:
    """Convert structured bout array to columnar float32/int32 arrays."""

    columns: Dict[str, np.ndarray] = {}
    if bouts.size == 0 or bouts.dtype.names is None:
        return columns

    for name in bouts.dtype.names:
        data = bouts[name]
        kind = data.dtype.kind
        if kind in {"f", "c"}:  # floats (complex not expected but guard)
            columns[name] = np.asarray(data, dtype=np.float32)
        elif kind in {"i", "u"}:
            columns[name] = np.asarray(data, dtype=np.int32)
        else:
            # skip unsupported fields (e.g. strings)
            continue
    return columns


def _mirror_swim_bouts_to_tracks(
    root: zarr.Group,
    run_group: zarr.Group,
    track_ids: Iterable[int],
    swim_bout_run: Optional[str],
    console: Console,
    *,
    expected_track_kinematics_run: Optional[str] = None,
) -> Optional[str]:
    analysis = root.get("analysis")
    if analysis is None or "swim_bout_runs" not in analysis:
        return None

    bouts_parent = analysis["swim_bout_runs"]
    run_name = swim_bout_run
    if not run_name or run_name.lower() == "latest":
        candidate = bouts_parent.attrs.get("latest")
        if isinstance(candidate, str) and candidate:
            run_name = candidate
    if not run_name:
        console.print(
            "[yellow]Warning:[/yellow] Unable to mirror swim bouts (no swim_bout_runs/latest attribute)."
        )
        return None
    if run_name not in bouts_parent:
        console.print(
            f"[yellow]Warning:[/yellow] Swim bout run '{run_name}' not found; skipping mirror."
        )
        return None

    try:
        default_payload = load_default_swim_bout_tables(root, run_name=run_name)
    except SwimBoutIOError as exc:
        console.print(
            f"[yellow]Warning:[/yellow] Unable to resolve swim bout run '{run_name}' "
            f"for legacy mirror: {exc}"
        )
        return None

    bout_group = bouts_parent[run_name]
    ordered_track_ids = [int(track_id) for track_id in track_ids]
    source_track_kinematics_run = (
        default_payload.candidate.source_track_kinematics_run
        or default_payload.run_attrs.get("source_track_kinematics_run")
    )
    if (
        expected_track_kinematics_run
        and source_track_kinematics_run
        and str(source_track_kinematics_run) != str(expected_track_kinematics_run)
    ):
        console.print(
            "[yellow]Warning:[/yellow] Swim bout run "
            f"'{run_name}' comes from track_kinematics run "
            f"'{source_track_kinematics_run}', not '{expected_track_kinematics_run}'; "
            "skipping mirror."
        )
        return None

    source_track_id = default_payload.candidate.track_id
    mirror_scope = "single_track_legacy"
    if source_track_id is not None:
        source_track_id = int(source_track_id)
        if source_track_id not in ordered_track_ids:
            console.print(
                "[yellow]Warning:[/yellow] Swim bout run "
                f"'{run_name}' is for track_id={source_track_id}, which is not present "
                "in this track_kinematics run; skipping mirror."
            )
            return None
        mirror_track_ids = [source_track_id]
        mirror_scope = "matched_track_id"
    elif len(ordered_track_ids) == 1:
        mirror_track_ids = ordered_track_ids
    else:
        console.print(
            "[yellow]Warning:[/yellow] Swim bout run "
            f"'{run_name}' has no track_id metadata and this track_kinematics run has "
            f"{len(ordered_track_ids)} tracks; skipping mirror to avoid copying one "
            "bout artifact into multiple identities."
        )
        return None

    track_subgroup_attrs = {
        "source_swim_bout_run": run_name,
        "source_track_kinematics_run": source_track_kinematics_run,
        "source_swim_bout_track_id": source_track_id,
        "source_swim_bout_candidate_id": int(default_payload.candidate.candidate_id),
        "source_swim_bout_default_signal_id": int(default_payload.signal.signal_id),
        "mirror_scope": mirror_scope,
        "default_level": default_payload.signal.speed_level or default_payload.signal.signal_name,
        "layout": str(default_payload.run_attrs.get("layout", "hierarchical_v1")),
        "is_hierarchical": all(level in bout_group for level in SPEED_DERIVATIVE_LEVELS),
    }

    signals = [signal for signal in default_payload.candidate.signals if signal.n_bouts > 0]
    if not signals:
        console.print(
            f"[yellow]Warning:[/yellow] Swim bout run '{run_name}' has no non-empty logical signals to mirror."
        )
        return None

    if len(signals) == 1 and not signals[0].speed_level:
        # Legacy flat structure - mirror as before at the track swim_bouts level.
        columns = _columnar_bout_data(default_payload.bouts)
        if not columns:
            console.print(
                f"[yellow]Warning:[/yellow] Swim bout run '{run_name}' contains no numeric bout fields to mirror."
            )
            return None
        tracks_parent = run_group["tracks"]
        for track_id in mirror_track_ids:
            subgroup = tracks_parent[f"id_{track_id}"].require_group("swim_bouts")
            for name in list(subgroup.array_keys()):
                del subgroup[name]
            for name, array in columns.items():
                chunks = _track_preload_chunks(array.shape)
                kwargs: Dict[str, Any] = {"data": array, "overwrite": True}
                if chunks is not None:
                    kwargs["chunks"] = chunks
                subgroup.create_array(name, **kwargs)
                stamp_geometry_preload_attrs(subgroup[name])
            subgroup.attrs.update({**track_subgroup_attrs, "mirrored_fields": list(columns.keys())})

        console.print(
            f"[dim]Mirrored legacy flat swim bouts from swim_bout_runs/{run_name} into "
            f"{len(mirror_track_ids)} track kinematics track(s).[/dim]"
        )
        return run_name

    tracks_parent = run_group["tracks"]
    for track_id in mirror_track_ids:
        track_subgroup = tracks_parent[f"id_{track_id}"].require_group("swim_bouts")
        track_subgroup.attrs.update(track_subgroup_attrs)

        for signal in signals:
            payload = (
                default_payload
                if signal.signal_id == default_payload.signal.signal_id
                else load_swim_bout_tables(
                    root,
                    run_name=run_name,
                    candidate_id=default_payload.candidate.candidate_id,
                    signal_id=signal.signal_id,
                )
            )
            columns = _columnar_bout_data(payload.bouts)
            if not columns:
                continue

            level_name = signal.speed_level or signal.signal_name or f"signal_{signal.signal_id}"
            level_subgroup = track_subgroup.require_group(level_name)
            for name in list(level_subgroup.array_keys()):
                del level_subgroup[name]
            for name, array in columns.items():
                chunks = _track_preload_chunks(array.shape)
                kwargs = {"data": array, "overwrite": True}
                if chunks is not None:
                    kwargs["chunks"] = chunks
                level_subgroup.create_array(name, **kwargs)
                stamp_geometry_preload_attrs(level_subgroup[name])
            level_subgroup.attrs.update(
                {
                    "speed_level": signal.speed_level,
                    "signal_id": int(signal.signal_id),
                    "signal_name": signal.signal_name,
                    "signal_role": signal.role,
                    "signal_source_level": signal.source_level,
                    "source_swim_bout_path": payload.level_path,
                    "n_bouts": len(payload.bouts),
                    "mirrored_fields": list(columns.keys()),
                }
            )

    console.print(
        f"[dim]Mirrored {len(signals)} logical swim-bout signal(s) from "
        f"swim_bout_runs/{run_name} into {len(mirror_track_ids)} "
        "track kinematics track(s).[/dim]"
    )
    return run_name


def summarize_to_table(
    summaries: List[Dict[str, float]],
    pixel_to_mm: Optional[float],
    console: Console,
) -> Tuple[float, float]:
    """Render a Rich table summarizing track metrics."""

    table = Table(title="Movement summary", show_lines=False)
    table.add_column("Track ID", justify="right")
    table.add_column("Samples", justify="right")
    # Speed metrics for all processing levels (mm/s)
    table.add_column("Mean raw mm/s", justify="right")
    table.add_column("Mean filt mm/s", justify="right")
    table.add_column("Mean smooth mm/s", justify="right")
    table.add_column("Mean avg mm/s", justify="right")
    # Path-distance totals (mm)
    table.add_column("Path raw mm", justify="right")
    table.add_column("Path filt mm", justify="right")
    table.add_column("Path smooth mm", justify="right")
    table.add_column("Cumul path mm", justify="right")
    # Other metrics
    table.add_column("Heading (deg)", justify="right")
    table.add_column("Head result", justify="right")
    table.add_column("Accel mm/s²", justify="right")

    total_px = 0.0
    total_mm = 0.0
    total_path_raw_mm = 0.0
    total_path_filt_mm = 0.0
    total_path_smooth_mm = 0.0

    for row in summaries:
        total_px += float(row.get("total_distance_px", 0.0))
        dist_mm = row.get("total_distance_mm", float("nan"))
        if not math.isnan(dist_mm):
            total_mm += float(dist_mm)

        # Track path-distance totals
        for key, var in [
            ("total_path_distance_raw_mm", "total_path_raw_mm"),
            ("total_path_distance_filtered_mm", "total_path_filt_mm"),
            ("total_path_distance_smoothed_mm", "total_path_smooth_mm"),
        ]:
            val = row.get(key, float("nan"))
            if not math.isnan(val):
                if var == "total_path_raw_mm":
                    total_path_raw_mm += float(val)
                elif var == "total_path_filt_mm":
                    total_path_filt_mm += float(val)
                elif var == "total_path_smooth_mm":
                    total_path_smooth_mm += float(val)

        table.add_row(
            str(int(row["track_id"])),
            str(int(row["samples"])),
            f"{row['mean_speed_raw_mm']:.2f}" if np.isfinite(row["mean_speed_raw_mm"]) else "nan",
            f"{row['mean_speed_filtered_mm']:.2f}" if np.isfinite(row["mean_speed_filtered_mm"]) else "nan",
            f"{row['mean_speed_smoothed_mm']:.2f}" if np.isfinite(row["mean_speed_smoothed_mm"]) else "nan",
            f"{row['mean_speed_averaged_mm']:.2f}" if np.isfinite(row["mean_speed_averaged_mm"]) else "nan",
            f"{row['total_path_distance_raw_mm']:.2f}" if np.isfinite(row["total_path_distance_raw_mm"]) else "nan",
            f"{row['total_path_distance_filtered_mm']:.2f}" if np.isfinite(row["total_path_distance_filtered_mm"]) else "nan",
            f"{row['total_path_distance_smoothed_mm']:.2f}" if np.isfinite(row["total_path_distance_smoothed_mm"]) else "nan",
            f"{row['total_distance_mm']:.2f}" if np.isfinite(row["total_distance_mm"]) else "nan",
            f"{row['heading_mean_deg']:.2f}" if np.isfinite(row["heading_mean_deg"]) else "nan",
            f"{row['heading_resultant']:.2f}" if np.isfinite(row["heading_resultant"]) else "nan",
            f"{row['mean_acceleration_mm']:.2f}" if np.isfinite(row["mean_acceleration_mm"]) else "nan",
        )

    console.print(table)
    if pixel_to_mm is not None:
        console.print(f"Total cumulative distance: {total_px:.2f} px ({total_mm:.2f} mm)")
        console.print(f"Total path distance (raw): {total_path_raw_mm:.2f} mm")
        console.print(f"Total path distance (filtered): {total_path_filt_mm:.2f} mm")
        console.print(f"Total path distance (smoothed): {total_path_smooth_mm:.2f} mm")
    else:
        console.print(f"Total cumulative distance: {total_px:.2f} px")
    return total_px, total_mm if pixel_to_mm is not None else float("nan")


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Create track_kinematics_runs entries consolidating detections, IDs, keypoints, and calibration.",
    )
    parser.add_argument("zarr_path", help="Path to the Palette Zarr archive.")
    parser.add_argument(
        "--output-zarr-path",
        help=(
            "Optional separate output Zarr. When provided, the source archive is "
            "opened read-only and all track-kinematics writes and run pointers are "
            "created in this output archive. Intended for node-local materialization."
        ),
    )
    parser.add_argument(
        "--_unbound-coordinate-stage",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--keypoint-run",
        help="Keypoint run to use. Prefix with 'refined/' to target a refined run. Default: latest refined if available.",
    )
    parser.add_argument(
        "--position-source-run",
        help=(
            "Optional current-v2 crop_runs successor that proves exact row identity "
            "with the selected keypoint run's immutable historical crop source."
        ),
    )
    parser.add_argument(
        "--run-name", help="Optional name for the output track kinematics run."
    )
    parser.add_argument(
        "--smooth-seconds",
        type=float,
        default=DEFAULT_SMOOTH_SECONDS,
        help=f"Smoothing window in seconds (default: {DEFAULT_SMOOTH_SECONDS}).",
    )
    parser.add_argument(
        "--distance-interpolation-seconds",
        type=float,
        default=0.0,
        help="Maximum gap duration (seconds) to fill via linear interpolation for chaser distances (default: 0).",
    )
    parser.add_argument(
        "--include-unassigned",
        action="store_true",
        help="Include track_id < 0 rows in offline outputs for diagnostic use.",
    )
    parser.add_argument("--fps", type=float, default=None, help="Override frames-per-second value.")
    parser.add_argument("--no-write", action="store_true", help="Do not write results back to the Zarr archive.")
    parser.add_argument(
        "--offline-only",
        action="store_true",
        help="Skip detection-based track kinematics run; only compute offline metrics run.",
    )
    parser.add_argument(
        "--online-only",
        action="store_true",
        help="Only compute detection-based track kinematics run, skipping offline metrics.",
    )
    parser.add_argument(
        "--metrics-run",
        help="Specific legacy analysis/chaser_fish_metrics/<run> to use for offline track kinematics (default: latest).",
    )
    parser.add_argument(
        "--swim-bout-run",
        help=(
            "Legacy compatibility only: analysis/swim_bout_runs/<run> to mirror into "
            "the offline track kinematics run (default: latest). New consumers "
            "should read authoritative bouts from analysis/swim_bout_runs via swim_bout_io."
        ),
    )
    parser.add_argument(
        "--chaser-index",
        type=int,
        default=0,
        help="Chaser index for offline metrics (default: 0).",
    )
    parser.add_argument(
        "--offline-run-name",
        help="Optional name for the offline track kinematics run (auto-generated if omitted).",
    )
    parser.add_argument(
        "--stimulus-run",
        help="Stimulus run name to filter online track kinematics data to the experimental period (default: latest).",
    )
    parser.add_argument(
        "--refined-online-run",
        help="Use refined online positions from refined_online_runs/<run> instead of raw online data (default: None).",
    )
    parser.add_argument(
        "--hysteresis-high-px",
        type=float,
        default=DEFAULT_HYSTERESIS_HIGH_PX,
        help=(
            "High threshold in pixels for hysteresis filter in offline analysis "
            f"(enter 'moving' state, default: {DEFAULT_HYSTERESIS_HIGH_PX})."
        ),
    )
    parser.add_argument(
        "--hysteresis-low-px",
        type=float,
        default=DEFAULT_HYSTERESIS_LOW_PX,
        help=(
            "Low threshold in pixels for hysteresis filter in offline analysis "
            f"(exit 'moving' state, default: {DEFAULT_HYSTERESIS_LOW_PX})."
        ),
    )
    parser.add_argument(
        "--hysteresis-min-frames",
        type=int,
        default=DEFAULT_HYSTERESIS_MIN_FRAMES,
        help=(
            "Minimum consecutive frames below low threshold to exit 'moving' state "
            f"in offline analysis (default: {DEFAULT_HYSTERESIS_MIN_FRAMES})."
        ),
    )
    parser.add_argument(
        "--hysteresis-band-policy",
        choices=HYSTERESIS_BAND_POLICIES,
        default=DEFAULT_HYSTERESIS_BAND_POLICY,
        help=(
            "How in-band displacements between low and high affect exit debounce "
            "in offline analysis: 'reset' preserves historical Palette behavior; "
            "'latch' keeps the counter unchanged, matching Schmitt-style hysteresis "
            f"(default: {DEFAULT_HYSTERESIS_BAND_POLICY})."
        ),
    )
    parser.add_argument(
        "--no-hysteresis",
        action="store_true",
        help="Disable hysteresis filter in offline analysis (allow all sub-pixel frame path-distance increments).",
    )
    parser.add_argument(
        "--smoothing-method",
        type=str,
        choices=["moving_average", "savitzky_golay"],
        default="moving_average",
        help="Smoothing method for frame path-distance in offline analysis: 'moving_average' (simple averaging) or 'savitzky_golay' (shape-preserving polynomial fit, better for derivatives) (default: moving_average)",
    )
    parser.add_argument(
        "--smoothing-alignment",
        type=str,
        choices=SMOOTHING_ALIGNMENTS,
        default=DEFAULT_SMOOTHING_ALIGNMENT,
        help=(
            "Temporal smoothing alignment: 'centered' uses past and future samples; "
            "'causal' uses only current/past samples and is supported for moving_average "
            f"(default: {DEFAULT_SMOOTHING_ALIGNMENT})."
        ),
    )
    parser.add_argument(
        "--savgol-polyorder",
        type=int,
        default=3,
        help="Polynomial order for Savitzky-Golay filter in offline analysis (default: 3, typical for biomechanics). Auto-adjusted if window too small.",
    )

    args = parser.parse_args(argv)

    console = Console()
    source_path = Path(args.zarr_path).expanduser().resolve()
    output_path = (
        Path(args.output_zarr_path).expanduser().resolve()
        if args.output_zarr_path
        else source_path
    )
    separate_output = output_path != source_path
    if args.no_write and separate_output:
        raise ValueError("--no-write cannot be combined with --output-zarr-path.")
    deferred_coordinate_stage = bool(args._unbound_coordinate_stage)
    if deferred_coordinate_stage and (
        not separate_output
        or args.no_write
        or not args.offline_only
        or args.online_only
        or not args.keypoint_run
        or not args.offline_run_name
    ):
        raise ValueError(
            "Internal unbound coordinate staging requires one separate output "
            "archive, --offline-only, --keypoint-run, and --offline-run-name."
        )
    root = open_zarr_root(
        source_path,
        mode="r" if args.no_write or separate_output else "a",
    )
    output_root = (
        open_zarr_root(output_path, mode="a") if separate_output else root
    )

    output_acquisition_frame = None
    if not args.no_write and not deferred_coordinate_stage:
        try:
            _, output_acquisition_frame = (
                load_persisted_acquisition_camera_authority(output_root)
            )
        except PixelFrameAuthorityError as exc:
            raise ValueError(
                "Canonical track publication requires a persisted acquisition-camera "
                "authority in the exact output archive. Historical archives must be "
                "classified and migrated explicitly; track kinematics will not infer "
                "frame identity from dimensions or row counts."
            ) from exc

    render_online = not args.offline_only
    render_offline = not args.online_only
    if args.offline_only and args.online_only:
        render_online = render_offline = True
    if not render_online and not render_offline:
        render_online = render_offline = True

    fps = float(args.fps) if args.fps else find_fps(root, console)
    if fps <= 0:
        raise ValueError("FPS must be positive.")

    if render_online:
        # Online track kinematics prefers the sealed refined surface and otherwise
        # selects an exact row subset from the sealed stimulus point surface.
        use_refined_online = False
        refined_run_name = None
        online_positions_px_source: BoundCanonicalCoordinateDescriptor | None = None
        online_positions_px_source_path: str | None = None
        online_positions_px_descriptor_sha256: str | None = None
        online_source_row_index: np.ndarray | None = None
        online_source_temporal_authority = None
        raw_online_rowset: zarr.Group | None = None
        handoff: CanonicalOnlineCoordinateHandoff | None = None
        online_heading_authority_node: Any | None = None

        # Check for refined online data (use by default if available)
        if "refined_online_runs" in root:
            refined_runs = root["refined_online_runs"]

            # Use specified run, or latest if not specified
            if args.refined_online_run is not None:
                refined_run_name = args.refined_online_run
            else:
                refined_run_name = refined_runs.attrs.get("latest")

            if refined_run_name and refined_run_name in refined_runs:
                console.print("[blue]Building online track kinematics run from refined_online_runs (refined positions)...[/blue]")

                refined_group = refined_runs[refined_run_name]
                console.print(f"[cyan]Using refined online run:[/cyan] {refined_run_name}")

                from fisheye.refinement.refine_online_detect import (
                    load_bound_refined_online_coordinate_evidence,
                )

                refined_evidence = load_bound_refined_online_coordinate_evidence(
                    root,
                    refined_group,
                )
                refined_evidence.assert_verified()
                interp_grp = refined_group["interpolated"]
                online_positions_px_source = refined_evidence.descriptor_for(
                    "interpolated"
                )
                positions_refined_array = online_positions_px_source.coordinate_node
                online_positions_px_source_path = positions_refined_array.path
                online_positions_px_descriptor_sha256 = (
                    online_positions_px_source.descriptor.digest()
                )
                frames_all = np.array(
                    refined_evidence.source_acquisition_frame_index,
                    copy=True,
                )
                positions_refined = np.array(
                    positions_refined_array[:],
                    copy=True,
                    order="C",
                )
                valid_mask_refined = interp_grp["valid_mask"][:]
                online_source_row_index = np.arange(
                    positions_refined.shape[0],
                    dtype=np.int64,
                )
                online_source_temporal_authority = (
                    refined_evidence.source_temporal_authority
                )

                # Get source stimulus run for provenance
                stimulus_run_name = refined_group.attrs.get("source_stimulus_run")
                coordinate_space = online_positions_px_source.descriptor.space_id
                # A numeric projector scale alone does not define a physical
                # frame for arena-relative/canvas coordinates.  Preserve px
                # coordinates and fail closed on physical publication until a
                # sealed direction-labelled physical authority exists.
                pixel_to_mm_online = None

                positions_online = positions_refined
                use_refined_online = True

                console.print(f"  Source stimulus run: {stimulus_run_name}")
                console.print(f"  Coordinate space: {coordinate_space}")
                console.print(f"  Refined frames: {len(frames_all)}")
                console.print(f"  Valid frames: {valid_mask_refined.sum()} ({valid_mask_refined.sum()/len(frames_all)*100:.1f}%)")
            else:
                console.print(f"[yellow]Note:[/yellow] Refined run '{refined_run_name}' not found; using raw online data.")
        else:
            console.print("[yellow]Note:[/yellow] No refined_online_runs found; using raw online data.")

        if not use_refined_online:
            console.print(
                "[blue]Building online track kinematics directly from the "
                "canonical stimulus coordinate surface...[/blue]"
            )

            pixel_to_mm_online = None

            try:
                analysis_group = root["analysis"]
                stimulus_runs = analysis_group["stimulus_runs"]
                stimulus_run_name = args.stimulus_run or stimulus_runs.attrs.get(
                    "latest_complete"
                )
                if (
                    not isinstance(stimulus_run_name, str)
                    or stimulus_run_name not in stimulus_runs
                ):
                    raise ValueError(
                        "No exact complete canonical stimulus run is selected."
                    )
                stimulus_group = stimulus_runs[stimulus_run_name]
                raw_online_rowset = stimulus_group["tracking_data/chaser_states"]
                _, _, handoff = load_canonical_online_coordinate_surface(
                    root,
                    stimulus_group,
                    raw_online_rowset,
                )
            except Exception as exc:
                console.print(f"[yellow]Warning:[/yellow] Unable to load stimulus run data ({exc}).")
                console.print("[yellow]Skipping online track kinematics run.[/yellow]")
                render_online = False

            if render_online:
                if handoff is None:
                    raise ValueError(
                        "Canonical online track publication requires the exact "
                        "stimulus coordinate handoff."
                    )
                online_positions_px_source = (
                    require_bound_canonical_coordinate_descriptor(
                        handoff.coordinate_descriptor
                    )
                )
                (
                    online_source_row_index,
                    frames_all,
                    positions_online,
                ) = select_canonical_online_track_rows(
                    handoff,
                    chaser_index=args.chaser_index,
                )
                online_source_temporal_authority = handoff.source_temporal_authority
                online_positions_px_source_path = (
                    online_positions_px_source.coordinate_node.path
                )
                online_positions_px_descriptor_sha256 = (
                    online_positions_px_source.descriptor.digest()
                )
                coordinate_space = online_positions_px_source.descriptor.space_id

        if render_online:
            # Get heading from online fields if available, otherwise NaN
            if use_refined_online:
                # Refined data doesn't have heading, use NaN
                heading_online = np.full(frames_all.shape, np.nan, dtype=np.float64)
            else:
                assert raw_online_rowset is not None
                heading_node = raw_online_rowset.get("visual_angle_deg")
                if heading_node is not None:
                    online_heading_authority_node = heading_node
                    heading_values = np.asarray(heading_node[:], dtype=np.float64)
                    assert online_source_row_index is not None
                    if heading_values.shape != (
                        online_positions_px_source.row_identity.leading_dimension,
                    ):
                        raise ValueError(
                            "Stimulus visual_angle_deg is not aligned to the exact "
                            "coordinate row identity."
                        )
                    heading_online = heading_values[online_source_row_index]
                else:
                    heading_online = np.full(frames_all.shape, np.nan, dtype=np.float64)

            # Use ALL frames from stimulus run (PRE + TRAINING + POST periods)
            # No filtering needed since chaser positions are logged for all trial states
            frames_online = frames_all

            # Single track ID for online (chaser)
            track_ids_online = np.zeros(frames_online.shape[0], dtype=np.int64)
            keypoint_success_online = np.ones(frames_online.shape[0], dtype=bool)

            online_input_authority = build_track_motion_input_authority(
                root,
                source_positions=online_positions_px_source,
                mode="online_exact_or_generated_v1",
                heading_node=online_heading_authority_node,
                generated_track_id=0,
            )

            console.print(f"[blue]Online frames:[/blue] {frames_online.shape[0]} (full experimental session: PRE + TRAINING + POST)")

            tracks_online, summaries_online = build_track_datasets(
                track_ids=track_ids_online,
                frames=frames_online,
                positions_px=positions_online,
                headings_deg=heading_online,
                keypoint_success=keypoint_success_online,
                detection_source=None,
                fps=fps,
                smooth_seconds=args.smooth_seconds,
                pixel_to_mm=pixel_to_mm_online,
                smoothing_method=args.smoothing_method,
                smoothing_alignment=args.smoothing_alignment,
                savgol_polyorder=args.savgol_polyorder,
                source_row_index=(
                    online_source_row_index
                ),
                source_temporal_authority=online_source_temporal_authority,
            )

            if not summaries_online:
                console.print("[yellow]Warning:[/yellow] Online data produced no tracks.")
            else:
                total_px_online, total_mm_online = summarize_to_table(summaries_online, pixel_to_mm_online, console)

                if args.no_write:
                    console.print("[green]Skipping online write (--no-write).[/green]")
                else:
                    if online_positions_px_source is None:
                        raise ValueError(
                            "Canonical online track publication lacks a sealed "
                            "positions source."
                        )
                    run_name, run_group = ensure_track_kinematics_run_group(
                        output_root, args.run_name, run_type="online"
                    )
                    online_publication_owner_uuid = (
                        _track_publication_owner_uuid(run_group)
                    )
                    ordered_track_ids = save_track_kinematics_tracks(
                        run_group,
                        tracks_online,
                        summaries_online,
                        source_temporal_authority=(
                            online_source_temporal_authority
                        ),
                        positions_px_source=online_positions_px_source,
                        input_authority=online_input_authority,
                    )

                    created_at = datetime.now(timezone.utc).isoformat()

                    # Gather git and environment info for provenance
                    git_info = get_git_info()
                    env_info = get_environment_info()

                    if use_refined_online:
                        inputs = {
                            "refined_online_run": refined_run_name,
                            "stimulus_run": stimulus_run_name,
                            "chaser_index": args.chaser_index,
                            "positions_px_source_path": online_positions_px_source_path,
                            "positions_px_coordinate_descriptor_sha256": (
                                online_positions_px_descriptor_sha256
                            ),
                        }
                        method = "track_kinematics_online_refined"
                        saved_coordinate_space = coordinate_space
                    else:
                        inputs = {
                            "stimulus_run": stimulus_run_name,
                            "chaser_index": int(args.chaser_index),
                            "positions_px_source_path": online_positions_px_source_path,
                            "positions_px_coordinate_descriptor_sha256": (
                                online_positions_px_descriptor_sha256
                            ),
                        }
                        method = "track_kinematics_online"
                        saved_coordinate_space = coordinate_space

                    # Canonical stage provenance.
                    online_params = {
                        "fps": fps,
                        "smoothing_seconds": args.smooth_seconds,
                        "smoothing_method": args.smoothing_method,
                        "smoothing_alignment": args.smoothing_alignment,
                        "savgol_polyorder": int(args.savgol_polyorder) if args.smoothing_method == "savitzky_golay" else None,
                        "hysteresis_enabled": False,
                        "hysteresis_high_px": None,
                        "hysteresis_low_px": None,
                        "hysteresis_min_frames": None,
                        "hysteresis_band_policy": (
                            DEFAULT_HYSTERESIS_BAND_POLICY
                        ),
                        "coordinate_space": saved_coordinate_space,
                    }
                    provenance = build_stage_provenance(
                        stage="track_kinematics",
                        created_at_utc=created_at,
                        parameters=online_params,
                        inputs=inputs,
                        command=" ".join(sys.argv),
                        git=git_info,
                        environment=env_info.get("platform"),
                    )
                    write_stage_provenance(run_group, provenance)

                    # Backward-compatible top-level attrs.
                    run_group.attrs.update(
                        {
                            **_track_kinematics_contract_attrs(
                                run_type="online",
                                method=method,
                                parameters=online_params,
                                inputs=inputs,
                            ),
                            "created_at_utc": created_at,
                            "fps": fps,
                            "smoothing_seconds": args.smooth_seconds,
                            "smoothing_method": args.smoothing_method,
                            "smoothing_alignment": args.smoothing_alignment,
                            "savgol_polyorder": int(args.savgol_polyorder) if args.smoothing_method == "savitzky_golay" else None,
                            "hysteresis_enabled": False,
                            "hysteresis_high_px": None,
                            "hysteresis_low_px": None,
                            "hysteresis_min_frames": None,
                            "hysteresis_band_policy": (
                                DEFAULT_HYSTERESIS_BAND_POLICY
                            ),
                            "inputs": inputs,
                            "coordinate_space": saved_coordinate_space,
                            "positions_px_source_path": online_positions_px_source_path,
                            "positions_px_source_coordinate_descriptor_sha256": (
                                online_positions_px_descriptor_sha256
                            ),
                            "num_tracks": len(ordered_track_ids),
                            "source_zarr": str(source_path),
                            "output_zarr": str(output_path),
                        }
                    )
                    write_best_effort_run_lineage_attrs(
                        run_group,
                        run_family="track_kinematics_run",
                    )
                    mark_track_kinematics_run_complete(
                        output_root,
                        run_group,
                        run_name=run_name,
                        run_type="online",
                        publication_owner_uuid=online_publication_owner_uuid,
                        validate_complete_run=lambda fresh_run: (
                            _validate_direct_track_kinematics_run_before_selection(
                                output_root,
                                fresh_run,
                                run_name=run_name,
                                run_type="online",
                                source_positions=online_positions_px_source,
                                source_temporal_authority=(
                                    online_source_temporal_authority
                                ),
                                physical_authority=None,
                            )
                        ),
                    )

                    console.print(
                        f"[green]✓[/green] Saved track kinematics run to [bold]analysis/track_kinematics_runs/online/{run_name}[/bold]"
                    )

    if render_offline:
        # Offline track kinematics now uses all keypoint frames across the video.
        console.print("[blue]Building offline track kinematics run from all keypoint frames...[/blue]")

        keypoints_offline = resolve_keypoint_group(root, args.keypoint_run, console)
        position_crop_run = (
            _controlled_run_leaf(
                args.position_source_run,
                label="position_source_run",
            )
            if args.position_source_run
            else keypoints_offline.crop_run
        )
        crop_group_offline = root[f"crop_runs/{position_crop_run}"]
        position_source_offline = load_canonical_offline_position_source(
            root,
            crop_group_offline,
            crop_run_name=position_crop_run,
        )
        positions_offline = position_source_offline.positions_px
        frame_indices_offline = position_source_offline.frame_indices
        detection_source_offline = position_source_offline.detection_source
        canonical_position_surface = position_source_offline.position_surface
        if canonical_position_surface is None:
            raise ValueError(
                "Canonical offline track publication requires a sealed source-camera "
                "position surface."
            )
        if args.position_source_run:
            successor_tracking = resolve_collection_proxy_successor_tracking(
                root,
                keypoints=keypoints_offline,
                position_crop_run=position_crop_run,
            )
            detection_path_offline = (
                "source_detect_run:" + successor_tracking.expected_detect_run
            )
            expected_detect_run = successor_tracking.expected_detect_run
            expected_refined_run = None
            tracking_source_rowset_path = (
                successor_tracking.historical_source_rowset_path
            )
            tracking_source_fingerprint = (
                successor_tracking.expected_source_rowset_fingerprint
            )
            console.print(
                "[cyan]Using verified coordinate-successor geometry:[/cyan] "
                f"crop_runs/{position_crop_run} "
                f"(historical_tracking_rowset={tracking_source_rowset_path})"
            )
        else:
            detection_path_offline = _canonical_crop_detection_rowset_path(
                canonical_position_surface.coordinates
            )
            detection_offline = resolve_detection_from_path(
                root,
                detection_path_offline,
            )
            expected_detect_run = (
                detection_offline.source_detect_run or detection_offline.run_name
            )
            expected_refined_run = (
                detection_offline.run_name if detection_offline.is_refined else None
            )
            tracking_source_rowset_path = position_source_offline.path
            tracking_source_fingerprint = position_source_offline.rowset_fingerprint
            console.print(
                "[cyan]Using canonical crop-bound offline geometry:[/cyan] "
                f"crop_runs/{position_crop_run} "
                f"(detection_rowset={detection_path_offline})"
            )
        (
            offline_physical_authority,
            offline_physical_calibration_info,
        ) = resolve_track_physical_authority(
            root,
            stimulus_run=args.stimulus_run,
        )
        if offline_physical_authority is None and not args.no_write:
            raise ValueError(
                "Offline track-kinematics publication requires sealed source-camera "
                "physical calibration authority; none was available "
                "(reason_code="
                f"{offline_physical_calibration_info.get('reason_code')!r})."
            )
        offline_mm_per_pixel = (
            offline_physical_authority.mm_per_pixel
            if offline_physical_authority is not None
            else None
        )
        offline_physical_reason_code = str(
            offline_physical_calibration_info["reason_code"]
        )
        offline_source_temporal_authority = (
            canonical_position_surface.temporal_authority
            if not args.no_write
            else None
        )

        heading_offline = keypoints_offline.group["heading"][:]
        if heading_offline.shape[0] != positions_offline.shape[0]:
            raise ValueError(
                "Offline: Heading array length does not match position source row count "
                f"(heading={heading_offline.shape[0]}, "
                f"position_source={positions_offline.shape[0]}, "
                f"position_source_path={position_source_offline.path})."
            )
        if frame_indices_offline.shape[0] != positions_offline.shape[0]:
            raise ValueError(
                "Offline: Frame-index array length does not match position source row count "
                f"(frame_indices={frame_indices_offline.shape[0]}, "
                f"position_source={positions_offline.shape[0]}, "
                f"position_source_path={position_source_offline.path})."
            )

        keypoint_success_offline, keypoint_usability_dataset = (
            load_keypoint_usability_array(
                keypoints_offline.group,
                expected_length=heading_offline.shape[0],
            )
        )

        if not expected_detect_run:
            raise ValueError(
                "Offline: unable to determine source_detect_run for tracking lookup."
            )
        track_ids_offline, tracking_metadata = load_tracking_ids(
            root,
            frame_indices_offline.shape[0],
            expected_detect_run=expected_detect_run,
            expected_refined_run=expected_refined_run,
            expected_source_rowset_path=tracking_source_rowset_path,
            expected_instance_key=position_source_offline.instance_key,
            expected_source_rowset_fingerprint=tracking_source_fingerprint,
            return_metadata=True,
        )
        track_ids_offline = track_ids_offline.astype(np.int64, copy=False)
        track_id_to_arena_id = {
            int(track_id): int(arena_id)
            for track_id, arena_id in (
                tracking_metadata.get("track_id_to_arena_id", {}) or {}
            ).items()
        }
        tracking_run_name = tracking_metadata.get("track_run")
        if not isinstance(tracking_run_name, str) or not tracking_run_name:
            raise ValueError("Offline tracking metadata lacks one exact run name.")
        tracking_group_offline = root[f"tracking_runs/{tracking_run_name}"]
        keypoint_row_key_node = _motion_input_child(
            keypoints_offline.group,
            "instance_key",
        )
        keypoint_usability_node = (
            None
            if keypoint_usability_dataset == "implicit_all_true"
            else keypoints_offline.group[keypoint_usability_dataset]
        )
        offline_input_authority = build_track_motion_input_authority(
            root,
            source_positions=canonical_position_surface.coordinates,
            mode="offline_exact_sources_v1",
            heading_node=keypoints_offline.group["heading"],
            keypoint_usability_node=keypoint_usability_node,
            keypoint_row_key_node=keypoint_row_key_node,
            tracking_group=tracking_group_offline,
            detection_source_node=_motion_input_child(
                position_source_offline.rowset_group,
                "detection_source",
            ),
        )

        console.print(f"[blue]Offline frames:[/blue] {frame_indices_offline.shape[0]} (all keypoint detections)")

        if frame_indices_offline.size == 0:
            console.print("[yellow]Warning:[/yellow] No offline frames available; skipping.")
        else:
            proceed_offline = True
            public_row_mask = (
                np.ones(track_ids_offline.shape[0], dtype=bool)
                if args.include_unassigned
                else track_ids_offline >= 0
            )
            source_row_index_offline = np.flatnonzero(public_row_mask).astype(
                np.int64,
                copy=False,
            )
            (
                track_ids_offline,
                frame_indices_offline,
                positions_offline,
                heading_offline,
                keypoint_success_offline,
                detection_source_offline,
            ) = _filter_public_track_rows(
                track_ids=track_ids_offline,
                frames=frame_indices_offline,
                positions_px=positions_offline,
                headings_deg=heading_offline,
                keypoint_success=keypoint_success_offline,
                detection_source=detection_source_offline,
                include_unassigned=args.include_unassigned,
            )
            if frame_indices_offline.size == 0:
                console.print(
                    "[yellow]Warning:[/yellow] All offline detections are unassigned; skipping public offline track kinematics run."
                )
                proceed_offline = False

            if not proceed_offline or frame_indices_offline.size == 0:
                console.print("[yellow]Warning:[/yellow] No offline detections remaining after filtering; skipping.")
            else:
                # Prepare hysteresis parameters for offline analysis
                hysteresis_high = None if args.no_hysteresis else args.hysteresis_high_px
                hysteresis_low = None if args.no_hysteresis else args.hysteresis_low_px
                hysteresis_min = None if args.no_hysteresis else args.hysteresis_min_frames

                tracks_offline, summaries_offline = build_track_datasets(
                    track_ids=track_ids_offline,
                    frames=frame_indices_offline,
                    positions_px=positions_offline,
                    headings_deg=heading_offline,
                    keypoint_success=keypoint_success_offline,
                    detection_source=detection_source_offline,
                    fps=fps,
                    smooth_seconds=args.smooth_seconds,
                    pixel_to_mm=offline_mm_per_pixel,
                    hysteresis_high_px=hysteresis_high,
                    hysteresis_low_px=hysteresis_low,
                    hysteresis_min_frames=hysteresis_min,
                    hysteresis_band_policy=args.hysteresis_band_policy,
                    smoothing_method=args.smoothing_method,
                    smoothing_alignment=args.smoothing_alignment,
                    savgol_polyorder=args.savgol_polyorder,
                    source_row_index=(
                        source_row_index_offline
                        if offline_source_temporal_authority is not None
                        else None
                    ),
                    source_temporal_authority=(
                        offline_source_temporal_authority
                    ),
                )

                if not summaries_offline:
                    console.print("[yellow]Warning:[/yellow] Offline metrics produced no tracks.")
                else:
                    total_px_offline, total_mm_offline = summarize_to_table(
                        summaries_offline, offline_mm_per_pixel, console
                    )

                    if args.no_write:
                        console.print("[green]Skipping offline write (--no-write).[/green]")
                    else:
                        offline_run_name = args.offline_run_name
                        if not offline_run_name:
                            # Use keypoint run name as basis for the offline run name.
                            offline_run_name = (
                                f"{keypoints_offline.run_name}_track_kinematics"
                            )

                        offline_run_name, offline_group = ensure_track_kinematics_run_group(
                            output_root,
                            offline_run_name,
                            run_type="offline",
                            overwrite=True,
                        )
                        offline_publication_owner_uuid = (
                            _track_publication_owner_uuid(offline_group)
                        )
                        ordered_ids_offline = save_track_kinematics_tracks(
                            offline_group,
                            tracks_offline,
                            summaries_offline,
                            source_temporal_authority=(
                                offline_source_temporal_authority
                            ),
                            positions_px_source=canonical_position_surface.coordinates,
                            input_authority=offline_input_authority,
                            physical_authority=offline_physical_authority,
                            physical_omission_reason_code=(
                                offline_physical_reason_code
                            ),
                            track_id_to_arena_id=track_id_to_arena_id,
                            defer_coordinate_binding=deferred_coordinate_stage,
                            staging_keypoint_run=(
                                str(args.keypoint_run)
                                if deferred_coordinate_stage
                                else None
                            ),
                            staging_run_name=(
                                offline_run_name
                                if deferred_coordinate_stage
                                else None
                            ),
                        )

                        metrics_metadata: Optional[Dict[str, object]] = None
                        swim_bout_mirror: Optional[str] = None
                        if deferred_coordinate_stage:
                            chaser_bundle = None
                        else:
                            try:
                                chaser_bundle = load_chaser_metrics(
                                    args.zarr_path,
                                    stimulus_run=args.stimulus_run,
                                    metrics_run=args.metrics_run,
                                    chaser_index=args.chaser_index,
                                )
                            except Exception as exc:
                                console.print(
                                    f"[yellow]Warning:[/yellow] Failed to load chaser metrics for offline run: {exc}"
                                )
                                chaser_bundle = None

                        if chaser_bundle is not None:
                            has_offline = chaser_bundle.offline.get("has_offline")
                            has_values = bool(has_offline is not None and np.any(has_offline))
                            if has_values:
                                metrics_metadata = _persist_chaser_metrics_to_run(
                                    offline_group,
                                    chaser_bundle,
                                    fps=fps,
                                    smooth_seconds=args.smooth_seconds,
                                    distance_interp_seconds=args.distance_interpolation_seconds,
                                )
                                run_id = metrics_metadata.get("metrics_run") or "latest"
                                console.print(
                                    f"[cyan]Stored chaser metrics arrays[/cyan] "
                                    f"(analysis/chaser_fish_metrics/{run_id})."
                                )
                            else:
                                console.print(
                                    "[yellow]Warning:[/yellow] Chaser metrics bundle contains no valid offline data; "
                                    "skipping shared metrics write."
                                )

                        if not deferred_coordinate_stage:
                            try:
                                swim_bout_mirror = _mirror_swim_bouts_to_tracks(
                                    root,
                                    offline_group,
                                    ordered_ids_offline,
                                    args.swim_bout_run,
                                    console,
                                    expected_track_kinematics_run=offline_run_name,
                                )
                            except Exception as exc:
                                console.print(
                                    f"[yellow]Warning:[/yellow] Failed to mirror swim bouts: {exc}"
                                )

                        created_at = datetime.now(timezone.utc).isoformat()

                        # Gather git and environment info for provenance
                        git_info = get_git_info()
                        env_info = get_environment_info()

                        offline_inputs = {
                            "detection_path": detection_path_offline,
                            **_offline_position_source_inputs(
                                position_source_offline
                            ),
                            "keypoint_path": (
                                "refined_keypoints_runs/"
                                if keypoints_offline.is_refined
                                else "keypoints_runs/"
                            )
                            + keypoints_offline.run_name,
                            "crop_run": position_crop_run,
                            "keypoint_source_crop_run": keypoints_offline.crop_run,
                            "tracking_source_rowset_path": (
                                tracking_source_rowset_path
                            ),
                            "tracking_path": f"tracking_runs/{tracking_run_name}",
                        }
                        if metrics_metadata:
                            offline_inputs["chaser_metrics"] = metrics_metadata
                        if swim_bout_mirror:
                            offline_inputs["swim_bout_run"] = swim_bout_mirror

                        # Canonical stage provenance.
                        offline_params = {
                            "fps": fps,
                            "smoothing_seconds": args.smooth_seconds,
                            "smoothing_method": args.smoothing_method,
                            "smoothing_alignment": args.smoothing_alignment,
                            "savgol_polyorder": int(args.savgol_polyorder) if args.smoothing_method == "savitzky_golay" else None,
                            "distance_interpolation_seconds": args.distance_interpolation_seconds,
                            "coordinate_space": "source_camera_image_px",
                            "hysteresis_enabled": not args.no_hysteresis,
                            "hysteresis_high_px": float(args.hysteresis_high_px) if not args.no_hysteresis else None,
                            "hysteresis_low_px": float(args.hysteresis_low_px) if not args.no_hysteresis else None,
                            "hysteresis_min_frames": int(args.hysteresis_min_frames) if not args.no_hysteresis else None,
                            "hysteresis_band_policy": args.hysteresis_band_policy,
                        }
                        offline_provenance = build_stage_provenance(
                            stage="track_kinematics",
                            created_at_utc=created_at,
                            parameters=offline_params,
                            inputs=offline_inputs,
                            command=" ".join(sys.argv),
                            git=git_info,
                            environment=env_info.get("platform"),
                        )
                        write_stage_provenance(offline_group, offline_provenance)

                        # Backward-compatible top-level attrs.
                        offline_group.attrs.update(
                            {
                                **_track_kinematics_contract_attrs(
                                    run_type="offline",
                                    method="track_kinematics_offline",
                                    parameters=offline_params,
                                    inputs=offline_inputs,
                                ),
                                "created_at_utc": created_at,
                                "fps": fps,
                                "smoothing_seconds": args.smooth_seconds,
                                "smoothing_method": args.smoothing_method,
                                "smoothing_alignment": args.smoothing_alignment,
                                "savgol_polyorder": int(args.savgol_polyorder) if args.smoothing_method == "savitzky_golay" else None,
                                "distance_interpolation_seconds": args.distance_interpolation_seconds,
                                "hysteresis_enabled": not args.no_hysteresis,
                                "hysteresis_high_px": float(args.hysteresis_high_px) if not args.no_hysteresis else None,
                                "hysteresis_low_px": float(args.hysteresis_low_px) if not args.no_hysteresis else None,
                                "hysteresis_min_frames": int(args.hysteresis_min_frames) if not args.no_hysteresis else None,
                                "hysteresis_band_policy": args.hysteresis_band_policy,
                                "inputs": offline_inputs,
                                "num_tracks": len(ordered_ids_offline),
                                "source_zarr": str(source_path),
                                "output_zarr": str(output_path),
                            }
                        )
                        write_best_effort_run_lineage_attrs(
                            offline_group,
                            run_family="track_kinematics_run",
                        )
                        if deferred_coordinate_stage:
                            track_parent = output_root["analysis"][
                                "track_kinematics_runs"
                            ]
                            mark_run_complete(
                                offline_group,
                                parent_group=track_parent,
                                run_name=f"offline/{offline_run_name}",
                                run_provenance=(
                                    build_run_provenance_from_stage_record(
                                        offline_group.attrs.get("provenance", {}),
                                        fallback_command="track_kinematics",
                                    )
                                ),
                            )
                            if (
                                offline_group.attrs.get(
                                    "stage_selector_eligible"
                                )
                                is not False
                            ):
                                raise RuntimeError(
                                    "Unbound track stage became selector-eligible."
                                )
                        else:
                            mark_track_kinematics_run_complete(
                                output_root,
                                offline_group,
                                run_name=offline_run_name,
                                run_type="offline",
                                publication_owner_uuid=(
                                    offline_publication_owner_uuid
                                ),
                                validate_complete_run=lambda fresh_run: (
                                    _validate_direct_track_kinematics_run_before_selection(
                                        output_root,
                                        fresh_run,
                                        run_name=offline_run_name,
                                        run_type="offline",
                                        source_positions=(
                                            canonical_position_surface.coordinates
                                        ),
                                        source_temporal_authority=(
                                            offline_source_temporal_authority
                                        ),
                                        physical_authority=(
                                            offline_physical_authority
                                        ),
                                    )
                                ),
                            )

                        console.print(
                            f"[green]✓[/green] Saved offline track kinematics run to [bold]analysis/track_kinematics_runs/offline/{offline_run_name}[/bold]"
                        )


__all__ = [
    "BoundTrackMotionRun",
    "BoundTrackMotionSurface",
    "BoundTrackMotionTrack",
    "BoundTrackPositionBindings",
    "DeferredTrackKinematicsSelectorActivation",
    "TrackPhysicalAuthority",
    "TrackSpeeds",
    "_ordered_track_arena_ids",
    "bind_staged_offline_track_kinematics_run",
    "compute_track_speed",
    "find_fps",
    "_filter_public_track_rows",
    "load_arena_ids",
    "load_bound_track_motion_run",
    "load_bound_track_position_bindings",
    "resolve_dimensions",
    "rollback_deferred_track_kinematics_selector_activation",
    "resolve_track_physical_authority",
    "stage_offline_track_kinematics_run",
    "validate_bound_track_motion_run",
    "validate_bound_track_position_bindings",
    "main",
]


if __name__ == "__main__":  # pragma: no cover
    main()
