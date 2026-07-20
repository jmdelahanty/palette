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
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
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
    load_persisted_source_camera_position_surface,
    require_bound_source_camera_position_surface,
)
from fisheye.shared.stage_provenance import (
    build_stage_provenance,
    write_stage_provenance,
)
from fisheye.shared.run_provenance import build_run_provenance_from_stage_record
from fisheye.shared.run_lineage_fingerprint import write_best_effort_run_lineage_attrs
from fisheye.shared.rowset_fingerprint import (
    RowsetFingerprint,
    build_rowset_fingerprint,
    build_group_rowset_fingerprint,
    resolve_rowset_edit_revision,
)
from fisheye.shared.zarr.chunk_profiles import (
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
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.shared.zarr.columnar import load_structured_dataset
from .swim_bout_io import SwimBoutIOError, load_default_swim_bout_tables, load_swim_bout_tables


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
    physical_authority: BoundStimulusPhysicalCoordinateAuthority | None
    track_positions: tuple[tuple[int, TrackPositionPublicationResult], ...]
    run_group: zarr.Group

    def position_for_track(self, track_id: int) -> TrackPositionPublicationResult:
        for candidate_id, binding in self.track_positions:
            if candidate_id == int(track_id):
                return binding
        raise KeyError(f"Track {track_id} is not present in /{self.run_group.path}.")


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


def _track_kinematics_source_refs(
    *,
    run_type: str,
    inputs: Mapping[str, Any],
) -> Dict[str, Any]:
    """Normalize exact archive-relative dependencies for one run."""

    refs: Dict[str, Any] = {}

    def _set_path(key: str, prefix: str, value: Any) -> None:
        if value in (None, ""):
            return
        text = str(value)
        refs[key] = text if "/" in text else f"{prefix}/{text}"

    if run_type == "online":
        _set_path(
            "source_refined_online_path",
            "refined_online_runs",
            inputs.get("refined_online_run"),
        )
        _set_path(
            "source_stimulus_path",
            "analysis/stimulus_runs",
            inputs.get("stimulus_run"),
        )
        positions_path = inputs.get("positions_px_source_path")
        if positions_path not in (None, ""):
            refs["source_positions_px_path"] = str(positions_path)
        descriptor_digest = inputs.get(
            "positions_px_coordinate_descriptor_sha256"
        )
        if descriptor_digest not in (None, ""):
            refs["source_positions_px_coordinate_descriptor_sha256"] = str(
                descriptor_digest
            )
        if inputs.get("chaser_index") is not None:
            refs["source_chaser_index"] = int(inputs["chaser_index"])
        return refs

    for source_key in ("detection_path", "position_source_path"):
        value = inputs.get(source_key)
        if value not in (None, ""):
            refs[f"source_{source_key}"] = str(value)

    keypoint_parent = (
        "refined_keypoints_runs"
        if inputs.get("keypoint_variant") == "refined"
        else "keypoints_runs"
    )
    _set_path(
        "source_keypoint_path",
        keypoint_parent,
        inputs.get("keypoint_run"),
    )
    _set_path("source_crop_path", "crop_runs", inputs.get("crop_run"))
    _set_path(
        "source_tracking_path",
        "tracking_runs",
        inputs.get("source_tracking_run"),
    )
    _set_path(
        "source_arena_assignment_path",
        "arena_assignment_runs",
        inputs.get("source_arena_assignment_run"),
    )
    _set_path(
        "source_swim_bout_path",
        "analysis/swim_bout_runs",
        inputs.get("swim_bout_run"),
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


def ensure_track_kinematics_run_group(
    root: zarr.Group,
    run_name: Optional[str],
    *,
    run_type: str = "online",
    overwrite: bool = False,
) -> Tuple[str, zarr.Group]:
    """Create /analysis/track_kinematics_runs/<type>/<run_name>."""

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
            del type_parent[run_name]
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        prefix = (
            "track_kinematics"
            if run_type == "online"
            else "track_kinematics_offline"
        )
        run_name = f"{prefix}_{timestamp}"

    run_group = type_parent.create_group(
        run_name,
        attributes={"stage_selector_eligible": False},
    )
    try:
        mark_run_started(
            run_group,
            run_name=f"{run_type}/{run_name}",
            stage="track_kinematics",
        )
        if run_group.attrs.get("stage_selector_eligible") is not False:
            raise RuntimeError(
                "Track run did not persist fail-closed selector eligibility at "
                "creation."
            )
    except BaseException as exc:
        try:
            run_group.attrs["stage_selector_eligible"] = False
            mark_run_failed(
                run_group,
                parent_group=track_parent,
                run_name=f"{run_type}/{run_name}",
                error=str(exc),
            )
        except BaseException as rollback_exc:  # pragma: no cover - hostile store
            raise RuntimeError(
                "Track run creation failed and could not be left explicitly failed "
                "and selector-ineligible."
            ) from rollback_exc
        raise

    return run_name, run_group


def mark_track_kinematics_run_complete(
    root: zarr.Group,
    run_group: zarr.Group,
    *,
    run_name: str,
    run_type: str,
    validate_complete_run: Callable[[], Mapping[str, Any]],
) -> None:
    """Validate a complete ineligible run, prepare pointers, and expose it last."""

    track_parent = root["analysis"]["track_kinematics_runs"]
    type_parent = track_parent[run_type]
    qualified_name = f"{run_type}/{run_name}"
    expected_path = f"analysis/track_kinematics_runs/{qualified_name}"
    if str(run_group.path) != expected_path:
        raise ValueError(
            f"Track run path /{run_group.path} differs from /{expected_path}."
        )
    if run_group.attrs.get("stage_selector_eligible") is not False:
        raise ValueError(
            "Track completion requires literal stage_selector_eligible=false."
        )
    parent_snapshots = (
        (track_parent.attrs, copy.deepcopy(dict(track_parent.attrs))),
        (type_parent.attrs, copy.deepcopy(dict(type_parent.attrs))),
    )
    try:
        mark_run_complete(
            run_group,
            parent_group=track_parent,
            run_name=qualified_name,
            run_provenance=build_run_provenance_from_stage_record(
                run_group.attrs.get("provenance", {}),
                fallback_command="track_kinematics",
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
        validation = validate_complete_run()
        if not isinstance(validation, Mapping) or validation.get("valid") is not True:
            raise RuntimeError(
                "Complete track pre-selection validation did not report valid=true: "
                f"{validation!r}."
            )
        track_parent.attrs["latest_complete"] = qualified_name
        track_parent.attrs["latest"] = qualified_name
        type_parent.attrs["latest"] = run_name
        attr_key = "latest_online" if run_type == "online" else "latest_offline"
        track_parent.attrs[attr_key] = run_name
        # Persistent publication commit point: no fallible store mutation follows.
        run_group.attrs["stage_selector_eligible"] = True
    except BaseException as exc:
        rollback_errors: list[str] = []
        try:
            run_group.attrs["stage_selector_eligible"] = False
        except BaseException as rollback_exc:  # pragma: no cover - hostile store
            rollback_errors.append(f"disarm selector eligibility: {rollback_exc}")
        try:
            mark_run_failed(
                run_group,
                parent_group=track_parent,
                run_name=qualified_name,
                error=str(exc),
            )
        except BaseException as rollback_exc:  # pragma: no cover - hostile store
            rollback_errors.append(f"mark failed: {rollback_exc}")
        for attrs, snapshot in parent_snapshots:
            try:
                _restore_track_attrs(attrs, snapshot)
            except BaseException as rollback_exc:  # pragma: no cover - hostile store
                rollback_errors.append(f"restore parent attrs: {rollback_exc}")
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


def _smooth_acceleration_trace(acceleration_px: np.ndarray, window: int) -> np.ndarray:
    """Return a centered moving average of acceleration, ignoring NaNs."""

    acceleration = np.asarray(acceleration_px, dtype=np.float64)
    if window <= 1 or acceleration.size == 0:
        return acceleration.copy()

    kernel = np.ones(window, dtype=np.float64)
    val_mask = np.isfinite(acceleration).astype(np.float64)
    accel_values = np.nan_to_num(acceleration, nan=0.0, copy=True)
    sum_values = np.convolve(accel_values, kernel, mode="same")
    count_values = np.convolve(val_mask, kernel, mode="same")
    smoothed = np.full_like(acceleration, np.nan)
    valid = count_values > 0
    smoothed[valid] = sum_values[valid] / count_values[valid]
    return smoothed


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

    if pixel_to_mm is not None and np.isfinite(pixel_to_mm):
        acceleration_mm = acceleration_px * pixel_to_mm
    else:
        acceleration_mm = _nan_array(acceleration_px.shape, dtype=np.float64)

    post_window = max(1, int(round(fps * smooth_seconds)))
    smoothed_acceleration_px = _smooth_acceleration_trace(acceleration_px, post_window)
    if pixel_to_mm is not None and np.isfinite(pixel_to_mm):
        smoothed_acceleration_mm = smoothed_acceleration_px * pixel_to_mm
    else:
        smoothed_acceleration_mm = _nan_array(smoothed_acceleration_px.shape, dtype=np.float64)

    return {
        "acceleration_px": acceleration_px,
        "acceleration_mm": acceleration_mm,
        "smoothed_acceleration_px": smoothed_acceleration_px,
        "smoothed_acceleration_mm": smoothed_acceleration_mm,
        "derivative_method": "first_difference",
        "post_smoothing_method": "moving_average",
        "post_smoothing_alignment": "centered",
        "post_smoothing_window_frames": int(post_window),
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
        headings_track = headings_deg[mask]
        kp_success_track = keypoint_success[mask]
        detection_index = np.where(mask)[0]
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
        detection_indices_sorted = detection_index[order]

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

        speed_raw_px = speeds.speed_raw
        speed_filtered_px = speeds.speed_filtered
        speed_smoothed_px = speeds.speed_smoothed
        speed_averaged_px = speeds.speed_averaged
        frame_path_distance_raw_px = speeds.frame_path_distance_raw
        frame_path_distance_filtered_px = speeds.frame_path_distance_filtered
        frame_path_distance_smoothed_px = speeds.frame_path_distance_smoothed
        cumulative_path_px = speeds.cumulative_path_distance
        speed_per_second_px = speeds.speed_per_second
        delta_frames = speeds.delta_frames
        delta_seconds = speeds.delta_seconds
        transition_valid = speeds.transition_valid
        transition_reason_code = speeds.transition_reason_code

        if pixel_to_mm_val is not None:
            coords_mm = coords_px * pixel_to_mm_val
            speed_raw_mm = speed_raw_px * pixel_to_mm_val
            speed_filtered_mm = speed_filtered_px * pixel_to_mm_val
            speed_smoothed_mm = speed_smoothed_px * pixel_to_mm_val
            speed_averaged_mm = speed_averaged_px * pixel_to_mm_val
            frame_path_distance_raw_mm = frame_path_distance_raw_px * pixel_to_mm_val
            frame_path_distance_filtered_mm = frame_path_distance_filtered_px * pixel_to_mm_val
            frame_path_distance_smoothed_mm = frame_path_distance_smoothed_px * pixel_to_mm_val
            cumulative_path_mm = cumulative_path_px * pixel_to_mm_val
            speed_per_second_mm = speed_per_second_px * pixel_to_mm_val
        else:
            coords_mm = _nan_array(coords_px.shape)
            speed_raw_mm = _nan_array(speed_raw_px.shape)
            speed_filtered_mm = _nan_array(speed_filtered_px.shape)
            speed_smoothed_mm = _nan_array(speed_smoothed_px.shape)
            speed_averaged_mm = _nan_array(speed_averaged_px.shape)
            frame_path_distance_raw_mm = _nan_array(frame_path_distance_raw_px.shape)
            frame_path_distance_filtered_mm = _nan_array(frame_path_distance_filtered_px.shape)
            frame_path_distance_smoothed_mm = _nan_array(frame_path_distance_smoothed_px.shape)
            cumulative_path_mm = _nan_array(cumulative_path_px.shape)
            speed_per_second_mm = _nan_array(speed_per_second_px.shape)

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
        acceleration_px = np.asarray(default_speed_derivative["acceleration_px"], dtype=np.float64)
        accel_mm = np.asarray(default_speed_derivative["acceleration_mm"], dtype=np.float64)
        smoothed_accel_px = np.asarray(default_speed_derivative["smoothed_acceleration_px"], dtype=np.float64)
        smoothed_accel_mm = np.asarray(default_speed_derivative["smoothed_acceleration_mm"], dtype=np.float64)

        heading_window = max(1, int(round(fps * smooth_seconds)))
        if heading_window > 1 and heading_rad.size > 0:
            kernel = np.ones(heading_window, dtype=np.float64)
            valid_weights = np.convolve(heading_valid.astype(np.float64), kernel, mode="same")
            cos_vals = np.cos(np.where(heading_valid, heading_rad, 0.0))
            sin_vals = np.sin(np.where(heading_valid, heading_rad, 0.0))
            cos_sum = np.convolve(cos_vals, kernel, mode="same")
            sin_sum = np.convolve(sin_vals, kernel, mode="same")
            with np.errstate(invalid="ignore"):
                cos_mean = np.where(valid_weights > 0, cos_sum / valid_weights, np.nan)
                sin_mean = np.where(valid_weights > 0, sin_sum / valid_weights, np.nan)
            smoothed_heading_rad = np.arctan2(sin_mean, cos_mean)
        else:
            smoothed_heading_rad = heading_rad.copy()

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
            "detection_indices": detection_indices_sorted.astype(np.int64),
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
            "second_indices": seconds_per_frame,
            "speed_per_second_px": _float32(speed_per_second_px),
            "speed_per_second_mm": _float32(speed_per_second_mm),
            "heading_per_second_degrees": _float32(heading_per_second_deg),
            "heading_per_second_resultant": heading_per_second_resultant.astype(np.float32),
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
    physical_authority: BoundStimulusPhysicalCoordinateAuthority,
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
                "the exact selected-stimulus mm_per_pixel authority."
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
            f"{label} does not use the exact selected-stimulus mm_per_pixel authority."
        )


def _validate_track_summary_physical_fields(
    summary: Mapping[str, Any],
    *,
    physical_authority: BoundStimulusPhysicalCoordinateAuthority | None,
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
    value: BoundStimulusPhysicalCoordinateAuthority | None,
) -> dict[str, Any] | None:
    if value is None:
        return None
    authority = require_bound_stimulus_physical_coordinate_authority(value)
    physical = authority.physical_frame
    return {
        "stimulus_run": authority.stimulus_run,
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
    return {
        "relative_ref": relative_ref,
        "dtype": dtype.str,
        "dtype_fields": dtype_fields,
        "itemsize": int(dtype.itemsize),
        "shape": [int(item) for item in shape],
        "content_sha256": array_payload_sha256(node),
    }


def _build_track_staging_manifest(
    run_group: zarr.Group,
    *,
    ordered_ids: List[int],
    source_positions: BoundCanonicalCoordinateDescriptor,
    source_temporal_authority: Any,
    physical_authority: BoundStimulusPhysicalCoordinateAuthority | None,
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
    physical_frame: BoundPhysicalFrameCalibration | None = None,
    physical_authority: BoundStimulusPhysicalCoordinateAuthority | None = None,
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
            "outputs; supply a sealed selected-stimulus physical_authority."
        )
    if physical_authority is not None:
        physical_authority = (
            require_bound_stimulus_physical_coordinate_authority(
                physical_authority
            )
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
        subgroup.create_array("detection_indices", data=data["detection_indices"], chunks=base_chunk, overwrite=True)
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
    physical_authority: BoundStimulusPhysicalCoordinateAuthority | None,
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
) -> BoundStimulusPhysicalCoordinateAuthority | None:
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
    if type(record) is not dict or set(record) != {
        "stimulus_run",
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
    }:
        raise ValueError(
            "Track staging physical authority record is absent or not closed."
        )
    if reason != "NONE":
        raise ValueError(
            "A staged physical authority cannot carry an omission reason."
        )
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
    if _physical_authority_manifest_record(authority) != record:
        raise ValueError(
            "Selected stimulus physical authority, frame, calibration, scale, or "
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
    physical_authority: BoundStimulusPhysicalCoordinateAuthority | None,
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
    physical_authority: BoundStimulusPhysicalCoordinateAuthority | None,
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
    physical_authority: BoundStimulusPhysicalCoordinateAuthority | None,
) -> TrackPositionPublicationResult:
    time_lineage = load_bound_track_sample_time_lineage(
        subgroup,
        subgroup["track_sample_key"],
        subgroup["source_row_index"],
        subgroup["source_acquisition_frame_index"],
        subgroup["source_frame_interpolation"],
        subgroup["source_instance_key"],
        source_temporal_authority=source_temporal_authority,
    )
    identity = load_bound_row_identity_contract(
        subgroup,
        subgroup["track_sample_key"],
        track_time_lineage=time_lineage,
    )
    return load_track_position_coordinates(
        subgroup,
        subgroup["positions_px"],
        subgroup["source_row_index"],
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
    if not callable(group_keys):
        raise ValueError("Track run tracks node is not a persisted group.")
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
) -> BoundStimulusPhysicalCoordinateAuthority | None:
    expected = run_group.attrs.get("physical_coordinate_authority")
    if expected is None:
        return None
    if not isinstance(expected, Mapping):
        raise ValueError("Track run physical authority record is not an object.")
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
    parts = str(run_group.path).split("/")
    if (
        len(parts) != 4
        or parts[:2] != ["analysis", "track_kinematics_runs"]
        or parts[2] not in {"online", "offline"}
        or not parts[3]
    ):
        raise ValueError("Track run is not at one canonical typed run path.")
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


def _validate_direct_track_kinematics_run_before_selection(
    authoritative_root: zarr.Group,
    run_group: zarr.Group,
    *,
    run_name: str,
    run_type: str,
    source_positions: BoundCanonicalCoordinateDescriptor,
    source_temporal_authority: Any,
    physical_authority: BoundStimulusPhysicalCoordinateAuthority | None,
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
            time_lineage = stamp_track_sample_time_lineage(
                subgroup,
                subgroup["track_sample_key"],
                subgroup["source_row_index"],
                subgroup["source_acquisition_frame_index"],
                subgroup["source_frame_interpolation"],
                subgroup["source_instance_key"],
                source_temporal_authority=surface.temporal_authority,
            )
            key_values = np.array(
                subgroup["track_sample_key"][:],
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
                subgroup["track_sample_key"],
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

    _, legacy_calibration_info = resolve_calibration(root)

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
                texture_to_camera_scale = None

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

        if render_online and not use_refined_online:
            # Scalar texture/canvas ratios are not calibration or transform
            # authority. The exact native stimulus coordinates remain unchanged.
            texture_to_camera_scale = None

        if render_online:
            # Get heading from online fields if available, otherwise NaN
            if use_refined_online:
                # Refined data doesn't have heading, use NaN
                heading_online = np.full(frames_all.shape, np.nan, dtype=np.float64)
            else:
                assert raw_online_rowset is not None
                heading_node = raw_online_rowset.get("visual_angle_deg")
                if heading_node is not None:
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
                    ordered_track_ids = save_track_kinematics_tracks(
                        run_group,
                        tracks_online,
                        summaries_online,
                        source_temporal_authority=(
                            online_source_temporal_authority
                        ),
                        positions_px_source=online_positions_px_source,
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
                        # For refined online data, save the coordinate space and calibration used
                        saved_coordinate_space = coordinate_space
                        saved_pixel_to_mm = pixel_to_mm_online
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
                        saved_pixel_to_mm = None

                    # Canonical stage provenance.
                    online_params = {
                        "fps": fps,
                        "smoothing_seconds": args.smooth_seconds,
                        "smoothing_method": args.smoothing_method,
                        "smoothing_alignment": args.smoothing_alignment,
                        "savgol_polyorder": int(args.savgol_polyorder) if args.smoothing_method == "savitzky_golay" else None,
                        "coordinate_space": saved_coordinate_space,
                        "calibration_used": saved_pixel_to_mm,
                        "texture_to_camera_scale": texture_to_camera_scale,
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
                            "calibration": legacy_calibration_info,
                            "inputs": inputs,
                            "texture_to_camera_scale": texture_to_camera_scale,
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
                        validate_complete_run=lambda: (
                            _validate_direct_track_kinematics_run_before_selection(
                                output_root,
                                run_group,
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
        crop_group_offline = root[f"crop_runs/{keypoints_offline.crop_run}"]
        detection_path_offline = crop_group_offline.attrs.get("source_coords_path")
        crop_row_source_label = _crop_row_source_label(crop_group_offline.attrs)
        if detection_path_offline:
            detection_offline: Optional[DetectionResolution] = resolve_detection_from_path(
                root,
                detection_path_offline,
            )
            preferred_detection_offline = prefer_refined_detection(
                root,
                detection_offline,
                console,
            )
            detection_offline = preferred_detection_offline
        else:
            if (
                "frame_indices" not in crop_group_offline
                or (
                    "bbox_img_xyxy" not in crop_group_offline
                    and "bbox_norm_coords" not in crop_group_offline
                )
            ):
                raise ValueError(
                    f"Crop run '{keypoints_offline.crop_run}' missing 'source_coords_path'; cannot determine detection source."
                )
            if not crop_row_source_label:
                raise ValueError(
                    f"Crop run '{keypoints_offline.crop_run}' missing 'source_coords_path' "
                    "and source_detect_run/detection_source_type; cannot determine tracking lineage."
                )
            console.print(
                "[cyan]Using row-aligned crop metadata as offline position source:[/cyan] "
                f"crop_runs/{keypoints_offline.crop_run} "
                f"(source={crop_row_source_label})"
            )
            detection_offline = None

        position_source_offline = load_canonical_offline_position_source(
            root,
            crop_group_offline,
            crop_run_name=keypoints_offline.crop_run,
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
        (
            offline_physical_authority,
            offline_physical_calibration_info,
        ) = resolve_canonical_track_physical_authority(
            root,
            stimulus_run=args.stimulus_run,
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

        if detection_offline is not None:
            expected_detect_run = (
                detection_offline.source_detect_run or detection_offline.run_name
            )
        else:
            expected_detect_run = crop_row_source_label
        if not expected_detect_run:
            raise ValueError(
                "Offline: unable to determine source_detect_run for tracking lookup."
            )
        track_ids_offline, tracking_metadata = load_tracking_ids(
            root,
            frame_indices_offline.shape[0],
            expected_detect_run=expected_detect_run,
            expected_refined_run=(
                detection_offline.run_name
                if detection_offline is not None and detection_offline.is_refined
                else None
            ),
            expected_source_rowset_path=position_source_offline.path,
            expected_instance_key=position_source_offline.instance_key,
            expected_source_rowset_fingerprint=position_source_offline.rowset_fingerprint,
            return_metadata=True,
        )
        track_ids_offline = track_ids_offline.astype(np.int64, copy=False)
        track_id_to_arena_id = {
            int(track_id): int(arena_id)
            for track_id, arena_id in (
                tracking_metadata.get("track_id_to_arena_id", {}) or {}
            ).items()
        }

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
                        ordered_ids_offline = save_track_kinematics_tracks(
                            offline_group,
                            tracks_offline,
                            summaries_offline,
                            source_temporal_authority=(
                                offline_source_temporal_authority
                            ),
                            positions_px_source=canonical_position_surface.coordinates,
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
                            "detection_path": (
                                detection_offline.path
                                if detection_offline is not None
                                else None
                            ),
                            "detection_run": (
                                detection_offline.run_name
                                if detection_offline is not None
                                else crop_row_source_label
                            ),
                            "detection_variant": (
                                detection_offline.variant
                                if detection_offline is not None
                                else "crop_rows"
                            ),
                            "source_detect_run": (
                                detection_offline.source_detect_run
                                if detection_offline is not None
                                else crop_row_source_label
                            ),
                            "position_source_path": position_source_offline.path,
                            "position_source_kind": position_source_offline.kind,
                            "position_geometry_path": (
                                position_source_offline.geometry_path
                            ),
                            "keypoint_run": keypoints_offline.run_name,
                            "keypoint_variant": "refined" if keypoints_offline.is_refined else "raw",
                            "base_keypoint_run": keypoints_offline.base_run_name,
                            "keypoint_usability_dataset": keypoint_usability_dataset,
                            "crop_run": keypoints_offline.crop_run,
                            "source_tracking_run": tracking_metadata.get("track_run"),
                            "source_arena_assignment_run": tracking_metadata.get("source_arena_assignment_run"),
                            "source_tracking_rowset_fingerprint": tracking_metadata.get("source_rowset_fingerprint"),
                        }
                        if tracking_metadata:
                            offline_inputs["tracking_metadata"] = tracking_metadata
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
                            "coordinate_space": "camera",
                            "physical_calibration": (
                                offline_physical_calibration_info
                            ),
                            "hysteresis_enabled": not args.no_hysteresis,
                            "hysteresis_high_px": float(args.hysteresis_high_px) if not args.no_hysteresis else None,
                            "hysteresis_low_px": float(args.hysteresis_low_px) if not args.no_hysteresis else None,
                            "hysteresis_min_frames": int(args.hysteresis_min_frames) if not args.no_hysteresis else None,
                            "hysteresis_band_policy": args.hysteresis_band_policy if not args.no_hysteresis else None,
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
                                "physical_calibration": (
                                    offline_physical_calibration_info
                                ),
                                "hysteresis_enabled": not args.no_hysteresis,
                                "hysteresis_high_px": float(args.hysteresis_high_px) if not args.no_hysteresis else None,
                                "hysteresis_low_px": float(args.hysteresis_low_px) if not args.no_hysteresis else None,
                                "hysteresis_min_frames": int(args.hysteresis_min_frames) if not args.no_hysteresis else None,
                                "hysteresis_band_policy": args.hysteresis_band_policy if not args.no_hysteresis else None,
                                "source_tracking_run": tracking_metadata.get("track_run"),
                                "source_arena_assignment_run": tracking_metadata.get("source_arena_assignment_run"),
                                "source_tracking_rowset_fingerprint": tracking_metadata.get("source_rowset_fingerprint"),
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
                                validate_complete_run=lambda: (
                                    _validate_direct_track_kinematics_run_before_selection(
                                        output_root,
                                        offline_group,
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
    "BoundTrackPositionBindings",
    "TrackSpeeds",
    "_ordered_track_arena_ids",
    "bind_staged_offline_track_kinematics_run",
    "compute_track_speed",
    "find_fps",
    "_filter_public_track_rows",
    "load_arena_ids",
    "load_bound_track_position_bindings",
    "resolve_dimensions",
    "stage_offline_track_kinematics_run",
    "validate_bound_track_position_bindings",
    "main",
]


if __name__ == "__main__":  # pragma: no cover
    main()
