"""Immutable, selector-ineligible motion successors from explicit providers.

The legacy track-kinematics writer remains unchanged.  This module publishes a
compact canary successor under ``analysis/track_kinematics_runs/provider``
after exact position, body-frame, and tracking joins have already been sealed.
It never resolves or mutates a selector and it keeps linear and angular source
lineage separate in the run manifest.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
import json
import math
from pathlib import Path
import re
import shutil
from types import MappingProxyType
from typing import Any, Mapping, Sequence
import uuid

import numpy as np
import zarr

from fisheye.analysis.track_kinematics import (
    ANGULAR_SAMPLE_REASON_CODES,
    LINEAR_SAMPLE_REASON_CODES,
    TRACK_SAMPLE_VALIDITY_INDEPENDENT_PROFILE,
    TRANSITION_REASON_CODES,
)
from fisheye.analysis_workflows.position_body_frame_motion import (
    BoundTrackedProviderMotionInput,
)
from fisheye.analysis_workflows.tracking_source_handle import (
    TrackingSourceHandle,
    require_tracking_source_handle,
)
from fisheye.shared.atomic_run_publisher import (
    AtomicRunPublishSpec,
    atomic_publish_run_group,
)
from fisheye.shared.json_safety import json_attr_safe
from fisheye.shared.run_provenance import build_writer_run_provenance
from fisheye.shared.source_camera_physical_authority import (
    BoundSourceCameraPhysicalAuthority,
    load_source_camera_physical_authority,
    require_bound_source_camera_physical_authority,
)
from fisheye.shared.zarr.analysis_array_contracts import (
    AnalysisArrayDeclaration,
    AnalysisAuthorityRole,
)
from fisheye.shared.zarr.analysis_storage_planning import (
    AnalysisArrayStorageFacts,
    AnalysisStoragePlanReceipt,
    analysis_storage_plan_receipt_from_manifest,
    plan_analysis_storage,
)
from fisheye.shared.zarr.array_contracts import (
    ArrayContract,
    BOOL,
    FLOAT32,
    INT16,
    INT32,
    INT64,
    UINT16,
    UINT64,
)
from fisheye.shared.zarr.array_factory import (
    create_array_from_plan,
    validate_array_metadata_declaration_from_plan,
)
from fisheye.shared.zarr.benchmark_runtime import sha256_array
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.metadata_equivalence import (
    validate_direct_consolidated_subtree,
)
from fisheye.shared.zarr.storage_profiles import PUBLISHED_HTTP_V1, StorageProfile
from fisheye.shared.zarr_helpers import (
    consolidate_metadata_capture_expected_warnings,
)
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.shared.zarr_run_completion import (
    COMPLETION_EPOCH_REQUIRE_PROVENANCE,
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
    RUN_STATUS_RUNNING,
    mark_run_complete,
    mark_run_started,
    require_runs_parent,
)

PROVIDER_TRACK_MOTION_PARENT_PATH = "analysis/track_kinematics_runs/provider"
PROVIDER_TRACK_MOTION_SCHEMA_ID = "palette.provider_track_motion_run"
PROVIDER_TRACK_MOTION_SCHEMA_VERSION = 1
PROVIDER_TRACK_MOTION_MANIFEST_SCHEMA_ID = "palette.provider_track_motion_run_manifest"
PROVIDER_TRACK_MOTION_MANIFEST_SCHEMA_VERSION = 1
PROVIDER_TRACK_MOTION_MANIFEST_ATTR = "provider_track_motion_manifest"
PROVIDER_TRACK_MOTION_MANIFEST_DIGEST_ATTR = "provider_track_motion_manifest_sha256"
PROVIDER_TRACK_MOTION_STORAGE_PLAN_ATTR = "provider_track_motion_storage_plan"
PROVIDER_TRACK_MOTION_PUBLICATION_ATTEMPT_ATTR = (
    "provider_track_motion_publication_attempt_uuid"
)
PROVIDER_TRACK_MOTION_PUBLICATION_POLICY = (
    "provider_track_motion_atomic_nonpromoting_v1"
)
PROVIDER_TRACK_MOTION_RETRY_POLICY = "new_immutable_run_name_required"
PROVIDER_TRACK_MOTION_COMPUTATION_ID = "track_motion_provider_successor.v1"

_RUN_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_SELECTOR_ATTRS = (
    "latest",
    "latest_complete",
    "latest_pending",
    "latest_provider",
    "authoritative_run",
)

_PIXEL_SAMPLE_ARRAYS: tuple[str, ...] = (
    "track_sample_key",
    "source_acquisition_frame_index",
    "source_observation_instance_key",
    "source_provider_row_index",
    "source_position_row_index",
    "source_body_frame_row_index",
    "source_tracking_row_index",
    "time_seconds",
    "positions_px",
    "position_source_valid",
    "heading_degrees",
    "heading_radians",
    "body_frame_source_valid",
    "linear_sample_valid",
    "angular_sample_valid",
    "linear_sample_reason_code",
    "angular_sample_reason_code",
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
    "delta_heading_degrees",
    "angular_velocity_raw_deg_s",
    "angular_speed_raw_deg_s",
    "smoothed_heading_degrees",
    "smoothed_heading_radians",
    "delta_heading_smoothed_degrees",
    "angular_velocity_smoothed_deg_s",
    "angular_speed_smoothed_deg_s",
)

_PHYSICAL_SAMPLE_ARRAYS: tuple[str, ...] = (
    "positions_mm",
    "speed_raw_mm",
    "speed_filtered_mm",
    "speed_smoothed_mm",
    "speed_averaged_mm",
    "acceleration_mm",
    "smoothed_acceleration_mm",
    "frame_path_distance_raw_mm",
    "frame_path_distance_filtered_mm",
    "frame_path_distance_smoothed_mm",
    "cumulative_path_distance_mm",
)

_PIXEL_PER_SECOND_ARRAYS: tuple[str, ...] = (
    "per_second/track_second_key",
    "per_second/speed_px",
    "per_second/heading_degrees",
    "per_second/heading_resultant",
)

_PHYSICAL_PER_SECOND_ARRAYS: tuple[str, ...] = ("per_second/speed_mm",)

_REQUIRED_ARRAYS = (
    "track_ids",
    "track_row_offsets",
    *_PIXEL_SAMPLE_ARRAYS,
    *_PIXEL_PER_SECOND_ARRAYS,
)

_PHYSICAL_ARRAYS = (*_PHYSICAL_SAMPLE_ARRAYS, *_PHYSICAL_PER_SECOND_ARRAYS)
_ALL_ARRAYS = (*_REQUIRED_ARRAYS, *_PHYSICAL_ARRAYS)

_DTYPE_BY_PATH: dict[str, np.dtype[Any]] = {
    "track_ids": np.dtype("int64"),
    "track_row_offsets": np.dtype("int64"),
    "track_sample_key": np.dtype("int64"),
    "source_acquisition_frame_index": np.dtype("int64"),
    "source_observation_instance_key": np.dtype("uint64"),
    "source_provider_row_index": np.dtype("int64"),
    "source_position_row_index": np.dtype("int64"),
    "source_body_frame_row_index": np.dtype("int64"),
    "source_tracking_row_index": np.dtype("int64"),
    "time_seconds": np.dtype("float32"),
    "positions_px": np.dtype("float32"),
    "positions_mm": np.dtype("float32"),
    "position_source_valid": np.dtype(bool),
    "heading_degrees": np.dtype("float32"),
    "heading_radians": np.dtype("float32"),
    "body_frame_source_valid": np.dtype(bool),
    "linear_sample_valid": np.dtype(bool),
    "angular_sample_valid": np.dtype(bool),
    "linear_sample_reason_code": np.dtype("uint16"),
    "angular_sample_reason_code": np.dtype("uint16"),
    "delta_frames": np.dtype("int32"),
    "delta_seconds": np.dtype("float32"),
    "transition_valid": np.dtype(bool),
    "transition_reason_code": np.dtype("int16"),
    "speed_raw_px": np.dtype("float32"),
    "speed_filtered_px": np.dtype("float32"),
    "speed_smoothed_px": np.dtype("float32"),
    "speed_averaged_px": np.dtype("float32"),
    "speed_raw_mm": np.dtype("float32"),
    "speed_filtered_mm": np.dtype("float32"),
    "speed_smoothed_mm": np.dtype("float32"),
    "speed_averaged_mm": np.dtype("float32"),
    "acceleration_px": np.dtype("float32"),
    "smoothed_acceleration_px": np.dtype("float32"),
    "acceleration_mm": np.dtype("float32"),
    "smoothed_acceleration_mm": np.dtype("float32"),
    "frame_path_distance_raw_px": np.dtype("float32"),
    "frame_path_distance_filtered_px": np.dtype("float32"),
    "frame_path_distance_smoothed_px": np.dtype("float32"),
    "cumulative_path_distance_px": np.dtype("float32"),
    "frame_path_distance_raw_mm": np.dtype("float32"),
    "frame_path_distance_filtered_mm": np.dtype("float32"),
    "frame_path_distance_smoothed_mm": np.dtype("float32"),
    "cumulative_path_distance_mm": np.dtype("float32"),
    "delta_heading_degrees": np.dtype("float32"),
    "angular_velocity_raw_deg_s": np.dtype("float32"),
    "angular_speed_raw_deg_s": np.dtype("float32"),
    "smoothed_heading_degrees": np.dtype("float32"),
    "smoothed_heading_radians": np.dtype("float32"),
    "delta_heading_smoothed_degrees": np.dtype("float32"),
    "angular_velocity_smoothed_deg_s": np.dtype("float32"),
    "angular_speed_smoothed_deg_s": np.dtype("float32"),
    "per_second/track_second_key": np.dtype("int64"),
    "per_second/speed_px": np.dtype("float32"),
    "per_second/speed_mm": np.dtype("float32"),
    "per_second/heading_degrees": np.dtype("float32"),
    "per_second/heading_resultant": np.dtype("float32"),
}

_TRAILING_SHAPE: dict[str, tuple[int, ...]] = {
    "track_sample_key": (2,),
    "positions_px": (2,),
    "positions_mm": (2,),
    "per_second/track_second_key": (2,),
}


class ProviderTrackMotionError(ValueError):
    """Raised when a provider-motion successor cannot be sealed."""


def _canonical_record(value: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ProviderTrackMotionError(f"{name} must be one nonempty mapping.")
    try:
        encoded = json.dumps(
            json_attr_safe(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ProviderTrackMotionError(f"{name} must be strict JSON: {exc}") from exc
    result = json.loads(encoded)
    if not isinstance(result, dict):  # pragma: no cover - defensive
        raise ProviderTrackMotionError(f"{name} is not a JSON object.")
    return result


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _safe_run_name(value: object) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or _RUN_NAME_RE.fullmatch(value) is None
    ):
        raise ProviderTrackMotionError(f"Invalid provider-motion run name: {value!r}.")
    return value


def _selector_snapshot(parent: Any | None) -> dict[str, Any]:
    if parent is None:
        return {}
    attrs = getattr(parent, "attrs", {})
    return {
        name: json_attr_safe(attrs[name]) for name in _SELECTOR_ATTRS if name in attrs
    }


def _node(group: Any, path: str) -> Any:
    current = group
    for component in path.strip("/").split("/"):
        current = current[component]
    return current


def _concat(
    tracks: Mapping[int, Mapping[str, Any]],
    ordered_track_ids: Sequence[int],
    name: str,
) -> np.ndarray:
    dtype = _DTYPE_BY_PATH[name]
    trailing = _TRAILING_SHAPE.get(name, ())
    values = [np.asarray(tracks[track_id][name]) for track_id in ordered_track_ids]
    if not values:
        return np.empty((0, *trailing), dtype=dtype)
    for index, value in enumerate(values):
        if (
            value.dtype != dtype
            or value.ndim != 1 + len(trailing)
            or value.shape[1:] != trailing
        ):
            raise ProviderTrackMotionError(
                f"Track {ordered_track_ids[index]} array {name!r} has an invalid dtype or shape."
            )
    return np.ascontiguousarray(np.concatenate(values, axis=0), dtype=dtype)


def _flatten_motion(
    tracks: Mapping[int, Mapping[str, Any]],
    *,
    include_physical: bool,
) -> dict[str, np.ndarray]:
    ordered = tuple(sorted(int(track_id) for track_id in tracks))
    track_ids = np.asarray(ordered, dtype=np.int64)
    counts = [
        int(np.asarray(tracks[track_id]["track_sample_key"]).shape[0])
        for track_id in ordered
    ]
    offsets = np.zeros(len(ordered) + 1, dtype=np.int64)
    if counts:
        offsets[1:] = np.cumsum(np.asarray(counts, dtype=np.int64))
    arrays: dict[str, np.ndarray] = {
        "track_ids": track_ids,
        "track_row_offsets": offsets,
    }
    sample_arrays = (
        (*_PIXEL_SAMPLE_ARRAYS, *_PHYSICAL_SAMPLE_ARRAYS)
        if include_physical
        else _PIXEL_SAMPLE_ARRAYS
    )
    for name in sample_arrays:
        arrays[name] = _concat(tracks, ordered, name)

    second_keys: list[np.ndarray] = []
    second_values: dict[str, list[np.ndarray]] = {
        "per_second/speed_px": [],
        "per_second/heading_degrees": [],
        "per_second/heading_resultant": [],
    }
    if include_physical:
        second_values["per_second/speed_mm"] = []
    for track_id in ordered:
        data = tracks[track_id]
        seconds = np.asarray(data["second_indices"])
        if seconds.dtype != np.dtype("int64") or seconds.ndim != 1:
            raise ProviderTrackMotionError("second_indices must be exact int64[N].")
        second_keys.append(
            np.column_stack(
                (
                    np.full(seconds.shape, track_id, dtype=np.int64),
                    seconds,
                )
            )
        )
        for output_name, input_name in (
            ("per_second/speed_px", "speed_per_second_px"),
            ("per_second/heading_degrees", "heading_per_second_degrees"),
            ("per_second/heading_resultant", "heading_per_second_resultant"),
            *(
                (("per_second/speed_mm", "speed_per_second_mm"),)
                if include_physical
                else ()
            ),
        ):
            value = np.asarray(data[input_name])
            if value.dtype != np.dtype("float32") or value.shape != seconds.shape:
                raise ProviderTrackMotionError(
                    f"{input_name} must be exact float32 and align with second_indices."
                )
            second_values[output_name].append(value)
    arrays["per_second/track_second_key"] = (
        np.ascontiguousarray(np.concatenate(second_keys, axis=0), dtype=np.int64)
        if second_keys
        else np.empty((0, 2), dtype=np.int64)
    )
    for name, values in second_values.items():
        arrays[name] = (
            np.ascontiguousarray(np.concatenate(values), dtype=np.float32)
            if values
            else np.empty((0,), dtype=np.float32)
        )
    return arrays


def _validate_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    paths = set(arrays)
    if paths == set(_REQUIRED_ARRAYS):
        include_physical = False
    elif paths == set(_ALL_ARRAYS):
        include_physical = True
    else:
        raise ProviderTrackMotionError(
            "Provider-motion arrays do not match the exact pixel-only or "
            "calibrated v1 field set."
        )
    for name in arrays:
        value = np.asarray(arrays[name])
        trailing = _TRAILING_SHAPE.get(name, ())
        if (
            value.dtype != _DTYPE_BY_PATH[name]
            or value.ndim != 1 + len(trailing)
            or value.shape[1:] != trailing
        ):
            raise ProviderTrackMotionError(
                f"Provider-motion array {name!r} has an invalid dtype or shape."
            )
    track_ids = arrays["track_ids"]
    offsets = arrays["track_row_offsets"]
    keys = arrays["track_sample_key"]
    row_count = int(keys.shape[0])
    if (
        offsets.shape != (track_ids.shape[0] + 1,)
        or int(offsets[0]) != 0
        or int(offsets[-1]) != row_count
        or np.any(np.diff(offsets) < 0)
        or (track_ids.size and not np.all(np.diff(track_ids) > 0))
    ):
        raise ProviderTrackMotionError("Track IDs and row offsets are not canonical.")
    if not np.array_equal(keys[:, 1], arrays["source_acquisition_frame_index"]):
        raise ProviderTrackMotionError(
            "Track keys differ from acquisition-frame identity."
        )
    for index, track_id in enumerate(track_ids):
        start, stop = int(offsets[index]), int(offsets[index + 1])
        if np.any(keys[start:stop, 0] != track_id):
            raise ProviderTrackMotionError("Track-key segments differ from track_ids.")
    if np.unique(keys, axis=0).shape[0] != row_count:
        raise ProviderTrackMotionError(
            "Provider-motion track_sample_key is duplicated."
        )
    provider_rows = arrays["source_provider_row_index"]
    if not np.array_equal(np.sort(provider_rows), np.arange(row_count, dtype=np.int64)):
        raise ProviderTrackMotionError(
            "Provider-motion rows are not an exact permutation of the source provider rowset."
        )
    if np.unique(arrays["source_observation_instance_key"]).shape[0] != row_count:
        raise ProviderTrackMotionError("Source observation identity is duplicated.")
    position_valid = arrays["position_source_valid"]
    body_valid = arrays["body_frame_source_valid"]
    if np.any(arrays["linear_sample_valid"] & ~position_valid):
        raise ProviderTrackMotionError(
            "Linear validity exceeds position-source validity."
        )
    if np.any(arrays["angular_sample_valid"] & ~body_valid):
        raise ProviderTrackMotionError("Angular validity exceeds body-frame validity.")
    if np.any(position_valid & ~np.all(np.isfinite(arrays["positions_px"]), axis=1)):
        raise ProviderTrackMotionError("Valid position rows contain non-finite values.")
    if np.any(body_valid & ~np.isfinite(arrays["heading_degrees"])):
        raise ProviderTrackMotionError(
            "Valid body-frame rows contain non-finite headings."
        )
    second_keys = arrays["per_second/track_second_key"]
    second_count = int(second_keys.shape[0])
    if np.unique(second_keys, axis=0).shape[0] != second_count:
        raise ProviderTrackMotionError(
            "Provider-motion track_second_key is duplicated."
        )
    for name in _PIXEL_PER_SECOND_ARRAYS[1:]:
        if arrays[name].shape != (second_count,):
            raise ProviderTrackMotionError(
                f"Per-second array {name!r} is not row aligned."
            )
    if include_physical and arrays["per_second/speed_mm"].shape != (second_count,):
        raise ProviderTrackMotionError("Per-second physical speed is not row aligned.")


@dataclass(frozen=True)
class PreparedProviderTrackMotion:
    arrays: Mapping[str, np.ndarray]
    source_authority_record: Mapping[str, Any]
    source_authority_sha256: str
    tracked_input_record: Mapping[str, Any]
    tracked_input_sha256: str
    tracking_source: TrackingSourceHandle = dataclass_field(
        repr=False, compare=False
    )
    physical_authority_record: Mapping[str, Any] | None
    physical_authority_sha256: str | None
    computation_record: Mapping[str, Any]
    computation_sha256: str


def _load_provider_physical_authority(
    tracked: BoundTrackedProviderMotionInput,
    *,
    allow_pixel_only: bool,
) -> BoundSourceCameraPhysicalAuthority | None:
    root = open_zarr_root(
        tracked.source_authority.analysis_zarr_path,
        mode="r",
        use_consolidated=True,
    )
    try:
        physical = require_bound_source_camera_physical_authority(
            load_source_camera_physical_authority(root)
        )
    except (KeyError, TypeError, ValueError) as exc:
        if allow_pixel_only:
            return None
        raise ProviderTrackMotionError(
            "Provider motion requires the archive's typed source-camera physical "
            "authority; use allow_pixel_only=True only for an explicitly "
            "selector-ineligible pixel-domain canary."
        ) from exc
    source_frame_sha256 = physical.physical_frame.source_camera_pixels.record_sha256
    if source_frame_sha256 != tracked.source_authority.position_camera_frame_sha256:
        raise ProviderTrackMotionError(
            "Source-camera physical authority frame differs from the position "
            "provider's camera-frame authority."
        )
    return physical


def _validate_prepared_tracking_binding(
    prepared: PreparedProviderTrackMotion,
) -> None:
    tracking = require_tracking_source_handle(prepared.tracking_source)
    record = prepared.tracked_input_record
    tracking_record = record.get("tracking_source")
    expected = {
        "run_path": tracking.run_path,
        "manifest_sha256": tracking.manifest_sha256,
        "verification_digest": tracking.verification_digest,
        "instance_key_sha256": sha256_array(tracking.instance_key),
        "track_ids_sha256": sha256_array(tracking.track_ids),
    }
    if not isinstance(tracking_record, Mapping) or dict(tracking_record) != expected:
        raise ProviderTrackMotionError(
            "Prepared motion tracking authority differs from its live sealed run."
        )


def _physical_authority_record(
    physical: BoundSourceCameraPhysicalAuthority,
) -> dict[str, Any]:
    verified = require_bound_source_camera_physical_authority(physical)
    frame = verified.physical_frame
    return _canonical_record(
        {
            "schema_id": "palette.provider_track_motion_physical_authority",
            "schema_version": 1,
            "camera_id": verified.camera_id,
            "source_kind": verified.source_kind,
            "authority_manifest_ref": verified.manifest.record_ref,
            "authority_manifest_sha256": verified.manifest.record_sha256,
            "physical_frame_ref": frame.record_ref,
            "physical_frame_sha256": frame.record_sha256,
            "selected_camera_evidence_ref": (frame.selected_camera_evidence.record_ref),
            "selected_camera_evidence_sha256": (
                frame.selected_camera_evidence.record_sha256
            ),
            "source_camera_frame_ref": frame.source_camera_pixels.record_ref,
            "source_camera_frame_sha256": frame.source_camera_pixels.record_sha256,
            "mm_per_pixel": float(verified.mm_per_pixel),
            "derivation": "physical_array_equals_pixel_peer_times_mm_per_pixel_v1",
        },
        name="provider track-motion physical authority",
    )


def _validate_physical_array_pairs(
    arrays: Mapping[str, np.ndarray],
    *,
    mm_per_pixel: float,
) -> None:
    scale_value = float(mm_per_pixel)
    if not math.isfinite(scale_value) or scale_value <= 0:
        raise ProviderTrackMotionError("Physical authority has invalid mm_per_pixel.")
    for physical_path in _PHYSICAL_ARRAYS:
        pixel_path = (
            "per_second/speed_px"
            if physical_path == "per_second/speed_mm"
            else f"{physical_path[:-3]}_px"
        )
        physical = np.asarray(arrays[physical_path])
        pixel = np.asarray(arrays[pixel_path])
        if physical.dtype != pixel.dtype or physical.shape != pixel.shape:
            raise ProviderTrackMotionError(
                f"Physical array {physical_path!r} differs from its pixel peer."
            )
        scale = np.asarray(scale_value, dtype=pixel.dtype)
        with np.errstate(over="ignore", invalid="ignore"):
            expected = np.asarray(pixel * scale, dtype=pixel.dtype)
        if not np.array_equal(physical, expected, equal_nan=True):
            raise ProviderTrackMotionError(
                f"Physical array {physical_path!r} does not exactly use the "
                "bound mm_per_pixel authority."
            )


def _validate_prepared_physical_binding(
    prepared: PreparedProviderTrackMotion,
) -> None:
    has_physical_arrays = set(prepared.arrays) == set(_ALL_ARRAYS)
    has_physical_record = prepared.physical_authority_record is not None
    has_physical_digest = prepared.physical_authority_sha256 is not None
    if not (has_physical_arrays == has_physical_record == has_physical_digest):
        raise ProviderTrackMotionError(
            "Provider-motion physical arrays and authority binding are incomplete."
        )
    if not has_physical_record:
        return
    assert prepared.physical_authority_record is not None
    assert prepared.physical_authority_sha256 is not None
    record = _canonical_record(
        _thaw(prepared.physical_authority_record),
        name="provider track-motion physical authority",
    )
    if canonical_json_sha256(record) != prepared.physical_authority_sha256:
        raise ProviderTrackMotionError(
            "Provider-motion physical-authority digest is stale."
        )
    _validate_physical_array_pairs(
        prepared.arrays,
        mm_per_pixel=float(record["mm_per_pixel"]),
    )


def prepare_provider_track_motion(
    tracked: BoundTrackedProviderMotionInput,
    *,
    fps: float,
    smooth_seconds: float,
    hysteresis_high_px: float | None = None,
    hysteresis_low_px: float | None = None,
    hysteresis_min_frames: int | None = None,
    smoothing_method: str = "moving_average",
    smoothing_alignment: str = "centered",
    savgol_polyorder: int = 3,
    allow_pixel_only: bool = False,
) -> PreparedProviderTrackMotion:
    """Compute one immutable calibrated successor from a sealed provider join.

    The archive's typed source-camera physical authority is required by default.
    ``allow_pixel_only`` exists only for explicit selector-ineligible canaries.
    """

    if type(tracked) is not BoundTrackedProviderMotionInput:
        raise ProviderTrackMotionError("A sealed tracked provider input is required.")
    tracked.assert_verified()
    if (
        isinstance(fps, bool)
        or not isinstance(fps, (int, float))
        or not math.isfinite(float(fps))
        or float(fps) <= 0
    ):
        raise ProviderTrackMotionError("fps must be one positive finite number.")
    if (
        isinstance(smooth_seconds, bool)
        or not isinstance(smooth_seconds, (int, float))
        or not math.isfinite(float(smooth_seconds))
        or float(smooth_seconds) < 0
    ):
        raise ProviderTrackMotionError(
            "smooth_seconds must be one finite nonnegative number."
        )
    if type(allow_pixel_only) is not bool:
        raise ProviderTrackMotionError("allow_pixel_only must be the exact boolean.")
    physical = _load_provider_physical_authority(
        tracked,
        allow_pixel_only=allow_pixel_only,
    )
    parameters = {
        "fps": float(fps),
        "smooth_seconds": float(smooth_seconds),
        "pixel_to_mm": physical.mm_per_pixel if physical is not None else None,
        "hysteresis_high_px": hysteresis_high_px,
        "hysteresis_low_px": hysteresis_low_px,
        "hysteresis_min_frames": hysteresis_min_frames,
        "smoothing_method": str(smoothing_method),
        "smoothing_alignment": str(smoothing_alignment),
        "savgol_polyorder": int(savgol_polyorder),
    }
    tracks, _summaries = tracked.build_track_datasets(**parameters)
    arrays = _flatten_motion(tracks, include_physical=physical is not None)
    _validate_arrays(arrays)
    physical_record = (
        _physical_authority_record(physical) if physical is not None else None
    )
    physical_digest = (
        canonical_json_sha256(physical_record) if physical_record is not None else None
    )
    if physical is not None:
        _validate_physical_array_pairs(
            arrays,
            mm_per_pixel=physical.mm_per_pixel,
        )
    computation = _canonical_record(
        {
            "schema_id": "palette.provider_track_motion_computation",
            "schema_version": 1,
            "computation_id": PROVIDER_TRACK_MOTION_COMPUTATION_ID,
            "validity_profile": TRACK_SAMPLE_VALIDITY_INDEPENDENT_PROFILE,
            "linear_sample_reason_codes": dict(LINEAR_SAMPLE_REASON_CODES),
            "angular_sample_reason_codes": dict(ANGULAR_SAMPLE_REASON_CODES),
            "transition_reason_codes": dict(TRANSITION_REASON_CODES),
            "parameters": parameters,
            "physical_outputs": {
                "status": (
                    "bound_typed_source_camera_mm_v1"
                    if physical is not None
                    else "omitted_explicit_pixel_only_canary"
                ),
                "reason_code": (
                    "NONE"
                    if physical is not None
                    else "EXPLICIT_SELECTOR_INELIGIBLE_PIXEL_ONLY_CANARY"
                ),
                "physical_authority_sha256": physical_digest,
            },
            "implicit_fallback": "forbidden",
        },
        name="provider track-motion computation",
    )
    return PreparedProviderTrackMotion(
        arrays=MappingProxyType(
            {name: np.ascontiguousarray(value).copy() for name, value in arrays.items()}
        ),
        source_authority_record=_freeze(
            _thaw(tracked.source_authority.authority_record)
        ),
        source_authority_sha256=tracked.source_authority.authority_sha256,
        tracked_input_record=_freeze(_thaw(tracked.authority_record)),
        tracked_input_sha256=tracked.authority_sha256,
        tracking_source=tracked.tracking_source,
        physical_authority_record=(
            _freeze(physical_record) if physical_record is not None else None
        ),
        physical_authority_sha256=physical_digest,
        computation_record=_freeze(computation),
        computation_sha256=canonical_json_sha256(computation),
    )


def _array_contracts() -> dict[str, ArrayContract]:
    common = PROVIDER_TRACK_MOTION_SCHEMA_ID
    result: dict[str, ArrayContract] = {}
    for path in _ALL_ARRAYS:
        dtype = _DTYPE_BY_PATH[path]
        dtype_contract = {
            np.dtype(bool): BOOL,
            np.dtype("int16"): INT16,
            np.dtype("int32"): INT32,
            np.dtype("int64"): INT64,
            np.dtype("uint16"): UINT16,
            np.dtype("uint64"): UINT64,
            np.dtype("float32"): FLOAT32,
        }[dtype]
        if path == "track_ids":
            shape, axes = ("T",), ("track",)
        elif path == "track_row_offsets":
            shape, axes = ("O",), ("track_offset",)
        elif path.startswith("per_second/"):
            trailing = _TRAILING_SHAPE.get(path, ())
            shape = ("S", *trailing)
            axes = ("track_second",) + (("key_component",) if trailing else ())
        else:
            trailing = _TRAILING_SHAPE.get(path, ())
            shape = ("N", *trailing)
            axes = ("track_sample",) + (
                ("key_component",)
                if path == "track_sample_key"
                else ("xy",) if trailing else ()
            )
        units = None
        if path in {
            "speed_raw_px",
            "speed_filtered_px",
            "speed_smoothed_px",
            "speed_averaged_px",
            "per_second/speed_px",
        }:
            units = "px/s"
        elif path in {
            "speed_raw_mm",
            "speed_filtered_mm",
            "speed_smoothed_mm",
            "speed_averaged_mm",
            "per_second/speed_mm",
        }:
            units = "mm/s"
        elif path in {"acceleration_px", "smoothed_acceleration_px"}:
            units = "px/s^2"
        elif path in {"acceleration_mm", "smoothed_acceleration_mm"}:
            units = "mm/s^2"
        elif path in {
            "positions_px",
            "frame_path_distance_raw_px",
            "frame_path_distance_filtered_px",
            "frame_path_distance_smoothed_px",
            "cumulative_path_distance_px",
        }:
            units = "px"
        elif path in {
            "positions_mm",
            "frame_path_distance_raw_mm",
            "frame_path_distance_filtered_mm",
            "frame_path_distance_smoothed_mm",
            "cumulative_path_distance_mm",
        }:
            units = "mm"
        elif path.endswith("_deg_s"):
            units = "deg/s"
        elif "heading" in path and (path.endswith("degrees") or "_degrees" in path):
            units = "deg"
        elif path == "time_seconds" or path == "delta_seconds":
            units = "s"
        result[path] = ArrayContract(
            common,
            PROVIDER_TRACK_MOTION_SCHEMA_VERSION,
            dtype_contract,
            shape,
            axes,
            f"provider track-motion {path}",
            units=units,
            coordinate_space=(
                "source_camera_image_px.top_left_y_down.v1"
                if path == "positions_px"
                else (
                    "physical_mm.source_camera_y_down.v1"
                    if path == "positions_mm"
                    else None
                )
            ),
        )
    return result


def _authority_role(path: str) -> AnalysisAuthorityRole:
    if path in {
        "track_ids",
        "track_row_offsets",
        "track_sample_key",
        "source_acquisition_frame_index",
        "source_observation_instance_key",
        "source_provider_row_index",
        "source_position_row_index",
        "source_body_frame_row_index",
        "source_tracking_row_index",
        "per_second/track_second_key",
    }:
        return AnalysisAuthorityRole.LINEAGE_INDEX
    if path.endswith("_valid") or path.endswith("_reason_code"):
        return AnalysisAuthorityRole.QUALITY_DIAGNOSTIC
    return AnalysisAuthorityRole.SCIENTIFIC_AUTHORITY


def _storage_receipt(
    arrays: Mapping[str, np.ndarray], profile: StorageProfile
) -> AnalysisStoragePlanReceipt:
    contracts = _array_contracts()
    declarations = tuple(
        AnalysisArrayDeclaration(
            path=path,
            contract=contracts[path],
            required=True,
            access_pattern="per_row",
            write_mode="immutable",
            authority_role=_authority_role(path),
            fill_semantics="fully materialized immutable value",
            null_semantics="explicit validity and reason arrays govern missingness",
            physical_policy_owner=(
                "fisheye.analysis_workflows.materializers.provider_track_motion"
            ),
            byte_planner_adopted=True,
        )
        for path in sorted(arrays)
    )
    facts = {
        path: AnalysisArrayStorageFacts(
            path=path,
            shape=tuple(int(item) for item in value.shape),
            dtype=value.dtype,
            access_unit_semantics="one complete logical row with fixed trailing axes",
        )
        for path, value in arrays.items()
    }
    dimensions = {
        "N": int(arrays["track_sample_key"].shape[0]),
        "T": int(arrays["track_ids"].shape[0]),
        "O": int(arrays["track_row_offsets"].shape[0]),
        "S": int(arrays["per_second/track_second_key"].shape[0]),
    }
    return plan_analysis_storage(
        declarations,
        facts,
        profile=profile,
        dimensions=dimensions,
    )


@dataclass(frozen=True)
class ProviderTrackMotionRunPlan:
    source_zarr: Path
    run_name: str
    scratch_root: Path
    local_zarr: Path
    prepared: PreparedProviderTrackMotion
    storage_profile: StorageProfile
    storage_receipt: AnalysisStoragePlanReceipt
    parent_selector_attrs: Mapping[str, Any]
    publication_attempt_uuid: str
    run_provenance: Mapping[str, Any]

    @property
    def run_path(self) -> str:
        return f"{PROVIDER_TRACK_MOTION_PARENT_PATH}/{self.run_name}"

    @property
    def local_run_path(self) -> Path:
        return self.local_zarr.joinpath(*self.run_path.split("/"))

    @property
    def target_run_path(self) -> Path:
        return self.source_zarr.joinpath(*self.run_path.split("/"))

    @property
    def manifest_sha256(self) -> str:
        return provider_track_motion_manifest_digest(
            build_provider_track_motion_manifest(self, status=RUN_STATUS_COMPLETE)
        )


def plan_provider_track_motion_run(
    source_zarr: str | Path,
    prepared: PreparedProviderTrackMotion,
    *,
    run_name: str | None = None,
    scratch_root: str | Path,
    storage_profile: StorageProfile = PUBLISHED_HTTP_V1,
    publication_attempt_uuid: str | None = None,
) -> ProviderTrackMotionRunPlan:
    if not isinstance(prepared, PreparedProviderTrackMotion):
        raise TypeError("prepared must be PreparedProviderTrackMotion.")
    _validate_prepared_tracking_binding(prepared)
    _validate_arrays(prepared.arrays)
    _validate_prepared_physical_binding(prepared)
    source = Path(source_zarr).expanduser().resolve()
    if prepared.tracking_source.analysis_zarr_path != source:
        raise ProviderTrackMotionError(
            "Tracking authority and provider-motion destination must be the same archive."
        )
    scratch = Path(scratch_root).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Authoritative analysis Zarr does not exist: {source}")
    if scratch == source or scratch.is_relative_to(source):
        raise ProviderTrackMotionError("Scratch root must be outside the archive.")
    attempt = (
        str(uuid.UUID(publication_attempt_uuid))
        if publication_attempt_uuid
        else str(uuid.uuid4())
    )
    name = _safe_run_name(run_name or f"provider_motion_{uuid.UUID(attempt).hex}")
    local = scratch / f"{name}.zarr"
    target = source.joinpath(*f"{PROVIDER_TRACK_MOTION_PARENT_PATH}/{name}".split("/"))
    if local.exists() or target.exists():
        raise FileExistsError("Provider-motion publication name is already occupied.")
    root = open_zarr_root(source, mode="r", use_consolidated=False)
    parent = None
    try:
        parent = _node(root, PROVIDER_TRACK_MOTION_PARENT_PATH)
    except (KeyError, ValueError):
        pass
    receipt = _storage_receipt(prepared.arrays, storage_profile)
    provenance = build_writer_run_provenance(
        command="provider_track_motion_materializer",
        params={
            "source_authority_sha256": prepared.source_authority_sha256,
            "tracked_input_sha256": prepared.tracked_input_sha256,
            "physical_authority_sha256": prepared.physical_authority_sha256,
            "computation_sha256": prepared.computation_sha256,
            "storage_profile_id": storage_profile.profile_id,
        },
        input_run_ids={
            "position_body_frame_authority": prepared.source_authority_sha256,
            "tracked_provider_input": prepared.tracked_input_sha256,
            **(
                {"source_camera_physical_authority": prepared.physical_authority_sha256}
                if prepared.physical_authority_sha256 is not None
                else {}
            ),
        },
        cwd=source,
        include_system_context=False,
    )
    return ProviderTrackMotionRunPlan(
        source_zarr=source,
        run_name=name,
        scratch_root=scratch,
        local_zarr=local,
        prepared=prepared,
        storage_profile=storage_profile,
        storage_receipt=receipt,
        parent_selector_attrs=MappingProxyType(_selector_snapshot(parent)),
        publication_attempt_uuid=attempt,
        run_provenance=_freeze(provenance),
    )


def build_provider_track_motion_manifest(
    plan: ProviderTrackMotionRunPlan,
    *,
    status: str,
) -> dict[str, Any]:
    if status not in {RUN_STATUS_RUNNING, RUN_STATUS_COMPLETE}:
        raise ProviderTrackMotionError("Unsupported provider-motion lifecycle status.")
    _validate_arrays(plan.prepared.arrays)
    _validate_prepared_physical_binding(plan.prepared)
    array_records = [
        {
            "path": path,
            "dtype": np.dtype(value.dtype).str,
            "shape": list(value.shape),
            "sha256": sha256_array(value),
        }
        for path, value in sorted(plan.prepared.arrays.items())
    ]
    payload = {
        "namespace": PROVIDER_TRACK_MOTION_PARENT_PATH,
        "row_axis": "track_sample",
        "run_name": plan.run_name,
        "run_path": plan.run_path,
        "status": status,
        "stage_selector_eligible": False,
        "source_authority": {
            "record": _thaw(plan.prepared.source_authority_record),
            "sha256": plan.prepared.source_authority_sha256,
        },
        "tracked_input": {
            "record": _thaw(plan.prepared.tracked_input_record),
            "sha256": plan.prepared.tracked_input_sha256,
        },
        "physical_authority": (
            {
                "status": "bound",
                "record": _thaw(plan.prepared.physical_authority_record),
                "sha256": plan.prepared.physical_authority_sha256,
            }
            if plan.prepared.physical_authority_record is not None
            else {
                "status": "omitted_explicit_pixel_only_canary",
                "record": None,
                "sha256": None,
            }
        ),
        "computation": {
            "record": _thaw(plan.prepared.computation_record),
            "sha256": plan.prepared.computation_sha256,
        },
        "lineage_partition": {
            "linear": {
                "position_source": plan.prepared.source_authority_sha256,
                "validity_array": "linear_sample_valid",
                "reason_array": "linear_sample_reason_code",
            },
            "angular": {
                "body_frame_source": plan.prepared.source_authority_sha256,
                "validity_array": "angular_sample_valid",
                "reason_array": "angular_sample_reason_code",
            },
        },
        "arrays": array_records,
        "physical_storage_plan": plan.storage_receipt.as_manifest(),
        "publication": {
            "policy_id": PROVIDER_TRACK_MOTION_PUBLICATION_POLICY,
            "retry_policy": PROVIDER_TRACK_MOTION_RETRY_POLICY,
            "publication_attempt_uuid": plan.publication_attempt_uuid,
            "selector_activation": "forbidden",
            "parent_selector_mutation": "forbidden",
        },
    }
    return {
        "schema_id": PROVIDER_TRACK_MOTION_MANIFEST_SCHEMA_ID,
        "schema_version": PROVIDER_TRACK_MOTION_MANIFEST_SCHEMA_VERSION,
        "payload": payload,
        "payload_digest": canonical_json_sha256(payload),
    }


def provider_track_motion_manifest_digest(manifest: Mapping[str, Any]) -> str:
    if not isinstance(manifest, Mapping) or set(manifest) != {
        "schema_id",
        "schema_version",
        "payload",
        "payload_digest",
    }:
        raise ProviderTrackMotionError(
            "Provider-motion manifest envelope is not exact."
        )
    if (
        manifest["schema_id"] != PROVIDER_TRACK_MOTION_MANIFEST_SCHEMA_ID
        or manifest["schema_version"] != PROVIDER_TRACK_MOTION_MANIFEST_SCHEMA_VERSION
        or not isinstance(manifest["payload"], Mapping)
        or manifest["payload_digest"] != canonical_json_sha256(manifest["payload"])
    ):
        raise ProviderTrackMotionError(
            "Provider-motion manifest identity or digest is invalid."
        )
    return str(manifest["payload_digest"])


def _validate_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_run_name: str,
    expected_status: str,
) -> tuple[Mapping[str, Any], AnalysisStoragePlanReceipt]:
    provider_track_motion_manifest_digest(manifest)
    payload = manifest["payload"]
    expected_fields = {
        "namespace",
        "row_axis",
        "run_name",
        "run_path",
        "status",
        "stage_selector_eligible",
        "source_authority",
        "tracked_input",
        "physical_authority",
        "computation",
        "lineage_partition",
        "arrays",
        "physical_storage_plan",
        "publication",
    }
    if set(payload) != expected_fields:
        raise ProviderTrackMotionError(
            "Provider-motion manifest field set is not exact."
        )
    if (
        payload["namespace"] != PROVIDER_TRACK_MOTION_PARENT_PATH
        or payload["row_axis"] != "track_sample"
        or payload["run_name"] != expected_run_name
        or payload["run_path"]
        != f"{PROVIDER_TRACK_MOTION_PARENT_PATH}/{expected_run_name}"
        or payload["status"] != expected_status
        or payload["stage_selector_eligible"] is not False
    ):
        raise ProviderTrackMotionError(
            "Provider-motion manifest identity or lifecycle is invalid."
        )
    for name in ("source_authority", "tracked_input", "computation"):
        binding = payload[name]
        if not isinstance(binding, Mapping) or set(binding) != {"record", "sha256"}:
            raise ProviderTrackMotionError(
                f"Provider-motion {name} binding is not exact."
            )
        if canonical_json_sha256(binding["record"]) != binding["sha256"]:
            raise ProviderTrackMotionError(f"Provider-motion {name} binding is stale.")
    physical = payload["physical_authority"]
    if not isinstance(physical, Mapping) or set(physical) != {
        "status",
        "record",
        "sha256",
    }:
        raise ProviderTrackMotionError(
            "Provider-motion physical-authority binding is not exact."
        )
    if physical["status"] == "bound":
        if (
            not isinstance(physical["record"], Mapping)
            or canonical_json_sha256(physical["record"]) != physical["sha256"]
        ):
            raise ProviderTrackMotionError(
                "Provider-motion physical-authority binding is stale."
            )
    elif physical != {
        "status": "omitted_explicit_pixel_only_canary",
        "record": None,
        "sha256": None,
    }:
        raise ProviderTrackMotionError(
            "Provider-motion physical-authority omission is not explicit."
        )
    if (
        payload["computation"]["record"].get("validity_profile")
        != TRACK_SAMPLE_VALIDITY_INDEPENDENT_PROFILE
    ):
        raise ProviderTrackMotionError(
            "Provider-motion validity profile is not independent."
        )
    physical_outputs = payload["computation"]["record"].get("physical_outputs")
    expected_physical_outputs = (
        {
            "status": "bound_typed_source_camera_mm_v1",
            "reason_code": "NONE",
            "physical_authority_sha256": physical["sha256"],
        }
        if physical["status"] == "bound"
        else {
            "status": "omitted_explicit_pixel_only_canary",
            "reason_code": "EXPLICIT_SELECTOR_INELIGIBLE_PIXEL_ONLY_CANARY",
            "physical_authority_sha256": None,
        }
    )
    if physical_outputs != expected_physical_outputs:
        raise ProviderTrackMotionError(
            "Provider-motion physical-output computation record is stale."
        )
    if payload["lineage_partition"] != {
        "linear": {
            "position_source": payload["source_authority"]["sha256"],
            "validity_array": "linear_sample_valid",
            "reason_array": "linear_sample_reason_code",
        },
        "angular": {
            "body_frame_source": payload["source_authority"]["sha256"],
            "validity_array": "angular_sample_valid",
            "reason_array": "angular_sample_reason_code",
        },
    }:
        raise ProviderTrackMotionError(
            "Provider-motion linear/angular lineage partition is stale."
        )
    array_records = payload["arrays"]
    expected_arrays = (
        sorted(_ALL_ARRAYS)
        if physical["status"] == "bound"
        else sorted(_REQUIRED_ARRAYS)
    )
    if (
        not isinstance(array_records, list)
        or [item.get("path") for item in array_records] != expected_arrays
    ):
        raise ProviderTrackMotionError(
            "Provider-motion array manifest is not exact and sorted."
        )
    receipt = analysis_storage_plan_receipt_from_manifest(
        payload["physical_storage_plan"]
    )
    if [entry.declaration.path for entry in receipt.entries] != expected_arrays:
        raise ProviderTrackMotionError(
            "Provider-motion storage plan differs from its arrays."
        )
    for record, entry in zip(array_records, receipt.entries, strict=True):
        if set(record) != {"path", "dtype", "shape", "sha256"}:
            raise ProviderTrackMotionError(
                "Provider-motion array declaration is not exact."
            )
        if (
            record["dtype"] != np.dtype(entry.facts.dtype).str
            or tuple(record["shape"]) != entry.facts.shape
        ):
            raise ProviderTrackMotionError(
                "Provider-motion array declaration differs from storage plan."
            )
    publication = payload["publication"]
    if (
        not isinstance(publication, Mapping)
        or publication.get("policy_id") != PROVIDER_TRACK_MOTION_PUBLICATION_POLICY
        or publication.get("retry_policy") != PROVIDER_TRACK_MOTION_RETRY_POLICY
        or publication.get("selector_activation") != "forbidden"
        or publication.get("parent_selector_mutation") != "forbidden"
    ):
        raise ProviderTrackMotionError("Provider-motion publication policy is invalid.")
    str(uuid.UUID(str(publication["publication_attempt_uuid"])))
    return payload, receipt


def _fill_value(path: str) -> Any:
    dtype = _DTYPE_BY_PATH[path]
    if dtype.kind == "f":
        # Logical NaNs are written explicitly.  A finite physical fill keeps
        # the exact Zarr metadata declaration JSON-comparable during direct /
        # consolidated publication validation.
        return np.float32(0.0)
    if dtype == np.dtype(bool):
        return False
    return dtype.type(0)


def _write_arrays(run: Any, plan: ProviderTrackMotionRunPlan) -> None:
    entries = {entry.declaration.path: entry for entry in plan.storage_receipt.entries}
    for path, values in sorted(plan.prepared.arrays.items()):
        parent_path, _, leaf = path.rpartition("/")
        parent = run.require_group(parent_path) if parent_path else run
        entry = entries[path]
        contract = entry.declaration.contract
        semantic_attributes = {
            **({"units": contract.units} if contract.units is not None else {}),
            **(
                {"coordinate_space": contract.coordinate_space}
                if contract.coordinate_space is not None
                else {}
            ),
            "authority_role": entry.declaration.authority_role.value,
        }
        destination = create_array_from_plan(
            parent,
            name=leaf or path,
            contract=contract,
            plan=entry.plan,
            fill_value=_fill_value(path),
            attributes=semantic_attributes,
        )
        if values.size:
            destination[...] = values


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads((path / "zarr.json").read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ProviderTrackMotionError(f"Zarr declaration at {path} is invalid.")
    return value


def _validate_run_group(
    run: Any,
    run_path: Path,
    *,
    expected_run_name: str,
    expected_status: str,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    manifest = run.attrs.get(PROVIDER_TRACK_MOTION_MANIFEST_ATTR)
    payload, receipt = _validate_manifest(
        manifest,
        expected_run_name=expected_run_name,
        expected_status=expected_status,
    )
    digest = provider_track_motion_manifest_digest(manifest)
    if expected_manifest_sha256 is not None and digest != expected_manifest_sha256:
        raise ProviderTrackMotionError(
            "Provider-motion manifest differs from the plan."
        )
    if (
        run.attrs.get(RUN_COMPLETION_STATUS_ATTR) != expected_status
        or run.attrs.get("stage_selector_eligible") is not False
        or run.attrs.get(PROVIDER_TRACK_MOTION_MANIFEST_DIGEST_ATTR) != digest
    ):
        raise ProviderTrackMotionError(
            "Provider-motion run lifecycle attrs are invalid."
        )
    arrays: dict[str, np.ndarray] = {}
    records = {record["path"]: record for record in payload["arrays"]}
    for entry in receipt.entries:
        path = entry.declaration.path
        node = _node(run, path)
        if not isinstance(node, zarr.Array):
            raise ProviderTrackMotionError(
                f"Provider-motion array {path!r} is missing."
            )
        value = np.asarray(node[:])
        if sha256_array(value) != records[path]["sha256"]:
            raise ProviderTrackMotionError(f"Provider-motion array {path!r} is stale.")
        contract = entry.declaration.contract
        expected_semantics = {
            **({"units": contract.units} if contract.units is not None else {}),
            **(
                {"coordinate_space": contract.coordinate_space}
                if contract.coordinate_space is not None
                else {}
            ),
            "authority_role": entry.declaration.authority_role.value,
        }
        observed_semantics = {
            name: node.attrs[name]
            for name in ("units", "coordinate_space", "authority_role")
            if name in node.attrs
        }
        if observed_semantics != expected_semantics:
            raise ProviderTrackMotionError(
                f"Provider-motion array {path!r} semantic metadata is stale."
            )
        errors = validate_array_metadata_declaration_from_plan(
            _read_json(run_path.joinpath(*path.split("/"))),
            contract=entry.declaration.contract,
            plan=entry.plan,
            fill_value=_fill_value(path),
        )
        if errors:
            raise ProviderTrackMotionError(
                f"Provider-motion metadata failed for {path!r}: {errors!r}."
            )
        arrays[path] = value
    _validate_arrays(arrays)
    physical_binding = payload["physical_authority"]
    if physical_binding["status"] == "bound":
        _validate_physical_array_pairs(
            arrays,
            mm_per_pixel=float(physical_binding["record"]["mm_per_pixel"]),
        )
    return {
        "valid": True,
        "run_path": payload["run_path"],
        "status": expected_status,
        "row_count": int(arrays["track_sample_key"].shape[0]),
        "track_count": int(arrays["track_ids"].shape[0]),
        "manifest_sha256": digest,
    }


def validate_provider_track_motion_run(
    analysis_zarr: str | Path,
    run_path: str,
    *,
    use_consolidated: bool = True,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    archive = Path(analysis_zarr).expanduser().resolve()
    expected_prefix = f"{PROVIDER_TRACK_MOTION_PARENT_PATH}/"
    if (
        not run_path.startswith(expected_prefix)
        or "/" in run_path[len(expected_prefix) :]
    ):
        raise ProviderTrackMotionError(
            "run_path must name one exact provider-motion run."
        )
    root = open_zarr_root(archive, mode="r", use_consolidated=use_consolidated)
    run = _node(root, run_path)
    return _validate_run_group(
        run,
        archive.joinpath(*run_path.split("/")),
        expected_run_name=run_path.rsplit("/", 1)[1],
        expected_status=RUN_STATUS_COMPLETE,
        expected_manifest_sha256=expected_manifest_sha256,
    )


def _materialize_local(plan: ProviderTrackMotionRunPlan) -> dict[str, Any]:
    _validate_prepared_tracking_binding(plan.prepared)
    if plan.local_zarr.exists():
        raise FileExistsError(
            f"Local provider-motion Zarr already exists: {plan.local_zarr}"
        )
    plan.local_zarr.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(
        str(plan.local_zarr), mode="w-", zarr_format=3, use_consolidated=False
    )
    parent = require_runs_parent(
        root.require_group("analysis"),
        "track_kinematics_runs/provider",
        completion_epoch=COMPLETION_EPOCH_REQUIRE_PROVENANCE,
    )
    run = parent.create_group(plan.run_name)
    mark_run_started(run, run_name=plan.run_name, stage="provider_track_motion")
    run.attrs.update(
        {
            "schema_id": PROVIDER_TRACK_MOTION_SCHEMA_ID,
            "schema_version": PROVIDER_TRACK_MOTION_SCHEMA_VERSION,
            "stage_selector_eligible": False,
            "run_provenance": json_attr_safe(_thaw(plan.run_provenance)),
            PROVIDER_TRACK_MOTION_PUBLICATION_ATTEMPT_ATTR: plan.publication_attempt_uuid,
        }
    )
    _write_arrays(run, plan)
    complete_manifest = build_provider_track_motion_manifest(
        plan, status=RUN_STATUS_COMPLETE
    )
    mark_run_complete(
        run,
        parent_group=None,
        run_name=plan.run_name,
        run_provenance=_thaw(plan.run_provenance),
    )
    run.attrs[PROVIDER_TRACK_MOTION_MANIFEST_ATTR] = json_attr_safe(complete_manifest)
    run.attrs[PROVIDER_TRACK_MOTION_MANIFEST_DIGEST_ATTR] = (
        provider_track_motion_manifest_digest(complete_manifest)
    )
    run.attrs[PROVIDER_TRACK_MOTION_STORAGE_PLAN_ATTR] = json_attr_safe(
        plan.storage_receipt.as_manifest()
    )
    run.attrs["stage_selector_eligible"] = False
    consolidate_metadata_capture_expected_warnings(plan.local_zarr)
    validated = _validate_run_group(
        open_zarr_root(plan.local_zarr, mode="r", use_consolidated=True)[plan.run_path],
        plan.local_run_path,
        expected_run_name=plan.run_name,
        expected_status=RUN_STATUS_COMPLETE,
        expected_manifest_sha256=plan.manifest_sha256,
    )
    return {"local_zarr": str(plan.local_zarr), "validation": validated}


def publish_provider_track_motion_run(
    plan: ProviderTrackMotionRunPlan,
    *,
    copy_backend: str = "python",
    keep_scratch: bool = True,
) -> dict[str, Any]:
    """Publish a complete provider successor without changing selectors."""

    local = _materialize_local(plan)
    _validate_prepared_tracking_binding(plan.prepared)
    acceptance: dict[str, Any] = {}

    def validate(path: Path) -> Mapping[str, Any]:
        return _validate_run_group(
            open_zarr_root(path, mode="r", use_consolidated=False),
            path,
            expected_run_name=plan.run_name,
            expected_status=RUN_STATUS_COMPLETE,
            expected_manifest_sha256=plan.manifest_sha256,
        )

    def prepare(root: Any) -> tuple[Any]:
        return (
            require_runs_parent(
                root.require_group("analysis"),
                "track_kinematics_runs/provider",
                completion_epoch=COMPLETION_EPOCH_REQUIRE_PROVENANCE,
            ),
        )

    def complete(_root: Any, _parent: Any, run: Any) -> None:
        _validate_prepared_tracking_binding(plan.prepared)
        mark_run_complete(
            run,
            parent_group=None,
            run_name=plan.run_name,
            run_provenance=run.attrs.get("run_provenance"),
        )
        run.attrs["stage_selector_eligible"] = False

    def verify(root: Any) -> None:
        parent = _node(root, PROVIDER_TRACK_MOTION_PARENT_PATH)
        if _selector_snapshot(parent) != dict(plan.parent_selector_attrs):
            raise RuntimeError("Provider-motion publication changed parent selectors.")
        _validate_run_group(
            parent[plan.run_name],
            plan.target_run_path,
            expected_run_name=plan.run_name,
            expected_status=RUN_STATUS_COMPLETE,
            expected_manifest_sha256=plan.manifest_sha256,
        )

    def finalize(_root: Any, _parent: Any, _run: Any) -> None:
        validate_provider_track_motion_run(
            plan.source_zarr,
            plan.run_path,
            use_consolidated=False,
            expected_manifest_sha256=plan.manifest_sha256,
        )
        consolidate_metadata_capture_expected_warnings(plan.source_zarr)
        metadata = validate_direct_consolidated_subtree(
            plan.source_zarr, subtree_path=plan.run_path
        )
        consolidated = validate_provider_track_motion_run(
            plan.source_zarr,
            plan.run_path,
            use_consolidated=True,
            expected_manifest_sha256=plan.manifest_sha256,
        )
        acceptance.update(
            direct_consolidated=metadata.to_json(),
            consolidated_validation=consolidated,
        )

    publication = atomic_publish_run_group(
        AtomicRunPublishSpec(
            source_zarr=plan.source_zarr,
            local_run_path=plan.local_run_path,
            target_run_path=plan.target_run_path,
            run_name=plan.run_name,
            lock_suffix="provider-track-motion",
            publish_schema_id=PROVIDER_TRACK_MOTION_SCHEMA_ID,
            policy=PROVIDER_TRACK_MOTION_PUBLICATION_POLICY,
            rollback_policy=(
                "retain_failed_tombstone_leave_parent_selectors_untouched"
            ),
            content_checksum=True,
            publication_attempt_uuid=plan.publication_attempt_uuid,
        ),
        copy_backend=copy_backend,
        validate_run=validate,
        prepare_parents=prepare,
        complete_run=complete,
        verify_pointers=verify,
        activate_run=finalize,
        repair_failed_publication_visibility=(
            lambda _path: consolidate_metadata_capture_expected_warnings(
                plan.source_zarr
            )
        ),
        accept_persisted_activation_on_callback_error=False,
        payload_metadata={
            "manifest_sha256": plan.manifest_sha256,
            "source_authority_sha256": plan.prepared.source_authority_sha256,
            "tracked_input_sha256": plan.prepared.tracked_input_sha256,
            "selector_ineligible": True,
        },
    )
    result = {
        "plan": {
            "run_path": plan.run_path,
            "manifest_sha256": plan.manifest_sha256,
            "source_zarr": str(plan.source_zarr),
            "scratch_root": str(plan.scratch_root),
        },
        "local": local,
        "publication": publication,
        "acceptance": acceptance,
    }
    if not keep_scratch and plan.local_zarr.exists():
        shutil.rmtree(plan.local_zarr)
    return result


__all__ = [
    "PROVIDER_TRACK_MOTION_MANIFEST_ATTR",
    "PROVIDER_TRACK_MOTION_PARENT_PATH",
    "PROVIDER_TRACK_MOTION_SCHEMA_ID",
    "PROVIDER_TRACK_MOTION_SCHEMA_VERSION",
    "PreparedProviderTrackMotion",
    "ProviderTrackMotionError",
    "ProviderTrackMotionRunPlan",
    "build_provider_track_motion_manifest",
    "plan_provider_track_motion_run",
    "prepare_provider_track_motion",
    "provider_track_motion_manifest_digest",
    "publish_provider_track_motion_run",
    "validate_provider_track_motion_run",
]
