"""Exact composition of subject-position and body-frame motion inputs.

This module joins two independently published scientific providers.  It never
resolves a selector, chooses a fallback modality, or treats equal row counts as
lineage.  The resulting authority keeps linear and angular inputs independent
and can be projected onto one explicitly identified tracking rowset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from fisheye.analysis.track_kinematics import (
    TRACK_SAMPLE_VALIDITY_INDEPENDENT_PROFILE,
    build_track_datasets,
)
from fisheye.analysis_workflows.body_frame_source_handle import (
    BodyFrameSourceHandle,
    load_body_frame_source_handle,
    require_body_frame_source_handle,
)
from fisheye.analysis_workflows.subject_position_source_handle import (
    SubjectPositionSourceHandle,
    load_subject_position_source_handle,
    require_subject_position_source_handle,
)
from fisheye.analysis_workflows.tracking_source_handle import (
    TrackingSourceHandle,
    require_tracking_source_handle,
)
from fisheye.shared.subject_position_expression import (
    DETECTION_BBOX_CENTROID_ESTIMATOR_ID,
)
from fisheye.shared.traditional_heading_compatibility import (
    TraditionalHeadingCompatibilityReceipt,
)
from fisheye.shared.zarr.benchmark_runtime import sha256_array
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256

POSITION_BODY_FRAME_AUTHORITY_SCHEMA_ID = (
    "palette.position_body_frame_motion_source_authority"
)
POSITION_BODY_FRAME_AUTHORITY_SCHEMA_VERSION = 1
TRACKED_PROVIDER_MOTION_INPUT_SCHEMA_ID = (
    "palette.tracked_position_body_frame_motion_input"
)
TRACKED_PROVIDER_MOTION_INPUT_SCHEMA_VERSION = 2

EXPLICIT_POSITION_BODY_FRAME_PROFILE_ID = "explicit_position_body_frame.v1"
DETECTION_CENTROID_TRADITIONAL_V3_COMPATIBILITY_PROFILE_ID = (
    "detection_centroid_traditional_v3_heading_compatibility.v1"
)

_AUTHORITY_SEAL = object()
_TRACKED_INPUT_SEAL = object()
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class PositionBodyFrameMotionError(ValueError):
    """Raised when independently published providers cannot be composed."""


def _require_sha256(value: object, *, name: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise PositionBodyFrameMotionError(
            f"{name} must be one lowercase SHA-256 digest."
        )
    return value


def _readonly(value: Any, *, dtype: np.dtype[Any] | None = None) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _canonical_record(value: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise PositionBodyFrameMotionError(f"{name} must be one nonempty mapping.")
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise PositionBodyFrameMotionError(
            f"{name} must be strict JSON: {exc}"
        ) from exc
    result = json.loads(encoded)
    if not isinstance(result, dict):  # pragma: no cover - defensive
        raise PositionBodyFrameMotionError(f"{name} is not a JSON object.")
    return result


def _exact_key_join(
    left: np.ndarray,
    right: np.ndarray,
    *,
    left_name: str,
    right_name: str,
) -> tuple[np.ndarray, str]:
    """Return right-row indices for left order after exact key-set proof."""

    if (
        left.dtype != np.dtype("uint64")
        or right.dtype != np.dtype("uint64")
        or left.ndim != 1
        or right.ndim != 1
    ):
        raise PositionBodyFrameMotionError(
            f"{left_name} and {right_name} must be exact uint64[N] identities."
        )
    if np.unique(left).shape[0] != left.shape[0]:
        raise PositionBodyFrameMotionError(f"{left_name} contains duplicate keys.")
    if np.unique(right).shape[0] != right.shape[0]:
        raise PositionBodyFrameMotionError(f"{right_name} contains duplicate keys.")
    if left.shape == right.shape and np.array_equal(left, right):
        rows = np.arange(left.shape[0], dtype=np.int64)
        rows.setflags(write=False)
        return rows, "exact_ordered_instance_key_equality_v1"
    if left.shape != right.shape:
        raise PositionBodyFrameMotionError(
            f"{left_name} and {right_name} cardinalities differ."
        )
    right_order = np.argsort(right, kind="stable")
    left_order = np.argsort(left, kind="stable")
    if not np.array_equal(right[right_order], left[left_order]):
        raise PositionBodyFrameMotionError(
            f"{left_name} and {right_name} identify different rowsets."
        )
    rows = np.empty(left.shape[0], dtype=np.int64)
    rows[left_order] = right_order.astype(np.int64, copy=False)
    rows.setflags(write=False)
    return rows, "exact_instance_key_set_reorder_v1"


def _validate_position_coordinate_authority(
    position: SubjectPositionSourceHandle,
) -> str:
    record = position.coordinate_record
    descriptor = record.get("coordinate_descriptor")
    surface = record.get("coordinate_surface_contract")
    if not isinstance(descriptor, Mapping) or not isinstance(surface, Mapping):
        raise PositionBodyFrameMotionError(
            "Position provider lacks its exact coordinate descriptor and surface contract."
        )
    expected_descriptor = {
        "profile_id": "source_camera_image_px.top_left_y_down.v1",
        "space_id": "source_camera_image_px",
        "geometry_type": "point_xy",
        "components": ("x", "y"),
        "component_units": ("px", "px"),
        "origin": "top_left",
        "pixel_convention": "continuous",
    }
    for name, expected in expected_descriptor.items():
        actual = descriptor.get(name)
        if isinstance(expected, tuple):
            actual = tuple(actual) if isinstance(actual, (list, tuple)) else actual
        if actual != expected:
            raise PositionBodyFrameMotionError(
                f"Position coordinate descriptor differs at {name!r}."
            )
    if descriptor.get("positive_directions") != {"x": "right", "y": "down"}:
        raise PositionBodyFrameMotionError(
            "Position coordinate descriptor is not top-left, +X-right, +Y-down."
        )
    if (
        surface.get("surface_id") != "source_camera_point_xy_v1"
        or surface.get("source_camera_mapping")
        != "direct_source_camera_continuous_pixels"
    ):
        raise PositionBodyFrameMotionError(
            "Position provider is not a direct source-camera point surface."
        )
    frame_record = descriptor.get("frame_record")
    if not isinstance(frame_record, Mapping):
        raise PositionBodyFrameMotionError(
            "Position coordinate descriptor lacks source-camera frame authority."
        )
    return _require_sha256(
        frame_record.get("record_sha256"), name="position camera-frame digest"
    )


def _validate_body_frame_coordinate_semantics(
    body_frame: BodyFrameSourceHandle,
) -> Mapping[str, Any]:
    payload = body_frame.run_manifest.get("payload")
    recipe = payload.get("heading_recipe") if isinstance(payload, Mapping) else None
    if not isinstance(recipe, Mapping):
        raise PositionBodyFrameMotionError(
            "Body-frame manifest lacks its exact heading recipe."
        )
    required = {
        "coordinate_source": "keypoints_img_source_camera_pixels",
        "origin": "midpoint_eye_left_eye_right",
        "forward_axis": "unit_vector_swim_bladder_to_eye_midpoint",
        "left_axis": "fixed_clockwise_90_degrees_in_camera_xy",
        "axis_handedness": "determinant_negative_one_camera_xy",
        "heading_deg": "atan2_negative_forward_y_forward_x_degrees_float32",
        "invalid_geometry": "all_nan_and_axis_valid_false",
    }
    for name, expected in required.items():
        if recipe.get(name) != expected:
            raise PositionBodyFrameMotionError(
                f"Body-frame coordinate semantics differ at {name!r}."
            )
    return recipe


@dataclass(frozen=True, init=False)
class PositionBodyFrameMotionAuthority:
    """Sealed independent position and body-frame inputs in position-row order."""

    analysis_zarr_path: Path
    profile_id: str
    position_run_path: str
    position_manifest_sha256: str
    body_frame_run_path: str
    body_frame_manifest_sha256: str
    position_camera_frame_sha256: str
    row_alignment_mode: str
    instance_key: np.ndarray = field(repr=False, compare=False)
    source_acquisition_frame_index: np.ndarray = field(repr=False, compare=False)
    source_position_row_index: np.ndarray = field(repr=False, compare=False)
    body_frame_row_index: np.ndarray = field(repr=False, compare=False)
    position_xy: np.ndarray = field(repr=False, compare=False)
    position_valid: np.ndarray = field(repr=False, compare=False)
    heading_deg: np.ndarray = field(repr=False, compare=False)
    heading_valid: np.ndarray = field(repr=False, compare=False)
    authority_record: Mapping[str, Any] = field(repr=False)
    authority_sha256: str
    _seal: object = field(repr=False, compare=False)

    def __init__(
        self, *, _verification_seal: object | None = None, **values: Any
    ) -> None:
        if _verification_seal is not _AUTHORITY_SEAL:
            raise PositionBodyFrameMotionError(
                "Position/body-frame authorities must be minted by the strict composer."
            )
        for name, value in values.items():
            if name in {
                "instance_key",
                "source_acquisition_frame_index",
                "source_position_row_index",
                "body_frame_row_index",
                "position_xy",
                "position_valid",
                "heading_deg",
                "heading_valid",
            }:
                value = _readonly(value)
            elif name == "authority_record":
                value = _freeze(value)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_seal", _verification_seal)

    @property
    def row_count(self) -> int:
        return int(self.instance_key.shape[0])

    def assert_verified(self) -> None:
        if self._seal is not _AUTHORITY_SEAL:
            raise PositionBodyFrameMotionError(
                "Position/body-frame authority verification seal is absent."
            )
        if canonical_json_sha256(_thaw(self.authority_record)) != self.authority_sha256:
            raise PositionBodyFrameMotionError(
                "Position/body-frame authority record digest is stale."
            )


def compose_position_body_frame_motion_authority(
    position: SubjectPositionSourceHandle,
    body_frame: BodyFrameSourceHandle,
    *,
    profile_id: str = EXPLICIT_POSITION_BODY_FRAME_PROFILE_ID,
) -> PositionBodyFrameMotionAuthority:
    """Revalidate and compose two explicitly named provider publications."""

    position = require_subject_position_source_handle(position)
    body_frame = require_body_frame_source_handle(body_frame)
    if profile_id != EXPLICIT_POSITION_BODY_FRAME_PROFILE_ID:
        raise PositionBodyFrameMotionError(
            "Generic composition requires the explicit position/body-frame profile."
        )
    if position.analysis_zarr_path != body_frame.analysis_zarr_path:
        raise PositionBodyFrameMotionError(
            "Position and body-frame providers belong to different analysis archives."
        )
    # Reopen both immutable publications immediately before composition.  This
    # closes the time-of-check/time-of-use gap and rejects stale consolidated
    # metadata rather than silently falling back to direct traversal.
    position = load_subject_position_source_handle(
        position.analysis_zarr_path,
        position.run_path,
        expected_selector_eligible=position.selector_eligible,
        expected_manifest_sha256=position.manifest_sha256,
    )
    current_body_frame = load_body_frame_source_handle(
        body_frame.analysis_zarr_path,
        run_path=body_frame.run_path,
        expected_selector_eligible=body_frame.selector_eligible,
    )
    if current_body_frame.verification_digest != body_frame.verification_digest:
        raise PositionBodyFrameMotionError(
            "Body-frame publication changed after its handle was sealed."
        )
    body_frame = current_body_frame
    if position.selector_eligible is not body_frame.selector_eligible:
        raise PositionBodyFrameMotionError(
            "Position and body-frame eligibility dispositions differ."
        )

    camera_frame_sha256 = _validate_position_coordinate_authority(position)
    _validate_body_frame_coordinate_semantics(body_frame)

    position_keys = _readonly(position.instance_key[...], dtype=np.dtype("uint64"))
    body_keys = _readonly(body_frame.instance_key, dtype=np.dtype("uint64"))
    body_rows, row_mode = _exact_key_join(
        position_keys,
        body_keys,
        left_name="position instance_key",
        right_name="body-frame instance_key",
    )
    position_frames = _readonly(
        position.source_acquisition_frame_index[...], dtype=np.dtype("int64")
    )
    body_frames = _readonly(body_frame.frame_indices, dtype=np.dtype("int64"))[
        body_rows
    ]
    if not np.array_equal(position_frames, body_frames):
        raise PositionBodyFrameMotionError(
            "Position and body-frame acquisition-frame identities disagree after the exact key join."
        )
    source_rows = _readonly(position.source_row_index[...], dtype=np.dtype("int64"))
    if source_rows.shape != position_keys.shape:
        raise PositionBodyFrameMotionError(
            "Position source_row_index is not row aligned."
        )
    position_xy = _readonly(position.position_xy[...], dtype=np.dtype("float32"))
    position_valid = _readonly(position.valid[...], dtype=np.dtype(bool))
    heading_deg = _readonly(
        body_frame.heading_deg[body_rows], dtype=np.dtype("float32")
    )
    heading_valid = _readonly(body_frame.axis_valid[body_rows], dtype=np.dtype(bool))

    authority_record = _canonical_record(
        {
            "schema_id": POSITION_BODY_FRAME_AUTHORITY_SCHEMA_ID,
            "schema_version": POSITION_BODY_FRAME_AUTHORITY_SCHEMA_VERSION,
            "profile_id": profile_id,
            "analysis_zarr_path": str(position.analysis_zarr_path),
            "eligibility_disposition": {
                "stage_selector_eligible": position.selector_eligible,
                "selection": "explicit_named_sources_only",
                "fallback": "none",
            },
            "position_source": {
                "run_path": position.run_path,
                "manifest_sha256": position.manifest_sha256,
                "decoded_content_sha256": position.decoded_content_sha256,
                "estimator_id": position.estimator_record["estimator_id"],
                "estimator_sha256": position.estimator_sha256,
                "policy_sha256": position.policy_sha256,
                "source_sha256": position.source_sha256,
                "coordinate_sha256": position.coordinate_sha256,
                "camera_frame_sha256": camera_frame_sha256,
            },
            "body_frame_source": {
                "run_path": body_frame.run_path,
                "manifest_sha256": body_frame.run_manifest["payload_digest"],
                "verification_sha256": body_frame.verification_digest,
                "source_run_path": body_frame.source_run_path,
                "source_manifest_sha256": body_frame.source_manifest_digest,
                "source_skeleton_id": body_frame.source_skeleton_id,
                "source_skeleton_sha256": body_frame.source_skeleton_digest,
                "recipe_id": body_frame.recipe_id,
                "recipe_sha256": body_frame.recipe_digest,
                "heading_computation_sha256": body_frame.heading_computation_digest,
            },
            "row_alignment": {
                "mode": row_mode,
                "row_count": int(position_keys.shape[0]),
                "position_instance_key_sha256": sha256_array(position_keys),
                "body_frame_instance_key_sha256": sha256_array(body_keys),
                "body_frame_row_index_sha256": sha256_array(body_rows),
                "source_acquisition_frame_index_sha256": sha256_array(position_frames),
            },
            "linear_lineage": {
                "position_xy_sha256": sha256_array(position_xy),
                "position_valid_sha256": sha256_array(position_valid),
            },
            "angular_lineage": {
                "heading_deg_sha256": sha256_array(heading_deg),
                "heading_valid_sha256": sha256_array(heading_valid),
            },
        },
        name="position/body-frame authority",
    )
    authority_sha256 = canonical_json_sha256(authority_record)
    return PositionBodyFrameMotionAuthority(
        analysis_zarr_path=position.analysis_zarr_path,
        profile_id=profile_id,
        position_run_path=position.run_path,
        position_manifest_sha256=position.manifest_sha256,
        body_frame_run_path=body_frame.run_path,
        body_frame_manifest_sha256=str(body_frame.run_manifest["payload_digest"]),
        position_camera_frame_sha256=camera_frame_sha256,
        row_alignment_mode=row_mode,
        instance_key=position_keys,
        source_acquisition_frame_index=position_frames,
        source_position_row_index=source_rows,
        body_frame_row_index=body_rows,
        position_xy=position_xy,
        position_valid=position_valid,
        heading_deg=heading_deg,
        heading_valid=heading_valid,
        authority_record=authority_record,
        authority_sha256=authority_sha256,
        _verification_seal=_AUTHORITY_SEAL,
    )


def compose_detection_centroid_traditional_v3_compatibility_authority(
    position: SubjectPositionSourceHandle,
    body_frame: BodyFrameSourceHandle,
    *,
    compatibility: TraditionalHeadingCompatibilityReceipt,
) -> PositionBodyFrameMotionAuthority:
    """Expose the historical mixed modality only through a named receipt."""

    compatibility.validated_heading_computation()
    position = require_subject_position_source_handle(position)
    body_frame = require_body_frame_source_handle(body_frame)
    if (
        position.estimator_record.get("estimator_id")
        != DETECTION_BBOX_CENTROID_ESTIMATOR_ID
    ):
        raise PositionBodyFrameMotionError(
            "The compatibility profile requires detection_bbox_centroid.v1."
        )
    if (
        body_frame.source_skeleton_id != compatibility.skeleton_id
        or body_frame.source_skeleton_digest != compatibility.schema_sha256
        or body_frame.heading_computation_digest != compatibility.heading_sha256
    ):
        raise PositionBodyFrameMotionError(
            "Traditional-v3 compatibility receipt differs from the body-frame source."
        )
    composed = compose_position_body_frame_motion_authority(position, body_frame)
    record = _thaw(composed.authority_record)
    record["profile_id"] = DETECTION_CENTROID_TRADITIONAL_V3_COMPATIBILITY_PROFILE_ID
    record["compatibility_receipt"] = compatibility.as_dict()
    record["compatibility_receipt_sha256"] = compatibility.receipt_sha256
    record = _canonical_record(record, name="traditional-v3 compatibility authority")
    return PositionBodyFrameMotionAuthority(
        analysis_zarr_path=composed.analysis_zarr_path,
        profile_id=DETECTION_CENTROID_TRADITIONAL_V3_COMPATIBILITY_PROFILE_ID,
        position_run_path=composed.position_run_path,
        position_manifest_sha256=composed.position_manifest_sha256,
        body_frame_run_path=composed.body_frame_run_path,
        body_frame_manifest_sha256=composed.body_frame_manifest_sha256,
        position_camera_frame_sha256=composed.position_camera_frame_sha256,
        row_alignment_mode=composed.row_alignment_mode,
        instance_key=composed.instance_key,
        source_acquisition_frame_index=composed.source_acquisition_frame_index,
        source_position_row_index=composed.source_position_row_index,
        body_frame_row_index=composed.body_frame_row_index,
        position_xy=composed.position_xy,
        position_valid=composed.position_valid,
        heading_deg=composed.heading_deg,
        heading_valid=composed.heading_valid,
        authority_record=record,
        authority_sha256=canonical_json_sha256(record),
        _verification_seal=_AUTHORITY_SEAL,
    )


@dataclass(frozen=True, init=False)
class BoundTrackedProviderMotionInput:
    """Exact tracking projection of one position/body-frame authority."""

    source_authority: PositionBodyFrameMotionAuthority
    tracking_source: TrackingSourceHandle = field(repr=False, compare=False)
    tracking_run_path: str
    tracking_manifest_sha256: str
    tracking_row_alignment_mode: str
    tracking_row_index: np.ndarray = field(repr=False, compare=False)
    track_ids: np.ndarray = field(repr=False, compare=False)
    authority_record: Mapping[str, Any] = field(repr=False)
    authority_sha256: str
    _seal: object = field(repr=False, compare=False)

    def __init__(
        self, *, _verification_seal: object | None = None, **values: Any
    ) -> None:
        if _verification_seal is not _TRACKED_INPUT_SEAL:
            raise PositionBodyFrameMotionError(
                "Tracked provider inputs must be minted by the exact tracking join."
            )
        for name, value in values.items():
            if name in {"tracking_row_index", "track_ids"}:
                value = _readonly(value)
            elif name == "authority_record":
                value = _freeze(value)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_seal", _verification_seal)

    def assert_verified(self) -> None:
        """Reject a stale or caller-constructed tracking projection."""

        self.source_authority.assert_verified()
        require_tracking_source_handle(self.tracking_source)
        if self._seal is not _TRACKED_INPUT_SEAL:
            raise PositionBodyFrameMotionError(
                "Tracked provider input verification seal is absent."
            )
        if canonical_json_sha256(_thaw(self.authority_record)) != self.authority_sha256:
            raise PositionBodyFrameMotionError(
                "Tracked provider input authority record digest is stale."
            )

    def build_track_datasets(self, **parameters: Any):  # type: ignore[no-untyped-def]
        """Build numerical successors with independent validity arrays."""

        self.assert_verified()
        forbidden = {
            "track_ids",
            "frames",
            "positions_px",
            "headings_deg",
            "keypoint_success",
            "detection_source",
            "position_valid",
            "heading_valid",
            "validity_profile",
        }.intersection(parameters)
        if forbidden:
            raise PositionBodyFrameMotionError(
                f"Provider-bound track inputs cannot be overridden: {sorted(forbidden)!r}."
            )
        tracks, summaries = build_track_datasets(
            track_ids=self.track_ids,
            frames=self.source_authority.source_acquisition_frame_index,
            positions_px=self.source_authority.position_xy,
            headings_deg=self.source_authority.heading_deg,
            keypoint_success=self.source_authority.heading_valid,
            detection_source=None,
            position_valid=self.source_authority.position_valid,
            heading_valid=self.source_authority.heading_valid,
            validity_profile=TRACK_SAMPLE_VALIDITY_INDEPENDENT_PROFILE,
            **parameters,
        )
        # The legacy numerical builder groups and stably orders each track by
        # acquisition frame.  Reproduce that exact permutation here so the
        # provider successor retains every observation-to-provider/tracking
        # join instead of relying on equal cardinality or positional memory.
        source_rows = np.arange(self.source_authority.row_count, dtype=np.int64)
        for track_id, data in tracks.items():
            selected = source_rows[self.track_ids == track_id]
            order = np.argsort(
                self.source_authority.source_acquisition_frame_index[selected],
                kind="stable",
            )
            selected = selected[order]
            if not np.array_equal(
                data["source_acquisition_frame_index"],
                self.source_authority.source_acquisition_frame_index[selected],
            ):
                raise PositionBodyFrameMotionError(
                    "Numerical track ordering differs from the sealed provider rows."
                )
            data.update(
                {
                    "source_observation_instance_key": _readonly(
                        self.source_authority.instance_key[selected],
                        dtype=np.dtype("uint64"),
                    ),
                    "source_provider_row_index": _readonly(
                        selected, dtype=np.dtype("int64")
                    ),
                    "source_position_row_index": _readonly(
                        self.source_authority.source_position_row_index[selected],
                        dtype=np.dtype("int64"),
                    ),
                    "source_body_frame_row_index": _readonly(
                        self.source_authority.body_frame_row_index[selected],
                        dtype=np.dtype("int64"),
                    ),
                    "source_tracking_row_index": _readonly(
                        self.tracking_row_index[selected],
                        dtype=np.dtype("int64"),
                    ),
                    "position_source_valid": _readonly(
                        self.source_authority.position_valid[selected],
                        dtype=np.dtype(bool),
                    ),
                    "body_frame_source_valid": _readonly(
                        self.source_authority.heading_valid[selected],
                        dtype=np.dtype(bool),
                    ),
                }
            )
        return tracks, summaries


def bind_position_body_frame_to_tracking(
    source_authority: PositionBodyFrameMotionAuthority,
    tracking_source: TrackingSourceHandle,
) -> BoundTrackedProviderMotionInput:
    """Join one explicit tracking rowset by exact instance-key identity."""

    if type(source_authority) is not PositionBodyFrameMotionAuthority:
        raise PositionBodyFrameMotionError(
            "A verified PositionBodyFrameMotionAuthority is required."
    )
    source_authority.assert_verified()
    tracking = require_tracking_source_handle(tracking_source)
    if tracking.analysis_zarr_path != source_authority.analysis_zarr_path:
        raise PositionBodyFrameMotionError(
            "Tracking and provider authorities must belong to the same analysis archive."
        )
    tracking_keys = tracking.instance_key
    raw_track_ids = tracking.track_ids
    tracking_rows, alignment_mode = _exact_key_join(
        source_authority.instance_key,
        tracking_keys,
        left_name="position instance_key",
        right_name="tracking instance_key",
    )
    aligned_track_ids = _readonly(raw_track_ids[tracking_rows], dtype=np.dtype("int64"))
    record = _canonical_record(
        {
            "schema_id": TRACKED_PROVIDER_MOTION_INPUT_SCHEMA_ID,
            "schema_version": TRACKED_PROVIDER_MOTION_INPUT_SCHEMA_VERSION,
            "source_authority_sha256": source_authority.authority_sha256,
            "tracking_source": {
                "run_path": tracking.run_path,
                "manifest_sha256": tracking.manifest_sha256,
                "verification_digest": tracking.verification_digest,
                "instance_key_sha256": sha256_array(tracking_keys),
                "track_ids_sha256": sha256_array(raw_track_ids),
            },
            "row_alignment": {
                "mode": alignment_mode,
                "tracking_row_index_sha256": sha256_array(tracking_rows),
                "aligned_track_ids_sha256": sha256_array(aligned_track_ids),
            },
            "linear_lineage": {
                "source_authority_sha256": source_authority.authority_sha256,
                "position_manifest_sha256": source_authority.position_manifest_sha256,
            },
            "angular_lineage": {
                "source_authority_sha256": source_authority.authority_sha256,
                "body_frame_manifest_sha256": source_authority.body_frame_manifest_sha256,
            },
        },
        name="tracked provider motion input authority",
    )
    return BoundTrackedProviderMotionInput(
        source_authority=source_authority,
        tracking_source=tracking,
        tracking_run_path=tracking.run_path,
        tracking_manifest_sha256=tracking.manifest_sha256,
        tracking_row_alignment_mode=alignment_mode,
        tracking_row_index=tracking_rows,
        track_ids=aligned_track_ids,
        authority_record=record,
        authority_sha256=canonical_json_sha256(record),
        _verification_seal=_TRACKED_INPUT_SEAL,
    )


__all__ = [
    "DETECTION_CENTROID_TRADITIONAL_V3_COMPATIBILITY_PROFILE_ID",
    "EXPLICIT_POSITION_BODY_FRAME_PROFILE_ID",
    "POSITION_BODY_FRAME_AUTHORITY_SCHEMA_ID",
    "POSITION_BODY_FRAME_AUTHORITY_SCHEMA_VERSION",
    "TRACKED_PROVIDER_MOTION_INPUT_SCHEMA_ID",
    "TRACKED_PROVIDER_MOTION_INPUT_SCHEMA_VERSION",
    "BoundTrackedProviderMotionInput",
    "PositionBodyFrameMotionAuthority",
    "PositionBodyFrameMotionError",
    "bind_position_body_frame_to_tracking",
    "compose_detection_centroid_traditional_v3_compatibility_authority",
    "compose_position_body_frame_motion_authority",
]
