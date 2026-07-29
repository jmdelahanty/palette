"""Deterministic keypoint-derived producer for body-frame v1 snapshots."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Mapping

import numpy as np

from fisheye.shared.zarr.benchmark_runtime import sha256_array
from fisheye.shared.zarr.body_frame_schema import (
    BODY_FRAME_SCHEMA_V1,
    BodyFrameDimensions,
)
from fisheye.shared.zarr.keypoint_schema import (
    KEYPOINT_SCHEMA_V2,
    REFINED_KEYPOINT_SCHEMA_V2,
    KeypointDimensions,
    derive_frame_row_offsets,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.pose.heading import resolve_heading_computation
from fisheye.pose.schema import (
    resolve_keypoint_labels_from_attrs,
    resolve_required_keypoint_indices,
)


BODY_FRAME_RECIPE_SCHEMA_ID = "palette.body_frame.heading_recipe"
BODY_FRAME_RECIPE_SCHEMA_VERSION = 1
KEYPOINT_HEAD_AXIS_RECIPE_ID = "keypoint_eye_midpoint_head_axis_camera_xy_v1"
KEYPOINT_HEADING_COMPUTATION_SOURCE = "pose_schema.metadata.heading_computation"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_STAGES = {"keypoints", "refined_keypoints"}
_HEADING_DEPENDENCIES = ("swim_bladder", "eye_left", "eye_right")


def _require_sha256(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not _SHA256.fullmatch(normalized):
        raise ValueError(f"{name} must be lowercase hexadecimal SHA-256.")
    return normalized


def _array_values(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    try:
        return np.asarray(value[...])
    except (IndexError, KeyError, TypeError):
        return np.asarray(value)


def _canonical_heading_computation(value: object) -> dict[str, object]:
    """Require the exact three-landmark pose-schema heading recipe used by v1."""

    if not isinstance(value, Mapping):
        raise ValueError("Body-frame heading_computation must be an object.")
    canonical = json.loads(
        json.dumps(dict(value), sort_keys=True, separators=(",", ":"))
    )
    expected = {
        "version": 1,
        "enabled": True,
        "origin": {
            "op": "midpoint",
            "labels": ["eye_left", "eye_right"],
        },
        "direction_from": {"op": "keypoint", "label": "swim_bladder"},
        "direction_to": {
            "op": "midpoint",
            "labels": ["eye_left", "eye_right"],
        },
        "dependent_keypoints": list(_HEADING_DEPENDENCIES),
    }
    if canonical != expected:
        raise ValueError(
            "Body-frame v1 requires the exact enabled three-landmark "
            "pose-schema heading computation."
        )
    return canonical


@dataclass(frozen=True)
class BodyFrameSourceReference:
    """Immutable identity of the one complete keypoint-v2 source snapshot."""

    stage: str
    run_name: str
    manifest_digest: str
    skeleton_id: str
    skeleton_digest: str
    keypoint_row_signatures_digest: str

    def __post_init__(self) -> None:
        stage = str(self.stage).strip()
        if stage not in _SOURCE_STAGES:
            raise ValueError(
                "Body-frame source stage must be keypoints or refined_keypoints."
            )
        run_name = str(self.run_name).strip()
        if not run_name or "/" in run_name:
            raise ValueError("run_name must be one nonempty archive group name.")
        skeleton_id = str(self.skeleton_id).strip()
        if not skeleton_id:
            raise ValueError("skeleton_id cannot be empty.")
        object.__setattr__(self, "stage", stage)
        object.__setattr__(self, "run_name", run_name)
        object.__setattr__(self, "skeleton_id", skeleton_id)
        for name in (
            "manifest_digest",
            "skeleton_digest",
            "keypoint_row_signatures_digest",
        ):
            object.__setattr__(
                self,
                name,
                _require_sha256(getattr(self, name), name=name),
            )

    @property
    def run_path(self) -> str:
        family = (
            "keypoints_runs" if self.stage == "keypoints" else "refined_keypoints_runs"
        )
        return f"{family}/{self.run_name}"

    @property
    def schema(self):  # type: ignore[no-untyped-def]
        return (
            KEYPOINT_SCHEMA_V2
            if self.stage == "keypoints"
            else REFINED_KEYPOINT_SCHEMA_V2
        )

    def as_manifest(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "run_name": self.run_name,
            "run_path": self.run_path,
            "schema_id": self.schema.schema_id,
            "schema_version": self.schema.schema_version,
            "manifest_digest": self.manifest_digest,
            "skeleton_id": self.skeleton_id,
            "skeleton_digest": self.skeleton_digest,
            "keypoint_row_signatures_digest": self.keypoint_row_signatures_digest,
            "coverage": "complete_row_for_row_snapshot",
        }


@dataclass(frozen=True)
class KeypointBodyFrameRecipe:
    """Exact, digest-bound recipe for source-camera body axes and heading."""

    swim_bladder_index: int
    eye_left_index: int
    eye_right_index: int
    skeleton_digest: str
    heading_computation: Mapping[str, object]
    heading_computation_source: str = KEYPOINT_HEADING_COMPUTATION_SOURCE
    recipe_version: int = BODY_FRAME_RECIPE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        indices = (
            self.swim_bladder_index,
            self.eye_left_index,
            self.eye_right_index,
        )
        if any(type(value) is not int or value < 0 for value in indices):
            raise ValueError(
                "Body-frame keypoint indices must be nonnegative integers."
            )
        if len(set(indices)) != 3:
            raise ValueError("Body-frame keypoint indices must be distinct.")
        if type(self.recipe_version) is not int or self.recipe_version != 1:
            raise ValueError("Only body-frame recipe version 1 is supported.")
        object.__setattr__(
            self,
            "skeleton_digest",
            _require_sha256(self.skeleton_digest, name="skeleton_digest"),
        )
        source = str(self.heading_computation_source).strip()
        if source != KEYPOINT_HEADING_COMPUTATION_SOURCE:
            raise ValueError(
                "Body-frame v1 heading computation must come from "
                "pose_schema.metadata.heading_computation."
            )
        object.__setattr__(self, "heading_computation_source", source)
        object.__setattr__(
            self,
            "heading_computation",
            _canonical_heading_computation(self.heading_computation),
        )

    @property
    def heading_computation_digest(self) -> str:
        return canonical_json_sha256(self.heading_computation)

    def payload(self) -> dict[str, object]:
        return {
            "schema_id": BODY_FRAME_RECIPE_SCHEMA_ID,
            "schema_version": self.recipe_version,
            "recipe_id": KEYPOINT_HEAD_AXIS_RECIPE_ID,
            "skeleton_digest": self.skeleton_digest,
            "heading_computation_source": self.heading_computation_source,
            "heading_computation": dict(self.heading_computation),
            "heading_computation_digest": self.heading_computation_digest,
            "keypoint_indices": {
                "swim_bladder": self.swim_bladder_index,
                "eye_left": self.eye_left_index,
                "eye_right": self.eye_right_index,
            },
            "coordinate_source": "keypoints_img_source_camera_pixels",
            "origin": "midpoint_eye_left_eye_right",
            "forward_axis": "unit_vector_swim_bladder_to_eye_midpoint",
            "left_axis": "fixed_clockwise_90_degrees_in_camera_xy",
            "axis_handedness": "determinant_negative_one_camera_xy",
            "heading_deg": "atan2_negative_forward_y_forward_x_degrees_float32",
            "invalid_geometry": "all_nan_and_axis_valid_false",
        }

    @property
    def recipe_digest(self) -> str:
        return canonical_json_sha256(self.payload())

    def as_manifest(self) -> dict[str, object]:
        return {**self.payload(), "recipe_digest": self.recipe_digest}


def build_keypoint_body_frame_recipe(
    *,
    pose_schema: object,
    skeleton_digest: str,
    keypoint_count: int | None = None,
) -> KeypointBodyFrameRecipe:
    """Resolve the v1 recipe from canonical pose-schema metadata and labels."""

    resolved = resolve_heading_computation(pose_schema=pose_schema)
    if resolved.source != KEYPOINT_HEADING_COMPUTATION_SOURCE:
        raise ValueError(
            "Pose schema does not define canonical heading_computation metadata."
        )
    heading_computation = _canonical_heading_computation(resolved.spec)
    labels = resolve_keypoint_labels_from_attrs(
        {"pose_schema": pose_schema},
        keypoint_count=keypoint_count,
    )
    if not labels:
        raise ValueError("Pose schema does not define ordered keypoint labels.")
    indices = resolve_required_keypoint_indices(
        labels,
        _HEADING_DEPENDENCIES,
        keypoint_count=keypoint_count,
    )
    return KeypointBodyFrameRecipe(
        swim_bladder_index=indices["swim_bladder"],
        eye_left_index=indices["eye_left"],
        eye_right_index=indices["eye_right"],
        skeleton_digest=skeleton_digest,
        heading_computation=heading_computation,
    )


@dataclass(frozen=True)
class PreparedBodyFrameSnapshot:
    dimensions: BodyFrameDimensions
    source_dimensions: KeypointDimensions
    source: BodyFrameSourceReference
    recipe: KeypointBodyFrameRecipe
    arrays: Mapping[str, np.ndarray]
    source_arrays: Mapping[str, np.ndarray]


def derive_body_frame_geometry(
    source_arrays: Mapping[str, Any],
    *,
    recipe: KeypointBodyFrameRecipe,
) -> dict[str, np.ndarray]:
    """Derive v1 geometry without the legacy lateral-axis sign flip."""

    points = _array_values(source_arrays["keypoints_img"])
    valid_points = _array_values(source_arrays["keypoint_valid"])
    pose_usable = _array_values(source_arrays["pose_usable"])
    if points.dtype != np.dtype(np.float32) or points.ndim != 3 or points.shape[2] != 2:
        raise ValueError("keypoints_img must have exact float32 shape (N, K, 2).")
    if valid_points.dtype != np.dtype(bool) or valid_points.shape != points.shape[:2]:
        raise ValueError("keypoint_valid must have exact bool shape (N, K).")
    if pose_usable.dtype != np.dtype(bool) or pose_usable.shape != (points.shape[0],):
        raise ValueError("pose_usable must have exact bool shape (N,).")
    indices = np.asarray(
        [recipe.swim_bladder_index, recipe.eye_left_index, recipe.eye_right_index],
        dtype=np.int64,
    )
    if np.any(indices >= points.shape[1]):
        raise ValueError("Body-frame recipe index exceeds the source skeleton.")

    anchors = points[:, indices, :].astype(np.float64)
    anchor_valid = np.all(valid_points[:, indices], axis=1) & np.all(
        np.isfinite(anchors), axis=(1, 2)
    )
    bladder = anchors[:, 0, :]
    origin64 = 0.5 * (anchors[:, 1, :] + anchors[:, 2, :])
    forward_raw = origin64 - bladder
    norms = np.linalg.norm(forward_raw, axis=1)
    axis_valid = pose_usable & anchor_valid & np.isfinite(norms) & (norms > 1.0e-9)

    n_rows = int(points.shape[0])
    origin = np.full((n_rows, 2), np.nan, dtype=np.float32)
    forward = np.full((n_rows, 2), np.nan, dtype=np.float32)
    left = np.full((n_rows, 2), np.nan, dtype=np.float32)
    heading = np.full(n_rows, np.nan, dtype=np.float32)
    if np.any(axis_valid):
        origin[axis_valid] = origin64[axis_valid].astype(np.float32)
        forward[axis_valid] = (
            forward_raw[axis_valid] / norms[axis_valid, None]
        ).astype(np.float32)
        # Camera y increases down. This fixed clockwise rotation is the only
        # v1 lateral-axis construction and always has determinant -1.
        left[axis_valid, 0] = forward[axis_valid, 1]
        left[axis_valid, 1] = -forward[axis_valid, 0]
        heading[axis_valid] = np.rad2deg(
            np.arctan2(-forward[axis_valid, 1], forward[axis_valid, 0])
        ).astype(np.float32)
    return {
        "origin_xy": origin,
        "forward_axis_xy": forward,
        "left_axis_xy": left,
        "axis_valid": axis_valid,
        "heading_deg": heading,
    }


def prepare_keypoint_body_frame(
    source_keypoint_arrays: Mapping[str, Any],
    *,
    source_dimensions: KeypointDimensions,
    source_crop_arrays: Mapping[str, Any],
    source: BodyFrameSourceReference,
    source_manifest: Mapping[str, Any],
    recipe: KeypointBodyFrameRecipe,
    review_state_map: Mapping[int, str] | None = None,
    reason_code_map: Mapping[int, str] | None = None,
) -> PreparedBodyFrameSnapshot:
    """Validate exactly one complete keypoint-v2 snapshot and derive body frames."""

    if canonical_json_sha256(source_manifest) != source.manifest_digest:
        raise ValueError("Source manifest differs from the bound digest.")
    if source.skeleton_digest != recipe.skeleton_digest:
        raise ValueError("Source and body-frame recipe skeleton digests differ.")
    source.schema.require(
        source_keypoint_arrays,
        dimensions=source_dimensions,
        source_crop_arrays=source_crop_arrays,
        skeleton_digest=source.skeleton_digest,
        review_state_map=review_state_map,
        reason_code_map=reason_code_map,
    )
    pose_path = "pose_success" if source.stage == "keypoints" else "refined_success"
    evidence_paths = (
        "instance_key",
        "frame_indices",
        "keypoint_row_signature",
        "keypoints_img",
        "keypoint_valid",
    )
    # The source snapshot is immutable. Retain its materialized arrays without
    # another full-size copy; bounded-row DAG materialization is a separate
    # integration concern and is intentionally not invented by this publisher.
    evidence = {
        path: _array_values(source_keypoint_arrays[path]) for path in evidence_paths
    }
    evidence["pose_usable"] = _array_values(source_keypoint_arrays[pose_path])
    if sha256_array(evidence["keypoint_row_signature"]) != (
        source.keypoint_row_signatures_digest
    ):
        raise ValueError(
            "Source keypoint-row signature digest differs from its binding."
        )

    frames = evidence["frame_indices"]
    dimensions = BodyFrameDimensions(
        n_frames=source_dimensions.n_frames,
        n_instances=source_dimensions.n_instances,
    )
    arrays = {
        "instance_key": evidence["instance_key"],
        "source_keypoint_row_ids": np.arange(
            source_dimensions.n_instances, dtype=np.int64
        ),
        "source_keypoint_row_signature": evidence["keypoint_row_signature"],
        "frame_indices": frames,
        "frame_row_offsets": derive_frame_row_offsets(
            frames, n_frames=source_dimensions.n_frames
        ),
        **derive_body_frame_geometry(evidence, recipe=recipe),
    }
    BODY_FRAME_SCHEMA_V1.require(
        arrays,
        dimensions=dimensions,
        source_keypoint_arrays=evidence,
    )
    return PreparedBodyFrameSnapshot(
        dimensions=dimensions,
        source_dimensions=source_dimensions,
        source=source,
        recipe=recipe,
        arrays=arrays,
        source_arrays=evidence,
    )


__all__ = [
    "BODY_FRAME_RECIPE_SCHEMA_ID",
    "BODY_FRAME_RECIPE_SCHEMA_VERSION",
    "KEYPOINT_HEAD_AXIS_RECIPE_ID",
    "BodyFrameSourceReference",
    "KeypointBodyFrameRecipe",
    "PreparedBodyFrameSnapshot",
    "build_keypoint_body_frame_recipe",
    "derive_body_frame_geometry",
    "prepare_keypoint_body_frame",
]
