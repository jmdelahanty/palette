"""Mask-component body-frame producer for the shared body-frame-v1 schema.

This module is intentionally an array-level adapter.  It consumes a validated,
explicitly bound refined-subject-mask snapshot and never opens masks, derives a
centroid, resolves a selector, or publishes a run.  The existing body-frame-v1
array schema predates mask sources and names its two source-lineage arrays
``source_keypoint_*``.  The mask adapter preserves those physical arrays for
schema compatibility while carrying the exact mask source identity separately
in the source reference and publication payload.  This keeps the keypoint
contracts unchanged and gives the later shared-schema publication work an
unambiguous conversion surface.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import re
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from fisheye.shared.anatomy_profile import (
    AnatomyProfile,
    anatomy_profile_sha256,
    load_anatomy_profile,
    source_binding_sha256,
    source_schema_sha256,
    validate_source_binding,
)
from fisheye.shared.zarr.benchmark_runtime import sha256_array
from fisheye.shared.zarr.body_frame_schema import (
    BODY_FRAME_SCHEMA_V1,
    BodyFrameDimensions,
)
from fisheye.shared.zarr.keypoint_schema import derive_frame_row_offsets
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256

MASK_BODY_FRAME_RECIPE_SCHEMA_ID = "palette.body_frame.mask_heading_recipe"
MASK_BODY_FRAME_RECIPE_SCHEMA_VERSION = 1
MASK_BODY_FRAME_RECIPE_ID = "mask_component_anterior_axis_v1"
MASK_BODY_FRAME_SOURCE_STAGE = "refined_subject_masks"
MASK_BODY_FRAME_SOURCE_MODALITY = "subject_mask"
MASK_BODY_FRAME_RECIPE_PROFILE_RECIPE_ID = "anterior_axis"
MASK_BODY_FRAME_COORDINATE_SOURCE = (
    "subject_mask_component_centroid_source_camera_pixels"
)
MASK_BODY_FRAME_SOURCE_SCHEMA_ALIAS = (
    "source_keypoint_row_ids/source_keypoint_row_signature are body-frame-v1 "
    "compatibility aliases for exact refined-subject-mask source rows"
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_ROLES = ("swim_bladder", "eye_left", "eye_right")
_AXIS_EPSILON = 1.0e-9


class MaskBodyFrameProducerError(ValueError):
    """Raised when a mask body-frame source is not an exact authority."""


def _require_text(value: object, *, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise MaskBodyFrameProducerError(f"{name} must be a non-empty string.")
    return value.strip()


def _require_sha256(value: object, *, name: str) -> str:
    normalized = _require_text(value, name=name)
    if _SHA256.fullmatch(normalized) is None:
        raise MaskBodyFrameProducerError(
            f"{name} must be a lowercase hexadecimal SHA-256 digest."
        )
    return normalized


def _array(value: Any, *, name: str) -> np.ndarray:
    try:
        result = np.asarray(value if isinstance(value, np.ndarray) else value[...])
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        raise MaskBodyFrameProducerError(f"Unable to read {name}.") from exc
    return result


def _readonly(value: Any, *, dtype: np.dtype[Any] | None = None) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _profile(value: AnatomyProfile | Mapping[str, Any] | str | Path) -> AnatomyProfile:
    if isinstance(value, AnatomyProfile):
        return value
    if isinstance(value, (str, Path)):
        return load_anatomy_profile(value)
    return AnatomyProfile.from_mapping(value)


def _canonical_json(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(
        json.dumps(_thaw_json(value), sort_keys=True, separators=(",", ":"))
    )


def _axis_roles(profile: AnatomyProfile) -> tuple[str, str, str]:
    try:
        recipe = profile.recipe(MASK_BODY_FRAME_RECIPE_PROFILE_RECIPE_ID)
    except Exception as exc:
        raise MaskBodyFrameProducerError(
            "The bound anatomy profile does not define anterior_axis."
        ) from exc
    if recipe.kind != "axis" or set(recipe.required_roles) != set(_REQUIRED_ROLES):
        raise MaskBodyFrameProducerError(
            "anterior_axis must be the exact three-role zebrafish anterior axis."
        )
    expression = recipe.expression
    if expression.get("op") != "axis":
        raise MaskBodyFrameProducerError("anterior_axis must be an axis expression.")
    from_expression = expression.get("from")
    to_expression = expression.get("to")
    if not isinstance(from_expression, Mapping) or not isinstance(
        to_expression, Mapping
    ):
        raise MaskBodyFrameProducerError("anterior_axis endpoints are not canonical.")
    if (
        from_expression.get("op") != "role_point"
        or from_expression.get("role_id") != "swim_bladder"
    ):
        raise MaskBodyFrameProducerError(
            "anterior_axis must start at the swim_bladder role."
        )
    if to_expression.get("op") != "midpoint":
        raise MaskBodyFrameProducerError(
            "anterior_axis must terminate at the eye-pair midpoint."
        )
    points = to_expression.get("points")
    if not isinstance(points, Sequence) or isinstance(points, (str, bytes)):
        raise MaskBodyFrameProducerError(
            "anterior_axis eye endpoint must contain two point roles."
        )
    point_roles = tuple(
        point.get("role_id")
        for point in points
        if isinstance(point, Mapping) and point.get("op") == "role_point"
    )
    if set(point_roles) != {"eye_left", "eye_right"} or len(point_roles) != 2:
        raise MaskBodyFrameProducerError(
            "anterior_axis eye endpoint must use eye_left and eye_right exactly once."
        )
    return "swim_bladder", "eye_left", "eye_right"


def _controlled_binding(
    profile: AnatomyProfile,
    *,
    binding_id: str,
) -> tuple[dict[str, Any], dict[str, str], tuple[str, ...]]:
    binding_id = _require_text(binding_id, name="binding_id")
    try:
        binding = validate_source_binding(profile, profile.binding(binding_id))
    except Exception as exc:
        raise MaskBodyFrameProducerError(
            f"The anatomy source binding {binding_id!r} is invalid."
        ) from exc
    source_schema = binding.get("source_schema")
    if not isinstance(source_schema, Mapping):
        raise MaskBodyFrameProducerError("The source binding has no source schema.")
    if source_schema.get("modality") != MASK_BODY_FRAME_SOURCE_MODALITY:
        raise MaskBodyFrameProducerError(
            "A mask body frame requires a subject_mask source binding."
        )
    if source_schema.get("authority") != "declared_schema":
        raise MaskBodyFrameProducerError(
            "A mask body frame requires a declared-schema source binding."
        )
    declared_labels = tuple(str(label) for label in source_schema.get("labels", ()))
    if not declared_labels or len(set(declared_labels)) != len(declared_labels):
        raise MaskBodyFrameProducerError(
            "The bound subject-mask schema has no exact unique labels."
        )
    role_mapping = {
        str(item["role_id"]): str(item["source_label"])
        for item in binding.get("role_bindings", ())
    }
    _axis_roles(profile)
    if not set(_REQUIRED_ROLES).issubset(role_mapping):
        raise MaskBodyFrameProducerError(
            "The bound subject-mask schema is missing an anterior-axis role."
        )
    if not set(role_mapping.values()).issubset(set(declared_labels)):
        raise MaskBodyFrameProducerError(
            "The source binding maps an anatomy role to an undeclared component."
        )
    if MASK_BODY_FRAME_RECIPE_PROFILE_RECIPE_ID not in binding.get(
        "advertised_recipe_ids", ()
    ):
        raise MaskBodyFrameProducerError(
            "The subject-mask binding does not advertise anterior_axis."
        )
    expected_schema_digest = source_schema_sha256(source_schema)
    if source_schema.get("schema_sha256") != expected_schema_digest:
        raise MaskBodyFrameProducerError(
            "The bound subject-mask source schema digest is stale."
        )
    return binding, role_mapping, declared_labels


@dataclass(frozen=True)
class MaskBodyFrameSourceReference:
    """Exact identity of one refined subject-mask centroid source rowset."""

    run_path: str
    run_manifest_digest: str
    mask_schema_id: str
    mask_schema_version: int
    mask_schema_digest: str
    anatomy_profile_id: str
    anatomy_profile_version: int
    anatomy_profile_digest: str
    binding_id: str
    binding_digest: str
    row_identity_digest: str
    instance_key_digest: str
    frame_indices_digest: str
    source_acquisition_frame_index_digest: str
    source_row_ids_digest: str
    source_row_signature_digest: str

    def __post_init__(self) -> None:
        run_path = _require_text(self.run_path, name="run_path")
        if not run_path.startswith("refined_subject_masks_runs/"):
            raise MaskBodyFrameProducerError(
                "Mask body-frame sources must be explicit refined_subject_masks_runs paths."
            )
        run_name = run_path.split("/", 1)[1]
        if not run_name or "/" in run_name:
            raise MaskBodyFrameProducerError(
                "run_path must name exactly one refined run."
            )
        object.__setattr__(self, "run_path", run_path)
        for name in (
            "run_manifest_digest",
            "mask_schema_digest",
            "anatomy_profile_digest",
            "binding_digest",
            "row_identity_digest",
            "instance_key_digest",
            "frame_indices_digest",
            "source_acquisition_frame_index_digest",
            "source_row_ids_digest",
            "source_row_signature_digest",
        ):
            object.__setattr__(
                self,
                name,
                _require_sha256(getattr(self, name), name=name),
            )
        if type(self.mask_schema_version) is not int or self.mask_schema_version <= 0:
            raise MaskBodyFrameProducerError("mask_schema_version must be positive.")
        if (
            type(self.anatomy_profile_version) is not int
            or self.anatomy_profile_version <= 0
        ):
            raise MaskBodyFrameProducerError(
                "anatomy_profile_version must be positive."
            )

    @property
    def run_name(self) -> str:
        return self.run_path.split("/", 1)[1]

    def as_manifest(self) -> dict[str, Any]:
        return {
            "stage": MASK_BODY_FRAME_SOURCE_STAGE,
            "modality": MASK_BODY_FRAME_SOURCE_MODALITY,
            "run_path": self.run_path,
            "run_name": self.run_name,
            "run_manifest_digest": self.run_manifest_digest,
            "mask_schema_id": self.mask_schema_id,
            "mask_schema_version": self.mask_schema_version,
            "mask_schema_digest": self.mask_schema_digest,
            "anatomy_profile_id": self.anatomy_profile_id,
            "anatomy_profile_version": self.anatomy_profile_version,
            "anatomy_profile_digest": self.anatomy_profile_digest,
            "binding_id": self.binding_id,
            "binding_digest": self.binding_digest,
            "row_identity_digest": self.row_identity_digest,
            "instance_key_digest": self.instance_key_digest,
            "frame_indices_digest": self.frame_indices_digest,
            "source_acquisition_frame_index_digest": self.source_acquisition_frame_index_digest,
            "source_row_ids_digest": self.source_row_ids_digest,
            "source_row_signature_digest": self.source_row_signature_digest,
            "coverage": "complete_refined_subject_mask_centroid_rowset",
        }


@dataclass(frozen=True)
class MaskBodyFrameRecipe:
    """Digest-bound anatomy and mask-source recipe for body-frame-v1."""

    anatomy_profile_id: str
    anatomy_profile_version: int
    anatomy_profile_digest: str
    binding_id: str
    binding_digest: str
    mask_schema_id: str
    mask_schema_version: int
    mask_schema_digest: str
    role_bindings: Mapping[str, str]
    recipe_id: str = MASK_BODY_FRAME_RECIPE_ID
    recipe_version: int = MASK_BODY_FRAME_RECIPE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.recipe_id != MASK_BODY_FRAME_RECIPE_ID:
            raise MaskBodyFrameProducerError("Unknown mask body-frame recipe.")
        if self.recipe_version != MASK_BODY_FRAME_RECIPE_SCHEMA_VERSION:
            raise MaskBodyFrameProducerError(
                "Unsupported mask body-frame recipe version."
            )
        if (
            type(self.anatomy_profile_version) is not int
            or self.anatomy_profile_version <= 0
        ):
            raise MaskBodyFrameProducerError(
                "anatomy_profile_version must be positive."
            )
        if type(self.mask_schema_version) is not int or self.mask_schema_version <= 0:
            raise MaskBodyFrameProducerError("mask_schema_version must be positive.")
        for name in (
            "anatomy_profile_digest",
            "binding_digest",
            "mask_schema_digest",
        ):
            object.__setattr__(
                self,
                name,
                _require_sha256(getattr(self, name), name=name),
            )
        role_bindings = dict(self.role_bindings)
        if set(role_bindings) != set(_REQUIRED_ROLES) or len(
            set(role_bindings.values())
        ) != len(role_bindings):
            raise MaskBodyFrameProducerError(
                "Mask body-frame role bindings must contain the exact three roles."
            )
        object.__setattr__(self, "role_bindings", MappingProxyType(role_bindings))

    def payload(self) -> dict[str, Any]:
        return {
            "schema_id": MASK_BODY_FRAME_RECIPE_SCHEMA_ID,
            "schema_version": self.recipe_version,
            "recipe_id": self.recipe_id,
            "recipe_profile_recipe_id": MASK_BODY_FRAME_RECIPE_PROFILE_RECIPE_ID,
            "anatomy_profile_id": self.anatomy_profile_id,
            "anatomy_profile_version": self.anatomy_profile_version,
            "anatomy_profile_digest": self.anatomy_profile_digest,
            "binding_id": self.binding_id,
            "binding_digest": self.binding_digest,
            "mask_schema_id": self.mask_schema_id,
            "mask_schema_version": self.mask_schema_version,
            "mask_schema_digest": self.mask_schema_digest,
            "role_bindings": dict(sorted(self.role_bindings.items())),
            "coordinate_source": MASK_BODY_FRAME_COORDINATE_SOURCE,
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

    def as_manifest(self) -> dict[str, Any]:
        return {**self.payload(), "recipe_digest": self.recipe_digest}


@dataclass(frozen=True)
class PreparedMaskBodyFrameSnapshot:
    """Publication-ready body-frame arrays with exact mask source evidence."""

    dimensions: BodyFrameDimensions
    source: MaskBodyFrameSourceReference
    recipe: MaskBodyFrameRecipe
    arrays: Mapping[str, np.ndarray]
    source_arrays: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        object.__setattr__(self, "arrays", MappingProxyType(dict(self.arrays)))
        object.__setattr__(
            self, "source_arrays", MappingProxyType(dict(self.source_arrays))
        )

    def as_publication_payload(self) -> dict[str, Any]:
        """Return the explicit bridge payload for later mask-aware publication."""

        return {
            "schema": BODY_FRAME_SCHEMA_V1.as_manifest(dimensions=self.dimensions),
            "schema_compatibility": MASK_BODY_FRAME_SOURCE_SCHEMA_ALIAS,
            "arrays": dict(self.arrays),
            "source_mask_arrays": dict(self.source_arrays),
            "source_mask_snapshot": self.source.as_manifest(),
            "heading_recipe": self.recipe.as_manifest(),
            "publication": {
                "selector_changes": False,
                "source_stage": MASK_BODY_FRAME_SOURCE_STAGE,
                "source_modality": MASK_BODY_FRAME_SOURCE_MODALITY,
                "dense_masks_reopened": False,
                "centroids_recomputed": False,
            },
        }


def build_mask_body_frame_recipe(
    *,
    anatomy_profile: AnatomyProfile | Mapping[str, Any] | str | Path,
    binding_id: str,
    mask_schema_id: str,
    mask_schema_version: int,
    mask_schema_digest: str,
) -> MaskBodyFrameRecipe:
    """Bind ``anterior_axis`` to one explicit subject-mask schema."""

    profile = _profile(anatomy_profile)
    binding, role_mapping, declared_labels = _controlled_binding(
        profile, binding_id=binding_id
    )
    del declared_labels
    source_schema = binding["source_schema"]
    if mask_schema_id != source_schema["schema_id"]:
        raise MaskBodyFrameProducerError("mask_schema_id disagrees with the binding.")
    if mask_schema_version != source_schema["schema_version"]:
        raise MaskBodyFrameProducerError(
            "mask_schema_version disagrees with the binding."
        )
    expected_schema_digest = source_schema_sha256(source_schema)
    if mask_schema_digest != expected_schema_digest:
        raise MaskBodyFrameProducerError(
            "mask_schema_digest disagrees with the bound source schema."
        )
    return MaskBodyFrameRecipe(
        anatomy_profile_id=profile.profile_id,
        anatomy_profile_version=profile.profile_version,
        anatomy_profile_digest=anatomy_profile_sha256(profile),
        binding_id=binding_id,
        binding_digest=source_binding_sha256(binding),
        mask_schema_id=mask_schema_id,
        mask_schema_version=mask_schema_version,
        mask_schema_digest=mask_schema_digest,
        role_bindings={role: role_mapping[role] for role in _REQUIRED_ROLES},
    )


def _validate_source_arrays(
    *,
    component_centroid_xy: np.ndarray,
    component_centroid_valid: np.ndarray,
    component_labels: tuple[str, ...],
    available_channels: np.ndarray,
    instance_key: np.ndarray,
    frame_indices: np.ndarray,
    source_acquisition_frame_index: np.ndarray,
    source_row_ids: np.ndarray,
    source_row_signature: np.ndarray,
    n_frames: int,
) -> None:
    if component_centroid_xy.dtype != np.dtype("float32"):
        raise MaskBodyFrameProducerError(
            "component_centroid_xy must use the validated float32 dtype."
        )
    if component_centroid_xy.ndim != 3 or component_centroid_xy.shape[2] != 2:
        raise MaskBodyFrameProducerError(
            "component_centroid_xy must have shape [N, C, 2]."
        )
    rows, channels, _ = component_centroid_xy.shape
    if component_centroid_valid.dtype != np.dtype(
        bool
    ) or component_centroid_valid.shape != (
        rows,
        channels,
    ):
        raise MaskBodyFrameProducerError(
            "component_centroid_valid must be bool with shape [N, C]."
        )
    if available_channels.dtype != np.dtype(bool) or available_channels.shape != (
        channels,
    ):
        raise MaskBodyFrameProducerError(
            "available_channels must be bool with shape [C]."
        )
    if len(component_labels) != channels or not component_labels:
        raise MaskBodyFrameProducerError(
            "component_labels must exactly enumerate centroid channels."
        )
    if len(set(component_labels)) != len(component_labels):
        raise MaskBodyFrameProducerError("component_labels must be unique.")
    if instance_key.dtype != np.dtype("uint64") or instance_key.shape != (rows,):
        raise MaskBodyFrameProducerError("instance_key must be uint64 with shape [N].")
    if frame_indices.dtype != np.dtype("int64") or frame_indices.shape != (rows,):
        raise MaskBodyFrameProducerError("frame_indices must be int64 with shape [N].")
    if source_acquisition_frame_index.dtype != np.dtype(
        "int64"
    ) or source_acquisition_frame_index.shape != (rows,):
        raise MaskBodyFrameProducerError(
            "source_acquisition_frame_index must be int64 with shape [N]."
        )
    if source_row_ids.dtype != np.dtype("int64") or source_row_ids.shape != (rows,):
        raise MaskBodyFrameProducerError("source_row_ids must be int64 with shape [N].")
    if source_row_signature.dtype != np.dtype(
        "uint8"
    ) or source_row_signature.shape != (
        rows,
        32,
    ):
        raise MaskBodyFrameProducerError(
            "source_row_signature must be uint8 with shape [N, 32]."
        )
    if type(n_frames) is not int or n_frames <= 0:
        raise MaskBodyFrameProducerError("n_frames must be a positive exact integer.")
    if np.unique(instance_key).size != rows:
        raise MaskBodyFrameProducerError("instance_key values must be unique.")
    if np.any(source_row_ids < 0) or np.unique(source_row_ids).size != rows:
        raise MaskBodyFrameProducerError(
            "source_row_ids must be unique nonnegative exact source IDs."
        )
    if np.any(frame_indices < 0) or np.any(frame_indices >= n_frames):
        raise MaskBodyFrameProducerError("frame_indices are outside the source extent.")
    if rows > 1 and np.any(np.diff(frame_indices) < 0):
        raise MaskBodyFrameProducerError(
            "frame_indices must be nondecreasing for body-frame-v1 row order."
        )
    if np.any(source_acquisition_frame_index < 0):
        raise MaskBodyFrameProducerError(
            "source_acquisition_frame_index must be nonnegative."
        )
    finite_valid = (
        np.all(np.isfinite(component_centroid_xy), axis=2) | ~component_centroid_valid
    )
    if not np.all(finite_valid):
        raise MaskBodyFrameProducerError(
            "Valid component centroids must be finite; invalid centroids may use the "
            "canonical source sentinel or NaN."
        )


def derive_mask_body_frame_geometry(
    component_centroid_xy: Any,
    component_centroid_valid: Any,
    *,
    component_labels: Sequence[str],
    role_bindings: Mapping[str, str],
    available_channels: Any | None = None,
) -> dict[str, np.ndarray]:
    """Derive deterministic body-frame arrays from persisted mask centroids.

    ``role_bindings`` is the already validated anatomy source binding.  The
    function resolves channels by label and never interprets a numeric channel
    position as a biological role.
    """

    centroids = _array(component_centroid_xy, name="component_centroid_xy")
    valid = _array(component_centroid_valid, name="component_centroid_valid")
    labels = tuple(
        _require_text(label, name="component label") for label in component_labels
    )
    if available_channels is None:
        available = np.ones(len(labels), dtype=bool)
    else:
        available = _array(available_channels, name="available_channels")
    if (
        centroids.dtype != np.dtype("float32")
        or centroids.ndim != 3
        or centroids.shape[2] != 2
    ):
        raise MaskBodyFrameProducerError(
            "component_centroid_xy must be float32 with shape [N, C, 2]."
        )
    rows, channels, _ = centroids.shape
    if valid.dtype != np.dtype(bool) or valid.shape != (rows, channels):
        raise MaskBodyFrameProducerError(
            "component_centroid_valid must be bool with shape [N, C]."
        )
    if available.dtype != np.dtype(bool) or available.shape != (channels,):
        raise MaskBodyFrameProducerError(
            "available_channels must be bool with shape [C]."
        )
    if len(labels) != channels or len(set(labels)) != channels:
        raise MaskBodyFrameProducerError(
            "component_labels must exactly enumerate unique centroid channels."
        )
    if not np.all(np.isfinite(centroids) | ~valid[:, :, None]):
        raise MaskBodyFrameProducerError(
            "Valid component centroids must be finite; invalid centroids may use "
            "the canonical source sentinel or NaN."
        )
    label_to_index = {label: index for index, label in enumerate(labels)}
    if not set(_REQUIRED_ROLES).issubset(role_bindings):
        raise MaskBodyFrameProducerError(
            "role_bindings must contain every anterior-axis role."
        )
    try:
        indices = [label_to_index[role_bindings[role]] for role in _REQUIRED_ROLES]
    except KeyError as exc:
        raise MaskBodyFrameProducerError(
            "The anatomy binding refers to a missing component label."
        ) from exc
    if not np.all(available[indices]):
        raise MaskBodyFrameProducerError(
            "A required anterior-axis component is unavailable in the source run."
        )

    anchors = centroids[:, indices, :].astype(np.float64, copy=False)
    required_valid = np.all(valid[:, indices], axis=1)
    required_finite = np.all(np.isfinite(anchors), axis=(1, 2))
    origin64 = 0.5 * (anchors[:, 1, :] + anchors[:, 2, :])
    forward_raw = origin64 - anchors[:, 0, :]
    norms = np.linalg.norm(forward_raw, axis=1)
    axis_valid = (
        required_valid & required_finite & np.isfinite(norms) & (norms > _AXIS_EPSILON)
    )
    origin = np.full((rows, 2), np.nan, dtype=np.float32)
    forward = np.full((rows, 2), np.nan, dtype=np.float32)
    left = np.full((rows, 2), np.nan, dtype=np.float32)
    heading = np.full(rows, np.nan, dtype=np.float32)
    if np.any(axis_valid):
        origin[axis_valid] = origin64[axis_valid].astype(np.float32)
        forward[axis_valid] = (
            forward_raw[axis_valid] / norms[axis_valid, None]
        ).astype(np.float32)
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


def prepare_refined_subject_mask_body_frame(
    *,
    run_path: str,
    run_manifest_digest: str,
    mask_schema_id: str,
    mask_schema_version: int,
    mask_schema_digest: str,
    anatomy_profile: AnatomyProfile | Mapping[str, Any] | str | Path,
    binding_id: str,
    component_labels: Sequence[str],
    component_centroid_xy: Any,
    component_centroid_valid: Any,
    available_channels: Any | None,
    instance_key: Any,
    frame_indices: Any,
    source_acquisition_frame_index: Any,
    source_row_ids: Any,
    source_row_signature: Any,
    row_identity_digest: str,
    n_frames: int,
) -> PreparedMaskBodyFrameSnapshot:
    """Prepare one mask-derived body-frame-v1 snapshot without publication."""

    profile = _profile(anatomy_profile)
    recipe = build_mask_body_frame_recipe(
        anatomy_profile=profile,
        binding_id=binding_id,
        mask_schema_id=mask_schema_id,
        mask_schema_version=mask_schema_version,
        mask_schema_digest=mask_schema_digest,
    )
    binding, role_mapping, declared_labels = _controlled_binding(
        profile, binding_id=binding_id
    )
    del binding
    labels = tuple(
        _require_text(label, name="component label") for label in component_labels
    )
    if set(labels) != set(declared_labels):
        raise MaskBodyFrameProducerError(
            "The refined run component labels do not match the bound mask schema."
        )
    centroids = _array(component_centroid_xy, name="component_centroid_xy")
    centroid_valid = _array(component_centroid_valid, name="component_centroid_valid")
    keys = _array(instance_key, name="instance_key")
    frames = _array(frame_indices, name="frame_indices")
    acquisition_frames = _array(
        source_acquisition_frame_index, name="source_acquisition_frame_index"
    )
    row_ids = _array(source_row_ids, name="source_row_ids")
    signatures = _array(source_row_signature, name="source_row_signature")
    rows = int(centroids.shape[0]) if centroids.ndim else -1
    available = (
        np.ones(len(labels), dtype=bool)
        if available_channels is None
        else _array(available_channels, name="available_channels")
    )
    _validate_source_arrays(
        component_centroid_xy=centroids,
        component_centroid_valid=centroid_valid,
        component_labels=labels,
        available_channels=available,
        instance_key=keys,
        frame_indices=frames,
        source_acquisition_frame_index=acquisition_frames,
        source_row_ids=row_ids,
        source_row_signature=signatures,
        n_frames=n_frames,
    )
    role_indices = {role: labels.index(label) for role, label in role_mapping.items()}
    for role in _REQUIRED_ROLES:
        if not available[role_indices[role]]:
            raise MaskBodyFrameProducerError(
                f"Required source component for {role!r} is unavailable."
            )
    dimensions = BodyFrameDimensions(n_frames=n_frames, n_instances=rows)
    geometry = derive_mask_body_frame_geometry(
        centroids,
        centroid_valid,
        component_labels=labels,
        role_bindings=role_mapping,
        available_channels=available,
    )
    body_arrays = {
        "instance_key": _readonly(keys),
        "source_keypoint_row_ids": _readonly(row_ids),
        "source_keypoint_row_signature": _readonly(signatures),
        "frame_indices": _readonly(frames),
        "frame_row_offsets": _readonly(
            derive_frame_row_offsets(frames, n_frames=n_frames)
        ),
        **{name: _readonly(value) for name, value in geometry.items()},
    }
    # Validate the shared body-frame geometry and array contracts.  The schema's
    # source-keypoint evidence check is intentionally not used: this adapter
    # carries exact mask evidence under source_arrays and must not claim it is a
    # keypoint run.  The common source row invariants are checked above.
    schema_issues = BODY_FRAME_SCHEMA_V1.validate(
        body_arrays,
        dimensions=dimensions,
        source_keypoint_arrays=None,
    )
    schema_issues = tuple(
        issue
        for issue in schema_issues
        if issue.code != "missing_source_keypoint_evidence"
    )
    if schema_issues:
        detail = "; ".join(f"{issue.code}: {issue.message}" for issue in schema_issues)
        raise MaskBodyFrameProducerError(
            f"Mask body-frame arrays violate BODY_FRAME_SCHEMA_V1: {detail}"
        )

    source_arrays = {
        "instance_key": _readonly(keys),
        "frame_indices": _readonly(frames),
        "source_acquisition_frame_index": _readonly(acquisition_frames),
        "source_row_ids": _readonly(row_ids),
        "source_row_signature": _readonly(signatures),
        "component_labels": labels,
        "component_centroid_xy": _readonly(centroids),
        "component_centroid_valid": _readonly(centroid_valid),
        "available_channels": _readonly(available),
    }
    source = MaskBodyFrameSourceReference(
        run_path=run_path,
        run_manifest_digest=run_manifest_digest,
        mask_schema_id=mask_schema_id,
        mask_schema_version=mask_schema_version,
        mask_schema_digest=mask_schema_digest,
        anatomy_profile_id=profile.profile_id,
        anatomy_profile_version=profile.profile_version,
        anatomy_profile_digest=recipe.anatomy_profile_digest,
        binding_id=binding_id,
        binding_digest=recipe.binding_digest,
        row_identity_digest=row_identity_digest,
        instance_key_digest=sha256_array(keys),
        frame_indices_digest=sha256_array(frames),
        source_acquisition_frame_index_digest=sha256_array(acquisition_frames),
        source_row_ids_digest=sha256_array(row_ids),
        source_row_signature_digest=sha256_array(signatures),
    )
    return PreparedMaskBodyFrameSnapshot(
        dimensions=dimensions,
        source=source,
        recipe=recipe,
        arrays=body_arrays,
        source_arrays=source_arrays,
    )


prepare_mask_body_frame = prepare_refined_subject_mask_body_frame


__all__ = [
    "MASK_BODY_FRAME_COORDINATE_SOURCE",
    "MASK_BODY_FRAME_RECIPE_ID",
    "MASK_BODY_FRAME_RECIPE_PROFILE_RECIPE_ID",
    "MASK_BODY_FRAME_RECIPE_SCHEMA_ID",
    "MASK_BODY_FRAME_RECIPE_SCHEMA_VERSION",
    "MASK_BODY_FRAME_SOURCE_MODALITY",
    "MASK_BODY_FRAME_SOURCE_STAGE",
    "MaskBodyFrameProducerError",
    "MaskBodyFrameRecipe",
    "MaskBodyFrameSourceReference",
    "PreparedMaskBodyFrameSnapshot",
    "build_mask_body_frame_recipe",
    "derive_mask_body_frame_geometry",
    "prepare_mask_body_frame",
    "prepare_refined_subject_mask_body_frame",
]
