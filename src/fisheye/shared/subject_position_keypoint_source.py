"""Strict source adapter for subject-position keypoints.

This module is deliberately narrower than the keypoint publication writers.  It
consumes one explicitly named, current canonical keypoint coordinate run and
binds its source-camera landmark arrays to the anatomy-profile expression
interface.  It never resolves a historical run, an active production-bundle
member, or another measurement modality as a fallback.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import zarr

from fisheye.shared.anatomy_profile import AnatomyProfile, AnatomyProfileError
from fisheye.shared.canonical_coordinate_publication import (
    BoundCanonicalCoordinateDescriptor,
)
from fisheye.shared.coordinate_identity import (
    BoundRowIdentityContract,
    require_bound_row_identity_contract,
)
from fisheye.shared.keypoint_coordinate_publication import (
    BoundKeypointCoordinateContext,
    BoundKeypointCoordinateSurfaces,
    load_persisted_keypoint_coordinate_surfaces,
    require_bound_keypoint_coordinate_surfaces,
)
from fisheye.shared.pixel_frame_authority import (
    BoundPixelFrameAuthority,
    require_source_camera_pixel_frame_authority,
)
from fisheye.shared.subject_position_expression import (
    PointArrayBinding,
    PointExpressionBindings,
)
from fisheye.shared.zarr.keypoint_manifest import (
    KEYPOINT_RUN_MANIFEST_ATTRIBUTE,
    keypoint_metadata_declarations_digest,
    keypoint_skeleton_digest,
    validate_keypoint_run_manifest,
)
from fisheye.shared.zarr.keypoint_publication import (
    keypoint_metadata_declaration_maps,
)
from fisheye.shared.zarr.keypoint_schema import (
    KEYPOINT_SCHEMA_V2,
    KeypointDimensions,
)
from fisheye.shared.zarr.keypoint_storage import plan_keypoint_storage
from fisheye.shared.zarr.metadata_equivalence import (
    validate_direct_consolidated_subtree,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.storage_profiles import storage_profile_from_manifest
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_CONTRACT,
    RUN_COMPLETION_CONTRACT_ATTR,
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
    is_run_complete_in_parent,
)
from fisheye.shared.coordinate_surface_contract import (
    SOURCE_CAMERA_POINT_PIXEL_CONVENTION,
    SOURCE_CAMERA_PROFILE_ID,
)


SOURCE_MODALITY = "keypoint"
SOURCE_KIND = "canonical_keypoint_coordinate_selector"
CANONICAL_COORDINATE_CONTRACT = "canonical_v2"

_BOUND_SOURCE_SEAL = object()
_REQUIRED_SOURCE_CROP_ARRAYS = (
    "instance_key",
    "frame_indices",
    "source_acquisition_frame_index",
    "source_row_signature",
    "roi_coordinates_full",
    "roi_sizes_full",
)


class KeypointPositionSourceError(ValueError):
    """Raised when a keypoint source cannot prove the complete authority chain."""


@dataclass(frozen=True)
class KeypointPositionSourcePolicy:
    """Explicit source-selection policy for one anatomy-profile binding."""

    anatomy_profile: AnatomyProfile | Mapping[str, Any]
    binding_id: str


@dataclass(frozen=True, init=False)
class BoundKeypointPositionSource:
    """Sealed, row-aligned keypoint source in source-camera coordinates."""

    source_modality: str
    source_kind: str
    run_path: str
    run_id: str
    row_identity: BoundRowIdentityContract
    instance_key: np.ndarray
    source_acquisition_frame_index: np.ndarray
    source_row_index: np.ndarray
    source_camera_frame: BoundPixelFrameAuthority
    source_binding_record: Mapping[str, Any]
    source_binding_digest: str
    expression_bindings: PointExpressionBindings
    keypoints_roi: np.ndarray
    keypoints_img: np.ndarray
    keypoint_valid: np.ndarray
    keypoint_confidences: np.ndarray
    confidence_valid: None
    skeleton_id: str
    skeleton_digest: str
    pose_schema_binding_digest: str
    coordinate_descriptor: BoundCanonicalCoordinateDescriptor
    coordinate_context: BoundKeypointCoordinateContext
    run_manifest_digest: str
    logical_content_digest: str
    metadata_declarations_digest: str
    _analysis_zarr: str | Path | Any = field(repr=False, compare=False)
    _anatomy_profile: AnatomyProfile = field(repr=False, compare=False)
    _binding_id: str = field(repr=False, compare=False)
    _seal: object = field(repr=False, compare=False)

    def __init__(self, *, _verification_seal: object | None = None, **values: Any) -> None:
        if _verification_seal is not _BOUND_SOURCE_SEAL:
            raise KeypointPositionSourceError(
                "Bound keypoint sources must be produced by the strict loader."
            )
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_seal", _verification_seal)

    def revalidate(self) -> "BoundKeypointPositionSource":
        """Reload every persisted authority before downstream consumption."""

        return revalidate_bound_keypoint_position_source(self)

    @property
    def anatomy_profile(self) -> AnatomyProfile:
        """Exact anatomy authority used to seal this source."""

        return self._anatomy_profile

    @property
    def binding_id(self) -> str:
        """Exact anatomy source-binding identity used by this adapter."""

        return self._binding_id


def _read_array(node: Any, *, name: str) -> np.ndarray:
    try:
        value = node[...] if hasattr(node, "__getitem__") else node
    except Exception as exc:  # pragma: no cover - backend-specific detail
        raise KeypointPositionSourceError(
            f"Unable to read keypoint array {name!r}: {exc}."
        ) from exc
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


def _canonical_run_path(run_path: str) -> tuple[str, str]:
    if type(run_path) is not str or not run_path or run_path.startswith("/"):
        raise KeypointPositionSourceError(
            "run_path must be an explicit relative keypoints_runs/<run> path."
        )
    parts = run_path.split("/")
    if len(parts) != 2 or parts[0] != "keypoints_runs" or not parts[1]:
        raise KeypointPositionSourceError(
            "run_path must name exactly one keypoints_runs/<run> child."
        )
    if parts[1] in {".", ".."}:
        raise KeypointPositionSourceError("run_path contains an unsafe run name.")
    return run_path, parts[1]


def _open_root(analysis_zarr: str | Path | Any) -> Any:
    if isinstance(analysis_zarr, (str, Path)):
        try:
            return zarr.open_group(
                str(Path(analysis_zarr).expanduser().resolve()),
                mode="r",
                zarr_format=3,
                use_consolidated=False,
            )
        except Exception as exc:  # pragma: no cover - backend-specific detail
            raise KeypointPositionSourceError(
                f"Unable to open analysis Zarr directly: {exc}."
            ) from exc
    return analysis_zarr


def _require_group(root: Any, path: str, *, label: str) -> Any:
    try:
        return root[path]
    except Exception as exc:
        raise KeypointPositionSourceError(
            f"Missing {label} at {path!r}: {exc}."
        ) from exc


def _require_current_selector(
    root: Any,
    parent: Any,
    run: Any,
    *,
    run_path: str,
    run_id: str,
) -> None:
    attrs = getattr(run, "attrs", {})
    if attrs.get(RUN_COMPLETION_CONTRACT_ATTR) != RUN_COMPLETION_CONTRACT:
        raise KeypointPositionSourceError(
            "Keypoint source lacks the exact Palette completion contract."
        )
    if attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE:
        raise KeypointPositionSourceError("Keypoint source is not complete.")
    if attrs.get("stage_selector_eligible") is not True:
        raise KeypointPositionSourceError(
            "Keypoint source is not selector-eligible for subject positions."
        )

    parent_attrs = getattr(parent, "attrs", {})
    root_attrs = getattr(root, "attrs", {})
    if root_attrs.get("current_keypoint_group_path") != run_path:
        raise KeypointPositionSourceError(
            "The named keypoint run is not the root current_keypoint_group_path."
        )
    if (
        parent_attrs.get("latest") != run_id
        or parent_attrs.get("latest_complete") != run_id
        or "latest_pending" in parent_attrs
    ):
        raise KeypointPositionSourceError(
            "The named keypoint run does not own the complete current selector pair."
        )
    try:
        complete = is_run_complete_in_parent(parent, run)
    except Exception as exc:
        raise KeypointPositionSourceError(
            f"Unable to validate parent-scoped keypoint completion: {exc}."
        ) from exc
    if complete is not True:
        raise KeypointPositionSourceError(
            "The named keypoint run is not complete under its parent policy."
        )


def _manifest_and_payload(run: Any, *, run_id: str) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    manifest = getattr(run, "attrs", {}).get(KEYPOINT_RUN_MANIFEST_ATTRIBUTE)
    if not isinstance(manifest, Mapping):
        raise KeypointPositionSourceError("Keypoint source lacks its run manifest.")
    errors = validate_keypoint_run_manifest(manifest)
    if errors:
        raise KeypointPositionSourceError(
            "Keypoint run manifest is invalid: " + "; ".join(errors)
        )
    payload = manifest.get("payload")
    if not isinstance(payload, Mapping) or payload.get("run_id") != run_id:
        raise KeypointPositionSourceError(
            "Keypoint run manifest does not bind the requested run ID."
        )
    return manifest, payload


def _dimensions(payload: Mapping[str, Any]) -> KeypointDimensions:
    try:
        raw = payload["logical_schema"]["dimensions"]
        return KeypointDimensions(
            n_frames=raw["n_frames"],
            n_instances=raw["n_instances"],
            n_keypoints=raw["n_keypoints"],
            source_width=raw["source_width"],
            source_height=raw["source_height"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise KeypointPositionSourceError(
            f"Keypoint manifest dimensions are invalid: {exc}."
        ) from exc


def _validate_published_metadata(
    analysis_zarr: str | Path | Any,
    *,
    run_path: str,
    run_id: str,
    payload: Mapping[str, Any],
) -> str:
    if not isinstance(analysis_zarr, (str, Path)):
        raise KeypointPositionSourceError(
            "Strict direct/consolidated validation requires the analysis Zarr path."
        )
    archive = Path(analysis_zarr).expanduser().resolve()
    try:
        validate_direct_consolidated_subtree(archive, subtree_path=run_path)
        dimensions = _dimensions(payload)
        storage = payload["storage_plan"]
        profile = storage_profile_from_manifest(storage["storage_profile"])
        plans = plan_keypoint_storage(dimensions, profile=profile)
        direct, consolidated = keypoint_metadata_declaration_maps(
            archive,
            run_id=run_id,
            plans=plans,
        )
        observed = keypoint_metadata_declarations_digest(
            direct,
            consolidated_by_path=consolidated,
        )
        expected = payload["publication"]["metadata_declarations_digest"]
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise KeypointPositionSourceError(
            f"Keypoint direct/consolidated metadata is invalid: {exc}."
        ) from exc
    if observed != expected:
        raise KeypointPositionSourceError(
            "Keypoint metadata declaration digest differs from the run manifest."
        )
    return observed


def _validate_source_camera_surface(
    surfaces: BoundKeypointCoordinateSurfaces,
) -> tuple[BoundCanonicalCoordinateDescriptor, BoundPixelFrameAuthority]:
    surface = surfaces.keypoints_img
    descriptor = surface.descriptor
    if (
        descriptor.profile_id != SOURCE_CAMERA_PROFILE_ID
        or descriptor.space_id != "source_camera_image_px"
        or descriptor.geometry_type != "point_xy"
        or descriptor.pixel_convention != SOURCE_CAMERA_POINT_PIXEL_CONVENTION
    ):
        raise KeypointPositionSourceError(
            "Canonical keypoints_img is not the source-camera continuous point surface."
        )
    frame = surface.reference_frame_authority
    if frame is None:
        raise KeypointPositionSourceError(
            "Canonical keypoints_img lacks source-camera frame authority."
        )
    try:
        return surface, require_source_camera_pixel_frame_authority(frame)
    except Exception as exc:
        raise KeypointPositionSourceError(
            f"Canonical keypoints_img source-camera frame is invalid: {exc}."
        ) from exc


def _resolve_anatomy_binding(
    policy: KeypointPositionSourcePolicy,
) -> tuple[AnatomyProfile, dict[str, Any]]:
    try:
        profile = (
            policy.anatomy_profile
            if isinstance(policy.anatomy_profile, AnatomyProfile)
            else AnatomyProfile.from_mapping(policy.anatomy_profile)
        )
        binding = profile.binding(policy.binding_id)
    except (AnatomyProfileError, TypeError, ValueError) as exc:
        raise KeypointPositionSourceError(
            f"Anatomy source binding is invalid: {exc}."
        ) from exc
    if binding["source_schema"].get("modality") != SOURCE_MODALITY:
        raise KeypointPositionSourceError(
            "The requested AnatomyProfile binding is not a keypoint source."
        )
    return profile, binding


def _bind_source_schema(
    binding: Mapping[str, Any],
    pose_binding: Mapping[str, Any],
) -> tuple[str, str, tuple[str, ...], dict[str, int]]:
    source_schema = binding.get("source_schema")
    pose_schema = pose_binding.get("pose_schema")
    package = source_schema.get("package_payload") if isinstance(source_schema, Mapping) else None
    if not isinstance(source_schema, Mapping) or not isinstance(package, Mapping):
        raise KeypointPositionSourceError(
            "Keypoint AnatomyProfile binding lacks an exact pose-schema package."
        )
    if not isinstance(pose_schema, Mapping):
        raise KeypointPositionSourceError("Keypoint run lacks its bound pose schema.")
    for field_name in ("skeleton_id", "kpt_shape", "keypoint_labels", "nodes", "edges"):
        if pose_schema.get(field_name) != package.get(field_name):
            raise KeypointPositionSourceError(
                f"Anatomy source binding disagrees with pose schema field {field_name!r}."
            )
    labels = tuple(str(item) for item in pose_schema["keypoint_labels"])
    role_to_index: dict[str, int] = {}
    for item in binding["role_bindings"]:
        role_id = item["role_id"]
        source_label = item["source_label"]
        if source_label not in labels:
            raise KeypointPositionSourceError(
                f"Anatomy role {role_id!r} maps to missing keypoint label {source_label!r}."
            )
        if role_id in role_to_index:
            raise KeypointPositionSourceError(
                f"Anatomy role {role_id!r} is bound more than once."
            )
        role_to_index[role_id] = labels.index(source_label)
    return (
        str(pose_schema["skeleton_id"]),
        str(pose_binding["binding_sha256"]),
        labels,
        role_to_index,
    )


def _require_schema(
    arrays: Mapping[str, np.ndarray],
    *,
    dimensions: KeypointDimensions,
    source_crop_arrays: Mapping[str, Any],
    skeleton_digest: str,
) -> None:
    try:
        KEYPOINT_SCHEMA_V2.require(
            arrays,
            dimensions=dimensions,
            source_crop_arrays=source_crop_arrays,
            skeleton_digest=skeleton_digest,
        )
    except Exception as exc:
        raise KeypointPositionSourceError(
            f"Keypoint V2 schema validation failed: {exc}."
        ) from exc


def _readonly_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    def freeze(item: Any) -> Any:
        if isinstance(item, Mapping):
            return MappingProxyType(
                {str(key): freeze(child) for key, child in item.items()}
            )
        if isinstance(item, (list, tuple)):
            return tuple(freeze(child) for child in item)
        return deepcopy(item)

    return freeze(value)


def _build_expression_bindings(
    *,
    role_to_index: Mapping[str, int],
    keypoints_img: np.ndarray,
    keypoint_valid: np.ndarray,
    keypoint_confidences: np.ndarray,
) -> PointExpressionBindings:
    bindings: dict[str, PointArrayBinding] = {}
    for role_id, index in role_to_index.items():
        bindings[role_id] = PointArrayBinding(
            values=keypoints_img[:, index, :],
            valid=keypoint_valid[:, index],
            # Raw keypoint v2 has confidence values but no independent
            # confidence-valid authority. Preserve the diagnostic source array
            # on the bound handle without inventing a row-validity rule.
            confidence=None,
            confidence_valid=None,
        )
    return PointExpressionBindings(keypoints=MappingProxyType(bindings))


def load_bound_keypoint_position_source(
    analysis_zarr: str | Path | Any,
    *,
    run_path: str,
    policy: KeypointPositionSourcePolicy,
) -> BoundKeypointPositionSource:
    """Load one exact current canonical keypoint position source.

    ``run_path`` is intentionally mandatory.  The loader accepts an already
    opened root for deterministic in-memory tests, but real published reads
    must pass the archive path so direct/consolidated metadata can be checked.
    """

    normalized_path, run_id = _canonical_run_path(run_path)
    if not isinstance(policy, KeypointPositionSourcePolicy):
        raise KeypointPositionSourceError("A strict keypoint source policy is required.")
    profile, binding = _resolve_anatomy_binding(policy)
    root = _open_root(analysis_zarr)
    parent = _require_group(root, "keypoints_runs", label="keypoint parent")
    run = _require_group(root, normalized_path, label="keypoint run")
    _require_current_selector(
        root,
        parent,
        run,
        run_path=normalized_path,
        run_id=run_id,
    )
    manifest, payload = _manifest_and_payload(run, run_id=run_id)
    metadata_digest = _validate_published_metadata(
        analysis_zarr,
        run_path=normalized_path,
        run_id=run_id,
        payload=payload,
    )

    try:
        pose_binding = payload["pose_model_schema_binding"]
        skeleton_digest = keypoint_skeleton_digest(pose_binding)
        expected_skeleton_digest = payload["logical_content"]["document"]["skeleton_digest"]
        if skeleton_digest != expected_skeleton_digest:
            raise KeypointPositionSourceError(
                "Keypoint skeleton digest differs from logical-content authority."
            )
        skeleton_id, pose_binding_digest, labels, role_to_index = _bind_source_schema(
            binding,
            pose_binding,
        )
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, KeypointPositionSourceError):
            raise
        raise KeypointPositionSourceError(
            f"Keypoint skeleton authority is invalid: {exc}."
        ) from exc

    arrays = {
        name: _read_array(_require_group(run, name, label="keypoint array"), name=name)
        for name in KEYPOINT_SCHEMA_V2.binding_paths
    }
    dimensions = _dimensions(payload)
    surfaces = require_bound_keypoint_coordinate_surfaces(
        load_persisted_keypoint_coordinate_surfaces(root, normalized_path)
    )
    coordinate_descriptor, source_camera_frame = _validate_source_camera_surface(surfaces)
    keypoints_img = _read_array(
        coordinate_descriptor.coordinate_node,
        name="canonical keypoints_img",
    )
    if not np.array_equal(keypoints_img, arrays["keypoints_img"], equal_nan=True):
        raise KeypointPositionSourceError(
            "Canonical keypoints_img differs from the keypoint V2 source array."
        )
    source_crop = surfaces.context.source._rowset_node
    source_crop_arrays = {
        name: source_crop[name] for name in _REQUIRED_SOURCE_CROP_ARRAYS
    }
    _require_schema(
        arrays,
        dimensions=dimensions,
        source_crop_arrays=source_crop_arrays,
        skeleton_digest=skeleton_digest,
    )

    row_identity = require_bound_row_identity_contract(surfaces.context.row_identity)
    n_rows = row_identity.leading_dimension
    instance_key = arrays["instance_key"]
    acquisition_frames = arrays["source_acquisition_frame_index"]
    if instance_key.dtype != np.dtype("<u8") or instance_key.ndim != 1:
        raise KeypointPositionSourceError("instance_key must be exact uint64[N].")
    if acquisition_frames.dtype != np.dtype("<i8") or acquisition_frames.ndim != 1:
        raise KeypointPositionSourceError(
            "source_acquisition_frame_index must be exact int64[N]."
        )
    if instance_key.shape != (n_rows,) or acquisition_frames.shape != (n_rows,):
        raise KeypointPositionSourceError(
            "Keypoint rows do not match the sealed row-identity leading dimension."
        )
    source_row_index = np.arange(n_rows, dtype=np.int64)
    source_row_index.setflags(write=False)

    keypoints_roi = arrays["keypoints_roi"]
    keypoint_valid = arrays["keypoint_valid"]
    keypoint_confidences = arrays["keypoint_confidences"]
    expression_bindings = _build_expression_bindings(
        role_to_index=role_to_index,
        keypoints_img=keypoints_img,
        keypoint_valid=keypoint_valid,
        keypoint_confidences=keypoint_confidences,
    )
    logical_content = payload["logical_content"]
    source = BoundKeypointPositionSource(
        source_modality=SOURCE_MODALITY,
        source_kind=SOURCE_KIND,
        run_path=normalized_path,
        run_id=run_id,
        row_identity=row_identity,
        instance_key=instance_key,
        source_acquisition_frame_index=acquisition_frames,
        source_row_index=source_row_index,
        source_camera_frame=source_camera_frame,
        source_binding_record=_readonly_mapping(binding),
        source_binding_digest=str(binding["binding_sha256"]),
        expression_bindings=expression_bindings,
        keypoints_roi=keypoints_roi,
        keypoints_img=keypoints_img,
        keypoint_valid=keypoint_valid,
        keypoint_confidences=keypoint_confidences,
        confidence_valid=None,
        skeleton_id=skeleton_id,
        skeleton_digest=skeleton_digest,
        pose_schema_binding_digest=pose_binding_digest,
        coordinate_descriptor=coordinate_descriptor,
        coordinate_context=surfaces.context,
        run_manifest_digest=canonical_json_sha256(manifest),
        logical_content_digest=str(logical_content["digest"]),
        metadata_declarations_digest=metadata_digest,
        _analysis_zarr=analysis_zarr,
        _anatomy_profile=profile,
        _binding_id=policy.binding_id,
        _verification_seal=_BOUND_SOURCE_SEAL,
    )
    return source


def _same_array(left: np.ndarray, right: np.ndarray) -> bool:
    return left.dtype == right.dtype and np.array_equal(left, right, equal_nan=True)


def revalidate_bound_keypoint_position_source(
    source: BoundKeypointPositionSource,
) -> BoundKeypointPositionSource:
    """Reload and compare every persisted authority behind a bound source."""

    if (
        type(source) is not BoundKeypointPositionSource
        or source._seal is not _BOUND_SOURCE_SEAL
    ):
        raise KeypointPositionSourceError("A sealed keypoint position source is required.")
    current = load_bound_keypoint_position_source(
        source._analysis_zarr,
        run_path=source.run_path,
        policy=KeypointPositionSourcePolicy(
            anatomy_profile=source._anatomy_profile,
            binding_id=source._binding_id,
        ),
    )
    for name in (
        "run_path",
        "run_id",
        "source_binding_digest",
        "skeleton_id",
        "skeleton_digest",
        "pose_schema_binding_digest",
        "run_manifest_digest",
        "logical_content_digest",
        "metadata_declarations_digest",
    ):
        if getattr(current, name) != getattr(source, name):
            raise KeypointPositionSourceError(
                f"Keypoint source authority changed at {name}."
            )
    if current.row_identity.contract.digest() != source.row_identity.contract.digest():
        raise KeypointPositionSourceError("Keypoint source row identity changed.")
    if current.source_camera_frame.record_sha256 != source.source_camera_frame.record_sha256:
        raise KeypointPositionSourceError("Keypoint source camera frame authority changed.")
    if current.coordinate_descriptor.descriptor.digest() != source.coordinate_descriptor.descriptor.digest():
        raise KeypointPositionSourceError("Keypoint source coordinate descriptor changed.")
    for name in (
        "instance_key",
        "source_acquisition_frame_index",
        "source_row_index",
        "keypoints_roi",
        "keypoints_img",
        "keypoint_valid",
        "keypoint_confidences",
    ):
        if not _same_array(getattr(current, name), getattr(source, name)):
            raise KeypointPositionSourceError(f"Keypoint source array changed at {name}.")
    return source


def require_bound_keypoint_position_source(
    source: BoundKeypointPositionSource,
) -> BoundKeypointPositionSource:
    """Require and revalidate a sealed source before downstream consumption."""

    return revalidate_bound_keypoint_position_source(source)


__all__ = [
    "BoundKeypointPositionSource",
    "KeypointPositionSourceError",
    "KeypointPositionSourcePolicy",
    "SOURCE_KIND",
    "SOURCE_MODALITY",
    "load_bound_keypoint_position_source",
    "revalidate_bound_keypoint_position_source",
    "require_bound_keypoint_position_source",
]
