"""Strict source adapter for subject-position keypoints.

This module is deliberately narrower than the keypoint publication writers.  It
consumes one explicitly named keypoint coordinate run and binds its source-camera
landmark arrays to the anatomy-profile expression interface.  The default mode
requires the current canonical selector.  An explicitly named canary mode may
bind one sealed raw-keypoint member from the root bundle authority; it never
falls back between those authorities or to another measurement modality.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
import re
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
    load_persisted_ineligible_keypoint_coordinate_surfaces,
    require_bound_ineligible_keypoint_coordinate_surfaces,
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
    keypoint_skeleton_document,
    validate_keypoint_run_manifest,
)
from fisheye.shared.zarr.keypoint_bundle_activation import (
    KEYPOINT_BUNDLE_AUTHORITY_ATTR,
    KEYPOINT_BUNDLE_AUTHORITY_GENERATION_ATTR,
    KEYPOINT_BUNDLE_AUTHORITY_LEASE_ATTR,
    KEYPOINT_BUNDLE_AUTHORITY_SCHEMA_ID,
    KEYPOINT_BUNDLE_AUTHORITY_SCHEMA_VERSION,
    resolve_active_keypoint_bundle_from_root,
)
from fisheye.shared.zarr.keypoint_publication_mode import (
    ATOMIC_PUBLICATION_OWNER_ATTR,
)
from fisheye.shared.zarr.keypoint_publication import (
    keypoint_metadata_declaration_maps,
)
from fisheye.shared.zarr.coordinate_successor_authority import (
    KEYPOINT_COORDINATE_SUCCESSOR_KIND,
    CoordinateSuccessorAuthorityError,
    load_coordinate_successor_authority,
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
from fisheye.shared.zarr_io import open_zarr_root
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
SEALED_BUNDLE_SOURCE_KIND = "sealed_keypoint_bundle_member_canary"
COORDINATE_SUCCESSOR_SOURCE_KIND = "sealed_keypoint_coordinate_successor_canary"
CANONICAL_COORDINATE_CONTRACT = "canonical_v2"
KEYPOINT_AUTHORITY_MODE_CANONICAL_SELECTOR = "canonical_selector"
KEYPOINT_AUTHORITY_MODE_SEALED_BUNDLE_CANARY = "sealed_keypoint_bundle_canary_v1"
KEYPOINT_AUTHORITY_MODE_COORDINATE_SUCCESSOR_CANARY = (
    "sealed_keypoint_coordinate_successor_canary_v1"
)
SEALED_BUNDLE_PRODUCTION_SELECTOR_ACTIVATION = (
    "deferred_separate_reviewed_change"
)
_PUBLICATION_OWNER_UUID = re.compile(r"^[0-9a-f]{32}$")
_SELECTOR_ALIAS_NAMES = frozenset(
    {"latest", "latest_complete", "latest_pending", "authoritative_run"}
)

_BOUND_SOURCE_SEAL = object()
_BOUND_COORDINATE_SUCCESSOR_ADMISSION_SEAL = object()
_BOUND_COORDINATE_SUCCESSOR_SEAL = object()
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
    authority_mode: str = KEYPOINT_AUTHORITY_MODE_CANONICAL_SELECTOR


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
    authority_mode: str
    keypoint_bundle_authority: Mapping[str, Any] | None
    keypoint_bundle_authority_digest: str | None
    coordinate_successor_authority: Mapping[str, Any] | None
    coordinate_successor_authority_digest: str | None
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


@dataclass(frozen=True, init=False)
class BoundKeypointCoordinateSuccessorAdmission:
    """Metadata-only admission for one sealed coordinate successor.

    The admission proves the same manifest, successor authority, lifecycle,
    selector, and direct/consolidated metadata grammar as the full source
    loader.  It deliberately does not open or derive coordinate payload
    surfaces.  Consumers that need scientific values must use
    :func:`load_keypoint_coordinate_successor_source` instead.
    """

    analysis_zarr: Path
    run_path: str
    run_id: str
    run_group: Any = field(repr=False, compare=False)
    manifest: Mapping[str, Any] = field(repr=False)
    manifest_digest: str
    metadata_declarations_digest: str
    successor_authority: Mapping[str, Any] = field(repr=False)
    successor_authority_digest: str
    active_keypoint_bundle_authority: Mapping[str, Any] = field(repr=False)
    active_keypoint_bundle_authority_digest: str
    _root: Any = field(repr=False, compare=False)
    _seal: object = field(repr=False, compare=False)

    def __init__(
        self,
        *,
        _verification_seal: object | None = None,
        **values: Any,
    ) -> None:
        if _verification_seal is not _BOUND_COORDINATE_SUCCESSOR_ADMISSION_SEAL:
            raise KeypointPositionSourceError(
                "Coordinate-successor admissions must be produced by the shared "
                "resolver."
            )
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_seal", _verification_seal)


@dataclass(frozen=True, init=False)
class BoundKeypointCoordinateSuccessorSource:
    """Full-strength, selector-ineligible coordinate-successor profile."""

    analysis_zarr: Path
    run_path: str
    run_id: str
    run_group: Any = field(repr=False, compare=False)
    manifest: Mapping[str, Any] = field(repr=False)
    manifest_digest: str
    surfaces: BoundKeypointCoordinateSurfaces = field(repr=False, compare=False)
    successor_authority: Mapping[str, Any] = field(repr=False)
    successor_authority_digest: str
    active_keypoint_bundle_authority: Mapping[str, Any] = field(repr=False)
    active_keypoint_bundle_authority_digest: str
    _seal: object = field(repr=False, compare=False)

    def __init__(
        self,
        *,
        _verification_seal: object | None = None,
        **values: Any,
    ) -> None:
        if _verification_seal is not _BOUND_COORDINATE_SUCCESSOR_SEAL:
            raise KeypointPositionSourceError(
                "Coordinate-successor sources must be produced by the shared resolver."
            )
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_seal", _verification_seal)


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


def _validate_bundle_authority_direct_consolidated(
    analysis_zarr: str | Path | Any,
    *,
    authority: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Require the root authority to agree in direct and consolidated metadata."""

    if not isinstance(analysis_zarr, (str, Path)):
        raise KeypointPositionSourceError(
            "Sealed keypoint bundle authority requires the analysis Zarr path "
            "for direct/consolidated root validation."
        )
    archive = Path(analysis_zarr).expanduser().resolve()
    try:
        direct_root = open_zarr_root(
            archive,
            mode="r",
            use_consolidated=False,
        )
        consolidated_root = open_zarr_root(
            archive,
            mode="r",
            use_consolidated=True,
        )
    except Exception as exc:
        raise KeypointPositionSourceError(
            f"Unable to open direct/consolidated root authority: {exc}."
        ) from exc
    direct_attrs = getattr(direct_root, "attrs", {})
    consolidated_attrs = getattr(consolidated_root, "attrs", {})
    for name in (
        KEYPOINT_BUNDLE_AUTHORITY_ATTR,
        KEYPOINT_BUNDLE_AUTHORITY_GENERATION_ATTR,
        KEYPOINT_BUNDLE_AUTHORITY_LEASE_ATTR,
    ):
        if (
            (name in direct_attrs) != (name in consolidated_attrs)
            or direct_attrs.get(name) != consolidated_attrs.get(name)
        ):
            raise KeypointPositionSourceError(
                "Direct and consolidated root keypoint bundle authority state differs."
            )
    direct_authority = direct_attrs.get(KEYPOINT_BUNDLE_AUTHORITY_ATTR)
    consolidated_authority = consolidated_attrs.get(KEYPOINT_BUNDLE_AUTHORITY_ATTR)
    if direct_authority != consolidated_authority:
        raise KeypointPositionSourceError(
            "Direct and consolidated root keypoint_bundle_authority differ."
        )
    if KEYPOINT_BUNDLE_AUTHORITY_LEASE_ATTR in direct_attrs:
        raise KeypointPositionSourceError(
            "The root keypoint bundle authority is still under an activation lease."
        )
    if direct_authority != dict(authority):
        raise KeypointPositionSourceError(
            "Root keypoint_bundle_authority differs from the opened authority."
        )
    return dict(direct_authority)


def _require_sealed_bundle_authority(
    root: Any,
    analysis_zarr: str | Path | Any,
    *,
    run: Any,
    parent: Any,
    run_path: str,
    run_id: str,
    manifest: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> tuple[Mapping[str, Any], str]:
    """Bind one raw keypoint member from the committed root bundle authority."""

    if run_id in _SELECTOR_ALIAS_NAMES:
        raise KeypointPositionSourceError(
            "Sealed keypoint bundle canaries require a concrete run ID; selector "
            "aliases are forbidden."
        )
    try:
        resolved = resolve_active_keypoint_bundle_from_root(root)
    except Exception as exc:
        raise KeypointPositionSourceError(
            f"Sealed keypoint bundle authority is invalid: {exc}."
        ) from exc
    if not isinstance(resolved, Mapping):
        raise KeypointPositionSourceError(
            "The root has no committed keypoint bundle authority."
        )
    authority = resolved.get("authority")
    if not isinstance(authority, Mapping):
        raise KeypointPositionSourceError(
            "The root keypoint bundle authority is not one object."
        )
    if (
        authority.get("schema_id") != KEYPOINT_BUNDLE_AUTHORITY_SCHEMA_ID
        or authority.get("schema_version")
        != KEYPOINT_BUNDLE_AUTHORITY_SCHEMA_VERSION
    ):
        raise KeypointPositionSourceError(
            "The root keypoint bundle authority schema or version is unsupported."
        )
    persisted_authority = _validate_bundle_authority_direct_consolidated(
        analysis_zarr,
        authority=authority,
    )
    if persisted_authority != dict(authority):
        raise KeypointPositionSourceError(
            "The persisted root keypoint bundle authority changed during binding."
        )

    members = authority.get("members")
    raw_member = members.get("raw_keypoints") if isinstance(members, Mapping) else None
    expected_member_fields = {
        "role",
        "run_id",
        "run_path",
        "publication_owner_uuid",
        "manifest_payload_digest",
        "manifest_document_digest",
        "logical_content_digest",
    }
    if not isinstance(raw_member, Mapping) or set(raw_member) != expected_member_fields:
        raise KeypointPositionSourceError(
            "The root keypoint bundle raw_keypoints member is malformed."
        )
    if (
        raw_member.get("role") != "raw_keypoints"
        or raw_member.get("run_id") != run_id
        or raw_member.get("run_path") != run_path
        or run_path != f"keypoints_runs/{run_id}"
    ):
        raise KeypointPositionSourceError(
            "The root authority does not name the exact requested raw keypoint run."
        )

    logical_content = payload.get("logical_content")
    logical_digest = (
        logical_content.get("digest")
        if isinstance(logical_content, Mapping)
        else None
    )
    manifest_payload_digest = manifest.get("payload_digest")
    manifest_document_digest = canonical_json_sha256(manifest)
    if (
        raw_member.get("manifest_payload_digest") != manifest_payload_digest
        or raw_member.get("manifest_document_digest") != manifest_document_digest
        or raw_member.get("logical_content_digest") != logical_digest
    ):
        raise KeypointPositionSourceError(
            "The root raw keypoint member does not bind the exact manifest or "
            "logical-content digest."
        )

    attrs = getattr(run, "attrs", {})
    if (
        attrs.get(RUN_COMPLETION_CONTRACT_ATTR) != RUN_COMPLETION_CONTRACT
        or attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE
        or attrs.get("status") != RUN_STATUS_COMPLETE
        or attrs.get("palette_run_completion_status") != RUN_STATUS_COMPLETE
        or attrs.get("stage_selector_eligible") is not False
        or attrs.get("production_candidate") is not True
        or attrs.get("production_selector_activation")
        != SEALED_BUNDLE_PRODUCTION_SELECTOR_ACTIVATION
    ):
        raise KeypointPositionSourceError(
            "The raw keypoint bundle member is not a sealed selector-ineligible "
            "production candidate."
        )
    owner = attrs.get(ATOMIC_PUBLICATION_OWNER_ATTR)
    if (
        type(owner) is not str
        or _PUBLICATION_OWNER_UUID.fullmatch(owner) is None
        or raw_member.get("publication_owner_uuid") != owner
    ):
        raise KeypointPositionSourceError(
            "The raw keypoint bundle member publication owner is invalid or stale."
        )
    try:
        complete = is_run_complete_in_parent(parent, run)
    except Exception as exc:
        raise KeypointPositionSourceError(
            f"Unable to validate raw keypoint bundle completion: {exc}."
        ) from exc
    if complete is not True:
        raise KeypointPositionSourceError(
            "The raw keypoint bundle member is not complete under its parent policy."
        )

    parent_attrs = getattr(parent, "attrs", {})
    root_attrs = getattr(root, "attrs", {})
    selector_refs = (
        parent_attrs.get("latest"),
        parent_attrs.get("latest_complete"),
        parent_attrs.get("latest_pending"),
        parent_attrs.get("authoritative_run"),
    )
    if run_id in selector_refs or root_attrs.get("current_keypoint_group_path") == run_path:
        raise KeypointPositionSourceError(
            "The sealed raw keypoint member is referenced by a family selector."
        )
    return _readonly_mapping(authority), canonical_json_sha256(authority)


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


def _require_authority_mode(value: Any) -> str:
    if value not in {
        KEYPOINT_AUTHORITY_MODE_CANONICAL_SELECTOR,
        KEYPOINT_AUTHORITY_MODE_SEALED_BUNDLE_CANARY,
        KEYPOINT_AUTHORITY_MODE_COORDINATE_SUCCESSOR_CANARY,
    }:
        raise KeypointPositionSourceError(
            "Unsupported keypoint position authority mode; use the canonical "
            "selector, explicit sealed-bundle canary, or explicit coordinate-"
            "successor canary mode."
        )
    return str(value)


def _require_coordinate_successor_authority(
    root: Any,
    analysis_zarr: str | Path | Any,
    *,
    run: Any,
    parent: Any,
    run_path: str,
    run_id: str,
    manifest: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> tuple[Mapping[str, Any], str]:
    """Bind an ineligible successor back to its still-current source authority."""

    try:
        authority = load_coordinate_successor_authority(
            run,
            expected_kind=KEYPOINT_COORDINATE_SUCCESSOR_KIND,
            expected_successor_run_path=run_path,
        )
    except CoordinateSuccessorAuthorityError as exc:
        raise KeypointPositionSourceError(
            f"Keypoint coordinate-successor authority is invalid: {exc}."
        ) from exc
    authority_payload = authority["payload"]
    source_binding = authority_payload["source"]
    source_path = source_binding["run_path"]
    try:
        source_run = root[source_path]
    except Exception as exc:
        raise KeypointPositionSourceError(
            "The immutable keypoint coordinate-successor source is absent."
        ) from exc
    source_manifest = getattr(source_run, "attrs", {}).get(
        KEYPOINT_RUN_MANIFEST_ATTRIBUTE
    )
    if not isinstance(source_manifest, Mapping):
        raise KeypointPositionSourceError(
            "The keypoint coordinate-successor source manifest is absent."
        )
    source_logical = source_manifest.get("payload", {}).get("logical_content", {})
    if (
        source_manifest.get("payload_digest")
        != source_binding.get("manifest_payload_digest")
        or canonical_json_sha256(source_manifest)
        != source_binding.get("manifest_document_digest")
        or source_logical.get("digest")
        != source_binding.get("logical_content_digest")
    ):
        raise KeypointPositionSourceError(
            "The keypoint coordinate-successor source manifest changed."
        )
    successor_logical = payload.get("logical_content")
    if (
        not isinstance(successor_logical, Mapping)
        or successor_logical.get("digest") != source_logical.get("digest")
    ):
        raise KeypointPositionSourceError(
            "The keypoint coordinate successor changed the logical observation payload."
        )

    try:
        resolved = resolve_active_keypoint_bundle_from_root(root)
    except Exception as exc:
        raise KeypointPositionSourceError(
            f"Current source keypoint bundle authority is invalid: {exc}."
        ) from exc
    current_source_authority = (
        resolved.get("authority") if isinstance(resolved, Mapping) else None
    )
    sealed_source_authority = authority_payload["source_authority"]
    if (
        not isinstance(current_source_authority, Mapping)
        or current_source_authority != sealed_source_authority.get("record")
        or canonical_json_sha256(current_source_authority)
        != sealed_source_authority.get("record_sha256")
    ):
        raise KeypointPositionSourceError(
            "The source keypoint bundle authority changed after successor publication."
        )
    persisted = _validate_bundle_authority_direct_consolidated(
        analysis_zarr,
        authority=current_source_authority,
    )
    if persisted != dict(current_source_authority):
        raise KeypointPositionSourceError(
            "The published source keypoint authority differs between metadata views."
        )

    attrs = getattr(run, "attrs", {})
    owner = attrs.get(ATOMIC_PUBLICATION_OWNER_ATTR)
    if (
        attrs.get(RUN_COMPLETION_CONTRACT_ATTR) != RUN_COMPLETION_CONTRACT
        or attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE
        or attrs.get("status") != RUN_STATUS_COMPLETE
        or attrs.get("stage_selector_eligible") is not False
        or attrs.get("production_candidate") is not True
        or attrs.get("production_selector_activation")
        != SEALED_BUNDLE_PRODUCTION_SELECTOR_ACTIVATION
        or type(owner) is not str
        or _PUBLICATION_OWNER_UUID.fullmatch(owner) is None
    ):
        raise KeypointPositionSourceError(
            "The keypoint coordinate successor is not one sealed ineligible candidate."
        )
    try:
        complete = is_run_complete_in_parent(parent, run)
    except Exception as exc:
        raise KeypointPositionSourceError(
            f"Unable to validate keypoint successor completion: {exc}."
        ) from exc
    if complete is not True:
        raise KeypointPositionSourceError(
            "The keypoint coordinate successor is incomplete under its parent policy."
        )
    selector_refs = tuple(parent.attrs.get(name) for name in _SELECTOR_ALIAS_NAMES)
    if run_id in selector_refs or root.attrs.get("current_keypoint_group_path") == run_path:
        raise KeypointPositionSourceError(
            "The coordinate successor canary is unexpectedly selected."
        )
    return _readonly_mapping(authority), canonical_json_sha256(authority)


def load_keypoint_coordinate_successor_admission(
    analysis_zarr: str | Path,
    *,
    run_path: str,
) -> BoundKeypointCoordinateSuccessorAdmission:
    """Admit the maintained coordinate-successor profile without payload reads.

    This is the shared evidence branch for read-only planning and dependency
    admission.  It performs full-strength manifest, authority, lifecycle,
    selector, and metadata validation.  Coordinate payload validation remains
    owned by the full source loader, which reuses this exact admission.
    """

    archive = Path(analysis_zarr).expanduser().resolve()
    normalized_path, run_id = _canonical_run_path(run_path)
    root = _open_root(archive)
    parent = _require_group(root, "keypoints_runs", label="keypoint parent")
    run = _require_group(root, normalized_path, label="keypoint run")
    manifest, payload = _manifest_and_payload(run, run_id=run_id)
    authority, authority_digest = _require_coordinate_successor_authority(
        root,
        archive,
        run=run,
        parent=parent,
        run_path=normalized_path,
        run_id=run_id,
        manifest=manifest,
        payload=payload,
    )
    metadata_declarations_digest = _validate_published_metadata(
        archive,
        run_path=normalized_path,
        run_id=run_id,
        payload=payload,
    )
    active = authority["payload"]["source_authority"]["record"]
    active_digest = authority["payload"]["source_authority"]["record_sha256"]
    return BoundKeypointCoordinateSuccessorAdmission(
        analysis_zarr=archive,
        run_path=normalized_path,
        run_id=run_id,
        run_group=run,
        manifest=_readonly_mapping(manifest),
        manifest_digest=canonical_json_sha256(manifest),
        metadata_declarations_digest=metadata_declarations_digest,
        successor_authority=authority,
        successor_authority_digest=authority_digest,
        active_keypoint_bundle_authority=_readonly_mapping(active),
        active_keypoint_bundle_authority_digest=active_digest,
        _root=root,
        _verification_seal=_BOUND_COORDINATE_SUCCESSOR_ADMISSION_SEAL,
    )


def load_keypoint_coordinate_successor_source(
    analysis_zarr: str | Path,
    *,
    run_path: str,
) -> BoundKeypointCoordinateSuccessorSource:
    """Resolve and fully load the maintained coordinate-successor profile.

    Scientific consumers use this interface.  It first invokes the same
    metadata-only admission used by planners, then validates and materializes
    the complete coordinate surfaces.
    """

    admission = load_keypoint_coordinate_successor_admission(
        analysis_zarr,
        run_path=run_path,
    )
    surfaces = require_bound_ineligible_keypoint_coordinate_surfaces(
        load_persisted_ineligible_keypoint_coordinate_surfaces(
            admission._root,
            admission.run_path,
        )
    )
    return BoundKeypointCoordinateSuccessorSource(
        analysis_zarr=admission.analysis_zarr,
        run_path=admission.run_path,
        run_id=admission.run_id,
        run_group=admission.run_group,
        manifest=admission.manifest,
        manifest_digest=admission.manifest_digest,
        surfaces=surfaces,
        successor_authority=admission.successor_authority,
        successor_authority_digest=admission.successor_authority_digest,
        active_keypoint_bundle_authority=admission.active_keypoint_bundle_authority,
        active_keypoint_bundle_authority_digest=(
            admission.active_keypoint_bundle_authority_digest
        ),
        _verification_seal=_BOUND_COORDINATE_SUCCESSOR_SEAL,
    )


def _bind_source_schema(
    binding: Mapping[str, Any],
    pose_binding: Mapping[str, Any],
) -> tuple[str, str, tuple[str, ...], dict[str, int]]:
    source_schema = binding.get("source_schema")
    pose_schema = pose_binding.get("pose_schema")
    if not isinstance(source_schema, Mapping):
        raise KeypointPositionSourceError(
            "Keypoint AnatomyProfile binding lacks exact skeleton authority."
        )
    if not isinstance(pose_schema, Mapping):
        raise KeypointPositionSourceError("Keypoint run lacks its bound pose schema.")
    authority = source_schema.get("authority")
    if authority == "pose_schema_package":
        expected_schema = source_schema.get("package_payload")
        if not isinstance(expected_schema, Mapping):
            raise KeypointPositionSourceError(
                "Keypoint AnatomyProfile binding lacks an exact pose-schema package."
            )
        for field_name in (
            "skeleton_id",
            "kpt_shape",
            "keypoint_labels",
            "nodes",
            "edges",
        ):
            if pose_schema.get(field_name) != expected_schema.get(field_name):
                raise KeypointPositionSourceError(
                    "Anatomy source binding disagrees with pose schema field "
                    f"{field_name!r}."
                )
    elif authority == "keypoint_skeleton_semantics":
        expected_document = source_schema.get("skeleton_document")
        actual_document = keypoint_skeleton_document(pose_binding)
        if actual_document != expected_document:
            raise KeypointPositionSourceError(
                "Anatomy source binding disagrees with the exact keypoint skeleton "
                "semantics document."
            )
        if keypoint_skeleton_digest(pose_binding) != source_schema.get(
            "skeleton_sha256"
        ):
            raise KeypointPositionSourceError(
                "Anatomy source binding disagrees with the exact keypoint skeleton "
                "semantics digest."
            )
    else:
        raise KeypointPositionSourceError(
            "Keypoint AnatomyProfile binding uses unsupported skeleton authority."
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
    authority_mode = _require_authority_mode(policy.authority_mode)
    profile, binding = _resolve_anatomy_binding(policy)
    root = _open_root(analysis_zarr)
    parent = _require_group(root, "keypoints_runs", label="keypoint parent")
    run = _require_group(root, normalized_path, label="keypoint run")
    authority_record: Mapping[str, Any] | None = None
    authority_digest: str | None = None
    successor_authority: Mapping[str, Any] | None = None
    successor_authority_digest: str | None = None
    if authority_mode == KEYPOINT_AUTHORITY_MODE_CANONICAL_SELECTOR:
        _require_current_selector(
            root,
            parent,
            run,
            run_path=normalized_path,
            run_id=run_id,
        )
    manifest, payload = _manifest_and_payload(run, run_id=run_id)
    if authority_mode == KEYPOINT_AUTHORITY_MODE_SEALED_BUNDLE_CANARY:
        authority_record, authority_digest = _require_sealed_bundle_authority(
            root,
            analysis_zarr,
            run=run,
            parent=parent,
            run_path=normalized_path,
            run_id=run_id,
            manifest=manifest,
            payload=payload,
        )
    elif authority_mode == KEYPOINT_AUTHORITY_MODE_COORDINATE_SUCCESSOR_CANARY:
        successor_authority, successor_authority_digest = (
            _require_coordinate_successor_authority(
                root,
                analysis_zarr,
                run=run,
                parent=parent,
                run_path=normalized_path,
                run_id=run_id,
                manifest=manifest,
                payload=payload,
            )
        )
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
    if authority_mode == KEYPOINT_AUTHORITY_MODE_CANONICAL_SELECTOR:
        surfaces = require_bound_keypoint_coordinate_surfaces(
            load_persisted_keypoint_coordinate_surfaces(root, normalized_path)
        )
    else:
        surfaces = require_bound_ineligible_keypoint_coordinate_surfaces(
            load_persisted_ineligible_keypoint_coordinate_surfaces(
                root, normalized_path
            )
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
        source_kind=(
            SOURCE_KIND
            if authority_mode == KEYPOINT_AUTHORITY_MODE_CANONICAL_SELECTOR
            else (
                COORDINATE_SUCCESSOR_SOURCE_KIND
                if authority_mode
                == KEYPOINT_AUTHORITY_MODE_COORDINATE_SUCCESSOR_CANARY
                else SEALED_BUNDLE_SOURCE_KIND
            )
        ),
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
        authority_mode=authority_mode,
        keypoint_bundle_authority=authority_record,
        keypoint_bundle_authority_digest=authority_digest,
        coordinate_successor_authority=successor_authority,
        coordinate_successor_authority_digest=successor_authority_digest,
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
            authority_mode=source.authority_mode,
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
        "authority_mode",
        "keypoint_bundle_authority",
        "keypoint_bundle_authority_digest",
        "coordinate_successor_authority",
        "coordinate_successor_authority_digest",
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
    "BoundKeypointCoordinateSuccessorAdmission",
    "BoundKeypointCoordinateSuccessorSource",
    "BoundKeypointPositionSource",
    "KeypointPositionSourceError",
    "KeypointPositionSourcePolicy",
    "KEYPOINT_AUTHORITY_MODE_CANONICAL_SELECTOR",
    "KEYPOINT_AUTHORITY_MODE_COORDINATE_SUCCESSOR_CANARY",
    "KEYPOINT_AUTHORITY_MODE_SEALED_BUNDLE_CANARY",
    "COORDINATE_SUCCESSOR_SOURCE_KIND",
    "SEALED_BUNDLE_PRODUCTION_SELECTOR_ACTIVATION",
    "SEALED_BUNDLE_SOURCE_KIND",
    "SOURCE_KIND",
    "SOURCE_MODALITY",
    "load_bound_keypoint_position_source",
    "load_keypoint_coordinate_successor_admission",
    "load_keypoint_coordinate_successor_source",
    "revalidate_bound_keypoint_position_source",
    "require_bound_keypoint_position_source",
]
