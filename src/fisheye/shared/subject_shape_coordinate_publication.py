"""Strict source-camera publication for future subject-shape runs.

Subject-shape algorithms operate on ROI-local refined-mask rasters.  Their
published point geometry, however, is a camera-overlay surface.  This module
owns that boundary: it verifies the exact refined-mask coordinate authority,
applies its row-specific directed ROI-to-camera placement, seals canonical
array descriptors and the anatomical body-frame record, and validates the
complete payload before a run can become selector eligible.

The first publication version deliberately accepts translation-only crop
placement.  That is the normal crop contract and preserves vectors, angles,
and distances.  Scale, padding, affine, and projective placement fail closed;
supporting them requires explicit vector/Jacobian and scalar re-derivations,
not relabelling or a resolution ratio.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from fisheye.shared.archive_identity import archive_identity
from fisheye.shared.canonical_coordinate_publication import (
    BoundCanonicalCoordinateDescriptor,
    build_bound_canonical_coordinate_descriptor,
    load_bound_canonical_coordinate_descriptor,
    stamp_bound_canonical_coordinate_descriptors,
)
from fisheye.shared.coordinate_descriptor import (
    CANONICAL_OVERLAY_DIRECT,
    CANONICAL_OVERLAY_NOT_SUITABLE,
    COORDINATE_DESCRIPTOR_ATTR,
    CanonicalCollectionAxis,
    DigestBoundCoordinateRecordRef,
)
from fisheye.shared.coordinate_frame_record import (
    BODY_ESTIMATOR_SOURCE_MANIFEST_ATTR,
    BODY_FRAME_CONTRACT_ATTR,
    BODY_FRAME_ESTIMATOR_ATTR,
    FISH_ANATOMICAL_BODY_FRAME_ATTR,
    BoundFishAnatomicalBodyFrame,
    array_payload_sha256,
    array_values_sha256,
    bind_body_frame_geometry,
    bind_body_source_coordinate_descriptor,
    bind_mask_component_axis_source,
    build_body_estimator_source_manifest_record,
    build_body_frame_contract_record,
    build_body_frame_estimator_record,
    build_fish_anatomical_body_frame_record,
    load_bound_body_frame_contract,
    load_bound_body_frame_estimator,
    load_bound_fish_anatomical_body_frame,
    stamp_body_frame_contract,
    stamp_body_frame_estimator,
    stamp_fish_anatomical_body_frame_record,
)
from fisheye.shared.coordinate_identity import (
    OBSERVATION_INSTANCE_DOMAIN,
    ROW_IDENTITY_CONTRACT_ATTR,
    ROW_IDENTITY_CONTRACT_DIGEST_ATTR,
    ROW_IDENTITY_CONTRACT_REF_ATTR,
    ROW_IDENTITY_KEY_ATTR,
    ROW_IDENTITY_KEY_DIGEST_ATTR,
    SOURCE_ROW_TEMPORAL_AUTHORITY_ATTR,
    SOURCE_ROW_TEMPORAL_AUTHORITY_DIGEST_ATTR,
    BoundRowIdentityContract,
    BoundSourceRowTemporalAuthority,
    build_row_identity_contract,
    load_bound_row_identity_contract,
    load_bound_source_row_temporal_authority,
    stamp_and_bind_row_identity_contract,
    stamp_source_row_temporal_authority,
)
from fisheye.shared.coordinate_record import (
    BoundCoordinateRecord,
    bind_persisted_coordinate_record,
    stamp_and_bind_persisted_coordinate_record,
)
from fisheye.shared.directed_transform_chain import (
    apply_bound_directed_transform_chain,
)
from fisheye.shared.refined_subject_mask_coordinate_publication import (
    BoundRefinedSubjectMaskCoordinateSurfaces,
    load_persisted_refined_subject_mask_coordinate_surfaces,
)
from fisheye.shared.pixel_frame_authority import (
    BoundAcquisitionCameraFrame,
    BoundPixelFrameAuthority,
)
from fisheye.shared.proof_verification import (
    finish_proof_verification,
    proof_verification_operation,
    restart_proof_verification,
)
from fisheye.shared.coordinate_reference import canonical_node_path
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_CONTRACT,
    RUN_COMPLETION_CONTRACT_ATTR,
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
)
from fisheye.shared.zarr_payload_receipt import (
    build_payload_integrity_receipt,
    build_payload_validation_receipt,
    canonical_payload_integrity_receipt,
    verify_payload_integrity_receipt,
    verify_payload_validation_receipt,
)
from fisheye.shared.zarr.subject_shape_bundle_source import (
    SUBJECT_SHAPE_BUNDLE_SOURCE_KIND,
    BoundSubjectShapeBundleSource,
    assignment_rebinding_run_id_from_source_record,
    load_subject_shape_bundle_source,
    require_bound_subject_shape_bundle_source,
)


SUBJECT_SHAPE_COORDINATE_CONTRACT = "canonical_v2"
SUBJECT_SHAPE_COORDINATE_BINDING_STATUS_ATTR = "coordinate_binding_status"
SUBJECT_SHAPE_COMPUTING_UNBOUND_STATUS = "computing_unbound_numeric_stage_v1"
SUBJECT_SHAPE_UNBOUND_STAGE_STATUS = "unbound_numeric_stage_complete_v1"
SUBJECT_SHAPE_PUBLISHING_BINDING_STATUS = "publishing_canonical_binding_v1"
SUBJECT_SHAPE_BOUND_CANONICAL_STATUS = "bound_canonical_v2"
SUBJECT_SHAPE_PUBLICATION_OWNER_ATTR = "subject_shape_publication_owner_uuid"
SUBJECT_SHAPE_PUBLICATION_GENERATION_ATTR = "publication_generation"
SUBJECT_SHAPE_PUBLICATION_POLICY_ATTR = "publication_policy"
SUBJECT_SHAPE_PARENT_PUBLICATION_LEASE_ATTR = "subject_shape_publication_lease"
SUBJECT_SHAPE_DERIVATION_ATTR = "subject_shape_coordinate_derivation"
SUBJECT_SHAPE_COMPONENT_SCHEMA_ATTR = "subject_shape_component_schema"
SUBJECT_SHAPE_MANIFEST_ATTR = "subject_shape_publication_manifest"
SUBJECT_SHAPE_SOURCE_BINDING_ATTR = "subject_shape_source_binding"
SUBJECT_SHAPE_SOURCE_BINDING_DIGEST_ATTR = "subject_shape_source_binding_sha256"
SUBJECT_SHAPE_SOURCE_KIND_ATTR = "subject_shape_source_kind"
SUBJECT_SHAPE_BUNDLE_ID_ATTR = "source_subject_mask_bundle_id"
SUBJECT_SHAPE_BUNDLE_ACTIVE_AT_DERIVATION_ATTR = (
    "source_subject_mask_bundle_active_at_derivation"
)
SUBJECT_SHAPE_HISTORICAL_SOURCE_KIND = "historical_refined_coordinate_surfaces_v4"
SUBJECT_SHAPE_SCIENTIFIC_CONFIGURATION_ATTR = (
    "subject_shape_scientific_configuration"
)
SUBJECT_SHAPE_TAIL_SAMPLE_AXIS_ATTR = "subject_shape_tail_sample_axis"
SUBJECT_SHAPE_HEADING_SEMANTICS_ATTR = "subject_shape_heading_semantics"
SUBJECT_SHAPE_SCALAR_SURFACE_ATTR = "subject_shape_scalar_surface"
SUBJECT_SHAPE_SCALAR_SURFACE_INVENTORY_ATTR = (
    "subject_shape_scalar_surface_inventory"
)
SUBJECT_SHAPE_UNBOUND_MANIFEST_ATTR = "subject_shape_unbound_numeric_manifest"
SUBJECT_SHAPE_UNBOUND_MANIFEST_SCHEMA_ID = (
    "palette.subject_shape_unbound_numeric_manifest"
)
SUBJECT_SHAPE_PROJECTED_UNBOUND_MANIFEST_SCHEMA_VERSION = 2
SUBJECT_SHAPE_PROJECTED_UNBOUND_BINDING_STATUS = (
    "unbound_source_camera_projected_numeric_payload"
)
SUBJECT_SHAPE_CONSUMED_UNBOUND_STAGE_ATTR = (
    "subject_shape_consumed_unbound_stage"
)
SUBJECT_SHAPE_SCHEMA_INVENTORY_SCHEMA_ID = (
    "palette.subject_shape_closed_schema_inventory"
)
SUBJECT_SHAPE_STORAGE_PLAN_ATTR = "subject_shape_storage_plan"
SUBJECT_SHAPE_STORAGE_PLAN_DIGEST_ATTR = "subject_shape_storage_plan_payload_sha256"
SUBJECT_SHAPE_STORAGE_PROFILE_ID_ATTR = "subject_shape_storage_profile_id"
SUBJECT_SHAPE_STORAGE_PROFILE_ROLE_ATTR = "subject_shape_storage_profile_role"
SUBJECT_SHAPE_STORAGE_PROFILE_ROLE = "explicit_unpromoted_candidate"
SUBJECT_SHAPE_SUPPORTED_STORAGE_PROFILE_ROLE = "supported_selector_publication"
SUBJECT_SHAPE_ACCESS_AWARE_SUPPORTED_PROFILE_ID = "subject_shape_access_aware_v1"
SUBJECT_SHAPE_STORAGE_CANDIDATE_ATTR = "subject_shape_storage_candidate"
SUBJECT_SHAPE_STORAGE_SOURCE_MANIFEST_ATTR = (
    "subject_shape_storage_source_unbound_manifest"
)
SUBJECT_SHAPE_DEFERRED_STORAGE_RECEIPT_ATTR = (
    "subject_shape_deferred_storage_transform_receipt"
)
SUBJECT_SHAPE_STORAGE_SOURCE_MANIFEST_SCHEMA_ID = (
    "palette.subject_shape_storage_source_unbound_manifest"
)
SUBJECT_SHAPE_STORAGE_METADATA_POLICY_ATTR = (
    "subject_shape_storage_metadata_visibility_policy"
)
SUBJECT_SHAPE_PAYLOAD_INTEGRITY_RECEIPT_ATTR = (
    "subject_shape_payload_integrity_receipt"
)
SUBJECT_SHAPE_PAYLOAD_VALIDATION_RECEIPT_ATTR = (
    "subject_shape_payload_validation_receipt"
)
SUBJECT_SHAPE_PAYLOAD_RECEIPT_PROFILE_ATTR = (
    "subject_shape_payload_receipt_profile"
)
SUBJECT_SHAPE_NUMERIC_PROJECTION_ATTR = (
    "subject_shape_source_camera_numeric_projection"
)
SUBJECT_SHAPE_NUMERIC_PROJECTION_DIGEST_ATTR = (
    f"{SUBJECT_SHAPE_NUMERIC_PROJECTION_ATTR}_sha256"
)
SUBJECT_SHAPE_NUMERIC_PROJECTION_SCHEMA_ID = (
    "palette.subject_shape_source_camera_numeric_projection"
)
SUBJECT_SHAPE_PAYLOAD_RECEIPT_PROFILE = (
    "single_locked_bound_payload_receipt_v1"
)
SUBJECT_SHAPE_PAYLOAD_RECEIPT_PROFILE_V1 = SUBJECT_SHAPE_PAYLOAD_RECEIPT_PROFILE
SUBJECT_SHAPE_PAYLOAD_RECEIPT_PROFILE_V2 = (
    "verified_staged_transfer_plus_binding_append_receipt_v2"
)
SUBJECT_SHAPE_PAYLOAD_RECEIPT_PROFILES = frozenset(
    {
        SUBJECT_SHAPE_PAYLOAD_RECEIPT_PROFILE_V1,
        SUBJECT_SHAPE_PAYLOAD_RECEIPT_PROFILE_V2,
    }
)
SUBJECT_SHAPE_BINDING_APPENDED_ARRAY_PATHS = (
    "instance_key",
    "source_crop_row_ids",
    "source_acquisition_frame_index",
    "component_centroid_xy",
    "component_centroid_valid",
    "body_frame/axis_valid",
)
SUBJECT_SHAPE_PAYLOAD_VALIDATOR_SCHEMA_ID = (
    "palette.subject_shape_bound_payload_validator"
)
SUBJECT_SHAPE_PAYLOAD_VALIDATOR_SCHEMA_VERSION = 1
SUBJECT_SHAPE_STORAGE_ARRAY_ATTRS = frozenset(
    {
        "logical_schema_id",
        "logical_schema_version",
        "storage_policy_version",
        "storage_profile_id",
        "codec_profile_id",
        "access_pattern",
        "write_mode",
    }
)


_ACTIVE_SUBJECT_SHAPE_ARRAY_DIGESTS: ContextVar[Mapping[str, str] | None] = (
    ContextVar("palette_subject_shape_array_digests", default=None)
)


def is_supported_subject_shape_payload_receipt_profile(value: Any) -> bool:
    """Return whether ``value`` names one maintained receipt grammar."""

    return type(value) is str and value in SUBJECT_SHAPE_PAYLOAD_RECEIPT_PROFILES


@contextmanager
def _subject_shape_array_digest_scope(
    values: Mapping[str, str],
) -> Iterator[None]:
    token = _ACTIVE_SUBJECT_SHAPE_ARRAY_DIGESTS.set(dict(values))
    try:
        yield
    finally:
        _ACTIVE_SUBJECT_SHAPE_ARRAY_DIGESTS.reset(token)


def _uses_access_aware_subject_shape_storage(run: Any) -> bool:
    return run.attrs.get(SUBJECT_SHAPE_STORAGE_PROFILE_ROLE_ATTR) in {
        SUBJECT_SHAPE_STORAGE_PROFILE_ROLE,
        SUBJECT_SHAPE_SUPPORTED_STORAGE_PROFILE_ROLE,
    }


SUBJECT_SHAPE_SCHEMA_VERSION = 1
SUBJECT_SHAPE_DERIVATION_SCHEMA_ID = "palette.subject_shape_coordinate_derivation"
SUBJECT_SHAPE_COMPONENT_SCHEMA_ID = "palette.mask_component_geometry_schema"
SUBJECT_SHAPE_MANIFEST_SCHEMA_ID = "palette.subject_shape_publication_manifest"
SUBJECT_SHAPE_SCIENTIFIC_CONFIGURATION_SCHEMA_ID = (
    "palette.subject_shape_scientific_configuration"
)
SUBJECT_SHAPE_TAIL_SAMPLE_AXIS_SCHEMA_ID = "palette.subject_shape_tail_sample_axis"
SUBJECT_SHAPE_HEADING_SEMANTICS_SCHEMA_ID = (
    "palette.subject_shape_row_bound_heading_semantics"
)
SUBJECT_SHAPE_SCALAR_SURFACE_SCHEMA_ID = (
    "palette.subject_shape_scalar_surface"
)
SUBJECT_SHAPE_SCALAR_SURFACE_INVENTORY_SCHEMA_ID = (
    "palette.subject_shape_scalar_surface_inventory"
)
SUBJECT_SHAPE_PUBLICATION_POLICY = (
    "immutable_complete_ineligible_validate_selectors_then_eligibility_v1"
)
CANONICAL_SUBJECT_SHAPE_PROFILE_ID = "analysis.subject_shape.full_anatomy_v4"
CANONICAL_SUBJECT_SHAPE_RUN_SCHEMA_ID = "analysis.subject_shape_runs"
CANONICAL_SUBJECT_SHAPE_RUN_SCHEMA_VERSION = 4
CANONICAL_SUBJECT_SHAPE_METHOD = "subject_shape_from_refined_masks_v11"
CANONICAL_SUBJECT_SHAPE_METHOD_VERSION = 11
CANONICAL_SUBJECT_SHAPE_BUNDLE_PROFILE_ID = "analysis.subject_shape.full_anatomy_v5"
CANONICAL_SUBJECT_SHAPE_BUNDLE_RUN_SCHEMA_VERSION = 5
CANONICAL_SUBJECT_SHAPE_BUNDLE_METHOD = "subject_shape_from_recording_mask_bundle_v12"
CANONICAL_SUBJECT_SHAPE_BUNDLE_METHOD_VERSION = 12
SUBJECT_SHAPE_BUNDLE_DERIVATION_SCHEMA_VERSION = 2
SUBJECT_SHAPE_BUNDLE_MANIFEST_SCHEMA_VERSION = 2
CANONICAL_SUBJECT_SHAPE_COMPONENT_ORDER = (
    "subject_body",
    "swim_bladder",
    "eye_left",
    "eye_right",
)
CANONICAL_SUBJECT_SHAPE_RELATION_ORDER = (
    "eye_pair",
    "swim_bladder_to_body",
    "eyes_to_body",
)
CANONICAL_SUBJECT_SHAPE_ROW_INDEX_ARRAYS = (
    "source_crop_row_ids",
    "instance_key",
)
CANONICAL_SUBJECT_SHAPE_ROW_LINEAGE_MISSING = (
    "frame_indices",
    "detection_indices",
    "source_refined_row_ids",
    "source_detect_row_index",
)
BODY_FRAME_IMPLEMENTATION_VERSION = "subject_shape_from_refined_masks_v11"

_OWNER_RE = re.compile(r"^[0-9a-f]{32}$")
_BOUND_PUBLICATION_SEAL = object()
_METADATA_PUBLICATION_SEAL = object()
_DEFERRED_ACTIVATION_SEAL = object()
_GUARDED_SELECTORS = (
    "latest",
    "latest_complete",
    "authoritative_run",
    "authoritative_run_provenance",
)
_ACTIVATION_SNAPSHOT_ATTRS = (
    *_GUARDED_SELECTORS,
    SUBJECT_SHAPE_PUBLICATION_GENERATION_ATTR,
    SUBJECT_SHAPE_PUBLICATION_POLICY_ATTR,
    SUBJECT_SHAPE_PARENT_PUBLICATION_LEASE_ATTR,
)


class SubjectShapeCoordinatePublicationError(ValueError):
    """Raised when a future subject-shape coordinate contract is incomplete."""


def _fail(message: str) -> None:
    raise SubjectShapeCoordinatePublicationError(message)


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        _fail(f"Subject-shape metadata is not canonical JSON: {exc}.")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _record_pointer(value: BoundCoordinateRecord) -> dict[str, str]:
    return {
        "record_ref": value.record_ref,
        "record_sha256": value.record_sha256,
    }


def _temporal_pointer(value: BoundSourceRowTemporalAuthority) -> dict[str, str]:
    return {
        "record_ref": value.record_ref,
        "record_sha256": value.record_sha256,
    }


def _canonical_run_path(value: str) -> str:
    if not isinstance(value, str):
        _fail("Subject-shape run path must be text.")
    path = value.strip().strip("/")
    if (
        path != value
        or not path.startswith("analysis/subject_shape_runs/")
        or len(path.split("/")) != 3
        or any(part in {"", ".", ".."} for part in path.split("/"))
    ):
        _fail(f"Subject-shape path {value!r} is not canonical.")
    return path


def _node(root: Any, path: str, *, label: str) -> Any:
    try:
        result = root[path]
    except Exception as exc:
        _fail(f"Persisted {label} is unavailable at {path!r}: {exc}.")
    if canonical_node_path(result) != path:
        _fail(f"Persisted {label} resolved outside its canonical path.")
    return result


def _owner(run: Any, *, expected: str | None = None) -> str:
    value = run.attrs.get(SUBJECT_SHAPE_PUBLICATION_OWNER_ATTR)
    if not isinstance(value, str) or _OWNER_RE.fullmatch(value) is None:
        _fail("Subject-shape run lacks a canonical publication owner token.")
    if expected is not None and value != expected:
        _fail("Subject-shape publication owner changed.")
    return value


def _require_state(
    run: Any,
    *,
    complete: bool,
    eligible: bool,
    expected_owner: str | None = None,
) -> str:
    owner = _owner(run, expected=expected_owner)
    expected_status = RUN_STATUS_COMPLETE if complete else "running"
    if (
        run.attrs.get(RUN_COMPLETION_CONTRACT_ATTR) != RUN_COMPLETION_CONTRACT
        or run.attrs.get(RUN_COMPLETION_STATUS_ATTR) != expected_status
        or run.attrs.get("stage_selector_eligible") is not eligible
    ):
        _fail(
            "Subject-shape publication state must be exact "
            f"status={expected_status!r}, eligible={eligible!r}."
        )
    return owner


def _refined_source_path(run: Any) -> str:
    name = run.attrs.get("source_refined_subject_masks_run")
    if not isinstance(name, str) or not name or "/" in name:
        _fail("Subject-shape source_refined_subject_masks_run is invalid.")
    return f"refined_subject_masks_runs/{name}"


SubjectShapeCoordinateSource = (
    BoundRefinedSubjectMaskCoordinateSurfaces | BoundSubjectShapeBundleSource
)


def _is_bundle_source(source: SubjectShapeCoordinateSource) -> bool:
    return type(source) is BoundSubjectShapeBundleSource


def _local_archive_path(root: Any) -> str:
    identity = archive_identity(root)
    if identity.kind != "local_store_root" or not identity.key:
        _fail(
            "Recording-bundle subject-shape publication currently requires one "
            "local Zarr archive so its sealed bundle validator can reopen it."
        )
    return str(identity.key[0])


def _source_run_group(source: SubjectShapeCoordinateSource) -> Any:
    if _is_bundle_source(source):
        return require_bound_subject_shape_bundle_source(source).authority.refined_run
    return source.context._run_group


def _source_row_node(source: SubjectShapeCoordinateSource, name: str) -> Any:
    if _is_bundle_source(source):
        bundle = require_bound_subject_shape_bundle_source(source)
        nodes = {
            "instance_key": bundle.instance_key_node,
            "source_crop_row_ids": bundle.source_crop_row_ids_node,
            "source_acquisition_frame_index": bundle.source_acquisition_frame_index_node,
        }
        node = nodes.get(name)
    else:
        node = source.context._run_group.get(name)
    if node is None:
        _fail(f"Subject-shape source lacks required row array {name!r}.")
    return node


def _source_acquisition_frame(
    source: SubjectShapeCoordinateSource,
) -> BoundAcquisitionCameraFrame:
    if _is_bundle_source(source):
        return require_bound_subject_shape_bundle_source(source).acquisition_frame
    return source.context.temporal_authority.acquisition_frame


def _source_camera_frames(
    source: SubjectShapeCoordinateSource,
) -> tuple[BoundPixelFrameAuthority, BoundPixelFrameAuthority]:
    if _is_bundle_source(source):
        bundle = require_bound_subject_shape_bundle_source(source)
        return (
            bundle.continuous_source_camera_frame,
            bundle.edge_source_camera_frame,
        )
    return (
        source.context.continuous_chain.source_camera_frame_authority,
        source.context.pixel_edge_chain.source_camera_frame_authority,
    )


def _source_binding_record(
    source: SubjectShapeCoordinateSource,
) -> Mapping[str, Any] | None:
    if not _is_bundle_source(source):
        return None
    return require_bound_subject_shape_bundle_source(source).source_record


def _stamp_source_binding(
    run: Any,
    source: SubjectShapeCoordinateSource,
) -> BoundCoordinateRecord | None:
    record = _source_binding_record(source)
    if record is None:
        return None
    expected_digest = _canonical_sha256(record)
    if (
        run.attrs.get(SUBJECT_SHAPE_SOURCE_BINDING_ATTR) != record
        or run.attrs.get(SUBJECT_SHAPE_SOURCE_BINDING_DIGEST_ATTR)
        != expected_digest
    ):
        _fail("Subject-shape unbound source-binding receipt is absent or stale.")
    node = run.require_group("coordinate_records").require_group("source_binding")
    return stamp_and_bind_persisted_coordinate_record(
        node,
        record,
        attr_name=SUBJECT_SHAPE_SOURCE_BINDING_ATTR,
    )


def _load_source_binding(
    run: Any,
    source: SubjectShapeCoordinateSource,
) -> BoundCoordinateRecord | None:
    record = _source_binding_record(source)
    if record is None:
        return None
    node = run["coordinate_records/source_binding"]
    binding = bind_persisted_coordinate_record(
        node,
        attr_name=SUBJECT_SHAPE_SOURCE_BINDING_ATTR,
    )
    if binding.record != record:
        _fail("Subject-shape bundle-source binding differs from live exact authority.")
    return binding


@proof_verification_operation
def load_exact_subject_shape_source(
    root: Any,
    run: Any,
) -> SubjectShapeCoordinateSource:
    kind = run.attrs.get(
        SUBJECT_SHAPE_SOURCE_KIND_ATTR,
        SUBJECT_SHAPE_HISTORICAL_SOURCE_KIND,
    )
    if kind == SUBJECT_SHAPE_HISTORICAL_SOURCE_KIND:
        return load_exact_subject_shape_refined_source(root, run)
    if kind != SUBJECT_SHAPE_BUNDLE_SOURCE_KIND:
        _fail(f"Unsupported subject-shape source kind {kind!r}.")
    bundle_id = run.attrs.get(SUBJECT_SHAPE_BUNDLE_ID_ATTR)
    if not isinstance(bundle_id, str) or not bundle_id or "/" in bundle_id:
        _fail("Bundle-bound subject-shape run lacks one exact source bundle ID.")
    source = load_subject_shape_bundle_source(
        Path(_local_archive_path(root)),
        bundle_id=bundle_id,
        allow_inactive=True,
        assignment_keypoint_rebinding_run_id=(
            assignment_rebinding_run_id_from_source_record(
                run.attrs.get(SUBJECT_SHAPE_SOURCE_BINDING_ATTR)
            )
        ),
    )
    if archive_identity(source.authority.refined_run) != archive_identity(run):
        _fail("Subject-shape and recording-bundle source span archives/stores.")
    if type(run.attrs.get(SUBJECT_SHAPE_BUNDLE_ACTIVE_AT_DERIVATION_ATTR)) is not bool:
        _fail("Subject-shape bundle derivation lacks an exact activation-state receipt.")
    return source


@proof_verification_operation
def load_exact_subject_shape_refined_source(
    root: Any,
    run: Any,
) -> BoundRefinedSubjectMaskCoordinateSurfaces:
    """Freshly prove the exact selected refined-mask source."""

    source = load_persisted_refined_subject_mask_coordinate_surfaces(
        root,
        _refined_source_path(run),
    )
    if archive_identity(source.context._run_group) != archive_identity(run):
        _fail("Subject-shape and refined-mask source span archives/stores.")
    return source


def _transform_rows(
    values: np.ndarray,
    *,
    source: BoundRefinedSubjectMaskCoordinateSurfaces,
    edge: bool,
) -> np.ndarray:
    chain = source.context.pixel_edge_chain if edge else source.context.continuous_chain
    return apply_bound_directed_transform_chain(
        values,
        chain,
        row_identity=source.context.row_identity,
    )


def require_translation_only_refined_placement(
    source: BoundRefinedSubjectMaskCoordinateSurfaces,
) -> tuple[np.ndarray, np.ndarray]:
    """Return continuous/edge row offsets after proving unit linear basis.

    This uses the sealed transform itself as the authority.  It therefore also
    catches a wrong direction, a same-sized but reordered placement rowset, or
    any scale/pad/projective path that a root-dimension ratio would conceal.
    """

    row_count = int(source.context.row_identity.leading_dimension)
    zero = np.zeros((row_count, 2), dtype=np.float64)
    basis_x = np.tile(np.asarray([[1.0, 0.0]], dtype=np.float64), (row_count, 1))
    basis_y = np.tile(np.asarray([[0.0, 1.0]], dtype=np.float64), (row_count, 1))
    offsets: list[np.ndarray] = []
    for edge in (False, True):
        origin = _transform_rows(zero, source=source, edge=edge)
        dx = _transform_rows(basis_x, source=source, edge=edge) - origin
        dy = _transform_rows(basis_y, source=source, edge=edge) - origin
        expected_x = np.tile(np.asarray([[1.0, 0.0]]), (row_count, 1))
        expected_y = np.tile(np.asarray([[0.0, 1.0]]), (row_count, 1))
        if (
            origin.shape != (row_count, 2)
            or not np.isfinite(origin).all()
            or not np.allclose(dx, expected_x, rtol=0.0, atol=1e-12)
            or not np.allclose(dy, expected_y, rtol=0.0, atol=1e-12)
        ):
            _fail(
                "Subject-shape v1 supports only exact translation ROI-to-camera "
                "placement; scale/pad/affine/projective placement needs explicit "
                "point, vector, angle, and distance re-derivation."
            )
        offsets.append(np.asarray(origin, dtype=np.float64))
    if not np.allclose(offsets[0], offsets[1], rtol=0.0, atol=1e-12):
        _fail("Continuous and half-open-edge crop placement translations disagree.")
    return offsets[0], offsets[1]


def require_translation_only_subject_shape_placement(
    source: SubjectShapeCoordinateSource,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact continuous/edge offsets for either maintained source kind."""

    if _is_bundle_source(source):
        offsets = require_bound_subject_shape_bundle_source(source).translation_offsets()
        return offsets, offsets.copy()
    return require_translation_only_refined_placement(source)


def build_subject_shape_source_camera_numeric_projection_record(
    source: SubjectShapeCoordinateSource,
) -> tuple[dict[str, Any], np.ndarray]:
    """Seal the exact translation used by the node-local numeric writer.

    This is projection evidence, not canonical coordinate authority.  The
    final-path binder freshly resolves the same source authority and requires
    this record byte-for-byte before it may skip the historical in-place
    translation pass.
    """

    continuous, edge = require_translation_only_subject_shape_placement(source)
    continuous = np.ascontiguousarray(continuous, dtype=np.float64)
    edge = np.ascontiguousarray(edge, dtype=np.float64)
    if continuous.shape != edge.shape or not np.array_equal(continuous, edge):
        _fail(
            "Source-camera numeric projection requires identical continuous "
            "and half-open-edge translation offsets."
        )
    if _is_bundle_source(source):
        bundle = require_bound_subject_shape_bundle_source(source)
        source_evidence: dict[str, Any] = {
            "source_kind": SUBJECT_SHAPE_BUNDLE_SOURCE_KIND,
            "source_binding_sha256": bundle.source_digest,
            "bundle_id": bundle.bundle_id,
            "bundle_manifest_payload_digest": bundle.authority.bundle_manifest[
                "payload_digest"
            ],
            "refined_manifest_payload_digest": bundle.authority.refined_manifest[
                "payload_digest"
            ],
        }
    else:
        source_evidence = {
            "source_kind": SUBJECT_SHAPE_HISTORICAL_SOURCE_KIND,
            "run_path": source.context.run_path,
            "context_sha256": source.context.context_record.record_sha256,
            "surface_inventory_sha256": source.inventory.record_sha256,
            "component_qc_inventory_sha256": (
                source.component_qc_inventory.record_sha256
            ),
        }
    record = {
        "schema_id": SUBJECT_SHAPE_NUMERIC_PROJECTION_SCHEMA_ID,
        "schema_version": 1,
        "projection_role": (
            "private_precanonical_numeric_evidence_not_coordinate_authority"
        ),
        "source_evidence": source_evidence,
        "row_count": int(continuous.shape[0]),
        "offset_dtype": np.dtype(continuous.dtype).str,
        "offset_shape": [int(value) for value in continuous.shape],
        "offset_content_sha256": array_values_sha256(continuous),
        "continuous_edge_offsets_equal": True,
        "transform_policy": "translation_only_roi_to_source_camera_v1",
        "application_policy": (
            "roi_science_then_source_camera_projection_at_chunk_write_v1"
        ),
        "point_policy": "add_row_xy_then_preserve_nan_and_mask_invalid",
        "bbox_policy": "add_row_xy_to_both_half_open_corners",
        "ellipse_policy": "add_row_xy_to_center_only",
        "vector_scalar_policy": "mask_invalid_vectors_and_do_not_translate_scalars",
    }
    return record, continuous


def stamp_subject_shape_source_camera_numeric_projection(
    run: Any,
    source: SubjectShapeCoordinateSource,
    *,
    component_names: Sequence[str],
) -> np.ndarray:
    if (
        SUBJECT_SHAPE_NUMERIC_PROJECTION_ATTR in run.attrs
        or SUBJECT_SHAPE_NUMERIC_PROJECTION_DIGEST_ATTR in run.attrs
    ):
        _fail("Subject-shape numeric projection evidence is already occupied.")
    record, offsets = build_subject_shape_source_camera_numeric_projection_record(
        source
    )
    digest = _canonical_sha256(record)
    run.attrs[SUBJECT_SHAPE_NUMERIC_PROJECTION_ATTR] = _json_copy(record)
    run.attrs[SUBJECT_SHAPE_NUMERIC_PROJECTION_DIGEST_ATTR] = digest
    components = run["components"]
    for name in component_names:
        group = components[str(name)]
        group.attrs["point_coordinate_space"] = (
            "source_camera_image_px_precanonical_numeric"
        )
        group.attrs["bbox_coordinate_space"] = (
            "source_camera_image_px_precanonical_numeric"
        )
    body = components.get("subject_body")
    if body is not None:
        body.attrs["principal_axis_semantics"] = (
            "unoriented_principal_axis_in_source_camera_xy_precanonical_numeric"
        )
    if (
        run.attrs.get(SUBJECT_SHAPE_NUMERIC_PROJECTION_ATTR) != record
        or run.attrs.get(SUBJECT_SHAPE_NUMERIC_PROJECTION_DIGEST_ATTR) != digest
    ):
        _fail("Subject-shape numeric projection evidence did not persist exactly.")
    return offsets


def require_subject_shape_source_camera_numeric_projection(
    run: Any,
    source: SubjectShapeCoordinateSource,
) -> Mapping[str, Any] | None:
    raw = run.attrs.get(SUBJECT_SHAPE_NUMERIC_PROJECTION_ATTR)
    raw_digest = run.attrs.get(SUBJECT_SHAPE_NUMERIC_PROJECTION_DIGEST_ATTR)
    if raw is None and raw_digest is None:
        return None
    if not isinstance(raw, Mapping) or not isinstance(raw_digest, str):
        _fail("Subject-shape numeric projection evidence is incomplete.")
    expected, _offsets = build_subject_shape_source_camera_numeric_projection_record(
        source
    )
    if (
        dict(raw) != expected
        or raw_digest != _canonical_sha256(expected)
        or int(expected["row_count"]) != int(_source_row_node(source, "instance_key").shape[0])
    ):
        _fail(
            "Subject-shape numeric projection differs from the freshly resolved "
            "source authority."
        )
    return raw


def _outer_row_block_size(node: Any, row_count: int) -> int:
    """Return one physical outer chunk/shard extent along the row axis."""

    try:
        metadata = node.metadata.to_dict()
        outer = metadata["chunk_grid"]["configuration"]["chunk_shape"]
        value = int(outer[0])
    except (AttributeError, KeyError, TypeError, ValueError, IndexError):
        chunks = getattr(node, "chunks", None)
        try:
            value = int(chunks[0]) if chunks else 0
        except (TypeError, ValueError, IndexError):
            value = 0
    if value <= 0:
        value = max(1, min(int(row_count), 65_536))
    return value


def _row_blocks(node: Any, row_count: int) -> Iterable[tuple[int, int]]:
    step = _outer_row_block_size(node, row_count)
    for start in range(0, int(row_count), step):
        yield start, min(int(row_count), start + step)


def _require_row_array_metadata(
    node: Any,
    *,
    row_count: int,
    trailing_shape: tuple[int, ...] | None = None,
) -> tuple[np.dtype[Any], tuple[int, ...]]:
    try:
        dtype = np.dtype(node.dtype)
        shape = tuple(int(value) for value in node.shape)
    except (AttributeError, TypeError, ValueError) as exc:
        _fail(f"/{canonical_node_path(node)} has invalid array metadata: {exc}")
    if (
        dtype.kind != "f"
        or not shape
        or shape[0] != int(row_count)
        or (trailing_shape is not None and shape[1:] != trailing_shape)
    ):
        _fail(f"/{canonical_node_path(node)} has an invalid row-aligned shape.")
    return dtype, shape


def project_subject_shape_points(
    values: Any,
    offsets_xy: Any,
    valid: Any,
) -> np.ndarray:
    """Apply the maintained point projection to one in-memory row block."""

    source = np.asarray(values)
    offsets = np.asarray(offsets_xy, dtype=np.float64)
    validity = np.asarray(valid, dtype=bool)
    if (
        source.dtype.kind != "f"
        or source.ndim < 2
        or source.shape[-1] != 2
        or offsets.shape != (source.shape[0], 2)
        or validity.shape != (source.shape[0],)
        or not np.isfinite(offsets).all()
    ):
        _fail("Subject-shape point projection inputs are not exact row-aligned XY.")
    output = source.astype(np.float64) + offsets.reshape(
        (source.shape[0],) + (1,) * (source.ndim - 2) + (2,)
    )
    output[np.isnan(source)] = np.nan
    output[~validity] = np.nan
    return output.astype(source.dtype)


def project_subject_shape_bboxes(
    values: Any,
    offsets_xy: Any,
    valid: Any,
) -> np.ndarray:
    source = np.asarray(values)
    offsets = np.asarray(offsets_xy, dtype=np.float64)
    validity = np.asarray(valid, dtype=bool)
    if (
        source.dtype.kind != "f"
        or source.shape != (offsets.shape[0], 4)
        or offsets.shape != (source.shape[0], 2)
        or validity.shape != (source.shape[0],)
        or not np.isfinite(offsets).all()
    ):
        _fail("Subject-shape bbox projection inputs are not exact row-aligned xyxy.")
    output = source.astype(np.float64)
    output[~validity] = np.nan
    return (
        output.reshape(source.shape[0], 2, 2) + offsets[:, None, :]
    ).reshape(source.shape).astype(source.dtype)


def project_subject_shape_ellipses(
    values: Any,
    offsets_xy: Any,
    valid: Any,
) -> np.ndarray:
    source = np.asarray(values)
    offsets = np.asarray(offsets_xy, dtype=np.float64)
    validity = np.asarray(valid, dtype=bool)
    if (
        source.dtype.kind != "f"
        or source.shape != (offsets.shape[0], 5)
        or offsets.shape != (source.shape[0], 2)
        or validity.shape != (source.shape[0],)
        or not np.isfinite(offsets).all()
    ):
        _fail("Subject-shape ellipse projection inputs are not exact row-aligned.")
    output = source.astype(np.float64)
    output[:, :2] += offsets
    output[~validity] = np.nan
    return output.astype(source.dtype)


def mask_invalid_subject_shape_vectors(values: Any, valid: Any) -> np.ndarray:
    source = np.asarray(values)
    validity = np.asarray(valid, dtype=bool)
    if (
        source.dtype.kind != "f"
        or not source.shape
        or validity.shape != (source.shape[0],)
    ):
        _fail("Subject-shape vector masking inputs are not row aligned.")
    output = source.copy()
    output[~validity] = np.nan
    return output


def _mask_invalid_and_translate_points_node(
    node: Any,
    offsets: np.ndarray,
    valid_node: Any,
) -> None:
    """Mask and translate one positional surface in one whole-shard pass."""

    row_count = int(offsets.shape[0])
    dtype, shape = _require_row_array_metadata(node, row_count=row_count)
    if len(shape) < 2 or shape[-1] != 2:
        _fail(f"/{canonical_node_path(node)} is not a row-aligned XY surface.")
    if tuple(int(value) for value in valid_node.shape) != (row_count,):
        _fail(f"/{canonical_node_path(valid_node)} validity alignment is invalid.")
    trailing = (slice(None),) * (len(shape) - 1)
    for start, stop in _row_blocks(node, row_count):
        selection = (slice(start, stop), *trailing)
        values = np.asarray(node[selection])
        valid = np.asarray(valid_node[start:stop], dtype=bool)
        node[selection] = project_subject_shape_points(
            values,
            offsets[start:stop],
            valid,
        ).astype(dtype, copy=False)


def _translate_bbox_node(node: Any, offsets: np.ndarray, valid_node: Any) -> None:
    row_count = int(offsets.shape[0])
    dtype, _shape = _require_row_array_metadata(
        node,
        row_count=row_count,
        trailing_shape=(4,),
    )
    if tuple(int(value) for value in valid_node.shape) != (row_count,):
        _fail(f"/{canonical_node_path(node)} bbox validity is misaligned.")
    for start, stop in _row_blocks(node, row_count):
        values = np.asarray(node[start:stop])
        valid = np.asarray(valid_node[start:stop], dtype=bool)
        node[start:stop] = project_subject_shape_bboxes(
            values,
            offsets[start:stop],
            valid,
        ).astype(dtype, copy=False)


def _translate_ellipse_node(node: Any, offsets: np.ndarray, valid_node: Any) -> None:
    row_count = int(offsets.shape[0])
    dtype, _shape = _require_row_array_metadata(
        node,
        row_count=row_count,
        trailing_shape=(5,),
    )
    if tuple(int(value) for value in valid_node.shape) != (row_count,):
        _fail(f"/{canonical_node_path(node)} ellipse validity is misaligned.")
    for start, stop in _row_blocks(node, row_count):
        values = np.asarray(node[start:stop])
        valid = np.asarray(valid_node[start:stop], dtype=bool)
        node[start:stop] = project_subject_shape_ellipses(
            values,
            offsets[start:stop],
            valid,
        ).astype(dtype, copy=False)


def _mask_invalid(node: Any, valid_node: Any) -> None:
    row_count = int(node.shape[0])
    dtype, shape = _require_row_array_metadata(node, row_count=row_count)
    if tuple(int(value) for value in valid_node.shape) != (row_count,):
        _fail(f"/{canonical_node_path(node)} validity alignment is invalid.")
    trailing = (slice(None),) * (len(shape) - 1)
    for start, stop in _row_blocks(node, row_count):
        selection = (slice(start, stop), *trailing)
        output = mask_invalid_subject_shape_vectors(
            np.asarray(node[selection]),
            np.asarray(valid_node[start:stop], dtype=bool),
        )
        node[selection] = output.astype(dtype, copy=False)


def _stamp_subject_shape_source_camera_coordinate_labels(
    run: Any,
    *,
    component_names: Sequence[str],
) -> None:
    components = run["components"]
    for name in component_names:
        group = components[str(name)]
        group.attrs["point_coordinate_space"] = "source_camera_image_px"
        group.attrs["bbox_coordinate_space"] = "source_camera_image_px"
        group.attrs["bbox_convention"] = "xyxy_pixel_edge_half_open"
    body = components.get("subject_body")
    if body is not None:
        body.attrs["principal_axis_semantics"] = (
            "unoriented_principal_axis_in_source_camera_xy"
        )
        body.attrs["tail_vector_coordinate_space"] = "source_camera_image_px"
        body.attrs["tail_vector_sampling_axis"] = "tail_sample_s"
    relations = run.get("relations")
    if relations is not None and "eyes_to_body" in relations:
        eyes = relations["eyes_to_body"]
        eyes.attrs["offset_coordinate_space"] = "source_camera_image_px"
        eyes.attrs["offset_semantics"] = "displacement_vector_not_overlay_position"
    run.attrs["point_coordinate_space"] = "source_camera_image_px"
    run.attrs["point_coordinate_transform"] = (
        "exact_rowwise_roi_to_source_camera_translation"
    )
    run.attrs["bbox_convention"] = "xyxy_pixel_edge_half_open"
    run.attrs["roi_local_point_arrays_retained"] = False


def transform_subject_shape_geometry_to_source_camera(
    run: Any,
    source: SubjectShapeCoordinateSource,
    *,
    component_names: Sequence[str],
) -> None:
    """Translate every persisted positional geometry surface exactly once."""

    continuous_offsets, edge_offsets = require_translation_only_subject_shape_placement(
        source
    )
    components = run["components"]
    for name in component_names:
        group = components[str(name)]
        _mask_invalid_and_translate_points_node(
            group["centroid_xy"],
            continuous_offsets,
            group["centroid_valid"],
        )
        _translate_bbox_node(group["bbox_xyxy"], edge_offsets, group["bbox_valid"])
        if "ellipse_params" in group:
            _translate_ellipse_node(group["ellipse_params"], continuous_offsets, group["ellipse_success"])
    body = components.get("subject_body")
    if body is not None:
        for point_name, valid_name in (
            ("centerline_xy", "centerline_valid"),
            ("bspline_control_points_xy", "bspline_valid"),
            ("bspline_sample_xy", "bspline_valid"),
            ("tail_sample_xy", "tail_sample_valid"),
            ("snout_tip_xy", "snout_tip_valid"),
            ("head_endpoint_xy", "centerline_valid"),
            ("tail_tip_xy", "centerline_valid"),
            ("tail_base_xy", "tail_base_valid"),
        ):
            if point_name in body:
                _mask_invalid_and_translate_points_node(
                    body[point_name],
                    continuous_offsets,
                    body[valid_name],
                )
        for vector_name, valid_name in (
            ("principal_axis_xy", "principal_axis_valid"),
            ("tail_tangent_xy", "tail_sample_valid"),
            ("tail_normal_xy", "tail_sample_valid"),
        ):
            if vector_name in body:
                _mask_invalid(body[vector_name], body[valid_name])

    swim = components.get("swim_bladder")
    if swim is not None and "caudal_contour_point_xy" in swim:
        _mask_invalid_and_translate_points_node(
            swim["caudal_contour_point_xy"],
            continuous_offsets,
            swim["caudal_contour_valid"],
        )

    relations = run.get("relations")
    if relations is not None and "eye_pair" in relations:
        eye_pair = relations["eye_pair"]
        _mask_invalid_and_translate_points_node(
            eye_pair["midpoint_xy"],
            continuous_offsets,
            eye_pair["midpoint_valid"],
        )
    if relations is not None and "eyes_to_body" in relations:
        eyes = relations["eyes_to_body"]
        for prefix in ("left", "right"):
            _mask_invalid(
                eyes[f"{prefix}_eye_offset_xy"],
                eyes[f"{prefix}_eye_relation_valid"],
            )
    _stamp_subject_shape_source_camera_coordinate_labels(
        run,
        component_names=component_names,
    )


def _create_array(group: Any, name: str, values: np.ndarray) -> Any:
    if name in group:
        _fail(f"Immutable subject-shape publication refuses occupied array {name!r}.")
    data = np.asarray(values)
    group_path = str(getattr(group, "path", "")).strip("/")
    path_parts = tuple(part for part in group_path.split("/") if part)
    if len(path_parts) >= 3 and path_parts[:2] == (
        "analysis",
        "subject_shape_runs",
    ):
        run_path = "/".join(path_parts[:3])
        try:
            import zarr

            run = zarr.open_group(
                store=group.store,
                path=run_path,
                mode="a",
                use_consolidated=False,
            )
        except (AttributeError, TypeError, ValueError):
            run = None
        if run is not None and _uses_access_aware_subject_shape_storage(run):
            # Import lazily to keep the logical coordinate contract independent
            # of either access-aware physical publication lifecycle.
            from fisheye.shared.subject_shape_storage import (
                create_bound_subject_shape_access_aware_array,
            )

            return create_bound_subject_shape_access_aware_array(
                run,
                group,
                name=name,
                values=data,
            )
    chunks = (max(1, min(int(data.shape[0]), 1024)), *data.shape[1:]) if data.ndim else None
    kwargs: dict[str, Any] = {"data": data, "overwrite": False}
    if chunks is not None:
        kwargs["chunks"] = chunks
    return group.create_array(name, **kwargs)


def _component_schema_record(component_names: Sequence[str]) -> dict[str, Any]:
    names = [str(value) for value in component_names]
    if tuple(names) != CANONICAL_SUBJECT_SHAPE_COMPONENT_ORDER:
        _fail(
            "Maintained subject-shape component order differs from the exact "
            f"{CANONICAL_SUBJECT_SHAPE_PROFILE_ID!r} profile."
        )
    return {
        "schema_id": SUBJECT_SHAPE_COMPONENT_SCHEMA_ID,
        "schema_version": SUBJECT_SHAPE_SCHEMA_VERSION,
        "components": names,
    }


_SCIENTIFIC_RUN_ATTRS = (
    "schema_id",
    "schema_version",
    "method",
    "method_version",
    "component_names",
    "relation_names",
    "body_frame_schema_id",
    "body_frame_schema_version",
    "body_frame_estimator",
    "tail_geometry_schema_id",
    "tail_geometry_schema_version",
    "snout_tip_semantic_label",
    "snout_tip_estimator",
    "snout_tip_estimator_version",
    "tail_anchor_method",
    "centerline_method",
    "centerline_skeleton_method",
    "centerline_snout_extension_method",
    "centerline_snout_join_method",
    "head_endpoint_semantics",
    "centerline_sample_count",
    "centerline_snout_check_method",
    "centerline_reaches_snout_threshold_px",
    "centerline_snout_join_max_arclength_px",
    "centerline_snout_extension_max_distance_px",
    "centerline_snout_extension_max_length_ratio",
    "centerline_snout_extension_max_extra_px",
    "bspline_method",
    "bspline_degree",
    "bspline_fit_mode",
    "bspline_smoothing",
    "bspline_arclength_sample_count",
    "tail_curvature_method",
    "tail_curvature_smoothing_px",
    "tail_sample_count",
    "tail_sample_domain",
    "centerline_crop_to_foreground",
)

_COMMON_COMPONENT_SCIENTIFIC_ATTRS = (
    "component_name",
    "source_component",
    "component_schema_id",
)
_SUBJECT_BODY_SCIENTIFIC_ATTRS = (
    "principal_axis_method",
    "snout_tip_semantic_label",
    "snout_tip_estimator",
    "snout_tip_estimator_version",
    "centerline_method",
    "centerline_skeleton_method",
    "centerline_snout_extension_method",
    "centerline_snout_join_method",
    "head_endpoint_semantics",
    "centerline_sample_count",
    "centerline_snout_check_method",
    "centerline_reaches_snout_threshold_px",
    "centerline_snout_join_max_arclength_px",
    "centerline_snout_extension_max_distance_px",
    "centerline_snout_extension_max_length_ratio",
    "centerline_snout_extension_max_extra_px",
    "bspline_method",
    "bspline_degree",
    "bspline_fit_mode",
    "bspline_smoothing",
    "bspline_arclength_sample_count",
    "tail_curvature_method",
    "tail_curvature_smoothing_px",
    "tail_sample_domain",
    "tail_sample_count",
    "tail_tip_semantic_label",
    "tail_tip_estimator",
    "tail_base_definition",
    "source_mask_qc_semantics",
)
_BODY_FRAME_SCIENTIFIC_ATTRS = (
    "body_frame_schema_id",
    "body_frame_schema_version",
    "body_frame_estimator",
    "body_frame_angle_convention",
    "origin_definition",
    "forward_axis_definition",
    "left_axis_definition",
)
_RELATION_SCIENTIFIC_ATTRS = {
    "eye_pair": ("relation_schema_id", "relation_components"),
    "swim_bladder_to_body": (
        "relation_schema_id",
        "relation_components",
        "axis_semantics",
    ),
    "eyes_to_body": (
        "relation_schema_id",
        "relation_components",
        "angle_semantics",
    ),
}

_COMMON_COMPONENT_ARRAYS = frozenset(
    {
        "mask_present",
        "area_px",
        "centroid_xy",
        "centroid_valid",
        "bbox_xyxy",
        "bbox_valid",
    }
)
_SUBJECT_BODY_ARRAYS = frozenset(
    {
        "principal_axis_xy",
        "principal_axis_valid",
        "principal_axis_length_px",
        "secondary_axis_length_px",
        "centerline_xy",
        "centerline_valid",
        "centerline_failure_reason_bytes",
        "head_endpoint_to_snout_distance_px",
        "centerline_reaches_snout",
        "centerline_snout_check_reason_bytes",
        "bspline_control_points_xy",
        "bspline_knots",
        "bspline_degree_used",
        "bspline_sample_xy",
        "bspline_valid",
        "bspline_failure_reason_bytes",
        "bspline_arc_length_px",
        "centerline_curvature_px_inv",
        "tail_sample_s",
        "tail_sample_xy",
        "tail_tangent_xy",
        "tail_normal_xy",
        "tail_curvature_px_inv",
        "tail_sample_valid",
        "tail_sample_failure_reason_bytes",
        "source_mask_qc_available",
        "source_mask_qc_severe_failure",
        "source_mask_qc_requires_review",
        "source_mask_qc_reason_bytes",
        "snout_tip_xy",
        "snout_tip_valid",
        "snout_tip_failure_reason_bytes",
        "head_endpoint_xy",
        "tail_tip_xy",
        "tail_base_xy",
        "tail_base_valid",
        "tail_base_arclength_px",
        "tail_base_failure_reason_bytes",
        "tail_segment_arclength_px",
        "body_arclength_px",
    }
)
_ELLIPSE_COMPONENT_ARRAYS = frozenset({"ellipse_params", "ellipse_success"})
_SWIM_BLADDER_ARRAYS = frozenset(
    {
        "caudal_contour_point_xy",
        "caudal_contour_projection_px",
        "caudal_contour_valid",
        "caudal_contour_failure_reason_bytes",
    }
)
_BODY_FRAME_ARRAYS = frozenset(
    {
        "origin_xy",
        "forward_axis_xy",
        "left_axis_xy",
        "heading_deg",
        "valid",
        "failure_reason_bytes",
    }
)
_RELATION_ARRAYS = {
    "eye_pair": frozenset(
        {"separation_px", "separation_valid", "midpoint_xy", "midpoint_valid"}
    ),
    "swim_bladder_to_body": frozenset(
        {
            "relation_valid",
            "distance_to_body_centroid_px",
            "longitudinal_offset_px",
            "lateral_offset_px",
        }
    ),
    "eyes_to_body": frozenset(
        {
            "left_eye_relation_valid",
            "left_eye_offset_xy",
            "left_eye_distance_to_body_centroid_px",
            "left_eye_axis_angle_to_body_rad",
            "right_eye_relation_valid",
            "right_eye_offset_xy",
            "right_eye_distance_to_body_centroid_px",
            "right_eye_axis_angle_to_body_rad",
        }
    ),
}
_ROW_INDEX_ALLOWED_ARRAYS = frozenset(
    {
        "frame_indices",
        "detection_indices",
        "source_refined_row_ids",
        "source_detect_row_index",
        "source_crop_row_ids",
        "instance_key",
    }
)
_ROW_INDEX_REQUIRED_ARRAYS = frozenset({"source_crop_row_ids", "instance_key"})

_REASON_ATTRS = frozenset(
    {"reason_encoding", "reason_bytes_width", "reason_bytes_null_terminated"}
)
_COMMON_COMPONENT_ATTRS = frozenset(
    {
        *_COMMON_COMPONENT_SCIENTIFIC_ATTRS,
        "point_coordinate_space",
        "bbox_coordinate_space",
        "bbox_convention",
    }
)
_SUBJECT_BODY_ATTRS = frozenset(
    {
        *_COMMON_COMPONENT_ATTRS,
        *_SUBJECT_BODY_SCIENTIFIC_ATTRS,
        *_REASON_ATTRS,
        "principal_axis_semantics",
    }
)
_SWIM_BLADDER_ATTRS = frozenset(
    {
        *_COMMON_COMPONENT_ATTRS,
        *_REASON_ATTRS,
        "ellipse_method",
        "caudal_anchor_method",
        "caudal_anchor_definition",
    }
)
_EYE_COMPONENT_ATTRS = frozenset(
    {*_COMMON_COMPONENT_ATTRS, "ellipse_method"}
)
_BODY_FRAME_ATTRS = frozenset(
    {
        *_BODY_FRAME_SCIENTIFIC_ATTRS,
        *_REASON_ATTRS,
        "body_frame_coordinate_space",
    }
)
_SOURCE_REVISION_ATTRS = frozenset(
    {
        "schema_id",
        "schema_version",
        "source_stage",
        "source_run",
        "source_path",
        "component_names",
        "row_revision_semantics",
    }
)


def subject_shape_maintained_profile_record() -> dict[str, Any]:
    """Return the one closed logical bundle accepted by maintained readers."""

    return {
        "profile_id": CANONICAL_SUBJECT_SHAPE_PROFILE_ID,
        "run_schema_id": CANONICAL_SUBJECT_SHAPE_RUN_SCHEMA_ID,
        "run_schema_version": CANONICAL_SUBJECT_SHAPE_RUN_SCHEMA_VERSION,
        "method": CANONICAL_SUBJECT_SHAPE_METHOD,
        "method_version": CANONICAL_SUBJECT_SHAPE_METHOD_VERSION,
        "component_order": list(CANONICAL_SUBJECT_SHAPE_COMPONENT_ORDER),
        "relation_order": list(CANONICAL_SUBJECT_SHAPE_RELATION_ORDER),
        "row_index_arrays": list(CANONICAL_SUBJECT_SHAPE_ROW_INDEX_ARRAYS),
        "row_lineage_copied": list(CANONICAL_SUBJECT_SHAPE_ROW_INDEX_ARRAYS),
        "row_lineage_missing": list(CANONICAL_SUBJECT_SHAPE_ROW_LINEAGE_MISSING),
        "historical_variant_policy": "explicit_historical_inspection_only",
        "closed_component_inventory": True,
        "closed_relation_inventory": True,
        "closed_row_index_inventory": True,
    }


def subject_shape_bundle_maintained_profile_record() -> dict[str, Any]:
    """Return the closed recording-bundle profile for new publications."""

    return {
        "profile_id": CANONICAL_SUBJECT_SHAPE_BUNDLE_PROFILE_ID,
        "run_schema_id": CANONICAL_SUBJECT_SHAPE_RUN_SCHEMA_ID,
        "run_schema_version": CANONICAL_SUBJECT_SHAPE_BUNDLE_RUN_SCHEMA_VERSION,
        "method": CANONICAL_SUBJECT_SHAPE_BUNDLE_METHOD,
        "method_version": CANONICAL_SUBJECT_SHAPE_BUNDLE_METHOD_VERSION,
        "source_kind": SUBJECT_SHAPE_BUNDLE_SOURCE_KIND,
        "component_order": list(CANONICAL_SUBJECT_SHAPE_COMPONENT_ORDER),
        "relation_order": list(CANONICAL_SUBJECT_SHAPE_RELATION_ORDER),
        "row_index_arrays": list(CANONICAL_SUBJECT_SHAPE_ROW_INDEX_ARRAYS),
        "row_lineage_copied": list(CANONICAL_SUBJECT_SHAPE_ROW_INDEX_ARRAYS),
        "row_lineage_missing": list(CANONICAL_SUBJECT_SHAPE_ROW_LINEAGE_MISSING),
        "historical_variant_policy": "separate_v4_reader_only",
        "closed_component_inventory": True,
        "closed_relation_inventory": True,
        "closed_row_index_inventory": True,
    }


def _require_subject_shape_maintained_profile(run: Any) -> dict[str, Any]:
    source_kind = run.attrs.get(
        SUBJECT_SHAPE_SOURCE_KIND_ATTR,
        SUBJECT_SHAPE_HISTORICAL_SOURCE_KIND,
    )
    bundle = source_kind == SUBJECT_SHAPE_BUNDLE_SOURCE_KIND
    if source_kind not in {
        SUBJECT_SHAPE_HISTORICAL_SOURCE_KIND,
        SUBJECT_SHAPE_BUNDLE_SOURCE_KIND,
    }:
        _fail(f"Unsupported maintained subject-shape source kind {source_kind!r}.")
    profile = (
        subject_shape_bundle_maintained_profile_record()
        if bundle
        else subject_shape_maintained_profile_record()
    )
    expected_schema_version = (
        CANONICAL_SUBJECT_SHAPE_BUNDLE_RUN_SCHEMA_VERSION
        if bundle
        else CANONICAL_SUBJECT_SHAPE_RUN_SCHEMA_VERSION
    )
    expected_method = (
        CANONICAL_SUBJECT_SHAPE_BUNDLE_METHOD
        if bundle
        else CANONICAL_SUBJECT_SHAPE_METHOD
    )
    expected_method_version = (
        CANONICAL_SUBJECT_SHAPE_BUNDLE_METHOD_VERSION
        if bundle
        else CANONICAL_SUBJECT_SHAPE_METHOD_VERSION
    )
    expected_row_axis = (
        "recording_subject_mask_bundle_rows"
        if bundle
        else "refined_subject_mask_rows"
    )
    if (
        run.attrs.get("schema_id") != CANONICAL_SUBJECT_SHAPE_RUN_SCHEMA_ID
        or run.attrs.get("schema_version") != expected_schema_version
        or run.attrs.get("method") != expected_method
        or run.attrs.get("method_version") != expected_method_version
        or run.attrs.get("row_axis") != expected_row_axis
    ):
        _fail(
            "Maintained subject-shape run identity differs from the exact "
            f"{profile['profile_id']!r} profile."
        )
    component_names = tuple(
        str(value) for value in (run.attrs.get("component_names") or ())
    )
    relation_names = tuple(
        str(value) for value in (run.attrs.get("relation_names") or ())
    )
    if component_names != CANONICAL_SUBJECT_SHAPE_COMPONENT_ORDER:
        _fail(
            "Maintained subject-shape component order differs from the exact "
            f"{profile['profile_id']!r} profile: "
            f"expected={CANONICAL_SUBJECT_SHAPE_COMPONENT_ORDER!r}, "
            f"observed={component_names!r}."
        )
    if relation_names != CANONICAL_SUBJECT_SHAPE_RELATION_ORDER:
        _fail(
            "Maintained subject-shape relation order differs from the exact "
            f"{profile['profile_id']!r} profile: "
            f"expected={CANONICAL_SUBJECT_SHAPE_RELATION_ORDER!r}, "
            f"observed={relation_names!r}."
        )
    row_index = run.get("row_index")
    observed_row_arrays = (
        frozenset(str(value) for value in row_index.array_keys())
        if row_index is not None
        else frozenset()
    )
    expected_row_arrays = frozenset(CANONICAL_SUBJECT_SHAPE_ROW_INDEX_ARRAYS)
    if observed_row_arrays != expected_row_arrays:
        _fail(
            "Maintained subject-shape row_index bundle differs from the exact "
            f"{profile['profile_id']!r} profile: "
            f"expected={sorted(expected_row_arrays)!r}, "
            f"observed={sorted(observed_row_arrays)!r}."
        )
    row_specs = {
        "source_crop_row_ids": np.dtype("int64"),
        "instance_key": np.dtype("uint64"),
    }
    row_count: int | None = None
    for name, dtype in row_specs.items():
        node = row_index[name]
        shape = tuple(int(value) for value in node.shape)
        if len(shape) != 1 or np.dtype(node.dtype) != dtype:
            _fail(
                f"Maintained subject-shape row_index/{name} requires exact "
                f"rank-1 {dtype.name}; got shape={shape!r}, dtype={node.dtype!r}."
            )
        if row_count is None:
            row_count = shape[0]
        elif shape[0] != row_count:
            _fail("Maintained subject-shape row_index arrays are not row aligned.")
    if tuple(run.attrs.get("row_lineage_copied") or ()) != (
        CANONICAL_SUBJECT_SHAPE_ROW_INDEX_ARRAYS
    ):
        _fail("Maintained subject-shape row_lineage_copied declaration is not exact.")
    if tuple(run.attrs.get("row_lineage_missing") or ()) != (
        CANONICAL_SUBJECT_SHAPE_ROW_LINEAGE_MISSING
    ):
        _fail("Maintained subject-shape row_lineage_missing declaration is not exact.")
    source_revisions = run.get("source_refined_subject_masks")
    source_component_names = (
        tuple(
            str(value)
            for value in (source_revisions.attrs.get("component_names") or ())
        )
        if source_revisions is not None
        else ()
    )
    if source_component_names != CANONICAL_SUBJECT_SHAPE_COMPONENT_ORDER:
        _fail(
            "Maintained subject-shape source-revision component order is not exact."
        )
    revision = source_revisions["row_revision"]
    revision_available = source_revisions["row_revision_available"]
    if (
        tuple(int(value) for value in revision.shape)
        != (row_count, len(CANONICAL_SUBJECT_SHAPE_COMPONENT_ORDER))
        or np.dtype(revision.dtype) != np.dtype("int64")
        or tuple(int(value) for value in revision_available.shape)
        != (len(CANONICAL_SUBJECT_SHAPE_COMPONENT_ORDER),)
        or np.dtype(revision_available.dtype) != np.dtype("bool")
    ):
        _fail("Maintained subject-shape source-revision arrays are not exact.")
    return profile

_RUN_OPERATIONAL_ATTRS = frozenset(
    {
        "stage_selector_eligible",
        SUBJECT_SHAPE_COORDINATE_BINDING_STATUS_ATTR,
        RUN_COMPLETION_CONTRACT_ATTR,
        RUN_COMPLETION_STATUS_ATTR,
        "palette_run_started_at_utc",
        "palette_run_completed_at_utc",
        "palette_run_failed_at_utc",
        "palette_run_error",
        "palette_run_name",
        "palette_run_stage",
        "palette_run_provenance",
        "run_provenance",
        "created_at_utc",
        "created_utc",
        "git_commit",
        "git_branch",
        "provenance",
        "execution_backend",
        "dask_execution_enabled",
        "dask_scheduler",
        "dask_num_workers",
        "dask_requested_chunk_size",
        "dask_chunk_size",
        "dask_chunk_alignment",
        "dask_version",
        "chunk_size",
        "worker_chunk_size",
        "chunk_count",
        "native_threads_per_worker",
        "duration_seconds",
        "rows_per_second",
        "rows_with_component",
        "subject_shape_timing_summary",
        "subject_shape_chunk_timing_count",
        "subject_shape_chunk_timing_storage",
        "subject_shape_chunk_timings",
        "source_fingerprint",
        "source_lineage_hash",
        "lineage_hash",
        "fingerprint_status",
        "lineage_fingerprint_schema_id",
        "lineage_fingerprint_schema_version",
        "lineage_fingerprint_canonicalization",
        "lineage_payload_json",
        "physical_storage_layout",
        "cluster_output_staging",
        SUBJECT_SHAPE_STORAGE_PLAN_ATTR,
        SUBJECT_SHAPE_STORAGE_PLAN_DIGEST_ATTR,
        SUBJECT_SHAPE_STORAGE_PROFILE_ID_ATTR,
        SUBJECT_SHAPE_STORAGE_PROFILE_ROLE_ATTR,
        SUBJECT_SHAPE_STORAGE_CANDIDATE_ATTR,
        SUBJECT_SHAPE_DEFERRED_STORAGE_RECEIPT_ATTR,
        SUBJECT_SHAPE_STORAGE_SOURCE_MANIFEST_ATTR,
        f"{SUBJECT_SHAPE_STORAGE_SOURCE_MANIFEST_ATTR}_sha256",
        SUBJECT_SHAPE_STORAGE_METADATA_POLICY_ATTR,
        SUBJECT_SHAPE_PAYLOAD_INTEGRITY_RECEIPT_ATTR,
        SUBJECT_SHAPE_PAYLOAD_VALIDATION_RECEIPT_ATTR,
    }
)
_RUN_AUTHORITATIVE_ATTRS = frozenset(
    {
        *_SCIENTIFIC_RUN_ATTRS,
        SUBJECT_SHAPE_PUBLICATION_OWNER_ATTR,
        "row_axis",
        "source_refined_subject_masks_run",
        "source_refined_subject_masks_stage",
        "source_mask_labels",
        "source_mask_label_schema_id",
        "source_mask_geometry_schema_id",
        "source_mask_store_encoding",
        "source_mask_storage_surface",
        "source_mask_store_path",
        "source_body_mask_qc_available",
        "source_body_mask_qc_schema_id",
        "source_component_review_states",
        "source_refs",
        "body_frame_source_refs",
        "row_lineage_copied",
        "row_lineage_missing",
        SUBJECT_SHAPE_SOURCE_KIND_ATTR,
        SUBJECT_SHAPE_BUNDLE_ID_ATTR,
        SUBJECT_SHAPE_BUNDLE_ACTIVE_AT_DERIVATION_ATTR,
        SUBJECT_SHAPE_SOURCE_BINDING_ATTR,
        SUBJECT_SHAPE_SOURCE_BINDING_DIGEST_ATTR,
        SUBJECT_SHAPE_PAYLOAD_RECEIPT_PROFILE_ATTR,
        SUBJECT_SHAPE_NUMERIC_PROJECTION_ATTR,
        SUBJECT_SHAPE_NUMERIC_PROJECTION_DIGEST_ATTR,
    }
)
_RUN_FINAL_AUTHORITATIVE_ATTRS = frozenset(
    {
        "point_coordinate_space",
        "point_coordinate_transform",
        "bbox_convention",
        "roi_local_point_arrays_retained",
        "unbound_numeric_stage_manifest_sha256_consumed",
        ROW_IDENTITY_CONTRACT_ATTR,
        ROW_IDENTITY_CONTRACT_DIGEST_ATTR,
        SOURCE_ROW_TEMPORAL_AUTHORITY_ATTR,
        SOURCE_ROW_TEMPORAL_AUTHORITY_DIGEST_ATTR,
        BODY_ESTIMATOR_SOURCE_MANIFEST_ATTR,
        f"{BODY_ESTIMATOR_SOURCE_MANIFEST_ATTR}_sha256",
        SUBJECT_SHAPE_DERIVATION_ATTR,
        f"{SUBJECT_SHAPE_DERIVATION_ATTR}_sha256",
    }
)
def _require_selected_attrs(
    node: Any,
    names: Sequence[str],
    *,
    label: str,
) -> dict[str, Any]:
    missing = [name for name in names if name not in node.attrs]
    if missing:
        _fail(f"Subject-shape {label} lacks controlled attrs {missing!r}.")
    return {name: _json_copy(node.attrs[name]) for name in names}


def _json_copy(value: Any) -> Any:
    return json.loads(_canonical_json(value))


def build_subject_shape_scientific_configuration_record(run: Any) -> dict[str, Any]:
    """Return the complete controlled configuration that produced the payload.

    Coordinate-publication attrs are intentionally absent: ROI/source labels,
    descriptors, and body-frame publication records legitimately change at the
    final-path binding boundary.  Every algorithm choice and anatomical
    estimator parameter is selected by an explicit vocabulary instead of by
    copying arbitrary live attrs into provenance.
    """

    maintained_profile = _require_subject_shape_maintained_profile(run)
    run_attrs = _require_selected_attrs(
        run,
        _SCIENTIFIC_RUN_ATTRS,
        label="scientific run configuration",
    )
    component_names = tuple(str(value) for value in run_attrs["component_names"])
    relation_names = tuple(str(value) for value in run_attrs["relation_names"])
    groups: dict[str, Any] = {}
    for component in component_names:
        path = f"components/{component}"
        group = run.get(path)
        if group is None:
            _fail(f"Scientific configuration group {path!r} is unavailable.")
        names: tuple[str, ...] = _COMMON_COMPONENT_SCIENTIFIC_ATTRS
        if component == "subject_body":
            names = (*names, *_SUBJECT_BODY_SCIENTIFIC_ATTRS)
        elif component == "swim_bladder":
            names = (*names, "ellipse_method", "caudal_anchor_method", "caudal_anchor_definition")
        elif component in {"eye_left", "eye_right"}:
            names = (*names, "ellipse_method")
        else:
            _fail(f"Unsupported subject-shape component {component!r}.")
        groups[path] = _require_selected_attrs(
            group,
            names,
            label=f"scientific group {path!r}",
        )
    for relation in relation_names:
        path = f"relations/{relation}"
        group = run.get(path)
        names = _RELATION_SCIENTIFIC_ATTRS.get(relation)
        if group is None or names is None:
            _fail(f"Unsupported scientific relation group {path!r}.")
        groups[path] = _require_selected_attrs(
            group,
            names,
            label=f"scientific group {path!r}",
        )
    body_frame = run.get("body_frame")
    if body_frame is None:
        _fail("Scientific configuration group 'body_frame' is unavailable.")
    groups["body_frame"] = _require_selected_attrs(
        body_frame,
        _BODY_FRAME_SCIENTIFIC_ATTRS,
        label="scientific group 'body_frame'",
    )
    return {
        "schema_id": SUBJECT_SHAPE_SCIENTIFIC_CONFIGURATION_SCHEMA_ID,
        "schema_version": SUBJECT_SHAPE_SCHEMA_VERSION,
        "maintained_profile": maintained_profile,
        "run_ref": f"/{canonical_node_path(run)}",
        "run_attrs": run_attrs,
        "group_attrs": groups,
        "closed_run_attr_vocabulary": list(_SCIENTIFIC_RUN_ATTRS),
        "closed_group_attr_vocabulary": {
            path: sorted(attrs) for path, attrs in groups.items()
        },
        "scope": (
            "all_controlled_scientific_parameters_used_by_subject_shape_v12"
            if run.attrs.get(SUBJECT_SHAPE_SOURCE_KIND_ATTR)
            == SUBJECT_SHAPE_BUNDLE_SOURCE_KIND
            else "all_controlled_scientific_parameters_used_by_subject_shape_v11"
        ),
    }


def _scientific_configuration_record(run: Any) -> dict[str, Any]:
    return build_subject_shape_scientific_configuration_record(run)


def _iter_group_paths(group: Any, prefix: str = "") -> Iterable[str]:
    for name in sorted(str(value) for value in group.group_keys()):
        path = f"{prefix}/{name}" if prefix else name
        yield path
        yield from _iter_group_paths(group[name], path)


def _component_array_names(component: str) -> frozenset[str]:
    names = set(_COMMON_COMPONENT_ARRAYS)
    if component == "subject_body":
        names.update(_SUBJECT_BODY_ARRAYS)
    elif component == "swim_bladder":
        names.update(_ELLIPSE_COMPONENT_ARRAYS)
        names.update(_SWIM_BLADDER_ARRAYS)
    elif component in {"eye_left", "eye_right"}:
        names.update(_ELLIPSE_COMPONENT_ARRAYS)
    else:
        _fail(f"Unsupported subject-shape component {component!r}.")
    return frozenset(names)


def _expected_subject_shape_group_paths(
    run: Any,
    *,
    phase: str,
) -> tuple[str, ...]:
    if phase not in {"unbound", "bound"}:
        _fail(f"Unsupported subject-shape schema phase {phase!r}.")
    components = tuple(str(value) for value in (run.attrs.get("component_names") or ()))
    relations = tuple(str(value) for value in (run.attrs.get("relation_names") or ()))
    paths = {
        "row_index",
        "components",
        *(f"components/{name}" for name in components),
        "relations",
        *(f"relations/{name}" for name in relations),
        "body_frame",
        "source_refined_subject_masks",
    }
    if phase == "bound":
        paths.update(
            {
                "coordinate_records",
                "coordinate_records/component_schema",
                "coordinate_records/scientific_configuration",
                "coordinate_records/consumed_unbound_stage",
                "coordinate_records/body_frame_contract",
                "coordinate_records/body_frame_estimator",
                "coordinate_records/scalar_surface_inventory",
            }
        )
        if (
            run.attrs.get(SUBJECT_SHAPE_SOURCE_KIND_ATTR)
            == SUBJECT_SHAPE_BUNDLE_SOURCE_KIND
        ):
            paths.add("coordinate_records/source_binding")
    return tuple(sorted(paths))


def _expected_subject_shape_array_paths(
    run: Any,
    *,
    phase: str,
) -> tuple[str, ...]:
    components = tuple(str(value) for value in (run.attrs.get("component_names") or ()))
    relations = tuple(str(value) for value in (run.attrs.get("relation_names") or ()))
    paths: set[str] = set()
    row_index = run.get("row_index")
    observed_row_names = (
        {str(value) for value in row_index.array_keys()} if row_index is not None else set()
    )
    unknown_row_names = observed_row_names - _ROW_INDEX_ALLOWED_ARRAYS
    missing_row_names = _ROW_INDEX_REQUIRED_ARRAYS - observed_row_names
    if unknown_row_names or missing_row_names:
        _fail(
            "Subject-shape row_index schema differs from its controlled vocabulary: "
            f"unknown={sorted(unknown_row_names)!r}, missing={sorted(missing_row_names)!r}."
        )
    paths.update(f"row_index/{name}" for name in observed_row_names)
    for component in components:
        paths.update(
            f"components/{component}/{name}"
            for name in _component_array_names(component)
        )
    for relation in relations:
        names = _RELATION_ARRAYS.get(relation)
        if names is None:
            _fail(f"Unsupported subject-shape relation {relation!r}.")
        paths.update(f"relations/{relation}/{name}" for name in names)
    paths.update(f"body_frame/{name}" for name in _BODY_FRAME_ARRAYS)
    paths.update(
        {
            "source_refined_subject_masks/row_revision",
            "source_refined_subject_masks/row_revision_available",
        }
    )
    if phase == "bound":
        paths.update(
            {
                "instance_key",
                "source_crop_row_ids",
                "source_acquisition_frame_index",
                "component_centroid_xy",
                "component_centroid_valid",
                "body_frame/axis_valid",
            }
        )
    elif phase != "unbound":
        _fail(f"Unsupported subject-shape schema phase {phase!r}.")
    return tuple(sorted(paths))


def _subject_shape_array_role(run: Any, path: str, *, phase: str) -> str:
    if phase == "bound" and path in _geometry_specs(
        run,
        tuple(str(value) for value in run.attrs["component_names"]),
    ):
        return "coordinate_geometry"
    if phase == "bound" and path in _scalar_surface_specs(run):
        return "scalar_measurement"
    if path == "instance_key" or path.endswith("/instance_key"):
        return "row_identity"
    if path in {"source_crop_row_ids", "source_acquisition_frame_index"}:
        return "source_row_identity_or_time"
    if path.startswith("row_index/"):
        return "compatibility_row_lineage"
    if path.endswith("/tail_sample_s"):
        return "sample_axis"
    if path.startswith("source_refined_subject_masks/"):
        return "source_revision_lineage"
    name = path.rsplit("/", 1)[-1]
    if name.endswith("failure_reason_bytes") or name.endswith("reason_bytes"):
        return "reason_code"
    if (
        name.endswith("_valid")
        or name.endswith("_success")
        or name.endswith("_present")
        or name.endswith("_available")
        or name.endswith("_failure")
        or name.endswith("_review")
        or name == "centerline_reaches_snout"
    ):
        return "validity_or_flag"
    return "scientific_support"


def _expected_subject_shape_group_attrs(
    run: Any,
    *,
    phase: str,
) -> dict[str, frozenset[str]]:
    components = tuple(str(value) for value in run.attrs["component_names"])
    relations = tuple(str(value) for value in run.attrs["relation_names"])
    expected: dict[str, frozenset[str]] = {
        "row_index": frozenset(),
        "components": frozenset(),
        "relations": frozenset(),
        "source_refined_subject_masks": _SOURCE_REVISION_ATTRS,
    }
    for component in components:
        path = f"components/{component}"
        if component == "subject_body":
            attrs = set(_SUBJECT_BODY_ATTRS)
            if phase == "bound":
                attrs.update({"tail_vector_coordinate_space", "tail_vector_sampling_axis"})
        elif component == "swim_bladder":
            attrs = set(_SWIM_BLADDER_ATTRS)
        elif component in {"eye_left", "eye_right"}:
            attrs = set(_EYE_COMPONENT_ATTRS)
        else:
            _fail(f"Unsupported subject-shape component {component!r}.")
        expected[path] = frozenset(attrs)
    for relation in relations:
        names = _RELATION_SCIENTIFIC_ATTRS.get(relation)
        if names is None:
            _fail(f"Unsupported subject-shape relation {relation!r}.")
        attrs = set(names)
        if phase == "bound" and relation == "eyes_to_body":
            attrs.update({"offset_coordinate_space", "offset_semantics"})
        expected[f"relations/{relation}"] = frozenset(attrs)
    body_attrs = set(_BODY_FRAME_ATTRS)
    if phase == "bound":
        body_attrs.update(
            {
                "axis_valid_array",
                "valid_compatibility_alias_of",
                FISH_ANATOMICAL_BODY_FRAME_ATTR,
                f"{FISH_ANATOMICAL_BODY_FRAME_ATTR}_sha256",
            }
        )
    expected["body_frame"] = frozenset(body_attrs)
    if phase == "bound":
        expected.update(
            {
                "coordinate_records": frozenset(),
                "coordinate_records/component_schema": frozenset(
                    {
                        SUBJECT_SHAPE_COMPONENT_SCHEMA_ATTR,
                        f"{SUBJECT_SHAPE_COMPONENT_SCHEMA_ATTR}_sha256",
                    }
                ),
                "coordinate_records/scientific_configuration": frozenset(
                    {
                        SUBJECT_SHAPE_SCIENTIFIC_CONFIGURATION_ATTR,
                        f"{SUBJECT_SHAPE_SCIENTIFIC_CONFIGURATION_ATTR}_sha256",
                    }
                ),
                "coordinate_records/consumed_unbound_stage": frozenset(
                    {
                        SUBJECT_SHAPE_CONSUMED_UNBOUND_STAGE_ATTR,
                        f"{SUBJECT_SHAPE_CONSUMED_UNBOUND_STAGE_ATTR}_sha256",
                    }
                ),
                "coordinate_records/body_frame_contract": frozenset(
                    {BODY_FRAME_CONTRACT_ATTR, f"{BODY_FRAME_CONTRACT_ATTR}_sha256"}
                ),
                "coordinate_records/body_frame_estimator": frozenset(
                    {BODY_FRAME_ESTIMATOR_ATTR, f"{BODY_FRAME_ESTIMATOR_ATTR}_sha256"}
                ),
                "coordinate_records/scalar_surface_inventory": frozenset(
                    {
                        SUBJECT_SHAPE_SCALAR_SURFACE_INVENTORY_ATTR,
                        f"{SUBJECT_SHAPE_SCALAR_SURFACE_INVENTORY_ATTR}_sha256",
                    }
                ),
            }
        )
        if (
            run.attrs.get(SUBJECT_SHAPE_SOURCE_KIND_ATTR)
            == SUBJECT_SHAPE_BUNDLE_SOURCE_KIND
        ):
            expected["coordinate_records/source_binding"] = frozenset(
                {
                    SUBJECT_SHAPE_SOURCE_BINDING_ATTR,
                    f"{SUBJECT_SHAPE_SOURCE_BINDING_ATTR}_sha256",
                }
            )
    return expected


def _expected_subject_shape_array_attrs(
    run: Any,
    array_paths: Sequence[str],
    *,
    phase: str,
) -> dict[str, frozenset[str]]:
    access_aware_storage = _uses_access_aware_subject_shape_storage(run)
    physical_attrs = SUBJECT_SHAPE_STORAGE_ARRAY_ATTRS if access_aware_storage else ()
    expected = {path: frozenset(physical_attrs) for path in array_paths}
    if phase == "unbound":
        return expected
    identity_attrs = {
        ROW_IDENTITY_KEY_ATTR,
        ROW_IDENTITY_KEY_DIGEST_ATTR,
        ROW_IDENTITY_CONTRACT_REF_ATTR,
        ROW_IDENTITY_CONTRACT_DIGEST_ATTR,
    }
    expected["instance_key"] = frozenset(
        set(expected["instance_key"]) | identity_attrs
    )
    descriptor_attrs = {
        COORDINATE_DESCRIPTOR_ATTR,
        f"{COORDINATE_DESCRIPTOR_ATTR}_sha256",
        f"{COORDINATE_DESCRIPTOR_ATTR}_owner_dtype",
    }
    for path in _geometry_specs(
        run,
        tuple(str(value) for value in run.attrs["component_names"]),
    ):
        expected[path] = frozenset(set(expected[path]) | descriptor_attrs)
    scalar_attrs = {
        SUBJECT_SHAPE_SCALAR_SURFACE_ATTR,
        f"{SUBJECT_SHAPE_SCALAR_SURFACE_ATTR}_sha256",
    }
    for path in _scalar_surface_specs(run):
        expected[path] = frozenset(set(expected[path]) | scalar_attrs)
    tail_axis_path = "components/subject_body/tail_sample_s"
    expected[tail_axis_path] = frozenset(
        set(expected[tail_axis_path])
        | {
            SUBJECT_SHAPE_TAIL_SAMPLE_AXIS_ATTR,
            f"{SUBJECT_SHAPE_TAIL_SAMPLE_AXIS_ATTR}_sha256",
        }
    )
    heading_path = "body_frame/heading_deg"
    expected[heading_path] = frozenset(
        set(expected[heading_path])
        | {
            SUBJECT_SHAPE_HEADING_SEMANTICS_ATTR,
            f"{SUBJECT_SHAPE_HEADING_SEMANTICS_ATTR}_sha256",
        }
    )
    return expected


def _attribute_inventory_entry(node: Any, names: Sequence[str]) -> dict[str, Any]:
    payload = {name: _json_copy(node.attrs[name]) for name in sorted(names)}
    return {
        "names": sorted(names),
        "content_sha256": _canonical_sha256(payload),
        "canonicalization": "finite_canonical_json_sorted_attribute_mapping_v1",
    }


def build_subject_shape_schema_inventory_record(
    run: Any,
    *,
    phase: str,
) -> dict[str, Any]:
    """Validate and describe the exact controlled subject-shape schema.

    Traversal observes the live tree only for comparison.  The schema itself is
    generated from explicit producer-owned group, array, role, and attribute
    vocabularies; an extra array, empty group, or attr is never promoted into
    the schema merely because it exists.
    """

    maintained_profile = _require_subject_shape_maintained_profile(run)
    group_paths = _expected_subject_shape_group_paths(run, phase=phase)
    array_paths = _expected_subject_shape_array_paths(run, phase=phase)
    observed_groups = tuple(sorted(_iter_group_paths(run)))
    observed_arrays = tuple(sorted(path for path, _node_value in _iter_arrays(run)))
    if observed_groups != group_paths:
        _fail(
            "Subject-shape group inventory differs from the controlled schema: "
            f"expected={group_paths!r}, observed={observed_groups!r}."
        )
    if observed_arrays != array_paths:
        _fail(
            "Subject-shape array inventory differs from the controlled schema: "
            f"expected={array_paths!r}, observed={observed_arrays!r}."
        )

    group_attr_names = _expected_subject_shape_group_attrs(run, phase=phase)
    array_attr_names = _expected_subject_shape_array_attrs(
        run,
        array_paths,
        phase=phase,
    )
    attrs: dict[str, Any] = {}
    for path in group_paths:
        node = run[path]
        expected_names = group_attr_names[path]
        observed_names = frozenset(str(value) for value in node.attrs.keys())
        if observed_names != expected_names:
            _fail(
                f"Subject-shape group {path!r} attrs differ from the controlled "
                f"schema: expected={sorted(expected_names)!r}, "
                f"observed={sorted(observed_names)!r}."
            )
        attrs[path] = _attribute_inventory_entry(node, expected_names)
    for path in array_paths:
        node = run[path]
        expected_names = array_attr_names[path]
        observed_names = frozenset(str(value) for value in node.attrs.keys())
        if observed_names != expected_names:
            _fail(
                f"Subject-shape array {path!r} attrs differ from the controlled "
                f"schema: expected={sorted(expected_names)!r}, "
                f"observed={sorted(observed_names)!r}."
            )
        attrs[path] = _attribute_inventory_entry(node, expected_names)

    authoritative_run_names = set(_RUN_AUTHORITATIVE_ATTRS)
    if phase == "bound":
        authoritative_run_names.update(_RUN_FINAL_AUTHORITATIVE_ATTRS)
        self_names = {
            SUBJECT_SHAPE_MANIFEST_ATTR,
            f"{SUBJECT_SHAPE_MANIFEST_ATTR}_sha256",
            "publication_manifest_sha256",
            "coordinate_contract",
        }
    else:
        self_names = {
            SUBJECT_SHAPE_UNBOUND_MANIFEST_ATTR,
            f"{SUBJECT_SHAPE_UNBOUND_MANIFEST_ATTR}_sha256",
        }
    observed_run_names = {str(value) for value in run.attrs.keys()}
    allowed_run_names = authoritative_run_names | set(_RUN_OPERATIONAL_ATTRS) | self_names
    unknown_run_names = observed_run_names - allowed_run_names
    if unknown_run_names:
        _fail(
            "Subject-shape run attrs contain names outside the controlled "
            f"vocabulary: {sorted(unknown_run_names)!r}."
        )
    required_run_names = {
        *_SCIENTIFIC_RUN_ATTRS,
        SUBJECT_SHAPE_PUBLICATION_OWNER_ATTR,
        "row_axis",
        "source_refined_subject_masks_run",
        "source_refined_subject_masks_stage",
        "source_mask_labels",
        "source_mask_label_schema_id",
        "source_mask_geometry_schema_id",
        "source_mask_store_encoding",
        "source_mask_storage_surface",
        "source_mask_store_path",
        "source_body_mask_qc_available",
        "source_body_mask_qc_schema_id",
        "source_component_review_states",
        "source_refs",
        "body_frame_source_refs",
        "row_lineage_copied",
        "row_lineage_missing",
    }
    if (
        run.attrs.get(SUBJECT_SHAPE_SOURCE_KIND_ATTR)
        == SUBJECT_SHAPE_BUNDLE_SOURCE_KIND
    ):
        required_run_names.update(
            {
                SUBJECT_SHAPE_SOURCE_KIND_ATTR,
                SUBJECT_SHAPE_BUNDLE_ID_ATTR,
                SUBJECT_SHAPE_BUNDLE_ACTIVE_AT_DERIVATION_ATTR,
                SUBJECT_SHAPE_SOURCE_BINDING_ATTR,
                SUBJECT_SHAPE_SOURCE_BINDING_DIGEST_ATTR,
            }
        )
    if phase == "bound":
        required_run_names.update(_RUN_FINAL_AUTHORITATIVE_ATTRS)
    missing_run_names = required_run_names - observed_run_names
    if missing_run_names:
        _fail(
            "Subject-shape run lacks required controlled attrs: "
            f"{sorted(missing_run_names)!r}."
        )
    stable_run_names = sorted(observed_run_names & authoritative_run_names)
    attrs["."] = _attribute_inventory_entry(run, stable_run_names)
    return {
        "schema_id": SUBJECT_SHAPE_SCHEMA_INVENTORY_SCHEMA_ID,
        "schema_version": SUBJECT_SHAPE_SCHEMA_VERSION,
        "maintained_profile": maintained_profile,
        "phase": phase,
        "run_ref": f"/{canonical_node_path(run)}",
        "groups": list(group_paths),
        "arrays": {
            path: {"role": _subject_shape_array_role(run, path, phase=phase)}
            for path in array_paths
        },
        "attrs": attrs,
        "excluded_operational_run_attrs": sorted(_RUN_OPERATIONAL_ATTRS),
        "excluded_self_referential_run_attrs": sorted(self_names),
        "closed_group_inventory": True,
        "closed_array_inventory": True,
        "closed_attr_inventory": True,
    }


def build_subject_shape_unbound_numeric_manifest_record(
    run: Any,
    *,
    array_content_sha256: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Build the producer seal for one unbound subject-shape numeric payload."""

    schema_inventory = build_subject_shape_schema_inventory_record(
        run,
        phase="unbound",
    )
    expected_paths = set(schema_inventory["arrays"])
    if array_content_sha256 is not None and set(array_content_sha256) != expected_paths:
        _fail("Subject-shape unbound digest receipt inventory is not exact.")
    arrays: dict[str, dict[str, Any]] = {}
    for path in schema_inventory["arrays"]:
        node = run[path]
        digest = (
            array_payload_sha256(node)
            if array_content_sha256 is None
            else array_content_sha256[path]
        )
        if type(digest) is not str or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            _fail(f"Subject-shape unbound digest receipt is invalid for {path!r}.")
        arrays[path] = {
            "relative_ref": path,
            "dtype": np.dtype(node.dtype).str,
            "shape": [int(value) for value in node.shape],
            "content_sha256": digest,
            "canonicalization": "numpy_dtype_shape_c_order_bytes_v1",
        }
    projection = run.attrs.get(SUBJECT_SHAPE_NUMERIC_PROJECTION_ATTR)
    projection_digest = run.attrs.get(SUBJECT_SHAPE_NUMERIC_PROJECTION_DIGEST_ATTR)
    projected = projection is not None or projection_digest is not None
    if projected and (
        not isinstance(projection, Mapping)
        or not isinstance(projection_digest, str)
        or projection_digest != _canonical_sha256(projection)
    ):
        _fail("Subject-shape numeric projection evidence is incomplete or stale.")
    result = {
        "schema_id": SUBJECT_SHAPE_UNBOUND_MANIFEST_SCHEMA_ID,
        "schema_version": (
            SUBJECT_SHAPE_PROJECTED_UNBOUND_MANIFEST_SCHEMA_VERSION
            if projected
            else 1
        ),
        "run_name": run.attrs.get("palette_run_name"),
        "binding_status": (
            SUBJECT_SHAPE_PROJECTED_UNBOUND_BINDING_STATUS
            if projected
            else "unbound_roi_local_numeric_payload"
        ),
        "source_refined_subject_masks_run": run.attrs.get(
            "source_refined_subject_masks_run"
        ),
        "method": run.attrs.get("method"),
        "method_version": run.attrs.get("method_version"),
        "component_names": list(run.attrs.get("component_names") or ()),
        "scientific_configuration": (
            build_subject_shape_scientific_configuration_record(run)
        ),
        "schema_inventory": schema_inventory,
        "arrays": arrays,
        "closed_array_inventory": True,
        "closed_group_inventory": True,
        "closed_attr_inventory": True,
        "coordinate_descriptors_present": False,
    }
    if (
        run.attrs.get(SUBJECT_SHAPE_SOURCE_KIND_ATTR)
        == SUBJECT_SHAPE_BUNDLE_SOURCE_KIND
    ):
        result["source_binding"] = {
            "source_kind": run.attrs.get(SUBJECT_SHAPE_SOURCE_KIND_ATTR),
            "bundle_id": run.attrs.get(SUBJECT_SHAPE_BUNDLE_ID_ATTR),
            "bundle_active_at_derivation": run.attrs.get(
                SUBJECT_SHAPE_BUNDLE_ACTIVE_AT_DERIVATION_ATTR
            ),
            "record_sha256": run.attrs.get(SUBJECT_SHAPE_SOURCE_BINDING_DIGEST_ATTR),
        }
    return result


def stamp_unbound_subject_shape_manifest(
    run: Any,
    *,
    array_content_sha256: Mapping[str, str] | None = None,
) -> BoundCoordinateRecord:
    """Persist and digest-bind the exact unbound producer seal."""

    return stamp_and_bind_persisted_coordinate_record(
        run,
        build_subject_shape_unbound_numeric_manifest_record(
            run,
            array_content_sha256=array_content_sha256,
        ),
        attr_name=SUBJECT_SHAPE_UNBOUND_MANIFEST_ATTR,
    )


def load_unbound_subject_shape_manifest(
    run: Any,
    *,
    array_content_sha256: Mapping[str, str] | None = None,
) -> BoundCoordinateRecord:
    """Load an unbound producer seal only when it matches the live payload."""

    result = bind_persisted_coordinate_record(
        run,
        attr_name=SUBJECT_SHAPE_UNBOUND_MANIFEST_ATTR,
    )
    if result.record != build_subject_shape_unbound_numeric_manifest_record(
        run,
        array_content_sha256=array_content_sha256,
    ):
        _fail("Subject-shape unbound numeric manifest differs from live arrays.")
    return result


def load_sealed_unbound_subject_shape_manifest(run: Any) -> BoundCoordinateRecord:
    """Require one complete, selector-ineligible, producer-sealed unbound run."""

    if (
        run.attrs.get(SUBJECT_SHAPE_COORDINATE_BINDING_STATUS_ATTR)
        != SUBJECT_SHAPE_UNBOUND_STAGE_STATUS
        or run.attrs.get("palette_run_completion_status") != "complete"
        or run.attrs.get("stage_selector_eligible") is not False
    ):
        _fail(
            "Subject-shape storage conversion requires one producer-sealed, "
            "complete, unbound, selector-ineligible stage."
        )
    return load_unbound_subject_shape_manifest(run)


def _stamp_scientific_configuration(run: Any) -> BoundCoordinateRecord:
    node = run["coordinate_records"].require_group("scientific_configuration")
    return stamp_and_bind_persisted_coordinate_record(
        node,
        _scientific_configuration_record(run),
        attr_name=SUBJECT_SHAPE_SCIENTIFIC_CONFIGURATION_ATTR,
    )


def _load_scientific_configuration(run: Any) -> BoundCoordinateRecord:
    node = run["coordinate_records/scientific_configuration"]
    result = bind_persisted_coordinate_record(
        node,
        attr_name=SUBJECT_SHAPE_SCIENTIFIC_CONFIGURATION_ATTR,
    )
    if result.record != _scientific_configuration_record(run):
        _fail("Subject-shape scientific configuration differs from live attrs.")
    return result


def _require_retained_inventory_entry(
    value: Any,
    *,
    expected_names: Sequence[str],
    label: str,
) -> None:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"names", "content_sha256", "canonicalization"}
        or value.get("names") != sorted(expected_names)
        or not isinstance(value.get("content_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", value["content_sha256"]) is None
        or value.get("canonicalization")
        != "finite_canonical_json_sorted_attribute_mapping_v1"
    ):
        _fail(f"Retained unbound-stage attr inventory for {label!r} is invalid.")


def _validate_retained_unbound_schema(run: Any, record: Mapping[str, Any]) -> None:
    """Validate the consumed receipt against the explicit unbound vocabulary."""

    inventory = record.get("schema_inventory")
    arrays = record.get("arrays")
    expected_run_ref = f"/{canonical_node_path(run)}"
    expected_groups = _expected_subject_shape_group_paths(run, phase="unbound")
    expected_arrays = _expected_subject_shape_array_paths(run, phase="unbound")
    expected_profile = (
        subject_shape_bundle_maintained_profile_record()
        if run.attrs.get(SUBJECT_SHAPE_SOURCE_KIND_ATTR)
        == SUBJECT_SHAPE_BUNDLE_SOURCE_KIND
        else subject_shape_maintained_profile_record()
    )
    if (
        not isinstance(inventory, Mapping)
        or inventory.get("schema_id") != SUBJECT_SHAPE_SCHEMA_INVENTORY_SCHEMA_ID
        or inventory.get("schema_version") != SUBJECT_SHAPE_SCHEMA_VERSION
        or inventory.get("maintained_profile") != expected_profile
        or inventory.get("phase") != "unbound"
        or inventory.get("run_ref") != expected_run_ref
        or inventory.get("groups") != list(expected_groups)
        or not isinstance(inventory.get("arrays"), Mapping)
        or set(inventory["arrays"]) != set(expected_arrays)
        or not isinstance(inventory.get("attrs"), Mapping)
        or inventory.get("excluded_operational_run_attrs")
        != sorted(_RUN_OPERATIONAL_ATTRS)
        or inventory.get("excluded_self_referential_run_attrs")
        != sorted(
            {
                SUBJECT_SHAPE_UNBOUND_MANIFEST_ATTR,
                f"{SUBJECT_SHAPE_UNBOUND_MANIFEST_ATTR}_sha256",
            }
        )
        or inventory.get("closed_group_inventory") is not True
        or inventory.get("closed_array_inventory") is not True
        or inventory.get("closed_attr_inventory") is not True
        or not isinstance(arrays, Mapping)
        or set(arrays) != set(expected_arrays)
    ):
        _fail("Retained subject-shape unbound-stage schema inventory is invalid.")

    expected_group_attrs = _expected_subject_shape_group_attrs(
        run,
        phase="unbound",
    )
    expected_array_attrs = _expected_subject_shape_array_attrs(
        run,
        expected_arrays,
        phase="unbound",
    )
    inventory_attrs = inventory["attrs"]
    if set(inventory_attrs) != {".", *expected_groups, *expected_arrays}:
        _fail("Retained subject-shape unbound-stage attr paths are invalid.")
    for path in expected_groups:
        _require_retained_inventory_entry(
            inventory_attrs[path],
            expected_names=expected_group_attrs[path],
            label=path,
        )
    for path in expected_arrays:
        expected_role = _subject_shape_array_role(run, path, phase="unbound")
        if inventory["arrays"].get(path) != {"role": expected_role}:
            _fail(f"Retained unbound-stage array role for {path!r} is invalid.")
        _require_retained_inventory_entry(
            inventory_attrs[path],
            expected_names=expected_array_attrs[path],
            label=path,
        )
        entry = arrays[path]
        live = run[path]
        if (
            not isinstance(entry, Mapping)
            or set(entry)
            != {
                "relative_ref",
                "dtype",
                "shape",
                "content_sha256",
                "canonicalization",
            }
            or entry.get("relative_ref") != path
            or entry.get("dtype") != np.dtype(live.dtype).str
            or entry.get("shape") != [int(value) for value in live.shape]
            or not isinstance(entry.get("content_sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", entry["content_sha256"]) is None
            or entry.get("canonicalization")
            != "numpy_dtype_shape_c_order_bytes_v1"
        ):
            _fail(f"Retained unbound-stage array record for {path!r} is invalid.")

    stable_run_names = sorted(
        (
            {str(value) for value in run.attrs.keys()}
            & set(_RUN_AUTHORITATIVE_ATTRS)
        )
        - {SUBJECT_SHAPE_PAYLOAD_RECEIPT_PROFILE_ATTR}
    )
    expected_root_entry = _attribute_inventory_entry(run, stable_run_names)
    if inventory_attrs["."] != expected_root_entry:
        _fail(
            "Retained unbound-stage authoritative run attrs differ from the bound run."
        )


def load_subject_shape_consumed_unbound_stage(run: Any) -> BoundCoordinateRecord:
    node = run.get("coordinate_records/consumed_unbound_stage")
    if node is None:
        _fail("Subject-shape publication lacks its retained unbound-stage record.")
    result = bind_persisted_coordinate_record(
        node,
        attr_name=SUBJECT_SHAPE_CONSUMED_UNBOUND_STAGE_ATTR,
    )
    record = result.record
    scientific = record.get("scientific_configuration")
    schema_inventory = record.get("schema_inventory")
    projected = (
        record.get("schema_version")
        == SUBJECT_SHAPE_PROJECTED_UNBOUND_MANIFEST_SCHEMA_VERSION
        and record.get("binding_status")
        == SUBJECT_SHAPE_PROJECTED_UNBOUND_BINDING_STATUS
    )
    legacy = (
        record.get("schema_version") == 1
        and record.get("binding_status") == "unbound_roi_local_numeric_payload"
    )
    if (
        record.get("schema_id") != SUBJECT_SHAPE_UNBOUND_MANIFEST_SCHEMA_ID
        or not (legacy or projected)
        or record.get("run_name") != run.attrs.get("palette_run_name")
        or record.get("source_refined_subject_masks_run")
        != run.attrs.get("source_refined_subject_masks_run")
        or record.get("method") != run.attrs.get("method")
        or record.get("method_version") != run.attrs.get("method_version")
        or scientific != _scientific_configuration_record(run)
        or not isinstance(schema_inventory, Mapping)
        or record.get("component_names")
        != list(run.attrs.get("component_names") or ())
        or record.get("closed_group_inventory") is not True
        or record.get("closed_array_inventory") is not True
        or record.get("closed_attr_inventory") is not True
        or record.get("coordinate_descriptors_present") is not False
    ):
        _fail(
            "Retained subject-shape unbound-stage record differs from the exact "
            "bound run identity or scientific configuration."
        )
    projection_present = (
        SUBJECT_SHAPE_NUMERIC_PROJECTION_ATTR in run.attrs
        and SUBJECT_SHAPE_NUMERIC_PROJECTION_DIGEST_ATTR in run.attrs
    )
    if projected is not projection_present:
        _fail(
            "Retained subject-shape unbound-stage projection profile differs "
            "from its bound run evidence."
        )
    _validate_retained_unbound_schema(run, record)
    consumed_digest = run.attrs.get(
        "unbound_numeric_stage_manifest_sha256_consumed"
    )
    if consumed_digest != result.record_sha256:
        _fail("Retained subject-shape unbound-stage digest is inconsistent.")
    return result


def _tail_sample_axis_record(run: Any) -> dict[str, Any]:
    body = run.get("components/subject_body")
    if body is None or "tail_sample_s" not in body:
        _fail("Canonical subject-shape publication lacks tail_sample_s.")
    samples = body["tail_sample_s"]
    values = np.asarray(samples[:])
    if (
        values.dtype.kind != "f"
        or values.ndim != 1
        or values.size < 2
        or not np.isfinite(values).all()
        or float(values[0]) != 0.0
        or float(values[-1]) != 1.0
        or not np.all(np.diff(values.astype(np.float64)) > 0.0)
    ):
        _fail("tail_sample_s must be a finite, strictly increasing closed [0,1] axis.")
    for name in ("tail_sample_xy", "tail_tangent_xy", "tail_normal_xy"):
        node = body.get(name)
        if node is None or tuple(int(value) for value in node.shape[1:]) != (
            int(values.size),
            2,
        ):
            _fail(f"{name} does not use the exact tail-sample cardinality.")
    return {
        "schema_id": SUBJECT_SHAPE_TAIL_SAMPLE_AXIS_SCHEMA_ID,
        "schema_version": SUBJECT_SHAPE_SCHEMA_VERSION,
        "axis_index": 1,
        "axis_role": "tail_sample",
        "cardinality": int(values.size),
        "coordinate_kind": "normalized_arclength",
        "coordinate_units": "unitless",
        "domain": "closed_0_to_1",
        "sample_direction": "tail_base_to_tail_tip",
        "sample_coordinate": _array_record(
            "components/subject_body/tail_sample_s",
            samples,
        ),
        "bound_surfaces": [
            "components/subject_body/tail_sample_xy",
            "components/subject_body/tail_tangent_xy",
            "components/subject_body/tail_normal_xy",
        ],
    }


def _stamp_tail_sample_axis(run: Any) -> BoundCoordinateRecord:
    node = run["components/subject_body/tail_sample_s"]
    return stamp_and_bind_persisted_coordinate_record(
        node,
        _tail_sample_axis_record(run),
        attr_name=SUBJECT_SHAPE_TAIL_SAMPLE_AXIS_ATTR,
    )


def _load_tail_sample_axis(run: Any) -> BoundCoordinateRecord:
    node = run["components/subject_body/tail_sample_s"]
    result = bind_persisted_coordinate_record(
        node,
        attr_name=SUBJECT_SHAPE_TAIL_SAMPLE_AXIS_ATTR,
    )
    if result.record != _tail_sample_axis_record(run):
        _fail("Subject-shape tail-sample axis differs from its exact live array.")
    return result


def prepare_subject_shape_identity_and_schema(
    run: Any,
    source: SubjectShapeCoordinateSource,
    *,
    component_names: Sequence[str],
) -> tuple[BoundRowIdentityContract, BoundCoordinateRecord]:
    """Copy exact identity to direct rowset children and seal it."""

    source_keys = np.asarray(_source_row_node(source, "instance_key")[:])
    if source_keys.dtype != np.dtype("uint64"):
        _fail("Future subject-shape identity requires exact uint64 instance_key.")
    key = _create_array(run, "instance_key", source_keys)
    for name in ("source_crop_row_ids", "source_acquisition_frame_index"):
        source_node = _source_row_node(source, name)
        _create_array(run, name, np.asarray(source_node[:]))
    identity = stamp_and_bind_row_identity_contract(
        run,
        key,
        contract=build_row_identity_contract(
            domain=OBSERVATION_INSTANCE_DOMAIN,
            values=source_keys,
        ),
    )
    records = run.require_group("coordinate_records")
    schema_node = records.require_group("component_schema")
    schema = stamp_and_bind_persisted_coordinate_record(
        schema_node,
        _component_schema_record(component_names),
        attr_name=SUBJECT_SHAPE_COMPONENT_SCHEMA_ATTR,
    )
    return identity, schema


def stamp_subject_shape_temporal_authority(
    run: Any,
    source: SubjectShapeCoordinateSource,
    identity: BoundRowIdentityContract,
) -> BoundSourceRowTemporalAuthority:
    return stamp_source_row_temporal_authority(
        run,
        run["source_acquisition_frame_index"],
        source_row_identity=identity,
        acquisition_frame=_source_acquisition_frame(source),
    )


def load_subject_shape_temporal_authority(
    run: Any,
    source: SubjectShapeCoordinateSource,
    identity: BoundRowIdentityContract,
) -> BoundSourceRowTemporalAuthority:
    result = load_bound_source_row_temporal_authority(
        run,
        run["source_acquisition_frame_index"],
        source_row_identity=identity,
        acquisition_frame=_source_acquisition_frame(source),
    )
    if _is_bundle_source(source):
        bundle = require_bound_subject_shape_bundle_source(source)
        expected = (
            bundle.authority.recording_identity,
            bundle.authority.camera_identity,
            bundle.authority.source_total_frames,
        )
    else:
        expected = (
            source.context.temporal_authority.record.recording_id,
            source.context.temporal_authority.record.camera_id,
            source.context.temporal_authority.record.source_total_frames,
        )
    observed = (
        result.record.recording_id,
        result.record.camera_id,
        result.record.source_total_frames,
    )
    if observed != expected:
        _fail("Subject-shape temporal authority differs from its exact refined source.")
    return result


def _transform_refs(
    source: SubjectShapeCoordinateSource,
    source_binding: BoundCoordinateRecord | None,
) -> dict[str, Any]:
    if _is_bundle_source(source):
        if source_binding is None:
            _fail("Bundle-bound subject shape lacks its exact source binding record.")
        bundle = require_bound_subject_shape_bundle_source(source)
        return {
            "recording_bundle_roi_to_source_camera": {
                "source_binding": _record_pointer(source_binding),
                "placement_array": copy.deepcopy(
                    bundle.source_record["row_arrays"]["source_crop_xywh"]
                ),
                "policy": copy.deepcopy(
                    bundle.source_record["coordinate_transform"]
                ),
            }
        }
    return {
        "continuous_roi_to_source_camera": [
            {
                "record_ref": item.record_ref,
                "record_sha256": item.record_sha256,
                "payload_sha256": item.payload_sha256,
                "authority_ref": item.authority_ref,
                "authority_sha256": item.authority_sha256,
            }
            for item in source.context.continuous_chain.transform_records
        ],
        "pixel_edge_half_open_roi_to_source_camera": [
            {
                "record_ref": item.record_ref,
                "record_sha256": item.record_sha256,
                "payload_sha256": item.payload_sha256,
                "authority_ref": item.authority_ref,
                "authority_sha256": item.authority_sha256,
            }
            for item in source.context.pixel_edge_chain.transform_records
        ],
    }


def _derivation_record(
    run: Any,
    source: SubjectShapeCoordinateSource,
    source_binding: BoundCoordinateRecord | None,
    identity: BoundRowIdentityContract,
    component_schema: BoundCoordinateRecord,
    temporal: BoundSourceRowTemporalAuthority,
    scientific_configuration: BoundCoordinateRecord,
    tail_sample_axis: BoundCoordinateRecord,
    scalar_surface_inventory: BoundCoordinateRecord,
) -> dict[str, Any]:
    unbound_manifest_sha256 = run.attrs.get(
        "unbound_numeric_stage_manifest_sha256_consumed"
    )
    if (
        not isinstance(unbound_manifest_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", unbound_manifest_sha256) is None
    ):
        _fail("Subject-shape derivation lacks its consumed unbound-stage digest.")
    consumed_unbound_stage = load_subject_shape_consumed_unbound_stage(run)
    if consumed_unbound_stage.record_sha256 != unbound_manifest_sha256:
        _fail("Subject-shape derivation names the wrong consumed unbound-stage record.")
    if _is_bundle_source(source):
        if source_binding is None:
            _fail("Bundle-bound derivation lacks its persisted source binding.")
        bundle = require_bound_subject_shape_bundle_source(source)
        source_section = {
            "source_kind": SUBJECT_SHAPE_BUNDLE_SOURCE_KIND,
            "bundle_id": bundle.bundle_id,
            "source_binding": _record_pointer(source_binding),
            "bundle_manifest_payload_digest": bundle.authority.bundle_manifest[
                "payload_digest"
            ],
            "bundle_coordinate_authority_digest": bundle.authority.authority_digest,
            "refined_run_path": bundle.authority.refined_run_path,
            "refined_manifest_payload_digest": bundle.authority.refined_manifest[
                "payload_digest"
            ],
        }
        schema_version = SUBJECT_SHAPE_BUNDLE_DERIVATION_SCHEMA_VERSION
    else:
        source_section = {
            "run_path": source.context.run_path,
            "context": _record_pointer(source.context.context_record),
            "surface_inventory": _record_pointer(source.inventory),
            "component_qc_inventory": _record_pointer(
                source.component_qc_inventory
            ),
            "source_authority": _record_pointer(source.context.source_authority),
            "refinement_authority": _record_pointer(
                source.context.refinement_authority
            ),
            "row_identity": {
                "record_ref": source.context.row_identity.record_ref,
                "record_sha256": source.context.row_identity.record_sha256,
            },
        }
        schema_version = SUBJECT_SHAPE_SCHEMA_VERSION
    unbound_record = consumed_unbound_stage.record
    projected = (
        unbound_record.get("schema_version")
        == SUBJECT_SHAPE_PROJECTED_UNBOUND_MANIFEST_SCHEMA_VERSION
        and unbound_record.get("binding_status")
        == SUBJECT_SHAPE_PROJECTED_UNBOUND_BINDING_STATUS
    )
    unbound_section: dict[str, Any] = {
        "schema_id": SUBJECT_SHAPE_UNBOUND_MANIFEST_SCHEMA_ID,
        "manifest_sha256": unbound_manifest_sha256,
        "record_ref": consumed_unbound_stage.record_ref,
        "record_sha256": consumed_unbound_stage.record_sha256,
        "coordinate_status": (
            "source_camera_numeric_precanonical"
            if projected
            else "roi_local_numeric_unbound"
        ),
        "consumption_policy": (
            "validate_projection_receipt_then_stamp_final_authority_v1"
            if projected
            else "validate_then_final_path_bind_and_transform_v1"
        ),
    }
    if projected:
        unbound_section["numeric_projection_sha256"] = run.attrs.get(
            SUBJECT_SHAPE_NUMERIC_PROJECTION_DIGEST_ATTR
        )
    return {
        "schema_id": SUBJECT_SHAPE_DERIVATION_SCHEMA_ID,
        "schema_version": schema_version,
        "run_ref": f"/{canonical_node_path(run)}",
        "method": run.attrs.get("method"),
        "method_version": run.attrs.get("method_version"),
        "unbound_numeric_stage": unbound_section,
        "source_refined_subject_masks": source_section,
        "output_row_identity": {
            "record_ref": identity.record_ref,
            "record_sha256": identity.record_sha256,
        },
        "output_temporal_authority": _temporal_pointer(temporal),
        "component_schema": _record_pointer(component_schema),
        "scientific_configuration": _record_pointer(scientific_configuration),
        "tail_sample_axis": _record_pointer(tail_sample_axis),
        "scalar_surface_inventory": _record_pointer(scalar_surface_inventory),
        "transform_direction": "roi_local_px_to_source_camera_image_px",
        "transform_policy": "exact_translation_only_v1",
        "transforms": _transform_refs(source, source_binding),
        "bbox_derivation": "foreground_half_open_pixel_edges_then_exact_placement_v1",
        "roi_local_point_arrays_retained": False,
    }


def stamp_subject_shape_derivation(
    run: Any,
    source: SubjectShapeCoordinateSource,
    source_binding: BoundCoordinateRecord | None,
    identity: BoundRowIdentityContract,
    component_schema: BoundCoordinateRecord,
    temporal: BoundSourceRowTemporalAuthority,
    scientific_configuration: BoundCoordinateRecord,
    tail_sample_axis: BoundCoordinateRecord,
    scalar_surface_inventory: BoundCoordinateRecord,
) -> BoundCoordinateRecord:
    return stamp_and_bind_persisted_coordinate_record(
        run,
        _derivation_record(
            run,
            source,
            source_binding,
            identity,
            component_schema,
            temporal,
            scientific_configuration,
            tail_sample_axis,
            scalar_surface_inventory,
        ),
        attr_name=SUBJECT_SHAPE_DERIVATION_ATTR,
    )


def _lineage_records(
    source: SubjectShapeCoordinateSource,
    source_binding: BoundCoordinateRecord | None,
    component_schema: BoundCoordinateRecord,
    scientific_configuration: BoundCoordinateRecord,
    derivation: BoundCoordinateRecord,
    tail_sample_axis: BoundCoordinateRecord | None = None,
) -> tuple[BoundCoordinateRecord, ...]:
    # Put the output collection-label authority first.  The canonical
    # descriptor builder inserts that same authority immediately after its
    # reference-frame authority; matching this order keeps collected and
    # non-collected array descriptors deterministic under the strict rebinder.
    if _is_bundle_source(source):
        if source_binding is None:
            _fail("Bundle-bound descriptor lineage lacks its source binding.")
        prefix = (component_schema, source_binding, scientific_configuration)
    else:
        prefix = (
            component_schema,
            source.context.component_labels,
            source.context.source_authority,
            source.context.refinement_authority,
            source.context.context_record,
            source.inventory,
            source.component_qc_inventory,
            scientific_configuration,
        )
    return (
        *prefix,
        *((tail_sample_axis,) if tail_sample_axis is not None else ()),
        derivation,
    )


def _geometry_specs(run: Any, component_names: Sequence[str]) -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {
        "component_centroid_xy": {
            # One labeled point per subject component.  The per-component
            # sibling arrays below are also `point_xy`, but this aggregate has
            # an explicit subject-component collection axis.
            "geometry_type": "point_xy",
            "components": ("x", "y"),
            "units": ("px", "px"),
            "pixel_convention": "continuous",
            "collection": True,
        },
        "body_frame/origin_xy": {
            "geometry_type": "point_xy",
            "components": ("x", "y"),
            "units": ("px", "px"),
            "pixel_convention": "continuous",
        },
        "body_frame/forward_axis_xy": {
            "profile_id": "source_camera_image_px.unit_vector_y_down.v1",
            "geometry_type": "vector_xy",
            "components": ("x", "y"),
            "units": ("unitless", "unitless"),
            "pixel_convention": "not_applicable",
            "overlay": CANONICAL_OVERLAY_NOT_SUITABLE,
        },
        "body_frame/left_axis_xy": {
            "profile_id": "source_camera_image_px.unit_vector_y_down.v1",
            "geometry_type": "vector_xy",
            "components": ("x", "y"),
            "units": ("unitless", "unitless"),
            "pixel_convention": "not_applicable",
            "overlay": CANONICAL_OVERLAY_NOT_SUITABLE,
        },
    }
    for component in component_names:
        prefix = f"components/{component}"
        specs[f"{prefix}/centroid_xy"] = {
            "geometry_type": "point_xy",
            "components": ("x", "y"),
            "units": ("px", "px"),
            "pixel_convention": "continuous",
        }
        specs[f"{prefix}/bbox_xyxy"] = {
            "geometry_type": "bbox_xyxy",
            "components": ("x_min", "y_min", "x_max", "y_max"),
            "units": ("px", "px", "px", "px"),
            "pixel_convention": "pixel_edge_half_open",
        }
        group = run[prefix]
        if "ellipse_params" in group:
            specs[f"{prefix}/ellipse_params"] = {
                "geometry_type": "ellipse_cxcy_wh_angle",
                "components": ("center_x", "center_y", "width", "height", "angle"),
                "units": ("px", "px", "px", "px", "deg"),
                "pixel_convention": "continuous",
            }
    body = run.get("components/subject_body")
    if body is not None:
        if "principal_axis_xy" in body:
            specs["components/subject_body/principal_axis_xy"] = {
                "profile_id": "source_camera_image_px.unit_vector_y_down.v1",
                "geometry_type": "vector_xy",
                "components": ("x", "y"),
                "units": ("unitless", "unitless"),
                "pixel_convention": "not_applicable",
                "overlay": CANONICAL_OVERLAY_NOT_SUITABLE,
            }
        for name in ("tail_tangent_xy", "tail_normal_xy"):
            if name in body:
                specs[f"components/subject_body/{name}"] = {
                    "profile_id": "source_camera_image_px.unit_vector_y_down.v1",
                    "geometry_type": "vector_sequence_xy",
                    "components": ("x", "y"),
                    "units": ("unitless", "unitless"),
                    "pixel_convention": "not_applicable",
                    "overlay": CANONICAL_OVERLAY_NOT_SUITABLE,
                    "tail_sample_axis": True,
                }
        for name in (
            "centerline_xy",
            "bspline_control_points_xy",
            "bspline_sample_xy",
            "tail_sample_xy",
        ):
            if name in body:
                specs[f"components/subject_body/{name}"] = {
                    "geometry_type": "polyline_xy",
                    "components": ("x", "y"),
                    "units": ("px", "px"),
                    "pixel_convention": "continuous",
                    "tail_sample_axis": name == "tail_sample_xy",
                }
        for name in ("snout_tip_xy", "head_endpoint_xy", "tail_tip_xy", "tail_base_xy"):
            if name in body:
                specs[f"components/subject_body/{name}"] = {
                    "geometry_type": "point_xy",
                    "components": ("x", "y"),
                    "units": ("px", "px"),
                    "pixel_convention": "continuous",
                }
    swim = run.get("components/swim_bladder")
    if swim is not None and "caudal_contour_point_xy" in swim:
        specs["components/swim_bladder/caudal_contour_point_xy"] = {
            "geometry_type": "point_xy",
            "components": ("x", "y"),
            "units": ("px", "px"),
            "pixel_convention": "continuous",
        }
    if run.get("relations/eye_pair") is not None:
        specs["relations/eye_pair/midpoint_xy"] = {
            "geometry_type": "point_xy",
            "components": ("x", "y"),
            "units": ("px", "px"),
            "pixel_convention": "continuous",
        }
    if run.get("relations/eyes_to_body") is not None:
        for prefix in ("left", "right"):
            specs[f"relations/eyes_to_body/{prefix}_eye_offset_xy"] = {
                "profile_id": "source_camera_image_px.displacement_vector_y_down.v1",
                "geometry_type": "vector_xy",
                "components": ("x", "y"),
                "units": ("px", "px"),
                "pixel_convention": "not_applicable",
                "overlay": CANONICAL_OVERLAY_NOT_SUITABLE,
            }
    return specs


def _descriptor_bindings(
    run: Any,
    *,
    source: SubjectShapeCoordinateSource,
    source_binding: BoundCoordinateRecord | None,
    identity: BoundRowIdentityContract,
    component_schema: BoundCoordinateRecord,
    scientific_configuration: BoundCoordinateRecord,
    tail_sample_axis: BoundCoordinateRecord,
    derivation: BoundCoordinateRecord,
    component_names: Sequence[str],
    load: bool,
) -> dict[str, BoundCanonicalCoordinateDescriptor]:
    specs = _geometry_specs(run, component_names)
    collection = CanonicalCollectionAxis(
        axis=1,
        role="subject_component",
        cardinality=len(component_names),
        label_authority=DigestBoundCoordinateRecordRef(
            record_ref=component_schema.record_ref,
            record_sha256=component_schema.record_sha256,
        ),
    )
    result: dict[str, BoundCanonicalCoordinateDescriptor] = {}
    for path, spec in sorted(specs.items()):
        node = run[path]
        lineage = _lineage_records(
            source,
            source_binding,
            component_schema,
            scientific_configuration,
            derivation,
            tail_sample_axis=(
                tail_sample_axis if spec.get("tail_sample_axis") else None
            ),
        )
        continuous_frame, edge_frame = _source_camera_frames(source)
        frame = edge_frame if spec["pixel_convention"] == "pixel_edge_half_open" else continuous_frame
        kwargs = {
            "row_identity": identity,
            "reference_frame_authority": frame,
            "lineage_records": lineage,
        }
        if load:
            binding = load_bound_canonical_coordinate_descriptor(node, **kwargs)
        else:
            binding = build_bound_canonical_coordinate_descriptor(
                node,
                profile_id=spec.get(
                    "profile_id",
                    "source_camera_image_px.top_left_y_down.v1",
                ),
                geometry_type=spec["geometry_type"],
                components=spec["components"],
                component_units=spec["units"],
                pixel_convention=spec["pixel_convention"],
                row_identity=identity,
                reference_frame_authority=frame,
                source_camera_overlay_status=spec.get(
                    "overlay",
                    CANONICAL_OVERLAY_DIRECT,
                ),
                lineage_records=lineage,
                collection_axis=collection if spec.get("collection") else None,
            )
        result[path] = binding
    return result


def _write_component_aggregate(
    run: Any,
    component_names: Sequence[str],
) -> tuple[Any, Any]:
    centroids = np.stack(
        [np.asarray(run[f"components/{name}/centroid_xy"][:], dtype=np.float32) for name in component_names],
        axis=1,
    )
    valid = np.stack(
        [np.asarray(run[f"components/{name}/centroid_valid"][:], dtype=bool) for name in component_names],
        axis=1,
    )
    centroids[~valid] = np.nan
    return (
        _create_array(run, "component_centroid_xy", centroids),
        _create_array(run, "component_centroid_valid", valid),
    )


def derive_canonical_subject_shape_body_frame(
    component_names: Sequence[str],
    component_centroid_xy: Any,
    component_centroid_valid: Any,
) -> dict[str, np.ndarray]:
    """Derive the maintained camera-space anatomical frame in memory."""

    labels = tuple(str(value) for value in component_names)
    required = ("eye_left", "eye_right", "swim_bladder")
    if any(name not in labels for name in required):
        _fail("Canonical subject-shape publication requires both eyes and swim_bladder.")
    anchors = np.asarray(component_centroid_xy, dtype=np.float32)
    validity = np.asarray(component_centroid_valid, dtype=bool)
    if (
        anchors.ndim != 3
        or anchors.shape[1:] != (len(labels), 2)
        or validity.shape != anchors.shape[:2]
    ):
        _fail("Canonical subject-shape body-frame inputs are not exact.")
    row_count = anchors.shape[0]
    origin = np.full((row_count, 2), np.nan, dtype=np.float32)
    forward = np.full((row_count, 2), np.nan, dtype=np.float32)
    left = np.full((row_count, 2), np.nan, dtype=np.float32)
    valid = np.zeros((row_count,), dtype=bool)
    reasons = ["missing_source_anchor"] * row_count
    li, ri, si = (labels.index(name) for name in required)
    for row in range(row_count):
        if not bool(validity[row, li] and validity[row, ri] and validity[row, si]):
            continue
        eye_left, eye_right, posterior = np.asarray(
            anchors[row, (li, ri, si), :], dtype=np.float64
        )
        if not np.all(np.isfinite((eye_left, eye_right, posterior))):
            continue
        row_origin = (eye_left + eye_right) / 2.0
        direction = row_origin - posterior
        norm = float(np.linalg.norm(direction))
        if not math.isfinite(norm) or norm <= 0.0:
            reasons[row] = "degenerate_forward_axis"
            continue
        row_forward = direction / norm
        row_left = np.asarray([row_forward[1], -row_forward[0]], dtype=np.float64)
        if float(np.dot(eye_left - eye_right, row_left)) <= 0.0:
            reasons[row] = "left_right_unresolved"
            continue
        origin[row] = row_origin.astype(np.float32)
        forward[row] = row_forward.astype(np.float32)
        left[row] = row_left.astype(np.float32)
        valid[row] = True
        reasons[row] = "ok"
    heading = np.asarray(
        [
            math.degrees(math.atan2(-float(value[1]), float(value[0])))
            if is_valid
            else np.nan
            for value, is_valid in zip(forward, valid, strict=True)
        ],
        dtype=np.float32,
    )
    return {
        "origin_xy": origin,
        "forward_axis_xy": forward,
        "left_axis_xy": left,
        "valid": valid,
        "heading_deg": heading,
        "failure_reason_bytes": np.asarray(
            [
                np.pad(
                    np.frombuffer(reason.encode("utf-8")[:63], dtype=np.uint8),
                    (0, 64 - min(len(reason.encode("utf-8")), 63)),
                )
                for reason in reasons
            ],
            dtype=np.uint8,
        ),
    }


def _rewrite_body_frame_from_camera_components(
    run: Any,
    component_names: Sequence[str],
) -> None:
    derived = derive_canonical_subject_shape_body_frame(
        component_names,
        np.asarray(run["component_centroid_xy"][:], dtype=np.float32),
        np.asarray(run["component_centroid_valid"][:], dtype=bool),
    )
    body = run["body_frame"]
    body["origin_xy"][:] = derived["origin_xy"]
    body["forward_axis_xy"][:] = derived["forward_axis_xy"]
    body["left_axis_xy"][:] = derived["left_axis_xy"]
    body["valid"][:] = derived["valid"]
    body["heading_deg"][:] = derived["heading_deg"]
    if "axis_valid" in body:
        _fail("Immutable body_frame/axis_valid already exists.")
    axis_valid = _create_array(body, "axis_valid", derived["valid"])
    body["failure_reason_bytes"][:] = derived["failure_reason_bytes"]
    body.attrs["body_frame_coordinate_space"] = "source_camera_image_px"
    body.attrs["axis_valid_array"] = "axis_valid"
    # Keep the historical `valid` sibling solely as an explicit compatibility
    # alias; the sealed body record authorizes axis_valid.
    body.attrs["valid_compatibility_alias_of"] = "axis_valid"
    if not np.array_equal(np.asarray(axis_valid[:], dtype=bool), np.asarray(body["valid"][:], dtype=bool)):
        _fail("Body-frame validity compatibility alias differs from axis_valid.")


def _finalize_preprojected_body_frame(run: Any) -> None:
    """Add the canonical validity alias without rewriting sealed geometry."""

    body = run["body_frame"]
    if "axis_valid" in body:
        _fail("Immutable body_frame/axis_valid already exists.")
    valid = np.asarray(body["valid"][:], dtype=bool)
    axis_valid = _create_array(body, "axis_valid", valid)
    body.attrs["body_frame_coordinate_space"] = "source_camera_image_px"
    body.attrs["axis_valid_array"] = "axis_valid"
    body.attrs["valid_compatibility_alias_of"] = "axis_valid"
    if not np.array_equal(
        np.asarray(axis_valid[:], dtype=bool),
        valid,
    ):
        _fail("Body-frame validity compatibility alias differs from axis_valid.")


def _stamp_body_frame(
    run: Any,
    *,
    source: SubjectShapeCoordinateSource,
    source_binding: BoundCoordinateRecord | None,
    identity: BoundRowIdentityContract,
    component_schema: BoundCoordinateRecord,
    scientific_configuration: BoundCoordinateRecord,
    tail_sample_axis: BoundCoordinateRecord,
    derivation: BoundCoordinateRecord,
) -> BoundFishAnatomicalBodyFrame:
    records = run["coordinate_records"]
    contract_node = records.require_group("body_frame_contract")
    estimator_node = records.require_group("body_frame_estimator")
    contract = stamp_body_frame_contract(
        contract_node,
        record=build_body_frame_contract_record(),
    )
    estimator = stamp_body_frame_estimator(
        estimator_node,
        record=build_body_frame_estimator_record(
            method="mask_component_axis",
            implementation_version=BODY_FRAME_IMPLEMENTATION_VERSION,
            configuration_schema_id="palette.mask_component_axis_parameters",
            configuration={
                "eye_left": "eye_left",
                "eye_right": "eye_right",
                "posterior_anchor": "swim_bladder",
            },
        ),
    )
    lineage = _lineage_records(
        source,
        source_binding,
        component_schema,
        scientific_configuration,
        derivation,
    )
    continuous_frame, _edge_frame = _source_camera_frames(source)
    source_descriptor = bind_body_source_coordinate_descriptor(
        run["component_centroid_xy"],
        row_identity=identity,
        source_camera_pixels=continuous_frame,
        lineage_records=lineage,
    )
    source_manifest = stamp_and_bind_persisted_coordinate_record(
        run,
        build_body_estimator_source_manifest_record(
            method="mask_component_axis",
            source_descriptor=source_descriptor,
            estimator=estimator,
            source_schema=component_schema,
            support_nodes={"validity": run["component_centroid_valid"]},
        ),
        attr_name=BODY_ESTIMATOR_SOURCE_MANIFEST_ATTR,
    )
    estimator_source = bind_mask_component_axis_source(
        source_descriptor=source_descriptor,
        estimator=estimator,
        component_schema=component_schema,
        validity_node=run["component_centroid_valid"],
        producer_manifest=source_manifest,
    )
    frame = run["body_frame"]
    geometry = bind_body_frame_geometry(
        frame,
        origin_xy_node=frame["origin_xy"],
        forward_axis_xy_node=frame["forward_axis_xy"],
        left_axis_xy_node=frame["left_axis_xy"],
        axis_valid_node=frame["axis_valid"],
        row_identity=identity,
        estimator_source=estimator_source,
    )
    record = build_fish_anatomical_body_frame_record(
        frame_id="subject_shape_mask_component_axis_v1",
        origin_definition="eye_pair_midpoint",
        body_frame_contract=contract,
        estimator_source=estimator_source,
        geometry=geometry,
        row_identity=identity,
    )
    return stamp_fish_anatomical_body_frame_record(
        frame,
        record,
        expected_record_ref=f"/{canonical_node_path(frame)}@{FISH_ANATOMICAL_BODY_FRAME_ATTR}",
        body_frame_contract=contract,
        estimator_source=estimator_source,
        geometry=geometry,
        row_identity=identity,
    )


def _heading_semantics_record(
    run: Any,
    *,
    identity: BoundRowIdentityContract,
    forward_descriptor: BoundCanonicalCoordinateDescriptor,
    body_frame: BoundFishAnatomicalBodyFrame,
) -> dict[str, Any]:
    frame = run["body_frame"]
    heading = np.asarray(frame["heading_deg"][:], dtype=np.float64)
    forward = np.asarray(frame["forward_axis_xy"][:], dtype=np.float64)
    valid = np.asarray(frame["axis_valid"][:], dtype=bool)
    if heading.shape != (identity.leading_dimension,) or forward.shape != (
        identity.leading_dimension,
        2,
    ) or valid.shape != (identity.leading_dimension,):
        _fail("Subject-shape heading arrays do not share the exact row identity.")
    expected = np.full(heading.shape, np.nan, dtype=np.float64)
    expected[valid] = np.degrees(
        np.arctan2(-forward[valid, 1], forward[valid, 0])
    )
    if not np.allclose(heading, expected, rtol=0.0, atol=1e-5, equal_nan=True):
        _fail("body_frame/heading_deg differs from its declared forward-axis formula.")
    return {
        "schema_id": SUBJECT_SHAPE_HEADING_SEMANTICS_SCHEMA_ID,
        "schema_version": SUBJECT_SHAPE_SCHEMA_VERSION,
        "heading": _array_record("body_frame/heading_deg", frame["heading_deg"]),
        "forward_axis": {
            "array": _array_record(
                "body_frame/forward_axis_xy",
                frame["forward_axis_xy"],
            ),
            "descriptor_ref": (
                f"/{canonical_node_path(forward_descriptor.coordinate_node)}"
                f"@{COORDINATE_DESCRIPTOR_ATTR}"
            ),
            "descriptor_sha256": forward_descriptor.descriptor.digest(),
        },
        "axis_valid": _array_record("body_frame/axis_valid", frame["axis_valid"]),
        "row_identity": {
            "record_ref": identity.record_ref,
            "record_sha256": identity.record_sha256,
        },
        "body_frame": {
            "record_ref": body_frame.record_ref,
            "record_sha256": body_frame.record_sha256,
        },
        "units": "deg",
        "formula": "degrees(atan2(-forward_y, forward_x))",
        "zero_direction": "source_camera_positive_x_right",
        "positive_rotation": "counterclockwise_after_source_camera_y_flip",
        "invalid_row_value": "nan_when_axis_valid_false",
    }


def _stamp_heading_semantics(
    run: Any,
    *,
    identity: BoundRowIdentityContract,
    forward_descriptor: BoundCanonicalCoordinateDescriptor,
    body_frame: BoundFishAnatomicalBodyFrame,
) -> BoundCoordinateRecord:
    return stamp_and_bind_persisted_coordinate_record(
        run["body_frame/heading_deg"],
        _heading_semantics_record(
            run,
            identity=identity,
            forward_descriptor=forward_descriptor,
            body_frame=body_frame,
        ),
        attr_name=SUBJECT_SHAPE_HEADING_SEMANTICS_ATTR,
    )


def _load_heading_semantics(
    run: Any,
    *,
    identity: BoundRowIdentityContract,
    forward_descriptor: BoundCanonicalCoordinateDescriptor,
    body_frame: BoundFishAnatomicalBodyFrame,
) -> BoundCoordinateRecord:
    result = bind_persisted_coordinate_record(
        run["body_frame/heading_deg"],
        attr_name=SUBJECT_SHAPE_HEADING_SEMANTICS_ATTR,
    )
    expected = _heading_semantics_record(
        run,
        identity=identity,
        forward_descriptor=forward_descriptor,
        body_frame=body_frame,
    )
    if result.record != expected:
        _fail("Subject-shape heading semantics differ from live row-bound geometry.")
    return result


def _iter_arrays(group: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    for name in sorted(str(value) for value in group.array_keys()):
        path = f"{prefix}/{name}" if prefix else name
        yield path, group[name]
    for name in sorted(str(value) for value in group.group_keys()):
        path = f"{prefix}/{name}" if prefix else name
        yield from _iter_arrays(group[name], path)


def _subject_shape_payload_numerical_policy(
    array_content_sha256: Mapping[str, str],
    *,
    receipt_profile: str,
    verified_physical_copy: Mapping[str, Any] | None = None,
    receipt_composition: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    common = {
        "array_payload_canonicalization": "numpy_dtype_shape_c_order_bytes_v1",
        "array_content_sha256": {
            path: array_content_sha256[path]
            for path in sorted(array_content_sha256)
        },
        "closed_array_inventory": True,
        "mutation_exclusion_contract": (
            "exclusive_archive_publication_lock_then_immutable_run_lifecycle_v1"
        ),
        "normal_load_physical_rehash": False,
        "deep_audit_physical_rehash_available": True,
    }
    if receipt_profile == SUBJECT_SHAPE_PAYLOAD_RECEIPT_PROFILE_V1:
        if verified_physical_copy is not None or receipt_composition is not None:
            _fail("Legacy subject-shape scan receipts cannot carry transfer evidence.")
        return {
            **common,
            "array_digest_source": "single_locked_post_binding_decoded_scan_v1",
        }
    if receipt_profile != SUBJECT_SHAPE_PAYLOAD_RECEIPT_PROFILE_V2:
        _fail("Subject-shape payload receipt profile is unsupported.")
    physical_copy = _canonical_subject_shape_physical_copy_receipt(
        verified_physical_copy
    )
    composition = _canonical_subject_shape_receipt_composition(
        receipt_composition
    )
    return {
        **common,
        "array_digest_source": (
            "verified_staged_transfer_plus_final_binding_readback_v2"
        ),
        "mutation_exclusion_contract": (
            "exclusive_node_local_storage_then_atomic_verified_copy_then_"
            "archive_publication_lock_v2"
        ),
        "verified_physical_copy": physical_copy,
        "receipt_composition": composition,
    }


def _canonical_subject_shape_physical_copy_receipt(
    value: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail("Subject-shape v2 receipt lacks verified physical-copy evidence.")
    receipt = _json_copy(value)
    expected = {
        "backend",
        "verification",
        "file_count",
        "physical_bytes",
        "inventory_sha256",
        "content_sha256",
    }
    if (
        not isinstance(receipt, dict)
        or set(receipt) != expected
        or receipt.get("backend") not in {"python", "rsync"}
        or receipt.get("verification")
        not in {"sha256_all_physical_files", "rsync_checksum_dry_run"}
        or type(receipt.get("file_count")) is not int
        or receipt["file_count"] < 1
        or type(receipt.get("physical_bytes")) is not int
        or receipt["physical_bytes"] < 0
        or re.fullmatch(r"[0-9a-f]{64}", str(receipt.get("inventory_sha256")))
        is None
        or re.fullmatch(r"[0-9a-f]{64}", str(receipt.get("content_sha256")))
        is None
        or (
            receipt["backend"] == "python"
            and receipt["verification"] != "sha256_all_physical_files"
        )
        or (
            receipt["backend"] == "rsync"
            and receipt["verification"] != "rsync_checksum_dry_run"
        )
    ):
        _fail("Subject-shape verified physical-copy receipt is malformed.")
    return receipt


def _canonical_subject_shape_receipt_composition(
    value: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail("Subject-shape v2 receipt lacks its composition evidence.")
    receipt = _json_copy(value)
    expected = {
        "schema_id",
        "schema_version",
        "receipt_origin",
        "staged_decoded_payload_root_sha256",
        "appended_paths",
        "appended_decoded_bytes",
    }
    if (
        not isinstance(receipt, dict)
        or set(receipt) != expected
        or receipt.get("schema_id")
        != "palette.subject_shape_bound_payload_receipt_composition"
        or receipt.get("schema_version") != 1
        or receipt.get("receipt_origin")
        != "verified_staged_transfer_plus_final_binding_readback_v2"
        or re.fullmatch(
            r"[0-9a-f]{64}",
            str(receipt.get("staged_decoded_payload_root_sha256")),
        )
        is None
        or receipt.get("appended_paths")
        != list(sorted(SUBJECT_SHAPE_BINDING_APPENDED_ARRAY_PATHS))
        or type(receipt.get("appended_decoded_bytes")) is not int
        or receipt["appended_decoded_bytes"] < 0
    ):
        _fail("Subject-shape bound receipt composition is malformed.")
    return receipt


def _scan_subject_shape_bound_payload(
    run: Any,
    *,
    workers: int,
) -> tuple[dict[str, str], Mapping[str, Any]]:
    """Read the final arrays once before metadata-only authority stamping."""

    from fisheye.shared.subject_shape_storage import (
        build_subject_shape_bound_payload_scan_receipt,
    )

    if (
        SUBJECT_SHAPE_PAYLOAD_INTEGRITY_RECEIPT_ATTR in run.attrs
        or SUBJECT_SHAPE_PAYLOAD_VALIDATION_RECEIPT_ATTR in run.attrs
    ):
        _fail("Subject-shape payload receipt attributes are already occupied.")
    scan = build_subject_shape_bound_payload_scan_receipt(
        run,
        workers=workers,
    )
    raw_digests = scan.get("array_content_sha256")
    copy_report = scan.get("decoded_copy_report")
    if not isinstance(raw_digests, Mapping) or not isinstance(copy_report, Mapping):
        _fail("Subject-shape payload scan did not return closed digest evidence.")
    return (
        {str(path): str(value) for path, value in raw_digests.items()},
        copy_report,
    )


def _build_subject_shape_payload_integrity(
    run: Any,
    *,
    decoded_copy_report: Mapping[str, Any],
    payload_run_path: str | Path | None,
    hash_workers: int,
) -> dict[str, Any]:
    """Seal final physical and immutable metadata after authority stamping."""

    if (
        SUBJECT_SHAPE_PAYLOAD_INTEGRITY_RECEIPT_ATTR in run.attrs
        or SUBJECT_SHAPE_PAYLOAD_VALIDATION_RECEIPT_ATTR in run.attrs
    ):
        _fail("Subject-shape payload receipt attributes are already occupied.")
    payload_path = _subject_shape_payload_path(run, payload_run_path)
    integrity = build_payload_integrity_receipt(
        payload_path,
        run_ref=f"/{canonical_node_path(run)}",
        decoded_copy_report=decoded_copy_report,
        hash_workers=max(1, int(hash_workers)),
    )
    run.attrs[SUBJECT_SHAPE_PAYLOAD_INTEGRITY_RECEIPT_ATTR] = _json_copy(
        integrity
    )
    if run.attrs.get(SUBJECT_SHAPE_PAYLOAD_INTEGRITY_RECEIPT_ATTR) != integrity:
        _fail("Subject-shape payload integrity receipt did not persist exactly.")
    return integrity


def _subject_shape_payload_path(
    run: Any,
    payload_run_path: str | Path | None,
) -> Path:
    if payload_run_path is not None:
        return Path(payload_run_path).expanduser().resolve()
    identity = archive_identity(run)
    if identity.kind != "local_store_root":
        _fail("Subject-shape payload receipts require an explicit local run path.")
    return Path(str(identity.key[0])).joinpath(
        *str(run.path).strip("/").split("/")
    )


def _stamp_subject_shape_payload_validation(
    run: Any,
    *,
    integrity_receipt: Mapping[str, Any],
    manifest_sha256: str,
    array_content_sha256: Mapping[str, str],
    receipt_profile: str,
    verified_physical_copy: Mapping[str, Any] | None = None,
    receipt_composition: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    validation = build_payload_validation_receipt(
        integrity_receipt,
        scientific_manifest_schema_id=SUBJECT_SHAPE_MANIFEST_SCHEMA_ID,
        scientific_manifest_schema_version=SUBJECT_SHAPE_SCHEMA_VERSION,
        scientific_manifest_sha256=manifest_sha256,
        validator_schema_id=SUBJECT_SHAPE_PAYLOAD_VALIDATOR_SCHEMA_ID,
        validator_schema_version=SUBJECT_SHAPE_PAYLOAD_VALIDATOR_SCHEMA_VERSION,
        numerical_policy=_subject_shape_payload_numerical_policy(
            array_content_sha256,
            receipt_profile=receipt_profile,
            verified_physical_copy=verified_physical_copy,
            receipt_composition=receipt_composition,
        ),
    )
    run.attrs[SUBJECT_SHAPE_PAYLOAD_VALIDATION_RECEIPT_ATTR] = _json_copy(
        validation
    )
    if run.attrs.get(SUBJECT_SHAPE_PAYLOAD_VALIDATION_RECEIPT_ATTR) != validation:
        _fail("Subject-shape payload validation receipt did not persist exactly.")
    return validation


def _load_subject_shape_payload_digest_evidence(
    run: Any,
    *,
    manifest_sha256: str,
) -> Mapping[str, str] | None:
    """Load receipt-backed digests, falling back only for historical artifacts."""

    raw_integrity = run.attrs.get(SUBJECT_SHAPE_PAYLOAD_INTEGRITY_RECEIPT_ATTR)
    raw_validation = run.attrs.get(SUBJECT_SHAPE_PAYLOAD_VALIDATION_RECEIPT_ATTR)
    receipt_profile = run.attrs.get(SUBJECT_SHAPE_PAYLOAD_RECEIPT_PROFILE_ATTR)
    if raw_integrity is None and raw_validation is None:
        if receipt_profile is not None:
            _fail("Subject-shape receipt-profile publication lacks its receipts.")
        return None
    if not is_supported_subject_shape_payload_receipt_profile(receipt_profile):
        _fail("Subject-shape payload receipts lack their exact profile marker.")
    if not isinstance(raw_integrity, Mapping) or not isinstance(
        raw_validation,
        Mapping,
    ):
        _fail("Subject-shape payload receipt pair is incomplete or malformed.")
    try:
        integrity = canonical_payload_integrity_receipt(raw_integrity)
        if integrity.get("run_ref") != f"/{canonical_node_path(run)}":
            raise ValueError("payload integrity receipt names another run")
        validation = verify_payload_validation_receipt(
            raw_validation,
            integrity_receipt=integrity,
            expected_scientific_manifest_sha256=manifest_sha256,
            expected_validator_schema_id=SUBJECT_SHAPE_PAYLOAD_VALIDATOR_SCHEMA_ID,
            expected_validator_schema_version=(
                SUBJECT_SHAPE_PAYLOAD_VALIDATOR_SCHEMA_VERSION
            ),
        )
    except Exception as exc:
        _fail(f"Subject-shape payload receipt validation failed: {exc}")
    policy = validation.get("numerical_policy")
    if not isinstance(policy, Mapping):
        _fail("Subject-shape payload validation lacks its numerical policy.")
    digests = policy.get("array_content_sha256")
    common_fields = {
        "array_digest_source",
        "array_payload_canonicalization",
        "array_content_sha256",
        "closed_array_inventory",
        "mutation_exclusion_contract",
        "normal_load_physical_rehash",
        "deep_audit_physical_rehash_available",
    }
    if receipt_profile == SUBJECT_SHAPE_PAYLOAD_RECEIPT_PROFILE_V1:
        policy_valid = (
            set(policy) == common_fields
            and policy.get("array_digest_source")
            == "single_locked_post_binding_decoded_scan_v1"
            and policy.get("mutation_exclusion_contract")
            == "exclusive_archive_publication_lock_then_immutable_run_lifecycle_v1"
        )
    else:
        policy_valid = (
            set(policy)
            == common_fields | {"verified_physical_copy", "receipt_composition"}
            and policy.get("array_digest_source")
            == "verified_staged_transfer_plus_final_binding_readback_v2"
            and policy.get("mutation_exclusion_contract")
            == (
                "exclusive_node_local_storage_then_atomic_verified_copy_then_"
                "archive_publication_lock_v2"
            )
        )
        if policy_valid:
            _canonical_subject_shape_physical_copy_receipt(
                policy.get("verified_physical_copy")
            )
            _canonical_subject_shape_receipt_composition(
                policy.get("receipt_composition")
            )
    if (
        not policy_valid
        or policy.get("array_payload_canonicalization")
        != "numpy_dtype_shape_c_order_bytes_v1"
        or policy.get("closed_array_inventory") is not True
        or policy.get("normal_load_physical_rehash") is not False
        or policy.get("deep_audit_physical_rehash_available") is not True
        or not isinstance(digests, Mapping)
    ):
        _fail("Subject-shape payload validation policy is not exact.")
    live_arrays = dict(_iter_arrays(run))
    decoded_arrays = integrity["decoded_payload"]["arrays"]
    decoded_by_path = {str(value["path"]): value for value in decoded_arrays}
    if set(digests) != set(live_arrays) or set(decoded_by_path) != set(live_arrays):
        _fail("Subject-shape payload receipt array inventory differs from the run.")
    normalized: dict[str, str] = {}
    for path in sorted(live_arrays):
        node = live_arrays[path]
        digest = digests[path]
        decoded = decoded_by_path[path]
        if (
            type(digest) is not str
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or decoded.get("dtype") != str(np.dtype(node.dtype))
            or decoded.get("shape") != [int(value) for value in node.shape]
        ):
            _fail(f"Subject-shape payload receipt declaration differs for {path!r}.")
        normalized[path] = digest
    return normalized


def deep_audit_subject_shape_payload_receipt(
    run: Any,
    *,
    payload_run_path: str | Path | None = None,
    hash_workers: int = 4,
) -> dict[str, Any]:
    """Rehash one immutable publication against its sealed payload receipt.

    Normal scientific loads validate the digest-bound receipt and metadata
    declarations without rereading immutable payload files.  This explicit
    maintenance/audit path performs the expensive physical rehash when archive
    custody, transfer, or out-of-band mutation is in question.
    """

    manifest = bind_persisted_coordinate_record(
        run,
        attr_name=SUBJECT_SHAPE_MANIFEST_ATTR,
    )
    digests = _load_subject_shape_payload_digest_evidence(
        run,
        manifest_sha256=manifest.record_sha256,
    )
    if digests is None:
        _fail("Historical subject-shape publication lacks a payload receipt.")
    raw_integrity = run.attrs.get(SUBJECT_SHAPE_PAYLOAD_INTEGRITY_RECEIPT_ATTR)
    try:
        integrity = verify_payload_integrity_receipt(
            _subject_shape_payload_path(run, payload_run_path),
            raw_integrity,
            expected_run_ref=f"/{canonical_node_path(run)}",
            hash_workers=max(1, int(hash_workers)),
            verify_physical_payload=True,
        )
    except Exception as exc:
        _fail(f"Subject-shape deep payload audit failed: {exc}")
    return {
        "valid": True,
        "run_ref": integrity["run_ref"],
        "manifest_sha256": manifest.record_sha256,
        "integrity_receipt_sha256": integrity["record_sha256"],
        "decoded_payload_root_sha256": integrity["decoded_payload"]["root_sha256"],
        "physical_payload_root_sha256": integrity["physical_payload"]["root_sha256"],
        "immutable_metadata_root_sha256": integrity["immutable_metadata"][
            "root_sha256"
        ],
        "array_count": len(digests),
        "physical_rehash_performed": True,
    }


def _array_record(path: str, node: Any) -> dict[str, Any]:
    digests = _ACTIVE_SUBJECT_SHAPE_ARRAY_DIGESTS.get()
    if digests is None:
        content_sha256 = array_payload_sha256(node)
    else:
        content_sha256 = digests.get(path)
        if (
            type(content_sha256) is not str
            or re.fullmatch(r"[0-9a-f]{64}", content_sha256) is None
        ):
            _fail(
                f"Subject-shape payload receipt omits exact array digest {path!r}."
            )
    return {
        "array_ref": f"/{canonical_node_path(node)}",
        "relative_ref": path,
        "dtype": np.dtype(node.dtype).str,
        "shape": [int(value) for value in node.shape],
        "content_sha256": content_sha256,
        "canonicalization": "numpy_dtype_shape_c_order_bytes_v1",
    }


def _scalar_surface_specs(run: Any) -> dict[str, dict[str, Any]]:
    """Return the closed producer-owned scalar/profile semantic inventory."""

    specs: dict[str, dict[str, Any]] = {}

    def add(
        path: str,
        *,
        quantity: str,
        units: str,
        validity_path: str,
        derivation_method: str,
        surface_kind: str = "row_scalar",
        sign_convention: str = "nonnegative_magnitude",
        basis_paths: Sequence[str] = (),
        profile_axis: str | None = None,
        validity_policy: str = "validity_array_true_and_value_finite",
    ) -> None:
        if run.get(path) is None:
            return
        if run.get(validity_path) is None:
            _fail(f"Scalar surface {path!r} lacks validity authority {validity_path!r}.")
        specs[path] = {
            "quantity": quantity,
            "units": units,
            "validity_path": validity_path,
            "validity_policy": validity_policy,
            "derivation_method": derivation_method,
            "surface_kind": surface_kind,
            "sign_convention": sign_convention,
            "basis_paths": tuple(str(value) for value in basis_paths),
            "profile_axis": profile_axis,
        }

    for component in tuple(str(value) for value in (run.attrs.get("component_names") or ())):
        prefix = f"components/{component}"
        add(
            f"{prefix}/area_px",
            quantity="foreground_area",
            units="px^2",
            validity_path=f"{prefix}/mask_present",
            derivation_method="foreground_pixel_count_v1",
        )

    body = "components/subject_body"
    add(
        f"{body}/principal_axis_length_px",
        quantity="principal_axis_length",
        units="px",
        validity_path=f"{body}/principal_axis_valid",
        derivation_method="pca_mask_pixels_v1",
    )
    add(
        f"{body}/secondary_axis_length_px",
        quantity="secondary_axis_length",
        units="px",
        validity_path=f"{body}/principal_axis_valid",
        derivation_method="pca_mask_pixels_v1",
    )
    add(
        f"{body}/head_endpoint_to_snout_distance_px",
        quantity="head_endpoint_to_snout_distance",
        units="px",
        validity_path=f"{body}/centerline_valid",
        derivation_method="euclidean_distance_head_endpoint_to_snout_v1",
        basis_paths=(f"{body}/head_endpoint_xy", f"{body}/snout_tip_xy"),
    )
    add(
        f"{body}/bspline_arc_length_px",
        quantity="bspline_arclength",
        units="px",
        validity_path=f"{body}/bspline_valid",
        derivation_method="dense_interpolating_bspline_arclength_v1",
        basis_paths=(f"{body}/bspline_sample_xy",),
    )
    add(
        f"{body}/centerline_curvature_px_inv",
        quantity="signed_centerline_curvature",
        units="px^-1",
        validity_path=f"{body}/bspline_valid",
        derivation_method="separate_smoothing_spline_signed_curvature_v1",
        surface_kind="row_profile",
        sign_convention=(
            "kappa=(x_prime*y_double_prime-y_prime*x_double_prime)/"
            "(x_prime^2+y_prime^2)^1.5_in_source_camera_x_right_y_down;"
            "positive_is_clockwise_when_viewed_as_an_image;sample_direction=snout_to_tail"
        ),
        basis_paths=(f"{body}/bspline_sample_xy",),
        profile_axis="implicit_centerline_normalized_arclength",
    )
    add(
        f"{body}/tail_curvature_px_inv",
        quantity="signed_tail_curvature",
        units="px^-1",
        validity_path=f"{body}/tail_sample_valid",
        derivation_method="separate_smoothing_spline_signed_curvature_v1",
        surface_kind="row_profile",
        sign_convention=(
            "kappa=(x_prime*y_double_prime-y_prime*x_double_prime)/"
            "(x_prime^2+y_prime^2)^1.5_in_source_camera_x_right_y_down;"
            "positive_is_clockwise_when_viewed_as_an_image;sample_direction=tail_base_to_tail_tip"
        ),
        basis_paths=(f"{body}/tail_sample_xy", f"{body}/tail_tangent_xy"),
        profile_axis="tail_sample_s",
    )
    add(
        f"{body}/tail_base_arclength_px",
        quantity="snout_to_tail_base_arclength",
        units="px",
        validity_path=f"{body}/tail_base_valid",
        derivation_method="project_tail_base_onto_ordered_centerline_v1",
        basis_paths=(f"{body}/centerline_xy", f"{body}/tail_base_xy"),
    )
    add(
        f"{body}/tail_segment_arclength_px",
        quantity="tail_base_to_tail_tip_arclength",
        units="px",
        validity_path=f"{body}/tail_base_valid",
        derivation_method="ordered_centerline_tail_segment_arclength_v1",
        basis_paths=(f"{body}/centerline_xy", f"{body}/tail_base_xy"),
    )
    add(
        f"{body}/body_arclength_px",
        quantity="snout_to_tail_tip_centerline_arclength",
        units="px",
        validity_path=f"{body}/centerline_valid",
        derivation_method="ordered_centerline_arclength_v1",
        basis_paths=(f"{body}/centerline_xy",),
    )

    swim = "components/swim_bladder"
    add(
        f"{swim}/caudal_contour_projection_px",
        quantity="body_forward_projection_from_body_frame_origin",
        units="px",
        validity_path=f"{swim}/caudal_contour_valid",
        derivation_method="minimum_swim_bladder_contour_projection_v1",
        sign_convention="positive_toward_anatomical_forward_axis;negative_posterior",
        basis_paths=(
            f"{swim}/caudal_contour_point_xy",
            "body_frame/origin_xy",
            "body_frame/forward_axis_xy",
            "body_frame/axis_valid",
        ),
    )

    eye_pair = "relations/eye_pair"
    add(
        f"{eye_pair}/separation_px",
        quantity="eye_centroid_separation",
        units="px",
        validity_path=f"{eye_pair}/separation_valid",
        derivation_method="euclidean_distance_between_eye_centroids_v1",
        basis_paths=(
            "components/eye_left/centroid_xy",
            "components/eye_right/centroid_xy",
        ),
    )

    swim_relation = "relations/swim_bladder_to_body"
    add(
        f"{swim_relation}/distance_to_body_centroid_px",
        quantity="swim_bladder_to_body_centroid_distance",
        units="px",
        validity_path=f"{swim_relation}/relation_valid",
        derivation_method="euclidean_component_centroid_distance_v1",
        basis_paths=(
            "components/swim_bladder/centroid_xy",
            "components/subject_body/centroid_xy",
        ),
    )
    add(
        f"{swim_relation}/longitudinal_offset_px",
        quantity="swim_bladder_longitudinal_offset_in_persisted_body_principal_axis",
        units="px",
        validity_path=f"{swim_relation}/relation_valid",
        derivation_method="dot_component_offset_with_persisted_body_principal_axis_v1",
        sign_convention=(
            "signed_along_persisted_unoriented_principal_axis_xy;"
            "axis_polarity_is_the_digest_bound_array_value_not_an_anatomical_forward_claim"
        ),
        basis_paths=(
            "components/swim_bladder/centroid_xy",
            "components/subject_body/centroid_xy",
            "components/subject_body/principal_axis_xy",
            "components/subject_body/principal_axis_valid",
        ),
    )
    add(
        f"{swim_relation}/lateral_offset_px",
        quantity="swim_bladder_lateral_offset_in_persisted_body_principal_axis",
        units="px",
        validity_path=f"{swim_relation}/relation_valid",
        derivation_method="dot_component_offset_with_left_normal_of_persisted_principal_axis_v1",
        sign_convention=(
            "positive_along_perpendicular=(-axis_y,axis_x)_in_source_camera_xy;"
            "axis_polarity_is_the_digest_bound_array_value_not_an_anatomical_left_claim"
        ),
        basis_paths=(
            "components/swim_bladder/centroid_xy",
            "components/subject_body/centroid_xy",
            "components/subject_body/principal_axis_xy",
            "components/subject_body/principal_axis_valid",
        ),
    )

    eyes_relation = "relations/eyes_to_body"
    for prefix, component in (("left", "eye_left"), ("right", "eye_right")):
        valid_path = f"{eyes_relation}/{prefix}_eye_relation_valid"
        add(
            f"{eyes_relation}/{prefix}_eye_distance_to_body_centroid_px",
            quantity=f"{prefix}_eye_to_body_centroid_distance",
            units="px",
            validity_path=valid_path,
            derivation_method="euclidean_component_centroid_distance_v1",
            basis_paths=(
                f"components/{component}/centroid_xy",
                "components/subject_body/centroid_xy",
            ),
        )
        add(
            f"{eyes_relation}/{prefix}_eye_axis_angle_to_body_rad",
            quantity=f"{prefix}_eye_axis_angle_to_unoriented_body_principal_axis",
            units="rad",
            validity_path=valid_path,
            derivation_method="signed_angle_between_unoriented_axes_wrapped_half_pi_v1",
            sign_convention=(
                "atan2(cross(body_axis,eye_axis),dot(body_axis,eye_axis))_wrapped_to_[-pi/2,pi/2]"
            ),
            basis_paths=(
                f"components/{component}/ellipse_params",
                f"components/{component}/ellipse_success",
                "components/subject_body/principal_axis_xy",
                "components/subject_body/principal_axis_valid",
            ),
            validity_policy="validity_array_true_and_value_finite;ellipse_success_is_additional_derivation_gate",
        )

    add(
        "body_frame/heading_deg",
        quantity="anatomical_body_heading",
        units="deg",
        validity_path="body_frame/axis_valid",
        derivation_method="degrees_atan2_negative_y_x_v1",
        sign_convention="zero_positive_x;positive_counterclockwise_in_y_up_math_view",
        basis_paths=("body_frame/forward_axis_xy", "body_frame/axis_valid"),
    )
    return specs


def _scalar_surface_record(
    run: Any,
    path: str,
    spec: Mapping[str, Any],
    *,
    identity: BoundRowIdentityContract,
    scientific_configuration: BoundCoordinateRecord,
    tail_sample_axis: BoundCoordinateRecord,
) -> dict[str, Any]:
    node = run[path]
    row_count = int(identity.leading_dimension)
    shape = tuple(int(value) for value in node.shape)
    kind = str(spec["surface_kind"])
    if np.dtype(node.dtype).kind != "f":
        _fail(f"Scalar surface {path!r} must use a floating dtype.")
    if kind == "row_scalar":
        if shape != (row_count,):
            _fail(f"Scalar surface {path!r} must have shape {(row_count,)!r}.")
    elif kind == "row_profile":
        if len(shape) != 2 or shape[0] != row_count or shape[1] < 2:
            _fail(f"Profile surface {path!r} must have shape (rows, samples>=2).")
    else:
        _fail(f"Scalar surface {path!r} uses unsupported kind {kind!r}.")

    validity_path = str(spec["validity_path"])
    validity = run[validity_path]
    if tuple(int(value) for value in validity.shape) != (row_count,) or np.dtype(
        validity.dtype
    ).kind != "b":
        _fail(f"Scalar surface {path!r} has an invalid row-validity authority.")
    basis = []
    for basis_path in tuple(str(value) for value in spec.get("basis_paths", ())):
        basis_node = run.get(basis_path)
        if basis_node is None:
            _fail(f"Scalar surface {path!r} lacks basis surface {basis_path!r}.")
        basis.append(_array_record(basis_path, basis_node))

    profile_axis: dict[str, Any] | None = None
    axis_kind = spec.get("profile_axis")
    if axis_kind == "tail_sample_s":
        cardinality = int(tail_sample_axis.record["cardinality"])
        if kind != "row_profile" or shape[1] != cardinality:
            _fail(
                f"Scalar surface {path!r} does not match exact tail_sample_s cardinality."
            )
        profile_axis = {
            "axis_index": 1,
            "axis_kind": "persisted_normalized_arclength",
            "cardinality": cardinality,
            "axis_record": _record_pointer(tail_sample_axis),
            "sample_coordinate": _array_record(
                "components/subject_body/tail_sample_s",
                run["components/subject_body/tail_sample_s"],
            ),
        }
    elif axis_kind == "implicit_centerline_normalized_arclength":
        profile_axis = {
            "axis_index": 1,
            "axis_kind": "implicit_evenly_spaced_normalized_arclength",
            "cardinality": int(shape[1]),
            "domain": "closed_0_to_1",
            "sample_direction": "snout_to_tail",
            "authority": _record_pointer(scientific_configuration),
        }
    elif axis_kind is not None:
        _fail(f"Scalar surface {path!r} uses unsupported profile axis {axis_kind!r}.")

    record: dict[str, Any] = {
        "schema_id": SUBJECT_SHAPE_SCALAR_SURFACE_SCHEMA_ID,
        "schema_version": SUBJECT_SHAPE_SCHEMA_VERSION,
        "run_ref": f"/{canonical_node_path(run)}",
        "relative_ref": path,
        "surface_kind": kind,
        "quantity": str(spec["quantity"]),
        "units": str(spec["units"]),
        "sign_convention": str(spec["sign_convention"]),
        "surface": _array_record(path, node),
        "validity": {
            "policy": str(spec["validity_policy"]),
            "surface": _array_record(validity_path, validity),
        },
        "derivation": {
            "method": str(spec["derivation_method"]),
            "producer_method": run.attrs.get("method"),
            "producer_method_version": run.attrs.get("method_version"),
            "scientific_configuration": _record_pointer(scientific_configuration),
            "basis_surfaces": basis,
        },
        "row_identity": {
            "record_ref": identity.record_ref,
            "record_sha256": identity.record_sha256,
        },
    }
    if profile_axis is not None:
        record["profile_axis"] = profile_axis
    return record


@dataclass(frozen=True)
class BoundSubjectShapeScalarSurface:
    """Typed, digest-bound semantics for one persisted scalar/profile array."""

    relative_ref: str
    quantity: str
    units: str
    surface_kind: str
    semantics: BoundCoordinateRecord = field(repr=False)
    _node: Any = field(repr=False, compare=False)
    _validity_node: Any = field(repr=False, compare=False)

    @property
    def array_node(self) -> Any:
        return self._node

    @property
    def validity_node(self) -> Any:
        return self._validity_node


def _scalar_surface_inventory_record(
    run: Any,
    surfaces: Mapping[str, BoundSubjectShapeScalarSurface],
    *,
    identity: BoundRowIdentityContract,
    scientific_configuration: BoundCoordinateRecord,
    tail_sample_axis: BoundCoordinateRecord,
) -> dict[str, Any]:
    return {
        "schema_id": SUBJECT_SHAPE_SCALAR_SURFACE_INVENTORY_SCHEMA_ID,
        "schema_version": SUBJECT_SHAPE_SCHEMA_VERSION,
        "run_ref": f"/{canonical_node_path(run)}",
        "producer_method": run.attrs.get("method"),
        "producer_method_version": run.attrs.get("method_version"),
        "row_identity": {
            "record_ref": identity.record_ref,
            "record_sha256": identity.record_sha256,
        },
        "scientific_configuration": _record_pointer(scientific_configuration),
        "tail_sample_axis": _record_pointer(tail_sample_axis),
        "surfaces": {
            path: _record_pointer(binding.semantics)
            for path, binding in sorted(surfaces.items())
        },
        "closed_surface_inventory": True,
    }


def _scalar_surface_bindings(
    run: Any,
    *,
    identity: BoundRowIdentityContract,
    scientific_configuration: BoundCoordinateRecord,
    tail_sample_axis: BoundCoordinateRecord,
    load: bool,
) -> tuple[dict[str, BoundSubjectShapeScalarSurface], BoundCoordinateRecord]:
    specs = _scalar_surface_specs(run)
    surfaces: dict[str, BoundSubjectShapeScalarSurface] = {}
    for path, spec in sorted(specs.items()):
        node = run[path]
        expected = _scalar_surface_record(
            run,
            path,
            spec,
            identity=identity,
            scientific_configuration=scientific_configuration,
            tail_sample_axis=tail_sample_axis,
        )
        if load:
            semantics = bind_persisted_coordinate_record(
                node,
                attr_name=SUBJECT_SHAPE_SCALAR_SURFACE_ATTR,
            )
            if semantics.record != expected:
                _fail(f"Scalar surface semantics for {path!r} differ from live evidence.")
        else:
            semantics = stamp_and_bind_persisted_coordinate_record(
                node,
                expected,
                attr_name=SUBJECT_SHAPE_SCALAR_SURFACE_ATTR,
            )
        surfaces[path] = BoundSubjectShapeScalarSurface(
            relative_ref=path,
            quantity=str(spec["quantity"]),
            units=str(spec["units"]),
            surface_kind=str(spec["surface_kind"]),
            semantics=semantics,
            _node=node,
            _validity_node=run[str(spec["validity_path"])],
        )
    records = run.get("coordinate_records")
    if records is None:
        _fail("Subject-shape scalar surfaces lack coordinate_records.")
    if load:
        inventory_node = records.get("scalar_surface_inventory")
        if inventory_node is None:
            _fail("Subject-shape scalar-surface inventory node is missing.")
    else:
        inventory_node = records.require_group("scalar_surface_inventory")
    expected_inventory = _scalar_surface_inventory_record(
        run,
        surfaces,
        identity=identity,
        scientific_configuration=scientific_configuration,
        tail_sample_axis=tail_sample_axis,
    )
    if load:
        inventory = bind_persisted_coordinate_record(
            inventory_node,
            attr_name=SUBJECT_SHAPE_SCALAR_SURFACE_INVENTORY_ATTR,
        )
        if inventory.record != expected_inventory:
            _fail("Subject-shape scalar-surface inventory differs from live bindings.")
    else:
        inventory = stamp_and_bind_persisted_coordinate_record(
            inventory_node,
            expected_inventory,
            attr_name=SUBJECT_SHAPE_SCALAR_SURFACE_INVENTORY_ATTR,
        )
    return surfaces, inventory


def _manifest_record(
    run: Any,
    *,
    source: SubjectShapeCoordinateSource,
    source_binding: BoundCoordinateRecord | None,
    identity: BoundRowIdentityContract,
    temporal: BoundSourceRowTemporalAuthority,
    component_schema: BoundCoordinateRecord,
    scientific_configuration: BoundCoordinateRecord,
    tail_sample_axis: BoundCoordinateRecord,
    scalar_surfaces: Mapping[str, BoundSubjectShapeScalarSurface],
    scalar_surface_inventory: BoundCoordinateRecord,
    derivation: BoundCoordinateRecord,
    descriptors: Mapping[str, BoundCanonicalCoordinateDescriptor],
    body_frame: BoundFishAnatomicalBodyFrame,
    heading_semantics: BoundCoordinateRecord,
) -> dict[str, Any]:
    # Operational attrs such as cluster_output_staging, timings, lifecycle
    # receipts, and scheduler diagnostics are deliberately outside this seal.
    # They are never coordinate authority.  Scientific attrs are covered only
    # through the explicit scientific_configuration record above.
    schema_inventory = build_subject_shape_schema_inventory_record(
        run,
        phase="bound",
    )
    arrays = {
        path: _array_record(path, run[path])
        for path in schema_inventory["arrays"]
    }
    consumed_unbound_stage = load_subject_shape_consumed_unbound_stage(run)
    if _is_bundle_source(source):
        if source_binding is None:
            _fail("Bundle-bound manifest lacks its persisted source binding.")
        bundle = require_bound_subject_shape_bundle_source(source)
        source_section = {
            "source_kind": SUBJECT_SHAPE_BUNDLE_SOURCE_KIND,
            "bundle_id": bundle.bundle_id,
            "source_binding": _record_pointer(source_binding),
            "bundle_manifest_payload_digest": bundle.authority.bundle_manifest[
                "payload_digest"
            ],
            "refined_run_path": bundle.authority.refined_run_path,
            "refined_manifest_payload_digest": bundle.authority.refined_manifest[
                "payload_digest"
            ],
        }
        schema_version = SUBJECT_SHAPE_BUNDLE_MANIFEST_SCHEMA_VERSION
    else:
        source_section = {
            "run_path": source.context.run_path,
            "context": _record_pointer(source.context.context_record),
            "surface_inventory": _record_pointer(source.inventory),
            "component_qc_inventory": _record_pointer(
                source.component_qc_inventory
            ),
        }
        schema_version = SUBJECT_SHAPE_SCHEMA_VERSION
    return {
        "schema_id": SUBJECT_SHAPE_MANIFEST_SCHEMA_ID,
        "schema_version": schema_version,
        "run_ref": f"/{canonical_node_path(run)}",
        "row_identity": {
            "record_ref": identity.record_ref,
            "record_sha256": identity.record_sha256,
        },
        "temporal_authority": _temporal_pointer(temporal),
        "source_refined_subject_masks": source_section,
        "component_schema": _record_pointer(component_schema),
        "scientific_configuration": _record_pointer(scientific_configuration),
        "consumed_unbound_stage": _record_pointer(consumed_unbound_stage),
        "tail_sample_axis": _record_pointer(tail_sample_axis),
        "scalar_surface_inventory": _record_pointer(scalar_surface_inventory),
        "scalar_surfaces": {
            path: _record_pointer(binding.semantics)
            for path, binding in sorted(scalar_surfaces.items())
        },
        "derivation": _record_pointer(derivation),
        "body_frame": {
            "record_ref": body_frame.record_ref,
            "record_sha256": body_frame.record_sha256,
        },
        "heading_semantics": _record_pointer(heading_semantics),
        "coordinate_descriptors": {
            path: {
                "record_ref": (
                    f"/{canonical_node_path(binding.coordinate_node)}"
                    f"@{COORDINATE_DESCRIPTOR_ATTR}"
                ),
                "descriptor_sha256": binding.descriptor.digest(),
            }
            for path, binding in sorted(descriptors.items())
        },
        "arrays": arrays,
        "schema_inventory": schema_inventory,
        "closed_array_inventory": True,
        "closed_group_inventory": True,
        "closed_attr_inventory": True,
    }


@dataclass(frozen=True, init=False)
class BoundSubjectShapeCoordinatePublication:
    run_path: str
    source: SubjectShapeCoordinateSource = field(repr=False)
    source_binding: BoundCoordinateRecord | None = field(repr=False)
    row_identity: BoundRowIdentityContract = field(repr=False)
    temporal_authority: BoundSourceRowTemporalAuthority = field(repr=False)
    component_schema: BoundCoordinateRecord = field(repr=False)
    scientific_configuration: BoundCoordinateRecord = field(repr=False)
    tail_sample_axis: BoundCoordinateRecord = field(repr=False)
    scalar_surfaces: Mapping[str, BoundSubjectShapeScalarSurface]
    scalar_surface_inventory: BoundCoordinateRecord = field(repr=False)
    derivation: BoundCoordinateRecord = field(repr=False)
    descriptors: Mapping[str, BoundCanonicalCoordinateDescriptor]
    body_frame: BoundFishAnatomicalBodyFrame = field(repr=False)
    heading_semantics: BoundCoordinateRecord = field(repr=False)
    manifest: BoundCoordinateRecord = field(repr=False)
    component_names: tuple[str, ...]
    selector_eligible: bool
    publication_owner: str
    _root: Any = field(repr=False, compare=False)
    _run: Any = field(repr=False, compare=False)
    _seal: object = field(repr=False, compare=False)

    def __init__(self, *, _verification_seal: object | None = None, **values: Any) -> None:
        if _verification_seal is not _BOUND_PUBLICATION_SEAL:
            _fail("Subject-shape publications cannot be constructed directly.")
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_seal", _verification_seal)

    def require_scalar_surface(
        self,
        relative_ref: str,
        *,
        units: str | None = None,
        surface_kind: str | None = None,
    ) -> BoundSubjectShapeScalarSurface:
        """Return one already-verified scalar binding with optional expectations."""

        binding = self.scalar_surfaces.get(str(relative_ref))
        if binding is None:
            _fail(f"Subject-shape scalar surface {relative_ref!r} is unsupported.")
        if units is not None and binding.units != units:
            _fail(
                f"Subject-shape scalar surface {relative_ref!r} uses units "
                f"{binding.units!r}, expected {units!r}."
            )
        if surface_kind is not None and binding.surface_kind != surface_kind:
            _fail(
                f"Subject-shape scalar surface {relative_ref!r} uses kind "
                f"{binding.surface_kind!r}, expected {surface_kind!r}."
            )
        return binding


@dataclass(frozen=True, init=False)
class SealedSubjectShapePublicationMetadataProof:
    """Process-local lifecycle proof that never decodes scientific arrays."""

    run_path: str
    manifest: BoundCoordinateRecord = field(repr=False)
    row_count: int
    selector_eligible: bool
    publication_owner: str
    _seal: object = field(repr=False, compare=False)

    def __init__(
        self,
        *,
        _verification_seal: object | None = None,
        **values: Any,
    ) -> None:
        if _verification_seal is not _METADATA_PUBLICATION_SEAL:
            _fail("Subject-shape metadata proofs cannot be constructed directly.")
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_seal", _verification_seal)


@proof_verification_operation
def validate_sealed_subject_shape_publication_metadata(
    root: Any,
    run_path: str,
    *,
    expected_selector_eligible: bool,
    expected_publication_owner: str,
    payload_run_path: str | Path | None = None,
) -> SealedSubjectShapePublicationMetadataProof:
    """Validate one publication lifecycle epoch without decoding its arrays.

    This validator is intentionally limited to atomic publication internals.
    Ordinary scientific consumers still use the full authority resolver.  The
    proof checks the closed schema, manifest, receipt chain, live array
    metadata, and installed immutable Zarr metadata.  Explicit deep audit is
    the only path that rehashes physical payload bytes.
    """

    path = _canonical_run_path(run_path)
    run = _node(root, path, label="subject-shape metadata-only publication")
    owner = _require_state(
        run,
        complete=True,
        eligible=expected_selector_eligible,
        expected_owner=expected_publication_owner,
    )
    if (
        run.attrs.get("coordinate_contract")
        != SUBJECT_SHAPE_COORDINATE_CONTRACT
        or run.attrs.get(SUBJECT_SHAPE_COORDINATE_BINDING_STATUS_ATTR)
        != SUBJECT_SHAPE_BOUND_CANONICAL_STATUS
    ):
        _fail("Subject-shape metadata proof requires a bound canonical child.")
    _require_subject_shape_maintained_profile(run)
    manifest = bind_persisted_coordinate_record(
        run,
        attr_name=SUBJECT_SHAPE_MANIFEST_ATTR,
    )
    payload_digests = _load_subject_shape_payload_digest_evidence(
        run,
        manifest_sha256=manifest.record_sha256,
    )
    if payload_digests is None:
        _fail("Metadata-only validation requires sealed payload receipts.")
    raw_integrity = run.attrs.get(SUBJECT_SHAPE_PAYLOAD_INTEGRITY_RECEIPT_ATTR)
    try:
        verify_payload_integrity_receipt(
            _subject_shape_payload_path(run, payload_run_path),
            raw_integrity,
            expected_run_ref=f"/{path}",
            verify_physical_payload=False,
        )
    except Exception as exc:
        _fail(f"Subject-shape immutable metadata validation failed: {exc}")

    record = manifest.record
    schema_inventory = build_subject_shape_schema_inventory_record(
        run,
        phase="bound",
    )
    arrays = record.get("arrays")
    live_arrays = dict(_iter_arrays(run))
    expected_manifest_fields = {
        "schema_id",
        "schema_version",
        "run_ref",
        "row_identity",
        "temporal_authority",
        "source_refined_subject_masks",
        "component_schema",
        "scientific_configuration",
        "consumed_unbound_stage",
        "tail_sample_axis",
        "scalar_surface_inventory",
        "scalar_surfaces",
        "derivation",
        "body_frame",
        "heading_semantics",
        "coordinate_descriptors",
        "arrays",
        "schema_inventory",
        "closed_array_inventory",
        "closed_group_inventory",
        "closed_attr_inventory",
    }
    expected_manifest_version = (
        SUBJECT_SHAPE_BUNDLE_MANIFEST_SCHEMA_VERSION
        if run.attrs.get(SUBJECT_SHAPE_SOURCE_KIND_ATTR)
        == SUBJECT_SHAPE_BUNDLE_SOURCE_KIND
        else SUBJECT_SHAPE_SCHEMA_VERSION
    )
    if (
        set(record) != expected_manifest_fields
        or record.get("schema_id") != SUBJECT_SHAPE_MANIFEST_SCHEMA_ID
        or record.get("schema_version") != expected_manifest_version
        or record.get("run_ref") != f"/{path}"
        or record.get("schema_inventory") != schema_inventory
        or record.get("closed_array_inventory") is not True
        or record.get("closed_group_inventory") is not True
        or record.get("closed_attr_inventory") is not True
        or not isinstance(arrays, Mapping)
        or set(arrays) != set(live_arrays)
        or set(payload_digests) != set(live_arrays)
    ):
        _fail("Subject-shape metadata proof differs from its closed manifest.")
    for relative_ref, node in live_arrays.items():
        entry = arrays[relative_ref]
        if (
            not isinstance(entry, Mapping)
            or set(entry)
            != {
                "array_ref",
                "relative_ref",
                "dtype",
                "shape",
                "content_sha256",
                "canonicalization",
            }
            or entry.get("array_ref") != f"/{canonical_node_path(node)}"
            or entry.get("relative_ref") != relative_ref
            or entry.get("dtype") != np.dtype(node.dtype).str
            or entry.get("shape") != [int(value) for value in node.shape]
            or entry.get("content_sha256") != payload_digests[relative_ref]
            or entry.get("canonicalization")
            != "numpy_dtype_shape_c_order_bytes_v1"
        ):
            _fail(
                "Subject-shape metadata proof array declaration differs for "
                f"{relative_ref!r}."
            )
    instance_key = live_arrays.get("instance_key")
    if instance_key is None or not instance_key.shape:
        _fail("Subject-shape metadata proof lacks its row-identity array.")
    return SealedSubjectShapePublicationMetadataProof(
        run_path=path,
        manifest=manifest,
        row_count=int(instance_key.shape[0]),
        selector_eligible=expected_selector_eligible,
        publication_owner=owner,
        _verification_seal=_METADATA_PUBLICATION_SEAL,
    )


@dataclass(frozen=True)
class DeferredSubjectShapeCoordinateActivation:
    """Process-local receipt for one selected but still-ineligible child."""

    root: Any = field(repr=False, compare=False)
    parent: Any = field(repr=False, compare=False)
    run_name: str
    owner: str
    manifest_sha256: str
    snapshot: Mapping[str, tuple[bool, Any]] = field(repr=False, compare=False)
    overrides: Mapping[str, tuple[bool, Any]] = field(repr=False, compare=False)
    pending: Mapping[str, Any] = field(repr=False, compare=False)
    lease: Mapping[str, Any] = field(repr=False, compare=False)
    receipt_sha256: str
    _seal: object = field(repr=False, compare=False)


def _load_body_frame(
    run: Any,
    *,
    source: SubjectShapeCoordinateSource,
    source_binding: BoundCoordinateRecord | None,
    identity: BoundRowIdentityContract,
    component_schema: BoundCoordinateRecord,
    scientific_configuration: BoundCoordinateRecord,
    tail_sample_axis: BoundCoordinateRecord,
    derivation: BoundCoordinateRecord,
) -> BoundFishAnatomicalBodyFrame:
    records = run["coordinate_records"]
    contract_node = records["body_frame_contract"]
    estimator_node = records["body_frame_estimator"]
    contract = load_bound_body_frame_contract(
        contract_node,
        expected_record_ref=f"/{canonical_node_path(contract_node)}@{BODY_FRAME_CONTRACT_ATTR}",
        expected_record_sha256=contract_node.attrs[f"{BODY_FRAME_CONTRACT_ATTR}_sha256"],
    )
    estimator = load_bound_body_frame_estimator(
        estimator_node,
        expected_record_ref=f"/{canonical_node_path(estimator_node)}@{BODY_FRAME_ESTIMATOR_ATTR}",
        expected_record_sha256=estimator_node.attrs[f"{BODY_FRAME_ESTIMATOR_ATTR}_sha256"],
    )
    continuous_frame, _edge_frame = _source_camera_frames(source)
    source_descriptor = bind_body_source_coordinate_descriptor(
        run["component_centroid_xy"],
        row_identity=identity,
        source_camera_pixels=continuous_frame,
        lineage_records=_lineage_records(
            source,
            source_binding,
            component_schema,
            scientific_configuration,
            derivation,
        ),
    )
    source_manifest = bind_persisted_coordinate_record(
        run,
        attr_name=BODY_ESTIMATOR_SOURCE_MANIFEST_ATTR,
    )
    estimator_source = bind_mask_component_axis_source(
        source_descriptor=source_descriptor,
        estimator=estimator,
        component_schema=component_schema,
        validity_node=run["component_centroid_valid"],
        producer_manifest=source_manifest,
    )
    frame = run["body_frame"]
    geometry = bind_body_frame_geometry(
        frame,
        origin_xy_node=frame["origin_xy"],
        forward_axis_xy_node=frame["forward_axis_xy"],
        left_axis_xy_node=frame["left_axis_xy"],
        axis_valid_node=frame["axis_valid"],
        row_identity=identity,
        estimator_source=estimator_source,
    )
    return load_bound_fish_anatomical_body_frame(
        frame,
        expected_record_ref=f"/{canonical_node_path(frame)}@{FISH_ANATOMICAL_BODY_FRAME_ATTR}",
        expected_record_sha256=frame.attrs[f"{FISH_ANATOMICAL_BODY_FRAME_ATTR}_sha256"],
        expected_source_profile_id="source_camera_image_px.top_left_y_down.v1",
        expected_coordinate_units="px",
        expected_estimator_method="mask_component_axis",
        body_frame_contract=contract,
        estimator_source=estimator_source,
        geometry=geometry,
        row_identity=identity,
    )


def _load_subject_shape_publication(
    root: Any,
    run_path: str,
    *,
    eligible: bool,
    expected_owner: str | None = None,
) -> BoundSubjectShapeCoordinatePublication:
    path = _canonical_run_path(run_path)
    run = _node(root, path, label="subject-shape run")
    owner = _require_state(
        run,
        complete=True,
        eligible=eligible,
        expected_owner=expected_owner,
    )
    if run.attrs.get("coordinate_contract") != SUBJECT_SHAPE_COORDINATE_CONTRACT:
        _fail("Subject-shape run lacks canonical_v2 coordinate publication.")
    if (
        run.attrs.get(SUBJECT_SHAPE_COORDINATE_BINDING_STATUS_ATTR)
        != SUBJECT_SHAPE_BOUND_CANONICAL_STATUS
    ):
        _fail("Subject-shape run lacks exact bound canonical coordinate status.")
    _require_subject_shape_maintained_profile(run)
    manifest = bind_persisted_coordinate_record(
        run,
        attr_name=SUBJECT_SHAPE_MANIFEST_ATTR,
    )
    payload_digests = _load_subject_shape_payload_digest_evidence(
        run,
        manifest_sha256=manifest.record_sha256,
    )

    def digest_scope():
        return (
            nullcontext()
            if payload_digests is None
            else _subject_shape_array_digest_scope(payload_digests)
        )

    component_names = tuple(str(value) for value in (run.attrs.get("component_names") or ()))
    source = load_exact_subject_shape_source(root, run)
    source_binding = _load_source_binding(run, source)
    identity = load_bound_row_identity_contract(run, run["instance_key"])
    if not np.array_equal(
        np.asarray(run["instance_key"][:]),
        np.asarray(_source_row_node(source, "instance_key")[:]),
    ):
        _fail("Subject-shape instance_key order differs from selected refined rows.")
    for name in ("source_crop_row_ids", "source_acquisition_frame_index"):
        if not np.array_equal(
            np.asarray(run[name][:]),
            np.asarray(_source_row_node(source, name)[:]),
        ):
            _fail(f"Subject-shape {name} differs from selected refined rows.")
    for name in CANONICAL_SUBJECT_SHAPE_ROW_INDEX_ARRAYS:
        if not np.array_equal(
            np.asarray(run[f"row_index/{name}"][:]),
            np.asarray(run[name][:]),
        ):
            _fail(
                f"Subject-shape row_index/{name} differs from its canonical "
                "direct row-identity array."
            )
    temporal = load_subject_shape_temporal_authority(run, source, identity)
    schema_node = run["coordinate_records/component_schema"]
    component_schema = bind_persisted_coordinate_record(
        schema_node,
        attr_name=SUBJECT_SHAPE_COMPONENT_SCHEMA_ATTR,
    )
    if component_schema.record != _component_schema_record(component_names):
        _fail("Subject-shape component schema differs from run inventory.")
    scientific_configuration = _load_scientific_configuration(run)
    with digest_scope():
        tail_sample_axis = _load_tail_sample_axis(run)
        scalar_surfaces, scalar_surface_inventory = _scalar_surface_bindings(
            run,
            identity=identity,
            scientific_configuration=scientific_configuration,
            tail_sample_axis=tail_sample_axis,
            load=True,
        )
    derivation = bind_persisted_coordinate_record(
        run,
        attr_name=SUBJECT_SHAPE_DERIVATION_ATTR,
    )
    if derivation.record != _derivation_record(
        run,
        source,
        source_binding,
        identity,
        component_schema,
        temporal,
        scientific_configuration,
        tail_sample_axis,
        scalar_surface_inventory,
    ):
        _fail("Subject-shape derivation differs from live exact source evidence.")
    descriptors = _descriptor_bindings(
        run,
        source=source,
        source_binding=source_binding,
        identity=identity,
        component_schema=component_schema,
        scientific_configuration=scientific_configuration,
        tail_sample_axis=tail_sample_axis,
        derivation=derivation,
        component_names=component_names,
        load=True,
    )
    body_frame = _load_body_frame(
        run,
        source=source,
        source_binding=source_binding,
        identity=identity,
        component_schema=component_schema,
        scientific_configuration=scientific_configuration,
        tail_sample_axis=tail_sample_axis,
        derivation=derivation,
    )
    with digest_scope():
        heading_semantics = _load_heading_semantics(
            run,
            identity=identity,
            forward_descriptor=descriptors["body_frame/forward_axis_xy"],
            body_frame=body_frame,
        )
        expected_manifest = _manifest_record(
            run,
            source=source,
            source_binding=source_binding,
            identity=identity,
            temporal=temporal,
            component_schema=component_schema,
            scientific_configuration=scientific_configuration,
            tail_sample_axis=tail_sample_axis,
            scalar_surfaces=scalar_surfaces,
            scalar_surface_inventory=scalar_surface_inventory,
            derivation=derivation,
            descriptors=descriptors,
            body_frame=body_frame,
            heading_semantics=heading_semantics,
        )
    if manifest.record != expected_manifest:
        _fail("Subject-shape publication manifest differs from live arrays or authorities.")
    if run.attrs.get("publication_manifest_sha256") != manifest.record_sha256:
        _fail("Subject-shape publication manifest digest alias is stale.")
    return BoundSubjectShapeCoordinatePublication(
        run_path=path,
        source=source,
        source_binding=source_binding,
        row_identity=identity,
        temporal_authority=temporal,
        component_schema=component_schema,
        scientific_configuration=scientific_configuration,
        tail_sample_axis=tail_sample_axis,
        scalar_surfaces=scalar_surfaces,
        scalar_surface_inventory=scalar_surface_inventory,
        derivation=derivation,
        descriptors=descriptors,
        body_frame=body_frame,
        heading_semantics=heading_semantics,
        manifest=manifest,
        component_names=component_names,
        selector_eligible=eligible,
        publication_owner=owner,
        _root=root,
        _run=run,
        _verification_seal=_BOUND_PUBLICATION_SEAL,
    )


@proof_verification_operation
def load_persisted_subject_shape_coordinate_publication(
    root: Any,
    run_path: str,
    *,
    expected_publication_owner: str | None = None,
) -> BoundSubjectShapeCoordinatePublication:
    """Strict future reader.  Historical/ROI-only runs fail closed."""

    try:
        return _load_subject_shape_publication(
            root,
            run_path,
            eligible=True,
            expected_owner=expected_publication_owner,
        )
    except SubjectShapeCoordinatePublicationError:
        raise
    except Exception as exc:
        raise SubjectShapeCoordinatePublicationError(
            f"Subject-shape coordinate publication is invalid: {exc}"
        ) from exc


@proof_verification_operation
def load_completed_ineligible_subject_shape_coordinate_publication(
    root: Any,
    run_path: str,
    *,
    expected_publication_owner: str,
) -> BoundSubjectShapeCoordinatePublication:
    try:
        return _load_subject_shape_publication(
            root,
            run_path,
            eligible=False,
            expected_owner=expected_publication_owner,
        )
    except SubjectShapeCoordinatePublicationError:
        raise
    except Exception as exc:
        raise SubjectShapeCoordinatePublicationError(
            f"Ineligible subject-shape coordinate publication is invalid: {exc}"
        ) from exc


@proof_verification_operation
def publish_subject_shape_coordinate_surfaces(
    root: Any,
    run: Any,
    source: SubjectShapeCoordinateSource,
    *,
    component_names: Sequence[str],
    identity: BoundRowIdentityContract,
    component_schema: BoundCoordinateRecord,
    payload_run_path: str | Path | None = None,
    payload_hash_workers: int = 4,
    staged_decoded_copy_report: Mapping[str, Any] | None = None,
    staged_array_content_sha256: Mapping[str, str] | None = None,
    verified_physical_copy: Mapping[str, Any] | None = None,
    runtime_telemetry: Any | None = None,
) -> BoundSubjectShapeCoordinatePublication:
    """Transform and seal one running/ineligible child without selecting it."""

    def phase(name: str):
        if runtime_telemetry is None:
            return nullcontext()
        phase_factory = getattr(runtime_telemetry, "phase", None)
        if not callable(phase_factory):
            _fail("Subject-shape binding runtime telemetry has no phase interface.")
        return phase_factory(name)

    with phase("coordinate_source_binding_and_projection"):
        owner = _require_state(run, complete=False, eligible=False)
        # The computation preflight supplied this sealed source.  Requiring the
        # directed chain below freshly rechecks placement/identity metadata.  The
        # owning writer performs one complete source+output reload immediately
        # before activation, avoiding another full refined-raster hash pass here.
        if archive_identity(_source_run_group(source)) != archive_identity(run):
            _fail("Subject-shape and refined-mask source span archives/stores.")
        source_binding = _stamp_source_binding(run, source)
        numeric_projection = require_subject_shape_source_camera_numeric_projection(
            run,
            source,
        )
        if numeric_projection is None:
            transform_subject_shape_geometry_to_source_camera(
                run,
                source,
                component_names=component_names,
            )
        else:
            _stamp_subject_shape_source_camera_coordinate_labels(
                run,
                component_names=component_names,
            )
    with phase("binding_array_append"):
        _write_component_aggregate(run, component_names)
        if numeric_projection is None:
            _rewrite_body_frame_from_camera_components(run, component_names)
        else:
            _finalize_preprojected_body_frame(run)
    staged_evidence = (
        staged_decoded_copy_report,
        staged_array_content_sha256,
        verified_physical_copy,
    )
    receipt_composition: Mapping[str, Any] | None = None
    if any(value is not None for value in staged_evidence):
        with phase("staged_receipt_composition"):
            if any(value is None for value in staged_evidence):
                _fail("Subject-shape staged receipt evidence is incomplete.")
            if numeric_projection is None:
                _fail(
                    "Subject-shape staged receipts cannot cover a final-path numeric "
                    "rewrite."
                )
            from fisheye.shared.subject_shape_storage import (
                compose_subject_shape_bound_payload_receipt,
            )

            composed = compose_subject_shape_bound_payload_receipt(
                run,
                staged_decoded_copy_report=staged_decoded_copy_report,
                staged_array_content_sha256=staged_array_content_sha256,
                appended_paths=SUBJECT_SHAPE_BINDING_APPENDED_ARRAY_PATHS,
                workers=max(1, int(payload_hash_workers)),
            )
            raw_digests = composed.get("array_content_sha256")
            raw_copy_report = composed.get("decoded_copy_report")
            if not isinstance(raw_digests, Mapping) or not isinstance(
                raw_copy_report,
                Mapping,
            ):
                _fail("Subject-shape bound receipt composition is incomplete.")
            array_digests = {
                str(path): str(value) for path, value in raw_digests.items()
            }
            decoded_copy_report = raw_copy_report
            receipt_profile = SUBJECT_SHAPE_PAYLOAD_RECEIPT_PROFILE_V2
            receipt_composition = {
                key: composed[key]
                for key in (
                    "schema_id",
                    "schema_version",
                    "receipt_origin",
                    "staged_decoded_payload_root_sha256",
                    "appended_paths",
                    "appended_decoded_bytes",
                )
            }
    else:
        with phase("legacy_post_binding_decoded_scan"):
            array_digests, decoded_copy_report = _scan_subject_shape_bound_payload(
                run,
                workers=max(1, int(payload_hash_workers)),
            )
            receipt_profile = SUBJECT_SHAPE_PAYLOAD_RECEIPT_PROFILE_V1
    with phase("authority_stamping"):
        run.attrs[SUBJECT_SHAPE_PAYLOAD_RECEIPT_PROFILE_ATTR] = receipt_profile
        with _subject_shape_array_digest_scope(array_digests):
            temporal = stamp_subject_shape_temporal_authority(run, source, identity)
            scientific_configuration = _stamp_scientific_configuration(run)
            tail_sample_axis = _stamp_tail_sample_axis(run)
            scalar_surfaces, scalar_surface_inventory = _scalar_surface_bindings(
                run,
                identity=identity,
                scientific_configuration=scientific_configuration,
                tail_sample_axis=tail_sample_axis,
                load=False,
            )
            derivation = stamp_subject_shape_derivation(
                run,
                source,
                source_binding,
                identity,
                component_schema,
                temporal,
                scientific_configuration,
                tail_sample_axis,
                scalar_surface_inventory,
            )
            descriptors = _descriptor_bindings(
                run,
                source=source,
                source_binding=source_binding,
                identity=identity,
                component_schema=component_schema,
                scientific_configuration=scientific_configuration,
                tail_sample_axis=tail_sample_axis,
                derivation=derivation,
                component_names=component_names,
                load=False,
            )
            stamp_bound_canonical_coordinate_descriptors(descriptors.values())
            body_frame = _stamp_body_frame(
                run,
                source=source,
                source_binding=source_binding,
                identity=identity,
                component_schema=component_schema,
                scientific_configuration=scientific_configuration,
                tail_sample_axis=tail_sample_axis,
                derivation=derivation,
            )
            heading_semantics = _stamp_heading_semantics(
                run,
                identity=identity,
                forward_descriptor=descriptors["body_frame/forward_axis_xy"],
                body_frame=body_frame,
            )
            manifest_record = _manifest_record(
                run,
                source=source,
                source_binding=source_binding,
                identity=identity,
                temporal=temporal,
                component_schema=component_schema,
                scientific_configuration=scientific_configuration,
                tail_sample_axis=tail_sample_axis,
                scalar_surfaces=scalar_surfaces,
                scalar_surface_inventory=scalar_surface_inventory,
                derivation=derivation,
                descriptors=descriptors,
                body_frame=body_frame,
                heading_semantics=heading_semantics,
            )
            manifest = stamp_and_bind_persisted_coordinate_record(
                run,
                manifest_record,
                attr_name=SUBJECT_SHAPE_MANIFEST_ATTR,
            )
        run.attrs["publication_manifest_sha256"] = manifest.record_sha256
        run.attrs["coordinate_contract"] = SUBJECT_SHAPE_COORDINATE_CONTRACT
    with phase("physical_payload_hash"):
        integrity_receipt = _build_subject_shape_payload_integrity(
            run,
            decoded_copy_report=decoded_copy_report,
            payload_run_path=payload_run_path,
            hash_workers=payload_hash_workers,
        )
    with phase("validation_receipt_stamping"):
        _stamp_subject_shape_payload_validation(
            run,
            integrity_receipt=integrity_receipt,
            manifest_sha256=manifest.record_sha256,
            array_content_sha256=array_digests,
            receipt_profile=receipt_profile,
            verified_physical_copy=(
                verified_physical_copy
                if receipt_profile == SUBJECT_SHAPE_PAYLOAD_RECEIPT_PROFILE_V2
                else None
            ),
            receipt_composition=receipt_composition,
        )
        if (
            run.attrs.get("publication_manifest_sha256") != manifest.record_sha256
            or run.attrs.get("coordinate_contract")
            != SUBJECT_SHAPE_COORDINATE_CONTRACT
            or _owner(run, expected=owner) != owner
        ):
            _fail("Subject-shape coordinate publication attrs did not persist exactly.")
    # Completion is performed by the owning writer.  Return the sealed pieces
    # now only for diagnostics; activation requires a fresh complete reload.
    return BoundSubjectShapeCoordinatePublication(
        run_path=canonical_node_path(run),
        source=source,
        source_binding=source_binding,
        row_identity=identity,
        temporal_authority=temporal,
        component_schema=component_schema,
        scientific_configuration=scientific_configuration,
        tail_sample_axis=tail_sample_axis,
        scalar_surfaces=scalar_surfaces,
        scalar_surface_inventory=scalar_surface_inventory,
        derivation=derivation,
        descriptors=descriptors,
        body_frame=body_frame,
        heading_semantics=heading_semantics,
        manifest=manifest,
        component_names=tuple(str(value) for value in component_names),
        selector_eligible=False,
        publication_owner=owner,
        _root=root,
        _run=run,
        _verification_seal=_BOUND_PUBLICATION_SEAL,
    )


def selector_snapshot(parent: Any) -> dict[str, tuple[bool, Any]]:
    names = (*_ACTIVATION_SNAPSHOT_ATTRS, "latest_pending")
    return {
        name: (name in parent.attrs, copy.deepcopy(parent.attrs.get(name)))
        for name in names
    }


def _snapshot_value(snapshot: Mapping[str, tuple[bool, Any]], name: str) -> tuple[bool, Any]:
    value = snapshot.get(name)
    if not isinstance(value, tuple) or len(value) != 2 or type(value[0]) is not bool:
        _fail(f"Subject-shape selector snapshot lacks {name!r}.")
    return value


def _snapshot_unchanged(
    parent: Any,
    snapshot: Mapping[str, tuple[bool, Any]],
    names: Sequence[str] = _ACTIVATION_SNAPSHOT_ATTRS,
) -> None:
    for name in names:
        present, value = _snapshot_value(snapshot, name)
        if (name in parent.attrs) is not present or (present and parent.attrs.get(name) != value):
            _fail(f"Concurrent subject-shape selector mutation detected for {name!r}.")


def _require_activation_state(
    parent: Any,
    snapshot: Mapping[str, tuple[bool, Any]],
    *,
    overrides: Mapping[str, tuple[bool, Any]],
) -> None:
    """Compare every guarded selector and lifecycle attr against one epoch."""

    for name in (*_ACTIVATION_SNAPSHOT_ATTRS, "latest_pending"):
        present, value = overrides.get(name, _snapshot_value(snapshot, name))
        if (name in parent.attrs) is not present or (
            present and parent.attrs.get(name) != value
        ):
            _fail(f"Concurrent subject-shape activation mutation detected for {name!r}.")


def _base_generation(snapshot: Mapping[str, tuple[bool, Any]]) -> int:
    present, value = _snapshot_value(snapshot, SUBJECT_SHAPE_PUBLICATION_GENERATION_ATTR)
    if not present:
        return 0
    if type(value) is not int or value < 0:
        _fail("Subject-shape publication generation is invalid.")
    return value


def build_subject_shape_pending_receipt(
    *,
    run_name: str,
    owner: str,
    manifest_sha256: str,
    snapshot: Mapping[str, tuple[bool, Any]],
) -> dict[str, Any]:
    """Build the exact owner-bound pending receipt for one activation attempt."""

    if not isinstance(run_name, str) or not run_name or "/" in run_name:
        _fail("Subject-shape pending receipt requires a canonical run name.")
    if not isinstance(owner, str) or _OWNER_RE.fullmatch(owner) is None:
        _fail("Subject-shape pending receipt requires an exact owner token.")
    if not isinstance(manifest_sha256, str) or re.fullmatch(
        r"[0-9a-f]{64}", manifest_sha256
    ) is None:
        _fail("Subject-shape pending receipt requires a manifest digest.")
    base = _base_generation(snapshot)
    return {
        "schema_id": "palette.subject_shape_publication_pending",
        "schema_version": 1,
        "policy": SUBJECT_SHAPE_PUBLICATION_POLICY,
        "run_path": f"analysis/subject_shape_runs/{run_name}",
        "publication_owner": owner,
        "owner_uuid": owner,
        "publication_manifest_sha256": manifest_sha256,
        "base_generation": base,
        "next_generation": base + 1,
    }


def _deferred_activation_receipt_payload(
    *,
    run_name: str,
    owner: str,
    manifest_sha256: str,
    snapshot: Mapping[str, tuple[bool, Any]],
    overrides: Mapping[str, tuple[bool, Any]],
    pending: Mapping[str, Any],
    lease: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_id": "palette.subject_shape_deferred_activation_receipt",
        "schema_version": 1,
        "run_name": run_name,
        "run_path": f"analysis/subject_shape_runs/{run_name}",
        "publication_owner": owner,
        "publication_manifest_sha256": manifest_sha256,
        "snapshot": copy.deepcopy(dict(snapshot)),
        "overrides": copy.deepcopy(dict(overrides)),
        "pending": copy.deepcopy(dict(pending)),
        "lease": copy.deepcopy(dict(lease)),
    }


def _install_subject_shape_pending_receipt(
    root: Any,
    parent: Any,
    proof: (
        BoundSubjectShapeCoordinatePublication
        | SealedSubjectShapePublicationMetadataProof
    ),
    *,
    run_name: str,
    owner: str,
    snapshot: Mapping[str, tuple[bool, Any]],
) -> dict[str, Any]:
    """Transaction-internal pending write against the exact captured epoch."""

    present, _value = _snapshot_value(snapshot, "latest_pending")
    if present:
        _fail("Subject-shape activation refuses an occupied latest_pending receipt.")
    pending = build_subject_shape_pending_receipt(
        run_name=run_name,
        owner=owner,
        manifest_sha256=proof.manifest.record_sha256,
        snapshot=snapshot,
    )

    def fresh_parent() -> Any:
        current = _node(
            root,
            "analysis/subject_shape_runs",
            label="subject-shape pending parent",
        )
        if archive_identity(current) != archive_identity(parent):
            _fail("Subject-shape pending parent changed archives/stores.")
        return current

    current = fresh_parent()
    _require_activation_state(current, snapshot, overrides={})
    current.attrs["latest_pending"] = copy.deepcopy(pending)
    current = fresh_parent()
    _require_activation_state(
        current,
        snapshot,
        overrides={"latest_pending": (True, pending)},
    )
    return pending


def rollback_subject_shape_activation(
    root: Any,
    parent: Any,
    run: Any,
    *,
    run_name: str,
    owner: str,
    snapshot: Mapping[str, tuple[bool, Any]],
    attempted_pending: Mapping[str, Any],
    attempted_lease: Mapping[str, Any],
) -> None:
    """Restore only values still proven to belong to this exact attempt."""

    failures: list[str] = []
    if run.attrs.get(SUBJECT_SHAPE_PUBLICATION_OWNER_ATTR) == owner:
        try:
            run.attrs["stage_selector_eligible"] = False
        except BaseException as exc:  # pragma: no cover - hostile store
            failures.append(f"eligibility: {exc}")
    def fresh_parent() -> Any:
        current = _node(
            root,
            "analysis/subject_shape_runs",
            label="subject-shape rollback parent",
        )
        if archive_identity(current) != archive_identity(parent):
            _fail("Subject-shape rollback parent changed archives/stores.")
        return current


    def attr_state(attrs: Mapping[str, Any], name: str) -> tuple[bool, Any]:
        return (name in attrs, attrs.get(name))

    snapshot_generation_state = _snapshot_value(
        snapshot,
        SUBJECT_SHAPE_PUBLICATION_GENERATION_ATTR,
    )
    attempted_generation_state = (True, _base_generation(snapshot) + 1)

    def fresh_owned_epoch(
        *,
        allowed_generation_states: Sequence[tuple[bool, Any]],
    ) -> Any | None:
        """Return a fresh parent only while this exact lease owns the epoch."""

        current = fresh_parent()
        attrs = current.attrs
        if attrs.get(SUBJECT_SHAPE_PARENT_PUBLICATION_LEASE_ATTR) != attempted_lease:
            return None
        if attr_state(
            attrs,
            SUBJECT_SHAPE_PUBLICATION_GENERATION_ATTR,
        ) not in allowed_generation_states:
            return None
        return current

    attempted_selectors: dict[str, Any] = {
        "latest": run_name,
        "latest_complete": run_name,
    }
    # Selector values include the unique run name, so they remain safe to
    # restore conditionally even if another writer has taken the lifecycle
    # lease.  Shared policy/generation values are not unique and therefore need
    # a fresh exact lease-and-generation ownership proof below.
    for name, attempted_value in attempted_selectors.items():
        try:
            current_parent = fresh_parent()
            attrs = current_parent.attrs
            if attrs.get(name) != attempted_value:
                continue
            present, value = _snapshot_value(snapshot, name)
            if present:
                attrs[name] = copy.deepcopy(value)
            else:
                del attrs[name]
        except BaseException as exc:  # pragma: no cover - hostile store
            failures.append(f"{name}: {exc}")

    name = SUBJECT_SHAPE_PUBLICATION_GENERATION_ATTR
    try:
        current_parent = fresh_owned_epoch(
            allowed_generation_states=(attempted_generation_state,),
        )
        if current_parent is not None:
            attrs = current_parent.attrs
            present, value = snapshot_generation_state
            if present:
                attrs[name] = copy.deepcopy(value)
            else:
                del attrs[name]
    except BaseException as exc:  # pragma: no cover - hostile store
        failures.append(f"{name}: {exc}")

    name = SUBJECT_SHAPE_PUBLICATION_POLICY_ATTR
    try:
        current_parent = fresh_owned_epoch(
            allowed_generation_states=(
                snapshot_generation_state,
                attempted_generation_state,
            ),
        )
        if (
            current_parent is not None
            and current_parent.attrs.get(name) == SUBJECT_SHAPE_PUBLICATION_POLICY
        ):
            attrs = current_parent.attrs
            present, value = _snapshot_value(snapshot, name)
            if present:
                attrs[name] = copy.deepcopy(value)
            else:
                del attrs[name]
    except BaseException as exc:  # pragma: no cover - hostile store
        failures.append(f"{name}: {exc}")

    try:
        current_parent = fresh_parent()
        attrs = current_parent.attrs
        if attrs.get("latest_pending") == attempted_pending:
            # The pending receipt is owner/digest bound, unlike shared policy
            # and generation literals.  Removing only this exact attempt's
            # receipt cannot erase a successor's pending state.
            present, value = _snapshot_value(snapshot, "latest_pending")
            if present:
                attrs["latest_pending"] = copy.deepcopy(value)
            else:
                del attrs["latest_pending"]
    except BaseException as exc:  # pragma: no cover - hostile store
        failures.append(f"latest_pending: {exc}")

    name = SUBJECT_SHAPE_PARENT_PUBLICATION_LEASE_ATTR
    try:
        current_parent = fresh_owned_epoch(
            allowed_generation_states=(
                snapshot_generation_state,
                attempted_generation_state,
            ),
        )
        # Restore the lease last, after every other non-selector mutation has
        # freshly proven ownership of this exact attempted epoch.
        if current_parent is not None:
            attrs = current_parent.attrs
            present, value = _snapshot_value(snapshot, name)
            if present:
                attrs[name] = copy.deepcopy(value)
            else:
                del attrs[name]
    except BaseException as exc:  # pragma: no cover - hostile store
        failures.append(f"{name}: {exc}")
    if failures:
        raise RuntimeError(f"Subject-shape activation rollback was incomplete: {failures!r}.")


@proof_verification_operation
def activate_subject_shape_coordinate_publication(
    root: Any,
    parent: Any,
    proof: (
        BoundSubjectShapeCoordinatePublication
        | SealedSubjectShapePublicationMetadataProof
    ),
    *,
    run_name: str,
    owner: str,
    snapshot: Mapping[str, tuple[bool, Any]],
    defer_eligibility: bool = False,
) -> DeferredSubjectShapeCoordinateActivation | None:
    """Publish selectors, then flip child eligibility as the final mutation."""

    full_proof = (
        type(proof) is BoundSubjectShapeCoordinatePublication
        and proof._seal is _BOUND_PUBLICATION_SEAL
    )
    metadata_proof = (
        type(proof) is SealedSubjectShapePublicationMetadataProof
        and proof._seal is _METADATA_PUBLICATION_SEAL
    )
    if not (full_proof or metadata_proof):
        _fail("Subject-shape activation requires a sealed proof.")
    expected_path = f"analysis/subject_shape_runs/{run_name}"
    run = _node(root, expected_path, label="subject-shape activation child")
    if (
        run.attrs.get(SUBJECT_SHAPE_SOURCE_KIND_ATTR)
        == SUBJECT_SHAPE_BUNDLE_SOURCE_KIND
    ):
        if (
            run.attrs.get(SUBJECT_SHAPE_STORAGE_PROFILE_ID_ATTR)
            != SUBJECT_SHAPE_ACCESS_AWARE_SUPPORTED_PROFILE_ID
            or run.attrs.get(SUBJECT_SHAPE_STORAGE_PROFILE_ROLE_ATTR)
            != SUBJECT_SHAPE_SUPPORTED_STORAGE_PROFILE_ROLE
            or SUBJECT_SHAPE_STORAGE_CANDIDATE_ATTR in run.attrs
        ):
            _fail(
                "Recording-bundle subject-shape v5 can become selector-visible "
                "only through its supported access-aware publication profile."
            )
    if proof.run_path != expected_path or proof.selector_eligible is not False:
        _fail("Subject-shape activation proof names the wrong child/state.")
    _require_state(run, complete=True, eligible=False, expected_owner=owner)
    base = _base_generation(snapshot)
    pending = build_subject_shape_pending_receipt(
        run_name=run_name,
        owner=owner,
        manifest_sha256=proof.manifest.record_sha256,
        snapshot=snapshot,
    )
    lease = {
        "schema_id": "palette.subject_shape_publication_lease",
        "schema_version": 1,
        "policy": SUBJECT_SHAPE_PUBLICATION_POLICY,
        "run_path": expected_path,
        "publication_owner": owner,
        "owner_uuid": owner,
        "base_generation": base,
        "next_generation": base + 1,
        "pending_receipt_sha256": _canonical_sha256(pending),
    }

    def fresh_parent() -> Any:
        current = _node(
            root,
            "analysis/subject_shape_runs",
            label="subject-shape activation parent",
        )
        if archive_identity(current) != archive_identity(parent):
            _fail("Subject-shape activation parent changed archives/stores.")
        return current

    overrides: dict[str, tuple[bool, Any]] = {}
    try:
        parent = fresh_parent()
        _require_activation_state(parent, snapshot, overrides=overrides)

        # Reconstruct the complete child while no parent selector has changed.
        # The supplied proof is an ownership/intent receipt; this live reload
        # establishes the exact child and source graph for this activation.
        if is_supported_subject_shape_payload_receipt_profile(
            run.attrs.get(SUBJECT_SHAPE_PAYLOAD_RECEIPT_PROFILE_ATTR)
        ):
            fresh_proof = validate_sealed_subject_shape_publication_metadata(
                root,
                expected_path,
                expected_selector_eligible=False,
                expected_publication_owner=owner,
            )
        else:
            fresh_proof = _load_subject_shape_publication(
                root,
                expected_path,
                eligible=False,
                expected_owner=owner,
            )
        if fresh_proof.manifest.record_sha256 != proof.manifest.record_sha256:
            _fail("Subject-shape publication changed before activation.")

        # Recheck and discard all reused proof state before the first parent
        # mutation.  Reuse is therefore only a validation optimization, never
        # authorization to publish stale evidence.
        finish_proof_verification()

        installed_pending = _install_subject_shape_pending_receipt(
            root,
            parent,
            proof,
            run_name=run_name,
            owner=owner,
            snapshot=snapshot,
        )
        if installed_pending != pending:
            _fail("Subject-shape activation installed the wrong pending receipt.")
        overrides["latest_pending"] = (True, pending)
        parent = fresh_parent()
        _require_activation_state(parent, snapshot, overrides=overrides)
        lease_present, prior_lease = _snapshot_value(
            snapshot,
            SUBJECT_SHAPE_PARENT_PUBLICATION_LEASE_ATTR,
        )
        if lease_present:
            if not isinstance(prior_lease, Mapping):
                _fail("Subject-shape parent publication lease is malformed.")
            prior_path = prior_lease.get("run_path")
            prior_name = (
                str(prior_path).rsplit("/", 1)[-1]
                if isinstance(prior_path, str)
                else ""
            )
            try:
                prior_child = _node(
                    root,
                    f"analysis/subject_shape_runs/{prior_name}",
                    label="prior subject-shape publication child",
                )
            except Exception as exc:
                _fail(f"Prior subject-shape publication receipt is unresolved: {exc}.")
            if (
                prior_lease.get("policy") != SUBJECT_SHAPE_PUBLICATION_POLICY
                or prior_lease.get("next_generation") != base
                or parent.attrs.get("latest") != prior_name
                or prior_child.attrs.get("stage_selector_eligible") is not True
                or prior_child.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE
            ):
                _fail(
                    "Subject-shape parent has an uncommitted or inconsistent publication lease."
                )
        policy_present, policy = _snapshot_value(snapshot, SUBJECT_SHAPE_PUBLICATION_POLICY_ATTR)
        if policy_present and policy != SUBJECT_SHAPE_PUBLICATION_POLICY:
            _fail("Subject-shape parent uses an unsupported publication policy.")
        parent = fresh_parent()
        _require_activation_state(parent, snapshot, overrides=overrides)
        parent.attrs[SUBJECT_SHAPE_PARENT_PUBLICATION_LEASE_ATTR] = copy.deepcopy(lease)
        overrides[SUBJECT_SHAPE_PARENT_PUBLICATION_LEASE_ATTR] = (True, lease)

        parent = fresh_parent()
        _require_activation_state(parent, snapshot, overrides=overrides)
        parent.attrs["latest_complete"] = run_name

        overrides["latest_complete"] = (True, run_name)
        parent = fresh_parent()
        _require_activation_state(parent, snapshot, overrides=overrides)
        parent.attrs["latest"] = run_name

        overrides["latest"] = (True, run_name)
        parent = fresh_parent()
        _require_activation_state(parent, snapshot, overrides=overrides)
        parent.attrs[SUBJECT_SHAPE_PUBLICATION_POLICY_ATTR] = SUBJECT_SHAPE_PUBLICATION_POLICY

        overrides[SUBJECT_SHAPE_PUBLICATION_POLICY_ATTR] = (
            True,
            SUBJECT_SHAPE_PUBLICATION_POLICY,
        )
        parent = fresh_parent()
        _require_activation_state(parent, snapshot, overrides=overrides)
        parent.attrs[SUBJECT_SHAPE_PUBLICATION_GENERATION_ATTR] = base + 1

        overrides[SUBJECT_SHAPE_PUBLICATION_GENERATION_ATTR] = (True, base + 1)
        parent = fresh_parent()
        _require_activation_state(parent, snapshot, overrides=overrides)
        del parent.attrs["latest_pending"]

        overrides["latest_pending"] = (False, None)
        parent = fresh_parent()
        _require_activation_state(parent, snapshot, overrides=overrides)

        # Re-read the complete child after the guarded parent-write sequence,
        # then compare the full selector/lifecycle epoch once more immediately
        # before commit.  The parent itself was freshly reloaded between every
        # individual write above.
        restart_proof_verification()
        run = _node(root, expected_path, label="subject-shape activation child")
        if is_supported_subject_shape_payload_receipt_profile(
            run.attrs.get(SUBJECT_SHAPE_PAYLOAD_RECEIPT_PROFILE_ATTR)
        ):
            final_proof = validate_sealed_subject_shape_publication_metadata(
                root,
                expected_path,
                expected_selector_eligible=False,
                expected_publication_owner=owner,
            )
        else:
            final_proof = _load_subject_shape_publication(
                root,
                expected_path,
                eligible=False,
                expected_owner=owner,
            )
        if final_proof.manifest.record_sha256 != proof.manifest.record_sha256:
            _fail("Subject-shape publication changed during activation.")
        # Close the post-selector proof phase while the child remains
        # ineligible. A closing recheck failure enters the ordinary rollback
        # path; no cached proof survives to the eligibility commit.
        finish_proof_verification()
        parent = fresh_parent()
        _require_activation_state(parent, snapshot, overrides=overrides)
        run = _node(root, expected_path, label="subject-shape eligibility child")
        _require_state(run, complete=True, eligible=False, expected_owner=owner)
        if defer_eligibility:
            receipt_payload = _deferred_activation_receipt_payload(
                run_name=run_name,
                owner=owner,
                manifest_sha256=proof.manifest.record_sha256,
                snapshot=snapshot,
                overrides=overrides,
                pending=pending,
                lease=lease,
            )
            return DeferredSubjectShapeCoordinateActivation(
                root=root,
                parent=parent,
                run_name=run_name,
                owner=owner,
                manifest_sha256=proof.manifest.record_sha256,
                snapshot=copy.deepcopy(snapshot),
                overrides=copy.deepcopy(overrides),
                pending=copy.deepcopy(pending),
                lease=copy.deepcopy(lease),
                receipt_sha256=_canonical_sha256(receipt_payload),
                _seal=_DEFERRED_ACTIVATION_SEAL,
            )
        # Literal final mutation.  No validation or metadata write follows.
        run.attrs["stage_selector_eligible"] = True
    except BaseException as exc:
        try:
            committed_run = _node(
                root,
                expected_path,
                label="subject-shape activation failure child",
            )
            committed = (
                committed_run.attrs.get(SUBJECT_SHAPE_PUBLICATION_OWNER_ATTR)
                == owner
                and committed_run.attrs.get("stage_selector_eligible") is True
            )
        except BaseException:
            committed = False
        # The final eligibility assignment is the last operation in the
        # transaction.  Some stores can persist it and then raise.  A fresh
        # owned/eligible read proves the attempt committed; returning success
        # prevents an outer publisher from deleting that valid publication.
        if committed:
            return
        try:
            rollback_subject_shape_activation(
                root,
                fresh_parent(),
                run,
                run_name=run_name,
                owner=owner,
                snapshot=snapshot,
                attempted_pending=pending,
                attempted_lease=lease,
            )
        except BaseException as rollback_exc:  # pragma: no cover - hostile store
            raise SubjectShapeCoordinatePublicationError(
                f"Subject-shape activation failed and rollback was incomplete: {rollback_exc}."
            ) from exc
        raise


@proof_verification_operation
def commit_deferred_subject_shape_coordinate_activation(
    activation: DeferredSubjectShapeCoordinateActivation,
    *,
    root: Any,
    parent: Any,
    run: Any,
    expected_run_attrs: Mapping[str, Any],
) -> None:
    """Rebind a deferred receipt, then expose the fresh child as the final write."""

    if (
        type(activation) is not DeferredSubjectShapeCoordinateActivation
        or activation._seal is not _DEFERRED_ACTIVATION_SEAL
    ):
        _fail("Subject-shape deferred activation receipt is invalid.")
    parent_path = "analysis/subject_shape_runs"
    expected_path = f"analysis/subject_shape_runs/{activation.run_name}"
    expected_archive = archive_identity(activation.root)
    if (
        archive_identity(root) != expected_archive
        or archive_identity(activation.parent) != expected_archive
        or archive_identity(parent) != expected_archive
        or archive_identity(run) != expected_archive
        or canonical_node_path(parent) != parent_path
        or canonical_node_path(run) != expected_path
    ):
        _fail(
            "Deferred subject-shape activation did not rebind the exact archive, "
            "parent, and child paths."
        )
    receipt_payload = _deferred_activation_receipt_payload(
        run_name=activation.run_name,
        owner=activation.owner,
        manifest_sha256=activation.manifest_sha256,
        snapshot=activation.snapshot,
        overrides=activation.overrides,
        pending=activation.pending,
        lease=activation.lease,
    )
    if (
        not isinstance(activation.receipt_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", activation.receipt_sha256) is None
        or _canonical_sha256(receipt_payload) != activation.receipt_sha256
    ):
        _fail("Deferred subject-shape activation receipt payload changed.")
    expected_pending = build_subject_shape_pending_receipt(
        run_name=activation.run_name,
        owner=activation.owner,
        manifest_sha256=activation.manifest_sha256,
        snapshot=activation.snapshot,
    )
    base_generation = _base_generation(activation.snapshot)
    expected_lease = {
        "schema_id": "palette.subject_shape_publication_lease",
        "schema_version": 1,
        "policy": SUBJECT_SHAPE_PUBLICATION_POLICY,
        "run_path": expected_path,
        "publication_owner": activation.owner,
        "owner_uuid": activation.owner,
        "base_generation": base_generation,
        "next_generation": base_generation + 1,
        "pending_receipt_sha256": _canonical_sha256(expected_pending),
    }
    expected_overrides = {
        "latest_pending": (False, None),
        SUBJECT_SHAPE_PARENT_PUBLICATION_LEASE_ATTR: (True, expected_lease),
        "latest_complete": (True, activation.run_name),
        "latest": (True, activation.run_name),
        SUBJECT_SHAPE_PUBLICATION_POLICY_ATTR: (
            True,
            SUBJECT_SHAPE_PUBLICATION_POLICY,
        ),
        SUBJECT_SHAPE_PUBLICATION_GENERATION_ATTR: (
            True,
            base_generation + 1,
        ),
    }
    if (
        dict(activation.pending) != expected_pending
        or dict(activation.lease) != expected_lease
        or dict(activation.overrides) != expected_overrides
    ):
        _fail("Deferred subject-shape activation receipt is semantically invalid.")
    expected_attrs = copy.deepcopy(dict(expected_run_attrs))
    if dict(run.attrs) != expected_attrs:
        _fail("Deferred subject-shape activation child payload changed before commit.")
    _require_state(
        run,
        complete=True,
        eligible=False,
        expected_owner=activation.owner,
    )
    _require_activation_state(
        parent,
        activation.snapshot,
        overrides=expected_overrides,
    )
    if is_supported_subject_shape_payload_receipt_profile(
        run.attrs.get(SUBJECT_SHAPE_PAYLOAD_RECEIPT_PROFILE_ATTR)
    ):
        proof = validate_sealed_subject_shape_publication_metadata(
            root,
            expected_path,
            expected_selector_eligible=False,
            expected_publication_owner=activation.owner,
        )
    else:
        proof = _load_subject_shape_publication(
            root,
            expected_path,
            eligible=False,
            expected_owner=activation.owner,
        )
    if proof.manifest.record_sha256 != activation.manifest_sha256:
        _fail("Deferred subject-shape publication changed before commit.")
    parent = _node(
        root,
        parent_path,
        label="deferred subject-shape activation parent",
    )
    if archive_identity(parent) != expected_archive:
        _fail("Deferred subject-shape activation parent changed archives/stores.")
    _require_activation_state(
        parent,
        activation.snapshot,
        overrides=expected_overrides,
    )
    run = _node(root, expected_path, label="deferred subject-shape activation child")
    if archive_identity(run) != expected_archive:
        _fail("Deferred subject-shape activation child changed archives/stores.")
    _require_state(
        run,
        complete=True,
        eligible=False,
        expected_owner=activation.owner,
    )
    if dict(run.attrs) != expected_attrs:
        _fail("Deferred subject-shape activation child payload changed during commit.")
    # The deferred commit is a separate publication operation. Recheck every
    # proof gathered above before the literal final eligibility mutation.
    finish_proof_verification()
    try:
        # Literal final mutation. No ordinary-path read or metadata write follows.
        run.attrs["stage_selector_eligible"] = True
    except BaseException:
        try:
            committed = _node(
                root,
                expected_path,
                label="deferred subject-shape committed child",
            )
            if (
                committed.attrs.get(SUBJECT_SHAPE_PUBLICATION_OWNER_ATTR)
                == activation.owner
                and committed.attrs.get("stage_selector_eligible") is True
                and dict(committed.attrs)
                == {
                    **expected_attrs,
                    "stage_selector_eligible": True,
                }
            ):
                return
        except BaseException:
            pass
        raise


def rollback_deferred_subject_shape_coordinate_activation(
    activation: DeferredSubjectShapeCoordinateActivation,
) -> None:
    """Rollback only the selector epoch proven by one deferred receipt."""

    if (
        type(activation) is not DeferredSubjectShapeCoordinateActivation
        or activation._seal is not _DEFERRED_ACTIVATION_SEAL
    ):
        _fail("Subject-shape deferred activation receipt is invalid.")
    run = _node(
        activation.root,
        f"analysis/subject_shape_runs/{activation.run_name}",
        label="deferred subject-shape rollback child",
    )
    if run.attrs.get(SUBJECT_SHAPE_PUBLICATION_OWNER_ATTR) != activation.owner:
        return
    rollback_subject_shape_activation(
        activation.root,
        activation.parent,
        run,
        run_name=activation.run_name,
        owner=activation.owner,
        snapshot=activation.snapshot,
        attempted_pending=activation.pending,
        attempted_lease=activation.lease,
    )


__all__ = [
    "CANONICAL_SUBJECT_SHAPE_BUNDLE_METHOD",
    "CANONICAL_SUBJECT_SHAPE_BUNDLE_METHOD_VERSION",
    "CANONICAL_SUBJECT_SHAPE_BUNDLE_PROFILE_ID",
    "CANONICAL_SUBJECT_SHAPE_BUNDLE_RUN_SCHEMA_VERSION",
    "CANONICAL_SUBJECT_SHAPE_COMPONENT_ORDER",
    "CANONICAL_SUBJECT_SHAPE_METHOD",
    "CANONICAL_SUBJECT_SHAPE_METHOD_VERSION",
    "CANONICAL_SUBJECT_SHAPE_PROFILE_ID",
    "CANONICAL_SUBJECT_SHAPE_RELATION_ORDER",
    "CANONICAL_SUBJECT_SHAPE_ROW_INDEX_ARRAYS",
    "CANONICAL_SUBJECT_SHAPE_ROW_LINEAGE_MISSING",
    "CANONICAL_SUBJECT_SHAPE_RUN_SCHEMA_ID",
    "CANONICAL_SUBJECT_SHAPE_RUN_SCHEMA_VERSION",
    "SUBJECT_SHAPE_COMPONENT_SCHEMA_ATTR",
    "SUBJECT_SHAPE_BOUND_CANONICAL_STATUS",
    "SUBJECT_SHAPE_BUNDLE_ACTIVE_AT_DERIVATION_ATTR",
    "SUBJECT_SHAPE_BUNDLE_ID_ATTR",
    "SUBJECT_SHAPE_BUNDLE_SOURCE_KIND",
    "SUBJECT_SHAPE_COMPUTING_UNBOUND_STATUS",
    "SUBJECT_SHAPE_COORDINATE_BINDING_STATUS_ATTR",
    "SUBJECT_SHAPE_COORDINATE_CONTRACT",
    "SUBJECT_SHAPE_CONSUMED_UNBOUND_STAGE_ATTR",
    "SUBJECT_SHAPE_DERIVATION_ATTR",
    "SUBJECT_SHAPE_DEFERRED_STORAGE_RECEIPT_ATTR",
    "SUBJECT_SHAPE_HEADING_SEMANTICS_ATTR",
    "SUBJECT_SHAPE_MANIFEST_ATTR",
    "SUBJECT_SHAPE_PARENT_PUBLICATION_LEASE_ATTR",
    "SUBJECT_SHAPE_PAYLOAD_INTEGRITY_RECEIPT_ATTR",
    "SUBJECT_SHAPE_PAYLOAD_RECEIPT_PROFILE",
    "SUBJECT_SHAPE_PAYLOAD_RECEIPT_PROFILE_V1",
    "SUBJECT_SHAPE_PAYLOAD_RECEIPT_PROFILE_V2",
    "SUBJECT_SHAPE_PAYLOAD_RECEIPT_PROFILES",
    "SUBJECT_SHAPE_PAYLOAD_RECEIPT_PROFILE_ATTR",
    "SUBJECT_SHAPE_PAYLOAD_VALIDATION_RECEIPT_ATTR",
    "SUBJECT_SHAPE_PAYLOAD_VALIDATOR_SCHEMA_ID",
    "SUBJECT_SHAPE_PAYLOAD_VALIDATOR_SCHEMA_VERSION",
    "SUBJECT_SHAPE_NUMERIC_PROJECTION_ATTR",
    "SUBJECT_SHAPE_NUMERIC_PROJECTION_DIGEST_ATTR",
    "SUBJECT_SHAPE_NUMERIC_PROJECTION_SCHEMA_ID",
    "SUBJECT_SHAPE_PROJECTED_UNBOUND_BINDING_STATUS",
    "SUBJECT_SHAPE_PROJECTED_UNBOUND_MANIFEST_SCHEMA_VERSION",
    "SUBJECT_SHAPE_PUBLICATION_GENERATION_ATTR",
    "SUBJECT_SHAPE_PUBLICATION_OWNER_ATTR",
    "SUBJECT_SHAPE_PUBLICATION_POLICY_ATTR",
    "SUBJECT_SHAPE_PUBLISHING_BINDING_STATUS",
    "SUBJECT_SHAPE_SCALAR_SURFACE_ATTR",
    "SUBJECT_SHAPE_SCALAR_SURFACE_INVENTORY_ATTR",
    "SUBJECT_SHAPE_SCIENTIFIC_CONFIGURATION_ATTR",
    "SUBJECT_SHAPE_SOURCE_BINDING_ATTR",
    "SUBJECT_SHAPE_SOURCE_BINDING_DIGEST_ATTR",
    "SUBJECT_SHAPE_SOURCE_KIND_ATTR",
    "SUBJECT_SHAPE_HISTORICAL_SOURCE_KIND",
    "SUBJECT_SHAPE_STORAGE_ARRAY_ATTRS",
    "SUBJECT_SHAPE_STORAGE_CANDIDATE_ATTR",
    "SUBJECT_SHAPE_STORAGE_METADATA_POLICY_ATTR",
    "SUBJECT_SHAPE_STORAGE_PLAN_ATTR",
    "SUBJECT_SHAPE_STORAGE_PLAN_DIGEST_ATTR",
    "SUBJECT_SHAPE_STORAGE_PROFILE_ID_ATTR",
    "SUBJECT_SHAPE_STORAGE_PROFILE_ROLE",
    "SUBJECT_SHAPE_STORAGE_PROFILE_ROLE_ATTR",
    "SUBJECT_SHAPE_SUPPORTED_STORAGE_PROFILE_ROLE",
    "SUBJECT_SHAPE_ACCESS_AWARE_SUPPORTED_PROFILE_ID",
    "SUBJECT_SHAPE_STORAGE_SOURCE_MANIFEST_ATTR",
    "SUBJECT_SHAPE_STORAGE_SOURCE_MANIFEST_SCHEMA_ID",
    "SUBJECT_SHAPE_TAIL_SAMPLE_AXIS_ATTR",
    "SUBJECT_SHAPE_UNBOUND_MANIFEST_ATTR",
    "SUBJECT_SHAPE_UNBOUND_MANIFEST_SCHEMA_ID",
    "SUBJECT_SHAPE_UNBOUND_STAGE_STATUS",
    "BoundSubjectShapeScalarSurface",
    "BoundSubjectShapeCoordinatePublication",
    "DeferredSubjectShapeCoordinateActivation",
    "SealedSubjectShapePublicationMetadataProof",
    "SubjectShapeCoordinateSource",
    "SubjectShapeCoordinatePublicationError",
    "activate_subject_shape_coordinate_publication",
    "build_subject_shape_schema_inventory_record",
    "build_subject_shape_scientific_configuration_record",
    "build_subject_shape_source_camera_numeric_projection_record",
    "build_subject_shape_unbound_numeric_manifest_record",
    "build_subject_shape_pending_receipt",
    "commit_deferred_subject_shape_coordinate_activation",
    "deep_audit_subject_shape_payload_receipt",
    "derive_canonical_subject_shape_body_frame",
    "load_completed_ineligible_subject_shape_coordinate_publication",
    "load_exact_subject_shape_source",
    "load_exact_subject_shape_refined_source",
    "load_subject_shape_consumed_unbound_stage",
    "load_subject_shape_temporal_authority",
    "load_persisted_subject_shape_coordinate_publication",
    "load_sealed_unbound_subject_shape_manifest",
    "load_unbound_subject_shape_manifest",
    "is_supported_subject_shape_payload_receipt_profile",
    "prepare_subject_shape_identity_and_schema",
    "project_subject_shape_bboxes",
    "project_subject_shape_ellipses",
    "project_subject_shape_points",
    "publish_subject_shape_coordinate_surfaces",
    "require_translation_only_refined_placement",
    "require_translation_only_subject_shape_placement",
    "require_subject_shape_source_camera_numeric_projection",
    "rollback_subject_shape_activation",
    "rollback_deferred_subject_shape_coordinate_activation",
    "selector_snapshot",
    "subject_shape_maintained_profile_record",
    "subject_shape_bundle_maintained_profile_record",
    "stamp_subject_shape_derivation",
    "stamp_subject_shape_source_camera_numeric_projection",
    "stamp_subject_shape_temporal_authority",
    "stamp_unbound_subject_shape_manifest",
    "transform_subject_shape_geometry_to_source_camera",
    "validate_sealed_subject_shape_publication_metadata",
    "mask_invalid_subject_shape_vectors",
]
