"""Fail-closed publication boundary for canonical coordinate descriptors.

``coordinate_descriptor`` defines the compact persisted schema and its pure
parser.  Pure mappings cannot prove that referenced records exist.  This module
is the writer/reader boundary: it derives descriptors only from sealed evidence
loaded from the exact persisted identity, extent, frame, and transform nodes,
and it revalidates that evidence immediately before every attrs write or read.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from fisheye.shared.archive_identity import (
    ArchiveIdentity,
    ArchiveIdentityError,
    archive_identity,
)
from fisheye.shared.coordinate_descriptor import (
    CANONICAL_COORDINATE_PROFILES,
    CANONICAL_OVERLAY_DIRECT,
    CANONICAL_OVERLAY_NOT_SUITABLE,
    CANONICAL_OVERLAY_REQUIRES_TRANSFORM,
    COORDINATE_DESCRIPTOR_ATTR,
    CanonicalCoordinateDescriptor,
    CanonicalFrameRecord,
    CoordinateDescriptorError,
    CoordinateIssue,
    DigestBoundCoordinateRecordRef,
    build_canonical_coordinate_descriptor,
    canonical_coordinate_descriptor_v2_attrs,
    load_canonical_coordinate_descriptor_attrs,
    parse_canonical_coordinate_descriptor,
    verify_canonical_coordinate_descriptor_identity,
)
from fisheye.shared.coordinate_identity import (
    BoundRowIdentityContract,
    RowIdentityContractError,
    require_bound_row_identity_contract,
)
from fisheye.shared.coordinate_reference import (
    BoundReferenceExtent,
    CoordinateReferenceError,
    canonical_node_path,
    verify_bound_reference_extent,
)
from fisheye.shared.coordinate_record import (
    BoundCoordinateRecord,
    CoordinateRecordError,
    verify_bound_coordinate_record,
)
from fisheye.shared.directed_transform_chain import (
    BoundDirectedTransformChain,
    DirectedTransformChainError,
    require_bound_directed_transform_chain,
)
from fisheye.shared.pixel_frame_authority import (
    ARENA_RELATIVE_CANVAS_FRAME_KIND,
    DETECTOR_NORMALIZED_FRAME_KIND,
    MODEL_INPUT_FRAME_KIND,
    ROI_FRAME_KIND,
    SELECTED_CANVAS_FRAME_KIND,
    SOURCE_CAMERA_FRAME_KIND,
    SOURCE_CAMERA_NORMALIZED_FRAME_KIND,
    BoundPixelFrameAuthority,
    PixelFrameAuthorityError,
    require_bound_pixel_frame_authority,
)


_BOUND_COORDINATE_DESCRIPTOR_SEAL = object()
_OWNER_DTYPE_ATTR_SUFFIX = "_owner_dtype"

_TYPED_PIXEL_REFERENCE_KIND_BY_SPACE = {
    "source_camera_image_px": SOURCE_CAMERA_FRAME_KIND,
    "source_camera_normalized_xy": SOURCE_CAMERA_NORMALIZED_FRAME_KIND,
    "detector_model_input_px": MODEL_INPUT_FRAME_KIND,
    "detector_normalized_xy": DETECTOR_NORMALIZED_FRAME_KIND,
    "roi_local_px": ROI_FRAME_KIND,
    "stimulus_canvas_px": SELECTED_CANVAS_FRAME_KIND,
    "arena_relative_canvas_px": ARENA_RELATIVE_CANVAS_FRAME_KIND,
}


class CanonicalCoordinatePublicationError(CoordinateDescriptorError):
    """Raised when persisted evidence cannot authorize a coordinate surface."""


def _fail(code: str, path: str, message: str) -> None:
    raise CanonicalCoordinatePublicationError(
        (CoordinateIssue(code=code, path=path, message=message),)
    )


def _exact_value_equal(left: Any, right: Any) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return (
            type(left) is type(right)
            and set(left) == set(right)
            and all(_exact_value_equal(left[name], right[name]) for name in left)
        )
    if isinstance(left, (list, tuple)) and type(left) is type(right):
        return len(left) == len(right) and all(
            _exact_value_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return (
            type(left) is type(right)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and np.array_equal(left, right, equal_nan=True)
        )
    return type(left) is type(right) and bool(left == right)


def _node_shape(node: Any) -> tuple[int, ...]:
    raw = getattr(node, "shape", None)
    if not isinstance(raw, (tuple, list)):
        _fail(
            "coordinate_owner_shape_invalid",
            "$.coordinate_node.shape",
            "Coordinate array must expose an exact integer shape.",
        )
    shape: list[int] = []
    for index, item in enumerate(raw):
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            _fail(
                "coordinate_owner_shape_invalid",
                f"$.coordinate_node.shape[{index}]",
                "Coordinate dimensions must be exact nonnegative integers.",
            )
        shape.append(item)
    return tuple(shape)


def _node_numeric_dtype(node: Any) -> str:
    try:
        dtype = np.dtype(getattr(node, "dtype"))
    except (AttributeError, TypeError) as exc:
        _fail(
            "coordinate_owner_dtype_invalid",
            "$.coordinate_node.dtype",
            f"Coordinate array must expose one exact NumPy dtype: {exc}.",
        )
    if dtype.kind not in {"i", "u", "f"}:
        _fail(
            "coordinate_owner_dtype_nonnumeric",
            "$.coordinate_node.dtype",
            "Coordinate geometry must use an integer or floating dtype; bool, object, bytes, Unicode, complex, and structured dtypes are forbidden.",
        )
    return dtype.str


def _require_coordinate_under_rowset(
    coordinate_node: Any,
    identity: BoundRowIdentityContract,
) -> str:
    path = canonical_node_path(coordinate_node)
    prefix = f"{identity.rowset_path}/"
    if not path.startswith(prefix) or path == identity.key_array_path:
        _fail(
            "coordinate_rowset_path_mismatch",
            "$.coordinate_node.path",
            "Coordinate array must be a non-key descendant of the exact identity rowset.",
        )
    return path


def _same_reference_extent(
    left: BoundReferenceExtent,
    right: BoundReferenceExtent,
) -> bool:
    return (
        left.archive_identity == right.archive_identity
        and left.record_ref == right.record_ref
        and left.record_sha256 == right.record_sha256
        and left.selector == right.selector
        and left.width == right.width
        and left.height == right.height
        and left.units == right.units
    )


def _same_pixel_frame(
    left: BoundPixelFrameAuthority,
    right: BoundPixelFrameAuthority,
) -> bool:
    return (
        left.archive_identity == right.archive_identity
        and left.record_ref == right.record_ref
        and left.record_sha256 == right.record_sha256
        and left.endpoint == right.endpoint
    )


def _same_row_identity(
    left: BoundRowIdentityContract,
    right: BoundRowIdentityContract,
) -> bool:
    return (
        left.archive_identity == right.archive_identity
        and left.record_ref == right.record_ref
        and left.record_sha256 == right.record_sha256
        and left.leading_dimension == right.leading_dimension
        and left.rowset_path == right.rowset_path
        and left.key_array_path == right.key_array_path
        and left.contract.domain == right.contract.domain
        and left.contract.mode == right.contract.mode
    )


def _frame_evidence(
    value: Any,
    *,
    expected_kind: str,
) -> tuple[Any, CanonicalFrameRecord]:
    try:
        from fisheye.shared.coordinate_frame_record import (
            verify_bound_coordinate_frame,
        )

        bound = verify_bound_coordinate_frame(value, expected_kind=expected_kind)
    except (ImportError, ValueError, TypeError) as exc:
        _fail(
            "coordinate_frame_record_unverified",
            "$.frame_record",
            f"Exact persisted frame record is required: {exc}.",
        )
    return bound, CanonicalFrameRecord(
        kind=bound.kind,
        record_ref=bound.record_ref,
        record_sha256=bound.record_sha256,
    )


@dataclass(frozen=True, init=False)
class BoundCanonicalCoordinateDescriptor:
    """Sealed descriptor plus every exact node required to revalidate it."""

    descriptor: CanonicalCoordinateDescriptor
    coordinate_node: Any = field(repr=False, compare=False)
    row_identity: BoundRowIdentityContract = field(repr=False, compare=False)
    reference_extent: BoundReferenceExtent | None = field(
        repr=False,
        compare=False,
    )
    reference_frame_authority: BoundPixelFrameAuthority | None = field(
        repr=False,
        compare=False,
    )
    transform_chain: BoundDirectedTransformChain | None = field(
        repr=False,
        compare=False,
    )
    lineage_records: tuple[BoundCoordinateRecord, ...] = field(
        repr=False,
        compare=False,
    )
    frame_record: Any = field(repr=False, compare=False)
    owner_dtype: str = field(repr=False, compare=False)
    _archive_identity: ArchiveIdentity = field(repr=False, compare=False)
    _publication_seal: object = field(repr=False, compare=False)

    def __init__(
        self,
        *,
        descriptor: CanonicalCoordinateDescriptor,
        coordinate_node: Any,
        row_identity: BoundRowIdentityContract,
        reference_extent: BoundReferenceExtent | None,
        reference_frame_authority: BoundPixelFrameAuthority | None,
        transform_chain: BoundDirectedTransformChain | None,
        lineage_records: tuple[BoundCoordinateRecord, ...],
        frame_record: Any,
        owner_dtype: str,
        archive: ArchiveIdentity,
        _seal: object | None = None,
    ) -> None:
        if _seal is not _BOUND_COORDINATE_DESCRIPTOR_SEAL:
            _fail(
                "coordinate_descriptor_binding_unverified",
                "$",
                "Bound coordinate descriptors must be produced by the exact-evidence builder.",
            )
        object.__setattr__(self, "descriptor", descriptor)
        object.__setattr__(self, "coordinate_node", coordinate_node)
        object.__setattr__(self, "row_identity", row_identity)
        object.__setattr__(self, "reference_extent", reference_extent)
        object.__setattr__(
            self,
            "reference_frame_authority",
            reference_frame_authority,
        )
        object.__setattr__(self, "transform_chain", transform_chain)
        object.__setattr__(self, "lineage_records", lineage_records)
        object.__setattr__(self, "frame_record", frame_record)
        object.__setattr__(self, "owner_dtype", owner_dtype)
        object.__setattr__(self, "_archive_identity", archive)
        object.__setattr__(self, "_publication_seal", _seal)


def _verify_bound_canonical_coordinate_descriptor(
    value: Any,
) -> BoundCanonicalCoordinateDescriptor:
    if (
        type(value) is not BoundCanonicalCoordinateDescriptor
        or getattr(value, "_publication_seal", None)
        is not _BOUND_COORDINATE_DESCRIPTOR_SEAL
    ):
        _fail(
            "coordinate_descriptor_binding_unverified",
            "$",
            "A sealed exact-evidence coordinate descriptor is required.",
        )
    bound = value
    try:
        coordinate_archive = archive_identity(bound.coordinate_node)
    except ArchiveIdentityError as exc:
        _fail("coordinate_archive_unverified", "$.coordinate_node", str(exc))
    if coordinate_archive != bound._archive_identity:
        _fail(
            "coordinate_archive_mismatch",
            "$.coordinate_node",
            "Coordinate surface moved to a different archive/store.",
        )
    descriptor = parse_canonical_coordinate_descriptor(bound.descriptor)
    try:
        verified_identity = require_bound_row_identity_contract(bound.row_identity)
    except RowIdentityContractError as exc:
        _fail(
            "coordinate_row_identity_unverified",
            "$.row_identity",
            str(exc),
        )
    if verified_identity is not bound.row_identity:
        _fail(
            "coordinate_row_identity_unverified",
            "$.row_identity",
            "Coordinate row identity was not returned by the central sealed gate.",
        )
    if bound.row_identity.archive_identity != coordinate_archive:
        _fail(
            "coordinate_archive_mismatch",
            "$.row_identity",
            "Coordinate surface and row identity come from different archives/stores.",
        )
    _require_coordinate_under_rowset(bound.coordinate_node, bound.row_identity)
    shape = _node_shape(bound.coordinate_node)
    current_dtype = _node_numeric_dtype(bound.coordinate_node)
    if current_dtype != bound.owner_dtype:
        _fail(
            "coordinate_owner_dtype_changed",
            "$.coordinate_node.dtype",
            "Coordinate array dtype changed after the descriptor was bound.",
        )
    try:
        verify_canonical_coordinate_descriptor_identity(
            descriptor,
            row_identity_contract=bound.row_identity.contract,
            expected_row_identity_record_ref=bound.row_identity.record_ref,
            owner_shape=shape,
        )
    except CoordinateDescriptorError as exc:
        raise CanonicalCoordinatePublicationError(exc.issues) from exc

    profile = CANONICAL_COORDINATE_PROFILES[descriptor.profile_id]
    if profile.publication_status != "available":
        _fail(
            "coordinate_profile_publication_unavailable",
            "$.profile_id",
            "This profile is reserved but has no exact typed lineage authority for canonical publication.",
        )
    if (
        descriptor.source_camera_overlay.status
        == CANONICAL_OVERLAY_REQUIRES_TRANSFORM
        and profile.frame_record_kind != "pixel_frame_authority"
    ):
        _fail(
            "coordinate_nonpixel_overlay_transform_unsupported",
            "$.source_camera_overlay",
            "Physical/body-frame overlays fail closed until generalized typed transform endpoints are implemented.",
        )
    expected_frame: CanonicalFrameRecord | None = None
    expected_pixel_kind = _TYPED_PIXEL_REFERENCE_KIND_BY_SPACE.get(profile.space_id)
    if (
        profile.frame_record_kind == "pixel_frame_authority"
        and expected_pixel_kind is None
    ):
        _fail(
            "coordinate_pixel_frame_space_unsupported",
            "$.space_id",
            "No sealed pixel-frame authority subtype exists yet for this controlled space.",
        )
    if expected_pixel_kind is not None:
        if (
            bound.frame_record is not None
            or bound.reference_extent is not None
            or bound.reference_frame_authority is None
        ):
            _fail(
                "coordinate_pixel_frame_evidence_mismatch",
                "$.reference_extent",
                "This controlled pixel/normalized profile requires its exact sealed pixel-frame authority; a generic extent cannot substitute.",
            )
        try:
            pixel_frame = require_bound_pixel_frame_authority(
                bound.reference_frame_authority,
                expected_kind=expected_pixel_kind,
            )
        except PixelFrameAuthorityError as exc:
            _fail(
                "coordinate_pixel_frame_evidence_unverified",
                "$.reference_extent",
                str(exc),
            )
        if pixel_frame.archive_identity != coordinate_archive:
            _fail(
                "coordinate_archive_mismatch",
                "$.reference_extent",
                "Coordinate surface and pixel-frame authority come from different archives/stores.",
            )
        if pixel_frame.row_identity is not None and not _same_row_identity(
            pixel_frame.row_identity,
            bound.row_identity,
        ):
            _fail(
                "coordinate_pixel_frame_row_identity_mismatch",
                "$.row_identity",
                "Rowwise pixel-frame authority does not equal the coordinate surface identity.",
            )
        if (
            descriptor.space_id == pixel_frame.space_id
            and descriptor.pixel_convention != pixel_frame.pixel_convention
        ):
            _fail(
                "coordinate_pixel_frame_convention_mismatch",
                "$.pixel_convention",
                "Descriptor convention does not match its exact pixel-frame authority.",
            )
        reference_ref = pixel_frame.record_ref
        reference_sha256 = pixel_frame.record_sha256
        reference_selector = "record"
        reference_width = pixel_frame.endpoint.width
        reference_height = pixel_frame.endpoint.height
        # Coordinate endpoint units and reference-extent units are distinct for
        # normalized frames: coordinates are normalized, while W/H still name
        # an exact pixel reference extent.
        reference_units = pixel_frame.reference_extent.units
        if profile.frame_record_kind is not None:
            if profile.frame_record_kind != "pixel_frame_authority":
                _fail(
                    "coordinate_pixel_frame_kind_mismatch",
                    "$.frame_record.kind",
                    "Controlled pixel profiles must require the pixel-frame authority record kind.",
                )
            expected_frame = CanonicalFrameRecord(
                kind="pixel_frame_authority",
                record_ref=pixel_frame.record_ref,
                record_sha256=pixel_frame.record_sha256,
            )
    elif profile.frame_record_kind is None:
        if (
            bound.frame_record is not None
            or bound.reference_frame_authority is not None
            or bound.reference_extent is None
        ):
            _fail(
                "coordinate_reference_evidence_mismatch",
                "$.reference_extent",
                "This profile requires one exact non-frame reference authority.",
            )
        try:
            reference = verify_bound_reference_extent(bound.reference_extent)
        except CoordinateReferenceError as exc:
            _fail(
                "coordinate_reference_evidence_unverified",
                "$.reference_extent",
                str(exc),
            )
        if reference.archive_identity != coordinate_archive:
            _fail(
                "coordinate_archive_mismatch",
                "$.reference_extent",
                "Coordinate surface and extent authority come from different archives/stores.",
            )
        reference_ref = reference.record_ref
        reference_sha256 = reference.record_sha256
        reference_selector = reference.selector
        reference_width = reference.width
        reference_height = reference.height
        reference_units = reference.units
    else:
        if (
            bound.reference_extent is not None
            or bound.reference_frame_authority is not None
            or bound.frame_record is None
        ):
            _fail(
                "coordinate_frame_evidence_mismatch",
                "$.frame_record",
                "Frame profiles require one exact bound frame record and no substitute authority.",
            )
        frame, expected_frame = _frame_evidence(
            bound.frame_record,
            expected_kind=profile.frame_record_kind,
        )
        if getattr(frame, "archive_identity", None) != coordinate_archive:
            _fail(
                "coordinate_archive_mismatch",
                "$.frame_record",
                "Coordinate surface and frame authority come from different archives/stores.",
            )
        compatible_profiles = getattr(frame, "compatible_profile_ids", None)
        if compatible_profiles is not None and descriptor.profile_id not in tuple(
            compatible_profiles
        ):
            _fail(
                "coordinate_frame_profile_incompatible",
                "$.profile_id",
                "Frame authority does not explicitly authorize this coordinate profile.",
            )
        frame_units = getattr(frame, "coordinate_units", None)
        unit_vector_profile = (
            descriptor.profile_id == "fish_anatomical_body_frame.unit_vector.v1"
        )
        units_match = (
            frame_units in {"px", "mm"}
            if unit_vector_profile
            else frame_units == profile.coordinate_unit
        )
        if not units_match:
            _fail(
                "coordinate_frame_units_mismatch",
                "$.component_units",
                "Frame spatial basis is incompatible with the controlled profile units.",
            )
        if (
            getattr(frame, "positive_x", None) != profile.positive_x
            or getattr(frame, "positive_y", None) != profile.positive_y
        ):
            _fail(
                "coordinate_frame_axes_mismatch",
                "$.positive_directions",
                "Frame positive-axis directions do not match the controlled profile.",
            )
        if getattr(frame, "origin", None) != profile.origin:
            _fail(
                "coordinate_frame_origin_mismatch",
                "$.origin",
                "Frame origin does not match the controlled coordinate profile.",
            )
        frame_identity = getattr(frame, "row_identity", None)
        if frame_identity is not None and not _same_row_identity(
            frame_identity,
            bound.row_identity,
        ):
            _fail(
                "coordinate_frame_row_identity_mismatch",
                "$.row_identity",
                "Rowwise frame authority does not equal the coordinate surface identity.",
            )
        reference_ref = frame.record_ref
        reference_sha256 = frame.record_sha256
        reference_selector = frame.selector
        reference_width = frame.reference_width
        reference_height = frame.reference_height
        reference_units = frame.reference_units

    extent = descriptor.reference_extent
    if (
        extent.authority.record_ref != reference_ref
        or extent.authority.record_sha256 != reference_sha256
        or extent.authority.selector != reference_selector
        or extent.width != reference_width
        or extent.height != reference_height
        or extent.units != reference_units
    ):
        _fail(
            "coordinate_reference_evidence_mismatch",
            "$.reference_extent",
            "Descriptor extent does not equal the exact resolved authority.",
        )
    if descriptor.frame_record != expected_frame:
        _fail(
            "coordinate_frame_evidence_mismatch",
            "$.frame_record",
            "Descriptor frame record does not equal the exact resolved frame.",
        )

    verified_lineage: list[BoundCoordinateRecord] = []
    for index, record in enumerate(bound.lineage_records):
        try:
            current = verify_bound_coordinate_record(record)
        except CoordinateRecordError as exc:
            _fail(
                "coordinate_lineage_unverified",
                f"$.lineage_records[{index}]",
                str(exc),
            )
        if current.archive_identity != coordinate_archive:
            _fail(
                "coordinate_archive_mismatch",
                f"$.lineage_records[{index}]",
                "Coordinate lineage comes from a different archive/store.",
            )
        verified_lineage.append(current)
    expected_lineage = [(reference_ref, reference_sha256)]
    for record in verified_lineage:
        pair = (record.record_ref, record.record_sha256)
        if pair not in expected_lineage:
            expected_lineage.append(pair)
    actual_lineage = [
        (record.record_ref, record.record_sha256)
        for record in descriptor.lineage_refs
    ]
    if actual_lineage != expected_lineage:
        _fail(
            "coordinate_lineage_mismatch",
            "$.lineage_refs",
            "Descriptor lineage does not equal the exact resolved persisted records.",
        )

    overlay = descriptor.source_camera_overlay
    if overlay.status == CANONICAL_OVERLAY_DIRECT:
        if bound.transform_chain is not None:
            _fail(
                "coordinate_overlay_evidence_unexpected",
                "$.source_camera_overlay",
                "Direct source-camera coordinates cannot carry a transform chain.",
            )
    elif overlay.status == CANONICAL_OVERLAY_NOT_SUITABLE:
        if bound.transform_chain is not None:
            _fail(
                "coordinate_overlay_evidence_unexpected",
                "$.source_camera_overlay",
                "Not-suitable coordinates cannot claim a source-camera transform.",
            )
    elif overlay.status == CANONICAL_OVERLAY_REQUIRES_TRANSFORM:
        if (
            bound.transform_chain is None
            or bound.reference_frame_authority is None
        ):
            _fail(
                "coordinate_transform_evidence_missing",
                "$.source_camera_overlay",
                "Transform-required overlay needs an exact chain and typed descriptor-frame authority.",
            )
        try:
            chain = require_bound_directed_transform_chain(bound.transform_chain)
            descriptor_reference = require_bound_pixel_frame_authority(
                bound.reference_frame_authority
            )
        except (DirectedTransformChainError, PixelFrameAuthorityError) as exc:
            _fail(
                "coordinate_transform_evidence_unverified",
                "$.source_camera_overlay",
                str(exc),
            )
        if (
            chain.archive_identity != coordinate_archive
            or descriptor_reference.archive_identity != coordinate_archive
        ):
            _fail(
                "coordinate_archive_mismatch",
                "$.source_camera_overlay",
                "Coordinate transform evidence comes from different archives/stores.",
            )
        if chain.row_identity is not None and not _same_row_identity(
            chain.row_identity,
            bound.row_identity,
        ):
            _fail(
                "coordinate_transform_row_identity_mismatch",
                "$.source_camera_overlay",
                "Rowwise transform identity does not equal the coordinate surface identity.",
            )
        if chain.descriptor_space_id != descriptor.space_id:
            _fail(
                "coordinate_transform_start_mismatch",
                "$.source_camera_overlay",
                "Transform chain does not start in the descriptor space.",
            )
        if not _same_pixel_frame(
            chain.descriptor_frame_authority,
            descriptor_reference,
        ):
            _fail(
                "coordinate_transform_source_frame_mismatch",
                "$.source_camera_overlay",
                "Transform-chain source frame is not the descriptor authority.",
            )
        if descriptor.pixel_convention != chain.descriptor_pixel_convention:
            _fail(
                "coordinate_transform_pixel_convention_mismatch",
                "$.pixel_convention",
                "Descriptor convention does not match the transform source frame.",
            )
        transform_refs = tuple(
            (item.record_ref, item.record_sha256)
            for item in chain.transform_records
        )
        descriptor_refs = tuple(
            (item.record_ref, item.record_sha256)
            for item in overlay.transform_refs
        )
        if descriptor_refs != transform_refs:
            _fail(
                "coordinate_transform_record_mismatch",
                "$.source_camera_overlay.transform_refs",
                "Descriptor transform refs are not the exact resolved ordered chain.",
            )
    return bound


def build_bound_canonical_coordinate_descriptor(
    coordinate_node: Any,
    *,
    profile_id: str,
    geometry_type: str,
    components: Sequence[str],
    component_units: Sequence[str],
    pixel_convention: str,
    row_identity: BoundRowIdentityContract,
    reference_extent: BoundReferenceExtent | None = None,
    reference_frame_authority: BoundPixelFrameAuthority | None = None,
    source_camera_overlay_status: str,
    transform_chain: BoundDirectedTransformChain | None = None,
    lineage_records: Sequence[BoundCoordinateRecord] = (),
    frame_record: Any = None,
) -> BoundCanonicalCoordinateDescriptor:
    """Build one descriptor solely from exact persisted evidence objects."""

    try:
        row_identity = require_bound_row_identity_contract(row_identity)
    except RowIdentityContractError as exc:
        _fail("coordinate_row_identity_unverified", "$.row_identity", str(exc))
    _require_coordinate_under_rowset(coordinate_node, row_identity)
    owner_dtype = _node_numeric_dtype(coordinate_node)
    try:
        coordinate_archive = archive_identity(coordinate_node)
    except ArchiveIdentityError as exc:
        _fail("coordinate_archive_unverified", "$.coordinate_node", str(exc))
    if row_identity.archive_identity != coordinate_archive:
        _fail(
            "coordinate_archive_mismatch",
            "$.row_identity",
            "Coordinate surface and row identity come from different archives/stores.",
        )
    profile = CANONICAL_COORDINATE_PROFILES.get(profile_id)
    if profile is None:
        _fail("profile_id_unsupported", "$.profile_id", f"Unsupported profile {profile_id!r}.")
    if profile.publication_status != "available":
        _fail(
            "coordinate_profile_publication_unavailable",
            "$.profile_id",
            "This profile is reserved but has no exact typed lineage authority for canonical publication.",
        )

    if (
        source_camera_overlay_status == CANONICAL_OVERLAY_REQUIRES_TRANSFORM
        and profile.frame_record_kind != "pixel_frame_authority"
    ):
        _fail(
            "coordinate_nonpixel_overlay_transform_unsupported",
            "$.source_camera_overlay",
            "Physical/body-frame overlays fail closed until generalized typed transform endpoints are implemented.",
        )

    chain: BoundDirectedTransformChain | None = None
    if source_camera_overlay_status == CANONICAL_OVERLAY_REQUIRES_TRANSFORM:
        if transform_chain is None:
            _fail(
                "coordinate_transform_evidence_missing",
                "$.source_camera_overlay",
                "Transform-required overlay needs a sealed typed-endpoint chain.",
            )
        try:
            chain = require_bound_directed_transform_chain(transform_chain)
        except DirectedTransformChainError as exc:
            _fail(
                "coordinate_transform_evidence_unverified",
                "$.source_camera_overlay",
                str(exc),
            )
        if reference_frame_authority is None:
            reference_frame_authority = chain.descriptor_frame_authority

    frame_descriptor: CanonicalFrameRecord | None = None
    expected_pixel_kind = _TYPED_PIXEL_REFERENCE_KIND_BY_SPACE.get(profile.space_id)
    if (
        profile.frame_record_kind == "pixel_frame_authority"
        and expected_pixel_kind is None
    ):
        _fail(
            "coordinate_pixel_frame_space_unsupported",
            "$.space_id",
            "No sealed pixel-frame authority subtype exists yet for this controlled space.",
        )
    if expected_pixel_kind is not None:
        if (
            frame_record is not None
            or reference_extent is not None
            or reference_frame_authority is None
        ):
            _fail(
                "coordinate_pixel_frame_evidence_mismatch",
                "$.reference_extent",
                "This controlled profile requires its exact sealed pixel-frame authority; a generic extent cannot substitute.",
            )
        try:
            reference_frame_authority = require_bound_pixel_frame_authority(
                reference_frame_authority,
                expected_kind=expected_pixel_kind,
            )
        except PixelFrameAuthorityError as exc:
            _fail(
                "coordinate_pixel_frame_evidence_unverified",
                "$.reference_extent",
                str(exc),
            )
        width = reference_frame_authority.endpoint.width
        height = reference_frame_authority.endpoint.height
        units = reference_frame_authority.reference_extent.units
        authority_ref = reference_frame_authority.record_ref
        authority_digest = reference_frame_authority.record_sha256
        selector = "record"
        if profile.frame_record_kind is not None:
            if profile.frame_record_kind != "pixel_frame_authority":
                _fail(
                    "coordinate_pixel_frame_kind_mismatch",
                    "$.frame_record.kind",
                    "Controlled pixel profiles must require the pixel-frame authority record kind.",
                )
            frame_descriptor = CanonicalFrameRecord(
                kind="pixel_frame_authority",
                record_ref=reference_frame_authority.record_ref,
                record_sha256=reference_frame_authority.record_sha256,
            )
    elif profile.frame_record_kind is None:
        if (
            frame_record is not None
            or reference_frame_authority is not None
            or reference_extent is None
        ):
            _fail(
                "coordinate_reference_evidence_mismatch",
                "$.reference_extent",
                "Non-frame profiles require one exact reference extent.",
            )
        try:
            reference = verify_bound_reference_extent(reference_extent)
        except CoordinateReferenceError as exc:
            _fail("coordinate_reference_evidence_unverified", "$.reference_extent", str(exc))
        width = reference.width
        height = reference.height
        units = reference.units
        authority_ref = reference.record_ref
        authority_digest = reference.record_sha256
        selector = reference.selector
    else:
        if (
            reference_extent is not None
            or reference_frame_authority is not None
            or frame_record is None
        ):
            _fail(
                "coordinate_frame_evidence_mismatch",
                "$.frame_record",
                "Frame profiles require one exact frame record.",
            )
        frame, frame_descriptor = _frame_evidence(
            frame_record,
            expected_kind=profile.frame_record_kind,
        )
        width = frame.reference_width
        height = frame.reference_height
        units = frame.reference_units
        authority_ref = frame.record_ref
        authority_digest = frame.record_sha256
        selector = frame.selector
    if units != profile.reference_units:
        _fail(
            "coordinate_reference_units_mismatch",
            "$.reference_extent.units",
            "Resolved reference units do not match the controlled profile.",
        )

    verified_lineage: list[BoundCoordinateRecord] = []
    for index, item in enumerate(lineage_records):
        try:
            record = verify_bound_coordinate_record(item)
        except CoordinateRecordError as exc:
            _fail(
                "coordinate_lineage_unverified",
                f"$.lineage_records[{index}]",
                str(exc),
            )
        if record.archive_identity != coordinate_archive:
            _fail(
                "coordinate_archive_mismatch",
                f"$.lineage_records[{index}]",
                "Coordinate lineage comes from a different archive/store.",
            )
        verified_lineage.append(record)

    overlay_refs: tuple[DigestBoundCoordinateRecordRef, ...] = ()
    if source_camera_overlay_status == CANONICAL_OVERLAY_REQUIRES_TRANSFORM:
        assert chain is not None  # established before reference authority resolution
        assert reference_frame_authority is not None
        if chain.descriptor_space_id != profile.space_id:
            _fail(
                "coordinate_transform_start_mismatch",
                "$.source_camera_overlay",
                "Transform chain does not start in the controlled profile space.",
            )
        if chain.row_identity is not None and not _same_row_identity(
            chain.row_identity,
            row_identity,
        ):
            _fail(
                "coordinate_transform_row_identity_mismatch",
                "$.source_camera_overlay",
                "Rowwise transform identity does not equal the coordinate surface identity.",
            )
        if not _same_pixel_frame(
            chain.descriptor_frame_authority,
            reference_frame_authority,
        ):
            _fail(
                "coordinate_transform_source_frame_mismatch",
                "$.source_camera_overlay",
                "Transform source frame does not match exact descriptor authority.",
            )
        if pixel_convention != chain.descriptor_pixel_convention:
            _fail(
                "coordinate_transform_pixel_convention_mismatch",
                "$.pixel_convention",
                "Descriptor convention does not match the transform source frame.",
            )
        overlay_refs = tuple(
            DigestBoundCoordinateRecordRef(
                record_ref=item.record_ref,
                record_sha256=item.record_sha256,
            )
            for item in chain.transform_records
        )
    elif transform_chain is not None:
        _fail(
            "coordinate_overlay_evidence_unexpected",
            "$.source_camera_overlay",
            "Only a transform-required overlay can carry transform evidence.",
        )

    authority = DigestBoundCoordinateRecordRef(
        record_ref=authority_ref,
        record_sha256=authority_digest,
    )
    descriptor = build_canonical_coordinate_descriptor(
        profile_id=profile_id,
        geometry_type=geometry_type,
        components=components,
        component_units=component_units,
        reference_width=width,
        reference_height=height,
        reference_authority=authority,
        reference_selector=selector,
        pixel_convention=pixel_convention,
        row_identity_contract=row_identity.contract,
        row_identity_record_ref=row_identity.record_ref,
        source_camera_overlay_status=source_camera_overlay_status,
        overlay_transform_refs=overlay_refs,
        frame_record=frame_descriptor,
        lineage_refs=tuple(
            DigestBoundCoordinateRecordRef(
                record_ref=item.record_ref,
                record_sha256=item.record_sha256,
            )
            for item in verified_lineage
        ),
    )
    bound = BoundCanonicalCoordinateDescriptor(
        descriptor=descriptor,
        coordinate_node=coordinate_node,
        row_identity=row_identity,
        reference_extent=reference_extent,
        reference_frame_authority=reference_frame_authority,
        transform_chain=transform_chain,
        lineage_records=tuple(verified_lineage),
        frame_record=frame_record,
        owner_dtype=owner_dtype,
        archive=coordinate_archive,
        _seal=_BOUND_COORDINATE_DESCRIPTOR_SEAL,
    )
    return _verify_bound_canonical_coordinate_descriptor(bound)


def stamp_bound_canonical_coordinate_descriptors(
    values: Iterable[BoundCanonicalCoordinateDescriptor],
    *,
    attr_name: str = COORDINATE_DESCRIPTOR_ATTR,
) -> tuple[CanonicalCoordinateDescriptor, ...]:
    """Atomically stamp a same-rowset surface set after complete preflight."""

    bindings = tuple(_verify_bound_canonical_coordinate_descriptor(item) for item in values)
    if not bindings:
        _fail("coordinate_surface_set_empty", "$", "At least one coordinate surface is required.")
    identity_pairs = {
        (item.row_identity.record_ref, item.row_identity.record_sha256)
        for item in bindings
    }
    if len(identity_pairs) != 1:
        _fail(
            "coordinate_sibling_identity_drift",
            "$.row_identity",
            "All coordinate siblings in one publication must share the exact identity record.",
        )
    identity_surfaces = {
        (
            item.row_identity.archive_identity,
            item.row_identity.rowset_path,
            item.row_identity.key_array_path,
            item.row_identity.contract.domain,
            item.row_identity.contract.mode,
        )
        for item in bindings
    }
    if len(identity_surfaces) != 1:
        _fail(
            "coordinate_sibling_rowset_drift",
            "$.row_identity",
            "Coordinate siblings must share one exact archive, rowset path, key path, domain, and mode.",
        )
    coordinate_archives = {
        archive_identity(item.coordinate_node) for item in bindings
    }
    if len(coordinate_archives) != 1:
        _fail(
            "coordinate_sibling_archive_drift",
            "$.coordinate_node",
            "A publication batch cannot span archives/stores.",
        )
    paths = tuple(canonical_node_path(item.coordinate_node) for item in bindings)
    if len(set(paths)) != len(paths):
        _fail(
            "coordinate_surface_duplicate",
            "$",
            "A coordinate surface cannot occur twice in one publication.",
        )
    attrs_targets: list[Any] = []
    snapshots: list[dict[str, Any]] = []
    intended_payloads: list[dict[str, Any]] = []
    expected_final_attrs: list[dict[str, Any]] = []
    for item in bindings:
        attrs = getattr(item.coordinate_node, "attrs", None)
        if attrs is None:
            _fail(
                "coordinate_attrs_unavailable",
                "$.coordinate_node.attrs",
                "Coordinate node must expose mutable attrs.",
            )
        trusted = type(attrs) is dict
        if not trusted:
            try:
                from zarr.core.attributes import Attributes as ZarrAttributes
            except ImportError:  # pragma: no cover - Palette depends on zarr
                ZarrAttributes = None  # type: ignore[assignment,misc]
            trusted = ZarrAttributes is not None and type(attrs) is ZarrAttributes
        if not trusted or not all(
            callable(getattr(attrs, operation, None))
            for operation in ("update", "__setitem__", "__delitem__")
        ):
            _fail(
                "coordinate_attrs_transaction_untrusted",
                "$.coordinate_node.attrs",
                "Coordinate attrs must be an exact built-in dict or exact Zarr "
                "Attributes implementation; no write was attempted.",
            )
        attrs_targets.append(attrs)
        snapshot = copy.deepcopy(dict(attrs))
        occupied = snapshot.get(attr_name)
        if isinstance(occupied, Mapping) and occupied.get("schema_version") == 1:
            _fail(
                "coordinate_v1_migration_required",
                f"$.{attr_name}",
                "Canonical v2 publication refuses an occupied historical-v1 descriptor; use the explicit migration API.",
            )
        if occupied is not None and (
            not isinstance(occupied, Mapping)
            or type(occupied.get("schema_version")) is not int
            or occupied.get("schema_version") != 2
        ):
            _fail(
                "coordinate_descriptor_occupied_unknown",
                f"$.{attr_name}",
                "Canonical publication refuses to overwrite an occupied non-v2 descriptor.",
            )
        payload = canonical_coordinate_descriptor_v2_attrs(
            item.descriptor,
            attr_name=attr_name,
        )
        payload[f"{attr_name}{_OWNER_DTYPE_ATTR_SUFFIX}"] = item.owner_dtype
        expected = copy.deepcopy(snapshot)
        expected.update(copy.deepcopy(payload))
        snapshots.append(snapshot)
        intended_payloads.append(payload)
        expected_final_attrs.append(expected)
    try:
        for attrs, payload, expected in zip(
            attrs_targets,
            intended_payloads,
            expected_final_attrs,
            strict=True,
        ):
            attrs.update(payload)
            if not _exact_value_equal(dict(attrs), expected):
                _fail(
                    "coordinate_surface_stamp_exact_attrs_mismatch",
                    "$",
                    "Coordinate attrs differ from the exact pre-call snapshot plus intended canonical changes.",
                )
        # A successful ``attrs.update`` is not proof that the requested bytes
        # were persisted.  Reload every descriptor through the full evidence
        # boundary before committing the transaction result.
        reloaded = tuple(
            load_bound_canonical_coordinate_descriptor(
                item.coordinate_node,
                row_identity=item.row_identity,
                reference_extent=item.reference_extent,
                reference_frame_authority=item.reference_frame_authority,
                transform_chain=item.transform_chain,
                lineage_records=item.lineage_records,
                frame_record=item.frame_record,
                attr_name=attr_name,
            )
            for item in bindings
        )
        if any(
            loaded.descriptor != requested.descriptor
            for loaded, requested in zip(reloaded, bindings, strict=True)
        ):
            _fail(
                "coordinate_surface_stamp_reload_mismatch",
                "$",
                "Reloaded descriptor differs from the requested canonical record.",
            )
        for attrs, expected in zip(
            attrs_targets,
            expected_final_attrs,
            strict=True,
        ):
            if not _exact_value_equal(dict(attrs), expected):
                _fail(
                    "coordinate_surface_reload_mutated_attrs",
                    "$",
                    "Descriptor reload unexpectedly mutated or coerced coordinate attrs.",
                )
    except Exception:
        rollback_failures: list[str] = []
        for path, attrs, snapshot in zip(
            paths,
            attrs_targets,
            snapshots,
            strict=True,
        ):
            try:
                for name in tuple(attrs.keys()):
                    if name not in snapshot:
                        del attrs[name]
                attrs.update(copy.deepcopy(snapshot))
                if not _exact_value_equal(dict(attrs), snapshot):
                    raise RuntimeError("restored attrs differ from snapshot")
            except Exception as rollback_exc:  # pragma: no cover - hostile mapping
                rollback_failures.append(f"{path}: {rollback_exc}")
        code = (
            "coordinate_surface_stamp_rollback_incomplete"
            if rollback_failures
            else "coordinate_surface_stamp_failed"
        )
        details = f" Rollback failures: {rollback_failures!r}." if rollback_failures else ""
        _fail(code, "$", f"Coordinate surface publication failed.{details}")
    return tuple(item.descriptor for item in reloaded)


def migrate_historical_coordinate_descriptor_v1_to_v2(
    coordinate_node: Any,
    *,
    attr_name: str = COORDINATE_DESCRIPTOR_ATTR,
) -> None:
    """Explicit fail-closed migration boundary for historical descriptors.

    V1 descriptors do not contain digest-bound typed frame, exact row-record,
    or direction-labelled transform authority.  Equivalence therefore cannot
    be proved from the descriptor alone.  Archive migration must first resolve
    and seal that external lineage, then use a future evidence-complete
    migration implementation; normal publication never overwrites v1.
    """

    attrs = getattr(coordinate_node, "attrs", None)
    raw = attrs.get(attr_name) if isinstance(attrs, Mapping) else None
    if not isinstance(raw, Mapping) or raw.get("schema_version") != 1:
        _fail(
            "coordinate_v1_migration_source_required",
            f"$.{attr_name}",
            "Explicit migration requires one occupied historical-v1 descriptor.",
        )
    _fail(
        "coordinate_v1_migration_lineage_unproven",
        f"$.{attr_name}",
        "Historical descriptor lineage cannot be proven from v1 metadata alone; this archive must remain unchanged and fail closed.",
    )


def stamp_bound_canonical_coordinate_descriptor(
    value: BoundCanonicalCoordinateDescriptor,
    *,
    attr_name: str = COORDINATE_DESCRIPTOR_ATTR,
) -> CanonicalCoordinateDescriptor:
    """Stamp one surface; multi-surface runs must use the atomic set API."""

    return stamp_bound_canonical_coordinate_descriptors(
        (value,),
        attr_name=attr_name,
    )[0]


def require_bound_canonical_coordinate_descriptor(
    value: Any,
) -> BoundCanonicalCoordinateDescriptor:
    """Reverify and return one sealed exact-evidence descriptor binding.

    Producers that copy or subset a coordinate surface use this public gate to
    consume its frame and transform evidence.  Merely parsing descriptor attrs
    is insufficient because paths and digests without their retained persisted
    nodes are not authority.
    """

    return _verify_bound_canonical_coordinate_descriptor(value)


def load_bound_canonical_coordinate_descriptor(
    coordinate_node: Any,
    *,
    row_identity: BoundRowIdentityContract,
    reference_extent: BoundReferenceExtent | None = None,
    reference_frame_authority: BoundPixelFrameAuthority | None = None,
    transform_chain: BoundDirectedTransformChain | None = None,
    lineage_records: Sequence[BoundCoordinateRecord] = (),
    frame_record: Any = None,
    attr_name: str = COORDINATE_DESCRIPTOR_ATTR,
) -> BoundCanonicalCoordinateDescriptor:
    """Load stored attrs and prove every external record against exact nodes."""

    try:
        row_identity = require_bound_row_identity_contract(row_identity)
    except RowIdentityContractError as exc:
        _fail("coordinate_row_identity_unverified", "$.row_identity", str(exc))
    attrs = getattr(coordinate_node, "attrs", None)
    if attrs is None:
        _fail("coordinate_attrs_unavailable", "$.coordinate_node.attrs", "Coordinate attrs are missing.")
    raw_descriptor = attrs.get(attr_name) if isinstance(attrs, Mapping) else None
    if not isinstance(raw_descriptor, Mapping):
        _fail(
            "descriptor_attr_missing",
            f"$.{attr_name}",
            "Persisted canonical coordinate-array descriptor attr is missing.",
        )
    if type(raw_descriptor.get("schema_version")) is not int:
        _fail(
            "canonical_schema_version_required",
            f"$.{attr_name}.schema_version",
            "Persisted canonical descriptors require an exact integer schema version.",
        )
    owner_dtype = _node_numeric_dtype(coordinate_node)
    dtype_attr = f"{attr_name}{_OWNER_DTYPE_ATTR_SUFFIX}"
    if attrs.get(dtype_attr) != owner_dtype:
        _fail(
            "coordinate_owner_dtype_mismatch",
            f"$.{dtype_attr}",
            "Persisted owner dtype is missing or differs from the exact coordinate-array dtype.",
        )
    descriptor = load_canonical_coordinate_descriptor_attrs(
        attrs,
        row_identity_contract=row_identity.contract,
        expected_row_identity_record_ref=row_identity.record_ref,
        owner_shape=_node_shape(coordinate_node),
        attr_name=attr_name,
    )
    bound = BoundCanonicalCoordinateDescriptor(
        descriptor=descriptor,
        coordinate_node=coordinate_node,
        row_identity=row_identity,
        reference_extent=reference_extent,
        reference_frame_authority=reference_frame_authority,
        transform_chain=transform_chain,
        lineage_records=tuple(lineage_records),
        frame_record=frame_record,
        owner_dtype=owner_dtype,
        archive=archive_identity(coordinate_node),
        _seal=_BOUND_COORDINATE_DESCRIPTOR_SEAL,
    )
    return _verify_bound_canonical_coordinate_descriptor(bound)


__all__ = [
    "BoundCanonicalCoordinateDescriptor",
    "CanonicalCoordinatePublicationError",
    "build_bound_canonical_coordinate_descriptor",
    "load_bound_canonical_coordinate_descriptor",
    "migrate_historical_coordinate_descriptor_v1_to_v2",
    "require_bound_canonical_coordinate_descriptor",
    "stamp_bound_canonical_coordinate_descriptor",
    "stamp_bound_canonical_coordinate_descriptors",
]
