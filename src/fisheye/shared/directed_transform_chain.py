"""Resolve sealed v2 links into an acyclic typed-endpoint transform chain."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from fisheye.shared.archive_identity import ArchiveIdentity
from fisheye.shared.coordinate_identity import BoundRowIdentityContract, RowIdentityContractError
from fisheye.shared.directed_transform_v2 import (
    AFFINE_2D_ROWWISE_KIND,
    BoundDirectedTransformV2,
    DirectedTransformV2Error,
    apply_bound_directed_transform_v2,
    require_bound_directed_transform_v2,
)
from fisheye.shared.pixel_frame_authority import (
    SOURCE_CAMERA_IMAGE_SPACE_ID,
    BoundPixelFrameAuthority,
    PixelFrameAuthorityError,
    require_bound_pixel_frame_authority,
    require_source_camera_pixel_frame_authority,
)


_BOUND_CHAIN_SEAL = object()


class DirectedTransformChainError(DirectedTransformV2Error):
    """Raised when exact v2 links do not form one safe directed chain."""


@dataclass(frozen=True)
class BoundDirectedTransformRecordRef:
    array_path: str
    record_ref: str
    record_sha256: str
    payload_sha256: str
    authority_ref: str
    authority_sha256: str

    def to_descriptor_dict(self) -> dict[str, str]:
        return {"record_ref": self.record_ref, "record_sha256": self.record_sha256}


def _same_frame(left: BoundPixelFrameAuthority, right: BoundPixelFrameAuthority) -> bool:
    return (
        left.archive_identity == right.archive_identity
        and left.record_ref == right.record_ref
        and left.record_sha256 == right.record_sha256
        and left.endpoint == right.endpoint
    )


def _same_identity(left: BoundRowIdentityContract, right: BoundRowIdentityContract) -> bool:
    return (
        left.archive_identity == right.archive_identity
        and left.record_ref == right.record_ref
        and left.record_sha256 == right.record_sha256
        and left.rowset_path == right.rowset_path
        and left.key_array_path == right.key_array_path
        and left.leading_dimension == right.leading_dimension
    )


def _record(link: BoundDirectedTransformV2) -> BoundDirectedTransformRecordRef:
    return BoundDirectedTransformRecordRef(
        array_path=link.array_path,
        record_ref=link.record_ref,
        record_sha256=link.transform_sha256,
        payload_sha256=link.transform.payload.record_sha256,
        authority_ref=link.authority.record_ref,
        authority_sha256=link.authority.record_sha256,
    )


def _links(values: Sequence[BoundDirectedTransformV2]) -> tuple[BoundDirectedTransformV2, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise DirectedTransformChainError("links must be an ordered transform sequence.")
    try:
        raw = tuple(values)
    except TypeError as exc:
        raise DirectedTransformChainError("links must be an ordered transform sequence.") from exc
    if not raw:
        raise DirectedTransformChainError("A source-camera overlay chain cannot be empty.")
    result: list[BoundDirectedTransformV2] = []
    for index, value in enumerate(raw):
        try:
            result.append(require_bound_directed_transform_v2(value))
        except DirectedTransformV2Error as exc:
            raise DirectedTransformChainError(f"Link {index} is stale: {exc}") from exc
    paths = [item.array_path for item in result]
    if len(paths) != len(set(paths)):
        raise DirectedTransformChainError("A transform record cannot occur twice in a chain.")
    return tuple(result)


def _validate(
    links: tuple[BoundDirectedTransformV2, ...],
    records: tuple[BoundDirectedTransformRecordRef, ...],
) -> tuple[
    BoundPixelFrameAuthority,
    BoundPixelFrameAuthority,
    str | None,
    BoundRowIdentityContract | None,
    ArchiveIdentity,
]:
    if len(links) != len(records):
        raise DirectedTransformChainError("Transform-record count changed.")
    archive = links[0].archive_identity
    if any(item.archive_identity != archive for item in links):
        raise DirectedTransformChainError("Chain links come from different archives/stores.")
    try:
        descriptor_frame = require_bound_pixel_frame_authority(links[0].source_frame)
        camera_frame = require_bound_pixel_frame_authority(links[-1].target_frame)
    except PixelFrameAuthorityError as exc:
        raise DirectedTransformChainError(f"Chain endpoint is stale: {exc}") from exc
    if descriptor_frame.space_id == SOURCE_CAMERA_IMAGE_SPACE_ID:
        raise DirectedTransformChainError(
            "Direct source-camera coordinates must not use a transform chain."
        )

    camera_ids: set[str] = set()
    row_identity: BoundRowIdentityContract | None = None
    vertices: list[BoundPixelFrameAuthority] = [descriptor_frame]
    for index, (link, expected_record) in enumerate(zip(links, records, strict=True)):
        if _record(link) != expected_record:
            raise DirectedTransformChainError(f"Transform record {index} changed.")
        try:
            source = require_bound_pixel_frame_authority(link.source_frame)
            target = require_bound_pixel_frame_authority(link.target_frame)
        except PixelFrameAuthorityError as exc:
            raise DirectedTransformChainError(f"Link {index} endpoint is stale: {exc}") from exc
        if index == 0:
            if not _same_frame(source, descriptor_frame):
                raise DirectedTransformChainError("First source frame changed.")
        else:
            previous = links[index - 1]
            if not _same_frame(previous.target_frame, source):
                raise DirectedTransformChainError(
                    f"Exact endpoint discontinuity between links {index-1} and {index}."
                )
            if previous.transform.target != link.transform.source:
                raise DirectedTransformChainError(
                    f"Persisted endpoint discontinuity between links {index-1} and {index}."
                )
            if previous.transform.target.pixel_convention != link.transform.source.pixel_convention:
                raise DirectedTransformChainError(
                    f"Pixel-convention discontinuity between links {index-1} and {index}."
                )
        if index < len(links) - 1 and (
            source.space_id == SOURCE_CAMERA_IMAGE_SPACE_ID
            or target.space_id == SOURCE_CAMERA_IMAGE_SPACE_ID
        ):
            raise DirectedTransformChainError(
                "source_camera_image_px may occur only as the final chain endpoint."
            )
        if link.transform.camera_id is not None:
            camera_ids.add(link.transform.camera_id)
        if link.transform.kind == AFFINE_2D_ROWWISE_KIND:
            if link.row_identity is None:
                raise DirectedTransformChainError("Rowwise link lacks identity.")
            try:
                link.row_identity.assert_verified()
            except RowIdentityContractError as exc:
                raise DirectedTransformChainError(f"Rowwise identity is stale: {exc}") from exc
            if row_identity is None:
                row_identity = link.row_identity
            elif not _same_identity(row_identity, link.row_identity):
                raise DirectedTransformChainError(
                    "Rowwise links do not share one exact rowset identity/path."
                )
        vertices.append(target)

    # A simple directed chain cannot revisit a controlled space or exact frame.
    spaces = [item.space_id for item in vertices]
    if len(spaces) != len(set(spaces)):
        raise DirectedTransformChainError(
            "Transform chain repeats a coordinate space and is cyclic/ambiguous."
        )
    endpoint_keys = [(item.record_ref, item.record_sha256) for item in vertices]
    if len(endpoint_keys) != len(set(endpoint_keys)):
        raise DirectedTransformChainError(
            "Transform chain repeats an exact endpoint and is cyclic."
        )
    if vertices[-1].space_id != SOURCE_CAMERA_IMAGE_SPACE_ID:
        raise DirectedTransformChainError("Final endpoint must be source_camera_image_px.")
    try:
        require_source_camera_pixel_frame_authority(camera_frame)
    except PixelFrameAuthorityError as exc:
        raise DirectedTransformChainError(
            f"Final source-camera endpoint is invalid: {exc}"
        ) from exc
    if not _same_frame(vertices[-1], camera_frame):
        raise DirectedTransformChainError("Final camera frame changed.")
    if len(camera_ids) > 1:
        raise DirectedTransformChainError("Chain mixes different selected cameras.")
    return (
        descriptor_frame,
        camera_frame,
        next(iter(camera_ids)) if camera_ids else None,
        row_identity,
        archive,
    )


@dataclass(frozen=True, init=False)
class BoundDirectedTransformChain:
    descriptor_space_id: str
    source_camera_space_id: str
    descriptor_frame_authority: BoundPixelFrameAuthority = field(repr=False, compare=False)
    source_camera_frame_authority: BoundPixelFrameAuthority = field(repr=False, compare=False)
    camera_id: str | None
    row_identity: BoundRowIdentityContract | None = field(repr=False, compare=False)
    links: tuple[BoundDirectedTransformV2, ...]
    transform_records: tuple[BoundDirectedTransformRecordRef, ...]
    _archive_identity: ArchiveIdentity = field(repr=False, compare=False)
    _seal: object = field(repr=False, compare=False)

    def __init__(
        self,
        *,
        descriptor_frame: BoundPixelFrameAuthority,
        camera_frame: BoundPixelFrameAuthority,
        camera_id: str | None,
        row_identity: BoundRowIdentityContract | None,
        links: tuple[BoundDirectedTransformV2, ...],
        records: tuple[BoundDirectedTransformRecordRef, ...],
        archive: ArchiveIdentity,
        _verification_seal: object | None = None,
    ) -> None:
        if _verification_seal is not _BOUND_CHAIN_SEAL:
            raise DirectedTransformChainError("Bound chains cannot be constructed directly.")
        object.__setattr__(self, "descriptor_space_id", descriptor_frame.space_id)
        object.__setattr__(self, "source_camera_space_id", SOURCE_CAMERA_IMAGE_SPACE_ID)
        object.__setattr__(self, "descriptor_frame_authority", descriptor_frame)
        object.__setattr__(self, "source_camera_frame_authority", camera_frame)
        object.__setattr__(self, "camera_id", camera_id)
        object.__setattr__(self, "row_identity", row_identity)
        object.__setattr__(self, "links", links)
        object.__setattr__(self, "transform_records", records)
        object.__setattr__(self, "_archive_identity", archive)
        object.__setattr__(self, "_seal", _verification_seal)

    @property
    def archive_identity(self) -> ArchiveIdentity:
        return self._archive_identity

    @property
    def descriptor_pixel_convention(self) -> str:
        return self.descriptor_frame_authority.pixel_convention

    @property
    def source_camera_pixel_convention(self) -> str:
        return self.source_camera_frame_authority.pixel_convention

    @property
    def descriptor_transform_refs(self) -> tuple[dict[str, str], ...]:
        return tuple(item.to_descriptor_dict() for item in self.transform_records)

    @property
    def transform_authorities(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (item.authority_ref, item.authority_sha256)
            for item in self.transform_records
        )


def resolve_bound_directed_transform_chain(
    links: Sequence[BoundDirectedTransformV2],
) -> BoundDirectedTransformChain:
    bound_links = _links(links)
    records = tuple(_record(item) for item in bound_links)
    descriptor, camera, camera_id, identity, archive = _validate(bound_links, records)
    return BoundDirectedTransformChain(
        descriptor_frame=descriptor,
        camera_frame=camera,
        camera_id=camera_id,
        row_identity=identity,
        links=bound_links,
        records=records,
        archive=archive,
        _verification_seal=_BOUND_CHAIN_SEAL,
    )


def require_bound_directed_transform_chain(value: Any) -> BoundDirectedTransformChain:
    if type(value) is not BoundDirectedTransformChain or value._seal is not _BOUND_CHAIN_SEAL:
        raise DirectedTransformChainError("A sealed v2 transform chain is required.")
    descriptor, camera, camera_id, identity, archive = _validate(
        _links(value.links), value.transform_records
    )
    if (
        not _same_frame(descriptor, value.descriptor_frame_authority)
        or not _same_frame(camera, value.source_camera_frame_authority)
        or camera_id != value.camera_id
        or archive != value.archive_identity
        or ((identity is None) != (value.row_identity is None))
        or (
            identity is not None
            and value.row_identity is not None
            and not _same_identity(identity, value.row_identity)
        )
    ):
        raise DirectedTransformChainError("Bound chain changed after resolution.")
    return value


def apply_bound_directed_transform_chain(
    points_xy: Any,
    chain: BoundDirectedTransformChain,
    *,
    row_identity: BoundRowIdentityContract | None = None,
) -> np.ndarray:
    bound = require_bound_directed_transform_chain(chain)
    if bound.row_identity is None:
        if row_identity is not None:
            raise DirectedTransformChainError("Constant chain rejects row identity.")
    else:
        if row_identity is None or not _same_identity(bound.row_identity, row_identity):
            raise DirectedTransformChainError("Application row identity differs from chain.")
    result = points_xy
    for link in bound.links:
        result = apply_bound_directed_transform_v2(
            result,
            link,
            row_identity=(
                row_identity if link.transform.kind == AFFINE_2D_ROWWISE_KIND else None
            ),
        )
    return result


__all__ = [
    "BoundDirectedTransformChain",
    "BoundDirectedTransformRecordRef",
    "DirectedTransformChainError",
    "apply_bound_directed_transform_chain",
    "require_bound_directed_transform_chain",
    "resolve_bound_directed_transform_chain",
]
