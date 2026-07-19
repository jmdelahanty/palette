"""Canonical coordinate publication for detection and crop observation rows.

This module is the future-writer boundary for observation geometry.  It never
infers a frame from array names, root dimensions, numerical ranges, or a legacy
space label.  Callers must supply sealed source-camera, normalized-frame, and
direction-labelled transform evidence created from the exact acquisition
authority.

Detection publication persists three deliberately redundant surfaces:

* ``bbox_norm_coords`` -- source-camera-normalized ``cx,cy,w,h``;
* ``bbox_img_xyxy`` -- source-camera continuous pixel edges; and
* ``centers_img_xy`` -- source-camera continuous points derived from the exact
  persisted pixel bbox.

All three share one exact ``instance_key`` identity and one sealed
``source_acquisition_frame_index`` temporal authority.  The normalized-to-pixel
projection and bbox-to-center operation are digest-bound records, so a consumer
can distinguish a genuine persisted derivation from matching-looking numbers.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from fisheye.shared.archive_identity import archive_identity
from fisheye.shared.canonical_coordinate_publication import (
    BoundCanonicalCoordinateDescriptor,
    build_bound_canonical_coordinate_descriptor,
    load_bound_canonical_coordinate_descriptor,
    require_bound_canonical_coordinate_descriptor,
    stamp_bound_canonical_coordinate_descriptors,
)
from fisheye.shared.coordinate_descriptor import (
    CANONICAL_OVERLAY_DIRECT,
    CANONICAL_OVERLAY_REQUIRES_TRANSFORM,
)
from fisheye.shared.coordinate_frame_record import array_payload_sha256
from fisheye.shared.coordinate_identity import (
    OBSERVATION_INSTANCE_DOMAIN,
    BoundRowIdentityContract,
    BoundSourceRowTemporalAuthority,
    build_row_identity_contract,
    load_bound_row_identity_contract,
    load_bound_source_row_temporal_authority,
    require_bound_source_row_temporal_authority,
    resolve_source_acquisition_frame_indices,
    stamp_and_bind_row_identity_contract,
    stamp_source_row_temporal_authority,
)
from fisheye.shared.coordinate_record import (
    BoundCoordinateRecord,
    bind_persisted_coordinate_record,
    stamp_and_bind_persisted_coordinate_record,
    verify_bound_coordinate_record,
)
from fisheye.shared.coordinate_reference import canonical_node_path
from fisheye.shared.directed_transform_chain import (
    BoundDirectedTransformChain,
    apply_bound_directed_transform_chain,
    require_bound_directed_transform_chain,
    resolve_bound_directed_transform_chain,
)
from fisheye.shared.directed_transform_v2 import load_bound_directed_transform_v2
from fisheye.shared.pixel_frame_authority import (
    ROI_FRAME_KIND,
    SOURCE_CAMERA_FRAME_KIND,
    SOURCE_CAMERA_NORMALIZED_FRAME_KIND,
    BoundAcquisitionCameraFrame,
    BoundCropPlacementOwnership,
    BoundPixelFrameAuthority,
    load_normalized_pixel_frame_authority,
    load_persisted_acquisition_camera_authority,
    load_source_camera_pixel_frame_authority,
    require_bound_acquisition_camera_frame,
    require_bound_crop_placement_ownership,
    require_normalized_pixel_frame_authority,
    require_roi_pixel_frame_authority,
    require_source_camera_pixel_frame_authority,
    require_trusted_coordinate_attrs,
)
from fisheye.shared.transform_authority import load_bound_transform_authority


DETECTION_BBOX_PROJECTION_ATTR = "detection_bbox_projection"
DETECTION_BBOX_PROJECTION_SCHEMA_ID = "palette.detection_bbox_projection"
DETECTION_BBOX_PROJECTION_SCHEMA_VERSION = 1
DETECTION_BBOX_PROJECTION_OPERATION = "source_camera_normalized_cxcywh_to_image_xyxy_v1"

DETECTION_ACQUISITION_MAPPING_ATTR = "detection_acquisition_frame_mapping"
DETECTION_ACQUISITION_MAPPING_SCHEMA_ID = "palette.detection_acquisition_frame_mapping"
DETECTION_ACQUISITION_MAPPING_SCHEMA_VERSION = 1

BBOX_CENTER_DERIVATION_ATTR = "bbox_center_derivation"
BBOX_CENTER_DERIVATION_SCHEMA_ID = "palette.bbox_center_derivation"
BBOX_CENTER_DERIVATION_SCHEMA_VERSION = 1
BBOX_CENTER_DERIVATION_OPERATION = "xyxy_midpoint_v1"

CROP_GEOMETRY_SELECTION_ATTR = "crop_geometry_selection"
CROP_GEOMETRY_SELECTION_SCHEMA_ID = "palette.crop_geometry_selection"
CROP_GEOMETRY_SELECTION_SCHEMA_VERSION = 1
CROP_GEOMETRY_SELECTION_OPERATION = "exact_instance_key_subset_reorder_v1"

CROP_ROI_GEOMETRY_DERIVATION_ATTR = "crop_roi_geometry_derivation"
CROP_ROI_GEOMETRY_DERIVATION_SCHEMA_ID = "palette.crop_roi_geometry_derivation"
CROP_ROI_GEOMETRY_DERIVATION_SCHEMA_VERSION = 1
CROP_ROI_GEOMETRY_DERIVATION_OPERATION = (
    "roi_bbox_to_source_camera_via_crop_placement_v1"
)

SOURCE_CAMERA_PROFILE_ID = "source_camera_image_px.top_left_y_down.v1"
SOURCE_CAMERA_NORMALIZED_PROFILE_ID = "source_camera_normalized_xy.top_left_y_down.v1"
SOURCE_CAMERA_PIXEL_CONVENTION = "continuous"

_BOUND_DETECTION_FRAME_EVIDENCE_SEAL = object()
_BOUND_DETECTION_GEOMETRY_SEAL = object()
_BOUND_CROP_GEOMETRY_SEAL = object()
_BOUND_POSITION_SURFACE_SEAL = object()


class ObservationCoordinatePublicationError(ValueError):
    """Raised when observation geometry lacks exact coordinate authority."""


def _fail(message: str) -> None:
    raise ObservationCoordinatePublicationError(message)


def _same_row_identity(
    left: BoundRowIdentityContract,
    right: BoundRowIdentityContract,
) -> bool:
    return (
        left.archive_identity == right.archive_identity
        and left.record_ref == right.record_ref
        and left.record_sha256 == right.record_sha256
        and left.rowset_path == right.rowset_path
        and left.key_array_path == right.key_array_path
        and left.contract == right.contract
    )


def _same_pixel_frame(left: Any, right: Any) -> bool:
    return (
        type(left) is type(right)
        and left.archive_identity == right.archive_identity
        and left.record_ref == right.record_ref
        and left.record_sha256 == right.record_sha256
        and left.endpoint == right.endpoint
    )


def _array(node: Any, *, label: str) -> np.ndarray:
    try:
        values = np.array(node[:], copy=True, order="C")
        declared_dtype = np.dtype(getattr(node, "dtype"))
    except Exception as exc:
        _fail(f"Unable to read exact {label} array: {exc}.")
    declared_shape = getattr(node, "shape", None)
    if values.shape != declared_shape or values.dtype != declared_dtype:
        _fail(f"{label} values disagree with the declared dtype or shape.")
    if values.dtype.hasobject:
        _fail(f"{label} cannot use object dtype.")
    return values


def _payload(node: Any, values: np.ndarray) -> dict[str, Any]:
    return {
        "array_ref": f"/{canonical_node_path(node)}",
        "dtype": values.dtype.str,
        "shape": [int(item) for item in values.shape],
        "content_sha256": array_payload_sha256(node),
    }


def _require_child_path(node: Any, rowset_node: Any, name: str) -> None:
    expected = f"{canonical_node_path(rowset_node)}/{name}"
    if canonical_node_path(node) != expected:
        _fail(f"Canonical {name} must be persisted at exact path {expected!r}.")


def _raw_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return set(left) == set(right) and all(
            _raw_equal(left[name], right[name]) for name in left
        )
    if type(left) in {list, tuple}:
        return len(left) == len(right) and all(
            _raw_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, np.ndarray):
        return (
            left.dtype == right.dtype
            and left.shape == right.shape
            and np.array_equal(left, right, equal_nan=True)
        )
    try:
        result = left == right
    except Exception:
        return False
    return type(result) is bool and result


def _restore_attrs(attrs: Any, snapshot: Mapping[str, Any]) -> None:
    for name in tuple(attrs.keys()):
        del attrs[name]
    attrs.update(copy.deepcopy(dict(snapshot)))
    if not _raw_equal(dict(attrs), dict(snapshot)):
        raise RuntimeError("restored attrs differ from their exact snapshot")


def _attrs_snapshots(*nodes: Any) -> tuple[list[Any], list[dict[str, Any]]]:
    attrs_targets: list[Any] = []
    snapshots: list[dict[str, Any]] = []
    seen: set[int] = set()
    for index, node in enumerate(nodes):
        attrs = require_trusted_coordinate_attrs(
            node,
            label=f"Observation coordinate publication target {index}",
        )
        marker = id(attrs)
        if marker in seen:
            continue
        seen.add(marker)
        attrs_targets.append(attrs)
        snapshots.append(copy.deepcopy(dict(attrs)))
    return attrs_targets, snapshots


def _rollback_attrs(
    attrs_targets: list[Any],
    snapshots: list[dict[str, Any]],
    *,
    cause: Exception,
) -> None:
    failures: list[str] = []
    for attrs, snapshot in zip(attrs_targets, snapshots, strict=True):
        try:
            _restore_attrs(attrs, snapshot)
        except Exception as exc:  # pragma: no cover - hostile persistent mapping
            failures.append(str(exc))
    if failures:
        raise ObservationCoordinatePublicationError(
            "Observation coordinate publication failed and attrs rollback was "
            f"incomplete: {failures!r}."
        ) from cause


@dataclass(frozen=True, init=False)
class BoundDetectionFrameEvidence:
    """Sealed normalized-to-source-camera frame and transform evidence."""

    source_camera_frame: BoundPixelFrameAuthority = field(repr=False)
    normalized_frame: BoundPixelFrameAuthority = field(repr=False)
    normalized_to_source_camera: BoundDirectedTransformChain = field(repr=False)
    _seal: object = field(repr=False, compare=False)

    def __init__(
        self,
        *,
        source_camera_frame: BoundPixelFrameAuthority,
        normalized_frame: BoundPixelFrameAuthority,
        normalized_to_source_camera: BoundDirectedTransformChain,
        _verification_seal: object | None = None,
    ) -> None:
        if _verification_seal is not _BOUND_DETECTION_FRAME_EVIDENCE_SEAL:
            _fail("Detection frame evidence must be built by the sealed verifier.")
        object.__setattr__(self, "source_camera_frame", source_camera_frame)
        object.__setattr__(self, "normalized_frame", normalized_frame)
        object.__setattr__(
            self,
            "normalized_to_source_camera",
            normalized_to_source_camera,
        )
        object.__setattr__(self, "_seal", _verification_seal)

    @property
    def acquisition_frame(self) -> BoundAcquisitionCameraFrame:
        value = self.source_camera_frame.reference_extent
        return require_bound_acquisition_camera_frame(value)

    def assert_verified(self) -> None:
        require_bound_detection_frame_evidence(self)


def build_bound_detection_frame_evidence(
    *,
    source_camera_frame: BoundPixelFrameAuthority,
    normalized_frame: BoundPixelFrameAuthority,
    normalized_to_source_camera: BoundDirectedTransformChain,
) -> BoundDetectionFrameEvidence:
    """Verify one exact continuous source-camera normalized-to-pixel chain."""

    camera = require_source_camera_pixel_frame_authority(source_camera_frame)
    normalized = require_normalized_pixel_frame_authority(normalized_frame)
    chain = require_bound_directed_transform_chain(normalized_to_source_camera)
    if camera.record.kind != SOURCE_CAMERA_FRAME_KIND:
        _fail("Detection geometry requires a source-camera pixel frame.")
    if camera.pixel_convention != SOURCE_CAMERA_PIXEL_CONVENTION:
        _fail("Detection source-camera geometry requires continuous coordinates.")
    if normalized.record.kind != SOURCE_CAMERA_NORMALIZED_FRAME_KIND:
        _fail(
            "Detection normalized geometry requires a source-camera normalized frame."
        )
    expected_camera_ref = {
        "record_ref": camera.record_ref,
        "record_sha256": camera.record_sha256,
    }
    if normalized.record.lineage.get("pixel_frame") != expected_camera_ref:
        _fail("Normalized frame does not bind the exact source-camera frame.")
    if (
        not _same_pixel_frame(chain.descriptor_frame_authority, normalized)
        or not _same_pixel_frame(chain.source_camera_frame_authority, camera)
        or chain.row_identity is not None
        or chain.descriptor_space_id != "source_camera_normalized_xy"
        or chain.source_camera_space_id != "source_camera_image_px"
    ):
        _fail(
            "Detection normalized-to-camera chain has the wrong direction, "
            "endpoints, or row domain."
        )
    if not (
        camera.archive_identity == normalized.archive_identity == chain.archive_identity
    ):
        _fail("Detection frame evidence spans different archives.")
    return BoundDetectionFrameEvidence(
        source_camera_frame=camera,
        normalized_frame=normalized,
        normalized_to_source_camera=chain,
        _verification_seal=_BOUND_DETECTION_FRAME_EVIDENCE_SEAL,
    )


def require_bound_detection_frame_evidence(
    value: Any,
) -> BoundDetectionFrameEvidence:
    if (
        type(value) is not BoundDetectionFrameEvidence
        or value._seal is not _BOUND_DETECTION_FRAME_EVIDENCE_SEAL
    ):
        _fail("A sealed detection frame-evidence bundle is required.")
    current = build_bound_detection_frame_evidence(
        source_camera_frame=value.source_camera_frame,
        normalized_frame=value.normalized_frame,
        normalized_to_source_camera=value.normalized_to_source_camera,
    )
    if current != value:
        _fail("Detection frame evidence changed after binding.")
    return value


def derive_detection_source_camera_geometry(
    bbox_norm_coords: Any,
    *,
    frame_evidence: BoundDetectionFrameEvidence,
) -> tuple[np.ndarray, np.ndarray]:
    """Derive exact source-camera bbox edges and centers from normalized boxes."""

    evidence = require_bound_detection_frame_evidence(frame_evidence)
    normalized = np.asarray(bbox_norm_coords)
    if (
        normalized.dtype.kind != "f"
        or normalized.ndim != 2
        or normalized.shape[1:] != (4,)
        or not np.isfinite(normalized).all()
    ):
        _fail("bbox_norm_coords must be a finite floating (N,4) cxcywh array.")
    dtype = normalized.dtype
    half = np.asarray(0.5, dtype=dtype)
    one = np.asarray(1.0, dtype=dtype)
    cx = normalized[:, 0]
    cy = normalized[:, 1]
    width = normalized[:, 2]
    height = normalized[:, 3]
    x_min_norm = cx - width * half
    y_min_norm = cy - height * half
    x_max_norm = cx + width * half
    y_max_norm = cy + height * half
    if normalized.shape[0] and (
        np.any(width <= 0)
        or np.any(height <= 0)
        or np.any(x_min_norm < 0)
        or np.any(y_min_norm < 0)
        or np.any(x_max_norm > one)
        or np.any(y_max_norm > one)
    ):
        _fail(
            "Canonical normalized detection boxes must have positive extents "
            "and remain inside the exact source-camera extent."
        )
    width_px = np.asarray(evidence.source_camera_frame.endpoint.width, dtype=dtype)
    height_px = np.asarray(evidence.source_camera_frame.endpoint.height, dtype=dtype)
    bbox_img = np.column_stack(
        (
            x_min_norm * width_px,
            y_min_norm * height_px,
            x_max_norm * width_px,
            y_max_norm * height_px,
        )
    ).astype(dtype, copy=False)
    centers = np.column_stack(
        (
            (bbox_img[:, 0] + bbox_img[:, 2]) * half,
            (bbox_img[:, 1] + bbox_img[:, 3]) * half,
        )
    ).astype(dtype, copy=False)
    return (
        np.array(bbox_img, copy=True, order="C"),
        np.array(centers, copy=True, order="C"),
    )


def _detection_records(
    *,
    row_identity: BoundRowIdentityContract,
    temporal: BoundSourceRowTemporalAuthority,
    frame_evidence: BoundDetectionFrameEvidence,
    bbox_norm_node: Any,
    bbox_norm: np.ndarray,
    bbox_img_node: Any,
    bbox_img: np.ndarray,
    centers_img_node: Any,
    centers_img: np.ndarray,
    source_lineage_records: tuple[BoundCoordinateRecord, ...],
) -> tuple[dict[str, Any], dict[str, Any]]:
    camera = frame_evidence.source_camera_frame
    normalized = frame_evidence.normalized_frame
    chain = frame_evidence.normalized_to_source_camera
    projection = {
        "schema_id": DETECTION_BBOX_PROJECTION_SCHEMA_ID,
        "schema_version": DETECTION_BBOX_PROJECTION_SCHEMA_VERSION,
        "operation": DETECTION_BBOX_PROJECTION_OPERATION,
        "source_bbox": _payload(bbox_norm_node, bbox_norm),
        "source_frame": {
            "record_ref": normalized.record_ref,
            "record_sha256": normalized.record_sha256,
        },
        "destination_bbox": _payload(bbox_img_node, bbox_img),
        "destination_frame": {
            "record_ref": camera.record_ref,
            "record_sha256": camera.record_sha256,
        },
        "direction": "source_camera_normalized_xy_to_source_camera_image_px",
        "transform_chain": [
            {
                "record_ref": item.record_ref,
                "record_sha256": item.record_sha256,
            }
            for item in chain.transform_records
        ],
        "reference_width_px": int(camera.endpoint.width),
        "reference_height_px": int(camera.endpoint.height),
        "formula": ("cxcywh_normalized_to_xyxy_edges_using_exact_reference_extent_v1"),
        "row_identity": {
            "record_ref": row_identity.record_ref,
            "record_sha256": row_identity.record_sha256,
        },
        "temporal_authority": {
            "record_ref": temporal.record_ref,
            "record_sha256": temporal.record_sha256,
        },
        "source_lineage": [
            {
                "record_ref": item.record_ref,
                "record_sha256": item.record_sha256,
            }
            for item in source_lineage_records
        ],
    }
    center = {
        "schema_id": BBOX_CENTER_DERIVATION_SCHEMA_ID,
        "schema_version": BBOX_CENTER_DERIVATION_SCHEMA_VERSION,
        "operation": BBOX_CENTER_DERIVATION_OPERATION,
        "source_bbox": _payload(bbox_img_node, bbox_img),
        "output_centers": _payload(centers_img_node, centers_img),
        "coordinate_frame": {
            "record_ref": camera.record_ref,
            "record_sha256": camera.record_sha256,
        },
        "formula": "center_x=(x_min+x_max)/2;center_y=(y_min+y_max)/2",
        "row_identity": {
            "record_ref": row_identity.record_ref,
            "record_sha256": row_identity.record_sha256,
        },
    }
    return projection, center


@dataclass(frozen=True, init=False)
class BoundSourceCameraPositionSurface:
    """Sealed exact source-camera point surface plus acquisition time."""

    coordinates: BoundCanonicalCoordinateDescriptor
    temporal_authority: BoundSourceRowTemporalAuthority = field(repr=False)
    _seal: object = field(repr=False, compare=False)

    def __init__(
        self,
        *,
        coordinates: BoundCanonicalCoordinateDescriptor,
        temporal_authority: BoundSourceRowTemporalAuthority,
        _verification_seal: object | None = None,
    ) -> None:
        if _verification_seal is not _BOUND_POSITION_SURFACE_SEAL:
            _fail("Position surfaces must be created by a canonical geometry loader.")
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "temporal_authority", temporal_authority)
        object.__setattr__(self, "_seal", _verification_seal)

    def assert_verified(self) -> None:
        require_bound_source_camera_position_surface(self)


def _position_surface(
    coordinates: BoundCanonicalCoordinateDescriptor,
    temporal: BoundSourceRowTemporalAuthority,
) -> BoundSourceCameraPositionSurface:
    descriptor = require_bound_canonical_coordinate_descriptor(coordinates)
    temporal = require_bound_source_row_temporal_authority(temporal)
    if not _same_row_identity(descriptor.row_identity, temporal.source_row_identity):
        _fail("Position coordinates and temporal authority use different row identity.")
    if (
        descriptor.descriptor.profile_id != SOURCE_CAMERA_PROFILE_ID
        or descriptor.descriptor.geometry_type != "point_xy"
        or descriptor.descriptor.components != ("x", "y")
        or descriptor.descriptor.component_units != ("px", "px")
        or descriptor.descriptor.pixel_convention != SOURCE_CAMERA_PIXEL_CONVENTION
        or descriptor.descriptor.source_camera_overlay.status
        != CANONICAL_OVERLAY_DIRECT
    ):
        _fail("Position surface is not the canonical source-camera point profile.")
    return BoundSourceCameraPositionSurface(
        coordinates=descriptor,
        temporal_authority=temporal,
        _verification_seal=_BOUND_POSITION_SURFACE_SEAL,
    )


def require_bound_source_camera_position_surface(
    value: Any,
) -> BoundSourceCameraPositionSurface:
    if (
        type(value) is not BoundSourceCameraPositionSurface
        or value._seal is not _BOUND_POSITION_SURFACE_SEAL
    ):
        _fail("A sealed source-camera position surface is required.")
    current = _position_surface(value.coordinates, value.temporal_authority)
    if current.coordinates.descriptor != value.coordinates.descriptor:
        _fail("Source-camera position surface changed after binding.")
    return value


@dataclass(frozen=True, init=False)
class BoundDetectionObservationGeometry:
    """Freshly validated canonical detection bbox and center surfaces."""

    bbox_normalized: BoundCanonicalCoordinateDescriptor
    bbox_image: BoundCanonicalCoordinateDescriptor
    centers_image: BoundCanonicalCoordinateDescriptor
    row_identity: BoundRowIdentityContract = field(repr=False)
    temporal_authority: BoundSourceRowTemporalAuthority = field(repr=False)
    frame_evidence: BoundDetectionFrameEvidence = field(repr=False)
    bbox_projection: BoundCoordinateRecord = field(repr=False)
    bbox_center_derivation: BoundCoordinateRecord = field(repr=False)
    source_lineage_records: tuple[BoundCoordinateRecord, ...] = field(repr=False)
    position_surface: BoundSourceCameraPositionSurface
    _rowset_node: Any = field(repr=False, compare=False)
    _key_node: Any = field(repr=False, compare=False)
    _source_frame_index_node: Any = field(repr=False, compare=False)
    _bbox_norm_node: Any = field(repr=False, compare=False)
    _bbox_img_node: Any = field(repr=False, compare=False)
    _centers_img_node: Any = field(repr=False, compare=False)
    _seal: object = field(repr=False, compare=False)

    def __init__(
        self,
        *,
        bbox_normalized: BoundCanonicalCoordinateDescriptor,
        bbox_image: BoundCanonicalCoordinateDescriptor,
        centers_image: BoundCanonicalCoordinateDescriptor,
        row_identity: BoundRowIdentityContract,
        temporal_authority: BoundSourceRowTemporalAuthority,
        frame_evidence: BoundDetectionFrameEvidence,
        bbox_projection: BoundCoordinateRecord,
        bbox_center_derivation: BoundCoordinateRecord,
        source_lineage_records: tuple[BoundCoordinateRecord, ...],
        rowset_node: Any,
        key_node: Any,
        source_frame_index_node: Any,
        bbox_norm_node: Any,
        bbox_img_node: Any,
        centers_img_node: Any,
        _verification_seal: object | None = None,
    ) -> None:
        if _verification_seal is not _BOUND_DETECTION_GEOMETRY_SEAL:
            _fail("Detection geometry must be created by the canonical loader.")
        object.__setattr__(self, "bbox_normalized", bbox_normalized)
        object.__setattr__(self, "bbox_image", bbox_image)
        object.__setattr__(self, "centers_image", centers_image)
        object.__setattr__(self, "row_identity", row_identity)
        object.__setattr__(self, "temporal_authority", temporal_authority)
        object.__setattr__(self, "frame_evidence", frame_evidence)
        object.__setattr__(self, "bbox_projection", bbox_projection)
        object.__setattr__(self, "bbox_center_derivation", bbox_center_derivation)
        object.__setattr__(self, "source_lineage_records", source_lineage_records)
        object.__setattr__(
            self,
            "position_surface",
            _position_surface(centers_image, temporal_authority),
        )
        object.__setattr__(self, "_rowset_node", rowset_node)
        object.__setattr__(self, "_key_node", key_node)
        object.__setattr__(self, "_source_frame_index_node", source_frame_index_node)
        object.__setattr__(self, "_bbox_norm_node", bbox_norm_node)
        object.__setattr__(self, "_bbox_img_node", bbox_img_node)
        object.__setattr__(self, "_centers_img_node", centers_img_node)
        object.__setattr__(self, "_seal", _verification_seal)

    def assert_verified(self) -> None:
        require_bound_detection_observation_geometry(self)


def _validate_detection_arrays(
    *,
    rowset_node: Any,
    key_node: Any,
    source_frame_index_node: Any,
    bbox_norm_node: Any,
    bbox_img_node: Any,
    centers_img_node: Any,
    frame_evidence: BoundDetectionFrameEvidence,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    for node, name in (
        (key_node, "instance_key"),
        (source_frame_index_node, "source_acquisition_frame_index"),
        (bbox_norm_node, "bbox_norm_coords"),
        (bbox_img_node, "bbox_img_xyxy"),
        (centers_img_node, "centers_img_xy"),
    ):
        _require_child_path(node, rowset_node, name)
    keys = _array(key_node, label="instance_key")
    source_frames = _array(
        source_frame_index_node,
        label="source_acquisition_frame_index",
    )
    bbox_norm = _array(bbox_norm_node, label="bbox_norm_coords")
    bbox_img = _array(bbox_img_node, label="bbox_img_xyxy")
    centers_img = _array(centers_img_node, label="centers_img_xy")
    if keys.dtype != np.dtype("<u8") or keys.ndim != 1:
        _fail("Canonical instance_key must be exact uint64 rank 1.")
    if source_frames.dtype != np.dtype("<i8") or source_frames.shape != keys.shape:
        _fail("source_acquisition_frame_index must be exact int64 and row-aligned.")
    if np.any(source_frames < 0):
        _fail("source_acquisition_frame_index cannot contain negative values.")
    if (
        bbox_norm.dtype.kind != "f"
        or bbox_norm.shape != (keys.shape[0], 4)
        or bbox_img.dtype != bbox_norm.dtype
        or bbox_img.shape != bbox_norm.shape
        or centers_img.dtype != bbox_norm.dtype
        or centers_img.shape != (keys.shape[0], 2)
    ):
        _fail(
            "Canonical bbox/center arrays must share one floating dtype and exact "
            "row-aligned (N,4)/(N,2) shapes."
        )
    expected_bbox, expected_centers = derive_detection_source_camera_geometry(
        bbox_norm,
        frame_evidence=frame_evidence,
    )
    if not np.array_equal(bbox_img, expected_bbox, equal_nan=True):
        _fail(
            "bbox_img_xyxy is not the exact dtype-preserving projection of "
            "bbox_norm_coords through the sealed normalized-to-camera frame."
        )
    if not np.array_equal(centers_img, expected_centers, equal_nan=True):
        _fail(
            "centers_img_xy is not the exact dtype-preserving midpoint of the "
            "persisted bbox_img_xyxy rows."
        )
    return keys, source_frames, bbox_norm, bbox_img, centers_img


def _verified_detection_source_lineage(
    records: tuple[BoundCoordinateRecord, ...],
    *,
    rowset_node: Any,
) -> tuple[BoundCoordinateRecord, ...]:
    if type(records) is not tuple:
        _fail("Detection source lineage must be an exact tuple of sealed records.")
    archive = archive_identity(rowset_node)
    verified = tuple(verify_bound_coordinate_record(item) for item in records)
    refs = tuple(item.record_ref for item in verified)
    if len(set(refs)) != len(refs):
        _fail("Detection source lineage cannot repeat a persisted record.")
    if any(item.archive_identity != archive for item in verified):
        _fail("Detection source lineage and output rowset use different archives.")
    return verified


def publish_detection_observation_geometry(
    rowset_node: Any,
    key_node: Any,
    source_frame_index_node: Any,
    bbox_norm_node: Any,
    bbox_img_node: Any,
    centers_img_node: Any,
    *,
    frame_evidence: BoundDetectionFrameEvidence,
    source_lineage_records: tuple[BoundCoordinateRecord, ...] = (),
) -> BoundDetectionObservationGeometry:
    """Publish canonical detection identity, time, bbox, and center semantics."""

    evidence = require_bound_detection_frame_evidence(frame_evidence)
    source_lineage = _verified_detection_source_lineage(
        source_lineage_records,
        rowset_node=rowset_node,
    )
    keys, _, bbox_norm, bbox_img, centers_img = _validate_detection_arrays(
        rowset_node=rowset_node,
        key_node=key_node,
        source_frame_index_node=source_frame_index_node,
        bbox_norm_node=bbox_norm_node,
        bbox_img_node=bbox_img_node,
        centers_img_node=centers_img_node,
        frame_evidence=evidence,
    )
    common_archive = archive_identity(rowset_node)
    if evidence.source_camera_frame.archive_identity != common_archive:
        _fail("Detection geometry and frame evidence use different archives.")
    for label, node in (
        ("instance_key", key_node),
        ("source_acquisition_frame_index", source_frame_index_node),
        ("bbox_norm_coords", bbox_norm_node),
        ("bbox_img_xyxy", bbox_img_node),
        ("centers_img_xy", centers_img_node),
    ):
        if archive_identity(node) != common_archive:
            _fail(f"{label} and its rowset use different archives.")
    contract = build_row_identity_contract(
        domain=OBSERVATION_INSTANCE_DOMAIN,
        values=keys,
    )
    attrs_targets, snapshots = _attrs_snapshots(
        rowset_node,
        key_node,
        source_frame_index_node,
        bbox_norm_node,
        bbox_img_node,
        centers_img_node,
    )
    try:
        identity = stamp_and_bind_row_identity_contract(
            rowset_node,
            key_node,
            contract=contract,
        )
        temporal = stamp_source_row_temporal_authority(
            rowset_node,
            source_frame_index_node,
            source_row_identity=identity,
            acquisition_frame=evidence.acquisition_frame,
        )
        projection_record, center_record = _detection_records(
            row_identity=identity,
            temporal=temporal,
            frame_evidence=evidence,
            bbox_norm_node=bbox_norm_node,
            bbox_norm=bbox_norm,
            bbox_img_node=bbox_img_node,
            bbox_img=bbox_img,
            centers_img_node=centers_img_node,
            centers_img=centers_img,
            source_lineage_records=source_lineage,
        )
        projection = stamp_and_bind_persisted_coordinate_record(
            rowset_node,
            projection_record,
            attr_name=DETECTION_BBOX_PROJECTION_ATTR,
        )
        center = stamp_and_bind_persisted_coordinate_record(
            rowset_node,
            center_record,
            attr_name=BBOX_CENTER_DERIVATION_ATTR,
        )
        bbox_normalized = build_bound_canonical_coordinate_descriptor(
            bbox_norm_node,
            profile_id=SOURCE_CAMERA_NORMALIZED_PROFILE_ID,
            geometry_type="bbox_cxcywh",
            components=("center_x", "center_y", "width", "height"),
            component_units=(
                "normalized",
                "normalized",
                "normalized",
                "normalized",
            ),
            pixel_convention="continuous",
            row_identity=identity,
            reference_frame_authority=evidence.normalized_frame,
            source_camera_overlay_status=CANONICAL_OVERLAY_REQUIRES_TRANSFORM,
            transform_chain=evidence.normalized_to_source_camera,
            lineage_records=(*source_lineage, projection),
        )
        bbox_image = build_bound_canonical_coordinate_descriptor(
            bbox_img_node,
            profile_id=SOURCE_CAMERA_PROFILE_ID,
            geometry_type="bbox_xyxy",
            components=("x_min", "y_min", "x_max", "y_max"),
            component_units=("px", "px", "px", "px"),
            pixel_convention=SOURCE_CAMERA_PIXEL_CONVENTION,
            row_identity=identity,
            reference_frame_authority=evidence.source_camera_frame,
            source_camera_overlay_status=CANONICAL_OVERLAY_DIRECT,
            lineage_records=(*source_lineage, projection),
        )
        centers_image = build_bound_canonical_coordinate_descriptor(
            centers_img_node,
            profile_id=SOURCE_CAMERA_PROFILE_ID,
            geometry_type="point_xy",
            components=("x", "y"),
            component_units=("px", "px"),
            pixel_convention=SOURCE_CAMERA_PIXEL_CONVENTION,
            row_identity=identity,
            reference_frame_authority=evidence.source_camera_frame,
            source_camera_overlay_status=CANONICAL_OVERLAY_DIRECT,
            lineage_records=(*source_lineage, projection, center),
        )
        stamp_bound_canonical_coordinate_descriptors(
            (bbox_normalized, bbox_image, centers_image)
        )
        return load_detection_observation_geometry(
            rowset_node,
            key_node,
            source_frame_index_node,
            bbox_norm_node,
            bbox_img_node,
            centers_img_node,
            frame_evidence=evidence,
            source_lineage_records=source_lineage,
        )
    except Exception as exc:
        _rollback_attrs(attrs_targets, snapshots, cause=exc)
        raise


def load_detection_observation_geometry(
    rowset_node: Any,
    key_node: Any,
    source_frame_index_node: Any,
    bbox_norm_node: Any,
    bbox_img_node: Any,
    centers_img_node: Any,
    *,
    frame_evidence: BoundDetectionFrameEvidence,
    source_lineage_records: tuple[BoundCoordinateRecord, ...] = (),
) -> BoundDetectionObservationGeometry:
    """Freshly revalidate one canonical detection geometry publication."""

    evidence = require_bound_detection_frame_evidence(frame_evidence)
    source_lineage = _verified_detection_source_lineage(
        source_lineage_records,
        rowset_node=rowset_node,
    )
    _, _, bbox_norm, bbox_img, centers_img = _validate_detection_arrays(
        rowset_node=rowset_node,
        key_node=key_node,
        source_frame_index_node=source_frame_index_node,
        bbox_norm_node=bbox_norm_node,
        bbox_img_node=bbox_img_node,
        centers_img_node=centers_img_node,
        frame_evidence=evidence,
    )
    identity = load_bound_row_identity_contract(rowset_node, key_node)
    temporal = load_bound_source_row_temporal_authority(
        rowset_node,
        source_frame_index_node,
        source_row_identity=identity,
        acquisition_frame=evidence.acquisition_frame,
    )
    projection = bind_persisted_coordinate_record(
        rowset_node,
        attr_name=DETECTION_BBOX_PROJECTION_ATTR,
    )
    center = bind_persisted_coordinate_record(
        rowset_node,
        attr_name=BBOX_CENTER_DERIVATION_ATTR,
    )
    expected_projection, expected_center = _detection_records(
        row_identity=identity,
        temporal=temporal,
        frame_evidence=evidence,
        bbox_norm_node=bbox_norm_node,
        bbox_norm=bbox_norm,
        bbox_img_node=bbox_img_node,
        bbox_img=bbox_img,
        centers_img_node=centers_img_node,
        centers_img=centers_img,
        source_lineage_records=source_lineage,
    )
    if projection.record != expected_projection:
        _fail(
            "Persisted detection bbox projection differs from exact live frame, "
            "transform, identity, temporal, or payload evidence."
        )
    if center.record != expected_center:
        _fail(
            "Persisted bbox-center derivation differs from exact live bbox, "
            "point, frame, or identity evidence."
        )
    bbox_normalized = load_bound_canonical_coordinate_descriptor(
        bbox_norm_node,
        row_identity=identity,
        reference_frame_authority=evidence.normalized_frame,
        transform_chain=evidence.normalized_to_source_camera,
        lineage_records=(*source_lineage, projection),
    )
    bbox_image = load_bound_canonical_coordinate_descriptor(
        bbox_img_node,
        row_identity=identity,
        reference_frame_authority=evidence.source_camera_frame,
        lineage_records=(*source_lineage, projection),
    )
    centers_image = load_bound_canonical_coordinate_descriptor(
        centers_img_node,
        row_identity=identity,
        reference_frame_authority=evidence.source_camera_frame,
        lineage_records=(*source_lineage, projection, center),
    )
    expected_profiles = (
        (
            bbox_normalized.descriptor.profile_id,
            bbox_normalized.descriptor.geometry_type,
        ),
        (bbox_image.descriptor.profile_id, bbox_image.descriptor.geometry_type),
        (
            centers_image.descriptor.profile_id,
            centers_image.descriptor.geometry_type,
        ),
    )
    if expected_profiles != (
        (SOURCE_CAMERA_NORMALIZED_PROFILE_ID, "bbox_cxcywh"),
        (SOURCE_CAMERA_PROFILE_ID, "bbox_xyxy"),
        (SOURCE_CAMERA_PROFILE_ID, "point_xy"),
    ):
        _fail("Detection coordinate descriptors use unsupported canonical profiles.")
    return BoundDetectionObservationGeometry(
        bbox_normalized=bbox_normalized,
        bbox_image=bbox_image,
        centers_image=centers_image,
        row_identity=identity,
        temporal_authority=temporal,
        frame_evidence=evidence,
        bbox_projection=projection,
        bbox_center_derivation=center,
        source_lineage_records=source_lineage,
        rowset_node=rowset_node,
        key_node=key_node,
        source_frame_index_node=source_frame_index_node,
        bbox_norm_node=bbox_norm_node,
        bbox_img_node=bbox_img_node,
        centers_img_node=centers_img_node,
        _verification_seal=_BOUND_DETECTION_GEOMETRY_SEAL,
    )


def require_bound_detection_observation_geometry(
    value: Any,
) -> BoundDetectionObservationGeometry:
    if (
        type(value) is not BoundDetectionObservationGeometry
        or value._seal is not _BOUND_DETECTION_GEOMETRY_SEAL
    ):
        _fail("A sealed canonical detection-geometry binding is required.")
    current = load_detection_observation_geometry(
        value._rowset_node,
        value._key_node,
        value._source_frame_index_node,
        value._bbox_norm_node,
        value._bbox_img_node,
        value._centers_img_node,
        frame_evidence=value.frame_evidence,
        source_lineage_records=value.source_lineage_records,
    )
    if (
        current.bbox_normalized.descriptor != value.bbox_normalized.descriptor
        or current.bbox_image.descriptor != value.bbox_image.descriptor
        or current.centers_image.descriptor != value.centers_image.descriptor
        or current.bbox_projection.record_sha256 != value.bbox_projection.record_sha256
        or current.bbox_center_derivation.record_sha256
        != value.bbox_center_derivation.record_sha256
        or current.temporal_authority.record_sha256
        != value.temporal_authority.record_sha256
    ):
        _fail("Canonical detection geometry changed after binding.")
    return value


def detection_observation_geometry_values(
    value: BoundDetectionObservationGeometry,
) -> dict[str, np.ndarray]:
    """Return defensive copies of one freshly verified detection rowset.

    This is the only value-level handoff used by crop writers.  Returning the
    exact persisted dtype/order avoids encouraging consumers to reconstruct
    pixels from dimensions or parsed descriptor dictionaries.
    """

    source = require_bound_detection_observation_geometry(value)
    keys, frames, bbox_norm, bbox_img, centers = _copy_source_arrays(source)
    return {
        "instance_key": np.array(keys, copy=True, order="C"),
        "source_acquisition_frame_index": np.array(
            frames,
            copy=True,
            order="C",
        ),
        "bbox_norm_coords": np.array(bbox_norm, copy=True, order="C"),
        "bbox_img_xyxy": np.array(bbox_img, copy=True, order="C"),
        "centers_img_xy": np.array(centers, copy=True, order="C"),
    }


def _copy_source_arrays(
    source: BoundDetectionObservationGeometry,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    keys = _array(source._key_node, label="source instance_key")
    frames = _array(
        source._source_frame_index_node,
        label="source source_acquisition_frame_index",
    )
    bbox_norm = _array(source._bbox_norm_node, label="source bbox_norm_coords")
    bbox_img = _array(source._bbox_img_node, label="source bbox_img_xyxy")
    centers = _array(source._centers_img_node, label="source centers_img_xy")
    return keys, frames, bbox_norm, bbox_img, centers


def _validate_crop_copy_arrays(
    *,
    rowset_node: Any,
    key_node: Any,
    source_row_index_node: Any,
    source_frame_index_node: Any,
    bbox_norm_node: Any,
    bbox_img_node: Any,
    centers_img_node: Any,
    source_geometry: BoundDetectionObservationGeometry,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    source = require_bound_detection_observation_geometry(source_geometry)
    for node, name in (
        (key_node, "instance_key"),
        (source_frame_index_node, "source_acquisition_frame_index"),
        (bbox_norm_node, "bbox_norm_coords"),
        (bbox_img_node, "bbox_img_xyxy"),
        (centers_img_node, "centers_img_xy"),
    ):
        _require_child_path(node, rowset_node, name)
    if not canonical_node_path(source_row_index_node).startswith(
        f"{canonical_node_path(rowset_node)}/"
    ):
        _fail("Crop source-row selection must be persisted under its exact rowset.")
    source_rows = _array(source_row_index_node, label="crop source_row_index")
    keys = _array(key_node, label="crop instance_key")
    frames = _array(
        source_frame_index_node,
        label="crop source_acquisition_frame_index",
    )
    bbox_norm = _array(bbox_norm_node, label="crop bbox_norm_coords")
    bbox_img = _array(bbox_img_node, label="crop bbox_img_xyxy")
    centers = _array(centers_img_node, label="crop centers_img_xy")
    if source_rows.dtype != np.dtype("<i8") or source_rows.ndim != 1:
        _fail("Crop source-row selection must be exact signed int64 rank 1.")
    if source_rows.size and int(np.unique(source_rows).shape[0]) != source_rows.size:
        _fail("Crop source-row selection must be one-to-one.")
    source_keys, source_frames, source_norm, source_img, source_centers = (
        _copy_source_arrays(source)
    )
    if np.any(source_rows < 0) or np.any(
        source_rows >= source.row_identity.leading_dimension
    ):
        _fail("Crop source-row selection resolves outside the exact source rowset.")
    selected_keys = source_keys[source_rows]
    expected_frames = resolve_source_acquisition_frame_indices(
        source.temporal_authority,
        source_rows,
    )
    if keys.dtype != np.dtype("<u8") or not np.array_equal(keys, selected_keys):
        _fail("Crop instance_key is not the exact selected/reordered source identity.")
    if frames.dtype != np.dtype("<i8") or not np.array_equal(
        frames,
        expected_frames,
    ):
        _fail(
            "Crop source_acquisition_frame_index is not the exact selected "
            "source temporal mapping."
        )
    expected_shapes = (
        (bbox_norm, source_norm, (source_rows.shape[0], 4), "bbox_norm_coords"),
        (bbox_img, source_img, (source_rows.shape[0], 4), "bbox_img_xyxy"),
        (centers, source_centers, (source_rows.shape[0], 2), "centers_img_xy"),
    )
    for output, source_values, expected_shape, label in expected_shapes:
        if (
            output.shape != expected_shape
            or output.dtype != source_values.dtype
            or not np.array_equal(output, source_values[source_rows], equal_nan=True)
        ):
            _fail(
                f"Crop {label} is not an exact dtype-preserving subset/reorder "
                "of the selected source surface."
            )
    return source_rows, keys, frames, bbox_norm, bbox_img, centers


def _crop_selection_record(
    *,
    source: BoundDetectionObservationGeometry,
    source_row_index_node: Any,
    source_rows: np.ndarray,
    row_identity: BoundRowIdentityContract,
    temporal: BoundSourceRowTemporalAuthority,
    key_node: Any,
    keys: np.ndarray,
    source_frame_index_node: Any,
    frames: np.ndarray,
    bbox_norm_node: Any,
    bbox_norm: np.ndarray,
    bbox_img_node: Any,
    bbox_img: np.ndarray,
    centers_img_node: Any,
    centers: np.ndarray,
) -> dict[str, Any]:
    source_keys, source_frames, source_norm, source_img, source_centers = (
        _copy_source_arrays(source)
    )
    return {
        "schema_id": CROP_GEOMETRY_SELECTION_SCHEMA_ID,
        "schema_version": CROP_GEOMETRY_SELECTION_SCHEMA_VERSION,
        "operation": CROP_GEOMETRY_SELECTION_OPERATION,
        "source_rowset": {
            "row_identity_ref": source.row_identity.record_ref,
            "row_identity_sha256": source.row_identity.record_sha256,
            "temporal_authority_ref": source.temporal_authority.record_ref,
            "temporal_authority_sha256": source.temporal_authority.record_sha256,
            "instance_key": _payload(source._key_node, source_keys),
            "source_acquisition_frame_index": _payload(
                source._source_frame_index_node,
                source_frames,
            ),
            "bbox_norm_coords": _payload(source._bbox_norm_node, source_norm),
            "bbox_img_xyxy": _payload(source._bbox_img_node, source_img),
            "centers_img_xy": _payload(source._centers_img_node, source_centers),
        },
        "selection": _payload(source_row_index_node, source_rows),
        "output_rowset": {
            "row_identity_ref": row_identity.record_ref,
            "row_identity_sha256": row_identity.record_sha256,
            "temporal_authority_ref": temporal.record_ref,
            "temporal_authority_sha256": temporal.record_sha256,
            "instance_key": _payload(key_node, keys),
            "source_acquisition_frame_index": _payload(
                source_frame_index_node,
                frames,
            ),
            "bbox_norm_coords": _payload(bbox_norm_node, bbox_norm),
            "bbox_img_xyxy": _payload(bbox_img_node, bbox_img),
            "centers_img_xy": _payload(centers_img_node, centers),
        },
    }


@dataclass(frozen=True, init=False)
class BoundCropObservationGeometry:
    """Canonical crop rows that exactly select/reorder detection geometry."""

    bbox_normalized: BoundCanonicalCoordinateDescriptor
    bbox_image: BoundCanonicalCoordinateDescriptor
    centers_image: BoundCanonicalCoordinateDescriptor
    row_identity: BoundRowIdentityContract = field(repr=False)
    temporal_authority: BoundSourceRowTemporalAuthority = field(repr=False)
    source_geometry: BoundDetectionObservationGeometry = field(repr=False)
    selection_derivation: BoundCoordinateRecord = field(repr=False)
    position_surface: BoundSourceCameraPositionSurface
    _rowset_node: Any = field(repr=False, compare=False)
    _key_node: Any = field(repr=False, compare=False)
    _source_row_index_node: Any = field(repr=False, compare=False)
    _source_frame_index_node: Any = field(repr=False, compare=False)
    _bbox_norm_node: Any = field(repr=False, compare=False)
    _bbox_img_node: Any = field(repr=False, compare=False)
    _centers_img_node: Any = field(repr=False, compare=False)
    _seal: object = field(repr=False, compare=False)

    def __init__(
        self,
        *,
        bbox_normalized: BoundCanonicalCoordinateDescriptor,
        bbox_image: BoundCanonicalCoordinateDescriptor,
        centers_image: BoundCanonicalCoordinateDescriptor,
        row_identity: BoundRowIdentityContract,
        temporal_authority: BoundSourceRowTemporalAuthority,
        source_geometry: BoundDetectionObservationGeometry,
        selection_derivation: BoundCoordinateRecord,
        rowset_node: Any,
        key_node: Any,
        source_row_index_node: Any,
        source_frame_index_node: Any,
        bbox_norm_node: Any,
        bbox_img_node: Any,
        centers_img_node: Any,
        _verification_seal: object | None = None,
    ) -> None:
        if _verification_seal is not _BOUND_CROP_GEOMETRY_SEAL:
            _fail("Crop geometry must be created by the canonical loader.")
        object.__setattr__(self, "bbox_normalized", bbox_normalized)
        object.__setattr__(self, "bbox_image", bbox_image)
        object.__setattr__(self, "centers_image", centers_image)
        object.__setattr__(self, "row_identity", row_identity)
        object.__setattr__(self, "temporal_authority", temporal_authority)
        object.__setattr__(self, "source_geometry", source_geometry)
        object.__setattr__(self, "selection_derivation", selection_derivation)
        object.__setattr__(
            self,
            "position_surface",
            _position_surface(centers_image, temporal_authority),
        )
        object.__setattr__(self, "_rowset_node", rowset_node)
        object.__setattr__(self, "_key_node", key_node)
        object.__setattr__(self, "_source_row_index_node", source_row_index_node)
        object.__setattr__(self, "_source_frame_index_node", source_frame_index_node)
        object.__setattr__(self, "_bbox_norm_node", bbox_norm_node)
        object.__setattr__(self, "_bbox_img_node", bbox_img_node)
        object.__setattr__(self, "_centers_img_node", centers_img_node)
        object.__setattr__(self, "_seal", _verification_seal)

    def assert_verified(self) -> None:
        require_bound_crop_observation_geometry(self)


def _copied_descriptor(
    output_node: Any,
    *,
    source: BoundCanonicalCoordinateDescriptor,
    row_identity: BoundRowIdentityContract,
    selection: BoundCoordinateRecord,
) -> BoundCanonicalCoordinateDescriptor:
    source = require_bound_canonical_coordinate_descriptor(source)
    descriptor = source.descriptor
    return build_bound_canonical_coordinate_descriptor(
        output_node,
        profile_id=descriptor.profile_id,
        geometry_type=descriptor.geometry_type,
        components=descriptor.components,
        component_units=descriptor.component_units,
        pixel_convention=descriptor.pixel_convention,
        row_identity=row_identity,
        reference_extent=source.reference_extent,
        reference_frame_authority=source.reference_frame_authority,
        source_camera_overlay_status=descriptor.source_camera_overlay.status,
        transform_chain=source.transform_chain,
        lineage_records=(*source.lineage_records, selection),
        frame_record=source.frame_record,
    )


def publish_crop_observation_geometry(
    rowset_node: Any,
    key_node: Any,
    source_row_index_node: Any,
    source_frame_index_node: Any,
    bbox_norm_node: Any,
    bbox_img_node: Any,
    centers_img_node: Any,
    *,
    source_geometry: BoundDetectionObservationGeometry,
) -> BoundCropObservationGeometry:
    """Publish exact source-camera geometry copied into canonical crop rows."""

    source = require_bound_detection_observation_geometry(source_geometry)
    source_rows, keys, frames, bbox_norm, bbox_img, centers = (
        _validate_crop_copy_arrays(
            rowset_node=rowset_node,
            key_node=key_node,
            source_row_index_node=source_row_index_node,
            source_frame_index_node=source_frame_index_node,
            bbox_norm_node=bbox_norm_node,
            bbox_img_node=bbox_img_node,
            centers_img_node=centers_img_node,
            source_geometry=source,
        )
    )
    common_archive = archive_identity(rowset_node)
    if source.row_identity.archive_identity != common_archive:
        _fail("Crop output and selected detection geometry use different archives.")
    for node in (
        key_node,
        source_row_index_node,
        source_frame_index_node,
        bbox_norm_node,
        bbox_img_node,
        centers_img_node,
    ):
        if archive_identity(node) != common_archive:
            _fail("Crop coordinate publication spans different archives.")
    contract = build_row_identity_contract(
        domain=OBSERVATION_INSTANCE_DOMAIN,
        values=keys,
    )
    attrs_targets, snapshots = _attrs_snapshots(
        rowset_node,
        key_node,
        source_row_index_node,
        source_frame_index_node,
        bbox_norm_node,
        bbox_img_node,
        centers_img_node,
    )
    try:
        identity = stamp_and_bind_row_identity_contract(
            rowset_node,
            key_node,
            contract=contract,
        )
        temporal = stamp_source_row_temporal_authority(
            rowset_node,
            source_frame_index_node,
            source_row_identity=identity,
            acquisition_frame=source.frame_evidence.acquisition_frame,
        )
        selection_record = _crop_selection_record(
            source=source,
            source_row_index_node=source_row_index_node,
            source_rows=source_rows,
            row_identity=identity,
            temporal=temporal,
            key_node=key_node,
            keys=keys,
            source_frame_index_node=source_frame_index_node,
            frames=frames,
            bbox_norm_node=bbox_norm_node,
            bbox_norm=bbox_norm,
            bbox_img_node=bbox_img_node,
            bbox_img=bbox_img,
            centers_img_node=centers_img_node,
            centers=centers,
        )
        selection = stamp_and_bind_persisted_coordinate_record(
            rowset_node,
            selection_record,
            attr_name=CROP_GEOMETRY_SELECTION_ATTR,
        )
        bindings = (
            _copied_descriptor(
                bbox_norm_node,
                source=source.bbox_normalized,
                row_identity=identity,
                selection=selection,
            ),
            _copied_descriptor(
                bbox_img_node,
                source=source.bbox_image,
                row_identity=identity,
                selection=selection,
            ),
            _copied_descriptor(
                centers_img_node,
                source=source.centers_image,
                row_identity=identity,
                selection=selection,
            ),
        )
        stamp_bound_canonical_coordinate_descriptors(bindings)
        return load_crop_observation_geometry(
            rowset_node,
            key_node,
            source_row_index_node,
            source_frame_index_node,
            bbox_norm_node,
            bbox_img_node,
            centers_img_node,
            source_geometry=source,
        )
    except Exception as exc:
        _rollback_attrs(attrs_targets, snapshots, cause=exc)
        raise


def load_crop_observation_geometry(
    rowset_node: Any,
    key_node: Any,
    source_row_index_node: Any,
    source_frame_index_node: Any,
    bbox_norm_node: Any,
    bbox_img_node: Any,
    centers_img_node: Any,
    *,
    source_geometry: BoundDetectionObservationGeometry,
) -> BoundCropObservationGeometry:
    """Freshly verify a crop's exact instance-key geometry selection."""

    source = require_bound_detection_observation_geometry(source_geometry)
    source_rows, keys, frames, bbox_norm, bbox_img, centers = (
        _validate_crop_copy_arrays(
            rowset_node=rowset_node,
            key_node=key_node,
            source_row_index_node=source_row_index_node,
            source_frame_index_node=source_frame_index_node,
            bbox_norm_node=bbox_norm_node,
            bbox_img_node=bbox_img_node,
            centers_img_node=centers_img_node,
            source_geometry=source,
        )
    )
    identity = load_bound_row_identity_contract(rowset_node, key_node)
    temporal = load_bound_source_row_temporal_authority(
        rowset_node,
        source_frame_index_node,
        source_row_identity=identity,
        acquisition_frame=source.frame_evidence.acquisition_frame,
    )
    selection = bind_persisted_coordinate_record(
        rowset_node,
        attr_name=CROP_GEOMETRY_SELECTION_ATTR,
    )
    expected = _crop_selection_record(
        source=source,
        source_row_index_node=source_row_index_node,
        source_rows=source_rows,
        row_identity=identity,
        temporal=temporal,
        key_node=key_node,
        keys=keys,
        source_frame_index_node=source_frame_index_node,
        frames=frames,
        bbox_norm_node=bbox_norm_node,
        bbox_norm=bbox_norm,
        bbox_img_node=bbox_img_node,
        bbox_img=bbox_img,
        centers_img_node=centers_img_node,
        centers=centers,
    )
    if selection.record != expected:
        _fail(
            "Persisted crop selection differs from exact source/output arrays, "
            "instance keys, temporal mapping, or row ordering."
        )
    bbox_normalized = load_bound_canonical_coordinate_descriptor(
        bbox_norm_node,
        row_identity=identity,
        reference_frame_authority=source.frame_evidence.normalized_frame,
        transform_chain=source.frame_evidence.normalized_to_source_camera,
        lineage_records=(*source.bbox_normalized.lineage_records, selection),
    )
    bbox_image = load_bound_canonical_coordinate_descriptor(
        bbox_img_node,
        row_identity=identity,
        reference_frame_authority=source.frame_evidence.source_camera_frame,
        lineage_records=(*source.bbox_image.lineage_records, selection),
    )
    centers_image = load_bound_canonical_coordinate_descriptor(
        centers_img_node,
        row_identity=identity,
        reference_frame_authority=source.frame_evidence.source_camera_frame,
        lineage_records=(*source.centers_image.lineage_records, selection),
    )
    for output, selected in (
        (bbox_normalized.descriptor, source.bbox_normalized.descriptor),
        (bbox_image.descriptor, source.bbox_image.descriptor),
        (centers_image.descriptor, source.centers_image.descriptor),
    ):
        comparable_output = (
            output.profile_id,
            output.space_id,
            output.geometry_type,
            output.components,
            output.component_units,
            output.origin,
            output.positive_directions,
            output.reference_extent,
            output.pixel_convention,
            output.source_camera_overlay,
            output.frame_record,
        )
        comparable_source = (
            selected.profile_id,
            selected.space_id,
            selected.geometry_type,
            selected.components,
            selected.component_units,
            selected.origin,
            selected.positive_directions,
            selected.reference_extent,
            selected.pixel_convention,
            selected.source_camera_overlay,
            selected.frame_record,
        )
        if comparable_output != comparable_source:
            _fail("Crop coordinate semantics differ from the exact selected source.")
    return BoundCropObservationGeometry(
        bbox_normalized=bbox_normalized,
        bbox_image=bbox_image,
        centers_image=centers_image,
        row_identity=identity,
        temporal_authority=temporal,
        source_geometry=source,
        selection_derivation=selection,
        rowset_node=rowset_node,
        key_node=key_node,
        source_row_index_node=source_row_index_node,
        source_frame_index_node=source_frame_index_node,
        bbox_norm_node=bbox_norm_node,
        bbox_img_node=bbox_img_node,
        centers_img_node=centers_img_node,
        _verification_seal=_BOUND_CROP_GEOMETRY_SEAL,
    )


def require_bound_crop_observation_geometry(
    value: Any,
) -> BoundCropObservationGeometry:
    if (
        type(value) is not BoundCropObservationGeometry
        or value._seal is not _BOUND_CROP_GEOMETRY_SEAL
    ):
        _fail("A sealed canonical crop-geometry binding is required.")
    current = load_crop_observation_geometry(
        value._rowset_node,
        value._key_node,
        value._source_row_index_node,
        value._source_frame_index_node,
        value._bbox_norm_node,
        value._bbox_img_node,
        value._centers_img_node,
        source_geometry=value.source_geometry,
    )
    if (
        current.selection_derivation.record_sha256
        != value.selection_derivation.record_sha256
        or current.centers_image.descriptor != value.centers_image.descriptor
        or current.temporal_authority.record_sha256
        != value.temporal_authority.record_sha256
    ):
        _fail("Canonical crop geometry changed after binding.")
    return value


@dataclass(frozen=True)
class CropRoiGeometryPublicationResult:
    """Canonical crop placement and ROI-local bbox descriptors."""

    source_crop_xywh: BoundCanonicalCoordinateDescriptor
    bbox_roi_xyxy: BoundCanonicalCoordinateDescriptor
    derivation: BoundCoordinateRecord


def _validate_crop_roi_evidence(
    *,
    crop_geometry: BoundCropObservationGeometry,
    source_crop_xywh_node: Any,
    bbox_roi_xyxy_node: Any,
    crop_placement_ownership: BoundCropPlacementOwnership,
    roi_frame: BoundPixelFrameAuthority,
    roi_to_source_camera: BoundDirectedTransformChain,
) -> tuple[
    BoundCropObservationGeometry,
    BoundCropPlacementOwnership,
    BoundPixelFrameAuthority,
    BoundDirectedTransformChain,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    crop = require_bound_crop_observation_geometry(crop_geometry)
    ownership = require_bound_crop_placement_ownership(crop_placement_ownership)
    roi = require_roi_pixel_frame_authority(roi_frame)
    chain = require_bound_directed_transform_chain(roi_to_source_camera)
    if roi.record.kind != ROI_FRAME_KIND or roi.pixel_convention != "continuous":
        _fail("Canonical ROI bbox geometry requires a continuous crop-ROI frame.")
    if not _same_row_identity(ownership.row_identity, crop.row_identity):
        _fail("Crop placement ownership uses a different observation identity.")
    if not _same_row_identity(roi.row_identity, crop.row_identity):
        _fail("ROI frame uses a different observation identity.")
    if not _same_pixel_frame(
        ownership.source_camera_frame,
        crop.source_geometry.frame_evidence.source_camera_frame,
    ):
        _fail("Crop placement targets a different source-camera frame.")
    if (
        not _same_pixel_frame(chain.descriptor_frame_authority, roi)
        or not _same_pixel_frame(
            chain.source_camera_frame_authority,
            ownership.source_camera_frame,
        )
        or chain.row_identity is None
        or not _same_row_identity(chain.row_identity, crop.row_identity)
    ):
        _fail("ROI-to-camera transform chain has wrong direction or row identity.")
    _require_child_path(source_crop_xywh_node, crop._rowset_node, "source_crop_xywh")
    _require_child_path(bbox_roi_xyxy_node, crop._rowset_node, "bbox_roi_xyxy")
    if canonical_node_path(ownership._placement_node) != canonical_node_path(
        source_crop_xywh_node
    ):
        _fail("Crop placement ownership does not bind exact source_crop_xywh bytes.")
    placement = _array(source_crop_xywh_node, label="source_crop_xywh")
    bbox_roi = _array(bbox_roi_xyxy_node, label="bbox_roi_xyxy")
    bbox_img = _array(crop._bbox_img_node, label="bbox_img_xyxy")
    if (
        placement.dtype.kind not in {"i", "u", "f"}
        or placement.shape != (crop.row_identity.leading_dimension, 4)
        or bbox_roi.dtype != bbox_img.dtype
        or bbox_roi.shape != bbox_img.shape
    ):
        _fail("Crop placement/ROI bbox arrays have incompatible dtype or shape.")
    roi_corners = bbox_roi.reshape(bbox_roi.shape[0], 2, 2)
    projected = np.asarray(
        apply_bound_directed_transform_chain(
            roi_corners,
            chain,
            row_identity=crop.row_identity,
        ),
        dtype=bbox_img.dtype,
    ).reshape(bbox_img.shape)
    if not np.array_equal(projected, bbox_img, equal_nan=True):
        _fail(
            "bbox_roi_xyxy does not project exactly to bbox_img_xyxy through "
            "the direction-labelled crop-placement chain."
        )
    return crop, ownership, roi, chain, placement, bbox_roi, bbox_img


def _crop_roi_record(
    *,
    crop: BoundCropObservationGeometry,
    ownership: BoundCropPlacementOwnership,
    roi: BoundPixelFrameAuthority,
    chain: BoundDirectedTransformChain,
    source_crop_xywh_node: Any,
    placement: np.ndarray,
    bbox_roi_xyxy_node: Any,
    bbox_roi: np.ndarray,
    bbox_img: np.ndarray,
) -> dict[str, Any]:
    return {
        "schema_id": CROP_ROI_GEOMETRY_DERIVATION_SCHEMA_ID,
        "schema_version": CROP_ROI_GEOMETRY_DERIVATION_SCHEMA_VERSION,
        "operation": CROP_ROI_GEOMETRY_DERIVATION_OPERATION,
        "source_crop_xywh": _payload(source_crop_xywh_node, placement),
        "crop_placement_ownership": {
            "record_ref": ownership.record_ref,
            "record_sha256": ownership.record_sha256,
        },
        "roi_frame": {
            "record_ref": roi.record_ref,
            "record_sha256": roi.record_sha256,
        },
        "direction": "roi_local_px_to_source_camera_image_px",
        "transform_chain": [
            {
                "record_ref": item.record_ref,
                "record_sha256": item.record_sha256,
            }
            for item in chain.transform_records
        ],
        "bbox_roi_xyxy": _payload(bbox_roi_xyxy_node, bbox_roi),
        "bbox_img_xyxy": _payload(crop._bbox_img_node, bbox_img),
        "row_identity": {
            "record_ref": crop.row_identity.record_ref,
            "record_sha256": crop.row_identity.record_sha256,
        },
        "formula": "apply_exact_rowwise_crop_placement_to_each_xyxy_corner_v1",
    }


def publish_crop_roi_geometry(
    source_crop_xywh_node: Any,
    bbox_roi_xyxy_node: Any,
    *,
    crop_geometry: BoundCropObservationGeometry,
    crop_placement_ownership: BoundCropPlacementOwnership,
    roi_frame: BoundPixelFrameAuthority,
    roi_to_source_camera: BoundDirectedTransformChain,
) -> CropRoiGeometryPublicationResult:
    """Publish ROI-local geometry only through exact crop-placement lineage."""

    crop, ownership, roi, chain, placement, bbox_roi, bbox_img = (
        _validate_crop_roi_evidence(
            crop_geometry=crop_geometry,
            source_crop_xywh_node=source_crop_xywh_node,
            bbox_roi_xyxy_node=bbox_roi_xyxy_node,
            crop_placement_ownership=crop_placement_ownership,
            roi_frame=roi_frame,
            roi_to_source_camera=roi_to_source_camera,
        )
    )
    attrs_targets, snapshots = _attrs_snapshots(
        crop._rowset_node,
        source_crop_xywh_node,
        bbox_roi_xyxy_node,
    )
    try:
        record = _crop_roi_record(
            crop=crop,
            ownership=ownership,
            roi=roi,
            chain=chain,
            source_crop_xywh_node=source_crop_xywh_node,
            placement=placement,
            bbox_roi_xyxy_node=bbox_roi_xyxy_node,
            bbox_roi=bbox_roi,
            bbox_img=bbox_img,
        )
        derivation = stamp_and_bind_persisted_coordinate_record(
            crop._rowset_node,
            record,
            attr_name=CROP_ROI_GEOMETRY_DERIVATION_ATTR,
        )
        source_crop = build_bound_canonical_coordinate_descriptor(
            source_crop_xywh_node,
            profile_id=SOURCE_CAMERA_PROFILE_ID,
            geometry_type="bbox_xywh",
            components=("x", "y", "width", "height"),
            component_units=("px", "px", "px", "px"),
            pixel_convention=SOURCE_CAMERA_PIXEL_CONVENTION,
            row_identity=crop.row_identity,
            reference_frame_authority=ownership.source_camera_frame,
            source_camera_overlay_status=CANONICAL_OVERLAY_DIRECT,
            lineage_records=(crop.selection_derivation, derivation),
        )
        bbox_roi_binding = build_bound_canonical_coordinate_descriptor(
            bbox_roi_xyxy_node,
            profile_id="roi_local_px.top_left_y_down.v1",
            geometry_type="bbox_xyxy",
            components=("x_min", "y_min", "x_max", "y_max"),
            component_units=("px", "px", "px", "px"),
            pixel_convention="continuous",
            row_identity=crop.row_identity,
            reference_frame_authority=roi,
            source_camera_overlay_status=CANONICAL_OVERLAY_REQUIRES_TRANSFORM,
            transform_chain=chain,
            lineage_records=(crop.selection_derivation, derivation),
        )
        stamp_bound_canonical_coordinate_descriptors((source_crop, bbox_roi_binding))
        return load_crop_roi_geometry(
            source_crop_xywh_node,
            bbox_roi_xyxy_node,
            crop_geometry=crop,
            crop_placement_ownership=ownership,
            roi_frame=roi,
            roi_to_source_camera=chain,
        )
    except Exception as exc:
        _rollback_attrs(attrs_targets, snapshots, cause=exc)
        raise


def load_crop_roi_geometry(
    source_crop_xywh_node: Any,
    bbox_roi_xyxy_node: Any,
    *,
    crop_geometry: BoundCropObservationGeometry,
    crop_placement_ownership: BoundCropPlacementOwnership,
    roi_frame: BoundPixelFrameAuthority,
    roi_to_source_camera: BoundDirectedTransformChain,
) -> CropRoiGeometryPublicationResult:
    """Freshly verify source-camera crop placement and ROI-local bbox metadata."""

    crop, ownership, roi, chain, placement, bbox_roi, bbox_img = (
        _validate_crop_roi_evidence(
            crop_geometry=crop_geometry,
            source_crop_xywh_node=source_crop_xywh_node,
            bbox_roi_xyxy_node=bbox_roi_xyxy_node,
            crop_placement_ownership=crop_placement_ownership,
            roi_frame=roi_frame,
            roi_to_source_camera=roi_to_source_camera,
        )
    )
    derivation = bind_persisted_coordinate_record(
        crop._rowset_node,
        attr_name=CROP_ROI_GEOMETRY_DERIVATION_ATTR,
    )
    expected = _crop_roi_record(
        crop=crop,
        ownership=ownership,
        roi=roi,
        chain=chain,
        source_crop_xywh_node=source_crop_xywh_node,
        placement=placement,
        bbox_roi_xyxy_node=bbox_roi_xyxy_node,
        bbox_roi=bbox_roi,
        bbox_img=bbox_img,
    )
    if derivation.record != expected:
        _fail("Persisted crop ROI derivation differs from exact live placement.")
    source_crop = load_bound_canonical_coordinate_descriptor(
        source_crop_xywh_node,
        row_identity=crop.row_identity,
        reference_frame_authority=ownership.source_camera_frame,
        lineage_records=(crop.selection_derivation, derivation),
    )
    bbox_roi_binding = load_bound_canonical_coordinate_descriptor(
        bbox_roi_xyxy_node,
        row_identity=crop.row_identity,
        reference_frame_authority=roi,
        transform_chain=chain,
        lineage_records=(crop.selection_derivation, derivation),
    )
    if (
        source_crop.descriptor.profile_id != SOURCE_CAMERA_PROFILE_ID
        or source_crop.descriptor.geometry_type != "bbox_xywh"
        or bbox_roi_binding.descriptor.profile_id != "roi_local_px.top_left_y_down.v1"
        or bbox_roi_binding.descriptor.geometry_type != "bbox_xyxy"
    ):
        _fail("Crop ROI coordinate descriptors use unsupported profiles.")
    return CropRoiGeometryPublicationResult(
        source_crop_xywh=source_crop,
        bbox_roi_xyxy=bbox_roi_binding,
        derivation=derivation,
    )


def _persisted_node(root_node: Any, path: str, *, label: str) -> Any:
    normalized = str(path).strip().strip("/")
    if not normalized or any(part in {"", ".", ".."} for part in normalized.split("/")):
        _fail(f"{label} path is not canonical.")
    try:
        node = root_node[normalized]
    except Exception as exc:
        _fail(f"Unable to open persisted {label} at {normalized!r}: {exc}.")
    if canonical_node_path(node) != normalized:
        _fail(f"Persisted {label} resolved to an unexpected path.")
    return node


def _require_detection_acquisition_mapping(
    rowset: Any,
    mapping: BoundCoordinateRecord,
    *,
    acquisition: BoundAcquisitionCameraFrame,
) -> None:
    base = canonical_node_path(rowset)
    try:
        decode_node = rowset["frame_indices"]
        source_node = rowset["source_acquisition_frame_index"]
    except Exception as exc:
        _fail(f"Detection acquisition mapping arrays are unavailable: {exc}.")
    if canonical_node_path(decode_node) != f"{base}/frame_indices" or (
        canonical_node_path(source_node) != f"{base}/source_acquisition_frame_index"
    ):
        _fail("Detection acquisition mapping arrays resolved to unexpected paths.")
    decode = _array(decode_node, label="detection decode frame index")
    source = _array(source_node, label="detection acquisition frame index")
    if (
        decode.dtype.kind not in "iu"
        or source.dtype != np.dtype("<i8")
        or decode.shape != source.shape
        or not np.array_equal(decode.astype(np.int64), source)
    ):
        _fail("Persisted detection decode-to-acquisition mapping is not identity.")
    metadata = acquisition.record.source_video_metadata
    expected = {
        "schema_id": DETECTION_ACQUISITION_MAPPING_SCHEMA_ID,
        "schema_version": DETECTION_ACQUISITION_MAPPING_SCHEMA_VERSION,
        "operation": "full_untrimmed_video_decode_identity_to_acquisition_v1",
        "direction": "decode_frame_index_to_source_acquisition_frame_index",
        "decode_frame_index": _payload(decode_node, decode),
        "source_acquisition_frame_index": _payload(source_node, source),
        "acquisition_camera_frame": {
            "record_ref": acquisition.record_ref,
            "record_sha256": acquisition.record_sha256,
        },
        "source_video_locator": metadata["locator"],
        "source_video_fingerprint": metadata["file_fingerprint"],
        "source_total_frames": int(acquisition.record.source_total_frames),
        "proof": "exact_locator_and_stat_fingerprint_revalidated_after_full_decode",
    }
    if mapping.record != expected:
        differing = sorted(
            name
            for name in set(mapping.record) | set(expected)
            if mapping.record.get(name) != expected.get(name)
        )
        _fail(
            "Persisted detection acquisition mapping differs from the exact live "
            f"arrays or acquisition authority for {base!r}; fields={differing!r}."
        )


def load_persisted_detection_observation_geometry(
    root_node: Any,
    rowset_path: str,
) -> BoundDetectionObservationGeometry:
    """Resolve a canonical detection geometry from persisted refs and nodes only."""

    rowset = _persisted_node(root_node, rowset_path, label="detection rowset")
    attrs = require_trusted_coordinate_attrs(rowset, label="Detection rowset")
    if attrs.get("coordinate_contract") != "canonical_v2":
        _fail("Selected detection rowset is not explicitly canonical_v2.")
    _, acquisition = load_persisted_acquisition_camera_authority(root_node)
    camera_id = acquisition.record.camera_id
    camera_node = _persisted_node(
        root_node,
        f"analysis/coordinate_frames/source_camera/{camera_id}/continuous",
        label="source-camera frame authority",
    )
    camera = load_source_camera_pixel_frame_authority(
        camera_node,
        acquisition_frame=acquisition,
    )
    normalized_node = _persisted_node(
        root_node,
        f"{canonical_node_path(rowset)}/coordinate_frames/source_camera_normalized",
        label="detection normalized frame",
    )
    normalized = load_normalized_pixel_frame_authority(
        normalized_node,
        pixel_frame=camera,
    )
    matrix_node = _persisted_node(
        root_node,
        (
            f"{canonical_node_path(rowset)}/coordinate_transforms/"
            "source_camera_normalized_to_image"
        ),
        label="normalized-to-camera transform",
    )
    authority_node = _persisted_node(
        root_node,
        (
            f"{canonical_node_path(rowset)}/coordinate_transforms/"
            "source_camera_normalized_to_image_authority"
        ),
        label="normalized-to-camera transform authority",
    )
    authority = load_bound_transform_authority(
        authority_node,
        payload_node=matrix_node,
        source_frame=normalized,
        target_frame=camera,
    )
    transform = load_bound_directed_transform_v2(
        matrix_node,
        authority=authority,
        source_frame=normalized,
        target_frame=camera,
    )
    evidence = build_bound_detection_frame_evidence(
        source_camera_frame=camera,
        normalized_frame=normalized,
        normalized_to_source_camera=resolve_bound_directed_transform_chain(
            (transform,)
        ),
    )
    mapping = bind_persisted_coordinate_record(
        rowset,
        attr_name=DETECTION_ACQUISITION_MAPPING_ATTR,
    )
    _require_detection_acquisition_mapping(
        rowset,
        mapping,
        acquisition=acquisition,
    )
    return load_detection_observation_geometry(
        rowset,
        _persisted_node(
            root_node,
            f"{canonical_node_path(rowset)}/instance_key",
            label="detection instance_key",
        ),
        _persisted_node(
            root_node,
            f"{canonical_node_path(rowset)}/source_acquisition_frame_index",
            label="detection acquisition frame",
        ),
        _persisted_node(
            root_node,
            f"{canonical_node_path(rowset)}/bbox_norm_coords",
            label="detection normalized bbox",
        ),
        _persisted_node(
            root_node,
            f"{canonical_node_path(rowset)}/bbox_img_xyxy",
            label="detection image bbox",
        ),
        _persisted_node(
            root_node,
            f"{canonical_node_path(rowset)}/centers_img_xy",
            label="detection image centers",
        ),
        frame_evidence=evidence,
        source_lineage_records=(mapping,),
    )


def load_persisted_crop_observation_geometry(
    root_node: Any,
    rowset_path: str,
) -> BoundCropObservationGeometry:
    """Resolve a canonical crop and its exact selected detection from disk."""

    rowset = _persisted_node(root_node, rowset_path, label="crop rowset")
    attrs = require_trusted_coordinate_attrs(rowset, label="Crop rowset")
    if attrs.get("coordinate_contract") != "canonical_v2":
        _fail("Selected crop rowset is not explicitly canonical_v2.")
    selection = bind_persisted_coordinate_record(
        rowset,
        attr_name=CROP_GEOMETRY_SELECTION_ATTR,
    )
    source_key = selection.record.get("source_rowset", {}).get("instance_key")
    source_ref = (
        source_key.get("array_ref") if isinstance(source_key, Mapping) else None
    )
    suffix = "/instance_key"
    if (
        not isinstance(source_ref, str)
        or not source_ref.startswith("/")
        or not source_ref.endswith(suffix)
    ):
        _fail("Crop selection does not identify one exact source instance_key array.")
    source_path = source_ref[1 : -len(suffix)]
    source = load_persisted_detection_observation_geometry(root_node, source_path)
    base = canonical_node_path(rowset)
    return load_crop_observation_geometry(
        rowset,
        _persisted_node(root_node, f"{base}/instance_key", label="crop instance_key"),
        _persisted_node(
            root_node,
            f"{base}/detection_indices",
            label="crop source-row selection",
        ),
        _persisted_node(
            root_node,
            f"{base}/source_acquisition_frame_index",
            label="crop acquisition frame",
        ),
        _persisted_node(
            root_node,
            f"{base}/bbox_norm_coords",
            label="crop normalized bbox",
        ),
        _persisted_node(
            root_node,
            f"{base}/bbox_img_xyxy",
            label="crop image bbox",
        ),
        _persisted_node(
            root_node,
            f"{base}/centers_img_xy",
            label="crop image centers",
        ),
        source_geometry=source,
    )


def load_persisted_source_camera_position_surface(
    root_node: Any,
    rowset_path: str,
) -> BoundSourceCameraPositionSurface:
    """Track-facing resolver for one canonical detection or crop rowset."""

    rowset = _persisted_node(root_node, rowset_path, label="position rowset")
    attrs = require_trusted_coordinate_attrs(rowset, label="Position rowset")
    has_mapping = DETECTION_ACQUISITION_MAPPING_ATTR in attrs
    has_selection = CROP_GEOMETRY_SELECTION_ATTR in attrs
    if has_mapping == has_selection:
        _fail(
            "Canonical position rowset must declare exactly one detection-mapping "
            "or crop-selection lineage."
        )
    geometry = (
        load_persisted_detection_observation_geometry(root_node, rowset_path)
        if has_mapping
        else load_persisted_crop_observation_geometry(root_node, rowset_path)
    )
    return require_bound_source_camera_position_surface(geometry.position_surface)


__all__ = [
    "BBOX_CENTER_DERIVATION_ATTR",
    "BBOX_CENTER_DERIVATION_OPERATION",
    "BBOX_CENTER_DERIVATION_SCHEMA_ID",
    "BBOX_CENTER_DERIVATION_SCHEMA_VERSION",
    "DETECTION_BBOX_PROJECTION_ATTR",
    "DETECTION_BBOX_PROJECTION_OPERATION",
    "DETECTION_BBOX_PROJECTION_SCHEMA_ID",
    "DETECTION_BBOX_PROJECTION_SCHEMA_VERSION",
    "DETECTION_ACQUISITION_MAPPING_ATTR",
    "DETECTION_ACQUISITION_MAPPING_SCHEMA_ID",
    "DETECTION_ACQUISITION_MAPPING_SCHEMA_VERSION",
    "CROP_GEOMETRY_SELECTION_ATTR",
    "CROP_GEOMETRY_SELECTION_OPERATION",
    "CROP_GEOMETRY_SELECTION_SCHEMA_ID",
    "CROP_GEOMETRY_SELECTION_SCHEMA_VERSION",
    "CROP_ROI_GEOMETRY_DERIVATION_ATTR",
    "CROP_ROI_GEOMETRY_DERIVATION_OPERATION",
    "CROP_ROI_GEOMETRY_DERIVATION_SCHEMA_ID",
    "CROP_ROI_GEOMETRY_DERIVATION_SCHEMA_VERSION",
    "BoundCropObservationGeometry",
    "BoundDetectionFrameEvidence",
    "BoundDetectionObservationGeometry",
    "BoundSourceCameraPositionSurface",
    "CropRoiGeometryPublicationResult",
    "ObservationCoordinatePublicationError",
    "build_bound_detection_frame_evidence",
    "derive_detection_source_camera_geometry",
    "detection_observation_geometry_values",
    "load_crop_observation_geometry",
    "load_crop_roi_geometry",
    "load_detection_observation_geometry",
    "load_persisted_crop_observation_geometry",
    "load_persisted_detection_observation_geometry",
    "load_persisted_source_camera_position_surface",
    "publish_crop_observation_geometry",
    "publish_crop_roi_geometry",
    "publish_detection_observation_geometry",
    "require_bound_crop_observation_geometry",
    "require_bound_detection_frame_evidence",
    "require_bound_detection_observation_geometry",
    "require_bound_source_camera_position_surface",
]
