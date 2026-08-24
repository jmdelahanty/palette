"""Canonical coordinate publication for detection and crop observation rows.

This module is the future-writer boundary for observation geometry.  It never
infers a frame from array names, root dimensions, numerical ranges, or a legacy
space label.  Callers must supply sealed source-camera, normalized-frame, and
direction-labelled transform evidence created from the exact acquisition
authority.

Detection publication persists three deliberately redundant surfaces:

* ``bbox_norm_coords`` -- source-camera-normalized ``cx,cy,w,h``;
* ``bbox_img_xyxy`` -- source-camera half-open pixel edges; and
* ``centers_img_xy`` -- source-camera continuous points derived from the exact
  persisted pixel bbox.

All three share one exact ``instance_key`` identity and one sealed
``source_acquisition_frame_index`` temporal authority.  The normalized-to-pixel
projection and bbox-to-center operation are digest-bound records, so a consumer
can distinguish a genuine persisted derivation from matching-looking numbers.
"""

from __future__ import annotations

import copy
import hashlib
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
from fisheye.shared.coordinate_surface_contract import (
    ROI_BBOX_XYXY,
    SOURCE_CAMERA_BBOX_PIXEL_CONVENTION,
    SOURCE_CAMERA_BBOX_XYXY,
    SOURCE_CAMERA_CROP_XYWH,
    SOURCE_CAMERA_EXTRACTION_ORIGIN_XY,
    SOURCE_CAMERA_NORMALIZED_BBOX_CXCYWH,
    SOURCE_CAMERA_NORMALIZED_PROFILE_ID,
    SOURCE_CAMERA_POINT_PIXEL_CONVENTION,
    SOURCE_CAMERA_POINT_XY,
    SOURCE_CAMERA_PROFILE_ID,
)
from fisheye.shared.coordinate_reference import (
    BoundReferenceExtent,
    bind_array_reference_extent,
    bind_persisted_record_reference_extent,
    canonical_node_path,
)
from fisheye.shared.directed_transform_chain import (
    BoundDirectedTransformChain,
    apply_bound_directed_transform_chain,
    require_bound_directed_transform_chain,
    resolve_bound_directed_transform_chain,
)
from fisheye.shared.directed_transform_v2 import (
    DIRECTED_TRANSFORM_V2_PIXEL_EDGE_ATTR,
    load_bound_directed_transform_v2,
)
from fisheye.shared.immutable_yolo_storage import (
    IMMUTABLE_YOLO_STORAGE_ATTR,
    IMMUTABLE_YOLO_STORAGE_SCHEMA,
)
from fisheye.shared.historical_collection_proxy_v1 import (
    BoundHistoricalMergedCollectionProxyV1,
    load_historical_merged_collection_proxy_v1,
    require_bound_historical_merged_collection_proxy_v1,
)
from fisheye.shared.instance_keys import (
    INSTANCE_KEY_ALGORITHM,
    INSTANCE_KEY_BBOX_QUANTIZATION,
    INSTANCE_KEY_DUPLICATE_POLICY,
    mint_detection_instance_keys,
)
from fisheye.shared.pixel_frame_authority import (
    CROP_PLACEMENT_OWNERSHIP_ATTR,
    CROP_PLACEMENT_PIXEL_EDGE_OWNERSHIP_ATTR,
    ROI_FRAME_KIND,
    SOURCE_CAMERA_FRAME_KIND,
    SOURCE_CAMERA_NORMALIZED_FRAME_KIND,
    BoundAcquisitionCameraFrame,
    BoundCropPlacementOwnership,
    BoundPixelFrameAuthority,
    load_crop_placement_ownership,
    load_normalized_pixel_frame_authority,
    load_persisted_acquisition_camera_authority,
    load_roi_pixel_frame_authority,
    load_source_camera_pixel_frame_authority,
    require_bound_acquisition_camera_frame,
    require_bound_crop_placement_ownership,
    require_normalized_pixel_frame_authority,
    require_roi_pixel_frame_authority,
    require_source_camera_pixel_frame_authority,
    require_trusted_coordinate_attrs,
)
from fisheye.shared.proof_verification import proof_verification_operation
from fisheye.shared.sampled_training_detection_selection import (
    SELECTION_REASON_LABELS,
    select_strong_single_detections,
    strong_single_policy_record,
)
from fisheye.shared.transform_authority import (
    TRANSFORM_AUTHORITY_PIXEL_EDGE_ATTR,
    load_bound_transform_authority,
)
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_CONTRACT,
    RUN_COMPLETION_CONTRACT_ATTR,
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
)


DETECTION_BBOX_PROJECTION_ATTR = "detection_bbox_projection"
DETECTION_BBOX_PROJECTION_SCHEMA_ID = "palette.detection_bbox_projection"
DETECTION_BBOX_PROJECTION_SCHEMA_VERSION = 2
DETECTION_BBOX_PROJECTION_OPERATION = (
    "source_camera_normalized_cxcywh_to_half_open_image_xyxy_v2"
)

DETECTION_BACKEND_RESULT_PROJECTION_ATTR = "detection_backend_result_projection"
DETECTION_BACKEND_RESULT_PROJECTION_SCHEMA_ID = (
    "palette.detection_backend_result_projection"
)
DETECTION_BACKEND_RESULT_PROJECTION_SCHEMA_VERSION = 1
DETECTION_BACKEND_RESULT_PROJECTION_OPERATION = (
    "validated_yolo_result_xyxy_to_source_camera_normalized_cxcywh_v1"
)

DETECTION_ACQUISITION_MAPPING_ATTR = "detection_acquisition_frame_mapping"
DETECTION_ACQUISITION_MAPPING_SCHEMA_ID = "palette.detection_acquisition_frame_mapping"
DETECTION_ACQUISITION_MAPPING_SCHEMA_VERSION = 1

DETECTION_INSTANCE_KEY_DERIVATION_ATTR = "detection_instance_key_derivation"
DETECTION_INSTANCE_KEY_DERIVATION_SCHEMA_ID = (
    "palette.detection_instance_key_derivation"
)
DETECTION_INSTANCE_KEY_DERIVATION_SCHEMA_VERSION = 1

DETECTION_OBSERVATION_CARDINALITY_ATTR = "detection_observation_cardinality"
DETECTION_OBSERVATION_CARDINALITY_SCHEMA_ID = (
    "palette.detection_observation_cardinality"
)
DETECTION_OBSERVATION_CARDINALITY_LEGACY_FLOAT64_SCHEMA_VERSION = 1
DETECTION_OBSERVATION_CARDINALITY_SCHEMA_VERSION = 2

EMPTY_OBSERVATION_DECLARATION_ATTR = "empty_observation_declaration"
EMPTY_OBSERVATION_DECLARATION_SCHEMA_ID = (
    "palette.empty_detection_observation_declaration"
)
EMPTY_OBSERVATION_DECLARATION_SCHEMA_VERSION = 1
OBSERVATION_ROW_COUNT_ATTR = "observation_row_count"
SUPPORTED_DETECTION_DECODE_DOMAIN_PROOFS = frozenset(
    {
        "decord_index_domain_and_exact_batches_v1",
        "opencv_stream_eof_and_exact_count_v1",
        "pynvvc_exact_count_and_eof_probe_v1",
    }
)

BBOX_CENTER_DERIVATION_ATTR = "bbox_center_derivation"
BBOX_CENTER_DERIVATION_SCHEMA_ID = "palette.bbox_center_derivation"
BBOX_CENTER_DERIVATION_SCHEMA_VERSION = 2
BBOX_CENTER_DERIVATION_OPERATION = (
    "half_open_xyxy_edges_to_continuous_midpoint_v2"
)

CROP_GEOMETRY_SELECTION_ATTR = "crop_geometry_selection"
CROP_GEOMETRY_SELECTION_SCHEMA_ID = "palette.crop_geometry_selection"
CROP_GEOMETRY_SELECTION_SCHEMA_VERSION = 1
CROP_GEOMETRY_SELECTION_OPERATION = "exact_instance_key_subset_reorder_v1"

COLLECTION_PROXY_SUCCESSOR_MAPPING_ATTR = (
    "collection_proxy_coordinate_successor_mapping"
)
COLLECTION_PROXY_SUCCESSOR_MAPPING_SCHEMA_ID = (
    "palette.collection_proxy_coordinate_successor_mapping"
)
COLLECTION_PROXY_SUCCESSOR_MAPPING_SCHEMA_VERSION = 1
COLLECTION_PROXY_SUCCESSOR_MAPPING_OPERATION = (
    "verified_historical_v1_rows_to_current_v2_geometry_v1"
)
COLLECTION_PROXY_SUCCESSOR_RUN_SCHEMA = (
    "palette.collection_proxy_coordinate_successor_run"
)
COLLECTION_PROXY_SUCCESSOR_SOURCE_KIND = (
    "historical_collection_proxy_coordinate_successor"
)

SAMPLED_TRAINING_DETECTION_SELECTION_ATTR = (
    "sampled_training_detection_selection"
)
SAMPLED_TRAINING_DETECTION_SELECTION_SCHEMA_ID = (
    "palette.sampled_training_detection.selection"
)
SAMPLED_TRAINING_DETECTION_SELECTION_SCHEMA_VERSION = 1
SAMPLED_TRAINING_DETECTION_RUN_SCHEMA = (
    "palette.sampled_training_detection_run.v1"
)
SAMPLED_TRAINING_DETECTION_SOURCE_KIND = (
    "strong_single_full_frame_detection_selection"
)

CROP_ROI_GEOMETRY_DERIVATION_ATTR = "crop_roi_geometry_derivation"
CROP_ROI_GEOMETRY_DERIVATION_SCHEMA_ID = "palette.crop_roi_geometry_derivation"
CROP_ROI_GEOMETRY_DERIVATION_SCHEMA_VERSION = 1
CROP_ROI_GEOMETRY_DERIVATION_OPERATION = (
    "roi_bbox_to_source_camera_via_crop_placement_v1"
)

CROP_ROI_TOP_LEFT_DERIVATION_ATTR = "crop_roi_top_left_derivation"
CROP_ROI_TOP_LEFT_DERIVATION_SCHEMA_ID = "palette.crop_roi_top_left_derivation"
CROP_ROI_TOP_LEFT_DERIVATION_SCHEMA_VERSION = 2
CROP_ROI_TOP_LEFT_DERIVATION_OPERATION = (
    "source_crop_xywh_to_source_camera_continuous_top_left_v2"
)

CROP_ROI_BBOX_EDGE_FRAME_RELATIVE_PATH = "coordinate_frames/roi_bbox_edge"
CROP_ROI_BBOX_EDGE_EXTENT_ATTR = "crop_roi_bbox_edge_reference_extent"
CROP_ROI_BBOX_EDGE_EXTENT_SCHEMA_ID = (
    "palette.crop_roi_bbox_edge_reference_extent"
)
CROP_ROI_BBOX_EDGE_EXTENT_SCHEMA_VERSION = 1

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


def _strict_detection_model_artifact(value: Any) -> dict[str, Any]:
    if type(value) is not dict:
        _fail(
            "Canonical detection requires one exact model content-fingerprint "
            "mapping."
        )
    expected_fields = {
        "role",
        "path",
        "fingerprint_scheme",
        "sha256",
        "size_bytes",
        "mtime_ns",
        "source",
    }
    if set(value) != expected_fields:
        _fail(
            "Canonical detection model evidence must contain only the strict "
            f"content-v1 fields {sorted(expected_fields)!r}."
        )
    artifact = copy.deepcopy(value)
    digest = artifact.get("sha256")
    if (
        artifact.get("role") != "detect_model"
        or artifact.get("fingerprint_scheme") != "content_v1"
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        _fail("Canonical detection model fingerprint is missing or unsupported.")
    path = artifact.get("path")
    if (
        not isinstance(path, str)
        or not path.startswith("/")
        or not path.strip()
        or path != path.strip()
    ):
        _fail("Canonical detection model fingerprint requires an absolute path.")
    for name in ("size_bytes", "mtime_ns"):
        if type(artifact.get(name)) is not int or artifact[name] < 0:
            _fail(f"Canonical detection model {name} must be nonnegative int.")
    if artifact.get("source") not in {"computed", "sidecar", "registry"}:
        _fail("Canonical detection model fingerprint source is unsupported.")
    return artifact


def _optional_hw(value: Any, *, label: str) -> list[int] | None:
    if value is None:
        return None
    if (
        type(value) is not list
        or len(value) != 2
        or any(type(item) is not int or item <= 0 for item in value)
    ):
        _fail(f"{label} must be null or one exact positive [height, width] list.")
    return [int(value[0]), int(value[1])]


def _runtime_detection_result_contract(
    rowset_node: Any,
    *,
    acquisition_frame: BoundAcquisitionCameraFrame,
) -> dict[str, Any]:
    attrs = require_trusted_coordinate_attrs(
        rowset_node,
        label="Canonical detection backend-result rowset",
    )
    parameters = attrs.get("parameters")
    if type(parameters) is not dict:
        _fail("Canonical detection lacks exact runtime parameter metadata.")
    backend = attrs.get("decode_backend_effective")
    reader = attrs.get("video_reader_type")
    if (
        not isinstance(backend, str)
        or not backend
        or backend != backend.strip()
        or not isinstance(reader, str)
        or not reader
        or reader != reader.strip()
    ):
        _fail("Canonical detection lacks an exact decoder/backend identity.")
    height = attrs.get("inference_height")
    width = attrs.get("inference_width")
    result_count = attrs.get("validated_backend_result_count")
    result_shape = _optional_hw(
        attrs.get("validated_backend_result_orig_shape_hw"),
        label="validated_backend_result_orig_shape_hw",
    )
    if (
        type(height) is not int
        or height <= 0
        or type(width) is not int
        or width <= 0
        or type(result_count) is not int
        or result_count != acquisition_frame.record.source_total_frames
        or result_shape != [height, width]
    ):
        _fail(
            "Canonical detection backend-result count/orig_shape does not match "
            "the exact acquisition and runtime input extent."
        )
    if parameters.get("decode_backend_effective") != backend:
        _fail("Detection parameters and backend-result decoder identity disagree.")
    requested = _optional_hw(
        parameters.get("resize_dims"),
        label="parameters.resize_dims",
    )
    pre_resize = _optional_hw(
        parameters.get("pre_resize_dims"),
        label="parameters.pre_resize_dims",
    )
    effective = _optional_hw(
        parameters.get("effective_input_resize_dims"),
        label="parameters.effective_input_resize_dims",
    )
    tensor_resize = _optional_hw(
        parameters.get("tensor_resize_dims"),
        label="parameters.tensor_resize_dims",
    )
    imgsz = parameters.get("imgsz_applied")
    if imgsz is not None and not (
        type(imgsz) is int
        and imgsz > 0
        or type(imgsz) is list
        and len(imgsz) == 2
        and all(type(item) is int and item > 0 for item in imgsz)
    ):
        _fail("parameters.imgsz_applied is not one exact supported shape.")
    return {
        "decode_backend_effective": backend,
        "video_reader_type": reader,
        "validated_result_count": result_count,
        "validated_result_orig_shape_hw": result_shape,
        "requested_resize_dims_hw": requested,
        "pre_resize_dims_hw": pre_resize,
        "effective_runtime_input_resize_dims_hw": effective,
        "tensor_resize_dims_hw": tensor_resize,
        "ultralytics_imgsz_applied": copy.deepcopy(imgsz),
        "result_coordinate_contract": (
            "ultralytics_boxes_xyxy_in_validated_result_orig_shape_px"
        ),
        "network_preprocessing_authority": (
            "not_persisted_not_used_as_coordinate_projection_authority"
        ),
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
    cause: BaseException,
) -> None:
    failures: list[str] = []
    for attrs, snapshot in zip(attrs_targets, snapshots, strict=True):
        try:
            _restore_attrs(attrs, snapshot)
        except BaseException as exc:  # pragma: no cover - hostile persistent mapping
            failures.append(str(exc))
    if failures:
        raise ObservationCoordinatePublicationError(
            "Observation coordinate publication failed and attrs rollback was "
            f"incomplete: {failures!r}."
        ) from cause


@dataclass(frozen=True)
class ObservationCoordinatePublicationCheckpoint:
    """Exact attrs checkpoint for a multi-step coordinate publication.

    Writers capture this immediately before publishing coordinate records and
    descriptors.  If completion or selector publication subsequently fails,
    restoring the checkpoint removes every partially trusted coordinate attr
    while preserving the writer's already-validated non-coordinate metadata.
    """

    _attrs_targets: tuple[Any, ...] = field(repr=False)
    _snapshots: tuple[dict[str, Any], ...] = field(repr=False)


def capture_observation_coordinate_publication_checkpoint(
    *nodes: Any,
) -> ObservationCoordinatePublicationCheckpoint:
    """Capture exact attrs for every coordinate-publication target."""

    attrs_targets, snapshots = _attrs_snapshots(*nodes)
    return ObservationCoordinatePublicationCheckpoint(
        _attrs_targets=tuple(attrs_targets),
        _snapshots=tuple(snapshots),
    )


def restore_observation_coordinate_publication_checkpoint(
    checkpoint: ObservationCoordinatePublicationCheckpoint,
    *,
    cause: BaseException,
) -> None:
    """Restore a checkpoint exactly, raising loudly on incomplete rollback."""

    if type(checkpoint) is not ObservationCoordinatePublicationCheckpoint:
        _fail("An observation coordinate publication checkpoint is required.")
    _rollback_attrs(
        list(checkpoint._attrs_targets),
        list(checkpoint._snapshots),
        cause=cause,
    )


@dataclass(frozen=True, init=False)
class BoundDetectionFrameEvidence:
    """Sealed point, bbox-edge, and normalized source-camera evidence."""

    source_camera_frame: BoundPixelFrameAuthority = field(repr=False)
    bbox_source_camera_frame: BoundPixelFrameAuthority = field(repr=False)
    normalized_frame: BoundPixelFrameAuthority = field(repr=False)
    normalized_to_source_camera: BoundDirectedTransformChain = field(repr=False)
    _seal: object = field(repr=False, compare=False)

    def __init__(
        self,
        *,
        source_camera_frame: BoundPixelFrameAuthority,
        bbox_source_camera_frame: BoundPixelFrameAuthority,
        normalized_frame: BoundPixelFrameAuthority,
        normalized_to_source_camera: BoundDirectedTransformChain,
        _verification_seal: object | None = None,
    ) -> None:
        if _verification_seal is not _BOUND_DETECTION_FRAME_EVIDENCE_SEAL:
            _fail("Detection frame evidence must be built by the sealed verifier.")
        object.__setattr__(self, "source_camera_frame", source_camera_frame)
        object.__setattr__(
            self,
            "bbox_source_camera_frame",
            bbox_source_camera_frame,
        )
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


@proof_verification_operation
def build_bound_detection_frame_evidence(
    *,
    source_camera_frame: BoundPixelFrameAuthority,
    bbox_source_camera_frame: BoundPixelFrameAuthority,
    normalized_frame: BoundPixelFrameAuthority,
    normalized_to_source_camera: BoundDirectedTransformChain,
) -> BoundDetectionFrameEvidence:
    """Verify exact point and half-open bbox source-camera authorities."""

    camera = require_source_camera_pixel_frame_authority(source_camera_frame)
    bbox_camera = require_source_camera_pixel_frame_authority(
        bbox_source_camera_frame
    )
    normalized = require_normalized_pixel_frame_authority(normalized_frame)
    chain = require_bound_directed_transform_chain(normalized_to_source_camera)
    if camera.record.kind != SOURCE_CAMERA_FRAME_KIND:
        _fail("Detection geometry requires a source-camera pixel frame.")
    if camera.pixel_convention != SOURCE_CAMERA_POINT_PIXEL_CONVENTION:
        _fail("Detection source-camera point geometry requires continuous coordinates.")
    if bbox_camera.pixel_convention != SOURCE_CAMERA_BBOX_PIXEL_CONVENTION:
        _fail(
            "Detection source-camera bbox geometry requires half-open pixel-edge "
            "coordinates."
        )
    if (
        camera.reference_extent.record_ref
        != bbox_camera.reference_extent.record_ref
        or camera.reference_extent.record_sha256
        != bbox_camera.reference_extent.record_sha256
        or camera.endpoint.width != bbox_camera.endpoint.width
        or camera.endpoint.height != bbox_camera.endpoint.height
    ):
        _fail(
            "Detection point and bbox source-camera frames do not bind the exact "
            "same acquisition extent."
        )
    if normalized.record.kind != SOURCE_CAMERA_NORMALIZED_FRAME_KIND:
        _fail(
            "Detection normalized geometry requires a source-camera normalized frame."
        )
    expected_camera_ref = {
        "record_ref": bbox_camera.record_ref,
        "record_sha256": bbox_camera.record_sha256,
    }
    if normalized.record.lineage.get("pixel_frame") != expected_camera_ref:
        _fail("Normalized frame does not bind the exact source-camera frame.")
    if (
        not _same_pixel_frame(chain.descriptor_frame_authority, normalized)
        or not _same_pixel_frame(chain.source_camera_frame_authority, bbox_camera)
        or chain.row_identity is not None
        or chain.descriptor_space_id != "source_camera_normalized_xy"
        or chain.source_camera_space_id != "source_camera_image_px"
    ):
        _fail(
            "Detection normalized-to-camera chain has the wrong direction, "
            "endpoints, or row domain."
        )
    if not (
        camera.archive_identity
        == bbox_camera.archive_identity
        == normalized.archive_identity
        == chain.archive_identity
    ):
        _fail("Detection frame evidence spans different archives.")
    return BoundDetectionFrameEvidence(
        source_camera_frame=camera,
        bbox_source_camera_frame=bbox_camera,
        normalized_frame=normalized,
        normalized_to_source_camera=chain,
        _verification_seal=_BOUND_DETECTION_FRAME_EVIDENCE_SEAL,
    )


@proof_verification_operation
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
        bbox_source_camera_frame=value.bbox_source_camera_frame,
        normalized_frame=value.normalized_frame,
        normalized_to_source_camera=value.normalized_to_source_camera,
    )
    if current != value:
        _fail("Detection frame evidence changed after binding.")
    return value


def _detection_backend_result_projection_record(
    rowset_node: Any,
    bbox_norm_node: Any,
    *,
    frame_evidence: BoundDetectionFrameEvidence,
    model_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    evidence = require_bound_detection_frame_evidence(frame_evidence)
    _require_child_path(bbox_norm_node, rowset_node, "bbox_norm_coords")
    bbox_norm = _array(bbox_norm_node, label="bbox_norm_coords")
    if (
        bbox_norm.dtype.kind != "f"
        or bbox_norm.ndim != 2
        or bbox_norm.shape[1:] != (4,)
        or not np.isfinite(bbox_norm).all()
    ):
        _fail(
            "Canonical detection backend-result projection requires a finite "
            "floating (N,4) bbox_norm_coords surface."
        )
    artifact = _strict_detection_model_artifact(model_artifact)
    attrs = require_trusted_coordinate_attrs(
        rowset_node,
        label="Canonical detection backend-result rowset",
    )
    if (
        attrs.get("model_path") != artifact["path"]
        or attrs.get("model_name") != artifact["path"].rsplit("/", 1)[-1]
    ):
        _fail(
            "Canonical detection model attrs differ from the exact fingerprinted "
            "model artifact."
        )
    runtime = _runtime_detection_result_contract(
        rowset_node,
        acquisition_frame=evidence.acquisition_frame,
    )
    result_height, result_width = runtime["validated_result_orig_shape_hw"]
    source_width = int(evidence.bbox_source_camera_frame.endpoint.width)
    source_height = int(evidence.bbox_source_camera_frame.endpoint.height)
    return {
        "schema_id": DETECTION_BACKEND_RESULT_PROJECTION_SCHEMA_ID,
        "schema_version": DETECTION_BACKEND_RESULT_PROJECTION_SCHEMA_VERSION,
        "operation": DETECTION_BACKEND_RESULT_PROJECTION_OPERATION,
        "direction": (
            "detector_backend_result_image_px_to_source_camera_normalized_xy"
        ),
        "backend_result_space": {
            "space_id": "detector_backend_result_image_px",
            "geometry_type": "bbox_xyxy",
            "pixel_convention": SOURCE_CAMERA_BBOX_PIXEL_CONVENTION,
            "reference_width_px": int(result_width),
            "reference_height_px": int(result_height),
        },
        "source_camera_normalized_frame": {
            "record_ref": evidence.normalized_frame.record_ref,
            "record_sha256": evidence.normalized_frame.record_sha256,
        },
        "source_camera_bbox_frame": {
            "record_ref": evidence.bbox_source_camera_frame.record_ref,
            "record_sha256": evidence.bbox_source_camera_frame.record_sha256,
        },
        "acquisition_camera_frame": {
            "record_ref": evidence.acquisition_frame.record_ref,
            "record_sha256": evidence.acquisition_frame.record_sha256,
        },
        "published_bbox_normalized": _payload(bbox_norm_node, bbox_norm),
        "result_px_to_source_camera_normalized_matrix": [
            [1.0 / float(result_width), 0.0, 0.0],
            [0.0, 1.0 / float(result_height), 0.0],
            [0.0, 0.0, 1.0],
        ],
        "result_px_to_source_camera_bbox_matrix": [
            [float(source_width) / float(result_width), 0.0, 0.0],
            [0.0, float(source_height) / float(result_height), 0.0],
            [0.0, 0.0, 1.0],
        ],
        "runtime_result_validation": runtime,
        "model_artifact": artifact,
        "proof": (
            "every_backend_result_orig_shape_equaled_its_exact_runtime_input_"
            "shape_before_bbox_normalization_v1"
        ),
    }


@proof_verification_operation
def publish_detection_backend_result_projection(
    rowset_node: Any,
    bbox_norm_node: Any,
    *,
    frame_evidence: BoundDetectionFrameEvidence,
    model_artifact: Mapping[str, Any],
) -> BoundCoordinateRecord:
    """Persist the verified YOLO result-space projection without guessing letterbox."""

    record = _detection_backend_result_projection_record(
        rowset_node,
        bbox_norm_node,
        frame_evidence=frame_evidence,
        model_artifact=model_artifact,
    )
    return stamp_and_bind_persisted_coordinate_record(
        rowset_node,
        record,
        attr_name=DETECTION_BACKEND_RESULT_PROJECTION_ATTR,
    )


@proof_verification_operation
def load_detection_backend_result_projection(
    rowset_node: Any,
    bbox_norm_node: Any,
    *,
    frame_evidence: BoundDetectionFrameEvidence,
) -> BoundCoordinateRecord:
    """Freshly verify exact persisted backend-result projection evidence."""

    bound = bind_persisted_coordinate_record(
        rowset_node,
        attr_name=DETECTION_BACKEND_RESULT_PROJECTION_ATTR,
    )
    artifact = _strict_detection_model_artifact(
        bound.record.get("model_artifact")
    )
    expected = _detection_backend_result_projection_record(
        rowset_node,
        bbox_norm_node,
        frame_evidence=frame_evidence,
        model_artifact=artifact,
    )
    if bound.record != expected:
        _fail(
            "Persisted detection backend-result projection differs from the "
            "exact live result shape, source frames, model, or bbox payload."
        )
    return bound


@proof_verification_operation
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
    width_px = np.asarray(
        evidence.bbox_source_camera_frame.endpoint.width,
        dtype=dtype,
    )
    height_px = np.asarray(
        evidence.bbox_source_camera_frame.endpoint.height,
        dtype=dtype,
    )
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
    point_camera = frame_evidence.source_camera_frame
    bbox_camera = frame_evidence.bbox_source_camera_frame
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
            "record_ref": bbox_camera.record_ref,
            "record_sha256": bbox_camera.record_sha256,
        },
        "direction": "source_camera_normalized_xy_to_source_camera_image_px",
        "transform_chain": [
            {
                "record_ref": item.record_ref,
                "record_sha256": item.record_sha256,
            }
            for item in chain.transform_records
        ],
        "destination_pixel_convention": SOURCE_CAMERA_BBOX_PIXEL_CONVENTION,
        "reference_width_px": int(bbox_camera.endpoint.width),
        "reference_height_px": int(bbox_camera.endpoint.height),
        "formula": (
            "cxcywh_normalized_to_xyxy_half_open_edges_using_exact_"
            "reference_extent_v2"
        ),
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
        "source_frame": {
            "record_ref": bbox_camera.record_ref,
            "record_sha256": bbox_camera.record_sha256,
        },
        "destination_frame": {
            "record_ref": point_camera.record_ref,
            "record_sha256": point_camera.record_sha256,
        },
        "direction": (
            "source_camera_bbox_pixel_edge_half_open_to_"
            "source_camera_point_continuous"
        ),
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
        or descriptor.descriptor.pixel_convention
        != SOURCE_CAMERA_POINT_PIXEL_CONVENTION
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


@proof_verification_operation
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
    if (
        evidence.source_camera_frame.archive_identity != common_archive
        or evidence.bbox_source_camera_frame.archive_identity != common_archive
    ):
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
            **SOURCE_CAMERA_NORMALIZED_BBOX_CXCYWH.descriptor_kwargs(),
            row_identity=identity,
            reference_frame_authority=evidence.normalized_frame,
            transform_chain=evidence.normalized_to_source_camera,
            lineage_records=(*source_lineage, projection),
        )
        bbox_image = build_bound_canonical_coordinate_descriptor(
            bbox_img_node,
            **SOURCE_CAMERA_BBOX_XYXY.descriptor_kwargs(),
            row_identity=identity,
            reference_frame_authority=evidence.bbox_source_camera_frame,
            lineage_records=(*source_lineage, projection),
        )
        centers_image = build_bound_canonical_coordinate_descriptor(
            centers_img_node,
            **SOURCE_CAMERA_POINT_XY.descriptor_kwargs(),
            row_identity=identity,
            reference_frame_authority=evidence.source_camera_frame,
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
    except BaseException as exc:
        _rollback_attrs(attrs_targets, snapshots, cause=exc)
        raise


@proof_verification_operation
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
        reference_frame_authority=evidence.bbox_source_camera_frame,
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


@proof_verification_operation
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
    except BaseException as exc:
        _rollback_attrs(attrs_targets, snapshots, cause=exc)
        raise


@proof_verification_operation
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
        reference_frame_authority=(
            source.frame_evidence.bbox_source_camera_frame
        ),
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


def _crop_roi_bbox_edge_extent_record(
    roi_images_node: Any,
) -> tuple[dict[str, Any], BoundReferenceExtent]:
    source_extent = bind_array_reference_extent(roi_images_node, units="px")
    return (
        {
            "schema_id": CROP_ROI_BBOX_EDGE_EXTENT_SCHEMA_ID,
            "schema_version": CROP_ROI_BBOX_EDGE_EXTENT_SCHEMA_VERSION,
            "width_px": int(source_extent.width),
            "height_px": int(source_extent.height),
            "units": "px",
            "source_roi_images_extent": {
                "record_ref": source_extent.record_ref,
                "record_sha256": source_extent.record_sha256,
                "selector": source_extent.selector,
            },
            "purpose": "half_open_bbox_edge_frame_extent_only",
        },
        source_extent,
    )


def _require_crop_roi_bbox_edge_frame_path(
    frame_node: Any,
    roi_images_node: Any,
) -> None:
    roi_path = canonical_node_path(roi_images_node)
    if not roi_path.endswith("/roi_images"):
        _fail("Crop ROI pixel authority must be the exact roi_images child.")
    rowset_path = roi_path[: -len("/roi_images")]
    expected = f"{rowset_path}/{CROP_ROI_BBOX_EDGE_FRAME_RELATIVE_PATH}"
    if canonical_node_path(frame_node) != expected:
        _fail(
            "Crop ROI bbox-edge frame must use its exact run-local canonical "
            f"path {expected!r}."
        )


@proof_verification_operation
def publish_crop_roi_bbox_edge_reference_extent(
    frame_node: Any,
    roi_images_node: Any,
) -> BoundReferenceExtent:
    """Bind a separate half-open bbox frame to exact live ROI image metadata."""

    _require_crop_roi_bbox_edge_frame_path(frame_node, roi_images_node)
    record, _ = _crop_roi_bbox_edge_extent_record(roi_images_node)
    attrs_targets, snapshots = _attrs_snapshots(frame_node)
    try:
        attrs = require_trusted_coordinate_attrs(
            frame_node,
            label="Crop ROI bbox-edge extent frame",
        )
        attrs.update(
            {
                "width_px": int(record["width_px"]),
                "height_px": int(record["height_px"]),
                "units": "px",
            }
        )
        stamp_and_bind_persisted_coordinate_record(
            frame_node,
            record,
            attr_name=CROP_ROI_BBOX_EDGE_EXTENT_ATTR,
        )
        return load_crop_roi_bbox_edge_reference_extent(
            frame_node,
            roi_images_node,
        )
    except BaseException as exc:
        _rollback_attrs(attrs_targets, snapshots, cause=exc)
        raise


@proof_verification_operation
def load_crop_roi_bbox_edge_reference_extent(
    frame_node: Any,
    roi_images_node: Any,
) -> BoundReferenceExtent:
    """Freshly validate a separate ROI bbox-edge frame extent authority."""

    _require_crop_roi_bbox_edge_frame_path(frame_node, roi_images_node)
    expected, source_extent = _crop_roi_bbox_edge_extent_record(roi_images_node)
    record = bind_persisted_coordinate_record(
        frame_node,
        attr_name=CROP_ROI_BBOX_EDGE_EXTENT_ATTR,
    )
    if record.record != expected:
        _fail(
            "Crop ROI bbox-edge extent differs from exact live roi_images "
            "metadata."
        )
    extent = bind_persisted_record_reference_extent(
        frame_node,
        record_attr=CROP_ROI_BBOX_EDGE_EXTENT_ATTR,
        digest_attr=f"{CROP_ROI_BBOX_EDGE_EXTENT_ATTR}_sha256",
        width_field="width_px",
        height_field="height_px",
        units_field="units",
    )
    if (
        extent.width != source_extent.width
        or extent.height != source_extent.height
        or extent.units != source_extent.units
    ):
        _fail("Crop ROI bbox-edge extent does not equal roi_images shape[-2:].")
    return extent


@dataclass(frozen=True)
class CropRoiGeometryPublicationResult:
    """Canonical crop placement and ROI-local bbox descriptors."""

    source_crop_xywh: BoundCanonicalCoordinateDescriptor
    bbox_roi_xyxy: BoundCanonicalCoordinateDescriptor
    derivation: BoundCoordinateRecord
    roi_top_left_xy: BoundCanonicalCoordinateDescriptor | None = None
    top_left_derivation: BoundCoordinateRecord | None = None


def _validate_crop_roi_top_left(
    *,
    crop: BoundCropObservationGeometry,
    source_crop_xywh_node: Any,
    placement: np.ndarray,
    roi_top_left_node: Any,
) -> np.ndarray:
    """Require the compatibility top-left surface to equal placement ``x,y``."""

    _require_child_path(roi_top_left_node, crop._rowset_node, "roi_coordinates_full")
    top_left = _array(roi_top_left_node, label="roi_coordinates_full")
    if (
        top_left.dtype != np.dtype("<i4")
        or top_left.shape != (crop.row_identity.leading_dimension, 2)
    ):
        _fail("roi_coordinates_full must be one exact int32 (N,2) points_xy surface.")
    if not np.array_equal(top_left, placement[:, :2], equal_nan=True):
        _fail(
            "roi_coordinates_full is not the exact source-camera top-left "
            "projection of source_crop_xywh[:, :2]."
        )
    if canonical_node_path(source_crop_xywh_node) != (
        f"{canonical_node_path(crop._rowset_node)}/source_crop_xywh"
    ):
        _fail("Crop top-left derivation uses an unexpected placement path.")
    return top_left


def _crop_roi_top_left_record(
    *,
    crop: BoundCropObservationGeometry,
    point_placement_ownership: BoundCropPlacementOwnership,
    source_crop_xywh_node: Any,
    placement: np.ndarray,
    roi_top_left_node: Any,
    top_left: np.ndarray,
) -> dict[str, Any]:
    point_camera = point_placement_ownership.source_camera_frame
    return {
        "schema_id": CROP_ROI_TOP_LEFT_DERIVATION_SCHEMA_ID,
        "schema_version": CROP_ROI_TOP_LEFT_DERIVATION_SCHEMA_VERSION,
        "operation": CROP_ROI_TOP_LEFT_DERIVATION_OPERATION,
        "source_crop_xywh": _payload(source_crop_xywh_node, placement),
        "roi_coordinates_full": _payload(roi_top_left_node, top_left),
        "row_identity": {
            "record_ref": crop.row_identity.record_ref,
            "record_sha256": crop.row_identity.record_sha256,
        },
        "crop_placement_ownership": {
            "record_ref": point_placement_ownership.record_ref,
            "record_sha256": point_placement_ownership.record_sha256,
        },
        "reference_frame": {
            "record_ref": point_camera.record_ref,
            "record_sha256": point_camera.record_sha256,
        },
        "direction": "source_crop_xywh_to_source_camera_top_left_points_xy",
        "pixel_convention": SOURCE_CAMERA_POINT_PIXEL_CONVENTION,
        "formula": "roi_coordinates_full = source_crop_xywh[:, :2]",
    }


def _validate_crop_roi_top_left_point_ownership(
    *,
    crop: BoundCropObservationGeometry,
    source_crop_xywh_node: Any,
    point_placement_ownership: BoundCropPlacementOwnership,
) -> BoundCropPlacementOwnership:
    ownership = require_bound_crop_placement_ownership(
        point_placement_ownership
    )
    expected_camera = crop.source_geometry.frame_evidence.source_camera_frame
    if ownership.attr_name != CROP_PLACEMENT_OWNERSHIP_ATTR:
        _fail(
            "Crop ROI top-left point geometry requires the canonical continuous "
            "crop-placement ownership attr."
        )
    if not _same_row_identity(ownership.row_identity, crop.row_identity):
        _fail("Crop ROI top-left ownership uses a different observation identity.")
    if not _same_pixel_frame(ownership.source_camera_frame, expected_camera):
        _fail(
            "Crop ROI top-left ownership targets a different source-camera "
            "point frame."
        )
    if (
        ownership.source_camera_frame.pixel_convention
        != SOURCE_CAMERA_POINT_PIXEL_CONVENTION
    ):
        _fail("Crop ROI top-left ownership must target continuous point coordinates.")
    if canonical_node_path(ownership._placement_node) != canonical_node_path(
        source_crop_xywh_node
    ):
        _fail(
            "Crop ROI top-left ownership does not bind the exact "
            "source_crop_xywh payload."
        )
    return ownership


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
    if (
        roi.record.kind != ROI_FRAME_KIND
        or roi.pixel_convention != SOURCE_CAMERA_BBOX_PIXEL_CONVENTION
    ):
        _fail(
            "Canonical ROI bbox geometry requires a half-open pixel-edge "
            "crop-ROI frame."
        )
    if not _same_row_identity(ownership.row_identity, crop.row_identity):
        _fail("Crop placement ownership uses a different observation identity.")
    if not _same_row_identity(roi.row_identity, crop.row_identity):
        _fail("ROI frame uses a different observation identity.")
    if not _same_pixel_frame(
        ownership.source_camera_frame,
        crop.source_geometry.frame_evidence.bbox_source_camera_frame,
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


@proof_verification_operation
def publish_crop_roi_geometry(
    source_crop_xywh_node: Any,
    bbox_roi_xyxy_node: Any,
    *,
    crop_geometry: BoundCropObservationGeometry,
    crop_placement_ownership: BoundCropPlacementOwnership,
    roi_frame: BoundPixelFrameAuthority,
    roi_to_source_camera: BoundDirectedTransformChain,
    roi_top_left_node: Any | None = None,
    roi_top_left_placement_ownership: BoundCropPlacementOwnership | None = None,
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
    if (roi_top_left_node is None) != (
        roi_top_left_placement_ownership is None
    ):
        _fail(
            "Crop ROI top-left publication requires both its array and its "
            "continuous point-placement ownership."
        )
    point_ownership = None
    if roi_top_left_placement_ownership is not None:
        point_ownership = _validate_crop_roi_top_left_point_ownership(
            crop=crop,
            source_crop_xywh_node=source_crop_xywh_node,
            point_placement_ownership=roi_top_left_placement_ownership,
        )
    attrs_targets, snapshots = _attrs_snapshots(
        crop._rowset_node,
        source_crop_xywh_node,
        bbox_roi_xyxy_node,
        *((roi_top_left_node,) if roi_top_left_node is not None else ()),
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
        top_left_derivation = None
        top_left_binding = None
        if roi_top_left_node is not None and point_ownership is not None:
            top_left = _validate_crop_roi_top_left(
                crop=crop,
                source_crop_xywh_node=source_crop_xywh_node,
                placement=placement,
                roi_top_left_node=roi_top_left_node,
            )
            top_left_record = _crop_roi_top_left_record(
                crop=crop,
                point_placement_ownership=point_ownership,
                source_crop_xywh_node=source_crop_xywh_node,
                placement=placement,
                roi_top_left_node=roi_top_left_node,
                top_left=top_left,
            )
            top_left_derivation = stamp_and_bind_persisted_coordinate_record(
                crop._rowset_node,
                top_left_record,
                attr_name=CROP_ROI_TOP_LEFT_DERIVATION_ATTR,
            )
        source_crop = build_bound_canonical_coordinate_descriptor(
            source_crop_xywh_node,
            **SOURCE_CAMERA_CROP_XYWH.descriptor_kwargs(),
            row_identity=crop.row_identity,
            reference_frame_authority=ownership.source_camera_frame,
            lineage_records=(crop.selection_derivation, derivation),
        )
        bbox_roi_binding = build_bound_canonical_coordinate_descriptor(
            bbox_roi_xyxy_node,
            **ROI_BBOX_XYXY.descriptor_kwargs(),
            row_identity=crop.row_identity,
            reference_frame_authority=roi,
            transform_chain=chain,
            lineage_records=(crop.selection_derivation, derivation),
        )
        bindings = [source_crop, bbox_roi_binding]
        if roi_top_left_node is not None and top_left_derivation is not None:
            top_left_binding = build_bound_canonical_coordinate_descriptor(
                roi_top_left_node,
                **SOURCE_CAMERA_EXTRACTION_ORIGIN_XY.descriptor_kwargs(),
                row_identity=crop.row_identity,
                reference_frame_authority=point_ownership.source_camera_frame,
                lineage_records=(
                    crop.selection_derivation,
                    top_left_derivation,
                ),
            )
            bindings.append(top_left_binding)
        stamp_bound_canonical_coordinate_descriptors(tuple(bindings))
        return load_crop_roi_geometry(
            source_crop_xywh_node,
            bbox_roi_xyxy_node,
            crop_geometry=crop,
            crop_placement_ownership=ownership,
            roi_frame=roi,
            roi_to_source_camera=chain,
            roi_top_left_node=roi_top_left_node,
            roi_top_left_placement_ownership=point_ownership,
        )
    except BaseException as exc:
        _rollback_attrs(attrs_targets, snapshots, cause=exc)
        raise


@proof_verification_operation
def load_crop_roi_geometry(
    source_crop_xywh_node: Any,
    bbox_roi_xyxy_node: Any,
    *,
    crop_geometry: BoundCropObservationGeometry,
    crop_placement_ownership: BoundCropPlacementOwnership,
    roi_frame: BoundPixelFrameAuthority,
    roi_to_source_camera: BoundDirectedTransformChain,
    roi_top_left_node: Any | None = None,
    roi_top_left_placement_ownership: BoundCropPlacementOwnership | None = None,
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
    if (roi_top_left_node is None) != (
        roi_top_left_placement_ownership is None
    ):
        _fail(
            "Crop ROI top-left loading requires both its array and its "
            "continuous point-placement ownership."
        )
    point_ownership = None
    if roi_top_left_placement_ownership is not None:
        point_ownership = _validate_crop_roi_top_left_point_ownership(
            crop=crop,
            source_crop_xywh_node=source_crop_xywh_node,
            point_placement_ownership=roi_top_left_placement_ownership,
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
    top_left_derivation = None
    top_left_binding = None
    if roi_top_left_node is not None and point_ownership is not None:
        top_left = _validate_crop_roi_top_left(
            crop=crop,
            source_crop_xywh_node=source_crop_xywh_node,
            placement=placement,
            roi_top_left_node=roi_top_left_node,
        )
        top_left_derivation = bind_persisted_coordinate_record(
            crop._rowset_node,
            attr_name=CROP_ROI_TOP_LEFT_DERIVATION_ATTR,
        )
        expected_top_left = _crop_roi_top_left_record(
            crop=crop,
            point_placement_ownership=point_ownership,
            source_crop_xywh_node=source_crop_xywh_node,
            placement=placement,
            roi_top_left_node=roi_top_left_node,
            top_left=top_left,
        )
        if top_left_derivation.record != expected_top_left:
            _fail(
                "Persisted crop top-left derivation differs from exact live "
                "source_crop_xywh projection."
            )
        top_left_binding = load_bound_canonical_coordinate_descriptor(
            roi_top_left_node,
            row_identity=crop.row_identity,
            reference_frame_authority=point_ownership.source_camera_frame,
            lineage_records=(
                crop.selection_derivation,
                top_left_derivation,
            ),
        )
    if (
        source_crop.descriptor.profile_id != SOURCE_CAMERA_PROFILE_ID
        or source_crop.descriptor.geometry_type != "bbox_xywh"
        or bbox_roi_binding.descriptor.profile_id != "roi_local_px.top_left_y_down.v1"
        or bbox_roi_binding.descriptor.geometry_type != "bbox_xyxy"
        or (
            top_left_binding is not None
            and (
                top_left_binding.descriptor.profile_id != SOURCE_CAMERA_PROFILE_ID
                or top_left_binding.descriptor.geometry_type != "point_xy"
                or top_left_binding.descriptor.source_camera_overlay.status
                != CANONICAL_OVERLAY_DIRECT
            )
        )
    ):
        _fail("Crop ROI coordinate descriptors use unsupported profiles.")
    return CropRoiGeometryPublicationResult(
        source_crop_xywh=source_crop,
        bbox_roi_xyxy=bbox_roi_binding,
        derivation=derivation,
        roi_top_left_xy=top_left_binding,
        top_left_derivation=top_left_derivation,
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


def _require_complete_canonical_observation_rowset(
    rowset: Any,
    *,
    run_family: str,
    label: str,
    require_selector_eligible: bool = True,
) -> None:
    """Fail closed unless ``rowset`` is one exact complete canonical stage run."""

    path = canonical_node_path(rowset)
    parts = path.split("/")
    if len(parts) != 2 or parts[0] != run_family or not parts[1]:
        _fail(
            f"{label} must be an exact {run_family}/<run> rowset; found {path!r}."
        )
    attrs = require_trusted_coordinate_attrs(rowset, label=label)
    if attrs.get("coordinate_contract") != "canonical_v2":
        _fail(f"{label} is not explicitly canonical_v2.")
    if attrs.get(RUN_COMPLETION_CONTRACT_ATTR) != RUN_COMPLETION_CONTRACT:
        _fail(
            f"{label} lacks the explicit {RUN_COMPLETION_CONTRACT!r} completion "
            "contract."
        )
    if attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE:
        _fail(
            f"{label} is not explicitly complete; found "
            f"{attrs.get(RUN_COMPLETION_STATUS_ATTR)!r}."
        )
    if type(require_selector_eligible) is not bool:
        _fail("Selector-eligibility validation mode must be one exact boolean.")
    expected_eligibility = require_selector_eligible
    if attrs.get("stage_selector_eligible") is not expected_eligibility:
        state = "eligible" if expected_eligibility else "selector-ineligible"
        _fail(f"{label} is not explicitly {state} for this validation phase.")


def _require_detection_observation_row_count(
    rowset: Any,
    key_node: Any,
) -> int:
    """Bind the declared observation count to the exact row-identity dimension."""

    _require_child_path(key_node, rowset, "instance_key")
    attrs = require_trusted_coordinate_attrs(
        rowset,
        label="Detection observation rowset",
    )
    declared = attrs.get(OBSERVATION_ROW_COUNT_ATTR)
    if type(declared) is not int or declared < 0:
        _fail(
            f"Detection {OBSERVATION_ROW_COUNT_ATTR} must be one exact "
            "nonnegative integer."
        )
    try:
        key_shape = tuple(int(value) for value in key_node.shape)
    except (AttributeError, TypeError, ValueError) as exc:
        _fail(f"Detection instance_key has no exact persisted shape: {exc}.")
    if len(key_shape) != 1:
        _fail("Detection instance_key must be one exact one-dimensional row key.")
    if declared != key_shape[0]:
        _fail(
            f"Detection {OBSERVATION_ROW_COUNT_ATTR} disagrees with the exact "
            f"instance_key row dimension: declared={declared}, rows={key_shape[0]}."
        )
    return declared


def _detection_observation_cardinality_record(
    rowset: Any,
    *,
    acquisition_frame: BoundAcquisitionCameraFrame,
    schema_version: int | None = None,
) -> dict[str, Any]:
    """Build an exact all-row/all-frame cardinality and payload seal."""

    acquisition = require_bound_acquisition_camera_frame(acquisition_frame)
    try:
        geometry_dtype = np.dtype(rowset["bbox_norm_coords"].dtype)
    except Exception as exc:
        _fail(f"Detection normalized geometry dtype is unavailable: {exc}.")
    if schema_version is None:
        if geometry_dtype == np.dtype("<f8"):
            resolved_schema_version = (
                DETECTION_OBSERVATION_CARDINALITY_LEGACY_FLOAT64_SCHEMA_VERSION
            )
        elif geometry_dtype == np.dtype("<f4"):
            resolved_schema_version = DETECTION_OBSERVATION_CARDINALITY_SCHEMA_VERSION
        else:
            _fail(
                "Detection cardinality geometry must use exact float32 v2 or "
                "legacy float64 v1."
            )
    else:
        resolved_schema_version = schema_version
    expected_geometry_dtype = {
        DETECTION_OBSERVATION_CARDINALITY_LEGACY_FLOAT64_SCHEMA_VERSION: np.dtype(
            "<f8"
        ),
        DETECTION_OBSERVATION_CARDINALITY_SCHEMA_VERSION: np.dtype("<f4"),
    }.get(resolved_schema_version)
    if expected_geometry_dtype is None or geometry_dtype != expected_geometry_dtype:
        _fail(
            "Detection cardinality schema version does not match the exact "
            "persisted geometry dtype."
        )
    row_specs = {
        "frame_indices": (np.dtype("<i4"), ()),
        "source_acquisition_frame_index": (np.dtype("<i8"), ()),
        "bbox_norm_coords": (expected_geometry_dtype, (4,)),
        "bbox_img_xyxy": (expected_geometry_dtype, (4,)),
        "centers_img_xy": (expected_geometry_dtype, (2,)),
        "scores": (np.dtype("<f4"), ()),
        "class_ids": (np.dtype("<i4"), ()),
        "instance_key": (np.dtype("<u8"), ()),
    }
    try:
        row_nodes = {name: rowset[name] for name in row_specs}
        frame_counts_node = rowset["frame_counts"]
        n_detections_node = rowset["n_detections"]
    except Exception as exc:
        _fail(f"Detection cardinality arrays are incomplete: {exc}.")
    for name, node in (
        *row_nodes.items(),
        ("frame_counts", frame_counts_node),
        ("n_detections", n_detections_node),
    ):
        _require_child_path(node, rowset, name)

    row_count = _require_detection_observation_row_count(
        rowset,
        row_nodes["instance_key"],
    )
    row_values: dict[str, np.ndarray] = {}
    for name, (expected_dtype, trailing_shape) in row_specs.items():
        values = _array(node=row_nodes[name], label=f"detection cardinality {name}")
        expected_shape = (row_count, *trailing_shape)
        if values.dtype != expected_dtype or values.shape != expected_shape:
            _fail(
                f"Detection cardinality array {name!r} must have exact "
                f"shape/dtype {expected_shape}/{expected_dtype}; found "
                f"{values.shape}/{values.dtype}."
            )
        row_values[name] = values

    source_total_frames = int(acquisition.record.source_total_frames)
    frame_indices = row_values["frame_indices"]
    source_frames = row_values["source_acquisition_frame_index"]
    if not np.array_equal(frame_indices.astype(np.int64), source_frames):
        _fail(
            "Detection frame_indices and source_acquisition_frame_index must be "
            "the exact full-video identity mapping."
        )
    if np.any(source_frames < 0) or np.any(source_frames >= source_total_frames):
        _fail(
            "Detection observation rows contain a frame outside the exact "
            "acquisition frame domain."
        )

    frame_counts = _array(
        frame_counts_node,
        label="detection cardinality frame_counts",
    )
    n_detections = _array(
        n_detections_node,
        label="detection cardinality n_detections",
    )
    expected_frame_shape = (source_total_frames,)
    if (
        frame_counts.dtype != np.dtype("<i4")
        or frame_counts.shape != expected_frame_shape
        or n_detections.dtype != np.dtype("<i4")
        or n_detections.shape != expected_frame_shape
    ):
        _fail(
            "Detection frame_counts and n_detections must be exact int32 arrays "
            "over the complete acquisition frame domain."
        )
    expected_counts = np.bincount(
        frame_indices.astype(np.int64, copy=False),
        minlength=source_total_frames,
    ).astype(np.int32, copy=False)
    if not np.array_equal(frame_counts, expected_counts):
        _fail("Detection frame_counts differs from exact frame_indices cardinality.")
    if not np.array_equal(n_detections, expected_counts):
        _fail("Detection n_detections differs from exact frame_indices cardinality.")
    if int(frame_counts.sum(dtype=np.int64)) != row_count:
        _fail("Detection frame-count arrays do not sum to observation_row_count.")

    attrs = require_trusted_coordinate_attrs(
        rowset,
        label="Detection cardinality rowset",
    )
    summary = attrs.get("summary_statistics")
    expected_summary_counts = {
        "total_detections": row_count,
        "frames_with_detections": int(np.count_nonzero(frame_counts)),
        "frames_with_zero_detections": int(np.count_nonzero(frame_counts == 0)),
        "frames_with_multiple_detections": int(np.count_nonzero(frame_counts > 1)),
    }
    if not isinstance(summary, Mapping) or any(
        summary.get(name) != value for name, value in expected_summary_counts.items()
    ):
        _fail(
            "Detection summary count authorities disagree with the exact live rowset."
        )

    storage_validation = attrs.get(IMMUTABLE_YOLO_STORAGE_ATTR)
    if not isinstance(storage_validation, Mapping) or any(
        (
            storage_validation.get("schema_id") != IMMUTABLE_YOLO_STORAGE_SCHEMA,
            storage_validation.get("status") != "ok",
            storage_validation.get("stage") != "detect",
            storage_validation.get("row_count") != row_count,
            storage_validation.get("frame_count") != source_total_frames,
            storage_validation.get("instance_key_present") is not True,
            storage_validation.get("instance_key_unique") is not True,
        )
    ):
        _fail(
            "Detection immutable-storage count authority is missing or disagrees "
            "with the exact live rowset."
        )

    shard_write = attrs.get("detect_shard_write")
    row_shard_rows = attrs.get("detect_row_shard_rows")
    if row_shard_rows is None:
        if shard_write is not None:
            _fail("Regular-chunk detection must not carry a shard-write claim.")
    elif (
        type(row_shard_rows) is not int
        or row_shard_rows <= 0
        or not isinstance(shard_write, Mapping)
        or shard_write.get("status") != "complete"
        or shard_write.get("exact_match") is not True
        or shard_write.get("detection_row_count") != row_count
        or shard_write.get("frame_row_count") != source_total_frames
        or shard_write.get("source_sha256_by_array")
        != shard_write.get("destination_sha256_by_array")
        or set(shard_write.get("destination_sha256_by_array") or {})
        != {*row_specs, "frame_counts", "n_detections"}
    ):
        _fail(
            "Detection shard-write count authority is missing or disagrees with "
            "the exact live rowset."
        )

    return {
        "schema_id": DETECTION_OBSERVATION_CARDINALITY_SCHEMA_ID,
        "schema_version": resolved_schema_version,
        "observation_row_count": row_count,
        "source_total_frames": source_total_frames,
        "row_arrays": {
            name: _payload(row_nodes[name], values)
            for name, values in row_values.items()
        },
        "frame_count_arrays": {
            "frame_counts": _payload(frame_counts_node, frame_counts),
            "n_detections": _payload(n_detections_node, n_detections),
        },
        "count_authorities": {
            OBSERVATION_ROW_COUNT_ATTR: row_count,
            "summary_statistics": copy.deepcopy(dict(summary)),
            IMMUTABLE_YOLO_STORAGE_ATTR: copy.deepcopy(dict(storage_validation)),
            "detect_shard_write": (
                copy.deepcopy(dict(shard_write))
                if isinstance(shard_write, Mapping)
                else None
            ),
        },
        "proof": "exact_live_row_payloads_and_frame_bincount_v1",
    }


@proof_verification_operation
def publish_detection_observation_cardinality(
    rowset: Any,
    *,
    acquisition_frame: BoundAcquisitionCameraFrame,
) -> BoundCoordinateRecord:
    """Persist the exact cardinality/payload seal for every detection rowset."""

    record = _detection_observation_cardinality_record(
        rowset,
        acquisition_frame=acquisition_frame,
    )
    attrs_targets, snapshots = _attrs_snapshots(rowset)
    try:
        return stamp_and_bind_persisted_coordinate_record(
            rowset,
            record,
            attr_name=DETECTION_OBSERVATION_CARDINALITY_ATTR,
        )
    except BaseException as exc:
        _rollback_attrs(attrs_targets, snapshots, cause=exc)
        raise


def _require_detection_observation_cardinality(
    rowset: Any,
    *,
    acquisition_frame: BoundAcquisitionCameraFrame,
) -> BoundCoordinateRecord:
    cardinality = bind_persisted_coordinate_record(
        rowset,
        attr_name=DETECTION_OBSERVATION_CARDINALITY_ATTR,
    )
    persisted_schema_version = cardinality.record.get("schema_version")
    if type(persisted_schema_version) is not int:
        _fail("Persisted detection cardinality schema version is not exact.")
    expected = _detection_observation_cardinality_record(
        rowset,
        acquisition_frame=acquisition_frame,
        schema_version=persisted_schema_version,
    )
    if not _raw_equal(cardinality.record, expected):
        _fail(
            "Persisted detection cardinality seal differs from the exact live "
            "row arrays, frame-count arrays, or count authorities."
        )
    return cardinality


def _full_acquisition_identity_mapping_sha256(
    acquisition_frame: BoundAcquisitionCameraFrame,
) -> str:
    acquisition = require_bound_acquisition_camera_frame(acquisition_frame)
    mapping = np.arange(
        int(acquisition.record.source_total_frames),
        dtype=np.int64,
    )
    return hashlib.sha256(
        np.ascontiguousarray(mapping).view(np.uint8)
    ).hexdigest()


def _detection_instance_key_derivation_record(
    rowset: Any,
    key_node: Any,
    source_frame_index_node: Any,
    bbox_norm_node: Any,
    class_id_node: Any,
    *,
    acquisition_frame: BoundAcquisitionCameraFrame,
    acquisition_mapping: BoundCoordinateRecord,
) -> dict[str, Any]:
    """Build and verify the exact detect-origin ``instance_key`` derivation."""

    acquisition = require_bound_acquisition_camera_frame(acquisition_frame)
    mapping = verify_bound_coordinate_record(acquisition_mapping)
    if mapping.record.get("schema_id") != DETECTION_ACQUISITION_MAPPING_SCHEMA_ID:
        _fail("Detection instance-key derivation requires the acquisition mapping.")
    if mapping.archive_identity != archive_identity(rowset):
        _fail("Detection instance-key derivation and mapping use different archives.")
    for node, name in (
        (key_node, "instance_key"),
        (source_frame_index_node, "source_acquisition_frame_index"),
        (bbox_norm_node, "bbox_norm_coords"),
        (class_id_node, "class_ids"),
    ):
        _require_child_path(node, rowset, name)
        if archive_identity(node) != archive_identity(rowset):
            _fail(f"Detection {name} and its rowset use different archives.")

    keys = _array(key_node, label="detection instance_key")
    source_frames = _array(
        source_frame_index_node,
        label="detection source acquisition frame index",
    )
    bbox_norm = _array(bbox_norm_node, label="detection normalized bbox")
    class_ids = _array(class_id_node, label="detection class id")
    row_count = keys.shape[0] if keys.ndim == 1 else -1
    if keys.dtype != np.dtype("<u8") or keys.ndim != 1:
        _fail("Detection instance_key must be exact uint64 rank 1.")
    if (
        source_frames.dtype != np.dtype("<i8")
        or source_frames.shape != (row_count,)
        or np.any(source_frames < 0)
    ):
        _fail(
            "Detection instance-key source frames must be exact nonnegative "
            "int64 rows."
        )
    if bbox_norm.dtype.kind != "f" or bbox_norm.shape != (row_count, 4):
        _fail("Detection instance-key bboxes must be floating (N,4) rows.")
    if class_ids.dtype != np.dtype("<i4") or class_ids.shape != (row_count,):
        _fail("Detection instance-key class IDs must be exact int32 rows.")

    expected_keys = mint_detection_instance_keys(
        recording_identity=acquisition.record.recording_id,
        frame_indices=source_frames,
        bbox_norm_coords=bbox_norm,
        class_ids=class_ids,
    )
    if not np.array_equal(keys, expected_keys):
        _fail(
            "Detection instance_key is not the exact detect-origin derivation "
            "from the sealed acquisition recording, frame, bbox, and class."
        )

    attrs = require_trusted_coordinate_attrs(
        rowset,
        label="Detection instance-key rowset",
    )
    expected_mapping_source = (
        f"{acquisition.record_ref}#full_untrimmed_video_decode_identity_v1"
    )
    expected_mapping_sha256 = _full_acquisition_identity_mapping_sha256(acquisition)
    expected_attrs = {
        "instance_key_algorithm": INSTANCE_KEY_ALGORITHM,
        "instance_key_recording_identity": acquisition.record.recording_id,
        "instance_key_frame_domain": "recording_parent_frame_index",
        "instance_key_bbox_quantization": int(INSTANCE_KEY_BBOX_QUANTIZATION),
        "instance_key_duplicate_policy": INSTANCE_KEY_DUPLICATE_POLICY,
        "instance_key_frame_mapping_source": expected_mapping_source,
        "instance_key_frame_mapping_sha256": expected_mapping_sha256,
    }
    mismatched_attrs = sorted(
        name for name, value in expected_attrs.items() if attrs.get(name) != value
    )
    if mismatched_attrs:
        _fail(
            "Detection instance-key metadata disagrees with the sealed acquisition "
            f"identity or mapping: fields={mismatched_attrs!r}."
        )

    return {
        "schema_id": DETECTION_INSTANCE_KEY_DERIVATION_SCHEMA_ID,
        "schema_version": DETECTION_INSTANCE_KEY_DERIVATION_SCHEMA_VERSION,
        "operation": INSTANCE_KEY_ALGORITHM,
        "recording_id": acquisition.record.recording_id,
        "acquisition_camera_frame": {
            "record_ref": acquisition.record_ref,
            "record_sha256": acquisition.record_sha256,
        },
        "acquisition_mapping": {
            "record_ref": mapping.record_ref,
            "record_sha256": mapping.record_sha256,
        },
        "instance_key_policy": expected_attrs,
        "source_acquisition_frame_index": _payload(
            source_frame_index_node,
            source_frames,
        ),
        "bbox_norm_coords": _payload(bbox_norm_node, bbox_norm),
        "class_ids": _payload(class_id_node, class_ids),
        "instance_key": _payload(key_node, keys),
        "proof": "exact_detect_origin_instance_key_recomputation_v1",
    }


@proof_verification_operation
def publish_detection_instance_key_derivation(
    rowset: Any,
    key_node: Any,
    source_frame_index_node: Any,
    bbox_norm_node: Any,
    class_id_node: Any,
    *,
    acquisition_frame: BoundAcquisitionCameraFrame,
    acquisition_mapping: BoundCoordinateRecord,
) -> BoundCoordinateRecord:
    """Persist the acquisition-bound semantic derivation of detection row keys."""

    record = _detection_instance_key_derivation_record(
        rowset,
        key_node,
        source_frame_index_node,
        bbox_norm_node,
        class_id_node,
        acquisition_frame=acquisition_frame,
        acquisition_mapping=acquisition_mapping,
    )
    attrs_targets, snapshots = _attrs_snapshots(rowset)
    try:
        return stamp_and_bind_persisted_coordinate_record(
            rowset,
            record,
            attr_name=DETECTION_INSTANCE_KEY_DERIVATION_ATTR,
        )
    except BaseException as exc:
        _rollback_attrs(attrs_targets, snapshots, cause=exc)
        raise


def _require_detection_instance_key_derivation(
    rowset: Any,
    key_node: Any,
    source_frame_index_node: Any,
    bbox_norm_node: Any,
    class_id_node: Any,
    *,
    acquisition_frame: BoundAcquisitionCameraFrame,
    acquisition_mapping: BoundCoordinateRecord,
) -> BoundCoordinateRecord:
    derivation = bind_persisted_coordinate_record(
        rowset,
        attr_name=DETECTION_INSTANCE_KEY_DERIVATION_ATTR,
    )
    expected = _detection_instance_key_derivation_record(
        rowset,
        key_node,
        source_frame_index_node,
        bbox_norm_node,
        class_id_node,
        acquisition_frame=acquisition_frame,
        acquisition_mapping=acquisition_mapping,
    )
    if derivation.record != expected:
        _fail(
            "Persisted detection instance-key derivation differs from the exact "
            "sealed acquisition identity or live input arrays."
        )
    return derivation


def _empty_detection_observation_record(
    rowset: Any,
    *,
    acquisition_frame: BoundAcquisitionCameraFrame,
    decoded_frame_count: int,
    decode_domain_proof: str,
) -> dict[str, Any]:
    """Build the only supported proof record for a canonical empty detection."""

    row_shapes_and_dtypes = {
        "frame_indices": ((0,), np.dtype("<i4")),
        "source_acquisition_frame_index": ((0,), np.dtype("<i8")),
        "bbox_norm_coords": ((0, 4), np.dtype("<f8")),
        "bbox_img_xyxy": ((0, 4), np.dtype("<f8")),
        "centers_img_xy": ((0, 2), np.dtype("<f8")),
        "scores": ((0,), np.dtype("<f4")),
        "class_ids": ((0,), np.dtype("<i4")),
        "instance_key": ((0,), np.dtype("<u8")),
    }
    row_names = tuple(row_shapes_and_dtypes)
    try:
        row_nodes = {name: rowset[name] for name in row_names}
        frame_counts_node = rowset["frame_counts"]
        n_detections_node = rowset["n_detections"]
    except Exception as exc:
        _fail(f"Empty detection observation arrays are incomplete: {exc}.")
    for name, node in (
        *row_nodes.items(),
        ("frame_counts", frame_counts_node),
        ("n_detections", n_detections_node),
    ):
        _require_child_path(node, rowset, name)

    observation_count = _require_detection_observation_row_count(
        rowset,
        row_nodes["instance_key"],
    )
    if observation_count != 0:
        _fail("An empty-observation declaration requires exactly zero rows.")
    for name, node in row_nodes.items():
        values = _array(node, label=f"empty detection {name}")
        expected_shape, expected_dtype = row_shapes_and_dtypes[name]
        if values.shape != expected_shape or values.dtype != expected_dtype:
            _fail(
                f"Empty detection array {name!r} must have exact shape/dtype "
                f"{expected_shape}/{expected_dtype}; found "
                f"{values.shape}/{values.dtype}."
            )

    source_total_frames = int(acquisition_frame.record.source_total_frames)
    if type(decoded_frame_count) is not int or decoded_frame_count != source_total_frames:
        _fail(
            "Empty-observation declaration requires the complete acquisition "
            f"frame domain: decoded={decoded_frame_count!r}, "
            f"source_total={source_total_frames}."
        )
    if (
        type(decode_domain_proof) is not str
        or decode_domain_proof not in SUPPORTED_DETECTION_DECODE_DOMAIN_PROOFS
    ):
        _fail(
            "Empty-observation declaration requires a supported decode-domain "
            f"proof; found {decode_domain_proof!r}."
        )

    frame_counts = _array(
        frame_counts_node,
        label="empty detection frame_counts",
    )
    n_detections = _array(
        n_detections_node,
        label="empty detection n_detections",
    )
    if (
        frame_counts.shape != (source_total_frames,)
        or frame_counts.dtype != np.dtype("<i4")
        or np.any(frame_counts != 0)
        or n_detections.shape != frame_counts.shape
        or n_detections.dtype != frame_counts.dtype
        or not np.array_equal(n_detections, frame_counts)
    ):
        _fail(
            "Empty-observation declaration requires matching exact int32 zero "
            "frame_counts and n_detections arrays for every acquisition frame."
        )

    attrs = require_trusted_coordinate_attrs(
        rowset,
        label="Empty detection observation rowset",
    )
    summary = attrs.get("summary_statistics")
    if not isinstance(summary, Mapping):
        _fail("Empty detection summary_statistics must be an exact mapping.")
    expected_summary_counts = {
        "total_detections": 0,
        "frames_with_detections": 0,
        "frames_with_zero_detections": source_total_frames,
        "frames_with_multiple_detections": 0,
    }
    if any(summary.get(name) != value for name, value in expected_summary_counts.items()):
        _fail("Empty detection summary count authorities disagree with zero rows.")

    storage_validation = attrs.get(IMMUTABLE_YOLO_STORAGE_ATTR)
    if not isinstance(storage_validation, Mapping) or any(
        (
            storage_validation.get("schema_id") != IMMUTABLE_YOLO_STORAGE_SCHEMA,
            storage_validation.get("status") != "ok",
            storage_validation.get("stage") != "detect",
            storage_validation.get("row_count") != 0,
            storage_validation.get("frame_count") != source_total_frames,
            storage_validation.get("instance_key_present") is not True,
            storage_validation.get("instance_key_unique") is not True,
        )
    ):
        _fail(
            "Empty detection immutable-storage count authority is missing or "
            "disagrees with the exact empty rowset."
        )

    shard_write = attrs.get("detect_shard_write")
    row_shard_rows = attrs.get("detect_row_shard_rows")
    if row_shard_rows is None:
        if shard_write is not None:
            _fail("Regular-chunk empty detection must not carry a shard-write claim.")
    elif (
        type(row_shard_rows) is not int
        or row_shard_rows <= 0
        or not isinstance(shard_write, Mapping)
        or shard_write.get("status") != "complete"
        or shard_write.get("exact_match") is not True
        or shard_write.get("detection_row_count") != 0
        or shard_write.get("frame_row_count") != source_total_frames
        or shard_write.get("source_sha256_by_array")
        != shard_write.get("destination_sha256_by_array")
    ):
        _fail(
            "Empty detection shard-write count authority is missing or disagrees "
            "with the exact empty rowset."
        )

    run_decode_proof = attrs.get("decode_domain_proof")
    timing = attrs.get("timing_summary")
    if (
        run_decode_proof != decode_domain_proof
        or not isinstance(timing, Mapping)
        or timing.get("decode_domain_proof") != decode_domain_proof
        or timing.get("frames_processed") != source_total_frames
    ):
        _fail(
            "Empty detection decode/count authorities disagree with the exact "
            "full-domain proof."
        )
    backend = attrs.get("decode_backend_effective")
    supported_backend = (
        (decode_domain_proof == "opencv_stream_eof_and_exact_count_v1" and backend == "opencv")
        or (
            decode_domain_proof == "decord_index_domain_and_exact_batches_v1"
            and backend in {"decord_cpu", "decord_gpu"}
        )
        or (
            decode_domain_proof == "pynvvc_exact_count_and_eof_probe_v1"
            and backend in {"pynvvc_luma_rgb", "pynvvc_nv12_rgb"}
        )
    )
    if not supported_backend or timing.get("decode_backend_effective") != backend:
        _fail("Empty detection decode proof does not match its exact decoder backend.")

    row_payloads = {
        name: _payload(
            node,
            _array(node, label=f"empty detection {name}"),
        )
        for name, node in row_nodes.items()
    }
    return {
        "schema_id": EMPTY_OBSERVATION_DECLARATION_SCHEMA_ID,
        "schema_version": EMPTY_OBSERVATION_DECLARATION_SCHEMA_VERSION,
        "is_empty": True,
        "observation_row_count": 0,
        "decoded_frame_count": source_total_frames,
        "source_total_frames": source_total_frames,
        "decode_domain_proof": decode_domain_proof,
        "model_result_cardinality_proof": "per_batch_exact_v1",
        "model_result_orig_shape_proof": "per_input_exact_v1",
        "row_arrays": row_payloads,
        "frame_count_arrays": {
            "frame_counts": _payload(frame_counts_node, frame_counts),
            "n_detections": _payload(n_detections_node, n_detections),
        },
        "count_authorities": {
            OBSERVATION_ROW_COUNT_ATTR: 0,
            "summary_statistics": copy.deepcopy(dict(summary)),
            IMMUTABLE_YOLO_STORAGE_ATTR: copy.deepcopy(dict(storage_validation)),
            "detect_shard_write": (
                copy.deepcopy(dict(shard_write))
                if isinstance(shard_write, Mapping)
                else None
            ),
        },
        "decode_authority": {
            "decode_backend_effective": backend,
            "decode_domain_proof": decode_domain_proof,
            "timing_summary": copy.deepcopy(dict(timing)),
        },
        "proof": "full_acquisition_domain_processed_with_zero_observations_v1",
    }


@proof_verification_operation
def publish_empty_detection_observation_declaration(
    rowset: Any,
    *,
    acquisition_frame: BoundAcquisitionCameraFrame,
    decoded_frame_count: int,
    decode_domain_proof: str,
) -> BoundCoordinateRecord:
    """Persist and bind a full-domain proof for one genuine empty detection run."""

    record = _empty_detection_observation_record(
        rowset,
        acquisition_frame=acquisition_frame,
        decoded_frame_count=decoded_frame_count,
        decode_domain_proof=decode_domain_proof,
    )
    attrs_targets, snapshots = _attrs_snapshots(rowset)
    try:
        return stamp_and_bind_persisted_coordinate_record(
            rowset,
            record,
            attr_name=EMPTY_OBSERVATION_DECLARATION_ATTR,
        )
    except BaseException as exc:
        _rollback_attrs(attrs_targets, snapshots, cause=exc)
        raise


def _require_detection_observation_publication(
    rowset: Any,
    key_node: Any,
    *,
    acquisition_frame: BoundAcquisitionCameraFrame,
) -> None:
    """Require exact row cardinality and the zero-row full-domain proof regime."""

    observation_count = _require_detection_observation_row_count(rowset, key_node)
    attrs = require_trusted_coordinate_attrs(
        rowset,
        label="Detection observation rowset",
    )
    digest_attr = f"{EMPTY_OBSERVATION_DECLARATION_ATTR}_sha256"
    has_record = EMPTY_OBSERVATION_DECLARATION_ATTR in attrs
    has_digest = digest_attr in attrs
    if observation_count > 0:
        if has_record or has_digest:
            _fail(
                "A nonempty detection output must not declare an empty-observation "
                "full-domain proof."
            )
        return
    if not has_record or not has_digest:
        _fail(
            "A zero-row detection output requires an exact persisted "
            "empty-observation declaration and digest."
        )

    declaration = bind_persisted_coordinate_record(
        rowset,
        attr_name=EMPTY_OBSERVATION_DECLARATION_ATTR,
    )
    stored_proof = declaration.record.get("decode_domain_proof")
    expected = _empty_detection_observation_record(
        rowset,
        acquisition_frame=acquisition_frame,
        decoded_frame_count=int(acquisition_frame.record.source_total_frames),
        decode_domain_proof=stored_proof,
    )
    if not _raw_equal(declaration.record, expected):
        differing = sorted(
            name
            for name in set(declaration.record) | set(expected)
            if name not in declaration.record
            or name not in expected
            or not _raw_equal(
                declaration.record[name],
                expected[name],
            )
        )
        _fail(
            "Persisted empty-observation declaration differs from the exact "
            f"acquisition domain or live zero-row arrays; fields={differing!r}."
        )


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


def _collection_proxy_successor_mapping_record(
    rowset: Any,
    *,
    historical: BoundHistoricalMergedCollectionProxyV1,
    acquisition: BoundAcquisitionCameraFrame,
) -> dict[str, Any]:
    source = require_bound_historical_merged_collection_proxy_v1(historical)
    acquisition = require_bound_acquisition_camera_frame(acquisition)
    if (
        source.archive_identity != archive_identity(rowset)
        or source.camera_id != acquisition.record.camera_id
    ):
        _fail("Collection-proxy successor and historical source span archives or cameras.")
    attrs = require_trusted_coordinate_attrs(
        rowset,
        label="Collection-proxy coordinate successor",
    )
    if (
        attrs.get("schema") != COLLECTION_PROXY_SUCCESSOR_RUN_SCHEMA
        or attrs.get("source_kind") != COLLECTION_PROXY_SUCCESSOR_SOURCE_KIND
        or attrs.get("historical_source_rowset_path") != source.rowset_path
    ):
        _fail("Collection-proxy successor identity is absent or unsupported.")
    array_names = (
        "instance_key",
        "frame_indices",
        "source_frame_indices",
        "source_acquisition_frame_index",
        "source_proxy_crop_run_index",
        "source_proxy_crop_row_ids",
        "bbox_norm_coords",
    )
    row_nodes: dict[str, Any] = {}
    row_values: dict[str, np.ndarray] = {}
    for name in array_names:
        try:
            node = rowset[name]
        except Exception as exc:
            _fail(f"Collection-proxy successor array {name!r} is absent: {exc}.")
        values = _array(node, label=f"collection-proxy successor {name}")
        source_values = source.read_array(name)
        if values.dtype != source_values.dtype or values.shape != source_values.shape:
            _fail(f"Collection-proxy successor {name!r} changed dtype or shape.")
        if not np.array_equal(values, source_values, equal_nan=values.dtype.kind in "fc"):
            _fail(f"Collection-proxy successor {name!r} differs from historical source.")
        row_nodes[name] = node
        row_values[name] = values
    if row_values["instance_key"].shape != (source.row_count,):
        _fail("Collection-proxy successor row count differs from historical source.")
    historical_mapping = source.acquisition_mapping.record
    return {
        "schema_id": COLLECTION_PROXY_SUCCESSOR_MAPPING_SCHEMA_ID,
        "schema_version": COLLECTION_PROXY_SUCCESSOR_MAPPING_SCHEMA_VERSION,
        "operation": COLLECTION_PROXY_SUCCESSOR_MAPPING_OPERATION,
        "direction": "historical_merged_proxy_row_to_current_observation_row",
        "historical_source": {
            "rowset_ref": f"/{source.rowset_path}",
            "acquisition_mapping": {
                "record_ref": source.acquisition_mapping.record_ref,
                "record_sha256": source.acquisition_mapping.record_sha256,
            },
            "bbox_projection": {
                "record_ref": source.bbox_projection.record_ref,
                "record_sha256": source.bbox_projection.record_sha256,
            },
            "center_derivation": {
                "record_ref": source.center_derivation.record_ref,
                "record_sha256": source.center_derivation.record_sha256,
            },
        },
        "row_arrays": {
            name: _payload(row_nodes[name], row_values[name]) for name in array_names
        },
        "source_proxy_crop_runs": list(
            historical_mapping["source_proxy_crop_runs"]
        ),
        "source_refined_run_paths": list(
            historical_mapping["source_refined_run_paths"]
        ),
        "source_collection_id": historical_mapping["source_collection_id"],
        "acquisition_camera_frame": {
            "record_ref": acquisition.record_ref,
            "record_sha256": acquisition.record_sha256,
        },
        "source_total_frames": int(acquisition.record.source_total_frames),
        "proof": "exact_historical_payload_digest_and_source_row_revalidation_v1",
    }


@proof_verification_operation
def publish_collection_proxy_successor_mapping(
    rowset: Any,
    *,
    historical_source: BoundHistoricalMergedCollectionProxyV1,
    acquisition_frame: BoundAcquisitionCameraFrame,
) -> BoundCoordinateRecord:
    """Bind copied historical rows without treating v1 geometry as current."""

    record = _collection_proxy_successor_mapping_record(
        rowset,
        historical=historical_source,
        acquisition=acquisition_frame,
    )
    return stamp_and_bind_persisted_coordinate_record(
        rowset,
        record,
        attr_name=COLLECTION_PROXY_SUCCESSOR_MAPPING_ATTR,
    )


def _require_collection_proxy_successor_mapping(
    rowset: Any,
    mapping: BoundCoordinateRecord,
    *,
    root_node: Any,
    acquisition: BoundAcquisitionCameraFrame,
) -> BoundHistoricalMergedCollectionProxyV1:
    record = mapping.record
    historical_pointer = record.get("historical_source")
    rowset_ref = (
        historical_pointer.get("rowset_ref")
        if isinstance(historical_pointer, Mapping)
        else None
    )
    if not isinstance(rowset_ref, str) or not rowset_ref.startswith("/"):
        _fail("Collection-proxy successor lacks an exact historical rowset reference.")
    historical = load_historical_merged_collection_proxy_v1(
        root_node,
        rowset_ref[1:],
    )
    expected = _collection_proxy_successor_mapping_record(
        rowset,
        historical=historical,
        acquisition=acquisition,
    )
    if mapping.record != expected:
        _fail("Collection-proxy successor mapping differs from exact live lineage.")
    return historical


def _sampled_training_source_path(
    value: Any,
    *,
    run_family: str,
    field_name: str,
) -> str:
    path = str(value).strip().strip("/")
    parts = path.split("/")
    if len(parts) != 2 or parts[0] != run_family or not parts[1]:
        _fail(f"{field_name} must be one exact {run_family}/<run> path.")
    return path


def _sampled_training_detection_selection_record(
    root_node: Any,
    rowset: Any,
    *,
    acquisition: BoundAcquisitionCameraFrame,
) -> dict[str, Any]:
    """Recompute one sampled strong-single selection from exact live arrays."""

    attrs = require_trusted_coordinate_attrs(
        rowset,
        label="Sampled-training detection rowset",
    )
    if (
        attrs.get("schema") != SAMPLED_TRAINING_DETECTION_RUN_SCHEMA
        or attrs.get("source_kind") != SAMPLED_TRAINING_DETECTION_SOURCE_KIND
    ):
        _fail("Sampled-training detection run schema is unsupported.")
    source_detection_path = _sampled_training_source_path(
        attrs.get("source_detection_path"),
        run_family="detect_runs",
        field_name="source_detection_path",
    )
    source_crop_path = _sampled_training_source_path(
        attrs.get("source_proposal_crop_path"),
        run_family="crop_runs",
        field_name="source_proposal_crop_path",
    )
    _persisted_node(
        root_node,
        source_detection_path,
        label="sampled source detection artifact",
    )
    _persisted_node(
        root_node,
        source_crop_path,
        label="sampled source proposal crop",
    )
    policy = attrs.get("selection_policy")
    if not isinstance(policy, Mapping):
        _fail("Sampled-training detection selection_policy is missing.")
    target_size = policy.get("target_roi_size_height_width")
    if (
        not isinstance(target_size, (list, tuple))
        or len(target_size) != 2
        or any(type(value) is not int or value <= 0 for value in target_size)
    ):
        _fail("Sampled-training target ROI size is invalid.")
    minimum_score = policy.get("minimum_score_inclusive")
    minimum_iou = policy.get("minimum_proposal_iou_inclusive")
    policy_schema_version = policy.get("schema_version")
    try:
        expected_policy = strong_single_policy_record(
            minimum_score=float(minimum_score),
            minimum_proposal_iou=float(minimum_iou),
            target_roi_size=(int(target_size[0]), int(target_size[1])),
            policy_schema_version=policy_schema_version,
        )
    except (TypeError, ValueError) as exc:
        _fail(f"Sampled-training selection policy is invalid: {exc}.")
    if dict(policy) != expected_policy:
        _fail("Sampled-training selection policy is not its exact canonical form.")

    source_nodes = {
        "crop_frame_indices": _persisted_node(
            root_node,
            f"{source_crop_path}/frame_indices",
            label="sampled crop acquisition frames",
        ),
        "proposal_bbox_img_xyxy": _persisted_node(
            root_node,
            f"{source_crop_path}/bbox_img_xyxy",
            label="sampled proposal image bbox",
        ),
        "target_roi_top_left_xy": _persisted_node(
            root_node,
            f"{source_crop_path}/roi_coordinates_full",
            label="sampled target ROI placement",
        ),
        "detection_source_frame_indices": _persisted_node(
            root_node,
            f"{source_detection_path}/source_frame_indices",
            label="sampled detector acquisition frames",
        ),
        "detection_bbox_norm_coords": _persisted_node(
            root_node,
            f"{source_detection_path}/bbox_norm_coords",
            label="sampled detector normalized bboxes",
        ),
        "detection_scores": _persisted_node(
            root_node,
            f"{source_detection_path}/scores",
            label="sampled detector scores",
        ),
        "detection_class_ids": _persisted_node(
            root_node,
            f"{source_detection_path}/class_ids",
            label="sampled detector class IDs",
        ),
    }
    source_values = {
        name: _array(node, label=f"sampled selection source {name}")
        for name, node in source_nodes.items()
    }
    selection = select_strong_single_detections(
        crop_source_acquisition_frame_index=source_values["crop_frame_indices"],
        proposal_bbox_img_xyxy=source_values["proposal_bbox_img_xyxy"],
        target_roi_top_left_xy=source_values["target_roi_top_left_xy"],
        target_roi_size=(int(target_size[0]), int(target_size[1])),
        detection_source_acquisition_frame_index=source_values[
            "detection_source_frame_indices"
        ],
        detection_bbox_norm_coords=source_values["detection_bbox_norm_coords"],
        detection_scores=source_values["detection_scores"],
        source_width=int(acquisition.record.width_px),
        source_height=int(acquisition.record.height_px),
        minimum_score=float(minimum_score),
        minimum_proposal_iou=float(minimum_iou),
        policy_schema_version=policy_schema_version,
    )

    base = canonical_node_path(rowset)
    output_nodes = {
        name: _persisted_node(
            root_node,
            f"{base}/{name}",
            label=f"sampled detection output {name}",
        )
        for name in (
            "instance_key",
            "source_acquisition_frame_index",
            "bbox_norm_coords",
            "bbox_img_xyxy",
            "centers_img_xy",
            "scores",
            "class_ids",
            "source_training_crop_row_index",
            "source_detection_row_index",
        )
    }
    output_values = {
        name: _array(node, label=f"sampled detection output {name}")
        for name, node in output_nodes.items()
    }
    receipt_nodes = {
        name: _persisted_node(
            root_node,
            f"{base}/selection_receipt/{name}",
            label=f"sampled selection receipt {name}",
        )
        for name in (
            "candidate_count",
            "selected_detection_row_index",
            "selected_score",
            "proposal_iou",
            "included",
            "reason_code",
        )
    }
    receipt_values = {
        name: _array(node, label=f"sampled selection receipt {name}")
        for name, node in receipt_nodes.items()
    }
    expected_receipt = {
        "candidate_count": selection.candidate_count,
        "selected_detection_row_index": selection.selected_detection_row_index,
        "selected_score": selection.selected_score,
        "proposal_iou": selection.proposal_iou,
        "included": selection.included,
        "reason_code": selection.reason_code,
    }
    for name, expected in expected_receipt.items():
        if not np.array_equal(receipt_values[name], expected, equal_nan=True):
            _fail(f"Sampled selection receipt {name!r} differs from recomputation.")

    accepted_crop_rows = selection.accepted_crop_row_indices
    accepted_detection_rows = selection.accepted_detection_row_indices
    detect_frames = source_values["detection_source_frame_indices"].astype(
        np.int64, copy=False
    )
    expected_frames = detect_frames[accepted_detection_rows]
    expected_bbox_norm = source_values["detection_bbox_norm_coords"][
        accepted_detection_rows
    ]
    expected_scores = source_values["detection_scores"][accepted_detection_rows]
    expected_class_ids = source_values["detection_class_ids"][
        accepted_detection_rows
    ]
    expected_keys = mint_detection_instance_keys(
        recording_identity=acquisition.record.recording_id,
        frame_indices=expected_frames,
        bbox_norm_coords=expected_bbox_norm,
        class_ids=expected_class_ids,
    )
    expected_outputs = {
        "instance_key": expected_keys,
        "source_acquisition_frame_index": expected_frames.astype(
            np.int64, copy=False
        ),
        "bbox_norm_coords": expected_bbox_norm,
        "scores": expected_scores,
        "class_ids": expected_class_ids,
        "source_training_crop_row_index": accepted_crop_rows,
        "source_detection_row_index": accepted_detection_rows,
    }
    for name, expected in expected_outputs.items():
        if not np.array_equal(output_values[name], expected, equal_nan=True):
            _fail(f"Sampled detection output {name!r} differs from exact selection.")
    declared_rows = attrs.get(OBSERVATION_ROW_COUNT_ATTR)
    if (
        type(declared_rows) is not int
        or declared_rows != selection.accepted_row_count
    ):
        _fail("Sampled detection observation_row_count differs from selection.")

    return {
        "schema_id": SAMPLED_TRAINING_DETECTION_SELECTION_SCHEMA_ID,
        "schema_version": SAMPLED_TRAINING_DETECTION_SELECTION_SCHEMA_VERSION,
        "operation": "strict_strong_single_sampled_detection_selection_v1",
        "recording_id": acquisition.record.recording_id,
        "acquisition_camera_frame": {
            "record_ref": acquisition.record_ref,
            "record_sha256": acquisition.record_sha256,
        },
        "source_detection_rowset_ref": f"/{source_detection_path}",
        "source_proposal_crop_rowset_ref": f"/{source_crop_path}",
        "selection_policy": copy.deepcopy(expected_policy),
        "source_arrays": {
            name: _payload(node, source_values[name])
            for name, node in source_nodes.items()
        },
        "selection_receipt": {
            name: _payload(node, receipt_values[name])
            for name, node in receipt_nodes.items()
        },
        "output_arrays": {
            name: _payload(node, output_values[name])
            for name, node in output_nodes.items()
        },
        "source_row_count": selection.source_row_count,
        "accepted_row_count": selection.accepted_row_count,
        "reason_counts": selection.reason_counts(),
        "reason_code_labels": {
            str(code): label for code, label in SELECTION_REASON_LABELS.items()
        },
        "proof": "exact_live_source_recomputation_and_output_projection_v1",
    }


@proof_verification_operation
def publish_sampled_training_detection_selection(
    root_node: Any,
    rowset: Any,
    *,
    acquisition_frame: BoundAcquisitionCameraFrame,
) -> BoundCoordinateRecord:
    """Seal the exact source, policy, all-row receipt, and accepted projection."""

    acquisition = require_bound_acquisition_camera_frame(acquisition_frame)
    record = _sampled_training_detection_selection_record(
        root_node,
        rowset,
        acquisition=acquisition,
    )
    return stamp_and_bind_persisted_coordinate_record(
        rowset,
        record,
        attr_name=SAMPLED_TRAINING_DETECTION_SELECTION_ATTR,
    )


def _load_detection_frame_evidence_for_rowset(
    root_node: Any,
    rowset: Any,
    *,
    acquisition: BoundAcquisitionCameraFrame,
) -> BoundDetectionFrameEvidence:
    camera_id = acquisition.record.camera_id
    camera = load_source_camera_pixel_frame_authority(
        _persisted_node(
            root_node,
            "analysis/coordinate_frames/source_camera/"
            f"{camera_id}/{SOURCE_CAMERA_POINT_PIXEL_CONVENTION}",
            label="source-camera frame authority",
        ),
        acquisition_frame=acquisition,
    )
    bbox_camera = load_source_camera_pixel_frame_authority(
        _persisted_node(
            root_node,
            "analysis/coordinate_frames/source_camera/"
            f"{camera_id}/{SOURCE_CAMERA_BBOX_PIXEL_CONVENTION}",
            label="source-camera bbox-edge frame authority",
        ),
        acquisition_frame=acquisition,
    )
    base = canonical_node_path(rowset)
    normalized = load_normalized_pixel_frame_authority(
        _persisted_node(
            root_node,
            f"{base}/coordinate_frames/source_camera_normalized",
            label="detection normalized frame",
        ),
        pixel_frame=bbox_camera,
    )
    matrix_node = _persisted_node(
        root_node,
        f"{base}/coordinate_transforms/source_camera_normalized_to_image",
        label="normalized-to-camera transform",
    )
    authority = load_bound_transform_authority(
        _persisted_node(
            root_node,
            f"{base}/coordinate_transforms/"
            "source_camera_normalized_to_image_authority",
            label="normalized-to-camera transform authority",
        ),
        payload_node=matrix_node,
        source_frame=normalized,
        target_frame=bbox_camera,
    )
    transform = load_bound_directed_transform_v2(
        matrix_node,
        authority=authority,
        source_frame=normalized,
        target_frame=bbox_camera,
    )
    return build_bound_detection_frame_evidence(
        source_camera_frame=camera,
        bbox_source_camera_frame=bbox_camera,
        normalized_frame=normalized,
        normalized_to_source_camera=resolve_bound_directed_transform_chain(
            (transform,)
        ),
    )


def _load_persisted_sampled_training_detection_geometry(
    root_node: Any,
    rowset_path: str,
    *,
    require_selector_eligible: bool = False,
) -> BoundDetectionObservationGeometry:
    """Load a complete selector-free sampled detection geometry authority."""

    rowset = _persisted_node(
        root_node,
        rowset_path,
        label="sampled-training detection rowset",
    )
    _require_complete_canonical_observation_rowset(
        rowset,
        run_family="sampled_detection_runs",
        label="Sampled-training detection rowset",
        require_selector_eligible=require_selector_eligible,
    )
    attrs = require_trusted_coordinate_attrs(
        rowset,
        label="Sampled-training detection rowset",
    )
    if (
        attrs.get("schema") != SAMPLED_TRAINING_DETECTION_RUN_SCHEMA
        or attrs.get("source_kind") != SAMPLED_TRAINING_DETECTION_SOURCE_KIND
    ):
        _fail("Sampled-training detection rowset schema is unsupported.")
    _, acquisition = load_persisted_acquisition_camera_authority(root_node)
    evidence = _load_detection_frame_evidence_for_rowset(
        root_node,
        rowset,
        acquisition=acquisition,
    )
    selection = bind_persisted_coordinate_record(
        rowset,
        attr_name=SAMPLED_TRAINING_DETECTION_SELECTION_ATTR,
    )
    expected = _sampled_training_detection_selection_record(
        root_node,
        rowset,
        acquisition=acquisition,
    )
    if selection.record != expected:
        _fail("Persisted sampled-training detection selection is stale.")
    base = canonical_node_path(rowset)
    return load_detection_observation_geometry(
        rowset,
        _persisted_node(root_node, f"{base}/instance_key", label="sampled instance_key"),
        _persisted_node(
            root_node,
            f"{base}/source_acquisition_frame_index",
            label="sampled acquisition frame",
        ),
        _persisted_node(
            root_node,
            f"{base}/bbox_norm_coords",
            label="sampled normalized bbox",
        ),
        _persisted_node(
            root_node,
            f"{base}/bbox_img_xyxy",
            label="sampled image bbox",
        ),
        _persisted_node(
            root_node,
            f"{base}/centers_img_xy",
            label="sampled image center",
        ),
        frame_evidence=evidence,
        source_lineage_records=(selection,),
    )


@proof_verification_operation
def load_persisted_sampled_training_detection_geometry(
    root_node: Any,
    rowset_path: str,
) -> BoundDetectionObservationGeometry:
    """Load one complete selector-ineligible sampled detection authority."""

    return _load_persisted_sampled_training_detection_geometry(
        root_node,
        rowset_path,
        require_selector_eligible=False,
    )


def _load_persisted_detection_observation_geometry(
    root_node: Any,
    rowset_path: str,
    *,
    require_selector_eligible: bool,
) -> BoundDetectionObservationGeometry:
    """Internal persisted loader supporting the producer's staged validation."""

    rowset = _persisted_node(root_node, rowset_path, label="detection rowset")
    _require_complete_canonical_observation_rowset(
        rowset,
        run_family="detect_runs",
        label="Detection rowset",
        require_selector_eligible=require_selector_eligible,
    )
    _, acquisition = load_persisted_acquisition_camera_authority(root_node)
    camera_id = acquisition.record.camera_id
    camera_node = _persisted_node(
        root_node,
        (
            "analysis/coordinate_frames/source_camera/"
            f"{camera_id}/{SOURCE_CAMERA_POINT_PIXEL_CONVENTION}"
        ),
        label="source-camera frame authority",
    )
    camera = load_source_camera_pixel_frame_authority(
        camera_node,
        acquisition_frame=acquisition,
    )
    bbox_camera_node = _persisted_node(
        root_node,
        (
            "analysis/coordinate_frames/source_camera/"
            f"{camera_id}/{SOURCE_CAMERA_BBOX_PIXEL_CONVENTION}"
        ),
        label="source-camera bbox-edge frame authority",
    )
    bbox_camera = load_source_camera_pixel_frame_authority(
        bbox_camera_node,
        acquisition_frame=acquisition,
    )
    normalized_node = _persisted_node(
        root_node,
        f"{canonical_node_path(rowset)}/coordinate_frames/source_camera_normalized",
        label="detection normalized frame",
    )
    normalized = load_normalized_pixel_frame_authority(
        normalized_node,
        pixel_frame=bbox_camera,
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
        target_frame=bbox_camera,
    )
    transform = load_bound_directed_transform_v2(
        matrix_node,
        authority=authority,
        source_frame=normalized,
        target_frame=bbox_camera,
    )
    evidence = build_bound_detection_frame_evidence(
        source_camera_frame=camera,
        bbox_source_camera_frame=bbox_camera,
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
    base = canonical_node_path(rowset)
    key_node = _persisted_node(
        root_node,
        f"{base}/instance_key",
        label="detection instance_key",
    )
    source_frame_index_node = _persisted_node(
        root_node,
        f"{base}/source_acquisition_frame_index",
        label="detection acquisition frame",
    )
    bbox_norm_node = _persisted_node(
        root_node,
        f"{base}/bbox_norm_coords",
        label="detection normalized bbox",
    )
    backend_result_projection = load_detection_backend_result_projection(
        rowset,
        bbox_norm_node,
        frame_evidence=evidence,
    )
    class_id_node = _persisted_node(
        root_node,
        f"{base}/class_ids",
        label="detection class IDs",
    )
    bbox_img_node = _persisted_node(
        root_node,
        f"{base}/bbox_img_xyxy",
        label="detection image bbox",
    )
    centers_img_node = _persisted_node(
        root_node,
        f"{base}/centers_img_xy",
        label="detection image centers",
    )
    instance_key_derivation = _require_detection_instance_key_derivation(
        rowset,
        key_node,
        source_frame_index_node,
        bbox_norm_node,
        class_id_node,
        acquisition_frame=acquisition,
        acquisition_mapping=mapping,
    )
    _require_detection_observation_cardinality(
        rowset,
        acquisition_frame=acquisition,
    )
    _require_detection_observation_publication(
        rowset,
        key_node,
        acquisition_frame=acquisition,
    )
    geometry = load_detection_observation_geometry(
        rowset,
        key_node,
        source_frame_index_node,
        bbox_norm_node,
        bbox_img_node,
        centers_img_node,
        frame_evidence=evidence,
        source_lineage_records=(
            mapping,
            backend_result_projection,
            instance_key_derivation,
        ),
    )
    return geometry


@proof_verification_operation
def load_persisted_detection_observation_geometry(
    root_node: Any,
    rowset_path: str,
) -> BoundDetectionObservationGeometry:
    """Load only a complete, selector-eligible canonical detection run."""

    return _load_persisted_detection_observation_geometry(
        root_node,
        rowset_path,
        require_selector_eligible=True,
    )


@proof_verification_operation
def load_persisted_ineligible_detection_observation_geometry(
    root_node: Any,
    rowset_path: str,
) -> BoundDetectionObservationGeometry:
    """Load one complete detection candidate that remains selector-ineligible.

    This value-level loader does not authorize scientific consumption by
    itself.  A caller must separately prove the exact immutable candidate or
    bundle authority that permits the named ineligible rowset.
    """

    return _load_persisted_detection_observation_geometry(
        root_node,
        rowset_path,
        require_selector_eligible=False,
    )


def _load_persisted_collection_proxy_successor_geometry(
    root_node: Any,
    rowset_path: str,
    *,
    require_selector_eligible: bool = True,
) -> BoundDetectionObservationGeometry:
    rowset = _persisted_node(
        root_node,
        rowset_path,
        label="collection-proxy coordinate successor",
    )
    _require_complete_canonical_observation_rowset(
        rowset,
        run_family="crop_runs",
        label="Collection-proxy coordinate successor",
        require_selector_eligible=require_selector_eligible,
    )
    attrs = require_trusted_coordinate_attrs(
        rowset,
        label="Collection-proxy coordinate successor",
    )
    if (
        attrs.get("schema") != COLLECTION_PROXY_SUCCESSOR_RUN_SCHEMA
        or attrs.get("source_kind") != COLLECTION_PROXY_SUCCESSOR_SOURCE_KIND
    ):
        _fail("Collection-proxy coordinate successor schema is unsupported.")
    _, acquisition = load_persisted_acquisition_camera_authority(root_node)
    camera_id = acquisition.record.camera_id
    point_camera = load_source_camera_pixel_frame_authority(
        _persisted_node(
            root_node,
            "analysis/coordinate_frames/source_camera/"
            f"{camera_id}/{SOURCE_CAMERA_POINT_PIXEL_CONVENTION}",
            label="successor source-camera point frame",
        ),
        acquisition_frame=acquisition,
    )
    bbox_camera = load_source_camera_pixel_frame_authority(
        _persisted_node(
            root_node,
            "analysis/coordinate_frames/source_camera/"
            f"{camera_id}/{SOURCE_CAMERA_BBOX_PIXEL_CONVENTION}",
            label="successor source-camera bbox frame",
        ),
        acquisition_frame=acquisition,
    )
    base = canonical_node_path(rowset)
    normalized = load_normalized_pixel_frame_authority(
        _persisted_node(
            root_node,
            f"{base}/coordinate_frames/source_camera_normalized",
            label="successor normalized frame",
        ),
        pixel_frame=bbox_camera,
    )
    matrix_node = _persisted_node(
        root_node,
        f"{base}/coordinate_transforms/source_camera_normalized_to_image",
        label="successor normalized-to-camera transform",
    )
    transform_authority = load_bound_transform_authority(
        _persisted_node(
            root_node,
            f"{base}/coordinate_transforms/source_camera_normalized_to_image_authority",
            label="successor normalized-to-camera transform authority",
        ),
        payload_node=matrix_node,
        source_frame=normalized,
        target_frame=bbox_camera,
    )
    transform = load_bound_directed_transform_v2(
        matrix_node,
        authority=transform_authority,
        source_frame=normalized,
        target_frame=bbox_camera,
    )
    evidence = build_bound_detection_frame_evidence(
        source_camera_frame=point_camera,
        bbox_source_camera_frame=bbox_camera,
        normalized_frame=normalized,
        normalized_to_source_camera=resolve_bound_directed_transform_chain(
            (transform,)
        ),
    )
    mapping = bind_persisted_coordinate_record(
        rowset,
        attr_name=COLLECTION_PROXY_SUCCESSOR_MAPPING_ATTR,
    )
    _require_collection_proxy_successor_mapping(
        rowset,
        mapping,
        root_node=root_node,
        acquisition=acquisition,
    )
    geometry = load_detection_observation_geometry(
        rowset,
        _persisted_node(root_node, f"{base}/instance_key", label="successor instance_key"),
        _persisted_node(
            root_node,
            f"{base}/source_acquisition_frame_index",
            label="successor acquisition frame",
        ),
        _persisted_node(
            root_node,
            f"{base}/bbox_norm_coords",
            label="successor normalized bbox",
        ),
        _persisted_node(
            root_node,
            f"{base}/bbox_img_xyxy",
            label="successor image bbox",
        ),
        _persisted_node(
            root_node,
            f"{base}/centers_img_xy",
            label="successor centers",
        ),
        frame_evidence=evidence,
        source_lineage_records=(mapping,),
    )
    return geometry


@proof_verification_operation
def load_persisted_collection_proxy_successor_geometry(
    root_node: Any,
    rowset_path: str,
) -> BoundDetectionObservationGeometry:
    """Load only a complete, current-v2 merged-proxy coordinate successor."""

    return _load_persisted_collection_proxy_successor_geometry(
        root_node,
        rowset_path,
        require_selector_eligible=True,
    )


@proof_verification_operation
def load_collection_proxy_successor_source_rowset(
    root_node: Any,
    rowset_path: str,
) -> str:
    """Return the exact historical rowset proven identical by one successor."""

    geometry = load_persisted_collection_proxy_successor_geometry(
        root_node,
        rowset_path,
    )
    matches = [
        record
        for record in geometry.source_lineage_records
        if record.record.get("schema_id")
        == COLLECTION_PROXY_SUCCESSOR_MAPPING_SCHEMA_ID
    ]
    if len(matches) != 1:
        _fail(
            "Collection-proxy successor geometry must carry exactly one verified "
            "historical-row mapping."
        )
    historical = matches[0].record.get("historical_source")
    rowset_ref = (
        historical.get("rowset_ref")
        if isinstance(historical, Mapping)
        else None
    )
    if (
        not isinstance(rowset_ref, str)
        or not rowset_ref.startswith("/crop_runs/")
        or len(rowset_ref.split("/")) != 3
    ):
        _fail("Collection-proxy successor historical rowset reference is invalid.")
    return rowset_ref[1:]


def _load_persisted_crop_source_observation_geometry(
    root_node: Any,
    source_path: str,
) -> BoundDetectionObservationGeometry:
    """Resolve the exact supported detection-source family for one crop."""

    if source_path.startswith("sampled_detection_runs/"):
        return load_persisted_sampled_training_detection_geometry(
            root_node,
            source_path,
        )
    return load_persisted_detection_observation_geometry(root_node, source_path)


def _load_persisted_crop_observation_geometry(
    root_node: Any,
    rowset_path: str,
    *,
    require_selector_eligible: bool = True,
) -> BoundCropObservationGeometry:
    """Internal crop loader supporting an explicit staged-validation phase."""

    rowset = _persisted_node(root_node, rowset_path, label="crop rowset")
    _require_complete_canonical_observation_rowset(
        rowset,
        run_family="crop_runs",
        label="Crop rowset",
        require_selector_eligible=require_selector_eligible,
    )
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
    source = _load_persisted_crop_source_observation_geometry(
        root_node,
        source_path,
    )
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


@proof_verification_operation
def load_persisted_crop_observation_geometry(
    root_node: Any,
    rowset_path: str,
) -> BoundCropObservationGeometry:
    """Load only a complete, selector-eligible canonical crop run."""

    return _load_persisted_crop_observation_geometry(
        root_node,
        rowset_path,
        require_selector_eligible=True,
    )


def _load_persisted_ordinary_crop_observation_geometry(
    root_node: Any,
    rowset_path: str,
    *,
    require_selector_eligible: bool = True,
) -> BoundCropObservationGeometry:
    """Resolve and fully validate one materialized ordinary crop run.

    The base crop loader deliberately remains useful to geometry-only and
    incremental compatibility surfaces.  Ordinary crop publication uses this
    stricter boundary so selection cannot expose a run whose materialized ROI,
    source-camera placement, ROI-local bbox, compatibility top-left surface,
    or direction-labelled transform has drifted.
    """

    crop = _load_persisted_crop_observation_geometry(
        root_node,
        rowset_path,
        require_selector_eligible=require_selector_eligible,
    )
    rowset = crop._rowset_node
    attrs = require_trusted_coordinate_attrs(
        rowset,
        label="Materialized ordinary crop rowset",
    )
    if attrs.get("crop_storage_mode") != "materialized":
        _fail(
            "Ordinary crop coordinate publication requires exact "
            "crop_storage_mode='materialized'."
        )
    base = canonical_node_path(rowset)
    roi_images_node = _persisted_node(
        root_node,
        f"{base}/roi_images",
        label="ordinary crop ROI images",
    )
    placement_node = _persisted_node(
        root_node,
        f"{base}/source_crop_xywh",
        label="ordinary crop source-camera placement",
    )
    bbox_roi_node = _persisted_node(
        root_node,
        f"{base}/bbox_roi_xyxy",
        label="ordinary crop ROI-local bbox",
    )
    top_left_node = _persisted_node(
        root_node,
        f"{base}/roi_coordinates_full",
        label="ordinary crop source-camera top-left compatibility surface",
    )
    point_camera = crop.source_geometry.frame_evidence.source_camera_frame
    point_ownership = load_crop_placement_ownership(
        placement_node,
        row_identity=crop.row_identity,
        source_camera_frame=point_camera,
    )
    point_roi_extent = bind_array_reference_extent(roi_images_node, units="px")
    point_roi_frame = load_roi_pixel_frame_authority(
        roi_images_node,
        reference_extent=point_roi_extent,
        crop_placement_ownership=point_ownership,
    )
    point_transform_authority = load_bound_transform_authority(
        placement_node,
        payload_node=placement_node,
        source_frame=point_roi_frame,
        target_frame=point_camera,
        row_identity=crop.row_identity,
    )
    point_transform = load_bound_directed_transform_v2(
        placement_node,
        authority=point_transform_authority,
        source_frame=point_roi_frame,
        target_frame=point_camera,
        row_identity=crop.row_identity,
    )
    resolve_bound_directed_transform_chain((point_transform,))

    bbox_camera = crop.source_geometry.frame_evidence.bbox_source_camera_frame
    bbox_ownership = load_crop_placement_ownership(
        placement_node,
        row_identity=crop.row_identity,
        source_camera_frame=bbox_camera,
        attr_name=CROP_PLACEMENT_PIXEL_EDGE_OWNERSHIP_ATTR,
    )
    bbox_frame_node = _persisted_node(
        root_node,
        f"{base}/{CROP_ROI_BBOX_EDGE_FRAME_RELATIVE_PATH}",
        label="ordinary crop ROI bbox-edge frame",
    )
    bbox_roi_extent = load_crop_roi_bbox_edge_reference_extent(
        bbox_frame_node,
        roi_images_node,
    )
    bbox_roi_frame = load_roi_pixel_frame_authority(
        bbox_frame_node,
        reference_extent=bbox_roi_extent,
        crop_placement_ownership=bbox_ownership,
    )
    bbox_transform_authority = load_bound_transform_authority(
        placement_node,
        payload_node=placement_node,
        source_frame=bbox_roi_frame,
        target_frame=bbox_camera,
        row_identity=crop.row_identity,
        attr_name=TRANSFORM_AUTHORITY_PIXEL_EDGE_ATTR,
    )
    bbox_transform = load_bound_directed_transform_v2(
        placement_node,
        authority=bbox_transform_authority,
        source_frame=bbox_roi_frame,
        target_frame=bbox_camera,
        row_identity=crop.row_identity,
        attr_name=DIRECTED_TRANSFORM_V2_PIXEL_EDGE_ATTR,
    )
    load_crop_roi_geometry(
        placement_node,
        bbox_roi_node,
        crop_geometry=crop,
        crop_placement_ownership=bbox_ownership,
        roi_frame=bbox_roi_frame,
        roi_to_source_camera=resolve_bound_directed_transform_chain(
            (bbox_transform,)
        ),
        roi_top_left_node=top_left_node,
        roi_top_left_placement_ownership=point_ownership,
    )
    return crop


@proof_verification_operation
def load_persisted_ordinary_crop_observation_geometry(
    root_node: Any,
    rowset_path: str,
) -> BoundCropObservationGeometry:
    """Load only a complete, eligible, fully validated ordinary crop run."""

    return _load_persisted_ordinary_crop_observation_geometry(
        root_node,
        rowset_path,
        require_selector_eligible=True,
    )


@proof_verification_operation
def load_persisted_source_camera_position_surface(
    root_node: Any,
    rowset_path: str,
) -> BoundSourceCameraPositionSurface:
    """Track-facing resolver for one canonical observation position rowset."""

    rowset = _persisted_node(root_node, rowset_path, label="position rowset")
    attrs = require_trusted_coordinate_attrs(rowset, label="Position rowset")
    has_mapping = DETECTION_ACQUISITION_MAPPING_ATTR in attrs
    has_selection = CROP_GEOMETRY_SELECTION_ATTR in attrs
    has_proxy_successor = COLLECTION_PROXY_SUCCESSOR_MAPPING_ATTR in attrs
    from fisheye.shared.manifest_crop_position_authority import (
        _is_manifest_bound_crop_position_rowset,
        _load_manifest_crop_position_proof,
        _require_manifest_crop_position_proof,
    )

    has_manifest_crop = _is_manifest_bound_crop_position_rowset(rowset)
    if sum((has_mapping, has_selection, has_proxy_successor, has_manifest_crop)) != 1:
        _fail(
            "Canonical position rowset must declare exactly one detection, crop, "
            "manifest-crop, or collection-proxy-successor lineage."
        )
    if has_mapping:
        geometry = load_persisted_detection_observation_geometry(
            root_node,
            rowset_path,
        )
    elif has_selection:
        geometry = load_persisted_crop_observation_geometry(root_node, rowset_path)
    elif has_proxy_successor:
        geometry = load_persisted_collection_proxy_successor_geometry(
            root_node,
            rowset_path,
        )
    else:
        proof = _require_manifest_crop_position_proof(
            _load_manifest_crop_position_proof(root_node, rowset_path)
        )
        return require_bound_source_camera_position_surface(
            _position_surface(proof.coordinates, proof.temporal_authority)
        )
    return require_bound_source_camera_position_surface(
        geometry.position_surface
    )


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
    "DETECTION_INSTANCE_KEY_DERIVATION_ATTR",
    "DETECTION_INSTANCE_KEY_DERIVATION_SCHEMA_ID",
    "DETECTION_INSTANCE_KEY_DERIVATION_SCHEMA_VERSION",
    "DETECTION_OBSERVATION_CARDINALITY_ATTR",
    "DETECTION_OBSERVATION_CARDINALITY_SCHEMA_ID",
    "DETECTION_OBSERVATION_CARDINALITY_LEGACY_FLOAT64_SCHEMA_VERSION",
    "DETECTION_OBSERVATION_CARDINALITY_SCHEMA_VERSION",
    "EMPTY_OBSERVATION_DECLARATION_ATTR",
    "EMPTY_OBSERVATION_DECLARATION_SCHEMA_ID",
    "EMPTY_OBSERVATION_DECLARATION_SCHEMA_VERSION",
    "OBSERVATION_ROW_COUNT_ATTR",
    "SUPPORTED_DETECTION_DECODE_DOMAIN_PROOFS",
    "CROP_GEOMETRY_SELECTION_ATTR",
    "CROP_GEOMETRY_SELECTION_OPERATION",
    "CROP_GEOMETRY_SELECTION_SCHEMA_ID",
    "CROP_GEOMETRY_SELECTION_SCHEMA_VERSION",
    "COLLECTION_PROXY_SUCCESSOR_MAPPING_ATTR",
    "COLLECTION_PROXY_SUCCESSOR_MAPPING_OPERATION",
    "COLLECTION_PROXY_SUCCESSOR_MAPPING_SCHEMA_ID",
    "COLLECTION_PROXY_SUCCESSOR_MAPPING_SCHEMA_VERSION",
    "COLLECTION_PROXY_SUCCESSOR_RUN_SCHEMA",
    "COLLECTION_PROXY_SUCCESSOR_SOURCE_KIND",
    "SAMPLED_TRAINING_DETECTION_RUN_SCHEMA",
    "SAMPLED_TRAINING_DETECTION_SELECTION_ATTR",
    "SAMPLED_TRAINING_DETECTION_SELECTION_SCHEMA_ID",
    "SAMPLED_TRAINING_DETECTION_SELECTION_SCHEMA_VERSION",
    "SAMPLED_TRAINING_DETECTION_SOURCE_KIND",
    "CROP_ROI_GEOMETRY_DERIVATION_ATTR",
    "CROP_ROI_GEOMETRY_DERIVATION_OPERATION",
    "CROP_ROI_GEOMETRY_DERIVATION_SCHEMA_ID",
    "CROP_ROI_GEOMETRY_DERIVATION_SCHEMA_VERSION",
    "CROP_ROI_TOP_LEFT_DERIVATION_ATTR",
    "CROP_ROI_TOP_LEFT_DERIVATION_OPERATION",
    "CROP_ROI_TOP_LEFT_DERIVATION_SCHEMA_ID",
    "CROP_ROI_TOP_LEFT_DERIVATION_SCHEMA_VERSION",
    "BoundCropObservationGeometry",
    "BoundDetectionFrameEvidence",
    "BoundDetectionObservationGeometry",
    "BoundSourceCameraPositionSurface",
    "CropRoiGeometryPublicationResult",
    "ObservationCoordinatePublicationError",
    "ObservationCoordinatePublicationCheckpoint",
    "build_bound_detection_frame_evidence",
    "capture_observation_coordinate_publication_checkpoint",
    "derive_detection_source_camera_geometry",
    "detection_observation_geometry_values",
    "load_crop_observation_geometry",
    "load_collection_proxy_successor_source_rowset",
    "load_crop_roi_geometry",
    "load_detection_observation_geometry",
    "load_persisted_crop_observation_geometry",
    "load_persisted_collection_proxy_successor_geometry",
    "load_persisted_detection_observation_geometry",
    "load_persisted_ineligible_detection_observation_geometry",
    "load_persisted_ordinary_crop_observation_geometry",
    "load_persisted_sampled_training_detection_geometry",
    "load_persisted_source_camera_position_surface",
    "publish_crop_observation_geometry",
    "publish_collection_proxy_successor_mapping",
    "publish_crop_roi_geometry",
    "publish_detection_observation_geometry",
    "publish_detection_observation_cardinality",
    "publish_detection_instance_key_derivation",
    "publish_sampled_training_detection_selection",
    "publish_empty_detection_observation_declaration",
    "require_bound_crop_observation_geometry",
    "require_bound_detection_frame_evidence",
    "require_bound_detection_observation_geometry",
    "require_bound_source_camera_position_surface",
    "restore_observation_coordinate_publication_checkpoint",
]
