"""Sealed subject-shape input adapter for recording subject-mask bundles."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from fisheye.shared.zarr.assignment_keypoint_rebinding import (
    ASSIGNMENT_CANONICAL_KEYPOINT_PROFILE,
    load_assignment_keypoint_rebinding_manifest,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.subject_mask_bundle_coordinate_authority import (
    BoundRecordingSubjectMaskCoordinateAuthority,
    load_recording_subject_mask_coordinate_authority,
)
from fisheye.shared.pixel_frame_authority import (
    BoundAcquisitionCameraFrame,
    BoundPixelFrameAuthority,
    load_persisted_acquisition_camera_authority,
    load_source_camera_pixel_frame_authority,
)
from fisheye.shared.proof_verification import verify_persisted_proof

SUBJECT_SHAPE_BUNDLE_SOURCE_SCHEMA_ID = (
    "palette.subject_shape.recording_mask_bundle_source"
)
SUBJECT_SHAPE_BUNDLE_SOURCE_SCHEMA_VERSION = 1
SUBJECT_SHAPE_BUNDLE_REBOUND_SOURCE_SCHEMA_VERSION = 2
SUBJECT_SHAPE_BUNDLE_SOURCE_KIND = "recording_subject_mask_bundle_v3"
SUBJECT_SHAPE_REQUIRED_MASK_COMPONENTS = (
    "subject_body",
    "swim_bladder",
    "eye_left",
    "eye_right",
)
_ASSIGNMENT_REBINDING_PREFIX = "subject_mask_assignment_keypoint_rebinding_runs/"


class SubjectShapeBundleSourceError(ValueError):
    """Raised when a bundle cannot serve as an exact subject-shape input."""


_BOUND_SOURCE_SEAL = object()


def assignment_rebinding_run_id_from_source_record(
    value: Mapping[str, Any] | None,
) -> str | None:
    """Return the exact optional rebinding ID sealed by a source record."""

    if not isinstance(value, Mapping) or value.get("schema_version") == 1:
        return None
    if value.get("schema_version") != 2:
        raise SubjectShapeBundleSourceError(
            "Unsupported subject-shape bundle source-record version."
        )
    assignment = value.get("assignment_keypoints")
    path = (
        assignment.get("rebinding_run_path")
        if isinstance(assignment, Mapping)
        else None
    )
    if (
        not isinstance(path, str)
        or not path.startswith(_ASSIGNMENT_REBINDING_PREFIX)
        or "/" in path[len(_ASSIGNMENT_REBINDING_PREFIX) :]
        or not path[len(_ASSIGNMENT_REBINDING_PREFIX) :]
    ):
        raise SubjectShapeBundleSourceError(
            "Bundle source record lacks one exact assignment rebinding path."
        )
    return path[len(_ASSIGNMENT_REBINDING_PREFIX) :]


def _array_declarations(
    authority: BoundRecordingSubjectMaskCoordinateAuthority,
) -> dict[str, object]:
    logical = authority.refined_manifest["payload"]["logical_content"]
    document = logical["document"]
    arrays = document["arrays"]
    paths = (
        "instance_key",
        "source_crop_row_ids",
        "source_acquisition_frame_index",
        "frame_row_offsets",
        "source_crop_xywh",
    )
    if not isinstance(arrays, Mapping) or any(path not in arrays for path in paths):
        raise SubjectShapeBundleSourceError(
            "Refined mask manifest lacks exact subject-shape row declarations."
        )
    return {path: dict(arrays[path]) for path in paths}


def _camera_frame_authorities(
    authority: BoundRecordingSubjectMaskCoordinateAuthority,
) -> tuple[
    BoundAcquisitionCameraFrame,
    BoundPixelFrameAuthority,
    BoundPixelFrameAuthority,
]:
    _ownership, acquisition = load_persisted_acquisition_camera_authority(
        authority._root,
        expected_camera_id=authority.camera_identity,
    )
    camera_root = authority._root[
        f"analysis/coordinate_frames/source_camera/{authority.camera_identity}"
    ]
    continuous = load_source_camera_pixel_frame_authority(
        camera_root["continuous"],
        acquisition_frame=acquisition,
    )
    edge = load_source_camera_pixel_frame_authority(
        camera_root["pixel_edge_half_open"],
        acquisition_frame=acquisition,
    )
    expected_extent = (authority.source_width, authority.source_height)
    if (
        (acquisition.width, acquisition.height) != expected_extent
        or (continuous.endpoint.width, continuous.endpoint.height) != expected_extent
        or (edge.endpoint.width, edge.endpoint.height) != expected_extent
        or continuous.pixel_convention != "continuous"
        or edge.pixel_convention != "pixel_edge_half_open"
    ):
        raise SubjectShapeBundleSourceError(
            "Bundle camera-frame authorities differ from the exact source extent "
            "or required pixel conventions."
        )
    return acquisition, continuous, edge


def _source_record(
    authority: BoundRecordingSubjectMaskCoordinateAuthority,
    *,
    acquisition_frame: BoundAcquisitionCameraFrame,
    continuous_frame: BoundPixelFrameAuthority,
    edge_frame: BoundPixelFrameAuthority,
    assignment_keypoint_rebinding: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    components = authority.refined_manifest["payload"]["logical_schema"]["components"]
    labels = tuple(str(value) for value in components.get("labels") or ())
    if len(labels) != len(SUBJECT_SHAPE_REQUIRED_MASK_COMPONENTS) or set(labels) != set(
        SUBJECT_SHAPE_REQUIRED_MASK_COMPONENTS
    ):
        raise SubjectShapeBundleSourceError(
            "Subject-shape bundle source requires each maintained component exactly once."
        )
    historical_assignment = dict(authority.assignment_keypoint_collection)
    source_record: dict[str, object] = {
        "schema_id": SUBJECT_SHAPE_BUNDLE_SOURCE_SCHEMA_ID,
        "schema_version": SUBJECT_SHAPE_BUNDLE_SOURCE_SCHEMA_VERSION,
        "source_kind": SUBJECT_SHAPE_BUNDLE_SOURCE_KIND,
        "recording_identity": authority.recording_identity,
        "camera_identity": authority.camera_identity,
        "frame_axis": {
            "domain": "zero_based_acquisition_camera_frame",
            "source_total_frames": authority.source_total_frames,
            "frame_row_offsets_length": authority.source_total_frames + 1,
        },
        "source_camera_extent": {
            "width_px": authority.source_width,
            "height_px": authority.source_height,
        },
        "source_camera_authorities": {
            "acquisition_frame": {
                "record_ref": acquisition_frame.record_ref,
                "record_sha256": acquisition_frame.record_sha256,
            },
            "continuous_pixel_frame": {
                "record_ref": continuous_frame.record_ref,
                "record_sha256": continuous_frame.record_sha256,
            },
            "pixel_edge_half_open_frame": {
                "record_ref": edge_frame.record_ref,
                "record_sha256": edge_frame.record_sha256,
            },
        },
        "roi_raster_extent": {
            "width_px": authority.roi_width,
            "height_px": authority.roi_height,
        },
        "row_count": authority.n_rois,
        "component_labels": list(labels),
        "row_arrays": _array_declarations(authority),
        "coordinate_transform": {
            "profile": "rowwise_roi_to_source_camera_translation_v1",
            "source": "refined_subject_mask_core.source_crop_xywh",
            "size_policy": "source_crop_wh_must_equal_dense_roi_extent",
            "continuous_points": "add_source_crop_xy",
            "pixel_edge_half_open_boxes": "add_source_crop_xy_to_both_corners",
        },
        "authorities": {
            "bundle_id": authority.bundle_id,
            "bundle_manifest_payload_digest": authority.bundle_manifest[
                "payload_digest"
            ],
            "bundle_coordinate_authority_digest": authority.authority_digest,
            "crop_run_path": authority.crop_run_path,
            "crop_manifest_payload_digest": authority.crop_manifest["payload_digest"],
            "raw_run_path": authority.raw_run_path,
            "raw_manifest_payload_digest": authority.raw_manifest["payload_digest"],
            "refined_run_path": authority.refined_run_path,
            "refined_manifest_payload_digest": authority.refined_manifest[
                "payload_digest"
            ],
        },
        "assignment_keypoints": historical_assignment,
        "assignment_keypoints_digest": canonical_json_sha256(historical_assignment),
    }
    if assignment_keypoint_rebinding is not None:
        payload = assignment_keypoint_rebinding.get("payload")
        subject = (
            payload.get("subject_mask_source") if isinstance(payload, Mapping) else None
        )
        if (
            not isinstance(payload, Mapping)
            or not isinstance(subject, Mapping)
            or payload.get("assignment_state") != "used"
            or payload.get("recording_identity") != authority.recording_identity
            or payload.get("camera_identity") != authority.camera_identity
            or payload.get("row_count") != authority.n_rois
            or subject.get("bundle_id") != authority.bundle_id
            or subject.get("bundle_manifest_payload_digest")
            != authority.bundle_manifest.get("payload_digest")
            or subject.get("bundle_coordinate_authority_digest")
            != authority.authority_digest
            or subject.get("assignment_collection_digest")
            != canonical_json_sha256(historical_assignment)
        ):
            raise SubjectShapeBundleSourceError(
                "Assignment-keypoint rebinding does not bind this exact mask bundle."
            )
        normalized = {
            "status": "used",
            "authority_profile": ASSIGNMENT_CANONICAL_KEYPOINT_PROFILE,
            "rebinding_run_path": (
                "subject_mask_assignment_keypoint_rebinding_runs/"
                f"{payload['rebinding_run_id']}"
            ),
            "rebinding_manifest_payload_digest": assignment_keypoint_rebinding[
                "payload_digest"
            ],
            "canonical_keypoint_source": dict(payload["canonical_keypoint_source"]),
            "equivalence": dict(payload["equivalence"]),
            "selection_policy": payload["selection_policy"],
        }
        source_record.update(
            {
                "schema_version": (SUBJECT_SHAPE_BUNDLE_REBOUND_SOURCE_SCHEMA_VERSION),
                "historical_assignment_keypoints": historical_assignment,
                "historical_assignment_keypoints_digest": canonical_json_sha256(
                    historical_assignment
                ),
                "assignment_keypoints": normalized,
                "assignment_keypoints_digest": canonical_json_sha256(normalized),
            }
        )
    return source_record


@dataclass(frozen=True, init=False)
class BoundSubjectShapeBundleSource:
    archive_path: Path
    bundle_id: str
    source_record: Mapping[str, Any] = field(repr=False)
    source_digest: str
    active: bool
    assignment_keypoint_rebinding_run_id: str | None
    assignment_keypoint_rebinding_manifest: Mapping[str, Any] | None = field(
        repr=False,
        compare=False,
    )
    authority: BoundRecordingSubjectMaskCoordinateAuthority = field(
        repr=False,
        compare=False,
    )
    acquisition_frame: BoundAcquisitionCameraFrame = field(
        repr=False,
        compare=False,
    )
    continuous_source_camera_frame: BoundPixelFrameAuthority = field(
        repr=False,
        compare=False,
    )
    edge_source_camera_frame: BoundPixelFrameAuthority = field(
        repr=False,
        compare=False,
    )
    _seal: object = field(repr=False, compare=False)

    def __init__(
        self,
        *,
        _verification_seal: object | None = None,
        **values: Any,
    ) -> None:
        if _verification_seal is not _BOUND_SOURCE_SEAL:
            raise SubjectShapeBundleSourceError(
                "Bound subject-shape bundle sources cannot be constructed directly."
            )
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_seal", _verification_seal)

    @property
    def row_count(self) -> int:
        return self.authority.n_rois

    @property
    def instance_key_node(self) -> Any:
        return self.authority.instance_key_node

    @property
    def source_crop_row_ids_node(self) -> Any:
        return self.authority.source_crop_row_ids_node

    @property
    def source_acquisition_frame_index_node(self) -> Any:
        return self.authority.source_acquisition_frame_index_node

    def translation_offsets(self) -> np.ndarray:
        return self.authority.require_translation_only_offsets()

    def transform_roi_points(self, values: Any) -> np.ndarray:
        points = np.asarray(values)
        if (
            points.dtype.kind != "f"
            or points.ndim < 2
            or points.shape[0] != self.row_count
            or points.shape[-1] != 2
        ):
            raise SubjectShapeBundleSourceError(
                "ROI points must be floating [row,...,xy] values."
            )
        offsets = self.translation_offsets()
        shape = (self.row_count,) + (1,) * (points.ndim - 2) + (2,)
        return points.astype(np.float64) + offsets.reshape(shape)

    def transform_roi_boxes(self, values: Any) -> np.ndarray:
        boxes = np.asarray(values)
        if boxes.dtype.kind != "f" or boxes.shape != (self.row_count, 4):
            raise SubjectShapeBundleSourceError(
                "ROI boxes must be floating [row,xyxy] values."
            )
        offsets = self.translation_offsets()
        return (
            boxes.astype(np.float64).reshape(self.row_count, 2, 2) + offsets[:, None, :]
        ).reshape(self.row_count, 4)

    def assert_verified(self) -> None:
        def verify() -> None:
            current = _load_receipt_bound_subject_shape_bundle_source(
                self.archive_path,
                bundle_id=self.bundle_id,
                allow_inactive=True,
                assignment_keypoint_rebinding_run_id=(
                    self.assignment_keypoint_rebinding_run_id
                ),
            )
            if current.source_digest != self.source_digest:
                raise SubjectShapeBundleSourceError(
                    "Subject-shape bundle source changed after binding."
                )

        verify_persisted_proof(
            (
                "palette.bound_subject_shape_bundle_source.v1",
                str(self.archive_path.expanduser().resolve()),
                self.bundle_id,
                self.source_digest,
                self.assignment_keypoint_rebinding_run_id,
            ),
            verify,
        )


def _load_receipt_bound_subject_shape_bundle_source(
    analysis_zarr: Path,
    *,
    bundle_id: str | None = None,
    allow_inactive: bool = False,
    assignment_keypoint_rebinding_run_id: str | None = None,
) -> BoundSubjectShapeBundleSource:
    authority = load_recording_subject_mask_coordinate_authority(
        analysis_zarr,
        bundle_id=bundle_id,
        allow_inactive=allow_inactive,
    )
    acquisition, continuous, edge = _camera_frame_authorities(authority)
    rebinding_manifest = (
        load_assignment_keypoint_rebinding_manifest(
            analysis_zarr,
            rebinding_run_id=assignment_keypoint_rebinding_run_id,
            subject_mask_authority=authority,
        )
        if assignment_keypoint_rebinding_run_id is not None
        else None
    )
    record = _source_record(
        authority,
        acquisition_frame=acquisition,
        continuous_frame=continuous,
        edge_frame=edge,
        assignment_keypoint_rebinding=rebinding_manifest,
    )
    return BoundSubjectShapeBundleSource(
        archive_path=authority.archive_path,
        bundle_id=authority.bundle_id,
        source_record=record,
        source_digest=canonical_json_sha256(record),
        active=authority.active,
        assignment_keypoint_rebinding_run_id=(
            str(assignment_keypoint_rebinding_run_id)
            if assignment_keypoint_rebinding_run_id is not None
            else None
        ),
        assignment_keypoint_rebinding_manifest=rebinding_manifest,
        authority=authority,
        acquisition_frame=acquisition,
        continuous_source_camera_frame=continuous,
        edge_source_camera_frame=edge,
        _verification_seal=_BOUND_SOURCE_SEAL,
    )


def load_subject_shape_bundle_source(
    analysis_zarr: Path,
    *,
    bundle_id: str | None = None,
    allow_inactive: bool = False,
    assignment_keypoint_rebinding_run_id: str | None = None,
) -> BoundSubjectShapeBundleSource:
    """Load one exact source and validate its translation geometry once.

    Later process-local proof rechecks reconstruct only the receipt-bound
    metadata identity. They do not reread ``source_crop_xywh`` merely to prove
    that the already-bound source digest remained stable.
    """

    source = _load_receipt_bound_subject_shape_bundle_source(
        analysis_zarr,
        bundle_id=bundle_id,
        allow_inactive=allow_inactive,
        assignment_keypoint_rebinding_run_id=assignment_keypoint_rebinding_run_id,
    )
    source.authority.require_translation_only_offsets()
    return source


def require_bound_subject_shape_bundle_source(
    value: Any,
) -> BoundSubjectShapeBundleSource:
    if type(value) is not BoundSubjectShapeBundleSource or value._seal is not (
        _BOUND_SOURCE_SEAL
    ):
        raise SubjectShapeBundleSourceError(
            "A sealed exact subject-shape bundle source is required."
        )
    value.assert_verified()
    return value


__all__ = [
    "BoundSubjectShapeBundleSource",
    "SUBJECT_SHAPE_BUNDLE_SOURCE_SCHEMA_ID",
    "SUBJECT_SHAPE_BUNDLE_SOURCE_SCHEMA_VERSION",
    "SUBJECT_SHAPE_BUNDLE_REBOUND_SOURCE_SCHEMA_VERSION",
    "SUBJECT_SHAPE_BUNDLE_SOURCE_KIND",
    "SUBJECT_SHAPE_REQUIRED_MASK_COMPONENTS",
    "SubjectShapeBundleSourceError",
    "assignment_rebinding_run_id_from_source_record",
    "load_subject_shape_bundle_source",
    "require_bound_subject_shape_bundle_source",
]
