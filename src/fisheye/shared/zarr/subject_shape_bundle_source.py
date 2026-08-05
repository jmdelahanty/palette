"""Sealed subject-shape input adapter for recording subject-mask bundles."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.subject_mask_bundle_coordinate_authority import (
    BoundRecordingSubjectMaskCoordinateAuthority,
    load_recording_subject_mask_coordinate_authority,
)

SUBJECT_SHAPE_BUNDLE_SOURCE_SCHEMA_ID = (
    "palette.subject_shape.recording_mask_bundle_source"
)
SUBJECT_SHAPE_BUNDLE_SOURCE_SCHEMA_VERSION = 1
SUBJECT_SHAPE_REQUIRED_MASK_COMPONENTS = (
    "subject_body",
    "swim_bladder",
    "eye_left",
    "eye_right",
)


class SubjectShapeBundleSourceError(ValueError):
    """Raised when a bundle cannot serve as an exact subject-shape input."""


_BOUND_SOURCE_SEAL = object()


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


def _source_record(
    authority: BoundRecordingSubjectMaskCoordinateAuthority,
) -> dict[str, object]:
    components = authority.refined_manifest["payload"]["logical_schema"][
        "components"
    ]
    labels = tuple(str(value) for value in components.get("labels") or ())
    if len(labels) != len(SUBJECT_SHAPE_REQUIRED_MASK_COMPONENTS) or set(
        labels
    ) != set(SUBJECT_SHAPE_REQUIRED_MASK_COMPONENTS):
        raise SubjectShapeBundleSourceError(
            "Subject-shape bundle source requires each maintained component exactly once."
        )
    assignment = dict(authority.assignment_keypoint_collection)
    return {
        "schema_id": SUBJECT_SHAPE_BUNDLE_SOURCE_SCHEMA_ID,
        "schema_version": SUBJECT_SHAPE_BUNDLE_SOURCE_SCHEMA_VERSION,
        "source_kind": "recording_subject_mask_bundle_v3",
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
            "crop_manifest_payload_digest": authority.crop_manifest[
                "payload_digest"
            ],
            "raw_run_path": authority.raw_run_path,
            "raw_manifest_payload_digest": authority.raw_manifest[
                "payload_digest"
            ],
            "refined_run_path": authority.refined_run_path,
            "refined_manifest_payload_digest": authority.refined_manifest[
                "payload_digest"
            ],
        },
        "assignment_keypoints": assignment,
        "assignment_keypoints_digest": canonical_json_sha256(assignment),
    }


@dataclass(frozen=True, init=False)
class BoundSubjectShapeBundleSource:
    archive_path: Path
    bundle_id: str
    source_record: Mapping[str, Any] = field(repr=False)
    source_digest: str
    active: bool
    authority: BoundRecordingSubjectMaskCoordinateAuthority = field(
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
            boxes.astype(np.float64).reshape(self.row_count, 2, 2)
            + offsets[:, None, :]
        ).reshape(self.row_count, 4)

    def assert_verified(self) -> None:
        current = load_subject_shape_bundle_source(
            self.archive_path,
            bundle_id=self.bundle_id,
            allow_inactive=True,
        )
        if current.source_digest != self.source_digest:
            raise SubjectShapeBundleSourceError(
                "Subject-shape bundle source changed after binding."
            )


def load_subject_shape_bundle_source(
    analysis_zarr: Path,
    *,
    bundle_id: str | None = None,
    allow_inactive: bool = False,
) -> BoundSubjectShapeBundleSource:
    authority = load_recording_subject_mask_coordinate_authority(
        analysis_zarr,
        bundle_id=bundle_id,
        allow_inactive=allow_inactive,
    )
    record = _source_record(authority)
    authority.require_translation_only_offsets()
    return BoundSubjectShapeBundleSource(
        archive_path=authority.archive_path,
        bundle_id=authority.bundle_id,
        source_record=record,
        source_digest=canonical_json_sha256(record),
        active=authority.active,
        authority=authority,
        _verification_seal=_BOUND_SOURCE_SEAL,
    )


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
    "SUBJECT_SHAPE_REQUIRED_MASK_COMPONENTS",
    "SubjectShapeBundleSourceError",
    "load_subject_shape_bundle_source",
    "require_bound_subject_shape_bundle_source",
]
