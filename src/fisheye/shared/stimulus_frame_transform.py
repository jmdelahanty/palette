"""Sealed frame, transform, and temporal evidence for future stimulus rows.

This module is deliberately canonical-only.  It materializes and reloads the
typed path from arena-relative stimulus canvas pixels to the selected source
camera and binds that path to an explicit acquisition-frame mapping.  Historic
camera-frame identifiers and resolution ratios are not accepted as temporal or
spatial authority.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from fisheye.shared.archive_identity import ArchiveIdentity, require_same_archive
from fisheye.shared.coordinate_identity import (
    SOURCE_ROW_ACQUISITION_FRAME_INDEX_REF,
    BoundRowIdentityContract,
    BoundSourceRowTemporalAuthority,
    load_bound_source_row_temporal_authority,
    require_bound_source_row_temporal_authority,
)
from fisheye.shared.coordinate_record import (
    BoundCoordinateRecord,
    bind_persisted_coordinate_record,
    stamp_and_bind_persisted_coordinate_record,
)
from fisheye.shared.coordinate_reference import (
    BoundReferenceExtent,
    bind_persisted_record_reference_extent,
    verify_bound_reference_extent,
)
from fisheye.shared.directed_transform_chain import (
    BoundDirectedTransformChain,
    resolve_bound_directed_transform_chain,
)
from fisheye.shared.directed_transform_v2 import (
    BoundDirectedTransformV2,
    load_bound_directed_transform_v2,
    stamp_directed_transform_v2,
    stamp_explicit_inverse_directed_transform_v2,
)
from fisheye.shared.pixel_frame_authority import (
    BoundAcquisitionCameraFrame,
    BoundPixelFrameAuthority,
    load_arena_relative_canvas_pixel_frame_authority,
    load_persisted_acquisition_camera_authority,
    load_selected_canvas_pixel_frame_authority,
    load_source_camera_pixel_frame_authority,
    stamp_arena_relative_canvas_pixel_frame_authority,
    stamp_selected_canvas_pixel_frame_authority,
    stamp_source_camera_pixel_frame_authority,
)
from fisheye.shared.selected_calibration import (
    CANONICAL_HOMOGRAPHY_FROM_SPACE_ID,
    CANONICAL_HOMOGRAPHY_TO_SPACE_ID,
    SelectedCalibrationSnapshot,
    TransformReferenceExtent,
    load_selected_calibration_manifest_attrs,
    load_selected_calibration_snapshot,
    require_bound_selected_calibration_snapshot,
)
from fisheye.shared.transform_authority import (
    arena_to_selected_canvas_matrix,
    load_bound_transform_authority,
    stamp_arena_to_selected_canvas_transform_authority,
    stamp_selected_calibration_transform_authority,
)


STIMULUS_FRAME_TRANSFORM_MANIFEST_ATTR = "stimulus_frame_transform_manifest"
STIMULUS_FRAME_TRANSFORM_MANIFEST_DIGEST_ATTR = (
    f"{STIMULUS_FRAME_TRANSFORM_MANIFEST_ATTR}_sha256"
)
STIMULUS_FRAME_TRANSFORM_MANIFEST_SCHEMA_ID = (
    "palette.stimulus_frame_transform_manifest"
)
STIMULUS_FRAME_TRANSFORM_MANIFEST_SCHEMA_VERSION = 1
STIMULUS_CANVAS_EXTENT_ATTR = "selected_canvas_extent_record"
STIMULUS_CANVAS_EXTENT_DIGEST_ATTR = f"{STIMULUS_CANVAS_EXTENT_ATTR}_sha256"
STIMULUS_ARENA_EXTENT_ATTR = "arena_relative_extent_record"
STIMULUS_ARENA_EXTENT_DIGEST_ATTR = f"{STIMULUS_ARENA_EXTENT_ATTR}_sha256"

_PIXEL_CONVENTION = "continuous"
_SELECTED_CANVAS_NODE = "coordinate_frames/selected_canvas"
_ARENA_FRAME_NODE = "coordinate_frames/arena_relative_canvas"
_ARENA_GEOMETRY_ARRAY = "coordinate_frames/arena_geometry_xywh"
_ARENA_TO_CANVAS_ARRAY = "transforms/arena_to_selected_canvas"
_ARENA_TO_CANVAS_AUTHORITY = "transforms/arena_to_selected_canvas_authority"


class StimulusFrameTransformError(ValueError):
    """Raised when future stimulus frame/transform evidence is incomplete."""


@dataclass(frozen=True)
class BoundStimulusFrameTransformEvidence:
    """Live, sealed evidence needed by canonical stimulus coordinate surfaces."""

    archive_identity: ArchiveIdentity
    selected_calibration: SelectedCalibrationSnapshot = field(repr=False, compare=False)
    acquisition_frame: BoundAcquisitionCameraFrame = field(repr=False, compare=False)
    source_temporal_authority: BoundSourceRowTemporalAuthority = field(
        repr=False,
        compare=False,
    )
    source_camera_frame: BoundPixelFrameAuthority = field(repr=False, compare=False)
    selected_canvas_frame: BoundPixelFrameAuthority = field(repr=False, compare=False)
    arena_relative_frame: BoundPixelFrameAuthority = field(repr=False, compare=False)
    arena_to_canvas: BoundDirectedTransformV2 = field(repr=False, compare=False)
    canvas_to_source_camera: BoundDirectedTransformV2 = field(
        repr=False,
        compare=False,
    )
    transform_chain: BoundDirectedTransformChain = field(repr=False, compare=False)
    manifest: BoundCoordinateRecord


def _node_path(node: Any) -> str:
    path = getattr(node, "path", None)
    if not isinstance(path, str) or not path or path.startswith("/"):
        raise StimulusFrameTransformError("Persisted node path is not canonical.")
    return path


def _child(root: Any, path: str) -> Any:
    try:
        return root[path]
    except Exception as exc:
        raise StimulusFrameTransformError(
            f"Required canonical stimulus node /{path} is missing."
        ) from exc


def _create_group(parent: Any, name: str) -> Any:
    if name in parent:
        raise StimulusFrameTransformError(
            f"Future canonical publication refuses occupied child {name!r}."
        )
    try:
        return parent.create_group(name)
    except Exception as exc:
        raise StimulusFrameTransformError(
            f"Unable to create canonical group {name!r}: {exc}."
        ) from exc


def _require_group(parent: Any, path: str) -> Any:
    try:
        return parent.require_group(path)
    except Exception as exc:
        raise StimulusFrameTransformError(
            f"Unable to resolve canonical group {path!r}: {exc}."
        ) from exc


def _create_array(parent: Any, name: str, values: np.ndarray) -> Any:
    if name in parent:
        raise StimulusFrameTransformError(
            f"Future canonical publication refuses occupied array {name!r}."
        )
    array = np.asarray(values)
    chunks = tuple(max(1, int(value)) for value in array.shape)
    try:
        return parent.create_array(name, data=array, chunks=chunks)
    except Exception as exc:
        raise StimulusFrameTransformError(
            f"Unable to create canonical array {name!r}: {exc}."
        ) from exc


def _pointer(record_ref: str, record_sha256: str) -> dict[str, str]:
    return {"record_ref": record_ref, "record_sha256": record_sha256}


def _selected_snapshot_from_persisted_run(
    root_node: Any,
    run_group: Any,
    *,
    stimulus_run: str,
) -> SelectedCalibrationSnapshot:
    calibration = _child(run_group, "calibration")
    attrs = getattr(calibration, "attrs", None)
    if not isinstance(attrs, Mapping):
        raise StimulusFrameTransformError(
            "Selected calibration group does not expose persisted attrs."
        )
    try:
        manifest = load_selected_calibration_manifest_attrs(attrs)
    except Exception as exc:
        raise StimulusFrameTransformError(
            f"Selected calibration manifest is invalid: {exc}."
        ) from exc
    camera = manifest.camera_calibration
    display = manifest.display_snapshot
    source_extent = TransformReferenceExtent(
        width=camera.native_width_px,
        height=camera.native_height_px,
        units="px",
        authority=(
            f"{manifest.camera_calibration_ref}@native_width_px,native_height_px"
        ),
    )
    target_extent = TransformReferenceExtent(
        width=display.width_px,
        height=display.height_px,
        units="px",
        authority=(
            f"analysis/stimulus_runs/{stimulus_run}/display_snapshot"
            "@selected_output_geometry"
        ),
    )
    try:
        return load_selected_calibration_snapshot(
            root_node,
            stimulus_run=stimulus_run,
            expected_camera_id=manifest.camera_id,
            expected_from_space_id=CANONICAL_HOMOGRAPHY_FROM_SPACE_ID,
            expected_to_space_id=CANONICAL_HOMOGRAPHY_TO_SPACE_ID,
            expected_source_reference_extent=source_extent,
            expected_target_reference_extent=target_extent,
        )
    except Exception as exc:
        raise StimulusFrameTransformError(
            f"Selected calibration snapshot cannot be rebound: {exc}."
        ) from exc


def _canvas_extent_record(selected: SelectedCalibrationSnapshot) -> dict[str, Any]:
    return {
        "schema_id": "palette.selected_stimulus_canvas_extent",
        "schema_version": 1,
        "width_px": selected.target_reference_extent.width,
        "height_px": selected.target_reference_extent.height,
        "units": "px",
        "selected_calibration": _pointer(
            selected.manifest_record_ref,
            selected.manifest_sha256,
        ),
    }


def _arena_extent_record(
    arena_reference: BoundReferenceExtent,
) -> dict[str, Any]:
    reference = verify_bound_reference_extent(arena_reference)
    return {
        "schema_id": "palette.arena_relative_canvas_extent",
        "schema_version": 1,
        "width_px": reference.width,
        "height_px": reference.height,
        "units": "px",
        "selected_arena_geometry": _pointer(
            reference.record_ref,
            reference.record_sha256,
        ),
    }


def _bind_canvas_extent(node: Any) -> BoundReferenceExtent:
    return bind_persisted_record_reference_extent(
        node,
        record_attr=STIMULUS_CANVAS_EXTENT_ATTR,
        digest_attr=STIMULUS_CANVAS_EXTENT_DIGEST_ATTR,
        width_field="width_px",
        height_field="height_px",
        units_field="units",
    )


def _bind_arena_extent(node: Any) -> BoundReferenceExtent:
    return bind_persisted_record_reference_extent(
        node,
        record_attr=STIMULUS_ARENA_EXTENT_ATTR,
        digest_attr=STIMULUS_ARENA_EXTENT_DIGEST_ATTR,
        width_field="width_px",
        height_field="height_px",
        units_field="units",
    )


def _geometry_xywh(
    arena_record: BoundCoordinateRecord,
    arena_reference: BoundReferenceExtent,
) -> np.ndarray:
    reference = verify_bound_reference_extent(arena_reference)
    record = arena_record.record
    values = (
        record.get("arena_origin_in_canvas_x_px"),
        record.get("arena_origin_in_canvas_y_px"),
        reference.width,
        reference.height,
    )
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise StimulusFrameTransformError(
            "Canonical arena placement requires exact integer x,y,width,height pixels."
        )
    return np.asarray(values, dtype="<i8")


def _manifest_record(
    *,
    stimulus_run: str,
    selected: SelectedCalibrationSnapshot,
    acquisition: BoundAcquisitionCameraFrame,
    temporal: BoundSourceRowTemporalAuthority,
    source_camera: BoundPixelFrameAuthority,
    canvas: BoundPixelFrameAuthority,
    arena: BoundPixelFrameAuthority,
    arena_to_canvas: BoundDirectedTransformV2,
    canvas_to_camera: BoundDirectedTransformV2,
) -> dict[str, Any]:
    return {
        "schema_id": STIMULUS_FRAME_TRANSFORM_MANIFEST_SCHEMA_ID,
        "schema_version": STIMULUS_FRAME_TRANSFORM_MANIFEST_SCHEMA_VERSION,
        "stimulus_run": stimulus_run,
        "pixel_convention": _PIXEL_CONVENTION,
        "selected_calibration": _pointer(
            selected.manifest_record_ref,
            selected.manifest_sha256,
        ),
        "acquisition_camera_frame": _pointer(
            acquisition.record_ref,
            acquisition.record_sha256,
        ),
        "source_row_temporal_authority": _pointer(
            temporal.record_ref,
            temporal.record_sha256,
        ),
        "frames": {
            "arena_relative_canvas": _pointer(
                arena.record_ref,
                arena.record_sha256,
            ),
            "selected_canvas": _pointer(canvas.record_ref, canvas.record_sha256),
            "source_camera": _pointer(
                source_camera.record_ref,
                source_camera.record_sha256,
            ),
        },
        "descriptor_to_source_camera_chain": [
            _pointer(arena_to_canvas.record_ref, arena_to_canvas.transform_sha256),
            _pointer(canvas_to_camera.record_ref, canvas_to_camera.transform_sha256),
        ],
    }


def publish_stimulus_frame_transform_evidence(
    root_node: Any,
    run_group: Any,
    manifest_owner: Any,
    *,
    stimulus_run: str,
    selected_calibration: SelectedCalibrationSnapshot,
    arena_reference: BoundReferenceExtent,
    arena_record: BoundCoordinateRecord,
    source_temporal_authority: BoundSourceRowTemporalAuthority,
) -> BoundStimulusFrameTransformEvidence:
    """Publish the one future-canonical arena-to-source-camera path."""

    selected = require_bound_selected_calibration_snapshot(selected_calibration)
    temporal = require_bound_source_row_temporal_authority(
        source_temporal_authority
    )
    reference = verify_bound_reference_extent(arena_reference)
    if selected.stimulus_run != stimulus_run:
        raise StimulusFrameTransformError(
            "Selected calibration belongs to a different stimulus run."
        )
    try:
        _, acquisition = load_persisted_acquisition_camera_authority(
            root_node,
            expected_camera_id=selected.camera_id,
        )
    except Exception as exc:
        raise StimulusFrameTransformError(
            f"Canonical acquisition camera authority is unavailable: {exc}."
        ) from exc
    if (
        temporal.record.acquisition_camera_frame.record_ref != acquisition.record_ref
        or temporal.record.acquisition_camera_frame.record_sha256
        != acquisition.record_sha256
    ):
        raise StimulusFrameTransformError(
            "Stimulus temporal mapping and selected acquisition camera differ."
        )

    camera_node = _require_group(
        root_node,
        "analysis/coordinate_frames/source_camera/"
        f"{selected.camera_id}/{_PIXEL_CONVENTION}",
    )
    source_camera = stamp_source_camera_pixel_frame_authority(
        camera_node,
        frame_id=f"{selected.camera_id}_source_camera",
        pixel_convention=_PIXEL_CONVENTION,
        acquisition_frame=acquisition,
    )

    frame_group = _create_group(run_group, "coordinate_frames")
    canvas_node = _create_group(frame_group, "selected_canvas")
    canvas_extent_record = _canvas_extent_record(selected)
    canvas_node.attrs.update(
        {
            "width_px": canvas_extent_record["width_px"],
            "height_px": canvas_extent_record["height_px"],
            "units": canvas_extent_record["units"],
        }
    )
    stamp_and_bind_persisted_coordinate_record(
        canvas_node,
        canvas_extent_record,
        attr_name=STIMULUS_CANVAS_EXTENT_ATTR,
        digest_attr_name=STIMULUS_CANVAS_EXTENT_DIGEST_ATTR,
    )
    canvas_extent = _bind_canvas_extent(canvas_node)
    canvas = stamp_selected_canvas_pixel_frame_authority(
        canvas_extent,
        frame_id=f"{stimulus_run}_selected_canvas",
        pixel_convention=_PIXEL_CONVENTION,
        selected_calibration_snapshot=selected,
    )

    arena_node = _create_group(frame_group, "arena_relative_canvas")
    arena_extent_record = _arena_extent_record(reference)
    arena_node.attrs.update(
        {
            "width_px": arena_extent_record["width_px"],
            "height_px": arena_extent_record["height_px"],
            "units": arena_extent_record["units"],
        }
    )
    stamp_and_bind_persisted_coordinate_record(
        arena_node,
        arena_extent_record,
        attr_name=STIMULUS_ARENA_EXTENT_ATTR,
        digest_attr_name=STIMULUS_ARENA_EXTENT_DIGEST_ATTR,
    )
    arena_extent = _bind_arena_extent(arena_node)
    geometry_node = _create_array(
        frame_group,
        "arena_geometry_xywh",
        _geometry_xywh(arena_record, reference),
    )
    arena = stamp_arena_relative_canvas_pixel_frame_authority(
        arena_extent,
        frame_id=f"{stimulus_run}_arena_relative_canvas",
        pixel_convention=_PIXEL_CONVENTION,
        geometry_node=geometry_node,
        selected_canvas_frame=canvas,
    )

    calibration = _child(run_group, "calibration")
    camera_calibration = _child(calibration, selected.camera_id)
    matrix_node = _child(camera_calibration, "homography_matrix")
    selected_authority_node = _create_group(
        camera_calibration,
        "camera_to_selected_canvas_authority",
    )
    selected_authority = stamp_selected_calibration_transform_authority(
        selected_authority_node,
        authority_id=f"{stimulus_run}_{selected.camera_id}_selected_calibration",
        source_matrix_node=matrix_node,
        source_frame=source_camera,
        target_frame=canvas,
        selected_calibration_snapshot=selected,
    )
    camera_to_canvas = stamp_directed_transform_v2(
        matrix_node,
        transform_id=f"{stimulus_run}_source_camera_to_selected_canvas",
        authority=selected_authority,
        source_frame=source_camera,
        target_frame=canvas,
    )
    inverse_node = _create_array(
        camera_calibration,
        "selected_canvas_to_source_camera",
        np.asarray(np.linalg.inv(selected.homography.matrix), dtype="<f8"),
    )
    canvas_to_camera = stamp_explicit_inverse_directed_transform_v2(
        inverse_node,
        transform_id=f"{stimulus_run}_selected_canvas_to_source_camera",
        forward=camera_to_canvas,
    )

    transforms = _create_group(run_group, "transforms")
    arena_matrix_node = _create_array(
        transforms,
        "arena_to_selected_canvas",
        np.asarray(arena_to_selected_canvas_matrix(arena, canvas), dtype="<f8"),
    )
    arena_authority_node = _create_group(
        transforms,
        "arena_to_selected_canvas_authority",
    )
    arena_authority = stamp_arena_to_selected_canvas_transform_authority(
        arena_authority_node,
        authority_id=f"{stimulus_run}_arena_to_selected_canvas",
        matrix_node=arena_matrix_node,
        source_frame=arena,
        target_frame=canvas,
    )
    arena_to_canvas = stamp_directed_transform_v2(
        arena_matrix_node,
        transform_id=f"{stimulus_run}_arena_to_selected_canvas",
        authority=arena_authority,
        source_frame=arena,
        target_frame=canvas,
    )
    chain = resolve_bound_directed_transform_chain(
        (arena_to_canvas, canvas_to_camera)
    )
    manifest_record = _manifest_record(
        stimulus_run=stimulus_run,
        selected=selected,
        acquisition=acquisition,
        temporal=temporal,
        source_camera=source_camera,
        canvas=canvas,
        arena=arena,
        arena_to_canvas=arena_to_canvas,
        canvas_to_camera=canvas_to_camera,
    )
    manifest = stamp_and_bind_persisted_coordinate_record(
        manifest_owner,
        manifest_record,
        attr_name=STIMULUS_FRAME_TRANSFORM_MANIFEST_ATTR,
        digest_attr_name=STIMULUS_FRAME_TRANSFORM_MANIFEST_DIGEST_ATTR,
    )
    archive = require_same_archive(
        root_node,
        run_group,
        manifest_owner,
        camera_node,
        canvas_node,
        arena_node,
        geometry_node,
        matrix_node,
        inverse_node,
        arena_matrix_node,
    )
    return BoundStimulusFrameTransformEvidence(
        archive_identity=archive,
        selected_calibration=selected,
        acquisition_frame=acquisition,
        source_temporal_authority=temporal,
        source_camera_frame=source_camera,
        selected_canvas_frame=canvas,
        arena_relative_frame=arena,
        arena_to_canvas=arena_to_canvas,
        canvas_to_source_camera=canvas_to_camera,
        transform_chain=chain,
        manifest=manifest,
    )


def load_bound_stimulus_frame_transform_evidence(
    root_node: Any,
    run_group: Any,
    manifest_owner: Any,
    *,
    stimulus_run: str,
    row_identity: BoundRowIdentityContract,
) -> BoundStimulusFrameTransformEvidence:
    """Freshly rebind the complete persisted future stimulus evidence."""

    selected = _selected_snapshot_from_persisted_run(
        root_node,
        run_group,
        stimulus_run=stimulus_run,
    )
    try:
        _, acquisition = load_persisted_acquisition_camera_authority(
            root_node,
            expected_camera_id=selected.camera_id,
        )
        source_frame_node = _child(
            root_node,
            "analysis/coordinate_frames/source_camera/"
            f"{selected.camera_id}/{_PIXEL_CONVENTION}",
        )
        source_camera = load_source_camera_pixel_frame_authority(
            source_frame_node,
            acquisition_frame=acquisition,
        )
        source_acquisition_node = _child(
            manifest_owner,
            SOURCE_ROW_ACQUISITION_FRAME_INDEX_REF,
        )
        temporal = load_bound_source_row_temporal_authority(
            manifest_owner,
            source_acquisition_node,
            source_row_identity=row_identity,
            acquisition_frame=acquisition,
        )

        canvas_node = _child(run_group, _SELECTED_CANVAS_NODE)
        canvas_extent = _bind_canvas_extent(canvas_node)
        canvas = load_selected_canvas_pixel_frame_authority(
            canvas_node,
            reference_extent=canvas_extent,
            selected_calibration_snapshot=selected,
        )
        arena_node = _child(run_group, _ARENA_FRAME_NODE)
        arena_extent = _bind_arena_extent(arena_node)
        geometry_node = _child(run_group, _ARENA_GEOMETRY_ARRAY)
        arena = load_arena_relative_canvas_pixel_frame_authority(
            arena_node,
            reference_extent=arena_extent,
            geometry_node=geometry_node,
            selected_canvas_frame=canvas,
        )

        calibration = _child(run_group, "calibration")
        camera_calibration = _child(calibration, selected.camera_id)
        matrix_node = _child(camera_calibration, "homography_matrix")
        selected_authority_node = _child(
            camera_calibration,
            "camera_to_selected_canvas_authority",
        )
        selected_authority = load_bound_transform_authority(
            selected_authority_node,
            payload_node=matrix_node,
            source_frame=source_camera,
            target_frame=canvas,
            selected_calibration_snapshot=selected,
        )
        camera_to_canvas = load_bound_directed_transform_v2(
            matrix_node,
            authority=selected_authority,
            source_frame=source_camera,
            target_frame=canvas,
        )
        inverse_node = _child(
            camera_calibration,
            "selected_canvas_to_source_camera",
        )
        canvas_to_camera = load_bound_directed_transform_v2(
            inverse_node,
            authority=selected_authority,
            source_frame=canvas,
            target_frame=source_camera,
            inverse_of=camera_to_canvas,
        )

        arena_matrix_node = _child(run_group, _ARENA_TO_CANVAS_ARRAY)
        arena_authority_node = _child(run_group, _ARENA_TO_CANVAS_AUTHORITY)
        arena_authority = load_bound_transform_authority(
            arena_authority_node,
            payload_node=arena_matrix_node,
            source_frame=arena,
            target_frame=canvas,
        )
        arena_to_canvas = load_bound_directed_transform_v2(
            arena_matrix_node,
            authority=arena_authority,
            source_frame=arena,
            target_frame=canvas,
        )
        chain = resolve_bound_directed_transform_chain(
            (arena_to_canvas, canvas_to_camera)
        )
        manifest = bind_persisted_coordinate_record(
            manifest_owner,
            attr_name=STIMULUS_FRAME_TRANSFORM_MANIFEST_ATTR,
            digest_attr_name=STIMULUS_FRAME_TRANSFORM_MANIFEST_DIGEST_ATTR,
        )
    except StimulusFrameTransformError:
        raise
    except Exception as exc:
        raise StimulusFrameTransformError(
            f"Canonical stimulus frame/transform evidence is invalid: {exc}."
        ) from exc
    expected_manifest = _manifest_record(
        stimulus_run=stimulus_run,
        selected=selected,
        acquisition=acquisition,
        temporal=temporal,
        source_camera=source_camera,
        canvas=canvas,
        arena=arena,
        arena_to_canvas=arena_to_canvas,
        canvas_to_camera=canvas_to_camera,
    )
    if manifest.record != expected_manifest:
        raise StimulusFrameTransformError(
            "Stimulus frame/transform manifest is stale or incomplete."
        )
    archive = require_same_archive(
        root_node,
        run_group,
        manifest_owner,
        source_frame_node,
        source_acquisition_node,
        canvas_node,
        arena_node,
        geometry_node,
        matrix_node,
        inverse_node,
        arena_matrix_node,
    )
    return BoundStimulusFrameTransformEvidence(
        archive_identity=archive,
        selected_calibration=selected,
        acquisition_frame=acquisition,
        source_temporal_authority=temporal,
        source_camera_frame=source_camera,
        selected_canvas_frame=canvas,
        arena_relative_frame=arena,
        arena_to_canvas=arena_to_canvas,
        canvas_to_source_camera=canvas_to_camera,
        transform_chain=chain,
        manifest=manifest,
    )


__all__ = [
    "BoundStimulusFrameTransformEvidence",
    "STIMULUS_ARENA_EXTENT_ATTR",
    "STIMULUS_ARENA_EXTENT_DIGEST_ATTR",
    "STIMULUS_CANVAS_EXTENT_ATTR",
    "STIMULUS_CANVAS_EXTENT_DIGEST_ATTR",
    "STIMULUS_FRAME_TRANSFORM_MANIFEST_ATTR",
    "STIMULUS_FRAME_TRANSFORM_MANIFEST_DIGEST_ATTR",
    "StimulusFrameTransformError",
    "load_bound_stimulus_frame_transform_evidence",
    "publish_stimulus_frame_transform_evidence",
]
