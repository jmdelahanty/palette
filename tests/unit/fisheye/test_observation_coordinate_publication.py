from __future__ import annotations

import inspect
from dataclasses import replace

import numpy as np
import pytest

from fisheye.shared import observation_coordinate_publication as observation_publication
from fisheye.shared.coordinate_descriptor import COORDINATE_DESCRIPTOR_ATTR
from fisheye.shared.coordinate_reference import bind_array_reference_extent
from fisheye.shared.directed_transform_chain import (
    resolve_bound_directed_transform_chain,
)
from fisheye.shared.directed_transform_v2 import (
    DIRECTED_TRANSFORM_V2_PIXEL_EDGE_ATTR,
    stamp_directed_transform_v2,
)
from fisheye.shared.observation_coordinate_publication import (
    BBOX_CENTER_DERIVATION_ATTR,
    DETECTION_BACKEND_RESULT_PROJECTION_ATTR,
    DETECTION_BBOX_PROJECTION_ATTR,
    ObservationCoordinatePublicationError,
    build_bound_detection_frame_evidence,
    capture_observation_coordinate_publication_checkpoint,
    derive_detection_source_camera_geometry,
    load_crop_observation_geometry,
    load_crop_roi_bbox_edge_reference_extent,
    load_crop_roi_geometry,
    load_detection_observation_geometry,
    load_detection_backend_result_projection,
    publish_crop_observation_geometry,
    publish_crop_roi_bbox_edge_reference_extent,
    publish_crop_roi_geometry,
    publish_detection_observation_geometry,
    publish_detection_backend_result_projection,
    require_bound_source_camera_position_surface,
    restore_observation_coordinate_publication_checkpoint,
)
from fisheye.shared.pixel_frame_authority import (
    CROP_PLACEMENT_PIXEL_EDGE_OWNERSHIP_ATTR,
    normalized_to_pixel_matrix,
    stamp_crop_placement_ownership,
    stamp_normalized_pixel_frame_authority,
    stamp_roi_pixel_frame_authority,
    stamp_source_camera_pixel_frame_authority,
)
from fisheye.shared.transform_authority import (
    TRANSFORM_AUTHORITY_PIXEL_EDGE_ATTR,
    stamp_crop_placement_transform_authority,
    stamp_normalized_to_pixel_transform_authority,
)
from fisheye.tracking.incremental_crop import (
    CropSourceSnapshot,
    IncrementalCropError,
    _canonical_crop_coordinate_arrays,
)
from tests.unit.fisheye.test_directed_transform_chain import (
    FakeArray,
    FakeGroup,
    _world,
)


def _frame_evidence(world):
    token = world["archive_token"]
    bbox_camera = stamp_source_camera_pixel_frame_authority(
        FakeGroup(
            path=(
                "analysis/coordinate_frames/source_camera/"
                "camera-a/pixel_edge_half_open"
            ),
            archive_token=token,
        ),
        frame_id="camera_a_native_pixel_edge_half_open",
        pixel_convention="pixel_edge_half_open",
        acquisition_frame=world["acquisition_frame"],
    )
    normalized = stamp_normalized_pixel_frame_authority(
        FakeGroup(
            path="detect_runs/d1/coordinate_frames/source_camera_normalized",
            archive_token=token,
        ),
        frame_id="d1_source_camera_normalized",
        pixel_frame=bbox_camera,
    )
    matrix = FakeArray(
        normalized_to_pixel_matrix(bbox_camera),
        path="detect_runs/d1/transforms/source_camera_normalized_to_image",
        archive_token=token,
    )
    authority = stamp_normalized_to_pixel_transform_authority(
        FakeGroup(
            path=(
                "detect_runs/d1/transforms/source_camera_normalized_to_image_authority"
            ),
            archive_token=token,
        ),
        authority_id="d1_source_camera_normalized_to_image",
        matrix_node=matrix,
        source_frame=normalized,
        target_frame=bbox_camera,
    )
    link = stamp_directed_transform_v2(
        matrix,
        transform_id="d1_source_camera_normalized_to_image",
        authority=authority,
        source_frame=normalized,
        target_frame=bbox_camera,
    )
    return build_bound_detection_frame_evidence(
        source_camera_frame=world["camera_frame"],
        bbox_source_camera_frame=bbox_camera,
        normalized_frame=normalized,
        normalized_to_source_camera=resolve_bound_directed_transform_chain((link,)),
    )


def _surface(world, *, frames=(0, 1), alter_center: bool = False):
    token = world["archive_token"]
    evidence = _frame_evidence(world)
    rowset = FakeGroup(path="detect_runs/d1", archive_token=token)
    key = FakeArray(
        np.asarray([101, 202], dtype=np.uint64),
        path=f"{rowset.path}/instance_key",
        archive_token=token,
    )
    source_frames = FakeArray(
        np.asarray(frames, dtype=np.int64),
        path=f"{rowset.path}/source_acquisition_frame_index",
        archive_token=token,
    )
    normalized_values = np.asarray(
        [[0.25, 0.50, 0.20, 0.25], [0.75, 0.25, 0.10, 0.20]],
        dtype=np.float64,
    )
    bbox_values, center_values = derive_detection_source_camera_geometry(
        normalized_values,
        frame_evidence=evidence,
    )
    if alter_center:
        center_values[0, 0] += 1.0
    bbox_norm = FakeArray(
        normalized_values,
        path=f"{rowset.path}/bbox_norm_coords",
        archive_token=token,
    )
    bbox_img = FakeArray(
        bbox_values,
        path=f"{rowset.path}/bbox_img_xyxy",
        archive_token=token,
    )
    centers = FakeArray(
        center_values,
        path=f"{rowset.path}/centers_img_xy",
        archive_token=token,
    )
    return (
        evidence,
        rowset,
        key,
        source_frames,
        bbox_norm,
        bbox_img,
        centers,
    )


@pytest.mark.parametrize(
    ("status", "include_contract"),
    [
        ("running", True),
        ("failed", True),
        (None, True),
        (None, False),
    ],
)
def test_persisted_observation_rowset_gate_requires_explicit_complete_lifecycle(
    status: str | None,
    include_contract: bool,
) -> None:
    rowset = FakeGroup(path="detect_runs/d1", archive_token=object())
    rowset.attrs["coordinate_contract"] = "canonical_v2"
    if include_contract:
        rowset.attrs["palette_run_completion_contract"] = (
            "palette.zarr_run_completion.v1"
        )
    if status is not None:
        rowset.attrs["palette_run_completion_status"] = status

    with pytest.raises(
        ObservationCoordinatePublicationError,
        match="completion|complete",
    ):
        observation_publication._require_complete_canonical_observation_rowset(
            rowset,
            run_family="detect_runs",
            label="Detection rowset",
        )


def test_persisted_observation_rowset_gate_accepts_only_normal_complete_run() -> None:
    rowset = FakeGroup(path="crop_runs/c1", archive_token=object())
    rowset.attrs.update(
        {
            "coordinate_contract": "canonical_v2",
            "palette_run_completion_contract": "palette.zarr_run_completion.v1",
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": True,
        }
    )

    observation_publication._require_complete_canonical_observation_rowset(
        rowset,
        run_family="crop_runs",
        label="Crop rowset",
    )
    for value in (False, None):
        if value is None:
            del rowset.attrs["stage_selector_eligible"]
        else:
            rowset.attrs["stage_selector_eligible"] = value
        with pytest.raises(
            ObservationCoordinatePublicationError,
            match="not explicitly eligible",
        ):
            observation_publication._require_complete_canonical_observation_rowset(
                rowset,
                run_family="crop_runs",
                label="Crop rowset",
            )


def test_persisted_observation_rowset_gate_supports_explicit_staged_validation() -> None:
    rowset = FakeGroup(path="crop_runs/c1", archive_token=object())
    rowset.attrs.update(
        {
            "coordinate_contract": "canonical_v2",
            "palette_run_completion_contract": "palette.zarr_run_completion.v1",
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": False,
        }
    )

    observation_publication._require_complete_canonical_observation_rowset(
        rowset,
        run_family="crop_runs",
        label="Crop rowset",
        require_selector_eligible=False,
    )
    rowset.attrs["stage_selector_eligible"] = True
    with pytest.raises(
        ObservationCoordinatePublicationError,
        match="selector-ineligible",
    ):
        observation_publication._require_complete_canonical_observation_rowset(
            rowset,
            run_family="crop_runs",
            label="Crop rowset",
            require_selector_eligible=False,
        )


def test_coordinate_publication_checkpoint_restores_exact_attrs() -> None:
    node = FakeGroup(path="detect_runs/d1", archive_token=object())
    node.attrs.update({"keep": {"nested": [1, 2]}, "status": "running"})
    checkpoint = capture_observation_coordinate_publication_checkpoint(node)
    node.attrs.clear()
    node.attrs.update(
        {
            "coordinate_contract": "canonical_v2",
            "coordinate_descriptor": {"partial": True},
        }
    )

    restore_observation_coordinate_publication_checkpoint(
        checkpoint,
        cause=RuntimeError("completion failed"),
    )

    assert node.attrs == {"keep": {"nested": [1, 2]}, "status": "running"}


def test_detection_geometry_publishes_exact_identity_time_and_v2_descriptors() -> None:
    world = _world(convention="continuous", archive_token=object())
    values = _surface(world)
    evidence, rowset, key, source_frames, bbox_norm, bbox_img, centers = values

    result = publish_detection_observation_geometry(
        rowset,
        key,
        source_frames,
        bbox_norm,
        bbox_img,
        centers,
        frame_evidence=evidence,
    )

    assert result.row_identity.contract.domain == "observation_instance"
    assert result.temporal_authority.record.source_identity_mode == "instance_key"
    assert result.bbox_normalized.descriptor.space_id == ("source_camera_normalized_xy")
    assert result.bbox_normalized.descriptor.source_camera_overlay.status == (
        "requires_transform"
    )
    assert result.bbox_image.descriptor.geometry_type == "bbox_xyxy"
    assert result.bbox_image.descriptor.pixel_convention == "pixel_edge_half_open"
    assert result.centers_image.descriptor.geometry_type == "point_xy"
    assert result.centers_image.descriptor.pixel_convention == "continuous"
    assert (
        result.bbox_image.reference_frame_authority.record_ref
        != result.centers_image.reference_frame_authority.record_ref
    )
    assert centers.attrs[COORDINATE_DESCRIPTOR_ATTR]["schema_version"] == 2
    assert rowset.attrs[DETECTION_BBOX_PROJECTION_ATTR]["direction"] == (
        "source_camera_normalized_xy_to_source_camera_image_px"
    )
    assert rowset.attrs[BBOX_CENTER_DERIVATION_ATTR]["operation"] == (
        "half_open_xyxy_edges_to_continuous_midpoint_v2"
    )
    assert (
        require_bound_source_camera_position_surface(result.position_surface)
        is result.position_surface
    )
    loaded = load_detection_observation_geometry(
        rowset,
        key,
        source_frames,
        bbox_norm,
        bbox_img,
        centers,
        frame_evidence=evidence,
    )
    assert loaded.centers_image.descriptor == result.centers_image.descriptor


def test_detection_backend_result_projection_binds_exact_orig_shape_and_model() -> None:
    world = _world(convention="continuous", archive_token=object())
    evidence, rowset, _, _, bbox_norm, _, _ = _surface(world)
    rowset.attrs.update(
        {
            "model_path": "/models/detect.pt",
            "model_name": "detect.pt",
            "inference_height": 40,
            "inference_width": 50,
            "validated_backend_result_count": 2,
            "validated_backend_result_orig_shape_hw": [40, 50],
            "decode_backend_effective": "opencv",
            "video_reader_type": "opencv",
            "parameters": {
                "decode_backend_effective": "opencv",
                "resize_dims": [40, 50],
                "pre_resize_dims": [40, 50],
                "effective_input_resize_dims": [40, 50],
                "tensor_resize_dims": None,
                "imgsz_applied": [40, 50],
            },
        }
    )
    model_artifact = {
        "role": "detect_model",
        "path": "/models/detect.pt",
        "fingerprint_scheme": "content_v1",
        "sha256": "a" * 64,
        "size_bytes": 123,
        "mtime_ns": 456,
        "source": "computed",
    }

    published = publish_detection_backend_result_projection(
        rowset,
        bbox_norm,
        frame_evidence=evidence,
        model_artifact=model_artifact,
    )

    assert published.record["direction"] == (
        "detector_backend_result_image_px_to_source_camera_normalized_xy"
    )
    assert published.record["backend_result_space"]["pixel_convention"] == (
        "pixel_edge_half_open"
    )
    assert published.record["result_px_to_source_camera_bbox_matrix"] == [
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    assert published.record["runtime_result_validation"][
        "network_preprocessing_authority"
    ] == "not_persisted_not_used_as_coordinate_projection_authority"
    loaded = load_detection_backend_result_projection(
        rowset,
        bbox_norm,
        frame_evidence=evidence,
    )
    assert loaded.record_sha256 == published.record_sha256

    rowset.attrs["validated_backend_result_orig_shape_hw"] = [20, 25]
    with pytest.raises(
        ObservationCoordinatePublicationError,
        match="count/orig_shape",
    ):
        load_detection_backend_result_projection(
            rowset,
            bbox_norm,
            frame_evidence=evidence,
        )

    assert DETECTION_BACKEND_RESULT_PROJECTION_ATTR in rowset.attrs


def test_detection_geometry_rejects_inexact_bbox_center_before_attrs_mutation() -> None:
    world = _world(convention="continuous", archive_token=object())
    values = _surface(world, alter_center=True)
    evidence, rowset, key, source_frames, bbox_norm, bbox_img, centers = values

    with pytest.raises(
        ObservationCoordinatePublicationError,
        match="exact dtype-preserving midpoint",
    ):
        publish_detection_observation_geometry(
            rowset,
            key,
            source_frames,
            bbox_norm,
            bbox_img,
            centers,
            frame_evidence=evidence,
        )

    assert rowset.attrs == {}
    assert key.attrs == {}
    assert bbox_norm.attrs == {}
    assert bbox_img.attrs == {}
    assert centers.attrs == {}


def test_detection_geometry_loader_rejects_bbox_payload_tampering() -> None:
    world = _world(convention="continuous", archive_token=object())
    values = _surface(world)
    evidence, rowset, key, source_frames, bbox_norm, bbox_img, centers = values
    publish_detection_observation_geometry(
        rowset,
        key,
        source_frames,
        bbox_norm,
        bbox_img,
        centers,
        frame_evidence=evidence,
    )

    bbox_img.data[0, 0] += 1.0
    with pytest.raises(
        ObservationCoordinatePublicationError,
        match="exact dtype-preserving projection",
    ):
        load_detection_observation_geometry(
            rowset,
            key,
            source_frames,
            bbox_norm,
            bbox_img,
            centers,
            frame_evidence=evidence,
        )


def test_detection_geometry_rolls_back_identity_when_temporal_mapping_is_invalid() -> (
    None
):
    world = _world(convention="continuous", archive_token=object())
    values = _surface(world, frames=(0, 2))
    evidence, rowset, key, source_frames, bbox_norm, bbox_img, centers = values

    with pytest.raises(Exception, match="range|frame"):
        publish_detection_observation_geometry(
            rowset,
            key,
            source_frames,
            bbox_norm,
            bbox_img,
            centers,
            frame_evidence=evidence,
        )

    assert rowset.attrs == {}
    assert key.attrs == {}
    assert source_frames.attrs == {}
    assert bbox_norm.attrs == {}
    assert bbox_img.attrs == {}
    assert centers.attrs == {}


@pytest.mark.parametrize(
    "failure",
    [
        KeyboardInterrupt("injected coordinate publication interrupt"),
        SystemExit("injected coordinate publication exit"),
    ],
    ids=["keyboard_interrupt", "system_exit"],
)
def test_detection_geometry_publication_rolls_back_baseexception(
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException,
) -> None:
    world = _world(convention="continuous", archive_token=object())
    values = _surface(world)
    evidence, rowset, key, source_frames, bbox_norm, bbox_img, centers = values
    nodes = (rowset, key, source_frames, bbox_norm, bbox_img, centers)
    snapshots = [dict(node.attrs) for node in nodes]

    def interrupt_temporal(*_args, **_kwargs):
        source_frames.attrs["source_row_temporal_authority"] = {"partial": True}
        raise failure

    monkeypatch.setattr(
        observation_publication,
        "stamp_source_row_temporal_authority",
        interrupt_temporal,
    )

    with pytest.raises(type(failure), match="injected coordinate publication"):
        publish_detection_observation_geometry(
            rowset,
            key,
            source_frames,
            bbox_norm,
            bbox_img,
            centers,
            frame_evidence=evidence,
        )

    assert [dict(node.attrs) for node in nodes] == snapshots


def _published_detection(world):
    values = _surface(world)
    evidence, rowset, key, source_frames, bbox_norm, bbox_img, centers = values
    published = publish_detection_observation_geometry(
        rowset,
        key,
        source_frames,
        bbox_norm,
        bbox_img,
        centers,
        frame_evidence=evidence,
    )
    return published


def _crop_copy(world, source, *, wrong_keys: bool = False):
    token = world["archive_token"]
    rowset = FakeGroup(path="crop_runs/c1", archive_token=token)
    source_rows_values = np.asarray([1, 0], dtype=np.int64)
    source_rows = FakeArray(
        source_rows_values,
        path=f"{rowset.path}/detection_indices",
        archive_token=token,
    )
    source_keys = np.asarray(source._key_node[:])
    keys_values = source_keys[source_rows_values].copy()
    if wrong_keys:
        keys_values[0] += np.uint64(1)
    key = FakeArray(
        keys_values,
        path=f"{rowset.path}/instance_key",
        archive_token=token,
    )
    source_frames_values = np.asarray(source._source_frame_index_node[:])
    source_frames = FakeArray(
        source_frames_values[source_rows_values],
        path=f"{rowset.path}/source_acquisition_frame_index",
        archive_token=token,
    )
    bbox_norm_values = np.asarray(source._bbox_norm_node[:])
    bbox_img_values = np.asarray(source._bbox_img_node[:])
    centers_values = np.asarray(source._centers_img_node[:])
    bbox_norm = FakeArray(
        bbox_norm_values[source_rows_values],
        path=f"{rowset.path}/bbox_norm_coords",
        archive_token=token,
    )
    bbox_img = FakeArray(
        bbox_img_values[source_rows_values],
        path=f"{rowset.path}/bbox_img_xyxy",
        archive_token=token,
    )
    centers = FakeArray(
        centers_values[source_rows_values],
        path=f"{rowset.path}/centers_img_xy",
        archive_token=token,
    )
    return rowset, key, source_rows, source_frames, bbox_norm, bbox_img, centers


def _crop_roi(world, crop, *, roi_height: int = 40, alter_bbox: bool = False):
    token = world["archive_token"]
    placements_values = np.asarray(
        [[60.0, 0.0, 40.0, 40.0], [0.0, 20.0, 40.0, 40.0]],
        dtype=np.float64,
    )
    placements = FakeArray(
        placements_values,
        path=f"{crop._rowset_node.path}/source_crop_xywh",
        archive_token=token,
    )
    point_ownership = stamp_crop_placement_ownership(
        placements,
        row_identity=crop.row_identity,
        source_camera_frame=crop.source_geometry.frame_evidence.source_camera_frame,
    )
    ownership = stamp_crop_placement_ownership(
        placements,
        row_identity=crop.row_identity,
        source_camera_frame=(
            crop.source_geometry.frame_evidence.bbox_source_camera_frame
        ),
        attr_name=CROP_PLACEMENT_PIXEL_EDGE_OWNERSHIP_ATTR,
    )
    roi_pixels = FakeArray(
        np.zeros((2, roi_height, 40), dtype=np.uint8),
        path=f"{crop._rowset_node.path}/roi_images",
        archive_token=token,
    )
    point_roi_frame = stamp_roi_pixel_frame_authority(
        bind_array_reference_extent(roi_pixels, units="px"),
        frame_id="c1_roi_continuous",
        pixel_convention="continuous",
        crop_placement_ownership=point_ownership,
    )
    point_authority = stamp_crop_placement_transform_authority(
        placements,
        authority_id="c1_roi_continuous_to_source_camera",
        source_frame=point_roi_frame,
        target_frame=crop.source_geometry.frame_evidence.source_camera_frame,
    )
    stamp_directed_transform_v2(
        placements,
        transform_id="c1_roi_continuous_to_source_camera",
        authority=point_authority,
        source_frame=point_roi_frame,
        target_frame=crop.source_geometry.frame_evidence.source_camera_frame,
        row_identity=crop.row_identity,
    )
    bbox_frame_node = FakeGroup(
        path=(
            f"{crop._rowset_node.path}/coordinate_frames/roi_bbox_edge"
        ),
        archive_token=token,
    )
    coordinate_frames = FakeGroup(
        path=f"{crop._rowset_node.path}/coordinate_frames",
        archive_token=token,
    )
    coordinate_frames["roi_bbox_edge"] = bbox_frame_node
    crop._rowset_node["coordinate_frames"] = coordinate_frames
    crop._rowset_node["roi_images"] = roi_pixels
    roi_frame = stamp_roi_pixel_frame_authority(
        publish_crop_roi_bbox_edge_reference_extent(
            bbox_frame_node,
            roi_pixels,
        ),
        frame_id="c1_roi_bbox_edge",
        pixel_convention="pixel_edge_half_open",
        crop_placement_ownership=ownership,
    )
    authority = stamp_crop_placement_transform_authority(
        placements,
        authority_id="c1_roi_to_source_camera",
        source_frame=roi_frame,
        target_frame=(
            crop.source_geometry.frame_evidence.bbox_source_camera_frame
        ),
        attr_name=TRANSFORM_AUTHORITY_PIXEL_EDGE_ATTR,
    )
    link = stamp_directed_transform_v2(
        placements,
        transform_id="c1_roi_to_source_camera",
        authority=authority,
        source_frame=roi_frame,
        target_frame=(
            crop.source_geometry.frame_evidence.bbox_source_camera_frame
        ),
        row_identity=crop.row_identity,
        attr_name=DIRECTED_TRANSFORM_V2_PIXEL_EDGE_ATTR,
    )
    bbox_img = np.asarray(crop._bbox_img_node[:])
    offsets = np.column_stack(
        (
            placements_values[:, 0],
            placements_values[:, 1],
            placements_values[:, 0],
            placements_values[:, 1],
        )
    )
    bbox_roi_values = bbox_img - offsets
    if roi_height != 40:
        # Keep the naive translation to prove a changed ROI reference extent
        # cannot be hidden behind matching-looking local coordinates.
        bbox_roi_values = bbox_roi_values.copy()
    if alter_bbox:
        bbox_roi_values[0, 0] += 1.0
    bbox_roi = FakeArray(
        bbox_roi_values,
        path=f"{crop._rowset_node.path}/bbox_roi_xyxy",
        archive_token=token,
    )
    chain = resolve_bound_directed_transform_chain((link,))
    return placements, bbox_roi, ownership, roi_frame, chain, point_ownership


def test_crop_geometry_copies_exact_instance_key_selection_and_roi_lineage() -> None:
    world = _world(convention="continuous", archive_token=object())
    source = _published_detection(world)
    nodes = _crop_copy(world, source)
    rowset, key, source_rows, source_frames, bbox_norm, bbox_img, centers = nodes
    crop = publish_crop_observation_geometry(
        rowset,
        key,
        source_rows,
        source_frames,
        bbox_norm,
        bbox_img,
        centers,
        source_geometry=source,
    )
    placements, bbox_roi, ownership, roi_frame, chain, _ = _crop_roi(world, crop)
    roi = publish_crop_roi_geometry(
        placements,
        bbox_roi,
        crop_geometry=crop,
        crop_placement_ownership=ownership,
        roi_frame=roi_frame,
        roi_to_source_camera=chain,
    )

    assert np.array_equal(key[:], np.asarray([202, 101], dtype=np.uint64))
    assert crop.centers_image.descriptor.space_id == "source_camera_image_px"
    assert crop.position_surface.temporal_authority.record.camera_id == "camera-a"
    assert roi.source_crop_xywh.descriptor.space_id == "source_camera_image_px"
    assert roi.source_crop_xywh.descriptor.pixel_convention == (
        "pixel_edge_half_open"
    )
    assert roi.bbox_roi_xyxy.descriptor.space_id == "roi_local_px"
    assert roi.bbox_roi_xyxy.descriptor.pixel_convention == "pixel_edge_half_open"
    assert roi.bbox_roi_xyxy.descriptor.source_camera_overlay.status == (
        "requires_transform"
    )
    loaded_crop = load_crop_observation_geometry(
        rowset,
        key,
        source_rows,
        source_frames,
        bbox_norm,
        bbox_img,
        centers,
        source_geometry=source,
    )
    loaded_roi = load_crop_roi_geometry(
        placements,
        bbox_roi,
        crop_geometry=loaded_crop,
        crop_placement_ownership=ownership,
        roi_frame=roi_frame,
        roi_to_source_camera=chain,
    )
    assert loaded_roi.derivation.record_sha256 == roi.derivation.record_sha256


def test_crop_bbox_edge_frame_extent_is_bound_to_exact_roi_image_metadata() -> None:
    token = object()
    roi_images = FakeArray(
        np.zeros((2, 40, 60), dtype=np.uint8),
        path="crop_runs/c1/roi_images",
        archive_token=token,
    )
    frame_node = FakeGroup(
        path="crop_runs/c1/coordinate_frames/roi_bbox_edge",
        archive_token=token,
    )

    published = publish_crop_roi_bbox_edge_reference_extent(
        frame_node,
        roi_images,
    )

    assert (published.width, published.height, published.units) == (60, 40, "px")
    loaded = load_crop_roi_bbox_edge_reference_extent(frame_node, roi_images)
    assert loaded.record_sha256 == published.record_sha256

    changed_extent = FakeArray(
        np.zeros((2, 20, 60), dtype=np.uint8),
        path="crop_runs/c1/roi_images",
        archive_token=token,
    )
    with pytest.raises(
        ObservationCoordinatePublicationError,
        match="differs from exact live roi_images metadata",
    ):
        load_crop_roi_bbox_edge_reference_extent(
            frame_node,
            changed_extent,
        )


def test_crop_top_left_compatibility_surface_is_descriptor_bound_and_drift_checked() -> None:
    world = _world(convention="continuous", archive_token=object())
    source = _published_detection(world)
    crop = publish_crop_observation_geometry(
        *_crop_copy(world, source),
        source_geometry=source,
    )
    placements, bbox_roi, ownership, roi_frame, chain, point_ownership = _crop_roi(
        world,
        crop,
    )
    top_left = FakeArray(
        np.asarray(placements[:])[:, :2].astype(np.int32),
        path=f"{crop._rowset_node.path}/roi_coordinates_full",
        archive_token=world["archive_token"],
    )

    published = publish_crop_roi_geometry(
        placements,
        bbox_roi,
        crop_geometry=crop,
        crop_placement_ownership=ownership,
        roi_frame=roi_frame,
        roi_to_source_camera=chain,
        roi_top_left_node=top_left,
        roi_top_left_placement_ownership=point_ownership,
    )

    assert published.roi_top_left_xy is not None
    assert published.top_left_derivation is not None
    descriptor = published.roi_top_left_xy.descriptor
    assert descriptor.profile_id == "source_camera_image_px.top_left_y_down.v1"
    assert descriptor.geometry_type == "point_xy"
    assert descriptor.pixel_convention == "continuous"
    assert (
        published.roi_top_left_xy.reference_frame_authority.record_ref
        == crop.source_geometry.frame_evidence.source_camera_frame.record_ref
    )
    assert descriptor.source_camera_overlay.status == "direct"
    assert descriptor.row_identity is not None
    loaded = load_crop_roi_geometry(
        placements,
        bbox_roi,
        crop_geometry=crop,
        crop_placement_ownership=ownership,
        roi_frame=roi_frame,
        roi_to_source_camera=chain,
        roi_top_left_node=top_left,
        roi_top_left_placement_ownership=point_ownership,
    )
    assert loaded.top_left_derivation is not None
    assert (
        loaded.top_left_derivation.record_sha256
        == published.top_left_derivation.record_sha256
    )

    top_left.data[0, 0] += 1
    with pytest.raises(
        ObservationCoordinatePublicationError,
        match="exact source-camera top-left",
    ):
        load_crop_roi_geometry(
            placements,
            bbox_roi,
            crop_geometry=crop,
            crop_placement_ownership=ownership,
            roi_frame=roi_frame,
            roi_to_source_camera=chain,
            roi_top_left_node=top_left,
            roi_top_left_placement_ownership=point_ownership,
        )


def test_crop_geometry_rejects_identity_mismatch_before_publication() -> None:
    world = _world(convention="continuous", archive_token=object())
    source = _published_detection(world)
    nodes = _crop_copy(world, source, wrong_keys=True)
    rowset, key, source_rows, source_frames, bbox_norm, bbox_img, centers = nodes

    with pytest.raises(
        ObservationCoordinatePublicationError,
        match="exact selected/reordered source identity",
    ):
        publish_crop_observation_geometry(
            rowset,
            key,
            source_rows,
            source_frames,
            bbox_norm,
            bbox_img,
            centers,
            source_geometry=source,
        )
    assert rowset.attrs == {}
    assert key.attrs == {}


def test_crop_roi_rejects_roi_image_mixing_and_reference_extent_drift() -> None:
    world = _world(convention="continuous", archive_token=object())
    source = _published_detection(world)
    nodes = _crop_copy(world, source)
    crop = publish_crop_observation_geometry(
        *nodes,
        source_geometry=source,
    )
    placements, bbox_roi, ownership, roi_frame, chain, _ = _crop_roi(
        world,
        crop,
        alter_bbox=True,
    )
    with pytest.raises(
        ObservationCoordinatePublicationError,
        match="does not project exactly",
    ):
        publish_crop_roi_geometry(
            placements,
            bbox_roi,
            crop_geometry=crop,
            crop_placement_ownership=ownership,
            roi_frame=roi_frame,
            roi_to_source_camera=chain,
        )

    placements2, bbox_roi2, ownership2, roi_frame2, chain2, _ = _crop_roi(
        world,
        crop,
        roi_height=20,
    )
    with pytest.raises(
        ObservationCoordinatePublicationError,
        match="does not project exactly",
    ):
        publish_crop_roi_geometry(
            placements2,
            bbox_roi2,
            crop_geometry=crop,
            crop_placement_ownership=ownership2,
            roi_frame=roi_frame2,
            roi_to_source_camera=chain2,
        )


def test_incremental_crop_preflight_uses_exact_source_geometry_and_placement() -> None:
    world = _world(convention="continuous", archive_token=object())
    source = _published_detection(world)
    values = {
        "instance_key": np.asarray(source._key_node[:]),
        "source_acquisition_frame_index": np.asarray(
            source._source_frame_index_node[:]
        ),
        "bbox_norm_coords": np.asarray(source._bbox_norm_node[:]),
    }
    snapshot = CropSourceSnapshot(
        source_path=source.row_identity.rowset_path,
        instance_keys=values["instance_key"],
        frame_indices=values["source_acquisition_frame_index"],
        bbox_norm_coords=values["bbox_norm_coords"],
        optional_row_arrays={},
        signatures=np.empty((2, 32), dtype=np.uint8),
        signature_spec=None,  # not consulted by coordinate preflight
        rowset_fingerprint=None,  # not consulted by coordinate preflight
    )
    roi_coordinates = np.asarray([[15, 30], [65, 10]], dtype=np.int32)

    arrays = _canonical_crop_coordinate_arrays(
        snapshot,
        frame_shape=(80, 100),
        roi_size=(20, 20),
        roi_coordinates_full=roi_coordinates,
        source_geometry=source,
    )

    assert np.array_equal(
        arrays["source_acquisition_frame_index"],
        values["source_acquisition_frame_index"],
    )
    assert np.array_equal(
        arrays["source_crop_xywh"],
        np.asarray([[15, 30, 20, 20], [65, 10, 20, 20]], dtype=np.float64),
    )
    offsets = np.asarray(
        [[15, 30, 15, 30], [65, 10, 65, 10]],
        dtype=np.float64,
    )
    assert np.array_equal(
        arrays["bbox_roi_xyxy"],
        np.asarray(source._bbox_img_node[:]) - offsets,
    )

    with pytest.raises(IncrementalCropError, match="acquisition-frame mapping"):
        _canonical_crop_coordinate_arrays(
            replace(snapshot, frame_indices=snapshot.frame_indices[::-1]),
            frame_shape=(80, 100),
            roi_size=(20, 20),
            roi_coordinates_full=roi_coordinates,
            source_geometry=source,
        )
    with pytest.raises(IncrementalCropError, match="fully contained"):
        _canonical_crop_coordinate_arrays(
            snapshot,
            frame_shape=(80, 100),
            roi_size=(20, 20),
            roi_coordinates_full=np.asarray([[-1, 30], [65, 10]], dtype=np.int32),
            source_geometry=source,
        )


def test_public_persisted_crop_loaders_cannot_bypass_selector_eligibility() -> None:
    for loader_name in (
        "load_persisted_crop_observation_geometry",
        "load_persisted_ordinary_crop_observation_geometry",
    ):
        signature = inspect.signature(getattr(observation_publication, loader_name))
        assert "require_selector_eligible" not in signature.parameters

    for loader_name in (
        "_load_persisted_crop_observation_geometry",
        "_load_persisted_ordinary_crop_observation_geometry",
    ):
        signature = inspect.signature(getattr(observation_publication, loader_name))
        parameter = signature.parameters["require_selector_eligible"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is True
