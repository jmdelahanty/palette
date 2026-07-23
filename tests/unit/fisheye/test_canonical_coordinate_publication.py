from __future__ import annotations

import copy

import numpy as np
import pytest

from fisheye.shared.canonical_coordinate_publication import (
    CanonicalCoordinatePublicationError,
    build_bound_canonical_coordinate_descriptor,
    load_bound_canonical_coordinate_descriptor,
    migrate_historical_coordinate_descriptor_v1_to_v2,
    require_bound_canonical_coordinate_descriptor,
    stamp_bound_canonical_coordinate_descriptor,
    stamp_bound_canonical_coordinate_descriptors,
)
from fisheye.shared.coordinate_descriptor import (
    CANONICAL_OVERLAY_DIRECT,
    CANONICAL_OVERLAY_NOT_SUITABLE,
    CANONICAL_OVERLAY_REQUIRES_TRANSFORM,
)
from fisheye.shared.coordinate_frame_record import (
    build_physical_frame_calibration_record,
    stamp_physical_frame_calibration_record,
    stamp_selected_camera_frame_evidence,
)
from fisheye.shared.coordinate_identity import (
    OBSERVATION_INSTANCE_DOMAIN,
    build_row_identity_contract,
    stamp_and_bind_row_identity_contract,
)
from fisheye.shared.coordinate_reference import bind_array_reference_extent
from fisheye.shared.coordinate_record import (
    stamp_and_bind_persisted_coordinate_record,
)
from fisheye.shared.directed_transform_chain import (
    resolve_bound_directed_transform_chain,
)
from fisheye.shared.pixel_frame_authority import (
    stamp_normalized_pixel_frame_authority,
    stamp_source_camera_pixel_frame_authority,
)
_ARCHIVE_TOKEN = object()


class _Array:
    def __init__(
        self,
        values,
        *,
        path: str,
        dtype=None,
        archive_token: object = _ARCHIVE_TOKEN,
    ) -> None:
        self.values = np.asarray(values, dtype=dtype)
        self._coordinate_archive_token = archive_token
        self.path = path
        self.shape = self.values.shape
        self.dtype = self.values.dtype
        self.attrs: dict[str, object] = {}

    def __getitem__(self, item):
        return self.values[item]


class _Group:
    def __init__(
        self,
        *,
        path: str,
        archive_token: object = _ARCHIVE_TOKEN,
    ) -> None:
        self.path = path
        self._coordinate_archive_token = archive_token
        self.attrs: dict[str, object] = {}


def _identity(
    path: str = "analysis/detect_runs/d1",
    *,
    archive_token: object = _ARCHIVE_TOKEN,
):
    values = np.asarray([11, 12, 13], dtype=np.uint64)
    rowset = _Group(path=path, archive_token=archive_token)
    key = _Array(
        values,
        path=f"{path}/instance_key",
        archive_token=archive_token,
    )
    contract = build_row_identity_contract(
        domain=OBSERVATION_INSTANCE_DOMAIN,
        values=values,
    )
    return rowset, key, stamp_and_bind_row_identity_contract(
        rowset,
        key,
        contract=contract,
    )


def _direct_binding(
    *,
    archive_token: object = _ARCHIVE_TOKEN,
    dtype=np.float64,
):
    from tests.unit.fisheye.test_directed_transform_chain import _world

    world = _world(convention="pixel_center", archive_token=archive_token)
    _, _, identity = _identity(archive_token=archive_token)
    positions = _Array(
        np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=dtype),
        path="analysis/detect_runs/d1/centers_img_xy",
        archive_token=archive_token,
    )
    binding = build_bound_canonical_coordinate_descriptor(
        positions,
        profile_id="source_camera_image_px.top_left_y_down.v1",
        geometry_type="point_xy",
        components=("x", "y"),
        component_units=("px", "px"),
        pixel_convention="pixel_center",
        row_identity=identity,
        reference_frame_authority=world["camera_frame"],
        source_camera_overlay_status=CANONICAL_OVERLAY_DIRECT,
    )
    return positions, world["camera_array"], binding


@pytest.mark.parametrize("dtype", [np.bool_, object, "S4", "U4", np.complex128])
def test_publication_rejects_nonnumeric_coordinate_dtypes(dtype) -> None:
    with pytest.raises(CanonicalCoordinatePublicationError) as exc_info:
        _direct_binding(dtype=dtype)
    assert "coordinate_owner_dtype_nonnumeric" in {
        issue.code for issue in exc_info.value.issues
    }


def test_public_verifier_accepts_only_the_sealed_live_binding() -> None:
    _, _, binding = _direct_binding()
    assert require_bound_canonical_coordinate_descriptor(binding) is binding

    with pytest.raises(CanonicalCoordinatePublicationError) as exc_info:
        require_bound_canonical_coordinate_descriptor(object())
    assert "coordinate_descriptor_binding_unverified" in {
        issue.code for issue in exc_info.value.issues
    }


def test_source_camera_unit_vectors_round_trip_without_inheriting_point_semantics() -> None:
    from tests.unit.fisheye.test_directed_transform_chain import _world

    world = _world(convention="continuous")
    _, _, identity = _identity(archive_token=world["archive_token"])
    vectors = _Array(
        np.asarray([[1.0, 0.0], [0.0, -1.0], [np.nan, np.nan]], dtype=np.float32),
        path="analysis/detect_runs/d1/body_forward_axis_xy",
        archive_token=world["archive_token"],
    )
    binding = build_bound_canonical_coordinate_descriptor(
        vectors,
        profile_id="source_camera_image_px.unit_vector_y_down.v1",
        geometry_type="vector_xy",
        components=("x", "y"),
        component_units=("unitless", "unitless"),
        pixel_convention="not_applicable",
        row_identity=identity,
        reference_frame_authority=world["camera_frame"],
        source_camera_overlay_status=CANONICAL_OVERLAY_NOT_SUITABLE,
    )
    stamp_bound_canonical_coordinate_descriptor(binding)
    loaded = load_bound_canonical_coordinate_descriptor(
        vectors,
        row_identity=identity,
        reference_frame_authority=world["camera_frame"],
    )
    assert loaded.descriptor.profile_id == "source_camera_image_px.unit_vector_y_down.v1"
    assert loaded.descriptor.component_units == ("unitless", "unitless")

    with pytest.raises(ValueError, match="Profile does not permit 'direct'"):
        build_bound_canonical_coordinate_descriptor(
            vectors,
            profile_id="source_camera_image_px.unit_vector_y_down.v1",
            geometry_type="vector_xy",
            components=("x", "y"),
            component_units=("unitless", "unitless"),
            pixel_convention="not_applicable",
            row_identity=identity,
            reference_frame_authority=world["camera_frame"],
            source_camera_overlay_status=CANONICAL_OVERLAY_DIRECT,
        )


@pytest.mark.parametrize(
    ("profile_id", "geometry_type", "shape", "units"),
    (
        (
            "source_camera_image_px.unit_vector_y_down.v1",
            "vector_sequence_xy",
            (3, 5, 2),
            ("unitless", "unitless"),
        ),
        (
            "source_camera_image_px.displacement_vector_y_down.v1",
            "vector_xy",
            (3, 2),
            ("px", "px"),
        ),
    ),
)
def test_nonpositional_source_camera_vectors_bind_exact_frame_without_pixel_sampling(
    profile_id: str,
    geometry_type: str,
    shape: tuple[int, ...],
    units: tuple[str, str],
) -> None:
    from tests.unit.fisheye.test_directed_transform_chain import _world

    world = _world(convention="continuous")
    _, _, identity = _identity(archive_token=world["archive_token"])
    values = _Array(
        np.zeros(shape, dtype=np.float32),
        path="analysis/detect_runs/d1/nonpositional_vector",
        archive_token=world["archive_token"],
    )
    binding = build_bound_canonical_coordinate_descriptor(
        values,
        profile_id=profile_id,
        geometry_type=geometry_type,
        components=("x", "y"),
        component_units=units,
        pixel_convention="not_applicable",
        row_identity=identity,
        reference_frame_authority=world["camera_frame"],
        source_camera_overlay_status=CANONICAL_OVERLAY_NOT_SUITABLE,
    )
    stamp_bound_canonical_coordinate_descriptor(binding)
    loaded = load_bound_canonical_coordinate_descriptor(
        values,
        row_identity=identity,
        reference_frame_authority=world["camera_frame"],
    )
    assert loaded.descriptor.profile_id == profile_id
    assert loaded.descriptor.pixel_convention == "not_applicable"


def test_generic_extent_cannot_authorize_source_camera_coordinates() -> None:
    _, _, identity = _identity()
    camera = _Array(
        np.zeros((3, 80, 100), dtype=np.uint8),
        path="analysis/debug/native_sized",
    )
    points = _Array(
        np.zeros((3, 2), dtype=np.float64),
        path="analysis/detect_runs/d1/points",
    )
    with pytest.raises(CanonicalCoordinatePublicationError) as exc_info:
        build_bound_canonical_coordinate_descriptor(
            points,
            profile_id="source_camera_image_px.top_left_y_down.v1",
            geometry_type="point_xy",
            components=("x", "y"),
            component_units=("px", "px"),
            pixel_convention="pixel_center",
            row_identity=identity,
            reference_extent=bind_array_reference_extent(camera, units="px"),
            source_camera_overlay_status=CANONICAL_OVERLAY_DIRECT,
        )
    assert "coordinate_pixel_frame_evidence_mismatch" in {
        issue.code for issue in exc_info.value.issues
    }


def test_pixel_profile_without_typed_endpoint_authority_fails_closed() -> None:
    _, _, identity = _identity()
    points = _Array(
        np.zeros((3, 2), dtype=np.float64),
        path="analysis/detect_runs/d1/texture_points",
    )
    with pytest.raises(CanonicalCoordinatePublicationError) as exc_info:
        build_bound_canonical_coordinate_descriptor(
            points,
            profile_id="stimulus_texture_px.top_left_y_down.v1",
            geometry_type="point_xy",
            components=("x", "y"),
            component_units=("px", "px"),
            pixel_convention="pixel_center",
            row_identity=identity,
            source_camera_overlay_status=CANONICAL_OVERLAY_NOT_SUITABLE,
        )
    assert "coordinate_profile_publication_unavailable" in {
        issue.code for issue in exc_info.value.issues
    }


def test_arena_relative_canvas_profile_requires_exact_typed_arena_authority() -> None:
    from tests.unit.fisheye.test_directed_transform_chain import _arena_frame, _world

    world = _world(convention="pixel_center")
    _, _, _, arena_frame = _arena_frame(world)
    _, _, identity = _identity(archive_token=world["archive_token"])
    points = _Array(
        np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        path="analysis/detect_runs/d1/arena_points",
        archive_token=world["archive_token"],
    )
    binding = build_bound_canonical_coordinate_descriptor(
        points,
        profile_id="arena_relative_canvas_px.top_left_y_down.v1",
        geometry_type="point_xy",
        components=("x", "y"),
        component_units=("px", "px"),
        pixel_convention="pixel_center",
        row_identity=identity,
        reference_frame_authority=arena_frame,
        source_camera_overlay_status=CANONICAL_OVERLAY_NOT_SUITABLE,
    )
    assert binding.descriptor.origin == "arena_top_left"
    assert binding.descriptor.reference_extent.width == 80
    assert binding.descriptor.frame_record.record_ref == arena_frame.record_ref


def test_normalized_camera_coordinates_bind_exact_normalized_frame() -> None:
    from tests.unit.fisheye.test_directed_transform_chain import _world

    world = _world(convention="pixel_center")
    _, _, identity = _identity(archive_token=world["archive_token"])
    points = _Array(
        np.asarray([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
        path="analysis/detect_runs/d1/centers_normalized_xy",
        archive_token=world["archive_token"],
    )
    normalized_frame = stamp_normalized_pixel_frame_authority(
        _Group(
            path="analysis/coordinate_frames/source_camera/camera-a/normalized",
            archive_token=world["archive_token"],
        ),
        frame_id="camera_a_normalized",
        pixel_frame=world["camera_frame"],
    )
    binding = build_bound_canonical_coordinate_descriptor(
        points,
        profile_id="source_camera_normalized_xy.top_left_y_down.v1",
        geometry_type="point_xy",
        components=("x", "y"),
        component_units=("normalized", "normalized"),
        pixel_convention="continuous",
        row_identity=identity,
        reference_frame_authority=normalized_frame,
        source_camera_overlay_status=CANONICAL_OVERLAY_NOT_SUITABLE,
    )
    assert binding.descriptor.reference_extent.width == 100
    assert binding.descriptor.frame_record.record_ref == normalized_frame.record_ref


def test_source_camera_point_and_bbox_conventions_coexist_without_overwrite() -> None:
    from tests.unit.fisheye.test_directed_transform_chain import _world

    world = _world(convention="pixel_center")
    token = world["archive_token"]
    _, _, identity = _identity(archive_token=token)
    point_frame = world["camera_frame"]
    bbox_frame = stamp_source_camera_pixel_frame_authority(
        _Group(
            path="analysis/coordinate_frames/source_camera/camera-a/pixel_edge_half_open",
            archive_token=token,
        ),
        frame_id="camera_a_native_bbox_edges",
        pixel_convention="pixel_edge_half_open",
        acquisition_frame=world["acquisition_frame"],
    )
    points = _Array(
        np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        path="analysis/detect_runs/d1/centers_img_xy",
        archive_token=token,
    )
    boxes = _Array(
        np.asarray(
            [[0.0, 0.0, 4.0, 5.0], [1.0, 2.0, 5.0, 6.0], [2.0, 3.0, 7.0, 8.0]]
        ),
        path="analysis/detect_runs/d1/bboxes_img_xyxy",
        archive_token=token,
    )
    point_binding = build_bound_canonical_coordinate_descriptor(
        points,
        profile_id="source_camera_image_px.top_left_y_down.v1",
        geometry_type="point_xy",
        components=("x", "y"),
        component_units=("px", "px"),
        pixel_convention="pixel_center",
        row_identity=identity,
        reference_frame_authority=point_frame,
        source_camera_overlay_status=CANONICAL_OVERLAY_DIRECT,
    )
    bbox_binding = build_bound_canonical_coordinate_descriptor(
        boxes,
        profile_id="source_camera_image_px.top_left_y_down.v1",
        geometry_type="bbox_xyxy",
        components=("x_min", "y_min", "x_max", "y_max"),
        component_units=("px", "px", "px", "px"),
        pixel_convention="pixel_edge_half_open",
        row_identity=identity,
        reference_frame_authority=bbox_frame,
        source_camera_overlay_status=CANONICAL_OVERLAY_DIRECT,
    )

    point_binding.reference_frame_authority.assert_verified()
    bbox_binding.reference_frame_authority.assert_verified()
    assert point_binding.reference_frame_authority.pixel_convention == "pixel_center"
    assert (
        bbox_binding.reference_frame_authority.pixel_convention
        == "pixel_edge_half_open"
    )
    assert point_frame.record_ref != bbox_frame.record_ref


def test_direct_surface_build_stamp_and_load_revalidate_exact_nodes() -> None:
    positions, _, binding = _direct_binding()

    stamped = stamp_bound_canonical_coordinate_descriptor(binding)
    assert stamped.space_id == "source_camera_image_px"
    assert stamped.frame_record is not None
    assert stamped.frame_record.kind == "pixel_frame_authority"
    assert stamped.frame_record.record_ref == binding.reference_frame_authority.record_ref
    assert stamped.row_identity.record_ref == (
        "/analysis/detect_runs/d1@row_identity_contract"
    )
    loaded = load_bound_canonical_coordinate_descriptor(
        positions,
        row_identity=binding.row_identity,
        reference_frame_authority=binding.reference_frame_authority,
    )
    assert loaded.descriptor == stamped


def test_owner_dtype_is_persisted_and_revalidated_exactly() -> None:
    positions, _, binding = _direct_binding()
    stamp_bound_canonical_coordinate_descriptor(binding)
    assert positions.attrs["coordinate_descriptor_owner_dtype"] == "<f8"

    positions.dtype = np.dtype("<f4")
    with pytest.raises(CanonicalCoordinatePublicationError) as bound_exc:
        stamp_bound_canonical_coordinate_descriptor(binding)
    assert "coordinate_owner_dtype_changed" in {
        issue.code for issue in bound_exc.value.issues
    }
    with pytest.raises(CanonicalCoordinatePublicationError) as load_exc:
        load_bound_canonical_coordinate_descriptor(
            positions,
            row_identity=binding.row_identity,
            reference_frame_authority=binding.reference_frame_authority,
        )
    assert "coordinate_owner_dtype_mismatch" in {
        issue.code for issue in load_exc.value.issues
    }


def test_loader_rejects_noninteger_schema_version_at_publication_boundary() -> None:
    positions, _, binding = _direct_binding()
    stamp_bound_canonical_coordinate_descriptor(binding)
    raw = copy.deepcopy(positions.attrs["coordinate_descriptor"])
    raw["schema_version"] = 2.0
    positions.attrs["coordinate_descriptor"] = raw
    with pytest.raises(CanonicalCoordinatePublicationError) as exc_info:
        load_bound_canonical_coordinate_descriptor(
            positions,
            row_identity=binding.row_identity,
            reference_frame_authority=binding.reference_frame_authority,
        )
    assert "canonical_schema_version_required" in {
        issue.code for issue in exc_info.value.issues
    }


def test_loader_reports_missing_array_descriptor_before_schema_validation() -> None:
    positions, _, binding = _direct_binding()

    with pytest.raises(CanonicalCoordinatePublicationError) as exc_info:
        load_bound_canonical_coordinate_descriptor(
            positions,
            row_identity=binding.row_identity,
            reference_frame_authority=binding.reference_frame_authority,
        )

    assert {issue.code for issue in exc_info.value.issues} == {
        "descriptor_attr_missing"
    }


def test_stamping_rejects_reference_or_identity_changed_after_binding() -> None:
    positions, camera, binding = _direct_binding()
    camera.shape = (3, 4512, 4511)
    with pytest.raises(CanonicalCoordinatePublicationError) as reference_exc:
        stamp_bound_canonical_coordinate_descriptor(binding)
    assert "coordinate_pixel_frame_evidence_unverified" in {
        issue.code for issue in reference_exc.value.issues
    }
    assert positions.attrs == {}

    positions, _, binding = _direct_binding()
    binding.row_identity._key_array_node.values[1] = 99
    with pytest.raises(CanonicalCoordinatePublicationError) as identity_exc:
        stamp_bound_canonical_coordinate_descriptor(binding)
    assert "coordinate_row_identity_unverified" in {
        issue.code for issue in identity_exc.value.issues
    }
    assert positions.attrs == {}


def test_wrong_geometry_layout_and_unrelated_rowset_fail_before_attrs_write() -> None:
    _, _, identity = _identity()
    wrong = _Array(
        np.zeros((3, 99), dtype=np.float32),
        path="analysis/detect_runs/d1/positions",
    )
    with pytest.raises(CanonicalCoordinatePublicationError) as shape_exc:
        build_bound_canonical_coordinate_descriptor(
            wrong,
            profile_id="source_camera_image_px.top_left_y_down.v1",
            geometry_type="points_xy",
            components=("x", "y"),
            component_units=("px", "px"),
            pixel_convention="pixel_center",
            row_identity=identity,
            reference_frame_authority=_direct_binding()[2].reference_frame_authority,
            source_camera_overlay_status=CANONICAL_OVERLAY_DIRECT,
        )
    assert "geometry_owner_shape_mismatch" in {
        issue.code for issue in shape_exc.value.issues
    }
    assert wrong.attrs == {}

    unrelated = _Array(
        np.zeros((3, 2), dtype=np.float32),
        path="analysis/other/positions",
    )
    with pytest.raises(CanonicalCoordinatePublicationError) as path_exc:
        build_bound_canonical_coordinate_descriptor(
            unrelated,
            profile_id="source_camera_image_px.top_left_y_down.v1",
            geometry_type="point_xy",
            components=("x", "y"),
            component_units=("px", "px"),
            pixel_convention="pixel_center",
            row_identity=identity,
            reference_frame_authority=_direct_binding()[2].reference_frame_authority,
            source_camera_overlay_status=CANONICAL_OVERLAY_DIRECT,
        )
    assert "coordinate_rowset_path_mismatch" in {
        issue.code for issue in path_exc.value.issues
    }


def test_transform_required_surface_uses_exact_ordered_resolved_chain() -> None:
    from tests.unit.fisheye.test_directed_transform_chain import _crop_link, _world

    world = _world(convention="pixel_center")
    rowset_path = world["rowset"].path
    _, transform = _crop_link(world)
    chain = resolve_bound_directed_transform_chain((transform,))
    points = _Array(
        np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        path=f"{rowset_path}/keypoints_roi_xy",
        archive_token=world["archive_token"],
    )
    binding = build_bound_canonical_coordinate_descriptor(
        points,
        profile_id="roi_local_px.top_left_y_down.v1",
        geometry_type="point_xy",
        components=("x", "y"),
        component_units=("px", "px"),
        pixel_convention="pixel_center",
        row_identity=world["identity"],
        source_camera_overlay_status=CANONICAL_OVERLAY_REQUIRES_TRANSFORM,
        transform_chain=chain,
    )
    assert binding.descriptor.source_camera_overlay.transform_refs[0].record_ref == (
        f"/{rowset_path}/source_crop_xywh@directed_transform_v2"
    )

    other_path = "analysis/crop_runs/crop_2"
    other_rowset = _Group(path=other_path, archive_token=world["archive_token"])
    other_key = _Array(
        np.asarray([101, 202], dtype=np.uint64),
        path=f"{other_path}/instance_key",
        archive_token=world["archive_token"],
    )
    other_identity = stamp_and_bind_row_identity_contract(
        other_rowset,
        other_key,
        contract=build_row_identity_contract(
            domain=OBSERVATION_INSTANCE_DOMAIN,
            values=other_key.values,
        ),
    )
    other_points = _Array(
        np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        path=f"{other_path}/keypoints_roi_xy",
        archive_token=world["archive_token"],
    )
    with pytest.raises(CanonicalCoordinatePublicationError) as identity_exc:
        build_bound_canonical_coordinate_descriptor(
            other_points,
            profile_id="roi_local_px.top_left_y_down.v1",
            geometry_type="point_xy",
            components=("x", "y"),
            component_units=("px", "px"),
            pixel_convention="pixel_center",
            row_identity=other_identity,
            source_camera_overlay_status=CANONICAL_OVERLAY_REQUIRES_TRANSFORM,
            transform_chain=chain,
        )
    assert {
        "coordinate_transform_row_identity_mismatch",
        "coordinate_pixel_frame_row_identity_mismatch",
    } & {issue.code for issue in identity_exc.value.issues}

    world["placements"].data[0, 0] += 1.0
    with pytest.raises(CanonicalCoordinatePublicationError) as stale_exc:
        stamp_bound_canonical_coordinate_descriptor(binding)
    assert {
        "coordinate_transform_evidence_unverified",
        "coordinate_pixel_frame_evidence_unverified",
    } & {
        issue.code for issue in stale_exc.value.issues
    }
    assert points.attrs == {}


def test_surface_set_rejects_identity_drift_and_rolls_back_all_attrs() -> None:
    first_node, _, first = _direct_binding()
    second_node = _Array(
        np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        path="analysis/detect_runs/d2/centers_img_xy",
    )
    _, _, second_identity = _identity("analysis/detect_runs/d2")
    second = build_bound_canonical_coordinate_descriptor(
        second_node,
        profile_id="source_camera_image_px.top_left_y_down.v1",
        geometry_type="point_xy",
        components=("x", "y"),
        component_units=("px", "px"),
        pixel_convention="pixel_center",
        row_identity=second_identity,
        reference_frame_authority=first.reference_frame_authority,
        source_camera_overlay_status=CANONICAL_OVERLAY_DIRECT,
    )
    with pytest.raises(CanonicalCoordinatePublicationError) as drift_exc:
        stamp_bound_canonical_coordinate_descriptors((first, second))
    assert "coordinate_sibling_identity_drift" in {
        issue.code for issue in drift_exc.value.issues
    }
    assert first_node.attrs == {}
    assert second_node.attrs == {}

    class FailAttrs(dict):
        failed = False

        def update(self, *args, **kwargs):
            if not self.failed:
                self.failed = True
                super().update({"partial": True})
                raise RuntimeError("injected")
            super().update(*args, **kwargs)

    same_path_second = _Array(
        np.asarray([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]),
        path="analysis/detect_runs/d1/other_centers_img_xy",
    )
    same_identity_binding = build_bound_canonical_coordinate_descriptor(
        same_path_second,
        profile_id="source_camera_image_px.top_left_y_down.v1",
        geometry_type="point_xy",
        components=("x", "y"),
        component_units=("px", "px"),
        pixel_convention="pixel_center",
        row_identity=first.row_identity,
        reference_frame_authority=first.reference_frame_authority,
        source_camera_overlay_status=CANONICAL_OVERLAY_DIRECT,
    )
    first_node.attrs = {"keep": 1}
    same_path_second.attrs = FailAttrs({"keep": 2})
    snapshots = (copy.deepcopy(first_node.attrs), {"keep": 2})
    with pytest.raises(CanonicalCoordinatePublicationError):
        stamp_bound_canonical_coordinate_descriptors(
            (first, same_identity_binding)
        )
    assert first_node.attrs == snapshots[0]
    assert same_path_second.attrs == snapshots[1]


def test_external_lineage_is_exact_persisted_evidence_and_revalidated() -> None:
    positions, _, direct = _direct_binding()
    lineage_node = _Group(path="analysis/detect_runs/d1")
    lineage = stamp_and_bind_persisted_coordinate_record(
        lineage_node,
        {
            "schema_id": "palette.coordinate_source_lineage",
            "schema_version": 1,
            "source_array_ref": "/raw_video/images_full",
        },
        attr_name="coordinate_source_lineage",
    )
    binding = build_bound_canonical_coordinate_descriptor(
        positions,
        profile_id="source_camera_image_px.top_left_y_down.v1",
        geometry_type="point_xy",
        components=("x", "y"),
        component_units=("px", "px"),
        pixel_convention="pixel_center",
        row_identity=direct.row_identity,
        reference_frame_authority=direct.reference_frame_authority,
        source_camera_overlay_status=CANONICAL_OVERLAY_DIRECT,
        lineage_records=(lineage,),
    )
    assert [item.record_ref for item in binding.descriptor.lineage_refs] == [
        "/analysis/coordinate_frames/source_camera/camera-a/pixel_center@pixel_frame_authority",
        "/analysis/detect_runs/d1@coordinate_source_lineage",
    ]

    lineage_node.attrs["coordinate_source_lineage"] = {
        "schema_id": "palette.coordinate_source_lineage",
        "schema_version": 2,
        "source_array_ref": "/raw_video/images_full",
    }
    with pytest.raises(CanonicalCoordinatePublicationError) as exc_info:
        stamp_bound_canonical_coordinate_descriptor(binding)
    assert "coordinate_lineage_unverified" in {
        issue.code for issue in exc_info.value.issues
    }
    assert positions.attrs == {}


def test_same_paths_from_different_archives_cannot_be_composed() -> None:
    from tests.unit.fisheye.test_directed_transform_chain import _world

    first_archive = object()
    second_archive = object()
    values = np.asarray([11, 12], dtype=np.uint64)
    rowset = _Group(
        path="analysis/detect_runs/d1",
        archive_token=first_archive,
    )
    key = _Array(
        values,
        path="analysis/detect_runs/d1/instance_key",
        archive_token=first_archive,
    )
    identity = stamp_and_bind_row_identity_contract(
        rowset,
        key,
        contract=build_row_identity_contract(
            domain=OBSERVATION_INSTANCE_DOMAIN,
            values=values,
        ),
    )
    world = _world(convention="pixel_center", archive_token=second_archive)
    positions = _Array(
        np.zeros((2, 2), dtype=np.float64),
        path="analysis/detect_runs/d1/centers_img_xy",
        archive_token=second_archive,
    )

    with pytest.raises(CanonicalCoordinatePublicationError) as exc_info:
        build_bound_canonical_coordinate_descriptor(
            positions,
            profile_id="source_camera_image_px.top_left_y_down.v1",
            geometry_type="point_xy",
            components=("x", "y"),
            component_units=("px", "px"),
            pixel_convention="pixel_center",
            row_identity=identity,
            reference_frame_authority=world["camera_frame"],
            source_camera_overlay_status=CANONICAL_OVERLAY_DIRECT,
        )
    assert "coordinate_archive_mismatch" in {
        issue.code for issue in exc_info.value.issues
    }


def test_batch_rejects_same_paths_and_digests_from_different_archives() -> None:
    first_node, _, first = _direct_binding(archive_token=object())
    second_node, _, second = _direct_binding(archive_token=object())
    assert first.row_identity.record_ref == second.row_identity.record_ref
    assert first.row_identity.record_sha256 == second.row_identity.record_sha256

    with pytest.raises(CanonicalCoordinatePublicationError) as exc_info:
        stamp_bound_canonical_coordinate_descriptors((first, second))
    assert {
        "coordinate_sibling_rowset_drift",
        "coordinate_sibling_archive_drift",
    } & {issue.code for issue in exc_info.value.issues}
    assert first_node.attrs == {}
    assert second_node.attrs == {}


def test_nonthrowing_corrupt_attrs_update_is_reloaded_and_rolled_back() -> None:
    first_node, _, first = _direct_binding()
    second_node = _Array(
        np.asarray([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]),
        path="analysis/detect_runs/d1/other_centers_img_xy",
    )
    second = build_bound_canonical_coordinate_descriptor(
        second_node,
        profile_id="source_camera_image_px.top_left_y_down.v1",
        geometry_type="point_xy",
        components=("x", "y"),
        component_units=("px", "px"),
        pixel_convention="pixel_center",
        row_identity=first.row_identity,
        reference_frame_authority=first.reference_frame_authority,
        source_camera_overlay_status=CANONICAL_OVERLAY_DIRECT,
    )

    class CorruptOnceAttrs(dict):
        corrupt = True

        def update(self, other=(), /, **kwargs):
            incoming = copy.deepcopy(dict(other, **kwargs))
            if self.corrupt and "coordinate_descriptor" in incoming:
                self.corrupt = False
                incoming["coordinate_descriptor"]["profile_id"] = "silently_corrupted"
            super().update(incoming)

    first_node.attrs = {"keep": 1}
    second_node.attrs = CorruptOnceAttrs({"keep": 2})
    with pytest.raises(CanonicalCoordinatePublicationError):
        stamp_bound_canonical_coordinate_descriptors((first, second))
    assert first_node.attrs == {"keep": 1}
    assert second_node.attrs == {"keep": 2}


def test_publication_refuses_occupied_v1_and_explicit_migration_fails_closed() -> None:
    node, _, binding = _direct_binding()
    node.attrs = {
        "unrelated_keep_me": {"nested": [1, 2]},
        "coordinate_descriptor": {
            "schema_id": "palette.coordinate_descriptor",
            "schema_version": 1,
        },
        "coordinate_descriptor_sha256": "f" * 64,
    }
    before = copy.deepcopy(node.attrs)
    with pytest.raises(CanonicalCoordinatePublicationError) as exc_info:
        stamp_bound_canonical_coordinate_descriptor(binding)
    assert "coordinate_v1_migration_required" in {
        issue.code for issue in exc_info.value.issues
    }
    with pytest.raises(CanonicalCoordinatePublicationError) as migration_error:
        migrate_historical_coordinate_descriptor_v1_to_v2(node)
    assert "coordinate_v1_migration_lineage_unproven" in {
        issue.code for issue in migration_error.value.issues
    }
    assert node.attrs == before


def test_publication_detects_unrelated_attr_clearing_and_rolls_back() -> None:
    node, _, binding = _direct_binding()

    class ClearingOnceAttrs(dict):
        armed = True

        def update(self, other=(), /, **kwargs):
            incoming = copy.deepcopy(dict(other, **kwargs))
            if self.armed and "coordinate_descriptor" in incoming:
                self.armed = False
                self.pop("unrelated_keep_me", None)
            super().update(incoming)

    node.attrs = ClearingOnceAttrs(
        {"unrelated_keep_me": {"nested": [1, 2]}}
    )
    before = copy.deepcopy(dict(node.attrs))
    with pytest.raises(CanonicalCoordinatePublicationError):
        stamp_bound_canonical_coordinate_descriptor(binding)
    assert dict(node.attrs) == before


def test_loader_rejects_duck_typed_noop_row_identity() -> None:
    positions, _, binding = _direct_binding()
    stamp_bound_canonical_coordinate_descriptor(binding)

    class DuckIdentity:
        contract = binding.row_identity.contract
        record_ref = binding.row_identity.record_ref
        record_sha256 = binding.row_identity.record_sha256
        rowset_path = binding.row_identity.rowset_path
        key_array_path = binding.row_identity.key_array_path
        leading_dimension = binding.row_identity.leading_dimension
        archive_identity = binding.row_identity.archive_identity

        def assert_verified(self) -> None:
            return None

    with pytest.raises(CanonicalCoordinatePublicationError) as exc_info:
        load_bound_canonical_coordinate_descriptor(
            positions,
            row_identity=DuckIdentity(),  # type: ignore[arg-type]
            reference_frame_authority=binding.reference_frame_authority,
        )
    assert "coordinate_row_identity_unverified" in {
        issue.code for issue in exc_info.value.issues
    }


def test_physical_frame_profile_requires_exact_origin_axes_units_and_allowlist() -> None:
    from tests.unit.fisheye.test_directed_transform_chain import _world

    token = object()
    world = _world(archive_token=token)
    selected_node = _Group(path="analysis/calibration/selected", archive_token=token)
    selected = stamp_selected_camera_frame_evidence(
        selected_node,
        source_camera=world["camera_evidence"],
    )
    source_frame = world["camera_frame"]
    frame_node = _Group(
        path="analysis/coordinate_frames/camera_mm",
        archive_token=token,
    )
    physical = stamp_physical_frame_calibration_record(
        frame_node,
        record=build_physical_frame_calibration_record(
            frame_id="camera_mm",
            source_camera_pixels=source_frame,
            selected_camera_evidence=selected,
        ),
        expected_record_ref=(
            "/analysis/coordinate_frames/camera_mm"
            "@physical_frame_calibration"
        ),
        source_camera_pixels=source_frame,
        selected_camera_evidence=selected,
    )
    rowset, key, identity = _identity(
        "analysis/metrics/m1",
        archive_token=token,
    )
    del rowset, key
    positions = _Array(
        np.zeros((3, 2), dtype=np.float64),
        path="analysis/metrics/m1/positions_mm",
        archive_token=token,
    )
    accepted = build_bound_canonical_coordinate_descriptor(
        positions,
        profile_id="physical_mm.source_camera_y_down.v1",
        geometry_type="point_xy",
        components=("x", "y"),
        component_units=("mm", "mm"),
        pixel_convention="not_applicable",
        row_identity=identity,
        source_camera_overlay_status=CANONICAL_OVERLAY_NOT_SUITABLE,
        frame_record=physical,
    )
    assert accepted.descriptor.origin == physical.origin == "physical_frame_origin"
    assert accepted.descriptor.positive_directions.y == physical.positive_y == "down"

    with pytest.raises(CanonicalCoordinatePublicationError) as exc_info:
        build_bound_canonical_coordinate_descriptor(
            positions,
            profile_id="physical_mm.cartesian_y_up.v1",
            geometry_type="point_xy",
            components=("x", "y"),
            component_units=("mm", "mm"),
            pixel_convention="not_applicable",
            row_identity=identity,
            source_camera_overlay_status=CANONICAL_OVERLAY_NOT_SUITABLE,
            frame_record=physical,
        )
    assert {
        "coordinate_profile_publication_unavailable",
        "coordinate_frame_profile_incompatible",
        "coordinate_frame_axes_mismatch",
        "coordinate_frame_origin_mismatch",
        "coordinate_reference_units_mismatch",
    } & {issue.code for issue in exc_info.value.issues}


def test_body_frame_unit_vector_profile_uses_spatial_frame_basis() -> None:
    from tests.unit.fisheye.test_coordinate_frame_record import (
        _stamp_body,
        body_inputs as body_inputs_fixture,
        physical_inputs as physical_inputs_fixture,
    )

    physical_inputs = physical_inputs_fixture.__wrapped__()
    body_inputs = body_inputs_fixture.__wrapped__(physical_inputs)
    body_frame = _stamp_body(body_inputs)
    vectors = _Array(
        np.asarray([[1.0, 0.0], [1.0, 0.0], [np.nan, np.nan]], dtype=np.float32),
        path=f"{body_inputs['rowset'].path}/heading_body_xy",
        archive_token=body_inputs["token"],
    )
    binding = build_bound_canonical_coordinate_descriptor(
        vectors,
        profile_id="fish_anatomical_body_frame.unit_vector.v1",
        geometry_type="vector_xy",
        components=("x", "y"),
        component_units=("unitless", "unitless"),
        pixel_convention="not_applicable",
        row_identity=body_inputs["identity"],
        source_camera_overlay_status=CANONICAL_OVERLAY_NOT_SUITABLE,
        frame_record=body_frame,
    )
    assert body_frame.coordinate_units == "px"
    assert binding.descriptor.component_units == ("unitless", "unitless")
    assert binding.descriptor.frame_record.record_ref == body_frame.record_ref
