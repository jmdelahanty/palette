from __future__ import annotations

import numpy as np
import pytest

from fisheye.shared.canonical_coordinate_publication import (
    build_bound_canonical_coordinate_descriptor,
    load_bound_canonical_coordinate_descriptor,
    stamp_bound_canonical_coordinate_descriptor,
)
from fisheye.shared.coordinate_descriptor import (
    CANONICAL_OVERLAY_DIRECT,
    COORDINATE_DESCRIPTOR_ATTR,
)
from fisheye.shared.coordinate_frame_record import (
    build_physical_frame_calibration_record,
    stamp_physical_frame_calibration_record,
    stamp_selected_camera_frame_evidence,
)
from fisheye.shared.coordinate_identity import (
    OBSERVATION_INSTANCE_DOMAIN,
    TRACK_SAMPLE_INTERPOLATION_DTYPE,
    TRACK_SAMPLE_DOMAIN,
    build_row_identity_contract,
    build_track_sample_key,
    derive_track_source_instance_values,
    stamp_and_bind_row_identity_contract,
    stamp_source_row_temporal_authority,
    stamp_track_sample_time_lineage,
)
from fisheye.shared.track_coordinate_publication import (
    TRACK_POSITION_DERIVATION_ATTR,
    TrackCoordinatePublicationError,
    load_track_position_coordinates,
    publish_track_position_coordinates,
)
from tests.unit.fisheye.test_directed_transform_chain import (
    FakeArray,
    FakeGroup,
    _world,
)


def _source(world, *, dtype: np.dtype | type = np.float32):
    token = world["archive_token"]
    rowset = FakeGroup(
        path="analysis/refined_online_runs/refined_1/interpolated",
        archive_token=token,
    )
    keys = FakeArray(
        np.asarray([301, 302], dtype=np.uint64),
        path=f"{rowset.path}/instance_key",
        archive_token=token,
    )
    identity = stamp_and_bind_row_identity_contract(
        rowset,
        keys,
        contract=build_row_identity_contract(
            domain=OBSERVATION_INSTANCE_DOMAIN,
            values=keys.data,
        ),
    )
    frames = FakeArray(
        np.asarray([0, 1], dtype=np.int64),
        path=f"{rowset.path}/source_acquisition_frame_index",
        archive_token=token,
    )
    temporal = stamp_source_row_temporal_authority(
        rowset,
        frames,
        source_row_identity=identity,
        acquisition_frame=world["acquisition_frame"],
    )
    values = np.asarray([[11.0, 12.0], [21.0, 22.0]], dtype=dtype)
    positions = FakeArray(
        values,
        path=f"{rowset.path}/positions_px",
        archive_token=token,
    )
    binding = build_bound_canonical_coordinate_descriptor(
        positions,
        profile_id="source_camera_image_px.top_left_y_down.v1",
        geometry_type="points_xy",
        components=("x", "y"),
        component_units=("px", "px"),
        pixel_convention="pixel_center",
        row_identity=identity,
        reference_frame_authority=world["camera_frame"],
        source_camera_overlay_status=CANONICAL_OVERLAY_DIRECT,
    )
    stamp_bound_canonical_coordinate_descriptor(binding)
    return rowset, positions, binding, temporal


def _track(world, source_temporal, *, source_rows=(0, 1), output_values=None):
    token = world["archive_token"]
    group = FakeGroup(
        path=(
            "analysis/track_kinematics_runs/offline/tk_1/tracks/id_7"
        ),
        archive_token=token,
    )
    source_rows_values = np.asarray(source_rows, dtype=np.int64)
    source_frames = np.asarray(source_rows_values, dtype=np.int64)
    key_values = build_track_sample_key(
        np.full(source_frames.shape, 7, dtype=np.int64),
        source_frames,
    )
    key = FakeArray(
        key_values,
        path=f"{group.path}/track_sample_key",
        archive_token=token,
    )
    source_row_index = FakeArray(
        source_rows_values,
        path=f"{group.path}/source_row_index",
        archive_token=token,
    )
    source_frame = FakeArray(
        source_frames,
        path=f"{group.path}/source_acquisition_frame_index",
        archive_token=token,
    )
    interpolation_values = np.zeros(
        source_frames.shape,
        dtype=TRACK_SAMPLE_INTERPOLATION_DTYPE,
    )
    interpolation_values["left_source_frame_index"] = source_frames
    interpolation_values["right_source_frame_index"] = source_frames
    interpolation = FakeArray(
        interpolation_values,
        path=f"{group.path}/source_frame_interpolation",
        archive_token=token,
    )
    source_instances = FakeArray(
        derive_track_source_instance_values(
            source_temporal,
            source_rows_values,
        ),
        path=f"{group.path}/source_instance_key",
        archive_token=token,
    )
    time_lineage = stamp_track_sample_time_lineage(
        group,
        key,
        source_row_index,
        source_frame,
        interpolation,
        source_instances,
        source_temporal_authority=source_temporal,
    )
    identity = stamp_and_bind_row_identity_contract(
        group,
        key,
        contract=build_row_identity_contract(
            domain=TRACK_SAMPLE_DOMAIN,
            values=key_values,
            track_time_lineage=time_lineage,
        ),
        track_time_lineage=time_lineage,
    )
    values = (
        np.asarray([[11.0, 12.0], [21.0, 22.0]], dtype=np.float32)
        if output_values is None
        else np.asarray(output_values)
    )
    positions = FakeArray(
        values,
        path=f"{group.path}/positions_px",
        archive_token=token,
    )
    return group, source_row_index, identity, positions


def _physical(world):
    token = world["archive_token"]
    selected_node = FakeGroup(
        path="analysis/coordinate_frames/selected_camera/camera-a",
        archive_token=token,
    )
    selected = stamp_selected_camera_frame_evidence(
        selected_node,
        source_camera=world["camera_evidence"],
    )
    frame_node = FakeGroup(
        path="analysis/coordinate_frames/physical/camera-a",
        archive_token=token,
    )
    return stamp_physical_frame_calibration_record(
        frame_node,
        build_physical_frame_calibration_record(
            frame_id="camera_a_mm",
            source_camera_pixels=world["camera_frame"],
            selected_camera_evidence=selected,
        ),
        expected_record_ref=(
            "/analysis/coordinate_frames/physical/camera-a"
            "@physical_frame_calibration"
        ),
        source_camera_pixels=world["camera_frame"],
        selected_camera_evidence=selected,
    )


def test_track_positions_publish_v2_from_exact_selected_source() -> None:
    world = _world(convention="pixel_center", archive_token=object())
    _, _, source, temporal = _source(world)
    group, source_rows, identity, positions = _track(world, temporal)

    result = publish_track_position_coordinates(
        group,
        positions,
        source_rows,
        track_row_identity=identity,
        source_positions=source,
        source_temporal_authority=temporal,
    )

    assert result.positions_px.descriptor.profile_id == source.descriptor.profile_id
    assert result.positions_px.row_identity.record_ref == identity.record_ref
    assert result.positions_mm is None
    assert positions.attrs[COORDINATE_DESCRIPTOR_ATTR]["schema_version"] == 2
    assert group.attrs[TRACK_POSITION_DERIVATION_ATTR]["operation"] == (
        "exact_subset_reorder_v1"
    )
    loaded = load_bound_canonical_coordinate_descriptor(
        positions,
        row_identity=identity,
        reference_frame_authority=world["camera_frame"],
        lineage_records=(result.derivation,),
    )
    assert loaded.descriptor == result.positions_px.descriptor
    reloaded = load_track_position_coordinates(
        group,
        positions,
        source_rows,
        track_row_identity=identity,
        source_positions=source,
        source_temporal_authority=temporal,
    )
    assert reloaded.positions_px.descriptor == result.positions_px.descriptor


def test_fresh_loader_rejects_source_payload_tampering() -> None:
    world = _world(convention="pixel_center", archive_token=object())
    _, source_node, source, temporal = _source(world)
    group, source_rows, identity, positions = _track(world, temporal)
    publish_track_position_coordinates(
        group,
        positions,
        source_rows,
        track_row_identity=identity,
        source_positions=source,
        source_temporal_authority=temporal,
    )

    source_node.data[0, 0] += np.float32(1.0)
    with pytest.raises(TrackCoordinatePublicationError, match="source subset"):
        load_track_position_coordinates(
            group,
            positions,
            source_rows,
            track_row_identity=identity,
            source_positions=source,
            source_temporal_authority=temporal,
        )


def test_track_positions_publish_physical_only_with_exact_typed_frame() -> None:
    world = _world(convention="pixel_center", archive_token=object())
    _, _, source, temporal = _source(world)
    group, source_rows, identity, positions = _track(world, temporal)
    physical = _physical(world)
    mm_values = positions.data * np.asarray(
        physical.record.mm_per_pixel,
        dtype=positions.dtype,
    )
    positions_mm = FakeArray(
        mm_values,
        path=f"{group.path}/positions_mm",
        archive_token=world["archive_token"],
    )

    result = publish_track_position_coordinates(
        group,
        positions,
        source_rows,
        track_row_identity=identity,
        source_positions=source,
        source_temporal_authority=temporal,
        positions_mm_node=positions_mm,
        physical_frame=physical,
    )

    assert result.positions_mm is not None
    assert (
        result.positions_mm.descriptor.profile_id
        == "physical_mm.source_camera_y_down.v1"
    )
    assert positions_mm.attrs[COORDINATE_DESCRIPTOR_ATTR]["schema_version"] == 2


def test_track_positions_reject_roi_image_mixing_before_publication() -> None:
    world = _world(convention="pixel_center", archive_token=object())
    _, _, source, temporal = _source(world)
    group, source_rows, identity, positions = _track(
        world,
        temporal,
        output_values=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
    )

    with pytest.raises(
        TrackCoordinatePublicationError,
        match="exact dtype-preserving subset/reorder",
    ):
        publish_track_position_coordinates(
            group,
            positions,
            source_rows,
            track_row_identity=identity,
            source_positions=source,
            source_temporal_authority=temporal,
        )
    assert TRACK_POSITION_DERIVATION_ATTR not in group.attrs
    assert COORDINATE_DESCRIPTOR_ATTR not in positions.attrs


def test_track_positions_reject_unframed_or_wrong_scale_mm() -> None:
    world = _world(convention="pixel_center", archive_token=object())
    _, _, source, temporal = _source(world)
    group, source_rows, identity, positions = _track(world, temporal)
    positions_mm = FakeArray(
        positions.data.copy(),
        path=f"{group.path}/positions_mm",
        archive_token=world["archive_token"],
    )
    with pytest.raises(TrackCoordinatePublicationError, match="present exactly"):
        publish_track_position_coordinates(
            group,
            positions,
            source_rows,
            track_row_identity=identity,
            source_positions=source,
            source_temporal_authority=temporal,
            positions_mm_node=positions_mm,
        )

    physical = _physical(world)
    with pytest.raises(TrackCoordinatePublicationError, match="does not equal"):
        publish_track_position_coordinates(
            group,
            positions,
            source_rows,
            track_row_identity=identity,
            source_positions=source,
            source_temporal_authority=temporal,
            positions_mm_node=positions_mm,
            physical_frame=physical,
        )


def test_track_publication_rolls_back_derivation_when_descriptor_stamp_fails() -> None:
    world = _world(convention="pixel_center", archive_token=object())
    _, _, source, temporal = _source(world)
    group, source_rows, identity, positions = _track(world, temporal)
    group.attrs["preexisting"] = {"version": 7}
    positions.attrs.update(
        {
            "preexisting": ["preserve", 7],
            COORDINATE_DESCRIPTOR_ATTR: {
                "schema_id": "palette.coordinate_descriptor",
                "schema_version": 1,
            },
        }
    )
    group_before = dict(group.attrs)
    positions_before = dict(positions.attrs)

    with pytest.raises(ValueError):
        publish_track_position_coordinates(
            group,
            positions,
            source_rows,
            track_row_identity=identity,
            source_positions=source,
            source_temporal_authority=temporal,
        )

    assert dict(group.attrs) == group_before
    assert dict(positions.attrs) == positions_before
    assert TRACK_POSITION_DERIVATION_ATTR not in group.attrs
