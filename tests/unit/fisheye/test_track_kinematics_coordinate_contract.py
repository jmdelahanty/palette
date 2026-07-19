from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from fisheye.analysis import track_kinematics as mod
from fisheye.shared.coordinate_descriptor import COORDINATE_DESCRIPTOR_ATTR
from fisheye.shared.coordinate_identity import (
    TRACK_SAMPLE_KEY_MODE,
    TRACK_SAMPLE_SOURCE_INSTANCE_KEY_DTYPE,
    TRACK_SAMPLE_DOMAIN,
    load_bound_row_identity_contract,
    load_bound_track_sample_time_lineage,
)
from fisheye.shared.observation_coordinate_publication import (
    publish_crop_observation_geometry,
)
from tests.unit.fisheye.test_directed_transform_chain import (
    FakeArray,
    FakeGroup,
    _world,
)
from tests.unit.fisheye.test_track_coordinate_publication import (
    _physical,
    _source,
)
from tests.unit.fisheye.test_observation_coordinate_publication import (
    _crop_copy,
    _published_detection,
)


class _WritableGroup(FakeGroup):
    def create_group(self, name: str) -> "_WritableGroup":
        child = _WritableGroup(
            path=f"{self.path}/{name}",
            archive_token=self._coordinate_archive_token,
        )
        self.children[name] = child
        return child

    def create_array(
        self,
        name: str,
        *,
        data: Any,
        chunks: tuple[int, ...] | None = None,
        overwrite: bool = False,
    ) -> FakeArray:
        if not overwrite and name in self.children:
            raise ValueError(f"{name!r} already exists")
        child = FakeArray(
            data,
            path=f"{self.path}/{name}",
            archive_token=self._coordinate_archive_token,
            chunks=chunks,
        )
        self.children[name] = child
        return child

    def __getitem__(self, name: str) -> Any:
        node: Any = self
        for part in name.split("/"):
            node = node.children[part]
        return node

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        try:
            self[name]
        except (KeyError, AttributeError):
            return False
        return True

    def array_keys(self) -> list[str]:
        return [
            name
            for name, child in self.children.items()
            if isinstance(child, FakeArray)
        ]

    def group_keys(self) -> list[str]:
        return [
            name
            for name, child in self.children.items()
            if isinstance(child, _WritableGroup)
        ]


def _build_from_source(source, temporal, *, pixel_to_mm=None):
    values = np.asarray(source.coordinate_node[:])
    source_rows = np.asarray([0, 1], dtype=np.int64)
    frames = mod.resolve_source_acquisition_frame_indices(
        temporal,
        source_rows,
    )
    return mod.build_track_datasets(
        track_ids=np.asarray([7, 7], dtype=np.int64),
        frames=frames,
        positions_px=values,
        headings_deg=np.zeros(2, dtype=np.float32),
        keypoint_success=np.ones(2, dtype=bool),
        detection_source=None,
        fps=1.0,
        smooth_seconds=1.0,
        pixel_to_mm=pixel_to_mm,
        source_row_index=source_rows,
        source_temporal_authority=temporal,
    )


def test_offline_loader_uses_exact_persisted_crop_center_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = _world(convention="continuous", archive_token=object())
    detection = _published_detection(world)
    nodes = _crop_copy(world, detection)
    rowset, key, source_rows, source_frames, bbox_norm, bbox_img, centers = nodes
    crop = publish_crop_observation_geometry(
        *nodes,
        source_geometry=detection,
    )
    for name, node in (
        ("instance_key", key),
        ("detection_indices", source_rows),
        ("source_acquisition_frame_index", source_frames),
        ("bbox_norm_coords", bbox_norm),
        ("bbox_img_xyxy", bbox_img),
        ("centers_img_xy", centers),
    ):
        rowset.children[name] = node
    monkeypatch.setattr(
        mod,
        "load_persisted_source_camera_position_surface",
        lambda _root, path: crop.position_surface
        if path == "crop_runs/c1"
        else None,
    )

    loaded = mod.load_canonical_offline_position_source(
        FakeGroup(path="", archive_token=world["archive_token"]),
        rowset,
        crop_run_name="c1",
    )

    assert loaded.position_surface is crop.position_surface
    assert loaded.geometry_path == "crop_runs/c1/centers_img_xy"
    assert loaded.kind == "canonical_crop_rows_source_camera_centers"
    np.testing.assert_array_equal(loaded.positions_px, centers[:])
    np.testing.assert_array_equal(loaded.frame_indices, source_frames[:])
    np.testing.assert_array_equal(loaded.instance_key, key[:])


def test_offline_loader_rejects_same_path_from_a_different_archive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = _world(convention="continuous", archive_token=object())
    detection = _published_detection(world)
    nodes = _crop_copy(world, detection)
    rowset = nodes[0]
    crop = publish_crop_observation_geometry(
        *nodes,
        source_geometry=detection,
    )
    monkeypatch.setattr(
        mod,
        "load_persisted_source_camera_position_surface",
        lambda _root, _path: crop.position_surface,
    )

    with pytest.raises(ValueError, match="exact selected crop rowset"):
        mod.load_canonical_offline_position_source(
            FakeGroup(path="", archive_token=object()),
            rowset,
            crop_run_name="c1",
        )


def test_builder_derives_observation_lineage_but_track_key_is_primary() -> None:
    world = _world(convention="pixel_center", archive_token=object())
    _, _, source, temporal = _source(world)
    tracks, _ = _build_from_source(source, temporal)

    track = tracks[7]
    assert track["track_sample_key"].dtype == np.dtype("<i8")
    np.testing.assert_array_equal(
        track["track_sample_key"],
        np.asarray([[7, 0], [7, 1]], dtype=np.int64),
    )
    assert track["source_instance_key"].dtype == (
        TRACK_SAMPLE_SOURCE_INSTANCE_KEY_DTYPE
    )
    np.testing.assert_array_equal(
        track["source_instance_key"]["instance_key"],
        np.asarray([301, 302], dtype=np.uint64),
    )
    np.testing.assert_array_equal(
        track["source_instance_key"]["valid"],
        np.asarray([True, True]),
    )


def test_builder_rejects_caller_frames_that_disagree_with_source_mapping() -> None:
    world = _world(convention="pixel_center", archive_token=object())
    _, _, source, temporal = _source(world)
    with pytest.raises(ValueError, match="must exactly equal"):
        mod.build_track_datasets(
            track_ids=np.asarray([7, 7], dtype=np.int64),
            frames=np.asarray([10, 11], dtype=np.int64),
            positions_px=np.asarray(source.coordinate_node[:]),
            headings_deg=np.zeros(2, dtype=np.float32),
            keypoint_success=np.ones(2, dtype=bool),
            detection_source=None,
            fps=1.0,
            smooth_seconds=1.0,
            pixel_to_mm=None,
            source_row_index=np.asarray([0, 1], dtype=np.int64),
            source_temporal_authority=temporal,
        )


def test_writer_publishes_v2_track_coordinates_and_omits_unframed_mm() -> None:
    world = _world(convention="pixel_center", archive_token=object())
    _, _, source, temporal = _source(world)
    tracks, summaries = _build_from_source(source, temporal)
    run = _WritableGroup(
        path="analysis/track_kinematics_runs/offline/tk_1",
        archive_token=world["archive_token"],
    )

    ordered = mod.save_track_kinematics_tracks(
        run,
        tracks,
        summaries,
        source_temporal_authority=temporal,
        positions_px_source=source,
    )

    assert ordered == [7]
    track = run["tracks/id_7"]
    time_lineage = load_bound_track_sample_time_lineage(
        track,
        track["track_sample_key"],
        track["source_row_index"],
        track["source_acquisition_frame_index"],
        track["source_frame_interpolation"],
        track["source_instance_key"],
        source_temporal_authority=temporal,
    )
    identity = load_bound_row_identity_contract(
        track,
        track["track_sample_key"],
        track_time_lineage=time_lineage,
    )
    assert identity.contract.domain == TRACK_SAMPLE_DOMAIN
    assert identity.contract.mode == TRACK_SAMPLE_KEY_MODE
    descriptor = track["positions_px"].attrs[COORDINATE_DESCRIPTOR_ATTR]
    assert descriptor["schema_version"] == 2
    assert descriptor["row_identity"]["record_ref"] == identity.record_ref
    assert "positions_mm" not in track
    assert not any(name.endswith("_mm") for name in track.array_keys())
    assert track.attrs["physical_outputs_status"] == (
        "omitted_no_compatible_typed_physical_frame"
    )
    assert not any("_mm" in name for name in track.attrs["summary"])


def test_writer_publishes_positions_mm_only_with_exact_physical_frame() -> None:
    world = _world(convention="pixel_center", archive_token=object())
    _, _, source, temporal = _source(world)
    physical = _physical(world)
    tracks, summaries = _build_from_source(
        source,
        temporal,
        pixel_to_mm=physical.record.mm_per_pixel,
    )
    run = _WritableGroup(
        path="analysis/track_kinematics_runs/offline/tk_physical",
        archive_token=world["archive_token"],
    )

    mod.save_track_kinematics_tracks(
        run,
        tracks,
        summaries,
        source_temporal_authority=temporal,
        positions_px_source=source,
        physical_frame=physical,
    )

    track = run["tracks/id_7"]
    assert track["positions_mm"].attrs[COORDINATE_DESCRIPTOR_ATTR][
        "profile_id"
    ] == "physical_mm.source_camera_y_down.v1"
    assert track.attrs["physical_outputs_status"] == (
        "available_typed_source_camera_frame"
    )
    assert "speed_smoothed_mm" in track


def test_writer_preserves_float64_source_position_payload_exactly() -> None:
    world = _world(convention="pixel_center", archive_token=object())
    _, _, source, temporal = _source(world, dtype=np.float64)
    tracks, summaries = _build_from_source(source, temporal)
    assert tracks[7]["positions_px"].dtype == np.dtype("<f8")
    run = _WritableGroup(
        path="analysis/track_kinematics_runs/offline/tk_float64",
        archive_token=world["archive_token"],
    )

    mod.save_track_kinematics_tracks(
        run,
        tracks,
        summaries,
        source_temporal_authority=temporal,
        positions_px_source=source,
    )

    stored = np.asarray(run["tracks/id_7/positions_px"][:])
    assert stored.dtype == np.dtype("<f8")
    np.testing.assert_array_equal(stored, source.coordinate_node[:])


def _canonical_crop_position_surface(world):
    detection = _published_detection(world)
    nodes = _crop_copy(world, detection)
    crop = publish_crop_observation_geometry(
        *nodes,
        source_geometry=detection,
    )
    return crop.position_surface


def test_deferred_track_stage_binds_only_at_authoritative_final_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = _world(convention="continuous", archive_token=object())
    surface = _canonical_crop_position_surface(world)
    tracks, summaries = _build_from_source(
        surface.coordinates,
        surface.temporal_authority,
    )
    run_name = "tk_staged"
    run = _WritableGroup(
        path=f"analysis/track_kinematics_runs/offline/{run_name}",
        archive_token=world["archive_token"],
    )
    mod.save_track_kinematics_tracks(
        run,
        tracks,
        summaries,
        source_temporal_authority=surface.temporal_authority,
        positions_px_source=surface.coordinates,
        defer_coordinate_binding=True,
        staging_keypoint_run="kp_1",
        staging_run_name=run_name,
    )
    track = run["tracks/id_7"]
    assert run.attrs[mod.TRACK_KINEMATICS_COORDINATE_BINDING_STATUS_ATTR] == (
        mod.TRACK_KINEMATICS_UNBOUND_STAGE_STATUS
    )
    assert COORDINATE_DESCRIPTOR_ATTR not in track["positions_px"].attrs
    assert not any(name.startswith("row_identity_") for name in track.attrs)
    assert not any(
        name.startswith("track_sample_time_lineage") for name in track.attrs
    )

    monkeypatch.setattr(
        mod,
        "load_persisted_source_camera_position_surface",
        lambda _root, path: surface if path == "crop_runs/c1" else None,
    )
    run.attrs["palette_run_completion_status"] = "running"
    run.attrs[mod.TRACK_KINEMATICS_COORDINATE_BINDING_STATUS_ATTR] = (
        mod.TRACK_KINEMATICS_PUBLISHING_BINDING_STATUS
    )
    root = FakeGroup(path="", archive_token=world["archive_token"])

    bound = mod.bind_staged_offline_track_kinematics_run(
        root,
        run,
        expected_keypoint_run="kp_1",
        expected_run_name=run_name,
    )

    assert bound["valid"] is True
    assert run.attrs[mod.TRACK_KINEMATICS_COORDINATE_BINDING_STATUS_ATTR] == (
        mod.TRACK_KINEMATICS_BOUND_CANONICAL_STATUS
    )
    assert track["positions_px"].attrs[COORDINATE_DESCRIPTOR_ATTR][
        "space_id"
    ] == "source_camera_image_px"
    assert mod.validate_bound_offline_track_kinematics_run(
        root,
        run,
        expected_keypoint_run="kp_1",
        expected_run_name=run_name,
        require_complete=False,
    )["valid"] is True
    run.attrs["palette_run_completion_status"] = "complete"
    assert mod.validate_bound_offline_track_kinematics_run(
        root,
        run,
        expected_keypoint_run="kp_1",
        expected_run_name=run_name,
        require_complete=True,
    )["valid"] is True


def test_deferred_track_binding_rejects_source_drift_without_partial_claims(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = _world(convention="continuous", archive_token=object())
    surface = _canonical_crop_position_surface(world)
    tracks, summaries = _build_from_source(
        surface.coordinates,
        surface.temporal_authority,
    )
    run_name = "tk_source_drift"
    run = _WritableGroup(
        path=f"analysis/track_kinematics_runs/offline/{run_name}",
        archive_token=world["archive_token"],
    )
    mod.save_track_kinematics_tracks(
        run,
        tracks,
        summaries,
        source_temporal_authority=surface.temporal_authority,
        positions_px_source=surface.coordinates,
        defer_coordinate_binding=True,
        staging_keypoint_run="kp_1",
        staging_run_name=run_name,
    )
    monkeypatch.setattr(
        mod,
        "load_persisted_source_camera_position_surface",
        lambda _root, _path: surface,
    )
    run.attrs["palette_run_completion_status"] = "running"
    run.attrs[mod.TRACK_KINEMATICS_COORDINATE_BINDING_STATUS_ATTR] = (
        mod.TRACK_KINEMATICS_PUBLISHING_BINDING_STATUS
    )
    surface.coordinates.coordinate_node.data[0, 0] += 1.0
    track = run["tracks/id_7"]

    with pytest.raises(ValueError, match="changed after numerical staging"):
        mod.bind_staged_offline_track_kinematics_run(
            FakeGroup(path="", archive_token=world["archive_token"]),
            run,
            expected_keypoint_run="kp_1",
            expected_run_name=run_name,
        )

    assert run.attrs[mod.TRACK_KINEMATICS_COORDINATE_BINDING_STATUS_ATTR] == (
        mod.TRACK_KINEMATICS_PUBLISHING_BINDING_STATUS
    )
    assert COORDINATE_DESCRIPTOR_ATTR not in track["positions_px"].attrs
    assert not any(name.startswith("row_identity_") for name in track.attrs)


def test_refined_source_path_and_digest_are_normalized_in_provenance() -> None:
    world = _world(convention="pixel_center", archive_token=object())
    _, _, source, _ = _source(world)
    descriptor = source.descriptor
    source_path = "refined_online_runs/refined_a/interpolated/positions_px"

    attrs = mod._track_kinematics_contract_attrs(
        run_type="online",
        method="track_kinematics_online_refined",
        parameters={"coordinate_space": descriptor.space_id},
        inputs={
            "refined_online_run": "refined_a",
            "stimulus_run": "stimulus_a",
            "chaser_index": 0,
            "positions_px_source_path": source_path,
            "positions_px_coordinate_descriptor_sha256": descriptor.digest(),
        },
    )

    assert attrs["source_refs"]["source_positions_px_path"] == source_path
    assert attrs["source_refs"][
        "source_positions_px_coordinate_descriptor_sha256"
    ] == descriptor.digest()
