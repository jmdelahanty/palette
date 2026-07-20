from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pytest

from fisheye.analysis import track_kinematics as mod
from fisheye.shared.acquisition_publication_status import (
    ACQUISITION_AUTHORITY_PUBLISHED,
    MATERIALIZED_ACQUISITION_AUTHORITY_MODE,
    MATERIALIZED_ACQUISITION_PUBLISHED_REASON,
    stamp_acquisition_authority_publication_status,
)
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
from fisheye.shared.stimulus_physical_coordinate import (
    BoundStimulusPhysicalCoordinateAuthority,
    load_stimulus_physical_coordinate_authority,
    publish_stimulus_physical_coordinate_authority,
    require_bound_stimulus_physical_coordinate_authority,
)
from fisheye.shared.source_camera_physical_authority import (
    publish_source_camera_physical_authority,
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
    def __init__(self, *, path: str, archive_token: object) -> None:
        super().__init__(path=path, archive_token=archive_token)
        self.fresh_array_handle_names: set[str] = set()
        parts = path.split("/")
        if parts[:2] == ["analysis", "track_kinematics_runs"] and len(parts) == 4:
            self.attrs["stage_selector_eligible"] = False

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
            child = node.children[part]
            if (
                part in getattr(node, "fresh_array_handle_names", ())
                and isinstance(child, FakeArray)
            ):
                child = FakeArray(
                    child.data,
                    path=child.path,
                    attrs=child.attrs,
                    archive_token=child._coordinate_archive_token,
                    chunks=child.chunks,
                    shards=child.shards,
                )
            node = child
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
    assert mod._offline_position_source_inputs(loaded) == {
        "position_source_path": "crop_runs/c1/centers_img_xy",
        "position_source_rowset_path": "crop_runs/c1",
        "position_source_kind": "canonical_crop_rows_source_camera_centers",
    }
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
    assert not mod._track_physical_array_nodes(track)
    assert "total_distance_mm" not in run.attrs
    assert not any(name.endswith("_mm") for name in run.attrs)
    assert not any(
        name.endswith("_mm")
        for summary in run.attrs["summary"]
        for name in summary
    )
    assert not any(
        name.endswith("_mm")
        for entry in run.attrs["track_manifest"]
        for name in entry
    )
    for _, node in mod._iter_track_array_nodes(track):
        descriptor = node.attrs.get(COORDINATE_DESCRIPTOR_ATTR)
        assert not (
            isinstance(descriptor, dict)
            and str(descriptor.get("profile_id", "")).startswith("physical_mm.")
        )


def test_writer_rejects_detached_physical_frame_before_mutation() -> None:
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

    with pytest.raises(
        ValueError,
        match="Detached physical_frame values cannot authorize",
    ):
        mod.save_track_kinematics_tracks(
            run,
            tracks,
            summaries,
            source_temporal_authority=temporal,
            positions_px_source=source,
            physical_frame=physical,
        )

    assert "tracks" not in run


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


def _selected_stimulus_physical_authority(world):
    token = world["archive_token"]
    root = world["root"]
    root.get = root.children.get
    stamp_acquisition_authority_publication_status(
        root,
        root["raw_video"],
        status=ACQUISITION_AUTHORITY_PUBLISHED,
        reason_code=MATERIALIZED_ACQUISITION_PUBLISHED_REASON,
        authority_mode=MATERIALIZED_ACQUISITION_AUTHORITY_MODE,
        authority_path="analysis/acquisition_camera_frames/camera-a",
    )
    analysis = root["analysis"]
    analysis.get = analysis.children.get
    analysis["stimulus_runs"].get = analysis["stimulus_runs"].children.get
    coordinate_frames = FakeGroup(
        path="analysis/coordinate_frames",
        archive_token=token,
    )
    source_camera = FakeGroup(
        path="analysis/coordinate_frames/source_camera",
        archive_token=token,
    )
    camera = FakeGroup(
        path="analysis/coordinate_frames/source_camera/camera-a",
        archive_token=token,
    )
    camera.children["continuous"] = world["camera_frame_node"]
    source_camera.children["camera-a"] = camera
    coordinate_frames.children["source_camera"] = source_camera
    analysis.children["coordinate_frames"] = coordinate_frames

    run = analysis["stimulus_runs"]["stim_1"]
    calibration_camera = run["calibration"]["camera-a"]
    run_frames = FakeGroup(
        path=(
            "analysis/stimulus_runs/stim_1/calibration/camera-a/"
            "coordinate_frames"
        ),
        archive_token=token,
    )
    run_frames.children["selected_camera_evidence"] = FakeGroup(
        path=f"{run_frames.path}/selected_camera_evidence",
        archive_token=token,
    )
    run_frames.children["source_camera_physical_mm"] = FakeGroup(
        path=f"{run_frames.path}/source_camera_physical_mm",
        archive_token=token,
    )
    calibration_camera.children["coordinate_frames"] = run_frames
    run.attrs["palette_run_completion_status"] = "running"
    run.attrs["stage_selector_eligible"] = False
    authority = publish_stimulus_physical_coordinate_authority(
        root,
        run,
        stimulus_run="stim_1",
        selected_calibration=world["selected_snapshot"],
    )
    assert authority is not None
    run.attrs["palette_run_completion_status"] = "complete"
    run.attrs["stage_selector_eligible"] = True
    reloaded = load_stimulus_physical_coordinate_authority(
        root,
        stimulus_run="stim_1",
    )
    assert reloaded is not None
    return reloaded


def _recording_physical_authority(world):
    _selected_stimulus_physical_authority(world)
    return publish_source_camera_physical_authority(
        world["root"],
        source_camera_evidence=world["camera_evidence"],
        source_kind="operator_verified_donor",
        provenance={"operator_verified": True, "donor_zarr": "/donor.zarr"},
    )


def test_writer_accepts_recording_physical_authority_without_stimulus_selector() -> None:
    world = _world(convention="continuous", archive_token=object())
    surface = _canonical_crop_position_surface(world)
    physical = _recording_physical_authority(world)
    tracks, summaries = _build_from_source(
        surface.coordinates,
        surface.temporal_authority,
        pixel_to_mm=physical.mm_per_pixel,
    )
    run = _WritableGroup(
        path="analysis/track_kinematics_runs/offline/tk_recording_physical",
        archive_token=world["archive_token"],
    )

    mod.save_track_kinematics_tracks(
        run,
        tracks,
        summaries,
        source_temporal_authority=surface.temporal_authority,
        positions_px_source=surface.coordinates,
        physical_authority=physical,
    )

    manifest = run.attrs["physical_coordinate_authority"]
    assert manifest["authority_kind"] == "recording_calibration"
    assert manifest["recording_calibration"] is True
    assert "stimulus_run" not in manifest
    assert run["tracks/id_7/positions_mm"].shape == (2, 2)


def test_track_resolver_falls_back_to_recording_physical_authority() -> None:
    world = _world(convention="continuous", archive_token=object())
    physical = _recording_physical_authority(world)

    resolved, info = mod.resolve_track_physical_authority(
        world["root"],
        stimulus_run=None,
    )

    assert resolved is not None
    assert resolved.manifest.record_sha256 == physical.manifest.record_sha256
    assert info["authority_kind"] == "recording_calibration"
    assert info["reason_code"] == "NONE"


def test_track_stage_fails_before_compute_without_physical_authority(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.zarr"
    source.mkdir()
    staging = tmp_path / "staging.zarr"
    called = False

    monkeypatch.setattr(mod, "open_zarr_root", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        mod,
        "resolve_track_physical_authority",
        lambda *_args, **_kwargs: (
            None,
            {"reason_code": "NO_RECORDING_PHYSICAL_AUTHORITY"},
        ),
    )

    def _unexpected_main(_argv):
        nonlocal called
        called = True

    monkeypatch.setattr(mod, "main", _unexpected_main)

    with pytest.raises(ValueError, match="requires sealed source-camera"):
        mod.stage_offline_track_kinematics_run(
            source,
            staging,
            keypoint_run="keypoints",
            run_name="track",
        )

    assert called is False
    assert not staging.exists()


def test_writer_publishes_all_mm_surfaces_with_selected_stimulus_authority() -> None:
    world = _world(convention="continuous", archive_token=object())
    surface = _canonical_crop_position_surface(world)
    physical = _selected_stimulus_physical_authority(world)
    tracks, summaries = _build_from_source(
        surface.coordinates,
        surface.temporal_authority,
        pixel_to_mm=physical.mm_per_pixel,
    )
    run = _WritableGroup(
        path="analysis/track_kinematics_runs/offline/tk_physical",
        archive_token=world["archive_token"],
    )

    mod.save_track_kinematics_tracks(
        run,
        tracks,
        summaries,
        source_temporal_authority=surface.temporal_authority,
        positions_px_source=surface.coordinates,
        physical_authority=physical,
    )

    track = run["tracks/id_7"]
    assert track["positions_mm"].attrs[COORDINATE_DESCRIPTOR_ATTR][
        "profile_id"
    ] == "physical_mm.source_camera_y_down.v1"
    assert track.attrs["physical_outputs_status"] == (
        "available_typed_source_camera_frame"
    )
    assert "speed_smoothed_mm" in track
    assert run.attrs["physical_coordinate_authority"][
        "authority_manifest_sha256"
    ] == physical.manifest.record_sha256
    assert run.attrs["total_distance_mm"] == (
        run.attrs["total_distance_px"] * physical.mm_per_pixel
    )
    assert run.attrs["summary"] == [track.attrs["summary"]]
    assert run.attrs["track_manifest"][0]["total_distance_mm"] == (
        track.attrs["summary"]["total_distance_mm"]
    )


@pytest.mark.parametrize("tamper_kind", ["array", "summary"])
def test_direct_physical_writer_rejects_unsealed_mm_payload_before_mutation(
    tamper_kind: str,
) -> None:
    world = _world(convention="continuous", archive_token=object())
    surface = _canonical_crop_position_surface(world)
    physical = _selected_stimulus_physical_authority(world)
    tracks, summaries = _build_from_source(
        surface.coordinates,
        surface.temporal_authority,
        pixel_to_mm=physical.mm_per_pixel,
    )
    if tamper_kind == "array":
        tracks[7]["cumulative_path_distance_mm"][0] += 1.0
    else:
        summaries[0]["total_distance_mm"] += 1.0
    run = _WritableGroup(
        path=f"analysis/track_kinematics_runs/offline/tk_tamper_{tamper_kind}",
        archive_token=world["archive_token"],
    )

    with pytest.raises(ValueError, match="exact source-camera mm_per_pixel"):
        mod.save_track_kinematics_tracks(
            run,
            tracks,
            summaries,
            source_temporal_authority=surface.temporal_authority,
            positions_px_source=surface.coordinates,
            physical_authority=physical,
        )

    assert "tracks" not in run


def test_track_physical_selector_rejects_failed_explicit_stimulus_run() -> None:
    world = _world(convention="continuous", archive_token=object())
    _selected_stimulus_physical_authority(world)
    run = world["root"]["analysis"]["stimulus_runs"]["stim_1"]
    run.attrs["palette_run_completion_status"] = "failed"

    with pytest.raises(ValueError, match="requires parent run status 'complete'"):
        mod.resolve_canonical_track_physical_authority(
            world["root"],
            stimulus_run="stim_1",
        )


def test_stimulus_physical_authority_cannot_be_constructed_or_forged() -> None:
    world = _world(convention="continuous", archive_token=object())
    authority = _selected_stimulus_physical_authority(world)

    with pytest.raises(
        ValueError,
        match="cannot be constructed directly",
    ):
        BoundStimulusPhysicalCoordinateAuthority(
            stimulus_run=authority.stimulus_run,
            camera_id=authority.camera_id,
            archive_identity=authority.archive_identity,
            selected_calibration=authority.selected_calibration,
            acquisition_frame=authority.acquisition_frame,
            source_camera_frame=authority.source_camera_frame,
            selected_camera_evidence=authority.selected_camera_evidence,
            physical_frame=authority.physical_frame,
            manifest=authority.manifest,
            root_node=world["root"],
        )

    forged = object.__new__(BoundStimulusPhysicalCoordinateAuthority)
    for name in BoundStimulusPhysicalCoordinateAuthority.__dataclass_fields__:
        if name != "_seal":
            object.__setattr__(forged, name, getattr(authority, name))
    object.__setattr__(forged, "_seal", object())
    with pytest.raises(ValueError, match="freshly loader-minted"):
        require_bound_stimulus_physical_coordinate_authority(forged)


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
    # Real Zarr groups may return a distinct Python Array wrapper for every
    # lookup.  The final-path binder must retain one handle while stamping the
    # time lineage and the row-identity contract because those two operations
    # intentionally require exact in-memory authority continuity.
    track.fresh_array_handle_names.add("track_sample_key")
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
    root = _WritableGroup(path="", archive_token=world["archive_token"])
    analysis = root.create_group("analysis")
    runs = analysis.create_group("track_kinematics_runs")
    offline = runs.create_group("offline")
    offline.children[run_name] = run

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
    assert mod._validate_bound_offline_track_kinematics_run_before_selection(
        root,
        run,
        expected_keypoint_run="kp_1",
        expected_run_name=run_name,
        require_complete=False,
    )["valid"] is True
    run.attrs["palette_run_completion_status"] = "complete"
    assert mod._validate_bound_offline_track_kinematics_run_before_selection(
        root,
        run,
        expected_keypoint_run="kp_1",
        expected_run_name=run_name,
        require_complete=True,
    )["valid"] is True
    run.attrs["stage_selector_eligible"] = True
    published = mod.load_bound_track_position_bindings(root, run)
    assert isinstance(published, mod.BoundTrackPositionBindings)
    assert published.run_type == "offline"
    assert published.run_name == run_name
    assert published.source_positions.descriptor.space_id == (
        "source_camera_image_px"
    )
    track_position = published.position_for_track(7)
    assert track_position.positions_px.coordinate_node is track["positions_px"]
    assert track_position.positions_mm is None


def test_deferred_physical_payload_is_unbound_until_final_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = _world(convention="continuous", archive_token=object())
    surface = _canonical_crop_position_surface(world)
    physical = _selected_stimulus_physical_authority(world)
    tracks, summaries = _build_from_source(
        surface.coordinates,
        surface.temporal_authority,
        pixel_to_mm=physical.mm_per_pixel,
    )
    run_name = "tk_staged_physical"
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
        physical_authority=physical,
        defer_coordinate_binding=True,
        staging_keypoint_run="kp_1",
        staging_run_name=run_name,
    )
    track = run["tracks/id_7"]
    physical_manifest = run.attrs[
        mod.TRACK_KINEMATICS_STAGING_MANIFEST_ATTR
    ]["physical_authority"]
    assert physical_manifest["authority_manifest_ref"] == (
        physical.manifest.record_ref
    )
    assert physical_manifest["authority_manifest_sha256"] == (
        physical.manifest.record_sha256
    )
    assert "positions_mm" in track
    assert COORDINATE_DESCRIPTOR_ATTR not in track["positions_mm"].attrs
    assert run.attrs["physical_outputs_reason_code"] == "NONE"

    monkeypatch.setattr(
        mod,
        "load_persisted_source_camera_position_surface",
        lambda _root, path: surface if path == "crop_runs/c1" else None,
    )
    run.attrs["palette_run_completion_status"] = "running"
    run.attrs[mod.TRACK_KINEMATICS_COORDINATE_BINDING_STATUS_ATTR] = (
        mod.TRACK_KINEMATICS_PUBLISHING_BINDING_STATUS
    )
    bound = mod.bind_staged_offline_track_kinematics_run(
        world["root"],
        run,
        expected_keypoint_run="kp_1",
        expected_run_name=run_name,
    )

    assert bound["valid"] is True
    assert track["positions_mm"].attrs[COORDINATE_DESCRIPTOR_ATTR][
        "profile_id"
    ] == "physical_mm.source_camera_y_down.v1"
    assert mod._validate_bound_offline_track_kinematics_run_before_selection(
        world["root"],
        run,
        expected_keypoint_run="kp_1",
        expected_run_name=run_name,
        require_complete=False,
    )["valid"] is True


def test_deferred_physical_binding_rejects_selected_calibration_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = _world(convention="continuous", archive_token=object())
    surface = _canonical_crop_position_surface(world)
    physical = _selected_stimulus_physical_authority(world)
    tracks, summaries = _build_from_source(
        surface.coordinates,
        surface.temporal_authority,
        pixel_to_mm=physical.mm_per_pixel,
    )
    run_name = "tk_staged_physical_drift"
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
        physical_authority=physical,
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
    selected_camera = world["root"]["analysis"]["stimulus_runs"]["stim_1"][
        "calibration"
    ]["camera-a"]
    selected_camera.attrs["pixels_per_mm_camera"] = 12.5

    with pytest.raises(ValueError, match="freshly rebound"):
        mod.bind_staged_offline_track_kinematics_run(
            world["root"],
            run,
            expected_keypoint_run="kp_1",
            expected_run_name=run_name,
        )

    track = run["tracks/id_7"]
    assert COORDINATE_DESCRIPTOR_ATTR not in track["positions_px"].attrs
    assert COORDINATE_DESCRIPTOR_ATTR not in track["positions_mm"].attrs


def test_deferred_physical_binding_rejects_mm_payload_drift_without_partial_claims(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = _world(convention="continuous", archive_token=object())
    surface = _canonical_crop_position_surface(world)
    physical = _selected_stimulus_physical_authority(world)
    tracks, summaries = _build_from_source(
        surface.coordinates,
        surface.temporal_authority,
        pixel_to_mm=physical.mm_per_pixel,
    )
    run_name = "tk_staged_physical_payload_drift"
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
        physical_authority=physical,
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
    track = run["tracks/id_7"]
    track["cumulative_path_distance_mm"].data[0] += 1.0

    with pytest.raises(ValueError, match="changed after physical staging"):
        mod.bind_staged_offline_track_kinematics_run(
            world["root"],
            run,
            expected_keypoint_run="kp_1",
            expected_run_name=run_name,
        )

    assert run.attrs[mod.TRACK_KINEMATICS_COORDINATE_BINDING_STATUS_ATTR] == (
        mod.TRACK_KINEMATICS_PUBLISHING_BINDING_STATUS
    )
    assert COORDINATE_DESCRIPTOR_ATTR not in track["positions_px"].attrs
    assert COORDINATE_DESCRIPTOR_ATTR not in track["positions_mm"].attrs


@pytest.mark.parametrize(
    "tamper_kind",
    ["track_summary", "run_summary", "track_manifest", "aggregate"],
)
def test_deferred_physical_binding_rejects_unsealed_mm_metadata_surfaces(
    monkeypatch: pytest.MonkeyPatch,
    tamper_kind: str,
) -> None:
    world = _world(convention="continuous", archive_token=object())
    surface = _canonical_crop_position_surface(world)
    physical = _selected_stimulus_physical_authority(world)
    tracks, summaries = _build_from_source(
        surface.coordinates,
        surface.temporal_authority,
        pixel_to_mm=physical.mm_per_pixel,
    )
    run_name = f"tk_staged_physical_{tamper_kind}"
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
        physical_authority=physical,
        defer_coordinate_binding=True,
        staging_keypoint_run="kp_1",
        staging_run_name=run_name,
    )
    track = run["tracks/id_7"]
    if tamper_kind == "track_summary":
        payload = copy.deepcopy(track.attrs["summary"])
        payload["total_distance_mm"] += 1.0
        track.attrs["summary"] = payload
    elif tamper_kind == "run_summary":
        payload = copy.deepcopy(run.attrs["summary"])
        payload[0]["total_distance_mm"] += 1.0
        run.attrs["summary"] = payload
    elif tamper_kind == "track_manifest":
        payload = copy.deepcopy(run.attrs["track_manifest"])
        payload[0]["total_distance_mm"] += 1.0
        run.attrs["track_manifest"] = payload
    else:
        run.attrs["total_distance_mm"] += 1.0
    monkeypatch.setattr(
        mod,
        "load_persisted_source_camera_position_surface",
        lambda _root, _path: surface,
    )
    run.attrs["palette_run_completion_status"] = "running"
    run.attrs[mod.TRACK_KINEMATICS_COORDINATE_BINDING_STATUS_ATTR] = (
        mod.TRACK_KINEMATICS_PUBLISHING_BINDING_STATUS
    )

    with pytest.raises(ValueError):
        mod.bind_staged_offline_track_kinematics_run(
            world["root"],
            run,
            expected_keypoint_run="kp_1",
            expected_run_name=run_name,
        )

    assert run.attrs[mod.TRACK_KINEMATICS_COORDINATE_BINDING_STATUS_ATTR] == (
        mod.TRACK_KINEMATICS_PUBLISHING_BINDING_STATUS
    )
    assert COORDINATE_DESCRIPTOR_ATTR not in track["positions_px"].attrs
    assert COORDINATE_DESCRIPTOR_ATTR not in track["positions_mm"].attrs


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
