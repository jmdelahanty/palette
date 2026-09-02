from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
import uuid

import numpy as np
import pytest
import zarr

from fisheye.analysis import track_kinematics as track_mod
from fisheye.shared import coordinate_identity as identity_mod
from fisheye.shared import coordinate_record as record_mod
from fisheye.shared import manifest_crop_position_authority as profile_mod
from fisheye.shared import observation_coordinate_publication as position_mod
from fisheye.shared.zarr import crop_snapshot_publication as crop_publication_mod
from fisheye.shared.keypoint_coordinate_publication import (
    KeypointCoordinatePublicationError,
    load_persisted_keypoint_crop_source,
    require_direct_keypoint_crop_pixel_source,
)
from fisheye.shared.observation_coordinate_publication import (
    load_persisted_source_camera_position_surface,
    resolve_source_detection_rowset_from_position_coordinates,
)
from fisheye.shared.subject_mask_coordinate_publication import (
    load_persisted_subject_mask_crop_source,
)
from fisheye.shared.zarr.crop_snapshot_publication import (
    publish_crop_geometry_production_candidate,
)
from tests.unit.fisheye.test_crop_shadow import _pixel, _policy, _refined_source
from tests.unit.fisheye.test_crop_snapshot_publication import (
    _BoundPixels,
    _wire_authorities,
)


def test_geometry_crop_profile_has_one_advertised_position_interface() -> None:
    """Keep profile proof construction behind the shared resolver boundary."""

    assert profile_mod.__all__ == []
    assert "build_bound_source_camera_position_surface" not in position_mod.__all__
    assert "bind_persisted_manifest_coordinate_record" not in record_mod.__all__
    assert "bind_manifest_row_identity_contract" not in identity_mod.__all__
    assert "bind_manifest_source_row_temporal_authority" not in identity_mod.__all__
    assert "load_persisted_source_camera_position_surface" in position_mod.__all__


def test_real_geometry_crop_publisher_round_trips_shared_crop_source_resolver(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Guard the real writer -> unpatched keypoint/mask resolver boundary."""

    source = replace(
        _refined_source(tmp_path),
        selection_mode="approved_authoritative_refined_v1",
    )
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    pixels = _BoundPixels(
        pixel_authority=_pixel(),
        source_video_path=tmp_path / "camera.mp4",
    )
    _wire_authorities(monkeypatch, source, pixels)
    writable_root = zarr.open_group(
        str(source.archive_path),
        mode="a",
        use_consolidated=False,
    )
    provider = writable_root.require_group("crop_runs").create_group(
        "hybrid_provider"
    )
    provider.attrs.update(
        {
            "status": "completed",
            "stage_selector_eligible": False,
            "provider_record_sha256": "a" * 64,
            "source_refined_run_id": source.run_id,
            "source_refined_manifest_digest": source.manifest["payload_digest"],
        }
    )
    source_paths = {
        "instance_key": "instances/instance_key",
        "source_refined_row_ids": "instances/refined_row_ids",
        "frame_indices": "instances/frame_indices",
        "source_acquisition_frame_index": (
            "instances/source_acquisition_frame_index"
        ),
    }
    for provider_path, source_path in source_paths.items():
        provider.create_array(
            provider_path,
            data=np.asarray(source.arrays[source_path][...]),
        )
    provider.create_array(
        "roi_coordinates_full",
        data=np.asarray([[14, 10], [67, 11], [46, 51], [20, 55]], dtype=np.int32),
    )
    provider.create_array(
        "roi_sizes_full",
        data=np.full((4, 2), 8, dtype=np.int32),
    )
    monkeypatch.setattr(
        crop_publication_mod,
        "validate_hybrid_crop_signed_identity",
        lambda *_args, **_kwargs: {
            "provider_record_sha256": "a" * 64,
            "row_count": 4,
            "source_row_signature_spec_digest": "b" * 64,
            "source_pixel_fingerprint": "c" * 64,
            "source_rowset_fingerprint": "d" * 64,
        },
    )
    publish_crop_geometry_production_candidate(
        analysis_zarr=source.archive_path,
        run_id="crop_shared_resolver_roundtrip",
        policy=_policy(),
        expected_camera_identity="cam2010095",
        scratch_root=scratch,
        geometry_origin_provider_run_id="hybrid_provider",
    )

    root = zarr.open_group(
        str(source.archive_path),
        mode="a",
        use_consolidated=False,
    )
    path = "crop_runs/crop_shared_resolver_roundtrip"
    keypoint_source = load_persisted_keypoint_crop_source(root, path)
    mask_source = load_persisted_subject_mask_crop_source(root, path)

    assert keypoint_source.crop_profile == "sealed_geometry_only_v2"
    assert mask_source.crop_profile == keypoint_source.crop_profile
    assert mask_source.crop_path == keypoint_source.crop_path == path
    assert keypoint_source._roi_images_node is None
    assert keypoint_source.crop_geometry.row_identity.leading_dimension > 0
    with pytest.raises(KeypointCoordinatePublicationError):
        require_direct_keypoint_crop_pixel_source(
            keypoint_source,
            root[f"{path}/centers_img_xy"],
        )


def test_real_geometry_crop_publisher_round_trips_track_motion_v2(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Guard the real writer -> resolver -> motion-publication boundary."""

    source = replace(
        _refined_source(tmp_path),
        selection_mode="approved_authoritative_refined_v1",
    )
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    pixels = _BoundPixels(
        pixel_authority=_pixel(),
        source_video_path=tmp_path / "camera.mp4",
    )
    _wire_authorities(monkeypatch, source, pixels)
    publish_crop_geometry_production_candidate(
        analysis_zarr=source.archive_path,
        run_id="crop_track_roundtrip",
        policy=_policy(),
        expected_camera_identity="cam2010095",
        scratch_root=scratch,
    )

    root = zarr.open_group(
        str(source.archive_path),
        mode="a",
        use_consolidated=False,
    )
    surface = load_persisted_source_camera_position_surface(
        root,
        "crop_runs/crop_track_roundtrip",
    )
    offline = track_mod.load_canonical_offline_position_source(
        root,
        root["crop_runs/crop_track_roundtrip"],
        crop_run_name="crop_track_roundtrip",
    )
    detection_path = resolve_source_detection_rowset_from_position_coordinates(
        surface.coordinates
    )
    assert detection_path == "refined_detect_runs/refined_crop_source"

    row_count = int(offline.positions_px.shape[0])
    instance_keys = np.asarray(offline.instance_key, dtype=np.uint64)
    keypoint = root.require_group("keypoints_runs").create_group("kp_geometry_profile")
    keypoint.attrs.update(
        {
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": True,
            "source_crop_run": "crop_track_roundtrip",
            "keypoints_processed": row_count,
        }
    )
    heading = keypoint.create_array(
        "heading",
        data=np.zeros(row_count, dtype=np.float32),
    )
    heading_usable = keypoint.create_array(
        "heading_usable",
        data=np.ones(row_count, dtype=bool),
    )
    keypoint_keys = keypoint.create_array(
        "instance_key",
        data=instance_keys,
    )
    tracking = root.require_group("tracking_runs").create_group("trk_geometry_profile")
    track_ids = np.arange(1, row_count + 1, dtype=np.int32)
    tracking.create_array("track_ids", data=track_ids)
    tracking.create_array(
        "arena_ids",
        data=np.full(row_count, 3, dtype=np.int32),
    )
    tracking.create_array("instance_key", data=instance_keys)
    tracking.create_array("track_ids_present", data=track_ids)
    tracking.create_array(
        "track_arena_ids",
        data=np.full(row_count, 3, dtype=np.int32),
    )
    input_authority = track_mod.build_track_motion_input_authority(
        root,
        source_positions=surface.coordinates,
        mode="offline_exact_sources_v1",
        heading_node=heading,
        keypoint_usability_node=heading_usable,
        keypoint_row_key_node=keypoint_keys,
        tracking_group=tracking,
    )
    source_rows = np.arange(row_count, dtype=np.int64)
    frames = track_mod.resolve_source_acquisition_frame_indices(
        surface.temporal_authority,
        source_rows,
    )
    tracks, summaries = track_mod.build_track_datasets(
        track_ids=track_ids.astype(np.int64),
        frames=frames,
        positions_px=offline.positions_px,
        headings_deg=np.zeros(row_count, dtype=np.float32),
        keypoint_success=np.ones(row_count, dtype=bool),
        detection_source=None,
        fps=1.0,
        smooth_seconds=1.0,
        pixel_to_mm=None,
        hysteresis_high_px=None,
        hysteresis_low_px=None,
        hysteresis_min_frames=None,
        hysteresis_band_policy="reset",
        smoothing_alignment="centered",
        source_row_index=source_rows,
        source_temporal_authority=surface.temporal_authority,
    )
    run = (
        root.require_group("analysis")
        .require_group("track_kinematics_runs")
        .require_group("offline")
        .create_group("geometry_profile_roundtrip")
    )
    run.attrs[track_mod.TRACK_KINEMATICS_PUBLICATION_OWNER_ATTR] = str(uuid.uuid4())
    track_mod.save_track_kinematics_tracks(
        run,
        tracks,
        summaries,
        source_temporal_authority=surface.temporal_authority,
        positions_px_source=surface.coordinates,
        input_authority=input_authority,
        track_id_to_arena_id={int(track_id): 3 for track_id in track_ids},
    )
    inputs = {
        "detection_path": detection_path,
        "position_source_path": ("crop_runs/crop_track_roundtrip/centers_img_xy"),
        "position_source_rowset_path": "crop_runs/crop_track_roundtrip",
        "position_source_kind": ("canonical_crop_rows_source_camera_centers"),
        "position_lineage_mode": (
            track_mod.TRACK_POSITION_LINEAGE_GEOMETRY_ONLY_CROP_V2
        ),
        "keypoint_path": "keypoints_runs/kp_geometry_profile",
        "crop_run": "crop_track_roundtrip",
        "tracking_path": "tracking_runs/trk_geometry_profile",
    }
    parameters = {
        "fps": 1.0,
        "smoothing_seconds": 1.0,
        "smoothing_method": "moving_average",
        "smoothing_alignment": "centered",
        "savgol_polyorder": None,
        "distance_interpolation_seconds": 0.0,
        "coordinate_space": "source_camera_image_px",
        "hysteresis_enabled": False,
        "hysteresis_high_px": None,
        "hysteresis_low_px": None,
        "hysteresis_min_frames": None,
        "hysteresis_band_policy": "reset",
    }
    run.attrs.update(
        track_mod._track_kinematics_contract_attrs(
            run_type="offline",
            method="track_kinematics_offline",
            parameters=parameters,
            inputs=inputs,
            publication_schema_version=(
                track_mod.TRACK_MOTION_PUBLICATION_MANIFEST_SCHEMA_VERSION_V2
            ),
        )
    )
    provenance = {
        "stage": "track_kinematics",
        "parameters": copy.deepcopy(parameters),
        "inputs": copy.deepcopy(inputs),
    }
    run.attrs.update(
        {
            "inputs": inputs,
            "fps": 1.0,
            "smoothing_seconds": 1.0,
            "smoothing_method": "moving_average",
            "smoothing_alignment": "centered",
            "savgol_polyorder": None,
            "distance_interpolation_seconds": 0.0,
            "hysteresis_enabled": False,
            "hysteresis_high_px": None,
            "hysteresis_low_px": None,
            "hysteresis_min_frames": None,
            "hysteresis_band_policy": "reset",
            "provenance": provenance,
            "run_provenance": {
                "schema": "palette.run_provenance.v1",
                "git_sha": "a" * 40,
                "config_hash": track_mod.sha256_payload(parameters),
                "params": copy.deepcopy(parameters),
                "input_run_ids": copy.deepcopy(inputs),
                "command": "test_geometry_profile_roundtrip",
                "fisheye_version": None,
            },
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": False,
        }
    )

    sealed_track = track_mod._seal_and_load_track_motion_run_before_selection(
        root,
        run,
    )
    assert sealed_track.manifest["schema_version"] == 2
    assert sealed_track.manifest["position_lineage_mode"] == (
        track_mod.TRACK_POSITION_LINEAGE_GEOMETRY_ONLY_CROP_V2
    )
    lineage = sealed_track.manifest["source_authority"]["position_lineage"]
    assert lineage["crop_manifest"]["record_ref"] == (
        "/crop_runs/crop_track_roundtrip@run_manifest"
    )
