from __future__ import annotations

from pathlib import Path
import json

import h5py
import numpy as np
import pytest
import zarr

from fisheye.analysis import import_stimulus_to_zarr as mod
from fisheye.analysis import chaser_metrics_loader
from fisheye.shared.experiment_setup import (
    resolve_experiment_setup,
)
from fisheye.shared.subject_metadata import resolve_subject_metadata
from fisheye.analysis.stimulus_response_coordinate_authority import (
    STIMULUS_RESPONSE_COORDINATE_LINEAGE_SCHEMA_ID,
    STIMULUS_RESPONSE_COORDINATE_LINEAGE_SCHEMA_VERSION,
    StimulusResponseCoordinateAuthorityError,
    load_stimulus_response_coordinate_authority,
)
from fisheye.shared.acquisition_publication_status import (
    ACQUISITION_AUTHORITY_NOT_PUBLISHED,
    ACQUISITION_AUTHORITY_PENDING,
    ACQUISITION_AUTHORITY_PUBLISHED,
    ACQUISITION_AUTHORITY_STATUS_ATTR,
    EXTERNAL_ACQUISITION_AUTHORITY_MODE,
    EXTERNAL_ACQUISITION_PENDING_REASON,
    EXTERNAL_ACQUISITION_PUBLISHED_REASON,
    MATERIALIZED_ACQUISITION_AUTHORITY_MODE,
    MATERIALIZED_ACQUISITION_PUBLISHED_REASON,
    build_acquisition_authority_publication_status,
    stamp_acquisition_authority_publication_status,
)
from fisheye.shared.coordinate_descriptor import (
    PIXEL_FRAME_AUTHORITY_RECORD_KIND,
    CanonicalFrameRecord,
    DigestBoundCoordinateRecordRef,
    build_canonical_coordinate_descriptor,
    canonical_coordinate_descriptor_v2_attrs,
    load_canonical_coordinate_descriptor_attrs,
)
from fisheye.shared.directed_transform import TransformReferenceExtent
from fisheye.shared.coordinate_identity import (
    ROW_IDENTITY_CONTRACT_ATTR,
    ROW_IDENTITY_CONTRACT_DIGEST_ATTR,
    ROW_IDENTITY_KEY_ATTR,
    ROW_IDENTITY_KEY_DIGEST_ATTR,
    STIMULUS_STATE_DOMAIN,
    STIMULUS_STATE_KEY_ARRAY_REF,
    STIMULUS_STATE_KEY_MODE,
    build_row_identity_contract,
    identity_array_content_sha256,
    row_identity_contract_attrs,
    row_identity_key_attrs,
    validate_stamped_row_identity,
)
from fisheye.shared.selected_calibration import (
    ACTIVE_CAMERA_ID_ATTR,
    SelectedCalibrationSnapshot,
    load_selected_calibration_snapshot,
    selected_calibration_paths,
)
from fisheye.shared.pixel_frame_authority import (
    PIXEL_FRAME_AUTHORITY_ATTR,
    PIXEL_FRAME_AUTHORITY_DIGEST_ATTR,
    stamp_acquisition_camera_frame,
    stamp_acquisition_import_ownership,
)
from fisheye.shared.stimulus_coordinate_contract import (
    ARENA_GEOMETRY_RECORD_ATTR,
    ARENA_GEOMETRY_RECORD_DIGEST_ATTR,
    COORDINATE_CONTRACT_EPOCH,
    COORDINATE_IMPORT_LINEAGE_ATTR,
    COORDINATE_IMPORT_LINEAGE_DIGEST_ATTR,
    COORDINATE_OUTPUT_MANIFEST_ATTR,
    COORDINATE_OUTPUT_MANIFEST_DIGEST_ATTR,
    COORDINATE_SURFACE_MANIFEST_ATTR,
    COORDINATE_SURFACE_MANIFEST_DIGEST_ATTR,
    COORDINATE_SURFACE_MANIFEST_SCHEMA,
    COORDINATE_SURFACE_MANIFEST_VERSION,
    CAMERA_FRAME_IDS_ARRAY,
    CAMERA_MAPPING_RECORD_ATTR,
    CAMERA_MAPPING_RECORD_DIGEST_ATTR,
    SOURCE_ROW_INDICES_ARRAY,
    SOURCE_ACQUISITION_FRAME_INDEX_ARRAY,
    SOURCE_ACQUISITION_MAPPING_ARRAY_PATH,
    SOURCE_ACQUISITION_MAPPING_RECORD_ATTR,
    SOURCE_ACQUISITION_MAPPING_RECORD_DIGEST_ATTR,
    SOURCE_ACQUISITION_MAPPING_SCHEMA_ID,
    SOURCE_ACQUISITION_MAPPING_SCHEMA_VERSION,
    SOURCE_COORDINATE_POLICY_METADATA_ONLY,
    STIMULUS_IMPORT_VERSION,
    StimulusCoordinateContractError,
    arena_geometry_record,
    canonical_mapping_digest,
    numpy_content_digest,
    source_arena_pixel_frame_record,
)
from fisheye.shared.stimulus_physical_coordinate import (
    STIMULUS_PHYSICAL_COORDINATE_BOUND_STATUS,
    STIMULUS_PHYSICAL_COORDINATE_INVALIDATED_STATUS,
    STIMULUS_PHYSICAL_COORDINATE_MANIFEST_REF_ATTR,
    STIMULUS_PHYSICAL_COORDINATE_MANIFEST_SHA256_ATTR,
    STIMULUS_PHYSICAL_COORDINATE_OMITTED_NO_ACQUISITION_STATUS,
    STIMULUS_PHYSICAL_COORDINATE_OMITTED_NO_SCALE_STATUS,
    STIMULUS_PHYSICAL_COORDINATE_REASON_BOUND,
    STIMULUS_PHYSICAL_COORDINATE_REASON_CODE_ATTR,
    STIMULUS_PHYSICAL_COORDINATE_REASON_NO_ACQUISITION,
    STIMULUS_PHYSICAL_COORDINATE_REASON_NO_SCALE,
    STIMULUS_PHYSICAL_COORDINATE_REASON_PARENT_RUN_FAILED,
    STIMULUS_PHYSICAL_COORDINATE_STATUS_ATTR,
    StimulusPhysicalCoordinateError,
    load_stimulus_physical_coordinate_authority,
)


def _write_minimal_stimulus_h5(path: Path) -> None:
    dtype = np.dtype(
        [
            ("stimulus_frame_num", np.uint64),
            ("triggering_camera_frame_id", np.uint64),
            ("timestamp_ns", np.int64),
        ]
    )
    frame_metadata = np.array(
        [
            (1000, 2000, 1_000_000_000),
            (1001, 2001, 1_008_333_333),
        ],
        dtype=dtype,
    )
    with h5py.File(path, "w") as h5:
        video_metadata = h5.create_group("video_metadata")
        video_metadata.create_dataset("frame_metadata", data=frame_metadata)
        arena_config = {
            "active_camera_id": "2010093",
            "calculated_z_eff_mm": 0.0,
            "experimental_area_center_x_px": 172.0,
            "experimental_area_center_y_px": 172.0,
            "experimental_area_radius_px": 166.0,
            "experimental_area_shape": "CIRCLE",
            "sub_arena_x_px": 270,
            "sub_arena_y_px": 520,
            "sub_arena_width_px": 344,
            "sub_arena_height_px": 344,
            "camera_calibrations": [
                {
                    "camera_id": "2010093",
                    "native_width_px": 4512,
                    "native_height_px": 4512,
                    "pixels_per_mm_camera": 50.0,
                    "pixels_per_mm_projector": 5.0,
                    "real_world_ref_mm": 10.0,
                }
            ],
        }
        homography_yml = """%YAML:1.0
---
homography_matrix:
  rows: 3
  cols: 3
  dt: d
  data: [1, 0, 10, 0, 1, 20, 0, 0, 1]
"""
        calib = h5.create_group("calibration_snapshot")
        calib.create_dataset("arena_config_json", data=json.dumps(arena_config).encode("utf-8"))
        cam = calib.create_group("2010093")
        cam.attrs["pixels_per_mm_camera"] = 50.0
        cam.attrs["pixels_per_mm_projector"] = 5.0
        cam.attrs["real_world_ref_mm"] = 10.0
        matrix = np.array(
            [[1.0, 0.0, 10.0], [0.0, 1.0, 20.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        common_homography_attrs = {
            "arena_config_name": "arena_1",
            "axes": "x_right_y_down",
            "camera_id": "2010093",
            "canvas_name": "shadow",
            "coordinate_origin": "top_left",
            "dest_frame": "final_display_canvas_px",
            "homography_artifact_checksum_algorithm": "fnv1a64",
            "homography_artifact_checksum_fnv1a64": "f999f671b0ebd9fd",
            "homography_artifact_exists": "true",
            "homography_artifact_mtime_unix_ns": 1779901497426733358,
            "homography_artifact_path": "/rig/calibration/homography.yml",
            "homography_artifact_size_bytes": 347,
            "homography_provenance_schema": "citrus.homography_provenance.v1",
            "image_space": "raw",
            "source_frame": "camera_view_px",
        }
        numeric = cam.create_dataset("homography_matrix", data=matrix)
        numeric.attrs.update(
            {
                **common_homography_attrs,
                "homography_payload_source": (
                    "runtime_arena_config.homography_matrix"
                ),
            }
        )
        yaml_node = cam.create_dataset(
            "homography_matrix_yml",
            data=homography_yml.encode("utf-8"),
        )
        yaml_node.attrs.update(
            {
                **common_homography_attrs,
                "homography_payload_source": "resolved_calibration_artifact_file",
                "serialization_format": "opencv_yml",
            }
        )
        projected_surface = cam.create_group("scale_models").create_group("projected_surface")
        projected_surface.attrs["model_name"] = "projected_surface"
        projected_surface.create_dataset("scale_image_png_buffer", data=np.array([1, 2, 3], dtype=np.uint8))
        arena_geometry = calib.create_group("arena_geometry")
        arena_geometry.attrs["arena_region_width_px"] = 344
        arena_geometry.attrs["arena_region_height_px"] = 344
        arena_geometry.attrs["arena_origin_in_canvas_x_px"] = 270
        arena_geometry.attrs["arena_origin_in_canvas_y_px"] = 520
        display = h5.create_group("display_snapshot")
        display.attrs.update(
            {
                "selected_output_name": "DP-3",
                "selected_output_connection_state": "connected",
                "selected_output_geometry": "1920x1080+3840+0",
                "selected_output_transform_token": "normal",
                "selected_output_transform_raw": (
                    "normal left inverted right x axis y axis"
                ),
            }
        )
        display.create_dataset(
            "selected_output_block",
            data=(
                b"DP-3 connected 1920x1080+3840+0 "
                b"(normal left inverted right x axis y axis) 0mm x 0mm\n"
                b"   1920x1080 119.88*"
            ),
        )


def _write_stimulus_h5_with_calibration(path: Path) -> None:
    _write_minimal_stimulus_h5(path)


def _prepare_acquisition_authority(
    zarr_path: Path,
    *,
    total_frames: int = 2,
) -> None:
    root = zarr.open_group(str(zarr_path), mode="a")
    root.attrs.update(
        {
            "recording_id": "recording-1",
            "source_video_metadata": {
                "schema_id": "palette.source_video_metadata.v2",
                "layout": "single_video",
                "camera_id": "2010093",
                "source_path": "/recording/camera.mp4",
                "width": 4512,
                "height": 4512,
                "total_frames": total_frames,
                "locator": {
                    "kind": "recording_relative",
                    "relative_path": "camera.mp4",
                },
                "file_fingerprint": {
                    "strategy": "size_mtime_sha256_v1",
                    "value": "a" * 64,
                    "size_bytes": 1234,
                    "mtime_ns": 5678,
                    "relocation_stable": False,
                },
            },
        }
    )
    analysis = root.require_group("analysis")
    authorities = analysis.require_group("acquisition_camera_frames")
    camera = authorities.create_group("2010093")
    ownership = stamp_acquisition_import_ownership(root, camera)
    stamp_acquisition_camera_frame(
        root,
        camera,
        import_ownership=ownership,
    )
    raw_video = root.require_group("raw_video")
    stamp_acquisition_authority_publication_status(
        root,
        raw_video,
        status=ACQUISITION_AUTHORITY_PUBLISHED,
        reason_code=EXTERNAL_ACQUISITION_PUBLISHED_REASON,
        authority_mode=EXTERNAL_ACQUISITION_AUTHORITY_MODE,
        authority_path="analysis/acquisition_camera_frames/2010093",
    )


def _load_output_coordinate_descriptor(
    chaser_group: zarr.Group,
    node: zarr.Array,
):
    identity = validate_stamped_row_identity(
        chaser_group,
        chaser_group[STIMULUS_STATE_KEY_ARRAY_REF],
    )
    return load_canonical_coordinate_descriptor_attrs(
        node.attrs,
        row_identity_contract=identity,
        expected_row_identity_record_ref=(
            f"/{chaser_group.path}@{ROW_IDENTITY_CONTRACT_ATTR}"
        ),
        owner_shape=node.shape,
    )


def _write_h5_contract_attrs(
    node: h5py.Group | h5py.Dataset,
    attrs: dict[str, object],
) -> None:
    for name, value in attrs.items():
        node.attrs[name] = (
            json.dumps(value, separators=(",", ":"), sort_keys=True)
            if isinstance(value, (dict, list))
            else value
        )


def _refresh_source_acquisition_identity_binding(
    h5: h5py.File,
    *,
    row_values: np.ndarray,
    identity_contract: object,
) -> None:
    node = h5[SOURCE_ACQUISITION_MAPPING_ARRAY_PATH]
    record = json.loads(node.attrs[SOURCE_ACQUISITION_MAPPING_RECORD_ATTR])
    record["source_row_identity_sha256"] = identity_array_content_sha256(
        row_values
    )
    record["source_row_identity_contract_sha256"] = identity_contract.digest()
    _write_h5_contract_attrs(
        node,
        {
            SOURCE_ACQUISITION_MAPPING_RECORD_ATTR: record,
            SOURCE_ACQUISITION_MAPPING_RECORD_DIGEST_ATTR: (
                canonical_mapping_digest(record)
            ),
        },
    )


def _write_stimulus_h5_with_arena_relative_chaser_states(
    path: Path,
    *,
    include_legacy_attrs: bool = False,
    multi_chaser: bool = False,
) -> None:
    _write_stimulus_h5_with_calibration(path)
    base_fields = [
        ("stimulus_frame_num", np.uint64),
        ("chaser_pos_x", np.float32),
        ("chaser_pos_y", np.float32),
        ("target_pos_x", np.float32),
        ("target_pos_y", np.float32),
        ("target_clamped_pos_x", np.float32),
        ("target_clamped_pos_y", np.float32),
    ]
    if multi_chaser:
        chaser_dtype = np.dtype([("chaser_index", np.uint16), *base_fields])
        chaser_states = np.array(
            [
                (0, 1000, 20.0, 30.0, 350.0, 358.5, 343.0, 344.0),
                (1, 1000, 40.0, 50.0, 330.0, 320.0, 330.0, 320.0),
                (0, 1001, 21.0, 31.0, 340.0, 330.0, 340.0, 330.0),
                (1, 1001, 41.0, 51.0, 329.0, 319.0, 329.0, 319.0),
            ],
            dtype=chaser_dtype,
        )
        row_identity_fields = ["chaser_index", "stimulus_frame_num"]
        row_identity_values = np.column_stack(
            [chaser_states[field] for field in row_identity_fields]
        ).astype(np.int64)
    else:
        chaser_dtype = np.dtype(base_fields)
        chaser_states = np.array(
            [
                (1000, 20.0, 30.0, 350.0, 358.5, 343.0, 344.0),
                (1001, 21.0, 31.0, 340.0, 330.0, 340.0, 330.0),
            ],
            dtype=chaser_dtype,
        )
        row_identity_fields = ["stimulus_frame_num"]
        row_identity_values = np.asarray(
            chaser_states["stimulus_frame_num"],
            dtype=np.int64,
        )

    component_fields = {
        "chaser_pos_x",
        "chaser_pos_y",
        "target_pos_x",
        "target_pos_y",
        "target_clamped_pos_x",
        "target_clamped_pos_y",
    }
    field_classifications = {
        name: (
            "row_identity"
            if name in row_identity_fields
            else "coordinate_component"
            if name in component_fields
            else "non_spatial"
        )
        for name in chaser_dtype.names or ()
    }

    manifest = {
        "schema_id": COORDINATE_SURFACE_MANIFEST_SCHEMA,
        "schema_version": COORDINATE_SURFACE_MANIFEST_VERSION,
        "coordinate_fields_complete": True,
        "field_classifications": field_classifications,
        "row_identity_fields": row_identity_fields,
        "surfaces": [
            {
                "array_name": "chaser_position_xy",
                "semantic_role": "chaser_position",
                "component_fields": ["chaser_pos_x", "chaser_pos_y"],
            },
            {
                "array_name": "target_position_xy",
                "semantic_role": "target_position",
                "component_fields": ["target_pos_x", "target_pos_y"],
            },
            {
                "array_name": "target_clamped_position_xy",
                "semantic_role": "target_clamped_position",
                "component_fields": ["target_clamped_pos_x", "target_clamped_pos_y"],
            },
        ],
    }
    arena_lineage = arena_geometry_record(
        {
            "arena_region_width_px": 344,
            "arena_region_height_px": 344,
            "arena_origin_in_canvas_x_px": 270,
            "arena_origin_in_canvas_y_px": 520,
        }
    )
    arena_digest = canonical_mapping_digest(arena_lineage)
    arena_frame = source_arena_pixel_frame_record(arena_lineage)
    arena_frame_digest = canonical_mapping_digest(arena_frame)
    arena_frame_ref = (
        f"/calibration_snapshot/arena_geometry@{PIXEL_FRAME_AUTHORITY_ATTR}"
    )
    identity_contract = build_row_identity_contract(
        domain=STIMULUS_STATE_DOMAIN,
        values=row_identity_values,
        components=row_identity_fields,
    )
    descriptor = build_canonical_coordinate_descriptor(
        profile_id="arena_relative_canvas_px.top_left_y_down.v1",
        geometry_type="point_xy",
        components=("x", "y"),
        component_units=("px", "px"),
        reference_width=344,
        reference_height=344,
        reference_authority=DigestBoundCoordinateRecordRef(
            record_ref=arena_frame_ref,
            record_sha256=arena_frame_digest,
        ),
        reference_selector="record",
        pixel_convention="continuous",
        row_identity_contract=identity_contract,
        row_identity_record_ref=(
            f"/tracking_data/chaser_states@{ROW_IDENTITY_CONTRACT_ATTR}"
        ),
        source_camera_overlay_status="not_suitable",
        frame_record=CanonicalFrameRecord(
            kind=PIXEL_FRAME_AUTHORITY_RECORD_KIND,
            record_ref=arena_frame_ref,
            record_sha256=arena_frame_digest,
        ),
    )
    with h5py.File(path, "a") as h5:
        arena = h5["/calibration_snapshot/arena_geometry"]
        _write_h5_contract_attrs(
            arena,
            {
                ARENA_GEOMETRY_RECORD_ATTR: arena_lineage,
                ARENA_GEOMETRY_RECORD_DIGEST_ATTR: arena_digest,
                PIXEL_FRAME_AUTHORITY_ATTR: arena_frame,
                PIXEL_FRAME_AUTHORITY_DIGEST_ATTR: arena_frame_digest,
            },
        )
        tracking = h5.create_group("tracking_data")
        ds = tracking.create_dataset("chaser_states", data=chaser_states)
        row_identity = tracking.create_dataset(
            STIMULUS_STATE_KEY_ARRAY_REF,
            data=row_identity_values,
        )
        source_acquisition_values = np.asarray(
            [0, 0, 1, 1] if multi_chaser else [0, 1],
            dtype="<i8",
        )
        source_acquisition = tracking.create_dataset(
            SOURCE_ACQUISITION_FRAME_INDEX_ARRAY,
            data=source_acquisition_values,
            dtype="<i8",
        )
        source_acquisition_record = {
            "schema_id": SOURCE_ACQUISITION_MAPPING_SCHEMA_ID,
            "schema_version": SOURCE_ACQUISITION_MAPPING_SCHEMA_VERSION,
            "mapping_method": "explicit_per_stimulus_state_v1",
            "source_rowset_ref": "/tracking_data/chaser_states",
            "source_row_identity_ref": (
                f"/tracking_data/{STIMULUS_STATE_KEY_ARRAY_REF}"
            ),
            "source_row_identity_sha256": identity_array_content_sha256(
                row_identity_values
            ),
            "source_row_identity_contract_sha256": identity_contract.digest(),
            "acquisition_recording_id": "recording-1",
            "acquisition_camera_id": "2010093",
            "source_total_frames": 2,
            "target_domain": "acquisition_frame_index",
            "array_ref": SOURCE_ACQUISITION_MAPPING_ARRAY_PATH,
            "array_dtype": np.dtype("<i8").str,
            "array_shape": [int(source_acquisition_values.shape[0])],
            "array_content_sha256": numpy_content_digest(
                source_acquisition_values
            ),
            "canonicalization": "canonical_json_sort_keys_v1",
        }
        _write_h5_contract_attrs(
            source_acquisition,
            {
                SOURCE_ACQUISITION_MAPPING_RECORD_ATTR: (
                    source_acquisition_record
                ),
                SOURCE_ACQUISITION_MAPPING_RECORD_DIGEST_ATTR: (
                    canonical_mapping_digest(source_acquisition_record)
                ),
            },
        )
        _write_h5_contract_attrs(
            ds,
            row_identity_contract_attrs(identity_contract),
        )
        _write_h5_contract_attrs(
            row_identity,
            row_identity_key_attrs(identity_contract),
        )
        _write_h5_contract_attrs(
            ds,
            canonical_coordinate_descriptor_v2_attrs(descriptor),
        )
        ds.attrs[COORDINATE_SURFACE_MANIFEST_ATTR] = json.dumps(
            manifest, sort_keys=True
        )
        ds.attrs[COORDINATE_SURFACE_MANIFEST_DIGEST_ATTR] = (
            canonical_mapping_digest(manifest)
        )
        if include_legacy_attrs:
            ds.attrs["coordinate_frame"] = "arena_relative_canvas_px"
            ds.attrs["coordinate_units"] = "px"
            ds.attrs["coordinate_origin"] = "top_left_of_active_arena"
            ds.attrs[
                "position_fields"
            ] = "chaser_pos_x,chaser_pos_y,target_pos_x,target_pos_y,target_clamped_pos_x,target_clamped_pos_y"
            ds.attrs["x_axis_direction"] = "right"
            ds.attrs["y_axis_direction"] = "down"


def _write_stimulus_h5_with_protocol_steps(path: Path) -> None:
    _write_stimulus_h5_with_calibration(path)
    events_dtype = np.dtype(
        [
            ("event_name", "S32"),
            ("current_step_index", np.int32),
            ("stimulus_mode_id", np.int32),
            ("camera_frame_id", np.int64),
        ]
    )
    events = np.array(
        [
            (b"STEP_START", 0, 3, 10),
            (b"STEP_END", 0, 3, 70),
            (b"STEP_START", 1, 6, 80),
            (b"STEP_END", 1, 6, 140),
        ],
        dtype=events_dtype,
    )
    protocol = {
        "steps": [
            {
                "name": "left grating",
                "stimulus_mode_str": "MOVING_GRATING",
                "duration_seconds": 1.0,
                "parameters": {
                    "type": "ProtocolMovingGratingParams",
                    "orientation_degrees": 180.0,
                    "speed_mm_per_sec": 3.5,
                    "speed_pps": 17.5,
                    "spatial_freq_cycles_per_mm": 0.2,
                    "spatial_freq_rpp": 0.04,
                    "duty_cycle": 0.5,
                },
            },
            {
                "name": "concentric center",
                "stimulus_mode_str": "CONCENTRIC_GRATING",
                "duration_seconds": 1.0,
                "parameters": {
                    "type": "ProtocolConcentricGratingParams",
                    "is_expanding": False,
                    "speed_mm_per_sec": 4.0,
                    "speed_pps": 20.0,
                    "spatial_freq_cycles_per_mm": 0.25,
                    "spatial_freq_rpp": 0.05,
                    "stimulus_role": "centering_utility",
                    "target_radius_min_mm": 8.0,
                    "target_radius_max_mm": 14.0,
                    "centering_success_fraction_threshold": 0.8,
                },
            },
        ]
    }
    with h5py.File(path, "a") as h5:
        h5.create_dataset("events", data=events)
        protocol_group = h5.create_group("protocol_snapshot")
        protocol_group.create_dataset(
            "protocol_definition_json",
            data=json.dumps(protocol).encode("utf-8"),
        )
        coords = h5.create_group("stimulus_coordinates")
        arena = coords.create_group("arena_1")
        custom = arena.create_group("custom_coordinates")
        custom.attrs["texture_center_x"] = 172.0
        custom.attrs["texture_center_y"] = 173.0


def test_import_sets_source_stimulus_video_path_when_rendered_mp4_exists(tmp_path: Path) -> None:
    h5_path = tmp_path / "session.h5"
    rendered_mp4 = tmp_path / "session.mp4"
    zarr_path = tmp_path / "sample_analysis.zarr"

    _write_minimal_stimulus_h5(h5_path)
    rendered_mp4.touch()
    zarr.open_group(str(zarr_path), mode="w")

    run_name = mod.import_stimulus_to_zarr(
        stimulus_h5=h5_path,
        zarr_path=zarr_path,
        run_name="stimulus_test",
        overwrite=False,
        verbose=False,
        repair_chaser_gaps=False,
    )

    root = zarr.open_group(str(zarr_path), mode="r")
    run_group = root["analysis"]["stimulus_runs"][run_name]
    assert run_group.attrs.get("source_h5") == str(h5_path.resolve())
    assert run_group.attrs.get("source_stimulus_video_path") == str(rendered_mp4.resolve())
    assert run_group.attrs[STIMULUS_PHYSICAL_COORDINATE_STATUS_ATTR] == (
        STIMULUS_PHYSICAL_COORDINATE_OMITTED_NO_ACQUISITION_STATUS
    )
    assert run_group.attrs[STIMULUS_PHYSICAL_COORDINATE_REASON_CODE_ATTR] == (
        STIMULUS_PHYSICAL_COORDINATE_REASON_NO_ACQUISITION
    )
    assert load_stimulus_physical_coordinate_authority(
        root,
        stimulus_run=run_name,
    ) is None
    camera = run_group["calibration/2010093"]
    assert "coordinate_frames" not in camera


def test_stimulus_run_binds_versioned_subject_and_setup_authorities(
    tmp_path: Path,
) -> None:
    h5_path = tmp_path / "session_subject.h5"
    zarr_path = tmp_path / "sample_subject_analysis.zarr"
    _write_minimal_stimulus_h5(h5_path)
    subject_metadata = {
        "subject_count": "1",
        "subject_type": "individual",
        "fish_id": "40d99fea-846b-4890-bad2-b4e152dfdde0",
        "fish_count": "35",
    }
    with h5py.File(h5_path, "a") as h5:
        subject = h5.create_group("subject_metadata")
        subject.attrs.update(subject_metadata)

    zarr.open_group(str(zarr_path), mode="w", zarr_format=3)

    run_name = mod.import_stimulus_to_zarr(
        stimulus_h5=h5_path,
        zarr_path=zarr_path,
        run_name="stimulus_subject_authority",
        overwrite=False,
        verbose=False,
        repair_chaser_gaps=False,
    )

    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    subject_authority = resolve_subject_metadata(root, allow_legacy=False)
    setup_authority = resolve_experiment_setup(root, allow_legacy=False)
    source = root[f"analysis/stimulus_runs/{run_name}/source_metadata"]
    assert source.attrs["subject_metadata_ref"] == subject_authority.group_path
    assert source.attrs["subject_metadata_sha256"] == subject_authority.record_sha256
    assert source.attrs["experiment_setup_ref"] == setup_authority.group_path
    assert source.attrs["experiment_setup_sha256"] == setup_authority.record_sha256
    assert "subject_metadata" not in source.attrs
    assert "experiment_setup" not in source.attrs


def test_import_omits_source_stimulus_video_path_when_rendered_mp4_missing(tmp_path: Path) -> None:
    h5_path = tmp_path / "session.h5"
    zarr_path = tmp_path / "sample_analysis.zarr"

    _write_minimal_stimulus_h5(h5_path)
    zarr.open_group(str(zarr_path), mode="w")

    run_name = mod.import_stimulus_to_zarr(
        stimulus_h5=h5_path,
        zarr_path=zarr_path,
        run_name="stimulus_test_no_video",
        overwrite=False,
        verbose=False,
        repair_chaser_gaps=False,
    )

    root = zarr.open_group(str(zarr_path), mode="r")
    run_group = root["analysis"]["stimulus_runs"][run_name]
    assert run_group.attrs.get("source_h5") == str(h5_path.resolve())
    assert "source_stimulus_video_path" not in run_group.attrs


def test_import_explicitly_omits_physical_authority_without_camera_scale(
    tmp_path: Path,
) -> None:
    h5_path = tmp_path / "session_no_scale.h5"
    zarr_path = tmp_path / "sample_no_scale.zarr"
    _write_minimal_stimulus_h5(h5_path)
    with h5py.File(h5_path, "a") as h5:
        arena_node = h5["/calibration_snapshot/arena_config_json"]
        arena = json.loads(bytes(arena_node[()]).decode("utf-8"))
        arena.pop("calculated_z_eff_mm", None)
        camera_record = arena["camera_calibrations"][0]
        for name in (
            "pixels_per_mm_camera",
            "pixels_per_mm_projector",
            "real_world_ref_mm",
        ):
            camera_record.pop(name, None)
        del h5["/calibration_snapshot/arena_config_json"]
        h5["/calibration_snapshot"].create_dataset(
            "arena_config_json",
            data=json.dumps(arena).encode("utf-8"),
        )
        camera = h5["/calibration_snapshot/2010093"]
        for name in (
            "pixels_per_mm_camera",
            "pixels_per_mm_projector",
            "real_world_ref_mm",
        ):
            if name in camera.attrs:
                del camera.attrs[name]
    zarr.open_group(str(zarr_path), mode="w")

    run_name = mod.import_stimulus_to_zarr(
        stimulus_h5=h5_path,
        zarr_path=zarr_path,
        run_name="stimulus_no_scale",
        overwrite=False,
        verbose=False,
        repair_chaser_gaps=False,
    )

    root = zarr.open_group(str(zarr_path), mode="r")
    run = root[f"analysis/stimulus_runs/{run_name}"]
    assert run.attrs[STIMULUS_PHYSICAL_COORDINATE_STATUS_ATTR] == (
        STIMULUS_PHYSICAL_COORDINATE_OMITTED_NO_SCALE_STATUS
    )
    assert run.attrs[STIMULUS_PHYSICAL_COORDINATE_REASON_CODE_ATTR] == (
        STIMULUS_PHYSICAL_COORDINATE_REASON_NO_SCALE
    )
    assert load_stimulus_physical_coordinate_authority(
        root,
        stimulus_run=run_name,
    ) is None
    assert "coordinate_frames" not in run["calibration/2010093"]


def test_import_rejects_statusless_present_acquisition_physical_authority(
    tmp_path: Path,
) -> None:
    h5_path = tmp_path / "session_statusless_acquisition.h5"
    zarr_path = tmp_path / "statusless_acquisition.zarr"
    _write_stimulus_h5_with_arena_relative_chaser_states(h5_path)
    _prepare_acquisition_authority(zarr_path)
    root = zarr.open_group(str(zarr_path), mode="a", use_consolidated=False)
    del root.attrs[ACQUISITION_AUTHORITY_STATUS_ATTR]
    del root["raw_video"].attrs[ACQUISITION_AUTHORITY_STATUS_ATTR]

    with pytest.raises(
        StimulusPhysicalCoordinateError,
        match="lacks exact mirrored typed publication status",
    ):
        mod.import_stimulus_to_zarr(
            stimulus_h5=h5_path,
            zarr_path=zarr_path,
            run_name="stimulus_statusless_acquisition",
            overwrite=False,
            verbose=False,
            repair_chaser_gaps=False,
        )


def test_import_creates_missing_zarr_root_after_h5_resolves(tmp_path: Path) -> None:
    h5_path = tmp_path / "session.h5"
    zarr_path = tmp_path / "created_analysis.zarr"

    _write_minimal_stimulus_h5(h5_path)

    run_name = mod.import_stimulus_to_zarr(
        stimulus_h5=h5_path,
        zarr_path=zarr_path,
        run_name="stimulus_created_root",
        overwrite=False,
        verbose=False,
        repair_chaser_gaps=False,
    )

    root = zarr.open_group(str(zarr_path), mode="r")
    assert run_name in root["analysis"]["stimulus_runs"]


def test_import_preserves_global_calibration_and_copies_run_local_snapshot(
    tmp_path: Path,
) -> None:
    h5_path = tmp_path / "session.h5"
    zarr_path = tmp_path / "sample_analysis.zarr"

    _write_stimulus_h5_with_calibration(h5_path)
    root = zarr.open_group(str(zarr_path), mode="w")
    global_calibration = root.require_group("analysis").require_group("calibration")
    global_calibration.attrs.update(
        {
            "authority": "preexisting_global_calibration",
            "sentinel": 17,
        }
    )
    global_calibration.create_array(
        "sentinel_values",
        data=np.array([11.0, 13.0], dtype=np.float64),
        chunks=(2,),
    )

    run_name = mod.import_stimulus_to_zarr(
        stimulus_h5=h5_path,
        zarr_path=zarr_path,
        run_name="stimulus_with_calibration",
        overwrite=False,
        verbose=False,
        repair_chaser_gaps=False,
    )

    root = zarr.open_group(str(zarr_path), mode="r")
    calib = root["analysis"]["calibration"]
    assert dict(calib.attrs) == {
        "authority": "preexisting_global_calibration",
        "sentinel": 17,
    }
    assert list(calib.array_keys()) == ["sentinel_values"]
    np.testing.assert_array_equal(calib["sentinel_values"][:], [11.0, 13.0])

    run_calib = root["analysis"]["stimulus_runs"][run_name]["calibration"]["2010093"]
    np.testing.assert_allclose(
        run_calib["homography_matrix"][:],
        np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 20.0], [0.0, 0.0, 1.0]]),
    )
    assert run_calib["homography_matrix"].attrs["source_frame"] == "camera_view_px"
    paths = selected_calibration_paths(
        stimulus_run=run_name,
        camera_id="2010093",
    )
    snapshot = load_selected_calibration_snapshot(
        root,
        stimulus_run=run_name,
        expected_camera_id="2010093",
        expected_from_space_id="source_camera_image_px",
        expected_to_space_id="stimulus_canvas_px",
        expected_source_reference_extent=TransformReferenceExtent(
            width=4512,
            height=4512,
            units="px",
            authority=(
                f"{paths.camera_calibration_path}"
                "@native_width_px,native_height_px"
            ),
        ),
        expected_target_reference_extent=TransformReferenceExtent(
            width=1920,
            height=1080,
            units="px",
            authority=(
                f"{paths.display_snapshot_path}@selected_output_geometry"
            ),
        ),
    )
    assert snapshot.camera_id == "2010093"
    assert snapshot.homography.transform.from_space_id == "source_camera_image_px"
    assert snapshot.homography.transform.to_space_id == "stimulus_canvas_px"
    assert snapshot.manifest.source_camera.source_h5_path == str(h5_path.resolve())
    assert snapshot.manifest.source_display.source_h5_path == str(h5_path.resolve())
    assert snapshot.manifest.source_homography.source_h5_path == str(h5_path.resolve())
    assert snapshot.manifest.source_homography.numeric_matrix_sha256 == (
        snapshot.manifest.matrix_sha256
    )
    assert "source_h5_evidence" not in run_calib
    projected_surface = run_calib["scale_models"]["projected_surface"]
    assert projected_surface.attrs["model_name"] == "projected_surface"
    np.testing.assert_array_equal(
        projected_surface["scale_image_png_buffer"][:],
        np.array([1, 2, 3], dtype=np.uint8),
    )


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("missing_numeric", "requires exact dataset.*homography_matrix"),
        ("wrong_direction", "source_frame.*camera_view_px"),
        ("wrong_camera", "camera_id.*2010093"),
        ("numeric_yaml_mismatch", "payloads disagree"),
        ("missing_display", "requires /display_snapshot"),
        ("display_extent_mismatch", "bytes and display group attrs disagree"),
        ("camera_scalar_mismatch", "camera-group attr.*disagree"),
    ],
)
def test_import_rejects_incoherent_selected_calibration_before_zarr_mutation(
    tmp_path: Path,
    mutation: str,
    expected: str,
) -> None:
    h5_path = tmp_path / f"bad_calibration_{mutation}.h5"
    zarr_path = tmp_path / f"bad_calibration_{mutation}.zarr"
    _write_minimal_stimulus_h5(h5_path)
    with h5py.File(h5_path, "a") as h5:
        camera = h5["/calibration_snapshot/2010093"]
        if mutation == "missing_numeric":
            del camera["homography_matrix"]
        elif mutation == "wrong_direction":
            camera["homography_matrix"].attrs["source_frame"] = (
                "final_display_canvas_px"
            )
        elif mutation == "wrong_camera":
            camera["homography_matrix_yml"].attrs["camera_id"] = "other_camera"
        elif mutation == "numeric_yaml_mismatch":
            yaml_node = camera["homography_matrix_yml"]
            attrs = dict(yaml_node.attrs)
            del camera["homography_matrix_yml"]
            yaml_node = camera.create_dataset(
                "homography_matrix_yml",
                data=(
                    b"homography_matrix:\n  rows: 3\n  cols: 3\n  dt: d\n"
                    b"  data: [1, 0, 11, 0, 1, 20, 0, 0, 1]\n"
                ),
            )
            yaml_node.attrs.update(attrs)
        elif mutation == "missing_display":
            del h5["/display_snapshot"]
        elif mutation == "display_extent_mismatch":
            h5["/display_snapshot"].attrs[
                "selected_output_geometry"
            ] = "1280x720+3840+0"
        else:
            camera.attrs["pixels_per_mm_camera"] = 51.0

    with pytest.raises(StimulusCoordinateContractError, match=expected):
        mod.import_stimulus_to_zarr(
            stimulus_h5=h5_path,
            zarr_path=zarr_path,
            run_name=f"stimulus_{mutation}",
            overwrite=False,
            verbose=False,
            repair_chaser_gaps=False,
        )

    assert not zarr_path.exists()


def test_import_selects_only_exact_active_camera_not_first_camera_record(
    tmp_path: Path,
) -> None:
    h5_path = tmp_path / "active_camera_not_first.h5"
    zarr_path = tmp_path / "active_camera_not_first.zarr"
    _write_minimal_stimulus_h5(h5_path)
    with h5py.File(h5_path, "a") as h5:
        node = h5["/calibration_snapshot/arena_config_json"]
        arena_config = json.loads(node[()].decode("utf-8"))
        arena_config["camera_calibrations"].insert(
            0,
            {
                "camera_id": "first_but_inactive",
                "native_width_px": 99,
                "native_height_px": 77,
                "pixels_per_mm_camera": 2.0,
                "pixels_per_mm_projector": 3.0,
                "real_world_ref_mm": 10.0,
            },
        )
        del h5["/calibration_snapshot/arena_config_json"]
        h5["/calibration_snapshot"].create_dataset(
            "arena_config_json",
            data=json.dumps(arena_config).encode("utf-8"),
        )
        h5["/calibration_snapshot"].create_group("first_but_inactive")

    run_name = mod.import_stimulus_to_zarr(
        stimulus_h5=h5_path,
        zarr_path=zarr_path,
        run_name="stimulus_exact_active_camera",
        overwrite=False,
        verbose=False,
        repair_chaser_gaps=False,
    )

    root = zarr.open_group(str(zarr_path), mode="r")
    calibration = root["analysis"]["stimulus_runs"][run_name]["calibration"]
    assert calibration.attrs["active_camera_id"] == "2010093"
    assert "2010093" in calibration
    assert "first_but_inactive" not in calibration


def test_importer_has_no_independent_transform_construction_path() -> None:
    assert not hasattr(mod, "build_directed_homography")
    assert not hasattr(mod, "stamp_directed_homography")


def test_import_cli_has_no_historical_coordinate_compatibility_flag() -> None:
    with pytest.raises(SystemExit):
        mod.parse_args(
            [
                "session.h5",
                "analysis.zarr",
                "--historical-coordinate-compatibility",
            ]
        )


def test_import_materializes_canonical_array_specific_chaser_surfaces(tmp_path: Path) -> None:
    h5_path = tmp_path / "session.h5"
    zarr_path = tmp_path / "sample_analysis.zarr"

    _write_stimulus_h5_with_arena_relative_chaser_states(h5_path)
    _prepare_acquisition_authority(zarr_path)

    run_name = mod.import_stimulus_to_zarr(
        stimulus_h5=h5_path,
        zarr_path=zarr_path,
        run_name="stimulus_group_local_coordinate_transform",
        overwrite=False,
        verbose=False,
        repair_chaser_gaps=False,
    )

    root = zarr.open_group(str(zarr_path), mode="r")
    run_group = root["analysis"]["stimulus_runs"][run_name]
    chaser_group = run_group["tracking_data"]["chaser_states"]

    physical = load_stimulus_physical_coordinate_authority(
        root,
        stimulus_run=run_name,
    )
    assert physical is not None
    assert physical.mm_per_pixel == 0.02
    assert physical.camera_id == "2010093"
    assert physical.physical_frame.record_ref == (
        f"/{run_group.path}/calibration/2010093/coordinate_frames/"
        "source_camera_physical_mm@physical_frame_calibration"
    )
    assert physical.manifest.record["selected_calibration"] == {
        "record_ref": f"/{run_group.path}/calibration@selected_calibration_manifest",
        "record_sha256": physical.selected_calibration.manifest_sha256,
    }
    assert run_group.attrs[STIMULUS_PHYSICAL_COORDINATE_STATUS_ATTR] == (
        STIMULUS_PHYSICAL_COORDINATE_BOUND_STATUS
    )
    assert run_group.attrs[STIMULUS_PHYSICAL_COORDINATE_REASON_CODE_ATTR] == (
        STIMULUS_PHYSICAL_COORDINATE_REASON_BOUND
    )

    for frame_name, record_attr in (
        ("selected_canvas", "selected_canvas_extent_record"),
        ("arena_relative_canvas", "arena_relative_extent_record"),
    ):
        frame_node = run_group["coordinate_frames"][frame_name]
        extent_record = frame_node.attrs[record_attr]
        assert frame_node.attrs["width_px"] == extent_record["width_px"]
        assert frame_node.attrs["height_px"] == extent_record["height_px"]
        assert frame_node.attrs["units"] == extent_record["units"] == "px"

    assert "coordinate_frame" not in chaser_group.attrs
    assert "coordinate_units" not in chaser_group.attrs
    assert "coordinate_origin" not in chaser_group.attrs
    assert "position_fields" not in chaser_group.attrs
    assert "coordinate_transform" not in run_group.attrs
    assert "legacy_texture_to_camera_transform" not in run_group.attrs
    assert "coordinate_transform_status" not in run_group.attrs
    assert "coordinate_descriptor" not in chaser_group.attrs
    assert chaser_group.attrs["coordinate_descriptor_status"] == "canonical"
    assert run_group.attrs["chaser_states_coordinate_descriptor_status"] == "canonical"

    np.testing.assert_array_equal(
        chaser_group[STIMULUS_STATE_KEY_ARRAY_REF][:],
        np.array([1000, 1001], dtype=np.int64),
    )
    assert "coordinate_row_identity" not in chaser_group
    assert "instance_key" not in chaser_group
    assert chaser_group[STIMULUS_STATE_KEY_ARRAY_REF].dtype == np.dtype(np.int64)
    identity = validate_stamped_row_identity(
        chaser_group,
        chaser_group[STIMULUS_STATE_KEY_ARRAY_REF],
    )
    assert identity.domain == STIMULUS_STATE_DOMAIN
    assert identity.mode == STIMULUS_STATE_KEY_MODE
    assert identity.key_array.components == ("stimulus_frame_num",)
    assert chaser_group.attrs[ROW_IDENTITY_CONTRACT_DIGEST_ATTR] == identity.digest()
    expected = {
        "chaser_position_xy": np.array([[20.0, 30.0], [21.0, 31.0]]),
        "target_position_xy": np.array([[350.0, 358.5], [340.0, 330.0]]),
        "target_clamped_position_xy": np.array([[343.0, 344.0], [340.0, 330.0]]),
    }
    for array_name, values in expected.items():
        node = chaser_group[array_name]
        np.testing.assert_allclose(node[:], values)
        descriptor = _load_output_coordinate_descriptor(chaser_group, node)
        assert descriptor.space_id == "arena_relative_canvas_px"
        assert descriptor.geometry_type == "point_xy"
        assert descriptor.components == ("x", "y")
        assert descriptor.reference_extent.width == 344
        assert descriptor.reference_extent.height == 344
        assert descriptor.row_identity.record_ref == (
            f"/{chaser_group.path}@{ROW_IDENTITY_CONTRACT_ATTR}"
        )
        assert descriptor.row_identity.record_sha256 == identity.digest()
        assert descriptor.source_camera_overlay.status == "requires_transform"
        assert len(descriptor.source_camera_overlay.transform_refs) == 2
        assert descriptor.reference_extent.authority.record_ref == (
            f"/{run_group.path}/coordinate_frames/arena_relative_canvas"
            "@pixel_frame_authority"
        )
        assert descriptor.reference_extent.authority.selector == "record"
        assert node.shape[0] == chaser_group[STIMULUS_STATE_KEY_ARRAY_REF].shape[0]
        lineage_refs = [item.record_ref for item in descriptor.lineage_refs]
        assert any(
            value.endswith("@stimulus_frame_transform_manifest")
            for value in lineage_refs
        )
        assert any(value.endswith("@coordinate_import_lineage") for value in lineage_refs)
        assert any(value.endswith("@coordinate_output_manifest") for value in lineage_refs)
        assert any(value.endswith("@camera_mapping_record") for value in lineage_refs)

    component_surfaces = {
        "chaser_position_xy": ("chaser_pos_x", "chaser_pos_y"),
        "target_position_xy": ("target_pos_x", "target_pos_y"),
        "target_clamped_position_xy": (
            "target_clamped_pos_x",
            "target_clamped_pos_y",
        ),
    }
    for surface_name, component_fields in component_surfaces.items():
        surface_descriptor = _load_output_coordinate_descriptor(
            chaser_group,
            chaser_group[surface_name],
        )
        for component, field_name in zip(("x", "y"), component_fields, strict=True):
            scalar = chaser_group[field_name]
            scalar_descriptor = _load_output_coordinate_descriptor(
                chaser_group,
                scalar,
            )
            assert scalar.attrs["coordinate_component"] == component
            assert scalar.attrs["coordinate_surface_array_ref"] == surface_name
            assert "semantic_role" not in scalar.attrs
            assert scalar.attrs["parent_semantic_role"] == chaser_group[
                surface_name
            ].attrs["semantic_role"]
            assert scalar_descriptor.geometry_type == "coordinate_component"
            assert scalar_descriptor.components == (component,)
            assert scalar_descriptor.space_id == surface_descriptor.space_id
            assert scalar_descriptor.origin == surface_descriptor.origin
            assert scalar_descriptor.positive_directions == (
                surface_descriptor.positive_directions
            )
            assert scalar_descriptor.reference_extent == (
                surface_descriptor.reference_extent
            )
            assert scalar_descriptor.row_identity == surface_descriptor.row_identity
            assert scalar_descriptor.lineage_refs == surface_descriptor.lineage_refs

    lineage = chaser_group.attrs[COORDINATE_IMPORT_LINEAGE_ATTR]
    assert chaser_group.attrs[COORDINATE_IMPORT_LINEAGE_DIGEST_ATTR] == (
        canonical_mapping_digest(lineage)
    )
    assert chaser_group.attrs[COORDINATE_OUTPUT_MANIFEST_DIGEST_ATTR] == (
        canonical_mapping_digest(chaser_group.attrs[COORDINATE_OUTPUT_MANIFEST_ATTR])
    )
    assert chaser_group.attrs[CAMERA_MAPPING_RECORD_DIGEST_ATTR] == (
        canonical_mapping_digest(chaser_group.attrs[CAMERA_MAPPING_RECORD_ATTR])
    )
    np.testing.assert_array_equal(chaser_group[CAMERA_FRAME_IDS_ARRAY][:], [2000, 2001])
    np.testing.assert_array_equal(chaser_group[SOURCE_ROW_INDICES_ARRAY][:], [0, 1])
    np.testing.assert_array_equal(
        chaser_group[SOURCE_ACQUISITION_FRAME_INDEX_ARRAY][:],
        [0, 1],
    )
    assert lineage["source_dataset_ref"] == (
        f"{h5_path.resolve()}#/tracking_data/chaser_states"
    )
    assert lineage["interpolation"]["applied"] is False
    assert lineage["source_coordinate_descriptor"]["reference_extent"][
        "authority"
    ]["record_ref"] == (
        f"/calibration_snapshot/arena_geometry@{PIXEL_FRAME_AUTHORITY_ATTR}"
    )
    assert lineage["source_arena_geometry_ref"] == (
        f"{h5_path.resolve()}#/calibration_snapshot/arena_geometry"
        f"@{ARENA_GEOMETRY_RECORD_ATTR}"
    )
    assert lineage["selected_arena_geometry_ref"] == (
        f"/{run_group.path}/calibration/arena_geometry"
        f"@{ARENA_GEOMETRY_RECORD_ATTR}"
    )
    arena = run_group["calibration"]["arena_geometry"]
    assert arena.attrs[ARENA_GEOMETRY_RECORD_ATTR] == lineage[
        "selected_arena_geometry"
    ]
    assert arena.attrs[ARENA_GEOMETRY_RECORD_DIGEST_ATTR] == lineage[
        "selected_arena_geometry_sha256"
    ]
    assert lineage["source_arena_geometry"] == lineage["selected_arena_geometry"]
    assert lineage["source_arena_geometry_sha256"] == (
        lineage["selected_arena_geometry_sha256"]
    )
    assert canonical_mapping_digest(lineage) == chaser_group.attrs[
        COORDINATE_IMPORT_LINEAGE_DIGEST_ATTR
    ]


def test_stimulus_response_authority_uses_typed_inverse_transform_direction(
    tmp_path: Path,
) -> None:
    h5_path = tmp_path / "session.h5"
    zarr_path = tmp_path / "sample_analysis.zarr"
    _write_stimulus_h5_with_arena_relative_chaser_states(h5_path)
    _prepare_acquisition_authority(zarr_path)

    run_name = mod.import_stimulus_to_zarr(
        stimulus_h5=h5_path,
        zarr_path=zarr_path,
        run_name="stimulus_response_coordinates",
        overwrite=False,
        verbose=False,
        repair_chaser_gaps=False,
    )
    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    physical = load_stimulus_physical_coordinate_authority(
        root,
        stimulus_run=run_name,
    )
    assert physical is not None

    authority = load_stimulus_response_coordinate_authority(
        root,
        stimulus_run=run_name,
        track_physical_authority=physical,
    )

    # Arena center (172, 172) is placed at canvas origin (270, 520).
    # The selected calibration is camera -> canvas translation (+10, +20),
    # so the explicit canvas -> camera inverse yields camera (432, 672).
    # The selected 50 camera px/mm scale then yields (8.64, 13.44) mm.
    np.testing.assert_allclose(authority.arena_center_mm(), (8.64, 13.44))
    np.testing.assert_allclose(
        authority.selected_canvas_to_source_camera_mm([442.0, 692.0]),
        [8.64, 13.44],
    )
    assert authority.arena_axis_extent_mm([1.0, 0.0]) == pytest.approx(3.44)
    assert authority.record["schema_id"] == (
        STIMULUS_RESPONSE_COORDINATE_LINEAGE_SCHEMA_ID
    )
    assert authority.record["schema_version"] == (
        STIMULUS_RESPONSE_COORDINATE_LINEAGE_SCHEMA_VERSION
    )
    record = dict(authority.record)
    digest = record.pop("record_sha256")
    assert canonical_mapping_digest(record) == digest


def test_stimulus_response_authority_rejects_another_run_physical_identity(
    tmp_path: Path,
) -> None:
    h5_path = tmp_path / "session.h5"
    zarr_path = tmp_path / "sample_analysis.zarr"
    _write_stimulus_h5_with_arena_relative_chaser_states(h5_path)
    _prepare_acquisition_authority(zarr_path)

    first_run = mod.import_stimulus_to_zarr(
        stimulus_h5=h5_path,
        zarr_path=zarr_path,
        run_name="stimulus_response_first",
        overwrite=False,
        verbose=False,
        repair_chaser_gaps=False,
    )
    second_run = mod.import_stimulus_to_zarr(
        stimulus_h5=h5_path,
        zarr_path=zarr_path,
        run_name="stimulus_response_second",
        overwrite=False,
        verbose=False,
        repair_chaser_gaps=False,
    )
    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    first_physical = load_stimulus_physical_coordinate_authority(
        root,
        stimulus_run=first_run,
    )
    assert first_physical is not None

    with pytest.raises(
        StimulusResponseCoordinateAuthorityError,
        match="do not share the exact source-camera physical authority",
    ):
        load_stimulus_response_coordinate_authority(
            root,
            stimulus_run=second_run,
            track_physical_authority=first_physical,
        )


def test_stimulus_physical_authority_fails_after_selected_scale_drift(
    tmp_path: Path,
) -> None:
    h5_path = tmp_path / "session.h5"
    zarr_path = tmp_path / "sample_analysis.zarr"
    _write_stimulus_h5_with_arena_relative_chaser_states(h5_path)
    _prepare_acquisition_authority(zarr_path)
    run_name = mod.import_stimulus_to_zarr(
        stimulus_h5=h5_path,
        zarr_path=zarr_path,
        run_name="stimulus_physical_drift",
        overwrite=False,
        verbose=False,
        repair_chaser_gaps=False,
    )
    root = zarr.open_group(str(zarr_path), mode="a", use_consolidated=False)
    camera = root[f"analysis/stimulus_runs/{run_name}/calibration/2010093"]
    camera.attrs["pixels_per_mm_camera"] = 25.0

    with pytest.raises(
        StimulusPhysicalCoordinateError,
        match="Selected calibration snapshot cannot be freshly rebound",
    ):
        load_stimulus_physical_coordinate_authority(
            root,
            stimulus_run=run_name,
        )


def test_stimulus_physical_loader_requires_exact_published_acquisition_status(
    tmp_path: Path,
) -> None:
    h5_path = tmp_path / "session.h5"
    zarr_path = tmp_path / "sample_analysis.zarr"
    _write_stimulus_h5_with_arena_relative_chaser_states(h5_path)
    _prepare_acquisition_authority(zarr_path)
    run_name = mod.import_stimulus_to_zarr(
        stimulus_h5=h5_path,
        zarr_path=zarr_path,
        run_name="stimulus_acquisition_status_gate",
        overwrite=False,
        verbose=False,
        repair_chaser_gaps=False,
    )
    root = zarr.open_group(str(zarr_path), mode="a", use_consolidated=False)
    raw_video = root["raw_video"]
    original = dict(root.attrs[ACQUISITION_AUTHORITY_STATUS_ATTR])

    del root.attrs[ACQUISITION_AUTHORITY_STATUS_ATTR]
    del raw_video.attrs[ACQUISITION_AUTHORITY_STATUS_ATTR]
    with pytest.raises(
        StimulusPhysicalCoordinateError,
        match="lacks exact mirrored typed publication status",
    ):
        load_stimulus_physical_coordinate_authority(
            root,
            stimulus_run=run_name,
        )

    cases = (
        (
            build_acquisition_authority_publication_status(
                status=ACQUISITION_AUTHORITY_PENDING,
                reason_code=EXTERNAL_ACQUISITION_PENDING_REASON,
                authority_mode=EXTERNAL_ACQUISITION_AUTHORITY_MODE,
                authority_path="analysis/acquisition_camera_frames/2010093",
            ).to_dict(),
            "requires a published acquisition authority",
        ),
        (
            build_acquisition_authority_publication_status(
                status=ACQUISITION_AUTHORITY_NOT_PUBLISHED,
                reason_code="organized_recording_identity_absent",
            ).to_dict(),
            "requires a published acquisition authority",
        ),
        (
            build_acquisition_authority_publication_status(
                status=ACQUISITION_AUTHORITY_PUBLISHED,
                reason_code=MATERIALIZED_ACQUISITION_PUBLISHED_REASON,
                authority_mode=MATERIALIZED_ACQUISITION_AUTHORITY_MODE,
                authority_path="analysis/acquisition_camera_frames/2010093",
            ).to_dict(),
            "mode/path disagrees",
        ),
        (
            build_acquisition_authority_publication_status(
                status=ACQUISITION_AUTHORITY_PUBLISHED,
                reason_code=EXTERNAL_ACQUISITION_PUBLISHED_REASON,
                authority_mode=EXTERNAL_ACQUISITION_AUTHORITY_MODE,
                authority_path="analysis/acquisition_camera_frames/other-camera",
            ).to_dict(),
            "mode/path disagrees",
        ),
    )
    for status_record, error_match in cases:
        root.attrs[ACQUISITION_AUTHORITY_STATUS_ATTR] = status_record
        raw_video.attrs[ACQUISITION_AUTHORITY_STATUS_ATTR] = status_record
        with pytest.raises(StimulusPhysicalCoordinateError, match=error_match):
            load_stimulus_physical_coordinate_authority(
                root,
                stimulus_run=run_name,
            )

    root.attrs[ACQUISITION_AUTHORITY_STATUS_ATTR] = original
    raw_video.attrs[ACQUISITION_AUTHORITY_STATUS_ATTR] = original
    assert (
        load_stimulus_physical_coordinate_authority(
            root,
            stimulus_run=run_name,
        )
        is not None
    )


def test_imported_stimulus_round_trips_through_exact_chaser_handoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h5_path = tmp_path / "roundtrip.h5"
    zarr_path = tmp_path / "roundtrip.zarr"
    _write_stimulus_h5_with_arena_relative_chaser_states(h5_path)
    _prepare_acquisition_authority(zarr_path)

    selected_calibration_checks = 0
    original = SelectedCalibrationSnapshot.assert_verified

    def counted(value: SelectedCalibrationSnapshot) -> None:
        nonlocal selected_calibration_checks
        selected_calibration_checks += 1
        original(value)

    monkeypatch.setattr(SelectedCalibrationSnapshot, "assert_verified", counted)

    run_name = mod.import_stimulus_to_zarr(
        stimulus_h5=h5_path,
        zarr_path=zarr_path,
        run_name="stimulus_roundtrip",
        overwrite=False,
        verbose=False,
        repair_chaser_gaps=False,
    )
    assert selected_calibration_checks == 26
    bundle = chaser_metrics_loader.load_chaser_metrics(
        zarr_path,
        stimulus_run=run_name,
    )

    np.testing.assert_array_equal(bundle.camera_frame_ids, [2000, 2001])
    np.testing.assert_allclose(
        bundle.online["target_position_xy"],
        [[350.0, 358.5], [340.0, 330.0]],
    )
    json.dumps(bundle.online_coordinate_metadata, allow_nan=False, sort_keys=True)
    handoff = bundle.online_coordinate_handoff
    assert handoff is not None
    handoff.assert_verified()
    assert handoff.import_lineage.attr_name == COORDINATE_IMPORT_LINEAGE_ATTR
    assert handoff.output_manifest.attr_name == COORDINATE_OUTPUT_MANIFEST_ATTR
    assert handoff.camera_mapping.attr_name == CAMERA_MAPPING_RECORD_ATTR
    np.testing.assert_array_equal(handoff.camera_frame_ids, [2000, 2001])
    np.testing.assert_array_equal(handoff.source_row_indices, [0, 1])
    np.testing.assert_array_equal(
        handoff.source_acquisition_frame_index,
        [0, 1],
    )
    assert len(handoff.frame_transform.transform_chain.transform_records) == 2


def test_stimulus_activation_rechecks_calibration_after_lease_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h5_path = tmp_path / "activation_calibration_drift.h5"
    zarr_path = tmp_path / "activation_calibration_drift.zarr"
    run_name = "stimulus_activation_calibration_drift"
    _write_stimulus_h5_with_arena_relative_chaser_states(h5_path)
    _prepare_acquisition_authority(zarr_path)
    original_write = mod._write_stimulus_activation_attr
    drift_injected = False

    def drift_after_lease(attrs, name, value):
        nonlocal drift_injected
        original_write(attrs, name, value)
        if name == mod.STIMULUS_PARENT_PUBLICATION_LEASE_ATTR and not drift_injected:
            drift_injected = True
            root = zarr.open_group(str(zarr_path), mode="a", use_consolidated=False)
            calibration = root[
                f"analysis/stimulus_runs/{run_name}/calibration"
            ]
            calibration.attrs[ACTIVE_CAMERA_ID_ATTR] = "tampered-camera"

    monkeypatch.setattr(mod, "_write_stimulus_activation_attr", drift_after_lease)

    with pytest.raises(
        StimulusCoordinateContractError,
        match="cannot be rebound.*active_camera_id.*mismatch",
    ):
        mod.import_stimulus_to_zarr(
            stimulus_h5=h5_path,
            zarr_path=zarr_path,
            run_name=run_name,
            overwrite=False,
            verbose=False,
            repair_chaser_gaps=False,
        )

    assert drift_injected is True
    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    run = root[f"analysis/stimulus_runs/{run_name}"]
    assert run.attrs["palette_run_completion_status"] == "failed"
    assert run.attrs["stage_selector_eligible"] is False


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("fractional_camera", "fractional mappings are forbidden"),
        ("duplicate_stimulus", "duplicate stimulus-frame mappings"),
    ],
)
def test_import_rejects_ambiguous_or_fractional_camera_mapping_before_publish(
    tmp_path: Path,
    mutation: str,
    expected: str,
) -> None:
    h5_path = tmp_path / f"{mutation}.h5"
    zarr_path = tmp_path / f"{mutation}.zarr"
    run_name = f"stimulus_{mutation}"
    _write_stimulus_h5_with_arena_relative_chaser_states(h5_path)
    _prepare_acquisition_authority(zarr_path)
    with h5py.File(h5_path, "a") as h5:
        source = h5["/video_metadata/frame_metadata"]
        attrs = dict(source.attrs)
        values = source[:]
        if mutation == "fractional_camera":
            replacement = np.empty(
                values.shape,
                dtype=np.dtype(
                    [
                        ("stimulus_frame_num", np.uint64),
                        ("triggering_camera_frame_id", np.float64),
                        ("timestamp_ns", np.int64),
                    ]
                ),
            )
            replacement["stimulus_frame_num"] = values["stimulus_frame_num"]
            replacement["triggering_camera_frame_id"] = [2000.0, 2001.5]
            replacement["timestamp_ns"] = values["timestamp_ns"]
            values = replacement
        else:
            values["stimulus_frame_num"][1] = values["stimulus_frame_num"][0]
        del h5["/video_metadata/frame_metadata"]
        destination = h5["/video_metadata"].create_dataset(
            "frame_metadata",
            data=values,
        )
        destination.attrs.update(attrs)

    with pytest.raises(StimulusCoordinateContractError, match=expected):
        mod.import_stimulus_to_zarr(
            stimulus_h5=h5_path,
            zarr_path=zarr_path,
            run_name=run_name,
            overwrite=False,
            verbose=False,
            repair_chaser_gaps=False,
        )

    root = zarr.open_group(str(zarr_path), mode="r")
    runs = root["analysis"]["stimulus_runs"]
    assert runs[run_name].attrs["palette_run_completion_status"] == "failed"
    assert runs.attrs.get("latest") != run_name
    assert runs.attrs.get("latest_complete") != run_name


def test_import_preserves_duplicate_external_camera_ids_without_using_them_as_time(
    tmp_path: Path,
) -> None:
    h5_path = tmp_path / "duplicate_external_camera.h5"
    zarr_path = tmp_path / "duplicate_external_camera.zarr"
    _write_stimulus_h5_with_arena_relative_chaser_states(h5_path)
    with h5py.File(h5_path, "a") as h5:
        source = h5["/video_metadata/frame_metadata"]
        attrs = dict(source.attrs)
        values = source[:]
        values["triggering_camera_frame_id"][:] = 2000
        del h5["/video_metadata/frame_metadata"]
        destination = h5["/video_metadata"].create_dataset(
            "frame_metadata",
            data=values,
        )
        destination.attrs.update(attrs)
    _prepare_acquisition_authority(zarr_path)

    run_name = mod.import_stimulus_to_zarr(
        stimulus_h5=h5_path,
        zarr_path=zarr_path,
        run_name="stimulus_duplicate_external_camera",
        overwrite=False,
        verbose=False,
        repair_chaser_gaps=False,
    )

    root = zarr.open_group(str(zarr_path), mode="r")
    run = root["analysis/stimulus_runs"][run_name]
    chaser = run["tracking_data/chaser_states"]
    np.testing.assert_array_equal(chaser[CAMERA_FRAME_IDS_ARRAY][:], [2000, 2000])
    np.testing.assert_array_equal(
        chaser[SOURCE_ACQUISITION_FRAME_INDEX_ARRAY][:],
        [0, 1],
    )
    _, _, handoff = chaser_metrics_loader.load_canonical_online_coordinate_surface(
        root,
        run,
        chaser,
    )
    np.testing.assert_array_equal(handoff.camera_frame_ids, [2000, 2000])
    np.testing.assert_array_equal(
        handoff.source_acquisition_frame_index,
        [0, 1],
    )
    handoff.assert_verified()


def test_import_rejects_descriptor_free_chaser_and_does_not_publish_latest(
    tmp_path: Path,
) -> None:
    h5_path = tmp_path / "session.h5"
    zarr_path = tmp_path / "sample_analysis.zarr"
    _write_stimulus_h5_with_arena_relative_chaser_states(h5_path)
    with h5py.File(h5_path, "a") as h5:
        del h5["/tracking_data/chaser_states"].attrs["coordinate_descriptor"]
        del h5["/tracking_data/chaser_states"].attrs["coordinate_descriptor_sha256"]
    root = zarr.open_group(str(zarr_path), mode="w")
    global_calibration = root.require_group("analysis").require_group("calibration")
    global_calibration.attrs["sentinel"] = "must_remain_unchanged"

    with pytest.raises(StimulusCoordinateContractError):
        mod.import_stimulus_to_zarr(
            stimulus_h5=h5_path,
            zarr_path=zarr_path,
            run_name="stimulus_incomplete_chaser_coordinates",
            overwrite=False,
            verbose=False,
            repair_chaser_gaps=False,
        )

    root = zarr.open_group(str(zarr_path), mode="r")
    analysis = root["analysis"]
    assert dict(analysis["calibration"].attrs) == {
        "sentinel": "must_remain_unchanged"
    }
    assert "stimulus_runs" not in analysis


def test_import_accepts_canonical_only_source_without_legacy_adapter(tmp_path: Path) -> None:
    h5_path = tmp_path / "session.h5"
    zarr_path = tmp_path / "sample_analysis.zarr"
    _write_stimulus_h5_with_arena_relative_chaser_states(
        h5_path,
        include_legacy_attrs=False,
    )
    _prepare_acquisition_authority(zarr_path)

    run_name = mod.import_stimulus_to_zarr(
        stimulus_h5=h5_path,
        zarr_path=zarr_path,
        run_name="stimulus_canonical_only",
        overwrite=False,
        verbose=False,
        repair_chaser_gaps=False,
    )

    root = zarr.open_group(str(zarr_path), mode="r")
    run_group = root["analysis"]["stimulus_runs"][run_name]
    chaser_group = run_group["tracking_data"]["chaser_states"]
    assert run_group.attrs["palette_run_completion_status"] == "complete"
    assert run_group.attrs["import_version"] == STIMULUS_IMPORT_VERSION
    assert run_group.attrs["coordinate_contract_epoch"] == COORDINATE_CONTRACT_EPOCH
    assert "target_position_xy" in chaser_group
    assert "coordinate_transform" not in run_group.attrs
    assert "legacy_texture_to_camera_transform" not in run_group.attrs
    assert not any(
        key in chaser_group.attrs
        for key in (
            "coordinate_frame",
            "coordinate_units",
            "coordinate_origin",
            "position_fields",
            "x_axis_direction",
            "y_axis_direction",
            "pixel_convention",
        )
    )


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("missing", "requires persisted /tracking_data/stimulus_state_key"),
        ("value_mismatch", "values do not exactly equal"),
        ("stale_digest", "row-identity metadata is invalid"),
        ("missing_key_attr", "attrs must be exactly"),
    ],
)
def test_import_requires_exact_digest_bound_source_row_identity(
    tmp_path: Path,
    mutation: str,
    expected: str,
) -> None:
    h5_path = tmp_path / f"row_identity_{mutation}.h5"
    zarr_path = tmp_path / f"row_identity_{mutation}.zarr"
    _write_stimulus_h5_with_arena_relative_chaser_states(h5_path)
    with h5py.File(h5_path, "a") as h5:
        path = "/tracking_data/stimulus_state_key"
        if mutation == "missing":
            del h5[path]
        elif mutation == "value_mismatch":
            h5[path][0] = 999
        elif mutation == "stale_digest":
            key_record = json.loads(h5[path].attrs[ROW_IDENTITY_KEY_ATTR])
            key_record["content_sha256"] = "0" * 64
            h5[path].attrs[ROW_IDENTITY_KEY_ATTR] = json.dumps(
                key_record,
                sort_keys=True,
            )
        else:
            del h5[path].attrs[ROW_IDENTITY_KEY_DIGEST_ATTR]

    with pytest.raises(StimulusCoordinateContractError, match=expected):
        mod.import_stimulus_to_zarr(
            stimulus_h5=h5_path,
            zarr_path=zarr_path,
            run_name=f"stimulus_row_identity_{mutation}",
            overwrite=False,
            verbose=False,
            repair_chaser_gaps=False,
        )

    assert not zarr_path.exists()


def test_normal_import_rejects_historical_coordinate_row_identity_adapter(
    tmp_path: Path,
) -> None:
    h5_path = tmp_path / "historical_rows.h5"
    zarr_path = tmp_path / "historical_rows.zarr"
    _write_stimulus_h5_with_arena_relative_chaser_states(h5_path)
    with h5py.File(h5_path, "a") as h5:
        tracking = h5["/tracking_data"]
        values = tracking[STIMULUS_STATE_KEY_ARRAY_REF][:]
        del tracking[STIMULUS_STATE_KEY_ARRAY_REF]
        tracking.create_dataset("coordinate_row_identity", data=values)

    with pytest.raises(
        StimulusCoordinateContractError,
        match="does not consume coordinate_row_identity",
    ):
        mod.import_stimulus_to_zarr(
            stimulus_h5=h5_path,
            zarr_path=zarr_path,
            run_name="stimulus_historical_rows",
            overwrite=False,
            verbose=False,
            repair_chaser_gaps=False,
        )

    assert not zarr_path.exists()


def test_import_accepts_explicitly_classified_non_spatial_future_field(
    tmp_path: Path,
) -> None:
    h5_path = tmp_path / "classified_future_field.h5"
    zarr_path = tmp_path / "classified_future_field.zarr"
    _write_stimulus_h5_with_arena_relative_chaser_states(h5_path)
    with h5py.File(h5_path, "a") as h5:
        source = h5["/tracking_data/chaser_states"]
        attrs = dict(source.attrs)
        values = source[:]
        extended = np.empty(
            values.shape,
            dtype=np.dtype([*values.dtype.descr, ("future_state_code", "<i4")]),
        )
        for field_name in values.dtype.names or ():
            extended[field_name] = values[field_name]
        extended["future_state_code"] = np.array([7, 8], dtype=np.int32)
        manifest = json.loads(attrs[COORDINATE_SURFACE_MANIFEST_ATTR])
        manifest["field_classifications"]["future_state_code"] = "non_spatial"
        attrs[COORDINATE_SURFACE_MANIFEST_ATTR] = json.dumps(
            manifest,
            sort_keys=True,
        )
        attrs[COORDINATE_SURFACE_MANIFEST_DIGEST_ATTR] = (
            canonical_mapping_digest(manifest)
        )
        del h5["/tracking_data/chaser_states"]
        destination = h5["/tracking_data"].create_dataset(
            "chaser_states",
            data=extended,
        )
        destination.attrs.update(attrs)

    _prepare_acquisition_authority(zarr_path)
    run_name = mod.import_stimulus_to_zarr(
        stimulus_h5=h5_path,
        zarr_path=zarr_path,
        run_name="stimulus_classified_future_field",
        overwrite=False,
        verbose=False,
        repair_chaser_gaps=False,
    )
    root = zarr.open_group(str(zarr_path), mode="r")
    chaser = root["analysis"]["stimulus_runs"][run_name]["tracking_data"][
        "chaser_states"
    ]
    np.testing.assert_array_equal(chaser["future_state_code"][:], [7, 8])
    assert chaser.attrs[COORDINATE_SURFACE_MANIFEST_ATTR][
        "field_classifications"
    ]["future_state_code"] == "non_spatial"


def test_import_rejects_overwrite_of_published_latest_run(tmp_path: Path) -> None:
    h5_path = tmp_path / "session.h5"
    zarr_path = tmp_path / "analysis.zarr"
    run_name = "immutable_published_stimulus"
    _write_stimulus_h5_with_arena_relative_chaser_states(h5_path)
    _prepare_acquisition_authority(zarr_path)

    mod.import_stimulus_to_zarr(
        stimulus_h5=h5_path,
        zarr_path=zarr_path,
        run_name=run_name,
        overwrite=False,
        verbose=False,
        repair_chaser_gaps=False,
    )
    before = zarr.open_group(str(zarr_path), mode="r")
    before_run = before["analysis"]["stimulus_runs"][run_name]
    created_at = before_run.attrs["created_at_utc"]
    target_values = before_run["tracking_data"]["chaser_states"][
        "target_position_xy"
    ][:]

    with pytest.raises(ValueError, match="Refusing to overwrite published stimulus run"):
        mod.import_stimulus_to_zarr(
            stimulus_h5=h5_path,
            zarr_path=zarr_path,
            run_name=run_name,
            overwrite=True,
            verbose=False,
            repair_chaser_gaps=False,
        )

    root = zarr.open_group(str(zarr_path), mode="r")
    runs = root["analysis"]["stimulus_runs"]
    run_group = runs[run_name]
    assert runs.attrs["latest"] == run_name
    assert runs.attrs["latest_complete"] == run_name
    assert run_group.attrs["palette_run_completion_status"] == "complete"
    assert run_group.attrs["stage_selector_eligible"] is True
    assert run_group.attrs[mod.STIMULUS_PUBLICATION_OWNER_ATTR]
    assert (
        runs.attrs[mod.STIMULUS_PUBLICATION_POLICY_ATTR]
        == mod.STIMULUS_PUBLICATION_POLICY
    )
    assert runs.attrs[mod.STIMULUS_PUBLICATION_GENERATION_ATTR] == 1
    assert runs.attrs[mod.STIMULUS_PARENT_PUBLICATION_LEASE_ATTR]["run_path"] == (
        f"analysis/stimulus_runs/{run_name}"
    )
    assert run_group.attrs["created_at_utc"] == created_at
    np.testing.assert_array_equal(
        run_group["tracking_data"]["chaser_states"]["target_position_xy"][:],
        target_values,
    )


def test_import_preserves_hostile_concurrent_selector_takeover(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h5_path = tmp_path / "concurrent_takeover.h5"
    zarr_path = tmp_path / "concurrent_takeover.zarr"
    run_name = "stimulus_concurrent_takeover"
    _write_stimulus_h5_with_arena_relative_chaser_states(h5_path)
    _prepare_acquisition_authority(zarr_path)
    root = zarr.open_group(str(zarr_path), mode="a", use_consolidated=False)
    runs = root["analysis"].require_group("stimulus_runs")
    runs.attrs["latest"] = "prior"
    runs.attrs["latest_complete"] = "prior"
    original_write = mod._write_stimulus_activation_attr
    takeover_injected = False

    def hostile_write(attrs, name, value):
        nonlocal takeover_injected
        original_write(attrs, name, value)
        if name == "latest_complete" and not takeover_injected:
            takeover_injected = True
            runs.attrs["latest"] = "alien-concurrent-run"

    monkeypatch.setattr(mod, "_write_stimulus_activation_attr", hostile_write)

    with pytest.raises(RuntimeError, match="lost exact ownership"):
        mod.import_stimulus_to_zarr(
            stimulus_h5=h5_path,
            zarr_path=zarr_path,
            run_name=run_name,
            overwrite=False,
            verbose=False,
            repair_chaser_gaps=False,
        )

    reloaded = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    reloaded_runs = reloaded["analysis/stimulus_runs"]
    failed = reloaded_runs[run_name]
    assert reloaded_runs.attrs["latest"] == "alien-concurrent-run"
    assert reloaded_runs.attrs["latest_complete"] == "prior"
    assert failed.attrs["palette_run_completion_status"] == "failed"
    assert failed.attrs["stage_selector_eligible"] is False


def test_import_rejects_overwrite_of_historical_complete_run(tmp_path: Path) -> None:
    h5_path = tmp_path / "session.h5"
    zarr_path = tmp_path / "analysis.zarr"
    historical_run = "historical_complete_stimulus"
    latest_run = "newer_complete_stimulus"
    _write_minimal_stimulus_h5(h5_path)

    for run_name in (historical_run, latest_run):
        mod.import_stimulus_to_zarr(
            stimulus_h5=h5_path,
            zarr_path=zarr_path,
            run_name=run_name,
            overwrite=False,
            verbose=False,
            repair_chaser_gaps=False,
        )

    with pytest.raises(ValueError, match="Refusing to overwrite non-failed stimulus run"):
        mod.import_stimulus_to_zarr(
            stimulus_h5=h5_path,
            zarr_path=zarr_path,
            run_name=historical_run,
            overwrite=True,
            verbose=False,
            repair_chaser_gaps=False,
        )

    root = zarr.open_group(str(zarr_path), mode="r")
    runs = root["analysis"]["stimulus_runs"]
    assert runs.attrs["latest"] == latest_run
    assert runs[historical_run].attrs["palette_run_completion_status"] == "complete"
    assert runs[latest_run].attrs["palette_run_completion_status"] == "complete"


def test_import_never_reuses_failed_public_tombstone(tmp_path: Path) -> None:
    h5_path = tmp_path / "failed_retry.h5"
    zarr_path = tmp_path / "failed_retry.zarr"
    run_name = "failed_stimulus_tombstone"
    _write_minimal_stimulus_h5(h5_path)
    root = zarr.open_group(str(zarr_path), mode="w", use_consolidated=False)
    runs = root.require_group("analysis").require_group("stimulus_runs")
    failed = runs.create_group(
        run_name,
        attributes={
            "palette_run_completion_status": "failed",
            "stage_selector_eligible": False,
            mod.STIMULUS_PUBLICATION_OWNER_ATTR: "existing-owner",
            "sentinel": "preserve",
        },
    )

    with pytest.raises(ValueError, match="failed public children are immutable"):
        mod.import_stimulus_to_zarr(
            stimulus_h5=h5_path,
            zarr_path=zarr_path,
            run_name=run_name,
            overwrite=True,
            verbose=False,
            repair_chaser_gaps=False,
        )

    assert runs[run_name].attrs["sentinel"] == "preserve"
    assert runs[run_name].attrs[mod.STIMULUS_PUBLICATION_OWNER_ATTR] == (
        "existing-owner"
    )
    assert failed.path == runs[run_name].path


def test_postcommit_stimulus_log_is_nonthrowing() -> None:
    class HostileConsole:
        def log(self, _message: str) -> None:
            raise RuntimeError("injected postcommit logger failure")

    mod._log_after_commit(HostileConsole(), "committed")


def test_import_marks_arbitrary_post_start_exception_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h5_path = tmp_path / "session.h5"
    zarr_path = tmp_path / "analysis.zarr"
    run_name = "stimulus_generic_failure"
    _write_minimal_stimulus_h5(h5_path)
    root = zarr.open_group(str(zarr_path), mode="w")
    root.attrs.update(
        {
            "experiment_setup": {"sentinel": "root"},
            "experimental_chamber": "root_chamber",
            "dish_design": "root_dish",
        }
    )
    global_calibration = root.require_group("analysis").require_group("calibration")
    global_calibration.attrs["sentinel"] = "unchanged"
    global_metadata = root.require_group("analysis_metadata")
    global_metadata.attrs["sentinel"] = "unchanged"
    global_enums = root["analysis"].require_group("enums")
    global_enums.attrs["sentinel"] = "unchanged"

    def _raise_after_start(*_args, **_kwargs):
        raise RuntimeError("synthetic post-start failure")

    monkeypatch.setattr(
        mod,
        "_materialize_selected_calibration_snapshot",
        _raise_after_start,
    )
    with pytest.raises(RuntimeError, match="synthetic post-start failure"):
        mod.import_stimulus_to_zarr(
            stimulus_h5=h5_path,
            zarr_path=zarr_path,
            run_name=run_name,
            overwrite=False,
            verbose=False,
            repair_chaser_gaps=False,
        )

    root = zarr.open_group(str(zarr_path), mode="r")
    analysis = root["analysis"]
    runs = analysis["stimulus_runs"]
    failed = runs[run_name]
    assert failed.attrs["palette_run_completion_status"] == "failed"
    assert failed.attrs["stage_selector_eligible"] is False
    assert failed.attrs["palette_run_error"] == "synthetic post-start failure"
    assert failed.attrs[mod.STIMULUS_PUBLICATION_OWNER_ATTR]
    tombstone = failed.attrs[mod.STIMULUS_PUBLICATION_TOMBSTONE_ATTR]
    assert tombstone["public_path_retained"] is True
    assert tombstone["retry_policy"] == "new_immutable_run_name_required"
    assert runs.attrs.get("latest") != run_name
    assert runs.attrs.get("latest_complete") != run_name
    assert runs.attrs.get("authoritative_run") != run_name
    assert dict(analysis["calibration"].attrs) == {"sentinel": "unchanged"}
    assert dict(root["analysis_metadata"].attrs) == {"sentinel": "unchanged"}
    assert dict(analysis["enums"].attrs) == {"sentinel": "unchanged"}
    assert dict(root.attrs) == {
        "experiment_setup": {"sentinel": "root"},
        "experimental_chamber": "root_chamber",
        "dish_design": "root_dish",
    }


def test_import_persist_then_raise_create_recovers_owned_tombstone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h5_path = tmp_path / "stimulus_create_ambiguous.h5"
    zarr_path = tmp_path / "stimulus_create_ambiguous.zarr"
    run_name = "stimulus_create_ambiguous"
    _write_minimal_stimulus_h5(h5_path)
    root = zarr.open_group(str(zarr_path), mode="a", use_consolidated=False)
    runs = root.require_group("analysis/stimulus_runs")
    runs.attrs["latest_pending"] = run_name
    original_create = mod._create_stimulus_public_candidate

    def persist_then_raise(parent, *, run_name, publication_owner_uuid):
        original_create(
            parent,
            run_name=run_name,
            publication_owner_uuid=publication_owner_uuid,
        )
        raise RuntimeError("injected stimulus create acknowledgement loss")

    monkeypatch.setattr(
        mod,
        "_create_stimulus_public_candidate",
        persist_then_raise,
    )

    with pytest.raises(RuntimeError, match="create acknowledgement loss"):
        mod.import_stimulus_to_zarr(
            stimulus_h5=h5_path,
            zarr_path=zarr_path,
            run_name=run_name,
            overwrite=False,
            verbose=False,
            repair_chaser_gaps=False,
        )

    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    failed = root[f"analysis/stimulus_runs/{run_name}"]
    owner = failed.attrs[mod.STIMULUS_PUBLICATION_OWNER_ATTR]
    tombstone = failed.attrs[mod.STIMULUS_PUBLICATION_TOMBSTONE_ATTR]
    assert failed.attrs["palette_run_completion_status"] == "failed"
    assert failed.attrs["stage_selector_eligible"] is False
    assert tombstone["publication_owner_uuid"] == owner
    assert tombstone["run_name"] == run_name
    assert root["analysis/stimulus_runs"].attrs["latest_pending"] == run_name


def test_import_failure_cleanup_never_clobbers_recreated_successor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h5_path = tmp_path / "stimulus_cleanup_takeover.h5"
    zarr_path = tmp_path / "stimulus_cleanup_takeover.zarr"
    run_name = "stimulus_cleanup_takeover"
    _write_minimal_stimulus_h5(h5_path)
    root = zarr.open_group(str(zarr_path), mode="w", use_consolidated=False)
    parent = root.require_group("analysis/stimulus_runs")
    original_write = mod._write_stimulus_failure_attr
    takeover_injected = False

    monkeypatch.setattr(
        mod,
        "_materialize_selected_calibration_snapshot",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("injected stimulus source failure")
        ),
    )

    def hostile_write(attrs, name, value):
        nonlocal takeover_injected
        original_write(attrs, name, value)
        if name == "stage_selector_eligible" and not takeover_injected:
            takeover_injected = True
            del parent[run_name]
            parent.create_group(
                run_name,
                attributes={
                    mod.STIMULUS_PUBLICATION_OWNER_ATTR: "alien-stimulus-owner",
                    "palette_run_completion_status": "complete",
                    "stage_selector_eligible": True,
                    "sentinel": "successor-preserved",
                },
            )

    monkeypatch.setattr(mod, "_write_stimulus_failure_attr", hostile_write)

    with pytest.raises(RuntimeError, match="injected stimulus source failure"):
        mod.import_stimulus_to_zarr(
            stimulus_h5=h5_path,
            zarr_path=zarr_path,
            run_name=run_name,
            overwrite=False,
            verbose=False,
            repair_chaser_gaps=False,
        )

    reloaded = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    successor = reloaded[f"analysis/stimulus_runs/{run_name}"]
    assert takeover_injected is True
    assert successor.attrs[mod.STIMULUS_PUBLICATION_OWNER_ATTR] == (
        "alien-stimulus-owner"
    )
    assert successor.attrs["palette_run_completion_status"] == "complete"
    assert successor.attrs["stage_selector_eligible"] is True
    assert successor.attrs["sentinel"] == "successor-preserved"
    assert mod.STIMULUS_PUBLICATION_TOMBSTONE_ATTR not in successor.attrs


def test_import_invalidates_physical_authority_when_final_h5_reverify_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h5_path = tmp_path / "physical_reverify_failure.h5"
    zarr_path = tmp_path / "physical_reverify_failure.zarr"
    run_name = "stimulus_physical_reverify_failure"
    _write_stimulus_h5_with_arena_relative_chaser_states(h5_path)
    _prepare_acquisition_authority(zarr_path)
    root = zarr.open_group(str(zarr_path), mode="a", use_consolidated=False)
    runs = root["analysis"].require_group("stimulus_runs")
    pointer_snapshot = {
        "latest": "previous_latest",
        "latest_complete": "previous_complete",
        "latest_pending": "previous_pending",
        "authoritative_run": "previous_authority",
        "authoritative_run_provenance": {"sentinel": "unchanged"},
    }
    runs.attrs.update(pointer_snapshot)

    def _fail_final_reverify(*_args, **_kwargs):
        raise RuntimeError("synthetic final H5 reverify failure")

    monkeypatch.setattr(
        mod,
        "reverify_stimulus_coordinate_contract",
        _fail_final_reverify,
    )
    with pytest.raises(RuntimeError, match="synthetic final H5 reverify failure"):
        mod.import_stimulus_to_zarr(
            stimulus_h5=h5_path,
            zarr_path=zarr_path,
            run_name=run_name,
            overwrite=False,
            verbose=False,
            repair_chaser_gaps=False,
        )

    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    runs = root["analysis/stimulus_runs"]
    failed = runs[run_name]
    assert failed.attrs["palette_run_completion_status"] == "failed"
    assert failed.attrs[STIMULUS_PHYSICAL_COORDINATE_STATUS_ATTR] == (
        STIMULUS_PHYSICAL_COORDINATE_INVALIDATED_STATUS
    )
    assert failed.attrs[STIMULUS_PHYSICAL_COORDINATE_REASON_CODE_ATTR] == (
        STIMULUS_PHYSICAL_COORDINATE_REASON_PARENT_RUN_FAILED
    )
    assert STIMULUS_PHYSICAL_COORDINATE_MANIFEST_REF_ATTR not in failed.attrs
    assert STIMULUS_PHYSICAL_COORDINATE_MANIFEST_SHA256_ATTR not in failed.attrs
    for name, value in pointer_snapshot.items():
        assert runs.attrs[name] == value
    with pytest.raises(
        StimulusPhysicalCoordinateError,
        match="requires parent run status 'complete'",
    ):
        load_stimulus_physical_coordinate_authority(
            root,
            stimulus_run=run_name,
        )


def test_import_rolls_back_parent_pointers_when_postcompletion_reload_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h5_path = tmp_path / "physical_reload_failure.h5"
    zarr_path = tmp_path / "physical_reload_failure.zarr"
    run_name = "stimulus_physical_reload_failure"
    _write_stimulus_h5_with_arena_relative_chaser_states(h5_path)
    _prepare_acquisition_authority(zarr_path)
    root = zarr.open_group(str(zarr_path), mode="a", use_consolidated=False)
    runs = root["analysis"].require_group("stimulus_runs")
    pointer_snapshot = {
        "latest": "previous_latest",
        "latest_complete": "previous_complete",
        "latest_pending": "previous_pending",
        "authoritative_run": "previous_authority",
        "authoritative_run_provenance": {"sentinel": "unchanged"},
    }
    runs.attrs.update(pointer_snapshot)

    def _fail_public_reload(*_args, **_kwargs):
        raise RuntimeError("synthetic postcompletion authority reload failure")

    monkeypatch.setattr(
        mod,
        "_load_stimulus_physical_coordinate_authority_before_selection",
        _fail_public_reload,
    )
    with pytest.raises(
        RuntimeError,
        match="synthetic postcompletion authority reload failure",
    ):
        mod.import_stimulus_to_zarr(
            stimulus_h5=h5_path,
            zarr_path=zarr_path,
            run_name=run_name,
            overwrite=False,
            verbose=False,
            repair_chaser_gaps=False,
        )

    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    runs = root["analysis/stimulus_runs"]
    failed = runs[run_name]
    assert failed.attrs["palette_run_completion_status"] == "failed"
    assert failed.attrs[STIMULUS_PHYSICAL_COORDINATE_STATUS_ATTR] == (
        STIMULUS_PHYSICAL_COORDINATE_INVALIDATED_STATUS
    )
    assert STIMULUS_PHYSICAL_COORDINATE_MANIFEST_REF_ATTR not in failed.attrs
    assert STIMULUS_PHYSICAL_COORDINATE_MANIFEST_SHA256_ATTR not in failed.attrs
    for name, value in pointer_snapshot.items():
        assert runs.attrs[name] == value


def test_import_keyboard_interrupt_during_complete_validation_restores_selectors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h5_path = tmp_path / "physical_interrupt.h5"
    zarr_path = tmp_path / "physical_interrupt.zarr"
    run_name = "stimulus_physical_interrupt"
    _write_stimulus_h5_with_arena_relative_chaser_states(h5_path)
    _prepare_acquisition_authority(zarr_path)
    root = zarr.open_group(str(zarr_path), mode="a", use_consolidated=False)
    runs = root["analysis"].require_group("stimulus_runs")
    pointer_snapshot = {
        "latest": "previous_latest",
        "latest_complete": "previous_complete",
        "latest_pending": "previous_pending",
        "authoritative_run": "previous_authority",
        "authoritative_run_provenance": {"sentinel": "unchanged"},
    }
    runs.attrs.update(pointer_snapshot)

    def _interrupt_complete_validation(root_group, *, stimulus_run, require_complete):
        candidate_parent = root_group["analysis/stimulus_runs"]
        candidate = candidate_parent[stimulus_run]
        assert require_complete is True
        assert candidate.attrs["palette_run_completion_status"] == "complete"
        assert candidate.attrs["stage_selector_eligible"] is False
        for name, value in pointer_snapshot.items():
            assert candidate_parent.attrs[name] == value
        raise KeyboardInterrupt("synthetic complete-validation interrupt")

    monkeypatch.setattr(
        mod,
        "_load_stimulus_physical_coordinate_authority_before_selection",
        _interrupt_complete_validation,
    )
    with pytest.raises(
        KeyboardInterrupt,
        match="synthetic complete-validation interrupt",
    ):
        mod.import_stimulus_to_zarr(
            stimulus_h5=h5_path,
            zarr_path=zarr_path,
            run_name=run_name,
            overwrite=False,
            verbose=False,
            repair_chaser_gaps=False,
        )

    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    runs = root["analysis/stimulus_runs"]
    failed = runs[run_name]
    assert failed.attrs["palette_run_completion_status"] == "failed"
    assert failed.attrs["stage_selector_eligible"] is False
    for name, value in pointer_snapshot.items():
        assert runs.attrs[name] == value


def test_import_rechecks_open_h5_file_identity_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h5_path = tmp_path / "source.h5"
    replacement_path = tmp_path / "replacement.h5"
    zarr_path = tmp_path / "analysis.zarr"
    run_name = "stimulus_toctou_replacement"
    _write_minimal_stimulus_h5(h5_path)
    _write_minimal_stimulus_h5(replacement_path)
    original = mod._materialize_selected_calibration_snapshot

    def _replace_source_after_copy(*args, **kwargs):
        original(*args, **kwargs)
        replacement_path.replace(h5_path)

    monkeypatch.setattr(
        mod,
        "_materialize_selected_calibration_snapshot",
        _replace_source_after_copy,
    )
    with pytest.raises(
        StimulusCoordinateContractError,
        match="no longer identifies the open file handle",
    ):
        mod.import_stimulus_to_zarr(
            stimulus_h5=h5_path,
            zarr_path=zarr_path,
            run_name=run_name,
            overwrite=False,
            verbose=False,
            repair_chaser_gaps=False,
        )

    root = zarr.open_group(str(zarr_path), mode="r")
    runs = root["analysis"]["stimulus_runs"]
    assert runs[run_name].attrs["palette_run_completion_status"] == "failed"
    assert runs.attrs.get("latest_complete") != run_name


@pytest.mark.parametrize(
    "tamper_kind",
    ["descriptor_digest", "manifest_digest", "row_identity_ref"],
)
def test_import_rejects_tampered_canonical_coordinate_metadata(
    tmp_path: Path,
    tamper_kind: str,
) -> None:
    h5_path = tmp_path / f"{tamper_kind}.h5"
    zarr_path = tmp_path / f"{tamper_kind}.zarr"
    _write_stimulus_h5_with_arena_relative_chaser_states(
        h5_path,
        include_legacy_attrs=False,
    )
    with h5py.File(h5_path, "a") as h5:
        attrs = h5["/tracking_data/chaser_states"].attrs
        if tamper_kind in {"descriptor_digest", "row_identity_ref"}:
            payload = json.loads(attrs["coordinate_descriptor"])
            if tamper_kind == "descriptor_digest":
                payload["origin"] = "top_left"
            else:
                payload["row_identity"]["record_ref"] = (
                    f"/tracking_data/wrong_rows@{ROW_IDENTITY_CONTRACT_ATTR}"
                )
            attrs["coordinate_descriptor"] = json.dumps(payload, sort_keys=True)
            if tamper_kind == "row_identity_ref":
                attrs["coordinate_descriptor_sha256"] = canonical_mapping_digest(
                    payload
                )
        else:
            manifest = json.loads(attrs[COORDINATE_SURFACE_MANIFEST_ATTR])
            manifest["surfaces"][1]["semantic_role"] = "silently_changed"
            attrs[COORDINATE_SURFACE_MANIFEST_ATTR] = json.dumps(
                manifest, sort_keys=True
            )
    zarr.open_group(str(zarr_path), mode="w")

    with pytest.raises(StimulusCoordinateContractError):
        mod.import_stimulus_to_zarr(
            stimulus_h5=h5_path,
            zarr_path=zarr_path,
            run_name=f"stimulus_{tamper_kind}",
            overwrite=False,
            verbose=False,
            repair_chaser_gaps=False,
        )


@pytest.mark.parametrize(
    ("manifest_error", "expected"),
    [
        ("duplicate_role", "semantic_role 'target_position' is duplicated"),
        ("missing_required_role", "Required coordinate semantic roles are missing"),
        ("reused_component", "requires component_fields"),
        ("bad_component_cardinality", "requires two unique component_fields"),
        ("unbound_known_role", "surface component_fields disagree"),
    ],
)
def test_import_enforces_controlled_surface_roles_and_cardinality(
    tmp_path: Path,
    manifest_error: str,
    expected: str,
) -> None:
    h5_path = tmp_path / f"{manifest_error}.h5"
    zarr_path = tmp_path / f"{manifest_error}.zarr"
    _write_stimulus_h5_with_arena_relative_chaser_states(h5_path)
    with h5py.File(h5_path, "a") as h5:
        attrs = h5["/tracking_data/chaser_states"].attrs
        manifest = json.loads(attrs[COORDINATE_SURFACE_MANIFEST_ATTR])
        if manifest_error == "duplicate_role":
            manifest["surfaces"].append(dict(manifest["surfaces"][1]))
        elif manifest_error == "missing_required_role":
            manifest["surfaces"] = [
                surface
                for surface in manifest["surfaces"]
                if surface["semantic_role"] != "target_position"
            ]
        elif manifest_error == "reused_component":
            manifest["surfaces"][2]["component_fields"][0] = (
                manifest["surfaces"][0]["component_fields"][0]
            )
        elif manifest_error == "bad_component_cardinality":
            manifest["surfaces"][1]["component_fields"] = ["target_pos_x"]
        else:
            manifest["surfaces"] = [
                surface
                for surface in manifest["surfaces"]
                if surface["semantic_role"] != "chaser_position"
            ]
        attrs[COORDINATE_SURFACE_MANIFEST_ATTR] = json.dumps(
            manifest,
            sort_keys=True,
        )
        attrs[COORDINATE_SURFACE_MANIFEST_DIGEST_ATTR] = (
            canonical_mapping_digest(manifest)
        )

    with pytest.raises(StimulusCoordinateContractError, match=expected):
        mod.import_stimulus_to_zarr(
            stimulus_h5=h5_path,
            zarr_path=zarr_path,
            run_name=f"stimulus_{manifest_error}",
            overwrite=False,
            verbose=False,
            repair_chaser_gaps=False,
        )

    assert not zarr_path.exists()


def test_import_rejects_arbitrary_unclassified_future_chaser_field(
    tmp_path: Path,
) -> None:
    h5_path = tmp_path / "unclassified_future_field.h5"
    zarr_path = tmp_path / "unclassified_future_field.zarr"
    _write_stimulus_h5_with_arena_relative_chaser_states(h5_path)
    with h5py.File(h5_path, "a") as h5:
        source = h5["/tracking_data/chaser_states"]
        attrs = dict(source.attrs)
        values = source[:]
        extended = np.empty(
            values.shape,
            dtype=np.dtype([*values.dtype.descr, ("future_scalar", "<f4")]),
        )
        for field_name in values.dtype.names or ():
            extended[field_name] = values[field_name]
        extended["future_scalar"] = np.array([4.0, 4.5], dtype=np.float32)
        del h5["/tracking_data/chaser_states"]
        destination = h5["/tracking_data"].create_dataset(
            "chaser_states",
            data=extended,
        )
        destination.attrs.update(attrs)

    with pytest.raises(
        StimulusCoordinateContractError,
        match="field_classifications must cover the structured dtype exactly",
    ):
        mod.import_stimulus_to_zarr(
            stimulus_h5=h5_path,
            zarr_path=zarr_path,
            run_name="stimulus_unclassified_future_field",
            overwrite=False,
            verbose=False,
            repair_chaser_gaps=False,
        )

    assert not zarr_path.exists()


def test_import_rejects_chaser_space_without_exact_authority_resolver(
    tmp_path: Path,
) -> None:
    h5_path = tmp_path / "unsupported_space.h5"
    zarr_path = tmp_path / "unsupported_space.zarr"
    _write_stimulus_h5_with_arena_relative_chaser_states(h5_path)
    with h5py.File(h5_path, "a") as h5:
        attrs = h5["/tracking_data/chaser_states"].attrs
        descriptor = json.loads(attrs["coordinate_descriptor"])
        descriptor["space_id"] = "stimulus_texture_px"
        attrs["coordinate_descriptor"] = json.dumps(descriptor, sort_keys=True)
        attrs["coordinate_descriptor_sha256"] = canonical_mapping_digest(
            descriptor
        )

    with pytest.raises(
        StimulusCoordinateContractError,
        match="profile_field_mismatch.*space_id",
    ):
        mod.import_stimulus_to_zarr(
            stimulus_h5=h5_path,
            zarr_path=zarr_path,
            run_name="stimulus_unsupported_space",
            overwrite=False,
            verbose=False,
            repair_chaser_gaps=False,
        )

    assert not zarr_path.exists()


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("units", "profile_component_unit_mismatch"),
        ("origin", "profile_field_mismatch.*origin"),
        ("x_direction", "profile_field_mismatch.*positive_directions.x"),
        ("pixel_convention", "continuous pixel convention"),
    ],
)
def test_import_rejects_contradictory_arena_coordinate_semantics(
    tmp_path: Path,
    mutation: str,
    expected: str,
) -> None:
    h5_path = tmp_path / f"bad_{mutation}.h5"
    zarr_path = tmp_path / f"bad_{mutation}.zarr"
    _write_stimulus_h5_with_arena_relative_chaser_states(h5_path)
    with h5py.File(h5_path, "a") as h5:
        attrs = h5["/tracking_data/chaser_states"].attrs
        descriptor = json.loads(attrs["coordinate_descriptor"])
        if mutation == "units":
            descriptor["component_units"] = ["mm", "mm"]
        elif mutation == "origin":
            descriptor["origin"] = "top_left"
        elif mutation == "x_direction":
            descriptor["positive_directions"]["x"] = "left"
        else:
            descriptor["pixel_convention"] = "pixel_center"
        attrs["coordinate_descriptor"] = json.dumps(descriptor, sort_keys=True)
        attrs["coordinate_descriptor_sha256"] = canonical_mapping_digest(
            descriptor
        )

    with pytest.raises(StimulusCoordinateContractError, match=expected):
        mod.import_stimulus_to_zarr(
            stimulus_h5=h5_path,
            zarr_path=zarr_path,
            run_name=f"stimulus_bad_{mutation}",
            overwrite=False,
            verbose=False,
            repair_chaser_gaps=False,
        )

    assert not zarr_path.exists()


@pytest.mark.parametrize("overlay_status", ["requires_transform", "unknown"])
def test_import_rejects_unresolvable_or_unknown_overlay_contract(
    tmp_path: Path,
    overlay_status: str,
) -> None:
    h5_path = tmp_path / f"overlay_{overlay_status}.h5"
    zarr_path = tmp_path / f"overlay_{overlay_status}.zarr"
    _write_stimulus_h5_with_arena_relative_chaser_states(
        h5_path,
        include_legacy_attrs=False,
    )
    with h5py.File(h5_path, "a") as h5:
        attrs = h5["/tracking_data/chaser_states"].attrs
        payload = json.loads(attrs["coordinate_descriptor"])
        payload["source_camera_overlay"] = {"status": overlay_status}
        attrs["coordinate_descriptor"] = json.dumps(payload, sort_keys=True)
        attrs["coordinate_descriptor_sha256"] = canonical_mapping_digest(payload)
    zarr.open_group(str(zarr_path), mode="w")

    expected = (
        "source_camera_overlay.chain_direction"
        if overlay_status == "requires_transform"
        else "overlay_status_unsupported"
    )
    with pytest.raises(StimulusCoordinateContractError, match=expected):
        mod.import_stimulus_to_zarr(
            stimulus_h5=h5_path,
            zarr_path=zarr_path,
            run_name=f"stimulus_overlay_{overlay_status}",
            overwrite=False,
            verbose=False,
            repair_chaser_gaps=False,
        )


def test_import_rejects_stale_source_arena_digest_before_zarr_mutation(
    tmp_path: Path,
) -> None:
    h5_path = tmp_path / "stale_arena_digest.h5"
    zarr_path = tmp_path / "stale_arena_digest.zarr"
    _write_stimulus_h5_with_arena_relative_chaser_states(h5_path)
    with h5py.File(h5_path, "a") as h5:
        h5["/calibration_snapshot/arena_geometry"].attrs[
            "arena_origin_in_canvas_x_px"
        ] = 271

    with pytest.raises(
        StimulusCoordinateContractError,
        match="Persisted arena_geometry record or digest disagrees",
    ):
        mod.import_stimulus_to_zarr(
            stimulus_h5=h5_path,
            zarr_path=zarr_path,
            run_name="stimulus_stale_arena_digest",
            overwrite=False,
            verbose=False,
            repair_chaser_gaps=False,
        )

    assert not zarr_path.exists()


def test_import_rejects_stale_typed_source_arena_frame_before_zarr_mutation(
    tmp_path: Path,
) -> None:
    h5_path = tmp_path / "stale_arena_frame.h5"
    zarr_path = tmp_path / "stale_arena_frame.zarr"
    _write_stimulus_h5_with_arena_relative_chaser_states(h5_path)
    with h5py.File(h5_path, "a") as h5:
        h5["/calibration_snapshot/arena_geometry"].attrs[
            PIXEL_FRAME_AUTHORITY_DIGEST_ATTR
        ] = "0" * 64

    with pytest.raises(
        StimulusCoordinateContractError,
        match="source arena pixel-frame record or digest is stale",
    ):
        mod.import_stimulus_to_zarr(
            stimulus_h5=h5_path,
            zarr_path=zarr_path,
            run_name="stimulus_stale_arena_frame",
            overwrite=False,
            verbose=False,
            repair_chaser_gaps=False,
        )

    assert not zarr_path.exists()


def test_import_rejects_same_size_selected_arena_with_wrong_origin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h5_path = tmp_path / "wrong_selected_origin.h5"
    zarr_path = tmp_path / "wrong_selected_origin.zarr"
    run_name = "stimulus_wrong_selected_origin"
    _write_stimulus_h5_with_arena_relative_chaser_states(h5_path)
    _prepare_acquisition_authority(zarr_path)
    copy_group = mod._copy_h5_group_to_zarr_mirror

    def _copy_with_wrong_origin(source, destination):
        copy_group(source, destination)
        if source.name == "/calibration_snapshot/arena_geometry":
            destination.attrs["arena_origin_in_canvas_y_px"] = 521

    monkeypatch.setattr(mod, "_copy_h5_group_to_zarr_mirror", _copy_with_wrong_origin)
    with pytest.raises(
        StimulusCoordinateContractError,
        match="Selected arena_geometry snapshot differs from the verified source record",
    ):
        mod.import_stimulus_to_zarr(
            stimulus_h5=h5_path,
            zarr_path=zarr_path,
            run_name=run_name,
            overwrite=False,
            verbose=False,
            repair_chaser_gaps=False,
        )

    root = zarr.open_group(str(zarr_path), mode="r")
    runs = root["analysis"]["stimulus_runs"]
    failed = runs[run_name]
    selected = failed["calibration"]["arena_geometry"]
    assert selected.attrs["arena_region_width_px"] == 344
    assert selected.attrs["arena_region_height_px"] == 344
    assert selected.attrs["arena_origin_in_canvas_y_px"] == 521
    assert failed.attrs["palette_run_completion_status"] == "failed"
    assert runs.attrs.get("latest") != run_name
    assert runs.attrs.get("latest_complete") != run_name


def test_import_rejects_reference_extent_mismatch_with_selected_arena(
    tmp_path: Path,
) -> None:
    h5_path = tmp_path / "extent_mismatch.h5"
    zarr_path = tmp_path / "extent_mismatch.zarr"
    _write_stimulus_h5_with_arena_relative_chaser_states(
        h5_path,
        include_legacy_attrs=False,
    )
    with h5py.File(h5_path, "a") as h5:
        attrs = h5["/tracking_data/chaser_states"].attrs
        payload = json.loads(attrs["coordinate_descriptor"])
        payload["reference_extent"]["width"] = 345
        attrs["coordinate_descriptor"] = json.dumps(payload, sort_keys=True)
        attrs["coordinate_descriptor_sha256"] = canonical_mapping_digest(payload)
    zarr.open_group(str(zarr_path), mode="w")

    with pytest.raises(
        StimulusCoordinateContractError,
        match="Source descriptor extent disagrees with source arena_geometry",
    ):
        mod.import_stimulus_to_zarr(
            stimulus_h5=h5_path,
            zarr_path=zarr_path,
            run_name="stimulus_extent_mismatch",
            overwrite=False,
            verbose=False,
            repair_chaser_gaps=False,
        )


def test_import_rejects_canonical_legacy_coordinate_conflict(tmp_path: Path) -> None:
    h5_path = tmp_path / "conflict.h5"
    zarr_path = tmp_path / "conflict.zarr"
    _write_stimulus_h5_with_arena_relative_chaser_states(h5_path)
    with h5py.File(h5_path, "a") as h5:
        h5["/tracking_data/chaser_states"].attrs["coordinate_origin"] = "top_left"
    zarr.open_group(str(zarr_path), mode="w")

    with pytest.raises(
        StimulusCoordinateContractError,
        match="must not carry legacy coordinate attrs",
    ):
        mod.import_stimulus_to_zarr(
            stimulus_h5=h5_path,
            zarr_path=zarr_path,
            run_name="stimulus_conflict",
            overwrite=False,
            verbose=False,
            repair_chaser_gaps=False,
        )


def test_import_rejects_unsigned_bounding_boxes_without_publishing(tmp_path: Path) -> None:
    h5_path = tmp_path / "bounding_boxes.h5"
    zarr_path = tmp_path / "bounding_boxes.zarr"
    _write_minimal_stimulus_h5(h5_path)
    bbox_dtype = np.dtype(
        [
            ("stimulus_frame_num", np.uint64),
            ("x_min", np.float32),
            ("y_min", np.float32),
            ("width", np.float32),
            ("height", np.float32),
        ]
    )
    with h5py.File(h5_path, "a") as h5:
        tracking = h5.create_group("tracking_data")
        tracking.create_dataset(
            "bounding_boxes",
            data=np.array([(1000, 10.0, 20.0, 30.0, 40.0)], dtype=bbox_dtype),
        )
    with pytest.raises(
        StimulusCoordinateContractError,
        match="bounding_boxes lacks canonical array-specific geometry support",
    ):
        mod.import_stimulus_to_zarr(
            stimulus_h5=h5_path,
            zarr_path=zarr_path,
            run_name="stimulus_unsigned_bboxes",
            overwrite=False,
            verbose=False,
            repair_chaser_gaps=False,
        )

    assert not zarr_path.exists()


def test_import_materializes_unique_composite_multi_chaser_row_identity(
    tmp_path: Path,
) -> None:
    h5_path = tmp_path / "multi.h5"
    zarr_path = tmp_path / "multi.zarr"
    _write_stimulus_h5_with_arena_relative_chaser_states(
        h5_path,
        include_legacy_attrs=False,
        multi_chaser=True,
    )
    _prepare_acquisition_authority(zarr_path)

    run_name = mod.import_stimulus_to_zarr(
        stimulus_h5=h5_path,
        zarr_path=zarr_path,
        run_name="stimulus_multi_chaser",
        overwrite=False,
        verbose=False,
        repair_chaser_gaps=False,
    )

    root = zarr.open_group(str(zarr_path), mode="r")
    chaser_group = root["analysis"]["stimulus_runs"][run_name]["tracking_data"][
        "chaser_states"
    ]
    assert "coordinate_row_identity" not in chaser_group
    row_identity = chaser_group[STIMULUS_STATE_KEY_ARRAY_REF][:]
    np.testing.assert_array_equal(
        row_identity,
        np.array([[0, 1000], [1, 1000], [0, 1001], [1, 1001]], dtype=np.int64),
    )
    assert np.unique(row_identity, axis=0).shape[0] == row_identity.shape[0]
    identity = validate_stamped_row_identity(
        chaser_group,
        chaser_group[STIMULUS_STATE_KEY_ARRAY_REF],
    )
    descriptor = _load_output_coordinate_descriptor(
        chaser_group,
        chaser_group["target_position_xy"],
    )
    assert descriptor.row_identity.record_ref == (
        f"/{chaser_group.path}@{ROW_IDENTITY_CONTRACT_ATTR}"
    )
    assert descriptor.row_identity.record_sha256 == identity.digest()
    assert identity.key_array.components == ("chaser_index", "stimulus_frame_num")
    assert chaser_group["target_position_xy"].shape == (4, 2)


def test_import_rejects_nonunique_composite_chaser_row_identity(tmp_path: Path) -> None:
    h5_path = tmp_path / "duplicate_rows.h5"
    zarr_path = tmp_path / "duplicate_rows.zarr"
    _write_stimulus_h5_with_arena_relative_chaser_states(
        h5_path,
        include_legacy_attrs=False,
        multi_chaser=True,
    )
    with h5py.File(h5_path, "a") as h5:
        dataset = h5["/tracking_data/chaser_states"]
        attrs = dict(dataset.attrs)
        states = dataset[:]
        states[1]["chaser_index"] = states[0]["chaser_index"]
        states[1]["stimulus_frame_num"] = states[0]["stimulus_frame_num"]
        del h5["/tracking_data/chaser_states"]
        dataset = h5["/tracking_data"].create_dataset("chaser_states", data=states)
        dataset.attrs.update(attrs)
    zarr.open_group(str(zarr_path), mode="w")

    with pytest.raises(
        StimulusCoordinateContractError,
        match="Source row identity is not unique",
    ):
        mod.import_stimulus_to_zarr(
            stimulus_h5=h5_path,
            zarr_path=zarr_path,
            run_name="stimulus_duplicate_rows",
            overwrite=False,
            verbose=False,
            repair_chaser_gaps=False,
        )


def test_import_preserves_source_row_identity_instead_of_interpolating_coordinates(
    tmp_path: Path,
) -> None:
    h5_path = tmp_path / "interpolated.h5"
    zarr_path = tmp_path / "interpolated.zarr"
    _write_stimulus_h5_with_arena_relative_chaser_states(
        h5_path,
        include_legacy_attrs=False,
    )
    with h5py.File(h5_path, "a") as h5:
        frame_ds = h5["/video_metadata/frame_metadata"]
        frame_attrs = dict(frame_ds.attrs)
        frames = frame_ds[:]
        frames[1]["stimulus_frame_num"] = 1002
        frames[1]["triggering_camera_frame_id"] = 2002
        del h5["/video_metadata/frame_metadata"]
        frame_ds = h5["/video_metadata"].create_dataset("frame_metadata", data=frames)
        frame_ds.attrs.update(frame_attrs)

        chaser_ds = h5["/tracking_data/chaser_states"]
        chaser_attrs = dict(chaser_ds.attrs)
        states = chaser_ds[:]
        states[1]["stimulus_frame_num"] = 1002
        del h5["/tracking_data/chaser_states"]
        chaser_ds = h5["/tracking_data"].create_dataset("chaser_states", data=states)
        chaser_ds.attrs.update(chaser_attrs)
        row_identity = h5["/tracking_data/stimulus_state_key"]
        row_identity[:] = np.array([1000, 1002], dtype=np.int64)
        identity_contract = build_row_identity_contract(
            domain=STIMULUS_STATE_DOMAIN,
            values=row_identity[:],
            components=("stimulus_frame_num",),
        )
        _write_h5_contract_attrs(
            chaser_ds,
            row_identity_contract_attrs(identity_contract),
        )
        _write_h5_contract_attrs(
            row_identity,
            row_identity_key_attrs(identity_contract),
        )
        _refresh_source_acquisition_identity_binding(
            h5,
            row_values=row_identity[:],
            identity_contract=identity_contract,
        )
        descriptor = json.loads(chaser_ds.attrs["coordinate_descriptor"])
        descriptor["row_identity"]["record_sha256"] = identity_contract.digest()
        chaser_ds.attrs["coordinate_descriptor"] = json.dumps(
            descriptor,
            sort_keys=True,
        )
        chaser_ds.attrs["coordinate_descriptor_sha256"] = (
            canonical_mapping_digest(descriptor)
        )
    _prepare_acquisition_authority(zarr_path)

    run_name = mod.import_stimulus_to_zarr(
        stimulus_h5=h5_path,
        zarr_path=zarr_path,
        run_name="stimulus_interpolated",
        overwrite=False,
        verbose=False,
        repair_chaser_gaps=True,
    )

    root = zarr.open_group(str(zarr_path), mode="r")
    run_group = root["analysis"]["stimulus_runs"][run_name]
    tracking_group = run_group["tracking_data"]
    chaser_group = tracking_group["chaser_states"]
    np.testing.assert_array_equal(
        chaser_group[STIMULUS_STATE_KEY_ARRAY_REF][:],
        np.array([1000, 1002], dtype=np.int64),
    )
    assert "chaser_interpolation_mask" not in tracking_group
    assert run_group.attrs["chaser_interpolation_skipped"] is True
    assert run_group.attrs["chaser_interpolation_skipped_reason"] == (
        "canonical_coordinate_rows_must_copy_source_identity"
    )
    lineage = chaser_group.attrs[COORDINATE_IMPORT_LINEAGE_ATTR]
    assert lineage["interpolation"]["applied"] is False
    assert lineage["interpolation"]["mask_ref"] is None
    assert lineage["interpolation"]["mask_sha256"] is None
    assert chaser_group["target_position_xy"].shape == (2, 2)


def test_import_does_not_infer_step_geometry_from_events_or_calibration(
    tmp_path: Path,
) -> None:
    h5_path = tmp_path / "event_only.h5"
    zarr_path = tmp_path / "event_only.zarr"
    _write_minimal_stimulus_h5(h5_path)
    events_dtype = np.dtype(
        [
            ("event_name", "S32"),
            ("current_step_index", np.int32),
            ("stimulus_mode_id", np.int32),
            ("camera_frame_id", np.int64),
        ]
    )
    with h5py.File(h5_path, "a") as h5:
        h5.create_dataset(
            "events",
            data=np.array(
                [
                    (b"STEP_START", 0, 6, 2000),
                    (b"STEP_END", 0, 6, 2001),
                ],
                dtype=events_dtype,
            ),
        )

    run_name = mod.import_stimulus_to_zarr(
        stimulus_h5=h5_path,
        zarr_path=zarr_path,
        run_name="stimulus_event_only",
        overwrite=False,
        verbose=False,
        repair_chaser_gaps=False,
    )
    root = zarr.open_group(str(zarr_path), mode="r")
    run = root["analysis"]["stimulus_runs"][run_name]
    assert "events" in run
    assert "steps" not in run
    assert "stimulus_coordinates" not in run


def test_import_materializes_canonical_protocol_step_metadata(tmp_path: Path) -> None:
    h5_path = tmp_path / "session.h5"
    zarr_path = tmp_path / "sample_analysis.zarr"

    _write_stimulus_h5_with_protocol_steps(h5_path)
    with pytest.raises(
        StimulusCoordinateContractError,
        match="stimulus_coordinates lacks canonical array-specific geometry support",
    ):
        mod.import_stimulus_to_zarr(
            stimulus_h5=h5_path,
            zarr_path=zarr_path,
            run_name="stimulus_with_steps",
            overwrite=False,
            verbose=False,
            repair_chaser_gaps=False,
        )

    assert not zarr_path.exists()


def test_metadata_and_calibration_only_import_omits_uncontracted_coordinate_surfaces(
    tmp_path: Path,
) -> None:
    h5_path = tmp_path / "session.h5"
    zarr_path = tmp_path / "sample_analysis.zarr"
    _write_stimulus_h5_with_protocol_steps(h5_path)
    with h5py.File(h5_path, "a") as h5:
        tracking = h5.create_group("tracking_data")
        tracking.create_dataset(
            "chaser_states",
            data=np.asarray([(1000, 1.0, 2.0)], dtype=[
                ("stimulus_frame_num", "<u8"),
                ("chaser_pos_x", "<f4"),
                ("chaser_pos_y", "<f4"),
            ]),
        )
        tracking.create_dataset(
            "bounding_boxes",
            data=np.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32),
        )

    run_name = mod.import_stimulus_to_zarr(
        stimulus_h5=h5_path,
        zarr_path=zarr_path,
        run_name="stimulus_metadata_only",
        overwrite=False,
        verbose=False,
        repair_chaser_gaps=False,
        metadata_and_calibration_only=True,
    )

    root = zarr.open_group(str(zarr_path), mode="r")
    run = root["analysis/stimulus_runs"][run_name]
    assert run.attrs["palette_run_completion_status"] == "complete"
    assert run.attrs["source_coordinate_policy"] == (
        SOURCE_COORDINATE_POLICY_METADATA_ONLY
    )
    assert run.attrs["source_coordinate_surface_status"] == (
        "omitted_uncontracted_source_surfaces"
    )
    assert run.attrs["omitted_coordinate_source_paths"] == [
        "/stimulus_coordinates",
        "/tracking_data/bounding_boxes",
        "/tracking_data/chaser_states",
    ]
    assert run.attrs["chaser_states_coordinate_descriptor_status"] == "not_present"
    assert "tracking_data" not in run
    assert "stimulus_coordinates" not in run
    assert "protocol_json" in run.attrs
    assert "events" in run
    assert "steps" in run
    assert "calibration" in run


def test_metadata_only_import_initializes_empty_source_camera_frame_placeholder(
    tmp_path: Path,
) -> None:
    h5_path = tmp_path / "session.h5"
    zarr_path = tmp_path / "sample_analysis.zarr"
    _write_stimulus_h5_with_protocol_steps(h5_path)
    _prepare_acquisition_authority(zarr_path, total_frames=138_000)
    root = zarr.open_group(str(zarr_path), mode="a", use_consolidated=False)
    placeholder = root.require_group(
        "analysis/coordinate_frames/source_camera/2010093/continuous"
    )
    assert dict(placeholder.attrs) == {}

    run_name = mod.import_stimulus_to_zarr(
        stimulus_h5=h5_path,
        zarr_path=zarr_path,
        run_name="stimulus_metadata_empty_source_frame",
        overwrite=False,
        verbose=False,
        repair_chaser_gaps=False,
        metadata_and_calibration_only=True,
    )

    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    source_frame = root[
        "analysis/coordinate_frames/source_camera/2010093/continuous"
    ]
    assert PIXEL_FRAME_AUTHORITY_ATTR in source_frame.attrs
    assert PIXEL_FRAME_AUTHORITY_DIGEST_ATTR in source_frame.attrs
    physical = load_stimulus_physical_coordinate_authority(
        root,
        stimulus_run=run_name,
    )
    assert physical is not None
    assert physical.camera_id == "2010093"
