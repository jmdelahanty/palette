from __future__ import annotations

import copy

import numpy as np
import zarr

from fisheye.shared.pose_model_schema_binding import (
    build_explicit_pose_model_schema_binding,
)
from fisheye.shared.zarr.array_factory import array_metadata_declaration_from_plan
from fisheye.shared.zarr.benchmark_runtime import sha256_array
from fisheye.shared.zarr.crop_manifest import (
    CropPixelAuthority,
    CropRefinedSourceIdentity,
    build_coordinate_crop_run_manifest,
    build_crop_row_source_signatures,
)
from fisheye.shared.zarr.crop_schema import (
    CROP_GEOMETRY_SCHEMA_V1,
    CropDimensions,
    CropGeometryPolicy,
    CropPaddingMode,
    CropSizeMode,
    derive_crop_placement_geometry,
    derive_frame_row_offsets,
)
from fisheye.shared.zarr.crop_storage import plan_crop_geometry_storage
from fisheye.shared.zarr.detection_schema import derive_canonical_detection_geometry
from fisheye.shared.zarr.keypoint_manifest import (
    KeypointPreprocessingReference,
    keypoint_skeleton_digest,
)
from fisheye.shared.zarr.keypoint_publication import (
    prepare_raw_keypoint_v2_snapshot,
    publish_selector_ineligible_keypoint_snapshot,
)
from fisheye.shared.zarr.keypoint_quality_producer import (
    ObservationLocalKeypointQualityPolicy,
    prepare_observation_local_keypoint_quality,
)
from fisheye.shared.zarr.keypoint_quality_publication import (
    publish_selector_ineligible_keypoint_quality_snapshot,
)
from fisheye.shared.zarr.keypoint_quality_schema import KeypointQualitySourceReference
from fisheye.shared.zarr.keypoint_schema import (
    REFINED_KEYPOINT_SCHEMA_V2,
    KeypointDimensions,
    derive_keypoint_row_signatures,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.refined_keypoint_manifest import (
    build_refined_keypoint_source_bindings,
    initial_refined_keypoint_snapshot_identity,
    validate_refined_keypoint_publication,
    validate_refined_keypoint_run_manifest,
)
from fisheye.shared.zarr.refined_keypoint_producer import (
    LandmarkCoordinateEdit,
    RefinedKeypointDecision,
    prepare_refined_keypoint_snapshot,
)
from fisheye.shared.zarr.refined_keypoint_publication import (
    publish_selector_ineligible_refined_keypoint_snapshot,
    validate_refined_keypoint_shadow_publication,
)


REVIEW_STATE_MAP = {0: "unreviewed", 1: "accepted", 2: "rejected"}
REASON_CODE_MAP = {0: "none", 1: "manual_reject", 2: "manual_correction"}


def _crop_source() -> CropRefinedSourceIdentity:
    return CropRefinedSourceIdentity(
        run_id="refined_source",
        run_manifest_digest="a" * 64,
        logical_content_digest="b" * 64,
        recording_identity="refined_keypoint_test_recording",
        lineage_id="11111111-1111-4111-8111-111111111111",
        snapshot_id="22222222-2222-4222-8222-222222222222",
    )


def _pixel() -> CropPixelAuthority:
    return CropPixelAuthority(
        authority_id="camera_video_manifest_v1",
        authority_manifest_digest="c" * 64,
        recording_identity="refined_keypoint_test_recording",
        camera_identity="cam_test",
        n_frames=4,
        source_width=100,
        source_height=80,
    )


def _crop_fixture():  # type: ignore[no-untyped-def]
    dimensions = CropDimensions(
        n_frames=4,
        n_instances=4,
        source_width=100,
        source_height=80,
    )
    policy = CropGeometryPolicy(
        purpose="subject_analysis",
        size_mode=CropSizeMode.FIXED_PER_RUN,
        fixed_size_wh=(20, 20),
        padding_mode=CropPaddingMode.ZERO_OUTSIDE_SOURCE_FRAME,
    )
    frames = np.asarray([0, 0, 2, 3], dtype=np.int64)
    bbox_norm = np.asarray(
        [
            [0.20, 0.20, 0.10, 0.10],
            [0.70, 0.20, 0.10, 0.10],
            [0.50, 0.70, 0.20, 0.10],
            [0.25, 0.75, 0.10, 0.15],
        ],
        dtype=np.float32,
    )
    bbox_img, centers = derive_canonical_detection_geometry(
        bbox_norm,
        source_width=100,
        source_height=80,
    )
    sizes = np.repeat(np.asarray([[20, 20]], dtype=np.int32), 4, axis=0)
    coordinates, source_crop, bbox_roi = derive_crop_placement_geometry(
        centers,
        bbox_img,
        sizes,
    )
    arrays = {
        "instance_key": np.asarray([101, 102, 201, 301], dtype=np.uint64),
        "source_refined_row_ids": np.arange(4, dtype=np.int64),
        "frame_indices": frames,
        "source_acquisition_frame_index": frames.copy(),
        "frame_row_offsets": derive_frame_row_offsets(frames, n_frames=4),
        "bbox_norm_coords": bbox_norm,
        "bbox_img_xyxy": bbox_img,
        "centers_img_xy": centers,
        "roi_coordinates_full": coordinates,
        "roi_sizes_full": sizes,
        "source_crop_xywh": source_crop,
        "bbox_roi_xyxy": bbox_roi,
    }
    arrays["source_row_signature"] = build_crop_row_source_signatures(
        arrays,
        source=_crop_source(),
        policy=policy,
        pixel_authority=_pixel(),
    ).signatures
    plans = plan_crop_geometry_storage(dimensions)
    declarations: dict[str, dict[str, object]] = {
        "": {
            "zarr_format": 3,
            "node_type": "group",
            "attributes": {
                "status": "complete",
                "stage_selector_eligible": False,
                "shadow_only": True,
            },
        }
    }
    bindings = {binding.path: binding for binding in CROP_GEOMETRY_SCHEMA_V1.bindings}
    for entry in plans.entries:
        binding = bindings[entry.rule.path]
        contract = CROP_GEOMETRY_SCHEMA_V1.contracts.resolve(
            binding.contract_id,
            binding.contract_version,
        )
        declarations[entry.rule.path] = {
            "zarr_format": 3,
            "node_type": "array",
            **array_metadata_declaration_from_plan(
                contract=contract,
                plan=entry.plan,
                fill_value=0,
                attributes={"artifact_class": "geometry_only_analysis"},
            ),
        }
    manifest = build_coordinate_crop_run_manifest(
        run_id="crop_v2_source",
        dimensions=dimensions,
        policy=policy,
        storage_plan=plans,
        arrays=arrays,
        source=_crop_source(),
        pixel_authority=_pixel(),
        direct_metadata_declarations=declarations,
        consolidated_metadata_declarations=copy.deepcopy(declarations),
        selector_eligible=False,
    )
    return dimensions, arrays, manifest


def _pose_binding() -> dict[str, object]:
    return build_explicit_pose_model_schema_binding(
        model_sha256="d" * 64,
        assertion_id="selector_ineligible_refined_keypoint_test",
        skeleton_id="sleepyfish_three_point_v1",
        model_kpt_shape=[3, 3],
        keypoint_labels=["swim_bladder", "eye_left", "eye_right"],
        edges=[[0, 1], [0, 2], [1, 2]],
    )


def _raw_fixture():  # type: ignore[no-untyped-def]
    crop_dimensions, crop, crop_manifest = _crop_fixture()
    binding = _pose_binding()
    dimensions = KeypointDimensions(
        n_frames=crop_dimensions.n_frames,
        n_instances=crop_dimensions.n_instances,
        n_keypoints=3,
        source_width=crop_dimensions.source_width,
        source_height=crop_dimensions.source_height,
    )
    points_roi = np.asarray(
        [
            [[5, 10], [15, 5], [15, 15]],
            [[np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]],
            [[5, 10], [15, 5], [15, 15]],
            [[5, 10], [15, 5], [15, 15]],
        ],
        dtype=np.float32,
    )
    valid = np.all(np.isfinite(points_roi), axis=2)
    origins = crop["roi_coordinates_full"]
    success = np.any(valid, axis=1)
    bbox_roi = np.asarray(
        [[1, 1, 19, 19], [np.nan] * 4, [1, 1, 19, 19], [1, 1, 19, 19]],
        dtype=np.float32,
    )
    arrays = {
        "instance_key": crop["instance_key"].copy(),
        "source_crop_row_ids": np.arange(4, dtype=np.int64),
        "source_acquisition_frame_index": crop[
            "source_acquisition_frame_index"
        ].copy(),
        "frame_indices": crop["frame_indices"].copy(),
        "frame_row_offsets": crop["frame_row_offsets"].copy(),
        "source_crop_row_signature": crop["source_row_signature"].copy(),
        "keypoints_roi": points_roi,
        "keypoints_img": points_roi + origins.astype(np.float32)[:, None, :],
        "keypoint_confidences": np.where(
            valid, np.float32(0.9), np.float32(np.nan)
        ),
        "keypoint_valid": valid,
        "pose_confidence": np.where(
            success, np.float32(0.95), np.float32(np.nan)
        ),
        "pose_bbox_xyxy_roi": bbox_roi,
        "pose_bbox_xyxy_img": bbox_roi
        + np.column_stack((origins, origins)).astype(np.float32),
        "pose_success": success,
    }
    skeleton_digest = keypoint_skeleton_digest(binding)
    arrays["keypoint_row_signature"] = derive_keypoint_row_signatures(
        instance_key=arrays["instance_key"],
        source_crop_row_signature=arrays["source_crop_row_signature"],
        keypoints_roi=points_roi,
        keypoint_valid=valid,
        skeleton_digest=skeleton_digest,
    )
    preprocessing = KeypointPreprocessingReference(
        profile_id="yolo_pose_crop_v1",
        profile_version=1,
        input_mode="crop_pixel_package",
        document={
            "decoded_dtype": "uint8",
            "channels": "grayscale",
            "resize": "letterbox_model_shape",
            "normalization": "ultralytics_runtime_v1",
        },
    )
    prepared = prepare_raw_keypoint_v2_snapshot(
        arrays,
        dimensions=dimensions,
        source_crop_arrays=crop,
        source_crop_manifest=crop_manifest,
        pose_model_schema_binding=binding,
        preprocessing=preprocessing,
    )
    return prepared, crop, crop_manifest, skeleton_digest


def _published_sources(tmp_path: object):  # type: ignore[no-untyped-def]
    prepared, crop, crop_manifest, skeleton_digest = _raw_fixture()
    raw_root = tmp_path / "raw_root"  # type: ignore[operator]
    raw = publish_selector_ineligible_keypoint_snapshot(
        prepared,
        destination=raw_root / "raw.zarr",
        run_id="raw_v2_001",
        shadow_root=raw_root,
        created_by="pytest",
    )
    source = KeypointQualitySourceReference(
        run_name=raw.run_id,
        manifest_digest=canonical_json_sha256(raw.manifest),
        skeleton_id="sleepyfish_three_point_v1",
        skeleton_digest=skeleton_digest,
        keypoint_row_signatures_digest=sha256_array(
            prepared.arrays["keypoint_row_signature"]
        ),
    )
    quality_prepared = prepare_observation_local_keypoint_quality(
        prepared.arrays,
        source_dimensions=prepared.dimensions,
        source_crop_arrays=crop,
        source=source,
        skeleton_digest=skeleton_digest,
        policy=ObservationLocalKeypointQualityPolicy(
            confidence_threshold=0.8,
            minimum_valid_keypoints=2,
        ),
    )
    quality_root = tmp_path / "quality_root"  # type: ignore[operator]
    quality = publish_selector_ineligible_keypoint_quality_snapshot(
        quality_prepared,
        source_manifest=raw.manifest,
        destination=quality_root / "quality.zarr",
        run_id="quality_v1_001",
        shadow_root=quality_root,
        created_by="pytest",
    )
    return raw, quality, quality_prepared, crop, crop_manifest, skeleton_digest


def _publish_refined(tmp_path: object):  # type: ignore[no-untyped-def]
    raw, quality, quality_prepared, crop, crop_manifest, skeleton_digest = (
        _published_sources(tmp_path)
    )
    prepared = prepare_refined_keypoint_snapshot(
        raw.prepared.arrays,
        dimensions=raw.prepared.dimensions,
        source_crop_arrays=crop,
        skeleton_digest=skeleton_digest,
        keypoint_quality_arrays=quality.prepared.arrays,
        quality_dimensions=quality.prepared.dimensions,
        quality_profile=quality.prepared.profile,
        decisions=(
            RefinedKeypointDecision(
                instance_key=201,
                accepted=True,
                review_state_code=1,
                reason_code=2,
                coordinate_edits=(LandmarkCoordinateEdit(0, (6.0, 11.0)),),
                confidence_valid=True,
                geometry_valid=True,
            ),
            RefinedKeypointDecision(
                instance_key=301,
                accepted=False,
                review_state_code=2,
                reason_code=1,
            ),
        ),
        review_state_map=REVIEW_STATE_MAP,
        reason_code_map=REASON_CODE_MAP,
    )
    source = build_refined_keypoint_source_bindings(
        raw_manifest=raw.manifest,
        quality_manifest=quality.manifest,
        crop_manifest=crop_manifest,
    )
    identity = initial_refined_keypoint_snapshot_identity(
        recording_identity=source.recording_identity,
        lineage_id="33333333-3333-4333-8333-333333333333",
        snapshot_id="44444444-4444-4444-8444-444444444444",
    )
    refined_root = tmp_path / "refined_root"  # type: ignore[operator]
    publication = publish_selector_ineligible_refined_keypoint_snapshot(
        prepared,
        source=source,
        raw_manifest=raw.manifest,
        quality_manifest=quality.manifest,
        crop_manifest=crop_manifest,
        raw_arrays=raw.prepared.arrays,
        quality_arrays=quality_prepared.arrays,
        source_crop_arrays=crop,
        identity=identity,
        review_state_map=REVIEW_STATE_MAP,
        reason_code_map=REASON_CODE_MAP,
        destination=refined_root / "refined.zarr",
        run_id="refined_v2_001",
        shadow_root=refined_root,
        created_by="pytest",
    )
    return publication, raw, quality, crop_manifest


def test_refined_publication_round_trip_is_exact_and_selector_ineligible(
    tmp_path: object,
) -> None:
    publication, _, _, _ = _publish_refined(tmp_path)

    assert validate_refined_keypoint_run_manifest(publication.manifest) == ()
    assert validate_refined_keypoint_shadow_publication(publication) == ()
    family = zarr.open_group(
        str(publication.output_path / "refined_keypoints_runs"),
        mode="r",
        use_consolidated=False,
    )
    assert all(
        family.attrs.get(name) is None
        for name in ("latest", "latest_complete", "latest_pending", "authoritative_run")
    )
    run = family[publication.run_id]
    assert run.attrs["status"] == "complete"
    assert run.attrs["stage_selector_eligible"] is False
    assert set(run.array_keys()) == set(REFINED_KEYPOINT_SCHEMA_V2.binding_paths)
    assert "heading" not in " ".join(run.array_keys())
    assert np.asarray(run["keypoint_edit_flags"][:])[2, 0]
    assert not np.asarray(run["refined_success"][:])[3]


def test_refined_manifest_rejects_recomputed_nested_tampering(tmp_path: object) -> None:
    publication, _, _, _ = _publish_refined(tmp_path)
    tampered = copy.deepcopy(publication.manifest)
    tampered["payload"]["storage_plan"]["arrays"][0]["plan"]["chunk_shape"][0] = 1
    tampered["payload_digest"] = canonical_json_sha256(tampered["payload"])

    errors = validate_refined_keypoint_run_manifest(tampered)

    assert any("storage plan differs" in error for error in errors)


def test_refined_publication_rejects_source_fact_and_retired_key_tampering(
    tmp_path: object,
) -> None:
    publication, raw, quality, crop_manifest = _publish_refined(tmp_path)
    arrays = dict(publication.prepared.arrays)
    arrays["frame_indices"] = arrays["frame_indices"].copy()
    arrays["frame_indices"][0] = 1

    errors = validate_refined_keypoint_publication(
        publication.manifest,
        arrays=arrays,
        source_crop_arrays=publication.source_crop_arrays,
        raw_manifest=raw.manifest,
        quality_manifest=quality.manifest,
        crop_manifest=crop_manifest,
        raw_arrays=raw.prepared.arrays,
        quality_arrays=quality.prepared.arrays,
        retired_instance_keys=(101,),
        direct_metadata_declarations={},
        consolidated_metadata_declarations={},
    )

    assert any("source fact differs" in error for error in errors)
    assert any("live and retired" in error for error in errors)
