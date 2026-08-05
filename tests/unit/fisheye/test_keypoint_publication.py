from __future__ import annotations

import copy

import numpy as np
import pytest
import zarr

from fisheye.shared.pose_model_schema_binding import (
    build_explicit_pose_model_schema_binding,
)
from fisheye.shared.zarr.array_factory import array_metadata_declaration_from_plan
from fisheye.shared.zarr.benchmark_runtime import sha256_array
from fisheye.shared.zarr.body_frame_producer import (
    BodyFrameSourceReference,
    build_keypoint_body_frame_recipe,
    prepare_keypoint_body_frame,
)
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
    keypoint_crop_source_from_manifest,
    keypoint_skeleton_digest,
    validate_keypoint_run_manifest,
)
from fisheye.shared.zarr.keypoint_publication import (
    prepare_raw_keypoint_v2_from_yolo_arrays,
    prepare_raw_keypoint_v2_snapshot,
    publish_selector_ineligible_keypoint_snapshot,
    validate_keypoint_shadow_publication,
)
from fisheye.shared.zarr.keypoint_schema import (
    KEYPOINT_SCHEMA_V2,
    KeypointDimensions,
    derive_keypoint_row_signatures,
)
from fisheye.shared.zarr.keypoint_storage import plan_keypoint_storage
from fisheye.shared.zarr.manifest_digest import (
    canonical_json_sha256,
)


def _crop_source() -> CropRefinedSourceIdentity:
    return CropRefinedSourceIdentity(
        run_id="refined_source",
        run_manifest_digest="a" * 64,
        logical_content_digest="b" * 64,
        recording_identity="keypoint_v2_canary",
        lineage_id="11111111-1111-4111-8111-111111111111",
        snapshot_id="22222222-2222-4222-8222-222222222222",
    )


def _pixel() -> CropPixelAuthority:
    return CropPixelAuthority(
        authority_id="camera_video_manifest_v1",
        authority_manifest_digest="c" * 64,
        recording_identity="keypoint_v2_canary",
        camera_identity="cam2010095",
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
        bbox_norm, source_width=100, source_height=80
    )
    sizes = np.repeat(np.asarray([[20, 20]], dtype=np.int32), 4, axis=0)
    coordinates, source_crop, bbox_roi = derive_crop_placement_geometry(
        centers, bbox_img, sizes
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
            binding.contract_id, binding.contract_version
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
        assertion_id="selector_ineligible_yolo_canary",
        skeleton_id="sleepyfish_three_point_v1",
        model_kpt_shape=[3, 3],
        keypoint_labels=["swim_bladder", "eye_left", "eye_right"],
        edges=[[0, 1], [0, 2], [1, 2]],
    )


def _keypoint_fixture():  # type: ignore[no-untyped-def]
    crop_dimensions, crop, crop_manifest = _crop_fixture()
    pose_binding = _pose_binding()
    skeleton_digest = keypoint_skeleton_digest(pose_binding)
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
            [[5, 10], [15, 5], [15, 15]],
            [[5, 10], [15, 5], [15, 15]],
            [[np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]],
        ],
        dtype=np.float32,
    )
    valid = np.all(np.isfinite(points_roi), axis=2)
    origins = crop["roi_coordinates_full"]
    points_img = points_roi + origins.astype(np.float32)[:, None, :]
    pose_success = np.all(valid, axis=1)
    confidences = np.where(valid, np.float32(0.9), np.float32(np.nan))
    pose_confidence = np.where(pose_success, np.float32(0.95), np.float32(np.nan))
    bbox_roi = np.asarray(
        [[1, 1, 19, 19], [1, 1, 19, 19], [1, 1, 19, 19], [np.nan] * 4],
        dtype=np.float32,
    )
    bbox_img = bbox_roi + np.column_stack((origins, origins)).astype(np.float32)
    arrays = {
        "instance_key": crop["instance_key"].copy(),
        "source_crop_row_ids": np.arange(4, dtype=np.int64),
        "source_acquisition_frame_index": crop["source_acquisition_frame_index"].copy(),
        "frame_indices": crop["frame_indices"].copy(),
        "frame_row_offsets": crop["frame_row_offsets"].copy(),
        "source_crop_row_signature": crop["source_row_signature"].copy(),
        "keypoints_roi": points_roi,
        "keypoints_img": points_img,
        "keypoint_confidences": confidences,
        "keypoint_valid": valid,
        "pose_confidence": pose_confidence,
        "pose_bbox_xyxy_roi": bbox_roi,
        "pose_bbox_xyxy_img": bbox_img,
        "pose_success": pose_success,
    }
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
    return dimensions, arrays, crop, crop_manifest, pose_binding, preprocessing


def test_raw_keypoint_storage_and_manifest_round_trip(tmp_path: object) -> None:
    dimensions, arrays, crop, crop_manifest, pose_binding, preprocessing = (
        _keypoint_fixture()
    )
    prepared = prepare_raw_keypoint_v2_snapshot(
        arrays,
        dimensions=dimensions,
        source_crop_arrays=crop,
        source_crop_manifest=crop_manifest,
        pose_model_schema_binding=pose_binding,
        preprocessing=preprocessing,
    )
    root = tmp_path / "keypoints"  # type: ignore[operator]
    publication = publish_selector_ineligible_keypoint_snapshot(
        prepared,
        destination=root / "canary.zarr",
        run_id="yolo_keypoints_v2_canary",
        shadow_root=root,
        created_by="pytest",
    )

    assert validate_keypoint_shadow_publication(publication) == ()
    assert validate_keypoint_run_manifest(publication.manifest) == ()
    assert publication.prepared.crop_source == keypoint_crop_source_from_manifest(
        crop_manifest
    )
    assert len(plan_keypoint_storage(dimensions).entries) == len(
        KEYPOINT_SCHEMA_V2.binding_paths
    )
    zarr_root = zarr.open_group(
        str(publication.output_path), mode="r", use_consolidated=False
    )
    family = zarr_root["keypoints_runs"]
    assert all(
        family.attrs.get(name) is None
        for name in ("latest", "latest_complete", "authoritative_run")
    )
    run = family["yolo_keypoints_v2_canary"]
    assert run.attrs["stage_selector_eligible"] is False
    assert set(run.array_keys()) == set(KEYPOINT_SCHEMA_V2.binding_paths)


def test_legacy_yolo_payload_prepares_exact_v2_without_embedded_qc_or_heading() -> None:
    dimensions, arrays, crop, crop_manifest, pose_binding, preprocessing = (
        _keypoint_fixture()
    )
    legacy = {
        "instance_key": arrays["instance_key"],
        "source_crop_row_ids": arrays["source_crop_row_ids"],
        "source_acquisition_frame_index": arrays["source_acquisition_frame_index"],
        "frame_indices": arrays["frame_indices"],
        "keypoints_roi": arrays["keypoints_roi"].astype(np.float64),
        "keypoints_img": arrays["keypoints_img"].astype(np.float64),
        "keypoint_confidences": arrays["keypoint_confidences"].astype(np.float64),
        "confidence": arrays["pose_confidence"].astype(np.float64),
        "detection_success": arrays["pose_success"],
        "pose_failure_codes": np.asarray([0, 0, 0, 1], dtype=np.uint8),
        "pose_bbox_xyxy_roi": arrays["pose_bbox_xyxy_roi"].astype(np.float64),
        "pose_bbox_xyxy_img": arrays["pose_bbox_xyxy_img"].astype(np.float64),
        "heading": np.zeros(dimensions.n_instances, dtype=np.float64),
        "heading_temporal_outlier": np.zeros(dimensions.n_instances, dtype=bool),
    }

    conversion = prepare_raw_keypoint_v2_from_yolo_arrays(
        legacy,
        dimensions=dimensions,
        source_crop_arrays=crop,
        source_crop_manifest=crop_manifest,
        pose_model_schema_binding=pose_binding,
        preprocessing=preprocessing,
    )

    assert set(conversion.prepared.arrays) == set(KEYPOINT_SCHEMA_V2.binding_paths)
    assert conversion.prepared.arrays["keypoints_roi"].dtype == np.dtype(np.float32)
    assert "heading" not in conversion.prepared.arrays
    assert "heading_temporal_outlier" not in conversion.prepared.arrays
    assert conversion.conversion_receipt["ignored_legacy_families"] == [
        "heading",
        "normalized_coordinates",
        "count_aliases",
        "embedded_quality",
    ]
    assert conversion.conversion_receipt["terminal_pose_failure_evidence"] == {
        "present": True,
        "array_path": "pose_failure_codes",
        "dtype": "uint8",
        "code_map": {
            "0": "none",
            "1": "no_pose_detection_above_threshold",
            "2": "keypoint_payload_missing",
            "3": "keypoint_payload_empty",
            "4": "insufficient_keypoint_count",
        },
        "histogram": {
            "none": 3,
            "no_pose_detection_above_threshold": 1,
            "keypoint_payload_missing": 0,
            "keypoint_payload_empty": 0,
            "insufficient_keypoint_count": 0,
        },
        "public_raw_v2_array": False,
    }

    tampered = dict(legacy)
    tampered["pose_failure_codes"] = np.zeros(
        dimensions.n_instances, dtype=np.uint8
    )
    with pytest.raises(ValueError, match="zero exactly"):
        prepare_raw_keypoint_v2_from_yolo_arrays(
            tampered,
            dimensions=dimensions,
            source_crop_arrays=crop,
            source_crop_manifest=crop_manifest,
            pose_model_schema_binding=pose_binding,
            preprocessing=preprocessing,
        )


def test_keypoint_manifest_rejects_recomputed_nested_tampering(
    tmp_path: object,
) -> None:
    dimensions, arrays, crop, crop_manifest, pose_binding, preprocessing = (
        _keypoint_fixture()
    )
    prepared = prepare_raw_keypoint_v2_snapshot(
        arrays,
        dimensions=dimensions,
        source_crop_arrays=crop,
        source_crop_manifest=crop_manifest,
        pose_model_schema_binding=pose_binding,
        preprocessing=preprocessing,
    )
    root = tmp_path / "keypoints"  # type: ignore[operator]
    publication = publish_selector_ineligible_keypoint_snapshot(
        prepared,
        destination=root / "canary.zarr",
        run_id="yolo_keypoints_v2_canary",
        shadow_root=root,
        created_by="pytest",
    )
    tampered = copy.deepcopy(publication.manifest)
    tampered["payload"]["preprocessing"]["document"]["resize"] = "different"
    tampered["payload"]["storage_plan"]["arrays"][0]["access_unit_semantics"] = (
        "different"
    )
    tampered["payload_digest"] = canonical_json_sha256(tampered["payload"])

    errors = validate_keypoint_run_manifest(tampered)

    assert "Keypoint preprocessing differs from its frozen builder." in errors
    assert "keypoint storage plan differs from planner output" in errors


def test_keypoint_body_frame_requires_no_mask_inputs(tmp_path: object) -> None:
    dimensions, arrays, crop, crop_manifest, pose_binding, preprocessing = (
        _keypoint_fixture()
    )
    prepared = prepare_raw_keypoint_v2_snapshot(
        arrays,
        dimensions=dimensions,
        source_crop_arrays=crop,
        source_crop_manifest=crop_manifest,
        pose_model_schema_binding=pose_binding,
        preprocessing=preprocessing,
    )
    root = tmp_path / "keypoints"  # type: ignore[operator]
    publication = publish_selector_ineligible_keypoint_snapshot(
        prepared,
        destination=root / "canary.zarr",
        run_id="yolo_keypoints_v2_canary",
        shadow_root=root,
        created_by="pytest",
    )
    skeleton_digest = keypoint_skeleton_digest(pose_binding)
    body_recipe = build_keypoint_body_frame_recipe(
        pose_schema=pose_binding["pose_schema"],
        skeleton_digest=skeleton_digest,
        keypoint_count=3,
    )
    body_source = BodyFrameSourceReference(
        stage="keypoints",
        run_name=publication.run_id,
        manifest_digest=canonical_json_sha256(publication.manifest),
        skeleton_id=pose_binding["pose_schema"]["skeleton_id"],
        skeleton_digest=skeleton_digest,
        keypoint_row_signatures_digest=sha256_array(arrays["keypoint_row_signature"]),
    )

    body = prepare_keypoint_body_frame(
        arrays,
        source_dimensions=dimensions,
        source_crop_arrays=crop,
        source=body_source,
        source_manifest=publication.manifest,
        recipe=body_recipe,
    )

    assert body.arrays["axis_valid"].tolist() == [True, True, True, False]
    assert "masks_roi" not in prepared.arrays
    assert "masks_roi" not in body.source_arrays
