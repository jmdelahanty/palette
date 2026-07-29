from __future__ import annotations

import copy

import numpy as np
import zarr

from fisheye.shared.zarr.benchmark_runtime import sha256_array
from fisheye.shared.zarr.body_frame_manifest import validate_body_frame_run_manifest
from fisheye.shared.zarr.body_frame_producer import (
    BodyFrameSourceReference,
    KeypointBodyFrameRecipe,
    build_keypoint_body_frame_recipe,
    prepare_keypoint_body_frame,
)
from fisheye.shared.zarr.body_frame_publication import (
    publish_selector_ineligible_body_frame_snapshot,
    validate_body_frame_shadow_publication,
)
from fisheye.shared.zarr.body_frame_schema import BODY_FRAME_SCHEMA_V1
from fisheye.shared.zarr.body_frame_storage import plan_body_frame_storage
from fisheye.shared.zarr.keypoint_schema import (
    KEYPOINT_SCHEMA_V2,
    REFINED_KEYPOINT_SCHEMA_V2,
    KeypointDimensions,
    derive_frame_row_offsets,
    derive_keypoint_row_signatures,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


SKELETON_DIGEST = "42" * 32
HEADING_COMPUTATION = {
    "version": 1,
    "enabled": True,
    "origin": {"op": "midpoint", "labels": ["eye_left", "eye_right"]},
    "direction_from": {"op": "keypoint", "label": "swim_bladder"},
    "direction_to": {"op": "midpoint", "labels": ["eye_left", "eye_right"]},
    "dependent_keypoints": ["swim_bladder", "eye_left", "eye_right"],
}


def _source_fixture():  # type: ignore[no-untyped-def]
    dimensions = KeypointDimensions(
        n_frames=4,
        n_instances=4,
        n_keypoints=3,
        source_width=640,
        source_height=480,
    )
    frames = np.asarray([0, 0, 2, 3], dtype=np.int64)
    keys = np.asarray([101, 102, 201, 301], dtype=np.uint64)
    origins = np.asarray([[10, 20], [100, 50], [200, 100], [300, 200]], dtype=np.int32)
    sizes = np.full((4, 2), 100, dtype=np.int32)
    crop_signatures = np.arange(4 * 32, dtype=np.uint8).reshape(4, 32)
    crop = {
        "instance_key": keys.copy(),
        "frame_indices": frames.copy(),
        "source_acquisition_frame_index": frames.copy(),
        "source_row_signature": crop_signatures.copy(),
        "roi_coordinates_full": origins,
        "roi_sizes_full": sizes,
    }
    points_roi = np.asarray(
        [
            [[5, 15], [15, 10], [15, 20]],
            [[np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]],
            [[20, 20], [25, 15], [15, 25]],
            [[40, 50], [45, 35], [35, 45]],
        ],
        dtype=np.float32,
    )
    valid = np.all(np.isfinite(points_roi), axis=2)
    points_img = points_roi + origins.astype(np.float32)[:, None, :]
    bbox_roi = np.asarray(
        [[1, 2, 50, 60], [np.nan] * 4, [2, 3, 60, 70], [5, 6, 80, 90]],
        dtype=np.float32,
    )
    bbox_img = bbox_roi + np.column_stack((origins, origins)).astype(np.float32)
    row_signatures = derive_keypoint_row_signatures(
        instance_key=keys,
        source_crop_row_signature=crop_signatures,
        keypoints_roi=points_roi,
        keypoint_valid=valid,
        skeleton_digest=SKELETON_DIGEST,
    )
    arrays = {
        "instance_key": keys,
        "source_crop_row_ids": np.arange(4, dtype=np.int64),
        "source_acquisition_frame_index": frames.copy(),
        "frame_indices": frames,
        "frame_row_offsets": derive_frame_row_offsets(frames, n_frames=4),
        "source_crop_row_signature": crop_signatures,
        "keypoint_row_signature": row_signatures,
        "keypoints_roi": points_roi,
        "keypoints_img": points_img,
        "keypoint_confidences": np.asarray(
            [[0.9] * 3, [np.nan] * 3, [0.8] * 3, [0.95] * 3], dtype=np.float32
        ),
        "keypoint_valid": valid,
        "pose_confidence": np.asarray([0.9, np.nan, 0.8, 0.95], dtype=np.float32),
        "pose_bbox_xyxy_roi": bbox_roi,
        "pose_bbox_xyxy_img": bbox_img,
        "pose_success": np.asarray([True, False, True, True], dtype=bool),
    }
    source_manifest: dict[str, object] = {
        "schema_id": "palette.keypoint.test_source_manifest",
        "schema_version": 1,
        "run_id": "raw_pose_v2_001",
        "logical_schema": KEYPOINT_SCHEMA_V2.as_manifest(dimensions=dimensions),
    }
    source = BodyFrameSourceReference(
        stage="keypoints",
        run_name="raw_pose_v2_001",
        manifest_digest=canonical_json_sha256(source_manifest),
        skeleton_id="sleepyfish_three_point_test",
        skeleton_digest=SKELETON_DIGEST,
        keypoint_row_signatures_digest=sha256_array(row_signatures),
    )
    recipe = KeypointBodyFrameRecipe(
        swim_bladder_index=0,
        eye_left_index=1,
        eye_right_index=2,
        skeleton_digest=SKELETON_DIGEST,
        heading_computation=HEADING_COMPUTATION,
    )
    return dimensions, arrays, crop, source_manifest, source, recipe


def _prepared():  # type: ignore[no-untyped-def]
    dimensions, arrays, crop, source_manifest, source, recipe = _source_fixture()
    prepared = prepare_keypoint_body_frame(
        arrays,
        source_dimensions=dimensions,
        source_crop_arrays=crop,
        source=source,
        source_manifest=source_manifest,
        recipe=recipe,
    )
    return prepared, source_manifest


def _refined_prepared():  # type: ignore[no-untyped-def]
    dimensions, arrays, crop, _, raw_source, recipe = _source_fixture()
    refined = dict(arrays)
    source_success = refined.pop("pose_success")
    refined.update(
        {
            "source_success": source_success,
            "refined_success": source_success.copy(),
            "keypoint_edit_flags": np.zeros(
                (dimensions.n_instances, dimensions.n_keypoints), dtype=bool
            ),
            "flip_corrected": np.zeros(dimensions.n_instances, dtype=bool),
            "confidence_valid": source_success.copy(),
            "geometry_valid": source_success.copy(),
            "usable_keypoints": source_success.copy(),
            "review_state_codes": np.zeros(dimensions.n_instances, dtype=np.uint8),
            "reason_codes": np.zeros(dimensions.n_instances, dtype=np.uint16),
        }
    )
    source_manifest: dict[str, object] = {
        "schema_id": "palette.refined_keypoint.test_source_manifest",
        "schema_version": 1,
        "run_id": "refined_pose_v2_001",
        "logical_schema": REFINED_KEYPOINT_SCHEMA_V2.as_manifest(dimensions=dimensions),
    }
    source = BodyFrameSourceReference(
        stage="refined_keypoints",
        run_name="refined_pose_v2_001",
        manifest_digest=canonical_json_sha256(source_manifest),
        skeleton_id=raw_source.skeleton_id,
        skeleton_digest=raw_source.skeleton_digest,
        keypoint_row_signatures_digest=raw_source.keypoint_row_signatures_digest,
    )
    return prepare_keypoint_body_frame(
        refined,
        source_dimensions=dimensions,
        source_crop_arrays=crop,
        source=source,
        source_manifest=source_manifest,
        recipe=recipe,
        review_state_map={0: "unreviewed"},
        reason_code_map={0: "none"},
    )


def test_producer_binds_rows_and_uses_negative_determinant_camera_axes() -> None:
    prepared, _ = _prepared()
    arrays = prepared.arrays

    assert arrays["source_keypoint_row_ids"].tolist() == [0, 1, 2, 3]
    assert arrays["frame_row_offsets"].tolist() == [0, 2, 2, 3, 4]
    assert arrays["axis_valid"].tolist() == [True, False, False, True]
    np.testing.assert_array_equal(arrays["forward_axis_xy"][0], [1.0, 0.0])
    np.testing.assert_array_equal(arrays["left_axis_xy"][0], [0.0, -1.0])
    valid = arrays["axis_valid"]
    determinant = (
        arrays["forward_axis_xy"][valid, 0] * arrays["left_axis_xy"][valid, 1]
        - arrays["forward_axis_xy"][valid, 1] * arrays["left_axis_xy"][valid, 0]
    )
    np.testing.assert_allclose(determinant, -1.0, atol=5e-6, rtol=0.0)
    assert np.all(np.isnan(arrays["origin_xy"][~valid]))
    assert np.all(np.isnan(arrays["heading_deg"][~valid]))
    assert prepared.recipe.as_manifest()["left_axis"] == (
        "fixed_clockwise_90_degrees_in_camera_xy"
    )
    assert prepared.recipe.as_manifest()["heading_computation"] == HEADING_COMPUTATION


def test_recipe_resolves_indices_from_canonical_pose_schema() -> None:
    pose_schema = {
        "keypoint_labels": ["eye_right", "swim_bladder", "eye_left", "tail"],
        "metadata": {"heading_computation": HEADING_COMPUTATION},
    }

    recipe = build_keypoint_body_frame_recipe(
        pose_schema=pose_schema,
        skeleton_digest=SKELETON_DIGEST,
        keypoint_count=4,
    )

    assert recipe.swim_bladder_index == 1
    assert recipe.eye_left_index == 2
    assert recipe.eye_right_index == 0
    assert recipe.as_manifest()["heading_computation_digest"] == (
        canonical_json_sha256(HEADING_COMPUTATION)
    )


def test_recipe_rejects_noncanonical_or_disabled_heading_computation() -> None:
    noncanonical = copy.deepcopy(HEADING_COMPUTATION)
    noncanonical["enabled"] = False

    try:
        KeypointBodyFrameRecipe(
            swim_bladder_index=0,
            eye_left_index=1,
            eye_right_index=2,
            skeleton_digest=SKELETON_DIGEST,
            heading_computation=noncanonical,
        )
    except ValueError as exc:
        assert "exact enabled three-landmark" in str(exc)
    else:
        raise AssertionError("Disabled heading computation must fail closed.")


def test_producer_accepts_one_complete_refined_keypoint_v2_snapshot() -> None:
    prepared = _refined_prepared()

    assert prepared.source.stage == "refined_keypoints"
    assert prepared.source.run_path == "refined_keypoints_runs/refined_pose_v2_001"
    assert prepared.arrays["source_keypoint_row_ids"].tolist() == [0, 1, 2, 3]
    assert prepared.arrays["axis_valid"].tolist() == [True, False, False, True]


def test_storage_plan_is_byte_derived_and_complete_row_aligned() -> None:
    prepared, _ = _prepared()
    plans = plan_body_frame_storage(prepared.dimensions)

    assert len(plans.entries) == len(BODY_FRAME_SCHEMA_V1.binding_paths)
    for entry in plans.entries:
        assert entry.plan.chunk_shape[1:] == tuple(
            max(1, value) for value in entry.plan.logical_shape[1:]
        )
    offsets = next(
        entry for entry in plans.entries if entry.rule.path == "frame_row_offsets"
    )
    assert offsets.rule.access.value == "eager"


def test_selector_ineligible_body_frame_publication_round_trip(
    tmp_path: object,
) -> None:
    prepared, source_manifest = _prepared()
    root = tmp_path / "body_frame_root"  # type: ignore[operator]
    destination = root / "fixture.zarr"

    publication = publish_selector_ineligible_body_frame_snapshot(
        prepared,
        source_manifest=source_manifest,
        destination=destination,
        run_id="body_frame_v1_001",
        shadow_root=root,
        created_by="pytest",
    )

    assert validate_body_frame_shadow_publication(publication) == ()
    assert validate_body_frame_run_manifest(publication.manifest) == ()
    family = zarr.open_group(
        str(destination / "analysis" / "body_frame_runs"),
        mode="r",
        use_consolidated=False,
    )
    assert all(
        family.attrs.get(name) is None
        for name in ("latest", "latest_complete", "latest_pending", "authoritative_run")
    )
    run = family["body_frame_v1_001"]
    assert run.attrs["stage_selector_eligible"] is False
    assert run.attrs["keypoint_authority"] is False
    assert set(run.array_keys()) == set(BODY_FRAME_SCHEMA_V1.binding_paths)
    for path in BODY_FRAME_SCHEMA_V1.binding_paths:
        np.testing.assert_array_equal(
            np.asarray(run[path][...]),
            np.asarray(prepared.arrays[path]),
        )


def test_manifest_rejects_recomputed_digest_recipe_and_storage_tampering(
    tmp_path: object,
) -> None:
    prepared, source_manifest = _prepared()
    root = tmp_path / "body_frame_root"  # type: ignore[operator]
    publication = publish_selector_ineligible_body_frame_snapshot(
        prepared,
        source_manifest=source_manifest,
        destination=root / "fixture.zarr",
        run_id="body_frame_v1_001",
        shadow_root=root,
        created_by="pytest",
    )
    tampered = copy.deepcopy(publication.manifest)
    tampered["payload"]["heading_recipe"]["left_axis"] = "counterclockwise"
    tampered["payload"]["heading_recipe"]["heading_computation"][
        "dependent_keypoints"
    ] = ["eye_left"]
    tampered["payload"]["storage_plan"]["arrays"][0]["access_unit_semantics"] = (
        "tampered"
    )
    tampered["payload_digest"] = canonical_json_sha256(tampered["payload"])

    errors = validate_body_frame_run_manifest(tampered)

    assert any(
        "exact enabled three-landmark" in error
        or error == "Body-frame recipe differs from its frozen builder."
        for error in errors
    )
    assert "body-frame storage plan differs from planner output" in errors
