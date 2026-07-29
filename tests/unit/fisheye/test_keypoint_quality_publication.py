from __future__ import annotations

import copy

import numpy as np
import pytest
import zarr

from fisheye.shared.zarr.benchmark_runtime import sha256_array
from fisheye.shared.zarr.keypoint_quality_manifest import (
    validate_keypoint_quality_run_manifest,
)
from fisheye.shared.zarr.keypoint_quality_producer import (
    KEYPOINT_FLAG_LOW_CONFIDENCE,
    KEYPOINT_FLAG_SOURCE_INVALID,
    POSE_FLAG_INSUFFICIENT_VALID_LANDMARKS,
    POSE_FLAG_SOURCE_FAILED,
    ObservationLocalKeypointQualityPolicy,
    prepare_observation_local_keypoint_quality,
)
from fisheye.shared.zarr.keypoint_quality_publication import (
    publish_selector_ineligible_keypoint_quality_snapshot,
    validate_keypoint_quality_shadow_publication,
)
from fisheye.shared.zarr.keypoint_quality_schema import (
    KEYPOINT_QUALITY_SCHEMA_V1,
    KeypointQualityDimensions,
    KeypointQualitySourceReference,
)
from fisheye.shared.zarr.keypoint_quality_storage import (
    plan_keypoint_quality_storage,
)
from fisheye.shared.zarr.keypoint_schema import (
    KEYPOINT_SCHEMA_V2,
    KeypointDimensions,
    derive_frame_row_offsets,
    derive_keypoint_row_signatures,
)
from fisheye.shared.zarr.manifest_digest import (
    canonical_json_sha256,
)


SKELETON_DIGEST = "42" * 32


def _source_fixture() -> tuple[
    KeypointDimensions,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, object],
    KeypointQualitySourceReference,
]:
    dimensions = KeypointDimensions(
        n_frames=4,
        n_instances=4,
        n_keypoints=3,
        source_width=640,
        source_height=480,
    )
    frames = np.asarray([0, 0, 2, 3], dtype=np.int64)
    keys = np.asarray([101, 102, 201, 301], dtype=np.uint64)
    origins = np.asarray(
        [[10, 20], [100, 50], [200, 100], [300, 200]], dtype=np.int32
    )
    sizes = np.asarray([[64, 48], [80, 60], [96, 72], [112, 88]], dtype=np.int32)
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
            [[5, 6], [10, 11], [15, 16]],
            [[np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]],
            [[20, 21], [30, 31], [np.nan, np.nan]],
            [[40, 41], [50, 51], [60, 61]],
        ],
        dtype=np.float32,
    )
    valid = np.all(np.isfinite(points_roi), axis=2)
    points_img = points_roi + origins.astype(np.float32)[:, None, :]
    confidence = np.asarray(
        [
            [0.95, 0.90, 0.70],
            [np.nan, np.nan, np.nan],
            [0.75, 0.85, np.nan],
            [0.99, 0.40, 0.97],
        ],
        dtype=np.float32,
    )
    bbox_roi = np.asarray(
        [
            [1, 2, 30, 35],
            [np.nan, np.nan, np.nan, np.nan],
            [4, 5, 60, 65],
            [6, 7, 100, 80],
        ],
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
        "keypoint_confidences": confidence,
        "keypoint_valid": valid,
        "pose_confidence": np.asarray(
            [0.92, np.nan, 0.81, 0.97], dtype=np.float32
        ),
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
    source = KeypointQualitySourceReference(
        run_name="raw_pose_v2_001",
        manifest_digest=canonical_json_sha256(source_manifest),
        skeleton_id="sleepyfish_three_point_test",
        skeleton_digest=SKELETON_DIGEST,
        keypoint_row_signatures_digest=sha256_array(row_signatures),
    )
    return dimensions, arrays, crop, source_manifest, source


def _prepared() -> tuple[object, dict[str, object]]:
    dimensions, arrays, crop, source_manifest, source = _source_fixture()
    prepared = prepare_observation_local_keypoint_quality(
        arrays,
        source_dimensions=dimensions,
        source_crop_arrays=crop,
        source=source,
        skeleton_digest=SKELETON_DIGEST,
        policy=ObservationLocalKeypointQualityPolicy(
            confidence_threshold=0.8,
            minimum_valid_keypoints=2,
        ),
    )
    return prepared, source_manifest


def test_observation_local_producer_keeps_decisions_and_heading_separate() -> None:
    prepared, _ = _prepared()
    arrays = prepared.arrays

    assert arrays["frame_row_offsets"].tolist() == [0, 2, 2, 3, 4]
    assert arrays["proposed_keypoint_valid"].tolist() == [
        [True, True, False],
        [False, False, False],
        [False, True, False],
        [True, False, True],
    ]
    assert arrays["proposed_pose_usable"].tolist() == [True, False, False, True]
    assert arrays["keypoint_quality_flags"][0, 2] == KEYPOINT_FLAG_LOW_CONFIDENCE
    assert arrays["keypoint_quality_flags"][1, 0] == KEYPOINT_FLAG_SOURCE_INVALID
    assert arrays["pose_quality_flags"][1] == (
        POSE_FLAG_SOURCE_FAILED | POSE_FLAG_INSUFFICIENT_VALID_LANDMARKS
    )
    assert "heading" not in " ".join(KEYPOINT_QUALITY_SCHEMA_V1.binding_paths)
    assert set(arrays) == set(KEYPOINT_QUALITY_SCHEMA_V1.binding_paths)


def test_storage_plan_is_byte_derived_and_preserves_complete_rows() -> None:
    dimensions = KeypointQualityDimensions(
        n_frames=1_000_000,
        n_instances=1_000_000,
        n_keypoints=5,
        n_keypoint_metrics=1,
        n_pose_metrics=1,
    )
    plans = plan_keypoint_quality_storage(dimensions)

    assert len(plans.entries) == 13
    assert plans.sharded_array_count == 10
    for entry in plans.entries:
        assert entry.plan.chunk_shape[1:] == tuple(
            max(1, value) for value in entry.plan.logical_shape[1:]
        )
        if entry.plan.shard_shape is None:
            assert entry.plan.estimated_payload_objects == 1
    offsets = next(
        entry for entry in plans.entries if entry.rule.path == "frame_row_offsets"
    )
    assert offsets.rule.access.value == "eager"
    assert offsets.plan.chunk_nbytes >= 512 * 1024


def test_selector_ineligible_publication_round_trip(tmp_path: object) -> None:
    prepared, source_manifest = _prepared()
    root = tmp_path / "quality_root"  # type: ignore[operator]
    destination = root / "fixture.zarr"

    publication = publish_selector_ineligible_keypoint_quality_snapshot(
        prepared,
        source_manifest=source_manifest,
        destination=destination,
        run_id="quality_v1_001",
        shadow_root=root,
        created_by="pytest",
    )

    assert validate_keypoint_quality_shadow_publication(publication) == ()
    assert validate_keypoint_quality_run_manifest(publication.manifest) == ()
    family = zarr.open_group(
        str(destination / "keypoint_quality_runs"),
        mode="r",
        use_consolidated=False,
    )
    assert all(family.attrs.get(name) is None for name in (
        "latest",
        "latest_complete",
        "latest_pending",
        "authoritative_run",
    ))
    run = family["quality_v1_001"]
    assert run.attrs["stage_selector_eligible"] is False
    assert run.attrs["status"] == "complete"
    assert set(run.array_keys()) == set(KEYPOINT_QUALITY_SCHEMA_V1.binding_paths)
    for path in KEYPOINT_QUALITY_SCHEMA_V1.binding_paths:
        np.testing.assert_array_equal(
            np.asarray(run[path][...]),
            np.asarray(prepared.arrays[path]),
        )


def test_manifest_rejects_recomputed_digest_nested_tampering(tmp_path: object) -> None:
    prepared, source_manifest = _prepared()
    root = tmp_path / "quality_root"  # type: ignore[operator]
    publication = publish_selector_ineligible_keypoint_quality_snapshot(
        prepared,
        source_manifest=source_manifest,
        destination=root / "fixture.zarr",
        run_id="quality_v1_001",
        shadow_root=root,
        created_by="pytest",
    )
    tampered = copy.deepcopy(publication.manifest)
    tampered["payload"]["storage_plan"]["arrays"][0][
        "access_unit_semantics"
    ] = "tampered"
    logical_content = tampered["payload"]["logical_content"]
    logical_content["document"]["arrays"]["instance_key"]["shape"] = [999]
    logical_content["digest"] = canonical_json_sha256(logical_content["document"])
    tampered["payload_digest"] = canonical_json_sha256(tampered["payload"])

    errors = validate_keypoint_quality_run_manifest(tampered)

    assert "quality storage plan differs from planner output" in errors
    assert "quality logical_content shape mismatch at instance_key" in errors


def test_producer_rejects_source_signature_digest_mismatch() -> None:
    dimensions, arrays, crop, _, source = _source_fixture()
    wrong_source = KeypointQualitySourceReference(
        run_name=source.run_name,
        manifest_digest=source.manifest_digest,
        skeleton_id=source.skeleton_id,
        skeleton_digest=source.skeleton_digest,
        keypoint_row_signatures_digest="00" * 32,
    )

    with pytest.raises(ValueError, match="signature digest differs"):
        prepare_observation_local_keypoint_quality(
            arrays,
            source_dimensions=dimensions,
            source_crop_arrays=crop,
            source=wrong_source,
            skeleton_digest=SKELETON_DIGEST,
        )
