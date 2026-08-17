from __future__ import annotations

import copy
from typing import Mapping

import numpy as np
import pytest
import zarr

from fisheye.analysis_workflows.body_frame_source_handle import (
    BodyFrameSourceHandleError,
    load_body_frame_source_handle,
)
from fisheye.shared.zarr.benchmark_runtime import sha256_array
from fisheye.shared.zarr.body_frame_producer import (
    BodyFrameSourceReference,
    KeypointBodyFrameRecipe,
    prepare_keypoint_body_frame,
)
from fisheye.shared.zarr.body_frame_publication import (
    publish_selector_ineligible_body_frame_snapshot,
)
from fisheye.shared.zarr.keypoint_schema import (
    KeypointDimensions,
    derive_frame_row_offsets,
    derive_keypoint_row_signatures,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256

_SKELETON_DIGEST = "42" * 32
_HEADING_COMPUTATION = {
    "version": 1,
    "enabled": True,
    "origin": {"op": "midpoint", "labels": ["eye_left", "eye_right"]},
    "direction_from": {"op": "keypoint", "label": "swim_bladder"},
    "direction_to": {
        "op": "midpoint",
        "labels": ["eye_left", "eye_right"],
    },
    "dependent_keypoints": ["swim_bladder", "eye_left", "eye_right"],
}


def _published_fixture(
    tmp_path: object,
    *,
    skeleton_id: str = "test_zebrafish_skeleton",
    skeleton_digest: str = _SKELETON_DIGEST,
    heading_computation: Mapping[str, object] | None = None,
):  # type: ignore[no-untyped-def]
    if heading_computation is None:
        heading_computation = _HEADING_COMPUTATION
    dimensions = KeypointDimensions(
        n_frames=3,
        n_instances=3,
        n_keypoints=3,
        source_width=640,
        source_height=480,
    )
    frames = np.asarray([0, 1, 2], dtype=np.int64)
    keys = np.asarray([101, 102, 103], dtype=np.uint64)
    crop_signatures = np.arange(3 * 32, dtype=np.uint8).reshape(3, 32)
    crop = {
        "instance_key": keys.copy(),
        "frame_indices": frames.copy(),
        "source_acquisition_frame_index": frames.copy(),
        "source_row_signature": crop_signatures.copy(),
        "roi_coordinates_full": np.zeros((3, 2), dtype=np.int32),
        "roi_sizes_full": np.full((3, 2), 100, dtype=np.int32),
    }
    keypoints = np.asarray(
        [
            [[20, 20], [10, 10], [10, 30]],
            [[21, 20], [11, 10], [11, 30]],
            [[22, 20], [12, 10], [12, 30]],
        ],
        dtype=np.float32,
    )
    valid = np.ones((3, 3), dtype=bool)
    row_signatures = derive_keypoint_row_signatures(
        instance_key=keys,
        source_crop_row_signature=crop_signatures,
        keypoints_roi=keypoints,
        keypoint_valid=valid,
        skeleton_digest=skeleton_digest,
    )
    source_arrays = {
        "instance_key": keys,
        "source_crop_row_ids": np.arange(3, dtype=np.int64),
        "source_acquisition_frame_index": frames,
        "frame_indices": frames,
        "frame_row_offsets": derive_frame_row_offsets(frames, n_frames=3),
        "source_crop_row_signature": crop_signatures,
        "keypoint_row_signature": row_signatures,
        "keypoints_roi": keypoints,
        "keypoints_img": keypoints,
        "keypoint_confidences": np.ones((3, 3), dtype=np.float32),
        "keypoint_valid": valid,
        "pose_confidence": np.ones(3, dtype=np.float32),
        "pose_bbox_xyxy_roi": np.tile(
            np.asarray([1, 1, 10, 10], dtype=np.float32), (3, 1)
        ),
        "pose_bbox_xyxy_img": np.tile(
            np.asarray([1, 1, 10, 10], dtype=np.float32), (3, 1)
        ),
        "pose_success": np.ones(3, dtype=bool),
    }
    source_manifest = {
        "schema_id": "palette.keypoint.test_source_manifest",
        "schema_version": 1,
        "run_id": "raw_pose_v2_001",
    }
    source = BodyFrameSourceReference(
        stage="keypoints",
        run_name="raw_pose_v2_001",
        manifest_digest=canonical_json_sha256(source_manifest),
        skeleton_id=skeleton_id,
        skeleton_digest=skeleton_digest,
        keypoint_row_signatures_digest=sha256_array(row_signatures),
    )
    recipe = KeypointBodyFrameRecipe(
        swim_bladder_index=0,
        eye_left_index=1,
        eye_right_index=2,
        skeleton_digest=skeleton_digest,
        heading_computation=heading_computation,
    )
    prepared = prepare_keypoint_body_frame(
        source_arrays,
        source_dimensions=dimensions,
        source_crop_arrays=crop,
        source=source,
        source_manifest=source_manifest,
        recipe=recipe,
    )
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
    return destination, publication


def test_loads_exact_immutable_body_frame_source_handle(tmp_path: object) -> None:
    destination, publication = _published_fixture(tmp_path)

    handle = load_body_frame_source_handle(
        destination,
        run_path="analysis/body_frame_runs/body_frame_v1_001",
        expected_selector_eligible=False,
    )

    assert handle.run_name == "body_frame_v1_001"
    assert handle.selector_eligible is False
    assert handle.source_run_path == "keypoints_runs/raw_pose_v2_001"
    assert handle.recipe_id == "keypoint_eye_midpoint_head_axis_camera_xy_v1"
    np.testing.assert_array_equal(
        handle.heading_deg, publication.prepared.arrays["heading_deg"]
    )
    with pytest.raises(ValueError):
        handle.heading_deg[0] = 12.0
    with pytest.raises((AttributeError, TypeError)):
        handle.arrays["heading_deg"] = handle.heading_deg  # type: ignore[index]

    family = zarr.open_group(
        str(destination / "analysis" / "body_frame_runs"),
        mode="r",
        use_consolidated=False,
    )
    assert all(
        family.attrs.get(name) is None
        for name in ("latest", "latest_complete", "latest_pending", "authoritative_run")
    )


def test_rejects_selector_eligibility_mismatch(tmp_path: object) -> None:
    destination, _ = _published_fixture(tmp_path)

    with pytest.raises(BodyFrameSourceHandleError, match="eligibility mismatch"):
        load_body_frame_source_handle(
            destination,
            run_path="analysis/body_frame_runs/body_frame_v1_001",
            expected_selector_eligible=True,
        )


def test_rejects_incomplete_status(tmp_path: object) -> None:
    destination, _ = _published_fixture(tmp_path)
    run = zarr.open_group(
        str(destination / "analysis" / "body_frame_runs" / "body_frame_v1_001"),
        mode="r+",
        use_consolidated=False,
    )
    run.attrs["status"] = "running"

    with pytest.raises(BodyFrameSourceHandleError, match="not complete"):
        load_body_frame_source_handle(
            destination,
            run_path="analysis/body_frame_runs/body_frame_v1_001",
            expected_selector_eligible=False,
            use_consolidated=False,
        )


def test_rejects_run_manifest_payload_digest_tampering(tmp_path: object) -> None:
    destination, _ = _published_fixture(tmp_path)
    run = zarr.open_group(
        str(destination / "analysis" / "body_frame_runs" / "body_frame_v1_001"),
        mode="r+",
        use_consolidated=False,
    )
    manifest = copy.deepcopy(dict(run.attrs["run_manifest"]))
    manifest["payload"]["run_id"] = "another_run"
    run.attrs["run_manifest"] = manifest

    with pytest.raises(BodyFrameSourceHandleError, match="payload_digest mismatch"):
        load_body_frame_source_handle(
            destination,
            run_path="analysis/body_frame_runs/body_frame_v1_001",
            expected_selector_eligible=False,
            use_consolidated=False,
        )


def test_rejects_tampered_row_identity_against_manifest_array_digest(
    tmp_path: object,
) -> None:
    destination, _ = _published_fixture(tmp_path)
    run = zarr.open_group(
        str(destination / "analysis" / "body_frame_runs" / "body_frame_v1_001"),
        mode="r+",
        use_consolidated=False,
    )
    values = np.asarray(run["instance_key"][...])
    run["instance_key"][:] = values[::-1]

    with pytest.raises(
        BodyFrameSourceHandleError, match="digest mismatch at instance_key"
    ):
        load_body_frame_source_handle(
            destination,
            run_path="analysis/body_frame_runs/body_frame_v1_001",
            expected_selector_eligible=False,
            use_consolidated=False,
        )
