from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fisheye.utils import finalize_recording_keypoint_v2_benchmark_adapter as mod


def test_row_lookup_reorders_by_stable_instance_key() -> None:
    available = np.asarray([90, 10, 40], dtype=np.uint64)
    requested = np.asarray([10, 40, 90], dtype=np.uint64)

    rows = mod._row_lookup(requested=requested, available=available)

    np.testing.assert_array_equal(rows, np.asarray([1, 2, 0], dtype=np.int64))


@pytest.mark.parametrize(
    "requested,available,message",
    [
        ([10, 10], [10, 20], "Requested crop-v2 instance keys are not unique"),
        ([10, 20], [10, 10], "Historical keypoint instance keys are not unique"),
        ([10, 30], [10, 20], "do not cover"),
        ([10], [10, 20], "contain rows absent"),
    ],
)
def test_row_lookup_fails_closed_on_nonbijective_keys(
    requested: list[int],
    available: list[int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        mod._row_lookup(
            requested=np.asarray(requested, dtype=np.uint64),
            available=np.asarray(available, dtype=np.uint64),
        )


def test_node_local_scratch_rejects_shared_storage(tmp_path: Path) -> None:
    assert mod._require_node_local_scratch(tmp_path) == tmp_path.resolve()
    with pytest.raises(ValueError, match="node-local"):
        mod._require_node_local_scratch(Path("/groups/example"))


def test_source_selector_evidence_accepts_eligible_superseded_run() -> None:
    evidence = mod._source_selector_evidence(
        {
            "latest": "newer",
            "latest_complete": "newer",
            "latest_pending": None,
        },
        run_id="historical",
        stage_selector_eligible=True,
    )

    assert evidence == {
        "run_id": "historical",
        "stage_selector_eligible": True,
        "selectors": {
            "latest": "newer",
            "latest_complete": "newer",
            "latest_pending": None,
        },
        "selected_by": [],
        "explicit_metadata_pin_required": True,
    }


@pytest.mark.parametrize("selector", mod._SELECTOR_ATTRIBUTE_NAMES)
def test_source_selector_evidence_rejects_selected_source(selector: str) -> None:
    with pytest.raises(ValueError, match=f"currently selected by {selector}"):
        mod._source_selector_evidence(
            {selector: "historical"},
            run_id="historical",
            stage_selector_eligible=True,
        )


def test_source_selector_evidence_requires_exact_eligibility_bool() -> None:
    with pytest.raises(ValueError, match="must be an exact bool"):
        mod._source_selector_evidence(
            {},
            run_id="historical",
            stage_selector_eligible=1,
        )


def test_rebase_legacy_roi_coordinates_preserves_camera_pixels() -> None:
    points = np.asarray([[[4.5, 8.0]], [[9.0, 2.0]]], dtype=np.float64)
    boxes = np.asarray(
        [[1.0, 2.0, 10.0, 12.0], [3.0, 4.0, 11.0, 13.0]],
        dtype=np.float64,
    )
    old_origins = np.asarray([[100, 200], [300, 400]], dtype=np.int32)
    new_origins = np.asarray([[100, 199], [299, 400]], dtype=np.int32)
    old_camera_points = points + old_origins[:, None, :]

    rebased_points, rebased_boxes, evidence = (
        mod._rebase_legacy_roi_coordinates(
            keypoints_roi=points,
            pose_bbox_xyxy_roi=boxes,
            old_origins=old_origins,
            new_origins=new_origins,
        )
    )

    np.testing.assert_array_equal(
        rebased_points + new_origins[:, None, :], old_camera_points
    )
    np.testing.assert_array_equal(
        rebased_boxes,
        boxes + np.asarray([[0, 1, 0, 1], [1, 0, 1, 0]]),
    )
    assert evidence["rebased_row_count"] == 2
    assert evidence["maximum_origin_delta_pixels_observed"] == 1


def test_rebase_legacy_roi_coordinates_rejects_larger_geometry_change() -> None:
    with pytest.raises(ValueError, match="more than the 1-pixel"):
        mod._rebase_legacy_roi_coordinates(
            keypoints_roi=np.zeros((1, 5, 2), dtype=np.float64),
            pose_bbox_xyxy_roi=np.zeros((1, 4), dtype=np.float64),
            old_origins=np.asarray([[10, 20]], dtype=np.int32),
            new_origins=np.asarray([[8, 20]], dtype=np.int32),
        )


def test_rebound_receipt_changes_only_output_locations(tmp_path: Path) -> None:
    payload = {
        "status": "complete",
        "selector_eligible": False,
        "registry_registered": False,
        "crop": {},
        "clip_inputs": [],
        "preparation": {},
        "outputs": {
            name: {
                "path": f"/tmp/local/{name}.zarr",
                "selector_eligible": False,
            }
            for name in (
                "raw_keypoints",
                "keypoint_quality",
                "refined_keypoints",
                "body_frame",
            )
        },
        "selector_activation": "none_direct_path_only",
        "production_state_changes": [],
    }
    receipt = {
        "schema_id": "palette.keypoint.clipped_recording_finalization",
        "schema_version": 1,
        "digest_algorithm": "sha256_canonical_json_v1",
        "payload_digest": mod.canonical_json_sha256(payload),
        "payload": payload,
    }

    rebound = mod._rebind_finalization_receipt(
        receipt,
        destination=tmp_path / "published",
    )

    assert rebound["payload_digest"] != receipt["payload_digest"]
    for binding in rebound["payload"]["outputs"].values():
        assert Path(binding["path"]).parent == tmp_path / "published"
