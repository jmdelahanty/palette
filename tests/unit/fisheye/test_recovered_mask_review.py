"""Real producer-to-editor checks: coordinates, missing fins, and provenance."""

import numpy as np
import pytest
import zarr

from fisheye.training.recovered_mask_review_payload import (
    build_review_payload,
    array_hashes,
)
from fisheye.training.recover_merged_subject_masks import (
    validate_initial_payload,
    review_tasks,
)
from fisheye.tune import keypoint_review_backend as editor
from fisheye.tune.refined_subject_mask_review import prepare_refined_subject_run


@pytest.fixture
def review_archive(tmp_path):
    path = tmp_path / "training.zarr"
    root = zarr.open_group(str(path), mode="w", use_consolidated=False)
    root.attrs.update(
        {
            "zarr_purpose": "training",
            "stage_selector_eligible": False,
            "schema_id": "palette.training.merged_pose_detect_recovery_source.v1",
        }
    )
    yy, xx = np.mgrid[:128, :128]
    body = ((xx - 64) / 13) ** 2 + ((yy - 60) / 48) ** 2 <= 1
    bladder = ((xx - 64) / 7) ** 2 + ((yy - 43) / 9) ** 2 <= 1
    eyes = (((xx - 58) ** 2 + (yy - 26) ** 2) <= 9) | (
        ((xx - 70) ** 2 + (yy - 26) ** 2) <= 9
    )
    masks = np.repeat(np.stack([body, eyes, bladder])[None], 2, axis=0).astype(np.uint8)
    masks[1, 0, 1, 1] = 1
    arrays = {
        "roi_images": np.repeat((body * 100).astype(np.uint8)[None], 2, axis=0),
        "masks_roi": masks,
        "head_keypoints_roi": np.array(
            [[[64, 43], [58, 26], [70, 26]]] * 2, np.float32
        ),
        "frame_indices": np.array([15, 31]),
        "source_merged_row": np.array([200, 300]),
        "source_pose_local_row": np.array([3, 5]),
        "source_bbox_norm_coords": np.zeros((2, 4)),
        "detection_source": np.array([0, 0], np.int8),
        "target_valid_channels": np.ones((2, 3), bool),
    }
    binding = {
        "mask_run": "original",
        "recording_id": "rec",
        "mask_run_attrs": {"label_schema_id": "subject_v1_union"},
    }
    result = build_review_payload(
        root,
        arrays,
        ("subject_body", "eyes_union", "swim_bladder"),
        binding,
        version="v1",
    )
    return path, root, arrays, result


def test_producer_opens_in_existing_mask_and_pose_editors(review_archive):
    path, root, arrays, result = review_archive
    paths = result["paths"]
    for run in paths.values():
        assert validate_initial_payload(path / run)["valid"]
    session = editor.resolve_review_session(
        str(path), refined_run=paths["pose_edit"].split("/")[1], include_all=True
    )
    assert session.recovered_roi_only and session.roi_coordinates_full is None
    with pytest.raises(RuntimeError, match="recording authority approval"):
        editor.apply_review_status(session, state="approved")
    assert "authoritative_run" not in root["refined_keypoints_runs"].attrs
    payload = editor.load_roi_payload(session, 0)
    assert payload["points"][14:18] == [[None, None]] * 4
    assert len(payload["points"]) == 19
    assert payload["labels"][-1] == "snout_tip"
    assert payload["frame_idx"] == 15
    assert "roi_coordinates_full" not in session.crop
    seed_before = array_hashes(root[paths["seed"]])
    points = session.kp_roi_arr[0]
    with pytest.raises(ValueError, match="incomplete"):
        editor.save_roi_correction(session, position=0, points=points)
    points[14:18] = [[74, 43], [80, 46], [54, 43], [48, 46]]
    missing_snout = points.copy()
    missing_snout[18] = np.nan
    with pytest.raises(ValueError, match="incomplete"):
        editor.save_roi_correction(session, position=0, points=missing_snout)
    outside = points.copy()
    outside[-1, 0] = 128
    with pytest.raises(ValueError, match="inside the crop"):
        editor.save_roi_correction(session, position=0, points=outside)
    saved = editor.save_roi_correction(session, position=0, points=points)
    assert saved["geometry_ok"]
    assert session.refined["training_eligible"][:].tolist() == [True, False]
    assert session.refined["keypoint_origin"][0].tolist() == [1] * 3 + [2] * 11 + [
        3
    ] * 4 + [2]
    assert session.refined["keypoint_manual_edit"][0].tolist() == [False] * 14 + [
        True
    ] * 4 + [False]
    assert array_hashes(root[paths["seed"]]) == seed_before
    points[4, 0] += 1
    points[18, 1] += 1
    editor.save_roi_correction(session, position=0, points=points)
    assert session.refined["keypoint_origin"][0, 4] == 3
    assert session.refined["keypoint_origin"][0, 18] == 3
    reopened = editor.resolve_review_session(
        str(path), refined_run=paths["pose_edit"].split("/")[1], include_all=True
    )
    np.testing.assert_array_equal(reopened.kp_roi_arr[0], points)
    source, refined = prepare_refined_subject_run(
        root,
        subject_run=paths["mask"].split("/")[1],
        refined_run=paths["mask_edit"].split("/")[1],
        components=["subject_body"],
    )
    np.testing.assert_array_equal(refined.group["masks_roi"][:], arrays["masks_roi"])
    assert source.crop_run == paths["crop"].split("/")[1]
    assert result["failures"][0]["source_merged_row"] == 300
    tasks = review_tasks(path, "rec", result, "v1")
    assert tasks[1]["scope"]["target_roi_indices"] == [1]
    editor.mark_no_keypoints(reopened, position=0)
    assert not reopened.refined["training_eligible"][0]
    assert not reopened.refined["usable_keypoints"][0]
    assert array_hashes(root[paths["seed"]]) == seed_before


def test_tampered_and_wrong_coordinate_contracts_refuse(review_archive):
    path, root, _, result = review_archive
    paths = result["paths"]
    root[paths["seed"]]["keypoints_roi"][0, 0] = [2, 3]
    assert not validate_initial_payload(path / paths["seed"])["valid"]
    root[paths["crop"]].attrs["sensor_pixel_origin_available"] = True
    with pytest.raises(ValueError, match="crop-only"):
        editor.resolve_review_session(
            str(path), refined_run=paths["pose_edit"].split("/")[1]
        )


def test_metadata_tampering_is_rejected_even_with_unchanged_arrays(review_archive):
    path, root, _, result = review_archive
    seed = root[result["paths"]["seed"]]
    seed.attrs["keypoint_labels"] = list(reversed(seed.attrs["keypoint_labels"]))
    assert not validate_initial_payload(path / result["paths"]["seed"])["valid"]
