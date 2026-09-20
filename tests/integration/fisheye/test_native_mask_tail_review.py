"""Native row joins, label preservation, and the real publication/editor path."""

import numpy as np
import pytest
import zarr

from fisheye.training.native_mask_tail_review import generate_native_mask_review
from fisheye.tune.keypoint_review_backend import (
    resolve_review_session,
    save_roi_correction,
)


@pytest.fixture
def native_source(tmp_path):
    path = tmp_path / "native_training.zarr"
    root = zarr.open_group(str(path), mode="w", use_consolidated=False)
    root.attrs.update(zarr_purpose="training", recording_id="native")
    crop = root.create_group("crop_runs/crop")
    crop.create_array("roi_images", data=np.zeros((2, 128, 128), np.uint8))
    crop.create_array("frame_indices", data=np.array([17, 29], np.int64))
    crop.create_array("roi_coordinates_full", data=np.array([[10, 20], [30, 40]]))
    crop.create_array("bbox_norm_coords", data=np.ones((2, 4), np.float32) / 2)
    crop.create_array("detection_source", data=np.zeros(2, np.int8))
    yy, xx = np.mgrid[:128, :128]
    body = ((xx - 64) / 15) ** 2 + ((yy - 68) / 44) ** 2 <= 1
    swim = ((xx - 64) / 8) ** 2 + ((yy - 46) / 9) ** 2 <= 1
    mask = root.create_group("refined_subject_masks_runs/masks")
    labels = ["subject_body", "eyes_union", "swim_bladder"]
    approved = {"state": "approved", "method": "manual", "intended_use": "training"}
    mask.attrs.update(
        source_crop_run="crop",
        mask_labels=labels,
        label_schema_id="subject_v1_union",
        component_review_statuses={name: approved for name in labels},
        palette_run_completion_status="complete",
    )
    mask.create_array(
        "masks_roi",
        data=np.repeat(np.stack([body, swim, swim])[None], 2, axis=0).astype(np.uint8),
    )
    mask.create_array("available_channels", data=np.ones(3, bool))
    mask.create_array("source_crop_row_ids", data=np.array([1, 0], np.int64))
    mask.create_array("frame_indices", data=np.array([29, 17], np.int64))
    pose = root.create_group("refined_keypoints_runs/pose")
    pose.attrs.update(
        source_crop_run="crop",
        keypoint_labels=[
            "eye_right",
            "snout_tip",
            "swim_bladder",
            "eye_left",
            "left_pectoral_fin_tip",
        ],
        keypoint_review_status=approved,
        palette_run_completion_status="complete",
    )
    points = np.array(
        [
            [[74, 34], [64, 25], [64, 46], [54, 34], [80, 50]],
            [[74, 34], [64, 26], [64, 46], [54, 34], [81, 50]],
        ],
        np.float32,
    )
    pose.create_array("keypoints_roi", data=points)
    pose.create_array("source_crop_row_ids", data=np.arange(2, dtype=np.int64))
    pose.create_array("frame_indices", data=np.array([17, 29], np.int64))
    for family in ("crop_runs", "refined_keypoints_runs", "refined_subject_masks_runs"):
        root[family].attrs.update(latest="unchanged", authoritative_run="unchanged")
    return path


def produce(path, **kwargs):
    return generate_native_mask_review(
        archive=path, mask_run="masks", keypoint_run="pose", version="v1", **kwargs
    )


def test_native_publication_and_editor_preserve_source_identity(native_source):
    result = produce(native_source, apply=True)
    root = zarr.open_group(str(native_source), mode="r", use_consolidated=True)
    crop, pose = (root[result["paths"][name]] for name in ("crop", "pose_edit"))
    assert crop.attrs["frame_index_domain"] == "source_crop_frame_index"
    np.testing.assert_array_equal(crop["frame_indices"][:], [29, 17])
    np.testing.assert_array_equal(crop["source_training_crop_row_ids"][:], [1, 0])
    np.testing.assert_array_equal(
        crop["source_roi_coordinates_full"][:], [[30, 40], [10, 20]]
    )
    np.testing.assert_array_equal(pose["keypoints_roi"][:, 18], [[64, 26], [64, 25]])
    np.testing.assert_array_equal(pose["keypoints_roi"][:, 17], [[81, 50], [80, 50]])
    assert (pose["keypoint_origin"][:, 18] == 4).all()
    assert not pose["training_eligible"][:].any()
    assert root["crop_runs"].attrs["latest"] == "unchanged"
    assert root["refined_keypoints_runs"].attrs["authoritative_run"] == "unchanged"
    session = resolve_review_session(
        str(native_source),
        refined_run=result["paths"]["pose_edit"].split("/")[1],
        crop_run=result["paths"]["crop"].split("/")[1],
        include_all=True,
    )
    points = np.asarray(session.kp_roi_arr[0]).copy()
    points[~np.isfinite(points).all(axis=1)] = [64, 60]
    save_roi_correction(session, position=0, points=points)
    assert session.refined["training_eligible"][0]
    assert session.refined["keypoint_origin"][0, 18] == 4
    assert session.refined["keypoint_origin"][0, 14] == 3
    with pytest.raises(ValueError, match="incomplete keypoints"):
        save_roi_correction(session, position=0, points=np.full((19, 2), np.nan))
    with pytest.raises(ValueError, match="visible inside"):
        save_roi_correction(session, position=0, points=np.full((19, 2), 128))
    with pytest.raises(FileExistsError):
        produce(native_source, apply=True)


@pytest.mark.parametrize(
    "fault",
    [
        "pending",
        "duplicate_rows",
        "wrong_frames",
        "wrong_crop",
        "missing_head",
        "nonbinary",
    ],
)
def test_native_source_refuses_bad_inputs_without_publication(native_source, fault):
    root = zarr.open_group(str(native_source), mode="a", use_consolidated=False)
    mask, pose = (
        root["refined_subject_masks_runs/masks"],
        root["refined_keypoints_runs/pose"],
    )
    if fault == "pending":
        mask.attrs["component_review_statuses"] = {}
    elif fault == "duplicate_rows":
        mask["source_crop_row_ids"][:] = [0, 0]
    elif fault == "wrong_frames":
        mask["frame_indices"][:] = [17, 29]
    elif fault == "wrong_crop":
        pose.attrs["source_crop_run"] = "different"
    elif fault == "missing_head":
        pose.attrs["keypoint_labels"] = ["a", "b", "c", "d", "e"]
    else:
        mask["masks_roi"][0, 0, 0, 0] = 2
    with pytest.raises(ValueError):
        produce(native_source, apply=True)
    assert not (native_source / "crop_runs/mask_tail_full_roi_v1").exists()


def test_native_source_change_refuses_publication(native_source, monkeypatch):
    from fisheye.training import native_mask_tail_review as mod

    publish = mod.publish_review_payload

    def changed(*args, **kwargs):
        root = zarr.open_group(str(native_source), mode="a", use_consolidated=False)
        root["refined_keypoints_runs/pose/keypoints_roi"][0, 0, 0] += 1
        return publish(*args, **kwargs)

    monkeypatch.setattr(mod, "publish_review_payload", changed)
    with pytest.raises(ValueError, match="source identity.*changed"):
        produce(native_source, apply=True)
    assert not (native_source / "crop_runs/mask_tail_full_roi_v1").exists()


def test_native_review_refuses_wrong_recording_and_tampered_contract(native_source):
    result = produce(native_source, apply=True)
    root = zarr.open_group(str(native_source), mode="a", use_consolidated=False)
    kwargs = dict(
        refined_run=result["paths"]["pose_edit"].split("/")[1],
        crop_run=result["paths"]["crop"].split("/")[1],
        include_all=True,
    )
    root.attrs["recording_id"] = "wrong"
    with pytest.raises(ValueError, match="native crop-only review source binding"):
        resolve_review_session(str(native_source), **kwargs)
    root.attrs["recording_id"] = "native"
    root[result["paths"]["pose_edit"]].attrs["keypoint_labels"] = ["wrong"] * 19
    with pytest.raises(ValueError, match="crop-only review contract"):
        resolve_review_session(str(native_source), **kwargs)
