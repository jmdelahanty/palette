"""Crop-only compatibility for recovered training labels with missing sensor origins.

This is a training annotation surface, never a camera-coordinate publication.
The immutable seed retains the original per-landmark origin after manual edits.
"""

import numpy as np

from fisheye.shared.keypoint_motion_authority import (
    keypoint_source_crop_run_from_attributes,
)

from fisheye.shared.recovered_training_review_contract import (
    COORDINATE_SYSTEM,
    REVIEW_SCHEMA,
    initial_contract_digest,
)


def is_recovered_roi_review(root, refined, crop):
    if refined.attrs.get("schema_id") != REVIEW_SCHEMA:
        return False
    if (
        root.attrs.get("zarr_purpose") != "training"
        or root.attrs.get("schema_id")
        != "palette.training.merged_pose_detect_recovery_source.v1"
        or any(
            g.attrs.get("stage_selector_eligible") is not False
            for g in (root, crop, refined)
        )
        or crop.attrs.get("schema_id") != REVIEW_SCHEMA
        or any(
            g.attrs.get("coordinate_system") != COORDINATE_SYSTEM
            for g in (crop, refined)
        )
        or crop.attrs.get("sensor_pixel_origin_available") is not False
        or crop.attrs.get("frame_index_domain") != "legacy_training_sample_row"
        or refined.attrs.get("source_bindings") != crop.attrs.get("source_bindings")
        or keypoint_source_crop_run_from_attributes(refined.attrs)
        != str(crop.path).split("/")[-1]
        or any(
            initial_contract_digest(g) != g.attrs.get("initial_contract_sha256")
            for g in (crop, refined)
        )
        or any(name in refined for name in ("keypoints_img", "keypoints_norm"))
        or "roi_coordinates_full" in crop
        or not np.array_equal(crop["frame_indices"][:], refined["frame_indices"][:])
    ):
        raise ValueError("Invalid recovered crop-only review contract")
    n, k, dims = refined["keypoints_roi"].shape
    if dims != 2 or n != crop["frame_indices"].shape[0]:
        raise ValueError("Recovered keypoint/crop row axes disagree")
    for name, shape in (
        ("keypoint_origin", (n, k)),
        ("keypoint_manual_edit", (n, k)),
        ("training_eligible", (n,)),
    ):
        if name not in refined or refined[name].shape != shape:
            raise ValueError(f"Recovered review lacks aligned {name}")
    return True


def validate_points_inside_crop(points, image_shape):
    height, width = image_shape[:2]
    if not (
        np.isfinite(points).all()
        and np.all(points >= 0)
        and np.all(points[:, 0] < width)
        and np.all(points[:, 1] < height)
    ):
        raise ValueError("Every training keypoint must be visible inside the crop")


def record_point_edits(refined, row, old_points, new_points):
    """Only changed landmarks acquire manual origin; untouched seeds keep lineage."""
    dtype = refined["keypoints_roi"].dtype
    new_points = np.asarray(new_points, dtype=dtype)
    changed = ~np.all(
        (old_points == new_points) | (np.isnan(old_points) & np.isnan(new_points)),
        axis=1,
    )
    origins = np.asarray(refined["keypoint_origin"][row]).copy()
    edited = np.asarray(refined["keypoint_manual_edit"][row]).copy()
    origins[changed] = 3
    edited[changed] = True
    refined["keypoint_origin"][row] = origins
    refined["keypoint_manual_edit"][row] = edited
