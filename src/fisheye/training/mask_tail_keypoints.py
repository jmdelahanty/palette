"""Named training-label recipe using the maintained subject-shape geometry."""

from __future__ import annotations

import numpy as np

from fisheye.analysis.subject_shape_runs import (
    BodyFrameBatch,
    CENTERLINE_SAMPLE_COUNT,
    _compute_caudal_anchor_batch,
    _compute_centerline_batch,
    _compute_snout_tip_batch,
    _decode_reason_rows,
)
from fisheye.analysis.subject_shape_spline import (
    _resample_polyline,
    fit_subject_body_spline_batch,
    sample_spline_segment_by_arclength,
)
from fisheye.shared.subject_shape_coordinate_publication import (
    derive_canonical_subject_shape_body_frame,
)

SCHEMA_NAME = "head_tail11_fins_v1"
RECIPE = {
    "id": "recovered_head_oriented_subject_shape_tail11_arclength_v1",
    "orientation": "existing_head3_camera_y_down_axes_in_roi",
    "centerline": "subject_shape_snout_extended_longest_skeleton_endpoint_path",
    "tail_base": "caudal_swim_contour_projection_centerline_arclength_fraction",
    "spline_degree": 3,
    "spline_smoothing": 0.0,
    "centerline_sample_count": CENTERLINE_SAMPLE_COUNT,
    "sampling": "integrated_spline_speed_inverse_interpolation",
    "integration_samples": 4097,
    "tail_sample_s": np.linspace(0, 1, 11).tolist(),
    "mask_cleanup": "none",
    "outside_body_policy": "fail_row",
}
ORIGIN_CODES = {"missing": 0, "recovered_head": 1, "mask_derived": 2, "manual": 3}


def derive_tail_seed(
    masks: np.ndarray, mask_labels: tuple[str, ...], head_points: np.ndarray
) -> dict[str, np.ndarray]:
    """Keep every row, preserve head coordinates, leave fins/failures as NaNs."""
    masks, head = np.asarray(masks), np.asarray(head_points)
    if (
        masks.ndim != 4
        or masks.dtype != np.uint8
        or head.shape != (len(masks), 3, 2)
        or head.dtype.kind != "f"
        or len(mask_labels) != masks.shape[1]
        or len(set(mask_labels)) != len(mask_labels)
    ):
        raise ValueError("Expected uint8 (N,C,H,W) masks and float (N,3,2) head points")
    if not {"subject_body", "swim_bladder"}.issubset(mask_labels):
        raise ValueError("Body and swim-bladder mask channels are required")
    body, swim = (
        masks[:, mask_labels.index(name)] for name in ("subject_body", "swim_bladder")
    )
    frame = BodyFrameBatch(
        **derive_canonical_subject_shape_body_frame(
            ("swim_bladder", "eye_left", "eye_right"),
            head,
            np.isfinite(head).all(axis=2),
        )
    )
    snout = _compute_snout_tip_batch(body, frame, source_body_qc=None)
    anchor = _compute_caudal_anchor_batch(swim, frame)
    center = _compute_centerline_batch(
        body, frame, anchor, snout_tip=snout, crop_to_foreground=True
    )
    spline = fit_subject_body_spline_batch(
        center.centerline_xy,
        center.centerline_valid,
        center.tail_base_valid,
        center.tail_base_arclength_px,
        centerline_failure_reasons=_decode_reason_rows(
            center.centerline_failure_reason_bytes
        ),
        tail_base_failure_reasons=_decode_reason_rows(
            center.tail_base_failure_reason_bytes
        ),
        centerline_sample_count=CENTERLINE_SAMPLE_COUNT,
        tail_sample_count=11,
    )
    n, _, h, w = masks.shape
    points = np.full((n, 18, 2), np.nan, dtype=head.dtype)
    points[:, :3] = head
    origins = np.zeros((n, 18), dtype=np.uint8)
    origins[:, :3] = np.where(np.isfinite(head).all(axis=2), 1, 0)
    valid = np.zeros(n, dtype=bool)
    reasons = np.asarray(spline.tail_sample_failure_reasons, dtype=object).copy()
    length = np.full(n, np.nan, dtype=np.float64)
    parameters = np.full((n, 11), np.nan, dtype=np.float64)
    for row in range(n):
        if not spline.tail_sample_valid[row]:
            continue
        if np.any(body[row, (0, -1), :]) or np.any(body[row, :, (0, -1)]):
            reasons[row] = "body_touches_crop_border"
            continue
        _, total = _resample_polyline(
            center.centerline_xy[row], CENTERLINE_SAMPLE_COUNT
        )
        start = float(center.tail_base_arclength_px[row]) / total
        degree = int(spline.bspline_degree_used[row])
        knots = spline.bspline_knots[row]
        knots = knots[np.isfinite(knots)]
        control = spline.bspline_control_points_xy[row, : len(knots) - degree - 1]
        try:
            xy, u, arc = sample_spline_segment_by_arclength(
                (knots, control.T, degree),
                start_u=start,
                integration_samples=RECIPE["integration_samples"],
            )
        except ValueError as exc:
            reasons[row] = f"tail_sampling_failed:{exc}"
            continue
        pixel = np.rint(xy).astype(int)
        if (
            np.any(pixel < 0)
            or np.any(pixel[:, 0] >= w)
            or np.any(pixel[:, 1] >= h)
            or not np.all(body[row, pixel[:, 1], pixel[:, 0]] > 0)
        ):
            reasons[row] = "tail_station_outside_body"
            continue
        points[row, 3:14] = xy
        origins[row, 3:14] = ORIGIN_CODES["mask_derived"]
        valid[row], reasons[row], length[row], parameters[row] = True, "ok", arc, u
    return {
        "keypoints_roi": points,
        "keypoint_origin": origins,
        "tail_valid": valid,
        "tail_failure_reason": reasons,
        "tail_arc_length_px": length,
        "tail_spline_u": parameters,
        "tail_base_polyline_xy": center.tail_base_xy,
        "swim_caudal_anchor_xy": anchor.point_xy,
        "training_eligible": np.zeros(n, dtype=bool),
    }
