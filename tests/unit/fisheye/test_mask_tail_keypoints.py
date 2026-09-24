"""Scientific preservation checks for recovered head + mask tail + manual fins."""

import numpy as np
import pytest
from scipy import integrate, interpolate

from fisheye.shared.pose_schema import schema_from_package
from fisheye.analysis.subject_shape_spline import sample_spline_segment_by_arclength
from fisheye.training.mask_tail_keypoints import derive_tail_seed, recipe_for_schema, recipe_with_visible_endpoint
from fisheye.shared.detect_reason_codec import decode_reason_bytes


def test_schema_preserves_head_indices_and_has_eleven_tail_stations():
    schema = schema_from_package("head_tail11_fins_v1")
    assert schema.node_names[:3] == schema_from_package("traditional_v1").node_names
    assert schema.node_names[3:14] == [
        "tail_base",
        *[f"tail_point_{i:02d}" for i in range(1, 10)],
        "tail_tip",
    ]
    assert (
        schema.node_names[14:] == schema_from_package("traditional_v3").node_names[6:]
    )
    assert schema.num_keypoints == 18


def test_snout_schema_extends_existing_indices_without_reinterpreting_v1():
    legacy = schema_from_package("head_tail11_fins_v1")
    current = schema_from_package("head_tail11_fins_v2")
    assert current.node_names[:-1] == legacy.node_names
    assert current.node_names[-1] == "snout_tip"
    assert current.num_keypoints == 19
    assert [18, 1] in current.edges and [18, 2] in current.edges


def test_spline_stations_are_arclength_spaced_not_parameter_spaced():
    u = np.linspace(0, 1, 32)
    tck, _ = interpolate.splprep([100 * u, 90 * u**3], u=u, s=0, k=3)
    xy, stations, length = sample_spline_segment_by_arclength(tck, start_u=0.2)
    arc = [
        integrate.quad(
            lambda v: np.linalg.norm(interpolate.splev(v, tck, der=1)),
            a,
            b,
            epsabs=1e-6,
        )[0]
        for a, b in zip(stations[:-1], stations[1:])
    ]
    np.testing.assert_allclose(arc, length / 10, atol=2e-4)
    np.testing.assert_allclose(
        xy[[0, -1]], np.array(interpolate.splev([0.2, 1], tck)).T
    )
    assert np.ptp(np.diff(stations)) > 0.02


@pytest.mark.parametrize("start", [np.nan, -0.1, 1, 1.1])
def test_spline_sampling_refuses_invalid_segment(start):
    tck, _ = interpolate.splprep([[0, 1, 2, 3], [0, 0, 0, 0]], s=0)
    with pytest.raises(ValueError):
        sample_spline_segment_by_arclength(tck, start_u=start)


def _fish():
    yy, xx = np.mgrid[:128, :128]
    body = ((xx - 64) / 13) ** 2 + ((yy - 60) / 48) ** 2 <= 1
    bladder = ((xx - 64) / 7) ** 2 + ((yy - 43) / 9) ** 2 <= 1
    eyes = (((xx - 58) ** 2 + (yy - 26) ** 2) <= 9) | (
        ((xx - 70) ** 2 + (yy - 26) ** 2) <= 9
    )
    masks = np.stack([body, eyes, bladder]).astype(np.uint8)[None]
    head = np.array([[[64, 43], [58, 26], [70, 26]]], dtype=np.float32)
    return masks, head


def test_seed_keeps_head_exact_and_fins_missing_and_reports_fragmentation():
    masks, head = _fish()
    before = masks.copy()
    result = derive_tail_seed(
        masks, ("subject_body", "eyes_union", "swim_bladder"), head
    )
    assert result["tail_valid"].tolist() == [True]
    np.testing.assert_array_equal(result["keypoints_roi"][:, :3], head)
    assert np.isnan(result["keypoints_roi"][:, 14:18]).all()
    # The contour is at the pixel edge, half a pixel before the first body row.
    np.testing.assert_allclose(result["keypoints_roi"][0, 18], [64, 11.5])
    assert result["snout_valid"].tolist() == [True]
    assert decode_reason_bytes(result["snout_failure_reason_bytes"]).tolist() == ["ok"]
    assert not result["training_eligible"].any()
    assert result["keypoint_origin"][0].tolist() == [1] * 3 + [2] * 11 + [0] * 4 + [2]
    legacy = derive_tail_seed(
        masks,
        ("subject_body", "eyes_union", "swim_bladder"),
        head,
        schema_name="head_tail11_fins_v1",
    )
    np.testing.assert_array_equal(
        result["keypoints_roi"][:, :18], legacy["keypoints_roi"]
    )
    np.testing.assert_array_equal(
        result["keypoint_origin"][:, :18], legacy["keypoint_origin"]
    )
    assert "snout_valid" not in legacy
    np.testing.assert_array_equal(masks, before)
    masks[0, 0, 1, 1] = 1
    failed = derive_tail_seed(
        masks, ("subject_body", "eyes_union", "swim_bladder"), head
    )
    assert not failed["tail_valid"][0]
    assert not failed["snout_valid"][0]
    assert (
        decode_reason_bytes(failed["snout_failure_reason_bytes"])[0]
        == "fragmented_subject_body_mask"
    )
    assert failed["tail_failure_reason"][0] == "fragmented_subject_body_mask"
    assert np.isnan(failed["keypoints_roi"][0, 3:]).all()
    np.testing.assert_array_equal(failed["keypoints_roi"][:, :3], head)


def test_valid_snout_is_retained_when_tail_cannot_be_anchored():
    masks, head = _fish()
    masks[:, 2] = 0
    result = derive_tail_seed(
        masks, ("subject_body", "eyes_union", "swim_bladder"), head
    )
    assert result["snout_valid"][0]
    assert not result["tail_valid"][0]
    assert np.isnan(result["keypoints_roi"][0, 3:14]).all()
    assert np.isfinite(result["keypoints_roi"][0, 18]).all()


def test_explicit_crop_edge_acceptance_keeps_visible_endpoint_and_other_checks():
    masks, head = _fish()
    clipped = masks[:, :, :108, :].copy()
    strict = derive_tail_seed(clipped, ("subject_body", "eyes_union", "swim_bladder"), head)
    assert strict["tail_failure_reason"].tolist() == ["body_touches_crop_border"]
    assert "tail_tip_truncated" not in strict
    accepted = derive_tail_seed(
        clipped, ("subject_body", "eyes_union", "swim_bladder"), head,
        accepted_crop_border_rows=np.array([True]),
    )
    assert accepted["tail_valid"].tolist() == [True]
    assert accepted["tail_tip_truncated"].tolist() == [True]
    assert accepted["tail_visible_endpoint_accepted"].tolist() == [True]
    np.testing.assert_array_equal(accepted["keypoints_roi"][:, :3], head)
    assert accepted["keypoint_origin"][0, 13] == 2
    assert accepted["keypoints_roi"][0, 13, 1] < 108
    assert recipe_for_schema("head_tail11_fins_v2")["id"].endswith("v2")
    assert recipe_with_visible_endpoint("head_tail11_fins_v2")["id"].endswith("v3")
    fragmented = clipped.copy()
    fragmented[0, 0, 1, 1] = 1
    refused = derive_tail_seed(
        fragmented, ("subject_body", "eyes_union", "swim_bladder"), head,
        accepted_crop_border_rows=np.array([True]),
    )
    assert not refused["tail_valid"][0]
    assert refused["tail_failure_reason"][0] == "fragmented_subject_body_mask"
