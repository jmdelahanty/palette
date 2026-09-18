"""Scientific preservation checks for recovered head + mask tail + manual fins."""

import numpy as np
import pytest
from scipy import integrate, interpolate

from fisheye.shared.pose_schema import schema_from_package
from fisheye.analysis.subject_shape_spline import sample_spline_segment_by_arclength
from fisheye.training.mask_tail_keypoints import derive_tail_seed


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
    assert np.isnan(result["keypoints_roi"][:, 14:]).all()
    assert not result["training_eligible"].any()
    assert result["keypoint_origin"][0].tolist() == [1] * 3 + [2] * 11 + [0] * 4
    np.testing.assert_array_equal(masks, before)
    masks[0, 0, 1, 1] = 1
    failed = derive_tail_seed(
        masks, ("subject_body", "eyes_union", "swim_bladder"), head
    )
    assert not failed["tail_valid"][0]
    assert failed["tail_failure_reason"][0] == "fragmented_subject_body_mask"
    assert np.isnan(failed["keypoints_roi"][0, 3:]).all()
    np.testing.assert_array_equal(failed["keypoints_roi"][:, :3], head)
