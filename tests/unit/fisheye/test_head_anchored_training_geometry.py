"""Frozen training mask examples for the opt-in head-anchored tail method."""

from pathlib import Path

import numpy as np

from fisheye.analysis.subject_shape_runs import HEAD_ANCHORED_CENTERLINE_METHOD
from fisheye.training.mask_tail_keypoints import derive_tail_seed


FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "redscare_head_anchored_tail_rows_v1.npz"
GOLDEN = Path(__file__).resolve().parents[2] / "fixtures" / "redscare_head_anchored_legacy_golden_v1.npz"


def test_legacy_seed_stays_identical_to_frozen_audited_outputs():
    with np.load(FIXTURE) as frozen:
        masks, labels, head = frozen["masks"], tuple(frozen["labels"]), frozen["head"]
    default = derive_tail_seed(masks, labels, head)
    with np.load(GOLDEN) as frozen_outputs:
        assert set(default) == set(frozen_outputs.files)
        for name, values in default.items():
            actual = np.asarray(values, dtype=str) if np.asarray(values).dtype == object else values
            np.testing.assert_array_equal(actual, frozen_outputs[name], err_msg=name)
    assert default["tail_valid"].tolist() == [False] * 8 + [True, True]
    assert default["tail_failure_reason"].tolist() == ["snout_extension_too_long"] * 8 + ["ok", "ok"]
    np.testing.assert_allclose(default["tail_arc_length_px"][8:], [135.33733688, 146.63149176], atol=1e-5)


def test_head_anchored_path_recovers_diagnosed_curl_join_and_fin_cases():
    with np.load(FIXTURE) as frozen:
        masks, labels, head = frozen["masks"], tuple(frozen["labels"]), frozen["head"]
        rows = frozen["rows"]
    derived = derive_tail_seed(masks, labels, head, method=HEAD_ANCHORED_CENTERLINE_METHOD)
    for roi_idx in (16, 72, 113, 121, 171, 187):
        row = int(np.flatnonzero(rows == roi_idx)[0])
        assert bool(derived["tail_valid"][row]), (roi_idx, derived["tail_failure_reason"][row])
        assert np.isfinite(derived["tail_arc_length_px"][row])
        stations = derived["keypoints_roi"][row, 3:14]
        assert np.isfinite(stations).all()
        pixels = np.rint(stations).astype(int)
        body = masks[row, labels.index("subject_body")]
        assert np.all(body[pixels[:, 1], pixels[:, 0]] > 0)
        assert np.linalg.norm(stations[0] - derived["tail_base_polyline_xy"][row]) < 5
        assert np.linalg.norm(stations[-1] - head[row, 1:3].mean(axis=0)) > 40


def test_ambiguous_head_endpoints_are_refused():
    from fisheye.analysis.subject_shape_runs import _head_anchored_skeleton_path_xy

    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[28:36, 12:52] = 1
    path, reason = _head_anchored_skeleton_path_xy(
        mask, np.array([32., 31.]), np.array([32., 31.]),
    )
    assert path is None
    assert reason == "ambiguous_head_endpoint"
