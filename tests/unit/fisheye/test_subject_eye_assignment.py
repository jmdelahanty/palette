from __future__ import annotations

import numpy as np
import pytest

from fisheye.refinement import subject_eye_assignment as assignment_mod

from fisheye.refinement.subject_eye_assignment import (
    _split_union_by_keypoints,
    _split_union_by_keypoints_batch_into,
    _split_union_by_keypoints_distance_batch_into,
    _split_union_by_keypoints_halfplane_batch_into,
    _split_union_by_keypoints_sparse_batch_into,
    assign_eyes_union_to_lr,
    reconcile_keypoint_mask_row_identity,
)


def test_reconcile_keypoint_mask_row_identity_accepts_matching_source_crop_rows() -> (
    None
):
    summary = reconcile_keypoint_mask_row_identity(
        keypoint_source_crop_row_ids=np.asarray([4, 8, 9], dtype=np.int64),
        mask_source_crop_row_ids=np.asarray([4, 8, 9], dtype=np.int64),
        expected_rows=3,
        keypoint_run_name="kp_001",
        keypoint_group_name="refined_keypoints_runs",
        mask_run_name="subject_001",
    )

    assert summary["row_identity_check"] == "source_crop_row_ids_match"
    assert summary["rows_checked"] == 3


def test_reconcile_keypoint_mask_row_identity_rejects_same_length_mismatch() -> None:
    with pytest.raises(
        ValueError, match="row identity mismatch.*row 1.*keypoint=7.*mask=6"
    ):
        reconcile_keypoint_mask_row_identity(
            keypoint_source_crop_row_ids=np.asarray([5, 7, 9], dtype=np.int64),
            mask_source_crop_row_ids=np.asarray([5, 6, 9], dtype=np.int64),
            expected_rows=3,
            keypoint_run_name="kp_001",
            keypoint_group_name="refined_keypoints_runs",
            mask_run_name="subject_001",
        )


def test_reconcile_keypoint_mask_row_identity_skips_when_legacy_ids_missing() -> None:
    summary = reconcile_keypoint_mask_row_identity(
        keypoint_source_crop_row_ids=None,
        mask_source_crop_row_ids=np.asarray([0, 1], dtype=np.int64),
        expected_rows=2,
        keypoint_run_name="legacy_kp",
        keypoint_group_name="keypoints_runs",
        mask_run_name="legacy_subject",
    )

    assert summary["row_identity_check"] == "skipped_missing_source_crop_row_ids"
    assert summary["keypoint_has_source_crop_row_ids"] is False
    assert summary["mask_has_source_crop_row_ids"] is True


def test_reconcile_keypoint_mask_row_identity_rejects_keypoint_row_count_mismatch() -> (
    None
):
    with pytest.raises(
        ValueError,
        match=r"keypoints_runs/kp_001: keypoint source_crop_row_ids has 2 rows, expected 3",
    ):
        reconcile_keypoint_mask_row_identity(
            keypoint_source_crop_row_ids=np.asarray([4, 8], dtype=np.int64),
            mask_source_crop_row_ids=np.asarray([4, 8, 9], dtype=np.int64),
            expected_rows=3,
            keypoint_run_name="kp_001",
            mask_run_name="subject_001",
        )


def test_reconcile_keypoint_mask_row_identity_rejects_mask_row_count_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match=r"mask source_crop_row_ids has 4 rows, expected 3",
    ):
        reconcile_keypoint_mask_row_identity(
            keypoint_source_crop_row_ids=np.asarray([4, 8, 9], dtype=np.int64),
            mask_source_crop_row_ids=np.asarray([4, 8, 9, 11], dtype=np.int64),
            expected_rows=3,
            keypoint_run_name="kp_001",
            mask_run_name="subject_001",
        )


def test_reconcile_keypoint_mask_row_identity_skips_when_mask_ids_missing() -> None:
    summary = reconcile_keypoint_mask_row_identity(
        keypoint_source_crop_row_ids=np.asarray([0, 1], dtype=np.int64),
        mask_source_crop_row_ids=None,
        expected_rows=2,
        keypoint_run_name="kp_001",
        mask_run_name="legacy_subject",
    )

    assert summary["row_identity_check"] == "skipped_missing_source_crop_row_ids"
    assert summary["keypoint_has_source_crop_row_ids"] is True
    assert summary["mask_has_source_crop_row_ids"] is False
    assert summary["expected_rows"] == 2


def test_reconcile_keypoint_mask_row_identity_skips_when_both_sides_missing() -> None:
    summary = reconcile_keypoint_mask_row_identity(
        keypoint_source_crop_row_ids=None,
        mask_source_crop_row_ids=None,
        expected_rows=5,
        keypoint_run_name="legacy_kp",
        mask_run_name="legacy_subject",
    )

    assert summary["row_identity_check"] == "skipped_missing_source_crop_row_ids"
    assert summary["keypoint_has_source_crop_row_ids"] is False
    assert summary["mask_has_source_crop_row_ids"] is False


def test_reconcile_keypoint_mask_row_identity_rejects_reordered_ids_without_reordering() -> (
    None
):
    # Same id set, different order: the check is strict positional equality.
    # The reconciler never reorders rows to make the sides match.
    with pytest.raises(
        ValueError, match=r"row identity mismatch.*row 0.*keypoint=4.*mask=9"
    ):
        reconcile_keypoint_mask_row_identity(
            keypoint_source_crop_row_ids=np.asarray([4, 8, 9], dtype=np.int64),
            mask_source_crop_row_ids=np.asarray([9, 8, 4], dtype=np.int64),
            expected_rows=3,
            keypoint_run_name="kp_001",
            mask_run_name="subject_001",
        )


def test_reconcile_keypoint_mask_row_identity_accepts_sliceable_zarr_like_sources() -> (
    None
):
    # Call sites pass zarr arrays; the reader consumes them via value[:].
    summary = reconcile_keypoint_mask_row_identity(
        keypoint_source_crop_row_ids=[4, 8, 9],
        mask_source_crop_row_ids=[4, 8, 9],
        expected_rows=3,
        keypoint_run_name="kp_001",
        mask_run_name="subject_001",
    )

    assert summary["row_identity_check"] == "source_crop_row_ids_match"
    assert summary["rows_checked"] == 3


def test_vectorized_keypoint_split_matches_row_reference() -> None:
    union = np.zeros((4, 18, 20), dtype=np.uint8)
    union[0, 4:9, 3:8] = 1
    union[0, 4:9, 12:17] = 1
    union[1, 2:16, 8:12] = 1
    union[2, 8, :] = 1
    union[3, :, 10] = 1

    eye_left = np.asarray(
        [
            [5.0, 6.0],
            [8.0, 8.0],
            [4.0, 8.0],
            [10.0, 4.0],
        ],
        dtype=np.float32,
    )
    eye_right = np.asarray(
        [
            [14.0, 6.0],
            [12.0, 8.0],
            [15.0, 8.0],
            [10.0, 14.0],
        ],
        dtype=np.float32,
    )

    left_out = np.zeros_like(union, dtype=np.uint8)
    right_out = np.zeros_like(union, dtype=np.uint8)

    _split_union_by_keypoints_batch_into(
        union,
        eye_left,
        eye_right,
        row_indices=np.arange(int(union.shape[0])),
        left_out=left_out,
        right_out=right_out,
        batch_size=2,
    )

    for row_idx in range(int(union.shape[0])):
        expected_left, expected_right = _split_union_by_keypoints(
            union[row_idx],
            eye_left[row_idx],
            eye_right[row_idx],
        )
        np.testing.assert_array_equal(left_out[row_idx].astype(bool), expected_left)
        np.testing.assert_array_equal(right_out[row_idx].astype(bool), expected_right)


def test_halfplane_split_matches_distance_batch_reference() -> None:
    rng = np.random.default_rng(123)
    union = (rng.random((12, 24, 26)) > 0.78).astype(np.uint8)
    union[1] = 0
    union[4, 8:16, 9:18] = 1

    # Integer-like positions avoid making this a floating-point roundoff test.
    eye_left = rng.integers(2, 12, size=(12, 2)).astype(np.float32)
    eye_right = rng.integers(13, 24, size=(12, 2)).astype(np.float32)
    row_indices = np.asarray([0, 1, 2, 4, 7, 9, 11], dtype=np.int64)

    distance_left = np.zeros_like(union, dtype=np.uint8)
    distance_right = np.zeros_like(union, dtype=np.uint8)
    halfplane_left = np.zeros_like(union, dtype=np.uint8)
    halfplane_right = np.zeros_like(union, dtype=np.uint8)

    _split_union_by_keypoints_distance_batch_into(
        union,
        eye_left,
        eye_right,
        row_indices=row_indices,
        left_out=distance_left,
        right_out=distance_right,
        batch_size=3,
    )
    _split_union_by_keypoints_halfplane_batch_into(
        union,
        eye_left,
        eye_right,
        row_indices=row_indices,
        left_out=halfplane_left,
        right_out=halfplane_right,
        batch_size=5,
    )

    np.testing.assert_array_equal(halfplane_left, distance_left)
    np.testing.assert_array_equal(halfplane_right, distance_right)


def test_sparse_split_matches_distance_batch_reference() -> None:
    rng = np.random.default_rng(456)
    union = (rng.random((16, 32, 34)) > 0.93).astype(np.uint8)
    union[2] = 0
    union[5, 4:20, 3:16] = 1
    union[7, 12:15, 10:22] = 1

    eye_left = rng.uniform(-5.0, 18.0, size=(16, 2)).astype(np.float32)
    eye_right = rng.uniform(12.0, 40.0, size=(16, 2)).astype(np.float32)
    row_indices = np.asarray([0, 2, 4, 5, 7, 8, 11, 15], dtype=np.int64)

    distance_left = np.zeros_like(union, dtype=np.uint8)
    distance_right = np.zeros_like(union, dtype=np.uint8)
    sparse_left = np.zeros_like(union, dtype=np.uint8)
    sparse_right = np.zeros_like(union, dtype=np.uint8)

    _split_union_by_keypoints_distance_batch_into(
        union,
        eye_left,
        eye_right,
        row_indices=row_indices,
        left_out=distance_left,
        right_out=distance_right,
        batch_size=4,
    )
    _split_union_by_keypoints_sparse_batch_into(
        union,
        eye_left,
        eye_right,
        row_indices=row_indices,
        left_out=sparse_left,
        right_out=sparse_right,
        batch_size=3,
    )

    np.testing.assert_array_equal(sparse_left, distance_left)
    np.testing.assert_array_equal(sparse_right, distance_right)


def test_halfplane_split_preserves_tie_goes_left_contract() -> None:
    union = np.zeros((1, 11, 11), dtype=np.uint8)
    union[0, 2:9, 5] = 1
    union[0, 5, 2:9] = 1
    eye_left = np.asarray([[3.0, 5.0]], dtype=np.float32)
    eye_right = np.asarray([[7.0, 5.0]], dtype=np.float32)

    left_out = np.zeros_like(union, dtype=np.uint8)
    right_out = np.zeros_like(union, dtype=np.uint8)
    _split_union_by_keypoints_halfplane_batch_into(
        union,
        eye_left,
        eye_right,
        row_indices=np.asarray([0], dtype=np.int64),
        left_out=left_out,
        right_out=right_out,
        batch_size=1,
    )

    assert bool(left_out[0, 5, 5])
    assert not bool(right_out[0, 5, 5])
    expected_left, expected_right = _split_union_by_keypoints(
        union[0], eye_left[0], eye_right[0]
    )
    np.testing.assert_array_equal(left_out[0].astype(bool), expected_left)
    np.testing.assert_array_equal(right_out[0].astype(bool), expected_right)


def test_halfplane_split_corrects_float32_boundary_roundoff() -> None:
    union = np.zeros((1, 512, 512), dtype=np.uint8)
    union[0, 267, 262] = 1
    eye_left = np.asarray([[272.0487365722656, 265.84039306640625]], dtype=np.float32)
    eye_right = np.asarray([[252.3084259033203, 269.89666748046875]], dtype=np.float32)

    distance_left = np.zeros_like(union, dtype=np.uint8)
    distance_right = np.zeros_like(union, dtype=np.uint8)
    halfplane_left = np.zeros_like(union, dtype=np.uint8)
    halfplane_right = np.zeros_like(union, dtype=np.uint8)

    _split_union_by_keypoints_distance_batch_into(
        union,
        eye_left,
        eye_right,
        row_indices=np.asarray([0], dtype=np.int64),
        left_out=distance_left,
        right_out=distance_right,
        batch_size=1,
    )
    _split_union_by_keypoints_halfplane_batch_into(
        union,
        eye_left,
        eye_right,
        row_indices=np.asarray([0], dtype=np.int64),
        left_out=halfplane_left,
        right_out=halfplane_right,
        batch_size=1,
    )

    np.testing.assert_array_equal(halfplane_left, distance_left)
    np.testing.assert_array_equal(halfplane_right, distance_right)


def test_sparse_split_preserves_float32_boundary_reference() -> None:
    union = np.zeros((1, 512, 512), dtype=np.uint8)
    union[0, 267, 262] = 1
    eye_left = np.asarray([[272.0487365722656, 265.84039306640625]], dtype=np.float32)
    eye_right = np.asarray([[252.3084259033203, 269.89666748046875]], dtype=np.float32)

    distance_left = np.zeros_like(union, dtype=np.uint8)
    distance_right = np.zeros_like(union, dtype=np.uint8)
    sparse_left = np.zeros_like(union, dtype=np.uint8)
    sparse_right = np.zeros_like(union, dtype=np.uint8)

    _split_union_by_keypoints_distance_batch_into(
        union,
        eye_left,
        eye_right,
        row_indices=np.asarray([0], dtype=np.int64),
        left_out=distance_left,
        right_out=distance_right,
        batch_size=1,
    )
    _split_union_by_keypoints_sparse_batch_into(
        union,
        eye_left,
        eye_right,
        row_indices=np.asarray([0], dtype=np.int64),
        left_out=sparse_left,
        right_out=sparse_right,
        batch_size=1,
    )

    np.testing.assert_array_equal(sparse_left, distance_left)
    np.testing.assert_array_equal(sparse_right, distance_right)


def test_assign_eyes_union_records_subphase_timings() -> None:
    union = np.zeros((4, 32, 32), dtype=np.uint8)
    union[0, 8:16, 7:15] = 1
    union[0, 8:16, 18:26] = 1
    union[2, 8:16, 7:15] = 1
    union[2, 8:16, 18:26] = 1
    union[3, 8:16, 7:15] = 1
    union[3, 8:16, 18:26] = 1

    keypoints = np.zeros((4, 5, 2), dtype=np.float32)
    keypoints[:, 0, :] = np.asarray([11.0, 12.0], dtype=np.float32)
    keypoints[:, 1, :] = np.asarray([22.0, 12.0], dtype=np.float32)
    keypoints[3, 1, :] = keypoints[3, 0, :]
    success = np.asarray([True, True, False, True], dtype=bool)

    result = assign_eyes_union_to_lr(
        union,
        keypoints_roi=keypoints,
        keypoint_success=success,
        eye_keypoint_indices=(0, 1),
        split_batch_size=2,
    )

    assert set(result.phase_seconds) >= {
        "split_by_keypoint",
        "select_components",
        "measure_ellipse",
        "reason_labels",
    }
    assert result.summary["status_counts"]["assigned"] == 1
    assert result.summary["status_counts"]["failed_empty_union"] == 1
    assert result.summary["status_counts"]["failed_keypoint_status"] == 1
    assert result.summary["status_counts"]["failed_coincident_eye_keypoints"] == 1
    assert np.count_nonzero(result.masks["eye_left"][0]) > 0
    assert np.count_nonzero(result.masks["eye_right"][0]) > 0
    assert np.asarray(result.eye_geometry["ellipse_params"]).shape == (4, 2, 5)
    assert np.asarray(result.eye_geometry["ellipse_success"]).shape == (4, 2)
    assert np.asarray(result.eye_geometry["ellipse_success"], dtype=bool)[0].all()
    assert set(result.eye_geometry["contours"]) == {"eye_left", "eye_right"}


def test_assign_eyes_union_can_skip_ellipse_measurement_for_diagnostics() -> None:
    union = np.zeros((2, 32, 32), dtype=np.uint8)
    union[0, 8:16, 7:15] = 1
    union[0, 8:16, 18:26] = 1
    union[1, 10:18, 7:15] = 1
    union[1, 10:18, 18:26] = 1

    keypoints = np.zeros((2, 5, 2), dtype=np.float32)
    keypoints[:, 0, :] = np.asarray([11.0, 12.0], dtype=np.float32)
    keypoints[:, 1, :] = np.asarray([22.0, 12.0], dtype=np.float32)
    success = np.ones((2,), dtype=bool)

    measured = assign_eyes_union_to_lr(
        union,
        keypoints_roi=keypoints,
        keypoint_success=success,
        eye_keypoint_indices=(0, 1),
        split_batch_size=2,
    )
    skipped = assign_eyes_union_to_lr(
        union,
        keypoints_roi=keypoints,
        keypoint_success=success,
        eye_keypoint_indices=(0, 1),
        split_batch_size=2,
        measure_ellipses=False,
    )

    assert measured.summary["ellipse_measurement_requested"] is True
    assert skipped.summary["ellipse_measurement_requested"] is False
    assert "measure_ellipse" in measured.phase_seconds
    assert "measure_ellipse" not in skipped.phase_seconds
    for component in ("eye_left", "eye_right"):
        np.testing.assert_array_equal(
            skipped.masks[component], measured.masks[component]
        )
        np.testing.assert_array_equal(
            skipped.reason_labels[component], measured.reason_labels[component]
        )
    np.testing.assert_array_equal(skipped.assignment_status, measured.assignment_status)
    assert not np.asarray(skipped.eye_geometry["ellipse_success"], dtype=bool).any()
    assert np.isnan(
        np.asarray(skipped.eye_geometry["ellipse_params"], dtype=np.float32)
    ).all()
    assert all(item is None for item in skipped.eye_geometry["contours"]["eye_left"])
    assert all(item is None for item in skipped.eye_geometry["contours"]["eye_right"])


def test_assign_eyes_union_rejects_below_support_eye_before_ellipse_measurement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    union = np.zeros((1, 40, 40), dtype=np.uint8)
    union[0, 8:11, 7:10] = 1
    union[0, 8:16, 24:32] = 1
    keypoints = np.zeros((1, 2, 2), dtype=np.float32)
    keypoints[0, 0] = np.asarray([8.0, 9.0], dtype=np.float32)
    keypoints[0, 1] = np.asarray([28.0, 12.0], dtype=np.float32)
    original_measure = assignment_mod._measure_mask
    measured_areas: list[int] = []

    def guarded_measure(mask: np.ndarray):
        area = int(np.count_nonzero(mask))
        measured_areas.append(area)
        assert area >= 20
        return original_measure(mask)

    monkeypatch.setattr(assignment_mod, "_measure_mask", guarded_measure)

    result = assign_eyes_union_to_lr(
        union,
        keypoints_roi=keypoints,
        keypoint_success=np.asarray([True], dtype=bool),
        eye_keypoint_indices=(0, 1),
        minimum_component_area_px_by_name={"eye_left": 20, "eye_right": 20},
    )

    assert measured_areas == [64]
    assert int(result.masks["eye_left"].sum()) == 0
    assert int(result.masks["eye_right"].sum()) == 64
    assert result.assignment_status.tolist() == ["assigned_needs_review"]
    assert "needs_review_below_model_supported_area" in str(
        result.reason_labels["eye_left"][0]
    )
    assert "needs_review_below_model_supported_area" not in str(
        result.reason_labels["eye_right"][0]
    )
    assert result.summary["minimum_component_area_px_by_name"] == {
        "eye_left": 20,
        "eye_right": 20,
    }
    assert result.summary["below_model_supported_area_rows_by_component"] == {
        "eye_left": 1,
        "eye_right": 0,
    }


def test_component_fast_path_assignment_matches_standard_assignment() -> None:
    union = np.zeros((4, 48, 48), dtype=np.uint8)
    union[0, 10:20, 7:17] = 1
    union[0, 10:20, 30:40] = 1
    union[1, 8:30, 21:27] = 1  # crosses the left/right split plane; must fall back.
    union[2, 8:15, 5:12] = 1
    union[2, 30:38, 8:16] = 1
    union[2, 18:28, 32:42] = 1
    union[3, 14:24, 6:16] = 1
    union[3, 14:24, 33:43] = 1

    keypoints = np.zeros((4, 3, 2), dtype=np.float32)
    keypoints[:, 0, :] = np.asarray([10.0, 16.0], dtype=np.float32)
    keypoints[:, 1, :] = np.asarray([36.0, 16.0], dtype=np.float32)
    keypoints[2, 0, :] = np.asarray([11.0, 34.0], dtype=np.float32)
    keypoints[2, 1, :] = np.asarray([37.0, 23.0], dtype=np.float32)
    success = np.ones((4,), dtype=bool)

    standard = assign_eyes_union_to_lr(
        union,
        keypoints_roi=keypoints,
        keypoint_success=success,
        eye_keypoint_indices=(0, 1),
        split_batch_size=2,
    )
    fast = assign_eyes_union_to_lr(
        union,
        keypoints_roi=keypoints,
        keypoint_success=success,
        eye_keypoint_indices=(0, 1),
        split_batch_size=2,
        use_component_fast_path=True,
    )

    assert fast.summary["component_fast_path_requested"] is True
    assert fast.summary["component_fast_path_rows"] >= 1
    assert fast.summary["distance_split_rows"] >= 1
    for component in ("eye_left", "eye_right"):
        np.testing.assert_array_equal(fast.masks[component], standard.masks[component])
        np.testing.assert_array_equal(
            fast.reason_labels[component], standard.reason_labels[component]
        )
    np.testing.assert_array_equal(fast.assignment_status, standard.assignment_status)
    np.testing.assert_allclose(
        np.asarray(fast.eye_geometry["ellipse_params"], dtype=np.float32),
        np.asarray(standard.eye_geometry["ellipse_params"], dtype=np.float32),
        equal_nan=True,
    )
    np.testing.assert_array_equal(
        np.asarray(fast.eye_geometry["ellipse_success"], dtype=bool),
        np.asarray(standard.eye_geometry["ellipse_success"], dtype=bool),
    )
