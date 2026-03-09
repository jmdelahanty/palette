from __future__ import annotations

import numpy as np
import pytest
from zarr.core.dtype import VariableLengthUTF8

from fisheye.refinement import refine_eye_masks as mod


class _FakeGroup(dict):
    def create_group(self, name: str) -> "_FakeGroup":
        group = _FakeGroup()
        self[name] = group
        return group

    def create_array(
        self,
        name: str,
        *,
        data=None,
        shape=None,
        dtype=None,
        fill_value=0,
        **_kwargs,
    ):
        if data is None:
            arr_dtype = object if isinstance(dtype, VariableLengthUTF8) else dtype
            arr = np.full(shape, fill_value, dtype=arr_dtype)
        else:
            if isinstance(dtype, VariableLengthUTF8):
                arr = np.asarray(data, dtype=object)
            else:
                arr = np.asarray(data, dtype=dtype)
        self[name] = arr
        return arr


def _create_metrics_group(run_group: _FakeGroup, *, total_rois: int, chunk_rois: int) -> _FakeGroup:
    metrics = run_group.create_group("metrics")

    for name in (
        "area_refined",
        "area_source",
        "area_zscore",
        "area_delta_vs_source",
        "area_ratio_vs_source",
        "centroid_error",
        "symmetry_offsets",
        "axis_ratio",
        "circularity",
        "probability_mean",
        "probability_max",
        "probability_var",
        "probability_high_fraction",
    ):
        metrics.create_array(
            name,
            shape=(total_rois, 2),
            chunks=(chunk_rois, 2),
            dtype=np.float32,
            fill_value=np.float32(np.nan),
        )

    for name in (
        "area_union_refined",
        "area_union_source",
        "area_ratio_left_right",
        "area_diff_left_right",
        "area_union_delta",
        "area_union_ratio",
        "symmetry_sum",
        "symmetry_abs_diff",
        "separation_refined",
        "separation_keypoint",
        "separation_delta",
    ):
        metrics.create_array(
            name,
            shape=(total_rois,),
            chunks=(chunk_rois,),
            dtype=np.float32,
            fill_value=np.float32(np.nan),
        )

    metrics.create_array(
        "connectivity_flags",
        shape=(total_rois,),
        chunks=(chunk_rois,),
        dtype=np.uint8,
        fill_value=0,
    )
    metrics.create_array(
        "smoothing_flags",
        shape=(total_rois, 2),
        chunks=(chunk_rois, 2),
        dtype=np.uint8,
        fill_value=0,
    )
    metrics.create_array(
        "pixels_reassigned",
        shape=(total_rois,),
        chunks=(chunk_rois,),
        dtype=np.int32,
        fill_value=0,
    )
    metrics.create_array(
        "probabilities_used",
        shape=(total_rois,),
        chunks=(chunk_rois,),
        dtype=bool,
        fill_value=False,
    )
    metrics.create_array(
        "filter_flags",
        shape=(total_rois, 2),
        chunks=(chunk_rois, 2),
        dtype=bool,
        fill_value=False,
    )
    metrics.create_array(
        mod._LOCAL_REASON_STAGING_DATASET,
        shape=(total_rois,),
        chunks=(chunk_rois,),
        dtype=VariableLengthUTF8(),
        fill_value="",
    )
    return metrics


def test_write_chunk_metrics_stages_reason_without_return_payload() -> None:
    run_group = _FakeGroup()
    metrics = _create_metrics_group(run_group, total_rois=1, chunk_rois=1)

    roi_output = mod.ROIOutput(
        masks=np.ones((2, 4, 4), dtype=np.uint8),
        ellipse_params=np.zeros((2, 5), dtype=np.float32),
        ellipse_success=np.array([True, True], dtype=bool),
        centroids=np.zeros((2, 2), dtype=np.float32),
        contours=(None, None),
        eye_separation=8.0,
        reason="union_source",
        smoothing_changed=np.array([True, False], dtype=bool),
        reassigned_pixels=3,
        used_probabilities=True,
        probabilities=np.ones((2, 4, 4), dtype=np.float32),
        refined_areas=np.array([10.0, 60.0], dtype=np.float32),
        source_areas=np.array([12.0, 55.0], dtype=np.float32),
        source_union_area=70.0,
        refined_union_area=72.0,
        centroid_errors=np.array([1.0, 2.0], dtype=np.float32),
        symmetry_offsets=np.array([1.0, -1.0], dtype=np.float32),
        keypoint_separation=7.5,
        separation_delta=0.5,
        axis_ratio=np.array([2.0, 3.0], dtype=np.float32),
        circularity=np.array([0.4, 0.5], dtype=np.float32),
        probability_mean=np.array([0.8, 0.9], dtype=np.float32),
        probability_max=np.array([1.0, 1.0], dtype=np.float32),
        probability_var=np.array([0.1, 0.2], dtype=np.float32),
        probability_high_fraction=np.array([0.6, 0.7], dtype=np.float32),
    )

    result = mod._write_chunk_metrics(
        run_group,
        start=0,
        stop=1,
        results=[(0, roi_output)],
        success_min_eye_area_px=50.0,
    )

    assert result is None
    assert metrics[mod._LOCAL_REASON_STAGING_DATASET][0] == "union_source|small_area_left|small_area_pair"
    assert float(np.asarray(metrics["area_ratio_left_right"][0])) == np.float32(10.0 / 60.0)
    assert int(np.asarray(metrics["connectivity_flags"][0])) == 13
    assert bool(np.asarray(metrics["probabilities_used"][0]))


def test_load_source_mask_slices_reads_masks_and_probabilities(monkeypatch: pytest.MonkeyPatch) -> None:
    store = {
        "masks": np.array([[[[0, 1], [1, 0]]]], dtype=np.uint8),
        "probs": np.array([[[[0.25, 0.75], [0.9, 0.1]]]], dtype=np.float32),
    }
    monkeypatch.setattr(mod, "_get_zarr_array", lambda _zarr_path, path: store[path])

    masks, probs = mod._load_source_mask_slices(
        "demo.zarr",
        start=0,
        stop=1,
        source_masks_path="masks",
        source_probs_path="probs",
        probability_threshold=0.5,
    )

    assert masks.shape == (1, 1, 2, 2)
    assert probs is not None
    assert probs.shape == (1, 1, 2, 2)
    assert masks.dtype == np.uint8
    assert probs.dtype == np.float32


def test_load_source_mask_slices_synthesizes_binary_from_probabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = {
        "probs": np.array([[[[0.25, 0.75], [0.9, 0.1]]]], dtype=np.float32),
    }
    monkeypatch.setattr(mod, "_get_zarr_array", lambda _zarr_path, path: store[path])

    masks, probs = mod._load_source_mask_slices(
        "demo.zarr",
        start=0,
        stop=1,
        source_masks_path=None,
        source_probs_path="probs",
        probability_threshold=0.5,
    )

    assert probs is not None
    np.testing.assert_array_equal(
        masks,
        np.array([[[[0, 1], [1, 0]]]], dtype=np.uint8),
    )


def test_compute_local_refine_stats_rebuilds_counts_from_staged_arrays() -> None:
    stats = mod._compute_local_refine_stats(
        local_reason_array=np.array(
            [
                "union_source|assigned_by_keypoint",
                "union_source|heading_split",
                "keypoint_fail|union_source",
                "empty_union|union_source",
                "copied_original",
            ],
            dtype=object,
        ),
        smoothing_flags=np.array(
            [
                [1, 0],
                [0, 0],
                [1, 1],
                [0, 0],
                [0, 0],
            ],
            dtype=np.uint8,
        ),
        pixels_reassigned=np.array([3, 0, 2, 0, 0], dtype=np.int32),
        probabilities_used=np.array([True, False, True, False, False], dtype=bool),
    )

    assert stats == {
        "refined": 2,
        "fallback_heading": 1,
        "copied_original": 3,
        "keypoint_fail": 1,
        "empty_union": 1,
        "smoothed_rois": 2,
        "smoothed_channels": 3,
        "components_reassigned": 5,
        "probability_split": 2,
        "assigned_by_keypoint": 1,
    }


def test_apply_driver_global_reason_tags_appends_only_filter_tags() -> None:
    values = np.array(
        [
            "union_source|small_area_pair",
            "refined",
        ],
        dtype=object,
    )
    filter_flags = np.array([[True, False], [False, True]], dtype=bool)
    pair_filter_flags = np.array([True, False], dtype=bool)

    result = mod._apply_driver_global_reason_tags(
        values,
        filter_flags=filter_flags,
        pair_filter_flags=pair_filter_flags,
    )

    assert result.tolist() == [
        "union_source|small_area_pair|filtered_left|filtered_pair",
        "refined|filtered_right",
    ]


def test_write_contours_from_masks_builds_packed_arrays() -> None:
    run_group = _FakeGroup()

    masks = np.zeros((2, 2, 8, 8), dtype=np.uint8)
    masks[0, 0, 1:5, 1:5] = 1
    masks[0, 1, 2:6, 2:6] = 1
    run_group.create_array("masks_roi", data=masks)

    mod._write_contours_from_masks(run_group, total_rois=2, chunk_rois=1)

    left_ptr = np.asarray(run_group["contour_left_ptr"][:], dtype=np.int64)
    left_len = np.asarray(run_group["contour_left_len"][:], dtype=np.int32)
    right_ptr = np.asarray(run_group["contour_right_ptr"][:], dtype=np.int64)
    right_len = np.asarray(run_group["contour_right_len"][:], dtype=np.int32)

    assert left_ptr.tolist() == [0, -1]
    assert right_ptr.tolist() == [0, -1]
    assert left_len[0] > 0
    assert right_len[0] > 0
    assert left_len[1] == 0
    assert right_len[1] == 0
    assert tuple(run_group["contours_left"].shape[1:]) == (2,)
    assert tuple(run_group["contours_right"].shape[1:]) == (2,)


def test_resolve_worker_chunk_rois_tracks_requested_task_width() -> None:
    assert mod._resolve_worker_chunk_rois(total_rois=0, chunk_size=256) == 1
    assert mod._resolve_worker_chunk_rois(total_rois=3, chunk_size=256) == 3
    assert mod._resolve_worker_chunk_rois(total_rois=32, chunk_size=8) == 8
    assert mod._resolve_worker_chunk_rois(total_rois=32, chunk_size=0) == 1
