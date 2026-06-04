from __future__ import annotations

import numpy as np
import pytest

from fisheye.diagnostics.compare_realtime_offline_detections import DetectionRows
from fisheye.diagnostics.compare_realtime_offline_detections import compare_detection_rows
from fisheye.diagnostics.compare_realtime_offline_detections import compute_bbox_iou
from fisheye.diagnostics.compare_realtime_offline_detections import summarize_comparison
from fisheye.diagnostics.compare_realtime_offline_detections import _select_top_one_per_frame
from fisheye.visualization.plot_detection_epoch_heatmaps import EpochWindow


def _rows(
    *,
    frames: list[int],
    centers: list[tuple[float, float]],
    confidence: list[float] | None = None,
) -> DetectionRows:
    centers_arr = np.asarray(centers, dtype=np.float64)
    bbox = np.column_stack(
        [
            centers_arr[:, 0] - 5.0,
            centers_arr[:, 1] - 5.0,
            centers_arr[:, 0] + 5.0,
            centers_arr[:, 1] + 5.0,
        ]
    )
    return DetectionRows(
        source_path="source/path",
        source_kind="test",
        run_name="run",
        frame_indices=np.asarray(frames, dtype=np.int64),
        bbox_img_xyxy=bbox,
        centers_xy=centers_arr,
        confidence=np.asarray(confidence if confidence is not None else [np.nan] * len(frames), dtype=np.float64),
        row_indices=np.arange(len(frames), dtype=np.int64),
    )


def test_select_top_one_per_frame_prefers_highest_confidence() -> None:
    selected = _select_top_one_per_frame(
        np.array([2, 1, 1, 2, 3], dtype=np.int64),
        confidence=np.array([0.2, 0.3, 0.9, 0.8, 0.1], dtype=np.float64),
    )

    assert selected.tolist() == [2, 3, 4]


def test_compute_bbox_iou_reports_overlap() -> None:
    iou = compute_bbox_iou(
        np.array([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]], dtype=np.float64),
        np.array([[5.0, 5.0, 15.0, 15.0], [20.0, 20.0, 30.0, 30.0]], dtype=np.float64),
    )

    assert iou[0] == pytest.approx(25.0 / 175.0)
    assert iou[1] == pytest.approx(0.0)


def test_compare_detection_rows_aligns_union_frames_and_metrics() -> None:
    offline = _rows(frames=[1, 2, 4], centers=[(10.0, 10.0), (20.0, 20.0), (40.0, 40.0)])
    realtime = _rows(frames=[2, 3, 4], centers=[(23.0, 24.0), (30.0, 30.0), (40.0, 40.0)])

    arrays = compare_detection_rows(
        offline,
        realtime,
        epoch_windows=[EpochWindow("stim", 0.0, 1.0, start_frame=2, end_frame=4, source="stimulus_events")],
    )

    assert arrays.frame_indices.tolist() == [1, 2, 3, 4]
    assert arrays.offline_present.tolist() == [True, True, False, True]
    assert arrays.realtime_present.tolist() == [False, True, True, True]
    assert arrays.epoch_label_code.tolist() == [0, 1, 1, 1]
    assert arrays.centroid_delta_px[1] == pytest.approx(5.0)
    assert arrays.centroid_delta_px[3] == pytest.approx(0.0)
    assert np.isnan(arrays.centroid_delta_px[0])


def test_summarize_comparison_counts_presence_and_epochs() -> None:
    offline = _rows(frames=[1, 2, 4], centers=[(10.0, 10.0), (20.0, 20.0), (40.0, 40.0)])
    realtime = _rows(frames=[2, 3, 4], centers=[(23.0, 24.0), (30.0, 30.0), (40.0, 40.0)])
    windows = [EpochWindow("stim", 0.0, 1.0, start_frame=2, end_frame=4, source="stimulus_events")]

    arrays = compare_detection_rows(offline, realtime, epoch_windows=windows)
    summary = summarize_comparison(arrays, total_frames=5, epoch_windows=windows)

    assert summary["offline_present_count"] == 3
    assert summary["realtime_present_count"] == 3
    assert summary["both_present_count"] == 2
    assert summary["offline_only_count"] == 1
    assert summary["realtime_only_count"] == 1
    assert summary["neither_present_count"] == 1
    assert summary["centroid_delta_px"]["p50"] == pytest.approx(2.5)
    assert summary["epochs"]["stim"]["both_present_count"] == 2
    assert summary["epochs"]["stim"]["offline_only_count"] == 0
    assert summary["epochs"]["stim"]["realtime_only_count"] == 1
