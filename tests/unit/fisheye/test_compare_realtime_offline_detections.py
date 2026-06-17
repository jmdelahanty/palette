from __future__ import annotations

import json

import numpy as np
import pytest

from fisheye.diagnostics.compare_realtime_offline_detections import CropMetaRows
from fisheye.diagnostics.compare_realtime_offline_detections import DetectionRows
from fisheye.diagnostics.compare_realtime_offline_detections import compare_detection_rows
from fisheye.diagnostics.compare_realtime_offline_detections import compute_bbox_iou
from fisheye.diagnostics.compare_realtime_offline_detections import load_crop_meta_realtime_detection_rows
from fisheye.diagnostics.compare_realtime_offline_detections import resolve_crop_meta_path
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


def test_resolve_crop_meta_path_prefers_recording_manifest(tmp_path) -> None:
    recording_dir = tmp_path / "recording"
    zarr_dir = recording_dir / "zarr"
    crop_dir = recording_dir / "derived" / "external_crop_recorder"
    zarr_dir.mkdir(parents=True)
    crop_dir.mkdir(parents=True)
    crop_meta = crop_dir / "Cam2010093_session_crop_meta.csv"
    crop_meta.write_text("recording_frame_id,has_detection,blank_frame\n", encoding="utf-8")
    (recording_dir / "recording_manifest.json").write_text(
        json.dumps(
            {
                "video_streams": {
                    "streams": {
                        "crop": {
                            "metadata": "derived/external_crop_recorder/Cam2010093_session_crop_meta.csv",
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    resolved = resolve_crop_meta_path(zarr_dir / "recording_analysis.zarr")

    assert resolved == crop_meta


def test_load_crop_meta_realtime_detection_rows_filters_blank_and_converts_frames(tmp_path) -> None:
    crop_meta = tmp_path / "crop_meta.csv"
    crop_meta.write_text(
        "\n".join(
            [
                "recording_frame_id,local_frame_id,camera_frame_id,timestamp,timestamp_sys,has_detection,blank_frame,detection_confidence,crop_x,crop_y,crop_w,crop_h,detection_x,detection_y,detection_w,detection_h",
                "1,10,100,0,0,1,0,0.5,0,0,100,100,10,20,30,40",
                "2,11,101,0,0,1,1,0.6,0,0,100,100,20,30,30,40",
                "3,12,102,0,0,0,0,0.0,0,0,100,100,,,,",
                "4,13,103,0,0,1,0,0.9,5,6,100,100,30,40,20,10",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    detections, crop_rows = load_crop_meta_realtime_detection_rows(crop_meta)

    assert crop_rows.frame_indices.tolist() == [0, 1, 2, 3]
    assert crop_rows.has_detection.tolist() == [True, True, False, True]
    assert crop_rows.blank_frame.tolist() == [False, True, False, False]
    assert detections.frame_indices.tolist() == [0, 3]
    assert detections.confidence.tolist() == pytest.approx([0.5, 0.9])
    np.testing.assert_allclose(
        detections.bbox_img_xyxy,
        np.asarray([[10.0, 20.0, 40.0, 60.0], [30.0, 40.0, 50.0, 50.0]], dtype=np.float64),
    )


def test_crop_sufficiency_arrays_and_summary_reason_codes() -> None:
    offline = DetectionRows(
        source_path="offline",
        source_kind="test",
        run_name="offline",
        frame_indices=np.asarray([0, 1, 2, 3, 4], dtype=np.int64),
        bbox_img_xyxy=np.asarray(
            [
                [10.0, 10.0, 20.0, 20.0],
                [10.0, 10.0, 20.0, 20.0],
                [10.0, 10.0, 20.0, 20.0],
                [10.0, 10.0, 20.0, 20.0],
                [200.0, 200.0, 210.0, 210.0],
            ],
            dtype=np.float64,
        ),
        centers_xy=np.asarray(
            [(15.0, 15.0), (15.0, 15.0), (15.0, 15.0), (15.0, 15.0), (205.0, 205.0)],
            dtype=np.float64,
        ),
        confidence=np.full((5,), np.nan, dtype=np.float64),
        row_indices=np.arange(5, dtype=np.int64),
    )
    realtime = _rows(frames=[0, 4], centers=[(15.0, 15.0), (205.0, 205.0)])
    crop_meta = CropMetaRows(
        source_path="crop_meta.csv",
        frame_indices=np.asarray([0, 1, 2, 4], dtype=np.int64),
        crop_xywh=np.asarray(
            [
                [0.0, 0.0, 100.0, 100.0],
                [0.0, 0.0, 100.0, 100.0],
                [0.0, 0.0, 100.0, 100.0],
                [0.0, 0.0, 100.0, 100.0],
            ],
            dtype=np.float64,
        ),
        has_detection=np.asarray([True, True, False, True], dtype=bool),
        blank_frame=np.asarray([False, True, False, False], dtype=bool),
        row_indices=np.arange(4, dtype=np.int64),
    )

    arrays = compare_detection_rows(offline, realtime, crop_meta=crop_meta)
    summary = summarize_comparison(arrays, total_frames=5, epoch_windows=[], crop_meta=crop_meta)

    assert arrays.crop_sufficiency_reason_code.tolist() == [2, 4, 5, 3, 6]
    assert arrays.offline_bbox_inside_realtime_crop.tolist() == [True, False, False, False, False]
    assert arrays.offline_crop_edge_margins[0].tolist() == pytest.approx([10.0, 10.0, 80.0, 80.0])
    assert summary["crop_sufficiency_available"] is True
    assert summary["offline_full_bbox_inside_crop_count"] == 1
    assert summary["offline_full_bbox_inside_crop_pct"] == pytest.approx(20.0)
    assert summary["blank_crop_rows_for_offline_count"] == 1
    assert summary["no_detection_crop_rows_for_offline_count"] == 1
    assert summary["missing_crop_rows_for_offline_count"] == 1
    assert summary["crop_elsewhere_rows_for_offline_count"] == 1
    assert summary["crop_meta_row_count"] == 4
