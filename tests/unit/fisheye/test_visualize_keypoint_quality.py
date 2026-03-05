from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import numpy as np

from fisheye.visualization.visualize_keypoint_quality import (
    _normalize_summary_statistics,
    create_keypoint_quality_visualization,
    create_keypoint_refinement_pipeline_visualization,
)


def test_normalize_summary_statistics_merges_refine_postprocess() -> None:
    summary = {
        "refine": {
            "total_rois": 231,
            "source_success": 188,
            "refined_success": 188,
            "usable_keypoints": 187,
            "flips_corrected": 0,
            "low_confidence": 1,
        },
        "postprocess": {
            "refined_success": 221,
            "usable_keypoints": 221,
            "flip_corrected": 0,
        },
    }

    merged = _normalize_summary_statistics(summary)
    assert merged["total_rois"] == 231
    assert merged["source_success"] == 188
    assert merged["refined_success"] == 221
    assert merged["usable_keypoints"] == 221
    assert merged["low_confidence"] == 1
    assert merged["flips_corrected"] == 0
    assert merged["flip_corrected"] == 0


def test_pipeline_visualization_uses_nested_summary_counts_and_review_timestamp() -> None:
    quality_data = {
        "refined_run": "refined_keypoints_2026-02-04_12-43-25",
        "source_keypoints_run": "keypoints_2026-02-04_17-33-09",
        "source_crop_run": "crop_2026-02-03_23-31-56",
        "source_detect_run": "detect_2026-02-02_18-14-04",
        "summary_statistics": {
            "refine": {
                "total_rois": 231,
                "source_success": 188,
                "refined_success": 188,
                "usable_keypoints": 187,
                "flips_corrected": 0,
            },
            "postprocess": {
                "total_rois": 231,
                "source_success": 188,
                "refined_success": 221,
                "usable_keypoints": 221,
                "flip_corrected": 0,
            },
        },
        "review_status": {
            "state": "approved",
            "method": "manual",
            "intended_use": "training",
            "timestamp": "2026-02-20T02:49:04.412207+00:00",
        },
        "detection_source_counts": {
            "manual_or_clean": 231,
            "interpolated": 0,
            "other": 0,
        },
    }

    fig = create_keypoint_refinement_pipeline_visualization(quality_data)
    try:
        assert len(fig.axes) == 3
        ax_meta, ax_counts, ax_edges = fig.axes

        bar_heights = [float(patch.get_height()) for patch in ax_counts.patches]
        assert bar_heights[:5] == [231.0, 188.0, 221.0, 221.0, 0.0]

        meta_text = ax_meta.texts[0].get_text()
        assert "2026-02-20T02:49:04.412207+00:00" in meta_text
        assert ax_edges.texts[0].get_text() == "No edge-distance metrics"
    finally:
        fig.clf()


def test_pipeline_visualization_renders_edge_distance_panel_when_available() -> None:
    quality_data = {
        "refined_run": "refined_keypoints_2026-03-02_16-01-45",
        "source_keypoints_run": "keypoints_2026-03-02_21-00-45",
        "source_crop_run": "crop_2026-02-10_21-20-47",
        "source_detect_run": "detect_2026-02-09_14-28-54",
        "summary_statistics": {
            "total_rois": 4,
            "source_success": 4,
            "refined_success": 4,
            "usable_keypoints": 4,
            "flips_corrected": 0,
        },
        "review_status": {
            "state": "approved",
            "method": "algorithmic",
            "intended_use": "full_recording",
            "timestamp_utc": "2026-03-02T21:01:51.452861+00:00",
        },
        "detection_source_counts": {
            "manual_or_clean": 4,
            "interpolated": 0,
            "other": 0,
        },
        "edge_pairs": np.asarray([[0, 1], [0, 2], [1, 2]], dtype=np.int32),
        "edge_labels": ["swim_bladder-eye_left", "swim_bladder-eye_right", "eye_left-eye_right"],
        "edge_distances_norm": np.asarray(
            [
                [0.10, 0.20, 0.30],
                [0.12, 0.22, 0.32],
                [0.14, 0.24, 0.34],
                [0.16, 0.26, np.nan],
            ],
            dtype=np.float64,
        ),
        "edge_distance_valid": np.asarray(
            [
                [True, True, True],
                [True, True, True],
                [True, True, True],
                [True, True, False],
            ],
            dtype=bool,
        ),
        "edge_distance_normalization": {"mode": "roi_diagonal", "roi_diagonal": 90.5},
    }

    fig = create_keypoint_refinement_pipeline_visualization(quality_data)
    try:
        assert len(fig.axes) == 3
        _ax_meta, _ax_counts, ax_edges = fig.axes
        assert "Edge Distance P50" in ax_edges.get_title()
        assert len(ax_edges.patches) == 3
    finally:
        fig.clf()


def test_quality_visualization_renders_edge_distance_panel_when_available() -> None:
    quality_data = {
        "refined_run": "refined_keypoints_2026-03-02_16-01-45",
        "source_keypoints_run": "keypoints_2026-03-02_21-00-45",
        "source_crop_run": "crop_2026-02-10_21-20-47",
        "source_detect_run": "detect_2026-02-09_14-28-54",
        "summary_statistics": {
            "total_rois": 4,
            "source_success": 4,
            "refined_success": 4,
            "usable_keypoints": 4,
            "flips_corrected": 0,
        },
        "parameters": {
            "confidence_threshold": 0.3,
            "min_triangle_angle": 10.0,
            "min_triangle_area": 100.0,
        },
        "review_status": {
            "state": "approved",
            "method": "algorithmic",
            "intended_use": "full_recording",
        },
        "total_rows": 4,
        "triangle_area": np.asarray([120.0, 130.0, 140.0, 150.0], dtype=np.float64),
        "min_angle": np.asarray([12.0, 15.0, 18.0, 20.0], dtype=np.float64),
        "confidence": np.asarray([0.91, 0.93, 0.96, 0.97], dtype=np.float64),
        "reason_counts": {"clean": 4},
        "detection_source_counts": {"manual_or_clean": 4, "interpolated": 0, "other": 0},
        "edge_pairs": np.asarray([[0, 1], [0, 2], [1, 2]], dtype=np.int32),
        "edge_labels": ["swim_bladder-eye_left", "swim_bladder-eye_right", "eye_left-eye_right"],
        "edge_distances_norm": np.asarray(
            [
                [0.10, 0.20, 0.30],
                [0.12, 0.22, 0.32],
                [0.14, 0.24, 0.34],
                [0.16, 0.26, np.nan],
            ],
            dtype=np.float64,
        ),
        "edge_distance_valid": np.asarray(
            [
                [True, True, True],
                [True, True, True],
                [True, True, True],
                [True, True, False],
            ],
            dtype=bool,
        ),
        "edge_distance_normalization": {"mode": "roi_diagonal", "roi_diagonal": 90.5},
    }

    fig = create_keypoint_quality_visualization(quality_data)
    try:
        assert len(fig.axes) == 8
        ax_edges = fig.axes[3]
        assert "Edge Distance P50" in ax_edges.get_title()
        assert len(ax_edges.patches) == 3
    finally:
        fig.clf()
