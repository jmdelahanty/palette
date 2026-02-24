from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")

from fisheye.visualization.visualize_detect_quality import create_quality_visualization


def test_create_quality_visualization_tolerates_missing_score_fields() -> None:
    quality_data = {
        "quality_flags": np.array([0, 0, 2, 3], dtype=np.int32),
        "detection_quality_labels": np.array([0, 0, 2, 3], dtype=np.int32),
        "empty_frames": np.array([], dtype=np.int32),
        "clean_frames": np.array([0, 1], dtype=np.int32),
        "blip_frames": np.array([2], dtype=np.int32),
        "jump_frames": np.array([3], dtype=np.int32),
        "multi_frames": np.array([], dtype=np.int32),
        "quality_score": {
            "grade": "B",
            "overall_score": 88.5,
            "coverage_score": 96.0,
            # legacy payload may omit artifact_score / bbox_score
        },
        "coverage_stats": {
            "coverage_percent": 100.0,
            "gaps": {"total_count": 0, "longest_gap": 0, "mean_gap_size": 0.0},
        },
        "bbox_validation": {
            "total_bboxes": 4,
            "out_of_range": 0,
            "size_outliers": 0,
            "malformed": 0,
            "mean_size": 0.2,
            "std_size": 0.01,
            "size_cv": 0.05,
        },
        "detection_summary": {
            "clean_detections": 2,
            "clean_percentage": 50.0,
            "total_frames": 4,
            "empty_frames": 0,
            "frames_with_detections": 4,
            "clean_frames": 2,
        },
    }
    detection_data = {
        "centroids": np.array([[10.0, 10.0], [11.0, 11.0]], dtype=np.float32),
        "frame_indices": np.array([0, 1], dtype=np.int32),
        "frame_counts": np.array([1, 1, 1, 1], dtype=np.int32),
        "bbox_coords": np.array([[0.5, 0.5, 0.1, 0.1]], dtype=np.float32),
        "width": 640,
        "height": 640,
    }

    fig = create_quality_visualization(quality_data, detection_data)
    assert fig is not None
    fig.clf()
