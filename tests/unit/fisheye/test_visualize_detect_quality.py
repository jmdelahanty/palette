from __future__ import annotations

import matplotlib
import numpy as np
import zarr

matplotlib.use("Agg")

from fisheye.visualization.visualize_detect_quality import create_quality_visualization, load_quality_report


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


def test_load_quality_report_handles_raw_video_without_images_ds(tmp_path) -> None:
    zarr_path = tmp_path / "analysis.zarr"
    root = zarr.open_group(store=zarr_path, mode="w")
    root.attrs["width"] = 4512
    root.attrs["height"] = 4512

    raw = root.create_group("raw_video")
    raw.attrs["downsampled_resolution"] = [720, 1280]

    detect_parent = root.create_group("detect_runs")
    detect = detect_parent.create_group("detect_2026-02-27_00-00-00")
    detect_parent.attrs["latest"] = "detect_2026-02-27_00-00-00"

    detect.create_array("frame_indices", data=np.array([0, 1], dtype=np.int32), overwrite=True)
    detect.create_array(
        "bbox_norm_coords",
        data=np.array(
            [
                [0.5, 0.25, 0.1, 0.1],
                [0.2, 0.4, 0.1, 0.1],
            ],
            dtype=np.float32,
        ),
        overwrite=True,
    )
    detect.create_array("frame_counts", data=np.array([1, 1], dtype=np.int32), overwrite=True)

    quality_parent = detect.create_group("quality_reports")
    quality = quality_parent.create_group("quality_2026-02-27_00-00-00")
    quality_parent.attrs["latest"] = "quality_2026-02-27_00-00-00"

    quality.create_array("quality_flags", data=np.array([0, 0], dtype=np.int32), overwrite=True)
    quality.create_array("detection_quality_labels", data=np.array([0, 0], dtype=np.int32), overwrite=True)
    quality.attrs["quality_score"] = {
        "grade": "A",
        "overall_score": 99.0,
        "coverage_score": 100.0,
        "artifact_score": 100.0,
        "bbox_score": 99.0,
    }
    quality.attrs["coverage_stats"] = {
        "coverage_percent": 100.0,
        "gaps": {"total_count": 0, "longest_gap": 0, "mean_gap_size": 0.0},
    }
    quality.attrs["bbox_validation"] = {
        "total_bboxes": 2,
        "out_of_range": 0,
        "size_outliers": 0,
        "malformed": 0,
        "mean_size": 0.1,
        "std_size": 0.01,
        "size_cv": 0.1,
    }
    quality.attrs["detection_quality_summary"] = {
        "total_detections": 2,
        "clean_detections": 2,
        "clean_percentage": 100.0,
        "empty_frames": 0,
        "blip_detections": 0,
        "jump_detections": 0,
        "total_frames": 2,
        "frames_with_detections": 2,
        "clean_frames": 2,
    }

    _, detection_data = load_quality_report(str(zarr_path))
    assert detection_data["width"] == 1280
    assert detection_data["height"] == 720
    assert np.isclose(detection_data["centroids"][0, 0], 640.0)
    assert np.isclose(detection_data["centroids"][0, 1], 180.0)
