from __future__ import annotations

import matplotlib
import numpy as np
import zarr

matplotlib.use("Agg")

from fisheye.visualization.detection_coverage_dashboard import (
    DetectionCoverageSeries,
    DETECTION_COVERAGE_DASHBOARD_ARTIFACT,
    compute_missing_segments,
    create_detection_coverage_dashboard,
    load_raw_detect_coverage_series,
    load_refined_detect_coverage_series,
    render_detection_coverage_png,
    summarize_detection_coverage,
    write_detection_coverage_dashboard_artifact,
)


def test_compute_missing_segments_uses_end_exclusive_ranges() -> None:
    present = np.array([True, False, False, True, False, True], dtype=bool)

    assert compute_missing_segments(present) == [(1, 3), (4, 5)]


def test_summarize_detection_coverage_reports_neutral_statuses() -> None:
    series = DetectionCoverageSeries(
        name="raw/test",
        frame_counts=np.array([1, 1, 0, 0, 1, 1, 1, 0], dtype=np.int32),
    )

    summary = summarize_detection_coverage(series)

    assert summary.total_frames == 8
    assert summary.frames_with_detections == 5
    assert summary.missing_segment_count == 2
    assert summary.max_missing_segment_frames == 2
    assert summary.coverage_status == "low"
    assert summary.review_priority == "high"
    text = "\n".join(summary.as_text_lines()).lower()
    assert "coverage status" in text
    assert "review priority" in text


def test_dashboard_renders_single_and_comparison_series() -> None:
    raw = DetectionCoverageSeries(
        name="raw/test",
        frame_counts=np.array([1, 0, 1, 1, 0, 1], dtype=np.int32),
    )
    refined = DetectionCoverageSeries(
        name="refined/test",
        frame_counts=np.array([1, 1, 1, 1, 0, 1], dtype=np.int32),
    )

    fig = create_detection_coverage_dashboard([raw, refined], rolling_window=2, frames_per_row=3)
    assert fig is not None
    fig.clf()

    png = render_detection_coverage_png([raw], rolling_window=2, frames_per_row=3, dpi=80)
    assert png.startswith(b"\x89PNG")


def test_raw_and_refined_loaders_read_shared_coverage_inputs(tmp_path) -> None:
    zarr_path = tmp_path / "analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    root.attrs["total_frames"] = 6

    detect_parent = root.create_group("detect_runs")
    detect_parent.attrs["latest"] = "detect_a"
    detect = detect_parent.create_group("detect_a")
    detect.create_array("frame_indices", data=np.array([0, 2, 2, 5], dtype=np.int32), overwrite=True)
    detect.create_array(
        "bbox_norm_coords",
        data=np.zeros((4, 4), dtype=np.float32),
        overwrite=True,
    )

    quality_parent = detect.create_group("quality_reports")
    quality_parent.attrs["latest"] = "quality_a"
    quality = quality_parent.create_group("quality_a")
    quality.create_array(
        "quality_flags",
        data=np.array([0, -1, 0, -1, -1, 0], dtype=np.int32),
        overwrite=True,
    )

    refined_parent = root.create_group("refined_detect_runs")
    refined_parent.attrs["latest"] = "refined_a"
    refined = refined_parent.create_group("refined_a")
    refined.attrs["source_detect_run"] = "detect_a"
    instances = refined.create_group("instances")
    instances.create_array("frame_indices", data=np.array([0, 1, 5], dtype=np.int32), overwrite=True)
    instances.create_array(
        "frame_counts",
        data=np.array([1, 1, 0, 0, 0, 1], dtype=np.int32),
        overwrite=True,
    )

    raw_series = load_raw_detect_coverage_series(zarr_path)
    refined_series = load_refined_detect_coverage_series(zarr_path)

    assert raw_series.name == "raw/detect_a"
    assert raw_series.frame_counts.tolist() == [1, 0, 2, 0, 0, 1]
    assert raw_series.quality_flags is not None
    assert refined_series.name == "refined/refined_a"
    assert refined_series.frame_counts.tolist() == [1, 1, 0, 0, 0, 1]
    assert refined_series.attrs is not None
    assert refined_series.attrs["source_detect_run"] == "detect_a"


def test_write_dashboard_artifacts_targets_raw_and_refined_runs(tmp_path) -> None:
    zarr_path = tmp_path / "analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    root.attrs["total_frames"] = 4

    detect_parent = root.create_group("detect_runs")
    detect_parent.attrs["latest"] = "detect_a"
    detect = detect_parent.create_group("detect_a")
    detect.create_array("frame_counts", data=np.array([1, 0, 1, 1], dtype=np.int32), overwrite=True)
    quality_parent = detect.create_group("quality_reports")
    quality_parent.attrs["latest"] = "quality_a"
    quality_parent.create_group("quality_a")

    refined_parent = root.create_group("refined_detect_runs")
    refined_parent.attrs["latest"] = "refined_a"
    refined = refined_parent.create_group("refined_a")
    refined.attrs["source_detect_run"] = "detect_a"
    instances = refined.create_group("instances")
    instances.create_array("frame_counts", data=np.array([1, 1, 1, 1], dtype=np.int32), overwrite=True)

    raw_series = load_raw_detect_coverage_series(zarr_path)
    raw_png = render_detection_coverage_png([raw_series], dpi=80)
    raw_result = write_detection_coverage_dashboard_artifact(
        zarr_path,
        [raw_series],
        raw_png,
        mode="raw",
        dpi=80,
    )

    refined_series = load_refined_detect_coverage_series(zarr_path)
    compare_png = render_detection_coverage_png([raw_series, refined_series], dpi=80)
    refined_result = write_detection_coverage_dashboard_artifact(
        zarr_path,
        [raw_series, refined_series],
        compare_png,
        mode="compare",
        dpi=80,
    )

    reopened = zarr.open_group(str(zarr_path), mode="r")
    raw_artifact = reopened["detect_runs/detect_a/visualizations"][DETECTION_COVERAGE_DASHBOARD_ARTIFACT]
    refined_artifact = reopened["refined_detect_runs/refined_a/visualizations"][DETECTION_COVERAGE_DASHBOARD_ARTIFACT]

    assert raw_result.path == f"visualizations/{DETECTION_COVERAGE_DASHBOARD_ARTIFACT}"
    assert refined_result.path == f"visualizations/{DETECTION_COVERAGE_DASHBOARD_ARTIFACT}"
    assert raw_artifact.attrs["dashboard_mode"] == "raw"
    assert raw_artifact.attrs["target_kind"] == "raw_detection"
    assert refined_artifact.attrs["dashboard_mode"] == "compare"
    assert refined_artifact.attrs["target_kind"] == "refined_detection"
    assert refined_artifact.attrs["source_runs"]["detect_run"] == "detect_a"
    assert refined_artifact.attrs["source_runs"]["refined_run"] == "refined_a"
