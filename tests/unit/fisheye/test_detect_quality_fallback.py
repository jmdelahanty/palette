from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr

from fisheye.refinement.detect_quality import analyze_detect_quality


def test_analyze_detect_quality_handles_raw_video_without_images_ds(tmp_path: Path) -> None:
    zarr_path = tmp_path / "analysis.zarr"
    root = zarr.open_group(store=zarr_path, mode="w")
    root.attrs["width"] = 4512
    root.attrs["height"] = 4512
    root.attrs["total_frames"] = 10

    # Archive has raw_video metadata, but no imported images dataset.
    raw = root.create_group("raw_video")
    raw.attrs["source_video"] = "Cam2010093.mp4"

    detect_parent = root.create_group("detect_runs")
    detect = detect_parent.create_group("detect_2026-02-09_12-00-00")
    detect_parent.attrs["latest"] = "detect_2026-02-09_12-00-00"

    frame_indices = np.array([0, 2, 3], dtype=np.int32)
    bbox_norm = np.array(
        [
            [0.5, 0.5, 0.1, 0.1],
            [0.51, 0.5, 0.1, 0.1],
            [0.52, 0.5, 0.1, 0.1],
        ],
        dtype=np.float64,
    )
    frame_counts = np.array([1, 0, 1, 1, 0], dtype=np.int32)

    detect.create_array("frame_indices", data=frame_indices, overwrite=True)
    detect.create_array("bbox_norm_coords", data=bbox_norm, overwrite=True)
    detect.create_array("frame_counts", data=frame_counts, overwrite=True)

    report = analyze_detect_quality(str(zarr_path), run_name="detect_2026-02-09_12-00-00")
    assert report["source_run"] == "detect_2026-02-09_12-00-00"
    # Uses detect/frame_counts as frame universe when imported frames are absent.
    assert report["coverage"]["total_frames"] == 5
    assert report["bbox_validation"]["total_bboxes"] == 3


def test_analyze_detect_quality_scaled_threshold_is_resolution_invariant(tmp_path: Path) -> None:
    zarr_path = tmp_path / "analysis_scaled.zarr"
    root = zarr.open_group(store=zarr_path, mode="w")
    root.attrs["width"] = 4512
    root.attrs["height"] = 4512
    root.attrs["total_frames"] = 2

    raw = root.create_group("raw_video")
    raw.attrs["source_video"] = "Cam2010093.mp4"

    detect_parent = root.create_group("detect_runs")
    detect = detect_parent.create_group("detect_2026-02-09_12-00-00")
    detect_parent.attrs["latest"] = "detect_2026-02-09_12-00-00"

    frame_indices = np.array([0, 1], dtype=np.int32)
    # ~135 px jump at 4512 width (0.03 * 4512), above 100px absolute threshold.
    bbox_norm = np.array(
        [
            [0.50, 0.50, 0.1, 0.1],
            [0.53, 0.50, 0.1, 0.1],
        ],
        dtype=np.float64,
    )
    frame_counts = np.array([1, 1], dtype=np.int32)

    detect.create_array("frame_indices", data=frame_indices, overwrite=True)
    detect.create_array("bbox_norm_coords", data=bbox_norm, overwrite=True)
    detect.create_array("frame_counts", data=frame_counts, overwrite=True)

    report_pixels = analyze_detect_quality(
        str(zarr_path),
        run_name="detect_2026-02-09_12-00-00",
        jump_threshold=100.0,
        threshold_mode="pixels",
    )
    report_scaled = analyze_detect_quality(
        str(zarr_path),
        run_name="detect_2026-02-09_12-00-00",
        jump_threshold=100.0,
        threshold_mode="scaled",
        threshold_reference_width=640.0,
    )

    assert report_pixels["artifacts"]["jump_threshold_pixels_effective"] == 100.0
    assert len(report_pixels["artifacts"]["jumps"]) == 1
    assert report_scaled["artifacts"]["jump_threshold_pixels_effective"] == 705.0
    assert len(report_scaled["artifacts"]["jumps"]) == 0
