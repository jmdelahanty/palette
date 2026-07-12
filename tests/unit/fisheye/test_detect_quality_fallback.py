from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr

from fisheye.refinement import detect_quality as detect_quality_mod
from fisheye.refinement.detect_quality import analyze_detect_quality, save_quality_report


def _normalize_report(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _normalize_report(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_report(item) for item in value]
    return value


def _write_frame_count_precedence_archive(zarr_path: Path) -> None:
    root = zarr.open_group(store=zarr_path, mode="w")
    root.attrs["width"] = 4512
    root.attrs["height"] = 4512
    root.attrs["total_frames"] = 10

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


def test_analyze_detect_quality_handles_raw_video_without_images_ds(tmp_path: Path) -> None:
    zarr_path = tmp_path / "analysis.zarr"
    _write_frame_count_precedence_archive(zarr_path)

    report = analyze_detect_quality(str(zarr_path), run_name="detect_2026-02-09_12-00-00")
    assert report["source_run"] == "detect_2026-02-09_12-00-00"
    # Uses detect/frame_counts as frame universe when imported frames are absent.
    assert report["coverage"]["total_frames"] == 5
    assert report["bbox_validation"]["total_bboxes"] == 3


def test_analyze_detect_quality_reads_run_when_consolidated_metadata_is_stale(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "analysis_stale.zarr"
    _write_frame_count_precedence_archive(zarr_path)
    zarr.consolidate_metadata(str(zarr_path))

    live_root = zarr.open_group(str(zarr_path), mode="a", use_consolidated=False)
    detect = live_root["detect_runs"].create_group("detect_fresh")
    detect.create_array("frame_indices", data=np.array([0], dtype=np.int32))
    detect.create_array(
        "bbox_norm_coords",
        data=np.array([[0.5, 0.5, 0.1, 0.1]], dtype=np.float64),
    )
    detect.create_array("frame_counts", data=np.array([1], dtype=np.int32))
    live_root["detect_runs"].attrs["latest"] = "detect_fresh"

    report = analyze_detect_quality(str(zarr_path), run_name="detect_fresh")

    assert report["source_run"] == "detect_fresh"
    assert report["bbox_validation"]["total_bboxes"] == 1


def test_detect_quality_frame_domains_matches_legacy_report_and_saved_content(
    monkeypatch,
    tmp_path: Path,
) -> None:
    normal_path = tmp_path / "analysis_domains.zarr"
    legacy_path = tmp_path / "analysis_legacy.zarr"
    _write_frame_count_precedence_archive(normal_path)
    _write_frame_count_precedence_archive(legacy_path)

    normal_report = analyze_detect_quality(
        str(normal_path),
        run_name="detect_2026-02-09_12-00-00",
    )
    normal_quality_path = save_quality_report(
        str(normal_path),
        normal_report,
        quality_run_name="detect_quality_domains",
    )

    monkeypatch.setattr(
        detect_quality_mod,
        "_run_frame_count_from_domains",
        lambda _root, _detect_group: None,
    )
    legacy_report = analyze_detect_quality(
        str(legacy_path),
        run_name="detect_2026-02-09_12-00-00",
    )
    legacy_quality_path = save_quality_report(
        str(legacy_path),
        legacy_report,
        quality_run_name="detect_quality_legacy",
    )

    assert _normalize_report(normal_report) == _normalize_report(legacy_report)
    assert normal_report["coverage"]["total_frames"] == 5

    normal_quality = zarr.open_group(str(normal_path), mode="r")[normal_quality_path]
    legacy_quality = zarr.open_group(str(legacy_path), mode="r")[legacy_quality_path]
    assert normal_quality.attrs["coverage_stats"] == legacy_quality.attrs["coverage_stats"]
    assert normal_quality.attrs["detection_quality_summary"] == (
        legacy_quality.attrs["detection_quality_summary"]
    )
    np.testing.assert_array_equal(
        normal_quality["quality_flags"][:],
        legacy_quality["quality_flags"][:],
    )
    np.testing.assert_array_equal(
        normal_quality["detection_quality_labels"][:],
        legacy_quality["detection_quality_labels"][:],
    )


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


def test_detect_quality_expected_subject_count_marks_only_over_expected_frames(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "analysis_multi_subject.zarr"
    root = zarr.open_group(store=zarr_path, mode="w")
    root.attrs["width"] = 4512
    root.attrs["height"] = 4512
    root.attrs["total_frames"] = 4

    detect_parent = root.create_group("detect_runs")
    detect = detect_parent.create_group("detect_multi")
    detect_parent.attrs["latest"] = "detect_multi"

    frame_indices = np.array(
        [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3],
        dtype=np.int32,
    )
    centers = np.linspace(0.1, 0.9, frame_indices.shape[0], dtype=np.float64)
    bbox_norm = np.column_stack(
        [
            centers,
            centers[::-1],
            np.full(frame_indices.shape[0], 0.05, dtype=np.float64),
            np.full(frame_indices.shape[0], 0.05, dtype=np.float64),
        ]
    )
    frame_counts = np.array([4, 4, 5, 3], dtype=np.int32)

    detect.create_array("frame_indices", data=frame_indices, overwrite=True)
    detect.create_array("bbox_norm_coords", data=bbox_norm, overwrite=True)
    detect.create_array("frame_counts", data=frame_counts, overwrite=True)

    report = analyze_detect_quality(
        str(zarr_path),
        run_name="detect_multi",
        expected_subject_count=4,
    )

    assert report["coverage"]["expected_count"] == 4
    assert report["coverage"]["frames_with_expected_count"] == 2
    assert report["coverage"]["frames_under_expected"] == 1
    assert report["coverage"]["frames_over_expected"] == 1
    assert report["coverage"]["multi_detection_frames"] == 1
    assert report["artifacts"]["temporal_artifact_policy"] == "skipped_expected_subject_count_gt_1"
    assert report["quality_score"]["mode"] == "expected_count"

    quality_path = save_quality_report(
        str(zarr_path),
        report,
        quality_run_name="detect_quality_expected4",
    )
    quality = zarr.open_group(str(zarr_path), mode="r")[quality_path]

    np.testing.assert_array_equal(
        quality["quality_flags"][:],
        np.array([0, 0, 4, 0], dtype=np.int8),
    )
    labels = quality["detection_quality_labels"][:]
    assert labels[:8].tolist() == [0] * 8
    assert labels[8:13].tolist() == [4] * 5
    assert labels[13:].tolist() == [0] * 3
    assert quality.attrs["expected_subject_count"] == 4
    assert quality.attrs["count_policy"]["over_expected_rule"] == "frame_counts > 4"


def test_detect_quality_without_expected_count_preserves_legacy_single_subject_multi_labels(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "analysis_single_subject.zarr"
    root = zarr.open_group(store=zarr_path, mode="w")
    root.attrs["width"] = 640
    root.attrs["height"] = 640
    root.attrs["total_frames"] = 2

    detect_parent = root.create_group("detect_runs")
    detect = detect_parent.create_group("detect_single")
    detect_parent.attrs["latest"] = "detect_single"

    frame_indices = np.array([0, 1, 1], dtype=np.int32)
    bbox_norm = np.array(
        [
            [0.50, 0.50, 0.05, 0.05],
            [0.51, 0.50, 0.05, 0.05],
            [0.52, 0.50, 0.05, 0.05],
        ],
        dtype=np.float64,
    )
    frame_counts = np.array([1, 2], dtype=np.int32)

    detect.create_array("frame_indices", data=frame_indices, overwrite=True)
    detect.create_array("bbox_norm_coords", data=bbox_norm, overwrite=True)
    detect.create_array("frame_counts", data=frame_counts, overwrite=True)

    report = analyze_detect_quality(str(zarr_path), run_name="detect_single")
    assert report["coverage"]["multi_detection_frames"] == 1

    explicit_single_report = analyze_detect_quality(
        str(zarr_path),
        run_name="detect_single",
        expected_subject_count=1,
    )
    assert explicit_single_report["coverage"]["multi_detection_frames"] == 1
    assert explicit_single_report["artifacts"]["temporal_artifact_policy"] == "global_row_sequence"
    assert "mode" not in explicit_single_report["quality_score"]

    quality_path = save_quality_report(
        str(zarr_path),
        report,
        quality_run_name="detect_quality_legacy",
    )
    quality = zarr.open_group(str(zarr_path), mode="r")[quality_path]
    np.testing.assert_array_equal(
        quality["quality_flags"][:],
        np.array([0, 4], dtype=np.int8),
    )
    np.testing.assert_array_equal(
        quality["detection_quality_labels"][:],
        np.array([0, 4, 4], dtype=np.int8),
    )
