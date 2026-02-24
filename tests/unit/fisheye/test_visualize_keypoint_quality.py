from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from fisheye.visualization.visualize_keypoint_quality import (
    _normalize_summary_statistics,
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
        assert len(fig.axes) == 2
        ax_meta, ax_counts = fig.axes

        bar_heights = [float(patch.get_height()) for patch in ax_counts.patches]
        assert bar_heights[:5] == [231.0, 188.0, 221.0, 221.0, 0.0]

        meta_text = ax_meta.texts[0].get_text()
        assert "2026-02-20T02:49:04.412207+00:00" in meta_text
    finally:
        fig.clf()

