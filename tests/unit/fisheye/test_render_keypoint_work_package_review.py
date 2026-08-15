from __future__ import annotations

import numpy as np
from PIL import Image

from fisheye.utils.render_keypoint_work_package_review import (
    _annotated_panel,
    render_review_montage,
    select_review_rows,
)


def test_select_review_rows_keeps_failures_low_scores_and_spread_distinct() -> None:
    success = np.asarray([False, True, True, True, True, True, True, True, True, False])
    confidence = np.asarray([np.nan, 0.9, 0.2, 0.8, 0.1, 0.7, 0.3, 0.6, 0.4, np.nan])
    providers = np.asarray([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    selected = select_review_rows(
        detection_success=success,
        confidence=confidence,
        provider_codes=providers,
        low_confidence_per_provider=1,
        spread_per_provider=2,
    )

    assert selected["failures"] == [0, 9]
    assert selected["provider_0_low_confidence"] == [4]
    assert selected["provider_1_low_confidence"] == [6]
    flattened = [row for rows in selected.values() for row in rows]
    assert len(flattened) == len(set(flattened))
    assert set(selected["provider_0_spread"]).issubset({1, 2, 3})
    assert set(selected["provider_1_spread"]).issubset({5, 7, 8})


def test_render_review_montage_has_expected_grid_extent() -> None:
    panels = [Image.new("RGB", (100, 125), (index, 0, 0)) for index in range(5)]
    montage = render_review_montage(panels, columns=2, panel_width=80)
    assert montage.size == (160, 300)


def test_annotated_panel_marks_success_and_failure_borders() -> None:
    pixel = np.full((64, 64), 100, dtype=np.uint8)
    keypoints = np.asarray(
        [[20, 30], [25, 20], [35, 20], [30, 10], [20, 50]], dtype=np.float64
    )
    success = _annotated_panel(
        pixel=pixel,
        output_row=1,
        crop_row=2,
        frame_index=3,
        provider_name="acquisition_crop_video",
        success=True,
        confidence=0.9,
        failure_code=0,
        keypoints_roi=keypoints,
        pose_bbox_xyxy_roi=np.asarray([10, 5, 50, 55]),
    )
    failure = _annotated_panel(
        pixel=pixel,
        output_row=4,
        crop_row=5,
        frame_index=6,
        provider_name="acquisition_crop_video",
        success=False,
        confidence=np.nan,
        failure_code=1,
        keypoints_roi=np.full((5, 2), np.nan),
        pose_bbox_xyxy_roi=np.full(4, np.nan),
    )

    assert success.size == (64, 116)
    assert failure.size == (64, 116)
    assert success.getpixel((1, 1)) == (0, 220, 80)
    assert failure.getpixel((1, 1)) == (255, 40, 40)
