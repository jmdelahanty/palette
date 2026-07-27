from __future__ import annotations

import numpy as np
import pytest

from fisheye.diagnostics.audit_arena_geometry_detection_gates import (
    Circle,
    classify_gate_results,
    select_boundary_sentinel_rows,
    select_review_rows,
    signed_circle_distance,
)


def test_signed_circle_distance_is_positive_inside_and_inclusive_at_boundary() -> None:
    circle = Circle(center_x_px=10.0, center_y_px=20.0, radius_px=5.0)
    centers = np.asarray(
        [
            [10.0, 20.0],
            [13.0, 24.0],
            [16.0, 20.0],
        ]
    )

    distance = signed_circle_distance(centers, circle)

    np.testing.assert_allclose(distance, [5.0, 0.0, -1.0], atol=0.0, rtol=0.0)
    categories = classify_gate_results(distance, np.asarray([1.0, -1.0, 1.0]))
    assert categories.tolist() == [
        "both_inside",
        "palette_only",
        "acquisition_only",
    ]


def test_classify_gate_results_covers_all_four_categories() -> None:
    categories = classify_gate_results(
        np.asarray([1.0, -1.0, 0.0, -0.1]),
        np.asarray([1.0, -1.0, -0.1, 0.0]),
    )

    assert categories.tolist() == [
        "both_inside",
        "both_outside",
        "palette_only",
        "acquisition_only",
    ]


def test_review_selection_uses_temporal_quantiles_per_disagreement_class() -> None:
    categories = np.asarray(
        [
            "palette_only",
            "both_inside",
            "palette_only",
            "acquisition_only",
            "palette_only",
            "acquisition_only",
            "palette_only",
            "acquisition_only",
        ]
    )
    frames = np.asarray([40, 10, 10, 70, 30, 20, 20, 50])

    selected = select_review_rows(
        categories=categories,
        frame_indices=frames,
        max_per_category=2,
    )

    assert selected.tolist() == [2, 5, 0, 3]


def test_gate_helpers_fail_closed_on_invalid_shapes_or_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="shape"):
        signed_circle_distance(np.asarray([1.0, 2.0]), Circle(0.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="finite"):
        classify_gate_results(np.asarray([np.nan]), np.asarray([0.0]))
    with pytest.raises(ValueError, match="positive"):
        select_review_rows(
            categories=np.asarray(["palette_only"]),
            frame_indices=np.asarray([0]),
            max_per_category=0,
        )


def test_boundary_sentinels_choose_nearest_row_in_each_temporal_partition() -> None:
    selected = select_boundary_sentinel_rows(
        frame_indices=np.asarray([0, 10, 20, 30, 40, 50]),
        palette_signed_distance_px=np.asarray([9.0, 2.0, 8.0, 4.0, 1.0, 3.0]),
        acquisition_signed_distance_px=np.asarray([8.0, 3.0, 7.0, 5.0, 2.0, 4.0]),
        max_rows=3,
    )

    assert selected.tolist() == [1, 3, 4]
