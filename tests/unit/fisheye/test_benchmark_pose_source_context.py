from __future__ import annotations

import numpy as np
import pytest

from fisheye.diagnostics.benchmark_pose_source_context import (
    derive_centered_context_coordinates,
    select_sample_indices,
    summarize_confidence_scores,
)


def test_select_sample_indices_is_deterministic_and_strictly_increasing() -> None:
    assert select_sample_indices(10, 4, mode="first").tolist() == [0, 1, 2, 3]
    even = select_sample_indices(10, 4, mode="even")
    assert even.tolist() == [1, 3, 6, 8]
    assert np.all(np.diff(even) > 0)


def test_derive_centered_context_coordinates_binds_exact_translation() -> None:
    coordinates = np.asarray([[100, 200], [0, 0]], dtype=np.int32)
    derived, transform = derive_centered_context_coordinates(
        coordinates,
        native_shape_hw=(348, 348),
        context_shape_hw=(512, 512),
    )
    assert derived.tolist() == [[18, 118], [-82, -82]]
    assert transform["context_top_left_from_native_top_left_xy"] == [-82, -82]
    assert transform["native_xy_from_context_xy"] == [-82, -82]


@pytest.mark.parametrize("context_shape", [(347, 512), (511, 512)])
def test_derive_centered_context_coordinates_rejects_unsafe_extent(
    context_shape: tuple[int, int],
) -> None:
    with pytest.raises(ValueError, match="centered"):
        derive_centered_context_coordinates(
            np.zeros((1, 2), dtype=np.int32),
            native_shape_hw=(348, 348),
            context_shape_hw=context_shape,
        )


def test_summarize_confidence_scores_uses_one_maximum_per_row() -> None:
    summary = summarize_confidence_scores(
        [None, 0.5, 0.02, 0.001],
        thresholds=[0.25, 0.01, 0.001],
    )
    assert summary["detected_at_prediction_floor"] == 3
    assert summary["count_by_threshold"] == {
        "0.25": 1,
        "0.01": 2,
        "0.001": 3,
    }
    assert summary["max_confidence"] == 0.5
