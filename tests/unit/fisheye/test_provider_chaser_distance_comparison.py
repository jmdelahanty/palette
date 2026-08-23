from __future__ import annotations

import argparse

import numpy as np
import pytest

from fisheye.analysis.provider_chaser_distance_comparison import (
    ProviderChaserDistanceComparisonError,
    _interval_masks,
    _parse_thresholds,
    _require_repeated_frame_field,
)


class _Interval:
    def __init__(self, start: int, end: int):
        self.start_frame = start
        self.end_frame = end


class _Selection:
    def __init__(self, intervals):
        self.intervals = intervals


def test_repeated_frame_field_returns_one_value_per_frame() -> None:
    values = np.asarray([[1, 1], [4, 4], [8, 8]], dtype=np.int64)
    observed = _require_repeated_frame_field(values, name="frame")
    np.testing.assert_array_equal(observed, np.asarray([1, 4, 8]))


def test_repeated_frame_field_rejects_chaser_dependent_values() -> None:
    values = np.asarray([[1, 1], [4, 5]], dtype=np.int64)
    with pytest.raises(
        ProviderChaserDistanceComparisonError,
        match="differs across chaser rows",
    ):
        _require_repeated_frame_field(values, name="frame")


def test_interval_masks_preserve_gaps_and_reject_overlap() -> None:
    frame_ids = np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64)
    masks = _interval_masks(
        frame_ids,
        _Selection([_Interval(0, 2), _Interval(4, 6)]),
    )
    np.testing.assert_array_equal(masks[0][1], [True, True, False, False, False, False])
    np.testing.assert_array_equal(masks[1][1], [False, False, False, False, True, True])

    with pytest.raises(ProviderChaserDistanceComparisonError, match="overlap"):
        _interval_masks(
            frame_ids,
            _Selection([_Interval(0, 4), _Interval(3, 6)]),
        )


def test_threshold_parser_is_numeric_and_nonempty() -> None:
    assert _parse_thresholds("0,2.5,10") == (0.0, 2.5, 10.0)
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_thresholds("not-a-number")
