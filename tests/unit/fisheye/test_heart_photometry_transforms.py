from __future__ import annotations

import numpy as np
import pytest

from fisheye.analysis.heart_photometry_transforms import (
    masked_gaussian_smooth,
    normalized_signed_lag_difference,
    reference_normalize,
    regional_pool,
    regional_spatial_std,
    segmented_savgol_derivative,
)


def test_regional_pool_rejects_outliers_and_honors_validity() -> None:
    values = np.asarray(
        [
            [10.0, 11.0, 12.0, 1000.0, -50.0],
            [20.0, 21.0, 22.0, 2000.0, -50.0],
            [30.0, 31.0, np.nan, 3000.0, -50.0],
        ]
    )
    region = np.asarray([True, True, True, True, False])
    valid = np.isfinite(values)
    valid[2, 1] = False

    mean = regional_pool(values, region, valid=valid, method="mean")
    trimmed = regional_pool(
        values,
        region,
        valid=valid,
        method="trimmed_mean",
        trim_fraction=0.25,
        min_valid_pixels=2,
    )
    huber = regional_pool(values, region, valid=valid, method="huber")
    insufficient = regional_pool(
        values,
        region,
        valid=valid,
        method="median",
        min_valid_pixels=3,
    )

    assert mean[0] > 250.0
    assert trimmed[:2] == pytest.approx([11.5, 21.5])
    assert trimmed[2] == pytest.approx((30.0 + 3000.0) / 2.0)
    assert huber[0] < 20.0
    assert np.isnan(insufficient[2])


def test_masked_gaussian_smooth_has_no_cross_mask_leakage() -> None:
    pixel_xy = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    region = np.asarray([True, True, False, False])
    values = np.asarray(
        [
            [0.0, 10.0, 1000.0, 2000.0],
            [10.0, 20.0, -1000.0, -2000.0],
            [5.0, 15.0, 1e9, 1e9],
        ]
    )
    changed_outside = values.copy()
    changed_outside[:, 2:] *= -12345.0
    valid = np.ones(values.shape, dtype=bool)
    valid[2, :2] = False

    smoothed = masked_gaussian_smooth(
        values,
        pixel_xy,
        region,
        sigma_px=1.0,
        valid=valid,
    )
    changed = masked_gaussian_smooth(
        changed_outside,
        pixel_xy,
        region,
        sigma_px=1.0,
        valid=valid,
    )

    assert np.allclose(smoothed[:, :2], changed[:, :2], equal_nan=True)
    assert np.isnan(smoothed[:, 2:]).all()
    assert 0.0 < smoothed[0, 0] < 10.0
    assert 0.0 < smoothed[0, 1] < 10.0
    assert np.isnan(smoothed[2, :]).all()


def test_reference_normalizations_match_fixed_formulas() -> None:
    signal_values = np.asarray([100.0, 120.0, 0.0, -2.0])
    reference_values = np.asarray([100.0, 100.0, 0.0, 1.0])
    valid = np.asarray([True, True, False, True])

    log_ratio = reference_normalize(
        signal_values,
        reference_values,
        mode="log_ratio",
        valid=valid,
        epsilon=1e-3,
    )
    fractional = reference_normalize(
        signal_values,
        reference_values,
        mode="fractional_difference",
        valid=valid,
        epsilon=1e-3,
    )

    assert log_ratio[0] == pytest.approx(0.0)
    assert log_ratio[1] == pytest.approx(np.log(120.001) - np.log(100.001))
    assert np.isnan(log_ratio[2])
    assert np.isnan(log_ratio[3])
    assert fractional[0] == pytest.approx(0.0)
    assert fractional[1] == pytest.approx(40.0 / 220.001)
    assert np.isnan(fractional[2])
    assert fractional[3] == pytest.approx(-6.0 / -0.999)


def test_segmented_savgol_derivative_is_signed_and_does_not_bridge_invalid_gap() -> None:
    timestamps = np.arange(40, dtype=np.float64) * 0.1
    values = timestamps**2
    valid = np.ones(40, dtype=bool)
    valid[17:22] = False

    derivative = segmented_savgol_derivative(
        values,
        timestamps,
        valid=valid,
        window_length=5,
        polyorder=2,
    )

    assert derivative[5:15] == pytest.approx(2.0 * timestamps[5:15], abs=1e-10)
    assert derivative[24:35] == pytest.approx(2.0 * timestamps[24:35], abs=1e-10)
    assert np.isnan(derivative[15:24]).all()


def test_segmented_savgol_uses_actual_irregular_timestamps_and_timestamp_gaps() -> None:
    steps = np.resize(np.asarray([0.08, 0.12, 0.09, 0.11]), 35)
    timestamps = np.concatenate([[0.0], np.cumsum(steps)])
    timestamps[20:] += 4.0
    values = 3.0 * timestamps**2 - 2.0 * timestamps + 7.0

    derivative = segmented_savgol_derivative(
        values,
        timestamps,
        window_length=5,
        polyorder=2,
        max_gap_factor=1.75,
    )

    expected = 6.0 * timestamps - 2.0
    assert derivative[4:18] == pytest.approx(expected[4:18], abs=1e-9)
    assert derivative[22:33] == pytest.approx(expected[22:33], abs=1e-9)
    assert np.isnan(derivative[18:22]).all()


def test_centered_lag_difference_stays_inside_contiguous_valid_segments() -> None:
    timestamps = np.arange(20, dtype=np.float64) * 0.01
    values = 100.0 + np.arange(20, dtype=np.float64)
    valid = np.ones(20, dtype=bool)
    valid[8:11] = False

    difference = normalized_signed_lag_difference(
        values,
        timestamps,
        lag_frames=4,
        valid=valid,
        alignment="center",
        epsilon=1e-6,
    )

    expected_at_four = 2.0 * (values[6] - values[2]) / (values[6] + values[2] + 1e-6)
    assert difference[4] == pytest.approx(expected_at_four)
    assert np.isnan(difference[:2]).all()
    assert np.isnan(difference[6:13]).all()
    assert np.isfinite(difference[13:18]).all()
    assert np.isnan(difference[18:]).all()


def test_regional_spatial_std_uses_only_valid_selected_pixels() -> None:
    values = np.asarray(
        [
            [1.0, 2.0, 3.0, 100.0],
            [4.0, 6.0, 8.0, -100.0],
            [5.0, 7.0, 9.0, 0.0],
        ]
    )
    region = np.asarray([True, True, True, False])
    valid = np.ones(values.shape, dtype=bool)
    valid[1, 1] = False
    valid[2, :2] = False

    result = regional_spatial_std(
        values,
        region,
        valid=valid,
        ddof=1,
        min_valid_pixels=2,
    )

    assert result[0] == pytest.approx(1.0)
    assert result[1] == pytest.approx(np.std([4.0, 8.0], ddof=1))
    assert np.isnan(result[2])


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (
            lambda: regional_pool(np.ones((3, 2)), np.ones(3, dtype=bool)),
            "region_mask shape",
        ),
        (
            lambda: segmented_savgol_derivative(
                np.ones(10), np.arange(10), window_length=4
            ),
            "odd integer",
        ),
        (
            lambda: normalized_signed_lag_difference(
                np.ones(10), np.arange(10), lag_frames=3, alignment="center"
            ),
            "even lag_frames",
        ),
    ],
)
def test_transform_validation_rejects_ambiguous_inputs(call: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        call()  # type: ignore[operator]
