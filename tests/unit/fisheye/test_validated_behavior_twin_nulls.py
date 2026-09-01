"""Unit tests for rotated virtual-twin null computation on synthetic frames."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fisheye.analytics_exports.validated_behavior_twin_nulls import (
    ROTATION_DEGREES,
    TwinChaserTrack,
    TwinEpochWindow,
    TwinNullError,
    compute_twin_excess,
    compute_twin_rows_for_provider,
    fit_arena_center_mm,
    hysteresis_entry_stats,
    rotate_points_about_center,
    summarize_distances,
)


class TestRotationMath:
    def test_rotation_set_matches_legacy_and_excludes_90(self) -> None:
        assert ROTATION_DEGREES == (0, 60, 120, 180, 240, 300)
        assert 90 not in ROTATION_DEGREES

    def test_distance_from_center_is_preserved(self) -> None:
        rng = np.random.default_rng(7)
        points = rng.uniform(-50.0, 50.0, size=(200, 2))
        center = np.array([12.5, -3.0])
        before = np.hypot(*(points - center).T)
        for degrees in ROTATION_DEGREES:
            rotated = rotate_points_about_center(points, center, degrees)
            after = np.hypot(*(rotated - center).T)
            np.testing.assert_allclose(after, before, atol=1e-9)

    def test_rotation_zero_is_identity_and_180_reflects(self) -> None:
        points = np.array([[10.0, 4.0], [-2.0, 8.0]])
        center = np.array([1.0, 2.0])
        np.testing.assert_allclose(
            rotate_points_about_center(points, center, 0), points
        )
        np.testing.assert_allclose(
            rotate_points_about_center(points, center, 180),
            2 * center[None, :] - points,
            atol=1e-12,
        )

    def test_60_degree_rotation_matches_complex_arithmetic(self) -> None:
        point = np.array([[7.0, -5.0]])
        center = np.array([2.0, 3.0])
        rotated = rotate_points_about_center(point, center, 60)
        expected = (
            complex(2.0, 3.0)
            + (complex(7.0, -5.0) - complex(2.0, 3.0))
            * np.exp(1j * math.radians(60))
        )
        np.testing.assert_allclose(rotated[0], [expected.real, expected.imag])

    def test_nan_points_propagate(self) -> None:
        points = np.array([[np.nan, 1.0]])
        rotated = rotate_points_about_center(points, np.zeros(2), 60)
        assert np.isnan(rotated).any()

    def test_nonfinite_center_is_rejected(self) -> None:
        with pytest.raises(TwinNullError):
            rotate_points_about_center(
                np.zeros((1, 2)), np.array([np.nan, 0.0]), 60
            )


class TestCenterFit:
    def test_recovers_known_center_from_mean_radius_constraints(self) -> None:
        rng = np.random.default_rng(11)
        true_center = np.array([61.0, 44.5])
        angles = rng.uniform(0, 2 * math.pi, size=3000)
        radii = 40.0 * np.sqrt(rng.uniform(0, 1, size=3000))
        points = true_center + np.column_stack(
            [radii * np.cos(angles), radii * np.sin(angles)]
        )
        masks = [np.zeros(3000, dtype=bool) for _ in range(3)]
        for index, mask in enumerate(masks):
            mask[index * 1000 : (index + 1) * 1000] = True
        constraints = [
            (
                mask,
                float(np.hypot(*(points[mask] - true_center).T).mean()),
            )
            for mask in masks
        ]
        fit = fit_arena_center_mm(points, constraints)
        np.testing.assert_allclose(fit.center_mm, true_center, atol=1e-6)
        assert fit.max_abs_residual_mm < 1e-6

    def test_requires_three_constraints(self) -> None:
        points = np.zeros((10, 2))
        mask = np.ones(10, dtype=bool)
        with pytest.raises(TwinNullError):
            fit_arena_center_mm(points, [(mask, 1.0), (mask, 1.0)])


class TestSummarizeDistances:
    def test_counts_quantiles_and_near_fraction(self) -> None:
        distance = np.array([1.0, 2.0, 5.0, 8.0, 100.0, np.nan])
        valid = np.array([True, True, True, True, False, False])
        row = summarize_distances(distance, valid, near_zone_radius_mm=5.0)
        assert row["valid_distance_frame_count"] == 4
        # near zone is inclusive: 1, 2, and exactly 5 are near.
        assert row["near_zone_frame_count"] == 3
        assert row["near_zone_fraction_valid"] == pytest.approx(0.75)
        assert row["distance_mean_mm"] == pytest.approx(4.0)
        assert row["distance_p50_mm"] == pytest.approx(
            float(np.percentile([1.0, 2.0, 5.0, 8.0], 50))
        )

    def test_empty_valid_set_yields_none(self) -> None:
        row = summarize_distances(
            np.array([1.0]), np.array([False]), near_zone_radius_mm=5.0
        )
        assert row["valid_distance_frame_count"] == 0
        assert row["near_zone_fraction_valid"] is None
        assert row["distance_p50_mm"] is None


def _stats(distance, valid, *, enter=5.0, exit_=6.0, frame=None, ts=None):
    distance = np.asarray(distance, dtype=np.float64)
    n = distance.size
    if frame is None:
        frame = np.arange(n, dtype=np.int64)
    if ts is None:
        ts = np.arange(n, dtype=np.int64) * 10_000_000
    return hysteresis_entry_stats(
        frame_id=frame,
        timestamp_ns=ts,
        timestamp_valid=np.ones(n, dtype=bool),
        distance_mm=distance,
        distance_valid=np.asarray(valid, dtype=bool),
        enter_mm=enter,
        exit_mm=exit_,
    )


class TestHysteresisEntries:
    def test_simple_entry_counted_once(self) -> None:
        # outside -> inside -> stays -> exits -> re-enters
        distance = [10.0, 4.0, 3.0, 7.0, 4.5]
        stats = _stats(distance, [True] * 5)
        assert stats.entry_count == 2

    def test_dead_band_does_not_toggle(self) -> None:
        # 5..6 band: crossing into [5, 6] neither enters nor exits.
        distance = [10.0, 5.5, 5.9, 5.2, 10.0, 5.5, 4.0]
        stats = _stats(distance, [True] * 7)
        assert stats.entry_count == 1

    def test_start_inside_is_censored_not_entered(self) -> None:
        distance = [3.0, 2.0, 10.0, 4.0]
        stats = _stats(distance, [True] * 4)
        assert stats.entry_count == 1  # only the re-entry after going outside

    def test_gap_censors_active_visit_and_requires_new_outside(self) -> None:
        # Valid gap in the middle: fish inside before and after the gap.
        distance = [10.0, 4.0, np.nan, 3.0, 10.0, 4.0]
        valid = [True, True, False, True, True, True]
        stats = _stats(distance, valid)
        # entry 1 before the gap; inside-at-segment-start after the gap is
        # censored; final approach after an outside sample is entry 2.
        assert stats.entry_count == 2
        assert stats.invalid_gap_count == 1

    def test_frame_discontinuity_breaks_segment(self) -> None:
        frame = np.array([0, 1, 5, 6], dtype=np.int64)
        ts = frame * 10_000_000
        distance = [10.0, 4.0, 3.0, 2.0]
        stats = _stats(distance, [True] * 4, frame=frame, ts=ts)
        # inside state does not survive the frame jump: one entry only.
        assert stats.entry_count == 1

    def test_tracked_duration_sums_contiguous_intervals_only(self) -> None:
        frame = np.array([0, 1, 2, 10, 11], dtype=np.int64)
        ts = frame * 10_000_000
        stats = _stats([10.0] * 5, [True] * 5, frame=frame, ts=ts)
        assert stats.valid_tracked_duration_s == pytest.approx(0.03)

    def test_no_observed_samples(self) -> None:
        stats = _stats([np.nan, np.nan], [False, False])
        assert stats.entry_count == 0
        assert stats.valid_tracked_duration_s == 0.0
        assert math.isnan(stats.entry_rate_per_min_valid_time)


class TestTwinRowsAndExcess:
    def _make_rows(self):
        n = 240
        frame = np.arange(n, dtype=np.int64)
        ts = frame * 10_000_000
        center = np.array([50.0, 50.0])
        # Fish sits near the chaser's observed side: strong object effect.
        fish = np.tile(center + np.array([20.0, 0.0]), (n, 1))
        chaser = np.tile(center + np.array([30.0, 0.0]), (n, 1))
        track = TwinChaserTrack(
            chaser_identity_code=0,
            chaser_identity="c0",
            behavior_role="aggressive",
            chaser_xy_mm=chaser,
            valid=np.ones(n, dtype=bool),
        )
        epochs = [TwinEpochWindow(0, "chaser_training", 0, n)]
        rows = compute_twin_rows_for_provider(
            frame_id=frame,
            timestamp_ns=ts,
            timestamp_valid=np.ones(n, dtype=bool),
            fish_xy_mm=fish,
            chasers=[track],
            epochs=epochs,
            center_mm=center,
            near_zone_radius_mm=11.0,
            near_entry_radius_mm=11.0,
            near_exit_radius_mm=12.0,
        )
        for row in rows:
            row["recording_id"] = "rec"
            row["provider_role"] = "keypoint"
        return rows

    def test_row_grain_and_rotation_zero_matches_direct_distance(self) -> None:
        rows = self._make_rows()
        assert len(rows) == len(ROTATION_DEGREES)
        by_rotation = {row["rotation_deg"]: row for row in rows}
        assert by_rotation[0]["distance_p50_mm"] == pytest.approx(10.0)
        assert by_rotation[0]["near_zone_fraction_valid"] == pytest.approx(1.0)
        # 180-degree twin: chaser reflected to the far side, 50 mm away.
        assert by_rotation[180]["distance_p50_mm"] == pytest.approx(50.0)
        assert by_rotation[180]["near_zone_fraction_valid"] == pytest.approx(0.0)
        # 60-degree twin distance by the law of cosines (r1=20, r2=30).
        expected_60 = math.sqrt(20**2 + 30**2 - 2 * 20 * 30 * math.cos(math.radians(60)))
        assert by_rotation[60]["distance_p50_mm"] == pytest.approx(expected_60)

    def test_excess_is_observed_minus_twin_mean(self) -> None:
        rows = self._make_rows()
        excess = compute_twin_excess(rows)
        assert len(excess) == 1
        entry = excess[0]
        twins = [
            row for row in rows if row["rotation_deg"] != 0
        ]
        twin_mean = float(np.mean([row["distance_p50_mm"] for row in twins]))
        assert entry["distance_p50_mm_observed"] == pytest.approx(10.0)
        assert entry["distance_p50_mm_twin_mean"] == pytest.approx(twin_mean)
        assert entry["distance_p50_mm_excess"] == pytest.approx(10.0 - twin_mean)
        # Fish glued to the observed chaser: near-fraction excess is positive.
        assert entry["near_zone_fraction_valid_excess"] == pytest.approx(1.0)

    def test_excess_requires_full_rotation_set(self) -> None:
        rows = [row for row in self._make_rows() if row["rotation_deg"] != 180]
        with pytest.raises(TwinNullError):
            compute_twin_excess(rows)

    def test_excess_with_missing_metric_yields_none(self) -> None:
        rows = self._make_rows()
        for row in rows:
            if row["rotation_deg"] == 120:
                row["distance_p50_mm"] = None
        entry = compute_twin_excess(rows)[0]
        assert entry["distance_p50_mm_excess"] is None
        assert entry["near_zone_fraction_valid_excess"] is not None
