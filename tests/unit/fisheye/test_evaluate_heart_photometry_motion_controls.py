from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest


_PLAYGROUND = (
    Path(__file__).resolve().parents[3] / "playgrounds" / "heartrate_stabilization"
)
sys.path.insert(0, str(_PLAYGROUND))

import evaluate_heart_photometry_motion_controls as runner  # noqa: E402


def _ok_row(
    *,
    source: str,
    region: str,
    transform: str,
    spectral_ratio: float,
    searched_ratio: float | None = None,
    searched_frequency_hz: float = 3.1,
) -> dict[str, object]:
    searched = spectral_ratio if searched_ratio is None else float(searched_ratio)
    return {
        "window_index": 0,
        "source": source,
        "region": region,
        "transform": transform,
        "status": "ok",
        "spectral_ratio": spectral_ratio,
        "frozen_frequency_spectral_ratio": spectral_ratio,
        "external_control_ratio": 1.5,
        "searched_best_frequency_hz": searched_frequency_hz,
        "searched_best_cycles_per_min": 60.0 * searched_frequency_hz,
        "searched_best_spectral_ratio": searched,
        "searched_external_control_ratio_at_best": 1.25,
        "searched_best_frequency_at_boundary": False,
        "searched_frequency_search_complete": (
            transform != "crossfit_matched_spatial_projection"
        ),
        "searched_adaptive_peak_claim_eligible": False,
        "searched_claim_requires_paired_support": True,
        "frozen_frequency_peak_claim_eligible": False,
        "source_step_spearman_r": 0.1,
        "gradient_displacement_spearman_r": 0.2,
        "transform_uncertainty_spearman_r": 0.3,
        "heldout_block_count": 3,
    }


def _paired_row(
    *,
    control_source: str,
    region: str,
    transform: str,
    observed_ratio: float,
    control_ratio: float,
    block_count: int = 4,
    block_fraction: float = 1.0,
) -> dict[str, object]:
    eligible = transform != "crossfit_matched_spatial_projection"
    return {
        "window_index": 0,
        "control_source": control_source,
        "region": region,
        "transform": transform,
        "status": "ok",
        "paired_observed_logical_block_count": 4,
        "paired_block_count": block_count,
        "paired_block_fraction": block_fraction,
        "paired_row_count": 1200,
        "paired_observed_row_count": 1200,
        "paired_row_fraction": 1.0,
        "minimum_paired_block_count": 4,
        "minimum_paired_block_fraction": 0.5,
        "paired_support_gate_passed": True,
        "paired_observed_frozen_spectral_ratio": observed_ratio,
        "paired_control_frozen_spectral_ratio": control_ratio,
        "paired_observed_to_control_frozen_ratio": observed_ratio / control_ratio,
        "paired_observed_searched_best_frequency_hz": 3.1,
        "paired_control_searched_best_frequency_hz": 2.8,
        "paired_observed_searched_best_spectral_ratio": observed_ratio,
        "paired_control_searched_best_spectral_ratio": control_ratio,
        "paired_observed_to_control_searched_max_ratio": (
            observed_ratio / control_ratio
        ),
        "paired_observed_best_frequency_at_boundary": False,
        "paired_control_best_frequency_at_boundary": False,
        "paired_frequency_interior_gate_passed": True,
        "paired_adaptive_peak_claim_eligible": eligible,
        "paired_adaptive_peak_eligibility_reason": (
            "eligible"
            if eligible
            else "matched_projection_requires_frequency_grid_refit"
        ),
        "paired_frozen_frequency_peak_claim_eligible": False,
    }


def test_region_masks_use_four_connected_erosion() -> None:
    yy, xx = np.mgrid[0:9, 0:9]
    dataset = SimpleNamespace(
        pixel_xy=np.column_stack([xx.ravel(), yy.ravel()]),
        pixel_count=81,
    )
    mask = np.zeros((9, 9), dtype=bool)
    mask[2:7, 2:7] = True

    regions = runner._region_masks(mask, dataset, erosion_iterations=1)

    assert np.count_nonzero(regions["full_mask"]) == 25
    assert np.count_nonzero(regions["boundary"]) == 16
    assert np.count_nonzero(regions["eroded_interior"]) == 9
    assert not np.any(regions["boundary"] & regions["eroded_interior"])


def test_comparison_rows_report_observed_motion_and_boundary_ratios() -> None:
    transform = runner._TRANSFORM_NAMES[0]
    rows = [
        _ok_row(
            source="observed",
            region="full_mask",
            transform=transform,
            spectral_ratio=4.0,
        ),
        _ok_row(
            source="gradient_displacement_control",
            region="full_mask",
            transform=transform,
            spectral_ratio=2.0,
        ),
        _ok_row(
            source="static_reference_control",
            region="full_mask",
            transform=transform,
            spectral_ratio=1.0,
        ),
        _ok_row(
            source="observed",
            region="boundary",
            transform=transform,
            spectral_ratio=3.0,
        ),
        _ok_row(
            source="observed",
            region="eroded_interior",
            transform=transform,
            spectral_ratio=1.5,
        ),
    ]

    paired = [
        _paired_row(
            control_source="gradient_displacement_control",
            region="full_mask",
            transform=transform,
            observed_ratio=4.0,
            control_ratio=2.0,
        ),
        _paired_row(
            control_source="static_reference_control",
            region="full_mask",
            transform=transform,
            observed_ratio=4.0,
            control_ratio=1.0,
        ),
    ]
    comparisons = runner._comparison_rows(rows, paired)
    observed = next(
        row
        for row in comparisons
        if row["comparison"] == "observed_vs_motion_controls"
        and row["region"] == "full_mask"
        and row["transform"] == transform
    )
    boundary = next(
        row
        for row in comparisons
        if row["comparison"] == "boundary_vs_eroded_interior"
        and row["source"] == "observed"
        and row["transform"] == transform
    )

    assert observed["observed_to_gradient_ratio"] == 2.0
    assert observed["observed_to_static_ratio"] == 4.0
    assert observed["observed_to_gradient_searched_max_ratio"] == 2.0
    assert observed["observed_to_gradient_searched_claim_eligible"] is True
    assert observed["frozen_frequency_peak_claim_eligible"] is False
    assert boundary["boundary_to_interior_ratio"] == 2.0


def test_array_output_is_numeric_and_pickle_free(tmp_path: Path) -> None:
    transform = runner._TRANSFORM_NAMES[0]
    rows = [
        _ok_row(
            source="observed",
            region="full_mask",
            transform=transform,
            spectral_ratio=2.5,
        )
    ]
    comparisons = runner._comparison_rows(rows)
    path = tmp_path / "motion_controls.npz"

    runner._write_arrays(path, rows, comparisons)

    with np.load(path, allow_pickle=False) as data:
        assert data["spectral_ratio"].shape == (1, 3, 3, 5)
        assert float(data["spectral_ratio"][0, 0, 0, 0]) == 2.5
        assert data["source_names"].dtype.kind == "U"
        assert (
            data["gradient_paired_adaptive_peak_eligibility_reason"].dtype.kind
            == "U"
        )
        assert data["interpretation"].item() == runner._INTERPRETATION


def test_independent_frequency_search_finds_control_peak_away_from_frozen_frequency() -> None:
    timestamps = np.arange(1600, dtype=np.float64) / 100.0
    control_peak_hz = 2.5
    target = np.sin(2.0 * np.pi * control_peak_hz * timestamps)
    traces = runner.photometry.TraceSet(
        target=target,
        upper=np.full_like(target, np.nan),
        lower=np.full_like(target, np.nan),
        control=np.sin(2.0 * np.pi * 3.7 * timestamps),
    )
    dataset = SimpleNamespace(
        timestamps_s=timestamps,
        frame_valid=np.ones(timestamps.size, dtype=bool),
    )

    searched = runner._searched_frequency_metrics(
        dataset,
        traces,
        frequency_min_hz=2.0,
        frequency_max_hz=4.0,
        frequency_step_hz=0.05,
        block_seconds=4.0,
        min_block_seconds=2.0,
        min_valid_fraction=0.7,
        max_interpolated_gap_seconds=0.02,
    )

    assert searched["searched_best_frequency_hz"] == pytest.approx(control_peak_hz)
    assert searched["searched_best_spectral_ratio"] > 10.0
    assert not searched["searched_best_frequency_at_boundary"]


def test_comparison_uses_each_sources_own_max_and_excludes_frozen_matched_projection() -> None:
    scalar = "regional_spatial_std"
    matched = "crossfit_matched_spatial_projection"
    rows = [
        _ok_row(
            source="observed",
            region="full_mask",
            transform=scalar,
            spectral_ratio=4.0,
            searched_ratio=4.0,
            searched_frequency_hz=3.2,
        ),
        _ok_row(
            source="gradient_displacement_control",
            region="full_mask",
            transform=scalar,
            spectral_ratio=0.5,
            searched_ratio=5.0,
            searched_frequency_hz=2.5,
        ),
        _ok_row(
            source="observed",
            region="full_mask",
            transform=matched,
            spectral_ratio=4.0,
            searched_ratio=4.5,
        ),
        _ok_row(
            source="gradient_displacement_control",
            region="full_mask",
            transform=matched,
            spectral_ratio=0.5,
            searched_ratio=1.0,
        ),
    ]

    paired = [
        _paired_row(
            control_source="gradient_displacement_control",
            region="full_mask",
            transform=scalar,
            observed_ratio=4.0,
            control_ratio=5.0,
        ),
        _paired_row(
            control_source="gradient_displacement_control",
            region="full_mask",
            transform=matched,
            observed_ratio=4.5,
            control_ratio=1.0,
        ),
    ]
    comparisons = runner._comparison_rows(rows, paired)
    scalar_row = next(
        row
        for row in comparisons
        if row["comparison"] == "observed_vs_motion_controls"
        and row["region"] == "full_mask"
        and row["transform"] == scalar
    )
    matched_row = next(
        row
        for row in comparisons
        if row["comparison"] == "observed_vs_motion_controls"
        and row["region"] == "full_mask"
        and row["transform"] == matched
    )

    assert scalar_row["observed_to_gradient_ratio"] == 8.0
    assert scalar_row["observed_to_gradient_searched_max_ratio"] == 0.8
    assert scalar_row["observed_to_gradient_searched_claim_eligible"] is True
    assert math.isnan(matched_row["observed_to_gradient_searched_max_ratio"])
    assert matched_row["observed_to_gradient_searched_claim_eligible"] is False


def test_frequency_grid_rejects_search_at_or_above_timestamp_nyquist() -> None:
    timestamps = np.arange(100, dtype=np.float64) / 10.0

    with pytest.raises(ValueError, match="below timestamp Nyquist"):
        runner._validated_frequency_grid(
            timestamps,
            frequency_min_hz=2.0,
            frequency_max_hz=5.0,
            frequency_step_hz=0.05,
        )


def test_paired_support_prevents_nonstationary_missing_control_from_favorable_claim() -> None:
    timestamps = np.arange(1600, dtype=np.float64) / 100.0
    rng = np.random.default_rng(19)
    noise = rng.normal(0.0, 1.0, timestamps.size)
    oscillation = np.sin(2.0 * np.pi * 3.0 * timestamps)
    observed_target = noise + np.where(np.arange(timestamps.size) < 800, 0.2, 8.0) * oscillation
    control_target = observed_target.copy()
    control_target[800:] = np.nan
    unavailable = np.full(timestamps.shape, np.nan, dtype=np.float64)
    observed = runner.photometry.TraceSet(
        target=observed_target,
        upper=unavailable.copy(),
        lower=unavailable.copy(),
        control=noise.copy(),
    )
    control = runner.photometry.TraceSet(
        target=control_target,
        upper=unavailable.copy(),
        lower=unavailable.copy(),
        control=noise.copy(),
    )
    dataset = SimpleNamespace(
        timestamps_s=timestamps,
        frame_valid=np.ones(timestamps.size, dtype=bool),
    )
    search_kwargs = {
        "frequency_min_hz": 2.0,
        "frequency_max_hz": 4.0,
        "frequency_step_hz": 0.05,
        "block_seconds": 4.0,
        "min_block_seconds": 2.0,
        "min_valid_fraction": 0.7,
        "max_interpolated_gap_seconds": 0.02,
    }

    unpaired_observed = runner._searched_frequency_metrics(
        dataset,
        observed,
        **search_kwargs,
    )
    unpaired_control = runner._searched_frequency_metrics(
        dataset,
        control,
        **search_kwargs,
    )
    paired = runner._paired_support_metrics(
        dataset,
        observed,
        dataset,
        control,
        frozen_frequency_hz=3.0,
        minimum_paired_block_count=2,
        minimum_paired_block_fraction=0.5,
        **search_kwargs,
    )

    unpaired_ratio = (
        unpaired_observed["searched_best_spectral_ratio"]
        / unpaired_control["searched_best_spectral_ratio"]
    )
    assert unpaired_ratio > 2.0
    assert paired["paired_block_count"] == 2
    assert paired["paired_block_fraction"] == 0.5
    assert paired["paired_support_gate_passed"] is True
    assert paired["paired_observed_to_control_searched_max_ratio"] == pytest.approx(1.0)

    sparse_control = runner.photometry.TraceSet(
        target=np.where(np.arange(timestamps.size) < 400, control_target, np.nan),
        upper=unavailable.copy(),
        lower=unavailable.copy(),
        control=noise.copy(),
    )
    sparse = runner._paired_support_metrics(
        dataset,
        observed,
        dataset,
        sparse_control,
        frozen_frequency_hz=3.0,
        minimum_paired_block_count=2,
        minimum_paired_block_fraction=0.5,
        **search_kwargs,
    )
    assert sparse["paired_block_count"] == 1
    assert sparse["paired_block_fraction"] == 0.25
    assert sparse["paired_support_gate_passed"] is False

    failed_gate_record = _paired_row(
        control_source="gradient_displacement_control",
        region="full_mask",
        transform="regional_spatial_std",
        observed_ratio=4.0,
        control_ratio=1.0,
        block_count=1,
        block_fraction=0.25,
    )
    failed_gate_record["paired_support_gate_passed"] = False
    # Even inconsistent upstream eligibility metadata cannot bypass the gate.
    failed_gate_record["paired_adaptive_peak_claim_eligible"] = True
    comparison = runner._comparison_rows(
        [
            _ok_row(
                source="observed",
                region="full_mask",
                transform="regional_spatial_std",
                spectral_ratio=4.0,
            ),
            _ok_row(
                source="gradient_displacement_control",
                region="full_mask",
                transform="regional_spatial_std",
                spectral_ratio=1.0,
            ),
        ],
        [failed_gate_record],
    )
    comparison_row = next(
        row
        for row in comparison
        if row["comparison"] == "observed_vs_motion_controls"
        and row["region"] == "full_mask"
        and row["transform"] == "regional_spatial_std"
    )
    assert comparison_row["observed_to_gradient_searched_claim_eligible"] is False
    assert math.isnan(comparison_row["observed_to_gradient_searched_max_ratio"])


@pytest.mark.parametrize(
    ("observed_hz", "control_hz", "expected_reason"),
    [
        (2.0, 3.0, "observed_maximum_at_search_boundary"),
        (3.0, 4.0, "control_maximum_at_search_boundary"),
    ],
)
def test_paired_boundary_maximum_is_descriptive_but_claim_ineligible(
    observed_hz: float,
    control_hz: float,
    expected_reason: str,
) -> None:
    timestamps = np.arange(1600, dtype=np.float64) / 100.0
    unavailable = np.full(timestamps.shape, np.nan, dtype=np.float64)

    def traces(frequency_hz: float) -> runner.photometry.TraceSet:
        target = np.sin(2.0 * np.pi * frequency_hz * timestamps)
        return runner.photometry.TraceSet(
            target=target,
            upper=unavailable.copy(),
            lower=unavailable.copy(),
            control=np.sin(2.0 * np.pi * 3.5 * timestamps),
        )

    dataset = SimpleNamespace(
        timestamps_s=timestamps,
        frame_valid=np.ones(timestamps.size, dtype=bool),
    )
    paired = runner._paired_support_metrics(
        dataset,
        traces(observed_hz),
        dataset,
        traces(control_hz),
        frozen_frequency_hz=3.0,
        frequency_min_hz=2.0,
        frequency_max_hz=4.0,
        frequency_step_hz=0.05,
        block_seconds=4.0,
        min_block_seconds=2.0,
        min_valid_fraction=0.7,
        max_interpolated_gap_seconds=0.02,
        minimum_paired_block_count=4,
        minimum_paired_block_fraction=1.0,
    )
    eligible, reason = runner._paired_claim_decision(
        paired,
        transform_name="regional_spatial_std",
    )

    assert paired["paired_support_gate_passed"] is True
    assert paired["paired_frequency_interior_gate_passed"] is False
    assert paired["paired_observed_to_control_searched_max_ratio"] > 0.0
    assert eligible is False
    assert reason == expected_reason
