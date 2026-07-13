from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np


_PLAYGROUND = (
    Path(__file__).resolve().parents[3] / "playgrounds" / "heartrate_stabilization"
)
sys.path.insert(0, str(_PLAYGROUND))

import compare_heart_photometry_transforms as runner  # noqa: E402


def _summary_row(
    candidate: str,
    role: str,
    *,
    spectral_ratio: float,
    control_ratio: float,
    phase_locking_value: float = 0.7,
    tracking_spearman_r: float = 0.1,
) -> dict[str, object]:
    return {
        "candidate": candidate,
        "outer_role": role,
        "status": "ok",
        "spectral_ratio": spectral_ratio,
        "control_ratio": control_ratio,
        "phase_locking_value": phase_locking_value,
        "tracking_spearman_r": tracking_spearman_r,
    }


def test_candidate_summary_allows_no_winner() -> None:
    names = ["baseline", "derivative"]
    rows = [
        _summary_row(
            "baseline", "discovery", spectral_ratio=1.2, control_ratio=1.3
        ),
        _summary_row(
            "derivative", "discovery", spectral_ratio=2.0, control_ratio=0.8
        ),
        _summary_row(
            "baseline", "confirmation", spectral_ratio=50.0, control_ratio=50.0
        ),
        _summary_row(
            "derivative", "confirmation", spectral_ratio=50.0, control_ratio=50.0
        ),
    ]

    summaries, winner = runner._candidate_summary(
        rows,
        names,
        min_discovery_windows=1,
        min_discovery_spectral_ratio=1.5,
        min_discovery_control_ratio=1.1,
    )

    assert winner is None
    assert not summaries["baseline"]["passes_descriptive_discovery_gate"]
    assert not summaries["derivative"]["passes_descriptive_discovery_gate"]


def test_candidate_selection_uses_discovery_rows_only() -> None:
    names = ["baseline", "projection"]
    discovery = [
        _summary_row(
            "baseline", "discovery", spectral_ratio=4.0, control_ratio=2.0
        ),
        _summary_row(
            "projection", "discovery", spectral_ratio=2.0, control_ratio=1.5
        ),
    ]
    first_confirmation = [
        _summary_row(
            "baseline", "confirmation", spectral_ratio=0.01, control_ratio=0.01
        ),
        _summary_row(
            "projection", "confirmation", spectral_ratio=1e6, control_ratio=1e6
        ),
    ]
    reversed_confirmation = [
        _summary_row(
            "baseline", "confirmation", spectral_ratio=1e9, control_ratio=1e9
        ),
        _summary_row(
            "projection", "confirmation", spectral_ratio=1e-9, control_ratio=1e-9
        ),
    ]
    kwargs = {
        "min_discovery_windows": 1,
        "min_discovery_spectral_ratio": 1.0,
        "min_discovery_control_ratio": 1.0,
    }

    first_summaries, first_winner = runner._candidate_summary(
        discovery + first_confirmation, names, **kwargs
    )
    second_summaries, second_winner = runner._candidate_summary(
        discovery + reversed_confirmation, names, **kwargs
    )

    assert first_winner == second_winner == "baseline"
    assert (
        first_summaries["baseline"]["discovery_selection_score"]
        == second_summaries["baseline"]["discovery_selection_score"]
    )
    assert (
        first_summaries["projection"]["discovery_selection_score"]
        == second_summaries["projection"]["discovery_selection_score"]
    )
    assert (
        first_summaries["baseline"]["confirmation_display_only"]
        != second_summaries["baseline"]["confirmation_display_only"]
    )


def test_logical_blocks_split_long_invalid_gap_without_reusing_rows() -> None:
    timestamps = np.arange(1200, dtype=np.float64) / 100.0
    frame_valid = np.ones(1200, dtype=bool)
    frame_valid[260:340] = False
    target = np.sin(2.0 * np.pi * 3.0 * timestamps)
    target[~frame_valid] = np.nan
    traces = runner.TraceSet(
        target=target,
        upper=target.copy(),
        lower=target.copy(),
        control=target.copy(),
    )
    dataset = SimpleNamespace(timestamps_s=timestamps, frame_valid=frame_valid)

    blocks = runner._logical_blocks(
        dataset,
        traces,
        block_seconds=4.0,
        min_block_seconds=0.5,
        min_valid_fraction=0.7,
        max_interpolated_gap_seconds=0.02,
    )

    assert len(blocks) == 4
    assert all(np.all(frame_valid[rows]) for rows in blocks)
    assert all(not (np.any(rows < 260) and np.any(rows >= 340)) for rows in blocks)
    concatenated = np.concatenate(blocks)
    assert np.unique(concatenated).size == concatenated.size
    assert not np.any(np.isin(np.arange(260, 340), concatenated))


def test_candidate_family_builds_predeclared_transforms_on_small_dataset() -> None:
    timestamps = np.arange(300, dtype=np.float64) / 100.0
    pixel_xy = np.column_stack(
        [np.arange(12, dtype=np.float64), np.zeros(12, dtype=np.float64)]
    )
    phase = 2.0 * np.pi * 3.1 * timestamps[:, None]
    pixel_scale = np.linspace(0.7, 1.3, 12)[None, :]
    values = 100.0 + pixel_scale * np.sin(phase + np.arange(12)[None, :] * 0.1)
    dataset = SimpleNamespace(
        traces=values,
        pixel_valid=np.ones(values.shape, dtype=bool),
        frame_valid=np.ones(timestamps.size, dtype=bool),
        pixel_xy=pixel_xy,
        timestamps_s=timestamps,
    )
    target = np.arange(12) < 6
    upper = np.arange(12) < 3
    lower = (np.arange(12) >= 3) & (np.arange(12) < 6)
    reference = (np.arange(12) >= 6) & (np.arange(12) < 9)
    control = np.arange(12) >= 9

    candidates = runner._candidate_traces(
        dataset,
        target=target,
        upper=upper,
        lower=lower,
        reference=reference,
        control=control,
        sg_windows=(5, 7, 11),
        lag_frames=(8, 12, 16),
        gaussian_sigma_px=0.8,
    )

    assert list(candidates) == [
        "baseline_mean_intensity",
        "robust_huber_intensity",
        "reference_log_ratio",
        "reference_fractional_difference",
        "masked_gaussian_huber_sigma0.8",
        "regional_spatial_std",
        "huber_savgol_derivative_w5",
        "huber_savgol_derivative_w7",
        "huber_savgol_derivative_w11",
        "gaussian_savgol_derivative_w7_sigma0.8",
        "huber_normalized_signed_lag8",
        "huber_normalized_signed_lag12",
        "huber_normalized_signed_lag16",
    ]
    assert all(trace.target.shape == timestamps.shape for trace in candidates.values())
    assert all(np.any(np.isfinite(trace.target)) for trace in candidates.values())
