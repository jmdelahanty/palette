from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from fisheye.analysis.local_rostral_heartrate import (
    LocalCoordinateDataset,
    autocorrelation_preserving_surrogate,
)


_BATCH_SCHEMA = "palette.heart_photometry_family_null_batch.v2"


@dataclass(frozen=True)
class PhotometryFamilyEvaluation:
    """One complete evaluation of a predeclared transform family."""

    candidate_names: tuple[str, ...]
    window_indices: np.ndarray
    discovery_windows: np.ndarray
    spectral_ratios: np.ndarray
    control_ratios: np.ndarray
    scorable: np.ndarray
    cell_statistics: np.ndarray
    discovery_selection_scores: np.ndarray
    selected_candidate_index: int
    selected_confirmation_window_count: int
    total_confirmation_window_count: int
    selected_confirmation_scorable_fraction: float
    selected_confirmation_gate_passed: bool
    selected_confirmation_statistic: float
    maximum_cell_statistic: float
    maximum_window_index: int
    maximum_candidate_index: int


@dataclass(frozen=True)
class PhotometryFamilyNullBatch:
    """Deterministic global-index null samples for the full transform family."""

    surrogate_indices: np.ndarray
    maximum_cell_statistics: np.ndarray
    selected_confirmation_statistics: np.ndarray
    selected_candidate_indices: np.ndarray
    selected_confirmation_window_counts: np.ndarray
    selected_confirmation_scorable_fractions: np.ndarray
    selected_confirmation_gate_passed: np.ndarray
    maximum_window_indices: np.ndarray
    maximum_candidate_indices: np.ndarray


def _finite_median(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.median(finite)) if finite.size else float("nan")


def _combined_statistic(
    spectral_ratios: np.ndarray,
    control_ratios: np.ndarray,
) -> np.ndarray:
    spectral = np.asarray(spectral_ratios, dtype=np.float64)
    control = np.asarray(control_ratios, dtype=np.float64)
    valid = (
        np.isfinite(spectral)
        & (spectral > 0.0)
        & np.isfinite(control)
        & (control > 0.0)
    )
    output = np.full(spectral.shape, np.nan, dtype=np.float64)
    output[valid] = np.log2(spectral[valid]) + np.log2(control[valid])
    return output


def evaluate_photometry_family(
    *,
    candidate_names: Sequence[str],
    window_indices: np.ndarray,
    discovery_windows: np.ndarray,
    spectral_ratios: np.ndarray,
    control_ratios: np.ndarray,
    scorable: np.ndarray | None = None,
    min_discovery_windows: int = 3,
    min_discovery_spectral_ratio: float = 1.5,
    min_discovery_control_ratio: float = 1.1,
    min_confirmation_windows: int = 3,
    min_confirmation_scorable_fraction: float = 0.5,
) -> PhotometryFamilyEvaluation:
    """Apply the frozen discovery gate and both family-level statistics.

    The strict statistic is the maximum combined log-ratio over every scorable
    transform/window cell. The adaptive statistic selects a transform using
    discovery windows only, then summarizes that frozen transform on confirmation
    windows. A selection index of ``-1`` is a valid no-candidate result.
    """

    names = tuple(str(name) for name in candidate_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("candidate_names must be nonempty and unique")
    windows = np.asarray(window_indices, dtype=np.int64)
    discovery = np.asarray(discovery_windows, dtype=bool)
    spectral = np.asarray(spectral_ratios, dtype=np.float64)
    control = np.asarray(control_ratios, dtype=np.float64)
    expected = (windows.size, len(names))
    if windows.ndim != 1 or discovery.shape != (windows.size,):
        raise ValueError("window_indices and discovery_windows must be one-dimensional")
    if np.unique(windows).size != windows.size:
        raise ValueError("window_indices must be unique")
    if spectral.shape != expected or control.shape != expected:
        raise ValueError(f"ratio arrays must have shape {expected}")
    if int(min_discovery_windows) < 1:
        raise ValueError("min_discovery_windows must be positive")
    if int(min_confirmation_windows) < 1:
        raise ValueError("min_confirmation_windows must be positive")
    if not 0.0 < float(min_confirmation_scorable_fraction) <= 1.0:
        raise ValueError("min_confirmation_scorable_fraction must be in (0, 1]")
    if (
        float(min_discovery_spectral_ratio) <= 0.0
        or float(min_discovery_control_ratio) <= 0.0
    ):
        raise ValueError("discovery ratio thresholds must be positive")
    available = (
        np.ones(expected, dtype=bool)
        if scorable is None
        else np.asarray(scorable, dtype=bool)
    )
    if available.shape != expected:
        raise ValueError(f"scorable must have shape {expected}")
    cell = _combined_statistic(spectral, control)
    cell[~available] = np.nan

    finite_cell = np.isfinite(cell)
    if np.any(finite_cell):
        flattened = np.where(finite_cell, cell, -np.inf).argmax()
        maximum_row, maximum_candidate = np.unravel_index(flattened, expected)
        maximum = float(cell[maximum_row, maximum_candidate])
        maximum_window = int(windows[maximum_row])
    else:
        maximum = -np.inf
        maximum_window = -1
        maximum_candidate = -1

    selection_scores = np.full(len(names), np.nan, dtype=np.float64)
    eligible: list[tuple[float, int]] = []
    for candidate_index in range(len(names)):
        rows = discovery & available[:, candidate_index]
        spectral_median = _finite_median(spectral[rows, candidate_index])
        control_median = _finite_median(control[rows, candidate_index])
        selection_score = (
            float(np.log2(spectral_median) + np.log2(control_median))
            if np.isfinite(spectral_median)
            and spectral_median > 0.0
            and np.isfinite(control_median)
            and control_median > 0.0
            else float("nan")
        )
        selection_scores[candidate_index] = selection_score
        if (
            int(np.count_nonzero(rows)) >= int(min_discovery_windows)
            and spectral_median >= float(min_discovery_spectral_ratio)
            and control_median >= float(min_discovery_control_ratio)
            and np.isfinite(selection_score)
        ):
            eligible.append((selection_score, candidate_index))

    selected = max(
        eligible,
        default=(-np.inf, -1),
        key=lambda item: (item[0], names[item[1]] if item[1] >= 0 else ""),
    )[1]
    confirmation_statistic = -np.inf
    total_confirmation_count = int(np.count_nonzero(~discovery))
    selected_confirmation_count = 0
    selected_confirmation_fraction = 0.0
    confirmation_gate_passed = False
    if selected >= 0:
        rows = (~discovery) & available[:, selected]
        selected_confirmation_count = int(np.count_nonzero(rows))
        selected_confirmation_fraction = float(
            selected_confirmation_count / total_confirmation_count
        ) if total_confirmation_count else 0.0
        confirmation_gate_passed = bool(
            selected_confirmation_count >= int(min_confirmation_windows)
            and selected_confirmation_fraction
            >= float(min_confirmation_scorable_fraction)
        )
        spectral_median = _finite_median(spectral[rows, selected])
        control_median = _finite_median(control[rows, selected])
        if (
            confirmation_gate_passed
            and np.isfinite(spectral_median)
            and spectral_median > 0.0
            and np.isfinite(control_median)
            and control_median > 0.0
        ):
            confirmation_statistic = float(
                np.log2(spectral_median) + np.log2(control_median)
            )

    return PhotometryFamilyEvaluation(
        candidate_names=names,
        window_indices=windows.copy(),
        discovery_windows=discovery.copy(),
        spectral_ratios=spectral.copy(),
        control_ratios=control.copy(),
        scorable=available.copy(),
        cell_statistics=cell,
        discovery_selection_scores=selection_scores,
        selected_candidate_index=int(selected),
        selected_confirmation_window_count=selected_confirmation_count,
        total_confirmation_window_count=total_confirmation_count,
        selected_confirmation_scorable_fraction=selected_confirmation_fraction,
        selected_confirmation_gate_passed=confirmation_gate_passed,
        selected_confirmation_statistic=confirmation_statistic,
        maximum_cell_statistic=maximum,
        maximum_window_index=maximum_window,
        maximum_candidate_index=int(maximum_candidate),
    )


def compute_photometry_family_null_batch(
    dataset: LocalCoordinateDataset,
    active_rows: np.ndarray,
    *,
    surrogate_indices: Sequence[int],
    seed: int,
    scorer: Callable[[LocalCoordinateDataset], PhotometryFamilyEvaluation],
    spatial_block_px: int = 2,
    min_shift_seconds: float = 1.0,
    max_gap_factor: float = 1.75,
    workers: int = 1,
) -> PhotometryFamilyNullBatch:
    """Rerun a transform-family scorer on autocorrelation-preserving nulls.

    Each random stream depends only on ``seed`` and the global surrogate index,
    so splitting work into batches or changing thread count cannot alter samples.
    """

    dataset.validated()
    active = np.asarray(active_rows, dtype=bool)
    if active.shape != (dataset.frame_count,):
        raise ValueError("active_rows must match the dataset frame axis")
    indices = np.asarray(surrogate_indices, dtype=np.int64)
    if indices.ndim != 1 or np.any(indices < 0):
        raise ValueError("surrogate_indices must be one-dimensional and nonnegative")
    if np.unique(indices).size != indices.size:
        raise ValueError("surrogate_indices must be unique")
    if int(workers) < 1:
        raise ValueError("workers must be positive")
    if int(spatial_block_px) < 1:
        raise ValueError("spatial_block_px must be positive")
    if float(min_shift_seconds) < 0.0:
        raise ValueError("min_shift_seconds cannot be negative")
    if float(max_gap_factor) <= 1.0:
        raise ValueError("max_gap_factor must exceed one")

    def score_one(surrogate_index: int) -> PhotometryFamilyEvaluation:
        rng = np.random.default_rng(
            np.random.SeedSequence([int(seed), int(surrogate_index)])
        )
        surrogate = autocorrelation_preserving_surrogate(
            dataset,
            active,
            rng=rng,
            spatial_block_px=int(spatial_block_px),
            min_shift_seconds=float(min_shift_seconds),
            max_gap_factor=float(max_gap_factor),
        )
        return scorer(surrogate)

    if int(workers) == 1:
        evaluations = [score_one(int(index)) for index in indices]
    else:
        with ThreadPoolExecutor(max_workers=int(workers)) as executor:
            evaluations = list(executor.map(score_one, indices.tolist()))
    return PhotometryFamilyNullBatch(
        surrogate_indices=indices.copy(),
        maximum_cell_statistics=np.asarray(
            [item.maximum_cell_statistic for item in evaluations], dtype=np.float64
        ),
        selected_confirmation_statistics=np.asarray(
            [item.selected_confirmation_statistic for item in evaluations],
            dtype=np.float64,
        ),
        selected_candidate_indices=np.asarray(
            [item.selected_candidate_index for item in evaluations], dtype=np.int32
        ),
        selected_confirmation_window_counts=np.asarray(
            [item.selected_confirmation_window_count for item in evaluations],
            dtype=np.int16,
        ),
        selected_confirmation_scorable_fractions=np.asarray(
            [item.selected_confirmation_scorable_fraction for item in evaluations],
            dtype=np.float64,
        ),
        selected_confirmation_gate_passed=np.asarray(
            [item.selected_confirmation_gate_passed for item in evaluations],
            dtype=bool,
        ),
        maximum_window_indices=np.asarray(
            [item.maximum_window_index for item in evaluations], dtype=np.int32
        ),
        maximum_candidate_indices=np.asarray(
            [item.maximum_candidate_index for item in evaluations], dtype=np.int32
        ),
    )


def familywise_p_values(observed: np.ndarray, null_maximum: np.ndarray) -> np.ndarray:
    """Plus-one maximum-statistic p-values for arbitrary observed cells."""

    values = np.asarray(observed, dtype=np.float64)
    null = np.asarray(null_maximum, dtype=np.float64)
    if null.ndim != 1 or not null.size or np.any(np.isnan(null)):
        raise ValueError("null_maximum must be a nonempty one-dimensional array without NaN")
    flat = values.reshape(-1)
    output = np.full(flat.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(flat)
    output[finite] = (
        1.0 + np.sum(null[None, :] >= flat[finite, None], axis=1)
    ) / float(null.size + 1)
    return output.reshape(values.shape)


def plus_one_p_value(observed: float, null: np.ndarray) -> float:
    samples = np.asarray(null, dtype=np.float64)
    if samples.ndim != 1 or not samples.size or np.any(np.isnan(samples)):
        raise ValueError("null must be a nonempty one-dimensional array without NaN")
    return float((1.0 + np.count_nonzero(samples >= float(observed))) / (samples.size + 1.0))


def higher_quantile(samples: np.ndarray, *, alpha: float) -> float:
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 1 or not values.size or np.any(np.isnan(values)):
        raise ValueError("samples must be a nonempty one-dimensional array without NaN")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be between zero and one")
    return float(np.quantile(values, 1.0 - float(alpha), method="higher"))


def write_photometry_null_batch(
    path: Path,
    *,
    identity: str,
    batch: PhotometryFamilyNullBatch,
) -> None:
    """Atomically persist one resumable null batch without pickle payloads."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            schema=np.asarray(_BATCH_SCHEMA),
            identity=np.asarray(str(identity)),
            surrogate_indices=np.asarray(batch.surrogate_indices, dtype=np.int64),
            maximum_cell_statistics=np.asarray(
                batch.maximum_cell_statistics, dtype=np.float64
            ),
            selected_confirmation_statistics=np.asarray(
                batch.selected_confirmation_statistics, dtype=np.float64
            ),
            selected_candidate_indices=np.asarray(
                batch.selected_candidate_indices, dtype=np.int32
            ),
            selected_confirmation_window_counts=np.asarray(
                batch.selected_confirmation_window_counts, dtype=np.int16
            ),
            selected_confirmation_scorable_fractions=np.asarray(
                batch.selected_confirmation_scorable_fractions, dtype=np.float64
            ),
            selected_confirmation_gate_passed=np.asarray(
                batch.selected_confirmation_gate_passed, dtype=bool
            ),
            maximum_window_indices=np.asarray(
                batch.maximum_window_indices, dtype=np.int32
            ),
            maximum_candidate_indices=np.asarray(
                batch.maximum_candidate_indices, dtype=np.int32
            ),
        )
    os.replace(temporary, output)


def load_photometry_null_batch(
    path: Path,
    *,
    identity: str,
    expected_indices: np.ndarray,
) -> PhotometryFamilyNullBatch | None:
    """Load an exactly matching batch, returning ``None`` for stale artifacts."""

    source = Path(path)
    if not source.exists():
        return None
    try:
        with np.load(source, allow_pickle=False) as payload:
            if str(payload["schema"].item()) != _BATCH_SCHEMA:
                return None
            if str(payload["identity"].item()) != str(identity):
                return None
            indices = np.asarray(payload["surrogate_indices"], dtype=np.int64)
            if not np.array_equal(indices, np.asarray(expected_indices, dtype=np.int64)):
                return None
            arrays = {
                name: np.asarray(payload[name]).copy()
                for name in (
                    "maximum_cell_statistics",
                    "selected_confirmation_statistics",
                    "selected_candidate_indices",
                    "selected_confirmation_window_counts",
                    "selected_confirmation_scorable_fractions",
                    "selected_confirmation_gate_passed",
                    "maximum_window_indices",
                    "maximum_candidate_indices",
                )
            }
    except (KeyError, OSError, ValueError):
        return None
    if any(np.asarray(value).shape != indices.shape for value in arrays.values()):
        return None
    return PhotometryFamilyNullBatch(
        surrogate_indices=indices,
        maximum_cell_statistics=np.asarray(
            arrays["maximum_cell_statistics"], dtype=np.float64
        ),
        selected_confirmation_statistics=np.asarray(
            arrays["selected_confirmation_statistics"], dtype=np.float64
        ),
        selected_candidate_indices=np.asarray(
            arrays["selected_candidate_indices"], dtype=np.int32
        ),
        selected_confirmation_window_counts=np.asarray(
            arrays["selected_confirmation_window_counts"], dtype=np.int16
        ),
        selected_confirmation_scorable_fractions=np.asarray(
            arrays["selected_confirmation_scorable_fractions"], dtype=np.float64
        ),
        selected_confirmation_gate_passed=np.asarray(
            arrays["selected_confirmation_gate_passed"], dtype=bool
        ),
        maximum_window_indices=np.asarray(
            arrays["maximum_window_indices"], dtype=np.int32
        ),
        maximum_candidate_indices=np.asarray(
            arrays["maximum_candidate_indices"], dtype=np.int32
        ),
    )


__all__ = [
    "PhotometryFamilyEvaluation",
    "PhotometryFamilyNullBatch",
    "compute_photometry_family_null_batch",
    "evaluate_photometry_family",
    "familywise_p_values",
    "higher_quantile",
    "load_photometry_null_batch",
    "plus_one_p_value",
    "write_photometry_null_batch",
]
