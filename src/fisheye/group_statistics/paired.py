"""Paired recording-level contrast statistics."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ContrastDefinition:
    name: str
    condition_a: str
    condition_b: str


@dataclass(frozen=True)
class PairedContrastStats:
    unit_count: int
    paired_unit_count: int
    excluded_unit_count: int
    mean_a: float | None
    mean_b: float | None
    mean_difference: float | None
    median_difference: float | None
    std_difference: float | None
    effect_size: float | None
    ci_low: float | None
    ci_high: float | None
    p_value: float | None
    test_method: str
    bootstrap_iterations: int
    permutation_iterations: int
    status: str
    skip_reason: str | None


@dataclass(frozen=True)
class OneSampleSignedRankStats:
    unit_count: int
    paired_unit_count: int
    excluded_unit_count: int
    mean_observed: float | None
    median_observed: float | None
    mean_difference: float | None
    median_difference: float | None
    std_difference: float | None
    effect_size: float | None
    ci_low: float | None
    ci_high: float | None
    p_value: float | None
    test_method: str
    bootstrap_iterations: int
    permutation_iterations: int
    status: str
    skip_reason: str | None


def _finite_array(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return array[np.isfinite(array)]


def _safe_float(value: float | np.floating | None) -> float | None:
    if value is None:
        return None
    out = float(value)
    return out if math.isfinite(out) else None


def bootstrap_mean_ci(
    differences: Sequence[float],
    *,
    iterations: int,
    confidence_level: float,
    rng: np.random.Generator,
) -> tuple[float | None, float | None]:
    diffs = _finite_array(differences)
    if diffs.size == 0 or iterations <= 0:
        return None, None
    if diffs.size == 1:
        value = float(diffs[0])
        return value, value
    samples = rng.choice(diffs, size=(int(iterations), int(diffs.size)), replace=True)
    means = np.mean(samples, axis=1)
    alpha = 1.0 - float(confidence_level)
    low, high = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return _safe_float(low), _safe_float(high)


def bootstrap_median_ci(
    values: Sequence[float],
    *,
    iterations: int,
    confidence_level: float,
    rng: np.random.Generator,
) -> tuple[float | None, float | None]:
    finite = _finite_array(values)
    if finite.size == 0 or iterations <= 0:
        return None, None
    if finite.size == 1:
        value = float(finite[0])
        return value, value
    samples = rng.choice(finite, size=(int(iterations), int(finite.size)), replace=True)
    medians = np.median(samples, axis=1)
    alpha = 1.0 - float(confidence_level)
    low, high = np.quantile(medians, [alpha / 2.0, 1.0 - alpha / 2.0])
    return _safe_float(low), _safe_float(high)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < sorted_values.size:
        end = start + 1
        while end < sorted_values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = (float(start + 1) + float(end)) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def wilcoxon_signed_rank_p_value(
    differences: Sequence[float],
    *,
    exact_max_n: int = 20,
) -> tuple[float | None, str, float | None, float | None]:
    diffs = _finite_array(differences)
    diffs = diffs[diffs != 0.0]
    n = int(diffs.size)
    if n == 0:
        return None, "wilcoxon_signed_rank_unavailable", None, None
    ranks = _average_ranks(np.abs(diffs))
    w_plus = float(np.sum(ranks[diffs > 0.0]))
    w_minus = float(np.sum(ranks[diffs < 0.0]))
    rank_sum = float(np.sum(ranks))
    rank_biserial = (w_plus - w_minus) / rank_sum if rank_sum > 0 else None

    if n <= int(exact_max_n):
        observed = min(w_plus, w_minus)
        tail_count = 0
        total = 2**n
        for signs in product((0.0, 1.0), repeat=n):
            candidate_w_plus = float(np.sum(ranks * np.asarray(signs, dtype=np.float64)))
            candidate_stat = min(candidate_w_plus, rank_sum - candidate_w_plus)
            if candidate_stat <= observed + 1e-12:
                tail_count += 1
        return float(tail_count) / float(total), "wilcoxon_signed_rank_exact", rank_biserial, w_plus

    mean = rank_sum / 2.0
    variance = float(np.sum(ranks * ranks)) / 4.0
    if variance <= 0:
        return None, "wilcoxon_signed_rank_unavailable", rank_biserial, w_plus
    # Continuity-corrected two-sided normal approximation.
    z = (abs(w_plus - mean) - 0.5) / math.sqrt(variance)
    p_value = math.erfc(abs(z) / math.sqrt(2.0))
    return _safe_float(p_value), "wilcoxon_signed_rank_normal", rank_biserial, w_plus


def paired_sign_flip_p_value(
    differences: Sequence[float],
    *,
    iterations: int,
    rng: np.random.Generator,
) -> tuple[float | None, str, int]:
    diffs = _finite_array(differences)
    n = int(diffs.size)
    if n == 0:
        return None, "paired_sign_flip_unavailable", 0
    observed = abs(float(np.mean(diffs)))
    if observed == 0.0:
        return 1.0, "paired_sign_flip_exact", 2**n if n <= 20 else int(iterations)

    if n <= 20:
        null_means = []
        for signs in product((-1.0, 1.0), repeat=n):
            null_means.append(abs(float(np.mean(diffs * np.asarray(signs, dtype=np.float64)))))
        null = np.asarray(null_means, dtype=np.float64)
        p_value = float(np.mean(null >= observed - 1e-12))
        return p_value, "paired_sign_flip_exact", int(null.size)

    draws = int(iterations)
    if draws <= 0:
        return None, "paired_sign_flip_unavailable", 0
    signs = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float64), size=(draws, n), replace=True)
    null = np.abs(np.mean(signs * diffs.reshape(1, -1), axis=1))
    # Add-one correction for sampled randomization tests.
    p_value = float((np.sum(null >= observed - 1e-12) + 1.0) / (draws + 1.0))
    return p_value, "paired_sign_flip_random", draws


def compute_paired_contrast(
    values_a: Sequence[float],
    values_b: Sequence[float],
    *,
    unit_count: int,
    minimum_recordings: int,
    bootstrap_iterations: int,
    permutation_iterations: int,
    confidence_level: float,
    rng: np.random.Generator,
) -> PairedContrastStats:
    a = np.asarray(values_a, dtype=np.float64).reshape(-1)
    b = np.asarray(values_b, dtype=np.float64).reshape(-1)
    n = int(min(a.size, b.size))
    a = a[:n]
    b = b[:n]
    valid = np.isfinite(a) & np.isfinite(b)
    paired_a = a[valid]
    paired_b = b[valid]
    diffs = paired_b - paired_a
    paired_n = int(diffs.size)
    excluded = max(0, int(unit_count) - paired_n)

    if paired_n < int(minimum_recordings):
        return PairedContrastStats(
            unit_count=int(unit_count),
            paired_unit_count=paired_n,
            excluded_unit_count=excluded,
            mean_a=_safe_float(np.mean(paired_a)) if paired_a.size else None,
            mean_b=_safe_float(np.mean(paired_b)) if paired_b.size else None,
            mean_difference=None,
            median_difference=None,
            std_difference=None,
            effect_size=None,
            ci_low=None,
            ci_high=None,
            p_value=None,
            test_method="skipped",
            bootstrap_iterations=0,
            permutation_iterations=0,
            status="skipped",
            skip_reason=f"paired_unit_count<{int(minimum_recordings)}",
        )

    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1)) if paired_n > 1 else None
    effect_size = mean_diff / std_diff if std_diff is not None and std_diff > 0 else None
    ci_low, ci_high = bootstrap_mean_ci(
        diffs,
        iterations=int(bootstrap_iterations),
        confidence_level=float(confidence_level),
        rng=rng,
    )
    p_value, test_method, effective_permutations = paired_sign_flip_p_value(
        diffs,
        iterations=int(permutation_iterations),
        rng=rng,
    )
    return PairedContrastStats(
        unit_count=int(unit_count),
        paired_unit_count=paired_n,
        excluded_unit_count=excluded,
        mean_a=_safe_float(np.mean(paired_a)),
        mean_b=_safe_float(np.mean(paired_b)),
        mean_difference=_safe_float(mean_diff),
        median_difference=_safe_float(np.median(diffs)),
        std_difference=_safe_float(std_diff),
        effect_size=_safe_float(effect_size),
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=_safe_float(p_value),
        test_method=test_method,
        bootstrap_iterations=int(bootstrap_iterations),
        permutation_iterations=int(effective_permutations),
        status="computed",
        skip_reason=None,
    )


def compute_one_sample_signed_rank(
    values: Sequence[float],
    *,
    unit_count: int,
    minimum_recordings: int,
    bootstrap_iterations: int,
    confidence_level: float,
    rng: np.random.Generator,
) -> OneSampleSignedRankStats:
    all_values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = all_values[np.isfinite(all_values)]
    nonzero = finite[finite != 0.0]
    paired_n = int(finite.size)
    excluded = max(0, int(unit_count) - paired_n)

    if paired_n < int(minimum_recordings):
        return OneSampleSignedRankStats(
            unit_count=int(unit_count),
            paired_unit_count=paired_n,
            excluded_unit_count=excluded,
            mean_observed=_safe_float(np.mean(finite)) if finite.size else None,
            median_observed=_safe_float(np.median(finite)) if finite.size else None,
            mean_difference=None,
            median_difference=None,
            std_difference=None,
            effect_size=None,
            ci_low=None,
            ci_high=None,
            p_value=None,
            test_method="skipped",
            bootstrap_iterations=0,
            permutation_iterations=0,
            status="skipped",
            skip_reason=f"paired_unit_count<{int(minimum_recordings)}",
        )
    if nonzero.size == 0:
        return OneSampleSignedRankStats(
            unit_count=int(unit_count),
            paired_unit_count=paired_n,
            excluded_unit_count=excluded,
            mean_observed=0.0,
            median_observed=0.0,
            mean_difference=0.0,
            median_difference=0.0,
            std_difference=0.0 if paired_n > 1 else None,
            effect_size=0.0,
            ci_low=0.0,
            ci_high=0.0,
            p_value=1.0,
            test_method="wilcoxon_signed_rank_all_zero",
            bootstrap_iterations=0,
            permutation_iterations=0,
            status="computed",
            skip_reason=None,
        )

    std_diff = float(np.std(finite, ddof=1)) if paired_n > 1 else None
    ci_low, ci_high = bootstrap_median_ci(
        finite,
        iterations=int(bootstrap_iterations),
        confidence_level=float(confidence_level),
        rng=rng,
    )
    p_value, test_method, rank_biserial, _w_plus = wilcoxon_signed_rank_p_value(finite)
    return OneSampleSignedRankStats(
        unit_count=int(unit_count),
        paired_unit_count=paired_n,
        excluded_unit_count=excluded,
        mean_observed=_safe_float(np.mean(finite)),
        median_observed=_safe_float(np.median(finite)),
        mean_difference=_safe_float(np.mean(finite)),
        median_difference=_safe_float(np.median(finite)),
        std_difference=_safe_float(std_diff),
        effect_size=_safe_float(rank_biserial),
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=_safe_float(p_value),
        test_method=test_method,
        bootstrap_iterations=int(bootstrap_iterations),
        permutation_iterations=0,
        status="computed" if p_value is not None else "skipped",
        skip_reason=None if p_value is not None else "wilcoxon_unavailable",
    )


def benjamini_hochberg(p_values: Sequence[float | None]) -> list[float | None]:
    indexed = [
        (index, float(value))
        for index, value in enumerate(p_values)
        if value is not None and math.isfinite(float(value))
    ]
    out: list[float | None] = [None] * len(p_values)
    m = len(indexed)
    if m == 0:
        return out
    indexed.sort(key=lambda item: item[1])
    adjusted = [0.0] * m
    running = 1.0
    for rank_from_end, (index, p_value) in enumerate(reversed(indexed), start=1):
        rank = m - rank_from_end + 1
        running = min(running, p_value * m / rank)
        adjusted[rank - 1] = running
    for (index, _p_value), q_value in zip(indexed, adjusted):
        out[index] = min(1.0, float(q_value))
    return out
