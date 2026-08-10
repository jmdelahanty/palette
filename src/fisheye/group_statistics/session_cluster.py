"""Session-clustered inference for recording-level group statistics."""

from __future__ import annotations

from dataclasses import dataclass
import math
import warnings
from typing import Sequence

import numpy as np
from scipy import stats


SESSION_RANDOM_INTERCEPT_METHOD = "session_random_intercept_reml_v1"
# A random-effect variance and asymptotic Wald test are fragile with only a
# handful of groups. Ten independent sessions is the fail-closed publication
# default; smaller predeclared designs remain possible down to three sessions.
DEFAULT_MINIMUM_SESSIONS = 10


@dataclass(frozen=True)
class SessionClusterResult:
    """One intercept-only random-effects fit with explicit availability state."""

    status: str
    reason: str | None
    method: str
    unit: str
    unit_count: int
    cluster_count: int
    mean: float | None
    standard_error: float | None
    ci_low: float | None
    ci_high: float | None
    p_value: float | None
    cluster_variance: float | None
    residual_variance: float | None
    intraclass_correlation: float | None


def _unavailable(
    *,
    reason: str,
    unit_count: int,
    cluster_count: int,
) -> SessionClusterResult:
    return SessionClusterResult(
        status="unavailable",
        reason=reason,
        method=SESSION_RANDOM_INTERCEPT_METHOD,
        unit="session",
        unit_count=unit_count,
        cluster_count=cluster_count,
        mean=None,
        standard_error=None,
        ci_low=None,
        ci_high=None,
        p_value=None,
        cluster_variance=None,
        residual_variance=None,
        intraclass_correlation=None,
    )


def fit_session_random_intercept(
    values: Sequence[float] | np.ndarray,
    session_ids: Sequence[object] | np.ndarray,
    *,
    confidence_level: float,
    minimum_sessions: int = DEFAULT_MINIMUM_SESSIONS,
) -> SessionClusterResult:
    """Fit ``value ~ 1 + (1 | session)`` without parsing recording names.

    The caller must supply persisted registry session identities. Fit failure is
    reported as unavailable; it never falls back to a model that ignores the
    requested clustering.
    """

    if not 0.0 < float(confidence_level) < 1.0:
        raise ValueError("confidence_level must be in (0, 1).")
    if type(minimum_sessions) is not int or minimum_sessions < 3:
        raise ValueError("minimum_sessions must be an integer >= 3.")
    y = np.asarray(values, dtype=np.float64).reshape(-1)
    raw_sessions = np.asarray(session_ids, dtype=object).reshape(-1)
    if y.shape != raw_sessions.shape:
        raise ValueError("Session-cluster values and identities must have equal length.")

    normalized_sessions = np.asarray(
        [str(value).strip() if value is not None else "" for value in raw_sessions],
        dtype=object,
    )
    usable = np.isfinite(y) & (normalized_sessions != "")
    y = y[usable]
    sessions = normalized_sessions[usable]
    unit_count = int(y.size)
    cluster_count = int(np.unique(sessions).size)
    if unit_count < 2:
        return _unavailable(
            reason="insufficient_complete_recordings",
            unit_count=unit_count,
            cluster_count=cluster_count,
        )
    if cluster_count < int(minimum_sessions):
        return _unavailable(
            reason=f"session_count<{int(minimum_sessions)}",
            unit_count=unit_count,
            cluster_count=cluster_count,
        )
    if unit_count == cluster_count:
        return _unavailable(
            reason="no_repeated_session_observations",
            unit_count=unit_count,
            cluster_count=cluster_count,
        )

    try:
        from statsmodels.regression.mixed_linear_model import MixedLM

        exog = np.ones((unit_count, 1), dtype=np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = MixedLM(y, exog, groups=sessions).fit(
                reml=True,
                method=["lbfgs", "powell"],
                disp=False,
            )
        if not bool(fitted.converged):
            raise ValueError("mixed model did not converge")
        mean = float(np.asarray(fitted.fe_params, dtype=np.float64)[0])
        standard_error = float(np.asarray(fitted.bse_fe, dtype=np.float64)[0])
        cluster_variance = float(
            np.asarray(fitted.cov_re, dtype=np.float64).reshape(-1)[0]
        )
        residual_variance = float(fitted.scale)
        if not all(
            math.isfinite(value)
            for value in (
                mean,
                standard_error,
                cluster_variance,
                residual_variance,
            )
        ) or standard_error <= 0.0:
            raise ValueError("mixed model returned non-finite estimates")
    except Exception as exc:
        return _unavailable(
            reason=f"fit_failed:{type(exc).__name__}",
            unit_count=unit_count,
            cluster_count=cluster_count,
        )

    alpha = 1.0 - float(confidence_level)
    critical = float(stats.norm.ppf(1.0 - alpha / 2.0))
    z_value = mean / standard_error
    p_value = float(2.0 * stats.norm.sf(abs(z_value)))
    variance_sum = cluster_variance + residual_variance
    icc = cluster_variance / variance_sum if variance_sum > 0.0 else math.nan
    if not math.isfinite(icc):
        return _unavailable(
            reason="non_finite_intraclass_correlation",
            unit_count=unit_count,
            cluster_count=cluster_count,
        )
    boundary = cluster_variance <= np.finfo(np.float64).eps
    return SessionClusterResult(
        status="boundary_zero_variance" if boundary else "computed",
        reason=None,
        method=SESSION_RANDOM_INTERCEPT_METHOD,
        unit="session",
        unit_count=unit_count,
        cluster_count=cluster_count,
        mean=mean,
        standard_error=standard_error,
        ci_low=mean - critical * standard_error,
        ci_high=mean + critical * standard_error,
        p_value=p_value,
        cluster_variance=max(0.0, cluster_variance),
        residual_variance=max(0.0, residual_variance),
        intraclass_correlation=max(0.0, min(1.0, icc)),
    )
