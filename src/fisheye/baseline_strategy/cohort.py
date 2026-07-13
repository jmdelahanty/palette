"""Cohort-relative scoring and optional cluster discovery."""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .contracts import (
    BASELINE_STRATEGY_CLASSIFICATION_TABLE,
    BASELINE_STRATEGY_CLUSTERS_TABLE,
    METHOD,
    METHOD_VERSION,
    SCHEMA_ID,
    SCHEMA_VERSION,
    StrategyFeatureConfig,
)


def _float(value: object) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _identity(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "recording_id": row.get("recording_id"),
        "track_id": row.get("track_id"),
        "baseline_window_id": row.get("baseline_window_id"),
        "baseline_window_label": row.get("baseline_window_label"),
        "source_export_run_id": row.get("source_export_run_id"),
        "zarr_path": row.get("zarr_path"),
    }


def _common(table_name: str, row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "table_name": table_name,
        "method": METHOD,
        "method_version": METHOD_VERSION,
        **_identity(row),
    }


def _identity_key(row: Mapping[str, Any]) -> tuple[object, object, object]:
    return row.get("recording_id"), row.get("track_id"), row.get("baseline_window_id")


def _log1p(value: float) -> float:
    return math.log1p(max(0.0, value))


def _identity_transform(value: float) -> float:
    return value


def _negative(value: float) -> float:
    return -value


Metric = tuple[str, float, Callable[[float], float]]

AXIS_METRICS: dict[str, tuple[Metric, ...]] = {
    "activity": (
        ("path_per_min_mm", 1.0, _log1p),
        ("bout_rate_per_min", 1.0, _log1p),
        ("active_sample_fraction", 1.0, _identity_transform),
        ("p95_speed_mm_s", 0.75, _log1p),
    ),
    "boundary": (
        ("wall_enrichment_log2", 1.0, _identity_transform),
        ("mean_center_distance_norm", 1.0, _identity_transform),
        ("active_wall_fraction", 1.0, _identity_transform),
        ("wall_following_episode_fraction", 1.0, _identity_transform),
    ),
    "spatial_distribution": (
        ("occupancy_coverage_fraction", 1.0, _identity_transform),
        ("occupancy_entropy_accessible_normalized", 1.0, _identity_transform),
        ("occupancy_js_divergence_uniform", 1.0, _negative),
        ("occupancy_max_cell_fraction", 0.75, _negative),
        ("source_spatial_entropy_normalized", 0.5, _identity_transform),
        ("source_quadrant_entropy_normalized", 0.5, _identity_transform),
    ),
    "home_base": (
        ("dominant_dwell_cell_fraction", 1.0, _identity_transform),
        ("dominant_to_second_dwell_ratio", 0.75, _log1p),
        ("dominant_dwell_return_fraction", 1.0, _identity_transform),
        ("dominant_dwell_visit_count", 0.5, _log1p),
    ),
    "temporal_expansion": (
        ("wall_fraction_delta_late_minus_early", 1.0, _negative),
        ("wall_fraction_slope_per_baseline", 0.75, _negative),
        ("center_distance_norm_delta_late_minus_early", 1.0, _negative),
        ("center_distance_norm_slope_per_baseline", 0.75, _negative),
    ),
}


def _robust_scale(values: Sequence[float]) -> tuple[float, float] | None:
    array = np.asarray(values, dtype=float)
    if array.size < 2:
        return None
    center = float(np.median(array))
    scale = 1.4826 * float(np.median(np.abs(array - center)))
    if not math.isfinite(scale) or scale <= 1e-12:
        q25, q75 = np.percentile(array, [25.0, 75.0])
        scale = float(q75 - q25) / 1.349
    if not math.isfinite(scale) or scale <= 1e-12:
        scale = float(np.std(array))
    if not math.isfinite(scale) or scale <= 1e-12:
        return None
    return center, scale


def _metric_scalers(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[float, float, Callable[[float], float]]]:
    scalers: dict[str, tuple[float, float, Callable[[float], float]]] = {}
    metrics = {metric for definitions in AXIS_METRICS.values() for metric in definitions}
    for name, _weight, transform in metrics:
        values = []
        for row in rows:
            value = _float(row.get(name))
            if value is not None:
                values.append(transform(value))
        scale = _robust_scale(values)
        if scale is not None:
            scalers[name] = (*scale, transform)
    return scalers


def _axis_score(
    row: Mapping[str, Any],
    definitions: Sequence[Metric],
    scalers: Mapping[str, tuple[float, float, Callable[[float], float]]],
) -> tuple[float | None, int]:
    weighted = []
    weights = []
    for name, weight, _declared_transform in definitions:
        value = _float(row.get(name))
        scaler = scalers.get(name)
        if value is None or scaler is None:
            continue
        center, scale, transform = scaler
        weighted.append(weight * ((transform(value) - center) / scale))
        weights.append(weight)
    if not weighted:
        return None, 0
    return float(sum(weighted) / sum(weights)), len(weighted)


def _state(score: float | None, *, low: str, middle: str, high: str, threshold: float) -> str:
    if score is None:
        return "unavailable"
    if score <= -threshold:
        return low
    if score >= threshold:
        return high
    return middle


def _primary_strategy(
    *,
    activity: float | None,
    boundary: float | None,
    spatial: float | None,
    home_base: float | None,
    temporal: float | None,
    threshold: float,
) -> str:
    if activity is not None and activity <= -threshold:
        return "inactive_or_low_activity"
    if (
        boundary is not None
        and temporal is not None
        and boundary >= 0
        and temporal >= threshold
    ):
        return "initial_wall_bias_then_expansion"
    if boundary is not None and boundary >= threshold:
        return "active_wall_following"
    if home_base is not None and home_base >= threshold:
        return "home_base_like_explorer"
    if spatial is not None and spatial >= threshold:
        return "broad_even_explorer"
    if spatial is not None and spatial <= -threshold:
        return "localized_explorer"
    return "mixed_or_uncertain"


def classify_strategy_features(
    feature_rows: Sequence[Mapping[str, Any]],
    *,
    config: StrategyFeatureConfig | None = None,
) -> list[dict[str, Any]]:
    """Assign factorized cohort-relative states without claiming anxiety."""

    config = config or StrategyFeatureConfig()
    config.validate()
    complete = [row for row in feature_rows if row.get("feature_status") == "complete"]
    scalers = _metric_scalers(complete)
    output = []
    threshold = config.relative_score_threshold
    for source in feature_rows:
        row = dict(source)
        result = _common(BASELINE_STRATEGY_CLASSIFICATION_TABLE, row)
        if row.get("feature_status") != "complete":
            result.update(
                {
                    "classification_status": "invalid",
                    "classification_reason": row.get("feature_reason") or "feature_row_invalid",
                    "activity_state": "unavailable",
                    "boundary_strategy": "unavailable",
                    "spatial_organization": "unavailable",
                    "temporal_pattern": "unavailable",
                    "primary_strategy": "unavailable",
                    "classification_confidence_score": None,
                }
            )
            output.append(result)
            continue
        scores: dict[str, float | None] = {}
        for axis, definitions in AXIS_METRICS.items():
            score, count = _axis_score(row, definitions, scalers)
            scores[axis] = score
            result[f"{axis}_score"] = score
            result[f"{axis}_metric_count"] = count
        activity_state = _state(
            scores["activity"],
            low="inactive",
            middle="typical_activity",
            high="active",
            threshold=threshold,
        )
        boundary_strategy = _state(
            scores["boundary"],
            low="boundary_neutral",
            middle="mixed_boundary",
            high="wall_following",
            threshold=threshold,
        )
        home_base_state = _state(
            scores["home_base"],
            low="diffuse",
            middle="weakly_localized",
            high="home_base_like",
            threshold=threshold,
        )
        if scores["home_base"] is not None and scores["home_base"] >= threshold:
            spatial_organization = "home_base_like"
        else:
            spatial_organization = _state(
                scores["spatial_distribution"],
                low="localized",
                middle="intermediate",
                high="broad_even",
                threshold=threshold,
            )
        temporal_pattern = _state(
            scores["temporal_expansion"],
            low="contracting",
            middle="stable_or_mixed",
            high="expanding",
            threshold=threshold,
        )
        primary = _primary_strategy(
            activity=scores["activity"],
            boundary=scores["boundary"],
            spatial=scores["spatial_distribution"],
            home_base=scores["home_base"],
            temporal=scores["temporal_expansion"],
            threshold=threshold,
        )
        finite_scores = [abs(score) for score in scores.values() if score is not None]
        strongest = max(finite_scores) if finite_scores else None
        confidence = (
            0.5 + 0.5 * math.tanh(max(0.0, strongest - threshold))
            if strongest is not None and primary != "mixed_or_uncertain"
            else 0.0 if strongest is not None else None
        )
        result.update(
            {
                "classification_status": "complete" if finite_scores else "insufficient_cohort_variation",
                "classification_reason": None if finite_scores else "no_robustly_scalable_metrics",
                "reference_scope": "source_export_cohort_relative",
                "relative_score_threshold": threshold,
                "activity_state": activity_state,
                "boundary_strategy": boundary_strategy,
                "home_base_state": home_base_state,
                "spatial_organization": spatial_organization,
                "temporal_pattern": temporal_pattern,
                "primary_strategy": primary,
                "classification_confidence_score": confidence,
                "confidence_semantics": "descriptive_distance_not_probability",
                "anxiety_inference_permitted": False,
            }
        )
        output.append(result)
    return output


def discover_strategy_clusters(
    classification_rows: Sequence[Mapping[str, Any]],
    *,
    config: StrategyFeatureConfig | None = None,
) -> list[dict[str, Any]]:
    """Discover optional Gaussian-mixture clusters with BIC and stability evidence."""

    from sklearn.metrics import adjusted_rand_score
    from sklearn.mixture import GaussianMixture

    config = config or StrategyFeatureConfig()
    config.validate()
    axes = (
        "activity_score",
        "boundary_score",
        "spatial_distribution_score",
        "home_base_score",
        "temporal_expansion_score",
    )
    valid_indexes = []
    matrix_rows = []
    for index, row in enumerate(classification_rows):
        values = [_float(row.get(axis)) for axis in axes]
        if row.get("classification_status") == "complete" and sum(value is not None for value in values) >= 3:
            valid_indexes.append(index)
            matrix_rows.append([0.0 if value is None else value for value in values])
    base_rows = [
        {
            **_common(BASELINE_STRATEGY_CLUSTERS_TABLE, row),
            "cluster_status": "excluded",
            "cluster_reason": "classification_not_cluster_eligible",
            "cluster_id": None,
            "cluster_probability": None,
        }
        for row in classification_rows
    ]
    if len(matrix_rows) < 6:
        for index in valid_indexes:
            base_rows[index].update(
                cluster_status="insufficient_cohort_size",
                cluster_reason="at_least_six_complete_rows_required",
            )
        return base_rows

    matrix = np.asarray(matrix_rows, dtype=float)
    max_components = min(config.cluster_max_components, max(1, len(matrix_rows) // 3))
    candidates = []
    for component_count in range(1, max_components + 1):
        model = GaussianMixture(
            n_components=component_count,
            covariance_type="full",
            reg_covar=1e-5,
            n_init=10,
            random_state=config.random_seed,
        ).fit(matrix)
        candidates.append((float(model.bic(matrix)), model))
    selected_bic, selected = min(candidates, key=lambda item: item[0])
    labels = selected.predict(matrix)
    probabilities = np.max(selected.predict_proba(matrix), axis=1)
    component_count = int(selected.n_components)
    stability_values = []
    if component_count > 1 and config.cluster_stability_resamples > 0:
        rng = np.random.default_rng(config.random_seed)
        sample_size = max(component_count * 2, int(math.ceil(0.8 * len(matrix))))
        for _ in range(config.cluster_stability_resamples):
            selected_rows = rng.choice(len(matrix), size=sample_size, replace=False)
            try:
                replicate = GaussianMixture(
                    n_components=component_count,
                    covariance_type="full",
                    reg_covar=1e-5,
                    n_init=5,
                    random_state=int(rng.integers(0, 2**31 - 1)),
                ).fit(matrix[selected_rows])
                stability_values.append(
                    adjusted_rand_score(labels, replicate.predict(matrix))
                )
            except ValueError:
                continue
    stability = float(np.median(stability_values)) if stability_values else None
    bic_by_components = {
        str(model.n_components): bic for bic, model in sorted(candidates, key=lambda item: item[1].n_components)
    }
    for matrix_index, source_index in enumerate(valid_indexes):
        probability = float(probabilities[matrix_index])
        status = "complete"
        reason = None
        if component_count == 1:
            status = "no_multimodal_structure"
            reason = "bic_selected_one_component"
        elif probability < config.cluster_probability_threshold:
            status = "uncertain"
            reason = "assignment_probability_below_threshold"
        base_rows[source_index].update(
            {
                "cluster_status": status,
                "cluster_reason": reason,
                "cluster_id": int(labels[matrix_index]) if component_count > 1 else None,
                "cluster_probability": probability if component_count > 1 else None,
                "cluster_probability_threshold": config.cluster_probability_threshold,
                "selected_component_count": component_count,
                "selected_bic": selected_bic,
                "bic_by_component_count": bic_by_components,
                "cluster_stability_median_ari": stability,
                "cluster_stability_resample_count": len(stability_values),
                "cluster_axes": list(axes),
                "cluster_semantics": "unsupervised_ids_require_posthoc_interpretation",
            }
        )
    return base_rows


__all__ = [
    "AXIS_METRICS",
    "classify_strategy_features",
    "discover_strategy_clusters",
]
