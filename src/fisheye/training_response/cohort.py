"""Cohort-relative scoring for whole-training response features."""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .contracts import (
    METHOD,
    METHOD_VERSION,
    SCHEMA_ID,
    SCHEMA_VERSION,
    TRAINING_RESPONSE_CLASSIFICATION_TABLE,
    TRAINING_RESPONSE_CLUSTERS_TABLE,
    TrainingResponseConfig,
)


def _float(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _identity(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "recording_id": row.get("recording_id"),
        "training_window_id": row.get("training_window_id"),
        "source_export_run_id": row.get("source_export_run_id"),
        "protocol_name": row.get("protocol_name"),
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


def _identity_transform(value: float) -> float:
    return value


def _negative(value: float) -> float:
    return -value


Metric = tuple[str, float, Callable[[float], float]]

AXIS_METRICS: dict[str, tuple[Metric, ...]] = {
    "locomotor_response": (
        ("mean_speed_mm_s_log2_ratio", 1.0, _identity_transform),
        ("p95_speed_mm_s_log2_ratio", 0.75, _identity_transform),
        ("path_per_min_log2_ratio", 1.0, _identity_transform),
        ("bout_rate_per_min_log2_ratio", 1.0, _identity_transform),
    ),
    "boundary_response": (
        ("wall_fraction_delta", 1.0, _identity_transform),
        ("center_distance_norm_delta", 1.0, _identity_transform),
    ),
    "aggressive_proximity": (
        ("aggressive_training_p05_distance_mm", 1.0, _identity_transform),
        ("aggressive_training_p50_distance_mm", 1.0, _identity_transform),
        (
            "aggressive_training_fraction_within_threshold",
            1.0,
            _negative,
        ),
    ),
    "role_distance_selectivity": (
        ("training_role_p05_distance_contrast_mm", 1.0, _identity_transform),
        ("training_role_p50_distance_contrast_mm", 1.0, _identity_transform),
        ("training_role_within_threshold_contrast", 1.0, _negative),
    ),
    "close_contact_vigor": (
        ("aggressive_near_minus_far_speed_mm_s", 1.0, _identity_transform),
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
    metrics = {definition for axis in AXIS_METRICS.values() for definition in axis}
    for name, _weight, transform in metrics:
        values = [
            transform(value)
            for row in rows
            if (value := _float(row.get(name))) is not None
        ]
        scale = _robust_scale(values)
        if scale is not None:
            scalers[name] = (*scale, transform)
    return scalers


def _axis_score(
    row: Mapping[str, Any],
    definitions: Sequence[Metric],
    scalers: Mapping[str, tuple[float, float, Callable[[float], float]]],
) -> tuple[float | None, int]:
    weighted: list[float] = []
    weights: list[float] = []
    for name, weight, _transform in definitions:
        value = _float(row.get(name))
        scaler = scalers.get(name)
        if value is None or scaler is None:
            continue
        center, scale, transform = scaler
        weighted.append(weight * ((transform(value) - center) / scale))
        weights.append(weight)
    return (
        (float(sum(weighted) / sum(weights)), len(weighted))
        if weighted
        else (None, 0)
    )


def _state(
    score: float | None,
    *,
    low: str,
    middle: str,
    high: str,
    threshold: float,
) -> str:
    if score is None:
        return "unavailable"
    if score <= -threshold:
        return low
    if score >= threshold:
        return high
    return middle


def _primary_profile(
    scores: Mapping[str, float | None], threshold: float
) -> str:
    locomotor = scores.get("locomotor_response")
    proximity = scores.get("aggressive_proximity")
    selectivity = scores.get("role_distance_selectivity")
    boundary = scores.get("boundary_response")
    if locomotor is not None and proximity is not None:
        if locomotor >= threshold and proximity >= threshold:
            return "active_distance_maintenance"
        if locomotor >= threshold and proximity <= -threshold:
            return "high_activity_close_proximity"
        if locomotor <= -threshold and proximity >= threshold:
            return "low_activity_distance_maintenance"
        if locomotor <= -threshold and proximity <= -threshold:
            return "low_activity_close_proximity"
    if selectivity is not None and abs(selectivity) >= threshold:
        return "role_selective_proximity"
    if boundary is not None and abs(boundary) >= threshold:
        return "boundary_relocation_response"
    finite = [abs(value) for value in scores.values() if value is not None]
    if finite and max(finite) < threshold:
        return "limited_summary_change"
    return "mixed_training_response"


def classify_training_response_features(
    feature_rows: Sequence[Mapping[str, Any]],
    *,
    config: TrainingResponseConfig | None = None,
) -> list[dict[str, Any]]:
    """Assign factorized descriptive states relative to the valid cohort."""

    config = config or TrainingResponseConfig()
    config.validate()
    complete = [row for row in feature_rows if row.get("feature_status") == "complete"]
    scalers = _metric_scalers(complete)
    threshold = config.relative_score_threshold
    output: list[dict[str, Any]] = []
    for source in feature_rows:
        row = dict(source)
        result = _common(TRAINING_RESPONSE_CLASSIFICATION_TABLE, row)
        if row.get("feature_status") != "complete":
            for axis in AXIS_METRICS:
                result[f"{axis}_score"] = None
                result[f"{axis}_metric_count"] = 0
            result.update(
                {
                    "classification_status": "invalid",
                    "classification_reason": row.get("feature_reason")
                    or "feature_row_invalid",
                    "locomotor_response": "unavailable",
                    "boundary_response": "unavailable",
                    "aggressive_proximity_state": "unavailable",
                    "role_distance_selectivity": "unavailable",
                    "close_contact_vigor": "unavailable",
                    "primary_training_profile": "unavailable",
                    "profile_separation_score": None,
                    "profile_separation_semantics": (
                        "descriptive_distance_not_probability"
                    ),
                    "causal_avoidance_inference_permitted": False,
                    "temporal_adaptation_inference_permitted": False,
                }
            )
            output.append(result)
            continue
        scores: dict[str, float | None] = {}
        for axis, definitions in AXIS_METRICS.items():
            score, metric_count = _axis_score(row, definitions, scalers)
            scores[axis] = score
            result[f"{axis}_score"] = score
            result[f"{axis}_metric_count"] = metric_count
        result.update(
            {
                "classification_status": "complete",
                "classification_reason": None,
                "reference_scope": "combined_source_export_cohort_relative",
                "relative_score_threshold": threshold,
                "locomotor_response": _state(
                    scores["locomotor_response"],
                    low="suppressed",
                    middle="stable_or_mixed",
                    high="activated",
                    threshold=threshold,
                ),
                "boundary_response": _state(
                    scores["boundary_response"],
                    low="decreased_boundary_bias",
                    middle="stable_or_mixed",
                    high="increased_boundary_bias",
                    threshold=threshold,
                ),
                "aggressive_proximity_state": _state(
                    scores["aggressive_proximity"],
                    low="closer_than_cohort",
                    middle="cohort_typical_proximity",
                    high="farther_than_cohort",
                    threshold=threshold,
                ),
                "role_distance_selectivity": _state(
                    scores["role_distance_selectivity"],
                    low="farther_from_inert",
                    middle="similar_role_proximity",
                    high="farther_from_aggressive",
                    threshold=threshold,
                ),
                "close_contact_vigor": _state(
                    scores["close_contact_vigor"],
                    low="reduced_near_aggressive",
                    middle="similar_near_and_far",
                    high="elevated_near_aggressive",
                    threshold=threshold,
                ),
                "primary_training_profile": _primary_profile(scores, threshold),
                "profile_separation_semantics": (
                    "descriptive_distance_not_probability"
                ),
                "causal_avoidance_inference_permitted": False,
                "temporal_adaptation_inference_permitted": False,
            }
        )
        finite = [abs(value) for value in scores.values() if value is not None]
        strongest = max(finite) if finite else None
        result["profile_separation_score"] = (
            0.5 + 0.5 * math.tanh(max(0.0, strongest - threshold))
            if strongest is not None
            and result["primary_training_profile"]
            not in {"limited_summary_change", "mixed_training_response"}
            else 0.0 if strongest is not None else None
        )
        output.append(result)
    return output


def discover_training_response_clusters(
    classification_rows: Sequence[Mapping[str, Any]],
    *,
    config: TrainingResponseConfig | None = None,
) -> list[dict[str, Any]]:
    """Discover optional Gaussian-mixture clusters with BIC and stability."""

    from sklearn.metrics import adjusted_rand_score
    from sklearn.mixture import GaussianMixture

    config = config or TrainingResponseConfig()
    config.validate()
    axes = tuple(f"{axis}_score" for axis in AXIS_METRICS)
    valid_indexes: list[int] = []
    matrix_rows: list[list[float]] = []
    for index, row in enumerate(classification_rows):
        values = [_float(row.get(axis)) for axis in axes]
        if row.get("classification_status") == "complete" and sum(
            value is not None for value in values
        ) >= 3:
            valid_indexes.append(index)
            matrix_rows.append([0.0 if value is None else value for value in values])
    output = [
        {
            **_common(TRAINING_RESPONSE_CLUSTERS_TABLE, row),
            "cluster_status": "excluded",
            "cluster_reason": "classification_not_cluster_eligible",
            "cluster_id": None,
            "cluster_probability": None,
        }
        for row in classification_rows
    ]
    if len(matrix_rows) < 6:
        for index in valid_indexes:
            output[index].update(
                cluster_status="insufficient_cohort_size",
                cluster_reason="at_least_six_complete_rows_required",
            )
        return output
    matrix = np.asarray(matrix_rows, dtype=float)
    max_components = min(
        config.cluster_max_components,
        max(1, len(matrix) // config.cluster_min_rows_per_component),
    )
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
        str(model.n_components): bic
        for bic, model in sorted(candidates, key=lambda item: item[1].n_components)
    }
    for matrix_index, source_index in enumerate(valid_indexes):
        probability = float(probabilities[matrix_index])
        status = "complete"
        reason = None
        if component_count == 1:
            status = "no_multimodal_structure"
            reason = "bic_selected_one_component"
        elif stability is None:
            status = "stability_unavailable"
            reason = "no_successful_stability_resamples"
        elif stability < config.cluster_stability_threshold:
            status = "unstable_model"
            reason = "median_ari_below_stability_threshold"
        elif probability < config.cluster_probability_threshold:
            status = "uncertain"
            reason = "assignment_probability_below_threshold"
        output[source_index].update(
            {
                "cluster_status": status,
                "cluster_reason": reason,
                "cluster_id": (
                    int(labels[matrix_index]) if component_count > 1 else None
                ),
                "cluster_probability": probability if component_count > 1 else None,
                "cluster_probability_threshold": config.cluster_probability_threshold,
                "selected_component_count": component_count,
                "selected_bic": selected_bic,
                "bic_by_component_count": bic_by_components,
                "cluster_stability_median_ari": stability,
                "cluster_stability_threshold": config.cluster_stability_threshold,
                "cluster_stability_resample_count": len(stability_values),
                "cluster_min_rows_per_component": (
                    config.cluster_min_rows_per_component
                ),
                "cluster_axes": list(axes),
                "cluster_semantics": "unsupervised_ids_require_posthoc_interpretation",
            }
        )
    return output


__all__ = [
    "AXIS_METRICS",
    "classify_training_response_features",
    "discover_training_response_clusters",
]
