"""Pure feature derivation for whole-training chaser responses."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .contracts import (
    METHOD,
    METHOD_VERSION,
    SCHEMA_ID,
    SCHEMA_VERSION,
    TRAINING_RESPONSE_FEATURES_TABLE,
    TrainingResponseConfig,
)


def _float(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _int(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _ratio_log2(training: object, pre: object, *, epsilon: float = 1e-6) -> float | None:
    training_value = _float(training)
    pre_value = _float(pre)
    if training_value is None or pre_value is None:
        return None
    return math.log2((max(0.0, training_value) + epsilon) / (max(0.0, pre_value) + epsilon))


def _delta(training: object, pre: object) -> float | None:
    training_value = _float(training)
    pre_value = _float(pre)
    if training_value is None or pre_value is None:
        return None
    return training_value - pre_value


def _one_by_key(
    rows: Sequence[Mapping[str, Any]], key: str
) -> dict[str, Mapping[str, Any]]:
    output: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        value = str(row.get(key) or "").strip().lower()
        if not value:
            continue
        if value in output:
            raise ValueError(f"duplicate {key}={value!r}")
        output[value] = row
    return output


def _role_windows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    output: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        window = str(row.get("window_label") or "").strip().lower()
        role = str(row.get("behavior_class") or "").strip().lower()
        if not window or not role:
            continue
        key = (window, role)
        if key in output:
            raise ValueError(f"duplicate distance/orientation row {key!r}")
        output[key] = row
    return output


def _weighted_speed(
    rows: Sequence[Mapping[str, Any]], *, maximum_distance_mm: float | None
) -> float | None:
    speed_sum = 0.0
    sample_count = 0
    for row in rows:
        center = _float(row.get("distance_bin_center_mm"))
        count = _int(row.get("speed_sample_count"))
        total = _float(row.get("speed_sum_mm_s"))
        mean = _float(row.get("mean_speed_mm_s"))
        if total is None and mean is not None and count is not None:
            total = mean * count
        if center is None or count is None or total is None or count <= 0:
            continue
        if maximum_distance_mm is not None and center > maximum_distance_mm:
            continue
        speed_sum += total
        sample_count += count
    return speed_sum / sample_count if sample_count else None


def _weighted_far_speed(
    rows: Sequence[Mapping[str, Any]], *, minimum_distance_mm: float
) -> float | None:
    speed_sum = 0.0
    sample_count = 0
    for row in rows:
        center = _float(row.get("distance_bin_center_mm"))
        count = _int(row.get("speed_sample_count"))
        total = _float(row.get("speed_sum_mm_s"))
        mean = _float(row.get("mean_speed_mm_s"))
        if total is None and mean is not None and count is not None:
            total = mean * count
        if center is None or count is None or total is None or count <= 0:
            continue
        if center <= minimum_distance_mm:
            continue
        speed_sum += total
        sample_count += count
    return speed_sum / sample_count if sample_count else None


def derive_training_response_features(
    *,
    recording_id: str,
    source_export_run_id: str,
    behavior_rows: Sequence[Mapping[str, Any]],
    distance_rows: Sequence[Mapping[str, Any]],
    egocentric_rows: Sequence[Mapping[str, Any]] = (),
    speed_distance_rows: Sequence[Mapping[str, Any]] = (),
    protocol_name: str | None = None,
    config: TrainingResponseConfig | None = None,
) -> dict[str, Any]:
    """Build one recording-level pre-to-training response feature row."""

    config = config or TrainingResponseConfig()
    config.validate()
    behaviors = _one_by_key(behavior_rows, "window_label")
    pre = behaviors.get(config.pre_window_label.lower())
    training = behaviors.get(config.training_window_label.lower())
    distances = _role_windows(distance_rows)
    orientations = _role_windows(egocentric_rows)
    training_window_id = _int(training.get("window_id")) if training else None
    row: dict[str, Any] = {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "table_name": TRAINING_RESPONSE_FEATURES_TABLE,
        "method": METHOD,
        "method_version": METHOD_VERSION,
        "recording_id": recording_id,
        "training_window_id": training_window_id,
        "source_export_run_id": source_export_run_id,
        "protocol_name": protocol_name,
        "pre_window_label": config.pre_window_label,
        "training_window_label": config.training_window_label,
        "zarr_path": (training or pre or {}).get("zarr_path"),
        "temporal_training_features_available": False,
        "temporal_training_feature_reason": "training_time_bins_and_samples_not_exported",
    }
    if pre is None or training is None:
        row.update(
            feature_status="invalid",
            feature_reason="missing_pre_or_training_behavior_summary",
            training_duration_s=None,
        )
        return row

    pre_duration = _float(pre.get("duration_s"))
    training_duration = _float(training.get("duration_s"))
    pre_dropout = _float(pre.get("tracking_dropout_fraction"))
    training_dropout = _float(training.get("tracking_dropout_fraction"))
    row.update(
        {
            "pre_duration_s": pre_duration,
            "training_duration_s": training_duration,
            "pre_tracking_dropout_fraction": pre_dropout,
            "training_tracking_dropout_fraction": training_dropout,
            "pre_valid_position_fraction": (
                1.0 - pre_dropout if pre_dropout is not None else None
            ),
            "training_valid_position_fraction": (
                1.0 - training_dropout if training_dropout is not None else None
            ),
        }
    )

    positive_metrics = {
        "mean_speed_mm_s": "mean_speed_mm_s",
        "p95_speed_mm_s": "p95_speed_mm_s",
        "bout_rate_per_min": "bout_rate_per_min",
        "mean_bout_path_length_mm": "mean_bout_path_length_mm",
        "mean_abs_bout_net_heading_change_deg": "mean_abs_bout_net_heading_change_deg",
    }
    pre_path_per_min = (
        _float(pre.get("total_path_mm")) / pre_duration * 60.0
        if _float(pre.get("total_path_mm")) is not None
        and pre_duration is not None
        and pre_duration > 0
        else None
    )
    training_path_per_min = (
        _float(training.get("total_path_mm")) / training_duration * 60.0
        if _float(training.get("total_path_mm")) is not None
        and training_duration is not None
        and training_duration > 0
        else None
    )
    row.update(
        pre_path_per_min_mm=pre_path_per_min,
        training_path_per_min_mm=training_path_per_min,
        path_per_min_log2_ratio=_ratio_log2(training_path_per_min, pre_path_per_min),
    )
    for output_name, source_name in positive_metrics.items():
        pre_value = _float(pre.get(source_name))
        training_value = _float(training.get(source_name))
        row[f"pre_{output_name}"] = pre_value
        row[f"training_{output_name}"] = training_value
        row[f"{output_name}_log2_ratio"] = _ratio_log2(training_value, pre_value)

    arena_radius = _float(training.get("arena_radius_mm")) or _float(
        pre.get("arena_radius_mm")
    )
    pre_center = _float(pre.get("median_distance_from_arena_center_mm"))
    training_center = _float(training.get("median_distance_from_arena_center_mm"))
    pre_center_norm = pre_center / arena_radius if pre_center is not None and arena_radius else None
    training_center_norm = (
        training_center / arena_radius
        if training_center is not None and arena_radius
        else None
    )
    row.update(
        {
            "arena_radius_mm": arena_radius,
            "pre_wall_fraction": _float(pre.get("wall_fraction")),
            "training_wall_fraction": _float(training.get("wall_fraction")),
            "wall_fraction_delta": _delta(
                training.get("wall_fraction"), pre.get("wall_fraction")
            ),
            "pre_center_distance_norm": pre_center_norm,
            "training_center_distance_norm": training_center_norm,
            "center_distance_norm_delta": _delta(training_center_norm, pre_center_norm),
        }
    )

    required_role_rows = True
    for role in (config.aggressive_role, config.inert_role):
        pre_distance = distances.get((config.pre_window_label.lower(), role.lower()))
        training_distance = distances.get(
            (config.training_window_label.lower(), role.lower())
        )
        if pre_distance is None or training_distance is None:
            required_role_rows = False
            continue
        for metric in ("p05_distance_mm", "p50_distance_mm", "fraction_within_threshold"):
            pre_value = _float(pre_distance.get(metric))
            training_value = _float(training_distance.get(metric))
            row[f"{role}_pre_{metric}"] = pre_value
            row[f"{role}_training_{metric}"] = training_value
            row[f"{role}_{metric}_delta"] = _delta(training_value, pre_value)
        row[f"{role}_threshold_mm"] = _float(training_distance.get("threshold_mm"))

        pre_orientation = orientations.get(
            (config.pre_window_label.lower(), role.lower())
        )
        training_orientation = orientations.get(
            (config.training_window_label.lower(), role.lower())
        )
        for metric in (
            "mean_alignment_cos",
            "fraction_front_45",
            "fraction_behind_45",
            "circular_resultant_length",
        ):
            pre_value = _float(pre_orientation.get(metric)) if pre_orientation else None
            training_value = (
                _float(training_orientation.get(metric))
                if training_orientation
                else None
            )
            row[f"{role}_pre_{metric}"] = pre_value
            row[f"{role}_training_{metric}"] = training_value
            row[f"{role}_{metric}_delta"] = _delta(training_value, pre_value)

        threshold = _float(training_distance.get("threshold_mm"))
        role_speed_rows = [
            source
            for source in speed_distance_rows
            if str(source.get("window_label") or "").strip().lower()
            == config.training_window_label.lower()
            and str(source.get("behavior_class") or source.get("role") or "")
            .strip()
            .lower()
            == role.lower()
        ]
        if not role_speed_rows:
            chaser_index = _int(training_distance.get("chaser_index"))
            role_speed_rows = [
                source
                for source in speed_distance_rows
                if str(source.get("window_label") or "").strip().lower()
                == config.training_window_label.lower()
                and _int(source.get("chaser_index")) == chaser_index
            ]
        near_speed = (
            _weighted_speed(role_speed_rows, maximum_distance_mm=threshold)
            if threshold is not None
            else None
        )
        far_speed = (
            _weighted_far_speed(role_speed_rows, minimum_distance_mm=threshold)
            if threshold is not None
            else None
        )
        row[f"{role}_training_near_speed_mm_s"] = near_speed
        row[f"{role}_training_far_speed_mm_s"] = far_speed
        row[f"{role}_near_minus_far_speed_mm_s"] = _delta(near_speed, far_speed)

    aggressive = config.aggressive_role
    inert = config.inert_role
    row["training_role_p50_distance_contrast_mm"] = _delta(
        row.get(f"{aggressive}_training_p50_distance_mm"),
        row.get(f"{inert}_training_p50_distance_mm"),
    )
    row["training_role_p05_distance_contrast_mm"] = _delta(
        row.get(f"{aggressive}_training_p05_distance_mm"),
        row.get(f"{inert}_training_p05_distance_mm"),
    )
    row["training_role_within_threshold_contrast"] = _delta(
        row.get(f"{aggressive}_training_fraction_within_threshold"),
        row.get(f"{inert}_training_fraction_within_threshold"),
    )

    reasons = []
    if training_duration is None or training_duration < config.min_training_duration_s:
        reasons.append("training_duration_below_threshold")
    minimum = config.min_valid_position_fraction
    if pre_dropout is None or 1.0 - pre_dropout < minimum:
        reasons.append("pre_tracking_coverage_below_threshold")
    if training_dropout is None or 1.0 - training_dropout < minimum:
        reasons.append("training_tracking_coverage_below_threshold")
    if not required_role_rows:
        reasons.append("missing_pre_or_training_role_distance_summary")
    row["feature_status"] = "invalid" if reasons else "complete"
    row["feature_reason"] = ";".join(reasons) if reasons else None
    row["interpretation_guardrail"] = (
        "descriptive response profile; fear, anxiety, and escape success are not inferred"
    )
    return row


__all__ = ["derive_training_response_features"]
