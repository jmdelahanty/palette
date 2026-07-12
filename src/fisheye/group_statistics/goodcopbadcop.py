"""GoodCopBadCop profile for cross-recording group statistics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import socket
from typing import Any, Mapping, Sequence

import numpy as np
import polars as pl

from fisheye.analytics_exports.capabilities import resolve_capabilities
from fisheye.analytics_exports.contracts import (
    CHASER_CRA_NEAR_FIELD_SUMMARY_TABLE,
    CHASER_CRA_SUMMARY_TABLE,
    CHASER_DISTANCE_SUMMARY_TABLE,
    CHASER_EGOCENTRIC_SUMMARY_TABLE,
    CHASER_EPOCH_BEHAVIOR_TABLE,
    CHASER_SPATIAL_TABLE,
    DESCRIPTIVE_TABLE,
    EXPORT_SCHEMA_ID,
    EXPORT_SCHEMA_VERSION,
    STATISTICS_TABLE,
    TABLE_CONTRACTS,
    canonicalize_export_row,
    contract_snapshot,
    validate_table_columns,
)
from fisheye.shared.json_safety import json_attr_safe, strict_json_dumps
from fisheye.shared.system_metadata import get_git_info

from .paired import (
    ContrastDefinition,
    benjamini_hochberg,
    compute_one_sample_signed_rank,
    compute_paired_contrast,
)


SUMMARY_TABLE = STATISTICS_TABLE
CRA_SUMMARY_TABLE = CHASER_CRA_SUMMARY_TABLE
CRA_NEAR_FIELD_SUMMARY_TABLE = CHASER_CRA_NEAR_FIELD_SUMMARY_TABLE
EPOCH_BEHAVIOR_TABLE = CHASER_EPOCH_BEHAVIOR_TABLE
SCHEMA_VERSION = EXPORT_SCHEMA_VERSION
DEFAULT_CONTRASTS = (
    ContrastDefinition("training-pre", "pre", "training"),
    ContrastDefinition("post-pre", "pre", "post"),
    ContrastDefinition("post-training", "training", "post"),
)


@dataclass(frozen=True)
class MetricSpec:
    metric_family: str
    source_table: str
    metric_name: str
    group_keys: tuple[str, ...]
    primary: bool = True
    exploratory: bool = False


DEFAULT_METRICS: tuple[MetricSpec, ...] = (
    MetricSpec(
        metric_family="chaser_distance",
        source_table=CHASER_DISTANCE_SUMMARY_TABLE,
        metric_name="mean_distance_mm",
        group_keys=("chaser_index",),
    ),
    MetricSpec(
        metric_family="chaser_distance",
        source_table=CHASER_DISTANCE_SUMMARY_TABLE,
        metric_name="p50_distance_mm",
        group_keys=("chaser_index",),
    ),
    MetricSpec(
        metric_family="chaser_distance",
        source_table=CHASER_DISTANCE_SUMMARY_TABLE,
        metric_name="fraction_within_threshold",
        group_keys=("chaser_index",),
    ),
    MetricSpec(
        metric_family="spatial_occupancy",
        source_table=CHASER_SPATIAL_TABLE,
        metric_name="time_s",
        group_keys=("zone_set_id", "zone_id", "zone_label"),
    ),
    MetricSpec(
        metric_family="spatial_occupancy",
        source_table=CHASER_SPATIAL_TABLE,
        metric_name="fraction_of_epoch",
        group_keys=("zone_set_id", "zone_id", "zone_label"),
    ),
    MetricSpec(
        metric_family="spatial_occupancy",
        source_table=CHASER_SPATIAL_TABLE,
        metric_name="fraction_of_detected",
        group_keys=("zone_set_id", "zone_id", "zone_label"),
    ),
    MetricSpec(
        metric_family="epoch_behavior",
        source_table=EPOCH_BEHAVIOR_TABLE,
        metric_name="mean_speed_mm_s",
        group_keys=(),
        primary=False,
        exploratory=True,
    ),
    MetricSpec(
        metric_family="epoch_behavior",
        source_table=EPOCH_BEHAVIOR_TABLE,
        metric_name="bout_rate_per_min",
        group_keys=(),
        primary=False,
        exploratory=True,
    ),
    MetricSpec(
        metric_family="epoch_behavior",
        source_table=EPOCH_BEHAVIOR_TABLE,
        metric_name="mean_bout_duration_s",
        group_keys=(),
        primary=False,
        exploratory=True,
    ),
    MetricSpec(
        metric_family="epoch_behavior",
        source_table=EPOCH_BEHAVIOR_TABLE,
        metric_name="mean_bout_path_length_mm",
        group_keys=(),
        primary=False,
        exploratory=True,
    ),
    MetricSpec(
        metric_family="epoch_behavior",
        source_table=EPOCH_BEHAVIOR_TABLE,
        metric_name="mean_bout_net_heading_change_deg",
        group_keys=(),
        primary=False,
        exploratory=True,
    ),
    MetricSpec(
        metric_family="epoch_behavior",
        source_table=EPOCH_BEHAVIOR_TABLE,
        metric_name="mean_abs_bout_net_heading_change_deg",
        group_keys=(),
        primary=False,
        exploratory=True,
    ),
    MetricSpec(
        metric_family="epoch_behavior",
        source_table=EPOCH_BEHAVIOR_TABLE,
        metric_name="mean_bout_heading_path_deg",
        group_keys=(),
        primary=False,
        exploratory=True,
    ),
    MetricSpec(
        metric_family="epoch_behavior",
        source_table=EPOCH_BEHAVIOR_TABLE,
        metric_name="inter_bout_interval_count",
        group_keys=(),
        primary=False,
        exploratory=True,
    ),
    MetricSpec(
        metric_family="epoch_behavior",
        source_table=EPOCH_BEHAVIOR_TABLE,
        metric_name="wall_fraction",
        group_keys=(),
        primary=False,
        exploratory=True,
    ),
    MetricSpec(
        metric_family="epoch_behavior",
        source_table=EPOCH_BEHAVIOR_TABLE,
        metric_name="median_distance_from_arena_center_mm",
        group_keys=(),
        primary=False,
        exploratory=True,
    ),
    MetricSpec(
        metric_family="epoch_behavior",
        source_table=EPOCH_BEHAVIOR_TABLE,
        metric_name="mean_inter_bout_interval_s",
        group_keys=(),
        primary=False,
        exploratory=True,
    ),
    MetricSpec(
        metric_family="epoch_behavior",
        source_table=EPOCH_BEHAVIOR_TABLE,
        metric_name="median_inter_bout_interval_s",
        group_keys=(),
        primary=False,
        exploratory=True,
    ),
    MetricSpec(
        metric_family="cra_primary_endpoint",
        source_table=CRA_SUMMARY_TABLE,
        metric_name="delta_agg",
        group_keys=(),
        primary=False,
    ),
    MetricSpec(
        metric_family="cra_primary_endpoint",
        source_table=CRA_SUMMARY_TABLE,
        metric_name="delta_occ_agg",
        group_keys=(),
        primary=False,
    ),
    MetricSpec(
        metric_family="cra_primary_endpoint",
        source_table=CRA_SUMMARY_TABLE,
        metric_name="specificity_distance",
        group_keys=(),
    ),
    MetricSpec(
        metric_family="cra_primary_endpoint",
        source_table=CRA_SUMMARY_TABLE,
        metric_name="specificity_occupancy",
        group_keys=(),
        primary=False,
    ),
    MetricSpec(
        metric_family="cra_primary_endpoint",
        source_table=CRA_SUMMARY_TABLE,
        metric_name="delta_inert",
        group_keys=(),
        primary=False,
    ),
    MetricSpec(
        metric_family="cra_primary_endpoint",
        source_table=CRA_SUMMARY_TABLE,
        metric_name="delta_occ_inert",
        group_keys=(),
        primary=False,
    ),
    MetricSpec(
        metric_family="cra_near_field",
        source_table=CRA_NEAR_FIELD_SUMMARY_TABLE,
        metric_name="approach_p05_delta_agg",
        group_keys=(),
    ),
    MetricSpec(
        metric_family="cra_near_field",
        source_table=CRA_NEAR_FIELD_SUMMARY_TABLE,
        metric_name="approach_p05_specificity",
        group_keys=(),
    ),
    MetricSpec(
        metric_family="cra_near_field",
        source_table=CRA_NEAR_FIELD_SUMMARY_TABLE,
        metric_name="nearzone_occ_delta_agg",
        group_keys=(),
    ),
    MetricSpec(
        metric_family="cra_near_field",
        source_table=CRA_NEAR_FIELD_SUMMARY_TABLE,
        metric_name="nearzone_occ_specificity",
        group_keys=(),
    ),
    MetricSpec(
        metric_family="cra_near_field",
        source_table=CRA_NEAR_FIELD_SUMMARY_TABLE,
        metric_name="nearzone_entry_rate_delta_agg",
        group_keys=(),
    ),
    MetricSpec(
        metric_family="cra_near_field",
        source_table=CRA_NEAR_FIELD_SUMMARY_TABLE,
        metric_name="nearzone_entry_rate_specificity",
        group_keys=(),
    ),
    MetricSpec(
        metric_family="cra_near_field",
        source_table=CRA_NEAR_FIELD_SUMMARY_TABLE,
        metric_name="approach_p05_delta_inert",
        group_keys=(),
        primary=False,
    ),
    MetricSpec(
        metric_family="cra_near_field",
        source_table=CRA_NEAR_FIELD_SUMMARY_TABLE,
        metric_name="nearzone_occ_delta_inert",
        group_keys=(),
        primary=False,
    ),
    MetricSpec(
        metric_family="cra_near_field",
        source_table=CRA_NEAR_FIELD_SUMMARY_TABLE,
        metric_name="nearzone_entry_rate_delta_inert",
        group_keys=(),
        primary=False,
    ),
    MetricSpec(
        metric_family="egocentric_alignment",
        source_table=CHASER_EGOCENTRIC_SUMMARY_TABLE,
        metric_name="mean_alignment_cos",
        group_keys=("chaser_index",),
    ),
    MetricSpec(
        metric_family="egocentric_alignment",
        source_table=CHASER_EGOCENTRIC_SUMMARY_TABLE,
        metric_name="mean_lateral_sin",
        group_keys=("chaser_index",),
    ),
    MetricSpec(
        metric_family="egocentric_alignment",
        source_table=CHASER_EGOCENTRIC_SUMMARY_TABLE,
        metric_name="fraction_front_45",
        group_keys=("chaser_index",),
    ),
    MetricSpec(
        metric_family="egocentric_alignment",
        source_table=CHASER_EGOCENTRIC_SUMMARY_TABLE,
        metric_name="fraction_lateral_45",
        group_keys=("chaser_index",),
    ),
    MetricSpec(
        metric_family="egocentric_alignment",
        source_table=CHASER_EGOCENTRIC_SUMMARY_TABLE,
        metric_name="fraction_behind_45",
        group_keys=("chaser_index",),
    ),
)


@dataclass(frozen=True)
class GoodCopBadCopStatisticsConfig:
    export_root: Path
    source_export_run_id: str
    stats_run_id: str
    metrics: tuple[MetricSpec, ...] = DEFAULT_METRICS
    contrasts: tuple[ContrastDefinition, ...] = DEFAULT_CONTRASTS
    bootstrap_iterations: int = 10000
    permutation_iterations: int = 10000
    confidence_level: float = 0.95
    minimum_recordings: int = 3
    random_seed: int = 0
    overwrite: bool = False


def utc_run_id() -> str:
    return "stats_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def source_manifest_path(export_root: Path, source_export_run_id: str) -> Path:
    return export_root / "v1" / "manifests" / f"export_run_id={source_export_run_id}.json"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_file(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _json_dumps(value: Any) -> str:
    return strict_json_dumps(json_attr_safe(value))


def canonical_condition(label: Any) -> str | None:
    text = str(label or "").strip().lower()
    if not text:
        return None
    if text in {"pre", "pre_event", "pre-event"} or text.startswith("pre"):
        return "pre"
    if text in {"training", "training_event", "train", "train_event"} or text.startswith(("training", "train")):
        return "training"
    if text in {"post", "post_event", "post-event"} or text.startswith("post"):
        return "post"
    return text


def _table_dir(export_root: Path, source_export_run_id: str, table: str) -> Path:
    return export_root / "v1" / table / f"export_run_id={source_export_run_id}"


def _read_export_table(export_root: Path, source_export_run_id: str, table: str) -> pl.DataFrame:
    table_dir = _table_dir(export_root, source_export_run_id, table)
    files = sorted(table_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet parts found for {table}: {table_dir}")
    return pl.scan_parquet([str(path) for path in files]).collect()


def _require_v2_source_manifest(manifest: Mapping[str, Any], *, path: Path) -> None:
    if manifest.get("schema_id") != EXPORT_SCHEMA_ID or manifest.get("schema_version") != EXPORT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported analytics export contract at {path}: "
            f"{manifest.get('schema_id')!r} version {manifest.get('schema_version')!r}; "
            f"re-export with {EXPORT_SCHEMA_ID} version {EXPORT_SCHEMA_VERSION}."
        )


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _group_key_payload(row: Mapping[str, Any], group_keys: Sequence[str]) -> dict[str, Any]:
    return {key: row.get(key) for key in group_keys}


def _result_id(row: Mapping[str, Any]) -> str:
    payload = {
        "metric_family": row.get("metric_family"),
        "metric_name": row.get("metric_name"),
        "contrast_name": row.get("contrast_name"),
        "group_key_json": row.get("group_key_json"),
        "source_export_run_id": row.get("source_export_run_id"),
    }
    return hashlib.sha256(_json_dumps(payload).encode("utf-8")).hexdigest()


def _descriptive_result_id(row: Mapping[str, Any]) -> str:
    payload = {
        "metric_family": row.get("metric_family"),
        "metric_name": row.get("metric_name"),
        "condition_name": row.get("condition_name"),
        "group_key_json": row.get("group_key_json"),
        "source_export_run_id": row.get("source_export_run_id"),
    }
    return hashlib.sha256(_json_dumps(payload).encode("utf-8")).hexdigest()


def _finite_values(values: Sequence[Any]) -> np.ndarray:
    data = np.asarray([_safe_float(value) for value in values], dtype=object)
    finite = [float(value) for value in data.tolist() if value is not None]
    return np.asarray(finite, dtype=np.float64)


def _descriptive_stats(values: Sequence[Any]) -> dict[str, Any]:
    finite = _finite_values(values)
    if finite.size == 0:
        return {
            "unit_count": 0,
            "sum": None,
            "mean": None,
            "median": None,
            "std_dev": None,
            "sem": None,
            "min": None,
            "max": None,
        }
    std_dev = float(np.std(finite, ddof=1)) if finite.size > 1 else None
    sem = float(std_dev / math.sqrt(float(finite.size))) if std_dev is not None else None
    return {
        "unit_count": int(finite.size),
        "sum": float(np.sum(finite)),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "std_dev": std_dev,
        "sem": sem,
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def _prepare_metric_frame(frame: pl.DataFrame, spec: MetricSpec) -> pl.DataFrame:
    required = {"recording_id", "window_label", spec.metric_name, *spec.group_keys}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{spec.source_table} is missing required column(s) for {spec.metric_name}: {missing}")

    selected = frame.select(
        [
            "recording_id",
            "window_label",
            spec.metric_name,
            *spec.group_keys,
        ]
    ).with_columns(
        pl.col("window_label")
        .map_elements(canonical_condition, return_dtype=pl.String)
        .alias("condition")
    )
    value_column = "_value"
    selected = selected.rename({spec.metric_name: value_column})
    return (
        selected.drop_nulls(["recording_id", "condition", value_column])
        .group_by(["recording_id", "condition", *spec.group_keys])
        .agg(pl.col(value_column).mean().alias(value_column))
    )


def _base_descriptive_row(
    *,
    spec: MetricSpec,
    config: GoodCopBadCopStatisticsConfig,
    source_manifest: Mapping[str, Any],
    source_manifest_sha256: str,
    source_row_count: int,
    group_key: Mapping[str, Any],
    condition_name: str,
    stats: Mapping[str, Any],
) -> dict[str, Any]:
    collection = source_manifest.get("collection_manifest")
    if not isinstance(collection, Mapping):
        collection = {}
    source_row_counts = source_manifest.get("row_counts_by_table")
    if not isinstance(source_row_counts, Mapping):
        source_row_counts = {}
    parameters = {
        "minimum_recordings": int(config.minimum_recordings),
        "missing_policy": "complete_recording_values_by_condition",
        "random_seed": int(config.random_seed),
    }
    row: dict[str, Any] = {
        "export_schema_version": SCHEMA_VERSION,
        "stats_run_id": config.stats_run_id,
        "source_export_run_id": config.source_export_run_id,
        "source_export_manifest_path": str(source_manifest_path(config.export_root, config.source_export_run_id)),
        "source_export_manifest_sha256": source_manifest_sha256,
        "collection_id": collection.get("collection_id"),
        "collection_manifest_sha256": collection.get("manifest_sha256"),
        "source_table": spec.source_table,
        "source_row_count": int(source_row_counts.get(spec.source_table) or source_row_count),
        "metric_family": spec.metric_family,
        "metric_name": spec.metric_name,
        "condition_name": condition_name,
        "group_key_json": _json_dumps(dict(group_key)),
        "primary": bool(spec.primary),
        "exploratory": bool(spec.exploratory),
        "unit": "recording",
        "unit_count": int(stats.get("unit_count") or 0),
        "sum": stats.get("sum"),
        "mean": stats.get("mean"),
        "median": stats.get("median"),
        "std_dev": stats.get("std_dev"),
        "sem": stats.get("sem"),
        "min": stats.get("min"),
        "max": stats.get("max"),
        "missing_policy": "complete_recording_values_by_condition",
        "parameters_json": _json_dumps(parameters),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    row["descriptive_result_id"] = _descriptive_result_id(row)
    return row


def _descriptive_rows_for_metric(
    *,
    frame: pl.DataFrame,
    spec: MetricSpec,
    config: GoodCopBadCopStatisticsConfig,
    source_manifest: Mapping[str, Any],
    source_manifest_sha256: str,
) -> list[dict[str, Any]]:
    source_row_count = int(frame.height)
    prepared = _prepare_metric_frame(frame, spec)
    group_rows = (
        prepared.select(list(spec.group_keys)).unique().sort(list(spec.group_keys)).to_dicts()
        if spec.group_keys
        else [{}]
    )
    rows: list[dict[str, Any]] = []
    for group_row in group_rows:
        group_frame = prepared
        for key in spec.group_keys:
            group_frame = group_frame.filter(pl.col(key) == group_row.get(key))
        for condition in sorted(
            str(value)
            for value in group_frame["condition"].drop_nulls().unique().to_list()
        ):
            condition_frame = group_frame.filter(pl.col("condition") == condition).sort("recording_id")
            values = condition_frame["_value"].to_numpy() if condition_frame.height else np.asarray([], dtype=np.float64)
            rows.append(
                _base_descriptive_row(
                    spec=spec,
                    config=config,
                    source_manifest=source_manifest,
                    source_manifest_sha256=source_manifest_sha256,
                    source_row_count=source_row_count,
                    group_key=_group_key_payload(group_row, spec.group_keys),
                    condition_name=condition,
                    stats=_descriptive_stats(values),
                )
            )
    return rows


def _descriptive_rows_for_cra_summary_metric(
    *,
    frame: pl.DataFrame,
    spec: MetricSpec,
    config: GoodCopBadCopStatisticsConfig,
    source_manifest: Mapping[str, Any],
    source_manifest_sha256: str,
) -> list[dict[str, Any]]:
    required = {"recording_id", spec.metric_name}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{spec.source_table} is missing required column(s) for {spec.metric_name}: {missing}")
    selected = (
        frame.select(["recording_id", spec.metric_name])
        .rename({spec.metric_name: "_value"})
        .drop_nulls(["recording_id", "_value"])
        .group_by("recording_id")
        .agg(pl.col("_value").mean().alias("_value"))
        .sort("recording_id")
    )
    values = selected["_value"].to_numpy() if selected.height else np.asarray([], dtype=np.float64)
    return [
        _base_descriptive_row(
            spec=spec,
            config=config,
            source_manifest=source_manifest,
            source_manifest_sha256=source_manifest_sha256,
            source_row_count=int(frame.height),
            group_key={},
            condition_name="observed",
            stats=_descriptive_stats(values),
        )
    ]


def _rows_for_metric(
    *,
    frame: pl.DataFrame,
    spec: MetricSpec,
    config: GoodCopBadCopStatisticsConfig,
    source_manifest: Mapping[str, Any],
    source_manifest_sha256: str,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    source_row_count = int(frame.height)
    prepared = _prepare_metric_frame(frame, spec)
    collection = source_manifest.get("collection_manifest")
    if not isinstance(collection, Mapping):
        collection = {}
    source_row_counts = source_manifest.get("row_counts_by_table")
    if not isinstance(source_row_counts, Mapping):
        source_row_counts = {}

    group_rows = (
        prepared.select(list(spec.group_keys)).unique().sort(list(spec.group_keys)).to_dicts()
        if spec.group_keys
        else [{}]
    )
    rows: list[dict[str, Any]] = []
    for group_row in group_rows:
        group_frame = prepared
        for key in spec.group_keys:
            group_frame = group_frame.filter(pl.col(key) == group_row.get(key))
        unit_count = int(group_frame.select("recording_id").unique().height)
        for contrast in config.contrasts:
            a = (
                group_frame.filter(pl.col("condition") == contrast.condition_a)
                .select(["recording_id", pl.col("_value").alias("value_a")])
            )
            b = (
                group_frame.filter(pl.col("condition") == contrast.condition_b)
                .select(["recording_id", pl.col("_value").alias("value_b")])
            )
            paired = a.join(b, on="recording_id", how="inner").sort("recording_id")
            values_a = paired["value_a"].to_numpy() if paired.height else np.asarray([], dtype=np.float64)
            values_b = paired["value_b"].to_numpy() if paired.height else np.asarray([], dtype=np.float64)
            stats = compute_paired_contrast(
                values_a,
                values_b,
                unit_count=unit_count,
                minimum_recordings=int(config.minimum_recordings),
                bootstrap_iterations=int(config.bootstrap_iterations),
                permutation_iterations=int(config.permutation_iterations),
                confidence_level=float(config.confidence_level),
                rng=rng,
            )
            group_key = _group_key_payload(group_row, spec.group_keys)
            parameters = {
                "confidence_level": float(config.confidence_level),
                "minimum_recordings": int(config.minimum_recordings),
                "missing_policy": "paired_complete_recordings",
                "random_seed": int(config.random_seed),
            }
            multiple_comparison_family = "|".join(
                [
                    spec.metric_family,
                    spec.metric_name,
                    contrast.name,
                ]
            )
            row: dict[str, Any] = {
                "export_schema_version": SCHEMA_VERSION,
                "stats_run_id": config.stats_run_id,
                "source_export_run_id": config.source_export_run_id,
                "source_export_manifest_path": str(source_manifest_path(config.export_root, config.source_export_run_id)),
                "source_export_manifest_sha256": source_manifest_sha256,
                "collection_id": collection.get("collection_id"),
                "collection_manifest_sha256": collection.get("manifest_sha256"),
                "source_table": spec.source_table,
                "source_row_count": int(source_row_counts.get(spec.source_table) or source_row_count),
                "metric_family": spec.metric_family,
                "metric_name": spec.metric_name,
                "contrast_name": contrast.name,
                "condition_a": contrast.condition_a,
                "condition_b": contrast.condition_b,
                "group_key_json": _json_dumps(group_key),
                "primary": bool(spec.primary),
                "exploratory": bool(spec.exploratory),
                "unit": "recording",
                "unit_count": stats.unit_count,
                "paired_unit_count": stats.paired_unit_count,
                "excluded_unit_count": stats.excluded_unit_count,
                "missing_policy": "paired_complete_recordings",
                "mean_a": stats.mean_a,
                "mean_b": stats.mean_b,
                "mean_difference": stats.mean_difference,
                "median_difference": stats.median_difference,
                "std_difference": stats.std_difference,
                "effect_size": stats.effect_size,
                "ci_low": stats.ci_low,
                "ci_high": stats.ci_high,
                "p_value": stats.p_value,
                "q_value": None,
                "multiple_comparison_family": multiple_comparison_family,
                "test_method": stats.test_method,
                "bootstrap_iterations": stats.bootstrap_iterations,
                "permutation_iterations": stats.permutation_iterations,
                "status": stats.status,
                "skip_reason": stats.skip_reason,
                "parameters_json": _json_dumps(parameters),
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            row["stat_result_id"] = _result_id(row)
            rows.append(row)
    return rows


def _rows_for_cra_summary_metric(
    *,
    frame: pl.DataFrame,
    spec: MetricSpec,
    config: GoodCopBadCopStatisticsConfig,
    source_manifest: Mapping[str, Any],
    source_manifest_sha256: str,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    required = {"recording_id", spec.metric_name}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{spec.source_table} is missing required column(s) for {spec.metric_name}: {missing}")

    selected = (
        frame.select(["recording_id", spec.metric_name])
        .rename({spec.metric_name: "_value"})
        .drop_nulls(["recording_id", "_value"])
        .group_by("recording_id")
        .agg(pl.col("_value").mean().alias("_value"))
        .sort("recording_id")
    )
    values = selected["_value"].to_numpy() if selected.height else np.asarray([], dtype=np.float64)
    unit_count = int(frame.select("recording_id").drop_nulls().unique().height)
    stats = compute_one_sample_signed_rank(
        values,
        unit_count=unit_count,
        minimum_recordings=int(config.minimum_recordings),
        bootstrap_iterations=int(config.bootstrap_iterations),
        confidence_level=float(config.confidence_level),
        rng=rng,
    )
    collection = source_manifest.get("collection_manifest")
    if not isinstance(collection, Mapping):
        collection = {}
    source_row_counts = source_manifest.get("row_counts_by_table")
    if not isinstance(source_row_counts, Mapping):
        source_row_counts = {}

    parameters = {
        "confidence_level": float(config.confidence_level),
        "minimum_recordings": int(config.minimum_recordings),
        "missing_policy": "one_row_per_recording_complete_cases",
        "random_seed": int(config.random_seed),
        "test_target": 0.0,
    }
    row: dict[str, Any] = {
        "export_schema_version": SCHEMA_VERSION,
        "stats_run_id": config.stats_run_id,
        "source_export_run_id": config.source_export_run_id,
        "source_export_manifest_path": str(source_manifest_path(config.export_root, config.source_export_run_id)),
        "source_export_manifest_sha256": source_manifest_sha256,
        "collection_id": collection.get("collection_id"),
        "collection_manifest_sha256": collection.get("manifest_sha256"),
        "source_table": spec.source_table,
        "source_row_count": int(source_row_counts.get(spec.source_table) or int(frame.height)),
        "metric_family": spec.metric_family,
        "metric_name": spec.metric_name,
        "contrast_name": "vs-zero",
        "condition_a": "zero",
        "condition_b": "observed",
        "group_key_json": _json_dumps({}),
        "primary": bool(spec.primary),
        "exploratory": bool(spec.exploratory),
        "unit": "recording",
        "unit_count": stats.unit_count,
        "paired_unit_count": stats.paired_unit_count,
        "excluded_unit_count": stats.excluded_unit_count,
        "missing_policy": "one_row_per_recording_complete_cases",
        "mean_a": 0.0 if stats.mean_observed is not None else None,
        "mean_b": stats.mean_observed,
        "mean_difference": stats.mean_difference,
        "median_difference": stats.median_difference,
        "std_difference": stats.std_difference,
        "effect_size": stats.effect_size,
        "ci_low": stats.ci_low,
        "ci_high": stats.ci_high,
        "p_value": stats.p_value,
        "q_value": None,
        "multiple_comparison_family": f"{spec.metric_family}|{spec.metric_name}|vs-zero",
        "test_method": stats.test_method,
        "bootstrap_iterations": stats.bootstrap_iterations,
        "permutation_iterations": stats.permutation_iterations,
        "status": stats.status,
        "skip_reason": stats.skip_reason,
        "parameters_json": _json_dumps(parameters),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    row["stat_result_id"] = _result_id(row)
    return [row]


def apply_fdr(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_family: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        if row.get("status") != "computed":
            continue
        by_family.setdefault(str(row.get("multiple_comparison_family") or ""), []).append(index)
    for indices in by_family.values():
        q_values = benjamini_hochberg([_safe_float(rows[index].get("p_value")) for index in indices])
        for index, q_value in zip(indices, q_values):
            rows[index]["q_value"] = q_value
    return rows


def compute_goodcopbadcop_statistics(config: GoodCopBadCopStatisticsConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    export_root = Path(config.export_root).expanduser().resolve()
    source_manifest_file = source_manifest_path(export_root, config.source_export_run_id)
    source_manifest = _json_file(source_manifest_file)
    _require_v2_source_manifest(source_manifest, path=source_manifest_file)
    source_manifest_sha256 = _sha256_file(source_manifest_file)
    rng = np.random.default_rng(int(config.random_seed))
    tables = sorted({spec.source_table for spec in config.metrics})
    frames = {
        table: _read_export_table(export_root, config.source_export_run_id, table)
        for table in tables
    }

    rows: list[dict[str, Any]] = []
    for spec in config.metrics:
        metric_config = GoodCopBadCopStatisticsConfig(
            export_root=export_root,
            source_export_run_id=config.source_export_run_id,
            stats_run_id=config.stats_run_id,
            metrics=config.metrics,
            contrasts=config.contrasts,
            bootstrap_iterations=config.bootstrap_iterations,
            permutation_iterations=config.permutation_iterations,
            confidence_level=config.confidence_level,
            minimum_recordings=config.minimum_recordings,
            random_seed=config.random_seed,
            overwrite=config.overwrite,
        )
        if spec.source_table in {CRA_SUMMARY_TABLE, CRA_NEAR_FIELD_SUMMARY_TABLE}:
            rows.extend(
                _rows_for_cra_summary_metric(
                    frame=frames[spec.source_table],
                    spec=spec,
                    config=metric_config,
                    source_manifest=source_manifest,
                    source_manifest_sha256=source_manifest_sha256,
                    rng=rng,
                )
            )
        else:
            rows.extend(
                _rows_for_metric(
                    frame=frames[spec.source_table],
                    spec=spec,
                    config=metric_config,
                    source_manifest=source_manifest,
                    source_manifest_sha256=source_manifest_sha256,
                    rng=rng,
                )
            )
    apply_fdr(rows)
    rows = [canonicalize_export_row(SUMMARY_TABLE, row) for row in rows]

    manifest = _build_stats_manifest(
        config=GoodCopBadCopStatisticsConfig(
            export_root=export_root,
            source_export_run_id=config.source_export_run_id,
            stats_run_id=config.stats_run_id,
            metrics=config.metrics,
            contrasts=config.contrasts,
            bootstrap_iterations=config.bootstrap_iterations,
            permutation_iterations=config.permutation_iterations,
            confidence_level=config.confidence_level,
            minimum_recordings=config.minimum_recordings,
            random_seed=config.random_seed,
            overwrite=config.overwrite,
        ),
        rows=rows,
        source_manifest=source_manifest,
        source_manifest_sha256=source_manifest_sha256,
    )
    return rows, manifest


def compute_goodcopbadcop_descriptive_summaries(config: GoodCopBadCopStatisticsConfig) -> list[dict[str, Any]]:
    export_root = Path(config.export_root).expanduser().resolve()
    source_manifest_file = source_manifest_path(export_root, config.source_export_run_id)
    source_manifest = _json_file(source_manifest_file)
    _require_v2_source_manifest(source_manifest, path=source_manifest_file)
    source_manifest_sha256 = _sha256_file(source_manifest_file)
    tables = sorted({spec.source_table for spec in config.metrics})
    frames = {
        table: _read_export_table(export_root, config.source_export_run_id, table)
        for table in tables
    }

    rows: list[dict[str, Any]] = []
    for spec in config.metrics:
        metric_config = GoodCopBadCopStatisticsConfig(
            export_root=export_root,
            source_export_run_id=config.source_export_run_id,
            stats_run_id=config.stats_run_id,
            metrics=config.metrics,
            contrasts=config.contrasts,
            bootstrap_iterations=config.bootstrap_iterations,
            permutation_iterations=config.permutation_iterations,
            confidence_level=config.confidence_level,
            minimum_recordings=config.minimum_recordings,
            random_seed=config.random_seed,
            overwrite=config.overwrite,
        )
        if spec.source_table in {CRA_SUMMARY_TABLE, CRA_NEAR_FIELD_SUMMARY_TABLE}:
            rows.extend(
                _descriptive_rows_for_cra_summary_metric(
                    frame=frames[spec.source_table],
                    spec=spec,
                    config=metric_config,
                    source_manifest=source_manifest,
                    source_manifest_sha256=source_manifest_sha256,
                )
            )
        else:
            rows.extend(
                _descriptive_rows_for_metric(
                    frame=frames[spec.source_table],
                    spec=spec,
                    config=metric_config,
                    source_manifest=source_manifest,
                    source_manifest_sha256=source_manifest_sha256,
                )
            )
    return [canonicalize_export_row(DESCRIPTIVE_TABLE, row) for row in rows]


def _build_stats_manifest(
    *,
    config: GoodCopBadCopStatisticsConfig,
    rows: Sequence[Mapping[str, Any]],
    descriptive_rows: Sequence[Mapping[str, Any]] = (),
    source_manifest: Mapping[str, Any],
    source_manifest_sha256: str,
) -> dict[str, Any]:
    git = get_git_info(Path(__file__).resolve().parents[3])
    row_count = len(rows)
    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    output_tables = [SUMMARY_TABLE]
    row_counts_by_table = {SUMMARY_TABLE: row_count}
    if descriptive_rows:
        output_tables.append(DESCRIPTIVE_TABLE)
        row_counts_by_table[DESCRIPTIVE_TABLE] = len(descriptive_rows)
    return {
        "export_run_id": config.stats_run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "schema_id": EXPORT_SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "tool": "fisheye.utils.compute_group_statistics",
        "profile": "chaser",
        "hostname": socket.gethostname(),
        "palette_git_commit": git.get("commit_hash"),
        "palette_git_branch": git.get("branch"),
        "palette_git_dirty": git.get("is_dirty"),
        "source_export_run_id": config.source_export_run_id,
        "source_export_manifest_path": str(source_manifest_path(config.export_root, config.source_export_run_id)),
        "source_export_manifest_sha256": source_manifest_sha256,
        "source_collection_manifest": source_manifest.get("collection_manifest"),
        "input_tables": sorted({spec.source_table for spec in config.metrics}),
        "source_row_counts_by_table": source_manifest.get("row_counts_by_table"),
        "output_tables": output_tables,
        "table_contracts": contract_snapshot(output_tables),
        "row_counts_by_table": row_counts_by_table,
        "status_counts": status_counts,
        "metrics": [
            {
                "metric_family": spec.metric_family,
                "source_table": spec.source_table,
                "metric_name": spec.metric_name,
                "group_keys": list(spec.group_keys),
                "primary": spec.primary,
                "exploratory": spec.exploratory,
            }
            for spec in config.metrics
        ],
        "contrasts": [
            {
                "name": contrast.name,
                "condition_a": contrast.condition_a,
                "condition_b": contrast.condition_b,
            }
            for contrast in config.contrasts
        ],
        "parameters": {
            "bootstrap_iterations": int(config.bootstrap_iterations),
            "permutation_iterations": int(config.permutation_iterations),
            "confidence_level": float(config.confidence_level),
            "minimum_recordings": int(config.minimum_recordings),
            "random_seed": int(config.random_seed),
            "fdr_method": "benjamini_hochberg",
        },
    }


def _normalize_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    columns = sorted({key for row in rows for key in row.keys()})
    return [{column: row.get(column) for column in columns} for row in rows]


def write_goodcopbadcop_statistics(
    rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    *,
    export_root: Path,
    stats_run_id: str,
    descriptive_rows: Sequence[Mapping[str, Any]] = (),
    overwrite: bool = False,
) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    output_root = Path(export_root).expanduser().resolve()

    canonical_rows_by_table = {
        SUMMARY_TABLE: [canonicalize_export_row(SUMMARY_TABLE, row) for row in rows],
    }
    if descriptive_rows:
        canonical_rows_by_table[DESCRIPTIVE_TABLE] = [
            canonicalize_export_row(DESCRIPTIVE_TABLE, row) for row in descriptive_rows
        ]

    def write_table(table_name: str, table_rows: Sequence[Mapping[str, Any]]) -> list[str]:
        table_dir = output_root / "v1" / table_name / f"export_run_id={stats_run_id}"
        if table_dir.exists() and any(table_dir.iterdir()) and not overwrite:
            raise FileExistsError(f"Statistics table directory already exists: {table_dir}")
        table_dir.mkdir(parents=True, exist_ok=True)
        normalized = _normalize_rows(table_rows)
        part_files: list[str] = []
        if normalized:
            columns = tuple(normalized[0])
            missing = validate_table_columns(table_name, columns)
            if missing:
                raise ValueError(
                    f"{table_name} does not satisfy its V2 table contract; "
                    f"missing columns: {list(missing)}"
                )
            part_path = table_dir / "part-00000.parquet"
            tmp_path = table_dir / ".part-00000.parquet.tmp"
            if tmp_path.exists():
                tmp_path.unlink()
            arrow_table = pa.Table.from_pylist(normalized)
            metadata = dict(arrow_table.schema.metadata or {})
            metadata.update(
                {
                    b"palette.export_schema_id": EXPORT_SCHEMA_ID.encode("utf-8"),
                    b"palette.export_schema_version": str(EXPORT_SCHEMA_VERSION).encode("utf-8"),
                    b"palette.table_contract": json.dumps(
                        TABLE_CONTRACTS[table_name].to_dict(),
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8"),
                }
            )
            pq.write_table(arrow_table.replace_schema_metadata(metadata), tmp_path)
            os.replace(tmp_path, part_path)
            part_files.append(str(part_path))
        return part_files

    part_files_by_table: dict[str, list[str]] = {
        SUMMARY_TABLE: write_table(SUMMARY_TABLE, canonical_rows_by_table[SUMMARY_TABLE]),
    }
    if descriptive_rows:
        part_files_by_table[DESCRIPTIVE_TABLE] = write_table(
            DESCRIPTIVE_TABLE,
            canonical_rows_by_table[DESCRIPTIVE_TABLE],
        )

    manifest_dir = output_root / "v1" / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"export_run_id={stats_run_id}.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"Statistics manifest already exists: {manifest_path}")
    payload = dict(manifest)
    output_tables = list(payload.get("output_tables") or [])
    if SUMMARY_TABLE not in output_tables:
        output_tables.append(SUMMARY_TABLE)
    row_counts_by_table = dict(payload.get("row_counts_by_table") or {})
    row_counts_by_table[SUMMARY_TABLE] = len(rows)
    if descriptive_rows:
        if DESCRIPTIVE_TABLE not in output_tables:
            output_tables.append(DESCRIPTIVE_TABLE)
        row_counts_by_table[DESCRIPTIVE_TABLE] = len(descriptive_rows)
    payload["output_tables"] = output_tables
    payload["row_counts_by_table"] = row_counts_by_table
    payload["part_files_by_table"] = part_files_by_table
    payload["schema_id"] = EXPORT_SCHEMA_ID
    payload["schema_version"] = EXPORT_SCHEMA_VERSION
    payload["table_contracts"] = contract_snapshot(output_tables)
    columns_by_table = {
        SUMMARY_TABLE: sorted(
            {key for row in canonical_rows_by_table[SUMMARY_TABLE] for key in row}
        ),
    }
    if descriptive_rows:
        columns_by_table[DESCRIPTIVE_TABLE] = sorted(
            {key for row in canonical_rows_by_table[DESCRIPTIVE_TABLE] for key in row}
        )
    capability_statuses = resolve_capabilities(columns_by_table)
    payload["capabilities"] = [
        status.capability_id for status in capability_statuses if status.available
    ]
    payload["capability_statuses"] = [status.to_dict() for status in capability_statuses]
    payload["manifest_path"] = str(manifest_path)
    tmp_manifest = manifest_path.with_suffix(".json.tmp")
    tmp_manifest.write_text(json.dumps(json_attr_safe(payload), indent=2, sort_keys=True) + "\n")
    os.replace(tmp_manifest, manifest_path)
    return payload


def metric_specs_for_families(families: Sequence[str] | None) -> tuple[MetricSpec, ...]:
    if not families:
        return DEFAULT_METRICS
    wanted = {str(item).strip() for item in families if str(item).strip()}
    unknown = wanted - {spec.metric_family for spec in DEFAULT_METRICS}
    if unknown:
        known = ", ".join(sorted({spec.metric_family for spec in DEFAULT_METRICS}))
        raise ValueError(f"Unknown metric family/families: {', '.join(sorted(unknown))}. Known: {known}")
    return tuple(spec for spec in DEFAULT_METRICS if spec.metric_family in wanted)


def contrast_definitions(names: Sequence[str] | None) -> tuple[ContrastDefinition, ...]:
    if not names:
        return DEFAULT_CONTRASTS
    known = {contrast.name: contrast for contrast in DEFAULT_CONTRASTS}
    unknown = [name for name in names if name not in known]
    if unknown:
        raise ValueError(f"Unknown contrast(s): {', '.join(unknown)}. Known: {', '.join(sorted(known))}")
    return tuple(known[name] for name in names)
