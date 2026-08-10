"""GoodCopBadCop profile for cross-recording group statistics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import socket
from typing import Any, Mapping, Sequence
import uuid

import numpy as np
import polars as pl

from fisheye.analytics_exports.arrow_contracts import (
    ARROW_TABLE_CONTRACTS,
    arrow_contract_envelope,
    exact_arrow_schema,
)
from fisheye.analytics_exports.capabilities import resolve_capabilities
from fisheye.analytics_exports.contracts import (
    CHASER_NEAR_FIELD_OCCUPANCY_CHASER_PHASE_TABLE,
    CHASER_NEAR_FIELD_OCCUPANCY_SUMMARY_TABLE,
    CHASER_QUADRANT_OCCUPANCY_CHASER_PHASE_TABLE,
    CHASER_QUADRANT_OCCUPANCY_SUMMARY_TABLE,
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
from fisheye.analytics_exports.publication import (
    PUBLICATION_SCHEMA_ID,
    PUBLICATION_SCHEMA_VERSION,
    export_manifest_path,
    generation_relative_path,
    manifest_commit_lock,
    manifest_identity,
    manifest_selected_part_files_from_payload,
    publication_generation_root,
    publication_staging_root,
    safe_component,
    sha256_file,
    validate_staged_publication,
)
from fisheye.analytics_exports.validation import validate_export_payload
from fisheye.shared.json_safety import json_attr_safe, strict_json_dumps
from fisheye.shared.system_metadata import get_git_info

from .paired import (
    ContrastDefinition,
    benjamini_hochberg,
    compute_one_sample_signed_rank,
    compute_paired_contrast,
)
from .session_cluster import (
    DEFAULT_MINIMUM_SESSIONS,
    SESSION_RANDOM_INTERCEPT_METHOD,
    SessionClusterResult,
    fit_session_random_intercept,
)


SUMMARY_TABLE = STATISTICS_TABLE
CRA_SUMMARY_TABLE = CHASER_QUADRANT_OCCUPANCY_SUMMARY_TABLE
CRA_OBJECT_PHASE_TABLE = CHASER_QUADRANT_OCCUPANCY_CHASER_PHASE_TABLE
CRA_NEAR_FIELD_SUMMARY_TABLE = CHASER_NEAR_FIELD_OCCUPANCY_SUMMARY_TABLE
CRA_NEAR_FIELD_OBJECT_PHASE_TABLE = CHASER_NEAR_FIELD_OCCUPANCY_CHASER_PHASE_TABLE
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
    primary: bool
    exploratory: bool


_METRIC_UNITS: dict[tuple[str, str], str] = {
    (CHASER_DISTANCE_SUMMARY_TABLE, "mean_distance_mm"): "mm",
    (CHASER_DISTANCE_SUMMARY_TABLE, "p50_distance_mm"): "mm",
    (CHASER_DISTANCE_SUMMARY_TABLE, "fraction_within_threshold"): "fraction",
    (CHASER_SPATIAL_TABLE, "time_s"): "s",
    (CHASER_SPATIAL_TABLE, "fraction_of_epoch"): "fraction",
    (CHASER_SPATIAL_TABLE, "fraction_of_detected"): "fraction",
    (EPOCH_BEHAVIOR_TABLE, "mean_speed_mm_s"): "mm/s",
    (EPOCH_BEHAVIOR_TABLE, "bout_rate_per_min"): "1/min",
    (EPOCH_BEHAVIOR_TABLE, "mean_bout_duration_s"): "s",
    (EPOCH_BEHAVIOR_TABLE, "mean_bout_path_length_mm"): "mm",
    (EPOCH_BEHAVIOR_TABLE, "mean_bout_net_heading_change_deg"): "deg",
    (EPOCH_BEHAVIOR_TABLE, "mean_abs_bout_net_heading_change_deg"): "deg",
    (EPOCH_BEHAVIOR_TABLE, "mean_bout_heading_path_deg"): "deg",
    (EPOCH_BEHAVIOR_TABLE, "inter_bout_interval_count"): "count",
    (EPOCH_BEHAVIOR_TABLE, "wall_fraction"): "fraction",
    (EPOCH_BEHAVIOR_TABLE, "median_distance_from_arena_center_mm"): "mm",
    (EPOCH_BEHAVIOR_TABLE, "mean_inter_bout_interval_s"): "s",
    (EPOCH_BEHAVIOR_TABLE, "median_inter_bout_interval_s"): "s",
    (CRA_SUMMARY_TABLE, "delta_agg"): "mm",
    (CRA_SUMMARY_TABLE, "delta_occ_agg"): "fraction",
    (CRA_SUMMARY_TABLE, "specificity_distance"): "mm",
    (CRA_SUMMARY_TABLE, "specificity_occupancy"): "fraction",
    (CRA_SUMMARY_TABLE, "delta_inert"): "mm",
    (CRA_SUMMARY_TABLE, "delta_occ_inert"): "fraction",
    (CRA_NEAR_FIELD_SUMMARY_TABLE, "approach_p05_delta_agg"): "mm",
    (CRA_NEAR_FIELD_SUMMARY_TABLE, "approach_p05_specificity"): "mm",
    (CRA_NEAR_FIELD_SUMMARY_TABLE, "nearzone_occ_delta_agg"): "fraction",
    (CRA_NEAR_FIELD_SUMMARY_TABLE, "nearzone_occ_specificity"): "fraction",
    (CRA_NEAR_FIELD_SUMMARY_TABLE, "nearzone_entry_rate_delta_agg"): "1/min",
    (CRA_NEAR_FIELD_SUMMARY_TABLE, "nearzone_entry_rate_specificity"): "1/min",
    (CRA_NEAR_FIELD_SUMMARY_TABLE, "approach_p05_delta_inert"): "mm",
    (CRA_NEAR_FIELD_SUMMARY_TABLE, "nearzone_occ_delta_inert"): "fraction",
    (CRA_NEAR_FIELD_SUMMARY_TABLE, "nearzone_entry_rate_delta_inert"): "1/min",
    (CHASER_EGOCENTRIC_SUMMARY_TABLE, "mean_alignment_cos"): "dimensionless",
    (CHASER_EGOCENTRIC_SUMMARY_TABLE, "mean_lateral_sin"): "dimensionless",
    (CHASER_EGOCENTRIC_SUMMARY_TABLE, "fraction_front_45"): "fraction",
    (CHASER_EGOCENTRIC_SUMMARY_TABLE, "fraction_lateral_45"): "fraction",
    (CHASER_EGOCENTRIC_SUMMARY_TABLE, "fraction_behind_45"): "fraction",
}


def _metric_unit(spec: MetricSpec) -> str:
    return _metric_unit_for_values(spec.source_table, spec.metric_name)


def _metric_unit_for_values(source_table: object, metric_name: object) -> str:
    try:
        return _METRIC_UNITS[(str(source_table), str(metric_name))]
    except KeyError as exc:
        raise ValueError(
            "No exact measurement unit is registered for "
            f"{source_table}.{metric_name}"
        ) from exc


DEFAULT_METRICS: tuple[MetricSpec, ...] = (
    MetricSpec(
        metric_family="chaser_distance",
        source_table=CHASER_DISTANCE_SUMMARY_TABLE,
        metric_name="mean_distance_mm",
        group_keys=("behavior_class",),
        primary=True,
        exploratory=False,
    ),
    MetricSpec(
        metric_family="chaser_distance",
        source_table=CHASER_DISTANCE_SUMMARY_TABLE,
        metric_name="p50_distance_mm",
        group_keys=("behavior_class",),
        primary=True,
        exploratory=False,
    ),
    MetricSpec(
        metric_family="chaser_distance",
        source_table=CHASER_DISTANCE_SUMMARY_TABLE,
        metric_name="fraction_within_threshold",
        group_keys=("behavior_class",),
        primary=True,
        exploratory=False,
    ),
    MetricSpec(
        metric_family="spatial_occupancy",
        source_table=CHASER_SPATIAL_TABLE,
        metric_name="time_s",
        group_keys=("zone_set_id", "zone_id", "zone_label"),
        primary=True,
        exploratory=False,
    ),
    MetricSpec(
        metric_family="spatial_occupancy",
        source_table=CHASER_SPATIAL_TABLE,
        metric_name="fraction_of_epoch",
        group_keys=("zone_set_id", "zone_id", "zone_label"),
        primary=True,
        exploratory=False,
    ),
    MetricSpec(
        metric_family="spatial_occupancy",
        source_table=CHASER_SPATIAL_TABLE,
        metric_name="fraction_of_detected",
        group_keys=("zone_set_id", "zone_id", "zone_label"),
        primary=True,
        exploratory=False,
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
        exploratory=True,
    ),
    MetricSpec(
        metric_family="cra_primary_endpoint",
        source_table=CRA_SUMMARY_TABLE,
        metric_name="delta_occ_agg",
        group_keys=(),
        primary=False,
        exploratory=True,
    ),
    MetricSpec(
        metric_family="cra_primary_endpoint",
        source_table=CRA_SUMMARY_TABLE,
        metric_name="specificity_distance",
        group_keys=(),
        primary=True,
        exploratory=False,
    ),
    MetricSpec(
        metric_family="cra_primary_endpoint",
        source_table=CRA_SUMMARY_TABLE,
        metric_name="specificity_occupancy",
        group_keys=(),
        primary=False,
        exploratory=True,
    ),
    MetricSpec(
        metric_family="cra_primary_endpoint",
        source_table=CRA_SUMMARY_TABLE,
        metric_name="delta_inert",
        group_keys=(),
        primary=False,
        exploratory=True,
    ),
    MetricSpec(
        metric_family="cra_primary_endpoint",
        source_table=CRA_SUMMARY_TABLE,
        metric_name="delta_occ_inert",
        group_keys=(),
        primary=False,
        exploratory=True,
    ),
    MetricSpec(
        metric_family="cra_near_field",
        source_table=CRA_NEAR_FIELD_SUMMARY_TABLE,
        metric_name="approach_p05_delta_agg",
        group_keys=(),
        primary=True,
        exploratory=False,
    ),
    MetricSpec(
        metric_family="cra_near_field",
        source_table=CRA_NEAR_FIELD_SUMMARY_TABLE,
        metric_name="approach_p05_specificity",
        group_keys=(),
        primary=True,
        exploratory=False,
    ),
    MetricSpec(
        metric_family="cra_near_field",
        source_table=CRA_NEAR_FIELD_SUMMARY_TABLE,
        metric_name="nearzone_occ_delta_agg",
        group_keys=(),
        primary=True,
        exploratory=False,
    ),
    MetricSpec(
        metric_family="cra_near_field",
        source_table=CRA_NEAR_FIELD_SUMMARY_TABLE,
        metric_name="nearzone_occ_specificity",
        group_keys=(),
        primary=True,
        exploratory=False,
    ),
    MetricSpec(
        metric_family="cra_near_field",
        source_table=CRA_NEAR_FIELD_SUMMARY_TABLE,
        metric_name="nearzone_entry_rate_delta_agg",
        group_keys=(),
        primary=True,
        exploratory=False,
    ),
    MetricSpec(
        metric_family="cra_near_field",
        source_table=CRA_NEAR_FIELD_SUMMARY_TABLE,
        metric_name="nearzone_entry_rate_specificity",
        group_keys=(),
        primary=True,
        exploratory=False,
    ),
    MetricSpec(
        metric_family="cra_near_field",
        source_table=CRA_NEAR_FIELD_SUMMARY_TABLE,
        metric_name="approach_p05_delta_inert",
        group_keys=(),
        primary=False,
        exploratory=True,
    ),
    MetricSpec(
        metric_family="cra_near_field",
        source_table=CRA_NEAR_FIELD_SUMMARY_TABLE,
        metric_name="nearzone_occ_delta_inert",
        group_keys=(),
        primary=False,
        exploratory=True,
    ),
    MetricSpec(
        metric_family="cra_near_field",
        source_table=CRA_NEAR_FIELD_SUMMARY_TABLE,
        metric_name="nearzone_entry_rate_delta_inert",
        group_keys=(),
        primary=False,
        exploratory=True,
    ),
    MetricSpec(
        metric_family="egocentric_alignment",
        source_table=CHASER_EGOCENTRIC_SUMMARY_TABLE,
        metric_name="mean_alignment_cos",
        group_keys=("behavior_class",),
        primary=True,
        exploratory=False,
    ),
    MetricSpec(
        metric_family="egocentric_alignment",
        source_table=CHASER_EGOCENTRIC_SUMMARY_TABLE,
        metric_name="mean_lateral_sin",
        group_keys=("behavior_class",),
        primary=True,
        exploratory=False,
    ),
    MetricSpec(
        metric_family="egocentric_alignment",
        source_table=CHASER_EGOCENTRIC_SUMMARY_TABLE,
        metric_name="fraction_front_45",
        group_keys=("behavior_class",),
        primary=True,
        exploratory=False,
    ),
    MetricSpec(
        metric_family="egocentric_alignment",
        source_table=CHASER_EGOCENTRIC_SUMMARY_TABLE,
        metric_name="fraction_lateral_45",
        group_keys=("behavior_class",),
        primary=True,
        exploratory=False,
    ),
    MetricSpec(
        metric_family="egocentric_alignment",
        source_table=CHASER_EGOCENTRIC_SUMMARY_TABLE,
        metric_name="fraction_behind_45",
        group_keys=("behavior_class",),
        primary=True,
        exploratory=False,
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
    minimum_sessions: int = DEFAULT_MINIMUM_SESSIONS
    random_seed: int = 0
    cluster: str = "session"
    overwrite: bool = False
    allow_legacy_export_layout: bool = False

    def __post_init__(self) -> None:
        if type(self.minimum_recordings) is not int or self.minimum_recordings < 1:
            raise ValueError("minimum_recordings must be an integer >= 1")
        if type(self.minimum_sessions) is not int or self.minimum_sessions < 3:
            raise ValueError("minimum_sessions must be an integer >= 3")
        if self.cluster not in {"none", "session"}:
            raise ValueError("cluster must be 'none' or 'session'")
        for spec in self.metrics:
            _analysis_tier(spec)


def _analysis_tier(spec: MetricSpec) -> str:
    if type(spec.primary) is not bool or type(spec.exploratory) is not bool:
        raise ValueError(
            f"Metric {spec.source_table}.{spec.metric_name} tier flags must be booleans."
        )
    if bool(spec.primary) == bool(spec.exploratory):
        raise ValueError(
            f"Metric {spec.source_table}.{spec.metric_name} must be exactly one of "
            "primary or exploratory."
        )
    return "primary" if spec.primary else "exploratory"


def _multiple_comparison_family(spec: MetricSpec) -> str:
    return f"{_analysis_tier(spec)}|{spec.metric_family}"


def _disabled_cluster_result(unit_count: int) -> SessionClusterResult:
    return SessionClusterResult(
        status="disabled",
        reason=None,
        method="none",
        unit="session",
        unit_count=int(unit_count),
        cluster_count=0,
        mean=None,
        standard_error=None,
        ci_low=None,
        ci_high=None,
        p_value=None,
        cluster_variance=None,
        residual_variance=None,
        intraclass_correlation=None,
    )


def _session_cluster_result(
    *,
    config: GoodCopBadCopStatisticsConfig,
    values: np.ndarray,
    session_ids: np.ndarray,
    naive_status: str,
    naive_skip_reason: str | None,
) -> SessionClusterResult:
    if config.cluster == "none":
        return _disabled_cluster_result(int(values.size))
    if naive_status != "computed":
        numeric_values = np.asarray(values, dtype=np.float64).reshape(-1)
        raw_sessions = np.asarray(session_ids, dtype=object).reshape(-1)
        if numeric_values.shape != raw_sessions.shape:
            raise ValueError(
                "Cluster values and session identities must have equal length."
            )
        normalized_session_values = np.asarray(
            [
                str(value).strip() if value is not None else ""
                for value in raw_sessions
            ],
            dtype=object,
        )
        usable = np.isfinite(numeric_values) & (normalized_session_values != "")
        normalized_sessions = {
            str(value) for value in normalized_session_values[usable]
        }
        return SessionClusterResult(
            status="unavailable",
            reason=(
                "naive_inference_ineligible:"
                f"{naive_skip_reason or naive_status}"
            ),
            method=SESSION_RANDOM_INTERCEPT_METHOD,
            unit="session",
            unit_count=int(np.count_nonzero(usable)),
            cluster_count=len(normalized_sessions),
            mean=None,
            standard_error=None,
            ci_low=None,
            ci_high=None,
            p_value=None,
            cluster_variance=None,
            residual_variance=None,
            intraclass_correlation=None,
        )
    return fit_session_random_intercept(
        values,
        session_ids,
        confidence_level=float(config.confidence_level),
        minimum_sessions=int(config.minimum_sessions),
    )


def _cluster_fields(result: SessionClusterResult) -> dict[str, Any]:
    return {
        "cluster_mode": "none" if result.status == "disabled" else "session",
        "cluster_unit": result.unit,
        "cluster_method": result.method,
        "cluster_status": result.status,
        "cluster_reason": result.reason,
        "cluster_count": result.cluster_count,
        "clustered_unit_count": result.unit_count,
        "clustered_mean_difference": result.mean,
        "clustered_standard_error": result.standard_error,
        "clustered_ci_low": result.ci_low,
        "clustered_ci_high": result.ci_high,
        "clustered_p_value": result.p_value,
        "clustered_q_value": None,
        "session_variance": result.cluster_variance,
        "residual_variance": result.residual_variance,
        "intraclass_correlation": result.intraclass_correlation,
    }


def _test_count_status(test_count: int) -> str:
    if test_count == 0:
        return "no_eligible_tests"
    if test_count == 1:
        return "singleton_test"
    return "multiple_tests"


def _fdr_family_test_registry(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        family_id = str(row.get("multiple_comparison_family") or "")
        grouped.setdefault(family_id, []).append(row)

    registry: list[dict[str, Any]] = []
    for family_id, family_rows in sorted(grouped.items()):
        try:
            analysis_tier, metric_family = family_id.split("|", 1)
        except ValueError as exc:
            raise ValueError(
                f"Invalid multiple-comparison family identifier: {family_id!r}"
            ) from exc
        naive_test_count = sum(
            row.get("status") == "computed" and row.get("p_value") is not None
            for row in family_rows
        )
        clustered_test_count = sum(
            row.get("status") == "computed"
            and row.get("cluster_status") in {"computed", "boundary_zero_variance"}
            and row.get("clustered_p_value") is not None
            for row in family_rows
        )
        registry.append(
            {
                "family_id": family_id,
                "analysis_tier": analysis_tier,
                "metric_family": metric_family,
                "result_count": len(family_rows),
                "naive_test_count": naive_test_count,
                "naive_test_status": _test_count_status(naive_test_count),
                "clustered_test_count": clustered_test_count,
                "clustered_test_status": _test_count_status(
                    clustered_test_count
                ),
            }
        )
    return registry


def _validate_fdr_family_test_registry(
    declared: object,
    *,
    rows: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    if not isinstance(declared, list):
        raise ValueError("Statistics manifest lacks FDR family test counts")
    exact_fields = {
        "family_id",
        "analysis_tier",
        "metric_family",
        "result_count",
        "naive_test_count",
        "naive_test_status",
        "clustered_test_count",
        "clustered_test_status",
    }
    previous_family = ""
    for item in declared:
        if not isinstance(item, Mapping) or set(item) != exact_fields:
            raise ValueError("Statistics manifest FDR family test counts are invalid")
        family_id = item.get("family_id")
        tier = item.get("analysis_tier")
        metric_family = item.get("metric_family")
        if (
            not isinstance(family_id, str)
            or not family_id
            or family_id <= previous_family
            or tier not in {"primary", "exploratory"}
            or not isinstance(metric_family, str)
            or not metric_family
            or family_id != f"{tier}|{metric_family}"
        ):
            raise ValueError("Statistics manifest FDR family identity is invalid")
        previous_family = family_id
        for count_name, status_name in (
            ("naive_test_count", "naive_test_status"),
            ("clustered_test_count", "clustered_test_status"),
        ):
            count = item.get(count_name)
            if type(count) is not int or count < 0:
                raise ValueError("Statistics manifest FDR family test count is invalid")
            if item.get(status_name) != _test_count_status(count):
                raise ValueError("Statistics manifest FDR family test status is invalid")
        result_count = item.get("result_count")
        if (
            type(result_count) is not int
            or result_count < 1
            or item["naive_test_count"] > result_count
            or item["clustered_test_count"] > result_count
        ):
            raise ValueError("Statistics manifest FDR family result count is invalid")
    if rows is not None and declared != _fdr_family_test_registry(rows):
        raise ValueError("Statistics manifest FDR family test counts do not match rows")


def utc_run_id() -> str:
    return "stats_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def source_manifest_path(export_root: Path, source_export_run_id: str) -> Path:
    return export_manifest_path(export_root, source_export_run_id)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json_snapshot(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload, hashlib.sha256(raw).hexdigest()


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


def _read_export_table(
    export_root: Path,
    source_manifest: Mapping[str, Any],
    table: str,
    *,
    allow_legacy_layout: bool = False,
) -> pl.DataFrame:
    files = manifest_selected_part_files_from_payload(
        export_root,
        source_manifest,
        table,
        allow_legacy_layout=allow_legacy_layout,
    )
    if not files:
        raise FileNotFoundError(
            f"Manifest selects no Parquet parts for {table}: "
            f"{source_manifest.get('export_run_id')}"
        )
    return pl.scan_parquet([str(path) for path in files]).collect()


ROLE_GROUPED_SOURCE_TABLES = {
    CHASER_DISTANCE_SUMMARY_TABLE,
    CHASER_EGOCENTRIC_SUMMARY_TABLE,
}


def _statistics_input_tables(metrics: Sequence[MetricSpec]) -> list[str]:
    tables = {spec.source_table for spec in metrics}
    if any(spec.source_table == CRA_SUMMARY_TABLE for spec in metrics):
        tables.add(CRA_OBJECT_PHASE_TABLE)
    if any(spec.source_table == CRA_NEAR_FIELD_SUMMARY_TABLE for spec in metrics):
        tables.add(CRA_NEAR_FIELD_OBJECT_PHASE_TABLE)
    if any(
        spec.source_table in ROLE_GROUPED_SOURCE_TABLES
        and "behavior_class" in spec.group_keys
        for spec in metrics
    ):
        tables.add(CRA_OBJECT_PHASE_TABLE)
    return sorted(tables)


def _derive_profile_role_contrasts(
    frames: dict[str, pl.DataFrame],
    metrics: Sequence[MetricSpec],
) -> None:
    """Add protocol-profile contrasts to generic recording summaries.

    The recording contracts remain N-chaser and do not persist a privileged
    aggressive/inert comparison.  This GoodCopBadCop statistics profile owns
    that comparison and averages repeated chasers within a role before taking
    pre/post differences.
    """

    requested_by_table: dict[str, set[str]] = {}
    for spec in metrics:
        if spec.source_table in {CRA_SUMMARY_TABLE, CRA_NEAR_FIELD_SUMMARY_TABLE}:
            requested_by_table.setdefault(spec.source_table, set()).add(
                spec.metric_name
            )

    def phase_kind(row: Mapping[str, Any]) -> str:
        for key in ("phase_label", "source_window_label", "window_label"):
            label = str(row.get(key) or "").strip().lower()
            if label.startswith("pre"):
                return "pre"
            if label.startswith("post"):
                return "post"
        return ""

    def derive(
        *,
        summary_table: str,
        phase_table: str,
        source_metrics: Mapping[str, tuple[str, str, str]],
    ) -> None:
        requested = requested_by_table.get(summary_table, set())
        if not requested:
            return
        summary = frames[summary_table]
        phase = frames[phase_table]
        required = {
            "recording_id",
            "object_role",
            "phase_label",
            *(spec[0] for spec in source_metrics.values()),
        }
        missing = sorted(required - set(phase.columns))
        if missing:
            raise ValueError(
                f"{phase_table} is missing columns required for profile contrasts: {missing}"
            )
        grouped: dict[tuple[str, str, str, str], list[float]] = {}
        for row in phase.to_dicts():
            recording_id = str(row.get("recording_id") or "")
            role = (
                str(row.get("object_role") or row.get("behavior_class") or "")
                .strip()
                .lower()
            )
            kind = phase_kind(row)
            if (
                not recording_id
                or role not in {"aggressive", "inert"}
                or kind not in {"pre", "post"}
            ):
                continue
            for analysis_key, (
                source_column,
                _delta_stem,
                _specificity_name,
            ) in source_metrics.items():
                value = _safe_float(row.get(source_column))
                if value is not None:
                    grouped.setdefault(
                        (recording_id, role, kind, analysis_key), []
                    ).append(value)

        def mean(key: tuple[str, str, str, str]) -> float | None:
            values = grouped.get(key, [])
            return float(sum(values) / len(values)) if values else None

        derived_by_recording: dict[str, dict[str, float | None]] = {}
        for recording_id in (
            summary.get_column("recording_id").cast(pl.String).to_list()
        ):
            values: dict[str, float | None] = {}
            for role, suffix in (("aggressive", "agg"), ("inert", "inert")):
                for analysis_key, (
                    _source_column,
                    delta_stem,
                    _specificity_name,
                ) in source_metrics.items():
                    pre = mean((recording_id, role, "pre", analysis_key))
                    post = mean((recording_id, role, "post", analysis_key))
                    values[f"{delta_stem}_{suffix}"] = (
                        None if pre is None or post is None else post - pre
                    )
            for _analysis_key, (
                _source_column,
                delta_stem,
                specificity_name,
            ) in source_metrics.items():
                aggressive = values.get(f"{delta_stem}_agg")
                inert = values.get(f"{delta_stem}_inert")
                values[specificity_name] = (
                    None if aggressive is None or inert is None else aggressive - inert
                )
            derived_by_recording[recording_id] = values

        additions: list[pl.Series] = []
        recording_ids = summary.get_column("recording_id").cast(pl.String).to_list()
        for metric_name in sorted(requested):
            if metric_name in summary.columns:
                continue
            additions.append(
                pl.Series(
                    metric_name,
                    [
                        derived_by_recording.get(recording_id, {}).get(metric_name)
                        for recording_id in recording_ids
                    ],
                    dtype=pl.Float64,
                )
            )
        if additions:
            frames[summary_table] = summary.with_columns(additions)

    derive(
        summary_table=CRA_SUMMARY_TABLE,
        phase_table=CRA_OBJECT_PHASE_TABLE,
        source_metrics={
            "distance": ("median_distance_mm", "delta", "specificity_distance"),
            "occupancy": ("occupancy_fraction", "delta_occ", "specificity_occupancy"),
        },
    )
    derive(
        summary_table=CRA_NEAR_FIELD_SUMMARY_TABLE,
        phase_table=CRA_NEAR_FIELD_OBJECT_PHASE_TABLE,
        source_metrics={
            "approach_p05": (
                "approach_p05_mm",
                "approach_p05_delta",
                "approach_p05_specificity",
            ),
            "nearzone_occ": (
                "near_zone_occupancy_fraction",
                "nearzone_occ_delta",
                "nearzone_occ_specificity",
            ),
            "nearzone_entry_rate": (
                "near_zone_entry_rate_per_min",
                "nearzone_entry_rate_delta",
                "nearzone_entry_rate_specificity",
            ),
        },
    )


def _enrich_role_grouped_frames(
    frames: dict[str, pl.DataFrame],
    metrics: Sequence[MetricSpec],
) -> None:
    role_grouped_tables = {
        spec.source_table
        for spec in metrics
        if spec.source_table in ROLE_GROUPED_SOURCE_TABLES
        and "behavior_class" in spec.group_keys
    }
    if not role_grouped_tables:
        return
    role_frame = frames.get(CRA_OBJECT_PHASE_TABLE)
    if role_frame is None:
        raise ValueError(
            f"Role-grouped statistics require {CRA_OBJECT_PHASE_TABLE}"
        )
    required_role_columns = {
        "recording_id",
        "object_column_index",
        "object_role",
    }
    missing_role_columns = sorted(required_role_columns - set(role_frame.columns))
    if missing_role_columns:
        raise ValueError(
            f"{CRA_OBJECT_PHASE_TABLE} is missing role mapping columns: "
            f"{missing_role_columns}"
        )
    role_rows = (
        role_frame.select(
            [
                "recording_id",
                pl.col("object_column_index").alias("chaser_column_index"),
                pl.col("object_role")
                .cast(pl.String)
                .str.strip_chars()
                .str.to_lowercase()
                .alias("_resolved_behavior_class"),
            ]
        )
        .drop_nulls(
            [
                "recording_id",
                "chaser_column_index",
                "_resolved_behavior_class",
            ]
        )
        .unique()
    )
    conflicts = (
        role_rows.group_by(["recording_id", "chaser_column_index"])
        .agg(pl.col("_resolved_behavior_class").n_unique().alias("role_count"))
        .filter(pl.col("role_count") > 1)
    )
    if conflicts.height:
        raise ValueError(
            "CRA object-role mapping is not unique by recording and object column"
        )
    role_map = role_rows.group_by(
        ["recording_id", "chaser_column_index"]
    ).agg(pl.col("_resolved_behavior_class").first())

    for table in role_grouped_tables:
        frame = frames[table]
        required = {"recording_id", "chaser_column_index", "behavior_class"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{table} is missing role-grouping columns: {missing}")
        joined = frame.join(
            role_map,
            on=["recording_id", "chaser_column_index"],
            how="left",
        ).with_columns(
            pl.col("behavior_class")
            .cast(pl.String)
            .str.strip_chars()
            .str.to_lowercase()
        )
        disagreements = joined.filter(
            pl.col("behavior_class").is_not_null()
            & (pl.col("behavior_class") != "unknown")
            & pl.col("_resolved_behavior_class").is_not_null()
            & (pl.col("behavior_class") != pl.col("_resolved_behavior_class"))
        )
        if disagreements.height:
            raise ValueError(
                f"{table} behavior_class conflicts with CRA object-role mapping"
            )
        joined = joined.with_columns(
            pl.when(
                pl.col("behavior_class").is_null()
                | (pl.col("behavior_class") == "unknown")
            )
            .then(pl.col("_resolved_behavior_class"))
            .otherwise(pl.col("behavior_class"))
            .alias("behavior_class")
        )
        unresolved = joined.filter(
            pl.col("behavior_class").is_null()
            | (pl.col("behavior_class") == "unknown")
        )
        if unresolved.height:
            raise ValueError(
                f"{table} contains chaser rows without a resolvable behavior role"
            )
        frames[table] = joined.drop("_resolved_behavior_class")


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
        "metric_unit": row.get("metric_unit"),
        "source_table": row.get("source_table"),
        "contrast_name": row.get("contrast_name"),
        "condition_a": row.get("condition_a"),
        "condition_b": row.get("condition_b"),
        "group_key_json": row.get("group_key_json"),
        "effect_size_kind": row.get("effect_size_kind"),
        "ci_estimand": row.get("ci_estimand"),
        "missing_policy": row.get("missing_policy"),
        "multiple_comparison_family": row.get("multiple_comparison_family"),
        "parameters_json": row.get("parameters_json"),
        "test_method": row.get("test_method"),
        "bootstrap_iterations": row.get("bootstrap_iterations"),
        "permutation_iterations": row.get("permutation_iterations"),
        "source_export_run_id": row.get("source_export_run_id"),
        "source_export_manifest_sha256": row.get(
            "source_export_manifest_sha256"
        ),
    }
    return hashlib.sha256(_json_dumps(payload).encode("utf-8")).hexdigest()


def _descriptive_result_id(row: Mapping[str, Any]) -> str:
    payload = {
        "metric_family": row.get("metric_family"),
        "metric_name": row.get("metric_name"),
        "metric_unit": row.get("metric_unit"),
        "source_table": row.get("source_table"),
        "condition_name": row.get("condition_name"),
        "group_key_json": row.get("group_key_json"),
        "unit": row.get("unit"),
        "missing_policy": row.get("missing_policy"),
        "parameters_json": row.get("parameters_json"),
        "source_export_run_id": row.get("source_export_run_id"),
        "source_export_manifest_sha256": row.get(
            "source_export_manifest_sha256"
        ),
    }
    return hashlib.sha256(_json_dumps(payload).encode("utf-8")).hexdigest()


def _validate_result_rows(
    table_name: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_stats_run_id: str | None = None,
    expected_source_export_run_id: str | None = None,
    expected_source_manifest_sha256: str | None = None,
) -> None:
    if table_name not in {SUMMARY_TABLE, DESCRIPTIVE_TABLE}:
        raise ValueError(f"Unsupported group-statistics result table: {table_name}")
    id_field = (
        "stat_result_id" if table_name == SUMMARY_TABLE else "descriptive_result_id"
    )
    seen: set[str] = set()
    for row_index, row in enumerate(rows):
        for field_name, expected in (
            ("stats_run_id", expected_stats_run_id),
            ("source_export_run_id", expected_source_export_run_id),
            (
                "source_export_manifest_sha256",
                expected_source_manifest_sha256,
            ),
        ):
            if expected is not None and row.get(field_name) != expected:
                raise ValueError(
                    f"{table_name}: row {row_index} has an invalid {field_name}"
                )
        source_digest = row.get("source_export_manifest_sha256")
        if (
            not isinstance(source_digest, str)
            or len(source_digest) != 64
            or source_digest != source_digest.lower()
        ):
            raise ValueError(
                f"{table_name}: row {row_index} has an invalid source manifest digest"
            )
        try:
            int(source_digest, 16)
        except ValueError as exc:
            raise ValueError(
                f"{table_name}: row {row_index} has an invalid source manifest digest"
            ) from exc
        expected_unit = _metric_unit_for_values(
            row.get("source_table"), row.get("metric_name")
        )
        if row.get("metric_unit") != expected_unit:
            raise ValueError(
                f"{table_name}: row {row_index} has an invalid metric_unit"
            )
        if row.get("unit") != "recording":
            raise ValueError(f"{table_name}: row {row_index} has invalid unit")
        try:
            group_key = json.loads(str(row.get("group_key_json")))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"{table_name}: row {row_index} has invalid group_key_json"
            ) from exc
        if not isinstance(group_key, dict) or _json_dumps(group_key) != row.get(
            "group_key_json"
        ):
            raise ValueError(
                f"{table_name}: row {row_index} has noncanonical group_key_json"
            )
        expected_id = (
            _result_id(row)
            if table_name == SUMMARY_TABLE
            else _descriptive_result_id(row)
        )
        actual_id = row.get(id_field)
        if actual_id != expected_id:
            raise ValueError(
                f"{table_name}: row {row_index} has an invalid {id_field}"
            )
        if expected_id in seen:
            raise ValueError(f"{table_name}: duplicate {id_field} {expected_id}")
        seen.add(expected_id)

        if table_name == DESCRIPTIVE_TABLE:
            if row.get("missing_policy") != (
                "available_recording_values_by_condition"
            ):
                raise ValueError(
                    f"{table_name}: row {row_index} has an invalid missing_policy"
                )
            unit_count = row.get("unit_count")
            if type(unit_count) is not int or unit_count < 0:
                raise ValueError(
                    f"{table_name}: row {row_index} has an invalid unit_count"
                )
            values = tuple(
                row.get(name)
                for name in ("sum", "mean", "median", "std_dev", "sem", "min", "max")
            )
            if any(
                value is not None
                and (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float, np.number))
                    or not math.isfinite(float(value))
                )
                for value in values
            ):
                raise ValueError(
                    f"{table_name}: row {row_index} has a nonfinite descriptive value"
                )
            if unit_count == 0 and any(value is not None for value in values):
                raise ValueError(
                    f"{table_name}: empty row {row_index} has descriptive values"
                )
            required_location = ("sum", "mean", "median", "min", "max")
            if unit_count > 0 and any(
                row.get(name) is None for name in required_location
            ):
                raise ValueError(
                    f"{table_name}: row {row_index} lacks descriptive location values"
                )
            if unit_count < 2 and any(
                row.get(name) is not None for name in ("std_dev", "sem")
            ):
                raise ValueError(
                    f"{table_name}: row {row_index} has singleton dispersion values"
                )
            if unit_count >= 2 and any(
                row.get(name) is None for name in ("std_dev", "sem")
            ):
                raise ValueError(
                    f"{table_name}: row {row_index} lacks descriptive dispersion values"
                )
            continue

        status = row.get("status")
        if status not in {"computed", "skipped"}:
            raise ValueError(f"{table_name}: row {row_index} has invalid status")
        if (status == "computed") != (row.get("skip_reason") is None):
            raise ValueError(
                f"{table_name}: row {row_index} has inconsistent skip_reason"
            )
        if status != "computed" and row.get("q_value") is not None:
            raise ValueError(
                f"{table_name}: row {row_index} has q_value for skipped result"
            )
        cluster_mode = row.get("cluster_mode")
        cluster_status = row.get("cluster_status")
        if cluster_mode not in {"none", "session"}:
            raise ValueError(
                f"{table_name}: row {row_index} has invalid cluster_mode"
            )
        if row.get("cluster_unit") != "session":
            raise ValueError(
                f"{table_name}: row {row_index} has invalid cluster_unit"
            )
        cluster_count = row.get("cluster_count")
        clustered_unit_count = row.get("clustered_unit_count")
        if (
            type(cluster_count) is not int
            or cluster_count < 0
            or type(clustered_unit_count) is not int
            or clustered_unit_count < 0
        ):
            raise ValueError(
                f"{table_name}: row {row_index} has invalid cluster counts"
            )
        clustered_values = (
            "clustered_mean_difference",
            "clustered_standard_error",
            "clustered_ci_low",
            "clustered_ci_high",
            "clustered_p_value",
            "clustered_q_value",
            "session_variance",
            "residual_variance",
            "intraclass_correlation",
        )
        if status != "computed" and cluster_status in {
            "computed",
            "boundary_zero_variance",
        }:
            raise ValueError(
                f"{table_name}: row {row_index} has clustered inference for a "
                "naive-ineligible result"
            )
        if cluster_mode == "none":
            if (
                cluster_status != "disabled"
                or row.get("cluster_method") != "none"
                or row.get("cluster_reason") is not None
                or cluster_count != 0
                or any(row.get(name) is not None for name in clustered_values)
            ):
                raise ValueError(
                    f"{table_name}: row {row_index} has inconsistent disabled clustering"
                )
        elif cluster_status == "unavailable":
            if (
                row.get("cluster_method") != "session_random_intercept_reml_v1"
                or not isinstance(row.get("cluster_reason"), str)
                or not row.get("cluster_reason")
                or any(row.get(name) is not None for name in clustered_values)
            ):
                raise ValueError(
                    f"{table_name}: row {row_index} has inconsistent unavailable clustering"
                )
        elif cluster_status in {"computed", "boundary_zero_variance"}:
            if (
                row.get("cluster_method") != "session_random_intercept_reml_v1"
                or row.get("cluster_reason") is not None
                or cluster_count < 2
                or any(
                    row.get(name) is None
                    for name in clustered_values
                    if name != "clustered_q_value"
                )
            ):
                raise ValueError(
                    f"{table_name}: row {row_index} has inconsistent computed clustering"
                )
        else:
            raise ValueError(
                f"{table_name}: row {row_index} has invalid cluster_status"
            )
        one_sample = (
            row.get("contrast_name") == "vs-zero"
            and row.get("condition_a") == "zero"
        )
        expected_missing_policy = (
            "one_row_per_recording_complete_cases"
            if one_sample
            else "paired_complete_recordings"
        )
        if row.get("missing_policy") != expected_missing_policy:
            raise ValueError(
                f"{table_name}: row {row_index} has invalid missing_policy"
            )
        counts = tuple(
            row.get(name)
            for name in ("unit_count", "paired_unit_count", "excluded_unit_count")
        )
        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError(f"{table_name}: row {row_index} has invalid counts")
        if counts[1] + counts[2] != counts[0]:
            raise ValueError(
                f"{table_name}: row {row_index} has inconsistent counts"
            )
        for field_name in ("bootstrap_iterations", "permutation_iterations"):
            value = row.get(field_name)
            if type(value) is not int or value < 0:
                raise ValueError(
                    f"{table_name}: row {row_index} has invalid {field_name}"
                )
        numeric_names = (
            "mean_a",
            "mean_b",
            "mean_difference",
            "median_difference",
            "std_difference",
            "effect_size",
            "ci_low",
            "ci_high",
            "p_value",
            "q_value",
            "clustered_mean_difference",
            "clustered_standard_error",
            "clustered_ci_low",
            "clustered_ci_high",
            "clustered_p_value",
            "clustered_q_value",
            "session_variance",
            "residual_variance",
            "intraclass_correlation",
        )
        if any(
            row.get(name) is not None
            and (
                isinstance(row.get(name), bool)
                or not isinstance(row.get(name), (int, float, np.number))
                or not math.isfinite(float(row[name]))
            )
            for name in numeric_names
        ):
            raise ValueError(
                f"{table_name}: row {row_index} has a nonfinite statistical value"
            )
        for field_name in (
            "p_value",
            "q_value",
            "clustered_p_value",
            "clustered_q_value",
            "intraclass_correlation",
        ):
            value = row.get(field_name)
            if value is not None and not 0.0 <= float(value) <= 1.0:
                raise ValueError(
                    f"{table_name}: row {row_index} has invalid {field_name}"
                )
        ci_low = row.get("ci_low")
        ci_high = row.get("ci_high")
        if (ci_low is None) != (ci_high is None) or (
            ci_low is not None and float(ci_low) > float(ci_high)
        ):
            raise ValueError(
                f"{table_name}: row {row_index} has invalid CI bounds"
            )
        clustered_ci_low = row.get("clustered_ci_low")
        clustered_ci_high = row.get("clustered_ci_high")
        if (clustered_ci_low is None) != (clustered_ci_high is None) or (
            clustered_ci_low is not None
            and float(clustered_ci_low) > float(clustered_ci_high)
        ):
            raise ValueError(
                f"{table_name}: row {row_index} has invalid clustered CI bounds"
            )
        allowed_methods = (
            {
                "skipped",
                "wilcoxon_signed_rank_all_zero",
                "wilcoxon_signed_rank_exact",
                "wilcoxon_signed_rank_normal",
                "wilcoxon_signed_rank_unavailable",
            }
            if one_sample
            else {
                "skipped",
                "paired_sign_flip_exact",
                "paired_sign_flip_random",
                "paired_sign_flip_unavailable",
            }
        )
        if row.get("test_method") not in allowed_methods:
            raise ValueError(
                f"{table_name}: row {row_index} has invalid test_method"
            )
        test_method = row.get("test_method")
        if test_method == "skipped":
            if status != "skipped":
                raise ValueError(
                    f"{table_name}: row {row_index} has inconsistent skipped method"
                )
            unavailable_fields = (
                "mean_difference",
                "median_difference",
                "std_difference",
                "effect_size",
                "ci_low",
                "ci_high",
                "p_value",
                "q_value",
            )
            if any(row.get(name) is not None for name in unavailable_fields):
                raise ValueError(
                    f"{table_name}: row {row_index} has inferential values for a low-count skip"
                )
            if row.get("bootstrap_iterations") != 0 or row.get(
                "permutation_iterations"
            ) != 0:
                raise ValueError(
                    f"{table_name}: row {row_index} has iterations for a low-count skip"
                )
        elif status == "skipped":
            if not (
                one_sample
                and test_method == "wilcoxon_signed_rank_unavailable"
                and row.get("skip_reason") == "wilcoxon_unavailable"
                and row.get("p_value") is None
                and row.get("q_value") is None
            ):
                raise ValueError(
                    f"{table_name}: row {row_index} has inconsistent skipped inference"
                )
        elif test_method == "wilcoxon_signed_rank_unavailable":
            raise ValueError(
                f"{table_name}: row {row_index} has unavailable method for a computed result"
            )
        expected_effect = (
            "rank_biserial_correlation"
            if one_sample
            else "paired_mean_difference_over_sample_sd"
        )
        expected_estimand = (
            "one_sample_median" if one_sample else "paired_mean_difference"
        )
        if row.get("effect_size_kind") != expected_effect:
            raise ValueError(
                f"{table_name}: row {row_index} has invalid effect_size_kind"
            )
        if row.get("ci_estimand") != expected_estimand:
            raise ValueError(
                f"{table_name}: row {row_index} has invalid ci_estimand"
            )

    if table_name == SUMMARY_TABLE:
        by_family: dict[str, list[tuple[int, Mapping[str, Any]]]] = {}
        clustered_by_family: dict[str, list[tuple[int, Mapping[str, Any]]]] = {}
        for row_index, row in enumerate(rows):
            if row.get("status") == "computed":
                family = str(row.get("multiple_comparison_family") or "")
                by_family.setdefault(family, []).append((row_index, row))
                if row.get("cluster_status") in {
                    "computed",
                    "boundary_zero_variance",
                }:
                    clustered_by_family.setdefault(family, []).append(
                        (row_index, row)
                    )
        for family_rows in by_family.values():
            expected_q_values = benjamini_hochberg(
                [_safe_float(row.get("p_value")) for _index, row in family_rows]
            )
            for (row_index, row), expected_q in zip(
                family_rows,
                expected_q_values,
                strict=True,
            ):
                actual_q = row.get("q_value")
                if expected_q is None:
                    valid = actual_q is None
                else:
                    valid = actual_q is not None and math.isclose(
                        float(actual_q),
                        float(expected_q),
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                if not valid:
                    raise ValueError(
                        f"{table_name}: row {row_index} has an invalid FDR q_value"
                    )
        for family_rows in clustered_by_family.values():
            expected_q_values = benjamini_hochberg(
                [
                    _safe_float(row.get("clustered_p_value"))
                    for _index, row in family_rows
                ]
            )
            for (row_index, row), expected_q in zip(
                family_rows,
                expected_q_values,
                strict=True,
            ):
                actual_q = row.get("clustered_q_value")
                if expected_q is None:
                    valid = actual_q is None
                else:
                    valid = actual_q is not None and math.isclose(
                        float(actual_q),
                        float(expected_q),
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                if not valid:
                    raise ValueError(
                        f"{table_name}: row {row_index} has an invalid clustered FDR q_value"
                    )


def _validate_result_rows_against_manifest(
    table_name: str,
    rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> None:
    _validate_fdr_family_test_registry(
        manifest.get("fdr_families"),
        rows=rows if table_name == SUMMARY_TABLE else None,
    )
    raw_metrics = manifest.get("metrics")
    if not isinstance(raw_metrics, list):
        raise ValueError("Statistics manifest lacks an exact metric registry")
    metric_fields = {
        "metric_family",
        "source_table",
        "metric_name",
        "metric_unit",
        "group_keys",
        "primary",
        "exploratory",
    }
    metric_specs: dict[tuple[str, str], Mapping[str, Any]] = {}
    for item in raw_metrics:
        if not isinstance(item, Mapping) or set(item) != metric_fields:
            raise ValueError("Statistics manifest metric registry is invalid")
        key = (str(item.get("source_table")), str(item.get("metric_name")))
        if key in metric_specs:
            raise ValueError("Statistics manifest metric registry is not unique")
        group_keys = item.get("group_keys")
        if not isinstance(group_keys, list) or any(
            not isinstance(value, str) or not value for value in group_keys
        ):
            raise ValueError("Statistics manifest metric group keys are invalid")
        if item.get("metric_unit") != _metric_unit_for_values(*key):
            raise ValueError("Statistics manifest metric unit is invalid")
        if type(item.get("primary")) is not bool or type(
            item.get("exploratory")
        ) is not bool:
            raise ValueError("Statistics manifest metric flags are invalid")
        if bool(item.get("primary")) == bool(item.get("exploratory")):
            raise ValueError(
                "Statistics manifest metrics must be exactly one of primary or exploratory"
            )
        metric_specs[key] = item

    raw_contrasts = manifest.get("contrasts")
    if not isinstance(raw_contrasts, list):
        raise ValueError("Statistics manifest lacks exact contrast declarations")
    contrast_fields = {"name", "condition_a", "condition_b"}
    contrasts: dict[str, tuple[str, str]] = {}
    for item in raw_contrasts:
        if not isinstance(item, Mapping) or set(item) != contrast_fields:
            raise ValueError("Statistics manifest contrast registry is invalid")
        name = item.get("name")
        conditions = (item.get("condition_a"), item.get("condition_b"))
        if (
            not isinstance(name, str)
            or not name
            or name in contrasts
            or any(not isinstance(value, str) or not value for value in conditions)
        ):
            raise ValueError("Statistics manifest contrast registry is invalid")
        contrasts[name] = (str(conditions[0]), str(conditions[1]))

    manifest_parameters = manifest.get("parameters")
    if not isinstance(manifest_parameters, Mapping):
        raise ValueError("Statistics manifest parameters are invalid")
    required_parameters = {
        "allow_legacy_export_layout",
        "bootstrap_iterations",
        "confidence_level",
        "cluster",
        "fdr_method",
        "fdr_family_rule",
        "minimum_recordings",
        "minimum_sessions",
        "permutation_iterations",
        "random_seed",
        "role_mapping_table",
    }
    if set(manifest_parameters) != required_parameters:
        raise ValueError("Statistics manifest parameters have an invalid field set")
    if manifest_parameters.get("fdr_method") != "benjamini_hochberg":
        raise ValueError("Statistics manifest FDR method is invalid")
    if manifest_parameters.get("fdr_family_rule") != "analysis_tier_metric_family_v1":
        raise ValueError("Statistics manifest FDR family rule is invalid")
    if manifest_parameters.get("cluster") not in {"none", "session"}:
        raise ValueError("Statistics manifest cluster mode is invalid")
    if type(manifest_parameters.get("allow_legacy_export_layout")) is not bool:
        raise ValueError("Statistics manifest legacy-layout policy is invalid")
    for parameter_name in (
        "bootstrap_iterations",
        "permutation_iterations",
        "random_seed",
    ):
        if type(manifest_parameters.get(parameter_name)) is not int or int(
            manifest_parameters[parameter_name]
        ) < 0:
            raise ValueError(
                f"Statistics manifest parameter {parameter_name} is invalid"
            )
    if type(manifest_parameters.get("minimum_recordings")) is not int or int(
        manifest_parameters["minimum_recordings"]
    ) < 1:
        raise ValueError("Statistics manifest minimum_recordings is invalid")
    if type(manifest_parameters.get("minimum_sessions")) is not int or int(
        manifest_parameters["minimum_sessions"]
    ) < 3:
        raise ValueError("Statistics manifest minimum_sessions is invalid")
    confidence = manifest_parameters.get("confidence_level")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0.0 < float(confidence) < 1.0
    ):
        raise ValueError("Statistics manifest confidence level is invalid")

    expected_input_tables = {source_table for source_table, _metric in metric_specs}
    if CRA_SUMMARY_TABLE in expected_input_tables:
        expected_input_tables.add(CRA_OBJECT_PHASE_TABLE)
    if CRA_NEAR_FIELD_SUMMARY_TABLE in expected_input_tables:
        expected_input_tables.add(CRA_NEAR_FIELD_OBJECT_PHASE_TABLE)
    if any(
        str(item.get("source_table")) in ROLE_GROUPED_SOURCE_TABLES
        and "behavior_class" in (item.get("group_keys") or ())
        for item in metric_specs.values()
    ):
        expected_input_tables.add(CRA_OBJECT_PHASE_TABLE)
    if manifest.get("input_tables") != sorted(expected_input_tables):
        raise ValueError("Statistics manifest input table registry is invalid")
    expected_role_mapping = (
        CRA_OBJECT_PHASE_TABLE
        if CRA_OBJECT_PHASE_TABLE in expected_input_tables
        else None
    )
    if manifest_parameters.get("role_mapping_table") != expected_role_mapping:
        raise ValueError("Statistics manifest role mapping table is invalid")

    source_path = manifest.get("source_export_manifest_path")
    source_counts = manifest.get("source_row_counts_by_table")
    if not isinstance(source_path, str) or not source_path:
        raise ValueError("Statistics manifest source path is invalid")
    if not isinstance(source_counts, Mapping):
        source_counts = {}
    collection = manifest.get("source_collection_manifest")
    if not isinstance(collection, Mapping):
        collection = {}
    expected_collection = (
        collection.get("collection_id"),
        collection.get("manifest_sha256"),
    )

    for row_index, row in enumerate(rows):
        key = (str(row.get("source_table")), str(row.get("metric_name")))
        spec = metric_specs.get(key)
        if spec is None:
            raise ValueError(
                f"{table_name}: row {row_index} is absent from manifest metrics"
            )
        if row.get("metric_family") != spec.get("metric_family"):
            raise ValueError(
                f"{table_name}: row {row_index} has an invalid metric_family"
            )
        if row.get("metric_unit") != spec.get("metric_unit"):
            raise ValueError(
                f"{table_name}: row {row_index} has an invalid metric_unit"
            )
        if row.get("primary") != spec.get("primary") or row.get(
            "exploratory"
        ) != spec.get("exploratory"):
            raise ValueError(
                f"{table_name}: row {row_index} has invalid metric flags"
            )
        group_key = json.loads(str(row.get("group_key_json")))
        if set(group_key) != set(spec.get("group_keys") or ()):
            raise ValueError(
                f"{table_name}: row {row_index} has invalid grouping keys"
            )
        if row.get("source_export_manifest_path") != source_path:
            raise ValueError(
                f"{table_name}: row {row_index} has an invalid source manifest path"
            )
        if (
            row.get("collection_id"),
            row.get("collection_manifest_sha256"),
        ) != expected_collection:
            raise ValueError(
                f"{table_name}: row {row_index} has invalid collection binding"
            )
        declared_count = source_counts.get(key[0])
        if declared_count is not None and row.get("source_row_count") != declared_count:
            raise ValueError(
                f"{table_name}: row {row_index} has an invalid source_row_count"
            )
        try:
            parameters = json.loads(str(row.get("parameters_json")))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"{table_name}: row {row_index} has invalid parameters_json"
            ) from exc
        one_sample = (
            table_name == SUMMARY_TABLE
            and row.get("contrast_name") == "vs-zero"
            and row.get("condition_a") == "zero"
        )
        expected_parameters: dict[str, object] = {
            "allow_legacy_export_layout": bool(
                manifest_parameters["allow_legacy_export_layout"]
            ),
            "bootstrap_iterations_requested": int(
                manifest_parameters["bootstrap_iterations"]
            ),
            "confidence_level": float(manifest_parameters["confidence_level"]),
            "cluster": str(manifest_parameters["cluster"]),
            "minimum_recordings": int(manifest_parameters["minimum_recordings"]),
            "minimum_sessions": int(manifest_parameters["minimum_sessions"]),
            "missing_policy": row.get("missing_policy"),
            "permutation_iterations_requested": int(
                manifest_parameters["permutation_iterations"]
            ),
            "random_seed": int(manifest_parameters["random_seed"]),
        }
        if one_sample:
            expected_parameters["test_target"] = 0.0
        if parameters != expected_parameters:
            raise ValueError(
                f"{table_name}: row {row_index} has invalid parameters_json"
            )
        if table_name == SUMMARY_TABLE:
            if row.get("cluster_status") in {
                "computed",
                "boundary_zero_variance",
            } and int(row.get("cluster_count") or 0) < int(
                manifest_parameters["minimum_sessions"]
            ):
                raise ValueError(
                    f"{table_name}: row {row_index} has clustered inference below "
                    "minimum_sessions"
                )
            tier = "primary" if row.get("primary") else "exploratory"
            expected_family = f"{tier}|{row.get('metric_family')}"
            if row.get("multiple_comparison_family") != expected_family:
                raise ValueError(
                    f"{table_name}: row {row_index} has invalid multiple-comparison binding"
                )
            if row.get("contrast_name") == "vs-zero":
                if (row.get("condition_a"), row.get("condition_b")) != (
                    "zero",
                    "observed",
                ) or key[0] not in {CRA_SUMMARY_TABLE, CRA_NEAR_FIELD_SUMMARY_TABLE}:
                    raise ValueError(
                        f"{table_name}: row {row_index} has invalid one-sample contrast"
                    )
            elif contrasts.get(str(row.get("contrast_name"))) != (
                row.get("condition_a"),
                row.get("condition_b"),
            ):
                raise ValueError(
                    f"{table_name}: row {row_index} has invalid contrast binding"
                )


def validate_goodcopbadcop_result_rows(
    table_name: str,
    rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> None:
    """Validate persisted exact rows against their semantic publication manifest."""

    stats_run_id = manifest.get("export_run_id")
    source_run_id = manifest.get("source_export_run_id")
    source_digest = manifest.get("source_export_manifest_sha256")
    if not isinstance(stats_run_id, str) or not stats_run_id:
        raise ValueError("Statistics manifest lacks a statistics run identity")
    if not isinstance(source_run_id, str) or not source_run_id:
        raise ValueError("Statistics manifest lacks a source export run identity")
    if (
        not isinstance(source_digest, str)
        or len(source_digest) != 64
        or source_digest != source_digest.lower()
    ):
        raise ValueError("Statistics manifest lacks a source manifest digest")
    try:
        int(source_digest, 16)
    except ValueError as exc:
        raise ValueError("Statistics manifest lacks a source manifest digest") from exc
    _validate_result_rows(
        table_name,
        rows,
        expected_stats_run_id=stats_run_id,
        expected_source_export_run_id=source_run_id,
        expected_source_manifest_sha256=source_digest,
    )
    _validate_result_rows_against_manifest(table_name, rows, manifest)


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
    required = {
        "recording_id",
        "session_id",
        "subject_id",
        "window_label",
        spec.metric_name,
        *spec.group_keys,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{spec.source_table} is missing required column(s) for {spec.metric_name}: {missing}")

    selected = frame.select(
        [
            "recording_id",
            "session_id",
            "subject_id",
            "window_label",
            spec.metric_name,
            *spec.group_keys,
        ]
    ).with_columns(
        pl.col("window_label")
        .map_elements(canonical_condition, return_dtype=pl.String)
        .alias("condition")
    )
    invalid_identity = selected.filter(
        pl.col("recording_id").is_null()
        | pl.col("session_id").is_null()
        | pl.col("subject_id").is_null()
        | (pl.col("recording_id").cast(pl.String).str.strip_chars() == "")
        | (pl.col("session_id").cast(pl.String).str.strip_chars() == "")
        | (pl.col("subject_id").cast(pl.String).str.strip_chars() == "")
    )
    if invalid_identity.height:
        raise ValueError(
            f"{spec.source_table} contains rows without registry-backed "
            "recording_id, session_id, or subject_id."
        )
    value_column = "_value"
    selected = selected.rename({spec.metric_name: value_column})
    return (
        selected.drop_nulls(["condition", value_column])
        .group_by(
            [
                "recording_id",
                "session_id",
                "subject_id",
                "condition",
                *spec.group_keys,
            ]
        )
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
        "allow_legacy_export_layout": bool(config.allow_legacy_export_layout),
        "bootstrap_iterations_requested": int(config.bootstrap_iterations),
        "confidence_level": float(config.confidence_level),
        "cluster": config.cluster,
        "minimum_recordings": int(config.minimum_recordings),
        "minimum_sessions": int(config.minimum_sessions),
        "missing_policy": "available_recording_values_by_condition",
        "permutation_iterations_requested": int(config.permutation_iterations),
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
        "metric_unit": _metric_unit(spec),
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
        "missing_policy": "available_recording_values_by_condition",
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
    required = {"recording_id", "session_id", "subject_id", spec.metric_name}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{spec.source_table} is missing required column(s) for {spec.metric_name}: {missing}")
    selected = (
        frame.select(["recording_id", "session_id", "subject_id", spec.metric_name])
        .rename({spec.metric_name: "_value"})
        .drop_nulls(["recording_id", "session_id", "subject_id", "_value"])
        .group_by(["recording_id", "session_id", "subject_id"])
        .agg(pl.col("_value").mean().alias("_value"))
        .sort("recording_id")
    )
    if selected.height != frame.select("recording_id").drop_nulls().unique().height:
        raise ValueError(
            f"{spec.source_table} lacks one registry-backed session_id/subject_id "
            "identity per recording."
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
                .select(
                    [
                        "recording_id",
                        "session_id",
                        "subject_id",
                        pl.col("_value").alias("value_a"),
                    ]
                )
            )
            b = (
                group_frame.filter(pl.col("condition") == contrast.condition_b)
                .select(
                    [
                        "recording_id",
                        "session_id",
                        "subject_id",
                        pl.col("_value").alias("value_b"),
                    ]
                )
            )
            paired = a.join(
                b,
                on=["recording_id", "session_id", "subject_id"],
                how="inner",
            ).sort("recording_id")
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
            differences = np.asarray(values_b, dtype=np.float64) - np.asarray(
                values_a,
                dtype=np.float64,
            )
            clustered = _session_cluster_result(
                config=config,
                values=differences,
                session_ids=(
                    paired["session_id"].to_numpy()
                    if paired.height
                    else np.asarray([], dtype=object)
                ),
                naive_status=stats.status,
                naive_skip_reason=stats.skip_reason,
            )
            group_key = _group_key_payload(group_row, spec.group_keys)
            parameters = {
                "allow_legacy_export_layout": bool(config.allow_legacy_export_layout),
                "bootstrap_iterations_requested": int(config.bootstrap_iterations),
                "confidence_level": float(config.confidence_level),
                "minimum_recordings": int(config.minimum_recordings),
                "minimum_sessions": int(config.minimum_sessions),
                "missing_policy": "paired_complete_recordings",
                "permutation_iterations_requested": int(
                    config.permutation_iterations
                ),
                "random_seed": int(config.random_seed),
                "cluster": config.cluster,
            }
            multiple_comparison_family = _multiple_comparison_family(spec)
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
                "metric_unit": _metric_unit(spec),
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
                "effect_size_kind": "paired_mean_difference_over_sample_sd",
                "ci_estimand": "paired_mean_difference",
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
                **_cluster_fields(clustered),
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
    required = {"recording_id", "session_id", "subject_id", spec.metric_name}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{spec.source_table} is missing required column(s) for {spec.metric_name}: {missing}")

    selected = (
        frame.select(["recording_id", "session_id", "subject_id", spec.metric_name])
        .rename({spec.metric_name: "_value"})
        .drop_nulls(["recording_id", "session_id", "subject_id", "_value"])
        .group_by(["recording_id", "session_id", "subject_id"])
        .agg(pl.col("_value").mean().alias("_value"))
        .sort("recording_id")
    )
    if selected.height != frame.select("recording_id").drop_nulls().unique().height:
        raise ValueError(
            f"{spec.source_table} lacks one registry-backed session_id/subject_id "
            "identity per recording."
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
    clustered = _session_cluster_result(
        config=config,
        values=np.asarray(values, dtype=np.float64),
        session_ids=(
            selected["session_id"].to_numpy()
            if selected.height
            else np.asarray([], dtype=object)
        ),
        naive_status=stats.status,
        naive_skip_reason=stats.skip_reason,
    )
    collection = source_manifest.get("collection_manifest")
    if not isinstance(collection, Mapping):
        collection = {}
    source_row_counts = source_manifest.get("row_counts_by_table")
    if not isinstance(source_row_counts, Mapping):
        source_row_counts = {}

    parameters = {
        "allow_legacy_export_layout": bool(config.allow_legacy_export_layout),
        "bootstrap_iterations_requested": int(config.bootstrap_iterations),
        "confidence_level": float(config.confidence_level),
        "minimum_recordings": int(config.minimum_recordings),
        "minimum_sessions": int(config.minimum_sessions),
        "missing_policy": "one_row_per_recording_complete_cases",
        "permutation_iterations_requested": int(config.permutation_iterations),
        "random_seed": int(config.random_seed),
        "cluster": config.cluster,
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
        "metric_unit": _metric_unit(spec),
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
        "effect_size_kind": "rank_biserial_correlation",
        "ci_estimand": "one_sample_median",
        "ci_low": stats.ci_low,
        "ci_high": stats.ci_high,
        "p_value": stats.p_value,
        "q_value": None,
        "multiple_comparison_family": _multiple_comparison_family(spec),
        "test_method": stats.test_method,
        "bootstrap_iterations": stats.bootstrap_iterations,
        "permutation_iterations": stats.permutation_iterations,
        "status": stats.status,
        "skip_reason": stats.skip_reason,
        "parameters_json": _json_dumps(parameters),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **_cluster_fields(clustered),
    }
    row["stat_result_id"] = _result_id(row)
    return [row]


def apply_fdr(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_family: dict[str, list[int]] = {}
    clustered_by_family: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        if row.get("status") != "computed":
            continue
        family = str(row.get("multiple_comparison_family") or "")
        by_family.setdefault(family, []).append(index)
        if row.get("cluster_status") in {"computed", "boundary_zero_variance"}:
            clustered_by_family.setdefault(family, []).append(index)
    for indices in by_family.values():
        q_values = benjamini_hochberg([_safe_float(rows[index].get("p_value")) for index in indices])
        for index, q_value in zip(indices, q_values):
            rows[index]["q_value"] = q_value
    for indices in clustered_by_family.values():
        q_values = benjamini_hochberg(
            [_safe_float(rows[index].get("clustered_p_value")) for index in indices]
        )
        for index, q_value in zip(indices, q_values, strict=True):
            rows[index]["clustered_q_value"] = q_value
    return rows


@dataclass(frozen=True)
class _GoodCopBadCopInputSnapshot:
    export_root: Path
    source_manifest: Mapping[str, Any]
    source_manifest_sha256: str
    frames: Mapping[str, pl.DataFrame]


def _load_goodcopbadcop_input_snapshot(
    config: GoodCopBadCopStatisticsConfig,
) -> _GoodCopBadCopInputSnapshot:
    """Bind one manifest byte identity, validated inventory, and table scan."""

    export_root = Path(config.export_root).expanduser().resolve()
    source_run_id = safe_component(
        config.source_export_run_id,
        label="source export run ID",
    )
    source_manifest_file = source_manifest_path(export_root, source_run_id)
    source_manifest, source_manifest_sha256 = _load_json_snapshot(source_manifest_file)
    _require_v2_source_manifest(source_manifest, path=source_manifest_file)
    if source_manifest.get("export_run_id") != source_run_id:
        raise ValueError(
            "source export manifest run identity does not match the requested run"
        )
    if not config.allow_legacy_export_layout:
        validate_export_payload(
            export_root,
            source_run_id,
            source_manifest,
        )
    tables = _statistics_input_tables(config.metrics)
    frames = {
        table: _read_export_table(
            export_root,
            source_manifest,
            table,
            allow_legacy_layout=config.allow_legacy_export_layout,
        )
        for table in tables
    }
    _enrich_role_grouped_frames(frames, config.metrics)
    _derive_profile_role_contrasts(frames, config.metrics)
    return _GoodCopBadCopInputSnapshot(
        export_root=export_root,
        source_manifest=source_manifest,
        source_manifest_sha256=source_manifest_sha256,
        frames=frames,
    )


def _compute_goodcopbadcop_statistics_from_snapshot(
    config: GoodCopBadCopStatisticsConfig,
    snapshot: _GoodCopBadCopInputSnapshot,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    export_root = snapshot.export_root
    source_manifest = snapshot.source_manifest
    source_manifest_sha256 = snapshot.source_manifest_sha256
    frames = snapshot.frames
    rng = np.random.default_rng(int(config.random_seed))

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
            minimum_sessions=config.minimum_sessions,
            random_seed=config.random_seed,
            cluster=config.cluster,
            overwrite=config.overwrite,
            allow_legacy_export_layout=config.allow_legacy_export_layout,
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
    _validate_result_rows(
        SUMMARY_TABLE,
        rows,
        expected_stats_run_id=config.stats_run_id,
        expected_source_export_run_id=config.source_export_run_id,
        expected_source_manifest_sha256=source_manifest_sha256,
    )

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
            minimum_sessions=config.minimum_sessions,
            random_seed=config.random_seed,
            cluster=config.cluster,
            overwrite=config.overwrite,
            allow_legacy_export_layout=config.allow_legacy_export_layout,
        ),
        rows=rows,
        source_manifest=source_manifest,
        source_manifest_sha256=source_manifest_sha256,
    )
    _validate_result_rows_against_manifest(SUMMARY_TABLE, rows, manifest)
    return rows, manifest


def _compute_goodcopbadcop_descriptive_from_snapshot(
    config: GoodCopBadCopStatisticsConfig,
    snapshot: _GoodCopBadCopInputSnapshot,
) -> list[dict[str, Any]]:
    export_root = snapshot.export_root
    source_manifest = snapshot.source_manifest
    source_manifest_sha256 = snapshot.source_manifest_sha256
    frames = snapshot.frames

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
            minimum_sessions=config.minimum_sessions,
            random_seed=config.random_seed,
            cluster=config.cluster,
            overwrite=config.overwrite,
            allow_legacy_export_layout=config.allow_legacy_export_layout,
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
    result = [canonicalize_export_row(DESCRIPTIVE_TABLE, row) for row in rows]
    _validate_result_rows(
        DESCRIPTIVE_TABLE,
        result,
        expected_stats_run_id=config.stats_run_id,
        expected_source_export_run_id=config.source_export_run_id,
        expected_source_manifest_sha256=source_manifest_sha256,
    )
    return result


def compute_goodcopbadcop_statistics(
    config: GoodCopBadCopStatisticsConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    snapshot = _load_goodcopbadcop_input_snapshot(config)
    return _compute_goodcopbadcop_statistics_from_snapshot(config, snapshot)


def compute_goodcopbadcop_descriptive_summaries(
    config: GoodCopBadCopStatisticsConfig,
) -> list[dict[str, Any]]:
    snapshot = _load_goodcopbadcop_input_snapshot(config)
    return _compute_goodcopbadcop_descriptive_from_snapshot(config, snapshot)


def compute_goodcopbadcop_outputs(
    config: GoodCopBadCopStatisticsConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Compute both published tables from one immutable input snapshot."""

    snapshot = _load_goodcopbadcop_input_snapshot(config)
    rows, manifest = _compute_goodcopbadcop_statistics_from_snapshot(config, snapshot)
    descriptive_rows = _compute_goodcopbadcop_descriptive_from_snapshot(
        config,
        snapshot,
    )
    _validate_result_rows_against_manifest(
        DESCRIPTIVE_TABLE,
        descriptive_rows,
        manifest,
    )
    return rows, descriptive_rows, manifest


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
        "input_tables": _statistics_input_tables(config.metrics),
        "source_row_counts_by_table": source_manifest.get("row_counts_by_table"),
        "output_tables": output_tables,
        "table_contracts": contract_snapshot(output_tables),
        "arrow_schema_contracts": arrow_contract_envelope(output_tables),
        "row_counts_by_table": row_counts_by_table,
        "status_counts": status_counts,
        "fdr_families": _fdr_family_test_registry(rows),
        "metrics": [
            {
                "metric_family": spec.metric_family,
                "source_table": spec.source_table,
                "metric_name": spec.metric_name,
                "metric_unit": _metric_unit(spec),
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
            "allow_legacy_export_layout": bool(config.allow_legacy_export_layout),
            "bootstrap_iterations": int(config.bootstrap_iterations),
            "permutation_iterations": int(config.permutation_iterations),
            "confidence_level": float(config.confidence_level),
            "minimum_recordings": int(config.minimum_recordings),
            "minimum_sessions": int(config.minimum_sessions),
            "random_seed": int(config.random_seed),
            "cluster": config.cluster,
            "fdr_method": "benjamini_hochberg",
            "fdr_family_rule": "analysis_tier_metric_family_v1",
            "role_mapping_table": (
                CRA_OBJECT_PHASE_TABLE
                if CRA_OBJECT_PHASE_TABLE in _statistics_input_tables(config.metrics)
                else None
            ),
        },
    }


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
    stats_run_id = safe_component(stats_run_id, label="statistics run ID")
    manifest_path = export_manifest_path(output_root, stats_run_id)
    manifest_dir = manifest_path.parent
    baseline_manifest_identity = manifest_identity(manifest_path)
    if manifest.get("export_run_id") != stats_run_id:
        raise ValueError(
            "Statistics manifest export_run_id does not match the requested run"
        )
    source_run_id = manifest.get("source_export_run_id")
    source_manifest_sha256 = manifest.get("source_export_manifest_sha256")
    if not isinstance(source_run_id, str) or not source_run_id:
        raise ValueError("Statistics manifest lacks a source export run identity")
    if (
        not isinstance(source_manifest_sha256, str)
        or len(source_manifest_sha256) != 64
        or source_manifest_sha256 != source_manifest_sha256.lower()
    ):
        raise ValueError("Statistics manifest lacks a source manifest digest")
    try:
        int(source_manifest_sha256, 16)
    except ValueError as exc:
        raise ValueError("Statistics manifest lacks a source manifest digest") from exc
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"Statistics manifest already exists: {manifest_path}")
    generation_id = uuid.uuid4().hex
    generation_path = generation_relative_path(stats_run_id, generation_id)
    staging_root = publication_staging_root(
        output_root,
        stats_run_id,
        generation_id,
    )
    final_generation_root = publication_generation_root(
        output_root,
        stats_run_id,
        generation_id,
    )

    canonical_rows_by_table = {
        SUMMARY_TABLE: [canonicalize_export_row(SUMMARY_TABLE, row) for row in rows],
    }
    _validate_result_rows(
        SUMMARY_TABLE,
        canonical_rows_by_table[SUMMARY_TABLE],
        expected_stats_run_id=stats_run_id,
        expected_source_export_run_id=source_run_id,
        expected_source_manifest_sha256=source_manifest_sha256,
    )
    _validate_result_rows_against_manifest(
        SUMMARY_TABLE,
        canonical_rows_by_table[SUMMARY_TABLE],
        manifest,
    )
    if descriptive_rows:
        canonical_rows_by_table[DESCRIPTIVE_TABLE] = [
            canonicalize_export_row(DESCRIPTIVE_TABLE, row) for row in descriptive_rows
        ]
        _validate_result_rows(
            DESCRIPTIVE_TABLE,
            canonical_rows_by_table[DESCRIPTIVE_TABLE],
            expected_stats_run_id=stats_run_id,
            expected_source_export_run_id=source_run_id,
            expected_source_manifest_sha256=source_manifest_sha256,
        )
        _validate_result_rows_against_manifest(
            DESCRIPTIVE_TABLE,
            canonical_rows_by_table[DESCRIPTIVE_TABLE],
            manifest,
        )

    def write_table(table_name: str, table_rows: Sequence[Mapping[str, Any]]) -> list[str]:
        table_dir = staging_root / "tables" / table_name
        table_dir.mkdir(parents=True, exist_ok=True)
        contract = ARROW_TABLE_CONTRACTS[table_name]
        columns = tuple(field.name for field in contract.fields)
        expected = set(columns)
        unexpected = sorted(
            {str(key) for row in table_rows for key in row} - expected
        )
        if unexpected:
            raise ValueError(
                f"{table_name}: unexpected fields for exact Arrow schema: "
                f"{unexpected}"
            )
        nonnullable = {
            field.name for field in contract.fields if not field.nullable
        }
        for row_index, row in enumerate(table_rows):
            missing = sorted(
                name for name in nonnullable if row.get(name) is None
            )
            if missing:
                raise ValueError(
                    f"{table_name}: row {row_index} has null/missing "
                    f"non-nullable fields: {missing}"
                )
        missing = validate_table_columns(table_name, columns)
        if missing:
            raise ValueError(
                f"{table_name} does not satisfy its V2 table contract; "
                f"missing columns: {list(missing)}"
            )
        normalized = [
            {column: row.get(column) for column in columns}
            for row in table_rows
        ]
        part_path = table_dir / "part-00000.parquet"
        tmp_path = table_dir / ".part-00000.parquet.tmp"
        if tmp_path.exists():
            tmp_path.unlink()
        schema = exact_arrow_schema(
            table_name,
            metadata={
                b"palette.export_schema_id": EXPORT_SCHEMA_ID.encode("utf-8"),
                b"palette.export_schema_version": str(
                    EXPORT_SCHEMA_VERSION
                ).encode("utf-8"),
                b"palette.table_contract": json.dumps(
                    TABLE_CONTRACTS[table_name].to_dict(),
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8"),
            },
        )
        arrow_table = pa.Table.from_pylist(normalized, schema=schema)
        pq.write_table(arrow_table, tmp_path)
        os.replace(tmp_path, part_path)
        return [str(part_path)]

    try:
        part_files_by_table: dict[str, list[str]] = {
            SUMMARY_TABLE: write_table(SUMMARY_TABLE, canonical_rows_by_table[SUMMARY_TABLE]),
        }
        if descriptive_rows:
            part_files_by_table[DESCRIPTIVE_TABLE] = write_table(
                DESCRIPTIVE_TABLE,
                canonical_rows_by_table[DESCRIPTIVE_TABLE],
            )
    except Exception:
        if staging_root.exists():
            shutil.rmtree(staging_root)
        raise

    def prepare_payload() -> dict[str, Any]:
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
        part_inventory: dict[str, list[dict[str, Any]]] = {}
        relative_part_files: dict[str, list[str]] = {}
        for table_name, staged_paths in part_files_by_table.items():
            entries: list[dict[str, Any]] = []
            relative_paths: list[str] = []
            for staged_value in staged_paths:
                staged_path = Path(staged_value)
                relative = generation_path / "tables" / table_name / staged_path.name
                entries.append(
                    {
                        "path": relative.as_posix(),
                        "sha256": sha256_file(staged_path),
                        "size_bytes": int(staged_path.stat().st_size),
                        "row_count": int(
                            pq.ParquetFile(staged_path).metadata.num_rows
                        ),
                    }
                )
                relative_paths.append(relative.as_posix())
            part_inventory[table_name] = entries
            relative_part_files[table_name] = relative_paths
        payload["part_files_by_table"] = relative_part_files
        payload["publication"] = {
            "schema_id": PUBLICATION_SCHEMA_ID,
            "schema_version": PUBLICATION_SCHEMA_VERSION,
            "state": "complete",
            "generation_id": generation_id,
            "generation_path": generation_path.as_posix(),
            "parts_by_table": part_inventory,
        }
        payload["schema_id"] = EXPORT_SCHEMA_ID
        payload["schema_version"] = EXPORT_SCHEMA_VERSION
        payload["table_contracts"] = contract_snapshot(output_tables)
        payload["arrow_schema_contracts"] = arrow_contract_envelope(output_tables)
        columns_by_table = {
            table_name: [
                field.name for field in ARROW_TABLE_CONTRACTS[table_name].fields
            ]
            for table_name in part_files_by_table
        }
        capability_statuses = resolve_capabilities(columns_by_table)
        payload["capabilities"] = [
            status.capability_id
            for status in capability_statuses
            if status.available
        ]
        payload["capability_statuses"] = [
            status.to_dict() for status in capability_statuses
        ]
        payload["manifest_path"] = str(manifest_path)
        return payload

    try:
        manifest_dir.mkdir(parents=True, exist_ok=True)
        payload = prepare_payload()
        validate_staged_publication(staging_root, payload)
    except Exception:
        if staging_root.exists():
            shutil.rmtree(staging_root)
        raise
    tmp_manifest = manifest_dir / (
        f".export_run_id={stats_run_id}.generation={generation_id}.json.tmp"
    )
    final_generation_root.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging_root, final_generation_root)
    try:
        tmp_manifest.write_text(
            json.dumps(json_attr_safe(payload), indent=2, sort_keys=True) + "\n"
        )
        with manifest_commit_lock(manifest_path):
            if manifest_identity(manifest_path) != baseline_manifest_identity:
                raise RuntimeError(
                    "Statistics manifest changed during publication; "
                    "the staged generation was not committed"
                )
            os.replace(tmp_manifest, manifest_path)
    except Exception:
        tmp_manifest.unlink(missing_ok=True)
        if final_generation_root.exists():
            shutil.rmtree(final_generation_root)
        raise
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
