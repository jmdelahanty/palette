"""Recording-scoped grouped statistics over validated-behavior exports.

This module is a read-only consumer.  It opens one receipt-validated export,
uses only manifest-selected Parquet parts, and reduces every metric to one
exact row per ``recording_id`` and declared condition/group before computing
cohort summaries or paired contrasts.

The v1 result is deliberately exploratory.  It cannot claim confirmatory or
acquisition-batch-adjusted inference because the historical GoodBatBadBat
cohort does not carry authoritative acquisition-batch identity.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from hashlib import sha256
from itertools import product
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from fisheye.analytics_exports.publication import safe_component, sha256_file
from fisheye.analytics_exports.validated_behavior_dataset import (
    ValidatedBehaviorExportDataset,
)
from fisheye.group_statistics.paired import (
    benjamini_hochberg,
    compute_paired_contrast,
)
from fisheye.shared.json_safety import (
    json_attr_safe,
    strict_json_dumps,
    write_json_atomic,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256

from .validated_behavior_specs import (
    CONTRAST_SETS,
    ValidatedBehaviorHistogramAxisSpec,
    ValidatedBehaviorHistogramSpec,
    ValidatedBehaviorMetricSpec,
    validate_histogram_specs,
    validate_metric_specs,
)
from .validated_behavior_appearance import (
    build_chaser_appearance_dimension,
    validate_chaser_appearance_dimension,
)

SCHEMA_ID = "palette.analytics.validated_behavior.group_statistics"
SCHEMA_VERSION = 2
LEGACY_SCHEMA_VERSION = 1
METHOD_ID = "recording_scoped_equal_weight_paired_and_histogram_v2"
LEGACY_METHOD_ID = "recording_scoped_equal_weight_paired_v1"
SANDBOX_STATUS = "selector_ineligible_exploratory_candidate"
ALL_CONDITIONS = "__all__"


class ValidatedBehaviorGroupStatisticsError(ValueError):
    """Raised when a grouped-statistics request is scientifically unsafe."""


def _fail(message: str) -> None:
    raise ValidatedBehaviorGroupStatisticsError(message)


@dataclass(frozen=True, slots=True)
class ValidatedBehaviorGroupStatisticsConfig:
    """Deterministic exploratory statistics configuration."""

    statistics_run_id: str
    metric_specs: tuple[ValidatedBehaviorMetricSpec, ...]
    histogram_specs: tuple[ValidatedBehaviorHistogramSpec, ...] = ()
    bootstrap_iterations: int = 5000
    permutation_iterations: int = 5000
    confidence_level: float = 0.95
    minimum_recordings: int = 3
    random_seed: int = 0

    def __post_init__(self) -> None:
        safe_component(self.statistics_run_id, label="statistics_run_id")
        validate_metric_specs(self.metric_specs, allow_empty=True)
        validate_histogram_specs(self.histogram_specs, allow_empty=True)
        if not self.metric_specs and not self.histogram_specs:
            raise ValueError(
                "At least one metric or histogram specification is required"
            )
        if type(self.bootstrap_iterations) is not int or self.bootstrap_iterations < 0:
            raise ValueError("bootstrap_iterations must be one nonnegative integer")
        if (
            type(self.permutation_iterations) is not int
            or self.permutation_iterations < 0
        ):
            raise ValueError("permutation_iterations must be one nonnegative integer")
        if not 0.0 < float(self.confidence_level) < 1.0:
            raise ValueError("confidence_level must lie strictly between zero and one")
        if type(self.minimum_recordings) is not int or self.minimum_recordings < 2:
            raise ValueError("minimum_recordings must be at least two")
        if type(self.random_seed) is not int or self.random_seed < 0:
            raise ValueError("random_seed must be one nonnegative integer")

    @property
    def record(self) -> Mapping[str, object]:
        value = {
            "statistics_run_id": self.statistics_run_id,
            "method_id": METHOD_ID,
            "analysis_status": "exploratory",
            "experimental_unit": "recording_id",
            "cohort_weighting": "equal_weight_per_finite_recording",
            "bootstrap_iterations": self.bootstrap_iterations,
            "permutation_iterations": self.permutation_iterations,
            "confidence_level": self.confidence_level,
            "minimum_recordings": self.minimum_recordings,
            "random_seed": self.random_seed,
            "multiplicity_adjustment": (
                "benjamini_hochberg_within_declared_metric_family"
            ),
            "acquisition_batch_adjustment": "not_performed_identity_unavailable",
            "metric_specs": [spec.to_dict() for spec in self.metric_specs],
            "histogram_specs": [spec.to_dict() for spec in self.histogram_specs],
            "contrast_sets": {
                key: [item.to_dict() for item in values]
                for key, values in CONTRAST_SETS.items()
                if any(spec.contrast_set_id == key for spec in self.metric_specs)
            },
        }
        return MappingProxyType(
            {**value, "config_sha256": canonical_json_sha256(value)}
        )


@dataclass(frozen=True, slots=True)
class ValidatedBehaviorGroupStatisticsResult:
    """In-memory normalized result ready for sandbox publication."""

    config: ValidatedBehaviorGroupStatisticsConfig
    source_export: Mapping[str, object]
    cohort_summary: Mapping[str, object]
    source_queries: tuple[Mapping[str, object], ...]
    recording_values: tuple[Mapping[str, object], ...]
    descriptive_statistics: tuple[Mapping[str, object], ...]
    paired_contrasts: tuple[Mapping[str, object], ...]
    histogram_recipes: tuple[Mapping[str, object], ...]
    recording_histogram_bins: tuple[Mapping[str, object], ...]
    histogram_descriptive_statistics: tuple[Mapping[str, object], ...]


@dataclass(frozen=True, slots=True)
class _MetricBatch:
    metric_family: str
    source_table: str
    condition_column: str | None
    expected_conditions: tuple[str, ...]
    group_columns: tuple[str, ...]
    recording_reducer: str
    source_identity_columns: tuple[str, ...]
    reducer_order_column: str | None
    specs: tuple[ValidatedBehaviorMetricSpec, ...]


def _plain(value: Any) -> Any:
    return json_attr_safe(value)


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _group_record(row: Mapping[str, Any], columns: Sequence[str]) -> dict[str, Any]:
    return {name: _plain(row[name]) for name in columns}


def _group_identity(group: Mapping[str, Any]) -> tuple[str, str]:
    payload = strict_json_dumps(group)
    return payload, canonical_json_sha256(group)


def _condition_axis(spec: ValidatedBehaviorMetricSpec) -> str:
    return spec.condition_column or "none"


def _query_identity(
    dataset: ValidatedBehaviorExportDataset,
    spec: ValidatedBehaviorMetricSpec,
) -> dict[str, object]:
    columns = (
        "recording_id",
        *((spec.condition_column,) if spec.condition_column else ()),
        *spec.group_columns,
        *spec.source_identity_columns,
        *((spec.reducer_order_column,) if spec.reducer_order_column else ()),
        spec.value_column,
    )
    table = dataset.table(spec.source_table)
    identity = table.query_identity(
        columns=columns,
        predicate_description=(
            "all manifest-selected rows; fail-closed unique recording/condition/"
            "group grain; null and nonfinite values retained as exclusions"
            if spec.recording_reducer == "unique_exact_row"
            else (
                "all manifest-selected rows; require constant declared source identity "
                "and unique gapless reducer order within recording/condition/group; "
                "select the exact maximum-order row; null and nonfinite metric values "
                "retained as exclusions"
            )
        ),
    )
    body = {
        **identity,
        "metric_spec_sha256": spec.spec_sha256,
        "recording_reducer": spec.recording_reducer,
        "cohort_weighting": "equal_weight_per_finite_recording",
    }
    return {**body, "query_sha256": canonical_json_sha256(body)}


def _metric_batches(
    specs: Sequence[ValidatedBehaviorMetricSpec],
) -> tuple[_MetricBatch, ...]:
    grouped: dict[
        tuple[
            str,
            str,
            str | None,
            tuple[str, ...],
            tuple[str, ...],
            str,
            tuple[str, ...],
            str | None,
        ],
        list[ValidatedBehaviorMetricSpec],
    ] = defaultdict(list)
    for spec in specs:
        key = (
            spec.metric_family,
            spec.source_table,
            spec.condition_column,
            spec.expected_conditions,
            spec.group_columns,
            spec.recording_reducer,
            spec.source_identity_columns,
            spec.reducer_order_column,
        )
        grouped[key].append(spec)
    return tuple(
        _MetricBatch(
            metric_family=key[0],
            source_table=key[1],
            condition_column=key[2],
            expected_conditions=key[3],
            group_columns=key[4],
            recording_reducer=key[5],
            source_identity_columns=key[6],
            reducer_order_column=key[7],
            specs=tuple(sorted(values, key=lambda item: item.metric_id)),
        )
        for key, values in sorted(grouped.items(), key=lambda item: str(item[0]))
    )


def _validate_analysis_units(
    dataset: ValidatedBehaviorExportDataset,
) -> tuple[dict[str, object], tuple[str, ...]]:
    import polars as pl

    policy_envelope = dataset.manifest.get("analysis_unit_policy")
    if not isinstance(policy_envelope, Mapping):
        _fail("Source export lacks an analysis-unit policy envelope")
    policy = policy_envelope.get("record")
    if not isinstance(policy, Mapping):
        _fail("Source export lacks an analysis-unit policy record")
    if policy.get("analysis_unit_kind") != "recording":
        _fail("Grouped statistics require recording-scoped analysis units")
    if policy.get("member_id_field") != "recording_id":
        _fail("Grouped statistics require recording_id as the authorized unit key")

    columns = (
        "recording_id",
        "analysis_unit_kind",
        "analysis_unit_id",
        "membership_state",
        "acquisition_batch_id",
        "acquisition_batch_identity_status",
    )
    frame = dataset.table("cohort_recordings").scan(columns=columns).collect()
    if frame.height == 0:
        _fail("Source export contains no cohort recordings")
    if frame.get_column("recording_id").n_unique() != frame.height:
        _fail("Cohort recording IDs are not unique")
    if frame.filter(pl.col("analysis_unit_kind") != "recording").height:
        _fail("A cohort row does not use the recording analysis-unit kind")
    if frame.filter(pl.col("analysis_unit_id") != pl.col("recording_id")).height:
        _fail("A cohort row aliases recording_id through another analysis-unit ID")

    batch_counts = (
        frame.group_by("acquisition_batch_identity_status")
        .agg(pl.len().alias("recording_count"))
        .sort("acquisition_batch_identity_status")
        .to_dicts()
    )
    membership_counts = (
        frame.group_by("membership_state")
        .agg(pl.len().alias("recording_count"))
        .sort("membership_state")
        .to_dicts()
    )
    recording_ids = tuple(sorted(frame.get_column("recording_id").to_list()))

    bundle_counts = (
        dataset.table("recording_bundles")
        .scan(columns=("bundle_state",))
        .group_by("bundle_state")
        .agg(pl.len().alias("recording_count"))
        .sort("bundle_state")
        .collect()
        .to_dicts()
    )
    capability_counts = (
        dataset.table("recording_capabilities")
        .scan(columns=("capability_id", "state"))
        .group_by(("capability_id", "state"))
        .agg(pl.len().alias("recording_count"))
        .sort(("capability_id", "state"))
        .collect()
        .to_dicts()
    )
    summary = {
        "parent_recording_count": frame.height,
        "authorized_unit_key": "recording_id",
        "analysis_unit_kind": "recording",
        "analysis_unit_policy_sha256": policy_envelope.get("sha256"),
        "analysis_unit_policy": _plain(policy),
        "membership_state_counts": _plain(membership_counts),
        "bundle_state_counts": _plain(bundle_counts),
        "acquisition_batch_identity_status_counts": _plain(batch_counts),
        "capability_state_counts": _plain(capability_counts),
        "source_subject_id_use": "prohibited_for_grouping_joining_or_sample_size",
        "acquisition_batch_adjustment": "not_performed",
        "inference_status": "exploratory",
    }
    return summary, recording_ids


def _unit_lazy_frame(
    dataset: ValidatedBehaviorExportDataset,
    batch: _MetricBatch,
) -> Any:
    import polars as pl

    selected = ["recording_id"]
    if batch.condition_column is not None:
        selected.append(batch.condition_column)
    selected.extend(batch.group_columns)
    selected.extend(batch.source_identity_columns)
    if batch.reducer_order_column is not None:
        selected.append(batch.reducer_order_column)
    selected.extend(spec.value_column for spec in batch.specs)
    if len(set(selected)) != len(selected):
        _fail(f"{batch.source_table}: metric batch selects duplicate columns")

    table = dataset.table(batch.source_table)
    known = {item.name for item in table.spec.contract.fields}
    missing = sorted(set(selected) - known)
    if missing:
        _fail(f"{batch.source_table}: metric columns are absent: {missing}")
    lazy = table.scan(columns=selected)
    if batch.condition_column is None:
        lazy = lazy.with_columns(pl.lit(ALL_CONDITIONS).alias("__condition"))
    else:
        lazy = lazy.with_columns(
            pl.col(batch.condition_column).cast(pl.String).alias("__condition")
        )

    unit_keys = ("recording_id", "__condition", *batch.group_columns)
    null_key = pl.any_horizontal([pl.col(name).is_null() for name in unit_keys])
    if lazy.filter(null_key).limit(1).collect().height:
        _fail(f"{batch.source_table}: a recording/condition/group key is null")

    if batch.recording_reducer == "unique_exact_row":
        aggregations = [pl.len().alias("__source_row_count")]
        aggregations.extend(
            pl.col(spec.value_column).first().alias(spec.value_column)
            for spec in batch.specs
        )
        units = lazy.group_by(unit_keys).agg(aggregations)
        duplicate = units.filter(pl.col("__source_row_count") != 1).limit(1).collect()
        if duplicate.height:
            example = duplicate.to_dicts()[0]
            _fail(
                f"{batch.source_table}: metric grain is not one exact row per "
                f"recording/condition/group; example={example!r}"
            )
    elif batch.recording_reducer == "terminal_at_max_order_v1":
        assert batch.reducer_order_column is not None
        order = batch.reducer_order_column
        terminal_required = (*batch.source_identity_columns, order)
        null_terminal = pl.any_horizontal(
            [pl.col(name).is_null() for name in terminal_required]
        )
        null_example = lazy.filter(null_terminal).limit(1).collect()
        if null_example.height:
            _fail(
                f"{batch.source_table}: terminal reducer identity/order is null; "
                f"example={null_example.to_dicts()[0]!r}"
            )
        identity_aliases = {
            name: f"__identity_n_unique_{index}"
            for index, name in enumerate(batch.source_identity_columns)
        }
        checks = lazy.group_by(unit_keys).agg(
            pl.len().alias("__source_row_count"),
            pl.col(order).n_unique().alias("__order_n_unique"),
            pl.col(order).min().alias("__order_min"),
            pl.col(order).max().alias("__order_max"),
            *(
                pl.col(name).n_unique().alias(alias)
                for name, alias in identity_aliases.items()
            ),
        )
        invalid_check = (
            (pl.col("__source_row_count") != pl.col("__order_n_unique"))
            | (
                pl.col("__source_row_count")
                != (pl.col("__order_max") - pl.col("__order_min") + 1)
            )
        )
        for alias in identity_aliases.values():
            invalid_check = invalid_check | (pl.col(alias) != 1)
        invalid = checks.filter(invalid_check).limit(1).collect()
        if invalid.height:
            _fail(
                f"{batch.source_table}: terminal reducer requires one constant "
                "source identity and a unique gapless order axis per recording/"
                f"condition/group; example={invalid.to_dicts()[0]!r}"
            )
        terminal_keys = checks.select(
            *unit_keys,
            pl.col("__order_max").alias(order),
        )
        terminal_rows = lazy.join(
            terminal_keys,
            on=(*unit_keys, order),
            how="inner",
        )
        aggregations = [pl.len().alias("__source_row_count")]
        aggregations.extend(
            pl.col(spec.value_column).first().alias(spec.value_column)
            for spec in batch.specs
        )
        units = terminal_rows.group_by(unit_keys).agg(aggregations)
        ambiguous = units.filter(pl.col("__source_row_count") != 1).limit(1).collect()
        if ambiguous.height:
            _fail(
                f"{batch.source_table}: terminal reducer did not resolve exactly "
                f"one maximum-order row; example={ambiguous.to_dicts()[0]!r}"
            )
    else:  # pragma: no cover - specification validation owns this branch
        _fail(f"Unsupported recording reducer: {batch.recording_reducer}")

    observed = {
        str(value)
        for value in units.select("__condition")
        .unique()
        .collect()
        .get_column("__condition")
    }
    expected = set(batch.expected_conditions or (ALL_CONDITIONS,))
    if observed != expected:
        _fail(
            f"{batch.source_table}: observed condition registry {sorted(observed)!r} "
            f"does not equal the exact expected registry {sorted(expected)!r}"
        )
    return units


def _descriptive_records(
    dataset: ValidatedBehaviorExportDataset,
    config: ValidatedBehaviorGroupStatisticsConfig,
    batch: _MetricBatch,
    units: Any,
    *,
    parent_recording_count: int,
) -> tuple[dict[str, object], ...]:
    import polars as pl

    finite_names: dict[str, str] = {}
    finite_exprs = []
    for spec in batch.specs:
        name = f"__finite_{spec.value_column}"
        finite_names[spec.metric_id] = name
        numeric = pl.col(spec.value_column).cast(pl.Float64, strict=True)
        finite_exprs.append(
            pl.when(numeric.is_not_null() & numeric.is_finite())
            .then(numeric)
            .otherwise(None)
            .alias(name)
        )
    prepared = units.with_columns(finite_exprs)
    summary_keys = ("__condition", *batch.group_columns)
    aggregations: list[Any] = [pl.len().alias("__contributor_count")]
    for spec in batch.specs:
        finite = pl.col(finite_names[spec.metric_id])
        prefix = spec.value_column
        aggregations.extend(
            (
                finite.count().alias(f"{prefix}__finite_count"),
                finite.mean().alias(f"{prefix}__mean"),
                finite.median().alias(f"{prefix}__median"),
                finite.std(ddof=1).alias(f"{prefix}__std"),
                finite.min().alias(f"{prefix}__min"),
                finite.quantile(0.25, interpolation="linear").alias(f"{prefix}__p25"),
                finite.quantile(0.75, interpolation="linear").alias(f"{prefix}__p75"),
                finite.max().alias(f"{prefix}__max"),
            )
        )
    summary = prepared.group_by(summary_keys).agg(aggregations).sort(summary_keys)
    rows: list[dict[str, object]] = []
    query_by_metric = {
        spec.metric_id: _query_identity(dataset, spec) for spec in batch.specs
    }
    for row in summary.collect().to_dicts():
        group = _group_record(row, batch.group_columns)
        group_json, group_sha256 = _group_identity(group)
        condition = str(row["__condition"])
        contributor_count = int(row["__contributor_count"])
        for spec in batch.specs:
            prefix = spec.value_column
            finite_count = int(row[f"{prefix}__finite_count"])
            std = _finite_float(row[f"{prefix}__std"])
            sem = (
                std / math.sqrt(float(finite_count))
                if std is not None and finite_count > 0
                else None
            )
            query = query_by_metric[spec.metric_id]
            rows.append(
                {
                    "statistics_run_id": config.statistics_run_id,
                    "source_export_run_id": dataset.export_run_id,
                    "source_export_manifest_sha256": dataset.cache_identity,
                    "metric_id": spec.metric_id,
                    "metric_spec_sha256": spec.spec_sha256,
                    "metric_family": spec.metric_family,
                    "source_table": spec.source_table,
                    "source_value_column": spec.value_column,
                    "source_table_contract_sha256": query["table_contract_sha256"],
                    "source_query_sha256": query["query_sha256"],
                    "unit": spec.unit,
                    "analysis_status": spec.analysis_status,
                    "condition_axis": _condition_axis(spec),
                    "condition": condition,
                    "group_key_json": group_json,
                    "group_key_sha256": group_sha256,
                    "parent_recording_count": parent_recording_count,
                    "contributor_recording_count": contributor_count,
                    "finite_recording_count": finite_count,
                    "excluded_nonfinite_recording_count": (
                        contributor_count - finite_count
                    ),
                    "noncontributor_recording_count": (
                        parent_recording_count - contributor_count
                    ),
                    "mean": _finite_float(row[f"{prefix}__mean"]),
                    "median": _finite_float(row[f"{prefix}__median"]),
                    "sample_std": std,
                    "sem": _finite_float(sem),
                    "minimum": _finite_float(row[f"{prefix}__min"]),
                    "p25": _finite_float(row[f"{prefix}__p25"]),
                    "p75": _finite_float(row[f"{prefix}__p75"]),
                    "maximum": _finite_float(row[f"{prefix}__max"]),
                    "cohort_weighting": "equal_weight_per_finite_recording",
                }
            )
    return tuple(rows)


def _recording_value_records(
    dataset: ValidatedBehaviorExportDataset,
    config: ValidatedBehaviorGroupStatisticsConfig,
    batch: _MetricBatch,
    units: Any,
) -> tuple[dict[str, object], ...]:
    retained = tuple(spec for spec in batch.specs if spec.retain_recording_values)
    if not retained:
        return ()
    selected = (
        "recording_id",
        "__condition",
        *batch.group_columns,
        *(spec.value_column for spec in retained),
    )
    frame = (
        units.select(selected)
        .sort(("recording_id", "__condition", *batch.group_columns))
        .collect()
    )
    query_by_metric = {
        spec.metric_id: _query_identity(dataset, spec) for spec in retained
    }
    rows: list[dict[str, object]] = []
    for row in frame.to_dicts():
        group = _group_record(row, batch.group_columns)
        group_json, group_sha256 = _group_identity(group)
        for spec in retained:
            raw = row[spec.value_column]
            value = _finite_float(raw)
            state = (
                "finite"
                if value is not None
                else ("source_null" if raw is None else "source_nonfinite")
            )
            query = query_by_metric[spec.metric_id]
            rows.append(
                {
                    "statistics_run_id": config.statistics_run_id,
                    "source_export_run_id": dataset.export_run_id,
                    "source_export_manifest_sha256": dataset.cache_identity,
                    "metric_id": spec.metric_id,
                    "metric_spec_sha256": spec.spec_sha256,
                    "metric_family": spec.metric_family,
                    "source_table": spec.source_table,
                    "source_value_column": spec.value_column,
                    "source_query_sha256": query["query_sha256"],
                    "unit": spec.unit,
                    "analysis_status": spec.analysis_status,
                    "recording_id": str(row["recording_id"]),
                    "condition_axis": _condition_axis(spec),
                    "condition": str(row["__condition"]),
                    "group_key_json": group_json,
                    "group_key_sha256": group_sha256,
                    "value": value,
                    "value_state": state,
                    "recording_reducer": spec.recording_reducer,
                }
            )
    return tuple(rows)


def _rng_for_contrast(
    config: ValidatedBehaviorGroupStatisticsConfig,
    spec: ValidatedBehaviorMetricSpec,
    contrast_id: str,
    group_sha256: str,
) -> np.random.Generator:
    payload = (
        f"{config.random_seed}\x00{spec.spec_sha256}\x00{contrast_id}\x00"
        f"{group_sha256}"
    ).encode("utf-8")
    seed = int.from_bytes(sha256(payload).digest()[:8], "big", signed=False)
    return np.random.default_rng(seed)


def _contrast_records(
    dataset: ValidatedBehaviorExportDataset,
    config: ValidatedBehaviorGroupStatisticsConfig,
    batch: _MetricBatch,
    recording_rows: Sequence[Mapping[str, object]],
    *,
    parent_recording_count: int,
) -> tuple[dict[str, object], ...]:
    contrasted = tuple(spec for spec in batch.specs if spec.contrast_eligible)
    if not contrasted:
        return ()

    by_metric: dict[
        str,
        dict[
            tuple[str, str],
            dict[str, dict[str, float | None]],
        ],
    ] = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for row in recording_rows:
        metric_id = str(row["metric_id"])
        if metric_id not in {spec.metric_id for spec in contrasted}:
            continue
        group_key = (str(row["group_key_json"]), str(row["group_key_sha256"]))
        condition = str(row["condition"])
        recording_id = str(row["recording_id"])
        by_metric[metric_id][group_key][condition][recording_id] = (
            None if row["value"] is None else float(row["value"])
        )

    rows: list[dict[str, object]] = []
    for spec in contrasted:
        assert spec.contrast_set_id is not None
        assert spec.multiplicity_family is not None
        query = _query_identity(dataset, spec)
        for group_key in sorted(by_metric[spec.metric_id]):
            group_json, group_sha256 = group_key
            condition_values = by_metric[spec.metric_id][group_key]
            for contrast in CONTRAST_SETS[spec.contrast_set_id]:
                a_map = condition_values.get(contrast.condition_a, {})
                b_map = condition_values.get(contrast.condition_b, {})
                eligible_ids = tuple(sorted(set(a_map) | set(b_map)))
                values_a = [
                    np.nan if a_map.get(key) is None else float(a_map[key])
                    for key in eligible_ids
                ]
                values_b = [
                    np.nan if b_map.get(key) is None else float(b_map[key])
                    for key in eligible_ids
                ]
                stats = compute_paired_contrast(
                    values_a,
                    values_b,
                    unit_count=len(eligible_ids),
                    minimum_recordings=config.minimum_recordings,
                    bootstrap_iterations=config.bootstrap_iterations,
                    permutation_iterations=config.permutation_iterations,
                    confidence_level=config.confidence_level,
                    rng=_rng_for_contrast(
                        config, spec, contrast.contrast_id, group_sha256
                    ),
                )
                result = asdict(stats)
                rows.append(
                    {
                        "statistics_run_id": config.statistics_run_id,
                        "source_export_run_id": dataset.export_run_id,
                        "source_export_manifest_sha256": dataset.cache_identity,
                        "metric_id": spec.metric_id,
                        "metric_spec_sha256": spec.spec_sha256,
                        "metric_family": spec.metric_family,
                        "multiplicity_family": spec.multiplicity_family,
                        "source_table": spec.source_table,
                        "source_value_column": spec.value_column,
                        "source_query_sha256": query["query_sha256"],
                        "unit": spec.unit,
                        "analysis_status": spec.analysis_status,
                        "condition_axis": _condition_axis(spec),
                        "contrast_id": contrast.contrast_id,
                        "condition_a": contrast.condition_a,
                        "condition_b": contrast.condition_b,
                        "difference_direction": "condition_b_minus_condition_a",
                        "group_key_json": group_json,
                        "group_key_sha256": group_sha256,
                        "parent_recording_count": parent_recording_count,
                        "eligible_recording_count": len(eligible_ids),
                        "noncontributor_recording_count": (
                            parent_recording_count - len(eligible_ids)
                        ),
                        **result,
                        "q_value": None,
                        "multiplicity_method": (
                            "benjamini_hochberg_within_declared_metric_family"
                        ),
                        "acquisition_batch_adjustment": "not_performed",
                    }
                )

    family_indices: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        family_indices[str(row["multiplicity_family"])].append(index)
    for indices in family_indices.values():
        adjusted = benjamini_hochberg([rows[index]["p_value"] for index in indices])
        for index, q_value in zip(indices, adjusted):
            rows[index]["q_value"] = q_value
    return tuple(rows)


def _histogram_query_identity(
    dataset: ValidatedBehaviorExportDataset,
    spec: ValidatedBehaviorHistogramSpec,
) -> dict[str, object]:
    columns = (
        "recording_id",
        spec.condition_column,
        *spec.group_columns,
        *spec.identity_columns,
        *spec.membership_columns,
        *spec.validity_columns,
        *(axis.source_column for axis in spec.axes),
    )
    table = dataset.table(spec.source_table)
    identity = table.query_identity(
        columns=columns,
        predicate_description=(
            "exact declared conditions only; null condition rows retained as "
            "nonmember audit evidence; conjunction of declared membership and "
            "validity columns; fixed-width terminal-right-closed binning; one "
            "histogram normalized within each recording/condition/group"
        ),
    )
    body = {
        **identity,
        "histogram_spec_sha256": spec.spec_sha256,
        "recording_reducer": spec.recording_reducer,
        "cohort_reducer": spec.cohort_reducer,
    }
    return {**body, "query_sha256": canonical_json_sha256(body)}


def _resolved_histogram_axis(
    axis: ValidatedBehaviorHistogramAxisSpec,
    *,
    valid_maximum: float | None,
) -> dict[str, object]:
    width = float(axis.bin_width)
    lower = float(axis.lower_bound)
    if axis.coverage_policy == "fixed_closed_terminal":
        assert axis.upper_bound is not None
        upper = float(axis.upper_bound)
    else:
        if valid_maximum is None:
            _fail(
                f"Histogram axis {axis.axis_id!r} has no valid rows from which "
                "to resolve its covering range"
            )
        if not math.isfinite(valid_maximum) or valid_maximum < lower:
            _fail(f"Histogram axis {axis.axis_id!r} has an invalid maximum")
        count = max(1, int(math.ceil((valid_maximum - lower) / width)))
        upper = lower + float(count) * width
    bin_count = int(round((upper - lower) / width))
    edges = [lower + float(index) * width for index in range(bin_count + 1)]
    record = {
        **axis.to_dict(),
        "resolved_upper_bound": upper,
        "bin_count": bin_count,
        "bin_edges": edges,
        "resolution_scope": (
            "declared_fixed_range"
            if axis.coverage_policy == "fixed_closed_terminal"
            else "all_exact_valid_rows_in_selected_export"
        ),
    }
    return {**record, "axis_recipe_sha256": canonical_json_sha256(record)}


def _histogram_bin_expression(
    axis: ValidatedBehaviorHistogramAxisSpec,
    resolved: Mapping[str, object],
    *,
    output_name: str,
) -> Any:
    import polars as pl

    lower = float(axis.lower_bound)
    width = float(axis.bin_width)
    count = int(resolved["bin_count"])
    return (
        ((pl.col(axis.source_column).cast(pl.Float64) - lower) / width)
        .floor()
        .cast(pl.Int64)
        .clip(0, count - 1)
        .alias(output_name)
    )


def _compute_histogram_spec(
    dataset: ValidatedBehaviorExportDataset,
    config: ValidatedBehaviorGroupStatisticsConfig,
    spec: ValidatedBehaviorHistogramSpec,
    *,
    parent_recording_count: int,
) -> tuple[
    Mapping[str, object],
    tuple[Mapping[str, object], ...],
    tuple[Mapping[str, object], ...],
]:
    import polars as pl

    table = dataset.table(spec.source_table)
    selected = (
        "recording_id",
        spec.condition_column,
        *spec.group_columns,
        *spec.identity_columns,
        *spec.membership_columns,
        *spec.validity_columns,
        *(axis.source_column for axis in spec.axes),
    )
    if len(set(selected)) != len(selected):
        _fail(f"{spec.metric_id}: histogram query selects duplicate columns")
    known = {item.name for item in table.spec.contract.fields}
    missing = sorted(set(selected) - known)
    if missing:
        _fail(f"{spec.source_table}: histogram columns are absent: {missing}")

    lazy = table.scan(columns=selected).with_columns(
        pl.col(spec.condition_column).cast(pl.String).alias("__condition")
    )
    expected = tuple(spec.expected_conditions)
    condition_member = pl.col("__condition").is_in(expected)
    unexpected = (
        lazy.filter(pl.col("__condition").is_not_null() & ~condition_member)
        .limit(1)
        .collect()
    )
    if unexpected.height:
        _fail(
            f"{spec.source_table}: histogram query found an undeclared non-null "
            f"condition: {unexpected.to_dicts()[0]!r}"
        )
    observed = {
        str(value)
        for value in lazy.filter(condition_member)
        .select("__condition")
        .unique()
        .collect()
        .get_column("__condition")
    }
    if observed != set(expected):
        _fail(
            f"{spec.source_table}: histogram condition registry "
            f"{sorted(observed)!r} does not equal {sorted(expected)!r}"
        )

    unit_keys = ("recording_id", "__condition", *spec.group_columns)
    semantic_columns = (*spec.group_columns, *spec.identity_columns)
    null_semantic = pl.any_horizontal(
        [pl.col(name).is_null() for name in semantic_columns]
    )
    bad_semantic = lazy.filter(condition_member & null_semantic).limit(1).collect()
    if bad_semantic.height:
        _fail(
            f"{spec.source_table}: an exact histogram condition has a null "
            f"group or identity: {bad_semantic.to_dicts()[0]!r}"
        )
    flag_columns = (*spec.membership_columns, *spec.validity_columns)
    null_flag = pl.any_horizontal([pl.col(name).is_null() for name in flag_columns])
    bad_flag = lazy.filter(condition_member & null_flag).limit(1).collect()
    if bad_flag.height:
        _fail(
            f"{spec.source_table}: an exact histogram condition has a null "
            f"membership or validity flag: {bad_flag.to_dicts()[0]!r}"
        )

    membership = pl.all_horizontal([pl.col(name) for name in spec.membership_columns])
    scientific_validity = pl.all_horizontal(
        [pl.col(name) for name in spec.validity_columns]
    )
    selected_valid = condition_member & membership & scientific_validity
    for axis in spec.axes:
        value = pl.col(axis.source_column).cast(pl.Float64)
        invalid = ~value.is_finite() | (value < float(axis.lower_bound))
        if axis.upper_bound is not None:
            invalid = invalid | (value > float(axis.upper_bound))
        bad_value = lazy.filter(selected_valid & invalid).limit(1).collect()
        if bad_value.height:
            _fail(
                f"{spec.source_table}: declared-valid histogram axis "
                f"{axis.axis_id!r} is outside its contract: "
                f"{bad_value.to_dicts()[0]!r}"
            )

    audit_expressions: list[Any] = [
        pl.len().alias("source_row_count"),
        condition_member.sum().alias("exact_condition_row_count"),
        pl.col("__condition").is_null().sum().alias("null_condition_row_count"),
        (pl.col("__condition").is_null() & membership & scientific_validity)
        .sum()
        .alias("null_condition_membership_valid_row_count"),
        (condition_member & ~membership).sum().alias("excluded_membership_row_count"),
        (condition_member & membership & ~scientific_validity)
        .sum()
        .alias("excluded_validity_row_count"),
        selected_valid.sum().alias("valid_denominator_row_count"),
    ]
    for index, axis in enumerate(spec.axes):
        audit_expressions.append(
            pl.col(axis.source_column)
            .cast(pl.Float64)
            .filter(selected_valid)
            .max()
            .alias(f"axis_{index}_valid_maximum")
        )
    audit = lazy.select(audit_expressions).collect().to_dicts()[0]
    if int(audit["valid_denominator_row_count"]) <= 0:
        _fail(f"{spec.metric_id}: histogram has no valid source rows")

    resolved_axes: list[dict[str, object]] = []
    for index, axis in enumerate(spec.axes):
        maximum = _finite_float(audit[f"axis_{index}_valid_maximum"])
        resolved_axes.append(_resolved_histogram_axis(axis, valid_maximum=maximum))

    candidate = lazy.filter(condition_member)
    identity_aggregations = [pl.len().alias("__candidate_row_count")]
    for name in spec.identity_columns:
        identity_aggregations.extend(
            (
                pl.col(name).n_unique().alias(f"__identity_count_{name}"),
                pl.col(name).first().alias(name),
            )
        )
    strata = candidate.group_by(unit_keys).agg(identity_aggregations)
    ambiguous_identity = pl.any_horizontal(
        [pl.col(f"__identity_count_{name}") != 1 for name in spec.identity_columns]
    )
    ambiguous = strata.filter(ambiguous_identity).limit(1).collect()
    if ambiguous.height:
        _fail(
            f"{spec.source_table}: recording/condition/group does not map to "
            f"one exact histogram identity: {ambiguous.to_dicts()[0]!r}"
        )
    strata = strata.drop(
        *[f"__identity_count_{name}" for name in spec.identity_columns]
    )

    indexed = lazy.filter(selected_valid).with_columns(
        [
            _histogram_bin_expression(
                axis,
                resolved_axes[index],
                output_name=f"__axis_{index}_bin_index",
            )
            for index, axis in enumerate(spec.axes)
        ]
    )
    bin_index_columns = tuple(
        f"__axis_{index}_bin_index" for index in range(len(spec.axes))
    )
    denominators = indexed.group_by(unit_keys).agg(
        pl.len().alias("__denominator_count")
    )
    counts = indexed.group_by((*unit_keys, *bin_index_columns)).agg(
        pl.len().alias("__bin_count")
    )

    bin_records: list[dict[str, object]] = []
    ranges = [range(int(axis["bin_count"])) for axis in resolved_axes]
    for indices in product(*ranges):
        row: dict[str, object] = {}
        for axis_index, bin_index in enumerate(indices):
            edges = resolved_axes[axis_index]["bin_edges"]
            assert isinstance(edges, list)
            row[f"__axis_{axis_index}_bin_index"] = int(bin_index)
            row[f"__axis_{axis_index}_bin_start"] = float(edges[bin_index])
            row[f"__axis_{axis_index}_bin_end"] = float(edges[bin_index + 1])
        bin_records.append(row)
    bin_grid = pl.DataFrame(bin_records)
    expanded = (
        strata.join(bin_grid.lazy(), how="cross")
        .join(denominators, on=unit_keys, how="left")
        .join(counts, on=(*unit_keys, *bin_index_columns), how="left")
        .with_columns(
            pl.col("__denominator_count").fill_null(0).cast(pl.Int64),
            pl.col("__bin_count").fill_null(0).cast(pl.Int64),
        )
        .with_columns(
            pl.when(pl.col("__denominator_count") > 0)
            .then(
                pl.col("__bin_count").cast(pl.Float64)
                / pl.col("__denominator_count").cast(pl.Float64)
            )
            .otherwise(None)
            .alias("__fraction")
        )
        .sort((*unit_keys, *bin_index_columns))
        .collect()
    )

    query = _histogram_query_identity(dataset, spec)
    recipe_body = {
        "metric_id": spec.metric_id,
        "metric_family": spec.metric_family,
        "histogram_spec_sha256": spec.spec_sha256,
        "source_query_sha256": query["query_sha256"],
        "resolved_axes": resolved_axes,
        "source_audit": {
            key: _plain(value)
            for key, value in audit.items()
            if not key.startswith("axis_")
        },
        "normalization": "bin_count_divided_by_recording_condition_group_denominator",
        "cohort_weighting": "equal_weight_per_finite_recording",
        "source_identity_use": (
            "validated_one_identity_per_recording_condition_semantic_group_"
            "then_excluded_from_cohort_grouping"
        ),
        "null_condition_policy": "retained_as_nonmember_source_audit_evidence",
        "interpolation": "prohibited",
    }
    recipe = {
        **recipe_body,
        "histogram_recipe_sha256": canonical_json_sha256(recipe_body),
    }

    recording_rows: list[dict[str, object]] = []
    for raw in expanded.to_dicts():
        group = _group_record(raw, spec.group_columns)
        group_json, group_sha256 = _group_identity(group)
        identity = _group_record(raw, spec.identity_columns)
        identity_json, identity_sha256 = _group_identity(identity)
        axis_1 = resolved_axes[1] if len(resolved_axes) == 2 else None
        denominator = int(raw["__denominator_count"])
        fraction = _finite_float(raw["__fraction"])
        recording_rows.append(
            {
                "statistics_run_id": config.statistics_run_id,
                "source_export_run_id": dataset.export_run_id,
                "source_export_manifest_sha256": dataset.cache_identity,
                "metric_id": spec.metric_id,
                "metric_spec_sha256": spec.spec_sha256,
                "histogram_recipe_sha256": recipe["histogram_recipe_sha256"],
                "metric_family": spec.metric_family,
                "source_table": spec.source_table,
                "source_query_sha256": query["query_sha256"],
                "analysis_status": spec.analysis_status,
                "recording_id": str(raw["recording_id"]),
                "condition_axis": spec.condition_column,
                "condition": str(raw["__condition"]),
                "group_key_json": group_json,
                "group_key_sha256": group_sha256,
                "source_identity_key_json": identity_json,
                "source_identity_key_sha256": identity_sha256,
                "axis_0_id": str(resolved_axes[0]["axis_id"]),
                "axis_0_unit": str(resolved_axes[0]["unit"]),
                "axis_0_bin_index": int(raw["__axis_0_bin_index"]),
                "axis_0_bin_start": float(raw["__axis_0_bin_start"]),
                "axis_0_bin_end": float(raw["__axis_0_bin_end"]),
                "axis_1_id": None if axis_1 is None else str(axis_1["axis_id"]),
                "axis_1_unit": None if axis_1 is None else str(axis_1["unit"]),
                "axis_1_bin_index": (
                    None if axis_1 is None else int(raw["__axis_1_bin_index"])
                ),
                "axis_1_bin_start": (
                    None if axis_1 is None else float(raw["__axis_1_bin_start"])
                ),
                "axis_1_bin_end": (
                    None if axis_1 is None else float(raw["__axis_1_bin_end"])
                ),
                "candidate_row_count": int(raw["__candidate_row_count"]),
                "bin_count": int(raw["__bin_count"]),
                "denominator_count": denominator,
                "fraction": fraction,
                "value_state": "finite" if denominator > 0 else "zero_denominator",
                "recording_reducer": spec.recording_reducer,
            }
        )

    recording_frame = pl.DataFrame(recording_rows)
    summary_keys = (
        "condition",
        "group_key_json",
        "group_key_sha256",
        "axis_0_id",
        "axis_0_unit",
        "axis_0_bin_index",
        "axis_0_bin_start",
        "axis_0_bin_end",
        "axis_1_id",
        "axis_1_unit",
        "axis_1_bin_index",
        "axis_1_bin_start",
        "axis_1_bin_end",
    )
    summary = (
        recording_frame.lazy()
        .group_by(summary_keys)
        .agg(
            pl.len().alias("__contributor_count"),
            pl.col("fraction").count().alias("__finite_count"),
            pl.col("bin_count").sum().alias("__source_bin_count_sum"),
            pl.col("denominator_count").sum().alias("__source_denominator_count_sum"),
            pl.col("fraction").mean().alias("__mean"),
            pl.col("fraction").median().alias("__median"),
            pl.col("fraction").std(ddof=1).alias("__std"),
            pl.col("fraction").min().alias("__minimum"),
            pl.col("fraction").quantile(0.25, interpolation="linear").alias("__p25"),
            pl.col("fraction").quantile(0.75, interpolation="linear").alias("__p75"),
            pl.col("fraction").max().alias("__maximum"),
        )
        .sort(summary_keys)
        .collect()
    )
    descriptive_rows: list[dict[str, object]] = []
    for raw in summary.to_dicts():
        contributor_count = int(raw["__contributor_count"])
        finite_count = int(raw["__finite_count"])
        std = _finite_float(raw["__std"])
        sem = (
            std / math.sqrt(float(finite_count))
            if std is not None and finite_count > 0
            else None
        )
        descriptive_rows.append(
            {
                "statistics_run_id": config.statistics_run_id,
                "source_export_run_id": dataset.export_run_id,
                "source_export_manifest_sha256": dataset.cache_identity,
                "metric_id": spec.metric_id,
                "metric_spec_sha256": spec.spec_sha256,
                "histogram_recipe_sha256": recipe["histogram_recipe_sha256"],
                "metric_family": spec.metric_family,
                "source_table": spec.source_table,
                "source_table_contract_sha256": query["table_contract_sha256"],
                "source_query_sha256": query["query_sha256"],
                "analysis_status": spec.analysis_status,
                "condition_axis": spec.condition_column,
                **{key: raw[key] for key in summary_keys},
                "parent_recording_count": parent_recording_count,
                "contributor_recording_count": contributor_count,
                "finite_recording_count": finite_count,
                "excluded_zero_denominator_recording_count": (
                    contributor_count - finite_count
                ),
                "noncontributor_recording_count": (
                    parent_recording_count - contributor_count
                ),
                "source_bin_count_sum": int(raw["__source_bin_count_sum"]),
                "source_denominator_count_sum": int(
                    raw["__source_denominator_count_sum"]
                ),
                "mean_fraction": _finite_float(raw["__mean"]),
                "median_fraction": _finite_float(raw["__median"]),
                "sample_std_fraction": std,
                "sem_fraction": _finite_float(sem),
                "minimum_fraction": _finite_float(raw["__minimum"]),
                "p25_fraction": _finite_float(raw["__p25"]),
                "p75_fraction": _finite_float(raw["__p75"]),
                "maximum_fraction": _finite_float(raw["__maximum"]),
                "cohort_weighting": "equal_weight_per_finite_recording",
            }
        )
    return recipe, tuple(recording_rows), tuple(descriptive_rows)


def compute_validated_behavior_group_statistics(
    dataset: ValidatedBehaviorExportDataset,
    config: ValidatedBehaviorGroupStatisticsConfig,
) -> ValidatedBehaviorGroupStatisticsResult:
    """Compute descriptive and exploratory paired recording-level statistics."""

    specs = validate_metric_specs(config.metric_specs, allow_empty=True)
    histogram_specs = validate_histogram_specs(config.histogram_specs, allow_empty=True)
    missing_tables = sorted(
        {spec.source_table for spec in (*specs, *histogram_specs)}
        - set(dataset.table_names)
    )
    if missing_tables:
        _fail(f"Source export lacks requested metric tables: {missing_tables}")
    cohort_summary, _recording_ids = _validate_analysis_units(dataset)
    parent_count = int(cohort_summary["parent_recording_count"])

    all_recording_rows: list[Mapping[str, object]] = []
    all_descriptive_rows: list[Mapping[str, object]] = []
    all_contrast_rows: list[Mapping[str, object]] = []
    all_histogram_recipes: list[Mapping[str, object]] = []
    all_recording_histogram_rows: list[Mapping[str, object]] = []
    all_histogram_descriptive_rows: list[Mapping[str, object]] = []
    for batch in _metric_batches(specs):
        units = _unit_lazy_frame(dataset, batch)
        descriptive = _descriptive_records(
            dataset,
            config,
            batch,
            units,
            parent_recording_count=parent_count,
        )
        recording = _recording_value_records(dataset, config, batch, units)
        contrasts = _contrast_records(
            dataset,
            config,
            batch,
            recording,
            parent_recording_count=parent_count,
        )
        all_descriptive_rows.extend(descriptive)
        all_recording_rows.extend(recording)
        all_contrast_rows.extend(contrasts)

    for spec in histogram_specs:
        recipe, recording, descriptive = _compute_histogram_spec(
            dataset,
            config,
            spec,
            parent_recording_count=parent_count,
        )
        all_histogram_recipes.append(recipe)
        all_recording_histogram_rows.extend(recording)
        all_histogram_descriptive_rows.extend(descriptive)

    source_queries_by_digest: dict[str, Mapping[str, object]] = {}
    for spec in specs:
        query = _query_identity(dataset, spec)
        source_queries_by_digest[str(query["query_sha256"])] = query
    for spec in histogram_specs:
        query = _histogram_query_identity(dataset, spec)
        source_queries_by_digest[str(query["query_sha256"])] = query
    source_export = {
        "root": str(dataset.root),
        "export_run_id": dataset.export_run_id,
        "export_manifest_record_sha256": dataset.cache_identity,
        "export_plan_sha256": dataset.manifest["export_plan"]["plan_sha256"],
        "profile_id": dataset.manifest.get("export_profile", {}).get("profile_id"),
        "validation_mode": dataset.validation_mode,
        "analysis_unit_policy_sha256": dataset.manifest["analysis_unit_policy"][
            "sha256"
        ],
    }
    chaser_appearance = build_chaser_appearance_dimension(dataset)
    if chaser_appearance is not None:
        source_export["chaser_appearance_dimension"] = chaser_appearance
    return ValidatedBehaviorGroupStatisticsResult(
        config=config,
        source_export=MappingProxyType(source_export),
        cohort_summary=MappingProxyType(cohort_summary),
        source_queries=tuple(
            source_queries_by_digest[key] for key in sorted(source_queries_by_digest)
        ),
        recording_values=tuple(all_recording_rows),
        descriptive_statistics=tuple(all_descriptive_rows),
        paired_contrasts=tuple(all_contrast_rows),
        histogram_recipes=tuple(all_histogram_recipes),
        recording_histogram_bins=tuple(all_recording_histogram_rows),
        histogram_descriptive_statistics=tuple(all_histogram_descriptive_rows),
    )


_RECORDING_VALUE_SCHEMA = (
    ("statistics_run_id", "string", False),
    ("source_export_run_id", "string", False),
    ("source_export_manifest_sha256", "string", False),
    ("metric_id", "string", False),
    ("metric_spec_sha256", "string", False),
    ("metric_family", "string", False),
    ("source_table", "string", False),
    ("source_value_column", "string", False),
    ("source_query_sha256", "string", False),
    ("unit", "string", False),
    ("analysis_status", "string", False),
    ("recording_id", "string", False),
    ("condition_axis", "string", False),
    ("condition", "string", False),
    ("group_key_json", "string", False),
    ("group_key_sha256", "string", False),
    ("value", "float64", True),
    ("value_state", "string", False),
    ("recording_reducer", "string", False),
)

_DESCRIPTIVE_SCHEMA = (
    ("statistics_run_id", "string", False),
    ("source_export_run_id", "string", False),
    ("source_export_manifest_sha256", "string", False),
    ("metric_id", "string", False),
    ("metric_spec_sha256", "string", False),
    ("metric_family", "string", False),
    ("source_table", "string", False),
    ("source_value_column", "string", False),
    ("source_table_contract_sha256", "string", False),
    ("source_query_sha256", "string", False),
    ("unit", "string", False),
    ("analysis_status", "string", False),
    ("condition_axis", "string", False),
    ("condition", "string", False),
    ("group_key_json", "string", False),
    ("group_key_sha256", "string", False),
    ("parent_recording_count", "int64", False),
    ("contributor_recording_count", "int64", False),
    ("finite_recording_count", "int64", False),
    ("excluded_nonfinite_recording_count", "int64", False),
    ("noncontributor_recording_count", "int64", False),
    ("mean", "float64", True),
    ("median", "float64", True),
    ("sample_std", "float64", True),
    ("sem", "float64", True),
    ("minimum", "float64", True),
    ("p25", "float64", True),
    ("p75", "float64", True),
    ("maximum", "float64", True),
    ("cohort_weighting", "string", False),
)

_CONTRAST_SCHEMA = (
    ("statistics_run_id", "string", False),
    ("source_export_run_id", "string", False),
    ("source_export_manifest_sha256", "string", False),
    ("metric_id", "string", False),
    ("metric_spec_sha256", "string", False),
    ("metric_family", "string", False),
    ("multiplicity_family", "string", False),
    ("source_table", "string", False),
    ("source_value_column", "string", False),
    ("source_query_sha256", "string", False),
    ("unit", "string", False),
    ("analysis_status", "string", False),
    ("condition_axis", "string", False),
    ("contrast_id", "string", False),
    ("condition_a", "string", False),
    ("condition_b", "string", False),
    ("difference_direction", "string", False),
    ("group_key_json", "string", False),
    ("group_key_sha256", "string", False),
    ("parent_recording_count", "int64", False),
    ("eligible_recording_count", "int64", False),
    ("noncontributor_recording_count", "int64", False),
    ("unit_count", "int64", False),
    ("paired_unit_count", "int64", False),
    ("excluded_unit_count", "int64", False),
    ("mean_a", "float64", True),
    ("mean_b", "float64", True),
    ("mean_difference", "float64", True),
    ("median_difference", "float64", True),
    ("std_difference", "float64", True),
    ("effect_size", "float64", True),
    ("ci_low", "float64", True),
    ("ci_high", "float64", True),
    ("p_value", "float64", True),
    ("q_value", "float64", True),
    ("test_method", "string", False),
    ("bootstrap_iterations", "int64", False),
    ("permutation_iterations", "int64", False),
    ("status", "string", False),
    ("skip_reason", "string", True),
    ("multiplicity_method", "string", False),
    ("acquisition_batch_adjustment", "string", False),
)

_RECORDING_HISTOGRAM_SCHEMA = (
    ("statistics_run_id", "string", False),
    ("source_export_run_id", "string", False),
    ("source_export_manifest_sha256", "string", False),
    ("metric_id", "string", False),
    ("metric_spec_sha256", "string", False),
    ("histogram_recipe_sha256", "string", False),
    ("metric_family", "string", False),
    ("source_table", "string", False),
    ("source_query_sha256", "string", False),
    ("analysis_status", "string", False),
    ("recording_id", "string", False),
    ("condition_axis", "string", False),
    ("condition", "string", False),
    ("group_key_json", "string", False),
    ("group_key_sha256", "string", False),
    ("source_identity_key_json", "string", False),
    ("source_identity_key_sha256", "string", False),
    ("axis_0_id", "string", False),
    ("axis_0_unit", "string", False),
    ("axis_0_bin_index", "int64", False),
    ("axis_0_bin_start", "float64", False),
    ("axis_0_bin_end", "float64", False),
    ("axis_1_id", "string", True),
    ("axis_1_unit", "string", True),
    ("axis_1_bin_index", "int64", True),
    ("axis_1_bin_start", "float64", True),
    ("axis_1_bin_end", "float64", True),
    ("candidate_row_count", "int64", False),
    ("bin_count", "int64", False),
    ("denominator_count", "int64", False),
    ("fraction", "float64", True),
    ("value_state", "string", False),
    ("recording_reducer", "string", False),
)

_HISTOGRAM_DESCRIPTIVE_SCHEMA = (
    ("statistics_run_id", "string", False),
    ("source_export_run_id", "string", False),
    ("source_export_manifest_sha256", "string", False),
    ("metric_id", "string", False),
    ("metric_spec_sha256", "string", False),
    ("histogram_recipe_sha256", "string", False),
    ("metric_family", "string", False),
    ("source_table", "string", False),
    ("source_table_contract_sha256", "string", False),
    ("source_query_sha256", "string", False),
    ("analysis_status", "string", False),
    ("condition_axis", "string", False),
    ("condition", "string", False),
    ("group_key_json", "string", False),
    ("group_key_sha256", "string", False),
    ("axis_0_id", "string", False),
    ("axis_0_unit", "string", False),
    ("axis_0_bin_index", "int64", False),
    ("axis_0_bin_start", "float64", False),
    ("axis_0_bin_end", "float64", False),
    ("axis_1_id", "string", True),
    ("axis_1_unit", "string", True),
    ("axis_1_bin_index", "int64", True),
    ("axis_1_bin_start", "float64", True),
    ("axis_1_bin_end", "float64", True),
    ("parent_recording_count", "int64", False),
    ("contributor_recording_count", "int64", False),
    ("finite_recording_count", "int64", False),
    ("excluded_zero_denominator_recording_count", "int64", False),
    ("noncontributor_recording_count", "int64", False),
    ("source_bin_count_sum", "int64", False),
    ("source_denominator_count_sum", "int64", False),
    ("mean_fraction", "float64", True),
    ("median_fraction", "float64", True),
    ("sample_std_fraction", "float64", True),
    ("sem_fraction", "float64", True),
    ("minimum_fraction", "float64", True),
    ("p25_fraction", "float64", True),
    ("p75_fraction", "float64", True),
    ("maximum_fraction", "float64", True),
    ("cohort_weighting", "string", False),
)


def _arrow_schema(fields: Sequence[tuple[str, str, bool]]) -> Any:
    import pyarrow as pa

    types = {
        "string": pa.string(),
        "float64": pa.float64(),
        "int64": pa.int64(),
    }
    return pa.schema(
        [
            pa.field(name, types[data_type], nullable=nullable)
            for name, data_type, nullable in fields
        ]
    )


def _write_parquet(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    schema_fields: Sequence[tuple[str, str, bool]],
) -> dict[str, object]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = _arrow_schema(schema_fields)
    table = pa.Table.from_pylist([dict(row) for row in rows], schema=schema)
    pq.write_table(
        table,
        path,
        compression="zstd",
        row_group_size=65536,
        write_statistics=True,
    )
    return {
        "path": path.name,
        "row_count": table.num_rows,
        "size_bytes": path.stat().st_size,
        "file_sha256": sha256_file(path),
        "arrow_schema_sha256": canonical_json_sha256(
            {
                "fields": [
                    {
                        "name": field.name,
                        "type": str(field.type),
                        "nullable": field.nullable,
                    }
                    for field in schema
                ]
            }
        ),
    }


def _validate_group_identities(
    frame: Any,
    *,
    table_name: str,
    json_column: str = "group_key_json",
    sha256_column: str = "group_key_sha256",
) -> None:
    pairs = frame.select((json_column, sha256_column)).unique()
    for row in pairs.to_dicts():
        try:
            group = json.loads(str(row[json_column]))
        except json.JSONDecodeError:
            _fail(f"{table_name}: group key is not strict JSON")
        if not isinstance(group, dict):
            _fail(f"{table_name}: group key must decode to one object")
        if canonical_json_sha256(group) != row[sha256_column]:
            _fail(f"{table_name}: group-key digest is stale")


def _validate_statistics_directory(
    root: Path,
    manifest: Mapping[str, object],
) -> None:
    import polars as pl
    import pyarrow.parquet as pq

    version = manifest.get("schema_version")
    if manifest.get("schema_id") != SCHEMA_ID or version not in {
        LEGACY_SCHEMA_VERSION,
        SCHEMA_VERSION,
    }:
        _fail("Grouped-statistics manifest schema is unsupported")
    expected_method = (
        LEGACY_METHOD_ID if version == LEGACY_SCHEMA_VERSION else METHOD_ID
    )
    if (
        manifest.get("method_id") != expected_method
        or manifest.get("status") != SANDBOX_STATUS
    ):
        _fail("Grouped-statistics manifest method or status is unsupported")
    for field in (
        "selector_eligible",
        "production_authority",
        "selector_activation",
        "registry_update",
        "source_export_mutation",
    ):
        if manifest.get(field) is not False:
            _fail(f"Grouped-statistics sandbox safety flag is not false: {field}")
    body = {key: value for key, value in manifest.items() if key != "record_sha256"}
    if manifest.get("record_sha256") != canonical_json_sha256(body):
        _fail("Grouped-statistics manifest record digest is stale")

    output_records = manifest.get("outputs")
    if not isinstance(output_records, list):
        _fail("Grouped-statistics manifest lacks output records")
    expected = {
        "recording_metric_values.parquet": _RECORDING_VALUE_SCHEMA,
        "descriptive_statistics.parquet": _DESCRIPTIVE_SCHEMA,
        "paired_contrasts.parquet": _CONTRAST_SCHEMA,
    }
    if version == SCHEMA_VERSION:
        expected.update(
            {
                "recording_histogram_bins.parquet": (_RECORDING_HISTOGRAM_SCHEMA),
                "histogram_descriptive_statistics.parquet": (
                    _HISTOGRAM_DESCRIPTIVE_SCHEMA
                ),
            }
        )
    by_name = {
        str(record.get("path")): record
        for record in output_records
        if isinstance(record, Mapping)
    }
    if set(by_name) != set(expected):
        _fail("Grouped-statistics output inventory is not exact")
    actual_names = {path.name for path in root.iterdir()}
    if actual_names != {"manifest.json", *expected}:
        _fail("Grouped-statistics directory contains an unrecorded file")

    statistics_run_id = manifest.get("statistics_run_id")
    source_export = manifest.get("source_export")
    if not isinstance(source_export, Mapping):
        _fail("Grouped-statistics manifest lacks source-export identity")
    source_digest = source_export.get("export_manifest_record_sha256")
    chaser_appearance = source_export.get("chaser_appearance_dimension")
    if chaser_appearance is not None:
        validate_chaser_appearance_dimension(
            chaser_appearance,
            expected_export_manifest_sha256=str(source_digest),
        )
    primary_keys = {
        "recording_metric_values.parquet": (
            "statistics_run_id",
            "metric_id",
            "recording_id",
            "condition",
            "group_key_sha256",
        ),
        "descriptive_statistics.parquet": (
            "statistics_run_id",
            "metric_id",
            "condition",
            "group_key_sha256",
        ),
        "paired_contrasts.parquet": (
            "statistics_run_id",
            "metric_id",
            "contrast_id",
            "group_key_sha256",
        ),
    }
    if version == SCHEMA_VERSION:
        primary_keys.update(
            {
                "recording_histogram_bins.parquet": (
                    "statistics_run_id",
                    "metric_id",
                    "recording_id",
                    "condition",
                    "group_key_sha256",
                    "axis_0_bin_index",
                    "axis_1_bin_index",
                ),
                "histogram_descriptive_statistics.parquet": (
                    "statistics_run_id",
                    "metric_id",
                    "condition",
                    "group_key_sha256",
                    "axis_0_bin_index",
                    "axis_1_bin_index",
                ),
            }
        )
    recipe_digests: set[str] = set()
    if version == SCHEMA_VERSION:
        configuration = manifest.get("configuration")
        if not isinstance(configuration, Mapping):
            _fail("Grouped-statistics manifest lacks its v2 configuration")
        histogram_specs = configuration.get("histogram_specs")
        recipes = manifest.get("histogram_recipes")
        if not isinstance(histogram_specs, list) or not isinstance(recipes, list):
            _fail("Grouped-statistics v2 manifest lacks histogram contracts")
        spec_ids = {
            str(record.get("metric_id"))
            for record in histogram_specs
            if isinstance(record, Mapping)
        }
        recipe_ids: set[str] = set()
        for recipe in recipes:
            if not isinstance(recipe, Mapping):
                _fail("Grouped-statistics histogram recipe is malformed")
            digest = recipe.get("histogram_recipe_sha256")
            recipe_body = {
                key: value
                for key, value in recipe.items()
                if key != "histogram_recipe_sha256"
            }
            if digest != canonical_json_sha256(recipe_body):
                _fail("Grouped-statistics histogram recipe digest is stale")
            metric_id = str(recipe.get("metric_id"))
            recipe_ids.add(metric_id)
            recipe_digests.add(str(digest))
        if spec_ids != recipe_ids or len(spec_ids) != len(histogram_specs):
            _fail("Histogram specifications and resolved recipes do not match")
    for name, schema_fields in expected.items():
        record = by_name[name]
        path = root / name
        if Path(name).name != name or not path.is_file():
            _fail(f"Grouped-statistics output path is unsafe or missing: {name}")
        if path.stat().st_size != record.get("size_bytes"):
            _fail(f"{name}: file size does not match the manifest")
        if sha256_file(path) != record.get("file_sha256"):
            _fail(f"{name}: file digest does not match the manifest")
        parquet = pq.ParquetFile(path)
        if parquet.metadata.num_rows != record.get("row_count"):
            _fail(f"{name}: row count does not match the manifest")
        expected_schema = _arrow_schema(schema_fields)
        if not parquet.schema_arrow.equals(expected_schema, check_metadata=False):
            _fail(f"{name}: Arrow schema does not match the exact contract")
        schema_sha256 = canonical_json_sha256(
            {
                "fields": [
                    {
                        "name": field.name,
                        "type": str(field.type),
                        "nullable": field.nullable,
                    }
                    for field in expected_schema
                ]
            }
        )
        if schema_sha256 != record.get("arrow_schema_sha256"):
            _fail(f"{name}: Arrow schema digest is stale")

        frame = pl.read_parquet(path)
        duplicate = (
            frame.group_by(primary_keys[name])
            .agg(pl.len().alias("row_count"))
            .filter(pl.col("row_count") != 1)
        )
        if duplicate.height:
            _fail(f"{name}: primary key is not unique")
        if frame.height:
            if frame.get_column("statistics_run_id").unique().to_list() != [
                statistics_run_id
            ]:
                _fail(f"{name}: statistics-run identity is inconsistent")
            if frame.get_column("source_export_manifest_sha256").unique().to_list() != [
                source_digest
            ]:
                _fail(f"{name}: source-export identity is inconsistent")
            if frame.get_column("analysis_status").unique().to_list() != [
                "exploratory"
            ]:
                _fail(f"{name}: result claims an unsupported analysis status")
            _validate_group_identities(frame, table_name=name)
            if name == "recording_histogram_bins.parquet":
                _validate_group_identities(
                    frame,
                    table_name=name,
                    json_column="source_identity_key_json",
                    sha256_column="source_identity_key_sha256",
                )
            if "histogram_recipe_sha256" in frame.columns and not set(
                frame.get_column("histogram_recipe_sha256").unique().to_list()
            ).issubset(recipe_digests):
                _fail(f"{name}: histogram recipe identity is unrecorded")

        if name == "recording_metric_values.parquet" and frame.height:
            bad_state = frame.filter(
                ((pl.col("value_state") == "finite") & pl.col("value").is_null())
                | ((pl.col("value_state") != "finite") & pl.col("value").is_not_null())
                | ~pl.col("value_state").is_in(
                    ("finite", "source_null", "source_nonfinite")
                )
            )
            if bad_state.height:
                _fail(f"{name}: value and explicit validity state disagree")
        elif name == "descriptive_statistics.parquet" and frame.height:
            bad_counts = frame.filter(
                (
                    pl.col("finite_recording_count")
                    + pl.col("excluded_nonfinite_recording_count")
                    != pl.col("contributor_recording_count")
                )
                | (
                    pl.col("contributor_recording_count")
                    + pl.col("noncontributor_recording_count")
                    != pl.col("parent_recording_count")
                )
            )
            if bad_counts.height:
                _fail(f"{name}: recording denominator accounting is inconsistent")
        elif name == "paired_contrasts.parquet" and frame.height:
            bad_counts = frame.filter(
                (pl.col("eligible_recording_count") != pl.col("unit_count"))
                | (
                    pl.col("paired_unit_count") + pl.col("excluded_unit_count")
                    != pl.col("unit_count")
                )
                | (
                    pl.col("eligible_recording_count")
                    + pl.col("noncontributor_recording_count")
                    != pl.col("parent_recording_count")
                )
            )
            bad_probability = frame.filter(
                pl.any_horizontal(
                    [
                        pl.col(field).is_not_null()
                        & ~pl.col(field).is_between(0.0, 1.0, closed="both")
                        for field in ("p_value", "q_value")
                    ]
                )
            )
            if bad_counts.height or bad_probability.height:
                _fail(f"{name}: contrast denominators or probabilities are invalid")
        elif name == "recording_histogram_bins.parquet" and frame.height:
            bad_state = frame.filter(
                (
                    (pl.col("value_state") == "finite")
                    & (
                        (pl.col("denominator_count") <= 0)
                        | pl.col("fraction").is_null()
                    )
                )
                | (
                    (pl.col("value_state") == "zero_denominator")
                    & (
                        (pl.col("denominator_count") != 0)
                        | pl.col("fraction").is_not_null()
                        | (pl.col("bin_count") != 0)
                    )
                )
                | ~pl.col("value_state").is_in(("finite", "zero_denominator"))
                | (pl.col("bin_count") < 0)
                | (pl.col("denominator_count") < 0)
                | (
                    pl.col("fraction").is_not_null()
                    & ~pl.col("fraction").is_between(0.0, 1.0, closed="both")
                )
            )
            histogram_units = frame.group_by(
                (
                    "metric_id",
                    "recording_id",
                    "condition",
                    "group_key_sha256",
                )
            ).agg(
                pl.col("denominator_count").n_unique().alias("denominator_n"),
                pl.col("denominator_count").first().alias("denominator"),
                pl.col("bin_count").sum().alias("count_sum"),
                pl.col("fraction").sum().alias("fraction_sum"),
            )
            bad_unit = histogram_units.filter(
                (pl.col("denominator_n") != 1)
                | (pl.col("count_sum") != pl.col("denominator"))
                | (
                    (pl.col("denominator") > 0)
                    & ((pl.col("fraction_sum") - 1.0).abs() > 1e-9)
                )
            )
            if bad_state.height or bad_unit.height:
                _fail(f"{name}: histogram denominator accounting is invalid")
        elif name == "histogram_descriptive_statistics.parquet" and frame.height:
            bad_counts = frame.filter(
                (
                    pl.col("finite_recording_count")
                    + pl.col("excluded_zero_denominator_recording_count")
                    != pl.col("contributor_recording_count")
                )
                | (
                    pl.col("contributor_recording_count")
                    + pl.col("noncontributor_recording_count")
                    != pl.col("parent_recording_count")
                )
            )
            fraction_columns = (
                "mean_fraction",
                "median_fraction",
                "minimum_fraction",
                "p25_fraction",
                "p75_fraction",
                "maximum_fraction",
            )
            bad_fraction = frame.filter(
                pl.any_horizontal(
                    [
                        pl.col(column).is_not_null()
                        & ~pl.col(column).is_between(0.0, 1.0, closed="both")
                        for column in fraction_columns
                    ]
                )
            )
            if bad_counts.height or bad_fraction.height:
                _fail(f"{name}: histogram cohort accounting is invalid")


def read_validated_behavior_group_statistics_sandbox(
    output_dir: str | Path,
) -> Mapping[str, object]:
    """Read and fully validate one immutable sandbox statistics generation."""

    root = Path(output_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidatedBehaviorGroupStatisticsError(
            f"Cannot read grouped-statistics manifest: {manifest_path}"
        ) from exc
    if not isinstance(manifest, dict):
        _fail("Grouped-statistics manifest must be one JSON object")
    _validate_statistics_directory(root, manifest)
    return MappingProxyType({**manifest, "manifest_path": str(manifest_path)})


def write_validated_behavior_group_statistics_sandbox(
    result: ValidatedBehaviorGroupStatisticsResult,
    output_dir: str | Path,
) -> Mapping[str, object]:
    """Atomically write one selector-ineligible sandbox statistics generation."""

    target = Path(output_dir).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite statistics output: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    )
    try:
        assert temporary is not None
        outputs = (
            _write_parquet(
                temporary / "recording_metric_values.parquet",
                result.recording_values,
                _RECORDING_VALUE_SCHEMA,
            ),
            _write_parquet(
                temporary / "descriptive_statistics.parquet",
                result.descriptive_statistics,
                _DESCRIPTIVE_SCHEMA,
            ),
            _write_parquet(
                temporary / "paired_contrasts.parquet",
                result.paired_contrasts,
                _CONTRAST_SCHEMA,
            ),
            _write_parquet(
                temporary / "recording_histogram_bins.parquet",
                result.recording_histogram_bins,
                _RECORDING_HISTOGRAM_SCHEMA,
            ),
            _write_parquet(
                temporary / "histogram_descriptive_statistics.parquet",
                result.histogram_descriptive_statistics,
                _HISTOGRAM_DESCRIPTIVE_SCHEMA,
            ),
        )
        body: dict[str, object] = {
            "schema_id": SCHEMA_ID,
            "schema_version": SCHEMA_VERSION,
            "method_id": METHOD_ID,
            "status": SANDBOX_STATUS,
            "statistics_run_id": result.config.statistics_run_id,
            "source_export": _plain(result.source_export),
            "cohort_summary": _plain(result.cohort_summary),
            "configuration": _plain(result.config.record),
            "source_queries": _plain(result.source_queries),
            "histogram_recipes": _plain(result.histogram_recipes),
            "outputs": list(outputs),
            "scientific_claim": (
                "exploratory_descriptive_paired_and_recording_histogram_statistics"
            ),
            "experimental_unit": "recording_id",
            "source_subject_id_use": "prohibited",
            "acquisition_batch_adjustment": "not_performed_identity_unavailable",
            "selector_eligible": False,
            "production_authority": False,
            "selector_activation": False,
            "registry_update": False,
            "source_export_mutation": False,
        }
        manifest = {**body, "record_sha256": canonical_json_sha256(body)}
        write_json_atomic(
            temporary / "manifest.json",
            manifest,
            overwrite=False,
        )
        _validate_statistics_directory(temporary, manifest)
        os.replace(temporary, target)
        temporary = None
        return MappingProxyType(
            {**manifest, "manifest_path": str(target / "manifest.json")}
        )
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)


__all__ = [
    "METHOD_ID",
    "SANDBOX_STATUS",
    "SCHEMA_ID",
    "SCHEMA_VERSION",
    "ValidatedBehaviorGroupStatisticsConfig",
    "ValidatedBehaviorGroupStatisticsError",
    "ValidatedBehaviorGroupStatisticsResult",
    "compute_validated_behavior_group_statistics",
    "read_validated_behavior_group_statistics_sandbox",
    "write_validated_behavior_group_statistics_sandbox",
]
