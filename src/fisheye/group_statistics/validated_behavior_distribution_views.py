"""Shared, receipt-bound view payloads for validated-behavior distributions.

Static and interactive renderers consume these payloads instead of querying
Parquet independently.  The distribution generation remains the authority for
all histogram values; optional traces reopen only the exact parent export named
by that generation and use display-only bounded decimation.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from fisheye.analytics_exports.validated_behavior_dataset import (
    ValidatedBehaviorExportDataset,
)
from fisheye.group_statistics.validated_behavior_appearance import (
    behavior_role_styles,
    validate_chaser_appearance_dimension,
)
from fisheye.shared.json_safety import json_attr_safe
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256

from .validated_behavior_distribution_specs import SCOPE_LABELS, SCOPE_ORDER
from .validated_behavior_distributions import read_validated_behavior_distributions

SCHEMA_ID = "palette.analytics.validated_behavior.distributions.view_payload"
SCHEMA_VERSION = 1
METHOD_ID = "shared_exact_distribution_view_payload_v1"
TRACE_SCHEMA_ID = "palette.analytics.validated_behavior.distributions.trace_payload"
TRACE_SCHEMA_VERSION = 1
TRACE_METHOD_ID = "exact_parent_motion_display_trace_v1"

DEFAULT_COHORT_STATISTIC = "mean_recording_fraction"
COHORT_STATISTIC_LABELS: Mapping[str, str] = MappingProxyType(
    {
        "mean_recording_fraction": "Mean recording fraction",
        "median_recording_fraction": "Median recording fraction",
        "pooled_fraction": "Pooled observations (diagnostic)",
    }
)
FULL_EVIDENCE_RANGE = "full_evidence"
CENTRAL_99_RANGE = "central_99"
DEFAULT_DISPLAY_RANGE = CENTRAL_99_RANGE
DISPLAY_RANGE_LABELS: Mapping[str, str] = MappingProxyType(
    {
        CENTRAL_99_RANGE: "Central 99% (tails retained)",
        FULL_EVIDENCE_RANGE: "Full evidence range",
    }
)
DISPLAY_RANGE_REFERENCE_STATISTIC = DEFAULT_COHORT_STATISTIC
CENTRAL_DISPLAY_FRACTION = 0.99
DISPLAY_RANGE_REASONS = frozenset(
    {
        "complete_sealed_histogram_axis",
        "whole_bin_minimum_99_percent_each_equal_recording_series",
        "central_target_requires_complete_axis",
        "complete_log_axis_already_exposes_tail",
        "no_finite_equal_recording_series",
    }
)
SCOPE_COLORS: Mapping[str, str] = MappingProxyType(
    {
        "whole_session": "#4C78A8",
        "chaser_pre": "#72B7B2",
        "chaser_training": "#E45756",
        "chaser_post": "#54A24B",
    }
)
PROVIDER_COLORS: Mapping[str, str] = MappingProxyType(
    {"keypoint": "#4C78A8", "detection": "#F58518"}
)
PROVIDER_LINE_STYLES: Mapping[str, str] = MappingProxyType(
    {"keypoint": "solid", "detection": "dashed"}
)
LEGACY_ROLE_COLORS: Mapping[str, str] = MappingProxyType(
    {"aggressive": "#D1495B", "inert": "#4C78A8"}
)


class ValidatedBehaviorDistributionViewError(ValueError):
    """Raised when an exact distribution cannot form a safe view."""


def _fail(message: str) -> None:
    raise ValidatedBehaviorDistributionViewError(message)


def _digest(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{label} is not one lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class ValidatedBehaviorDistributionViewSource:
    """Strict handle over one exact immutable distribution generation."""

    root: Path
    manifest: Mapping[str, Any]
    output_paths: Mapping[str, Path]

    @classmethod
    def open(cls, root: str | Path) -> "ValidatedBehaviorDistributionViewSource":
        resolved = Path(root).expanduser().resolve()
        validated = read_validated_behavior_distributions(resolved)
        manifest = {
            key: value for key, value in validated.items() if key != "manifest_path"
        }
        output_records = manifest.get("outputs")
        if not isinstance(output_records, list):  # strict reader proves this
            _fail("Distribution manifest lacks its exact outputs")
        paths: dict[str, Path] = {}
        for record in output_records:
            if not isinstance(record, Mapping):
                _fail("Distribution output record is malformed")
            name = str(record.get("path"))
            path = (resolved / name).resolve()
            if path.parent != resolved or path.name != name:
                _fail("Distribution view output escapes its exact generation")
            paths[name] = path
        return cls(
            root=resolved,
            manifest=MappingProxyType(manifest),
            output_paths=MappingProxyType(paths),
        )

    @property
    def cache_identity(self) -> str:
        return str(self.manifest["record_sha256"])

    @property
    def distribution_run_id(self) -> str:
        return str(self.manifest["distribution_run_id"])

    def scan(self, name: str) -> Any:
        import polars as pl

        try:
            path = self.output_paths[name]
        except KeyError as exc:
            raise KeyError(f"Unknown distribution output: {name}") from exc
        return pl.scan_parquet(path)


def available_distribution_metrics(
    source: ValidatedBehaviorDistributionViewSource,
) -> tuple[Mapping[str, object], ...]:
    configuration = source.manifest.get("configuration")
    if not isinstance(configuration, Mapping):
        _fail("Distribution manifest lacks its configuration")
    raw = configuration.get("metric_specs")
    if not isinstance(raw, list) or not raw:
        _fail("Distribution configuration has no metric specifications")
    records = tuple(record for record in raw if isinstance(record, Mapping))
    if len(records) != len(raw):
        _fail("Distribution metric specification is malformed")
    ids = [str(record.get("metric_id")) for record in records]
    if len(ids) != len(set(ids)):
        _fail("Distribution metric IDs are duplicated")
    return records


def _metric(
    source: ValidatedBehaviorDistributionViewSource, metric_id: str
) -> Mapping[str, object]:
    matches = [
        record
        for record in available_distribution_metrics(source)
        if record.get("metric_id") == metric_id
    ]
    if len(matches) != 1:
        raise KeyError(f"Unknown distribution metric: {metric_id}")
    return matches[0]


def _recipe(
    source: ValidatedBehaviorDistributionViewSource, metric_id: str
) -> Mapping[str, object]:
    matches = [
        record
        for record in source.manifest.get("histogram_recipes", [])
        if isinstance(record, Mapping) and record.get("metric_id") == metric_id
    ]
    if len(matches) != 1:
        _fail(f"Distribution metric {metric_id!r} lacks one exact histogram recipe")
    return matches[0]


def _decode_group(row: Mapping[str, Any]) -> dict[str, Any]:
    try:
        group = json.loads(str(row["group_key_json"]))
    except (KeyError, json.JSONDecodeError) as exc:
        raise ValidatedBehaviorDistributionViewError(
            "Distribution row has an invalid group key"
        ) from exc
    if not isinstance(group, dict):
        _fail("Distribution group key must be one JSON object")
    if canonical_json_sha256(group) != row.get("group_key_sha256"):
        _fail("Distribution group-key digest is stale")
    return {str(key): json_attr_safe(value) for key, value in group.items()}


def _normalized_rows(frame: Any) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for raw in frame.to_dicts():
        group = _decode_group(raw)
        collisions = set(group) & set(raw)
        if collisions:
            _fail(f"Decoded group fields collide with source columns: {collisions}")
        row = {key: json_attr_safe(value) for key, value in raw.items()}
        row.update(group)
        rows.append(row)
    return tuple(rows)


def build_distribution_view_payload(
    source: ValidatedBehaviorDistributionViewSource,
    metric_id: str,
    weighting_id: str,
) -> Mapping[str, object]:
    """Build one small exact-metric payload shared by both figure backends."""

    import polars as pl

    metric = _metric(source, metric_id)
    allowed_weightings = tuple(str(value) for value in metric["weighting_ids"])
    if weighting_id not in allowed_weightings:
        raise ValueError(
            f"{metric_id}: weighting {weighting_id!r} is unavailable; "
            f"choose one of {allowed_weightings}"
        )
    recipe = _recipe(source, metric_id)
    expected_spec_sha256 = canonical_json_sha256(metric)
    if recipe.get("metric_spec_sha256") != expected_spec_sha256:
        _fail("Distribution recipe refers to a stale metric specification")

    predicate = (pl.col("metric_id") == metric_id) & (
        pl.col("weighting_id") == weighting_id
    )
    cohort_frame = (
        source.scan("cohort_distribution_bins.parquet")
        .filter(predicate)
        .sort("scope_id", "group_key_sha256", "bin_index")
        .collect()
    )
    support_frame = (
        source.scan("recording_distribution_support.parquet")
        .filter(predicate)
        .sort("recording_id", "scope_id", "group_key_sha256")
        .collect()
    )
    if cohort_frame.height == 0 or support_frame.height == 0:
        _fail("Exact distribution selection has no cohort or support rows")
    cohort_rows = _normalized_rows(cohort_frame)
    support_rows = _normalized_rows(support_frame)

    source_export = source.manifest.get("source_export")
    if not isinstance(source_export, Mapping):
        _fail("Distribution manifest lacks its exact parent export")
    source_export_digest = _digest(
        source_export.get("export_manifest_record_sha256"),
        label="source export digest",
    )
    appearance = validate_chaser_appearance_dimension(
        source.manifest.get("chaser_appearance_dimension"),
        expected_export_manifest_sha256=source_export_digest,
    )
    role_styles = behavior_role_styles(
        appearance, legacy_display_colors=LEGACY_ROLE_COLORS
    )
    body: dict[str, object] = {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "metric": json_attr_safe(metric),
        "histogram_recipe": json_attr_safe(recipe),
        "weighting_id": weighting_id,
        "cohort_statistic_default": DEFAULT_COHORT_STATISTIC,
        "cohort_statistic_labels": dict(COHORT_STATISTIC_LABELS),
        "scope_order": list(SCOPE_ORDER),
        "scope_labels": dict(SCOPE_LABELS),
        "scope_colors": dict(SCOPE_COLORS),
        "provider_colors": dict(PROVIDER_COLORS),
        "provider_line_styles": dict(PROVIDER_LINE_STYLES),
        "behavior_role_styles": json_attr_safe(role_styles),
        "chaser_appearance_dimension": json_attr_safe(appearance),
        "cohort_rows": list(cohort_rows),
        "recording_support_rows": list(support_rows),
        "source_distribution": {
            "path": str(source.root),
            "distribution_run_id": source.distribution_run_id,
            "distribution_manifest_sha256": source.cache_identity,
            "source_export_run_id": source_export["export_run_id"],
            "source_export_manifest_sha256": source_export_digest,
            "experimental_unit": source.manifest["experimental_unit"],
            "cohort_weighting": source.manifest["cohort_weighting"],
            "pooled_observation_statistic": source.manifest[
                "pooled_observation_statistic"
            ],
        },
    }
    result = MappingProxyType({**body, "payload_sha256": canonical_json_sha256(body)})
    validate_distribution_view_payload(result)
    return result


def validate_distribution_view_payload(payload: Mapping[str, object]) -> None:
    """Fail closed when a shared distribution payload is stale or ambiguous."""

    if (
        payload.get("schema_id") != SCHEMA_ID
        or payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("method_id") != METHOD_ID
    ):
        _fail("Distribution view payload schema or method is unsupported")
    body = {key: value for key, value in payload.items() if key != "payload_sha256"}
    if payload.get("payload_sha256") != canonical_json_sha256(body):
        _fail("Distribution view payload digest is stale")
    metric = payload.get("metric")
    recipe = payload.get("histogram_recipe")
    if not isinstance(metric, Mapping) or not isinstance(recipe, Mapping):
        _fail("Distribution view lacks its metric or histogram recipe")
    if recipe.get("metric_id") != metric.get("metric_id"):
        _fail("Distribution view metric and recipe identities differ")
    if recipe.get("metric_spec_sha256") != canonical_json_sha256(metric):
        _fail("Distribution view metric-spec digest is stale")
    recipe_body = {
        key: value for key, value in recipe.items() if key != "histogram_recipe_sha256"
    }
    if recipe.get("histogram_recipe_sha256") != canonical_json_sha256(recipe_body):
        _fail("Distribution view histogram-recipe digest is stale")
    weighting_id = payload.get("weighting_id")
    if weighting_id not in metric.get("weighting_ids", []):
        _fail("Distribution view weighting is unsupported for this metric")
    if payload.get("cohort_statistic_default") != DEFAULT_COHORT_STATISTIC:
        _fail("Distribution view does not default to equal recording weight")
    if payload.get("cohort_statistic_labels") != dict(COHORT_STATISTIC_LABELS):
        _fail("Distribution view cohort-statistic labels are stale")
    if payload.get("scope_order") != list(SCOPE_ORDER):
        _fail("Distribution view scope order is stale")
    source = payload.get("source_distribution")
    if not isinstance(source, Mapping):
        _fail("Distribution view lacks source identity")
    for label in (
        "distribution_manifest_sha256",
        "source_export_manifest_sha256",
    ):
        _digest(source.get(label), label=label)
    appearance = validate_chaser_appearance_dimension(
        payload.get("chaser_appearance_dimension"),
        expected_export_manifest_sha256=str(source["source_export_manifest_sha256"]),
    )
    expected_styles = json_attr_safe(
        behavior_role_styles(appearance, legacy_display_colors=LEGACY_ROLE_COLORS)
    )
    if payload.get("behavior_role_styles") != expected_styles:
        _fail("Distribution view behavior-role styles are stale")

    cohort_rows = payload.get("cohort_rows")
    support_rows = payload.get("recording_support_rows")
    if not isinstance(cohort_rows, list) or not cohort_rows:
        _fail("Distribution view contains no cohort bins")
    if not isinstance(support_rows, list) or not support_rows:
        _fail("Distribution view contains no recording support")
    series: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for row in cohort_rows:
        if not isinstance(row, Mapping):
            _fail("Distribution cohort row is malformed")
        if (
            row.get("metric_id") != metric.get("metric_id")
            or row.get("weighting_id") != weighting_id
            or row.get("histogram_recipe_sha256")
            != recipe.get("histogram_recipe_sha256")
        ):
            _fail("Distribution cohort row escapes the exact view selection")
        _decode_group(row)
        for field in COHORT_STATISTIC_LABELS:
            value = row.get(field)
            if value is not None and (
                not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0
            ):
                _fail(f"Distribution cohort fraction is invalid: {field}")
        key = (str(row["scope_id"]), str(row["group_key_sha256"]))
        series.setdefault(key, []).append(row)
    expected_count = int(recipe["bin_count"])
    for (scope_id, _group), rows in series.items():
        if scope_id not in SCOPE_ORDER:
            _fail("Distribution cohort row has an unknown scope")
        ordered = sorted(rows, key=lambda row: int(row["bin_index"]))
        if [int(row["bin_index"]) for row in ordered] != list(range(expected_count)):
            _fail("Distribution cohort bin axis is not complete and gapless")
        if not all(
            math.isclose(
                float(left["bin_right"]),
                float(right["bin_left"]),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            for left, right in zip(ordered[:-1], ordered[1:], strict=True)
        ):
            _fail("Distribution cohort bin edges are discontinuous")
    for row in support_rows:
        if not isinstance(row, Mapping):
            _fail("Distribution support row is malformed")
        if (
            row.get("metric_id") != metric.get("metric_id")
            or row.get("weighting_id") != weighting_id
        ):
            _fail("Distribution support row escapes the exact view selection")
        _decode_group(row)


def distribution_dimension_options(
    payload: Mapping[str, object], dimension: str
) -> tuple[str, ...]:
    validate_distribution_view_payload(payload)
    values = {
        str(row[dimension])
        for row in payload["cohort_rows"]
        if isinstance(row, Mapping) and row.get(dimension) is not None
    }
    return tuple(sorted(values))


def resolve_distribution_display_range(
    payload: Mapping[str, object],
    *,
    display_range_id: str = DEFAULT_DISPLAY_RANGE,
    provider_role: str | None = None,
    behavior_role: str | None = None,
) -> Mapping[str, object]:
    """Resolve a transparent display-only x range from complete sealed bins.

    ``central_99`` retains whole bins covering at least 99% of the
    equal-recording mean in every finite scope/group series currently shown.
    It never removes rows from the payload and it never changes persisted
    counts. Logarithmic evidence axes already expose their tails and therefore
    remain on their complete range.
    """

    validate_distribution_view_payload(payload)
    if display_range_id not in DISPLAY_RANGE_LABELS:
        raise ValueError(
            f"Unknown display range {display_range_id!r}; choose one of "
            f"{tuple(DISPLAY_RANGE_LABELS)}"
        )
    recipe = payload.get("histogram_recipe")
    if not isinstance(recipe, Mapping):  # validated above
        _fail("Distribution view lacks its histogram recipe")
    evidence_lower = float(recipe["resolved_lower_bound"])
    evidence_upper = float(recipe["resolved_upper_bound"])
    selected = [
        row
        for row in payload["cohort_rows"]
        if isinstance(row, Mapping)
        and (provider_role is None or row.get("provider_role") == provider_role)
        and (behavior_role is None or row.get("behavior_role") == behavior_role)
    ]
    if not selected:
        raise ValueError("No distribution series matches the selected dimensions")
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for row in selected:
        key = (str(row["scope_id"]), str(row["group_key_sha256"]))
        grouped.setdefault(key, []).append(row)

    finite_series: list[list[tuple[float, float, float]]] = []
    for rows in grouped.values():
        values: list[tuple[float, float, float]] = []
        for row in sorted(rows, key=lambda value: int(value["bin_index"])):
            fraction = row.get(DISPLAY_RANGE_REFERENCE_STATISTIC)
            if fraction is None:
                continue
            numeric = float(fraction)
            if numeric > 0.0:
                values.append(
                    (float(row["bin_left"]), float(row["bin_right"]), numeric)
                )
        if values and sum(value[2] for value in values) > 0.0:
            finite_series.append(values)

    applied = (
        display_range_id == CENTRAL_99_RANGE
        and recipe.get("axis_scale") != "log10"
        and bool(finite_series)
    )
    display_lower = evidence_lower
    display_upper = evidence_upper
    reason = "complete_sealed_histogram_axis"
    if applied:
        signed_axis = evidence_lower < 0.0 < evidence_upper
        series_limits: list[float] = []
        for series in finite_series:
            total = sum(value[2] for value in series)
            target = CENTRAL_DISPLAY_FRACTION * total
            cumulative = 0.0
            if signed_axis:
                ordered = sorted(
                    series,
                    key=lambda value: max(abs(value[0]), abs(value[1])),
                )
                for left, right, fraction in ordered:
                    cumulative += fraction
                    if cumulative + 1e-15 >= target:
                        series_limits.append(max(abs(left), abs(right)))
                        break
            else:
                for _left, right, fraction in series:
                    cumulative += fraction
                    if cumulative + 1e-15 >= target:
                        series_limits.append(right)
                        break
        if series_limits:
            if signed_axis:
                extent = max(series_limits)
                display_lower = max(evidence_lower, -extent)
                display_upper = min(evidence_upper, extent)
            else:
                display_upper = min(evidence_upper, max(series_limits))
            applied = not (
                math.isclose(display_lower, evidence_lower, abs_tol=1e-12)
                and math.isclose(display_upper, evidence_upper, abs_tol=1e-12)
            )
            reason = (
                "whole_bin_minimum_99_percent_each_equal_recording_series"
                if applied
                else "central_target_requires_complete_axis"
            )
    elif display_range_id == CENTRAL_99_RANGE and recipe.get("axis_scale") == "log10":
        reason = "complete_log_axis_already_exposes_tail"
    elif display_range_id == CENTRAL_99_RANGE:
        reason = "no_finite_equal_recording_series"

    retained: list[float] = []
    for series in finite_series:
        total = sum(value[2] for value in series)
        inside = sum(
            fraction
            for left, right, fraction in series
            if left >= display_lower - 1e-12 and right <= display_upper + 1e-12
        )
        retained.append(min(1.0, max(0.0, inside / total)))
    minimum_retained = min(retained) if retained else 1.0
    body: dict[str, object] = {
        "requested_display_range_id": display_range_id,
        "effective_display_range_id": (
            CENTRAL_99_RANGE if applied else FULL_EVIDENCE_RANGE
        ),
        "display_only": True,
        "range_reference_statistic": DISPLAY_RANGE_REFERENCE_STATISTIC,
        "central_fraction_target": CENTRAL_DISPLAY_FRACTION,
        "evidence_lower_bound": evidence_lower,
        "evidence_upper_bound": evidence_upper,
        "display_lower_bound": display_lower,
        "display_upper_bound": display_upper,
        "finite_series_count": len(finite_series),
        "minimum_series_fraction_retained": minimum_retained,
        "maximum_series_fraction_omitted": max(0.0, 1.0 - minimum_retained),
        "reason": reason,
    }
    result = MappingProxyType(
        {**body, "display_range_sha256": canonical_json_sha256(body)}
    )
    validate_distribution_display_range(result)
    return result


def validate_distribution_display_range(record: Mapping[str, object]) -> None:
    """Reject stale or semantically impossible display-only range records."""

    body = {
        key: value for key, value in record.items() if key != "display_range_sha256"
    }
    if record.get("display_range_sha256") != canonical_json_sha256(body):
        _fail("Distribution display-range digest is stale")
    requested = record.get("requested_display_range_id")
    effective = record.get("effective_display_range_id")
    if requested not in DISPLAY_RANGE_LABELS or effective not in DISPLAY_RANGE_LABELS:
        _fail("Distribution display-range identity is unsupported")
    if record.get("display_only") is not True:
        _fail("Distribution display range is not explicitly display-only")
    if record.get(
        "range_reference_statistic"
    ) != DISPLAY_RANGE_REFERENCE_STATISTIC or not math.isclose(
        float(record.get("central_fraction_target", math.nan)),
        CENTRAL_DISPLAY_FRACTION,
        abs_tol=1e-12,
    ):
        _fail("Distribution display-range reference contract is stale")
    numeric = {
        name: float(record.get(name, math.nan))
        for name in (
            "evidence_lower_bound",
            "evidence_upper_bound",
            "display_lower_bound",
            "display_upper_bound",
            "minimum_series_fraction_retained",
            "maximum_series_fraction_omitted",
        )
    }
    if not all(math.isfinite(value) for value in numeric.values()):
        _fail("Distribution display range contains a non-finite value")
    if not (
        numeric["evidence_lower_bound"] < numeric["evidence_upper_bound"]
        and numeric["evidence_lower_bound"]
        <= numeric["display_lower_bound"]
        < numeric["display_upper_bound"]
        <= numeric["evidence_upper_bound"]
    ):
        _fail("Distribution display bounds escape the sealed evidence axis")
    retained = numeric["minimum_series_fraction_retained"]
    omitted = numeric["maximum_series_fraction_omitted"]
    if not (
        0.0 <= retained <= 1.0
        and 0.0 <= omitted <= 1.0
        and math.isclose(retained + omitted, 1.0, abs_tol=1e-12)
    ):
        _fail("Distribution retained and omitted display fractions disagree")
    count = record.get("finite_series_count")
    if type(count) is not int or count < 0:
        _fail("Distribution display range has an invalid finite-series count")
    if record.get("reason") not in DISPLAY_RANGE_REASONS:
        _fail("Distribution display-range reason is unsupported")
    full_bounds = math.isclose(
        numeric["display_lower_bound"],
        numeric["evidence_lower_bound"],
        abs_tol=1e-12,
    ) and math.isclose(
        numeric["display_upper_bound"],
        numeric["evidence_upper_bound"],
        abs_tol=1e-12,
    )
    if effective == CENTRAL_99_RANGE:
        if (
            requested != CENTRAL_99_RANGE
            or retained + 1e-12 < CENTRAL_DISPLAY_FRACTION
            or full_bounds
            or record.get("reason")
            != "whole_bin_minimum_99_percent_each_equal_recording_series"
        ):
            _fail("Distribution central display-range claim is inconsistent")
    elif not (
        full_bounds
        and math.isclose(retained, 1.0, abs_tol=1e-12)
        and math.isclose(omitted, 0.0, abs_tol=1e-12)
    ):
        _fail("Distribution full display-range claim is inconsistent")


def distribution_recording_ids(
    source: ValidatedBehaviorDistributionViewSource,
) -> tuple[str, ...]:
    frame = (
        source.scan("recording_distribution_support.parquet")
        .select("recording_id")
        .unique()
        .sort("recording_id")
        .collect()
    )
    return tuple(str(value) for value in frame.get_column("recording_id").to_list())


def _parent_export(
    source: ValidatedBehaviorDistributionViewSource,
) -> ValidatedBehaviorExportDataset:
    record = source.manifest.get("source_export")
    if not isinstance(record, Mapping):
        _fail("Distribution source lacks the exact parent export")
    dataset = ValidatedBehaviorExportDataset.open(
        str(record["path"]),
        str(record["export_run_id"]),
        validate=True,
        full_part_hashes=False,
    )
    if dataset.cache_identity != record.get("export_manifest_record_sha256"):
        _fail("Exact parent export manifest differs from the distribution binding")
    return dataset


def _even_display_indices(row_count: int, max_points: int) -> np.ndarray:
    if row_count <= max_points:
        return np.arange(row_count, dtype=np.int64)
    return np.unique(np.linspace(0, row_count - 1, num=max_points, dtype=np.int64))


def build_motion_trace_payload(
    source: ValidatedBehaviorDistributionViewSource,
    *,
    metric_id: str,
    recording_id: str,
    coordinate_id: str,
    provider_role: str | None = None,
    max_display_points: int = 5000,
) -> Mapping[str, object]:
    """Read one exact parent motion series with bounded display-only decimation."""

    import polars as pl

    metric = _metric(source, metric_id)
    if metric.get("source_surface") != "provider_motion_samples":
        raise ValueError("Trace view currently supports provider-motion metrics only")
    if coordinate_id not in {"frame", "time"}:
        raise ValueError("coordinate_id must be 'frame' or 'time'")
    if type(max_display_points) is not int or not 100 <= max_display_points <= 100_000:
        raise ValueError(
            "max_display_points must be an integer from 100 through 100000"
        )
    dataset = _parent_export(source)
    if recording_id not in set(distribution_recording_ids(source)):
        raise KeyError(f"Unknown contributing recording: {recording_id}")
    table = dataset.table("provider_motion_samples")
    value_column = str(metric["value_column"])
    columns = (
        "recording_id",
        "provider_role",
        "position_provider_id",
        "position_provider_digest",
        "acquisition_frame_id",
        "time_s",
        "linear_sample_valid",
        "angular_sample_valid",
        "transition_valid",
        "delta_s",
        value_column,
    )
    predicate = pl.col("recording_id") == recording_id
    if provider_role is not None:
        predicate &= pl.col("provider_role") == provider_role
    frame = (
        table.scan(columns=columns, predicate=predicate)
        .sort("provider_role", "acquisition_frame_id")
        .collect()
    )
    if frame.height == 0:
        _fail("Exact parent motion trace selection has no rows")
    providers = tuple(sorted(str(value) for value in frame["provider_role"].unique()))
    if provider_role is None:
        if len(providers) != 1:
            raise ValueError(
                "provider_role is required when one recording has multiple providers"
            )
        provider_role = providers[0]
    elif providers != (provider_role,):
        _fail("Exact parent motion trace provider selection is ambiguous")
    frame_ids = frame["acquisition_frame_id"].to_numpy().astype(np.int64)
    if np.any(np.diff(frame_ids) <= 0):
        _fail("Exact parent motion trace frame axis is not strictly increasing")
    values = frame[value_column].to_numpy().astype(np.float64)
    validity_name = (
        "angular_sample_valid"
        if str(metric["validity_policy_id"]).startswith("angular_")
        else "linear_sample_valid"
    )
    valid = (
        frame[validity_name].to_numpy().astype(bool)
        & frame["transition_valid"].to_numpy().astype(bool)
        & np.isfinite(values)
    )
    indices = _even_display_indices(frame.height, max_display_points)
    coordinate_column = "acquisition_frame_id" if coordinate_id == "frame" else "time_s"
    coordinate = frame[coordinate_column].to_numpy()[indices]
    display_values = values[indices]
    display_valid = valid[indices]
    points = [
        {
            "source_row_index": int(index),
            "acquisition_frame_id": int(frame_ids[index]),
            "time_s": float(frame["time_s"][int(index)]),
            "coordinate": (
                int(coordinate[position])
                if coordinate_id == "frame"
                else float(coordinate[position])
            ),
            "value": (
                float(display_values[position]) if display_valid[position] else None
            ),
            "valid": bool(display_valid[position]),
        }
        for position, index in enumerate(indices)
    ]
    provider_ids = frame["position_provider_id"].unique().to_list()
    provider_digests = frame["position_provider_digest"].unique().to_list()
    if len(provider_ids) != 1 or len(provider_digests) != 1:
        _fail("Exact parent motion trace does not bind one provider identity")
    query = table.query_identity(
        columns=columns,
        predicate_description=(
            f"recording_id == {recording_id!r} and provider_role == {provider_role!r}; "
            "ordered by acquisition_frame_id"
        ),
    )
    body: dict[str, object] = {
        "schema_id": TRACE_SCHEMA_ID,
        "schema_version": TRACE_SCHEMA_VERSION,
        "method_id": TRACE_METHOD_ID,
        "metric": json_attr_safe(metric),
        "recording_id": recording_id,
        "provider_role": provider_role,
        "position_provider_id": str(provider_ids[0]),
        "position_provider_digest": str(provider_digests[0]),
        "coordinate_id": coordinate_id,
        "coordinate_column": coordinate_column,
        "coordinate_choice_semantics": "display_only_same_exact_rows",
        "source_row_count": frame.height,
        "source_valid_count": int(np.count_nonzero(valid)),
        "display_point_count": len(points),
        "max_display_points": max_display_points,
        "decimation_id": (
            "identity_all_rows"
            if frame.height <= max_display_points
            else "deterministic_even_index_endpoint_preserving_display_only_v1"
        ),
        "points": points,
        "source_query_identity": query,
        "source_distribution_manifest_sha256": source.cache_identity,
        "source_export_manifest_sha256": dataset.cache_identity,
    }
    result = MappingProxyType({**body, "payload_sha256": canonical_json_sha256(body)})
    validate_motion_trace_payload(result)
    return result


def validate_motion_trace_payload(payload: Mapping[str, object]) -> None:
    if (
        payload.get("schema_id") != TRACE_SCHEMA_ID
        or payload.get("schema_version") != TRACE_SCHEMA_VERSION
        or payload.get("method_id") != TRACE_METHOD_ID
    ):
        _fail("Motion trace payload schema or method is unsupported")
    body = {key: value for key, value in payload.items() if key != "payload_sha256"}
    if payload.get("payload_sha256") != canonical_json_sha256(body):
        _fail("Motion trace payload digest is stale")
    if payload.get("coordinate_id") not in {"frame", "time"}:
        _fail("Motion trace coordinate is unsupported")
    for label in (
        "position_provider_digest",
        "source_distribution_manifest_sha256",
        "source_export_manifest_sha256",
    ):
        _digest(payload.get(label), label=label)
    points = payload.get("points")
    if not isinstance(points, list) or len(points) != payload.get(
        "display_point_count"
    ):
        _fail("Motion trace point roster is incomplete")
    if int(payload.get("display_point_count", -1)) > int(
        payload.get("max_display_points", -1)
    ):
        _fail("Motion trace exceeds its display bound")
    row_indices: list[int] = []
    for point in points:
        if not isinstance(point, Mapping):
            _fail("Motion trace point is malformed")
        row_indices.append(int(point["source_row_index"]))
        if bool(point.get("valid")) != (point.get("value") is not None):
            _fail("Motion trace validity and value disagree")
    if row_indices != sorted(set(row_indices)):
        _fail("Motion trace display rows are duplicated or unordered")
    if row_indices and (
        row_indices[0] != 0 or row_indices[-1] != int(payload["source_row_count"]) - 1
    ):
        _fail("Decimated motion trace does not preserve both endpoints")


__all__ = [
    "CENTRAL_99_RANGE",
    "CENTRAL_DISPLAY_FRACTION",
    "COHORT_STATISTIC_LABELS",
    "DEFAULT_COHORT_STATISTIC",
    "DEFAULT_DISPLAY_RANGE",
    "DISPLAY_RANGE_LABELS",
    "DISPLAY_RANGE_REASONS",
    "DISPLAY_RANGE_REFERENCE_STATISTIC",
    "FULL_EVIDENCE_RANGE",
    "METHOD_ID",
    "SCHEMA_ID",
    "SCHEMA_VERSION",
    "TRACE_METHOD_ID",
    "TRACE_SCHEMA_ID",
    "TRACE_SCHEMA_VERSION",
    "ValidatedBehaviorDistributionViewError",
    "ValidatedBehaviorDistributionViewSource",
    "available_distribution_metrics",
    "build_distribution_view_payload",
    "build_motion_trace_payload",
    "distribution_dimension_options",
    "distribution_recording_ids",
    "resolve_distribution_display_range",
    "validate_distribution_view_payload",
    "validate_distribution_display_range",
    "validate_motion_trace_payload",
]
