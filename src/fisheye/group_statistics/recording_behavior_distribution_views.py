"""Renderer-neutral views over one sealed recording distribution run.

The persisted run owns scope membership, histogram binning, denominators, and
normalization.  This module only restores structural zero bins and presents the
same immutable series to static and interactive renderers.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from fisheye.analysis_workflows.recording_behavior_distribution_storage import (
    RecordingBehaviorDistributionSourceHandle,
)
from fisheye.shared.json_safety import json_attr_safe
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256

from .recording_behavior_distribution_specs import histogram_recipe_record
from .recording_behavior_distributions import canonical_grid_bounds
from .validated_behavior_distribution_specs import DistributionMetricSpec


SCHEMA_ID = "palette.analysis.recording_behavior_distribution_view"
SCHEMA_VERSION = 1
METHOD_ID = "sealed_sparse_bins_structural_zero_projection_v1"
MAX_PROJECTED_BINS = 10_000


class RecordingBehaviorDistributionViewError(ValueError):
    """A persisted recording distribution cannot form an exact view."""


def _fail(message: str) -> None:
    raise RecordingBehaviorDistributionViewError(message)


def _readonly(values: Any, *, dtype: Any) -> np.ndarray:
    result = np.asarray(values, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _decode_registry_json(
    value: object,
    *,
    expected_sha256: object,
    field: str,
) -> Mapping[str, Any]:
    if type(value) is not str:
        _fail(f"{field} is not canonical JSON text.")
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise RecordingBehaviorDistributionViewError(
            f"{field} is not valid JSON."
        ) from exc
    if not isinstance(decoded, dict):
        _fail(f"{field} must decode to one object.")
    plain = json_attr_safe(decoded)
    if canonical_json_sha256(plain) != expected_sha256:
        _fail(f"{field} digest is stale.")
    return MappingProxyType(plain)


def _metric_spec(row: Mapping[str, Any]) -> DistributionMetricSpec:
    try:
        spec = DistributionMetricSpec(
            metric_id=row["metric_id"],
            metric_family=row["metric_family"],
            source_surface=row["source_surface"],
            value_column=row["value_column"],
            unit=row["unit"],
            bin_width=row["bin_width"],
            lower_bound=row["lower_bound"],
            upper_bound=row["upper_bound"],
            coverage_policy=row["coverage_policy"],
            weighting_ids=tuple(row["weighting_ids"]),
            group_columns=tuple(row["group_columns"]),
            validity_policy_id=row["validity_policy_id"],
            scope_binding_id=row["scope_binding_id"],
            interpretation=row["interpretation"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RecordingBehaviorDistributionViewError(
            "Metric registry row cannot reconstruct its declared specification."
        ) from exc
    if row.get("metric_spec_sha256") != spec.spec_sha256:
        _fail(f"{spec.metric_id}: metric specification digest is stale.")
    recipe = histogram_recipe_record(spec)
    if row.get("histogram_recipe_sha256") != recipe["histogram_recipe_sha256"]:
        _fail(f"{spec.metric_id}: histogram recipe digest is stale.")
    return spec


def _axis_indices(
    spec: DistributionMetricSpec, audit: Mapping[str, Any]
) -> np.ndarray:
    valid_count = int(audit.get("valid_finite_count", -1))
    minimum = audit.get("minimum_value")
    maximum = audit.get("maximum_value")
    minimum_grid = audit.get("minimum_grid_index")
    maximum_grid = audit.get("maximum_grid_index")
    if valid_count < 0:
        _fail(f"{spec.metric_id}: axis audit has an invalid valid count.")
    if valid_count == 0:
        if any(
            value is not None
            for value in (minimum, maximum, minimum_grid, maximum_grid)
        ):
            _fail(f"{spec.metric_id}: empty axis audit retains observed bounds.")
        if spec.coverage_policy != "fixed_closed_terminal":
            return _readonly([], dtype=np.int64)
    elif any(
        value is None for value in (minimum, maximum, minimum_grid, maximum_grid)
    ):
        _fail(f"{spec.metric_id}: finite axis audit lacks observed bounds.")

    width = float(spec.bin_width)
    if spec.coverage_policy == "fixed_closed_terminal":
        assert spec.upper_bound is not None
        count = int(round((float(spec.upper_bound) - float(spec.lower_bound)) / width))
        indices = np.arange(count, dtype=np.int64)
    elif spec.coverage_policy == "zero_anchored_cover_valid_max":
        if int(minimum_grid) < 0 or float(minimum) < 0:
            _fail(f"{spec.metric_id}: nonnegative axis has negative evidence.")
        indices = np.arange(0, int(maximum_grid) + 1, dtype=np.int64)
    elif spec.coverage_policy == "symmetric_cover_valid_abs_max":
        half = max(
            1,
            math.floor(max(abs(float(minimum)), abs(float(maximum))) / width) + 1,
        )
        indices = np.arange(-half, half, dtype=np.int64)
    elif spec.coverage_policy == "log10_cover_valid_positive_range":
        if float(minimum) <= 0:
            _fail(f"{spec.metric_id}: logarithmic axis has nonpositive evidence.")
        indices = np.arange(
            int(minimum_grid), int(maximum_grid) + 1, dtype=np.int64
        )
    else:  # pragma: no cover - DistributionMetricSpec closes this vocabulary
        raise AssertionError(spec.coverage_policy)
    if indices.size > MAX_PROJECTED_BINS:
        _fail(
            f"{spec.metric_id}: resolved {indices.size} bins, above the safe "
            f"view limit {MAX_PROJECTED_BINS}."
        )
    return _readonly(indices, dtype=np.int64)


@dataclass(frozen=True, slots=True)
class RecordingDistributionSeries:
    """One scope/group/source histogram with restored structural zero bins."""

    support_key_sha256: str
    scope_id: str
    scope_label: str
    scope_order: int
    group: Mapping[str, Any]
    group_key_sha256: str
    source_identity: Mapping[str, Any]
    source_identity_key_sha256: str
    weighting_id: str
    weight_unit: str
    support: Mapping[str, Any]
    grid_index: np.ndarray
    bin_left: np.ndarray
    bin_right: np.ndarray
    bin_center: np.ndarray
    bin_count: np.ndarray
    bin_weight: np.ndarray
    fraction: np.ndarray

    @property
    def label(self) -> str:
        if not self.group:
            return "All observations"
        return " · ".join(
            f"{name}={value}" for name, value in sorted(self.group.items())
        )


@dataclass(frozen=True, slots=True)
class RecordingBehaviorDistributionView:
    """An exact metric/weighting projection shared by every renderer."""

    recording_id: str
    distribution_run_id: str
    run_path: str
    manifest_sha256: str
    verification_digest: str
    result_record_sha256: str
    metric: Mapping[str, Any]
    weighting_id: str
    scopes: tuple[Mapping[str, Any], ...]
    series: tuple[RecordingDistributionSeries, ...]
    view_sha256: str


def available_recording_distribution_metrics(
    handle: RecordingBehaviorDistributionSourceHandle,
) -> tuple[Mapping[str, Any], ...]:
    """Return the validated metric registry in deterministic order."""

    if type(handle) is not RecordingBehaviorDistributionSourceHandle:
        _fail("source must be one validated recording-distribution handle.")
    result = []
    seen: set[str] = set()
    for raw in handle.tables["metric_registry"]:
        spec = _metric_spec(raw)
        if spec.metric_id in seen:
            _fail("Metric registry contains duplicate metric IDs.")
        seen.add(spec.metric_id)
        result.append(MappingProxyType(dict(raw)))
    return tuple(sorted(result, key=lambda row: str(row["metric_id"])))


def available_recording_distribution_scopes(
    handle: RecordingBehaviorDistributionSourceHandle,
) -> tuple[Mapping[str, Any], ...]:
    """Return the exact dynamic scope roster, never a hard-coded epoch list."""

    if type(handle) is not RecordingBehaviorDistributionSourceHandle:
        _fail("source must be one validated recording-distribution handle.")
    scopes = tuple(
        sorted(handle.tables["scope_registry"], key=lambda row: int(row["order"]))
    )
    if [int(row["order"]) for row in scopes] != list(range(len(scopes))):
        _fail("Scope registry order is not gapless.")
    if len({str(row["scope_id"]) for row in scopes}) != len(scopes):
        _fail("Scope registry contains duplicate scope IDs.")
    return scopes


def build_recording_behavior_distribution_view(
    handle: RecordingBehaviorDistributionSourceHandle,
    *,
    metric_id: str,
    weighting_id: str,
    scope_ids: Sequence[str] = (),
) -> RecordingBehaviorDistributionView:
    """Restore one exact persisted metric without rebinning or reassignment."""

    metrics = available_recording_distribution_metrics(handle)
    matches = [row for row in metrics if row["metric_id"] == metric_id]
    if len(matches) != 1:
        raise KeyError(f"Unknown recording distribution metric: {metric_id!r}")
    metric = matches[0]
    spec = _metric_spec(metric)
    if weighting_id not in tuple(spec.weighting_ids):
        raise ValueError(
            f"{metric_id}: weighting {weighting_id!r} is unavailable; "
            f"choose one of {spec.weighting_ids}."
        )
    all_scopes = available_recording_distribution_scopes(handle)
    scope_by_id = {str(row["scope_id"]): row for row in all_scopes}
    requested = tuple(dict.fromkeys(str(value) for value in scope_ids))
    unknown = sorted(set(requested) - set(scope_by_id))
    if unknown:
        raise KeyError(f"Unknown recording distribution scopes: {unknown!r}")
    selected_scopes = (
        all_scopes
        if not requested
        else tuple(row for row in all_scopes if row["scope_id"] in requested)
    )

    audits = [
        row for row in handle.tables["axis_audits"] if row["metric_id"] == metric_id
    ]
    if len(audits) != 1:
        _fail(f"{metric_id}: expected exactly one axis audit.")
    audit = audits[0]
    if (
        audit.get("metric_spec_sha256") != spec.spec_sha256
        or audit.get("histogram_recipe_sha256")
        != metric.get("histogram_recipe_sha256")
    ):
        _fail(f"{metric_id}: axis audit differs from the metric registry.")
    grid = _axis_indices(spec, audit)
    bounds = [canonical_grid_bounds(int(index), spec) for index in grid]
    left = _readonly([item[0] for item in bounds], dtype=np.float64)
    right = _readonly([item[1] for item in bounds], dtype=np.float64)
    center = _readonly((left + right) / 2.0, dtype=np.float64)
    grid_to_offset = {int(value): index for index, value in enumerate(grid)}

    groups = {
        str(row["group_key_sha256"]): _decode_registry_json(
            row["group_key_json"],
            expected_sha256=row["group_key_sha256"],
            field="group registry row",
        )
        for row in handle.tables["group_registry"]
    }
    sources = {
        str(row["source_identity_key_sha256"]): _decode_registry_json(
            row["source_identity_key_json"],
            expected_sha256=row["source_identity_key_sha256"],
            field="source-identity registry row",
        )
        for row in handle.tables["source_identity_registry"]
    }
    if len(groups) != len(handle.tables["group_registry"]) or len(sources) != len(
        handle.tables["source_identity_registry"]
    ):
        _fail("Distribution identity registries contain duplicate digests.")

    sparse_by_support: dict[str, list[Mapping[str, Any]]] = {}
    for row in handle.tables["sparse_bins"]:
        if row["metric_id"] == metric_id and row["weighting_id"] == weighting_id:
            sparse_by_support.setdefault(str(row["support_key_sha256"]), []).append(row)

    series = []
    selected_ids = {str(row["scope_id"]) for row in selected_scopes}
    for support in handle.tables["support"]:
        if (
            support["metric_id"] != metric_id
            or support["weighting_id"] != weighting_id
            or support["scope_id"] not in selected_ids
        ):
            continue
        scope = scope_by_id[str(support["scope_id"])]
        group_digest = str(support["group_key_sha256"])
        source_digest = str(support["source_identity_key_sha256"])
        try:
            group = groups[group_digest]
            source = sources[source_digest]
        except KeyError as exc:
            raise RecordingBehaviorDistributionViewError(
                "Support row refers to an absent identity registry."
            ) from exc
        counts = np.zeros(grid.shape, dtype=np.int64)
        weights = np.zeros(grid.shape, dtype=np.float64)
        for row in sparse_by_support.get(str(support["support_key_sha256"]), []):
            index = int(row["grid_index"])
            if index not in grid_to_offset:
                _fail(f"{metric_id}: sparse bin lies outside the resolved view axis.")
            offset = grid_to_offset[index]
            expected_left, expected_right = canonical_grid_bounds(index, spec)
            if not (
                math.isclose(float(row["bin_left"]), expected_left)
                and math.isclose(float(row["bin_right"]), expected_right)
            ):
                _fail(f"{metric_id}: sparse bin edges differ from the recipe.")
            counts[offset] = int(row["bin_count"])
            weights[offset] = float(row["bin_weight"])
        denominator = float(support["denominator_weight"])
        if int(np.sum(counts)) != int(support["valid_count"]) or not math.isclose(
            float(np.sum(weights)), denominator, rel_tol=1e-10, abs_tol=1e-10
        ):
            _fail(f"{metric_id}: restored series does not conserve support.")
        fractions = (
            np.full(grid.shape, np.nan, dtype=np.float64)
            if denominator == 0
            else weights / denominator
        )
        series.append(
            RecordingDistributionSeries(
                support_key_sha256=str(support["support_key_sha256"]),
                scope_id=str(scope["scope_id"]),
                scope_label=str(scope["scope_label"]),
                scope_order=int(scope["order"]),
                group=group,
                group_key_sha256=group_digest,
                source_identity=source,
                source_identity_key_sha256=source_digest,
                weighting_id=weighting_id,
                weight_unit=str(support["weight_unit"]),
                support=MappingProxyType(dict(support)),
                grid_index=grid,
                bin_left=left,
                bin_right=right,
                bin_center=center,
                bin_count=_readonly(counts, dtype=np.int64),
                bin_weight=_readonly(weights, dtype=np.float64),
                fraction=_readonly(fractions, dtype=np.float64),
            )
        )
    series.sort(
        key=lambda item: (
            item.scope_order,
            item.group_key_sha256,
            item.source_identity_key_sha256,
        )
    )
    if not series:
        _fail("Exact metric/weighting selection has no persisted support rows.")
    manifest_sha256 = canonical_json_sha256(dict(handle.manifest))
    view_body = {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "recording_id": handle.recording_id,
        "distribution_run_id": handle.run_name,
        "run_path": handle.run_path,
        "manifest_sha256": manifest_sha256,
        "verification_digest": handle.verification_digest,
        "result_record_sha256": handle.result_record["record_sha256"],
        "metric_spec_sha256": spec.spec_sha256,
        "histogram_recipe_sha256": metric["histogram_recipe_sha256"],
        "weighting_id": weighting_id,
        "scope_sha256": [row["scope_sha256"] for row in selected_scopes],
        "support_key_sha256": [item.support_key_sha256 for item in series],
    }
    return RecordingBehaviorDistributionView(
        recording_id=handle.recording_id,
        distribution_run_id=handle.run_name,
        run_path=handle.run_path,
        manifest_sha256=manifest_sha256,
        verification_digest=handle.verification_digest,
        result_record_sha256=str(handle.result_record["record_sha256"]),
        metric=metric,
        weighting_id=weighting_id,
        scopes=tuple(MappingProxyType(dict(row)) for row in selected_scopes),
        series=tuple(series),
        view_sha256=canonical_json_sha256(view_body),
    )


__all__ = [
    "METHOD_ID",
    "RecordingBehaviorDistributionView",
    "RecordingBehaviorDistributionViewError",
    "RecordingDistributionSeries",
    "SCHEMA_ID",
    "SCHEMA_VERSION",
    "available_recording_distribution_metrics",
    "available_recording_distribution_scopes",
    "build_recording_behavior_distribution_view",
]
