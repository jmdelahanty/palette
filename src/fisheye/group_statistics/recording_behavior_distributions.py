"""Pure recording-local reduction for composable behavior distributions.

The reducer receives already validated metric values and already resolved scope
membership.  It owns histogram arithmetic, but it never discovers a selector,
infers a protocol epoch, or reads a recording archive.  Source adapters retain
those responsibilities and bind their evidence in ``source_identity_arrays``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from fisheye.shared.json_safety import json_attr_safe, strict_json_dumps
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256

from .recording_distribution_scopes import (
    RecordingDistributionScope,
    ScopeMaskProjection,
    scope_registry_record,
    validate_scope_registry,
)
from .recording_behavior_distribution_specs import (
    histogram_recipe_record,
    recording_metric_registry_record,
)
from .validated_behavior_distribution_specs import DistributionMetricSpec


SCHEMA_ID = "palette.analysis.recording_behavior_distributions"
SCHEMA_VERSION = 1
METHOD_ID = "explicit_metric_scope_sparse_grid_histograms_v1"


class RecordingBehaviorDistributionError(ValueError):
    """A recording distribution input or reduction is unsafe."""


def _fail(message: str) -> None:
    raise RecordingBehaviorDistributionError(message)


def _text(value: object, *, field: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        _fail(f"{field} must be one nonempty stripped string.")
    return value


def _plain(value: Any) -> Any:
    return json_attr_safe(value)


def _json_identity(value: Mapping[str, Any]) -> tuple[str, str]:
    plain = _plain(dict(value))
    return strict_json_dumps(plain), canonical_json_sha256(plain)


def _readonly_1d(values: Any, *, dtype: Any, field: str) -> np.ndarray:
    result = np.asarray(values, dtype=dtype)
    if result.ndim != 1:
        raise RecordingBehaviorDistributionError(
            f"{field} must be one-dimensional."
        )
    result = result.copy()
    result.setflags(write=False)
    return result


def _readonly_arrays(
    values: Mapping[str, Any], *, row_count: int, field: str
) -> Mapping[str, np.ndarray]:
    if not isinstance(values, Mapping):
        _fail(f"{field} must be one mapping.")
    result: dict[str, np.ndarray] = {}
    for raw_name, raw_values in values.items():
        name = _text(raw_name, field=f"{field} key")
        array = np.asarray(raw_values)
        if array.shape != (row_count,):
            _fail(f"{field}.{name} does not share the metric row axis.")
        array = array.copy()
        array.setflags(write=False)
        result[name] = array
    return MappingProxyType(result)


def _validated_projection(
    value: ScopeMaskProjection,
    *,
    row_count: int,
    field: str,
) -> ScopeMaskProjection:
    if type(value) is not ScopeMaskProjection:
        _fail(f"{field} must be one ScopeMaskProjection.")
    _text(value.membership_policy_id, field=f"{field}.membership_policy_id")
    masks: dict[str, np.ndarray] = {}
    uncovered: dict[str, np.ndarray] = {}
    if set(value.masks) != set(value.uncovered):
        _fail(f"{field} mask and uncovered scope rosters differ.")
    for scope_id in value.masks:
        masks[scope_id] = _readonly_1d(
            value.masks[scope_id], dtype=bool, field=f"{field}.masks.{scope_id}"
        )
        uncovered[scope_id] = _readonly_1d(
            value.uncovered[scope_id],
            dtype=bool,
            field=f"{field}.uncovered.{scope_id}",
        )
        if masks[scope_id].shape != (row_count,):
            _fail(f"{field}.{scope_id} does not share the metric row axis.")
        if np.any(masks[scope_id] & uncovered[scope_id]):
            _fail(f"{field}.{scope_id} marks rows both selected and uncovered.")
    return ScopeMaskProjection(
        masks=MappingProxyType(masks),
        uncovered=MappingProxyType(uncovered),
        membership_policy_id=value.membership_policy_id,
    )


@dataclass(frozen=True, slots=True)
class RecordingDistributionMetricInput:
    """One validated value axis and its exact scope/source projections."""

    spec: DistributionMetricSpec
    values: np.ndarray
    valid: np.ndarray
    scope_projection: ScopeMaskProjection
    source_identity_arrays: Mapping[str, np.ndarray]
    source_identity_fallback: Mapping[str, Any]
    group_arrays: Mapping[str, np.ndarray] = field(default_factory=dict)
    time_weights_s: np.ndarray | None = None
    time_scope_projection: ScopeMaskProjection | None = None
    valid_duration_s_by_scope: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.spec) is not DistributionMetricSpec:
            _fail("spec must be one DistributionMetricSpec.")
        values = _readonly_1d(self.values, dtype=np.float64, field="values")
        valid = _readonly_1d(self.valid, dtype=bool, field="valid")
        if valid.shape != values.shape:
            _fail("values and valid do not share one row axis.")
        row_count = int(values.size)
        groups = _readonly_arrays(
            self.group_arrays, row_count=row_count, field="group_arrays"
        )
        if tuple(groups) != tuple(self.spec.group_columns):
            _fail(
                f"{self.spec.metric_id}: group arrays must exactly follow the "
                "metric specification."
            )
        identities = _readonly_arrays(
            self.source_identity_arrays,
            row_count=row_count,
            field="source_identity_arrays",
        )
        if not identities:
            _fail(f"{self.spec.metric_id}: source identity must not be empty.")
        fallback = _plain(dict(self.source_identity_fallback))
        if set(fallback) != set(identities):
            _fail(
                f"{self.spec.metric_id}: source identity fallback must exactly "
                "match the identity-array field roster."
            )
        scope_projection = _validated_projection(
            self.scope_projection, row_count=row_count, field="scope_projection"
        )
        time_projection = None
        weights = None
        if "time" in self.spec.weighting_ids:
            if self.time_weights_s is None or self.time_scope_projection is None:
                _fail(
                    f"{self.spec.metric_id}: time weighting requires elapsed-time "
                    "weights and a transition scope projection."
                )
            weights = _readonly_1d(
                self.time_weights_s, dtype=np.float64, field="time_weights_s"
            )
            if weights.shape != values.shape:
                _fail("time_weights_s does not share the metric row axis.")
            time_projection = _validated_projection(
                self.time_scope_projection,
                row_count=row_count,
                field="time_scope_projection",
            )
        elif self.time_weights_s is not None or self.time_scope_projection is not None:
            _fail(
                f"{self.spec.metric_id}: non-time metric supplied time-only inputs."
            )
        durations: dict[str, float] = {}
        for scope_id, raw in self.valid_duration_s_by_scope.items():
            value = float(raw)
            if not math.isfinite(value) or value < 0:
                _fail(f"{self.spec.metric_id}: invalid duration for {scope_id!r}.")
            durations[str(scope_id)] = value
        if "event" in self.spec.weighting_ids and not durations:
            _fail(
                f"{self.spec.metric_id}: event weighting requires valid-duration "
                "support for rate provenance."
            )
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "group_arrays", groups)
        object.__setattr__(self, "source_identity_arrays", identities)
        object.__setattr__(
            self, "source_identity_fallback", MappingProxyType(fallback)
        )
        object.__setattr__(self, "scope_projection", scope_projection)
        object.__setattr__(self, "time_weights_s", weights)
        object.__setattr__(self, "time_scope_projection", time_projection)
        object.__setattr__(
            self, "valid_duration_s_by_scope", MappingProxyType(durations)
        )


@dataclass(frozen=True, slots=True)
class RecordingBehaviorDistributionConfig:
    distribution_run_id: str
    recording_id: str
    scopes: tuple[RecordingDistributionScope, ...]
    source_record: Mapping[str, Any]

    def __post_init__(self) -> None:
        _text(self.distribution_run_id, field="distribution_run_id")
        _text(self.recording_id, field="recording_id")
        object.__setattr__(self, "scopes", validate_scope_registry(self.scopes))
        source = _plain(dict(self.source_record))
        if not source:
            _fail("source_record must not be empty.")
        object.__setattr__(self, "source_record", MappingProxyType(source))

    @property
    def record(self) -> Mapping[str, Any]:
        body = {
            "schema_id": SCHEMA_ID,
            "schema_version": SCHEMA_VERSION,
            "method_id": METHOD_ID,
            "distribution_run_id": self.distribution_run_id,
            "recording_id": self.recording_id,
            "scope_registry": dict(scope_registry_record(self.scopes)),
            "source_record": dict(self.source_record),
            "interpolation": "prohibited",
            "histogram_storage": "sparse_canonical_grid_indices",
        }
        return MappingProxyType(
            {**body, "config_sha256": canonical_json_sha256(body)}
        )


@dataclass(frozen=True, slots=True)
class RecordingBehaviorDistributionResult:
    config: RecordingBehaviorDistributionConfig
    metric_registry: tuple[Mapping[str, Any], ...]
    axis_audits: tuple[Mapping[str, Any], ...]
    support: tuple[Mapping[str, Any], ...]
    sparse_bins: tuple[Mapping[str, Any], ...]

    @property
    def record(self) -> Mapping[str, Any]:
        body = {
            "config": dict(self.config.record),
            "metric_registry": [dict(row) for row in self.metric_registry],
            "axis_audits": [dict(row) for row in self.axis_audits],
            "support": [dict(row) for row in self.support],
            "sparse_bins": [dict(row) for row in self.sparse_bins],
        }
        return MappingProxyType(
            {**body, "record_sha256": canonical_json_sha256(body)}
        )


def _validate_scope_roster(
    config: RecordingBehaviorDistributionConfig,
    item: RecordingDistributionMetricInput,
) -> None:
    expected = tuple(scope.scope_id for scope in config.scopes)
    if tuple(item.scope_projection.masks) != expected:
        _fail(f"{item.spec.metric_id}: sample scope roster or order differs.")
    duration_scopes = set(item.valid_duration_s_by_scope)
    if duration_scopes and duration_scopes != set(expected):
        _fail(f"{item.spec.metric_id}: duration scope roster differs.")
    if item.time_scope_projection is not None and tuple(
        item.time_scope_projection.masks
    ) != expected:
        _fail(f"{item.spec.metric_id}: transition scope roster or order differs.")


def _validate_metric_values(spec: DistributionMetricSpec, values: np.ndarray) -> None:
    if not values.size:
        return
    if spec.coverage_policy == "fixed_closed_terminal":
        assert spec.upper_bound is not None
        if np.any(values < spec.lower_bound) or np.any(values > spec.upper_bound):
            _fail(f"{spec.metric_id}: valid values escape the fixed axis.")
    elif spec.coverage_policy == "zero_anchored_cover_valid_max":
        if np.any(values < 0):
            _fail(f"{spec.metric_id}: valid values violate nonnegativity.")
    elif spec.coverage_policy == "log10_cover_valid_positive_range":
        if np.any(values <= 0):
            _fail(f"{spec.metric_id}: logarithmic values must be positive.")


def canonical_grid_indices(
    values: Any, spec: DistributionMetricSpec
) -> np.ndarray:
    """Return globally composable grid coordinates for one metric recipe."""

    data = np.asarray(values, dtype=np.float64).reshape(-1)
    if np.any(~np.isfinite(data)):
        _fail(f"{spec.metric_id}: histogram input contains non-finite values.")
    _validate_metric_values(spec, data)
    width = float(spec.bin_width)
    if spec.coverage_policy == "log10_cover_valid_positive_range":
        return np.floor(np.log10(data) / width).astype(np.int64)
    if spec.coverage_policy == "fixed_closed_terminal":
        assert spec.upper_bound is not None
        indices = np.floor((data - float(spec.lower_bound)) / width).astype(
            np.int64
        )
        terminal = data == float(spec.upper_bound)
        bin_count = int(
            round((float(spec.upper_bound) - float(spec.lower_bound)) / width)
        )
        indices[terminal] = bin_count - 1
        return indices
    return np.floor(data / width).astype(np.int64)


def canonical_grid_bounds(
    grid_index: int, spec: DistributionMetricSpec
) -> tuple[float, float]:
    width = float(spec.bin_width)
    if spec.coverage_policy == "log10_cover_valid_positive_range":
        return (
            10.0 ** (int(grid_index) * width),
            10.0 ** ((int(grid_index) + 1) * width),
        )
    origin = (
        float(spec.lower_bound)
        if spec.coverage_policy == "fixed_closed_terminal"
        else 0.0
    )
    left = origin + int(grid_index) * width
    return left, left + width


def _group_roster(
    arrays: Mapping[str, np.ndarray], *, row_count: int
) -> tuple[tuple[tuple[Any, ...], np.ndarray], ...]:
    if not arrays:
        return (((), np.ones(row_count, dtype=bool)),)
    names = tuple(arrays)
    columns = tuple(np.asarray(arrays[name], dtype=object) for name in names)
    if any(any(value is None for value in column) for column in columns):
        _fail("Distribution grouping dimensions contain null evidence.")
    keys = sorted(set(zip(*columns, strict=True)), key=lambda key: tuple(map(str, key)))
    return tuple(
        (
            tuple(value.item() if isinstance(value, np.generic) else value for value in key),
            np.logical_and.reduce(
                [column == value for column, value in zip(columns, key, strict=True)]
            ),
        )
        for key in keys
    )


def _identity_for_group(
    arrays: Mapping[str, np.ndarray],
    group_mask: np.ndarray,
    *,
    fallback: Mapping[str, Any],
) -> Mapping[str, Any]:
    result: dict[str, Any] = {}
    for name, raw in arrays.items():
        values = np.asarray(raw, dtype=object)[group_mask]
        unique = {
            value.item() if isinstance(value, np.generic) else value for value in values
        }
        if not unique:
            result[name] = fallback[name]
            continue
        if len(unique) != 1:
            _fail(f"One histogram group maps to multiple exact {name!r} values.")
        result[name] = unique.pop()
    return MappingProxyType(result)


def _axis_audit(
    config: RecordingBehaviorDistributionConfig,
    item: RecordingDistributionMetricInput,
) -> Mapping[str, Any]:
    finite = item.valid & np.isfinite(item.values)
    values = item.values[finite]
    _validate_metric_values(item.spec, values)
    indices = (
        canonical_grid_indices(values, item.spec)
        if values.size
        else np.asarray([], dtype=np.int64)
    )
    recipe = histogram_recipe_record(item.spec)
    body = {
        "distribution_run_id": config.distribution_run_id,
        "recording_id": config.recording_id,
        "metric_id": item.spec.metric_id,
        "metric_spec_sha256": item.spec.spec_sha256,
        "histogram_recipe_sha256": recipe["histogram_recipe_sha256"],
        "candidate_count": int(item.values.size),
        "valid_finite_count": int(values.size),
        "minimum_value": None if not values.size else float(np.min(values)),
        "maximum_value": None if not values.size else float(np.max(values)),
        "minimum_grid_index": None if not indices.size else int(np.min(indices)),
        "maximum_grid_index": None if not indices.size else int(np.max(indices)),
        "grid_origin": (
            "log10_zero_exponent"
            if item.spec.coverage_policy == "log10_cover_valid_positive_range"
            else (
                float(item.spec.lower_bound)
                if item.spec.coverage_policy == "fixed_closed_terminal"
                else 0.0
            )
        ),
        "grid_width": float(item.spec.bin_width),
    }
    return MappingProxyType(
        {**body, "axis_audit_sha256": canonical_json_sha256(body)}
    )


def compute_recording_behavior_distributions(
    config: RecordingBehaviorDistributionConfig,
    metric_inputs: Sequence[RecordingDistributionMetricInput],
) -> RecordingBehaviorDistributionResult:
    """Reduce explicit inputs into recording support and sparse histogram bins."""

    inputs = tuple(metric_inputs)
    if not inputs:
        _fail("At least one recording distribution metric input is required.")
    metric_ids = [item.spec.metric_id for item in inputs]
    if len(set(metric_ids)) != len(metric_ids):
        _fail("Recording distribution metric IDs must be unique.")
    support_rows: list[Mapping[str, Any]] = []
    bin_rows: list[Mapping[str, Any]] = []
    audits: list[Mapping[str, Any]] = []

    for item in sorted(inputs, key=lambda value: value.spec.metric_id):
        _validate_scope_roster(config, item)
        audits.append(_axis_audit(config, item))
        finite_valid = item.valid & np.isfinite(item.values)
        recipe_sha256 = histogram_recipe_record(item.spec)[
            "histogram_recipe_sha256"
        ]
        _validate_metric_values(item.spec, item.values[finite_valid])
        group_names = tuple(item.group_arrays)
        for group_values, group_mask in _group_roster(
            item.group_arrays, row_count=item.values.size
        ):
            group = {
                name: _plain(value)
                for name, value in zip(group_names, group_values, strict=True)
            }
            group_json, group_sha256 = _json_identity(group)
            identity = _identity_for_group(
                item.source_identity_arrays,
                group_mask,
                fallback=item.source_identity_fallback,
            )
            identity_json, identity_sha256 = _json_identity(identity)
            for scope in config.scopes:
                for weighting_id in item.spec.weighting_ids:
                    projection = (
                        item.time_scope_projection
                        if weighting_id == "time"
                        else item.scope_projection
                    )
                    assert projection is not None
                    scope_mask = np.asarray(
                        projection.masks[scope.scope_id], dtype=bool
                    ) & group_mask
                    uncovered_mask = np.asarray(
                        projection.uncovered[scope.scope_id], dtype=bool
                    ) & group_mask
                    candidate_count = int(np.count_nonzero(scope_mask))
                    selected = scope_mask & finite_valid
                    if weighting_id == "time":
                        assert item.time_weights_s is not None
                        weight_valid = np.isfinite(item.time_weights_s) & (
                            item.time_weights_s > 0
                        )
                        selected &= weight_valid
                        selected_weights = item.time_weights_s[selected]
                        weight_unit = "s"
                    else:
                        selected_weights = np.ones(
                            int(np.count_nonzero(selected)), dtype=np.float64
                        )
                        weight_unit = "count"
                    selected_values = item.values[selected]
                    valid_count = int(selected_values.size)
                    denominator_weight = float(np.sum(selected_weights))
                    duration = item.valid_duration_s_by_scope.get(scope.scope_id)
                    rate = (
                        float(valid_count) / (float(duration) / 60.0)
                        if weighting_id == "event"
                        and duration is not None
                        and float(duration) > 0
                        else None
                    )
                    support_key = canonical_json_sha256(
                        {
                            "distribution_run_id": config.distribution_run_id,
                            "recording_id": config.recording_id,
                            "metric_id": item.spec.metric_id,
                            "scope_sha256": scope.record["scope_sha256"],
                            "group_key_sha256": group_sha256,
                            "source_identity_key_sha256": identity_sha256,
                            "weighting_id": weighting_id,
                            "membership_policy_id": projection.membership_policy_id,
                        }
                    )
                    support_rows.append(
                        MappingProxyType(
                            {
                                "distribution_run_id": config.distribution_run_id,
                                "recording_id": config.recording_id,
                                "metric_id": item.spec.metric_id,
                                "metric_spec_sha256": item.spec.spec_sha256,
                                "histogram_recipe_sha256": recipe_sha256,
                                "metric_family": item.spec.metric_family,
                                "source_surface": item.spec.source_surface,
                                "scope_id": scope.scope_id,
                                "scope_sha256": scope.record["scope_sha256"],
                                "scope_provider_id": scope.scope_provider_id,
                                "membership_policy_id": projection.membership_policy_id,
                                "group_key_json": group_json,
                                "group_key_sha256": group_sha256,
                                "source_identity_key_json": identity_json,
                                "source_identity_key_sha256": identity_sha256,
                                "weighting_id": weighting_id,
                                "weight_unit": weight_unit,
                                "candidate_count": candidate_count,
                                "valid_count": valid_count,
                                "excluded_count": candidate_count - valid_count,
                                "uncovered_count": int(
                                    np.count_nonzero(uncovered_mask)
                                ),
                                "denominator_weight": denominator_weight,
                                "valid_duration_s": duration,
                                "event_rate_per_valid_min": rate,
                                "minimum_value": (
                                    None
                                    if not valid_count
                                    else float(np.min(selected_values))
                                ),
                                "maximum_value": (
                                    None
                                    if not valid_count
                                    else float(np.max(selected_values))
                                ),
                                "support_state": (
                                    "finite"
                                    if denominator_weight > 0
                                    else "zero_denominator"
                                ),
                                "support_key_sha256": support_key,
                            }
                        )
                    )
                    if not valid_count:
                        continue
                    grid = canonical_grid_indices(selected_values, item.spec)
                    unique, inverse = np.unique(grid, return_inverse=True)
                    counts = np.bincount(inverse).astype(np.int64)
                    weights = np.bincount(
                        inverse, weights=selected_weights
                    ).astype(np.float64)
                    for grid_index, count, weight in zip(
                        unique, counts, weights, strict=True
                    ):
                        left, right = canonical_grid_bounds(
                            int(grid_index), item.spec
                        )
                        bin_rows.append(
                            MappingProxyType(
                                {
                                    "distribution_run_id": config.distribution_run_id,
                                    "recording_id": config.recording_id,
                                    "metric_id": item.spec.metric_id,
                                    "metric_spec_sha256": item.spec.spec_sha256,
                                    "histogram_recipe_sha256": recipe_sha256,
                                    "scope_id": scope.scope_id,
                                    "scope_sha256": scope.record["scope_sha256"],
                                    "group_key_sha256": group_sha256,
                                    "source_identity_key_sha256": identity_sha256,
                                    "weighting_id": weighting_id,
                                    "support_key_sha256": support_key,
                                    "grid_index": int(grid_index),
                                    "bin_left": float(left),
                                    "bin_right": float(right),
                                    "bin_center": float((left + right) / 2.0),
                                    "bin_count": int(count),
                                    "bin_weight": float(weight),
                                    "fraction": float(weight) / denominator_weight,
                                }
                            )
                        )

    order = {scope.scope_id: scope.order for scope in config.scopes}
    support_rows.sort(
        key=lambda row: (
            row["metric_id"],
            order[str(row["scope_id"])],
            row["group_key_sha256"],
            row["weighting_id"],
        )
    )
    bin_rows.sort(
        key=lambda row: (
            row["metric_id"],
            order[str(row["scope_id"])],
            row["group_key_sha256"],
            row["weighting_id"],
            row["grid_index"],
        )
    )
    support_keys = [row["support_key_sha256"] for row in support_rows]
    if len(set(support_keys)) != len(support_keys):
        _fail("Recording distribution support primary key is duplicated.")
    metric_registry = tuple(
        recording_metric_registry_record(item.spec)
        for item in sorted(inputs, key=lambda value: value.spec.metric_id)
    )
    return RecordingBehaviorDistributionResult(
        config=config,
        metric_registry=metric_registry,
        axis_audits=tuple(audits),
        support=tuple(support_rows),
        sparse_bins=tuple(bin_rows),
    )


__all__ = [
    "METHOD_ID",
    "RecordingBehaviorDistributionConfig",
    "RecordingBehaviorDistributionError",
    "RecordingBehaviorDistributionResult",
    "RecordingDistributionMetricInput",
    "SCHEMA_ID",
    "SCHEMA_VERSION",
    "canonical_grid_bounds",
    "canonical_grid_indices",
    "compute_recording_behavior_distributions",
]
