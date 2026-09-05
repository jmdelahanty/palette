"""Paradigm-neutral metric recipes for recording-local distributions."""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType
from typing import Mapping, Sequence

from fisheye.shared.zarr.manifest_digest import canonical_json_sha256

from .validated_behavior_distribution_specs import (
    DEFAULT_DISTRIBUTION_METRICS,
    DistributionMetricSpec,
    validate_distribution_metric_specs,
)


_SCOPE_CONTRACT_BY_SURFACE: Mapping[str, str] = MappingProxyType(
    {
        "bout_observations": (
            "source_membership_or_event_full_containment_in_requested_scope_v1"
        ),
        "inter_bout_interval_observations": (
            "both_interval_endpoints_inside_requested_scope_v1"
        ),
        "provider_motion_samples": (
            "sample_or_both_transition_endpoints_inside_requested_scope_v1"
        ),
        "chaser_relative_samples": (
            "sample_or_both_transition_endpoints_inside_requested_scope_v1"
        ),
    }
)


def histogram_recipe_record(spec: DistributionMetricSpec) -> Mapping[str, object]:
    """Return the scope-independent histogram identity used for safe fan-in."""

    body = {
        "metric_id": spec.metric_id,
        "source_surface": spec.source_surface,
        "value_column": spec.value_column,
        "unit": spec.unit,
        "bin_width": float(spec.bin_width),
        "lower_bound": float(spec.lower_bound),
        "upper_bound": (
            None if spec.upper_bound is None else float(spec.upper_bound)
        ),
        "coverage_policy": spec.coverage_policy,
        "terminal_bin_policy": "right_closed_only_for_final_bin",
        "axis_scale": (
            "log10"
            if spec.coverage_policy == "log10_cover_valid_positive_range"
            else "linear"
        ),
        "weighting_ids": list(spec.weighting_ids),
        "group_columns": list(spec.group_columns),
        "validity_policy_id": spec.validity_policy_id,
    }
    return MappingProxyType(
        {**body, "histogram_recipe_sha256": canonical_json_sha256(body)}
    )


DEFAULT_RECORDING_DISTRIBUTION_METRICS: tuple[DistributionMetricSpec, ...] = tuple(
    replace(spec, scope_binding_id=_SCOPE_CONTRACT_BY_SURFACE[spec.source_surface])
    for spec in DEFAULT_DISTRIBUTION_METRICS
)

_PARENT_SPEC_SHA256: Mapping[str, str] = MappingProxyType(
    {spec.metric_id: spec.spec_sha256 for spec in DEFAULT_DISTRIBUTION_METRICS}
)


def recording_metric_registry_record(
    spec: DistributionMetricSpec,
) -> Mapping[str, object]:
    """Add cohort lineage and the scope-independent recipe digest."""

    return MappingProxyType(
        {
            **spec.to_dict(),
            "metric_spec_sha256": spec.spec_sha256,
            "parent_cohort_metric_spec_sha256": _PARENT_SPEC_SHA256.get(
                spec.metric_id
            ),
            **dict(histogram_recipe_record(spec)),
        }
    )


def recording_distribution_metric_specs_for_families(
    families: Sequence[str] = (),
) -> tuple[DistributionMetricSpec, ...]:
    requested = tuple(str(value).strip() for value in families if str(value).strip())
    available = {
        spec.metric_family for spec in DEFAULT_RECORDING_DISTRIBUTION_METRICS
    }
    unknown = sorted(set(requested) - available)
    if unknown:
        raise KeyError(f"Unknown recording distribution metric families: {unknown}")
    selected = (
        DEFAULT_RECORDING_DISTRIBUTION_METRICS
        if not requested
        else tuple(
            spec
            for spec in DEFAULT_RECORDING_DISTRIBUTION_METRICS
            if spec.metric_family in requested
        )
    )
    return validate_distribution_metric_specs(selected)


__all__ = [
    "DEFAULT_RECORDING_DISTRIBUTION_METRICS",
    "histogram_recipe_record",
    "recording_distribution_metric_specs_for_families",
    "recording_metric_registry_record",
]
