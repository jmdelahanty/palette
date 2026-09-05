"""Normalized view payloads for validated-behavior grouped statistics.

Static and interactive renderers consume these payloads instead of querying
Parquet independently.  This keeps filtering, group-key decoding, metric
identity, missingness, and display provenance identical across viewers.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from fisheye.group_statistics.validated_behavior import (
    read_validated_behavior_group_statistics_sandbox,
)
from fisheye.group_statistics.validated_behavior_appearance import (
    behavior_role_styles,
    validate_chaser_appearance_dimension,
)
from fisheye.shared.bounded_identity_cache import BoundedIdentityCache
from fisheye.shared.json_safety import json_attr_safe
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256

SCHEMA_ID = "palette.analytics.validated_behavior.group_statistics.view_payload"
SCHEMA_VERSION = 2
LEGACY_SCHEMA_VERSION = 1
METHOD_ID = "shared_normalized_statistics_view_payload_v2"
LEGACY_METHOD_ID = "shared_normalized_statistics_view_payload_v1"

CONDITION_ORDER = ("chaser_pre", "chaser_training", "chaser_post")
CONDITION_LABELS: Mapping[str, str] = MappingProxyType(
    {
        "chaser_pre": "Pre",
        "chaser_training": "Training",
        "chaser_post": "Post",
        "__all__": "All",
    }
)
CONDITION_COLORS: Mapping[str, str] = MappingProxyType(
    {
        "chaser_pre": "#4C78A8",
        "chaser_training": "#E45756",
        "chaser_post": "#54A24B",
    }
)
BEHAVIOR_ROLE_COLORS: Mapping[str, str] = MappingProxyType(
    {
        "aggressive": "#D1495B",
        "inert": "#4C78A8",
    }
)
PROVIDER_LINE_STYLES: Mapping[str, str] = MappingProxyType(
    {"keypoint": "solid", "detection": "dashed"}
)


@dataclass(frozen=True, slots=True)
class StatisticsViewDefinition:
    view_id: str
    label: str
    metric_family: str
    renderer_kind: str
    description: str
    default_metric_id: str


VIEW_DEFINITIONS: tuple[StatisticsViewDefinition, ...] = (
    StatisticsViewDefinition(
        "core_behavior",
        "Core behavior",
        "core_behavior",
        "condition",
        "Recording-weighted speed, bout, heading, IBI, and tracking summaries.",
        "core_behavior.mean_speed_mm_s",
    ),
    StatisticsViewDefinition(
        "distance_traveled",
        "Distance traveled",
        "distance_traveled",
        "distance_traveled",
        (
            "Whole-session and exact-epoch observed path distance with "
            "recording-equal cohort summaries."
        ),
        "distance_traveled.epoch_total_path_mm",
    ),
    StatisticsViewDefinition(
        "near_field",
        "Near field",
        "near_field",
        "condition_group",
        "Distance and near-zone behavior by provider, role, and epoch.",
        "near_field.near_zone_fraction_valid",
    ),
    StatisticsViewDefinition(
        "same_quadrant",
        "Same quadrant",
        "same_quadrant",
        "condition_group",
        "Same-quadrant occupancy by provider, role, and epoch.",
        "same_quadrant.same_quadrant_fraction_valid",
    ),
    StatisticsViewDefinition(
        "occupancy_support",
        "Tracking and arena coverage",
        "occupancy_support",
        "condition_group",
        "Position-provider coverage and in-arena support by epoch.",
        "occupancy_support.in_arena_coverage_fraction_candidate",
    ),
    StatisticsViewDefinition(
        "bout_response_by_distance",
        "Bout response by distance",
        "bout_response_by_distance",
        "distance_curve",
        "Recording-weighted bout rates and kinematics over persisted distance bins.",
        "bout_response_by_distance.bout_rate_per_min",
    ),
    StatisticsViewDefinition(
        "body_alignment_by_distance",
        "Body alignment by distance",
        "body_alignment_by_distance",
        "distance_curve",
        "Anatomical body alignment and bearing summaries over distance.",
        "body_alignment_by_distance.mean_alignment_cos",
    ),
    StatisticsViewDefinition(
        "body_bearing_polar",
        "Signed anatomical bearing",
        "body_bearing_polar",
        "bearing_polar",
        (
            "Equal-recording-weight signed anatomical body-bearing "
            "distributions by epoch and semantic chaser role."
        ),
        "body_bearing_polar.recording_fraction",
    ),
    StatisticsViewDefinition(
        "body_bearing_distance",
        "Signed bearing by chaser distance",
        "body_bearing_distance",
        "bearing_distance_heatmap",
        (
            "Equal-recording-weight joint signed anatomical bearing and "
            "fish--chaser distance distributions."
        ),
        "body_bearing_distance.recording_joint_fraction",
    ),
    StatisticsViewDefinition(
        "trial_response",
        "Trial response",
        "trial_response",
        "trial_curve",
        "Escape, freeze, recapture, and separation summaries by trial ordinal.",
        "trial_response.escape_event_rate_per_min",
    ),
    StatisticsViewDefinition(
        "spatial_occupancy",
        "Spatial occupancy",
        "spatial_occupancy",
        "spatial_heatmap",
        "Equal-recording-weight spatial occupancy by provider and epoch.",
        "spatial_occupancy.occupancy_density_valid_in_arena",
    ),
    StatisticsViewDefinition(
        "radial_distribution",
        "Radial distribution",
        "radial_distribution",
        "distance_curve",
        "Observed and geometry-corrected radial distributions.",
        "radial_distribution.observed_fraction",
    ),
    StatisticsViewDefinition(
        "distance_cdf",
        "Distance CDF",
        "distance_cdf",
        "distance_curve",
        "Recording-weighted fish-to-chaser distance CDFs.",
        "distance_cdf.fraction_at_or_below",
    ),
)


class ValidatedBehaviorStatisticsViewError(ValueError):
    """Raised when a statistics generation cannot form a safe view payload."""


def _fail(message: str) -> None:
    raise ValidatedBehaviorStatisticsViewError(message)


@dataclass(frozen=True, slots=True)
class ValidatedBehaviorStatisticsViewSource:
    root: Path
    manifest: Mapping[str, Any]
    output_paths: Mapping[str, Path]

    @classmethod
    def open(cls, root: str | Path) -> "ValidatedBehaviorStatisticsViewSource":
        resolved = Path(root).expanduser().resolve()
        validated = read_validated_behavior_group_statistics_sandbox(resolved)
        manifest = {
            key: value for key, value in validated.items() if key != "manifest_path"
        }
        records = manifest.get("outputs")
        if not isinstance(
            records, list
        ):  # pragma: no cover - strict reader proves this
            _fail("Statistics manifest lacks exact outputs")
        paths: dict[str, Path] = {}
        for record in records:
            if not isinstance(record, Mapping):
                _fail("Statistics output record is malformed")
            name = str(record.get("path"))
            path = (resolved / name).resolve()
            if path.parent != resolved or path.name != name:
                _fail("Statistics view output escapes its exact generation")
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
    def statistics_run_id(self) -> str:
        return str(self.manifest["statistics_run_id"])

    def scan(self, name: str) -> Any:
        import polars as pl

        try:
            path = self.output_paths[name]
        except KeyError as exc:
            raise KeyError(f"Unknown grouped-statistics output: {name}") from exc
        return pl.scan_parquet(path)


def available_statistics_views(
    source: ValidatedBehaviorStatisticsViewSource,
) -> tuple[StatisticsViewDefinition, ...]:
    configuration = source.manifest.get("configuration", {})
    metric_specs = configuration.get("metric_specs", [])
    histogram_specs = configuration.get("histogram_specs", [])
    families = {
        str(record.get("metric_family"))
        for record in (*metric_specs, *histogram_specs)
        if isinstance(record, Mapping)
    }
    return tuple(
        definition
        for definition in VIEW_DEFINITIONS
        if definition.metric_family in families
    )


def _metric_catalog(
    source: ValidatedBehaviorStatisticsViewSource,
    family: str,
) -> tuple[dict[str, object], ...]:
    configuration = source.manifest.get("configuration", {})
    raw = configuration.get("metric_specs", [])
    metric_records = [
        record
        for record in raw
        if isinstance(record, Mapping) and record.get("metric_family") == family
    ]
    histogram_raw = configuration.get("histogram_specs", [])
    histogram_records = [
        record
        for record in histogram_raw
        if isinstance(record, Mapping) and record.get("metric_family") == family
    ]
    if metric_records and histogram_records:
        _fail(f"Statistics family {family!r} mixes scalar and histogram specs")
    if not metric_records and not histogram_records:
        _fail(f"Statistics generation lacks metric family {family!r}")
    if histogram_records:
        return tuple(
            {
                "metric_id": str(record["metric_id"]),
                "value_column": "fraction",
                "unit": "fraction",
                "interpretation": str(record["interpretation"]),
                "condition_column": record.get("condition_column"),
                "group_columns": list(record.get("group_columns", [])),
                "contrast_eligible": False,
                "retain_recording_values": True,
                "metric_spec_sha256": canonical_json_sha256(record),
                "recording_reducer": record.get("recording_reducer"),
                "cohort_reducer": record.get("cohort_reducer"),
                "axes": list(record.get("axes", [])),
            }
            for record in histogram_records
        )
    return tuple(
        {
            "metric_id": str(record["metric_id"]),
            "value_column": str(record["value_column"]),
            "unit": str(record["unit"]),
            "interpretation": str(record["interpretation"]),
            "condition_column": record.get("condition_column"),
            "group_columns": list(record.get("group_columns", [])),
            "contrast_eligible": record.get("contrast_set_id") is not None,
            "retain_recording_values": bool(record.get("retain_recording_values")),
            "metric_spec_sha256": str(
                canonical_json_sha256(
                    {
                        key: value
                        for key, value in record.items()
                        if key not in {"metric_spec_sha256"}
                    }
                )
            ),
        }
        for record in metric_records
    )


def _decode_group(row: Mapping[str, Any]) -> dict[str, Any]:
    try:
        group = json.loads(str(row["group_key_json"]))
    except (KeyError, json.JSONDecodeError) as exc:
        raise ValidatedBehaviorStatisticsViewError(
            "Statistics row has an invalid group key"
        ) from exc
    if not isinstance(group, dict):
        _fail("Statistics group key must be one object")
    return {str(key): json_attr_safe(value) for key, value in group.items()}


def _normalized_rows(
    source: ValidatedBehaviorStatisticsViewSource,
    *,
    output_name: str,
    family: str,
    columns: Sequence[str],
) -> tuple[dict[str, object], ...]:
    import polars as pl

    lazy = source.scan(output_name).filter(pl.col("metric_family") == family)
    known = set(lazy.collect_schema().names())
    selected = tuple(columns)
    missing = sorted(set(selected) - known)
    if missing:
        _fail(f"{output_name}: normalized view columns are absent: {missing}")
    frame = (
        lazy.select(selected)
        .sort(
            tuple(
                name
                for name in (
                    "metric_id",
                    "condition",
                    "contrast_id",
                    "recording_id",
                    "group_key_sha256",
                    "axis_0_bin_index",
                    "axis_1_bin_index",
                )
                if name in selected
            )
        )
        .collect()
    )
    rows: list[dict[str, object]] = []
    for raw in frame.to_dicts():
        group = _decode_group(raw)
        collisions = set(group) & set(raw)
        if collisions:
            _fail(f"Decoded group dimensions collide with result columns: {collisions}")
        row = {
            key: json_attr_safe(value)
            for key, value in raw.items()
            if key not in {"group_key_json"}
        }
        row.update(group)
        rows.append(row)
    return tuple(rows)


def build_statistics_view_payload(
    source: ValidatedBehaviorStatisticsViewSource,
    view_id: str,
    *,
    payload_cache: (
        BoundedIdentityCache[tuple[str, str, str, str], Mapping[str, object]]
        | None
    ) = None,
) -> Mapping[str, object]:
    """Build one view, optionally reusing its exact immutable identity."""

    if payload_cache is not None:
        cache_key = (METHOD_ID, str(source.root), source.cache_identity, view_id)
        return payload_cache.get_or_load(
            cache_key,
            lambda: build_statistics_view_payload(source, view_id),
        )
    try:
        definition = next(item for item in VIEW_DEFINITIONS if item.view_id == view_id)
    except StopIteration as exc:
        raise KeyError(
            f"Unknown validated-behavior statistics view: {view_id}"
        ) from exc
    available = {item.view_id for item in available_statistics_views(source)}
    if definition.view_id not in available:
        _fail(f"Statistics view is unavailable: {definition.view_id}")

    catalog = _metric_catalog(source, definition.metric_family)
    histogram_view = definition.renderer_kind in {
        "bearing_polar",
        "bearing_distance_heatmap",
    }
    if histogram_view:
        descriptive = _normalized_rows(
            source,
            output_name="histogram_descriptive_statistics.parquet",
            family=definition.metric_family,
            columns=(
                "metric_id",
                "condition_axis",
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
                "parent_recording_count",
                "contributor_recording_count",
                "finite_recording_count",
                "excluded_zero_denominator_recording_count",
                "noncontributor_recording_count",
                "source_bin_count_sum",
                "source_denominator_count_sum",
                "mean_fraction",
                "median_fraction",
                "sample_std_fraction",
                "sem_fraction",
                "minimum_fraction",
                "p25_fraction",
                "p75_fraction",
                "maximum_fraction",
            ),
        )
        recording = _normalized_rows(
            source,
            output_name="recording_histogram_bins.parquet",
            family=definition.metric_family,
            columns=(
                "metric_id",
                "recording_id",
                "condition_axis",
                "condition",
                "group_key_json",
                "group_key_sha256",
                "source_identity_key_json",
                "source_identity_key_sha256",
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
                "candidate_row_count",
                "bin_count",
                "denominator_count",
                "fraction",
                "value_state",
            ),
        )
        contrasts = ()
    else:
        descriptive = _normalized_rows(
            source,
            output_name="descriptive_statistics.parquet",
            family=definition.metric_family,
            columns=(
                "metric_id",
                "unit",
                "condition_axis",
                "condition",
                "group_key_json",
                "group_key_sha256",
                "parent_recording_count",
                "contributor_recording_count",
                "finite_recording_count",
                "excluded_nonfinite_recording_count",
                "noncontributor_recording_count",
                "mean",
                "median",
                "sample_std",
                "sem",
                "minimum",
                "p25",
                "p75",
                "maximum",
            ),
        )
        recording = _normalized_rows(
            source,
            output_name="recording_metric_values.parquet",
            family=definition.metric_family,
            columns=(
                "metric_id",
                "unit",
                "recording_id",
                "condition_axis",
                "condition",
                "group_key_json",
                "group_key_sha256",
                "value",
                "value_state",
            ),
        )
        contrasts = _normalized_rows(
            source,
            output_name="paired_contrasts.parquet",
            family=definition.metric_family,
            columns=(
                "metric_id",
                "unit",
                "condition_axis",
                "contrast_id",
                "condition_a",
                "condition_b",
                "group_key_json",
                "group_key_sha256",
                "parent_recording_count",
                "eligible_recording_count",
                "paired_unit_count",
                "excluded_unit_count",
                "mean_difference",
                "median_difference",
                "ci_low",
                "ci_high",
                "p_value",
                "q_value",
                "test_method",
                "status",
                "skip_reason",
            ),
        )
    recipes = [
        record
        for record in source.manifest.get("histogram_recipes", [])
        if isinstance(record, Mapping)
        and record.get("metric_family") == definition.metric_family
    ]
    source_export = source.manifest.get("source_export")
    if not isinstance(source_export, Mapping):
        _fail("Statistics source lacks its exact export identity")
    raw_appearance = source_export.get("chaser_appearance_dimension")
    appearance = (
        None
        if raw_appearance is None
        else validate_chaser_appearance_dimension(
            raw_appearance,
            expected_export_manifest_sha256=str(
                source_export.get("export_manifest_record_sha256")
            ),
        )
    )
    role_styles = behavior_role_styles(
        appearance,
        legacy_display_colors=BEHAVIOR_ROLE_COLORS,
    )
    role_colors = {
        role: str(style["aggregate_color_hex"]) for role, style in role_styles.items()
    }
    body: dict[str, object] = {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "view_id": definition.view_id,
        "label": definition.label,
        "description": definition.description,
        "renderer_kind": definition.renderer_kind,
        "default_metric_id": definition.default_metric_id,
        "metric_family": definition.metric_family,
        "metric_catalog": list(catalog),
        "recording_rows": list(recording),
        "descriptive_rows": list(descriptive),
        "contrast_rows": list(contrasts),
        "histogram_recipes": recipes,
        "condition_order": list(CONDITION_ORDER),
        "condition_labels": dict(CONDITION_LABELS),
        "condition_colors": dict(CONDITION_COLORS),
        "behavior_role_colors": role_colors,
        "behavior_role_styles": json_attr_safe(role_styles),
        "chaser_appearance_dimension": (
            None if appearance is None else json_attr_safe(appearance)
        ),
        "provider_line_styles": dict(PROVIDER_LINE_STYLES),
        "source_statistics": {
            "statistics_run_id": source.statistics_run_id,
            "statistics_manifest_sha256": source.cache_identity,
            "source_export_manifest_sha256": source.manifest["source_export"][
                "export_manifest_record_sha256"
            ],
            "analysis_status": "exploratory",
            "experimental_unit": "recording_id",
            "cohort_weighting": "equal_weight_per_finite_recording",
            "acquisition_batch_adjustment": "not_performed_identity_unavailable",
        },
    }
    return MappingProxyType({**body, "payload_sha256": canonical_json_sha256(body)})


def validate_statistics_view_payload(payload: Mapping[str, object]) -> None:
    """Fail closed when a normalized view payload is stale or unsupported."""

    if payload.get("schema_id") != SCHEMA_ID or payload.get("schema_version") not in {
        LEGACY_SCHEMA_VERSION,
        SCHEMA_VERSION,
    }:
        _fail("Statistics view payload schema is unsupported")
    expected_method = (
        LEGACY_METHOD_ID
        if payload.get("schema_version") == LEGACY_SCHEMA_VERSION
        else METHOD_ID
    )
    if payload.get("method_id") != expected_method:
        _fail("Statistics view payload method is unsupported")
    body = {key: value for key, value in payload.items() if key != "payload_sha256"}
    if payload.get("payload_sha256") != canonical_json_sha256(body):
        _fail("Statistics view payload digest is stale")
    definitions = {definition.view_id: definition for definition in VIEW_DEFINITIONS}
    definition = definitions.get(str(payload.get("view_id")))
    if definition is None:
        _fail("Statistics view payload identifies an unknown view")
    expected = {
        "label": definition.label,
        "description": definition.description,
        "renderer_kind": definition.renderer_kind,
        "default_metric_id": definition.default_metric_id,
        "metric_family": definition.metric_family,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            _fail(f"Statistics view payload has stale {field}")
    catalog = payload.get("metric_catalog")
    if not isinstance(catalog, list) or not catalog:
        _fail("Statistics view payload has no metric catalog")
    metric_ids = {
        str(record.get("metric_id"))
        for record in catalog
        if isinstance(record, Mapping)
    }
    if (
        len(metric_ids) != len(catalog)
        or definition.default_metric_id not in metric_ids
    ):
        _fail("Statistics view payload metric catalog is incomplete or ambiguous")
    for field in ("recording_rows", "descriptive_rows", "contrast_rows"):
        if not isinstance(payload.get(field), list):
            _fail(f"Statistics view payload lacks exact {field}")
    recipes = payload.get("histogram_recipes")
    if not isinstance(recipes, list):
        _fail("Statistics view payload lacks its histogram-recipe inventory")
    histogram_view = definition.renderer_kind in {
        "bearing_polar",
        "bearing_distance_heatmap",
    }
    if histogram_view and len(recipes) != 1:
        _fail("Histogram view payload does not bind exactly one resolved recipe")
    if not histogram_view and recipes:
        _fail("Scalar statistics view unexpectedly binds a histogram recipe")
    for recipe in recipes:
        if not isinstance(recipe, Mapping):
            _fail("Statistics histogram recipe is malformed")
        recipe_body = {
            key: value
            for key, value in recipe.items()
            if key != "histogram_recipe_sha256"
        }
        if recipe.get("histogram_recipe_sha256") != canonical_json_sha256(recipe_body):
            _fail("Statistics histogram recipe digest is stale")
    if payload.get("schema_version") == SCHEMA_VERSION:
        source = payload.get("source_statistics")
        if not isinstance(source, Mapping):
            _fail("Statistics view payload lacks source-statistics identity")
        raw_appearance = payload.get("chaser_appearance_dimension")
        appearance = (
            None
            if raw_appearance is None
            else validate_chaser_appearance_dimension(
                raw_appearance,
                expected_export_manifest_sha256=str(
                    source.get("source_export_manifest_sha256")
                ),
            )
        )
        expected_styles = json_attr_safe(
            behavior_role_styles(
                appearance,
                legacy_display_colors=BEHAVIOR_ROLE_COLORS,
            )
        )
        if payload.get("behavior_role_styles") != expected_styles:
            _fail("Statistics view payload has stale behavior-role styles")
        expected_colors = {
            role: str(style["aggregate_color_hex"])
            for role, style in expected_styles.items()
        }
        if payload.get("behavior_role_colors") != expected_colors:
            _fail("Statistics view payload has stale aggregate role colors")
    source = payload.get("source_statistics")
    if not isinstance(source, Mapping):
        _fail("Statistics view payload lacks source-statistics identity")
    for field in (
        "statistics_manifest_sha256",
        "source_export_manifest_sha256",
    ):
        value = source.get(field)
        if not isinstance(value, str) or len(value) != 64:
            _fail(f"Statistics view payload has invalid source digest: {field}")


def build_initial_statistics_view_payloads(
    source: ValidatedBehaviorStatisticsViewSource,
) -> Mapping[str, Mapping[str, object]]:
    return MappingProxyType(
        {
            definition.view_id: build_statistics_view_payload(
                source, definition.view_id
            )
            for definition in available_statistics_views(source)
        }
    )


__all__ = [
    "BEHAVIOR_ROLE_COLORS",
    "CONDITION_COLORS",
    "CONDITION_LABELS",
    "CONDITION_ORDER",
    "METHOD_ID",
    "PROVIDER_LINE_STYLES",
    "SCHEMA_ID",
    "SCHEMA_VERSION",
    "StatisticsViewDefinition",
    "VIEW_DEFINITIONS",
    "ValidatedBehaviorStatisticsViewError",
    "ValidatedBehaviorStatisticsViewSource",
    "available_statistics_views",
    "build_initial_statistics_view_payloads",
    "build_statistics_view_payload",
    "validate_statistics_view_payload",
]
