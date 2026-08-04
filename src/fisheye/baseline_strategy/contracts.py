"""Contracts and configuration for baseline strategy analytics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from typing import Any, Mapping, Sequence

from fisheye.analytics_exports.arrow_contract_core import (
    ArrowTableContract,
    canonical_bytes,
    contract_envelope,
    exact_schema,
    field,
    validate_contract_envelope,
    validate_exact_schema,
)


SCHEMA_ID = "palette.baseline_strategy_analytics"
LEGACY_SCHEMA_VERSION = 1
SCHEMA_VERSION = 2
METHOD = "fish_rodent_open_field_strategy_features"
METHOD_VERSION = "1"

ARROW_CONTRACT_ENVELOPE_SCHEMA_ID = (
    "palette.baseline_strategy_analytics.arrow_contracts"
)
ARROW_CONTRACT_ENVELOPE_SCHEMA_VERSION = 1
ARROW_TABLE_SCHEMA_NAMESPACE = "palette.baseline_strategy_analytics.arrow_table"

BASELINE_STRATEGY_FEATURES_TABLE = "baseline_strategy_features"
BASELINE_EXPLORATION_EPISODES_TABLE = "baseline_exploration_episodes"
BASELINE_STRATEGY_CLASSIFICATION_TABLE = "baseline_strategy_classification"
BASELINE_STRATEGY_CLUSTERS_TABLE = "baseline_strategy_clusters"

BASELINE_STRATEGY_TABLES = (
    BASELINE_STRATEGY_FEATURES_TABLE,
    BASELINE_EXPLORATION_EPISODES_TABLE,
    BASELINE_STRATEGY_CLASSIFICATION_TABLE,
    BASELINE_STRATEGY_CLUSTERS_TABLE,
)

IDENTITY_COLUMNS = (
    "recording_id",
    "track_id",
    "baseline_window_id",
)

_ARROW_COMMON_FIELDS = (
    field("schema_id", "string"),
    field("schema_version", "int32"),
    field("table_name", "string"),
    field("method", "string"),
    field("method_version", "string"),
    field("recording_id", "string"),
    field("track_id", "int64"),
    field("baseline_window_id", "int64"),
    field("baseline_window_label", "string"),
    field("source_export_run_id", "string"),
    field("zarr_path", "string"),
)

_BASELINE_STRATEGY_FEATURES_ARROW_FIELDS = _ARROW_COMMON_FIELDS + (
    field("sample_features_available", "bool"),
    field("time_bin_features_available", "bool"),
    field("duration_s", "float64", nullable=True),
    field("tracking_dropout_fraction", "float64", nullable=True),
    field("mean_speed_mm_s", "float64", nullable=True),
    field("median_speed_mm_s", "float64", nullable=True),
    field("p95_speed_mm_s", "float64", nullable=True),
    field("total_path_mm", "float64", nullable=True),
    field("path_per_min_mm", "float64", nullable=True),
    field("bout_count", "int64", nullable=True),
    field("bout_rate_per_min", "float64", nullable=True),
    field("wall_fraction", "float64", nullable=True),
    field("wall_band_mm", "float64", nullable=True),
    field("experimental_area_geometry_type", "string"),
    field("boundary_distance_method", "string"),
    field("wall_fraction_denominator", "string"),
    field("active_wall_fraction_denominator", "string"),
    field("expected_uniform_wall_fraction", "float64", nullable=True),
    field("wall_enrichment_ratio", "float64", nullable=True),
    field("wall_enrichment_log2", "float64", nullable=True),
    field("mean_center_distance_norm", "float64", nullable=True),
    field("source_spatial_entropy_normalized", "float64", nullable=True),
    field("source_quadrant_entropy_normalized", "float64", nullable=True),
    field("source_spatial_max_cell_fraction", "float64", nullable=True),
    field("source_quadrant_max_fraction", "float64", nullable=True),
    field("active_speed_threshold_mm_s", "float64"),
    field("spatial_grid_size", "int64"),
    field("dwell_grid_size", "int64"),
    field("time_bin_count", "int64"),
    field("wall_fraction_early", "float64", nullable=True),
    field("wall_fraction_late", "float64", nullable=True),
    field("wall_fraction_delta_late_minus_early", "float64", nullable=True),
    field("wall_fraction_slope_per_baseline", "float64", nullable=True),
    field("mean_speed_mm_s_early", "float64", nullable=True),
    field("mean_speed_mm_s_late", "float64", nullable=True),
    field("mean_speed_mm_s_delta_late_minus_early", "float64", nullable=True),
    field("mean_speed_mm_s_slope_per_baseline", "float64", nullable=True),
    field("distance_travelled_mm_early", "float64", nullable=True),
    field("distance_travelled_mm_late", "float64", nullable=True),
    field(
        "distance_travelled_mm_delta_late_minus_early",
        "float64",
        nullable=True,
    ),
    field("distance_travelled_mm_slope_per_baseline", "float64", nullable=True),
    field("center_distance_norm_early", "float64", nullable=True),
    field("center_distance_norm_late", "float64", nullable=True),
    field(
        "center_distance_norm_delta_late_minus_early",
        "float64",
        nullable=True,
    ),
    field("center_distance_norm_slope_per_baseline", "float64", nullable=True),
    field("bout_count_early", "float64", nullable=True),
    field("bout_count_late", "float64", nullable=True),
    field("bout_count_delta_late_minus_early", "float64", nullable=True),
    field("bout_count_slope_per_baseline", "float64", nullable=True),
    field("valid_sample_count", "int64"),
    field("active_sample_fraction", "float64", nullable=True),
    field("boundary_distance_sample_source", "string"),
    field("active_wall_fraction", "float64", nullable=True),
    field("occupancy_accessible_cell_count", "int64", nullable=True),
    field("occupancy_visited_cell_count", "int64", nullable=True),
    field("occupancy_visited_cell_fraction", "float64", nullable=True),
    field("occupancy_coverage_fraction", "float64", nullable=True),
    field(
        "occupancy_entropy_accessible_normalized",
        "float64",
        nullable=True,
    ),
    field("occupancy_js_divergence_uniform", "float64", nullable=True),
    field("occupancy_uniform_reference", "string", nullable=True),
    field("occupancy_max_cell_fraction", "float64", nullable=True),
    field("latency_to_half_final_coverage_s", "float64", nullable=True),
    field("dominant_dwell_cell_fraction", "float64", nullable=True),
    field("dominant_to_second_dwell_ratio", "float64", nullable=True),
    field("dominant_dwell_visit_count", "int64", nullable=True),
    field("dominant_dwell_center_distance_norm", "float64", nullable=True),
    field("exploration_episode_count", "int64", nullable=True),
    field("wall_following_episode_fraction", "float64", nullable=True),
    field("dominant_dwell_excursion_count", "int64", nullable=True),
    field("dominant_dwell_return_fraction", "float64", nullable=True),
    field("median_episode_path_length_mm", "float64", nullable=True),
    field("median_episode_tortuosity", "float64", nullable=True),
    field("feature_status", "string"),
    field("feature_reason", "string", nullable=True),
    field("analysis_run_id", "string"),
)

_BASELINE_EXPLORATION_EPISODES_ARROW_FIELDS = _ARROW_COMMON_FIELDS + (
    field("episode_id", "int64"),
    field("episode_start_s", "float64"),
    field("episode_end_s", "float64"),
    field("episode_duration_s", "float64"),
    field("sample_count", "int64"),
    field("path_length_mm", "float64"),
    field("net_displacement_mm", "float64"),
    field("tortuosity", "float64", nullable=True),
    field("minimum_center_distance_mm", "float64"),
    field("maximum_inward_excursion_mm", "float64"),
    field("wall_sample_fraction", "float64", nullable=True),
    field("mean_tangential_alignment", "float64", nullable=True),
    field("wall_following", "bool"),
    field("origin_dominant_dwell_zone", "bool"),
    field("destination_dominant_dwell_zone", "bool"),
    field("returned_to_dominant_dwell_zone", "bool"),
    field("path_length_method", "string"),
    field("boundary_distance_method", "string"),
    field("analysis_run_id", "string"),
)

_BASELINE_STRATEGY_CLASSIFICATION_ARROW_FIELDS = _ARROW_COMMON_FIELDS + (
    field("activity_score", "float64", nullable=True),
    field("activity_metric_count", "int64", nullable=True),
    field("boundary_score", "float64", nullable=True),
    field("boundary_metric_count", "int64", nullable=True),
    field("spatial_distribution_score", "float64", nullable=True),
    field("spatial_distribution_metric_count", "int64", nullable=True),
    field("home_base_score", "float64", nullable=True),
    field("home_base_metric_count", "int64", nullable=True),
    field("temporal_expansion_score", "float64", nullable=True),
    field("temporal_expansion_metric_count", "int64", nullable=True),
    field("classification_status", "string"),
    field("classification_reason", "string", nullable=True),
    field("reference_scope", "string", nullable=True),
    field("relative_score_threshold", "float64", nullable=True),
    field("activity_state", "string"),
    field("boundary_strategy", "string"),
    field("home_base_state", "string", nullable=True),
    field("spatial_organization", "string"),
    field("temporal_pattern", "string"),
    field("primary_strategy", "string"),
    field("classification_confidence_score", "float64", nullable=True),
    field("confidence_semantics", "string", nullable=True),
    field("anxiety_inference_permitted", "bool", nullable=True),
    field("analysis_run_id", "string"),
)

_BASELINE_STRATEGY_CLUSTERS_ARROW_FIELDS = _ARROW_COMMON_FIELDS + (
    field("cluster_status", "string"),
    field("cluster_reason", "string", nullable=True),
    field("cluster_id", "int64", nullable=True),
    field("cluster_probability", "float64", nullable=True),
    field("cluster_probability_threshold", "float64", nullable=True),
    field("selected_component_count", "int64", nullable=True),
    field("selected_bic", "float64", nullable=True),
    field("bic_by_component_count_json", "string", nullable=True),
    field("cluster_stability_median_ari", "float64", nullable=True),
    field("cluster_stability_resample_count", "int64", nullable=True),
    field("cluster_axes", "list<string>", nullable=True),
    field("cluster_semantics", "string", nullable=True),
    field("analysis_run_id", "string"),
)

BASELINE_STRATEGY_PRIMARY_KEYS = {
    BASELINE_STRATEGY_FEATURES_TABLE: (
        "analysis_run_id",
        *IDENTITY_COLUMNS,
    ),
    BASELINE_EXPLORATION_EPISODES_TABLE: (
        "analysis_run_id",
        *IDENTITY_COLUMNS,
        "episode_id",
    ),
    BASELINE_STRATEGY_CLASSIFICATION_TABLE: (
        "analysis_run_id",
        *IDENTITY_COLUMNS,
    ),
    BASELINE_STRATEGY_CLUSTERS_TABLE: (
        "analysis_run_id",
        *IDENTITY_COLUMNS,
    ),
}

BASELINE_STRATEGY_ARROW_CONTRACTS = {
    BASELINE_STRATEGY_FEATURES_TABLE: ArrowTableContract(
        table_name=BASELINE_STRATEGY_FEATURES_TABLE,
        fields=_BASELINE_STRATEGY_FEATURES_ARROW_FIELDS,
        schema_version=SCHEMA_VERSION,
        schema_namespace=ARROW_TABLE_SCHEMA_NAMESPACE,
        primary_key=BASELINE_STRATEGY_PRIMARY_KEYS[BASELINE_STRATEGY_FEATURES_TABLE],
    ),
    BASELINE_EXPLORATION_EPISODES_TABLE: ArrowTableContract(
        table_name=BASELINE_EXPLORATION_EPISODES_TABLE,
        fields=_BASELINE_EXPLORATION_EPISODES_ARROW_FIELDS,
        schema_version=SCHEMA_VERSION,
        schema_namespace=ARROW_TABLE_SCHEMA_NAMESPACE,
        primary_key=BASELINE_STRATEGY_PRIMARY_KEYS[
            BASELINE_EXPLORATION_EPISODES_TABLE
        ],
    ),
    BASELINE_STRATEGY_CLASSIFICATION_TABLE: ArrowTableContract(
        table_name=BASELINE_STRATEGY_CLASSIFICATION_TABLE,
        fields=_BASELINE_STRATEGY_CLASSIFICATION_ARROW_FIELDS,
        schema_version=SCHEMA_VERSION,
        schema_namespace=ARROW_TABLE_SCHEMA_NAMESPACE,
        primary_key=BASELINE_STRATEGY_PRIMARY_KEYS[
            BASELINE_STRATEGY_CLASSIFICATION_TABLE
        ],
    ),
    BASELINE_STRATEGY_CLUSTERS_TABLE: ArrowTableContract(
        table_name=BASELINE_STRATEGY_CLUSTERS_TABLE,
        fields=_BASELINE_STRATEGY_CLUSTERS_ARROW_FIELDS,
        schema_version=SCHEMA_VERSION,
        schema_namespace=ARROW_TABLE_SCHEMA_NAMESPACE,
        primary_key=BASELINE_STRATEGY_PRIMARY_KEYS[BASELINE_STRATEGY_CLUSTERS_TABLE],
    ),
}


@dataclass(frozen=True)
class StrategyFeatureConfig:
    """Declared policy for geometry, activity, and QC feature derivation.

    Defaults are operational starting points, not biological truths.  They are
    serialized into every output row so a cohort result can be reproduced and
    compared only with compatible policies.
    """

    active_speed_mm_s: float = 1.0
    max_episode_gap_s: float = 0.5
    min_episode_duration_s: float = 0.2
    spatial_grid_size: int = 12
    dwell_grid_size: int = 8
    early_late_fraction: float = 0.25
    min_valid_position_fraction: float = 0.75
    min_sample_count: int = 20
    wall_following_tangential_threshold: float = 0.7
    relative_score_threshold: float = 0.75
    cluster_probability_threshold: float = 0.65
    cluster_max_components: int = 6
    cluster_stability_resamples: int = 25
    random_seed: int = 0

    def validate(self) -> None:
        if self.active_speed_mm_s < 0:
            raise ValueError("active_speed_mm_s must be non-negative")
        if self.max_episode_gap_s <= 0:
            raise ValueError("max_episode_gap_s must be positive")
        if self.min_episode_duration_s < 0:
            raise ValueError("min_episode_duration_s must be non-negative")
        if self.spatial_grid_size < 2 or self.dwell_grid_size < 2:
            raise ValueError("spatial and dwell grids must each be at least 2")
        if not 0 < self.early_late_fraction <= 0.5:
            raise ValueError("early_late_fraction must be in (0, 0.5]")
        if not 0 <= self.min_valid_position_fraction <= 1:
            raise ValueError("min_valid_position_fraction must be in [0, 1]")
        if self.min_sample_count < 1:
            raise ValueError("min_sample_count must be positive")
        if not 0 <= self.wall_following_tangential_threshold <= 1:
            raise ValueError("wall_following_tangential_threshold must be in [0, 1]")
        if self.relative_score_threshold <= 0:
            raise ValueError("relative_score_threshold must be positive")
        if not 0 < self.cluster_probability_threshold <= 1:
            raise ValueError("cluster_probability_threshold must be in (0, 1]")
        if self.cluster_max_components < 1:
            raise ValueError("cluster_max_components must be positive")
        if self.cluster_stability_resamples < 0:
            raise ValueError("cluster_stability_resamples must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


def baseline_strategy_arrow_contract_envelope(
    table_names: Sequence[str] = BASELINE_STRATEGY_TABLES,
) -> dict[str, object]:
    """Build the closed, all-exact Arrow envelope for baseline strategy v2."""

    return contract_envelope(
        table_names,
        known_table_names=BASELINE_STRATEGY_TABLES,
        contracts=BASELINE_STRATEGY_ARROW_CONTRACTS,
        schema_id=ARROW_CONTRACT_ENVELOPE_SCHEMA_ID,
        schema_version=ARROW_CONTRACT_ENVELOPE_SCHEMA_VERSION,
    )


def validate_baseline_strategy_arrow_contract_envelope(
    value: object,
    table_names: Sequence[str] = BASELINE_STRATEGY_TABLES,
) -> dict[str, object]:
    """Validate an envelope against the installed v2 declarations."""

    return validate_contract_envelope(
        value,
        table_names,
        known_table_names=BASELINE_STRATEGY_TABLES,
        contracts=BASELINE_STRATEGY_ARROW_CONTRACTS,
        schema_id=ARROW_CONTRACT_ENVELOPE_SCHEMA_ID,
        schema_version=ARROW_CONTRACT_ENVELOPE_SCHEMA_VERSION,
    )


def exact_baseline_strategy_arrow_schema(
    table_name: str,
    *,
    metadata: Mapping[bytes, bytes] | None = None,
) -> Any:
    """Return the exact digest-bound physical schema for one v2 table."""

    try:
        contract = BASELINE_STRATEGY_ARROW_CONTRACTS[table_name]
    except KeyError as exc:
        raise KeyError(f"unknown baseline strategy table: {table_name}") from exc
    return exact_schema(contract, metadata=metadata or {})


def validate_baseline_strategy_arrow_schema(table_name: str, schema: Any) -> None:
    """Reject any field or footer drift from one installed v2 schema."""

    try:
        contract = BASELINE_STRATEGY_ARROW_CONTRACTS[table_name]
    except KeyError as exc:
        raise KeyError(f"unknown baseline strategy table: {table_name}") from exc
    validate_exact_schema(contract, schema)


def _canonical_bic_json(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        def reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
            decoded: dict[str, object] = {}
            for key, item in pairs:
                if key in decoded:
                    raise ValueError(
                        "bic_by_component_count_json has duplicate component keys"
                    )
                decoded[key] = item
            return decoded

        try:
            decoded = json.loads(value, object_pairs_hook=reject_duplicate_pairs)
        except json.JSONDecodeError as exc:
            raise ValueError("bic_by_component_count_json is not valid JSON") from exc
        canonical = _canonical_bic_json(decoded)
        if canonical != value:
            raise ValueError("bic_by_component_count_json is not canonical JSON")
        return canonical
    if not isinstance(value, Mapping):
        raise TypeError("bic_by_component_count must be a mapping, JSON string, or null")
    normalized: dict[str, float] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key)
        try:
            component_count = int(key)
        except ValueError as exc:
            raise ValueError(
                "bic_by_component_count keys must be positive decimal integers"
            ) from exc
        if component_count <= 0 or key != str(component_count):
            raise ValueError(
                "bic_by_component_count keys must be canonical positive decimal integers"
            )
        if key in normalized:
            raise ValueError("bic_by_component_count has duplicate component keys")
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise TypeError("bic_by_component_count values must be finite numbers")
        number = float(raw_value)
        if not math.isfinite(number):
            raise ValueError("bic_by_component_count values must be finite numbers")
        normalized[key] = number
    return canonical_bytes(normalized).decode("ascii")


def _normalize_arrow_value(name: str, arrow_type: str, value: object) -> object:
    if arrow_type == "string":
        if type(value) is not str:
            raise ValueError(f"{name} must be a string")
        return value
    if arrow_type == "bool":
        if type(value) is not bool:
            raise ValueError(f"{name} must be a bool")
        return value
    if arrow_type in {"int32", "int64"}:
        if type(value) is not int:
            raise ValueError(f"{name} must be an integer")
        lower, upper = (
            (-(2**31), 2**31) if arrow_type == "int32" else (-(2**63), 2**63)
        )
        if not lower <= value < upper:
            raise ValueError(f"{name} is outside the {arrow_type} range")
        return value
    if arrow_type == "float64":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a finite number")
        normalized = float(value)
        if not math.isfinite(normalized):
            raise ValueError(f"{name} must be a finite number")
        return normalized
    if arrow_type == "list<string>":
        if not isinstance(value, (list, tuple)) or any(
            type(item) is not str for item in value
        ):
            raise ValueError(f"{name} must be a list of strings")
        return list(value)
    raise ValueError(f"unsupported baseline-strategy Arrow type: {arrow_type}")


def normalize_baseline_strategy_row(
    table_name: str,
    row: Mapping[str, Any],
    *,
    analysis_run_id: str | None = None,
) -> dict[str, Any]:
    """Return one complete, ordered v2 row and reject open-ended input.

    The cluster builder's v1 ``bic_by_component_count`` mapping is accepted only
    at this boundary and converted to canonical strict JSON. Persisted v2 rows
    expose only ``bic_by_component_count_json``.
    """

    try:
        contract = BASELINE_STRATEGY_ARROW_CONTRACTS[table_name]
    except KeyError as exc:
        raise KeyError(f"unknown baseline strategy table: {table_name}") from exc
    source = dict(row)
    if analysis_run_id is not None:
        declared = source.get("analysis_run_id")
        if declared is not None and declared != analysis_run_id:
            raise ValueError(
                f"{table_name}: analysis_run_id differs from the publication run"
            )
        source["analysis_run_id"] = analysis_run_id
    if source.get("schema_version") == LEGACY_SCHEMA_VERSION:
        source["schema_version"] = SCHEMA_VERSION
    if table_name == BASELINE_STRATEGY_CLUSTERS_TABLE:
        legacy_present = "bic_by_component_count" in source
        exact_present = "bic_by_component_count_json" in source
        if legacy_present and exact_present:
            raise ValueError(
                "baseline_strategy_clusters: legacy and exact BIC fields cannot coexist"
            )
        if legacy_present:
            source["bic_by_component_count_json"] = _canonical_bic_json(
                source.pop("bic_by_component_count")
            )
        elif exact_present:
            source["bic_by_component_count_json"] = _canonical_bic_json(
                source["bic_by_component_count_json"]
            )

    fields = contract.fields
    names = {item.name for item in fields}
    unexpected = sorted(set(source) - names)
    if unexpected:
        raise ValueError(f"{table_name}: unexpected fields {unexpected}")
    output: dict[str, Any] = {}
    for item in fields:
        value = source.get(item.name)
        if value is None:
            if not item.nullable:
                state = "missing" if item.name not in source else "null"
                raise ValueError(
                    f"{table_name}: required field is {state}: {item.name}"
                )
            output[item.name] = None
        else:
            output[item.name] = _normalize_arrow_value(
                item.name,
                item.arrow_type,
                value,
            )
    expected_constants = {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "table_name": table_name,
        "method": METHOD,
        "method_version": METHOD_VERSION,
    }
    for name, expected in expected_constants.items():
        if output[name] != expected:
            raise ValueError(
                f"{table_name}: {name} must equal the installed v2 value {expected!r}"
            )
    return output


def validate_baseline_strategy_primary_keys(
    table_name: str,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    """Reject null or duplicate v2 primary keys within one table batch."""

    try:
        key_fields = BASELINE_STRATEGY_PRIMARY_KEYS[table_name]
    except KeyError as exc:
        raise KeyError(f"unknown baseline strategy table: {table_name}") from exc
    contract = BASELINE_STRATEGY_ARROW_CONTRACTS[table_name]
    fields_by_name = {item.name: item for item in contract.fields}
    observed: set[tuple[object, ...]] = set()
    for row_index, row in enumerate(rows):
        values: list[object] = []
        for name in key_fields:
            value = row.get(name)
            if value is None or (type(value) is str and not value.strip()):
                raise ValueError(
                    f"{table_name}: row {row_index} has a null or empty primary-key "
                    f"field {name}"
                )
            values.append(
                _normalize_arrow_value(name, fields_by_name[name].arrow_type, value)
            )
        key = tuple(values)
        if key in observed:
            raise ValueError(f"{table_name}: duplicate primary key {key!r}")
        observed.add(key)


def normalize_baseline_strategy_rows(
    table_name: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    analysis_run_id: str | None = None,
) -> list[dict[str, Any]]:
    """Normalize a complete table batch, then validate its global primary key."""

    normalized = [
        normalize_baseline_strategy_row(
            table_name,
            row,
            analysis_run_id=analysis_run_id,
        )
        for row in rows
    ]
    validate_baseline_strategy_primary_keys(table_name, normalized)
    return normalized


def contract_fields(table_name: str) -> tuple[str, ...]:
    common = (
        "schema_id",
        "schema_version",
        "table_name",
        "method",
        "method_version",
        *IDENTITY_COLUMNS,
    )
    specific = {
        BASELINE_STRATEGY_FEATURES_TABLE: (
            "feature_status",
            "feature_reason",
            "sample_features_available",
            "time_bin_features_available",
        ),
        BASELINE_EXPLORATION_EPISODES_TABLE: (
            "episode_id",
            "episode_start_s",
            "episode_end_s",
            "episode_duration_s",
        ),
        BASELINE_STRATEGY_CLASSIFICATION_TABLE: (
            "classification_status",
            "activity_state",
            "boundary_strategy",
            "spatial_organization",
            "temporal_pattern",
            "primary_strategy",
        ),
        BASELINE_STRATEGY_CLUSTERS_TABLE: (
            "cluster_status",
            "cluster_id",
            "cluster_probability",
        ),
    }
    if table_name not in specific:
        raise KeyError(f"unknown baseline strategy table: {table_name}")
    return (*common, *specific[table_name])


__all__ = [
    "ARROW_CONTRACT_ENVELOPE_SCHEMA_ID",
    "ARROW_CONTRACT_ENVELOPE_SCHEMA_VERSION",
    "ARROW_TABLE_SCHEMA_NAMESPACE",
    "BASELINE_STRATEGY_ARROW_CONTRACTS",
    "BASELINE_EXPLORATION_EPISODES_TABLE",
    "BASELINE_STRATEGY_PRIMARY_KEYS",
    "BASELINE_STRATEGY_TABLES",
    "BASELINE_STRATEGY_CLASSIFICATION_TABLE",
    "BASELINE_STRATEGY_CLUSTERS_TABLE",
    "BASELINE_STRATEGY_FEATURES_TABLE",
    "IDENTITY_COLUMNS",
    "LEGACY_SCHEMA_VERSION",
    "METHOD",
    "METHOD_VERSION",
    "SCHEMA_ID",
    "SCHEMA_VERSION",
    "StrategyFeatureConfig",
    "baseline_strategy_arrow_contract_envelope",
    "contract_fields",
    "exact_baseline_strategy_arrow_schema",
    "normalize_baseline_strategy_row",
    "normalize_baseline_strategy_rows",
    "validate_baseline_strategy_arrow_contract_envelope",
    "validate_baseline_strategy_arrow_schema",
    "validate_baseline_strategy_primary_keys",
]
